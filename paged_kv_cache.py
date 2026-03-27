"""
Paged KV-Cache Manager
======================

A vLLM-inspired paged memory manager for KV caches, designed to integrate
with ``flash_attn_varlen_func`` for batched multi-sequence decoding in
the Prefix-Tree UQ generator.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import torch


# ============================================================================
# 1. BLOCK ALLOCATOR
# ============================================================================

class BlockAllocator:
    """
    Manages a fixed pool of block indices with reference counting.
    """

    def __init__(self, num_blocks: int):
        self.num_blocks = num_blocks
        self.refcounts = [0] * num_blocks
        self._free: set[int] = set(range(num_blocks))
        self._peak_used: int = 0  # high-water mark for blocks ever in use

    @property
    def num_free(self) -> int:
        return len(self._free)

    @property
    def num_used(self) -> int:
        return self.num_blocks - len(self._free)

    @property
    def peak_used(self) -> int:
        """High-water mark: maximum number of blocks ever simultaneously allocated."""
        return self._peak_used

    def alloc(self) -> int:
        if not self._free:
            raise RuntimeError(
                f"PagedKVCache: out of blocks (pool size={self.num_blocks}).  "
                "Increase max_blocks or prune more aggressively."
            )
        block_idx = self._free.pop()
        self.refcounts[block_idx] = 1
        current_used = self.num_blocks - len(self._free)
        if current_used > self._peak_used:
            self._peak_used = current_used
        return block_idx

    def ref(self, block_idx: int) -> None:
        assert self.refcounts[block_idx] > 0, f"ref() on free block {block_idx}"
        self.refcounts[block_idx] += 1

    def release(self, block_idx: int) -> None:
        assert self.refcounts[block_idx] > 0, f"release() on free block {block_idx}"
        self.refcounts[block_idx] -= 1
        if self.refcounts[block_idx] == 0:
            self._free.add(block_idx)

    def is_shared(self, block_idx: int) -> bool:
        return self.refcounts[block_idx] > 1

    def can_alloc(self, n: int = 1) -> bool:
        return len(self._free) >= n


# ============================================================================
# 2. SEQUENCE METADATA
# ============================================================================

@dataclass
class SequenceState:
    """
    Per-sequence bookkeeping. Fixed to track tokens per layer.
    """
    seq_id: int
    num_layers: int
    num_tokens: list[int] = field(default_factory=list)
    tokens_in_last_block: list[int] = field(default_factory=list)
    page_table: list[list[int]] = field(default_factory=list)

    def __post_init__(self):
        if not self.page_table:
            self.page_table = [[] for _ in range(self.num_layers)]
        if not self.num_tokens:
            self.num_tokens = [0 for _ in range(self.num_layers)]
        if not self.tokens_in_last_block:
            self.tokens_in_last_block = [0 for _ in range(self.num_layers)]

# ============================================================================
# 3. PAGED KV-CACHE MANAGER
# ============================================================================

class PagedKVCacheManager:

    def __init__(
        self,
        num_layers: int,
        num_key_value_heads: int,
        head_dim: int,
        max_blocks: int = 4096,
        block_size: int = 16,
        device: str | torch.device = "cuda",
        dtype: torch.dtype = torch.bfloat16,
    ):
        self.num_layers = num_layers
        self.num_key_value_heads = num_key_value_heads
        self.head_dim = head_dim
        self.max_blocks = max_blocks
        self.block_size = block_size
        self.device = torch.device(device)
        self.dtype = dtype

        self.allocator = BlockAllocator(max_blocks)

        self.k_pools: list[torch.Tensor] = []
        self.v_pools: list[torch.Tensor] = []
        for _ in range(num_layers):
            self.k_pools.append(
                torch.zeros(
                    max_blocks, block_size, num_key_value_heads, head_dim,
                    device=self.device, dtype=self.dtype,
                )
            )
            self.v_pools.append(
                torch.zeros(
                    max_blocks, block_size, num_key_value_heads, head_dim,
                    device=self.device, dtype=self.dtype,
                )
            )

        self._sequences: dict[int, SequenceState] = {}
        self._next_seq_id: int = 0

    def allocate_sequence(self) -> int:
        seq_id = self._next_seq_id
        self._next_seq_id += 1
        self._sequences[seq_id] = SequenceState(
            seq_id=seq_id,
            num_layers=self.num_layers,
        )
        return seq_id

    def fork_sequence(self, parent_id: int) -> int:
        parent = self._sequences[parent_id]
        child_id = self._next_seq_id
        self._next_seq_id += 1

        child_page_table: list[list[int]] = []
        for layer_blocks in parent.page_table:
            child_page_table.append(list(layer_blocks))
            for blk_idx in layer_blocks:
                self.allocator.ref(blk_idx)

        self._sequences[child_id] = SequenceState(
            seq_id=child_id,
            num_layers=self.num_layers,
            num_tokens=list(parent.num_tokens),
            tokens_in_last_block=list(parent.tokens_in_last_block),
            page_table=child_page_table,
        )
        return child_id

    def free_sequence(self, seq_id: int) -> None:
        seq = self._sequences.pop(seq_id)
        for layer_blocks in seq.page_table:
            for blk_idx in layer_blocks:
                self.allocator.release(blk_idx)

    def append_tokens(
        self,
        seq_id: int,
        layer_idx: int,
        k: torch.Tensor,
        v: torch.Tensor,
    ) -> None:
        seq = self._sequences[seq_id]
        page_table = seq.page_table[layer_idx]
        num_new = k.shape[0]

        written = 0
        while written < num_new:
            # Check the specific layer's token count
            if not page_table or seq.tokens_in_last_block[layer_idx] == self.block_size:
                new_blk = self.allocator.alloc()
                page_table.append(new_blk)
                seq.tokens_in_last_block[layer_idx] = 0

            tail_blk = page_table[-1]

            if self.allocator.is_shared(tail_blk):
                new_blk = self.allocator.alloc()
                self.k_pools[layer_idx][new_blk].copy_(
                    self.k_pools[layer_idx][tail_blk]
                )
                self.v_pools[layer_idx][new_blk].copy_(
                    self.v_pools[layer_idx][tail_blk]
                )
                self.allocator.release(tail_blk)
                page_table[-1] = new_blk
                tail_blk = new_blk

            slot_start = seq.tokens_in_last_block[layer_idx]
            space = self.block_size - slot_start
            to_write = min(space, num_new - written)

            self.k_pools[layer_idx][tail_blk, slot_start:slot_start + to_write] = (
                k[written:written + to_write]
            )
            self.v_pools[layer_idx][tail_blk, slot_start:slot_start + to_write] = (
                v[written:written + to_write]
            )

            written += to_write
            seq.tokens_in_last_block[layer_idx] += to_write

        seq.num_tokens[layer_idx] += num_new
        
    def prepare_append_batched(
        self,
        seq_ids: list[int],
    ) -> tuple[list[list[int]], list[int]]:
        """
        Prepare write destinations for appending 1 token per sequence,
        performing all block allocations and CoW copies for every layer
        in one pass.

        Returns:
            per_layer_block_indices: list of num_layers lists, each of
                length N — the destination block index for each
                sequence in each layer.  In the common case (no CoW)
                all layers share the same list; after CoW they may
                diverge.
            slot_indices: list[int] of N slot offsets — identical
                across all layers.
        """
        N = len(seq_ids)
        # Pre-fill per-layer block index storage
        per_layer_block_indices: list[list[int]] = [
            [] for _ in range(self.num_layers)
        ]
        slot_indices: list[int] = []

        for seq_id in seq_ids:
            seq = self._sequences[seq_id]

            # Need a new block?  Check layer 0 as the reference.
            need_new_block = (
                not seq.page_table[0]
                or seq.tokens_in_last_block[0] == self.block_size
            )
            if need_new_block:
                # Allocate one block per layer so pools stay independent.
                for layer_idx in range(self.num_layers):
                    new_blk = self.allocator.alloc()
                    seq.page_table[layer_idx].append(new_blk)
                    seq.tokens_in_last_block[layer_idx] = 0

            # CoW: check layer 0's tail block as the shared-refcount proxy.
            tail_blk_0 = seq.page_table[0][-1]
            if self.allocator.is_shared(tail_blk_0):
                for layer_idx in range(self.num_layers):
                    old_blk = seq.page_table[layer_idx][-1]
                    new_blk = self.allocator.alloc()
                    self.k_pools[layer_idx][new_blk].copy_(
                        self.k_pools[layer_idx][old_blk]
                    )
                    self.v_pools[layer_idx][new_blk].copy_(
                        self.v_pools[layer_idx][old_blk]
                    )
                    self.allocator.release(old_blk)
                    seq.page_table[layer_idx][-1] = new_blk

            # Record the slot (same for all layers) and per-layer blocks.
            slot_indices.append(seq.tokens_in_last_block[0])
            for layer_idx in range(self.num_layers):
                per_layer_block_indices[layer_idx].append(
                    seq.page_table[layer_idx][-1]
                )

            # Update counters for ALL layers
            for layer_idx in range(self.num_layers):
                seq.tokens_in_last_block[layer_idx] += 1
                seq.num_tokens[layer_idx] += 1

        return per_layer_block_indices, slot_indices

    def apply_append_batched(
        self,
        layer_idx: int,
        k: torch.Tensor,
        v: torch.Tensor,
        block_idx_t: torch.Tensor,
        slot_idx_t: torch.Tensor,
    ) -> None:
        """
        Execute the GPU scatter-write for a single layer using
        precomputed index tensors.

        Args:
            layer_idx:   which layer's pool to write into.
            k, v:        (N, num_kv_heads, head_dim) new KV to write.
            block_idx_t: (N,) int64 tensor of destination block indices.
            slot_idx_t:  (N,) int64 tensor of destination slot offsets.
        """
        self.k_pools[layer_idx][block_idx_t, slot_idx_t] = k
        self.v_pools[layer_idx][block_idx_t, slot_idx_t] = v

    def append_tokens_batched(
        self,
        seq_ids: list[int],
        layer_idx: int,
        k: torch.Tensor,
        v: torch.Tensor,
    ) -> None:
        """
        Append 1 token for N sequences in a single batched GPU operation.
        k, v expected shapes: (N, num_key_value_heads, head_dim)

        Legacy single-layer interface — kept for backward compatibility.
        For the optimised path, use prepare_append_batched() once then
        apply_append_batched() per layer.
        """
        N = len(seq_ids)
        if N == 0:
            return

        block_indices = []
        slot_indices = []

        # 1. State updates & allocations (Fast Python logic)
        for i, seq_id in enumerate(seq_ids):
            seq = self._sequences[seq_id]
            page_table = seq.page_table[layer_idx]

            # Need a new block?
            if not page_table or seq.tokens_in_last_block[layer_idx] == self.block_size:
                new_blk = self.allocator.alloc()
                page_table.append(new_blk)
                seq.tokens_in_last_block[layer_idx] = 0

            tail_blk = page_table[-1]

            # CoW copy if shared
            if self.allocator.is_shared(tail_blk):
                new_blk = self.allocator.alloc()
                self.k_pools[layer_idx][new_blk].copy_(self.k_pools[layer_idx][tail_blk])
                self.v_pools[layer_idx][new_blk].copy_(self.v_pools[layer_idx][tail_blk])
                self.allocator.release(tail_blk)
                page_table[-1] = new_blk
                tail_blk = new_blk

            # Record destinations for this sequence
            slot_start = seq.tokens_in_last_block[layer_idx]
            block_indices.append(tail_blk)
            slot_indices.append(slot_start)

            # Update counters (assuming decode step = 1 token at a time)
            seq.tokens_in_last_block[layer_idx] += 1
            seq.num_tokens[layer_idx] += 1


        # 2. Vectorized write (Single GPU Kernel)
        # Convert index lists to tensors for advanced indexing
        block_idx_t = torch.tensor(block_indices, device=self.device, dtype=torch.long)
        slot_idx_t = torch.tensor(slot_indices, device=self.device, dtype=torch.long)

        # Write all N tokens to their respective blocks and slots simultaneously
        self.k_pools[layer_idx][block_idx_t, slot_idx_t] = k
        self.v_pools[layer_idx][block_idx_t, slot_idx_t] = v
    def get_kv_length(self, seq_id: int) -> int:
        # All layers should have the same logical length, so reading layer 0 is safe
        return self._sequences[seq_id].num_tokens[0]

    def prepare_gather_indices(
        self,
        seq_ids: list[int],
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, int] | None:
        """
        Precompute the gather index tensors and cu_seqlens_k that are
        shared across all layers for a given batch of sequences.

        Because blocks are allocated symmetrically (every layer gets the
        same page-table structure), the (block_idx, slot_idx) gather
        pattern is identical for every layer.  Call this once before the
        layer loop and pass the result to ``build_packed_kv`` to avoid
        re-building and re-uploading these small tensors L times per
        decode step.

        Returns:
            (blk_t, slt_t, cu_seqlens_k, max_seqlen_k) or None if the
            batch is empty.

            blk_t:        (total_tokens,) int64 block indices on device
            slt_t:        (total_tokens,) int64 slot  indices on device
            cu_seqlens_k: (N+1,) int32 cumulative seq lengths on device
            max_seqlen_k: int — longest sequence in the batch
        """
        seq_lengths: list[int] = []
        block_indices: list[int] = []
        slot_indices: list[int] = []

        # Use layer 0 — symmetric with all other layers.
        for sid in seq_ids:
            seq = self._sequences[sid]
            seq_len = seq.num_tokens[0]
            seq_lengths.append(seq_len)

            tokens_remaining = seq_len
            for blk_idx in seq.page_table[0]:
                n = min(self.block_size, tokens_remaining)
                block_indices.extend([blk_idx] * n)
                slot_indices.extend(range(n))
                tokens_remaining -= n
                if tokens_remaining == 0:
                    break

        total_tokens = sum(seq_lengths)
        if total_tokens == 0:
            return None

        max_seqlen_k = max(seq_lengths) if seq_lengths else 0

        blk_t = torch.tensor(block_indices, device=self.device, dtype=torch.long)
        slt_t = torch.tensor(slot_indices,  device=self.device, dtype=torch.long)

        cu_cpu = torch.zeros(len(seq_ids) + 1, dtype=torch.int32)
        for i, l in enumerate(seq_lengths):
            cu_cpu[i + 1] = cu_cpu[i] + l
        cu_seqlens_k = cu_cpu.to(self.device, non_blocking=True)

        return blk_t, slt_t, cu_seqlens_k, max_seqlen_k

    def build_packed_kv(
        self,
        seq_ids: list[int],
        layer_idx: int,
        cached_indices: tuple[torch.Tensor, torch.Tensor, torch.Tensor, int] | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, int]:
        """
        Gather K and V for all sequences into a single packed (varlen) tensor.

        If ``cached_indices`` is provided (from ``prepare_gather_indices``),
        skips index construction and GPU upload entirely — just does the
        two gather kernels.  This saves 2 × torch.tensor() allocations and
        CPU→GPU transfers per layer after the first.

        Args:
            seq_ids:        list of N paged-cache sequence ids.
            layer_idx:      which transformer layer to gather from.
            cached_indices: optional precomputed (blk_t, slt_t,
                            cu_seqlens_k, max_seqlen_k) from
                            ``prepare_gather_indices``.

        Returns:
            packed_k:    (total_tokens, num_kv_heads, head_dim)
            packed_v:    (total_tokens, num_kv_heads, head_dim)
            cu_seqlens_k: (N+1,) int32 cumulative sequence lengths on device
            max_seqlen_k: int — longest sequence in the batch (for FlashAttn)
        """
        k_pool = self.k_pools[layer_idx]
        v_pool = self.v_pools[layer_idx]

        # --- Fast path: use precomputed indices ---
        if cached_indices is not None:
            blk_t, slt_t, cu_seqlens_k, max_seqlen_k = cached_indices
            packed_k = k_pool[blk_t, slt_t]
            packed_v = v_pool[blk_t, slt_t]
            return packed_k, packed_v, cu_seqlens_k, max_seqlen_k

        # --- Slow path: build indices from scratch (fallback) ---
        seq_lengths: list[int] = []
        block_indices: list[int] = []
        slot_indices: list[int] = []

        for sid in seq_ids:
            seq = self._sequences[sid]
            seq_len = seq.num_tokens[layer_idx]
            seq_lengths.append(seq_len)

            tokens_remaining = seq_len
            for blk_idx in seq.page_table[layer_idx]:
                n = min(self.block_size, tokens_remaining)
                block_indices.extend([blk_idx] * n)
                slot_indices.extend(range(n))
                tokens_remaining -= n
                if tokens_remaining == 0:
                    break

        total_tokens = sum(seq_lengths)
        max_seqlen_k = max(seq_lengths) if seq_lengths else 0

        if total_tokens == 0:
            empty = torch.empty(
                0, self.num_key_value_heads, self.head_dim,
                device=self.device, dtype=self.dtype,
            )
            cu = torch.zeros(
                len(seq_ids) + 1, device=self.device, dtype=torch.int32,
            )
            return empty, empty, cu, 0

        blk_t = torch.tensor(block_indices, device=self.device, dtype=torch.long)
        slt_t = torch.tensor(slot_indices,  device=self.device, dtype=torch.long)

        packed_k = k_pool[blk_t, slt_t]
        packed_v = v_pool[blk_t, slt_t]

        cu_cpu = torch.zeros(len(seq_ids) + 1, dtype=torch.int32)
        for i, l in enumerate(seq_lengths):
            cu_cpu[i + 1] = cu_cpu[i] + l
        cu_seqlens_k = cu_cpu.to(self.device, non_blocking=True)

        return packed_k, packed_v, cu_seqlens_k, max_seqlen_k

    def trim_to_window(self, seq_id: int, window_size: int) -> None:
        seq = self._sequences[seq_id]
        # Just check layer 0 to see if we need to evict
        if seq.num_tokens[0] <= window_size:
            return

        for layer_idx in range(self.num_layers):
            tokens_to_evict = seq.num_tokens[layer_idx] - window_size
            page_table = seq.page_table[layer_idx]
            remaining = tokens_to_evict
            
            while remaining > 0 and page_table:
                blk_idx = page_table[0]
                tokens_in_blk = min(self.block_size, remaining)

                if tokens_in_blk >= self.block_size:
                    page_table.pop(0)
                    self.allocator.release(blk_idx)
                    remaining -= self.block_size
                else:
                    if self.allocator.is_shared(blk_idx):
                        new_blk = self.allocator.alloc()
                        keep = self.block_size - tokens_in_blk
                        self.k_pools[layer_idx][new_blk, :keep] = (
                            self.k_pools[layer_idx][blk_idx, tokens_in_blk:tokens_in_blk + keep]
                        )
                        self.v_pools[layer_idx][new_blk, :keep] = (
                            self.v_pools[layer_idx][blk_idx, tokens_in_blk:tokens_in_blk + keep]
                        )
                        self.allocator.release(blk_idx)
                        page_table[0] = new_blk
                    else:
                        keep = self.block_size - tokens_in_blk
                        self.k_pools[layer_idx][blk_idx, :keep] = (
                            self.k_pools[layer_idx][blk_idx, tokens_in_blk:tokens_in_blk + keep].clone()
                        )
                        self.v_pools[layer_idx][blk_idx, :keep] = (
                            self.v_pools[layer_idx][blk_idx, tokens_in_blk:tokens_in_blk + keep].clone()
                        )
                    remaining = 0

            seq.num_tokens[layer_idx] = window_size
            if seq.num_tokens[layer_idx] == 0:
                seq.tokens_in_last_block[layer_idx] = 0
            else:
                seq.tokens_in_last_block[layer_idx] = seq.num_tokens[layer_idx] % self.block_size
                if seq.tokens_in_last_block[layer_idx] == 0:
                    seq.tokens_in_last_block[layer_idx] = self.block_size

    def get_stats(self) -> dict:
        num_seqs = len(self._sequences)
        # Safely count tokens via layer 0
        total_tokens = sum(s.num_tokens[0] if s.num_tokens else 0 for s in self._sequences.values())
        elem_bytes = self.dtype_size
        # bytes per block across all layers, K and V tensors
        bytes_per_block = (
            2 * self.num_layers * self.block_size * self.num_key_value_heads * self.head_dim * elem_bytes
        )
        return {
            "num_sequences": num_seqs,
            "total_cached_tokens": total_tokens,
            "blocks_used": self.allocator.num_used,
            "blocks_free": self.allocator.num_free,
            "blocks_total": self.max_blocks,
            "blocks_peak": self.allocator.peak_used,
            "utilization": self.allocator.num_used / self.max_blocks,
            "peak_utilization": self.allocator.peak_used / self.max_blocks,
            "pool_memory_bytes": bytes_per_block * self.max_blocks,
            "peak_kv_cache_bytes": bytes_per_block * self.allocator.peak_used,
        }

    @property
    def dtype_size(self) -> int:
        return torch.tensor([], dtype=self.dtype).element_size()

    def __repr__(self) -> str:
        stats = self.get_stats()
        return (
            f"PagedKVCacheManager("
            f"layers={self.num_layers}, "
            f"heads={self.num_key_value_heads}, "
            f"head_dim={self.head_dim}, "
            f"block_size={self.block_size}, "
            f"blocks={stats['blocks_used']}/{self.max_blocks}, "
            f"seqs={stats['num_sequences']}, "
            f"tokens={stats['total_cached_tokens']})"
        )