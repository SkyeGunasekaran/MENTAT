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

    def build_packed_kv(
        self,
        seq_ids: list[int],
        layer_idx: int,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, int]:
        """
        Gather K and V for all sequences into a single packed (varlen) tensor.

        Previous implementation issued one GPU copy per block per sequence in a
        Python loop — O(N × blocks_per_seq) kernel launches per layer per step.

        This implementation:
          1. Walks the page tables in Python to build two flat integer lists:
             ``block_indices`` and ``slot_indices``, each of length
             ``total_tokens``.  This is pure CPU integer arithmetic on small
             lists and is negligible.
          2. Issues a single advanced-index gather on each pool tensor:
               pool[block_indices, slot_indices]   →  (total_tokens, H, D)
             This is one GPU kernel regardless of N, sequence length, or
             branching depth.
          3. Builds ``cu_seqlens_k`` from the sequence lengths, which are
             already tracked in Python state — no GPU reduction needed.

        Returns:
            packed_k:    (total_tokens, num_kv_heads, head_dim)
            packed_v:    (total_tokens, num_kv_heads, head_dim)
            cu_seqlens_k: (N+1,) int32 cumulative sequence lengths on device
            max_seqlen_k: int — longest sequence in the batch (for FlashAttn)
        """
        k_pool = self.k_pools[layer_idx]
        v_pool = self.v_pools[layer_idx]

        # --- 1. Collect lengths and build flat gather indices in Python ---
        # All arithmetic is on small Python ints; no GPU ops here.
        # The gather indices (block_idx, slot_idx) are identical across all
        # layers because blocks are allocated symmetrically.  We therefore
        # cache them on each SequenceState after the first build and reuse
        # them for layers 1-N, saving N-1 list-construction passes per step.
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

        # --- 2. Early-exit for empty batch ---
        if total_tokens == 0:
            empty = torch.empty(
                0, self.num_key_value_heads, self.head_dim,
                device=self.device, dtype=self.dtype,
            )
            cu = torch.zeros(
                len(seq_ids) + 1, device=self.device, dtype=torch.int32,
            )
            return empty, empty, cu, 0

        # --- 3. Single GPU gather (one kernel each for K and V) ---
        # Upload index tensors once; they are small (total_tokens ints).
        blk_t = torch.tensor(block_indices, device=self.device, dtype=torch.long)
        slt_t = torch.tensor(slot_indices,  device=self.device, dtype=torch.long)

        # pool shape: (max_blocks, block_size, num_kv_heads, head_dim)
        # Result:     (total_tokens, num_kv_heads, head_dim)
        packed_k = k_pool[blk_t, slt_t]
        packed_v = v_pool[blk_t, slt_t]

        # --- 4. cu_seqlens_k — cumulative sum of Python ints, pinned transfer ---
        # torch.tensor() on a tiny CPU list + .to(device) is faster than
        # computing cumsum on GPU for the small N typical here.
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