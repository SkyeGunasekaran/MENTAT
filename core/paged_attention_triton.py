"""
paged_attention_triton.py
=========================

Triton-based PagedAttention decode kernel + GPU-native metadata extensions
for the PagedKVCacheManager.  Eliminates the Python-level ``build_packed_kv``
gather that was the main CPU-GPU IO bottleneck during tree-based batched
decoding.

Architecture
------------

1. **Triton Kernel** (``_paged_attn_decode_kernel``):
   Fused QK^T → softmax → V accumulation that reads K/V directly from the
   paged pool using a per-sequence block table.  Each Triton program handles
   one (sequence, head) pair.  Uses online safe-softmax (Milakov-Gimelshein)
   so it streams through KV blocks in a single pass with O(HEAD_DIM) state.

2. **GPU-Native Metadata** (``TritonCacheMetadata``):
   Maintains ``block_tables`` and ``context_lens`` as contiguous GPU tensors
   (row-mapped to sequence IDs) so the Triton kernel can launch with zero
   Python-loop gather.  Bolted onto the existing ``PagedKVCacheManager`` via
   a thin wrapper — all existing CoW / fork / free logic is preserved.

3. **Pool Layout Compatibility**:
   The production ``PagedKVCacheManager`` stores pools as:
       k_pools[layer]: (max_blocks, block_size, num_kv_heads, head_dim)
   
   The Triton kernel expects:
       (max_blocks, num_kv_heads, block_size, head_dim)
   
   We handle this by reshuffling the pool on-the-fly inside the wrapper's
   ``get_pool_for_triton()`` method via a permute (which is a view, not a
   copy).  Alternatively, you can change the pool layout at allocation time
   for zero overhead — see the docstring on ``TritonPagedCacheManager``.

Integration
-----------

Drop-in replacement for ``build_packed_kv`` + ``flash_attn_varlen_func`` in
the decode path.  The ``model_adapter.py`` ``forward_paged_decode`` method
is the integration point — see ``TritonPagedCacheManager.decode_attention``.

Usage::

    from paged_attention_triton import TritonPagedCacheManager

    # Wraps your existing PagedKVCacheManager
    mgr = TritonPagedCacheManager(base_kv_cache_mgr)

    # In the decode loop (replaces build_packed_kv + flash_attn_varlen_func):
    attn_out = mgr.decode_attention(q, seq_ids, layer_idx)
"""

from __future__ import annotations

import math
from typing import Optional

import torch

try:
    import triton
    import triton.language as tl
    HAS_TRITON = True
except ImportError:
    HAS_TRITON = False


# ============================================================================
# 1. TRITON PAGED ATTENTION KERNEL (DECODE PHASE)
# ============================================================================

if HAS_TRITON:
    @triton.jit
    def _paged_attn_decode_kernel(
        # Tensor pointers
        Q,                              # [N, num_q_heads, head_dim]
        K_pool,                         # [max_blocks, num_kv_heads, block_size, head_dim]
        V_pool,                         # [max_blocks, num_kv_heads, block_size, head_dim]
        O,                              # [N, num_q_heads, head_dim]
        # Metadata
        block_tables,                   # [max_seqs, max_blocks_per_seq]  (int32)
        context_lens,                   # [max_seqs]  (int32)
        seq_row_map,                    # [N]  (int64) — maps batch idx → row in block_tables
        # Scalars
        softmax_scale,
        # Q strides
        stride_q_seq, stride_q_head, stride_q_dim,
        # Pool strides (shared for K and V since they have the same layout)
        stride_pool_block, stride_pool_head, stride_pool_bslot, stride_pool_dim,
        # O strides
        stride_o_seq, stride_o_head, stride_o_dim,
        # Block table strides
        stride_bt_row, stride_bt_col,
        # GQA factor: num_q_heads // num_kv_heads
        GQA_FACTOR: tl.constexpr,
        # Compile-time constants
        BLOCK_SIZE_KV: tl.constexpr,
        HEAD_DIM: tl.constexpr,
    ):
        """
        PagedAttention decode kernel.  Query length = 1 per sequence.

        Each program instance handles one (batch_seq, q_head) pair.
        It walks the block table for the sequence, loading K/V blocks
        from the physical pool, and accumulates the attention output
        using online safe-softmax (single pass, O(HEAD_DIM) state).

        GQA: multiple Q heads map to the same KV head via
        ``kv_head = q_head // GQA_FACTOR``.
        """
        # --- Program identity ---
        batch_idx = tl.program_id(0)    # which sequence in the batch
        q_head_idx = tl.program_id(1)   # which query head

        # Map Q head → KV head for GQA
        kv_head_idx = q_head_idx // GQA_FACTOR

        # --- Look up the metadata row for this sequence ---
        row_idx = tl.load(seq_row_map + batch_idx)

        # Context length (total KV tokens including the just-appended one)
        ctx_len = tl.load(context_lens + row_idx)
        if ctx_len == 0:
            return

        # --- Load Q vector: [HEAD_DIM] ---
        q_offset = batch_idx * stride_q_seq + q_head_idx * stride_q_head
        dim_offsets = tl.arange(0, HEAD_DIM)
        q = tl.load(Q + q_offset + dim_offsets * stride_q_dim)

        # --- Block table base pointer for this sequence ---
        bt_base = block_tables + row_idx * stride_bt_row

        # --- Online softmax accumulators ---
        m_i = -float("inf")                        # running max
        l_i = 0.0                                   # running sum of exp
        acc = tl.zeros([HEAD_DIM], dtype=tl.float32)  # weighted V accumulator

        # --- Iterate over KV blocks ---
        num_blocks = (ctx_len + BLOCK_SIZE_KV - 1) // BLOCK_SIZE_KV

        for logical_blk in range(num_blocks):
            # Physical block index from the block table
            phys_blk = tl.load(bt_base + logical_blk * stride_bt_col)

            # Offsets within this block
            start_pos = logical_blk * BLOCK_SIZE_KV
            slot_offsets = tl.arange(0, BLOCK_SIZE_KV)
            valid_mask = (start_pos + slot_offsets) < ctx_len

            # --- Load K block: [BLOCK_SIZE_KV, HEAD_DIM] ---
            # Pool layout: [max_blocks, num_kv_heads, block_size, head_dim]
            k_base = (
                phys_blk * stride_pool_block
                + kv_head_idx * stride_pool_head
            )
            k_ptrs = (
                K_pool + k_base
                + slot_offsets[:, None] * stride_pool_bslot
                + dim_offsets[None, :] * stride_pool_dim
            )
            k = tl.load(k_ptrs, mask=valid_mask[:, None], other=0.0)

            # --- QK^T: [BLOCK_SIZE_KV] ---
            # q is [HEAD_DIM], k is [BLOCK_SIZE_KV, HEAD_DIM]
            qk = tl.sum(q[None, :] * k, axis=1) * softmax_scale
            qk = tl.where(valid_mask, qk, -float("inf"))

            # --- Load V block: [BLOCK_SIZE_KV, HEAD_DIM] ---
            v_ptrs = (
                V_pool + k_base
                + slot_offsets[:, None] * stride_pool_bslot
                + dim_offsets[None, :] * stride_pool_dim
            )
            v = tl.load(v_ptrs, mask=valid_mask[:, None], other=0.0)

            # --- Online softmax update ---
            m_ij = tl.max(qk, axis=0)
            m_new = tl.maximum(m_i, m_ij)
            alpha = tl.math.exp(m_i - m_new)       # rescale old accumulators
            p = tl.math.exp(qk - m_new)            # new attention weights
            l_ij = tl.sum(p, axis=0)

            # Update running state
            acc = acc * alpha + tl.sum(p[:, None] * v, axis=0)
            l_i = l_i * alpha + l_ij
            m_i = m_new

        # --- Normalize and write output ---
        out = acc / l_i

        o_offset = batch_idx * stride_o_seq + q_head_idx * stride_o_head
        tl.store(
            O + o_offset + dim_offsets * stride_o_dim,
            out.to(O.dtype.element_ty),
        )


def paged_attention_decode(
    q: torch.Tensor,                    # [N, num_q_heads, head_dim]
    k_pool: torch.Tensor,               # [max_blocks, num_kv_heads, block_size, head_dim]
    v_pool: torch.Tensor,               # [max_blocks, num_kv_heads, block_size, head_dim]
    block_tables: torch.Tensor,         # [max_seqs, max_blocks_per_seq]  (int32, on GPU)
    context_lens: torch.Tensor,         # [max_seqs]  (int32, on GPU)
    seq_row_map: torch.Tensor,          # [N]  (int64) — batch idx → row in block_tables
    num_kv_heads: int,
) -> torch.Tensor:
    """
    Python launcher for the Triton paged-attention decode kernel.

    Args:
        q:            Query vectors, one per sequence.  [N, num_q_heads, head_dim]
        k_pool:       Physical K pool.  [max_blocks, num_kv_heads, block_size, head_dim]
        v_pool:       Physical V pool.  Same layout as k_pool.
        block_tables: GPU tensor mapping (seq_row, logical_block) → physical_block.
        context_lens: GPU tensor with the total token count per metadata row.
        seq_row_map:  Maps each batch index [0..N) to the metadata row in
                      block_tables / context_lens for that sequence.
        num_kv_heads: Number of KV heads (for GQA factor calculation).

    Returns:
        output: [N, num_q_heads, head_dim]
    """
    if not HAS_TRITON:
        raise RuntimeError(
            "Triton is required for paged_attention_decode.  "
            "Install with: pip install triton"
        )

    N, num_q_heads, head_dim = q.shape
    block_size = k_pool.shape[2]
    gqa_factor = num_q_heads // num_kv_heads

    out = torch.empty_like(q)
    softmax_scale = 1.0 / math.sqrt(head_dim)

    # Grid: one program per (sequence, q_head)
    grid = (N, num_q_heads)

    _paged_attn_decode_kernel[grid](
        q, k_pool, v_pool, out,
        block_tables, context_lens, seq_row_map,
        softmax_scale,
        # Q strides
        q.stride(0), q.stride(1), q.stride(2),
        # Pool strides — same for K and V
        k_pool.stride(0), k_pool.stride(1), k_pool.stride(2), k_pool.stride(3),
        # O strides
        out.stride(0), out.stride(1), out.stride(2),
        # Block table strides
        block_tables.stride(0), block_tables.stride(1),
        # Compile-time constants
        GQA_FACTOR=gqa_factor,
        BLOCK_SIZE_KV=block_size,
        HEAD_DIM=head_dim,
    )
    return out


# ============================================================================
# 2. GPU-NATIVE METADATA WRAPPER
# ============================================================================

class TritonCacheMetadata:
    """
    Maintains GPU-resident block_tables and context_lens tensors that the
    Triton kernel reads directly — no Python-loop gathering at decode time.

    This is a *metadata overlay* on the existing ``PagedKVCacheManager``.
    The physical KV pools and all CoW / alloc / free logic remain in the
    base manager; this class just shadows the per-sequence metadata as GPU
    tensors.

    Design: each sequence gets a "row" in the metadata tensors.  Rows are
    recycled via a free-list (identical to the base manager's approach but
    tracked here for the GPU tensors).
    """

    def __init__(
        self,
        max_sequences: int = 2048,
        max_blocks_per_seq: int = 1024,
        device: torch.device = torch.device("cuda"),
    ):
        self.max_sequences = max_sequences
        self.max_blocks_per_seq = max_blocks_per_seq
        self.device = device

        # GPU-resident metadata
        self.block_tables = torch.zeros(
            (max_sequences, max_blocks_per_seq),
            dtype=torch.int32, device=device,
        )
        self.context_lens = torch.zeros(
            max_sequences, dtype=torch.int32, device=device,
        )

        # Row management (CPU-side, tiny)
        self._seq_to_row: dict[int, int] = {}
        self._free_rows: list[int] = list(range(max_sequences - 1, -1, -1))

    # ---- Row lifecycle ---------------------------------------------------

    def register_sequence(self, seq_id: int) -> int:
        """Assign a metadata row to a new sequence.  Returns the row index."""
        if not self._free_rows:
            raise RuntimeError("TritonCacheMetadata: out of metadata rows")
        row = self._free_rows.pop()
        self._seq_to_row[seq_id] = row
        self.context_lens[row] = 0
        return row

    def fork_metadata(self, parent_id: int, child_id: int) -> int:
        """Copy the parent's block table row → child.  Returns child row."""
        parent_row = self._seq_to_row[parent_id]
        child_row = self.register_sequence(child_id)

        ctx_len = self.context_lens[parent_row].item()
        num_blocks = (ctx_len + 16 - 1) // 16  # conservative; exact block_size comes from mgr
        self.block_tables[child_row, :num_blocks] = self.block_tables[parent_row, :num_blocks]
        self.context_lens[child_row] = ctx_len
        return child_row

    def release_sequence(self, seq_id: int):
        """Free the metadata row for reuse."""
        row = self._seq_to_row.pop(seq_id, None)
        if row is not None:
            self.context_lens[row] = 0
            self._free_rows.append(row)

    # ---- Metadata updates (called from append_tokens_batched) ------------

    def update_block_table(self, seq_id: int, logical_block: int, physical_block: int):
        """Write a single block-table entry on GPU."""
        row = self._seq_to_row[seq_id]
        self.block_tables[row, logical_block] = physical_block

    def set_context_len(self, seq_id: int, length: int):
        """Update the context length for a sequence."""
        row = self._seq_to_row[seq_id]
        self.context_lens[row] = length

    def increment_context_len(self, seq_id: int):
        """Increment context length by 1 (decode step)."""
        row = self._seq_to_row[seq_id]
        self.context_lens[row] += 1

    # ---- Batch query (for kernel launch) ---------------------------------

    def get_row_map(self, seq_ids: list[int]) -> torch.Tensor:
        """
        Build a [N] int64 tensor mapping batch index → metadata row.
        This is the only thing we need to build per decode step.
        """
        rows = [self._seq_to_row[sid] for sid in seq_ids]
        return torch.tensor(rows, dtype=torch.long, device=self.device)

    def get_row(self, seq_id: int) -> int:
        return self._seq_to_row[seq_id]


# ============================================================================
# 3. INTEGRATED MANAGER (wraps existing PagedKVCacheManager)
# ============================================================================

class TritonPagedCacheManager:
    """
    Drop-in enhancement for ``PagedKVCacheManager`` that adds:

    1. GPU-native metadata (``TritonCacheMetadata``) kept in sync with the
       base manager's Python-side bookkeeping.
    2. A ``decode_attention`` method that launches the Triton kernel directly
       against the paged pools — replacing ``build_packed_kv`` +
       ``flash_attn_varlen_func``.

    The base manager's pool layout is:
        k_pools[layer]: (max_blocks, block_size, num_kv_heads, head_dim)

    The Triton kernel expects:
        (max_blocks, num_kv_heads, block_size, head_dim)

    We handle this with a ``.permute(0, 2, 1, 3)`` which is a zero-copy
    view.  For maximum performance, you can change the base manager's
    allocation to use the Triton-native layout directly.

    Usage::

        base_mgr = PagedKVCacheManager(...)
        triton_mgr = TritonPagedCacheManager(base_mgr)

        # Use triton_mgr everywhere you'd use base_mgr.
        # It delegates all existing methods and adds decode_attention.
    """

    def __init__(self, base_mgr):
        self.base = base_mgr
        self.metadata = TritonCacheMetadata(
            max_sequences=2048,
            max_blocks_per_seq=1024,
            device=base_mgr.device,
        )
        # Cache the permuted pool views (they're just views, no memory cost).
        # We'll rebuild them lazily since the underlying storage doesn't change.
        self._pool_views_valid = False

    # ---- Pool view management --------------------------------------------

    def _get_triton_pools(self, layer_idx: int):
        """
        Return (k_pool, v_pool) in Triton layout:
            (max_blocks, num_kv_heads, block_size, head_dim)

        This is a zero-copy permute of the base manager's pools.
        """
        k = self.base.k_pools[layer_idx].permute(0, 2, 1, 3)
        v = self.base.v_pools[layer_idx].permute(0, 2, 1, 3)
        return k, v

    # ---- Delegated methods (keep metadata in sync) -----------------------

    def allocate_sequence(self) -> int:
        seq_id = self.base.allocate_sequence()
        self.metadata.register_sequence(seq_id)
        return seq_id

    def fork_sequence(self, parent_id: int) -> int:
        child_id = self.base.fork_sequence(parent_id)
        self.metadata.fork_metadata(parent_id, child_id)
        # Fix: fork_metadata uses a hardcoded block_size of 16 for num_blocks.
        # Re-sync with the actual block_size.
        parent_row = self.metadata._seq_to_row[parent_id]
        child_row = self.metadata._seq_to_row[child_id]
        ctx_len = self.base._sequences[child_id].num_tokens[0]
        bs = self.base.block_size
        num_blocks = (ctx_len + bs - 1) // bs
        self.metadata.block_tables[child_row, :num_blocks] = (
            self.metadata.block_tables[parent_row, :num_blocks]
        )
        self.metadata.context_lens[child_row] = ctx_len
        return child_id

    def free_sequence(self, seq_id: int) -> None:
        self.base.free_sequence(seq_id)
        self.metadata.release_sequence(seq_id)

    def get_kv_length(self, seq_id: int) -> int:
        return self.base.get_kv_length(seq_id)

    def append_tokens(self, seq_id: int, layer_idx: int, k, v) -> None:
        """Prefill append — delegates to base, then syncs metadata."""
        self.base.append_tokens(seq_id, layer_idx, k, v)
        self._sync_metadata_for_seq(seq_id)

    def append_tokens_batched(
        self, seq_ids: list[int], layer_idx: int, k, v,
    ) -> None:
        """
        Batched decode append — delegates to base, then syncs metadata.

        The metadata sync is lightweight: we just read the updated
        context_len and tail block index from the base manager's Python
        state (which was already updated by the base's append logic).
        """
        self.base.append_tokens_batched(seq_ids, layer_idx, k, v)
        # Only sync metadata once per decode step (layer 0 increments ctx_len)
        if layer_idx == 0:
            for sid in seq_ids:
                self._sync_metadata_for_seq(sid)
        else:
            # Still need to sync block table entries in case CoW allocated
            # new blocks on non-zero layers.
            for sid in seq_ids:
                self._sync_block_table(sid, layer_idx)

    def _sync_metadata_for_seq(self, seq_id: int):
        """
        Full sync of a sequence's metadata row from the base manager's
        Python-side state.  Called after append_tokens / prefill.
        """
        seq = self.base._sequences[seq_id]
        row = self.metadata.get_row(seq_id)
        # Use layer 0's token count as the canonical context length
        ctx_len = seq.num_tokens[0]
        self.metadata.context_lens[row] = ctx_len

        # Sync the block table for layer 0 (decode attention reads layer-specific pools)
        # We sync ALL layers' block tables since CoW can change them.
        for layer_idx in range(self.base.num_layers):
            page_table = seq.page_table[layer_idx]
            if page_table:
                n = len(page_table)
                self.metadata.block_tables[row, :n] = torch.tensor(
                    page_table, dtype=torch.int32, device=self.base.device,
                )

    def _sync_block_table(self, seq_id: int, layer_idx: int):
        """Sync just the block table for a specific layer after CoW."""
        seq = self.base._sequences[seq_id]
        row = self.metadata.get_row(seq_id)
        page_table = seq.page_table[layer_idx]
        if page_table:
            n = len(page_table)
            self.metadata.block_tables[row, :n] = torch.tensor(
                page_table, dtype=torch.int32, device=self.base.device,
            )

    # ---- Triton decode attention (replaces build_packed_kv + flash) ------

    def decode_attention(
        self,
        q: torch.Tensor,           # [N, num_q_heads, head_dim]
        seq_ids: list[int],
        layer_idx: int,
    ) -> torch.Tensor:
        """
        Run paged attention decode via the Triton kernel.

        This replaces the entire ``build_packed_kv`` → ``flash_attn_varlen_func``
        pipeline.  The kernel reads K/V directly from the paged pools using
        the GPU-resident block tables.

        Args:
            q:         Query vectors, [N, num_q_heads, head_dim].
            seq_ids:   List of N sequence IDs (same order as q).
            layer_idx: Which transformer layer we're computing attention for.

        Returns:
            attn_output: [N, num_q_heads, head_dim]
        """
        # Get the pool views in Triton layout (zero-copy permute)
        k_pool, v_pool = self._get_triton_pools(layer_idx)

        # Build the batch→row mapping tensor (tiny: N int64s)
        seq_row_map = self.metadata.get_row_map(seq_ids)

        # For the Triton kernel, we need per-layer block tables.
        # IMPORTANT: The base manager has per-layer page tables that can
        # differ due to CoW.  The metadata.block_tables is synced from
        # layer 0 by default.  For correctness with per-layer CoW, we
        # need layer-specific block tables.
        #
        # Optimization: if no CoW has happened (common case for decode
        # after the first token post-fork), all layers share the same
        # block table and we can use metadata.block_tables directly.
        # For safety, we always use the layer-specific tables here.
        layer_block_tables = self._build_layer_block_tables(seq_ids, layer_idx)

        return paged_attention_decode(
            q=q,
            k_pool=k_pool,
            v_pool=v_pool,
            block_tables=layer_block_tables,
            context_lens=self.metadata.context_lens,
            seq_row_map=seq_row_map,
            num_kv_heads=self.base.num_key_value_heads,
        )

    def _build_layer_block_tables(
        self, seq_ids: list[int], layer_idx: int,
    ) -> torch.Tensor:
        """
        Build a compact block_tables tensor for the given layer.

        In the common case (no CoW divergence between layers), this returns
        ``self.metadata.block_tables`` directly.  When CoW has caused
        layer-specific differences, we build a corrected table.

        TODO: For maximum performance, maintain per-layer GPU block tables
        and update them incrementally during append_tokens_batched.
        """
        # Quick check: are any sequences' layer tables different from layer 0?
        needs_layer_specific = False
        for sid in seq_ids:
            seq = self.base._sequences[sid]
            if seq.page_table[layer_idx] != seq.page_table[0]:
                needs_layer_specific = True
                break

        if not needs_layer_specific:
            return self.metadata.block_tables

        # Build a corrected table — still fast because we only touch
        # the rows for active sequences (typically < 64).
        bt = self.metadata.block_tables.clone()
        for sid in seq_ids:
            seq = self.base._sequences[sid]
            row = self.metadata.get_row(sid)
            pt = seq.page_table[layer_idx]
            if pt:
                bt[row, :len(pt)] = torch.tensor(
                    pt, dtype=torch.int32, device=self.base.device,
                )
        return bt

    # ---- Pass-through for methods the rest of the codebase calls ---------

    def build_packed_kv(self, seq_ids, layer_idx, **kwargs):
        """
        Fallback: delegates to the base manager's gather-based path.
        Use ``decode_attention`` instead for the Triton fast path.
        """
        return self.base.build_packed_kv(seq_ids, layer_idx)

    def trim_to_window(self, seq_id: int, window_size: int):
        self.base.trim_to_window(seq_id, window_size)
        self._sync_metadata_for_seq(seq_id)

    def get_stats(self):
        return self.base.get_stats()

    @property
    def num_layers(self):
        return self.base.num_layers

    @property
    def num_key_value_heads(self):
        return self.base.num_key_value_heads

    @property
    def head_dim(self):
        return self.base.head_dim

    @property
    def block_size(self):
        return self.base.block_size

    @property
    def device(self):
        return self.base.device

    @property
    def dtype(self):
        return self.base.dtype

    @property
    def allocator(self):
        return self.base.allocator

    @property
    def k_pools(self):
        return self.base.k_pools

    @property
    def v_pools(self):
        return self.base.v_pools

    @property
    def _sequences(self):
        return self.base._sequences

    def __repr__(self):
        return f"TritonPagedCacheManager(base={self.base!r})"