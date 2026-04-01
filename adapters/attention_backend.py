"""
Attention Backend Abstraction
=============================

Auto-detects whether ``flash_attn`` is available and exposes a unified
pair of helpers — ``attn_prefill`` and ``attn_decode`` — that dispatch to
FlashAttention when possible and fall back to PyTorch SDPA otherwise.

The SDPA path works on **any** PyTorch backend (CPU, MPS, CUDA-without-
flash-attn) and requires no compiled extensions.

Usage:

    from attention_backend import BACKEND, attn_prefill, attn_decode

    # BACKEND is "flash" or "sdpa" — purely informational.
    o = attn_prefill(q, k, v, window_size=None)     # prefill (causal)
    o = attn_decode(q, k, v, cu_q, cu_k, max_k, N)  # varlen decode
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn.functional as F

# ---------------------------------------------------------------------------
#  Detect backend
# ---------------------------------------------------------------------------

try:
    from flash_attn import flash_attn_func, flash_attn_varlen_func
    BACKEND = "flash"
except ImportError:
    BACKEND = "sdpa"


# ---------------------------------------------------------------------------
#  SDPA helpers (used when flash_attn is not installed)
# ---------------------------------------------------------------------------

def _sdpa_prefill(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    window_size: Optional[int] = None,
) -> torch.Tensor:
    """
    Causal self-attention for prefill via ``F.scaled_dot_product_attention``.

    Args:
        q, k, v: (batch, seq, heads, head_dim)  — FlashAttention layout.
                  K/V may have fewer heads than Q (GQA).
        window_size: if not None, use sliding-window causal mask of this width.

    Returns:
        output: (batch, seq, num_q_heads, head_dim)
    """
    # SDPA expects (batch, heads, seq, head_dim)
    q = q.transpose(1, 2)  # (B, num_heads, S, D)
    k = k.transpose(1, 2)  # (B, num_kv_heads, S, D)
    v = v.transpose(1, 2)

    num_heads = q.size(1)
    num_kv_heads = k.size(1)
    seq_len = q.size(2)

    # Expand KV heads for GQA when num_heads > num_kv_heads.
    # This makes it work on all PyTorch versions without enable_gqa.
    if num_kv_heads != num_heads:
        gqa_factor = num_heads // num_kv_heads
        # (B, kv_heads, S, D) → (B, kv_heads, 1, S, D) → expand → (B, heads, S, D)
        k = k.unsqueeze(2).expand(-1, -1, gqa_factor, -1, -1).reshape(
            k.size(0), num_heads, seq_len, k.size(-1)
        )
        v = v.unsqueeze(2).expand(-1, -1, gqa_factor, -1, -1).reshape(
            v.size(0), num_heads, seq_len, v.size(-1)
        )

    if window_size is not None and window_size < seq_len:
        # Build a sliding-window causal mask.
        # Position i can attend to positions max(0, i - window_size + 1) .. i.
        row = torch.arange(seq_len, device=q.device)
        col = torch.arange(seq_len, device=q.device)
        mask = (col <= row.unsqueeze(1)) & (col >= (row - window_size + 1).unsqueeze(1))
        attn_mask = torch.where(
            mask, torch.zeros(1, device=q.device, dtype=q.dtype),
            torch.full((1,), float("-inf"), device=q.device, dtype=q.dtype),
        )
        o = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask)
    else:
        o = F.scaled_dot_product_attention(q, k, v, is_causal=True)

    # Back to FA layout: (batch, heads, seq, head_dim) → (batch, seq, heads, head_dim)
    return o.transpose(1, 2)


def _sdpa_decode(
    q_flat: torch.Tensor,
    packed_k: torch.Tensor,
    packed_v: torch.Tensor,
    cu_seqlens_q: torch.Tensor,
    cu_seqlens_k: torch.Tensor,
    max_seqlen_k: int,
    N: int,
    window_size: Optional[int] = None,
) -> torch.Tensor:
    """
    Batched single-query decode attention via per-sequence SDPA loops.

    ``flash_attn_varlen_func`` packs N variable-length sequences into a
    single flat tensor.  SDPA has no native varlen mode, so we pad to
    ``max_seqlen_k`` and call SDPA once per sequence.  For the typical
    decode batch sizes (< 64 sequences), this is perfectly fast enough
    on CPU / MPS.

    Args:
        q_flat:       (N, num_heads, head_dim)
        packed_k:     (total_tokens, num_kv_heads, head_dim)
        packed_v:     (total_tokens, num_kv_heads, head_dim)
        cu_seqlens_q: (N+1,) int32
        cu_seqlens_k: (N+1,) int32
        max_seqlen_k: int
        N:            number of sequences
        window_size:  unused for decode (single query always in-window)

    Returns:
        output: (N, num_heads, head_dim)
    """
    num_heads = q_flat.shape[1]
    num_kv_heads = packed_k.shape[1]
    head_dim = q_flat.shape[2]
    device = q_flat.device
    dtype = q_flat.dtype

    # GQA repeat factor: if num_heads > num_kv_heads, we need to expand K/V.
    gqa_factor = num_heads // num_kv_heads

    outputs = []
    cu_k = cu_seqlens_k.tolist()

    for i in range(N):
        # Extract this sequence's K/V slice from the packed tensor
        k_start, k_end = cu_k[i], cu_k[i + 1]
        seq_len = k_end - k_start

        ki = packed_k[k_start:k_end]  # (seq_len, num_kv_heads, head_dim)
        vi = packed_v[k_start:k_end]  # (seq_len, num_kv_heads, head_dim)

        # Expand KV heads for GQA if needed
        if gqa_factor > 1:
            # (seq_len, kv_heads, dim) → (seq_len, kv_heads, 1, dim) → repeat → reshape
            ki = ki.unsqueeze(2).expand(-1, -1, gqa_factor, -1).reshape(seq_len, num_heads, head_dim)
            vi = vi.unsqueeze(2).expand(-1, -1, gqa_factor, -1).reshape(seq_len, num_heads, head_dim)

        # Reshape for SDPA: (1, heads, seq, dim)
        qi = q_flat[i].unsqueeze(0).unsqueeze(2)          # (1, heads, 1, dim)
        ki = ki.permute(1, 0, 2).unsqueeze(0)              # (1, heads, seq_len, dim)
        vi = vi.permute(1, 0, 2).unsqueeze(0)              # (1, heads, seq_len, dim)

        # Single-query decode: is_causal=False because Q has length 1 and
        # should attend to all K positions (they are already the causal
        # history from the KV cache).
        oi = F.scaled_dot_product_attention(qi, ki, vi, is_causal=False)
        outputs.append(oi.squeeze(0).squeeze(1))           # (heads, dim)

    return torch.stack(outputs, dim=0)  # (N, heads, dim)


# ---------------------------------------------------------------------------
#  Unified dispatch functions
# ---------------------------------------------------------------------------

def attn_prefill(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    window_size: Optional[int] = None,
) -> torch.Tensor:
    """
    Causal attention for prefill.  (batch, seq, heads, head_dim) layout.

    Dispatches to FlashAttention when available, SDPA otherwise.
    """
    if BACKEND == "flash":
        window = (-1, -1) if window_size is None else (window_size - 1, 0)
        return flash_attn_func(q, k, v, causal=True, window_size=window)
    return _sdpa_prefill(q, k, v, window_size)


def attn_decode(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    cu_seqlens_q: torch.Tensor,
    cu_seqlens_k: torch.Tensor,
    max_seqlen_k: int,
    N: int,
    window_size: Optional[int] = None,
) -> torch.Tensor:
    """
    Variable-length batched decode attention.

    Dispatches to ``flash_attn_varlen_func`` when available, otherwise
    falls back to a per-sequence SDPA loop.

    Args:
        q: (N, num_heads, head_dim)   — one query per sequence.
        k: (total_tokens, num_kv_heads, head_dim) — packed cache keys.
        v: (total_tokens, num_kv_heads, head_dim) — packed cache values.
        cu_seqlens_q: (N+1,)
        cu_seqlens_k: (N+1,)
        max_seqlen_k: int
        N: number of sequences
        window_size: sliding window size (or None for full context)
    """
    if BACKEND == "flash":
        window = (-1, -1) if window_size is None else (window_size - 1, 0)
        return flash_attn_varlen_func(
            q, k, v,
            cu_seqlens_q=cu_seqlens_q,
            cu_seqlens_k=cu_seqlens_k,
            max_seqlen_q=1,
            max_seqlen_k=max_seqlen_k,
            causal=True,
            window_size=window,
        )
    return _sdpa_decode(q, k, v, cu_seqlens_q, cu_seqlens_k, max_seqlen_k, N, window_size)