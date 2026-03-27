# Copyright (c) 2023-2025, Songlin Yang, Yu Zhang
#
# Modified to add paged-attention batched decode path.
# Original forward() is preserved for HuggingFace compatibility.
# New forward_paged() handles both prefill and batched decode
# via PagedKVCacheManager + flash_attn_varlen_func.
#
# PATCHED: Replaced fla.modules.{RMSNorm, RotaryEmbedding} with
# Qwen3-native implementations so HF checkpoint weights load
# correctly and RoPE frequencies / attention_scaling match.

from __future__ import annotations

import math
import warnings
from typing import TYPE_CHECKING, Optional

import torch
import torch.nn as nn
from einops import rearrange
from transformers.utils import logging

# We still use fla's pad/unpad for the HF-compatible forward()
from fla.layers.utils import pad_input, unpad_input
from fla.ops.utils.index import prepare_lens_from_mask

if TYPE_CHECKING:
    from fla.models.utils import Cache
    from paged_kv_cache import PagedKVCacheManager

try:
    from flash_attn import flash_attn_func, flash_attn_varlen_func
except ImportError:
    warnings.warn(
        "Flash Attention is not installed. Please install it via "
        "`pip install flash-attn --no-build-isolation`",
        category=ImportWarning,
    )
    flash_attn_func = None

logger = logging.get_logger(__name__)


# ======================================================================
# Qwen3-compatible RMSNorm
# ======================================================================
# Uses the same parameter name (`weight`) and epsilon attribute
# (`variance_epsilon`) as Qwen3RMSNorm in the HF checkpoint, so
# `from_pretrained` loads the q_norm / k_norm weights correctly.
# ======================================================================

class Qwen3RMSNorm(nn.Module):
    """RMSNorm matching the HuggingFace Qwen3 checkpoint layout."""

    def __init__(self, hidden_size: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(
            variance + self.variance_epsilon
        )
        return self.weight * hidden_states.to(input_dtype)


# ======================================================================
# Qwen3-compatible Rotary Embedding
# ======================================================================
# Computes inv_freq from the full `rope_parameters` dict (including
# rope_theta and attention_scaling), matching Qwen3RotaryEmbedding.
#
# Provides TWO application interfaces:
#   1. apply_rotary_standard(q, k, cos, sin)
#        HF-style: tensors are (batch, heads, seq, dim), cos/sin are
#        (batch, seq, dim).  Used by the HF-compatible forward().
#   2. apply_rotary_varlen(x, cos_full, positions)
#        Varlen-style: x is (total_tokens, heads, dim), positions is
#        a 1-D tensor of per-token position ids.  Used by the paged
#        prefill and decode paths.
# ======================================================================

def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


class Qwen3CompatRotaryEmbedding(nn.Module):
    """
    RoPE module that replicates Qwen3RotaryEmbedding's frequency
    computation and supports both standard and varlen application.

    inv_freq is computed lazily on first use, so it works correctly
    even when the model is initialized on meta device during
    from_pretrained().
    """

    def __init__(
        self,
        head_dim: int,
        rope_theta: float = 10000.0,
        max_position_embeddings: int = 32768,
        attention_scaling: float = 1.0,
        device: Optional[torch.device] = None,
    ):
        super().__init__()
        self.head_dim = head_dim
        self.rope_theta = rope_theta
        self.max_position_embeddings = max_position_embeddings
        self.attention_scaling = attention_scaling
        # inv_freq will be built lazily on the correct device
        self._inv_freq: Optional[torch.Tensor] = None

    def _get_inv_freq(self, device: torch.device) -> torch.Tensor:
        """Compute or return cached inv_freq on the given device."""
        if self._inv_freq is not None and self._inv_freq.device == device:
            return self._inv_freq
        self._inv_freq = 1.0 / (
            self.rope_theta ** (
                torch.arange(0, self.head_dim, 2, dtype=torch.int64)
                .to(device=device, dtype=torch.float32)
                / self.head_dim
            )
        )
        return self._inv_freq

    # ----- cos/sin cache builder -----

    def _build_cos_sin(
        self,
        positions: torch.Tensor,
        device: torch.device,
        dtype: torch.dtype,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Build cos/sin tensors for arbitrary position ids.

        Args:
            positions: int tensor of any shape; each element is a
                       position index.
        Returns:
            cos, sin: same shape as positions + (head_dim,), in dtype.
        """
        inv_freq = self._get_inv_freq(device)

        # positions may be (batch, seq) or (total_tokens,)
        orig_shape = positions.shape
        pos_flat = positions.reshape(-1).float()  # (P,)

        # (P, head_dim/2)
        freqs = torch.outer(pos_flat, inv_freq)
        # (P, head_dim)
        emb = torch.cat((freqs, freqs), dim=-1)

        cos = (emb.cos() * self.attention_scaling).to(dtype)
        sin = (emb.sin() * self.attention_scaling).to(dtype)

        # Restore leading dims
        cos = cos.view(*orig_shape, self.head_dim)
        sin = sin.view(*orig_shape, self.head_dim)
        return cos, sin

    # ----- Standard HF-style application -----

    def apply_standard(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        position_ids: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Apply RoPE the HuggingFace way.

        Args:
            q: (batch, seq, num_heads, head_dim)
            k: (batch, seq, num_kv_heads, head_dim)
            position_ids: (batch, seq) int tensor.

        Returns:
            q_rot, k_rot: same shapes as q, k.
        """
        cos, sin = self._build_cos_sin(
            position_ids, q.device, q.dtype,
        )
        # cos, sin: (batch, seq, head_dim) → unsqueeze for heads dim
        cos = cos.unsqueeze(2)  # (batch, seq, 1, head_dim)
        sin = sin.unsqueeze(2)

        q_rot = (q * cos) + (_rotate_half(q) * sin)
        k_rot = (k * cos) + (_rotate_half(k) * sin)
        return q_rot, k_rot

    # ----- Varlen application (paged paths) -----

    def apply_varlen(
        self,
        x: torch.Tensor,
        positions: torch.Tensor,
    ) -> torch.Tensor:
        """
        Apply RoPE to a packed varlen tensor.

        Args:
            x: (total_tokens, num_heads, head_dim)
            positions: (total_tokens,) int tensor — one position per token.

        Returns:
            x_rot: same shape as x.
        """
        cos, sin = self._build_cos_sin(positions, x.device, x.dtype)
        # cos, sin: (total_tokens, head_dim) → unsqueeze for heads
        cos = cos.unsqueeze(1)  # (total_tokens, 1, head_dim)
        sin = sin.unsqueeze(1)
        return (x * cos) + (_rotate_half(x) * sin)


# ======================================================================
# PagedAttention — patched to use Qwen3-compatible modules
# ======================================================================

class PagedAttention(nn.Module):

    def __init__(
        self,
        hidden_size: int = 2048,
        num_heads: int = 32,
        num_kv_heads: int | None = None,
        head_dim: int | None = None,
        qkv_bias: bool = False,
        qk_norm: bool = False,
        window_size: int | None = None,
        # --- Qwen3-specific RoPE config ---
        rope_theta: float | None = 10000.,
        max_position_embeddings: int | None = None,
        attention_scaling: float = 1.0,
        rms_norm_eps: float = 1e-6,
        layer_idx: int = None,
    ):
        super().__init__()

        self.hidden_size = hidden_size
        self.num_heads = num_heads
        if num_kv_heads is None:
            self.num_kv_heads = self.num_heads
        else:
            self.num_kv_heads = num_kv_heads
        self.num_kv_groups = num_heads // self.num_kv_heads

        # Use explicit head_dim from config (Qwen3 sets it to 128
        # regardless of hidden_size // num_heads).
        if head_dim is not None:
            self.head_dim = head_dim
        else:
            self.head_dim = self.hidden_size // self.num_heads

        # q_proj output dim = num_heads * head_dim (may differ from hidden_size!)
        self.q_dim = self.num_heads * self.head_dim
        self.kv_dim = self.num_kv_heads * self.head_dim
        self.qkv_bias = qkv_bias
        self.qk_norm = qk_norm

        self.window_size = window_size
        self.rope_theta = rope_theta
        self.max_position_embeddings = max_position_embeddings
        self.layer_idx = layer_idx

        if flash_attn_func is None:
            raise ImportError(
                "Please install Flash Attention via "
                "`pip install flash-attn --no-build-isolation` first"
            )

        self.q_proj = nn.Linear(
            self.hidden_size, self.q_dim, bias=self.qkv_bias,
        )
        self.k_proj = nn.Linear(
            self.hidden_size, self.kv_dim, bias=self.qkv_bias,
        )
        self.v_proj = nn.Linear(
            self.hidden_size, self.kv_dim, bias=self.qkv_bias,
        )
        self.o_proj = nn.Linear(
            self.q_dim, self.hidden_size, bias=False,
        )

        # ---- PATCHED: use Qwen3RMSNorm (same param name "weight") ----
        if qk_norm:
            self.q_norm = Qwen3RMSNorm(self.head_dim, eps=rms_norm_eps)
            self.k_norm = Qwen3RMSNorm(self.head_dim, eps=rms_norm_eps)

        # ---- PATCHED: use Qwen3-compatible RoPE ----
        self.rotary = Qwen3CompatRotaryEmbedding(
            head_dim=self.head_dim,
            rope_theta=rope_theta or 10000.0,
            max_position_embeddings=max_position_embeddings or 32768,
            attention_scaling=attention_scaling,
        )

    # ==================================================================
    # ORIGINAL FORWARD — for HuggingFace-compatible usage
    # ==================================================================

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.LongTensor | None = None,
        past_key_values: Cache | None = None,
        output_attentions: bool = False,
        use_cache: bool = False,
        **kwargs,
    ) -> tuple[torch.Tensor, torch.Tensor | None, tuple[torch.Tensor] | None]:
        if attention_mask is not None:
            assert len(attention_mask.shape) == 2, (
                "Expected attention_mask as a 0-1 matrix with shape "
                "[batch_size, seq_len] for padding purposes (0 indicating "
                "padding). Arbitrary attention masks of shape "
                "[batch_size, seq_len, seq_len] are not allowed."
            )

        batch_size, q_len, _ = hidden_states.size()

        q = rearrange(
            self.q_proj(hidden_states),
            '... (h d) -> ... h d', d=self.head_dim,
        )
        k = rearrange(
            self.k_proj(hidden_states),
            '... (h d) -> ... h d', d=self.head_dim,
        )
        v = rearrange(
            self.v_proj(hidden_states),
            '... (h d) -> ... h d', d=self.head_dim,
        )
        # q: (batch, seq, num_heads, head_dim)
        # k: (batch, seq, num_kv_heads, head_dim)

        if self.qk_norm:
            q, k = self.q_norm(q), self.k_norm(k)

        # Build position_ids
        cu_seqlens = kwargs.get('cu_seqlens')

        seqlen_offset = 0
        if past_key_values is not None:
            seqlen_offset = past_key_values.get_seq_length(self.layer_idx)
            if attention_mask is not None:
                seqlen_offset = (
                    seqlen_offset
                    + prepare_lens_from_mask(attention_mask)
                    - attention_mask.shape[-1]
                )

        # Construct position_ids: (batch, seq)
        if isinstance(seqlen_offset, torch.Tensor):
            # Per-sequence offsets from padding
            position_ids = (
                torch.arange(q_len, device=q.device).unsqueeze(0)
                + seqlen_offset.unsqueeze(1)
            )
        else:
            position_ids = torch.arange(
                seqlen_offset, seqlen_offset + q_len, device=q.device,
            ).unsqueeze(0).expand(batch_size, -1)

        # PATCHED: apply RoPE using Qwen3-compatible method
        q, k = self.rotary.apply_standard(q, k, position_ids)

        if past_key_values is not None:
            cache_has_content = (
                past_key_values.get_seq_length(self.layer_idx) > 0
            )
            k_cached, v_cached = past_key_values.update(
                attn_state=(k.flatten(-2, -1), v.flatten(-2, -1)),
                layer_idx=self.layer_idx,
                offset=q_len,
                cache_kwargs=dict(window_size=self.window_size),
            )['attn_state']
            if cache_has_content:
                k, v = k_cached, v_cached
                k = rearrange(k, '... (h d) -> ... h d', d=self.head_dim)
                v = rearrange(v, '... (h d) -> ... h d', d=self.head_dim)

        # flash_attn expects (batch, seq, heads, dim) — already in that layout
        if attention_mask is not None:
            if q.shape[1] == 1 and self.window_size is not None:
                attention_mask = attention_mask[:, -self.window_size:]
            q, (k, v), indices_q, cu_seqlens_pair, max_seq_lens = (
                unpad_input(q, (k, v), attention_mask, q_len)
            )
            cu_seqlens_q, cu_seqlens_k = cu_seqlens_pair
            max_seqlen_q, max_seqlen_k = max_seq_lens
            o = flash_attn_varlen_func(
                q, k, v,
                cu_seqlens_q=cu_seqlens_q,
                cu_seqlens_k=cu_seqlens_k,
                max_seqlen_q=max_seqlen_q,
                max_seqlen_k=max_seqlen_k,
                causal=True,
                window_size=(
                    (-1, -1)
                    if self.window_size is None
                    else (self.window_size - 1, 0)
                ),
            )
            o = pad_input(o, indices_q, batch_size, q_len)
        elif cu_seqlens is not None:
            max_seqlen = q_len
            if past_key_values is not None:
                max_seqlen = q.shape[1] + past_key_values.get_seq_length(
                    self.layer_idx
                )
            o = flash_attn_varlen_func(
                q.squeeze(0), k.squeeze(0), v.squeeze(0),
                cu_seqlens_q=cu_seqlens,
                cu_seqlens_k=cu_seqlens,
                max_seqlen_q=max_seqlen,
                max_seqlen_k=max_seqlen,
                causal=True,
                window_size=(
                    (-1, -1)
                    if self.window_size is None
                    else (self.window_size - 1, 0)
                ),
            ).unsqueeze(0)
        else:
            o = flash_attn_func(
                q, k, v,
                causal=True,
                window_size=(
                    (-1, -1)
                    if self.window_size is None
                    else (self.window_size - 1, 0)
                ),
            )

        o = o.reshape(batch_size, q_len, -1)
        o = self.o_proj(o)

        if not output_attentions:
            attentions = None

        return o, attentions, past_key_values

    # ==================================================================
    # PAGED PREFILL
    # ==================================================================

    def forward_paged_prefill(
        self,
        hidden_states: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Paged prefill: full Q/K/V projection, RoPE, causal self-attn.

        Parameters
        ----------
        hidden_states : (1, seq_len, hidden_size)

        Returns
        -------
        output : (1, seq_len, hidden_size)
        k_new  : (seq_len, num_kv_heads, head_dim) post-RoPE
        v_new  : (seq_len, num_kv_heads, head_dim)
        """
        _, seq_len, _ = hidden_states.size()

        q = rearrange(
            self.q_proj(hidden_states),
            '... (h d) -> ... h d', d=self.head_dim,
        )
        k = rearrange(
            self.k_proj(hidden_states),
            '... (h d) -> ... h d', d=self.head_dim,
        )
        v = rearrange(
            self.v_proj(hidden_states),
            '... (h d) -> ... h d', d=self.head_dim,
        )
        # q: (1, seq_len, num_heads, head_dim)
        # k: (1, seq_len, num_kv_heads, head_dim)

        if self.qk_norm:
            q, k = self.q_norm(q), self.k_norm(k)

        # Position ids for prefill: 0, 1, 2, ..., seq_len-1
        position_ids = torch.arange(
            seq_len, device=q.device, dtype=torch.long,
        ).unsqueeze(0)  # (1, seq_len)

        # PATCHED: Qwen3-compatible RoPE
        q, k = self.rotary.apply_standard(q, k, position_ids)

        window = (
            (-1, -1)
            if self.window_size is None
            else (self.window_size - 1, 0)
        )
        o = flash_attn_func(q, k, v, causal=True, window_size=window)

        k_new = k.squeeze(0)  # (seq_len, num_kv_heads, head_dim)
        v_new = v.squeeze(0)

        o = o.reshape(1, seq_len, -1)
        o = self.o_proj(o)
        return o, k_new, v_new

    # ==================================================================
    # PAGED BATCHED DECODE
    # ==================================================================

    def forward_paged_decode(
        self,
        hidden_states: torch.Tensor,
        kv_cache_mgr: 'PagedKVCacheManager',
        seq_ids: list[int],
        seqlen_offsets: list[int] | torch.Tensor,
        cached_gather_indices: tuple | None = None,
    ) -> torch.Tensor:
        """
        Paged batched decode: Q-only projection + varlen attention.

        Parameters
        ----------
        hidden_states : (1, N, hidden_size)
        kv_cache_mgr : PagedKVCacheManager (already contains this step's KV)
        seq_ids : list[int], one per active sequence
        seqlen_offsets : per-sequence RoPE offset (position of new token)
        cached_gather_indices : optional precomputed indices from
            ``kv_cache_mgr.prepare_gather_indices()``.  When provided,
            ``build_packed_kv`` skips index construction entirely.

        Returns
        -------
        output : (1, N, hidden_size)
        """
        N = len(seq_ids)
        _, total_q, _ = hidden_states.size()

        # -- Q projection --
        q = rearrange(
            self.q_proj(hidden_states),
            '... (h d) -> ... h d', d=self.head_dim,
        )
        # q: (1, N, num_heads, head_dim)

        if self.qk_norm:
            q = self.q_norm(q)

        # -- Build position tensor for each decode token --
        if isinstance(seqlen_offsets, list):
            positions = torch.tensor(
                seqlen_offsets, device=q.device, dtype=torch.long,
            )
        else:
            positions = seqlen_offsets.to(device=q.device, dtype=torch.long)
        # positions: (N,) — one position per decode token

        # -- Apply RoPE to Q only (varlen style) --
        # Squeeze to (N, num_heads, head_dim) for varlen application
        q_squeezed = q.squeeze(0)
        q_rotated = self.rotary.apply_varlen(q_squeezed, positions)

        # -- Gather packed KV from paged cache --
        packed_k, packed_v, cu_seqlens_k, max_seqlen_k = (
            kv_cache_mgr.build_packed_kv(
                seq_ids, self.layer_idx,
                cached_indices=cached_gather_indices,
            )
        )

        # -- Varlen attention --
        cu_seqlens_q = torch.arange(
            0, N + 1, device=q.device, dtype=torch.int32,
        )
        window = (
            (-1, -1)
            if self.window_size is None
            else (self.window_size - 1, 0)
        )

        o = flash_attn_varlen_func(
            q_rotated,       # (N, num_heads, head_dim)
            packed_k,
            packed_v,
            cu_seqlens_q=cu_seqlens_q,
            cu_seqlens_k=cu_seqlens_k,
            max_seqlen_q=1,
            max_seqlen_k=max_seqlen_k,
            causal=True,
            window_size=window,
        )
        # o: (N, num_heads, head_dim)
        o = o.unsqueeze(0)  # -> (1, N, num_heads, head_dim)

        o = o.reshape(1, total_q, -1)
        o = self.o_proj(o)
        return o