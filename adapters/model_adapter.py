"""
Model Adapter System
====================

Provides a unified interface for different HuggingFace model architectures
to work with the paged KV-cache inference pipeline.

Instead of monkey-patching attention modules (the old ``inject_paged_attention``
approach), each model family gets a concrete ``ModelAdapter`` subclass that
knows how to:

  1. Extract structural components (embeddings, layers, norms, lm_head).
  2. Project Q, K, V from hidden states.
  3. Apply RoPE (standard cos/sin, pre-rotated Gemma style, or none/internal).
  4. Run prefill attention and decode attention via FlashAttention.

The pipeline (``PagedModelWrapper``) calls the adapter's uniform interface
and never touches model internals directly.

Usage::

    adapter = get_adapter(model)
    wrapper = PagedModelWrapper(model, kv_cache_mgr, adapter, eos_token_id=eos)
"""

from __future__ import annotations

import abc
from typing import Optional

import torch
from einops import rearrange

try:
    from flash_attn import flash_attn_func, flash_attn_varlen_func
except ImportError:
     "Flash attention not found! Please install with `pip install flash_attn --no-build-isolation`!"


# Inheritor Class

class ModelAdapter(abc.ABC):
    """
    Abstract base that wraps a HuggingFace CausalLM model and exposes a
    uniform interface for the paged inference pipeline.

    Subclasses must implement the abstract methods / properties below.
    Most models share 90 %+ of the logic; the base class provides concrete
    defaults for the common (Llama-like) pattern so that new adapters only
    need to override the bits that differ.
    """

    def __init__(self, model):
        self.model = model
        self.config = model.config
        self.device = next(model.parameters()).device
        self.dtype = next(model.parameters()).dtype

        # Cache these once — subclasses set them in their __init__
        self._embeddings = None
        self._layers = None
        self._final_norm = None
        self._lm_head = None

    # ------------------------------------------------------------------
    #  Structural accessors (must be set by subclass __init__)
    # ------------------------------------------------------------------

    @property
    def embeddings(self) -> torch.nn.Module:
        return self._embeddings

    @property
    def layers(self) -> torch.nn.ModuleList:
        return self._layers

    @property
    def final_norm(self) -> torch.nn.Module:
        return self._final_norm

    @property
    def lm_head(self) -> torch.nn.Module:
        return self._lm_head

    # ------------------------------------------------------------------
    #  Per-layer accessors
    # ------------------------------------------------------------------

    @abc.abstractmethod
    def get_self_attn(self, layer) -> torch.nn.Module:
        """Return the self-attention sub-module from a decoder layer."""
        ...

    @abc.abstractmethod
    def get_mlp(self, layer) -> torch.nn.Module:
        """Return the MLP sub-module from a decoder layer."""
        ...

    @abc.abstractmethod
    def apply_pre_attn_norm(self, layer, hidden: torch.Tensor) -> torch.Tensor:
        """Apply the pre-attention layer norm."""
        ...

    @abc.abstractmethod
    def apply_pre_mlp_norm(self, layer, hidden: torch.Tensor) -> torch.Tensor:
        """Apply the pre-MLP (post-attention) layer norm."""
        ...

    def apply_post_attn_residual(
        self, layer, residual: torch.Tensor, attn_out: torch.Tensor
    ) -> torch.Tensor:
        """Combine residual + attention output.  Override for non-standard patterns."""
        return residual + attn_out

    def apply_post_mlp_residual(
        self, layer, residual: torch.Tensor, mlp_out: torch.Tensor
    ) -> torch.Tensor:
        """Combine residual + MLP output.  Override for non-standard patterns."""
        return residual + mlp_out

    # ------------------------------------------------------------------
    #  Geometry helpers
    # ------------------------------------------------------------------

    @property
    def num_layers(self) -> int:
        return self.config.num_hidden_layers

    @property
    def num_attention_heads(self) -> int:
        return self.config.num_attention_heads

    @property
    def num_key_value_heads(self) -> int:
        return getattr(self.config, "num_key_value_heads", None) or self.num_attention_heads
    
    @property
    def num_hidden_layers(self) -> int:
        return getattr(self.config, "num_hidden_layers", None) 
    
    @property
    def head_dim(self) -> int:
        return getattr(self.config, "head_dim", self.config.hidden_size // self.num_attention_heads)

    @property
    def hidden_size(self) -> int:
        return self.config.hidden_size

    # ------------------------------------------------------------------
    #  Attention projection + RoPE  (the core model-specific logic)
    # ------------------------------------------------------------------

    @abc.abstractmethod
    def project_qkv(
        self,
        attn: torch.nn.Module,
        hidden_states: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Project hidden_states → (Q, K, V), each shaped
        (batch, seq, heads, head_dim).  Includes any Q/K norms but
        does NOT apply RoPE.
        """
        ...

    @abc.abstractmethod
    def apply_rope(
        self,
        attn: torch.nn.Module,
        q: Optional[torch.Tensor],
        k: Optional[torch.Tensor],
        position_ids: torch.Tensor,
        v_for_shape: Optional[torch.Tensor] = None,
    ) -> tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
        """
        Apply rotary positional embeddings to Q and/or K.

        Args:
            attn:         the attention module (holds the rope sub-module).
            q:            (batch, seq, heads, dim) or None.
            k:            (batch, seq, heads, dim) or None.
            position_ids: (batch, seq) long tensor of position indices.
            v_for_shape:  value tensor used as shape hint for standard-style
                          RoPE modules that take (hidden, position_ids).

        Returns:
            (q_rotated, k_rotated) — None where input was None.
        """
        ...

    # ------------------------------------------------------------------
    #  Flash-attention wrappers (shared — rarely need overriding)
    # ------------------------------------------------------------------

    def flash_prefill(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        window_size: Optional[int] = None,
    ) -> torch.Tensor:
        """Causal flash attention for prefill.  (B, S, H, D) layout."""
        window = (-1, -1) if window_size is None else (window_size - 1, 0)
        return flash_attn_func(q, k, v, causal=True, window_size=window)

    def flash_decode(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        cu_seqlens_q: torch.Tensor,
        cu_seqlens_k: torch.Tensor,
        max_seqlen_k: int,
        N: int,
        window_size: Optional[int] = None,
    ) -> torch.Tensor:
        """Varlen flash attention for batched decode."""
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

    # ------------------------------------------------------------------
    #  High-level ops used by PagedModelWrapper
    # ------------------------------------------------------------------

    def forward_paged_prefill(
        self,
        attn: torch.nn.Module,
        hidden_states: torch.Tensor,
        layer_idx: int,
        window_size: Optional[int] = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Full prefill attention for one layer.

        Returns:
            attn_output: (1, seq_len, hidden_size)
            k_new:       (seq_len, num_kv_heads, head_dim) — post-RoPE K for cache
            v_new:       (seq_len, num_kv_heads, head_dim) — V for cache
        """
        batch_size, seq_len, _ = hidden_states.size()

        q, k, v = self.project_qkv(attn, hidden_states)

        position_ids = torch.arange(
            seq_len, device=q.device, dtype=torch.long
        ).unsqueeze(0)

        q_rot, k_rot = self.apply_rope(attn, q, k, position_ids, v_for_shape=v)

        o = self.flash_prefill(q_rot, k_rot, v, window_size)

        # Extract K/V for the paged cache — drop batch dim.
        k_new = k_rot[0]  # (seq_len, num_kv_heads, head_dim)
        v_new = v[0]

        o = o.reshape(1, seq_len, -1)
        o = attn.o_proj(o)
        return o, k_new, v_new

    def project_kv_for_cache(
        self,
        attn: torch.nn.Module,
        hidden_states: torch.Tensor,
        seqlen_offsets: list[int],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Project K/V + RoPE for new decode tokens (no Q, no attention).

        Args:
            attn:            attention sub-module.
            hidden_states:   (1, N, D) normed hidden states.
            seqlen_offsets:  list of N position offsets.

        Returns:
            k: (N, num_kv_heads, head_dim) post-RoPE
            v: (N, num_kv_heads, head_dim)
        """
        _, k, v = self.project_qkv(attn, hidden_states)
        # k, v are (1, N, heads, dim).  Reshape to (N, 1, heads, dim) so
        # rope sees batch=N, seq=1.
        k_4d = k.permute(1, 0, 2, 3)
        v_4d = v.permute(1, 0, 2, 3)

        position_ids = torch.tensor(
            seqlen_offsets, device=k.device, dtype=torch.long
        ).unsqueeze(1)  # (N, 1)

        _, k_rot_4d = self.apply_rope(attn, None, k_4d, position_ids, v_for_shape=v_4d)

        k_out = k_rot_4d.squeeze(1)  # (N, heads, dim)
        v_out = v_4d.squeeze(1)
        return k_out, v_out

    def forward_paged_decode(
        self,
        attn: torch.nn.Module,
        hidden_states: torch.Tensor,
        kv_cache_mgr,
        seq_ids: list[int],
        seqlen_offsets: list[int],
        layer_idx: int,
        window_size: Optional[int] = None,
        cached_gather_indices=None,
    ) -> torch.Tensor:
        """
        Batched decode attention for one layer: project Q, apply RoPE,
        gather packed KV from cache, run varlen flash attention.

        Returns:
            attn_output: (1, N, hidden_size)
        """
        N = len(seq_ids)
        _, total_q, _ = hidden_states.size()

        q, _, _ = self.project_qkv(attn, hidden_states)
        # q is (1, N, num_heads, head_dim) — only need Q

        position_ids = torch.tensor(
            seqlen_offsets, device=q.device, dtype=torch.long
        ).unsqueeze(1)  # (N, 1)

        # Permute to (N, 1, heads, dim) for per-sequence RoPE
        q_4d = q.permute(1, 0, 2, 3)
        q_rot_4d, _ = self.apply_rope(attn, q_4d, None, position_ids, v_for_shape=q_4d)
        q_rot = q_rot_4d.squeeze(1)  # (N, num_heads, head_dim)

        packed_k, packed_v, cu_seqlens_k, max_seqlen_k = kv_cache_mgr.build_packed_kv(
            seq_ids, layer_idx, cached_metadata=cached_gather_indices,
        )

        cu_seqlens_q = torch.arange(0, N + 1, device=q.device, dtype=torch.int32)

        o = self.flash_decode(
            q_rot, packed_k, packed_v,
            cu_seqlens_q, cu_seqlens_k,
            max_seqlen_k, N, window_size,
        )

        o = o.unsqueeze(0).reshape(1, total_q, -1)
        return attn.o_proj(o)