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
    flash_attn_func = None
    flash_attn_varlen_func = None


# ============================================================================
# 1. ABSTRACT BASE
# ============================================================================

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


# ============================================================================
# 2. ROPE HELPERS
# ============================================================================

def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def apply_rope_cos_sin(
    q: Optional[torch.Tensor],
    k: Optional[torch.Tensor],
    cos: torch.Tensor,
    sin: torch.Tensor,
) -> tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
    """
    Standard cos/sin RoPE application.
    q, k: (batch, seq, heads, dim)
    cos, sin: (batch, seq, dim) — standard HF output.
    """
    cos = cos.unsqueeze(-2)  # (batch, seq, 1, dim)
    sin = sin.unsqueeze(-2)
    q_rot = (q * cos + _rotate_half(q) * sin) if q is not None else None
    k_rot = (k * cos + _rotate_half(k) * sin) if k is not None else None
    return q_rot, k_rot


# ============================================================================
# 3. CONCRETE ADAPTERS
# ============================================================================

class LlamaAdapter(ModelAdapter):
    """
    Adapter for Llama-family models (Llama 2/3, CodeLlama, etc.)
    and models that follow the same architecture:
      - Mistral, Mixtral
      - Qwen 2/2.5/3 (standard attention variant)
      - Yi, DeepSeek, InternLM2, etc.

    Architecture pattern:
      - embed_tokens → layers[ input_layernorm → self_attn → post_attention_layernorm → mlp ] → norm → lm_head
      - self_attn has: q_proj, k_proj, v_proj, o_proj, rotary_emb
      - rotary_emb(value_placeholder, position_ids) → (cos, sin)
    """

    def __init__(self, model):
        super().__init__(model)
        base = model.model if hasattr(model, "model") else model

        # Embeddings
        for name in ["embed_tokens", "word_embeddings", "wte", "embed"]:
            if hasattr(base, name):
                self._embeddings = getattr(base, name)
                break
        if self._embeddings is None:
            raise AttributeError(f"Cannot find embeddings in {type(base).__name__}")

        self._layers = base.layers
        self._final_norm = base.norm
        self._lm_head = model.lm_head

        # Detect norm attribute names from the first layer
        first = self._layers[0]
        self._norm1_attr = self._detect_attr(first, [
            "input_layernorm", "pre_attention_layernorm",
            "ln_1", "attention_norm",
        ], "pre-attention norm")
        self._norm2_attr = self._detect_attr(first, [
            "post_attention_layernorm", "pre_feedforward_layernorm",
            "post_feedforward_layernorm", "ln_2", "ffn_norm",
        ], "post-attention norm")

        # Detect RoPE attribute on the attention module
        attn0 = self.get_self_attn(first)
        self._rope_attr = self._detect_attr(attn0, [
            "rotary_emb", "rotary_fn", "rotary", "rope",
        ], "RoPE module")

        # Cache head_dim from model if available
        if not hasattr(attn0, "head_dim"):
            attn0.head_dim = self.head_dim

    @staticmethod
    def _detect_attr(module, candidates: list[str], label: str) -> str:
        for name in candidates:
            if hasattr(module, name):
                return name
        raise AttributeError(
            f"Could not find {label} in {type(module).__name__}. "
            f"Checked: {candidates}"
        )

    # -- Layer accessors --
    def get_self_attn(self, layer):
        return layer.self_attn

    def get_mlp(self, layer):
        return layer.mlp

    def apply_pre_attn_norm(self, layer, hidden):
        return getattr(layer, self._norm1_attr)(hidden)

    def apply_pre_mlp_norm(self, layer, hidden):
        return getattr(layer, self._norm2_attr)(hidden)

    # -- QKV projection --
    def project_qkv(self, attn, hidden_states):
        hd = self.head_dim
        q = rearrange(attn.q_proj(hidden_states), "... (h d) -> ... h d", d=hd)
        k = rearrange(attn.k_proj(hidden_states), "... (h d) -> ... h d", d=hd)
        v = rearrange(attn.v_proj(hidden_states), "... (h d) -> ... h d", d=hd)
        if hasattr(attn, "q_norm"):
            q = attn.q_norm(q)
        if hasattr(attn, "k_norm"):
            k = attn.k_norm(k)
        return q, k, v

    # -- RoPE --
    def apply_rope(self, attn, q, k, position_ids, v_for_shape=None):
        rope_module = getattr(attn, self._rope_attr)
        if v_for_shape is None:
            # Use q or k as shape hint
            v_for_shape = q if q is not None else k
        cos, sin = rope_module(v_for_shape, position_ids)
        return apply_rope_cos_sin(q, k, cos, sin)


class GemmaAdapter(LlamaAdapter):
    """
    Adapter for Gemma / Gemma-2 models.

    Key differences from Llama:
      - rotary_emb(q, k, position_ids) → (q_rotated, k_rotated)
        (pre-rotated style — no separate cos/sin)
      - Gemma-2 has pre_feedforward_layernorm + post_feedforward_layernorm
        and a post_attention_layernorm that acts as a *post*-norm on attn
        output rather than a pre-MLP norm.  The residual path is:
          residual + post_attention_layernorm(attn_out)
        then:
          residual + post_feedforward_layernorm(mlp(pre_feedforward_layernorm(hidden)))
    """

    def __init__(self, model):
        super().__init__(model)

        # Gemma-2 uses a post-norm on attention output
        first = self._layers[0]
        self._has_post_attn_norm = hasattr(first, "post_attention_layernorm")
        self._has_post_ffn_norm = hasattr(first, "post_feedforward_layernorm")

        # For Gemma-2, the pre-MLP norm is pre_feedforward_layernorm
        if hasattr(first, "pre_feedforward_layernorm"):
            self._norm2_attr = "pre_feedforward_layernorm"

    def apply_rope(self, attn, q, k, position_ids, v_for_shape=None):
        rope_module = getattr(attn, self._rope_attr)

        # Gemma-style: rope(q, k, position_ids) → (q_rot, k_rot)
        # We need to handle None q or k by passing through zeros
        if q is None:
            dummy_q = torch.zeros_like(k)
            _, k_rot = rope_module(dummy_q, k, position_ids)
            return None, k_rot
        if k is None:
            dummy_k = torch.zeros_like(q)
            q_rot, _ = rope_module(q, dummy_k, position_ids)
            return q_rot, None
        return rope_module(q, k, position_ids)

    def apply_post_attn_residual(self, layer, residual, attn_out):
        if self._has_post_attn_norm:
            return residual + layer.post_attention_layernorm(attn_out)
        return residual + attn_out

    def apply_post_mlp_residual(self, layer, residual, mlp_out):
        if self._has_post_ffn_norm:
            return residual + layer.post_feedforward_layernorm(mlp_out)
        return residual + mlp_out


class Phi3Adapter(LlamaAdapter):
    """
    Adapter for Phi-3 / Phi-3.5 models.

    Key differences:
      - Uses a fused qkv_proj instead of separate q/k/v projections.
      - May have SuRoPE (scaled/extended rotary) but the interface is
        still (hidden, position_ids) → (cos, sin).
    """

    def project_qkv(self, attn, hidden_states):
        hd = self.head_dim
        if hasattr(attn, "qkv_proj"):
            qkv = attn.qkv_proj(hidden_states)
            # Phi-3: qkv is (batch, seq, (num_heads + 2*num_kv_heads) * head_dim)
            num_q = self.num_attention_heads
            num_kv = self.num_key_value_heads
            q, k, v = qkv.split(
                [num_q * hd, num_kv * hd, num_kv * hd], dim=-1
            )
            q = rearrange(q, "... (h d) -> ... h d", d=hd)
            k = rearrange(k, "... (h d) -> ... h d", d=hd)
            v = rearrange(v, "... (h d) -> ... h d", d=hd)
        else:
            # Fall back to separate projections
            return super().project_qkv(attn, hidden_states)
        if hasattr(attn, "q_norm"):
            q = attn.q_norm(q)
        if hasattr(attn, "k_norm"):
            k = attn.k_norm(k)
        return q, k, v


class CohereAdapter(LlamaAdapter):
    """
    Adapter for Cohere Command-R models.

    Key difference:
      - Uses a single layernorm before both attention and MLP (parallel
        residual pattern): both attn and MLP receive the same normed input.
      - The residual is: hidden = hidden + attn_out + mlp_out
    """

    def __init__(self, model):
        super().__init__(model)
        # Command-R uses input_layernorm for the single shared norm
        self._parallel_residual = True

    def apply_pre_mlp_norm(self, layer, hidden):
        # In parallel residual, MLP uses the same normed input as attention.
        # The wrapper handles this by passing the pre-attn normed tensor.
        # This method won't actually be called in the standard pipeline flow
        # when parallel residual is active — see PagedModelWrapper.
        return getattr(layer, self._norm1_attr)(hidden)


class MistralAdapter(LlamaAdapter):
    """
    Adapter for Mistral / Mixtral models.

    Architecturally identical to Llama for our purposes. Sliding window
    attention is handled via the window_size parameter passed through the
    pipeline, not in the adapter.
    """
    pass


class Qwen2Adapter(LlamaAdapter):
    """
    Adapter for Qwen2 / Qwen2.5 / Qwen3 models.

    Architecturally identical to Llama for our purposes.
    Qwen3 has q_norm/k_norm which the base LlamaAdapter already handles.
    """
    pass


# ============================================================================
# 4. ADAPTER REGISTRY & FACTORY
# ============================================================================

# Maps config.model_type → adapter class.
# If a model's config.model_type isn't in this dict, we fall back to LlamaAdapter
# which handles the vast majority of HF models.
ADAPTER_REGISTRY: dict[str, type[ModelAdapter]] = {
    # Llama family
    "llama": LlamaAdapter,
    "codellama": LlamaAdapter,
    # Mistral / Mixtral
    "mistral": MistralAdapter,
    "mixtral": MistralAdapter,
    # Qwen
    "qwen2": Qwen2Adapter,
    "qwen2_moe": Qwen2Adapter,
    "qwen3": Qwen2Adapter,
    "qwen3_moe": Qwen2Adapter,
    # Gemma
    "gemma": GemmaAdapter,
    "gemma2": GemmaAdapter,
    "gemma3": GemmaAdapter,
    # Phi
    "phi3": Phi3Adapter,
    "phi": Phi3Adapter,
    # Cohere
    "cohere": CohereAdapter,
    # Yi / DeepSeek / InternLM share the Llama pattern
    "yi": LlamaAdapter,
    "deepseek": LlamaAdapter,
    "deepseek_v2": LlamaAdapter,
    "internlm2": LlamaAdapter,
    # StarCoder2
    "starcoder2": LlamaAdapter,
}


def get_adapter(model, adapter_cls: Optional[type[ModelAdapter]] = None) -> ModelAdapter:
    """
    Create the appropriate ModelAdapter for a HuggingFace model.

    Args:
        model:       a loaded ``AutoModelForCausalLM`` instance.
        adapter_cls: optionally force a specific adapter class.

    Returns:
        A concrete ``ModelAdapter`` instance.
    """
    if adapter_cls is not None:
        return adapter_cls(model)

    model_type = getattr(model.config, "model_type", "").lower()

    cls = ADAPTER_REGISTRY.get(model_type)
    if cls is None:
        # Fall back to the most common pattern
        print(
            f"[model_adapter] No specific adapter for model_type='{model_type}'. "
            f"Falling back to LlamaAdapter (works for most HF models)."
        )
        cls = LlamaAdapter

    return cls(model)