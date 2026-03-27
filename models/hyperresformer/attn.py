"""HyperResFormer Attention Module - Implementation of HyperResFormer.

This implements a token-dependent value residual mechanism inspired by Hyper-Connections.
The key formula is:

    V'_n = V_n + alpha_n(h) * V_1

where alpha_n(h) is a token-wise mixing coefficient computed by a lightweight linear
projection from the (post-norm) hidden states, followed by a configurable gating function:

    alpha_n(h) = gate(W_alpha @ h)   shape: [B, T, num_kv_heads, 1]

The gating function is selected via the ``value_residual_gate`` config parameter.
Supported modes (see ``HyperValueResidualMixing`` for details):

    relu            — ReLU(Wh)                            non-convex, sparse, default
    sigmoid         — σ(Wh)                               convex, bounded [0, 1]
    softmax         — softmax(Wh) * num_kv_heads          convex combination across heads
    softmax_sigmoid — softmax(Wh) * σ(Wh) * num_kv_heads convex + per-head gate
    tanh            — tanh(Wh)                            allows subtractive mixing
    identity        — Wh (no activation)                  linear baseline

This allows each token to dynamically control how much of the first layer's value
representation is mixed in, rather than using a single global scalar per layer.

Compared to Learnable ResFormer Plus:
- No shared lambda_scale parameter across layers
- No softmax-normalised lambda logits
- Per-token, per-head alpha computed from the hidden state at each layer
- Gating function is configurable for ablation studies
"""

from __future__ import annotations

import warnings
from typing import TYPE_CHECKING

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from transformers.utils import logging

from fla.layers.utils import pad_input, unpad_input
from fla.modules import RMSNorm, RotaryEmbedding
from fla.ops.utils.index import prepare_lens_from_mask

if TYPE_CHECKING:
    from fla.models.utils import Cache

try:
    from flash_attn import flash_attn_func, flash_attn_varlen_func
except ImportError:
    warnings.warn(
        "Flash Attention is not installed. Please install it via `pip install flash-attn --no-build-isolation`",
        category=ImportWarning,
    )
    flash_attn_func = None

logger = logging.get_logger(__name__)


class HyperValueResidualMixing(nn.Module):
    """Token-dependent value residual mixing module for HyperResFormer.

    This module replaces the global learned scalar lambda_{n,1} from Learnable
    ResFormer Plus with a lightweight linear projection from hidden states,
    producing a per-token, per-head mixing coefficient alpha:

        alpha_n(h) = gate(W_alpha @ h)   shape: [B, T, num_kv_heads, 1]
        V'_n = V_n + alpha_n(h) * V_1

    The gating function is selected at construction time via the ``gate`` argument
    and held fixed for the lifetime of the module, enabling clean ablation studies.

    Supported gates
    ---------------
    relu
        ``ReLU(Wh)`` — the original baseline. Non-convex; alpha is sparse (can be
        exactly 0) and unbounded above. Allows aggressive amplification of V_1.
    sigmoid
        ``σ(Wh)`` — convex, bounded in [0, 1]. Each head independently gates V_1
        residual with a smooth probability-like weight.
    softmax
        ``softmax(Wh) * num_kv_heads`` — softmax is computed *across the head
        dimension*, making alpha a convex combination (heads compete for a shared
        V_1 budget). Scaled by num_kv_heads so the expected per-head value is 1
        at uniform distribution, matching the scale of other gates.
    softmax_sigmoid
        ``softmax(Wh) * σ(Wh) * num_kv_heads`` — combines across-head competition
        (softmax) with a per-head on/off gate (sigmoid). Strictly more expressive
        than either alone; the product is still convex in the logits.
    tanh
        ``tanh(Wh)`` — bounded in [−1, 1]. Allows *subtractive* mixing where
        alpha < 0 suppresses features already present in V_n.
    identity
        ``Wh`` — no activation; purely linear. Useful as a lower bound to confirm
        that nonlinearity is necessary.

    The projection W_alpha maps hidden_size -> num_kv_heads, so it is lightweight
    (hidden_size * num_kv_heads parameters per layer) and adds negligible overhead.
    """

    VALID_GATES = frozenset({'relu', 'sigmoid', 'softmax', 'softmax_sigmoid', 'tanh', 'identity'})

    def __init__(
        self,
        hidden_size: int,
        num_kv_heads: int,
        proj_bias: bool = False,
        gate: str = 'relu',
    ):
        """Initialise the hyper value residual mixing module.

        Args:
            hidden_size: Dimension of the hidden states (input to the projection).
            num_kv_heads: Number of key-value heads; one alpha scalar is produced per head.
            proj_bias: Whether to add a bias term to the alpha projection.
            gate: Gating function to apply to the projection output. Must be one of
                ``'relu'``, ``'sigmoid'``, ``'softmax'``, ``'softmax_sigmoid'``,
                ``'tanh'``, or ``'identity'``.

        Raises:
            ValueError: If ``gate`` is not a recognised gating function.
        """
        super().__init__()
        if gate not in self.VALID_GATES:
            raise ValueError(
                f"gate must be one of {sorted(self.VALID_GATES)}, got '{gate}'."
            )
        self.hidden_size = hidden_size
        self.num_kv_heads = num_kv_heads
        self.gate = gate

        # Lightweight projection: hidden_size -> num_kv_heads.
        # Produces one mixing scalar per kv-head per token.
        self.alpha_proj = nn.Linear(hidden_size, num_kv_heads, bias=proj_bias)

    def _apply_gate(self, logits: torch.Tensor) -> torch.Tensor:
        """Apply the configured gating function to raw projection logits.

        Args:
            logits: Raw output of alpha_proj, shape [B, T, num_kv_heads].

        Returns:
            Gated alpha values with the same shape.
        """
        if self.gate == 'relu':
            return F.relu(logits)
        elif self.gate == 'sigmoid':
            return torch.sigmoid(logits)
        elif self.gate == 'softmax':
            # softmax over the head dimension; scale so mean alpha ≈ 1
            return F.softmax(logits, dim=-1) * self.num_kv_heads
        elif self.gate == 'softmax_sigmoid':
            # convex combination (heads) × per-head gate; scale as above
            return F.softmax(logits, dim=-1) * torch.sigmoid(logits) * self.num_kv_heads
        elif self.gate == 'tanh':
            return torch.tanh(logits)
        elif self.gate == 'identity':
            return logits
        else:
            # Should never reach here due to __init__ validation, but be explicit.
            raise RuntimeError(f"Unknown gate '{self.gate}'")  # pragma: no cover

    def forward(
        self,
        v: torch.Tensor,
        v1: torch.Tensor,
        hidden_states: torch.Tensor,
    ) -> torch.Tensor:
        """Apply token-dependent value residual mixing.

        Args:
            v: Current layer's value tensor  [B, T, num_kv_heads, head_dim].
            v1: First layer's value tensor   [B, T, num_kv_heads, head_dim].
            hidden_states: Post-norm hidden states [B, T, hidden_size] used to
                compute the per-token mixing coefficient alpha.

        Returns:
            Mixed value tensor V'_n = V_n + alpha(h) * V_1,
            shape [B, T, num_kv_heads, head_dim].
        """
        # Project hidden states to per-head alpha scalars: [B, T, num_kv_heads]
        logits = self.alpha_proj(hidden_states.to(dtype=v.dtype))

        # Apply the selected gate, then unsqueeze for head_dim broadcast:
        # [B, T, num_kv_heads] -> [B, T, num_kv_heads, 1]
        alpha = self._apply_gate(logits).unsqueeze(-1)

        # Ensure v1 is on the same device/dtype as v.
        v1_cast = v1.to(dtype=v.dtype, device=v.device)

        # V'_n = V_n + alpha(h) * V_1
        return v + alpha * v1_cast


class Attention(nn.Module):
    """Multi-head attention with optional HyperResFormer token-dependent value residual.

    This implements flash attention with rotary position embeddings and optional
    hyper value residual connections where mixing coefficients are computed
    dynamically from hidden states rather than from global learned scalars.
    """

    def __init__(
        self,
        hidden_size: int = 2048,
        num_heads: int = 32,
        num_kv_heads: int | None = None,
        qkv_bias: bool = False,
        qk_norm: bool = False,
        window_size: int | None = None,
        rope_theta: float | None = 10000.,
        max_position_embeddings: int | None = None,
        layer_idx: int = None,
        # Value residual hyperparameters
        use_value_residual: bool = False,
        num_hidden_layers: int | None = None,
        value_residual_gate: str = 'relu',
        value_residual_proj_bias: bool = False,
        value_residual_last_k: int | None = None,
    ):
        """Initialise the attention module.

        Args:
            hidden_size: Total hidden dimension.
            num_heads: Number of attention heads.
            num_kv_heads: Number of key-value heads (for GQA).
            qkv_bias: Whether to use bias in QKV projections.
            qk_norm: Whether to apply RMSNorm to Q and K.
            window_size: Sliding window size for attention.
            rope_theta: Base for rotary position embeddings.
            max_position_embeddings: Maximum sequence length.
            layer_idx: Index of this layer in the model.
            use_value_residual: Whether to use hyper value residual connections.
            num_hidden_layers: Total number of layers (used for last-k determination).
            value_residual_gate: Gating function for the alpha projection. One of
                ``'relu'``, ``'sigmoid'``, ``'softmax'``, ``'softmax_sigmoid'``,
                ``'tanh'``, or ``'identity'``. See ``HyperValueResidualMixing`` for details.
            value_residual_proj_bias: Whether to add bias to the alpha projection.
            value_residual_last_k: Apply value residual to last k layers only.
                None = apply to all layers after layer 0.
        """
        super().__init__()

        self.hidden_size = hidden_size
        self.num_heads = num_heads
        if num_kv_heads is None:
            self.num_kv_heads = self.num_heads
        else:
            self.num_kv_heads = num_kv_heads
        self.num_kv_groups = num_heads // self.num_kv_heads
        self.head_dim = self.hidden_size // self.num_heads
        self.kv_dim = self.num_kv_heads * self.head_dim
        self.qkv_bias = qkv_bias
        self.qk_norm = qk_norm

        self.window_size = window_size
        self.rope_theta = rope_theta
        self.max_position_embeddings = max_position_embeddings
        self.layer_idx = layer_idx

        if flash_attn_func is None:
            raise ImportError(
                "Please install Flash Attention via `pip install flash-attn --no-build-isolation` first"
            )

        self.q_proj = nn.Linear(self.hidden_size, self.hidden_size, bias=self.qkv_bias)
        self.k_proj = nn.Linear(self.hidden_size, self.kv_dim, bias=self.qkv_bias)
        self.v_proj = nn.Linear(self.hidden_size, self.kv_dim, bias=self.qkv_bias)
        self.o_proj = nn.Linear(self.hidden_size, self.hidden_size, bias=False)

        if qk_norm:
            self.q_norm = RMSNorm(self.head_dim, dtype=torch.float32)
            self.k_norm = RMSNorm(self.head_dim, dtype=torch.float32)

        self.rotary = RotaryEmbedding(dim=self.head_dim, base=self.rope_theta)

        # Value residual configuration
        self.use_value_residual = use_value_residual
        self.num_hidden_layers = num_hidden_layers
        self.value_residual_gate = value_residual_gate
        self.value_residual_last_k = value_residual_last_k

        # Determine whether this layer should apply / produce value residual
        self._should_apply_value_residual = self._check_should_apply_value_residual()
        self._should_produce_v1 = (use_value_residual and layer_idx == 0)

        # Create the hyper value residual mixing module if needed
        if self._should_apply_value_residual:
            self.value_residual_mixing = HyperValueResidualMixing(
                hidden_size=hidden_size,
                num_kv_heads=self.num_kv_heads,
                proj_bias=value_residual_proj_bias,
                gate=value_residual_gate,
            )
        else:
            self.value_residual_mixing = None

    def _check_should_apply_value_residual(self) -> bool:
        """Check if this layer should apply value residual mixing."""
        if not self.use_value_residual:
            return False
        if self.layer_idx == 0:
            # First layer produces V1; it does not consume it
            return False
        if self.value_residual_last_k is None:
            # Apply to all layers after the first
            return True
        if self.num_hidden_layers is None:
            warnings.warn(
                "value_residual_last_k is set but num_hidden_layers is None. "
                "Cannot determine which layers to apply value residual to."
            )
            return False
        # Apply to last k layers only
        return self.layer_idx >= (self.num_hidden_layers - self.value_residual_last_k)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.LongTensor | None = None,
        past_key_values: Cache | None = None,
        output_attentions: bool = False,
        use_cache: bool = False,
        v1: torch.Tensor | None = None,
        **kwargs,
    ) -> tuple[torch.Tensor, torch.Tensor | None, Cache | None, torch.Tensor | None]:
        """Forward pass for attention with optional hyper value residual.

        Args:
            hidden_states: Post-norm input tensor [batch_size, seq_len, hidden_size].
                These are the same features used for Q/K/V projections and are also
                passed to HyperValueResidualMixing to compute the per-token alpha.
            attention_mask: Optional mask [batch_size, seq_len].
            past_key_values: Optional KV cache.
            output_attentions: Whether to output attention weights.
            use_cache: Whether to update cache.
            v1: Value tensor from layer 0 [batch_size, seq_len, num_kv_heads, head_dim].

        Returns:
            output: Attention output [batch_size, seq_len, hidden_size].
            attentions: None (not implemented).
            past_key_values: Updated cache.
            v_base: Raw V projection from this layer (only non-None for layer 0).
        """
        if attention_mask is not None:
            assert len(attention_mask.shape) == 2, (
                "Expected attention_mask as a 0-1 matrix with shape [batch_size, seq_len] "
                "for padding purposes (0 indicating padding). "
                "Arbitrary attention masks of shape [batch_size, seq_len, seq_len] are not allowed."
            )

        batch_size, q_len, _ = hidden_states.size()

        q = rearrange(self.q_proj(hidden_states), '... (h d) -> ... h d', d=self.head_dim)
        k = rearrange(self.k_proj(hidden_states), '... (h d) -> ... h d', d=self.head_dim)
        v = rearrange(self.v_proj(hidden_states), '... (h d) -> ... h d', d=self.head_dim)

        # Store the original V projection from layer 0 for downstream layers
        v_base = v.clone() if self._should_produce_v1 else None

        # Apply hyper value residual: V'_n = V_n + alpha(hidden_states) * V_1.
        # hidden_states is the post-norm input — the same content-rich features used
        # for Q/K/V, making the mixing decision fully token-aware at negligible cost.
        if self._should_apply_value_residual and v1 is not None:
            v = self.value_residual_mixing(v, v1, hidden_states)

        if self.qk_norm:
            q, k = self.q_norm(q), self.k_norm(k)

        # Handle cu_seqlens for variable-length sequences
        cu_seqlens = kwargs.get('cu_seqlens')

        seqlen_offset, max_seqlen = 0, q_len
        if past_key_values is not None:
            seqlen_offset = past_key_values.get_seq_length(self.layer_idx)
            max_seqlen = q.shape[1] + seqlen_offset

            if attention_mask is not None:
                seqlen_offset = seqlen_offset + prepare_lens_from_mask(attention_mask) - attention_mask.shape[-1]
                max_seqlen = q.shape[1] + max(seqlen_offset)

        if self.max_position_embeddings is not None:
            max_seqlen = max(max_seqlen, self.max_position_embeddings)
        q, k = self.rotary(q, k, seqlen_offset=seqlen_offset, max_seqlen=max_seqlen, cu_seqlens=cu_seqlens)

        if past_key_values is not None:
            cache_has_content = past_key_values.get_seq_length(self.layer_idx) > 0
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

        # Flash attention computation
        if attention_mask is not None:
            if q.shape[1] == 1 and self.window_size is not None:
                attention_mask = attention_mask[:, -self.window_size:]
            q, (k, v), indices_q, cu_seqlens, max_seq_lens = unpad_input(q, (k, v), attention_mask, q_len)
            cu_seqlens_q, cu_seqlens_k = cu_seqlens
            max_seqlen_q, max_seqlen_k = max_seq_lens
            o = flash_attn_varlen_func(
                q, k, v,
                cu_seqlens_q=cu_seqlens_q,
                cu_seqlens_k=cu_seqlens_k,
                max_seqlen_q=max_seqlen_q,
                max_seqlen_k=max_seqlen_k,
                causal=True,
                window_size=(-1, -1) if self.window_size is None else (self.window_size-1, 0),
            )
            o = pad_input(o, indices_q, batch_size, q_len)
        elif cu_seqlens is not None:
            o = flash_attn_varlen_func(
                q.squeeze(0), k.squeeze(0), v.squeeze(0),
                cu_seqlens_q=cu_seqlens,
                cu_seqlens_k=cu_seqlens,
                max_seqlen_q=max_seqlen,
                max_seqlen_k=max_seqlen,
                causal=True,
                window_size=(-1, -1) if self.window_size is None else (self.window_size-1, 0),
            ).unsqueeze(0)
        else:
            o = flash_attn_func(
                q, k, v,
                causal=True,
                window_size=(-1, -1) if self.window_size is None else (self.window_size-1, 0),
            )
        o = o.reshape(batch_size, q_len, -1)
        o = self.o_proj(o)

        if not output_attentions:
            attentions = None

        return o, attentions, past_key_values, v_base