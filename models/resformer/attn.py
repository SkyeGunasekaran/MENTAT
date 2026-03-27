"""ResFormer Attention Module - Implementation of Learnable ResFormer Plus.

This implements the value residual mechanism from the paper "Value Residual Learning"
(Zhou et al., 2024). The key formula is:

    V'_n = λ_{n,1} * V_1 + λ_{n,2} * V_n

For Learnable ResFormer Plus:
- λ_{n,2} is initialized to 0.5 for all layers
- λ_{n,1} is initialized to: λ_scale * softmax(λ'_{n,1})
  where λ_scale is initialized to N (total layers) and shared across all layers
"""

from __future__ import annotations

import warnings
from typing import TYPE_CHECKING

import torch
import torch.nn as nn
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


class ValueResidualMixing(nn.Module):
    """Learnable ResFormer Plus value residual mixing module.
    
    This module implements the value residual mixing formula:
        V'_n = λ_{n,1} * V_1 + λ_{n,2} * V_n
    
    For Learnable ResFormer Plus initialization:
    - λ_{n,2} is initialized to 0.5
    - λ_{n,1} = λ_scale * softmax(λ'_{n,1}) across all layers
    - λ_scale is initialized to N (total layers)
    
    The λ_scale and λ' (lambda_logits) parameters are shared across all layers
    and managed by the parent model, while each layer has its own λ_{n,2}.
    """
    
    def __init__(
        self,
        layer_idx: int,
        num_hidden_layers: int,
        lambda2_init: float = 0.5,
        head_dim: int | None = None,
    ):
        """Initialize the value residual mixing module.
        
        Args:
            layer_idx: Index of this layer (0-indexed).
            num_hidden_layers: Total number of layers in the model.
            lambda2_init: Initial value for λ_{n,2}.
            head_dim: Dimension of each attention head (for V1 normalization).
        """
        super().__init__()
        self.layer_idx = layer_idx
        self.num_hidden_layers = num_hidden_layers
        
        # λ_{n,2} is a learnable scalar for this layer, initialized to lambda2_init
        self.lambda2 = nn.Parameter(torch.tensor(lambda2_init))
            
    def forward(
        self,
        v: torch.Tensor,
        v1: torch.Tensor,
        lambda1: torch.Tensor,
    ) -> torch.Tensor:
        """Apply value residual mixing.
        
        Args:
            v: Current layer's value tensor [B, T, H, D]
            v1: First layer's value tensor [B, T, H, D]
            lambda1: The λ_{n,1} coefficient for this layer (scalar tensor)
        
        Returns:
            Mixed value tensor V' = λ_{n,1} * V_1 + λ_{n,2} * V_n
        """
        # Ensure v1 is on same device/dtype
        v1_cast = v1.to(dtype=v.dtype, device=v.device)
                
        # V'_n = λ_{n,1} * V_1 + λ_{n,2} * V_n
        v_mixed = lambda1 * v1_cast + self.lambda2 * v
        
        return v_mixed


class Attention(nn.Module):
    """Multi-head attention with optional Learnable ResFormer Plus value residual.
    
    This implements flash attention with rotary position embeddings and optional
    value residual connections as described in the ResFormer paper.
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
        value_residual_lambda2_init: float = 0.5,
        value_residual_last_k: int | None = None,
    ):
        """Initialize the attention module.
        
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
            use_value_residual: Whether to use value residual connections.
            num_hidden_layers: Total number of layers (required for value residual).
            value_residual_lambda2_init: Initial value for λ_{n,2}.
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
            raise ImportError("Please install Flash Attention via `pip install flash-attn --no-build-isolation` first")

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
        self.value_residual_last_k = value_residual_last_k

        # Determine if this layer should apply value residual
        self._should_apply_value_residual = self._check_should_apply_value_residual()
        self._should_produce_v1 = (use_value_residual and layer_idx == 0)
        
        # Create value residual mixing module if needed
        if self._should_apply_value_residual:
            self.value_residual_mixing = ValueResidualMixing(
                layer_idx=layer_idx,
                num_hidden_layers=num_hidden_layers,
                lambda2_init=value_residual_lambda2_init,
                head_dim=self.head_dim,
            )
        else:
            self.value_residual_mixing = None

    def _check_should_apply_value_residual(self) -> bool:
        """Check if this layer should apply value residual mixing."""
        if not self.use_value_residual:
            return False
        if self.layer_idx == 0:
            # First layer produces V1, doesn't consume it
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
        lambda1: torch.Tensor | None = None,
        **kwargs,
    ) -> tuple[torch.Tensor, torch.Tensor | None, Cache | None, torch.Tensor | None]:
        """Forward pass for attention with optional value residual.
        
        Args:
            hidden_states: Input tensor [batch_size, seq_len, hidden_size]
            attention_mask: Optional mask [batch_size, seq_len]
            past_key_values: Optional KV cache
            output_attentions: Whether to output attention weights
            use_cache: Whether to update cache
            v1: Value tensor from layer 0 [batch_size, seq_len, num_kv_heads, head_dim]
            lambda1: The λ_{n,1} coefficient for this layer (from model's softmax)
        
        Returns:
            output: Attention output [batch_size, seq_len, hidden_size]
            attentions: None (not implemented)
            past_key_values: Updated cache
            v_base: Raw V projection from this layer (only for layer 0 with value residual)
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
        
        # Store the original V projection from layer 0
        v_base = v.clone() if self._should_produce_v1 else None

        # Apply value residual: V'_n = λ_{n,1} * V_1 + λ_{n,2} * V_n
        if self._should_apply_value_residual and v1 is not None and lambda1 is not None:
            v = self.value_residual_mixing(v, v1, lambda1)

        if self.qk_norm:
            q, k = self.q_norm(q), self.k_norm(k)

        # Handle cu_seqlens for variable length sequences
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