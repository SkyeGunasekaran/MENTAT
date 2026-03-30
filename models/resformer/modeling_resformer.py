"""ResFormer Model - Implementation of Learnable ResFormer Plus.

This implements the Learnable ResFormer Plus architecture from the paper
"Value Residual Learning" (Zhou et al., 2024).

Key features:
- Value residual connections: V'_n = λ_{n,1} * V_1 + λ_{n,2} * V_n
- Learnable λ parameters with softmax-normalized λ_{n,1} values
- λ_scale parameter shared across all layers, initialized to N
"""

from __future__ import annotations

import math
import warnings
from typing import TYPE_CHECKING, Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers.modeling_outputs import BaseModelOutputWithPast, CausalLMOutputWithPast
from transformers.modeling_utils import PreTrainedModel
from transformers.utils import logging
from transformers.utils.deprecation import deprecate_kwarg

from models.resformer.attn import Attention
from models.resformer.configuration_resformer import ResFormerConfig
from fla.models.utils import Cache, FLAGenerationMixin
from fla.modules import FusedCrossEntropyLoss, FusedLinearCrossEntropyLoss, RMSNorm
from fla.modules import GatedMLP as TransformerMLP
from fla.modules.l2warp import l2_warp

if TYPE_CHECKING:
    from transformers.processing_utils import Unpack

try:
    from transformers.modeling_layers import GradientCheckpointingLayer
except ImportError:
    from fla.models.modeling_layers import GradientCheckpointingLayer

logger = logging.get_logger(__name__)


class ResFormerBlock(GradientCheckpointingLayer):
    """A single ResFormer transformer block with optional value residual."""

    def __init__(self, config: ResFormerConfig, layer_idx: int):
        super().__init__()

        self.config = config
        self.layer_idx = layer_idx

        self.attn_norm = (RMSNorm if config.fuse_norm else nn.RMSNorm)(config.hidden_size, eps=config.norm_eps)
        self.attn = Attention(
            hidden_size=config.hidden_size,
            num_heads=config.num_heads,
            num_kv_heads=config.num_kv_heads,
            qkv_bias=config.qkv_bias,
            qk_norm=config.qk_norm,
            window_size=config.window_size,
            rope_theta=config.rope_theta,
            max_position_embeddings=config.max_position_embeddings,
            layer_idx=layer_idx,
            # Value residual config
            use_value_residual=config.use_value_residual,
            num_hidden_layers=config.num_hidden_layers,
            value_residual_lambda2_init=config.value_residual_lambda2_init,
            value_residual_last_k=config.value_residual_last_k,
        )

        self.mlp_norm = (RMSNorm if config.fuse_norm else nn.RMSNorm)(config.hidden_size, eps=config.norm_eps)
        self.mlp = TransformerMLP(
            hidden_size=config.hidden_size,
            hidden_ratio=config.hidden_ratio,
            intermediate_size=config.intermediate_size,
            hidden_act=config.hidden_act,
            fuse_swiglu=config.fuse_swiglu,
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        past_key_values: tuple[torch.Tensor] | None = None,
        output_attentions: bool | None = False,
        use_cache: bool | None = False,
        v1: torch.Tensor | None = None,
        lambda1: torch.Tensor | None = None,
        **kwargs: Unpack[Any],
    ) -> tuple[torch.FloatTensor, ...]:
        """Forward pass for ResFormer block.
        
        Args:
            hidden_states: Input tensor [batch_size, seq_len, hidden_size]
            attention_mask: Optional attention mask
            past_key_values: Optional KV cache
            output_attentions: Whether to output attention weights
            use_cache: Whether to use caching
            v1: Value tensor from layer 0 for value residual
            lambda1: The λ_{n,1} coefficient for this layer
            
        Returns:
            Tuple containing:
            - hidden_states: Output tensor
            - attentions (optional): Attention weights if output_attentions=True
            - past_key_values (optional): Updated cache if use_cache=True  
            - v_base: V projection from this layer (only non-None for layer 0)
        """
        residual = hidden_states
        hidden_states = self.attn_norm(hidden_states)
        hidden_states, attentions, past_key_values, v_base = self.attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            use_cache=use_cache,
            output_attentions=output_attentions,
            v1=v1,
            lambda1=lambda1,
            **kwargs,
        )
        if self.config.fuse_norm:
            hidden_states, residual = self.mlp_norm(hidden_states, residual, True)
        else:
            hidden_states = residual + hidden_states
            residual = hidden_states
            hidden_states = self.mlp_norm(hidden_states)
        hidden_states = self.mlp(hidden_states, **kwargs)
        hidden_states = residual + hidden_states

        outputs = (hidden_states,)

        if output_attentions:
            outputs += (attentions,)

        if use_cache:
            outputs += (past_key_values,)
        
        # Always include v_base at the end
        outputs += (v_base,)

        return outputs


class ValueResidualLambdaManager(nn.Module):
    """Manages the shared λ parameters for Learnable ResFormer Plus.
    
    This implements the Learnable ResFormer Plus initialization strategy:
    - λ_scale is initialized to N (total layers) and is shared across all layers
    - λ_{n,1} = λ_scale * softmax(λ'_{n,1}) for n = 2, ..., N
    
    The λ'_{n,1} (lambda_logits) are learnable parameters that get softmax-normalized.
    """
    
    def __init__(self, num_hidden_layers: int):
        """Initialize the lambda manager.
        
        Args:
            num_hidden_layers: Total number of transformer layers (N).
        """
        super().__init__()
        self.num_hidden_layers = num_hidden_layers
        
        # λ_scale: shared scalar, initialized to N
        self.lambda_scale = nn.Parameter(torch.tensor(float(num_hidden_layers)))
        
        # λ'_{n,1} (lambda_logits) for layers 1 to N-1 (layer 0 produces V1, doesn't use it)
        # These are the pre-softmax logits, initialized to 0 for uniform distribution
        # Index i corresponds to layer i+1 (since layer 0 doesn't use value residual)
        self.lambda_logits = nn.Parameter(torch.zeros(num_hidden_layers - 1))
    
    def get_lambda1_values(self) -> torch.Tensor:
        """Compute the λ_{n,1} values for all layers that use value residual.
        
        Returns:
            Tensor of shape [num_hidden_layers - 1] containing λ_{n,1} values
            for layers 1, 2, ..., N-1 (0-indexed: layers that apply value residual)
        """
        # Apply softmax to get normalized weights
        softmax_weights = F.softmax(self.lambda_logits, dim=0)
        # Scale by λ_scale
        lambda1_values = self.lambda_scale * softmax_weights
        return lambda1_values
    
    def get_lambda1_for_layer(self, layer_idx: int) -> torch.Tensor:
        """Get λ_{n,1} for a specific layer.
        
        Args:
            layer_idx: The layer index (0-indexed). Layer 0 doesn't use value residual.
        
        Returns:
            Scalar tensor with λ_{n,1} value for this layer.
        """
        if layer_idx == 0:
            raise ValueError("Layer 0 produces V1 and doesn't use value residual")
        
        lambda1_values = self.get_lambda1_values()
        # layer_idx 1 corresponds to index 0 in lambda1_values
        return lambda1_values[layer_idx - 1]


class ResFormerPreTrainedModel(PreTrainedModel):
    """Base class for ResFormer pretrained models."""

    config_class = ResFormerConfig
    base_model_prefix = 'model'
    supports_gradient_checkpointing = True
    _no_split_modules = ['ResFormerBlock']
    _supports_cache_class = True

    def __init__(self, *inputs, **kwargs):
        super().__init__(*inputs, **kwargs)

    def _init_weights(
        self,
        module: nn.Module,
        rescale_prenorm_residual: bool = False,
        num_residuals_per_layer: int = 2,
    ):
        if isinstance(module, (nn.Linear, nn.Conv1d)):
            nn.init.normal_(module.weight, mean=0.0, std=self.config.initializer_range)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=self.config.initializer_range)
        elif hasattr(module, 'reset_parameters'):
            module.reset_parameters()

        if rescale_prenorm_residual:
            # Reinitialize selected weights subject to the OpenAI GPT-2 Paper Scheme
            p = None
            if hasattr(module, 'o_proj'):
                p = module.o_proj.weight
            elif hasattr(module, 'down_proj'):
                p = module.down_proj.weight
            if p is not None:
                nn.init.kaiming_uniform_(p, a=math.sqrt(5))
                with torch.no_grad():
                    p /= math.sqrt(num_residuals_per_layer * self.config.num_hidden_layers)


class ResFormerModel(ResFormerPreTrainedModel):
    """ResFormer model (Learnable ResFormer Plus) without language model head."""

    def __init__(
        self,
        config: ResFormerConfig,
    ) -> ResFormerModel:
        super().__init__(config)
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size

        self.embeddings = nn.Embedding(config.vocab_size, config.hidden_size, self.padding_idx)
        self.layers = nn.ModuleList([ResFormerBlock(config, layer_idx) for layer_idx in range(config.num_hidden_layers)])
        self.norm = (RMSNorm if config.fuse_norm else nn.RMSNorm)(config.hidden_size, eps=config.norm_eps)

        # Value residual lambda manager (shared across all layers)
        if config.use_value_residual:
            self.lambda_manager = ValueResidualLambdaManager(config.num_hidden_layers)
        else:
            self.lambda_manager = None

        self.gradient_checkpointing = False

        self.post_init()

    def get_input_embeddings(self):
        return self.embeddings

    def set_input_embeddings(self, value):
        self.embeddings = value

    def forward(
        self,
        input_ids: torch.LongTensor | None = None,
        attention_mask: torch.Tensor | None = None,
        past_key_values: list[torch.FloatTensor] | None = None,
        inputs_embeds: torch.FloatTensor | None = None,
        use_cache: bool | None = None,
        output_attentions: bool | None = None,
        output_hidden_states: bool | None = None,
        return_dict: bool | None = None,
        **kwargs: Unpack[Any],
    ) -> tuple | CausalLMOutputWithPast:
        if output_attentions:
            warnings.warn(
                "`ResFormerModel` does not support output attention weights now, "
                "so `output_attentions` is set to `False`.",
            )
            output_attentions = False
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        use_cache = use_cache if use_cache is not None else (self.config.use_cache if not self.training else False)
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # Validate inputs
        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both input_ids and inputs_embeds at the same time")
        elif input_ids is None and inputs_embeds is None:
            raise ValueError("You have to specify either input_ids or inputs_embeds")

        if use_cache and not isinstance(past_key_values, Cache):
            past_key_values = Cache.from_legacy_cache(past_key_values)

        if inputs_embeds is None:
            inputs_embeds = self.embeddings(input_ids)

        hidden_states = inputs_embeds

        all_hidden_states = () if output_hidden_states else None
        all_attns = () if output_attentions else None
        next_cache = None

        # Pre-compute all λ_{n,1} values for efficiency
        if self.lambda_manager is not None:
            lambda1_values = self.lambda_manager.get_lambda1_values()
        else:
            lambda1_values = None

        v1 = None  # Will hold V from layer 0 for value residual
        
        for layer_idx, layer in enumerate(self.layers):
            if output_hidden_states:
                all_hidden_states += (hidden_states,)

            # Get λ_{n,1} for this layer (if using value residual and not layer 0)
            if lambda1_values is not None and layer_idx > 0:
                lambda1 = lambda1_values[layer_idx - 1]
            else:
                lambda1 = None

            layer_outputs = layer(
                hidden_states,
                attention_mask=attention_mask,
                past_key_values=past_key_values,
                output_attentions=output_attentions,
                use_cache=use_cache,
                v1=v1,
                lambda1=lambda1,
                **kwargs,
            )

            hidden_states = layer_outputs[0]
            
            # v_base is always the last element of outputs
            v_base = layer_outputs[-1]
            
            # Capture V from layer 0 to pass to later layers
            if v_base is not None:
                v1 = v_base
            
            # Extract cache and attentions from the middle of outputs
            idx = 1
            if output_attentions:
                all_attns += (layer_outputs[idx],)
                idx += 1
            if use_cache:
                next_cache = layer_outputs[idx]

        hidden_states = self.norm(hidden_states)

        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        if not return_dict:
            return tuple(v for v in [hidden_states, next_cache, all_hidden_states, all_attns] if v is not None)

        return BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=next_cache,
            hidden_states=all_hidden_states,
            attentions=all_attns,
        )


class ResFormerForCausalLM(ResFormerPreTrainedModel, FLAGenerationMixin):
    """ResFormer model with causal language modeling head."""

    _tied_weights_keys = ["lm_head.weight"]

    def __init__(self, config):
        super().__init__(config)
        self.model = ResFormerModel(config)
        self.vocab_size = config.vocab_size
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.criterion = None

        self.post_init()

    def get_input_embeddings(self):
        return self.model.embeddings

    def set_input_embeddings(self, value):
        self.model.embeddings = value

    def get_output_embeddings(self):
        return self.lm_head

    def set_output_embeddings(self, new_embeddings):
        self.lm_head = new_embeddings

    def set_decoder(self, decoder):
        self.model = decoder

    def get_decoder(self):
        return self.model

    @deprecate_kwarg("num_logits_to_keep", version="4.50", new_name="logits_to_keep")
    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: torch.Tensor | None = None,
        past_key_values: Cache | list[torch.FloatTensor] | None = None,
        inputs_embeds: torch.FloatTensor | None = None,
        labels: torch.LongTensor | None = None,
        use_cache: bool | None = None,
        output_attentions: bool | None = None,
        output_hidden_states: bool | None = None,
        return_dict: bool | None = None,
        logits_to_keep: int | None = 0,
        **kwargs: Unpack[Any],
    ) -> tuple | CausalLMOutputWithPast:
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            **kwargs,
        )

        hidden_states = outputs[0]

        logits = None if self.config.fuse_linear_cross_entropy else self.lm_head(hidden_states[:, -logits_to_keep:])

        loss = None
        if labels is not None:
            if getattr(self, 'criterion', None) is None:
                if self.config.fuse_linear_cross_entropy:
                    criterion = FusedLinearCrossEntropyLoss(use_l2warp=self.config.use_l2warp)
                elif self.config.fuse_cross_entropy:
                    criterion = FusedCrossEntropyLoss(inplace_backward=True)
                else:
                    criterion = nn.CrossEntropyLoss()
            else:
                criterion = self.criterion
            labels = labels.to(hidden_states.device)
            labels = torch.cat((labels[..., 1:], torch.full_like(labels[:, :1], criterion.ignore_index)), 1)
            if self.config.fuse_linear_cross_entropy:
                loss = criterion(hidden_states, labels, self.lm_head.weight, self.lm_head.bias)
            else:
                loss = criterion(logits.view(labels.numel(), -1), labels.view(-1))
                loss = l2_warp(loss, logits) if self.config.use_l2warp else loss

        if not return_dict:
            output = (logits,) + outputs[1:]
            return (loss,) + output if loss is not None else output

        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )