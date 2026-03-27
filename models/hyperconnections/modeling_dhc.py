# Copyright (c) 2023-2025, Songlin Yang, Yu Zhang
# Transformer with Dynamic Hyper-Connections (DHC)
# Implementation based on "Hyper-Connections" (ICLR 2025)

from __future__ import annotations

import math
import warnings
from typing import TYPE_CHECKING, Any, Optional, Tuple

import torch
import torch.nn as nn
from transformers.modeling_outputs import BaseModelOutputWithPast, CausalLMOutputWithPast
from transformers.modeling_utils import PreTrainedModel
from transformers.utils import logging
from transformers.utils.deprecation import deprecate_kwarg

from fla.layers.attn import Attention
from fla.models.utils import Cache, FLAGenerationMixin
from fla.modules import FusedCrossEntropyLoss, FusedLinearCrossEntropyLoss, RMSNorm
from fla.modules import GatedMLP as TransformerMLP
from fla.modules.l2warp import l2_warp

from models.hyperconnections.configuration_dhc import DHCConfig
from models.hyperconnections.hyper_connections import (
    HyperConnection,
    expand_to_hyper_hidden,
    collapse_hyper_hidden,
)

if TYPE_CHECKING:
    from transformers.processing_utils import Unpack


try:
    from transformers.modeling_layers import GradientCheckpointingLayer
except ImportError:
    from fla.models.modeling_layers import GradientCheckpointingLayer

logger = logging.get_logger(__name__)


class DHCBlock(GradientCheckpointingLayer):
    """
    Transformer block with Hyper-Connections.
    
    This block replaces standard residual connections with hyper-connections,
    allowing the network to learn optimal connection strengths between features
    at different depths.
    """

    def __init__(self, config: DHCConfig, layer_idx: int):
        super().__init__()

        self.config = config
        self.layer_idx = layer_idx
        
        # Layer index for hyper-connections (each block has 2 HC: one for attn, one for FFN)
        # Following the paper's initialization scheme
        attn_hc_idx = layer_idx * 2
        ffn_hc_idx = layer_idx * 2 + 1

        # Attention normalization and module
        self.attn_norm = (RMSNorm if config.fuse_norm else nn.RMSNorm)(
            config.hidden_size, eps=config.norm_eps
        )
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
        )
        
        # Hyper-connection for attention
        self.attn_hc = HyperConnection(
            hidden_size=config.hidden_size,
            expansion_rate=config.expansion_rate,
            layer_idx=attn_hc_idx,
            dynamic=config.use_dynamic_hc,
            use_tanh=config.use_tanh,
            norm_type=config.hc_norm_type,
        )

        # MLP normalization and module
        self.mlp_norm = (RMSNorm if config.fuse_norm else nn.RMSNorm)(
            config.hidden_size, eps=config.norm_eps
        )
        self.mlp = TransformerMLP(
            hidden_size=config.hidden_size,
            hidden_ratio=config.hidden_ratio,
            intermediate_size=config.intermediate_size,
            hidden_act=config.hidden_act,
            fuse_swiglu=config.fuse_swiglu,
        )
        
        # Hyper-connection for MLP
        self.mlp_hc = HyperConnection(
            hidden_size=config.hidden_size,
            expansion_rate=config.expansion_rate,
            layer_idx=ffn_hc_idx,
            dynamic=config.use_dynamic_hc,
            use_tanh=config.use_tanh,
            norm_type=config.hc_norm_type,
        )

    def forward(
        self,
        hyper_hidden: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        past_key_values: tuple[torch.Tensor] | None = None,
        output_attentions: bool | None = False,
        use_cache: bool | None = False,
        **kwargs: Unpack[Any],
    ) -> tuple[torch.FloatTensor, ...]:
        """
        Forward pass with hyper-connections.
        
        Args:
            hyper_hidden: Hyper hidden matrix of shape (batch, seq_len, n, hidden_size)
            attention_mask: Attention mask
            past_key_values: Cached key-value pairs
            output_attentions: Whether to output attention weights
            use_cache: Whether to use KV cache
            
        Returns:
            Tuple containing:
                - Updated hyper_hidden
                - Attention weights (if output_attentions)
                - Updated past_key_values (if use_cache)
        """
        # ============ Attention Block with Hyper-Connection ============
        # Width connection for attention
        attn_mix_h, attn_beta = self.attn_hc.width_connection(hyper_hidden)
        
        # Get layer input (h_0 from the mixed states)
        attn_input = attn_mix_h[:, :, 0, :]  # (B, L, D)
        
        # Apply normalization
        attn_input_normed = self.attn_norm(attn_input)
        
        # Apply attention
        attn_output, attentions, past_key_values = self.attn(
            hidden_states=attn_input_normed,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            use_cache=use_cache,
            output_attentions=output_attentions,
            **kwargs,
        )
        
        # Depth connection for attention
        hyper_hidden = self.attn_hc.depth_connection(attn_mix_h, attn_output, attn_beta)
        
        # ============ MLP Block with Hyper-Connection ============
        # Width connection for MLP
        mlp_mix_h, mlp_beta = self.mlp_hc.width_connection(hyper_hidden)
        
        # Get layer input
        mlp_input = mlp_mix_h[:, :, 0, :]  # (B, L, D)
        
        # Apply normalization
        mlp_input_normed = self.mlp_norm(mlp_input)
        
        # Apply MLP
        mlp_output = self.mlp(mlp_input_normed, **kwargs)
        
        # Depth connection for MLP
        hyper_hidden = self.mlp_hc.depth_connection(mlp_mix_h, mlp_output, mlp_beta)

        # Prepare outputs
        outputs = (hyper_hidden,)

        if output_attentions:
            outputs += (attentions,)

        if use_cache:
            outputs += (past_key_values,)

        return outputs


class DHCPreTrainedModel(PreTrainedModel):
    """Base class for DHC models with weight initialization."""

    config_class = DHCConfig
    base_model_prefix = 'model'
    supports_gradient_checkpointing = True
    _no_split_modules = ['DHCBlock']
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
            # Scale output weights by 1/sqrt(n) where n is expansion_rate
            # This ensures the std of output remains consistent after summing hyper hidden vectors
            p = None
            if hasattr(module, 'o_proj'):
                p = module.o_proj.weight
            elif hasattr(module, 'down_proj'):
                p = module.down_proj.weight
            if p is not None:
                nn.init.kaiming_uniform_(p, a=math.sqrt(5))
                with torch.no_grad():
                    # Scale by sqrt(n) as mentioned in the paper's implementation section
                    p /= math.sqrt(num_residuals_per_layer * self.config.num_hidden_layers)
                    # Additional scaling for hyper-connections
                    p *= math.sqrt(self.config.expansion_rate)


class DHCModel(DHCPreTrainedModel):
    """
    DHC Transformer Model (decoder-only) with Hyper-Connections.
    
    This model uses hyper-connections instead of residual connections,
    allowing dynamic adjustment of connection strengths between layers.
    """

    def __init__(
        self,
        config: DHCConfig,
    ) -> 'DHCModel':
        super().__init__(config)
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size
        self.expansion_rate = config.expansion_rate

        self.embeddings = nn.Embedding(config.vocab_size, config.hidden_size, self.padding_idx)
        self.layers = nn.ModuleList([
            DHCBlock(config, layer_idx) 
            for layer_idx in range(config.num_hidden_layers)
        ])
        self.norm = (RMSNorm if config.fuse_norm else nn.RMSNorm)(
            config.hidden_size, eps=config.norm_eps
        )

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
    ) -> tuple | BaseModelOutputWithPast:
        if output_attentions:
            warnings.warn(
                "`DHCModel` does not support output attention weights now, "
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

        # Get embeddings
        if inputs_embeds is None:
            inputs_embeds = self.embeddings(input_ids)

        # Expand to hyper hidden matrix: (B, L, D) -> (B, L, N, D)
        hyper_hidden = expand_to_hyper_hidden(inputs_embeds, self.expansion_rate)

        all_hidden_states = () if output_hidden_states else None
        all_attns = () if output_attentions else None
        next_cache = None

        for layer in self.layers:
            if output_hidden_states:
                # Collapse hyper hidden for output
                all_hidden_states += (collapse_hyper_hidden(hyper_hidden),)

            layer_outputs = layer(
                hyper_hidden,
                attention_mask=attention_mask,
                past_key_values=past_key_values,
                output_attentions=output_attentions,
                use_cache=use_cache,
                **kwargs,
            )

            hyper_hidden = layer_outputs[0]

            if use_cache:
                next_cache = layer_outputs[2 if output_attentions else 1]

            if output_attentions:
                all_attns += (layer_outputs[1],)

        # Collapse hyper hidden matrix to single hidden state by summing
        hidden_states = collapse_hyper_hidden(hyper_hidden)
        
        # Apply final normalization
        hidden_states = self.norm(hidden_states)

        # Add final hidden states
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


class DHCForCausalLM(DHCPreTrainedModel, FLAGenerationMixin):
    """
    DHC Transformer Model with a language modeling head for causal LM.
    """

    _tied_weights_keys = ["lm_head.weight"]

    def __init__(self, config: DHCConfig):
        super().__init__(config)
        self.model = DHCModel(config)
        self.vocab_size = config.vocab_size
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.criterion = None

        # Initialize weights and apply final processing
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
            # Enable model parallelism
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