from __future__ import annotations

import math
import warnings
from typing import TYPE_CHECKING, Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from transformers.modeling_outputs import BaseModelOutputWithPast, CausalLMOutputWithPast
from transformers.modeling_utils import PreTrainedModel
from transformers.utils import logging
from transformers.utils.deprecation import deprecate_kwarg

from models.muddformer.attn import Attention
from models.muddformer.configuration_muddformer import MUDDFormerConfig
from fla.models.utils import Cache, FLAGenerationMixin
from fla.modules import FusedCrossEntropyLoss, FusedLinearCrossEntropyLoss, RMSNorm
from fla.modules import GatedMLP as MUDDFormerMLP
from fla.modules.l2warp import l2_warp

if TYPE_CHECKING:
    from transformers.processing_utils import Unpack

try:
    from transformers.modeling_layers import GradientCheckpointingLayer
except ImportError:
    from fla.models.modeling_layers import GradientCheckpointingLayer

logger = logging.get_logger(__name__)


class RMSNormNoScale(nn.Module):
    def __init__(self, eps=1e-6):
        super().__init__()
        self.eps = eps

    def forward(self, x):
        return x * torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + self.eps)

class DepthAggregateModule(nn.Module):
    """
    Depth-wise Aggregate (DA) module for MUDD connections.

    Generates dynamic per-position, per-way connection weights to aggregate
    outputs of all preceding layers into multiple input streams (Q, K, V, R)
    for the next layer.

    Following Eq. (5), (6) from the paper:
        A_i(X_i) = GELU(RMSNorm(X_i) @ W1) @ W2 + a_i
        X_bar_i = sum_j A_ij * X_j  (element-wise with broadcasting)

    With multiway (Eq. (8)), four separate sets of weights are generated
    for Q, K, V, R streams respectively.
    """

    def __init__(
        self,
        hidden_size: int,
        layer_idx: int,
        num_ways: int = 4,
        norm_eps: float = 1e-6,
        use_pre_post_da_norm: bool = False,
        is_last_layer: bool = False,
    ):
        super().__init__()

        self.hidden_size = hidden_size
        self.layer_idx = layer_idx
        self.num_ways = num_ways
        self.num_layers_accessible = layer_idx + 1  # layers 0..layer_idx (embedding + preceding blocks)
        self.use_pre_post_da_norm = use_pre_post_da_norm

        # If last layer, only initialize 1 way (the R stream)
        self.actual_num_ways = 1 if is_last_layer else num_ways

        # DA hidden dim K = num_ways * (layer_idx + 1), as per paper pseudocode
        if is_last_layer:
            da_hidden_dim = self.num_layers_accessible  # C=1 for last layer
        else:
            da_hidden_dim = num_ways * self.num_layers_accessible


        # Weight generation MLP: RMSNorm -> Linear(D, K) -> GELU -> Linear(K, C*(L+1))
        # W1: D -> K,  W2: K -> C*(L+1)
        self.da_norm = RMSNormNoScale(eps=norm_eps)
        self.w1 = nn.Linear(hidden_size, da_hidden_dim, bias=False)

        # Use actual_num_ways here
        self.w2 = nn.Linear(da_hidden_dim, self.actual_num_ways * self.num_layers_accessible, bias=False)
        self.static_weight = nn.Parameter(torch.zeros(self.actual_num_ways, self.num_layers_accessible))

        if use_pre_post_da_norm:
            self.pre_da_norms = nn.ModuleList([nn.RMSNorm(hidden_size, eps=norm_eps) for _ in range(self.num_layers_accessible)])
            # Use actual_num_ways here
            self.post_da_norms = nn.ModuleList([nn.RMSNorm(hidden_size, eps=norm_eps) for _ in range(self.actual_num_ways)])

    def forward(
        self,
        layer_outputs: list[torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Args:
            layer_outputs: List of (layer_idx + 1) tensors, each [B, T, D].
                           layer_outputs[0] = embedding, layer_outputs[i] = output of block i.

        Returns:
            Tuple of 4 tensors (xq, xk, xv, xr), each [B, T, D].
        """
        assert len(layer_outputs) == self.num_layers_accessible

        # Use the most recent layer output to generate dynamic weights
        x_current = layer_outputs[-1]  # [B, T, D]

        # Generate dynamic connection weights: Eq. (6)
        # dw: [B, T, C * (L+1)]
        dw = F.gelu(self.w1(self.da_norm(x_current)))
        dw = self.w2(dw)  # [B, T, C * num_layers_accessible]

        # Add static weight prior
        # dw: [B, T, C * num_layers_accessible] -> [C, B, T, num_layers_accessible]
        dw = rearrange(dw, 'B T (C L) -> C B T L', C=self.actual_num_ways)
        dw = dw + self.static_weight[:, None, None, :]  # broadcast static weights

        # Optionally normalize layer outputs (PreDANorm)
        if self.use_pre_post_da_norm:
            xs_normed = [self.pre_da_norms[j](layer_outputs[j]) for j in range(self.num_layers_accessible)]
        else:
            xs_normed = layer_outputs

        # Stack layer outputs: [L+1, B, T, D]
        xs_stacked = torch.stack(xs_normed, dim=0)

        # Compute weighted sum across all layers and ways simultaneously in C++/CUDA
        # dw: [C, B, T, L+1], xs_stacked: [L+1, B, T, D] -> aggregated_all: [C, B, T, D]
        aggregated_all = torch.einsum('c b t l, l b t d -> c b t d', dw, xs_stacked)

        results = []
        for c in range(self.actual_num_ways):
            aggregated = aggregated_all[c]

            # PostDANorm + residual (Section 2.5)
            if self.use_pre_post_da_norm:
                aggregated = self.post_da_norms[c](aggregated) + x_current
            results.append(aggregated)

        return results  # (xq, xk, xv, xr)


class MUDDFormerBlock(GradientCheckpointingLayer):
    """
    MUDDFormer block with multi-input attention (Eq. 7):
        X'_A = MHA(LN(X^Q), LN(X^K), LN(X^V)) + X^R
        B'(X^Q, X^K, X^V, X^R) = FFN(LN(X'_A)) + X'_A
    """

    def __init__(self, config: MUDDFormerConfig, layer_idx: int):
        super().__init__()

        self.config = config
        self.layer_idx = layer_idx

        # Three separate norms for Q, K, V inputs
        self.attn_norm_q = (RMSNorm if config.fuse_norm else nn.RMSNorm)(config.hidden_size, eps=config.norm_eps)
        self.attn_norm_k = (RMSNorm if config.fuse_norm else nn.RMSNorm)(config.hidden_size, eps=config.norm_eps)
        self.attn_norm_v = (RMSNorm if config.fuse_norm else nn.RMSNorm)(config.hidden_size, eps=config.norm_eps)

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

        self.mlp_norm = (RMSNorm if config.fuse_norm else nn.RMSNorm)(config.hidden_size, eps=config.norm_eps)

        # Compute re-allocated FFN intermediate size (Section 2.4)
        # Determine base intermediate size from either explicit config or hidden_ratio
        base_intermediate_size = config.intermediate_size
        if base_intermediate_size is None and config.hidden_ratio is not None:
            base_intermediate_size = int(config.hidden_size * config.hidden_ratio * 2 / 3)
            base_intermediate_size = max(256, (base_intermediate_size + 127) // 256 * 256)

        intermediate_size = base_intermediate_size
        if config.param_realloc and base_intermediate_size is not None:
            L = config.num_hidden_layers
            i = layer_idx + 1  # 1-indexed layer
            # D'_f(i) = (0.5*(L-i) + 1.5*(i-1)) / (L-1) * D_f
            if L > 1:
                scale = (0.5 * (L - i) + 1.5 * (i - 1)) / (L - 1)
                intermediate_size = int(round(scale * base_intermediate_size))
                # Round to nearest multiple of 256 for efficiency
                intermediate_size = max(256, (intermediate_size + 127) // 256 * 256)
            # else: single layer, keep original

        self.mlp = MUDDFormerMLP(
            hidden_size=config.hidden_size,
            hidden_ratio=None if intermediate_size is not None else config.hidden_ratio,
            intermediate_size=intermediate_size,
            hidden_act=config.hidden_act,
            fuse_swiglu=config.fuse_swiglu,
        )

        # DA module for this block (generates inputs for the NEXT block)
        # layer_idx is 0-based; DA at block i accesses outputs from embedding + blocks 0..i
        is_last_layer = layer_idx == (config.num_hidden_layers - 1)
        self.da = DepthAggregateModule(
            hidden_size=config.hidden_size,
            layer_idx=layer_idx + 1,  # +1 because we include the embedding as layer 0
            num_ways=config.num_ways,
            norm_eps=config.norm_eps,
            use_pre_post_da_norm=config.use_pre_post_da_norm,
            is_last_layer=is_last_layer,
        )

    def forward(
        self,
        xq: torch.Tensor,
        xk: torch.Tensor,
        xv: torch.Tensor,
        xr: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        past_key_values: tuple[torch.Tensor] | None = None,
        output_attentions: bool | None = False,
        use_cache: bool | None = False,
        **kwargs: Any,
    ) -> tuple[torch.FloatTensor, ...]:
        """
        Forward pass for a MUDDFormer block with decoupled Q/K/V/R inputs.

        Args:
            xq, xk, xv, xr: Decoupled input streams [B, T, D].
            attention_mask: Padding mask.
            past_key_values: KV cache.
            output_attentions: Whether to return attention weights.
            use_cache: Whether to return KV cache.

        Returns:
            Tuple of (block_output, [attentions], [past_key_values]).
        """
        # Eq. (7): X'_A = MHA(LN(X^Q), LN(X^K), LN(X^V)) + X^R
        normed_q = self.attn_norm_q(xq)
        normed_k = self.attn_norm_k(xk)
        normed_v = self.attn_norm_v(xv)

        attn_out, attentions, past_key_values = self.attn(
            hidden_states=normed_q,
            hidden_states_k=normed_k,
            hidden_states_v=normed_v,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            use_cache=use_cache,
            output_attentions=output_attentions,
            **kwargs,
        )

        # Add residual from R stream
        hidden_states = attn_out + xr

        # FFN with Pre-Norm
        if self.config.fuse_norm:
            hidden_states, residual = self.mlp_norm(hidden_states, None, True)
        else:
            residual = hidden_states
            hidden_states = self.mlp_norm(hidden_states)
        hidden_states = self.mlp(hidden_states, **kwargs)

        hidden_states = residual + hidden_states

        outputs = (hidden_states,)

        if output_attentions:
            outputs += (attentions,)

        if use_cache:
            outputs += (past_key_values,)

        return outputs


class MUDDFormerPreTrainedModel(PreTrainedModel):

    config_class = MUDDFormerConfig
    base_model_prefix = 'model'
    supports_gradient_checkpointing = True
    _no_split_modules = ['MUDDFormerBlock']
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

        # Special initialization for DA modules (Section 3, Implementation Details)
        if isinstance(module, DepthAggregateModule):
            D = self.config.hidden_size
            # W1 initialized with N(0, 1/D)
            nn.init.normal_(module.w1.weight, mean=0.0, std=1.0 / D)
            # W2 initialized with 0
            nn.init.zeros_(module.w2.weight)

            # Static weight: identity-like (a_ii = 1, rest = 0)
            # i.e., the most recent layer gets weight 1, others get 0
            # This reduces MUDDFormer to Transformer at init
            with torch.no_grad():
                module.static_weight.zero_()
                if not module.use_pre_post_da_norm:
                    # Set a_ii = 1 for the last (most recent) layer for all ways
                    module.static_weight[:, -1] = 1.0
                # If PrePostDANorm, static weights are 0 because X_i is added as residual

            # Initialize PrePostDANorm scale parameters
            if module.use_pre_post_da_norm:
                for norm in module.pre_da_norms:
                    if hasattr(norm, 'weight') and norm.weight is not None:
                        nn.init.ones_(norm.weight)
                for norm in module.post_da_norms:
                    if hasattr(norm, 'weight') and norm.weight is not None:
                        nn.init.constant_(norm.weight, 1e-3)

        if rescale_prenorm_residual:
            p = None
            if hasattr(module, 'o_proj'):
                p = module.o_proj.weight
            elif hasattr(module, 'down_proj'):
                p = module.down_proj.weight
            if p is not None:
                nn.init.kaiming_uniform_(p, a=math.sqrt(5))
                with torch.no_grad():
                    p /= math.sqrt(num_residuals_per_layer * self.config.num_hidden_layers)


class MUDDFormerModel(MUDDFormerPreTrainedModel):

    def __init__(
        self,
        config: MUDDFormerConfig,
    ) -> MUDDFormerModel:
        super().__init__(config)
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size

        self.embeddings = nn.Embedding(config.vocab_size, config.hidden_size, self.padding_idx)
        self.layers = nn.ModuleList([
            MUDDFormerBlock(config, layer_idx)
            for layer_idx in range(config.num_hidden_layers)
        ])
        self.norm = (RMSNorm if config.fuse_norm else nn.RMSNorm)(config.hidden_size, eps=config.norm_eps)

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
        **kwargs: Any,
    ) -> tuple | BaseModelOutputWithPast:
        if output_attentions:
            warnings.warn(
                "`MUDDFormerModel` does not support output attention weights now, "
                "so `output_attentions` is set to `False`.",
            )
            output_attentions = False
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        use_cache = use_cache if use_cache is not None else (self.config.use_cache if not self.training else False)
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both input_ids and inputs_embeds at the same time")
        elif input_ids is None and inputs_embeds is None:
            raise ValueError("You have to specify either input_ids or inputs_embeds")

        if use_cache:
            if past_key_values is None:
                past_key_values = Cache()  # Initialize fresh cache for step 1 of generation
            elif not isinstance(past_key_values, Cache):
                past_key_values = Cache.from_legacy_cache(past_key_values)

        if inputs_embeds is None:
            inputs_embeds = self.embeddings(input_ids)

        hidden_states = inputs_embeds

        if self.gradient_checkpointing and self.training:
            if hasattr(hidden_states, "requires_grad") and not hidden_states.requires_grad:
                hidden_states.requires_grad_(True)

        # MUDDFormer: maintain list of all layer outputs for dense connections
        # Eq. (8): X^Q_0 = X^K_0 = X^V_0 = X^R_0 = X_0 = Embedding(X)
        layer_outputs_list = [hidden_states]  # Start with embedding output
        xq, xk, xv, xr = hidden_states, hidden_states, hidden_states, hidden_states

        all_hidden_states = () if output_hidden_states else None
        all_attns = () if output_attentions else None
        next_cache = None

        for i, layer in enumerate(self.layers):
            if output_hidden_states:
                all_hidden_states += (hidden_states,)

            # Forward through block with decoupled inputs
            block_outputs = layer(
                xq,
                xk,
                xv,
                xr,
                attention_mask=attention_mask,
                past_key_values=past_key_values,
                output_attentions=output_attentions,
                use_cache=use_cache,
                **kwargs,
            )

            hidden_states = block_outputs[0]

            if use_cache:
                next_cache = block_outputs[2 if output_attentions else 1]

            if output_attentions:
                all_attns += (block_outputs[1],)

            # Append current block output to the list
            layer_outputs_list.append(hidden_states)

            # DA module: generate Q, K, V, R inputs for the next block
            # DA is called outside the checkpointed block because it needs the
            # full layer_outputs_list which contains tensors from prior blocks.
            # Gradient checkpointing cannot handle this growing list of external tensors.
            da_outputs = layer.da(layer_outputs_list)

            if i == len(self.layers) - 1:
                # Last layer: only the R stream is generated
                xr = da_outputs[0]
            else:
                # Normal layers: Q, K, V, R streams are generated
                xq, xk, xv, xr = da_outputs

        # Final output is the R stream from the last DA, as per Eq. (8):
        # MUDDFormer(X) = X^R_L
        hidden_states = xr
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


class MUDDFormerForCausalLM(MUDDFormerPreTrainedModel, FLAGenerationMixin):

    _tied_weights_keys = ["lm_head.weight"]

    def __init__(self, config):
        super().__init__(config)
        self.model = MUDDFormerModel(config)
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
        **kwargs: Any,
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

        # Ensure safe slicing if logits_to_keep is provided
        slice_idx = -logits_to_keep if (logits_to_keep is not None and logits_to_keep > 0) else None

        logits = None if self.config.fuse_linear_cross_entropy else self.lm_head(
            hidden_states[:, slice_idx:] if slice_idx else hidden_states
        )

        loss = None
        if labels is not None:
            if getattr(self, 'criterion', None) is None:
                if self.config.fuse_linear_cross_entropy:
                    self.criterion = FusedLinearCrossEntropyLoss(use_l2warp=self.config.use_l2warp)
                elif self.config.fuse_cross_entropy:
                    self.criterion = FusedCrossEntropyLoss(inplace_backward=True)
                else:
                    self.criterion = nn.CrossEntropyLoss()
            criterion = self.criterion

            labels = labels.to(hidden_states.device)
            # Shift labels
            labels = torch.cat((labels[..., 1:], torch.full_like(labels[:, :1], criterion.ignore_index)), 1)

            # Apply the same truncation to labels and fused hidden_states
            if slice_idx:
                labels = labels[:, slice_idx:]
                if self.config.fuse_linear_cross_entropy:
                    hidden_states = hidden_states[:, slice_idx:]

            if self.config.fuse_linear_cross_entropy:
                loss = criterion(hidden_states, labels, self.lm_head.weight, self.lm_head.bias)
            else:
                # Use .reshape() to avoid memory contiguous errors
                loss = criterion(logits.reshape(labels.numel(), -1), labels.reshape(-1))
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