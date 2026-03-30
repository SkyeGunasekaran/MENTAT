# Copyright (c) 2023-2025, Songlin Yang, Yu Zhang
# Configuration for Transformer with Dynamic Hyper-Connections (DHC)

import warnings

from transformers.configuration_utils import PretrainedConfig


class DHCConfig(PretrainedConfig):
    """
    Configuration class for Transformer with Dynamic Hyper-Connections.
    
    This configuration extends the standard Transformer config with 
    hyper-connection specific parameters.
    
    Args:
        hidden_size: Dimension of the hidden states
        num_hidden_layers: Number of transformer layers
        num_heads: Number of attention heads
        num_kv_heads: Number of key-value heads (for GQA)
        qkv_bias: Whether to use bias in QKV projections
        qk_norm: Whether to apply normalization to Q and K
        window_size: Window size for sliding window attention
        rope_theta: Base for rotary position embeddings
        max_position_embeddings: Maximum sequence length
        hidden_ratio: Ratio for FFN hidden dimension
        intermediate_size: FFN intermediate dimension (overrides hidden_ratio)
        hidden_act: Activation function
        initializer_range: Standard deviation for weight initialization
        elementwise_affine: Whether to use learnable affine in normalization
        norm_eps: Epsilon for normalization
        use_cache: Whether to use KV cache
        pad_token_id: Padding token ID
        bos_token_id: Beginning of sequence token ID
        eos_token_id: End of sequence token ID
        tie_word_embeddings: Whether to tie input and output embeddings
        fuse_norm: Whether to use fused normalization
        fuse_swiglu: Whether to use fused SwiGLU
        fuse_cross_entropy: Whether to use fused cross entropy
        fuse_linear_cross_entropy: Whether to use fused linear cross entropy
        use_l2warp: Whether to use L2 warp
        vocab_size: Vocabulary size
        
        # Hyper-connection specific parameters
        expansion_rate: Number of hyper hidden vectors (n in paper)
        use_dynamic_hc: Whether to use dynamic hyper-connections
        use_tanh: Whether to apply tanh to dynamic weights
        hc_norm_type: Normalization type for dynamic HC ('layer' or 'rms')
    """

    model_type = 'dhc'
    keys_to_ignore_at_inference = ['past_key_values']

    def __init__(
        self,
        hidden_size: int = 2048,
        num_hidden_layers: int = 24,
        num_heads: int = 32,
        num_kv_heads: int | None = None,
        qkv_bias: bool = False,
        qk_norm: bool = False,
        window_size: int | None = None,
        rope_theta: float | None = 10000.,
        max_position_embeddings: int = 2048,
        hidden_ratio: int | None = 4,
        intermediate_size: int | None = None,
        hidden_act: str = "swish",
        initializer_range: float = 0.02,
        elementwise_affine: bool | None = True,
        norm_eps: float = 1e-6,
        use_cache: bool = True,
        pad_token_id: int | None = None,
        bos_token_id: int = 1,
        eos_token_id: int = 2,
        tie_word_embeddings: bool = False,
        fuse_norm: bool = True,
        fuse_swiglu: bool = True,
        fuse_cross_entropy: bool = True,
        fuse_linear_cross_entropy: bool = False,
        use_l2warp: bool = False,
        vocab_size: int = 32000,
        # Hyper-connection specific parameters
        expansion_rate: int = 4,
        use_dynamic_hc: bool = True,
        use_tanh: bool = True,
        hc_norm_type: str = 'rms',
        **kwargs,
    ):
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.qkv_bias = qkv_bias
        self.qk_norm = qk_norm
        self.window_size = window_size
        self.rope_theta = rope_theta
        self.max_position_embeddings = max_position_embeddings

        self.hidden_ratio = hidden_ratio
        self.intermediate_size = intermediate_size
        self.hidden_act = hidden_act

        self.initializer_range = initializer_range
        self.elementwise_affine = elementwise_affine
        self.norm_eps = norm_eps
        self.use_cache = use_cache

        self.fuse_norm = fuse_norm
        self.fuse_swiglu = fuse_swiglu
        self.fuse_cross_entropy = fuse_cross_entropy
        self.fuse_linear_cross_entropy = fuse_linear_cross_entropy
        self.use_l2warp = use_l2warp
        self.vocab_size = vocab_size

        # Hyper-connection specific parameters
        self.expansion_rate = expansion_rate
        self.use_dynamic_hc = use_dynamic_hc
        self.use_tanh = use_tanh
        self.hc_norm_type = hc_norm_type

        if fuse_cross_entropy and fuse_linear_cross_entropy:
            raise ValueError(
                "`fuse_cross_entropy` and `fuse_linear_cross_entropy` cannot be True at the same time.",
            )
        if fuse_linear_cross_entropy:
            warnings.warn(
                "`fuse_linear_cross_entropy` is enabled, which can improve memory efficiency "
                "at the potential cost of reduced precision. "
                "If you observe issues like loss divergence, consider disabling this setting.",
            )
        
        if expansion_rate < 1:
            raise ValueError(
                f"`expansion_rate` must be at least 1, got {expansion_rate}."
            )
        
        if expansion_rate == 1 and use_dynamic_hc:
            warnings.warn(
                "Using expansion_rate=1 with dynamic hyper-connections may not provide "
                "benefits over the baseline. Consider using expansion_rate >= 2."
            )

        super().__init__(
            pad_token_id=pad_token_id,
            bos_token_id=bos_token_id,
            eos_token_id=eos_token_id,
            tie_word_embeddings=tie_word_embeddings,
            **kwargs,
        )