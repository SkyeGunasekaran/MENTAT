"""ResFormer configuration - Implementation of Learnable ResFormer Plus from the paper
"Value Residual Learning" (Zhou et al., 2024).

This implements the Learnable ResFormer Plus variant which uses:
- Learnable λ_{n,1} and λ_{n,2} coefficients for value residual mixing
- Softmax-normalized λ_{n,1} values scaled by a learnable λ_scale parameter
- V'_n = λ_{n,1} * V_1 + λ_{n,2} * V_n
"""

import warnings
from transformers.configuration_utils import PretrainedConfig


class ResFormerConfig(PretrainedConfig):
    """Configuration class for ResFormer (Learnable ResFormer Plus).
    
    This configuration implements the "Learnable ResFormer Plus" variant from the paper,
    which applies value residual connections with learnable mixing coefficients.
    
    The value residual formula is: V'_n = λ_{n,1} * V_1 + λ_{n,2} * V_n
    
    For Learnable ResFormer Plus:
    - λ_{n,2} is initialized to 0.5 for all layers n = 1, 2, ..., N
    - λ_{n,1} is initialized to: λ_scale * softmax(λ'_{n,1})
      where λ_scale is initialized to N (total layers) and shared across all layers
    """

    model_type = 'ResFormer'
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
        # Value residual hyperparameters (Learnable ResFormer Plus)
        use_value_residual: bool = True,
        value_residual_lambda2_init: float = 0.5,
        value_residual_last_k: int | None = None,
        **kwargs,
    ):
        """Initialize ResFormer configuration.
        
        Args:
            hidden_size: Dimension of the hidden representations.
            num_hidden_layers: Number of transformer layers.
            num_heads: Number of attention heads.
            num_kv_heads: Number of key-value heads (for GQA). None = same as num_heads.
            qkv_bias: Whether to use bias in Q, K, V projections.
            qk_norm: Whether to apply RMSNorm to Q and K.
            window_size: Sliding window attention size. None = full attention.
            rope_theta: Base for rotary position embeddings.
            max_position_embeddings: Maximum sequence length.
            hidden_ratio: MLP hidden dimension ratio (hidden_size * hidden_ratio).
            intermediate_size: Explicit MLP intermediate size (overrides hidden_ratio).
            hidden_act: Activation function for MLP.
            initializer_range: Standard deviation for weight initialization.
            elementwise_affine: Whether to use elementwise affine in norms.
            norm_eps: Epsilon for layer normalization.
            use_cache: Whether to use KV cache.
            pad_token_id: Padding token ID.
            bos_token_id: Beginning of sequence token ID.
            eos_token_id: End of sequence token ID.
            tie_word_embeddings: Whether to tie input/output embeddings.
            fuse_norm: Whether to use fused RMSNorm.
            fuse_swiglu: Whether to use fused SwiGLU.
            fuse_cross_entropy: Whether to use fused cross entropy.
            fuse_linear_cross_entropy: Whether to use fused linear cross entropy.
            use_l2warp: Whether to use L2 warp.
            vocab_size: Vocabulary size.
            use_value_residual: Whether to use value residual connections.
            value_residual_lambda2_init: Initial value for λ_{n,2} (default 0.5).
            value_residual_last_k: Number of final layers to apply value residual.
                None = apply to all layers after layer 0.
        """
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

        # Value residual hyperparameters (Learnable ResFormer Plus)
        self.use_value_residual = use_value_residual
        self.value_residual_lambda2_init = value_residual_lambda2_init
        self.value_residual_last_k = value_residual_last_k

        super().__init__(
            pad_token_id=pad_token_id,
            bos_token_id=bos_token_id,
            eos_token_id=eos_token_id,
            tie_word_embeddings=tie_word_embeddings,
            **kwargs,
        )