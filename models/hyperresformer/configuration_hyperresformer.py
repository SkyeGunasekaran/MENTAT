"""HyperResFormer configuration - Token-dependent value residual mixing inspired by
Hyper-Connections (Zhu et al., 2024) and Value Residual Learning (Zhou et al., 2024).

The core idea replaces the global learned lambda_{n,1} scalar of Learnable ResFormer Plus
with a per-token, per-head mixing coefficient derived from the hidden state:

    alpha_n(h) = gate(W_alpha @ h)
    V'_n = V_n + alpha_n(h) * V_1

The gating function is configurable via ``value_residual_gate`` to support ablation studies.
Supported modes and their properties:

+--------------------+-----------------------------------+----------+-----------+---------------------------+
| Gate               | Formula                           | Convex   | Bounded   | Notes                     |
+====================+===================================+==========+===========+===========================+
| ``relu``           | ReLU(Wh)                          | No       | [0, ∞)   | Default; sparse & free    |
| ``sigmoid``        | σ(Wh)                             | Yes      | [0, 1]    | Per-head soft gate        |
| ``softmax``        | softmax(Wh) * num_kv_heads        | Yes      | convex    | Across-head competition   |
| ``softmax_sigmoid``| softmax(Wh) * σ(Wh) * num_kv_heads| Yes     | convex    | Convex + per-head gate    |
| ``tanh``           | tanh(Wh)                          | No       | [−1, 1]  | Allows subtraction        |
| ``identity``       | Wh (no activation)                | —        | (−∞, ∞)  | Linear baseline           |
+--------------------+-----------------------------------+----------+-----------+---------------------------+

``softmax`` and ``softmax_sigmoid`` compute softmax over the num_kv_heads dimension,
making alpha a convex combination (heads compete for V_1 budget). They are then scaled
by num_kv_heads so that the expected per-head alpha is ~1 at initialisation, matching
the scale of the other gates.

This allows each token to independently decide how much of the first layer's value
representation to mix in, giving the model strictly more expressive power than a
shared scalar while remaining lightweight (hidden_size * num_kv_heads extra params
per mixing layer).
"""

import warnings
from transformers.configuration_utils import PretrainedConfig


class HyperResFormerConfig(PretrainedConfig):
    """Configuration class for HyperResFormer.

    HyperResFormer extends the value residual idea from Learnable ResFormer Plus by
    making the mixing coefficient token-dependent rather than a global scalar.

    The value residual formula is:

        alpha_n(h) = gate(W_alpha @ h)   [B, T, num_kv_heads, 1]
        V'_n = V_n + alpha_n(h) * V_1

    The ``gate`` function is controlled by ``value_residual_gate`` and supports
    several ablation modes — see module docstring for the full comparison table.

    Compared to Learnable ResFormer Plus:
    - No shared lambda_scale parameter (removed)
    - No per-layer softmax-normalised lambda logits (removed)
    - Each mixing layer has a small linear projection W_alpha (hidden_size -> num_kv_heads)
    - The gate function is configurable for ablation studies
    """

    model_type = 'HyperResFormer'
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
        # Hyper value residual hyperparameters
        use_value_residual: bool = True,
        value_residual_gate: str = 'relu',
        value_residual_proj_bias: bool = False,
        value_residual_last_k: int | None = None,
        **kwargs,
    ):
        """Initialise HyperResFormer configuration.

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
            initializer_range: Standard deviation for weight initialisation.
            elementwise_affine: Whether to use elementwise affine in norms.
            norm_eps: Epsilon for layer normalisation.
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
            use_value_residual: Whether to use hyper value residual connections.
            value_residual_gate: Gating function applied to the alpha projection output.
                Choices:
                - ``'relu'``           — ReLU(Wh); non-convex, sparse, unbounded (default).
                - ``'sigmoid'``        — σ(Wh); convex, bounded in [0, 1].
                - ``'softmax'``        — softmax over heads; convex combination across heads.
                - ``'softmax_sigmoid'``— softmax * sigmoid; convex + per-head gating.
                - ``'tanh'``           — tanh(Wh); allows negative (subtractive) mixing.
                - ``'identity'``       — no activation; linear baseline.
            value_residual_proj_bias: Whether to add a bias term to the alpha projection
                (W_alpha). Defaults to False.
            value_residual_last_k: Number of final layers to apply value residual to.
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

        # Hyper value residual hyperparameters
        _valid_gates = {'relu', 'sigmoid', 'softmax', 'softmax_sigmoid', 'tanh', 'identity'}
        if value_residual_gate not in _valid_gates:
            raise ValueError(
                f"value_residual_gate must be one of {sorted(_valid_gates)}, "
                f"got '{value_residual_gate}'."
            )
        self.use_value_residual = use_value_residual
        self.value_residual_gate = value_residual_gate
        self.value_residual_proj_bias = value_residual_proj_bias
        self.value_residual_last_k = value_residual_last_k

        super().__init__(
            pad_token_id=pad_token_id,
            bos_token_id=bos_token_id,
            eos_token_id=eos_token_id,
            tie_word_embeddings=tie_word_embeddings,
            **kwargs,
        )