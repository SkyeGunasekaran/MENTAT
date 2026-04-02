from adapters.model_adapter import ModelAdapter
import torch
from einops import rearrange

from adapters.attention_backend import attn_prefill, attn_decode

class Qwen3Adapter(ModelAdapter):
    """
    Adapter for the stock HuggingFace Qwen3ForCausalLM checkpoint.

    Three things make Qwen3 distinct from the base-class assumptions:

    **RoPE is model-level, not per-attention.**
    **QKV tensor layout is ``(batch, heads, seq, head_dim)``.**
    **QK-norm is unconditional.**

    Because points 1–2 break the layout contract assumed by the base-class
    forward_paged_prefill, project_kv_for_cache, and
    forward_paged_decode, we override all three of those methods here rather
    than trying to patch the base class with layout flags.

    Qwen3 model layout:
        model.model.embed_tokens
        model.model.rotary_emb          
        model.model.layers[i]
            .input_layernorm
            .self_attn                  
                .q_proj / .k_proj / .v_proj / .o_proj
                .q_norm / .k_norm       
            .post_attention_layernorm
            .mlp
        model.model.norm
        model.lm_head
    """

    def __init__(self, model):
        super().__init__(model)
        base = model.model if hasattr(model, "model") else model
        self._embeddings = base.embed_tokens
        self._layers = base.layers
        self._final_norm = base.norm
        self._lm_head = model.lm_head
        # Cache the single shared RoPE module.
        self._rotary_emb = base.rotary_emb

    # ------------------------------------------------------------------
    #  Per-layer accessors
    # ------------------------------------------------------------------

    def get_self_attn(self, layer):
        return layer.self_attn

    def get_mlp(self, layer):
        return layer.mlp

    def apply_pre_attn_norm(self, layer, hidden):
        return layer.input_layernorm(hidden)

    def apply_pre_mlp_norm(self, layer, hidden):
        return layer.post_attention_layernorm(hidden)

    # ------------------------------------------------------------------
    #  QKV projection
    #
    #  Returns tensors in FlashAttention layout: (batch, seq, heads, head_dim).
    #  The HF Qwen3Attention uses (batch, heads, seq, head_dim) internally
    #  (i.e. it transposes immediately after projection), but our paged paths
    #  need the FA layout, so we stay in FA layout throughout.
    # ------------------------------------------------------------------

    def project_qkv(self, attn, hidden_states):
        """
        Project hidden_states → (Q, K, V) in (batch, seq, heads, head_dim) layout.

        QK-norm is applied here, on the pre-transpose view, matching the HF
        implementation exactly (RMSNorm is element-wise on the last dim so
        layout does not affect the numerical result).
        """
        head_dim = self.head_dim
        
        # Project and reshape to (batch, seq, heads, head_dim) — FA layout.
        # Extract batch and sequence length dynamically from the input tensor
        b, s, _ = hidden_states.shape

        # Project and reshape to (batch, seq, heads, head_dim) — FA layout.
        # Using -1 automatically infers the number of heads (h) based on head_dim.
        q = attn.q_proj(hidden_states).view(b, s, -1, head_dim)
        k = attn.k_proj(hidden_states).view(b, s, -1, head_dim)
        v = attn.v_proj(hidden_states).view(b, s, -1, head_dim)
        # QK-norm is unconditional on Qwen3 (always present).
        q = attn.q_norm(q)
        k = attn.k_norm(k)
        return q, k, v

    # ------------------------------------------------------------------
    #  RoPE
    #
    #  ``apply_rope`` is the abstract interface used by the *base class* high-
    #  level ops.  We override all three of those ops below, so this method is
    #  only called if a subclass delegates back up.  We implement it for
    #  completeness using the shared rotary_emb.
    #
    #  NOTE: ``Qwen3RotaryEmbedding.forward(x, position_ids)`` uses x only for
    #  dtype/device — it does NOT need to be a (batch, heads, seq, dim) tensor.
    #  We pass any available tensor as the device/dtype hint.
    # ------------------------------------------------------------------

    def apply_rope(self, attn, q, k, position_ids, v_for_shape=None):
        """
        Apply RoPE using the shared model-level rotary_emb.

        Inputs/outputs are in (batch, seq, heads, head_dim) layout (FA layout).
        ``Qwen3RotaryEmbedding.forward`` returns (cos, sin) shaped
        (batch, seq, head_dim); ``apply_rotary_pos_emb`` with unsqueeze_dim=2
        broadcasts them to the heads dimension correctly for FA layout.
        """
        ref = q if q is not None else (k if k is not None else v_for_shape)
        cos, sin = self._rotary_emb(ref, position_ids)
        # cos/sin: (batch, seq, head_dim) → unsqueeze at dim 2 for heads
        cos = cos.unsqueeze(2)
        sin = sin.unsqueeze(2)

        def _rotate_half(x):
            x1 = x[..., : x.shape[-1] // 2]
            x2 = x[..., x.shape[-1] // 2 :]
            return torch.cat((-x2, x1), dim=-1)

        q_rot = (q * cos + _rotate_half(q) * sin) if q is not None else None
        k_rot = (k * cos + _rotate_half(k) * sin) if k is not None else None
        return q_rot, k_rot

    # ------------------------------------------------------------------
    #  Override high-level paged ops
    #
    #  The base-class versions assume FlashAttention layout throughout and
    #  compute RoPE via attn.rotary — neither holds for stock Qwen3.  We
    #  override all three ops here with Qwen3-correct implementations.
    # ------------------------------------------------------------------

    def _build_cos_sin(self, position_ids: torch.Tensor, ref: torch.Tensor):
        """
        Call the shared rotary_emb and return (cos, sin) broadcast-ready for
        FA layout (batch, seq, heads, head_dim).

        ``Qwen3RotaryEmbedding.forward(x, position_ids)``:
          - x        : used only for dtype / device
          - returns  : cos, sin each (batch, seq, head_dim)
        """
        cos, sin = self._rotary_emb(ref, position_ids)
        # (batch, seq, head_dim) → (batch, seq, 1, head_dim) for heads broadcast
        return cos.unsqueeze(2), sin.unsqueeze(2)

    @staticmethod
    def _apply_cos_sin(t, cos, sin):
        x1 = t[..., : t.shape[-1] // 2]
        x2 = t[..., t.shape[-1] // 2 :]
        rotated = torch.cat((-x2, x1), dim=-1)
        return t * cos + rotated * sin

    def forward_paged_prefill(self, attn, hidden_states, layer_idx, window_size=None):
        """
        Prefill for one Qwen3 layer.

        All intermediate tensors are in FA layout (batch, seq, heads, head_dim).
        ``flash_attn_func`` requires this layout, so no transpose is needed
        before/after the attention call.

        Returns:
            attn_output : (1, seq_len, hidden_size)
            k_new       : (seq_len, num_kv_heads, head_dim)  — post-RoPE, for cache
            v_new       : (seq_len, num_kv_heads, head_dim)
        """
        batch_size, seq_len, _ = hidden_states.size()

        q, k, v = self.project_qkv(attn, hidden_states)
        # q: (1, seq_len, num_heads, head_dim)  — FA layout

        position_ids = torch.arange(seq_len, device=q.device, dtype=torch.long).unsqueeze(0)
        cos, sin = self._build_cos_sin(position_ids, q)
        q = self._apply_cos_sin(q, cos, sin)
        k = self._apply_cos_sin(k, cos, sin)

        window = None if window_size is None else window_size
        o = attn_prefill(q, k, v, window_size=window)
        # o: (1, seq_len, num_heads, head_dim)

        # Cache needs (seq_len, num_kv_heads, head_dim) — just drop batch dim.
        k_new = k[0]
        v_new = v[0]

        o = o.reshape(1, seq_len, -1)
        o = attn.o_proj(o)
        return o, k_new, v_new

    def project_kv_for_cache(self, attn, hidden_states, seqlen_offsets):
        """
        Project K/V + RoPE for N new decode tokens (no Q, no attention).

        Each token has its own position offset (seqlen_offsets[i]).
        We treat each decode token as a batch of N independent sequences of
        length 1, so position_ids is (N, 1).

        Returns:
            k : (N, num_kv_heads, head_dim)  — post-RoPE
            v : (N, num_kv_heads, head_dim)
        """
        N = len(seqlen_offsets)
        # hidden_states: (1, N, D) → treat as (N, 1, D) for per-token RoPE
        hs = hidden_states.squeeze(0).unsqueeze(1)  # (N, 1, D)

        _, k, v = self.project_qkv(attn, hs)
        # k, v: (N, 1, num_kv_heads, head_dim)

        position_ids = torch.tensor(
            seqlen_offsets, device=k.device, dtype=torch.long
        ).unsqueeze(1)  # (N, 1)
        cos, sin = self._build_cos_sin(position_ids, k)
        k = self._apply_cos_sin(k, cos, sin)

        # Squeeze the seq-len-1 dimension → (N, num_kv_heads, head_dim)
        return k.squeeze(1), v.squeeze(1)

    def forward_paged_decode(self, attn, hidden_states, kv_cache_mgr,
                             seq_ids, seqlen_offsets, layer_idx,
                             window_size=None, cached_gather_indices=None):
        """
        Batched decode attention for one Qwen3 layer.

        Projects Q only (K/V already in cache), applies per-token RoPE,
        then either:
          - Triton path: calls kv_cache_mgr.decode_attention (fused paged
            attention reading directly from the KV pools), or
          - Fallback path: gathers packed KV with build_packed_kv, then runs
            flash_attn_varlen_func / SDPA.

        Returns:
            attn_output : (1, N, hidden_size)
        """
        N = len(seq_ids)
        _, total_q, _ = hidden_states.size()

        # --- Q projection + RoPE (unchanged) ---
        hs = hidden_states.squeeze(0).unsqueeze(1)  # (N, 1, D)
        q, _, _ = self.project_qkv(attn, hs)
        # q: (N, 1, num_heads, head_dim)

        position_ids = torch.tensor(
            seqlen_offsets, device=q.device, dtype=torch.long
        ).unsqueeze(1)  # (N, 1)
        cos, sin = self._build_cos_sin(position_ids, q)
        q = self._apply_cos_sin(q, cos, sin)
        q_flat = q.squeeze(1)  # (N, num_heads, head_dim)

        # --- Attention: Triton fast path vs. gather fallback ---
        if hasattr(kv_cache_mgr, 'decode_attention'):
            # Triton PagedAttention — reads K/V directly from paged pools.
            # No build_packed_kv gather, no index tensor construction.
            o = kv_cache_mgr.decode_attention(q_flat, seq_ids, layer_idx)
        else:
            # Original path: gather KV into packed tensors + varlen attention.
            packed_k, packed_v, cu_seqlens_k, max_seqlen_k = kv_cache_mgr.build_packed_kv(
                seq_ids, layer_idx,
            )
            cu_seqlens_q = torch.arange(0, N + 1, device=q.device, dtype=torch.int32)
            window = None if window_size is None else window_size
            o = attn_decode(
                q_flat, packed_k, packed_v,
                cu_seqlens_q=cu_seqlens_q,
                cu_seqlens_k=cu_seqlens_k,
                max_seqlen_k=max_seqlen_k,
                N=N,
                window_size=window,
            )

        # o: (N, num_heads, head_dim) → reshape and project
        o = o.unsqueeze(0).reshape(1, total_q, -1)
        return attn.o_proj(o)