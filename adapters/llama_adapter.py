import torch
from einops import rearrange

try:
    from flash_attn import flash_attn_func, flash_attn_varlen_func
except ImportError:
    "Flash attention not found! Please install with `pip install flash_attn --no-build-isolation`!"

from adapters.model_adapter import ModelAdapter

class LlamaAdapter(ModelAdapter):
    """
    Adapter for LlamaForCausalLM / Llama-3.x HuggingFace checkpoints.

    The architecture is structurally identical to Qwen3 in every way that
    matters for the paged pipeline:

    1. **RoPE is model-level, not per-attention.**
    2. **QKV layout is ``(batch, heads, seq, head_dim)`` after projection.**
    3. **No QK-norm.** Unlike Qwen3, ``LlamaAttention`` has no ``q_norm`` /
    4. **No sliding window.** All layers use full causal attention.

    Because points 1–2 are the same caveats that required overriding all three
    high-level ops in ``Qwen3Adapter``, we do the same here.

    Llama model layout:
        model.model.embed_tokens
        model.model.rotary_emb          
        model.model.layers[i]
            .input_layernorm
            .self_attn                 
                .q_proj / .k_proj / .v_proj / .o_proj
                (no q_norm / k_norm)
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
    #  Returns tensors in FA layout (batch, seq, heads, head_dim).
    #  No QK-norm applied (Llama has none).
    # ------------------------------------------------------------------

    def project_qkv(self, attn, hidden_states):
        head_dim = self.head_dim
        q = rearrange(attn.q_proj(hidden_states), "b s (h d) -> b s h d", d=head_dim)
        k = rearrange(attn.k_proj(hidden_states), "b s (h d) -> b s h d", d=head_dim)
        v = rearrange(attn.v_proj(hidden_states), "b s (h d) -> b s h d", d=head_dim)
        return q, k, v

    # ------------------------------------------------------------------
    #  RoPE  (abstract interface — implemented for completeness, but the
    #  three overridden high-level ops call _build_cos_sin directly)
    # ------------------------------------------------------------------

    def apply_rope(self, attn, q, k, position_ids, v_for_shape=None):
        ref = q if q is not None else (k if k is not None else v_for_shape)
        cos, sin = self._rotary_emb(ref, position_ids)
        cos = cos.unsqueeze(2)   # (batch, seq, 1, head_dim) for heads broadcast
        sin = sin.unsqueeze(2)
        q_rot = self._apply_cos_sin(q, cos, sin) if q is not None else None
        k_rot = self._apply_cos_sin(k, cos, sin) if k is not None else None
        return q_rot, k_rot

    # ------------------------------------------------------------------
    #  RoPE helpers  (same pattern as Qwen3Adapter)
    # ------------------------------------------------------------------

    def _build_cos_sin(self, position_ids: torch.Tensor, ref: torch.Tensor):
        """
        Call the shared rotary_emb and return (cos, sin) broadcast-ready for
        FA layout (batch, seq, heads, head_dim).

        ``LlamaRotaryEmbedding.forward(x, position_ids)``:
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

    # ------------------------------------------------------------------
    #  Override high-level paged ops
    #
    #  Identical structure to Qwen3Adapter but without QK-norm.
    # ------------------------------------------------------------------

    def forward_paged_prefill(self, attn, hidden_states, layer_idx, window_size=None):
        """
        Prefill for one Llama layer.  All tensors in FA layout (b, s, h, d).

        Returns:
            attn_output : (1, seq_len, hidden_size)
            k_new       : (seq_len, num_kv_heads, head_dim)  — post-RoPE, for cache
            v_new       : (seq_len, num_kv_heads, head_dim)
        """
        _, seq_len, _ = hidden_states.size()

        q, k, v = self.project_qkv(attn, hidden_states)
        # q, k, v: (1, seq_len, heads, head_dim)

        position_ids = torch.arange(seq_len, device=q.device, dtype=torch.long).unsqueeze(0)
        cos, sin = self._build_cos_sin(position_ids, q)
        q = self._apply_cos_sin(q, cos, sin)
        k = self._apply_cos_sin(k, cos, sin)

        # Llama has no sliding window, so window_size is always None here,
        # but we honour the parameter for forward-compatibility.
        window = (-1, -1) if window_size is None else (window_size - 1, 0)
        o = flash_attn_func(q, k, v, causal=True, window_size=window)

        k_new = k[0]   # (seq_len, num_kv_heads, head_dim)
        v_new = v[0]

        o = o.reshape(1, seq_len, -1)
        o = attn.o_proj(o)
        return o, k_new, v_new

    def project_kv_for_cache(self, attn, hidden_states, seqlen_offsets):
        """
        Project K/V + RoPE for N new decode tokens (no Q, no attention).

        Returns:
            k : (N, num_kv_heads, head_dim)  — post-RoPE
            v : (N, num_kv_heads, head_dim)
        """
        # hidden_states: (1, N, D) → treat as (N, 1, D) for per-token RoPE
        hs = hidden_states.squeeze(0).unsqueeze(1)   # (N, 1, D)

        _, k, v = self.project_qkv(attn, hs)
        # k, v: (N, 1, num_kv_heads, head_dim)

        position_ids = torch.tensor(
            seqlen_offsets, device=k.device, dtype=torch.long
        ).unsqueeze(1)   # (N, 1)
        cos, sin = self._build_cos_sin(position_ids, k)
        k = self._apply_cos_sin(k, cos, sin)

        return k.squeeze(1), v.squeeze(1)   # (N, num_kv_heads, head_dim)

    def forward_paged_decode(self, attn, hidden_states, kv_cache_mgr,
                             seq_ids, seqlen_offsets, layer_idx,
                             window_size=None, cached_gather_indices=None):
        """
        Batched decode attention for one Llama layer.

        Returns:
            attn_output : (1, N, hidden_size)
        """
        N = len(seq_ids)
        _, total_q, _ = hidden_states.size()

        hs = hidden_states.squeeze(0).unsqueeze(1)   # (N, 1, D)
        q, _, _ = self.project_qkv(attn, hs)
        # q: (N, 1, num_heads, head_dim)

        position_ids = torch.tensor(
            seqlen_offsets, device=q.device, dtype=torch.long
        ).unsqueeze(1)   # (N, 1)
        cos, sin = self._build_cos_sin(position_ids, q)
        q = self._apply_cos_sin(q, cos, sin)
        q_flat = q.squeeze(1)   # (N, num_heads, head_dim)

        packed_k, packed_v, cu_seqlens_k, max_seqlen_k = kv_cache_mgr.build_packed_kv(
            seq_ids, layer_idx,
        )

        cu_seqlens_q = torch.arange(0, N + 1, device=q.device, dtype=torch.int32)

        o = flash_attn_varlen_func(
            q_flat, packed_k, packed_v,
            cu_seqlens_q=cu_seqlens_q,
            cu_seqlens_k=cu_seqlens_k,
            max_seqlen_q=1,
            max_seqlen_k=max_seqlen_k,
            causal=True,
            window_size=(-1, -1),
        )
        # o: (N, num_heads, head_dim)
        o = o.unsqueeze(0).reshape(1, total_q, -1)
        return attn.o_proj(o)