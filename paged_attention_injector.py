import types
import torch
from einops import rearrange

from flash_attn import flash_attn_func, flash_attn_varlen_func

def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    """Standard rotation for RoPE."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)

def apply_rope_agnostic(q: torch.Tensor, k: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor):
    """
    Applies RoPE to Q and K. 
    Expects q, k to be (batch, seq, heads, dim).
    Expects cos, sin to be (batch, seq, dim) - standard HF output.
    """
    cos = cos.unsqueeze(-2) # (batch, seq, 1, dim)
    sin = sin.unsqueeze(-2)
    
    q_rot = (q * cos) + (_rotate_half(q) * sin) if q is not None else None
    k_rot = (k * cos) + (_rotate_half(k) * sin) if k is not None else None
    return q_rot, k_rot

def inject_paged_attention(model, window_size=None):
    """
    Dynamically injects paged prefill, KV projection, and batched decode 
    methods into standard Hugging Face attention modules.
    """
    if flash_attn_func is None:
        raise ImportError("Flash Attention is required for paged execution.")

    base_model = model.model if hasattr(model, 'model') else model
    
    for layer_idx, layer in enumerate(base_model.layers):
        attn = layer.self_attn
        
        # Ensure structural variables are set on the instance
        if not hasattr(attn, "head_dim"):
            attn.head_dim = attn.hidden_size // attn.num_heads
        attn.layer_idx = layer_idx
        attn.window_size = window_size

        # ------------------------------------------------------------------
        # 1. Paged Prefill
        # ------------------------------------------------------------------
        def forward_paged_prefill(self, hidden_states: torch.Tensor):
            batch_size, seq_len, _ = hidden_states.size()
            
            q = rearrange(self.q_proj(hidden_states), '... (h d) -> ... h d', d=self.head_dim)
            k = rearrange(self.k_proj(hidden_states), '... (h d) -> ... h d', d=self.head_dim)
            v = rearrange(self.v_proj(hidden_states), '... (h d) -> ... h d', d=self.head_dim)
            
            if hasattr(self, 'q_norm'): q = self.q_norm(q)
            if hasattr(self, 'k_norm'): k = self.k_norm(k)

            # Native HF RoPE expects (value_states, position_ids)
            position_ids = torch.arange(seq_len, device=q.device, dtype=torch.long).unsqueeze(0)
            cos, sin = self.rotary_emb(v, position_ids)
            q_rot, k_rot = apply_rope_agnostic(q, k, cos, sin)

            window = (-1, -1) if getattr(self, "window_size", None) is None else (self.window_size - 1, 0)
            o = flash_attn_func(q_rot, k_rot, v, causal=True, window_size=window)
            
            # Extract K/V to save to paged cache
            k_new = k_rot.squeeze(0)
            v_new = v.squeeze(0)
            
            o = o.reshape(1, seq_len, -1)
            o = self.o_proj(o)
            return o, k_new, v_new

        # ------------------------------------------------------------------
        # 2. KV Projection (Used during step 1 of decode loop)
        # ------------------------------------------------------------------
        def project_kv_paged(self, hidden_states: torch.Tensor, seqlen_offsets: list[int]):
            k = rearrange(self.k_proj(hidden_states), '... (h d) -> ... h d', d=self.head_dim)
            v = rearrange(self.v_proj(hidden_states), '... (h d) -> ... h d', d=self.head_dim)
            
            if hasattr(self, 'k_norm'): k = self.k_norm(k)
            
            position_ids = torch.tensor(seqlen_offsets, device=k.device, dtype=torch.long).unsqueeze(0)
            cos, sin = self.rotary_emb(v, position_ids)
            
            _, k_rot = apply_rope_agnostic(None, k, cos, sin)
            
            k_out = k_rot.squeeze(0) # (N, num_kv_heads, head_dim)
            v_out = v.squeeze(0)
            return k_out, v_out

        # ------------------------------------------------------------------
        # 3. Paged Decode (Gather + Attention)
        # ------------------------------------------------------------------
        def forward_paged_decode(self, hidden_states, kv_cache_mgr, seq_ids, seqlen_offsets, cached_gather_indices=None):
            N = len(seq_ids)
            _, total_q, _ = hidden_states.size()
            
            q = rearrange(self.q_proj(hidden_states), '... (h d) -> ... h d', d=self.head_dim)
            if hasattr(self, 'q_norm'): q = self.q_norm(q)
            
            # Apply native RoPE just to Q
            position_ids = torch.tensor(seqlen_offsets, device=q.device, dtype=torch.long).unsqueeze(0)
            cos, sin = self.rotary_emb(q, position_ids)
            q_rot, _ = apply_rope_agnostic(q, None, cos, sin)
            
            q_rot = q_rot.squeeze(0) # (N, num_heads, head_dim)
            
            packed_k, packed_v, cu_seqlens_k, max_seqlen_k = kv_cache_mgr.build_packed_kv(
                seq_ids, self.layer_idx, cached_metadata=cached_gather_indices
            )
            
            cu_seqlens_q = torch.arange(0, N + 1, device=q.device, dtype=torch.int32)
            window = (-1, -1) if getattr(self, "window_size", None) is None else (self.window_size - 1, 0)
            
            o = flash_attn_varlen_func(
                q_rot, packed_k, packed_v,
                cu_seqlens_q=cu_seqlens_q, cu_seqlens_k=cu_seqlens_k,
                max_seqlen_q=1, max_seqlen_k=max_seqlen_k,
                causal=True, window_size=window
            )
            
            o = o.unsqueeze(0).reshape(1, total_q, -1)
            return self.o_proj(o)

        # Bind the methods to the instance dynamically
        attn.forward_paged_prefill = types.MethodType(forward_paged_prefill, attn)
        attn.project_kv_paged = types.MethodType(project_kv_paged, attn)
        attn.forward_paged_decode = types.MethodType(forward_paged_decode, attn)