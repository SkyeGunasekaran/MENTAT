from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from typing import Optional

import torch
import torch.nn.functional as F

from paged_kv_cache import PagedKVCacheManager


# ============================================================================
# 1.  PREFIX TREE DATA STRUCTURES (paged variant)
# ============================================================================

@dataclass
class TreeNode:
    node_id: int
    token_ids: list[int] = field(default_factory=list)
    seq_id: Optional[int] = None           
    parent: Optional['TreeNode'] = None
    children: dict[int, 'TreeNode'] = field(default_factory=dict)
    cumulative_log_prob: float = 0.0
    depth: int = 0                          
    is_active: bool = True
    is_eos: bool = False
    semantic_vector: Optional[torch.Tensor] = None  
    entropy_ema: Optional[float] = None   # Track local entropy baseline          
    creation_step: int = 0

    def get_full_sequence(self) -> list[int]:
        """Collect all generated token ids from root to this node."""
        parts: list[list[int]] = []
        node = self
        while node is not None:
            if node.token_ids:
                parts.append(node.token_ids)
            node = node.parent
        parts.reverse()
        return [tok for seg in parts for tok in seg]


class PrefixTree:
    """
    Manages a tree of generation branches.  Supports extend, branch,
    and prune — all backed by paged KV-cache operations.
    """

    def __init__(self, kv_cache_mgr: PagedKVCacheManager):
        self.root = TreeNode(node_id=0)
        self._next_id = 1
        self.kv_cache_mgr = kv_cache_mgr

    def new_id(self) -> int:
        nid = self._next_id
        self._next_id += 1
        return nid

    # ---- extend -----------------------------------------------------------
    def extend_leaf(
        self,
        leaf: TreeNode,
        token_id: int,
        log_prob: float,
    ) -> TreeNode:
        """Append one token to *leaf* in-place."""
        leaf.token_ids.append(token_id)
        leaf.cumulative_log_prob += log_prob
        leaf.depth += 1
        return leaf

    # ---- branch -----------------------------------------------------------
    def branch_leaf(
        self,
        leaf: TreeNode,
        token_ids: list[int],
        log_probs: list[float],
    ) -> list[TreeNode]:
        """
        Create len(token_ids) children from *leaf*.

        Each child gets a forked paged-cache sequence (O(1) page-table
        copy).  The parent leaf is deactivated and its paged sequence
        is freed (refcounts handle sharing).
        """
        children: list[TreeNode] = []
        for tid, lp in zip(token_ids, log_probs):
            child_seq_id = self.kv_cache_mgr.fork_sequence(leaf.seq_id)
            child = TreeNode(
                node_id=self.new_id(),
                token_ids=[tid],
                seq_id=child_seq_id,
                parent=leaf,
                cumulative_log_prob=leaf.cumulative_log_prob + lp,
                depth=leaf.depth + 1,
                is_active=True,
                entropy_ema=leaf.entropy_ema
            )
            leaf.children[child.node_id] = child
            children.append(child)

        # Parent is no longer a leaf; free its paged sequence
        leaf.is_active = False
        if leaf.seq_id is not None:
            self.kv_cache_mgr.free_sequence(leaf.seq_id)
            leaf.seq_id = None

        return children

    # ---- prune ------------------------------------------------------------
    def prune_branch(self, node: TreeNode):
        """Remove *node*, free its paged cache, cascade up if needed."""
        if node.seq_id is not None:
            self.kv_cache_mgr.free_sequence(node.seq_id)
            node.seq_id = None
        node.is_active = False

        if node.parent is not None:
            parent = node.parent
            parent.children = {
                k: v for k, v in parent.children.items()
                if v.node_id != node.node_id
            }
            if (not parent.children and not parent.is_active
                    and parent.parent is not None):
                self.prune_branch(parent)

    # ---- queries ----------------------------------------------------------
    def get_active_leaves(self) -> list[TreeNode]:
        leaves: list[TreeNode] = []
        self._walk(self.root, leaves, want_active=True)
        return leaves

    def get_complete_sequences(self) -> list[TreeNode]:
        complete: list[TreeNode] = []
        self._walk(self.root, complete, want_active=False)
        return complete

    def _walk(self, node: TreeNode, out: list, *, want_active: bool):
        if want_active:
            if node.is_active and not node.children and not node.is_eos:
                out.append(node)
        else:
            if node.is_eos:
                out.append(node)
        for child in node.children.values():
            self._walk(child, out, want_active=want_active)


# ============================================================================
# 2.  Entropy & Penalty Utils  
# ============================================================================

def compute_entropy(logits: torch.Tensor, temperature: float = 1.0) -> float:
    """Shannon entropy of the softmax distribution (in nats)."""
    probs = F.softmax(logits / temperature, dim=-1)
    log_probs = F.log_softmax(logits / temperature, dim=-1)
    plogp = probs * log_probs
    plogp = torch.nan_to_num(plogp, nan=0.0)
    return -plogp.sum().item()


def apply_repetition_penalty(
    logits: torch.Tensor,
    token_ids: list[int],
    penalty: float = 1.2,
    frequency_penalty: float = 0.3,
    max_ngram_block: int = 3,
) -> torch.Tensor:
    """Apply repetition suppression to raw logits."""
    if not token_ids:
        return logits

    logits = logits.clone()

    if penalty != 1.0 or frequency_penalty > 0:
        counts: dict[int, int] = {}
        for tid in token_ids:
            counts[tid] = counts.get(tid, 0) + 1

        token_indices = list(counts.keys())
        token_counts = torch.tensor(
            [counts[t] for t in token_indices],
            device=logits.device, dtype=logits.dtype,
        )
        gathered = logits[token_indices]

        if penalty != 1.0:
            gathered = torch.where(
                gathered > 0,
                gathered / penalty,
                gathered * penalty,
            )
        if frequency_penalty > 0:
            gathered = gathered - frequency_penalty * token_counts

        logits[token_indices] = gathered

    if max_ngram_block >= 2 and len(token_ids) >= max_ngram_block:
        blocked = _find_blocked_tokens(token_ids, max_ngram_block)
        if blocked:
            logits[list(blocked)] = float('-inf')

    return logits


def _find_blocked_tokens(token_ids: list[int], max_n: int) -> set[int]:
    blocked: set[int] = set()
    seq_len = len(token_ids)
    for n in range(2, max_n + 1):
        if seq_len < n:
            continue
        suffix = tuple(token_ids[-(n - 1):])
        for i in range(seq_len - n + 1):
            if tuple(token_ids[i:i + n - 1]) == suffix:
                if i + n - 1 < seq_len:
                    blocked.add(token_ids[i + n - 1])
    return blocked


def adaptive_threshold(
    base_val: float,
    max_active: int,
    current_active: int,
) -> float:
    util = current_active / max(max_active, 1)
    return base_val * (1.0 + 2.0 * util * util)

# ============================================================================
# 3.  PAGED MODEL WRAPPER
# ============================================================================

class PagedModelWrapper:
    """
    Thin wrapper over TransformerForCausalLM that provides:

        prefill(input_ids, seq_id)  → logits
        decode_batch(token_ids_per_seq, seq_ids) → list[logits]

    It reaches inside the model to call ``attn.forward_paged_prefill()``
    and ``attn.forward_paged_decode()`` on each layer, threading the
    ``PagedKVCacheManager`` through.  The embedding, MLP, norm, and
    lm_head layers are called directly.

    This bypasses the standard ``model.forward()`` / ``FLACache`` path
    entirely.
    """

    def __init__(
        self,
        model,
        kv_cache_mgr: PagedKVCacheManager,
        eos_token_id: int | list[int] = 2,
    ):
        self.model = model
        self.kv_cache_mgr = kv_cache_mgr
        self.eos_token_id = eos_token_id
        self.device = next(model.parameters()).device
        self.dtype = next(model.parameters()).dtype

        # Extract model internals
        # TransformerForCausalLM → .model (TransformerModel)
        base = model.model if hasattr(model, 'model') else model
        self.embeddings = base.embed_tokens
        self.layers = base.layers          # nn.ModuleList[TransformerBlock]
        self.final_norm = base.norm
        self.lm_head = model.lm_head
        self.config = model.config

    # ------------------------------------------------------------------
    #  Prefill — single sequence
    # ------------------------------------------------------------------

    @torch.no_grad()
    def prefill(
        self,
        input_ids: torch.LongTensor,
        seq_id: int,
    ) -> torch.Tensor:
        """
        Run the full prompt through the model, writing KV into the
        paged cache for ``seq_id``.

        Args:
            input_ids: (1, seq_len) on the correct device.
            seq_id:    allocated sequence id in the kv_cache_mgr.

        Returns:
            logits: (vocab_size,) — logits for the *last* position.
        """
        hidden = self.embeddings(input_ids)  # (1, seq_len, D)

        for layer_idx, block in enumerate(self.layers):
            # -- Attention (paged prefill) --
            residual = hidden
            # REPLACED: attn_input = block.attn_norm(hidden)
            attn_input = block.input_layernorm(hidden)

            # REPLACED: block.attn.forward_paged_prefill
            attn_out, k_new, v_new = block.self_attn.forward_paged_prefill(
                hidden_states=attn_input,
            )

            self.kv_cache_mgr.append_tokens(seq_id, layer_idx, k_new, v_new)

            # -- Residual + MLP (replicating Qwen3DecoderLayer logic) --
            hidden = residual + attn_out
            residual = hidden

            # REPLACED: hidden = block.mlp_norm(hidden)
            hidden = block.post_attention_layernorm(hidden)

            hidden = block.mlp(hidden)
            hidden = residual + hidden

        hidden = self.final_norm(hidden)
        logits = self.lm_head(hidden[0, -1, :])  # (vocab_size,)
        return logits

    # ------------------------------------------------------------------
    #  Batched decode — all active leaves in one forward pass
    # ------------------------------------------------------------------

    @torch.no_grad()
    def decode_batch(
        self,
        token_ids_per_seq: list[int],
        seq_ids: list[int],
    ) -> tuple[list[torch.Tensor], torch.Tensor]:
        """
        Decode one token per sequence for a batch of active leaves.

        For each sequence, the new token's KV is appended to the paged
        cache layer-by-layer, then attention gathers from the full
        history (including the just-appended token).

        Optimisations over the naive per-layer path:
          - Append: slot-index GPU tensor is built once and reused
            across all layers (slots are always layer-invariant).
            Block allocations, CoW, and counter updates remain per-layer
            to preserve correctness.
          - Gather: slot-index tensor, cu_seqlens_k, and seq_lengths
            are computed once (after layer 0's append) and reused for
            layers 1..L-1.  Only the per-layer block-index tensor is
            rebuilt each layer.

        Net savings per step: eliminates (L-1) slot-index tensor
        allocations for append + (L-1) slot-index, cu_seqlens, and
        seq_lengths constructions for gather.

        Args:
            token_ids_per_seq: list of N token ids (one per leaf).
            seq_ids:           list of N paged-cache sequence ids.

        Returns:
            logits_list: List of N logits tensors, each (vocab_size,).
            hidden_states: (N, D) final hidden states before lm_head.
        """
        N = len(seq_ids)
        assert N == len(token_ids_per_seq)

        if N == 0:
            D = self.config.hidden_size
            empty_hidden = torch.empty(
                0, D, device=self.device, dtype=self.dtype,
            )
            return [], empty_hidden

        # -- Embedding --
        ids = torch.tensor(
            token_ids_per_seq, device=self.device, dtype=torch.long,
        ).unsqueeze(0)  # (1, N)
        hidden = self.embeddings(ids)  # (1, N, D)

        # Per-sequence RoPE position = number of tokens already cached
        # (this is the position of the NEW token being decoded).
        seqlen_offsets = [
            self.kv_cache_mgr.get_kv_length(sid) for sid in seq_ids
        ]

        # -- Precompute the shared slot-index tensor for appends --
        # Slot offsets are identical across all layers because
        # tokens_in_last_block is always in sync.  Build the GPU
        # tensor once and reuse it L times.
        append_slt_indices = (
            self.kv_cache_mgr.prepare_append_slot_indices(seq_ids)
        )
        append_slt_t = torch.tensor(
            append_slt_indices, device=self.device, dtype=torch.long,
        )

        cached_gather_meta = None

        for layer_idx, block in enumerate(self.layers):
            residual = hidden

            attn_input = block.input_layernorm(hidden)

            # Step 1: Project K/V + RoPE for the new decode tokens.
            k_new, v_new = self._project_kv(
                block.self_attn, attn_input, seq_ids, seqlen_offsets,
            )

            # Step 2: Append new K/V into paged cache.
            # Per-layer bookkeeping (block alloc, CoW, counter updates)
            # happens inside; only the slot-index tensor is shared.
            self.kv_cache_mgr.append_tokens_batched_fast(
                seq_ids=seq_ids,
                layer_idx=layer_idx,
                k=k_new,
                v=v_new,
                slot_idx_t=append_slt_t,
            )

            # Step 3: After layer 0's append, build the gather metadata
            # (slot indices, cu_seqlens, seq_lengths) that is shared
            # across all layers.  Block indices still differ per layer
            # and are built inside build_packed_kv.
            if layer_idx == 0:
                cached_gather_meta = (
                    self.kv_cache_mgr.prepare_gather_metadata(seq_ids)
                )

            # Step 4: Run Q-only attention reading from paged cache
            attn_out = block.self_attn.forward_paged_decode(
                hidden_states=attn_input,
                kv_cache_mgr=self.kv_cache_mgr,
                seq_ids=seq_ids,
                seqlen_offsets=seqlen_offsets,
                cached_gather_indices=cached_gather_meta,
            )

            # -- Residual + MLP --
            hidden = residual + attn_out
            residual = hidden

            hidden = block.post_attention_layernorm(hidden)

            hidden = block.mlp(hidden)
            hidden = residual + hidden

        hidden = self.final_norm(hidden)
        # hidden: (1, N, D) → hidden_flat: (N, D)
        hidden_flat = hidden.squeeze(0)
        # logits: (N, vocab_size)
        all_logits = self.lm_head(hidden_flat)

        return [all_logits[i] for i in range(N)], hidden_flat

    # ------------------------------------------------------------------
    #  Internal: lightweight K/V projection + RoPE (no attention)
    # ------------------------------------------------------------------

    def _project_kv(
        self,
        attn_module,
        hidden_states: torch.Tensor,
        seq_ids: list[int],
        seqlen_offsets: list[int],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Compute post-RoPE K and V for the new decode tokens without
        running the full attention computation.

        This avoids double-projecting: we project K/V here, append to
        the cache, then ``forward_paged_decode`` only needs to project
        Q and gather the full KV from the cache.

        Args:
            attn_module: the PagedAttention nn.Module for this layer.
            hidden_states: (1, N, D) — normed hidden states.
            seq_ids: list of N seq_ids.
            seqlen_offsets: list of N position offsets.

        Returns:
            k: (N, num_key_value_heads, head_dim) — post-RoPE K.
            v: (N, num_key_value_heads, head_dim) — V (no RoPE applied).
        """
        from einops import rearrange as _rearrange

        head_dim = attn_module.head_dim
        N = len(seq_ids)

        k = _rearrange(
            attn_module.k_proj(hidden_states),
            '... (h d) -> ... h d', d=head_dim,
        )
        v = _rearrange(
            attn_module.v_proj(hidden_states),
            '... (h d) -> ... h d', d=head_dim,
        )
        # k, v: (1, N, num_key_value_heads, head_dim)

        if attn_module.qk_norm:
            k = attn_module.k_norm(k)

        # Apply RoPE to K using Qwen3-compatible varlen interface.
        # Each decode token has its own position = seqlen_offsets[i].
        positions = torch.tensor(
            seqlen_offsets, device=k.device, dtype=torch.long,
        )
        # Squeeze to (N, num_kv_heads, head_dim) for varlen application
        k_squeezed = k.squeeze(0)
        k_rotated = attn_module.rotary.apply_varlen(k_squeezed, positions)

        v_out = v.squeeze(0)  # (N, num_key_value_heads, head_dim)

        return k_rotated, v_out

    # ------------------------------------------------------------------
    def is_eos(self, token_id: int) -> bool:
        if isinstance(self.eos_token_id, (list, tuple)):
            return token_id in self.eos_token_id
        return token_id == self.eos_token_id