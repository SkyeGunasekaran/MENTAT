"""
Prefix-Tree Uncertainty Quantification Generator — Paged Batched Edition
=========================================================================

Entropy-gated branching over a TransformerForCausalLM model, using
paged KV-cache and batched decoding for throughput.

Key differences from the original:
  • TreeNode stores a ``seq_id`` (int) instead of a full FLACache.
  • Branching uses ``kv_cache_mgr.fork_sequence()`` (O(1) page-table
    copy + refcount bumps) instead of ``clone_cache()`` (full tensor
    deep-copy).
  • All active leaves are decoded in a **single batched forward pass**
    per step via ``flash_attn_varlen_func``, rather than N sequential
    ``decode_one`` calls.
  • Pruning calls ``kv_cache_mgr.free_sequence()`` to release pages.

Usage:
    from paged_kv_cache import PagedKVCacheManager

    model = TransformerForCausalLM.from_pretrained(...)
    tokenizer = AutoTokenizer.from_pretrained(...)
    gen = PagedPrefixTreeUQGenerator(model, tokenizer)
    results = gen.generate("The meaning of life is")
"""

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
    entropy_ema: Optional[float] = None             # NEW: Track local entropy baseline
    creation_step: int = 0
    is_pruned: bool = False

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
                entropy_ema=leaf.entropy_ema,
                semantic_vector=leaf.semantic_vector.clone() if leaf.semantic_vector is not None else None
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
        """
        Deactivate *node*, free its paged cache, and cascade upward.

        Nodes are marked ``is_pruned = True`` but **kept in the tree** so
        the visualisation can render them (red for directly-pruned leaves,
        orange for cascade-pruned internal nodes).

        Cascade rule: after marking *node*, check its parent.  If the
        parent has no surviving (non-pruned) children and is itself
        inactive (i.e. it was already branched, not currently generating),
        mark the parent as pruned too and recurse upward.  The root node
        is never pruned.
        """
        if node.seq_id is not None:
            self.kv_cache_mgr.free_sequence(node.seq_id)
            node.seq_id = None
        node.is_active = False
        node.is_pruned = True

        # Cascade upward through dead internal ancestors
        if node.parent is not None:
            parent = node.parent
            has_surviving_child = any(
                not child.is_pruned for child in parent.children.values()
            )
            if (not has_surviving_child and not parent.is_active
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
# 2.  ENTROPY / PENALTY UTILITIES  (unchanged from original)
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
) -> torch.Tensor:
    """Apply repetition suppression to raw logits."""
    # Fast exit if there are no tokens or the penalty is exactly 1.0
    if not token_ids or penalty == 1.0:
        return logits

    logits = logits.clone()

    # Standard repetition penalty 
    unique_tokens = list(set(token_ids))
    gathered = logits[unique_tokens]

    # Apply the multiplicative penalty
    gathered = torch.where(
        gathered > 0,
        gathered / penalty,
        gathered * penalty,
    )

    logits[unique_tokens] = gathered

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
    Model-agnostic wrapper that drives the paged KV-cache inference loop.

    All model-family-specific logic (layer access, QKV projection, RoPE,
    norm names) is delegated to a ``ModelAdapter`` instance.  The wrapper
    itself is pure pipeline mechanics: embed → per-layer (norm, attn, mlp,
    residual) → final-norm → lm_head.

    Interface::

        wrapper = PagedModelWrapper(model, kv_cache_mgr, adapter, eos_token_id=eos)
        logits           = wrapper.prefill(input_ids, seq_id)
        logits_list, hs  = wrapper.decode_batch(token_ids, seq_ids)
    """

    def __init__(
        self,
        model,
        kv_cache_mgr: PagedKVCacheManager,
        adapter,                            # ModelAdapter instance
        eos_token_id: int | list[int] = 2,
    ):
        self.model = model
        self.kv_cache_mgr = kv_cache_mgr
        self.adapter = adapter
        self.eos_token_id = eos_token_id
        self.device = adapter.device
        self.dtype = adapter.dtype
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
        adapter = self.adapter
        hidden = adapter.embeddings(input_ids)  # (1, seq_len, D)

        for layer_idx, block in enumerate(adapter.layers):
            attn = adapter.get_self_attn(block)

            # -- Pre-attention norm + paged prefill attention --
            residual = hidden
            attn_input = adapter.apply_pre_attn_norm(block, hidden)
            attn_out, k_new, v_new = adapter.forward_paged_prefill(
                attn, attn_input, layer_idx,
            )
            self.kv_cache_mgr.append_tokens(seq_id, layer_idx, k_new, v_new)
            hidden = adapter.apply_post_attn_residual(block, residual, attn_out)

            # -- Pre-MLP norm + MLP --
            residual = hidden
            mlp_input = adapter.apply_pre_mlp_norm(block, hidden)
            mlp_out = adapter.get_mlp(block)(mlp_input)
            hidden = adapter.apply_post_mlp_residual(block, residual, mlp_out)

        hidden = adapter.final_norm(hidden)
        logits = adapter.lm_head(hidden[0, -1, :])  # (vocab_size,)
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

        For each sequence the new token's KV is appended to the paged cache
        layer-by-layer, then attention gathers from the full history
        (including the just-appended token).

        Args:
            token_ids_per_seq: list of N token ids (one per leaf).
            seq_ids:           list of N paged-cache sequence ids.

        Returns:
            logits_list:   List of N logits tensors, each (vocab_size,).
            hidden_states: (N, D) final hidden states before lm_head.
        """
        adapter = self.adapter
        N = len(seq_ids)
        assert N == len(token_ids_per_seq)

        if N == 0:
            D = self.config.hidden_size
            return [], torch.empty(0, D, device=self.device, dtype=self.dtype)

        # -- Embedding --
        ids = torch.tensor(
            token_ids_per_seq, device=self.device, dtype=torch.long,
        ).unsqueeze(0)  # (1, N)
        hidden = adapter.embeddings(ids)  # (1, N, D)

        # Per-sequence RoPE position = number of tokens already cached
        # (this is the position of the NEW token being decoded).
        seqlen_offsets = [
            self.kv_cache_mgr.get_kv_length(sid) for sid in seq_ids
        ]

        for layer_idx, block in enumerate(adapter.layers):
            attn = adapter.get_self_attn(block)

            residual = hidden
            attn_input = adapter.apply_pre_attn_norm(block, hidden)

            # Step 1: Project K/V + RoPE for the new decode tokens and
            # append them to the paged cache *before* running attention so
            # the new token attends to itself.
            k_new, v_new = adapter.project_kv_for_cache(
                attn, attn_input, seqlen_offsets,
            )
            self.kv_cache_mgr.append_tokens_batched(
                seq_ids=seq_ids,
                layer_idx=layer_idx,
                k=k_new,
                v=v_new,
            )

            # Step 2: Q-only projection + varlen flash attention over cache.
            attn_out = adapter.forward_paged_decode(
                attn, attn_input, self.kv_cache_mgr,
                seq_ids, seqlen_offsets, layer_idx,
            )

            hidden = adapter.apply_post_attn_residual(block, residual, attn_out)

            # -- MLP --
            residual = hidden
            mlp_input = adapter.apply_pre_mlp_norm(block, hidden)
            mlp_out = adapter.get_mlp(block)(mlp_input)
            hidden = adapter.apply_post_mlp_residual(block, residual, mlp_out)

        hidden = adapter.final_norm(hidden)
        hidden_flat = hidden.squeeze(0)         # (N, D)
        all_logits = adapter.lm_head(hidden_flat)  # (N, vocab_size)

        return [all_logits[i] for i in range(N)], hidden_flat

    # ------------------------------------------------------------------
    def is_eos(self, token_id: int) -> bool:
        if isinstance(self.eos_token_id, (list, tuple)):
            return token_id in self.eos_token_id
        return token_id == self.eos_token_id