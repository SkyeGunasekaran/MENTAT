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
# 3.  PAGED MODEL WRAPPER  (adapter-based)
# ============================================================================

class PagedModelWrapper:
    """
    Thin wrapper over TransformerForCausalLM that provides:

        prefill(input_ids, seq_id)  → logits
        decode_batch(token_ids_per_seq, seq_ids) → list[logits]

    All model-specific logic (norm names, Q/K/V projection, RoPE style,
    residual patterns) is delegated to a ``ModelAdapter`` instance.
    The wrapper itself is fully model-agnostic.
    """

    def __init__(
        self,
        model,
        kv_cache_mgr: PagedKVCacheManager,
        adapter,  # ModelAdapter instance
        eos_token_id: int | list[int] = 2,
        window_size: int | None = None,
    ):
        self.model = model
        self.kv_cache_mgr = kv_cache_mgr
        self.adapter = adapter
        self.eos_token_id = eos_token_id
        self.window_size = window_size

        # Convenience aliases (read from the adapter)
        self.device = adapter.device
        self.dtype = adapter.dtype
        self.config = adapter.config
        self.embeddings = adapter.embeddings
        self.layers = adapter.layers
        self.final_norm = adapter.final_norm
        self.lm_head = adapter.lm_head

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

        Returns:
            logits: (vocab_size,) — logits for the *last* position.
        """
        hidden = self.embeddings(input_ids)  # (1, seq_len, D)
        adapter = self.adapter

        for layer_idx, block in enumerate(self.layers):
            residual = hidden
            attn_input = adapter.apply_pre_attn_norm(block, hidden)

            attn = adapter.get_self_attn(block)
            attn_out, k_new, v_new = adapter.forward_paged_prefill(
                attn, attn_input, layer_idx, self.window_size,
            )

            self.kv_cache_mgr.append_tokens(seq_id, layer_idx, k_new, v_new)

            hidden = adapter.apply_post_attn_residual(block, residual, attn_out)
            residual = hidden

            hidden = adapter.apply_pre_mlp_norm(block, hidden)
            hidden = adapter.get_mlp(block)(hidden)
            hidden = adapter.apply_post_mlp_residual(block, residual, hidden)

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

        Returns:
            logits_list: List of N logits tensors, each (vocab_size,).
            hidden_states: (N, D) final hidden states before lm_head.
        """
        N = len(seq_ids)
        assert N == len(token_ids_per_seq)
        adapter = self.adapter

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

        seqlen_offsets = [
            self.kv_cache_mgr.get_kv_length(sid) for sid in seq_ids
        ]

        # -- Precompute the shared slot-index tensor for appends --
        append_slt_indices = (
            self.kv_cache_mgr.prepare_append_slot_indices(seq_ids)
        )
        append_slt_t = torch.tensor(
            append_slt_indices, device=self.device, dtype=torch.long,
        )

        cached_gather_meta = None

        for layer_idx, block in enumerate(self.layers):
            residual = hidden

            attn_input = adapter.apply_pre_attn_norm(block, hidden)
            attn = adapter.get_self_attn(block)

            # Step 1: Project K/V + RoPE for the new decode tokens
            k_new, v_new = adapter.project_kv_for_cache(
                attn, attn_input, seqlen_offsets,
            )

            # Step 2: Append new K/V into paged cache
            self.kv_cache_mgr.append_tokens_batched_fast(
                seq_ids=seq_ids,
                layer_idx=layer_idx,
                k=k_new,
                v=v_new,
                slot_idx_t=append_slt_t,
            )

            # Step 3: Build gather metadata once (after layer 0)
            if layer_idx == 0:
                cached_gather_meta = (
                    self.kv_cache_mgr.prepare_gather_metadata(seq_ids)
                )

            # Step 4: Q-only attention reading from paged cache
            attn_out = adapter.forward_paged_decode(
                attn, attn_input,
                kv_cache_mgr=self.kv_cache_mgr,
                seq_ids=seq_ids,
                seqlen_offsets=seqlen_offsets,
                layer_idx=layer_idx,
                window_size=self.window_size,
                cached_gather_indices=cached_gather_meta,
            )

            # -- Residual + MLP --
            hidden = adapter.apply_post_attn_residual(block, residual, attn_out)
            residual = hidden

            hidden = adapter.apply_pre_mlp_norm(block, hidden)
            hidden = adapter.get_mlp(block)(hidden)
            hidden = adapter.apply_post_mlp_residual(block, residual, hidden)

        hidden = self.final_norm(hidden)
        hidden_flat = hidden.squeeze(0)  # (N, D)
        all_logits = self.lm_head(hidden_flat)  # (N, vocab_size)

        return [all_logits[i] for i in range(N)], hidden_flat

    # ------------------------------------------------------------------
    #  KV projection helper (used by generator for step-0 cache writes)
    # ------------------------------------------------------------------

    def _project_kv(
        self,
        attn_module,
        hidden_states: torch.Tensor,
        seq_ids: list[int],
        seqlen_offsets: list[int],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Delegate to adapter.project_kv_for_cache."""
        return self.adapter.project_kv_for_cache(
            attn_module, hidden_states, seqlen_offsets,
        )

    # ------------------------------------------------------------------
    def is_eos(self, token_id: int) -> bool:
        if isinstance(self.eos_token_id, (list, tuple)):
            return token_id in self.eos_token_id
        return token_id == self.eos_token_id