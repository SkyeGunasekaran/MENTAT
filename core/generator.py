from __future__ import annotations

import math
import time
from dataclasses import dataclass, field

import torch
import torch.nn.functional as F

from core.paged_kv_cache import PagedKVCacheManager
from core.generator_utils import TreeNode, PrefixTree, compute_entropy, apply_repetition_penalty, _find_blocked_tokens, adaptive_threshold, PagedModelWrapper
from adapters.adapter_factory import ModelAdapter, get_adapter

class MentatGenerator:
    """
    Uncertainty-quantification generator using entropy-gated prefix-tree
    branching, backed by paged KV-cache and batched decoding.

    At each decode step:
      1. Collect all active leaves.
      2. Run a single batched forward pass for all of them.
      3. For each leaf, decide branch vs. extend based on entropy.
      4. Prune low-probability branches.

    Branching is O(1) in memory (page-table fork + refcount bumps).
    Pruning releases pages via refcount decrement.
    """

    def __init__(
        self,
        model,
        tokenizer,
        *,
        adapter: ModelAdapter | None = None,
        max_active_branches: int = 10,
        branching_factor: int = 3,
        relative_entropy_multiplier: float = 1.25,
        entropy_ema_alpha: float = 0.3,
        max_new_tokens: int = 256,
        temperature: float = 0.7,
        repetition_penalty: float = 1.2,
        # Paged cache config
        block_size: int = 16,
        max_blocks: int = 8192,
        # Semantic diversity pruning
        semantic_similarity_threshold: float = 0.85,
        ema_alpha: float = 0.3,
        min_steps_before_prune: int = 5,
        # Soft exploration warmup
        soft_explore_window: int = 10,
        soft_explore_initial: float = 0.3,
    ):
        eos = tokenizer.eos_token_id
        if eos is None:
            eos = 2
        self.tokenizer = tokenizer

        self.M = max_active_branches
        self.K = branching_factor
        self.relative_entropy_multiplier = relative_entropy_multiplier # CHANGED
        self.entropy_ema_alpha = entropy_ema_alpha                     # NEW
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.rep_penalty = repetition_penalty
        self.semantic_similarity_threshold = semantic_similarity_threshold
        self.ema_alpha = ema_alpha
        self.min_steps_before_prune = min_steps_before_prune
        self.soft_explore_window = soft_explore_window
        self.soft_explore_initial = soft_explore_initial

        # Resolve adapter — auto-detect if not supplied.
        if adapter is None:
            adapter = get_adapter(model)
        self.adapter = adapter

        # Derive cache geometry from the adapter (model-agnostic).
        num_key_value_heads = adapter.num_key_value_heads
        num_hidden_layers = adapter.num_hidden_layers
        head_dim = adapter.head_dim
        device = adapter.device
        dtype = adapter.dtype

        # Auto-size block pool if not specified.
        # Worst case: max_active branches × max_new_tokens tokens each,
        # plus headroom for CoW copies during branching.
        if max_blocks is None:
            tokens_budget = max_active_branches * (max_new_tokens + 512)
            max_blocks = (tokens_budget // block_size + 1) * 2
            max_blocks = max(max_blocks, 256)

        # Create the paged KV-cache manager
        self.kv_cache_mgr = PagedKVCacheManager(
            num_layers=num_hidden_layers,
            num_key_value_heads=num_key_value_heads,
            head_dim=head_dim,
            max_blocks=max_blocks,
            block_size=block_size,
            device=device,
            dtype=dtype,
        )

        # Create paged model wrapper
        self.wrapper = PagedModelWrapper(
            model, self.kv_cache_mgr, adapter, eos_token_id=eos,
        )

        # Diagnostics
        self.branch_points: list[tuple[int, float, int]] = []
        self.pruning_events: list[tuple[int, int]] = []
        self.entropy_trace: list[tuple[int, int, float]] = []
        self.diversity_pruning_events: list[tuple[int, int]] = []  # (step, count)

        # Performance / throughput metrics (populated during generate())
        self.active_branch_trace: list[int] = []   # active branches at each decode step
        self._total_tokens_decoded: int = 0        # decode-phase tokens (excluding prefill)
        self._prefill_tokens: int = 0              # prompt token count
        self._prefill_time_s: float = 0.0          # wall time for prefill
        self._decode_time_s: float = 0.0           # wall time for decode loop only
        self._peak_vram_bytes: int = 0             # peak CUDA memory allocated (bytes)

    # ------------------------------------------------------------------
    #  Public API
    # ------------------------------------------------------------------

    def generate(self, prompt: str) -> list[dict]:
        """
        Run the full prefix-tree generation with batched decoding.

        Returns a list of dicts sorted by cumulative log-prob (desc):
            {'token_ids': [...], 'text': str, 'log_prob': float}
        """
        # Reset per-run metrics so the object is reusable across prompts.
        self.active_branch_trace = []
        self._total_tokens_decoded = 0
        self._prefill_tokens = 0
        self._prefill_time_s = 0.0
        self._decode_time_s = 0.0
        self._peak_vram_bytes = 0

        _on_cuda = (self.wrapper.device.type == "cuda")
        if _on_cuda:
            torch.cuda.reset_peak_memory_stats(self.wrapper.device)

        # Encode prompt
        enc = self.tokenizer(prompt, return_tensors='pt')
        input_ids = enc['input_ids'].to(self.wrapper.device)
        self._prefill_tokens = input_ids.shape[-1]

        # ---- Prefill ----
        _t0_prefill = time.perf_counter()
        seq_id = self.kv_cache_mgr.allocate_sequence()
        logits = self.wrapper.prefill(input_ids, seq_id)
        if _on_cuda:
            torch.cuda.synchronize(self.wrapper.device)
        self._prefill_time_s = time.perf_counter() - _t0_prefill

        # Build tree
        tree = PrefixTree(self.kv_cache_mgr)
        first_leaf = TreeNode(
            node_id=tree.new_id(),
            token_ids=[],
            seq_id=seq_id,
            parent=tree.root,
            cumulative_log_prob=0.0,
            depth=0,
            is_active=True,
        )
        tree.root.children[0] = first_leaf

        self._step_leaf(tree, first_leaf, logits, step=0, num_active=1)

        # ---- Main decode loop ----
        _t0_decode = time.perf_counter()
        for step in range(1, self.max_new_tokens):
            active = tree.get_active_leaves()
            if not active:
                break

            num_active = len(active)
            self.active_branch_trace.append(num_active)

            # -- Batched forward pass for all active leaves --
            token_ids_batch = [leaf.token_ids[-1] for leaf in active]
            seq_ids_batch = [leaf.seq_id for leaf in active]

            logits_list, hidden_states = self.wrapper.decode_batch(
                token_ids_batch, seq_ids_batch,
            )
            # Each forward pass decodes one token per active sequence.
            self._total_tokens_decoded += num_active

            # -- Update semantic vectors via EMA --
            self._update_semantic_vectors(active, hidden_states)

            # -- Per-leaf branch/extend decisions (unified path with
            #    soft-exploration warmup baked into the threshold) --
            for leaf, leaf_logits in zip(list(active), logits_list):
                if not leaf.is_active:
                    continue
                self._step_leaf(tree, leaf, leaf_logits, step, num_active)

            # -- Semantic diversity prune  --
            self._prune_similar(tree, step)

            # -- Early stop --
            if not tree.get_active_leaves():
                break

        if _on_cuda:
            torch.cuda.synchronize(self.wrapper.device)
        self._decode_time_s = time.perf_counter() - _t0_decode

        # Snapshot peak CUDA memory (covers both prefill and decode).
        if _on_cuda:
            self._peak_vram_bytes = torch.cuda.max_memory_allocated(self.wrapper.device)

        # ---- Collect results ----
        # ---- Collect results ----
        results = []
        
        # Combine both complete and active leaves for processing
        all_nodes = tree.get_complete_sequences() + tree.get_active_leaves()
        
        for node in all_nodes:
            tids = node.get_full_sequence()
            length = max(len(tids), 1) # Prevent division by zero
            
            # Calculate length-normalized probability
            norm_prob = math.exp(node.cumulative_log_prob / length)
            
            results.append({
                'token_ids': tids,
                'text': self.tokenizer.decode(tids, skip_special_tokens=True),
                'log_prob': node.cumulative_log_prob,
                'norm_prob': norm_prob, # Add the new metric
            })

        # Free any remaining paged sequences
        self._free_all_sequences(tree)

        # Sort using the newly calculated normalized probability
        results.sort(key=lambda r: r['norm_prob'], reverse=True)

        self.tree = tree
        return results

    # ------------------------------------------------------------------
    #  Internals
    # ------------------------------------------------------------------
    def _handle_eos(self, tree: PrefixTree, node: TreeNode) -> None:
        """Checks if the node's last token is EOS, and if so, deactivates and frees it."""
        if self.wrapper.is_eos(node.token_ids[-1]):
            node.is_eos = True
            node.is_active = False
            if node.seq_id is not None:
                tree.kv_cache_mgr.free_sequence(node.seq_id)
                node.seq_id = None

    def _free_all_sequences(self, tree: PrefixTree):
        """Release all paged sequences still held by tree nodes."""
        self._free_walk(tree.root)

    def _free_walk(self, node: TreeNode):
        """Recursively free paged sequences."""
        if node.seq_id is not None:
            self.kv_cache_mgr.free_sequence(node.seq_id)
            node.seq_id = None
        for child in node.children.values():
            self._free_walk(child)

    # ------------------------------------------------------------------
    #  Branching helpers
    # ------------------------------------------------------------------

    def _do_branch(
        self,
        tree: PrefixTree,
        leaf: TreeNode,
        logits: torch.Tensor,
        step: int,
    ):
        """Branch *leaf* into top-K children via paged fork."""
        log_probs = F.log_softmax(logits, dim=-1)
        topk_vals, topk_ids = torch.topk(log_probs, self.K)

        token_ids = topk_ids.tolist()
        token_lps = topk_vals.tolist()

        # branch_leaf internally calls kv_cache_mgr.fork_sequence()
        # for each child — O(1) per fork, no tensor copies.
        children = tree.branch_leaf(leaf, token_ids, token_lps)
        self.branch_points.append(
            (step, compute_entropy(logits), len(children))
        )

        for child in children: 
            child.creation_step = step
            self._handle_eos(tree, child)


    def _do_extend(
        self,
        tree: PrefixTree,
        leaf: TreeNode,
        logits: torch.Tensor,
    ):
        """Greedy-extend *leaf* by one token."""
        log_probs = F.log_softmax(logits, dim=-1)
        best_id = torch.argmax(log_probs).item()
        best_lp = log_probs[best_id].item()

        tree.extend_leaf(leaf, best_id, best_lp)

        self._handle_eos(tree, leaf)
    # ------------------------------------------------------------------
    #  Entropy-gated branching with soft-exploration warmup
    # ------------------------------------------------------------------

    def _soft_explore_factor(self, step: int) -> float:
        """
        Returns a multiplier in [soft_explore_initial, 1.0] that linearly
        ramps from a permissive value to 1.0 over ``soft_explore_window``
        steps.  During early steps the effective threshold is lower,
        making branching easier (soft exploration) without the memory
        explosion of unconditional fan-out.
        """
        if step >= self.soft_explore_window:
            return 1.0
        # Linear ramp: initial → 1.0
        t = step / max(self.soft_explore_window, 1)
        return self.soft_explore_initial + (1.0 - self.soft_explore_initial) * t

    def _step_leaf(
        self,
        tree: PrefixTree,
        leaf: TreeNode,
        logits: torch.Tensor,
        step: int,
        num_active: int,
    ):
        """Decide whether to branch or greedy-extend *leaf*."""
        entropy = compute_entropy(logits, self.temperature)
        self.entropy_trace.append((step, leaf.node_id, entropy))

        # Update the sequence's specific entropy EMA
        if leaf.entropy_ema is None:
            leaf.entropy_ema = entropy
        else:
            leaf.entropy_ema = (self.entropy_ema_alpha * entropy) + ((1.0 - self.entropy_ema_alpha) * leaf.entropy_ema)

        # Calculate the dynamic multiplier based on tree capacity
        dynamic_multiplier = adaptive_threshold(self.relative_entropy_multiplier, self.M, num_active)

        # Apply soft-exploration warmup: early steps scale the threshold
        # down so branching is easier (more permissive), then ramp to
        # full strictness over soft_explore_window steps.
        warmup = self._soft_explore_factor(step)
        
        # The threshold is EMA × capacity multiplier × warmup factor
        tau = leaf.entropy_ema * dynamic_multiplier * warmup

        seq_so_far = leaf.get_full_sequence()
        penalized = apply_repetition_penalty(
            logits, seq_so_far,
            penalty=self.rep_penalty
        )

        headroom = self.M - num_active
        can_branch = (entropy > tau) and (headroom >= self.K)

        if can_branch:
            self._do_branch(tree, leaf, penalized, step)
        else:
            self._do_extend(tree, leaf, penalized)

    def _update_semantic_vectors(
        self,
        leaves: list[TreeNode],
        hidden_states: torch.Tensor,
    ):
        """
        Update each leaf's semantic vector using EMA of the new hidden state.

        All arithmetic stays on GPU — no .item() or .tolist() calls.

        Args:
            leaves: list of N active leaves (same order as the batch).
            hidden_states: (N, D) final hidden states from decode_batch,
                           already on GPU.
        """
        alpha = self.ema_alpha
        for i, leaf in enumerate(leaves):
            h = hidden_states[i]  # (D,) — stays on device
            if leaf.semantic_vector is None:
                leaf.semantic_vector = h.clone()
            else:
                # EMA: v_new = α * h + (1 − α) * v_old
                leaf.semantic_vector.mul_(1.0 - alpha).add_(h, alpha=alpha)

    def _prune_similar(self, tree: PrefixTree, step: int):
        """
        Pairwise cosine-similarity diversity pruning.
 
        Stacks all active-leaf semantic vectors into an (N, D) matrix,
        computes the full (N, N) cosine similarity on GPU in one
        batched operation, and prunes the lower-probability branch
        from any pair exceeding the similarity threshold.

        Guards:
          1. A leaf must have lived for at least ``min_steps_before_prune``
             decode steps so its EMA semantic vector has diverged from the
             parent's cloned vector.  With ema_alpha=0.3, after 3 steps the
             parent contribution is 0.7^3 ≈ 34 % — enough for siblings to
             have differentiated.
          2. Two siblings that share the same parent (i.e. were born from
             the same branch point) are never compared against each other,
             because their vectors start identical and need extra time to
             diverge.
        """
        leaves = tree.get_active_leaves()
        N = len(leaves)
        if N <= 1:
            return
 
        # Filter to leaves whose EMA has had enough steps to warm up.
        min_age = self.min_steps_before_prune
        sv_leaves: list[TreeNode] = []
        sv_list: list[torch.Tensor] = []
        for leaf in leaves:
            age = step - leaf.creation_step
            if leaf.semantic_vector is not None and age >= min_age:
                sv_leaves.append(leaf)
                sv_list.append(leaf.semantic_vector)
 
        M = len(sv_leaves)
        if M <= 1:
            return
 
        # (M, D) — stack on GPU, no copies to CPU
        sv_matrix = torch.stack(sv_list, dim=0)  # already on device
 
        # Batched pairwise cosine similarity: O(M^2) on GPU
        # F.normalize + mm is a single fused path on CUDA.
        sv_normed = F.normalize(sv_matrix, p=2, dim=1)  # (M, D)
        sim_matrix = torch.mm(sv_normed, sv_normed.t())  # (M, M)
 
        # Mask the diagonal and lower triangle so each pair is only
        # considered once (and a leaf is never compared to itself).
        mask = torch.triu(torch.ones(M, M, device=sim_matrix.device, dtype=torch.bool), diagonal=1)

        # Also mask out sibling pairs (same parent) — their vectors
        # were cloned from the same source and need extra time to diverge.
        for i in range(M):
            for j in range(i + 1, M):
                if (sv_leaves[i].parent is not None
                        and sv_leaves[i].parent is sv_leaves[j].parent):
                    mask[i, j] = False

        sim_matrix = sim_matrix * mask
 
        # Find pairs above threshold — GPU comparison, then a small
        # transfer of the indices (typically very few pairs).
        above = (sim_matrix > self.semantic_similarity_threshold).nonzero(as_tuple=False)
        # above: (P, 2) where P is number of violating pairs
 
        if above.shape[0] == 0:
            return
 
        # Move only the small index tensor to CPU for the prune logic
        pairs = above.tolist()
 
        # Track which leaves are already marked for pruning this step
        to_prune: set[int] = set()
        for i_idx, j_idx in pairs:
            leaf_i = sv_leaves[i_idx]
            leaf_j = sv_leaves[j_idx]
 
            # Skip if either was already pruned this step
            if leaf_i.node_id in to_prune or leaf_j.node_id in to_prune:
                continue
 
            # Keep the branch with higher cumulative log-prob (raw, not depth-normalised —
            # consistent with _prune and avoids depth-bias on recently branched children).
            if leaf_i.cumulative_log_prob >= leaf_j.cumulative_log_prob:
                to_prune.add(leaf_j.node_id)
            else:
                to_prune.add(leaf_i.node_id)
 
        # Execute pruning
        pruned = 0
        for leaf in sv_leaves:
            if leaf.node_id in to_prune and leaf.is_active:
                tree.prune_branch(leaf)
                pruned += 1
 
        if pruned:
            self.diversity_pruning_events.append((step, pruned))

    # ------------------------------------------------------------------
    #  Diagnostics
    # ------------------------------------------------------------------

    def get_diagnostics(self) -> dict:
        # ---- Throughput ----
        # decode_tps: tokens emitted per wall-second during the decode loop.
        # Each forward pass emits one token per active branch, so this is the
        # total aggregate throughput (sum across all branches).
        decode_tps = (
            self._total_tokens_decoded / self._decode_time_s
            if self._decode_time_s > 0 else 0.0
        )
        prefill_tps = (
            self._prefill_tokens / self._prefill_time_s
            if self._prefill_time_s > 0 else 0.0
        )

        # ---- VRAM ----
        kv_stats = self.kv_cache_mgr.get_stats()

        return {
            # --- branch / prune diagnostics ---
            'num_branch_points': len(self.branch_points),
            'total_pruned': sum(n for _, n in self.pruning_events),
            'total_diversity_pruned': sum(n for _, n in self.diversity_pruning_events),
            'branch_points': self.branch_points,
            'pruning_events': self.pruning_events,
            'diversity_pruning_events': self.diversity_pruning_events,
            'entropy_trace': self.entropy_trace,

            # --- active-branch evolution (length == number of decode steps taken) ---
            # active_branch_trace[i] is the number of live branches at step i.
            # Step 0 is always 1 (single root sequence after prefill).
            'active_branch_trace': self.active_branch_trace,

            # --- throughput ---
            'prefill_tokens': self._prefill_tokens,
            'prefill_time_s': round(self._prefill_time_s, 4),
            'prefill_throughput_tps': round(prefill_tps, 2),
            'decode_tokens_total': self._total_tokens_decoded,
            'decode_time_s': round(self._decode_time_s, 4),
            'decode_throughput_tps': round(decode_tps, 2),

            # --- VRAM ---
            # peak_vram_bytes: peak CUDA memory allocated across both prefill
            #   and decode (weights + activations + KV cache).  0 on CPU.
            'peak_vram_bytes': self._peak_vram_bytes,
            'peak_vram_mb': round(self._peak_vram_bytes / 1024 ** 2, 2),
            # peak_kv_cache_bytes: high-water mark for KV-cache pages only
            #   (derived from peak block count × bytes-per-block).
            'peak_kv_cache_bytes': kv_stats['peak_kv_cache_bytes'],
            'peak_kv_cache_mb': round(kv_stats['peak_kv_cache_bytes'] / 1024 ** 2, 2),
            'kv_blocks_peak': kv_stats['blocks_peak'],
            'kv_blocks_total': kv_stats['blocks_total'],
            'kv_peak_utilization': round(kv_stats['peak_utilization'], 4),
        }