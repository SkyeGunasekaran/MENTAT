from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

# ---------------------------------------------------------------------------
# 1. Import the model-agnostic Paged Attention Injector
# ---------------------------------------------------------------------------
from paged_attention_injector import inject_paged_attention

# ---------------------------------------------------------------------------
# 2. Import the paged UQ generator
# ---------------------------------------------------------------------------
from generator import PagedPrefixTreeUQGenerator

# ---------------------------------------------------------------------------
# 3. Default prompts
# ---------------------------------------------------------------------------
DEFAULT_PROMPTS = [
    "The meaning of life is",
    "In a shocking turn of events, scientists discovered that",
    "The best way to learn a new programming language is to",
    "Once upon a time, in a kingdom far away,",
    "The key difference between classical and quantum computing is",
]

# ---------------------------------------------------------------------------
# 4. Pretty-printing helpers
# ---------------------------------------------------------------------------

def print_header(text: str, width: int = 80):
    print("\n" + "=" * width)
    print(f"  {text}")
    print("=" * width)


def print_results_for_prompt(prompt: str, results: list[dict], diagnostics: dict):
    print_header(f'PROMPT: "{prompt}"')

    n_branch = diagnostics['num_branch_points']
    n_div_pruned = diagnostics.get('total_diversity_pruned', 0)
    n_explore_bp = diagnostics.get('exploration_branch_points', 0)
    n_converge_bp = diagnostics.get('convergence_branch_points', 0)
    print(f"  Branch points: {n_branch} (explore: {n_explore_bp}, converge: {n_converge_bp})")
    print(f"  Pruned: {n_div_pruned}")
    print(f"  Sequences returned: {len(results)}")

    if diagnostics['branch_points']:
        entropies = [e for _, e, _ in diagnostics['branch_points']]
        print(f"  Branch entropies: min={min(entropies):.2f}  "
              f"max={max(entropies):.2f}  mean={sum(entropies)/len(entropies):.2f}")

    # ---- Performance metrics ----
    print(f"\n  --- Throughput ---")
    print(f"  Prefill:  {diagnostics['prefill_tokens']} tokens  "
          f"in {diagnostics['prefill_time_s']:.3f}s  "
          f"({diagnostics['prefill_throughput_tps']:.1f} tok/s)")
    print(f"  Decode:   {diagnostics['decode_tokens_total']} tokens (aggregate across branches)  "
          f"in {diagnostics['decode_time_s']:.3f}s  "
          f"({diagnostics['decode_throughput_tps']:.1f} tok/s)")

    # ---- VRAM metrics ----
    print(f"\n  --- Memory ---")
    if diagnostics['peak_vram_bytes'] > 0:
        print(f"  Peak VRAM (total):    {diagnostics['peak_vram_mb']:.1f} MB")
    else:
        print(f"  Peak VRAM (total):    N/A (CPU run)")
    print(f"  Peak KV-cache:        {diagnostics['peak_kv_cache_mb']:.1f} MB  "
          f"({diagnostics['kv_blocks_peak']}/{diagnostics['kv_blocks_total']} blocks, "
          f"{diagnostics['kv_peak_utilization']:.1%} utilization)")

    # ---- Active-branch trace (compact) ----
    trace = diagnostics.get('active_branch_trace', [])
    if trace:
        peak_branches = max(trace)
        mean_branches = sum(trace) / len(trace)
        print(f"\n  --- Branch Evolution (over {len(trace)} steps) ---")
        print(f"  Peak active branches: {peak_branches}  |  Mean: {mean_branches:.1f}")
        # Print a compact ASCII sparkline (width ≤ 60 chars)
        width = min(len(trace), 60)
        step = max(1, len(trace) // width)
        sampled = trace[::step][:width]
        bar_chars = " ▁▂▃▄▅▆▇█"
        scaled = [int(v / max(peak_branches, 1) * (len(bar_chars) - 1)) for v in sampled]
        sparkline = "".join(bar_chars[s] for s in scaled)
        print(f"  Active branches:  [{sparkline}]")

    for i, res in enumerate(results):
        tag = "COMPLETE" if res.get('complete', True) else "PARTIAL"
        pct = res['norm_prob'] * 100
        
        # Update the print statement to show the percentage
        print(f"\n  [{i+1}] ({tag})  prob={pct:.2f}%  (log_prob={res['log_prob']:.3f})")
        text = res['text']
        print(f"      {text}")
    print()


# ---------------------------------------------------------------------------
# 5. Serialization helpers
# ---------------------------------------------------------------------------

def serialize_results(
    prompt: str,
    results: list[dict],
    diagnostics: dict,
    elapsed: float,
) -> dict:
    def safe_float(x):
        if isinstance(x, float) and (math.isnan(x) or math.isinf(x)):
            return None
        return round(x, 4) if isinstance(x, float) else x
    
    return {
        'prompt': prompt,
        'elapsed_seconds': round(elapsed, 3),
        'num_sequences': len(results),
        'sequences': [
            {
                'text': r['text'],
                'log_prob': safe_float(r['log_prob']),
                'avg_log_prob': r['log_prob'] / max(len(r['token_ids']), 1),
                'normalized_prob': safe_float(r['norm_prob']), # Save to JSON
                'normalized_prob_pct': safe_float(r['norm_prob'] * 100), # Optional: save as strict %
                'num_tokens': len(r['token_ids']),
            }
            for r in results
        ],
        'performance': {
            # throughput
            'prefill_tokens': diagnostics['prefill_tokens'],
            'prefill_time_s': diagnostics['prefill_time_s'],
            'prefill_throughput_tps': diagnostics['prefill_throughput_tps'],
            'decode_tokens_total': diagnostics['decode_tokens_total'],
            'decode_time_s': diagnostics['decode_time_s'],
            'decode_throughput_tps': diagnostics['decode_throughput_tps'],
            # VRAM
            'peak_vram_bytes': diagnostics['peak_vram_bytes'],
            'peak_vram_mb': diagnostics['peak_vram_mb'],
            'peak_kv_cache_bytes': diagnostics['peak_kv_cache_bytes'],
            'peak_kv_cache_mb': diagnostics['peak_kv_cache_mb'],
            'kv_blocks_peak': diagnostics['kv_blocks_peak'],
            'kv_blocks_total': diagnostics['kv_blocks_total'],
            'kv_peak_utilization': diagnostics['kv_peak_utilization'],
            # branch evolution
            'active_branch_trace': diagnostics.get('active_branch_trace', []),
        },
        'diagnostics': {
            'num_branch_points': diagnostics['num_branch_points'],
            'exploration_branch_points': diagnostics.get('exploration_branch_points', 0),
            'convergence_branch_points': diagnostics.get('convergence_branch_points', 0),
            'total_diversity_pruned': diagnostics.get('total_diversity_pruned', 0),
            'branch_points': [
                {'step': s, 'entropy': safe_float(e), 'num_children': k}
                for s, e, k in diagnostics['branch_points']
            ],
            'pruning_events': [
                {'step': s, 'count': c}
                for s, c in diagnostics['pruning_events']
            ],
            'diversity_pruning_events': [
                {'step': s, 'count': c}
                for s, c in diagnostics.get('diversity_pruning_events', [])
            ],
            'entropy_trace': [
                {'step': s, 'node_id': n, 'entropy': safe_float(e)}
                for s, n, e in diagnostics['entropy_trace']
            ],
        },
    }

# ---------------------------------------------------------------------------
# 6. Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Paged Batched Prefix-Tree UQ Generator"
    )
    parser.add_argument("--model_path", type=str, required=True,
                        help="Path to pretrained model directory")
    parser.add_argument("--tokenizer_path", type=str, default=None,
                        help="Path to tokenizer (defaults to model_path)")
    parser.add_argument("--device", type=str, default="auto",
                        help="Device: 'auto', 'cuda', 'cpu'")
    parser.add_argument("--dtype", type=str, default="bfloat16",
                        choices=["float32", "float16", "bfloat16"],
                        help="Model dtype")

    # UQ parameters
    parser.add_argument("--max_new_tokens", type=int, default=256)
    parser.add_argument("--max_active", type=int, default=25)
    parser.add_argument("--branching_factor", type=int, default=3)
    parser.add_argument("--entropy_ema_alpha", type=float, default=0.2)
    parser.add_argument("--relative_entropy_multiplier", type=float, default=1.25)
    parser.add_argument("--temperature", type=float, default=0.7)

    # Repetition penalty
    parser.add_argument("--rep_penalty", type=float, default=1.2)
    parser.add_argument("--freq_penalty", type=float, default=0.3)
    parser.add_argument("--ngram_block", type=int, default=3)

    # Semantic diversity pruning
    parser.add_argument("--sim_threshold", type=float, default=0.90)

    # Exploration window
    parser.add_argument("--exploration_window", type=int, default=15)
    parser.add_argument("--exploration_percentile", type=float, default=0.99)

    # Paged cache config
    parser.add_argument("--block_size", type=int, default=16,
                        help="Tokens per KV cache block (default: 16)")
    parser.add_argument("--max_blocks", type=int, default=8192,
                        help="Max blocks in KV pool (auto-sized if omitted)")

    # Prompts & output
    parser.add_argument("--prompts", type=str, nargs="+", default=None)
    parser.add_argument("--output_json", type=str, default="uq_results.json")

    args = parser.parse_args()

    # ---- resolve device / dtype ----
    dtype_map = {
        "float32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
    }
    dtype = dtype_map[args.dtype]

    if args.device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = args.device

    prompts = args.prompts or DEFAULT_PROMPTS
    tokenizer_path = args.tokenizer_path or args.model_path

    # ---- load model ----
    print_header("Loading model")
    print(f"  Model path:  {args.model_path}")
    print(f"  Device:      {device}")
    print(f"  Dtype:       {args.dtype}")

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # We now load standard HF models directly without custom AutoModel registrations
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        torch_dtype=dtype,
        trust_remote_code=True,
    ).to(device).eval()

    num_params = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"  Parameters:  {num_params:.1f}M")
    print(f"  Vocab size:  {model.config.vocab_size}")
    print(f"  Layers:      {model.config.num_hidden_layers}")
    print(f"  Hidden:      {model.config.hidden_size}")

    # ---- INJECT PAGED ATTENTION ----
    print("  Injecting model-agnostic paged attention...")
    inject_paged_attention(model)

    # ---- UQ parameters summary ----
    print_header("UQ Configuration (Paged Batched)")
    print(f"  max_new_tokens:    {args.max_new_tokens}")
    print(f"  max_active:        {args.max_active}")
    print(f"  branching_factor:  {args.branching_factor}")
    print(f"  relative_entropy_multiplier:{args.relative_entropy_multiplier}")
    print(f"  entropy_ema_alpha: {args.entropy_ema_alpha}")
    print(f"  temperature:       {args.temperature}")
    print(f"  rep_penalty:       {args.rep_penalty}")
    print(f"  freq_penalty:      {args.freq_penalty}")
    print(f"  ngram_block:       {args.ngram_block}")
    print(f"  sim_threshold:     {args.sim_threshold}")
    print(f"  explore_window:    {args.exploration_window}")
    print(f"  explore_pctile:    {args.exploration_percentile}")
    print(f"  block_size:        {args.block_size}")
    print(f"  max_blocks:        {args.max_blocks or 'auto'}")
    print(f"  num_prompts:       {len(prompts)}")

    # ---- run generation ----
    all_results = []

    for i, prompt in enumerate(prompts):
        print_header(f"Generating [{i+1}/{len(prompts)}]")
        print(f'  Prompt: "{prompt}"')

        # Notice how we just pass the standard model, but under the hood, 
        # the generator will now be calling the newly injected paged methods!
        gen = PagedPrefixTreeUQGenerator(
            model=model,
            tokenizer=tokenizer,
            max_active_branches=args.max_active,
            branching_factor=args.branching_factor,
            relative_entropy_multiplier=args.relative_entropy_multiplier,
            entropy_ema_alpha=args.entropy_ema_alpha,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            repetition_penalty=args.rep_penalty,
            frequency_penalty=args.freq_penalty,
            max_ngram_block=args.ngram_block,
            block_size=args.block_size,
            max_blocks=args.max_blocks,
            semantic_similarity_threshold=args.sim_threshold,
            exploration_window=args.exploration_window,
            exploration_percentile=args.exploration_percentile,
        )

        t0 = time.time()
        results = gen.generate(prompt)
        elapsed = time.time() - t0

        diag = gen.get_diagnostics()

        # Print paged cache stats
        stats = gen.kv_cache_mgr.get_stats()
        print(f"  Done in {elapsed:.2f}s")
        print(f"  Peak KV-cache utilization: ({stats['peak_utilization']:.2%}) ")

        # Tag completeness
        for r in results:
            eos_id = tokenizer.eos_token_id
            if isinstance(eos_id, list):
                r['complete'] = any(
                    r['token_ids'][-1] == e for e in eos_id
                ) if r['token_ids'] else False
            else:
                r['complete'] = (
                    r['token_ids'][-1] == eos_id
                ) if r['token_ids'] else False

        print_results_for_prompt(prompt, results, diag)

        serialized = serialize_results(prompt, results, diag, elapsed)
        all_results.append(serialized)

    # ---- save JSON ----
    output_path = Path(args.output_json)
    with open(output_path, 'w') as f:
        json.dump({
            'config': {
                'model_path': args.model_path,
                'max_new_tokens': args.max_new_tokens,
                'max_active': args.max_active,
                'branching_factor': args.branching_factor,
                'entropy_ema_alpha': args.entropy_ema_alpha,
                'relative_entropy_multiplier': args.relative_entropy_multiplier,
                'temperature': args.temperature,
                'rep_penalty': args.rep_penalty,
                'freq_penalty': args.freq_penalty,
                'ngram_block': args.ngram_block,
                'sim_threshold': args.sim_threshold,
                'exploration_window': args.exploration_window,
                'exploration_percentile': args.exploration_percentile,
                'block_size': args.block_size,
                'max_blocks': args.max_blocks,
                'backend': 'paged_batched',
            },
            'prompts': all_results,
        }, f, indent=2)

    print_header("Done")
    print(f"  Results saved to: {output_path.resolve()}")
    print(f"  Total prompts:    {len(all_results)}")
    total_seqs = sum(r['num_sequences'] for r in all_results)
    total_time = sum(r['elapsed_seconds'] for r in all_results)
    total_decode_tokens = sum(r['performance']['decode_tokens_total'] for r in all_results)
    total_decode_time = sum(r['performance']['decode_time_s'] for r in all_results)
    total_prefill_tokens = sum(r['performance']['prefill_tokens'] for r in all_results)
    total_prefill_time = sum(r['performance']['prefill_time_s'] for r in all_results)
    print(f"  Total sequences:  {total_seqs}")
    print(f"  Total time:       {total_time:.2f}s")
    if total_decode_time > 0:
        print(f"  Aggregate decode throughput: "
              f"{total_decode_tokens / total_decode_time:.1f} tok/s "
              f"({total_decode_tokens} tokens in {total_decode_time:.2f}s)")
    if total_prefill_time > 0:
        print(f"  Aggregate prefill throughput: "
              f"{total_prefill_tokens / total_prefill_time:.1f} tok/s "
              f"({total_prefill_tokens} tokens in {total_prefill_time:.2f}s)")


if __name__ == '__main__':
    main()