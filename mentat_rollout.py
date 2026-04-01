"""
Usage:

    python rollout_uq.py \
        --model meta-llama/Llama-3.2-1B \
        --prompt "The meaning of life is" \
        --max-active 10 \
        --branching-factor 2 \
        --chat-template auto \
        ...
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

from adapters.adapter_factory import get_adapter
from core.generator import MentatGenerator
from utils.visualize_tree import export_tree_visualization
from utils.shared import (
    extract_answer,
    resolve_chat_template,
    format_prompt_from_messages,
    safe_float
)


# generation stuff

def run_mentat_rollout(
    model,
    tokenizer,
    adapter,
    prompt: str,
    args: argparse.Namespace,
) -> dict:
    """Run Mentat prefix-tree generation on a single prompt."""

    # 1. Resolve and format the prompt
    resolved_tmpl = resolve_chat_template(tokenizer, args.chat_template)
    if resolved_tmpl:
        formatted = format_prompt_from_messages(
            tokenizer, resolved_tmpl, [{"role": "user", "content": prompt}]
        )
    else:
        formatted = prompt

    inputs = tokenizer(formatted, return_tensors="pt").to(model.device)
    input_len = inputs["input_ids"].shape[1]

    # 2. Initialize the generator
    gen = MentatGenerator(
        model,
        tokenizer,
        adapter=adapter,
        max_active_branches=args.max_active,
        branching_factor=args.branching_factor,
        relative_entropy_multiplier=args.relative_entropy_multiplier,
        entropy_ema_alpha=args.entropy_ema_alpha,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        repetition_penalty=args.repetition_penalty,
        block_size=args.block_size,
        max_blocks=args.max_blocks,
        semantic_similarity_threshold=args.sim_threshold,
        ema_alpha=args.ema_alpha,
        soft_explore_window=args.soft_explore_window,
        soft_explore_initial=args.soft_explore_initial,
    )

    # 3. Generate
    t0 = time.perf_counter()
    results = gen.generate(formatted)
    elapsed = time.perf_counter() - t0

    diag = gen.get_diagnostics()

    # 4. Process Sequences & Extract Answers
    sequences = []
    answer_counts: dict[str, int] = {}
    eos_id = tokenizer.eos_token_id

    for r in results:
        tids = r['token_ids']
        text = r['text']
        length = max(len(tids), 1)

        # Check completion status
        if isinstance(eos_id, (list, tuple)):
            complete = any(tids[-1] == e for e in eos_id) if tids else False
        else:
            complete = (tids[-1] == eos_id) if tids else False

        extracted = extract_answer(text)
        if extracted is not None:
            answer_counts[extracted] = answer_counts.get(extracted, 0) + 1

        sequences.append({
            "text": text,
            "token_ids": tids,
            "log_prob": safe_float(r['log_prob']),
            "avg_log_prob": safe_float(r['log_prob'] / length),
            "num_tokens": len(tids),
            "finish_reason": "stop" if complete else "length",
            "extracted_answer": extracted,
        })

    # 5. Compute Summaries
    n_with_answer = sum(1 for s in sequences if s["extracted_answer"] is not None)
    majority_answer = None
    agreement_ratio = 0.0
    if answer_counts:
        majority_answer = max(answer_counts, key=answer_counts.get)
        agreement_ratio = answer_counts[majority_answer] / max(n_with_answer, 1)

    return {
        "method": "mentat_uq",
        "prompt": prompt,
        "num_sequences": len(sequences),
        "sequences": sequences,
        "answer_summary": {
            "distinct_answers": len(answer_counts),
            "majority_answer": majority_answer,
            "agreement_ratio": round(agreement_ratio, 4),
            "answer_distribution": answer_counts,
        },
        "elapsed_seconds": round(elapsed, 4),
        "performance": {
            "prefill_tokens": diag['prefill_tokens'],
            "prefill_time_s": diag['prefill_time_s'],
            "prefill_throughput_tps": diag['prefill_throughput_tps'],
            "decode_tokens_total": diag['decode_tokens_total'],
            "decode_time_s": diag['decode_time_s'],
            "decode_throughput_tps": diag['decode_throughput_tps'],
            "peak_vram_mb": diag['peak_vram_mb'],
            "peak_kv_cache_mb": diag['peak_kv_cache_mb'],
            "active_branch_trace": diag.get('active_branch_trace', []),
        },
        "diagnostics": {
            "num_branch_points": diag['num_branch_points'],
            "total_diversity_pruned": diag.get('total_diversity_pruned', 0),
            "branch_points": [
                {'step': s, 'entropy': safe_float(e), 'num_children': k}
                for s, e, k in diag['branch_points']
            ]
        },
        "usage": {
            "prompt_tokens": input_len,
            "completion_tokens": diag['decode_tokens_total'],
        },
        "generator_ref": gen  # Temporarily attached for PyVis export if needed
    }


# look cool

def print_result(result: dict):
    print(f"\n{'=' * 72}")
    print(f"  PROMPT: \"{result['prompt']}\"")
    print(f"  Method: Mentat UQ | Time: {result['elapsed_seconds']:.2f}s | Seqs: {result['num_sequences']}")
    print(f"{'=' * 72}")

    summary = result["answer_summary"]
    if summary["majority_answer"]:
        print(f"  Majority Answer: \"{summary['majority_answer']}\" ({summary['agreement_ratio']:.0%})")

    diag = result["diagnostics"]
    perf = result["performance"]
    print(f"  Branch points: {diag['num_branch_points']} | Pruned: {diag['total_diversity_pruned']}")
    print(f"  Throughput: {perf['decode_throughput_tps']:.1f} tok/s | Peak KV: {perf['peak_kv_cache_mb']:.1f} MB")

    for i, seq in enumerate(result["sequences"]):
        tag = seq["finish_reason"].upper()
        ans = f"  → {seq['extracted_answer']}" if seq["extracted_answer"] else ""
        print(f"\n  [{i + 1}] ({tag}) log_p={seq['log_prob']:.2f} | tokens={seq['num_tokens']}{ans}")
        print(f"      {seq['text'][:200]}..." if len(seq['text']) > 200 else f"      {seq['text']}")


# ---------------------------------------------------------------------------
# CLI (Merged baseline arguments + Mentat arguments)
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="RLVR Rollout UQ (Mentat)")

    model_group = parser.add_argument_group("Model")
    model_group.add_argument("--model", type=str, required=True, help="Path or HF repo")
    model_group.add_argument("--tokenizer", type=str, default=None, help="Tokenizer (defaults to model)")
    model_group.add_argument("--dtype", type=str, default="bfloat16", choices=["float32", "float16", "bfloat16"])
    model_group.add_argument("--device", type=str, default="auto")
    model_group.add_argument("--trust-remote-code", action="store_true")
    model_group.add_argument("--chat-template", type=str, default="none")

    gen_group = parser.add_argument_group("Generation Defaults")
    gen_group.add_argument("--max-new-tokens", type=int, default=512)
    gen_group.add_argument("--temperature", type=float, default=0.7)
    gen_group.add_argument("--repetition-penalty", type=float, default=1.0)

    uq_group = parser.add_argument_group("Mentat UQ Parameters")
    uq_group.add_argument("--max-active", type=int, default=10, help="Max active branches")
    uq_group.add_argument("--branching-factor", type=int, default=2, help="Children per branch point")
    uq_group.add_argument("--entropy-ema-alpha", type=float, default=0.2)
    uq_group.add_argument("--relative-entropy-multiplier", type=float, default=1.15)
    
    prune_group = parser.add_argument_group("Semantic Diversity Pruning")
    prune_group.add_argument("--sim-threshold", type=float, default=0.75)
    prune_group.add_argument("--ema-alpha", type=float, default=0.25)
    prune_group.add_argument("--soft-explore-window", type=int, default=15)
    prune_group.add_argument("--soft-explore-initial", type=float, default=0.3)

    cache_group = parser.add_argument_group("Paged KV Cache")
    cache_group.add_argument("--block-size", type=int, default=16)
    cache_group.add_argument("--max-blocks", type=int, default=8192)

    io_group = parser.add_argument_group("Input / Output")
    io_group.add_argument("--prompt", type=str, nargs="+")
    io_group.add_argument("--prompt-file", type=str)
    io_group.add_argument("--output-json", type=str)
    io_group.add_argument("--export-visuals", action="store_true", help="Export PyVis HTML trees")

    args = parser.parse_args()
    if args.tokenizer is None:
        args.tokenizer = args.model
    return args


def main():
    args = parse_args()

    # Resolve prompts
    prompts = []
    if args.prompt: prompts.extend(args.prompt)
    if args.prompt_file:
        prompts.extend(line.strip() for line in Path(args.prompt_file).read_text().splitlines() if line.strip())
    if not prompts: prompts = ["What is 2+2?"]

    # Load model
    dtype_map = {"float32": torch.float32, "float16": torch.float16, "bfloat16": torch.bfloat16}
    dtype = dtype_map[args.dtype]
    device = args.device if args.device != "auto" else ("cuda" if torch.cuda.is_available() else "cpu")

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, trust_remote_code=args.trust_remote_code)
    if tokenizer.pad_token is None: tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=dtype, trust_remote_code=args.trust_remote_code
    ).to(device).eval()

    adapter = get_adapter(model)

    all_results = []
    for i, prompt in enumerate(prompts):
        print(f"[{i + 1}/{len(prompts)}] Generating UQ Rollout...")
        
        result = run_mentat_rollout(model, tokenizer, adapter, prompt, args)
        
        # Optionally export PyVis interactive graph
        if args.export_visuals:
            safe_prompt_name = f"tree_prompt_{i}.html"
            export_tree_visualization(result["generator_ref"].tree, tokenizer, output_path=safe_prompt_name)
        
        # Clean up the generator ref before serialization
        del result["generator_ref"]

        print_result(result)
        all_results.append(result)

    if args.output_json:
        # Strip out the non-serializable args namespace into a dict
        safe_config = {k: v for k, v in vars(args).items() if not k.startswith('_')}
        with open(args.output_json, "w") as f:
            json.dump({"config": safe_config, "results": all_results}, f, indent=2)

if __name__ == "__main__":
    main()