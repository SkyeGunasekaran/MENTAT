"""
DPO Candidate Generation Pipeline
===================================

Loads prompts from a dataset (HuggingFace Hub, local JSONL, or local JSON),
runs each through the prefix-tree UQ generator sequentially, and writes
structured JSONL output with all N candidate completions per prompt.

Output schema (one JSON object per line in the .jsonl):
{
    "prompt_id": 0,
    "prompt": "...",
    "source_dataset": "...",
    "num_completions": 9,
    "completions": [
        {
            "rank": 0,
            "text": "...",
            "log_prob": -49.07,
            "avg_log_prob": -0.383,
            "norm_prob": 0.0312,
            "num_tokens": 128,
            "is_complete": false
        },
        ...
    ],
    "generation_meta": {
        "elapsed_seconds": 5.89,
        "prefill_time_s": 1.065,
        "decode_time_s": 7.355,
        "decode_throughput_tps": 133.5,
        "peak_vram_mb": 20480.9,
        "peak_kv_cache_mb": 2744.0,
        "num_branch_points": 44,
        "total_diversity_pruned": 36,
        "peak_active_branches": 9,
        "mean_active_branches": 7.7
    }
}

Usage examples:

    # From HuggingFace Hub (auto-detects prompt column)
    python generate_dpo_candidates.py \\
        --model_path /models/my_llm \\
        --dataset Anthropic/hh-rlhf \\
        --split train \\
        --prompt_column chosen \\
        --max_prompts 500 \\
        --output dpo_candidates.jsonl

    # From local JSONL file
    python generate_dpo_candidates.py \\
        --model_path /models/my_llm \\
        --dataset ./my_prompts.jsonl \\
        --prompt_column prompt \\
        --output dpo_candidates.jsonl

    # From local JSON file (list of objects or {"prompts": [...]})
    python generate_dpo_candidates.py \\
        --model_path /models/my_llm \\
        --dataset ./prompts.json \\
        --prompt_column text \\
        --output dpo_candidates.jsonl
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

from adapters.adapter_factory import get_adapter
from generator import PagedPrefixTreeUQGenerator


# ---------------------------------------------------------------------------
#  Dataset loading
# ---------------------------------------------------------------------------

def _combine_prompt(instruction: str, input_text: str,
                    template: str) -> str:
    """
    Merge an instruction and an optional input field into a single prompt
    string.  If *input_text* is empty (or whitespace-only), only the
    instruction is used.

    The ``template`` uses ``{instruction}`` and ``{input}`` placeholders:
        "{instruction}\n\n{input}"          ← default
        "### Instruction:\n{instruction}\n### Input:\n{input}\n### Response:\n"
    """
    input_text = input_text.strip()
    if not input_text:
        return instruction.strip()
    return template.format(instruction=instruction.strip(),
                           input=input_text)


def _load_from_huggingface(dataset_name: str, split: str, prompt_column: str,
                           input_column: str | None, prompt_template: str,
                           max_prompts: int | None) -> list[str]:
    """Load prompts from a HuggingFace dataset."""
    try:
        from datasets import load_dataset
    except ImportError:
        print("ERROR: `datasets` library required for HuggingFace datasets.")
        print("       Install with: pip install datasets")
        sys.exit(1)

    print(f"  Loading HuggingFace dataset: {dataset_name} (split={split})")
    ds = load_dataset(dataset_name, split=split)

    if prompt_column not in ds.column_names:
        available = ", ".join(ds.column_names)
        print(f"  ERROR: Column '{prompt_column}' not found.")
        print(f"         Available columns: {available}")
        sys.exit(1)

    if input_column and input_column not in ds.column_names:
        available = ", ".join(ds.column_names)
        print(f"  ERROR: Input column '{input_column}' not found.")
        print(f"         Available columns: {available}")
        sys.exit(1)

    n = len(ds) if max_prompts is None else min(max_prompts, len(ds))
    prompts: list[str] = []
    for i in range(n):
        instruction = str(ds[i][prompt_column])
        input_text = str(ds[i][input_column]) if input_column else ""
        prompts.append(_combine_prompt(instruction, input_text,
                                       prompt_template))

    return prompts


def _load_from_jsonl(path: Path, prompt_column: str,
                     input_column: str | None, prompt_template: str,
                     max_prompts: int | None) -> list[str]:
    """Load prompts from a local .jsonl file (one JSON object per line)."""
    prompts: list[str] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            if prompt_column not in obj:
                available = ", ".join(obj.keys())
                print(f"  ERROR: Column '{prompt_column}' not found in JSONL row.")
                print(f"         Available keys: {available}")
                sys.exit(1)
            instruction = str(obj[prompt_column])
            input_text = str(obj.get(input_column, "")) if input_column else ""
            prompts.append(_combine_prompt(instruction, input_text,
                                           prompt_template))
            if max_prompts is not None and len(prompts) >= max_prompts:
                break
    return prompts


def _load_from_json(path: Path, prompt_column: str,
                    input_column: str | None, prompt_template: str,
                    max_prompts: int | None) -> list[str]:
    """
    Load prompts from a local .json file.
    Supports two formats:
      - A JSON array of objects: [{"prompt": "..."}, ...]
      - A JSON object with a key containing an array: {"prompts": [{"prompt": "..."}, ...]}
    """
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    # If top-level is a list, use directly
    if isinstance(data, list):
        rows = data
    elif isinstance(data, dict):
        # Try to find the first key whose value is a list
        rows = None
        for key, val in data.items():
            if isinstance(val, list) and len(val) > 0:
                rows = val
                break
        if rows is None:
            print("  ERROR: JSON file must contain a list of objects or a dict "
                  "with a list-valued key.")
            sys.exit(1)
    else:
        print("  ERROR: Unexpected JSON structure.")
        sys.exit(1)

    prompts: list[str] = []
    for obj in rows:
        if isinstance(obj, str):
            # Plain list of strings — no input column possible
            prompts.append(obj)
        elif isinstance(obj, dict):
            if prompt_column not in obj:
                available = ", ".join(obj.keys())
                print(f"  ERROR: Column '{prompt_column}' not found in JSON row.")
                print(f"         Available keys: {available}")
                sys.exit(1)
            instruction = str(obj[prompt_column])
            input_text = str(obj.get(input_column, "")) if input_column else ""
            prompts.append(_combine_prompt(instruction, input_text,
                                           prompt_template))
        else:
            continue
        if max_prompts is not None and len(prompts) >= max_prompts:
            break

    return prompts


def load_prompts(dataset: str, split: str, prompt_column: str,
                 input_column: str | None, prompt_template: str,
                 max_prompts: int | None) -> list[str]:
    """
    Unified loader: detects whether ``dataset`` is a local file path or a
    HuggingFace dataset name and dispatches accordingly.

    When *input_column* is set, each prompt is built by combining the
    instruction (``prompt_column``) with the supplementary context
    (``input_column``) using *prompt_template*.
    """
    path = Path(dataset)

    if path.exists() and path.is_file():
        suffix = path.suffix.lower()
        if suffix == ".jsonl":
            prompts = _load_from_jsonl(path, prompt_column, input_column,
                                       prompt_template, max_prompts)
        elif suffix == ".json":
            prompts = _load_from_json(path, prompt_column, input_column,
                                      prompt_template, max_prompts)
        else:
            # Treat any other text file as one prompt per line
            # (input_column not applicable for plain text files)
            with open(path, "r", encoding="utf-8") as f:
                prompts = [l.strip() for l in f if l.strip()]
            if max_prompts is not None:
                prompts = prompts[:max_prompts]
    else:
        # Assume HuggingFace dataset
        prompts = _load_from_huggingface(dataset, split, prompt_column,
                                         input_column, prompt_template,
                                         max_prompts)

    return prompts


# ---------------------------------------------------------------------------
#  Serialization
# ---------------------------------------------------------------------------

def safe_float(x):
    """Sanitize floats for JSON (NaN/Inf → None)."""
    if isinstance(x, float) and (math.isnan(x) or math.isinf(x)):
        return None
    return round(x, 4) if isinstance(x, float) else x


def serialize_prompt_result(
    prompt_id: int,
    prompt: str,
    source_dataset: str,
    results: list[dict],
    diagnostics: dict,
    elapsed: float,
    eos_token_id,
) -> dict:
    """
    Build a single JSONL record for one prompt's generation output.
    Completions are ordered by norm_prob descending (as returned by the
    generator) and tagged with a rank index.
    """
    completions = []
    for rank, r in enumerate(results):
        # Determine if the completion reached EOS naturally
        is_complete = False
        if r["token_ids"]:
            last_tok = r["token_ids"][-1]
            if isinstance(eos_token_id, (list, tuple)):
                is_complete = last_tok in eos_token_id
            else:
                is_complete = last_tok == eos_token_id

        completions.append({
            "rank": rank,
            "text": r["text"],
            "log_prob": safe_float(r["log_prob"]),
            "avg_log_prob": safe_float(
                r["log_prob"] / max(len(r["token_ids"]), 1)
            ),
            "norm_prob": safe_float(r["norm_prob"]),
            "num_tokens": len(r["token_ids"]),
            "is_complete": is_complete,
        })

    # Compact generation metadata (skip the heavy per-step traces)
    trace = diagnostics.get("active_branch_trace", [])
    peak_branches = max(trace) if trace else 0
    mean_branches = (sum(trace) / len(trace)) if trace else 0

    meta = {
        "elapsed_seconds": round(elapsed, 3),
        "prefill_time_s": diagnostics["prefill_time_s"],
        "decode_time_s": diagnostics["decode_time_s"],
        "decode_throughput_tps": diagnostics["decode_throughput_tps"],
        "peak_vram_mb": diagnostics["peak_vram_mb"],
        "peak_kv_cache_mb": diagnostics["peak_kv_cache_mb"],
        "num_branch_points": diagnostics["num_branch_points"],
        "total_diversity_pruned": diagnostics.get("total_diversity_pruned", 0),
        "peak_active_branches": peak_branches,
        "mean_active_branches": round(mean_branches, 1),
    }

    return {
        "prompt_id": prompt_id,
        "prompt": prompt,
        "source_dataset": source_dataset,
        "num_completions": len(completions),
        "completions": completions,
        "generation_meta": meta,
    }


# ---------------------------------------------------------------------------
#  Progress display
# ---------------------------------------------------------------------------

def print_header(text: str, width: int = 80):
    print("\n" + "=" * width)
    print(f"  {text}")
    print("=" * width)


def print_progress(prompt_id: int, total: int, prompt: str, num_completions: int,
                   elapsed: float, decode_tps: float):
    bar_width = 30
    pct = (prompt_id + 1) / total
    filled = int(bar_width * pct)
    bar = "█" * filled + "░" * (bar_width - filled)
    prompt_preview = prompt[:50] + "..." if len(prompt) > 50 else prompt
    print(f"  [{bar}] {prompt_id + 1}/{total}  "
          f"({num_completions} completions, {elapsed:.1f}s, {decode_tps:.0f} tok/s)  "
          f'"{prompt_preview}"')


# ---------------------------------------------------------------------------
#  Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Generate DPO candidate completions from a prompt dataset",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # ---- Dataset ----
    parser.add_argument("--dataset", type=str, required=True,
                        help="HuggingFace dataset name or local file path "
                             "(.json, .jsonl, or plain text)")
    parser.add_argument("--split", type=str, default="train",
                        help="Dataset split (for HuggingFace datasets)")
    parser.add_argument("--prompt_column", type=str, default="instruction",
                        help="Column/key containing the prompt/instruction text")
    parser.add_argument("--input_column", type=str, default="input",
                        help="Optional column/key with supplementary input context "
                             "(e.g. 'input' for Alpaca). When set, the prompt is "
                             "built by combining prompt_column + input_column via "
                             "--prompt_template. Rows with an empty input field "
                             "use prompt_column alone.")
    parser.add_argument("--prompt_template", type=str,
                        default="{instruction}\n\n{input}",
                        help="Template for combining instruction + input. "
                             "Uses {instruction} and {input} placeholders.")
    parser.add_argument("--max_prompts", type=int, default=None,
                        help="Max number of prompts to process (None = all)")

    # ---- Model ----
    parser.add_argument("--model_path", type=str, required=True,
                        help="Path to pretrained model directory")
    parser.add_argument("--tokenizer_path", type=str, default=None,
                        help="Path to tokenizer (defaults to model_path)")
    parser.add_argument("--device", type=str, default="auto",
                        choices=["auto", "cuda", "cpu"],
                        help="Device for inference")
    parser.add_argument("--dtype", type=str, default="bfloat16",
                        choices=["float32", "float16", "bfloat16"],
                        help="Model dtype")

    # ---- Generation hyperparameters ----
    parser.add_argument("--max_new_tokens", type=int, default=256)
    parser.add_argument("--max_active", type=int, default=14)
    parser.add_argument("--branching_factor", type=int, default=2)
    parser.add_argument("--entropy_ema_alpha", type=float, default=0.35)
    parser.add_argument("--relative_entropy_multiplier", type=float, default=1.1)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--rep_penalty", type=float, default=1.15)

    # ---- Semantic diversity pruning ----
    parser.add_argument("--sim_threshold", type=float, default=0.78)
    parser.add_argument("--ema_alpha", type=float, default=0.25)

    # ---- Soft exploration warmup ----
    parser.add_argument("--soft_explore_window", type=int, default=18)
    parser.add_argument("--soft_explore_initial", type=float, default=0.3)

    # ---- Paged KV cache ----
    parser.add_argument("--block_size", type=int, default=16)
    parser.add_argument("--max_blocks", type=int, default=8192)

    # ---- Output ----
    parser.add_argument("--output", type=str, default="dpo_candidates.jsonl",
                        help="Output JSONL file path")
    parser.add_argument("--save_config", type=str, default=None,
                        help="Also save a standalone JSON config file "
                             "(defaults to <output>.config.json)")

    args = parser.parse_args()

    # ---- Resolve device / dtype ----
    dtype_map = {
        "float32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
    }
    dtype = dtype_map[args.dtype]
    device = (
        "cuda" if args.device == "auto" and torch.cuda.is_available()
        else args.device if args.device != "auto"
        else "cpu"
    )
    tokenizer_path = args.tokenizer_path or args.model_path

    # ---- Build the run config dict (written to output for reproducibility) ----
    run_config = {
        "dataset": args.dataset,
        "split": args.split,
        "prompt_column": args.prompt_column,
        "input_column": args.input_column,
        "prompt_template": args.prompt_template,
        "max_prompts": args.max_prompts,
        "model_path": args.model_path,
        "device": device,
        "dtype": args.dtype,
        "generation": {
            "max_new_tokens": args.max_new_tokens,
            "max_active": args.max_active,
            "branching_factor": args.branching_factor,
            "entropy_ema_alpha": args.entropy_ema_alpha,
            "relative_entropy_multiplier": args.relative_entropy_multiplier,
            "temperature": args.temperature,
            "rep_penalty": args.rep_penalty,
            "sim_threshold": args.sim_threshold,
            "ema_alpha": args.ema_alpha,
            "soft_explore_window": args.soft_explore_window,
            "soft_explore_initial": args.soft_explore_initial,
            "block_size": args.block_size,
            "max_blocks": args.max_blocks,
        },
    }

    # ---- Load prompts ----
    print_header("Loading prompts")
    prompts = load_prompts(args.dataset, args.split, args.prompt_column,
                           args.input_column, args.prompt_template,
                           args.max_prompts)
    print(f"  Loaded {len(prompts)} prompts from: {args.dataset}")
    if args.input_column:
        print(f"  Combined columns: {args.prompt_column} + {args.input_column}")

    if not prompts:
        print("  ERROR: No prompts loaded. Check --dataset and --prompt_column.")
        sys.exit(1)

    # ---- Load model (once) ----
    print_header("Loading model")
    print(f"  Model:   {args.model_path}")
    print(f"  Device:  {device}")
    print(f"  Dtype:   {args.dtype}")

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path,
                                              trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        torch_dtype=dtype,
        trust_remote_code=True,
    ).to(device).eval()

    adapter = get_adapter(model)

    num_params = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"  Parameters:  {num_params:.1f}M")
    print(f"  Vocab size:  {model.config.vocab_size}")

    # ---- Print config ----
    print_header("Generation config")
    for key, val in run_config["generation"].items():
        print(f"  {key:32s} {val}")

    # ---- Sequential generation loop ----
    print_header(f"Generating completions for {len(prompts)} prompts")

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Track aggregate stats
    total_completions = 0
    total_time = 0.0
    total_decode_tokens = 0
    total_decode_time = 0.0

    # Open output file in write mode (overwrite if exists).
    # We write one JSONL line per prompt as soon as it's done so that
    # partial results are preserved if the run is interrupted.
    with open(output_path, "w", encoding="utf-8") as out_f:
        # Write the config as the first line (tagged so the reader can
        # distinguish it from prompt records)
        config_line = json.dumps({"__config__": run_config}, ensure_ascii=False)
        out_f.write(config_line + "\n")

        for idx, prompt in enumerate(prompts):
            # Create a fresh generator per prompt (resets KV cache pool)
            gen = PagedPrefixTreeUQGenerator(
                model,
                tokenizer,
                adapter=adapter,
                max_active_branches=args.max_active,
                branching_factor=args.branching_factor,
                relative_entropy_multiplier=args.relative_entropy_multiplier,
                entropy_ema_alpha=args.entropy_ema_alpha,
                max_new_tokens=args.max_new_tokens,
                temperature=args.temperature,
                repetition_penalty=args.rep_penalty,
                block_size=args.block_size,
                max_blocks=args.max_blocks,
                semantic_similarity_threshold=args.sim_threshold,
                ema_alpha=args.ema_alpha,
                soft_explore_window=args.soft_explore_window,
                soft_explore_initial=args.soft_explore_initial,
            )

            t0 = time.perf_counter()
            results = gen.generate(prompt)
            elapsed = time.perf_counter() - t0

            diag = gen.get_diagnostics()

            record = serialize_prompt_result(
                prompt_id=idx,
                prompt=prompt,
                source_dataset=args.dataset,
                results=results,
                diagnostics=diag,
                elapsed=elapsed,
                eos_token_id=tokenizer.eos_token_id,
            )

            # Write immediately (streaming / crash-safe)
            out_f.write(json.dumps(record, ensure_ascii=False) + "\n")
            out_f.flush()

            # Accumulate stats
            total_completions += record["num_completions"]
            total_time += elapsed
            total_decode_tokens += diag["decode_tokens_total"]
            total_decode_time += diag["decode_time_s"]

            print_progress(
                idx, len(prompts), prompt,
                record["num_completions"], elapsed,
                diag["decode_throughput_tps"],
            )

    # ---- Save standalone config file ----
    config_path = args.save_config or str(output_path) + ".config.json"
    with open(config_path, "w", encoding="utf-8") as f:
        json.dump(run_config, f, indent=2)

    # ---- Summary ----
    print_header("Generation complete")
    print(f"  Output:             {output_path.resolve()}")
    print(f"  Config:             {Path(config_path).resolve()}")
    print(f"  Prompts processed:  {len(prompts)}")
    print(f"  Total completions:  {total_completions}")
    print(f"  Avg completions:    {total_completions / max(len(prompts), 1):.1f} per prompt")
    print(f"  Total time:         {total_time:.1f}s")
    print(f"  Avg time/prompt:    {total_time / max(len(prompts), 1):.2f}s")
    if total_decode_time > 0:
        print(f"  Aggregate decode:   {total_decode_tokens / total_decode_time:.1f} tok/s")


if __name__ == "__main__":
    main()