#!/usr/bin/env python3
"""
Benchmarks
----------
- GSM8K: grade-school math word problems 
- MATH: competition math                  
- ARC-Challenge: science multiple-choice           

Metrics
-------
- acc@k: fraction of problems where at least one of the top-K completions contains the correct answer.
- efficiency@k: ratio  acc_mentat@k / acc_standard@k
- mean branches: average active branches Mentat used per problem
- wall-time per problem: total generation time / num_problems

Usage
-----
    python eval_harness.py \\
        --model Qwen/Qwen3-1.7B \\
        --benchmarks gsm8k math arc \\
        --k_values 1 3 5 10 \\
        --num_problems 100 \\
        --chat-template auto \\
        --temperature 0.7 \\
        --output_dir ./eval_results
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import os
import re
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset

from core.generator import MentatGenerator
from adapters.adapter_factory import get_adapter
from utils.shared import BUILTIN_TEMPLATES

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(message)s")
from utils.shared import resolve_chat_template, format_prompt_from_messages


# Per-benchmark token budgets — most problems don't need 512 tokens.
# These are defaults; --max-new-tokens overrides if explicitly set.
DEFAULT_TOKEN_BUDGETS: dict[str, int] = {
    "gsm8k": 256,
    "math":  384,
    "arc":   128,
}

# Utils 

_GSM8K_ANS_RE = re.compile(r"####\s*(-?[\d,]+\.?\d*)")
_NUMERIC_RE = re.compile(r"(-?\d[\d,]*\.?\d*)")
_BOXED_RE = re.compile(r"\\boxed\{([^}]*)\}")


def format_prompt(tokenizer, chat_template: str | None, system_msg: str, user_msg: str) -> str:
    if chat_template is None:
        return f"{system_msg}\n\n{user_msg}" if system_msg else user_msg

    messages = []
    if system_msg: messages.append({"role": "system", "content": system_msg})
    messages.append({"role": "user", "content": user_msg})

    return format_prompt_from_messages(tokenizer, chat_template, messages)

def extract_gsm8k_gold(answer_text: str) -> str:
    m = _GSM8K_ANS_RE.search(answer_text)
    if m:
        return m.group(1).replace(",", "").strip()
    nums = _NUMERIC_RE.findall(answer_text)
    return nums[-1].replace(",", "").strip() if nums else ""


def extract_gsm8k_pred(completion: str) -> str:
    m = _GSM8K_ANS_RE.search(completion)
    if m:
        return m.group(1).replace(",", "").strip()
    m = _BOXED_RE.search(completion)
    if m:
        inner = m.group(1).strip()
        nums = _NUMERIC_RE.findall(inner)
        if nums:
            return nums[-1].replace(",", "").strip()
    nums = _NUMERIC_RE.findall(completion)
    return nums[-1].replace(",", "").strip() if nums else ""


def grade_numeric(pred: str, gold: str) -> bool:
    if not pred or not gold:
        return False
    try:
        p = float(pred.replace(",", ""))
        g = float(gold.replace(",", ""))
        return math.isclose(p, g, rel_tol=1e-5, abs_tol=1e-8)
    except ValueError:
        return pred.strip() == gold.strip()


def extract_math_gold(solution_text: str) -> str:
    m = _BOXED_RE.search(solution_text)
    if m:
        return _normalize_math_answer(m.group(1))
    return _normalize_math_answer(solution_text.strip().split("\n")[-1])


def extract_math_pred(completion: str) -> str:
    m = _BOXED_RE.search(completion)
    if m:
        return _normalize_math_answer(m.group(1))
    m2 = _GSM8K_ANS_RE.search(completion)
    if m2:
        return _normalize_math_answer(m2.group(1))
    lines = [l.strip() for l in completion.strip().split("\n") if l.strip()]
    return _normalize_math_answer(lines[-1]) if lines else ""


def _normalize_math_answer(s: str) -> str:
    s = s.strip().replace(" ", "")
    if s.endswith("."):
        s = s[:-1]
    s = s.replace("$", "").replace("\\(", "").replace("\\)", "")
    s = s.replace("\\[", "").replace("\\]", "")
    return s


def grade_math(pred: str, gold: str) -> bool:
    if not pred or not gold:
        return False
    try:
        p = float(pred.replace(",", ""))
        g = float(gold.replace(",", ""))
        return math.isclose(p, g, rel_tol=1e-5, abs_tol=1e-8)
    except ValueError:
        pass
    return pred.lower() == gold.lower()


def extract_arc_gold(example: dict) -> str:
    return example["answerKey"].strip().upper()


def extract_arc_pred(completion: str) -> str:
    completion = completion.strip()
    patterns = [
        r"(?:the\s+)?answer\s+is\s*[:\s]*\(?([A-Da-d])\)?",
        r"(?:^|\n)\s*\(?([A-Da-d])\)?\s*$",
        r"\b([A-Da-d])\b\s*(?:is\s+(?:the\s+)?(?:correct|right|answer))",
    ]
    for pat in patterns:
        m = re.search(pat, completion, re.IGNORECASE)
        if m:
            return m.group(1).upper()
    m = re.search(r"\b([A-D])\b", completion)
    if m:
        return m.group(1)
    return ""


def grade_arc(pred: str, gold: str) -> bool:
    return pred.strip().upper() == gold.strip().upper()



@dataclass
class Problem:
    idx: int
    prompt: str           # Fully formatted, model-ready string
    gold_answer: str
    benchmark: str
    max_new_tokens: int   # Per-benchmark budget
    raw_example: dict = field(default_factory=dict, repr=False)


# System prompts — concise, format-instructive
SYSTEM_PROMPTS: dict[str, str] = {
    "gsm8k": (
        "You are a math assistant. Solve the problem step by step. "
        "End with your final numeric answer on its own line after ####."
    ),
    "math": (
        "You are a math assistant. Solve the problem step by step. "
        "Put your final answer inside \\boxed{}."
    ),
    "arc": (
        "You are a science assistant. Pick the correct answer. "
        "Briefly explain your reasoning, then state your final answer "
        "as a single letter (A, B, C, or D)."
    ),
}


def load_gsm8k(
    num_problems: int, tokenizer, chat_template: str | None,
    max_new_tokens_override: int | None, split: str = "test",
) -> list[Problem]:
    ds = load_dataset("openai/gsm8k", "main", split=split)
    budget = max_new_tokens_override or DEFAULT_TOKEN_BUDGETS["gsm8k"]
    problems = []
    for i, ex in enumerate(ds):
        if i >= num_problems:
            break
        prompt = format_prompt(
            tokenizer, chat_template,
            system_msg=SYSTEM_PROMPTS["gsm8k"],
            user_msg=ex["question"],
        )
        gold = extract_gsm8k_gold(ex["answer"])
        problems.append(Problem(
            idx=i, prompt=prompt, gold_answer=gold,
            benchmark="gsm8k", max_new_tokens=budget,
            raw_example=dict(ex),
        ))
    return problems


def load_math(
    num_problems: int, tokenizer, chat_template: str | None,
    max_new_tokens_override: int | None, split: str = "test",
) -> list[Problem]:
    for name in ["HuggingFaceH4/MATH-500"]:
        try:
            ds = load_dataset(name, split=split, trust_remote_code=True)
            break
        except Exception:
            continue
    else:
        print("[WARN] Could not load MATH dataset -- skipping.")
        return []

    budget = max_new_tokens_override or DEFAULT_TOKEN_BUDGETS["math"]
    problems = []
    for i, ex in enumerate(ds):
        if i >= num_problems:
            break
        prompt = format_prompt(
            tokenizer, chat_template,
            system_msg=SYSTEM_PROMPTS["math"],
            user_msg=ex["problem"],
        )
        gold = extract_math_gold(ex["solution"])
        problems.append(Problem(
            idx=i, prompt=prompt, gold_answer=gold,
            benchmark="math", max_new_tokens=budget,
            raw_example=dict(ex),
        ))
    return problems


def load_arc(
    num_problems: int, tokenizer, chat_template: str | None,
    max_new_tokens_override: int | None, split: str = "test",
) -> list[Problem]:
    try:
        ds = load_dataset("allenai/ai2_arc", "ARC-Challenge", split=split)
    except Exception:
        print("[WARN] Could not load ARC-Challenge -- skipping.")
        return []

    budget = max_new_tokens_override or DEFAULT_TOKEN_BUDGETS["arc"]
    problems = []
    for i, ex in enumerate(ds):
        if i >= num_problems:
            break
        choices = ex["choices"]
        choice_text = "\n".join(
            f"({label}) {text}"
            for label, text in zip(choices["label"], choices["text"])
        )
        user_msg = f"{ex['question']}\n\n{choice_text}"
        prompt = format_prompt(
            tokenizer, chat_template,
            system_msg=SYSTEM_PROMPTS["arc"],
            user_msg=user_msg,
        )
        gold = extract_arc_gold(ex)
        problems.append(Problem(
            idx=i, prompt=prompt, gold_answer=gold,
            benchmark="arc", max_new_tokens=budget,
            raw_example=dict(ex),
        ))
    return problems


BENCHMARK_LOADERS = {
    "gsm8k": load_gsm8k,
    "math": load_math,
    "arc": load_arc,
}

EXTRACTORS = {
    "gsm8k": (extract_gsm8k_pred, grade_numeric),
    "math": (extract_math_pred, grade_math),
    "arc": (extract_arc_pred, grade_arc),
}


# standard baseline 

@torch.no_grad()
def standard_sample(
    model,
    tokenizer,
    prompt: str,
    n_samples: int,
    max_new_tokens: int = 256,
    temperature: float = 0.7,
    top_p: float = 0.95,
) -> list[dict]:
    """
    Generate n_samples completions in a single batched generate() call.
    """
    device = next(model.parameters()).device
    enc = tokenizer(prompt, return_tensors="pt").to(device)
    input_ids = enc["input_ids"]
    attention_mask = enc["attention_mask"]
    prompt_len = input_ids.shape[1]

    input_ids = input_ids.expand(n_samples, -1)
    attention_mask = attention_mask.expand(n_samples, -1)

    out = model.generate(
        input_ids,
        attention_mask=attention_mask,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_p=top_p,
        do_sample=True,
        pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
    )

    results = []
    for i in range(n_samples):
        gen_ids = out[i, prompt_len:]
        text = tokenizer.decode(gen_ids, skip_special_tokens=True)
        results.append({"text": text, "log_prob": 0.0})

    return results


# scoring 

def compute_acc_at_k(
    completions: list[dict],
    gold: str,
    benchmark: str,
    k_values: list[int],
) -> dict[int, bool]:
    extractor, grader = EXTRACTORS[benchmark]
    grades = []
    for comp in completions:
        pred = extractor(comp["text"])
        correct = grader(pred, gold)
        grades.append(correct)
    results = {}
    for k in k_values:
        top_k = grades[:k]
        results[k] = any(top_k)
    return results


# main eval

def run_evaluation(args):
    print("=" * 72)
    print("  Mentat vs. Standard Sampling — Reasoning Benchmark Evaluation")
    print("=" * 72)

    # ---- Load model -------------------------------------------------------
    print(f"\n[1/5] Loading model: {args.model}")
    dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = "mps"
        dtype = torch.float32

    tokenizer = AutoTokenizer.from_pretrained(
        args.model, trust_remote_code=True,
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=dtype,
        device_map=device if device == "cuda" else None,
        trust_remote_code=True,
    )
    if device != "cuda":
        model = model.to(device)
    model.eval()

    print(f"       Device: {device}  |  Dtype: {dtype}")
    print(f"       Parameters: {sum(p.numel() for p in model.parameters()):,}")

    # ---- Resolve chat template --------------------------------------------
    print(f"\n[2/5] Resolving chat template: '{args.chat_template}'")
    chat_template = resolve_chat_template(tokenizer, args.chat_template)
    if chat_template is not None:
        preview = format_prompt(tokenizer, chat_template, "System.", "Hello?")
        print(f"       Preview:\n         {preview[:120]}...")
    else:
        print("       Mode: raw completion (no chat template)")

    # ---- Build Mentat generator -------------------------------------------
    print(f"\n[3/5] Building Mentat generator")
    adapter = get_adapter(model)

    # Use the largest budget across requested benchmarks as the generator's
    # upper bound. Per-problem budgets are set dynamically in the loop.
    max_budget = args.max_new_tokens or max(
        DEFAULT_TOKEN_BUDGETS.get(b, 256) for b in args.benchmarks
    )
    mentat_gen = MentatGenerator(
        model, tokenizer,
        adapter=adapter,
        max_new_tokens=max_budget,
        temperature=args.temperature,
        max_active_branches=args.max_branches,
        branching_factor=args.branching_factor,
        repetition_penalty=args.repetition_penalty,
    )

    # ---- Load benchmarks --------------------------------------------------
    print(f"\n[4/5] Loading benchmarks: {args.benchmarks}")
    all_problems: dict[str, list[Problem]] = {}
    for bench_name in args.benchmarks:
        if bench_name not in BENCHMARK_LOADERS:
            print(f"  [WARN] Unknown benchmark '{bench_name}' -- skipping.")
            continue
        problems = BENCHMARK_LOADERS[bench_name](
            args.num_problems, tokenizer, chat_template, args.max_new_tokens,
        )
        if problems:
            all_problems[bench_name] = problems
            budget = problems[0].max_new_tokens
            print(f"  {bench_name}: {len(problems)} problems loaded "
                  f"(token budget: {budget})")

    if not all_problems:
        print("[ERROR] No problems loaded. Exiting.")
        sys.exit(1)

    k_values = sorted(args.k_values)
    max_k = max(k_values)
    n_standard_samples = max_k

    # ---- Run evaluation ---------------------------------------------------
    print(f"\n[5/5] Running evaluation (k_values={k_values})")
    print(f"       Mentat: max_branches={args.max_branches}, "
          f"branching_factor={args.branching_factor}")
    print(f"       Standard: n_samples={n_standard_samples}")
    print(f"       Temperature: {args.temperature}")
    print("-" * 72)

    all_results = {}

    for bench_name, problems in all_problems.items():
        print(f"\n  --- {bench_name.upper()} ({len(problems)} problems) ---")

        mentat_acc = {k: 0 for k in k_values}
        standard_acc = {k: 0 for k in k_values}
        mentat_times = []
        standard_times = []
        mentat_branch_counts = []
        per_problem_details = []

        for pi, problem in enumerate(problems):
            if (pi + 1) % 5 == 0 or pi == 0:
                print(f"    Problem {pi + 1}/{len(problems)}...", flush=True)

            detail = {
                "idx": problem.idx,
                "gold": problem.gold_answer,
                "prompt_preview": problem.prompt[:150] + "...",
                "max_new_tokens": problem.max_new_tokens,
            }

            # ---- Mentat generation ----
            t0 = time.perf_counter()
            try:
                # Adjust the generator's token budget per benchmark.
                mentat_gen.max_new_tokens = problem.max_new_tokens
                mentat_completions = mentat_gen.generate(problem.prompt)
                mentat_time = time.perf_counter() - t0
                mentat_times.append(mentat_time)

                diag = mentat_gen.get_diagnostics()
                avg_branches = (
                    sum(diag["active_branch_trace"])
                    / max(len(diag["active_branch_trace"]), 1)
                )
                mentat_branch_counts.append(avg_branches)
                detail["mentat_num_completions"] = len(mentat_completions)
                detail["mentat_avg_branches"] = round(avg_branches, 2)
                detail["mentat_time"] = round(mentat_time, 2)

                m_scores = compute_acc_at_k(
                    mentat_completions, problem.gold_answer,
                    bench_name, k_values,
                )
                for k in k_values:
                    mentat_acc[k] += int(m_scores[k])
                detail["mentat_acc_at_k"] = {
                    str(k): m_scores[k] for k in k_values
                }

                extractor, _ = EXTRACTORS[bench_name]
                detail["mentat_preds"] = [
                    extractor(c["text"]) for c in mentat_completions[:max_k]
                ]

            except Exception as e:
                print(f"      [WARN] Mentat failed on problem {pi}: {e}")
                mentat_times.append(0.0)
                mentat_branch_counts.append(0.0)
                detail["mentat_error"] = str(e)

            # ---- Standard sampling ----
            t0 = time.perf_counter()
            try:
                std_completions = standard_sample(
                    model, tokenizer, problem.prompt,
                    n_samples=n_standard_samples,
                    max_new_tokens=problem.max_new_tokens,
                    temperature=args.temperature,
                )
                standard_time = time.perf_counter() - t0
                standard_times.append(standard_time)
                detail["standard_time"] = round(standard_time, 2)

                s_scores = compute_acc_at_k(
                    std_completions, problem.gold_answer,
                    bench_name, k_values,
                )
                for k in k_values:
                    standard_acc[k] += int(s_scores[k])
                detail["standard_acc_at_k"] = {
                    str(k): s_scores[k] for k in k_values
                }

                extractor, _ = EXTRACTORS[bench_name]
                detail["standard_preds"] = [
                    extractor(c["text"]) for c in std_completions[:max_k]
                ]

            except Exception as e:
                print(f"      [WARN] Standard sampling failed on problem {pi}: {e}")
                standard_times.append(0.0)
                detail["standard_error"] = str(e)

            per_problem_details.append(detail)

        # ---- Aggregate results ----
        n = len(problems)
        bench_result = {
            "benchmark": bench_name,
            "num_problems": n,
            "k_values": k_values,
            "token_budget": problems[0].max_new_tokens,
            "chat_template": args.chat_template,
            "mentat": {
                "acc_at_k": {
                    str(k): round(mentat_acc[k] / n, 4) for k in k_values
                },
                "acc_at_k_raw": {str(k): mentat_acc[k] for k in k_values},
                "mean_time_per_problem": round(
                    sum(mentat_times) / max(len(mentat_times), 1), 3
                ),
                "mean_branches": round(
                    sum(mentat_branch_counts)
                    / max(len(mentat_branch_counts), 1), 2
                ),
            },
            "standard": {
                "acc_at_k": {
                    str(k): round(standard_acc[k] / n, 4) for k in k_values
                },
                "acc_at_k_raw": {str(k): standard_acc[k] for k in k_values},
                "mean_time_per_problem": round(
                    sum(standard_times) / max(len(standard_times), 1), 3
                ),
                "n_samples": n_standard_samples,
            },
            "efficiency": {},
            "per_problem": per_problem_details,
        }

        for k in k_values:
            m = mentat_acc[k] / n
            s = standard_acc[k] / n
            if s > 0:
                bench_result["efficiency"][str(k)] = round(m / s, 4)
            else:
                bench_result["efficiency"][str(k)] = (
                    float("inf") if m > 0 else 1.0
                )

        all_results[bench_name] = bench_result

        # ---- Print summary ----
        print(f"\n  {bench_name.upper()} Results ({n} problems, "
              f"budget={problems[0].max_new_tokens} tokens):")
        print(f"  {'k':>6}  {'Mentat acc@k':>14}  "
              f"{'Standard acc@k':>16}  {'Efficiency':>12}")
        print(f"  {'-'*6}  {'-'*14}  {'-'*16}  {'-'*12}")
        for k in k_values:
            m_acc = mentat_acc[k] / n
            s_acc = standard_acc[k] / n
            eff = bench_result["efficiency"][str(k)]
            print(f"  {k:>6}  {m_acc:>14.1%}  {s_acc:>16.1%}  {eff:>12.2f}x")

        m_time = bench_result["mentat"]["mean_time_per_problem"]
        s_time = bench_result["standard"]["mean_time_per_problem"]
        m_branches = bench_result["mentat"]["mean_branches"]
        print(f"\n  Mentat avg time: {m_time:.1f}s  |  Std avg time: {s_time:.1f}s")
        print(f"  Mentat avg branches used: {m_branches:.1f}")

    # ---- Save results -----------------------------------------------------
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    results_path = output_dir / "eval_results.json"
    with open(results_path, "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\n  Full results saved to: {results_path}")

    summary_path = output_dir / "eval_summary.txt"
    with open(summary_path, "w") as f:
        f.write("=" * 72 + "\n")
        f.write("  Mentat vs. Standard Sampling -- Evaluation Summary\n")
        f.write("=" * 72 + "\n\n")
        f.write(f"Model:            {args.model}\n")
        f.write(f"Chat template:    {args.chat_template}\n")
        f.write(f"Temperature:      {args.temperature}\n")
        f.write(f"Device:           {device}\n\n")

        for bench_name, br in all_results.items():
            f.write(f"--- {bench_name.upper()} "
                    f"({br['num_problems']} problems, "
                    f"budget={br['token_budget']} tokens) ---\n\n")
            f.write(f"  {'k':>6}  {'Mentat':>10}  "
                    f"{'Standard':>10}  {'Efficiency':>12}\n")
            f.write(f"  {'-'*6}  {'-'*10}  {'-'*10}  {'-'*12}\n")
            for k in k_values:
                m = br["mentat"]["acc_at_k"][str(k)]
                s = br["standard"]["acc_at_k"][str(k)]
                e = br["efficiency"][str(k)]
                f.write(f"  {k:>6}  {m:>10.1%}  {s:>10.1%}  {e:>12.2f}x\n")
            f.write(
                f"\n  Mentat  -- avg time: "
                f"{br['mentat']['mean_time_per_problem']:.1f}s, "
                f"avg branches: {br['mentat']['mean_branches']:.1f}\n"
            )
            f.write(
                f"  Standard -- avg time: "
                f"{br['standard']['mean_time_per_problem']:.1f}s\n\n"
            )
    print(f"  Summary saved to: {summary_path}")

    csv_path = output_dir / "per_problem.csv"
    with open(csv_path, "w") as f:
        header_parts = ["benchmark", "problem_idx", "gold", "token_budget"]
        for k in k_values:
            header_parts.append(f"mentat_acc@{k}")
            header_parts.append(f"standard_acc@{k}")
        header_parts.extend([
            "mentat_time", "standard_time",
            "mentat_avg_branches", "mentat_num_completions",
        ])
        f.write(",".join(header_parts) + "\n")

        for bench_name, br in all_results.items():
            for detail in br["per_problem"]:
                row = [
                    bench_name, str(detail["idx"]),
                    detail["gold"], str(detail["max_new_tokens"]),
                ]
                for k in k_values:
                    m_val = detail.get("mentat_acc_at_k", {}).get(str(k), "")
                    s_val = detail.get("standard_acc_at_k", {}).get(str(k), "")
                    row.append(str(m_val))
                    row.append(str(s_val))
                row.append(str(detail.get("mentat_time", "")))
                row.append(str(detail.get("standard_time", "")))
                row.append(str(detail.get("mentat_avg_branches", "")))
                row.append(str(detail.get("mentat_num_completions", "")))
                f.write(",".join(row) + "\n")
    print(f"  Per-problem CSV: {csv_path}")

    # ---- Final summary ----------------------------------------------------
    print("\n" + "=" * 72)
    print("  FINAL SUMMARY")
    print("=" * 72)
    for bench_name, br in all_results.items():
        n = br["num_problems"]
        print(f"\n  {bench_name.upper()} ({n} problems, "
              f"budget={br['token_budget']} tokens):")
        for k in k_values:
            m = br["mentat"]["acc_at_k"][str(k)]
            s = br["standard"]["acc_at_k"][str(k)]
            delta = m - s
            symbol = "+" if delta >= 0 else ""
            print(f"    acc@{k:<3}  Mentat: {m:.1%}  "
                  f"Standard: {s:.1%}  ({symbol}{delta:.1%})")

    print("\n" + "=" * 72)
    print("  Done.")
    print("=" * 72)


# ============================================================================
#  6.  CLI
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Mentat vs. Standard Sampling -- Reasoning Benchmark Eval",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--model", type=str, required=True,
        help="HuggingFace model name or path",
    )
    parser.add_argument(
        "--benchmarks", nargs="+", default=["gsm8k"],
        choices=list(BENCHMARK_LOADERS.keys()),
        help="Which benchmarks to evaluate (default: gsm8k)",
    )
    parser.add_argument(
        "--k_values", nargs="+", type=int, default=[1, 3, 5, 10],
        help="k values for acc@k computation (default: 1 3 5 10)",
    )
    parser.add_argument(
        "--num_problems", type=int, default=100,
        help="Number of problems per benchmark (default: 100)",
    )
    parser.add_argument(
        "--max-new-tokens", type=int, default=None,
        help="Override token budget for all benchmarks. "
             "If not set, uses per-benchmark defaults "
             "(gsm8k=256, math=384, arc=128)",
    )
    parser.add_argument(
        "--chat-template", type=str, default="auto",
        help="Chat template: 'auto' (detect from tokenizer), "
             "'none' (raw completion), or a built-in name "
             f"({', '.join(BUILTIN_TEMPLATES.keys())}). Default: auto",
    )
    parser.add_argument(
        "--temperature", type=float, default=0.7,
        help="Sampling temperature for both methods (default: 0.7)",
    )
    parser.add_argument(
        "--max_branches", type=int, default=10,
        help="Mentat max_active_branches (default: 10)",
    )
    parser.add_argument(
        "--branching_factor", type=int, default=3,
        help="Mentat branching_factor (default: 3)",
    )
    parser.add_argument(
        "--repetition_penalty", type=float, default=1.2,
        help="Repetition penalty (default: 1.2)",
    )
    parser.add_argument(
        "--output_dir", type=str, default="./eval_results",
        help="Directory for output files (default: ./eval_results)",
    )

    args = parser.parse_args()
    run_evaluation(args)


if __name__ == "__main__":
    main()