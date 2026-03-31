"""
Beam Search Baseline
====================

Minimal beam search generation using HuggingFace's built-in
`model.generate()` for direct comparison against the prefix-tree
UQ pipeline.

Produces the same output shape: a list of sequences ranked by
log-probability, with optional answer extraction.

Usage::

    python beam_baseline.py \
        --model meta-llama/Llama-3.2-1B \
        --prompt "The meaning of life is" \
        --num-beams 10

    # Chat-template mode (instruct models)
    python beam_baseline.py \
        --model Qwen/Qwen2.5-7B-Instruct \
        --prompt "What is 37 * 29?" \
        --chat-template auto \
        --num-beams 10

    # Batch mode from a file (one prompt per line)
    python beam_baseline.py \
        --model Qwen/Qwen2.5-7B \
        --prompt-file prompts.txt \
        --num-beams 5 \
        --output-json beam_results.json
"""

from __future__ import annotations

import argparse
import json
import math
import re
import time
from pathlib import Path

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM


# ---------------------------------------------------------------------------
# Answer extraction (mirrors api_server.py)
# ---------------------------------------------------------------------------

def extract_answer(text: str) -> str | None:
    r"""
    Try to pull a final answer from model output.

    Checks (in order):
        1. \boxed{...}          — standard math RLVR format
        2. <answer>...</answer> — common RLVR tag format
        3. ####  ...            — GSM8K-style
    """
    m = re.search(r"\\boxed\{([^{}]*(?:\{[^{}]*\}[^{}]*)*)\}", text)
    if m:
        return m.group(1).strip()

    m = re.search(r"<answer>(.*?)</answer>", text, re.DOTALL)
    if m:
        return m.group(1).strip()

    m = re.search(r"####\s*(.+?)(?:\n|$)", text)
    if m:
        return m.group(1).strip()

    return None


# ---------------------------------------------------------------------------
# Chat template handling (subset of api_server.py logic)
# ---------------------------------------------------------------------------

def format_prompt(
    raw_prompt: str,
    tokenizer,
    chat_template: str,
) -> str:
    """Wrap a raw prompt string with a chat template if requested."""
    if chat_template == "none":
        return raw_prompt

    messages = [{"role": "user", "content": raw_prompt}]

    if chat_template == "auto":
        tok_tmpl = getattr(tokenizer, "chat_template", None)
        if tok_tmpl:
            return tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True,
            )
        # Tokenizer has no template — fall through to raw
        return raw_prompt

    # Named or custom Jinja string — pass directly
    return tokenizer.apply_chat_template(
        messages,
        chat_template=chat_template,
        tokenize=False,
        add_generation_prompt=True,
    )


# ---------------------------------------------------------------------------
# Generation
# ---------------------------------------------------------------------------

def run_beam_search(
    model,
    tokenizer,
    prompt: str,
    num_beams: int,
    num_return: int,
    max_new_tokens: int,
    temperature: float,
    repetition_penalty: float,
    length_penalty: float,
    chat_template: str,
) -> dict:
    """Run beam search on a single prompt and return structured results."""

    formatted = format_prompt(prompt, tokenizer, chat_template)

    inputs = tokenizer(formatted, return_tensors="pt").to(model.device)
    input_len = inputs["input_ids"].shape[1]

    t0 = time.perf_counter()
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            num_beams=num_beams,
            num_return_sequences=num_return,
            temperature=temperature,
            repetition_penalty=repetition_penalty,
            length_penalty=length_penalty,
            early_stopping=True,
            return_dict_in_generate=True,
            output_scores=True,
        )
    elapsed = time.perf_counter() - t0

    # Extract sequences and scores
    sequences = []
    answer_counts: dict[str, int] = {}

    for i in range(num_return):
        full_ids = outputs.sequences[i]
        gen_ids = full_ids[input_len:]
        text = tokenizer.decode(gen_ids, skip_special_tokens=True)
        num_tokens = len(gen_ids)

        # Beam search scores: sequences_scores is log-prob of the full beam
        score = (
            outputs.sequences_scores[i].item()
            if hasattr(outputs, "sequences_scores") and outputs.sequences_scores is not None
            else 0.0
        )

        # Check for EOS
        eos_id = tokenizer.eos_token_id
        if isinstance(eos_id, (list, tuple)):
            finished = gen_ids[-1].item() in eos_id if len(gen_ids) > 0 else False
        else:
            finished = gen_ids[-1].item() == eos_id if len(gen_ids) > 0 else False

        extracted = extract_answer(text)
        if extracted is not None:
            answer_counts[extracted] = answer_counts.get(extracted, 0) + 1

        sequences.append({
            "text": text,
            "token_ids": gen_ids.tolist(),
            "log_prob": round(score, 6),
            "avg_log_prob": round(score / max(num_tokens, 1), 6),
            "num_tokens": num_tokens,
            "finish_reason": "stop" if finished else "length",
            "extracted_answer": extracted,
        })

    # Answer summary
    n_with_answer = sum(1 for s in sequences if s["extracted_answer"] is not None)
    majority_answer = None
    agreement_ratio = 0.0
    if answer_counts:
        majority_answer = max(answer_counts, key=answer_counts.get)
        agreement_ratio = answer_counts[majority_answer] / max(n_with_answer, 1)

    return {
        "method": "beam_search",
        "prompt": prompt,
        "formatted_prompt": formatted if formatted != prompt else None,
        "num_beams": num_beams,
        "num_sequences": len(sequences),
        "sequences": sequences,
        "answer_summary": {
            "distinct_answers": len(answer_counts),
            "majority_answer": majority_answer,
            "agreement_ratio": round(agreement_ratio, 4),
            "answer_distribution": answer_counts,
            "sequences_with_answer": n_with_answer,
        },
        "elapsed_seconds": round(elapsed, 4),
        "usage": {
            "prompt_tokens": input_len,
            "completion_tokens": sum(s["num_tokens"] for s in sequences),
        },
    }


# ---------------------------------------------------------------------------
# Pretty printing
# ---------------------------------------------------------------------------

def print_result(result: dict):
    prompt = result["prompt"]
    print(f"\n{'=' * 72}")
    print(f"  PROMPT: \"{prompt}\"")
    print(f"  Method: beam_search  |  Beams: {result['num_beams']}  "
          f"|  Sequences: {result['num_sequences']}  "
          f"|  Time: {result['elapsed_seconds']:.2f}s")
    print(f"{'=' * 72}")

    summary = result["answer_summary"]
    if summary["sequences_with_answer"] > 0:
        print(f"  Answer summary: {summary['distinct_answers']} distinct, "
              f"majority=\"{summary['majority_answer']}\" "
              f"(agreement={summary['agreement_ratio']:.0%})")

    for i, seq in enumerate(result["sequences"]):
        tag = seq["finish_reason"].upper()
        ans = f"  → {seq['extracted_answer']}" if seq["extracted_answer"] else ""
        print(f"\n  [{i + 1}] ({tag})  score={seq['log_prob']:.3f}  "
              f"tokens={seq['num_tokens']}{ans}")
        print(f"      {seq['text']}")
    print()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="beam_baseline",
        description="Beam Search Baseline — for comparison with prefix-tree UQ",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    model_group = parser.add_argument_group("Model")
    model_group.add_argument("--model", type=str, required=True,
                             help="HuggingFace model name or local path")
    model_group.add_argument("--tokenizer", type=str, default=None,
                             help="Tokenizer (defaults to --model)")
    model_group.add_argument("--dtype", type=str, default="bfloat16",
                             choices=["float32", "float16", "bfloat16"])
    model_group.add_argument("--device", type=str, default="auto",
                             choices=["auto", "cuda", "cpu"])
    model_group.add_argument("--trust-remote-code", action="store_true")
    model_group.add_argument("--chat-template", type=str, default="none",
                             help="'none', 'auto', or a built-in name / Jinja2 string")

    gen_group = parser.add_argument_group("Generation")
    gen_group.add_argument("--num-beams", type=int, default=10,
                           help="Number of beams")
    gen_group.add_argument("--num-return", type=int, default=None,
                           help="Sequences to return (defaults to --num-beams)")
    gen_group.add_argument("--max-new-tokens", type=int, default=256)
    gen_group.add_argument("--temperature", type=float, default=1.0,
                           help="Temperature (1.0 = no scaling for beam search)")
    gen_group.add_argument("--repetition-penalty", type=float, default=1.2)
    gen_group.add_argument("--length-penalty", type=float, default=1.0,
                           help="Length penalty (>1 = longer, <1 = shorter)")

    io_group = parser.add_argument_group("Input / Output")
    io_group.add_argument("--prompt", type=str, nargs="+", default=None,
                          help="One or more prompts")
    io_group.add_argument("--prompt-file", type=str, default=None,
                          help="Text file with one prompt per line")
    io_group.add_argument("--output-json", type=str, default=None,
                          help="Save results to JSON file")

    args = parser.parse_args()

    if args.tokenizer is None:
        args.tokenizer = args.model
    if args.num_return is None:
        args.num_return = args.num_beams

    return args


def main():
    args = parse_args()

    # ── Resolve prompts ─────────────────────────────────────────────────
    prompts: list[str] = []
    if args.prompt:
        prompts.extend(args.prompt)
    if args.prompt_file:
        p = Path(args.prompt_file)
        if not p.exists():
            print(f"Error: prompt file not found: {p}")
            return
        prompts.extend(
            line.strip() for line in p.read_text().splitlines() if line.strip()
        )
    if not prompts:
        prompts = [
            "The meaning of life is",
            "The best way to learn a new programming language is to",
        ]
        print(f"No prompts specified — using {len(prompts)} defaults.\n")

    # ── Load model ──────────────────────────────────────────────────────
    dtype_map = {"float32": torch.float32, "float16": torch.float16, "bfloat16": torch.bfloat16}
    dtype = dtype_map[args.dtype]
    device = args.device
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"Loading model: {args.model}  (dtype={args.dtype}, device={device})")
    tokenizer = AutoTokenizer.from_pretrained(
        args.tokenizer, trust_remote_code=args.trust_remote_code,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=dtype, trust_remote_code=args.trust_remote_code,
    ).to(device).eval()

    num_params = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"Model ready: {num_params:.1f}M params\n")

    # ── Run generation ──────────────────────────────────────────────────
    all_results = []
    for i, prompt in enumerate(prompts):
        print(f"[{i + 1}/{len(prompts)}] Generating...")
        result = run_beam_search(
            model, tokenizer, prompt,
            num_beams=args.num_beams,
            num_return=args.num_return,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            repetition_penalty=args.repetition_penalty,
            length_penalty=args.length_penalty,
            chat_template=args.chat_template,
        )
        print_result(result)
        all_results.append(result)

    # ── Save JSON ───────────────────────────────────────────────────────
    if args.output_json:
        out = Path(args.output_json)
        with open(out, "w") as f:
            json.dump({
                "config": {
                    "model": args.model,
                    "num_beams": args.num_beams,
                    "num_return": args.num_return,
                    "max_new_tokens": args.max_new_tokens,
                    "temperature": args.temperature,
                    "repetition_penalty": args.repetition_penalty,
                    "length_penalty": args.length_penalty,
                    "chat_template": args.chat_template,
                    "method": "beam_search",
                },
                "prompts": all_results,
            }, f, indent=2)
        print(f"Results saved to: {out.resolve()}")

    # ── Summary ─────────────────────────────────────────────────────────
    total_time = sum(r["elapsed_seconds"] for r in all_results)
    total_seqs = sum(r["num_sequences"] for r in all_results)
    print(f"\nDone: {len(all_results)} prompts, {total_seqs} total sequences, "
          f"{total_time:.2f}s total")


if __name__ == "__main__":
    main()