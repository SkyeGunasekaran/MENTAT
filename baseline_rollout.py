"""
RLVR Rollout Baseline
====================

Standard sampling rollout (prompting N times) using HuggingFace's
`model.generate()` for RLVR-style data collection.

Usage::

    python rollout_baseline.py \
        --model meta-llama/Llama-3.2-1B \
        --prompt "The meaning of life is" \
        --n 10 \
        --temperature 0.7

    # Chat-template mode
    python rollout_baseline.py \
        --model Qwen/Qwen2.5-7B-Instruct \
        --prompt "What is 37 * 29?" \
        --chat-template auto \
        --n 10
"""

from __future__ import annotations

import argparse
import json
import re
import time
from pathlib import Path

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM


# ---------------------------------------------------------------------------
# Answer extraction
# ---------------------------------------------------------------------------

def extract_answer(text: str) -> str | None:
    r"""Try to pull a final answer from model output."""
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
# Chat template handling
# ---------------------------------------------------------------------------

def format_prompt(raw_prompt: str, tokenizer, chat_template: str) -> str:
    if chat_template == "none":
        return raw_prompt

    messages = [{"role": "user", "content": raw_prompt}]

    if chat_template == "auto":
        tok_tmpl = getattr(tokenizer, "chat_template", None)
        if tok_tmpl:
            return tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True,
            )
        return raw_prompt

    return tokenizer.apply_chat_template(
        messages,
        chat_template=chat_template,
        tokenize=False,
        add_generation_prompt=True,
    )


# ---------------------------------------------------------------------------
# Generation (Updated for Sampling Rollouts)
# ---------------------------------------------------------------------------

def run_rollout(
    model,
    tokenizer,
    prompt: str,
    n: int,
    max_new_tokens: int,
    temperature: float,
    top_p: float,
    repetition_penalty: float,
    chat_template: str,
) -> dict:
    """Run N independent sampling rollouts on a single prompt."""

    formatted = format_prompt(prompt, tokenizer, chat_template)
    inputs = tokenizer(formatted, return_tensors="pt").to(model.device)
    input_len = inputs["input_ids"].shape[1]

    t0 = time.perf_counter()
    with torch.no_grad():
        # Standard RLVR rollout uses do_sample=True and num_return_sequences=N
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=True,            # Enable sampling
            num_return_sequences=n,    # Number of rollouts (N)
            temperature=temperature,
            top_p=top_p,
            repetition_penalty=repetition_penalty,
            return_dict_in_generate=True,
            output_scores=True,
        )
    elapsed = time.perf_counter() - t0

    sequences = []
    answer_counts: dict[str, int] = {}

    for i in range(n):
        full_ids = outputs.sequences[i]
        gen_ids = full_ids[input_len:]
        text = tokenizer.decode(gen_ids, skip_special_tokens=True)
        num_tokens = len(gen_ids)

        # In sampling mode, sequences_scores contains the cumulative log-probs
        score = (
            outputs.sequences_scores[i].item()
            if hasattr(outputs, "sequences_scores") and outputs.sequences_scores is not None
            else 0.0
        )

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

    n_with_answer = sum(1 for s in sequences if s["extracted_answer"] is not None)
    majority_answer = None
    agreement_ratio = 0.0
    if answer_counts:
        majority_answer = max(answer_counts, key=answer_counts.get)
        agreement_ratio = answer_counts[majority_answer] / max(n_with_answer, 1)

    return {
        "method": "sampling_rollout",
        "prompt": prompt,
        "n": n,
        "num_sequences": len(sequences),
        "sequences": sequences,
        "answer_summary": {
            "distinct_answers": len(answer_counts),
            "majority_answer": majority_answer,
            "agreement_ratio": round(agreement_ratio, 4),
            "answer_distribution": answer_counts,
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
    print(f"\n{'=' * 72}")
    print(f"  PROMPT: \"{result['prompt']}\"")
    print(f"  Method: sampling | N: {result['n']} | Time: {result['elapsed_seconds']:.2f}s")
    print(f"{'=' * 72}")

    summary = result["answer_summary"]
    if summary["majority_answer"]:
        print(f"  Majority Answer: \"{summary['majority_answer']}\" ({summary['agreement_ratio']:.0%})")

    for i, seq in enumerate(result["sequences"]):
        tag = seq["finish_reason"].upper()
        ans = f"  → {seq['extracted_answer']}" if seq["extracted_answer"] else ""
        print(f"\n  [{i + 1}] ({tag}) log_p={seq['log_prob']:.2f} | tokens={seq['num_tokens']}{ans}")
        print(f"      {seq['text'][:200]}..." if len(seq['text']) > 200 else f"      {seq['text']}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="RLVR Rollout Baseline")

    model_group = parser.add_argument_group("Model")
    model_group.add_argument("--model", type=str, required=True)
    model_group.add_argument("--dtype", type=str, default="bfloat16", choices=["float32", "float16", "bfloat16"])
    model_group.add_argument("--device", type=str, default="auto")
    model_group.add_argument("--trust-remote-code", action="store_true")
    model_group.add_argument("--chat-template", type=str, default="none")

    gen_group = parser.add_argument_group("Generation")
    gen_group.add_argument("--n", type=int, default=10, help="Number of rollouts per prompt")
    gen_group.add_argument("--max-new-tokens", type=int, default=512)
    gen_group.add_argument("--temperature", type=float, default=0.7)
    gen_group.add_argument("--top-p", type=float, default=0.9)
    gen_group.add_argument("--repetition-penalty", type=float, default=1.0)

    io_group = parser.add_argument_group("Input / Output")
    io_group.add_argument("--prompt", type=str, nargs="+")
    io_group.add_argument("--prompt-file", type=str)
    io_group.add_argument("--output-json", type=str)

    args = parser.parse_args()
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
    dtype = {"float32": torch.float32, "float16": torch.float16, "bfloat16": torch.bfloat16}[args.dtype]
    device = args.device if args.device != "auto" else ("cuda" if torch.cuda.is_available() else "cpu")

    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=args.trust_remote_code)
    if tokenizer.pad_token is None: tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=dtype, trust_remote_code=args.trust_remote_code
    ).to(device).eval()

    all_results = []
    for i, prompt in enumerate(prompts):
        print(f"[{i + 1}/{len(prompts)}] Rolling out...")
        result = run_rollout(
            model, tokenizer, prompt,
            n=args.n,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            top_p=args.top_p,
            repetition_penalty=args.repetition_penalty,
            chat_template=args.chat_template,
        )
        print_result(result)
        all_results.append(result)

    if args.output_json:
        with open(args.output_json, "w") as f:
            json.dump({"config": vars(args), "results": all_results}, f, indent=2)

if __name__ == "__main__":
    main()