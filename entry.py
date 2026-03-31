"""
Endpoints:

    POST /v1/completions
        Body: {"prompt": "...", <optional overrides>}
          or: {"messages": [{"role": "user", "content": "..."}], ...}
        Returns: JSON with input prompt and all branched output sequences.

    POST /v1/completions/batch
        Body: newline-delimited JSON (JSONL), one request object per line.
        Returns: JSON array of results, one per input line.

    GET  /v1/models
        Returns: model metadata.

    GET  /health
        Returns: {"status": "ok"}

Chat template usage:

    # Auto-detect from tokenizer (recommended for instruct/chat models)
    python api_server.py --model Qwen/Qwen2.5-7B-Instruct --chat-template auto

    # Raw completion (no template, for base models)
    python api_server.py --model Qwen/Qwen2.5-7B --chat-template none

    # Force a specific built-in template
    python api_server.py --model mymodel --chat-template chatml

    # Provide a custom Jinja2 string
    python api_server.py --model mymodel \\
        --chat-template "{% for m in messages %}..."
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import re
import sys
import time
import uuid
from http.server import HTTPServer, BaseHTTPRequestHandler
from threading import Lock
from typing import Any

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

from adapters.adapter_factory import get_adapter
from core.generator import MentatGenerator

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------
# Styling & Logging
# ---------------------------------------------------------------------------
class Style:
    RESET = "\033[0m"
    BOLD = "\033[1m"
    DIM = "\033[2m"
    ORANGE = "\033[38;5;208m" # Mentat spice orange
    CYAN = "\033[36m"
    GREEN = "\033[32m"
    YELLOW = "\033[33m"
    RED = "\033[31m"

class MentatFormatter(logging.Formatter):
    def format(self, record):
        time_str = self.formatTime(record, "%m-%d %H:%M:%S")
        
        if record.levelno == logging.INFO:
            lvl = f"{Style.GREEN}INFO{Style.RESET}"
        elif record.levelno == logging.WARNING:
            lvl = f"{Style.YELLOW}WARN{Style.RESET}"
        elif record.levelno == logging.ERROR:
            lvl = f"{Style.RED}ERR {Style.RESET}"
        else:
            lvl = f"{Style.DIM}DBUG{Style.RESET}"
            
        msg = record.getMessage()
        
        # Colorize specific keywords dynamically to make logs pop
        if "Request:" in msg:
            msg = msg.replace("Request:", f"{Style.CYAN}Request:{Style.RESET}")
        if "Completed:" in msg:
            msg = msg.replace("Completed:", f"{Style.GREEN}Completed:{Style.RESET}")
        if "Model ready:" in msg:
            msg = msg.replace("Model ready:", f"{Style.ORANGE}{Style.BOLD}Model ready:{Style.RESET}")
            
        return f"{Style.DIM}[{time_str}]{Style.RESET} {lvl} {msg}"

logger = logging.getLogger("mentat_api")
logger.setLevel(logging.INFO)
ch = logging.StreamHandler()
ch.setFormatter(MentatFormatter())
if logger.hasHandlers():
    logger.handlers.clear()
logger.addHandler(ch)


# ---------------------------------------------------------------------------
# Globals set during startup
# ---------------------------------------------------------------------------
_model = None
_tokenizer = None
_adapter = None
_args = None
_lock = Lock()          # serialize requests (single-request-at-a-time)


# ---------------------------------------------------------------------------
# Built-in chat templates (fallbacks when tokenizer doesn't ship one)
# ---------------------------------------------------------------------------
BUILTIN_TEMPLATES: dict[str, str] = {
    # ── ChatML (OpenAI / Qwen default) ──────────────────────────────────
    "chatml": (
        "{% for message in messages %}"
        "<|im_start|>{{ message['role'] }}\n"
        "{{ message['content'] }}<|im_end|>\n"
        "{% endfor %}"
        "{% if add_generation_prompt %}"
        "<|im_start|>assistant\n"
        "{% endif %}"
    ),

    # ── Llama 3 Instruct ────────────────────────────────────────────────
    "llama3": (
        "{% for message in messages %}"
        "<|start_header_id|>{{ message['role'] }}<|end_header_id|>\n\n"
        "{{ message['content'] }}<|eot_id|>"
        "{% endfor %}"
        "{% if add_generation_prompt %}"
        "<|start_header_id|>assistant<|end_header_id|>\n\n"
        "{% endif %}"
    ),

    # ── Mistral Instruct ────────────────────────────────────────────────
    "mistral": (
        "{% for message in messages %}"
        "{% if message['role'] == 'user' %}"
        "[INST] {{ message['content'] }} [/INST]"
        "{% elif message['role'] == 'assistant' %}"
        "{{ message['content'] }}</s> "
        "{% endif %}"
        "{% endfor %}"
    ),

    # ── Zephyr / Gemma style ────────────────────────────────────────────
    "zephyr": (
        "{% for message in messages %}"
        "<|{{ message['role'] }}|>\n"
        "{{ message['content'] }}<|endoftext|>\n"
        "{% endfor %}"
        "{% if add_generation_prompt %}"
        "<|assistant|>\n"
        "{% endif %}"
    ),

    # ── DeepSeek R1 (think + answer) ───────────────────────────────────
    "deepseek-r1": (
        "{% for message in messages %}"
        "{% if message['role'] == 'user' %}"
        "User: {{ message['content'] }}\n\n"
        "{% elif message['role'] == 'assistant' %}"
        "Assistant: {{ message['content'] }}\n\n"
        "{% elif message['role'] == 'system' %}"
        "{{ message['content'] }}\n\n"
        "{% endif %}"
        "{% endfor %}"
        "{% if add_generation_prompt %}"
        "Assistant: <think>\n"
        "{% endif %}"
    ),

    # ── No template (raw passthrough) ──────────────────────────────────
    "none": "",
}


# ---------------------------------------------------------------------------
# Chat template resolution
# ---------------------------------------------------------------------------

def resolve_chat_template(args: argparse.Namespace) -> str | None:
    """
    Resolve the effective chat template string.

    Returns:
        A Jinja2 template string, or None meaning "don't apply any template,
        treat prompt as raw text".

    Resolution order:
        1. --chat-template none          → None  (raw completion mode)
        2. --chat-template auto          → tokenizer's built-in template
        3. --chat-template <builtin>     → one of BUILTIN_TEMPLATES
        4. --chat-template <jinja str>   → user-supplied Jinja2 string
    """
    spec = args.chat_template

    if spec == "none":
        return None

    if spec == "auto":
        # Try the tokenizer's own template
        tok_tmpl = getattr(_tokenizer, "chat_template", None)
        if tok_tmpl:
            # Could be a string or a dict of named templates
            if isinstance(tok_tmpl, dict):
                tmpl = tok_tmpl.get("default") or next(iter(tok_tmpl.values()))
                logger.info(f"Chat template: tokenizer (named, using '{next(iter(tok_tmpl))}')")
                return tmpl
            logger.info("Chat template: tokenizer built-in")
            return tok_tmpl
        logger.info("Chat template: tokenizer has none → raw completion mode")
        return None

    if spec in BUILTIN_TEMPLATES:
        tmpl = BUILTIN_TEMPLATES[spec]
        if not tmpl:      # "none" entry
            return None
        logger.info(f"Chat template: built-in '{spec}'")
        return tmpl

    # Treat as a raw Jinja2 string
    logger.info("Chat template: user-supplied Jinja2 string")
    return spec


def format_prompt(
    body: dict,
    chat_template: str | None,
) -> str:
    """
    Turn the request body into a flat prompt string for the generator.

    Supports two input shapes:
        1. {"prompt": "raw text"}        → returned as-is
        2. {"messages": [{role, content}]} → formatted via chat template

    If the body has "messages" but no chat template is configured, we
    concatenate the content fields with newlines (best-effort fallback).
    """
    # ── Raw prompt path ─────────────────────────────────────────────────
    if "prompt" in body:
        return body["prompt"]

    # ── Messages path ───────────────────────────────────────────────────
    messages = body.get("messages")
    if not messages or not isinstance(messages, list):
        raise ValueError(
            "Request must contain either 'prompt' (string) or "
            "'messages' (list of {role, content} dicts)."
        )

    # Validate structure
    for i, msg in enumerate(messages):
        if not isinstance(msg, dict) or "role" not in msg or "content" not in msg:
            raise ValueError(
                f"messages[{i}] must be a dict with 'role' and 'content' keys."
            )

    if chat_template is not None:
        # Use HuggingFace's apply_chat_template with our resolved template
        return _tokenizer.apply_chat_template(
            messages,
            chat_template=chat_template,
            tokenize=False,
            add_generation_prompt=True,
        )

    # No template — plain concatenation fallback
    parts = []
    for msg in messages:
        role = msg["role"]
        content = msg["content"]
        if role == "system":
            parts.append(content)
        elif role == "user":
            parts.append(f"User: {content}")
        elif role == "assistant":
            parts.append(f"Assistant: {content}")
        else:
            parts.append(content)
    # Add generation cue
    parts.append("Assistant:")
    return "\n".join(parts)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def safe_float(x: Any) -> Any:
    if isinstance(x, float) and (math.isnan(x) or math.isinf(x)):
        return None
    return round(x, 6) if isinstance(x, float) else x


def extract_answer(text: str) -> str | None:
    r"""
    Try to pull a final answer from model output.

    Checks (in order):
        1. \boxed{...}          — standard math RLVR format
        2. <answer>...</answer> — common RLVR tag format
        3. ####  ...            — GSM8K-style
    Returns None if nothing matched.
    """
    # \boxed{...} (handles nested braces one level deep)
    m = re.search(r"\\boxed\{([^{}]*(?:\{[^{}]*\}[^{}]*)*)\}", text)
    if m:
        return m.group(1).strip()

    # <answer>...</answer>
    m = re.search(r"<answer>(.*?)</answer>", text, re.DOTALL)
    if m:
        return m.group(1).strip()

    # #### answer
    m = re.search(r"####\s*(.+?)(?:\n|$)", text)
    if m:
        return m.group(1).strip()

    return None


def build_response(
    prompt: str,
    results: list[dict],
    diagnostics: dict,
    elapsed: float,
) -> dict:
    """Build the JSON response object for a single generation request."""

    # ── Per-sequence output ─────────────────────────────────────────────
    sequences = []
    answer_counts: dict[str, int] = {}

    for r in results:
        tids = r["token_ids"]
        text = r["text"]
        length = max(len(tids), 1)

        extracted = extract_answer(text)
        if extracted is not None:
            answer_counts[extracted] = answer_counts.get(extracted, 0) + 1

        sequences.append({
            "text": text,
            "token_ids": tids,
            "log_prob": safe_float(r["log_prob"]),
            "avg_log_prob": safe_float(r["log_prob"] / length),
            "norm_prob": safe_float(r.get("norm_prob")),
            "num_tokens": len(tids),
            "finish_reason": "stop" if r.get("complete") else "length",
            "extracted_answer": extracted,
        })

    # ── Group-level answer statistics ───────────────────────────────────
    n_with_answer = sum(1 for s in sequences if s["extracted_answer"] is not None)
    majority_answer = None
    agreement_ratio = 0.0
    if answer_counts:
        majority_answer = max(answer_counts, key=answer_counts.get)
        agreement_ratio = answer_counts[majority_answer] / max(n_with_answer, 1)

    return {
        "id": f"uq-{uuid.uuid4().hex[:12]}",
        "object": "uq.completion",
        "created": int(time.time()),
        "model": _args.model,
        "prompt": prompt,
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
        "performance": {
            "prefill_tokens": diagnostics["prefill_tokens"],
            "prefill_time_s": diagnostics["prefill_time_s"],
            "prefill_throughput_tps": diagnostics["prefill_throughput_tps"],
            "decode_tokens_total": diagnostics["decode_tokens_total"],
            "decode_time_s": diagnostics["decode_time_s"],
            "decode_throughput_tps": diagnostics["decode_throughput_tps"],
            "peak_vram_mb": diagnostics["peak_vram_mb"],
            "peak_kv_cache_mb": diagnostics["peak_kv_cache_mb"],
        },
        "usage": {
            "prompt_tokens": diagnostics["prefill_tokens"],
            "completion_tokens": diagnostics["decode_tokens_total"],
        },
    }


# ---------------------------------------------------------------------------
# Core generation logic
# ---------------------------------------------------------------------------

def run_generation(body: dict, chat_template: str | None) -> dict:
    """Resolve prompt, instantiate a fresh generator, run, return response."""
    global _model, _tokenizer, _adapter, _args

    # ── Resolve prompt ──────────────────────────────────────────────────
    prompt = format_prompt(body, chat_template)

    # ── Merge per-request overrides with server defaults ────────────────
    overrides = {k: v for k, v in body.items() if k not in ("prompt", "messages")}

    max_new_tokens = overrides.get("max_new_tokens", _args.max_new_tokens)
    max_active = overrides.get("max_active_branches", _args.max_active_branches)
    branching_factor = overrides.get("branching_factor", _args.branching_factor)
    temperature = overrides.get("temperature", _args.temperature)
    rep_penalty = overrides.get("repetition_penalty", _args.repetition_penalty)
    entropy_ema_alpha = overrides.get("entropy_ema_alpha", _args.entropy_ema_alpha)
    relative_entropy_multiplier = overrides.get(
        "relative_entropy_multiplier", _args.relative_entropy_multiplier,
    )
    sim_threshold = overrides.get("sim_threshold", _args.sim_threshold)
    ema_alpha = overrides.get("ema_alpha", _args.ema_alpha)
    soft_explore_window = overrides.get("soft_explore_window", _args.soft_explore_window)
    soft_explore_initial = overrides.get("soft_explore_initial", _args.soft_explore_initial)

    gen = MentatGenerator(
        _model,
        _tokenizer,
        adapter=_adapter,
        max_active_branches=max_active,
        branching_factor=branching_factor,
        relative_entropy_multiplier=relative_entropy_multiplier,
        entropy_ema_alpha=entropy_ema_alpha,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        repetition_penalty=rep_penalty,
        block_size=_args.block_size,
        max_blocks=_args.max_blocks,
        semantic_similarity_threshold=sim_threshold,
        ema_alpha=ema_alpha,
        soft_explore_window=soft_explore_window,
        soft_explore_initial=soft_explore_initial,
    )

    t0 = time.perf_counter()
    results = gen.generate(prompt)
    elapsed = time.perf_counter() - t0

    diagnostics = gen.get_diagnostics()

    # Tag completeness
    eos_id = _tokenizer.eos_token_id
    for r in results:
        if not r["token_ids"]:
            r["complete"] = False
        elif isinstance(eos_id, (list, tuple)):
            r["complete"] = r["token_ids"][-1] in eos_id
        else:
            r["complete"] = r["token_ids"][-1] == eos_id

    return build_response(prompt, results, diagnostics, elapsed)


# ---------------------------------------------------------------------------
# HTTP Handler
# ---------------------------------------------------------------------------

class MENTATRequestHandler(BaseHTTPRequestHandler):
    """Minimal HTTP handler — no framework dependencies."""

    def log_message(self, format, *args):
        logger.info(f"{self.client_address[0]} - {format % args}")

    # -- helpers --

    def _send_json(self, obj: Any, status: int = 200):
        body = json.dumps(obj, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _read_body(self) -> bytes:
        length = int(self.headers.get("Content-Length", 0))
        return self.rfile.read(length) if length else b""

    # -- routes --

    def do_GET(self):
        if self.path == "/health":
            self._send_json({"status": "ok"})

        elif self.path == "/v1/models":
            self._send_json({
                "object": "list",
                "data": [{
                    "id": _args.model,
                    "object": "model",
                    "owned_by": "local",
                    "pipeline": "prefix-tree-uq",
                    "chat_template": _args.chat_template,
                }],
            })
        else:
            self._send_json({"error": "Not found"}, 404)

    def do_POST(self):
        if self.path == "/v1/completions":
            self._handle_single()
        elif self.path == "/v1/completions/batch":
            self._handle_batch()
        else:
            self._send_json({"error": "Not found"}, 404)

    # -- single request --

    def _handle_single(self):
        raw = self._read_body()
        try:
            body = json.loads(raw) if raw else {}
        except json.JSONDecodeError as exc:
            self._send_json({"error": f"Invalid JSON: {exc}"}, 400)
            return

        if "prompt" not in body and "messages" not in body:
            self._send_json(
                {"error": "Request must contain 'prompt' (string) or 'messages' (list)."},
                400,
            )
            return

        chat_template = resolve_chat_template(_args)

        preview = body.get("prompt") or body.get("messages", [{}])[0].get("content", "")
        logger.info(f"Request: {preview!r:.80s}")

        with _lock:
            try:
                response = run_generation(body, chat_template)
            except ValueError as exc:
                self._send_json({"error": str(exc)}, 400)
                return
            except Exception as exc:
                logger.exception("Generation failed")
                self._send_json({"error": str(exc)}, 500)
                return

        n_seq = response["num_sequences"]
        elapsed = response["elapsed_seconds"]
        logger.info(f"Completed: {n_seq} sequences in {elapsed:.2f}s")
        self._send_json(response)

    # -- batch request (JSONL) --

    def _handle_batch(self):
        raw = self._read_body()
        if not raw:
            self._send_json({"error": "Empty body. Send newline-delimited JSON."}, 400)
            return

        # Parse JSONL
        lines = raw.decode("utf-8").strip().splitlines()
        requests: list[dict] = []
        for i, line in enumerate(lines):
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError as exc:
                self._send_json(
                    {"error": f"Invalid JSON on line {i + 1}: {exc}"}, 400,
                )
                return
            if "prompt" not in obj and "messages" not in obj:
                self._send_json(
                    {"error": f"Line {i + 1}: must contain 'prompt' or 'messages'."},
                    400,
                )
                return
            requests.append(obj)

        if not requests:
            self._send_json({"error": "No valid request lines found."}, 400)
            return

        logger.info(f"Batch request: {len(requests)} prompts")
        chat_template = resolve_chat_template(_args)

        results: list[dict] = []
        with _lock:
            for i, req_body in enumerate(requests):
                preview = req_body.get("prompt") or req_body.get("messages", [{}])[0].get("content", "")
                logger.info(f"  [{i + 1}/{len(requests)}] {preview!r:.60s}")
                try:
                    resp = run_generation(req_body, chat_template)
                    results.append(resp)
                except Exception as exc:
                    logger.exception(f"  [{i + 1}] failed")
                    results.append({
                        "error": str(exc),
                        "index": i,
                        "prompt": preview,
                    })

        logger.info(f"Batch done: {len(results)} results")
        self._send_json(results)


# ---------------------------------------------------------------------------
# CLI — vLLM-style argument groups
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="api_server",
        description="MENTAT API Server",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # ── Server ──────────────────────────────────────────────────────────
    server = parser.add_argument_group("Server")
    server.add_argument("--host", type=str, default="0.0.0.0",
                        help="Bind address")
    server.add_argument("--port", type=int, default=8000,
                        help="Listen port")

    # ── Model ───────────────────────────────────────────────────────────
    model = parser.add_argument_group("Model")
    model.add_argument("--model", type=str, required=True,
                       help="HuggingFace model name or local path")
    model.add_argument("--tokenizer", type=str, default=None,
                       help="Tokenizer name/path (defaults to --model)")
    model.add_argument("--dtype", type=str, default="bfloat16",
                       choices=["float32", "float16", "bfloat16"],
                       help="Model weight dtype")
    model.add_argument("--device", type=str, default="auto",
                       choices=["auto", "cuda", "cpu"],
                       help="Compute device")
    model.add_argument("--trust-remote-code", action="store_true",
                       help="Trust remote code in model/tokenizer")
    model.add_argument(
        "--chat-template", type=str, default="none",
        help=(
            "Chat template to apply to 'messages' input. Options: "
            "'none' (raw completion, default for base models), "
            "'auto' (use tokenizer's built-in template), "
            "or a built-in name: chatml, llama3, mistral, zephyr, deepseek-r1. "
            "You can also pass a raw Jinja2 template string."
        ),
    )

    # ── Generation defaults ─────────────────────────────────────────────
    gen = parser.add_argument_group("Generation defaults (overridable per-request)")
    gen.add_argument("--max-new-tokens", type=int, default=256,
                     help="Maximum tokens to generate per sequence")
    gen.add_argument("--temperature", type=float, default=0.7,
                     help="Sampling temperature")
    gen.add_argument("--repetition-penalty", type=float, default=1.2,
                     help="Repetition penalty multiplier")

    # ── Branching ───────────────────────────────────────────────────────
    branch = parser.add_argument_group("Branching / UQ")
    branch.add_argument("--max-active-branches", type=int, default=10,
                        help="Maximum concurrent active branches")
    branch.add_argument("--branching-factor", type=int, default=2,
                        help="Children per branch point")
    branch.add_argument("--entropy-ema-alpha", type=float, default=0.2,
                        help="EMA smoothing for per-sequence entropy baseline")
    branch.add_argument("--relative-entropy-multiplier", type=float, default=1.15,
                        help="Base threshold = EMA × this multiplier")

    # ── Diversity pruning ───────────────────────────────────────────────
    prune = parser.add_argument_group("Diversity pruning")
    prune.add_argument("--sim-threshold", type=float, default=0.75,
                       help="Cosine-similarity ceiling before pruning")
    prune.add_argument("--ema-alpha", type=float, default=0.25,
                       help="EMA alpha for semantic-vector updates")

    # ── Exploration warmup ──────────────────────────────────────────────
    explore = parser.add_argument_group("Soft exploration warmup")
    explore.add_argument("--soft-explore-window", type=int, default=15,
                         help="Steps over which threshold ramps up")
    explore.add_argument("--soft-explore-initial", type=float, default=0.3,
                         help="Initial threshold scale (0..1, lower = more permissive)")

    # ── Paged KV-cache ──────────────────────────────────────────────────
    cache = parser.add_argument_group("Paged KV-cache")
    cache.add_argument("--block-size", type=int, default=16,
                       help="Tokens per KV-cache block")
    cache.add_argument("--max-blocks", type=int, default=8192,
                       help="Maximum blocks in KV pool")

    args = parser.parse_args()

    # Post-processing
    if args.tokenizer is None:
        args.tokenizer = args.model

    return args


# ---------------------------------------------------------------------------
# Startup
# ---------------------------------------------------------------------------

BANNER = f"""{Style.ORANGE}{Style.BOLD}
 ███╗   ███╗███████╗███╗   ██╗████████╗███████╗████████╗
 ████╗ ████║██╔════╝████╗  ██║╚══██╔══╝██╔══██║╚══██╔══╝
 ██╔████╔██║█████╗  ██╔██╗ ██║   ██║   ███████║   ██║   
 ██║╚██╔╝██║██╔══╝  ██║╚██╗██║   ██║   ██╔══██║   ██║   
 ██║ ╚═╝ ██║███████╗██║ ╚████║   ██║   ██║  ██║   ██║   
 ╚═╝     ╚═╝╚══════╝╚═╝  ╚═══╝   ╚═╝   ╚═╝  ╚═╝   ╚═╝   
{Style.RESET}{Style.CYAN}       API Server v1.0{Style.RESET}
"""


def load_model(args: argparse.Namespace):
    global _model, _tokenizer, _adapter

    dtype_map = {"float32": torch.float32, "float16": torch.float16, "bfloat16": torch.bfloat16}
    dtype = dtype_map[args.dtype]
    device = args.device
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"

    logger.info(f"Loading tokenizer from {args.tokenizer}")
    _tokenizer = AutoTokenizer.from_pretrained(
        args.tokenizer, trust_remote_code=args.trust_remote_code,
    )
    if _tokenizer.pad_token is None:
        _tokenizer.pad_token = _tokenizer.eos_token

    logger.info(f"Loading model from {args.model}  (dtype={args.dtype}, device={device})")
    _model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=dtype,
        trust_remote_code=args.trust_remote_code,
    ).to(device).eval()

    _adapter = get_adapter(_model)

    num_params = sum(p.numel() for p in _model.parameters()) / 1e6
    cfg = _model.config
    logger.info(
        f"Model ready: {num_params:.1f}M params | "
        f"{cfg.num_hidden_layers}L / {cfg.hidden_size}D / "
        f"vocab={cfg.vocab_size} | device={device}"
    )


def print_server_info(args: argparse.Namespace):
    print(BANNER)

    tmpl_label = args.chat_template
    if tmpl_label == "auto":
        tok_tmpl = getattr(_tokenizer, "chat_template", None)
        if tok_tmpl:
            tmpl_label = "auto (tokenizer built-in)"
        else:
            tmpl_label = "auto → none (tokenizer has no template)"

    info = [
        ("Model",               args.model),
        ("Dtype",               args.dtype),
        ("Device",              args.device),
        ("Chat template",       tmpl_label),
        ("Max new tokens",      args.max_new_tokens),
        ("Max active branches", args.max_active_branches),
        ("Branching factor",    args.branching_factor),
        ("Temperature",         args.temperature),
        ("Repetition penalty",  args.repetition_penalty),
        ("Entropy EMA α",       args.entropy_ema_alpha),
        ("Entropy multiplier",  args.relative_entropy_multiplier),
        ("Sim threshold",       args.sim_threshold),
        ("Block size",          args.block_size),
        ("Max blocks",          args.max_blocks),
    ]
    
    max_label = max(len(label) for label, _ in info)
    
    logger.info(f"{Style.ORANGE}--- Mentat Configuration ---{Style.RESET}")
    for label, value in info:
        # Pad the label and colorize the keys vs values
        formatted_label = f"{Style.CYAN}{label:<{max_label}}{Style.RESET}"
        logger.info(f"  {formatted_label} : {Style.BOLD}{value}{Style.RESET}")

    logger.info(f"{Style.ORANGE}----------------------------{Style.RESET}")
    logger.info(f"  Listening on {Style.BOLD}{Style.GREEN}http://{args.host}:{args.port}{Style.RESET}")
    logger.info(f"  Endpoints:")
    logger.info(f"    {Style.CYAN}POST /v1/completions{Style.RESET}       — single prompt (raw or messages)")
    logger.info(f"    {Style.CYAN}POST /v1/completions/batch{Style.RESET} — JSONL batch of prompts")
    logger.info(f"    {Style.CYAN}GET  /v1/models{Style.RESET}            — model info")
    logger.info(f"    {Style.CYAN}GET  /health{Style.RESET}               — health check")
    logger.info("")

def main():
    global _args
    _args = parse_args()

    load_model(_args)
    print_server_info(_args)

    server = HTTPServer((_args.host, _args.port), MENTATRequestHandler)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        logger.info("Shutting down...")
        server.server_close()


if __name__ == "__main__":
    main()