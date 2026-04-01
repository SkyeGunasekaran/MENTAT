import math
import re
import logging
from typing import Any

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Built-in chat templates
# ---------------------------------------------------------------------------
BUILTIN_TEMPLATES: dict[str, str] = {
    "chatml": (
        "{% for message in messages %}"
        "<|im_start|>{{ message['role'] }}\n"
        "{{ message['content'] }}<|im_end|>\n"
        "{% endfor %}"
        "{% if add_generation_prompt %}"
        "<|im_start|>assistant\n"
        "{% endif %}"
    ),
    "llama3": (
        "{% for message in messages %}"
        "<|start_header_id|>{{ message['role'] }}<|end_header_id|>\n\n"
        "{{ message['content'] }}<|eot_id|>"
        "{% endfor %}"
        "{% if add_generation_prompt %}"
        "<|start_header_id|>assistant<|end_header_id|>\n\n"
        "{% endif %}"
    ),
    "mistral": (
        "{% for message in messages %}"
        "{% if message['role'] == 'user' %}"
        "[INST] {{ message['content'] }} [/INST]"
        "{% elif message['role'] == 'assistant' %}"
        "{{ message['content'] }}</s> "
        "{% endif %}"
        "{% endfor %}"
    ),
    "zephyr": (
        "{% for message in messages %}"
        "<|{{ message['role'] }}|>\n"
        "{{ message['content'] }}<|endoftext|>\n"
        "{% endfor %}"
        "{% if add_generation_prompt %}"
        "<|assistant|>\n"
        "{% endif %}"
    ),
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
    "none": "",
}

# ---------------------------------------------------------------------------
# Chat Template Handling
# ---------------------------------------------------------------------------
def resolve_chat_template(tokenizer, spec: str) -> str | None:
    """Resolve the effective chat template string."""
    if spec == "none":
        return None

    if spec == "auto":
        tok_tmpl = getattr(tokenizer, "chat_template", None)
        if tok_tmpl:
            if isinstance(tok_tmpl, dict):
                tmpl = tok_tmpl.get("default") or next(iter(tok_tmpl.values()))
                logger.info(f"Chat template: tokenizer (named, using '{next(iter(tok_tmpl))}')")
                return tmpl
            logger.info("Chat template: tokenizer built-in")
            return tok_tmpl
        logger.info("Chat template: tokenizer has none -> raw completion mode")
        return None

    if spec in BUILTIN_TEMPLATES:
        tmpl = BUILTIN_TEMPLATES[spec]
        if not tmpl:
            return None
        logger.info(f"Chat template: built-in '{spec}'")
        return tmpl

    logger.info("Chat template: user-supplied Jinja2 string")
    return spec

def format_prompt_from_messages(tokenizer, chat_template: str | None, messages: list[dict]) -> str:
    """Format a list of message dicts into a model-ready string."""
    if chat_template is not None:
        old_template = getattr(tokenizer, "chat_template", None)
        try:
            tokenizer.chat_template = chat_template
            return tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
        finally:
            tokenizer.chat_template = old_template

    # Fallback raw concatenation if no template
    parts = []
    for msg in messages:
        role, content = msg.get("role"), msg.get("content")
        if role == "system": parts.append(content)
        elif role == "user": parts.append(f"User: {content}")
        elif role == "assistant": parts.append(f"Assistant: {content}")
        else: parts.append(content)
    parts.append("Assistant:")
    return "\n".join(parts)

# ---------------------------------------------------------------------------
# Text & Answer Extraction Utilities
# ---------------------------------------------------------------------------
def extract_answer(text: str) -> str | None:
    """Try to pull a final answer from model output."""
    m = re.search(r"\\boxed\{([^{}]*(?:\{[^{}]*\}[^{}]*)*)\}", text)
    if m: return m.group(1).strip()

    m = re.search(r"<answer>(.*?)</answer>", text, re.DOTALL)
    if m: return m.group(1).strip()

    m = re.search(r"####\s*(.+?)(?:\n|$)", text)
    if m: return m.group(1).strip()

    return None

def safe_float(x: Any) -> Any:
    """Safely convert to rounded float, handling NaNs and Infs."""
    if isinstance(x, float) and (math.isnan(x) or math.isinf(x)):
        return None
    return round(x, 6) if isinstance(x, float) else x