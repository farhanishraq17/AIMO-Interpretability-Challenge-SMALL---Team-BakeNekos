"""Final-answer extraction and grading for sampled solutions (E7).

Olympiad answers in the pool are single integers, so grading is exact integer match. The
extractor looks, in order, for:

1. the last `\\boxed{...}` (or `\\fbox{...}`) after the reasoning block, with braces balanced;
2. an `ANSWER: ...` line, the probe baseline's system-prompt format;
3. a "final answer is ..." phrase.

Anything that does not normalise to an integer counts as no answer, which is graded wrong.
"""

from __future__ import annotations

import re

_BOX_COMMANDS = ("\\boxed", "\\fbox")
_ANSWER_LINE = re.compile(r"ANSWER:\s*([^\n]+)", re.I)
_FINAL_PHRASE = re.compile(r"final answer\s*(?:is|:)\s*([^\n.]+)", re.I)


def last_boxed(text: str) -> str | None:
    """Content of the last \\boxed{...}, respecting nested braces."""
    start = max(text.rfind(command) for command in _BOX_COMMANDS)
    if start < 0:
        return None
    brace = text.find("{", start)
    if brace < 0:
        return None
    depth = 0
    for index in range(brace, len(text)):
        if text[index] == "{":
            depth += 1
        elif text[index] == "}":
            depth -= 1
            if depth == 0:
                return text[brace + 1 : index]
    return None  # unbalanced: the generation was probably cut off


def normalize_integer(raw: str | None) -> int | None:
    """'$1{,}024$', '\\text{42}', 'n = 7', '0.0', '7^\\circ' -> int; anything else -> None."""
    if raw is None:
        return None
    text = raw.strip()
    text = re.sub(r"\\(?:text|mathrm|mathbf|textbf)\s*\{([^{}]*)\}", r"\1", text)
    text = text.replace("{,}", ",").replace("\\!", "").replace("\\,", "").replace("$", "")
    text = re.sub(r"\^\s*\{?\\circ\}?|\\%|%", "", text)
    text = re.sub(r"^[A-Za-z]\w*\s*=\s*", "", text)  # "n = 7"
    text = text.strip().rstrip(".").strip()
    if re.fullmatch(r"-?\d{1,3}(,\d{3})+", text):
        text = text.replace(",", "")
    if re.fullmatch(r"-?\d+\.0*", text):
        text = text.split(".")[0]
    return int(text) if re.fullmatch(r"-?\d+", text) else None


def extract_answer(text: str) -> int | None:
    """Integer final answer of one completion, or None."""
    if not text:
        return None
    # Reasoning models often box intermediate results while thinking; prefer the conclusion.
    tail = text.rsplit("</think>", 1)[-1] if "</think>" in text else text
    for candidate in (tail, text):
        boxed = last_boxed(candidate)
        if boxed is not None:
            return normalize_integer(boxed)
    for pattern in (_ANSWER_LINE, _FINAL_PHRASE):
        matches = pattern.findall(tail)
        if matches:
            return normalize_integer(matches[-1])
    return None
