"""Deterministic answer-preserving rewrites shared by E14 and E31."""

from __future__ import annotations

import random
import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE / "perturb"))

from transforms import _MATH_SPAN, latex_to_plaintext, premise_reorder  # noqa: E402

_LONG_WORD = re.compile(r"\b[A-Za-z]{6,}\b")
DISTRACTORS = (
    "Note that this problem appeared in a regional competition held in the spring.",
    "A student once spent 45 minutes on a similar question.",
    "Recall that 2024 is divisible by 11.",
    "Some solvers find it helpful to draw a diagram first.",
    "The number 7 is the fourth prime number.",
)


def typos(problem: str, rng: random.Random) -> str | None:
    math = [match.span() for match in _MATH_SPAN.finditer(problem)]
    words = [
        match
        for match in _LONG_WORD.finditer(problem)
        if not any(start <= match.start() < end for start, end in math)
    ]
    if len(words) < 3:
        return None
    pieces, last = [], 0
    for match in sorted(rng.sample(words, 3), key=lambda item: item.start()):
        word = match.group()
        index = rng.randrange(1, len(word) - 2)
        pieces += [
            problem[last : match.start()],
            word[:index] + word[index + 1] + word[index] + word[index + 2 :],
        ]
        last = match.end()
    return "".join(pieces) + problem[last:]


def distract(problem: str, rng: random.Random) -> str:
    return f"{problem.rstrip()} {rng.choice(DISTRACTORS)}"


def rewrites(problem: str) -> dict[str, str]:
    """Return deterministic applicable rewrites as ``kind -> problem``."""
    candidates = {
        "latex_to_plaintext": getattr(
            latex_to_plaintext(problem, 0, random.Random(problem)), "problem", None
        ),
        "premise_reorder": getattr(
            premise_reorder(problem, 0, random.Random(problem)), "problem", None
        ),
        "typos": typos(problem, random.Random(problem)),
        "distract": distract(problem, random.Random(problem)),
    }
    return {key: value for key, value in candidates.items() if value and value.strip() != problem.strip()}
