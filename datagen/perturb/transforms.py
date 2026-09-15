"""Answer-aware perturbations for self-generated robustness labels (PLAN.md §3.3-§3.4).

**Compliance comes first.** The competition forbids extra *labelled* data for the official
perturbation types: rephrase, rename, domain, distract, typos, expert_perturbations,
expert_no_solution. Every category below carries a compliance tag:

* `distinct`: changes what is asked or how the answer must be reported, and matches no
  official type. Safe to label.
* `confirm`: meaning-preserving surface changes that an organizer could reasonably count as
  `rephrase` or `typos`. Use them unlabelled, or labelled only after written confirmation.

The label pipeline must filter on this tag. The tag records a judgement call, and the
report must state it.

Each transform is deterministic given its `random.Random` and returns a `Variant` with the
**expected answer computed exactly**, or None when it does not apply. Returning None beats
emitting a garbled problem. `needs_verification` marks outputs whose well-posedness a
check should confirm before they are used, even though the answer is known by construction.

LLM-based categories (translation round-trip, unit systems) are intentionally absent here:
they need a model and re-solve verification, and a translation round-trip is close enough to
`rephrase` that it would start as `confirm` anyway.
"""

from __future__ import annotations

import random
import re
from collections.abc import Callable
from dataclasses import dataclass, field


@dataclass(frozen=True)
class Variant:
    category: str
    problem: str
    expected_answer: int
    params: dict = field(default_factory=dict)
    needs_verification: bool = False


@dataclass(frozen=True)
class Category:
    transform: Callable[[str, int, random.Random], Variant | None]
    compliance: str  # "distinct" | "confirm"
    rationale: str


# ---------------------------------------------------------------------------
# answer_modulus: change the requested output, answer computed exactly
# ---------------------------------------------------------------------------

# Prime or round moduli, large enough that guessing a residue stays unlikely.
LARGE_MODULI = (97, 100, 101, 103, 107, 109, 113)
SMALL_MODULI = (13, 17, 19, 23)


def answer_modulus(problem: str, answer: int, rng: random.Random) -> Variant | None:
    """Ask for the answer's remainder mod m, with m <= answer so the result actually changes."""
    if answer < SMALL_MODULI[0]:
        return None  # residues of tiny answers are either unchanged or trivially guessable
    pool = [m for m in LARGE_MODULI if m <= answer] or [m for m in SMALL_MODULI if m <= answer]
    modulus = rng.choice(pool)
    text = problem.rstrip() + (
        f"\n\nOnce you have found the answer, report only its remainder when divided by {modulus}."
    )
    return Variant(
        category="answer_modulus",
        problem=text,
        expected_answer=answer % modulus,
        params={"modulus": modulus, "chance_rate": round(1 / modulus, 4)},
    )


# ---------------------------------------------------------------------------
# latex_to_plaintext: notation change, answer unchanged
# ---------------------------------------------------------------------------

_UNSUPPORTED = re.compile(r"\\(begin|end|sum|prod|int|lim|array|matrix|cases|mathbb|mathcal|tag|label)(?![A-Za-z])")

SYMBOLS = {
    "cdot": "*", "times": "*", "div": "/",
    "le": "<=", "leq": "<=", "leqslant": "<=", "ge": ">=", "geq": ">=", "geqslant": ">=",
    "ne": "!=", "neq": "!=", "pm": "+/-", "ldots": "...", "cdots": "...", "dots": "...",
    "infty": "infinity", "pi": "pi", "alpha": "alpha", "beta": "beta", "gamma": "gamma",
    "delta": "delta", "theta": "theta", "lambda": "lambda", "mu": "mu", "sigma": "sigma",
    "phi": "phi", "omega": "omega", "log": "log", "ln": "ln", "sin": "sin", "cos": "cos",
    "tan": "tan", "angle": "angle ", "triangle": "triangle ", "in": " in ", "mid": " | ",
    "bmod": " mod ", "mod": " mod ", "lfloor": "floor(", "rfloor": ")", "lceil": "ceil(",
    "rceil": ")", "left": "", "right": "", "quad": " ", "qquad": " ", "displaystyle": "",
    "cup": " union ", "cap": " intersect ", "ell": "l", "prime": "'",
}


class _Unsupported(Exception):
    pass


def _group(content: str) -> str:
    content = content.strip()
    return content if re.fullmatch(r"[A-Za-z0-9.]+", content) else f"({content})"


_STRUCTURAL: list[tuple[str, Callable[[re.Match], str] | str]] = [
    (r"\^\s*\{?\s*\\circ\s*\}?", " degrees"),
    (r"\\(?:text|textbf|textit|mathrm|mathbf|mathit|operatorname|mbox|overline)\s*\{([^{}]*)\}", r"\1"),
    (r"\\[dt]?frac\s*\{([^{}]*)\}\s*\{([^{}]*)\}", lambda m: f"({_group(m.group(1))}/{_group(m.group(2))})"),
    (r"\\binom\s*\{([^{}]*)\}\s*\{([^{}]*)\}", lambda m: f"C({m.group(1).strip()}, {m.group(2).strip()})"),
    (r"\\sqrt\s*\[([^\]]*)\]\s*\{([^{}]*)\}", lambda m: f"({m.group(2).strip()})^(1/{m.group(1).strip()})"),
    (r"\\sqrt\s*\{([^{}]*)\}", lambda m: f"sqrt({m.group(1).strip()})"),
    (r"\\pmod\s*\{([^{}]*)\}", r" (mod \1)"),
    (r"([\^_])\s*\{([^{}]*)\}", lambda m: m.group(1) + _group(m.group(2))),
]


def latex_to_plaintext(problem: str, answer: int, rng: random.Random) -> Variant | None:
    """Rewrite inline LaTeX as plain ASCII math; None if anything unsupported remains."""
    if "$" not in problem and "\\" not in problem:
        return None
    if _UNSUPPORTED.search(problem) or "&" in problem or "[asy]" in problem:
        return None  # Asymptote figure code is not LaTeX and would be mangled

    text = problem.replace("\\\\", "\n")  # LaTeX line break
    for escaped, replacement in (
        ("\\{", "\x01"), ("\\}", "\x02"), ("\\$", "\x03"), ("\\%", "%"),
        ("\\,", " "), ("\\;", " "), ("\\!", ""), ("\\ ", " "),
    ):
        text = text.replace(escaped, replacement)

    # Innermost-first: repeat until no structural rule fires, so nesting unwinds.
    while True:
        before = text
        for pattern, replacement in _STRUCTURAL:
            text = re.sub(pattern, replacement, text)
        if text == before:
            break

    text = re.sub(r"\\\[|\\\]", "\n", text)
    text = re.sub(r"\\\(|\\\)", "", text)

    def symbol(match: re.Match) -> str:
        if match.group(1) not in SYMBOLS:
            raise _Unsupported(match.group(1))
        return SYMBOLS[match.group(1)]

    try:
        text = re.sub(r"\\([A-Za-z]+)", symbol, text)
    except _Unsupported:
        return None

    text = text.replace("$", "").replace("{", "").replace("}", "")
    if "\\" in text:
        return None
    text = text.replace("\x01", "{").replace("\x02", "}").replace("\x03", "$")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r" *\n *", "\n", text).strip()
    if text == problem.strip():
        return None
    # Mechanical, but operator precedence in plain text can read ambiguously; spot-check.
    return Variant("latex_to_plaintext", text, answer, {}, needs_verification=True)


# ---------------------------------------------------------------------------
# premise_reorder: same premises, different order, question last
# ---------------------------------------------------------------------------

_MATH_SPAN = re.compile(r"\$\$.*?\$\$|\$[^$]*\$|\\\[.*?\\\]|\\\(.*?\\\)", re.S)
_QUESTION_CUE = re.compile(r"\?|\b(find|compute|determine|what|how many|evaluate)\b", re.I)
# Sentences opening like this lean on an earlier sentence, so they stay in place.
_ANAPHORA = {
    "then", "it", "this", "these", "those", "such", "they", "also", "moreover", "furthermore",
    "here", "similarly", "thus", "hence", "so", "additionally", "finally", "its", "their",
    "now", "next", "again", "otherwise", "however", "but", "and", "each", "both", "there",
}


def split_sentences(text: str) -> list[str]:
    """Sentence split that never cuts inside math."""
    spans: list[str] = []

    def hide(match: re.Match) -> str:
        spans.append(match.group(0))
        return f"\x00{len(spans) - 1}\x00"

    hidden = _MATH_SPAN.sub(hide, text.strip())
    parts = re.split(r"(?<=[.!?])\s+(?=[A-Z\x00])", hidden)
    return [
        re.sub(r"\x00(\d+)\x00", lambda m: spans[int(m.group(1))], part).strip()
        for part in parts
        if part.strip()
    ]


def _first_word(sentence: str) -> str:
    match = re.match(r"[A-Za-z']+", sentence)
    return match.group(0).lower() if match else ""


# Sentences just before the question that set it up belong to it: hypotheticals ("Suppose ...")
# and answer-format statements ("... can be written as m/n, where m and n are coprime").
_QUESTION_LEADS = {"suppose", "if", "assume", "given", "when", "consider"}
_ANSWER_FORM = re.compile(
    r"can be (written|expressed)|in the form|relatively prime|in simplest form|where \S+ and \S+ are",
    re.I,
)


def _belongs_to_question(sentence: str) -> bool:
    return _first_word(sentence) in _QUESTION_LEADS or bool(_ANSWER_FORM.search(sentence))


def premise_reorder(problem: str, answer: int, rng: random.Random) -> Variant | None:
    """Permute free-standing premise sentences; the question (and its lead-in) stays last."""
    if "\n\n" in problem.strip() or "\\[" in problem or "$$" in problem or "[asy]" in problem:
        return None  # multi-paragraph, display math, or figure code: sentence structure is unreliable
    sentences = split_sentences(problem)
    if len(sentences) < 3 or not _QUESTION_CUE.search(sentences[-1]):
        return None
    question_start = len(sentences) - 1
    while question_start > 0 and _belongs_to_question(sentences[question_start - 1]):
        question_start -= 1
    premises, question = sentences[:question_start], sentences[question_start:]
    movable = [i for i, s in enumerate(premises) if _first_word(s) not in _ANAPHORA]
    if len(movable) < 2:
        return None

    order = movable[:]
    for _ in range(10):
        rng.shuffle(order)
        if order != movable:
            break
    else:
        return None
    reordered = premises[:]
    for position, source in zip(movable, order):
        reordered[position] = premises[source]
    text = " ".join([*reordered, *question])
    return Variant(
        "premise_reorder",
        text,
        answer,
        {"moved_positions": movable, "new_order": order},
        needs_verification=True,  # a definition can now follow its first use
    )


CATEGORIES: dict[str, Category] = {
    "answer_modulus": Category(
        answer_modulus,
        "distinct",
        "Changes the requested output (a residue); no official type alters what is reported.",
    ),
    "latex_to_plaintext": Category(
        latex_to_plaintext,
        "confirm",
        "Meaning-preserving notation change; could be read as rephrase or typos.",
    ),
    "premise_reorder": Category(
        premise_reorder,
        "confirm",
        "Same premises in a different order; could be read as rephrase.",
    ),
}
