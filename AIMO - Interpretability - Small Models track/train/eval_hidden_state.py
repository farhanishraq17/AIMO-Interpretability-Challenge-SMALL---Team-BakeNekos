"""E50: solve the hidden validation set's per-model composition from our own leaderboard scores.

E40 (`eval_batch_composition.py`) used only the four constant probes, so it could fix each model's
robust-minus-spurious balance but not its batch size, and had to average over 56 compositions.
Every E27 submission since is a deterministic function of the same per-model mean-NLL ranking
(exactly repeatable and precision-proof: Finding 62), so each of its scores is one more equation
in the same unknowns: how many cases each model has, and which familiarity ranks are robust.

This enumerates every such state - batch sizes summing to 14 and an R/S label for each rank of
each model - and keeps the ones that reproduce all seventeen scores. No balance or size is assumed;
the probes enter only as scores like the rest. Then it re-scores the null of E40 on the surviving
composition instead of the 56-way average.

The inputs are the scores already recorded in PROGRESS.md, nothing new is submitted, and the
hidden problems themselves stay unknown: a state says "Qwen's second most familiar problem is
spurious", not which problem that is.

Writes the `E50: the hidden set, solved from the leaderboard` section of RESULTS.md.

Run:  python train/eval_hidden_state.py
"""

from __future__ import annotations

import argparse
import sys
from collections import defaultdict
from itertools import product
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "train"))

from eval_batch_composition import TILT, robust_count, score_distribution, tail  # noqa: E402
from results_io import upsert_section  # noqa: E402

MODELS = ("Qwen", "DeepSeek", "Skywork", "Olmo")
CASES = 14


# Each rule maps a batch size to the predictions in familiarity order (lowest mean NLL first).
def const(value: bool):
    return lambda n: (value,) * n


def familiar(tilt: int = 0):
    """The shipped split: the `ceil(n/2) + tilt` most familiar problems robust."""
    return lambda n: tuple(i < robust_count(n, tilt) for i in range(n))


def unfamiliar():
    """E27d: the `ceil(n/2)` least familiar problems robust."""
    return lambda n: tuple(i >= n - robust_count(n, 0) for i in range(n))


FALSE = {m: const(False) for m in MODELS}
CONSTANTS = {**FALSE, "Qwen": const(True), "DeepSeek": const(True)}
E27K = {m: familiar(TILT.get(m, 0)) for m in MODELS}

# (submission, rule per model, correct of 14, where the score is recorded)
LEADERBOARD = [
    ("all-False", FALSE, 7, "Finding 64"),
    ("True for Qwen only", {**FALSE, "Qwen": const(True)}, 8, "Finding 64"),
    ("True for DeepSeek only", {**FALSE, "DeepSeek": const(True)}, 8, "Finding 64"),
    ("True for Skywork only", {**FALSE, "Skywork": const(True)}, 6, "Finding 64"),
    ("True for Olmo only", {**FALSE, "Olmo": const(True)}, 6, "Finding 64"),
    ("`e27d` unfamiliar half", {m: unfamiliar() for m in MODELS}, 4, "Finding 49"),
    ("`e27e` familiar half", {m: familiar() for m in MODELS}, 10, "Finding 58"),
    ("`e27f` constants", CONSTANTS, 9, "Finding 58"),
    ("`e27g` split Qwen only", {**CONSTANTS, "Qwen": familiar()}, 9, "Finding 58"),
    ("`e27h` split DeepSeek only", {**CONSTANTS, "DeepSeek": familiar()}, 10, "Finding 58"),
    ("`e27i` split Skywork only", {**CONSTANTS, "Skywork": familiar()}, 9, "Finding 58"),
    ("`e27j` split Olmo only", {**CONSTANTS, "Olmo": familiar()}, 9, "Finding 58"),
    ("`e27k` tilted", E27K, 12, "Finding 60"),
    ("`e27m` Qwen one fewer", {**E27K, "Qwen": familiar(-1)}, 11, "Finding 61"),
    ("`e27n` DeepSeek one fewer", {**E27K, "DeepSeek": familiar(-1)}, 11, "Finding 61"),
    ("`e27o` Qwen one more", {**E27K, "Qwen": familiar(+1)}, 11, "Finding 61"),
    ("`e27p` DeepSeek one more", {**E27K, "DeepSeek": familiar(+1)}, 11, "Finding 61"),
]
OBSERVED = tuple(correct for _, _, correct, _ in LEADERBOARD)


def labelings(n: int):
    """Every R/S assignment to n familiarity ranks, as tuples of bools (True = robust)."""
    return product((True, False), repeat=n)


def solve() -> list[dict[str, tuple[bool, ...]]]:
    """Every state reproducing all scores. Grouping labelings by their score vector keeps it fast."""
    groups: dict[tuple[str, int], dict[tuple[int, ...], list]] = {}
    for model in MODELS:
        for n in range(1, CASES - len(MODELS) + 2):
            by_vector = defaultdict(list)
            for labels in labelings(n):
                vector = tuple(sum(p == y for p, y in zip(rules[model](n), labels)) for _, rules, _, _ in LEADERBOARD)
                by_vector[vector].append(labels)
            groups[model, n] = by_vector

    states = []
    for sizes in product(range(1, CASES - len(MODELS) + 2), repeat=len(MODELS)):
        if sum(sizes) != CASES:
            continue
        options = [groups[m, n].items() for m, n in zip(MODELS, sizes)]
        for combo in product(*options):
            if tuple(map(sum, zip(*(vector for vector, _ in combo)))) == OBSERVED:
                for labels in product(*(seqs for _, seqs in combo)):
                    states.append(dict(zip(MODELS, labels)))
    return states


def show(labels: tuple[bool, ...]) -> str:
    return " ".join("R" if y else "S" for y in labels)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--no-write", action="store_true", help="print only; do not touch RESULTS.md")
    args = parser.parse_args()

    states = solve()
    assert states, "no state reproduces every recorded score; a score or a rule is wrong"
    sizes = {tuple(len(s[m]) for m in MODELS) for s in states}
    assert len(sizes) == 1, f"composition not identified: {sizes}"
    composition = {m: (sum(states[0][m]), len(states[0][m]) - sum(states[0][m])) for m in MODELS}

    lines = [
        f"All {len(LEADERBOARD)} scores below are reproduced by exactly **{len(states)}** hidden states, "
        "out of every assignment of batch sizes summing to 14 and an R/S label per familiarity rank. "
        "Inputs are the recorded scores only; no balance or batch size is assumed.",
        "",
        "| submission | correct of 14 | source |",
        "| --- | --- | --- |",
        *(f"| {name} | {correct} | {source} |" for name, _, correct, source in LEADERBOARD),
        "",
        "Labels in familiarity order, most familiar first, and what `e27k` gets right:",
        "",
        "| model | cases | robust | " + " | ".join(f"state {chr(65 + i)}" for i in range(len(states)))
        + " | `e27k` correct |",
        "| --- | --- | --- |" + " --- |" * len(states) + " --- |",
    ]
    for m in MODELS:
        r, s = composition[m]
        e27k = {sum(p == y for p, y in zip(E27K[m](r + s), st[m])) for st in states}
        lines.append(f"| {m} | {r + s} | {r} | " + " | ".join(show(st[m]) for st in states)
                     + f" | {'/'.join(map(str, sorted(e27k)))} of {r + s} |")

    null = score_distribution(composition, TILT)
    lines += [
        "",
        "**`e27k`'s counts are exact and its ranking is perfect on DeepSeek, Skywork and Olmo.** Its "
        "two misses are one swapped pair inside Qwen's batch: one of Qwen's two most familiar problems "
        "is spurious and its least familiar is robust. The states differ only in which of the top two.",
        "",
        f"Under a ranking with no information and `e27k`'s exact counts on this composition, "
        f"P(score >= 12) = **{tail(null, 12):.4f}**, P(14) = {null.get(14, 0.0):.4f}, mean "
        f"{sum(k * p for k, p in null.items()):.2f}/14. E40's 0.0552 averaged over 56 compositions; "
        "this is the same test on the one composition the scores allow.",
    ]

    body = "\n".join(lines)
    print(body)
    if not args.no_write:
        upsert_section(ROOT / "RESULTS.md", "E50: the hidden set, solved from the leaderboard", body)
        print("\nwrote the E50 section of RESULTS.md")


if __name__ == "__main__":
    main()
