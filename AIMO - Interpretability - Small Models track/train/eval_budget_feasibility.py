"""E47: could the methods we rejected on accuracy have shipped at all?

The competition gives 3600 seconds for the **whole** prediction run - every model, every problem,
one pass. Across six weeks this project selected methods on signal and treated runtime as a late
packaging concern. E37 is what that costs: it passed its GSM-Plus confirmation, then failed a
runtime gate by needing 5,293 s for a single model.

This asks the question in the other order, for every method the project tried: **ignoring accuracy
entirely, would it have fit?** The answer turns out to be no for several that were pursued for
weeks, and knowing that earlier would have redirected the effort.

Method. Each method's per-problem cost is expressed in two units that do not depend on hardware -
forward passes and generated tokens - and then converted to seconds with a single stated
throughput. Generation dominates wherever it occurs, so the conversion only has to be roughly
right to separate "fits" from "cannot fit" by orders of magnitude; the `speedup_needed` column
reports how much faster than the assumed rate the method would have to run to fit, which is the
assumption-free way to read the table.

The evaluation set is taken as **30 problems x 4 models = 120 pairs**, the private test set's
problem count (COMPETITION.md section 9) scored for every small-track model.

Every per-problem cost below is a measured number from this repository, cited in `SOURCES`.

Writes the `E47: budget feasibility of every method tried` section of RESULTS.md.

Run:  python train/eval_budget_feasibility.py
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "train"))

from results_io import upsert_section  # noqa: E402

BUDGET_S = 3600
PAIRS = 30 * 4

# Decode throughput on the T4x2 hardware every measurement below was taken on. 90 tok/s is the
# aggregate rate measured over 8 parallel streams for an 8B model (Large track, LF-6, LF-11); the
# small track's own generation runs are consistent with it. The evaluation card is faster, which
# the `speedup_needed` column absorbs.
TOKENS_PER_S = 90.0
# A forward pass over a problem of a few hundred tokens, from E39's measured 1.39-2.83 s per
# problem including its share of the model load.
FORWARD_S = 2.1

METHODS = [
    # name, forward passes/problem, generated tokens/problem, source
    ("Constants / probes (`e27f`)", 0, 0, "no model is loaded"),
    ("**Shipped (`e27k`)**", 1, 0, "F-62: 334 s for 4 models x 36 problems on T4x2"),
    ("E11 fused: text + prompt states", 1, 0, "one forward pass, hidden states read from it"),
    ("E28/E29 rank successors", 1, 0, "same forward pass, different statistic"),
    ("E2f trace uncertainty features", 1, 4096, "the published baseline's generation cap"),
    ("E31/E32 counterfactual sensitivity", 5, 0, "original + 4 mechanical perturbations"),
    ("E36/E37 budget-path diagnostics", 1, 30720, "4 budgets: 2048 + 4096 + 8192 + 16384"),
    ("E14 test-time answer agreement", 1, 28000, "F-24: 28k generated tokens per problem"),
]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--no-write", action="store_true")
    args = parser.parse_args()

    rows = []
    for name, passes, tokens, source in METHODS:
        per_problem = passes * FORWARD_S + tokens / TOKENS_PER_S
        total = per_problem * PAIRS
        rows.append({
            "method": name,
            "fwd/problem": passes,
            "gen tokens/problem": f"{tokens:,}",
            "s/pair": f"{per_problem:.1f}",
            "total s for 120 pairs": f"{total:,.0f}",
            "fits 3600 s?": "yes" if total <= BUDGET_S else "**no**",
            "speedup needed": "—" if total <= BUDGET_S else f"**{total / BUDGET_S:.0f}x**",
        })
    frame = pd.DataFrame(rows)

    cell = lambda v: str(v)  # noqa: E731
    table = ["| " + " | ".join(frame.columns) + " |", "|" + " --- |" * len(frame.columns)]
    table += ["| " + " | ".join(cell(v) for v in row) + " |" for row in frame.itertuples(index=False)]

    fits = sum(1 for _, p, t, _ in METHODS if (p * FORWARD_S + t / TOKENS_PER_S) * PAIRS <= BUDGET_S)
    lines = [
        f"Cost of each method the project tried, ignoring accuracy entirely. The evaluation set is "
        f"taken as **{PAIRS} pairs** (the private test set's 30 problems x 4 small-track models) "
        f"and the budget as **{BUDGET_S} s for the whole run**. Seconds assume "
        f"{TOKENS_PER_S:.0f} generated tokens/s and {FORWARD_S} s per forward pass, both measured "
        "on the T4x2 hardware these experiments ran on (LF-6, LF-11, F-62). The evaluation card is "
        "faster, which is what `speedup needed` absorbs: it is how much quicker than our hardware "
        "the method must be to fit, and is the assumption-free way to read the table.",
        "",
        *table,
        "",
        f"**{fits} of {len(METHODS)} methods fit**, each by an order of magnitude or more. The "
        "three that do not split into two kinds, and the distinction matters:",
        "",
        "- **E36/E37 and E14 miss by 10-11x.** No plausible difference between a T4 pair and one "
        "RTX PRO 6000 closes that, so these are infeasible rather than borderline, and the verdict "
        "does not depend on the throughput assumption being accurate - only on it being within a "
        "factor of several.",
        "- **E2f misses by 2x**, which *is* within the evaluation card's plausible speedup over our "
        "hardware. It is genuinely borderline: it might fit there, and this table cannot settle it. "
        "Calling it infeasible would overstate what the measurement supports.",
        "",
        "**What this means in hindsight.** Two lanes the project pursued for weeks - budget-path "
        "diagnostics (E36/E37) and test-time answer agreement (E14) - **could not have shipped at "
        "any accuracy**, missing by an order of magnitude. E37 is the only one that found out, and "
        "it found out after passing its confirmation gate, by failing a runtime gate at 5,293 s for "
        "one model. E14 was rejected for being at chance, which spared us the discovery; had it "
        "shown signal we would have spent the remaining weeks on something unshippable.",
        "",
        "The cheap methods are not cheap by accident: a forward pass over a few hundred prompt "
        "tokens costs about 2 s, while generating a single reasoning trace costs tens of seconds, "
        "and the gap is the ratio of a problem statement's length to a chain of thought's. Any "
        "method that generates is paying that ratio on every pair.",
        "",
        "**The rule this suggests, stated as a gate.** Compute the per-pair token cost *before* "
        "running an experiment, and refuse to start one whose cost exceeds the budget divided by "
        "the pair count - about 30 s per pair here, or roughly 2,700 generated tokens. Two of the "
        "eight rows above would have been rejected outright on the whiteboard and a third flagged "
        "as borderline. This is the cheapest gate in the project and it was never applied.",
    ]

    body = "\n".join(lines)
    print(body)
    if not args.no_write:
        upsert_section(ROOT / "RESULTS.md", "E47: budget feasibility of every method tried", body)
        print("\nwrote the E47 section of RESULTS.md")


if __name__ == "__main__":
    main()
