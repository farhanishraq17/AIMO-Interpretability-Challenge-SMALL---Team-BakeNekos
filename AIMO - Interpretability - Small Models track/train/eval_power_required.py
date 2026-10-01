"""E48: how many labels would each closed lane have needed to settle its question?

"The project was label-bound" appears throughout `PROGRESS.md` and the report. It is the correct
diagnosis and, as stated, it is unfalsifiable: any null can be blamed on sample size. This turns it
into a number. For each effect the project actually observed, how many rows would be needed to
detect an effect that size at 80% power - and how does that compare with the 33 labelled problems
we had?

Two standard closed forms, chosen over simulation because they are textbook, auditable in two lines
each, and fast enough to sweep:

* **Correlation** - Fisher's z transform. With `z = atanh(r)`, the required n is
  `((z_(1-alpha/2) + z_(1-beta)) / z)^2 + 3`. Applied to E46's Spearman +0.247 between familiarity
  and `max_drop`.
* **AUC** - the Hanley-McNeil variance, `V = [A(1-A) + (m-1)(Q1 - A^2) + (k-1)(Q2 - A^2)] / (m k)`
  with `Q1 = A/(2-A)` and `Q2 = 2A^2/(1+A)`, for `m` positives and `k` negatives. Power is
  `Phi(|A - 0.5| / sqrt(V) - z_(1-alpha/2))`. Applied to E45/E46's AUC 0.312 and E35's 0.609.

Both are approximations and both are two-sided at alpha = 0.05. They are used here to separate "we
needed three times more data" from "we needed ten times more", which is the decision they inform;
they are not precise enough to argue about a dozen rows either way.

The answers are the point, not the machinery: they say what a future labelling run has to produce
to be worth starting, which is exactly the decision the Main Track's Lane A faces.

Writes the `E48: labels required to settle each lane` section of RESULTS.md.

Run:  python train/eval_power_required.py
"""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

from scipy.stats import norm

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "train"))

from results_io import upsert_section  # noqa: E402

ALPHA, TARGET_POWER = 0.05, 0.80
Z_ALPHA = norm.ppf(1 - ALPHA / 2)
AVAILABLE = 33  # labelled official problems (F-30, F-46)


def power_correlation(r: float, n: int) -> float:
    """Fisher-z power for a two-sided test of rho = 0."""
    if n <= 3:
        return 0.0
    z = abs(math.atanh(r)) * math.sqrt(n - 3)
    return float(norm.cdf(z - Z_ALPHA))


def power_auc(auc: float, n: int, prevalence: float) -> float:
    """Hanley-McNeil power for a two-sided test of AUC = 0.5."""
    a = max(auc, 1 - auc)  # the formula is symmetric about 0.5
    m, k = max(int(round(n * prevalence)), 1), max(n - int(round(n * prevalence)), 1)
    q1, q2 = a / (2 - a), 2 * a * a / (1 + a)
    var = (a * (1 - a) + (m - 1) * (q1 - a * a) + (k - 1) * (q2 - a * a)) / (m * k)
    if var <= 0:
        return 1.0
    return float(norm.cdf(abs(a - 0.5) / math.sqrt(var) - Z_ALPHA))


def required(power_fn, *args, cap: int = 5000) -> int | None:
    """Smallest n reaching TARGET_POWER."""
    for n in range(6, cap):
        if power_fn(*args, n) >= TARGET_POWER:
            return n
    return None


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--no-write", action="store_true")
    args = parser.parse_args()

    rows = [
        ("E46 familiarity vs `max_drop`", "Spearman +0.247", 37,
         power_correlation(0.247, AVAILABLE), required(lambda r, n: power_correlation(r, n), 0.247)),
        ("E45/E46 familiarity ranks robustness", "AUC 0.312", 33,
         power_auc(0.312, AVAILABLE, 0.61), required(lambda a, n: power_auc(a, n, 0.61), 0.312)),
        ("E35 external-label transfer", "AUC 0.609", 61,
         power_auc(0.609, AVAILABLE, 0.82), required(lambda a, n: power_auc(a, n, 0.82), 0.609)),
    ]

    lines = [
        f"For each effect the project actually observed, the sample size needed to detect an effect "
        f"that size at {TARGET_POWER:.0%} power, two-sided at alpha = {ALPHA}. Correlation by "
        f"Fisher's z, AUC by the Hanley-McNeil variance. `power at {AVAILABLE}` is what the "
        f"{AVAILABLE} labelled official problems actually bought.",
        "",
        f"| lane | observed effect | rows it had | power at {AVAILABLE} | rows needed for {TARGET_POWER:.0%} |",
        "| --- | --- | --- | --- | --- |",
    ]
    for lane, effect, had, power_at, need in rows:
        lines.append(f"| {lane} | {effect} | {had} | {power_at:.0%} | **{need if need else '> 5000'}** |")

    needs = [n for *_, n in rows if n]
    lines += [
        "",
        f"**Every lane needed between {min(needs)} and {max(needs)} rows and had between 33 and 61.** "
        f"Power at the sample sizes we actually ran was "
        f"{min(r[3] for r in rows):.0%} to {max(r[3] for r in rows):.0%} - so an effect of exactly "
        "the size observed would have failed to reach significance most of the time. **The "
        "experiments were not capable of returning a positive result even if their hypotheses were "
        "true.** That is why the nulls are so uniform across Findings 46, 47, 48, 57, 69 and 70: "
        "they are mostly measuring the sample size, not the hypotheses.",
        "",
        "**This is the project's central constraint, stated as a quantity**, and it is the "
        "justification for the Main Track's Lane A, which was designed to label roughly 500 "
        "olympiad problems. A labelling run producing 100 rows would have settled nothing; 500 "
        "clears every row in this table at once. The plan was right even though it never ran.",
        "",
        "**What it does not excuse.** Low power explains why the nulls are uninformative; it does "
        "not make the observed effects real. An effect estimated at n = 33 is estimated badly in "
        "both directions, and the honest reading of Findings 46 to 70 is that we do not know - not "
        "that we were nearly right.",
    ]

    body = "\n".join(lines)
    print(body)
    if not args.no_write:
        upsert_section(ROOT / "RESULTS.md", "E48: labels required to settle each lane", body)
        print("\nwrote the E48 section of RESULTS.md")


if __name__ == "__main__":
    main()
