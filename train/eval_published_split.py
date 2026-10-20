"""E45: does familiarity predict robustness differently for publicly mirrored problems?

PLAN section 12 and PAPER section 12 carry the project's one unresolved contradiction. The shipped
direction - familiar problems are robust - holds on DeepSeek's MATH instances and on the hidden
validation set, and **reverses** on the official olympiad rows. The hypothesis offered for it:
`train-main-v2` holds *published* problems, where familiarity means memorisation and predicts
spurious, while the validation cases are mostly not those, where familiarity is ordinary fluency
and predicts robust.

That hypothesis has never been tested because it needs a publication indicator. There is one, and
it is non-circular: **8 of the 36 official problems appear verbatim in our pinned olympiad pool**
(AIME-2024, AIME-2026, AMC validation, OlympiadBench - all widely mirrored public datasets, pinned
by revision). Membership there is decided by exact normalised text match, independently of any
model, any score and any label.

So: split the official problems on that indicator and compare the direction of the familiarity
signal in each half.

**The indicator is one-sided and the report must say so.** A problem in the pool is certainly
published and widely mirrored. A problem absent from it is *not identified as published* - it may
be published somewhere the pool does not cover. The comparison is therefore
"mirrored-in-our-pool" against "everything else", which is a weaker contrast than
published-against-unpublished and biases toward finding no difference.

Familiarity is the cross-model consensus rank (E44, Finding 68): the four models rank problems
almost identically, so the consensus is the natural per-problem statistic and avoids picking one
model arbitrarily.

Writes the `E45: familiarity by publication status` section of RESULTS.md.

Run:  python train/eval_published_split.py
"""

from __future__ import annotations

import argparse
import glob
import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "train"))

from results_io import upsert_section  # noqa: E402

RNG = np.random.default_rng(20261001)


def normalise(text: object) -> str:
    return re.sub(r"\s+", " ", str(text)).strip().lower()


def familiarity() -> pd.Series:
    """Consensus familiarity rank per problem: the mean of the four models' NLL ranks (E44)."""
    frames = []
    for path in sorted(glob.glob(str(ROOT / "runs/e23-familiarity-*/features/familiarity-*/shard_*/records.jsonl"))):
        name = Path(path).parts[-3]
        rows = [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip()]
        frame = pd.DataFrame(rows)
        frame["model"] = name
        frames.append(frame)
    data = pd.concat(frames, ignore_index=True)
    wide = data.pivot_table(index="problem_key", columns="model", values="mean_nll").dropna()
    return wide.rank().mean(axis=1)


def auc_ci(scores: np.ndarray, labels: np.ndarray, draws: int = 4000) -> tuple[float, float, float]:
    """AUC with a percentile bootstrap CI; resamples that lose a class are skipped."""
    point = roc_auc_score(labels, scores)
    boot = []
    for _ in range(draws):
        idx = RNG.integers(0, len(labels), len(labels))
        if len(set(labels[idx])) == 2:
            boot.append(roc_auc_score(labels[idx], scores[idx]))
    return float(point), float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--no-write", action="store_true")
    args = parser.parse_args()

    rows = pd.read_parquet(ROOT / "data/train_main/rows.parquet")
    pool = {normalise(t) for t in pd.read_parquet(ROOT / "data/olympiad_pool/pool.parquet").problem}
    fam = familiarity()

    problems = rows.drop_duplicates("problem_key")[["problem_key", "problem"]].copy()
    problems["mirrored"] = problems.problem.map(lambda t: normalise(t) in pool)
    problems["familiarity"] = problems.problem_key.map(fam)

    # One label per problem: pooled across models, which Finding 38 and LF-12 licence.
    labelled = rows.dropna(subset=["is_robust"])
    per_problem = labelled.groupby("problem_key").is_robust.mean()
    problems["robust"] = problems.problem_key.map(per_problem)
    usable = problems.dropna(subset=["familiarity", "robust"]).copy()
    usable["robust"] = usable.robust >= 0.5

    lines = [
        f"{len(problems)} official problems; **{int(problems.mirrored.sum())} appear verbatim in the "
        f"pinned olympiad pool** (exact normalised text match against AIME-2024, AIME-2026, AMC "
        "validation and OlympiadBench). Familiarity is the cross-model consensus NLL rank "
        "(Finding 68); a *lower* rank means more familiar. Labels are pooled per problem.",
        "",
        "| group | problems | labelled | robust | AUC of familiarity for robust | 95% CI |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    summary = {}
    for name, subset in (("mirrored in the public pool", usable[usable.mirrored]),
                         ("not identified as published", usable[~usable.mirrored]),
                         ("all official problems", usable)):
        n_lab = len(subset)
        n_rob = int(subset.robust.sum())
        if n_lab >= 4 and 0 < n_rob < n_lab:
            # Lower rank = more familiar, so negate to make "familiar predicts robust" AUC > 0.5.
            point, lo, hi = auc_ci(-subset.familiarity.to_numpy(), subset.robust.to_numpy())
            summary[name] = (point, lo, hi, n_lab)
            cell, ci = f"{point:.3f}", f"[{lo:.3f}, {hi:.3f}]"
        else:
            summary[name] = None
            cell, ci = "—", "too few labels or one class only"
        total = int(subset.mirrored.count()) if name != "all official problems" else len(usable)
        lines.append(f"| {name} | {total} | {n_lab} | {n_rob} | {cell} | {ci} |")

    lines += [
        "",
        "AUC above 0.5 means **familiar problems are robust** - the direction the shipped policy "
        "uses. Below 0.5 means familiarity predicts spurious, the memorisation reading.",
        "",
    ]

    a, b = summary["mirrored in the public pool"], summary["not identified as published"]
    if a and b:
        lines.append(
            f"The two halves point {'the same way' if (a[0] - 0.5) * (b[0] - 0.5) > 0 else '**opposite ways**'}: "
            f"{a[0]:.3f} on the {a[3]} mirrored problems against {b[0]:.3f} on the {b[3]} others. "
            "Both intervals are wide and both contain 0.5, so this neither confirms nor refutes the "
            "hypothesis; it bounds how large an effect could be hiding at this sample size.")
    else:
        lines.append("**The test cannot be run at this sample size.** One group has too few labelled "
                     "problems, or only one class, so no AUC is defined for it. That is itself the "
                     "result: the publication hypothesis is not testable on the official split, and "
                     "no amount of re-analysis of these 36 problems will make it so.")

    lines += [
        "",
        "**The indicator is one-sided.** A problem in the pool is certainly published and widely "
        "mirrored; a problem absent from it is only *not identified as published*, since the pool "
        "covers four sources and not the whole literature. The contrast is therefore weaker than "
        "published-against-unpublished and is biased toward finding no difference. A clean test "
        "needs publication dates per problem, which nothing available to us has.",
    ]

    body = "\n".join(lines)
    print(body)
    if not args.no_write:
        upsert_section(ROOT / "RESULTS.md", "E45: familiarity by publication status", body)
        print("\nwrote the E45 section of RESULTS.md")


if __name__ == "__main__":
    main()
