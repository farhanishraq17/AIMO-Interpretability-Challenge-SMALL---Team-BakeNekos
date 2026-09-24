"""Independent E32 confirmation of E31's selected signal on DeepSeek MATH.

E31 selected ``cf_cosine_auc`` after its five-way official-data screen.  E32 changes
nothing: lower rewrite sensitivity predicts robust.  The confirmation gate, fixed before
downloading E32, is AUC >= .65, ordinary bootstrap CI lower bound > .50, one-sided
permutation p <= .05, within-length-quintile AUC >= .60, and the deployable half-batch
rank policy balanced accuracy >= .60.
"""

from __future__ import annotations

import argparse
import glob
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import rankdata

ROOT = Path(__file__).resolve().parents[1]
PATTERN = "runs/e32-counterfactual-deepseek-math/features/e32-counterfactual-deepseek-math/shard_*/records.jsonl"


def binary_auc(labels, scores) -> float:
    labels = np.asarray(labels, dtype=bool)
    scores = np.asarray(scores, dtype=float)
    n_pos = int(labels.sum())
    n_neg = int((~labels).sum())
    ranks = rankdata(scores, method="average")
    return float((ranks[labels].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))


def within_length_auc(labels, scores, length, n_bins: int = 5) -> float:
    labels = np.asarray(labels, dtype=bool)
    scores = np.asarray(scores, dtype=float)
    length = np.asarray(length, dtype=float)
    bins = np.minimum((rankdata(length, method="ordinal") - 1) * n_bins // len(length), n_bins - 1)
    wins = pairs = 0.0
    for bin_index in range(n_bins):
        positive = scores[(bins == bin_index) & labels]
        negative = scores[(bins == bin_index) & ~labels]
        wins += (positive[:, None] > negative[None, :]).sum()
        wins += 0.5 * (positive[:, None] == negative[None, :]).sum()
        pairs += len(positive) * len(negative)
    return float(wins / pairs)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--write-results", action="store_true")
    args = parser.parse_args()
    paths = sorted(glob.glob(str(ROOT / PATTERN)))
    if not paths:
        raise SystemExit("no E32 records; download the Kaggle output first")
    rows = [
        json.loads(line)
        for path in paths
        for line in Path(path).read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    frame = pd.DataFrame(rows).sort_values("problem_key").reset_index(drop=True)
    if len(frame) != 142 or frame.problem_key.duplicated().any():
        raise SystemExit(f"expected 142 unique MATH rows, found {len(frame)}")
    if frame.truncated.any() or frame.cf_cosine_auc.isna().any():
        raise SystemExit("E32 output contains truncation or missing selected signals")

    labels = frame.is_robust.to_numpy(bool)
    scores = -frame.cf_cosine_auc.to_numpy(float)
    length = frame.n_prompt_tokens.to_numpy(int)
    rng = np.random.default_rng(20260924)
    auc = binary_auc(labels, scores)
    boot = []
    for _ in range(10000):
        indices = rng.integers(0, len(frame), len(frame))
        if labels[indices].any() and not labels[indices].all():
            boot.append(binary_auc(labels[indices], scores[indices]))
    low, high = np.quantile(boot, [0.025, 0.975])
    null = np.asarray([binary_auc(rng.permutation(labels), scores) for _ in range(10000)])
    pvalue = float((1 + np.sum(null >= auc)) / (len(null) + 1))
    within = within_length_auc(labels, scores, length)
    rho = float(pd.Series(scores).corr(pd.Series(length), method="spearman"))
    order = np.argsort(scores, kind="stable")
    predictions = np.zeros(len(frame), dtype=bool)
    predictions[order[len(frame) // 2 :]] = True
    split_ba = float(
        0.5 * (predictions[labels].mean() + (~predictions[~labels]).mean())
    )
    passed = bool(
        auc >= 0.65 and low > 0.50 and pvalue <= 0.05 and within >= 0.60 and split_ba >= 0.60
    )
    print(
        f"E32 n={len(frame)} robust={labels.sum()} AUC={auc:.3f} CI=[{low:.3f},{high:.3f}] "
        f"within-length={within:.3f} split-BA={split_ba:.3f} p={pvalue:.4f} "
        f"rho_len={rho:.3f} GATE={'PASS' if passed else 'FAIL'}"
    )

    if args.write_results:
        path = ROOT / "RESULTS.md"
        text = path.read_text(encoding="utf-8").rstrip()
        marker = "## E32 — independent counterfactual-sensitivity confirmation"
        if marker in text:
            text = text.split(marker, 1)[0].rstrip()
        section = f"""{marker}

E31 selected all-layer mean cosine distance only after its five-way official-data screen. E32 fixes that signal and its direction (lower rewrite sensitivity predicts robust) and tests it once on the untouched 142 DeepSeek Hendrycks MATH rows.

| n | Robust | AUC | bootstrap 95% CI | within-length AUC | split-half BA | one-sided p | length rho | Gate |
|---:|---:|---:|---:|---:|---:|---:|---:|---|
| {len(frame)} | {int(labels.sum())} | {auc:.3f} | [{low:.3f}, {high:.3f}] | {within:.3f} | {split_ba:.3f} | {pvalue:.4f} | {rho:.3f} | {'pass' if passed else 'fail'} |

**Decision: {'the signal independently confirms; proceed to a DeepSeek-only packaged GPU check before any submission.' if passed else 'the signal does not independently confirm; close the counterfactual-sensitivity lane without a submission.'}**
"""
        path.write_text(text + "\n\n" + section, encoding="utf-8")


if __name__ == "__main__":
    main()
