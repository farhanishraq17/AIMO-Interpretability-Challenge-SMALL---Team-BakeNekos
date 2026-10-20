"""E46: characterise the direction reversal against `max_drop` itself, not the binarised label.

Every test of the reversal so far (Findings 58, 62, 69) scored familiarity against `is_robust`,
which throws away two things. The three-valued rule leaves **11 rows unlabelled** in the
0.11-0.21 band, and binarising the rest discards how far each row sits from the cut. Both are
recoverable: `train-main-v2` ships `max_drop` for every row, labelled or not.

(In the event only **3** of those 11 band rows are small-track; the rest are gpt-oss-120b's, for
which no familiarity features exist. The power gain comes from the continuous target, not from the
recovered rows - see Finding 70.)

Scoring against the continuous quantity is a sharper instrument on the same data. It asks whether
familiarity tracks *how much* a model's accuracy falls under perturbation, rather than whether it
crosses a threshold, and it uses the band rows that the label rule discards - which is exactly
where a weak monotone effect would show up first if one exists.

Three questions:

1. **Overall.** Spearman between familiarity and `max_drop` over the small-track rows. A positive
   correlation is the memorisation reading - more familiar, larger drop, more spurious - and is the
   direction the official rows have shown throughout.
2. **Is it one model's?** Finding 50 showed the leaderboard *gain* was DeepSeek-specific. Per-model
   correlations say whether the official-row *reversal* is similarly concentrated or uniform.
3. **Is it length?** Problem length drives every text statistic here (Finding 22), so the same
   correlation is recomputed within length deciles.

gpt-oss-120b rows are excluded: 37 of the 82 rows are its, and we have no familiarity features for
a model we were never able to run (LF-3).

Writes the `E46: familiarity against continuous max_drop` section of RESULTS.md.

Run:  python train/eval_maxdrop_continuous.py
"""

from __future__ import annotations

import argparse
import glob
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "train"))

from results_io import upsert_section  # noqa: E402

RNG = np.random.default_rng(20261001)
FEATURE_DIR_TO_MODEL = {
    "familiarity-qwen35": "Qwen/Qwen3.5-4B",
    "familiarity-skywork": "Skywork/Skywork-OR1-Math-7B",
    "familiarity-olmo-think": "allenai/Olmo-3-7B-Think",
    "familiarity-deepseek": "deepseek-ai/DeepSeek-R1-0528-Qwen3-8B",
}


def features() -> pd.DataFrame:
    frames = []
    for path in sorted(glob.glob(str(ROOT / "runs/e23-familiarity-*/features/familiarity-*/shard_*/records.jsonl"))):
        name = Path(path).parts[-3]
        rows = [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip()]
        frame = pd.DataFrame(rows)[["problem_key", "mean_nll", "n_tokens"]].copy()
        # Familiarity is the negated NLL: higher means more familiar, so the sign of every
        # correlation below reads directly rather than needing to be flipped.
        frame["familiarity"] = -frame.mean_nll
        frame["model_id"] = FEATURE_DIR_TO_MODEL[name]
        frames.append(frame)
    return pd.concat(frames, ignore_index=True)


def spearman_ci(x: np.ndarray, y: np.ndarray, draws: int = 4000) -> tuple[float, float, float]:
    point = float(pd.Series(x).corr(pd.Series(y), method="spearman"))
    boot = []
    for _ in range(draws):
        idx = RNG.integers(0, len(x), len(x))
        if len(set(x[idx])) > 2 and len(set(y[idx])) > 2:
            boot.append(pd.Series(x[idx]).corr(pd.Series(y[idx]), method="spearman"))
    return point, float(np.nanpercentile(boot, 2.5)), float(np.nanpercentile(boot, 97.5))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--no-write", action="store_true")
    args = parser.parse_args()

    rows = pd.read_parquet(ROOT / "data/train_main/rows.parquet")
    small = rows[rows.model_id.isin(FEATURE_DIR_TO_MODEL.values())].copy()
    joined = small.merge(features(), on=["problem_key", "model_id"], how="inner").dropna(subset=["max_drop"])

    lines = [
        f"{len(joined)} small-track rows carry both `max_drop` and a familiarity score, of which "
        f"**{int(joined.is_labelled.sum())} are labelled** - the other "
        f"{len(joined) - int(joined.is_labelled.sum())} sit in the 0.11-0.21 band the three-valued "
        "rule discards and are usable here. Spearman throughout; a **positive** correlation means "
        "more familiar goes with a larger accuracy drop, i.e. the memorisation reading, which is "
        "the direction the official rows have shown throughout (Findings 58, 62, 69). "
        "gpt-oss-120b's 37 rows are excluded: no familiarity features exist for a model the fleet "
        "could never run (LF-3).",
        "",
        "| set | rows | Spearman(familiarity, max_drop) | 95% CI |",
        "| --- | --- | --- | --- |",
    ]

    for name, subset in (("all small-track rows", joined),
                         ("labelled rows only", joined[joined.is_labelled]),
                         ("band rows only (unlabelled)", joined[~joined.is_labelled])):
        if len(subset) >= 6:
            point, lo, hi = spearman_ci(subset.familiarity.to_numpy(), subset.max_drop.to_numpy())
            lines.append(f"| {name} | {len(subset)} | {point:+.3f} | [{lo:+.3f}, {hi:+.3f}] |")
        else:
            lines.append(f"| {name} | {len(subset)} | — | too few rows |")

    lines += ["", "### Per model", "",
              "| model | rows | Spearman | 95% CI |", "| --- | --- | --- | --- |"]
    per_model = {}
    for model, group in joined.groupby("model_id"):
        if len(group) >= 6:
            point, lo, hi = spearman_ci(group.familiarity.to_numpy(), group.max_drop.to_numpy())
            per_model[model] = point
            lines.append(f"| {model.split('/')[-1]} | {len(group)} | {point:+.3f} | [{lo:+.3f}, {hi:+.3f}] |")
        else:
            lines.append(f"| {model.split('/')[-1]} | {len(group)} | — | too few rows |")

    decile = pd.qcut(joined.n_tokens, 6, labels=False, duplicates="drop")
    within = [float(g.familiarity.corr(g.max_drop, method="spearman"))
              for _, g in joined.groupby(decile) if len(g) >= 5]
    overall, lo, hi = spearman_ci(joined.familiarity.to_numpy(), joined.max_drop.to_numpy())

    lines += [
        "",
        f"Within length sextiles, where problem length cannot drive it, the mean correlation is "
        f"**{np.nanmean(within):+.3f}** against {overall:+.3f} overall - so "
        f"{'length explains most of it' if abs(np.nanmean(within)) < abs(overall) / 2 else 'the effect is not a length artifact'}.",
        "",
        f"**Reading.** The overall correlation is {overall:+.3f}, CI [{lo:+.3f}, {hi:+.3f}]. "
        + ("The interval excludes zero, so the reversal on official rows is a real monotone effect "
           "rather than an artifact of where the label thresholds fall."
           if lo > 0 or hi < 0 else
           "The interval contains zero, so even with the band rows recovered and the continuous "
           "target in play, the reversal is not established at this sample size. That is the "
           "sharpest instrument these rows support, and it is still not sharp enough.")
        + (f" Per model the correlations run {min(per_model.values()):+.3f} to "
           f"{max(per_model.values()):+.3f}, so the effect is "
           f"{'concentrated in one model' if max(per_model.values()) - min(per_model.values()) > 0.6 else 'broadly consistent in sign across models'}."
           if per_model else ""),
    ]

    body = "\n".join(lines)
    print(body)
    if not args.no_write:
        upsert_section(ROOT / "RESULTS.md", "E46: familiarity against continuous max_drop", body)
        print("\nwrote the E46 section of RESULTS.md")


if __name__ == "__main__":
    main()
