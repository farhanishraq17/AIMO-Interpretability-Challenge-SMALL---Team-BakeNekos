"""E51: which perturbation families make a label spurious, on MATH and on olympiad problems?

Every model this project trained on DeepSeek's Hendrycks MATH labels failed to transfer to the
olympiad rows (Findings 32, 39, 40), and the familiarity direction flips between the two
(PLAN section 12.3). Nobody had asked what the two label sources actually measure. A label is
spurious when *any* perturbation family costs at least 25 points (the `train-main-v2` card), so
the question is which family does that, population by population.

Two tables answer it with no GPU:

1. `aimo-interp-challenge-sample-full`, the organizers' only public table with per-family
   results on olympiad problems (10 AIMO-2025 reference problems, 7 configurations, and the
   variant texts themselves). `train-main-v2` gives only `max_drop`, not the family behind it.
2. The DeepSeek rows of our consolidated set (`data/small_track/perturbation_rows.parquet`),
   split into the MATH tables and that olympiad sample.

Rows are restricted to `base_accuracy >= 1.0`, the eligibility rule used throughout (CLAUDE.md),
so a drop means the perturbation broke a problem the model had solved.

A third, exploratory table asks whether solving reliably goes with being robust: E24's base
samples (k = 4) against the official labels of the same pairs. Those 34 rows have been read many
times (Finding 74 closed them as a holdout), so it is a description, not a test.

Writes the `E51: what the labels measure` section of RESULTS.md.

Run:  python train/eval_label_mechanics.py
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from huggingface_hub import hf_hub_download
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "train"))

from results_io import upsert_section  # noqa: E402

SAMPLE = ("aimo-interp/aimo-interp-challenge-sample-full", "data/validation-00000-of-00001.parquet",
          "c0ffb7e678294cc8819d50d979ba28ba49bd02d0")
ROWS = ROOT / "data" / "small_track" / "perturbation_rows.parquet"
OFFICIAL = ROOT / "data" / "official_check" / "official_labels.parquet"
E24_BASE = "runs/e24b-*/e24b-*/base/*/records.jsonl"
SPURIOUS = 0.25  # `train-main-v2`: is_robust = false iff max_drop >= 0.25


def family_table(frame: pd.DataFrame, family: str) -> str:
    stats = (frame.groupby(family)["drop"]
             .agg(rows="size", mean_drop="mean", share=lambda d: (d >= SPURIOUS).mean())
             .sort_values("rows", ascending=False))
    lines = ["| family | rows | mean drop | share with drop >= 0.25 |", "| --- | --- | --- | --- |"]
    lines += [f"| {f} | {int(r.rows)} | {r.mean_drop:.2f} | **{r.share:.2f}** |" for f, r in stats.iterrows()]
    return "\n".join(lines)


def sample_section() -> list[str]:
    repo, filename, revision = SAMPLE
    frame = pd.read_parquet(hf_hub_download(repo, filename, repo_type="dataset", revision=revision))
    frame = frame[frame.base_accuracy >= 1.0].assign(drop=lambda f: f.base_accuracy - f.permuted_accuracy)
    worst = frame.loc[frame.groupby(["model_id", "problem_id"])["drop"].idxmax()]
    setters = worst[worst["drop"] >= SPURIOUS].permutation_type.value_counts()
    return [
        f"**Organizers' olympiad sample**, all {frame.model_id.nunique()} configurations, "
        f"{frame.problem_id.nunique()} problems, eligible rows only:",
        "",
        family_table(frame, "permutation_type"),
        "",
        f"The family that sets `max_drop` on the {int(setters.sum())} spurious (configuration, problem) "
        "pairs: " + ", ".join(f"{f} {n}" for f, n in setters.items()) + ".",
    ]


def deepseek_section() -> list[str]:
    rows = pd.read_parquet(ROWS)
    rows = rows[(rows.base_accuracy >= 1.0) & ~rows.is_perturbation_aggregate.astype(bool)
                & rows.model_id.str.contains("DeepSeek|qwen3-8b")]
    rows = rows.assign(drop=rows.base_accuracy - rows.permuted_accuracy,
                       family=rows.perturbation_types.map(lambda t: ", ".join(t)))
    out = []
    for name, keep in (("Hendrycks MATH tables (what every MATH-trained model learned)", "augmented"),
                       ("olympiad sample", "challenge")):
        part = rows[rows.source_repo.str.contains(keep)]
        out += ["", f"**DeepSeek, {name}**, {len(part)} eligible rows:", "", family_table(part, "family")]
    return out


def bootstrap_auc(y: np.ndarray, score: np.ndarray, n_boot: int = 5000, seed: int = 0) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    aucs = [roc_auc_score(y[i], score[i]) for i in rng.integers(0, len(y), (n_boot, len(y))) if 0 < y[i].sum() < len(i)]
    return float(np.percentile(aucs, 2.5)), float(np.percentile(aucs, 97.5))


def competence_section() -> list[str]:
    records = []
    for path in sorted(ROOT.glob(E24_BASE)):
        for line in path.read_text(encoding="utf-8").splitlines():
            r = json.loads(line)
            if r["kind"] == "original":
                records.append({"model_id": r["model_id"], "problem_key": r["pool_id"].split(":")[1],
                                "solved": r["n_correct"], "tokens": float(np.mean(r["completion_tokens"]))})
    assert records, f"no E24 base records under {E24_BASE}; download the e24b runs first"
    pairs = pd.read_parquet(OFFICIAL).merge(pd.DataFrame(records), on=["model_id", "problem_key"])
    y = pairs.is_robust.astype(int).to_numpy()
    lines = [
        "",
        f"**Exploratory: E24's own base samples (k = 4, temperature 0.6, 24k cap) against the official "
        f"labels of the same {len(pairs)} pairs** ({int(y.sum())} robust). AUC oriented so that above 0.5 "
        "means the stated direction predicts robust:",
        "",
        "| signal | direction | AUC | 95% bootstrap CI |",
        "| --- | --- | --- | --- |",
    ]
    for column, sign, direction in (("solved", 1, "more of 4 samples correct"),
                                    ("tokens", -1, "shorter reasoning traces")):
        score = sign * pairs[column].to_numpy()
        low, high = bootstrap_auc(y, score)
        lines.append(f"| {column} | {direction} | {roc_auc_score(y, score):.3f} | [{low:.3f}, {high:.3f}] |")
    return lines


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--no-write", action="store_true", help="print only; do not touch RESULTS.md")
    args = parser.parse_args()

    body = "\n".join(sample_section() + deepseek_section() + competence_section())
    print(body)
    if not args.no_write:
        upsert_section(ROOT / "RESULTS.md", "E51: what the labels measure", body)
        print("\nwrote the E51 section of RESULTS.md")


if __name__ == "__main__":
    main()
