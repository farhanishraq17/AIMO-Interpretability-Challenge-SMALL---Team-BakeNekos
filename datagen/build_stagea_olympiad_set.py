"""Build the instance table for E21: do prompt hidden states carry olympiad robustness?

Every block we have has now been scored against the organizers' olympiad labels except the
one the submission actually leans on. The text block reaches AUC 0.617 with a CI spanning 0.5
(Finding 37), and training it on olympiad labels instead of MATH changes nothing (Finding 39).
The prompt-hidden-state block has never been tested out of population at all: E1b measured it
on Hendrycks MATH, and `train-main-v2` needs a forward pass we have not run.

This writes one parquet that `extract_features.py` can consume directly, holding two disjoint
groups of `Qwen/Qwen3.5-4B` rows:

* **train** -- our self-generated olympiad labels (`data/qwen35_labels/instances_train.parquet`),
  the answer-modulus labels from E15/E18/E20;
* **eval** -- the `Qwen/Qwen3.5-4B` rows of `train-main-v2` that carry an official label.

`dataset_id` records which group a row belongs to, so the training script can split on it
rather than re-deriving the membership. The two groups are verified disjoint by
`problem_key`: an overlap would leak the evaluation labels into the probe.

Qwen is the only model where both halves exist -- it is the only one with enough self-labels
to fit a probe (26) and a useful number of official rows (12).

Run:  uv run python ../datagen/build_stagea_olympiad_set.py
"""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path

import pandas as pd
from huggingface_hub import hf_hub_download

ROOT = Path(__file__).resolve().parents[1]
SELF_LABELS = ROOT / "data" / "qwen35_labels" / "instances_train.parquet"
OUT = ROOT / "data" / "stagea_olympiad" / "instances.parquet"

DATASET = "aimo-interp/train-main-v2"
REVISION = "23791d9b120132087877b0459f953d5959fc8455"
PARQUET = "data/train-00000-of-00001.parquet"
MODEL_ID = "Qwen/Qwen3.5-4B"


def problem_key(text: str) -> str:
    """Identical to `ingest_public.problem_key`, so keys are comparable across tables."""
    normalized = " ".join((text or "").split())
    return hashlib.sha1(normalized.encode("utf-8")).hexdigest()[:16]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--revision", default=REVISION)
    parser.add_argument("--out", type=Path, default=OUT)
    args = parser.parse_args()

    if not SELF_LABELS.is_file():
        raise SystemExit(f"{SELF_LABELS} is missing; run datagen/perturb/label.py first")
    train = pd.read_parquet(SELF_LABELS)

    official = pd.read_parquet(
        hf_hub_download(DATASET, PARQUET, repo_type="dataset", revision=args.revision)
    )
    official = official[(official["model_id"] == MODEL_ID) & official["is_robust"].notna()].copy()
    official["problem_key"] = official["problem"].map(problem_key)
    # The same statement can appear under several reasoning efforts; one row per problem.
    official = official.drop_duplicates(subset="problem_key")

    rows = pd.DataFrame(
        [
            {
                "problem_key": row.problem_key,
                "model_id": MODEL_ID,
                "problem": row.problem,
                "is_robust": bool(row.is_robust),
                "dataset_id": "qwen35-self-labels",
            }
            for row in train.itertuples(index=False)
        ]
        + [
            {
                "problem_key": row.problem_key,
                "model_id": MODEL_ID,
                "problem": row.problem,
                "is_robust": bool(row.is_robust),
                "dataset_id": "train-main-v2",
            }
            for row in official.itertuples(index=False)
        ]
    )

    groups = {name: set(part["problem_key"]) for name, part in rows.groupby("dataset_id")}
    overlap = groups["qwen35-self-labels"] & groups["train-main-v2"]
    if overlap:
        raise SystemExit(
            f"{len(overlap)} problems appear in both groups; the evaluation would leak: {sorted(overlap)[:5]}"
        )
    if rows["problem_key"].duplicated().any():
        raise SystemExit("duplicate problem_key within a group; one vector file per key is assumed")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    rows.to_parquet(args.out, index=False)

    for name, part in rows.groupby("dataset_id"):
        print(f"  {name:20s} n={len(part):3d}  robust={int(part['is_robust'].sum()):3d}"
              f" ({part['is_robust'].mean():.3f})")
    print(f"  {'total':20s} n={len(rows):3d}  disjoint by problem_key: yes")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
