"""Recover the organizers' robust/spurious labelling rule from the labelled datasets.

The unlabelled public tables (challenge-sample-full, augmented-sample-math-full-filtered)
carry accuracy-decay columns but no `model_is_robust`. To reuse them as training data we
must derive labels the same way the organizers did, so this script fits the rule on the
tables where both the decay columns and the label are present.
"""

from __future__ import annotations

import pyarrow.parquet as pq
from huggingface_hub import hf_hub_download

LABELLED = [
    "aimo-interp/augmented-sample-math-agg",
    "aimo-interp/augmented-sample-math-agg-filtered",
    "aimo-interp/augmented-sample-math",
    "aimo-interp/val-sample",
]

CANDIDATES = {
    "absolute_accuracy_decay == 0": lambda r: r["absolute_accuracy_decay"] == 0,
    "absolute_accuracy_decay <= 0.1": lambda r: r["absolute_accuracy_decay"] <= 0.1,
    "absolute_accuracy_decay < 0.5": lambda r: r["absolute_accuracy_decay"] < 0.5,
    "relative_accuracy_decay == 0": lambda r: r["relative_accuracy_decay"] == 0,
    "relative_accuracy_decay < 0.5": lambda r: r["relative_accuracy_decay"] < 0.5,
    "n_detrimental_permutations == 0": lambda r: r["n_detrimental_permutations"] == 0,
}


def rows_for(repo: str) -> list[dict]:
    info_files = ["data/validation-00000-of-00001.parquet"]
    out: list[dict] = []
    for name in info_files:
        try:
            path = hf_hub_download(repo_id=repo, repo_type="dataset", filename=name)
        except Exception as exc:  # noqa: BLE001
            print(f"  cannot read {repo}/{name}: {type(exc).__name__}")
            continue
        out.extend(pq.read_table(path).to_pylist())
    return out


def main() -> None:
    pooled: list[dict] = []
    for repo in LABELLED:
        rows = rows_for(repo)
        label_col = next(
            (c for c in ("model_is_robust", "is_robust") if rows and c in rows[0]), None
        )
        if not label_col:
            print(f"{repo}: no label column, skipped")
            continue
        print(f"{repo}: {len(rows)} labelled rows")
        for row in rows:
            row["_label"] = bool(row[label_col])
        pooled.extend(rows)

    print(f"\npooled labelled rows: {len(pooled)}")
    truthy = sum(1 for r in pooled if r["_label"])
    print(f"robust={truthy} spurious={len(pooled) - truthy}\n")

    print("candidate rule agreement with the published label:")
    for name, rule in CANDIDATES.items():
        try:
            agree = sum(1 for r in pooled if rule(r) == r["_label"])
        except (KeyError, TypeError):
            print(f"  {name:38s} n/a (missing column)")
            continue
        print(f"  {name:38s} {agree}/{len(pooled)} = {agree / len(pooled):.4f}")

    print("\nlabel vs absolute_accuracy_decay cross-tab:")
    buckets: dict[tuple[str, bool], int] = {}
    for row in pooled:
        decay = row["absolute_accuracy_decay"]
        if decay == 0:
            key = "0"
        elif decay <= 0.1:
            key = "(0, 0.1]"
        elif decay <= 0.3:
            key = "(0.1, 0.3]"
        elif decay <= 0.5:
            key = "(0.3, 0.5]"
        else:
            key = "> 0.5"
        buckets[(key, row["_label"])] = buckets.get((key, row["_label"]), 0) + 1
    for bucket in ["0", "(0, 0.1]", "(0.1, 0.3]", "(0.3, 0.5]", "> 0.5"]:
        robust = buckets.get((bucket, True), 0)
        spurious = buckets.get((bucket, False), 0)
        print(f"  decay {bucket:11s} robust={robust:4d} spurious={spurious:4d}")


if __name__ == "__main__":
    main()
