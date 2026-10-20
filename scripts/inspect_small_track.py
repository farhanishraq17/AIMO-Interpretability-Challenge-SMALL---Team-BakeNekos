"""Summarize the consolidated small-track training set."""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "small_track"


def main() -> None:
    instances = pq.read_table(DATA / "instances.parquet").to_pylist()
    rows = pq.read_table(DATA / "perturbation_rows.parquet").to_pylist()
    summary = json.loads((DATA / "summary.json").read_text(encoding="utf-8"))

    print("folds (grouped by problem):")
    for fold, stats in summary["fold_stats"].items():
        print(f"  fold {fold}: instances={stats['instances']:3d} robust={stats['robust']:3d}")

    deepseek = [i for i in instances if i["model_id"].startswith("deepseek")]
    print(f"\nDeepSeek-8B instances: {len(deepseek)}")
    print(f"  difficulty mix: {dict(Counter(i['dataset_id'] for i in deepseek))}")

    print("\ndecay distribution of instance labels (all models):")
    buckets = Counter()
    for inst in instances:
        decay = inst["max_absolute_accuracy_decay"]
        if decay == 0:
            buckets["0 (robust)"] += 1
        elif decay <= 0.1:
            buckets["(0, 0.1] borderline"] += 1
        elif decay <= 0.3:
            buckets["(0.1, 0.3] borderline"] += 1
        elif decay <= 0.5:
            buckets["(0.3, 0.5]"] += 1
        else:
            buckets["> 0.5"] += 1
    for name in [
        "0 (robust)",
        "(0, 0.1] borderline",
        "(0.1, 0.3] borderline",
        "(0.3, 0.5]",
        "> 0.5",
    ]:
        print(f"  {name:24s} {buckets.get(name, 0):4d}")

    borderline = sum(
        1 for i in instances if 0 < i["max_absolute_accuracy_decay"] <= 0.3
    )
    print(
        f"\nborderline instances (0 < decay <= 0.3): {borderline}"
        f"  <- organizers exclude non-clear-cut cases from evaluation"
    )

    print("\nperturbation-row label provenance:")
    print(f"  {dict(Counter(r['label_source'] for r in rows))}")


if __name__ == "__main__":
    main()
