"""Compare base_accuracy between organizer-published rows and derived rows.

The task is only defined for problems the model already answers correctly, and the
proposal says non-clear-cut cases are dropped from evaluation. If the published rows
cluster at high base accuracy while the unlabelled table spans low values, we must
apply the same eligibility filter before training on derived rows.
"""

from __future__ import annotations

from collections import Counter
from pathlib import Path

import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "small_track"


def bucket(value: float | None) -> str:
    if value is None:
        return "none"
    if value >= 1.0:
        return "1.0"
    if value >= 0.9:
        return "[0.9, 1.0)"
    if value >= 0.7:
        return "[0.7, 0.9)"
    if value >= 0.5:
        return "[0.5, 0.7)"
    return "< 0.5"


ORDER = ["1.0", "[0.9, 1.0)", "[0.7, 0.9)", "[0.5, 0.7)", "< 0.5", "none"]


def main() -> None:
    rows = pq.read_table(DATA / "perturbation_rows.parquet").to_pylist()

    for source in ("published", "derived"):
        subset = [r for r in rows if r["label_source"] == source]
        counts = Counter(bucket(r["base_accuracy"]) for r in subset)
        print(f"{source} rows: {len(subset)}")
        for name in ORDER:
            if counts.get(name):
                print(f"    base_accuracy {name:12s} {counts[name]:4d}")
        print()

    eligible = [r for r in rows if (r["base_accuracy"] or 0) >= 1.0]
    print(f"rows with base_accuracy == 1.0: {len(eligible)} / {len(rows)}")

    by_repo = Counter(r["source_repo"].split("/")[-1] for r in rows if (r["base_accuracy"] or 0) < 1.0)
    print("\nrows below base_accuracy 1.0, by source table:")
    for repo, count in by_repo.most_common():
        print(f"    {count:4d}  {repo}")


if __name__ == "__main__":
    main()
