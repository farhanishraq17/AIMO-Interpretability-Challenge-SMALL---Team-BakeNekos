"""Check for negative accuracy decay, which the `decay == 0` label rule mishandles."""

from __future__ import annotations

from pathlib import Path

import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "small_track"


def main() -> None:
    rows = pq.read_table(DATA / "perturbation_rows.parquet").to_pylist()
    instances = pq.read_table(DATA / "instances.parquet").to_pylist()

    neg_rows = [r for r in rows if r["absolute_accuracy_decay"] < 0]
    neg_inst = [i for i in instances if i["max_absolute_accuracy_decay"] < 0]

    print(f"perturbation rows with negative decay: {len(neg_rows)} / {len(rows)}")
    print(f"instances with negative max decay:     {len(neg_inst)} / {len(instances)}")

    for row in neg_rows[:10]:
        print(
            f"\n  model={row['model_id']}"
            f"\n  decay={row['absolute_accuracy_decay']}"
            f"  base={row['base_accuracy']} permuted={row['permuted_accuracy']}"
            f"\n  label={row['is_robust']} source={row['label_source']}"
            f"  perturbations={row['perturbation_types']}"
            f"\n  repo={row['source_repo']}"
        )

    published_neg = [r for r in neg_rows if r["label_source"] == "published"]
    print(f"\nnegative-decay rows carrying a PUBLISHED label: {len(published_neg)}")
    for row in published_neg:
        print(
            f"  published is_robust={row['is_robust']} for decay={row['absolute_accuracy_decay']}"
        )


if __name__ == "__main__":
    main()
