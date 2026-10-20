"""Evaluate only E34a's pre-registered yield gate; fit no model and use no official labels."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


DEFAULT_LABELS = Path("runs/e34a-qwen-aime-surface/e34a-qwen-aime-labels")
SOURCES = ("aime-2024", "aime-2026")
CATEGORIES = ("latex_to_plaintext", "premise_reorder")


def evaluate(instances: pd.DataFrame, rows: pd.DataFrame) -> dict:
    clear = instances[instances["is_eligible"].astype(bool) & instances["is_clear_cut"].astype(bool)].copy()
    counts = {source: int((clear["dataset_id"] == source).sum()) for source in SOURCES}
    by_source_class = {
        source: {
            "robust": int(((clear["dataset_id"] == source) & clear["is_robust"].astype(bool)).sum()),
            "spurious": int(((clear["dataset_id"] == source) & ~clear["is_robust"].astype(bool)).sum()),
        }
        for source in SOURCES
    }
    eligible_keys = set(clear["problem_key"])
    sampled = rows[rows["problem_key"].isin(eligible_keys)]
    category_rows = {
        category: int((sampled["perturbation_key"] == category).sum()) for category in CATEGORIES
    }
    robust = int(clear["is_robust"].astype(bool).sum())
    spurious = int(len(clear) - robust)
    checks = {
        "aime_2024_clear_cut_at_least_8": counts["aime-2024"] >= 8,
        "aime_2026_clear_cut_at_least_5": counts["aime-2026"] >= 5,
        "combined_robust_at_least_4": robust >= 4,
        "combined_spurious_at_least_4": spurious >= 4,
        "both_classes_in_each_source": all(
            by_source_class[source]["robust"] >= 1 and by_source_class[source]["spurious"] >= 1
            for source in SOURCES
        ),
        "each_category_has_at_least_4_rows": all(n >= 4 for n in category_rows.values()),
    }
    return {
        "clear_cut_per_source": counts,
        "class_per_source": by_source_class,
        "clear_cut_total": len(clear),
        "robust": robust,
        "spurious": spurious,
        "eligible_rows_per_category": category_rows,
        "checks": checks,
        "passed": all(checks.values()),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--labels", type=Path, default=DEFAULT_LABELS)
    args = parser.parse_args()
    result = evaluate(
        pd.read_parquet(args.labels / "instances.parquet"),
        pd.read_parquet(args.labels / "perturbation_rows.parquet"),
    )
    print(json.dumps(result, indent=2))
    raise SystemExit(0 if result["passed"] else 2)


if __name__ == "__main__":
    main()
