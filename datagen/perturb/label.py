"""E7 labels: aggregate sampled accuracies into rows the training pipeline already reads.

Consumes the base and variants phases of `sample.py` and emits the same schema as
`ingest_public.py`, so self-generated olympiad labels merge with the public data without new
code paths:

    perturbation_rows.parquet  one row per (model, problem, perturbation category)
    instances.parquet          one row per (model, problem), via ingest_public.build_instances
    instances_train.parquet    eligible and clear-cut instances only
    summary.json               yield at each step, per category and tier

Label rule, identical to the public data (PROGRESS.md §5.2):

* `permuted_accuracy` pools every sample of every variant in a category.
* `absolute_accuracy_decay = base_accuracy - permuted_accuracy`; robust iff decay <= 0.
* Eligible iff base_accuracy >= --min-base-accuracy (1.0, the organizers' curation).
* Clear-cut iff decay <= 0 or decay > 0.3.

`label_source` is "self-generated" and `compliance` is carried through, so a training set
can always be rebuilt from official data plus `distinct` categories only.

Run:  python datagen/perturb/label.py --base-glob "runs/e7-*/e7/base/shard_*/records.jsonl" \\
          --variants-glob "runs/e7-*/e7/variants/shard_*/records.jsonl"
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

import pandas as pd

PERTURB = Path(__file__).resolve().parent
ROOT = PERTURB.parents[1]
for import_path in (PERTURB, PERTURB.parent):
    if str(import_path) not in sys.path:
        sys.path.insert(0, str(import_path))

from ingest_public import AMBIGUOUS_MAX_DECAY, build_instances, write_parquet  # noqa: E402
from sample import DEFAULT_POOL, DEFAULT_VARIANTS, load_records  # noqa: E402

DEFAULT_OUT = ROOT / "data" / "e7_labels"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--base-glob", required=True)
    parser.add_argument("--variants-glob", required=True)
    parser.add_argument("--pool", type=Path, default=DEFAULT_POOL)
    parser.add_argument("--variants", type=Path, default=DEFAULT_VARIANTS)
    parser.add_argument("--min-base-accuracy", type=float, default=1.0)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()

    base = load_records(args.base_glob)
    sampled = load_records(args.variants_glob)
    pool = pd.read_parquet(args.pool)
    # Pools built before the pool.py fix reuse 34 AMC ids for two different problems; a record
    # under such an id cannot say which problem it measured, so it is dropped (Finding 29).
    ambiguous = set(pool.loc[pool["pool_id"].duplicated(keep=False), "pool_id"])
    pool = pool[~pool["pool_id"].isin(ambiguous)].set_index("pool_id")
    variants = pd.read_parquet(args.variants).set_index("variant_id")
    model_ids = {r["model_id"] for r in [*base.values(), *sampled.values()]}
    if len(model_ids) != 1:
        raise SystemExit(f"records mix models: {sorted(model_ids)}")
    model_id = model_ids.pop()

    base_accuracy = {
        r["pool_id"]: (r["n_correct"] / r["n_samples"], r["n_samples"])
        for r in base.values() if r["n_samples"] and r["pool_id"] in pool.index
    }
    missing_variants = [vid for vid in sampled if vid not in variants.index]
    if missing_variants:
        raise SystemExit(f"{len(missing_variants)} sampled variants are not in {args.variants}; regenerate with the same seed")

    grouped: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for variant_id, record in sampled.items():
        grouped[(record["pool_id"], record["category"])].append(record)

    rows = []
    for (pool_id, category), records in sorted(grouped.items()):
        if pool_id not in base_accuracy:
            continue
        base_acc, n_base = base_accuracy[pool_id]
        n_permuted = sum(r["n_samples"] for r in records)
        permuted_acc = sum(r["n_correct"] for r in records) / n_permuted
        decay = base_acc - permuted_acc
        problem = pool.loc[pool_id]
        first_variant = variants.loc[records[0]["item_id"]]
        rows.append({
            "model_id": model_id,
            "problem_key": problem["problem_key"],
            "problem": problem["problem"],
            "problem_id": pool_id,
            "dataset_id": problem["source"],
            "perturbation_key": category,
            "perturbation_types": [category],
            "is_perturbation_aggregate": False,
            "base_accuracy": base_acc,
            "permuted_accuracy": permuted_acc,
            "absolute_accuracy_decay": decay,
            "relative_accuracy_decay": decay / base_acc if base_acc else None,
            "n_base_predictions": n_base,
            "n_permuted_predictions": n_permuted,
            "n_detrimental_permutations": sum(r["n_correct"] / r["n_samples"] < base_acc for r in records),
            "is_robust": decay <= 0.0,
            "label_source": "self-generated",
            "is_eligible": base_acc >= args.min_base_accuracy,
            "is_clear_cut": decay <= 0.0 or decay > AMBIGUOUS_MAX_DECAY,
            "source_repo": "self-generated:e7",
            "compliance": first_variant["compliance"],
            "tier": problem["tier"],
        })

    if not rows:
        raise SystemExit("no labelled rows: are the base and variants records from the same run?")
    instances = build_instances(rows)
    train = [i for i in instances if i["is_eligible"] and i["is_clear_cut"]]

    args.out_dir.mkdir(parents=True, exist_ok=True)
    write_parquet(rows, args.out_dir / "perturbation_rows.parquet")
    write_parquet(instances, args.out_dir / "instances.parquet")
    write_parquet(train, args.out_dir / "instances_train.parquet")

    eligible = sum(acc >= args.min_base_accuracy for acc, _ in base_accuracy.values())
    per_category = Counter(r["perturbation_key"] for r in rows)
    robust_per_category = Counter(r["perturbation_key"] for r in rows if r["is_robust"])
    summary = {
        "model_id": model_id,
        "base_problems": len(base_accuracy),
        "records_dropped_ambiguous_pool_id": sum(r["pool_id"] in ambiguous for r in [*base.values(), *sampled.values()]),
        "base_eligible": eligible,
        "base_eligible_rate": round(eligible / len(base_accuracy), 4),
        "base_eligible_per_tier": dict(Counter(
            pool.loc[p, "tier"] for p, (acc, _) in base_accuracy.items() if acc >= args.min_base_accuracy
        )),
        "perturbation_rows": len(rows),
        "rows_per_category": dict(per_category),
        "robust_rate_per_category": {c: round(robust_per_category[c] / n, 4) for c, n in per_category.items()},
        "compliance_counts": dict(Counter(r["compliance"] for r in rows)),
        "instances": len(instances),
        "train_instances": len(train),
        "train_robust": sum(i["is_robust"] for i in train),
        "label_rule": "is_robust == (absolute_accuracy_decay <= 0); eligible base_accuracy >= "
        f"{args.min_base_accuracy}; clear-cut decay <= 0 or > {AMBIGUOUS_MAX_DECAY}",
    }
    (args.out_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    print(f"wrote {args.out_dir}")


if __name__ == "__main__":
    main()
