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
* `absolute_accuracy_decay = base_accuracy - permuted_accuracy`; robust iff the drop is
  <= 0.10, spurious iff >= 0.25, and unlabelled in between (the organizers' own rule, stated
  in `aimo-interp/train-main-v2`; PROGRESS.md Finding 30).
* Eligible iff base_accuracy >= --min-base-accuracy (1.0, the organizers' curation).

`label_source` is "self-generated" and `compliance` is carried through, so a training set
can always be rebuilt from official data plus `distinct` categories only.

Run:  python datagen/perturb/label.py --base-glob "runs/e7-*/e7/base/shard_*/records.jsonl" \\
          --variants-glob "runs/e7-*/e7/variants/shard_*/records.jsonl"
"""

from __future__ import annotations

import argparse
import glob
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

from ingest_public import ROBUST_MAX_DROP, SPURIOUS_MIN_DROP, build_instances, write_parquet  # noqa: E402
from sample import DEFAULT_POOL, DEFAULT_VARIANTS, load_records  # noqa: E402

DEFAULT_OUT = ROOT / "data" / "e7_labels"


def load_base_records(pattern: str) -> list[dict]:
    """Every base record, without `load_records`' de-duplication by `item_id`.

    The base phase names items `orig:<pool_id>`, and `pool_id` is not unique (see `main`), so
    11 of the 208 Qwen records share an `item_id` with a *different* problem. Collapsing on it
    would discard one problem of each colliding pair along with the GPU time that produced it.
    De-duplication still has to happen -- a resumed session re-flushes items -- but on the
    (`item_id`, `expected_answer`) pair, which separates the two problems while still
    collapsing genuine repeats.
    """
    paths = sorted(glob.glob(pattern))
    if not paths:
        raise SystemExit(f"no records match {pattern}")
    records: dict[tuple[str, object], dict] = {}
    for path in paths:
        for line in Path(path).read_text(encoding="utf-8").splitlines():
            if line.strip():
                row = json.loads(line)
                records[(row["item_id"], row["expected_answer"])] = row
    return list(records.values())


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--base-glob", required=True)
    parser.add_argument("--variants-glob", required=True)
    parser.add_argument("--pool", type=Path, default=DEFAULT_POOL)
    parser.add_argument("--variants", type=Path, default=DEFAULT_VARIANTS)
    parser.add_argument("--min-base-accuracy", type=float, default=1.0)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()

    base = load_base_records(args.base_glob)
    sampled = load_records(args.variants_glob)
    pool = pd.read_parquet(args.pool)
    variants = pd.read_parquet(args.variants).set_index("variant_id")
    model_ids = {r["model_id"] for r in [*base, *sampled.values()]}
    if len(model_ids) != 1:
        raise SystemExit(f"records mix models: {sorted(model_ids)}")
    model_id = model_ids.pop()

    # `pool_id` is NOT unique in pools built before the `pool.py` fix: the upstream AMC table
    # reuses its `id` column, so 34 ids name two different problems each. Keying anything on it
    # silently pairs one problem's base accuracy with another's perturbations. `problem_key`
    # (SHA-1 of the statement) is unique across the pool, so everything below joins on that.
    #
    # Main now disambiguates at the source by suffixing colliding ids with `:<problem_key[:8]>`,
    # and drops records filed under the old ambiguous ids. We resolve them instead: base records
    # carry no `problem_key`, but (`pool_id`, `expected_answer`) identifies a pool row uniquely --
    # no colliding pair shares an answer -- so records sampled before the fix are recovered rather
    # than thrown away, along with the GPU time that produced them. Both spellings are registered,
    # so this works whichever side of the `pool.py` fix a record was sampled on.
    if pool["problem_key"].duplicated().any():
        raise SystemExit("pool has duplicate problem_key values; the disambiguation below is unsafe")
    key_by_pool_answer: dict[tuple[str, object], str] = {}
    for row in pool.itertuples(index=False):
        key_by_pool_answer[(row.pool_id, row.answer)] = row.problem_key
        suffix = f":{row.problem_key[:8]}"
        if row.pool_id.endswith(suffix):  # pool built after the fix; accept the pre-fix spelling too
            key_by_pool_answer.setdefault((row.pool_id[: -len(suffix)], row.answer), row.problem_key)
    pool = pool.set_index("problem_key")

    base_accuracy: dict[str, tuple[float, int]] = {}
    unresolved_base = 0
    for record in base:
        if not record["n_samples"]:
            continue
        problem_key = key_by_pool_answer.get((record["pool_id"], record["expected_answer"]))
        if problem_key is None:
            unresolved_base += 1
            continue
        base_accuracy[problem_key] = (record["n_correct"] / record["n_samples"], record["n_samples"])
    if unresolved_base:
        print(f"warning: {unresolved_base} base records did not match a pool problem and were skipped")

    missing_variants = [vid for vid in sampled if vid not in variants.index]
    if missing_variants:
        raise SystemExit(f"{len(missing_variants)} sampled variants are not in {args.variants}; regenerate with the same seed")

    # A variant record's own `pool_id` is equally ambiguous, but its `variant_id` is unique and
    # the variants table carries the `problem_key` it was derived from.
    grouped: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for variant_id, record in sampled.items():
        grouped[(variants.at[variant_id, "problem_key"], record["category"])].append(record)

    rows = []
    for (problem_key, category), records in sorted(grouped.items()):
        if problem_key not in base_accuracy:
            continue
        base_acc, n_base = base_accuracy[problem_key]
        n_permuted = sum(r["n_samples"] for r in records)
        permuted_acc = sum(r["n_correct"] for r in records) / n_permuted
        decay = base_acc - permuted_acc
        problem = pool.loc[problem_key]
        first_variant = variants.loc[records[0]["item_id"]]
        rows.append({
            "model_id": model_id,
            "problem_key": problem_key,
            "problem": problem["problem"],
            "problem_id": problem["pool_id"],
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
            "is_robust": decay <= ROBUST_MAX_DROP,
            "label_source": "self-generated",
            "is_eligible": base_acc >= args.min_base_accuracy,
            "is_clear_cut": decay <= ROBUST_MAX_DROP or decay >= SPURIOUS_MIN_DROP,
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
        # Main reports records dropped for an ambiguous `pool_id`; we resolve them instead,
        # so the number that matters is the one we could not place at all.
        "base_records_unresolved": unresolved_base,
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
        "label_rule": f"is_robust == (max drop <= {ROBUST_MAX_DROP}); spurious iff >= {SPURIOUS_MIN_DROP}; "
        f"eligible base_accuracy >= {args.min_base_accuracy}",
    }
    (args.out_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    print(f"wrote {args.out_dir}")


if __name__ == "__main__":
    main()
