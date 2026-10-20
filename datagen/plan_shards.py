"""Show how instances distribute across shards, for assigning Kaggle accounts.

Does not load any model, so it is safe to run locally before committing GPU time.

Run:  python datagen/plan_shards.py --model-id deepseek-ai/DeepSeek-R1-0528-Qwen3-8B
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from pathlib import Path

import pyarrow.parquet as pq

from extract_features import assign_shards

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INSTANCES = ROOT / "data" / "small_track" / "instances_train.parquet"

# Account roles come from PLAN.md section 4.1.
ACCOUNTS = ["ISHRAQ", "DIDHITI", "TASHIN", "SALAM", "ISHMAM1", "DESIKALAKAR"]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--instances", type=Path, default=DEFAULT_INSTANCES)
    parser.add_argument("--model-id", default=None, help="restrict to one model")
    parser.add_argument("--num-shards", type=int, default=6)
    parser.add_argument(
        "--seconds-per-instance",
        type=float,
        default=1.0,
        help="forward-pass cost per instance; 1.0 s measured on Kaggle T4x2 in bf16 (E1)",
    )
    parser.add_argument("--shard-strategy", default="roundrobin", choices=["roundrobin", "hash"])
    args = parser.parse_args()

    rows = pq.read_table(args.instances).to_pylist()
    models = Counter(r["model_id"] for r in rows)
    print("instances per model:")
    for model, count in models.most_common():
        print(f"  {count:4d}  {model}")

    if args.model_id:
        rows = [r for r in rows if r["model_id"] == args.model_id]
        print(f"\nrestricted to {args.model_id}: {len(rows)} instances")
    if not rows:
        raise SystemExit("no instances selected")

    assignment = assign_shards(
        [r["problem_key"] for r in rows], args.num_shards, args.shard_strategy
    )
    buckets: dict[int, list[dict]] = defaultdict(list)
    for row in rows:
        buckets[assignment[row["problem_key"]]].append(row)

    print(f"\nshard plan ({args.num_shards} shards, {args.shard_strategy}):")
    print(f"  {'shard':>5}  {'account':<12} {'instances':>9} {'robust':>6} {'est. min':>8}")
    for index in range(args.num_shards):
        group = buckets.get(index, [])
        robust = sum(1 for r in group if r["is_robust"])
        account = ACCOUNTS[index] if index < len(ACCOUNTS) else "-"
        minutes = len(group) * args.seconds_per_instance / 60
        print(f"  {index:>5}  {account:<12} {len(group):>9} {robust:>6} {minutes:>8.1f}")

    sizes = [len(buckets.get(i, [])) for i in range(args.num_shards)]
    print(
        f"\n  balance: min={min(sizes)} max={max(sizes)} "
        f"total={sum(sizes)}  (~{max(sizes) * args.seconds_per_instance / 60:.0f} min critical path)"
    )


if __name__ == "__main__":
    main()
