"""Generate perturbed variants of the olympiad pool (PLAN.md §3.4, feeds E7).

Reads `data/olympiad_pool/pool.parquet` (built by `pool.py`), applies every category in
`transforms.CATEGORIES`, and writes one row per variant with its exact expected answer and
compliance tag:

    data/perturbations/variants.parquet
    data/perturbations/summary.json    applicability and counts per category and tier

Randomness is seeded per (seed, problem, category, attempt), so any single variant is
reproducible without regenerating the rest. Labels do not exist yet; E7 samples the target
model on originals and variants to produce them. It must keep only `compliance == "distinct"`
rows unless the organizers approve the others (see transforms.py).

Run:  python datagen/perturb/generate.py
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path

import pandas as pd

PERTURB = Path(__file__).resolve().parent
ROOT = PERTURB.parents[1]
if str(PERTURB) not in sys.path:
    sys.path.insert(0, str(PERTURB))

from transforms import CATEGORIES  # noqa: E402

DEFAULT_POOL = ROOT / "data" / "olympiad_pool" / "pool.parquet"
DEFAULT_OUT = ROOT / "data" / "perturbations"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--pool", type=Path, default=DEFAULT_POOL)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--categories", nargs="+", choices=sorted(CATEGORIES), default=sorted(CATEGORIES))
    parser.add_argument("--per-category", type=int, default=2, help="distinct variants per problem per category")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--examples", type=int, default=1, help="examples to print per category")
    args = parser.parse_args()

    if not args.pool.is_file():
        raise SystemExit(f"{args.pool} missing; run datagen/perturb/pool.py first")
    pool = pd.read_parquet(args.pool)

    rows = []
    applicable: dict[str, Counter] = defaultdict(Counter)
    for problem in pool.itertuples(index=False):
        for name in args.categories:
            category = CATEGORIES[name]
            seen = {problem.problem.strip()}
            produced = 0
            for attempt in range(args.per_category * 4):
                rng = random.Random(f"{args.seed}:{problem.pool_id}:{name}:{attempt}")
                variant = category.transform(problem.problem, int(problem.answer), rng)
                if variant is None:
                    break  # applicability does not depend on the draw
                if variant.problem.strip() in seen:
                    continue
                seen.add(variant.problem.strip())
                produced += 1
                rows.append({
                    "variant_id": hashlib.sha1(f"{problem.pool_id}|{name}|{variant.problem}".encode()).hexdigest()[:16],
                    "pool_id": problem.pool_id,
                    "source": problem.source,
                    "tier": problem.tier,
                    "problem_key": problem.problem_key,
                    "original_problem": problem.problem,
                    "original_answer": int(problem.answer),
                    "category": name,
                    "compliance": category.compliance,
                    "variant_problem": variant.problem,
                    "expected_answer": variant.expected_answer,
                    "params_json": json.dumps(variant.params, sort_keys=True),
                    "needs_verification": variant.needs_verification,
                    "seed": args.seed,
                })
                if produced == args.per_category:
                    break
            if produced:
                applicable[name][problem.tier] += 1

    variants = pd.DataFrame(rows)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    variants.to_parquet(args.out_dir / "variants.parquet", index=False)

    tiers = pool["tier"].value_counts().to_dict()
    summary = {
        "pool": str(args.pool),
        "pool_size": len(pool),
        "pool_per_tier": tiers,
        "seed": args.seed,
        "per_category": args.per_category,
        "categories": {
            name: {
                "compliance": CATEGORIES[name].compliance,
                "rationale": CATEGORIES[name].rationale,
                "problems_applicable": dict(applicable[name]),
                "problems_applicable_total": sum(applicable[name].values()),
                "variants": int((variants["category"] == name).sum()) if len(variants) else 0,
            }
            for name in args.categories
        },
        "variants_total": len(variants),
        "variants_labelable_now": int((variants["compliance"] == "distinct").sum()) if len(variants) else 0,
    }
    (args.out_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(f"pool: {len(pool)} problems {tiers}")
    for name, stats in summary["categories"].items():
        print(
            f"  {name:20s} [{stats['compliance']:8s}] applicable to {stats['problems_applicable_total']:4d} "
            f"problems {stats['problems_applicable']}, {stats['variants']} variants"
        )
    print(f"variants: {len(variants)} total, {summary['variants_labelable_now']} labelable without organizer confirmation")
    for name in args.categories:
        sample = variants[variants["category"] == name].head(args.examples) if len(variants) else []
        for row in getattr(sample, "itertuples", lambda **_: [])(index=False):
            print(f"\n--- {name} ({row.pool_id}, answer {row.original_answer} -> {row.expected_answer}) ---")
            print(row.variant_problem[:700])
    print(f"\nwrote {args.out_dir}")


if __name__ == "__main__":
    main()
