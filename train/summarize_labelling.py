"""E15-E19: how far self-labelling has got for the three models without public labels.

The E7 pipeline labels a (model, problem) pair in two steps: a base phase finds the problems
the model solves in every sample (the organizers' eligibility rule), and a variants phase
samples the compliant `answer_modulus` variants of those problems. This summarises the base
phase from the downloaded records (deduplicated across sessions and shards by item id), and
the labels from `datagen/perturb/label.py`'s output, so the label rule lives in one place.
Records under pool ids the old pool reused for two problems are excluded (Finding 29).

Run:  python train/summarize_labelling.py
"""

from __future__ import annotations

import argparse
import glob
import json
import statistics
from collections import defaultdict
from pathlib import Path

import pandas as pd

from results_io import upsert_section

ROOT = Path(__file__).resolve().parents[1]
RUNS = ROOT / "runs"
VARIANTS = ROOT / "data" / "perturbations" / "variants.parquet"
LABELS = ROOT / "data" / "e7_labels"  # label.py --out-dir data/e7_labels/<short>
MODELS = {  # base-phase runs, variants-phase runs, label.py output
    "Qwen/Qwen3.5-4B": ("e15-base-qwen35-*/e15-qwen35", "e18-variants-qwen35/e18-qwen35", "qwen35"),
    "Skywork/Skywork-OR1-Math-7B": ("e16-base-skywork/e16-skywork", "e19-variants-skywork/e19-skywork", "skywork"),
    "allenai/Olmo-3-7B-Think": ("e17-base-olmo-think/e17-olmo-think", None, None),
}
TIERS = ("amc", "aime", "olympiad")


def load(pattern: str) -> list[dict]:
    """Records under runs/<pattern>/*/shard_*/records.jsonl, one per item id (the latest wins)."""
    rows: dict[str, dict] = {}
    for path in sorted(glob.glob(str(RUNS / pattern / "shard_*" / "records.jsonl"))):
        for line in Path(path).read_text(encoding="utf-8").splitlines():
            if line.strip():
                row = json.loads(line)
                rows[row["item_id"]] = row
    return list(rows.values())


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, default=ROOT / "RESULTS.md")
    args = parser.parse_args()

    pool = pd.read_parquet(ROOT / "data" / "olympiad_pool" / "pool.parquet")
    pool_size = len(pool)
    ambiguous = set(pool.loc[pool["pool_id"].duplicated(keep=False), "pool_id"])
    variants = pd.read_parquet(VARIANTS)
    variants = variants[variants["compliance"] == "distinct"]

    base_lines = [
        "| Model | Sampled | Eligible | Rate | Samples hitting the cap | Median tokens per sample | "
        + " | ".join(f"{t.upper() if t != 'olympiad' else 'Olympiad'} eligible/sampled" for t in TIERS) + " |",
        "| --- |" + " --- |" * (5 + len(TIERS)),
    ]
    label_lines = [
        "| Model | Compliant variants of eligible problems | Problems covered | Variants sampled | "
        "Problems labelled | Robust (all) | Clear-cut (training) | Robust (clear-cut) |",
        "| --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for model_id, (base_dir, variant_dir, label_dir) in MODELS.items():
        base = [r for r in load(f"{base_dir}/base") if r["pool_id"] not in ambiguous]
        if not base:
            base_lines.append(f"| `{model_id}` | 0 | — | — | — | — |" + " — |" * len(TIERS))
            continue
        eligible = {r["pool_id"] for r in base if r["n_samples"] and r["n_correct"] == r["n_samples"]}
        tokens = [t for r in base for t in r["completion_tokens"]]
        hit = sum(r["n_hit_length"] for r in base) / sum(r["n_samples"] for r in base)
        by_tier: dict[str, list[int]] = defaultdict(lambda: [0, 0])
        for r in base:
            by_tier[r["tier"]][0] += 1
            by_tier[r["tier"]][1] += r["pool_id"] in eligible
        base_lines.append(
            f"| `{model_id}` | {len(base)} of {pool_size} | {len(eligible)} | {len(eligible) / len(base):.0%} | "
            f"{hit:.0%} | {statistics.median(tokens):,.0f} | "
            + " | ".join(f"{by_tier[t][1]}/{by_tier[t][0]}" for t in TIERS) + " |"
        )

        selected = variants[variants["pool_id"].isin(eligible)]
        sampled = [r for r in load(f"{variant_dir}/variants") if r["pool_id"] not in ambiguous] if variant_dir else []
        labels = LABELS / label_dir / "instances.parquet" if label_dir else None
        if labels and labels.is_file():
            instances = pd.read_parquet(labels)
            train = pd.read_parquet(labels.with_name("instances_train.parquet"))
            counts = (f"{len(instances)} | {int(instances['is_robust'].sum())} ({instances['is_robust'].mean():.0%}) | "
                      f"{len(train)} | {int(train['is_robust'].sum())}")
        else:
            counts = "— | — | — | —"
        label_lines.append(
            f"| `{model_id}` | {len(selected)} | {selected['pool_id'].nunique()} | {len(sampled)} | {counts} |"
        )

    body = "\n".join([
        f"Olympiad pool: {pool_size} problems (AMC, AIME, OlympiadBench). k=4 samples per problem; vLLM, fp16, "
        "2x T4. Eligible means all 4 base samples correct, the organizers' rule. Only `answer_modulus` "
        "variants (compliance `distinct`) are sampled for labels, per the extra-data rule.",
        "",
        "### Base phase",
        "",
        *base_lines,
        "",
        "### Variants phase and labels",
        "",
        *label_lines,
        "",
        "Labels come from `datagen/perturb/label.py`: robust iff decay <= 0, where decay pools every "
        "sample of every variant; clear-cut drops the ambiguous 0 < decay <= 0.3 band, as the public data "
        "does. With 4 samples per variant, one wrong sample already lands a problem in that band. "
        f"{len(ambiguous)} AMC pool ids that the old pool reused for two problems are excluded (Finding 29). "
        "Olmo-3-7B-Think has no variants run: on 2x T4 it fits only at a 12k cap (Finding 26). "
        "Regenerate with `label.py`, then `train/summarize_labelling.py`.",
    ])
    print(body)
    upsert_section(args.results, "E15-E19: self-labelling progress", body)


if __name__ == "__main__":
    main()
