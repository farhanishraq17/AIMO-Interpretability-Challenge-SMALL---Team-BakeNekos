"""E15-E19: how far self-labelling has got for the three models without public labels.

The E7 pipeline labels a (model, problem) pair in two steps: a base phase finds the problems
the model solves in every sample (the organizers' eligibility rule), and a variants phase
samples the compliant `answer_modulus` variants of those problems. This summarises both from
the downloaded records, deduplicating across sessions and shards by item id, so the yields
quoted in PROGRESS.md (Findings 25-26) regenerate from the data.

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
MODELS = {
    "Qwen/Qwen3.5-4B": ("e15-base-qwen35-*/e15-qwen35", "e18-variants-qwen35/e18-qwen35"),
    "Skywork/Skywork-OR1-Math-7B": ("e16-base-skywork/e16-skywork", "e19-variants-skywork/e19-skywork"),
    "allenai/Olmo-3-7B-Think": ("e17-base-olmo-think/e17-olmo-think", None),
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

    pool_size = len(pd.read_parquet(ROOT / "data" / "olympiad_pool" / "pool.parquet"))
    variants = pd.read_parquet(VARIANTS)
    variants = variants[variants["compliance"] == "distinct"]

    base_lines = [
        "| Model | Sampled | Eligible | Rate | Samples hitting the cap | Median tokens per sample | "
        + " | ".join(f"{t.upper() if t != 'olympiad' else 'Olympiad'} eligible/sampled" for t in TIERS) + " |",
        "| --- |" + " --- |" * (5 + len(TIERS)),
    ]
    label_lines = [
        "| Model | Compliant variants of eligible problems | Problems covered | Variants sampled | Labelled | Robust |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    for model_id, (base_dir, variant_dir) in MODELS.items():
        base = load(f"{base_dir}/base")
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
        sampled = load(f"{variant_dir}/variants") if variant_dir else []
        # A problem is labelled once every one of its compliant variants has been sampled.
        done = defaultdict(list)
        for r in sampled:
            done[r["pool_id"]].append(r["n_correct"] / r["n_samples"])
        expected = selected.groupby("pool_id").size()
        labelled = {p: accs for p, accs in done.items() if len(accs) == expected.get(p, -1)}
        robust = sum(min(accs) >= 1.0 for accs in labelled.values())  # decay <= 0 against a base accuracy of 1
        label_lines.append(
            f"| `{model_id}` | {len(selected)} | {selected['pool_id'].nunique()} | {len(sampled)} | "
            f"{len(labelled)} | {robust if labelled else '—'} |"
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
        "A problem counts as labelled once all its compliant variants are sampled, and as robust when "
        "no variant loses accuracy against the base (decay <= 0). Olmo-3-7B-Think has no variants run: "
        "on 2x T4 it fits only at a 12k cap and runs one or two sequences at a time (Finding 26). "
        "Regenerate with `train/summarize_labelling.py`.",
    ])
    print(body)
    upsert_section(args.results, "E15-E19: self-labelling progress", body)


if __name__ == "__main__":
    main()
