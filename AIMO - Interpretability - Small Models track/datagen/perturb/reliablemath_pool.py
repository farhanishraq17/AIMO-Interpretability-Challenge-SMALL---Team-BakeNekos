"""E52: ReliableMath's AMC 2023 problems and their contradiction rewrites, in the sampler's pool format.

Finding 76: on olympiad problems a model has solved, the family that decides the official label is
`expert_no_solution`, an added premise that contradicts the rest. ReliableMath (CC-BY-4.0) pairs
each AMC 2023 problem with human-checked unsolvable rewrites, some made by adding a contradictory
condition, so that family can be measured on our models without writing a single variant.

Only `contradict` rewrites are kept: the organizers' family adds a premise rather than removing
one. Every contradict rewrite of a problem is used, so a problem's detection rate averages over
several contradictions instead of hanging on one rewrite's wording. A rewrite's `expected_answer`
is the original answer, which makes the sampler's `n_correct` count the samples that ignored the
contradiction and answered as if nothing had changed.

Writes data/reliablemath/{pool,variants,instances}.parquet. `instances` lists every
(small-track model, problem) pair for the one-pass extractors, with no label anywhere.

Run:  python datagen/perturb/reliablemath_pool.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd
from huggingface_hub import hf_hub_download

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "datagen"))

from ingest_public import problem_key  # noqa: E402

REPO, REVISION = "BeyondHsueh/ReliableMath", "c059d6e6dd612c0dd21c2a4844d508b482c0d4b2"
OUT = ROOT / "data" / "reliablemath"
MODELS = ("Qwen/Qwen3.5-4B", "Skywork/Skywork-OR1-Math-7B", "allenai/Olmo-3-7B-Think",
          "deepseek-ai/DeepSeek-R1-0528-Qwen3-8B")


def integer(value: str) -> int | None:
    try:
        number = float(str(value).strip())
    except ValueError:
        return None
    return int(number) if number.is_integer() else None


def main() -> None:
    download = lambda name: hf_hub_download(REPO, name, repo_type="dataset", revision=REVISION)  # noqa: E731
    solve = pd.DataFrame(json.loads(Path(download("solve/amc.json")).read_text(encoding="utf-8")))
    unsol = pd.DataFrame(json.loads(Path(download("unsol/amc.json")).read_text(encoding="utf-8")))

    solve["answer"] = solve["ground_truth"].map(integer)
    solve = solve[solve["answer"].notna()].astype({"answer": int, "id": str})
    pool = pd.DataFrame({
        "pool_id": "rm-amc:" + solve["id"],
        "problem": solve["question"],
        "answer": solve["answer"],
        "tier": "amc",
        "source": "reliablemath-amc-2023",
    })
    pool["problem_key"] = pool["problem"].map(problem_key)

    parts = unsol["data_id"].str.extract(r"^amc_(?P<id>\d+)_(?P<kind>[a-z]+)_(?P<n>\d+)$")
    unsol = unsol.join(parts)
    unsol = unsol[(unsol["kind"] == "contradict") & ("rm-amc:" + unsol["id"]).isin(pool["pool_id"])]
    variants = pd.DataFrame({
        "variant_id": "rm-amc:" + unsol["data_id"],
        "pool_id": "rm-amc:" + unsol["id"],
        "category": "contradict",
        "tier": "amc",
        "variant_problem": unsol["rewritten_question"],
        "compliance": "distinct",
        "difficulty_eval": unsol["difficulty_eval"].astype(str),
    })
    variants = variants.merge(pool[["pool_id", "answer"]].rename(columns={"answer": "expected_answer"}), on="pool_id")
    pool = pool[pool["pool_id"].isin(variants["pool_id"])].reset_index(drop=True)

    instances = pd.DataFrame([
        {"model_id": m, "problem_key": r.problem_key, "problem": r.problem, "dataset_id": r.source}
        for m in MODELS for r in pool.itertuples(index=False)
    ])

    assert pool["pool_id"].is_unique and pool["problem_key"].is_unique and variants["variant_id"].is_unique
    OUT.mkdir(parents=True, exist_ok=True)
    pool.to_parquet(OUT / "pool.parquet", index=False)
    variants.sort_values("variant_id").to_parquet(OUT / "variants.parquet", index=False)
    instances.to_parquet(OUT / "instances.parquet", index=False)
    print(f"{len(pool)} AMC problems, {len(variants)} contradict rewrites "
          f"({len(variants) / len(pool):.2f} per problem), {len(instances)} extraction instances -> {OUT}")


if __name__ == "__main__":
    main()
