"""E55: every problem the collaborator's self-labelling may cover, paired with all four models, unlabelled.

The per-problem predictor of PLAN section 12.8 will be trained on self-labels for some subset of these
problems, so its one-pass features are extracted for the whole union now (`protocols/E55_PROTOCOL.md`
section 1): the 36 `train-main-v2` problems, ReliableMath's AIME and AMC originals, and our olympiad
pool. Problems are deduplicated by `problem_key`, the whitespace-normalised hash every other table
uses. No label, answer or source-specific column is carried into the instance table.

Writes data/e55/instances.parquet (model_id, problem_key, problem, dataset_id) and
data/e55/summary.json.

Run:  python datagen/build_e55_instances.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd
from huggingface_hub import hf_hub_download

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "datagen"))

from ingest_public import problem_key  # noqa: E402

OFFICIAL = ("aimo-interp/train-main-v2", "data/train-00000-of-00001.parquet", "23791d9b120132087877b0459f953d5959fc8455")
RELIABLEMATH = ("BeyondHsueh/ReliableMath", "c059d6e6dd612c0dd21c2a4844d508b482c0d4b2")
POOL = ROOT / "data" / "olympiad_pool" / "pool.parquet"
MODELS = ("Qwen/Qwen3.5-4B", "Skywork/Skywork-OR1-Math-7B", "allenai/Olmo-3-7B-Think",
          "deepseek-ai/DeepSeek-R1-0528-Qwen3-8B")
OUT = ROOT / "data" / "e55"


def main() -> None:
    repo, filename, revision = OFFICIAL
    official = pd.read_parquet(hf_hub_download(repo, filename, repo_type="dataset", revision=revision))
    parts = [pd.DataFrame({"problem": official["problem"].unique(), "dataset_id": "train-main-v2"})]
    repo, revision = RELIABLEMATH
    for name in ("aime", "amc"):
        path = Path(hf_hub_download(repo, f"solve/{name}.json", repo_type="dataset", revision=revision))
        frame = pd.DataFrame(json.loads(path.read_text(encoding="utf-8")))
        parts.append(pd.DataFrame({"problem": frame["question"], "dataset_id": f"reliablemath-{name}"}))
    pool = pd.read_parquet(POOL)
    parts.append(pd.DataFrame({"problem": pool["problem"], "dataset_id": "pool-" + pool["source"].astype(str)}))

    problems = pd.concat(parts, ignore_index=True)
    problems["problem_key"] = problems["problem"].map(problem_key)
    by_source = problems.groupby("dataset_id").size().to_dict()
    problems = problems.drop_duplicates("problem_key").reset_index(drop=True)  # first source wins

    instances = pd.DataFrame([
        {"model_id": m, "problem_key": r.problem_key, "problem": r.problem, "dataset_id": r.dataset_id}
        for m in MODELS for r in problems.itertuples(index=False)
    ])
    OUT.mkdir(parents=True, exist_ok=True)
    instances.to_parquet(OUT / "instances.parquet", index=False)
    summary = {"problems": len(problems), "instances": len(instances), "rows_per_source_before_dedup": by_source,
               "problems_per_source_after_dedup": problems.groupby("dataset_id").size().to_dict()}
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
