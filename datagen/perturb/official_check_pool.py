"""E24: the labelled olympiad problems of `train-main-v2`, in pool form, for the G-V2 test.

E20 found that `answer_modulus` self-labels disagree with official labels (kappa -0.135), but on
Hendrycks MATH problems, and every other signal we have broke on the MATH -> olympiad shift.
Gate G-V2 repeats the comparison where it matters: on the 20 olympiad problems `train-main-v2`
labels for a small-track model, under the current official perturbation families.

`train-main-v2` ships no answers, and `answer_modulus` needs one. 8 of the 20 problems are in
our olympiad pool, so their answers are known; the rest get the placeholder -1, and E24a
recovers them as the unanimous answer of 8 Skywork samples (`answer_key.py`). Rows keep the
pool schema, so `sample.py`, `generate.py` and `label.py` run on them unchanged.

Run:  python datagen/perturb/official_check_pool.py
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd
from huggingface_hub import hf_hub_download

PERTURB = Path(__file__).resolve().parent
ROOT = PERTURB.parents[1]
sys.path.insert(0, str(PERTURB))

from pool import problem_key  # noqa: E402

DATASET = "aimo-interp/train-main-v2"
REVISION = "23791d9b120132087877b0459f953d5959fc8455"
PARQUET = "data/train-00000-of-00001.parquet"
SMALL_TRACK = {
    "Qwen/Qwen3.5-4B",
    "Skywork/Skywork-OR1-Math-7B",
    "allenai/Olmo-3-7B-Think",
    "deepseek-ai/DeepSeek-R1-0528-Qwen3-8B",
}
OLYMPIAD_POOL = ROOT / "data" / "olympiad_pool" / "pool.parquet"
OUT = ROOT / "data" / "official_check"
UNKNOWN_ANSWER = -1


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out-dir", type=Path, default=OUT)
    args = parser.parse_args()

    table = pd.read_parquet(hf_hub_download(DATASET, PARQUET, repo_type="dataset", revision=REVISION))
    labelled = table[table["model_id"].isin(SMALL_TRACK) & table["is_robust"].notna()].copy()
    labelled["problem_key"] = labelled["problem"].map(problem_key)
    problems = labelled.drop_duplicates("problem_key")[["problem_key", "problem"]]

    known = pd.read_parquet(OLYMPIAD_POOL)[["problem_key", "answer"]]
    pool = problems.merge(known, on="problem_key", how="left")
    pool = pd.DataFrame({
        "pool_id": "official:" + pool["problem_key"],
        "source": "train-main-v2",
        "tier": "official",
        "source_repo": DATASET,
        "source_revision": REVISION,
        "problem": pool["problem"],
        "answer": pool["answer"].fillna(UNKNOWN_ANSWER).astype("int64"),
        "answer_known": pool["answer"].notna(),
        "problem_key": pool["problem_key"],
    }).sort_values("pool_id").reset_index(drop=True)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    pool.to_parquet(args.out_dir / "pool.parquet", index=False)
    labels = labelled[["model_id", "problem_key", "is_robust", "max_drop"]]
    labels.to_parquet(args.out_dir / "official_labels.parquet", index=False)
    print(f"{len(pool)} labelled official problems ({int(pool['answer_known'].sum())} with a known answer), "
          f"{len(labels)} (model, problem) labels -> {args.out_dir}")


if __name__ == "__main__":
    main()
