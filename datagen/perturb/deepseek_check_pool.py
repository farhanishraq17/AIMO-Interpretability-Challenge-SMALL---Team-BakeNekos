"""E20: a pool of DeepSeek's officially labelled problems, to test whether our self-labels agree.

Self-generated labels (E15-E19) measure robustness under `answer_modulus`, which is not one of
the official perturbation types. Before training on them, check them where the answer is
known: DeepSeek-R1-0528-Qwen3-8B's 138 training problems carry official labels, and 137 of them
are Hendrycks MATH problems whose reference solutions end in a `\\boxed{}` answer.

This writes those problems in the olympiad-pool schema (`pool.py`), so `generate.py`,
`sample.py` and `label.py` run on them unchanged, plus the official label for the comparison.
Only problems `answer_modulus` applies to are kept (an integer answer of at least its smallest
modulus), so no GPU time goes to problems that could never get a self-label.

Run:  python datagen/perturb/deepseek_check_pool.py
      python datagen/perturb/generate.py --pool data/deepseek_check/pool.parquet \\
          --out-dir data/deepseek_check --categories answer_modulus
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd
from huggingface_hub import HfApi, hf_hub_download

PERTURB = Path(__file__).resolve().parent
ROOT = PERTURB.parents[1]
sys.path.insert(0, str(PERTURB))

from answers import last_boxed, normalize_integer  # noqa: E402
from pool import problem_key  # noqa: E402
from transforms import SMALL_MODULI  # noqa: E402

DEEPSEEK = "deepseek-ai/DeepSeek-R1-0528-Qwen3-8B"
MATH_REPO = "EleutherAI/hendrycks_math"
MATH_REVISION = "21a5633873b6a120296cce3e2df9d5550074f4a3"
INSTANCES = ROOT / "data" / "small_track" / "instances_train.parquet"
OUT = ROOT / "data" / "deepseek_check"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out-dir", type=Path, default=OUT)
    args = parser.parse_args()

    files = [f for f in HfApi().list_repo_files(MATH_REPO, repo_type="dataset", revision=MATH_REVISION)
             if f.endswith(".parquet")]
    math = pd.concat(
        pd.read_parquet(hf_hub_download(MATH_REPO, f, repo_type="dataset", revision=MATH_REVISION)) for f in files
    )
    math["problem_key"] = math["problem"].map(problem_key)
    math["answer"] = math["solution"].map(lambda s: normalize_integer(last_boxed(s)))
    math = math.drop_duplicates("problem_key")[["problem_key", "answer"]]

    labelled = pd.read_parquet(INSTANCES)
    labelled = labelled[labelled["model_id"] == DEEPSEEK]
    pool = labelled.merge(math, on="problem_key", how="inner").dropna(subset=["answer"])
    pool = pool[pool["answer"] >= SMALL_MODULI[0]]  # answer_modulus returns None below this
    pool = pd.DataFrame({
        "pool_id": "math:" + pool["problem_key"],
        "source": pool["dataset_id"],
        "tier": "math",
        "source_repo": MATH_REPO,
        "source_revision": MATH_REVISION,
        "problem": pool["problem"],
        "answer": pool["answer"].astype("int64"),
        "problem_key": pool["problem_key"],
        "official_is_robust": pool["is_robust"].astype(bool),
    }).sort_values("pool_id").reset_index(drop=True)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    pool.to_parquet(args.out_dir / "pool.parquet", index=False)
    print(f"{len(labelled)} labelled DeepSeek problems; {len(pool)} with an integer answer >= {SMALL_MODULI[0]} "
          f"({int(pool['official_is_robust'].sum())} officially robust) -> {args.out_dir / 'pool.parquet'}")


if __name__ == "__main__":
    main()
