"""E24a: recover answers for the official problems from unanimous samples.

`train-main-v2` ships no answers, and `answer_modulus` variants need one. E24a sampled every
labelled official problem 8 times with Skywork (the model most often correct in E16) under a
placeholder answer, so each record's `answers` field holds 8 parsed answers. A problem gets a
reference answer only when **all 8 agree**; the 6 problems our pool already knows check that
this rule gives the right answer, and any disagreement there stops the run.

Writes `pool_keyed.parquet` (the pool with reference answers filled in and unresolved problems
dropped, for `generate.py --categories answer_modulus`) and one `pool_<model>.parquet` per
small-track model, holding only the problems that model is officially labelled on (E24b).

Run:  python datagen/perturb/answer_key.py
"""

from __future__ import annotations

import argparse
import sys
from collections import Counter
from pathlib import Path

import pandas as pd

PERTURB = Path(__file__).resolve().parent
ROOT = PERTURB.parents[1]
sys.path.insert(0, str(PERTURB))

from sample import load_records  # noqa: E402
from transforms import SMALL_MODULI  # noqa: E402

POOL = ROOT / "data" / "official_check" / "pool.parquet"
MODEL_SHORT = {
    "Qwen/Qwen3.5-4B": "qwen35",
    "deepseek-ai/DeepSeek-R1-0528-Qwen3-8B": "deepseek",
    "Skywork/Skywork-OR1-Math-7B": "skywork",
    "allenai/Olmo-3-7B-Think": "olmo-think",
}
RECORDS = "runs/e24a-answer-key-skywork/e24a-skywork/base/shard_*/records.jsonl"


def consensus(answers: list) -> int | None:
    """The answer every sample gave, or None if any sample disagreed or failed to answer."""
    counts = Counter(answers)
    if len(counts) == 1 and None not in counts:
        return int(next(iter(counts)))
    return None


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--records", default=str(ROOT / RECORDS))
    parser.add_argument("--pool", type=Path, default=POOL)
    parser.add_argument("--out", type=Path, default=POOL.with_name("pool_keyed.parquet"))
    args = parser.parse_args()

    pool = pd.read_parquet(args.pool).set_index("pool_id")
    found = {r["pool_id"]: consensus(r["answers"]) for r in load_records(args.records).values()}

    wrong = [(pid, found[pid], int(pool.at[pid, "answer"]))
             for pid in found if pool.at[pid, "answer_known"] and found[pid] is not None
             and found[pid] != pool.at[pid, "answer"]]
    checked = sum(1 for pid in found if pool.at[pid, "answer_known"] and found[pid] is not None)
    if wrong:
        raise SystemExit(f"unanimous samples contradict {len(wrong)} known answers: {wrong}")
    print(f"known answers reproduced by unanimous samples: {checked} of {int(pool['answer_known'].sum())}")

    keyed = pool.copy()
    for pid, answer in found.items():
        if not keyed.at[pid, "answer_known"]:
            keyed.at[pid, "answer"] = -1 if answer is None else answer
    keyed = keyed[keyed["answer"] >= 0].reset_index()
    usable = int((keyed["answer"] >= SMALL_MODULI[0]).sum())
    keyed.to_parquet(args.out, index=False)
    print(f"{len(keyed)} of {len(pool)} problems have a reference answer; {usable} are >= {SMALL_MODULI[0]}, "
          f"so answer_modulus applies -> {args.out}")

    # E24b samples each model on its own labelled problems only, so each gets its own pool.
    labels = pd.read_parquet(args.pool.with_name("official_labels.parquet"))
    for model_id, short in MODEL_SHORT.items():
        mine = keyed[keyed["problem_key"].isin(labels.loc[labels["model_id"] == model_id, "problem_key"])]
        mine = mine[mine["answer"] >= SMALL_MODULI[0]]
        mine.to_parquet(args.out.with_name(f"pool_{short}.parquet"), index=False)
        print(f"  {model_id}: {len(mine)} problems with an answer_modulus-ready answer")


if __name__ == "__main__":
    main()
