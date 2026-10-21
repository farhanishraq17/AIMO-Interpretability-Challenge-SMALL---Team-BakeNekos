"""E62: the problems for the familiarity-versus-detection report experiment (`protocols/E62_PROTOCOL.md`).

E55 scored 548 problems with every model, so their familiarity is already known; E62 needs each one's reference
answer to write contradictions against. This keeps the E55 problems that have an answer in the olympiad pool and
are not official `train-main-v2` rows, takes E55's text (the text the NLL was computed on), and sorts by
`problem_key`, the order generation and sampling follow.

Run:  python datagen/perturb/e62_pool.py
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "data" / "e62" / "pool.parquet"


def main() -> None:
    e55 = pd.read_parquet(ROOT / "data" / "e55" / "instances.parquet").drop_duplicates("problem_key")
    e55 = e55[e55["dataset_id"] != "train-main-v2"][["problem_key", "problem", "dataset_id"]]
    pool = pd.read_parquet(ROOT / "data" / "olympiad_pool" / "pool.parquet").drop_duplicates("problem_key")
    out = e55.merge(pool[["problem_key", "pool_id", "tier", "source", "answer"]], on="problem_key", validate="one_to_one")
    out["answer"] = out["answer"].astype(int)
    # The olympiad pool reuses 34 AMC `pool_id`s for different problems (Finding 29), and the sampler drops every
    # row with a repeated id, so the id here comes from the problem itself.
    out["pool_id"] = "e62:" + out["problem_key"]
    out = out.sort_values("problem_key").reset_index(drop=True)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(OUT, index=False)
    print(f"{len(out)} problems -> {OUT}\n{out['dataset_id'].value_counts().to_string()}")


if __name__ == "__main__":
    main()
