"""E62: the problems for the familiarity-versus-detection report experiment (`protocols/E62_PROTOCOL.md`).

E55 scored 548 problems with every model, so their familiarity is already known; E62 needs each one's reference
answer to write contradictions against. This keeps the E55 problems that have an answer in the olympiad pool and
are not official `train-main-v2` rows, takes E55's text (the text the NLL was computed on), and sorts by
`problem_key`, the order generation and sampling follow.

`--eligible` (amendment 2) instead writes `eligible.parquet`: the pool's problems that the downloaded base records
call eligible (both samples correct within the cap), the only ones whose contradictions are ever sampled.

Run:  python datagen/perturb/e62_pool.py [--eligible]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "data" / "e62" / "pool.parquet"
BASE = "runs/e62-qwen-?-*/e62-qwen-?/base/shard_*/records.jsonl"


def eligible() -> None:
    sys.path.insert(0, str(ROOT / "train"))
    from eval_e62 import CAP
    from eval_e58 import correct

    records = {r["pool_id"]: r for p in ROOT.glob(BASE) for r in map(json.loads, p.read_text(encoding="utf-8").splitlines()) if r}
    keep = {i for i, r in records.items() if r["n_samples"] == 2 and all(correct(r, CAP))}
    pool = pd.read_parquet(OUT)
    out = pool[pool["pool_id"].isin(keep)].reset_index(drop=True)
    out.to_parquet(OUT.with_name("eligible.parquet"), index=False)
    print(f"{len(records)} of {len(pool)} problems have base records; {len(out)} eligible -> {OUT.with_name('eligible.parquet')}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--eligible", action="store_true")
    if parser.parse_args().eligible:
        return eligible()
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
