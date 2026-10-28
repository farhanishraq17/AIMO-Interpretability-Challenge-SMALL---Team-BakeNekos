"""E66: the problems whose E62 difficulty target is cut off by the 32k cap (`protocols/E66_PROTOCOL.md` §1).

E64's target is Qwen's mean log trace length at a 32,768-token cap; a problem with any sample at the cap has a censored
target. This keeps the rows of E62's pool whose downloaded base record has at least one sample at the cap, in
`problem_key` order, the order the 16 E66 slices follow. Rows are E62's unchanged (same text, same `pool_id`).

Run:  python datagen/perturb/e66_pool.py
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
BASE = "runs/e62-qwen-?-*/e62-qwen-?/base/shard_*/records.jsonl"
OUT = ROOT / "data" / "e66" / "pool.parquet"


def main() -> None:
    records = [json.loads(line) for path in ROOT.glob(BASE) for line in path.read_text(encoding="utf-8").splitlines()
               if line.strip()]
    assert len({r["pool_id"] for r in records}) == len(records) == 497, len(records)
    capped = {r["pool_id"] for r in records if r["n_hit_length"] >= 1}
    pool = pd.read_parquet(ROOT / "data" / "e62" / "pool.parquet")
    out = pool[pool["pool_id"].isin(capped)].sort_values("problem_key").reset_index(drop=True)
    assert len(out) == len(capped) == 194, (len(out), len(capped))
    OUT.parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(OUT, index=False)
    print(f"{len(out)} of {len(pool)} problems have a sample at the 32k cap -> {OUT.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
