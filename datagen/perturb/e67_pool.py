"""E67: the steering direction and the problems it is tested on (`protocols/E67_PROTOCOL.md` §1-§2).

Direction: E64's Qwen ridge at hidden-state index LAYER (index 0 is the embeddings, so this is the output of the
LAYER-th decoder layer, `layers[LAYER - 1]`), mapped back to raw state space (coef / scale) and normalised to u.
sigma is the SD over E55's 548 Qwen prompt states (`X_mean`) of their projection on u, so a push of 2 sigma moves
that layer's E64 prediction by two SDs of its spread over E55. The random control r is a Gaussian direction (seed 67), orthogonalised against u and
normalised. Only these ride inside the kernels; E64's arrays stay local (`train/eval_e67.py`).

Problems: E62's pool (no official problem) without the ones whose two E62 base samples both hit the 32k cap, ranked
by E64's Qwen predicted difficulty; the N closest to log(CAP / 2), a predicted trace of half E67's 16k cap, so a
push either way has room before the cap or the floor. Rows are E62's unchanged, in `problem_key` order.

Run:  python datagen/perturb/e67_pool.py
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
LAYER, SEED, N, CAP = 15, 67, 60, 16384
BASE = "runs/e62-qwen-?-*/e62-qwen-?/base/shard_*/records.jsonl"
FEATURES = "runs/e55-features-qwen-*/e55-features-qwen/stagea-qwen/features.npz"
OUT = ROOT / "data" / "e67"


def capped_twice() -> list[str]:
    records = [json.loads(line) for path in ROOT.glob(BASE) for line in path.read_text(encoding="utf-8").splitlines()
               if line.strip()]
    assert len(records) == 497, len(records)
    return sorted(r["pool_id"].removeprefix("e62:") for r in records if r["n_hit_length"] == r["n_samples"] == 2)


def main() -> None:
    arrays = np.load(ROOT / "runs" / "e64" / "artifact.npz")
    X = np.load(next(ROOT.glob(FEATURES)), allow_pickle=True)["X_mean"][:, LAYER].astype(np.float64)
    w = arrays["qwen__coef"][LAYER].astype(np.float64) / arrays["qwen__scale"][LAYER]
    u = w / np.linalg.norm(w)
    sigma = float((X @ u).std())
    r = np.random.default_rng(SEED).standard_normal(u.size)
    r -= (r @ u) * u
    r /= np.linalg.norm(r)
    OUT.mkdir(parents=True, exist_ok=True)
    np.savez(OUT / "directions.npz", u=u.astype(np.float32), r=r.astype(np.float32), sigma=np.float32(sigma),
             layer=np.int64(LAYER))

    pool = pd.read_parquet(ROOT / "data" / "e62" / "pool.parquet")
    capped = set(capped_twice())
    predicted = pd.read_csv(ROOT / "runs" / "e64" / "predicted_difficulty.csv", index_col=0)["Qwen/Qwen3.5-4B"]
    eligible = pool[~pool["problem_key"].isin(capped)].copy()
    eligible["predicted"] = predicted.loc[eligible["problem_key"]].to_numpy()
    eligible["distance"] = (eligible["predicted"] - np.log(CAP / 2)).abs()
    out = eligible.nsmallest(N, "distance").sort_values("problem_key").reset_index(drop=True)
    out.to_parquet(OUT / "pool.parquet", index=False)
    print(f"layer {LAYER}: sigma {sigma:.4f}, |w| {np.linalg.norm(w):.3f} (2 sigma moves the layer's prediction by "
          f"{2 * sigma * np.linalg.norm(w):.3f}); {len(capped)} capped twice dropped; {len(out)} problems, predicted "
          f"{out['predicted'].min():.2f}-{out['predicted'].max():.2f} -> {(OUT / 'pool.parquet').relative_to(ROOT)}")


if __name__ == "__main__":
    main()
