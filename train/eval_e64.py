"""E64: predicted difficulty, read from one forward pass, as the per-problem robustness score.

Finding 76's strongest exploratory signal on the official rows was difficulty (shorter traces, robust), not
familiarity, which reverses there. Traces cannot be generated inside the budget, so under `protocols/E64_PROTOCOL.md`
this learns difficulty from the prompt instead: for every model, a per-layer ridge from its own E55 prompt states
(`X_mean`) to Qwen's mean log trace length on E62's 497 problems, averaged over layers. No robustness label enters the
fit. Then:

* D1 (gate): 5-fold CV Spearman of predicted against observed difficulty, every model >= 0.30;
* D2 (gate): AUC of -(predicted difficulty) on the 34 official rows >= 0.60, template-bootstrap interval;
* D3: Spearman of predicted difficulty with mean NLL (is it familiarity in disguise?);
* cutoffs: each model's p_m quantile of predicted difficulty over the 36 `train-main-v2` problems (E63's calibration).

Writes runs/e64/artifact.npz (the arrays the ZIP ships), runs/e64/result.json and the `E64: predicted difficulty as
the per-problem score` section of RESULTS.md.

Run:  python train/eval_e64.py
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.linear_model import RidgeCV
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
for path in ("train", "scripts", "datagen"):
    sys.path.insert(0, str(ROOT / path))

from eval_e61 import E55_RECORDS, P_ROBUST, SHORT  # noqa: E402
from eval_problem_level import load_rows  # noqa: E402
from results_io import upsert_section  # noqa: E402

CAP, SEED, BOOT = 32768, 64, 2_000
ALPHAS = np.logspace(1, 5, 9)
FEATURES = "runs/e55-features-{short}-*/e55-features-{short}/stagea-{short}/features.npz"
BASE = "runs/e62-qwen-?-*/e62-qwen-?/base/shard_*/records.jsonl"
OUT = ROOT / "runs" / "e64"
HEADING = "E64: predicted difficulty as the per-problem score"


def difficulty() -> pd.Series:
    """Qwen's mean log trace length (capped) per problem key, from E62's base phase."""
    rows = [json.loads(line) for path in ROOT.glob(BASE) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    y = {r["pool_id"].removeprefix("e62:"): float(np.mean(np.log(np.minimum(r["completion_tokens"], CAP)))) for r in rows}
    assert len(y) == 497, len(y)
    return pd.Series(y)


def layer_fit(X: np.ndarray, y: np.ndarray) -> list[tuple[StandardScaler, RidgeCV]]:
    fits = []
    for layer in range(X.shape[1]):
        scaler = StandardScaler().fit(X[:, layer])
        fits.append((scaler, RidgeCV(alphas=ALPHAS).fit(scaler.transform(X[:, layer]), y)))
    return fits


def layer_predict(fits, X: np.ndarray) -> np.ndarray:
    return np.mean([ridge.predict(scaler.transform(X[:, layer])) for layer, (scaler, ridge) in enumerate(fits)], axis=0)


def template_auc(frame: pd.DataFrame, score: str) -> dict:
    y, s, groups = frame["is_robust"].astype(bool).to_numpy(), frame[score].to_numpy(), frame["template"].to_numpy(str)
    ids = np.unique(groups)
    index = {g: np.flatnonzero(groups == g) for g in ids}
    rng, values = np.random.default_rng(SEED), []
    for _ in range(BOOT):
        pick = np.concatenate([index[g] for g in rng.choice(ids, len(ids))])
        if np.unique(y[pick]).size == 2:
            values.append(roc_auc_score(y[pick], s[pick]))
    return {"auc": float(roc_auc_score(y, s)), "ci": np.percentile(values, [2.5, 97.5]).tolist(), "n": int(len(y))}


def official_keys() -> list[str]:
    return sorted(set(pd.read_parquet(ROOT / "data" / "e55" / "instances.parquet").query("dataset_id == 'train-main-v2'")["problem_key"]))


def fit_models(y_all: pd.Series):
    """Per model: D1 (out-of-fold Spearman), D3, the p_m cutoff, full-fit predictions on every E55 problem, and the
    arrays the ZIP ships. E66 calls this with its own target."""
    official = official_keys()
    arrays, d1, d3, cutoffs, predicted = {}, {}, {}, {}, {}
    for model, short in SHORT.items():
        bundle = np.load(next(ROOT.glob(FEATURES.format(short=short))), allow_pickle=True)
        keys = [str(k) for k in bundle["problem_keys"]]
        X = bundle["X_mean"].astype(np.float32)
        train = [i for i, k in enumerate(keys) if k in y_all.index]
        assert len(train) == len(y_all), f"{short}: {len(train)} of {len(y_all)} training problems"
        Xt, yt = X[train], y_all.loc[[keys[i] for i in train]].to_numpy()

        oof = np.zeros(len(yt))
        for fit_idx, test_idx in KFold(5, shuffle=True, random_state=SEED).split(Xt):
            oof[test_idx] = layer_predict(layer_fit(Xt[fit_idx], yt[fit_idx]), Xt[test_idx])
        d1[model] = float(spearmanr(oof, yt).statistic)

        fits = layer_fit(Xt, yt)
        pred = pd.Series(layer_predict(fits, X), index=keys)
        predicted[model] = pred
        nll = {r["problem_key"]: r["mean_nll"] for p in ROOT.glob(E55_RECORDS.format(short=short))
               for r in map(json.loads, p.read_text(encoding="utf-8").splitlines()) if r}
        d3[model] = float(spearmanr(pred.loc[list(nll)], list(nll.values())).statistic)
        cutoffs[model] = float(np.quantile(pred.loc[official], P_ROBUST[model]))
        arrays.update({f"{short}__mean": np.stack([s.mean_ for s, _ in fits]).astype(np.float32),
                       f"{short}__scale": np.stack([s.scale_ for s, _ in fits]).astype(np.float32),
                       f"{short}__coef": np.stack([r.coef_ for _, r in fits]).astype(np.float32),
                       f"{short}__intercept": np.array([r.intercept_ for _, r in fits], dtype=np.float32),
                       f"{short}__cutoff": np.float32(cutoffs[model])})
        print(f"{short}: D1 {d1[model]:.3f}, D3 {d3[model]:.3f}, cutoff {cutoffs[model]:.4f}", flush=True)
    return arrays, d1, d3, cutoffs, predicted


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--no-write", action="store_true", help="print only; do not touch RESULTS.md")
    args = parser.parse_args()

    official = official_keys()
    rows = load_rows()
    rows = rows[rows["model_id"].isin(SHORT)].copy()
    arrays, d1, d3, cutoffs, predicted = fit_models(difficulty())

    rows["neg_difficulty"] = [-predicted[m].loc[k] for m, k in zip(rows["model_id"], rows["problem_key"])]
    d2 = template_auc(rows, "neg_difficulty")
    gates = {"D1": all(v >= 0.30 for v in d1.values()), "D2": d2["auc"] >= 0.60}
    # Amendment 2 (reported only, decides nothing): the shipped decisions on the same 34 rows, against the constants.
    rows["e64"] = [bool(predicted[m].loc[k] <= cutoffs[m]) for m, k in zip(rows["model_id"], rows["problem_key"])]
    rows["constants"] = rows["model_id"].isin(["Qwen/Qwen3.5-4B", "deepseek-ai/DeepSeek-R1-0528-Qwen3-8B"])
    truth = rows["is_robust"].astype(bool)
    decisions = {k: {"accuracy": float((rows[k] == truth).mean()),
                     "balanced_accuracy": float(np.mean([(rows[k][truth] == True).mean(), (rows[k][~truth] == False).mean()]))}  # noqa: E712
                 for k in ("e64", "constants")}
    calls = {m: float(np.mean(predicted[m].loc[official] <= cutoffs[m])) for m in SHORT}

    lines = ["Fixed in `protocols/E64_PROTOCOL.md` before any fit. Target: Qwen's mean log trace length (cap 32k) on E62's 497 "
             "problems. Score: per-layer ridge from each model's own E55 prompt states, averaged over layers; lower = robust.", "",
             "| model | D1 CV Spearman (gate >= 0.30) | D3 Spearman with mean NLL | cutoff (p_m quantile, official 36) | share of the 36 called robust |",
             "| --- | --- | --- | --- | --- |"]
    lines += [f"| `{m}` | {d1[m]:.3f} | {d3[m]:.3f} | {cutoffs[m]:.4f} | {calls[m]:.2f} |" for m in SHORT]
    lines += ["", f"**D2 (gate: AUC >= 0.60), official rows:** AUC {d2['auc']:.3f} [{d2['ci'][0]:.3f}, {d2['ci'][1]:.3f}] on {d2['n']} rows.",
              "", f"Reported only (amendment 2): the shipped decisions on the same 34 rows score accuracy "
              f"{decisions['e64']['accuracy']:.3f} (balanced {decisions['e64']['balanced_accuracy']:.3f}); the per-model constants "
              f"{decisions['constants']['accuracy']:.3f} (balanced {decisions['constants']['balanced_accuracy']:.3f}).",
              "", f"**Gates: D1 {'pass' if gates['D1'] else 'fail'}, D2 {'pass' if gates['D2'] else 'fail'}.** D4 (DeepSeek MATH) is not "
              "computable: E55 holds no prompt states for those problems (amendment 1)."]
    body = "\n".join(lines)
    print(body)
    OUT.mkdir(parents=True, exist_ok=True)
    np.savez(OUT / "artifact.npz", **arrays)
    (OUT / "result.json").write_text(json.dumps({"d1": d1, "d2": d2, "d3": d3, "cutoffs": cutoffs, "gates": gates,
                                                 "official_called_robust": calls, "official_decisions": decisions}, indent=2),
                                     encoding="utf-8")
    pd.DataFrame(predicted).to_csv(OUT / "predicted_difficulty.csv")
    if not args.no_write:
        upsert_section(ROOT / "RESULTS.md", HEADING, body)
        print(f"\nwrote the {HEADING} section of RESULTS.md")


if __name__ == "__main__":
    main()
