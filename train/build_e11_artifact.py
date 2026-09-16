"""E11: fit the text + prompt-hidden-state fusion that ships in the first Lane B submission.

Findings 11 and 16: under nested CV, a mean fusion of the char tf-idf text model and the
all-layer prompt-state ensemble reached 0.7173 balanced accuracy and 0.7230 AUC on
DeepSeek-R1-0528-Qwen3-8B (138 instances). This script refits that recipe on all 138
instances for deployment:

* **text block:** char tf-idf logistic regression, fit on everything;
* **stagea block:** one standardised logistic probe per (pooling, layer), 3 x 37 = 111, fit on
  everything; the block score is the mean of their probabilities;
* **fusion:** each block's logit is standardised with the mean and std of its 5-fold
  out-of-fold logits; the two are averaged and compared with a threshold learned on the
  out-of-fold fused scores. A text-only threshold is learned the same way, for the fallback
  path used when the language model cannot run.

Probes are exported as float32 arrays rather than sklearn objects, so inference is a few
matrix products. The text model stays a sklearn pipeline, fit here in the pinned venv
(scikit-learn 1.8.0, as at evaluation).

The artifact also stores reference in-sample decision scores computed through the sklearn
models, so `check_e11_artifact.py` can verify the array-based inference against an
independent path.

The out-of-fold metrics printed here reuse the same folds for standardisation and threshold,
so they are mildly optimistic; the honest estimate is the nested one in RESULTS.md.

Run:  python train/build_e11_artifact.py
"""

from __future__ import annotations

import argparse
import hashlib
import json
import pickle
import platform
import time
from itertools import product
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import sklearn
from joblib import Parallel, delayed

from results_io import score
from train_fusion import learn_threshold, logit
from train_stagea_probes import load, probe, text_model

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_FEATURES = ROOT / "runs" / "e1b-stagea-deepseek" / "features" / "stagea-deepseek"
DEFAULT_OUT = ROOT / "submission" / "e11-text-stagea" / "artifacts" / "e11_deepseek.joblib"
INSTANCES = ROOT / "data" / "small_track" / "instances_train.parquet"
PROBE_ARTIFACT = ROOT / "getting-started" / "solutions" / "trained-probe" / "probe_artifacts" / "probe_artifact.pkl"
MODEL_ID = "deepseek-ai/DeepSeek-R1-0528-Qwen3-8B"
PROBE_C = 0.01
MAX_PROMPT_TOKENS = 2048  # extract_features.py --max-length default


def fit_layer(X_layer: np.ndarray, y: np.ndarray, folds: np.ndarray, c: float):
    """Out-of-fold and in-sample probabilities, plus the full-data probe as arrays."""
    oof = np.full(len(y), np.nan)
    for fold in np.unique(folds):
        val = folds == fold
        oof[val] = probe(c).fit(X_layer[~val], y[~val]).predict_proba(X_layer[val])[:, 1]
    model = probe(c).fit(X_layer, y)
    scaler, classifier = model[0], model[-1]
    return (
        oof,
        model.predict_proba(X_layer)[:, 1],
        scaler.mean_.astype(np.float32),
        scaler.scale_.astype(np.float32),
        classifier.coef_[0].astype(np.float32),
        np.float32(classifier.intercept_[0]),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--features-dir", type=Path, default=DEFAULT_FEATURES)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--n-jobs", type=int, default=-1)
    args = parser.parse_args()
    started = time.time()

    X, records = load(args.features_dir, allow_label_mismatch=False)
    shard_meta = json.loads(next(args.features_dir.glob("shard_*/metadata.json")).read_text(encoding="utf-8"))
    if not shard_meta.get("tokenizer_round_trip"):
        raise SystemExit("features were not extracted with a round-trip-checked tokenizer (Finding 14)")
    if shard_meta["model_id"] != MODEL_ID:
        raise SystemExit(f"features are for {shard_meta['model_id']}, expected {MODEL_ID}")

    y = records["is_robust"].astype(bool).to_numpy()
    folds = records["fold"].astype(int).to_numpy()
    keys = records["problem_key"].astype(str).tolist()
    poolings = sorted(X)
    _, n_layers, hidden = X[poolings[0]].shape

    problems = pd.read_parquet(INSTANCES).set_index(["model_id", "problem_key"])["problem"]
    texts = [problems[(MODEL_ID, k)] for k in keys]
    with PROBE_ARTIFACT.open("rb") as handle:
        system_prompt = str(pickle.load(handle)["system_prompt"])
    if hashlib.sha1(system_prompt.encode()).hexdigest() != shard_meta["system_prompt_sha1"]:
        raise SystemExit("system prompt differs from the one the features were extracted with")

    # Stagea block: 111 probes.
    configs = list(product(poolings, range(n_layers)))
    fitted = Parallel(n_jobs=args.n_jobs)(
        delayed(fit_layer)(X[p][:, layer, :], y, folds, PROBE_C) for p, layer in configs
    )
    shape = (len(poolings), n_layers)
    probe_mean = np.stack([f[2] for f in fitted]).reshape(*shape, hidden)
    probe_scale = np.stack([f[3] for f in fitted]).reshape(*shape, hidden)
    probe_coef = np.stack([f[4] for f in fitted]).reshape(*shape, hidden)
    probe_intercept = np.array([f[5] for f in fitted], dtype=np.float32).reshape(shape)
    stagea_oof = np.mean([f[0] for f in fitted], axis=0)
    stagea_full = np.mean([f[1] for f in fitted], axis=0)

    # Text block.
    text_oof = np.full(len(y), np.nan)
    for fold in np.unique(folds):
        val = folds == fold
        model = text_model().fit([t for t, v in zip(texts, val) if not v], y[~val])
        text_oof[val] = model.predict_proba([t for t, v in zip(texts, val) if v])[:, 1]
    text_pipeline = text_model().fit(texts, y)
    text_full = text_pipeline.predict_proba(texts)[:, 1]

    # Fusion parameters from out-of-fold scores.
    logits_oof = {"text": logit(text_oof), "stagea": logit(stagea_oof)}
    mean = {k: float(v.mean()) for k, v in logits_oof.items()}
    std = {k: float(v.std() + 1e-6) for k, v in logits_oof.items()}
    z = {k: (v - mean[k]) / std[k] for k, v in logits_oof.items()}
    fused_oof = (z["text"] + z["stagea"]) / 2
    threshold = learn_threshold(y, fused_oof)
    text_threshold = learn_threshold(y, z["text"])

    metrics = {
        "fused_oof": score(y, fused_oof - threshold, threshold=0.0),
        "text_oof": score(y, z["text"] - text_threshold, threshold=0.0),
        "stagea_oof_auc": score(y, stagea_oof)["roc_auc"],
    }
    reference_fused = (
        ((logit(text_full) - mean["text"]) / std["text"] + (logit(stagea_full) - mean["stagea"]) / std["stagea"]) / 2
        - threshold
    )

    artifact = {
        "schema_version": 1,
        "name": "e11-text-stagea",
        "model_id": MODEL_ID,
        "system_prompt": system_prompt,
        "system_prompt_sha1": shard_meta["system_prompt_sha1"],
        "max_prompt_tokens": MAX_PROMPT_TOKENS,
        "poolings": poolings,
        "n_layers": int(n_layers),
        "hidden_dim": int(hidden),
        "probe_mean": probe_mean,
        "probe_scale": probe_scale,
        "probe_coef": probe_coef,
        "probe_intercept": probe_intercept,
        "text_model": text_pipeline,
        "block_logit_mean": mean,
        "block_logit_std": std,
        "threshold": float(threshold),
        "text_threshold": float(text_threshold),
        "reference": {
            "problem_keys": keys,
            "fused": reference_fused.astype(np.float64),
            "text_prob": text_full.astype(np.float64),
            "stagea_prob": stagea_full.astype(np.float64),
        },
        "training": {
            "features_dir": args.features_dir.as_posix(),
            "extraction": {k: shard_meta.get(k) for k in ("dtype", "gpu", "torch", "tokenizer_class", "tokenizer_round_trip", "n_layers")},
            "n_instances": int(len(y)),
            "n_robust": int(y.sum()),
            "probe_C": PROBE_C,
            "oof_metrics_optimistic": metrics,
            "nested_estimate": "RESULTS.md, section 'E10: fusion v1', row 'mean fusion: text + stagea'",
            "sklearn": sklearn.__version__,
            "numpy": np.__version__,
            "joblib": joblib.__version__,
            "platform": platform.platform(),
            "created": time.strftime("%Y-%m-%dT%H:%M:%S"),
        },
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(artifact, args.out, compress=3)
    print(json.dumps({"out": str(args.out), "size_mb": round(args.out.stat().st_size / 1e6, 2),
                      "threshold": threshold, "text_threshold": text_threshold,
                      "logit_mean": mean, "logit_std": std, "oof_metrics_optimistic": metrics,
                      "seconds": round(time.time() - started, 1)}, indent=2))


if __name__ == "__main__":
    main()
