"""Build the E57 per-problem submission: probes fitted on self-labels, exported as plain arrays.

`protocols/E57_PROTOCOL.md` decides whether the probe ships; this only packages it. For each model it
fits `train_e57_probe.fit_probe` on every labelled non-official problem, chooses the threshold with
the same nested rule, and exports each layer's scaler and logistic weights as NumPy arrays, so the
ZIP loads with NumPy alone and no pickled sklearn object can break across library versions. Before
writing, it checks that the exported arithmetic reproduces sklearn's layer ensemble.

A real build requires `runs/e57/result.json` with `"ship": true` (written by the E57 evaluation once
its three rules pass). `--synthetic` builds the same ZIP from random labels for plumbing tests only;
it must never be uploaded.

Run:  python submission/make_e57.py --labels <self_labels.parquet>
      python submission/make_e57.py --synthetic
"""

from __future__ import annotations

import argparse
import io
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

SUBMISSION = Path(__file__).resolve().parent
ROOT = SUBMISSION.parent
for path in (ROOT / "getting-started" / "scripts", ROOT / "train", ROOT / "datagen", SUBMISSION / "e57-probe"):
    sys.path.insert(0, str(path))

from build import archive  # noqa: E402
from extract_features import DEFAULT_ARTIFACT, load_system_prompt  # noqa: E402
from train_e57_probe import (INSTANCES, SEED, SHORT, fit_probe, groups, load_features,  # noqa: E402
                             nested_threshold, predict_probe)

DIST = ROOT / "dist"
RESULT = ROOT / "runs" / "e57" / "result.json"


def export(models: list) -> dict[str, np.ndarray]:
    scalers = [m.named_steps["standardscaler"] for m in models]
    heads = [m.named_steps["logisticregression"] for m in models]
    return {"mean": np.stack([s.mean_ for s in scalers]).astype(np.float32),
            "scale": np.stack([s.scale_ for s in scalers]).astype(np.float32),
            "coef": np.stack([h.coef_[0] for h in heads]).astype(np.float32),
            "intercept": np.array([h.intercept_[0] for h in heads], dtype=np.float32)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--labels", type=Path)
    parser.add_argument("--synthetic", action="store_true")
    args = parser.parse_args()
    if bool(args.labels) == args.synthetic:
        parser.error("give exactly one of --labels or --synthetic")
    if args.labels and not (RESULT.is_file() and json.loads(RESULT.read_text(encoding="utf-8")).get("ship")):
        raise SystemExit(f"{RESULT} does not record a passed ship rule; E57 decides first")

    import solution  # the shipped arithmetic, checked against sklearn below

    instances = pd.read_parquet(INSTANCES)
    official = set(instances.loc[instances["dataset_id"] == "train-main-v2", "problem_key"])
    if args.synthetic:
        labels = instances[["model_id", "problem_key"]].copy()
        labels["is_robust"] = np.random.default_rng(SEED).random(len(labels)) < 0.6
    else:
        labels = pd.read_parquet(args.labels).dropna(subset=["is_robust"])
    labels = labels[~labels["problem_key"].isin(official)]
    labels["group"] = labels["problem_key"].map(groups(instances.drop_duplicates("problem_key")))

    arrays = {}
    for model, part in labels.groupby("model_id"):
        X_all, index = load_features(model)
        X = X_all[[index[k] for k in part["problem_key"]]]
        y, g = part["is_robust"].astype(int).to_numpy(), part["group"].to_numpy()
        models = fit_probe(X, y)
        threshold = nested_threshold(X, y, g)
        exported = export(models)
        short = SHORT[model]
        probe = {f"{short}__{k}": v for k, v in exported.items()}
        sample = X[:5].astype(np.float32)
        ours = [solution.probability(probe, short, row) for row in sample]
        assert np.allclose(ours, predict_probe(models, X[:5]), atol=1e-5), f"{model}: exported arithmetic drifts"
        arrays.update(probe, **{f"{short}__threshold": np.float32(threshold)})
        print(f"{model}: {len(y)} labels, threshold {threshold:.4f}, export matches sklearn")

    buffer = io.BytesIO()
    np.savez_compressed(buffer, **arrays)
    name = "solution-e57-probe-SYNTHETIC-small.zip" if args.synthetic else "solution-e57-probe-small.zip"
    out = archive(DIST / name, {
        "solution.py": (SUBMISSION / "e57-probe" / "solution.py").read_bytes(),
        "tokenizer_check.py": (ROOT / "datagen" / "tokenizer_check.py").read_bytes(),
        "system_prompt.txt": load_system_prompt(DEFAULT_ARTIFACT).encode("utf-8"),
        "e57_probe.npz": buffer.getvalue(),
        "small.txt": b"",
    })
    print(f"wrote {out.name} ({out.stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
