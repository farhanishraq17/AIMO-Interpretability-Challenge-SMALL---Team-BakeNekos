"""Build the E64 ZIP: predicted difficulty from one forward pass (`protocols/E64_PROTOCOL.md`).

`train/eval_e64.py` fits the per-layer ridges and cutoffs and writes runs/e64/artifact.npz; this ships those arrays as
plain NumPy (no pickled sklearn object to break across library versions) with the E64 solution, the tokenizer guard,
the baseline system prompt and `small.txt`. Before writing, it checks that the shipped arithmetic, applied to E55's
stored prompt states, reproduces the offline predicted difficulty for every problem of every model.

`--experiment e66` ships E66's refit (`protocols/E66_PROTOCOL.md`) instead: same solution, arrays from runs/e66.

Run:  python submission/make_e64.py [--experiment e66]
"""

from __future__ import annotations

import argparse
import io
import sys
from pathlib import Path

import numpy as np
import pandas as pd

SUBMISSION = Path(__file__).resolve().parent
ROOT = SUBMISSION.parent
for path in (ROOT / "getting-started" / "scripts", ROOT / "datagen", SUBMISSION / "e64-difficulty"):
    sys.path.insert(0, str(path))

from build import archive  # noqa: E402
from extract_features import DEFAULT_ARTIFACT, load_system_prompt  # noqa: E402

import solution  # noqa: E402  (the shipped arithmetic)

FEATURES = "runs/e55-features-{short}-*/e55-features-{short}/stagea-{short}/features.npz"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--experiment", default="e64", choices=["e64", "e66"], help="which fit under runs/ to ship")
    experiment = parser.parse_args().experiment
    run = ROOT / "runs" / experiment
    arrays = dict(np.load(run / "artifact.npz"))
    offline = pd.read_csv(run / "predicted_difficulty.csv", index_col=0)
    for model, short in solution.SHORT.items():
        bundle = np.load(next(ROOT.glob(FEATURES.format(short=short))), allow_pickle=True)
        X, keys = bundle["X_mean"].astype(np.float32), [str(k) for k in bundle["problem_keys"]]
        ours = np.array([solution.difficulty(arrays, short, X[i]) for i in range(len(keys))])
        drift = np.abs(ours - offline.loc[keys, model].to_numpy()).max()
        assert drift < 1e-3, f"{model}: shipped arithmetic drifts by {drift}"
        print(f"{model}: shipped arithmetic matches offline on {len(keys)} problems (max abs diff {drift:.2e})")
    buffer = io.BytesIO()
    np.savez_compressed(buffer, **arrays)
    out = archive(ROOT / "dist" / f"solution-{experiment}-difficulty-small.zip", {
        "solution.py": (SUBMISSION / "e64-difficulty" / "solution.py").read_bytes(),
        "tokenizer_check.py": (ROOT / "datagen" / "tokenizer_check.py").read_bytes(),
        "system_prompt.txt": load_system_prompt(DEFAULT_ARTIFACT).encode("utf-8"),
        "e64_difficulty.npz": buffer.getvalue(),
        "small.txt": b"",
    })
    print(f"wrote {out.name} ({out.stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
