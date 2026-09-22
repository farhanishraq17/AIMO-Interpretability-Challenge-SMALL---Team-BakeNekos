"""GPU check of the packaged E27d ZIP: every model loads, nothing falls back, and time is spent.

E11's first GPU check caught a solution that silently fell back to text-only (Finding 20); a
fallback here would look like a score of exactly v2's 0.64. This unpacks the ZIP exactly as
Codabench would, downloads each model into the cache (the worker has them pre-cached), then
calls the solution's `problem_nll` directly, so any failure raises instead of falling back, and
`are_robust` for the predictions. Problems are the official ones each model is labelled on, so
the NLLs can be compared with E23's records.

Run (on Kaggle, via kaggle_launch):  python submission/check_e27d.py --models Qwen/Qwen3.5-4B ...
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import zipfile
from pathlib import Path

import pandas as pd
from huggingface_hub import snapshot_download

ROOT = Path(__file__).resolve().parents[1]
ZIP = ROOT / "dist" / "solution-e27d-familiarity-small.zip"
OFFICIAL = ROOT / "data" / "official_check"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--models", nargs="+", required=True)
    parser.add_argument("--out", type=Path, default=Path("/kaggle/working/e27d_check.json"))
    args = parser.parse_args()

    unpacked = Path("/tmp/e27d")
    zipfile.ZipFile(ZIP).extractall(unpacked)
    sys.path.insert(0, str(unpacked))
    import solution  # noqa: E402  (the packaged file, not the repo copy)

    pool = pd.read_parquet(OFFICIAL / "pool.parquet").set_index("problem_key")["problem"]
    labels = pd.read_parquet(OFFICIAL / "official_labels.parquet")
    report = {}
    for model_id in args.models:
        snapshot_download(model_id)
        rows = labels[labels["model_id"] == model_id]
        problems = [pool[k] for k in rows["problem_key"]]
        started = time.time()
        nll = solution.problem_nll(model_id, problems)  # raises on any failure: no silent fallback
        predictions = solution.are_robust(model_id, problems)
        report[model_id] = {
            "seconds": time.time() - started,
            "problem_keys": list(rows["problem_key"]),
            "official": [bool(v) for v in rows["is_robust"]],
            "nll": nll,
            "predictions": predictions,
            "native_bools": all(type(p) is bool for p in predictions),
        }
        hits = sum(p == o for p, o in zip(predictions, report[model_id]["official"]))
        print(f"{model_id}: {len(problems)} problems in {report[model_id]['seconds']:.1f}s, "
              f"{hits}/{len(problems)} match official labels", flush=True)
    args.out.write_text(json.dumps(report, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
