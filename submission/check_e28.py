"""GPU contract check for the packaged E28 familiarity solution.

The checker imports the ZIP exactly as Codabench does, loads each requested cached model,
executes the public entry point, and separately calls the score extractor so a silent fallback
cannot pass. It verifies finite scores, native booleans, and the familiar-half policy.

Run on Kaggle: python submission/check_e28.py --models Qwen/Qwen3.5-4B ...
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
import zipfile
from pathlib import Path

import pandas as pd
from huggingface_hub import snapshot_download

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ZIP = ROOT / "dist" / "solution-e28d-min-k-gap-small.zip"
OFFICIAL = ROOT / "data" / "official_check"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--models", nargs="+", required=True)
    parser.add_argument("--zip", type=Path, default=DEFAULT_ZIP)
    parser.add_argument("--out", type=Path, default=Path("/kaggle/working/e28_check.json"))
    args = parser.parse_args()

    unpacked = Path("/tmp/e28")
    zipfile.ZipFile(args.zip).extractall(unpacked)
    sys.path.insert(0, str(unpacked))
    import solution  # noqa: E402

    pool = pd.read_parquet(OFFICIAL / "pool.parquet").set_index("problem_key")["problem"]
    labels = pd.read_parquet(OFFICIAL / "official_labels.parquet")
    report = {}
    for model_id in args.models:
        snapshot_download(model_id)
        rows = labels[labels["model_id"] == model_id].drop_duplicates("problem_key")
        problems = [pool[key] for key in rows["problem_key"]]
        started = time.time()
        records = solution.problem_scores(model_id, problems)
        raw_scores = solution.combine_scores(records)
        expected = solution.familiar_half(raw_scores)
        predictions = solution.are_robust(model_id, problems)
        elapsed = time.time() - started
        assert len(records) == len(problems)
        assert all(math.isfinite(value) for record in records for value in record.values())
        assert predictions == expected
        assert all(type(prediction) is bool for prediction in predictions)
        report[model_id] = {
            "seconds_for_two_passes": elapsed,
            "n_problems": len(problems),
            "score_mode": solution.SCORE_MODE,
            "records": records,
            "predictions": predictions,
        }
        print(f"{model_id}: {len(problems)} problems, two passes in {elapsed:.1f}s, contract OK", flush=True)
    args.out.write_text(json.dumps(report, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
