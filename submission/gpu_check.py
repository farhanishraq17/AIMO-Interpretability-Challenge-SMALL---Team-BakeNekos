"""GPU check for the packaged E11 submission before it spends a Codabench slot.

Local checks cannot load the 8B model, so the DeepSeek path of the ZIP has never run. On a
Kaggle GPU this script:

1. unpacks the exact ZIP (or finds its already-extracted files under the dataset mount);
2. calls `are_robust` on the 138 DeepSeek training problems, timing the full call including
   model load, and compares its decisions with the artifact's reference decisions (a silent
   fallback to the text block would show up here as disagreements);
3. recomputes each problem's stagea probability through the same inference functions and
   compares it with the reference computed from the E1b features at training time;
4. times a call for a model the bundle does not cover.

Results go to /kaggle/working/e11_gpu_check.json.

Run:  python submission/gpu_check.py --search-root /kaggle/input
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import zipfile
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
INSTANCES = ROOT / "data" / "small_track" / "instances_train.parquet"
ZIP_NAME = "solution-e11-text-stagea-small.zip"
MODEL_ID = "deepseek-ai/DeepSeek-R1-0528-Qwen3-8B"


def locate_submission(search_root: Path, workdir: Path) -> Path:
    archives = sorted(search_root.rglob(ZIP_NAME))
    if archives:
        with zipfile.ZipFile(archives[0]) as bundle:
            bundle.extractall(workdir)
        print(f"extracted {archives[0]} -> {workdir}")
        return workdir
    # Kaggle may unpack uploaded zips into the dataset mount.
    for candidate in sorted(search_root.rglob("e11_inference.py")):
        if (candidate.parent / "solution.py").is_file():
            print(f"using extracted submission at {candidate.parent}")
            return candidate.parent
    raise SystemExit(f"no {ZIP_NAME} or extracted submission under {search_root}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--search-root", type=Path, default=Path("/kaggle/input"))
    parser.add_argument("--workdir", type=Path, default=Path("/tmp/e11-submission"))
    parser.add_argument("--out", type=Path, default=Path("/kaggle/working/e11_gpu_check.json"))
    args = parser.parse_args()

    submission = locate_submission(args.search_root, args.workdir)
    sys.path.insert(0, str(submission))
    import numpy as np
    import e11_inference as inference
    from solution import are_robust

    artifact = inference.load_artifact()
    reference = artifact["reference"]
    order = {key: i for i, key in enumerate(reference["problem_keys"])}
    table = pd.read_parquet(INSTANCES)
    table = table[table["model_id"] == MODEL_ID]
    table = table.assign(position=table["problem_key"].map(order)).sort_values("position")
    problems = table["problem"].tolist()
    expected = reference["fused"] >= 0

    started = time.time()
    decisions = are_robust(MODEL_ID, reasoning_effort="default", problems=problems)
    public_seconds = time.time() - started
    valid = type(decisions) is list and len(decisions) == len(problems) and all(type(d) is bool for d in decisions)
    flips = int(np.sum(np.asarray(decisions) != expected)) if valid else None
    print(f"are_robust: {len(problems)} problems in {public_seconds:.1f}s, valid={valid}, "
          f"robust={sum(decisions) if valid else None}, flips vs reference={flips}")

    started = time.time()
    tokenizer, model = inference.load_language_model(MODEL_ID)
    load_seconds = time.time() - started
    started = time.time()
    stagea = np.array([
        inference.stagea_probability(inference.pooled_prompt_states(tokenizer, model, p, artifact), artifact)
        for p in problems
    ])
    forward_seconds = time.time() - started
    gap = np.abs(stagea - reference["stagea_prob"])
    recomputed = inference.fused_decisions(reference["text_prob"], stagea, artifact) >= 0
    print(f"stagea prob gap: max={gap.max():.4f} mean={gap.mean():.4f}; "
          f"decision flips from recomputed features={int((recomputed != expected).sum())}; "
          f"load {load_seconds:.1f}s, forward {forward_seconds:.1f}s")

    started = time.time()
    other = are_robust("Qwen/Qwen3.5-4B", reasoning_effort="default", problems=problems)
    other_seconds = time.time() - started

    import torch
    import transformers

    result = {
        "valid": valid,
        "n_problems": len(problems),
        "public_call_seconds": round(public_seconds, 1),
        "robust_predicted": int(sum(decisions)) if valid else None,
        "robust_reference": int(expected.sum()),
        "flips_vs_reference": flips,
        "stagea_gap_max": float(gap.max()),
        "stagea_gap_mean": float(gap.mean()),
        "flips_from_recomputed_features": int((recomputed != expected).sum()),
        "model_load_seconds": round(load_seconds, 1),
        "forward_seconds": round(forward_seconds, 1),
        "seconds_per_problem": round(forward_seconds / len(problems), 3),
        "other_model_all_false": other == [False] * len(problems),
        "other_model_seconds": round(other_seconds, 2),
        "dtype": str(model.dtype),
        "gpus": [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())],
        "torch": torch.__version__,
        "transformers": transformers.__version__,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))
    if not valid:
        raise SystemExit("are_robust returned invalid predictions")


if __name__ == "__main__":
    main()
