"""Verify the E11 submission without the 8B model.

1. **Array inference matches sklearn.** The E1b pooled features are fed through the
   submission's array-based probe math and compared with the reference decision scores that
   `build_e11_artifact.py` computed through the fitted sklearn pipelines.
2. **Contract.** Both harness call styles return native bools of the right length; any
   model other than DeepSeek gets all False.
3. **Fallback.** If the language model cannot load (it is not cached on this machine), the
   DeepSeek path still returns valid text-only decisions.

Run:  python train/check_e11_artifact.py
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SOLUTION = ROOT / "submission" / "e11-text-stagea"
FEATURES = ROOT / "runs" / "e1b-stagea-deepseek" / "features" / "stagea-deepseek"
sys.path.insert(0, str(SOLUTION))

import e11_inference as inference  # noqa: E402
from solution import are_robust  # noqa: E402


def main() -> None:
    artifact = inference.load_artifact()
    reference = artifact["reference"]
    payload = np.load(FEATURES / "features.npz")
    keys = [str(k) for k in payload["problem_keys"]]
    if keys != list(reference["problem_keys"]):
        raise SystemExit("feature order differs from the artifact's reference order")

    stagea = np.array([
        inference.stagea_probability(
            {p: payload[f"X_{p}"][i].astype(np.float32) for p in artifact["poolings"]}, artifact
        )
        for i in range(len(keys))
    ])
    scores = inference.fused_decisions(reference["text_prob"], stagea, artifact)
    stagea_gap = float(np.abs(stagea - reference["stagea_prob"]).max())
    score_gap = float(np.abs(scores - reference["fused"]).max())
    flips = int(((scores >= 0) != (reference["fused"] >= 0)).sum())
    print(f"1. stagea prob max |array - sklearn| = {stagea_gap:.2e}; fused score gap = {score_gap:.2e}; decision flips = {flips}")
    assert stagea_gap < 1e-4 and flips == 0, "array inference disagrees with the sklearn reference"
    print(f"   in-sample robust decisions: {int((scores >= 0).sum())} of {len(scores)}")

    problems = ["Find the remainder when $2^{100}$ is divided by 7.", "Compute $1+2+\\cdots+10$."]
    for model_id in ("Qwen/Qwen3.5-4B", "Skywork/Skywork-OR1-Math-7B", "allenai/Olmo-3-7B-Think", "unknown/model"):
        out = are_robust(model_id, problems)
        assert out == [False, False] and all(type(v) is bool for v in out), (model_id, out)
    new_style = are_robust("Qwen/Qwen3.5-4B", reasoning_effort="low", problems=problems)
    assert new_style == [False, False]
    print("2. non-DeepSeek models -> all False under both call styles")

    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    fallback = are_robust("deepseek-ai/DeepSeek-R1-0528-Qwen3-8B", reasoning_effort="default", problems=problems)
    assert len(fallback) == 2 and all(type(v) is bool for v in fallback), fallback
    legacy = are_robust("qwen3-8b:low", problems)
    assert len(legacy) == 2 and all(type(v) is bool for v in legacy), legacy
    print(f"3. DeepSeek without a cached model -> valid text-only fallback {fallback}; alias qwen3-8b:low {legacy}")
    print("all checks passed")


if __name__ == "__main__":
    main()
