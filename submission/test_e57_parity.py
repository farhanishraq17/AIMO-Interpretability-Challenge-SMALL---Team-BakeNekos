"""The E57 submission must compute exactly the features its probe was trained on.

E55's training features came from `datagen/extract_features.py`; the submission recomputes them in
`submission/e57-probe/solution.py` without that module (it imports pyarrow, which the evaluation image
does not pin). This runs both on the tiny cached test model and asserts the mean-pooled states match.

Run:  HF_HUB_OFFLINE=1 python submission/test_e57_parity.py
"""

from __future__ import annotations

import os
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
TINY = "hf-internal-testing/tiny-random-LlamaForCausalLM"
sys.path.insert(0, str(ROOT / "submission" / "e57-probe"))
sys.path.insert(0, str(ROOT / "datagen"))


def test_parity() -> None:
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    import solution
    from extract_features import DEFAULT_ARTIFACT, load_system_prompt

    rows = pd.read_parquet(ROOT / "data" / "e55" / "instances.parquet").drop_duplicates("problem_key").head(3)
    with tempfile.TemporaryDirectory() as tmp:
        table = Path(tmp) / "inst.parquet"
        rows.to_parquet(table, index=False)
        subprocess.run([sys.executable, "-W", "ignore", str(ROOT / "datagen" / "extract_features.py"), "--model-id", TINY,
                        "--instances", str(table), "--no-filter-model", "--out-dir", str(Path(tmp) / "out")],
                       check=True, capture_output=True)
        vectors = Path(tmp) / "out" / "shard_00_of_01" / "vectors"
        reference = {p.stem: np.load(p)["mean"].astype(np.float32) for p in vectors.glob("*.npz")}
    prompt = load_system_prompt(DEFAULT_ARTIFACT)
    tokenizer, model = solution._load(TINY)
    for row in rows.itertuples():
        ours = solution.mean_states(tokenizer, model, row.problem, prompt)
        assert np.array_equal(ours, reference[row.problem_key]), row.problem_key


if __name__ == "__main__":
    test_parity()
    print("ok: the submission's mean-pooled states equal extract_features.py's")
