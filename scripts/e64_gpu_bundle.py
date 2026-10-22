"""E64 GPU check (protocol checks C1 and C5), split so the 5 MB arrays never ride inside a kernel script.

On Kaggle T4s, the E64 solution's own `_load` and `mean_states` run on the four real checkpoints:

1. the prompt states of five problems per model are saved (runs/e64-gpu-check-*/e64-gpu-check/states_<short>.npz);
   `python scripts/e64_gpu_bundle.py --verify` then applies the shipped `difficulty` arithmetic to them locally and
   compares with the offline predictions (C1: max abs diff <= 1e-3);
2. fifteen one-problem forward passes per model, model load included and download excluded, are timed (C5: sixty
   calls under five minutes; the ridge read-out adds microseconds and is timed locally).

Run via:  python scripts/kaggle_followup.py push e64-gpu-check --account <ALIAS>
          python scripts/e64_gpu_bundle.py --verify      (after `output e64-gpu-check`)
"""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "datagen"))
sys.path.insert(0, str(ROOT / "scripts"))
from e52_gpu_bundle import HEADER, MODELS, payload  # noqa: E402
from extract_features import DEFAULT_ARTIFACT, load_system_prompt  # noqa: E402
from kaggle_launch import PIP_PINS  # noqa: E402

JOBS = ["gpu-check"]
N_PARITY, N_TIMING = 5, 15


def problems() -> dict:
    text = pd.read_parquet(ROOT / "data" / "e55" / "instances.parquet").drop_duplicates("problem_key").sort_values("problem_key")
    keys, texts = text["problem_key"].tolist(), text["problem"].tolist()
    return {"parity_keys": keys[:N_PARITY], "parity": texts[:N_PARITY], "timing": texts[N_PARITY:N_PARITY + N_TIMING]}


def render(job: str) -> tuple[str, dict]:
    assert job == "gpu-check"
    extra = {"e64/solution.py": (ROOT / "submission" / "e64-difficulty" / "solution.py").read_bytes(),
             "e64/tokenizer_check.py": (ROOT / "datagen" / "tokenizer_check.py").read_bytes(),
             "e64/system_prompt.txt": load_system_prompt(DEFAULT_ARTIFACT).encode("utf-8"),
             "problems.json": json.dumps(problems()).encode("utf-8")}
    entries, hashes = payload((), extra)
    source = HEADER.format(out="e64-gpu-check", entries=json.dumps(entries)) + f'''subprocess.run([sys.executable, "-m", "pip", "install", "--no-cache-dir", "-q", *{PIP_PINS!r}], check=True)
import numpy as np
sys.path.insert(0, str(root / "e64"))
import solution
problems = json.loads((root / "problems.json").read_text())
system_prompt = (root / "e64" / "system_prompt.txt").read_text(encoding="utf-8")
result = {{"models": {{}}}}
from huggingface_hub import snapshot_download
for short, model_id in {MODELS!r}.items():
    entry = {{}}
    began = time.time()
    snapshot_download(model_id)
    entry["download_seconds"] = round(time.time() - began, 1)
    began = time.time()
    tokenizer, model = solution._load(model_id)
    for text in problems["timing"]:
        solution.mean_states(tokenizer, model, text, system_prompt)
    entry["c5_seconds_load_and_15_calls"] = round(time.time() - began, 2)
    states = np.stack([solution.mean_states(tokenizer, model, text, system_prompt) for text in problems["parity"]])
    np.savez_compressed(out / f"states_{{short}}.npz", states=states.astype(np.float16), keys=np.array(problems["parity_keys"]))
    entry.update(dtype=str(model.dtype), state_shape=list(states.shape))
    result["models"][model_id] = entry
    print(model_id, entry, flush=True)
    del tokenizer, model
    solution._cache.clear()
    import gc, torch
    gc.collect()
    torch.cuda.empty_cache()
    shutil.rmtree(Path(os.environ["HF_HOME"]) / "hub", ignore_errors=True)
result["unknown_model"] = solution.are_robust("example/unknown-model", ["1 + 1 = ?"])
total = sum(e["c5_seconds_load_and_15_calls"] for e in result["models"].values())
result.update(c5_total_seconds=total, c5_pass=total < 300, end_to_end_seconds=time.time() - started)
(out / "result.json").write_text(json.dumps(result, indent=2))
print(json.dumps(result, indent=2))
'''
    compile(source, "run.py", "exec")
    return source, hashes


def verify() -> None:
    """C1: the shipped arithmetic on the GPU-computed states against the offline predictions."""
    sys.path.insert(0, str(ROOT / "submission" / "e64-difficulty"))
    import solution
    arrays = dict(np.load(ROOT / "runs" / "e64" / "artifact.npz"))
    offline = pd.read_csv(ROOT / "runs" / "e64" / "predicted_difficulty.csv", index_col=0)
    run = next(ROOT.glob("runs/e64-gpu-check-*/e64-gpu-check"))
    gpu = json.loads((run / "result.json").read_text())
    worst = 0.0
    for short, model_id in MODELS.items():
        z = np.load(run / f"states_{short}.npz")
        started = time.perf_counter()
        ours = [solution.difficulty(arrays, short, s.astype(np.float32)) for s in z["states"]]
        readout = (time.perf_counter() - started) / len(ours)
        diff = float(np.max(np.abs(np.array(ours) - offline.loc[[str(k) for k in z["keys"]], model_id].to_numpy())))
        worst = max(worst, diff)
        print(f"{model_id}: max abs diff {diff:.2e}, read-out {readout * 1e3:.2f} ms/call, "
              f"load + 15 calls {gpu['models'][model_id]['c5_seconds_load_and_15_calls']} s, {gpu['models'][model_id]['dtype']}")
    print(f"C1 {'pass' if worst <= 1e-3 else 'FAIL'} (max {worst:.2e}); C5 {'pass' if gpu['c5_pass'] else 'FAIL'} "
          f"({gpu['c5_total_seconds']:.0f} s for 60 calls); unknown model -> {gpu['unknown_model']}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--verify", action="store_true")
    if parser.parse_args().verify:
        verify()
