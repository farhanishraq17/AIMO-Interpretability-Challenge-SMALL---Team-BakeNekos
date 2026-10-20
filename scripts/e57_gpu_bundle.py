"""E57 GPU check: the per-problem submission template on the four real checkpoints, before any probe exists.

`submission/test_e57_parity.py` proved the template computes E55's features on the tiny test model.
This repeats the check on the real models, on Kaggle T4s like the ones E55 ran on:

1. each checkpoint loads through the template's own loader (tokenizer guard included);
2. its mean-pooled states for two problems match E55's stored features (three layers each, shipped as a
   small reference file because a kernel script cannot carry the full arrays);
3. twenty one-problem `are_robust` calls with a random probe return native bools without falling back,
   and their wall time is recorded with the model cached, as the per-problem evaluator calls it;
4. an unknown model id returns the constants.

The random probe exercises the arithmetic only; its predictions mean nothing.

Run via:  python scripts/kaggle_followup.py push e57-gpu-check --account <ALIAS>
"""

import io
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "datagen"))
sys.path.insert(0, str(ROOT / "scripts"))
from e52_gpu_bundle import HEADER, MODELS, payload  # noqa: E402
from extract_features import DEFAULT_ARTIFACT, load_system_prompt  # noqa: E402
from kaggle_launch import PIP_PINS  # noqa: E402

FILES = ("submission/e57-probe/solution.py", "datagen/tokenizer_check.py")
FEATURES = "runs/e55-features-{short}-*/e55-features-{short}/stagea-{short}/features.npz"
JOBS = ["gpu-check"]


def references() -> tuple[bytes, dict]:
    """Two problems per model: three layers of E55's stored mean states, and the problem texts."""
    instances = pd.read_parquet(ROOT / "data" / "e55" / "instances.parquet").drop_duplicates("problem_key")
    text = dict(zip(instances["problem_key"], instances["problem"]))
    arrays, problems = {}, {}
    for short in MODELS:
        bundle = np.load(next(ROOT.glob(FEATURES.format(short=short))), allow_pickle=True)
        keys = [str(k) for k in bundle["problem_keys"][:2]]
        n_layers = bundle["X_mean"].shape[1]
        layers = [0, n_layers // 2, n_layers - 1]
        arrays[f"{short}__layers"] = np.array(layers)
        arrays[f"{short}__states"] = bundle["X_mean"][:2][:, layers]
        problems[short] = [text[k] for k in keys]
    buffer = io.BytesIO()
    np.savez_compressed(buffer, **arrays)
    timing = instances.sort_values("problem_key")["problem"].head(20).tolist()
    return buffer.getvalue(), {"parity": problems, "timing": timing}


def render(job: str) -> tuple[str, dict]:
    assert job == "gpu-check"
    reference, problems = references()
    # The ZIP layout: tokenizer_check.py and system_prompt.txt sit next to solution.py, which imports from there.
    extra = {"submission/e57-probe/system_prompt.txt": load_system_prompt(DEFAULT_ARTIFACT).encode("utf-8"),
             "submission/e57-probe/tokenizer_check.py": (ROOT / "datagen" / "tokenizer_check.py").read_bytes(),
             "reference.npz": reference, "problems.json": json.dumps(problems).encode("utf-8")}
    entries, hashes = payload(FILES, extra)
    source = HEADER.format(out="e57-gpu-check", entries=json.dumps(entries)) + f'''subprocess.run([sys.executable, "-m", "pip", "install", "--no-cache-dir", "-q", *{PIP_PINS!r}], check=True)
import numpy as np
sys.path.insert(0, str(root / "submission" / "e57-probe"))
import solution
here = Path(solution.__file__).resolve().parent
reference = np.load(root / "reference.npz")
problems = json.loads((root / "problems.json").read_text())
system_prompt = (here / "system_prompt.txt").read_text(encoding="utf-8")
result, probe = {{"models": {{}}}}, {{}}
rng = np.random.default_rng(57)
from huggingface_hub import snapshot_download
for model_id, short in solution.SHORT.items():
    entry = {{}}
    # The evaluator serves weights from a warm, offline cache; download first, then load offline as it does.
    began = time.time()
    snapshot_download(model_id)
    entry["download_seconds"] = round(time.time() - began, 1)
    began = time.time()
    tokenizer, model = solution._load(model_id)
    entry["load_seconds"] = round(time.time() - began, 1)
    layers = reference[f"{{short}}__layers"]
    diffs = []
    for i, text in enumerate(problems["parity"][short]):
        ours = solution.mean_states(tokenizer, model, text, system_prompt)
        diffs.append(float(np.abs(ours[layers] - reference[f"{{short}}__states"][i].astype(np.float32)).max()))
    entry["parity_max_abs_diff"] = diffs
    n_layers, hidden = ours.shape
    probe.update({{f"{{short}}__mean": np.zeros((n_layers, hidden), np.float32), f"{{short}}__scale": np.ones((n_layers, hidden), np.float32),
                  f"{{short}}__coef": (rng.standard_normal((n_layers, hidden)) * 0.01).astype(np.float32),
                  f"{{short}}__intercept": np.zeros(n_layers, np.float32), f"{{short}}__threshold": np.float32(0.5)}})
    np.savez(here / "e57_probe.npz", **probe)
    calls, outputs = [], []
    for text in problems["timing"]:
        began = time.time()
        outputs += solution.are_robust(model_id, [text])
        calls.append(time.time() - began)
    entry.update(seconds_per_call=float(np.mean(calls)), max_call_seconds=float(np.max(calls)),
                 native_bools=all(type(o) is bool for o in outputs), n_calls=len(calls), robust_calls=int(sum(outputs)))
    result["models"][model_id] = entry
    print(model_id, entry, flush=True)
    # Free the GPU, then the disk: four checkpoints do not fit together. This harness's own references
    # must go too, or the template's eviction cannot release the model (version 2 ran out of memory).
    del tokenizer, model, ours
    solution._cache.clear()
    import gc, torch
    gc.collect()
    torch.cuda.empty_cache()
    shutil.rmtree(Path(os.environ["HF_HOME"]) / "hub", ignore_errors=True)
result["unknown_model"] = solution.are_robust("example/unknown-model", ["1 + 1 = ?"])
result["end_to_end_seconds"] = time.time() - started
(out / "result.json").write_text(json.dumps(result, indent=2))
print(json.dumps(result, indent=2))
'''
    compile(source, "run.py", "exec")
    return source, hashes
