"""E61 GPU check (protocol checks C1 and C5): the shipped E61b solution on the four real checkpoints.

`submission/test_e61_parity.py` proved the solution computes `extract_familiarity`'s mean NLL on the tiny
test model. This runs the solution.py inside `dist/solution-e61b-quantile-small.zip`, unchanged, on Kaggle
T4s like the ones E55 ran on:

1. C1: for five problems per model, the solution's mean NLL against E55's stored `mean_nll`
   (pass: max abs diff <= 1e-3);
2. C5: fifteen one-problem `are_robust` calls per model, sixty in all, timed with the model load
   included and the download excluded, as the evaluator's warm cache would have it (pass: under
   5 minutes in total); every call must return one native bool;
3. an unknown model id returns the constant.

Run via:  python scripts/kaggle_followup.py push e61-gpu-check --account <ALIAS>
"""

import json
import sys
import zipfile
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from e52_gpu_bundle import HEADER, MODELS, payload  # noqa: E402
from kaggle_launch import PIP_PINS  # noqa: E402

ZIP = ROOT / "dist" / "solution-e61b-quantile-small.zip"
RECORDS = "runs/e55-features-{short}-*/e55-features-{short}/familiarity-{short}/shard_*/records.jsonl"
JOBS = ["gpu-check"]


def problems() -> dict:
    """Per model: five parity problems with E55's NLL, and fifteen timing problems."""
    text = pd.read_parquet(ROOT / "data" / "e55" / "instances.parquet").drop_duplicates("problem_key")
    text = dict(zip(text["problem_key"], text["problem"]))
    out = {}
    for short in MODELS:
        rows = sorted((json.loads(line) for path in ROOT.glob(RECORDS.format(short=short))
                       for line in path.read_text(encoding="utf-8").splitlines() if line.strip()),
                      key=lambda r: r["problem_key"])
        out[short] = {"parity": [[text[r["problem_key"]], r["mean_nll"]] for r in rows[:5]],
                      "timing": [text[r["problem_key"]] for r in rows[5:20]]}
    return out


def render(job: str) -> tuple[str, dict]:
    assert job == "gpu-check"
    with zipfile.ZipFile(ZIP) as bundle:  # the ZIP's own files, so the check covers what ships
        extra = {f"e61/{name}": bundle.read(name) for name in ("solution.py", "tokenizer_check.py")}
    extra["problems.json"] = json.dumps(problems()).encode("utf-8")
    entries, hashes = payload((), extra)
    source = HEADER.format(out="e61-gpu-check", entries=json.dumps(entries)) + f'''subprocess.run([sys.executable, "-m", "pip", "install", "--no-cache-dir", "-q", *{PIP_PINS!r}], check=True)
sys.path.insert(0, str(root / "e61"))
import solution
problems = json.loads((root / "problems.json").read_text())
models = {MODELS!r}
result = {{"models": {{}}}}
from huggingface_hub import snapshot_download
for short, model_id in models.items():
    entry = {{}}
    # The evaluator serves weights from a warm, offline cache; download first, then time from the load.
    began = time.time()
    snapshot_download(model_id)
    entry["download_seconds"] = round(time.time() - began, 1)
    began = time.time()
    outputs = []
    for text in problems[short]["timing"]:
        outputs += solution.are_robust(model_id, [text])
    entry["c5_seconds_load_and_15_calls"] = round(time.time() - began, 2)
    tokenizer, model = solution._load(model_id)
    entry["parity_abs_diff"] = [abs(solution.mean_nll(tokenizer, model, text) - nll) for text, nll in problems[short]["parity"]]
    entry.update(native_bools=all(type(o) is bool for o in outputs), n_calls=len(outputs), robust_calls=int(sum(outputs)),
                 dtype=str(model.dtype))
    result["models"][model_id] = entry
    print(model_id, entry, flush=True)
    del tokenizer, model
    solution._cache.clear()
    import gc, torch
    gc.collect()
    torch.cuda.empty_cache()
    shutil.rmtree(Path(os.environ["HF_HOME"]) / "hub", ignore_errors=True)
result["unknown_model"] = solution.are_robust("example/unknown-model", ["1 + 1 = ?"])
max_diff = max(max(e["parity_abs_diff"]) for e in result["models"].values())
total = sum(e["c5_seconds_load_and_15_calls"] for e in result["models"].values())
result.update(c1_max_abs_diff=max_diff, c1_pass=max_diff <= 1e-3, c5_total_seconds=total,
              c5_pass=total < 300 and all(e["native_bools"] and e["n_calls"] == 15 for e in result["models"].values()),
              end_to_end_seconds=time.time() - started)
(out / "result.json").write_text(json.dumps(result, indent=2))
print(json.dumps(result, indent=2))
'''
    compile(source, "run.py", "exec")
    return source, hashes
