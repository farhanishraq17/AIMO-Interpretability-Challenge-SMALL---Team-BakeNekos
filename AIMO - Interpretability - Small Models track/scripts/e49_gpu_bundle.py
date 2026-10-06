"""Allowlisted E49 bundle: the batched `transformers` extractor and exactly the files it imports.

Unlike E37's bundle there is no vLLM, no separate venv and no CUDA linker repair. E49 runs the
evaluation image's libraries (the `Dockerfile.competition` pins, with Kaggle's own torch), as a
submission would. Per-model batch caps come from each checkpoint's KV cache per token on a T4 pair
in fp16 (`protocols/E49_PROTOCOL.md` §3). A cap that proves too large halves itself on
out-of-memory and records it.

Run via:  python scripts/kaggle_followup.py push e49-qwen --account <ALIAS>
"""

import base64
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "datagen"))
from kaggle_launch import PIP_PINS  # noqa: E402

FILES = (
    "datagen/extract_budget_path_hf.py", "datagen/extract_budget_path.py", "datagen/extract_features.py",
    "datagen/tokenizer_check.py", "datagen/perturb/answers.py", "datagen/perturb/sample.py",
    "train/eval_e36_budget_path.py", "protocols/E49_PROTOCOL.md",
    "data/train_main/rows.parquet", "data/train_main/summary.json", "scripts/kaggle_process_guard.py",
)
# model -> (checkpoint, --max-batch-tokens, --group-size)
SETTINGS = {
    "qwen": ("Qwen/Qwen3.5-4B", 400_000, 0),
    "skywork": ("Skywork/Skywork-OR1-Math-7B", 150_000, 0),
    "olmo": ("allenai/Olmo-3-7B-Think", 50_000, 3),
    "deepseek": ("deepseek-ai/DeepSeek-R1-0528-Qwen3-8B", 60_000, 5),
}
GUARD_SECONDS = 11 * 3600  # Kaggle ends sessions at 12 h; the extractor starts no new group after 10


def render(model: str) -> tuple[str, dict]:
    checkpoint, cap, group = SETTINGS[model]
    entries = {name: base64.b64encode((ROOT / name).read_bytes()).decode() for name in FILES}
    hashes = {name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest() for name in FILES}
    source = f'''import base64, json, os, runpy, subprocess, sys, time
from pathlib import Path
os.environ.update(HF_HUB_DISABLE_XET="1", HF_HUB_DISABLE_PROGRESS_BARS="1", HF_HOME="/tmp/e49-hf",
                  PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True")
started = time.time()
out = Path("/kaggle/working/e49-{model}")
out.mkdir(parents=True, exist_ok=True)
subprocess.run(["nvidia-smi"], check=True)
root = Path("/tmp/aimo-e49")
for name, encoded in json.loads({json.dumps(entries)!r}).items():
    dest = root / name
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_bytes(base64.b64decode(encoded))
subprocess.run([sys.executable, "-m", "pip", "install", "--no-cache-dir", "-q", *{PIP_PINS!r}], check=True)
freeze = subprocess.run([sys.executable, "-m", "pip", "freeze"], capture_output=True, text=True, check=True)
(out / "environment.txt").write_text(freeze.stdout)
cmd = [sys.executable, "datagen/extract_budget_path_hf.py", "--model-id", {checkpoint!r}, "--out-dir", str(out),
       "--dtype", "float16", "--max-batch-tokens", "{cap}", "--group-size", "{group}", "--max-hours", "10"]
result = runpy.run_path(str(root / "scripts/kaggle_process_guard.py"))["run_guarded"](cmd, root, out / "runner.log", timeout={GUARD_SECONDS})
(out / "execution.json").write_text(json.dumps({{**result, "end_to_end_seconds": time.time() - started, "command": cmd}}, indent=2))
raise SystemExit(result["exit_code"])
'''
    compile(source, "run.py", "exec")
    return source, hashes
