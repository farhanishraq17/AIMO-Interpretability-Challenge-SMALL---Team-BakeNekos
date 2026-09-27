"""Allowlisted E37 transfer bundle; no credential or unrelated data files."""
import base64
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MODELS = {
    "qwen": "Qwen/Qwen3.5-4B",
    "skywork": "Skywork/Skywork-OR1-Math-7B",
    "olmo": "allenai/Olmo-3-7B-Think",
    "deepseek": "deepseek-ai/DeepSeek-R1-0528-Qwen3-8B",
}
FILES = ("datagen/extract_budget_path.py", "datagen/tokenizer_check.py",
         "datagen/perturb/answers.py", "datagen/perturb/sample.py", "datagen/extract_features.py",
         "train/eval_e36_budget_path.py", "protocols/E37_PROTOCOL.md",
         "data/train_main/rows.parquet", "data/train_main/provenance.json")


def render(model):
    gate = json.loads((ROOT / "runs/e37-confirmation-Arifin/e37/result.json").read_text())
    if not gate["passed"] or not all(gate["gates"].values()):
        raise ValueError("E37 confirmation gate is closed")
    entries = {name: base64.b64encode((ROOT / name).read_bytes()).decode() for name in FILES}
    hashes = {name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest() for name in FILES}
    # vLLM 0.20's Olmo loader reads flat rope_parameters; Transformers 5.13 nests
    # them by layer type. This compatible pin preserves the checkpoint's values.
    transformers_pin = "5.5.3" if model == "olmo" else "5.13.0"
    source = f'''import base64, json, os, subprocess, sys, time
from pathlib import Path
os.environ.update(HF_HUB_DISABLE_XET="1", HF_HUB_DISABLE_PROGRESS_BARS="1",
                  HF_HOME="/tmp/e37-hf", OMP_NUM_THREADS="2", VLLM_WORKER_MULTIPROC_METHOD="spawn")
started = time.time()
out = Path("/kaggle/working/e37-{model}")
out.mkdir(parents=True, exist_ok=True)
subprocess.run(["nvidia-smi"], check=True)
# CUDA 12 generation; avoid the CUDA 13 default wheels in newer vLLM releases.
subprocess.run([sys.executable, "-m", "venv", "--without-pip", "/tmp/e37-env"], check=True)
py = "/tmp/e37-env/bin/python"
subprocess.run([sys.executable, "-m", "pip", "--python", py, "install", "--no-cache-dir", "-q", "vllm==0.20.0",
               "transformers=={transformers_pin}", "huggingface-hub==1.22.0", "scikit-learn==1.8.0",
               "pandas==3.0.3", "joblib==1.5.3", "pyarrow==24.0.0"], check=True)
root = Path("/tmp/aimo-e37")
for name, encoded in json.loads({json.dumps(entries)!r}).items():
    dest = root / name
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_bytes(base64.b64decode(encoded))
freeze = subprocess.run([sys.executable, "-m", "pip", "--python", py, "freeze"], capture_output=True, text=True, check=True)
(out / "environment.txt").write_text(freeze.stdout)
cmd = [py, "datagen/extract_budget_path.py", "--model-id", {MODELS[model]!r},
       "--out-dir", str(out), "--tensor-parallel", "2", "--max-num-seqs", "2", "--chunk", "2"]
if {model!r} == "qwen":
    cmd.append("--text-only")
result = subprocess.run(cmd, cwd=root)
(out / "execution.json").write_text(json.dumps({{"exit_code": result.returncode,
    "end_to_end_seconds": time.time() - started, "command": cmd}}, indent=2))
raise SystemExit(result.returncode)
'''
    compile(source, "run.py", "exec")
    return source, hashes
