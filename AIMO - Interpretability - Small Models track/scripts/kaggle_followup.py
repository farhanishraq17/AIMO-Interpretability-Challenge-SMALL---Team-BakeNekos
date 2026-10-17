"""Launch bounded Small Models setup and E37 jobs using the accounts present on this machine.

The E37 CPU job has an explicit source allowlist and needs no local training data or model
weights. Credentials stay in the local child environment and never enter a kernel payload.
Capability checks request T4 GPUs and finish after reporting actual GPU and network access.

Run: python scripts/kaggle_followup.py push e37-confirmation --account Arifin
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "datagen"))
from kaggle_launch import READ_ONLY_ACCOUNTS, load_accounts  # noqa: E402
from e49_gpu_bundle import JOBS as E49_JOBS  # noqa: E402
from e52_gpu_bundle import JOBS as E52_JOBS  # noqa: E402
from e55_gpu_bundle import JOBS as E55_JOBS  # noqa: E402
from e57_gpu_bundle import JOBS as E57_JOBS  # noqa: E402
from e58_gpu_bundle import JOBS as E58_JOBS  # noqa: E402

FILES = (
    "train/eval_e36_budget_path.py", "train/eval_e37_gsm_confirmation.py",
    "train/run_e37_confirmation.py", "train/test_eval_e36_budget_path.py",
    "train/test_eval_e37_gsm_confirmation.py", "protocols/E37_PROTOCOL.md",
)
PINS = ["scikit-learn==1.8.0", "pandas==3.0.3", "joblib==1.5.3",
        "huggingface-hub==1.22.0", "pyarrow>=20", "pytest"]

CAPABILITY = '''import json, subprocess, urllib.request
from pathlib import Path
r = subprocess.run(["sh", "-c", "command -v nvidia-smi >/dev/null && nvidia-smi --query-gpu=name,memory.total --format=csv,noheader"], capture_output=True, text=True)
result = {"gpu_available": r.returncode == 0, "gpus": r.stdout.strip().splitlines()}
try:
    with urllib.request.urlopen("https://huggingface.co/api/datasets/ssubhnil/Qwen3-Math-Eval", timeout=30) as response:
        result["internet_available"] = response.status == 200
except Exception as exc:
    result["internet_available"] = False
    result["network_error"] = type(exc).__name__
Path("/kaggle/working/capability.json").write_text(json.dumps(result, indent=2))
print(json.dumps(result), flush=True)
'''


def render(job: str) -> tuple[str, dict]:
    if job == "capability-check":
        return CAPABILITY, {}
    if job == "cuda-link-check":
        source = (ROOT / "scripts/kaggle_cuda_setup.py").read_text(encoding="utf-8")
        return source, {"scripts/kaggle_cuda_setup.py": hashlib.sha256(source.encode()).hexdigest()}
    if job.startswith(("e37-official-", "e37-retry-")):
        from e37_gpu_bundle import render as render_gpu
        return render_gpu(job.removeprefix("e37-official-").removeprefix("e37-retry-"))
    if job.startswith("e49-"):
        from e49_gpu_bundle import render as render_e49
        return render_e49(job.removeprefix("e49-"))
    if job.startswith("e58-"):
        from e58_gpu_bundle import render as render_e58
        return render_e58(job.removeprefix("e58-"))
    if job.startswith("e57-"):
        from e57_gpu_bundle import render as render_e57
        return render_e57(job.removeprefix("e57-"))
    if job.startswith("e55-"):
        from e55_gpu_bundle import render as render_e55
        return render_e55(job.removeprefix("e55-"))
    if job.startswith("e52-"):
        from e52_gpu_bundle import render as render_e52
        return render_e52(job.removeprefix("e52-"))
    files = FILES if job == "e37-confirmation" else (
        "train/audit_e38_prefix.py", "train/test_audit_e38_prefix.py", "protocols/E38_PROTOCOL.md")
    tests = (["train/test_eval_e36_budget_path.py", "train/test_eval_e37_gsm_confirmation.py"]
             if job == "e37-confirmation" else ["train/test_audit_e38_prefix.py"])
    runner = "train/run_e37_confirmation.py" if job == "e37-confirmation" else "train/audit_e38_prefix.py"
    output_dir = "e37" if job == "e37-confirmation" else "e38"
    entries = {f: (ROOT / f).read_text(encoding="utf-8") for f in files}
    hashes = {f: hashlib.sha256(text.encode()).hexdigest() for f, text in entries.items()}
    source = f'''import json, os, subprocess, sys
from pathlib import Path
os.environ["HF_HUB_DISABLE_XET"] = "1"
os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"
os.environ["OMP_NUM_THREADS"] = "2"
subprocess.run([sys.executable, "-m", "pip", "install", "-q", {", ".join(repr(p) for p in PINS)}], check=True)
root = Path("/tmp/aimo-e37")
entries = json.loads({json.dumps(entries)!r})
for name, text in entries.items():
    path = root / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
subprocess.run([sys.executable, "-m", "pytest", "-q", *{tests!r}], cwd=root, check=True)
subprocess.run([sys.executable, {runner!r}, "--cache-dir", "/tmp/e37-hf", "--out-dir", "/kaggle/working/{output_dir}"], cwd=root, check=True)
'''
    if "KGAT_" in source or "Huggingphase_token" in source:
        raise ValueError("credential marker in kernel source")
    compile(source, "run.py", "exec")
    return source, hashes


def call(alias: str, *args: str) -> tuple[int, str]:
    _, token = load_accounts()[alias]
    env = {k: v for k, v in os.environ.items() if k not in ("KAGGLE_USERNAME", "KAGGLE_KEY")}
    env.update(KAGGLE_API_TOKEN=token, PYTHONIOENCODING="utf-8", PYTHONUTF8="1")
    result = subprocess.run([sys.executable, "-m", "kaggle", *args], capture_output=True,
                            text=True, encoding="utf-8", errors="replace", env=env, timeout=180)
    output = (result.stdout or "") + (result.stderr or "")
    return result.returncode, re.sub(r"KGAT_[0-9A-Za-z]+", "[REDACTED]", output.replace(token, "[REDACTED]"))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["build", "push", "status", "output"])
    parser.add_argument("job", choices=["capability-check", "cuda-link-check", "e37-confirmation", "e37-official-qwen",
                                       "e37-official-skywork", "e37-official-olmo", "e37-official-deepseek", "e38-prefix-audit",
                                       "e37-retry-qwen", "e37-retry-skywork", "e37-retry-olmo", "e37-retry-deepseek",
                                       *(f"e49-{job}" for job in E49_JOBS), *(f"e52-{job}" for job in E52_JOBS), *(f"e55-{job}" for job in E55_JOBS), *(f"e57-{job}" for job in E57_JOBS), *(f"e58-{job}" for job in E58_JOBS)])
    parser.add_argument("--account", required=True, choices=sorted(load_accounts()))
    args = parser.parse_args()
    if args.command == "push" and args.account in READ_ONLY_ACCOUNTS:
        raise SystemExit(f"{args.account} is read-only: no pushes, ever (CLAUDE.md, Kaggle fleet)")
    username, _ = load_accounts()[args.account]
    slug = f"aimo-small-{args.job}"
    ref = f"{username}/{slug}"
    path = ROOT / "kaggle_runs" / f"{args.job}-{args.account}"
    if args.command in ("build", "push"):
        source, hashes = render(args.job)
        path.mkdir(parents=True, exist_ok=True)
        (path / "run.py").write_text(source, encoding="utf-8")
        metadata = {"id": ref, "title": slug, "code_file": "run.py", "language": "python",
                    "kernel_type": "script", "is_private": True, "enable_gpu": args.job not in ("e37-confirmation", "e38-prefix-audit"),
                    "enable_internet": True, "dataset_sources": [], "competition_sources": [], "kernel_sources": []}
        (path / "kernel-metadata.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
        print(f"Built {ref}; private={metadata['is_private']}; GPU={metadata['enable_gpu']}")
        if args.command == "build":
            return
        flags = ["--accelerator", "NvidiaTeslaT4"] if metadata["enable_gpu"] else []
        code, output = call(args.account, "kernels", "push", "-p", str(path), *flags)
        # Kaggle CLI can return zero even when the server rejects a push.
        if code == 0 and "successfully pushed" not in output:
            code = 1
        if code == 0:
            record = {"job": args.job, "account": args.account, "ref": ref, "source_sha256": hashes,
                      "kernel_sha256": hashlib.sha256(source.encode()).hexdigest(),
                      "submitted_at": datetime.now(timezone.utc).isoformat(), "metadata": metadata}
            dest = ROOT / "kernels" / "runs" / f"{args.job}-{args.account}"
            dest.mkdir(parents=True, exist_ok=True)
            (dest / "launch.json").write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
            (dest / "run.py").write_text(source, encoding="utf-8")
    elif args.command == "status":
        code, output = call(args.account, "kernels", "status", ref)
    else:
        dest = ROOT / "runs" / f"{args.job}-{args.account}"
        dest.mkdir(parents=True, exist_ok=True)
        code, output = call(args.account, "kernels", "output", ref, "-p", str(dest), "--page-size", "200", "-o")
    print(output.strip())
    raise SystemExit(code)


if __name__ == "__main__":
    main()
