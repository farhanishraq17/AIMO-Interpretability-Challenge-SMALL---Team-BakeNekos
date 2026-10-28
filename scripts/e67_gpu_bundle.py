"""Allowlisted E67 bundles: Qwen3.5-4B sampled under transformers with E64's difficulty direction pushed at prefill.

`datagen/perturb/steer.py` does the work; this packs it with E64's evaluation pins (Kaggle's torch kept, as the E64
GPU check ran) and E52's per-GPU layout: one process per T4, each on half of the kernel's problems, under the process
guard. vLLM is not used: it cannot take the forward hook.

`timing` came first, before `protocols/E67_PROTOCOL.md` fixed its sizes: two E62 problems with both base samples at the
32k cap (E67 excludes them), one per T4, at E67's 16k cap, so every row runs to the cap and the kernel times the worst
case. `a` to `d` are E67 proper: `data/e67/pool.parquet` dealt round-robin in `problem_key` order.

Run via:  python scripts/kaggle_followup.py push e67-a --account <ALIAS>
"""

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "datagen"))
sys.path.insert(0, str(ROOT / "datagen" / "perturb"))
sys.path.insert(0, str(ROOT / "scripts"))
from e52_gpu_bundle import HEADER, payload  # noqa: E402
from e67_pool import capped_twice  # noqa: E402
from extract_features import DEFAULT_ARTIFACT, load_system_prompt  # noqa: E402
from kaggle_launch import PIP_PINS  # noqa: E402

FILES = ("datagen/perturb/steer.py", "datagen/perturb/sample.py", "datagen/perturb/answers.py",
         "datagen/extract_features.py", "datagen/tokenizer_check.py", "scripts/kaggle_process_guard.py",
         "data/e67/directions.npz")
SYSTEM_PROMPT = "data/probe_system_prompt.txt"
KERNELS, HOURS = 4, 9.0  # E67 §3; --max-hours per process, the guard allows half an hour more
PARTS = "abcd"
JOBS = ["timing", *PARTS[:KERNELS]]


def render(job: str) -> tuple[str, dict]:
    extra = {SYSTEM_PROMPT: load_system_prompt(DEFAULT_ARTIFACT).encode("utf-8")}
    if job == "timing":
        files, pool, hours = FILES + ("data/e62/pool.parquet",), "data/e62/pool.parquet", 1.5
        keys = capped_twice()[:2]
    else:
        files = FILES + ("data/e67/pool.parquet", "protocols/E67_PROTOCOL.md")
        pool, hours = "data/e67/pool.parquet", HOURS
        import pandas as pd
        keys = sorted(pd.read_parquet(ROOT / pool)["problem_key"])[PARTS.index(job)::KERNELS]
    entries, hashes = payload(files, extra)
    out = f"e67-{job}"
    argv = ["datagen/perturb/steer.py", "--pool", pool, "--directions", "data/e67/directions.npz",
            "--system-prompt-file", SYSTEM_PROMPT, "--max-hours", str(hours), "--local-files-only",
            "--out-dir", f"/kaggle/working/{out}", "--problem-keys", *keys]
    source = HEADER.format(out=out, entries=json.dumps(entries)) + f'''subprocess.run([sys.executable, "-m", "pip", "install", "--no-cache-dir", "-q", *{PIP_PINS!r}], check=True)
freeze = subprocess.run([sys.executable, "-m", "pip", "freeze"], capture_output=True, text=True, check=True)
(out / "environment.txt").write_text(freeze.stdout)
from huggingface_hub import snapshot_download
snapshot_download("Qwen/Qwen3.5-4B")  # once; the two processes then load offline (e67-timing v2 lost one to the Hub)
guard = runpy.run_path(str(root / "scripts/kaggle_process_guard.py"))["run_guarded"]
argv = {argv!r}
# One process per T4, each on half of this kernel's problems.
launcher = ("import os, subprocess, sys\\n"
            f"argv = {{argv!r}}\\n"
            "procs = [subprocess.Popen([sys.executable, *argv, '--num-shards', '2', '--shard-index', str(i)],\\n"
            "         env={{**os.environ, 'CUDA_VISIBLE_DEVICES': str(i),\\n"
            "              'PYTORCH_CUDA_ALLOC_CONF': 'expandable_segments:True', 'HF_HUB_OFFLINE': '1'}}) for i in (0, 1)]\\n"
            "sys.exit(max(p.wait() for p in procs))")
result = guard([sys.executable, "-c", launcher], root, out / "steer.log", timeout={hours * 3600 + 1800})
(out / "execution.json").write_text(json.dumps({{**result, "end_to_end_seconds": time.time() - started}}, indent=2))
raise SystemExit(result["exit_code"])
'''
    compile(source, "run.py", "exec")
    return source, hashes
