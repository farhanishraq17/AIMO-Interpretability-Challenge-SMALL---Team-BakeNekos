"""Allowlisted E52 bundles: contradiction sampling under vLLM, and the one-pass features under HF.

Sampling follows E37's working recipe (CUDA linker check, vLLM 0.20.0 in its own venv) with E24's
engine layout: Qwen3.5-4B runs one engine per T4, Skywork tensor-parallel over both. Each model's
75 problems are split in two by sorted `problem_key`, and each half is one kernel running both
phases, so eligibility and rewrites stay in one session (`protocols/E52_PROTOCOL.md` §2).

The features kernel needs no vLLM: the evaluation pins, then familiarity and Stage A for all four
models, merged on Kaggle with the per-problem vectors deleted so the output stays one page.

Run via:  python scripts/kaggle_followup.py push e52-qwen-a --account <ALIAS>
"""

import base64
import hashlib
import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "datagen"))
from extract_features import DEFAULT_ARTIFACT, load_system_prompt  # noqa: E402
from kaggle_launch import PIP_PINS  # noqa: E402

MODELS = {"qwen": "Qwen/Qwen3.5-4B", "skywork": "Skywork/Skywork-OR1-Math-7B",
          "olmo": "allenai/Olmo-3-7B-Think", "deepseek": "deepseek-ai/DeepSeek-R1-0528-Qwen3-8B"}
SAMPLE_FILES = ("datagen/perturb/sample.py", "datagen/perturb/answers.py", "datagen/tokenizer_check.py",
                "datagen/extract_features.py", "data/reliablemath/pool.parquet", "data/reliablemath/variants.parquet",
                "scripts/kaggle_cuda_setup.py", "scripts/kaggle_process_guard.py", "protocols/E52_PROTOCOL.md")
FEATURE_FILES = ("datagen/extract_familiarity.py", "datagen/extract_features.py", "datagen/merge_features.py",
                 "datagen/tokenizer_check.py", "datagen/ingest_public.py", "data/reliablemath/instances.parquet",
                 "protocols/E52_PROTOCOL.md")
SYSTEM_PROMPT = "data/probe_system_prompt.txt"
# Sampling settings fixed by the protocol: the organizers' decoding, k = 2, 16k cap.
DECODING = ["--k", "2", "--temperature", "1.0", "--top-p", "0.95", "--top-k", "40", "--max-new-tokens", "16384"]
ENGINE = {"qwen": ["--tensor-parallel", "1", "--text-only", "--max-num-seqs", "64", "--chunk", "16"],
          "skywork": ["--tensor-parallel", "2", "--chunk", "12"],
          # E52b: Olmo keeps a full cache on 8 of 32 layers, so it gets all the memory a T4 pair allows.
          # Eager mode: CUDA-graph capture reserved 4.72 GiB a GPU and left 1.61 GiB of cache (E52b pilot).
          "olmo": ["--tensor-parallel", "2", "--gpu-memory-utilization", "0.95", "--enforce-eager", "--chunk", "8"]}
PER_GPU = {"qwen": True, "skywork": False, "olmo": False}
# --max-hours per phase; the guard allows half an hour more. E52b bounds Olmo tighter to cap quota.
HOURS = {"base": 3.5, "variants": 6.0}
MODEL_HOURS = {"olmo": {"base": 2.5, "variants": 4.0}}
# vLLM 0.20 reads Olmo's rope parameters flat; transformers 5.13 nests them (E37 config check).
TRANSFORMERS = {"olmo": "5.5.3"}
SPLITS = {"qwen": 2, "skywork": 2, "olmo": 4}  # kernels per model, each a slice of the sorted problem keys
JOBS = [f"{m}-{part}" for m, n in SPLITS.items() for part in "abcd"[:n]] + ["features"]


def part_keys(model: str, part: str) -> list[str]:
    keys = sorted(pd.read_parquet(ROOT / "data/reliablemath/pool.parquet")["problem_key"])
    return keys["abcd".index(part)::SPLITS[model]]


def payload(files: tuple[str, ...], extra: dict[str, bytes] | None = None) -> tuple[dict, dict]:
    blobs = {name: (ROOT / name).read_bytes() for name in files} | (extra or {})
    return ({n: base64.b64encode(b).decode() for n, b in blobs.items()},
            {n: hashlib.sha256(b).hexdigest() for n, b in blobs.items()})


HEADER = '''import base64, json, os, runpy, shutil, subprocess, sys, time
from pathlib import Path
os.environ.update(HF_HUB_DISABLE_XET="1", HF_HUB_DISABLE_PROGRESS_BARS="1", HF_HOME="/tmp/e52-hf",
                  OMP_NUM_THREADS="2", VLLM_WORKER_MULTIPROC_METHOD="spawn")
started = time.time()
out = Path("/kaggle/working/{out}")
out.mkdir(parents=True, exist_ok=True)
subprocess.run(["nvidia-smi"], check=True)
root = Path("/tmp/aimo-e52")
for name, encoded in json.loads({entries!r}).items():
    dest = root / name
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_bytes(base64.b64decode(encoded))
'''


def render_sampling(model: str, half: str) -> tuple[str, dict]:
    entries, hashes = payload(SAMPLE_FILES)
    out = f"e52-{model}-{half}"
    hours = MODEL_HOURS.get(model, HOURS)
    common = ["datagen/perturb/sample.py", "--model-id", MODELS[model], "--backend", "vllm",
              "--pool", "data/reliablemath/pool.parquet", "--variants", "data/reliablemath/variants.parquet",
              *DECODING, *ENGINE[model], "--out-dir", f"/kaggle/working/{out}", "--problem-keys", *part_keys(model, half)]
    phases = {
        "base": [*common, "--phase", "base", "--max-hours", str(hours["base"])],
        "variants": [*common, "--phase", "variants", "--max-hours", str(hours["variants"]),
                     "--base-records", f"/kaggle/working/{out}/base/shard_*/records.jsonl"],
    }
    source = HEADER.format(out=out, entries=json.dumps(entries)) + f'''cuda = runpy.run_path(str(root / "scripts/kaggle_cuda_setup.py"))["prepare_cuda_linker"]()
(out / "cuda-link-check.json").write_text(json.dumps(cuda, indent=2))
subprocess.run([sys.executable, "-m", "venv", "--without-pip", "/tmp/e52-env"], check=True)
py = "/tmp/e52-env/bin/python"
subprocess.run([sys.executable, "-m", "pip", "--python", py, "install", "--no-cache-dir", "-q", "vllm==0.20.0",
               "transformers=={TRANSFORMERS.get(model, "5.13.0")}", "huggingface-hub==1.22.0", "pandas==3.0.3", "pyarrow==24.0.0"], check=True)
freeze = subprocess.run([sys.executable, "-m", "pip", "--python", py, "freeze"], capture_output=True, text=True, check=True)
(out / "environment.txt").write_text(freeze.stdout)
guard = runpy.run_path(str(root / "scripts/kaggle_process_guard.py"))["run_guarded"]
phases = {phases!r}
results = {{}}
for phase, argv in phases.items():
    if {PER_GPU[model]!r}:
        # One engine per T4 (E24's layout for Qwen3.5-4B), each on half of this kernel's items.
        launcher = ("import os, subprocess, sys\\n"
                    f"argv = {{argv!r}}\\n"
                    "procs = [subprocess.Popen([sys.argv[1], *argv, '--num-shards', '2', '--shard-index', str(i)],\\n"
                    "         env={{**os.environ, 'CUDA_VISIBLE_DEVICES': str(i)}}) for i in (0, 1)]\\n"
                    "sys.exit(max(p.wait() for p in procs))")
        cmd = [py, "-c", launcher, py]
    else:
        cmd = [py, *argv]
    results[phase] = guard(cmd, root, out / f"{{phase}}.log", timeout={hours["variants"] * 3600 + 1800}
                           if phase == "variants" else {hours["base"] * 3600 + 1800})
    (out / "execution.json").write_text(json.dumps({{**results, "end_to_end_seconds": time.time() - started}}, indent=2))
raise SystemExit(max(r["exit_code"] for r in results.values()))
'''
    compile(source, "run.py", "exec")
    return source, hashes


def render_features() -> tuple[str, dict]:
    prompt = load_system_prompt(DEFAULT_ARTIFACT).encode("utf-8")
    entries, hashes = payload(FEATURE_FILES, {SYSTEM_PROMPT: prompt})
    source = HEADER.format(out="e52-features", entries=json.dumps(entries)) + f'''subprocess.run([sys.executable, "-m", "pip", "install", "--no-cache-dir", "-q", *{PIP_PINS!r}], check=True)
freeze = subprocess.run([sys.executable, "-m", "pip", "freeze"], capture_output=True, text=True, check=True)
(out / "environment.txt").write_text(freeze.stdout)
instances = "data/reliablemath/instances.parquet"
codes = {{}}
for short, model in {MODELS!r}.items():
    steps = {{
        "familiarity": ["datagen/extract_familiarity.py", "--model-id", model, "--instances", instances,
                        "--out-dir", str(out / f"familiarity-{{short}}")],
        "stagea": ["datagen/extract_features.py", "--model-id", model, "--instances", instances,
                   "--system-prompt-file", "{SYSTEM_PROMPT}", "--out-dir", str(out / f"stagea-{{short}}")],
        "merge": ["datagen/merge_features.py", "--features-dir", str(out / f"stagea-{{short}}")],
    }}
    for step, argv in steps.items():
        codes[f"{{short}}-{{step}}"] = subprocess.run([sys.executable, *argv], cwd=root).returncode
    for vectors in (out / f"stagea-{{short}}").glob("shard_*/vectors"):
        shutil.rmtree(vectors)  # merged into features.npz; one file per problem would overflow the output page
    shutil.rmtree("/tmp/e52-hf/hub", ignore_errors=True)  # four 4-8B checkpoints do not fit on disk together
    (out / "execution.json").write_text(json.dumps({{"exit_codes": codes, "end_to_end_seconds": time.time() - started}}, indent=2))
raise SystemExit(max(codes.values()))
'''
    compile(source, "run.py", "exec")
    return source, hashes


def render(job: str) -> tuple[str, dict]:
    if job == "features":
        return render_features()
    model, half = job.rsplit("-", 1)
    return render_sampling(model, half)
