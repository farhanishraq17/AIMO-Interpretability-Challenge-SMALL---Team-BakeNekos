"""Allowlisted E58 bundles: answer-property variants of the official problems, sampled at each model's T4 cap.

E52's recipe (`e52_gpu_bundle.py`: CUDA linker check, vLLM 0.20.0 in its own venv, the process guard)
with E58's settings from `protocols/E58_PROTOCOL.md` §3: k = 4 for the originals and every variant, and
the longest cap each model's T4 layout holds. Each kernel takes a slice of its model's sorted problem keys
and runs both phases, so eligibility and variants live in one session. Olmo-3-Think runs one sequence at
a time, the only setting under which vLLM 0.20 sampled it at 16k without dying (Finding 78), so it gets
one kernel per problem.

E59 (`protocols/E59_PROTOCOL.md`) reruns the same kernels with one change, `--allow-no-solution`, so the prompt
offers a way to decline; Qwen gets five kernels instead of three, since `e58-qwen-c` hit its guard.

E62 (`protocols/E62_PROTOCOL.md` §3) is Qwen's base phase only, on E62's own 497-problem pool: k = 2 at a 32k cap,
eight hash-ordered slices. `e62v-qwen-*` then samples each slice's contradictions (k = 4) on the problems its
downloaded base records call eligible, the records bundled as E60 bundled E59's.

Run via:  python scripts/kaggle_followup.py push e58-qwen-a --account <ALIAS>   (or e59-qwen-a)
"""

import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from e52_gpu_bundle import HEADER, MODELS, TRANSFORMERS, payload  # noqa: E402

FILES = ("datagen/perturb/sample.py", "datagen/perturb/answers.py", "datagen/tokenizer_check.py",
         "datagen/extract_features.py", "data/e58/pool.parquet", "data/e58/variants.parquet",
         "scripts/kaggle_cuda_setup.py", "scripts/kaggle_process_guard.py")
DECODING = ["--k", "4", "--temperature", "1.0", "--top-p", "0.95", "--top-k", "40"]
CAP = {"qwen": 65536, "deepseek": 32768, "skywork": 32768, "olmo": 16384}
ENGINE = {"qwen": ["--tensor-parallel", "1", "--text-only", "--max-num-seqs", "64", "--chunk", "4"],
          "deepseek": ["--tensor-parallel", "2", "--chunk", "4"],
          "skywork": ["--tensor-parallel", "2", "--chunk", "4"],
          # Finding 78: eager, utilisation 0.85 for FlashInfer's workspace, and one sequence at a time,
          # since vLLM 0.20 sizes Olmo's cache as full attention (20,288 tokens) and dies instead of preempting.
          "olmo": ["--tensor-parallel", "2", "--gpu-memory-utilization", "0.85", "--enforce-eager",
                   "--max-num-seqs", "1", "--chunk", "1"]}
PER_GPU = {"qwen": True, "deepseek": False, "skywork": False, "olmo": False}
HOURS = {"base": 2.5, "variants": 7.0}  # --max-hours per phase; the guard allows half an hour more
SPLITS = {"e58": {"qwen": 3, "deepseek": 4, "skywork": 2, "olmo": 8},
          "e59": {"qwen": 5, "deepseek": 4, "skywork": 1, "olmo": 8},
          "e60": {"qwen": 5, "deepseek": 4, "skywork": 1, "olmo": 8},
          "e62": {"qwen": 8}, "e62v": {"qwen": 8}}
# E62: its own pool, k = 2, a 32k cap, and base kernels long enough for 62 problems a slice (chunk 16 for throughput).
POOL = {"e62": "data/e62/pool.parquet", "e62v": "data/e62/pool.parquet"}
E62 = {"k": "2", "cap": 32768, "hours": 9.0, "chunk": "16"}
E62_BASE = "runs/e62-{model}-{part}-*/e62-{model}-{part}/base/shard_*/records.jsonl"
PROTOCOL = {"e62v": "E62"}  # bundled beside the run; default: the experiment's own
FLAGS = {"e58": [], "e59": ["--allow-no-solution"],
         # E60 §3: E59's prompt; DeepSeek saves each item as it finishes (E59 amendment 1's lesson).
         "e60": ["--allow-no-solution"], "e62": ["--allow-no-solution"], "e62v": ["--allow-no-solution"]}
VARIANTS = {"e58": "data/e58/variants.parquet", "e59": "data/e58/variants.parquet", "e60": "data/e60/variants.parquet",
            "e62v": "data/e62/variants.parquet"}
# E60 samples only its contradictions; eligibility comes from each model's E59 base records, bundled.
E59_BASE = "runs/e59-{model}-*/e59-{model}-*/base/shard_*/records.jsonl"
# Amendment 1: a kernel stopped by its guard resumes from its own downloaded records, so the sampler skips
# every item already recorded and base eligibility is not re-drawn.
# E59 amendment 1: e59-deepseek-b's first chunk of four items took 6.9 h, so its resume saves each item as it
# finishes (`--chunk 1`, an engine setting) instead of risking the guard with nothing flushed.
RESUME = {"e58": {"qwen-c": ("runs/e58-qwen-c-TASHIN/e58-qwen-c", None)},
          "e59": {"deepseek-b": ("runs/e59-deepseek-b-DIDHITI/e59-deepseek-b", "1")}}
JOBS = [f"{m}-{part}" for m, n in SPLITS["e58"].items() for part in "abcdefgh"[:n]] + [f"{job}-resume" for job in RESUME["e58"]]
E59_JOBS = [f"{m}-{part}" for m, n in SPLITS["e59"].items() for part in "abcdefgh"[:n]] + [f"{job}-resume" for job in RESUME["e59"]]
E60_JOBS = [f"{m}-{part}" for m, n in SPLITS["e60"].items() for part in "abcdefgh"[:n]]
E62_JOBS = [f"{m}-{part}" for m, n in SPLITS["e62"].items() for part in "abcdefgh"[:n]]
E62V_JOBS = [f"{m}-{part}" for m, n in SPLITS["e62v"].items() for part in "abcdefgh"[:n]]


def part_keys(model: str, part: str, experiment: str = "e58") -> list[str]:
    if experiment in POOL:
        keys = sorted(pd.read_parquet(ROOT / POOL[experiment])["problem_key"])
    else:
        rows = pd.read_parquet(ROOT / "data/e58/rows.parquet")
        keys = sorted(rows.loc[rows.model_id == MODELS[model], "problem_key"])
    return keys["abcdefgh".index(part)::SPLITS[experiment][model]]


def render(job: str, experiment: str = "e58") -> tuple[str, dict]:
    kernel = job.removesuffix("-resume")
    model, part = kernel.rsplit("-", 1)
    previous, chunk = RESUME[experiment][kernel] if job.endswith("-resume") else (None, None)
    previous = previous and ROOT / previous
    extra = {f"resume/{f.relative_to(previous).as_posix()}": f.read_bytes()
             for f in previous.glob("*/shard_*/*") if f.name in ("records.jsonl", "metadata.json")} if previous else {}
    if experiment == "e60":
        records = {json.loads(line)["item_id"]: line for path in sorted(ROOT.glob(E59_BASE.format(model=model)))
                   for line in path.read_text(encoding="utf-8").splitlines() if line.strip()}
        extra["e59_base/records.jsonl"] = ("\n".join(records.values()) + "\n").encode()
    if experiment == "e62v":
        records = {json.loads(line)["item_id"]: line for path in sorted(ROOT.glob(E62_BASE.format(model=model, part=part)))
                   for line in path.read_text(encoding="utf-8").splitlines() if line.strip()}
        if not records:
            raise SystemExit(f"no downloaded base records for e62-{model}-{part}; run `output e62-{model}-{part}` first")
        extra["e62_base/records.jsonl"] = ("\n".join(records.values()) + "\n").encode()
    pool = POOL.get(experiment, "data/e58/pool.parquet")
    files = tuple(f for f in FILES if f not in ("data/e58/variants.parquet", "data/e58/pool.parquet")) + (pool,)
    files += (VARIANTS[experiment],) if experiment in VARIANTS else ()
    entries, hashes = payload((*files, f"protocols/{PROTOCOL.get(experiment, experiment.upper())}_PROTOCOL.md"), extra)
    out = f"{experiment}-{model}-{part}"
    common = ["datagen/perturb/sample.py", "--model-id", MODELS[model], "--backend", "vllm",
              "--pool", pool, *(["--variants", VARIANTS[experiment]] if experiment in VARIANTS else []),
              *DECODING, "--max-new-tokens", str(CAP[model]), *ENGINE[model], *FLAGS[experiment],
              *(["--max-new-tokens", str(E62["cap"]), "--chunk", E62["chunk"]] if experiment.startswith("e62") else []),
              *(["--k", E62["k"]] if experiment == "e62" else []),
              *(["--chunk", chunk] if chunk else []),
              *(["--chunk", "1"] if experiment == "e60" and model == "deepseek" else []),
              "--out-dir", f"/kaggle/working/{out}", "--problem-keys", *part_keys(model, part, experiment)]
    phases = {
        "base": [*common, "--phase", "base", "--max-hours", str(HOURS["base"])],
        "variants": [*common, "--phase", "variants", "--max-hours", str(HOURS["variants"]),
                     "--base-records", f"/kaggle/working/{out}/base/shard_*/records.jsonl"],
    }
    if experiment == "e60":
        phases = {"variants": [*common, "--phase", "variants", "--max-hours", str(HOURS["variants"]),
                               "--base-records", "/tmp/aimo-e52/e59_base/records.jsonl"]}
    if experiment == "e62":
        phases = {"base": [*common, "--phase", "base", "--max-hours", str(E62["hours"])]}
    if experiment == "e62v":
        phases = {"variants": [*common, "--phase", "variants", "--max-hours", str(E62["hours"]),
                               "--base-records", "/tmp/aimo-e52/e62_base/records.jsonl"]}
    hours = {"base": E62["hours"], "variants": E62["hours"]} if experiment.startswith("e62") else HOURS
    source = HEADER.format(out=out, entries=json.dumps(entries)) + f'''cuda = runpy.run_path(str(root / "scripts/kaggle_cuda_setup.py"))["prepare_cuda_linker"]()
(out / "cuda-link-check.json").write_text(json.dumps(cuda, indent=2))
subprocess.run([sys.executable, "-m", "venv", "--without-pip", "/tmp/e52-env"], check=True)
py = "/tmp/e52-env/bin/python"
subprocess.run([sys.executable, "-m", "pip", "--python", py, "install", "--no-cache-dir", "-q", "vllm==0.20.0",
               "transformers=={TRANSFORMERS.get(model, "5.13.0")}", "huggingface-hub==1.22.0", "pandas==3.0.3", "pyarrow==24.0.0"], check=True)
freeze = subprocess.run([sys.executable, "-m", "pip", "--python", py, "freeze"], capture_output=True, text=True, check=True)
(out / "environment.txt").write_text(freeze.stdout)
guard = runpy.run_path(str(root / "scripts/kaggle_process_guard.py"))["run_guarded"]
if (root / "resume").is_dir():
    shutil.copytree(root / "resume", out, dirs_exist_ok=True)
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
