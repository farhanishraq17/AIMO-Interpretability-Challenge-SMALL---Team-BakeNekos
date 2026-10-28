"""E62 amendment 4: E62's contradictions written and checked on Kaggle by two open-weight models, not OpenRouter.

OpenRouter's free tier gave about 50 requests a day, so generation had covered 88 of 300 eligible problems in a week.
Each kernel here serves two models through vLLM's OpenAI-compatible server, one per T4, and runs
`datagen/perturb/premise_variants.py` against them unchanged except for its endpoints:

* generator, GPU 0: `Qwen/Qwen3-8B-AWQ` (thinking on, temperature 1.0 as before), amendment 4's fallback: the first
  smoke showed `Qwen/Qwen3-14B-AWQ` leaves 1.68 GiB of KV cache on a T4, short of the 2.5 GiB one 16k sequence needs.
  vLLM picks its AWQ kernel (awq_marlin runs on a T4);
* checker, GPU 1: `deepseek-ai/DeepSeek-R1-Distill-Llama-8B`, 4-bit bitsandbytes so it fits a T4, a Llama-based model
  from another family, as E60 required (temperature 0.6, its model card's setting: greedy decoding loops).

Both reasoning parsers keep the thinking out of the text the script parses; it is still logged for the audit. vLLM
0.20.0 runs in its own venv after the CUDA linker check, E52's recipe. `WORKERS` generator processes per kernel take
disjoint slices of the kernel's share of `data/e62/eligible.parquet`, so the servers batch them; every process saves
one record per finished problem. `smoke` runs two problems first and its output is discarded.

`python scripts/e62_gen_bundle.py merge` folds the downloaded kernels into `data/e62/` (the OpenRouter output moves to
`data/e62/superseded/openrouter/` first) and rewrites `variants.parquet`, the input of the `e62v-qwen-*` kernels.

Run via:  python scripts/kaggle_followup.py push e62g-a --account <ALIAS>
"""

from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "datagen" / "perturb"))
from e52_gpu_bundle import HEADER, payload  # noqa: E402

GENERATOR, CHECKER = "Qwen/Qwen3-8B-AWQ", "deepseek-ai/DeepSeek-R1-Distill-Llama-8B"
SERVERS = [  # (GPU, port, vLLM server arguments)
    (0, 8001, ["--model", GENERATOR, "--reasoning-parser", "qwen3"]),
    (1, 8002, ["--model", CHECKER, "--quantization", "bitsandbytes", "--reasoning-parser", "deepseek_r1"]),
]
ENGINE = ["--dtype", "float16", "--max-model-len", "16384", "--gpu-memory-utilization", "0.90", "--max-num-seqs", "4"]
MAX_TOKENS, CHECKER_TEMPERATURE = 12000, 0.6  # completion budget inside the 16k context
KERNELS, WORKERS, HOURS = 10, 4, 9.0
PARTS = "abcdefghij"
JOBS = ["smoke", *PARTS[:KERNELS]]
FILES = ("datagen/perturb/premise_variants.py", "datagen/perturb/transforms.py", "data/e62/eligible.parquet",
         "protocols/E62_PROTOCOL.md", "scripts/kaggle_cuda_setup.py", "scripts/kaggle_process_guard.py")
DOWNLOADED = "runs/e62g-[a-j]-*/e62g-*/w*/e62/raw"


def render(job: str) -> tuple[str, dict]:
    entries, hashes = payload(FILES)
    out = f"e62g-{job}"
    # Disjoint slices: kernel k's worker w takes every (KERNELS * WORKERS)-th problem from k + KERNELS * w.
    shards, workers, hours = ((150, [0], 2.0) if job == "smoke"
                              else (KERNELS * WORKERS, [PARTS.index(job) + KERNELS * w for w in range(WORKERS)], HOURS))
    argv = [["datagen/perturb/premise_variants.py", "--pool", "data/e62/eligible.parquet",
             "--out-dir", f"/kaggle/working/{out}/w{i}/e62", "--generator", GENERATOR, "--checker", CHECKER,
             "--generator-url", "http://127.0.0.1:8001/v1", "--checker-url", "http://127.0.0.1:8002/v1",
             "--checker-temperature", str(CHECKER_TEMPERATURE), "--max-tokens", str(MAX_TOKENS),
             "--num-shards", str(shards), "--shard-index", str(i)] for i in workers]
    source = HEADER.format(out=out, entries=json.dumps(entries)) + f'''import urllib.request
cuda = runpy.run_path(str(root / "scripts/kaggle_cuda_setup.py"))["prepare_cuda_linker"]()
(out / "cuda-link-check.json").write_text(json.dumps(cuda, indent=2))
subprocess.run([sys.executable, "-m", "venv", "--without-pip", "/tmp/e52-env"], check=True)
py = "/tmp/e52-env/bin/python"
subprocess.run([sys.executable, "-m", "pip", "--python", py, "install", "--no-cache-dir", "-q", "vllm==0.20.0",
               "transformers==5.13.0", "huggingface-hub==1.22.0", "pandas==3.0.3", "pyarrow==24.0.0",
               "bitsandbytes>=0.46.1"], check=True)
freeze = subprocess.run([sys.executable, "-m", "pip", "--python", py, "freeze"], capture_output=True, text=True, check=True)
(out / "environment.txt").write_text(freeze.stdout)
servers = []
for gpu, port, args in {SERVERS!r}:
    log = open(out / f"server-{{gpu}}.log", "w")
    servers.append(subprocess.Popen([py, "-m", "vllm.entrypoints.openai.api_server", *args, *{ENGINE!r},
                                     "--port", str(port)], env={{**os.environ, "CUDA_VISIBLE_DEVICES": str(gpu)}},
                                    stdout=log, stderr=subprocess.STDOUT))
for (gpu, port, _), server in zip({SERVERS!r}, servers):
    deadline = time.time() + 2400
    while True:
        try:
            if urllib.request.urlopen(f"http://127.0.0.1:{{port}}/health", timeout=5).status == 200:
                break
        except OSError:
            pass
        if server.poll() is not None or time.time() > deadline:
            raise SystemExit(f"server on GPU {{gpu}} did not come up; see server-{{gpu}}.log")
        time.sleep(10)
    print(f"server on GPU {{gpu}} ready after {{time.time() - started:.0f}} s", flush=True)
guard = runpy.run_path(str(root / "scripts/kaggle_process_guard.py"))["run_guarded"]
launcher = ("import subprocess, sys\\n"
            f"procs = [subprocess.Popen([sys.argv[1], *argv]) for argv in {argv!r}]\\n"
            "sys.exit(max(p.wait() for p in procs))")
result = guard([py, "-c", launcher, py], root, out / "generate.log", timeout={hours * 3600})
for server in servers:
    server.terminate()
(out / "execution.json").write_text(json.dumps({{**result, "end_to_end_seconds": time.time() - started}}, indent=2))
raise SystemExit(result["exit_code"])
'''
    compile(source, "run.py", "exec")
    return source, hashes


def merge() -> None:
    """Every downloaded kernel's records into `data/e62/`, the OpenRouter output set aside first."""
    from premise_variants import write_outputs

    e62 = ROOT / "data" / "e62"
    old = e62 / "superseded" / "openrouter"
    if not old.exists():
        old.mkdir(parents=True)
        for name in ("raw", "variants.parquet", "audit.parquet", "generation.log"):
            if (e62 / name).exists():
                shutil.move(str(e62 / name), str(old / name))
    (e62 / "raw").mkdir(exist_ok=True)
    progress = {}
    for name in ("progress", "generation", "check"):
        lines = [line for raw in sorted(ROOT.glob(DOWNLOADED)) for line in
                 (raw / f"{name}.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()]
        if name == "progress":
            progress = {json.loads(line)["pool_id"]: json.loads(line) for line in lines}
            lines = [json.dumps(p) for p in progress.values()]
        (e62 / "raw" / f"{name}.jsonl").write_text("".join(f"{line}\n" for line in lines), encoding="utf-8")
    write_outputs(list(progress.values()), e62)
    print(f"{len(progress)} problems from {len(list(ROOT.glob(DOWNLOADED)))} worker records -> {e62}")


if __name__ == "__main__" and sys.argv[1:] == ["merge"]:
    merge()
