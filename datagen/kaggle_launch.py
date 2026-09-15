"""Run pipeline stages as self-contained Kaggle kernels across the six-account fleet.

Each entry in `EXPERIMENTS` becomes one private GPU script kernel on its assigned account.
The kernel embeds a zip of exactly the repo files it needs (base64, well under 1 MB), so
there is no dataset upload, no guessing at mount paths, and every kernel version is a
frozen copy of the code (and any upstream outputs) that produced its result.

    python datagen/kaggle_launch.py list
    python datagen/kaggle_launch.py build e1-stagea-deepseek     # render only, no network
    python datagen/kaggle_launch.py push e1-stagea-deepseek
    python datagen/kaggle_launch.py status                       # every experiment
    python datagen/kaggle_launch.py logs e1-stagea-deepseek      # log tail of a finished run
    python datagen/kaggle_launch.py output e1-stagea-deepseek    # -> runs/<experiment>/

Tokens are read from Secrets.env and reach the Kaggle CLI only through the child process
environment; anything token-shaped is scrubbed from CLI output before printing.
"""

from __future__ import annotations

import argparse
import base64
import glob
import hashlib
import io
import json
import os
import pickle
import re
import subprocess
import zipfile
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SECRETS = ROOT / "Secrets.env"
BUILD_DIR = ROOT / "kaggle_runs"
RUNS_DIR = ROOT / "runs"
PROBE_ARTIFACT = (
    ROOT / "getting-started" / "solutions" / "trained-probe" / "probe_artifacts" / "probe_artifact.pkl"
)

DEEPSEEK = "deepseek-ai/DeepSeek-R1-0528-Qwen3-8B"
OLMO = "allenai/Olmo-3-7B-Instruct"

# Written into the bundle at build time from the probe artifact, the single source of truth.
SYSTEM_PROMPT_PATH = "data/probe_system_prompt.txt"
KAGGLE_FEATURES = "/kaggle/working/features"

# Library pins from Dockerfile.competition. torch stays as Kaggle ships it: the
# competition's CUDA 13.2 build targets Blackwell, and the T4 needs Kaggle's own wheel.
PIP_PINS = [
    "transformers==5.13.0",
    "accelerate==1.13.0",
    "huggingface-hub==1.22.0",
    "tokenizers==0.22.2",
    "safetensors==0.8.0",
]

BUNDLE_FILES = [
    "datagen/extract_features.py",
    "datagen/extract_generation_features.py",
    "datagen/extract_trace_states.py",
    "datagen/merge_features.py",
    "data/small_track/instances.parquet",
    "data/small_track/instances_train.parquet",
    "data/small_track/splits.json",
    "data/small_track/summary.json",
]
BUNDLE_DIRS = ["getting-started/solutions/uncertainty-profiling/uncertainty_profile"]

# Kaggle rejects very large script sources; past this, ship inputs as a private dataset instead.
BUNDLE_WARN_BYTES = 900_000

ACCOUNTS = ["ISHRAQ", "DIDHITI", "TASHIN", "SALAM", "ISHMAM1", "DESIKALAKAR"]

# Stage B traces used downstream (E4). fp16 only: the dtype benchmark showed bf16 and fp16
# greedy traces diverge, so the two sets must never be mixed.
E2_TRACES = "runs/e2f-stageb-fp16-s*/features/stageb-fp16-deepseek/shard_*/records.jsonl"


@dataclass(frozen=True)
class Experiment:
    account: str
    purpose: str
    steps: tuple[tuple[str, ...], ...]  # argv after `python`, run from the bundle root
    accelerator: str = "NvidiaTeslaT4"  # T4x2: 8B in half precision needs ~16.4 GB, more than one P100
    bundle_globs: tuple[str, ...] = ()  # extra repo files (e.g. upstream run outputs) to embed


def stage_a(model_id: str, out_name: str, *extra: str) -> tuple[str, ...]:
    return (
        "datagen/extract_features.py",
        "--model-id", model_id,
        "--system-prompt-file", SYSTEM_PROMPT_PATH,
        "--out-dir", f"{KAGGLE_FEATURES}/{out_name}",
        *extra,
    )


def stage_b(model_id: str, out_name: str, *extra: str) -> tuple[str, ...]:
    return (
        "datagen/extract_generation_features.py",
        "--model-id", model_id,
        "--out-dir", f"{KAGGLE_FEATURES}/{out_name}",
        *extra,
    )


def trace_states(model_id: str, traces_glob: str, out_name: str, *extra: str) -> tuple[str, ...]:
    return (
        "datagen/extract_trace_states.py",
        "--model-id", model_id,
        "--traces", traces_glob,
        "--out-dir", f"{KAGGLE_FEATURES}/{out_name}",
        *extra,
    )


def merge(features_dir: str) -> tuple[str, ...]:
    # Merging on Kaggle leaves one features.npz to download instead of one .npz per
    # instance, which `kaggle kernels output` would otherwise have to page through.
    return ("datagen/merge_features.py", "--features-dir", features_dir)


EXPERIMENTS: dict[str, Experiment] = {
    "e1-stagea-deepseek": Experiment(
        account="ISHMAM1",
        purpose="Stage A on DeepSeek-R1-0528-Qwen3-8B: pooled hidden states (last/mean/max) at "
        "every layer plus next-token uncertainty for the 138 train instances.",
        steps=(
            stage_a(DEEPSEEK, "stagea-deepseek"),
            merge(f"{KAGGLE_FEATURES}/stagea-deepseek"),
        ),
    ),
    # Superseded by e2f-*: emulated bf16 on T4 ran at ~617 s/instance and shard 1 hit CUDA OOM.
    # Kept so their kernels stay addressable; shard 0's output is the bf16 drift reference.
    "e2-stageb-deepseek-s0": Experiment(
        account="ISHRAQ",
        purpose="Stage B shard 0/2 on DeepSeek-R1-0528-Qwen3-8B: greedy generation (4096 cap) "
        "and the 14 trace-uncertainty features, plus generated text.",
        steps=(stage_b(DEEPSEEK, "stageb-deepseek", "--num-shards", "2", "--shard-index", "0"),),
    ),
    "e2-stageb-deepseek-s1": Experiment(
        account="DIDHITI",
        purpose="Stage B shard 1/2 on DeepSeek-R1-0528-Qwen3-8B (see shard 0).",
        steps=(stage_b(DEEPSEEK, "stageb-deepseek", "--num-shards", "2", "--shard-index", "1"),),
    ),
    "e2-bench-dtype": Experiment(
        account="TASHIN",
        purpose="Throughput benchmark after E2 ran at ~617 s/instance: the same 8 problems, 512-token "
        "greedy generation at batch size 8, first in emulated bf16 (the starter-kit default), then fp16.",
        steps=(
            # bf16 first, so its timing survives even if fp16 fails the non-finite preflight.
            stage_b(DEEPSEEK, "bench-bf16", "--limit", "8", "--max-new-tokens", "512"),
            stage_b(DEEPSEEK, "bench-fp16", "--limit", "8", "--max-new-tokens", "512", "--dtype", "float16"),
        ),
    ),
    "e2f-stageb-fp16-s0": Experiment(
        account="DESIKALAKAR",
        purpose="Stage B rerun in native fp16, shard 0/2: 3.3x the throughput of emulated bf16 on T4 "
        "(e2-bench-dtype), batch size 4 after the OOM at 8.",
        steps=(
            stage_b(DEEPSEEK, "stageb-fp16-deepseek", "--num-shards", "2", "--shard-index", "0",
                    "--batch-size", "4", "--dtype", "float16"),
        ),
    ),
    "e2f-stageb-fp16-s1": Experiment(
        account="DIDHITI",
        purpose="Stage B rerun in native fp16, shard 1/2 (see shard 0).",
        steps=(
            stage_b(DEEPSEEK, "stageb-fp16-deepseek", "--num-shards", "2", "--shard-index", "1",
                    "--batch-size", "4", "--dtype", "float16"),
        ),
    ),
    "e3-stagea-olmo-smoke": Experiment(
        account="TASHIN",
        purpose="Olmo-3-7B-Instruct plumbing check: loads under transformers 5.13 on T4x2, layer "
        "count, activation finiteness. Labels are NOT Olmo's (--no-filter-model).",
        steps=(
            stage_a(OLMO, "stagea-olmo-smoke", "--no-filter-model"),
            merge(f"{KAGGLE_FEATURES}/stagea-olmo-smoke"),
        ),
    ),
    "e4-tracestate-deepseek": Experiment(
        account="SALAM",
        purpose="E4 on DeepSeek-R1-0528-Qwen3-8B: teacher-forced forward pass over prompt + the fp16 E2 "
        "greedy trace; hidden states at every layer pooled over prompt_last, trace_mean, trace "
        "thirds, and trace_last. Needs the e2f outputs downloaded to runs/ first.",
        steps=(
            trace_states(DEEPSEEK, E2_TRACES, "tracestate-deepseek", "--dtype", "float16"),
            merge(f"{KAGGLE_FEATURES}/tracestate-deepseek"),
        ),
        bundle_globs=(E2_TRACES,),
    ),
}

KERNEL_TEMPLATE = r'''"""AIMO Interpretability, small-model track: __NAME__

Auto-generated by datagen/kaggle_launch.py. Edit the repo and re-push, not this kernel.

__PURPOSE__
"""

import base64
import io
import json
import os
import subprocess
import sys
import time
import zipfile
from pathlib import Path

EXPERIMENT = json.loads(__EXPERIMENT_JSON__)
BUNDLE_SHA1 = "__BUNDLE_SHA1__"
BUNDLE_B64 = "__BUNDLE_B64__"

WORK = Path("/kaggle/working")
REPO = Path("/tmp/aimo-repo")  # code stays out of the saved output
LOGS = WORK / "logs"

VERSIONS = (
    "import platform, torch, transformers, accelerate;"
    "print('python', platform.python_version());"
    "print('torch', torch.__version__, 'cuda', torch.version.cuda);"
    "print('transformers', transformers.__version__, 'accelerate', accelerate.__version__);"
    "print('gpus', [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]);"
    "print('bf16', torch.cuda.is_available() and torch.cuda.is_bf16_supported())"
)


def run(argv, log_name, **kwargs):
    LOGS.mkdir(parents=True, exist_ok=True)
    print("\n$ " + " ".join(argv), flush=True)
    with (LOGS / log_name).open("a", encoding="utf-8") as log:
        proc = subprocess.Popen(
            argv, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1, **kwargs
        )
        for line in proc.stdout:
            sys.stdout.write(line)
            log.write(line)
        return proc.wait()


def main():
    manifest = {
        "experiment": EXPERIMENT["name"],
        "account": EXPERIMENT["account"],
        "bundle_sha1": BUNDLE_SHA1,
        "steps": [],
    }
    started = time.time()
    try:
        run(["nvidia-smi"], "env.log")
        code = run([sys.executable, "-m", "pip", "install", "-q", *EXPERIMENT["pip"]], "pip.log")
        if code:
            raise SystemExit(f"pip install failed with exit code {code}")
        with zipfile.ZipFile(io.BytesIO(base64.b64decode(BUNDLE_B64))) as bundle:
            bundle.extractall(REPO)
        # Model weights go to /tmp: /kaggle/working is capped at 20 GB and saved as output.
        # Expandable segments: E2 shard 1 died of fragmentation (567 MB reserved but unallocated).
        env = dict(
            os.environ,
            HF_HOME="/tmp/hf",
            TOKENIZERS_PARALLELISM="false",
            PYTHONUNBUFFERED="1",
            PYTORCH_ALLOC_CONF="expandable_segments:True",
            PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True",
        )
        run([sys.executable, "-c", VERSIONS], "env.log", env=env)
        for index, argv in enumerate(EXPERIMENT["steps"]):
            step_started = time.time()
            code = run([sys.executable, *argv], f"step{index}.log", env=env, cwd=REPO)
            manifest["steps"].append(
                {"argv": argv, "returncode": code, "seconds": round(time.time() - step_started, 1)}
            )
            if code:
                raise SystemExit(f"step {index} failed with exit code {code}")
    finally:
        manifest["seconds"] = round(time.time() - started, 1)
        WORK.mkdir(parents=True, exist_ok=True)
        (WORK / "run_manifest.json").write_text(json.dumps(manifest, indent=2))


main()
'''


def load_accounts() -> dict[str, tuple[str, str]]:
    """alias -> (username, token), parsed from Secrets.env."""
    usernames: dict[str, str] = {}
    tokens: dict[str, str] = {}
    for raw in SECRETS.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        name, value = (part.strip().strip('"').strip("'") for part in line.split("=", 1))
        if name.startswith("KAGGLE_USERNAME_"):
            usernames[name.removeprefix("KAGGLE_USERNAME_")] = value
        elif name.startswith("KAGGLE_API_TOKEN_"):
            tokens[name.removeprefix("KAGGLE_API_TOKEN_")] = value
    return {alias: (usernames.get(alias, ""), token) for alias, token in tokens.items()}


def kaggle(alias: str, *args: str) -> tuple[int, str]:
    """Run the Kaggle CLI as `alias`, returning (exit code, token-scrubbed output)."""
    _username, token = load_accounts()[alias]
    env = {k: v for k, v in os.environ.items() if k not in ("KAGGLE_USERNAME", "KAGGLE_KEY")}
    env["KAGGLE_API_TOKEN"] = token
    # The CLI echoes kernel logs (tqdm bars, Unicode) after downloads; a cp1252 Windows
    # console makes it exit 1 even though every file arrived.
    env["PYTHONIOENCODING"] = "utf-8"
    env["PYTHONUTF8"] = "1"
    result = subprocess.run(
        ["kaggle", *args], capture_output=True, text=True, env=env, check=False, encoding="utf-8"
    )
    text = (result.stdout or "") + (result.stderr or "")
    return result.returncode, re.sub(r"KGAT_[0-9A-Za-z]+", "KGAT_***", text).strip()


def build_bundle(extra_globs: tuple[str, ...] = ()) -> tuple[bytes, str]:
    """Deterministic zip of the files a kernel needs, plus its SHA-1."""
    entries: dict[str, bytes] = {}
    for relative in BUNDLE_FILES:
        path = ROOT / relative
        if not path.is_file():
            raise SystemExit(f"bundle file missing: {relative} (run datagen/ingest_public.py first?)")
        entries[relative] = path.read_bytes()
    for relative in BUNDLE_DIRS:
        for path in sorted((ROOT / relative).rglob("*.py")):
            entries[path.relative_to(ROOT).as_posix()] = path.read_bytes()
    for pattern in extra_globs:
        matches = sorted(glob.glob(str(ROOT / pattern)))
        if not matches:
            raise SystemExit(f"no files match {pattern}; download the upstream run first")
        for match in matches:
            path = Path(match)
            entries[path.relative_to(ROOT).as_posix()] = path.read_bytes()
    with PROBE_ARTIFACT.open("rb") as handle:
        entries[SYSTEM_PROMPT_PATH] = str(pickle.load(handle)["system_prompt"]).encode("utf-8")

    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9) as bundle:
        for name in sorted(entries):
            info = zipfile.ZipInfo(name, date_time=(2020, 1, 1, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            bundle.writestr(info, entries[name])
    data = buffer.getvalue()
    return data, hashlib.sha1(data).hexdigest()


def kernel_ref(name: str) -> str:
    username, _ = load_accounts()[EXPERIMENTS[name].account]
    return f"{username}/aimo-small-{name}"


def build(name: str) -> Path:
    experiment = EXPERIMENTS[name]
    bundle, sha1 = build_bundle(experiment.bundle_globs)
    payload = {
        "name": name,
        "account": experiment.account,
        "pip": PIP_PINS,
        "steps": [list(step) for step in experiment.steps],
    }
    script = (
        KERNEL_TEMPLATE.replace("__NAME__", name)
        .replace("__PURPOSE__", experiment.purpose)
        .replace("__EXPERIMENT_JSON__", json.dumps(json.dumps(payload)))
        .replace("__BUNDLE_SHA1__", sha1)
        .replace("__BUNDLE_B64__", base64.b64encode(bundle).decode("ascii"))
    )
    ref = kernel_ref(name)
    metadata = {
        "id": ref,
        "title": ref.split("/", 1)[1],
        "code_file": "run.py",
        "language": "python",
        "kernel_type": "script",
        "is_private": True,
        "enable_gpu": True,
        "enable_internet": True,
        "machine_shape": experiment.accelerator,
        "dataset_sources": [],
        "competition_sources": [],
        "kernel_sources": [],
        "model_sources": [],
    }
    out = BUILD_DIR / name
    out.mkdir(parents=True, exist_ok=True)
    (out / "run.py").write_text(script, encoding="utf-8")
    (out / "kernel-metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(
        f"built {out}  kernel={ref}  bundle={len(bundle) / 1024:.0f} KB "
        f"script={len(script) / 1024:.0f} KB sha1={sha1[:12]}"
    )
    if len(script) > BUNDLE_WARN_BYTES:
        print(f"  warning: script is {len(script) / 1e6:.2f} MB; Kaggle may reject it. Ship inputs as a private dataset.")
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("list")
    for command in ("build", "push", "output"):
        sub.add_parser(command).add_argument("experiments", nargs="+", choices=sorted(EXPERIMENTS))
    for command in ("status", "logs"):
        sub.add_parser(command).add_argument("experiments", nargs="*", choices=sorted(EXPERIMENTS))
    args = parser.parse_args()

    if args.command == "list":
        for name, experiment in EXPERIMENTS.items():
            print(f"{name:26s} {experiment.account:12s} {experiment.purpose}")
        return

    for name in args.experiments or list(EXPERIMENTS):
        experiment = EXPERIMENTS[name]
        if args.command == "build":
            build(name)
        elif args.command == "push":
            out = build(name)
            code, text = kaggle(
                experiment.account, "kernels", "push", "-p", str(out), "--accelerator", experiment.accelerator
            )
            print(f"[{experiment.account}] push exit={code}\n{text}\n")
        elif args.command == "status":
            code, text = kaggle(experiment.account, "kernels", "status", kernel_ref(name))
            print(f"{name:26s} [{experiment.account}] {text}")
        elif args.command == "logs":
            code, text = kaggle(experiment.account, "kernels", "logs", kernel_ref(name))
            tail = [line for line in text.splitlines() if line.strip()][-40:]
            print(f"===== {name} [{experiment.account}] exit={code}\n" + "\n".join(tail) + "\n")
        elif args.command == "output":
            destination = RUNS_DIR / name
            destination.mkdir(parents=True, exist_ok=True)
            # The CLI fetches a single page per call. Per-instance vectors are redundant
            # once merged on Kaggle, so skip them and ask for the maximum page size.
            code, text = kaggle(
                experiment.account, "kernels", "output", kernel_ref(name),
                "-p", str(destination), "-o",
                "--page-size", "200", "--file-pattern", r"^(?!.*/vectors/)",
            )
            print(f"[{experiment.account}] output exit={code} -> {destination}\n{text[-2000:]}\n")


if __name__ == "__main__":
    main()
