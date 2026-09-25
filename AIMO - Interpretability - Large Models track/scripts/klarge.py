"""Kaggle fleet launcher for the Large Models (Main) track.

Why a second launcher: `datagen/kaggle_launch.py` is built around the small-track registry
(T4x2, transformers pins, a small-track file bundle). Main-track jobs need H100s, Kaggle model
mounts (gpt-oss-120b), competition sources and vLLM. So a job here is one plain script in
`jobs/`, plus the input files it embeds, pushed as N shard kernels over the fleet.

    python scripts/klarge.py accounts                      # which tokens authenticate + quota
    python scripts/klarge.py push l0-h100-probe --accounts ISHRAQ,Eshad
    python scripts/klarge.py push l1-gptoss-traces --shards 6 --accounts ISHRAQ,Eshad,Nasif
    python scripts/klarge.py status [job]
    python scripts/klarge.py output <job>                  # -> runs/<job>/s<i>/

Tokens come from `Secrets.env` in this folder, or the file named by `AIMO_SECRETS`, and reach
the CLI only through the child environment; token-shaped strings are scrubbed from output.
Every push is recorded in `runs/launches.json` (job, shard, account, kernel ref).

Run: python scripts/klarge.py accounts
"""

from __future__ import annotations

import argparse
import base64
import hashlib
import io
import json
import os
import re
import subprocess
import sys
import threading
import time
import zipfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SECRETS = Path(os.environ.get("AIMO_SECRETS", ROOT / "Secrets.env"))
JOBS_DIR = ROOT / "jobs"
BUILD_DIR = ROOT / "kaggle_runs"
RUNS_DIR = ROOT / "runs"
LAUNCHES = RUNS_DIR / "launches.json"
_LAUNCHES_LOCK = threading.Lock()  # pushes run in parallel threads

AIMO3 = "ai-mathematical-olympiad-progress-prize-3"
GPT_OSS_120B = "danielhanchen/gpt-oss-120b/Transformers/default/1"  # what AIMO3 H100 notebooks mount
VLLM_UTILS = "andreasbis/aimo-3-utils"  # offline vLLM wheels used by the AIMO3 notebooks

PANEL_ITEMS = "data/panel/items.jsonl"  # scripts/build_items.py


def panel(model: str, efforts: list[str], parallel: int, num_ctx: int, temperature: float, top_p: float,
          hours: float = 11.0) -> dict:
    # `hours` stops new requests; in-flight ones (up to ~45 min at 32k) must still end inside
    # Kaggle's 12 h session. The first launch used 8.3 to end before the quota reset.
    """An Ollama panel job on T4x2: `model` samples every panel item under each effort."""
    config = dict(model=model, efforts=efforts, k=4, parallel=parallel, num_ctx=num_ctx,
                  num_predict=num_ctx - 1024, temperature=temperature, top_p=top_p, hours=hours)
    return dict(script="ollama_panel.py", accelerator="NvidiaTeslaT4", internet=True, config=config, items=PANEL_ITEMS)


# job -> kernel settings. `inputs` are repo-relative files embedded in the kernel; panel jobs
# embed `config.json` + `items.jsonl` instead. Sampling follows each model card's recommendation.
JOBS: dict[str, dict] = {
    # L0 showed `NvidiaH100` silently falls back to T4x2 unless the AIMO3 competition is attached.
    "l0-h100-probe": dict(
        script="probe.py", accelerator="NvidiaH100", internet=True,
        models=[GPT_OSS_120B], kernels=[VLLM_UTILS], inputs=[],
    ),
    "l1-gptoss20b-low": panel("gpt-oss:20b", ["low"], 8, 16384, 1.0, 1.0),
    "l1-gptoss20b-medium": panel("gpt-oss:20b", ["medium"], 8, 24576, 1.0, 1.0),
    "l1-gptoss20b-high": panel("gpt-oss:20b", ["high"], 8, 32768, 1.0, 1.0),
    "l1-dsr1-8b": panel("deepseek-r1:8b", ["think"], 8, 16384, 0.6, 0.95),
    "l1-qwen3-4b": panel("qwen3:4b", ["think"], 8, 16384, 0.6, 0.95),
    "l1-qwen3-14b": panel("qwen3:14b", ["think"], 8, 16384, 0.6, 0.95),
    "l1-qwen3-30b-a3b": panel("qwen3:30b-a3b", ["think"], 8, 16384, 0.6, 0.95),
    "l1-qwen3-32b": panel("qwen3:32b", ["think"], 3, 16384, 0.6, 0.95),
    # Smoke tests: the first L1 push died at warm-load (streamed NDJSON), so check a short run end to end.
    # MUBAMAN's pushes ran with no GPU and no internet (unverified account), so these reran on RSD.
    "l1-smoke-qwen3-4b": panel("qwen3:4b", ["think"], 8, 4096, 0.6, 0.95, hours=0.2),
    "l1-smoke-gptoss20b": panel("gpt-oss:20b", ["low", "high"], 8, 4096, 1.0, 1.0, hours=0.25),
}


def load_accounts() -> dict[str, tuple[str, str]]:
    """alias -> (username, token). Tolerates `KEY = value` spacing."""
    found: dict[str, list[str]] = {}
    for raw in SECRETS.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = (part.strip().strip('"').strip("'") for part in line.split("=", 1))
        for prefix, slot in (("KAGGLE_USERNAME_", 0), ("KAGGLE_API_TOKEN_", 1)):
            if key.startswith(prefix):
                found.setdefault(key[len(prefix):], ["", ""])[slot] = value
    return {alias: (user, token) for alias, (user, token) in found.items() if user and token}


def scrub(text: str) -> str:
    return re.sub(r"KGAT_[0-9A-Za-z]+", "KGAT_***", text)


def child_env(token: str) -> dict[str, str]:
    env = {k: v for k, v in os.environ.items() if k not in ("KAGGLE_USERNAME", "KAGGLE_KEY")}
    env.update(KAGGLE_API_TOKEN=token, PYTHONIOENCODING="utf-8", PYTHONUTF8="1")
    return env


def kaggle(alias: str, *args: str, timeout: int = 600) -> tuple[int, str]:
    _user, token = load_accounts()[alias]
    result = subprocess.run(
        ["kaggle", *args], capture_output=True, text=True, env=child_env(token),
        encoding="utf-8", errors="replace", timeout=timeout, check=False,
    )
    return result.returncode, scrub((result.stdout or "") + (result.stderr or "")).strip()


# `kaggle quota` crashes in CLI 2.2.2: the SDK cannot parse a duration without a fractional part.
QUOTA_CODE = r"""
import json
from datetime import timedelta
import kagglesdk.kaggle_object as ko
ko.TimeDeltaSerializer._from_dict_value = staticmethod(lambda v: timedelta(seconds=float(v.rstrip("s"))))
from kaggle.api.kaggle_api_extended import KaggleApi
api = KaggleApi(); api.authenticate(); r = api.quota_view()
hours = lambda q: None if q is None else [round(q.time_used.total_seconds() / 3600, 2), round(q.total_time_allowed.total_seconds() / 3600, 2)]
print("JSON" + json.dumps({"refresh": str(r.quota_refresh_time), "gpu": hours(r.gpu_quota), "tpu": hours(r.tpu_quota)}))
"""


def quota(alias: str) -> dict | str:
    _user, token = load_accounts()[alias]
    result = subprocess.run([sys.executable, "-c", QUOTA_CODE], capture_output=True, text=True,
                            env=child_env(token), timeout=120, check=False)
    match = re.search(r"JSON(.*)", result.stdout)
    return json.loads(match.group(1)) if match else scrub(result.stderr.strip()[-200:])


def payload_files(job: str, shard: int, shards: int) -> dict[str, bytes]:
    """Files the kernel unpacks: repo inputs, or a panel's config, its shard of the items, and the
    keys every downloaded session already completed (so a continuation session skips them)."""
    spec = JOBS[job]
    files = {relative: (ROOT / relative).read_bytes() for relative in spec.get("inputs", [])}
    if "items" in spec:
        lines = (ROOT / spec["items"]).read_text(encoding="utf-8").splitlines()
        files["items.jsonl"] = "".join(line + "\n" for i, line in enumerate(lines) if i % shards == shard).encode()
        files["config.json"] = json.dumps(spec["config"]).encode()
        done = {tuple(json.loads(line)[k] for k in ("item_id", "effort", "sample"))
                for path in RUNS_DIR.glob(f"{job}/s*/records*.jsonl")
                for line in path.read_text(encoding="utf-8").splitlines() if line.strip()}
        files["done.json"] = json.dumps(sorted(done)).encode()
    return files


def payload_zip(files: dict[str, bytes]) -> tuple[str, str]:
    """Deterministic base64 zip of the job's payload files, plus its SHA-1."""
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED, compresslevel=9) as bundle:
        for name in sorted(files):
            info = zipfile.ZipInfo(name, date_time=(2020, 1, 1, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            bundle.writestr(info, files[name])
    data = buffer.getvalue()
    return base64.b64encode(data).decode("ascii"), hashlib.sha1(data).hexdigest()


def build(job: str, shard: int, shards: int, alias: str) -> tuple[Path, str]:
    spec = JOBS[job]
    user, _ = load_accounts()[alias]
    slug = f"aimo-large-{job}" + (f"-s{shard}" if shards > 1 else "")
    b64, sha1 = payload_zip(payload_files(job, shard, shards))
    header = (
        f"# Auto-generated by scripts/klarge.py from jobs/{spec['script']}; edit the repo, not this kernel.\n"
        f"JOB = {json.dumps(dict(job=job, shard=shard, shards=shards, account=alias, payload_sha1=sha1))!r}\n"
        f"PAYLOAD_B64 = {b64!r}\n\n"
    )
    out = BUILD_DIR / job / f"s{shard}"
    out.mkdir(parents=True, exist_ok=True)
    (out / "run.py").write_text(header + (JOBS_DIR / spec["script"]).read_text(encoding="utf-8"), encoding="utf-8")
    metadata = {
        "id": f"{user}/{slug}", "title": slug,  # Kaggle derives the slug from the title
        "code_file": "run.py", "language": "python", "kernel_type": "script", "is_private": True,
        "enable_gpu": spec["accelerator"].startswith("Nvidia"), "enable_tpu": spec["accelerator"].startswith("Tpu"),
        "enable_internet": spec["internet"], "machine_shape": spec["accelerator"],
        "dataset_sources": spec.get("datasets", []), "competition_sources": spec.get("competitions", []),
        "kernel_sources": spec.get("kernels", []), "model_sources": spec.get("models", []),
    }
    (out / "kernel-metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return out, metadata["id"]


def record(entry: dict) -> None:
    with _LAUNCHES_LOCK:
        RUNS_DIR.mkdir(parents=True, exist_ok=True)
        rows = json.loads(LAUNCHES.read_text()) if LAUNCHES.exists() else []
        # A relaunch supersedes the old kernel: both would download into runs/<job>/s<shard>/.
        rows = [r for r in rows if (r["job"], r["shard"]) != (entry["job"], entry["shard"])] + [entry]
        LAUNCHES.write_text(json.dumps(rows, indent=1))


def push(job: str, shards: int, aliases: list[str]) -> None:
    def one(shard: int) -> str:
        alias = aliases[shard % len(aliases)]
        out, ref = build(job, shard, shards, alias)
        code, text = kaggle(alias, "kernels", "push", "-p", str(out), "--accelerator", JOBS[job]["accelerator"])
        # The CLI exits 0 on a refused push; only a version number means it was accepted.
        if code == 0 and "error" not in text.lower():
            record(dict(job=job, shard=shard, shards=shards, account=alias, ref=ref,
                        pushed=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())))
        return f"[{alias}] s{shard} {ref} exit={code}\n  " + text.replace("\n", "\n  ")
    with ThreadPoolExecutor(8) as pool:
        for line in pool.map(one, range(shards)):
            print(line)


def launched(job: str | None) -> list[dict]:
    rows = json.loads(LAUNCHES.read_text()) if LAUNCHES.exists() else []
    return [r for r in rows if job in (None, r["job"])]


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("command", choices=["accounts", "push", "status", "output", "progress"])
    parser.add_argument("job", nargs="?")
    parser.add_argument("--shards", type=int, default=1)
    parser.add_argument("--accounts", help="comma-separated aliases; default: every account")
    args = parser.parse_args()
    aliases = args.accounts.split(",") if args.accounts else list(load_accounts())

    if args.command == "accounts":
        with ThreadPoolExecutor(8) as pool:
            for alias, result in zip(aliases, pool.map(quota, aliases)):
                print(f"{alias:10s} {load_accounts()[alias][0]:22s} {result}")
    elif args.command == "push":
        push(args.job, max(args.shards, 1), aliases)
    elif args.command == "status":
        rows = launched(args.job)
        with ThreadPoolExecutor(8) as pool:
            states = pool.map(lambda r: kaggle(r["account"], "kernels", "status", r["ref"])[1], rows)
            for row, state in zip(rows, states):
                print(f"{row['job']:24s} s{row['shard']:<2d} {row['account']:9s} {state.splitlines()[-1] if state else ''}")
    elif args.command == "progress":  # downloaded completions vs the panel's total, per job
        items = [json.loads(line) for line in (ROOT / PANEL_ITEMS).read_text(encoding="utf-8").splitlines()]
        panels = {j for j, s in JOBS.items() if "items" in s and "smoke" not in j}  # smoke runs are never continued
        for job in sorted({r["job"] for r in launched(args.job)} & panels):
            config = JOBS[job]["config"]
            total = len(config["efforts"]) * sum(item.get("k", config["k"]) for item in items)
            done = len(json.loads(payload_files(job, 0, 1)["done.json"]))
            print(f"{job:24s} {done:5d}/{total:<5d} {'DONE' if done >= total else 'UNFINISHED'}")
    elif args.command == "output":
        def fetch(row: dict) -> str:
            dest = RUNS_DIR / row["job"] / f"s{row['shard']}"
            dest.mkdir(parents=True, exist_ok=True)
            code, text = kaggle(row["account"], "kernels", "output", row["ref"], "-p", str(dest), "-o",
                                "--page-size", "200", timeout=1800)
            return f"[{row['account']}] {row['ref']} exit={code} -> {dest}\n{text[-600:]}"
        with ThreadPoolExecutor(8) as pool:
            for line in pool.map(fetch, launched(args.job)):
                print(line)


if __name__ == "__main__":
    main()
