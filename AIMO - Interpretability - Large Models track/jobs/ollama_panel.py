"""Sample Main-Track problems with an Ollama-served model on a Kaggle T4x2 (L1/L2 panels).

Why Ollama and not vLLM: vLLM cannot run gpt-oss on Turing (its MXFP4 kernels need sm80+),
and transformers dequantises gpt-oss-20b to ~42 GB of bf16, more than two T4s hold. Ollama
ships prebuilt CUDA runners for sm75 and serves every table model that fits T4x2
(gpt-oss:20b, qwen3:14b / 30b-a3b / 32b, ...) with batched decoding, so one code path covers
the whole panel. Weights are Ollama's quantised builds (gpt-oss: native MXFP4; qwen3: Q4_K_M):
record that next to every number.

The payload holds `items.jsonl` (item_id, problem_id, family, text). `CONFIG` in the payload
sets the model, reasoning efforts, samples per item and budgets. Every completion is appended
to /kaggle/working/records.jsonl as soon as it returns, so a timeout loses nothing already
sampled; nothing is scored here (answers are extracted and graded offline).
"""
import base64
import io
import json
import os
import subprocess
import sys
import threading
import time
import urllib.request
import zipfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

STARTED = time.time()
WORK = Path("/kaggle/working")
WORK.mkdir(parents=True, exist_ok=True)
with zipfile.ZipFile(io.BytesIO(base64.b64decode(PAYLOAD_B64))) as bundle:  # noqa: F821 (header)
    CONFIG = json.loads(bundle.read("config.json"))
    ITEMS = [json.loads(line) for line in bundle.read("items.jsonl").decode().splitlines() if line.strip()]
    # Keys finished by earlier sessions (downloaded, then embedded by klarge.py).
    DONE = {tuple(key) for key in json.loads(bundle.read("done.json"))} if "done.json" in bundle.namelist() else set()
JOB = json.loads(JOB)  # noqa: F821 (header)
# One file per session, so downloading a continuation never overwrites an earlier session's records.
RECORDS = WORK / f"records-{JOB['payload_sha1'][:8]}.jsonl"
HOST = "http://127.0.0.1:11434"
SUFFIX = "\n\nPlease reason step by step, and put your final answer within \\boxed{}."
DEADLINE = STARTED + 3600 * CONFIG["hours"]


def sh(cmd, check=False):
    print(f"$ {cmd}", flush=True)
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    print((result.stdout + result.stderr)[-3000:], flush=True)
    if check and result.returncode:
        raise SystemExit(f"failed ({result.returncode}): {cmd}")
    return result.stdout


def api(path, body=None, timeout=7200):
    data = None if body is None else json.dumps(body).encode()
    request = urllib.request.Request(HOST + path, data=data, headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.loads(response.read())


def start_ollama():
    # An unverified account's push is accepted but runs on CPU without internet: stop here.
    sh("nvidia-smi --query-gpu=name,memory.total --format=csv,noheader", check=True)
    sh("(apt-get -qq update && apt-get -qq install -y zstd) >/dev/null 2>&1; curl -fsSL https://ollama.com/install.sh | sh", check=True)
    env = dict(
        os.environ,
        OLLAMA_MODELS="/tmp/ollama",  # weights stay out of the saved output
        OLLAMA_NUM_PARALLEL=str(CONFIG["parallel"]),
        OLLAMA_CONTEXT_LENGTH=str(CONFIG["num_ctx"]),
        OLLAMA_FLASH_ATTENTION="1",
        OLLAMA_KV_CACHE_TYPE="q8_0",  # halves KV memory so more sequences decode together
        OLLAMA_MAX_LOADED_MODELS="1",
        OLLAMA_KEEP_ALIVE="-1",
    )
    subprocess.Popen(["ollama", "serve"], env=env, stdout=open(WORK / "ollama_serve.log", "w"), stderr=subprocess.STDOUT)
    for _ in range(120):
        try:
            print("ollama", api("/api/version", timeout=5), flush=True)
            break
        except Exception:  # noqa: BLE001 - server still starting
            time.sleep(2)
    sh(f"ollama pull {CONFIG['model']}", check=True)
    # Warm-load with the real context size so `ollama ps` shows the GPU/CPU split we will run with.
    api("/api/generate", {"model": CONFIG["model"], "prompt": "1+1=", "stream": False,  # default streams NDJSON
                          "options": {"num_ctx": CONFIG["num_ctx"], "num_predict": 1}})
    return sh("ollama ps") + sh("nvidia-smi --query-gpu=memory.used,memory.total --format=csv,noheader")


def work_list():
    done = set(DONE)
    if RECORDS.exists():
        for line in RECORDS.read_text().splitlines():
            record = json.loads(line)
            done.add((record["item_id"], record["effort"], record["sample"]))
    todo = []
    # Sample-major order: every item gets sample 0 before any gets sample 1, so a timeout
    # leaves full coverage at lower k rather than full k on a subset.
    for sample in range(max(item.get("k", CONFIG["k"]) for item in ITEMS)):
        for effort in CONFIG["efforts"]:
            for item in ITEMS:
                if sample < item.get("k", CONFIG["k"]) and (item["item_id"], effort, sample) not in done:
                    todo.append((item, effort, sample))
    return todo


LOCK = threading.Lock()


def solve(task):
    item, effort, sample = task
    if time.time() > DEADLINE:
        return 0
    think = effort if effort in ("low", "medium", "high") else (effort == "think")
    body = {
        "model": CONFIG["model"],
        "messages": [{"role": "user", "content": item["text"] + SUFFIX}],
        "think": think,
        "stream": False,
        "options": {
            "temperature": CONFIG["temperature"], "top_p": CONFIG["top_p"], "seed": 1000 * sample + 7,
            "num_ctx": CONFIG["num_ctx"], "num_predict": CONFIG["num_predict"],
        },
    }
    started = time.time()
    try:
        reply = api("/api/chat", body)
        error = None
    except Exception as exc:  # noqa: BLE001 - record and move on
        reply, error = {}, repr(exc)
    message = reply.get("message", {})
    record = {
        **{k: item[k] for k in ("item_id", "problem_id", "family")},
        "model": CONFIG["model"], "effort": effort, "sample": sample,
        "thinking": message.get("thinking", ""), "content": message.get("content", ""),
        "done_reason": reply.get("done_reason"), "eval_count": reply.get("eval_count"),
        "prompt_eval_count": reply.get("prompt_eval_count"),
        "seconds": round(time.time() - started, 1), "error": error,
    }
    with LOCK, RECORDS.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record) + "\n")
    return reply.get("eval_count") or 0


def main():
    placement = start_ollama()
    todo = work_list()
    print(f"{len(todo)} completions to run; deadline in {(DEADLINE - time.time()) / 3600:.2f} h", flush=True)
    tokens, t0 = 0, time.time()
    with ThreadPoolExecutor(CONFIG["parallel"]) as pool:
        for index, count in enumerate(pool.map(solve, todo), 1):
            tokens += count
            if index % 20 == 0:
                print(f"{index}/{len(todo)} done, {tokens / (time.time() - t0):.1f} tok/s", flush=True)
    summary = {
        "job": JOB, "config": CONFIG, "placement": placement, "items": len(ITEMS),
        "completions_this_session": len(todo), "generated_tokens": tokens,
        "tokens_per_second": round(tokens / max(time.time() - t0, 1), 1),
        "seconds": round(time.time() - STARTED, 1), "ollama": api("/api/version", timeout=5),
    }
    (WORK / f"summary-{JOB['payload_sha1'][:8]}.json").write_text(json.dumps(summary, indent=1))
    print(json.dumps(summary, indent=1))


main()
