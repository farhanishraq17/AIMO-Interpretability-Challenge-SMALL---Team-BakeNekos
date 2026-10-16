"""Allowlisted E55 bundles: one-pass features for the E55 problems, one small-track model per kernel.

Same recipe as E52's features kernel, which ran clean on Kaggle (Finding 77): the evaluation pins,
`extract_familiarity.py` and Stage A via `extract_features.py`, merged on Kaggle with the
per-problem vectors deleted. Each kernel ships only its own model's rows of
`data/e55/instances.parquet`, and records every step's wall time, model load included, which is
E55's one-pass timing (`protocols/E55_PROTOCOL.md`, amendment 1).

Run via:  python scripts/kaggle_followup.py push e55-features-qwen --account <ALIAS>
"""

import io
import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "datagen"))
sys.path.insert(0, str(ROOT / "scripts"))
from e52_gpu_bundle import HEADER, MODELS, SYSTEM_PROMPT, payload  # noqa: E402
from extract_features import DEFAULT_ARTIFACT, load_system_prompt  # noqa: E402
from kaggle_launch import PIP_PINS  # noqa: E402

FILES = ("datagen/extract_familiarity.py", "datagen/extract_features.py", "datagen/merge_features.py",
         "datagen/tokenizer_check.py", "datagen/ingest_public.py", "protocols/E55_PROTOCOL.md")
INSTANCES = ROOT / "data" / "e55" / "instances.parquet"
SHIPPED = "data/e55/instances_model.parquet"
JOBS = [f"features-{short}" for short in MODELS]


def model_rows(model_id: str) -> bytes:
    rows = pd.read_parquet(INSTANCES)
    rows = rows[rows["model_id"] == model_id]
    assert len(rows), f"no E55 instances for {model_id}"
    buffer = io.BytesIO()
    rows.to_parquet(buffer, index=False)
    return buffer.getvalue()


def render(job: str) -> tuple[str, dict]:
    short = job.removeprefix("features-")
    model = MODELS[short]
    extra = {SYSTEM_PROMPT: load_system_prompt(DEFAULT_ARTIFACT).encode("utf-8"), SHIPPED: model_rows(model)}
    entries, hashes = payload(FILES, extra)
    out = f"e55-{job}"
    source = HEADER.format(out=out, entries=json.dumps(entries)) + f'''subprocess.run([sys.executable, "-m", "pip", "install", "--no-cache-dir", "-q", *{PIP_PINS!r}], check=True)
freeze = subprocess.run([sys.executable, "-m", "pip", "freeze"], capture_output=True, text=True, check=True)
(out / "environment.txt").write_text(freeze.stdout)
steps = {{
    "familiarity": ["datagen/extract_familiarity.py", "--model-id", {model!r}, "--instances", {SHIPPED!r},
                    "--out-dir", str(out / "familiarity-{short}")],
    "stagea": ["datagen/extract_features.py", "--model-id", {model!r}, "--instances", {SHIPPED!r},
               "--system-prompt-file", "{SYSTEM_PROMPT}", "--out-dir", str(out / "stagea-{short}")],
    "merge": ["datagen/merge_features.py", "--features-dir", str(out / "stagea-{short}")],
}}
record = {{"model_id": {model!r}, "steps": {{}}}}
for step, argv in steps.items():
    began = time.time()
    code = subprocess.run([sys.executable, *argv], cwd=root).returncode
    record["steps"][step] = {{"exit_code": code, "seconds": round(time.time() - began, 1)}}
    (out / "execution.json").write_text(json.dumps({{**record, "end_to_end_seconds": time.time() - started}}, indent=2))
for vectors in (out / "stagea-{short}").glob("shard_*/vectors"):
    shutil.rmtree(vectors)  # merged into features.npz; one file per problem would overflow the output page
raise SystemExit(max(s["exit_code"] for s in record["steps"].values()))
'''
    compile(source, "run.py", "exec")
    return source, hashes
