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
QWEN35 = "Qwen/Qwen3.5-4B"
SKYWORK = "Skywork/Skywork-OR1-Math-7B"
OLMO_THINK = "allenai/Olmo-3-7B-Think"

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
    # Kaggle ships scikit-learn 1.6.1, which cannot unpickle 1.8.0 estimators (e11-gpu-check).
    "scikit-learn==1.8.0",
    "joblib==1.5.3",
]

BUNDLE_FILES = [
    "datagen/tokenizer_check.py",  # imported by every extractor (Finding 14)
    "datagen/extract_features.py",
    "datagen/extract_generation_features.py",
    "datagen/extract_trace_states.py",
    "datagen/merge_features.py",
    "datagen/extract_label_free.py",
    "datagen/extract_agreement.py",
    "datagen/perturb/transforms.py",
    "data/small_track/instances.parquet",
    "data/small_track/instances_train.parquet",
    "data/small_track/splits.json",
    "data/small_track/summary.json",
]
BUNDLE_DIRS = ["getting-started/solutions/uncertainty-profiling/uncertainty_profile"]

# Kaggle rejects very large script sources; past this, ship inputs as a private dataset instead.
BUNDLE_WARN_BYTES = 900_000

# ISHMAM1 and DESIKALAKAR stopped authenticating on 17 Sep; RSD and Labib were added on 18 Sep.
# NAKIB, CHOTON, MEDIUM and MR come from the `nakib`/`nakib_v2` branches; each alias needs its
# token in Secrets.env. CHOTON and MEDIUM are the *same Kaggle accounts* as Labib and RSD under
# a second spelling, so their quota is shared, not additional (Finding 42).
# Eshad was added on 22 Sep with a fresh quota.
ACCOUNTS = ["ISHRAQ", "DIDHITI", "TASHIN", "SALAM", "ISHMAM1", "DESIKALAKAR", "RSD", "Labib", "NAKIB", "CHOTON", "MEDIUM", "MR", "Eshad"]

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
    # Private Kaggle datasets to mount, for inputs too large to embed (e.g. a submission ZIP).
    dataset_sources: tuple[str, ...] = ()


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


def label_free(model_id: str, out_name: str, *extra: str) -> tuple[str, ...]:
    return (
        "datagen/extract_label_free.py",
        "--model-id", model_id,
        "--system-prompt-file", SYSTEM_PROMPT_PATH,
        "--out-dir", f"{KAGGLE_FEATURES}/{out_name}",
        *extra,
    )


def describe_model(model_id: str) -> tuple[str, ...]:
    """Download the checkpoint and print what transformers makes of it; never fails the run."""
    return ("-c", (
        "from huggingface_hub import snapshot_download; from transformers import AutoConfig\n"
        f"print(snapshot_download('{model_id}'))\n"
        "try:\n"
        f"    c = AutoConfig.from_pretrained('{model_id}'); print(type(c).__name__, c.architectures)\n"
        "except Exception as exc:\n"
        "    print('AutoConfig failed:', type(exc).__name__, exc)"
    ))


def plumbing_and_label_free(model_id: str, short: str) -> tuple[tuple[str, ...], ...]:
    """E13: does the model load and run on T4x2 under the pins, then E12 scores for every train problem."""
    return (
        describe_model(model_id),
        stage_a(model_id, f"stagea-{short}-smoke", "--no-filter-model", "--limit", "20"),
        label_free(model_id, f"labelfree-{short}", "--no-filter-model"),
    )


VLLM_INSTALL = ("-m", "pip", "install", "-q", "vllm")
E7_FILES = ("datagen/perturb/*.py", "data/olympiad_pool/pool.parquet")
# Base-phase records already sampled, shipped back so a new session resumes instead of redoing them.
QWEN_RECORDS = "runs/e15-base-qwen35-*/e15-qwen35/base/shard_*/records.jsonl"
SKYWORK_RECORDS = "runs/e16-base-skywork/e16-skywork/base/shard_*/records.jsonl"
VARIANTS = "data/perturbations/variants.parquet"
SKYWORK_VARIANT_RECORDS = "runs/e19-variants-skywork/e19-skywork/variants/shard_*/records.jsonl"
# E20: DeepSeek's officially labelled MATH problems in pool form (datagen/perturb/deepseek_check_pool.py).
CHECK_POOL = "data/deepseek_check/pool.parquet"
CHECK_VARIANTS = "data/deepseek_check/variants.parquet"
# E24: the labelled olympiad problems of train-main-v2 (datagen/perturb/official_check_pool.py).
OFFICIAL_POOL = "data/official_check/pool.parquet"
OFFICIAL_VARIANTS = "data/official_check/variants.parquet"

# Problems E18 sampled Qwen variants for while their base records stayed on an account this
# checkout cannot read. Listed literally so the kernel records exactly which problems it
# re-sampled (ported from `nakib_v2`, Finding 36).
QWEN35_GAPFILL_KEYS = (
    "09614ab696400998", "0b54136fc692e0c7", "10f97723a88077d2", "1135f49af4fcedbb",
    "1293217ebd97ed17", "139d89afb527f5e6", "1a17db537035c4f4", "1f0ccfd0d6792c8e",
    "27a4e7ab5d29661f", "5b10a0fd172a4dd6", "691cb9fbdb99b9a9", "9e3cfe84ec7606f1",
    "b5ac25ab547ec3db", "b93905359e1ad28d", "cd09ffc429cc54dd", "d4b1650741179e7b",
    "d4ccf4347ab82805", "ebcaa5cf29f5e809", "f72895c4fbcfa580",
)
# The same, for the 25 problems E19 sampled Skywork variants for.
SKYWORK_GAPFILL_KEYS = (
    "1a17db537035c4f4", "1d8ef1e3f2eaf603", "2025c735c4ffde03", "2053390e1ec47f98",
    "25aa0f4d78fcd6c3", "32d936881040ffa5", "3980c787bdea2215", "3aad65408a0f1875",
    "44206a7619bcad94", "4893606b65d43380", "506d378bc029f34d", "725ef63f684781bb",
    "79e1abbe53ef6879", "861cf79dfc623f20", "874b4a8d662051bb", "8f6a5fcacd3175bd",
    "921983dd11591226", "adeb6cb10b2fe831", "b0832a17441dcd5b", "b265d695c151c5c7",
    "c21ef3d4d630a47e", "d50ba6f1dcbb7572", "e53faab347086059", "e7c519bf1eea1396",
    "eadf4b95241b8c82",
)


def official_check(
    model_id: str, short: str, *engine: str, max_new_tokens: int = 24576, hours: tuple[float, float] = (4.0, 6.5),
    per_gpu_engines: bool = False,
) -> tuple[tuple[str, ...], ...]:
    """E24b: base phase, then `answer_modulus` variants, on the official problems `model_id` is labelled on.

    Olympiad problems truncate often at 16k (58% of Qwen's samples did, Finding 26), and these are
    problems the model solved in the organizers' runs, so the cap is 24k where memory allows.
    """
    pool, out = f"data/official_check/pool_{short}.parquet", f"e24b-{short}"
    base_records = f"/kaggle/working/{out}/base/shard_*/records.jsonl"
    shards = [("--num-shards", "2", "--shard-index", str(g)) for g in (0, 1)] if per_gpu_engines else [()]

    def base(shard: tuple[str, ...]) -> tuple[str, ...]:
        return base_sampling(model_id, out, "--pool", pool, *engine, *shard,
                             max_new_tokens=max_new_tokens, max_hours=hours[0])

    def variants(shard: tuple[str, ...]) -> tuple[str, ...]:
        return variant_sampling(model_id, out, base_records, "--pool", pool, "--variants", OFFICIAL_VARIANTS,
                                *engine, *shard, max_new_tokens=max_new_tokens, max_hours=hours[1])

    if per_gpu_engines:
        return (VLLM_INSTALL, per_gpu(*map(base, shards)), per_gpu(*map(variants, shards)))
    return (VLLM_INSTALL, base(()), variants(()))


def familiarity(model_id: str, out_name: str) -> tuple[str, ...]:
    return ("datagen/extract_familiarity.py", "--model-id", model_id, "--out-dir", f"{KAGGLE_FEATURES}/{out_name}")


FAMILIARITY_FILES = ("datagen/extract_familiarity.py", "datagen/ingest_public.py")


def agreement(model_id: str, out_name: str, *extra: str) -> tuple[str, ...]:
    return (
        "datagen/extract_agreement.py",
        "--model-id", model_id,
        "--out-dir", f"{KAGGLE_FEATURES}/{out_name}",
        *extra,
    )


def base_sampling(
    model_id: str, out_name: str, *extra: str, max_new_tokens: int = 16384, chunk: int = 4, max_hours: float = 8.5,
) -> tuple[str, ...]:
    """E7 base phase over the olympiad pool: k samples per problem, flushed every 4 problems.

    Sizing is the thing to get right. A chunk is one generate call and one flush, and Kaggle
    kills a session at 12 h, so a chunk that does not finish is work thrown away. The first
    attempt asked for 32 problems x 4 samples at a 16k cap, roughly 2M tokens per call:
    Qwen managed one chunk per GPU, Skywork five, and Olmo-3-Think none at all in 11.5 h
    (Finding 25). Four problems per call (16 sequences, about what a T4's KV cache holds
    anyway) flushes every 20-45 minutes, and an 8.5 h budget leaves room to exit cleanly.
    """
    return (
        "datagen/perturb/sample.py", "--phase", "base",
        "--model-id", model_id,
        "--backend", "vllm",
        "--k", "4", "--max-new-tokens", str(max_new_tokens), "--max-hours", str(max_hours), "--chunk", str(chunk),
        "--out-dir", f"/kaggle/working/{out_name}",
        *extra,
    )


def variant_sampling(
    model_id: str, out_name: str, base_records: str, *extra: str,
    max_new_tokens: int = 16384, chunk: int = 4, max_hours: float = 8.5,
) -> tuple[str, ...]:
    """E7 variants phase: k samples of every compliant variant of each problem the model always solves.

    Only `compliance == "distinct"` categories (today `answer_modulus`) are sampled, the ones the
    extra-data rule lets us label. Eligibility comes from the base-phase records in the bundle.
    """
    return (
        "datagen/perturb/sample.py", "--phase", "variants",
        "--model-id", model_id,
        "--backend", "vllm",
        "--base-records", base_records,
        "--k", "4", "--max-new-tokens", str(max_new_tokens), "--max-hours", str(max_hours), "--chunk", str(chunk),
        "--out-dir", f"/kaggle/working/{out_name}",
        *extra,
    )


def restore(out_name: str, records_glob: str, phase: str = "base") -> tuple[str, ...]:
    """Copy already-sampled records back into the Kaggle output directory, so a session resumes.

    Kernels start empty, but the sampler skips items whose records it already has. Downloaded
    records travel in the bundle and are put back before sampling starts.
    """
    return ("-c", (
        "import glob, pathlib, shutil\n"
        f"for src in sorted(glob.glob({records_glob!r})):\n"
        "    path = pathlib.Path(src)\n"
        f"    dst = pathlib.Path('/kaggle/working/{out_name}/{phase}') / path.parent.name / path.name\n"
        "    dst.parent.mkdir(parents=True, exist_ok=True)\n"
        "    shutil.copy(src, dst)\n"
        "    print('restored', dst, sum(1 for _ in open(dst)), 'records')"
    ))


def per_gpu(*argvs: tuple[str, ...]) -> tuple[str, ...]:
    """Run one argv per GPU in parallel (CUDA_VISIBLE_DEVICES=i); fails if any of them fails.

    For models that fit on one T4, two independent engines beat tensor parallelism over PCIe.
    """
    return ("-c", (
        "import os, subprocess, sys\n"
        f"argvs = {[list(a) for a in argvs]!r}\n"
        "procs = [subprocess.Popen([sys.executable, *a], env={**os.environ, 'CUDA_VISIBLE_DEVICES': str(i)})\n"
        "         for i, a in enumerate(argvs)]\n"
        "sys.exit(max(p.wait() for p in procs))"
    ))


def qwen_base_pair(first_shard: int) -> tuple[str, ...]:
    return per_gpu(*(
        base_sampling(QWEN35, "e15-qwen35", "--tensor-parallel", "1", "--text-only", "--max-num-seqs", "64",
                      "--num-shards", "4", "--shard-index", str(first_shard + gpu))
        for gpu in (0, 1)
    ))


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
    "e7-sample-smoke": Experiment(
        account="ISHMAM1",
        purpose="E7 plumbing and throughput check: installs vLLM, rebuilds the olympiad pool and variants "
        "from pinned revisions and a fixed seed, then samples AMC and AIME originals (k=4, 4,096-token cap) "
        "in fp16 with tensor parallelism across the two T4s. Measures tokens/s and base-eligibility yield.",
        steps=(
            ("-m", "pip", "install", "-q", "vllm"),
            ("datagen/perturb/pool.py",),
            ("datagen/perturb/generate.py", "--examples", "0"),
            ("datagen/perturb/sample.py", "--phase", "base", "--model-id", DEEPSEEK, "--backend", "vllm",
             "--tiers", "amc", "--limit", "16", "--k", "4", "--max-new-tokens", "4096",
             "--out-dir", "/kaggle/working/e7-amc"),
            ("datagen/perturb/sample.py", "--phase", "base", "--model-id", DEEPSEEK, "--backend", "vllm",
             "--tiers", "aime", "--limit", "8", "--k", "4", "--max-new-tokens", "4096",
             "--out-dir", "/kaggle/working/e7-aime"),
        ),
        bundle_globs=("datagen/ingest_public.py", "datagen/perturb/*.py"),
    ),
    "e1b-stagea-deepseek": Experiment(
        account="ISHMAM1",
        purpose="E1 re-extraction with the round-trip-checked tokenizer: e1-stagea-deepseek ran on "
        "AutoTokenizer, which strips all whitespace from DeepSeek-R1-0528-Qwen3-8B prompts (Finding 14).",
        steps=(
            stage_a(DEEPSEEK, "stagea-deepseek"),
            merge(f"{KAGGLE_FEATURES}/stagea-deepseek"),
        ),
    ),
    "e7b-sample-smoke": Experiment(
        account="TASHIN",
        purpose="E7 smoke rerun with the tokenizer fix (token-id prompts, own decoding) and longer caps: "
        "e7-sample-smoke hit its 4,096-token cap on 80% of AMC and 97% of AIME samples. AMC at 8,192 "
        "tokens, AIME at 16,384, k=4.",
        steps=(
            ("-m", "pip", "install", "-q", "vllm"),
            ("datagen/perturb/pool.py",),
            ("datagen/perturb/generate.py", "--examples", "0"),
            ("datagen/perturb/sample.py", "--phase", "base", "--model-id", DEEPSEEK, "--backend", "vllm",
             "--tiers", "amc", "--limit", "16", "--k", "4", "--max-new-tokens", "8192",
             "--out-dir", "/kaggle/working/e7b-amc"),
            ("datagen/perturb/sample.py", "--phase", "base", "--model-id", DEEPSEEK, "--backend", "vllm",
             "--tiers", "aime", "--limit", "8", "--k", "4", "--max-new-tokens", "16384",
             "--out-dir", "/kaggle/working/e7b-aime"),
        ),
        bundle_globs=("datagen/ingest_public.py", "datagen/perturb/*.py"),
    ),
    "e11-gpu-check": Experiment(
        account="ISHMAM1",
        purpose="Pre-submission check of dist/solution-e11-text-stagea-small.zip on a real GPU: runs the "
        "packaged are_robust on the 138 DeepSeek training problems, compares decisions and stagea "
        "probabilities with the artifact's reference, and times the call.",
        steps=(
            # The solution loads with local_files_only=True, like the evaluation worker's warm
            # cache; download the checkpoint first, or it silently falls back to text only.
            ("-c", f"from huggingface_hub import snapshot_download; print(snapshot_download('{DEEPSEEK}'))"),
            ("submission/gpu_check.py", "--search-root", "/kaggle/input"),
        ),
        bundle_globs=("submission/gpu_check.py",),
        dataset_sources=("ishmamahmid/aimo-small-e11-bundle",),
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
    "e12-labelfree-deepseek": Experiment(
        account="ISHMAM1",
        purpose="E12 label-free signals on DeepSeek's 138 labelled problems: immediate-answer confidence, "
        "answer stability under latex_to_plaintext / premise_reorder rewrites, problem NLL. Validates "
        "the signals before they are used for models without labels.",
        steps=(label_free(DEEPSEEK, "labelfree-deepseek"),),
    ),
    "e13-qwen35-plumbing": Experiment(
        account="TASHIN",
        purpose="Qwen3.5-4B (the model the leaderboard appears to score, Finding 21): loads under "
        "transformers 5.13 on T4x2? Stage A smoke on 20 problems, then E12 label-free scores for all "
        "train problems. Labels are NOT Qwen's (--no-filter-model).",
        steps=plumbing_and_label_free(QWEN35, "qwen35"),
    ),
    "e13-skywork-plumbing": Experiment(
        account="ISHRAQ",
        purpose="Skywork-OR1-Math-7B plumbing check plus E12 label-free scores (labels are NOT Skywork's).",
        steps=plumbing_and_label_free(SKYWORK, "skywork"),
    ),
    "e13-olmo-think-plumbing": Experiment(
        account="DESIKALAKAR",
        purpose="Olmo-3-7B-Think plumbing check plus E12 label-free scores (labels are NOT Olmo's).",
        steps=plumbing_and_label_free(OLMO_THINK, "olmo-think"),
    ),
    "e14-agreement-deepseek-s0": Experiment(
        account="ISHRAQ",
        purpose="E14 test-time answer agreement, shard 0/2: DeepSeek reasons (k=2, 8k cap) on each labelled "
        "problem and on up to four rewrites (latex_to_plaintext, premise_reorder, typos, distract). Does "
        "the answer survive? Validates a label-free, model-specific signal against DeepSeek's labels.",
        steps=(VLLM_INSTALL, agreement(DEEPSEEK, "agreement-deepseek", "--num-shards", "2", "--shard-index", "0",
                                       "--max-hours", "10")),
        bundle_globs=E7_FILES,
    ),
    "e14-agreement-deepseek-s1": Experiment(
        account="DIDHITI",
        purpose="E14 test-time answer agreement, shard 1/2 (see shard 0).",
        steps=(VLLM_INSTALL, agreement(DEEPSEEK, "agreement-deepseek", "--num-shards", "2", "--shard-index", "1",
                                       "--max-hours", "10")),
        bundle_globs=E7_FILES,
    ),
    "e15-base-qwen35-a": Experiment(
        account="TASHIN",
        purpose="E15 Qwen3.5-4B labelling, step 1 (E7 base phase) over the 505-problem olympiad pool, "
        "shards 0-1 of 4: one vLLM engine per T4 (the model fits in fp16), k=4, 16k cap. Finds the "
        "problems Qwen always solves, the only ones that can be labelled. Session 1 measured 27% "
        "eligible over 128 problems; this session resumes from those records.",
        steps=(VLLM_INSTALL, restore("e15-qwen35", QWEN_RECORDS), qwen_base_pair(0)),
        bundle_globs=(*E7_FILES, QWEN_RECORDS),
    ),
    "e15-base-qwen35-b": Experiment(
        account="RSD",
        purpose="E15 Qwen3.5-4B base phase, shards 2-3 of 4 (see e15-base-qwen35-a).",
        steps=(VLLM_INSTALL, restore("e15-qwen35", QWEN_RECORDS), qwen_base_pair(2)),
        bundle_globs=(*E7_FILES, QWEN_RECORDS),
    ),
    "e16-base-skywork": Experiment(
        account="ISHRAQ",
        purpose="E16 Skywork-OR1-Math-7B base phase: TP=2, k=4, 16k cap. Session 1 covered 160 problems "
        "at 63% eligible and 150 tok/s, the best yield of the three; this session resumes from it.",
        steps=(
            VLLM_INSTALL,
            restore("e16-skywork", SKYWORK_RECORDS),
            base_sampling(SKYWORK, "e16-skywork"),
        ),
        bundle_globs=(*E7_FILES, SKYWORK_RECORDS),
    ),
    "e17-base-olmo-think": Experiment(
        account="Labib",
        purpose="E17 Olmo-3-7B-Think base phase: TP=2, k=4, 16k cap. Session 1 produced nothing because "
        "no 32-problem chunk finished in 11.5 h (Finding 25); with 4-problem chunks it flushes as it "
        "goes, which also measures how slow this model really is.",
        steps=(VLLM_INSTALL, base_sampling(
            OLMO_THINK, "e17-olmo-think",
            # Olmo-3 keeps a full KV cache: 4.5 GiB for one 18k-token sequence, against 4.12 GiB
            # free on a T4 pair, so the engine refused to start (18 Sep). 12k fits, barely.
            "--gpu-memory-utilization", "0.95",
            max_new_tokens=12288,
        )),
        bundle_globs=E7_FILES,
    ),
    "e18-variants-qwen35": Experiment(
        account="RSD",
        purpose="E18 Qwen3.5-4B labelling, step 2 (E7 variants phase): k=4 samples of each answer_modulus "
        "variant of the 90 problems Qwen always solved in E15 (109 variants over 53 problems). Two "
        "single-GPU vLLM engines. With E15, gives Qwen's first robustness labels.",
        steps=(VLLM_INSTALL, per_gpu(*(
            variant_sampling(QWEN35, "e18-qwen35", QWEN_RECORDS, "--tensor-parallel", "1", "--text-only",
                             "--max-num-seqs", "64", "--num-shards", "2", "--shard-index", str(gpu))
            for gpu in (0, 1)
        ))),
        bundle_globs=(*E7_FILES, VARIANTS, QWEN_RECORDS),
    ),
    "e19-variants-skywork": Experiment(
        account="Labib",
        purpose="E19 Skywork-OR1-Math-7B labelling, step 2 (E7 variants phase): k=4 samples of each "
        "answer_modulus variant of the problems Skywork always solved in E16 (176 variants over 91 "
        "problems once ambiguous pool ids are dropped). TP=2. Session 1 ran on a slow host (19.9 tok/s) "
        "and covered 24; this session resumes from those records.",
        steps=(
            VLLM_INSTALL,
            restore("e19-skywork", SKYWORK_VARIANT_RECORDS, phase="variants"),
            variant_sampling(SKYWORK, "e19-skywork", SKYWORK_RECORDS),
        ),
        bundle_globs=(*E7_FILES, VARIANTS, SKYWORK_RECORDS, SKYWORK_VARIANT_RECORDS),
    ),
    "e20-validate-deepseek": Experiment(
        account="ISHRAQ",
        purpose="E20: do our self-labels agree with official ones? The E7 pipeline (base phase, then "
        "answer_modulus variants, k=4, 8k cap) on DeepSeek's 66 officially labelled MATH problems with an "
        "integer answer >= 13 (19 officially robust). label.py output is then compared with the official "
        "labels before any model is trained on E15-E19 labels.",
        steps=(
            VLLM_INSTALL,
            base_sampling(DEEPSEEK, "e20-deepseek", "--pool", CHECK_POOL,
                          max_new_tokens=8192, chunk=8, max_hours=3.5),
            variant_sampling(DEEPSEEK, "e20-deepseek", "/kaggle/working/e20-deepseek/base/shard_*/records.jsonl",
                             "--pool", CHECK_POOL, "--variants", CHECK_VARIANTS,
                             max_new_tokens=8192, chunk=8, max_hours=4.5),
        ),
        bundle_globs=(*E7_FILES, CHECK_POOL, CHECK_VARIANTS),
    ),
    "e21-stagea-deepseek-official": Experiment(
        account="DIDHITI",
        purpose="E21 Stage A on DeepSeek over the relabelled 152 train instances, which now include the "
        "10 official olympiad ones from train-main-v2. Gives the E11 probe its first out-of-sample test "
        "on the difficulty shift the private test set asks for (Finding 30).",
        steps=(
            stage_a(DEEPSEEK, "stagea-deepseek-official"),
            merge(f"{KAGGLE_FEATURES}/stagea-deepseek-official"),
        ),
    ),
    "e21-stagea-qwen35-official": Experiment(
        account="SALAM",
        purpose="E21 Stage A on Qwen3.5-4B over its 12 officially labelled problems (8 robust). With the "
        "other three models, the first pooled cross-model probe we can hold a model out of.",
        steps=(
            stage_a(QWEN35, "stagea-qwen35-official"),
            merge(f"{KAGGLE_FEATURES}/stagea-qwen35-official"),
        ),
    ),
    "e21-stagea-skywork-official": Experiment(
        account="TASHIN",
        purpose="E21 Stage A on Skywork-OR1-Math-7B over its 3 officially labelled problems (see "
        "e21-stagea-qwen35-official).",
        steps=(
            stage_a(SKYWORK, "stagea-skywork-official"),
            merge(f"{KAGGLE_FEATURES}/stagea-skywork-official"),
        ),
    ),
    "e21-stagea-olmo-think-official": Experiment(
        account="ISHRAQ",
        purpose="E21 Stage A on Olmo-3-7B-Think over its 9 officially labelled problems (see "
        "e21-stagea-qwen35-official).",
        steps=(
            stage_a(OLMO_THINK, "stagea-olmo-think-official"),
            merge(f"{KAGGLE_FEATURES}/stagea-olmo-think-official"),
        ),
    ),
    # --- Run on the teammate's accounts from the `nakib_v2` branch, ported so the whole fleet's
    # kernels stay addressable from this repository. Their slugs keep the numbering they were
    # pushed under, which collides with ours: `e20-base-qwen35-gapfill` is that branch's E23,
    # and `e21-stagea-qwen35-olympiad` is its E21, neither of which is our E20 or E21.
    "e20-base-qwen35-gapfill": Experiment(
        account="NAKIB",
        purpose="Base samples for the 19 problems E18 already sampled Qwen variants for but whose "
        "base records are unreachable (they live in the e15-base-qwen35-a shards on an account "
        "that checkout cannot read). Each costs one k=4 base sample and unlocks a variant set "
        "already paid for. Keyed on problem_key because pool_id is not unique (Finding 29). "
        "Result: 19/19 sampled, 13 eligible, Qwen reaches 26 train-ready labels (Finding 37) - "
        "since invalidated by E20 (Finding 31).",
        steps=(
            VLLM_INSTALL,
            base_sampling(
                QWEN35, "e20-qwen35", "--tensor-parallel", "1", "--text-only", "--max-num-seqs", "64",
                "--problem-keys", *QWEN35_GAPFILL_KEYS,
            ),
        ),
        bundle_globs=E7_FILES,
    ),
    "e22-base-skywork-gapfill": Experiment(
        account="NAKIB",
        purpose="The same gapfill for Skywork: base samples for the 25 problems E19 sampled "
        "variants for. Result: 24 of 25 eligible, 23 train-ready labels (Finding 37) - since "
        "invalidated by E20 (Finding 31). TP=2, as Skywork in fp16 does not fit one T4.",
        steps=(
            VLLM_INSTALL,
            base_sampling(SKYWORK, "e22-skywork", "--problem-keys", *SKYWORK_GAPFILL_KEYS),
        ),
        bundle_globs=E7_FILES,
    ),
    "e21-stagea-qwen35-olympiad": Experiment(
        account="MR",
        purpose="Do prompt hidden states carry olympiad robustness signal when text does not? "
        "Stage A over 38 Qwen3.5-4B problems from datagen/build_stagea_olympiad_set.py: 26 "
        "answer-modulus olympiad self-labels to fit probes on, and the 12 officially labelled "
        "train-main-v2 rows to score them against, disjoint by problem_key. Result: AUC 0.844 "
        "against text's 0.688, but the strict control gives p = 0.073 and the training labels "
        "were later invalidated by E20 (Findings 41, 31).",
        steps=(
            stage_a(QWEN35, "stagea-qwen35-olympiad",
                    "--instances", "data/stagea_olympiad/instances.parquet"),
            merge(f"{KAGGLE_FEATURES}/stagea-qwen35-olympiad"),
        ),
        bundle_globs=("data/stagea_olympiad/instances.parquet",),
    ),
    "e24a-answer-key-skywork": Experiment(
        # Eshad's first push ran without a GPU (no `nvidia-smi`): the account needs phone verification.
        account="ISHRAQ",
        purpose="E24a: an answer key for the 20 olympiad problems train-main-v2 labels for a small-track "
        "model, which ship without answers. Skywork base phase, k=8, 16k cap, TP=2; 14 problems carry a "
        "placeholder answer, so the reference is read from each record's sampled answers and kept only when "
        "all 8 agree. The 6 problems whose answers our pool already knows check the method. Feeds E24b, "
        "gate G-V2 (PLAN.md section 11).",
        steps=(
            VLLM_INSTALL,
            base_sampling(SKYWORK, "e24a-skywork", "--pool", OFFICIAL_POOL, "--k", "8", max_hours=6.0),
        ),
        bundle_globs=(*E7_FILES, OFFICIAL_POOL),
    ),
    "e23-familiarity-a": Experiment(
        account="RSD",
        purpose="E23: familiarity and contamination scores (mean NLL, Min-K%, Min-K%++, zlib ratio, suffix "
        "NLL) from one forward pass over each bare problem, for every labelled train problem and all 36 "
        "train-main-v2 problems. Tests whether spurious means memorized (gate G-M, PLAN.md section 11). "
        "Qwen3.5-4B, then DeepSeek.",
        steps=(familiarity(QWEN35, "familiarity-qwen35"), familiarity(DEEPSEEK, "familiarity-deepseek")),
        bundle_globs=FAMILIARITY_FILES,
    ),
    "e23-familiarity-b": Experiment(
        account="Labib",
        purpose="E23 for Skywork-OR1-Math-7B, then Olmo-3-7B-Think (see e23-familiarity-a).",
        steps=(familiarity(SKYWORK, "familiarity-skywork"), familiarity(OLMO_THINK, "familiarity-olmo-think")),
        bundle_globs=FAMILIARITY_FILES,
    ),
    "e27d-gpu-check-a": Experiment(
        account="TASHIN",
        purpose="GPU check of dist/solution-e27d-familiarity-small.zip before upload: unpack it as Codabench "
        "would, load each model from the cache, and score its officially labelled problems with no "
        "fallback allowed. Qwen3.5-4B and DeepSeek.",
        steps=(("submission/check_e27d.py", "--models", QWEN35, DEEPSEEK),),
        bundle_globs=("submission/check_e27d.py", "dist/solution-e27d-familiarity-small.zip",
                      "data/official_check/pool.parquet", "data/official_check/official_labels.parquet"),
    ),
    "e27d-gpu-check-b": Experiment(
        # SALAM's and DIDHITI's GPU slots were all taken on 22 Sep by another project sharing these
        # accounts (a segmentation challenge), so their quota is not ours alone either.
        account="RSD",
        purpose="The same check for Skywork-OR1-Math-7B and Olmo-3-7B-Think (see e27d-gpu-check-a).",
        steps=(("submission/check_e27d.py", "--models", SKYWORK, OLMO_THINK),),
        bundle_globs=("submission/check_e27d.py", "dist/solution-e27d-familiarity-small.zip",
                      "data/official_check/pool.parquet", "data/official_check/official_labels.parquet"),
    ),
    "e24b-qwen35": Experiment(
        account="TASHIN",
        purpose="E24b, gate G-V2: do answer_modulus self-labels agree with official labels on olympiad "
        "problems? Qwen3.5-4B on the 9 official problems it is labelled on that have a reference answer: "
        "base phase k=4, then 2 answer_modulus variants each, k=4, 24k cap, one vLLM engine per T4.",
        steps=official_check(QWEN35, "qwen35", "--tensor-parallel", "1", "--text-only", "--max-num-seqs", "64",
                             per_gpu_engines=True),
        bundle_globs=(*E7_FILES, "data/official_check/pool_qwen35.parquet", OFFICIAL_VARIANTS),
    ),
    "e24b-deepseek": Experiment(
        account="ISHRAQ",
        purpose="E24b for DeepSeek-R1-0528-Qwen3-8B on its 7 answerable official problems (see e24b-qwen35). TP=2.",
        steps=official_check(DEEPSEEK, "deepseek"),
        bundle_globs=(*E7_FILES, "data/official_check/pool_deepseek.parquet", OFFICIAL_VARIANTS),
    ),
    "e24b-skywork": Experiment(
        account="RSD",
        purpose="E24b for Skywork-OR1-Math-7B on its 3 official problems (see e24b-qwen35). TP=2.",
        steps=official_check(SKYWORK, "skywork"),
        bundle_globs=(*E7_FILES, "data/official_check/pool_skywork.parquet", OFFICIAL_VARIANTS),
    ),
    "e24b-olmo-think": Experiment(
        account="Labib",
        purpose="E24b for Olmo-3-7B-Think on its 8 answerable official problems (see e24b-qwen35). TP=2 at a "
        "12k cap, the most its full KV cache allows on a T4 pair (Finding 26).",
        steps=official_check(OLMO_THINK, "olmo-think", "--gpu-memory-utilization", "0.95",
                             max_new_tokens=12288, hours=(4.5, 6.5)),
        bundle_globs=(*E7_FILES, "data/official_check/pool_olmo-think.parquet", OFFICIAL_VARIANTS),
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


def kernel_ref(name: str, account: str | None = None) -> str:
    username, _ = load_accounts()[account or EXPERIMENTS[name].account]
    return f"{username}/aimo-small-{name}"


def build(name: str, account: str | None = None) -> Path:
    experiment = EXPERIMENTS[name]
    account = account or experiment.account
    bundle, sha1 = build_bundle(experiment.bundle_globs)
    payload = {
        "name": name,
        "account": account,
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
    ref = kernel_ref(name, account)
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
        "dataset_sources": list(experiment.dataset_sources),
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
    # --account runs a registered experiment on another fleet account without editing its
    # registry entry, so outputs keep the runs/<experiment>/ paths the train/ scripts expect.
    for command, nargs in (("build", "+"), ("push", "+"), ("output", "+"), ("status", "*"), ("logs", "*")):
        command_parser = sub.add_parser(command)
        command_parser.add_argument("experiments", nargs=nargs, choices=sorted(EXPERIMENTS))
        command_parser.add_argument("--account", choices=ACCOUNTS, help="override the registered account")
    args = parser.parse_args()

    if args.command == "list":
        for name, experiment in EXPERIMENTS.items():
            print(f"{name:26s} {experiment.account:12s} {experiment.purpose}")
        return

    for name in args.experiments or list(EXPERIMENTS):
        experiment = EXPERIMENTS[name]
        account = args.account or experiment.account
        if args.command == "build":
            build(name, account)
        elif args.command == "push":
            out = build(name, account)
            code, text = kaggle(account, "kernels", "push", "-p", str(out), "--accelerator", experiment.accelerator)
            print(f"[{account}] push exit={code}\n{text}\n")
        elif args.command == "status":
            code, text = kaggle(account, "kernels", "status", kernel_ref(name, account))
            print(f"{name:26s} [{account}] {text}")
        elif args.command == "logs":
            code, text = kaggle(account, "kernels", "logs", kernel_ref(name, account))
            tail = [line for line in text.splitlines() if line.strip()][-40:]
            print(f"===== {name} [{account}] exit={code}\n" + "\n".join(tail) + "\n")
        elif args.command == "output":
            destination = RUNS_DIR / name
            destination.mkdir(parents=True, exist_ok=True)
            # The CLI fetches a single page per call. Per-instance vectors are redundant
            # once merged on Kaggle, so skip them and ask for the maximum page size.
            code, text = kaggle(
                account, "kernels", "output", kernel_ref(name, account),
                "-p", str(destination), "-o",
                "--page-size", "200", "--file-pattern", r"^(?!.*/vectors/)",
            )
            print(f"[{account}] output exit={code} -> {destination}\n{text[-2000:]}\n")


if __name__ == "__main__":
    main()
