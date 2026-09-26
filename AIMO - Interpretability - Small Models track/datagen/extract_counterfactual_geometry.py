"""E31: prompt-state stability under deterministic, answer-preserving rewrites.

Robust solutions should depend on the mathematical content rather than incidental wording.
This extractor compares the model's prompt representation for an original problem with the
same representation after four deterministic rewrites from E14.  Unlike E14, it performs no
generation: each rewrite costs one teacher-forced forward pass.

The five signals were fixed before inspecting E31 labels/results.  Larger values always mean
*more* sensitivity, so the confirmatory prediction is that robust rows have lower values:

* ``cf_cosine_tail_mean``: mean cosine distance in the last quarter of layers;
* ``cf_cosine_tail_max``: maximum cosine distance in the last quarter;
* ``cf_cosine_auc``: mean cosine distance over all non-embedding layers;
* ``cf_cosine_slope``: last-quarter minus first-quarter distance;
* ``cf_js_next_mean``: mean Jensen-Shannon divergence of next-token distributions.

Only official ``train-main-v2`` rows are used by default.  Output is one resumable JSONL
record per labelled model/problem pair.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import sys
import time
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import torch
from transformers import AutoModelForCausalLM

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE / "perturb"))

from counterfactual_geometry import SIGNALS, counterfactual_signals  # noqa: E402
from counterfactual_rewrites import rewrites  # noqa: E402
from extract_features import (  # noqa: E402
    DEFAULT_INSTANCES,
    assign_shards,
    build_prompt,
    load_system_prompt,
    resolve_dtype,
)
from tokenizer_check import load_checked_tokenizer  # noqa: E402

DEFAULT_ARTIFACT = (
    HERE.parent
    / "getting-started"
    / "solutions"
    / "trained-probe"
    / "probe_artifacts"
    / "probe_artifact.pkl"
)
def forward_signature(model, tokenizer, prompt: str, max_length: int) -> tuple[np.ndarray, np.ndarray, int]:
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=max_length)
    inputs = {key: value.to(model.device) for key, value in inputs.items()}
    with torch.inference_mode():
        output = model(
            **inputs,
            output_hidden_states=True,
            return_dict=True,
            use_cache=False,
        )
    states = np.stack(
        [layer[0, -1].float().cpu().numpy() for layer in output.hidden_states]
    ).astype(np.float32)
    logits = output.logits[0, -1].float().cpu().numpy().astype(np.float32)
    n_tokens = int(inputs["attention_mask"][0].sum().item())
    del output, inputs
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return states, logits, n_tokens


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-id", required=True)
    parser.add_argument("--instances", type=Path, default=DEFAULT_INSTANCES)
    parser.add_argument(
        "--dataset-id",
        action="append",
        dest="dataset_ids",
        help="dataset to include; repeat for multiple datasets (default: train-main-v2)",
    )
    parser.add_argument("--artifact", type=Path, default=DEFAULT_ARTIFACT)
    parser.add_argument("--system-prompt-file", type=Path, default=None)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--dtype", default="auto", choices=["auto", "float16", "bfloat16", "float32"])
    parser.add_argument("--max-length", type=int, default=2048)
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

    if not 0 <= args.shard_index < args.num_shards:
        raise SystemExit("--shard-index must satisfy 0 <= index < --num-shards")
    system_prompt = (
        args.system_prompt_file.read_text(encoding="utf-8")
        if args.system_prompt_file
        else load_system_prompt(args.artifact)
    )
    # Select only flat columns.  Besides reducing I/O, this remains readable by Kaggle's
    # older Arrow build when the local file was written by Arrow 25 with nested-list stats.
    table = pq.read_table(
        args.instances,
        columns=["model_id", "problem_key", "problem", "dataset_id", "is_robust"],
    )
    dataset_ids = args.dataset_ids or ["train-main-v2"]
    rows = [
        row
        for row in table.to_pylist()
        if row["model_id"] == args.model_id and row["dataset_id"] in dataset_ids
    ]
    if not rows:
        raise SystemExit(f"no rows from {dataset_ids!r} for {args.model_id!r}")
    assignment = assign_shards([row["problem_key"] for row in rows], args.num_shards, "roundrobin")
    rows = sorted(
        (row for row in rows if assignment[row["problem_key"]] == args.shard_index),
        key=lambda row: row["problem_key"],
    )
    if args.limit:
        rows = rows[: args.limit]

    shard_dir = args.out_dir / f"shard_{args.shard_index:02d}_of_{args.num_shards:02d}"
    shard_dir.mkdir(parents=True, exist_ok=True)
    records_path = shard_dir / "records.jsonl"
    done: set[str] = set()
    if records_path.is_file():
        done = {
            json.loads(line)["problem_key"]
            for line in records_path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        }
    todo = [row for row in rows if row["problem_key"] not in done]
    print(f"{len(rows)} official rows, {len(done)} done, {len(todo)} to do", flush=True)
    if not todo:
        return

    dtype = resolve_dtype(args.dtype)
    tokenizer, tokenizer_info = load_checked_tokenizer(args.model_id)
    print(f"loading {args.model_id} as {dtype}", flush=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_id,
        dtype=dtype,
        device_map="auto" if torch.cuda.is_available() else None,
    )
    model.eval()
    started = time.time()
    n_forwards = 0
    rewrite_counts: dict[str, int] = {}

    for index, row in enumerate(todo, start=1):
        variants = rewrites(row["problem"])
        if not variants:
            raise RuntimeError(f"no applicable rewrites for {row['problem_key']}")
        prompts = {"original": row["problem"], **variants}
        signatures: dict[str, tuple[np.ndarray, np.ndarray, int]] = {}
        for kind, problem in prompts.items():
            prompt = build_prompt(tokenizer, problem, system_prompt)
            signatures[kind] = forward_signature(model, tokenizer, prompt, args.max_length)
            n_forwards += 1
            rewrite_counts[kind] = rewrite_counts.get(kind, 0) + 1

        anchor_states, anchor_logits, anchor_tokens = signatures.pop("original")
        ordered = sorted(signatures)
        record = {
            "problem_key": row["problem_key"],
            "model_id": args.model_id,
            "is_robust": bool(row["is_robust"]),
            "dataset_id": row["dataset_id"],
            "rewrite_kinds": ordered,
            "n_rewrites": len(ordered),
            "n_prompt_tokens": anchor_tokens,
            "max_rewrite_tokens": max(signatures[kind][2] for kind in ordered),
            "truncated": any(signatures[kind][2] >= args.max_length for kind in ordered)
            or anchor_tokens >= args.max_length,
            **counterfactual_signals(
                anchor_states,
                [signatures[kind][0] for kind in ordered],
                anchor_logits,
                [signatures[kind][1] for kind in ordered],
            ),
        }
        with records_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record) + "\n")
        elapsed = time.time() - started
        print(
            f"  [{index}/{len(todo)}] {n_forwards} forwards, "
            f"{elapsed / n_forwards:.1f}s/forward, ~{elapsed / index * (len(todo) - index) / 60:.1f} min left",
            flush=True,
        )

    metadata = {
        "stage": "E31 counterfactual activation geometry",
        "model_id": args.model_id,
        "dataset_ids": dataset_ids,
        "dtype": str(dtype),
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "signals": list(SIGNALS),
        "predicted_direction": "lower_is_robust",
        "n_rows": len(rows),
        "n_forwards": n_forwards,
        "rewrite_counts": rewrite_counts,
        "max_length": args.max_length,
        "system_prompt_sha1": hashlib.sha1(system_prompt.encode()).hexdigest(),
        "seconds": round(time.time() - started, 1),
        "torch": torch.__version__,
        "platform": platform.platform(),
        **tokenizer_info,
    }
    (shard_dir / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(f"done -> {shard_dir}", flush=True)


if __name__ == "__main__":
    main()
