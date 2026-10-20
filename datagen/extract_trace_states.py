"""E4: hidden states over the model's own reasoning trace (teacher-forced).

Stage A reads the prompt only. Finding 11 found no single prompt layer that generalises,
but an all-layer ensemble out-ranks problem text, and fused with text it lifts balanced
accuracy (Finding 16). The natural next question is whether internals *during reasoning*
carry more signal. Does the model's state while working through its own greedy solution
separate robust from spurious answers better than its state before it starts?

This script feeds each E2 greedy trace back through the model in a single teacher-forced
forward pass (prompt + generated text, no sampling, no KV cache). It then pools every
layer's hidden states over five segments:

    prompt_last        final prompt token: the state generation started from
    trace_mean         mean over all trace tokens
    trace_first_third  mean over the first third of the trace (setting up)
    trace_last_third   mean over the last third (answer and any self-checking)
    trace_last         final trace token

**Layers are pooled through forward hooks, not `output_hidden_states=True`.** Keeping all 37
layers of a 6k-token sequence costs about 1.9 GB on top of the model and ran a T4 out of
memory at instance 100 of 138. Hooks pool each layer the moment it is produced, so only one
layer's activations are ever held.

The output layout matches `extract_features.py` (vectors/*.npz, records.jsonl,
metadata.json), so `merge_features.py`, `train/train_stagea_probes.py`, and
`train/train_fusion.py` consume it unchanged.

Fidelity notes, recorded in metadata:

* Prompts use E2's user-only chat template (`uncertainty_profile.format_problem`), not the
  probe baseline's system prompt, because that is the prompt the trace was generated from.
* E2 kept decoded text, not token ids, so the trace is re-tokenised. The ids can differ
  slightly from the generated ones at token boundaries.
* Sequences are capped at --max-length tokens; a longer trace loses its end.

Run (after the E2 outputs are downloaded):

    python datagen/extract_trace_states.py \
        --model-id deepseek-ai/DeepSeek-R1-0528-Qwen3-8B \
        --traces "runs/e2f-stageb-fp16-s*/features/stageb-fp16-deepseek/shard_*/records.jsonl" \
        --out-dir features/tracestate-deepseek
"""

from __future__ import annotations

import argparse
import glob
import hashlib
import json
import platform
import sys
import time
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import torch
import transformers

DATAGEN = Path(__file__).resolve().parent
ROOT = DATAGEN.parent
UNCERTAINTY_SOLUTION = ROOT / "getting-started" / "solutions" / "uncertainty-profiling"
for import_path in (DATAGEN, UNCERTAINTY_SOLUTION):
    if str(import_path) not in sys.path:
        sys.path.insert(0, str(import_path))

from extract_features import DEFAULT_INSTANCES, assign_shards  # noqa: E402
from uncertainty_profile.extraction import (  # noqa: E402
    format_problem,
    get_model_input_device,
    load_model_and_tokenizer,
)

POOLINGS = ("prompt_last", "trace_mean", "trace_first_third", "trace_last_third", "trace_last")
PROMPT_FORMAT = "uncertainty_profile.format_problem: user-only chat template, add_generation_prompt=True"
MAX_PROMPT_TOKENS = 2048  # E2's max_prompt_length


def load_traces(pattern: str) -> dict[str, str]:
    """problem_key -> generated text, from any records.jsonl carrying both fields."""
    paths = sorted(glob.glob(pattern))
    if not paths:
        raise SystemExit(f"no trace records match {pattern}")
    traces: dict[str, str] = {}
    for path in paths:
        for line in Path(path).read_text(encoding="utf-8").splitlines():
            if line.strip():
                row = json.loads(line)
                traces[row["problem_key"]] = row["generated_text"]
    return traces


def pool_layer(hidden: torch.Tensor, n_prompt: int) -> tuple[dict[str, torch.Tensor], int]:
    """Segment poolings for one layer's (1, seq_len, hidden_dim) activations, moved to CPU."""
    tokens = hidden[0].float()
    n_nonfinite = int((~torch.isfinite(tokens)).sum().item())
    trace = tokens[n_prompt:]
    if len(trace) == 0:  # empty or fully truncated trace: fall back to the prompt state
        trace = tokens[n_prompt - 1 : n_prompt]
    third = max(1, len(trace) // 3)
    return {
        "prompt_last": tokens[n_prompt - 1].cpu(),
        "trace_mean": trace.mean(dim=0).cpu(),
        "trace_first_third": trace[:third].mean(dim=0).cpu(),
        "trace_last_third": trace[-third:].mean(dim=0).cpu(),
        "trace_last": trace[-1].cpu(),
    }, n_nonfinite


def layer_modules(model) -> tuple[list, bool]:
    """Modules to hook, and whether the final norm is among them.

    `output_hidden_states` does not return each layer's raw output: transformers appends every
    layer's *input*, then the last layer's output *after* the model's final norm. So the tuple
    is (embeddings, out_1, ..., out_{N-1}, norm(out_N)). Hooking the norm as well, and dropping
    the raw final layer, reproduces that sequence exactly — which matters because Stage A (E1b)
    used the standard path.
    """
    decoder = model.get_decoder() if hasattr(model, "get_decoder") else getattr(model, "model", None)
    layers = getattr(decoder, "layers", None)
    if layers is None:
        raise SystemExit(f"cannot locate decoder layers on {type(model).__name__}; use --no-hooks")
    norm = getattr(decoder, "norm", None)
    modules = [model.get_input_embeddings(), *layers]
    if norm is not None:
        modules.append(norm)
    return modules, norm is not None


@torch.inference_mode()
def pooled_states(model, input_ids: torch.Tensor, n_prompt: int, use_hooks: bool):
    """Per-layer segment poolings as (n_layers, hidden_dim) float16 arrays, plus non-finite count."""
    if not use_hooks:
        output = model(input_ids=input_ids, output_hidden_states=True, use_cache=False)
        collected, n_nonfinite = [], 0
        for hidden in output.hidden_states:
            pooled, bad = pool_layer(hidden, n_prompt)
            collected.append(pooled)
            n_nonfinite += bad
        del output
    else:
        collected, nonfinite = [], [0]

        def hook(_module, _args, output):
            hidden = output[0] if isinstance(output, tuple) else output
            pooled, bad = pool_layer(hidden.detach(), n_prompt)
            collected.append(pooled)
            nonfinite[0] += bad

        modules, has_norm = layer_modules(model)
        handles = [module.register_forward_hook(hook) for module in modules]
        try:
            # Hidden states are pooled and released layer by layer; keep only the last logit.
            try:
                model(input_ids=input_ids, use_cache=False, logits_to_keep=1)
            except TypeError:
                model(input_ids=input_ids, use_cache=False)
        finally:
            for handle in handles:
                handle.remove()
        if has_norm:
            # Keep norm(out_N), drop the raw out_N, matching the hidden_states convention.
            del collected[-2]
        n_nonfinite = nonfinite[0]

    return {
        name: torch.stack([layer[name] for layer in collected]).numpy().astype(np.float16)
        for name in POOLINGS
    }, n_nonfinite


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model-id", required=True, help="must be the model that generated the traces")
    parser.add_argument("--traces", required=True, help="glob of Stage B records.jsonl files")
    parser.add_argument("--instances", type=Path, default=DEFAULT_INSTANCES)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--shard-strategy", default="roundrobin", choices=["roundrobin", "hash"])
    parser.add_argument("--max-length", type=int, default=2048 + 4096, help="prompt + trace token cap")
    parser.add_argument("--limit", type=int, default=None, help="debug: stop after N instances")
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument(
        "--no-hooks",
        action="store_true",
        help="debug: materialise every layer with output_hidden_states (needs ~1.9 GB at 6k tokens)",
    )
    parser.add_argument(
        "--dtype",
        default="auto",
        choices=["auto", "float16"],
        help="auto follows the starter kit (bf16 whenever torch reports it, emulated and slow on a T4); "
        "match the dtype the traces were generated in",
    )
    parser.add_argument(
        "--filter-model",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="keep only instances whose model_id equals --model-id (default); see extract_features.py",
    )
    args = parser.parse_args()
    if not 0 <= args.shard_index < args.num_shards:
        raise SystemExit("--shard-index must satisfy 0 <= index < --num-shards")

    traces = load_traces(args.traces)
    rows = pq.read_table(args.instances).to_pylist()
    if args.filter_model:
        rows = [r for r in rows if r["model_id"] == args.model_id]
    else:
        seen: set[str] = set()
        rows = [r for r in rows if not (r["problem_key"] in seen or seen.add(r["problem_key"]))]
    missing = [r["problem_key"] for r in rows if r["problem_key"] not in traces]
    rows = [r for r in rows if r["problem_key"] in traces]
    if not rows:
        raise SystemExit("no instances have a trace")
    print(f"{len(rows)} instances with traces ({len(missing)} without, skipped)")

    assignment = assign_shards([r["problem_key"] for r in rows], args.num_shards, args.shard_strategy)
    shard = sorted((r for r in rows if assignment[r["problem_key"]] == args.shard_index), key=lambda r: r["problem_key"])
    if args.limit:
        shard = shard[: args.limit]

    out_dir = args.out_dir / f"shard_{args.shard_index:02d}_of_{args.num_shards:02d}"
    vectors_dir = out_dir / "vectors"
    vectors_dir.mkdir(parents=True, exist_ok=True)
    done = {p.stem for p in vectors_dir.glob("*.npz")}
    todo = [r for r in shard if r["problem_key"] not in done]
    print(f"shard {args.shard_index}/{args.num_shards}: {len(shard)} instances, {len(done)} done, {len(todo)} to do", flush=True)
    if not todo:
        print("nothing to do")
        return

    started = time.time()
    if args.dtype == "float16":
        # Same override as extract_generation_features.py: hide emulated bf16 from the loader.
        reported = torch.cuda.is_bf16_supported
        torch.cuda.is_bf16_supported = lambda *_, **__: False
        try:
            model, tokenizer = load_model_and_tokenizer(args.model_id, local_files_only=args.local_files_only)
        finally:
            torch.cuda.is_bf16_supported = reported
    else:
        model, tokenizer = load_model_and_tokenizer(args.model_id, local_files_only=args.local_files_only)
    device = get_model_input_device(model)
    print(f"loaded {args.model_id} dtype={model.dtype} in {time.time() - started:.0f}s", flush=True)

    metadata = {
        "stage": "E4 trace states",
        "model_id": args.model_id,
        "dtype": str(model.dtype),
        "gpus": [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())],
        "poolings": list(POOLINGS),
        "prompt_format": PROMPT_FORMAT,
        # merge_features.py checks this key for cross-shard consistency.
        "system_prompt_sha1": hashlib.sha1(PROMPT_FORMAT.encode()).hexdigest(),
        "trace_retokenised": True,
        "pooling_via_hooks": not args.no_hooks,
        "traces": args.traces,
        "max_length": args.max_length,
        "num_shards": args.num_shards,
        "shard_index": args.shard_index,
        "shard_strategy": args.shard_strategy,
        "n_instances": len(shard),
        "n_instances_without_trace": len(missing),
        "torch": torch.__version__,
        "transformers": transformers.__version__,
        "platform": platform.platform(),
    }

    loop_started = time.time()
    for position, row in enumerate(todo, start=1):
        prompt = format_problem(tokenizer, row["problem"])
        prompt_ids = tokenizer(prompt, truncation=True, max_length=MAX_PROMPT_TOKENS)["input_ids"]
        trace_ids = tokenizer(traces[row["problem_key"]], add_special_tokens=False)["input_ids"]
        ids = (prompt_ids + trace_ids)[: args.max_length]
        n_prompt = min(len(prompt_ids), len(ids))

        pooled, n_nonfinite = pooled_states(
            model, torch.tensor([ids], device=device), n_prompt, use_hooks=not args.no_hooks
        )
        if n_nonfinite:
            print(f"  warning: {n_nonfinite} non-finite activations for {row['problem_key']}")

        np.savez_compressed(vectors_dir / f"{row['problem_key']}.npz", **pooled)
        record = {
            "problem_key": row["problem_key"],
            "model_id": row["model_id"],
            "extracted_model_id": args.model_id,
            "is_robust": bool(row["is_robust"]),
            "dataset_id": row["dataset_id"],
            "n_prompt_tokens": n_prompt,
            "n_trace_tokens": len(ids) - n_prompt,
            "truncated": len(prompt_ids) + len(trace_ids) > args.max_length,
            "n_nonfinite_activations": n_nonfinite,
        }
        with (out_dir / "records.jsonl").open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record) + "\n")

        if position == 1 or position % 5 == 0 or position == len(todo):
            first = next(iter(pooled.values()))
            metadata.update(n_layers=int(first.shape[0]), hidden_dim=int(first.shape[1]))
            metadata["seconds_so_far"] = round(time.time() - loop_started, 1)
            (out_dir / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
            rate = (time.time() - loop_started) / position
            print(
                f"  [{position}/{len(todo)}] {len(ids)} tokens, {rate:.1f}s/instance, "
                f"~{rate * (len(todo) - position) / 60:.1f} min left",
                flush=True,
            )

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    print(f"\ndone in {(time.time() - started) / 60:.1f} min -> {out_dir}")


if __name__ == "__main__":
    main()
