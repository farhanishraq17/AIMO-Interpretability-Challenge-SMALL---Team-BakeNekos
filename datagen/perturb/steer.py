"""E67: sample Qwen3.5-4B with E64's difficulty direction added to the prompt's residual stream.

E64 reads difficulty from mean-pooled prompt states through a per-layer ridge (Finding 93). Whether that direction
*causes* longer reasoning is a steering question (`protocols/E67_PROTOCOL.md`): add `alpha * sigma * u` at one layer,
at every prompt token, during prefill only, and sample under E62's base recipe. Decode steps are left alone, so the
push reaches the trace only through the prompt's cached keys, values and recurrent state.

Each problem is one batch: every condition (-alpha, 0, +alpha along u, +alpha along the random control r) times k
samples, all on the same prompt, so no padding enters the hook and every condition shares one prefill. vLLM cannot
take a forward hook, so this generates with transformers. Before sampling, the same hook runs on the E55 prompt (the
probe's system prompt, E64's input) and the steered mean states are saved, so E64's own arithmetic can read them
locally (`train/eval_e67.py`, the protocol's sanity check) without its arrays riding inside the kernel.

Output: <out-dir>/shard_XX_of_YY/records.jsonl, one line per problem, flushed as each finishes and resumable;
states/<problem_key>.npz (conditions x layers x hidden, float16, as E64's `mean_states` stores them) and
metadata.json beside it.

Run (Kaggle, one process per T4):
    python datagen/perturb/steer.py --pool data/e67/pool.parquet --directions data/e67/directions.npz \\
        --system-prompt-file data/probe_system_prompt.txt --out-dir /kaggle/working/e67-a
"""

from __future__ import annotations

import argparse
import json
import platform
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

PERTURB = Path(__file__).resolve().parent
for import_path in (PERTURB, PERTURB.parent):
    if str(import_path) not in sys.path:
        sys.path.insert(0, str(import_path))

from answers import extract_answer  # noqa: E402
from extract_features import build_prompt  # noqa: E402
from sample import NO_SOLUTION_SUFFIX, encode_prompt, stop_token_ids  # noqa: E402
from tokenizer_check import load_checked_tokenizer  # noqa: E402

CONDITIONS = ("minus", "zero", "plus", "random")
MAX_LENGTH = 2048  # E55's prompt truncation (extract_features.py), for the sanity read only


class Steer:
    """Forward hook on one decoder layer: adds `delta[row]` to every position of a multi-token (prefill) pass."""

    def __init__(self):
        self.delta = None

    def __call__(self, module, args, output):
        hidden = output[0] if isinstance(output, tuple) else output
        if self.delta is None or hidden.shape[1] == 1:  # decode steps are untouched
            return None
        hidden = hidden + self.delta[:, None, :].to(hidden.device, hidden.dtype)
        return (hidden, *output[1:]) if isinstance(output, tuple) else hidden


def grouped_decode_sdpa():
    """transformers' sdpa, except at decode: one query token per row lets a KV group's query heads stand in for query
    positions over their shared key and value head, so no head is copied. On a T4, sdpa's GQA path copies K and V for
    every head each step (820 MB at 13k tokens of context, batch 8: the `e67-timing` OOM). Same arithmetic, exact."""
    import torch
    from transformers import AttentionInterface
    from transformers.integrations.sdpa_attention import sdpa_attention_forward

    def forward(module, query, key, value, attention_mask, dropout=0.0, scaling=None, **kwargs):
        groups = getattr(module, "num_key_value_groups", 1)
        if query.shape[2] != 1 or groups == 1:
            return sdpa_attention_forward(module, query, key, value, attention_mask, dropout=dropout, scaling=scaling, **kwargs)
        batch, heads, _, dim = query.shape
        mask = None if attention_mask is None else attention_mask[:, :, :, : key.shape[-2]]  # (batch, 1, 1, keys)
        out = torch.nn.functional.scaled_dot_product_attention(query.reshape(batch, heads // groups, groups, dim), key, value,
                                                               attn_mask=mask, scale=scaling)
        return out.reshape(batch, heads, 1, dim).transpose(1, 2).contiguous(), None

    AttentionInterface.register("sdpa", forward)


def decoder_layers(model, n: int):
    import torch

    return next(m for name, m in model.named_modules()
                if isinstance(m, torch.nn.ModuleList) and len(m) == n and "visual" not in name)


def steered_states(model, tokenizer, steer: Steer, deltas, problem: str, system_prompt: str) -> np.ndarray:
    """E64's `mean_states` on the steered E55 prompt, one row per condition: (conditions, layers, hidden), float16."""
    import torch

    ids = tokenizer(build_prompt(tokenizer, problem, system_prompt), return_tensors="pt", truncation=True,
                    max_length=MAX_LENGTH)["input_ids"].to(model.device)
    steer.delta = deltas
    with torch.no_grad():
        output = model(input_ids=ids.repeat(len(deltas), 1), output_hidden_states=True, use_cache=False)
    steer.delta = None
    # E64's `mean_states`: mean over the prompt, stored as float16.
    return torch.stack([h.float().mean(dim=1) for h in output.hidden_states], dim=1).cpu().numpy().astype(np.float16)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model-id", default="Qwen/Qwen3.5-4B")
    parser.add_argument("--pool", type=Path, required=True)
    parser.add_argument("--directions", type=Path, required=True, help="data/e67/directions.npz (e67_pool.py)")
    parser.add_argument("--system-prompt-file", type=Path, required=True, help="the probe's system prompt, E64's input")
    parser.add_argument("--problem-keys", nargs="+", default=None)
    parser.add_argument("--alpha", type=float, default=2.0, help="push size in units of sigma")
    parser.add_argument("--k", type=int, default=2, help="samples per condition")
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--top-k", type=int, default=40)
    parser.add_argument("--max-new-tokens", type=int, default=16384)
    parser.add_argument("--dtype", default="float16", choices=["float16", "bfloat16", "float32"])
    parser.add_argument("--seed", type=int, default=67)
    parser.add_argument("--max-hours", type=float, default=None, help="stop starting new problems after this long")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()

    import torch
    import transformers
    from transformers import AutoModelForCausalLM

    pool = pd.read_parquet(args.pool).sort_values("problem_key")
    if args.problem_keys:
        pool = pool[pool["problem_key"].isin(set(args.problem_keys))]
    shard = pool.iloc[args.shard_index::args.num_shards]
    shard = shard.iloc[: args.limit] if args.limit else shard
    out_dir = args.out_dir / f"shard_{args.shard_index:02d}_of_{args.num_shards:02d}"
    (out_dir / "states").mkdir(parents=True, exist_ok=True)
    records_path = out_dir / "records.jsonl"
    done = {json.loads(line)["problem_key"] for line in records_path.read_text(encoding="utf-8").splitlines()
            if line.strip()} if records_path.is_file() else set()
    todo = shard[~shard["problem_key"].isin(done)]
    print(f"shard {args.shard_index}/{args.num_shards}: {len(shard)} problems, {len(done)} done, {len(todo)} to do", flush=True)
    if todo.empty:
        return

    started = time.time()
    tokenizer, tokenizer_info = load_checked_tokenizer(args.model_id, local_files_only=args.local_files_only)
    cuda = torch.cuda.is_available()
    grouped_decode_sdpa()
    model = AutoModelForCausalLM.from_pretrained(args.model_id, dtype=getattr(torch, args.dtype) if cuda else torch.float32,
                                                 device_map="auto" if cuda else None,
                                                 local_files_only=args.local_files_only).eval()
    directions = np.load(args.directions)
    layer, sigma = int(directions["layer"]), float(directions["sigma"])
    push = args.alpha * sigma
    u, r = (torch.tensor(directions[name], dtype=torch.float32) for name in ("u", "r"))
    deltas = torch.stack([-push * u, 0 * u, push * u, push * r])  # CONDITIONS order
    steer = Steer()
    n_layers = model.config.get_text_config().num_hidden_layers
    decoder_layers(model, n_layers)[layer - 1].register_forward_hook(steer)  # hidden_states[layer] is its output
    system_prompt = args.system_prompt_file.read_text(encoding="utf-8")
    stop = stop_token_ids(tokenizer, args.model_id, args.local_files_only)
    pad = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else stop[0]
    metadata = {"stage": "E67 steering", "args": {k: (str(v) if isinstance(v, Path) else v) for k, v in vars(args).items()},
                "conditions": CONDITIONS, "layer": layer, "sigma": sigma, "push_norm": push, "n_layers": n_layers,
                "dtype": str(model.dtype), "transformers": transformers.__version__, "torch": torch.__version__,
                "gpu": torch.cuda.get_device_name(0) if cuda else None, "prompt_suffix": NO_SOLUTION_SUFFIX,
                "stop_token_ids": stop, "load_seconds": round(time.time() - started, 1), "platform": platform.platform(),
                **tokenizer_info}
    (out_dir / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    for n, row in enumerate(todo.itertuples(index=False), 1):
        if args.max_hours and time.time() - started > args.max_hours * 3600:
            print(f"time budget of {args.max_hours} h reached; stopping (resumable)", flush=True)
            break
        began = time.time()
        if cuda:
            torch.cuda.reset_peak_memory_stats()
        np.savez_compressed(out_dir / "states" / f"{row.problem_key}.npz",
                            states=steered_states(model, tokenizer, steer, deltas, row.problem, system_prompt))
        ids = encode_prompt(tokenizer, row.problem, NO_SOLUTION_SUFFIX)
        input_ids = torch.tensor([ids] * (len(CONDITIONS) * args.k), device=model.device)
        torch.manual_seed(args.seed + int(row.problem_key[:8], 16))  # one stream per problem, whatever the order
        steer.delta = deltas.repeat_interleave(args.k, dim=0)
        with torch.inference_mode():
            output = model.generate(input_ids=input_ids, attention_mask=torch.ones_like(input_ids), do_sample=True,
                                    temperature=args.temperature, top_p=args.top_p, top_k=args.top_k,
                                    max_new_tokens=args.max_new_tokens, eos_token_id=stop, pad_token_id=pad)
        steer.delta = None
        generated = output[:, len(ids):].tolist()
        record = {"problem_key": row.problem_key, "pool_id": row.pool_id, "expected_answer": int(row.answer),
                  "completion_tokens": {}, "hit_length": {}, "answers": {}, "n_correct": {}, "sample_tails": {}}
        for i, condition in enumerate(CONDITIONS):
            rows = generated[i * args.k:(i + 1) * args.k]
            lengths = [next((j for j, token in enumerate(g) if token in stop), len(g)) for g in rows]
            texts = [tokenizer.decode(g[:n_tokens], skip_special_tokens=True) for g, n_tokens in zip(rows, lengths)]
            answers = [extract_answer(text) for text in texts]
            record["completion_tokens"][condition] = lengths
            record["hit_length"][condition] = [n_tokens >= args.max_new_tokens for n_tokens in lengths]
            record["answers"][condition] = answers
            record["n_correct"][condition] = sum(a == int(row.answer) for a in answers)
            record["sample_tails"][condition] = [text[-800:] for text in texts]
        record.update(prompt_tokens=len(ids), decode_steps=len(generated[0]), seconds=round(time.time() - began, 1),
                      peak_memory_gb=round(torch.cuda.max_memory_allocated() / 2**30, 2) if cuda else None)
        with records_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record) + "\n")
        print(f"  [{n}/{len(todo)}] {row.problem_key}: {record['decode_steps']} steps in {record['seconds']} s "
              f"({record['seconds'] / max(record['decode_steps'], 1) * 1e3:.1f} ms/step), peak {record['peak_memory_gb']} GB",
              flush=True)
    print(f"done in {(time.time() - started) / 60:.1f} min -> {out_dir}")


if __name__ == "__main__":
    main()
