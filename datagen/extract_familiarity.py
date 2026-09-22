"""E23: does the model already know the problem? Familiarity and contamination scores.

A spurious case is one the model solves as written but not once it is perturbed, which is
what a memorized problem looks like. The organizers' own report splits results by published
vs unpublished problems. E12 measured only mean problem NLL (Finding 22), and mean NLL mostly
reads how predictable the text is. These are the membership-inference scores built to separate
familiarity from predictability, all from **one forward pass over the bare problem text** (no
chat template: the statement as it would appear in a training corpus):

* `mean_nll` - E12's familiarity, for reference;
* `min_k_10`, `min_k_20` - mean NLL of the 10% / 20% least likely tokens (Min-K% Prob), which
  ignores the easy tokens every model predicts;
* `min_k_pp_20` - Min-K%++: each token's log-probability standardised by the mean and spread of
  the model's own next-token distribution at that position, then the lowest 20% averaged;
* `zlib_ratio` - total NLL over the zlib-compressed length, dividing out plain compressibility;
* `suffix_nll` - mean NLL of the second half given the first, i.e. how well the model completes
  a statement it has started reading.

The relative score (one model's NLL against the other three) is computed offline from four
runs. Problems: every distinct train problem in `data/small_track/` plus all 36 problems of
`train-main-v2`. Labels are joined offline by `problem_key`, so no label is read here.

Run:  python datagen/extract_familiarity.py --model-id Qwen/Qwen3.5-4B --out-dir features/familiarity-qwen35
"""

from __future__ import annotations

import argparse
import json
import platform
import time
import zlib
from pathlib import Path

import pandas as pd
import torch
from huggingface_hub import hf_hub_download
from transformers import AutoModelForCausalLM

from extract_features import DEFAULT_INSTANCES, assign_shards, resolve_dtype
from ingest_public import problem_key
from tokenizer_check import load_checked_tokenizer

OFFICIAL = ("aimo-interp/train-main-v2", "data/train-00000-of-00001.parquet", "23791d9b120132087877b0459f953d5959fc8455")
MAX_TOKENS = 2048


def load_problems(instances: Path) -> pd.DataFrame:
    """Every distinct problem we hold labels for, anywhere: our train set plus `train-main-v2`."""
    ours = pd.read_parquet(instances)[["problem_key", "problem", "dataset_id"]]
    repo, filename, revision = OFFICIAL
    official = pd.read_parquet(hf_hub_download(repo, filename, repo_type="dataset", revision=revision))
    official = pd.DataFrame({"problem": official["problem"].unique()})
    official["problem_key"] = official["problem"].map(problem_key)
    official["dataset_id"] = "train-main-v2"
    both = pd.concat([official, ours], ignore_index=True)  # official first, so it wins a duplicate
    return both.drop_duplicates("problem_key").reset_index(drop=True)


def lowest_mean(values: torch.Tensor, share: float) -> float:
    k = max(1, int(round(share * values.numel())))
    return float(values.topk(k, largest=False).values.mean())


@torch.no_grad()
def familiarity(model, tokenizer, problem: str) -> dict:
    ids = tokenizer(problem, return_tensors="pt", truncation=True, max_length=MAX_TOKENS)["input_ids"].to(model.device)
    logits = model(input_ids=ids, use_cache=False).logits[0, :-1].float()
    log_probs = torch.log_softmax(logits, dim=-1)
    targets = ids[0, 1:].to(log_probs.device)
    token_logp = log_probs.gather(1, targets[:, None])[:, 0]
    # Min-K%++: standardise each token's log-prob by the next-token distribution it came from.
    probs = log_probs.exp()
    mu = (probs * log_probs).sum(-1)
    sigma = ((probs * log_probs.square()).sum(-1) - mu.square()).clamp_min(1e-8).sqrt()
    half = token_logp.numel() // 2
    nll = -token_logp
    return {
        "n_tokens": int(ids.shape[1]),
        "mean_nll": float(nll.mean()),
        "min_k_10": -lowest_mean(token_logp, 0.10),
        "min_k_20": -lowest_mean(token_logp, 0.20),
        "min_k_pp_20": -lowest_mean((token_logp - mu) / sigma, 0.20),
        "zlib_ratio": float(nll.sum()) / len(zlib.compress(problem.encode("utf-8"))),
        "suffix_nll": float(nll[half:].mean()) if half else float(nll.mean()),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model-id", required=True)
    parser.add_argument("--instances", type=Path, default=DEFAULT_INSTANCES)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--dtype", default="auto", choices=["auto", "float16", "bfloat16", "float32"])
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--local-files-only", action="store_true")
    args = parser.parse_args()

    problems = load_problems(args.instances)
    assignment = assign_shards(list(problems["problem_key"]), args.num_shards, "roundrobin")
    shard = problems[problems["problem_key"].map(assignment) == args.shard_index].sort_values("problem_key")
    shard = shard.head(args.limit) if args.limit else shard

    out_dir = args.out_dir / f"shard_{args.shard_index:02d}_of_{args.num_shards:02d}"
    out_dir.mkdir(parents=True, exist_ok=True)
    records_path = out_dir / "records.jsonl"
    done = set()
    if records_path.is_file():
        done = {json.loads(line)["problem_key"] for line in records_path.read_text(encoding="utf-8").splitlines() if line}
    todo = shard[~shard["problem_key"].isin(done)]
    print(f"{len(shard)} problems, {len(done)} done, {len(todo)} to do")
    if todo.empty:
        return

    dtype = resolve_dtype(args.dtype)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer, tokenizer_info = load_checked_tokenizer(args.model_id, local_files_only=args.local_files_only)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_id, dtype=dtype, device_map="auto" if device == "cuda" else None,
        local_files_only=args.local_files_only,
    ).eval()

    started = time.time()
    with records_path.open("a", encoding="utf-8") as handle:
        for row in todo.itertuples(index=False):
            scores = familiarity(model, tokenizer, row.problem)
            record = {"problem_key": row.problem_key, "dataset_id": row.dataset_id, "scored_model_id": args.model_id, **scores}
            handle.write(json.dumps(record) + "\n")
            handle.flush()
    (out_dir / "metadata.json").write_text(json.dumps({
        "model_id": args.model_id,
        "dtype": str(dtype),
        "gpu": torch.cuda.get_device_name(0) if device == "cuda" else None,
        "n_gpus": torch.cuda.device_count(),
        "torch": torch.__version__,
        "transformers": __import__("transformers").__version__,
        "platform": platform.platform(),
        "max_tokens": MAX_TOKENS,
        "seconds_per_problem": (time.time() - started) / len(todo),
        **tokenizer_info,
    }, indent=2), encoding="utf-8")
    print(f"{len(todo)} problems in {time.time() - started:.0f}s")


if __name__ == "__main__":
    main()
