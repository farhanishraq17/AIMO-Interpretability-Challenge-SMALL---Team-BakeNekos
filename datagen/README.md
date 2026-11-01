# `datagen/` — data consolidation and feature extraction

Two pipelines: one builds the labelled training set from public data (no GPU), the other
extracts model features for it (GPU, shardable across the Kaggle fleet).

## 1. Build the training set (local, no GPU, ~30 s)

```bash
cd getting-started
uv run python ../datagen/ingest_public.py
```

Pulls all six public `aimo-interp` datasets, normalises model aliases, derives labels for
the two tables that ship decay columns but no label, applies the organizers' curation
rules, and writes `data/small_track/`:

| File | Contents |
| --- | --- |
| `perturbation_rows.parquet` | 574 rows, one per (model, problem, perturbation) |
| `instances.parquet` | 213 rows, one per (model, problem) — the evaluation unit |
| `instances_train.parquet` | **187** rows after eligibility + clarity filters |
| `splits.json` | 5 folds grouped by problem text |
| `summary.json` | provenance, counts, and the rules applied |

Labelling rules, all derived rather than assumed (see `PLAN.md` §10):

- `is_robust == (absolute_accuracy_decay <= 0)` — agrees with all 379 published labels.
  `<= 0` rather than `== 0` because 15 rows show the model *improving* under perturbation.
- Eligibility `base_accuracy >= 1.0` — the task only applies where the model already
  solves the original, and 295 of 298 published rows sit at exactly 1.0.
- Ambiguous band `0 < decay <= 0.3` excluded — published labels contain nothing there,
  matching the proposal's statement that non-clear-cut cases are dropped.

Output is git-ignored on purpose; the script regenerates it deterministically.

## 2. Extract features (GPU)

Stage A: one forward pass per problem, no generation. Produces pooled hidden states at
**every** layer plus next-token uncertainty scalars.

```bash
uv run python ../datagen/extract_features.py \
    --model-id deepseek-ai/DeepSeek-R1-0528-Qwen3-8B \
    --out-dir ../features/deepseek-8b \
    --num-shards 1 --shard-index 0
```

Prompts are built exactly as the shipped probe baseline builds them — same system prompt
(read from `probe_artifact.pkl` at runtime, so it cannot drift) and the same chat template
with `add_generation_prompt=True`. The prompt's SHA-1 is recorded in `metadata.json`, and
`merge_features.py` refuses to merge shards whose prompt, dtype, or layer count disagree.

Model filtering is **on by default**: `is_robust` is specific to a (model, problem) pair,
so extracting one model's activations against another model's label would silently
mislabel the data. `--no-filter-model` exists only for pipeline smoke tests and
deduplicates by problem.

### How many accounts to use

Check the workload before spending GPU time:

```bash
uv run python ../datagen/plan_shards.py --model-id deepseek-ai/DeepSeek-R1-0528-Qwen3-8B
```

Stage A was measured at **1.0 s per instance** on Kaggle T4x2 in bf16 (E1): 2.4 minutes of
extraction for 138 instances, and about 5 minutes for the whole kernel including the 16 GB
model download. **Run Stage A on a single account**; sharding it would pay the download
several times over to save seconds. Sharding exists for Stage B (generation up to 4,096
tokens per sample), where per-instance cost is minutes and the fleet pays off.

When you do shard, `--shard-strategy roundrobin` (default) balances to within one
instance per shard; `hash` instead pins a problem to a fixed shard as the set grows.

Results land in `shard_XX_of_YY/` under `--out-dir`. Each instance is written as its own
`.npz` and appended to `records.jsonl` immediately, so a session killed at the 12-hour
limit loses nothing: rerunning the same command skips completed work.

On Kaggle T4 and P100 GPUs there is no hardware bf16, so `--dtype auto` resolves to
float16; the choice is recorded in `metadata.json`. fp16 can overflow on Qwen-family
models, so every record carries `n_nonfinite_activations`; check it before trusting the
features. If the probe turns out dtype-sensitive, re-extract with `--dtype bfloat16` on a
capable GPU.

### Stage B: generation-trace features

One greedy generation per problem (4,096-token cap), summarised into the
uncertainty-profiling baseline's 14 features, plus the generated text. It reuses the
starter kit's `uncertainty_profile` package unchanged, so features match what that
baseline computes at evaluation time.

```bash
uv run python ../datagen/extract_generation_features.py \
    --model-id deepseek-ai/DeepSeek-R1-0528-Qwen3-8B \
    --out-dir ../features/stageb-deepseek \
    --num-shards 2 --shard-index 0
```

Per-instance cost is minutes, so this is the stage worth sharding across accounts. Rows are
flushed after every batch, resume skips finished problems, and a preflight forward pass
aborts the run if fp16 logits are already non-finite (override with `--allow-nonfinite`).

### Running on Kaggle: `kaggle_launch.py`

Do not paste notebook cells. Experiments live in the `EXPERIMENTS` registry in
`datagen/kaggle_launch.py`, each pinned to one fleet account:

```bash
python datagen/kaggle_launch.py list                      # registry
python datagen/kaggle_launch.py build e1-stagea-deepseek  # render kaggle_runs/<name>/, no network
python datagen/kaggle_launch.py push e1-stagea-deepseek   # private GPU kernel (T4x2, internet on)
python datagen/kaggle_launch.py status                    # all experiments
python datagen/kaggle_launch.py logs e1-stagea-deepseek   # log tail; came back empty while running
python datagen/kaggle_launch.py output e1-stagea-deepseek # -> runs/<name>/
```

`push` renders a script kernel that embeds a base64 zip of exactly the files it needs
(about 63 KB), pip-installs the competition's library pins (keeping Kaggle's torch),
downloads the model to `/tmp/hf`, and runs the registry steps with logs in
`/kaggle/working/logs/` and a `run_manifest.json`. Stage A experiments also run
`merge_features.py` on Kaggle, because `kaggle kernels output` fetches a single page of at
most 200 files. To add an experiment, add a registry entry; to change code, edit the repo
and push again.

## 3. Merge shards

Download each account's output into one directory, then:

```bash
uv run python ../datagen/merge_features.py --features-dir ../features/deepseek-8b
```

Writes `features.npz` (`X_last`, `X_mean`, `X_max`, each `(n_instances, n_layers,
hidden_dim)`) and `records.parquet` with labels, uncertainty scalars, and fold ids.

Keeping every layer and three poolings is deliberate: the shipped probe reads only the
last prompt token at one layer and measures at chance (`PLAN.md` §9, Finding 2), so layer
and pooling should be selected on grouped CV rather than inherited from the baseline.

## 4. Later runs, outside the registry

Later experiments are not in `EXPERIMENTS`. Each is a self-contained bundle, `scripts/*_gpu_bundle.py` or
`scripts/e62_gen_bundle.py`, pushed and downloaded with `scripts/kaggle_followup.py`
(`push <job> --account <alias>`). Their pools and generators are in `datagen/perturb/`, and each run's launch
record is in `kernels/runs/`.
