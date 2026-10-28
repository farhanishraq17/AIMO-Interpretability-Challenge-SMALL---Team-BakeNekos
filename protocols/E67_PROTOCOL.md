# E67: does E64's difficulty direction cause longer reasoning? A steering test of Finding 93

Status: **pre-registered 28 October 2026, 07:50 UTC, before any E67 sample is drawn.** Report experiment: it decides
no submission and touches neither the Form ZIP (E64 stays) nor Codabench. Code: `datagen/perturb/e67_pool.py`
(direction and problems), `datagen/perturb/steer.py` (sampling with the push), `scripts/e67_gpu_bundle.py` (kernels
`e67-a` to `e67-d`; `e67-timing` ran first, §6), `train/eval_e67.py` (gates; written and run on synthetic records
before any sample). Any change after the first E67 sample is read is an amendment, dated in §9.

## Motivation

Finding 93 opened E64 up: a ridge reads difficulty from a 20-50-dimensional subspace of the prompt states, present at
every depth and best mid-depth, read alike by all four models. That is a readout. It does not say the model *uses*
that subspace to decide how long to think. If it does, pushing a prompt's states along E64's direction should lengthen
Qwen's traces and pushing against it should shorten them, while a push of the same size in a random direction should
do neither. If the push moves nothing, E64 reads a correlate of difficulty (Finding 93's "kind of task") that the
model's later computation does not consume along that direction.

## 1. Direction

From `runs/e64/artifact.npz` (E64's shipped Qwen arrays, unchanged) by `datagen/perturb/e67_pool.py`:

- **Layer: hidden-state index 15** of 33 (index 0 is the embeddings), the output of decoder layer 15 of 32. Relative
  depth 15/32 = 0.47, Qwen's best single layer in Finding 93.
- **u** = `qwen__coef[15] / qwen__scale[15]` (the ridge's direction in raw state space), normalised. Its norm |w| is
  7.587: E64's layer-15 read is w · x plus a constant.
- **sigma = 0.0719**, the SD over E55's 548 Qwen prompt states (`X_mean`, layer 15) of their projection on u. A push of
  2 sigma moves the layer-15 read by 2 x 0.0719 x 7.587 = **1.091 log-tokens, two SDs of that read over E55**: it
  moves a problem along u as far as the spread between problems does. It is a natural-scale push, not a large one
  (2 sigma = 0.144, against a mean prompt state of norm 5.5 at that layer), and the null it may return is a null at
  that dose (§5).
- **r**: a Gaussian direction (seed 67), orthogonalised against u and normalised. The control push is 2 sigma r, the
  same norm.

## 2. Problems

`data/e67/pool.parquet`: E62's 497-problem pool (no `train-main-v2` problem; 0 overlap re-checked), minus the 141
whose two E62 base samples both hit the 32k cap, ranked by E64's Qwen predicted difficulty
(`runs/e64/predicted_difficulty.csv`); the **60 closest to log(8192) = 9.01**, a predicted trace of half E67's cap.

The brief asked for mid predicted difficulty. The pool's middle cannot serve at a 16k cap: E62's median Qwen trace is
21,752 tokens, so at the pool's median a lengthening push would be cut off by the cap. "Mid" is therefore taken
relative to the cap, so a push either way has room. The 60: predicted difficulty 8.79-9.23; E62 traces median 7,168
tokens (10th-90th percentile 4,020-11,329), 2 of their 120 E62 samples at or above 16k; E62 accuracy 0.98;
OlympiadBench 51, AMC 8, AIME 1. Selecting on E62's lengths is selection on an independent earlier run; every
contrast is paired against the unpushed condition sampled in the same batch, so regression to the mean cancels.

## 3. Sampling

Qwen3.5-4B under transformers (vLLM cannot take a forward hook): E64's evaluation pins (transformers 5.13.0) on
Kaggle's torch, **fp16**, one process per T4, both T4s of a kernel. transformers' torch fallback runs the linear-
attention layers (no `fla` or `causal-conv1d` installed).

- **E62's base recipe:** the prompt that can decline (`sample.py`'s `NO_SOLUTION_SUFFIX`, user turn, chat template
  with the generation prompt), temperature 1.0, top-k 40, top-p 0.95, stop ids from the tokenizer and the generation
  config. **Cap: 16,384 new tokens. k = 2 per condition.**
- **Conditions:** `minus` (-2 sigma u), `zero` (no push), `plus` (+2 sigma u), `random` (+2 sigma r).
- **The push:** a forward hook on decoder layer 15 adds the condition's vector to the residual stream at every
  position of a multi-token pass (the prefill: every prompt token) and leaves single-token decode steps alone. The
  push reaches the trace only through the prompt's cached keys and values and the linear layers' recurrent state.
- **Attention:** transformers' sdpa, except at decode, where an exact grouped form avoids copying KV heads (§6).
- **Batching:** each problem is one batch of 4 conditions x 2 samples = 8 rows of the same prompt, so no padding
  reaches the hook and every condition shares one prefill. Seed per problem: 67 + int(first 8 hex digits of its key).
- **Sanity states:** before sampling, the same hook runs on E64's own input, the E55 prompt (the probe's system
  prompt, chat template, 2,048-token truncation), once per condition; the mean-pooled states at all 33 indices are
  saved (float16, as E64's `mean_states` stores them) and read locally with E64's arithmetic.
- **Kernels:** `e67-a` to `e67-d`, the pool dealt round-robin in `problem_key` order (15 problems each, half per
  T4), `--max-hours 9` per process with the guard at 9.5 h, on accounts with verified GPUs (never RSD; MUBAMAN only
  after a passing capability check). A kernel stopped short may be resumed once from its own downloaded records.

## 4. Outcome and gates

Per problem and condition, **y = mean over its 2 samples of log(min(tokens, 16,384))** (E64's target, at E67's cap).
Contrasts are per-problem differences against `zero`, averaged over problems, with **95% paired problem-bootstrap
intervals** (2,000 draws, seed 67).

| Gate | Contrast | Pass |
| --- | --- | --- |
| G1 lengthening | `plus` - `zero` | interval **above 0** |
| G2 shortening | `minus` - `zero` | interval **below 0** |
| G3 control | `random` - `zero` | interval **contains 0** |
| S sanity | E64's Qwen read of the steered E55 states, averaged over the layers past the steered one (indices 16-32): `plus` - `zero` and `minus` - `zero` | first **above 0**, second **below 0** |

S is read past the steered layer because the steered layer's own read moves by construction (1.091 / 33 = 0.033 of
the all-layer average); only propagation can move the later layers.

Reported only: `plus` - `minus`; E64's read averaged over all layers; accuracy, cap hits and declined answers
(`eval_e52.grade`) by condition; the parity of the unsteered read with E64's offline predictions (fp16 here, bf16 in
E55); run time.

**Population:** every problem with a complete record downloaded by **31 October 2026, 12:00 UTC**. Fewer than 45:
reported as underpowered, with no reading.

## 5. Reading

- **G1, G2 and G3 pass:** the direction E64 reads is a lever on how long Qwen reasons, specific to that direction.
  Finding 93's subspace is then partly mechanism, not only correlate.
- **G3 passes, G1 or G2 fails, S passes:** the push propagates (later layers read the prompt as harder or easier)
  and the trace length does not follow: E64 reads a correlate the model does not consume along u, at this dose.
- **S fails:** later layers absorb the push; at this dose the direction does not reach the computation, and a length
  null says nothing more.
- **G3 fails:** a push of this norm in any direction changes length; G1 and G2 cannot be read as specific.
- One of G1 and G2 alone: reported as one-sided.

## 6. Cost (measured before this protocol)

`e67-timing` (`kernels/runs/e67-timing-Nipun/` holds v2): two E62 problems whose base samples both hit the 32k cap
(E67 excludes them), one per T4, through `steer.py` at E67's settings. Two versions, both on Nipun:

- **v1 (27 Oct, 19:52 UTC):** both T4s ran out of memory at about 13k tokens of context. With no mask at decode,
  transformers' sdpa passes `enable_gqa`, and on a T4 sdpa then copies K and V for all 16 query heads every step
  (820 MB at batch 8 and 13k tokens). `steer.py` now replaces sdpa at decode only by an exact equivalent: one query
  token per row lets each KV group's 4 query heads stand in for query positions over their shared head, so nothing is
  copied (`datagen/perturb/test_steer.py`: the same sampled tokens as sdpa on a tiny random Qwen3.5, scores within
  1.2e-7). The children also run with `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`.
- **v2 (27 Oct, 20:27 UTC):** one T4 ran its problem with all 8 rows to the cap: **16,384 decode steps in
  1,762 s (107.6 ms a step, averaged over 0-16k tokens of context), peak 12.36 GB of 14.56.** The other process failed
  to load the checkpoint (`OSError` on a shard, while still making Hub requests after the kernel's own download): a
  Hub failure, not a missing file. E67's processes therefore load offline (`--local-files-only`, `HF_HUB_OFFLINE=1`).

**Cost:** at most 29.4 min a problem (every row to the cap), so at most 8 x 29.4 min = 3.9 h of wall time for a
kernel's busier T4, plus about 5 min of setup. Kaggle bills session time, not GPUs (the two timing runs, about 31 min
each, moved Nipun's use from 11.9 h to 12.9 h). **At most about 16 GPU-hours for the four kernels**, well inside the
brief's ceiling of about 60, so its sizes stand (60 problems, k = 2, 16k cap). A 32k cap is not possible at this batch:
the full-attention KV cache of 8 rows at 32k (8.6 GB) and the fp16 weights do not fit on one T4.

## 7. What was read before this protocol

E62's base records (lengths, cap hits, correctness: used for §2), E64's arrays and offline predictions, E55's Qwen
prompt states (sigma), and from `e67-timing`: run time, decode steps, peak memory, dtype, and E64's read of the saved
states of its two problems. The unsteered read matched E64's offline value within 0.0023 (fp16 against bf16). The same
print showed the steered reads on those two excluded problems: past layer 15, about +0.18 for `plus`, -0.18 for
`minus` and +0.007 for `random`. That is S's quantity on two problems outside E67; it changed no choice here. Of
their samples, only the decode-step count was read (v2's finished problem: every row at the cap). No sample of an E67
problem exists.

## 8. Not done

No other layer, push size, direction, cap, k or population is tried after a sample is read, and no variant is chosen
by any read. No other model is steered. The samples label nothing and enter no submission.

## 9. Amendments

None.
