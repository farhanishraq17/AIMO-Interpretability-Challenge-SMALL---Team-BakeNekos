# E66: E64 refitted on a difficulty target that is not cut off at 32k tokens

Status: **pre-registered 26 October 2026, 22:40 UTC, before any E66 sample is drawn.** Submission experiment: it may
replace E64 in the Google Form, under §6's rule only. Code: `datagen/perturb/e66_pool.py` (problems),
`scripts/e58_gpu_bundle.py` (kernels `e66-qwen-a` to `e66-qwen-p`), `train/eval_e66.py` (target, fit, gates),
`submission/make_e64.py --experiment e66` (ZIP), `scripts/e64_gpu_bundle.py --verify --experiment e66` (GPU states). Any change after the first sample is read is an amendment, dated in §8.

## Motivation

E64 learns difficulty from Qwen's mean log trace length on E62's 497 problems, sampled at a 32,768-token cap. **194 of
the 497 have at least one sample at the cap, and 141 have both**, so the top quarter of the target is flat at
log(32768) = 10.397. Qwen's and DeepSeek's cutoffs (10.386 and 10.393) sit inside that flat region: their robust/spurious
decisions are made exactly where the target cannot rank problems. The private test is 30 unpublished AIMO 3 problems,
expected to be at least as hard as the official ones, so most of its decisions fall there too. A cap of 65,536 tokens
restores resolution where the decisions are made. (Counts from the E62 base records; `train/eval_e66.py` recomputes them.)

E64 is board-verified (10/14) and in the Form. Nothing measured on 27 Oct beats it by more than noise, and this
change cannot be verified beyond the checks below, so the rule for replacing it is strict.

## 1. Problems

`data/e66/pool.parquet` from `datagen/perturb/e66_pool.py`: the rows of `data/e62/pool.parquet` whose E62 base record
has `n_hit_length >= 1` (**194 problems**), in `problem_key` order, split round-robin into 16 slices.

## 2. Sampling

Qwen3.5-4B with E62's base recipe unchanged: the prompt that can decline (`--allow-no-solution`), temperature 1.0,
top-k 40, top-p 0.95, vLLM 0.20, one engine per T4, fp16. Three changes: **cap 65,536 tokens**, **seed 66** (fresh
samples, independent of E62's), **`--chunk 2`** (records saved every two problems). k = 2 per problem. Kernels on
accounts with verified GPUs, `--max-hours 9` per kernel.

## 3. Target

- Re-sampled problems: mean over the two new samples of log(min(tokens, 65,536)).
- The other 303 problems keep E62's target (every sample below 32,768, so the cap changes nothing for them).
- A re-sampled problem with fewer than two new samples is dropped from training, and the count is reported. **If more
  than 10% of the 194 lack two new samples by 30 Oct 12:00 UTC, E66 stops and E64 stays.**

## 4. Fit

E64's fit, unchanged except for the target: per model, per-layer `StandardScaler` + `RidgeCV` (alphas
`logspace(1, 5, 9)`) on E55's `X_mean` prompt states, predictions averaged over layers. Cutoffs at **E64's own shares**
(Qwen 0.6, DeepSeek 2/3, Skywork 1/3, Olmo 1/3) of predicted difficulty over the 36 `train-main-v2` problems. No
change to shares, pooling, layers, alphas or folds.

## 5. Gates

| Gate | Measure | Pass |
| --- | --- | --- |
| G1 resolution (label-free) | share of the 497 with both samples at the target's cap | at most half of E64's 141/497, i.e. **<= 0.142** |
| G2 fit (label-free) | 5-fold CV Spearman of predicted against observed target, E64's folds (seed 64) | **>= 0.30 for every model** |
| G3 official ranking | AUC of -(predicted difficulty) on the 34 labelled small-track rows | **>= 0.883 (E64's)** |
| G4 official decisions | accuracy of the shipped decisions on the same 34 rows | **>= 0.824 (E64's)** |
| G5 contract | shipped arithmetic matches offline on all 548 problems x 4 models (max abs diff < 1e-3); the same check on the four checkpoints' GPU states saved by E64's check (`runs/e64-gpu-check-Nipun/`); `run_local.py --small`: coverage 1.0, 0 invalid | **all pass** |
| G6 board | one upload, after G1-G5 pass | **>= 10/14 (E64's read)** |

Reported only: D3 (Spearman with mean NLL), the E65 check on DeepSeek MATH, how many of the 36 official problems change
decision per model, and the cap-hit rate of the new samples.

## 6. Ship rule

**E66 replaces E64 in the Google Form only if G1 to G6 all pass by 31 Oct 12:00 UTC** and the organizers accept a
replacement Form response. Anything else (a failed gate, Codabench unreachable for G6, an incomplete run, no
replacement allowed) keeps E64. If E66 does not replace E64 after its board upload, E64 is uploaded again so the board
ends on the Form ZIP. G3 and G4 ask for non-inferiority on rows E64 was already scored on, so an equally good E66
fails them about half the time; that is the intended strictness. E66 can only win by being at least as good on every
measurement that exists, with the structural argument above as the reason to prefer it.

## 7. Not done

No second variant is built from these samples: no other shares, poolings, caps, layer choices or alphas, and no
choice among variants by any read. The samples are not used to label robustness.

## 8. Amendments

None.
