# E64: predicted difficulty, read from one forward pass, as the per-problem score

Status: **pre-registered 22 October 2026, before any E64 regressor was fitted or any E64 number was read.** Code:
`train/eval_e64.py` (fit, checks, cutoffs, artifact), `submission/make_e64.py` (ZIP), `scripts/e64_gpu_bundle.py`
(parity and timing). Any change after the first number is read is an amendment, dated in §8.

## Motivation

Per-problem familiarity has a ceiling of about 10.3 of 14 (Finding 87), and it reverses on the organizers' 34 official
rows. The strongest exploratory signal on those rows was not familiarity but difficulty: **shorter reasoning traces go
with robust labels** (Finding 76, AUC 0.714 [0.446, 0.935], traces from E24). Traces cannot be generated inside the
budget, but difficulty may be readable from the prompt: E62 measured Qwen's trace lengths on 497 olympiad problems,
and E55 holds every model's one-pass prompt states on the same problems. E64 learns difficulty from those states,
with no robustness label in training, and asks whether predicted difficulty is the better per-problem score.

## 1. Target, features, fit

- **Target y (difficulty):** for each of E62's 497 problems, the mean over Qwen's two base samples of
  log(min(completion tokens, 32768)). Correctness is not used.
- **Features:** each model's own E55 prompt states, `X_mean` (mean-pooled over the prompt, every layer; the
  baseline system prompt and chat template), so every model reads difficulty from its own internals.
- **Fit, per model:** at every layer, standardise, then `RidgeCV` (alphas 10^1 ... 10^5, nine values, efficient
  leave-one-out); the score is the mean of the per-layer predictions (Finding 11's layer ensemble). Fitted on all 497
  problems for shipping.
- **Direction:** lower predicted difficulty = **robust** (Finding 76).

## 2. Decision

Robust iff predicted difficulty ≤ the model's `p_m` quantile of predicted difficulty over the 36 `train-main-v2`
problems (E63's calibration; no label read). `p_m` as in E61: Qwen 0.6, DeepSeek 2/3, Skywork and Olmo 1/3. Qwen is
included: difficulty is a different signal from familiarity, so E63's reason for Qwen's constant does not carry. Any
failure returns the model's constant.

## 3. Checks

| id | check | rule |
| --- | --- | --- |
| D1 | fit: 5-fold grouped CV (by `problem_key`, seed 64) Spearman between out-of-fold predicted and observed difficulty on the 497, per model | **gate:** ≥ 0.30 for every model |
| D2 | the 34 official rows: AUC of −(predicted difficulty) against the official label, 95% template-bootstrap interval (2,000 draws, seed 64) | **gate:** point AUC ≥ 0.60 (familiarity's is below 0.5 there) |
| D3 | Spearman between predicted difficulty and mean NLL, per model, on the 548 | reported (is it familiarity in disguise?) |
| D4 | DeepSeek MATH (142): balanced accuracy at DeepSeek's cutoff, E54's bootstrap | reported (different label mechanism, Finding 76) |
| C1, C5 | GPU: the ZIP's predicted difficulty against the offline value for five problems per model (max abs diff ≤ 1e-3), and 60 one-problem calls in under 5 minutes | must pass |
| C6 | `run_local.py --small` coverage 1.0, 0 invalid | must pass |

## 4. Board read and ship rule

One upload, only if D1, D2, C1, C5 and C6 pass. **E64 goes in the Google Form if its read is at least 10/14**. If E63
also reads at least 10/14, E64 is preferred because it passed D2 (it does not reverse on the organizers' own rows) and
E63 did not; this is fixed now and not reopened by comparing the two reads. Otherwise E63 or `e27f` per their rules.

## 5. Not done

No robustness label enters the fit. The 34 official rows are read once, for D2. No layer, alpha, target or quantile
changes after any number is read. No state across calls; no use of `train-main-v2` revision `25e45665`.

## 8. Amendments

**1 (22 Oct, before the first run): D4 is dropped.** E55's prompt states cover its 548 problems only; the 142 DeepSeek
MATH problems have no states in that format, so D4 cannot be computed. It was reported-only and gates nothing.

**2 (22 Oct, after the first run): one reported-only line added.** The accuracy of the shipped decisions on the same 34
rows, beside the constants'. It is computed after D1-D3 were read and decides nothing.

**3 (22 Oct, before the GPU run): C1 and C5 are split between GPU and CPU.** The ZIP's arrays are 5.2 MB, several times
the largest script the fleet has accepted (0.8 MB). So the GPU kernel runs the ZIP's own `_load` and `mean_states` on
the four checkpoints, saves five problems' states per model and times fifteen forward passes per model; the ZIP's own
`difficulty` read-out is then applied locally to those GPU states and compared with the offline predictions (same
1e-3 tolerance), and its cost is timed locally. Together these cover the shipped path; the pass rules are unchanged.

**4 (27 Oct, after the board read): the Docker load test is waived (owner's decision), as E61's amendment 1 waived
it.** Docker Desktop has not started on this machine since 21 Oct. The Codabench run (submission 1010961, the
organizers' GPU runner) had coverage 1.0 and 0 invalid predictions, and its 10/14 differs from the constants' 9/14,
so the shipped model path ran in the evaluation runtime rather than the fallback. `e64-difficulty` goes into the
Google Form unchanged (`dist/solution-e64-difficulty-small.zip`).
