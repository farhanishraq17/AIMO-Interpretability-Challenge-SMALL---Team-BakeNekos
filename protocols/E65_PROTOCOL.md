# E65: does E64's difficulty score hold on DeepSeek's MATH rows?

Status: **pre-registered 24 October 2026, before E64's score was computed on any MATH row.** Check experiment for E64;
it builds no ZIP. Code: `train/eval_e65.py`. Any change after its first run is an amendment, dated in §6.

## Motivation

E64 passed every gate on the organizers' 34 official rows (AUC 0.883, Finding 88). Those rows have been read by every
labeller since E24b, and E64's direction (easier = robust) runs against the batch board's (familiar = robust), so a
second population is the best evidence available before the board can be read. E64's amendment 1 dropped D4 (DeepSeek
MATH) because E55's states do not cover those problems. **They exist in another run:** `e1b-stagea-deepseek` extracted
DeepSeek's prompt states for 138 training problems with the same extractor, system prompt (SHA-1 `9021e9d4`), chat
template, 2048-token limit, mean pooling, bf16 and T4s as E55 (only torch differs, 2.10 against 2.11). Codabench has
also been unreachable since 22 Oct, so the board read E64's ship rule depends on may not come before the 27 Oct freeze.

## 1. Rows

DeepSeek's Hendrycks MATH instances in `data/small_track/instances_train.parquet` (142) whose `problem_key` has E1b
states: **137 rows, 36 robust**, labels as that table holds them (older rule, `absolute_accuracy_decay <= 0`).

## 2. Score

E64's shipped DeepSeek arithmetic exactly, from `runs/e64/artifact.npz`: per layer, standardise E1b's `X_mean` (float16
read as float32) with the stored mean and scale, apply the stored ridge, average over layers. Nothing is refitted.

## 3. Measurement

- **Primary (gate): AUC of -(predicted difficulty) against `is_robust`, 95% row-bootstrap interval (2,000 draws,
  seed 65). Pass: lower bound above 0.5.**
- Reported: balanced accuracy at E64's DeepSeek cutoff (10.3925) with E54's bootstrap (most MATH problems may fall
  below an olympiad cutoff, so this can sit at 0.5 whatever the ranking); the same AUC for E61's statistic (mean NLL,
  lower = robust) on the same rows, for comparison; Spearman of predicted difficulty with mean NLL.

## 4. What each result means

- **Pass:** easier = robust holds on DeepSeek's home population as well as on the official rows. E64 then has
  support from two populations that share no problem.
- **Interval spanning 0.5:** no evidence either way from MATH.
- **Interval below 0.5:** E64 reverses on MATH, as familiarity reverses on the official rows; it would then read one
  population's structure, not robustness, and the report says so.

## 5. Use

E64's ship rule stands: it replaces `e27f` in the Google Form at 10/14 or better on the board. **If Codabench is still
unreachable on 27 Oct**, this result is the owner's evidence for the Form choice: E64 if E65 passes, otherwise `e27f`
(recommendation; the owner decides). E65 is reported whatever its result.

## 6. Amendments

None.
