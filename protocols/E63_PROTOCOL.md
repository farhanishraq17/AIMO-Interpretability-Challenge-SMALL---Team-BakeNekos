# E63: E61's familiarity cutoffs, recalibrated on the organizers' own problems

Status: **pre-registered 21 October 2026, before any E63 threshold or decision was computed.** Code:
`train/eval_e63.py` (diagnosis, thresholds, checks), `submission/make_e61.py` (the ZIP, same solution template).
Any change after the first number is read is an amendment, dated in §7.

## Motivation: why E61 failed (diagnosis, computed by `train/eval_e63.py`)

E61b drew its cutoffs from E55's 548 problems, 362 of them OlympiadBench. The 36 problems of the organizers' own
`train-main-v2`, built by the pipeline that built the validation and test sets, are far more familiar to every model
(lower mean NLL, and about twice as long). On those 36, E61b's cutoffs call 78% (Qwen), 83% (DeepSeek), 56%
(Skywork) and 53% (Olmo) robust, against intended 60%, 67%, 33%, 33%. If the hidden problems are drawn from the
organizers' distribution, with labels following familiarity inside each batch as E50 recovered, the expected board
scores are 9.1 for E61a and 9.3 for E61b: the observed 8 and 9.

The same model sets a ceiling: **a per-problem familiarity cutoff, perfectly calibrated, expects about 10.3 of 14**
(about 2.44 of 3 on DeepSeek, Skywork and Olmo; on Qwen no cutoff beats the constant `True`). The batch ranking's 12
came from relative information inside a batch, which per-problem calls remove by design. E63 is therefore the best
this family can do, not a route to 12.

## 1. Rule

Score: E61's, unchanged (mean token NLL of the bare problem; same solution code, parity 0.0 on four checkpoints).

| model | robust iff |
| --- | --- |
| deepseek-ai/DeepSeek-R1-0528-Qwen3-8B | NLL ≤ the 2/3 quantile of its NLLs on the 36 `train-main-v2` problems |
| Skywork/Skywork-OR1-Math-7B | NLL ≤ the 1/3 quantile, same reference |
| allenai/Olmo-3-7B-Think | NLL ≤ the 1/3 quantile, same reference |
| Qwen/Qwen3.5-4B | constant `True` (no cutoff beats it under the diagnosis model; E50: its order adds nothing) |

The quantiles are E61's `p_m`, unchanged; only the reference changes. The 36 problems enter as text only: no label
is read to set a cutoff. Any failure returns the model's constant, as in E61.

## 2. Checks before upload

| id | check | pass |
| --- | --- | --- |
| C2 | DeepSeek MATH (142), E54's loader and bootstrap (2,000 draws, seed 63) | interval lower bound > 0.5 |
| C3 | the 34 official rows | reported only |
| C5, C1 | carried from E61: the solution code is byte-identical but for the thresholds | — |
| C6 | `run_local.py --small`: coverage 1.0, 0 invalid | must pass |

## 3. Board read and ship rule

One upload. **E63 replaces `e27f` in the Google Form if C2 and C6 pass and the read is at least 10/14**, one case
above the constants; otherwise `e27f` ships. The Docker run is still owed before the Form (E61 amendment 1).

## 4. Not done

No cutoff, quantile or reference changes after the read. No per-case rule for any model, no state across calls, no use
of `train-main-v2` revision `25e45665`.

## 7. Amendments

None yet.
