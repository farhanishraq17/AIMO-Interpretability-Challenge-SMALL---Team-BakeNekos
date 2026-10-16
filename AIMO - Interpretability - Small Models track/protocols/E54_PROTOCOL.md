# E54: does any signal we already hold beat the constants as a per-problem rule?

Status: **pre-registered 16 October 2026, before any per-problem accuracy below was computed.** Code:
`train/eval_e54_per_problem.py`. Any change after the first number is read is an amendment, dated in §7.

## Motivation

Since 15 October the organizers call `are_robust` once per problem, and the final test runs the same
way (Finding 79). `e27k`'s batch ranking cannot act on one problem, so it collapses to the per-model
constants (0.64). A replacement must make an **absolute** decision per problem: a score and a fixed
threshold. Every signal in the repository was evaluated as a within-batch ranking; none was ever
scored this way.

**This is an audit, not a blind test.** The ranking quality (AUC) of familiarity and length on most
populations below is already reported (`REVIEW_GENERALIZATION.md` §2.2), and the budget-path rule's
absolute-threshold balanced accuracy on Putnam is in E36's result (0.5975). What E54 adds is the
per-problem decision accuracy in every other cell, under rules fixed here. Anything that qualifies
still needs a fresh read on labels nobody has looked at (the collaborator's self-labels, PLAN §12.8)
before it can ship.

## 1. Populations (labels already in the repository)

| id | population | labels | rows |
| --- | --- | --- | --- |
| P1 | official small-track rows, `train-main-v2` rev `23791d9b` | official `max_drop` rule | 34 |
| P2 | DeepSeek, Hendrycks MATH (`data/small_track`) | older decay rule | 142 |
| P3 | Putnam-AXIOM (E35, Qwen3 models) | E35's derived labels | E35's count |
| P4 | GSM-Plus (E37's confirmation set) | E37's clear-cut labels | E37's count |
| P5 | MATH-Perturb (E35, Qwen3 models) | E35's derived labels | E35's count |

P5 is the budget-path predictor's training set, so that signal is not scored there.

## 2. Signals and their frozen per-problem rules

No threshold is chosen on any labelled population.

| id | signal | robust iff | threshold source | scored on |
| --- | --- | --- | --- | --- |
| S1 | familiarity: mean token NLL, bare text (E23's statistic) | NLL < reference median of the same scoring model | median over the reference set R | P1, P2 (own models); P3, P5 (`Qwen/Qwen3-0.6B` proxy, as in the review) |
| S2 | length in characters | length < reference median | median over R | P1, P2, P3, P4, P5 |
| S3 | budget-path probability (E37's frozen predictor, trained on MATH-Perturb) | P ≥ E36's out-of-fold threshold | chosen by E36 on its training data | P1 (E49 features), P3 (E36 holdout), P4 (E37 features) |

**Reference set R:** the 75 unlabelled ReliableMath AMC 2023 originals of E52
(`data/reliablemath/pool.parquet`). Our four models' NLLs on R come from `e52-features`; the proxy's
NLLs on R are computed with the same `extract_familiarity.familiarity` function.

**Directions** are taken from where each signal was proposed, not from these populations: familiar
= robust (`e27k`), shorter = robust (E22's fitted length model, Finding 46 corrected in the review).

## 3. Metric

Pooled balanced accuracy of the per-problem decisions over every labelled row of the population,
with a 95% interval from 2,000 problem-level bootstrap resamples (seed 54). Any constant decision has
balanced accuracy 0.5 within a model, which is the per-model constants' value. On P1 the accuracy of
each signal is also reported beside the constants' accuracy on the same rows.

## 4. Qualification (fixed now)

A signal **qualifies** if both hold:

1. its balanced-accuracy interval lies above 0.5 on **at least two** populations;
2. its point balanced accuracy is **at least 0.5 on every** population it is scored on (no reversal).

## 5. Decisions

- **Something qualifies:** it becomes a per-problem candidate, but ships only after (a) E55's timing
  shows it fits the 1 h 10 min budget, and (b) it holds on the collaborator's fresh self-labels. No
  leaderboard read chooses anything (owner's rule, 14 Oct).
- **Nothing qualifies:** the per-model constants (`e27f`) remain the per-problem floor until the
  self-labels exist, and E56 (budget-path features at scale) is not run.

## 6. Not done

No threshold, direction, reference set or population is changed after a number is read. P1's 34 rows
are reported, not used to select. The 14-case leaderboard is not consulted.

## 7. Amendments

None yet.
