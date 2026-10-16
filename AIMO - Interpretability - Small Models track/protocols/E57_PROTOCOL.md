# E57: a per-problem probe trained on full-budget self-labels, and the rule that decides if it ships

Status: **pre-registered 17 October 2026, before any self-label exists.** Code:
`train/train_e57_probe.py` (training and evaluation), `submission/make_e57.py` (packaging, later). Any
change after the first real self-label is read is an amendment, dated in §8.

## Motivation

Evaluation is per-problem (Finding 79): a submission needs an absolute decision for one problem at a
time. No signal we hold beats the per-model constants that way (E54, Finding 80), and the competition's
label is decided mostly by whether a model notices an impossible premise at a 100k-token budget
(Findings 76, 77). The collaborator is self-labelling that behaviour at full budget (PLAN §12.6-12.8).
E57 fixes now, before any of those labels exist, how they become a predictor and when it ships.

## 1. Precondition

E57 runs only if the collaborator's gate passed: their self-labels agree with the official labels on
the 34 official small-track rows at the threshold they pre-registered. Otherwise E57 does not run and
`e27f` (the per-model constants) is the submission.

## 2. Data

- **Labels:** the collaborator's self-labels, (model, problem) -> robust / spurious / band, under the
  official rule (robust iff `max_drop` <= 0.10, spurious iff >= 0.25). Band rows are dropped.
- **Held out of training:** the 36 `train-main-v2` problems. Their official labels get one read at
  the end (§5.3); they never enter training or threshold selection.
- **Features:** E55's one-pass features for the same pairs (Finding 81): Stage A mean-pooled states at
  every layer, and the E23 familiarity statistics.
- **Groups:** problems whose normalised text shares at least 80% of word 3-grams are one group, so
  near-duplicates across sources (the pool's AIME 2024 against ReliableMath's) never straddle a split.

## 3. Model (per evaluated model)

- **Primary:** Finding 11's layer ensemble: per layer, a standardised L2 logistic regression (C = 0.01,
  `class_weight="balanced"`) on mean-pooled states; the probability is the mean over layers.
- **Threshold:** per model, the value maximising balanced accuracy on out-of-fold probabilities of the
  training folds only (nested: chosen inside each outer fold, applied to its held-out fold).
- **Reported, not shipped:** the same with the familiarity statistics alone.

## 4. Evaluation

Five outer folds grouped as in §2, seed 57. Held-out per-problem decisions, pooled over the four models:
accuracy and balanced accuracy, each with a 95% interval from 2,000 group bootstrap resamples, and the
paired difference against each baseline.

**Baselines:** (a) per-model majority class of the training labels, chosen inside each fold; (b) `e27f`'s
constants (Qwen and DeepSeek robust, Skywork and Olmo spurious).

## 5. Ship rule (all three)

1. Held-out balanced accuracy has an interval lower bound above 0.5.
2. Held-out accuracy beats **both** baselines, with paired-difference interval lower bounds above 0.
3. On the 36 official problems (34 labelled small-track rows), read once, the final model's balanced
   accuracy is at least 0.5 and its accuracy is not below the constants' (0.529, Finding 80).

If all three hold, `submission/make_e57.py` packages the probe (cached model, one forward pass per call,
fallback to the constants on any error), it passes the local scorer and the Docker load test, and it is
uploaded **once** as a sanity check. Its leaderboard score selects nothing. If any rule fails, `e27f`
ships.

## 6. Plumbing test (before real labels)

`train/train_e57_probe.py --synthetic` draws random labels for every E55 pair and must report held-out
balanced accuracy near 0.5. It reads no real label of any kind.

## 7. Not done

No hyperparameter, layer subset, threshold rule or baseline is changed after real self-labels are read.
The 34 official rows are read only at §5.3. The leaderboard selects nothing.

## 8. Amendments

None yet.
