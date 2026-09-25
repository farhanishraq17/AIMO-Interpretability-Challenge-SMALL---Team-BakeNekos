# E34: new official-like labels with a frozen external holdout

Status: E34a completed 25 September 2026 and failed its frozen yield gate. E34b is closed.

## Objective

Beat the current Small Models score of 0.79 without learning from the 14 hidden leaderboard
labels or re-tuning on the 34 exhausted public small-model labels. A score above 0.79 requires
at least 12/14 correct; E27e/E27h currently produce 10/14.

## E34a yield pilot

- Development population: all 30 `aime-2024` problems in `data/olympiad_pool/pool.parquet`.
- Independent pilot population: the 30 `aime-2026` problems, minus the eight problems that
  overlap `train-main-v2`, leaving 22 unseen problems.
- The excluded problem keys are frozen in `datagen/kaggle_launch.py::E34_EXCLUDED_KEYS`.
- No official labels and no leaderboard outcomes participate in fitting or selection.
- Target checkpoint: `Qwen/Qwen3.5-4B` only for the pilot.

## Frozen label generator

- Original: four samples, temperature 0.6, top-p 0.95, 12,288-token cap.
- Eligibility: original accuracy must be 1.0.
- Perturbations: exactly `latex_to_plaintext` and `premise_reorder`; no later family search.
- Four samples per variant.
- Per-category accuracy drops are pooled into a per-problem maximum.
- Robust if maximum drop is at most 0.10; spurious if at least 0.25; the middle band is omitted.

These two transformations are intentionally narrow. E34a is a falsification/yield pilot, not
permission to submit a surface-rewrite classifier.

E34a measures feasibility only. It fits no predictor and does not inspect official labels.

## E34a gates

E34a passes only if all conditions hold:

1. At least eight AIME 2024 and five AIME 2026 problems are eligible and clear-cut.
2. The combined clear-cut set contains at least four robust and four spurious problems.
3. Each source contains at least one robust and one spurious problem.
4. Each frozen perturbation family contributes at least four eligible problem rows.

Failure closes the two-family surface-label lane with no Codabench submission. Passing E34a
only authorizes E34b; it is not evidence that a classifier will transfer.

## E34b independent validation

- Training: `olympiadbench-en` plus `aime-2024`, excluding every `train-main-v2` overlap.
- Untouched holdout: `amc-validation` plus the 22 non-overlapping `aime-2026` problems.
- The E34a generator, sampling parameters and label bands remain unchanged.

If the pilot yields enough labels, fit one character TF-IDF logistic regression:

- analyzer `char`, n-grams `(3, 5)`, `min_df=2`, `max_features=50_000`;
- `LogisticRegression(C=1.0, class_weight="balanced", max_iter=2000, random_state=34)`;
- decision threshold 0.5;
- train on the frozen E34b training sources and score the holdout once.

No rewrite type, layer, threshold, rank fraction, feature family, or hyperparameter may be
selected using AIME 2026, `train-main-v2`, or Codabench.

## E34b gates

E34b passes only if all conditions hold:

1. At least 80 training and 25 holdout problems are eligible and clear-cut.
2. The holdout contains at least eight robust and eight spurious labels.
3. Holdout ROC-AUC is at least 0.75 and balanced accuracy at threshold 0.5 is at least 0.67.
4. A 2,000-resample problem bootstrap has an AUC lower bound above 0.5.
5. AUC is at least 0.60 separately on AMC and AIME 2026 whenever both classes occur.

Passing E34b authorizes one fixed transfer measurement on `train-main-v2`, with no refit or
selection. It must additionally clear the existing G2 official-transfer gates before
one—and only one—Codabench submission.

## Leakage guard

Scores from prior Codabench probes may be used to state the 10/14 baseline, but not to infer
which cases were missed, choose per-row predictions, or select an ensemble. The inaccessible
`problems-public` dataset must be obtained through organizer-granted access; cached snippets are
not reconstructed.

## E34a result

Kernel version 1 completed on `tashintahir` after sampling all 52 declared originals. Only
**1/52 (1.92%)** met the frozen eligibility rule of 4/4 correct original samples. That problem
produced one `latex_to_plaintext` row, labelled robust; there were no eligible
`premise_reorder` rows and no spurious labels.

The frozen gate therefore failed every substantive condition:

- AIME 2024 clear-cut: 1, required 8;
- AIME 2026 clear-cut: 0, required 5;
- robust/spurious: 1/0, required at least 4/4;
- perturbation-family coverage: 1/0 rows, required at least 4/4.

This is a scientific yield failure, not an execution failure: both base shards finished all
26 assigned problems, the tokenizer round-tripped correctly, and all 52 base records resolved.
E34b is not launched and no Codabench submission is made.
