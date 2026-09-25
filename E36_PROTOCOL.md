# E36: budget-path reasoning diagnostics

Status: completed 25 September 2026.  The ranking gates passed strongly, but the frozen
absolute-threshold balanced-accuracy gate missed by 0.0025, so E36 formally failed and did not
authorize AIMO extraction or a submission.  E37 is the one independently pre-registered
confirmation of the unchanged score with rank calibration.

## Motivation

E35 showed that original problem text alone does not transfer reliably from MATH-Perturb to
Putnam-AXIOM.  The same pinned public archive also contains each model's original reasoning
trace at several token budgets.  E36 asks a different question: does the *path by which the
model reaches an answer* predict whether its answer survives independently authored
perturbations?

The source remains `ssubhnil/Qwen3-Math-Eval` at revision
`f64c82f5ce9249155ad7c70991a9711d92963cce`.  E36 uses no AIMO labels, Codabench outcomes,
variant text, variant traces, ground-truth answers, or correctness fields as predictor inputs.
Correctness is read only to construct the external target label.

## Frozen examples and labels

An example is one `(problem, model)` configuration.  Use Qwen3 1.7B, 4B, 8B and 14B.

- MATH-Perturb is the training source.  A configuration is eligible when its 16,384-token
  `base` answer is strictly correct; it is robust when both its `simple` and `hard` answers
  are strictly correct at the same budget.
- Putnam-AXIOM is the untouched transfer set.  A configuration is eligible when its
  16,384-token `original` answer is strictly correct; it is robust when every corresponding
  `variation` answer is strictly correct at the same budget.

The ID canonicalisations are exactly those already tested in E35.  Labels remain
model-specific rather than being aggregated across models.  All configurations of one problem
stay together in resampling and cross-validation.

E35 reported only a text model's aggregate Putnam score; no E36 feature/label association or
per-problem Putnam outcome was inspected before this protocol.

## Frozen predictor inputs

For the original/base prompt only, use the deterministic traces at budgets 2,048, 4,096,
8,192 and 16,384.  Extract exactly these architecture-independent features:

1. natural-completion indicators at each budget (`answer_forced == false`);
2. fraction of budgets completing naturally and the first natural-completion budget;
3. fraction with a non-empty extracted answer;
4. normalized-answer modal fraction, unique fraction, adjacent agreement and agreement with
   the 16,384-token answer;
5. log word count at each budget and the 8,192-to-16,384 growth ratio;
6. fixed self-correction-marker rate in the 16,384-token trace for
   `wait`, `however`, `alternatively`, `instead`, `check`, `verify`, `reconsider`, `mistake`;
7. repeated non-empty line fraction in that trace.

Answer normalization is lower-casing plus removal of whitespace and the characters
`$[]{},`; no mathematical correctness or equivalence checker is used.  Missing numeric
features are median-imputed from MATH-Perturb only.

Fit one `StandardScaler` plus class-balanced logistic regression (`C=1`, seed 36).  No text
TF-IDF, model identity, subject, difficulty, ground truth, correctness-at-shorter-budget,
variant data, feature selection, hyperparameter sweep or model ensemble is allowed.

Choose the binary threshold from five-fold grouped out-of-fold predictions on MATH-Perturb by
maximising balanced accuracy over the finite set of OOF scores.  Ties use the larger threshold.
Putnam is scored once with the fitted MATH model and this frozen threshold.

## External transfer gates

E36 passes only if every condition holds:

1. MATH-Perturb has at least 100 eligible configurations in each class;
2. Putnam has at least 40 eligible configurations in each class;
3. Putnam configuration-level ROC-AUC is at least 0.65;
4. a 5,000-replicate bootstrap that resamples problem IDs has an AUC 95% lower bound above
   0.50;
5. fixed-threshold Putnam balanced accuracy is at least 0.60;
6. every non-degenerate model-specific Putnam AUC is at least 0.50 and at least three of four
   are at least 0.58.

Failure closes E36 without AIMO evaluation or a Codabench upload.  The individual frozen
features may be reported diagnostically but cannot replace or revise the predictor.

## Conditional AIMO transfer

Only if all external gates pass:

1. implement the same four-budget deterministic trace extraction for the four cached
   small-track checkpoints;
2. extract it for all labelled `train-main-v2` rows on Kaggle;
3. score the official rows once with the unchanged E36 predictor and threshold;
4. require problem-level AUC at least 0.70, a template-bootstrap lower bound above 0.50,
   balanced accuracy above the E27 constants, and a measured worst-case Codabench runtime
   below 3,000 seconds.

Only then may one frozen Small Models submission be built and uploaded.  No runtime budget,
feature, threshold, per-model direction or rank fraction may be changed after seeing Putnam,
official-label or leaderboard results.  E27e/E27h at 0.71 remains the deployed candidate until
all gates clear.

## Result

MATH-Perturb yielded 1,001 eligible model configurations (769 robust, 232 spurious) over 264
problems.  Putnam yielded 206 (100 robust, 106 spurious) over 72 problems.  The unchanged model
scored ROC-AUC **0.765**, problem-bootstrap 95% CI **[0.687, 0.839]**.  Per-model AUCs were
0.771 (1.7B), 0.814 (14B), 0.680 (4B), and 0.794 (8B).  The frozen MATH OOF threshold produced
balanced accuracy **0.598**, below the required 0.60.  Therefore `passed=false`; no official
AIMO label was loaded for E36 and no Codabench bundle was built.
