# E37: independent confirmation of the frozen budget-path score

Status: pre-registered before reading any GSM-Plus correctness field or computing any E37
feature/label association.

## Motivation

E36's unchanged reasoning-path classifier transferred from MATH-Perturb to Putnam-AXIOM with
ROC-AUC 0.765 and a problem-bootstrap interval wholly above chance, including AUC 0.680--0.814
within each of four Qwen3 sizes.  It nevertheless missed its absolute-threshold balanced-
accuracy gate by 0.0025.  The ranking signal is therefore promising but E36 remains a formal
failure and does not authorize an AIMO run.

E37 is the one allowed confirmation.  It keeps E36's training data, 18 features, preprocessing,
logistic model and score direction unchanged, replaces the distribution-sensitive absolute
threshold with a prespecified within-model median rank, and evaluates on a third perturbation
benchmark whose correctness outcomes have not been inspected.

## Frozen data and label

Use the `gsm_plus` config from `ssubhnil/Qwen3-Math-Eval` at revision
`f64c82f5ce9249155ad7c70991a9711d92963cce`.

Schema-only inspection established that each source problem has eight consecutive IDs and one
row for each documented GSM-Plus type.  Canonicalise `problem_id` as `int(problem_id) // 8`.
Exclude `critical thinking`, whose gold target is an unanswerable sentinel according to the
dataset card.  Use the `problem understanding` rewrite as the anchor because it is the closest
answer-preserving paraphrase among the seven scorable types.

For each `(canonical problem, model)` at 16,384 tokens:

- eligible: the anchor is strictly correct;
- robust: all seven scorable variants are strictly correct;
- spurious: the anchor is correct and at most five of seven scorable variants are correct;
- ambiguous: exactly six of seven are correct; exclude it.

Predictor inputs are the anchor trace at 2,048, 4,096, 8,192 and 16,384 tokens.  They are passed
through E36's committed `trace_features` unchanged.  Variant traces, correctness, ground truth,
question text and model identity are never predictor inputs.

## Frozen predictor and decision

Rebuild the E36 model on all eligible model-specific MATH-Perturb examples exactly as committed
in `train/eval_e36_budget_path.py`.  Do not refit on Putnam or GSM-Plus.

Within each model's GSM-Plus batch, call the higher-scoring half robust using stable canonical-ID
tie-breaking.  This rank policy was selected before GSM-Plus labels were read.  It matches the
near-balanced Codabench validation prior established by constant probes and avoids carrying a
MATH-calibrated probability threshold across sources.

## Gates

E37 passes only if every condition holds:

1. GSM-Plus yields at least 50 robust and 50 spurious clear-cut configurations;
2. overall configuration-level ROC-AUC is at least 0.65;
3. a 5,000-replicate canonical-problem bootstrap has an AUC 95% lower bound above 0.50;
4. the frozen within-model half-rank decision has balanced accuracy at least 0.60;
5. every non-degenerate model AUC is at least 0.50 and at least three of four are at least 0.58.

Failure closes the budget-path lane with no AIMO GPU run and no submission.  Passing authorizes
one unchanged AIMO transfer experiment:

- generate deterministic original traces at the same four budgets for all 34 labelled
  small-track `train-main-v2` configurations;
- use the frozen E36 model score and within-`(model_id, reasoning_effort)` higher-half decision;
- require problem-level AUC >= 0.70, template-bootstrap lower bound > 0.50, half-rank balanced
  accuracy above E27's constants, and an end-to-end measured runtime below 3,000 seconds.

Only if that transfer also passes may one frozen Codabench Small Models submission be uploaded.
No feature, budget, label band, model coefficient, direction, threshold or rank fraction may be
changed after GSM-Plus or AIMO outcomes are seen.
