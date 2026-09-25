# E35: external robustness labels with a source-disjoint holdout

Status: pre-registered 25 September 2026, before downloading or inspecting any row-level
correctness outcomes from the source archive.

## Motivation

E34a is closed. Its one eligible Qwen3.5-4B problem does not justify relaxing its eligibility
gate, changing its frozen transformations, or launching E34b. E35 is a different lane: it uses
an independently published archive of model outputs on established robustness benchmarks and
does not generate or tune perturbations.

The source is `ssubhnil/Qwen3-Math-Eval` at revision
`f64c82f5ce9249155ad7c70991a9711d92963cce`. The dataset card reports deterministic Qwen3
1.7B/4B/8B/14B outputs and strict correctness for MATH-Perturb and Putnam-AXIOM. Only the
published aggregate table and schema were read before this protocol; per-problem outcomes were
not inspected.

## Frozen label construction

Use only the 16,384-token rows and `correct_strict`.

For each `(problem_id, model)` configuration:

- MATH-Perturb is eligible iff `variant == "base"` is correct. It is robust iff both `simple`
  and `hard` are correct.
- Putnam-AXIOM is eligible iff `variant == "original"` is correct. It is robust iff every
  available `variation` row for the same configuration is correct.

Aggregate configurations to one problem label. Require at least two eligible models. A problem
is clear-cut robust when at least 75% of its eligible models are robust, clear-cut spurious when
at most 25% are robust, and ambiguous otherwise. Duplicate or incomplete configuration rows are
an error, not silently dropped.

MATH-Perturb is the only training source. Putnam-AXIOM is the untouched holdout and is scored
exactly once. No official AIMO labels, Codabench scores, AIME rows, or prior self-labels enter
this stage.

## Frozen predictor

Fit exactly two class-balanced logistic regressions on the original problem text:

1. character `char_wb` TF-IDF, 3--5 grams;
2. word TF-IDF, 1--2 grams.

Both use `min_df=2`, `max_features=20_000`, sublinear term frequency, `C=1`, and seed 35.
The sole E35 score is the arithmetic mean of their robust-class probabilities. There is no
candidate selection, feature selection, threshold sweep, or hyperparameter search.

For a binary decision, call the higher-scoring half of a batch robust, with stable
`problem_id` tie-breaking. This matches the near-balanced validation prior without estimating a
threshold from the holdout.

## External holdout gates

E35 passes the external stage only if all conditions hold:

1. the training set contains at least 30 robust and 30 spurious clear-cut problems;
2. the Putnam holdout contains at least 20 robust and 20 spurious clear-cut problems;
3. ensemble ROC-AUC on Putnam is at least 0.65;
4. a 5,000-replicate stratified problem bootstrap has an AUC 95% lower bound above 0.50;
5. the fixed half-rank decision has balanced accuracy at least 0.60.

Failure closes E35 with no inspection of `train-main-v2`, no Codabench upload, and no tuning.

## Conditional official transfer

Passing the external gates authorizes one transfer measurement, with no refit choices:

- refit the same ensemble on the union of clear-cut MATH-Perturb and Putnam rows;
- score all official small-track `train-main-v2` rows once;
- rank within each `(model_id, reasoning_effort)` batch and call the higher half robust;
- require problem-level ROC-AUC at least 0.65 with a template-bootstrap lower bound above 0.50,
  and require the fixed decisions to beat always-robust in balanced accuracy.

Only if every transfer gate passes may one frozen E35 submission be built and uploaded. There
is no leaderboard-conditioned revision. E27e/E27h at 0.71 remains the performance candidate and
E11 v2 at 0.64 remains the conservative fallback unless E35 clears every gate.
