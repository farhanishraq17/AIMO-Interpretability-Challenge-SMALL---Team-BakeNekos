# M3: do our own labels reproduce the organizers' labels?

Status: **pre-registered 27 September 2026, not yet run.** Nothing in §"Frozen decision" may be
changed after the first comparison between one of our labels and one official label.

## Motivation

Every offline lane in both tracks has failed for one of two reasons: 33 labelled problems, or the
wrong population. `PLAN.md` Lane A converts unlimited compute into 200–500 labelled olympiad
problems by running the organizers' own labelling procedure ourselves — sample the model on an
original, sample it on answer-preserving perturbations, and threshold the largest accuracy drop.

That is only worth anything if our labels *are* their labels. M3 measures it on the one overlap that
exists: the 71 labelled rows of `aimo-interp/train-main-v2`.

M3 is also the gate that can end Lane A. If it fails, the Main Track ships constants (M11a) and the
failure itself is reported: *the official labelling rule is not reproducible from surface
perturbations without the symbolic chains in `problems-public`.*

## Frozen population

The 36 problems of `train-main-v2` at revision `23791d9b`, and only those. 71 of the 82 rows carry a
label; the other 11 (`max_drop` in 0.11–0.21) are **excluded from every metric**, exactly as the
organizers exclude them from scoring.

All 7 (model, effort) cells are compared: the four small checkpoints at `default`, and
`openai/gpt-oss-120b` at `low`, `medium` and `high`.

Problems are grouped by **template** (whitespace-normalised text with numeric constants masked), not
by text, for every resampling step. `train-main-v2` repeats templates with one constant swapped under
different labels (small-track Finding 34).

Ground-truth answers come from published sources only — `MathArena/aime_2025` and the existing
pinned pool (LF-15). **The small track's E24a key is not used here**: it is model-derived, and a
label chain resting on one model's opinion cannot test another model's labels.

## Frozen procedure

Per (problem, model, effort):

1. **Base phase.** k = 8 samples on the original problem, temperature and top-p as the checkpoint's
   own recommended defaults, context ≥ 32,768 (≥ 65,536 at `high`). A problem is **eligible** iff
   base accuracy = 1.0. Ineligible problems are dropped and counted, never labelled.
2. **Variant phase.** k = 4 samples on each accepted variant, 4 variants per family, over the five
   official surface families (`rephrase`, `rename`, `domain`, `distract`, `typos`). Variants are
   generated and verified under M1 and its gate G-M1. `expert_perturbations` and
   `expert_no_solution` are **excluded** — the first needs expert authorship, the second inverts the
   correctness criterion.
3. **Aggregate.** Per family, permuted accuracy pools every sample of every variant of that family.
   `max_drop` = max over families of (base accuracy − permuted accuracy).
4. **Label.** Three-valued, from the two cut points fixed in the next section.

Sampling is from our own perturbations of these problems. **The 453 official variant texts are not
used as inputs at any point in M3** — they are the organizers' outputs and using them would leak the
quantity under test. Their only permitted use anywhere is as style exemplars for M1's generator,
which is recorded in M1's own log.

## Frozen decision

**The cut points are fitted, not assumed.** Grid-search the pair `(robust_cut, spurious_cut)` over
`robust_cut ∈ {0.05, 0.10, 0.15, 0.20}` and `spurious_cut ∈ {0.20, 0.25, 0.30, 0.35, 0.40}` with
`robust_cut < spurious_cut`, maximising Cohen's kappa against the 71 official labels. Report the
whole grid, the chosen pair, and kappa at the organizers' own (0.10, 0.25).

Because the cut points are fitted on the same 71 rows they are scored on, the headline kappa is
optimistic. The gate therefore uses the **leave-one-template-out** kappa: for each template, fit the
cut points on the remaining templates and predict that template's rows.

**G-M3 passes iff all three hold:**

1. leave-one-template-out Cohen's kappa **≥ 0.4**;
2. its template-cluster bootstrap CI (2,000 resamples, 95%) has a **lower bound > 0**;
3. Spearman correlation between our `max_drop` and the organizers' `max_drop` is **positive** with a
   95% CI excluding 0, over all 82 rows including the 11 unlabelled ones.

If (1) or (2) fails, **Lane A stops**: no 500-problem run, ship M11a, write the negative result. If
only (3) fails, the labels agree but the underlying quantity does not, and the run may proceed only
with the discrepancy documented in `PROGRESS.md` as a finding before M2 starts.

Also reported, and not gated: the 2×2 confusion matrix, per-cell kappa for all 7 cells, the
eligibility rate per model, and the number of problems our pipeline leaves unlabelled that the
organizers labelled (and the reverse).

## Leakage guard

- No Codabench outcome, no leaderboard reading and no validation-set text enters M3.
- The official `is_robust` and `max_drop` columns are read **only** in the comparison step, never as
  a predictor input, never to choose a generator, a model, a sampling parameter or a variant.
- The official variant texts are not sampled in M3 (see Frozen procedure).
- Variant acceptance under M1 uses a re-solving panel whose members are **not** the model being
  labelled, so a model cannot certify its own perturbations.

## Recorded before the run

Fill these in before the first sample, and do not edit afterwards:

| Field | Value |
| --- | --- |
| Hardware | _to record_ |
| Inference engine and version | _to record_ |
| Checkpoint revisions (5 models) | _to record_ |
| gpt-oss-120b precision (MXFP4 or bf16) | _to record_ |
| Generator model and revision (M1) | _to record_ |
| Verification panel (3 models) | _to record_ |
| Sampling seeds | _to record_ |
| Date frozen | 27 September 2026 |
| Date run | _to record_ |
