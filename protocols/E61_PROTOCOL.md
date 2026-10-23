# E61: drop the per-model split, then look for a high-precision spurious detector

Status: **open. Pre-registered 23 October 2026, before any E61 artifact was built, uploaded or scored.**
Nothing in §4 may be relaxed after a result is read; a change is an amendment, dated in §7.

Code to be written: `submission/make_e61.py` (candidates), `train/eval_e61_floor.py` (§4 gate A),
`train/eval_e61_detector.py` (§4 gates B1–B3).

## Motivation

The shipped submission is `e27f`: `True` for Qwen3.5-4B and DeepSeek-R1-0528-Qwen3-8B, `False` for
Skywork-OR1-Math-7B and Olmo-3-7B-Think. Per-problem evaluation (Finding 79) removed the ranking that
used to sit on top of those constants, so the four booleans are now the whole method.

Two measurements already in this repository say that method is the wrong shape, and neither was
re-read after 15 October.

**1. The label is a property of the problem, not of the model** (Finding 38,
`train/eval_train_main_v2.py`). Small-track models agree on 20 of 20 shared problems; across every
configuration including `gpt-oss-120b`, 83 of 85, against a chance rate of 0.608 at the base rate.
A policy whose entire content is a per-model distinction is modelling a difference this table says
is not there.

**2. All four models are majority robust in the official labels** (RESULTS, "E43: prior sensitivity
of the tilts", `train/eval_prior_sensitivity.py`): Qwen 8 of 12, DeepSeek 7 of 10, Skywork 2 of 3,
Olmo 7 of 9. Per-model constants derived from that source are therefore `True` four times — there is
no split to make.

The two sources we hold disagree, and the disagreement is recorded: on the 34 official rows the
constants score accuracy **0.529** and balanced accuracy **0.463** (**0.465** and **0.450** on the
43 rows the table now holds, Finding 87) (RESULTS, "E54: per-problem rules
against the constants"; PROGRESS Finding 58 table), while always-robust scores **0.706** and 0.500.
On the solved validation set (RESULTS, "E50: the hidden set, solved from the leaderboard") the
constants score **9 of 14** and always-robust **7 of 14**. Each source favours a different answer.

**What breaks the tie is the shape of the bet, not its size.** Write `w` for the share of test pairs
belonging to the two models the constants call robust. A policy that answers `True` on that share and
`False` on the rest scores `w·p + (1−w)·(1−p)` under a shared robust prior `p` — a weighted average of
`p` and `1−p`, which can never exceed `max(p, 1−p)`, and which equals exactly 0.5 at `w = 0.5`.
Always-robust scores `p`. So the constants can only win if the per-model priors genuinely differ in
the assumed direction, which is what Finding 38 and E43 both argue against. This is the same
closed-form pricing E43 applied to the tilts, carried over to the per-problem protocol.

**And the target was wrong.** At a robust prior near 0.7 with plain accuracy as the metric, the object
is not a classifier. It is a **flip-only detector**: answer `True` by default and flip to `False` only
on confident evidence. Its value is `flips × (2·precision − 1)`, so it needs precision above 0.5 on
the rows it flips — not balanced accuracy above 0.6, which is what every gate in this project
required. E61 changes the target before it changes the method.

**The affordable budget is also larger than we assumed.** The organizers' own `uncertainty-profiling`
baseline generates 4,096 tokens per problem (`docs/COMPETITION.md`, upstream `getting-started`) and is
published at 69.23% ± 10.13. A method costing about 4k generated tokens per problem is therefore
affordable by the organizers' own construction. The cost gate that closed E14 and E36/E37 was
rejecting methods at 28k tokens per problem (PAPER §10); it does not reach 4k.

## 1. Population and sources

- **Gate A (floor)** is scored on the **43** labelled `train-main-v2` small-track rows (Finding 87:
  the table grew from 34 on 15 Oct and nothing in this repository had re-read it), and on the solved
  validation labels of E50. Both are already in the repository; no new extraction.
- **Gate B (detector)** is scored on the same 43 official rows, read **once**, after the detector and
  its threshold are fixed by §3.
- No label from any other source is joined to either gate. The E50 solved labels are used only to
  report what a candidate would have scored on the validation set; **they never select a candidate**
  (owner's rule, 14 Oct).

## 2. C1 candidates (no model, or upstream code unchanged)

1. `e61a-always-true` — `True` for all four small-track models, `False` for anything else, no model
   loaded. Built by `make_e61.py`.
2. `e61b-uncertainty-baseline` — the upstream `solutions/uncertainty-profiling` solution, packaged
   **unmodified**, as the published 69.23% baseline. Requires the `getting-started` submodule, which
   is not checked out on this machine.

Both are packaged in the pinned local venv, never on Kaggle, and both must pass
`submission/docker_load_test.py` in `Dockerfile.competition`.

## 3. C2 detector (mechanism chosen by the literature sweep, not here)

The mechanism is **not fixed by this protocol**. It is chosen by the C0 literature sweep (Appendix A),
and the choice is recorded as amendment 1 before any detector code is written. What is fixed now is
everything else:

- It is a **flip-only rule on top of `e61a`**: the default answer is `True` for every pair, and the
  rule may only turn a pair to `False`.
- It makes an **absolute per-problem decision**. No batch statistic, no rank, no comparison between
  the problems in a call.
- Its threshold is **structural or fitted off the official rows** — fitted on unlabelled data, on an
  external population, or set by the mechanism itself. **It is not fitted on the 34 official rows**,
  which is the error that produced `e27f`.
- It costs at most **4,000 generated tokens per problem**, measured, and the models are cached across
  calls.

**Mechanism selected by the sweep (amendment 1, §7): uncertainty measured on the original problem.**
Premise-ablation answer stability was the candidate when this protocol was written. R5 refuted it before
any code existed: a competing entrant ran exactly that test and scored 0.50 on the validation set and
**0.38 on the official rows**, below always-robust. It is not built.

What replaces it is the signal three independent sources support, two of them written by this
competition's own organizers:

- **Grünefeld et al. (arXiv 2605.07776)**, the work the organizers' published baseline is built on:
  AUROC up to 0.807 for chain-of-thought correctness from the uncertainty-trace profile, and
  **AUROC 0.801 from only the first few hundred tokens**. Mondorf and Plank are AIMO organizers.
- **Fazla et al. (arXiv 2606.16583)**, the only published task structurally identical to ours — predict
  from the unperturbed input whether an already-correct answer flips under perturbation. Flipped items
  carried higher baseline uncertainty in **every** model/method combination under one perturbation and
  85 of 96 under another, and **answer-token NLL from a single pass was the best single-pass
  discriminator on 11 of 12 models**. Plank is an author here too.
- **The organizers' own `uncertainty-profiling` baseline**, published at 69.23% ± 10.13.

The detector therefore reads **answer-token log-probability plus uncertainty-trace shape from one
truncated trace**, per-model normalised by rank or z-score **on E55's 548-problem unlabelled pool**
(Finding 81), with a **single pooled threshold**. Per-model thresholds are not fitted: 7 to 8 official
rows per model cannot set them, which R5 names as the sharpest constraint in this setting.

## 4. Gates, fixed now

**Gate A — C1 floor.** Both C1 ZIPs load under `Dockerfile.competition`, return native Python `bool`s,
carry a root `small.txt`, and accept both call signatures. On pass, **`e61a-always-true` becomes the
incumbent** and is the rollback point for everything after it. On fail, `e27f` remains the incumbent
and only the packaging is repaired.

**Gate B — C2 detector.** All three must hold on the 34 official rows, read once:

- **B1 precision (revised 23 Oct, before any run; see §7).** Of the rows the detector flips to
  spurious, the number truly spurious must reach a **one-sided binomial p-value of 0.05 or below
  against the 0.279 spurious base rate** of this population (12 of 43; revised by Finding 87), with a
  minimum of **3 flips**. The rule is scale-free: it accepts 3 of 3, 4 of 4, 4 of 5, 5 of 6, 5 of 7,
  5 of 8, 6 of 9, 6 of 10, 7 of 11, 7 of 12 and so on, so a selective detector and a broad one are judged on the same standard. The
  detector is run at its **own natural coverage**; the flip count is not tuned to reach the gate.
- **B2 coverage.** It flips at most **40%** of all rows. A detector that flips most of them is
  always-spurious in disguise, which scores 0.279 on this population.
- **B3 cost.** Measured generated tokens per problem at most **4,000**, and projected whole-run time at
  one third of measured T4 time at most **50 minutes**, leaving 20 of the 70 for loading and fallbacks.
  The one-third factor is E55's assumption, restated so it cannot be tuned here.
- **B4 difficulty control (added by amendment 1).** The detector must beat a **difficulty-only
  baseline at matched solve rate**. Two independent entrants in this competition found their signals
  tracked solve rate rather than fragility (byeolki, "Prompt Probes Read Solve Rate, Not Robustness";
  EconLearn, "Difficulty is not fragility"), and our own E23 reached the same place by a different
  route. A detector that fails B4 is a difficulty predictor and does not ship, whatever B1 says.

On pass, the detector ships as `e61c` and becomes the incumbent. On fail of **B1**, roll back to
`e61a`: the threshold is not re-fitted, the flip count is not adjusted, and the gate is not relaxed. On fail of **B2 or B3 only**, one
re-parameterisation is permitted (fewer ablations, shorter forced answer), re-gated once; a second
failure rolls back to `e61a`.

## 5. Rollback rules

- `dist/` is append-only. No ZIP is deleted or overwritten; the shipped choice is a file selection.
- The incumbent is always the best artifact that has passed its own gate. It starts as `e27f`.
- A gate may be failed, never relaxed. The only legal response to a failure is the rollback named in §4.
- **Freeze 27 Oct, hard stop 30 Oct.** Codabench closes 1 Nov at 11:55 UTC per the competition page
  (`docs/COMPETITION.md` records 12:55; the earlier time governs). If anything is unresolved on 27 Oct,
  the incumbent is sent and work continues until 30 Oct — the Google Form can be resubmitted.

## 6. Not done here

- **No leaderboard selection.** Each surviving candidate is uploaded at most once, to detect a broken
  ZIP (0.00, invalid predictions, timeout), which is the only failure invisible locally. A score is
  never used to choose between candidates (owner's rule, 14 Oct). A candidate below 0.50 is treated as
  suspected broken and inspected, not discarded.
- **No change to E57.** It stays as pre-registered, with one correction: its ship rule 3 currently
  requires beating the constants' 0.529. That is raised to always-robust's **0.721** on the 43 rows,
  which is what gate G2 required in spirit on 10 Oct before E57 lowered it. A probe that cannot beat a constant does not ship.
- **No new labelling run on our compute.** E58, E59 and E60 closed that (Findings 83, 84, 85).

## 7. Amendments

**Revision 1 (23 October 2026, before any E61 code, artifact or result existed).** B1 was written as
"precision at least 0.60 over at least 3 flips" and is replaced by the binomial criterion now in §4.
This tightens the gate and makes it scale-free; it is recorded here rather than silently edited.

The first form had two defects. Its false-pass rate under the null was high: flipping 3 rows blindly
at this population's 0.294 base rate reaches 2 of 3 with probability **0.209**, so roughly one run in
five would have passed on noise. And a fixed ratio penalises exactly the detector we want — one that
flips few rows and is almost always right. Under the binomial form, 3 of 3 passes at p = 0.0254 while
3 of 5 does not, which is the correct ordering.

Power of the revised rule, if the detector's true precision is 0.70: 0.240 at 4 flips, 0.528 at 5,
0.552 at 8, **0.850 at 10**. Low power at small flip counts is accepted rather than engineered away —
the alternative is a gate that passes on noise, which is how this project accumulated uninformative
nulls (RESULTS, "E45: power of the lanes we closed"). It is also the reason the detector is run at its
own coverage: a mechanism that naturally flips about ten of these rows gets a well-powered test, and
one that flips three gets an honest, weak one.

**Amendment 1 (23 October 2026, after R5 of the sweep returned, before any detector code existed).**
The C2 mechanism is fixed to original-problem uncertainty (§3), premise-ablation is dropped, and gate
**B4** is added (§4).

R5 did the job C0 exists for. It found a competing entrant had already run our leading candidate and
scored 0.38 on the official rows, so three days of building were avoided. It also found the mechanism
with the strongest evidence behind it, and that two of its three sources are written by this
competition's own organizers — which is as close to a stated preference as this task offers.

Three further R5 results are recorded here because they bear on gates already written:

- **Always-robust scores 0.68 on the official rows** by an independent entrant's count, against our
  0.706. Gate A's premise is confirmed from outside this repository.
- **Honest small-track entries cluster at 0.64.** EconLearn reports 9 of 14 and byeolki 0.643; `e27f`
  is 9 of 14. The field's visible 0.93 to 1.00 is therefore not the field's method quality, which
  supports §6's refusal to read the board for selection.
- **The budget is about 35 s per pair** if 70 minutes covers roughly 30 problems × 4 models. B3's 4,000
  tokens stands, but a truncated trace of a few hundred tokens is the configuration the literature
  actually supports, and it is what §3 now specifies.

R1 is downgraded: its leading hypothesis is the one R5 refuted. **R3 is promoted** — setting one pooled
threshold from 34 rows is now the binding methodological question, not an aside.

---

## Appendix A — Literature sweep (C0)

The five prompts live in `../docs/LITERATURE_SWEEP.md`, with an answer slot and a status table under
each. They are kept there rather than here so that results can be appended as they arrive without
editing this frozen protocol.

**What the sweep decides.** A mechanism qualifies for C2 only with at least two independent papers
reporting a usable effect size. If none clears that bar, C2 does not run and `e61a` ships. The chosen
mechanism is recorded as amendment 1 of §7, dated, before any detector code exists.
