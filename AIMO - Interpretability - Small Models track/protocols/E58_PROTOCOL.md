# E58: does a full-budget answer-property labeller reproduce the official labels on the official rows?

Status: **closed 18 October 2026: the gate failed as predicted** (kappa 0.000; every labelled row spurious, 0 of 192 contradiction samples flagged; PROGRESS Finding 83). `e27f` stays.
Pre-registered 18 October 2026, before any E58 record existed. Code:
`datagen/perturb/e58_variants.py` (variants), `scripts/e58_gpu_bundle.py` (kernels), `train/eval_e58.py`
(grading, labels, gate; checked by `train/test_eval_e58.py`). Any change after the first record is read
is an amendment, dated in §9.

## Motivation

E57's probe needs self-labels that reproduce the official ones (E57 §1). The collaborator's full-budget
run has not reported, and the owner asked the fleet to test a labeller now (18 October). E24b tested one
on these same rows: `answer_modulus` variants at a 24k cap reached Cohen's kappa 0.286 with only 15 of
27 rows labelled, and its gate failed (Finding 48). That labeller measured the wrong family. Finding 76
names the family that decides olympiad labels, `expert_no_solution`, and E52 showed it cannot be read
at 16k (Finding 77). E58 builds that family automatically and samples at the longest caps a T4 pair
holds.

## 1. Population

- The 27 (model, problem) rows of `train-main-v2` (revision `23791d9b`) whose problem has a reference
  answer from E24a (6 known, 8 from 8-of-8 unanimous Skywork samples; `answer_key.py`): Qwen 9,
  DeepSeek 7, Olmo-3-Think 8, Skywork 3, over 14 problems. The other 7 official rows have no answer,
  so they cannot carry answer-property variants, and they are not used.
- The 7 validation cells exposed by revision `25e45665` (Finding 75) are never read.
- The official labels are read once, by `train/eval_e58.py`, after every kernel has finished or failed.

## 2. Variants (automatic, from the reference answer N)

Each problem gets four variants: its text, a blank line, and "It is also known that the answer is P."

| category | family | P |
| --- | --- | --- |
| `contradict_parity` | no_solution | the parity N does not have ("an odd number" / "an even number") |
| `contradict_bound` | no_solution | "greater than 2N + 10" |
| `consistent_parity` | consistent | the parity N has |
| `consistent_bound` | consistent | "less than 2N + 10" |

The contradictions mirror `expert_no_solution`: an added premise inconsistent with the rest, so no
answer satisfies the problem. The consistent premises mirror `expert_perturbations`. The generator
needs only the answer, so it extends to any problem with a known integer answer, including E55's 548.

**The main threat to validity, stated now.** The organizers' added premises are about the problem's
objects ("the system registers at least 21 departures" where at most 15 are possible). Ours are about
the answer, so a model can only notice them after solving. The gate measures whether that matters.

## 3. Models, decoding, budgets

| item | value |
| --- | --- |
| decoding | the organizers': temperature 1.0, top-k 40, top-p 0.95; chat template; the E7 prompt suffix |
| samples | k = 4 per original; k = 4 per variant, so 8 per family |
| caps | Qwen 65,536 (one engine per T4); DeepSeek 32,768 and Skywork 32,768 (tensor parallel 2); Olmo-3-Think 16,384 (tensor parallel 2, eager, utilisation 0.85, one sequence at a time: Finding 78) |
| engine | vLLM 0.20.0 in its own venv after the CUDA linker check (E52's recipe); `transformers` 5.5.3 for Olmo |
| kernels | each model's sorted problem keys are sliced: Qwen 3 kernels, DeepSeek 4, Skywork 2, Olmo 8 (one problem each); each kernel runs both phases |
| time guards | 2.5 h base, 7 h variants per kernel, each plus half an hour for the process guard |

The organizers sample to 100k tokens. These caps are the most each model's T4 layout holds within a
session (Olmo's cache holds 20,288 tokens). Every cap stays within the model's context length.

## 4. Grading and labels (fixed; `train/eval_e58.py`)

- **Base or consistent sample:** correct iff it finished within the cap and its answer equals N.
- **Contradiction sample:** correct iff E52's grader (`eval_e52.grade`, audited 30 of 30) calls it
  `detected`: finished, no integer answer, and an ending that says there is no valid answer.
- **Eligible** iff all 4 base samples are correct (the official eligibility rule, base accuracy 1.0).
- **Family accuracy** pools a family's two variants (8 samples). `max_drop = 1 - min` of the two.
- **Label** by the official rule: robust iff `max_drop` <= 0.10, spurious iff >= 0.25, band between.
  With 8 samples, robust needs 8 of 8 in both families, 7 of 8 is band, and 6 or fewer is spurious.
- A row is **labelled** iff it is eligible, both families have at least 4 graded samples, and it is
  not band. Rows not eligible at the cap, unfinished, or band are unlabelled.
- The five surface families are not sampled: on solved olympiad problems they almost never cost 25
  points (Finding 76).

## 5. Gate (one read)

**Passes iff both:**

1. at least 18 of the 27 rows are labelled; and
2. Cohen's kappa between self-labels and official labels on the labelled rows is at least 0.30, with
   a problem-bootstrap 95% interval (2,000 resamples of the 14 problems, seed 58) whose lower bound is
   above 0.

This is E24b's gate (G-V2) plus a coverage floor, so the two labellers compare directly.

**Reported, not gated:**

- per-model contradiction outcomes (detected, answered, answered with the original answer, truncated);
- which family set `max_drop`;
- the agreement of the self-labels and of `e27f`'s constants with the official labels on the labelled
  rows;
- the confusion table.

## 6. Prediction, recorded now (not a gate)

**The gate fails.** E52 found Skywork answering through every contradiction (0 of 274 flagged) and Qwen
flagging 2 of 128. A labeller that adds a contradiction to every problem then labels every such row
spurious. The official labels instead depend partly on whether the organizers wrote a contradiction
for that problem at all: only 28 of the 38 eligible rows in their sample carry `expert_no_solution`
(E51), and these 27 rows are 78% robust. The self-labels would then track "this model detects", not
"this problem was perturbed that way", and kappa stays near 0 unless detection at the longer caps
varies by problem. Qwen at 65k is the open case: E52's Qwen noticed the clash in truncated traces and
ran out of budget deciding.

## 7. Decisions

- **Gate passes:** E57 §1 is amended, dated, to accept this labeller in place of the collaborator's
  gate. Labelling E55's 548 problems with it is pre-registered as E59 with its compute. E57 then runs on
  those labels under its own ship rule. No submission comes out of E58 itself.
- **Gate fails:** the answer-property labeller is closed. `e27f` stays the submission, and E57 waits
  for the collaborator's gate, still its only precondition. The report carries E24b, E52 and E58
  together, as three labellers measured against the same official rows.
- **Either way**, the per-model detection rates at the longer caps go to the report, as the first
  measurement of the deciding family beyond 16k tokens on our models.

## 8. Not done

- No wording, template, k, cap, threshold or grading change after the first record is read.
- A kernel that fails for an engine reason may be relaunched unchanged, or with an engine-only fix
  recorded in §9 before its records are read. Decoding, k, caps and grading never change.
- Nothing from E58 is uploaded to the leaderboard, and the leaderboard selects nothing.

## 9. Amendments

**1, 18 October, after every kernel ended and before any E58 record's outcome was read.** `e58-qwen-c`'s
variant phase was stopped by its process guard at 7.5 h (`wall_time_limit`) with 8 of its 12 variant items
sampled; the other 16 kernels finished every phase. Two Qwen rows each lack one variant per family. They would
still be labelled under §4 (4 samples per family), but on half the samples of every other row. The four missing
items are therefore sampled by `e58-qwen-c-resume`: the same settings, the sampler's own resume path (it skips
items already recorded), and the kernel's own base records, so eligibility is not re-drawn. Only item ids and
exit codes were looked at to decide this. Decoding, k, caps, grading and the gate are unchanged. If the resume
kernel also fails, the rows are read as they stand under §4. The TASHIN copy was still queued after 2 h, so
the identical kernel was also pushed to Nasif (18 October, 13:00 UTC): whichever copy completes first is used,
and the other's output is not downloaded. `e58-skywork-b` moved from MUBAMAN, which had no
GPU, to Nasif before it ran (§8).
