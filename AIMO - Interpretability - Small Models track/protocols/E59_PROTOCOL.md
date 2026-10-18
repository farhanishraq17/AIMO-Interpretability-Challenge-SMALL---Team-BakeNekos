# E59: E58's labeller with one change, a prompt that can decline

Status: **pre-registered 19 October 2026, after E58 closed and before any E59 record existed.** Code: E58's,
with `datagen/perturb/sample.py --allow-no-solution`, `scripts/e58_gpu_bundle.py` (the `e59-*` jobs) and
`train/eval_e58.py --experiment e59`. Any change after the first E59 record is read is an amendment, dated in §8.

## Motivation

E58 (Finding 83): no model flagged one of 192 answer-property contradictions, so every row was labelled
spurious and the gate failed with kappa 0.000. Two things separate that from the organizers' setting, and
the evidence points at the first:

1. **The prompt.** E58 used the DeepSeek suffix, "Please reason step by step, and put your final answer
   within \boxed{}.", which offers no way to say there is no answer. The organizers grade
   `expert_no_solution` on detection (Finding 83: in their sample its accuracy is decoupled from the
   original's), so their prompt must accept a no-solution answer, and `qwen3-8b:low` reaches 0.56 there.
   Our models' own endings name the clash and then box the computed value, calling the claim a mistake.
2. **The variant.** A claim about the answer can read as a wrong hint; their premises contradict the
   problem's own objects.

E59 changes only the first, so its effect is measured alone, and the labeller stays fully automatic.

## 1. What changes

The prompt suffix becomes (`sample.NO_SOLUTION_SUFFIX`, fixed now, never tuned):

> Please reason step by step, and put your final answer within \boxed{}. If the problem has no solution,
> put \boxed{\text{no solution}} instead.

It applies to every prompt: originals, contradictions and consistent premises alike, so false abstention
costs eligibility and consistent-family accuracy exactly as it would under the organizers' rule.

Kernel layout only: Qwen's 9 problems go to five kernels instead of three, since `e58-qwen-c` hit its
7.5 h guard, and Skywork's 3 to one kernel.

## 2. What does not change

Everything else in `protocols/E58_PROTOCOL.md` §1-§5, by reference:

- the population: the same 27 rows and 14 problems;
- the variants: the same `data/e58/variants.parquet`;
- decoding, k = 4, the caps (Qwen 65,536; DeepSeek and Skywork 32,768; Olmo 16,384) and every engine
  setting;
- the grader (`eval_e52.grade`; a boxed "no solution" is `detected`);
- eligibility, family accuracies, the label rule;
- the gate: at least 18 of 27 rows labelled, and kappa >= 0.30 with a problem-bootstrap lower bound above 0
  (seed 58).

## 3. Reported, not gated

- Per model, contradiction samples flagged in E59 against E58: the effect of the prompt, row by row.
- False abstention: originals and consistent premises answered "no solution".
- The same agreement, confusion and deciding-family figures as E58.

## 4. Multiplicity

This is the third labeller scored against these 27 rows (E24b, E58, E59). The threshold is not relaxed for
it, and any pass is reported with that count beside it.

## 5. Prediction, recorded now (not a gate)

- **Detection rises for Qwen and DeepSeek,** the two that named the clash in E58: at least a quarter of
  their contradiction samples flagged.
- **It stays near zero for Skywork,** which almost never mentioned it (1 of 24).
- **Olmo stays mostly out of budget at 16k.**
- **The gate most likely still fails:** labels would then vary by model more than by problem, and the
  official rows are mostly robust for every model. A pass is possible if detection varies within a model
  across problems the way the official labels do.

## 6. Decisions

- **Gate passes:** E57 §1 is amended, dated, to accept this labeller. The collaborator is told to use
  this prompt. Labelling E55's problems with a known answer is pre-registered as E60, scoped to the fleet's
  remaining quota. E57 runs only on those labels and ships only under its own rule.
- **Gate fails:** the self-labelling route is closed on our compute for this competition, and `e27f` is
  the submission.
- **Either way:**
  - the detection change goes to the report;
  - the collaborator is told about Finding 83, because a prompt without a way to decline is a candidate
    reason any labeller misses the official labels.

## 7. Not done

- No wording, variant, k, cap, threshold or grading change after the first record is read.
- A kernel that fails for an engine or host reason is relaunched unchanged, or resumed from its own records
  as in E58's amendment 1.
- Nothing is uploaded to the leaderboard.

## 8. Amendments

None yet.
