# E60: premise-level contradictions written by an LLM, on the same gate

Status: **pre-registered 20 October 2026, before any E60 variant or record existed.** Code:
`datagen/perturb/premise_variants.py` (generation and check), `scripts/e58_gpu_bundle.py` (the `e60-*` jobs),
`train/eval_e58.py --experiment e60`. Any change after the first generated variant is read is an amendment,
dated in §9.

## Motivation

E58 and E59 failed the same way (Findings 83, 84): our models almost never declare an answer-property
contradiction unsolvable, and permission to decline barely changed that (0 of 192 to 6 of 216). Two
differences from the organizers' setting remain. **The variant:** a claim about the answer reads as a wrong
hint, while their `expert_no_solution` contradicts the problem's own objects. **The budget:** their 100k
tokens against our 16k-64k. E60 changes the first and keeps everything else as E59 had it, so the variant's
effect is measured alone. The budget stays out of our reach and is left to the collaborator's run.

## 1. Population

The same 27 official rows and 14 problems as E58 and E59.

## 2. Variants (fixed now; `premise_variants.py`)

- **Generator:** `deepseek/deepseek-v4-pro` through OpenRouter, temperature 1.0, under `GENERATION_PROMPT`.
  It is given the problem and its answer and returns two sentences, each stating a condition about the
  problem's own objects that contradicts a different part of it. It writes only the sentences.
- **Placement:** `insert` puts each sentence before the question when premises precede it and the text is
  plain (no display math, several paragraphs or open quotation). Otherwise it is appended. The original text
  is preserved exactly; for these 14 problems that is 5 before the question and 9 appended.
- **Check, blind:** `anthropic/claude-sonnet-5.5`, temperature 0, under `CHECK_PROMPT`. It sees each variant
  and, as a control, the original, without being told anything was added. A variant is kept iff the checker
  calls it CONTRADICTORY and the original SOLVABLE.
- **One retry:** a problem with fewer than two kept variants gets one more generation round with the same
  prompt. Whatever is kept is used. A problem with none kept leaves its rows unlabelled under §4 (fewer than
  4 contradiction samples).
- **The prompts, models and placement rule are not edited after any output is read.** Every request and
  response is saved to `data/e60/raw/`.

## 3. Sampling: only the contradictions are new

- **Reused from E59, already read:** each row's base samples (eligibility) and its two consistent-premise
  variants. They used the same prompt, decoding, k and caps, and they only enter the labels.
- **New:** the kept premise contradictions, k = 4 each. Same settings as E59: the organizers' decoding, the
  prompt that can decline (`--allow-no-solution`), the same caps (Qwen 65,536; DeepSeek and Skywork 32,768;
  Olmo 16,384), the same engines.
- **Kernels:** Qwen 5, DeepSeek 4 (one item per flush), Skywork 1, Olmo 8. Each runs the variant phase only,
  with its model's E59 base records bundled.

## 4. Labels and gate (E58's, unchanged)

- E58 §4 grading: `eval_e52.grade` for contradictions, the answer for consistent premises.
- No-solution family: the E60 contradictions. Consistent family: E59's.
- The official max_drop rule.
- **Gate:** at least 18 of 27 rows labelled, and kappa >= 0.30 with a problem-bootstrap lower bound above 0
  (seed 58).

## 5. Reported, not gated

- The generator's yield: kept / checked, per round.
- Contradictions flagged, E59 (answer-property) against E60 (premise), per model.
- The same agreement, confusion and deciding-family figures as E58.

## 6. Multiplicity

This is the fourth labeller scored against these 27 rows (E24b, E58, E59, E60). The threshold is not
relaxed, and any pass is reported with that count beside it.

## 7. Prediction, recorded now (not a gate)

**Detection stays low and the gate fails.** E52 measured human-written premise contradictions (ReliableMath)
on Qwen and Skywork at 16k: 2 of 130 and 0 of 274. These models deliberate past our caps. A pass would mean
the variant style, not the budget, was the barrier.

## 8. Decisions

- **Gate passes:**
  - E57 §1 is amended, dated, to accept this labeller.
  - The generator is run on E55's problems with known answers.
  - The collaborator gets those variants with the decline-capable prompt for the full 100k-token run (E61,
    pre-registered then).
- **Gate fails:**
  - The report carries E24b, E52, E58, E59 and E60 together.
  - The collaborator still gets the generator and prompt, since the 100k budget is the one difference
    left untested.
  - `e27f` stays.

## 9. Amendments

**1, 20 October, before any variant was generated.** The paid models could not run: the account behind the key
has no credit (HTTP 402 on the first call, which produced nothing). The owner cannot add credit. The generator
becomes `nvidia/nemotron-3-ultra-550b-a55b:free` and the checker `minimax/minimax-m3.1-flash:free`, two
different families. Both are free and rate-limited, so the script paces its calls, stops at the daily limit,
and resumes where it stopped. Prompts, temperatures, placement, the keep rule, the retry round, sampling and
the gate are unchanged. If either model proves unusable on the first problem (no parsable output), that is
recorded here before switching to the next free model of another family. (Both worked; nothing switched.)

**2, 20 October, before any E60 record existed.** Generation stopped at the first key's daily limit with 11 of 14
problems done (2 kept variants each), and resumed on the second key for the other 3. The 12 kernels whose
problems were all done were launched at once, with the 11-problem variants file. A problem's variants are final
once written, so each kernel samples exactly the variants it would have had. The other 6 kernels are launched
when the last 3 problems are done. Kernels went to accounts by GPU hours left this week
(`scripts/kaggle_quota.py`), never DIDHITI (over quota), RSD (read-only) or MUBAMAN (no GPU).
