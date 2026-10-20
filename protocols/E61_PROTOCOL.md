# E61: the familiarity ranking as an absolute per-problem threshold

Status: **pre-registered 21 October 2026, before any E61 threshold or decision below was computed.**
Code: `train/eval_e61.py` (thresholds and checks C2-C4), `submission/make_e61.py` (the ZIPs),
`scripts/e61_gpu_bundle.py` (C1 and C5). Any change after the first number is read is an amendment,
dated in §8.

## Motivation

`e27k` called the more familiar problems of each model's batch robust (lower mean token NLL over the
bare problem) and scored 12/14 under batch evaluation. Since 15 October the organizers call
`are_robust` once per problem (Finding 79), so a ranking cannot act and `e27k` collapses to its
constants (9/14). E50 (Finding 75) shows the ranking was right on every DeepSeek, Skywork and Olmo
case, so the statistic is worth up to three cases if it can make an **absolute** decision. E61 asks
whether a threshold fixed on unlabelled problems carries that value to new problems.

**This is not a blind test, and it says so.** E54 (Finding 80) already scored the median-threshold
version of this rule on the official rows (0.458) and on DeepSeek MATH (0.638 [0.549, 0.726]), and
E50 fixed the hidden set's labels *by familiarity rank*. What nobody has seen is where the hidden
problems' NLLs fall against a fixed threshold, which is exactly what the board read measures, and the
balanced accuracy of the prior-tilted thresholds below on any population.

## 1. Score

Mean token NLL of the bare problem under the model being scored: no system prompt, no chat template,
the tokenizer's default special tokens, truncation at 2,048 tokens, logits cast to float32 before the
log-softmax. This is `datagen/extract_familiarity.familiarity`'s `mean_nll`, the function that
produced every reference value below; the submission computes it with that function's arithmetic.
Lower NLL = more familiar = **robust**, the direction `e27k` shipped.

## 2. Reference set and thresholds

**Reference set R:** E55's 548 unlabelled problems (the 36 official problems, ReliableMath AIME and
AMC, the olympiad pool), with each model's `mean_nll` from `runs/e55-features-*/` (T4, bf16, the
pinned transformers). No label enters a threshold.

| id | role | robust iff |
| --- | --- | --- |
| E61b | **the candidate** | NLL ≤ the `p_m` quantile (numpy default, linear) of model `m`'s 548 reference NLLs |
| E61a | control, no new choice | NLL < E54's frozen 75-problem AMC median: qwen 1.269, skywork 2.012, olmo 1.579, deepseek 1.538 |

`p_m = (n_m + d_m) / (2 n_m)`, with `d_m` the robust-minus-spurious balances the five constant probes
measured (Findings 45, 64: Qwen +1, DeepSeek +1, Skywork -1, Olmo -1) and `n_m` the case counts of
E50's solved composition (Qwen 5, the others 3):

| model | `p_m` |
| --- | --- |
| Qwen/Qwen3.5-4B | 0.6 |
| deepseek-ai/DeepSeek-R1-0528-Qwen3-8B | 2/3 |
| Skywork/Skywork-OR1-Math-7B | 1/3 |
| allenai/Olmo-3-7B-Think | 1/3 |

These are the same balances the per-model constants already rest on (`e27f`, PLAN §12.8); E61b only
lets the score move individual problems across them.

Any model id outside the four, and any failure inside a call (no cached weights, a tokenizer that fails
its round trip, an exception), returns that model's constant (`True` for Qwen and DeepSeek, `False`
otherwise), as `e27f` does. The checkpoint is loaded once and reused across calls; nothing else carries
between calls.

## 3. Checks before any upload

| id | check | pass |
| --- | --- | --- |
| C1 | parity: the ZIP's NLL against E55's stored `mean_nll` for the same problems, on all four real checkpoints | max abs diff ≤ 1e-3 |
| C2 | E61b on DeepSeek's 142 Hendrycks MATH instances (P2 of E54, same loader and bootstrap: 2,000 problem draws, seed 61) | balanced-accuracy 95% interval lower bound > 0.5 |
| C3 | E61a and E61b on the 34 official rows (P1 of E54) | **reported only** (§12.3: the batch split reverses there) |
| C4 | E61b's thresholds against E54's medians; the share of R within 0.01 nats of each threshold | **reported only** |
| C5 | 60 one-problem calls through the shipped `are_robust`, model loads included, on a T4 pair | under 5 minutes in total |
| C6 | `getting-started/scripts/run_local.py <zip> --small` and `submission/docker_load_test.py` | coverage 1.0, 0 invalid, Docker passes |

## 4. Board reads

At most two uploads, made by the owner: E61a, then E61b, so the board ends on the candidate. A read is
a veto, never a selector: no threshold, quantile, reference set or per-model choice changes after it.

## 5. Ship rule (decides the Google Form ZIP at the 27 October freeze)

**E61b replaces `e27f`** if C1, C2, C5 and C6 pass **and** its board read is at least `e27f`'s 9/14.
Otherwise `e27f` ships. E61a's read is reported whatever it is and does not change the choice, including
when it scores above E61b.

## 6. What each E61b read means

- 12/14: the threshold places every non-Qwen hidden problem where the batch ranking did.
- 10-11/14: it does on some models. A gain over 9; ships under §5.
- 9/14 or below: the hidden problems' NLL scale differs from R (for example, less familiar than the
  published problems R is made of), and `e27f` ships.

## 7. Not done

No per-case fit of the 14 validation cases; no use of `train-main-v2` revision `25e45665`, which still
holds 7 validation labels (Finding 75); no state carried across calls to calibrate on the test problems
themselves; no third upload of an E61 variant.

## 8. Amendments

None yet.
