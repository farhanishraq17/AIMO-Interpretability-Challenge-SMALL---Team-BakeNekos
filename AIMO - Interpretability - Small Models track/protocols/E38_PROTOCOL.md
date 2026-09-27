# E38: exact-prefix audit for cheaper budget-path inference

Status: frozen before inspecting cross-budget trace equality. This is an execution-cost
experiment, not a new robustness predictor or another test of E37's labels.

## Question

Can one longest greedy trace supply the shorter-budget primary traces exactly? If so,
we may avoid repeated decoding while keeping E37's four budgets and feature definitions.
Forced-answer calls would still run separately at each budget. Temperature zero alone
does not establish equivalence: batch scheduling and numerical differences can alter tokens.

## E38a: archive screening (CPU, run first)

- Dataset: `ssubhnil/Qwen3-Math-Eval`, revision
  `f64c82f5ce9249155ad7c70991a9711d92963cce`.
- Use all four Qwen3 model shards in `math_perturb` and `gsm_plus`.
- Select only `base` (MATH) or `problem understanding` (GSM) rows at budgets
  2048, 4096, 8192 and 16384. Do not read correctness, answers, question text, or labels.
- Require exactly one row per source/model/problem/budget, nonempty traces, temperature
  zero and seed 42. Each model/source cell must have at least 50 complete problems.
- For each shorter budget compare its raw stored trace with the start of the 16384-token
  trace, byte for byte after UTF-8 encoding. Do not strip whitespace, alter text, or filter
  on forced-answer status. Report counts by model, source, budget and forced status.
- Gate: **zero mismatches in every cell**. Any mismatch closes archive-based trace reuse;
  preserve E37's independent budget generations. No near-match tolerance or selective reuse.
- Store hashes and counts, not traces, in the tracked result. This audit does not compute
  or reuse an accuracy metric. Neither success nor failure changes E37's scientific gates.

## E38b: target-engine equivalence (conditional draft)

Only if E38a passes, run a GPU equivalence test on the four actual Small-track models,
using the first six distinct official problem keys in sorted order, without inspecting
their robustness labels. Freeze the exact implementation and record model revisions first.
Compare independent-budget token IDs, all forced answers, all 18 features, and final scores
against a longest-trace implementation. Require exact token/answer equality, feature equality
to 1e-12, identical decisions, and at least 20% lower warm generation runtime on each model.
Failed equality rejects the optimization; do not change budgets, precision, rank fractions,
features or coefficients to rescue it. Cap the conditional pilot at one 2-T4 session per
model and reassess capacity before launching. E38b never overrides E37's existing runtime gate.

No E38 Codabench upload is authorized by the CPU audit alone.
