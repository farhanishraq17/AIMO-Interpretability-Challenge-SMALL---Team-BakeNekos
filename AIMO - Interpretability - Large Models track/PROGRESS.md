# Large Models (Main) Track: Progress

Narrative log of what was built and measured. Findings are numbered **LF-n**, separately from
the small track's `PROGRESS.md`. Every number comes from a script in this folder; the tables
behind them are in `RESULTS.md` (regenerate with `scripts/write_results.py`).

## 1. What exists (27 Sep 2026)

| Piece | What it does |
| --- | --- |
| `scripts/download_data.py` | Mirrors every public `aimo-interp` HF dataset at a pinned revision into `data/hf/`; records failures (`problems-public` still answers 401) in `data/MANIFEST.json` |
| `scripts/build_items.py` | Builds the panel list `data/panel/items.jsonl`: 36 `train-main-v2` originals + 10 reference originals (k=4) and 453 official variant texts (k=1); official per-variant outcomes go to `official_variants.jsonl` (821 rows) |
| `scripts/klarge.py` | Main-Track Kaggle launcher: H100/T4 shapes, model and kernel mounts, payload embedding, per-account tokens, quota check (works around the CLI 2.2.2 `kaggle quota` crash), `runs/launches.json` log |
| `jobs/probe.py` | L0 hardware probe |
| `jobs/ollama_panel.py` | L1 panel: Ollama on T4x2, append-per-completion `records-<payload sha>.jsonl`, sample-major order so a timeout leaves full coverage at lower k, resumes from keys earlier sessions finished |
| `scripts/write_results.py` | Regenerates `RESULTS.md`, including the coverage table LF-11 needed |

The small-track launcher (`datagen/kaggle_launch.py`) was not reused: it hard-codes T4x2, the
transformers pins and the small-track bundle, and has no model or competition mounts.

## 2. Findings

**LF-1. The Main Track has 71 official labels, 37 of them for gpt-oss-120b, and "robust" is the
majority everywhere.** `train-main-v2` holds 82 rows over 36 problems; 71 carry a label, 52 of
them robust (0.732). gpt-oss-120b has 7 labelled rows at `low`, 15 at `medium`, 15 at `high`,
with robust shares 0.714, 0.667 and 0.867. The best constant is therefore "robust" for every
(model, effort) group, and it is the floor for any gpt-oss method. With 7 to 15 rows per effort,
only a large gain over it would be measurable.

**LF-2. The reference-problem table carries the official perturbed texts, which makes proxy
calibration possible.** `aimo-interp-challenge-sample-full` covers 10 AIMO-2025 reference problems
for 8 systems. Its `permutations_causing_decay` column lists each official variant's text and
per-variant accuracy: 453 distinct variants across `rephrase`, `typos`, `domain`, `distract`,
`rename`, `expert_*` and `paraphrase`. Among the 8 systems, `gpt-oss-120b:low` has the largest mean
absolute decay (0.147, base 0.483 to permuted 0.336) and `qwen3-8b:low` the smallest (0.006). Those
two are the official references our proxies are checked against. Calibration use only: see the
extra-data constraint in the Plan.

**LF-3. Kaggle's `NvidiaH100` shape gives T4x2 on our accounts.** All 4 L0 probes (ISHRAQ,
Mashrafi, Samin, Nasif) got 2x Tesla T4 with internet on and the `danielhanchen/gpt-oss-120b`
weights mounted. The AIMO3 H100 notebooks attach the AIMO3 competition; our pushes that attached
it were refused because no account has accepted its rules. Whether accepting them unlocks H100 is
untested. Until then gpt-oss-120b cannot run on the fleet, so the panel uses gpt-oss-20b as a proxy.

**LF-4. The first L1 launch died at warm-load; fixed and smoke-tested before relaunch.** Seven
panel kernels were pushed at 15:03 UTC on 25 Sep (the eighth, gpt-oss low, was refused: ISHRAQ's
probe held its second slot). Every one that reached the model died about two minutes after load:
the warm-load `/api/generate` call omitted `"stream": false`, so Ollama returned NDJSON and
`json.loads` raised "Extra data". The sampling calls already disabled streaming. Five had errored
by 15:14 UTC; the two gpt-oss kernels were still pulling weights and carry the same bug. Loss was
minutes per kernel. The fix is one field; two short smoke kernels (`l1-smoke-qwen3-4b`,
`l1-smoke-gptoss20b` with `low`+`high`, 4k context, 6-minute deadline) went to MUBAMAN at
15:10 UTC to prove the whole path, including gpt-oss effort levels, before 8.3-hour sessions are
spent again. The qwen3-4b load also showed all 37 layers on the two GPUs with ~7 GB free on each,
so the T4x2 placement itself works.

**LF-5. MUBAMAN cannot run GPU jobs yet.** Both MUBAMAN smoke kernels were accepted by the push
but ran with no GPU (`nvidia-smi: not found`) and no internet (`Could not resolve host:
ollama.com`), and failed in under a minute. That is how Kaggle treats an account without phone
verification; the owner has to verify it on the website. The panel now exits on a missing GPU
instead of failing later, and the smoke tests were re-pushed to RSD at 15:21 UTC with longer
deadlines (12 and 15 minutes) so gpt-oss's 14 GB pull fits before sampling starts. The owner
reported verifying MUBAMAN at ~15:22 UTC, but its two panel kernels pushed at 15:23 still had no
GPU; the new guard stopped both within seconds, and they moved to Samin at 15:28.

**LF-6. The panel works end to end, but a T4x2 kernel decodes only ~90-100 tokens/s, so the long
jobs need continuation sessions.** The RSD smoke tests (4k budget, 15:21-15:37 UTC) loaded
`gpt-oss:20b` and `qwen3:4b` 100% on GPU and wrote 37 and 24 completions with no errors
(Results, "L1: panel run health"). gpt-oss at `low` put `\boxed{}` in 97% of replies, averaged 1,759
generated tokens and truncated 3%. qwen3-4b hit the 3,072-token cap on every item while still
thinking, as expected at that budget. Aggregate decode was 90.4 tok/s (gpt-oss) and 96.9 tok/s
(qwen3-4b), about 12 tok/s per stream over 8 streams; qwen3-4b used one GPU and left the other idle.
Estimate from those rates, not a measurement: gpt-oss `low` needs ~3.5 h for all 637 completions
and finishes, but a thinking model near a 16k budget needs roughly 20 h, so the 8.3 h first-launch
jobs will stop partway through sample 0 of the variants. Sessions now embed the keys earlier
sessions finished and write `records-<payload sha>.jsonl`, so a continuation resumes instead of
restarting and its download does not overwrite the previous one. The Nasif gpt-oss kernels from the
first launch were still running at 15:37, so gpt-oss survived the warm-load bug (its first token is
presumably swallowed by the harmony parser, leaving a single NDJSON line).

**LF-7. gpt-oss:20b at `low` finished the whole panel in one session; 1.1% of its replies are
tool calls Ollama cannot serve.** `l1-gptoss20b-low` (Samin, 15:28-19:12 UTC) wrote all 637
completions: 1.18M generated tokens at 89.6 tok/s, mean 1,876 tokens, none truncated, 98% with a
`\boxed{}` answer (Results, "L1: panel run health"). 7 completions failed with HTTP 500. The serve
log pairs each one with `harmony parser: no reverse mapping found for function name
harmonyFunctionName=python`: gpt-oss tried to call its built-in python tool, which the panel does
not provide. Seeds are fixed per sample, so a plain retry would most likely reproduce the call.
They stay recorded as errors and are graded as no answer. If `medium`/`high` show a much higher
rate, a retry with a fresh seed is the fix.

**LF-8. Idle accounts now run backwards "helper" copies of the slowest jobs.** A `-b` job samples
the same model and effort as its base job but walks the work list in reverse, so it meets the
forward session instead of duplicating it. For gpt-oss `medium`/`high`, that also puts the three
extra samples of each original first, the part G-L1b needs and the forward runs reach last.
Finished keys are now shared by model and effort across job folders, so a continuation of the base
job skips whatever a helper finished once both are downloaded. Pushed 19:47-19:50 UTC on 25 Sep:
DeepSeek-8B-b and qwen3-32b-b on RSD, gpt-oss medium-b and high-b on ISHRAQ. MUBAMAN was tried
first and still got no GPU (LF-5). Eshad refused with "Maximum batch CPU session count of 5
reached": another project's five `kagri-mk2-*` kernels fill that account's session limit, and the
same holds for every account running five of them.

**LF-9. First label-free read of gpt-oss:20b `low`: answers are unstable even on the originals,
so the planned calibration needs real reference answers.** Results, "L1: answer stability". Over
the 46 originals (k=4), 64.1% of samples give their problem's most common answer, and only 12
problems are unanimous. Only 3 of the 10 reference problems are unanimous, so variant stability
(75.4% of 122 variants keep the problem's answer) rests on 3 problems and says little per family.
Against the small track's E24a key (14 `train-main-v2` problems; unanimous Skywork samples, not
official answers), 76.8% of original samples match. No gate is decided on this: G-L1a needs
correctness against true answers for the 10 reference problems, which nothing in the repo has.
G-L1b needs the `medium`/`high` runs, since `low` has only 7 labelled gpt-oss-120b rows.

**LF-10. Every L1 kernel has finished, and eight outputs are still on Kaggle.**
(`klarge.py status`, 26 Sep.) **Corrected by LF-11: "finished" is kernel status, not coverage.**
Of the eight (model, effort) cells, one is complete.

- All 12 panel kernels report COMPLETE: the three `gpt-oss:20b` efforts, `deepseek-r1:8b`, the
  four Qwen3 rungs, and the four `-b` helpers. So do the four L0 probes and the two smokes. **No
  continuation is needed**, which is what the 26 Sep schedule row had assumed.
- **Downloaded so far:** `l1-gptoss20b-low`, `l1-gptoss20b-high-b`, `l1-qwen3-32b`,
  `l1-qwen3-4b`. **Still on Kaggle:** `l1-gptoss20b-medium`, `l1-gptoss20b-high`, `l1-dsr1-8b`,
  `l1-qwen3-14b`, `l1-qwen3-30b-a3b`, `l1-dsr1-8b-b`, `l1-qwen3-32b-b`,
  `l1-gptoss20b-medium-b`. Kaggle keeps kernel output indefinitely, so nothing is at risk, but
  G-L1b needs the `medium` and `high` runs and they are in that list.
- **Nothing is running on either track** as of 26 Sep, on the Large-track fleet or the
  small-track one.
- The next steps are unchanged and unblocked: download, `write_results.py`, then L1-a with
  whatever reference answers exist (LF-9).

**LF-11. The panel is not complete: one cell of eight is, and the binding constraint was the token
budget, not GPU hours.** All 12 panel outputs are now downloaded (`RESULTS.md`, "L1: panel
coverage" and "L1: panel run health").

| cell | completions of 637 | originals with k=4 of 46 | truncated | mean generated tokens |
| --- | --- | --- | --- | --- |
| `gpt-oss:20b` low | **637 (100%)** | 46 | 0.000 | 1,876 |
| `gpt-oss:20b` medium | 407 (64%) | 46 | 0.105-0.529 | 10,257-17,172 |
| `deepseek-r1:8b` | 211 (33%) | 29 | 0.950-0.967 | 15,166-15,274 |
| `gpt-oss:20b` high | 173 (27%) | 9 | 0.426-0.653 | 22,318-26,093 |
| `qwen3:4b` | 144 (23%) | 0 | 0.896 | 14,966 |
| `qwen3:30b-a3b` | 105 (17%) | 0 | 0.810 | 14,498 |
| `qwen3:14b` | 82 (13%) | 0 | 0.610 | 13,100 |
| `qwen3:32b` | 73 (11%) | 0 | 0.303-0.350 | 10,530-10,820 |

Three things this says, in order of how much they change the plan:

- **A 16k context truncates thinking models.** Between 43% and 97% of replies hit `num_predict`
  before producing an answer, so the completions exist and the answers do not. Only `low` effort
  is cheap enough to finish on this hardware, at about a seventh of `high`'s token count.
- **`summary.json` overstated every run.** Its `completions_this_session` field is `len(todo)`, the
  work list length, computed before sampling starts - not what was written. LF-10 read it as
  completion. Coverage now comes from counting records, which `write_results.py` does as of today.
- **Throughput per cell varied 9-fold**: 89.6 tok/s (`gpt-oss:20b` low), 69.2 (`qwen3:4b`), 65.1
  (medium), 57.9 (high), 49.5 (`qwen3:30b-a3b`), 44.5 (`deepseek-r1:8b`), 33.7 (`qwen3:14b`), 10.3
  (`qwen3:32b` at parallel 3). A 12-hour T4x2 session buys between 0.4M and 3.9M tokens, and a
  full cell at `high` needs about 16M.

**LF-12. Robustness is a property of the problem, not of the (model, problem) pair.** Of the 33
labelled problems in `train-main-v2`, 11 carry labels for two or more models, and **all 11 agree**.
Every pairwise comparison is exactly 1.000:

| pair | shared labelled problems | agreement |
| --- | --- | --- |
| Qwen3.5-4B vs gpt-oss-120b | 8 | 1.000 |
| Qwen3.5-4B vs DeepSeek-R1-0528-Qwen3-8B | 6 | 1.000 |
| Qwen3.5-4B vs Olmo-3-7B-Think | 5 | 1.000 |
| Olmo-3-7B-Think vs DeepSeek-R1-0528-Qwen3-8B | 5 | 1.000 |
| DeepSeek-R1-0528-Qwen3-8B vs gpt-oss-120b | 5 | 1.000 |
| Olmo-3-7B-Think vs gpt-oss-120b | 4 | 1.000 |

A **per-problem oracle** - the majority label of each problem, applied to every labelled row of
every model - scores **0.986** over all 71 rows, against **0.732** for always-robust. Per cell it is
1.000 everywhere except gpt-oss-120b at `low` (0.857, one row).

Read this as an architecture decision, not as an achievable accuracy: the oracle uses the labels it
is scored against. What it establishes is that **one score per problem, broadcast to every (model,
effort), loses almost nothing** - so the Main Track's fifth model needs no model-specific machinery,
and the small track's `e27k`, which can give one problem two different labels for two models, is
structurally at odds with this while still winning that leaderboard (small-track Finding 60). Two
caveats the report must state: n = 11 problems, and the three-valued rule discards the 11 rows whose
`max_drop` lands in 0.11-0.21, which is exactly where a disagreement would be most likely to sit.

**LF-13. `reasoning_effort` almost never changes the label.** gpt-oss-120b carries 37 labelled rows
over 21 problems (7 low, 15 medium, 15 high). On problems labelled at two efforts:

| comparison | shared problems | agreement | flips |
| --- | --- | --- | --- |
| high vs medium | 10 | **1.000** | 0 |
| high vs low | 6 | 0.833 | 1 |
| low vs medium | 6 | 0.833 | 1 |

Six problems are labelled at all three efforts and five are unanimous. The single dissenter is the
angle-bisector problem (`AB = 200, AC = 225`), spurious at `low` and robust at `medium` and `high` -
which is the sensible direction, and both flips in the table involve `low`. So a shared,
effort-blind prediction costs at most one row in 22 here, and a method that conditions on effort has
to earn that back. `PLAN.md` M5 tests it at n approximately 500.

**LF-14. The L1 panel could not answer the question it was built for, because no answer key
exists.** Grading needs each problem's true answer, since the official perturbations preserve it.
The 36 `train-main-v2` problems ship no answers; neither do the 10 AIMO-2025 reference problems.
The only key in reach is the small track's E24a key - 14 problems, from unanimous Skywork samples,
which is a model's opinion rather than ground truth. Agreement with it, over original samples:

| model | effort | agreement with the E24a key | truncated |
| --- | --- | --- | --- |
| `qwen3:32b` | think | 0.960 | 0.303 |
| `gpt-oss:20b` | medium | 0.946 | 0.105 |
| `qwen3:14b` | think | 0.929 | 0.610 |
| `gpt-oss:20b` | high | 0.884 | 0.426 |
| `qwen3:30b-a3b` | think | 0.786 | 0.810 |
| `gpt-oss:20b` | low | 0.768 | 0.000 |
| `qwen3:4b` | think | 0.571 | 0.896 |
| `deepseek-r1:8b` | think | **0.173** | 0.967 |

DeepSeek's 0.173 measures truncation, not ability: 97% of its replies never reached an answer. The
same artifact inflates its `self_consistency` to 0.976, because two missing answers count as
agreeing - a trap in any stability statistic computed over possibly-empty answers. Note also that
`expert_no_solution` variants report stability 0.000 by design: there the correct behaviour is to
*not* reproduce the original answer.

**Gates G-L1a and G-L1b are retired undecided.** Proxy fidelity was never measurable, and no
conclusion about gpt-oss-120b should be drawn from these eight cells.

**LF-15. The answer-key problem is solved by changing the population, not by building a key.** The
small track already has `data/olympiad_pool/pool.parquet`: **505 olympiad problems with a single
integer answer**, from pinned revisions of four Hugging Face datasets - OlympiadBench (362), AMC
validation (83), AIME-2024 (30) and AIME-2026 (30). Ground truth ships with the problems. Labelling
those under the official rule needs no key of our own, and it is the population the private test
set is drawn from - unlike the Hendrycks MATH problems that every earlier lane had to fall back on,
and on which nothing transferred (small-track Findings 32, 39, 40). Adding `MathArena/aime_2025`
would also give the `train-main-v2` calibration problems real answers, since 2025 AIME problems
appear there verbatim (small-track Finding 49). This is the foundation of `PLAN.md` Lane A.

**LF-16. M11a is built and passes the contract check; nothing about it needs a GPU.**
`submission/make_m11a.py` writes `dist/solution-m11a-constant-gptoss-main.zip`: 3.2 KB,
`solution.py` and `tokenizer_check.py` at the root, deterministic timestamps, **no `small.txt`**.
It ships no fitted estimator - the policy is source only - so building and testing it are CPU work.

The four small models run the small track's `e27k` policy unchanged (one forward pass per problem,
the familiar half robust, Skywork and Olmo one fewer). `openai/gpt-oss-120b` is deliberately left
out of `SPLIT_MODELS` and added to `FALLBACK_ROBUST`, so it returns `True` for every problem
**without the 62 GiB checkpoint being loaded at all**. That is both the best constant in the
official data at every effort (LF-1) and the fastest answer, and runtime breaks ties.

Checked, on CPU: `run_local.py` on the ZIP in Main-Track mode reports **coverage 1.0 and 0 invalid
predictions** through the same ingestion and scoring programs Codabench runs; `submission/
test_m11a.py` exercises every branch - the gpt-oss constant at all four effort values, the split's
robust counts with and without a tilt, an even batch, an empty batch, an unlisted `model_id`, and
the fallback each model takes when its weights are absent. Its accuracy there, 0.625, is **not** a
score estimate: it is the 24-case public `val-sample`, scored by the fallback path, because no
small-track weights are cached on this machine.

What is untested locally is the forward-pass path, which needs a GPU and cached weights. It is the
same source that scores 0.86 on the small-track leaderboard, so the new surface in M11a is the
gpt-oss branch and the absent marker file - both of them constants.

**One judgement call to revisit before the freeze.** The fallback constants inside `e27k` call
Skywork and Olmo spurious, from small-track leaderboard probes on 14 cases. The official labels
lean the other way for both (Skywork 2/3 robust, Olmo 7/9), and we have no Main-Track leaderboard
reading at all. The fallback only fires on an exception, so this changes nothing unless a model
fails to load - but if a Main-Track submission ever ships constants as its primary path, the
official labels are the better evidence for this track.

## 3. Fleet state (25 Sep, ~15:10 UTC; quota resets 26 Sep 00:00 UTC)

GPU hours used of 30, from `klarge.py accounts`:

| Account | Used | Note |
| --- | --- | --- |
| MUBAMAN | 0.0 | new; still no GPU at 15:27 and 19:47 UTC despite reported verification (LF-5, LF-8) |
| Eshad | 0.0 | slots held by small-track kernels |
| Nasif | 0.2 | |
| RSD | 10.8 | new to this track; same account as MEDIUM on the teammate's machine |
| Labib | 13.0 | new to this track; same account as CHOTON on the teammate's machine |
| ISHRAQ | 16.1 | L0 probe s0 still holding a slot |
| Mashrafi | 17.5 | |
| Samin | 19.8 | |
| DIDHITI, TASHIN, SALAM | 30+ | exhausted until the reset |

## 4. Open items (superseded by `PLAN.md` §4; kept for the record)

- After the first-launch jobs end (~23:45 UTC), download them and push continuation sessions
  (11 h each) for every job that did not finish (LF-6). Scheduled task
  `aimo-large-l1-continuations` does this at 00:30 UTC on 26 Sep. Accounts running five
  `kagri-mk2-*` kernels will refuse pushes until those finish (LF-8).
- Helpers (LF-8) run until ~07:00-07:45 UTC on 26 Sep; download them with `klarge.py output <job>-b`.
- Two Ollama servers per kernel, one per GPU, would roughly double throughput for models that fit
  one T4 (qwen3-4b, DeepSeek-8B). Not built; add if continuation time runs short.
- Reference answers, needed to grade L1 (L1-a): the 10 AIMO-2025 reference problems have none in
  the repo; the E24a key covers only 14 of the 36 `train-main-v2` problems, and it is model-derived (LF-9).
- Confirm Ollama's `deepseek-r1:8b` digest is DeepSeek-R1-0528-Qwen3-8B before comparing it with
  the official `qwen3-8b:low` rows.
- AIMO3 rules acceptance on one account, if the team wants to test L4.
