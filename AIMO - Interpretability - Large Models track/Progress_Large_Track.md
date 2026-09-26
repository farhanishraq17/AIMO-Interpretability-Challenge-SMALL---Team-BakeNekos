# Large Models (Main) Track: Progress

Narrative log of what was built and measured. Findings are numbered **LF-n**, separately from
the small track's `PROGRESS.md`. Every number comes from a script in this folder; the tables
behind them are in `Results_Large_Track.md` (regenerate with `scripts/write_results.py`).

## 1. What exists (25 Sep 2026)

| Piece | What it does |
| --- | --- |
| `scripts/download_data.py` | Mirrors every public `aimo-interp` HF dataset at a pinned revision into `data/hf/`; records failures (`problems-public` still answers 401) in `data/MANIFEST.json` |
| `scripts/build_items.py` | Builds the panel list `data/panel/items.jsonl`: 36 `train-main-v2` originals + 10 reference originals (k=4) and 453 official variant texts (k=1); official per-variant outcomes go to `official_variants.jsonl` (821 rows) |
| `scripts/klarge.py` | Main-Track Kaggle launcher: H100/T4 shapes, model and kernel mounts, payload embedding, per-account tokens, quota check (works around the CLI 2.2.2 `kaggle quota` crash), `runs/launches.json` log |
| `jobs/probe.py` | L0 hardware probe |
| `jobs/ollama_panel.py` | L1 panel: Ollama on T4x2, append-per-completion `records-<payload sha>.jsonl`, sample-major order so a timeout leaves full coverage at lower k, resumes from keys earlier sessions finished |
| `scripts/write_results.py` | Regenerates `Results_Large_Track.md` |

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

## 4. Open items

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
