# Experiment log — Main Track

One line per experiment, then the detail for each finished one. Numbers come from `RESULTS.md`
(regenerate with `scripts/write_results.py`); interpretation lives in `PROGRESS.md`.

Last updated: 27 September 2026. **Nothing is running on either track**
(`scripts/klarge.py status` and the small track's `kaggle_launch.py status` both report zero
RUNNING or QUEUED kernels).

---

## 1. Summary

### Done: the T4-fleet era (25–26 Sep)

| ID | Question | Compute | Status | Outcome |
| --- | --- | --- | --- | --- |
| L0 | What does Kaggle's `NvidiaH100` shape actually allocate, and do the gpt-oss-120b weights mount? | 4 short kernels | Done | **T4x2 on 4 of 4** accounts, internet on, weights mount. gpt-oss-120b cannot run on the fleet (LF-3) |
| L1 smoke | Does the Ollama panel work end to end at all three efforts? | 2 kernels, 0.5 h | Done | Yes, after a warm-load fix. Also measured 90–97 tok/s on T4x2 (LF-4, LF-6) |
| L1 | Proxy panel: sample 46 originals (k=4) and 453 official variants (k=1) with everything that fits T4x2 | 12 kernels, ~120 GPU-h | **Done and read** | **One cell of eight is complete.** `gpt-oss:20b` at `low` finished 637/637; the rest are 11–64%, truncated 43–97% at a 16k context (LF-11) |
| L1-a | Grade the panel against ground truth | CPU | **Abandoned** | The 10 reference problems have no answers in the repo, and the only key available is model-derived. The panel answered nothing it was built to answer (LF-14) |
| L2 | Label-free gpt-oss signal at test time | — | Superseded | Folded into the M-programme (M7, M8) |
| L3 | First Main-Track ZIP | CPU | **Renamed M11a**, still not started | Nothing blocks it |
| L4 | Direct gpt-oss-120b runs | H100 | Superseded | M0 does this on provisioned hardware instead of waiting on Kaggle |

### Done: zero-GPU analysis of the official labels (27 Sep)

| ID | Question | Compute | Status | Outcome |
| --- | --- | --- | --- | --- |
| L5 | Is a label a property of the (model, problem) pair, or of the problem? | CPU, seconds | Done | **Of the problem.** All 11 problems labelled for ≥2 models agree, gpt-oss-120b included; a per-problem oracle scores 0.986 against 0.732 for always-robust (LF-12) |
| L6 | Does `reasoning_effort` change the label? | CPU | Done | Almost never: 2 flips in 22 comparisons, both at `low`; high vs medium agree 10/10 (LF-13) |

### Planned: the M programme (`PLAN.md` §4)

| ID | Question | Compute | Status | Gate |
| --- | --- | --- | --- | --- |
| M0 | vLLM on real GPUs, 32–65k context, gpt-oss-120b directly | GPU | **Next** | G-M0 |
| M1 | Generate the five official perturbation families with verified answer preservation | GPU + 30 min of human reading | Planned | G-M1 |
| M2 | Label ~500 pooled olympiad problems under the official rule, 7 (model, effort) cells | ~250–400 GPU-h | Planned | — |
| M3 | **Do our labels agree with the 71 official ones?** | GPU | Planned | **G-M3**, pre-registered |
| M4 | Problem-level predictor at n≈500 | CPU | Planned | G-M4 |
| M5 | Is conditioning on effort worth anything? | CPU | Planned | — |
| M6 | White-box on gpt-oss-120b: layer probes and **MoE router statistics** | GPU | Planned | G-M6 |
| M7 | Perturbation sensitivity of internals, with labels this time | GPU | Planned | G-M7 |
| M8 | Sparse (SAE) features on the prompt representation | GPU | Planned | G-M8 |
| M9 | Does a signal fitted on 4B predict for 120B? | GPU | Planned | G-M9 |
| M10 | Test-time budget audit on one 96 GB card | GPU | Planned | G-M10 |
| M11a | Constants ZIP: small-track policy + `True` for gpt-oss-120b | CPU | **Ready to build** | G-L3 |
| M11b | The problem-level model as a ZIP | CPU | After G-M4 | G-L3, G-M10 |
| M12 | Technical report | — | 15 Nov | — |

---

## 2. Running now

**Nothing, on either track.** Checked 27 September: every one of the 18 launched kernels reports
`COMPLETE`, and all 12 panel outputs are downloaded.

**No Main-Track submission exists yet**, so there is no leaderboard reading for this track. The
small track's board, for reference, tops out at `e27k` = 12/14 = 0.86.

---

## 3. Completed experiments

### L0: what `NvidiaH100` allocates (ISHRAQ, Mashrafi, Samin, Nasif, 25 Sep, ~0.1 h each)

Four probe kernels requesting the `NvidiaH100` accelerator. All four got **2× Tesla T4** with
internet enabled and the `danielhanchen/gpt-oss-120b` weights mounted read-only. Pushes that
attached the AIMO3 competition were refused: no fleet account has accepted the AIMO3 rules, and
whether accepting them unlocks an H100 is still untested.

Consequence: gpt-oss-120b (62 GiB in MXFP4) cannot be loaded on the fleet, so L1 used
`gpt-oss:20b` as a proxy — a decision that M0 now reverses.

### L1 smoke: does the panel run? (RSD, 25 Sep, 0.5 h)

The first L1 launch died about two minutes after model load on every kernel: the warm-load
`/api/generate` call omitted `"stream": false`, so Ollama returned NDJSON and `json.loads` raised
"Extra data". One field. Two short smoke kernels then proved the whole path, including effort
levels, before 8-hour sessions were spent again.

The smokes also produced the throughput number that should have changed the plan on the spot:
**90.4 tok/s (gpt-oss:20b) and 96.9 tok/s (qwen3:4b) aggregate over 8 parallel streams**, i.e. about
12 tok/s per stream. `qwen3:4b` hit its 3,072-token cap on every item while still thinking.

### L1: the proxy panel (8 base jobs + 4 reverse helpers, 25–26 Sep, ~120 GPU-h)

Eight (model, effort) cells: `gpt-oss:20b` at low/medium/high, `deepseek-r1:8b`, and Qwen3 at
4B/14B/30B-A3B/32B. Four "`-b`" helpers walked the same work list backwards so an idle account
could meet the forward session instead of duplicating it.

Every kernel reported `COMPLETE`. **Coverage did not follow** (`RESULTS.md`, "L1: panel coverage"):

| cell | completions of 637 | originals with k=4 | truncated |
| --- | --- | --- | --- |
| gpt-oss:20b low | **637 (100%)** | 46 of 46 | 0% |
| gpt-oss:20b medium | 407 (64%) | 46 | 11–53% |
| deepseek-r1:8b | 211 (33%) | 29 | 95–97% |
| gpt-oss:20b high | 173 (27%) | 9 | 43–65% |
| qwen3:4b | 144 (23%) | 0 | 90% |
| qwen3:30b-a3b | 105 (17%) | 0 | 81% |
| qwen3:14b | 82 (13%) | 0 | 61% |
| qwen3:32b | 73 (11%) | 0 | 30% |

Only `low` effort is cheap enough to finish on this hardware: 1,876 generated tokens per reply
against 17,172 at medium and 26,093 at high. See LF-11 for why `summary.json` read as complete.

### L1-a: grading, abandoned (27 Sep)

The panel was built so that per-variant correctness could be compared with the official
`gpt-oss-120b:low` outcomes (gate G-L1a). Grading needs the *true* answer to each problem, because
the official perturbations preserve it. The 36 `train-main-v2` problems ship no answers, and the
10 AIMO-2025 reference problems ship none either. The only key in reach is the small track's E24a
key — 14 problems, derived from unanimous Skywork samples — which is a model's opinion, not ground
truth.

Against that key, original-sample agreement was: qwen3:32b 0.960, gpt-oss medium 0.946, qwen3:14b
0.929, gpt-oss high 0.884, qwen3:30b-a3b 0.786, gpt-oss low 0.768, qwen3:4b 0.571 and
`deepseek-r1:8b` **0.173** — the last one because 97% of its replies were truncated before an
answer appeared, not because it is wrong. The same truncation inflates its self-consistency to
0.976, since two missing answers count as agreeing.

G-L1a and G-L1b are therefore **undecided and retired**. The replacement is the M programme, which
labels problems that come with published answers (`PLAN.md` M2).

### L5 / L6: what the official labels say (local, 27 Sep, seconds)

Zero-GPU analysis of all 82 rows of `train-main-v2`, the highest-value-per-second work in either
track so far. Full numbers in LF-12 and LF-13. The two results:

- **Cross-model:** 11 problems are labelled for two or more models; all 11 agree. Every pairwise
  comparison is 1.000, including `gpt-oss-120b` against each of the four small models (n = 4 to 8).
  A per-problem majority label, scored on every labelled row, reaches **0.986** against the 0.732
  always-robust floor.
- **Cross-effort:** `high` vs `medium` agree on 10 of 10 shared problems; `high` vs `low` and
  `low` vs `medium` each flip once out of 6. Of the 6 problems labelled at all three efforts, 5 are
  unanimous. Both flips involve `low`.

---

## 4. Failures worth remembering

| What went wrong | Where | The fix |
| --- | --- | --- |
| A missing `"stream": false` killed 7 kernels two minutes after model load | L1, first launch | One field, plus a smoke test before every long launch (LF-4) |
| `NvidiaH100` silently gives T4x2 | L0 | Probe the hardware before planning around it (LF-3) |
| A 16k context truncated 43–97% of thinking-model replies | L1 | 32k minimum, 65k at `high` (LF-11) |
| `summary.json` said 637 completions when 82 were written | L1 | Count records, not the work list; `write_results.py` now prints coverage (LF-11) |
| A complete pipeline that could answer nothing, because no answer key existed | L1-a | Choose a population that ships ground-truth answers (LF-14, LF-15) |
| Kernels accepted onto accounts with no GPU | L1 | Exit early on missing `nvidia-smi`; phone-verify the account (LF-5) |
| Pushes refused on accounts shared with another project | L1 | Check `klarge.py accounts` first; five `kagri-*` kernels block an account (LF-8) |

---

## 5. Where this leaves the approach

The T4-fleet era answered one question — *can a quantised proxy on undersized hardware stand in for
gpt-oss-120b?* — with "we never found out, and it cost 120 GPU-hours". The zero-GPU analysis that
followed answered two much better questions in seconds.

The lesson is in the ordering, and `PLAN.md` §4 is built on it: **read the labels before sampling
anything.** With compute now unlimited, the programme spends it on the one thing that has blocked
every lane in both tracks — the 33 labelled problems — and keeps the shipped method inside one
forward pass per pair, because the 3600-second evaluation budget has not changed.
