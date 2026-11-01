# Submissions, best first

Every ZIP this track has built, ordered by its **Codabench leaderboard score**, highest first.
Narrative and evidence live in `../PROGRESS.md` (numbered Findings); this file is the index.

**What we nominate (1 Nov): `e64-difficulty`, Codabench 1010961**, on the OpenReview form by 5 Nov AoE
(`../docs/COMPETITION.md` §12). The form takes a second ID, and the better of the two counts; that choice is open.
The first table below is the old batch protocol: since 15 Oct `e27k`'s 0.86 reads 0.64.

**How to read a score.** The hidden validation set has **14 cases**, so every score is a multiple
of 1/14 and one case is 0.07. That is a small board: a one-case difference is noise, and the whole
series below was produced deliberately, one upload at a time, to measure structure rather than to
climb. Scores are accuracy only — `coverage` was 1.0 and `invalid_predictions` 0 on every entry.

**Two generations of readings.** The small-track task was rebuilt on **16 Sep 2026, 22:35 UTC**
(Finding 35). Readings taken before that scored a *different task* and are not comparable with the
ones above them; they are kept in their own table at the end.

**The built ZIPs are in the repository**, under `dist/` — clone it and they are ready to upload, no
rebuild and no Python needed. That is deliberate: three of them embed a fitted artifact that cannot
be rebuilt from a fresh clone, because the feature dumps and label tables they were fitted from are
git-ignored and regenerable only with GPU time. The builders are here too and are deterministic, so
a rebuild reproduces them byte for byte (see "Rebuilding", below).

---

## The leaderboard series

| Score | | Submission | What it does | What it settled |
| --- | --- | --- | --- | --- |
| **0.86** | 12/14 | **`e27k-tilted`** | Familiarity ranking, plus per-model robust counts: Skywork and Olmo call one fewer problem robust than the median | **The incumbent until 15 Oct**, when per-problem evaluation took it to 0.64. The count each model's measured class balance implies (Findings 60, 61) |
| 0.79 | 11/14 | `e27m-qwen-down` | `e27k`, Qwen one fewer robust | Every single-step neighbour of `e27k` loses exactly one case, so `e27k` is a strict local optimum and all six tests agree with the probe-measured balances (Finding 61) |
| 0.79 | 11/14 | `e27n-deepseek-down` | `e27k`, DeepSeek one fewer robust | ″ |
| 0.79 | 11/14 | `e27o-qwen-up` | `e27k`, Qwen one more robust | ″ |
| 0.79 | 11/14 | `e27p-deepseek-up` | `e27k`, DeepSeek one more robust | ″ |
| 0.71 | 10/14 | `e27e-familiar-robust` | Familiarity ranking, median split, no per-model counts | **The prior-free version of the incumbent.** Assumes nothing about the class balance (Finding 49) |
| 0.71 | 10/14 | `e27h-split-deepseek` | The same split for DeepSeek alone, constants elsewhere | **The entire gain of `e27e` is DeepSeek's.** Splitting any other single model adds nothing (Finding 50) |
| 0.71 | 10/14 | `e28a-suffix-nll` | Familiarity ranked by suffix NLL instead of mean NLL | Ties `e27e`: confirmation that the familiarity *family* works, not a new best (Finding 50) |
| 0.64 | 9/14 | `e27f-constants` | Best constant per model, no ranking at all | **The constants ceiling.** Per-model constants cannot exceed 9/14 (Finding 45) |
| 0.64 | 9/14 | `e27g-split-qwen35` | Split Qwen only | At the ceiling: Qwen's split adds zero net cases |
| 0.64 | 9/14 | `e27i-split-skywork` | Split Skywork only | ″ |
| 0.64 | 9/14 | `e27j-split-olmo-think` | Split Olmo-3-Think only | ″ |
| 0.64 | 9/14 | `e11-v2-qwen-true` | Text + hidden-state fusion for DeepSeek, constants elsewhere | **The zero-GPU fallback.** The best result before any per-problem ranking existed |
| 0.57 | 8/14 | `e27a-official-text` | Char tf-idf fitted on the 33 labelled official problems, applied to every model | **Models fitted on the official labels carry nothing to the hidden set.** n = 33 is too small, and the populations differ (Finding 49) |
| 0.57 | 8/14 | `e27b-official-text-e11` | The same, but E11's fusion kept for DeepSeek | ″ |
| 0.57 | 8/14 | `e27c-official-length` | Problem length alone, fitted on official labels | ″ — and length was the one signal near significance offline (Finding 46) |
| 0.57 | 8/14 | `e29a-deepseek-min-k-gap` | Min-K gap for DeepSeek only, constants elsewhere | Rejects the tail-gap family even where E27 found its gain, despite Min-K gap passing a stricter *offline* stability gate (AUC 0.808) than mean NLL ever did (Finding 50) |
| 0.50 | 7/14 | `probe-all-false` | Constant `False` everywhere | **7 of the 14 cases are spurious.** The board is balanced, unlike our training data at 74% spurious |
| 0.29 | 4/14 | `e27d-familiarity` | Familiarity ranking, **unfamiliar** half robust | Far below chance, which is informative: the ranking is real and simply inverted. Shipping its opposite (`e27e`) is what opened the whole series (Finding 49) |
| 0.29 | 4/14 | `e28d-min-k-gap` | Min-K gap for every model | The exact ZIP was retried once after an ingestion-host failure and scored the same, so this is a genuine transfer failure, not a packaging one (Finding 50) |

### Per-problem evaluation (since 15 Oct)

| Score | | Submission | What it does | What it settled |
| --- | --- | --- | --- | --- |
| **0.71** | **10/14** | **`e64-difficulty`** ⭐ (1010961, 26 Oct) | Difficulty read from one forward pass (per-layer ridge from prompt states to Qwen's log trace length on 497 unlabelled olympiad problems); easier = robust, cutoffs at the `p_m` quantiles of the 36 official problems | **The first per-problem method above the constants**, by one case. Its pre-registered ship rule (10/14 or better) makes it the submission we nominate (Finding 88), although it reverses on DeepSeek MATH (Finding 89) |
| 0.64 | 9/14 | `e63-official-quantile` (1010959, 26 Oct) | E61's familiarity cutoffs recalibrated on the 36 `train-main-v2` problems, Qwen at its constant | **Equals the constants**, below its expected 10.3. Per-problem familiarity has now read 8, 9 and 9 (Finding 87) |
| 0.64 | 9/14 | `e61b-quantile` (1007215, 21 Oct) | Robust iff mean NLL is at or below the `p_m` quantile of E55's 548 unlabelled problems (Qwen 0.6, DeepSeek 2/3, Skywork and Olmo 1/3) | **Equals the constants.** The batch ranking's 12/14 does not survive an absolute threshold (Finding 86) |
| 0.64 | 9/14 | `e27k-tilted` (990659, re-scored 15 Oct) | The batch ranking; with one problem a call it is the constants | Finding 79 |
| 0.57 | 8/14 | `e61a-amc-median` (1007214, 21 Oct) | Control: robust iff mean NLL is below E54's frozen 75-problem AMC median | One case below the constants |

**Uploaded 26 Oct, 21:13 and 21:14 UTC, E63 first so the board ends on E64.** Two earlier attempts the same evening (1010405, 1010444) never received a score and are not reads. The submission we nominate is `e64-difficulty`, under the rule fixed in its protocol before either read: it replaces `e27f` at 10/14 or better, and E63 did not reach 10. The Codabench run itself, on the organizers' GPU runner, had coverage 1.0 and 0 invalid predictions, and its read differs from the constants', so the model path ran there rather than the fallback: that covers what the Docker test checks, and the owner waived it on 27 Oct, as for E61 (`protocols/E64_PROTOCOL.md` amendment 4). **There is no Google Form (checked 31 Oct).** The OpenReview report nominates **Codabench submission 1010961**
(`dist/solution-e64-difficulty-small.zip`, SHA-256 `665f86ddd28730c2…`) in `leaderboard_submission_ids`, by 5 Nov
AoE; see `docs/COMPETITION.md` §12.

**E66 (27 Oct) built no ZIP for upload.** E64 refitted on a 64k-cap target failed its official-row gates (AUC 0.808,
26/34, against E64's 0.883 and 28/34; Finding 90), so it never reached the board and E64 stays the submission we nominate.
`make_e64.py --experiment e66` can rebuild it; the E64 ZIP rebuilds byte for byte after the builder change.

Both E61 ZIPs were uploaded without the Docker test (owner's decision, `protocols/E61_PROTOCOL.md` amendment 1).
The protocol's ship rule (§5: E61b ships at "at least 9/14") and its reading (§6: "9/14 or below" means the threshold
does not transfer and `e27f` ships) disagree at exactly 9/14; the owner decides (Finding 86).

### Built, never uploaded

| Submission | What it does | Why it sat |
| --- | --- | --- |
| `e28b-mean-suffix` | Familiarity ranked by mean + suffix NLL | Pre-registered E28 candidates. After `e28a` tied rather than beat `e27e` and `e28d` scored 4/14, the family was rejected and the remaining two were not worth an upload on a 14-case board |
| `e28c-min-k-20` | Familiarity ranked by Min-K% at k = 20 | ″ |

### Before the task rebuild (16 Sep 22:35 UTC) — not comparable with the above

| Score | Submission | Note |
| --- | --- | --- |
| 0.53 | `probe-true-qwen35-4b` | `True` for Qwen only. The one probe that beat all-False, which is why E11 v2 defaults Qwen to `True` |
| 0.47 | `probe-all-false` | Same ZIP as the 0.50 row above, scored against the old task |
| 0.47 | `probe-true-deepseek` | |
| 0.47 | `probe-true-skywork-7b` | |
| 0.47 | `probe-true-olmo3-think` | |
| 0.47 | `e11-text-stagea` | E11 v1, before Qwen defaulted to `True` |

**The four probe readings here were redone on 30 Sep** and are superseded by the rows above; only
E11 v1's 0.47 is still a pre-rebuild reading with no current equivalent.

---

## The re-probe (built 30 Sep, ready to upload)

**Since 15 Oct the per-model counts it protected no longer act** (Finding 79). No upload is planned; this is the record.

`python submission/make_probes.py` rebuilds five constant ZIPs into `dist/probes/`. They exist to
re-measure each model's class balance `d = robust - spurious` on the **current** task, because
Finding 63 shows `e27k`'s tilts are forced by those balances — and the balances were last measured
on **18 September**. The hidden task was silently rebuilt once before (Finding 35), and if it were
rebuilt again the tilts could be exactly wrong.

**Upload `all-false` first**: every other probe is read relative to it.

**Result, uploaded 30 Sep: all five landed exactly on the expected values** (submissions 952264-952269, Finding 64). The balances are unchanged twelve days on, the set is still 14 cases at 7 robust, and `e27k`'s counts are confirmed correct — the freeze goes to `e27k`.

| Order | ZIP | Expected | **Scored** |
| --- | --- | --- | --- |
| 1 | `solution-probe-all-false-small.zip` | **0.50** (7/14) | **0.50** |
| 2 | `solution-probe-true-qwen35-4b-small.zip` | 0.57 (8/14) | 0.57 |
| 3 | `solution-probe-true-deepseek-small.zip` | 0.57 (8/14) | 0.57 |
| 4 | `solution-probe-true-skywork-7b-small.zip` | 0.43 (6/14) | 0.43 |
| 5 | `solution-probe-true-olmo3-think-small.zip` | 0.43 (6/14) | 0.43 |

Five of the ten submissions allowed per day, and about 270 remain of 300.

**Reading the result.** With all-False scoring `a` cases out of 14, turning one model to `True`
gains its robust cases and loses its spurious ones, so a probe scoring `m` gives that model's
balance directly:

```
d_model = m - a        (in cases, not accuracy: multiply each score by 14)
```

- **All five match the table** → the balances hold, `e27k`'s counts are still exactly right, and
  the freeze goes to `e27k` on Finding 63.
- **All-False is not 7/14** → the case count or the class split changed; every arithmetic result
  from Finding 45 onward has to be re-derived before the freeze.
- **A balance flipped sign** → the tilt for that model is now wrong, and `e27k` would be calling
  the wrong number robust. Rebuild with the corrected `TILT` in `make_e27.py` and re-upload.

The probes are constants, so they need no GPU and cannot fall back or time out. Each is under
400 bytes.

## Rebuilding

```bash
python submission/make_e27.py     # the 13 e27* ZIPs, including the incumbent
python submission/make_e28.py     # e28a-d and e29a
python submission/make_probes.py  # the 5 constant probes -> dist/probes/
```

Each is deterministic: fixed ZIP timestamps, sorted entries, and every fitted estimator built in
the pinned venv (`getting-started/.venv`) so a rebuild is byte-identical. **Only fitted estimators
ship, never the label table** — the rules forbid submitting training data or private labels.

Before any upload: `unzip -l` must show `solution.py` **and `small.txt`** at the archive root, and
`getting-started/scripts/run_local.py <zip> --small` must report coverage 1.0 with 0 invalid
predictions.

## What the series is worth reading for

Read top to bottom and the shape of the problem appears, which is the part the report needs:

1. **Constants cap at 9/14** (`e27f`), so anything above that is per-problem information.
2. **The familiarity ranking is real but was shipped upside down** — `e27d` at 4/14 and `e27e` at
   10/14 are the same ranking, opposite directions.
3. **The gain is one model's** — `e27h` alone reaches 10/14, `e27g`/`e27i`/`e27j` reach none of it.
4. **Per-model counts add two more cases** (`e27k`), and all four neighbours lose exactly one, so
   the counts are at their optimum rather than merely lucky.
5. **Everything fitted offline failed to transfer** — `e27a`/`b`/`c` at 8/14, `e28d` at 4/14,
   `e29a` at 8/14, the last two despite passing stricter offline gates than the winner ever did.

Point 5 is the uncomfortable one and it is the honest headline: on this task, **offline validation
did not predict leaderboard behaviour**. `../PLAN.md` §12.3 records the unresolved contradiction
behind it, which the 27 Oct freeze has to decide.

## Not in this track

The Main Track's first ZIP is `submission/make_m11a.py` in its own repository,
`https://github.com/farhanishraq17/AIMO-Interpretability-Challenge-LARGE---Team-BakeNekos`, → `dist/solution-m11a-constant-gptoss-main.zip`. It ships `e27k`'s policy for these four models plus
a constant for `gpt-oss-120b`, and it must **not** contain `small.txt`.

E37 produced no ZIP: it passed its GSM-Plus confirmation and then failed its frozen runtime gate,
so it was never packaged or uploaded.
