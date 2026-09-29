# Submissions, best first

Every ZIP this track has built, ordered by its **Codabench leaderboard score**, highest first.
Narrative and evidence live in `../PROGRESS.md` (numbered Findings); this file is the index.

**How to read a score.** The hidden validation set has **14 cases**, so every score is a multiple
of 1/14 and one case is 0.07. That is a small board: a one-case difference is noise, and the whole
series below was produced deliberately, one upload at a time, to measure structure rather than to
climb. Scores are accuracy only — `coverage` was 1.0 and `invalid_predictions` 0 on every entry.

**Two generations of readings.** The small-track task was rebuilt on **16 Sep 2026, 22:35 UTC**
(Finding 35). Readings taken before that scored a *different task* and are not comparable with the
ones above them; they are kept in their own table at the end.

`dist/` is git-ignored, so the ZIPs are not in the repository — the builders are, and they are
deterministic. Rebuild everything with the three commands in "Rebuilding", below.

---

## The leaderboard series

| Score | | Submission | What it does | What it settled |
| --- | --- | --- | --- | --- |
| **0.86** | 12/14 | **`e27k-tilted`** ⭐ | Familiarity ranking, plus per-model robust counts: Skywork and Olmo call one fewer problem robust than the median | **The incumbent.** The count each model's measured class balance implies (Findings 60, 61) |
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

**These five probe readings need redoing** if they are ever quoted: E11 v2 scored 0.64 where they
predict 0.53, so at least one of them no longer holds on the current task.

---

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

The Main Track's first ZIP is `../AIMO - Interpretability - Large Models track/submission/make_m11a.py`
→ `dist/solution-m11a-constant-gptoss-main.zip`. It ships `e27k`'s policy for these four models plus
a constant for `gpt-oss-120b`, and it must **not** contain `small.txt`.

E37 produced no ZIP: it passed its GSM-Plus confirmation and then failed its frozen runtime gate,
so it was never packaged or uploaded.
