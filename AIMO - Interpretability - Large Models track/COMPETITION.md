# AIMO Interpretability Challenge — Main Track Brief

**This folder's entry:** the **Main Track** (no model-size cap, five evaluated models including
`openai/gpt-oss-120b`).

The small track's `../AIMO - Interpretability - Small Models track/COMPETITION.md` is the long-form
brief: official links, organizer contacts, key dates, the data inventory, the allowed runtime
libraries, the starter-kit walkthrough. All of it applies here. **This file records only what is
different for the Main Track**, so the two never drift into contradicting each other.

---

## 1. The task, unchanged

```python
are_robust(model_id: str, problems: list[str], reasoning_effort: str = "default") -> list[bool]
```

For each (model, problem) pair, predict **robust** (the answer survives answer-preserving
perturbations) or **spurious** (it depends on a brittle pattern). Not whether the answer is correct.

Hard constraints, identical in both tracks:

- One prediction per problem, same order, native Python `bool` (NumPy bools and ints are invalid).
- Offline: no network, no `pip install`, weights only from the worker cache with
  `local_files_only=True`.
- 3600 s for the whole prediction run, all models and all problems together.
- An unknown `model_id` must still return a valid `list[bool]`, not raise.

**The third argument is where the tracks differ.** `reasoning_effort` is merged upstream (PR #9,
commit `de79405`); the harness detects the parameter by name, and the two-argument form still
works. Documented values are `"default"`, `"low"`, `"medium"`; `train-main-v2` also carries
`"high"`, so do not treat the documented set as exhaustive. In the small track the value is always
`"default"`. **Here it varies, and only for `gpt-oss-120b`.**

---

## 2. The five evaluated models

| `model_id` (passed verbatim) | Params | Efforts seen | Official labelled rows |
| --- | --- | --- | --- |
| `Qwen/Qwen3.5-4B` | 4B | `default` | 12 |
| `Skywork/Skywork-OR1-Math-7B` | 7B | `default` | 3 |
| `allenai/Olmo-3-7B-Think` | 7B | `default` | 9 |
| `deepseek-ai/DeepSeek-R1-0528-Qwen3-8B` | 8B | `default` | 10 |
| **`openai/gpt-oss-120b`** | 117B MoE | `low`, `medium`, `high` | **37** (7 / 15 / 15) |

- The first four are the small track's checkpoints, so **the small-track solution covers them
  unchanged**. Only the fifth model is new work.
- No aliases are passed. The `:low`-suffixed names (`gpt-oss-120b:low`, `qwen3-8b:low`, …) exist
  only in the public tables.
- `gpt-oss-120b` is cached in **MXFP4, about 62 GiB**, and fits the 96 GB evaluation card. White-box
  and short-sampling methods on it are therefore possible at score time — but see §5.

---

## 3. Prizes and ranking

| Place | Main Track | Small Models Track |
| --- | --- | --- |
| 1st | **$3,000** | $2,000 |
| 2nd | **$1,000** | $600 |
| 3rd | **$600** | $300 |

Plus a shared **$5,000** writeup pool, independent of leaderboard rank, which the organizers say
rewards generalization, actionable interpretability, **negative results** and efficient methods.

Ranking: accuracy on the private test set first, ties broken by average runtime over three repeated
trials. The two tracks are ranked separately, and the same method may be uploaded to both as two
ZIPs. Require `coverage = 1.0` and `invalid_predictions = 0` before believing any accuracy.

**Submission budget** (read from the Codabench API, Competition phase): **10 per day, 300 in
total**. Spending them to probe the hidden set is a real option, and the small track's entire
leaderboard gain came from doing so deliberately — but it tunes on ~14 cases and overfits fast.

---

## 4. What to submit — and the one file that must not be there

```text
solution-our-method.zip
├── solution.py          # at the archive root
├── ...helper modules...
└── artifacts/           # probes, sklearn joblib, SAE weights
```

**There is no `small.txt`.** That marker selects the small track; a Main-Track ZIP containing it
scores **0.00**, which cost the small track four uploads (its Finding 21). Verify with
`unzip -l` before every upload.

Do not submit: training datasets or private labels, cached model weights, virtualenvs,
`__pycache__`, credentials, or a nested folder wrapping `solution.py`.

Upload: sign in to [Codabench 16180](https://www.codabench.org/competitions/16180/), open the
Competition phase, and select **exactly one task** — the Main Track one, i.e. *not* the task whose
name ends "Small Model Track" (task 36241). **Record the Main-Track task id on the first upload**;
the small track's task was silently rebuilt once (its Finding 35), and readings across a rebuild are
not comparable.

The uploads are the user's to make. Nobody signs in on their behalf.

---

## 5. The evaluation container

| Resource | Spec |
| --- | --- |
| GPU | **1×** RTX PRO 6000 (Blackwell, sm_120), 96 GB |
| Time | 3600 s prediction + 3600 s scoring, separately |
| Network | Disabled (`HF_HUB_OFFLINE=1`, `TRANSFORMERS_OFFLINE=1`) |
| Cache | Read-only HF cache at `/app/data/huggingface` |
| Install | None at evaluation time |

**The Main Track's budget is tighter than the small track's in practice**, because 3600 s now also
has to cover loading a 62 GiB checkpoint and scoring a 117B model. Measure the load time before
designing around it (`PLAN.md` M10). Runtime libraries are pinned in the upstream
`Dockerfile.competition` — torch 2.12.1/CUDA 13.2, transformers 5.13.0, scikit-learn 1.8.0,
pandas 3.0.3 — and any sklearn artifact must be serialised with **those** versions.

---

## 6. Labels and data for this track

`aimo-interp/train-main-v2` is the only officially labelled split covering all five models:
**82 rows, 71 labelled, 36 problems**, schema
`model_id / reasoning_effort / problem / is_robust / max_drop`.

The label rule is three-valued on `max_drop`, the largest accuracy drop across perturbation
families — verified with zero violations on all 82 rows:

| `max_drop` | `is_robust` |
| --- | --- |
| ≤ 0.10 | `true` (52 rows, observed min −0.09) |
| 0.11 – 0.21 | `null` — too close to call (11 rows), never scored |
| ≥ 0.25 | `false` (19 rows, observed max 1.00) |

This rule must never be mixed with the older `absolute_accuracy_decay` rule used by the reference
tables (small track, Finding 44).

Two further sources matter here:

- **`aimo-interp-challenge-sample-full`** — 10 AIMO-2025 reference problems for 8 systems, and,
  in `permutations_causing_decay`, **the official perturbed texts**: 453 distinct variants across
  `rephrase`, `typos`, `domain`, `distract`, `rename`, `expert_*`, `paraphrase`, with per-variant
  accuracies. Among the 8 systems, `gpt-oss-120b:low` has the largest mean absolute decay (0.147)
  and `qwen3-8b:low` the smallest (0.006). **These texts are style exemplars and calibration data
  only — never training data.**
- **`aimo-interp/problems-public`** — private; the organizers confirmed on 26 Sep 2026 that it will
  not be shared. Do not plan around it.

**The data rule, as it stands (26 Sep 2026).** Extra labelled training data is allowed; generating
additional training data from `train-main-v2` is explicitly permitted; evaluation-data leakage
remains forbidden. The official labels are final apart from a couple of Skywork rows.

---

## 7. Perturbation families

The seven the organizers use, and what each one is for:

| Family | What it does | Answer preserved? |
| --- | --- | --- |
| `rephrase` | Meaning-preserving rewrite | Yes |
| `rename` | Rename variables and entities | Yes |
| `domain` | Same math, different surface domain | Yes |
| `distract` | Adds irrelevant but plausible text | Yes |
| `typos` | Harmless noise | Yes |
| `expert_perturbations` | Expert adversarial edits, answer-preserving | Yes |
| `expert_no_solution` | Premises removed — the problem becomes unanswerable | **No**: the correct behaviour is to refuse |

`expert_no_solution` is the one family where "keeps the same answer" is the *wrong* outcome, and a
stability statistic that ignores this reads backwards on it (see `RESULTS.md`, where it shows
stability 0.000 for exactly that reason).

---

## 8. Track-specific working notes

- **Do the small models' work once.** The four small checkpoints are shared; reuse the small
  track's shipped policy rather than re-deriving it.
- **One prediction per problem is likely enough.** Every problem labelled for two or more models
  carries the same label (`PROGRESS.md`, LF-12), and effort almost never changes it (LF-13).
- **gpt-oss-120b at `low` is cheap** — 1,876 generated tokens per reply measured, against 17k–26k
  at medium/high (LF-11). If any sampling happens at score time, it happens at `low`.
- **Keep a constant fallback inside `solution.py`.** Any exception, unknown `model_id`, or guard
  trip returns the best constant (`True`) rather than crashing the run.
- **Report obligations are the same:** a technical report and open-sourced implementation by
  15 Nov 2026 for award consideration, reproducible from the report alone, plus peer review of up
  to three other reports.
