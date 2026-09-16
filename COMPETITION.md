# AIMO Interpretability Challenge — Competition Brief

**Team focus:** Small Models Track, open-weight models (< 10B parameters).

This file is the working brief for the NeurIPS 2026 competition. Official sources are the [competition website](https://aimo-interp.github.io/), the [getting-started repository](https://github.com/aimo-interp/getting-started), and the [proposal paper](https://arxiv.org/abs/2607.13899). Submissions are scored on **Codabench**, not Kaggle. Kaggle credentials in `Secrets.env` are for AIMO 3 / open-weight model assets.

---

## 1. What the competition is

**AIMO Interpretability Challenge** (NeurIPS 2026 Competition Track), part of the Fields Model Initiative.

The task is *not* to solve olympiad math problems. The task is to decide, for one **problem–model pair**, whether that model answers the problem **robustly** or **spuriously**.

| Label | Meaning |
| --- | --- |
| **Robust** | The model's answer holds up under perturbations that leave the deductive chain unchanged. |
| **Spurious** | Counterfactual evaluation shows the answer relies on at least one brittle / shortcut pattern. |

Strong final-answer accuracy is not enough. AIMO 3 systems can score well while failing under rephrase, rename, domain transfer, distractors, typos, or expert edits.

**One-line research question:** can interpretability (or a strong behavioural method inside the compute budget) tell whether a frontier reasoning model is actually generalizing?

---

## 2. Team decision (this workspace)

| Choice | Value |
| --- | --- |
| Track | **Small Models Track** |
| Model class | **Open-weight** models below 10B parameters |
| Why this track | Compute-heavy white-box methods (SAEs, transcoders, probes on internals) are feasible at 7B–8B; the Main Track includes 27B and 120B models |
| Submission platform | [Codabench competition 16180](https://www.codabench.org/competitions/16180/) |
| Marker file | Root-level `small.txt` in the ZIP (contents ignored; name is case-insensitive) |

The same method may also be submitted to the Main Track as a *separate* ZIP without `small.txt`. Each upload counts only for the chosen track.

---

## 3. Official links

| Resource | URL |
| --- | --- |
| Competition website | https://aimo-interp.github.io/ |
| Starter kit / submission interface | https://github.com/aimo-interp/getting-started |
| Reference baselines | https://github.com/aimo-interp/baselines |
| Org GitHub | https://github.com/aimo-interp |
| Codabench (submit + leaderboard) | https://www.codabench.org/competitions/16180/ |
| Proposal paper (arXiv) | https://arxiv.org/abs/2607.13899 |
| Proposal PDF | https://arxiv.org/pdf/2607.13899 |
| Discord | https://discord.gg/PaAMFmpsBQ |
| Email updates (registration form) | https://forms.gle/EmvfJz2uSEQpbttB9 |
| Compute-support proposal form | https://forms.gle/oYkYDN1EtSVeSfMd9 |
| Organizer email | aimo.interp@gmail.com |
| Fields Model Initiative (compute) | https://www.fieldsmodel.org/ |
| Public validation labels | https://huggingface.co/datasets/aimo-interp/val-sample |
| Public problems + symbolic chains | https://huggingface.co/datasets/aimo-interp/problems-public |
| Augmented training sample (Qwen3-8B) | https://huggingface.co/datasets/aimo-interp/augmented-sample-math-agg |
| AIMO Prize / Kaggle org (upstream models) | https://www.kaggle.com/organizations/aimoprize |
| AIMO 3 (Kaggle, open-weight solvers) | https://www.kaggle.com/competitions/ai-mathematical-olympiad-progress-prize-3 |

Local copies of the first two URLs are in `Competiton-Website-Portal.txt` and `Github-Link.txt`.

---

## 4. Task contract (what we must implement)

Every submission exposes:

```python
are_robust(model_id: str, problems: list[str]) -> list[bool]
```

Hard constraints:

- Exactly one prediction per input problem, same order.
- Native Python `bool` only (`True` = robust, `False` = not robust). Integers and NumPy bools are invalid.
- Evaluation is **offline**. No internet, no `pip install`, no Hugging Face downloads at score time.
- The harness passes only `model_id` and problem strings. No case IDs, no labels, no permutation-type metadata at inference.
- Load weights only from the worker cache, with `local_files_only=True`.

Codabench aliases must be resolved to cached checkpoints. The starter kit already maps:

| Codabench alias | Hugging Face checkpoint | Track relevance |
| --- | --- | --- |
| `qwen3-8b:low` | `deepseek-ai/DeepSeek-R1-0528-Qwen3-8B` | **Small / open-weight (primary)** |

Unknown aliases should still return a valid `list[bool]` (the uncertainty baseline returns all `False` rather than crashing).

---

## 5. Tracks, prizes, ranking

Total prize pool: **$12,500**.

### Main Track

Full scale of top AIMO 3 models. No size cap.

| Place | Prize |
| --- | --- |
| 1st | $3,000 |
| 2nd | $1,000 |
| 3rd | $600 |

### Small Models Track (our track)

Evaluation subset: **best-performing models below 10B parameters**. Intended for SAE / transcoder / other heavy white-box methods.

| Place | Prize |
| --- | --- |
| 1st | $2,000 |
| 2nd | $600 |
| 3rd | $300 |

### Writeup prizes

**$5,000** for selected technical reports, independent of leaderboard rank. Organizers call out:

- generalization of interpretability methods
- actionable interpretability
- negative results
- efficient methods

### Ranking

1. Primary metric: **accuracy** on the private test set (correct robust / spurious classification).
2. Ties broken by **efficiency**: faster average runtime over three repeated trials.
3. Leaderboard also reports `coverage` and `invalid_predictions`. Require `coverage = 1.0` and `invalid_predictions = 0` before trusting accuracy.
4. Main and Small tracks are ranked separately.

Reported analyses (organizers' final report, not the live metric): public vs private, per perturbation type, per model, published vs unpublished problems, with bootstrap CIs and paired tests.

### How results are measured (verified 16 Sep 2026 from the Codabench API and the website)

There are three levels, and only the last decides prizes:

| Level | What it scores | What it is good for |
| --- | --- | --- |
| Local: `getting-started/scripts/run_local.py` | The public `val-sample` (24 cases, only **1** small-model case), run through the same ingestion and scoring programs Codabench uses | Contract and packaging checks: `coverage 1.0`, `invalid_predictions 0`, `small.txt` placement. **Not** a score estimate. |
| **Live Codabench leaderboard** | Task "AIMO Robustness Validation v7 - Small Model Track" in the Competition Phase. Input and reference data are non-public, i.e. the hidden validation set, which the website says "covers all types of models contained in the test set". | The only external estimate of real performance. It is public and shows Accuracy, Coverage, and Invalid predictions. |
| Final ranking | Held-out private test set, scored after the deadline | Prizes. Ties are broken by average runtime over repeated runs. |

Scoring details from `components/scoring_program/scoring.py`:

- `accuracy = correct / total labels`.
- A missing, duplicated, or non-bool prediction counts as wrong and lowers `coverage`.
- The small track runs identical scoring; the track only enforces the `small.txt` marker.

Submission rules from the Codabench API:

- **10 submissions per day and 100 per person in total.**
- 3600 s execution limit per submission.
- Leaderboard rule `Force_Latest_Multiple`.

The leaderboard metric is raw accuracy, which depends on the class prior, while our development metric is balanced accuracy. Choose the decision threshold deliberately, and do not tune it by leaderboard probing: the budget is small and the validation set is not the test set.

---

## 6. Key dates

| When | What |
| --- | --- |
| 30 Apr – 31 Aug 2026 | Codabench Warmup Phase (task "AIMO Robustness Validation v8") |
| **1 Sep – 1 Nov 2026, 12:55 UTC** | **Codabench Competition Phase (we are in this window)**: Main Track and Small Model Track tasks. The website describes the competition phase as 15 Aug – 1 Nov. |
| **1 Nov 2026, 12:55 UTC** | **Final submission deadline**: Codabench closes |
| 1–15 Nov 2026 | Technical report + open-sourced implementation |
| 15 Nov 2026 | Report due for award-eligible teams |
| 15–30 Nov 2026 | Result validation, report review, rules compliance; award-eligible teams peer-review up to 3 reports |
| 11/12 Dec 2026 | Competition workshop at NeurIPS 2026 |

---

## 7. Competition rules (award-eligible submissions)

1. **Signals.** Weights, activations, token probabilities, or any other internal signals are allowed. White-box methods are encouraged. Black-box behavioural analysis is allowed if it fits the evaluation compute budget (for example, using reasoning traces).
2. **Tracks.** Each ZIP counts for one track. The same method can be uploaded twice (with / without `small.txt`).
3. **Report.** Award-considered teams must submit a technical report and open-sourced implementation by **15 Nov 2026**.
4. **Transparency.** Organizers must be able to reproduce the score from the report alone. Document training data, training codebase, and the submission bundle. The submission may depend only on documented resources.
5. **Reviewing.** Award-considered teams peer-review up to 3 other reports, 15–30 Nov 2026.

Proposal-paper extra (overfitting control): do not use extra **labeled** data for the same perturbation types as the final validation set. Unlabeled data and labels for *other* perturbation types are allowed.

Open participation: no affiliation, team-size, or career-level restrictions. You do **not** need to have competed in AIMO.

---

## 8. Open-weight vs closed models (why this matters)

The public `val-sample` mixes **closed APIs** and **open-weight** systems. White-box internals are only available for open-weight checkpoints cached on the workers.

### Models evaluated in the Competition Phase (upstream README, commit `995f992`, checked 16 Sep 2026)

| `model_id` (passed verbatim) | Main Track | Small Models Track | Public labels |
| --- | --- | --- | --- |
| `Qwen/Qwen3.5-4B` | yes | **yes** | none |
| `Skywork/Skywork-OR1-Math-7B` | yes | **yes** | none |
| `allenai/Olmo-3-7B-Think` | yes | **yes** | none |
| `deepseek-ai/DeepSeek-R1-0528-Qwen3-8B` | yes | **yes** | 138 train-ready |
| `openai/gpt-oss-120b` | yes | no | — |

- **No aliases.** The `:low`-suffixed names in the public datasets are never passed during the
  Competition Phase.
- **Warm-up only.** `allenai/Olmo-3-7B-Instruct` and `google/gemma-3-27b-it` were used in the
  warm-up phase only. `Olmo-3-7B-Think` is a different checkpoint from `Olmo-3-7B-Instruct`.
- **Hardware.** Each evaluation container sees **one** RTX PRO 6000 (96 GB), and every model
  fits on it. `gpt-oss-120b` is now cached in MXFP4 (about 62 GiB). A 4–8B checkpoint loads in
  about 2 s from the warm cache.
- **Interface.** The unmerged `reasoning_effort` branch adds an optional
  `are_robust(model_id, reasoning_effort, problems)`. It detects the parameter by name and
  keeps the legacy two-argument call working.

Anything outside this cache cannot be loaded during evaluation. Contact organizers if another checkpoint must be mounted.

### Aliases seen in public data (not all are cached / open)

| `model_id` in val-sample | Notes |
| --- | --- |
| `qwen3-8b:low` | Open-weight 8B. Resolves to DeepSeek-R1-0528-Qwen3-8B. **Our focus.** |
| `gpt-oss-120b:low` | Open-weight 120B. Main Track. |
| `huikang-gpt-oss-120b-aimo3:low` | Open-weight AIMO 3 submission variant. Main Track. |
| `Qwen/Qwen3.5-397B-A17B-FP8:low` | Open-weight, far above 10B. Main Track. |
| `gpt-5.2-2025-12-11:low` | Closed. Internals not in the HF cache. |
| `gemini-3.1-pro-preview:low` | Closed. |
| `lukealonso/GLM-5.1-NVFP4:low` | Not a worker-cached small open-weight target. |

Small-track scoring will subset to <10B. For white-box work, treat **DeepSeek-R1-0528-Qwen3-8B** and **Olmo-3-7B-Instruct** as the models we can actually inspect.

---

## 9. Data

### Problem sources (proposal)

About **180** olympiad-level problems from AIMO 2, AIMO 3, JMO, and recent AIME. ~50% of both train and test are unpublished. Licenses: public AIMO/AIME Apache 2.0; private AIMO/JMO cleared with owners.

Planned splits:

- Train: ~150 problems (published + unpublished)
- Test: **30 private AIMO 3 problems**, never public; only reachable through Codabench scoring
- Effective label volume is the product of problems × models (proposal target: ≥8 models → ~1200 train / ~240 test instances)

### Robustness labels

Symbolic reasoning chains (GSM-Symbolic-style, olympiad-level) generate answer-preserving and adversarial variants. A model is **not robust** when a majority of perturbations under a shift Δ flip a previously correct answer.

Perturbation types already used:

| Type | What it does |
| --- | --- |
| `rephrase` | Semantic-preserving rewrite |
| `rename` | Rename variables / entities |
| `domain` | Same math, different surface domain |
| `distract` | Irrelevant but plausible extra text |
| `typos` | Harmless noise |
| `expert_perturbations` | Expert answer-preserving adversarial edits |
| `expert_no_solution` | Unanswerable variant (missing premises) — tests unsolvability detection |

Public development label field: `model_is_robust`. Related numeric fields: `base_accuracy`, `permuted_accuracy`, `absolute_accuracy_decay`, `relative_accuracy_decay`.

The public sample is a **packaging and contract check**, not a preview of private-track scores. All model *types* in the private test set are represented in the public validation set.

### Hugging Face datasets

| Dataset | Use |
| --- | --- |
| [`aimo-interp/val-sample`](https://huggingface.co/datasets/aimo-interp/val-sample) | 28 public validation rows; mixed models; import with `uv run scripts/import_hf_dataset.py` |
| [`aimo-interp/problems-public`](https://huggingface.co/datasets/aimo-interp/problems-public) | Problems + symbolic chains (AIME 2026, AIMO 2 reference, …) |
| [`aimo-interp/augmented-sample-math-agg`](https://huggingface.co/datasets/aimo-interp/augmented-sample-math-agg) | Larger labeled aggregate for `qwen3-8b:low` (Hendrycks MATH). Used by the uncertainty baseline |

Local import layout after `import_hf_dataset.py`:

```text
data/val-sample/
├── input/cases.jsonl
├── reference/labels.jsonl
└── metadata.json
```

---

## 10. Evaluation environment (Codabench workers)

| Resource | Spec |
| --- | --- |
| GPU | 8× NVIDIA RTX PRO 6000 (Blackwell, sm_120), 96 GB each |
| CPU / RAM | 192 cores, ample RAM, no per-container memory cap |
| Time | 3600 s for prediction and 3600 s for scoring, separately. Loading the 8B model ≈ 1 minute |
| Network | Disabled. `HF_HUB_OFFLINE=1`, `TRANSFORMERS_OFFLINE=1` |
| Cache | Read-only HF cache at `/app/data/huggingface` (`HF_HOME` already set) |
| Install | No `pip`, no `uv`, no `requirements.txt` at eval time |

VRAM is shared with other jobs; `device_map="auto"` can shard, but free memory may be below the nominal total. OOM may be contention — resubmit, then ping organizers.

Proposal-paper compute grants (separate from scoring): teams may request up to **10,000 H200 GPU-hours** via the Fields Model form (compute in exchange for open math data / artifacts; details on fieldsmodel.org).

---

## 11. Allowed runtime libraries

Pinned in [`Dockerfile.competition`](https://github.com/aimo-interp/getting-started/blob/main/Dockerfile.competition):

| Package | Version |
| --- | --- |
| PyTorch | 2.12.1 + CUDA 13.2 + cuDNN 9 (Blackwell) |
| Accelerate | 1.13.0 |
| Hugging Face Hub | 1.22.0 |
| joblib | 1.5.3 |
| pandas | 3.0.3 |
| safetensors | 0.8.0 |
| scikit-learn | 1.8.0 |
| tokenizers | 0.22.2 |
| Transformers | 5.13.0 |

Serialize sklearn / joblib artifacts with **these** versions. Extra Python source and pretrained probes/SAEs may be bundled in the ZIP. Do not bundle model weights that the worker already caches, datasets, venvs, or credentials.

---

## 12. What to submit

### Small Models Track ZIP (required layout)

```text
solution-our-method-small.zip
├── solution.py          # must be at archive root
├── small.txt            # empty marker; required for this track
├── ...helper modules...
└── artifacts/           # probes, SAE weights, sklearn joblib, etc.
```

Build helper from the starter kit:

```bash
uv run scripts/build.py solutions --small
```

Example output: `dist/solution-trained-probe-small.zip`.

### Do not submit

- Training datasets or private labels
- Cached LM weights already on the worker
- Virtualenvs, `__pycache__`, installed packages
- API keys or any code that needs the network
- A nested folder wrapping `solution.py`

### How to upload

1. Sign in on [Codabench 16180](https://www.codabench.org/competitions/16180/#/pages-tab).
2. Open the active phase and select **exactly one task** (Small Models).
3. Upload the ZIP that contains `small.txt`.
4. Wait for ingestion/scoring. Check `coverage` and `invalid_predictions` first.

Invalid predictions usually mean an exception, a non-bool, or the wrong list length. Participant tracebacks may be hidden — reproduce locally in the competition Docker image.

---

## 13. Starter methods (starting points)

Clone and sync:

```bash
git clone https://github.com/aimo-interp/getting-started.git
cd getting-started
uv sync   # Python 3.11+
```

Quick local score:

```bash
uv run scripts/import_hf_dataset.py
uv run scripts/run_local.py solutions/trained-probe \
  --input-dir data/val-sample/input \
  --reference-dir data/val-sample/reference
```

| Example | Idea | Notes for our track |
| --- | --- | --- |
| `solutions/always-true` | Trivial constant predictor | Packaging check |
| `solutions/trained-probe` | Linear probes on last prompt-token hidden states, best layer, margin ensemble | White-box; natural SAE/transcoder extension |
| `solutions/uncertainty-profiling` | 14 token-level uncertainty features → sklearn regressor of `absolute_accuracy_decay` | Already wired to `qwen3-8b:low` / DeepSeek-R1-0528-Qwen3-8B. Greedy decode, 2048 prompt / 4096 gen tokens. OOF balanced acc ≈ 0.63 |

Published baseline numbers on DeepSeek-R1-0528-Qwen3-8B (proposal, stratified 10-fold, sample val):

- Linear probe: **58.37% ± 7.5**
- Uncertainty classifier: **69.23% ± 10.13**
- Chance: 50%

A third baseline (data attribution / memorization) is promised but not yet in the starter ZIP.

Training for uncertainty profiling uses `aimo-interp/augmented-sample-math-agg@f972ced…`. Do not ship that parquet in the submission — only the fitted joblib.

---

## 14. Local acceptance checks before a Codabench upload

- `coverage == 1.0`, `invalid_predictions == 0`
- Deterministic across repeated runs
- Succeeds in `Dockerfile.competition`, not only a laptop venv
- `unzip -l` shows `solution.py` and `small.txt` at the ZIP root
- `uv run scripts/run_local.py dist/solution-…-small.zip --small`

---

## 15. How to participate (checklist)

1. Register for email updates: https://forms.gle/EmvfJz2uSEQpbttB9
2. Join Discord: https://discord.gg/PaAMFmpsBQ
3. Create a Codabench account and enter competition 16180
4. Clone `getting-started`, import `val-sample`, run a baseline locally
5. Optional: request Fields Model compute — https://forms.gle/oYkYDN1EtSVeSfMd9
6. Implement `are_robust`, train artifacts offline on open-weight 7B/8B models
7. Package with `small.txt`, upload to the Small Models task
8. Iterate until **1 Nov 2026**
9. If award-eligible: report + open source by **15 Nov 2026**, then review peers

---

## 16. Kaggle in this workspace

AIMO Interpretability scoring is **Codabench**. Kaggle still matters because AIMO 3 (the upstream open-weight math competition) and model mirrors live there.

Local CLI (Kaggle CLI 2.2.2) is authenticated from `Secrets.env`. Tokens were introspected with `kaggle config view`; the active CLI account is **ISHRAQ**.

| Alias | Kaggle username |
| --- | --- |
| ISHRAQ (default) | `farhanishraqq` |
| DIDHITI | `didhitinahid` |
| TASHIN | `tashintahir` |
| SALAM | `salam2026` |
| ISHMAM1 | `ishmamahmid` |
| DESIKALAKAR | `fatkhato` |

| File | Role |
| --- | --- |
| `Secrets.env` | Per-alias `KAGGLE_USERNAME_*` and `KAGGLE_API_TOKEN_*` (KGAT_ tokens). **Do not commit.** |
| `scripts/use_kaggle_account.py` | Copies the chosen token to `%USERPROFILE%\.kaggle\access_token` |
| `%USERPROFILE%\.kaggle\accounts\` | One `*.access_token` file per alias |

```powershell
python scripts\use_kaggle_account.py ISHRAQ
kaggle config view
kaggle competitions list -s aimo
```

---

## 17. Organizers (contacts)

Lead: Michal Štefánik (NII) — website / communication.

Core: Philipp Mondorf (LMU) — baselines / leaderboard; Andreas Waldis (Tübingen) — baselines / beta.

Infrastructure / Codabench: Josef Kuchař, Adam Vawda-Oomerjee, Chaoran Liu.

Data / math: Chuan Yang, Qianying Liu, Róbert Belanec, Michal Spiegel, Marek Kadlčík.

Advisory: Simon Frieder (AIMO), Barbara Plank, Fazl Barez, Pontus Stenetorp.

Primary contacts: Discord and **aimo.interp@gmail.com** (they aim to reply within 48 hours). Technical issues: GitHub issues on `getting-started`.

---

## 18. Working notes for the Small / open-weight track

- Prefer methods that read **internals** of `DeepSeek-R1-0528-Qwen3-8B` and `Olmo-3-7B-Instruct`.
- Keep a black-box fallback (uncertainty / trace stats) so unknown `model_id`s still return valid bools.
- SAE / transcoder training happens **offline** on our machines or Fields Model GPUs; only compact artifacts go in the ZIP.
- Do not depend on closed-model APIs at evaluation time — the container has no network.
- Time budget is one hour; 8B load already costs ~1 minute. Generation-heavy methods must cap tokens (the uncertainty baseline uses 4096).
- Reproducibility for awards: log exact HF revisions, training scripts, and the exact ZIP contents in the November report.
