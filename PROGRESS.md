# Progress log: AIMO Interpretability Challenge, Small Models Track

A running record of what has been built, measured, and learned.

Companion documents:

- `COMPETITION.md`: rules and logistics.
- `PLAN.md`: forward plan and experiment registry.
- `RESULTS.md`: generated metrics.
- `CLAUDE.md`: working conventions and commands.

Last updated: 15 September 2026.

---

## 1. Where things stand

| Area | Status |
| --- | --- |
| Competition brief and rules | Done: `COMPETITION.md` |
| Execution plan | Redrafted 15 Sep: `PLAN.md` |
| Repo guide | Done: `CLAUDE.md` |
| Kaggle fleet (6 accounts) | **Verified 15 Sep**. Experiments run through `datagen/kaggle_launch.py` |
| Local dev environment | Working, 25/25 starter tests pass |
| Submission bundles | 6 ZIPs built and validated locally |
| Training set | Built: 187 instances, 138 for DeepSeek-8B |
| Stage A (prompt hidden states) | **E1 done** on Kaggle T4x2 in bf16, analysed (Finding 11) |
| Stage B (generation traces) | **E2 running** (2 shards, ISHRAQ + DIDHITI) |
| Olmo-3-7B plumbing | **E3 done**: loads and runs clean (Finding 12) |
| Text-only floor | 0.6732 balanced accuracy from problem text alone |
| Codabench entry | **Not created yet**: blocked on your account |
| Organizer questions | **Not sent yet** |

The track target is to beat the published baselines, 58.4% (trained probe) and 69.2%
(uncertainty profiling), on the Codabench leaderboard for competition 16180.

---

## 2. What the competition asks for

Given a `(model_id, problem)` pair, predict whether that model's answer is **robust**
(survives perturbations that preserve the deductive chain) or **spurious** (relies on brittle
pattern matching). The submission implements:

```python
def are_robust(model_id: str, problems: list[str]) -> list[bool]: ...
```

We are entering the **Small Models Track, open-weight category**: models below 10B
parameters. The track is selected by a root-level `small.txt` marker file in the bundle.

- **Platform:** scoring is on **Codabench** (competition 16180), not Kaggle. Kaggle matters
  only as free GPU capacity.
- **Metric:** accuracy is primary, with `coverage` and `invalid_predictions` reported
  alongside. Ties break on runtime over three trials.
- **Environment:** evaluation containers are offline (`HF_HUB_OFFLINE=1`,
  `local_files_only=True`) against a read-only Hugging Face cache.

---

## 3. Environment and credentials

**Kaggle fleet.** Six accounts, all re-verified on 15 Sep; each token resolves to its
expected username.

Kaggle CLI 2.x does not use the legacy `username` + `key` pair. It reads an **access token**
(`KGAT_…`) from `KAGGLE_API_TOKEN` or `~/.kaggle/access_token`. The first setup attempt
assumed HTTP Basic auth with legacy keys and got HTTP 401 on all six accounts. Reading
`kaggle/api/kaggle_api_extended.py` and `kagglesdk/kaggle_env.py` showed the actual mechanism.

- `scripts/_setup_kaggle_accounts.py` verifies each token by introspecting it through
  `kaggle config view`, then writes per-alias token files to
  `~/.kaggle/accounts/<alias>.access_token`.
- `scripts/use_kaggle_account.py ISHRAQ` switches the default CLI account.
- `datagen/kaggle_launch.py` passes the right token per call through the child process
  environment and scrubs token-shaped strings from anything it prints.

**CLI 2.2.2 quirks found on 15 Sep:**

- `kaggle quota` crashes with "not enough values to unpack".
- `kaggle kernels output` fetches a single page of files (default 20, max 200).
- `kaggle datasets list` has no `--page-size`.
- On a cp1252 Windows console, the CLI exits 1 after a successful download when it echoes a
  Unicode kernel log. The launcher now forces UTF-8.

**Toolchain.** `uv` 0.12.13 is available via `python -m pip install uv`. It must be invoked
as `python -m uv`, since the bare executable is not on `PATH`.

**Local hardware cannot run the target models.** The machine has an RTX 3050 6 GB and
31.7 GB RAM, against roughly 16 GB needed for an 8B model in half precision. It handles
packaging, data work, CPU training, and smoke tests on
`hf-internal-testing/tiny-random-LlamaForCausalLM`, which is locally cached. Every
model-dependent run goes to Kaggle.

---

## 4. Submission pipeline validated end to end

The starter kit is cloned at `getting-started/` (commit `e98c489`). All 25 tests pass under
the pinned runtime (transformers 5.13.0, torch ~2.12.1, scikit-learn 1.8.0).

All six bundles build, and `small.txt` placement is correct in each:

| Bundle | Size |
| --- | --- |
| `solution-always-true.zip` / `-small.zip` | <0.01 MB |
| `solution-trained-probe.zip` / `-small.zip` | 26.41 MB |
| `solution-uncertainty-profiling.zip` / `-small.zip` | 0.98 MB |

Local scoring against the imported validation sample runs clean. The always-true baseline
returns `accuracy 0.375, coverage 1.0, invalid_predictions 0`, consistent with the sample
being 37.5% robust.

---

## 5. Data work

### 5.1 Six public datasets, not two

The website links two datasets. Listing the Hub author revealed **six**, four of them
undocumented, and three of those carry small-model rows:

| Dataset | Rows | Small-model content |
| --- | --- | --- |
| `aimo-interp-challenge-sample-full` | 558 | 64 `qwen3-8b:low`, per-perturbation, **no label column** |
| `augmented-sample-math-full-filtered` | 259 | 259 `qwen3-8b:low`, per-perturbation, **no label column** |
| `augmented-sample-math` | 156 | 156 `meta-llama/Llama-3.1-8B-Instruct`, all spurious |
| `augmented-sample-math-agg` | 141 | 141 `qwen3-8b:low`, labelled 36/105 |
| `augmented-sample-math-agg-filtered` | 54 | 54 `deepseek-ai/DeepSeek-R1-0528-Qwen3-8B`, labelled 11/43 |
| `val-sample` | 28 | **1** `qwen3-8b:low` case, **0** Olmo |

`aimo-interp/problems-public`, the dataset holding the **symbolic reasoning chains**, returns
**401 Unauthorized** and is not listed, so it is gated or private. This rules out exact
answer-recomputing perturbations and pushes generator design toward answer-preserving
transforms. Worth requesting access.

A side discovery: `meta-llama/Llama-3.1-8B-Instruct` appears as an evaluated model. It is a
fourth open-weight model under 10B, and it is absent from the worker cache list.

### 5.2 The organizers' labelling rule, recovered

The two largest tables ship accuracy-decay columns without `model_is_robust`. Fitting
candidate rules against all 379 published labels (`scripts/infer_label_rule.py`):

| Rule | Agreement |
| --- | --- |
| `absolute_accuracy_decay == 0` | **379 / 379** |
| `n_detrimental_permutations == 0` | **379 / 379** |
| `relative_accuracy_decay == 0` | **379 / 379** |

The cross-tabulation separates perfectly: every robust row sits at decay 0, every spurious
row above 0.3, and nothing falls between. That made it safe to label roughly 800 previously
unusable rows.

Two corrections came out of auditing those derived labels:

1. **Negative decay exists.** Fifteen rows show the model scoring *better* under
   perturbation, down to −0.43. A literal `== 0` rule calls these spurious, which is wrong,
   so the implemented rule is `decay <= 0`. No published row has negative decay, so this
   agrees with the recovered rule everywhere it is testable.
2. **Eligibility filtering.** The task presumes the model already answers the original
   correctly. **295 of 298 published rows sit at `base_accuracy == 1.0`**, while derived rows
   include 36 below 0.7 (some as low as 0.167). We mirror that curation with
   `base_accuracy >= 1.0`, and also drop the ambiguous `0 < decay <= 0.3` band that the
   proposal says organizers exclude.

### 5.3 The resulting training set

`datagen/ingest_public.py` consolidates all six tables into `data/small_track/`:

| Stage | Count |
| --- | --- |
| Deduplicated perturbation rows (small models) | 574 |
| Collapsed to (model, problem) instances | 213 |
| Train-ready after eligibility and clarity filters | **187** |

| Model | Train instances | Robust / spurious |
| --- | --- | --- |
| `deepseek-ai/DeepSeek-R1-0528-Qwen3-8B` | **138** | 36 / 102 |
| `meta-llama/Llama-3.1-8B-Instruct` | 49 | 0 / 49 (degenerate) |

That is roughly 138× the small-model signal in `val-sample`, obtained with zero GPU hours.
Folds are grouped by normalised problem text. This matters because problems recur across
models: 31 of the 156 train problems, or 39 of 174 across all instances. Ungrouped
splitting would leak.

---

## 6. Findings

**1. The public validation sample gives the small track almost no signal.**

- Of its 24 deduplicated cases, only **one** is `qwen3-8b:low` and **zero** are Olmo-3-7B.
- Totals are 9 robust / 15 spurious, so always-false scores 0.625 and always-true 0.375.
- The site claims the validation set "covers all types of models contained in the test set".
  Either the full set is far richer, or Olmo is not in the small test set. This is an open
  question for the organizers.

**2. The shipped probe baseline is at chance, not 58%.**

- Reading `probe_artifact.pkl` (1,480 fits = 37 layers × 5 seeds × 4 folds × 2 control
  tasks), mean per-layer validation balanced accuracy is **0.5005**. The best layer (31)
  reaches only **0.5059**.
- The artifact's advertised `best_probe.score` of 0.7375 is a single cherry-picked fold.
- Artifact details needed to reproduce it:
  - `hf_model_id = deepseek-ai/DeepSeek-R1-0528-Qwen3-8B`
  - 37 hidden states
  - aggregation `mean_margin`
  - target `model_is_robust`
  - system prompt SHA-1 `9021e9d4…`, read at runtime and never transcribed

**3. Inference exposes very little.** Cases carry `{id, model_id, problem}`; labels carry
`{id, is_robust}`. There is no `permutation_type` at inference, so any
perturbation-type-aware method must infer it.

**4. Local hardware cannot run the target models.** See §3.

**5. Four undocumented datasets, one gated.** See §5.1.

**6. The labelling rule was recovered exactly.** See §5.2.

**7. The training set exists.** See §5.3.

**8. Stage A feature extraction does not need the fleet.**

- Round-robin shard assignment over hash-sorted keys balances 138 instances to 23 per shard
  across six. The original hash assignment gave 15–34.
- Stage A turned out to be even cheaper than planned (Finding 10), so it runs on one account.
- **Model filtering must be on by default.** `is_robust` is specific to a (model, problem)
  pair, and 31 train problems recur across models. Extracting one model's activations
  against another's label would silently mislabel data.

**9. Problem text alone beats the shipped probe, and it is not a confound.** See §7.

**10. Kaggle's T4s run bf16, the evaluation dtype, and Stage A takes 1 s per instance.**

- **Dtype.** The plan assumed fp16 on Kaggle because the T4 (Turing) has no native bf16.
  In practice Kaggle's torch 2.10.0+cu128 reports `torch.cuda.is_bf16_supported() == True`
  (emulated). Both our extractor and the starter kit's loader therefore picked bf16.
  - E1 and E3 recorded `dtype=torch.bfloat16` in `metadata.json`, with **zero** non-finite
    activations across 294 forward passes.
  - The planned fp16-vs-bf16 drift experiment is unnecessary.
  - The non-finite guards remain for any host that falls back to fp16.
- **Speed.** Measured on T4x2 with `device_map="auto"`: 1.0 s per instance and 2.4 minutes
  for 138 instances. The whole E1 kernel took 311 s, including pip, the 16 GB model download,
  and the on-Kaggle merge. The earlier estimate of 8 s per instance was 8× pessimistic.
- **Prompt length.** Median 95 tokens, max 248, none truncated.
- **A multi-GPU bug was caught before launch.** `pool_hidden_states` indexed each layer with
  an attention mask living on `cuda:0`, which fails for layers that `device_map="auto"`
  places on `cuda:1`. The earlier smoke test ran on one device and could not have shown it.

**11. Prompt hidden states: selecting a layer fails, averaging all layers is promising but
not established.** (E1, `train/train_stagea_probes.py`; 138 instances, 37 layers × 3
poolings, same folds as the text floor.)

| Method | Balanced acc. | ROC-AUC | AUC minus text (95% paired bootstrap CI) |
| --- | --- | --- | --- |
| Char tf-idf text (recomputed; matches the floor) | 0.6732 | 0.6686 | — |
| 5 next-token uncertainty scalars | 0.5809 | 0.6392 | −0.029 [−0.169, +0.123] |
| Nested-selected probe (layer, pooling chosen in-fold) | 0.5417 | 0.5929 | −0.076 [−0.195, +0.040] |
| Layer ensemble, `last` pooling (no selection) | 0.6340 | 0.6955 | +0.027 [−0.071, +0.126] |
| Layer ensemble, `mean` pooling | 0.6209 | 0.7320 | +0.064 [−0.029, +0.155] |
| Layer ensemble, all poolings | 0.6201 | 0.7331 | +0.065 [−0.024, +0.153] |
| Fusion: text + all-pooling ensemble | 0.6291 | 0.7260 | +0.058 [−0.007, +0.124] |

- **Selection overfits.** Inner-CV AUCs of 0.76–0.83 collapse to 0.59 on held-out folds.
  The chosen configuration differs in every fold (mean L23, max L16, max L7, mean L14,
  max L29). Picking a layer is fitting noise: the same effect behind the baseline's 58.4%.
- **Averaging recovers a ranking signal.** Ensembles over all layers out-rank text by about
  0.06 AUC. At n=138 every interval spans zero, so this is a lead, not a result.
- **Balanced accuracy hides it.** Hidden-state probabilities sit off-centre: the ensemble
  median is 0.25, and only 33 of 138 are predicted robust at 0.5. At a prior-matched
  threshold the mean- and max-pooling ensembles reach 0.643 against text's 0.624. A shipped
  version must learn its threshold in-fold.
- **Per-layer probes are not at chance here.** Mean per-layer balanced accuracy is 0.58–0.62
  across poolings, against the shipped artifact's 0.5005. The training setups differ, so
  this does not by itself show the baseline was mis-trained.
- **Implication for Lane B.** Drop single-layer probing; keep a layer-ensemble block as one
  input to fusion. Testing whether states over the *generated trace* carry more than prompt
  states is now a cheap experiment, because Stage A costs 1 s per instance.

**12. Olmo-3-7B-Instruct plumbing works (E3).**

- It loads under transformers 5.13.0 on T4x2 in bf16 and exposes **33 hidden states** (32
  layers plus embeddings) × 4096.
- It runs at 1.0 s per instance with zero non-finite activations; prompts have a median of
  101 tokens under its chat template.
- The run used `--no-filter-model`, so its labels are DeepSeek's and the features are for
  plumbing only.
- Mechanically, nothing blocks Olmo labelling now; only the organizer answer does.

---

## 7. Results so far

### Text-only floor

Out-of-fold over the five problem-grouped folds, on the 138 DeepSeek instances. Features come
from the problem statement only, which is all inference provides. Full detail in `RESULTS.md`.

| Method | Accuracy | Balanced acc. | ROC-AUC |
| --- | --- | --- | --- |
| always-spurious (majority) | 0.7391 | 0.5000 | 0.5000 |
| logreg, 44 handcrafted surface features | 0.6014 | 0.5417 | 0.5417 |
| hist-gradient-boosting, handcrafted | 0.7101 | 0.5882 | 0.6757 |
| logreg, word tf-idf | 0.6812 | 0.6315 | 0.6773 |
| **logreg, char tf-idf (3–5, `char_wb`)** | 0.7029 | **0.6732** | 0.6686 |

- **The floor for model-internals work is 0.673 balanced accuracy, not 0.5.** Differences
  must now also clear a paired bootstrap CI, because a single AUC at n=138 carries about
  ±0.09 of noise.
- **Raw accuracy is a trap on our split.** At 73.9% spurious, always-spurious scores 0.7391
  and beats every real method on the competition's primary metric while being useless.
- **The signal is not a dataset artifact.** Holding the source dataset constant, char tf-idf
  still reaches 0.670 balanced accuracy within `hendrycks-math-train-reference-500` (n=84)
  and 0.644 within `hendrycks-math-test-reference-100` (n=53).

Interpretable direction of the surface features: brace count, word count, and the keyword
"integer" push toward robust; backslash density, `\frac`, and `\pmod` push toward spurious.

### Stage A hidden states

See Finding 11 and the generated section in `RESULTS.md`.

### The measurement we cannot yet make

After filtering, the DeepSeek train set contains **exactly one olympiad-level instance**
alongside 137 Hendrycks MATH problems, while the private test set is entirely olympiad.
Transfer from MATH to olympiad, which is precisely what the competition scores, cannot be
measured with current data. This remains the highest-value gap to close.

---

## 8. Code inventory

### `datagen/`: data and features

| File | Purpose |
| --- | --- |
| `ingest_public.py` | Consolidates all six public datasets into `data/small_track/`: derives labels, applies eligibility and clarity filters, builds grouped folds |
| `extract_features.py` | Stage A: one forward pass per problem; pooled hidden states at every layer (`last`/`mean`/`max`) plus next-token uncertainty and non-finite counts. Shardable, resumable, multi-GPU safe |
| `extract_generation_features.py` | Stage B: greedy generation through the starter kit's `uncertainty_profile` (unchanged). Adds the 14 trace-uncertainty features, generated text, and a token-cap flag. Aborts on non-finite preflight logits |
| `merge_features.py` | Merges Stage A shards, refusing any that disagree on dtype, layer count, or prompt hash |
| `plan_shards.py` | Reports shard workload and balance without loading a model (default 1.0 s per instance, measured) |
| `kaggle_launch.py` | Experiment registry; renders self-contained private GPU kernels (embedded 63 KB code bundle) and runs `push` / `status` / `logs` / `output` per account |
| `README.md` | Runbook |

### `train/`: modelling

| File | Purpose |
| --- | --- |
| `train_text_baseline.py` | Text-only baselines under grouped CV; rewrites `RESULTS.md` |
| `check_dataset_confound.py` | Tests whether the text signal is really dataset detection; appends to `RESULTS.md` |
| `train_stagea_probes.py` | Nested probe, layer ensembles, text fusion, and bootstrap CIs on Stage A features; upserts its `RESULTS.md` section and saves out-of-fold probabilities |
| `train_stageb_features.py` | Trace-uncertainty and trace-shape classifiers, univariate AUCs, and text fusion on Stage B records; upserts its section |
| `results_io.py` | Shared metrics, paired bootstrap AUC difference, section upsert |

### `scripts/`: setup and one-off audits

| File | Purpose |
| --- | --- |
| `_setup_kaggle_accounts.py` | Verifies and installs the six Kaggle access tokens |
| `use_kaggle_account.py` | Switches the default Kaggle account |
| `check_hf_access.py` | Reports which datasets are reachable with current credentials |
| `profile_hf_datasets.py` | Schema, size, and model coverage for every public dataset |
| `infer_label_rule.py` | Recovers the organizers' labelling rule from published labels |
| `inspect_small_track.py` | Summarises the consolidated training set |
| `check_base_accuracy.py` | Compares base accuracy between published and derived rows |
| `check_negative_decay.py` | Finds rows where the model improved under perturbation |

**Verification before any GPU spend:**

- **Stage A:** extract, resume, and merge were exercised on the tiny cached Llama.
- **Stage B:** shard selection, preflight, per-batch flush, resume, and metadata were
  exercised the same way.
- **Kernels:** every rendered kernel was compiled and its bundle unpacked, and all three
  entry points were imported from the bundle root.
- **Analysis scripts:** both were run on synthetic data with planted signal, using the real
  keys and folds.
  - The nested probe found the planted layer in all five folds.
  - The Stage B script recovered the planted trace features.

Along the way this caught a pandas 3 copy-on-write read-only array bug in the Stage A script.

---

## 9. How to reproduce

```powershell
$py = "getting-started\.venv\Scripts\python.exe"

# Training set (no GPU, ~30 s), then the text floor and confound check
& $py datagen\ingest_public.py
& $py train\train_text_baseline.py
& $py train\check_dataset_confound.py

# GPU stages on the Kaggle fleet
& $py datagen\kaggle_launch.py push e1-stagea-deepseek
& $py datagen\kaggle_launch.py status
& $py datagen\kaggle_launch.py output e1-stagea-deepseek

# Analysis (each upserts its own RESULTS.md section)
& $py train\train_stagea_probes.py --features-dir runs\e1-stagea-deepseek\features\stagea-deepseek
& $py datagen\kaggle_launch.py output e2-stageb-deepseek-s0 e2-stageb-deepseek-s1
& $py train\train_stageb_features.py
```

`data/`, `features/`, `runs/`, and `kaggle_runs/` are git-ignored; all regenerate
deterministically.

---

## 10. Open questions and blockers

**For the organizers** (none sent yet; batch them into one message, drafted in `PLAN.md` §10):

1. Is Olmo-3-7B in the small-track test set, and under which alias?
2. Can we get access to `problems-public`?
3. Do labels for non-official perturbation categories satisfy the extra-data rule?
4. Is olympiad-difficulty labelled data for small models coming, and when is the train split
   released?

**Requiring your accounts:**

- Create a Codabench account, enter competition 16180, and upload
  `getting-started/dist/solution-trained-probe-small.zip` and
  `solution-uncertainty-profiling-small.zip`.
- Submit the Fields Model compute-grant form (`https://forms.gle/oYkYDN1EtSVeSfMd9`).

**Housekeeping:**

- **This workspace is still not a git repository**, so `.gitignore` is inert while
  `Secrets.env` sits in the tree. Initialise git before any commit or push and confirm
  `Secrets.env` is excluded.
- `General-Instructions.md` is still an empty placeholder, and `Inferred-Notebooks/` is empty.

---

## 11. Next steps

Ranked by value:

1. **Analyse E2** when both shards finish: `kaggle_launch.py output`, then
   `train_stageb_features.py`. Together with E1 this decides gate G1.
2. **Close the olympiad-difficulty gap.** Filter the locally cached AIME/OlympiadBench pool
   and build the generator for non-official perturbation categories (`PLAN.md` §3.4).
3. **Trace-state probes (E4).** Run Stage A over prompt + generated trace using E2's text, to
   test whether states during reasoning carry more than prompt states (Finding 11).
4. **Fusion v1.** Combine text, the layer ensemble, and trace features, with in-fold
   thresholds and bootstrap CIs against text.
5. **Send the organizer questions** and **get a baseline onto Codabench** (needs your accounts).
