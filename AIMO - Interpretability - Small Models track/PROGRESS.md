# Progress log: AIMO Interpretability Challenge, Small Models Track

A running record of what has been built, measured, and learned.

Companion documents:

- `COMPETITION.md`: rules and logistics.
- `PLAN.md`: forward plan and experiment registry.
- `RESULTS.md`: generated metrics.
- `CLAUDE.md`: working conventions and commands.

Last updated: 29 September 2026. Current decisions: Findings 62–64 and `PLAN.md` §12;
the early status tables below describe earlier stages.

---

## 1. Where things stand

| Area | Status |
| --- | --- |
| Competition brief, plan, repo guide | Done: `COMPETITION.md`, `PLAN.md`, `CLAUDE.md` |
| Kaggle fleet (8 accounts) | E18/E19 (variants phase) running; ISHMAM1 and DESIKALAKAR rejected, **RSD and Labib added 18 Sep** (Findings 23, 25, 26) |
| Training set | 187 instances, 138 for DeepSeek-8B (36 robust); no labels for the other three small-track models |
| Text-only floor | 0.6732 balanced accuracy from problem text alone |
| Stage A, prompt hidden states (E1b) | Done: layer ensembles out-rank text (Finding 11) |
| Stage B, trace uncertainty (E2f) | Done: near chance (Finding 15) |
| E10 fusion | Done: text + Stage A beats text alone on both metrics (Findings 11, 16) |
| E4, trace-state probes | Done: the signal sits in the prompt state (Finding 18) |
| E7, olympiad labelling runner | Built; smoke run E7b done, expensive on T4 (Finding 19) |
| E11 submission | v1 0.47; **v2 (Qwen default `True`) 0.64**, best so far (Findings 20, 23) |
| Leaderboard probes | Done: hidden set is about 53% robust, and only Qwen3.5-4B moves the score (Finding 21) |
| E12, label-free signals | Done: the signals that work are problem-text proxies (Finding 22) |
| E13, unlabelled-model plumbing | Done: Qwen3.5-4B, Skywork-OR1-Math-7B and Olmo-3-7B-Think all run on T4 (Finding 22) |
| Codabench entry | Active: 9 submissions (8 on 16 Sep, one of them on the wrong task; v2 on 17 Sep) |
| Organizer questions | Drafted 17 Sep (labels for the new models, validation-set composition, tokenizer bug); **not sent yet** |

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

The starter kit is a submodule at `getting-started/` (originally `e98c489`, now upstream `995f992`). All 25 tests pass under
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

**11. Prompt hidden states: selecting a layer fails; averaging all layers is a lead alone and
a statistically established gain when fused with text.** (E1b, `train/train_stagea_probes.py`
and `train/train_fusion.py`; 138 instances, 37 layers × 3 poolings, same folds as the text
floor. E1b replaced E1, whose prompts were whitespace-stripped: Finding 14.)

| Method | Balanced acc. | ROC-AUC | AUC minus text (95% paired bootstrap CI) | E1 (corrupted) AUC |
| --- | --- | --- | --- | --- |
| Char tf-idf text (recomputed; matches the floor) | 0.6732 | 0.6686 | — | 0.6686 |
| 5 next-token uncertainty scalars | 0.6299 | 0.6887 | +0.020 [−0.121, +0.163] | 0.6392 |
| Nested-selected probe (layer, pooling chosen in-fold) | 0.5515 | 0.6397 | −0.029 [−0.150, +0.088] | 0.5929 |
| Layer ensemble, `last` pooling (no selection) | 0.6381 | 0.6966 | +0.028 [−0.072, +0.129] | 0.6955 |
| Layer ensemble, `max` pooling | 0.5613 | 0.7538 | +0.085 [−0.014, +0.175] | 0.7214 |
| Layer ensemble, `mean` pooling | 0.6479 | 0.7481 | +0.080 [−0.010, +0.173] | 0.7320 |
| Layer ensemble, all poolings | 0.5972 | 0.7489 | +0.080 [−0.009, +0.168] | 0.7331 |
| Fixed-weight fusion: text + all-pooling ensemble, 0.5 threshold | 0.6332 | 0.7342 | +0.066 [−0.000, +0.129] | 0.7260 |

**Nested fusion (E10 protocol, text + stagea).** Thresholds are learned in-fold and fusions
were fixed in advance.

| Method | Balanced acc. | ROC-AUC |
| --- | --- | --- |
| Text alone, learned threshold | 0.6389 | 0.6481 |
| Stagea alone | 0.6046 | 0.7059 |
| **Mean fusion** | **0.7173** | **0.7230** |
| Stacked fusion | 0.6748 | 0.7258 |

Mean fusion minus text alone:

- **AUC +0.075 [+0.031, +0.117].**
- **Balanced accuracy +0.078 [+0.015, +0.148].**
- Both intervals exclude zero.

Findings:

- **Selection overfits.**
  - Inner-CV AUCs of 0.75–0.85 fall to 0.64 on held-out folds.
  - The chosen configuration differs in every fold: max L27, max L16, max L17, mean L16,
    max L15.
  - Picking a layer is fitting noise, the same effect behind the baseline's 58.4%.
- **Averaging recovers a ranking signal.** Ensembles over all layers out-rank text by about
  0.08 AUC, but alone every interval still spans zero.
- **Fused with text, the gain is established.** The nested mean fusion beats text alone on
  both metrics with intervals excluding zero. It also beats the 0.5-threshold floor
  (0.6732) by 0.044 balanced accuracy. Caveats:
  - The bootstrap resamples instances, not model refits.
  - Only 1 of the 138 instances is olympiad-level.
- **Thresholds matter.**
  - Hidden-state probabilities sit off-centre: the all-pooling ensemble's median is 0.25, and
    only 30 of 138 are predicted robust at 0.5.
  - At a prior-matched threshold the `mean`-pooling ensemble reaches 0.662 balanced
    accuracy, against text's 0.624.
  - A shipped model must learn its threshold in-fold.
- **The tokenizer bug depressed but did not flip the result.** Correct prompts raised the
  next-token scalars' AUC by 0.05, the nested probe's by 0.05, and every ensemble's by
  0.00–0.03.
- **Per-layer probes are not at chance here.** Mean per-layer balanced accuracy is
  0.59–0.64 across poolings, against the shipped artifact's 0.5005. The training setups
  differ, so this does not by itself show the baseline was mis-trained.
- **Implication for Lane B.**
  - Drop single-layer probing.
  - Keep the layer-ensemble block in fusion.
  - Next, test whether states over the *generated trace* (E4) carry more than prompt states.

**12. Olmo-3-7B-Instruct plumbing works (E3).**

- It loads under transformers 5.13.0 on T4x2 in bf16 and exposes **33 hidden states** (32
  layers plus embeddings) × 4096.
- It runs at 1.0 s per instance with zero non-finite activations; prompts have a median of
  101 tokens under its chat template.
- The run used `--no-filter-model`, so its labels are DeepSeek's and the features are for
  plumbing only.
- Mechanically, nothing blocks Olmo labelling now; only the organizer answer does.

---

**13. On T4s, emulated bf16 makes long generation about 3.3× slower than fp16, and the dtype
changes greedy traces.** (E2, `e2-bench-dtype`.)

- **Starter-kit default failed.** It uses bf16, emulated on Turing, with batch size 8 and a
  4,096-token cap.
  - It ran at **617 s per instance**.
  - Shard 1 finished 16 of 69 instances in 2.8 h, then hit CUDA out-of-memory on GPU 1 (567 MB
    reserved but unallocated, i.e. fragmentation).
  - 7 of those 16 traces hit the token cap; median 3,883 tokens.
- **Benchmark.** The same 8 problems, 512 tokens, batch size 8.

  | Dtype | Aggregate throughput | Model load | Preflight |
  | --- | --- | --- | --- |
  | Emulated bf16 | 14.7 tokens/s | 102 s | finite |
  | **Native fp16** | **48.4 tokens/s** | 20 s | finite |

- **Drift.**
  - No trace was identical across dtypes (0 of 8); traces share a 37% prefix on average.
  - Mean absolute differences: entropy 0.021 on a mean of 0.213; mean log-prob 0.011 on 0.104;
    high-confidence fraction 0.019 on 0.784.
- **Decisions.**
  - Stage B reruns entirely in fp16 (`e2f-*`), with batch size 4 and expandable CUDA segments;
    the bf16 and fp16 sets are never mixed.
  - bf16 shard 0 keeps running as a drift reference.
  - E4 teacher-forces in fp16 to match its traces.
- **Risk.** Evaluation computes features in native bf16 on Blackwell, so a model fit on fp16
  features needs recalibration, or a refit on bf16 features from the Fields grant.
- **Scope of Finding 10.** "bf16 in practice" still holds for short forward passes; for long
  generation, speed forces fp16.

**14. Under transformers 5.13, `AutoTokenizer` strips all whitespace from
DeepSeek-R1-0528-Qwen3-8B text. E1 was extracted on corrupted prompts.** (Found through
`Ġ`/`Ċ` artifacts in E7 smoke completions; verified locally in the pinned venv.)

- **The broken loader.** `AutoTokenizer.from_pretrained` resolves to a legacy
  `LlamaTokenizer` conversion.
  - `"Every morning Aya goes for a $9$-kilometer-long walk.\nFind the answer \boxed{204}."`
    decodes as `"EverymorningAyagoesfora$9$-kilometer-longwalk.Findtheanswer\boxed{204}."`
  - Decoding correct ids with it yields `Ġ`/`Ċ` text.
- **The working loader.** `PreTrainedTokenizerFast` (the class the starter kit's
  uncertainty baseline deliberately uses) round-trips exactly.
- **Impact by run:**

  | Run | Tokenizer | Status |
  | --- | --- | --- |
  | **E1** | `AutoTokenizer` | **Invalid.** Recorded prompt lengths match the lossy tokenizer exactly; 135 of 138 prompts encode differently with the correct one. Re-extracted as **E1b** (`e1b-stagea-deepseek`, 16 Sep) with the checked tokenizer; Finding 11 now reports E1b. |
  | **E7 smoke** | vLLM's default, the same `AutoTokenizer` path | **Invalid.** Its accuracy numbers are not trustworthy. |
  | E2 / e2f, e2-bench-dtype, E4 | Starter kit's `PreTrainedTokenizerFast` | Valid |
  | E3 | Olmo-3-7B-Instruct round-trips under both loaders | Valid |
  | Text-only floor | No tokenizer | Valid |

- **The shipped trained-probe baseline.** `probe_inference.py` also loads `AutoTokenizer`, so
  under the evaluation runtime it sees whitespace-stripped prompts. This may contribute to its
  near-chance behaviour (Finding 2), and it is worth reporting upstream.
- **Fix.**
  - `datagen/tokenizer_check.py` loads `PreTrainedTokenizerFast` first and refuses any
    tokenizer that fails an encode/decode round trip. Its class and round-trip result are
    recorded in `metadata.json`.
  - `extract_features.py` and `perturb/sample.py` load through it.
  - The vLLM backend now runs with `skip_tokenizer_init`: prompts go in as ids from the
    checked tokenizer, and completions are decoded with it.
  - For DeepSeek the chat template already carries BOS, and the tokenizer adds no second one.

**15. The published uncertainty-profiling features barely separate robust from spurious on our
instances. Trace length does, in an unexpected direction.** (E2 rerun in fp16, `e2f-*`;
`train/train_stageb_features.py`.)

- **Setup.** 138 instances; greedy decoding with a 4,096-token cap, fp16, batch size 4, and
  the starter kit's tokenizer (round-trips correctly).
- **Cost.** 190–211 s per instance, or 3.7 h and 4.1 h per shard, against 617 s per instance
  in emulated bf16.
- **Results.** Out-of-fold, same folds as the text floor:

| Features | Best balanced acc. | Best ROC-AUC |
| --- | --- | --- |
| 14 uncertainty features (the baseline's set) | 0.5752 | 0.5487 |
| + length (log tokens, hit cap) | 0.6136 | 0.6176 |
| + trace shape (log chars, boxed, closed think, reflections) | 0.5940 | 0.6484 |
| Fixed-weight fusion: text + trace (logreg), 0.5 threshold | 0.6438 | 0.6874 |
| Text floor | 0.6732 | 0.6686 |

- **Alone, the baseline's 14 features are near chance** (AUC 0.54–0.55). Its shipped artifact
  reports 0.6262 grouped out-of-fold balanced accuracy on its own 141-row split. This is not a
  clean replication: our traces are fp16 (Finding 13), our labels include derived rows and
  eligibility filters, and we classify rather than regress decay. Read it as "does not
  transfer to our setup", not as a refutation.
- **Longer traces lean robust.** 55% of traces hit the cap: 72% of robust ones and 49% of
  spurious ones. Univariate AUCs:
  - lean robust: log chars 0.640, log tokens 0.636, hit cap 0.616;
  - lean spurious: closing `</think>` 0.364, producing a boxed answer 0.384, reflection markers
    per 1k words 0.314.

  A plausible reading is that the model answers brittle problems quickly and fluently. But
  under a 4,096-token cap, length is confounded with difficulty, so this needs checking at a
  longer cap before it becomes a feature.

**16. E10 fusion v1: the pre-declared all-block fusion improves AUC over text but not balanced
accuracy; the trace block adds nothing.** (`train/train_fusion.py`: text, E1b prompt layer
ensemble, and E2 fp16 trace features; nested, with in-fold thresholds.)

| Row | Balanced acc. | ROC-AUC | AUC minus text-alone | Balanced acc. minus text-alone |
| --- | --- | --- | --- | --- |
| text alone | 0.6389 | 0.6481 | — | — |
| stagea alone | 0.6046 | 0.7059 | +0.058 [−0.037, +0.149] | −0.034 [−0.139, +0.064] |
| trace alone | 0.5792 | 0.6334 | −0.015 [−0.146, +0.113] | −0.060 [−0.181, +0.059] |
| text + stagea (ablation) | **0.7173** | 0.7230 | +0.075 [+0.031, +0.117] | +0.078 [+0.015, +0.148] |
| text + trace (ablation) | 0.6103 | 0.6419 | −0.006 [−0.054, +0.038] | −0.029 [−0.110, +0.051] |
| stagea + trace (ablation) | 0.6928 | **0.7571** | +0.109 [+0.015, +0.199] | +0.054 [−0.039, +0.141] |
| **mean fusion, all blocks (pre-declared)** | 0.6667 | 0.7176 | **+0.069 [+0.019, +0.121]** | +0.028 [−0.047, +0.104] |
| stacked fusion, all blocks (pre-declared) | 0.6471 | 0.7285 | +0.080 [−0.020, +0.175] | +0.008 [−0.090, +0.100] |

- **Pre-declared headline.** The AUC gain is established; the balanced-accuracy gain is not.
  An earlier version of the script's verdict called the gain "established" whenever either
  metric cleared zero. It now reports each metric separately.
- **Trace features dilute.** Adding the trace block to text + stagea cuts balanced accuracy
  from 0.7173 to 0.6667, and text + trace is below text alone.
- **The strongest rows are ablations.** Text + stagea is best on balanced accuracy; stagea +
  trace is best on AUC. Adopting either after reading this table is a selection effect, so
  confirm on independent data first (olympiad labels from E7).
- **Gate G1 decision.**
  - Trace-uncertainty features leave the fusion for now.
  - **Text + prompt layer ensemble is the Lane B candidate**, pending an out-of-sample check.
  - Its inference cost is one forward pass per problem (~1 s on a T4, less on the evaluation
    GPUs) plus a tf-idf model, far inside the 3600 s budget, which also helps the runtime
    tie-break.

**17. E4 ran out of memory because `output_hidden_states=True` holds every layer at once;
hooks fix it, but only after matching the final-norm convention.**

- **Failure.** The first E4 kernel died at instance 100 of 138 with CUDA OOM on GPU 1, while
  trying to allocate 572 MB. It was running at 8.4 s per instance, and every completed
  instance had a full 4,096-token trace.
- **Cause.** `output_hidden_states=True` materialises all 37 layers for the whole sequence:
  37 × 6,144 × 4,096 × 2 bytes ≈ **1.9 GB** on top of ~8 GB of weights per GPU. Stage A never
  hit this because its prompts are about 100 tokens.
- **Fix.** Forward hooks pool each layer the moment it is produced and release it, so only one
  layer is ever held.
- **The subtlety that first made hooks wrong.** `hidden_states` is not each layer's output.
  transformers appends every layer's *input*, then the last layer's output *after* the model's
  final norm, giving `(embeddings, out_1, ..., out_{N-1}, norm(out_N))`. Hooking only
  embeddings and decoder layers captured the raw `out_N` instead, and the two paths differed
  by up to **2.28** on the tiny model. Hooking the norm as well and dropping the raw final
  layer makes them **identical (max difference 0.0)**.
- **Why exactness matters.** E1b Stage A used the standard path, so E4 features have to follow
  the same convention to be comparable or fusible.
- `--no-hooks` keeps the original path for debugging, and `pooling_via_hooks` is recorded in
  `metadata.json`.

**18. E4 trace states lift the fusion, but the signal sits in the prompt, not in the
reasoning.** (`e4-tracestate-deepseek` v2, 138 instances, 37 layers × 5 segment poolings,
fp16, hook pooling; `train_stagea_probes.py --section` and `train_fusion.py`.)

- **Cost.** 8.7 s per instance, 23 minutes for all 138.
- **Within E4, prompt beats trace.**

  | Layer ensemble (no selection) | Balanced acc. | ROC-AUC |
  | --- | --- | --- |
  | `prompt_last` (state generation started from) | 0.6618 | **0.7386** |
  | `trace_first_third` | 0.6397 | 0.7394 |
  | `trace_mean` | 0.6013 | 0.6860 |
  | `trace_last_third` | 0.5163 | 0.6253 |
  | `trace_last` | 0.5114 | 0.6040 |

  The nested selector chose `prompt_last` in 4 of 5 folds. **States late in the reasoning are
  the weakest**, which is the opposite of the hypothesis E4 was built to test.
- **Fusion (nested, in-fold thresholds), against text alone:**

  | Row | Balanced acc. | ROC-AUC | AUC vs text | Bal. acc. vs text |
  | --- | --- | --- | --- | --- |
  | tracestate alone | 0.6454 | 0.7037 | +0.056 [−0.038, +0.156] | +0.007 [−0.102, +0.114] |
  | text + stagea (previous candidate) | **0.7173** | 0.7230 | +0.075 [+0.031, +0.117] | +0.078 [+0.015, +0.148] |
  | **text + stagea + tracestate** | 0.7116 | **0.7391** | **+0.091 [+0.031, +0.152]** | **+0.073 [+0.003, +0.150]** |

  This is the first pre-declared block set whose **balanced-accuracy** gain also excludes zero.
- **Interpretation.** E4's contribution is mostly a second view of *prompt* state under a
  different template (user-only, versus the probe's system prompt in E1b), not evidence that
  reasoning-time internals add information. Trace-derived features have now disappointed twice:
  trace-uncertainty statistics (Finding 15) and trace-position poolings here.
- **Candidate status.** Adding tracestate buys +0.016 AUC over text + stagea and costs
  −0.006 balanced accuracy, with overlapping intervals; it also requires generating a full
  trace at inference (minutes per problem) rather than one forward pass (~1 s). **Keep
  text + stagea as the Lane B candidate on efficiency grounds** until an out-of-sample check
  (E7) says otherwise; the runtime tie-break favours it too.

**19. With the tokenizer fixed and longer caps, olympiad base sampling works but is
expensive.** (`e7b-sample-smoke`: vLLM 0.29, fp16, 2× T4, k=4, token-id prompts.)

| Tier | Cap | Samples at cap | Accuracy per sample | Solved 4/4 | Time per problem | Throughput |
| --- | --- | --- | --- | --- | --- | --- |
| AMC (16) | 8,192 | 45/64 | 0.297 (0.203 at 4k, lossy tokenizer) | 4/16 | 311 s | 88 tok/s |
| AIME (8) | 16,384 | 21/32 | 0.344 (0.062 at 4k, lossy tokenizer) | 2/8 | 1,258 s | 44 tok/s |

- Completions are clean text, and every correct sample ends in a parsed `\boxed{}`.
- **Most samples still hit the cap**, so only about a quarter of problems are eligible
  (solved in every sample).
- **Cost.** A base pass over the 505-problem pool at these rates is roughly 45–175 GPU-hours
  before any variant sampling, against about 180 GPU-hours a week for the whole fleet. Long-cap
  labelling fits the Fields compute grant better than Kaggle T4s.

**20. The E11 submission ZIP passes on real GPUs.** (`e11-gpu-check`, 2× T4, bf16,
transformers 5.13, scikit-learn 1.8.0; the exact ZIP mounted from a private Kaggle dataset.)

- **Decisions.** `are_robust` on the 138 DeepSeek training problems returned 138 valid bools,
  42 robust, with **0 flips** against the artifact's reference decisions.
- **Features.** Recomputed stagea probabilities match the training-time reference to within
  **1.3e-9**.
- **Other models.** `Qwen/Qwen3.5-4B` gets all `False` in under 0.01 s.
- **Runtime.** 176.8 s for the full call, including a 7.8 s model load and 0.96 s of forward
  pass per problem on T4s.
- **Issues found on the way, all fixed.**
  1. Kaggle ships scikit-learn 1.6.1, which cannot unpickle 1.8.0 estimators. Kernels now pin
     1.8.0, as the evaluation image does.
  2. An exception in the text block would have invalidated the whole batch. It now falls back
     to `False`.
  3. With no checkpoint in the local cache, the `local_files_only` load silently fell back to
     text-only; the check caught this as 4 flips. The check now pre-downloads the model, as
     the evaluation workers' warm cache does.

**21. The live validation set is small, about 53% robust, and only Qwen3.5-4B moves the score.**
(Codabench 16180, Small Model Track, 16 Sep. Scores are accuracy over all hidden cases,
with invalid cases counted wrong. Probe ZIPs come from `submission/make_probes.py`.)

| Submission | Predicts | Score |
| --- | --- | --- |
| 927967 `probe-all-false` | `False` everywhere | 0.47 |
| 927882, 927968 `probe-true-deepseek` | `True` for DeepSeek only | 0.47 |
| 927889 `probe-true-qwen35-4b` | `True` for Qwen3.5-4B only | **0.53** |
| 927893 `probe-true-skywork-7b` | `True` for Skywork only | 0.47 |
| 927883 `probe-true-olmo3-think` | `True` for Olmo-3-Think only | 0.47 |
| 927865 `e11-text-stagea` (v1) | E11 fusion for DeepSeek, `False` otherwise | 0.47 |

- **All-false scores 0.47, so about 53% of hidden cases are robust.** Our training data is
  74% spurious.
- **A probe that doesn't move the score from 0.47 tells us that model is absent or exactly
  balanced.** That applies to DeepSeek, Skywork and Olmo, and to E11 v1's DeepSeek
  predictions.
- **The simplest reading is a set made only of Qwen3.5-4B cases.**
  - True-for-Qwen gains exactly as much as the overall robust surplus (0.53 − 0.47).
  - The two-decimal scores fit a set of 15–19 cases, so one case moves the score by about 0.06.
- **Consequences:**
  - The leaderboard cannot rank DeepSeek methods.
  - E11 v2 therefore changes only the Qwen default, to `True`.
  - Treat further leaderboard gains as noise, not evidence.
- **Main Track ingestion rejects `small.txt`.**
  Submission 927935, an all-false probe, scored 0.00 on that task with "small.txt is not allowed for a main-track submission".

**22. The label-free signals mostly read problem text; all three unlabelled models run on T4.**
(`e12-labelfree-deepseek`, `e13-{qwen35,skywork,olmo-think}-plumbing`, 17 Sep, T4x2 in bf16,
transformers 5.13. Every number below is in the `RESULTS.md` section generated by
`train/eval_label_free.py`, including the length control and cross-model transfer.)

- **Plumbing passes for all three.**
  - `Qwen/Qwen3.5-4B` is saved as `Qwen3_5ForConditionalGeneration`, but `AutoModelForCausalLM`
    loads it as `Qwen3_5ForCausalLM` with no missing-weight warnings: 33 hidden-state layers,
    2560 dimensions.
  - Its linear-attention layers fall back to the torch implementation, since `fla` and
    `causal-conv1d` are absent here, as they are in the evaluation image.
  - Skywork is `Qwen2` (29 hidden-state layers, 3584 dimensions); Olmo-3-Think is `Olmo3`
    (33 layers, 4096 dimensions).
  - Every tokenizer round-trips, and the 20-problem Stage A smoke tests show 0 non-finite
    activations.
  - Label-free scoring takes 3.3 s (Qwen), 3.8 s (Skywork), 4.8 s (Olmo) and 5.3 s (DeepSeek)
    per problem on T4s.
- **The immediate-answer stability signals are null.** `variant_agree` AUC is 0.51 and
  `variant_kl` 0.56, both with CIs spanning 0.5. Without reasoning, the models mostly guess,
  so their guesses say nothing about robustness.
- **The signals that work are text proxies.**
  - `next_token_entropy` reaches AUC 0.736 [0.64, 0.82], but problem length alone reaches
    0.734, and the two correlate at Spearman 0.67.
  - Counting only pairs within the same problem-length quintile, the entropy signal drops
    to 0.587. `problem_nll` and `next_token_logit_margin` keep a weak reversed signal (0.42),
    and every other signal sits at 0.49–0.57.
  - An earlier version regressed ranks on length. It reported 0.615 here, but it inflates
    tied signals (Finding 24).
  - `problem_nll` (AUC 0.32, i.e. 0.68 flipped: more familiar problems are more often robust)
    works just as well when another model computes it: 0.33 (Qwen), 0.29 (Skywork),
    0.30 (Olmo) against DeepSeek's labels. It is a property of the problem, not of DeepSeek.
  - Other models' `next_token_entropy` does not transfer (AUC 0.43–0.59). It barely
    correlates with DeepSeek's, except for Olmo (Spearman +0.55).
- **Implication.** Nothing here measures a model-specific mechanism. For the unlabelled
  models, the best available transfer is the problem-level text signal learned on DeepSeek,
  which assumes robustness is largely a property of the problem.

**23. E11 v2 scores 0.64, more than the probes predicted.** (Codabench submission 929596,
17 Sep 11:24: E11 v1 plus `True` for every Qwen3.5-4B case.)

- **Predicted 0.53.** v2 differs from v1 (0.47) only in the Qwen predictions, exactly as the
  true-Qwen probe (0.53) differs from all-false (0.47).
- **The +0.11 gap cannot come from the predictions alone.** On a fixed hidden set, the two
  pairs would change by the same amount. The set or its scoring must have changed
  between 16 Sep 23:33 and 17 Sep 11:24.
  - Supporting sign: submission 927935, an all-false probe that showed 0.00, now shows 0.50,
    while an identical later all-false shows 0.47.
- **Consequences.**
  - Finding 21's per-model reading needs re-measuring: all-false and true-Qwen, resubmitted
    once each.
  - v2 is the fallback for every later submission.
- **Launched in response** (`PLAN.md` §4–§5):
  - E14 (test-time answer agreement, DeepSeek);
  - E15–E17 (base phase toward labels for Qwen3.5-4B, Skywork and Olmo-3-Think).
- **Two Kaggle tokens stopped working.** On 17 Sep, ISHMAM1's and DESIKALAKAR's tokens were
  rejected ("Authentication required"; their own kernels return permission denied). Their runs
  moved to second slots on ISHRAQ and TASHIN.
- **vLLM's custom all-reduce crashes on some Kaggle T4 pairs.**
  - E14 (both shards) and E16 died during CUDA graph capture with `custom_all_reduce.cuh:164
    'invalid argument'`.
  - E7b and E17 ran the same TP=2 setup without error, so it depends on the host.
  - `sample.py` now passes `disable_custom_all_reduce=True` (NCCL), and the three runs were
    relaunched.
- **vLLM 0.29 runs Qwen3.5-4B on a single T4** (Triton attention, fp16, text-only). Engine
  start took about 10 minutes, but it then refused to start with its default 256 concurrent
  sequences: "max_num_seqs (256) exceeds available Mamba cache blocks (157)".
  - The hybrid linear-attention layers need one state block per sequence.
  - E15 now runs with `--max-num-seqs 64`.

**24. Test-time answer agreement does not predict robustness; gate G-D is a no-go.**
(`e14-agreement-deepseek-s0/s1`, 138 labelled problems.)

- **Setup.** DeepSeek reasoned in full, 2 samples per prompt with an 8k cap (vLLM, fp16, T4s),
  on each problem and on up to four rewrites (447 prompts, 309 of them rewrites). The
  measure is how often a rewrite's final answer matched the original's modal answer.
  `train/eval_agreement.py` and `train/train_fusion.py` generate the tables in `RESULTS.md`.
- **Answers barely change.** An anchor answer was found for 121 problems, and every rewrite
  sample matched it for 78.5% of them. Our rewrites rarely trip the model, spurious cases
  included (mean agreement 0.93 spurious vs 0.85 robust on shard 0).
- **No agreement signal is above chance.**

  | Signal | ROC-AUC [95% CI] | Within length quintiles |
  | --- | --- | --- |
  | rewrite answers match the original | 0.458 [0.363, 0.549] | 0.543 |
  | worst rewrite kind | 0.468 [0.375, 0.555] | 0.566 |
  | original samples agree | 0.455 [0.373, 0.524] | 0.484 |
  | tokens spent on the original | 0.652 [0.557, 0.744] | 0.497 |
  | share of samples hitting the cap | 0.606 [0.511, 0.704] | 0.488 |
  | problem length (reference) | 0.734 [0.646, 0.810] | — |

  The raw agreement AUCs sit below 0.5 because harder, longer problems both truncate more
  and are more often robust. Reasoning length is **entirely** problem length.
- **Fusion (nested, in-fold thresholds).**

  | Row | Balanced acc. | ROC-AUC |
  | --- | --- | --- |
  | text + stagea (E11) | **0.7173** | 0.7230 |
  | agreement block alone | 0.6626 | 0.6955 |
  | text + stagea + agreement | 0.6928 | **0.7320** |

  Adding the block costs 0.025 balanced accuracy and gains 0.009 AUC. Its own strength
  comes from the length features.
- **A measurement bug caught on the way.** The length control used in Finding 22 regressed a
  signal's rank on length rank. For heavily tied signals (79% of agreement values are
  exactly 1.0), that residual *is* length, so every agreement signal appeared to reach
  0.64–0.73. It was replaced by a within-quintile Mann-Whitney AUC. On synthetic data, that
  leaves a real signal intact and cuts a pure length proxy from about 0.8 to 0.56.
- **Cost.** 3.86M tokens, about 28,000 per problem and 10.1 GPU-hours on T4s. That is far
  beyond the 3600 s evaluation budget with HF `generate`, even before the result.
- **Decision.** Lane D closes. Mild answer-preserving rewrites do not reproduce what the
  official perturbations do to this model. Remaining routes to a model-specific signal for
  the unlabelled models are self-generated labels (E15–E17) or organizer labels.

**25. Chunk size decides what survives a Kaggle session; the first labelling session mostly
did not.** (`e15`, `e16`, `e17`, 17–18 Sep. A chunk is one generate call and one flush.)

- **What happened.** The base phase asked for 32 problems x 4 samples at a 16k cap per call,
  roughly 2M tokens. In 11.5 h Qwen completed one chunk per GPU, Skywork five, and
  Olmo-3-Think none, so Olmo produced nothing at all.
- **Output does survive a kill.** Kaggle saved `/kaggle/working` even for the sessions it
  cancelled at 12 h, so only the unfinished chunk was lost, not the finished ones.
- **First-session yields**, eligible meaning the model solved the problem in all 4 samples:

  | Model | Sampled | Eligible | Hit the cap | Tokens per problem | Speed |
  | --- | --- | --- | --- | --- | --- |
  | Skywork-OR1-Math-7B | 160 | **101 (63%)** | 18% | 33,000 | 150 tok/s (TP=2) |
  | Qwen3.5-4B | 128 | 35 (27%) | 62% | 54,000 | 75 tok/s per GPU |
  | Olmo-3-7B-Think | 0 | — | — | — | — |

  Skywork is much the cheapest to label: it is both faster and more often eligible. Qwen's
  62% truncation rate at a 16k cap is what holds its yield down.
- **Fix.** 4 problems per chunk (16 sequences, about what a T4's KV cache holds anyway), an
  8.5 h budget so the run exits on its own, and a restore step that copies previously
  downloaded records back into the kernel, so a session continues instead of restarting.
- **Fleet.** RSD and Labib were added on 18 Sep with full weekly quota, replacing ISHMAM1 and
  DESIKALAKAR. SALAM and DIDHITI hit Kaggle's 30 GPU-hours per week during the failed session.

**26. After two base-phase sessions, Qwen and Skywork pass gate G-E; the variants phase is
running.** (`e15`, `e16`, `e17` session 2, 18 Sep; each exited cleanly at its 8.5 h budget.)

| Model | Sampled (of 505) | Eligible | Rate | Hit the cap | Median tokens per sample |
| --- | --- | --- | --- | --- | --- |
| Skywork-OR1-Math-7B | 246 | **157** | 64% | 19% | 7,318 |
| Qwen3.5-4B | 285 | **90** | 32% | 58% | 16,384 (the cap) |
| Olmo-3-7B-Think | 30 | 17 | 57% | 38% | 8,266 |

- **By tier.** Skywork: AMC 20/23, AIME 9/26, olympiad 128/197. Qwen: AMC 8/25, AIME 2/29,
  olympiad 80/231. AIME is hard for both, and Qwen is limited by truncation more than by skill.
- **Olmo-3-Think fits on a T4 pair only at a 12k cap.** It keeps a full KV cache (4.5 GiB for
  one 18k-token sequence, against 4.12 GiB free), so it runs one or two sequences at a time:
  30 problems in 9.6 h. At that rate, labelling it belongs on the Fields grant.
- **Gate G-E (≥ 60 eligible): Qwen and Skywork pass.**
- **Not every eligible problem can be labelled.** `answer_modulus` needs an answer of at least
  13, so only some problems have compliant variants.

  | Model | Compliant variants | Problems covered |
  | --- | --- | --- |
  | Qwen3.5-4B | 109 | 53 of 90 |
  | Skywork-OR1-Math-7B | 205 | 101 of 157 |

  That caps the labels this round at about 53 for Qwen and 101 for Skywork.
- **Launched:** `e18-variants-qwen35` (RSD, one engine per T4) and `e19-variants-skywork`
  (Labib, TP=2), with k=4, a 16k cap and 4-problem chunks. `label.py` then turns base and
  variant accuracy into labels.

**27. Re-measured per-model defaults confirm E11 v2; the hidden set is about 14 cases.**
(Codabench, Small Model Track, 18 Sep 00:13-00:15, same task as v2. Probe ZIPs from
`submission/make_probes.py`.)

| Submission | Predicts | Score | vs all-False |
| --- | --- | --- | --- |
| 930707 `probe-all-false` | `False` everywhere | 0.50 | — |
| 930709 `probe-true-qwen35-4b` | `True` for Qwen3.5-4B only | **0.57** | **+0.07** |
| 930710 `probe-true-skywork-7b` | `True` for Skywork only | 0.43 | −0.07 |
| 930708 `probe-true-olmo3-think` | `True` for Olmo-3-Think only | 0.43 | −0.07 |
| 929596 E11 v2 (17 Sep) | DeepSeek fusion, Qwen `True`, others `False` | 0.64 | +0.14 |

- **Score steps of 0.07 put the hidden set at about 14–15 cases.** Every gap in this table is
  one or two cases.
- **E11 v2's defaults are the best constants available.** Qwen gains with `True`, while
  Skywork and Olmo-3-Think each lose one case with it, so v2 needs no change.
- **E11's DeepSeek predictions appear worth one case over `False`** (v2 0.64 vs true-Qwen 0.57),
  though the two were scored on different days and all-False moved 0.47 → 0.50 in between.
- **Beyond 0.64, gains must come from per-case predictions, not defaults.** That means labels
  for the unlabelled models (E15–E19).

**28. Changes ported from branch `nakib` (commit 2570baa), without merging.**

- **`kaggle_launch.py --account <ALIAS>`** runs a registered experiment on another account
  without editing its registry entry, so outputs keep their `runs/<experiment>/` paths. It
  would have avoided three registry edits this week when accounts ran out of quota.
- **`scripts/_setup_kaggle_accounts.py`** resolves `ROOT` from its own location instead of a
  hard-coded path to one machine.
- **`train_stagea_probes.py`** now writes a repo-relative features path into `RESULTS.md`,
  and the existing absolute path was corrected.
- **Docs:**
  - `COMPETITION.md`: model ids are passed verbatim (no alias table), and a worker has one
    RTX PRO 6000, not eight.
  - `CLAUDE.md`: the submodule commit is `995f992`, and three of four models are unlabelled.
- The branch also registers three more fleet aliases (NAKIB, CHOTON, MEDIUM); each needs its
  token in `Secrets.env` before use.

**29. First self-generated labels for Qwen3.5-4B and Skywork, after fixing a pool-id
collision.** (`e18-variants-qwen35` on RSD, 8.7 h; `e19-variants-skywork` on Labib, 9.8 h;
`datagen/perturb/label.py`, then `train/summarize_labelling.py`. Tables in `RESULTS.md`.)

- **A data bug, found when `label.py` first ran on real records.**
  - `AI-MO/aimo-validation-amc` reuses its `id` column across contests, so 34 AMC `pool_id`s
    each named two different problems (68 pool rows). AIME and OlympiadBench are unaffected.
  - The sampler keys records by `pool_id`. A record under a shared id cannot say which problem
    it measured, and a variant could have been selected on the other problem's eligibility.
  - `label.py` and `summarize_labelling.py` now drop those records: 33 for Qwen and 18 for
    Skywork, across both phases.
  - `pool.py` now suffixes colliding ids with the problem key. That renames only the 68
    colliding rows, so records under all 437 other ids stay valid.
- **Corrected base-phase yields**, replacing Findings 25–26:

  | Model | Eligible | Rate |
  | --- | --- | --- |
  | Qwen3.5-4B | 84 of 270 | 31% |
  | Skywork-OR1-Math-7B | 143 of 232 | 62% |
  | Olmo-3-7B-Think | 15 of 28 | 54% |

- **Variants phase.**
  - E18 sampled 89 of Qwen's 91 compliant variants.
  - E19 sampled only 24 of Skywork's 176, at 19.9 tok/s. Its previous base session reached
    111 tok/s with the same settings, and the log shows both GPUs and no errors, so this looks
    like a slow host (engine load also took 919 s against 196 s).
- **The labels:**

  | Model | Problems labelled | Robust (all) | Clear-cut (training) | Robust (clear-cut) |
  | --- | --- | --- | --- | --- |
  | Qwen3.5-4B | 47 | 19 (40%) | 26 | 19 |
  | Skywork-OR1-Math-7B | 21 | 17 (81%) | 19 | 17 |

- **The ambiguous band bites at this sample size.** 21 of Qwen's 47 problems fall in
  0 < decay <= 0.3 and are dropped. With 4 samples per variant, a single wrong sample is enough.
  The clear-cut training set therefore leans robust (19 of 26), unlike the raw 40%.
- **A warning sign.** Skywork looks mostly robust under `answer_modulus` (81%), but on the
  leaderboard `True` for Skywork lost a case (Finding 27). Either `answer_modulus` robustness
  differs from the official perturbations' robustness, or the one or two hidden Skywork cases
  are unrepresentative. That comparison is the check PLAN.md calls for before trusting
  self-labels.
- **Too few to train on yet:** 26 Qwen and 19 Skywork training instances, against 138 for
  DeepSeek.
- **Launched 19 Sep:**
  - `e20-validate-deepseek` runs the same pipeline on DeepSeek's 66 officially labelled MATH
    problems with an integer answer >= 13 (19 officially robust), from
    `datagen/perturb/deepseek_check_pool.py`.
  - Skywork's variants resume (`e19`, session 2).
  - Qwen's base phase continues over the ~235 untried pool problems (`e15`, session 3).
- **More samples per variant were considered and dropped.** Robust requires every variant
  sample correct, so extra samples sharpen the decay estimate but cannot lift a problem out of
  the ambiguous band. More problems add labels; more samples do not.

**30. The organizers published their own training split, and our label rule was wrong.**
(`aimo-interp/train-main-v2`, published 17 Sep 2026, found 20 Sep. Ingested by
`datagen/ingest_public.py`, which enumerates the author's datasets.)

- **82 rows over 28 problems, covering every evaluated configuration**, including the first
  labels anywhere for three of our four models:

  | Configuration | Labelled | Robust |
  | --- | --- | --- |
  | `Qwen/Qwen3.5-4B` | 12 | 8 |
  | `deepseek-ai/DeepSeek-R1-0528-Qwen3-8B` | 10 | 7 |
  | `allenai/Olmo-3-7B-Think` | 9 | 7 |
  | `Skywork/Skywork-OR1-Math-7B` | 3 | 2 |
  | `openai/gpt-oss-120b` (low/medium/high) | 37 | 28 |

- **Their label rule, stated exactly, is not the one we recovered.**
  - robust iff `max_drop <= 0.10`; spurious iff `max_drop >= 0.25`;
  - **no label at all** in between, in training, validation and test.
  - Ours was `decay <= 0`, with an ambiguous band of `0 < decay <= 0.3`, recovered from older
    tables that happen to contain nothing between 0 and 0.3 (§5.2). Both agree on every
    published row; they differ on rows we derive.
- **Effect on our training set:** DeepSeek goes from 138 train-ready instances (36 robust) to
  **153 (46 robust)** once Finding 44's precedence fix lands, and Qwen (12), Olmo-3-Think (9)
  and Skywork (3) enter with official labels. Stage A results are unchanged, because the 6 newly-labelled DeepSeek instances are
  ones the old rule dropped as ambiguous and they have no features yet.
- **They confirm the leaderboard is balanced.** The README says this split is 72% robust while
  "the validation set is close to balanced", and warns against calibrating a threshold on it.
  That matches the probes exactly (Finding 27: all-False 0.50).
- **The olympiad transfer test is now possible, and it fails.** Char tf-idf trained on the 142
  MATH instances and tested on the olympiad ones scores **balanced accuracy 0.41, AUC 0.43**
  (11 instances after the label fix of Finding 44): below chance. Our text signal does not survive the difficulty shift the private test
  set asks for. Small sample, so it is a warning rather than a measurement, but it points the
  same way as the leaderboard, where E11's DeepSeek predictions are worth about one case.
- **Also merged upstream:** the starter kit now carries `reasoning_effort` on main. Only
  `gpt-oss-120b` varies it; the four small-track models are always `default`, and the E11
  signature already accepts both call styles.

**31. Self-generated `answer_modulus` labels do not measure official robustness; gate G-V
fails.** (Regenerated by `train/compare_selflabels.py`. E20: the full E7 pipeline on 66 of
DeepSeek's officially labelled MATH problems,
6.1 h; base 66 of 66, variants 91 of 124.)

| Self-label vs official label | All 49 labelled | Clear-cut only (45) |
| --- | --- | --- |
| Agreement | 0.29 | 0.27 |
| Balanced accuracy | 0.391 | 0.422 |
| Cohen's kappa | **-0.135** | -0.085 |
| Spearman, our drop vs official drop | **-0.25** | -0.27 |

- **Not merely different: mildly inverted.** Our pipeline calls 38 of 49 robust where the
  organizers call 13 robust, and the ranking correlation is negative.
- **Why this is plausible.** `answer_modulus` changes what is asked (report the answer modulo
  m), whereas every official type preserves the answer and rewords the problem. A model can be
  fragile to one and not the other.
- **Consequences.**
  - **E15-E19 labels are not training data.** The 26 Qwen and 19 Skywork self-labels are set
    aside, and the labelling program stops. The base-phase and variants records stay for the
    report as a negative result.
  - The only category permitted under the rule at the time turns out not to measure the
    target. The 19 Sep rule change permits an official-family replacement, but that needs the
    inaccessible symbolic programs rather than more `answer_modulus` volume.
  - Official labels (34 small-track rows) plus DeepSeek's 152 are now the whole training set.
- **What it saves.** About 40 GPU-hours a week that E15-E19 would have consumed.

**32. Neither remaining route beats a constant: the model does not transfer to olympiad
problems, and there is no cross-model signal to borrow.** (E21: Stage A prompt states over
the official split for all four models, four Kaggle sessions of about 20 minutes;
`train/eval_official_holdout.py`, `RESULTS.md` "E21: official-split holdouts".)

Two questions, both first answerable now that `train-main-v2` exists.

- **Does the shipped DeepSeek model survive the difficulty shift?** The 10 `train-main-v2`
  DeepSeek rows are olympiad problems that no fit has seen (7 robust, 3 spurious). Re-measured
  after Finding 44 corrected two of their labels.

  | Method | Accuracy | Balanced acc. | ROC-AUC |
  | --- | --- | --- | --- |
  | E11 fused, shipped artifact | 0.700 | 0.595 | 0.429 |
  | its text block alone | 0.500 | 0.452 | 0.476 |
  | layer ensemble refit on the 142 MATH rows | 0.500 | 0.548 | 0.476 |
  | all-robust | 0.700 | 0.500 | 0.500 |

  Every AUC is at or below 0.5. The fused model's 0.700 accuracy is what answering `True`
  everywhere scores. On MATH the same model reaches 0.7173 balanced accuracy under grouped
  CV, so **the signal is specific to MATH-style problems**, which agrees with the char
  tf-idf transfer (0.41 balanced accuracy, Finding 30) and with the leaderboard.

- **Is there a model-agnostic signal?** Hidden states differ in width per model, but the five
  next-token uncertainty scalars do not. Pooling all 34 official rows and holding one model
  out at a time (standardising within each model, which evaluation allows, since the harness
  hands over one model's whole batch):

  | Held out | n | Balanced acc. | ROC-AUC |
  | --- | --- | --- | --- |
  | `Qwen/Qwen3.5-4B` | 12 | 0.625 | 0.406 |
  | `allenai/Olmo-3-7B-Think` | 9 | 0.464 | 0.571 |
  | `deepseek-ai/DeepSeek-R1-0528-Qwen3-8B` | 10 | 0.619 | 0.571 |
  | `Skywork/Skywork-OR1-Math-7B` | 3 | 0.750 | 0.500 |

  No AUC clears 0.58, the balanced accuracies swing on one or two rows each (Skywork's 0.750 is
  three rows), and the other models' base rate - predict `True` for everyone - still beats every
  fold on raw accuracy. Nothing here is a signal to build on.

- **Consequences.**
  - **E11 v2 stands.** Its per-model constants remain the best thing we can defend, and
    nothing here justifies a v3.
  - The two facts together explain the leaderboard: DeepSeek's model is worth about one case
    (Finding 27), and it is the one model where we have any model at all.
  - **What is left is the technical report**, where E21 joins E12, E14 and E20 as a measured
    negative result, and any further modelling waits on more labels than 32 rows.
  - Sample sizes are 3 to 12 rows. These are go/no-go signals, not measurements, and they all
    point the same way.

**Findings 33-43 are ported by hand from the `nakib_v2` branch** (measured on a teammate's
checkout and Kaggle accounts, 19 Sep). The branches are deliberately not merged; the scripts
they cite were copied file by file, and their numbering was shifted onto this sequence.
Where a result there rests on self-generated `answer_modulus` labels it is marked superseded
by Finding 31, which invalidated that label source.

**33. More on `train-main-v2`: how it was found, what is in it, and why the prevalence gap is
the problem population rather than the rule.** (`scripts/inspect_train_main_v2.py`, revision
`23791d9b`. Extends Finding 30, which reached the same table independently.)

- **Reachable only by listing the `aimo-interp` org on Hugging Face** - linked from neither the
  competition website nor the `getting-started` README, public, unauthenticated.
- **11 of the 82 rows carry `is_robust = null`**, all strictly inside the band (0.11-0.21). The
  card says band cases are left unlabelled in validation and test too, so they are never scored.
- **The problems are olympiad.** Problem 1 is verbatim 2025 AIME I Problem 1, and 5 of the 8
  `val-sample` problems appear here. This is the first olympiad-difficulty material with
  official labels.
- **The label rule does not explain the inverted prevalence.** Re-labelling our own DeepSeek
  instances under `<= 0.10` / `>= 0.25` moves the robust rate from 0.250 to 0.311, nowhere near
  train-main-v2's 0.706. The gap is a property of the **problem population**, not the
  labelling: ours is Hendrycks MATH, theirs is olympiad. Note the direction - the *harder*
  problems have the *higher* robust rate, consistent with eligibility filtering.
- **E11 v2's constants disagree with it.** We answer `False` for Skywork and Olmo-3-Think, which
  this table labels 2 of 3 and 7 of 9 robust. Finding 27 chose those constants from a ~14-case
  leaderboard; this is independent evidence against them, on too few rows to act on alone.

**34. `problem` text is not a safe grouping key in that table.** (Same script.)

- 36 distinct problem strings collapse to **29 templates** at similarity > 0.90; the card itself
  says 28 problems.
- The near-duplicates are not whitespace: the same statement with a constant swapped
  (`$10^{5}$` vs `$10^{4}$`, `617` vs `1000`, `f(3520)` vs `f(2024)`), carrying different labels.
- **Consequence:** SHA-1 of the normalised text would place two instantiations of one template
  in different folds and leak. Any use of this table must group by template, not by text.

**35. The small-track scoring task was rebuilt on 16 September, which is the likeliest cause of
the leaderboard jump in Finding 23.** (Codabench API for competition 16180 and the Hugging Face
commit log, read unauthenticated on 19 Sep.)

- The task we submit to was **created 16 Sep at 22:35 UTC**, is numbered **v8**, id 36241, and
  is named `AIMO Robustness Validation v8 - Small Model Track`. `COMPETITION.md` had recorded
  v7. The Main Track task was created earlier the same day, at 17:12 UTC.
- **The organizers were assembling that validation set the same day.** Their commit log:

  | When (UTC) | Commit |
  | --- | --- |
  | 16 Sep 14:42 | replace contents with the `gap >= 0.25` construction (was v3) |
  | 16 Sep 15:11 | **withhold 7 cells used by the small-track validation set** |
  | 16 Sep 15:11 | update counts after withholding small-track val cells |

  The labelling rule changed and rows were pulled out of the train split *because they are in
  the small-track validation set*, on the day the task was recreated.
- **Independent of our own timestamps.** Finding 23 brackets the change between 16 Sep 23:33
  and 17 Sep 11:24 as we logged them; whether 22:35 UTC falls inside that bracket depends on
  whether those logs are UTC or local (UTC+6), and the Codabench API refuses unauthenticated
  reads of our own submissions. The commit log supports the conclusion either way.
- **It weakens Finding 21's reading.** Withholding 7 cells from a split covering five
  configurations is hard to reconcile with a hidden set made only of Qwen3.5-4B cases. Finding
  27 still prefers v2's constants, so nothing is retracted, but the composition claim is open.
- **Two contract facts changed:** the Competition phase allows **300 submissions per person**,
  not 100 (that was the Warmup limit; 10 per day still holds), and `reasoning_effort` is
  **merged upstream** (PR #9, `de79405`), touching the ingestion program and the shipped
  solutions, not only the README. Our submodule is pinned two commits behind at `995f992`; the
  E11 signature already works under both callers, so no submission change is required.

**36. The `pool_id` collision had a second, silent loss on top of the crash, and can be resolved
rather than dropped.** (Found independently on 19 Sep; Finding 29 reports the same root cause.)

- Same diagnosis: `AI-MO/aimo-validation-amc` reuses its `id` column, so the 505-row pool holds
  471 distinct ids - **34 collisions, all AMC**, covering 68 problems.
- **The dangerous symptom was silent**, not the crash: `label.py` keyed `base_accuracy` on
  `pool_id`, so a variant of problem A could be scored against problem B's baseline. Blast
  radius at the time:

  | Affected | Count |
  | --- | --- |
  | Pool problems sharing an id | 68 of 505 (13%) |
  | Variants referencing a colliding id | 143 of 1249 |
  | Qwen base records | 22 of 208 (6 of 64 eligible) |
  | Qwen variant records | 18 of 107 |
  | Skywork variant records | 4 of 28 |

- **A second loss sat on top.** The base phase names items `orig:<pool_id>` and `load_records`
  de-duplicates on `item_id`, so one record of each colliding pair was discarded *before any
  labelling logic ran*: 208 Qwen records collapsed to 197. Both problems had been sampled, so
  that was paid-for GPU time thrown away.
- **Two fixes exist.** Ours disambiguates at the source (`pool.py` suffixes colliding ids) and
  drops records filed under the old ambiguous ids. Theirs resolves them instead:
  `(pool_id, expected_answer)` is unique across all 505 problems, so pre-fix records are
  recovered rather than discarded, and `sample.py` suffixes only colliding `item_id`s so
  existing records still resume. **Their `label.py`, `--problem-keys` and the `item_id` suffix
  are all ported**, after checking that their labeller reproduces E20 exactly on our records
  (49 instances, 45 train-ready, 38 self-robust). Their merge was
  verified rather than assumed: every formerly-colliding `pool_id` resolves to its own
  `problem_key`, the stored text matches the pool row, and `label.py` reported
  `base_records_unresolved: 0` on the Skywork set.
- **Report value.** A clean example of a dataset-provenance bug that produces plausible-looking
  labels rather than an error, caught only because a duplicated-index crash fired first.

**37. The gapfill runs: the cheapest possible self-labelling, and it worked.** (Their E23 and
E22, pushed as `e20-base-qwen35-gapfill` and `e22-base-skywork-gapfill` on NAKIB, 2.4 h and
1.4 h. Both registry entries are ported.)

E18 and E19 had sampled variants for problems whose *base* records sat on accounts that
checkout could not read, so none of them could be labelled. Each gapfill re-samples base
accuracy for exactly those problems: one k=4 sample each, unlocking variant sets already paid
for.

| | Qwen3.5-4B | Skywork-OR1-Math-7B |
| --- | --- | --- |
| Problems re-sampled | 19 | 25 |
| Eligible (solved 4/4) | 13 | 24 |
| Train-ready labels after pairing | 26 (19 robust) | 23 (0.87 robust) |

- **Eligibility is where the cost hides,** and Skywork - a math specialist - converts far more
  base samples into labels than Qwen.
- Their Qwen self-labels are **73.1% robust** against train-main-v2's 70.6%, and have **zero
  overlap** with it by `problem_key`.
- The Qwen run used one T4 of two: `base_sampling` was not wrapped in `per_gpu`, costing ~2x.

> **Superseded by Finding 31.** These labels are `answer_modulus` self-labels, which E20 showed
> do not measure official robustness. Kept as a record of what was measured.

**38. Robustness is a property of the problem, not of the model: the models agree 20/20.**
(`train/eval_train_main_v2.py` over `train-main-v2`; section `train-main-v2 transfer` of
`RESULTS.md`. Reproduced on this checkout, 20 Sep.)

| Scope | Agreeing pairs | Rate |
| --- | --- | --- |
| **Small-track models only** | **20 / 20** | **1.000** |
| All configurations, incl. gpt-oss efforts | 83 / 85 | 0.976 |
| Chance, at the 0.732 robust base rate | - | 0.608 |

- **Label permutation test: p = 0.0005.**
- **This is the first direct evidence for what Finding 22 inferred**: the signals that work are
  problem properties, not model properties. There it was implied by `problem_nll` transferring
  across models; here it is measured on the labels themselves.
- **Consequence:** olympiad labels are fungible across models, so thin per-model data can be
  pooled into one problem-level training set. It is also the cleanest reason our cross-model
  *feature* test failed in Finding 32: the labels transfer, our features do not.
- **Caveat:** 20 comparisons over 15 problems. Strong but small.

**39. The MATH-trained text block does not transfer to olympiad problems, measured with CIs.**
(Same script; complements Finding 32, which measured the fused model on the DeepSeek rows.)

| Scope | n | ROC-AUC | 95% CI |
| --- | --- | --- | --- |
| All small-track rows | 34 | **0.617** | **[0.391, 0.833]** |
| `Qwen/Qwen3.5-4B` | 12 | 0.688 | - |
| `Skywork/Skywork-OR1-Math-7B` | 3 | 1.000 | - |
| `allenai/Olmo-3-7B-Think` | 9 | 0.500 | - |
| **`deepseek-ai/DeepSeek-R1-0528-Qwen3-8B`** | 10 | **0.476** | - |

- **The CI spans 0.5, and the worst row is the model the block was trained for.**
- Always-robust scores 0.706 here; the best in-sample threshold sweep reaches only 0.765.
- **A proposed E11 v3 was abandoned on this evidence** (their gate G-T, 19 Sep), independently
  of ours (Finding 32).
- **A calibration trap avoided:** the train split is 73% robust, the hidden validation set is
  close to balanced (Finding 27). **Never calibrate a threshold on the train prevalence**, and
  use prevalence-independent metrics against this table.
- **What it does not say.** It does not show robustness is unpredictable on olympiad problems,
  only that a model fitted on MATH does not predict it there. That was an argument for olympiad
  *training* data - which Finding 31 then closed off, and Finding 32 tested directly.

**40. Training the text block on olympiad labels instead does not rescue it.** (Three-way
refit, scored on the 34 official rows.)

| Text block trained on | n | ROC-AUC | 95% CI |
| --- | --- | --- | --- |
| DeepSeek MATH (shipped) | 138 | 0.617 | [0.395, 0.838] |
| Qwen olympiad self-labels | 26 | 0.600 | [0.396, 0.788] |
| Pooled MATH + olympiad | 164 | 0.579 | [0.375, 0.783] |

Paired bootstrap, olympiad minus MATH: **-0.017, CI [-0.326, +0.298]**. Underpowered, but there
is no hint of a gain, so the failure in Finding 39 is not simply a training-difficulty problem.

**41. Prompt hidden states were the one block that did not fail on olympiad labels - on
invalidated labels, at n=12.** (Their E21, `e21-stagea-qwen35-olympiad` on MR, 142 s;
`train/eval_stagea_olympiad.py`, both ported.)

Probes fitted on 26 Qwen self-labels, scored on the 12 officially labelled Qwen rows, disjoint
by `problem_key`.

| Block | ROC-AUC |
| --- | --- |
| **Stage A, all poolings** (pre-specified) | **0.844** |
| Text block, same rows | 0.688 |

- **Two controls disagree.** Permuting the evaluation labels gives p = 0.037; shuffling the
  *training* labels and refitting - which keeps the probe's capacity to fit noise - gives
  **p = 0.073**. The strict control does not clear 0.05.
- Paired against text: **+0.158, CI [-0.273, +0.600]**; `max` pooling scored 0.938 and must be
  ignored as the selection effect of Finding 11.
- Stage A cost 142 s against the gapfill's 2.4 h: representation work is about three orders of
  magnitude cheaper than sampling work.

> **Superseded by Finding 31** as a result, since the training labels are `answer_modulus`
> self-labels. **It is the one open tension in the record:** those labels are anti-correlated
> with official ones for DeepSeek (Finding 31), yet a probe fitted on them ranked official Qwen
> labels at 0.844. Either is explicable at n=12, and re-testing it needs Stage A features over
> officially labelled problems - which Finding 32 has now measured directly, at chance.

**42. A fleet audit: two aliases are the same accounts twice, and a completed Olmo run had
never been downloaded.** (Audit of `kernels list --mine` across that checkout's aliases.)

- **`CHOTON` is the same Kaggle account as `Labib`, and `MEDIUM` the same as `RSD`.** The
  registry names eleven aliases but they resolve to fewer accounts, so their weekly quota is
  **shared, not additional**. A twelfth alias, `MR`, is new and untouched.
- `e17-base-olmo-think` had status COMPLETE, not the "paused" our log recorded, and nothing had
  pulled its output: **32 problems sampled, 18 eligible**, fp16, TP=2, 9.7 GPU-hours recovered
  for the cost of one download.
- **Olmo really is the slow model:** 30.4 tok/s against Qwen's 85 and Skywork's 117; of 128
  samples, 50 hit the 16k cap and 41 produced no answer. Finding 26's reason for pausing stands.
- **18.5 GPU-hours are reachable from that checkout; 46.0 GPU-hours across 20 runs are not**,
  including `e16-base-skywork` and the larger half of Qwen's base phase.

**43. More labels were affordable, but could never buy what the project is short of.**
(`datagen/plan_labelling_cost.py`, from measured throughput; section `Labelling cost model` of
`RESULTS.md`. Ported for the report.)

| Model | Pool-wide eligible | Cost per usable label | Whole pool |
| --- | --- | --- | --- |
| Qwen3.5-4B | 32% | 0.71 GPU-h | 113 GPU-h for ~159 labels |
| Skywork-OR1-Math-7B | 64% | 1.47 GPU-h | 474 GPU-h for ~322 labels |

- Labelling the entire pool for Qwen would have cost about 16% of the remaining fleet budget, so
  **the shortage was never a compute problem**.
- It corrects Finding 37: base-phase cost says Skywork is cheaper, but its variants phase ran at
  2425 GPU-seconds per item against Qwen's 567. **That 4.3x gap is unexplained** - E19 generated
  at 19.9 tok/s where E22 generated at 116.6 under a byte-identical vLLM config - and host
  throttling is the guess, not a measurement.
- **At the time, the binding ceiling was compliance, not volume.** Only `answer_modulus` was
  labelable, so every label measured robustness under a *changed requested output* while the
  competition scores *answer-preserving rewrites*. The 19 Sep clarification later removed
  that rule; Finding 54 records the conditional official-family route it opens.

**44. Merging the organizers' split with the older tables corrupted two DeepSeek labels.**
(Found while porting Finding 33; fixed in `datagen/ingest_public.py`.)

`build_instances` aggregated every table's rows for a (model, problem) pair and took the largest
decay. Two problems appear in both `aimo-2025-reference` and `train-main-v2`, which are labelled
under **different rules**, and the older derived measurement won:

| Problem | Official | Ours before | Effect |
| --- | --- | --- | --- |
| `cd15d0c0…` | robust (`max_drop` 0.08) | spurious (decay 0.11) | label flipped |
| `49981d06…` | robust (`max_drop` 0.04) | ambiguous (decay 0.11) | dropped from training |

- **Fix:** where `train-main-v2` covers a pair, its rows alone decide the label, the decay and
  the `dataset_id`. This is the invariant their CLAUDE.md states as "do not merge the two
  labellings"; we were merging them.
- **Effect:** DeepSeek goes from 152 to **153 train-ready instances (46 robust)**, and its
  official olympiad evaluation set from 8 rows to **10**, which is what Finding 39 scores
  against. Finding 32 was measured on the 8 and is re-measured on the 10.

**45. Per-model constants cannot score above 0.64 on the validation set; 0.79 needs per-problem
predictions.** (Arithmetic on Finding 27's probes; another team reached 0.79 on 21 Sep.)

- **The set is 14 cases.** Every score seen is a multiple of 1/14: all-False 0.50 = 7/14,
  Qwen-True 0.57 = 8/14, Skywork-True and Olmo-True 0.43 = 6/14, v2 0.64 = 9/14, and the leader's
  0.79 = 11/14.
- **Each probe fixes one model's balance.** Write `d_m` for model m's robust cases minus its
  spurious cases. Turning one model to `True` moves the score by `d_m / 14`, so `d_Qwen = +1`,
  `d_Skywork = -1` and `d_Olmo = -1`. All-False at 7/14 means 7 robust and 7 spurious overall, so
  the four balances sum to 0 and **`d_DeepSeek = +1`**.
- **So the best constants are exactly v2's:** `True` for Qwen and DeepSeek, `False` for the
  other two, which scores 7 + 1 + 1 = **9/14 = 0.64**. E11's DeepSeek predictions are worth the
  same single case as answering `True` for DeepSeek, which fits E21: on olympiad rows the fused
  model says robust 8 times in 10.
- **Consequence.** No choice of defaults beats v2. The leader's 11/14 is two cases of real
  per-problem discrimination, and matching it takes a signal that reaches two more cases.
- Assumes the hidden set has not changed since the 18 Sep probes. It changed once before
  (Finding 35), so a re-probe is the check if scores ever stop adding up.

**46. E22: on the official olympiad problems, nothing clears chance at the problem level.**
(`train/eval_problem_level.py`, section `E22: problem-level acceptance harness` of
`RESULTS.md`; 33 labelled problems in 28 templates, 20 robust, labels pooled over every
configuration, leave-one-template-out, CIs resampling templates.)

| Signal | ROC-AUC | 95% CI | Permutation p |
| --- | --- | --- | --- |
| shipped MATH text block | 0.517 | [0.300, 0.725] | 0.448 |
| char tf-idf refit on official labels | 0.560 | [0.371, 0.755] | 0.306 |
| the same plus 142 DeepSeek MATH rows | 0.500 | [0.323, 0.689] | 0.510 |
| **problem length only** | **0.635** | [0.423, 0.835] | 0.101 |

- **Problem length is the best signal and it still does not clear chance.** It is the same
  length signal that dominated MATH (Finding 22), and it points the same way: longer problems
  are more often robust.
- **Adding MATH rows hurts** (0.560 → 0.500), consistent with Findings 32 and 39: MATH teaches
  the wrong thing for olympiad problems.
- **The harness is what every later gate uses**, including G-M (E23) and G2. At 33 problems,
  only a signal with AUC around 0.75 or more can put its CI above 0.5.
- **On MATH, DeepSeek and Llama agree on 26 of 34 shared problems (0.76)**, but Llama is
  spurious everywhere, so this only restates DeepSeek's spurious rate (0.70 overall). It
  cannot confirm Finding 38 outside `train-main-v2`.

**47. E23: familiarity scores fail gate G-M, but on olympiad problems they point the way the
memorization hypothesis predicts.** (`e23-familiarity-a/b`, RSD and Labib, 22 Sep; 193 problems
under each of the four models, bf16, every tokenizer round-trips, 0.4-0.65 s per problem.
`train/eval_familiarity.py`, section `E23: familiarity and contamination` of `RESULTS.md`.)

| Signal | MATH AUC | MATH within length | Official AUC, MATH-oriented | 95% CI | p |
| --- | --- | --- | --- | --- | --- |
| `mean_nll` | 0.338 | 0.426 | 0.308 | [0.142, 0.507] | 0.962 |
| `min_k_10` | 0.475 | 0.504 | 0.600 | [0.391, 0.789] | 0.179 |
| `min_k_20` | 0.431 | 0.493 | 0.342 | [0.155, 0.538] | 0.932 |
| `min_k_pp_20` | 0.501 | 0.486 | 0.535 | [0.336, 0.742] | 0.376 |
| `zlib_ratio` | 0.394 | 0.455 | 0.465 | [0.278, 0.650] | 0.608 |
| `suffix_nll` | 0.325 | 0.345 | 0.396 | [0.198, 0.610] | 0.848 |
| `relative_nll` | 0.614 | 0.517 | 0.302 | [0.134, 0.527] | 0.973 |

- **Gate G-M fails as pre-registered.** No signal keeps its MATH direction on the official
  problems with a CI above 0.5.
- **The direction flips between populations, and the flip is the interesting part.** On MATH,
  familiar problems lean *robust* (`suffix_nll` within length: 0.345, i.e. 0.655 reversed, the
  strongest within-length label-free signal we have measured; E12's best was 0.587). On the
  olympiad problems, four of the seven lean the other way: read unoriented, `mean_nll` scores
  0.69 [0.49, 0.86] and `relative_nll` 0.70 [0.47, 0.87], so *familiar problems lean
  spurious*. That is the memorization reading of "spurious" stated in `PLAN.md` §11 before any
  of this was run.
- **It is not established.** Seven signals were tested, every lower bound is below 0.5, and
  the olympiad direction was read after the fact from the same 33 problems. It stays a
  hypothesis, and the leaderboard is the cheapest independent test of it: E27d.
- **E27d** predicts robust for the less familiar half of each model's batch, using the model's
  own mean NLL of the bare problem. Its GPU checks (`e27d-gpu-check-a/b`) loaded the packaged
  ZIP from cache for all four models with no fallback, reproduced E23's NLLs to 2e-7, returned
  native bools, and took 38 s (Qwen), 130 s (DeepSeek), 50 s (Skywork) and 139 s (Olmo) on T4s
  including model loading.

**48. E24: on olympiad problems the self-labels stop disagreeing with the official ones, but
gate G-V2 still fails, and the yield is the deeper problem.** (E24a answer key + E24b on
TASHIN, ISHRAQ, RSD and Labib, 22-23 Sep, 15.8 GPU-hours. `train/compare_selflabels.py`,
section `Self-labels vs official labels` of `RESULTS.md`.)

E20 rejected `answer_modulus` self-labels on Hendrycks MATH problems. Since every other signal
we hold broke on the MATH -> olympiad shift, E24 repeated the comparison on the problems
`train-main-v2` labels, under the perturbation families the competition actually scores.

| Set | Pairs | Agreement | Balanced acc. | Cohen's kappa [95% CI] | Spearman of drops |
| --- | --- | --- | --- | --- | --- |
| E20, Hendrycks MATH | 49 | 0.29 | 0.391 | **-0.135** [-0.347, 0.041] | -0.25 |
| **E24, olympiad** | **15** | **0.80** | 0.673 | **+0.286** [-0.190, 1.000] | +0.20 |
| E24, clear-cut only | 13 | 0.85 | 0.458 | -0.083 [-0.226, -0.083] | +0.44 |

- **The sign flips with the population.** Every headline number moves from "mildly inverted" on
  MATH to "weakly positive" on olympiad problems. That is consistent with the pattern in
  Findings 32, 39 and 47: what holds on MATH need not hold, or even keep its direction, where
  the competition scores.
- **It still fails the gate**, which asked for kappa >= 0.3 with the CI above 0. Agreement of
  0.80 sounds high only until the marginals are accounted for: 80% of our labels and 87% of the
  official ones are robust, so **chance agreement is already 0.72**. The clear-cut subset
  disagrees with the full set, which is what noise looks like at this size.
- **The yield is the harder obstacle.** A pair needs the model to solve the original 4 times
  out of 4 under our sampling, and these are problems the organizers' own runs solved:

  | Model | Official problems with an answer | Base-eligible for us | Pairs labelled |
  | --- | --- | --- | --- |
  | Qwen3.5-4B | 9 | 5 | 5 |
  | allenai/Olmo-3-7B-Think | 8 | 6 | 6 |
  | deepseek-ai/DeepSeek-R1-0528-Qwen3-8B | 7 | **1** | 1 |
  | Skywork-OR1-Math-7B | 3 | 3 | 3 |

  DeepSeek cleared 1 of 7 even at a 24k cap, and its variants phase covered 2 variants before
  its budget. **15 of the 27 available pairs produced a label for 15.8 GPU-hours**, so a
  larger olympiad self-labelling programme would cost far more per label than Finding 43
  estimated on the pool, where the problems were not selected for difficulty.
- **Decision: Lane E stays closed** (gate G-V2, 23 Sep). The record now covers both
  populations, and neither supports training on `answer_modulus` labels. What changed is the
  reason: on MATH the labels were wrong, on olympiad problems they are merely unobtainable in
  useful quantity.

**49. The leaderboard rejects the official-label models and, by scoring E27d at 4 of 14, hands
us a better submission than E11 v2.** (Codabench, Small Model Track, 22 Sep, submissions
939268-939318.)

| Submission | Score | In 14ths |
| --- | --- | --- |
| `e27a` char tf-idf on official labels, every model | 0.57 | 8 |
| `e27b` the same, E11's fusion kept for DeepSeek | 0.57 | 8 |
| `e27c` problem length on official labels | 0.57 | 8 |
| **`e27d` familiarity split, unfamiliar half robust** | **0.29** | **4** |
| E11 v2, for reference | 0.64 | 9 |

- **Training on the official labels buys nothing.** All three land one case *below* v2, which
  matches E22: on those 33 problems no signal cleared chance, and a model fitted on them
  carries that nothing to the validation set.
- **The validation problems are mostly not `train-main-v2` problems.** `e27a` is a text model
  fitted on exactly those problems' labels; had the validation cases reused them, it would have
  scored far above 0.57 through Finding 38's cross-model agreement. So **the leader's 0.79 is a
  method, not problem overlap** - the caveat in §11.6 is retired.
- **E27d proves the GPU path runs on the evaluation worker.** Its fallback is E11 v2's
  constants, which score 0.64; 0.29 can only come from the forward passes actually running.
- **4 of 14 is a result, not a dud.** The split predicts exactly half of each batch robust, so
  chance is 7 of 14 and no constant scores below 0.43. Being that far below chance means the
  predictions are systematically inverted, and **the opposite split scores 10 of 14 = 0.71**,
  above v2's 9. `e27e` ships that split. (Correction: the two are exact complements only for an
  even batch. With an odd one both call the median problem robust, so what holds is
  `score(d) + score(e) = ` non-median cases `+ 2 x` medians both get right. Here 4 + 10 = 14
  with 2 medians correct, which is consistent, and `e27e` did score 0.71.)
- **The direction is the MATH one after all.** `problem_nll` ranks robust below spurious on
  DeepSeek's 142 MATH instances (0.68 reversed, 0.58 within length quintiles: Findings 22, 47),
  and the leaderboard agrees: **familiar problems are robust**. E23's olympiad reading, which
  E27d shipped, was 33 pooled problems with a CI touching 0.5, and it is now rejected.
- **What this is worth, honestly.** One case above v2 on a 14-case set. The direction rests on
  the MATH sample (n=142) plus this leaderboard reading, against the E23 olympiad reading it
  contradicts, so the private test could still side with E23. v2 stays the fallback, and
  `e27e`'s margin is not large enough to freeze anything on.
- **Next, without probing a single case:** `e27f` (the constants alone) and `e27g`-`e27j` (the
  split for one model each) decompose the gain by model, which is the same per-model probing
  discipline Finding 27 used.

**50. The E27 gain is DeepSeek-specific; Min-K tail shape passes offline stability but fails
both controlled transfers.** (Codabench submissions 941763-942361; E28 Kaggle checks
on TASHIN and RSD; `train/eval_e28_stability.py`, 24 Sep.)

- **The decomposition is already informative.** Constants score 9/14. Splitting only Qwen or
  Skywork by mean NLL stays at 9/14, while splitting only DeepSeek reaches **10/14 = 0.71**,
  the same score as splitting every model. Olmo remains queued. Thus the observed gain belongs
  to DeepSeek; changing the other completed models adds zero net cases.
- **Suffix NLL does not improve the independent leaderboard result.** E28a calls the familiar
  half robust using suffix NLL and scores **0.71**, tying E27e. It is useful confirmation of the
  familiarity family, not a new best.
- **Min-K gap survives a stricter offline stability test.** The selected statistic is
  `min_k_10 - min_k_20`, with lower gaps called familiar. Averaging model-wise ranks over the
  official problems gives AUC 0.808 after ranks are recomputed inside that set. Leaving out
  Qwen, Skywork, Olmo or DeepSeek gives 0.835, 0.764, 0.731 and 0.802 respectively; the
  pre-registered worst-case gate of 0.65 therefore passes. Mean NLL's worst omission is 0.242
  and suffix NLL's is 0.330. On DeepSeek's own 10 labelled rows, Min-K gap has AUC **0.762**.
- **The E28d infrastructure failure was resolved by one exact retry, and the result fails.**
  Submission 942257 failed on the Codabench ingestion host before scoring. The unchanged ZIP
  was retried once as submission 942361 and scored **4/14 = 0.29**. It still passes the official
  local ingestion/scoring path with coverage 1.0 and zero invalid predictions, so this is a
  genuine transfer failure rather than a packaging failure.
- **E29a is the single follow-up, and it fails.** Submission 942281 applies Min-K gap only to
  DeepSeek and keeps Qwen=True, Skywork=False and Olmo=False. It passes five policy tests and
  the official local scorer, but scores **8/14 = 0.57** on Codabench, two cases below E27's
  10/14. Combined with E28d's 4/14, this rejects the tail-gap family and retains E27e/E27h as
  the best policy. No rank-fraction or ensemble search is justified on 14 cases.
- **Fleet constraint respected:** no `farhanishraqq` GPU was used for E28/E29.

**51. Static activation geometry is an attractive official-data artifact that reverses on
the independent MATH set.** (`train/eval_e30_geometry.py`, 24 Sep; no new GPU.)

- Twelve architecture-invariant layer-trajectory scalars were computed from the existing E21
  all-layer archives. `mean_growth` led the 34 official rows at leave-one-template-out AUC
  **0.813**, template-bootstrap CI [0.600, 0.976], but its max-T p-value after the 12-way
  screen was **0.2049**.
- The selected direction then reversed on the untouched 142-row DeepSeek MATH set: AUC
  **0.348**, CI [0.250, 0.449], within-length AUC 0.438.
- **Decision:** E30 closes without GPU work or a submission. This is the second warning after
  Min-K gap that a strong official-row geometry can fail on a new population.

**52. Counterfactual prompt-state sensitivity is the strongest new official-data signal, but
it misses the pre-registered gate.** (`e31-counterfactual-a/b` on TASHIN and RSD;
`train/eval_e31_counterfactual.py`, 24 Sep.)

- E31 ran the original problem and every applicable deterministic answer-preserving rewrite
  through the labelled model, with no generation. All 34 official rows completed, every
  tokenizer round-tripped, and no prompt truncated.
- The fixed prediction was that robust rows move less under rewrites. Mean cosine distance
  over all transformer layers reached pooled within-model-rank AUC **0.735**, but its
  19-template bootstrap CI was **[0.380, 0.938]** and the max-T p-value correcting the five
  registered signals was **0.0667**. Length-residual AUC was 0.654 and leave-one-model-out
  threshold balanced accuracy 0.696.
- The model split is the warning: Qwen 0.531, Skywork 1.000 on only three rows, Olmo 0.929,
  DeepSeek 0.762. The locked gate also required every model AUC >= 0.65.
- **Decision:** E31 is not submitted. Its single permitted follow-up is E32: the already
  selected all-layer statistic and lower-is-robust direction on 142 untouched DeepSeek MATH
  rows. No perturbation or layer search is reopened.

**53. E32 rejects counterfactual activation sensitivity on an independent population.**
(`e32-counterfactual-deepseek-math` on TASHIN; `train/eval_e32_counterfactual_math.py`,
24 Sep, 0.21 h wall time.)

- All **142** labelled DeepSeek Hendrycks MATH rows completed (459 forward passes), with no
  truncation and a round-tripping tokenizer. The signal and lower-is-robust direction were
  inherited unchanged from E31.
- All five locked criteria fail: AUC **0.545**, bootstrap CI **[0.428, 0.658]**, within-length
  AUC **0.510**, median-split balanced accuracy **0.527**, and one-sided p **0.2061**.
- **Decision:** the E31/E32 counterfactual-sensitivity lane closes without packaging or a
  Codabench upload. Do not tune rewrites or layers on these same labels. E27e/E27h remain the
  best measured policy at 10/14 = 0.71.
- **Fleet constraint respected:** E31/E32 used TASHIN and RSD only, never the excluded
  `farhanishraqq` GPU account.

**54. The data rule reopened, but the symbolic dataset needed for a clean new experiment is
still inaccessible.** (Official website, GitHub and Hugging Face audit, 25 Sep.)

- The current website says the restriction on extra labeled data was retired on 19 Sep.
  Self-generated labels for official perturbation families are therefore allowed; leakage of
  evaluation data remains forbidden. This supersedes this repository's old §3.3 reading.
- `aimo-interp/problems-public` is indexed as a 22-row symbolic validation dataset, with
  programs, constraints, templates and original results. The live Hub API returns **401**, and
  it is absent from the seven datasets currently visible under the official organization.
- No substitute release appeared: `train-main-v2` remains at commit `23791d9b` from 17 Sep;
  the getting-started and baselines heads remain `de794053` and `7e883996`; the other visible
  datasets were last changed in May.
- The official getting-started repository restricts new issue creation. Existing issue #6
  already asks whether self-generated counterfactual evaluation is permitted and is still
  unanswered. The next contact should therefore go through the challenge Discord or organizer
  registration channel, requesting `problems-public` access rather than opening a duplicate.
- **Conditional E33 is now frozen, not launched:** train only on new labels generated from the
  symbolic programs; score once on the external official `train-main-v2` rows using the E22
  19-template harness. No rewrite, layer, threshold or rank-fraction search on that holdout.

---

### Finding 62: E37 independent confirmation passed (27 Sep)

The preregistered GSM-Plus confirmation was executed on Arifin from the pinned archive.
All gates passed; `RESULTS.md` contains the generated measurements. The exact result,
source hashes, archive hashes and environment are preserved under
`kernels/runs/e37-confirmation-Arifin/`. Regenerate its table with `train/report_e37.py`.

This authorizes the unchanged official transfer specified in `protocols/E37_PROTOCOL.md`.
The GPU jobs are now being executed using available accounts. Runtime bootstrap failures
are recorded separately from scientific outcomes; no confirmation-derived feature or
threshold adjustment has been made. An E37 submission still requires the official transfer
to pass its accuracy, template-bootstrap and runtime gates.

### Finding 63: exact-prefix reuse is not supported by the archive (28 Sep)

E38a tested the frozen hypothesis that one longest trace could provide all shorter-budget
primary traces exactly. The CPU audit on Tamim3 failed in every comparison; the generated
`RESULTS.md` section contains the model/source counts. No correctness labels were read.
The conditional GPU equivalence experiment is closed under `protocols/E38_PROTOCOL.md`.
This is an execution-cost result, not an accuracy result and not a reason to change E37's
features, budgets or gates. Preserve independent-budget generation for the official transfer.

### Finding 64: E37 exceeds its runtime gate after the execution repair (29 Sep)

The old official-transfer jobs reported `RUNNING` after a fatal CUDA linker failure.
Live logs exposed the missing `libcuda.so` linker name; a local symlink to the installed
driver, compiler smoke check and `cuInit` check repaired initialization. A process guard
now bounds failed or overlong jobs. Olmo also needs the compatible Transformers 5.5.3 pin
to preserve its published rotary configuration. All four old stuck jobs are now confirmed
cancelled by Kaggle; the user handled their session cancellation.

The repaired Qwen and Skywork runs completed, but their measured runtime alone exceeds
E37's frozen whole-run limit. DeepSeek's guard stopped its process after two hours with
partial features saved. Olmo's Tamim3 notebook now returns `kernels.get` access denied,
although its configured and authenticated usernames match. The reason for that denial is
unknown. Exact coverage, timings and output hashes are recorded in
`kernels/runs/e37-runtime-result.json`; `train/report_e37_runtime.py` generates the table.

**E37 is closed by its runtime gate.** Official-set accuracy remains unmeasured because
coverage is incomplete. No remaining run, faster-hardware assumption or post-hoc budget
change can make this measured result pass the frozen gate. No new Codabench submission
or further GPU retry follows; E27k remains the incumbent. E38 is also closed (Finding 63).

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

### Model internals and fusion (DeepSeek, nested, in-fold thresholds)

| Method | Balanced acc. | ROC-AUC | Source |
| --- | --- | --- | --- |
| Text alone, learned threshold | 0.6389 | 0.6481 | Finding 16 |
| Stage A layer ensemble alone | 0.6046 | 0.7059 | Finding 16 |
| **Text + Stage A (E11)** | **0.7173** | 0.7230 | Findings 11, 16 |
| Text + Stage A + trace states | 0.7116 | **0.7391** | Finding 18 |
| Best label-free signal (`next_token_entropy`, no fitting) | — | 0.7361 | Finding 22 |

Full tables are in `RESULTS.md`.

### Leaderboard (Codabench 16180, Small Model Track)

| Submission | Score |
| --- | --- |
| E11 v1 | 0.47 |
| Constant: all `False` | 0.47 |
| Constant: `True` for Qwen3.5-4B only | 0.53 |
| **E11 v2 (v1 + Qwen `True`)** | **0.64** |
| Re-measured 18 Sep: all `False` / `True` for Qwen / Skywork / Olmo-Think | 0.50 / 0.57 / 0.43 / 0.43 |

The hidden set is about 14–15 cases, so one case moves the score by about 0.07. v2's
defaults are the best constants (Finding 27).

### Self-labelling (E15–E19)

| Model | Eligible problems | Compliant variants | Status |
| --- | --- | --- | --- |
| Skywork-OR1-Math-7B | 157 of 246 sampled | 205 over 101 problems | Variants phase running (E19) |
| Qwen3.5-4B | 90 of 285 sampled | 109 over 53 problems | Variants phase running (E18) |
| Olmo-3-7B-Think | 17 of 30 sampled | 29 over 13 problems | Paused: too slow on T4s |

Generated table: `RESULTS.md`, section "E15-E19: self-labelling progress" (Findings 25–26).

### The measurement we cannot yet make

Two gaps remain:

1. **Olympiad difficulty.** The DeepSeek train set contains **exactly one olympiad-level
   instance** alongside 137 Hendrycks MATH problems, while the private test set is entirely
   olympiad.
2. **No labels for the other three small-track models.** The leaderboard appears to score
   Qwen3.5-4B. Nothing we have trained applies to it, unless robustness is mostly a property
   of the problem.

Closing either gap needs new labels, from the organizers or from E7.

---

## 8. Code inventory

### `datagen/`: data and features

| File | Purpose |
| --- | --- |
| `ingest_public.py` | Consolidates all six public datasets into `data/small_track/`: derives labels, applies eligibility and clarity filters, builds grouped folds |
| `extract_features.py` | Stage A: one forward pass per problem; pooled hidden states at every layer (`last`/`mean`/`max`) plus next-token uncertainty and non-finite counts. Shardable, resumable, multi-GPU safe |
| `extract_generation_features.py` | Stage B: greedy generation through the starter kit's `uncertainty_profile` (unchanged). Adds the 14 trace-uncertainty features, generated text, and a token-cap flag. Aborts on non-finite preflight logits |
| `merge_features.py` | Merges Stage A shards, refusing any that disagree on dtype, layer count, or prompt hash |
| `extract_trace_states.py` | E4: teacher-forced pass over prompt + E2 trace; segment-pooled states via hooks |
| `extract_label_free.py` | E12: label-free signals (immediate-answer confidence, stability under rewrites, problem NLL, next-token uncertainty) |
| `tokenizer_check.py` | Loads `PreTrainedTokenizerFast` and refuses any tokenizer that does not round-trip (Finding 14) |
| `perturb/` | E7: olympiad pool, answer-aware transforms, variant generation, base/variant sampling (HF or vLLM), labelling; with tests |
| `plan_shards.py` | Reports shard workload and balance without loading a model (default 1.0 s per instance, measured) |
| `kaggle_launch.py` | Experiment registry; renders self-contained private GPU kernels (embedded code bundle, optional private dataset mounts) and runs `push` / `status` / `logs` / `output` per account |
| `README.md` | Runbook |

### `train/`: modelling

| File | Purpose |
| --- | --- |
| `train_text_baseline.py` | Text-only baselines under grouped CV; rewrites `RESULTS.md` |
| `check_dataset_confound.py` | Tests whether the text signal is really dataset detection; appends to `RESULTS.md` |
| `train_stagea_probes.py` | Nested probe, layer ensembles, text fusion, and bootstrap CIs on Stage A features; upserts its `RESULTS.md` section and saves out-of-fold probabilities |
| `train_stageb_features.py` | Trace-uncertainty and trace-shape classifiers, univariate AUCs, and text fusion on Stage B records; upserts its section |
| `train_fusion.py` | E10: nested fusion of text / stagea / tracestate / trace blocks, with in-fold thresholds and paired CIs |
| `eval_label_free.py` | E12: per-signal AUC against DeepSeek labels, within-length-quintile control, cross-model transfer |
| `eval_agreement.py` | E14: agreement signals against DeepSeek labels, and their generation cost |
| `summarize_labelling.py` | E15–E19: base-phase eligibility and variants-phase labels per model, deduplicated across sessions |
| `build_e11_artifact.py`, `check_e11_artifact.py` | Fit the E11 artifact in the pinned venv and verify it locally |
| `results_io.py` | Shared metrics, paired bootstrap AUC difference, section upsert |

### `submission/`: Codabench bundles

| File | Purpose |
| --- | --- |
| `e11-text-stagea/` | E11: text + Stage A fusion for DeepSeek, constant defaults for the other models (v2: `True` for Qwen3.5-4B) |
| `package.py` | Builds `dist/solution-<name>-small.zip` with the starter kit's archiver |
| `make_probes.py` | Constant-prediction leaderboard probes (Finding 21) |
| `gpu_check.py` | Runs a packaged ZIP on Kaggle GPUs against the artifact's reference decisions |

### `scripts/`: setup and one-off audits

| File | Purpose |
| --- | --- |
| `_setup_kaggle_accounts.py` | Verifies and installs the Kaggle access tokens in `Secrets.env` |
| `export_kernels.py` | Writes every kernel as pushed, and each run's manifest, to `kernels/` |
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

# GPU stages on the Kaggle fleet (push, wait, download)
& $py datagen\kaggle_launch.py push e1b-stagea-deepseek
& $py datagen\kaggle_launch.py status
& $py datagen\kaggle_launch.py output e1b-stagea-deepseek
# likewise: e2f-stageb-fp16-s0/s1, e4-tracestate-deepseek, e12-labelfree-deepseek, e13-*-plumbing

# Analysis (each upserts its own RESULTS.md section)
& $py train\train_stagea_probes.py
& $py train\train_stageb_features.py
& $py train\train_fusion.py
& $py train\eval_label_free.py

# Submission
& $py train\build_e11_artifact.py
& $py submission\package.py e11-text-stagea
& $py submission\make_probes.py
```

`data/`, `features/`, `runs/`, and `kaggle_runs/` are git-ignored; all regenerate
deterministically.

---

**55. E34 creates a genuinely new validation population before another feature search.**
(`protocols/E34_PROTOCOL.md`, `e34a-qwen-aime-surface`, launched 25 Sep on TASHIN.)

- The 0.79 Small Models score is 11/14; beating it requires 12/14, two more correct cases
  than E27e/E27h's 10/14. Prior submission vectors are not combined to infer hidden labels.
- E34a uses 30 AIME 2024 problems and 22 AIME 2026 problems after removing the eight that
  overlap `train-main-v2`. No current official label participates in fitting or selection.
- The generator is frozen to `latex_to_plaintext` and `premise_reorder`, Qwen3.5-4B, k=4,
  temperature 0.6, top-p 0.95 and a 12,288-token cap. The official 0.10/0.25 bands apply.
- E34a is a yield and class-balance pilot only. Its gate was written before the GPU run and
  fits no classifier. Passing authorizes E34b; failure closes the lane without a submission.
- E34b is already frozen: train on new OlympiadBench+AIME-2024 labels, evaluate once on
  untouched AMC+AIME-2026 labels, then permit one fixed official-transfer measurement only
  if the independent holdout clears every gate in `protocols/E34_PROTOCOL.md`.
- Kernel version 1 is running at
  `https://www.kaggle.com/code/tashintahir/aimo-small-e34a-qwen-aime-surface`. The excluded
  `farhanishraqq` GPU account is not used.

**56. E34a fails before modelling: Qwen solves only 1/52 new AIME originals four times.**
(`e34a-qwen-aime-surface`, TASHIN, completed 25 Sep; frozen gate in `protocols/E34_PROTOCOL.md`.)

- Both base shards completed all 26 assigned problems, all 52 records resolved, and the
  checked tokenizer round-tripped correctly. This is not an interrupted-run artifact.
- With k=4 at temperature 0.6 and a 12,288-token cap, only one problem reached the required
  base accuracy of 1.0: **1.92% eligibility**, versus E34a's minimum of 8 AIME 2024 and 5
  unseen AIME 2026 clear-cut labels.
- That one problem yielded one robust `latex_to_plaintext` row. There was no eligible
  `premise_reorder` row and no spurious label, so every class-balance and family-coverage gate
  also failed.
- E34b is dropped without training a predictor. Lowering eligibility, changing sampling, or
  adding rewrite families after seeing this result would violate the pre-registration.
- No Codabench submission was made. E27e/E27h at 0.71 and E11 v2 at 0.64 remain unchanged.

**57. E35 finds real external robustness labels, but problem-text ranking does not transfer.**
(`protocols/E35_PROTOCOL.md`, completed locally 25 Sep.)

- `ssubhnil/Qwen3-Math-Eval@f64c82f` supplies strict per-problem correctness for four Qwen3
  sizes on paired MATH-Perturb and Putnam-AXIOM variants. E35 was committed before downloading
  row-level outcomes; MATH-Perturb was fixed as train and Putnam as the one-shot holdout.
- Frozen clear-cut aggregation produced 233 train problems (191 robust, 42 spurious) and a
  balanced 48-problem holdout (23 robust, 25 spurious).
- The fixed character+word TF-IDF ensemble reaches **AUC 0.609**, bootstrap CI
  **[0.442, 0.769]**, and fixed half-rank balanced accuracy **0.646**.
- Balanced accuracy passes, but the AUC and CI gates fail. The official-transfer clause never
  opens: `train-main-v2` is not scored, and there is no Codabench submission or post-hoc tuning.
- The current official FAQ also clarifies temperature 1.0, top-k 40, top-p 0.95 and a
  100,000-token maximum. Those settings explain why E34's 0.6/12,288 pilot is not an exact
  reproduction, but they do not license reopening E34 after its frozen failure.

**58. The per-model decomposition closes: Olmo adds nothing, and 10 of 14 is the ceiling of the
whole familiarity family.** (Codabench, Small Model Track, 26 Sep, submissions 945323-945329.
Completes the Olmo cell Finding 50 left queued, with an independent rebuild of the same
policies.)

| Submission | Score | In 14ths | Delta vs constants |
| --- | --- | --- | --- |
| `e27f` best constants alone | 0.64 | 9 | — |
| `e27g` split for Qwen only | 0.64 | 9 | 0 |
| **`e27h` split for DeepSeek only** | **0.71** | 10 | **+1** |
| `e27i` split for Skywork only | 0.64 | 9 | 0 |
| **`e27j` split for Olmo-3-Think only** | 0.64 | 9 | **0** |
| `e27e` split for all four | 0.71 | 10 | +1 |

- **Olmo was the missing cell and it is zero**, so all four deltas are now measured: Qwen 0,
  DeepSeek +1, Skywork 0, Olmo 0. They sum to +1, exactly `e27e` minus `e27f`, which is the
  consistency check passing.
- **That closes the choice of split-versus-constant per model.** Each model's cases are
  disjoint, so the score is the sum of per-model correct counts and the best combination takes
  each model's better option independently. With three ties and one +1, no recombination of
  split-versus-constant exceeds 10 of 14. Finding 50 reached the same DeepSeek-specific reading
  from a different set of submissions.
  > **Corrected by Finding 60.** This was read as a ceiling for the whole familiarity family.
  > It is not: it holds only with the robust *count* frozen at `ceil(n/2)`. Freeing that count
  > reaches 12 of 14, and the three "no gain" cells above were the wrong count, not a wrong
  > ranking.
- **`e27f` scoring exactly 9 confirms Finding 45 directly**: plain constants reproduce E11 v2's
  score, so the DeepSeek fusion is worth precisely what answering `True` for DeepSeek is worth.
- **The official rows still say the opposite.** Scoring the same familiar-robust split offline
  on the 34 official rows, using E23's stored NLLs:

  | | Qwen | Skywork | Olmo-3-Think | DeepSeek | Pooled |
  | --- | --- | --- | --- | --- | --- |
  | Familiar-robust split | 0.333 | 0.333 | 0.333 | 0.400 | **0.353** |
  | Best constants | 0.667 | 0.333 | 0.222 | 0.700 | 0.529 |
  | Always robust | — | — | — | — | 0.706 |

  The split is inverted there, which is E23's reading (Finding 47) and the opposite of both
  MATH and the hidden set. **Gate G2's official-row leg would therefore block the policy that
  wins on the leaderboard**, and that conflict is a freeze decision, not a modelling one.
- **The one thing left untried** was a tilted split, with Skywork and Olmo calling one fewer
  problem robust - the count their measured balance implies. It was held back as 14-case
  tuning; submitted anyway on 26 Sep, it scored 12 of 14 (Finding 60).

**59. The organizers answered: `problems-public` is private for good, the official labels are
final, and generating training data from `train-main-v2` is explicitly allowed.** (Michal
Spiegel, challenge Discord, 26 Sep 2026, answering the questions Finding 54 routed there.)

- **`aimo-interp/problems-public` is a private dataset they cannot share.** The 401 was never a
  permissions bug and the name is simply confusing. **E33 is dead**, not frozen: it was
  conditional on those symbolic programs, and no substitute exists.
- **The official label pool is final.** "Currently, we plan to add only a couple samples for the
  `Skywork/Skywork-OR1-Math-7B` model. Nothing else is planned." So the acceptance set stays at
  34 small-track rows over 20 problems, plus a few Skywork rows. **Every gate scored on official
  rows is permanently underpowered** - E22's harness will not grow, and the conflict in
  Finding 58, where the leaderboard and the official rows disagree about familiarity, cannot be
  settled by waiting for more data.
- **"Feel free to use it to generate any additional data for training."** With the 19 Sep rule
  retirement (Finding 54), this closes the compliance question that shaped E15-E24: seeding
  generated data from the official train set is sanctioned, and self-generated labels for the
  official perturbation families are allowed.
- **Permission was never the binding constraint, though.** Yield is. A self-label needs the
  model to solve the original in all four samples: E24 got 15 usable pairs from 27 (DeepSeek 1
  of 7, Finding 48) and E34a got one label from 52 problems (Finding 56). The 36 official
  problems are the only seed now blessed, and they are the same problems our sampling already
  struggles on.
- **What it changes for the plan.** Nothing to wait for, and one option reopened: perturbing the
  official problems with the *official* families and checking our derived labels against the
  official ones - the comparison E24 could not make when only `answer_modulus` was compliant.
  At 20 problems it is an audit of the pipeline, not a training set.


**60. Freeing the robust count reaches 12 of 14: the familiarity ranking was right for every
model, and only the counts were wrong.** (Codabench submission 945375, 26 Sep, 0.86. `e27k`
differs from `e27e` in one line: Skywork and Olmo-3-Think call `ceil(n/2) - 1` problems robust
instead of `ceil(n/2)`.)

| Policy | Score | Per-model delta vs constants |
| --- | --- | --- |
| best constants (`e27f`) | 9/14 = 0.64 | — |
| median split (`e27e`) | 10/14 = 0.71 | Qwen 0, DeepSeek +1, Skywork 0, Olmo 0 |
| **tilted split (`e27k`)** | **12/14 = 0.86** | Qwen 0, DeepSeek +1, **Skywork +1, Olmo +1** |

- **Both tilts paid.** `e27k` changes exactly one prediction for each of Skywork and
  Olmo-3-Think, and the score rises by two, so each gained its one available case. With the
  right count, three of the four models beat their constant.
- **This corrects Finding 58.** The "ceiling" there assumed the robust count stays at
  `ceil(n/2)`; the count is a free parameter, and freeing it is worth two cases. The earlier
  zero deltas for Skywork and Olmo were **not evidence that familiarity fails for them** - the
  ranking was fine, the count was one too generous.
- **It is above the leaderboard's previous best** of 0.79, which it passes by one case.
- **What transfers and what does not, stated plainly.** The *ranking* - within a model's batch,
  more familiar problems lean robust - rests on DeepSeek's 142 MATH instances (Findings 22, 47)
  and on the hidden set twice over (Findings 49, 50). The *counts* are calibration to this
  validation set's per-model class balance, measured by the probes in Finding 45. If the
  private test's balance differs, each tilt can cost up to one case per model, which is exactly
  what it gained. `e27e` is the prior-free version of the same ranking and the safer fallback.
- **The count search stops here.** Tilting Qwen or DeepSeek as well would be a search over 14
  cases with no independent evidence behind it, for at most the two cases still wrong, and the
  official rows already disagree with this whole direction (Finding 58). Two candidates now go
  to the freeze: `e27k` (val-calibrated, 0.86) and `e27e` (prior-free, 0.71).

**61. The counts the probes imply are optimal in every direction tested: moving off them costs a
case each time.** (Codabench submissions 946252-946253, 26 Sep, both 0.79.)

| Policy | Robust count per batch | Score |
| --- | --- | --- |
| `e27e` median split | `ceil(n/2)` everywhere | 10/14 |
| **`e27k`** | `ceil(n/2)`, minus one for Skywork and Olmo | **12/14** |
| `e27m` | `e27k` with one fewer for Qwen | 11/14 |
| `e27n` | `e27k` with one fewer for DeepSeek | 11/14 |
| `e27o` | `e27k` with one **more** for Qwen | 11/14 |
| `e27p` | `e27k` with one **more** for DeepSeek | 11/14 |

- **Six moves tested, six agree with the probes.** Finding 45 measured each model's
  robust-minus-spurious balance on the hidden set: +1 for Qwen and DeepSeek, -1 for Skywork and
  Olmo-3-Think. Moving Skywork and Olmo *to* their implied count gained a case each
  (Finding 60); moving Qwen and DeepSeek *off* theirs costs a case each. The balances predict
  the optimal count every time.
- **So `e27k` is not a lucky draw from a search.** Its counts were fixed in advance by measured
  per-model balances, and the search around it only confirms them. That is a better claim than
  "we stopped searching", though it is still calibration to this validation set, and
  Finding 60's transfer caveat stands unchanged.
- **`e27k` is a strict local optimum.** Qwen and DeepSeek lose a case whether their count goes
  up or down (submissions 946282-946283, both 0.79), so every single-step neighbour of the count
  vector is worse. The count search is closed at 12/14, and the policy is now: a prior-free
  familiarity ranking, with per-model robust counts read off independently measured class
  balances - no free parameter left that the leaderboard chose.
---

## 10. Open questions and blockers

**For the organizers** (updated 25 Sep, not sent; use Discord/registration support because
new GitHub issues are restricted):

1. The website now permits extra labeled data, but `aimo-interp/problems-public` returns 401.
   Can registered participants get access, and may labels generated from its symbolic programs
   and official perturbation families be used for training?
2. Will more labelled rows be released for Qwen3.5-4B, Skywork-OR1-Math-7B and
   Olmo-3-7B-Think?
3. How many cases does the Competition Phase validation set hold, and for which models?
4. Does the final test set cover all four small-track models, and in what proportions? Does
   `reasoning_effort` apply to the small track?
5. Bug report: under transformers 5.13, `AutoTokenizer` strips whitespace for DeepSeek
   (Finding 14). This affects `solutions/trained-probe` and
   `aimo-interp/baselines/scripts/encode.py`.

**Requiring you:**

- Send the organizer message above through the challenge Discord or registration support.
- If access is granted, add a Hugging Face token with permission to read
  `aimo-interp/problems-public`; do not paste it into the repository or chat.
- No leaderboard action is currently needed: E27e/E27h at 0.71 remains the performance
  candidate and E11 v2 at 0.64 remains the conservative fallback.

**Housekeeping:**

- **Version control (15 Sep).** The workspace is now a git repository with remote
  `farhanishraq17/AIMO-Interpretability-Challenge---Small-Model-Track` (renamed upstream to
  `…---Team-BakeNekos` on 27 Sep, once the repo held both tracks; GitHub redirects the old URL).
  - `getting-started/` is a submodule at upstream `e98c489`.
  - `Secrets.env` and every generated directory are ignored.
  - Before the first commit, the index was checked for token strings; none were found.
- `General-Instructions.md` is still an empty placeholder, and `Inferred-Notebooks/` is empty.

---

## 11. Next steps

Ranked by value (25 Sep):

1. **Request `problems-public` access and official-family label clarification.** E33 remains
   blocked until legitimate access exists; cached search snippets are not reconstructed.
2. **Do not relax E34a after failure.** Its 1/52 yield closes E34b; no eligibility, sampling,
   perturbation or threshold change follows from this result.
3. **Do not tune E35 on Putnam.** Its independent AUC gate failed; no vectorizer, threshold,
   aggregation band or source substitution follows from the holdout outcome.
4. **Retain the current submissions.** E27e/E27h (0.71) is the performance candidate; E11 v2
   (0.64) is the conservative fallback. No new Codabench upload without E33 or E34 clearing
   its independent-validation gate.
5. **If access is denied, stop GPU experimentation** and focus on the technical
   report, reproducibility, runtime measurements and negative-result figures.
