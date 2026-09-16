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

- **Version control (15 Sep).** The workspace is now a git repository with remote
  `farhanishraq17/AIMO-Interpretability-Challenge---Small-Model-Track`.
  - `getting-started/` is a submodule at upstream `e98c489`.
  - `Secrets.env` and every generated directory are ignored.
  - Before the first commit, the index was checked for token strings; none were found.
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
