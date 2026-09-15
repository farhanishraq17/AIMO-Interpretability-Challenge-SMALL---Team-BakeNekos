# Results

## Text-only baseline (no GPU)

Model: `deepseek-ai/DeepSeek-R1-0528-Qwen3-8B`  
Instances: 138 (36 robust / 102 spurious)  
Evaluation: out-of-fold over the 5 problem-grouped folds in `data/small_track/splits.json`  
Features: problem text only, since inference exposes just `{id, model_id, problem}`

| Method | Accuracy | Balanced acc. | ROC-AUC | F1 (robust) | Predicted robust |
| --- | --- | --- | --- | --- | --- |
| always-spurious (majority) | 0.7391 | 0.5000 | 0.5000 | 0.0000 | 0 |
| logreg on handcrafted | 0.6014 | 0.5417 | 0.5417 | 0.3529 | 49 |
| hist-gradient-boosting on handcrafted | 0.7101 | 0.5882 | 0.6757 | 0.3750 | 28 |
| logreg on tf-idf words | 0.6812 | 0.6315 | 0.6773 | 0.4634 | 46 |
| logreg on tf-idf chars | 0.7029 | 0.6732 | 0.6686 | 0.5176 | 49 |

### Reading these numbers

Our training split is 73.9% spurious, so predicting *always spurious* already scores 0.7391 accuracy here. The competition's public validation sample is only 62.5% spurious, so raw accuracy on this table is inflated relative to the leaderboard and **balanced accuracy is the number to track**.

Best text-only method: **logreg on tf-idf chars** at 0.6732 balanced accuracy (0.6686 AUC).

This is the floor that model-internals work has to clear. Any probe or SAE feature set that cannot beat it is not reading robustness - it is reading problem surface form, which text features already capture for free.

### Surface features with the largest weight

| Feature | Coefficient (toward robust) |
| --- | --- |
| `n_brace` | +0.843 |
| `n_words` | +0.679 |
| `kw_integer` | +0.528 |
| `n_backslash` | -0.474 |
| `tex_frac` | -0.414 |
| `kw_how_many` | +0.398 |
| `n_sentences` | -0.356 |
| `kw_greatest` | -0.355 |
| `n_dollar` | -0.327 |
| `tex_pmod` | -0.322 |

Fit on all instances for inspection only, not used for the scores above.

Regenerate with `uv run python ../train/train_text_baseline.py`, then `uv run python ../train/check_dataset_confound.py` to append the confound analysis.

## Is the text baseline just detecting the source dataset?

The concern: the train-ready set is 137 Hendrycks MATH problems and **one** olympiad problem, while the private test set is entirely olympiad. A text model could score well by recognising dataset formatting rather than robustness.

### Base rate by source dataset

| Dataset | Instances | Robust | Rate |
| --- | --- | --- | --- |
| `hendrycks-math-train-reference-500` | 84 | 25 | 0.298 |
| `hendrycks-math-test-reference-100` | 53 | 11 | 0.208 |
| `aimo-2025-reference` | 1 | 0 | 0.000 |

### Performance within a single dataset (confound held constant)

| Dataset | Instances | Balanced acc. | ROC-AUC |
| --- | --- | --- | --- |
| `hendrycks-math-train-reference-500` | 84 | 0.6698 | 0.6976 |
| `hendrycks-math-test-reference-100` | 53 | 0.6439 | 0.6364 |

Within-dataset scores land close to the pooled 0.6732, and the two MATH subsets have similar base rates, so **the signal is not a dataset artifact**. Char n-grams really do carry robustness information about the problem statement.

### The limit of what we can currently measure

Olympiad instances in the train-ready set: **1**. A MATH-to-olympiad transfer test is therefore impossible today, and that transfer is exactly what the competition scores. Closing this gap is the highest-value data work available, ahead of any additional modelling.

Regenerate with `train_text_baseline.py` followed by `check_dataset_confound.py`.

## Stage A: hidden-state probes

Model: `deepseek-ai/DeepSeek-R1-0528-Qwen3-8B`  
Instances: 138 (36 robust / 102 spurious), 37 hidden-state layers x 4096 dims, poolings last, max, mean  
Features: `runs/e1-stagea-deepseek/features/stagea-deepseek`  
Probe: standardised L2 logistic regression, class-balanced, C searched over [0.01]

| Method | Accuracy | Balanced acc. | ROC-AUC | Predicted robust |
| --- | --- | --- | --- | --- |
| always-spurious (majority) | 0.7391 | 0.5000 | 0.5000 | 0 |
| logreg on tf-idf chars (text floor, same instances and folds) | 0.7029 | 0.6732 | 0.6686 | 49 |
| logreg on 5 next-token uncertainty scalars | 0.5797 | 0.5809 | 0.6392 | 64 |
| **nested-selected hidden-state probe** | 0.6812 | 0.5417 | 0.5929 | 26 |
| layer ensemble, last pooling (no selection) | 0.7246 | 0.6340 | 0.6955 | 34 |
| layer ensemble, max pooling (no selection) | 0.7391 | 0.5809 | 0.7214 | 18 |
| layer ensemble, mean pooling (no selection) | 0.7319 | 0.6209 | 0.7320 | 29 |
| layer ensemble, all pooling (no selection) | 0.7174 | 0.6201 | 0.7331 | 33 |
| fusion: text + all-pooling layer ensemble (partner fixed in advance) | 0.7174 | 0.6291 | 0.7260 | 35 |

**Verdict.** By balanced accuracy at the 0.5 threshold, no hidden-state row beats the text floor (0.6732). By ROC-AUC, rows ranking better than text: layer ensemble (last pooling), layer ensemble (max pooling), layer ensemble (mean pooling), layer ensemble (all pooling).

- All-pooling layer ensemble minus text, AUC: +0.0645 (95% paired bootstrap CI -0.0239 to +0.1526).
- Fusion (text + that ensemble) minus text, AUC: +0.0575 (95% paired bootstrap CI -0.0072 to +0.1239).
- An interval spanning zero means the difference is not established at this sample size.
- Hidden-state probabilities sit off-centre (all-pooling ensemble median 0.25; 33 of 138 predicted robust at 0.5), so any shipped version must learn its threshold in-fold.

### Configurations chosen by the inner CV

| Outer fold | Pooling | Layer | C | Inner AUC |
| --- | --- | --- | --- | --- |
| 0 | mean | 23 | 0.01 | 0.8335 |
| 1 | max | 16 | 0.01 | 0.7585 |
| 2 | max | 7 | 0.01 | 0.7552 |
| 3 | mean | 14 | 0.01 | 0.7796 |
| 4 | max | 29 | 0.01 | 0.7645 |

Choices vary across folds, and inner AUCs well above the nested score mean the selection over layers and poolings is fitting noise.

### Per-layer curve (inspection only, optimistic)

| Pooling | Mean bal. acc. over layers | Best layer | Best bal. acc. | Best AUC |
| --- | --- | --- | --- | --- |
| last | 0.5995 | 18 | 0.6618 | 0.7347 |
| max | 0.5833 | 11 | 0.6307 | 0.6939 |
| mean | 0.6187 | 24 | 0.6536 | 0.7478 |

For scale, the shipped last-token probe averages 0.5005 per layer. The best-in-hindsight row is what picking a layer after seeing every fold gives, the same selection effect behind the baseline's 58.4%.
Full curve: `layer_curve.csv`; out-of-fold probabilities: `stagea_oof.npz`. Regenerate with `train/train_stagea_probes.py`.
