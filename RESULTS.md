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
Features: `D:/Project Based Learning/AIMO - Interpretablity/runs/e1b-stagea-deepseek/features/stagea-deepseek`  
Probe: standardised L2 logistic regression, class-balanced, C searched over [0.01]

| Method | Accuracy | Balanced acc. | ROC-AUC | Predicted robust |
| --- | --- | --- | --- | --- |
| always-spurious (majority) | 0.7391 | 0.5000 | 0.5000 | 0 |
| logreg on tf-idf chars (text floor, same instances and folds) | 0.7029 | 0.6732 | 0.6686 | 49 |
| logreg on 5 next-token uncertainty scalars | 0.6522 | 0.6299 | 0.6887 | 54 |
| **nested-selected hidden-state probe** | 0.6957 | 0.5515 | 0.6397 | 24 |
| layer ensemble, last pooling (no selection) | 0.7174 | 0.6381 | 0.6966 | 37 |
| layer ensemble, max pooling (no selection) | 0.7101 | 0.5613 | 0.7538 | 22 |
| layer ensemble, mean pooling (no selection) | 0.7319 | 0.6479 | 0.7481 | 35 |
| layer ensemble, all pooling (no selection) | 0.7101 | 0.5972 | 0.7489 | 30 |
| fusion: text + all-pooling layer ensemble (partner fixed in advance) | 0.7101 | 0.6332 | 0.7342 | 38 |

**Verdict.** By balanced accuracy at the 0.5 threshold, no hidden-state row beats the text floor (0.6732). By ROC-AUC, rows ranking better than text: layer ensemble (last pooling), layer ensemble (max pooling), layer ensemble (mean pooling), layer ensemble (all pooling).

- All-pooling layer ensemble minus text, AUC: +0.0803 (95% paired bootstrap CI -0.0089 to +0.1675).
- Fusion (text + that ensemble) minus text, AUC: +0.0656 (95% paired bootstrap CI -0.0003 to +0.1291).
- An interval spanning zero means the difference is not established at this sample size.
- Hidden-state probabilities sit off-centre (all-pooling ensemble median 0.24; 30 of 138 predicted robust at 0.5), so any shipped version must learn its threshold in-fold.

### Configurations chosen by the inner CV

| Outer fold | Pooling | Layer | C | Inner AUC |
| --- | --- | --- | --- | --- |
| 0 | max | 27 | 0.01 | 0.8520 |
| 1 | max | 16 | 0.01 | 0.7678 |
| 2 | max | 17 | 0.01 | 0.7771 |
| 3 | mean | 16 | 0.01 | 0.7931 |
| 4 | max | 15 | 0.01 | 0.7455 |

Choices vary across folds, and inner AUCs well above the nested score mean the selection over layers and poolings is fitting noise.

### Per-layer curve (inspection only, optimistic)

| Pooling | Mean bal. acc. over layers | Best layer | Best bal. acc. | Best AUC |
| --- | --- | --- | --- | --- |
| last | 0.6031 | 9 | 0.6471 | 0.7075 |
| max | 0.5865 | 17 | 0.6732 | 0.7723 |
| mean | 0.6371 | 10 | 0.6716 | 0.7514 |

For scale, the shipped last-token probe averages 0.5005 per layer. The best-in-hindsight row is what picking a layer after seeing every fold gives, the same selection effect behind the baseline's 58.4%.
Full curve: `layer_curve.csv`; out-of-fold probabilities: `stagea_oof.npz`. Regenerate with `train/train_stagea_probes.py`.

## Stage B: generation-trace features

Model: `deepseek-ai/DeepSeek-R1-0528-Qwen3-8B`  
Instances: 138 (36 robust / 102 spurious)  
Traces: greedy, 4096-token cap; 55.1% hit the cap; median tokens robust=4096 vs spurious=3936  
Evaluation: out-of-fold over the problem-grouped folds; fusion averages OOF probabilities with fixed weights

| Method | Accuracy | Balanced acc. | ROC-AUC | Predicted robust |
| --- | --- | --- | --- | --- |
| always-spurious (majority) | 0.7391 | 0.5000 | 0.5000 | 0 |
| logreg on tf-idf chars (text floor, this subset) | 0.7029 | 0.6732 | 0.6686 | 49 |
| logreg on 14 uncertainty | 0.5290 | 0.5286 | 0.5392 | 67 |
| hist-gradient-boosting on 14 uncertainty | 0.6377 | 0.5752 | 0.5487 | 46 |
| random-forest on 14 uncertainty | 0.6739 | 0.5278 | 0.5125 | 25 |
| logreg on 14 uncertainty + length | 0.5362 | 0.6054 | 0.6176 | 82 |
| hist-gradient-boosting on 14 uncertainty + length | 0.6812 | 0.6136 | 0.6147 | 42 |
| random-forest on 14 uncertainty + length | 0.7029 | 0.5654 | 0.5651 | 25 |
| logreg on 14 uncertainty + length + trace shape | 0.5725 | 0.5940 | 0.6484 | 69 |
| hist-gradient-boosting on 14 uncertainty + length + trace shape | 0.6304 | 0.5523 | 0.6119 | 43 |
| random-forest on 14 uncertainty + length + trace shape | 0.6884 | 0.5286 | 0.6133 | 21 |
| **fusion: text + logreg on 14 uncertainty + length + trace shape** | 0.6594 | 0.6438 | 0.6874 | 55 |

The best trace model is picked by AUC across the rows above, so the fusion row carries a small selection effect; treat differences under ~0.03 as noise at this sample size.

### Univariate AUC per feature (direction: >0.5 means higher values lean robust)

| Feature | AUC |
| --- | --- |
| `trace_reflections_per_1k_words` | 0.314 |
| `trace_log_chars` | 0.640 |
| `trace_closed_think` | 0.364 |
| `log_generation_num_tokens` | 0.636 |
| `hit_token_cap` | 0.616 |
| `trace_has_boxed` | 0.384 |
| `generation_std_logprob` | 0.588 |
| `generation_mean_entropy` | 0.587 |
| `trace_reflections` | 0.420 |
| `generation_min_k_logprob` | 0.422 |
| `generation_mean_nll` | 0.576 |
| `generation_ppl` | 0.576 |

Reference: text floor on the full 138 is 0.6732 balanced accuracy. Regenerate with `train/train_stageb_features.py`.

## E10: fusion v1

Model: `deepseek-ai/DeepSeek-R1-0528-Qwen3-8B`  
Instances: 138 of 138 with every block (36 robust)  
Blocks: text, stagea, trace  
Protocol: nested grouped CV; block scores standardised on inner folds; thresholds learned on inner folds (a score >= 0 is robust); fusions fixed in advance

| Method | Accuracy | Balanced acc. | ROC-AUC | Predicted robust |
| --- | --- | --- | --- | --- |
| always-spurious (majority) | 0.7391 | 0.5000 | 0.5000 | 0 |
| text alone | 0.6522 | 0.6389 | 0.6481 | 56 |
| stagea alone | 0.6014 | 0.6046 | 0.7059 | 63 |
| trace alone | 0.5507 | 0.5792 | 0.6334 | 72 |
| mean fusion: text + stagea | 0.6884 | 0.7173 | 0.7230 | 63 |
| mean fusion: text + trace | 0.6232 | 0.6103 | 0.6419 | 58 |
| mean fusion: stagea + trace | 0.6522 | 0.6928 | 0.7571 | 68 |
| **mean fusion: all blocks** | 0.6667 | 0.6667 | 0.7176 | 58 |
| **stacked fusion: all blocks** | 0.6377 | 0.6471 | 0.7285 | 62 |

### Difference from text alone (95% paired bootstrap CI)

| Row | AUC difference | Balanced-accuracy difference |
| --- | --- | --- |
| stagea alone | +0.0577 [-0.0366, +0.1494] | -0.0343 [-0.1388, +0.0642] |
| trace alone | -0.0147 [-0.1456, +0.1129] | -0.0596 [-0.1806, +0.0587] |
| mean fusion: text + stagea | +0.0749 [+0.0308, +0.1174] | +0.0784 [+0.0145, +0.1477] |
| mean fusion: text + trace | -0.0063 [-0.0540, +0.0383] | -0.0286 [-0.1102, +0.0513] |
| mean fusion: stagea + trace | +0.1089 [+0.0154, +0.1992] | +0.0539 [-0.0390, +0.1411] |
| **mean fusion: all blocks** | +0.0694 [+0.0186, +0.1209] | +0.0278 [-0.0472, +0.1035] |
| **stacked fusion: all blocks** | +0.0803 [-0.0201, +0.1751] | +0.0082 [-0.0899, +0.0997] |

**Verdict:** relative to text alone, fusing all blocks changes AUC by +0.0694 [+0.0186, +0.1209] (established gain) and balanced accuracy by +0.0278 [-0.0472, +0.1035] (not established) at n=138.

Text alone here uses a learned threshold, so it can differ from the 0.5-threshold floor above. Pair rows are ablations; the all-block rows are the pre-declared result. Regenerate with `train/train_fusion.py`.

## E4: trace-state probes

Model: `deepseek-ai/DeepSeek-R1-0528-Qwen3-8B`  
Instances: 138 (36 robust / 102 spurious), 37 hidden-state layers x 4096 dims, poolings prompt_last, trace_first_third, trace_last, trace_last_third, trace_mean  
Features: `runs/e4-tracestate-deepseek/features/tracestate-deepseek`  
Probe: standardised L2 logistic regression, class-balanced, C searched over [0.01]

| Method | Accuracy | Balanced acc. | ROC-AUC | Predicted robust |
| --- | --- | --- | --- | --- |
| always-spurious (majority) | 0.7391 | 0.5000 | 0.5000 | 0 |
| logreg on tf-idf chars (text floor, same instances and folds) | 0.7029 | 0.6732 | 0.6686 | 49 |
| **nested-selected hidden-state probe** | 0.6739 | 0.5368 | 0.6245 | 27 |
| layer ensemble, prompt_last pooling (no selection) | 0.7391 | 0.6618 | 0.7386 | 36 |
| layer ensemble, trace_first_third pooling (no selection) | 0.7464 | 0.6397 | 0.7394 | 29 |
| layer ensemble, trace_last pooling (no selection) | 0.7029 | 0.5114 | 0.6040 | 13 |
| layer ensemble, trace_last_third pooling (no selection) | 0.6304 | 0.5163 | 0.6253 | 35 |
| layer ensemble, trace_mean pooling (no selection) | 0.7029 | 0.6013 | 0.6860 | 33 |
| layer ensemble, all pooling (no selection) | 0.6884 | 0.5645 | 0.7219 | 29 |
| fusion: text + all-pooling layer ensemble (partner fixed in advance) | 0.7174 | 0.6021 | 0.7274 | 29 |

**Verdict.** By balanced accuracy at the 0.5 threshold, no hidden-state row beats the text floor (0.6732). By ROC-AUC, rows ranking better than text: layer ensemble (prompt_last pooling), layer ensemble (trace_first_third pooling), layer ensemble (trace_mean pooling), layer ensemble (all pooling).

- All-pooling layer ensemble minus text, AUC: +0.0534 (95% paired bootstrap CI -0.0461 to +0.1528).
- Fusion (text + that ensemble) minus text, AUC: +0.0588 (95% paired bootstrap CI -0.0156 to +0.1370).
- An interval spanning zero means the difference is not established at this sample size.
- Hidden-state probabilities sit off-centre (all-pooling ensemble median 0.27; 29 of 138 predicted robust at 0.5), so any shipped version must learn its threshold in-fold.

### Configurations chosen by the inner CV

| Outer fold | Pooling | Layer | C | Inner AUC |
| --- | --- | --- | --- | --- |
| 0 | prompt_last | 16 | 0.01 | 0.8389 |
| 1 | prompt_last | 26 | 0.01 | 0.7653 |
| 2 | trace_last | 20 | 0.01 | 0.6747 |
| 3 | prompt_last | 31 | 0.01 | 0.7769 |
| 4 | prompt_last | 9 | 0.01 | 0.7342 |

Choices vary across folds, and inner AUCs well above the nested score mean the selection over layers and poolings is fitting noise.

### Per-layer curve (inspection only, optimistic)

| Pooling | Mean bal. acc. over layers | Best layer | Best bal. acc. | Best AUC |
| --- | --- | --- | --- | --- |
| prompt_last | 0.6342 | 21 | 0.6895 | 0.7239 |
| trace_first_third | 0.6227 | 16 | 0.6675 | 0.7135 |
| trace_last | 0.5146 | 3 | 0.5842 | 0.5539 |
| trace_last_third | 0.5370 | 25 | 0.5866 | 0.6694 |
| trace_mean | 0.5894 | 12 | 0.6381 | 0.6947 |

For scale, the shipped last-token probe averages 0.5005 per layer. The best-in-hindsight row is what picking a layer after seeing every fold gives, the same selection effect behind the baseline's 58.4%.
Full curve: `layer_curve.csv`; out-of-fold probabilities: `stagea_oof.npz`. Regenerate with `train/train_stagea_probes.py`.

## E4: fusion with trace states

Model: `deepseek-ai/DeepSeek-R1-0528-Qwen3-8B`  
Instances: 138 of 138 with every block (36 robust)  
Blocks: text, stagea, tracestate  
Protocol: nested grouped CV; block scores standardised on inner folds; thresholds learned on inner folds (a score >= 0 is robust); fusions fixed in advance

| Method | Accuracy | Balanced acc. | ROC-AUC | Predicted robust |
| --- | --- | --- | --- | --- |
| always-spurious (majority) | 0.7391 | 0.5000 | 0.5000 | 0 |
| text alone | 0.6522 | 0.6389 | 0.6481 | 56 |
| stagea alone | 0.6014 | 0.6046 | 0.7059 | 63 |
| tracestate alone | 0.6087 | 0.6454 | 0.7037 | 70 |
| mean fusion: text + stagea | 0.6884 | 0.7173 | 0.7230 | 63 |
| mean fusion: text + tracestate | 0.6377 | 0.6830 | 0.7179 | 70 |
| mean fusion: stagea + tracestate | 0.6377 | 0.6920 | 0.7505 | 72 |
| **mean fusion: all blocks** | 0.6667 | 0.7116 | 0.7391 | 68 |
| **stacked fusion: all blocks** | 0.6522 | 0.6658 | 0.7209 | 62 |

### Difference from text alone (95% paired bootstrap CI)

| Row | AUC difference | Balanced-accuracy difference |
| --- | --- | --- |
| stagea alone | +0.0577 [-0.0366, +0.1494] | -0.0343 [-0.1388, +0.0642] |
| tracestate alone | +0.0556 [-0.0383, +0.1556] | +0.0065 [-0.1021, +0.1138] |
| mean fusion: text + stagea | +0.0749 [+0.0308, +0.1174] | +0.0784 [+0.0145, +0.1477] |
| mean fusion: text + tracestate | +0.0697 [+0.0179, +0.1265] | +0.0441 [-0.0241, +0.1182] |
| mean fusion: stagea + tracestate | +0.1024 [+0.0219, +0.1881] | +0.0531 [-0.0419, +0.1468] |
| **mean fusion: all blocks** | +0.0910 [+0.0309, +0.1518] | +0.0727 [+0.0029, +0.1501] |
| **stacked fusion: all blocks** | +0.0727 [-0.0262, +0.1704] | +0.0270 [-0.0739, +0.1250] |

**Verdict:** relative to text alone, fusing all blocks changes AUC by +0.0910 [+0.0309, +0.1518] (established gain) and balanced accuracy by +0.0727 [+0.0029, +0.1501] (established gain) at n=138.

Text alone here uses a learned threshold, so it can differ from the 0.5-threshold floor above. Pair rows are ablations; the all-block rows are the pre-declared result. Regenerate with `train/train_fusion.py`.
