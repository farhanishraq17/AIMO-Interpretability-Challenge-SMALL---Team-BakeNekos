# Results

## Text-only baseline (no GPU)

Model: `deepseek-ai/DeepSeek-R1-0528-Qwen3-8B`  
Instances: 152 (44 robust / 108 spurious)  
Evaluation: out-of-fold over the 5 problem-grouped folds in `data/small_track/splits.json`  
Features: problem text only, since inference exposes just `{id, model_id, problem}`

| Method | Accuracy | Balanced acc. | ROC-AUC | F1 (robust) | Predicted robust |
| --- | --- | --- | --- | --- | --- |
| always-spurious (majority) | 0.7105 | 0.5000 | 0.5000 | 0.0000 | 0 |
| logreg on handcrafted | 0.6118 | 0.5787 | 0.5814 | 0.4272 | 59 |
| hist-gradient-boosting on handcrafted | 0.7237 | 0.6372 | 0.7024 | 0.4750 | 36 |
| logreg on tf-idf words | 0.6711 | 0.6406 | 0.6625 | 0.5000 | 56 |
| logreg on tf-idf chars | 0.6711 | 0.6406 | 0.6294 | 0.5000 | 56 |

### Reading these numbers

Our training split is 71.1% spurious, so predicting *always spurious* already scores 0.7105 accuracy here. The competition's public validation sample is only 62.5% spurious, so raw accuracy on this table is inflated relative to the leaderboard and **balanced accuracy is the number to track**.

Best text-only method: **logreg on tf-idf words** at 0.6406 balanced accuracy (0.6625 AUC).

This is the floor that model-internals work has to clear. Any probe or SAE feature set that cannot beat it is not reading robustness - it is reading problem surface form, which text features already capture for free.

### Surface features with the largest weight

| Feature | Coefficient (toward robust) |
| --- | --- |
| `n_backslash` | -0.851 |
| `n_words` | +0.774 |
| `n_brace` | +0.705 |
| `tex_log` | +0.595 |
| `tex_triangle` | -0.514 |
| `kw_integer` | +0.503 |
| `n_dollar` | -0.390 |
| `type_token_ratio` | -0.383 |
| `kw_how_many` | +0.367 |
| `kw_greatest` | -0.346 |

Fit on all instances for inspection only, not used for the scores above.

Regenerate with `uv run python ../train/train_text_baseline.py`, then `uv run python ../train/check_dataset_confound.py` to append the confound analysis.

## Is the text baseline just detecting the source dataset?

The concern: the train-ready set is mostly Hendrycks MATH problems while the private test set is entirely olympiad. A text model could score well by recognising dataset formatting rather than robustness.

### Base rate by source dataset

| Dataset | Instances | Robust | Rate |
| --- | --- | --- | --- |
| `hendrycks-math-train-reference-500` | 84 | 25 | 0.298 |
| `hendrycks-math-test-reference-100` | 58 | 14 | 0.241 |
| `train-main-v2` | 8 | 5 | 0.625 |
| `aimo-2025-reference` | 2 | 0 | 0.000 |

### Performance within a single dataset (confound held constant)

| Dataset | Instances | Balanced acc. | ROC-AUC |
| --- | --- | --- | --- |
| `hendrycks-math-train-reference-500` | 84 | 0.6698 | 0.6976 |
| `hendrycks-math-test-reference-100` | 58 | 0.6477 | 0.6185 |

Within-dataset scores land close to the pooled number in the table above, and the MATH subsets have similar base rates, so **the signal is not a dataset artifact** within MATH. Char n-grams do carry robustness information about the problem statement.

### MATH to olympiad: the transfer the competition actually scores

Char tf-idf trained on the 142 MATH instances and tested on the 10 olympiad instances (5 robust): **balanced accuracy 0.3000, ROC-AUC 0.2800**. Below 0.5 means the text signal does not survive the shift in difficulty, which is exactly what the private test set asks of it. At this sample size treat it as a warning, not a measurement.

Regenerate with `train_text_baseline.py` followed by `check_dataset_confound.py`.

## Stage A: hidden-state probes

Model: `deepseek-ai/DeepSeek-R1-0528-Qwen3-8B`  
Instances: 138 (36 robust / 102 spurious), 37 hidden-state layers x 4096 dims, poolings last, max, mean  
Features: `runs/e1b-stagea-deepseek/features/stagea-deepseek`  
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
Instances: 138 (36 robust / 102 spurious), **14 still missing**  
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
Instances: 138 of 152 with every block (36 robust)  
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
Instances: 138 of 152 with every block (36 robust)  
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

## E14: fusion with test-time agreement

Model: `deepseek-ai/DeepSeek-R1-0528-Qwen3-8B`  
Instances: 138 of 152 with every block (36 robust)  
Blocks: text, stagea, agreement  
Protocol: nested grouped CV; block scores standardised on inner folds; thresholds learned on inner folds (a score >= 0 is robust); fusions fixed in advance

| Method | Accuracy | Balanced acc. | ROC-AUC | Predicted robust |
| --- | --- | --- | --- | --- |
| always-spurious (majority) | 0.7391 | 0.5000 | 0.5000 | 0 |
| text alone | 0.6522 | 0.6389 | 0.6481 | 56 |
| stagea alone | 0.6014 | 0.6046 | 0.7059 | 63 |
| agreement alone | 0.5942 | 0.6626 | 0.6955 | 78 |
| mean fusion: text + stagea | 0.6884 | 0.7173 | 0.7230 | 63 |
| mean fusion: text + agreement | 0.6159 | 0.6683 | 0.7290 | 73 |
| mean fusion: stagea + agreement | 0.6232 | 0.7181 | 0.7563 | 82 |
| **mean fusion: all blocks** | 0.6522 | 0.6928 | 0.7320 | 68 |
| **stacked fusion: all blocks** | 0.6522 | 0.6748 | 0.7092 | 64 |

### Difference from text alone (95% paired bootstrap CI)

| Row | AUC difference | Balanced-accuracy difference |
| --- | --- | --- |
| stagea alone | +0.0577 [-0.0366, +0.1494] | -0.0343 [-0.1388, +0.0642] |
| agreement alone | +0.0474 [-0.0879, +0.1881] | +0.0237 [-0.1019, +0.1546] |
| mean fusion: text + stagea | +0.0749 [+0.0308, +0.1174] | +0.0784 [+0.0145, +0.1477] |
| mean fusion: text + agreement | +0.0809 [-0.0145, +0.1770] | +0.0294 [-0.0537, +0.1171] |
| mean fusion: stagea + agreement | +0.1081 [+0.0047, +0.2107] | +0.0792 [-0.0185, +0.1794] |
| **mean fusion: all blocks** | +0.0839 [+0.0229, +0.1463] | +0.0539 [-0.0255, +0.1405] |
| **stacked fusion: all blocks** | +0.0610 [-0.0422, +0.1628] | +0.0359 [-0.0601, +0.1303] |

**Verdict:** relative to text alone, fusing all blocks changes AUC by +0.0839 [+0.0229, +0.1463] (established gain) and balanced accuracy by +0.0539 [-0.0255, +0.1405] (not established) at n=138.

Text alone here uses a learned threshold, so it can differ from the 0.5-threshold floor above. Pair rows are ablations; the all-block rows are the pre-declared result. Regenerate with `train/train_fusion.py`.

## E12 label-free signals (DeepSeek)

138 DeepSeek training problems (36 robust). AUC > 0.5 means higher values indicate robust. The CI is a paired bootstrap against a constant score (AUC 0.5); rows where a signal is undefined are dropped for that signal.

| Signal | n | ROC-AUC | 95% CI vs chance | AUC within length quintiles |
| --- | --- | --- | --- | --- |
| variant_agree | 101 | 0.5065 | [0.3907, 0.6174] | 0.4967 |
| variant_kl | 101 | 0.5628 | [0.4425, 0.6806] | 0.5654 |
| answer_entropy | 138 | 0.6119 | [0.5158, 0.7042] | 0.4918 |
| answer_max_prob | 138 | 0.3922 | [0.2987, 0.4890] | 0.4934 |
| problem_nll | 138 | 0.3235 | [0.2309, 0.4299] | 0.4194 |
| next_token_entropy | 138 | 0.7361 | [0.6399, 0.8165] | 0.5872 |
| next_token_max_prob | 138 | 0.5000 | [0.5000, 0.5000] | 0.5000 |
| next_token_logit_margin | 138 | 0.2857 | [0.2017, 0.3876] | 0.4169 |
| problem_length | 138 | 0.7337 | [0.6491, 0.8110] | — |

Immediate answers stable across all rewrites: 65.3% of 101 problems. Spearman correlation with problem length: problem_nll -0.50, next_token_entropy +0.67, next_token_logit_margin -0.59, answer_entropy +0.43.

### Cross-model transfer: other models' signals against DeepSeek's labels

| Scored by | n | problem_nll | next_token_entropy | next_token_logit_margin | answer_entropy | Spearman with DeepSeek's next_token_entropy |
| --- | --- | --- | --- | --- | --- | --- |
| `allenai/Olmo-3-7B-Think` | 138 | 0.2985 | 0.5929 | 0.4127 | 0.5376 | +0.55 |
| `Qwen/Qwen3.5-4B` | 138 | 0.3306 | 0.5057 | 0.5162 | 0.5648 | -0.06 |
| `Skywork/Skywork-OR1-Math-7B` | 138 | 0.2922 | 0.4281 | 0.5347 | 0.6411 | +0.06 |

Labels are DeepSeek's throughout; no other model has labels.

## E14: test-time answer agreement (DeepSeek)

138 DeepSeek training problems (36 robust). AUC > 0.5 means higher values indicate robust. The CI is a paired bootstrap against a constant score (AUC 0.5); rows where a signal is undefined are dropped for that signal.

| Signal | n | ROC-AUC | 95% CI vs chance | AUC within length quintiles |
| --- | --- | --- | --- | --- |
| rewrite_agree | 121 | 0.4577 | [0.3627, 0.5489] | 0.5430 |
| rewrite_min_agree | 121 | 0.4681 | [0.3754, 0.5548] | 0.5656 |
| orig_self_agree | 121 | 0.4551 | [0.3733, 0.5244] | 0.4842 |
| orig_answered | 138 | 0.4372 | [0.3494, 0.5165] | 0.5121 |
| orig_tokens | 138 | 0.6520 | [0.5574, 0.7439] | 0.4968 |
| rewrite_token_ratio | 138 | 0.4658 | [0.3449, 0.5775] | 0.5307 |
| hit_length_frac | 138 | 0.6057 | [0.5108, 0.7036] | 0.4879 |
| agree_latex_to_plaintext | 85 | 0.4743 | [0.3933, 0.5427] | 0.5130 |
| agree_premise_reorder | 5 | 0.6667 | [0.5000, 1.0000] | nan |
| agree_typos | 55 | 0.5044 | [0.4086, 0.5875] | 0.5047 |
| agree_distract | 121 | 0.4500 | [0.3576, 0.5375] | 0.5305 |
| problem_length | 138 | 0.7337 | [0.6464, 0.8100] | — |

Setup: k=2 samples per prompt, 8192-token cap, vllm 0.29.0 in float16 on T4s. 447 prompts (309 rewrites: latex_to_plaintext 85, premise_reorder 5, typos 55, distract 121).
Anchor answer found for 121 of 138 problems; every rewrite sample matched it for 78.5% of them.
Cost: 3.86M generated tokens, 27,988 per problem, 10.1 GPU-session hours (107 tok/s on 2x T4).

## E15-E19: self-labelling progress

Olympiad pool: 505 problems (AMC, AIME, OlympiadBench). k=4 samples per problem; vLLM, fp16, 2x T4. Eligible means all 4 base samples correct, the organizers' rule. Only `answer_modulus` variants (compliance `distinct`) are sampled for labels, per the extra-data rule.

### Base phase

| Model | Sampled | Eligible | Rate | Samples hitting the cap | Median tokens per sample | AMC eligible/sampled | AIME eligible/sampled | Olympiad eligible/sampled |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `Qwen/Qwen3.5-4B` | 276 of 505 | 88 | 32% | 59% | 16,384 | 2/11 | 2/33 | 84/232 |
| `Skywork/Skywork-OR1-Math-7B` | 232 of 505 | 143 | 62% | 21% | 7,378 | 6/9 | 9/26 | 128/197 |
| `allenai/Olmo-3-7B-Think` | 28 of 505 | 15 | 54% | 41% | 9,278 | 1/2 | 1/3 | 13/23 |

### Variants phase and labels

| Model | Compliant variants of eligible problems | Problems covered | Variants sampled | Problems labelled | Robust (all) | Clear-cut (training) | Robust (clear-cut) |
| --- | --- | --- | --- | --- | --- | --- | --- |
| `Qwen/Qwen3.5-4B` | 100 | 53 | 89 | 47 | 19 (40%) | 26 | 19 |
| `Skywork/Skywork-OR1-Math-7B` | 176 | 91 | 176 | 21 | 17 (81%) | 19 | 17 |
| `allenai/Olmo-3-7B-Think` | 21 | 11 | 0 | — | — | — | — |

Labels come from `datagen/perturb/label.py`: robust iff decay <= 0, where decay pools every sample of every variant; clear-cut drops the ambiguous 0 < decay <= 0.3 band, as the public data does. With 4 samples per variant, one wrong sample already lands a problem in that band. 34 AMC pool ids that the old pool reused for two problems are excluded (Finding 29). Olmo-3-7B-Think has no variants run: on 2x T4 it fits only at a 12k cap (Finding 26). Regenerate with `label.py`, then `train/summarize_labelling.py`.
