# Results

## Text-only baseline (no GPU)

Model: `deepseek-ai/DeepSeek-R1-0528-Qwen3-8B`  
Instances: 153 (46 robust / 107 spurious)  
Evaluation: out-of-fold over the 5 problem-grouped folds in `data/small_track/splits.json`  
Features: problem text only, since inference exposes just `{id, model_id, problem}`

| Method | Accuracy | Balanced acc. | ROC-AUC | F1 (robust) | Predicted robust |
| --- | --- | --- | --- | --- | --- |
| always-spurious (majority) | 0.6993 | 0.5000 | 0.5000 | 0.0000 | 0 |
| logreg on handcrafted | 0.6209 | 0.5741 | 0.5937 | 0.4200 | 54 |
| hist-gradient-boosting on handcrafted | 0.7124 | 0.6457 | 0.7233 | 0.5000 | 42 |
| logreg on tf-idf words | 0.6601 | 0.6393 | 0.6770 | 0.5094 | 60 |
| logreg on tf-idf chars | 0.6601 | 0.6331 | 0.6491 | 0.5000 | 58 |

### Reading these numbers

Our training split is 69.9% spurious, so predicting *always spurious* already scores 0.6993 accuracy here. The competition's public validation sample is only 62.5% spurious, so raw accuracy on this table is inflated relative to the leaderboard and **balanced accuracy is the number to track**.

Best text-only method: **hist-gradient-boosting on handcrafted** at 0.6457 balanced accuracy (0.7233 AUC).

This is the floor that model-internals work has to clear. Any probe or SAE feature set that cannot beat it is not reading robustness - it is reading problem surface form, which text features already capture for free.

### Surface features with the largest weight

| Feature | Coefficient (toward robust) |
| --- | --- |
| `n_words` | +0.859 |
| `n_backslash` | -0.836 |
| `tex_triangle` | -0.578 |
| `kw_how_many` | +0.573 |
| `tex_log` | +0.534 |
| `n_underscore` | +0.521 |
| `kw_integer` | +0.480 |
| `n_brace` | +0.430 |
| `type_token_ratio` | -0.415 |
| `tex_pmod` | -0.289 |

Fit on all instances for inspection only, not used for the scores above.

Regenerate with `uv run python ../train/train_text_baseline.py`, then `uv run python ../train/check_dataset_confound.py` to append the confound analysis.

## Is the text baseline just detecting the source dataset?

The concern: the train-ready set is mostly Hendrycks MATH problems while the private test set is entirely olympiad. A text model could score well by recognising dataset formatting rather than robustness.

### Base rate by source dataset

| Dataset | Instances | Robust | Rate |
| --- | --- | --- | --- |
| `hendrycks-math-train-reference-500` | 84 | 25 | 0.298 |
| `hendrycks-math-test-reference-100` | 58 | 14 | 0.241 |
| `train-main-v2` | 10 | 7 | 0.700 |
| `aimo-2025-reference` | 1 | 0 | 0.000 |

### Performance within a single dataset (confound held constant)

| Dataset | Instances | Balanced acc. | ROC-AUC |
| --- | --- | --- | --- |
| `hendrycks-math-train-reference-500` | 84 | 0.6698 | 0.6976 |
| `hendrycks-math-test-reference-100` | 58 | 0.6477 | 0.6185 |

Within-dataset scores land close to the pooled number in the table above, and the MATH subsets have similar base rates, so **the signal is not a dataset artifact** within MATH. Char n-grams do carry robustness information about the problem statement.

### MATH to olympiad: the transfer the competition actually scores

Char tf-idf trained on the 142 MATH instances and tested on the 11 olympiad instances (7 robust): **balanced accuracy 0.4107, ROC-AUC 0.4286**. Below 0.5 means the text signal does not survive the shift in difficulty, which is exactly what the private test set asks of it. At this sample size treat it as a warning, not a measurement.

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
Instances: 138 (36 robust / 102 spurious), **15 still missing**  
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

## E10: fusion v1

Model: `deepseek-ai/DeepSeek-R1-0528-Qwen3-8B`  
Instances: 138 of 153 with every block (37 robust)  
Blocks: text, stagea  
Protocol: nested grouped CV; block scores standardised on inner folds; thresholds learned on inner folds (a score >= 0 is robust); fusions fixed in advance

| Method | Accuracy | Balanced acc. | ROC-AUC | Predicted robust |
| --- | --- | --- | --- | --- |
| always-spurious (majority) | 0.7319 | 0.5000 | 0.5000 | 0 |
| text alone | 0.6739 | 0.6573 | 0.6794 | 54 |
| stagea alone | 0.6232 | 0.6569 | 0.7490 | 69 |
| **mean fusion: all blocks** | 0.6739 | 0.6916 | 0.7305 | 62 |
| **stacked fusion: all blocks** | 0.6594 | 0.6903 | 0.7214 | 66 |

### Difference from text alone (95% paired bootstrap CI)

| Row | AUC difference | Balanced-accuracy difference |
| --- | --- | --- |
| stagea alone | +0.0696 [-0.0150, +0.1544] | -0.0004 [-0.0917, +0.0926] |
| **mean fusion: all blocks** | +0.0511 [+0.0103, +0.0953] | +0.0343 [-0.0342, +0.1013] |
| **stacked fusion: all blocks** | +0.0420 [-0.0465, +0.1298] | +0.0329 [-0.0629, +0.1273] |

**Verdict:** relative to text alone, fusing all blocks changes AUC by +0.0511 [+0.0103, +0.0953] (established gain) and balanced accuracy by +0.0343 [-0.0342, +0.1013] (not established) at n=138.

Text alone here uses a learned threshold, so it can differ from the 0.5-threshold floor above. Pair rows are ablations; the all-block rows are the pre-declared result. Regenerate with `train/train_fusion.py`.

## E4: fusion with trace states

Model: `deepseek-ai/DeepSeek-R1-0528-Qwen3-8B`  
Instances: 138 of 153 with every block (37 robust)  
Blocks: text, stagea, tracestate, trace  
Protocol: nested grouped CV; block scores standardised on inner folds; thresholds learned on inner folds (a score >= 0 is robust); fusions fixed in advance

| Method | Accuracy | Balanced acc. | ROC-AUC | Predicted robust |
| --- | --- | --- | --- | --- |
| always-spurious (majority) | 0.7319 | 0.5000 | 0.5000 | 0 |
| text alone | 0.6739 | 0.6573 | 0.6794 | 54 |
| stagea alone | 0.6232 | 0.6569 | 0.7490 | 69 |
| tracestate alone | 0.6232 | 0.6826 | 0.7198 | 75 |
| trace alone | 0.5507 | 0.5732 | 0.6064 | 71 |
| mean fusion: text + stagea | 0.6739 | 0.6916 | 0.7305 | 62 |
| mean fusion: text + tracestate | 0.6232 | 0.6569 | 0.7244 | 69 |
| mean fusion: text + trace | 0.6159 | 0.6092 | 0.6489 | 60 |
| mean fusion: stagea + tracestate | 0.6522 | 0.7281 | 0.7584 | 77 |
| mean fusion: stagea + trace | 0.6522 | 0.7024 | 0.7482 | 71 |
| mean fusion: tracestate + trace | 0.5942 | 0.6543 | 0.6941 | 77 |
| **mean fusion: all blocks** | 0.6594 | 0.7245 | 0.7410 | 74 |
| **stacked fusion: all blocks** | 0.6522 | 0.6596 | 0.7022 | 61 |

### Difference from text alone (95% paired bootstrap CI)

| Row | AUC difference | Balanced-accuracy difference |
| --- | --- | --- |
| stagea alone | +0.0696 [-0.0150, +0.1544] | -0.0004 [-0.0917, +0.0926] |
| tracestate alone | +0.0404 [-0.0651, +0.1502] | +0.0253 [-0.0770, +0.1307] |
| trace alone | -0.0731 [-0.2100, +0.0716] | -0.0842 [-0.2153, +0.0464] |
| mean fusion: text + stagea | +0.0511 [+0.0103, +0.0953] | +0.0343 [-0.0342, +0.1013] |
| mean fusion: text + tracestate | +0.0450 [-0.0269, +0.1147] | -0.0004 [-0.0855, +0.0798] |
| mean fusion: text + trace | -0.0305 [-0.0870, +0.0212] | -0.0482 [-0.1253, +0.0345] |
| mean fusion: stagea + tracestate | +0.0789 [-0.0153, +0.1793] | +0.0708 [-0.0197, +0.1666] |
| mean fusion: stagea + trace | +0.0688 [-0.0292, +0.1674] | +0.0451 [-0.0612, +0.1509] |
| mean fusion: tracestate + trace | +0.0147 [-0.1074, +0.1398] | -0.0031 [-0.1165, +0.1140] |
| **mean fusion: all blocks** | +0.0615 [-0.0153, +0.1385] | +0.0672 [-0.0195, +0.1552] |
| **stacked fusion: all blocks** | +0.0227 [-0.0702, +0.1151] | +0.0023 [-0.0971, +0.1013] |

**Verdict:** relative to text alone, fusing all blocks changes AUC by +0.0615 [-0.0153, +0.1385] (not established) and balanced accuracy by +0.0672 [-0.0195, +0.1552] (not established) at n=138.

Text alone here uses a learned threshold, so it can differ from the 0.5-threshold floor above. Pair rows are ablations; the all-block rows are the pre-declared result. Regenerate with `train/train_fusion.py`.

## E14: fusion with test-time agreement

Model: `deepseek-ai/DeepSeek-R1-0528-Qwen3-8B`  
Instances: 138 of 153 with every block (37 robust)  
Blocks: text, stagea, agreement  
Protocol: nested grouped CV; block scores standardised on inner folds; thresholds learned on inner folds (a score >= 0 is robust); fusions fixed in advance

| Method | Accuracy | Balanced acc. | ROC-AUC | Predicted robust |
| --- | --- | --- | --- | --- |
| always-spurious (majority) | 0.7319 | 0.5000 | 0.5000 | 0 |
| text alone | 0.6739 | 0.6573 | 0.6794 | 54 |
| stagea alone | 0.6232 | 0.6569 | 0.7490 | 69 |
| agreement alone | 0.5725 | 0.6651 | 0.7119 | 86 |
| mean fusion: text + stagea | 0.6739 | 0.6916 | 0.7305 | 62 |
| mean fusion: text + agreement | 0.6522 | 0.6853 | 0.7418 | 67 |
| mean fusion: stagea + agreement | 0.6377 | 0.7182 | 0.7731 | 79 |
| **mean fusion: all blocks** | 0.6594 | 0.7245 | 0.7624 | 74 |
| **stacked fusion: all blocks** | 0.6667 | 0.6866 | 0.7182 | 63 |

### Difference from text alone (95% paired bootstrap CI)

| Row | AUC difference | Balanced-accuracy difference |
| --- | --- | --- |
| stagea alone | +0.0696 [-0.0150, +0.1544] | -0.0004 [-0.0917, +0.0926] |
| agreement alone | +0.0325 [-0.0978, +0.1710] | +0.0078 [-0.1077, +0.1287] |
| mean fusion: text + stagea | +0.0511 [+0.0103, +0.0953] | +0.0343 [-0.0342, +0.1013] |
| mean fusion: text + agreement | +0.0623 [-0.0104, +0.1350] | +0.0280 [-0.0485, +0.1083] |
| mean fusion: stagea + agreement | +0.0937 [+0.0012, +0.1887] | +0.0609 [-0.0290, +0.1536] |
| **mean fusion: all blocks** | +0.0830 [+0.0143, +0.1569] | +0.0672 [-0.0192, +0.1566] |
| **stacked fusion: all blocks** | +0.0388 [-0.0602, +0.1375] | +0.0293 [-0.0750, +0.1347] |

**Verdict:** relative to text alone, fusing all blocks changes AUC by +0.0830 [+0.0143, +0.1569] (established gain) and balanced accuracy by +0.0672 [-0.0192, +0.1566] (not established) at n=138.

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

## train-main-v2 transfer

Source: `aimo-interp/train-main-v2` @ `23791d9b1201`, regenerated by `train/eval_train_main_v2.py`.

### Is robustness a property of the problem or of the model?

| Scope | Agreeing pairs | Rate |
| --- | --- | --- |
| all configurations | 83 / 85 | 0.976 |
| small-track models only | 20 / 20 | 1.000 |
| chance, at a 0.732 robust base rate | — | 0.608 |

Label permutation test over 2000 draws: **p = 0.0005**.

### Does the shipped text block transfer to olympiad problems?

| Scope | n | ROC-AUC | 95% CI |
| --- | --- | --- | --- |
| all small-track rows | 34 | 0.617 | [0.391, 0.833] |
| Qwen/Qwen3.5-4B | 12 | 0.688 | — |
| Skywork/Skywork-OR1-Math-7B | 3 | 1.000 | — |
| allenai/Olmo-3-7B-Think | 9 | 0.500 | — |
| deepseek-ai/DeepSeek-R1-0528-Qwen3-8B | 10 | 0.476 | — |

- Always-robust scores **0.706** on this set; the best threshold sweep reaches **0.765** at `p >= 0.469`, chosen in-sample and so optimistic.
- The train split is **0.732** robust, but the dataset card calls the validation set close to balanced and Finding 27's all-False probe scored 0.50. Do not calibrate a threshold on the train prevalence.

## E21: official-split holdouts

### DeepSeek: MATH-trained, olympiad-tested

| Method | n | Accuracy | Balanced acc. | ROC-AUC | Predicted robust |
| --- | --- | --- | --- | --- | --- |
| E11 fused (shipped artifact) | 10 | 0.700 | 0.595 | 0.429 | 8 |
| E11 text block only | 10 | 0.500 | 0.452 | 0.476 | 6 |
| all-spurious | 10 | 0.300 | 0.500 | 0.500 | 0 |
| all-robust | 10 | 0.700 | 0.500 | 0.500 | 10 |
| layer ensemble refit on 142 MATH rows | 10 | 0.500 | 0.548 | 0.476 | 4 |

### Cross-model: next-token uncertainty, one model held out

Five scalars from the same forward pass, standardised within each model.

| Method | n | Accuracy | Balanced acc. | ROC-AUC | Predicted robust |
| --- | --- | --- | --- | --- | --- |
| held out `Qwen/Qwen3.5-4B` | 12 | 0.583 | 0.625 | 0.406 | 5 |
| &nbsp;&nbsp;other models' base rate | 12 | 0.667 | 0.500 | 0.500 | 12 |
| held out `Skywork/Skywork-OR1-Math-7B` | 3 | 0.667 | 0.750 | 0.500 | 1 |
| &nbsp;&nbsp;other models' base rate | 3 | 0.667 | 0.500 | 0.500 | 3 |
| held out `allenai/Olmo-3-7B-Think` | 9 | 0.444 | 0.464 | 0.571 | 4 |
| &nbsp;&nbsp;other models' base rate | 9 | 0.778 | 0.500 | 0.500 | 9 |
| held out `deepseek-ai/DeepSeek-R1-0528-Qwen3-8B` | 10 | 0.600 | 0.619 | 0.571 | 5 |
| &nbsp;&nbsp;other models' base rate | 10 | 0.700 | 0.500 | 0.500 | 10 |

Regenerate with `train/eval_official_holdout.py`.

## Self-generated olympiad labels

> **Generated on the `nakib_v2` checkout** (19 Sep) and copied here verbatim: the script reads run outputs and label directories that only that machine holds. Do not hand-edit; regenerate there.

Self-generated `answer_modulus` labels on the olympiad pool, by `datagen/perturb/label.py`. Regenerated by `train/summarize_selflabels.py`.

| Model | Train-ready | Robust | Robust rate |
| --- | --- | --- | --- |
| `Qwen/Qwen3.5-4B` | 33 | 19 | 0.576 |
| `Skywork/Skywork-OR1-Math-7B` | 24 | 20 | 0.833 |

### Cross-model agreement on shared problems

| Pair | Shared | Agree | Rate | Chance | p |
| --- | --- | --- | --- | --- | --- |
| `Qwen/Qwen3.5-4B` vs `Skywork/Skywork-OR1-Math-7B` | 6 | 5 | 0.833 | 0.551 | 0.164 |

`Chance` is the agreement two independent labellers would reach drawing at each model's own robust rate. Both rates are high, so a high agreement rate is worth little until the shared set is large; read the p column, not the rate.

## Labelling cost model

> **Generated on the `nakib_v2` checkout** (19 Sep) and copied here verbatim: the script reads run outputs and label directories that only that machine holds. Do not hand-edit; regenerate there.

Cost of labelling the remaining olympiad pool, from measured throughput. Regenerated by `datagen/plan_labelling_cost.py --weeks 6`.

The pool holds **505 problems**; `answer_modulus` variants exist for **333** of them, 1.95 per problem. It is the only `distinct` category, so it is the only one that may carry labels.

| Model | Phase | Items measured | GPU-sec/item |
| --- | --- | --- | --- |
| Qwen3.5-4B | base | 19 | 454 |
| Qwen3.5-4B | variants | 107 | 567 |
| Skywork-7B | base | 25 | 363 |
| Skywork-7B | variants | 28 | 2425 |
| Olmo-7B-Think | base | 32 | 2138 |

Eligibility is pool-wide, from the full base phases (Findings 25-26) â€” not from the gapfill runs above, whose problems were picked *because* an earlier run found them eligible.

| Model | Eligible | Cost per usable label | Whole pool | Share of budget |
| --- | --- | --- | --- | --- |
| Qwen3.5-4B | 32% | 0.71 GPU-h | **113 GPU-h** for ~159 labels | 16% |
| Skywork-7B | 64% | 1.47 GPU-h | **474 GPU-h** for ~322 labels | 66% |
| Olmo-7B-Think | 57% | *no variants run yet* | -- | -- |

Budget: 4 reachable accounts x 30 GPU-hours/week x 6 weeks = **720 GPU-hours**.

Compute is not the binding constraint. The pool itself is: 505 problems cap what any amount of GPU time can produce, and only the share a model solves 4/4 can be labelled at all.

## E21: Stage A on olympiad labels

> **Generated on the `nakib_v2` checkout** (19 Sep) and copied here verbatim: the script reads run outputs and label directories that only that machine holds. Do not hand-edit; regenerate there.

> **Superseded by Finding 31.** The probes below are fitted on self-generated `answer_modulus` labels, which E20 showed do not measure official robustness (agreement 0.29, kappa -0.135). This section records what E21 measured; it is not a result to build on, and its borderline strict-control p-value is consistent with the labels carrying no signal.

Probes fitted on 26 answer-modulus self-labels, scored on 12 officially labelled `train-main-v2` rows for `Qwen/Qwen3.5-4B`, disjoint by `problem_key`. Regenerated by `train/eval_stagea_olympiad.py`.

Evaluation rows are 0.667 robust. The hidden validation set is close to balanced (Finding 27), so read ROC-AUC, not accuracy.

| Block | ROC-AUC | 95% CI |
| --- | --- | --- |
| Stage A, `last` pooling | 0.750 | [0.407, 1.000] |
| Stage A, `max` pooling | 0.938 | [0.743, 1.000] |
| Stage A, `mean` pooling | 0.875 | [0.611, 1.000] |
| **Stage A, all poolings** | **0.844** | [0.556, 1.000] |
| Text block (reference) | 0.688 | [0.222, 1.000] |

Paired bootstrap, Stage A minus text: **+0.158**, 95% CI [-0.273, +0.600] -- no difference detected.

### Controls

| Null | Null mean | 95th pct | p |
| --- | --- | --- | --- |
| Permuted **evaluation** labels (5000 draws) | 0.499 | 0.812 | **0.037** |
| Shuffled **training** labels, refitted (40 draws) | 0.541 | 0.816 | **0.073** |

The second control is the one to believe: a probe fitted on random labels still reaches 0.816 five percent of the time at this sample size, so the observed 0.844 sits inside its tail. **Suggestive, not established.**

At n=12 this can falsify a block, not confirm one. Per-pooling rows are reported for spread only -- the all-pooling ensemble is the pre-specified estimator, and picking the best pooling on 12 rows repeats the selection effect of Finding 11.
