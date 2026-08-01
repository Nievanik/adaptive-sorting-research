# Phase 3 — Complete: ML Model Training, Evaluation, and Analysis

**Status:** ✅ FROZEN  
**Completed:** 2026-08-01  
**Production model:** `ml/models/adaptive_sort_model.joblib`

---

## 1. Phase 3 Objective

Train, evaluate, tune, compare, and freeze a classification model that predicts the optimal sorting action (`continue`, `switch_insertion_sort`, `switch_merge_sort`, `switch_quick_sort`) at mid-execution checkpoints for the adaptive sorting system.

---

## 2. Completed Steps

| Step | Description | Status |
| :--- | :--- | :--- |
| 3.1 | Project structure and environment setup | ✅ |
| 3.2 | Data loading, preprocessing pipeline, Random Forest baseline | ✅ |
| 3.3 | Baseline evaluation: holdout metrics, cross-validation, feature importance | ✅ |
| 3.4 | Decision Tree baseline (explicit hyperparameters, same pipeline) | ✅ |
| 3.5 | Model comparison and production model selection | ✅ |
| 3.6 | Final verification of training and prediction workflow | ✅ |
| 3.7 | Decision Tree hyperparameter tuning (RandomizedSearchCV, 50 candidates) | ✅ |
| 3.8 | Baseline vs. tuned comparison and replacement decision | ✅ |
| 3.9 | Feature group ablation study | ✅ |
| 3.9B | Ablation audit and protocol correction | ✅ |
| 3.10 | Publication-ready tables, figures, and results report | ✅ |
| 3.11 | Final audit, documentation freeze, and sign-off | ✅ |

---

## 3. Dataset Summary

- **Path:** `ml/data/processed/checkpoint_training.csv`
- **Rows:** 90
- **Raw features:** 11 (`algorithm`, `input_type`, `size`, `checkpoint_pct`, `checkpoint_time_ms`, `checkpoint_comparisons`, `checkpoint_data_movements`, `comparisons_per_element`, `movements_per_element`, `work_ratio`, `time_per_element_ms`)
- **Transformed features:** 18 (after one-hot encoding of categorical features)
- **Target:** `best_action`

### Class Distribution
| Class | Count | % |
| :--- | :--- | :--- |
| continue | 28 | 31.11% |
| switch_insertion_sort | 32 | 35.56% |
| switch_merge_sort | 7 | **7.78%** ← underrepresented |
| switch_quick_sort | 23 | 25.56% |

---

## 4. Models Evaluated

| Model | Hyperparameters | CV Macro F1 | Holdout Macro F1 |
| :--- | :--- | :--- | :--- |
| Random Forest Baseline | default, `random_state=42` | 63.61% | 95.80% |
| **Decision Tree Baseline ★** | `max_depth=5`, `min_samples_split=5`, `min_samples_leaf=2` | **73.91%** | 90.80% |
| Decision Tree Tuned | `criterion=log_loss`, `max_depth=6`, `class_weight=balanced`, `max_features=sqrt` | 73.32% | 91.67% |

---

## 5. Final Model Selection

**Selected:** Decision Tree Baseline  
**Path:** `ml/models/adaptive_sort_model.joblib`

**Rationale:** CV macro F1 was defined as the primary selection criterion. The Decision Tree achieved 73.91% vs 63.61% for the Random Forest — a 10.30 pp advantage. The holdout gap (Random Forest 95.80% vs DT 90.80%) was not considered decisive because the 18-sample holdout is too small for reliable generalization inference on a 90-sample dataset.

**Tuning decision:** The tuned Decision Tree's CV macro F1 (73.32%) was 0.59 pp below the baseline, exceeding the 0.005 replacement threshold in the wrong direction. Baseline retained.

---

## 6. Key Metrics (Decision Tree Baseline)

| Metric | Value |
| :--- | :--- |
| Holdout Accuracy | 88.89% |
| Holdout Macro F1 | 90.80% |
| Holdout Weighted F1 | 88.24% |
| CV Accuracy (mean ± std) | 81.11% ± 5.67% |
| CV Macro F1 (mean ± std) | 73.91% ± 10.91% |
| CV Weighted F1 (mean ± std) | 81.45% ± 4.59% |

---

## 7. Tuning Conclusion

RandomizedSearchCV with 50 candidates and 5-fold CV optimizing `f1_macro` was run on the Decision Tree baseline. Best parameters: `criterion=log_loss, max_depth=6, min_samples_split=5, min_samples_leaf=2, max_features=sqrt, class_weight=balanced`. The tuned model improved holdout macro F1 slightly (+0.87 pp) but degraded CV macro F1 (−0.59 pp). Per predefined criteria, the baseline was retained.

---

## 8. Ablation Conclusion

Five feature groups were evaluated. Corrected baseline CV macro F1 std ≈ ±10.91 pp.

| Rank | Group Removed | Δ CV Macro F1 | Interpretation |
| :--- | :--- | :--- | :--- |
| 1 | A — Algorithm Metadata | −17.33 pp | Strong measured contribution (exceeds std) |
| 2 | C — Runtime | −4.40 pp | Moderate measured contribution |
| 3 | E — Data Movement | −3.03 pp | Small measured contribution |
| 4 | B — Checkpoint Progress | −2.75 pp | Small measured contribution |
| 5 | D — Comparison | 0.00 pp | No measurable unique contribution under current conditions |

Only Group A's drop exceeds the baseline CV macro F1 standard deviation.

---

## 9. Publication Outputs

All in `ml/results/publication/`:

**Tables:** `table_model_comparison`, `table_feature_importance`, `table_feature_ablation`, `table_experiment_summary` (CSV + Markdown)

**Figures (PNG 300 DPI + PDF):** `figure_model_performance`, `figure_cv_stability`, `figure_confusion_matrix`, `figure_feature_importance`, `figure_feature_ablation`, `figure_label_distribution`

**Report:** `phase3_research_results.md`  
**Manifest:** `publication_manifest.json`

---

## 10. Test Results

| Test Suite | Tests | Result |
| :--- | :--- | :--- |
| `test_train.py` | 19 | ✅ all passed |
| `test_evaluate.py` | — | ✅ all passed |
| `test_ablation.py` | 18 | ✅ all passed |
| `test_compare.py` | 10 | ✅ all passed |
| `test_tune.py` | 5 | ✅ all passed (1 transient OS/disk failure in combined run; passes alone) |
| `test_publication_results.py` | 20 | ✅ all passed |
| `test_phase3_audit.py` | 20 | ✅ all passed |
| **Full suite** | **129+** | **✅ clean single-run pass** |

---

## 11. Reproducibility Commands

From the repository root, in order:

```bash
./research_env/bin/python ml/train.py
./research_env/bin/python ml/src/tune.py
./research_env/bin/python ml/src/compare.py
./research_env/bin/python ml/src/ablation.py
MPLCONFIGDIR=/tmp/mpl_cache ./research_env/bin/python ml/src/publication_results.py
./research_env/bin/python -m pytest ml/tests -v
```

All commands are deterministic for the same `checkpoint_training.csv` dataset and `research_env` environment.

---

## 12. Known Limitations

1. 90-sample dataset; high CV variance (macro F1 std ≈ 10.91 pp).
2. `switch_merge_sort` has 7 samples — per-class metrics are unreliable for this class.
3. Single holdout split of 18 samples.
4. No repeated CV; single StratifiedKFold configuration.
5. 0.005 replacement tolerance is operational convention, not statistical significance.
6. Impurity-based feature importance reflects model usage, not causation.
7. Production model has not been integrated into the runtime sorter (Phase 4).

---

## 13. Phase 4 Entry Conditions

Phase 4 (**Runtime Integration**) may begin when:

- [x] Production model is frozen at `ml/models/adaptive_sort_model.joblib`
- [x] Model metadata is verified and complete
- [x] `predict.py` inference interface is validated
- [x] All Phase 3 tests pass
- [x] Publication outputs are generated and reviewed
- [ ] Phase 4 integration plan is approved

**Phase 4 scope:** Integrate `predict.py` inference into the C++/Java sorting runtime, measure checkpoint decision overhead, and evaluate end-to-end adaptive sorting performance on the benchmark suite.
