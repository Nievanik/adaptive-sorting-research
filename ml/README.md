# Machine Learning Pipeline for Adaptive Sorting

This directory contains the Phase 3 ML components that train, evaluate, tune, and freeze the production decision model used to predict the optimal sorting action (`continue`, `switch_insertion_sort`, `switch_merge_sort`, `switch_quick_sort`) at execution checkpoints.

---

## 1. Overview

Phase 3 implements a complete research-grade ML pipeline:

1. **Train** — fit Random Forest and Decision Tree baseline classifiers on checkpoint records.
2. **Evaluate** — compare holdout and cross-validation metrics under identical conditions.
3. **Select** — choose the production model using CV macro F1 as the primary criterion.
4. **Tune** — run hyperparameter search on the Decision Tree and test whether it improves generalization.
5. **Compare** — rigorously compare baseline vs. tuned candidate; replace only if the threshold is exceeded.
6. **Ablation** — quantify the contribution of each feature group by retraining with one group removed.
7. **Publication** — generate research-ready tables, figures, and a consolidated results report.
8. **Audit** — verify all artifacts, freeze the production model, and hand off to Phase 4.

**Selected production model:** Decision Tree Baseline (`ml/models/adaptive_sort_model.joblib`)
**Selection criterion:** CV macro F1 (73.91% vs 63.61% for Random Forest under stratified 5-fold CV)

---

## 2. Dataset

| Property | Value |
| :--- | :--- |
| Path | `ml/data/processed/checkpoint_training.csv` |
| Total rows | 90 |
| Raw features | 11 |
| Transformed features | 18 (after one-hot encoding) |
| Target column | `best_action` |
| Train/holdout split | 80% / 20% stratified, `random_state=42` |

### Raw Feature Schema (required order)

```
algorithm, input_type, size, checkpoint_pct, checkpoint_time_ms,
checkpoint_comparisons, checkpoint_data_movements, comparisons_per_element,
movements_per_element, work_ratio, time_per_element_ms
```

### Target Class Distribution

| Class | Count | Percentage |
| :--- | :--- | :--- |
| continue | 28 | 31.11% |
| switch_insertion_sort | 32 | 35.56% |
| switch_merge_sort | 7 | 7.78% ← underrepresented |
| switch_quick_sort | 23 | 25.56% |

### Leakage Exclusions

The following columns are present in the raw dataset but are **always excluded** from the feature matrix to prevent target leakage:

```
best_action, case,
continue_time_ms, continue_comparisons, continue_data_movements, continue_overhead_time_ms,
switch_insertion_sort_time_ms, switch_insertion_sort_comparisons, ...,
switch_merge_sort_time_ms, ..., switch_quick_sort_time_ms, ...,
best_action_total_ms, speedup_vs_continue
```

---

## 3. Models

### Random Forest Baseline
- **Class:** `sklearn.ensemble.RandomForestClassifier` (default parameters, `random_state=42`)
- **Saved at:** `ml/models/random_forest_baseline.joblib`
- **Holdout macro F1:** 95.80% | **CV macro F1:** 63.61%

### Decision Tree Baseline ✓ SELECTED PRODUCTION MODEL
- **Class:** `sklearn.tree.DecisionTreeClassifier`
- **Hyperparameters:** `max_depth=5`, `min_samples_split=5`, `min_samples_leaf=2`, `random_state=42`
- **Saved at:** `ml/models/decision_tree_baseline.joblib`
- **Holdout macro F1:** 90.80% | **CV macro F1:** 73.91%

### Decision Tree Tuned (candidate, not production)
- **Best params:** `criterion=log_loss`, `max_depth=6`, `min_samples_split=5`, `min_samples_leaf=2`, `max_features=sqrt`, `class_weight=balanced`
- **Saved at:** `ml/models/decision_tree_tuned.joblib`
- **CV macro F1:** 73.32% (below baseline by 0.59 pp; did not meet 0.005 replacement threshold)

---

## 4. Production Model Selection

**Primary metric:** CV macro F1 (Stratified 5-fold, `shuffle=True`, `random_state=42`)

**Priority hierarchy:**
1. CV mean macro F1
2. CV mean weighted F1
3. CV mean accuracy
4. Holdout macro F1
5. Holdout weighted F1
6. Simplicity

**Decision:** The Decision Tree Baseline was selected because it achieved higher CV macro F1 (73.91%) than the Random Forest (63.61%). Although the Random Forest had stronger holdout performance (95.80% vs 90.80%), the holdout set contains only 18 samples — insufficient to judge generalization reliably on a 90-sample dataset.

**Tuning decision:** The tuned Decision Tree degraded CV macro F1 by 0.59 pp (above the 0.005 replacement tolerance), so the baseline was retained.

**Production model path:** `ml/models/adaptive_sort_model.joblib`

---

## 5. Commands

Run all commands from the **repository root** using the virtual environment.

### Training (baseline models + comparison + selection)
```bash
./research_env/bin/python ml/train.py
```

### Hyperparameter Tuning
```bash
./research_env/bin/python ml/src/tune.py
```

### Baseline vs. Tuned Comparison
```bash
./research_env/bin/python ml/src/compare.py
```

### Feature Group Ablation Study
```bash
./research_env/bin/python ml/src/ablation.py
```

### Publication Tables and Figures
```bash
MPLCONFIGDIR=/tmp/mpl_cache ./research_env/bin/python ml/src/publication_results.py
```

### Single-Record Prediction (CLI)
```bash
./research_env/bin/python ml/predict.py \
  --model-path ml/models/adaptive_sort_model.joblib \
  --algorithm quick_sort \
  --input-type random \
  --size 1000 \
  --checkpoint-pct 50 \
  --checkpoint-time-ms 1.25 \
  --checkpoint-comparisons 4200 \
  --checkpoint-data-movements 1700 \
  --comparisons-per-element 4.2 \
  --movements-per-element 1.7 \
  --work-ratio 0.40 \
  --time-per-element-ms 0.00125
```

### Prediction from JSON File
```bash
./research_env/bin/python ml/predict.py \
  --model-path ml/models/adaptive_sort_model.joblib \
  --input-json path/to/checkpoint.json
```

### Run ML Tests Only
```bash
./research_env/bin/python -m pytest ml/tests -v
```

### Run Full Repository Tests
```bash
./research_env/bin/python -m pytest -v
```

> **Note:** If Matplotlib font-cache errors occur, prefix commands with `MPLCONFIGDIR=/tmp/mpl_cache`.

---

## 6. Outputs

### `ml/models/`

| File | Contents |
| :--- | :--- |
| `adaptive_sort_model.joblib` | **Production model** — Decision Tree Baseline (use this for inference) |
| `adaptive_sort_model_metadata.json` | Full audit trail: hyperparameters, metrics, selection rationale |
| `decision_tree_baseline.joblib` | Baseline Decision Tree pipeline (identical to production) |
| `decision_tree_tuned.joblib` | Tuned Decision Tree pipeline (not production) |
| `random_forest_baseline.joblib` | Random Forest baseline pipeline |

### `ml/results/`

| File | Contents |
| :--- | :--- |
| `classification_report.json` | Random Forest holdout per-class metrics |
| `confusion_matrix.csv` | Random Forest holdout confusion matrix |
| `feature_importance.csv` | Random Forest feature importances |
| `decision_tree_feature_importance.csv` | **Decision Tree baseline feature importances** (primary reference) |
| `baseline_analysis.md` | Random Forest baseline evaluation narrative |
| `model_comparison.{json,md}` | RF vs DT comparison table and narrative |
| `model_metrics.csv` | Machine-readable comparison metrics |
| `tuning_results.{json,csv}` | Hyperparameter search results |
| `decision_tree_tuning_summary.md` | Tuning narrative |
| `baseline_vs_tuned_comparison.{json,md}` | Baseline vs tuned decision record |
| `baseline_vs_tuned_metrics.csv` | Machine-readable BvT metrics |
| `feature_ablation.{json,csv,md}` | Corrected ablation study results |
| `feature_ablation_*.png` | Ablation bar charts |

### `ml/results/publication/`

Publication-ready outputs generated by `ml/src/publication_results.py`:

| Type | Files |
| :--- | :--- |
| Tables (CSV + Markdown) | `table_model_comparison`, `table_feature_importance`, `table_feature_ablation`, `table_experiment_summary` |
| Figures (PNG 300 DPI + PDF) | `figure_model_performance`, `figure_cv_stability`, `figure_confusion_matrix`, `figure_feature_importance`, `figure_feature_ablation`, `figure_label_distribution` |
| Report | `phase3_research_results.md` |
| Manifest | `publication_manifest.json` |

---

## 7. Reproducibility

| Parameter | Value |
| :--- | :--- |
| Random state | 42 (all models, splits, and CV) |
| Test size | 0.2 (stratified holdout) |
| CV folds | 5 (StratifiedKFold, shuffle=True, random_state=42) |
| CV data | Full dataset in original CSV row order |
| Environment | `research_env/` — see `requirements.txt` |
| scikit-learn | 1.9.0 |

**Command order for full reproduction:**
```
ml/train.py → ml/src/tune.py → ml/src/compare.py →
ml/src/ablation.py → ml/src/publication_results.py
```

**Deterministic limitation:** All scripts are deterministic given the same dataset and environment. Results may differ if `checkpoint_training.csv` is regenerated from new benchmark runs.

---

## 8. Limitations

1. **90 samples total** — CV fold variance is high (macro F1 std ≈ 10.91 pp); all results are preliminary.
2. **`switch_merge_sort` underrepresented** — only 7 samples; per-class metrics for this class are unreliable.
3. **Single holdout split** — 18-sample holdout cannot provide stable generalization estimates.
4. **CV variability** — no repeated CV was performed; a single StratifiedKFold configuration was used.
5. **Practical tolerance** — the 0.005 replacement threshold is an operational convention, not a statistical significance threshold.
6. **Impurity-based feature importance** — reflects model usage, not causal contribution.
7. **Not yet runtime-tested** — the production model has not been integrated into the running sorting system or measured for inference overhead.

---

## 9. Next Phase

**Phase 4** will integrate the production model (`ml/models/adaptive_sort_model.joblib`) into the runtime sorting system, measure decision latency and overhead per checkpoint call, and evaluate the adaptive sorter's end-to-end performance on the benchmark suite.

Phase 4 has **not** been started. The production model is frozen at the Decision Tree Baseline.
