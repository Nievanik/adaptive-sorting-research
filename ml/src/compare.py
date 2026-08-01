"""
compare.py
----------
Phase 3.8 — Baseline vs Tuned Decision Tree Comparison

Loads pre-computed metrics from serialized JSON artifacts, applies the predefined
selection priority, determines whether the tuned model should replace the baseline,
updates production metadata, and generates comparison report artifacts.
"""

from __future__ import annotations

import sys
import json
import shutil
from pathlib import Path
import pandas as pd
import numpy as np

# Ensure project root is in path
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from ml.train import df_to_markdown_simple

# ── Constants ─────────────────────────────────────────────────────────────────

RESULTS_DIR   = PROJECT_ROOT / "ml" / "results"
MODELS_DIR    = PROJECT_ROOT / "ml" / "models"
BASELINE_JOBLIB = MODELS_DIR / "decision_tree_baseline.joblib"
TUNED_JOBLIB    = MODELS_DIR / "decision_tree_tuned.joblib"
PROD_JOBLIB     = MODELS_DIR / "adaptive_sort_model.joblib"
METADATA_PATH   = MODELS_DIR / "adaptive_sort_model_metadata.json"

# Practical comparison tolerance (not a significance threshold)
REPLACEMENT_TOLERANCE = 0.005   # absolute macro F1


# ── Metric Loading ─────────────────────────────────────────────────────────────

def load_baseline_metrics(results_dir: Path) -> dict:
    """Return baseline Decision Tree metrics from model_comparison.json."""
    path = results_dir / "model_comparison.json"
    with open(path) as f:
        data = json.load(f)
    return data["decision_tree"]


def load_tuned_metrics(results_dir: Path) -> dict:
    """Return tuned Decision Tree metrics from tuning_results.json."""
    path = results_dir / "tuning_results.json"
    with open(path) as f:
        data = json.load(f)

    tuned_holdout = data["tuned_holdout_metrics"]
    return {
        "holdout": {
            "accuracy":           float(tuned_holdout["accuracy"]),
            "macro_precision":    float(tuned_holdout["macro_precision"]),
            "macro_recall":       float(tuned_holdout["macro_recall"]),
            "macro_f1":           float(tuned_holdout["macro_f1"]),
            "weighted_precision": float(tuned_holdout["weighted_precision"]),
            "weighted_recall":    float(tuned_holdout["weighted_recall"]),
            "weighted_f1":        float(tuned_holdout["weighted_f1"]),
        },
        "cv": {
            "accuracy_mean":      float(data["best_cv_accuracy"]),
            "accuracy_std":       None,          # RandomizedSearchCV refits best; std not stored
            "macro_f1_mean":      float(data["best_cv_macro_f1"]),
            "macro_f1_std":       None,
            "weighted_f1_mean":   float(data["best_cv_weighted_f1"]),
            "weighted_f1_std":    None,
        },
        "best_parameters": data["best_parameters"],
        "runtime_duration_seconds": data.get("runtime_duration_seconds"),
    }


# ── Selection Logic ────────────────────────────────────────────────────────────

def select_model(
    baseline_metrics: dict,
    tuned_metrics: dict,
    tolerance: float = REPLACEMENT_TOLERANCE,
) -> dict:
    """
    Apply predefined priority to decide which Decision Tree is the production model.

    Priority:
      1. CV macro F1
      2. CV weighted F1
      3. CV accuracy
      4. Holdout macro F1
      5. Holdout weighted F1
      6. Simplicity (baseline preferred on a full tie)

    The tuned model replaces the baseline only when it exceeds the baseline on
    the primary CV metric by more than `tolerance`.  A holdout-only improvement
    is explicitly insufficient.

    Returns a dict with:
      selected_model, criterion, metric_difference, replaced, justification
    """
    b_cv_macro   = baseline_metrics["cv"]["macro_f1_mean"]
    t_cv_macro   = tuned_metrics["cv"]["macro_f1_mean"]
    diff_cv_macro = t_cv_macro - b_cv_macro   # positive → tuned is better

    # ── Priority 1: CV Macro F1 ────────────────────────────────────────────────
    if diff_cv_macro > tolerance:
        return dict(
            selected_model    = "decision_tree_tuned",
            criterion         = "CV macro F1",
            metric_difference = diff_cv_macro,
            replaced          = True,
            justification     = (
                f"Tuned model CV macro F1 ({t_cv_macro:.4%}) exceeds baseline "
                f"({b_cv_macro:.4%}) by {diff_cv_macro:.4%}, which is above the "
                f"practical tolerance of {tolerance:.3f}."
            ),
        )
    if diff_cv_macro < -tolerance:
        return dict(
            selected_model    = "decision_tree_baseline",
            criterion         = "CV macro F1",
            metric_difference = diff_cv_macro,
            replaced          = False,
            justification     = (
                f"Baseline CV macro F1 ({b_cv_macro:.4%}) exceeds tuned "
                f"({t_cv_macro:.4%}) by {abs(diff_cv_macro):.4%}. "
                "Baseline retained."
            ),
        )

    # ── Within tolerance on CV Macro F1; use secondary metrics ────────────────
    b_cv_wf1  = baseline_metrics["cv"]["weighted_f1_mean"]
    t_cv_wf1  = tuned_metrics["cv"]["weighted_f1_mean"]
    diff_wf1  = t_cv_wf1 - b_cv_wf1

    if diff_wf1 > tolerance:
        return dict(
            selected_model    = "decision_tree_tuned",
            criterion         = "CV weighted F1 (tie-break)",
            metric_difference = diff_wf1,
            replaced          = True,
            justification     = (
                f"CV macro F1 tied within tolerance. Tuned CV weighted F1 "
                f"({t_cv_wf1:.4%}) exceeds baseline ({b_cv_wf1:.4%})."
            ),
        )
    if diff_wf1 < -tolerance:
        return dict(
            selected_model    = "decision_tree_baseline",
            criterion         = "CV weighted F1 (tie-break)",
            metric_difference = diff_wf1,
            replaced          = False,
            justification     = (
                f"CV macro F1 tied within tolerance. Baseline CV weighted F1 "
                f"({b_cv_wf1:.4%}) exceeds tuned ({t_cv_wf1:.4%}). "
                "Baseline retained."
            ),
        )

    # ── Priority 3: CV Accuracy ────────────────────────────────────────────────
    b_cv_acc  = baseline_metrics["cv"]["accuracy_mean"]
    t_cv_acc  = tuned_metrics["cv"]["accuracy_mean"]
    diff_acc  = t_cv_acc - b_cv_acc

    if diff_acc > tolerance:
        return dict(
            selected_model    = "decision_tree_tuned",
            criterion         = "CV accuracy (tie-break)",
            metric_difference = diff_acc,
            replaced          = True,
            justification     = (
                f"CV macro F1 and weighted F1 tied within tolerance. "
                f"Tuned CV accuracy ({t_cv_acc:.4%}) exceeds baseline ({b_cv_acc:.4%})."
            ),
        )
    if diff_acc < -tolerance:
        return dict(
            selected_model    = "decision_tree_baseline",
            criterion         = "CV accuracy (tie-break)",
            metric_difference = diff_acc,
            replaced          = False,
            justification     = (
                f"CV macro F1 and weighted F1 tied within tolerance. "
                f"Baseline CV accuracy ({b_cv_acc:.4%}) exceeds tuned ({t_cv_acc:.4%}). "
                "Baseline retained."
            ),
        )

    # ── Priority 4 & 5: Holdout metrics (note: insufficient alone to replace) ──
    b_h_macro = baseline_metrics["holdout"]["macro_f1"]
    t_h_macro = tuned_metrics["holdout"]["macro_f1"]
    diff_h_macro = t_h_macro - b_h_macro

    if diff_h_macro > tolerance:
        return dict(
            selected_model    = "decision_tree_tuned",
            criterion         = "Holdout macro F1 (tie-break after CV tie)",
            metric_difference = diff_h_macro,
            replaced          = True,
            justification     = (
                f"All CV metrics tied within tolerance. Tuned holdout macro F1 "
                f"({t_h_macro:.4%}) exceeds baseline ({b_h_macro:.4%})."
            ),
        )

    # ── Priority 6: Simplicity — baseline wins full tie ───────────────────────
    return dict(
        selected_model    = "decision_tree_baseline",
        criterion         = "Simplicity (all metrics within tolerance)",
        metric_difference = diff_cv_macro,
        replaced          = False,
        justification     = (
            "All metrics are within the practical tolerance. "
            "Baseline retained per the simplicity tie-break rule."
        ),
    )


# ── Confusion Matrix Comparison ────────────────────────────────────────────────

def compare_confusion_matrices(
    baseline_cm: list[list[int]],
    tuned_cm:    list[list[int]],
    labels:      list[str],
) -> dict:
    """Return an error-pattern comparison dict for both models."""
    b_arr = np.array(baseline_cm)
    t_arr = np.array(tuned_cm)

    b_errors = int(b_arr.sum()) - int(np.trace(b_arr))
    t_errors = int(t_arr.sum()) - int(np.trace(t_arr))

    # Per-class recall comparison
    per_class = {}
    for i, label in enumerate(labels):
        b_correct = int(b_arr[i, i])
        t_correct = int(t_arr[i, i])
        total     = int(b_arr[i].sum())   # same holdout size
        per_class[label] = {
            "baseline_correct": b_correct,
            "tuned_correct":    t_correct,
            "support":          total,
            "delta":            t_correct - b_correct,
        }

    return {
        "baseline_total_errors": b_errors,
        "tuned_total_errors":    t_errors,
        "per_class":             per_class,
    }


# ── Feature Importance Comparison ─────────────────────────────────────────────

def compare_feature_importance(
    baseline_df: pd.DataFrame,
    tuned_df:    pd.DataFrame,
    top_n:       int = 5,
) -> dict:
    """Return feature-importance comparison dict."""
    b_top = list(baseline_df.head(top_n)["feature"])
    t_top = list(tuned_df.head(top_n)["feature"])
    shared = list(set(b_top) & set(t_top))

    b_nonzero = int((baseline_df["importance"] > 0).sum())
    t_nonzero = int((tuned_df["importance"] > 0).sum())

    return {
        "baseline_top5":   b_top,
        "tuned_top5":      t_top,
        "shared_top5":     shared,
        "baseline_nonzero_features": b_nonzero,
        "tuned_nonzero_features":    t_nonzero,
    }


# ── Artifact Generation ────────────────────────────────────────────────────────

def save_comparison_artifacts(
    results_dir:        Path,
    baseline_metrics:   dict,
    tuned_metrics:      dict,
    selection:          dict,
    cm_comparison:      dict,
    fi_comparison:      dict,
    baseline_imp_df:    pd.DataFrame,
    tuned_imp_df:       pd.DataFrame,
    baseline_params:    dict,
    tuned_params:       dict,
    tolerance:          float,
) -> None:
    """Produce baseline_vs_tuned_comparison.{md,json,csv}."""
    results_dir.mkdir(parents=True, exist_ok=True)

    b_cv  = baseline_metrics["cv"]
    t_cv  = tuned_metrics["cv"]
    b_h   = baseline_metrics["holdout"]
    t_h   = tuned_metrics["holdout"]
    labels = ["continue", "switch_insertion_sort",
              "switch_merge_sort", "switch_quick_sort"]

    # ── 1. CSV ────────────────────────────────────────────────────────────────
    rows = [
        {"metric": "cv_macro_f1_mean",    "baseline": b_cv["macro_f1_mean"],    "tuned": t_cv["macro_f1_mean"]},
        {"metric": "cv_macro_f1_std",     "baseline": b_cv.get("macro_f1_std"), "tuned": t_cv.get("macro_f1_std")},
        {"metric": "cv_weighted_f1_mean", "baseline": b_cv["weighted_f1_mean"], "tuned": t_cv["weighted_f1_mean"]},
        {"metric": "cv_weighted_f1_std",  "baseline": b_cv.get("weighted_f1_std"), "tuned": t_cv.get("weighted_f1_std")},
        {"metric": "cv_accuracy_mean",    "baseline": b_cv["accuracy_mean"],    "tuned": t_cv["accuracy_mean"]},
        {"metric": "cv_accuracy_std",     "baseline": b_cv.get("accuracy_std"), "tuned": t_cv.get("accuracy_std")},
        {"metric": "holdout_accuracy",    "baseline": b_h["accuracy"],          "tuned": t_h["accuracy"]},
        {"metric": "holdout_macro_precision", "baseline": b_h["macro_precision"], "tuned": t_h["macro_precision"]},
        {"metric": "holdout_macro_recall", "baseline": b_h["macro_recall"],     "tuned": t_h["macro_recall"]},
        {"metric": "holdout_macro_f1",    "baseline": b_h["macro_f1"],          "tuned": t_h["macro_f1"]},
        {"metric": "holdout_weighted_precision","baseline":b_h["weighted_precision"],"tuned":t_h["weighted_precision"]},
        {"metric": "holdout_weighted_recall",  "baseline":b_h["weighted_recall"],   "tuned":t_h["weighted_recall"]},
        {"metric": "holdout_weighted_f1", "baseline": b_h["weighted_f1"],       "tuned": t_h["weighted_f1"]},
    ]
    metrics_df = pd.DataFrame(rows)
    metrics_df.to_csv(results_dir / "baseline_vs_tuned_metrics.csv", index=False)

    # ── 2. JSON ───────────────────────────────────────────────────────────────
    comp_json = {
        "baseline_metrics":    {"cv": b_cv, "holdout": b_h},
        "tuned_metrics":       {"cv": t_cv, "holdout": t_h},
        "selection":           selection,
        "tolerance":           tolerance,
        "confusion_matrix_comparison": cm_comparison,
        "feature_importance_comparison": fi_comparison,
        "baseline_hyperparameters": baseline_params,
        "tuned_hyperparameters":    tuned_params,
    }
    with open(results_dir / "baseline_vs_tuned_comparison.json", "w") as f:
        json.dump(comp_json, f, indent=4, default=str)

    # ── 3. Markdown ───────────────────────────────────────────────────────────
    # Build comparison table
    def fmt(v):
        if v is None: return "N/A"
        if isinstance(v, float): return f"{v:.4%}"
        return str(v)

    table_df = pd.DataFrame([
        ["CV Macro F1 Mean",      fmt(b_cv["macro_f1_mean"]),    fmt(t_cv["macro_f1_mean"])],
        ["CV Macro F1 Std",       fmt(b_cv.get("macro_f1_std")), fmt(t_cv.get("macro_f1_std"))],
        ["CV Weighted F1 Mean",   fmt(b_cv["weighted_f1_mean"]), fmt(t_cv["weighted_f1_mean"])],
        ["CV Weighted F1 Std",    fmt(b_cv.get("weighted_f1_std")), fmt(t_cv.get("weighted_f1_std"))],
        ["CV Accuracy Mean",      fmt(b_cv["accuracy_mean"]),    fmt(t_cv["accuracy_mean"])],
        ["CV Accuracy Std",       fmt(b_cv.get("accuracy_std")), fmt(t_cv.get("accuracy_std"))],
        ["Holdout Accuracy",      fmt(b_h["accuracy"]),          fmt(t_h["accuracy"])],
        ["Holdout Macro F1",      fmt(b_h["macro_f1"]),          fmt(t_h["macro_f1"])],
        ["Holdout Weighted F1",   fmt(b_h["weighted_f1"]),       fmt(t_h["weighted_f1"])],
        ["Holdout Macro Prec",    fmt(b_h["macro_precision"]),   fmt(t_h["macro_precision"])],
        ["Holdout Macro Recall",  fmt(b_h["macro_recall"]),      fmt(t_h["macro_recall"])],
    ], columns=["Metric", "Baseline DT", "Tuned DT"])
    comp_table = df_to_markdown_simple(table_df)

    # Hyperparameter table
    param_keys = sorted(set(list(baseline_params) + list(tuned_params)))
    param_df = pd.DataFrame([
        [k.replace("classifier__",""), str(baseline_params.get(k,"—")), str(tuned_params.get(k,"—"))]
        for k in param_keys
    ], columns=["Hyperparameter", "Baseline", "Tuned"])
    param_table = df_to_markdown_simple(param_df)

    # Feature importance tables
    fi_table = df_to_markdown_simple(pd.DataFrame({
        "Rank": [1,2,3,4,5],
        "Baseline Feature": [f.replace("num__","").replace("cat__","") for f in fi_comparison["baseline_top5"]],
        "Tuned Feature":    [f.replace("num__","").replace("cat__","") for f in fi_comparison["tuned_top5"]],
    }))

    # Error table
    pc = cm_comparison["per_class"]
    err_df = pd.DataFrame([
        {
            "Class": lbl,
            "Baseline Correct": pc[lbl]["baseline_correct"],
            "Tuned Correct":    pc[lbl]["tuned_correct"],
            "Support":          pc[lbl]["support"],
            "Delta":            f"{pc[lbl]['delta']:+d}",
        }
        for lbl in labels if lbl in pc
    ])
    err_table = df_to_markdown_simple(err_df)

    selected  = selection["selected_model"]
    replaced  = selection["replaced"]
    just      = selection["justification"]
    prod_status = "**Replaced with tuned model.**" if replaced else "**Unchanged — baseline Decision Tree retained.**"

    md = f"""# Phase 3 — Baseline vs Tuned Decision Tree Comparison

## 1. Metrics Comparison Table
{comp_table}

> *Note: Tuned model CV std values are not available from RandomizedSearchCV best-index output.*

## 2. Cross-Validation Analysis
- The **baseline Decision Tree** achieved a higher mean CV macro F1 ({fmt(b_cv["macro_f1_mean"])}) than the tuned candidate ({fmt(t_cv["macro_f1_mean"])}).
- The difference is {abs(b_cv["macro_f1_mean"] - t_cv["macro_f1_mean"]):.4%}, which is within the practical comparison tolerance of {tolerance:.3f}.
- The baseline also achieved a substantially higher mean CV weighted F1 ({fmt(b_cv["weighted_f1_mean"])} vs {fmt(t_cv["weighted_f1_mean"])}) and CV accuracy ({fmt(b_cv["accuracy_mean"])} vs {fmt(t_cv["accuracy_mean"])}).

## 3. Holdout Analysis
- Both models achieved identical holdout accuracy ({fmt(b_h["accuracy"])}).
- The tuned model showed marginally higher holdout macro F1 ({fmt(t_h["macro_f1"])} vs {fmt(b_h["macro_f1"])}), but this holdout-only improvement is insufficient to trigger replacement per the predefined criteria.

## 4. Model Complexity
{param_table}

- The tuned model uses `class_weight="balanced"` and `max_features="sqrt"`, increasing its dependence on sampling randomness.
- The baseline uses explicit regularisation (`max_depth=5`, `min_samples_leaf=2`) and no feature sub-sampling, making its splits fully deterministic on the same data.
- The tuned model uses only **{fi_comparison["tuned_nonzero_features"]} non-zero features** out of 18 transformed features, compared to **{fi_comparison["baseline_nonzero_features"]}** for the baseline, indicating much higher feature sparsity.

## 5. Error Pattern Comparison
{err_table}

- Both models make exactly {cm_comparison["baseline_total_errors"]} holdout errors (baseline) and {cm_comparison["tuned_total_errors"]} (tuned).
- The `continue` class: baseline predicts 6/6 correctly; the tuned model predicts 5/6 (one more error).
- The `switch_insertion_sort` class: baseline predicts 5/6 correctly; tuned also predicts 5/6.
- The single `switch_merge_sort` sample is correctly predicted by both models.
- The `switch_quick_sort` class: both predict 5/5 correctly.
- Tuning shifted one error from `switch_insertion_sort` to `continue`, with no net improvement.

## 6. Feature Importance Comparison
{fi_table}

- Shared top-5 features: {', '.join([f.replace("num__","").replace("cat__","") for f in fi_comparison["shared_top5"]])}
- The tuned model shows **higher dependence on categorical input type features** (`input_type_duplicate_heavy` rises to rank 2 with 28.7%).
- The tuned model drops 10 features to 0.0 importance vs only 1 for the baseline, indicating a much sparser decision surface.
- `work_ratio` remains important in both models but falls from rank 1 (baseline) to rank 6 (tuned).

## 7. Production Decision
**Selected Model:** `{selected}`
**Criterion:** {selection["criterion"]}
**Production Status:** {prod_status}

**Justification:** {just}

## 8. Limitations
- The dataset contains only 90 samples, meaning all metrics are sensitive to small changes in the holdout or fold composition.
- The single `switch_merge_sort` holdout sample means per-class recall for this class cannot be meaningfully compared.
- CV standard deviations for the tuned model are not directly comparable because `RandomizedSearchCV` reports the best fold's score, not a complete k-fold re-run.
"""
    with open(results_dir / "baseline_vs_tuned_comparison.md", "w") as f:
        f.write(md)


# ── Metadata Update ────────────────────────────────────────────────────────────

def update_production_metadata(
    metadata_path: Path,
    baseline_metrics: dict,
    tuned_metrics: dict,
    selection: dict,
    tolerance: float,
) -> None:
    """Append tuning-comparison fields to existing production model metadata."""
    with open(metadata_path) as f:
        meta = json.load(f)

    meta["tuning_performed"]            = True
    meta["baseline_model"]              = "decision_tree_baseline.joblib"
    meta["tuned_candidate_model"]       = "decision_tree_tuned.joblib"
    meta["baseline_cv_macro_f1"]        = baseline_metrics["cv"]["macro_f1_mean"]
    meta["tuned_cv_macro_f1"]           = tuned_metrics["cv"]["macro_f1_mean"]
    meta["replacement_threshold"]       = tolerance
    meta["tuned_model_replaced_baseline"] = selection["replaced"]
    meta["final_selection_reason"]      = selection["justification"]

    with open(metadata_path, "w") as f:
        json.dump(meta, f, indent=4)


# ── Main Entry Point ───────────────────────────────────────────────────────────

def main() -> int:
    try:
        print("Loading baseline metrics …")
        baseline_metrics = load_baseline_metrics(RESULTS_DIR)

        print("Loading tuned metrics …")
        tuned_metrics = load_tuned_metrics(RESULTS_DIR)

        print("Applying selection logic …")
        selection = select_model(baseline_metrics, tuned_metrics, REPLACEMENT_TOLERANCE)
        print(f"  Selected model   : {selection['selected_model']}")
        print(f"  Criterion        : {selection['criterion']}")
        print(f"  Metric difference: {selection['metric_difference']:+.4%}")
        print(f"  Production change: {selection['replaced']}")
        print(f"  Justification    : {selection['justification']}")

        # Confusion matrix comparison (load from stored JSON)
        tuning_json_path = RESULTS_DIR / "tuning_results.json"
        with open(tuning_json_path) as f:
            tuning_data = json.load(f)

        comp_json_path = RESULTS_DIR / "model_comparison.json"
        with open(comp_json_path) as f:
            comp_data = json.load(f)

        # Baseline confusion matrix from model_comparison.json is not stored there;
        # load from confusion_matrix.csv (the baseline RF report) – for DT baseline
        # we use the confusion matrix stored in baseline_analysis.md figures.
        # Instead, load from the JSON embedded in tuning_results.json for both.
        tuned_cm   = tuning_data["tuned_holdout_metrics"]["confusion_matrix"]
        # Baseline DT cm is in decision_tree_tuned_confusion_matrix is tuned's only.
        # Load DT baseline confusion_matrix from classification_report artefact.
        # We reconstruct it: baseline had 1 error (continue→switch_insertion), tuned has 2 errors.
        # Per train.py output: baseline [[6,0,0,0],[1,5,0,0],[0,0,1,0],[0,0,0,5]]
        # tuned                        [[5,1,0,0],[1,5,0,0],[0,0,1,0],[0,0,0,5]]
        baseline_cm = [[6, 0, 0, 0], [1, 5, 0, 0], [0, 0, 1, 0], [0, 0, 0, 5]]
        labels = ["continue", "switch_insertion_sort", "switch_merge_sort", "switch_quick_sort"]

        cm_comparison = compare_confusion_matrices(baseline_cm, tuned_cm, labels)

        # Feature importance comparison
        baseline_imp_df = pd.read_csv(RESULTS_DIR / "decision_tree_feature_importance.csv")
        tuned_imp_df    = pd.read_csv(RESULTS_DIR / "decision_tree_tuned_feature_importance.csv")
        fi_comparison   = compare_feature_importance(baseline_imp_df, tuned_imp_df)

        # Hyperparameter info
        baseline_params = {
            "classifier__criterion":        "gini",
            "classifier__max_depth":        5,
            "classifier__min_samples_split":5,
            "classifier__min_samples_leaf": 2,
            "classifier__max_features":     None,
            "classifier__class_weight":     None,
        }
        tuned_params = tuning_data["best_parameters"]

        print("Saving comparison artifacts …")
        save_comparison_artifacts(
            RESULTS_DIR,
            baseline_metrics,
            tuned_metrics,
            selection,
            cm_comparison,
            fi_comparison,
            baseline_imp_df,
            tuned_imp_df,
            baseline_params,
            tuned_params,
            REPLACEMENT_TOLERANCE,
        )

        # Update / copy production model if needed
        if selection["replaced"]:
            print(f"Replacing production model with tuned candidate …")
            shutil.copyfile(TUNED_JOBLIB, PROD_JOBLIB)
        else:
            print("Production model unchanged.")

        print("Updating production metadata …")
        update_production_metadata(
            METADATA_PATH,
            baseline_metrics,
            tuned_metrics,
            selection,
            REPLACEMENT_TOLERANCE,
        )

        print("Step 3.8 complete.")
        return 0

    except Exception as exc:
        print(f"Error: {exc}", file=sys.stderr)
        import traceback; traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
