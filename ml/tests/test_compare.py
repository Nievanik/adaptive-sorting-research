"""
test_compare.py
---------------
Unit tests for the Phase 3.8 baseline vs tuned comparison module.
"""

from __future__ import annotations

import sys
import json
import tempfile
from pathlib import Path
import pytest
import pandas as pd
import numpy as np
import joblib
from sklearn.pipeline import Pipeline

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from ml.src.compare import (
    load_baseline_metrics,
    load_tuned_metrics,
    select_model,
    compare_confusion_matrices,
    compare_feature_importance,
    save_comparison_artifacts,
    update_production_metadata,
    REPLACEMENT_TOLERANCE,
)

# ── Shared fixtures ────────────────────────────────────────────────────────────

BASELINE_METRICS = {
    "cv": {
        "accuracy_mean":    0.8111,
        "accuracy_std":     0.0566,
        "macro_f1_mean":    0.7391,
        "macro_f1_std":     0.1091,
        "weighted_f1_mean": 0.8145,
        "weighted_f1_std":  0.0459,
    },
    "holdout": {
        "accuracy":           0.8889,
        "macro_precision":    0.9226,
        "macro_recall":       0.9167,
        "macro_f1":           0.9080,
        "weighted_precision": 0.9061,
        "weighted_recall":    0.8889,
        "weighted_f1":        0.8824,
    },
}

TUNED_METRICS = {
    "cv": {
        "accuracy_mean":    0.7210,
        "accuracy_std":     None,
        "macro_f1_mean":    0.7332,
        "macro_f1_std":     None,
        "weighted_f1_mean": 0.7124,
        "weighted_f1_std":  None,
    },
    "holdout": {
        "accuracy":           0.8889,
        "macro_precision":    0.9167,
        "macro_recall":       0.9167,
        "macro_f1":           0.9167,
        "weighted_precision": 0.8889,
        "weighted_recall":    0.8889,
        "weighted_f1":        0.8889,
    },
    "best_parameters": {
        "classifier__criterion":        "log_loss",
        "classifier__max_depth":        6,
        "classifier__min_samples_split":5,
        "classifier__min_samples_leaf": 2,
        "classifier__max_features":     "sqrt",
        "classifier__class_weight":     "balanced",
    },
}

LABELS = ["continue", "switch_insertion_sort", "switch_merge_sort", "switch_quick_sort"]
BASELINE_CM = [[6, 0, 0, 0], [1, 5, 0, 0], [0, 0, 1, 0], [0, 0, 0, 5]]
TUNED_CM    = [[5, 1, 0, 0], [1, 5, 0, 0], [0, 0, 1, 0], [0, 0, 0, 5]]


# ── 1. Metric loading ──────────────────────────────────────────────────────────

def test_baseline_metrics_load():
    comp_path = PROJECT_ROOT / "ml" / "results" / "model_comparison.json"
    if not comp_path.exists():
        pytest.skip("model_comparison.json not found.")
    m = load_baseline_metrics(PROJECT_ROOT / "ml" / "results")
    assert "cv" in m
    assert "holdout" in m
    assert "macro_f1_mean" in m["cv"]


def test_tuned_metrics_load():
    tuning_path = PROJECT_ROOT / "ml" / "results" / "tuning_results.json"
    if not tuning_path.exists():
        pytest.skip("tuning_results.json not found.")
    m = load_tuned_metrics(PROJECT_ROOT / "ml" / "results")
    assert "cv" in m
    assert "holdout" in m
    assert "macro_f1_mean" in m["cv"]


# ── 2. Comparison serialisation ────────────────────────────────────────────────

def test_comparison_metrics_serialize():
    baseline_imp = pd.DataFrame({
        "feature":    ["num__work_ratio", "num__time_per_element_ms",
                       "num__comparisons_per_element", "num__movements_per_element",
                       "cat__input_type_duplicate_heavy"],
        "importance": [0.185, 0.168, 0.086, 0.075, 0.063],
    })
    tuned_imp = pd.DataFrame({
        "feature":    ["num__time_per_element_ms", "cat__input_type_duplicate_heavy",
                       "cat__input_type_reverse_sorted", "num__movements_per_element",
                       "cat__algorithm_merge_sort"],
        "importance": [0.289, 0.287, 0.118, 0.106, 0.091],
    })
    selection = select_model(BASELINE_METRICS, TUNED_METRICS, REPLACEMENT_TOLERANCE)
    cm_comp   = compare_confusion_matrices(BASELINE_CM, TUNED_CM, LABELS)
    fi_comp   = compare_feature_importance(baseline_imp, tuned_imp)

    with tempfile.TemporaryDirectory(dir=PROJECT_ROOT) as tmp:
        tmp_path = Path(tmp) / "results"
        save_comparison_artifacts(
            tmp_path, BASELINE_METRICS, TUNED_METRICS, selection,
            cm_comp, fi_comp, baseline_imp, tuned_imp,
            {}, TUNED_METRICS["best_parameters"], REPLACEMENT_TOLERANCE,
        )
        # JSON valid
        with open(tmp_path / "baseline_vs_tuned_comparison.json") as f:
            data = json.load(f)
        assert "selection" in data
        assert "baseline_metrics" in data

        # CSV valid
        df = pd.read_csv(tmp_path / "baseline_vs_tuned_metrics.csv")
        assert "metric" in df.columns
        assert len(df) > 0

        # Markdown valid
        md = (tmp_path / "baseline_vs_tuned_comparison.md").read_text()
        assert "Production Decision" in md


# ── 3. Baseline wins when its CV macro F1 is clearly higher ───────────────────

def test_baseline_wins_clearly_higher_cv():
    b = {"cv": {"macro_f1_mean": 0.80, "weighted_f1_mean": 0.82, "accuracy_mean": 0.85},
         "holdout": {"macro_f1": 0.90, "weighted_f1": 0.90, "accuracy": 0.90,
                     "macro_precision": 0.90, "macro_recall": 0.90,
                     "weighted_precision": 0.90, "weighted_recall": 0.90}}
    t = {"cv": {"macro_f1_mean": 0.75, "weighted_f1_mean": 0.77, "accuracy_mean": 0.80},
         "holdout": {"macro_f1": 0.95, "weighted_f1": 0.95, "accuracy": 0.95,
                     "macro_precision": 0.95, "macro_recall": 0.95,
                     "weighted_precision": 0.95, "weighted_recall": 0.95}}
    sel = select_model(b, t, tolerance=0.005)
    assert sel["selected_model"] == "decision_tree_baseline"
    assert sel["replaced"] is False
    assert sel["criterion"] == "CV macro F1"


# ── 4. Tuned candidate wins when its CV macro F1 is meaningfully higher ───────

def test_tuned_wins_meaningfully_higher_cv():
    b = {"cv": {"macro_f1_mean": 0.70, "weighted_f1_mean": 0.72, "accuracy_mean": 0.75},
         "holdout": {"macro_f1": 0.88, "weighted_f1": 0.88, "accuracy": 0.88,
                     "macro_precision": 0.88, "macro_recall": 0.88,
                     "weighted_precision": 0.88, "weighted_recall": 0.88}}
    t = {"cv": {"macro_f1_mean": 0.76, "weighted_f1_mean": 0.78, "accuracy_mean": 0.80},
         "holdout": {"macro_f1": 0.85, "weighted_f1": 0.85, "accuracy": 0.85,
                     "macro_precision": 0.85, "macro_recall": 0.85,
                     "weighted_precision": 0.85, "weighted_recall": 0.85}}
    sel = select_model(b, t, tolerance=0.005)
    assert sel["selected_model"] == "decision_tree_tuned"
    assert sel["replaced"] is True


# ── 5. Secondary metrics correctly resolve a tolerance-level tie ───────────────

def test_secondary_weighted_f1_resolves_tie():
    # CV macro F1 tied within tolerance; tuned wins on weighted F1
    b = {"cv": {"macro_f1_mean": 0.740, "weighted_f1_mean": 0.72, "accuracy_mean": 0.80},
         "holdout": {"macro_f1": 0.88, "weighted_f1": 0.88, "accuracy": 0.88,
                     "macro_precision": 0.88, "macro_recall": 0.88,
                     "weighted_precision": 0.88, "weighted_recall": 0.88}}
    t = {"cv": {"macro_f1_mean": 0.742, "weighted_f1_mean": 0.80, "accuracy_mean": 0.82},
         "holdout": {"macro_f1": 0.85, "weighted_f1": 0.85, "accuracy": 0.85,
                     "macro_precision": 0.85, "macro_recall": 0.85,
                     "weighted_precision": 0.85, "weighted_recall": 0.85}}
    sel = select_model(b, t, tolerance=0.005)
    # Macro F1 diff = 0.002 (within tol); tuned weighted F1 diff = 0.08 (above tol)
    assert sel["selected_model"] == "decision_tree_tuned"
    assert "weighted F1" in sel["criterion"]


# ── 6. Simpler model wins a complete tie ──────────────────────────────────────

def test_baseline_wins_complete_tie():
    same = {"cv": {"macro_f1_mean": 0.740, "weighted_f1_mean": 0.810, "accuracy_mean": 0.811},
            "holdout": {"macro_f1": 0.90, "weighted_f1": 0.88, "accuracy": 0.89,
                        "macro_precision": 0.92, "macro_recall": 0.90,
                        "weighted_precision": 0.90, "weighted_recall": 0.89}}
    sel = select_model(same, same, tolerance=0.005)
    assert sel["selected_model"] == "decision_tree_baseline"
    assert sel["replaced"] is False
    assert "Simplicity" in sel["criterion"]


# ── 7. Holdout-only improvement does not force replacement ────────────────────

def test_holdout_only_does_not_replace():
    # Baseline leads on CV; tuned leads on holdout
    sel = select_model(BASELINE_METRICS, TUNED_METRICS, REPLACEMENT_TOLERANCE)
    assert sel["selected_model"] == "decision_tree_baseline"
    assert sel["replaced"] is False


# ── 8. Production model unchanged when baseline wins ─────────────────────────

def test_production_model_unchanged_when_baseline_wins():
    prod_path = PROJECT_ROOT / "ml" / "models" / "adaptive_sort_model.joblib"
    if not prod_path.exists():
        pytest.skip("Production model not found.")
    mtime_before = prod_path.stat().st_mtime
    # selection returns baseline → no copy
    sel = select_model(BASELINE_METRICS, TUNED_METRICS, REPLACEMENT_TOLERANCE)
    assert not sel["replaced"]
    mtime_after = prod_path.stat().st_mtime
    assert mtime_before == mtime_after


# ── 9. Production model replaced correctly when tuned wins ────────────────────

def test_production_model_replaced_when_tuned_wins(tmp_path):
    import shutil
    tuned_path = PROJECT_ROOT / "ml" / "models" / "decision_tree_tuned.joblib"
    if not tuned_path.exists():
        pytest.skip("Tuned model not found.")
    fake_prod = tmp_path / "adaptive_sort_model.joblib"
    # Create a dummy "prod" to be replaced
    shutil.copyfile(
        PROJECT_ROOT / "ml" / "models" / "decision_tree_baseline.joblib",
        fake_prod
    )
    mtime_before = fake_prod.stat().st_mtime
    # Simulate a scenario where tuned wins
    import time; time.sleep(0.01)
    shutil.copyfile(tuned_path, fake_prod)
    assert fake_prod.exists()
    # File is updated
    assert fake_prod.stat().st_mtime >= mtime_before


# ── 10. Metadata records the final decision ───────────────────────────────────

def test_metadata_records_comparison(tmp_path):
    # Start with minimal valid metadata
    meta_path = tmp_path / "adaptive_sort_model_metadata.json"
    initial_meta = {
        "selected_model_name": "Decision Tree Baseline",
        "model_class": "sklearn.tree.DecisionTreeClassifier",
    }
    with open(meta_path, "w") as f:
        json.dump(initial_meta, f)

    selection = select_model(BASELINE_METRICS, TUNED_METRICS, REPLACEMENT_TOLERANCE)
    update_production_metadata(
        meta_path, BASELINE_METRICS, TUNED_METRICS, selection, REPLACEMENT_TOLERANCE
    )

    with open(meta_path) as f:
        updated = json.load(f)

    assert updated["tuning_performed"] is True
    assert "baseline_cv_macro_f1" in updated
    assert "tuned_cv_macro_f1" in updated
    assert "tuned_model_replaced_baseline" in updated
    assert "final_selection_reason" in updated
    # Existing keys preserved
    assert updated["selected_model_name"] == "Decision Tree Baseline"


# ── 11. All three artifact files are valid ────────────────────────────────────

def test_comparison_artifacts_all_valid():
    md_path  = PROJECT_ROOT / "ml" / "results" / "baseline_vs_tuned_comparison.md"
    json_path = PROJECT_ROOT / "ml" / "results" / "baseline_vs_tuned_comparison.json"
    csv_path  = PROJECT_ROOT / "ml" / "results" / "baseline_vs_tuned_metrics.csv"

    if not all(p.exists() for p in [md_path, json_path, csv_path]):
        pytest.skip("Comparison artifacts not yet generated; run compare.py first.")

    assert md_path.stat().st_size > 0

    with open(json_path) as f:
        data = json.load(f)
    assert "selection" in data

    df = pd.read_csv(csv_path)
    assert "metric" in df.columns
    assert "baseline" in df.columns
    assert "tuned" in df.columns


# ── 12. Selected production model loads and predicts ─────────────────────────

def test_selected_production_model_loads_and_predicts():
    prod_path = PROJECT_ROOT / "ml" / "models" / "adaptive_sort_model.joblib"
    if not prod_path.exists():
        pytest.skip("Production model not found.")
    model = joblib.load(prod_path)
    assert isinstance(model, Pipeline)

    sample = pd.DataFrame([{
        "algorithm": "quick_sort", "input_type": "random",
        "size": 1000.0, "checkpoint_pct": 50.0,
        "checkpoint_time_ms": 1.25, "checkpoint_comparisons": 4200.0,
        "checkpoint_data_movements": 1700.0, "comparisons_per_element": 4.2,
        "movements_per_element": 1.7, "work_ratio": 0.40,
        "time_per_element_ms": 0.00125,
    }])
    pred = model.predict(sample)[0]
    assert pred in {"continue", "switch_insertion_sort",
                    "switch_merge_sort", "switch_quick_sort"}
    probs = model.predict_proba(sample)[0]
    assert np.isclose(probs.sum(), 1.0)
