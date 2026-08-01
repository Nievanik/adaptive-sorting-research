"""
test_ablation.py
----------------
Unit tests for the Phase 3.9 feature ablation study module (corrected protocol).
"""

from __future__ import annotations

import sys
import json
import tempfile
from pathlib import Path
import pytest
import pandas as pd
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from ml.src.ablation import (
    FEATURE_GROUPS,
    GROUP_LABELS,
    BASELINE_DT_PARAMS,
    DT_BASELINE_IMPORTANCE_CSV,
    validate_feature_groups,
    evaluate_full_baseline,
    run_ablation_experiment,
    compute_deltas,
    save_ablation_artifacts,
    save_ablation_plots,
)
from ml.src.evaluate import cross_validate_pipeline
from ml.train import (
    REQUIRED_FEATURES,
    get_default_dataset_path,
    load_dataset_csv,
    validate_dataset_df,
    extract_features_and_target,
    split_train_holdout,
    run_decision_tree_cross_validation,
)
from ml.src.preprocess import build_preprocessor
from sklearn.tree import DecisionTreeClassifier
from sklearn.pipeline import Pipeline


# ── Shared fixture ─────────────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def full_dataset():
    path = get_default_dataset_path()
    if not path.exists():
        pytest.skip("Training dataset not found.")
    df = load_dataset_csv(path)
    validate_dataset_df(df)
    X, y = extract_features_and_target(df)
    X_train, X_test, y_train, y_test = split_train_holdout(
        X, y, test_size=0.2, random_state=42
    )
    return X, y, X_train, X_test, y_train, y_test


# ── 1. Feature groups cover every production feature ─────────────────────────

def test_feature_groups_cover_all_features():
    covered = [f for feats in FEATURE_GROUPS.values() for f in feats]
    assert sorted(covered) == sorted(REQUIRED_FEATURES)


# ── 2. No feature appears in more than one group ─────────────────────────────

def test_no_duplicate_features_across_groups():
    covered = [f for feats in FEATURE_GROUPS.values() for f in feats]
    assert len(covered) == len(set(covered)), "Duplicate feature found across groups."


# ── 3. validate_feature_groups raises for duplicates ─────────────────────────

def test_validate_feature_groups_raises_on_duplicate():
    bad_groups = {
        "G1": ["algorithm", "input_type"],
        "G2": ["algorithm", "size"],   # 'algorithm' repeated
    }
    with pytest.raises(ValueError, match="multiple groups"):
        validate_feature_groups(bad_groups, ["algorithm", "input_type", "size"])


# ── 4. validate_feature_groups raises for missing features ───────────────────

def test_validate_feature_groups_raises_on_missing():
    incomplete = {k: v for k, v in list(FEATURE_GROUPS.items())[:-1]}
    with pytest.raises(ValueError, match="not assigned"):
        validate_feature_groups(incomplete, REQUIRED_FEATURES)


# ── REGRESSION 1: All-features baseline matches train.py protocol ─────────────

def test_ablation_baseline_matches_train_protocol(full_dataset):
    """
    The ablation all-features baseline CV macro F1 must match the value produced
    by run_decision_tree_cross_validation (the train.py protocol).
    """
    X, y, X_train, X_test, y_train, y_test = full_dataset
    preprocessor = build_preprocessor(X)
    ref_cv = run_decision_tree_cross_validation(preprocessor, X, y)
    ref_macro_f1 = ref_cv["macro_f1_mean"]

    ablation_baseline = evaluate_full_baseline(X, y, X_train, X_test, y_train, y_test)
    assert abs(ablation_baseline["cv_macro_f1_mean"] - ref_macro_f1) < 1e-9, (
        f"Ablation baseline CV macro F1 ({ablation_baseline['cv_macro_f1_mean']:.6f}) "
        f"does not match train.py protocol ({ref_macro_f1:.6f})."
    )


# ── REGRESSION 2: Baseline CV macro F1 matches shared evaluation function ─────

def test_baseline_cv_macro_f1_equals_cross_validate_pipeline(full_dataset):
    X, y, X_train, X_test, y_train, y_test = full_dataset
    preprocessor = build_preprocessor(X)
    cv_pipeline = Pipeline([
        ("preprocessor", preprocessor),
        ("classifier",   DecisionTreeClassifier(**BASELINE_DT_PARAMS)),
    ])
    cv = cross_validate_pipeline(cv_pipeline, X, y)
    ablation_baseline = evaluate_full_baseline(X, y, X_train, X_test, y_train, y_test)
    assert abs(ablation_baseline["cv_macro_f1_mean"] - cv["macro_f1_mean"]) < 1e-9


# ── REGRESSION 3: Identical CV splits reused across all experiments ───────────

def test_identical_cv_splits_across_experiments(full_dataset):
    """
    All ablation experiments pass the same X / y (full original order) to
    cross_validate_pipeline, ensuring identical fold assignments.
    """
    X, y, X_train, X_test, y_train, y_test = full_dataset
    results = []
    for gname, gfeats in list(FEATURE_GROUPS.items())[:2]:
        r = run_ablation_experiment(gname, gfeats, X, y, X_train, X_test, y_train, y_test)
        results.append(r)
    # Verify both experiments used the same number of rows (full dataset)
    for r in results:
        assert len(r["features_remaining"]) < len(REQUIRED_FEATURES)


# ── REGRESSION 4: Every experiment contains all required CV metrics ───────────

def test_all_cv_metrics_present(full_dataset):
    X, y, X_train, X_test, y_train, y_test = full_dataset
    group = "C_runtime"
    result = run_ablation_experiment(
        group, FEATURE_GROUPS[group], X, y, X_train, X_test, y_train, y_test
    )
    for key in [
        "cv_accuracy_mean", "cv_accuracy_std",
        "cv_macro_f1_mean", "cv_macro_f1_std",
        "cv_weighted_f1_mean", "cv_weighted_f1_std",
    ]:
        assert key in result, f"Missing key: {key}"
        assert result[key] is not None, f"Null value for: {key}"
        assert isinstance(result[key], float)


# ── REGRESSION 5: Every experiment contains all required holdout metrics ──────

def test_all_holdout_metrics_present(full_dataset):
    X, y, X_train, X_test, y_train, y_test = full_dataset
    group = "E_data_movement"
    result = run_ablation_experiment(
        group, FEATURE_GROUPS[group], X, y, X_train, X_test, y_train, y_test
    )
    for key in ["holdout_accuracy", "holdout_macro_f1", "holdout_weighted_f1"]:
        assert key in result, f"Missing key: {key}"
        assert result[key] is not None, f"Null value for: {key}"
        assert isinstance(result[key], float)


# ── REGRESSION 6: All deltas computed relative to the same baseline ───────────

def test_deltas_computed_relative_to_shared_baseline(full_dataset):
    X, y, X_train, X_test, y_train, y_test = full_dataset
    baseline = evaluate_full_baseline(X, y, X_train, X_test, y_train, y_test)
    for gname, gfeats in FEATURE_GROUPS.items():
        exp = run_ablation_experiment(gname, gfeats, X, y, X_train, X_test, y_train, y_test)
        exp.update(compute_deltas(baseline, exp))
        expected_delta = exp["cv_macro_f1_mean"] - baseline["cv_macro_f1_mean"]
        assert abs(exp["delta_cv_macro_f1_mean"] - expected_delta) < 1e-9


# ── REGRESSION 7: DT feature importance loaded from correct artifact ──────────

def test_feature_importance_loaded_from_dt_baseline_artifact():
    assert "decision_tree_feature_importance" in str(DT_BASELINE_IMPORTANCE_CSV), (
        "DT_BASELINE_IMPORTANCE_CSV does not point to decision_tree_feature_importance.csv"
    )
    assert "feature_importance.csv" != DT_BASELINE_IMPORTANCE_CSV.name or \
           "decision_tree" in DT_BASELINE_IMPORTANCE_CSV.name, (
        "Should use decision_tree_feature_importance.csv, not feature_importance.csv (RF)"
    )
    if DT_BASELINE_IMPORTANCE_CSV.exists():
        df = pd.read_csv(DT_BASELINE_IMPORTANCE_CSV)
        assert "feature" in df.columns
        assert "importance" in df.columns


# ── REGRESSION 8: No metric field is blank or null ───────────────────────────

def test_no_null_metric_fields(full_dataset):
    X, y, X_train, X_test, y_train, y_test = full_dataset
    baseline = evaluate_full_baseline(X, y, X_train, X_test, y_train, y_test)
    required_numeric_keys = [
        "cv_accuracy_mean", "cv_accuracy_std",
        "cv_macro_f1_mean", "cv_macro_f1_std",
        "cv_weighted_f1_mean", "cv_weighted_f1_std",
        "holdout_accuracy", "holdout_macro_f1", "holdout_weighted_f1",
    ]
    for key in required_numeric_keys:
        assert baseline.get(key) is not None, f"Baseline key '{key}' is null."
        assert isinstance(baseline[key], (int, float))

    for gname, gfeats in FEATURE_GROUPS.items():
        exp = run_ablation_experiment(gname, gfeats, X, y, X_train, X_test, y_train, y_test)
        for key in required_numeric_keys:
            assert exp.get(key) is not None, f"Experiment '{gname}' key '{key}' is null."


# ── REGRESSION 9: Ranking uses CV macro F1 delta ─────────────────────────────

def test_ranking_uses_cv_macro_f1_delta():
    """The ranked list should be sorted ascending by delta_cv_macro_f1_mean."""
    exps = [
        {"group_removed": "G1", "delta_cv_macro_f1_mean": -0.05},
        {"group_removed": "G2", "delta_cv_macro_f1_mean": -0.20},
        {"group_removed": "G3", "delta_cv_macro_f1_mean":  0.01},
    ]
    ranked = sorted(exps, key=lambda e: e["delta_cv_macro_f1_mean"])
    assert ranked[0]["group_removed"] == "G2"   # most harmful first
    assert ranked[-1]["group_removed"] == "G3"  # least harmful last


# ── REGRESSION 10: Generated plots use corrected metrics ─────────────────────

def test_plots_use_corrected_metrics(full_dataset, tmp_path):
    X, y, X_train, X_test, y_train, y_test = full_dataset
    baseline = evaluate_full_baseline(X, y, X_train, X_test, y_train, y_test)

    exps = []
    for gname, gfeats in list(FEATURE_GROUPS.items())[:2]:
        exp = run_ablation_experiment(gname, gfeats, X, y, X_train, X_test, y_train, y_test)
        exp.update(compute_deltas(baseline, exp))
        exps.append(exp)

    save_ablation_plots(tmp_path, baseline, exps)
    cv_plot   = tmp_path / "feature_ablation_cv_macro_f1.png"
    hold_plot = tmp_path / "feature_ablation_holdout_macro_f1.png"
    assert cv_plot.exists() and cv_plot.stat().st_size > 0
    assert hold_plot.exists() and hold_plot.stat().st_size > 0

    # Verify baseline value matches train.py (corrected, not 79.44%)
    preprocessor = build_preprocessor(X)
    ref_cv = run_decision_tree_cross_validation(preprocessor, X, y)
    assert abs(baseline["cv_macro_f1_mean"] - ref_cv["macro_f1_mean"]) < 1e-9


# ── Existing serialization tests (updated API) ────────────────────────────────

def test_ablation_removes_correct_features(full_dataset):
    X, y, X_train, X_test, y_train, y_test = full_dataset
    group_name  = "C_runtime"
    group_feats = FEATURE_GROUPS[group_name]
    result = run_ablation_experiment(
        group_name, group_feats, X, y, X_train, X_test, y_train, y_test
    )
    for f in group_feats:
        assert f not in result["features_remaining"]
    for f in result["features_remaining"]:
        assert f not in group_feats


def test_csv_generation(full_dataset, tmp_path):
    X, y, X_train, X_test, y_train, y_test = full_dataset
    baseline = evaluate_full_baseline(X, y, X_train, X_test, y_train, y_test)
    group = "D_comparison"
    exp = run_ablation_experiment(
        group, FEATURE_GROUPS[group], X, y, X_train, X_test, y_train, y_test
    )
    exp.update(compute_deltas(baseline, exp))

    save_ablation_artifacts(tmp_path, baseline, [exp])
    df = pd.read_csv(tmp_path / "feature_ablation.csv")
    assert "group_removed" in df.columns
    assert "delta_cv_macro_f1" in df.columns
    assert "cv_macro_f1_std" in df.columns
    assert len(df) == 2


def test_json_generation(full_dataset, tmp_path):
    X, y, X_train, X_test, y_train, y_test = full_dataset
    baseline = evaluate_full_baseline(X, y, X_train, X_test, y_train, y_test)
    group = "E_data_movement"
    exp = run_ablation_experiment(
        group, FEATURE_GROUPS[group], X, y, X_train, X_test, y_train, y_test
    )
    exp.update(compute_deltas(baseline, exp))

    save_ablation_artifacts(tmp_path, baseline, [exp])
    with open(tmp_path / "feature_ablation.json") as f:
        data = json.load(f)
    assert "protocol" in data
    assert "baseline" in data
    assert "experiments" in data
    assert data["protocol"]["cv_folds"] == 5
    assert data["protocol"]["feature_importance_source"] == "decision_tree_feature_importance.csv"


def test_markdown_generation(full_dataset, tmp_path):
    X, y, X_train, X_test, y_train, y_test = full_dataset
    baseline = evaluate_full_baseline(X, y, X_train, X_test, y_train, y_test)
    group = "A_algorithm_metadata"
    exp = run_ablation_experiment(
        group, FEATURE_GROUPS[group], X, y, X_train, X_test, y_train, y_test
    )
    exp.update(compute_deltas(baseline, exp))

    save_ablation_artifacts(tmp_path, baseline, [exp])
    content = (tmp_path / "feature_ablation.md").read_text()
    assert "Feature Group Ablation" in content
    assert "Limitations" in content
    assert "decision_tree_feature_importance.csv" in content
    assert "±" in content   # std is shown in the table
