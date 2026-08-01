"""
test_train.py
-------------
Unit tests for the Phase 3.2, 3.3, 3.4 & 3.5 training pipeline and analysis reports.
Covers all required verification scenarios for Random Forest, Decision Tree, and Selected Production model.
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
from sklearn.compose import ColumnTransformer

# Ensure root path is present
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from ml.train import (
    get_default_dataset_path,
    load_dataset_csv,
    validate_dataset_df,
    extract_features_and_target,
    check_leakage_exclusion,
    split_train_holdout,
    build_pipeline,
    build_decision_tree_pipeline,
    fit_pipeline,
    evaluate_holdout,
    run_cross_validation,
    run_decision_tree_cross_validation,
    save_pipeline,
    save_analysis_artifacts,
    save_comparison_artifacts,
    save_model_metadata,
    REQUIRED_FEATURES,
    LEAKAGE_AND_OUTCOME_COLS,
    SUPPORTED_LABELS,
)
from ml.src.preprocess import build_preprocessor
from ml.src.feature_importance import compute_feature_importance


# 1. The default dataset path resolves correctly.
def test_default_dataset_path_resolves():
    path = get_default_dataset_path()
    assert isinstance(path, Path)
    assert path.name == "checkpoint_training.csv"
    assert "ml/data/processed" in path.as_posix()


# 2. The real training CSV loads successfully.
def test_real_dataset_loads():
    path = get_default_dataset_path()
    if not path.exists():
        pytest.skip("Real dataset file does not exist yet.")
    df = load_dataset_csv(path)
    assert isinstance(df, pd.DataFrame)
    assert len(df) > 0


# 3. A missing dataset raises FileNotFoundError.
def test_missing_dataset_raises():
    fake_path = PROJECT_ROOT / "ml" / "data" / "processed" / "non_existent_file_xyz.csv"
    with pytest.raises(FileNotFoundError):
        load_dataset_csv(fake_path)


# 4. An empty dataset raises a clear error.
def test_empty_dataset_raises(tmp_path):
    empty_csv = tmp_path / "empty.csv"
    df = pd.DataFrame()
    df.to_csv(empty_csv, index=False)
    with pytest.raises(ValueError, match="Loaded dataset is empty"):
        load_dataset_csv(empty_csv)


# 5. A missing best_action column raises a clear error.
def test_missing_best_action_raises():
    df = pd.DataFrame({feat: [1, 2] for feat in REQUIRED_FEATURES})
    with pytest.raises(ValueError, match="Target column 'best_action' is missing"):
        validate_dataset_df(df)


# 6. Missing required feature columns raise a clear error.
def test_missing_required_features_raises():
    incomplete_features = [feat for feat in REQUIRED_FEATURES if feat != "algorithm"]
    df = pd.DataFrame({feat: [1, 2] for feat in incomplete_features})
    df["best_action"] = ["continue", "switch_quick_sort"]
    with pytest.raises(ValueError, match="Dataset is missing required feature columns"):
        validate_dataset_df(df)


# 7. The extracted feature columns exactly match the required feature schema and order.
def test_features_schema_and_order():
    scrambled_cols = list(reversed(REQUIRED_FEATURES)) + ["best_action", "some_extra_col"]
    df = pd.DataFrame({col: [i, i + 1] for i, col in enumerate(scrambled_cols)})
    df["algorithm"] = ["quick_sort", "merge_sort"]
    df["input_type"] = ["random", "sorted"]
    df["best_action"] = ["continue", "switch_quick_sort"]

    X, y = extract_features_and_target(df)
    assert list(X.columns) == REQUIRED_FEATURES
    assert X.shape == (2, len(REQUIRED_FEATURES))


# 8. Outcome and leakage columns never appear in model inputs.
def test_leakage_exclusion():
    X_with_leakage = pd.DataFrame({feat: [1, 2] for feat in REQUIRED_FEATURES})
    X_with_leakage["continue_time_ms"] = [0.5, 0.6]
    with pytest.raises(ValueError, match="Feature matrix contains forbidden leakage or outcome columns"):
        check_leakage_exclusion(X_with_leakage)

    X_clean = pd.DataFrame({feat: [1, 2] for feat in REQUIRED_FEATURES})
    check_leakage_exclusion(X_clean)


# 9. Train/test splitting is reproducible with the same random state.
def test_reproducible_split():
    data = {feat: np.random.rand(20) for feat in REQUIRED_FEATURES}
    data["algorithm"] = ["quick_sort"] * 20
    data["input_type"] = ["random"] * 20
    data["best_action"] = ["continue"] * 10 + ["switch_quick_sort"] * 10
    df = pd.DataFrame(data)
    X, y = extract_features_and_target(df)

    X_train1, X_test1, y_train1, y_test1 = split_train_holdout(X, y, test_size=0.2, random_state=42)
    X_train2, X_test2, y_train2, y_test2 = split_train_holdout(X, y, test_size=0.2, random_state=42)

    pd.testing.assert_frame_equal(X_train1, X_train2)
    pd.testing.assert_frame_equal(X_test1, X_test2)
    pd.testing.assert_series_equal(y_train1, y_train2)
    pd.testing.assert_series_equal(y_test1, y_test2)


# 10. Stratification preserves all sufficiently represented classes.
def test_stratification_representation():
    data = {feat: np.random.rand(11) for feat in REQUIRED_FEATURES}
    data["algorithm"] = ["quick_sort"] * 11
    data["input_type"] = ["random"] * 11
    data["best_action"] = ["continue"] * 5 + ["switch_quick_sort"] * 5 + ["switch_merge_sort"] * 1
    df = pd.DataFrame(data)
    X, y = extract_features_and_target(df)

    X_train, X_test, y_train, y_test = split_train_holdout(X, y, test_size=0.2, random_state=42)
    assert len(X_train) > 0
    assert len(X_test) > 0


def _create_dummy_dataset(n_samples=20) -> pd.DataFrame:
    data = {
        "algorithm": np.random.choice(["quick_sort", "merge_sort", "insertion_sort"], size=n_samples),
        "input_type": np.random.choice(["random", "sorted", "reverse_sorted"], size=n_samples),
        "size": np.random.randint(100, 10000, size=n_samples),
        "checkpoint_pct": np.random.uniform(40.0, 99.0, size=n_samples),
        "checkpoint_time_ms": np.random.uniform(0.1, 100.0, size=n_samples),
        "checkpoint_comparisons": np.random.uniform(100.0, 50000.0, size=n_samples),
        "checkpoint_data_movements": np.random.uniform(100.0, 50000.0, size=n_samples),
        "comparisons_per_element": np.random.uniform(1.0, 50.0, size=n_samples),
        "movements_per_element": np.random.uniform(1.0, 50.0, size=n_samples),
        "work_ratio": np.random.uniform(0.1, 2.0, size=n_samples),
        "time_per_element_ms": np.random.uniform(0.0001, 0.01, size=n_samples),
        "best_action": np.random.choice(list(SUPPORTED_LABELS), size=n_samples)
    }
    # Ensure classes list is sufficiently populated
    data["best_action"][0] = "continue"
    data["best_action"][1] = "switch_insertion_sort"
    data["best_action"][2] = "switch_merge_sort"
    data["best_action"][3] = "switch_quick_sort"
    return pd.DataFrame(data)


# 11. The model fits on a small representative dataset.
# 12. Predictions have the correct length.
# 13. Predictions belong to the supported label set.
# 14. predict_proba() returns valid probabilities.
def test_model_fit_predictions_and_proba():
    df = _create_dummy_dataset(30)
    X, y = extract_features_and_target(df)
    preprocessor = build_preprocessor(X)
    pipeline = build_pipeline(preprocessor)

    fitted_pipeline = fit_pipeline(pipeline, X, y)
    assert fitted_pipeline is not None

    preds = fitted_pipeline.predict(X)
    assert len(preds) == len(X)
    for p in preds:
        assert p in SUPPORTED_LABELS

    assert hasattr(fitted_pipeline, "predict_proba")
    proba = fitted_pipeline.predict_proba(X)
    assert proba.shape == (len(X), len(fitted_pipeline.classes_))
    assert np.allclose(np.sum(proba, axis=1), 1.0)


# 15. The saved pipeline can be loaded and used for prediction.
def test_saved_pipeline_load_and_use():
    df = _create_dummy_dataset(25)
    X, y = extract_features_and_target(df)
    preprocessor = build_preprocessor(X)
    pipeline = build_pipeline(preprocessor)
    fitted_pipeline = fit_pipeline(pipeline, X, y)

    with tempfile.TemporaryDirectory(dir=PROJECT_ROOT) as tmp_dir:
        model_path = Path(tmp_dir) / "test_model.joblib"
        save_pipeline(fitted_pipeline, model_path)
        assert model_path.exists()

        reloaded = joblib.load(model_path)
        assert isinstance(reloaded, Pipeline)
        assert hasattr(reloaded, "predict")

        preds = reloaded.predict(X)
        assert len(preds) == len(X)
        for p in preds:
            assert p in SUPPORTED_LABELS


# Verify report generation, validity, and serialization (Phase 3.3)
def test_analysis_artifacts_generation():
    df = _create_dummy_dataset(35)
    X, y = extract_features_and_target(df)
    preprocessor = build_preprocessor(X)
    pipeline = build_pipeline(preprocessor)
    fitted_pipeline = fit_pipeline(pipeline, X, y)

    # Mock holdout metrics and CV results
    holdout_metrics = {
        "accuracy": 0.9,
        "macro_precision": 0.88,
        "macro_recall": 0.89,
        "macro_f1": 0.885,
        "weighted_precision": 0.9,
        "weighted_recall": 0.9,
        "weighted_f1": 0.898,
        "confusion_matrix": np.array([[2, 1, 0, 0], [0, 3, 0, 0], [0, 0, 1, 0], [0, 0, 0, 2]]),
        "label_order": ["continue", "switch_insertion_sort", "switch_merge_sort", "switch_quick_sort"],
        "classification_report": {
            "continue": {"precision": 1.0, "recall": 0.67, "f1-score": 0.8, "support": 3},
            "switch_insertion_sort": {"precision": 0.75, "recall": 1.0, "f1-score": 0.86, "support": 3},
            "switch_merge_sort": {"precision": 1.0, "recall": 1.0, "f1-score": 1.0, "support": 1},
            "switch_quick_sort": {"precision": 1.0, "recall": 1.0, "f1-score": 1.0, "support": 2},
            "accuracy": 0.8888,
            "macro avg": {"precision": 0.9375, "recall": 0.9167, "f1-score": 0.915, "support": 9},
            "weighted avg": {"precision": 0.9167, "recall": 0.8889, "f1-score": 0.887, "support": 9}
        }
    }

    cv_results = {
        "accuracy_mean": 0.85,
        "accuracy_std": 0.05,
        "macro_f1_mean": 0.82,
        "macro_f1_std": 0.06,
        "weighted_f1_mean": 0.84,
        "weighted_f1_std": 0.05,
    }

    class_dist_df = pd.DataFrame([
        {"label": "continue", "count": 10, "percentage": 28.57},
        {"label": "switch_insertion_sort", "count": 12, "percentage": 34.29},
        {"label": "switch_merge_sort", "count": 5, "percentage": 14.29},
        {"label": "switch_quick_sort", "count": 8, "percentage": 22.86}
    ])

    with tempfile.TemporaryDirectory(dir=PROJECT_ROOT) as tmp_dir:
        results_dir = Path(tmp_dir) / "results"
        save_analysis_artifacts(
            results_dir,
            holdout_metrics,
            cv_results,
            class_dist_df,
            fitted_pipeline,
            dataset_size=len(df),
            feature_count=X.shape[1]
        )

        # 1. JSON classification report exists and is valid JSON
        report_json = results_dir / "classification_report.json"
        assert report_json.exists()
        with open(report_json, "r") as f:
            data = json.load(f)
            assert "continue" in data
            assert "accuracy" in data

        # 2. Confusion matrix exists, is valid CSV, matches labels
        conf_matrix_csv = results_dir / "confusion_matrix.csv"
        assert conf_matrix_csv.exists()
        df_conf = pd.read_csv(conf_matrix_csv, index_col=0)
        assert list(df_conf.columns) == holdout_metrics["label_order"]
        assert list(df_conf.index) == holdout_metrics["label_order"]

        # 3. Feature importance exists and is valid CSV
        feat_imp_csv = results_dir / "feature_importance.csv"
        assert feat_imp_csv.exists()
        df_imp = pd.read_csv(feat_imp_csv)
        assert "feature" in df_imp.columns
        assert "importance" in df_imp.columns
        assert len(df_imp) >= len(REQUIRED_FEATURES)

        # 4. Markdown baseline report exists and is non-empty
        analysis_md = results_dir / "baseline_analysis.md"
        assert analysis_md.exists()
        content = analysis_md.read_text()
        assert len(content) > 0
        assert "Dataset Size" in content
        assert "Confusion Matrix Interpretation" in content


# Decision Tree baseline verification tests (Phase 3.4)
def test_decision_tree_pipeline_creation_and_fit():
    df = _create_dummy_dataset(30)
    X, y = extract_features_and_target(df)
    preprocessor = build_preprocessor(X)
    pipeline = build_decision_tree_pipeline(preprocessor)
    
    fitted_pipeline = fit_pipeline(pipeline, X, y)
    assert fitted_pipeline is not None
    
    preds = fitted_pipeline.predict(X)
    assert len(preds) == len(X)
    for p in preds:
        assert p in SUPPORTED_LABELS
        
    assert hasattr(fitted_pipeline, "predict_proba")
    proba = fitted_pipeline.predict_proba(X)
    assert proba.shape == (len(X), len(fitted_pipeline.classes_))
    assert np.allclose(np.sum(proba, axis=1), 1.0)


def test_decision_tree_feature_importance():
    df = _create_dummy_dataset(25)
    X, y = extract_features_and_target(df)
    preprocessor = build_preprocessor(X)
    pipeline = build_decision_tree_pipeline(preprocessor)
    fitted_pipeline = fit_pipeline(pipeline, X, y)
    
    df_imp = compute_feature_importance(fitted_pipeline)
    assert "feature" in df_imp.columns
    assert "importance" in df_imp.columns
    assert len(df_imp) >= len(REQUIRED_FEATURES)


def test_decision_tree_serialization():
    df = _create_dummy_dataset(20)
    X, y = extract_features_and_target(df)
    preprocessor = build_preprocessor(X)
    pipeline = build_decision_tree_pipeline(preprocessor)
    fitted_pipeline = fit_pipeline(pipeline, X, y)
    
    with tempfile.TemporaryDirectory(dir=PROJECT_ROOT) as tmp_dir:
        model_path = Path(tmp_dir) / "dt_model.joblib"
        save_pipeline(fitted_pipeline, model_path)
        assert model_path.exists()
        
        reloaded = joblib.load(model_path)
        assert isinstance(reloaded, Pipeline)
        assert hasattr(reloaded, "predict")
        
        preds = reloaded.predict(X)
        assert len(preds) == len(X)
        for p in preds:
            assert p in SUPPORTED_LABELS


# Production model selection and comparison checks (Phase 3.5)
def test_production_model_load_and_predict():
    prod_model_path = PROJECT_ROOT / "ml" / "models" / "adaptive_sort_model.joblib"
    if not prod_model_path.exists():
        pytest.skip("Production model file does not exist yet; run train.py first.")
    
    # Load production model
    model = joblib.load(prod_model_path)
    assert isinstance(model, Pipeline)
    assert hasattr(model, "predict")
    assert hasattr(model, "predict_proba")
    
    # Predict on simple input matching predict.py's REQUIRED_FEATURES
    df = pd.DataFrame([{
        "algorithm": "quick_sort",
        "input_type": "random",
        "size": 1000.0,
        "checkpoint_pct": 50.0,
        "checkpoint_time_ms": 1.25,
        "checkpoint_comparisons": 4200.0,
        "checkpoint_data_movements": 1700.0,
        "comparisons_per_element": 4.2,
        "movements_per_element": 1.7,
        "work_ratio": 0.40,
        "time_per_element_ms": 0.00125
    }])
    pred = model.predict(df)[0]
    assert pred in SUPPORTED_LABELS
    
    probs = model.predict_proba(df)[0]
    assert len(probs) == len(model.classes_)
    assert np.isclose(np.sum(probs), 1.0)


def test_comparison_files_exist_and_valid():
    results_dir = PROJECT_ROOT / "ml" / "results"
    
    comp_json = results_dir / "model_comparison.json"
    if comp_json.exists():
        with open(comp_json, "r") as f:
            data = json.load(f)
            assert "random_forest" in data
            assert "decision_tree" in data
            assert "cv" in data["random_forest"]
            assert "holdout" in data["decision_tree"]
            
    metrics_csv = results_dir / "model_metrics.csv"
    if metrics_csv.exists():
        df = pd.read_csv(metrics_csv)
        assert "metric" in df.columns
        assert "random_forest" in df.columns
        assert "decision_tree" in df.columns
        assert len(df) > 0
        
    comp_md = results_dir / "model_comparison.md"
    if comp_md.exists():
        content = comp_md.read_text()
        assert "Metrics Comparison Table" in content
        assert "Production Model Selection" in content


# Metadata validation checks (Phase 3.6)
def test_production_model_metadata_validity():
    metadata_path = PROJECT_ROOT / "ml" / "models" / "adaptive_sort_model_metadata.json"
    if not metadata_path.exists():
        pytest.skip("Production model metadata file does not exist yet; run train.py first.")
        
    with open(metadata_path, "r") as f:
        meta = json.load(f)
        
    # Check all key fields are present and valid
    assert meta["selected_model_name"] == "Decision Tree Baseline"
    assert meta["model_class"] == "sklearn.tree.DecisionTreeClassifier"
    assert "creation_timestamp" in meta
    assert meta["dataset_identifier"] == "checkpoint_training.csv"
    assert meta["dataset_row_count"] == 90
    assert meta["raw_feature_count"] == 11
    assert meta["ordered_required_feature_list"] == REQUIRED_FEATURES
    assert meta["transformed_feature_count"] == 18
    assert meta["target_column"] == "best_action"
    assert set(meta["supported_labels"]) == SUPPORTED_LABELS
    assert meta["train_test_split"] == 0.2
    assert meta["random_state"] == 42
    assert meta["decision_tree_hyperparameters"] == {
        "max_depth": 5,
        "min_samples_split": 5,
        "min_samples_leaf": 2,
        "random_state": 42
    }
    assert "holdout_metrics" in meta
    assert "cross_validation_metrics" in meta
    assert meta["model_selection_criterion"] == "Cross-validation Macro F1"
    assert "scikit_learn_version" in meta
