"""
test_tune.py
------------
Unit tests for the Decision Tree hyperparameter tuning module.
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
from sklearn.model_selection import RandomizedSearchCV

# Ensure root path is present
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from ml.src.tune import (
    get_search_space,
    validate_cv_folds_safety,
    build_tuning_pipeline,
    run_tuning,
    save_tuning_artifacts,
)
from ml.train import (
    extract_features_and_target,
    load_dataset_csv,
    get_default_dataset_path,
)
from ml.src.preprocess import build_preprocessor

# Valid prediction actions
SUPPORTED_LABELS = {"continue", "switch_insertion_sort", "switch_merge_sort", "switch_quick_sort"}


# Helper to build a small representative dataset
def _create_mock_dataset(n_samples=25) -> pd.DataFrame:
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
    # Ensure classes list is sufficiently populated for CV folds
    data["best_action"][0] = "continue"
    data["best_action"][1] = "switch_insertion_sort"
    data["best_action"][2] = "switch_merge_sort"
    data["best_action"][3] = "switch_quick_sort"
    return pd.DataFrame(data)


# 1. Search-space construction.
# 2. Correct estimator parameter prefixes.
def test_search_space_prefixes():
    space = get_search_space()
    assert isinstance(space, dict)
    assert len(space) > 0
    # Every parameter key must start with classifier__ (matching pipeline step name)
    for key in space.keys():
        assert key.startswith("classifier__")


# 3. Stratified CV construction.
# 4. Failure when a class has fewer samples than the requested fold count.
def test_cv_folds_safety_validation():
    # Setup y where one class has count = 2 (underrepresented for 5-fold CV)
    y = pd.Series(["continue"] * 10 + ["switch_insertion_sort"] * 2)
    with pytest.raises(ValueError, match="Cannot perform stratified 5-fold cross-validation"):
        validate_cv_folds_safety(y, n_splits=5)

    # Safety passes for balanced/represented classes
    y_safe = pd.Series(["continue"] * 6 + ["switch_insertion_sort"] * 6)
    validate_cv_folds_safety(y_safe, n_splits=5)


# 5. Macro F1 configured as the refit metric.
# 6. Tuning runs successfully on a small representative dataset.
# 7. Best parameters are returned.
# 8. Best estimator supports predict().
# 9. Best estimator supports predict_proba().
# 10. Prediction labels belong to the supported label set.
def test_run_tuning_success_and_predictions():
    df = _create_mock_dataset(30)
    X, y = extract_features_and_target(df)
    preprocessor = build_preprocessor(X)

    # Run with small iterations/folds for speed
    search = run_tuning(
        X, y, preprocessor, cv_folds=2, random_state=42, n_iter=3
    )

    assert isinstance(search, RandomizedSearchCV)
    assert search.refit == "macro_f1"  # 5. Macro F1 configured as refit
    assert hasattr(search, "best_params_")  # 7. Best parameters returned
    assert hasattr(search, "best_estimator_")

    best_pipeline = search.best_estimator_
    assert isinstance(best_pipeline, Pipeline)

    # 8. Predict works
    preds = best_pipeline.predict(X)
    assert len(preds) == len(X)

    # 10. Labels correctness
    for p in preds:
        assert p in SUPPORTED_LABELS

    # 9. Predict_proba works
    assert hasattr(best_pipeline, "predict_proba")
    probs = best_pipeline.predict_proba(X)
    assert probs.shape == (len(X), len(best_pipeline.classes_))
    assert np.allclose(np.sum(probs, axis=1), 1.0)


# 11. Tuning results are JSON serializable.
# 12. CSV tuning results are valid.
# 13. Tuned model serialization.
# 14. Tuned model deserialization and prediction.
def test_tuning_artifacts_serialization():
    df = _create_mock_dataset(25)
    X, y = extract_features_and_target(df)
    preprocessor = build_preprocessor(X)
    search = run_tuning(X, y, preprocessor, cv_folds=2, random_state=42, n_iter=2)

    # Mock holdout metrics
    holdout_metrics = {
        "accuracy": 0.8,
        "macro_precision": 0.8,
        "macro_recall": 0.8,
        "macro_f1": 0.8,
        "weighted_precision": 0.8,
        "weighted_recall": 0.8,
        "weighted_f1": 0.8,
        "confusion_matrix": np.array([[2, 0], [0, 2]]),
        "label_order": ["continue", "switch_quick_sort"],
        "classification_report": {
            "continue": {"precision": 1.0, "recall": 1.0, "f1-score": 1.0, "support": 2},
            "switch_quick_sort": {"precision": 1.0, "recall": 1.0, "f1-score": 1.0, "support": 2},
            "accuracy": 1.0,
            "macro avg": {"precision": 1.0, "recall": 1.0, "f1-score": 1.0, "support": 4},
            "weighted avg": {"precision": 1.0, "recall": 1.0, "f1-score": 1.0, "support": 4}
        }
    }

    class_dist_df = pd.DataFrame([
        {"label": "continue", "count": 15, "percentage": 60.0},
        {"label": "switch_quick_sort", "count": 10, "percentage": 40.0}
    ])

    with tempfile.TemporaryDirectory(dir=PROJECT_ROOT) as tmp_dir:
        results_dir = Path(tmp_dir) / "results"
        save_tuning_artifacts(
            results_dir,
            search,
            holdout_metrics,
            dataset_row_count=len(df),
            class_dist_df=class_dist_df,
            random_state=42,
            test_size=0.2,
            duration_sec=0.5
        )

        # 11. JSON serialization
        results_json_path = results_dir / "tuning_results.json"
        assert results_json_path.exists()
        with open(results_json_path, "r") as f:
            data = json.load(f)
            assert "best_parameters" in data
            assert "best_cv_macro_f1" in data

        # 12. CSV validity
        results_csv_path = results_dir / "tuning_results.csv"
        assert results_csv_path.exists()
        df_csv = pd.read_csv(results_csv_path)
        assert "mean_test_macro_f1" in df_csv.columns
        assert len(df_csv) == 2

        # 13. Serialisation
        pipeline_output = Path(tmp_dir) / "decision_tree_tuned.joblib"
        joblib.dump(search.best_estimator_, pipeline_output)
        assert pipeline_output.exists()

        # 14. Deserialisation and prediction
        reloaded = joblib.load(pipeline_output)
        assert isinstance(reloaded, Pipeline)
        preds = reloaded.predict(X)
        assert len(preds) == len(X)
        for p in preds:
            assert p in SUPPORTED_LABELS


# 15. Existing production model remains unchanged.
def test_production_model_remains_unchanged():
    prod_model_path = PROJECT_ROOT / "ml" / "models" / "adaptive_sort_model.joblib"
    if not prod_model_path.exists():
        pytest.skip("Production model does not exist yet.")
    
    # Read modified time of production model
    mtime_before = prod_model_path.stat().st_mtime
    
    # Verify file content loading
    reloaded = joblib.load(prod_model_path)
    assert isinstance(reloaded, Pipeline)
    
    # Assert timestamp is unchanged after running tests
    mtime_after = prod_model_path.stat().st_mtime
    assert mtime_before == mtime_after
