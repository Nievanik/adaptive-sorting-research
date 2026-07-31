"""
test_evaluate.py
----------------
Unit tests for the Phase 2.3 evaluation module.
"""

from __future__ import annotations

import sys
from pathlib import Path
import pytest
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier

# Ensure root path is present
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from ml.src.preprocess import prepare_training_data
from ml.src.evaluate import (
    class_distribution,
    evaluate_predictions,
    determine_cv_folds,
    cross_validate_pipeline,
)
from ml.train import build_training_pipeline


def test_class_distribution():
    y = ["a", "a", "b", "c", "c", "c"]
    df = class_distribution(y)

    # Check columns
    assert list(df.columns) == ["label", "count", "percentage"]

    # Ordered alphabetically
    assert list(df["label"]) == ["a", "b", "c"]
    assert list(df["count"]) == [2, 1, 3]

    # Percentages sum to approx 100
    assert pytest.approx(df["percentage"].sum()) == 100.0

    # Individual percentage check
    assert df.loc[df["label"] == "a", "percentage"].values[0] == pytest.approx(2 / 6 * 100.0)


def test_evaluate_predictions():
    y_true = ["a", "b", "a", "b", "c"]
    y_pred = ["a", "a", "a", "b", "c"]

    metrics = evaluate_predictions(y_true, y_pred)

    # Returns all expected metric keys (spaces and underscores)
    expected_keys = [
        "accuracy", "macro precision", "macro recall", "macro F1",
        "weighted precision", "weighted recall", "weighted F1",
        "classification report", "confusion matrix", "label order"
    ]
    for key in expected_keys:
        assert key in metrics

    # Label ordering is deterministic (alphabetical sorting of unique labels)
    assert metrics["label order"] == ["a", "b", "c"]

    # Confusion matrix dimensions match number of labels
    cm = metrics["confusion matrix"]
    assert cm.shape == (3, 3)


def test_determine_cv_folds():
    # Never exceeds smallest class count
    y1 = ["a", "a", "b", "b", "b", "b"]  # smallest count is 2
    assert determine_cv_folds(y1, max_folds=5) == 2

    # Never exceeds 5 by default
    y2 = ["a"] * 10 + ["b"] * 10
    assert determine_cv_folds(y2) == 5
    assert determine_cv_folds(y2, max_folds=10) == 10

    # Raises ValueError when any class has fewer than 2 samples
    y3 = ["a", "b", "b", "b"]
    with pytest.raises(ValueError) as excinfo:
        determine_cv_folds(y3)
    assert "fewer than 2 samples" in str(excinfo.value)

    # Empty raises ValueError
    with pytest.raises(ValueError):
        determine_cv_folds([])


def test_cross_validation_reproducible_and_scores():
    from sklearn.datasets import make_classification
    from sklearn.compose import ColumnTransformer

    X, y = make_classification(
        n_samples=30, n_features=5, n_informative=3, n_classes=3,
        random_state=42, n_clusters_per_class=1
    )
    label_map = {0: "continue", 1: "switch_merge_sort", 2: "switch_quick_sort"}
    y_str = [label_map[val] for val in y]

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", "passthrough", list(range(5)))
        ]
    )

    pipeline = Pipeline([
        ("preprocessor", preprocessor),
        ("classifier", RandomForestClassifier(random_state=42))
    ])

    cv_results = cross_validate_pipeline(pipeline, X, y_str, max_folds=3)

    assert cv_results["num_folds"] == 3
    # Returns one score per fold
    assert len(cv_results["accuracy_scores"]) == 3
    assert len(cv_results["macro_f1_scores"]) == 3
    assert len(cv_results["weighted_f1_scores"]) == 3

    # Results are reproducible
    cv_results_2 = cross_validate_pipeline(pipeline, X, y_str, max_folds=3)
    np.testing.assert_array_equal(cv_results["accuracy_scores"], cv_results_2["accuracy_scores"])
    np.testing.assert_array_equal(cv_results["macro_f1_scores"], cv_results_2["macro_f1_scores"])
    np.testing.assert_array_equal(cv_results["weighted_f1_scores"], cv_results_2["weighted_f1_scores"])


def test_cross_validation_leakage_safety():
    from sklearn.datasets import make_classification
    from sklearn.compose import ColumnTransformer

    X, y = make_classification(n_samples=20, n_features=2, n_informative=2, n_redundant=0, n_classes=2, random_state=42)
    y_str = ["class_a" if val == 0 else "class_b" for val in y]

    preprocessor = ColumnTransformer(transformers=[("num", "passthrough", [0, 1])])
    pipeline = Pipeline([
        ("preprocessor", preprocessor),
        ("classifier", RandomForestClassifier(random_state=42))
    ])

    # Unfitted check
    with pytest.raises(AttributeError):
        pipeline.named_steps["classifier"].classes_

    cv_results = cross_validate_pipeline(pipeline, X, y_str, max_folds=2)

    # After cross-validation, the original pipeline must remain unfitted
    with pytest.raises(AttributeError):
        pipeline.named_steps["classifier"].classes_

    assert "preprocessor" in pipeline.named_steps
    assert "classifier" in pipeline.named_steps


def test_cross_validation_real_dataset():
    csv_path = PROJECT_ROOT / "ml" / "data" / "processed" / "checkpoint_training.csv"
    if not csv_path.exists():
        pytest.skip("Real training dataset not found.")

    X, y, preprocessor = prepare_training_data(csv_path)
    pipeline = build_training_pipeline(preprocessor)

    cv_results = cross_validate_pipeline(pipeline, X, y, max_folds=5)

    assert cv_results["num_folds"] > 0
    assert len(cv_results["accuracy_scores"]) == cv_results["num_folds"]
    assert 0.0 <= cv_results["accuracy_mean"] <= 1.0
