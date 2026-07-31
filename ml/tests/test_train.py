"""
test_train.py
-------------
Unit tests for the Phase 2.2 training pipeline.
"""

from __future__ import annotations

import sys
from pathlib import Path
import pytest
import joblib
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

# Ensure root path is present
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from ml.src.preprocess import prepare_training_data

CSV_PATH = PROJECT_ROOT / "ml" / "data" / "processed" / "checkpoint_training.csv"
VALID_LABELS = {"continue", "switch_insertion_sort", "switch_merge_sort", "switch_quick_sort"}


def test_training_pipeline_construction_and_steps():
    if not CSV_PATH.exists():
        pytest.skip("Dataset file does not exist yet.")

    X, y, preprocessor = prepare_training_data(CSV_PATH)
    pipeline = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("classifier", RandomForestClassifier(random_state=42)),
        ]
    )

    # Check steps
    assert isinstance(pipeline, Pipeline)
    assert "preprocessor" in pipeline.named_steps
    assert "classifier" in pipeline.named_steps
    assert isinstance(pipeline.named_steps["classifier"], RandomForestClassifier)


def test_training_succeeds_and_evaluates():
    if not CSV_PATH.exists():
        pytest.skip("Dataset file does not exist yet.")

    X, y, preprocessor = prepare_training_data(CSV_PATH)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    pipeline = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("classifier", RandomForestClassifier(random_state=42)),
        ]
    )

    pipeline.fit(X_train, y_train)
    y_pred = pipeline.predict(X_test)

    # Length of predictions matches test labels
    assert len(y_pred) == len(y_test)

    # Predicted labels are valid known labels
    for pred in y_pred:
        assert pred in VALID_LABELS


def test_save_and_reload_pipeline():
    if not CSV_PATH.exists():
        pytest.skip("Dataset file does not exist yet.")

    X, y, preprocessor = prepare_training_data(CSV_PATH)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    pipeline = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("classifier", RandomForestClassifier(random_state=42)),
        ]
    )
    pipeline.fit(X_train, y_train)

    # Save to temp path inside the workspace
    temp_dir = PROJECT_ROOT / "ml" / "models" / "temp"
    temp_dir.mkdir(parents=True, exist_ok=True)
    temp_path = temp_dir / "test_rf_model.joblib"

    try:
        joblib.dump(pipeline, temp_path)
        assert temp_path.exists()

        # Reload
        reloaded_pipeline = joblib.load(temp_path)
        assert isinstance(reloaded_pipeline, Pipeline)

        # Reloaded model can make predictions
        y_pred = reloaded_pipeline.predict(X_test)
        assert len(y_pred) == len(y_test)
        for pred in y_pred:
            assert pred in VALID_LABELS
    finally:
        if temp_path.exists():
            temp_path.unlink()
        if temp_dir.exists():
            temp_dir.rmdir()
