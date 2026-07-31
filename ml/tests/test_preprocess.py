"""
test_preprocess.py
------------------
Unit tests for the Phase 2.1 preprocessing module.
"""

from __future__ import annotations

import sys
from pathlib import Path
import pandas as pd
import pytest
from sklearn.compose import ColumnTransformer

# Ensure root path is present
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from ml.src.preprocess import (
    load_dataset,
    split_features_labels,
    build_preprocessor,
    prepare_training_data,
    LEAKAGE_COLS
)

CSV_PATH = PROJECT_ROOT / "ml" / "data" / "processed" / "checkpoint_training.csv"


def test_load_dataset():
    if not CSV_PATH.exists():
        pytest.skip("Dataset file does not exist yet; run extract_dataset first.")
    df = load_dataset(CSV_PATH)
    assert isinstance(df, pd.DataFrame)
    assert not df.empty


def test_split_features_labels():
    # Construct a dummy dataframe with some features, target, and leakage columns
    df = pd.DataFrame({
        "algorithm": ["quick_sort", "merge_sort"],
        "size": [100, 200],
        "input_type": ["random", "sorted"],
        "checkpoint_pct": [50.0, 50.0],
        "continue_time_ms": [0.5, 0.6],
        "best_action": ["continue", "switch_quick_sort"],
        "speedup_vs_continue": [1.0, 1.2]
    })
    X, y = split_features_labels(df)
    
    # Check y
    assert isinstance(y, pd.Series)
    assert list(y) == ["continue", "switch_quick_sort"]
    
    # Check X
    assert "best_action" not in X.columns
    assert "continue_time_ms" not in X.columns
    assert "speedup_vs_continue" not in X.columns
    assert "algorithm" in X.columns
    assert "size" in X.columns
    assert X.shape[1] == 4  # algorithm, size, input_type, checkpoint_pct


def test_build_preprocessor():
    df = pd.DataFrame({
        "algorithm": ["quick_sort", "merge_sort"],
        "size": [100, 200],
        "input_type": ["random", "sorted"],
        "checkpoint_pct": [50.0, 50.0]
    })
    preprocessor = build_preprocessor(df)
    assert isinstance(preprocessor, ColumnTransformer)
    
    # Fit and transform
    transformed = preprocessor.fit_transform(df)
    # Categorical cols: algorithm (2 options), input_type (2 options) -> 4 hot columns
    # Numerical cols: size (1 col), checkpoint_pct (1 col) -> 2 cols
    # Total: 6 cols
    assert transformed.shape == (2, 6)


def test_prepare_training_data():
    if not CSV_PATH.exists():
        pytest.skip("Dataset file does not exist yet; run extract_dataset first.")
    X, y, preprocessor = prepare_training_data(CSV_PATH)
    assert isinstance(X, pd.DataFrame)
    assert isinstance(y, pd.Series)
    assert isinstance(preprocessor, ColumnTransformer)
    
    # Assert no leakage columns in X
    for col in LEAKAGE_COLS:
        assert col not in X.columns
    assert "best_action" not in X.columns
