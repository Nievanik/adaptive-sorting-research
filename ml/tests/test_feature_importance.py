"""
test_feature_importance.py
--------------------------
Unit tests for Phase 2.4 feature importance analysis.
"""

from __future__ import annotations

import sys
from pathlib import Path
import pytest
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder

# Ensure root path is present
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from ml.src.feature_importance import (
    load_pipeline,
    get_feature_names,
    compute_feature_importance,
    save_feature_importance,
    plot_feature_importance,
)

MODEL_PATH = PROJECT_ROOT / "ml" / "models" / "random_forest_baseline.joblib"


def test_feature_importance_with_dummy_pipeline(tmp_path):
    # Create dummy ColumnTransformer
    X_dummy = pd.DataFrame({
        "num_col1": [1.0, 2.0, 3.0],
        "cat_col": ["A", "B", "A"]
    })
    y_dummy = np.array([0, 1, 0])

    preprocessor = ColumnTransformer(
        transformers=[
            ("cat", OneHotEncoder(sparse_output=False), ["cat_col"]),
            ("num", "passthrough", ["num_col1"])
        ]
    )

    pipeline = Pipeline([
        ("preprocessor", preprocessor),
        ("classifier", RandomForestClassifier(random_state=42))
    ])

    # Fit dummy pipeline
    pipeline.fit(X_dummy, y_dummy)

    # 1. Recovered features check
    feature_names = get_feature_names(pipeline.named_steps["preprocessor"])
    assert len(feature_names) == 3  # cat_col_A, cat_col_B, num_col1
    assert "cat__cat_col_A" in feature_names or "cat_col_A" in "".join(feature_names)

    # 2. Compute importances
    df = compute_feature_importance(pipeline)
    assert isinstance(df, pd.DataFrame)
    assert list(df.columns) == ["feature", "importance"]

    # Number of features equals number of importances
    assert len(df) == len(feature_names)

    # Importance values sum approximately to 1
    assert pytest.approx(df["importance"].sum()) == 1.0

    # Returned DataFrame is sorted descending
    assert df["importance"].is_monotonic_decreasing

    # 3. CSV save check
    csv_out = tmp_path / "feat_imp.csv"
    save_feature_importance(df, csv_out)
    assert csv_out.exists()
    df_loaded = pd.read_csv(csv_out)
    assert len(df_loaded) == 3

    # 4. PNG plot check
    png_out = tmp_path / "feat_imp.png"
    plot_feature_importance(df, png_out, top_n=2)
    assert png_out.exists()


def test_feature_importance_with_real_pipeline():
    if not MODEL_PATH.exists():
        pytest.skip("Real baseline model joblib does not exist.")

    # 1. Model loads successfully
    pipeline = load_pipeline(MODEL_PATH)
    assert isinstance(pipeline, Pipeline)

    # 2. Pipeline contains both preprocessor and classifier
    assert "preprocessor" in pipeline.named_steps
    assert "classifier" in pipeline.named_steps

    # 3. Transformed feature names recovered correctly
    preprocessor = pipeline.named_steps["preprocessor"]
    feature_names = get_feature_names(preprocessor)
    assert len(feature_names) > 0

    # 4. Compute importances
    df = compute_feature_importance(pipeline)
    assert len(df) == len(feature_names)

    # Sums approximately to 1
    assert pytest.approx(df["importance"].sum(), abs=1e-5) == 1.0

    # Sorted descending
    assert df["importance"].is_monotonic_decreasing
