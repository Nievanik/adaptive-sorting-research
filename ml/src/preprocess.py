"""
preprocess.py
-------------
Phase 2.1 — Data Preprocessing

Provides utilities to load the checkpoint training dataset, split features
from labels, prevent target leakage, and build a preprocessing ColumnTransformer.
"""

from __future__ import annotations

import sys
from pathlib import Path
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder

# Ensure adaptive-sorting-research root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Leakage columns that must be dropped from feature matrix (X) to prevent target leakage
LEAKAGE_COLS = [
    "continue_time_ms", "continue_comparisons", "continue_data_movements", "continue_overhead_time_ms",
    "switch_insertion_sort_time_ms", "switch_insertion_sort_comparisons", "switch_insertion_sort_data_movements", "switch_insertion_sort_overhead_time_ms",
    "switch_merge_sort_time_ms", "switch_merge_sort_comparisons", "switch_merge_sort_data_movements", "switch_merge_sort_overhead_time_ms",
    "switch_quick_sort_time_ms", "switch_quick_sort_comparisons", "switch_quick_sort_data_movements", "switch_quick_sort_overhead_time_ms",
    "best_action_total_ms", "speedup_vs_continue"
]


def load_dataset(csv_path: str | Path) -> pd.DataFrame:
    """Load the dataset from a CSV file.

    Parameters
    ----------
    csv_path : str | Path
        Path to the training CSV file.

    Returns
    -------
    pd.DataFrame
        Loaded dataset.
    """
    return pd.read_csv(csv_path)


def split_features_labels(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """Separate features (X) and target labels (y), dropping target leakage columns.

    Parameters
    ----------
    df : pd.DataFrame
        Loaded dataset.

    Returns
    -------
    tuple[pd.DataFrame, pd.Series]
        Feature DataFrame (X) and target label Series (y).
    """
    y = df["best_action"]

    # Identify columns to drop (target column + leakage columns)
    to_drop = ["best_action"] + [col for col in LEAKAGE_COLS if col in df.columns]

    if "case" in df.columns:
        to_drop.append("case")

    X = df.drop(columns=to_drop)
    return X, y


def build_preprocessor(X: pd.DataFrame) -> ColumnTransformer:
    """Build a preprocessing pipeline using scikit-learn ColumnTransformer.

    Categorical features are OneHotEncoded.
    Numerical features are passed through unchanged.

    Parameters
    ----------
    X : pd.DataFrame
        Feature DataFrame.

    Returns
    -------
    ColumnTransformer
        Built preprocessor pipeline.
    """
    # Detect categorical columns: select all object/category/string data types
    categorical_cols = X.select_dtypes(include=["object", "category", "string"]).columns.tolist()

    # Detect numerical columns: select numeric data types
    numerical_cols = X.select_dtypes(include=["number"]).columns.tolist()

    preprocessor = ColumnTransformer(
        transformers=[
            ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), categorical_cols),
            ("num", "passthrough", numerical_cols)
        ]
    )
    return preprocessor


def prepare_training_data(csv_path: str | Path) -> tuple[pd.DataFrame, pd.Series, ColumnTransformer]:
    """Load, split, and build the preprocessor pipeline for the training data.

    Parameters
    ----------
    csv_path : str | Path
        Path to the training CSV file.

    Returns
    -------
    tuple[pd.DataFrame, pd.Series, ColumnTransformer]
        X (features), y (labels), and the built ColumnTransformer preprocessor.
    """
    df = load_dataset(csv_path)
    X, y = split_features_labels(df)
    preprocessor = build_preprocessor(X)
    return X, y, preprocessor
