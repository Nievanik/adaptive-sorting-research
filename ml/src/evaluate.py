"""
evaluate.py
-----------
Phase 2.3 — Model Evaluation

Provides leakage-safe evaluation helpers including class distribution calculation,
metric extraction, cross-validation wrapper, and evaluation reports.
"""

from __future__ import annotations

import sys
from pathlib import Path
from collections import Counter
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix,
)

# Ensure PROJECT_ROOT is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def class_distribution(y: pd.Series | np.ndarray | list) -> pd.DataFrame:
    """Return the number and percentage of samples in each class.

    Parameters
    ----------
    y : pd.Series | np.ndarray | list
        Target labels.

    Returns
    -------
    pd.DataFrame
        DataFrame containing columns: 'label', 'count', 'percentage'.
        Ordered alphabetically by label.
    """
    counts = Counter(y)
    total = sum(counts.values())

    data = []
    for label in sorted(counts.keys()):
        count = counts[label]
        percentage = (count / total) * 100.0 if total > 0 else 0.0
        data.append({
            "label": label,
            "count": count,
            "percentage": percentage
        })
    return pd.DataFrame(data)


def evaluate_predictions(y_true: pd.Series | np.ndarray | list, y_pred: pd.Series | np.ndarray | list) -> dict:
    """Calculate and return classification metrics.

    Parameters
    ----------
    y_true : pd.Series | np.ndarray | list
        True target labels.
    y_pred : pd.Series | np.ndarray | list
        Predicted target labels.

    Returns
    -------
    dict
        Dictionary containing metric keys (both with spaces and underscores for robustness).
    """
    labels = sorted(list(set(y_true) | set(y_pred)))

    accuracy = accuracy_score(y_true, y_pred)

    macro_precision = precision_score(y_true, y_pred, average="macro", zero_division=0)
    macro_recall = recall_score(y_true, y_pred, average="macro", zero_division=0)
    macro_f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)

    weighted_precision = precision_score(y_true, y_pred, average="weighted", zero_division=0)
    weighted_recall = recall_score(y_true, y_pred, average="weighted", zero_division=0)
    weighted_f1 = f1_score(y_true, y_pred, average="weighted", zero_division=0)

    report = classification_report(y_true, y_pred, labels=labels, output_dict=True, zero_division=0)
    matrix = confusion_matrix(y_true, y_pred, labels=labels)

    metrics = {
        "accuracy": accuracy,
        "macro precision": macro_precision,
        "macro_precision": macro_precision,
        "macro recall": macro_recall,
        "macro_recall": macro_recall,
        "macro F1": macro_f1,
        "macro_f1": macro_f1,
        "weighted precision": weighted_precision,
        "weighted_precision": weighted_precision,
        "weighted recall": weighted_recall,
        "weighted_recall": weighted_recall,
        "weighted F1": weighted_f1,
        "weighted_f1": weighted_f1,
        "classification report": report,
        "classification_report": report,
        "confusion matrix": matrix,
        "confusion_matrix": matrix,
        "label order": labels,
        "label_order": labels,
    }
    return metrics


def determine_cv_folds(y: pd.Series | np.ndarray | list, max_folds: int = 5) -> int:
    """Select a safe number of stratified cross-validation folds.

    Parameters
    ----------
    y : pd.Series | np.ndarray | list
        Target labels.
    max_folds : int, optional
        Maximum number of folds, by default 5.

    Returns
    -------
    int
        Safe fold count.

    Raises
    -------
    ValueError
        If any class has fewer than 2 samples or if target is empty.
    """
    counts = Counter(y)
    if not counts:
        raise ValueError("Target array/series is empty.")

    smallest_class_count = min(counts.values())
    if smallest_class_count < 2:
        raise ValueError(
            f"Cannot perform stratified cross-validation: at least one class has "
            f"fewer than 2 samples (smallest class count is {smallest_class_count})."
        )

    return min(max_folds, smallest_class_count)


def cross_validate_pipeline(pipeline: any, X: pd.DataFrame | np.ndarray, y: pd.Series | np.ndarray | list, max_folds: int = 5) -> dict:
    """Perform stratified cross-validation on an unfitted pipeline, fitting preprocessing within each fold.

    Parameters
    ----------
    pipeline : sklearn.pipeline.Pipeline
        The complete unfitted sklearn Pipeline.
    X : pd.DataFrame | np.ndarray
        Feature matrix.
    y : pd.Series | np.ndarray | list
        Target labels.
    max_folds : int, optional
        Maximum number of folds, by default 5.

    Returns
    -------
    dict
        Dictionary containing fold count, scores per fold, means, and standard deviations.
    """
    num_folds = determine_cv_folds(y, max_folds=max_folds)

    skf = StratifiedKFold(n_splits=num_folds, shuffle=True, random_state=42)

    accuracy_scores = []
    macro_f1_scores = []
    weighted_f1_scores = []

    # Standardize to pandas formats for robust index slicing
    X_data = X if isinstance(X, pd.DataFrame) else pd.DataFrame(X)
    y_data = y if isinstance(y, pd.Series) else pd.Series(y)

    for train_idx, test_idx in skf.split(X_data, y_data):
        X_train, X_test = X_data.iloc[train_idx], X_data.iloc[test_idx]
        y_train, y_test = y_data.iloc[train_idx], y_data.iloc[test_idx]

        # Clone to reset any fitted transformers/estimator within each fold
        fold_pipeline = clone(pipeline)
        fold_pipeline.fit(X_train, y_train)

        y_pred = fold_pipeline.predict(X_test)

        accuracy_scores.append(accuracy_score(y_test, y_pred))
        macro_f1_scores.append(f1_score(y_test, y_pred, average="macro", zero_division=0))
        weighted_f1_scores.append(f1_score(y_test, y_pred, average="weighted", zero_division=0))

    accuracy_scores = np.array(accuracy_scores)
    macro_f1_scores = np.array(macro_f1_scores)
    weighted_f1_scores = np.array(weighted_f1_scores)

    return {
        "num_folds": num_folds,
        "accuracy_scores": accuracy_scores,
        "macro_f1_scores": macro_f1_scores,
        "weighted_f1_scores": weighted_f1_scores,
        "accuracy_mean": np.mean(accuracy_scores),
        "accuracy_std": np.std(accuracy_scores),
        "macro_f1_mean": np.mean(macro_f1_scores),
        "macro_f1_std": np.std(macro_f1_scores),
        "weighted_f1_mean": np.mean(weighted_f1_scores),
        "weighted_f1_std": np.std(weighted_f1_scores),
    }


def print_evaluation_report(
    class_dist_df: pd.DataFrame,
    holdout_metrics: dict,
    cv_results: dict
) -> None:
    """Print a structured evaluation report.

    Parameters
    ----------
    class_dist_df : pd.DataFrame
        DataFrame with class distribution.
    holdout_metrics : dict
        Metrics dictionary from evaluate_predictions.
    cv_results : dict
        Metrics dictionary from cross_validate_pipeline.
    """
    print("\n================ EVALUATION REPORT ================")

    print("\n--- Class Distribution ---")
    print(class_dist_df.to_string(index=False))

    print("\n--- Holdout Evaluation Metrics ---")
    print(f"Accuracy:           {holdout_metrics['accuracy']:.4%}")
    print(f"Macro Precision:    {holdout_metrics['macro precision']:.4%}")
    print(f"Macro Recall:       {holdout_metrics['macro recall']:.4%}")
    print(f"Macro F1-Score:     {holdout_metrics['macro F1']:.4%}")
    print(f"Weighted Precision: {holdout_metrics['weighted precision']:.4%}")
    print(f"Weighted Recall:    {holdout_metrics['weighted recall']:.4%}")
    print(f"Weighted F1-Score:  {holdout_metrics['weighted F1']:.4%}")

    print("\n--- Confusion Matrix ---")
    print(f"Labels order: {holdout_metrics['label order']}")
    print(holdout_metrics['confusion matrix'])

    print(f"\n--- Stratified {cv_results['num_folds']}-Fold Cross-Validation ---")
    print(f"Accuracy:           {cv_results['accuracy_mean']:.4%} (+/- {cv_results['accuracy_std']:.4%})")
    print(f"Macro F1-Score:     {cv_results['macro_f1_mean']:.4%} (+/- {cv_results['macro_f1_std']:.4%})")
    print(f"Weighted F1-Score:  {cv_results['weighted_f1_mean']:.4%} (+/- {cv_results['weighted_f1_std']:.4%})")

    print("===================================================\n")
