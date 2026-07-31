"""
train.py
--------
Phase 2.2 — Baseline Model Training

Loads training data, applies the preprocessing pipeline, trains a Random Forest Classifier,
evaluates the model on a test split, and saves the trained model to disk.
"""

from __future__ import annotations

import sys
from pathlib import Path
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

# Ensure adaptive-sorting-research root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from ml.src.preprocess import prepare_training_data
from ml.src.evaluate import (
    class_distribution,
    evaluate_predictions,
    cross_validate_pipeline,
    print_evaluation_report,
)

CSV_PATH = PROJECT_ROOT / "ml" / "data" / "processed" / "checkpoint_training.csv"
MODEL_DIR = PROJECT_ROOT / "ml" / "models"
MODEL_OUTPUT_PATH = MODEL_DIR / "random_forest_baseline.joblib"


def build_training_pipeline(preprocessor) -> Pipeline:
    """Build the training pipeline with the preprocessor and classifier."""
    return Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("classifier", RandomForestClassifier(random_state=42)),
        ]
    )


def train_model() -> None:
    """Load preprocessed data, split into train/test, train a Random Forest,
    evaluate metrics, and save the model pipeline.
    """
    print(f"Loading and preprocessing data from {CSV_PATH}...")
    X, y, preprocessor = prepare_training_data(CSV_PATH)

    print("Calculating class distribution...")
    class_dist_df = class_distribution(y)

    print("Splitting dataset into train/test sets...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    print("Building scikit-learn pipeline...")
    pipeline = build_training_pipeline(preprocessor)

    print("Training model...")
    pipeline.fit(X_train, y_train)

    print("Evaluating model...")
    y_pred = pipeline.predict(X_test)
    holdout_metrics = evaluate_predictions(y_test, y_pred)

    print("Running stratified cross-validation...")
    unfitted_pipeline = build_training_pipeline(preprocessor)
    cv_results = cross_validate_pipeline(unfitted_pipeline, X, y)

    # Print integrated report
    print_evaluation_report(class_dist_df, holdout_metrics, cv_results)

    # Save model pipeline
    print(f"Saving model to {MODEL_OUTPUT_PATH}...")
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipeline, MODEL_OUTPUT_PATH)
    print("Model saved successfully.")


if __name__ == "__main__":
    train_model()

