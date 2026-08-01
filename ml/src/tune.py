"""
tune.py
-------
Phase 3.7 — Hyperparameter Tuning for the Decision Tree Model

Optimises the Decision Tree hyperparameters using Stratified K-Fold cross-validation
on the training portion of the dataset. Evaluates the best candidate on the holdout
set and serialises tuning results and the best fitted pipeline.
"""

from __future__ import annotations

import sys
import json
import argparse
import time
from pathlib import Path
import joblib
import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import StratifiedKFold, RandomizedSearchCV
from sklearn.pipeline import Pipeline

# Ensure adaptive-sorting-research root is in sys.path
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
    evaluate_holdout,
    save_pipeline,
    df_to_markdown_simple,
    SUPPORTED_LABELS,
)
from ml.src.preprocess import build_preprocessor
from ml.src.evaluate import evaluate_predictions, class_distribution
from ml.src.feature_importance import compute_feature_importance


def parse_args(args_list: list[str] | None = None) -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Tune Decision Tree hyperparameters on the adaptive sorting dataset."
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default=str(get_default_dataset_path()),
        help="Path to the training CSV file."
    )
    parser.add_argument(
        "--test-size",
        type=float,
        default=0.2,
        help="Fraction of dataset to reserve as holdout test."
    )
    parser.add_argument(
        "--random-state",
        type=int,
        default=42,
        help="Random state seed for reproducibility."
    )
    parser.add_argument(
        "--cv-folds",
        type=int,
        default=5,
        help="Number of cross-validation folds."
    )
    parser.add_argument(
        "--n-iter",
        type=int,
        default=50,
        help="Number of parameter settings sampled in RandomizedSearchCV."
    )
    parser.add_argument(
        "--model-output",
        type=str,
        default=str(PROJECT_ROOT / "ml" / "models" / "decision_tree_tuned.joblib"),
        help="Output path for the serialized tuned pipeline."
    )
    parser.add_argument(
        "--results-dir",
        type=str,
        default=str(PROJECT_ROOT / "ml" / "results"),
        help="Directory to save comparative reports and artifacts."
    )
    return parser.parse_args(args_list)


def get_search_space() -> dict:
    """Return the hyperparameter search grid for DecisionTreeClassifier."""
    return {
        "classifier__criterion": ["gini", "entropy", "log_loss"],
        "classifier__max_depth": [3, 4, 5, 6, 8, None],
        "classifier__min_samples_split": [2, 4, 5, 8, 10],
        "classifier__min_samples_leaf": [1, 2, 3, 4, 5],
        "classifier__max_features": [None, "sqrt", "log2"],
        "classifier__class_weight": [None, "balanced"],
    }


def validate_cv_folds_safety(y: pd.Series, n_splits: int) -> None:
    """Validate that every class in y has at least n_splits samples to support stratified CV."""
    class_counts = y.value_counts()
    for cls, count in class_counts.items():
        if count < n_splits:
            raise ValueError(
                f"Cannot perform stratified {n_splits}-fold cross-validation. "
                f"Class '{cls}' has only {count} samples in the dataset."
            )


def build_tuning_pipeline(preprocessor) -> Pipeline:
    """Create an unfitted DecisionTree pipeline targeting search space variables."""
    return Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("classifier", DecisionTreeClassifier(random_state=42)),
        ]
    )


def run_tuning(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    preprocessor,
    cv_folds: int = 5,
    random_state: int = 42,
    n_iter: int = 50
) -> RandomizedSearchCV:
    """Setup and run RandomizedSearchCV on the training subset refitting on macro F1."""
    validate_cv_folds_safety(y_train, cv_folds)
    
    pipeline = build_tuning_pipeline(preprocessor)
    search_space = get_search_space()
    
    cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=random_state)
    
    scoring = {
        "accuracy": "accuracy",
        "macro_f1": "f1_macro",
        "weighted_f1": "f1_weighted"
    }

    # Ensure n_iter does not exceed search space configurations
    search = RandomizedSearchCV(
        estimator=pipeline,
        param_distributions=search_space,
        n_iter=n_iter,
        scoring=scoring,
        refit="macro_f1",
        cv=cv,
        random_state=random_state,
        n_jobs=-1
    )
    
    return search.fit(X_train, y_train)


def save_tuning_artifacts(
    results_dir: Path,
    search: RandomizedSearchCV,
    holdout_metrics: dict,
    dataset_row_count: int,
    class_dist_df: pd.DataFrame,
    random_state: int,
    test_size: float,
    duration_sec: float
) -> None:
    """Save all tuning outputs (joblib pipeline, JSON metadata, CSV iteration scores, holdout reports)."""
    results_dir.mkdir(parents=True, exist_ok=True)
    
    # Extract baseline metrics if baseline comparison file exists
    comp_json_path = results_dir / "model_comparison.json"
    baseline_metrics = {}
    if comp_json_path.exists():
        try:
            with open(comp_json_path, "r") as f:
                baseline_data = json.load(f)
                baseline_metrics = baseline_data.get("decision_tree", {})
        except Exception:
            pass

    # Extract best CV metrics
    cv_results = search.cv_results_
    best_idx = search.best_index_
    
    best_cv_macro_f1 = float(cv_results["mean_test_macro_f1"][best_idx])
    best_cv_weighted_f1 = float(cv_results["mean_test_weighted_f1"][best_idx])
    best_cv_accuracy = float(cv_results["mean_test_accuracy"][best_idx])

    # Convert NumPy arrays in holdout metrics
    serialized_holdout = {}
    for k, v in holdout_metrics.items():
        if isinstance(v, np.ndarray):
            serialized_holdout[k] = v.tolist()
        elif k == "classification_report":
            serialized_holdout[k] = v
        elif isinstance(v, (np.float32, np.float64)):
            serialized_holdout[k] = float(v)
        elif isinstance(v, (np.int32, np.int64)):
            serialized_holdout[k] = int(v)
        else:
            serialized_holdout[k] = v

    # 1. Save tuning_results.json
    results_json = {
        "search_method": "RandomizedSearchCV",
        "search_space": get_search_space(),
        "number_of_candidates": int(search.n_iter),
        "number_of_cv_folds": int(search.cv.n_splits),
        "primary_scoring_metric": "f1_macro",
        "best_parameters": search.best_params_,
        "best_cv_macro_f1": best_cv_macro_f1,
        "best_cv_weighted_f1": best_cv_weighted_f1,
        "best_cv_accuracy": best_cv_accuracy,
        "tuned_holdout_metrics": serialized_holdout,
        "baseline_decision_tree_metrics": baseline_metrics,
        "random_state": random_state,
        "test_size": test_size,
        "dataset_row_count": dataset_row_count,
        "class_distribution": class_dist_df.to_dict(orient="records"),
        "runtime_duration_seconds": duration_sec
    }
    
    with open(results_dir / "tuning_results.json", "w") as f:
        json.dump(results_json, f, indent=4)

    # 2. Save tuning_results.csv
    cv_df = pd.DataFrame(cv_results)
    cv_df.to_csv(results_dir / "tuning_results.csv", index=False)

    # 3. Save decision_tree_tuned_classification_report.json
    with open(results_dir / "decision_tree_tuned_classification_report.json", "w") as f:
        json.dump(holdout_metrics["classification_report"], f, indent=4)

    # 4. Save decision_tree_tuned_confusion_matrix.csv
    labels = holdout_metrics["label_order"]
    matrix_df = pd.DataFrame(holdout_metrics["confusion_matrix"], index=labels, columns=labels)
    matrix_df.to_csv(results_dir / "decision_tree_tuned_confusion_matrix.csv")

    # 5. Save decision_tree_tuned_feature_importance.csv
    importance_df = compute_feature_importance(search.best_estimator_)
    importance_df.to_csv(results_dir / "decision_tree_tuned_feature_importance.csv", index=False)

    # Compile baseline vs tuned metrics table
    baseline_holdout_acc = baseline_metrics.get("holdout", {}).get("accuracy", 0.0)
    baseline_holdout_macro = baseline_metrics.get("holdout", {}).get("macro_f1", 0.0)
    baseline_cv_macro = baseline_metrics.get("cv", {}).get("macro_f1_mean", 0.0)
    
    comparison_table_df = pd.DataFrame([
        ["Cross-validation Macro F1", f"{baseline_cv_macro:.4%}", f"{best_cv_macro_f1:.4%}"],
        ["Holdout Accuracy", f"{baseline_holdout_acc:.4%}", f"{holdout_metrics['accuracy']:.4%}"],
        ["Holdout Macro F1", f"{baseline_holdout_macro:.4%}", f"{holdout_metrics['macro_f1']:.4%}"]
    ], columns=["Metric", "Baseline Decision Tree", "Tuned Decision Tree Candidate"])
    
    metrics_table = df_to_markdown_simple(comparison_table_df)
    param_table = df_to_markdown_simple(pd.DataFrame([
        [k.replace("classifier__", ""), str(v)] for k, v in search.best_params_.items()
    ], columns=["Hyperparameter", "Selected Optimal Value"]))
    
    top_5_features = df_to_markdown_simple(importance_df.head(5))

    # 6. Save decision_tree_tuning_summary.md
    summary_md = f"""# Phase 3 — Hyperparameter Tuning Summary: Decision Tree

This report summarizes the tuning process and evaluates the optimal hyperparameter candidate.

## 1. Tuning Process Overview
- **Search Method:** RandomizedSearchCV
- **Iterations / Candidates Tested:** {search.n_iter}
- **Cross-Validation Folds:** {search.cv.n_splits}
- **Primary Refit Metric:** macro F1 (`f1_macro`)
- **Tuning Time:** {duration_sec:.2f} seconds

## 2. Selected Optimal Parameters
{param_table}

## 3. Comparative Evaluation
{metrics_table}

## 4. Per-Class Tuned Performance
{df_to_markdown_simple(pd.DataFrame([
    {
        "class": label,
        "precision": f"{holdout_metrics['classification_report'][label]['precision']:.4f}",
        "recall": f"{holdout_metrics['classification_report'][label]['recall']:.4f}",
        "f1-score": f"{holdout_metrics['classification_report'][label]['f1-score']:.4f}",
        "support": holdout_metrics["classification_report"][label]["support"]
    }
    for label in labels if label in holdout_metrics["classification_report"]
]))}

## 5. Confusion Matrix (Holdout)
```
{matrix_df.to_string()}
```

## 6. Top 5 Features (Tuned Candidate)
{top_5_features}
"""
    with open(results_dir / "decision_tree_tuning_summary.md", "w") as f:
        f.write(summary_md)


def main() -> int:
    """Command Line Interface Entry Point."""
    args = parse_args()
    
    try:
        print(f"Loading dataset from {args.dataset}...")
        df = load_dataset_csv(Path(args.dataset))
        validate_dataset_df(df)
        
        X, y = extract_features_and_target(df)
        check_leakage_exclusion(X)
        
        class_dist_df = class_distribution(y)
        
        print(f"Splitting dataset (test_size={args.test_size}, random_state={args.random_state})...")
        X_train, X_test, y_train, y_test = split_train_holdout(
            X, y, test_size=args.test_size, random_state=args.random_state
        )
        
        preprocessor = build_preprocessor(X)
        
        print(f"Running randomized hyperparameter search (n_iter={args.n_iter}, folds={args.cv_folds})...")
        start_time = time.time()
        search = run_tuning(
            X_train,
            y_train,
            preprocessor,
            cv_folds=args.cv_folds,
            random_state=args.random_state,
            n_iter=args.n_iter
        )
        duration = time.time() - start_time
        print(f"Tuning completed in {duration:.2f} seconds.")
        print(f"Best CV Macro F1: {search.best_score_:.4f}")
        print("Best Parameters:")
        for k, v in search.best_params_.items():
            print(f"  - {k}: {v}")
            
        print("Evaluating best tuned model on untouched holdout split...")
        holdout_metrics = evaluate_holdout(search.best_estimator_, X_test, y_test)
        print(f"Holdout Accuracy: {holdout_metrics['accuracy']:.4%}")
        print(f"Holdout Macro F1: {holdout_metrics['macro_f1']:.4%}")
        
        model_output_path = Path(args.model_output)
        print(f"Saving best tuned pipeline candidate to {model_output_path}...")
        save_pipeline(search.best_estimator_, model_output_path)
        
        results_dir = Path(args.results_dir)
        print(f"Saving comparative tuning artifacts to {results_dir}...")
        save_tuning_artifacts(
            results_dir,
            search,
            holdout_metrics,
            dataset_row_count=len(df),
            class_dist_df=class_dist_df,
            random_state=args.random_state,
            test_size=args.test_size,
            duration_sec=duration
        )
        print("Tuning artifacts created successfully.")
        return 0
        
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
