"""
feature_importance.py
---------------------
Phase 2.4 — Feature Importance Analysis

Exposes functions to load the baseline model, extract feature importances,
map them to preprocessed feature names, save the results, and plot them.
"""

from __future__ import annotations

import sys
from pathlib import Path
import joblib  
import pandas as pd  
import numpy as np 
import matplotlib  
matplotlib.use('Agg')
import matplotlib.pyplot as plt 
from sklearn.pipeline import Pipeline 
from sklearn.compose import ColumnTransformer 

# Ensure PROJECT_ROOT is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def load_pipeline(model_path: str | Path) -> Pipeline:
    """Load the trained model pipeline from disk.

    Parameters
    ----------
    model_path : str | Path
        Path to the serialized joblib model file.

    Returns
    -------
    Pipeline
        Loaded scikit-learn Pipeline.
    """
    model_path = Path(model_path)
    if not model_path.exists():
        raise FileNotFoundError(f"No model found at {model_path}")
    pipeline = joblib.load(model_path)
    if not isinstance(pipeline, Pipeline):
        raise TypeError(f"Loaded object from {model_path} is not a scikit-learn Pipeline.")
    return pipeline


def get_feature_names(preprocessor: ColumnTransformer) -> list[str]:
    """Recover the transformed feature names after preprocessing.

    Parameters
    ----------
    preprocessor : ColumnTransformer
        The fitted preprocessor.

    Returns
    -------
    list[str]
        List of transformed feature names.
    """
    if not hasattr(preprocessor, "get_feature_names_out"):
        raise AttributeError("The preprocessor is not fitted or does not support get_feature_names_out.")
    return list(preprocessor.get_feature_names_out())


def compute_feature_importance(pipeline: Pipeline) -> pd.DataFrame:
    """Map every transformed feature with its corresponding Random Forest feature importance.

    Parameters
    ----------
    pipeline : Pipeline
        The fitted Pipeline containing 'preprocessor' and 'classifier'.

    Returns
    -------
    pd.DataFrame
        DataFrame with columns 'feature' and 'importance', sorted descending by importance.
    """
    if "preprocessor" not in pipeline.named_steps:
        raise KeyError("Pipeline does not contain a step named 'preprocessor'.")
    if "classifier" not in pipeline.named_steps:
        raise KeyError("Pipeline does not contain a step named 'classifier'.")

    preprocessor = pipeline.named_steps["preprocessor"]
    classifier = pipeline.named_steps["classifier"]

    if not hasattr(classifier, "feature_importances_"):
        raise AttributeError("Classifier is not fitted or does not expose feature_importances_.")

    feature_names = get_feature_names(preprocessor)
    importances = classifier.feature_importances_

    if len(feature_names) != len(importances):
        raise ValueError(
            f"Dimension mismatch: Number of features ({len(feature_names)}) "
            f"does not match number of importances ({len(importances)})."
        )

    df = pd.DataFrame({
        "feature": feature_names,
        "importance": importances
    })

    # Sort descending by importance
    return df.sort_values(by="importance", ascending=False).reset_index(drop=True)


def save_feature_importance(df: pd.DataFrame, output_csv: str | Path) -> None:
    """Save the feature importance DataFrame to a CSV file.

    Parameters
    ----------
    df : pd.DataFrame
        The feature importance DataFrame.
    output_csv : str | Path
        Output path for the CSV.
    """
    output_csv = Path(output_csv)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_csv, index=False)


def plot_feature_importance(df: pd.DataFrame, output_png: str | Path, top_n: int = 15) -> None:
    """Generate a horizontal bar chart of the Top N feature importances using Matplotlib.

    Parameters
    ----------
    df : pd.DataFrame
        The feature importance DataFrame.
    output_png : str | Path
        Output path for the PNG plot.
    top_n : int, optional
        Number of top features to plot, by default 15.
    """
    output_png = Path(output_png)
    output_png.parent.mkdir(parents=True, exist_ok=True)

    # Take the top N features
    plot_df = df.head(top_n).copy()

    # Sort ascending for plotting, so the most important feature is at the top of the horizontal bar chart
    plot_df = plot_df.iloc[::-1]

    plt.figure(figsize=(10, 6))
    plt.barh(plot_df["feature"], plot_df["importance"], color="#2c3e50", edgecolor="none")
    plt.xlabel("Importance Score")
    plt.ylabel("Feature")
    plt.title(f"Top {top_n} Feature Importances (Random Forest Baseline)")
    plt.tight_layout()
    plt.savefig(output_png, dpi=150)
    plt.close()


def main() -> None:
    """Run feature importance extraction, saving, and plotting."""
    model_path = PROJECT_ROOT / "ml" / "models" / "random_forest_baseline.joblib"
    output_csv = PROJECT_ROOT / "ml" / "results" / "feature_importance.csv"
    output_png = PROJECT_ROOT / "ml" / "results" / "feature_importance.png"

    print(f"Loading pipeline from {model_path}...")
    pipeline = load_pipeline(model_path)

    print("Computing feature importances...")
    importance_df = compute_feature_importance(pipeline)

    print(f"Saving feature importances to {output_csv}...")
    save_feature_importance(importance_df, output_csv)

    print(f"Generating feature importance plot to {output_png}...")
    plot_feature_importance(importance_df, output_png)

    print("\n================ Feature Importance Results ================")
    print(f"Total Transformed Features: {len(importance_df)}")
    print("\nTop 10 Most Important Features:")
    print(importance_df.head(10).to_string(index=False))
    print("============================================================\n")


if __name__ == "__main__":
    main()
