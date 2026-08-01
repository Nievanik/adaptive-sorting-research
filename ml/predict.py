"""
predict.py
----------
Phase 2.5 — Prediction and Inference Pipeline

Implements functions to load a fitted model pipeline, validate input schemas,
reject leakage features, predict single/batch records with confidence, and
provide CLI tools for inference.
"""

from __future__ import annotations

import sys
import argparse
import json
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline

# Ensure PROJECT_ROOT is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Supported prediction actions
SUPPORTED_LABELS = {
    "continue",
    "switch_insertion_sort",
    "switch_merge_sort",
    "switch_quick_sort"
}

# Ordered feature columns matching the training input schema
REQUIRED_FEATURES = [
    "algorithm",
    "input_type",
    "size",
    "checkpoint_pct",
    "checkpoint_time_ms",
    "checkpoint_comparisons",
    "checkpoint_data_movements",
    "comparisons_per_element",
    "movements_per_element",
    "work_ratio",
    "time_per_element_ms"
]

# Outcome or target leakage columns that must be rejected if supplied
LEAKAGE_AND_OUTCOME_COLS = [
    "best_action",
    "case",
    "continue_time_ms", "continue_comparisons", "continue_data_movements", "continue_overhead_time_ms",
    "switch_insertion_sort_time_ms", "switch_insertion_sort_comparisons", "switch_insertion_sort_data_movements", "switch_insertion_sort_overhead_time_ms",
    "switch_merge_sort_time_ms", "switch_merge_sort_comparisons", "switch_merge_sort_data_movements", "switch_merge_sort_overhead_time_ms",
    "switch_quick_sort_time_ms", "switch_quick_sort_comparisons", "switch_quick_sort_data_movements", "switch_quick_sort_overhead_time_ms",
    "best_action_total_ms", "speedup_vs_continue"
]


def load_model(model_path: str | Path) -> Pipeline:
    """Load the saved fitted sklearn Pipeline using joblib.

    Parameters
    ----------
    model_path : str | Path
        Path to the saved model file.

    Returns
    -------
    Pipeline
        Loaded model pipeline.
    """
    model_path = Path(model_path)
    if not model_path.exists():
        raise FileNotFoundError(f"Model file not found at: {model_path}")

    model = joblib.load(model_path)
    if not hasattr(model, "predict"):
        raise TypeError(f"Loaded object from {model_path} is not a valid predictor (lacks predict method).")
    return model


def validate_checkpoint_input(checkpoint: dict[str, object] | pd.Series | pd.DataFrame) -> pd.DataFrame:
    """Validate one checkpoint record before prediction.

    Parameters
    ----------
    checkpoint : dict | pd.Series | pd.DataFrame
        Single checkpoint record features.

    Returns
    -------
    pd.DataFrame
        One-row DataFrame containing valid features in training schema order.
    """
    if isinstance(checkpoint, pd.DataFrame):
        if len(checkpoint) != 1:
            raise ValueError(f"validate_checkpoint_input expects a single row, got {len(checkpoint)} rows.")
        df = checkpoint.copy()
    elif isinstance(checkpoint, pd.Series):
        df = pd.DataFrame([checkpoint.to_dict()])
    elif isinstance(checkpoint, dict):
        df = pd.DataFrame([checkpoint])
    else:
        raise TypeError("Input must be a dictionary, pandas Series, or a one-row pandas DataFrame.")

    # Check for target leakage or outcome columns
    supplied_leakage = [col for col in LEAKAGE_AND_OUTCOME_COLS if col in df.columns]
    if supplied_leakage:
        raise ValueError(f"Input contains forbidden leakage or outcome columns: {supplied_leakage}")

    # Check for missing required fields
    missing_fields = [field for field in REQUIRED_FEATURES if field not in df.columns]
    if missing_fields:
        raise ValueError(f"Input is missing required feature fields: {missing_fields}")

    # Reorder columns to deterministic training order
    df = df[REQUIRED_FEATURES].copy()

    # Validate 'size' field: positive integer/number
    size_val = df.iloc[0]["size"]
    try:
        size_numeric = float(size_val)
    except (ValueError, TypeError):
        raise ValueError(f"Field 'size' must be numeric, got: {size_val}")
    if size_numeric <= 0:
        raise ValueError(f"Field 'size' must be greater than 0, got: {size_numeric}")

    # Validate 'checkpoint_pct' field: in [0, 100]
    pct_val = df.iloc[0]["checkpoint_pct"]
    try:
        pct_numeric = float(pct_val)
    except (ValueError, TypeError):
        raise ValueError(f"Field 'checkpoint_pct' must be numeric, got: {pct_val}")
    if not (0.0 <= pct_numeric <= 100.0):
        raise ValueError(f"Field 'checkpoint_pct' must be within [0, 100], got: {pct_numeric}")

    # Validate remaining numeric fields
    numeric_fields = [
        "size", "checkpoint_pct", "checkpoint_time_ms", "checkpoint_comparisons",
        "checkpoint_data_movements", "comparisons_per_element", "movements_per_element",
        "work_ratio", "time_per_element_ms"
    ]
    non_negative_fields = [
        "checkpoint_time_ms", "checkpoint_comparisons", "checkpoint_data_movements",
        "comparisons_per_element", "movements_per_element", "work_ratio", "time_per_element_ms"
    ]

    for field in numeric_fields:
        val = df.iloc[0][field]
        try:
            num_val = float(val)
            df.loc[df.index[0], field] = num_val
        except (ValueError, TypeError):
            raise ValueError(f"Field '{field}' must be numeric, got: {val}")

    for field in non_negative_fields:
        num_val = df.iloc[0][field]
        if num_val < 0:
            raise ValueError(f"Field '{field}' must be non-negative, got: {num_val}")

    return df


def predict_action(model: Pipeline, checkpoint: dict[str, object] | pd.Series | pd.DataFrame) -> str:
    """Predict the best action for one checkpoint record.

    Parameters
    ----------
    model : Pipeline
        Fitted model pipeline.
    checkpoint : dict | pd.Series | pd.DataFrame
        Single checkpoint record.

    Returns
    -------
    str
        Predicted action name.
    """
    df_clean = validate_checkpoint_input(checkpoint)
    pred = model.predict(df_clean)[0]
    if pred not in SUPPORTED_LABELS:
        raise ValueError(f"Predicted action '{pred}' is not in supported labels: {SUPPORTED_LABELS}")
    return str(pred)


def predict_action_with_confidence(
    model: Pipeline,
    checkpoint: dict[str, object] | pd.Series | pd.DataFrame
) -> dict:
    """Predict the best action and confidence scores.

    Parameters
    ----------
    model : Pipeline
        Fitted model pipeline.
    checkpoint : dict | pd.Series | pd.DataFrame
        Single checkpoint record.

    Returns
    -------
    dict
        Dictionary containing 'predicted_action', 'confidence', and 'class_probabilities'.
    """
    df_clean = validate_checkpoint_input(checkpoint)

    # Check for predict_proba availability
    if not hasattr(model, "predict_proba"):
        raise AttributeError("The model pipeline does not support predict_proba().")

    probabilities = model.predict_proba(df_clean)[0]
    classes = model.classes_

    # Map classes to float probabilities
    class_probabilities = {str(cls): float(prob) for cls, prob in zip(classes, probabilities)}

    predicted_action = predict_action(model, df_clean)
    confidence = class_probabilities.get(predicted_action, 0.0)

    # Check sum approximately equals 1.0
    total_prob = sum(class_probabilities.values())
    if not np.isclose(total_prob, 1.0):
        raise ValueError(f"Class probabilities do not sum to approximately 1.0: {total_prob}")

    return {
        "predicted_action": predicted_action,
        "confidence": confidence,
        "class_probabilities": class_probabilities
    }


def predict_batch(model: Pipeline, checkpoints: pd.DataFrame | list[dict[str, object]]) -> pd.DataFrame:
    """Predict actions for multiple checkpoint records.

    Parameters
    ----------
    model : Pipeline
        Fitted model pipeline.
    checkpoints : pd.DataFrame | list[dict]
        DataFrame or list of feature dictionaries.

    Returns
    -------
    pd.DataFrame
        DataFrame with predictions and confidence scores.
    """
    if isinstance(checkpoints, list):
        df = pd.DataFrame(checkpoints)
    elif isinstance(checkpoints, pd.DataFrame):
        df = checkpoints.copy()
    else:
        raise TypeError("Input 'checkpoints' must be a list of dictionaries or a pandas DataFrame.")

    # Check for forbidden leakage/outcome columns
    supplied_leakage = [col for col in LEAKAGE_AND_OUTCOME_COLS if col in df.columns]
    if supplied_leakage:
        raise ValueError(f"Batch input contains forbidden leakage or outcome columns: {supplied_leakage}")

    # Check missing required fields
    missing_fields = [field for field in REQUIRED_FEATURES if field not in df.columns]
    if missing_fields:
        raise ValueError(f"Batch input is missing required fields: {missing_fields}")

    df_clean = df[REQUIRED_FEATURES].copy()

    # Enforce numeric types and value bounds on all rows
    numeric_fields = [
        "size", "checkpoint_pct", "checkpoint_time_ms", "checkpoint_comparisons",
        "checkpoint_data_movements", "comparisons_per_element", "movements_per_element",
        "work_ratio", "time_per_element_ms"
    ]
    non_negative_fields = [
        "checkpoint_time_ms", "checkpoint_comparisons", "checkpoint_data_movements",
        "comparisons_per_element", "movements_per_element", "work_ratio", "time_per_element_ms"
    ]

    for field in numeric_fields:
        try:
            df_clean[field] = df_clean[field].astype(float)
        except (ValueError, TypeError):
            # Pinpoint the offending row index
            for idx, val in enumerate(df_clean[field]):
                try:
                    float(val)
                except (ValueError, TypeError):
                    raise ValueError(f"Field '{field}' at row index {idx} is not numeric: {val}")

    # Check bounds
    invalid_size_rows = df_clean[df_clean["size"] <= 0]
    if not invalid_size_rows.empty:
        idx = invalid_size_rows.index[0]
        val = invalid_size_rows.loc[idx, "size"]
        raise ValueError(f"Field 'size' must be greater than 0, got: {val} at row index {idx}")

    invalid_pct_rows = df_clean[(df_clean["checkpoint_pct"] < 0.0) | (df_clean["checkpoint_pct"] > 100.0)]
    if not invalid_pct_rows.empty:
        idx = invalid_pct_rows.index[0]
        val = invalid_pct_rows.loc[idx, "checkpoint_pct"]
        raise ValueError(f"Field 'checkpoint_pct' must be within [0, 100], got: {val} at row index {idx}")

    for field in non_negative_fields:
        invalid_rows = df_clean[df_clean[field] < 0]
        if not invalid_rows.empty:
            idx = invalid_rows.index[0]
            val = invalid_rows.loc[idx, field]
            raise ValueError(f"Field '{field}' must be non-negative, got: {val} at row index {idx}")

    # Perform predictions
    predictions = model.predict(df_clean)
    result_df = pd.DataFrame({
        "predicted_action": predictions
    }, index=df.index)

    # Append confidence scores if available
    if hasattr(model, "predict_proba"):
        try:
            probabilities = model.predict_proba(df_clean)
            classes = list(model.classes_)
            confidences = []
            for pred_val, row_proba in zip(predictions, probabilities):
                pred_idx = classes.index(pred_val)
                confidences.append(float(row_proba[pred_idx]))
            result_df["confidence"] = confidences
        except Exception:
            pass

    return result_df


def parse_args(args: list[str]) -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Predict best adaptive sorting action for a checkpoint record."
    )
    # CLI mode 1: Predict from direct values
    parser.add_argument("--algorithm", type=str, help="Sorting algorithm name (e.g. quick_sort).")
    parser.add_argument("--input-type", type=str, help="Input array order/type (e.g. random).")
    parser.add_argument("--size", type=float, help="Size of the array.")
    parser.add_argument("--checkpoint-pct", type=float, help="Checkpoint percentage [0-100].")
    parser.add_argument("--checkpoint-time-ms", type=float, help="Accumulated checkpoint elapsed time.")
    parser.add_argument("--checkpoint-comparisons", type=float, help="Accumulated comparisons count.")
    parser.add_argument("--checkpoint-data-movements", type=float, help="Accumulated data movements count.")
    parser.add_argument("--comparisons-per-element", type=float, help="Accumulated comparisons per element.")
    parser.add_argument("--movements-per-element", type=float, help="Accumulated data movements per element.")
    parser.add_argument("--work-ratio", type=float, help="Ratio of work compared to expected bounds.")
    parser.add_argument("--time-per-element-ms", type=float, help="Average time per element.")

    # CLI mode 2: Predict from JSON file
    parser.add_argument("--input-json", type=str, help="Path to checkpoint features JSON file.")

    # Configs
    parser.add_argument(
        "--model-path",
        type=str,
        default=str(PROJECT_ROOT / "ml" / "models" / "adaptive_sort_model.joblib"),
        help="Path to the saved pipeline (default: production model)."
    )

    return parser.parse_args(args)


def main(args_list: list[str] | None = None) -> int:
    """Command line interface execution wrapper."""
    if args_list is None:
        args_list = sys.argv[1:]

    try:
        args = parse_args(args_list)

        # Load model first
        model = load_model(args.model_path)

        # Determine features source
        if args.input_json:
            json_path = Path(args.input_json)
            if not json_path.exists():
                print(f"Error: JSON input file does not exist: {json_path}", file=sys.stderr)
                return 1
            with open(json_path, "r") as f:
                checkpoint = json.load(f)
        else:
            # Build dictionary from CLI arguments
            checkpoint = {
                "algorithm": args.algorithm,
                "input_type": args.input_type,
                "size": args.size,
                "checkpoint_pct": args.checkpoint_pct,
                "checkpoint_time_ms": args.checkpoint_time_ms,
                "checkpoint_comparisons": args.checkpoint_comparisons,
                "checkpoint_data_movements": args.checkpoint_data_movements,
                "comparisons_per_element": args.comparisons_per_element,
                "movements_per_element": args.movements_per_element,
                "work_ratio": args.work_ratio,
                "time_per_element_ms": args.time_per_element_ms
            }

            # Check if any feature is missing
            missing_args = [k for k, v in checkpoint.items() if v is None]
            if missing_args:
                print(
                    f"Error: Please supply either --input-json or all feature arguments. "
                    f"Missing fields: {missing_args}",
                    file=sys.stderr
                )
                return 1

        # Run prediction
        res = predict_action_with_confidence(model, checkpoint)

        # Format output
        print(f"Predicted Action: {res['predicted_action']}")
        print(f"Confidence:       {res['confidence']:.4f}")
        print("Class Probabilities:")
        for cls, prob in sorted(res["class_probabilities"].items()):
            print(f"  - {cls}: {prob:.4f}")

        return 0

    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
