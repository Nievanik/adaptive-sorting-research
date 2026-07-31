"""
test_predict.py
----------------
Unit and integration tests for Phase 2.5 prediction and inference.
"""

from __future__ import annotations

import sys
import json
from pathlib import Path
import pytest
import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline

# Ensure root path is present
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from ml.predict import (
    load_model,
    validate_checkpoint_input,
    predict_action,
    predict_action_with_confidence,
    predict_batch,
    main,
    SUPPORTED_LABELS,
    REQUIRED_FEATURES,
)

CSV_PATH = PROJECT_ROOT / "ml" / "data" / "processed" / "checkpoint_training.csv"
MODEL_PATH = PROJECT_ROOT / "ml" / "models" / "random_forest_baseline.joblib"


def get_representative_record() -> dict:
    """Helper to extract a valid sample record from the processed dataset or fallback to a hardcoded dict."""
    if CSV_PATH.exists():
        df = pd.read_csv(CSV_PATH)
        row = df.iloc[0].to_dict()
        # Remove target/leakage
        clean_row = {k: v for k, v in row.items() if k in REQUIRED_FEATURES}
        return clean_row

    return {
        "algorithm": "quick_sort",
        "input_type": "random",
        "size": 1000.0,
        "checkpoint_pct": 20.0,
        "checkpoint_time_ms": 1.5,
        "checkpoint_comparisons": 3000.0,
        "checkpoint_data_movements": 1000.0,
        "comparisons_per_element": 3.0,
        "movements_per_element": 1.0,
        "work_ratio": 0.5,
        "time_per_element_ms": 0.0015
    }


def test_load_model():
    if not MODEL_PATH.exists():
        pytest.skip("Real baseline model joblib does not exist.")

    # valid saved model loads successfully
    model = load_model(MODEL_PATH)
    assert isinstance(model, Pipeline)
    assert hasattr(model, "predict")


def test_load_model_missing_raises_error():
    # missing model raises FileNotFoundError
    with pytest.raises(FileNotFoundError) as excinfo:
        load_model(PROJECT_ROOT / "ml" / "models" / "nonexistent_model_file.joblib")
    assert "Model file not found" in str(excinfo.value)


def test_validate_checkpoint_input():
    record = get_representative_record()

    # valid checkpoint dictionary is converted to a one-row DataFrame
    df = validate_checkpoint_input(record)
    assert isinstance(df, pd.DataFrame)
    assert len(df) == 1
    assert list(df.columns) == REQUIRED_FEATURES

    # Series validation
    series_df = validate_checkpoint_input(pd.Series(record))
    assert len(series_df) == 1
    assert list(series_df.columns) == REQUIRED_FEATURES

    # DataFrame validation
    df_df = validate_checkpoint_input(pd.DataFrame([record]))
    assert len(df_df) == 1
    assert list(df_df.columns) == REQUIRED_FEATURES


def test_validate_checkpoint_input_missing_raises_error():
    record = get_representative_record()
    del record["size"]

    # missing required feature raises ValueError
    with pytest.raises(ValueError) as excinfo:
        validate_checkpoint_input(record)
    assert "missing required feature fields" in str(excinfo.value)


def test_validate_checkpoint_input_rejects_leakage():
    record = get_representative_record()
    record["best_action"] = "switch_merge_sort"

    # outcome/leakage columns are rejected
    with pytest.raises(ValueError) as excinfo:
        validate_checkpoint_input(record)
    assert "forbidden leakage or outcome columns" in str(excinfo.value)


def test_validate_checkpoint_input_invalid_numeric():
    record = get_representative_record()
    record["size"] = "invalid_non_numeric"

    # invalid numeric values are rejected
    with pytest.raises(ValueError) as excinfo:
        validate_checkpoint_input(record)
    assert "must be numeric" in str(excinfo.value)


def test_validate_checkpoint_input_negative_values():
    fields_to_test = [
        "size", "checkpoint_time_ms", "checkpoint_comparisons",
        "checkpoint_data_movements", "comparisons_per_element", "movements_per_element",
        "work_ratio", "time_per_element_ms"
    ]

    for field in fields_to_test:
        record = get_representative_record()
        record[field] = -1.0

        # negative size, time, comparisons, or movements are rejected
        with pytest.raises(ValueError) as excinfo:
            validate_checkpoint_input(record)
        assert "greater than 0" in str(excinfo.value) or "must be non-negative" in str(excinfo.value)


def test_validate_checkpoint_input_invalid_pct():
    for pct in [-0.5, 100.1]:
        record = get_representative_record()
        record["checkpoint_pct"] = pct

        # invalid checkpoint percentage is rejected
        with pytest.raises(ValueError) as excinfo:
            validate_checkpoint_input(record)
        assert "must be within [0, 100]" in str(excinfo.value)


def test_predictions_deterministic_and_valid():
    if not MODEL_PATH.exists():
        pytest.skip("Real baseline model joblib does not exist.")

    model = load_model(MODEL_PATH)
    record = get_representative_record()

    # predict_action returns one of the four supported labels
    pred1 = predict_action(model, record)
    assert pred1 in SUPPORTED_LABELS

    # prediction is deterministic for the same record
    pred2 = predict_action(model, record)
    assert pred1 == pred2


def test_predict_action_with_confidence():
    if not MODEL_PATH.exists():
        pytest.skip("Real baseline model joblib does not exist.")

    model = load_model(MODEL_PATH)
    record = get_representative_record()

    # predict_action_with_confidence returns all required keys
    res = predict_action_with_confidence(model, record)
    assert "predicted_action" in res
    assert "confidence" in res
    assert "class_probabilities" in res

    # class probabilities sum approximately to 1
    probs = res["class_probabilities"]
    assert pytest.approx(sum(probs.values())) == 1.0

    # predicted confidence matches the predicted class probability
    assert probs[res["predicted_action"]] == res["confidence"]


def test_predict_batch():
    if not MODEL_PATH.exists():
        pytest.skip("Real baseline model joblib does not exist.")

    model = load_model(MODEL_PATH)
    record = get_representative_record()

    # Duplicate records to form a batch
    batch = [record.copy(), record.copy(), record.copy()]
    batch[1]["size"] = 9999.0  # make one slightly different

    result_df = predict_batch(model, batch)

    # batch prediction preserves row count and order
    assert len(result_df) == 3
    assert "predicted_action" in result_df.columns
    assert "confidence" in result_df.columns

    # Verify order is preserved
    single_res_0 = predict_action(model, batch[0])
    single_res_1 = predict_action(model, batch[1])
    assert result_df.iloc[0]["predicted_action"] == single_res_0
    assert result_df.iloc[1]["predicted_action"] == single_res_1


def test_no_refit_or_mutation():
    if not MODEL_PATH.exists():
        pytest.skip("Real baseline model joblib does not exist.")

    model = load_model(MODEL_PATH)
    record = get_representative_record()

    # Check fitted state marker before
    marker_before = model.named_steps["classifier"].estimators_[0].tree_.threshold

    _ = predict_action(model, record)

    # Check fitted state marker after
    marker_after = model.named_steps["classifier"].estimators_[0].tree_.threshold
    np.testing.assert_array_equal(marker_before, marker_after)


def test_cli_success(tmp_path):
    if not MODEL_PATH.exists():
        pytest.skip("Real baseline model joblib does not exist.")

    # Test via direct arguments
    args = [
        "--algorithm", "quick_sort",
        "--input-type", "random",
        "--size", "1000",
        "--checkpoint-pct", "50",
        "--checkpoint-time-ms", "1.25",
        "--checkpoint-comparisons", "4200",
        "--checkpoint-data-movements", "1700",
        "--comparisons-per-element", "4.2",
        "--movements-per-element", "1.7",
        "--work-ratio", "0.40",
        "--time-per-element-ms", "0.00125",
        "--model-path", str(MODEL_PATH)
    ]
    status = main(args)
    assert status == 0

    # Test via JSON file
    record = get_representative_record()
    json_file = tmp_path / "checkpoint.json"
    with open(json_file, "w") as f:
        json.dump(record, f)

    args_json = [
        "--input-json", str(json_file),
        "--model-path", str(MODEL_PATH)
    ]
    status_json = main(args_json)
    assert status_json == 0


def test_cli_clean_failure():
    # Test missing CLI argument fields
    args = [
        "--algorithm", "quick_sort",
        "--size", "1000"
    ]
    status = main(args)
    assert status == 1

    # Test invalid values causing exception
    args_invalid = [
        "--algorithm", "quick_sort",
        "--input-type", "random",
        "--size", "-500",  # negative size is invalid
        "--checkpoint-pct", "50",
        "--checkpoint-time-ms", "1.25",
        "--checkpoint-comparisons", "4200",
        "--checkpoint-data-movements", "1700",
        "--comparisons-per-element", "4.2",
        "--movements-per-element", "1.7",
        "--work-ratio", "0.40",
        "--time-per-element-ms", "0.00125",
        "--model-path", str(MODEL_PATH)
    ]
    status_invalid = main(args_invalid)
    assert status_invalid == 1
