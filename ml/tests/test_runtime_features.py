"""
test_runtime_features.py
------------------------
Phase 7.1 — Tests for ml/src/runtime_features.py

Covers:
  - Valid full payload (happy path)
  - Exact required feature keys
  - Deterministic feature generation
  - Feature order independence (dict-based)
  - Each supported starting algorithm
  - Each supported input distribution
  - Zero checkpoint_time and zero moves (zero denominator cases)
  - Invalid array_size
  - Negative comparisons
  - Negative moves
  - Missing required values (TypeError from Python)
  - Unknown categorical values
  - NaN inputs
  - Positive and negative infinity
  - Forbidden leakage fields not present in output
  - Every valid prediction action (validate_predicted_action)
  - Invalid prediction actions
  - continue action behavior
  - Switching to each supported algorithm
  - Switch-to-self detection
  - Cross-validation: generated features match a real training CSV row
"""

from __future__ import annotations

import math
import pandas as pd
import pytest
from pathlib import Path

from ml.src.runtime_features import (
    FALLBACK_ACTION,
    FORBIDDEN_LEAKAGE_FIELDS,
    REQUIRED_FEATURES,
    SUPPORTED_ALGORITHMS,
    SUPPORTED_INPUT_TYPES,
    VALID_PREDICTION_ACTIONS,
    build_runtime_features,
    validate_predicted_action,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _valid_kwargs(**overrides) -> dict:
    """Return a complete valid set of kwargs, with optional overrides."""
    defaults = dict(
        current_algorithm="quick_sort",
        input_type="random",
        array_size=1000,
        checkpoint_pct=50.0,
        checkpoint_time_ms=1.25,
        comparisons=4200,
        moves=1700,
    )
    defaults.update(overrides)
    return defaults


# ---------------------------------------------------------------------------
# 1. Valid full payload — happy path
# ---------------------------------------------------------------------------

class TestValidPayload:
    def test_returns_dict(self):
        feat = build_runtime_features(**_valid_kwargs())
        assert isinstance(feat, dict)

    def test_exact_required_keys(self):
        feat = build_runtime_features(**_valid_kwargs())
        assert set(feat.keys()) == set(REQUIRED_FEATURES)

    def test_exactly_11_keys(self):
        feat = build_runtime_features(**_valid_kwargs())
        assert len(feat) == 11

    def test_values_are_not_nan(self):
        feat = build_runtime_features(**_valid_kwargs())
        for key, val in feat.items():
            if isinstance(val, float):
                assert not math.isnan(val), f"{key} is NaN"

    def test_values_are_not_inf(self):
        feat = build_runtime_features(**_valid_kwargs())
        for key, val in feat.items():
            if isinstance(val, float):
                assert not math.isinf(val), f"{key} is Inf"

    def test_passthrough_fields_match_inputs(self):
        kw = _valid_kwargs()
        feat = build_runtime_features(**kw)
        assert feat["algorithm"] == kw["current_algorithm"]
        assert feat["input_type"] == kw["input_type"]
        assert feat["size"] == kw["array_size"]
        assert feat["checkpoint_pct"] == kw["checkpoint_pct"]
        assert feat["checkpoint_time_ms"] == kw["checkpoint_time_ms"]
        assert feat["checkpoint_comparisons"] == kw["comparisons"]
        assert feat["checkpoint_data_movements"] == kw["moves"]

    def test_no_forbidden_leakage_fields(self):
        feat = build_runtime_features(**_valid_kwargs())
        for field in FORBIDDEN_LEAKAGE_FIELDS:
            assert field not in feat, f"Leakage field '{field}' found in output"


# ---------------------------------------------------------------------------
# 2. Derived feature formulas
# ---------------------------------------------------------------------------

class TestDerivedFeatures:
    def test_comparisons_per_element(self):
        feat = build_runtime_features(**_valid_kwargs(comparisons=4200, array_size=1000))
        assert pytest.approx(feat["comparisons_per_element"]) == 4200 / 1000

    def test_movements_per_element(self):
        feat = build_runtime_features(**_valid_kwargs(moves=1700, array_size=1000))
        assert pytest.approx(feat["movements_per_element"]) == 1700 / 1000

    def test_work_ratio(self):
        feat = build_runtime_features(**_valid_kwargs(comparisons=4200, moves=1700))
        # Formula: comparisons / (moves + 1)
        assert pytest.approx(feat["work_ratio"]) == 4200 / (1700 + 1)

    def test_time_per_element_ms(self):
        feat = build_runtime_features(**_valid_kwargs(checkpoint_time_ms=1.25, array_size=1000))
        assert pytest.approx(feat["time_per_element_ms"]) == 1.25 / 1000

    def test_work_ratio_zero_moves(self):
        """moves=0 must not cause ZeroDivisionError; denominator is moves+1=1."""
        feat = build_runtime_features(**_valid_kwargs(moves=0, comparisons=500))
        assert pytest.approx(feat["work_ratio"]) == 500 / 1.0

    def test_zero_checkpoint_time(self):
        """checkpoint_time_ms=0 is valid; derived time fields become 0."""
        feat = build_runtime_features(**_valid_kwargs(checkpoint_time_ms=0.0))
        assert feat["checkpoint_time_ms"] == 0.0
        assert feat["time_per_element_ms"] == 0.0

    def test_zero_comparisons_zero_moves(self):
        """Both zero is valid (edge case: algorithm reached checkpoint in 0 ops)."""
        feat = build_runtime_features(**_valid_kwargs(comparisons=0, moves=0))
        assert feat["comparisons_per_element"] == 0.0
        assert feat["movements_per_element"] == 0.0
        assert feat["work_ratio"] == 0.0  # 0 / (0+1)


# ---------------------------------------------------------------------------
# 3. Determinism
# ---------------------------------------------------------------------------

class TestDeterminism:
    def test_same_inputs_same_output(self):
        kw = _valid_kwargs()
        assert build_runtime_features(**kw) == build_runtime_features(**kw)

    def test_dict_order_independence(self):
        """build_runtime_features uses keyword arguments — order is irrelevant."""
        kw = _valid_kwargs()
        # Call with reversed kwarg ordering (Python guarantees dict key order
        # doesn't affect named parameter passing)
        feat1 = build_runtime_features(**kw)
        feat2 = build_runtime_features(
            moves=kw["moves"],
            comparisons=kw["comparisons"],
            checkpoint_time_ms=kw["checkpoint_time_ms"],
            checkpoint_pct=kw["checkpoint_pct"],
            array_size=kw["array_size"],
            input_type=kw["input_type"],
            current_algorithm=kw["current_algorithm"],
        )
        assert feat1 == feat2


# ---------------------------------------------------------------------------
# 4. Each supported algorithm
# ---------------------------------------------------------------------------

class TestSupportedAlgorithms:
    @pytest.mark.parametrize("algo", sorted(SUPPORTED_ALGORITHMS))
    def test_each_algorithm_accepted(self, algo):
        feat = build_runtime_features(**_valid_kwargs(current_algorithm=algo))
        assert feat["algorithm"] == algo

    def test_all_algorithms_covered(self):
        assert SUPPORTED_ALGORITHMS == {"insertion_sort", "merge_sort", "quick_sort"}


# ---------------------------------------------------------------------------
# 5. Each supported input type
# ---------------------------------------------------------------------------

class TestSupportedInputTypes:
    @pytest.mark.parametrize("itype", sorted(SUPPORTED_INPUT_TYPES))
    def test_each_input_type_accepted(self, itype):
        feat = build_runtime_features(**_valid_kwargs(input_type=itype))
        assert feat["input_type"] == itype

    def test_all_input_types_covered(self):
        expected = {"all_equal", "duplicate_heavy", "nearly_sorted",
                    "random", "reverse_sorted", "sorted"}
        assert SUPPORTED_INPUT_TYPES == expected


# ---------------------------------------------------------------------------
# 6. Invalid array_size
# ---------------------------------------------------------------------------

class TestInvalidArraySize:
    def test_zero_size_raises(self):
        with pytest.raises(ValueError, match="array_size"):
            build_runtime_features(**_valid_kwargs(array_size=0))

    def test_negative_size_raises(self):
        with pytest.raises(ValueError, match="array_size"):
            build_runtime_features(**_valid_kwargs(array_size=-1))

    def test_size_one_accepted(self):
        feat = build_runtime_features(**_valid_kwargs(array_size=1, comparisons=0, moves=0))
        assert feat["size"] == 1

    def test_non_int_size_raises_typeerror(self):
        with pytest.raises(TypeError, match="array_size"):
            build_runtime_features(**_valid_kwargs(array_size=100.0))  # float not allowed


# ---------------------------------------------------------------------------
# 7. Negative comparisons
# ---------------------------------------------------------------------------

class TestNegativeComparisons:
    def test_negative_comparisons_raises(self):
        with pytest.raises(ValueError, match="comparisons"):
            build_runtime_features(**_valid_kwargs(comparisons=-1))

    def test_zero_comparisons_accepted(self):
        feat = build_runtime_features(**_valid_kwargs(comparisons=0))
        assert feat["checkpoint_comparisons"] == 0

    def test_non_int_comparisons_raises(self):
        with pytest.raises(TypeError, match="comparisons"):
            build_runtime_features(**_valid_kwargs(comparisons=100.5))


# ---------------------------------------------------------------------------
# 8. Negative moves
# ---------------------------------------------------------------------------

class TestNegativeMoves:
    def test_negative_moves_raises(self):
        with pytest.raises(ValueError, match="moves"):
            build_runtime_features(**_valid_kwargs(moves=-1))

    def test_zero_moves_accepted(self):
        feat = build_runtime_features(**_valid_kwargs(moves=0))
        assert feat["checkpoint_data_movements"] == 0

    def test_non_int_moves_raises(self):
        with pytest.raises(TypeError, match="moves"):
            build_runtime_features(**_valid_kwargs(moves=50.5))


# ---------------------------------------------------------------------------
# 9. Missing required values (caught by Python itself as TypeError)
# ---------------------------------------------------------------------------

class TestMissingRequiredValues:
    def test_missing_current_algorithm_raises(self):
        kw = _valid_kwargs()
        del kw["current_algorithm"]
        with pytest.raises(TypeError):
            build_runtime_features(**kw)

    def test_missing_input_type_raises(self):
        kw = _valid_kwargs()
        del kw["input_type"]
        with pytest.raises(TypeError):
            build_runtime_features(**kw)

    def test_missing_array_size_raises(self):
        kw = _valid_kwargs()
        del kw["array_size"]
        with pytest.raises(TypeError):
            build_runtime_features(**kw)

    def test_missing_checkpoint_pct_raises(self):
        kw = _valid_kwargs()
        del kw["checkpoint_pct"]
        with pytest.raises(TypeError):
            build_runtime_features(**kw)

    def test_missing_checkpoint_time_ms_raises(self):
        kw = _valid_kwargs()
        del kw["checkpoint_time_ms"]
        with pytest.raises(TypeError):
            build_runtime_features(**kw)

    def test_missing_comparisons_raises(self):
        kw = _valid_kwargs()
        del kw["comparisons"]
        with pytest.raises(TypeError):
            build_runtime_features(**kw)

    def test_missing_moves_raises(self):
        kw = _valid_kwargs()
        del kw["moves"]
        with pytest.raises(TypeError):
            build_runtime_features(**kw)


# ---------------------------------------------------------------------------
# 10. Unknown categorical values
# ---------------------------------------------------------------------------

class TestUnknownCategoricals:
    def test_unknown_algorithm_raises(self):
        with pytest.raises(ValueError, match="current_algorithm"):
            build_runtime_features(**_valid_kwargs(current_algorithm="heapsort"))

    def test_empty_algorithm_raises(self):
        with pytest.raises(ValueError, match="current_algorithm"):
            build_runtime_features(**_valid_kwargs(current_algorithm=""))

    def test_unknown_input_type_raises(self):
        with pytest.raises(ValueError, match="input_type"):
            build_runtime_features(**_valid_kwargs(input_type="real_world"))

    def test_empty_input_type_raises(self):
        with pytest.raises(ValueError, match="input_type"):
            build_runtime_features(**_valid_kwargs(input_type=""))


# ---------------------------------------------------------------------------
# 11. NaN inputs
# ---------------------------------------------------------------------------

class TestNaNInputs:
    def test_nan_checkpoint_pct_raises(self):
        with pytest.raises(ValueError, match="NaN"):
            build_runtime_features(**_valid_kwargs(checkpoint_pct=float("nan")))

    def test_nan_checkpoint_time_raises(self):
        with pytest.raises(ValueError, match="NaN"):
            build_runtime_features(**_valid_kwargs(checkpoint_time_ms=float("nan")))


# ---------------------------------------------------------------------------
# 12. Positive and negative infinity
# ---------------------------------------------------------------------------

class TestInfiniteInputs:
    def test_pos_inf_checkpoint_pct_raises(self):
        with pytest.raises(ValueError, match="infinite"):
            build_runtime_features(**_valid_kwargs(checkpoint_pct=float("inf")))

    def test_neg_inf_checkpoint_pct_raises(self):
        with pytest.raises(ValueError, match="infinite"):
            build_runtime_features(**_valid_kwargs(checkpoint_pct=float("-inf")))

    def test_pos_inf_checkpoint_time_raises(self):
        with pytest.raises(ValueError, match="infinite"):
            build_runtime_features(**_valid_kwargs(checkpoint_time_ms=float("inf")))

    def test_neg_inf_checkpoint_time_raises(self):
        with pytest.raises(ValueError, match="infinite"):
            build_runtime_features(**_valid_kwargs(checkpoint_time_ms=float("-inf")))


# ---------------------------------------------------------------------------
# 13. Forbidden leakage fields
# ---------------------------------------------------------------------------

class TestForbiddenLeakageFields:
    def test_leakage_fields_not_in_output(self):
        feat = build_runtime_features(**_valid_kwargs())
        for field in FORBIDDEN_LEAKAGE_FIELDS:
            assert field not in feat

    def test_all_expected_leakage_fields_defined(self):
        """Spot-check key leakage column names are in the constant."""
        spot_check = [
            "best_action",
            "continue_time_ms",
            "switch_insertion_sort_time_ms",
            "switch_merge_sort_time_ms",
            "switch_quick_sort_time_ms",
            "best_action_total_ms",
            "speedup_vs_continue",
        ]
        for field in spot_check:
            assert field in FORBIDDEN_LEAKAGE_FIELDS, f"Missing: {field}"

    def test_checkpoint_pct_range_boundary_low(self):
        feat = build_runtime_features(**_valid_kwargs(checkpoint_pct=0.0))
        assert feat["checkpoint_pct"] == 0.0

    def test_checkpoint_pct_range_boundary_high(self):
        feat = build_runtime_features(**_valid_kwargs(checkpoint_pct=100.0))
        assert feat["checkpoint_pct"] == 100.0

    def test_checkpoint_pct_above_100_raises(self):
        with pytest.raises(ValueError, match="checkpoint_pct"):
            build_runtime_features(**_valid_kwargs(checkpoint_pct=100.001))

    def test_checkpoint_pct_below_0_raises(self):
        with pytest.raises(ValueError, match="checkpoint_pct"):
            build_runtime_features(**_valid_kwargs(checkpoint_pct=-0.001))

    def test_negative_checkpoint_time_raises(self):
        with pytest.raises(ValueError, match="checkpoint_time_ms"):
            build_runtime_features(**_valid_kwargs(checkpoint_time_ms=-0.001))


# ---------------------------------------------------------------------------
# 14. Valid prediction actions (validate_predicted_action)
# ---------------------------------------------------------------------------

class TestValidPredictionActions:
    @pytest.mark.parametrize("action", sorted(VALID_PREDICTION_ACTIONS))
    def test_valid_action_returns_unchanged(self, action):
        # Choose an algorithm that doesn't conflict with the switch action.
        algo_map = {
            "continue": "quick_sort",
            "switch_insertion_sort": "quick_sort",
            "switch_merge_sort": "quick_sort",
            "switch_quick_sort": "insertion_sort",
        }
        algo = algo_map[action]
        result = validate_predicted_action(action, algo)
        assert result == action

    def test_continue_with_any_algorithm(self):
        for algo in sorted(SUPPORTED_ALGORITHMS):
            result = validate_predicted_action("continue", algo)
            assert result == "continue"

    def test_all_expected_actions_defined(self):
        expected = {"continue", "switch_insertion_sort", "switch_merge_sort", "switch_quick_sort"}
        assert VALID_PREDICTION_ACTIONS == expected

    def test_fallback_action_is_continue(self):
        assert FALLBACK_ACTION == "continue"


# ---------------------------------------------------------------------------
# 15. Invalid prediction actions
# ---------------------------------------------------------------------------

class TestInvalidPredictionActions:
    def test_unknown_action_raises(self):
        with pytest.raises(ValueError, match="not a valid model output"):
            validate_predicted_action("switch_heapsort", "quick_sort")

    def test_empty_action_raises(self):
        with pytest.raises(ValueError, match="not a valid model output"):
            validate_predicted_action("", "quick_sort")

    def test_non_string_action_raises(self):
        with pytest.raises(ValueError):
            validate_predicted_action(42, "quick_sort")  # type: ignore[arg-type]

    def test_none_action_raises(self):
        with pytest.raises(ValueError):
            validate_predicted_action(None, "quick_sort")  # type: ignore[arg-type]

    def test_unknown_current_algorithm_raises(self):
        with pytest.raises(ValueError, match="current_algorithm"):
            validate_predicted_action("continue", "heapsort")


# ---------------------------------------------------------------------------
# 16. Switch-to-self detection
# ---------------------------------------------------------------------------

class TestSwitchToSelf:
    def test_switch_to_insertion_from_insertion_raises(self):
        with pytest.raises(ValueError, match="degenerate"):
            validate_predicted_action("switch_insertion_sort", "insertion_sort")

    def test_switch_to_merge_from_merge_raises(self):
        with pytest.raises(ValueError, match="degenerate"):
            validate_predicted_action("switch_merge_sort", "merge_sort")

    def test_switch_to_quick_from_quick_raises(self):
        with pytest.raises(ValueError, match="degenerate"):
            validate_predicted_action("switch_quick_sort", "quick_sort")

    def test_switch_to_different_algo_is_fine(self):
        result = validate_predicted_action("switch_insertion_sort", "quick_sort")
        assert result == "switch_insertion_sort"


# ---------------------------------------------------------------------------
# 17. Switching to each supported algorithm
# ---------------------------------------------------------------------------

class TestSwitchingActions:
    def test_switch_insertion_sort_from_quick(self):
        assert validate_predicted_action("switch_insertion_sort", "quick_sort") == "switch_insertion_sort"

    def test_switch_insertion_sort_from_merge(self):
        assert validate_predicted_action("switch_insertion_sort", "merge_sort") == "switch_insertion_sort"

    def test_switch_merge_sort_from_quick(self):
        assert validate_predicted_action("switch_merge_sort", "quick_sort") == "switch_merge_sort"

    def test_switch_merge_sort_from_insertion(self):
        assert validate_predicted_action("switch_merge_sort", "insertion_sort") == "switch_merge_sort"

    def test_switch_quick_sort_from_insertion(self):
        assert validate_predicted_action("switch_quick_sort", "insertion_sort") == "switch_quick_sort"

    def test_switch_quick_sort_from_merge(self):
        assert validate_predicted_action("switch_quick_sort", "merge_sort") == "switch_quick_sort"


# ---------------------------------------------------------------------------
# 18. Cross-validation against real training CSV row
# ---------------------------------------------------------------------------

class TestCrossValidationAgainstTrainingData:
    """
    Verify that build_runtime_features reproduces derived features correctly
    by comparing against a real row from checkpoint_training.csv.

    This test directly proves that runtime feature engineering is consistent
    with training-time feature engineering in extract_dataset.py.
    """

    CSV_PATH = (
        Path(__file__).resolve().parents[2]
        / "ml" / "data" / "processed" / "checkpoint_training.csv"
    )

    @pytest.fixture(scope="class")
    @classmethod
    def csv_row(cls):
        """Load the first row from the training CSV."""
        if not cls.CSV_PATH.exists():
            pytest.skip(f"Training CSV not found at {cls.CSV_PATH}")
        df = pd.read_csv(cls.CSV_PATH)
        return df.iloc[0]

    def test_comparisons_per_element_matches_csv(self, csv_row):
        feat = build_runtime_features(
            current_algorithm=csv_row["algorithm"],
            input_type=csv_row["input_type"],
            array_size=int(csv_row["size"]),
            checkpoint_pct=float(csv_row["checkpoint_pct"]),
            checkpoint_time_ms=float(csv_row["checkpoint_time_ms"]),
            comparisons=int(csv_row["checkpoint_comparisons"]),
            moves=int(csv_row["checkpoint_data_movements"]),
        )
        assert pytest.approx(feat["comparisons_per_element"], rel=1e-6) == float(
            csv_row["comparisons_per_element"]
        )

    def test_movements_per_element_matches_csv(self, csv_row):
        feat = build_runtime_features(
            current_algorithm=csv_row["algorithm"],
            input_type=csv_row["input_type"],
            array_size=int(csv_row["size"]),
            checkpoint_pct=float(csv_row["checkpoint_pct"]),
            checkpoint_time_ms=float(csv_row["checkpoint_time_ms"]),
            comparisons=int(csv_row["checkpoint_comparisons"]),
            moves=int(csv_row["checkpoint_data_movements"]),
        )
        assert pytest.approx(feat["movements_per_element"], rel=1e-6) == float(
            csv_row["movements_per_element"]
        )

    def test_work_ratio_matches_csv(self, csv_row):
        feat = build_runtime_features(
            current_algorithm=csv_row["algorithm"],
            input_type=csv_row["input_type"],
            array_size=int(csv_row["size"]),
            checkpoint_pct=float(csv_row["checkpoint_pct"]),
            checkpoint_time_ms=float(csv_row["checkpoint_time_ms"]),
            comparisons=int(csv_row["checkpoint_comparisons"]),
            moves=int(csv_row["checkpoint_data_movements"]),
        )
        assert pytest.approx(feat["work_ratio"], rel=1e-6) == float(csv_row["work_ratio"])

    def test_time_per_element_ms_matches_csv(self, csv_row):
        feat = build_runtime_features(
            current_algorithm=csv_row["algorithm"],
            input_type=csv_row["input_type"],
            array_size=int(csv_row["size"]),
            checkpoint_pct=float(csv_row["checkpoint_pct"]),
            checkpoint_time_ms=float(csv_row["checkpoint_time_ms"]),
            comparisons=int(csv_row["checkpoint_comparisons"]),
            moves=int(csv_row["checkpoint_data_movements"]),
        )
        assert pytest.approx(feat["time_per_element_ms"], rel=1e-6) == float(
            csv_row["time_per_element_ms"]
        )

    def test_all_required_features_present_for_csv_row(self, csv_row):
        feat = build_runtime_features(
            current_algorithm=csv_row["algorithm"],
            input_type=csv_row["input_type"],
            array_size=int(csv_row["size"]),
            checkpoint_pct=float(csv_row["checkpoint_pct"]),
            checkpoint_time_ms=float(csv_row["checkpoint_time_ms"]),
            comparisons=int(csv_row["checkpoint_comparisons"]),
            moves=int(csv_row["checkpoint_data_movements"]),
        )
        assert set(feat.keys()) == set(REQUIRED_FEATURES)

    def test_csv_rows_for_all_algorithms(self):
        """Each algorithm appears in the training CSV and round-trips cleanly."""
        if not self.CSV_PATH.exists():
            pytest.skip(f"Training CSV not found at {self.CSV_PATH}")
        df = pd.read_csv(self.CSV_PATH)
        for algo in sorted(SUPPORTED_ALGORITHMS):
            rows = df[df["algorithm"] == algo]
            assert not rows.empty, f"No rows for algorithm '{algo}' in CSV"
            row = rows.iloc[0]
            feat = build_runtime_features(
                current_algorithm=row["algorithm"],
                input_type=row["input_type"],
                array_size=int(row["size"]),
                checkpoint_pct=float(row["checkpoint_pct"]),
                checkpoint_time_ms=float(row["checkpoint_time_ms"]),
                comparisons=int(row["checkpoint_comparisons"]),
                moves=int(row["checkpoint_data_movements"]),
            )
            assert feat["algorithm"] == algo

    def test_csv_rows_for_all_input_types(self):
        """Each input_type appears in the training CSV and round-trips cleanly."""
        if not self.CSV_PATH.exists():
            pytest.skip(f"Training CSV not found at {self.CSV_PATH}")
        df = pd.read_csv(self.CSV_PATH)
        for itype in sorted(SUPPORTED_INPUT_TYPES):
            rows = df[df["input_type"] == itype]
            assert not rows.empty, f"No rows for input_type '{itype}' in CSV"
            row = rows.iloc[0]
            feat = build_runtime_features(
                current_algorithm=row["algorithm"],
                input_type=row["input_type"],
                array_size=int(row["size"]),
                checkpoint_pct=float(row["checkpoint_pct"]),
                checkpoint_time_ms=float(row["checkpoint_time_ms"]),
                comparisons=int(row["checkpoint_comparisons"]),
                moves=int(row["checkpoint_data_movements"]),
            )
            assert feat["input_type"] == itype


# ---------------------------------------------------------------------------
# 19. Integration: features flow into predict_action without error
# ---------------------------------------------------------------------------

class TestIntegrationWithModel:
    """
    Light integration test: verify that build_runtime_features output is accepted
    by the existing predict.validate_checkpoint_input without modification.
    This confirms end-to-end compatibility.
    """

    def test_features_accepted_by_validate_checkpoint_input(self):
        from ml.predict import validate_checkpoint_input

        feat = build_runtime_features(**_valid_kwargs())
        df = validate_checkpoint_input(feat)
        assert isinstance(df, pd.DataFrame)
        assert list(df.columns) == list(REQUIRED_FEATURES)
        assert len(df) == 1

    def test_features_produce_valid_prediction(self):
        """With real model loaded, a valid runtime feature dict produces a known label."""
        import joblib

        model_path = (
            Path(__file__).resolve().parents[2]
            / "ml" / "models" / "adaptive_sort_model.joblib"
        )
        if not model_path.exists():
            pytest.skip("Production model not found")

        from ml.predict import predict_action

        model = joblib.load(model_path)
        feat = build_runtime_features(**_valid_kwargs())
        action = predict_action(model, feat)
        assert action in VALID_PREDICTION_ACTIONS
