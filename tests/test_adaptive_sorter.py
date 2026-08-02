"""
test_adaptive_sorter.py
-----------------------
Phase 7.3 — Tests for src/adaptive/adaptive_sorter.py

Covers:
  - Full start/action matrix (3 algos × 4 actions = 12 combos)
  - Multiple input distributions
  - Result structure (immutable, fields, ms properties, is_sorted)
  - Input safety (bad algo, bad input_type, no predictor, no predict())
  - Original input unchanged (copy semantics)
  - Fallback cases: PredictionError, switch-to-self, invalid action, malformed result
  - Edge cases: empty, 1-element, 2-element, already sorted, all-equal
  - Integration with real RuntimePredictor
  - Predictor called exactly once per non-trivial sort
"""

from __future__ import annotations

import random
import sys
from dataclasses import FrozenInstanceError
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.adaptive.adaptive_sorter import (
    FALLBACK_REASON_INVALID_ACTION,
    FALLBACK_REASON_MALFORMED_RESULT,
    FALLBACK_REASON_PREDICTION_FAILED,
    FALLBACK_REASON_SWITCH_TO_SELF,
    SUPPORTED_ALGORITHMS,
    SUPPORTED_INPUT_TYPES,
    VALID_ACTIONS,
    AdaptiveSortResult,
    adaptive_sort,
)
from ml.src.runtime_predictor import (
    PredictionError,
    RuntimePrediction,
    RuntimePredictorError,
)

# ---------------------------------------------------------------------------
# Fake predictor helpers
# ---------------------------------------------------------------------------

class FixedPredictor:
    """Fake predictor that always returns one fixed action."""

    def __init__(self, action: str):
        self.action = action
        self.calls = 0

    def predict(self, **kwargs) -> RuntimePrediction:
        self.calls += 1
        return RuntimePrediction(
            action=self.action,
            features={},
            feature_build_ns=10,
            inference_ns=20,
        )


class FailingPredictor:
    """Fake predictor that always raises a PredictionError."""

    def __init__(self, exc_class=PredictionError):
        self.exc_class = exc_class
        self.calls = 0

    def predict(self, **kwargs):
        self.calls += 1
        raise self.exc_class("Test prediction failure")


class MalformedResultPredictor:
    """Returns a result whose 'action' is not a string."""

    def predict(self, **kwargs):
        # Return an object with action=None (malformed)
        return RuntimePrediction(
            action="invalid_xyz",    # not in VALID_ACTIONS
            features={},
            feature_build_ns=0,
            inference_ns=0,
        )


class BadAttributePredictor:
    """Returns an object without an 'action' attribute."""

    class _FakeResult:
        pass

    def predict(self, **kwargs):
        return self._FakeResult()   # no 'action' str attribute


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

SMALL_RANDOM = [random.randint(0, 100) for _ in range(30)]


def _make_array(kind: str, n: int = 30) -> list:
    if kind == "random":
        return [random.randint(0, 100) for _ in range(n)]
    elif kind == "sorted":
        return list(range(n))
    elif kind == "reverse_sorted":
        return list(range(n, 0, -1))
    elif kind == "nearly_sorted":
        arr = list(range(n))
        arr[0], arr[-1] = arr[-1], arr[0]
        return arr
    elif kind == "duplicate_heavy":
        return [random.randint(1, 5) for _ in range(n)]
    elif kind == "all_equal":
        return [7] * n
    raise ValueError(kind)


# ===========================================================================
# 1. Full start/action matrix
# ===========================================================================

class TestStartActionMatrix:
    """3 starting algorithms × 4 actions = 12 combinations."""

    @pytest.mark.parametrize("starting_algo", sorted(SUPPORTED_ALGORITHMS))
    @pytest.mark.parametrize("action", sorted(VALID_ACTIONS))
    def test_sort_completes_and_is_correct(self, starting_algo, action):
        # Skip switch-to-self combinations — tested separately.
        from src.adaptive.adaptive_sorter import _SWITCH_TO_ALGO
        target = _SWITCH_TO_ALGO.get(action)
        if target == starting_algo:
            pytest.skip(f"switch-to-self: {starting_algo} → {action} tested separately")

        arr = _make_array("random")
        expected = sorted(arr)
        predictor = FixedPredictor(action)

        result = adaptive_sort(
            arr,
            starting_algorithm=starting_algo,
            input_type="random",
            predictor=predictor,
        )

        assert result.sorted_values == tuple(expected)
        assert result.is_sorted

    @pytest.mark.parametrize("starting_algo", sorted(SUPPORTED_ALGORITHMS))
    @pytest.mark.parametrize("action", sorted(VALID_ACTIONS))
    def test_predictor_called_exactly_once(self, starting_algo, action):
        from src.adaptive.adaptive_sorter import _SWITCH_TO_ALGO
        if _SWITCH_TO_ALGO.get(action) == starting_algo:
            pytest.skip("switch-to-self")

        predictor = FixedPredictor(action)
        adaptive_sort(
            _make_array("random"),
            starting_algorithm=starting_algo,
            input_type="random",
            predictor=predictor,
        )
        assert predictor.calls == 1

    @pytest.mark.parametrize("starting_algo", sorted(SUPPORTED_ALGORITHMS))
    @pytest.mark.parametrize("action", sorted(VALID_ACTIONS))
    def test_action_fields_correct(self, starting_algo, action):
        from src.adaptive.adaptive_sorter import _SWITCH_TO_ALGO
        if _SWITCH_TO_ALGO.get(action) == starting_algo:
            pytest.skip("switch-to-self")

        predictor = FixedPredictor(action)
        result = adaptive_sort(
            _make_array("random"),
            starting_algorithm=starting_algo,
            input_type="random",
            predictor=predictor,
        )

        assert result.requested_action == action
        assert result.executed_action == action
        assert result.prediction_succeeded is True
        assert result.fallback_used is False
        assert result.fallback_reason is None
        assert result.starting_algorithm == starting_algo

        if action == "continue":
            assert result.final_algorithm == starting_algo
        else:
            expected_final = _SWITCH_TO_ALGO[action]
            assert result.final_algorithm == expected_final

    @pytest.mark.parametrize("starting_algo", sorted(SUPPORTED_ALGORITHMS))
    def test_same_values_in_output(self, starting_algo):
        arr = [3, 1, 4, 1, 5, 9, 2, 6, 5]
        predictor = FixedPredictor("continue")
        result = adaptive_sort(
            arr,
            starting_algorithm=starting_algo,
            input_type="duplicate_heavy",
            predictor=predictor,
        )
        assert sorted(result.sorted_values) == sorted(arr)


# ===========================================================================
# 2. Input distribution coverage
# ===========================================================================

class TestInputDistributions:
    @pytest.mark.parametrize("itype", sorted(SUPPORTED_INPUT_TYPES))
    def test_all_input_types_produce_sorted_output(self, itype):
        arr = _make_array(itype, n=40)
        predictor = FixedPredictor("continue")
        result = adaptive_sort(
            arr,
            starting_algorithm="merge_sort",
            input_type=itype,
            predictor=predictor,
        )
        assert result.sorted_values == tuple(sorted(arr))
        assert result.is_sorted


# ===========================================================================
# 3. Result structure tests
# ===========================================================================

class TestResultStructure:

    @pytest.fixture(scope="class")
    @classmethod
    def sample_result(cls):
        predictor = FixedPredictor("continue")
        return adaptive_sort(
            _make_array("random"),
            starting_algorithm="insertion_sort",
            input_type="random",
            predictor=predictor,
        )

    def test_result_is_frozen(self, sample_result):
        with pytest.raises((FrozenInstanceError, TypeError, AttributeError)):
            sample_result.is_sorted = False  # type: ignore[misc]

    def test_sorted_values_is_tuple(self, sample_result):
        assert isinstance(sample_result.sorted_values, tuple)

    def test_all_timing_nonnegative(self, sample_result):
        assert sample_result.checkpoint_time_ns >= 0
        assert sample_result.feature_build_ns >= 0
        assert sample_result.inference_ns >= 0
        assert sample_result.execution_after_decision_ns >= 0
        assert sample_result.switch_overhead_ns >= 0
        assert sample_result.total_runtime_ns >= 0

    def test_ms_properties_match_ns(self, sample_result):
        assert pytest.approx(sample_result.checkpoint_time_ms) == sample_result.checkpoint_time_ns / 1e6
        assert pytest.approx(sample_result.feature_build_ms) == sample_result.feature_build_ns / 1e6
        assert pytest.approx(sample_result.inference_ms) == sample_result.inference_ns / 1e6
        assert pytest.approx(sample_result.execution_after_decision_ms) == sample_result.execution_after_decision_ns / 1e6
        assert pytest.approx(sample_result.switch_overhead_ms) == sample_result.switch_overhead_ns / 1e6
        assert pytest.approx(sample_result.total_runtime_ms) == sample_result.total_runtime_ns / 1e6

    def test_checkpoint_dict_present(self, sample_result):
        assert isinstance(sample_result.checkpoint, dict)
        # Must have key fields (arr excluded, arr_length added)
        assert "algo" in sample_result.checkpoint
        assert "comparisons" in sample_result.checkpoint
        assert "arr_length" in sample_result.checkpoint
        assert "arr" not in sample_result.checkpoint   # large array excluded

    def test_metrics_nonnegative(self, sample_result):
        assert sample_result.comparisons >= 0
        assert sample_result.data_movements >= 0

    def test_is_sorted_true(self, sample_result):
        assert sample_result.is_sorted is True

    def test_feature_build_ns_from_predictor(self):
        predictor = FixedPredictor("continue")
        result = adaptive_sort(
            _make_array("random"),
            starting_algorithm="insertion_sort",
            input_type="random",
            predictor=predictor,
        )
        assert result.feature_build_ns == 10   # FixedPredictor always returns 10
        assert result.inference_ns == 20


# ===========================================================================
# 4. Input safety tests
# ===========================================================================

class TestInputSafety:

    def test_unsupported_algorithm_rejected(self):
        predictor = FixedPredictor("continue")
        with pytest.raises(ValueError, match="starting_algorithm"):
            adaptive_sort([1, 2, 3], starting_algorithm="heapsort",
                          input_type="random", predictor=predictor)

    def test_unsupported_input_type_rejected(self):
        predictor = FixedPredictor("continue")
        with pytest.raises(ValueError, match="input_type"):
            adaptive_sort([1, 2, 3], starting_algorithm="merge_sort",
                          input_type="real_world", predictor=predictor)

    def test_none_predictor_rejected(self):
        with pytest.raises((ValueError, TypeError)):
            adaptive_sort([1, 2, 3], starting_algorithm="merge_sort",
                          input_type="random", predictor=None)

    def test_predictor_without_predict_rejected(self):
        class NoPredictMethod:
            pass
        with pytest.raises(TypeError, match="predict"):
            adaptive_sort([1, 2, 3], starting_algorithm="merge_sort",
                          input_type="random", predictor=NoPredictMethod())

    def test_original_input_unchanged_list(self):
        original = [5, 3, 8, 1, 9, 2]
        original_copy = original.copy()
        predictor = FixedPredictor("continue")
        adaptive_sort(original, starting_algorithm="insertion_sort",
                      input_type="random", predictor=predictor)
        assert original == original_copy, "Original list was mutated!"

    def test_original_input_unchanged_tuple(self):
        original = (5, 3, 8, 1, 9, 2)
        predictor = FixedPredictor("continue")
        adaptive_sort(original, starting_algorithm="merge_sort",
                      input_type="random", predictor=predictor)
        # Tuples are immutable, so mutation would raise TypeError anyway;
        # just verify the sort completes.
        assert sorted(original) == list(sorted(original))

    def test_duplicates_preserved(self):
        arr = [3, 1, 4, 1, 5, 9, 2, 6, 5, 3, 5]
        predictor = FixedPredictor("continue")
        result = adaptive_sort(arr, starting_algorithm="quick_sort",
                               input_type="duplicate_heavy", predictor=predictor)
        assert list(result.sorted_values) == sorted(arr)

    def test_negative_numbers_sorted_correctly(self):
        arr = [-5, 3, -1, 0, 7, -9, 2]
        predictor = FixedPredictor("continue")
        result = adaptive_sort(arr, starting_algorithm="insertion_sort",
                               input_type="random", predictor=predictor)
        assert list(result.sorted_values) == sorted(arr)


# ===========================================================================
# 5. Fallback tests
# ===========================================================================

class TestFallbacks:

    def test_prediction_error_triggers_fallback(self):
        predictor = FailingPredictor(PredictionError)
        arr = _make_array("random")
        result = adaptive_sort(arr, starting_algorithm="merge_sort",
                               input_type="random", predictor=predictor)

        assert result.prediction_succeeded is False
        assert result.fallback_used is True
        assert result.fallback_reason == FALLBACK_REASON_PREDICTION_FAILED
        assert result.executed_action == "continue"
        assert result.requested_action is None
        assert result.is_sorted
        assert list(result.sorted_values) == sorted(arr)

    def test_runtime_predictor_error_triggers_fallback(self):
        predictor = FailingPredictor(RuntimePredictorError)
        arr = _make_array("random")
        result = adaptive_sort(arr, starting_algorithm="insertion_sort",
                               input_type="random", predictor=predictor)
        assert result.fallback_used is True
        assert result.is_sorted

    def test_fallback_still_sorts_correctly(self):
        predictor = FailingPredictor(PredictionError)
        arr = [10, 5, 3, 8, 1]
        result = adaptive_sort(arr, starting_algorithm="quick_sort",
                               input_type="random", predictor=predictor)
        assert list(result.sorted_values) == sorted(arr)

    def test_invalid_action_triggers_fallback(self):
        predictor = MalformedResultPredictor()   # returns "invalid_xyz"
        arr = _make_array("random")
        result = adaptive_sort(arr, starting_algorithm="merge_sort",
                               input_type="random", predictor=predictor)

        assert result.fallback_used is True
        assert result.fallback_reason == FALLBACK_REASON_INVALID_ACTION
        assert result.executed_action == "continue"
        assert result.requested_action == "invalid_xyz"
        assert result.is_sorted

    def test_malformed_result_no_action_attribute(self):
        predictor = BadAttributePredictor()
        arr = _make_array("random")
        result = adaptive_sort(arr, starting_algorithm="insertion_sort",
                               input_type="random", predictor=predictor)
        assert result.fallback_used is True
        assert result.fallback_reason == FALLBACK_REASON_MALFORMED_RESULT
        assert result.is_sorted

    @pytest.mark.parametrize("starting_algo,switch_action", [
        ("insertion_sort", "switch_insertion_sort"),
        ("merge_sort",     "switch_merge_sort"),
        ("quick_sort",     "switch_quick_sort"),
    ])
    def test_switch_to_self_becomes_continue(self, starting_algo, switch_action):
        predictor = FixedPredictor(switch_action)
        arr = _make_array("random")
        result = adaptive_sort(arr, starting_algorithm=starting_algo,
                               input_type="random", predictor=predictor)

        assert result.requested_action == switch_action
        assert result.executed_action == "continue"
        assert result.final_algorithm == starting_algo
        assert result.fallback_used is True
        assert result.fallback_reason == FALLBACK_REASON_SWITCH_TO_SELF
        assert result.prediction_succeeded is True   # model DID succeed
        assert result.is_sorted

    def test_unexpected_exception_propagates(self):
        """KeyboardInterrupt must NOT be caught by the fallback handler."""
        class BombPredictor:
            def predict(self, **kwargs):
                raise KeyboardInterrupt("test interrupt")

        with pytest.raises(KeyboardInterrupt):
            adaptive_sort(
                _make_array("random"),
                starting_algorithm="merge_sort",
                input_type="random",
                predictor=BombPredictor(),
            )

    def test_fallback_does_not_swallow_system_exit(self):
        class ExitPredictor:
            def predict(self, **kwargs):
                raise SystemExit("test exit")

        with pytest.raises(SystemExit):
            adaptive_sort(
                _make_array("random"),
                starting_algorithm="insertion_sort",
                input_type="random",
                predictor=ExitPredictor(),
            )


# ===========================================================================
# 6. Edge cases
# ===========================================================================

class TestEdgeCases:

    @pytest.mark.parametrize("starting_algo", sorted(SUPPORTED_ALGORITHMS))
    def test_empty_input(self, starting_algo):
        predictor = FixedPredictor("continue")
        result = adaptive_sort([], starting_algorithm=starting_algo,
                               input_type="random", predictor=predictor)
        assert result.sorted_values == ()
        assert result.is_sorted
        # Trivial: predictor may or may not be called (n <= 1 path)
        assert result.executed_action == "continue"

    @pytest.mark.parametrize("starting_algo", sorted(SUPPORTED_ALGORITHMS))
    def test_one_element(self, starting_algo):
        predictor = FixedPredictor("continue")
        result = adaptive_sort([42], starting_algorithm=starting_algo,
                               input_type="random", predictor=predictor)
        assert result.sorted_values == (42,)
        assert result.is_sorted

    @pytest.mark.parametrize("starting_algo", sorted(SUPPORTED_ALGORITHMS))
    def test_two_elements_sorted(self, starting_algo):
        predictor = FixedPredictor("continue")
        result = adaptive_sort([2, 1], starting_algorithm=starting_algo,
                               input_type="random", predictor=predictor)
        assert result.sorted_values == (1, 2)
        assert result.is_sorted

    @pytest.mark.parametrize("starting_algo", sorted(SUPPORTED_ALGORITHMS))
    def test_already_sorted_input(self, starting_algo):
        arr = list(range(20))
        predictor = FixedPredictor("continue")
        result = adaptive_sort(arr, starting_algorithm=starting_algo,
                               input_type="sorted", predictor=predictor)
        assert list(result.sorted_values) == arr
        assert result.is_sorted

    @pytest.mark.parametrize("starting_algo", sorted(SUPPORTED_ALGORITHMS))
    def test_all_equal_input(self, starting_algo):
        arr = [5] * 20
        predictor = FixedPredictor("continue")
        result = adaptive_sort(arr, starting_algorithm=starting_algo,
                               input_type="all_equal", predictor=predictor)
        assert list(result.sorted_values) == arr
        assert result.is_sorted

    def test_predictor_called_exactly_once_for_n_gt_1(self):
        predictor = FixedPredictor("continue")
        adaptive_sort([3, 1, 2], starting_algorithm="merge_sort",
                      input_type="random", predictor=predictor)
        assert predictor.calls == 1

    def test_predictor_not_called_for_n_le_1(self):
        predictor = FixedPredictor("continue")
        adaptive_sort([], starting_algorithm="merge_sort",
                      input_type="random", predictor=predictor)
        assert predictor.calls == 0   # trivial path bypasses prediction


# ===========================================================================
# 7. Switch overhead tests
# ===========================================================================

class TestSwitchOverhead:

    @pytest.mark.parametrize("starting_algo,switch_action,target_algo", [
        ("insertion_sort", "switch_merge_sort",     "merge_sort"),
        ("insertion_sort", "switch_quick_sort",     "quick_sort"),
        ("merge_sort",     "switch_insertion_sort", "insertion_sort"),
        ("merge_sort",     "switch_quick_sort",     "quick_sort"),
        ("quick_sort",     "switch_insertion_sort", "insertion_sort"),
        ("quick_sort",     "switch_merge_sort",     "merge_sort"),
    ])
    def test_switch_overhead_nonnegative(self, starting_algo, switch_action, target_algo):
        predictor = FixedPredictor(switch_action)
        result = adaptive_sort(
            _make_array("random"),
            starting_algorithm=starting_algo,
            input_type="random",
            predictor=predictor,
        )
        assert result.switch_overhead_ns >= 0

    def test_continue_has_zero_switch_overhead(self):
        predictor = FixedPredictor("continue")
        result = adaptive_sort(
            _make_array("random"),
            starting_algorithm="quick_sort",
            input_type="random",
            predictor=predictor,
        )
        assert result.switch_overhead_ns == 0


# ===========================================================================
# 8. Reuse predictor across calls
# ===========================================================================

class TestPredictorReuse:

    def test_predictor_reused_across_multiple_sorts(self):
        predictor = FixedPredictor("continue")
        for _ in range(5):
            arr = _make_array("random")
            result = adaptive_sort(arr, starting_algorithm="merge_sort",
                                   input_type="random", predictor=predictor)
            assert result.is_sorted
        assert predictor.calls == 5   # one call per sort (n > 1)


# ===========================================================================
# 9. Integration with real RuntimePredictor
# ===========================================================================

class TestRealPredictorIntegration:

    _MODEL_PATH = PROJECT_ROOT / "ml" / "models" / "adaptive_sort_model.joblib"

    @pytest.fixture(scope="class")
    @classmethod
    def real_predictor(cls):
        if not cls._MODEL_PATH.exists():
            pytest.skip("Production model not found.")
        from ml.src.runtime_predictor import RuntimePredictor
        return RuntimePredictor()

    def test_end_to_end_with_real_predictor(self, real_predictor):
        arr = _make_array("random", n=50)
        result = adaptive_sort(
            arr,
            starting_algorithm="insertion_sort",
            input_type="random",
            predictor=real_predictor,
        )
        assert list(result.sorted_values) == sorted(arr)
        assert result.is_sorted

    def test_real_predictor_returns_valid_action(self, real_predictor):
        arr = _make_array("random", n=50)
        result = adaptive_sort(
            arr,
            starting_algorithm="insertion_sort",
            input_type="random",
            predictor=real_predictor,
        )
        if result.prediction_succeeded:
            assert result.requested_action in VALID_ACTIONS

    def test_real_predictor_timing_fields_populated(self, real_predictor):
        arr = _make_array("random", n=100)
        result = adaptive_sort(
            arr,
            starting_algorithm="merge_sort",
            input_type="random",
            predictor=real_predictor,
        )
        # When prediction succeeds, feature_build and inference > 0
        if result.prediction_succeeded:
            assert result.feature_build_ns > 0
            assert result.inference_ns > 0
        assert result.total_runtime_ns > 0

    def test_multiple_real_predictor_calls_no_reload(self, real_predictor):
        """Model must not reload between sort calls."""
        # RuntimePredictor.model_load_ns must not change between calls.
        ns_before = getattr(real_predictor, "model_load_ns", None)
        for itype in ("random", "sorted", "reverse_sorted"):
            adaptive_sort(
                _make_array(itype, n=30),
                starting_algorithm="insertion_sort",
                input_type=itype,
                predictor=real_predictor,
            )
        ns_after = getattr(real_predictor, "model_load_ns", None)
        if ns_before is not None:
            assert ns_before == ns_after

    def test_real_predictor_no_switch_to_self_error(self, real_predictor):
        """Even if the model predicts switch-to-self, sort completes safely."""
        # Try multiple algos/types; at least one should complete without unhandled error.
        for algo in sorted(SUPPORTED_ALGORITHMS):
            result = adaptive_sort(
                _make_array("random", n=50),
                starting_algorithm=algo,
                input_type="random",
                predictor=real_predictor,
            )
            assert result.is_sorted

    def test_checkpoint_framework_used_not_bypassed(self, real_predictor):
        """Verify checkpoint state is captured (framework is being used)."""
        arr = _make_array("random", n=40)
        result = adaptive_sort(
            arr,
            starting_algorithm="quick_sort",
            input_type="random",
            predictor=real_predictor,
        )
        assert result.checkpoint["algo"] == "quick_sort"
        assert result.checkpoint["arr_length"] == len(arr)
