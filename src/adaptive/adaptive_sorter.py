"""
adaptive_sorter.py
------------------
Phase 7.3 — Adaptive Sorter Controller

Provides the single public function ``adaptive_sort()`` and the accompanying
``AdaptiveSortResult`` dataclass.

The controller follows exactly one decision path per sort:

    1.  Copy the caller's input once.
    2.  Run the starting algorithm to its checkpoint.
    3.  Call ``RuntimePredictor.predict()`` exactly once.
    4.  Resolve the action (continue / switch / switch-to-self / fallback).
    5.  Execute the resolved action using the checkpoint framework.
    6.  Return a structured, immutable result.

Reusable functions already exist in ``src/checkpoint/runner.py``.
This module delegates to them directly — no sorting or switching logic
is reimplemented here.

Mutation policy
---------------
``adaptive_sort()`` copies the caller's input immediately (``list(values)``).
The original sequence is never mutated.  All checkpoint and switch operations
work on the internal copy.

Fallback policy
---------------
The controller falls back to ``continue`` — silently from the model's
perspective — when:
  * ``RuntimePredictor.predict()`` raises any ``RuntimePredictorError``
    subclass (includes ``PredictionError`` and ``ModelArtifactError``).
  * The model returns an unsupported action string.
  * The model predicts switching to the algorithm that is already running
    (switch-to-self).

Unexpected errors (``KeyboardInterrupt``, ``SystemExit``, programming bugs
unrelated to prediction) are **not** caught — they propagate normally.

Predictor lifetime
------------------
``RuntimePredictor`` must be created by the caller and reused:

    predictor = RuntimePredictor()
    for values in datasets:
        result = adaptive_sort(values, starting_algorithm="quick_sort",
                                input_type="random", predictor=predictor)

Do NOT create a new ``RuntimePredictor`` inside this function.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any, Sequence

from src.checkpoint.runner import (
    continue_sort,
    run_to_checkpoint,
    switch_sort,
)
from ml.src.runtime_predictor import RuntimePredictorError

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

#: Algorithms the controller can start with (must match checkpoint runner).
SUPPORTED_ALGORITHMS: frozenset[str] = frozenset({
    "insertion_sort",
    "merge_sort",
    "quick_sort",
})

#: Input distribution labels accepted by the ML model.
SUPPORTED_INPUT_TYPES: frozenset[str] = frozenset({
    "all_equal",
    "duplicate_heavy",
    "nearly_sorted",
    "random",
    "reverse_sorted",
    "sorted",
})

#: All action labels the model can return.
VALID_ACTIONS: frozenset[str] = frozenset({
    "continue",
    "switch_insertion_sort",
    "switch_merge_sort",
    "switch_quick_sort",
})

# Map switch labels → target algorithm name (used for action resolution).
_SWITCH_TO_ALGO: dict[str, str] = {
    "switch_insertion_sort": "insertion_sort",
    "switch_merge_sort":     "merge_sort",
    "switch_quick_sort":     "quick_sort",
}

# Fallback reason constants — stable machine-readable strings.
FALLBACK_REASON_PREDICTION_FAILED = "prediction_failed"
FALLBACK_REASON_SWITCH_TO_SELF    = "switch_to_self"
FALLBACK_REASON_INVALID_ACTION    = "invalid_action"
FALLBACK_REASON_MALFORMED_RESULT  = "malformed_result"


# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class AdaptiveSortResult:
    """Immutable result of one adaptive sort run.

    Timing fields
    -------------
    All ``_ns`` fields are measured with ``time.perf_counter_ns()``.
    ``total_runtime_ns`` is the wall-clock time for the *entire* adaptive sort
    call (checkpoint + prediction + post-decision execution).
    ``feature_build_ns`` and ``inference_ns`` come directly from the
    ``RuntimePrediction`` object; they are **not** included in any other
    timing field to avoid double-counting.
    ``checkpoint_time_ns`` is converted from ``state['time_ms']`` (the
    checkpoint framework measures in float ms; we store as int ns for
    consistency).
    ``execution_after_decision_ns`` is the post-checkpoint sorting time
    (``post_time_ms * 1_000_000``, rounded to int).
    ``switch_overhead_ns`` is ``result['overhead']['time_ms'] * 1_000_000``
    when a switch occurred; 0 otherwise.

    Metric aggregation
    ------------------
    ``comparisons`` = ``state['comparisons'] + post_comparisons``
        (= ``result['total_comparisons']`` from the checkpoint runner)
    ``data_movements`` = ``state['moves'] + post_moves``
        (= ``result['total_moves']`` from the checkpoint runner)
    Overhead comparisons/moves are included in post_comparisons/post_moves
    as returned by the checkpoint runner.

    Action fields
    -------------
    ``requested_action``  — the raw label returned by the ML model, or
                            ``None`` if prediction failed.
    ``executed_action``   — the action actually dispatched to the checkpoint
                            framework.
    ``final_algorithm``   — the algorithm that performed the post-checkpoint
                            work.
    ``prediction_succeeded`` — ``True`` iff the model returned a usable label.
    ``fallback_used``     — ``True`` iff ``executed_action != requested_action``
                            or if prediction failed.
    ``fallback_reason``   — machine-readable reason string or ``None``.
    """

    # Output
    sorted_values: tuple[int | float, ...]

    # Algorithm / action metadata
    starting_algorithm:   str
    requested_action:     str | None
    executed_action:      str
    final_algorithm:      str

    # Prediction metadata
    prediction_succeeded: bool
    fallback_used:        bool
    fallback_reason:      str | None

    # Checkpoint state snapshot
    checkpoint: dict[str, Any]

    # Timings (nanoseconds)
    checkpoint_time_ns:           int
    feature_build_ns:             int
    inference_ns:                 int
    execution_after_decision_ns:  int
    switch_overhead_ns:           int
    total_runtime_ns:             int

    # Aggregate metrics
    comparisons:    int
    data_movements: int

    # Correctness
    is_sorted: bool

    # ---- millisecond convenience properties ----------------------------

    @property
    def checkpoint_time_ms(self) -> float:
        """Checkpoint phase duration in milliseconds."""
        return self.checkpoint_time_ns / 1_000_000

    @property
    def feature_build_ms(self) -> float:
        """Feature-build phase duration in milliseconds."""
        return self.feature_build_ns / 1_000_000

    @property
    def inference_ms(self) -> float:
        """Model inference duration in milliseconds."""
        return self.inference_ns / 1_000_000

    @property
    def execution_after_decision_ms(self) -> float:
        """Post-decision sorting duration in milliseconds."""
        return self.execution_after_decision_ns / 1_000_000

    @property
    def switch_overhead_ms(self) -> float:
        """Switch overhead duration in milliseconds (0 for continue)."""
        return self.switch_overhead_ns / 1_000_000

    @property
    def total_runtime_ms(self) -> float:
        """Total adaptive sort wall-clock time in milliseconds."""
        return self.total_runtime_ns / 1_000_000


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def adaptive_sort(
    values: Sequence[int | float],
    *,
    starting_algorithm: str,
    input_type: str,
    predictor: Any,
    checkpoint_pct: float = 50.0,
) -> AdaptiveSortResult:
    """Run an adaptive sort on ``values``.

    The function:

    1. Validates arguments.
    2. Copies the caller's input (original is never mutated).
    3. Runs ``starting_algorithm`` to its checkpoint.
    4. Calls ``predictor.predict()`` exactly once.
    5. Resolves the predicted action (handling switch-to-self, invalid
       actions, and prediction failures via fallback to ``'continue'``).
    6. Executes the resolved action using ``src/checkpoint/runner.py``.
    7. Returns an immutable ``AdaptiveSortResult``.

    Parameters
    ----------
    values : sequence of int or float
        The data to sort.  Any sequence type is accepted; the caller's
        sequence is **not** mutated.
    starting_algorithm : str
        The algorithm to run first.  Must be one of:
        ``'insertion_sort'``, ``'merge_sort'``, ``'quick_sort'``.
    input_type : str
        The input distribution type — must be provided explicitly by the
        caller.  Must be one of: ``'all_equal'``, ``'duplicate_heavy'``,
        ``'nearly_sorted'``, ``'random'``, ``'reverse_sorted'``,
        ``'sorted'``.
    predictor : object with a ``predict(**kwargs)`` method
        Normally a ``RuntimePredictor`` instance created by the caller.
        Must be reused across calls — do **not** construct a new predictor
        inside this function.
    checkpoint_pct : float, optional
        Target checkpoint progress percentage (0–100).  Default 50.0.
        Note: the checkpoint framework uses this as a *guide*; the actual
        ``checkpoint_pct`` may differ slightly (especially for QuickSort).

    Returns
    -------
    AdaptiveSortResult
        Immutable result including sorted output, action metadata, timings,
        and aggregate metrics.

    Raises
    ------
    TypeError
        If ``predictor`` has no callable ``predict`` method, or if
        ``values`` is not a sequence.
    ValueError
        If ``starting_algorithm`` or ``input_type`` is not supported.
    """
    # ------------------------------------------------------------------ #
    # 1. Validate arguments                                                #
    # ------------------------------------------------------------------ #
    if not isinstance(values, (list, tuple, range)) and not hasattr(values, '__iter__'):
        raise TypeError(
            f"'values' must be a sequence, got {type(values).__name__}."
        )
    if not isinstance(starting_algorithm, str) or starting_algorithm not in SUPPORTED_ALGORITHMS:
        raise ValueError(
            f"'starting_algorithm' must be one of {sorted(SUPPORTED_ALGORITHMS)}, "
            f"got {starting_algorithm!r}."
        )
    if not isinstance(input_type, str) or input_type not in SUPPORTED_INPUT_TYPES:
        raise ValueError(
            f"'input_type' must be one of {sorted(SUPPORTED_INPUT_TYPES)}, "
            f"got {input_type!r}."
        )
    if predictor is None:
        raise ValueError("'predictor' must not be None.")
    if not callable(getattr(predictor, "predict", None)):
        raise TypeError(
            "'predictor' must have a callable 'predict()' method."
        )

    # ------------------------------------------------------------------ #
    # 2. Copy input — original is never mutated                            #
    # ------------------------------------------------------------------ #
    work_arr = list(values)

    # ------------------------------------------------------------------ #
    # 3. Start timing                                                       #
    # ------------------------------------------------------------------ #
    t_total_start = time.perf_counter_ns()

    # ------------------------------------------------------------------ #
    # 4. Edge case: trivial arrays bypass prediction                        #
    # ------------------------------------------------------------------ #
    n = len(work_arr)
    if n <= 1:
        t_total_end = time.perf_counter_ns()
        state = run_to_checkpoint(starting_algorithm, work_arr)
        runner_result = continue_sort(state)
        return AdaptiveSortResult(
            sorted_values=tuple(runner_result["sorted_arr"]),
            starting_algorithm=starting_algorithm,
            requested_action=None,
            executed_action="continue",
            final_algorithm=starting_algorithm,
            prediction_succeeded=False,
            fallback_used=True,
            fallback_reason="trivial_input",
            checkpoint=_safe_checkpoint_snapshot(state),
            checkpoint_time_ns=int(state["time_ms"] * 1_000_000),
            feature_build_ns=0,
            inference_ns=0,
            execution_after_decision_ns=int(runner_result["post_time_ms"] * 1_000_000),
            switch_overhead_ns=0,
            total_runtime_ns=time.perf_counter_ns() - t_total_start,
            comparisons=runner_result["total_comparisons"],
            data_movements=runner_result["total_moves"],
            is_sorted=_check_sorted(runner_result["sorted_arr"]),
        )

    # ------------------------------------------------------------------ #
    # 5. Run to checkpoint                                                  #
    # ------------------------------------------------------------------ #
    state = run_to_checkpoint(starting_algorithm, work_arr)
    checkpoint_time_ns = int(state["time_ms"] * 1_000_000)

    # ------------------------------------------------------------------ #
    # 6. Call predictor exactly once                                        #
    # ------------------------------------------------------------------ #
    requested_action: str | None = None
    prediction_succeeded = False
    fallback_used = False
    fallback_reason: str | None = None
    feature_build_ns = 0
    inference_ns = 0

    try:
        prediction = predictor.predict(
            current_algorithm=starting_algorithm,
            input_type=input_type,
            size=n,
            checkpoint_pct=float(state["checkpoint_pct"]),
            checkpoint_time_ms=float(state["time_ms"]),
            checkpoint_comparisons=int(state["comparisons"]),
            checkpoint_data_movements=int(state["moves"]),
        )
        # Extract timing from the RuntimePrediction result
        feature_build_ns = getattr(prediction, "feature_build_ns", 0)
        inference_ns     = getattr(prediction, "inference_ns", 0)

        raw_action = getattr(prediction, "action", None)
        if not isinstance(raw_action, str):
            # Malformed result object
            requested_action   = None
            fallback_used      = True
            fallback_reason    = FALLBACK_REASON_MALFORMED_RESULT
        elif raw_action not in VALID_ACTIONS:
            # Unsupported action label
            requested_action   = raw_action
            fallback_used      = True
            fallback_reason    = FALLBACK_REASON_INVALID_ACTION
        else:
            requested_action      = raw_action
            prediction_succeeded  = True

    except RuntimePredictorError:
        # Documented prediction failure — fall back silently, record it.
        fallback_used   = True
        fallback_reason = FALLBACK_REASON_PREDICTION_FAILED
        # requested_action stays None, prediction_succeeded stays False

    # Broad non-RuntimePredictorError exceptions bubble up (programming bugs,
    # KeyboardInterrupt, SystemExit, etc.).

    # ------------------------------------------------------------------ #
    # 7. Resolve action                                                     #
    # ------------------------------------------------------------------ #
    executed_action, final_algorithm = _resolve_action(
        requested_action=requested_action,
        starting_algorithm=starting_algorithm,
        prediction_succeeded=prediction_succeeded,
        # fallback_used may already be True from above
    )

    # Detect switch-to-self if prediction succeeded but action targets self
    if prediction_succeeded and not fallback_used:
        target_algo = _SWITCH_TO_ALGO.get(requested_action)  # type: ignore[arg-type]
        if target_algo is not None and target_algo == starting_algorithm:
            executed_action  = "continue"
            final_algorithm  = starting_algorithm
            fallback_used    = True
            fallback_reason  = FALLBACK_REASON_SWITCH_TO_SELF

    # ------------------------------------------------------------------ #
    # 8. Execute resolved action                                            #
    # ------------------------------------------------------------------ #
    t_exec_start = time.perf_counter_ns()

    if executed_action == "continue":
        runner_result = continue_sort(state)
        switch_overhead_ns = 0
    else:
        # switch_insertion_sort | switch_merge_sort | switch_quick_sort
        target_algo = _SWITCH_TO_ALGO[executed_action]
        runner_result = switch_sort(state, target_algo)
        switch_overhead_ns = int(runner_result["overhead"]["time_ms"] * 1_000_000)

    execution_after_decision_ns = int(
        (time.perf_counter_ns() - t_exec_start)
    )

    # ------------------------------------------------------------------ #
    # 9. Collect metrics                                                    #
    # ------------------------------------------------------------------ #
    total_runtime_ns = time.perf_counter_ns() - t_total_start
    sorted_arr = runner_result["sorted_arr"]

    return AdaptiveSortResult(
        sorted_values=tuple(sorted_arr),
        starting_algorithm=starting_algorithm,
        requested_action=requested_action,
        executed_action=executed_action,
        final_algorithm=final_algorithm,
        prediction_succeeded=prediction_succeeded,
        fallback_used=fallback_used,
        fallback_reason=fallback_reason,
        checkpoint=_safe_checkpoint_snapshot(state),
        checkpoint_time_ns=checkpoint_time_ns,
        feature_build_ns=feature_build_ns,
        inference_ns=inference_ns,
        execution_after_decision_ns=execution_after_decision_ns,
        switch_overhead_ns=switch_overhead_ns,
        total_runtime_ns=total_runtime_ns,
        comparisons=runner_result["total_comparisons"],
        data_movements=runner_result["total_moves"],
        is_sorted=_check_sorted(sorted_arr),
    )


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _resolve_action(
    *,
    requested_action: str | None,
    starting_algorithm: str,
    prediction_succeeded: bool,
) -> tuple[str, str]:
    """Return (executed_action, final_algorithm) given the request and starting state.

    Does NOT handle switch-to-self — that check happens in the caller after this.
    """
    if not prediction_succeeded or requested_action is None:
        return "continue", starting_algorithm

    if requested_action == "continue":
        return "continue", starting_algorithm

    target_algo = _SWITCH_TO_ALGO.get(requested_action)
    if target_algo is None:
        # Should never reach here after VALID_ACTIONS check, but be defensive.
        return "continue", starting_algorithm

    return requested_action, target_algo


def _check_sorted(arr: list) -> bool:
    """Return True if arr is non-decreasing."""
    return all(arr[i] <= arr[i + 1] for i in range(len(arr) - 1))


def _safe_checkpoint_snapshot(state: dict) -> dict:
    """Return a shallow copy of the checkpoint state, excluding the mutable array."""
    snap = {k: v for k, v in state.items() if k != "arr"}
    snap["arr_length"] = len(state.get("arr", []))
    return snap
