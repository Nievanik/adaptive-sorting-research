"""
runtime_features.py
-------------------
Phase 7.1 — Runtime Feature Contract

Provides the single public interface between the sorting checkpoint system and
the trained ML model.  Call ``build_runtime_features`` at a checkpoint to
produce a feature dictionary that can be passed directly to ``predict_action``
or ``predict_action_with_confidence`` in ``ml/predict.py``.

Source-of-truth audit (2026-08-01)
------------------------------------
The model artifact ``ml/models/adaptive_sort_model.joblib`` is a scikit-learn
``Pipeline`` with two steps:

  1. ``preprocessor`` — ``ColumnTransformer``
       - ``cat``  : ``OneHotEncoder(handle_unknown='ignore', sparse_output=False)``
                    applied to columns ``['algorithm', 'input_type']``
       - ``num``  : ``passthrough`` applied to the remaining 9 numeric columns

  2. ``classifier`` — ``DecisionTreeClassifier``
       (max_depth=5, min_samples_split=5, min_samples_leaf=2, random_state=42)

``model.predict()`` accepts a ``pd.DataFrame`` whose columns are exactly the
11 raw feature names in the order defined by ``REQUIRED_FEATURES``.

The pipeline handles encoding internally; callers must NOT pre-encode
categorical values.

Verified against
----------------
- ``ml/train.py``             (REQUIRED_FEATURES, LEAKAGE_AND_OUTCOME_COLS)
- ``ml/predict.py``           (validate_checkpoint_input, non-negative checks)
- ``ml/src/preprocess.py``    (build_preprocessor, categorical detection)
- ``ml/src/extract_dataset.py`` (derived-feature formulas)
- ``ml/data/processed/checkpoint_training.csv``  (categories, numeric ranges)
- ``ml/models/adaptive_sort_model.joblib``        (OneHotEncoder categories_)
"""

from __future__ import annotations

import math
from typing import Any

# ---------------------------------------------------------------------------
# Public constants
# ---------------------------------------------------------------------------

#: Categorical column: allowed starting algorithms (verified from OHE categories_)
SUPPORTED_ALGORITHMS: frozenset[str] = frozenset({
    "insertion_sort",
    "merge_sort",
    "quick_sort",
})

#: Categorical column: allowed input distribution types (verified from OHE categories_)
SUPPORTED_INPUT_TYPES: frozenset[str] = frozenset({
    "all_equal",
    "duplicate_heavy",
    "nearly_sorted",
    "random",
    "reverse_sorted",
    "sorted",
})

#: The four labels the model can output (verified from model.classes_)
VALID_PREDICTION_ACTIONS: frozenset[str] = frozenset({
    "continue",
    "switch_insertion_sort",
    "switch_merge_sort",
    "switch_quick_sort",
})

#: Exact ordered feature list fed to the pipeline (from train.py REQUIRED_FEATURES)
REQUIRED_FEATURES: tuple[str, ...] = (
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
    "time_per_element_ms",
)

#: Columns forbidden at inference time because they carry outcome/label information.
#: Mirrors LEAKAGE_AND_OUTCOME_COLS in train.py and predict.py exactly.
FORBIDDEN_LEAKAGE_FIELDS: tuple[str, ...] = (
    "best_action",
    "case",
    "continue_time_ms",
    "continue_comparisons",
    "continue_data_movements",
    "continue_overhead_time_ms",
    "switch_insertion_sort_time_ms",
    "switch_insertion_sort_comparisons",
    "switch_insertion_sort_data_movements",
    "switch_insertion_sort_overhead_time_ms",
    "switch_merge_sort_time_ms",
    "switch_merge_sort_comparisons",
    "switch_merge_sort_data_movements",
    "switch_merge_sort_overhead_time_ms",
    "switch_quick_sort_time_ms",
    "switch_quick_sort_comparisons",
    "switch_quick_sort_data_movements",
    "switch_quick_sort_overhead_time_ms",
    "best_action_total_ms",
    "speedup_vs_continue",
)

#: Recommended Phase 7 fallback action when prediction fails or is invalid.
#: Keeps the original algorithm running — the safest choice when the model
#: cannot produce a confident decision.
FALLBACK_ACTION: str = "continue"


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def build_runtime_features(
    *,
    current_algorithm: str,
    input_type: str,
    array_size: int,
    checkpoint_pct: float,
    checkpoint_time_ms: float,
    comparisons: int,
    moves: int,
) -> dict[str, Any]:
    """Build and validate a runtime feature dictionary for one checkpoint.

    This function reproduces the exact feature engineering applied in
    ``ml/src/extract_dataset.py`` (``flatten_record``) so that inference-time
    inputs are identical in structure and semantics to training-time inputs.

    Parameters
    ----------
    current_algorithm : str
        The sorting algorithm currently running.  Must be one of
        ``SUPPORTED_ALGORITHMS`` (``'insertion_sort'``, ``'merge_sort'``,
        ``'quick_sort'``).
    input_type : str
        The input array distribution type.  Must be one of
        ``SUPPORTED_INPUT_TYPES``.
    array_size : int
        Total number of elements in the array being sorted.  Must be >= 1.
    checkpoint_pct : float
        Percentage of algorithm progress at the checkpoint (0–100).
        Derived from the checkpoint module state (e.g. ``state['checkpoint_pct']``).
    checkpoint_time_ms : float
        Wall-clock time elapsed from sort start to checkpoint, in milliseconds.
        Must be >= 0.
    comparisons : int
        Total element comparisons performed from sort start to checkpoint.
        Must be >= 0.
    moves : int
        Total element moves (writes) performed from sort start to checkpoint.
        1 swap = 2 moves.  Must be >= 0.

    Returns
    -------
    dict[str, Any]
        A dictionary with exactly the 11 keys listed in ``REQUIRED_FEATURES``,
        ready to be passed to ``ml.predict.predict_action`` or
        ``ml.predict.validate_checkpoint_input``.

    Raises
    ------
    TypeError
        If any argument has an incompatible type.
    ValueError
        If any argument fails a range or categorical check, or if a
        derived value would be NaN / infinite after the computation.

    Notes
    -----
    Derived features (reproduced from ``extract_dataset.flatten_record``):

    * ``comparisons_per_element``  = comparisons / array_size
    * ``movements_per_element``    = moves / array_size
    * ``work_ratio``               = comparisons / (moves + 1)   [+1 avoids div-by-zero]
    * ``time_per_element_ms``      = checkpoint_time_ms / array_size

    Zero denominators are handled safely via the ``+1`` guard on ``work_ratio``
    and by the ``array_size >= 1`` precondition (which prevents division-by-zero
    in the other three formulas).
    """
    # ------------------------------------------------------------------ #
    # 1. Type validation                                                   #
    # ------------------------------------------------------------------ #
    if not isinstance(current_algorithm, str):
        raise TypeError(
            f"'current_algorithm' must be a str, got {type(current_algorithm).__name__}"
        )
    if not isinstance(input_type, str):
        raise TypeError(
            f"'input_type' must be a str, got {type(input_type).__name__}"
        )
    if not isinstance(array_size, int):
        raise TypeError(
            f"'array_size' must be an int, got {type(array_size).__name__}"
        )
    if not isinstance(checkpoint_pct, (int, float)):
        raise TypeError(
            f"'checkpoint_pct' must be numeric, got {type(checkpoint_pct).__name__}"
        )
    if not isinstance(checkpoint_time_ms, (int, float)):
        raise TypeError(
            f"'checkpoint_time_ms' must be numeric, got {type(checkpoint_time_ms).__name__}"
        )
    if not isinstance(comparisons, int):
        raise TypeError(
            f"'comparisons' must be an int, got {type(comparisons).__name__}"
        )
    if not isinstance(moves, int):
        raise TypeError(
            f"'moves' must be an int, got {type(moves).__name__}"
        )

    # ------------------------------------------------------------------ #
    # 2. NaN / Inf checks on float inputs                                  #
    # ------------------------------------------------------------------ #
    for name, val in (
        ("checkpoint_pct", float(checkpoint_pct)),
        ("checkpoint_time_ms", float(checkpoint_time_ms)),
    ):
        if math.isnan(val):
            raise ValueError(f"'{name}' must not be NaN.")
        if math.isinf(val):
            raise ValueError(f"'{name}' must not be infinite.")

    # ------------------------------------------------------------------ #
    # 3. Range / domain validation                                         #
    # ------------------------------------------------------------------ #
    if current_algorithm not in SUPPORTED_ALGORITHMS:
        raise ValueError(
            f"'current_algorithm' must be one of {sorted(SUPPORTED_ALGORITHMS)}, "
            f"got {current_algorithm!r}."
        )
    if input_type not in SUPPORTED_INPUT_TYPES:
        raise ValueError(
            f"'input_type' must be one of {sorted(SUPPORTED_INPUT_TYPES)}, "
            f"got {input_type!r}."
        )
    if array_size < 1:
        raise ValueError(
            f"'array_size' must be >= 1, got {array_size}."
        )
    if not (0.0 <= float(checkpoint_pct) <= 100.0):
        raise ValueError(
            f"'checkpoint_pct' must be in [0, 100], got {checkpoint_pct}."
        )
    if float(checkpoint_time_ms) < 0.0:
        raise ValueError(
            f"'checkpoint_time_ms' must be >= 0, got {checkpoint_time_ms}."
        )
    if comparisons < 0:
        raise ValueError(
            f"'comparisons' must be >= 0, got {comparisons}."
        )
    if moves < 0:
        raise ValueError(
            f"'moves' must be >= 0, got {moves}."
        )

    # ------------------------------------------------------------------ #
    # 4. Derived feature computation                                       #
    #    Formulas mirror extract_dataset.flatten_record exactly.           #
    # ------------------------------------------------------------------ #
    size_f = float(array_size)
    cmp_f = float(comparisons)
    mov_f = float(moves)
    time_f = float(checkpoint_time_ms)

    # array_size >= 1 guarantees no ZeroDivisionError here.
    comparisons_per_element: float = cmp_f / size_f
    movements_per_element: float = mov_f / size_f
    # +1 in denominator prevents ZeroDivisionError when moves == 0.
    work_ratio: float = cmp_f / (mov_f + 1.0)
    time_per_element_ms: float = time_f / size_f

    # ------------------------------------------------------------------ #
    # 5. Sanity-check derived values                                       #
    # ------------------------------------------------------------------ #
    derived = {
        "comparisons_per_element": comparisons_per_element,
        "movements_per_element": movements_per_element,
        "work_ratio": work_ratio,
        "time_per_element_ms": time_per_element_ms,
    }
    for fname, fval in derived.items():
        if math.isnan(fval):
            raise ValueError(
                f"Derived feature '{fname}' is NaN — check input values."
            )
        if math.isinf(fval):
            raise ValueError(
                f"Derived feature '{fname}' is infinite — check input values."
            )

    # ------------------------------------------------------------------ #
    # 6. Assemble the feature dict in required order                       #
    # ------------------------------------------------------------------ #
    features: dict[str, Any] = {
        "algorithm":                current_algorithm,
        "input_type":               input_type,
        "size":                     array_size,
        "checkpoint_pct":           float(checkpoint_pct),
        "checkpoint_time_ms":       time_f,
        "checkpoint_comparisons":   comparisons,
        "checkpoint_data_movements": moves,
        "comparisons_per_element":  comparisons_per_element,
        "movements_per_element":    movements_per_element,
        "work_ratio":               work_ratio,
        "time_per_element_ms":      time_per_element_ms,
    }

    # Confirm no leakage fields are present (belt-and-suspenders).
    _reject_leakage(features)

    return features


def validate_predicted_action(
    action: str,
    current_algorithm: str,
) -> str:
    """Validate a model-predicted action and return a normalized executable label.

    Parameters
    ----------
    action : str
        The raw string returned by the model (e.g. ``model.predict(df)[0]``).
    current_algorithm : str
        The algorithm that was running when the checkpoint was taken.  Used to
        detect the degenerate case where the model predicts switching to the
        algorithm that is already running.

    Returns
    -------
    str
        The validated action string, identical to ``action`` when valid.

    Raises
    ------
    ValueError
        If ``action`` is not one of ``VALID_PREDICTION_ACTIONS``, or if
        ``current_algorithm`` is not one of ``SUPPORTED_ALGORITHMS``.
        Prediction failures are intentionally surfaced — the caller is
        responsible for choosing the fallback (``FALLBACK_ACTION = 'continue'``
        is the Phase 7 recommendation).

    Notes
    -----
    Distinguishing ``continue`` from switching to the same algorithm
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    The model can only predict ``'continue'``, ``'switch_insertion_sort'``,
    ``'switch_merge_sort'``, or ``'switch_quick_sort'``.  It cannot predict
    switching *to* the algorithm that is already running (this never appears
    in the training labels), so no remapping is needed.  However, a defensive
    check is included: if the predicted action maps to the same algorithm as
    ``current_algorithm``, a ``ValueError`` is raised so the caller can handle
    it explicitly rather than silently treating it as a no-op.

    Phase 7 fallback guidance
    ~~~~~~~~~~~~~~~~~~~~~~~~~
    If this function raises, the recommended Phase 7 behaviour is:

        try:
            action = validate_predicted_action(raw, current_algorithm)
        except ValueError:
            action = FALLBACK_ACTION   # == 'continue'
    """
    if not isinstance(action, str):
        raise ValueError(
            f"Predicted action must be a str, got {type(action).__name__}: {action!r}"
        )
    if action not in VALID_PREDICTION_ACTIONS:
        raise ValueError(
            f"Predicted action {action!r} is not a valid model output. "
            f"Expected one of: {sorted(VALID_PREDICTION_ACTIONS)}."
        )
    if current_algorithm not in SUPPORTED_ALGORITHMS:
        raise ValueError(
            f"'current_algorithm' must be one of {sorted(SUPPORTED_ALGORITHMS)}, "
            f"got {current_algorithm!r}."
        )

    # Detect degenerate switch-to-self predictions.
    # Map switch labels → algorithm name.
    _switch_to_algo = {
        "switch_insertion_sort": "insertion_sort",
        "switch_merge_sort":     "merge_sort",
        "switch_quick_sort":     "quick_sort",
    }
    target_algo = _switch_to_algo.get(action)
    if target_algo is not None and target_algo == current_algorithm:
        raise ValueError(
            f"Model predicted {action!r}, but {current_algorithm!r} is already "
            f"running — this is a degenerate switch-to-self. "
            f"Use FALLBACK_ACTION ('{FALLBACK_ACTION}') or investigate the model."
        )

    return action


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _reject_leakage(features: dict[str, Any]) -> None:
    """Raise ValueError if any forbidden leakage field is present in the dict."""
    found = [k for k in features if k in FORBIDDEN_LEAKAGE_FIELDS]
    if found:
        raise ValueError(
            f"Feature dict contains forbidden leakage/outcome fields: {found}. "
            f"These fields must never be passed to the model at runtime."
        )
