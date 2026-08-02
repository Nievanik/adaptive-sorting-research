"""
runtime_predictor.py
--------------------
Phase 7.2 — In-Memory Runtime Predictor

Provides a production-safe, single-load prediction service for the adaptive
sorting runtime.  The model is loaded exactly once per ``RuntimePredictor``
instance and reused across all subsequent ``predict()`` calls.

Relationship to other modules
------------------------------
- ``ml/src/runtime_features.py``  — ``build_runtime_features()`` and
  ``validate_predicted_action()`` are called here; no feature engineering
  is duplicated.
- ``ml/predict.py``               — standalone CLI; ``load_model`` and
  ``validate_checkpoint_input`` from that module are intentionally kept
  separate.  The CLI can call ``RuntimePredictor`` for a unified code path,
  but the predictor itself does **not** depend on the CLI module.
- ``src/checkpoint/runner.py``    — Phase 7.3 adaptive controller will use
  checkpoint state to drive calls into this predictor.

Error philosophy
----------------
The predictor raises on every failure.  The Phase 7 adaptive sorter
controller is responsible for catching ``RuntimePredictorError`` (or its
subclasses) and deciding the fallback action (recommended: ``'continue'``).
The predictor will never silently replace a failure with ``'continue'``.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib
import pandas as pd

# ---------------------------------------------------------------------------
# Path defaults
# ---------------------------------------------------------------------------

_MODULE_DIR = Path(__file__).resolve().parent          # ml/src/
_ML_DIR = _MODULE_DIR.parent                           # ml/
_PROJECT_ROOT = _ML_DIR.parent                         # adaptive-sorting-research/

_DEFAULT_MODEL_PATH = _ML_DIR / "models" / "adaptive_sort_model.joblib"
_DEFAULT_METADATA_PATH = _ML_DIR / "models" / "adaptive_sort_model_metadata.json"

# Required metadata keys the JSON file must contain.
_REQUIRED_METADATA_KEYS: tuple[str, ...] = (
    "ordered_required_feature_list",
    "supported_labels",
    "model_class",
    "selected_model_name",
)


# ---------------------------------------------------------------------------
# Custom exceptions
# ---------------------------------------------------------------------------

class RuntimePredictorError(RuntimeError):
    """Base class for all RuntimePredictor errors."""


class ModelArtifactError(RuntimePredictorError):
    """Raised when the model or metadata artifact is missing or malformed."""


class PredictionError(RuntimePredictorError):
    """Raised when a prediction attempt fails at any stage."""


# ---------------------------------------------------------------------------
# Prediction result dataclass
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class RuntimePrediction:
    """Immutable result of a single checkpoint prediction.

    Attributes
    ----------
    action : str
        Validated model output label.  One of:
        ``'continue'``, ``'switch_insertion_sort'``,
        ``'switch_merge_sort'``, ``'switch_quick_sort'``.
    features : dict[str, object]
        Exact 11-feature payload supplied to the pipeline (raw, un-encoded).
    feature_build_ns : int
        Time spent building and validating features, in nanoseconds.
    inference_ns : int
        Time spent running the sklearn ``pipeline.predict()`` call, in
        nanoseconds.  Does not include model-loading time.
    """

    action: str
    features: dict[str, Any]
    feature_build_ns: int
    inference_ns: int

    @property
    def feature_build_ms(self) -> float:
        """Feature-build duration in milliseconds."""
        return self.feature_build_ns / 1_000_000

    @property
    def inference_ms(self) -> float:
        """Model inference duration in milliseconds."""
        return self.inference_ns / 1_000_000

    @property
    def total_prediction_ms(self) -> float:
        """Total per-prediction duration (feature build + inference) in ms.

        Does **not** include model-loading time.
        """
        return (self.feature_build_ns + self.inference_ns) / 1_000_000


# ---------------------------------------------------------------------------
# Runtime Predictor
# ---------------------------------------------------------------------------

class RuntimePredictor:
    """In-memory prediction service for the adaptive sorting runtime.

    The production model is loaded **once** during ``__init__`` and reused
    across all calls to ``predict()``.  Multiple predictions may be made on
    the same instance with no additional I/O or model reconstruction.

    Parameters
    ----------
    model_path : str | Path | None
        Path to the ``.joblib`` model artifact.  Defaults to the production
        model at ``ml/models/adaptive_sort_model.joblib``, resolved relative
        to this module's location (not the caller's CWD).
    metadata_path : str | Path | None
        Path to the JSON metadata file.  Defaults to
        ``ml/models/adaptive_sort_model_metadata.json``.

    Raises
    ------
    ModelArtifactError
        If either artifact is missing, unreadable, malformed, or
        incompatible with the expected feature/label schema.

    Examples
    --------
    >>> predictor = RuntimePredictor()
    >>> result = predictor.predict(
    ...     current_algorithm="quick_sort",
    ...     input_type="random",
    ...     size=1000,
    ...     checkpoint_pct=50.0,
    ...     checkpoint_time_ms=1.25,
    ...     checkpoint_comparisons=4200,
    ...     checkpoint_data_movements=1700,
    ... )
    >>> result.action
    'switch_insertion_sort'
    >>> result.total_prediction_ms
    0.123   # example only; actual value depends on hardware
    """

    def __init__(
        self,
        model_path: str | Path | None = None,
        metadata_path: str | Path | None = None,
    ) -> None:
        self._model_path = Path(model_path) if model_path is not None else _DEFAULT_MODEL_PATH
        self._metadata_path = Path(metadata_path) if metadata_path is not None else _DEFAULT_METADATA_PATH

        # Load metadata first (cheaper, validates artifact location)
        self._metadata: dict[str, Any] = self._load_metadata()
        self._validate_metadata(self._metadata)

        # Load and validate the model pipeline exactly once
        t0 = time.perf_counter_ns()
        self._pipeline = self._load_pipeline()
        self._model_load_ns: int = time.perf_counter_ns() - t0

        self._validate_pipeline(self._pipeline)

    # ------------------------------------------------------------------
    # Read-only public properties
    # ------------------------------------------------------------------

    @property
    def model_path(self) -> Path:
        """Path to the loaded model artifact."""
        return self._model_path

    @property
    def metadata_path(self) -> Path:
        """Path to the loaded metadata artifact."""
        return self._metadata_path

    @property
    def metadata(self) -> dict[str, Any]:
        """Loaded metadata dictionary (read-only reference)."""
        return self._metadata

    @property
    def model_load_ns(self) -> int:
        """Model-loading duration in nanoseconds."""
        return self._model_load_ns

    @property
    def model_load_ms(self) -> float:
        """Model-loading duration in milliseconds."""
        return self._model_load_ns / 1_000_000

    # ------------------------------------------------------------------
    # Public prediction method
    # ------------------------------------------------------------------

    def predict(
        self,
        *,
        current_algorithm: str,
        input_type: str,
        size: int,
        checkpoint_pct: float,
        checkpoint_time_ms: float,
        checkpoint_comparisons: int,
        checkpoint_data_movements: int,
    ) -> RuntimePrediction:
        """Predict the optimal sorting action at a mid-execution checkpoint.

        Parameters
        ----------
        current_algorithm : str
            The algorithm currently running.
            Must be one of ``{'insertion_sort', 'merge_sort', 'quick_sort'}``.
        input_type : str
            Input array distribution type, supplied by the caller at sort
            start.  Must be one of ``{'all_equal', 'duplicate_heavy',
            'nearly_sorted', 'random', 'reverse_sorted', 'sorted'}``.
        size : int
            Total number of elements in the array.  Must be >= 1.
        checkpoint_pct : float
            Percentage of algorithm progress at the checkpoint (0–100).
            From ``state['checkpoint_pct']`` in the checkpoint runner.
        checkpoint_time_ms : float
            Elapsed wall-clock time from sort start to checkpoint (ms).
            From ``state['time_ms']``.  Must be >= 0.
        checkpoint_comparisons : int
            Total element comparisons from sort start to checkpoint.
            From ``state['comparisons']``.  Must be >= 0.
        checkpoint_data_movements : int
            Total element writes from sort start to checkpoint.
            From ``state['moves']``.  Must be >= 0.

        Returns
        -------
        RuntimePrediction
            Immutable result containing the validated action, the feature
            payload, and per-phase timing measurements.

        Raises
        ------
        PredictionError
            If feature building, model inference, or action validation fails.
            The original exception is chained via ``__cause__``.

        Notes
        -----
        The model pipeline is **not** reloaded on each call.  The same
        ``Pipeline`` instance loaded during ``__init__`` is reused.
        """
        # ---- Stage 1: build features -----------------------------------
        t_feat_start = time.perf_counter_ns()
        try:
            from ml.src.runtime_features import build_runtime_features
            features = build_runtime_features(
                current_algorithm=current_algorithm,
                input_type=input_type,
                array_size=size,
                checkpoint_pct=checkpoint_pct,
                checkpoint_time_ms=checkpoint_time_ms,
                comparisons=checkpoint_comparisons,
                moves=checkpoint_data_movements,
            )
        except Exception as exc:
            raise PredictionError(
                f"Feature building failed: {exc}"
            ) from exc
        t_feat_end = time.perf_counter_ns()
        feature_build_ns = t_feat_end - t_feat_start

        # ---- Stage 2: construct DataFrame in required column order -----
        required_features: list[str] = self._metadata["ordered_required_feature_list"]
        try:
            df = pd.DataFrame([features])[required_features]
        except KeyError as exc:
            raise PredictionError(
                f"Feature dict is missing columns required by the pipeline: {exc}"
            ) from exc

        # ---- Stage 3: run sklearn pipeline inference -------------------
        t_inf_start = time.perf_counter_ns()
        try:
            raw_predictions = self._pipeline.predict(df)
        except Exception as exc:
            raise PredictionError(
                f"Model inference failed: {exc}"
            ) from exc
        t_inf_end = time.perf_counter_ns()
        inference_ns = t_inf_end - t_inf_start

        # ---- Stage 4: validate prediction output -----------------------
        if len(raw_predictions) == 0:
            raise PredictionError("Model returned an empty prediction array.")
        if len(raw_predictions) > 1:
            raise PredictionError(
                f"Model returned {len(raw_predictions)} predictions for a single-row input; "
                "expected exactly 1."
            )

        raw_action = str(raw_predictions[0])

        try:
            from ml.src.runtime_features import validate_predicted_action
            action = validate_predicted_action(raw_action, current_algorithm)
        except Exception as exc:
            raise PredictionError(
                f"Action validation failed for predicted label {raw_action!r}: {exc}"
            ) from exc

        return RuntimePrediction(
            action=action,
            features=dict(features),   # defensive copy of the feature dict
            feature_build_ns=feature_build_ns,
            inference_ns=inference_ns,
        )

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _load_metadata(self) -> dict[str, Any]:
        """Load and return the metadata JSON.  Raises ModelArtifactError."""
        if not self._metadata_path.exists():
            raise ModelArtifactError(
                f"Metadata artifact not found: {self._metadata_path}"
            )
        try:
            with self._metadata_path.open("r", encoding="utf-8") as f:
                data = json.load(f)
        except (json.JSONDecodeError, OSError) as exc:
            raise ModelArtifactError(
                f"Failed to read or parse metadata at {self._metadata_path}: {exc}"
            ) from exc
        if not isinstance(data, dict):
            raise ModelArtifactError(
                f"Metadata at {self._metadata_path} is not a JSON object."
            )
        return data

    def _validate_metadata(self, metadata: dict[str, Any]) -> None:
        """Check required metadata fields and cross-validate against runtime_features."""
        missing = [k for k in _REQUIRED_METADATA_KEYS if k not in metadata]
        if missing:
            raise ModelArtifactError(
                f"Metadata is missing required keys: {missing}"
            )

        # Validate feature list matches runtime_features.REQUIRED_FEATURES
        from ml.src.runtime_features import REQUIRED_FEATURES as RF_FEATURES
        meta_features: list[str] = metadata["ordered_required_feature_list"]
        if not isinstance(meta_features, list) or not meta_features:
            raise ModelArtifactError(
                "'ordered_required_feature_list' in metadata must be a non-empty list."
            )
        if list(meta_features) != list(RF_FEATURES):
            raise ModelArtifactError(
                f"Metadata feature order does not match runtime_features.REQUIRED_FEATURES.\n"
                f"  Metadata: {meta_features}\n"
                f"  Expected: {list(RF_FEATURES)}"
            )

        # Validate supported labels match VALID_PREDICTION_ACTIONS
        from ml.src.runtime_features import VALID_PREDICTION_ACTIONS
        meta_labels: list[str] = metadata.get("supported_labels", [])
        if not isinstance(meta_labels, list) or not meta_labels:
            raise ModelArtifactError(
                "'supported_labels' in metadata must be a non-empty list."
            )
        if set(meta_labels) != VALID_PREDICTION_ACTIONS:
            raise ModelArtifactError(
                f"Metadata supported_labels do not match VALID_PREDICTION_ACTIONS.\n"
                f"  Metadata: {sorted(meta_labels)}\n"
                f"  Expected: {sorted(VALID_PREDICTION_ACTIONS)}"
            )

    def _load_pipeline(self) -> Any:
        """Load the joblib model pipeline.  Raises ModelArtifactError."""
        if not self._model_path.exists():
            raise ModelArtifactError(
                f"Model artifact not found: {self._model_path}"
            )
        try:
            pipeline = joblib.load(self._model_path)
        except Exception as exc:
            raise ModelArtifactError(
                f"Failed to load model from {self._model_path}: {exc}"
            ) from exc
        return pipeline

    def _validate_pipeline(self, pipeline: Any) -> None:
        """Validate the loaded pipeline is a usable sklearn Pipeline."""
        if not callable(getattr(pipeline, "predict", None)):
            raise ModelArtifactError(
                f"Loaded artifact at {self._model_path} does not have a callable "
                "'predict()' method.  Is it a fitted sklearn Pipeline?"
            )

        # Validate model.classes_ match VALID_PREDICTION_ACTIONS
        from ml.src.runtime_features import VALID_PREDICTION_ACTIONS
        classes = getattr(pipeline, "classes_", None)
        if classes is None:
            raise ModelArtifactError(
                "Loaded model pipeline has no 'classes_' attribute — "
                "it may not be a fitted classifier."
            )
        model_classes = set(str(c) for c in classes)
        if model_classes != VALID_PREDICTION_ACTIONS:
            raise ModelArtifactError(
                f"Model classes_ do not match VALID_PREDICTION_ACTIONS.\n"
                f"  Model classes_: {sorted(model_classes)}\n"
                f"  Expected:       {sorted(VALID_PREDICTION_ACTIONS)}"
            )
