# Runtime Predictor

**Phase 7.2 — Adaptive Sorting Research**
**Module:** `ml/src/runtime_predictor.py`
**Status:** Production-ready (not for retraining)

---

## 1. Purpose

`RuntimePredictor` is the **in-memory prediction service** for the adaptive
sorting runtime.  It loads the trained production model once and reuses it
across an arbitrary number of checkpoint predictions.

> [!IMPORTANT]
> The predictor **raises exceptions on every failure**. It will never silently
> replace a failure with `'continue'`.  The future Phase 7 adaptive sorting
> controller is responsible for deciding the fallback action (recommended:
> `'continue'`) when a `RuntimePredictorError` or `PredictionError` is caught.

---

## 2. Initialization Lifecycle

```python
from ml.src.runtime_predictor import RuntimePredictor

predictor = RuntimePredictor()
```

During `__init__` the predictor:

1. Resolves the model path (default: `ml/models/adaptive_sort_model.joblib`)
2. Resolves the metadata path (default: `ml/models/adaptive_sort_model_metadata.json`)
3. Confirms both files exist
4. Loads and parses the metadata JSON
5. Validates required metadata keys
6. Cross-validates metadata feature list against `runtime_features.REQUIRED_FEATURES`
7. Cross-validates metadata labels against `runtime_features.VALID_PREDICTION_ACTIONS`
8. Loads the sklearn Pipeline via `joblib.load()` — **this is the only call to `joblib.load`**
9. Validates the pipeline has a callable `predict()` method
10. Validates `pipeline.classes_` matches the expected labels
11. Records the model-loading duration in nanoseconds

After `__init__` completes, the model is held in memory. No I/O or model
construction occurs during subsequent `predict()` calls.

### Custom paths

```python
predictor = RuntimePredictor(
    model_path="path/to/model.joblib",
    metadata_path="path/to/metadata.json",
)
```

Paths are resolved relative to the provided value, not the caller's CWD.
Default paths are resolved relative to the module file location, so they are
safe regardless of the working directory.

---

## 3. Model-Loading Behavior

| Property | Type | Description |
|---|---|---|
| `model_load_ns` | `int` | Nanoseconds spent on `joblib.load()` |
| `model_load_ms` | `float` | Same, in milliseconds |
| `model_path` | `Path` | Resolved path to the loaded artifact |
| `metadata_path` | `Path` | Resolved path to the metadata file |
| `metadata` | `dict` | Parsed metadata (reference, not copy) |

**Loading occurs exactly once per instance.** Multiple calls to `predict()`
reuse the same pipeline object — no reload, no re-parsing.

---

## 4. Prediction Input

Call `predict()` with keyword-only arguments derived from the checkpoint runner
state:

```python
result = predictor.predict(
    current_algorithm="quick_sort",    # str — algorithm currently running
    input_type="random",               # str — supplied at sort start
    size=1000,                         # int — total array elements
    checkpoint_pct=50.0,               # float — progress [0, 100]
    checkpoint_time_ms=1.25,           # float — elapsed time (ms) ≥ 0
    checkpoint_comparisons=4200,       # int — comparisons so far ≥ 0
    checkpoint_data_movements=1700,    # int — element writes so far ≥ 0
)
```

**`input_type` must be passed explicitly by the caller.** It is not inferred
from data.  It must be known when the sort begins (e.g., from experiment
metadata or a data classifier).

### Valid values

| Parameter | Type | Valid values / range |
|---|---|---|
| `current_algorithm` | str | `insertion_sort`, `merge_sort`, `quick_sort` |
| `input_type` | str | `all_equal`, `duplicate_heavy`, `nearly_sorted`, `random`, `reverse_sorted`, `sorted` |
| `size` | int | ≥ 1 |
| `checkpoint_pct` | float | [0, 100] |
| `checkpoint_time_ms` | float | ≥ 0, not NaN, not Inf |
| `checkpoint_comparisons` | int | ≥ 0 |
| `checkpoint_data_movements` | int | ≥ 0 |

Internally `predict()` calls `build_runtime_features()` (from
`runtime_features.py`) which also computes the four derived features before
feeding the DataFrame to the sklearn pipeline.

---

## 5. Prediction Output

`predict()` returns an immutable `RuntimePrediction` dataclass:

```python
@dataclass(frozen=True)
class RuntimePrediction:
    action:           str            # validated model output label
    features:         dict           # exact 11-field payload sent to pipeline
    feature_build_ns: int            # nanoseconds spent on build_runtime_features()
    inference_ns:     int            # nanoseconds spent on pipeline.predict()

    @property
    def feature_build_ms(self) -> float: ...
    @property
    def inference_ms(self) -> float: ...
    @property
    def total_prediction_ms(self) -> float: ...  # = feature_build_ms + inference_ms
```

### Possible `action` values

| Value | Runner call |
|---|---|
| `continue` | `continue_sort(state)` |
| `switch_insertion_sort` | `switch_sort(state, 'insertion_sort')` |
| `switch_merge_sort` | `switch_sort(state, 'merge_sort')` |
| `switch_quick_sort` | `switch_sort(state, 'quick_sort')` |

Actions are validated against `VALID_PREDICTION_ACTIONS` and checked for
switch-to-self (e.g. predicting `switch_quick_sort` while `quick_sort` is
running).

---

## 6. Timing Definitions

| Timing | Includes | Excludes |
|---|---|---|
| `model_load_ns` | `joblib.load()` call duration | Everything else |
| `feature_build_ns` | `build_runtime_features()` call, validation | Model loading, I/O |
| `inference_ns` | `pipeline.predict(df)` call | Feature building, model loading |
| `total_prediction_ms` | `feature_build_ns + inference_ns` | `model_load_ns` |

All timing uses `time.perf_counter_ns()`.

---

## 7. Error Behavior

### Exception hierarchy

```
RuntimePredictorError (RuntimeError)
├── ModelArtifactError    — raised during __init__
└── PredictionError       — raised during predict()
```

### `ModelArtifactError` conditions (raised in `__init__`)

| Condition | Message hint |
|---|---|
| Missing model file | `"Model artifact not found"` |
| Missing metadata file | `"Metadata artifact not found"` |
| Malformed metadata JSON | `"Failed to read or parse metadata"` |
| Missing metadata keys | `"missing required keys"` |
| Incompatible feature list | `"feature order does not match"` |
| Incompatible labels | `"supported_labels"` |
| Artifact without `predict()` | `"callable"` |
| Wrong `classes_` | `"classes_"` |

### `PredictionError` conditions (raised in `predict()`)

| Condition | Message hint |
|---|---|
| Unknown algorithm or input_type | `"Feature building failed"` |
| Negative counts, NaN, Inf | `"Feature building failed"` |
| Model returns empty array | `"empty prediction array"` |
| Model returns >1 predictions | `"Exactly 1"` |
| Unsupported label output | `"Action validation failed"` |
| Switch-to-self | `"Action validation failed"` |

Every `PredictionError` chains its original cause via `__cause__` so the root
exception is never lost.

---

## 8. Fallback Responsibility

```
FALLBACK_ACTION = 'continue'   # defined in runtime_features.py
```

The predictor does **not** apply the fallback.  The calling controller must:

```python
from ml.src.runtime_predictor import RuntimePredictor, RuntimePredictorError
from ml.src.runtime_features import FALLBACK_ACTION

predictor = RuntimePredictor()

try:
    result = predictor.predict(
        current_algorithm=state["algo"],
        input_type="random",
        size=len(state["arr"]),
        checkpoint_pct=state["checkpoint_pct"],
        checkpoint_time_ms=state["time_ms"],
        checkpoint_comparisons=state["comparisons"],
        checkpoint_data_movements=state["moves"],
    )
    action = result.action
except RuntimePredictorError as exc:
    # Log the failure — do not silently ignore it.
    print(f"[ML] prediction failed: {exc} — falling back to '{FALLBACK_ACTION}'")
    action = FALLBACK_ACTION
```

---

## 9. Example Usage

### Minimal usage with defaults

```python
from ml.src.runtime_predictor import RuntimePredictor

predictor = RuntimePredictor()
print(f"Model loaded in {predictor.model_load_ms:.2f} ms")

result = predictor.predict(
    current_algorithm="quick_sort",
    input_type="random",
    size=5000,
    checkpoint_pct=51.3,
    checkpoint_time_ms=4.87,
    checkpoint_comparisons=62500,
    checkpoint_data_movements=31250,
)

print(result.action)                 # e.g. "switch_insertion_sort"
print(result.total_prediction_ms)   # e.g. 0.041
print(result.features)              # dict with all 11 features
```

### Multiple predictions (model loaded once)

```python
predictor = RuntimePredictor()

for state in checkpoint_states:
    result = predictor.predict(
        current_algorithm=state["algo"],
        input_type=state["input_type"],
        size=state["size"],
        checkpoint_pct=state["checkpoint_pct"],
        checkpoint_time_ms=state["time_ms"],
        checkpoint_comparisons=state["comparisons"],
        checkpoint_data_movements=state["moves"],
    )
    # use result.action to drive the runtime
```

---

## 10. Relationship to Other Modules

### `ml/src/runtime_features.py`
`predict()` calls `build_runtime_features()` to construct and validate the
11-field raw feature dict, and `validate_predicted_action()` to validate the
model's output.  No feature engineering is duplicated inside the predictor.

### `ml/predict.py`
The existing standalone CLI continues to work independently.  The CLI uses
`validate_checkpoint_input()` and `predict_action()` from `predict.py` which
accept already-computed derived features.  The `RuntimePredictor` computes
derived features internally via `build_runtime_features()`.  Both paths route
through the same production `.joblib` artifact.

### Future Phase 7.3 — Adaptive Sorting Controller
The controller will:
1. Instantiate `RuntimePredictor` once at startup.
2. Call `predictor.predict(...)` at each checkpoint.
3. Catch `RuntimePredictorError` and fall back to `'continue'` when necessary.
4. Route `result.action` to `continue_sort()` or `switch_sort()` in
   `src/checkpoint/runner.py`.

The controller is **not** implemented in Phase 7.2.
