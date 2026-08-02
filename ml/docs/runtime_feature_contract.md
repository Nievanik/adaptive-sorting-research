# Runtime Feature Contract

**Phase 7.1 — Adaptive Sorting Research**
**Date:** 2026-08-01
**Status:** Verified against live codebase and model artifact

---

## 1. Purpose

This document defines the exact interface between the sorting checkpoint system
and the trained ML decision model.  Any code that calls the production model at
runtime **must** conform to this contract.  The contract was derived from direct
inspection of:

- `ml/train.py` — `REQUIRED_FEATURES`, `LEAKAGE_AND_OUTCOME_COLS`
- `ml/src/preprocess.py` — `build_preprocessor`, categorical column detection
- `ml/src/extract_dataset.py` — `flatten_record`, derived-feature formulas
- `ml/predict.py` — `validate_checkpoint_input`, numeric-range rules
- `ml/data/processed/checkpoint_training.csv` — actual categories and numeric ranges
- `ml/models/adaptive_sort_model.joblib` — OHE `categories_`, classifier `classes_`

---

## 2. Model Artifact Structure

| Component | Description |
|---|---|
| File | `ml/models/adaptive_sort_model.joblib` |
| Type | `sklearn.pipeline.Pipeline` |
| Step 1 | `preprocessor` — `ColumnTransformer` |
| Step 2 | `classifier` — `DecisionTreeClassifier` |

**The pipeline includes preprocessing.**  Callers pass **raw, un-encoded** feature
values.  The pipeline applies OneHotEncoding internally.

---

## 3. Runtime Input Schema

The model accepts a `pd.DataFrame` with exactly **11 columns** in this order:

| # | Field | Type | Description |
|---|---|---|---|
| 1 | `algorithm` | `str` (categorical) | Starting algorithm |
| 2 | `input_type` | `str` (categorical) | Input array distribution |
| 3 | `size` | `int` | Total array element count |
| 4 | `checkpoint_pct` | `float` | Progress at checkpoint (0–100) |
| 5 | `checkpoint_time_ms` | `float` | Elapsed time to checkpoint (ms) |
| 6 | `checkpoint_comparisons` | `int` | Comparisons performed to checkpoint |
| 7 | `checkpoint_data_movements` | `int` | Element writes performed to checkpoint |
| 8 | `comparisons_per_element` | `float` | **Derived** |
| 9 | `movements_per_element` | `float` | **Derived** |
| 10 | `work_ratio` | `float` | **Derived** |
| 11 | `time_per_element_ms` | `float` | **Derived** |

Column order matters only if the caller builds a DataFrame manually.
`build_runtime_features()` returns a dict already in this order.

---

## 4. Categorical Fields and Allowed Values

### `algorithm`

| Value | Meaning |
|---|---|
| `insertion_sort` | InsertionSort was the starting algorithm |
| `merge_sort` | MergeSort was the starting algorithm |
| `quick_sort` | QuickSort was the starting algorithm |

Values are **lowercase with underscores** — exactly as used in `src/checkpoint/runner.py`.

### `input_type`

| Value | Description |
|---|---|
| `all_equal` | Every element is identical |
| `duplicate_heavy` | High frequency of repeated values |
| `nearly_sorted` | Sorted with a small number of random swaps |
| `random` | Uniformly random values |
| `reverse_sorted` | Descending order |
| `sorted` | Ascending order |

> [!WARNING]
> The OneHotEncoder was trained with `handle_unknown='ignore'`.  Unknown values
> (e.g., `'real_world'`) produce an all-zeros OHE row — the model will not
> raise an error but predictions are unreliable.  Always use an allowed value.

---

## 5. Numeric Fields and Valid Ranges

| Field | Source | Valid Range | Notes |
|---|---|---|---|
| `size` | `array_size` | ≥ 1 | Integer |
| `checkpoint_pct` | `state['checkpoint_pct']` | [0, 100] | Float |
| `checkpoint_time_ms` | `state['time_ms']` | ≥ 0 | Float |
| `checkpoint_comparisons` | `state['comparisons']` | ≥ 0 | Integer |
| `checkpoint_data_movements` | `state['moves']` | ≥ 0 | Integer |

**Observed training-data ranges** (from the 90-row dataset):

| Field | Min | Max |
|---|---|---|
| `size` | 100 | 10 000 |
| `checkpoint_pct` | 49.97 | 58.70 |
| `checkpoint_time_ms` | 0.0068 | 849.62 |
| `checkpoint_comparisons` | 49 | 12 497 500 |
| `checkpoint_data_movements` | 49 | 12 502 499 |

---

## 6. Derived Feature Formulas

These four features are **computed** from the five source fields above.
The formulas reproduce `extract_dataset.flatten_record` exactly.

```python
comparisons_per_element = checkpoint_comparisons / array_size
movements_per_element   = checkpoint_data_movements / array_size
work_ratio              = checkpoint_comparisons / (checkpoint_data_movements + 1)
time_per_element_ms     = checkpoint_time_ms / array_size
```

**Zero denominator handling:**

- `comparisons_per_element`, `movements_per_element`, `time_per_element_ms`:
  No risk — `array_size >= 1` is required.
- `work_ratio`: `+1` added to the denominator so that `moves == 0` yields
  `comparisons / 1 = comparisons` instead of `ZeroDivisionError`.

---

## 7. Forbidden Leakage Fields

The following fields must **never** be present in a runtime input.  They are
post-checkpoint outcome measurements that the model cannot know during inference.

| Field |
|---|
| `best_action` |
| `case` |
| `continue_time_ms` |
| `continue_comparisons` |
| `continue_data_movements` |
| `continue_overhead_time_ms` |
| `switch_insertion_sort_time_ms` |
| `switch_insertion_sort_comparisons` |
| `switch_insertion_sort_data_movements` |
| `switch_insertion_sort_overhead_time_ms` |
| `switch_merge_sort_time_ms` |
| `switch_merge_sort_comparisons` |
| `switch_merge_sort_data_movements` |
| `switch_merge_sort_overhead_time_ms` |
| `switch_quick_sort_time_ms` |
| `switch_quick_sort_comparisons` |
| `switch_quick_sort_data_movements` |
| `switch_quick_sort_overhead_time_ms` |
| `best_action_total_ms` |
| `speedup_vs_continue` |

`build_runtime_features()` is structurally incapable of producing any of these.

---

## 8. Model Output Labels

The model produces exactly one of:

| Label | Meaning |
|---|---|
| `continue` | Finish the sort with the original algorithm |
| `switch_insertion_sort` | Switch to InsertionSort for the remaining work |
| `switch_merge_sort` | Switch to MergeSort for the remaining work |
| `switch_quick_sort` | Switch to QuickSort for the remaining work |

Output labels require **no normalization** before use in the sorting runtime.
They are identical to the algorithm names used in `src/checkpoint/runner.py`
(`switch_sort(state, 'insertion_sort')` etc.) except for the `continue` action
which maps directly to `continue_sort(state)`.

**Label-to-runner mapping:**

| Model output | Runner call |
|---|---|
| `continue` | `continue_sort(state)` |
| `switch_insertion_sort` | `switch_sort(state, 'insertion_sort')` |
| `switch_merge_sort` | `switch_sort(state, 'merge_sort')` |
| `switch_quick_sort` | `switch_sort(state, 'quick_sort')` |

---

## 9. Error Handling

### In `build_runtime_features`

| Error condition | Exception |
|---|---|
| Wrong type for any parameter | `TypeError` |
| Unknown `current_algorithm` | `ValueError` |
| Unknown `input_type` | `ValueError` |
| `array_size < 1` | `ValueError` |
| `checkpoint_pct` outside [0, 100] | `ValueError` |
| `checkpoint_time_ms < 0` | `ValueError` |
| `comparisons < 0` | `ValueError` |
| `moves < 0` | `ValueError` |
| NaN or Inf in float inputs | `ValueError` |
| NaN or Inf in derived outputs | `ValueError` |
| Leakage field in result (belt-and-suspenders) | `ValueError` |

### In `validate_predicted_action`

| Error condition | Exception |
|---|---|
| `action` not a str | `ValueError` |
| `action` not in `VALID_PREDICTION_ACTIONS` | `ValueError` |
| `current_algorithm` not in `SUPPORTED_ALGORITHMS` | `ValueError` |
| Switch-to-self (model predicts switching to already-running algorithm) | `ValueError` |

---

## 10. Fallback Recommendation

```
FALLBACK_ACTION = 'continue'
```

If `build_runtime_features` raises or `validate_predicted_action` raises, Phase 7
code should catch the exception, log it, and fall back to `continue_sort(state)`.
This ensures the sort always completes correctly even if the ML inference path fails.

```python
from ml.src.runtime_features import (
    build_runtime_features, validate_predicted_action, FALLBACK_ACTION
)
from ml.predict import load_model, predict_action

model = load_model("ml/models/adaptive_sort_model.joblib")

try:
    features = build_runtime_features(
        current_algorithm=state["algo"],
        input_type="random",
        array_size=len(state["arr"]),
        checkpoint_pct=state["checkpoint_pct"],
        checkpoint_time_ms=state["time_ms"],
        comparisons=state["comparisons"],
        moves=state["moves"],
    )
    raw_action = predict_action(model, features)
    action = validate_predicted_action(raw_action, state["algo"])
except Exception as exc:
    print(f"[ML] inference failed: {exc} — falling back to '{FALLBACK_ACTION}'")
    action = FALLBACK_ACTION
```

---

## 11. Example Payloads

### Valid payload (quick_sort, random, n=1000, 50% checkpoint)

```python
{
    "algorithm":                "quick_sort",
    "input_type":               "random",
    "size":                     1000,
    "checkpoint_pct":           50.0,
    "checkpoint_time_ms":       1.25,
    "checkpoint_comparisons":   4200,
    "checkpoint_data_movements": 1700,
    "comparisons_per_element":  4.2,      # 4200 / 1000
    "movements_per_element":    1.7,      # 1700 / 1000
    "work_ratio":               2.470588, # 4200 / (1700 + 1)
    "time_per_element_ms":      0.00125,  # 1.25 / 1000
}
```

### Invalid payload examples

```python
# Missing required field
{"algorithm": "quick_sort", "input_type": "random"}  # raises ValueError

# Unknown algorithm
{"algorithm": "heapsort", ...}  # raises ValueError

# Negative comparisons
{"comparisons": -5, ...}  # raises ValueError

# Leakage field present
{"continue_time_ms": 3.2, ...}  # raises ValueError (leakage guard)

# NaN value
{"checkpoint_time_ms": float('nan'), ...}  # raises ValueError
```

---

## 12. Mapping to Training Pipeline

```
Checkpoint state (runner.py)
        │
        ▼
build_runtime_features()          ← ml/src/runtime_features.py
        │  validates + computes derived features
        │  (mirrors extract_dataset.flatten_record)
        ▼
dict with 11 raw feature values
        │
        ▼
validate_checkpoint_input()       ← ml/predict.py (optional; already called
        │  reorders columns, checks types & bounds   by predict_action)
        ▼
pd.DataFrame (1 row, 11 cols)
        │
        ▼
model.predict(df)                 ← ml/models/adaptive_sort_model.joblib
  │ ColumnTransformer:
  │   OneHotEncoder(algorithm, input_type)
  │   passthrough(9 numeric cols)
  │ DecisionTreeClassifier
        │
        ▼
raw predicted label  (str)
        │
        ▼
validate_predicted_action()       ← ml/src/runtime_features.py
        │  checks label, guards switch-to-self
        ▼
executable action  (str)
        │
        ▼
continue_sort(state)  OR
switch_sort(state, target_algo)   ← src/checkpoint/runner.py
```
