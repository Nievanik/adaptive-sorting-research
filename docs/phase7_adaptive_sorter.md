# Phase 7.3 — Adaptive Sorter Controller

## Purpose
The Adaptive Sorter Controller provides the first end-to-end adaptive execution path that ties together our benchmarked algorithms, mid-execution checkpointing, switching mechanics, and the runtime machine learning model. It decides and executes a single optimal action during a live sorting run based on dynamically constructed runtime features, proving the feasibility of ML-driven adaptive sorting without the exhaustive overhead of full benchmarks.

## Lifecycle
The controller adheres strictly to a linear execution path:
1. **Validation**: Assert arguments and constraints.
2. **Copy Input**: The caller's sequence is cloned to avoid unexpected side effects.
3. **Execution to Checkpoint**: The starting algorithm processes the data until its predefined 50% checkpoint.
4. **Feature Extraction & Inference**: `RuntimePredictor.predict()` is called exactly once to query the ML model, supplying current metadata (time, metrics).
5. **Action Resolution**: The model's response is sanitized and mapped to a concrete sorting action.
6. **Continuation or Switch**: The controller delegates to the `src.checkpoint.runner` to either continue the existing algorithm or switch and complete sorting.
7. **Result Generation**: Aggregate all timing and performance metrics and return an immutable `AdaptiveSortResult`.

### Execution Flow Diagram

```mermaid
flowchart TD
    Start([Start adaptive_sort]) --> Validate[Validate Arguments]
    Validate --> Copy[Copy Input Array]
    Copy --> Trivial{Size <= 1?}
    Trivial -- Yes --> FinishTrivial[Sort & Return Instantly]
    Trivial -- No --> RunCP[Run to Checkpoint (50%)]
    RunCP --> Predict[Call RuntimePredictor once]
    
    Predict --> Valid{Action Valid?}
    Valid -- No / Error --> Fallback[Fallback: continue]
    Valid -- Yes --> SelfCheck{Switch to Self?}
    
    SelfCheck -- Yes --> Fallback
    SelfCheck -- No --> Resolve[Resolve Action]
    
    Fallback --> Exec[Execute Resolved Action]
    Resolve --> Exec
    
    Exec --> ExecCheck{Is Continue?}
    ExecCheck -- Yes --> Continue[Runner: continue_sort]
    ExecCheck -- No --> Switch[Runner: switch_sort]
    
    Continue --> Finish[Collect Metrics & Return]
    Switch --> Finish
    Finish --> End([Return AdaptiveSortResult])
```

## Public API

The primary interface is a single standalone function:

```python
from src.adaptive.adaptive_sorter import adaptive_sort

result = adaptive_sort(
    values=[5, 2, 9, 1, 5, 6],
    starting_algorithm="quick_sort",
    input_type="random",
    predictor=predictor,
    checkpoint_pct=50.0,
)
```

## Supported Algorithms
The controller allows the following starting algorithms, matching the capabilities of the underlying checkpoint runner:
* `insertion_sort`
* `merge_sort`
* `quick_sort`

## Supported Actions
The model must return one of the following exact string labels, which the controller resolves into execution pathways:
* `continue`
* `switch_insertion_sort`
* `switch_merge_sort`
* `switch_quick_sort`

## Input-Type Requirement
The input data distribution type (`input_type`) **cannot** be inferred directly by the controller. It must be explicitly provided by the caller using one of the supported strings (`random`, `sorted`, `reverse_sorted`, `nearly_sorted`, `duplicate_heavy`, `all_equal`).

## Mutation/Copying Policy
The adaptive controller uses a strict **copy-once** policy:
* It immediately calls `list(values)` on the input.
* The original input sequence is never mutated.
* All subsequent checkpoint tracking and switching logic mutates the internal copy.

## Prediction Fallback Behavior
The controller is designed to handle prediction failures gracefully by falling back to the `continue` action. This occurs when:
1. `RuntimePredictor.predict()` raises any `RuntimePredictorError` (e.g. invalid features, model artifact missing).
2. The predictor returns an unsupported action string (e.g., `"switch_heap_sort"`).
3. The predictor returns an object missing the `action` attribute.

In all fallback scenarios, `fallback_used=True` and a machine-readable string is provided in `fallback_reason`. The controller **does not** swallow unexpected system errors (e.g., `KeyboardInterrupt` or unrelated bugs).

## Switch-to-Self Behavior
If the model correctly predicts a switch action but the target algorithm is the *same* as the currently running algorithm (e.g., starting with `quick_sort` and predicting `switch_quick_sort`), the controller treats this as a fallback condition. 
It will:
* Set `executed_action = "continue"`
* Record `fallback_used = True`
* Record `fallback_reason = "switch_to_self"`

## Result Fields
The function returns an immutable `AdaptiveSortResult` dataclass with the following key attributes:
* `sorted_values`: The final sorted output (as a tuple).
* `starting_algorithm`, `final_algorithm`: Track the algorithm lifecycle.
* `requested_action`, `executed_action`: Differentiate between what the model asked for and what was actually performed.
* `prediction_succeeded`, `fallback_used`, `fallback_reason`: Status flags for the ML component.
* `checkpoint`: A safe dictionary snapshot of the mid-execution state (excluding the full array).
* `is_sorted`: Boolean validation of the result.

## Timing Definitions
Timings are measured dynamically using `time.perf_counter_ns()` and exposed as nanoseconds (`_ns`) or milliseconds (`_ms` properties):
* `checkpoint_time_ns`: Time spent sorting from the beginning until the checkpoint.
* `feature_build_ns` / `inference_ns`: Time spent inside the `RuntimePredictor` (these do not overlap with other timings).
* `execution_after_decision_ns`: The duration of post-decision sorting.
* `switch_overhead_ns`: Time spent performing setup/teardown strictly necessary to pivot algorithms (0 if continuing).
* `total_runtime_ns`: Wall-clock time for the entire `adaptive_sort()` function call.

## Metric Aggregation
* `comparisons`: The total comparisons for the entire run (`checkpoint comparisons + post-decision comparisons + overhead comparisons`).
* `data_movements`: The total data movements for the entire run (`checkpoint moves + post-decision moves + overhead moves`).
These metrics are automatically summed correctly by the underlying `runner.py` framework.

## Examples

### Example with the Real Predictor
```python
from ml.src.runtime_predictor import RuntimePredictor
from src.adaptive.adaptive_sorter import adaptive_sort

# Instantiate predictor ONCE
predictor = RuntimePredictor()

data = [99, 12, 4, 34, 10, 1]
result = adaptive_sort(
    data,
    starting_algorithm="merge_sort",
    input_type="random",
    predictor=predictor,
)

print(f"Executed: {result.executed_action}")
print(f"Total Runtime: {result.total_runtime_ms:.2f} ms")
```

### Example with a Fixed Test Predictor
Useful for forcing specific edge cases without loading the real model:
```python
class FixedPredictor:
    def __init__(self, action: str):
        self.action = action

    def predict(self, **kwargs):
        # Must return an object simulating RuntimePrediction
        class MockPrediction:
            action = self.action
            feature_build_ns = 1500
            inference_ns = 500000
        return MockPrediction()

test_predictor = FixedPredictor("switch_insertion_sort")
result = adaptive_sort([3, 1, 2], "quick_sort", "random", test_predictor)
assert result.executed_action == "switch_insertion_sort"
```

## Relationship to Future Phase 7 Benchmarking
This controller serves as the fundamental execution engine for Phase 7 benchmarking. While Phase 7.3 builds the isolated execution pathway for *a single live array*, the upcoming Phase 7 benchmark experiment will utilize this controller across large-scale datasets to compare the overall performance of the ML adaptive model against static monolithic algorithms.
