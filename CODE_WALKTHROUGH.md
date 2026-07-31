# Code Walkthrough — Benchmark Execution Sequence

This document walks through every file that runs when you execute a benchmark, in the exact order they are called. For each file, we explain **what it does**, **why it's written that way**, and **how the code works block by block**.

---

## Execution Chain Overview

When you run:
```bash
python main.py --algo quick_sort --size 1000
```

The files execute in this order:

```
main.py                                  ← 1. Entry point: parse CLI args
  └─► benchmark.py                       ← 2. Orchestrator: loop over algos × data types
        ├─► src/data_loader.py           ← 3. Load arrays from JSON files in data/
        └─► src/checkpoint/runner.py     ← 4. Unified API: checkpoint → continue / switch
              ├─► insertion_sort_checkpoint.py  ← 5a. IS-specific checkpoint logic
              ├─► merge_sort_checkpoint.py      ← 5b. MS-specific checkpoint logic
              └─► quick_sort_checkpoint.py      ← 5c. QS-specific checkpoint logic
```

---

## File 1: `main.py` — The Entry Point

**Purpose**: Parse command-line arguments and kick off the benchmark.

```python
import argparse
from benchmark import run_benchmark, ALGO_REGISTRY
```

- `argparse` is Python's built-in CLI argument parser.
- We import `run_benchmark` (the function that does all the work) and `ALGO_REGISTRY` (a dict of valid algorithm names) from `benchmark.py`.

```python
if __name__ == "__main__":
```

- This guard ensures the code below only runs when you execute `main.py` directly (not when it's imported by another file).

```python
    parser = argparse.ArgumentParser(
        description="Run adaptive sorting benchmarks with full metrics."
    )

    parser.add_argument(
        "--algo",
        nargs="+",
        required=True,
        help=f"Algorithms to benchmark: {' '.join(ALGO_REGISTRY.keys())}"
    )

    parser.add_argument(
        "--size",
        nargs="+",
        type=int,
        required=True,
        help="Dataset sizes to run (e.g. 100 500 1000 5000 10000)"
    )
```

- `--algo` accepts one or more algorithm names (e.g., `quick_sort merge_sort`). `nargs="+"` means "one or more values".
- `--size` accepts one or more integer sizes. `type=int` auto-converts the string arguments to integers.

```python
    args = parser.parse_args()

    valid   = [a for a in args.algo if a in ALGO_REGISTRY]
    invalid = [a for a in args.algo if a not in ALGO_REGISTRY]

    for name in invalid:
        print(f"⚠️  Unknown algorithm: {name} — skipping")

    if not valid:
        print("No valid algorithms selected. Exiting.")
        exit(1)
```

- We separate user-provided algo names into `valid` and `invalid` lists by checking against `ALGO_REGISTRY`.
- If there are no valid algorithms, the program exits with code 1 (error).

```python
    for size in args.size:
        run_benchmark(size=size, algo_names=valid, save=True)
```

- For each requested size, we call `run_benchmark()` from `benchmark.py` with the valid algorithm names and `save=True` (which writes results to JSON files).

**Why it's written this way**: `main.py` is intentionally thin — it only handles CLI parsing and validation. All the actual logic lives in `benchmark.py`, keeping concerns separated and making the benchmark engine importable from other scripts if needed.

---

## File 2: `benchmark.py` — The Orchestrator

**Purpose**: For each algorithm × data type combination, run the checkpoint pipeline (checkpoint → continue → switch to A → switch to B), collect metrics, print results, and save to JSON.

### Imports & Constants

```python
import json
from pathlib import Path

from src.data_loader import load_dataset
from src.checkpoint.runner import run_to_checkpoint, continue_sort, switch_sort
```

- `json` — for saving results as JSON files.
- `Path` — for OS-independent file path manipulation.
- `load_dataset` — loads a pre-generated array from `data/`.
- `run_to_checkpoint`, `continue_sort`, `switch_sort` — the three core checkpoint operations from `runner.py`.

```python
DATA_TYPES = [
    "random", "sorted", "reverse_sorted", "nearly_sorted",
    "duplicate_heavy", "all_equal", "edge_cases",
]
```

- The 7 input distribution categories we test against. Each one has pre-generated JSON files in `data/`.

```python
ALGO_REGISTRY = {
    "insertion_sort": None,
    "merge_sort":     None,
    "quick_sort":     None,
}
```

- A dictionary of valid algorithm names. The values are `None` because we only use the keys for validation. This dict is also imported by `main.py` to validate user input.

### The Core Runner Function

```python
def run_sort_with_checkpoint(algo_name, arr):
```

- This is the **single most important function** in the entire benchmark. It takes an algorithm name and an array, and produces a complete comparison of: continuing vs. switching to each alternative.

```python
    state = run_to_checkpoint(algo_name, arr)
```

- **Step 1**: Run the chosen algorithm to its 50% checkpoint. This returns a `state` dictionary containing:
  - `"algo"`: which algorithm was used (e.g., `"quick_sort"`)
  - `"arr"`: the array in its partially-sorted state
  - `"comparisons"`, `"moves"`, `"time_ms"`: metrics accumulated so far
  - `"checkpoint_pct"`: actual progress percentage
  - Plus algorithm-specific fields: `"sorted_end"` (for IS/MS) or `"remaining_stack"` (for QS)

```python
    cont_res = continue_sort(state)
```

- **Step 2**: Continue with the **same algorithm** from the checkpoint to completion. Returns a result dict with post-checkpoint metrics and totals.

```python
    other_algos = [name for name in ALGO_REGISTRY if name != algo_name]
    switch_res_a = switch_sort(state, other_algos[0])
    switch_res_b = switch_sort(state, other_algos[1])
```

- **Step 3**: Identify the two alternative algorithms and switch to each of them from the same checkpoint state. For example, if we started with `quick_sort`, the alternatives are `insertion_sort` and `merge_sort`.

> **Critical detail**: Both `switch_sort` calls use the **same** `state` dict from the checkpoint. The checkpoint is run once, and then three separate paths branch from it (continue, switch A, switch B). This ensures a fair comparison — all three paths start from the exact same partially-sorted array.

```python
    return {
        "checkpoint": {
            "checkpoint_pct": state["checkpoint_pct"],
            "time_ms":        state["time_ms"],
            "comparisons":    state["comparisons"],
            "moves":          state["moves"]
        },
        "continue": {
            "time_ms":      cont_res["total_time_ms"],
            "comparisons":  cont_res["total_comparisons"],
            "moves":        cont_res["total_moves"],
            "overhead":     cont_res["overhead"]
        },
        f"switch_{other_algos[0]}": {
            "time_ms":      switch_res_a["total_time_ms"],
            "comparisons":  switch_res_a["total_comparisons"],
            "moves":        switch_res_a["total_moves"],
            "overhead":     switch_res_a["overhead"]
        },
        f"switch_{other_algos[1]}": {
            "time_ms":      switch_res_b["total_time_ms"],
            "comparisons":  switch_res_b["total_comparisons"],
            "moves":        switch_res_b["total_moves"],
            "overhead":     switch_res_b["overhead"]
        }
    }
```

- The return value is a nested dictionary with four keys: `"checkpoint"`, `"continue"`, `"switch_<algo_a>"`, and `"switch_<algo_b>"`. Each contains the total metrics for that execution path.

**Why it's structured this way**: By recording all four paths in a single dict, we can later compare them side-by-side to build the **Switching Cost Matrix** — the ground truth for ML training labels. We can answer: "Starting from QuickSort at 50%, is it cheaper to continue or switch to MergeSort?"

### The Benchmark Loop

```python
def run_benchmark(size, algo_names, save=True):
```

- Called by `main.py` for each requested size. Loops over algorithms and data types.

```python
    for algo_name in algo_names:
        other_algos = [name for name in ALGO_REGISTRY if name != algo_name]
        results = []

        for dtype in DATA_TYPES:
            loaded = load_dataset(dtype, size)
```

- For each algorithm, we iterate over all 7 data types and load the corresponding dataset from `data/`.

```python
            if dtype == "edge_cases":
                # ... handle edge cases (empty array, single element)
                # Each edge case file is processed individually
            else:
                metrics = run_sort_with_checkpoint(algo_name, loaded)
                results.append({
                    "algorithm": algo_name,
                    "type":      dtype,
                    "size":      size,
                    "checkpoint": metrics["checkpoint"],
                    "continue":   metrics["continue"],
                    ...
                })
```

- **Edge cases** are special: the `data/edge_cases/` directory contains multiple small files (`empty.json`, `single_element.json`), so we loop through each one individually.
- **Normal cases** are straightforward: load one array, run the full checkpoint pipeline, and append the results.

```python
        if save:
            save_results(algo_name, size, results)
```

- After processing all data types for one algorithm at one size, save the results list to `results/<algo_name>/<size>.json`.

### The Save Function

```python
def save_results(algo_name, size, results):
    base_dir  = Path("results") / algo_name
    base_dir.mkdir(parents=True, exist_ok=True)
    file_path = base_dir / f"{size}.json"

    with open(file_path, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\n📁 Saved: {file_path}\n")
```

- Creates the directory structure `results/quick_sort/` if it doesn't exist.
- Writes the results list as a pretty-printed JSON file (e.g., `results/quick_sort/1000.json`).

---

## File 3: `src/data_loader.py` — Loading Pre-Generated Datasets

**Purpose**: Read dataset JSON files from `data/` and return them as Python lists.

```python
import json
from pathlib import Path

DATA_ROOT = Path(__file__).resolve().parents[1] / "data"
```

- `Path(__file__).resolve()` gives the absolute path to `data_loader.py` itself.
- `.parents[1]` goes up two directories: from `src/data_loader.py` → `src/` → project root.
- Then we append `/ "data"` to get the absolute path to the `data/` directory.

**Why not just `Path("data")`?** A relative path like `Path("data")` only works if you run the script from the project root. By resolving relative to `__file__`, the loader works regardless of which directory you run from. This was a bug we fixed early on.

```python
def load_dataset(data_type, size):
    if data_type == "edge_cases":
        edge_dir = DATA_ROOT / "edge_cases"
        datasets = []
        for file in sorted(edge_dir.glob("*.json")):
            with open(file, "r") as f:
                datasets.append(json.load(f))
        return datasets
```

- For `edge_cases`, we load all JSON files in `data/edge_cases/` and return them as a list of arrays.
- `sorted(edge_dir.glob("*.json"))` ensures consistent ordering (`empty.json` before `single_element.json`).

```python
    file_path = DATA_ROOT / data_type / f"{data_type}_{size}.json"
    with open(file_path, "r") as f:
        return json.load(f)
```

- For normal data types, the file path follows the pattern: `data/random/random_1000.json`.
- `json.load()` parses the JSON file and returns a Python list of integers.

---

## File 4: `src/checkpoint/runner.py` — The Unified Checkpoint API

**Purpose**: Provide a single, clean interface for the benchmark to interact with all three checkpoint implementations. This is the bridge between `benchmark.py` and the algorithm-specific checkpoint modules.

### Imports & Module Registry

```python
import time

from src.checkpoint import insertion_sort_checkpoint as is_cp
from src.checkpoint import merge_sort_checkpoint     as ms_cp
from src.checkpoint import quick_sort_checkpoint     as qs_cp

CHECKPOINT_MODULES = {
    "insertion_sort": is_cp,
    "merge_sort":     ms_cp,
    "quick_sort":     qs_cp,
}
```

- We import all three checkpoint modules and register them in a dictionary keyed by algorithm name.
- This allows us to look up the correct module dynamically: `CHECKPOINT_MODULES["quick_sort"]` gives us `qs_cp`.

**Why use a registry dict?** It avoids ugly `if/elif/else` chains. Adding a new algorithm is as simple as adding one entry to the dict and writing a new checkpoint module.

### Public API — `run_to_checkpoint()`

```python
def run_to_checkpoint(algo_name, arr):
    module = CHECKPOINT_MODULES[algo_name]
    return module.run_to_checkpoint(arr)
```

- Looks up the correct module and delegates to its `run_to_checkpoint()` function.
- Returns a state dictionary (the checkpoint snapshot).

### Public API — `continue_sort()`

```python
def continue_sort(state):
    module = CHECKPOINT_MODULES[state["algo"]]
    sorted_arr, post_cmp, post_mv, post_time = module.resume(state)

    return _build_result(
        sorted_arr=sorted_arr, state=state,
        post_cmp=post_cmp, post_mv=post_mv, post_time=post_time,
        switched=False, to_algo=state["algo"],
    )
```

- Reads `state["algo"]` to know which algorithm created this checkpoint.
- Calls that algorithm's `resume()` function, which picks up where the checkpoint left off.
- Wraps the raw results into a standardized result dict via `_build_result()`.

### Public API — `switch_sort()`

```python
def switch_sort(state, new_algo_name):
    if state["algo"] in ("insertion_sort", "merge_sort"):
        sorted_arr, post_cmp, post_mv, post_time, oh_cmp, oh_mv, oh_time = \
            _switch_from_split_state(state, new_algo_name)
    else:
        sorted_arr, post_cmp, post_mv, post_time, oh_cmp, oh_mv, oh_time = \
            _switch_from_qs_state(state, new_algo_name)

    return _build_result(
        sorted_arr=sorted_arr, state=state,
        post_cmp=post_cmp, post_mv=post_mv, post_time=post_time,
        switched=True, to_algo=new_algo_name,
        oh_cmp=oh_cmp, oh_mv=oh_mv, oh_time=oh_time,
    )
```

- **This is where the switching decision branches based on checkpoint type.**
- InsertionSort and MergeSort both produce a **split-state** checkpoint (sorted left half + untouched right half), so they share the same switching handler `_switch_from_split_state()`.
- QuickSort produces a **stack-state** checkpoint (partially partitioned array + remaining subproblem stack), so it has its own handler `_switch_from_qs_state()`.

### Internal — `_switch_from_split_state()`

This handles switching away from InsertionSort or MergeSort checkpoints.

```python
def _switch_from_split_state(state, new_algo_name):
    arr = state["arr"]
    mid = state["sorted_end"]

    left_sorted    = list(arr[:mid])    # already sorted at checkpoint
    right_unsorted = list(arr[mid:])
```

- Split the array at `sorted_end`. The left part is already sorted. The right part is completely untouched.

```python
    # Step 1 — sort right half with chosen algorithm
    sort_start = time.perf_counter_ns()
    right_sorted, rc, rm = _sort_with(new_algo_name, right_unsorted)
    sort_time_ns = time.perf_counter_ns() - sort_start
    cmp_total += rc
    mv_total  += rm
```

- Sort the untouched right half using the target algorithm's tracked sort function.
- We time this separately (`sort_time_ns`) so we can isolate pure sorting time from overhead.

```python
    # Step 2 — merge sorted halves
    merge_start = time.perf_counter_ns()
    merged, mc, mm = ms_cp._merge_tracked(left_sorted, right_sorted)
    merge_time_ns = time.perf_counter_ns() - merge_start
    cmp_total += mc
    mv_total  += mm
```

- Merge the two sorted halves using MergeSort's tracked merge function.
- The merge comparisons (`mc`) and merge moves (`mm`) are the **overhead** — this work wouldn't exist if we hadn't switched.

```python
    time_ms = (time.perf_counter_ns() - start) / 1_000_000

    # Overhead is setup time (slicing lists, constructor) + merge time
    overhead_time = time_ms - (sort_time_ns / 1_000_000)

    return merged, cmp_total, mv_total, time_ms, mc, mm, max(0.0, overhead_time)
```

- **Overhead time** = total wall time minus pure sorting time. This captures the cost of list slicing, constructors, and the merge step.
- `max(0.0, ...)` prevents negative values from floating-point imprecision.

### Internal — `_switch_from_qs_state()`

This handles switching away from QuickSort checkpoints.

```python
def _switch_from_qs_state(state, new_algo_name):
    arr   = list(state["arr"])
    stack = state["remaining_stack"]
```

- QuickSort's state contains a list of `(low, high)` tuples — the subproblems that still need sorting.
- Elements NOT covered by any stack entry are already in their final sorted positions (they were placed there by previous partitions).

```python
    for (low, high) in stack:
        if low >= high:
            continue

        subarray = arr[low : high + 1]

        sort_start = time.perf_counter_ns()
        sorted_sub, rc, rm = _sort_with(new_algo_name, subarray)
        pure_sort_time_ns += (time.perf_counter_ns() - sort_start)

        arr[low : high + 1] = sorted_sub
        cmp_total += rc
        mv_total  += rm
```

- For each unresolved subproblem, we:
  1. Extract the subarray.
  2. Sort it fully with the target algorithm.
  3. Write the sorted result back into the original array at the correct position.
- We accumulate pure sorting time separately from total time to calculate overhead.

```python
    return arr, cmp_total, mv_total, time_ms, 0, 0, overhead_time
```

- Overhead comparisons and moves are `0` because there's no merge step — the subarrays are independent and go back in-place.
- Overhead time captures the cost of slicing subarrays out and writing them back.

### Internal — `_sort_with()`

```python
def _sort_with(algo_name, arr):
    if algo_name == "insertion_sort":
        return is_cp._sort_tracked(arr)
    elif algo_name == "merge_sort":
        return ms_cp._merge_sort_tracked(arr)
    elif algo_name == "quick_sort":
        return qs_cp._sort_tracked(arr)
    else:
        raise ValueError(f"Unknown algorithm: {algo_name}")
```

- A dispatcher that calls the correct algorithm's full-sort function.
- Each checkpoint module exposes a `_sort_tracked()` or `_merge_sort_tracked()` function that sorts an array and returns `(sorted_arr, comparisons, moves)`.

**Why are these `_` prefixed?** The underscore signals they're "internal" functions — not meant to be called directly by the benchmark. They're helpers used by the runner during switching.

### Internal — `_build_result()`

```python
def _build_result(sorted_arr, state, post_cmp, post_mv, post_time,
                  switched, to_algo, oh_cmp=0, oh_mv=0, oh_time=0.0):
    return {
        "sorted_arr":        sorted_arr,
        "post_comparisons":  post_cmp,
        "post_moves":        post_mv,
        "post_time_ms":      post_time,
        "total_comparisons": state["comparisons"] + post_cmp,
        "total_moves":       state["moves"]        + post_mv,
        "total_time_ms":     state["time_ms"]      + post_time,
        "switched":          switched,
        "from_algo":         state["algo"],
        "to_algo":           to_algo,
        "checkpoint_pct":    state["checkpoint_pct"],
        "overhead": {
            "comparisons":   oh_cmp,
            "moves":         oh_mv,
            "time_ms":       oh_time
        }
    }
```

- Standardizes the output format so `benchmark.py` doesn't need to know which algorithm or path was taken.
- **Totals are computed here**: `total_comparisons = checkpoint_comparisons + post_checkpoint_comparisons`. This is the end-to-end cost of the entire sort (checkpoint phase + finishing phase).

---

## File 5a: `src/checkpoint/insertion_sort_checkpoint.py`

**Purpose**: Implement InsertionSort's checkpoint, resume, and tracked-sort functions.

### `_sort_tracked()` — Full InsertionSort with Metrics

```python
def _sort_tracked(arr):
    arr = list(arr)
    comparison_count = 0
    move_count = 0

    for i in range(1, len(arr)):
        key = arr[i]
        j   = i - 1

        while j >= 0:
            comparison_count += 1
            if arr[j] > key:
                arr[j + 1] = arr[j]    # shift element right
                move_count += 1
                j -= 1
            else:
                break

        arr[j + 1] = key               # place key in correct position
        move_count += 1

    return arr, comparison_count, move_count
```

- Standard InsertionSort: for each element, shift larger elements right until the correct position is found, then place the element.
- `list(arr)` creates a copy so we don't mutate the caller's array.
- Every comparison (`arr[j] > key`) increments `comparison_count`.
- Every element write (shift or final placement) increments `move_count`.

**Why count the final `arr[j + 1] = key` as a move?** Even though the key might be placed back where it started, it's still a write operation. We count all writes consistently.

### `run_to_checkpoint()` — Pause at 50%

```python
def run_to_checkpoint(arr):
    arr = list(arr)
    n   = len(arr)

    if n <= 1:
        return { ... checkpoint_pct: 100.0 ... }  # trivial case, already sorted

    checkpoint_i     = max(1, n // 2)
    comparison_count = 0
    move_count       = 0

    start = time.perf_counter_ns()

    for i in range(1, checkpoint_i):      # ← STOPS at n//2 instead of n
        key = arr[i]
        j   = i - 1
        while j >= 0:
            comparison_count += 1
            if arr[j] > key:
                arr[j + 1] = arr[j]
                move_count += 1
                j -= 1
            else:
                break
        arr[j + 1] = key
        move_count += 1

    time_ms = (time.perf_counter_ns() - start) / 1_000_000
```

- **The key difference from the full sort**: the loop is `range(1, checkpoint_i)` instead of `range(1, len(arr))`.
- After this loop, `arr[0 : checkpoint_i]` is sorted relative to itself, and `arr[checkpoint_i : n]` is completely untouched.

```python
    return {
        "algo":           "insertion_sort",
        "arr":            arr,
        "sorted_end":     checkpoint_i,
        "comparisons":    comparison_count,
        "moves":          move_count,
        "time_ms":        time_ms,
        "checkpoint_pct": (checkpoint_i / n) * 100,
    }
```

- `sorted_end` tells the resume function where to pick up.

### `resume()` — Continue from Checkpoint

```python
def resume(state):
    arr     = list(state["arr"])
    n       = len(arr)
    start_i = state["sorted_end"]

    comparison_count = 0
    move_count       = 0

    start = time.perf_counter_ns()

    for i in range(start_i, n):           # ← STARTS from where checkpoint stopped
        key = arr[i]
        j   = i - 1
        while j >= 0:
            comparison_count += 1
            if arr[j] > key:
                arr[j + 1] = arr[j]
                move_count += 1
                j -= 1
            else:
                break
        arr[j + 1] = key
        move_count += 1

    time_ms = (time.perf_counter_ns() - start) / 1_000_000
    return arr, comparison_count, move_count, time_ms
```

- Uses the same InsertionSort logic but starts the loop from `start_i` (the checkpoint position) instead of 1.
- Returns **only the post-checkpoint metrics** (not totals). The runner adds these to the checkpoint metrics to get totals.

---

## File 5b: `src/checkpoint/merge_sort_checkpoint.py`

**Purpose**: Implement MergeSort's checkpoint, resume, and tracked-sort functions.

### `_merge_tracked()` — Tracked Merge of Two Sorted Lists

```python
def _merge_tracked(left, right):
    sorted_arr = []
    i = j = 0
    comparisons = 0
    moves       = 0

    while i < len(left) and j < len(right):
        comparisons += 1
        if left[i] <= right[j]:
            sorted_arr.append(left[i])
            moves += 1
            i += 1
        else:
            sorted_arr.append(right[j])
            moves += 1
            j += 1

    # Remaining elements (no comparisons needed — one list is exhausted)
    while i < len(left):
        sorted_arr.append(left[i])
        moves += 1
        i += 1
    while j < len(right):
        sorted_arr.append(right[j])
        moves += 1
        j += 1

    return sorted_arr, comparisons, moves
```

- Standard two-pointer merge: compare the front elements of `left` and `right`, take the smaller one.
- Each `append` is a move (element write).
- The trailing `while` loops flush remaining elements — no comparisons needed because the other list is empty.

**Why is this function important?** It's used in three places:
1. MergeSort's own recursive sort (`_merge_sort_tracked`).
2. MergeSort's `resume()` to merge the two halves.
3. The runner's `_switch_from_split_state()` to merge sorted halves during a switch.

### `_merge_sort_tracked()` — Full Recursive MergeSort

```python
def _merge_sort_tracked(arr):
    arr = list(arr)
    if len(arr) <= 1:
        return arr, 0, 0

    mid   = len(arr) // 2
    left,  lc, lm = _merge_sort_tracked(arr[:mid])
    right, rc, rm = _merge_sort_tracked(arr[mid:])
    merged, mc, mm = _merge_tracked(left, right)

    return merged, lc + rc + mc, lm + rm + mm
```

- Standard recursive MergeSort: split → sort left → sort right → merge.
- Comparisons and moves are summed across all recursive calls.

### `run_to_checkpoint()` — Sort Only the Left Half

```python
def run_to_checkpoint(arr):
    arr = list(arr)
    n   = len(arr)

    if n <= 1:
        return { ... checkpoint_pct: 100.0 ... }

    mid = n // 2

    start = time.perf_counter_ns()

    left_sorted, comparisons, moves = _merge_sort_tracked(arr[:mid])

    arr[:mid] = left_sorted           # write sorted left half back
    # arr[mid:] intentionally left untouched

    time_ms = (time.perf_counter_ns() - start) / 1_000_000

    return {
        "algo":           "merge_sort",
        "arr":            arr,
        "sorted_end":     mid,
        "comparisons":    comparisons,
        "moves":          moves,
        "time_ms":        time_ms,
        "checkpoint_pct": 50.0,
    }
```

- **The trick**: We call `_merge_sort_tracked()` on only `arr[:mid]` (the left half) and then return immediately.
- The right half `arr[mid:]` is never touched — it stays in its original order.
- `checkpoint_pct` is always exactly `50.0` because MergeSort's checkpoint is defined structurally (half the array), not by comparison counting.

### `resume()` — Sort Right Half and Merge

```python
def resume(state):
    arr = state["arr"]
    mid = state["sorted_end"]

    start = time.perf_counter_ns()

    right_sorted, rc, rm  = _merge_sort_tracked(arr[mid:])
    merged,       mc, mm  = _merge_tracked(list(arr[:mid]), right_sorted)

    time_ms = (time.perf_counter_ns() - start) / 1_000_000
    return merged, rc + mc, rm + mm, time_ms
```

- **Step 1**: Recursively sort the right half.
- **Step 2**: Merge the already-sorted left half with the newly-sorted right half.
- Returns combined post-checkpoint metrics.

---

## File 5c: `src/checkpoint/quick_sort_checkpoint.py`

**Purpose**: Implement QuickSort's checkpoint, resume, and tracked-sort functions. This is the most complex checkpoint because QuickSort's execution state is fundamentally different from InsertionSort and MergeSort.

### `_partition_tracked()` — Lomuto Partition with Random Pivot

```python
def _partition_tracked(arr, low, high):
    pivot_index = random.randint(low, high)

    arr[pivot_index], arr[high] = arr[high], arr[pivot_index]   # swap pivot to end
    moves = 2

    pivot = arr[high]
    i     = low - 1
    comparisons = 0

    for j in range(low, high):
        comparisons += 1
        if arr[j] <= pivot:
            i += 1
            arr[i], arr[j] = arr[j], arr[i]
            moves += 2

    arr[i + 1], arr[high] = arr[high], arr[i + 1]              # place pivot in final position
    moves += 2

    return i + 1, comparisons, moves
```

- **Random pivot**: `random.randint(low, high)` picks a random element as pivot, protecting against worst-case on sorted/reverse-sorted input.
- **Lomuto partition scheme**: Elements ≤ pivot go to the left side, elements > pivot stay on the right.
- After partitioning, `arr[i + 1]` is the pivot's final sorted position. Everything left of it is ≤ pivot, everything right is > pivot.
- Each swap is 2 moves (2 element writes).

### `_sort_tracked()` — Full Iterative QuickSort

```python
def _sort_tracked(arr):
    arr = list(arr)
    n   = len(arr)
    if n <= 1:
        return arr, 0, 0

    comparison_count = 0
    move_count       = 0
    stack = [(0, n - 1)]

    while stack:
        low, high = stack.pop()
        if low < high:
            p, c, m = _partition_tracked(arr, low, high)
            comparison_count += c
            move_count       += m

            if (p - 1 - low) > (high - (p + 1)):
                stack.append((low, p - 1))
                stack.append((p + 1, high))
            else:
                stack.append((p + 1, high))
                stack.append((low, p - 1))

    return arr, comparison_count, move_count
```

- **Iterative, not recursive**: Uses an explicit stack of `(low, high)` subproblem ranges instead of recursion.
- The `if/else` block pushes the **larger** subproblem first so the **smaller** one is processed first (popped next). This is a standard optimization to reduce peak stack depth.

**Why iterative?** Recursive QuickSort stores its state in the call stack, which you can't inspect or serialize. By using an explicit stack, we can snapshot the remaining work at any point — which is exactly what the checkpoint needs.

### `run_to_checkpoint()` — Partition Until Budget Exhausted

```python
def run_to_checkpoint(arr):
    arr = list(arr)
    n   = len(arr)

    if n <= 1:
        return { ... checkpoint_pct: 100.0 ... }

    budget = max(1, int(n * math.log2(n) / 2))
```

- **Comparison budget**: `n * log₂(n) / 2` — half of the average-case total comparisons for QuickSort.
- `max(1, ...)` ensures the budget is at least 1 for tiny arrays.

```python
    comparison_count = 0
    move_count       = 0
    stack = [(0, n - 1)]

    start = time.perf_counter_ns()

    while stack and comparison_count < budget:       # ← STOPS when budget is spent
        low, high = stack.pop()
        if low < high:
            p, c, m = _partition_tracked(arr, low, high)
            comparison_count += c
            move_count       += m

            if (p - 1 - low) > (high - (p + 1)):
                stack.append((low, p - 1))
                stack.append((p + 1, high))
            else:
                stack.append((p + 1, high))
                stack.append((low, p - 1))

    time_ms = (time.perf_counter_ns() - start) / 1_000_000
```

- The loop runs normally until `comparison_count >= budget`, then it stops.
- Any subproblems still on the stack are **unfinished work**.

```python
    actual_pct = min(100.0, (comparison_count / (n * math.log2(n))) * 100)

    return {
        "algo":            "quick_sort",
        "arr":             arr,
        "remaining_stack": list(stack),
        "comparisons":     comparison_count,
        "moves":           move_count,
        "time_ms":         time_ms,
        "checkpoint_pct":  actual_pct,
    }
```

- `remaining_stack` is the key difference from IS/MS checkpoints. Instead of `sorted_end`, we have a list of `(low, high)` tuples representing unfinished subproblems.
- `checkpoint_pct` is calculated as the fraction of the total expected comparisons, capped at 100%.

**Why is QuickSort's checkpoint different?** QuickSort doesn't process elements left-to-right. It partitions recursively, placing pivot elements in their final positions while leaving subarrays unsorted. At any checkpoint, some elements are in their final positions and some aren't — and the "unsorted" elements are scattered across multiple non-contiguous ranges. That's why we need a stack of ranges instead of a single split point.

### `resume()` — Finish All Remaining Subproblems

```python
def resume(state):
    arr   = list(state["arr"])
    stack = list(state["remaining_stack"])

    comparison_count = 0
    move_count       = 0

    start = time.perf_counter_ns()

    while stack:
        low, high = stack.pop()
        if low < high:
            p, c, m = _partition_tracked(arr, low, high)
            comparison_count += c
            move_count       += m
            # ... push new subproblems ...

    time_ms = (time.perf_counter_ns() - start) / 1_000_000
    return arr, comparison_count, move_count, time_ms
```

- Identical to the checkpoint loop but without the budget check — it runs until the stack is empty and everything is sorted.

---

## Summary: The Complete Call Chain

Here's every function call that happens when you run `python main.py --algo quick_sort --size 1000`:

```
main.py
  └─ argparse parses --algo quick_sort --size 1000
  └─ run_benchmark(size=1000, algo_names=["quick_sort"], save=True)
       │
       ├── for dtype in ["random", "sorted", "reverse_sorted", ...]:
       │     │
       │     ├── load_dataset("random", 1000)                    # data_loader.py
       │     │     └── reads data/random/random_1000.json
       │     │
       │     └── run_sort_with_checkpoint("quick_sort", arr)     # benchmark.py
       │           │
       │           ├── run_to_checkpoint("quick_sort", arr)      # runner.py → qs_checkpoint.py
       │           │     └── partitions until comparison budget exhausted
       │           │     └── returns state dict with remaining_stack
       │           │
       │           ├── continue_sort(state)                      # runner.py → qs_checkpoint.py
       │           │     └── resume(state)
       │           │     └── finishes all remaining subproblems
       │           │
       │           ├── switch_sort(state, "insertion_sort")      # runner.py
       │           │     └── _switch_from_qs_state(state, "insertion_sort")
       │           │           └── for each (low, high) in remaining_stack:
       │           │                 └── _sort_with("insertion_sort", subarray)
       │           │                       └── is_cp._sort_tracked(subarray)
       │           │
       │           └── switch_sort(state, "merge_sort")          # runner.py
       │                 └── _switch_from_qs_state(state, "merge_sort")
       │                       └── for each (low, high) in remaining_stack:
       │                             └── _sort_with("merge_sort", subarray)
       │                                   └── ms_cp._merge_sort_tracked(subarray)
       │
       └── save_results("quick_sort", 1000, results)
             └── writes results/quick_sort/1000.json
```
