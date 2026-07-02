import time


# ---------------------------------------------------------------------------
# Internal helper — used by runner.py when switching TO insertion sort
# ---------------------------------------------------------------------------

def _sort_tracked(arr):
    """
    Run InsertionSort on arr, returning (sorted_arr, comparisons, moves).
    Works on any subarray handed to it — used internally by runner.py.
    """
    arr = list(arr)
    comparison_count = 0
    move_count = 0

    for i in range(1, len(arr)):
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

    return arr, comparison_count, move_count


# ---------------------------------------------------------------------------
# Checkpoint functions
# ---------------------------------------------------------------------------

def run_to_checkpoint(arr):
    """
    Run InsertionSort to the 50% checkpoint.

    Checkpoint definition: outer loop i == n // 2
    State at checkpoint:
      - arr[0 : sorted_end]  → sorted relative to each other
      - arr[sorted_end : n]  → completely untouched

    Returns a CheckpointState dict.
    """
    arr = list(arr)
    n   = len(arr)

    if n <= 1:
        return {
            "algo":           "insertion_sort",
            "arr":            arr,
            "sorted_end":     n,
            "comparisons":    0,
            "moves":          0,
            "time_ms":        0.0,
            "checkpoint_pct": 100.0,
        }

    checkpoint_i     = max(1, n // 2)
    comparison_count = 0
    move_count       = 0

    start = time.perf_counter_ns()

    for i in range(1, checkpoint_i):
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

    return {
        "algo":           "insertion_sort",
        "arr":            arr,
        "sorted_end":     checkpoint_i,
        "comparisons":    comparison_count,
        "moves":          move_count,
        "time_ms":        time_ms,
        "checkpoint_pct": (checkpoint_i / n) * 100,
    }


def resume(state):
    """
    Continue InsertionSort from the checkpoint to completion.

    Picks up from state["sorted_end"] and processes the rest of the array.
    Returns (sorted_arr, post_comparisons, post_moves, post_time_ms).
    """
    arr   = list(state["arr"])
    n     = len(arr)
    start_i = state["sorted_end"]

    comparison_count = 0
    move_count       = 0

    start = time.perf_counter_ns()

    for i in range(start_i, n):
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
