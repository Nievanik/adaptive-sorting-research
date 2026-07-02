import random
import time
import math


# ---------------------------------------------------------------------------
# Internal helpers — used by runner.py and by resume()
# ---------------------------------------------------------------------------

def _partition_tracked(arr, low, high):
    """
    Partition arr[low..high] around a random pivot.
    Returns (pivot_final_index, comparisons, moves).
    """
    pivot_index = random.randint(low, high)

    arr[pivot_index], arr[high] = arr[high], arr[pivot_index]
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

    arr[i + 1], arr[high] = arr[high], arr[i + 1]
    moves += 2

    return i + 1, comparisons, moves


def _sort_tracked(arr):
    """
    Full iterative QuickSort on arr.
    Returns (sorted_arr, comparisons, moves).
    Used internally by runner.py when switching TO quick sort.
    """
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


# ---------------------------------------------------------------------------
# Checkpoint functions
# ---------------------------------------------------------------------------

def run_to_checkpoint(arr):
    """
    Run QuickSort to the 50% checkpoint.

    Checkpoint definition (Option B): pause when
      comparison_count >= n * log2(n) / 2
    This approximates ~50% of the average-case total comparisons.

    State at checkpoint:
      - arr: partially partitioned (some elements in final positions)
      - remaining_stack: list of (low, high) subproblems not yet processed

    Returns a CheckpointState dict.
    """
    arr = list(arr)
    n   = len(arr)

    if n <= 1:
        return {
            "algo":            "quick_sort",
            "arr":             arr,
            "remaining_stack": [],
            "comparisons":     0,
            "moves":           0,
            "time_ms":         0.0,
            "checkpoint_pct":  100.0,
        }

    # n * log2(n) / 2 — the comparison budget for the 50% checkpoint
    budget = max(1, int(n * math.log2(n) / 2))

    comparison_count = 0
    move_count       = 0
    stack = [(0, n - 1)]

    start = time.perf_counter_ns()

    while stack and comparison_count < budget:
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

    # Actual progress as a % of n*log2(n) total expected comparisons
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


def resume(state):
    """
    Continue QuickSort from the checkpoint to completion.

    Processes all (low, high) subproblems still in remaining_stack.
    Returns (sorted_arr, post_comparisons, post_moves, post_time_ms).
    """
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

            if (p - 1 - low) > (high - (p + 1)):
                stack.append((low, p - 1))
                stack.append((p + 1, high))
            else:
                stack.append((p + 1, high))
                stack.append((low, p - 1))

    time_ms = (time.perf_counter_ns() - start) / 1_000_000

    return arr, comparison_count, move_count, time_ms
