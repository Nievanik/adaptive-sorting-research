import time


# ---------------------------------------------------------------------------
# Internal helpers — used by runner.py and by resume()
# ---------------------------------------------------------------------------

def _merge_tracked(left, right):
    """
    Merge two sorted lists.
    Returns (merged_list, comparisons, moves).
    """
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

    while i < len(left):
        sorted_arr.append(left[i])
        moves += 1
        i += 1

    while j < len(right):
        sorted_arr.append(right[j])
        moves += 1
        j += 1

    return sorted_arr, comparisons, moves


def _merge_sort_tracked(arr):
    """
    Full MergeSort on arr.
    Returns (sorted_arr, comparisons, moves).
    Used internally by runner.py when switching TO merge sort.
    """
    arr = list(arr)

    if len(arr) <= 1:
        return arr, 0, 0

    mid   = len(arr) // 2
    left,  lc, lm = _merge_sort_tracked(arr[:mid])
    right, rc, rm = _merge_sort_tracked(arr[mid:])
    merged, mc, mm = _merge_tracked(left, right)

    return merged, lc + rc + mc, lm + rm + mm


# ---------------------------------------------------------------------------
# Checkpoint functions
# ---------------------------------------------------------------------------

def run_to_checkpoint(arr):
    """
    Run MergeSort to the 50% checkpoint.

    Checkpoint definition: sort only the left half (arr[:n//2]).
    State at checkpoint:
      - arr[0 : sorted_end]  → fully sorted
      - arr[sorted_end : n]  → completely untouched

    Returns a CheckpointState dict.
    """
    arr = list(arr)
    n   = len(arr)

    if n <= 1:
        return {
            "algo":           "merge_sort",
            "arr":            arr,
            "sorted_end":     n,
            "comparisons":    0,
            "moves":          0,
            "time_ms":        0.0,
            "checkpoint_pct": 100.0,
        }

    mid = n // 2

    start = time.perf_counter_ns()

    left_sorted, comparisons, moves = _merge_sort_tracked(arr[:mid])

    # Write sorted left half back into the array
    arr[:mid] = left_sorted
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


def resume(state):
    """
    Continue MergeSort from the checkpoint to completion.

    Sorts the untouched right half, then merges with the already-sorted left half.
    Returns (sorted_arr, post_comparisons, post_moves, post_time_ms).
    """
    arr = state["arr"]
    mid = state["sorted_end"]

    start = time.perf_counter_ns()

    right_sorted, rc, rm  = _merge_sort_tracked(arr[mid:])
    merged,       mc, mm  = _merge_tracked(list(arr[:mid]), right_sorted)

    time_ms = (time.perf_counter_ns() - start) / 1_000_000

    return merged, rc + mc, rm + mm, time_ms
