"""
Unified checkpoint runner — the single interface used by Phase 5 (Switching Cost Matrix)
and Phase 6 (Feature Extraction).

Usage:
    state  = run_to_checkpoint("quick_sort", arr)
    result = continue_sort(state)           # stay with QuickSort
    result = switch_sort(state, "merge_sort") # switch to MergeSort at 50%

Each result dict contains:
    sorted_arr, post_comparisons, post_moves, post_time_ms,
    total_comparisons, total_moves, total_time_ms,
    switched, from_algo, to_algo
"""

import time

from src.checkpoint import insertion_sort_checkpoint as is_cp
from src.checkpoint import merge_sort_checkpoint     as ms_cp
from src.checkpoint import quick_sort_checkpoint     as qs_cp

CHECKPOINT_MODULES = {
    "insertion_sort": is_cp,
    "merge_sort":     ms_cp,
    "quick_sort":     qs_cp,
}


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def run_to_checkpoint(algo_name, arr):
    """
    Run algo_name to its 50% checkpoint.
    Returns a CheckpointState dict with array snapshot and metrics so far.
    """
    module = CHECKPOINT_MODULES[algo_name]
    return module.run_to_checkpoint(arr)


def continue_sort(state):
    """
    Continue with the same algorithm from the checkpoint.
    Returns a result dict with post-checkpoint and total metrics.
    """
    module = CHECKPOINT_MODULES[state["algo"]]
    sorted_arr, post_cmp, post_mv, post_time = module.resume(state)

    return _build_result(
        sorted_arr  = sorted_arr,
        state       = state,
        post_cmp    = post_cmp,
        post_mv     = post_mv,
        post_time   = post_time,
        switched    = False,
        to_algo     = state["algo"],
    )


def switch_sort(state, new_algo_name):
    """
    Switch to a different algorithm at the checkpoint.

    For InsertionSort / MergeSort checkpoints:
        sorted left half stays, new algo sorts the right half, then merge.

    For QuickSort checkpoint:
        new algo independently sorts each unresolved subarray region
        (each entry in remaining_stack), leaving already-partitioned
        elements in their correct positions.

    Returns a result dict with post-checkpoint and total metrics.
    """
    if state["algo"] in ("insertion_sort", "merge_sort"):
        sorted_arr, post_cmp, post_mv, post_time, oh_cmp, oh_mv, oh_time = _switch_from_split_state(
            state, new_algo_name
        )
    else:
        sorted_arr, post_cmp, post_mv, post_time, oh_cmp, oh_mv, oh_time = _switch_from_qs_state(
            state, new_algo_name
        )

    return _build_result(
        sorted_arr = sorted_arr,
        state      = state,
        post_cmp   = post_cmp,
        post_mv    = post_mv,
        post_time  = post_time,
        switched   = True,
        to_algo    = new_algo_name,
        oh_cmp     = oh_cmp,
        oh_mv      = oh_mv,
        oh_time    = oh_time,
    )


# ---------------------------------------------------------------------------
# Internal switch handlers
# ---------------------------------------------------------------------------

def _switch_from_split_state(state, new_algo_name):
    """
    Switch handler for IS and MS checkpoints.

    Both produce a clean split: arr[0:sorted_end] sorted, arr[sorted_end:] untouched.
    Steps:
      1. Sort the right half with the new algorithm.
      2. Merge the sorted left half + sorted right half.
    """
    arr = state["arr"]
    mid = state["sorted_end"]

    left_sorted   = list(arr[:mid])   # already sorted at checkpoint
    right_unsorted = list(arr[mid:])

    cmp_total = 0
    mv_total  = 0

    start = time.perf_counter_ns()

    # Step 1 — sort right half with chosen algorithm
    sort_start = time.perf_counter_ns()
    right_sorted, rc, rm = _sort_with(new_algo_name, right_unsorted)
    sort_time_ns = time.perf_counter_ns() - sort_start
    cmp_total += rc
    mv_total  += rm

    # Step 2 — merge sorted halves
    merge_start = time.perf_counter_ns()
    merged, mc, mm = ms_cp._merge_tracked(left_sorted, right_sorted)
    merge_time_ns = time.perf_counter_ns() - merge_start
    cmp_total += mc
    mv_total  += mm

    time_ms = (time.perf_counter_ns() - start) / 1_000_000
    
    # Overhead is setup time (slicing lists, constructor) + merge time
    overhead_time = time_ms - (sort_time_ns / 1_000_000)

    return merged, cmp_total, mv_total, time_ms, mc, mm, max(0.0, overhead_time)


def _switch_from_qs_state(state, new_algo_name):
    """
    Switch handler for QS checkpoint.

    The remaining_stack contains independent (low, high) subproblems.
    Each is sorted in-place with the new algorithm.
    Elements NOT in any remaining subproblem are already in their final positions.
    """
    arr   = list(state["arr"])
    stack = state["remaining_stack"]

    cmp_total = 0
    mv_total  = 0
    pure_sort_time_ns = 0

    start = time.perf_counter_ns()

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

    time_ms = (time.perf_counter_ns() - start) / 1_000_000
    sort_time_ms = pure_sort_time_ns / 1_000_000
    overhead_time = max(0.0, time_ms - sort_time_ms)

    return arr, cmp_total, mv_total, time_ms, 0, 0, overhead_time


# ---------------------------------------------------------------------------
# Helper: dispatch sort to the right checkpoint module
# ---------------------------------------------------------------------------

def _sort_with(algo_name, arr):
    """
    Sort arr fully with algo_name using the tracked helper functions.
    Returns (sorted_arr, comparisons, moves).
    """
    if algo_name == "insertion_sort":
        return is_cp._sort_tracked(arr)
    elif algo_name == "merge_sort":
        return ms_cp._merge_sort_tracked(arr)
    elif algo_name == "quick_sort":
        return qs_cp._sort_tracked(arr)
    else:
        raise ValueError(f"Unknown algorithm: {algo_name}")


# ---------------------------------------------------------------------------
# Helper: build a consistent result dict
# ---------------------------------------------------------------------------

def _build_result(sorted_arr, state, post_cmp, post_mv, post_time, switched, to_algo, oh_cmp=0, oh_mv=0, oh_time=0.0):
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
