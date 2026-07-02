"""
Tests for src/checkpoint/
Run with: PYTHONPATH=. python tests/test_checkpoint.py
"""

import random
from src.checkpoint.runner import run_to_checkpoint, continue_sort, switch_sort

ALGOS = ["insertion_sort", "merge_sort", "quick_sort"]


def run_test(name, func):
    try:
        func()
        print(f"  [PASS] {name}")
    except AssertionError as e:
        print(f"  [FAIL] {name} — {e}")
        raise


def is_sorted(arr):
    return all(arr[i] <= arr[i + 1] for i in range(len(arr) - 1))


# ---------------------------------------------------------------------------
# continue_sort correctness
# ---------------------------------------------------------------------------

def test_continue_correctness():
    arr = [random.randint(0, 1000) for _ in range(50)]
    expected = sorted(arr)
    for algo in ALGOS:
        state  = run_to_checkpoint(algo, arr)
        result = continue_sort(state)
        assert result["sorted_arr"] == expected, f"{algo} continue produced wrong result"


def test_continue_edge_empty():
    for algo in ALGOS:
        state  = run_to_checkpoint(algo, [])
        result = continue_sort(state)
        assert result["sorted_arr"] == [], f"{algo} failed on empty array"


def test_continue_edge_single():
    for algo in ALGOS:
        state  = run_to_checkpoint(algo, [42])
        result = continue_sort(state)
        assert result["sorted_arr"] == [42], f"{algo} failed on single element"


def test_continue_sorted_input():
    arr = list(range(30))
    for algo in ALGOS:
        state  = run_to_checkpoint(algo, arr)
        result = continue_sort(state)
        assert result["sorted_arr"] == arr, f"{algo} failed on sorted input"


def test_continue_reverse_sorted():
    arr = list(range(30, 0, -1))
    for algo in ALGOS:
        state  = run_to_checkpoint(algo, arr)
        result = continue_sort(state)
        assert result["sorted_arr"] == sorted(arr), f"{algo} failed on reverse sorted"


def test_continue_all_equal():
    arr = [7] * 40
    for algo in ALGOS:
        state  = run_to_checkpoint(algo, arr)
        result = continue_sort(state)
        assert result["sorted_arr"] == arr, f"{algo} failed on all_equal"


# ---------------------------------------------------------------------------
# switch_sort correctness — all 9 combinations
# ---------------------------------------------------------------------------

def test_switch_correctness_all_combos():
    arr = [random.randint(0, 1000) for _ in range(50)]
    expected = sorted(arr)
    for from_algo in ALGOS:
        state = run_to_checkpoint(from_algo, arr)
        for to_algo in ALGOS:
            result = switch_sort(state, to_algo)
            assert result["sorted_arr"] == expected, \
                f"switch {from_algo} -> {to_algo} produced wrong result"


def test_switch_nearly_sorted():
    arr = list(range(40))
    arr[5], arr[15] = arr[15], arr[5]
    expected = sorted(arr)
    for from_algo in ALGOS:
        state = run_to_checkpoint(from_algo, arr)
        for to_algo in ALGOS:
            result = switch_sort(state, to_algo)
            assert result["sorted_arr"] == expected, \
                f"switch {from_algo} -> {to_algo} failed on nearly_sorted"


def test_switch_duplicate_heavy():
    arr = [random.randint(1, 5) for _ in range(60)]
    expected = sorted(arr)
    for from_algo in ALGOS:
        state = run_to_checkpoint(from_algo, arr)
        for to_algo in ALGOS:
            result = switch_sort(state, to_algo)
            assert result["sorted_arr"] == expected, \
                f"switch {from_algo} -> {to_algo} failed on duplicate_heavy"


# ---------------------------------------------------------------------------
# Metrics validity
# ---------------------------------------------------------------------------

def test_metrics_sum_correctly():
    arr = [random.randint(0, 500) for _ in range(40)]
    for algo in ALGOS:
        state  = run_to_checkpoint(algo, arr)
        result = continue_sort(state)
        assert result["total_comparisons"] == state["comparisons"] + result["post_comparisons"]
        assert result["total_moves"]       == state["moves"]        + result["post_moves"]
        for key in ("post_comparisons", "post_moves", "total_comparisons", "total_moves"):
            assert result[key] >= 0, f"{algo} {key} is negative"


def test_checkpoint_state_fields():
    arr = [random.randint(0, 100) for _ in range(20)]
    for algo in ALGOS:
        state = run_to_checkpoint(algo, arr)
        for key in ("algo", "arr", "comparisons", "moves", "time_ms", "checkpoint_pct"):
            assert key in state, f"{algo} state missing key: {key}"
        assert 0 <= state["checkpoint_pct"] <= 100


def test_result_fields():
    arr = [random.randint(0, 100) for _ in range(20)]
    for algo in ALGOS:
        state  = run_to_checkpoint(algo, arr)
        result = continue_sort(state)
        for key in ("sorted_arr", "post_comparisons", "post_moves", "post_time_ms",
                    "total_comparisons", "total_moves", "total_time_ms",
                    "switched", "from_algo", "to_algo", "checkpoint_pct"):
            assert key in result, f"{algo} result missing key: {key}"


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("\nRunning Checkpoint Tests...\n")

    print("— continue_sort correctness —")
    run_test("continue: random input",          test_continue_correctness)
    run_test("continue: empty array",           test_continue_edge_empty)
    run_test("continue: single element",        test_continue_edge_single)
    run_test("continue: sorted input",          test_continue_sorted_input)
    run_test("continue: reverse sorted",        test_continue_reverse_sorted)
    run_test("continue: all equal",             test_continue_all_equal)

    print("\n— switch_sort correctness (all 9 combinations) —")
    run_test("switch: all combos random",       test_switch_correctness_all_combos)
    run_test("switch: nearly sorted",           test_switch_nearly_sorted)
    run_test("switch: duplicate heavy",         test_switch_duplicate_heavy)

    print("\n— metrics validity —")
    run_test("metrics sum correctly",           test_metrics_sum_correctly)
    run_test("checkpoint state has all fields", test_checkpoint_state_fields)
    run_test("result has all fields",           test_result_fields)

    print("\n✅ All checkpoint tests passed!")
