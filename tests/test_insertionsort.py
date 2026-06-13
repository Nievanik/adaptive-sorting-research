from src.algorithms.merge_sort import merge_sort
import random


def run_test(name, func):
    try:
        func()
        print(f"[PASS] {name}")
    except AssertionError:
        print(f"[FAIL] {name}")
        raise


def test_empty_array():
    assert merge_sort([]) == []


def test_single_element():
    assert merge_sort([5]) == [5]


def test_duplicates():
    assert merge_sort([3, 1, 2, 3, 1]) == [1, 1, 2, 3, 3]


def test_sorted_input():
    assert merge_sort([1, 2, 3, 4, 5]) == [1, 2, 3, 4, 5]


def test_reverse_sorted_input():
    assert merge_sort([5, 4, 3, 2, 1]) == [1, 2, 3, 4, 5]


def test_negative_numbers():
    assert merge_sort([10, -1, 4, -5, 0]) == [-5, -1, 0, 4, 10]


def test_all_equal():
    assert merge_sort([7, 7, 7, 7]) == [7, 7, 7, 7]


# 🔥 improved random test (important for research)
def test_random_input():
    arr = [random.randint(-100, 100) for _ in range(20)]
    assert merge_sort(arr) == sorted(arr)


# 🔥 large input test (useful for performance scaling later)
def test_large_input():
    arr = [random.randint(-1000, 1000) for _ in range(200)]
    assert merge_sort(arr) == sorted(arr)


if __name__ == "__main__":
    print("\nRunning Merge Sort Tests...\n")

    run_test("empty_array", test_empty_array)
    run_test("single_element", test_single_element)
    run_test("duplicates", test_duplicates)
    run_test("sorted_input", test_sorted_input)
    run_test("reverse_sorted_input", test_reverse_sorted_input)
    run_test("negative_numbers", test_negative_numbers)
    run_test("all_equal", test_all_equal)
    run_test("random_input", test_random_input)
    run_test("large_input", test_large_input)

    print("\nAll Merge Sort tests passed!")