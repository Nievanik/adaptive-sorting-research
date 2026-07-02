from src.algorithms.insertion_sort import insertion_sort
import random


def run_test(name, func):
    try:
        func()
        print(f"[PASS] {name}")
    except AssertionError:
        print(f"[FAIL] {name}")
        raise


def test_empty_array():
    assert insertion_sort([]) == []


def test_single_element():
    assert insertion_sort([5]) == [5]


def test_duplicates():
    assert insertion_sort([3, 1, 2, 3, 1]) == [1, 1, 2, 3, 3]


def test_sorted_input():
    assert insertion_sort([1, 2, 3, 4, 5]) == [1, 2, 3, 4, 5]


def test_reverse_sorted_input():
    assert insertion_sort([5, 4, 3, 2, 1]) == [1, 2, 3, 4, 5]


def test_negative_numbers():
    assert insertion_sort([10, -1, 4, -5, 0]) == [-5, -1, 0, 4, 10]


def test_all_equal():
    assert insertion_sort([7, 7, 7, 7]) == [7, 7, 7, 7]


def test_random_input():
    arr = [random.randint(-100, 100) for _ in range(20)]
    assert insertion_sort(arr) == sorted(arr)


def test_large_input():
    arr = [random.randint(-1000, 1000) for _ in range(200)]
    assert insertion_sort(arr) == sorted(arr)


if __name__ == "__main__":
    print("\nRunning Insertion Sort Tests...\n")

    run_test("empty_array",          test_empty_array)
    run_test("single_element",       test_single_element)
    run_test("duplicates",           test_duplicates)
    run_test("sorted_input",         test_sorted_input)
    run_test("reverse_sorted_input", test_reverse_sorted_input)
    run_test("negative_numbers",     test_negative_numbers)
    run_test("all_equal",            test_all_equal)
    run_test("random_input",         test_random_input)
    run_test("large_input",          test_large_input)

    print("\nAll Insertion Sort tests passed!")