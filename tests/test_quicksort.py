from src.algorithms.quick_sort import quick_sort


def run_test(name, func):
    try:
        func()
        print(f"[PASS] {name}")
    except AssertionError:
        print(f"[FAIL] {name}")
        raise


def test_empty_array():
    assert quick_sort([]) == []


def test_single_element():
    assert quick_sort([5]) == [5]


def test_duplicates():
    assert quick_sort([3, 1, 2, 3, 1]) == [1, 1, 2, 3, 3]


def test_sorted_input():
    assert quick_sort([1, 2, 3, 4, 5]) == [1, 2, 3, 4, 5]


def test_reverse_sorted_input():
    assert quick_sort([5, 4, 3, 2, 1]) == [1, 2, 3, 4, 5]


def test_negative_numbers():
    assert quick_sort([10, -1, 4, -5, 0]) == [-5, -1, 0, 4, 10]


def test_all_equal():
    assert quick_sort([7, 7, 7, 7]) == [7, 7, 7, 7]


def test_random_input():
    assert quick_sort([10, -1, 4, 8, 0]) == [-1, 0, 4, 8, 10]


if __name__ == "__main__":
    run_test("empty_array", test_empty_array)
    run_test("single_element", test_single_element)
    run_test("duplicates", test_duplicates)
    run_test("sorted_input", test_sorted_input)
    run_test("reverse_sorted_input", test_reverse_sorted_input)
    run_test("negative_numbers", test_negative_numbers)
    run_test("all_equal", test_all_equal)
    run_test("random_input", test_random_input)

    print("\nAll Quick Sort tests passed!")