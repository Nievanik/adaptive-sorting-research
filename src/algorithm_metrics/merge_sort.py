import time

comparison_count = 0
move_count       = 0
elapsed_ms       = 0


def merge_sort(arr):
    """
    Public entry point. Resets all metrics, times the full sort,
    and delegates to the recursive _merge_sort helper.
    """
    global comparison_count, move_count, elapsed_ms

    # Reset metrics at the start of every call
    comparison_count = 0
    move_count       = 0

    start  = time.perf_counter_ns()
    result = _merge_sort(arr)
    elapsed_ms = (time.perf_counter_ns() - start) / 1_000_000

    return result


def _merge_sort(arr):
    """
    Recursive merge sort — does NOT reset metrics (handled by merge_sort).
    """
    if len(arr) <= 1:
        return arr.copy()

    mid   = len(arr) // 2
    left  = _merge_sort(arr[:mid])
    right = _merge_sort(arr[mid:])

    return _merge(left, right)


def _merge(left, right):
    global comparison_count, move_count

    sorted_arr = []
    i = j = 0

    while i < len(left) and j < len(right):
        comparison_count += 1

        if left[i] <= right[j]:
            sorted_arr.append(left[i])
            move_count += 1
            i += 1
        else:
            sorted_arr.append(right[j])
            move_count += 1
            j += 1

    while i < len(left):
        sorted_arr.append(left[i])
        move_count += 1
        i += 1

    while j < len(right):
        sorted_arr.append(right[j])
        move_count += 1
        j += 1

    return sorted_arr


if __name__ == "__main__":
    arr    = [5, 2, 9, 1, 3]
    result = merge_sort(arr.copy())
    print(f"Sorted array     : {result}")
    print(f"Comparison count : {comparison_count}")
    print(f"Move count       : {move_count}")
    print(f"Elapsed (ms)     : {elapsed_ms:.6f}")