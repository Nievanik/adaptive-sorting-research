import time

comparison_count = 0
move_count = 0


def merge_sort(arr):
    if len(arr) <= 1:
        return arr.copy()

    mid = len(arr) // 2
    left = merge_sort(arr[:mid])
    right = merge_sort(arr[mid:])

    return merge(left, right)


def merge(left, right):
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

    arr = [5, 2, 9, 1, 3]

    comparison_count = 0
    move_count = 0

    start_time = time.perf_counter_ns()

    result = merge_sort(arr.copy())

    elapsed_ms = (time.perf_counter_ns() - start_time) / 1_000_000

    print(f"Sorted array     : {result}")
    print(f"Comparison count : {comparison_count}")
    print(f"Move count       : {move_count}")
    print(f"Elapsed (ms)     : {elapsed_ms:.6f}")