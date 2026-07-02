import time

comparison_count = 0
move_count       = 0
elapsed_ms       = 0


def insertion_sort(arr):
    global comparison_count, move_count, elapsed_ms

    # Reset metrics at the start of every call
    comparison_count = 0
    move_count       = 0

    start_time = time.perf_counter_ns()

    for i in range(1, len(arr)):
        key = arr[i]
        j   = i - 1

        while j >= 0:
            comparison_count += 1

            if arr[j] > key:
                arr[j + 1] = arr[j]
                move_count += 1          # shift right
                j -= 1
            else:
                break

        arr[j + 1] = key
        move_count += 1                  # final placement of key

    elapsed_ms = (time.perf_counter_ns() - start_time) / 1_000_000
    return arr


if __name__ == "__main__":
    arr    = [5, 2, 9, 1, 3]
    result = insertion_sort(arr.copy())
    print(f"Sorted array     : {result}")
    print(f"Comparison count : {comparison_count}")
    print(f"Move count       : {move_count}")
    print(f"Elapsed (ms)     : {elapsed_ms:.6f}")