import random
import time

comparison_count = 0
move_count = 0


def partition(arr, low, high):
    global comparison_count, move_count

    pivot_index = random.randint(low, high)

    # swap pivot to end (3 moves)
    arr[pivot_index], arr[high] = arr[high], arr[pivot_index]
    move_count += 3

    pivot = arr[high]
    i = low - 1

    for j in range(low, high):
        comparison_count += 1

        if arr[j] <= pivot:
            i += 1

            # swap arr[i], arr[j] (3 moves)
            arr[i], arr[j] = arr[j], arr[i]
            move_count += 3

    # final pivot swap (3 moves)
    arr[i + 1], arr[high] = arr[high], arr[i + 1]
    move_count += 3

    return i + 1


def quick_sort(arr):
    arr = arr.copy()
    _quick_sort_iterative(arr, 0, len(arr) - 1)
    return arr


def _quick_sort_iterative(arr, low, high):
    stack = [(low, high)]

    while stack:
        low, high = stack.pop()

        if low < high:
            p = partition(arr, low, high)

            # push larger partition first (optimization)
            if (p - 1 - low) > (high - (p + 1)):
                stack.append((low, p - 1))
                stack.append((p + 1, high))
            else:
                stack.append((p + 1, high))
                stack.append((low, p - 1))


if __name__ == "__main__":

    arr = [5, 2, 9, 1, 3]

    # reset metrics
    comparison_count = 0
    move_count = 0

    start = time.perf_counter_ns()

    result = quick_sort(arr.copy())

    elapsed_ms = (time.perf_counter_ns() - start) / 1_000_000

    print(f"Sorted array     : {result}")
    print(f"Comparison count : {comparison_count}")
    print(f"Move count       : {move_count}")
    print(f"Elapsed (ms)     : {elapsed_ms:.6f}")