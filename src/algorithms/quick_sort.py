import random

def partition(arr, low, high):
    pivot_index = random.randint(low, high)
    arr[pivot_index], arr[high] = arr[high], arr[pivot_index]

    pivot = arr[high]
    i = low - 1

    for j in range(low, high):
        if arr[j] <= pivot:
            i += 1
            arr[i], arr[j] = arr[j], arr[i]

    arr[i + 1], arr[high] = arr[high], arr[i + 1]
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

            # push larger partition first (slightly optimized stack usage)
            if p - 1 - low > high - (p + 1):
                stack.append((low, p - 1))
                stack.append((p + 1, high))
            else:
                stack.append((p + 1, high))
                stack.append((low, p - 1))