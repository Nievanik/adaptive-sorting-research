import time

comparison_count = 0
swap_count       = 0   # (actually MOVES in merge sort, but keeping name as you used)
step_count       = 0
elapsed_ms       = 0

start_time = None


def merge_sort(arr):
    global start_time, step_count

    if start_time is None:
        start_time = time.perf_counter_ns()

    step_count += 1  # function call

    if len(arr) <= 1:
        return arr

    mid   = len(arr) // 2
    left  = merge_sort(arr[:mid])
    right = merge_sort(arr[mid:])

    return merge(left, right)


def merge(left, right):
    global comparison_count, swap_count, step_count

    sorted_arr = []
    i = j = 0

    while i < len(left) and j < len(right):
        comparison_count += 1
        step_count += 1

        if left[i] <= right[j]:
            sorted_arr.append(left[i])
            swap_count += 1   # actually MOVE
            step_count += 1
            i += 1
        else:
            sorted_arr.append(right[j])
            swap_count += 1   # actually MOVE
            step_count += 1
            j += 1

    while i < len(left):
        sorted_arr.append(left[i])
        swap_count += 1
        step_count += 1
        i += 1

    while j < len(right):
        sorted_arr.append(right[j])
        swap_count += 1
        step_count += 1
        j += 1

    return sorted_arr


# ---------------- RUN ----------------

arr = [92, 150, 2, 134, 177, 99, 186, 45, 101, 143, 83, 132, 53, 190, 6,
 73, 95, 36, 191, 174, 65, 158, 29, 197, 8, 137, 48, 195, 172, 130,
 17, 171, 166, 94, 154, 183, 57, 188, 180, 114, 161, 69, 148, 179,
 87, 198, 120, 142, 200, 60, 92, 150, 2, 134, 177, 99, 186, 45, 101, 143, 83, 132, 53, 190, 6,
 73, 95, 36, 191, 174, 65, 158, 29, 197, 8, 137, 48, 195, 172, 130,
 17, 171, 166, 94, 154, 183, 57, 188, 180, 114, 161, 69, 148, 179,
 87, 198, 120, 142, 200, 60]

start_time = time.perf_counter_ns()

result = merge_sort(arr.copy())

elapsed_ms = (time.perf_counter_ns() - start_time) / 1_000_000


print(f"Sorted array     : {result}")
print(f"Comparison count : {comparison_count}")
print(f"Swap count       : {swap_count}")
print(f"Step count       : {step_count}")
print(f"Elapsed (ms)     : {elapsed_ms:.6f}")