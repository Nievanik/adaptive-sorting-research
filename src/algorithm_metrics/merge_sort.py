import time

comparison_count = 0
swap_count       = 0   # (actually MOVES in merge sort, but keeping name as you used)
elapsed_ms       = 0

start_time = None

def merge_sort(arr):
    global start_time

    if start_time is None:
        start_time = time.perf_counter_ns()

    if len(arr) <= 1:
        return arr

    mid   = len(arr) // 2
    left  = merge_sort(arr[:mid])
    right = merge_sort(arr[mid:])

    return merge(left, right)

def merge(left, right):
    global comparison_count, swap_count

    sorted_arr = []
    i = j = 0

    while i < len(left) and j < len(right):
        comparison_count += 1

        if left[i] <= right[j]:
            sorted_arr.append(left[i])
            swap_count += 1   # actually MOVE
            
            i += 1
        else:
            sorted_arr.append(right[j])
            swap_count += 1   # actually MOVE
            
            j += 1

    while i < len(left):
        sorted_arr.append(left[i])
        swap_count += 1
        
        i += 1

    while j < len(right):
        sorted_arr.append(right[j])
        swap_count += 1

        j += 1

    return sorted_arr


arr = [5,2,9,1,3]
start_time = time.perf_counter_ns()

result = merge_sort(arr.copy())

elapsed_ms = (time.perf_counter_ns() - start_time) / 1_000_000


print(f"Sorted array     : {result}")
print(f"Comparison count : {comparison_count}")
print(f"Swap count       : {swap_count}")
print(f"Elapsed (ms)     : {elapsed_ms:.6f}")