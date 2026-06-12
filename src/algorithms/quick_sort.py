import time

comparison_count = 0
swap_count       = 0
step_count       = 0
elapsed_ms       = 0

def partition(arr, low, high):
    global swap_count, comparison_count, step_count

    pivot = arr[high]
    i     = low - 1
    
    step_count += 2

    for j in range(low, high):
        comparison_count += 1
        step_count       += 1

        if arr[j] <= pivot:
            i += 1
            arr[i], arr[j] = arr[j], arr[i]
            swap_count += 1
            step_count += 2
        

    arr[i + 1], arr[high] = arr[high], arr[i + 1]
    swap_count += 1
    step_count += 1

    return i + 1

def quicksort(arr, low, high):
    global step_count
    
    step_count += 1
    
    if low < high:
        p = partition(arr, low, high)
        step_count += 1
        quicksort(arr, low, p - 1)
        quicksort(arr, p + 1, high)

arr = [92, 150, 2, 134, 177, 99, 186, 45, 101, 143, 83, 132, 53, 190, 6,
 73, 95, 36, 191, 174, 65, 158, 29, 197, 8, 137, 48, 195, 172, 130,
 17, 171, 166, 94, 154, 183, 57, 188, 180, 114, 161, 69, 148, 179,
 87, 198, 120, 142, 200, 60, 92, 150, 2, 134, 177, 99, 186, 45, 101, 143, 83, 132, 53, 190, 6,
 73, 95, 36, 191, 174, 65, 158, 29, 197, 8, 137, 48, 195, 172, 130,
 17, 171, 166, 94, 154, 183, 57, 188, 180, 114, 161, 69, 148, 179,
 87, 198, 120, 142, 200, 60, 92, 150, 2, 134, 177, 99, 186, 45, 101, 143, 83, 132, 53, 190, 6,
 73, 95, 36, 191, 174, 65, 158, 29, 197, 8, 137, 48, 195, 172, 130,
 17, 171, 166, 94, 154, 183, 57, 188, 180, 114, 161, 69, 148, 179,
 87, 198, 120, 142, 200, 60, 92, 150, 2, 134, 177, 99, 186, 45, 101, 143, 83, 132, 53, 190, 6,
 73, 95, 36, 191, 174, 65, 158, 29, 197, 8, 137, 48, 195, 172, 130,
 17, 171, 166, 94, 154, 183, 57, 188, 180, 114, 161, 69, 148, 179,
 87, 198, 120, 142, 200, 60, 92, 150, 2, 134, 177, 99, 186, 45, 101, 143, 83, 132, 53, 190, 6,
 73, 95, 36, 191, 174, 65, 158, 29, 197, 8, 137, 48, 195, 172, 130,
 17, 171, 166, 94, 154, 183, 57, 188, 180, 114, 161, 69, 148, 179,
 87, 198, 120, 142, 200, 60, 92, 150, 2, 134, 177, 99, 186, 45, 101, 143, 83, 132, 53, 190, 6,
 73, 95, 36, 191, 174, 65, 158, 29, 197, 8, 137, 48, 195, 172, 130,
 17, 171, 166, 94, 154, 183, 57, 188, 180, 114, 161, 69, 148, 179,
 87, 198, 120, 142, 200, 60]

start_time = time.perf_counter_ns()
quicksort(arr, 0, len(arr) - 1)
elapsed_ms = (time.perf_counter_ns() - start_time) / 1_000_000

print(f"Sorted array     : {arr}")
print(f"Comparison count : {comparison_count}")
print(f"Swap count       : {swap_count}")
print(f"Step count       : {step_count}")
print(f"Elapsed (ms)     : {elapsed_ms:.6f}")