import time

comparison_count = 0
swap_count       = 0
step_count       = 0
elapsed_ms       = 0


def insertion_sort(arr):
    global comparison_count, swap_count, step_count, elapsed_ms
    
    start_time = time.perf_counter_ns()
    
    for i in range(1, len(arr)):
        key = arr[i]
        j   = i - 1
        
        step_count += 2

        while j >= 0:
            comparison_count += 1
            step_count       += 1

            if arr[j] > key:
                arr[j + 1] = arr[j]
                swap_count  += 1
                step_count += 1
                
                j -= 1
                step_count += 1
            else:
                break

        arr[j + 1] = key
        step_count += 1

    elapsed_ms = (time.perf_counter_ns() - start_time) / 1_000_000
    return arr

arr = [5,2,9,1,3]
result = insertion_sort(arr.copy())
print(f"Sorted array     : {result}")
print(f"Comparison count : {comparison_count}")
print(f"Swap count       : {swap_count}")
print(f"Step count       : {step_count}")
print(f"Elapsed (ms)     : {elapsed_ms:.6f}")