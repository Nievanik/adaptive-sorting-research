import time
import hashlib
from typing import Any
from dataclasses import dataclass

from src.adaptive.adaptive_sorter import adaptive_sort
from src.checkpoint.runner import run_to_checkpoint, continue_sort, switch_sort, _sort_with, CHECKPOINT_MODULES

@dataclass
class BenchmarkResult:
    total_runtime_ns: int
    comparisons: int
    data_movements: int
    is_sorted: bool
    output_hash: str
    
    # Adaptive specific
    requested_action: str | None = None
    executed_action: str | None = None
    final_algorithm: str | None = None
    prediction_succeeded: bool | None = None
    fallback_used: bool | None = None
    switch_overhead_ns: int | None = None
    feature_build_ns: int | None = None
    inference_ns: int | None = None


def hash_array(arr: list) -> str:
    return hashlib.md5(str(arr).encode()).hexdigest()

def run_adaptive_ml(arr: list, starting_algorithm: str, input_type: str, predictor: Any, checkpoint_pct: float) -> BenchmarkResult:
    res = adaptive_sort(
        arr, 
        starting_algorithm=starting_algorithm, 
        input_type=input_type, 
        predictor=predictor, 
        checkpoint_pct=checkpoint_pct
    )
    return BenchmarkResult(
        total_runtime_ns=res.total_runtime_ns,
        comparisons=res.comparisons,
        data_movements=res.data_movements,
        is_sorted=res.is_sorted,
        output_hash=hash_array(list(res.sorted_values)),
        requested_action=res.requested_action,
        executed_action=res.executed_action,
        final_algorithm=res.final_algorithm,
        prediction_succeeded=res.prediction_succeeded,
        fallback_used=res.fallback_used,
        switch_overhead_ns=res.switch_overhead_ns,
        feature_build_ns=res.feature_build_ns,
        inference_ns=res.inference_ns
    )

def run_always_continue(arr: list, starting_algorithm: str, checkpoint_pct: float) -> BenchmarkResult:
    t_start = time.perf_counter_ns()
    
    state = run_to_checkpoint(starting_algorithm, arr)
    res = continue_sort(state)
    
    t_end = time.perf_counter_ns()
    total_ns = t_end - t_start
    
    sorted_arr = res["sorted_arr"]
    is_sorted = all(sorted_arr[i] <= sorted_arr[i+1] for i in range(len(sorted_arr)-1))

    return BenchmarkResult(
        total_runtime_ns=total_ns,
        comparisons=res["total_comparisons"],
        data_movements=res["total_moves"],
        is_sorted=is_sorted,
        output_hash=hash_array(sorted_arr),
        executed_action="continue",
        final_algorithm=starting_algorithm,
        switch_overhead_ns=0
    )

def run_pure_algorithm(arr: list, algo_name: str) -> BenchmarkResult:
    t_start = time.perf_counter_ns()
    sorted_arr, comparisons, moves = _sort_with(algo_name, arr)
    t_end = time.perf_counter_ns()
    
    is_sorted = all(sorted_arr[i] <= sorted_arr[i+1] for i in range(len(sorted_arr)-1))
    return BenchmarkResult(
        total_runtime_ns=t_end - t_start,
        comparisons=comparisons,
        data_movements=moves,
        is_sorted=is_sorted,
        output_hash=hash_array(sorted_arr),
        executed_action="pure_baseline",
        final_algorithm=algo_name,
        switch_overhead_ns=0
    )

def run_python_timsort(arr: list) -> BenchmarkResult:
    t_start = time.perf_counter_ns()
    arr.sort()
    t_end = time.perf_counter_ns()
    
    is_sorted = all(arr[i] <= arr[i+1] for i in range(len(arr)-1))
    return BenchmarkResult(
        total_runtime_ns=t_end - t_start,
        comparisons=0,
        data_movements=0,
        is_sorted=is_sorted,
        output_hash=hash_array(arr),
        executed_action="pure_baseline",
        final_algorithm="python_timsort",
        switch_overhead_ns=0
    )

def run_strategy(strategy: str, arr: list, starting_algorithm: str, input_type: str, predictor: Any, checkpoint_pct: float) -> BenchmarkResult:
    if strategy == "adaptive_ml":
        return run_adaptive_ml(arr, starting_algorithm, input_type, predictor, checkpoint_pct)
    elif strategy == "always_continue":
        return run_always_continue(arr, starting_algorithm, checkpoint_pct)
    elif strategy in ("insertion_sort", "merge_sort", "quick_sort"):
        return run_pure_algorithm(arr, strategy)
    elif strategy == "python_timsort":
        return run_python_timsort(arr)
    else:
        raise ValueError(f"Unknown strategy: {strategy}")

def get_oracle(arr: list, starting_algorithm: str) -> dict:
    t_start_cp = time.perf_counter_ns()
    state = run_to_checkpoint(starting_algorithm, list(arr))
    cp_time_ns = time.perf_counter_ns() - t_start_cp
    
    other_algos = [name for name in CHECKPOINT_MODULES if name != starting_algorithm]
    
    def measure_continue():
        s = _safe_checkpoint_snapshot(state, list(arr))
        t = time.perf_counter_ns()
        continue_sort(s)
        return time.perf_counter_ns() - t
        
    def measure_switch(target):
        s = _safe_checkpoint_snapshot(state, list(arr))
        t = time.perf_counter_ns()
        switch_sort(s, target)
        return time.perf_counter_ns() - t

    times = {
        "continue": cp_time_ns + measure_continue(),
        f"switch_{other_algos[0]}": cp_time_ns + measure_switch(other_algos[0]),
        f"switch_{other_algos[1]}": cp_time_ns + measure_switch(other_algos[1]),
    }
    
    best_action = min(times, key=times.get)
    return {
        "oracle_action": best_action,
        "oracle_runtime_ns": times[best_action]
    }

def _safe_checkpoint_snapshot(state: dict, original_arr: list) -> dict:
    # helper for oracle to copy the state and array without leaking mutations
    snap = dict(state)
    snap["arr"] = list(original_arr)
    if "remaining_stack" in snap:
        snap["remaining_stack"] = list(snap["remaining_stack"])
    return snap
