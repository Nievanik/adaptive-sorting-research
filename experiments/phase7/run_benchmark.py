import argparse
import csv
import json
import random
import sys
import time
from pathlib import Path
import numpy as np

from ml.src.runtime_predictor import RuntimePredictor
from experiments.phase7.benchmark_runner import run_strategy, get_oracle, hash_array
from src.datasets.generator import (
    random_dataset,
    sorted_dataset,
    reverse_sorted_dataset,
    nearly_sorted_dataset,
    duplicate_heavy_dataset,
    all_equal_dataset,
)

def generate_array(size: int, input_type: str) -> list[int]:
    if input_type == "random":
        return random_dataset(size)
    elif input_type == "sorted":
        return sorted_dataset(size)
    elif input_type == "reverse_sorted":
        return reverse_sorted_dataset(size)
    elif input_type == "nearly_sorted":
        return nearly_sorted_dataset(size)
    elif input_type == "duplicate_heavy":
        return duplicate_heavy_dataset(size)
    elif input_type == "all_equal":
        return all_equal_dataset(size)
    else:
        raise ValueError(f"Unsupported input_type: {input_type}")

def compute_percentiles(values: list[float]) -> dict:
    if not values:
        return {}
    arr = np.array(values)
    return {
        "count": len(arr),
        "mean": float(np.mean(arr)),
        "median": float(np.median(arr)),
        "std": float(np.std(arr)),
        "min": float(np.min(arr)),
        "max": float(np.max(arr)),
        "p95": float(np.percentile(arr, 95))
    }

def run_benchmark_experiment(config_path: Path, output_dir: Path, is_quick: bool = False):
    with config_path.open("r", encoding="utf-8") as f:
        config = json.load(f)

    if is_quick:
        config["repetitions"] = 2
        config["array_sizes"] = [100, 500]

    seed = config.get("random_seed", 42)
    random.seed(seed)
    np.random.seed(seed)

    predictor = RuntimePredictor()

    raw_results = []
    run_id = 0

    for size in config["array_sizes"]:
        for input_type in config["input_types"]:
            for starting_algo in config["starting_algorithms"]:
                for rep in range(config["repetitions"]):
                    run_id += 1
                    
                    # Generate identical array once per repetition
                    base_arr = generate_array(size, input_type)
                    base_sorted = sorted(base_arr)
                    base_hash = hash_array(base_sorted)

                    # Compute Oracle action 
                    # Only calculate if adaptive_ml is tested, but good for dataset metadata overall
                    oracle = get_oracle(base_arr, starting_algo) if size > 1 else {"oracle_action": "continue", "oracle_runtime_ns": 0}

                    for strategy in config["strategies"]:
                        # Deep copy the array for fairness
                        arr_copy = list(base_arr)
                        
                        try:
                            res = run_strategy(
                                strategy, 
                                arr_copy, 
                                starting_algo, 
                                input_type, 
                                predictor, 
                                config.get("checkpoint_pct", 50.0)
                            )
                        except Exception as e:
                            print(f"Error running strategy {strategy} on {starting_algo} size {size}: {e}")
                            raise

                        if not res.is_sorted:
                            raise RuntimeError(f"Strategy {strategy} failed to sort array of size {size} correctly!")
                        if res.output_hash != base_hash:
                            raise RuntimeError(f"Strategy {strategy} failed: output not a permutation of input!")
                        if res.total_runtime_ns < 0:
                            raise RuntimeError(f"Strategy {strategy} reported negative runtime: {res.total_runtime_ns}")

                        row = {
                            "run_id": run_id,
                            "seed": seed,
                            "repetition": rep,
                            "strategy": strategy,
                            "array_size": size,
                            "input_type": input_type,
                            "starting_algorithm": starting_algo,
                            
                            "requested_action": res.requested_action,
                            "executed_action": res.executed_action,
                            "final_algorithm": res.final_algorithm,
                            "prediction_succeeded": res.prediction_succeeded,
                            "fallback_used": res.fallback_used,
                            
                            "total_runtime_ns": res.total_runtime_ns,
                            "total_runtime_ms": res.total_runtime_ns / 1_000_000,
                            "comparisons": res.comparisons,
                            "data_movements": res.data_movements,
                            "switch_overhead_ns": res.switch_overhead_ns,
                            "feature_build_ns": res.feature_build_ns,
                            "inference_ns": res.inference_ns,
                            
                            "is_sorted": res.is_sorted,
                            "output_hash": res.output_hash,
                            
                            "oracle_action": oracle["oracle_action"],
                            "oracle_runtime_ns": oracle["oracle_runtime_ns"],
                            "oracle_matches_prediction": res.executed_action == oracle["oracle_action"] if strategy == "adaptive_ml" else None
                        }
                        raw_results.append(row)

    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / "benchmark_raw.csv"
    if raw_results:
        with csv_path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=raw_results[0].keys())
            writer.writeheader()
            writer.writerows(raw_results)

    # Summarize
    def group_by(keys):
        groups = {}
        for r in raw_results:
            key = tuple(r[k] for k in keys)
            groups.setdefault(key, []).append(r)
        return {
            str(k): {
                "total_runtime_ns": compute_percentiles([x["total_runtime_ns"] for x in v]),
                "wins": 0 # to be calculated later if needed
            }
            for k, v in groups.items()
        }

    grouped_stats = {
        "by_strategy": group_by(["strategy"]),
        "by_strategy_size": group_by(["strategy", "array_size"]),
        "by_strategy_input_type": group_by(["strategy", "input_type"]),
        "by_strategy_starting_algo": group_by(["strategy", "starting_algorithm"]),
    }

    summary = {
        "total_rows": len(raw_results),
        "grouped": grouped_stats
    }
    
    json_path = output_dir / "benchmark_summary.json"
    with json_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    return len(raw_results), csv_path, json_path

def main():
    parser = argparse.ArgumentParser(description="Phase 7.5 End-to-End Benchmark")
    parser.add_argument("--config", type=str, default="experiments/phase7/benchmark_config.json")
    parser.add_argument("--output-dir", type=str, default="results/phase7")
    parser.add_argument("--quick", action="store_true")
    
    args = parser.parse_args()
    
    print(f"Starting Phase 7.5 Benchmark (quick={args.quick})...")
    num_rows, csv_path, json_path = run_benchmark_experiment(Path(args.config), Path(args.output_dir), args.quick)
    
    print(f"\nDone! Generated {num_rows} benchmark rows.")
    print(f"Raw CSV: {csv_path}")
    print(f"Summary: {json_path}")

if __name__ == "__main__":
    main()
