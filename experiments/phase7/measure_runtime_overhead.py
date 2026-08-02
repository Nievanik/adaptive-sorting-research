import argparse
import csv
import json
import random
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

from src.adaptive.adaptive_sorter import adaptive_sort, SUPPORTED_ALGORITHMS, SUPPORTED_INPUT_TYPES
from ml.src.runtime_predictor import RuntimePredictor
from src.datasets.generator import (
    random_dataset,
    sorted_dataset,
    reverse_sorted_dataset,
    nearly_sorted_dataset,
    duplicate_heavy_dataset,
    all_equal_dataset,
)

def generate_array(size: int, input_type: str) -> list[int]:
    """Generate a fresh array based on input_type."""
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

def compute_percentiles(values: list[float]) -> dict[str, float]:
    """Compute summary statistics for a list of values."""
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
        "p95": float(np.percentile(arr, 95)),
        "p99": float(np.percentile(arr, 99)) if len(arr) >= 10 else None
    }

def run_experiment(config_path: Path, output_dir: Path, is_quick: bool = False, seed_override: int | None = None) -> tuple[int, dict, Path, Path]:
    # 1. Load Configuration
    with config_path.open("r", encoding="utf-8") as f:
        config = json.load(f)

    if is_quick:
        config["warmup_runs"] = 2
        config["measured_repetitions"] = 2
        config["array_sizes"] = [100, 500]

    seed = seed_override if seed_override is not None else config.get("random_seed", 42)
    random.seed(seed)
    np.random.seed(seed)

    # Validate config
    for size in config["array_sizes"]:
        if size <= 0:
            raise ValueError(f"Invalid array size: {size}")
    if config["measured_repetitions"] <= 0:
        raise ValueError("measured_repetitions must be > 0")

    for input_type in config["input_types"]:
        if input_type not in SUPPORTED_INPUT_TYPES:
            raise ValueError(f"Unsupported input_type: {input_type}")

    for algo in config["starting_algorithms"]:
        if algo not in SUPPORTED_ALGORITHMS:
            raise ValueError(f"Unsupported starting algorithm: {algo}")

    # 2. Instantiate Predictor
    predictor = RuntimePredictor()
    model_load_ns = predictor.model_load_ns
    model_load_ms = predictor.model_load_ms

    # 3. Warmup Runs
    warmup_runs = config.get("warmup_runs", 10)
    for _ in range(warmup_runs):
        arr = generate_array(500, "random")
        _ = adaptive_sort(arr, starting_algorithm="quick_sort", input_type="random", predictor=predictor)

    # 4. Measurement Loop
    raw_results = []
    run_id = 0
    measured_repetitions = config["measured_repetitions"]

    for size in config["array_sizes"]:
        for input_type in config["input_types"]:
            for algo in config["starting_algorithms"]:
                for rep in range(measured_repetitions):
                    run_id += 1
                    arr = generate_array(size, input_type)
                    arr_copy = list(arr) # for correctness check
                    
                    res = adaptive_sort(
                        arr,
                        starting_algorithm=algo,
                        input_type=input_type,
                        predictor=predictor,
                        checkpoint_pct=config.get("checkpoint_pct", 50.0)
                    )

                    # Correctness checks
                    if not res.is_sorted:
                        raise RuntimeError(f"Run {run_id} failed: output not sorted!")
                    if sorted(arr_copy) != list(res.sorted_values):
                        raise RuntimeError(f"Run {run_id} failed: output not permutation of input!")
                    if arr != arr_copy:
                        raise RuntimeError(f"Run {run_id} failed: input was unexpectedly mutated!")

                    prediction_total_ns = res.feature_build_ns + res.inference_ns
                    ml_overhead_ns = prediction_total_ns
                    action_resolution_ns = res.total_runtime_ns - (res.checkpoint_time_ns + prediction_total_ns + res.execution_after_decision_ns)
                    ml_overhead_pct = (ml_overhead_ns / res.total_runtime_ns * 100) if res.total_runtime_ns > 0 else 0.0

                    # Integrity checks
                    assert res.feature_build_ns >= 0
                    assert res.inference_ns >= 0
                    assert res.total_runtime_ns >= ml_overhead_ns
                    if res.executed_action == "continue":
                        assert res.switch_overhead_ns == 0

                    raw_results.append({
                        "run_id": run_id,
                        "seed": seed,
                        "repetition": rep,
                        "array_size": size,
                        "input_type": input_type,
                        "starting_algorithm": algo,
                        "checkpoint_pct": config.get("checkpoint_pct", 50.0),
                        
                        "requested_action": res.requested_action,
                        "executed_action": res.executed_action,
                        "final_algorithm": res.final_algorithm,
                        "prediction_succeeded": res.prediction_succeeded,
                        "fallback_used": res.fallback_used,
                        "fallback_reason": res.fallback_reason,

                        "model_load_ns": model_load_ns,
                        "checkpoint_time_ns": res.checkpoint_time_ns,
                        "feature_build_ns": res.feature_build_ns,
                        "inference_ns": res.inference_ns,
                        "prediction_total_ns": prediction_total_ns,
                        "action_resolution_ns": action_resolution_ns,
                        "execution_after_decision_ns": res.execution_after_decision_ns,
                        "switch_overhead_ns": res.switch_overhead_ns,
                        "total_runtime_ns": res.total_runtime_ns,

                        "feature_build_ms": res.feature_build_ms,
                        "inference_ms": res.inference_ms,
                        "prediction_total_ms": prediction_total_ns / 1_000_000,
                        "switch_overhead_ms": res.switch_overhead_ms,
                        "total_runtime_ms": res.total_runtime_ms,

                        "ml_overhead_ns": ml_overhead_ns,
                        "ml_overhead_pct": ml_overhead_pct,

                        "comparisons": res.comparisons,
                        "data_movements": res.data_movements,
                        "is_sorted": res.is_sorted,
                    })

    # 5. Save Raw CSV
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / "runtime_overhead_raw.csv"
    if raw_results:
        with csv_path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=raw_results[0].keys())
            writer.writeheader()
            writer.writerows(raw_results)

    # 6. Compute Summaries
    overall_stats = {
        "model_load_ns": model_load_ns,
        "model_load_ms": model_load_ms,
        "feature_build_ns": compute_percentiles([r["feature_build_ns"] for r in raw_results]),
        "inference_ns": compute_percentiles([r["inference_ns"] for r in raw_results]),
        "prediction_total_ns": compute_percentiles([r["prediction_total_ns"] for r in raw_results]),
        "switch_overhead_ns": compute_percentiles([r["switch_overhead_ns"] for r in raw_results]),
        "total_runtime_ns": compute_percentiles([r["total_runtime_ns"] for r in raw_results]),
        "ml_overhead_pct": compute_percentiles([r["ml_overhead_pct"] for r in raw_results]),
    }

    # Groupings
    def group_by(key):
        groups = {}
        for r in raw_results:
            groups.setdefault(r[key], []).append(r)
        return {
            str(k): {
                "feature_build_ns": compute_percentiles([x["feature_build_ns"] for x in v]),
                "inference_ns": compute_percentiles([x["inference_ns"] for x in v]),
                "total_runtime_ns": compute_percentiles([x["total_runtime_ns"] for x in v]),
                "ml_overhead_pct": compute_percentiles([x["ml_overhead_pct"] for x in v]),
                "switch_overhead_ns": compute_percentiles([x["switch_overhead_ns"] for x in v]),
            }
            for k, v in groups.items()
        }

    grouped_stats = {
        "by_array_size": group_by("array_size"),
        "by_input_type": group_by("input_type"),
        "by_starting_algorithm": group_by("starting_algorithm"),
        "by_executed_action": group_by("executed_action"),
        "by_fallback_used": group_by("fallback_used"),
    }
    
    research_metrics = {
        "fraction_continued": sum(1 for r in raw_results if r["executed_action"] == "continue") / max(1, len(raw_results)),
        "fraction_switched": sum(1 for r in raw_results if r["executed_action"] != "continue") / max(1, len(raw_results)),
        "fraction_fallback": sum(1 for r in raw_results if r["fallback_used"]) / max(1, len(raw_results)),
        "switch_to_self_rate": sum(1 for r in raw_results if r["fallback_reason"] == "switch_to_self") / max(1, len(raw_results)),
        "prediction_failure_rate": sum(1 for r in raw_results if r["fallback_reason"] == "prediction_failed") / max(1, len(raw_results)),
    }

    summary = {
        "overall": overall_stats,
        "grouped": grouped_stats,
        "research_metrics": research_metrics,
    }

    json_path = output_dir / "runtime_overhead_summary.json"
    with json_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    return len(raw_results), overall_stats, csv_path, json_path


def main():
    parser = argparse.ArgumentParser(description="Phase 7.4 Runtime Overhead Measurement")
    parser.add_argument("--config", type=str, default="experiments/phase7/overhead_config.json", help="Path to config JSON")
    parser.add_argument("--output-dir", type=str, default="results/phase7", help="Output directory")
    parser.add_argument("--seed", type=int, default=None, help="Random seed override")
    parser.add_argument("--quick", action="store_true", help="Run quick smoke test")
    
    args = parser.parse_args()

    config_path = Path(args.config)
    output_dir = Path(args.output_dir)

    print(f"Starting Phase 7.4 Runtime Overhead Measurement...")
    num_runs, overall, csv_path, json_path = run_experiment(config_path, output_dir, is_quick=args.quick, seed_override=args.seed)

    print("\n--- Completion Summary ---")
    print(f"Measured Runs:        {num_runs}")
    print(f"Model Load Time:      {overall['model_load_ms']:.2f} ms")
    if num_runs > 0:
        print(f"Median Feature Build: {overall['feature_build_ns'].get('median', 0) / 1_000_000:.3f} ms")
        print(f"Median Inference:     {overall['inference_ns'].get('median', 0) / 1_000_000:.3f} ms")
        print(f"Median ML Overhead:   {overall['ml_overhead_pct'].get('median', 0):.2f}%")
    print(f"Raw CSV saved to:     {csv_path}")
    print(f"Summary JSON saved to:{json_path}")


if __name__ == "__main__":
    main()
