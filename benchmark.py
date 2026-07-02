import json
from pathlib import Path

from src.data_loader import load_dataset

import src.algorithm_metrics.insertion_sort as ins_mod
import src.algorithm_metrics.merge_sort     as mrg_mod
import src.algorithm_metrics.quick_sort     as qck_mod


DATA_TYPES = [
    "random",
    "sorted",
    "reverse_sorted",
    "nearly_sorted",
    "duplicate_heavy",
    "all_equal",
    "edge_cases",
]

# Registry: algorithm name → its metrics module.
# The module's sort function resets counters and captures elapsed_ms internally,
# so no external timer wrapper is needed here.
ALGO_REGISTRY = {
    "insertion_sort": ins_mod,
    "merge_sort":     mrg_mod,
    "quick_sort":     qck_mod,
}


# =========================
# CORE RUNNER
# =========================

def run_sort(module, arr):
    """
    Call the public sort function from a metrics module, then read back
    the metrics it recorded internally.

    Each metrics module's sort function already handles:
      - resetting comparison_count / move_count to 0
      - timing the full sort (stored as module.elapsed_ms)

    We always pass arr.copy() so the original dataset is never mutated,
    even for insertion_sort which sorts in place.
    """
    fn_name = module.__name__.split(".")[-1]   # e.g. "insertion_sort"
    getattr(module, fn_name)(arr.copy())

    return {
        "time_ms":     module.elapsed_ms,
        "comparisons": module.comparison_count,
        "moves":       module.move_count,
    }


# =========================
# CORE BENCHMARK
# =========================

def run_benchmark(size, algo_names, save=True):

    print("\n" + "=" * 60)
    print(f"🚀 RUNNING SIZE = {size}")
    print("=" * 60)

    for algo_name in algo_names:

        module = ALGO_REGISTRY.get(algo_name)
        if module is None:
            print(f"⚠️  Unknown algorithm: {algo_name} — skipping")
            continue

        print(f"\n🔹 Algorithm: {algo_name}")
        results = []

        for dtype in DATA_TYPES:

            loaded = load_dataset(dtype, size)

            # =========================
            # EDGE CASES
            # =========================
            if dtype == "edge_cases":

                edge_dir   = Path("data") / "edge_cases"
                edge_files = sorted(edge_dir.glob("*.json"))

                for i, case in enumerate(loaded):
                    case_name = edge_files[i].stem if i < len(edge_files) else f"case_{i}"
                    data      = case if isinstance(case, list) else []

                    metrics = run_sort(module, data)

                    print(
                        f"  {algo_name} | edge_cases ({case_name}) | {size} → "
                        f"{metrics['time_ms']:.4f} ms  "
                        f"comparisons={metrics['comparisons']}  "
                        f"moves={metrics['moves']}"
                    )

                    results.append({
                        "algorithm":   algo_name,
                        "type":        "edge_cases",
                        "case":        case_name,
                        "size":        size,
                        "time_ms":     metrics["time_ms"],
                        "comparisons": metrics["comparisons"],
                        "moves":       metrics["moves"],
                    })

            # =========================
            # NORMAL CASES
            # =========================
            else:

                metrics = run_sort(module, loaded)

                print(
                    f"  {algo_name} | {dtype} | {size} → "
                    f"{metrics['time_ms']:.4f} ms  "
                    f"comparisons={metrics['comparisons']}  "
                    f"moves={metrics['moves']}"
                )

                results.append({
                    "algorithm":   algo_name,
                    "type":        dtype,
                    "size":        size,
                    "time_ms":     metrics["time_ms"],
                    "comparisons": metrics["comparisons"],
                    "moves":       metrics["moves"],
                })

        if save:
            save_results(algo_name, size, results)


# =========================
# SAVE RESULTS
# =========================

def save_results(algo_name, size, results):

    base_dir  = Path("results") / algo_name
    base_dir.mkdir(parents=True, exist_ok=True)

    file_path = base_dir / f"{size}.json"

    with open(file_path, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\n📁 Saved: {file_path}\n")