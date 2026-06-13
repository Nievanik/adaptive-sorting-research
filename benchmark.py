import time
import json
from pathlib import Path

from src.data_loader import load_dataset


DATA_TYPES = [
    "random",
    "sorted",
    "reverse_sorted",
    "nearly_sorted",
    "duplicate_heavy",
    "all_equal",
    "edge_cases"
]


# =========================
# TIMER
# =========================

def measure(sort_fn, arr):
    start = time.perf_counter()
    sort_fn(arr)
    end = time.perf_counter()
    return end - start


# =========================
# CORE BENCHMARK
# =========================

def run_benchmark(size, algorithms, save=True):

    print("\n" + "=" * 60)
    print(f"🚀 RUNNING SIZE = {size}")
    print("=" * 60)

    for algo_name, algo_fn in algorithms.items():

        print(f"\n🔹 Algorithm: {algo_name}")

        results = []

        for dtype in DATA_TYPES:

            loaded = load_dataset(dtype, size)

            # =========================
            # EDGE CASES (UPDATED LOGIC)
            # =========================
            if dtype == "edge_cases":

                # loaded = list of arrays from multiple files
                # we infer case name from file order via loader tweak OR fallback index
                edge_dir = Path("data") / "edge_cases"
                edge_files = sorted(edge_dir.glob("*.json"))

                for i, case in enumerate(loaded):

                    case_name = edge_files[i].stem if i < len(edge_files) else f"case_{i}"

                    data = case.copy() if isinstance(case, list) else case

                    t = measure(algo_fn, data)

                    print(f"{algo_name} | edge_cases ({case_name}) | {size} → {t:.6f}s")

                    results.append({
                        "algorithm": algo_name,
                        "type": "edge_cases",
                        "case": case_name,
                        "size": size,
                        "time": t
                    })

            # =========================
            # NORMAL CASES
            # =========================
            else:

                arr = loaded.copy()
                t = measure(algo_fn, arr)

                print(f"{algo_name} | {dtype} | {size} → {t:.6f}s")

                results.append({
                    "algorithm": algo_name,
                    "type": dtype,
                    "size": size,
                    "time": t
                })

        if save:
            save_results(algo_name, size, results)


# =========================
# SAVE RESULTS
# =========================

def save_results(algo_name, size, results):

    base_dir = Path("results") / algo_name
    base_dir.mkdir(parents=True, exist_ok=True)

    file_path = base_dir / f"{size}.json"

    with open(file_path, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\n📁 Saved: {file_path}\n")