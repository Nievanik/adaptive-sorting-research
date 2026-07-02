import json
from pathlib import Path

from src.data_loader import load_dataset
from src.checkpoint.runner import run_to_checkpoint, continue_sort, switch_sort

DATA_TYPES = [
    "random",
    "sorted",
    "reverse_sorted",
    "nearly_sorted",
    "duplicate_heavy",
    "all_equal",
    "edge_cases",
]

# Registry of supported algorithm names for validation.
ALGO_REGISTRY = {
    "insertion_sort": None,
    "merge_sort":     None,
    "quick_sort":     None,
}


# =========================
# CORE RUNNER
# =========================

def run_sort_with_checkpoint(algo_name, arr):
    """
    Run the algorithm to its 50% checkpoint, record the metrics,
    then run both continuation and all possible switching options.
    
    Returns a unified dictionary containing metrics for:
      - checkpoint state
      - continue path
      - switch to option A path
      - switch to option B path
    """
    state = run_to_checkpoint(algo_name, arr)
    cont_res = continue_sort(state)
    
    other_algos = [name for name in ALGO_REGISTRY if name != algo_name]
    switch_res_a = switch_sort(state, other_algos[0])
    switch_res_b = switch_sort(state, other_algos[1])
    
    return {
        "checkpoint": {
            "checkpoint_pct":         state["checkpoint_pct"],
            "time_ms":                state["time_ms"],
            "comparisons":            state["comparisons"],
            "moves":                  state["moves"]
        },
        "continue": {
            "time_ms":                cont_res["total_time_ms"],
            "comparisons":            cont_res["total_comparisons"],
            "moves":                  cont_res["total_moves"],
            "overhead":               cont_res["overhead"]
        },
        f"switch_{other_algos[0]}": {
            "time_ms":                switch_res_a["total_time_ms"],
            "comparisons":            switch_res_a["total_comparisons"],
            "moves":                  switch_res_a["total_moves"],
            "overhead":               switch_res_a["overhead"]
        },
        f"switch_{other_algos[1]}": {
            "time_ms":                switch_res_b["total_time_ms"],
            "comparisons":            switch_res_b["total_comparisons"],
            "moves":                  switch_res_b["total_moves"],
            "overhead":               switch_res_b["overhead"]
        }
    }


# =========================
# CORE BENCHMARK
# =========================

def run_benchmark(size, algo_names, save=True):

    print("\n" + "=" * 60)
    print(f"🚀 RUNNING SIZE = {size}")
    print("=" * 60)

    for algo_name in algo_names:

        if algo_name not in ALGO_REGISTRY:
            print(f"⚠️  Unknown algorithm: {algo_name} — skipping")
            continue

        other_algos = [name for name in ALGO_REGISTRY if name != algo_name]
        print(f"\n🔹 Algorithm: {algo_name} (Alternatives: {', '.join(other_algos)})")
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

                    metrics = run_sort_with_checkpoint(algo_name, data)

                    print(
                        f"  {algo_name} | edge_cases ({case_name}) | {size} →\n"
                        f"    [CP {metrics['checkpoint']['checkpoint_pct']:.1f}%] cmp={metrics['checkpoint']['comparisons']}  mv={metrics['checkpoint']['moves']}  time={metrics['checkpoint']['time_ms']:.4f} ms\n"
                        f"    [CONT     ] cmp={metrics['continue']['comparisons']}  mv={metrics['continue']['moves']}  time={metrics['continue']['time_ms']:.4f} ms\n"
                        f"    [SW {other_algos[0]:<10}] cmp={metrics[f'switch_{other_algos[0]}']['comparisons']}  mv={metrics[f'switch_{other_algos[0]}']['moves']}  time={metrics[f'switch_{other_algos[0]}']['time_ms']:.4f} ms\n"
                        f"    [SW {other_algos[1]:<10}] cmp={metrics[f'switch_{other_algos[1]}']['comparisons']}  mv={metrics[f'switch_{other_algos[1]}']['moves']}  time={metrics[f'switch_{other_algos[1]}']['time_ms']:.4f} ms"
                    )

                    results.append({
                        "algorithm":              algo_name,
                        "type":                   "edge_cases",
                        "case":                   case_name,
                        "size":                   size,
                        "checkpoint":             metrics["checkpoint"],
                        "continue":               metrics["continue"],
                        f"switch_{other_algos[0]}": metrics[f"switch_{other_algos[0]}"],
                        f"switch_{other_algos[1]}": metrics[f"switch_{other_algos[1]}"]
                    })

            # =========================
            # NORMAL CASES
            # =========================
            else:

                metrics = run_sort_with_checkpoint(algo_name, loaded)

                print(
                    f"  {algo_name} | {dtype} | {size} →\n"
                    f"    [CP {metrics['checkpoint']['checkpoint_pct']:.1f}%] cmp={metrics['checkpoint']['comparisons']}  mv={metrics['checkpoint']['moves']}  time={metrics['checkpoint']['time_ms']:.4f} ms\n"
                    f"    [CONT     ] cmp={metrics['continue']['comparisons']}  mv={metrics['continue']['moves']}  time={metrics['continue']['time_ms']:.4f} ms\n"
                    f"    [SW {other_algos[0]:<10}] cmp={metrics[f'switch_{other_algos[0]}']['comparisons']}  mv={metrics[f'switch_{other_algos[0]}']['moves']}  time={metrics[f'switch_{other_algos[0]}']['time_ms']:.4f} ms\n"
                    f"    [SW {other_algos[1]:<10}] cmp={metrics[f'switch_{other_algos[1]}']['comparisons']}  mv={metrics[f'switch_{other_algos[1]}']['moves']}  time={metrics[f'switch_{other_algos[1]}']['time_ms']:.4f} ms"
                )

                results.append({
                    "algorithm":              algo_name,
                    "type":                   dtype,
                    "size":                   size,
                    "checkpoint":             metrics["checkpoint"],
                    "continue":               metrics["continue"],
                    f"switch_{other_algos[0]}": metrics[f"switch_{other_algos[0]}"],
                    f"switch_{other_algos[1]}": metrics[f"switch_{other_algos[1]}"]
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