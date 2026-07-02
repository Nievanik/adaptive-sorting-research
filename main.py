import argparse
from benchmark import run_benchmark, ALGO_REGISTRY


if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description="Run adaptive sorting benchmarks with full metrics."
    )

    parser.add_argument(
        "--algo",
        nargs="+",
        required=True,
        help=f"Algorithms to benchmark: {' '.join(ALGO_REGISTRY.keys())}"
    )

    parser.add_argument(
        "--size",
        nargs="+",
        type=int,
        required=True,
        help="Dataset sizes to run (e.g. 100 500 1000 5000 10000)"
    )

    args = parser.parse_args()

    valid   = [a for a in args.algo if a in ALGO_REGISTRY]
    invalid = [a for a in args.algo if a not in ALGO_REGISTRY]

    for name in invalid:
        print(f"⚠️  Unknown algorithm: {name} — skipping")

    if not valid:
        print("No valid algorithms selected. Exiting.")
        exit(1)

    for size in args.size:
        run_benchmark(size=size, algo_names=valid, save=True)