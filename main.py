import argparse
from benchmark import run_benchmark

from src.algorithms.quick_sort import quick_sort
from src.algorithms.merge_sort import merge_sort
from src.algorithms.insertion_sort import insertion_sort


def get_algorithms():
    return {
        "quick_sort": quick_sort,
        "merge_sort": merge_sort,
        "insertion_sort": insertion_sort
    }


if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--algo",
        nargs="+",
        required=True,
        help="Choose algorithms: quick_sort merge_sort insertion_sort"
    )

    parser.add_argument(
        "--size",
        nargs="+",
        type=int,
        required=True,
        help="Dataset sizes: 100 500 1000 ..."
    )

    args = parser.parse_args()

    all_algos = get_algorithms()

    selected_algos = {}

    for a in args.algo:
        if a in all_algos:
            selected_algos[a] = all_algos[a]
        else:
            print(f"❌ Unknown algorithm: {a}")

    if not selected_algos:
        print("❌ No valid algorithms selected")
        exit()

    for size in args.size:
        run_benchmark(
            size=size,
            algorithms=selected_algos,
            save=True
        )