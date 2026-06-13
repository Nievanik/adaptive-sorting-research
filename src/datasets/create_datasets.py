import json
from pathlib import Path

from generator import (
    random_dataset,
    sorted_dataset,
    reverse_sorted_dataset,
    nearly_sorted_dataset,
    duplicate_heavy_dataset,
    all_equal_dataset,
    empty_dataset,
    single_element_dataset,
)

DATASET_SIZES = [
    100,
    500,
    1000,
    5000,
    10000,
]

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data"


def save_dataset(dataset, folder, filename):
    """
    Save dataset to a JSON file.
    """
    output_dir = DATA_DIR / folder
    output_dir.mkdir(parents=True, exist_ok=True)

    filepath = output_dir / filename

    with open(filepath, "w") as file:
        json.dump(dataset, file)

    print(f"Created {filepath}")


def main():

    # Random datasets
    for size in DATASET_SIZES:
        save_dataset(
            random_dataset(size),
            "random",
            f"random_{size}.json"
        )

    # Sorted datasets
    for size in DATASET_SIZES:
        save_dataset(
            sorted_dataset(size),
            "sorted",
            f"sorted_{size}.json"
        )

    # Reverse sorted datasets
    for size in DATASET_SIZES:
        save_dataset(
            reverse_sorted_dataset(size),
            "reverse_sorted",
            f"reverse_sorted_{size}.json"
        )

    # Nearly sorted datasets
    for size in DATASET_SIZES:
        save_dataset(
            nearly_sorted_dataset(size),
            "nearly_sorted",
            f"nearly_sorted_{size}.json"
        )

    # Duplicate-heavy datasets
    for size in DATASET_SIZES:
        save_dataset(
            duplicate_heavy_dataset(size),
            "duplicate_heavy",
            f"duplicate_heavy_{size}.json"
        )

    # All-equal datasets
    for size in DATASET_SIZES:
        save_dataset(
            all_equal_dataset(size),
            "all_equal",
            f"all_equal_{size}.json"
        )

    # Edge cases
    save_dataset(
        empty_dataset(),
        "edge_cases",
        "empty.json"
    )

    save_dataset(
        single_element_dataset(),
        "edge_cases",
        "single_element.json"
    )


if __name__ == "__main__":
    main()