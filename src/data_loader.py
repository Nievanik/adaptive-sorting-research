import json
from pathlib import Path

# Resolve data/ relative to this file's location so the loader works
# regardless of which directory the script is run from.
DATA_ROOT = Path(__file__).resolve().parents[1] / "data"


def load_dataset(data_type, size):

    if data_type == "edge_cases":
        edge_dir = DATA_ROOT / "edge_cases"
        datasets = []

        for file in sorted(edge_dir.glob("*.json")):
            with open(file, "r") as f:
                datasets.append(json.load(f))

        return datasets

    file_path = DATA_ROOT / data_type / f"{data_type}_{size}.json"

    with open(file_path, "r") as f:
        return json.load(f)