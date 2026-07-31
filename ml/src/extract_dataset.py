"""
extract_dataset.py
------------------
Phase 1 — Existing Result Data Extraction

Reads all benchmark result JSON files from results/{algorithm}/ directories,
flattens the nested checkpoint and outcome fields, normalises 'moves' ->
'data_movements', computes derived features, assigns best-action labels, and
saves the processed dataset as a CSV.

No sorting algorithm is imported or executed here.

Usage:
    python ml/src/extract_dataset.py
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import pandas as pd

# ---------------------------------------------------------------------------
# Path setup — all paths are relative to the project root
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[2]  # adaptive-sorting-research/
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

RESULTS_DIR = PROJECT_ROOT / "results"
OUTPUT_DIR = PROJECT_ROOT / "ml" / "data" / "processed"
OUTPUT_FILE = OUTPUT_DIR / "checkpoint_training.csv"

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
# Records whose checkpoint was reached at 100% (edge cases: empty / single-element)
# are excluded from training — there is nothing useful to decide at the checkpoint.
EDGE_CASE_PCT_THRESHOLD = 99.0

# Algorithms present in the results directory
EXPECTED_ALGORITHMS = ["insertion_sort", "merge_sort", "quick_sort"]

# Input distribution types that are considered "normal" (6 training types)
NORMAL_TYPES = {"random", "sorted", "reverse_sorted", "nearly_sorted",
                "duplicate_heavy", "all_equal"}


# ---------------------------------------------------------------------------
# Record validation
# ---------------------------------------------------------------------------

def is_valid_record(record: dict) -> bool:
    """Check if the record has all required fields with correct types."""
    required_top = ["algorithm", "type", "size", "checkpoint", "continue"]
    for field in required_top:
        if field not in record or record[field] is None:
            print(f"WARNING: Skipping record: missing top-level field '{field}'")
            return False

    # Check top-level numeric
    if not isinstance(record["size"], (int, float)) or isinstance(record["size"], bool):
        print(f"WARNING: Skipping record: 'size' is non-numeric: {record['size']}")
        return False

    # Check checkpoint nested fields
    chk = record["checkpoint"]
    if not isinstance(chk, dict):
        print("WARNING: Skipping record: 'checkpoint' is not a dictionary")
        return False
    for field in ["checkpoint_pct", "time_ms", "comparisons", "moves"]:
        if field not in chk or chk[field] is None:
            print(f"WARNING: Skipping record: missing checkpoint field '{field}'")
            return False
        if not isinstance(chk[field], (int, float)) or isinstance(chk[field], bool):
            print(f"WARNING: Skipping record: checkpoint field '{field}' is non-numeric: {chk[field]}")
            return False

    # Check continue nested fields
    cont = record["continue"]
    if not isinstance(cont, dict):
        print("WARNING: Skipping record: 'continue' is not a dictionary")
        return False
    for field in ["time_ms", "comparisons", "moves"]:
        if field not in cont or cont[field] is None:
            print(f"WARNING: Skipping record: missing continue field '{field}'")
            return False
        if not isinstance(cont[field], (int, float)) or isinstance(cont[field], bool):
            print(f"WARNING: Skipping record: continue field '{field}' is non-numeric: {cont[field]}")
            return False

    # Check switch nested fields if present
    for key, val in record.items():
        if key.startswith("switch_"):
            if not isinstance(val, dict):
                print(f"WARNING: Skipping record: '{key}' is not a dictionary")
                return False
            for field in ["time_ms", "comparisons", "moves"]:
                if field not in val or val[field] is None:
                    print(f"WARNING: Skipping record: missing switch field '{field}' in '{key}'")
                    return False
                if not isinstance(val[field], (int, float)) or isinstance(val[field], bool):
                    print(f"WARNING: Skipping record: switch field '{field}' in '{key}' is non-numeric")
                    return False

    return True


# ---------------------------------------------------------------------------
# Record flattening
# ---------------------------------------------------------------------------

def _extract_switch_outcomes(record: dict) -> dict:
    """Extract all switch_* outcome fields into a flat dictionary.

    Keys follow the pattern:
        switch_{target}_time_ms
        switch_{target}_comparisons
        switch_{target}_data_movements   (from 'moves')
        switch_{target}_overhead_time_ms
    """
    outcomes: dict = {}
    for key, value in record.items():
        if not key.startswith("switch_"):
            continue
        target = key  # e.g. 'switch_merge_sort'
        outcomes[f"{target}_time_ms"] = value.get("time_ms")
        outcomes[f"{target}_comparisons"] = value.get("comparisons")
        outcomes[f"{target}_data_movements"] = value.get("moves")  # normalised name
        overhead = value.get("overhead", {})
        outcomes[f"{target}_overhead_time_ms"] = overhead.get("time_ms")
    return outcomes


def flatten_record(record: dict) -> dict | None:
    """Flatten one JSON record into a single flat dictionary.

    Returns None if the record should be excluded (edge case or bad data).
    """
    # ---- Exclusion checks ----
    checkpoint = record.get("checkpoint", {})
    checkpoint_pct = checkpoint.get("checkpoint_pct", 0.0)

    if checkpoint_pct >= EDGE_CASE_PCT_THRESHOLD:
        return None  # edge case — skip

    record_type = record.get("type", "")
    if record_type == "edge_cases":
        return None  # belt-and-suspenders: also skip by type

    # ---- Core identifier fields ----
    row: dict = {
        "algorithm": record.get("algorithm"),
        "input_type": record_type,
        "size": record.get("size"),
    }

    # Optional 'case' sub-label (only edge cases have it; already excluded)
    if "case" in record:
        row["case"] = record["case"]

    # ---- Checkpoint features ----
    row["checkpoint_pct"] = checkpoint_pct
    row["checkpoint_time_ms"] = checkpoint.get("time_ms")
    row["checkpoint_comparisons"] = checkpoint.get("comparisons")
    row["checkpoint_data_movements"] = checkpoint.get("moves")  # normalised

    # ---- Continue outcome ----
    cont = record.get("continue", {})
    row["continue_time_ms"] = cont.get("time_ms")
    row["continue_comparisons"] = cont.get("comparisons")
    row["continue_data_movements"] = cont.get("moves")
    cont_overhead = cont.get("overhead", {})
    row["continue_overhead_time_ms"] = cont_overhead.get("time_ms")

    # ---- Switch outcomes (algorithm-specific) ----
    row.update(_extract_switch_outcomes(record))

    # ---- Derived features ----
    size = row["size"]
    chk_cmp = row["checkpoint_comparisons"]
    chk_mov = row["checkpoint_data_movements"]
    chk_time = row["checkpoint_time_ms"]

    row["comparisons_per_element"] = chk_cmp / size if size else None
    row["movements_per_element"] = chk_mov / size if size else None
    # work_ratio: how comparison-heavy vs move-heavy; +1 avoids div-by-zero
    row["work_ratio"] = chk_cmp / (chk_mov + 1) if chk_mov is not None else None
    row["time_per_element_ms"] = chk_time / size if size else None

    return row


# ---------------------------------------------------------------------------
# File discovery and loading
# ---------------------------------------------------------------------------

def discover_result_files() -> list[Path]:
    """Return all JSON result files found in results/{algorithm}/ directories."""
    files: list[Path] = []
    for algo_dir in sorted(RESULTS_DIR.iterdir()):
        if algo_dir.is_dir() and algo_dir.name in EXPECTED_ALGORITHMS:
            for json_file in sorted(algo_dir.glob("*.json")):
                files.append(json_file)
    return files


def load_and_flatten(json_path: Path) -> tuple[list[dict], int, int, int]:
    """Load a JSON file and flatten each record.

    Returns:
        (rows, total_records, excluded_count, invalid_count)
    """
    try:
        with json_path.open("r", encoding="utf-8") as f:
            records: list[dict] = json.load(f)
    except (json.JSONDecodeError, OSError) as e:
        print(f"WARNING: Failed to read/parse {json_path}: {e}")
        return [], 0, 0, 1

    rows: list[dict] = []
    excluded = 0
    invalid = 0
    for rec in records:
        if not is_valid_record(rec):
            invalid += 1
            continue
        flattened = flatten_record(rec)
        if flattened is None:
            excluded += 1
        else:
            rows.append(flattened)
    return rows, len(records), excluded, invalid


# ---------------------------------------------------------------------------
# Main extraction pipeline
# ---------------------------------------------------------------------------

STABLE_COLUMN_ORDER = [
    "algorithm",
    "input_type",
    "size",
    "checkpoint_pct",
    "checkpoint_time_ms",
    "checkpoint_comparisons",
    "checkpoint_data_movements",
    "continue_time_ms",
    "continue_comparisons",
    "continue_data_movements",
    "continue_overhead_time_ms",
    "switch_insertion_sort_time_ms",
    "switch_insertion_sort_comparisons",
    "switch_insertion_sort_data_movements",
    "switch_insertion_sort_overhead_time_ms",
    "switch_merge_sort_time_ms",
    "switch_merge_sort_comparisons",
    "switch_merge_sort_data_movements",
    "switch_merge_sort_overhead_time_ms",
    "switch_quick_sort_time_ms",
    "switch_quick_sort_comparisons",
    "switch_quick_sort_data_movements",
    "switch_quick_sort_overhead_time_ms",
    "comparisons_per_element",
    "movements_per_element",
    "work_ratio",
    "time_per_element_ms",
    "best_action",
    "best_action_total_ms",
    "speedup_vs_continue"
]

def extract_dataset() -> pd.DataFrame:
    """Run the full extraction pipeline and return the processed DataFrame."""
    files = discover_result_files()
    if not files:
        print(f"ERROR: No JSON files found under {RESULTS_DIR}", file=sys.stderr)
        sys.exit(1)

    print(f"Discovered {len(files)} result files across {RESULTS_DIR}")

    all_rows: list[dict] = []
    total_records = 0
    total_excluded = 0
    total_invalid = 0
    files_processed = 0

    for json_path in files:
        rows, n_records, n_excluded, n_invalid = load_and_flatten(json_path)
        all_rows.extend(rows)
        total_records += n_records
        total_excluded += n_excluded
        total_invalid += n_invalid
        files_processed += 1
        algo = json_path.parent.name
        size = json_path.stem
        status = f"  [{algo}/{size}.json] {n_records} records -> {n_excluded} excluded, {n_invalid} invalid, {len(rows)} kept"
        print(status)

    print(f"\nSummary of extraction:")
    print(f"  Files processed: {files_processed}")
    print(f"  Records processed: {total_records}")
    print(f"  Records excluded (edge cases): {total_excluded}")
    print(f"  Records skipped (invalid): {total_invalid}")
    print(f"  Total records kept: {len(all_rows)}")

    if not all_rows:
        print("ERROR: No valid records extracted.", file=sys.stderr)
        sys.exit(1)

    df = pd.DataFrame(all_rows)

    # ---- Drop Duplicates ----
    initial_len = len(df)
    df = df.drop_duplicates()
    n_duplicates = initial_len - len(df)
    if n_duplicates > 0:
        print(f"  WARNING: Detected and dropped {n_duplicates} duplicate records.")

    # ---- Assign labels ----
    # Import absolute package structure
    from ml.src.generate_labels import assign_labels
    df = assign_labels(df)

    # Ensure deterministic column order
    existing_cols = [col for col in STABLE_COLUMN_ORDER if col in df.columns]
    df = df[existing_cols]

    print(f"\nLabel distribution (best_action):")
    print(df["best_action"].value_counts().to_string())

    return df


def save_dataset(df: pd.DataFrame) -> None:
    """Save the processed DataFrame to CSV."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUTPUT_FILE, index=False)
    print(f"\nSaved {len(df)} rows -> {OUTPUT_FILE}")


if __name__ == "__main__":
    df = extract_dataset()
    save_dataset(df)

    # Quick sanity preview
    print("\n-- Column list --")
    for col in df.columns:
        print(f"  {col}")
    print(f"\n-- First row preview --")
    print(df.iloc[0].to_string())

