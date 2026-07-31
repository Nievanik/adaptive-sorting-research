"""
validate_dataset.py
-------------------
Phase 1 — Dataset Validation

Loads the processed CSV produced by extract_dataset.py and runs a suite of
integrity checks to verify correctness of the extraction pipeline.

No sorting algorithm is imported or executed here.

Usage:
    python ml/src/validate_dataset.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

CSV_PATH = PROJECT_ROOT / "ml" / "data" / "processed" / "checkpoint_training.csv"


# ---------------------------------------------------------------------------
# Expected values
# ---------------------------------------------------------------------------
EXPECTED_ROW_COUNT = 90          # 3 algorithms × 5 sizes × 6 normal types
CHECKPOINT_PCT_MIN = 40.0        # permissive lower bound
CHECKPOINT_PCT_MAX = 99.0        # strict upper bound (100 = edge case = excluded)

VALID_ALGORITHMS = {"insertion_sort", "merge_sort", "quick_sort"}
VALID_INPUT_TYPES = {"random", "sorted", "reverse_sorted",
                     "nearly_sorted", "duplicate_heavy", "all_equal"}
VALID_ACTIONS = {"continue", "switch_insertion_sort",
                 "switch_merge_sort", "switch_quick_sort"}

CORE_FEATURE_COLS = [
    "algorithm", "input_type", "size",
    "checkpoint_pct", "checkpoint_time_ms",
    "checkpoint_comparisons", "checkpoint_data_movements",
    "continue_time_ms",
    "comparisons_per_element", "movements_per_element",
    "work_ratio", "time_per_element_ms",
    "best_action", "best_action_total_ms", "speedup_vs_continue",
]

# ---------------------------------------------------------------------------
# Check runner
# ---------------------------------------------------------------------------

class CheckResult:
    def __init__(self) -> None:
        self.passed: list[str] = []
        self.failed: list[str] = []

    def ok(self, name: str, detail: str = "") -> None:
        msg = f"  [PASS] {name}" + (f" — {detail}" if detail else "")
        print(msg)
        self.passed.append(name)

    def fail(self, name: str, detail: str = "") -> None:
        msg = f"  [FAIL] {name}" + (f" — {detail}" if detail else "")
        print(msg, file=sys.stderr)
        self.failed.append(name)

    def summary(self) -> bool:
        total = len(self.passed) + len(self.failed)
        print(f"\n{'='*50}")
        print(f"Validation complete: {len(self.passed)}/{total} checks passed")
        if self.failed:
            print(f"FAILED checks: {', '.join(self.failed)}", file=sys.stderr)
            return False
        print("All checks PASSED ✓")
        return True


# ---------------------------------------------------------------------------
# Individual checks
# ---------------------------------------------------------------------------

def check_csv_loads(results: CheckResult) -> pd.DataFrame | None:
    """Check 1: CSV file exists and loads without error."""
    if not CSV_PATH.exists():
        results.fail("csv_exists", f"{CSV_PATH} not found — run extract_dataset.py first")
        return None
    try:
        df = pd.read_csv(CSV_PATH)
        results.ok("csv_loads", f"{len(df)} rows, {len(df.columns)} columns")
        return df
    except Exception as e:
        results.fail("csv_loads", str(e))
        return None


def check_row_count(df: pd.DataFrame, results: CheckResult) -> None:
    """Check 2: Exactly 90 rows (3 algos × 5 sizes × 6 types)."""
    n = len(df)
    if n == EXPECTED_ROW_COUNT:
        results.ok("row_count", f"{n} rows == expected {EXPECTED_ROW_COUNT}")
    else:
        results.fail("row_count", f"got {n} rows, expected {EXPECTED_ROW_COUNT}")


def check_no_edge_cases(df: pd.DataFrame, results: CheckResult) -> None:
    """Check 3: No records with checkpoint_pct >= 99 (edge cases excluded)."""
    bad = df[df["checkpoint_pct"] >= CHECKPOINT_PCT_MAX]
    if bad.empty:
        results.ok("no_edge_cases", "all checkpoint_pct < 99.0")
    else:
        results.fail("no_edge_cases",
                     f"{len(bad)} rows have checkpoint_pct >= 99: "
                     f"{bad[['algorithm','input_type','size','checkpoint_pct']].to_dict('records')}")


def check_checkpoint_pct_range(df: pd.DataFrame, results: CheckResult) -> None:
    """Check 4: checkpoint_pct is within [40, 99) for all rows."""
    out_of_range = df[(df["checkpoint_pct"] < CHECKPOINT_PCT_MIN) |
                      (df["checkpoint_pct"] >= CHECKPOINT_PCT_MAX)]
    if out_of_range.empty:
        lo, hi = df["checkpoint_pct"].min(), df["checkpoint_pct"].max()
        results.ok("checkpoint_pct_range", f"range [{lo:.2f}, {hi:.2f}]")
    else:
        results.fail("checkpoint_pct_range",
                     f"{len(out_of_range)} rows out of [{CHECKPOINT_PCT_MIN}, {CHECKPOINT_PCT_MAX})")


def check_no_nulls_in_core_cols(df: pd.DataFrame, results: CheckResult) -> None:
    """Check 5: No null values in required feature columns."""
    missing_cols = [c for c in CORE_FEATURE_COLS if c not in df.columns]
    if missing_cols:
        results.fail("core_cols_present", f"missing columns: {missing_cols}")
        return
    results.ok("core_cols_present", "all required columns present")

    null_report = {col: int(df[col].isna().sum())
                   for col in CORE_FEATURE_COLS if df[col].isna().any()}
    if not null_report:
        results.ok("no_nulls_in_core_cols", "zero nulls in all core columns")
    else:
        results.fail("no_nulls_in_core_cols", f"null counts: {null_report}")


def check_valid_algorithms(df: pd.DataFrame, results: CheckResult) -> None:
    """Check 6: algorithm column contains only expected values."""
    actual = set(df["algorithm"].unique())
    unexpected = actual - VALID_ALGORITHMS
    if not unexpected:
        results.ok("valid_algorithms", str(sorted(actual)))
    else:
        results.fail("valid_algorithms", f"unexpected values: {unexpected}")


def check_valid_input_types(df: pd.DataFrame, results: CheckResult) -> None:
    """Check 7: input_type contains only normal (non-edge-case) types."""
    actual = set(df["input_type"].unique())
    unexpected = actual - VALID_INPUT_TYPES
    if not unexpected:
        results.ok("valid_input_types", str(sorted(actual)))
    else:
        results.fail("valid_input_types", f"unexpected values: {unexpected}")


def check_best_action_valid(df: pd.DataFrame, results: CheckResult) -> None:
    """Check 8: best_action is non-null and from the set of valid actions."""
    null_count = int(df["best_action"].isna().sum())
    if null_count > 0:
        results.fail("best_action_not_null", f"{null_count} null best_action values")
    else:
        results.ok("best_action_not_null", "no nulls")

    actual = set(df["best_action"].unique())
    invalid = actual - VALID_ACTIONS
    if not invalid:
        results.ok("best_action_valid_values", str(sorted(actual)))
    else:
        results.fail("best_action_valid_values", f"invalid action values: {invalid}")


def check_best_action_self_switch(df: pd.DataFrame, results: CheckResult) -> None:
    """Check 9: An algorithm should never have best_action == switch to itself."""
    bad = df[df.apply(
        lambda r: r["best_action"] == f"switch_{r['algorithm']}", axis=1
    )]
    if bad.empty:
        results.ok("no_self_switch_label", "no rows labelled with self-switch")
    else:
        results.fail("no_self_switch_label",
                     f"{len(bad)} rows have best_action == switch to own algorithm")


def check_speedup_consistency(df: pd.DataFrame, results: CheckResult) -> None:
    """Check 10: speedup_vs_continue >= 1 when best_action is not 'continue'."""
    switched = df[df["best_action"] != "continue"]
    below_one = switched[switched["speedup_vs_continue"] < 0.999]  # small float tolerance
    if below_one.empty:
        results.ok("speedup_consistency",
                   f"{len(switched)} switch rows all have speedup >= 1.0")
    else:
        results.fail("speedup_consistency",
                     f"{len(below_one)} switch rows have speedup < 1.0 "
                     f"(min={switched['speedup_vs_continue'].min():.4f})")


def check_coverage(df: pd.DataFrame, results: CheckResult) -> None:
    """Check 11: Each (algorithm, size) pair has exactly 6 rows."""
    grouped = df.groupby(["algorithm", "size"]).size().reset_index(name="count")
    bad = grouped[grouped["count"] != 6]
    if bad.empty:
        results.ok("coverage_per_algo_size",
                   f"all (algorithm, size) pairs have exactly 6 rows")
    else:
        results.fail("coverage_per_algo_size",
                     f"mismatched pairs:\n{bad.to_string()}")


LEAKAGE_COLS = {
    "continue_time_ms", "continue_comparisons", "continue_data_movements", "continue_overhead_time_ms",
    "switch_insertion_sort_time_ms", "switch_insertion_sort_comparisons", "switch_insertion_sort_data_movements", "switch_insertion_sort_overhead_time_ms",
    "switch_merge_sort_time_ms", "switch_merge_sort_comparisons", "switch_merge_sort_data_movements", "switch_merge_sort_overhead_time_ms",
    "switch_quick_sort_time_ms", "switch_quick_sort_comparisons", "switch_quick_sort_data_movements", "switch_quick_sort_overhead_time_ms",
    "best_action_total_ms", "speedup_vs_continue"
}

ALLOWED_FEATURES = {
    "algorithm", "input_type", "size", "checkpoint_pct",
    "checkpoint_time_ms", "checkpoint_comparisons", "checkpoint_data_movements",
    "comparisons_per_element", "movements_per_element", "work_ratio", "time_per_element_ms"
}

def check_target_leakage(df: pd.DataFrame, results: CheckResult) -> None:
    """Check 12: Ensure allowed training features do not contain any leakage columns."""
    overlap = ALLOWED_FEATURES.intersection(LEAKAGE_COLS)
    if not overlap:
        results.ok("target_leakage_check", "No leakage columns in the allowed feature list.")
    else:
        results.fail("target_leakage_check", f"Leakage columns found in feature list: {overlap}")


# ---------------------------------------------------------------------------
# Summary statistics (informational, not a pass/fail check)
# ---------------------------------------------------------------------------

def print_summary_stats(df: pd.DataFrame) -> None:
    print("\n-- Total Rows and Columns --")
    print(f"Rows: {df.shape[0]}, Columns: {df.shape[1]}")

    print("\n-- Missing Values --")
    print(df.isna().sum().to_string())

    print("\n-- Duplicate Count --")
    print(f"Duplicates: {df.duplicated().sum()}")

    print("\n-- Feature Data Types --")
    print(df.dtypes.to_string())

    print("\n-- Algorithm Distribution --")
    print(df["algorithm"].value_counts().to_string())

    print("\n-- Input Distribution --")
    print(df["input_type"].value_counts().to_string())

    print("\n-- Label Distribution --")
    print(df["best_action"].value_counts().to_string())

    print("\n-- Checkpoint PCT Range --")
    print(f"Min: {df['checkpoint_pct'].min():.2f}%, Max: {df['checkpoint_pct'].max():.2f}%")

    print("\n-- Label distribution by algorithm --")
    print(df.groupby(["algorithm", "best_action"]).size()
          .unstack(fill_value=0).to_string())

    print("\n-- Rows where switching is clearly better (speedup >= 2x) --")
    big_wins = df[df["speedup_vs_continue"] >= 2.0]
    print(f"  {len(big_wins)} rows")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def validate() -> bool:
    print(f"Validating: {CSV_PATH}\n{'='*50}")
    results = CheckResult()

    df = check_csv_loads(results)
    if df is None:
        return results.summary()

    check_row_count(df, results)
    check_no_edge_cases(df, results)
    check_checkpoint_pct_range(df, results)
    check_no_nulls_in_core_cols(df, results)
    check_valid_algorithms(df, results)
    check_valid_input_types(df, results)
    check_best_action_valid(df, results)
    check_best_action_self_switch(df, results)
    check_speedup_consistency(df, results)
    check_coverage(df, results)
    check_target_leakage(df, results)

    print_summary_stats(df)

    return results.summary()


if __name__ == "__main__":
    ok = validate()
    sys.exit(0 if ok else 1)

