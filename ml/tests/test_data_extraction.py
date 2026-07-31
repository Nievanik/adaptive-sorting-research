"""
test_data_extraction.py
-----------------------
Unit tests for Phase 1 extraction and label generation.

Covers:
- Filtering of edge-case records (checkpoint_pct == 100)
- Inclusion of normal records (checkpoint_pct == 50)
- Normalisation of 'moves' -> 'data_movements'
- Correct best_action label selection
- Derived feature computation
- Guarantee that no sorting algorithm code is called

Run with:
    pytest ml/tests/test_data_extraction.py -v
"""

from __future__ import annotations
import sys
import pandas as pd
# pyrefly: ignore [missing-import]
import pytest

from ml.src.extract_dataset import flatten_record, _extract_switch_outcomes
from ml.src.generate_labels import assign_labels


# ---------------------------------------------------------------------------
# Fixtures — minimal synthetic records mirroring the real JSON structure
# ---------------------------------------------------------------------------

@pytest.fixture()
def normal_insertion_record() -> dict:
    """A normal insertion_sort record with 50% checkpoint."""
    return {
        "algorithm": "insertion_sort",
        "type": "random",
        "size": 100,
        "checkpoint": {
            "checkpoint_pct": 50.0,
            "time_ms": 0.08,
            "comparisons": 645,
            "moves": 646,
        },
        "continue": {
            "time_ms": 0.32,
            "comparisons": 2651,
            "moves": 2653,
            "overhead": {"comparisons": 0, "moves": 0, "time_ms": 0.0},
        },
        "switch_merge_sort": {
            "time_ms": 0.10,   # faster than continue → should be best_action
            "comparisons": 962,
            "moves": 1032,
            "overhead": {"comparisons": 99, "moves": 100, "time_ms": 0.02},
        },
        "switch_quick_sort": {
            "time_ms": 0.19,
            "comparisons": 988,
            "moves": 1142,
            "overhead": {"comparisons": 99, "moves": 100, "time_ms": 0.022},
        },
    }


@pytest.fixture()
def edge_case_record_100pct() -> dict:
    """An edge-case record with checkpoint_pct == 100 (empty array)."""
    return {
        "algorithm": "insertion_sort",
        "type": "edge_cases",
        "case": "empty",
        "size": 100,
        "checkpoint": {
            "checkpoint_pct": 100.0,
            "time_ms": 0.0,
            "comparisons": 0,
            "moves": 0,
        },
        "continue": {
            "time_ms": 0.0,
            "comparisons": 0,
            "moves": 0,
            "overhead": {"comparisons": 0, "moves": 0, "time_ms": 0.0},
        },
        "switch_merge_sort": {
            "time_ms": 0.001,
            "comparisons": 0,
            "moves": 0,
            "overhead": {"comparisons": 0, "moves": 0, "time_ms": 0.001},
        },
        "switch_quick_sort": {
            "time_ms": 0.001,
            "comparisons": 0,
            "moves": 0,
            "overhead": {"comparisons": 0, "moves": 0, "time_ms": 0.001},
        },
    }


@pytest.fixture()
def normal_merge_record() -> dict:
    """A normal merge_sort record with 50% checkpoint."""
    return {
        "algorithm": "merge_sort",
        "type": "sorted",
        "size": 1000,
        "checkpoint": {
            "checkpoint_pct": 50.0,
            "time_ms": 0.57,
            "comparisons": 2216,
            "moves": 4488,
        },
        "continue": {
            "time_ms": 1.20,
            "comparisons": 4932,
            "moves": 9976,
            "overhead": {"comparisons": 0, "moves": 0, "time_ms": 0.0},
        },
        "switch_insertion_sort": {
            "time_ms": 0.67,   # faster than continue
            "comparisons": 3215,
            "moves": 5987,
            "overhead": {"comparisons": 500, "moves": 1000, "time_ms": 0.067},
        },
        "switch_quick_sort": {
            "time_ms": 1.07,
            "comparisons": 7458,
            "moves": 11896,
            "overhead": {"comparisons": 500, "moves": 1000, "time_ms": 0.072},
        },
    }


# ---------------------------------------------------------------------------
# Tests — flatten_record
# ---------------------------------------------------------------------------

class TestFlattenRecord:

    def test_normal_record_is_included(self, normal_insertion_record):
        row = flatten_record(normal_insertion_record)
        assert row is not None, "Normal 50% record should not be excluded"

    def test_edge_case_100pct_is_excluded(self, edge_case_record_100pct):
        row = flatten_record(edge_case_record_100pct)
        assert row is None, "Record with checkpoint_pct==100 must be excluded"

    def test_moves_normalised_to_data_movements(self, normal_insertion_record):
        row = flatten_record(normal_insertion_record)
        assert "checkpoint_data_movements" in row
        assert row["checkpoint_data_movements"] == normal_insertion_record["checkpoint"]["moves"]

    def test_continue_data_movements_normalised(self, normal_insertion_record):
        row = flatten_record(normal_insertion_record)
        assert "continue_data_movements" in row
        assert row["continue_data_movements"] == normal_insertion_record["continue"]["moves"]

    def test_switch_data_movements_normalised(self, normal_insertion_record):
        row = flatten_record(normal_insertion_record)
        assert "switch_merge_sort_data_movements" in row
        assert row["switch_merge_sort_data_movements"] == \
               normal_insertion_record["switch_merge_sort"]["moves"]

    def test_core_fields_populated(self, normal_insertion_record):
        row = flatten_record(normal_insertion_record)
        assert row["algorithm"] == "insertion_sort"
        assert row["input_type"] == "random"
        assert row["size"] == 100
        assert row["checkpoint_pct"] == 50.0
        assert row["checkpoint_comparisons"] == 645

    def test_derived_comparisons_per_element(self, normal_insertion_record):
        row = flatten_record(normal_insertion_record)
        expected = 645 / 100
        assert abs(row["comparisons_per_element"] - expected) < 1e-9

    def test_derived_movements_per_element(self, normal_insertion_record):
        row = flatten_record(normal_insertion_record)
        expected = 646 / 100
        assert abs(row["movements_per_element"] - expected) < 1e-9

    def test_derived_work_ratio(self, normal_insertion_record):
        row = flatten_record(normal_insertion_record)
        # work_ratio = comparisons / (moves + 1)
        expected = 645 / (646 + 1)
        assert abs(row["work_ratio"] - expected) < 1e-9

    def test_derived_time_per_element(self, normal_insertion_record):
        row = flatten_record(normal_insertion_record)
        expected = 0.08 / 100
        assert abs(row["time_per_element_ms"] - expected) < 1e-9

    def test_edge_type_field_excluded_by_type(self):
        """Records with type='edge_cases' are excluded even if pct < 99."""
        rec = {
            "algorithm": "insertion_sort",
            "type": "edge_cases",
            "case": "single_element",
            "size": 100,
            "checkpoint": {"checkpoint_pct": 50.0, "time_ms": 0.0,
                           "comparisons": 0, "moves": 0},
            "continue": {"time_ms": 0.001, "comparisons": 0, "moves": 0,
                         "overhead": {"comparisons": 0, "moves": 0, "time_ms": 0.0}},
            "switch_merge_sort": {"time_ms": 0.001, "comparisons": 0, "moves": 0,
                                  "overhead": {"comparisons": 0, "moves": 0, "time_ms": 0.0}},
            "switch_quick_sort": {"time_ms": 0.001, "comparisons": 0, "moves": 0,
                                  "overhead": {"comparisons": 0, "moves": 0, "time_ms": 0.0}},
        }
        assert flatten_record(rec) is None


# ---------------------------------------------------------------------------
# Tests — _extract_switch_outcomes
# ---------------------------------------------------------------------------

class TestExtractSwitchOutcomes:

    def test_returns_all_switch_keys(self, normal_insertion_record):
        outcomes = _extract_switch_outcomes(normal_insertion_record)
        assert "switch_merge_sort_time_ms" in outcomes
        assert "switch_quick_sort_time_ms" in outcomes

    def test_does_not_return_non_switch_keys(self, normal_insertion_record):
        outcomes = _extract_switch_outcomes(normal_insertion_record)
        for key in outcomes:
            assert key.startswith("switch_"), f"Unexpected key: {key}"

    def test_moves_mapped_to_data_movements(self, normal_insertion_record):
        outcomes = _extract_switch_outcomes(normal_insertion_record)
        assert "switch_merge_sort_data_movements" in outcomes
        assert "switch_merge_sort_moves" not in outcomes  # old name must not appear


# ---------------------------------------------------------------------------
# Tests — assign_labels
# ---------------------------------------------------------------------------

class TestAssignLabels:

    def _make_df(self, rows: list[dict]) -> pd.DataFrame:
        return pd.DataFrame(rows)

    def test_best_action_is_switch_when_switch_is_faster(self, normal_insertion_record):
        row = flatten_record(normal_insertion_record)
        df = self._make_df([row])
        df = assign_labels(df)
        # switch_merge_sort total = 0.08 + 0.10 = 0.18
        # continue total = 0.08 + 0.32 = 0.40
        assert df.iloc[0]["best_action"] == "switch_merge_sort"

    def test_best_action_is_continue_when_continue_is_fastest(self):
        """When continue is faster than all switches, label is 'continue'."""
        row = {
            "algorithm": "insertion_sort",
            "input_type": "sorted",
            "size": 100,
            "checkpoint_pct": 50.0,
            "checkpoint_time_ms": 0.01,
            "checkpoint_comparisons": 49,
            "checkpoint_data_movements": 49,
            "continue_time_ms": 0.01,      # total = 0.02 — fastest
            "continue_comparisons": 99,
            "continue_data_movements": 99,
            "continue_overhead_time_ms": 0.0,
            "switch_merge_sort_time_ms": 0.20,   # total = 0.21
            "switch_quick_sort_time_ms": 0.15,   # total = 0.16
            "comparisons_per_element": 0.49,
            "movements_per_element": 0.49,
            "work_ratio": 1.0,
            "time_per_element_ms": 0.0001,
        }
        df = self._make_df([row])
        df = assign_labels(df)
        assert df.iloc[0]["best_action"] == "continue"

    def test_best_action_total_ms_matches_minimum(self, normal_insertion_record):
        row = flatten_record(normal_insertion_record)
        df = self._make_df([row])
        df = assign_labels(df)
        # Expected minimum total: checkpoint + switch_merge = 0.08 + 0.10 = 0.18
        assert abs(df.iloc[0]["best_action_total_ms"] - 0.18) < 1e-9

    def test_speedup_vs_continue_gte_1_for_switch(self, normal_insertion_record):
        row = flatten_record(normal_insertion_record)
        df = self._make_df([row])
        df = assign_labels(df)
        assert df.iloc[0]["speedup_vs_continue"] >= 1.0

    def test_speedup_equals_1_when_continue_is_best(self):
        row = {
            "algorithm": "insertion_sort",
            "input_type": "sorted",
            "size": 100,
            "checkpoint_pct": 50.0,
            "checkpoint_time_ms": 0.01,
            "checkpoint_comparisons": 49,
            "checkpoint_data_movements": 49,
            "continue_time_ms": 0.01,
            "continue_comparisons": 99,
            "continue_data_movements": 99,
            "continue_overhead_time_ms": 0.0,
            "switch_merge_sort_time_ms": 0.50,
            "switch_quick_sort_time_ms": 0.50,
            "comparisons_per_element": 0.49,
            "movements_per_element": 0.49,
            "work_ratio": 1.0,
            "time_per_element_ms": 0.0001,
        }
        df = self._make_df([row])
        df = assign_labels(df)
        assert df.iloc[0]["speedup_vs_continue"] == pytest.approx(1.0)

    def test_nan_switch_columns_ignored_in_labelling(self, normal_merge_record):
        """merge_sort has no switch_merge_sort column — should not be selected."""
        row = flatten_record(normal_merge_record)
        df = self._make_df([row])
        df = assign_labels(df)
        # switch_merge_sort is not present for merge_sort, so label must not be
        # 'switch_merge_sort'
        assert df.iloc[0]["best_action"] != "switch_merge_sort"

    def test_no_self_switch_label(self, normal_merge_record):
        """merge_sort must never get best_action == 'switch_merge_sort'."""
        row = flatten_record(normal_merge_record)
        df = self._make_df([row])
        df = assign_labels(df)
        assert df.iloc[0]["best_action"] != f"switch_{normal_merge_record['algorithm']}"


# ---------------------------------------------------------------------------
# Meta test — no sorting code imported
# ---------------------------------------------------------------------------

class TestNoSortingCodeImported:

    def test_extract_does_not_import_sorting_modules(self):
        """Confirm no sorting algorithm module was imported."""
        sorting_module_names = {"insertion_sort", "merge_sort", "quick_sort",
                                "benchmark", "main"}
        imported = set(sys.modules.keys())
        overlap = {m for m in imported if any(s in m for s in sorting_module_names)}
        # Filter to only modules that look like they contain sorting logic
        # (not our own extract/label modules)
        suspicious = {m for m in overlap
                      if "extract" not in m and "label" not in m and "test" not in m
                      and m != "__main__"}
        assert not suspicious, \
            f"Potential sorting modules imported: {suspicious}"


# ---------------------------------------------------------------------------
# New Pipeline & Edge Case Tests
# ---------------------------------------------------------------------------

from ml.src.extract_dataset import is_valid_record, extract_dataset, save_dataset
from ml.src.validate_dataset import LEAKAGE_COLS, ALLOWED_FEATURES

class TestPipelineAndEdgeCases:

    def test_invalid_records_detected(self):
        # Missing required field
        assert not is_valid_record({"algorithm": "insertion_sort"})

        # Non-numeric size
        assert not is_valid_record({
            "algorithm": "insertion_sort",
            "type": "random",
            "size": "one hundred",
            "checkpoint": {"checkpoint_pct": 50.0, "time_ms": 0.1, "comparisons": 10, "moves": 10},
            "continue": {"time_ms": 0.2, "comparisons": 20, "moves": 20}
        })

        # Correct structure
        assert is_valid_record({
            "algorithm": "insertion_sort",
            "type": "random",
            "size": 100,
            "checkpoint": {"checkpoint_pct": 50.0, "time_ms": 0.1, "comparisons": 10, "moves": 10},
            "continue": {"time_ms": 0.2, "comparisons": 20, "moves": 20}
        })

    def test_tie_breaking_behavior(self):
        # continue wins over switches if equal remaining time
        row = {
            "algorithm": "insertion_sort",
            "input_type": "random",
            "size": 100,
            "checkpoint_time_ms": 0.1,
            "continue_time_ms": 0.2,
            "switch_quick_sort_time_ms": 0.2,
            "switch_merge_sort_time_ms": 0.2,
        }
        df = pd.DataFrame([row])
        df = assign_labels(df)
        assert df.iloc[0]["best_action"] == "continue"

        # switch_quick_sort wins over switch_merge_sort if equal remaining time
        row = {
            "algorithm": "insertion_sort",
            "input_type": "random",
            "size": 100,
            "checkpoint_time_ms": 0.1,
            "continue_time_ms": 0.5,
            "switch_quick_sort_time_ms": 0.2,
            "switch_merge_sort_time_ms": 0.2,
        }
        df = pd.DataFrame([row])
        df = assign_labels(df)
        assert df.iloc[0]["best_action"] == "switch_quick_sort"

    def test_missing_switch_alternatives(self):
        row = {
            "algorithm": "insertion_sort",
            "input_type": "random",
            "size": 100,
            "checkpoint_time_ms": 0.1,
            "continue_time_ms": 0.5,
            # switch_quick_sort is missing/NaN
            "switch_quick_sort_time_ms": None,
            "switch_merge_sort_time_ms": 0.2,
        }
        df = pd.DataFrame([row])
        df = assign_labels(df)
        assert df.iloc[0]["best_action"] == "switch_merge_sort"

    def test_no_target_leakage(self):
        overlap = ALLOWED_FEATURES.intersection(LEAKAGE_COLS)
        assert not overlap, f"Leakage fields in feature list: {overlap}"

    def test_deterministic_sorting(self):
        # Verify columns are sorted as defined in STABLE_COLUMN_ORDER
        df = extract_dataset()
        cols = list(df.columns)
        # Check order is stable and matches subset of STABLE_COLUMN_ORDER
        from ml.src.extract_dataset import STABLE_COLUMN_ORDER
        expected = [c for c in STABLE_COLUMN_ORDER if c in cols]
        assert cols == expected

    def test_pipeline_csv_creation(self):
        df = extract_dataset()
        save_dataset(df)
        from ml.src.extract_dataset import OUTPUT_FILE
        assert OUTPUT_FILE.exists()

