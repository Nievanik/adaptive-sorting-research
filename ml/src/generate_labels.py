"""
generate_labels.py
------------------
Determines the best_action label for each row in the checkpoint dataset.

Best action = the candidate (continue OR switch_*) that produces the
lowest total wall-clock time:
    total_time = checkpoint_time_ms + action_time_ms

This module is pure transformation — it imports no sorting algorithm code
and runs no benchmarks.
"""

from __future__ import annotations

import pandas as pd

# All possible action column prefixes that may appear in the dataset.
# An action is present for a row if its corresponding *_time_ms column is non-null.
_POSSIBLE_ACTIONS: list[str] = [
    "continue",
    "switch_insertion_sort",
    "switch_merge_sort",
    "switch_quick_sort",
]


def _remaining_time(row: pd.Series, action: str) -> float | None:
    """Return remaining elapsed time for a given action.

    Returns None if the action column is absent or NaN for this row.
    """
    col = f"{action}_time_ms"
    if col not in row.index:
        return None
    val = row[col]
    if pd.isna(val):
        return None
    return float(val)


def assign_labels(df: pd.DataFrame) -> pd.DataFrame:
    """Assign best_action, best_action_total_ms, and speedup_vs_continue columns.

    Parameters
    ----------
    df : pd.DataFrame
        Flattened dataset produced by extract_dataset.py.  Must contain
        ``checkpoint_time_ms``, ``continue_time_ms``, and any available
        ``switch_*_time_ms`` columns.

    Returns
    -------
    pd.DataFrame
        Same DataFrame with three new columns appended:
        - best_action          : str  — action name with lowest remaining time
        - best_action_total_ms : float — winning total time in ms (checkpoint + remaining)
        - speedup_vs_continue  : float — continue_total / best_total (>=1 means switching wins)
    """
    best_actions: list[str] = []
    best_totals: list[float] = []
    speedups: list[float] = []

    # Deterministic tie-breaking order: continue has highest preference,
    # followed alphabetically/logically by others.
    tie_break_order = ["continue", "switch_quick_sort", "switch_merge_sort", "switch_insertion_sort"]

    for _, row in df.iterrows():
        checkpoint_time = float(row.get("checkpoint_time_ms", 0.0))
        continue_remaining = float(row.get("continue_time_ms", 0.0))
        continue_total = checkpoint_time + continue_remaining

        # Build candidate map: action_name -> remaining_time
        candidates: dict[str, float] = {"continue": continue_remaining}
        for action in _POSSIBLE_ACTIONS:
            if action == "continue":
                continue
            r_time = _remaining_time(row, action)
            if r_time is not None:
                candidates[action] = r_time

        # Find the minimum remaining time with deterministic tie-breaking.
        # We sort candidates by remaining time (ascending) and then by their position in tie_break_order.
        def sort_key(action_name: str) -> tuple[float, int]:
            val = candidates[action_name]
            try:
                prio = tie_break_order.index(action_name)
            except ValueError:
                prio = 999
            return (val, prio)

        best = min(candidates.keys(), key=sort_key)
        best_remaining = candidates[best]
        best_total = checkpoint_time + best_remaining
        speedup = continue_total / best_total if best_total > 0 else 1.0

        best_actions.append(best)
        best_totals.append(best_total)
        speedups.append(speedup)

    df = df.copy()
    df["best_action"] = best_actions
    df["best_action_total_ms"] = best_totals
    df["speedup_vs_continue"] = speedups

    return df



if __name__ == "__main__":
    print("generate_labels.py — import and call assign_labels(df) to use.")
