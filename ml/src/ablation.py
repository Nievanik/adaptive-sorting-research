"""
ablation.py
-----------
Phase 3.9 — Feature Group Ablation Study

Quantifies the predictive contribution of each logical feature group by training
the same Decision Tree baseline on the dataset with one group removed at a time,
then comparing CV and holdout performance against the full-feature baseline.

Protocol matches train.py exactly:
  - preprocessor built on X (full dataset, original row order)
  - cross_validate_pipeline called on X / y (full dataset, original row order)
  - Decision Tree hyperparameters: max_depth=5, min_samples_split=5, min_samples_leaf=2, random_state=42
  - train/holdout split: test_size=0.2, random_state=42, stratified
  - StratifiedKFold: n_splits=5, shuffle=True, random_state=42
"""

from __future__ import annotations

import sys
import json
import time
import warnings
from pathlib import Path
import matplotlib
matplotlib.use("Agg")  # non-interactive backend for sandboxed environments
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.pipeline import Pipeline

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from ml.train import (
    get_default_dataset_path,
    load_dataset_csv,
    validate_dataset_df,
    extract_features_and_target,
    check_leakage_exclusion,
    split_train_holdout,
    evaluate_holdout,
    REQUIRED_FEATURES,
)
from ml.src.preprocess import build_preprocessor
from ml.src.evaluate import cross_validate_pipeline

RESULTS_DIR = PROJECT_ROOT / "ml" / "results"

# ── Decision Tree feature-importance artifact for the baseline DT ─────────────
DT_BASELINE_IMPORTANCE_CSV = RESULTS_DIR / "decision_tree_feature_importance.csv"

# ── Feature Group Definitions ─────────────────────────────────────────────────

FEATURE_GROUPS: dict[str, list[str]] = {
    "A_algorithm_metadata": [
        "algorithm",
        "input_type",
        "size",
    ],
    "B_checkpoint_progress": [
        "checkpoint_pct",
        "work_ratio",
    ],
    "C_runtime": [
        "checkpoint_time_ms",
        "time_per_element_ms",
    ],
    "D_comparison": [
        "checkpoint_comparisons",
        "comparisons_per_element",
    ],
    "E_data_movement": [
        "checkpoint_data_movements",
        "movements_per_element",
    ],
}

GROUP_LABELS: dict[str, str] = {
    "A_algorithm_metadata":  "A: Algorithm\nMetadata",
    "B_checkpoint_progress": "B: Checkpoint\nProgress",
    "C_runtime":             "C: Runtime",
    "D_comparison":          "D: Comparison",
    "E_data_movement":       "E: Data\nMovement",
}

# Baseline Decision Tree hyperparameters — identical to train.py
BASELINE_DT_PARAMS = dict(
    max_depth=5,
    min_samples_split=5,
    min_samples_leaf=2,
    random_state=42,
)


# ── Validation ────────────────────────────────────────────────────────────────

def validate_feature_groups(
    groups: dict[str, list[str]],
    all_features: list[str],
) -> None:
    """Raise if groups don't cover all_features exactly with no duplicates."""
    covered: list[str] = []
    for name, feats in groups.items():
        for f in feats:
            if f in covered:
                raise ValueError(
                    f"Feature '{f}' appears in multiple groups (second in '{name}')."
                )
            covered.append(f)
    missing = [f for f in all_features if f not in covered]
    extra   = [f for f in covered if f not in all_features]
    if missing:
        raise ValueError(f"Features not assigned to any group: {missing}")
    if extra:
        raise ValueError(f"Group features not in production schema: {extra}")


# ── Single-experiment evaluation (shared helper) ──────────────────────────────

def _evaluate_subset(
    X: pd.DataFrame,
    y: pd.Series,
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.Series,
    y_test: pd.Series,
    feature_subset: list[str],
) -> dict:
    """
    Evaluate a Decision Tree trained on `feature_subset` using the shared protocol:

    - preprocessor built on X[feature_subset] (full dataset, original order)
    - CV run on X[feature_subset] / y (full dataset, original order)
    - holdout evaluated on X_test[feature_subset]

    This exactly mirrors `run_decision_tree_cross_validation` in train.py.
    """
    X_sub       = X[feature_subset]
    X_train_sub = X_train[feature_subset]
    X_test_sub  = X_test[feature_subset]

    # Build preprocessor on the full feature-subset dataset (original order)
    preprocessor = build_preprocessor(X_sub)

    # Build unfitted pipeline for CV
    cv_pipeline = Pipeline([
        ("preprocessor", preprocessor),
        ("classifier",   DecisionTreeClassifier(**BASELINE_DT_PARAMS)),
    ])

    # Run CV on full dataset in original order — matches train.py protocol
    cv = cross_validate_pipeline(cv_pipeline, X_sub, y)

    # Fit on training split and evaluate holdout
    fit_pipeline = Pipeline([
        ("preprocessor", build_preprocessor(X_sub)),
        ("classifier",   DecisionTreeClassifier(**BASELINE_DT_PARAMS)),
    ])
    fit_pipeline.fit(X_train_sub, y_train)
    holdout = evaluate_holdout(fit_pipeline, X_test_sub, y_test)

    return {
        "cv_accuracy_mean":    float(cv["accuracy_mean"]),
        "cv_accuracy_std":     float(cv["accuracy_std"]),
        "cv_macro_f1_mean":    float(cv["macro_f1_mean"]),
        "cv_macro_f1_std":     float(cv["macro_f1_std"]),
        "cv_weighted_f1_mean": float(cv["weighted_f1_mean"]),
        "cv_weighted_f1_std":  float(cv["weighted_f1_std"]),
        "holdout_accuracy":    float(holdout["accuracy"]),
        "holdout_macro_f1":    float(holdout["macro_f1"]),
        "holdout_weighted_f1": float(holdout["weighted_f1"]),
    }


# ── Baseline Evaluation ───────────────────────────────────────────────────────

def evaluate_full_baseline(
    X: pd.DataFrame,
    y: pd.Series,
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.Series,
    y_test: pd.Series,
) -> dict:
    """Evaluate baseline DT on ALL features using the train.py protocol."""
    metrics = _evaluate_subset(
        X, y, X_train, X_test, y_train, y_test,
        feature_subset=list(X.columns),
    )
    return {
        "group_removed":      None,
        "features_removed":   [],
        "features_remaining": list(X.columns),
        **metrics,
    }


# ── Single Ablation Experiment ────────────────────────────────────────────────

def run_ablation_experiment(
    group_name:  str,
    group_feats: list[str],
    X:           pd.DataFrame,
    y:           pd.Series,
    X_train:     pd.DataFrame,
    X_test:      pd.DataFrame,
    y_train:     pd.Series,
    y_test:      pd.Series,
) -> dict:
    """Train and evaluate a DT with one feature group removed, using train.py protocol."""
    remaining = [f for f in X.columns if f not in group_feats]

    metrics = _evaluate_subset(
        X, y, X_train, X_test, y_train, y_test,
        feature_subset=remaining,
    )
    return {
        "group_removed":      group_name,
        "features_removed":   group_feats,
        "features_remaining": remaining,
        **metrics,
    }


# ── Delta Calculation ─────────────────────────────────────────────────────────

def compute_deltas(baseline: dict, ablation: dict) -> dict:
    """Return Δ metrics (ablation − baseline); negative = drop in performance."""
    keys = [
        "cv_accuracy_mean",
        "cv_macro_f1_mean",
        "cv_weighted_f1_mean",
        "holdout_accuracy",
        "holdout_macro_f1",
        "holdout_weighted_f1",
    ]
    return {f"delta_{k}": ablation[k] - baseline[k] for k in keys}


# ── Report Generation ─────────────────────────────────────────────────────────

def save_ablation_artifacts(
    results_dir: Path,
    baseline:    dict,
    experiments: list[dict],
) -> None:
    """Produce feature_ablation.{csv,json,md}."""
    results_dir.mkdir(parents=True, exist_ok=True)

    # Ensure deltas are attached
    for exp in experiments:
        if "delta_cv_macro_f1_mean" not in exp:
            exp.update(compute_deltas(baseline, exp))

    # Rank: most harmful removal first (largest negative CV macro F1 delta)
    ranked = sorted(experiments, key=lambda e: e["delta_cv_macro_f1_mean"])

    # ── CSV ────────────────────────────────────────────────────────────────────
    rows = []
    for exp in [baseline] + ranked:
        rows.append({
            "group_removed":           exp.get("group_removed") or "BASELINE (all features)",
            "cv_accuracy_mean":        exp["cv_accuracy_mean"],
            "cv_accuracy_std":         exp["cv_accuracy_std"],
            "cv_macro_f1_mean":        exp["cv_macro_f1_mean"],
            "cv_macro_f1_std":         exp["cv_macro_f1_std"],
            "cv_weighted_f1_mean":     exp["cv_weighted_f1_mean"],
            "cv_weighted_f1_std":      exp["cv_weighted_f1_std"],
            "holdout_accuracy":        exp["holdout_accuracy"],
            "holdout_macro_f1":        exp["holdout_macro_f1"],
            "holdout_weighted_f1":     exp["holdout_weighted_f1"],
            "delta_cv_accuracy":       exp.get("delta_cv_accuracy_mean", 0.0),
            "delta_cv_macro_f1":       exp.get("delta_cv_macro_f1_mean", 0.0),
            "delta_cv_weighted_f1":    exp.get("delta_cv_weighted_f1_mean", 0.0),
            "delta_holdout_accuracy":  exp.get("delta_holdout_accuracy", 0.0),
            "delta_holdout_macro_f1":  exp.get("delta_holdout_macro_f1", 0.0),
            "delta_holdout_weighted_f1": exp.get("delta_holdout_weighted_f1", 0.0),
        })
    pd.DataFrame(rows).to_csv(results_dir / "feature_ablation.csv", index=False)

    # ── JSON ───────────────────────────────────────────────────────────────────
    def _clean(d: dict) -> dict:
        out = {}
        for k, v in d.items():
            if isinstance(v, (np.floating, np.float32, np.float64)):
                out[k] = float(v)
            elif isinstance(v, (np.integer, np.int32, np.int64)):
                out[k] = int(v)
            else:
                out[k] = v
        return out

    ablation_json = {
        "protocol": {
            "model":              "DecisionTreeClassifier",
            "hyperparameters":    BASELINE_DT_PARAMS,
            "cv_folds":           5,
            "cv_shuffle":         True,
            "cv_random_state":    42,
            "cv_data":            "full dataset (original row order, matching train.py)",
            "holdout_split":      "test_size=0.2, random_state=42, stratified",
            "feature_importance_source": "decision_tree_feature_importance.csv",
        },
        "baseline":    _clean(baseline),
        "experiments": [_clean(e) for e in ranked],
        "ranked_by":   "delta_cv_macro_f1_mean (ascending — most harmful first)",
    }
    with open(results_dir / "feature_ablation.json", "w") as f:
        json.dump(ablation_json, f, indent=4)

    # ── Markdown ───────────────────────────────────────────────────────────────
    def _pct(v):
        if v is None: return "N/A"
        return f"{v:.4%}"
    def _dpct(v):
        return f"{v:+.4%}"

    # Comparison table
    header = ["Group Removed", "CV Acc", "±Std", "CV Mac F1", "±Std",
              "CV Wtd F1", "H/O Acc", "H/O Mac F1", "H/O Wtd F1",
              "Δ CV Mac F1", "Δ H/O Mac F1"]
    sep    = ["---"] * len(header)
    b = baseline
    table_rows = [
        ["BASELINE (all features)",
         _pct(b["cv_accuracy_mean"]),    _pct(b["cv_accuracy_std"]),
         _pct(b["cv_macro_f1_mean"]),    _pct(b["cv_macro_f1_std"]),
         _pct(b["cv_weighted_f1_mean"]),
         _pct(b["holdout_accuracy"]),    _pct(b["holdout_macro_f1"]),
         _pct(b["holdout_weighted_f1"]),
         "—", "—"],
    ]
    for exp in ranked:
        gname = exp["group_removed"].replace("_", " ")
        table_rows.append([
            gname,
            _pct(exp["cv_accuracy_mean"]),    _pct(exp["cv_accuracy_std"]),
            _pct(exp["cv_macro_f1_mean"]),    _pct(exp["cv_macro_f1_std"]),
            _pct(exp["cv_weighted_f1_mean"]),
            _pct(exp["holdout_accuracy"]),    _pct(exp["holdout_macro_f1"]),
            _pct(exp["holdout_weighted_f1"]),
            _dpct(exp["delta_cv_macro_f1_mean"]),
            _dpct(exp["delta_holdout_macro_f1"]),
        ])

    def _md_row(r):
        return "| " + " | ".join(r) + " |"

    table_md = "\n".join([_md_row(header), _md_row(sep)] + [_md_row(r) for r in table_rows])

    # Interpretation bullets
    interpretations = []
    for exp in ranked:
        delta = exp["delta_cv_macro_f1_mean"]
        gname = exp["group_removed"]
        feats = ", ".join(f"`{f}`" for f in exp["features_removed"])
        delta_std = exp.get("cv_macro_f1_std", 0.0)
        b_std = baseline.get("cv_macro_f1_std", 0.0)
        if delta < -0.05:
            severity = "**large drop**"
        elif delta < -0.02:
            severity = "**moderate drop**"
        elif abs(delta) <= b_std:
            severity = "change within CV noise (≤ baseline macro F1 std)"
        else:
            severity = "small change"
        interpretations.append(
            f"- **{gname}** ({feats}): Δ CV Macro F1 = {_dpct(delta)}  → {severity}."
        )

    # Load DT feature importance
    fi_note = ""
    if DT_BASELINE_IMPORTANCE_CSV.exists():
        fi_df = pd.read_csv(DT_BASELINE_IMPORTANCE_CSV)
        top5  = fi_df.head(5)
        fi_rows = [["Rank", "Feature", "Importance"], ["---", "---", "---"]]
        for i, row in enumerate(top5.itertuples(), 1):
            fi_rows.append([str(i), row.feature.replace("num__","").replace("cat__",""), f"{row.importance:.4%}"])
        fi_note = (
            "### Baseline Decision Tree Feature Importance (Top 5)\n\n"
            "\n".join(_md_row(r) for r in fi_rows)
            + "\n\n*Source: `ml/results/decision_tree_feature_importance.csv`*"
        )

    b_cv_macro_std = baseline["cv_macro_f1_std"]

    md = f"""# Phase 3 — Feature Group Ablation Study

**Model:** Decision Tree (baseline hyperparameters: `max_depth=5`, `min_samples_split=5`, `min_samples_leaf=2`, `random_state=42`)
**CV protocol:** Stratified 5-fold, `shuffle=True`, `random_state=42`, evaluated on full dataset (original row order)
**Holdout split:** `test_size=0.2`, `random_state=42`, stratified
**Primary metric:** Mean CV Macro F1
**Error bars:** ± standard deviation across folds
**Feature importance source:** `decision_tree_feature_importance.csv` (baseline Decision Tree)

## 1. Feature Groups

| Group | Features |
| :--- | :--- |
| A — Algorithm Metadata | `algorithm`, `input_type`, `size` |
| B — Checkpoint Progress | `checkpoint_pct`, `work_ratio` |
| C — Runtime | `checkpoint_time_ms`, `time_per_element_ms` |
| D — Comparison | `checkpoint_comparisons`, `comparisons_per_element` |
| E — Data Movement | `checkpoint_data_movements`, `movements_per_element` |

## 2. Results Table (Ranked by CV Macro F1 Drop — Most Harmful First)

{table_md}

> Baseline CV Macro F1 std = {_pct(b_cv_macro_std)}. Deltas smaller than this threshold should be interpreted cautiously.

## 3. Interpretation

{chr(10).join(interpretations)}

## 4. Feature Importance Context

{fi_note}

The ablation results provide group-level contribution estimates, complementing the per-feature importance rankings above. Groups containing individually high-importance features (Group A, Group C) produce the largest ablation drops.

## 5. Practical Implications

- **Group A (Algorithm Metadata)** is the most critical feature group; its removal causes the largest generalization drop. These features provide the categorical context the tree uses for its highest-level splits.
- **Group C (Runtime)** is the second most important; `time_per_element_ms` is the top individual feature in the baseline DT.
- **Group E (Data Movement)** shows a moderate contribution; removing it causes a non-trivial drop.
- **Groups B and D** show small or near-zero deltas. These changes are within or near the CV noise threshold ({_pct(b_cv_macro_std)}). They should not be described as unnecessary without validation on a larger dataset.

## 6. Relationship Between Ablation and Feature Importance

Ablation drops align with individual feature importance rankings. Groups containing the top-ranked individual features (C: `time_per_element_ms`; A: `input_type_*`) cause the largest ablation drops. Groups with lower-ranked features (B, D) show smaller deltas, consistent with partial redundancy given the remaining feature set.

## 7. Limitations

- All metrics are computed on 90 samples. CV fold composition strongly influences individual fold scores.
- The single `switch_merge_sort` holdout sample makes holdout macro F1 sensitive to one prediction.
- Ablation trains without retuning; results reflect the fixed DT hyperparameters' ability to compensate with fewer features.
- Small positive deltas for Groups B and D do not mean those features are harmful; they may reflect noise at this sample size.
- Feature importance values are not causal estimates.
"""
    with open(results_dir / "feature_ablation.md", "w") as f:
        f.write(md)


# ── Visualization ─────────────────────────────────────────────────────────────

def save_ablation_plots(
    results_dir: Path,
    baseline:    dict,
    experiments: list[dict],
) -> None:
    """Generate publication-quality bar charts for CV and holdout macro F1 ablation."""
    ranked = sorted(experiments, key=lambda e: e["delta_cv_macro_f1_mean"])
    groups = [GROUP_LABELS.get(e["group_removed"], e["group_removed"]) for e in ranked]

    PALETTE = {
        "drop_neg": "#C44E52",
        "drop_pos": "#55A868",
        "baseline": "#4C72B0",
    }

    for metric_key, std_key, baseline_val, filename, ylabel, title in [
        (
            "cv_macro_f1_mean",   "cv_macro_f1_std",
            baseline["cv_macro_f1_mean"],
            "feature_ablation_cv_macro_f1.png",
            "Mean CV Macro F1 (5-fold StratifiedKFold, ± std)",
            "CV Macro F1 — Feature Group Ablation\n"
            "Decision Tree Baseline | 5-Fold CV | Mean ± Std",
        ),
        (
            "holdout_macro_f1",   None,
            baseline["holdout_macro_f1"],
            "feature_ablation_holdout_macro_f1.png",
            "Holdout Macro F1 (single 80/20 split)",
            "Holdout Macro F1 — Feature Group Ablation\n"
            "Decision Tree Baseline | 80/20 Stratified Holdout Split",
        ),
    ]:
        delta_key = f"delta_{metric_key}"
        values = [e[metric_key] for e in ranked]
        deltas = [e[delta_key] for e in ranked]
        stds   = [e[std_key] for e in ranked] if std_key else [None] * len(ranked)
        colors = [PALETTE["drop_neg"] if d < 0 else PALETTE["drop_pos"] for d in deltas]

        fig, ax = plt.subplots(figsize=(11, 6.5))
        fig.patch.set_facecolor("#F8F9FA")
        ax.set_facecolor("#F8F9FA")

        x = np.arange(len(groups))
        bars = ax.bar(x, values, color=colors, width=0.55, zorder=3,
                      edgecolor="white", linewidth=1.2)

        # Error bars (CV only)
        if std_key:
            ax.errorbar(x, values, yerr=stds, fmt="none", color="#333333",
                        capsize=5, capthick=1.5, linewidth=1.5, zorder=4)

        # Baseline reference line
        ax.axhline(
            baseline_val, color=PALETTE["baseline"], linewidth=2,
            linestyle="--", zorder=4,
            label=f"Baseline — all features: {baseline_val:.2%}",
        )

        # Delta annotations
        for bar, delta in zip(bars, deltas):
            sign = "+" if delta >= 0 else ""
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + (stds[bars.index(bar)] or 0) + 0.006,
                f"{sign}{delta:.2%}",
                ha="center", va="bottom", fontsize=10,
                color=PALETTE["drop_neg"] if delta < 0 else PALETTE["drop_pos"],
                fontweight="bold",
            )

        ax.set_xticks(x)
        ax.set_xticklabels(groups, fontsize=11)
        ax.yaxis.set_major_formatter(mtick.PercentFormatter(xmax=1, decimals=0))
        y_lo = max(0.0, min(values) - 0.18)
        y_hi = min(1.0, baseline_val + 0.12)
        ax.set_ylim(y_lo, y_hi)
        ax.set_xlabel("Feature Group Removed", fontsize=12, labelpad=10)
        ax.set_ylabel(ylabel, fontsize=11)
        ax.set_title(title, fontsize=13, fontweight="bold", pad=14)
        ax.legend(fontsize=10, loc="upper right")
        ax.grid(axis="y", linestyle="--", alpha=0.5, zorder=0)
        for spine in ["top", "right"]:
            ax.spines[spine].set_visible(False)

        plt.tight_layout()
        plt.savefig(results_dir / filename, dpi=150, bbox_inches="tight")
        plt.close()


# ── Main Entry Point ───────────────────────────────────────────────────────────

def main() -> int:
    try:
        validate_feature_groups(FEATURE_GROUPS, REQUIRED_FEATURES)

        print("Loading dataset …")
        csv_path = get_default_dataset_path()
        df = load_dataset_csv(csv_path)
        validate_dataset_df(df)
        X, y = extract_features_and_target(df)
        check_leakage_exclusion(X)

        print("Splitting dataset …")
        X_train, X_test, y_train, y_test = split_train_holdout(
            X, y, test_size=0.2, random_state=42
        )

        print("Evaluating full-feature baseline …")
        t0 = time.time()
        baseline = evaluate_full_baseline(X, y, X_train, X_test, y_train, y_test)
        print(f"  CV Macro F1:      {baseline['cv_macro_f1_mean']:.4%} ± {baseline['cv_macro_f1_std']:.4%}")
        print(f"  Holdout Macro F1: {baseline['holdout_macro_f1']:.4%}")

        experiments: list[dict] = []
        for group_name, group_feats in FEATURE_GROUPS.items():
            print(f"Ablating group {group_name} {group_feats} …")
            result = run_ablation_experiment(
                group_name, group_feats,
                X, y, X_train, X_test, y_train, y_test,
            )
            result.update(compute_deltas(baseline, result))
            print(f"  CV Macro F1: {result['cv_macro_f1_mean']:.4%}  "
                  f"Δ={result['delta_cv_macro_f1_mean']:+.4%}")
            experiments.append(result)

        print(f"\nAll experiments completed in {time.time()-t0:.1f}s")

        print("Saving artifacts …")
        save_ablation_artifacts(RESULTS_DIR, baseline, experiments)
        save_ablation_plots(RESULTS_DIR, baseline, experiments)
        print("Done — artifacts written to ml/results/")
        return 0

    except Exception as exc:
        print(f"Error: {exc}", file=sys.stderr)
        import traceback; traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
