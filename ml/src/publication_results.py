"""
publication_results.py
----------------------
Phase 3.10 — Publication-Ready Tables, Figures, and Results Summary

Loads verified result artifacts, validates required fields, and produces:
  - Tables  (CSV + Markdown)
  - Figures (PNG 300 DPI + vector PDF)
  - Consolidated research report (Markdown)
  - Publication manifest (JSON)

All outputs go to ml/results/publication/.
Source artifacts are never overwritten.
"""

from __future__ import annotations

import sys
import json
import datetime
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.ticker as mtick
import numpy as np
import pandas as pd
import sklearn

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

PUB_DIR     = PROJECT_ROOT / "ml" / "results" / "publication"
RESULTS_DIR = PROJECT_ROOT / "ml" / "results"
MODELS_DIR  = PROJECT_ROOT / "ml" / "models"

# ── Colour palette (colour-blind-safe, consistent throughout) ─────────────────
COLOURS = {
    "rf":      "#4878CF",  # blue   — Random Forest
    "dt":      "#6ACC65",  # green  — Decision Tree baseline
    "tuned":   "#D65F5F",  # red    — Tuned DT
    "neutral": "#B47CC7",  # purple
    "pos":     "#55A868",
    "neg":     "#C44E52",
    "zero":    "#888888",
}
LABEL_SHORT = {
    "continue":             "Continue",
    "switch_insertion_sort":"→ Insertion",
    "switch_merge_sort":    "→ Merge",
    "switch_quick_sort":    "→ Quick",
}
FONT_SIZE   = 11
TITLE_SIZE  = 13
FIG_DPI     = 300


# ═══════════════════════════════════════════════════════════════════════════════
# 1.  ARTIFACT LOADING
# ═══════════════════════════════════════════════════════════════════════════════

def _load_json(path: Path, label: str) -> dict:
    if not path.exists():
        raise FileNotFoundError(f"Required artifact missing: {path}  ({label})")
    with open(path) as f:
        try:
            return json.load(f)
        except json.JSONDecodeError as e:
            raise ValueError(f"Malformed JSON in {path} ({label}): {e}") from e


def _load_csv(path: Path, label: str) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Required artifact missing: {path}  ({label})")
    df = pd.read_csv(path)
    if df.empty:
        raise ValueError(f"Empty CSV artifact: {path}  ({label})")
    return df


def load_artifacts() -> dict:
    """Load and return all required source artifacts."""
    r = RESULTS_DIR
    m = MODELS_DIR
    return {
        "model_comparison":      _load_json(r / "model_comparison.json",       "model comparison"),
        "tuning_results":        _load_json(r / "tuning_results.json",          "tuning results"),
        "bvt_comparison":        _load_json(r / "baseline_vs_tuned_comparison.json", "bvt comparison"),
        "feature_ablation":      _load_json(r / "feature_ablation.json",        "feature ablation"),
        "metadata":              _load_json(m / "adaptive_sort_model_metadata.json", "prod metadata"),
        "classification_report": _load_json(r / "classification_report.json",   "classification report"),
        "confusion_matrix_df":   _load_csv(r / "confusion_matrix.csv",          "confusion matrix"),
        "dt_importance_df":      _load_csv(r / "decision_tree_feature_importance.csv", "DT feature importance"),
        "ablation_df":           _load_csv(r / "feature_ablation.csv",          "ablation metrics"),
    }


# ═══════════════════════════════════════════════════════════════════════════════
# 2.  VALIDATION
# ═══════════════════════════════════════════════════════════════════════════════

def validate_artifacts(arts: dict) -> None:
    """Raise clearly if required metric fields are absent."""
    comp = arts["model_comparison"]
    for model in ("random_forest", "decision_tree"):
        for section in ("holdout", "cv"):
            assert model in comp and section in comp[model], \
                f"model_comparison.json missing [{model}][{section}]"

    tuning = arts["tuning_results"]
    for key in ("best_cv_macro_f1", "best_cv_accuracy", "best_cv_weighted_f1",
                "tuned_holdout_metrics", "best_parameters"):
        assert key in tuning, f"tuning_results.json missing key: {key}"

    ablation = arts["feature_ablation"]
    assert "baseline" in ablation and "experiments" in ablation, \
        "feature_ablation.json missing baseline or experiments"

    fi_df = arts["dt_importance_df"]
    assert "feature" in fi_df.columns and "importance" in fi_df.columns, \
        "decision_tree_feature_importance.csv missing required columns"


# ═══════════════════════════════════════════════════════════════════════════════
# 3.  FORMATTING HELPERS
# ═══════════════════════════════════════════════════════════════════════════════

def _pct2(v) -> str:
    """Format float as percentage to 2 d.p."""
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return "N/A"
    return f"{v * 100:.2f}%"


def _pm(v) -> str:
    """Format std as ± percentage."""
    if v is None:
        return "—"
    return f"± {v * 100:.2f}%"


def _md_table(df: pd.DataFrame) -> str:
    header = "| " + " | ".join(df.columns) + " |"
    sep    = "| " + " | ".join(["---"] * len(df.columns)) + " |"
    rows   = ["| " + " | ".join(str(v) for v in row) + " |"
              for row in df.itertuples(index=False)]
    return "\n".join([header, sep] + rows)


def _clean_feature_name(raw: str) -> str:
    return raw.replace("num__", "").replace("cat__", "")


# ═══════════════════════════════════════════════════════════════════════════════
# 4.  TABLE GENERATORS
# ═══════════════════════════════════════════════════════════════════════════════

def make_model_comparison_table(arts: dict, pub_dir: Path) -> tuple[Path, Path]:
    comp   = arts["model_comparison"]
    tuning = arts["tuning_results"]

    rf  = comp["random_forest"]
    dt  = comp["decision_tree"]

    # Tuned DT — CV std not directly comparable (RandomizedSearchCV best index)
    rows = [
        {
            "Model":             "Random Forest",
            "Selected":          "",
            "H/O Accuracy":      _pct2(rf["holdout"]["accuracy"]),
            "H/O Macro F1":      _pct2(rf["holdout"]["macro_f1"]),
            "H/O Weighted F1":   _pct2(rf["holdout"]["weighted_f1"]),
            "CV Accuracy":       _pct2(rf["cv"]["accuracy_mean"]),
            "CV Acc Std":        _pm(rf["cv"]["accuracy_std"]),
            "CV Macro F1":       _pct2(rf["cv"]["macro_f1_mean"]),
            "CV Mac F1 Std":     _pm(rf["cv"]["macro_f1_std"]),
            "CV Weighted F1":    _pct2(rf["cv"]["weighted_f1_mean"]),
            "CV Wtd F1 Std":     _pm(rf["cv"]["weighted_f1_std"]),
        },
        {
            "Model":             "Decision Tree (baseline) ★",
            "Selected":          "✓ production",
            "H/O Accuracy":      _pct2(dt["holdout"]["accuracy"]),
            "H/O Macro F1":      _pct2(dt["holdout"]["macro_f1"]),
            "H/O Weighted F1":   _pct2(dt["holdout"]["weighted_f1"]),
            "CV Accuracy":       _pct2(dt["cv"]["accuracy_mean"]),
            "CV Acc Std":        _pm(dt["cv"]["accuracy_std"]),
            "CV Macro F1":       _pct2(dt["cv"]["macro_f1_mean"]),
            "CV Mac F1 Std":     _pm(dt["cv"]["macro_f1_std"]),
            "CV Weighted F1":    _pct2(dt["cv"]["weighted_f1_mean"]),
            "CV Wtd F1 Std":     _pm(dt["cv"]["weighted_f1_std"]),
        },
        {
            "Model":             "Decision Tree (tuned) †",
            "Selected":          "",
            "H/O Accuracy":      _pct2(tuning["tuned_holdout_metrics"]["accuracy"]),
            "H/O Macro F1":      _pct2(tuning["tuned_holdout_metrics"]["macro_f1"]),
            "H/O Weighted F1":   _pct2(tuning["tuned_holdout_metrics"]["weighted_f1"]),
            "CV Accuracy":       _pct2(tuning["best_cv_accuracy"]),
            "CV Acc Std":        "— †",
            "CV Macro F1":       _pct2(tuning["best_cv_macro_f1"]),
            "CV Mac F1 Std":     "— †",
            "CV Weighted F1":    _pct2(tuning["best_cv_weighted_f1"]),
            "CV Wtd F1 Std":     "— †",
        },
    ]
    df_md = pd.DataFrame(rows)

    # Raw CSV (decimals)
    raw_rows = [
        {"model": "random_forest",
         "selected": False,
         "holdout_accuracy": rf["holdout"]["accuracy"],
         "holdout_macro_f1": rf["holdout"]["macro_f1"],
         "holdout_weighted_f1": rf["holdout"]["weighted_f1"],
         "cv_accuracy_mean": rf["cv"]["accuracy_mean"],
         "cv_accuracy_std": rf["cv"]["accuracy_std"],
         "cv_macro_f1_mean": rf["cv"]["macro_f1_mean"],
         "cv_macro_f1_std": rf["cv"]["macro_f1_std"],
         "cv_weighted_f1_mean": rf["cv"]["weighted_f1_mean"],
         "cv_weighted_f1_std": rf["cv"]["weighted_f1_std"],
         "cv_std_comparable": True},
        {"model": "decision_tree_baseline",
         "selected": True,
         "holdout_accuracy": dt["holdout"]["accuracy"],
         "holdout_macro_f1": dt["holdout"]["macro_f1"],
         "holdout_weighted_f1": dt["holdout"]["weighted_f1"],
         "cv_accuracy_mean": dt["cv"]["accuracy_mean"],
         "cv_accuracy_std": dt["cv"]["accuracy_std"],
         "cv_macro_f1_mean": dt["cv"]["macro_f1_mean"],
         "cv_macro_f1_std": dt["cv"]["macro_f1_std"],
         "cv_weighted_f1_mean": dt["cv"]["weighted_f1_mean"],
         "cv_weighted_f1_std": dt["cv"]["weighted_f1_std"],
         "cv_std_comparable": True},
        {"model": "decision_tree_tuned",
         "selected": False,
         "holdout_accuracy": tuning["tuned_holdout_metrics"]["accuracy"],
         "holdout_macro_f1": tuning["tuned_holdout_metrics"]["macro_f1"],
         "holdout_weighted_f1": tuning["tuned_holdout_metrics"]["weighted_f1"],
         "cv_accuracy_mean": tuning["best_cv_accuracy"],
         "cv_accuracy_std": None,
         "cv_macro_f1_mean": tuning["best_cv_macro_f1"],
         "cv_macro_f1_std": None,
         "cv_weighted_f1_mean": tuning["best_cv_weighted_f1"],
         "cv_weighted_f1_std": None,
         "cv_std_comparable": False},
    ]
    csv_path = pub_dir / "table_model_comparison.csv"
    pd.DataFrame(raw_rows).to_csv(csv_path, index=False)

    note = (
        "★ = selected production model (criterion: CV macro F1)\n"
        "† = Tuned DT CV std omitted: RandomizedSearchCV reports best-index score, "
        "not a full k-fold re-run, making std values non-comparable with baseline std."
    )
    md_path = pub_dir / "table_model_comparison.md"
    md_path.write_text(
        "# Model Comparison Table\n\n"
        + _md_table(df_md)
        + f"\n\n_{note}_\n"
    )
    return csv_path, md_path


def make_feature_importance_table(arts: dict, pub_dir: Path) -> tuple[Path, Path]:
    fi_df = arts["dt_importance_df"].copy()
    fi_df = fi_df[fi_df["importance"] > 0].reset_index(drop=True)
    fi_df.insert(0, "rank", range(1, len(fi_df) + 1))
    fi_df["display_name"] = fi_df["feature"].apply(_clean_feature_name)
    fi_df["importance_pct"] = fi_df["importance"].apply(_pct2)

    csv_path = pub_dir / "table_feature_importance.csv"
    fi_df[["rank", "feature", "display_name", "importance"]].to_csv(csv_path, index=False)

    display_df = fi_df[["rank", "display_name", "importance_pct"]].rename(columns={
        "rank": "Rank", "display_name": "Feature", "importance_pct": "Importance"
    })
    md_path = pub_dir / "table_feature_importance.md"
    md_path.write_text(
        "# Decision Tree Baseline — Feature Importance\n\n"
        + _md_table(display_df)
        + "\n\n_Source: `ml/results/decision_tree_feature_importance.csv`. "
        "Impurity-based importance reflects model usage and does not establish causation._\n"
    )
    return csv_path, md_path


def make_ablation_table(arts: dict, pub_dir: Path) -> tuple[Path, Path]:
    abl   = arts["feature_ablation"]
    exps  = abl["experiments"]   # already ranked most-harmful first

    GROUP_DISPLAY = {
        "A_algorithm_metadata":  "A — Algorithm Metadata",
        "B_checkpoint_progress": "B — Checkpoint Progress",
        "C_runtime":             "C — Runtime",
        "D_comparison":          "D — Comparison",
        "E_data_movement":       "E — Data Movement",
    }
    baseline_std = abl["baseline"]["cv_macro_f1_std"]

    def _interp(delta: float) -> str:
        if delta < -0.10:
            return "strong measured contribution"
        elif delta < -0.025:
            return "moderate measured contribution"
        elif delta < 0.0:
            return "small measured contribution"
        else:
            return "no measurable unique contribution under current conditions"

    rows_csv = []
    rows_md  = []
    for rank, exp in enumerate(exps, 1):
        gname  = exp["group_removed"]
        delta  = exp["delta_cv_macro_f1_mean"]
        feats  = ", ".join(exp["features_removed"])
        interp = _interp(delta)
        rows_csv.append({
            "rank":                 rank,
            "group":                gname,
            "features_removed":     feats,
            "cv_macro_f1_mean":     exp["cv_macro_f1_mean"],
            "cv_macro_f1_std":      exp["cv_macro_f1_std"],
            "delta_cv_macro_f1":    delta,
            "cv_weighted_f1_mean":  exp["cv_weighted_f1_mean"],
            "cv_accuracy_mean":     exp["cv_accuracy_mean"],
            "holdout_macro_f1":     exp["holdout_macro_f1"],
            "holdout_accuracy":     exp["holdout_accuracy"],
            "interpretation":       interp,
        })
        rows_md.append({
            "Rank":         str(rank),
            "Group":        GROUP_DISPLAY.get(gname, gname),
            "Features":     feats,
            "CV Mac F1":    _pct2(exp["cv_macro_f1_mean"]),
            "CV Mac F1 Std":_pm(exp["cv_macro_f1_std"]),
            "Δ CV Mac F1":  f"{delta * 100:+.2f}%",
            "CV Wtd F1":    _pct2(exp["cv_weighted_f1_mean"]),
            "H/O Mac F1":   _pct2(exp["holdout_macro_f1"]),
            "Interpretation": interp,
        })

    csv_path = pub_dir / "table_feature_ablation.csv"
    pd.DataFrame(rows_csv).to_csv(csv_path, index=False)

    md_path = pub_dir / "table_feature_ablation.md"
    md_path.write_text(
        "# Feature Group Ablation Study Results\n\n"
        + _md_table(pd.DataFrame(rows_md))
        + f"\n\n_Baseline CV macro F1 std = {_pm(baseline_std)}. "
        "Only the Group A drop exceeds the baseline CV macro F1 standard deviation. "
        "All other differences should be interpreted cautiously._\n"
    )
    return csv_path, md_path


def make_experiment_summary_table(arts: dict, pub_dir: Path) -> tuple[Path, Path]:
    comp    = arts["model_comparison"]
    tuning  = arts["tuning_results"]
    abl_b   = arts["feature_ablation"]["baseline"]

    rows = [
        {
            "Experiment":     "Random Forest Baseline",
            "Objective":      "Establish RF classification baseline",
            "Model":          "RandomForestClassifier (default)",
            "Primary Metric": "CV macro F1",
            "Main Result":    _pct2(comp["random_forest"]["cv"]["macro_f1_mean"]),
            "Decision":       "Not selected; lower CV macro F1 than DT",
            "Artifact":       "ml/results/model_comparison.json",
        },
        {
            "Experiment":     "Decision Tree Baseline",
            "Objective":      "Establish DT classification baseline",
            "Model":          "DecisionTreeClassifier (max_depth=5, min_samples_split=5, min_samples_leaf=2)",
            "Primary Metric": "CV macro F1",
            "Main Result":    _pct2(comp["decision_tree"]["cv"]["macro_f1_mean"]),
            "Decision":       "Selected as production model",
            "Artifact":       "ml/results/model_comparison.json",
        },
        {
            "Experiment":     "DT Hyperparameter Tuning",
            "Objective":      "Test whether tuning improves generalization",
            "Model":          "DecisionTreeClassifier (RandomizedSearchCV, 50 candidates)",
            "Primary Metric": "CV macro F1",
            "Main Result":    _pct2(tuning["best_cv_macro_f1"]),
            "Decision":       "Baseline retained; tuned CV macro F1 below baseline by 0.59 pp",
            "Artifact":       "ml/results/tuning_results.json",
        },
        {
            "Experiment":     "Baseline vs Tuned Comparison",
            "Objective":      "Determine whether tuned DT replaces baseline in production",
            "Model":          "Both DT variants",
            "Primary Metric": "CV macro F1",
            "Main Result":    "Baseline leads by 0.59 pp (above 0.5 pp tolerance)",
            "Decision":       "Baseline retained; production model unchanged",
            "Artifact":       "ml/results/baseline_vs_tuned_comparison.json",
        },
        {
            "Experiment":     "Feature Group Ablation",
            "Objective":      "Quantify predictive contribution of each feature group",
            "Model":          "DecisionTreeClassifier (baseline params)",
            "Primary Metric": "Δ CV macro F1",
            "Main Result":    "Group A removal: −17.33 pp; Group D removal: 0.00 pp",
            "Decision":       "Algorithm metadata is most critical; comparison features show no unique contribution",
            "Artifact":       "ml/results/feature_ablation.json",
        },
    ]
    df   = pd.DataFrame(rows)
    csv_path = pub_dir / "table_experiment_summary.csv"
    df.to_csv(csv_path, index=False)
    md_path = pub_dir / "table_experiment_summary.md"
    md_path.write_text("# Phase 3 Experiment Summary\n\n" + _md_table(df) + "\n")
    return csv_path, md_path


# ═══════════════════════════════════════════════════════════════════════════════
# 5.  FIGURE GENERATORS
# ═══════════════════════════════════════════════════════════════════════════════

def _save_fig(fig: plt.Figure, stem: str, pub_dir: Path) -> tuple[Path, Path]:
    png = pub_dir / f"{stem}.png"
    pdf = pub_dir / f"{stem}.pdf"
    fig.savefig(png, dpi=FIG_DPI, bbox_inches="tight")
    fig.savefig(pdf, bbox_inches="tight")
    plt.close(fig)
    return png, pdf


def figure_model_performance(arts: dict, pub_dir: Path) -> tuple[Path, Path]:
    comp   = arts["model_comparison"]
    tuning = arts["tuning_results"]

    models  = ["Random Forest", "DT Baseline ★", "DT Tuned"]
    metrics = {
        "Holdout Macro F1": [
            comp["random_forest"]["holdout"]["macro_f1"],
            comp["decision_tree"]["holdout"]["macro_f1"],
            tuning["tuned_holdout_metrics"]["macro_f1"],
        ],
        "CV Macro F1 (mean)": [
            comp["random_forest"]["cv"]["macro_f1_mean"],
            comp["decision_tree"]["cv"]["macro_f1_mean"],
            tuning["best_cv_macro_f1"],
        ],
        "CV Weighted F1 (mean)": [
            comp["random_forest"]["cv"]["weighted_f1_mean"],
            comp["decision_tree"]["cv"]["weighted_f1_mean"],
            tuning["best_cv_weighted_f1"],
        ],
    }

    n_models  = len(models)
    n_metrics = len(metrics)
    x = np.arange(n_metrics)
    width = 0.25
    colours_list = [COLOURS["rf"], COLOURS["dt"], COLOURS["tuned"]]

    fig, ax = plt.subplots(figsize=(11, 6))
    fig.patch.set_facecolor("#F9F9F9")
    ax.set_facecolor("#F9F9F9")

    for i, (model_name, colour) in enumerate(zip(models, colours_list)):
        vals = [list(metrics.values())[j][i] for j in range(n_metrics)]
        bars = ax.bar(x + (i - 1) * width, vals, width, label=model_name,
                      color=colour, edgecolor="white", linewidth=0.8, zorder=3)
        for bar, val in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.005,
                    f"{val*100:.1f}%",
                    ha="center", va="bottom", fontsize=8.5, color="#333333")

    ax.set_xticks(x)
    ax.set_xticklabels(list(metrics.keys()), fontsize=FONT_SIZE)
    ax.yaxis.set_major_formatter(mtick.PercentFormatter(xmax=1, decimals=0))
    ax.set_ylim(0, 1.12)
    ax.set_ylabel("Score", fontsize=FONT_SIZE)
    ax.set_title(
        "Model Performance Comparison — Adaptive Sorting Decision Classifier\n"
        "★ Decision Tree Baseline selected on CV Macro F1 (not holdout performance alone)",
        fontsize=TITLE_SIZE, fontweight="bold", pad=12,
    )
    ax.legend(fontsize=FONT_SIZE, loc="upper right")
    ax.grid(axis="y", linestyle="--", alpha=0.4, zorder=0)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    plt.tight_layout()
    return _save_fig(fig, "figure_model_performance", pub_dir)


def figure_cv_stability(arts: dict, pub_dir: Path) -> tuple[Path, Path]:
    comp = arts["model_comparison"]

    models = ["Random Forest", "DT Baseline ★"]
    cv_keys = [
        ("CV Accuracy",    "accuracy_mean",    "accuracy_std"),
        ("CV Macro F1",    "macro_f1_mean",    "macro_f1_std"),
        ("CV Weighted F1", "weighted_f1_mean", "weighted_f1_std"),
    ]
    data_src = [comp["random_forest"]["cv"], comp["decision_tree"]["cv"]]

    x = np.arange(len(cv_keys))
    width = 0.35
    colours_list = [COLOURS["rf"], COLOURS["dt"]]

    fig, ax = plt.subplots(figsize=(10, 6))
    fig.patch.set_facecolor("#F9F9F9")
    ax.set_facecolor("#F9F9F9")

    for i, (model_name, colour, src) in enumerate(zip(models, colours_list, data_src)):
        means = [src[mk] for _, mk, _ in cv_keys]
        stds  = [src[sk] for _, _, sk in cv_keys]
        bars  = ax.bar(x + (i - 0.5) * width, means, width,
                       label=model_name, color=colour,
                       edgecolor="white", linewidth=0.8, zorder=3)
        ax.errorbar(x + (i - 0.5) * width, means, yerr=stds,
                    fmt="none", color="#333333", capsize=5,
                    capthick=1.5, linewidth=1.5, zorder=4)
        for bar, mean in zip(bars, means):
            ax.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.005,
                    f"{mean*100:.1f}%",
                    ha="center", va="bottom", fontsize=9)

    ax.set_xticks(x)
    ax.set_xticklabels([label for label, _, _ in cv_keys], fontsize=FONT_SIZE)
    ax.yaxis.set_major_formatter(mtick.PercentFormatter(xmax=1, decimals=0))
    ax.set_ylim(0, 1.10)
    ax.set_ylabel("Mean Score (error bars = ± 1 std)", fontsize=FONT_SIZE)
    ax.set_title(
        "Cross-Validation Stability — Random Forest vs Decision Tree Baseline\n"
        "Stratified 5-fold CV | Error bars = ± 1 standard deviation across folds\n"
        "(Tuned DT omitted: CV std not directly comparable due to RandomizedSearchCV protocol)",
        fontsize=TITLE_SIZE - 1, fontweight="bold", pad=12,
    )
    ax.legend(fontsize=FONT_SIZE)
    ax.grid(axis="y", linestyle="--", alpha=0.4, zorder=0)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    plt.tight_layout()
    return _save_fig(fig, "figure_cv_stability", pub_dir)


def figure_confusion_matrix(arts: dict, pub_dir: Path) -> tuple[Path, Path]:
    cm_df = arts["confusion_matrix_df"].set_index(arts["confusion_matrix_df"].columns[0])
    labels = list(cm_df.index)
    cm = cm_df.values.astype(float)
    n  = cm.shape[0]

    # Row-normalised for colour, raw counts as text
    row_sums = cm.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1
    cm_norm = cm / row_sums

    short_labels = [LABEL_SHORT.get(l, l) for l in labels]

    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(cm_norm, cmap="Blues", vmin=0, vmax=1)

    for i in range(n):
        for j in range(n):
            raw = int(cm[i, j])
            pct = f"\n({cm_norm[i,j]*100:.0f}%)" if row_sums[i, 0] > 0 else ""
            colour = "white" if cm_norm[i, j] > 0.6 else "#222222"
            ax.text(j, i, f"{raw}{pct}", ha="center", va="center",
                    fontsize=10, color=colour, fontweight="bold")

    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels(short_labels, fontsize=10, rotation=25, ha="right")
    ax.set_yticklabels(short_labels, fontsize=10)
    ax.set_xlabel("Predicted Label", fontsize=FONT_SIZE, labelpad=8)
    ax.set_ylabel("True Label", fontsize=FONT_SIZE, labelpad=8)
    ax.set_title(
        "Confusion Matrix — Decision Tree Baseline (Holdout Split)\n"
        "Raw counts; colour = row-normalised recall",
        fontsize=TITLE_SIZE - 1, fontweight="bold", pad=10,
    )
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04).set_label(
        "Recall (row-normalised)", fontsize=9
    )
    plt.tight_layout()
    return _save_fig(fig, "figure_confusion_matrix", pub_dir)


def figure_feature_importance(arts: dict, pub_dir: Path) -> tuple[Path, Path]:
    fi_df = arts["dt_importance_df"].copy()
    fi_df = fi_df[fi_df["importance"] > 0].head(10).copy()
    fi_df["display"] = fi_df["feature"].apply(_clean_feature_name)
    fi_df = fi_df.sort_values("importance")  # ascending for horizontal bar

    fig, ax = plt.subplots(figsize=(9, max(5, len(fi_df) * 0.55)))
    bars = ax.barh(fi_df["display"], fi_df["importance"],
                   color=COLOURS["dt"], edgecolor="white", height=0.65, zorder=3)
    for bar, val in zip(bars, fi_df["importance"]):
        ax.text(val + 0.003, bar.get_y() + bar.get_height() / 2,
                f"{val*100:.1f}%", va="center", fontsize=9)

    ax.xaxis.set_major_formatter(mtick.PercentFormatter(xmax=1, decimals=0))
    ax.set_xlabel("Impurity-Based Feature Importance", fontsize=FONT_SIZE)
    ax.set_title(
        "Decision Tree Baseline — Feature Importance (Top 10)\n"
        "Impurity-based importance reflects model usage; does not establish causation.",
        fontsize=TITLE_SIZE - 1, fontweight="bold", pad=10,
    )
    ax.grid(axis="x", linestyle="--", alpha=0.4, zorder=0)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    plt.tight_layout()
    return _save_fig(fig, "figure_feature_importance", pub_dir)


def figure_feature_ablation(arts: dict, pub_dir: Path) -> tuple[Path, Path]:
    abl_json = arts["feature_ablation"]
    exps     = abl_json["experiments"]    # already ranked
    baseline_std = abl_json["baseline"]["cv_macro_f1_std"]

    GROUP_SHORT = {
        "A_algorithm_metadata":  "A: Algorithm\nMetadata",
        "B_checkpoint_progress": "B: Checkpoint\nProgress",
        "C_runtime":             "C: Runtime",
        "D_comparison":          "D: Comparison",
        "E_data_movement":       "E: Data\nMovement",
    }

    labels = [GROUP_SHORT.get(e["group_removed"], e["group_removed"]) for e in exps]
    deltas = [e["delta_cv_macro_f1_mean"] for e in exps]
    colours = [COLOURS["neg"] if d < 0 else COLOURS["pos"] for d in deltas]

    fig, ax = plt.subplots(figsize=(10, 5.5))
    fig.patch.set_facecolor("#F9F9F9")
    ax.set_facecolor("#F9F9F9")

    x = np.arange(len(labels))
    bars = ax.bar(x, deltas, color=colours, width=0.55,
                  edgecolor="white", linewidth=1.0, zorder=3)

    # Zero reference line
    ax.axhline(0, color="#444444", linewidth=1.5, linestyle="-", zorder=4)
    # Baseline std band
    ax.axhspan(-baseline_std, baseline_std, alpha=0.10, color="#888888", zorder=1,
               label=f"Baseline CV macro F1 std band (±{baseline_std*100:.1f} pp)")

    for bar, delta in zip(bars, deltas):
        ypos = delta + 0.004 if delta >= 0 else delta - 0.012
        ax.text(bar.get_x() + bar.get_width() / 2, ypos,
                f"{delta*100:+.2f} pp", ha="center", va="bottom",
                fontsize=9.5, fontweight="bold",
                color=COLOURS["neg"] if delta < 0 else COLOURS["pos"])

    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=FONT_SIZE)
    ax.yaxis.set_major_formatter(mtick.PercentFormatter(xmax=1, decimals=0))
    ax.set_ylabel("Δ CV Macro F1 (feature group removed − baseline)", fontsize=FONT_SIZE)
    ax.set_title(
        "Feature Group Ablation — Δ CV Macro F1\n"
        "Decision Tree Baseline | Ranked most harmful → least harmful\n"
        "Shaded band = baseline CV macro F1 std (≈ ±10.91 pp); small diffs within band are not significant",
        fontsize=TITLE_SIZE - 1, fontweight="bold", pad=10,
    )
    ax.legend(fontsize=9, loc="lower right")
    ax.grid(axis="y", linestyle="--", alpha=0.4, zorder=0)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    plt.tight_layout()
    return _save_fig(fig, "figure_feature_ablation", pub_dir)


def figure_label_distribution(arts: dict, pub_dir: Path) -> tuple[Path, Path]:
    tuning = arts["tuning_results"]
    dist   = tuning["class_distribution"]

    labels = [d["label"] for d in dist]
    counts = [d["count"] for d in dist]
    pcts   = [d["percentage"] for d in dist]
    short  = [LABEL_SHORT.get(l, l) for l in labels]

    fig, ax = plt.subplots(figsize=(8, 5))
    fig.patch.set_facecolor("#F9F9F9")
    ax.set_facecolor("#F9F9F9")

    colours_bar = [COLOURS["neg"] if l == "switch_merge_sort" else COLOURS["dt"]
                   for l in labels]
    bars = ax.bar(short, counts, color=colours_bar, edgecolor="white",
                  linewidth=0.8, width=0.6, zorder=3)
    for bar, count, pct in zip(bars, counts, pcts):
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.3,
                f"{count}\n({pct:.1f}%)",
                ha="center", va="bottom", fontsize=10)

    ax.set_ylabel("Sample Count", fontsize=FONT_SIZE)
    ax.set_xlabel("Target Class", fontsize=FONT_SIZE, labelpad=8)
    ax.set_title(
        "Target Class Distribution (n = 90)\n"
        "Red bar = switch_merge_sort (underrepresented: 7 / 90 samples, 7.78%)",
        fontsize=TITLE_SIZE - 1, fontweight="bold", pad=10,
    )
    ax.set_ylim(0, max(counts) * 1.2)
    ax.grid(axis="y", linestyle="--", alpha=0.4, zorder=0)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    plt.tight_layout()
    return _save_fig(fig, "figure_label_distribution", pub_dir)


# ═══════════════════════════════════════════════════════════════════════════════
# 6.  RESEARCH REPORT
# ═══════════════════════════════════════════════════════════════════════════════

def make_research_report(arts: dict, pub_dir: Path) -> Path:
    comp    = arts["model_comparison"]
    tuning  = arts["tuning_results"]
    meta    = arts["metadata"]
    abl     = arts["feature_ablation"]
    fi_df   = arts["dt_importance_df"]

    rf = comp["random_forest"]
    dt = comp["decision_tree"]
    abl_b = abl["baseline"]
    exps  = abl["experiments"]

    top3_fi = fi_df[fi_df["importance"] > 0].head(3)
    fi_bullets = "\n".join(
        f"- **{_clean_feature_name(row.feature)}**: {row.importance*100:.1f}%"
        for row in top3_fi.itertuples()
    )

    abl_rows = "\n".join(
        f"| {i+1} | {e['group_removed'].replace('_',' ')} "
        f"| {e['cv_macro_f1_mean']*100:.2f}% "
        f"| {e['delta_cv_macro_f1_mean']*100:+.2f} pp |"
        for i, e in enumerate(exps)
    )

    report = f"""# Phase 3 — Research Results Summary

**Generated:** {datetime.datetime.now(datetime.timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}
**Production model:** `ml/models/adaptive_sort_model.joblib` (Decision Tree Baseline)

---

## 1. Dataset Overview

- **Total samples:** 90
- **Raw features:** 11 (see `REQUIRED_FEATURES` in `ml/train.py`)
- **Transformed features:** {meta.get('transformed_feature_count', 18)} (after one-hot encoding of categorical features)
- **Target classes:** 4 (`continue`, `switch_insertion_sort`, `switch_merge_sort`, `switch_quick_sort`)

### Class Distribution

| Class | Count | Percentage |
| :--- | :--- | :--- |
| continue | 28 | 31.11% |
| switch_insertion_sort | 32 | 35.56% |
| switch_merge_sort | 7 | **7.78%** |
| switch_quick_sort | 23 | 25.56% |

**Limitation:** `switch_merge_sort` is substantially underrepresented (7 of 90 samples). This causes high CV macro F1 variance and makes per-class metrics for this class unreliable.

---

## 2. Baseline Model Evaluation

Both models were trained on the same 80/20 stratified holdout split and evaluated using Stratified 5-fold cross-validation (shuffle=True, random_state=42) on the full dataset.

| Metric | Random Forest | Decision Tree |
| :--- | :--- | :--- |
| Holdout Accuracy | {_pct2(rf['holdout']['accuracy'])} | {_pct2(dt['holdout']['accuracy'])} |
| Holdout Macro F1 | {_pct2(rf['holdout']['macro_f1'])} | {_pct2(dt['holdout']['macro_f1'])} |
| Holdout Weighted F1 | {_pct2(rf['holdout']['weighted_f1'])} | {_pct2(dt['holdout']['weighted_f1'])} |
| CV Accuracy | {_pct2(rf['cv']['accuracy_mean'])} ± {_pm(rf['cv']['accuracy_std'])} | {_pct2(dt['cv']['accuracy_mean'])} ± {_pm(dt['cv']['accuracy_std'])} |
| CV Macro F1 | {_pct2(rf['cv']['macro_f1_mean'])} ± {_pm(rf['cv']['macro_f1_std'])} | {_pct2(dt['cv']['macro_f1_mean'])} ± {_pm(dt['cv']['macro_f1_std'])} |
| CV Weighted F1 | {_pct2(rf['cv']['weighted_f1_mean'])} ± {_pm(rf['cv']['weighted_f1_std'])} | {_pct2(dt['cv']['weighted_f1_mean'])} ± {_pm(dt['cv']['weighted_f1_std'])} |

- The Random Forest achieved stronger holdout performance ({_pct2(rf['holdout']['macro_f1'])} vs {_pct2(dt['holdout']['macro_f1'])} macro F1).
- The Decision Tree achieved stronger CV macro F1 ({_pct2(dt['cv']['macro_f1_mean'])} vs {_pct2(rf['cv']['macro_f1_mean'])}) and CV weighted F1.
- CV accuracy was tied at {_pct2(dt['cv']['accuracy_mean'])}.
- The predefined selection criterion was CV macro F1 as the primary generalization proxy.

---

## 3. Production Model Selection

**Selected model:** Decision Tree Baseline
**Path:** `ml/models/adaptive_sort_model.joblib`

The Decision Tree was selected because it achieved a higher CV macro F1 ({_pct2(dt['cv']['macro_f1_mean'])}) compared to the Random Forest ({_pct2(rf['cv']['macro_f1_mean'])}), a difference of {(dt['cv']['macro_f1_mean'] - rf['cv']['macro_f1_mean'])*100:.2f} percentage points. Holdout performance alone was not used because it represents a single 18-sample evaluation that is insufficient to infer generalization on 90 total samples.

---

## 4. Hyperparameter Tuning

RandomizedSearchCV (50 candidates, 5-fold CV, `f1_macro` scoring) was run on the training split.

**Best parameters:** `criterion=log_loss`, `max_depth=6`, `min_samples_split=5`, `min_samples_leaf=2`, `max_features=sqrt`, `class_weight=balanced`

| | Baseline DT | Tuned DT |
| :--- | :--- | :--- |
| CV Macro F1 | {_pct2(dt['cv']['macro_f1_mean'])} | {_pct2(tuning['best_cv_macro_f1'])} |
| CV Weighted F1 | {_pct2(dt['cv']['weighted_f1_mean'])} | {_pct2(tuning['best_cv_weighted_f1'])} |
| Holdout Macro F1 | {_pct2(dt['holdout']['macro_f1'])} | {_pct2(tuning['tuned_holdout_metrics']['macro_f1'])} |

The tuned model improved holdout macro F1 marginally ({_pct2(tuning['tuned_holdout_metrics']['macro_f1'])} vs {_pct2(dt['holdout']['macro_f1'])}) but degraded CV macro F1 by {(dt['cv']['macro_f1_mean'] - tuning['best_cv_macro_f1'])*100:.2f} pp. Per the predefined priority (CV macro F1 as criterion 1, tolerance 0.005), the baseline was retained.

---

## 5. Feature Importance

Source: `ml/results/decision_tree_feature_importance.csv` (baseline Decision Tree).

Top 3 features by impurity-based importance:

{fi_bullets}

Impurity-based importance reflects how the model allocates decision splits across features. It does not establish causal relationships and is known to overestimate the contribution of high-cardinality features.

---

## 6. Feature Ablation

Protocol: identical Decision Tree baseline hyperparameters; identical train/holdout split; CV evaluated on full dataset in original row order (matching train.py). Primary metric: Δ CV macro F1.

Baseline CV macro F1: {_pct2(abl_b['cv_macro_f1_mean'])} ± {_pm(abl_b['cv_macro_f1_std'])}

| Rank | Group Removed | CV Macro F1 | Δ CV Macro F1 |
| :--- | :--- | :--- | :--- |
{abl_rows}

**Key finding:** Only the Group A (algorithm metadata) removal exceeds the baseline CV macro F1 standard deviation ({_pm(abl_b['cv_macro_f1_std'])}). Groups B–E differences fall within or near the noise floor and should not be interpreted as statistically significant.

---

## 7. Limitations

1. **Dataset size:** 90 samples produce wide CV fold variance (macro F1 std ≈ 10.91 pp).
2. **Minority class:** `switch_merge_sort` has only 7 samples; one misclassification changes its recall by 100%.
3. **CV variability:** A single `StratifiedKFold` configuration was used; no repeated CV was performed.
4. **Single holdout split:** 18 holdout samples cannot provide stable per-class recall estimates.
5. **Impurity-based feature importance:** Known to overestimate importance of high-cardinality or correlated features.
6. **Ablation without retuning:** Ablation experiments use fixed hyperparameters; the model cannot adapt to the reduced feature set.
7. **Practical tolerance:** The 0.005 replacement threshold is an operational convention, not a statistical significance threshold.

---

## 8. Phase 3 Conclusion

The Decision Tree Baseline was selected as the production model on the basis of superior cross-validation macro F1 ({_pct2(dt['cv']['macro_f1_mean'])}). Hyperparameter tuning did not produce a CV improvement that exceeded the defined operational threshold, and the baseline was retained. The ablation study identified algorithm-metadata features (algorithm type, input type, size) as the most predictive group, with a measured CV macro F1 reduction of 17.33 percentage points upon removal. These results are based on a 90-sample dataset and should be treated as preliminary evidence to be validated on a larger benchmark corpus before drawing generalizable conclusions.
"""
    path = pub_dir / "phase3_research_results.md"
    path.write_text(report)
    return path


# ═══════════════════════════════════════════════════════════════════════════════
# 7.  PUBLICATION MANIFEST
# ═══════════════════════════════════════════════════════════════════════════════

def make_manifest(pub_dir: Path, generated: dict) -> Path:
    def _rel(p: Path) -> str:
        try:
            return str(p.relative_to(PROJECT_ROOT))
        except ValueError:
            return str(p)

    manifest = {
        "generated_at":         datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "generator":            "ml/src/publication_results.py",
        "selected_production_model": "ml/models/adaptive_sort_model.joblib",
        "dataset_size":         90,
        "primary_metric":       "CV Macro F1 (Stratified 5-fold)",
        "random_state":         42,
        "test_size":            0.2,
        "cv_folds":             5,
        "scikit_learn_version": sklearn.__version__,
        "source_artifacts":     {k: _rel(Path(v)) for k, v in {
            "model_comparison":      RESULTS_DIR / "model_comparison.json",
            "tuning_results":        RESULTS_DIR / "tuning_results.json",
            "bvt_comparison":        RESULTS_DIR / "baseline_vs_tuned_comparison.json",
            "feature_ablation":      RESULTS_DIR / "feature_ablation.json",
            "confusion_matrix":      RESULTS_DIR / "confusion_matrix.csv",
            "dt_feature_importance": RESULTS_DIR / "decision_tree_feature_importance.csv",
            "prod_metadata":         MODELS_DIR  / "adaptive_sort_model_metadata.json",
        }.items()},
        "generated_tables": {k: _rel(v) for k, v in generated.get("tables", {}).items()},
        "generated_figures": {k: _rel(v) for k, v in generated.get("figures", {}).items()},
        "warnings": [
            "Tuned DT CV standard deviations omitted: RandomizedSearchCV best-index "
            "score is not directly comparable with full stratified k-fold std values.",
            "All metrics are based on 90 samples; results should not be generalised "
            "beyond the tested dataset without further validation.",
        ],
    }
    path = pub_dir / "publication_manifest.json"
    with open(path, "w") as f:
        json.dump(manifest, f, indent=4)
    return path


# ═══════════════════════════════════════════════════════════════════════════════
# 8.  MAIN WORKFLOW
# ═══════════════════════════════════════════════════════════════════════════════

def run_publication_workflow(pub_dir: Path = PUB_DIR) -> dict:
    """Run the complete publication output workflow. Returns paths to all outputs."""
    pub_dir.mkdir(parents=True, exist_ok=True)

    print("Loading source artifacts …")
    arts = load_artifacts()
    validate_artifacts(arts)

    tables  = {}
    figures = {}

    print("Generating tables …")
    csv, md = make_model_comparison_table(arts, pub_dir)
    tables["model_comparison_csv"] = csv
    tables["model_comparison_md"]  = md

    csv, md = make_feature_importance_table(arts, pub_dir)
    tables["feature_importance_csv"] = csv
    tables["feature_importance_md"]  = md

    csv, md = make_ablation_table(arts, pub_dir)
    tables["ablation_csv"] = csv
    tables["ablation_md"]  = md

    csv, md = make_experiment_summary_table(arts, pub_dir)
    tables["experiment_summary_csv"] = csv
    tables["experiment_summary_md"]  = md

    print("Generating figures …")
    for name, fn in [
        ("model_performance",  figure_model_performance),
        ("cv_stability",       figure_cv_stability),
        ("confusion_matrix",   figure_confusion_matrix),
        ("feature_importance", figure_feature_importance),
        ("feature_ablation",   figure_feature_ablation),
        ("label_distribution", figure_label_distribution),
    ]:
        png, pdf = fn(arts, pub_dir)
        figures[f"{name}_png"] = png
        figures[f"{name}_pdf"] = pdf
        print(f"  {png.name}  {pdf.name}")

    print("Generating research report …")
    report = make_research_report(arts, pub_dir)

    print("Generating publication manifest …")
    manifest = make_manifest(pub_dir, {"tables": tables, "figures": figures})

    print(f"Done — all outputs in {pub_dir}")
    return {"tables": tables, "figures": figures,
            "report": report, "manifest": manifest}


if __name__ == "__main__":
    result = run_publication_workflow()
    sys.exit(0)
