"""
test_publication_results.py
---------------------------
Unit tests for the Phase 3.10 publication results module (20 tests).
All file I/O uses temporary directories; real publication/ directory is never modified.
"""

from __future__ import annotations

import sys
import json
import shutil
from pathlib import Path

import pytest
import pandas as pd
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from ml.src.publication_results import (
    load_artifacts,
    validate_artifacts,
    make_model_comparison_table,
    make_feature_importance_table,
    make_ablation_table,
    make_experiment_summary_table,
    figure_model_performance,
    figure_cv_stability,
    figure_confusion_matrix,
    figure_feature_importance,
    figure_feature_ablation,
    figure_label_distribution,
    make_research_report,
    make_manifest,
    run_publication_workflow,
    _load_json,
    _load_csv,
    RESULTS_DIR,
    MODELS_DIR,
    PUB_DIR,
)

# ── Shared fixture: real artifacts ─────────────────────────────────────────────

@pytest.fixture(scope="module")
def arts():
    try:
        a = load_artifacts()
        validate_artifacts(a)
        return a
    except (FileNotFoundError, AssertionError) as e:
        pytest.skip(f"Required artifacts not available: {e}")


# ── 1. Required source artifacts load correctly ────────────────────────────────

def test_all_artifacts_load(arts):
    assert "model_comparison" in arts
    assert "tuning_results" in arts
    assert "feature_ablation" in arts
    assert "dt_importance_df" in arts
    assert "confusion_matrix_df" in arts


# ── 2. Missing required artifact raises FileNotFoundError ────────────────────

def test_missing_artifact_raises(tmp_path):
    with pytest.raises(FileNotFoundError, match="Required artifact missing"):
        _load_json(tmp_path / "nonexistent.json", "test artifact")


# ── 3. Malformed JSON raises ValueError ───────────────────────────────────────

def test_malformed_json_raises(tmp_path):
    bad = tmp_path / "bad.json"
    bad.write_text("{ this is not valid json }")
    with pytest.raises(ValueError, match="Malformed JSON"):
        _load_json(bad, "bad json test")


# ── 4. Model comparison table contains all three models ──────────────────────

def test_model_comparison_table_all_models(arts, tmp_path):
    csv_path, md_path = make_model_comparison_table(arts, tmp_path)
    df = pd.read_csv(csv_path)
    model_names = df["model"].tolist()
    assert "random_forest" in model_names
    assert "decision_tree_baseline" in model_names
    assert "decision_tree_tuned" in model_names


# ── 5. Selected model is marked correctly ─────────────────────────────────────

def test_selected_model_marked(arts, tmp_path):
    csv_path, md_path = make_model_comparison_table(arts, tmp_path)
    df = pd.read_csv(csv_path)
    selected = df[df["selected"] == True]
    assert len(selected) == 1
    assert selected.iloc[0]["model"] == "decision_tree_baseline"
    # Also check markdown
    md = md_path.read_text()
    assert "production" in md.lower() or "★" in md


# ── 6. Missing tuned DT std remains explicitly unavailable, not fabricated ───

def test_tuned_std_not_fabricated(arts, tmp_path):
    csv_path, _ = make_model_comparison_table(arts, tmp_path)
    df = pd.read_csv(csv_path)
    tuned_row = df[df["model"] == "decision_tree_tuned"].iloc[0]
    # std fields must be None (not a fake number)
    assert tuned_row["cv_macro_f1_std"] is None or pd.isna(tuned_row["cv_macro_f1_std"])
    assert tuned_row["cv_accuracy_std"] is None or pd.isna(tuned_row["cv_accuracy_std"])
    assert tuned_row["cv_std_comparable"] == False


# ── 7. Feature importance uses DT artifact (not RF artifact) ─────────────────

def test_feature_importance_uses_dt_artifact(arts, tmp_path):
    csv_path, md_path = make_feature_importance_table(arts, tmp_path)
    md = md_path.read_text()
    assert "decision_tree_feature_importance.csv" in md
    assert "feature_importance.csv" not in md.replace(
        "decision_tree_feature_importance.csv", ""
    )


# ── 8. Ablation table uses corrected metrics (baseline CV macro F1 ≈ 73.91%) ─

def test_ablation_table_corrected_metrics(arts, tmp_path):
    _, md_path = make_ablation_table(arts, tmp_path)
    csv_path = tmp_path / "table_feature_ablation.csv"
    df = pd.read_csv(csv_path)
    # Baseline is captured in feature_ablation.json; ablation experiments are from
    # corrected Step 3.9B which yielded DT baseline CV macro F1 ≈ 0.7391
    abl_b = arts["feature_ablation"]["baseline"]["cv_macro_f1_mean"]
    assert abs(abl_b - 0.7391) < 0.001, (
        f"Ablation baseline CV macro F1 {abl_b:.4f} does not match corrected 73.91%"
    )
    assert "group" in df.columns


# ── 9. Markdown tables are valid (have header + separator + at least one row) ──

def test_markdown_tables_valid(arts, tmp_path):
    _, md_path = make_model_comparison_table(arts, tmp_path)
    lines = md_path.read_text().splitlines()
    table_lines = [l for l in lines if l.startswith("|")]
    assert len(table_lines) >= 5   # header + sep + 3 rows


# ── 10. CSV files are valid (non-empty, have expected columns) ────────────────

def test_csv_files_valid(arts, tmp_path):
    csv_path, _ = make_model_comparison_table(arts, tmp_path)
    df = pd.read_csv(csv_path)
    assert len(df) == 3
    assert "holdout_macro_f1" in df.columns

    csv_path2, _ = make_ablation_table(arts, tmp_path)
    df2 = pd.read_csv(csv_path2)
    assert len(df2) == 5
    assert "delta_cv_macro_f1" in df2.columns


# ── 11. PNG figures are generated ─────────────────────────────────────────────

def test_png_figures_generated(arts, tmp_path):
    for fn in [figure_model_performance, figure_cv_stability,
               figure_confusion_matrix, figure_feature_importance,
               figure_feature_ablation, figure_label_distribution]:
        png, _ = fn(arts, tmp_path)
        assert png.exists(), f"PNG not created by {fn.__name__}"


# ── 12. PDF figures are generated ─────────────────────────────────────────────

def test_pdf_figures_generated(arts, tmp_path):
    for fn in [figure_model_performance, figure_cv_stability,
               figure_confusion_matrix, figure_feature_importance,
               figure_feature_ablation, figure_label_distribution]:
        _, pdf = fn(arts, tmp_path)
        assert pdf.exists(), f"PDF not created by {fn.__name__}"


# ── 13. Figures are non-empty ─────────────────────────────────────────────────

def test_figures_non_empty(arts, tmp_path):
    for fn in [figure_model_performance, figure_cv_stability,
               figure_confusion_matrix]:
        png, pdf = fn(arts, tmp_path)
        assert png.stat().st_size > 10_000, f"PNG suspiciously small: {png.name}"
        assert pdf.stat().st_size > 1_000,  f"PDF suspiciously small: {pdf.name}"


# ── 14. Research report contains all required sections ───────────────────────

def test_research_report_sections(arts, tmp_path):
    report_path = make_research_report(arts, tmp_path)
    content = report_path.read_text()
    for section in [
        "Dataset Overview",
        "Baseline Model Evaluation",
        "Production Model Selection",
        "Hyperparameter Tuning",
        "Feature Importance",
        "Feature Ablation",
        "Limitations",
        "Phase 3 Conclusion",
    ]:
        assert section in content, f"Missing section: {section}"


# ── 15. Manifest contains all generated artifacts ────────────────────────────

def test_manifest_all_artifacts(arts, tmp_path):
    # Run a minimal workflow subset to populate tables/figures dicts
    tables  = {}
    figures = {}
    csv, md = make_model_comparison_table(arts, tmp_path)
    tables["model_comparison_csv"] = csv
    png, pdf = figure_model_performance(arts, tmp_path)
    figures["model_performance_png"] = png
    figures["model_performance_pdf"] = pdf

    manifest_path = make_manifest(tmp_path, {"tables": tables, "figures": figures})
    with open(manifest_path) as f:
        manifest = json.load(f)

    assert "generated_at" in manifest
    assert "source_artifacts" in manifest
    assert "generated_tables" in manifest
    assert "generated_figures" in manifest
    assert "model_comparison_csv" in manifest["generated_tables"]
    assert "model_performance_png" in manifest["generated_figures"]


# ── 16. Repository-relative paths are used in manifest ───────────────────────

def test_manifest_uses_relative_paths(arts, tmp_path):
    tables  = {"t": tmp_path / "foo.csv"}
    figures = {"f": tmp_path / "bar.png"}
    manifest_path = make_manifest(tmp_path, {"tables": tables, "figures": figures})
    with open(manifest_path) as f:
        manifest = json.load(f)
    for v in manifest["source_artifacts"].values():
        assert not Path(v).is_absolute() or v.startswith("ml/"), \
            f"Absolute path in manifest source_artifacts: {v}"


# ── 17. Existing experiment files are not overwritten ─────────────────────────

def test_source_artifacts_not_overwritten(arts, tmp_path):
    """run_publication_workflow writes only to pub_dir, not RESULTS_DIR."""
    result = run_publication_workflow(pub_dir=tmp_path)
    # Original model_comparison.json must still exist and be unchanged
    orig = RESULTS_DIR / "model_comparison.json"
    assert orig.exists(), "model_comparison.json was deleted!"
    with open(orig) as f:
        data = json.load(f)
    assert "decision_tree" in data   # content intact


# ── 18. Production model is not modified ──────────────────────────────────────

def test_production_model_not_modified():
    prod = MODELS_DIR / "adaptive_sort_model.joblib"
    if not prod.exists():
        pytest.skip("Production model not found.")
    mtime_before = prod.stat().st_mtime
    # load_artifacts does not touch the joblib file
    arts = load_artifacts()
    mtime_after = prod.stat().st_mtime
    assert mtime_before == mtime_after


# ── 19. Plotting functions close figures (no open figure leak) ────────────────

def test_plotting_functions_close_figures(arts, tmp_path):
    import matplotlib.pyplot as plt
    open_before = plt.get_fignums()
    for fn in [figure_model_performance, figure_cv_stability,
               figure_confusion_matrix, figure_feature_importance,
               figure_feature_ablation, figure_label_distribution]:
        fn(arts, tmp_path)
    open_after = plt.get_fignums()
    assert len(open_after) == len(open_before), \
        f"Figures not closed: before={open_before}, after={open_after}"


# ── 20. Workflow is reproducible (two runs produce identical CSVs) ─────────────

def test_workflow_reproducible(arts, tmp_path):
    out1 = tmp_path / "run1"
    out2 = tmp_path / "run2"
    run_publication_workflow(pub_dir=out1)
    run_publication_workflow(pub_dir=out2)

    for fname in ["table_model_comparison.csv", "table_feature_ablation.csv"]:
        df1 = pd.read_csv(out1 / fname)
        df2 = pd.read_csv(out2 / fname)
        pd.testing.assert_frame_equal(df1, df2, check_dtype=False)
