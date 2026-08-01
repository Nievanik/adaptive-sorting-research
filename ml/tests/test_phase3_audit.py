"""
test_phase3_audit.py
--------------------
Phase 3.11 — Final audit tests (20 tests).

Verifies that the complete Phase 3 artifact set exists, parses, and is internally
consistent without regenerating any expensive artifacts.
"""

from __future__ import annotations

import sys
import json
from pathlib import Path

import pytest
import joblib
import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier

PROJECT_ROOT = Path(__file__).resolve().parents[2]
ML           = PROJECT_ROOT / "ml"
RESULTS      = ML / "results"
PUB          = RESULTS / "publication"
MODELS       = ML / "models"

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from ml.train import REQUIRED_FEATURES, LEAKAGE_AND_OUTCOME_COLS, SUPPORTED_LABELS

REQUIRED_MODELS = [
    MODELS / "adaptive_sort_model.joblib",
    MODELS / "decision_tree_baseline.joblib",
    MODELS / "decision_tree_tuned.joblib",
    MODELS / "random_forest_baseline.joblib",
    MODELS / "adaptive_sort_model_metadata.json",
]

REQUIRED_RESULTS = [
    RESULTS / "classification_report.json",
    RESULTS / "confusion_matrix.csv",
    RESULTS / "feature_importance.csv",
    RESULTS / "decision_tree_feature_importance.csv",
    RESULTS / "baseline_analysis.md",
    RESULTS / "model_comparison.json",
    RESULTS / "model_comparison.md",
    RESULTS / "model_metrics.csv",
    RESULTS / "tuning_results.json",
    RESULTS / "tuning_results.csv",
    RESULTS / "decision_tree_tuning_summary.md",
    RESULTS / "decision_tree_tuned_classification_report.json",
    RESULTS / "decision_tree_tuned_confusion_matrix.csv",
    RESULTS / "decision_tree_tuned_feature_importance.csv",
    RESULTS / "baseline_vs_tuned_comparison.json",
    RESULTS / "baseline_vs_tuned_comparison.md",
    RESULTS / "baseline_vs_tuned_metrics.csv",
    RESULTS / "feature_ablation.json",
    RESULTS / "feature_ablation.csv",
    RESULTS / "feature_ablation.md",
    RESULTS / "feature_ablation_cv_macro_f1.png",
    RESULTS / "feature_ablation_holdout_macro_f1.png",
    RESULTS / "phase3_manifest.json",
]

REQUIRED_PUB = [
    PUB / "table_model_comparison.csv",
    PUB / "table_model_comparison.md",
    PUB / "table_feature_importance.csv",
    PUB / "table_feature_importance.md",
    PUB / "table_feature_ablation.csv",
    PUB / "table_feature_ablation.md",
    PUB / "table_experiment_summary.csv",
    PUB / "table_experiment_summary.md",
    PUB / "figure_model_performance.png",
    PUB / "figure_model_performance.pdf",
    PUB / "figure_cv_stability.png",
    PUB / "figure_cv_stability.pdf",
    PUB / "figure_confusion_matrix.png",
    PUB / "figure_confusion_matrix.pdf",
    PUB / "figure_feature_importance.png",
    PUB / "figure_feature_importance.pdf",
    PUB / "figure_feature_ablation.png",
    PUB / "figure_feature_ablation.pdf",
    PUB / "figure_label_distribution.png",
    PUB / "figure_label_distribution.pdf",
    PUB / "phase3_research_results.md",
    PUB / "publication_manifest.json",
]

# Representative checkpoint records (no outcome fields)
_RECORDS = [
    {"algorithm": "quick_sort",     "input_type": "random",         "size": 1000, "checkpoint_pct": 50,
     "checkpoint_time_ms": 1.5,  "checkpoint_comparisons": 4500, "checkpoint_data_movements": 1800,
     "comparisons_per_element": 4.5, "movements_per_element": 1.8, "work_ratio": 0.42, "time_per_element_ms": 0.0015},
    {"algorithm": "merge_sort",     "input_type": "duplicate_heavy","size": 500,  "checkpoint_pct": 30,
     "checkpoint_time_ms": 0.8,  "checkpoint_comparisons": 2000, "checkpoint_data_movements": 800,
     "comparisons_per_element": 4.0, "movements_per_element": 1.6, "work_ratio": 0.35, "time_per_element_ms": 0.0016},
    {"algorithm": "insertion_sort", "input_type": "nearly_sorted",  "size": 200,  "checkpoint_pct": 20,
     "checkpoint_time_ms": 0.4,  "checkpoint_comparisons": 600,  "checkpoint_data_movements": 200,
     "comparisons_per_element": 3.0, "movements_per_element": 1.0, "work_ratio": 0.20, "time_per_element_ms": 0.0020},
    {"algorithm": "quick_sort",     "input_type": "reverse_sorted", "size": 1000, "checkpoint_pct": 25,
     "checkpoint_time_ms": 3.0,  "checkpoint_comparisons": 8000, "checkpoint_data_movements": 5000,
     "comparisons_per_element": 8.0, "movements_per_element": 5.0, "work_ratio": 0.80, "time_per_element_ms": 0.0030},
]


# ── 1. Required model artifacts exist ─────────────────────────────────────────

def test_required_model_artifacts_exist():
    missing = [p.name for p in REQUIRED_MODELS if not p.exists()]
    assert not missing, f"Missing model artifacts: {missing}"


# ── 2. Required result artifacts exist ────────────────────────────────────────

def test_required_result_artifacts_exist():
    missing = [p.name for p in REQUIRED_RESULTS if not p.exists()]
    assert not missing, f"Missing result artifacts: {missing}"


# ── 3. Required publication artifacts exist ───────────────────────────────────

def test_required_publication_artifacts_exist():
    missing = [p.name for p in REQUIRED_PUB if not p.exists()]
    assert not missing, f"Missing publication artifacts: {missing}"


# ── 4. All JSON files parse ───────────────────────────────────────────────────

def test_all_json_files_parse():
    json_files = (
        list(RESULTS.glob("*.json"))
        + list(PUB.glob("*.json"))
        + list(MODELS.glob("*.json"))
    )
    errors = []
    for jf in json_files:
        try:
            with open(jf) as f:
                json.load(f)
        except json.JSONDecodeError as e:
            errors.append(f"{jf.name}: {e}")
    assert not errors, f"JSON parse errors:\n" + "\n".join(errors)


# ── 5. All CSV files are non-empty ────────────────────────────────────────────

def test_all_csv_files_non_empty():
    csv_files = list(RESULTS.glob("*.csv")) + list(PUB.glob("*.csv"))
    errors = []
    for cf in csv_files:
        df = pd.read_csv(cf)
        if df.empty:
            errors.append(cf.name)
    assert not errors, f"Empty CSVs: {errors}"


# ── 6. All Markdown files are non-empty ───────────────────────────────────────

def test_all_markdown_files_non_empty():
    md_files = (
        list(RESULTS.glob("*.md"))
        + list(PUB.glob("*.md"))
        + [ML / "README.md", ML / "PHASE_3_COMPLETE.md"]
    )
    errors = []
    for mf in md_files:
        if mf.exists() and mf.stat().st_size < 50:
            errors.append(mf.name)
    assert not errors, f"Suspiciously small Markdown files: {errors}"


# ── 7. All PNG and PDF files are non-empty ────────────────────────────────────

def test_all_figures_non_empty():
    pngs = list(RESULTS.glob("*.png")) + list(PUB.glob("*.png"))
    pdfs = list(PUB.glob("*.pdf"))
    errors = []
    for f in pngs:
        if f.stat().st_size < 5_000:
            errors.append(f"PNG too small: {f.name}")
    for f in pdfs:
        if f.stat().st_size < 500:
            errors.append(f"PDF too small: {f.name}")
    assert not errors, "\n".join(errors)


# ── 8. Production model loads ─────────────────────────────────────────────────

def test_production_model_loads():
    prod = joblib.load(MODELS / "adaptive_sort_model.joblib")
    assert hasattr(prod, "predict")
    assert hasattr(prod, "predict_proba")


# ── 9. Production and baseline DT predictions match ──────────────────────────

def test_production_and_baseline_predictions_match():
    prod = joblib.load(MODELS / "adaptive_sort_model.joblib")
    base = joblib.load(MODELS / "decision_tree_baseline.joblib")
    for rec in _RECORDS:
        df = pd.DataFrame([rec])
        pp = prod.predict(df)[0]
        bp = base.predict(df)[0]
        assert pp == bp, f"Prediction mismatch for {rec['algorithm']}/{rec['input_type']}: prod={pp}, base={bp}"
        pp_proba = prod.predict_proba(df)[0]
        bp_proba = base.predict_proba(df)[0]
        assert np.allclose(pp_proba, bp_proba), "Probability mismatch between production and baseline"


# ── 10. Metadata matches the production model ─────────────────────────────────

def test_metadata_matches_production_model():
    with open(MODELS / "adaptive_sort_model_metadata.json") as f:
        meta = json.load(f)
    assert meta["selected_model_name"] == "Decision Tree Baseline"
    assert meta["model_class"] == "sklearn.tree.DecisionTreeClassifier"
    assert meta["decision_tree_hyperparameters"]["max_depth"] == 5
    assert meta["decision_tree_hyperparameters"]["min_samples_split"] == 5
    assert meta["decision_tree_hyperparameters"]["min_samples_leaf"] == 2
    assert meta["random_state"] == 42
    assert meta["dataset_row_count"] == 90
    assert meta["raw_feature_count"] == 11
    assert sorted(meta["ordered_required_feature_list"]) == sorted(REQUIRED_FEATURES)


# ── 11. Manifest paths resolve ────────────────────────────────────────────────

def test_manifest_paths_resolve():
    with open(RESULTS / "phase3_manifest.json") as f:
        manifest = json.load(f)
    all_paths = (
        list(manifest.get("model_artifacts", []))
        + list(manifest.get("result_artifacts", []))
        + list(manifest.get("publication_artifacts", []))
    )
    missing = [p for p in all_paths if not (PROJECT_ROOT / p).exists()]
    assert not missing, f"Manifest lists non-existent paths: {missing}"


# ── 12. README contains required commands ─────────────────────────────────────

def test_readme_contains_required_commands():
    readme = (ML / "README.md").read_text()
    for cmd in ["ml/train.py", "ml/src/tune.py", "ml/src/compare.py",
                "ml/src/ablation.py", "ml/src/publication_results.py",
                "ml/predict.py", "pytest ml/tests"]:
        assert cmd in readme, f"README missing command reference: {cmd}"


# ── 13. Completion document covers Steps 3.1 – 3.11 ─────────────────────────

def test_completion_document_covers_all_steps():
    doc = (ML / "PHASE_3_COMPLETE.md").read_text()
    for step in ["3.1", "3.2", "3.3", "3.4", "3.5", "3.6",
                 "3.7", "3.8", "3.9", "3.10", "3.11"]:
        assert step in doc, f"PHASE_3_COMPLETE.md missing step: {step}"


# ── 14. Production predictions are deterministic ──────────────────────────────

def test_production_predictions_deterministic():
    prod = joblib.load(MODELS / "adaptive_sort_model.joblib")
    df = pd.DataFrame(_RECORDS)
    p1 = list(prod.predict(df))
    p2 = list(prod.predict(df))
    assert p1 == p2, "Predictions are not deterministic"


# ── 15. Probability outputs are valid ─────────────────────────────────────────

def test_probability_outputs_valid():
    prod = joblib.load(MODELS / "adaptive_sort_model.joblib")
    for rec in _RECORDS:
        df = pd.DataFrame([rec])
        proba = prod.predict_proba(df)[0]
        assert abs(sum(proba) - 1.0) < 1e-6, f"Probabilities do not sum to 1: {sum(proba)}"
        assert all(0.0 <= p <= 1.0 for p in proba), f"Probability out of [0,1]: {proba}"
        assert len(proba) == 4, f"Expected 4 class probabilities, got {len(proba)}"


# ── 16. Leakage fields are not in the production feature schema ───────────────

def test_no_leakage_fields_in_required_features():
    overlap = set(REQUIRED_FEATURES) & set(LEAKAGE_AND_OUTCOME_COLS)
    assert not overlap, f"Leakage columns found in REQUIRED_FEATURES: {overlap}"


# ── 17. Publication manifest is valid ─────────────────────────────────────────

def test_publication_manifest_valid():
    with open(PUB / "publication_manifest.json") as f:
        manifest = json.load(f)
    for key in ["generated_at", "source_artifacts", "generated_tables",
                "generated_figures", "selected_production_model", "warnings"]:
        assert key in manifest, f"Publication manifest missing key: {key}"
    assert manifest["selected_production_model"].endswith("adaptive_sort_model.joblib")
    assert len(manifest.get("warnings", [])) >= 1


# ── 18. No machine-specific absolute paths in metadata or manifests ───────────

def test_no_absolute_paths_in_manifests():
    files_to_check = [
        MODELS / "adaptive_sort_model_metadata.json",
        RESULTS / "phase3_manifest.json",
        PUB / "publication_manifest.json",
    ]
    for fpath in files_to_check:
        if not fpath.exists():
            continue
        content = fpath.read_text()
        # Paths like /Users/... or /home/... should not appear in path values
        data = json.loads(content)
        flat_vals = json.dumps(data)
        # Source artifact values should be relative (not starting with /)
        # We check by looking for the user home pattern
        import os
        home = str(Path.home())
        assert home not in flat_vals, (
            f"Machine-specific absolute path found in {fpath.name}: contains {home}"
        )


# ── 19. Final selected model is the Decision Tree Baseline ───────────────────

def test_final_selected_model_is_dt_baseline():
    with open(MODELS / "adaptive_sort_model_metadata.json") as f:
        meta = json.load(f)
    assert meta["selected_model_name"] == "Decision Tree Baseline"
    assert meta["model_class"] == "sklearn.tree.DecisionTreeClassifier"
    assert meta.get("tuned_model_replaced_baseline") == False

    prod = joblib.load(MODELS / "adaptive_sort_model.joblib")
    clf = prod.named_steps["classifier"]
    assert isinstance(clf, DecisionTreeClassifier)
    assert clf.max_depth == 5
    assert clf.min_samples_split == 5
    assert clf.min_samples_leaf == 2


# ── 20. Phase 4 is documented as not yet implemented ─────────────────────────

def test_phase4_documented_as_not_started():
    completion = (ML / "PHASE_3_COMPLETE.md").read_text()
    readme     = (ML / "README.md").read_text()
    # Phase 4 must be mentioned but clearly not marked complete
    assert "Phase 4" in completion, "PHASE_3_COMPLETE.md does not mention Phase 4"
    assert "Phase 4" in readme, "README does not mention Phase 4"
    # Must not claim Phase 4 is already done
    for doc, name in [(completion, "PHASE_3_COMPLETE.md"), (readme, "README.md")]:
        assert "Phase 4 is complete" not in doc and "Phase 4 complete" not in doc, \
            f"{name} incorrectly claims Phase 4 is complete"
