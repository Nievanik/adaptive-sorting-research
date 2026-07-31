# Machine Learning Pipeline for Adaptive Sorting

This directory contains the machine learning components designed to predict the optimal sorting action (`continue`, `switch_insertion_sort`, `switch_merge_sort`, or `switch_quick_sort`) at execution checkpoints.

---

## 1. Pipeline Phases

The ML workflow is structured into five sequential phases:

### Phase 2.1: Data Preprocessing (`ml/src/preprocess.py`)
- **Purpose**: Loads dataset from raw CSV records, segregates feature matrix ($X$) from target labels ($y$), and builds an isolated preprocessing `ColumnTransformer` (OneHotEncoding for categorical features and passthrough for numerical features).
- **Leakage Prevention**: Drops target outcome and execution metrics (e.g. baseline and switch execution times) to prevent target leakage.

### Phase 2.2: Baseline Model Training (`ml/train.py`)
- **Purpose**: Constructs a scikit-learn `Pipeline` binding the preprocessor and a `RandomForestClassifier`. Trains the model on a stratified holdout train/test split.
- **Model Checkpoint**: Saves the serialized, fitted pipeline to `ml/models/random_forest_baseline.joblib`.

### Phase 2.3: Model Evaluation (`ml/src/evaluate.py`)
- **Purpose**: Measures prediction performance using multiple metrics (Accuracy, Macro/Weighted Precision, Recall, F1-Scores), calculates class distributions, and conducts Stratified Cross-Validation on the pipeline.
- **Leakage Safety**: Fits and transforms the preprocessing steps independently *inside* each CV fold using `sklearn.base.clone`.

### Phase 2.4: Feature Importance Analysis (`ml/src/feature_importance.py`)
- **Purpose**: Extracts feature importances from the trained classifier, maps them back to the preprocessed/OneHotEncoded feature names, and ranks them.
- **Visualisation**: Saves a CSV report and plots a horizontal bar chart displaying the Top 15 features.

### Phase 2.5: Prediction and Inference (`ml/predict.py`)
- **Purpose**: Provides a reusable inference module and CLI that validates single or batch records to output actions and confidence scores without retraining.

---

## 2. Usage Instructions

Ensure you run these commands from the project root using the virtual environment:

### A. Training the Model
To execute the baseline training pipeline and print holdout/cross-validation metrics:
```bash
./research_env/bin/python ml/train.py
```

### B. Extracting Feature Importance
To rank features and generate visualization plots:
```bash
./research_env/bin/python ml/src/feature_importance.py
```

### C. Running Predictions (CLI)
You can run predictions either by supplying individual arguments or pointing to a JSON file.

#### Single-Record Prediction via Arguments:
```bash
./research_env/bin/python ml/predict.py \
  --algorithm quick_sort \
  --input-type random \
  --size 1000 \
  --checkpoint-pct 50 \
  --checkpoint-time-ms 1.25 \
  --checkpoint-comparisons 4200 \
  --checkpoint-data-movements 1700 \
  --comparisons-per-element 4.2 \
  --movements-per-element 1.7 \
  --work-ratio 0.40 \
  --time-per-element-ms 0.00125
```

#### Single-Record Prediction via JSON:
Create a JSON file (e.g. `checkpoint.json`):
```json
{
  "algorithm": "quick_sort",
  "input_type": "random",
  "size": 1000,
  "checkpoint_pct": 50,
  "checkpoint_time_ms": 1.25,
  "checkpoint_comparisons": 4200,
  "checkpoint_data_movements": 1700,
  "comparisons_per_element": 4.2,
  "movements_per_element": 1.7,
  "work_ratio": 0.40,
  "time_per_element_ms": 0.00125
}
```
Run command:
```bash
./research_env/bin/python ml/predict.py --input-json path/to/checkpoint.json
```

### D. Running Tests
To run the complete test suite:
```bash
./research_env/bin/python -m pytest
```

---

## 3. Directory Structure & Produced Files

When executing the workflow, the following workspace directories and files are produced:

```
ml/
├── data/
│   └── processed/
│       └── checkpoint_training.csv   # Preprocessed training dataset
├── models/
│   └── random_forest_baseline.joblib # Serialized training pipeline (preprocessor + model)
├── results/
│   ├── feature_importance.csv        # Sorted list of features with importance values
│   └── feature_importance.png        # Bar chart plot of top 15 features
├── src/
│   ├── evaluate.py                   # Metrics & cross-validation helpers
│   ├── feature_importance.py         # Feature ranking script
│   ├── generate_labels.py            # Label generation logic
│   ├── preprocess.py                 # Feature selection & transformers
│   └── validate_dataset.py           # Integrity validator
├── tests/
│   ├── test_data_extraction.py
│   ├── test_evaluate.py
│   ├── test_feature_importance.py
│   ├── test_predict.py
│   ├── test_preprocess.py
│   └── test_train.py
├── predict.py                        # Single/batch inference & CLI
└── train.py                          # Training orchestrator
```
