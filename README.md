# Adaptive Sorting via Mid-Execution Machine Learning

**Researcher**: Keepa Maharjan | **Advisor**: Prof. Vladislav D. Veksler

A research project investigating whether a machine learning model — trained on features extracted at a **50% execution checkpoint** — can accurately predict when switching sorting algorithms mid-execution yields a net performance gain.

---

## Research Overview

Traditional sorting algorithms excel under specific input conditions but cannot adapt once execution has begun. Hybrid approaches (TimSort, IntroSort) improve on this using hand-crafted heuristics, but commit to a strategy before any runtime behavior is observed.

This project explores a different question:

> *Can runtime signals captured mid-execution — comparison rates, move counts, elapsed time, and progress speed — provide enough information for an ML model to make better switching decisions than static heuristics?*

### Algorithms Studied
| Algorithm | Best Case | Average Case | Worst Case | Strength |
|-----------|-----------|--------------|------------|----------|
| QuickSort (random pivot) | O(n log n) | O(n log n) | O(n²) on uniform data | Fast in practice on random data |
| MergeSort | O(n log n) | O(n log n) | O(n log n) | Predictable, stable fallback |
| InsertionSort | O(n) | O(n²) | O(n²) | Excellent on sorted/nearly-sorted |

---

## Project Structure

```
adaptive-sorting-research/
├── src/
│   ├── algorithms/           ← Clean sort implementations (used by tests)
│   │   ├── insertion_sort.py
│   │   ├── merge_sort.py
│   │   └── quick_sort.py
│   ├── algorithm_metrics/    ← Instrumented versions (track comparisons, moves, time)
│   │   ├── insertion_sort.py
│   │   ├── merge_sort.py
│   │   └── quick_sort.py
│   ├── checkpoint/           ← (In progress) 50% execution checkpoint logic
│   ├── datasets/
│   │   ├── generator.py      ← Dataset generators for all 7 input types
│   │   └── create_datasets.py
│   └── data_loader.py        ← Loads datasets from data/
├── data/                     ← Generated JSON datasets
├── results/                  ← Benchmark output (JSON per algorithm per size)
├── tests/                    ← Correctness tests for all 3 algorithms
├── benchmark.py              ← Core benchmarking engine
├── main.py                   ← CLI entry point
├── findings.md               ← Key empirical findings from benchmarking
├── gemini.md                 ← Project memory and conventions for AI assistant
└── requirements.txt
```

---

## Setup

### Mac / Linux

```bash
# 1. Clone the repository
git clone https://github.com/Nievanik/adaptive-sorting-research.git
cd adaptive-sorting-research

# 2. Create virtual environment
python3 -m venv research_env
source research_env/bin/activate

# 3. Install dependencies
pip install -r requirements.txt
```

### Windows (PowerShell)

```powershell
# 1. Clone the repository
git clone https://github.com/Nievanik/adaptive-sorting-research.git
cd adaptive-sorting-research

# 2. Create virtual environment
python -m venv research_env
research_env\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt
```

---

## Running the Full Pipeline

All commands must be run from the **project root** (`adaptive-sorting-research/`).

---

### Step 1 — Generate Datasets

Creates JSON datasets for 5 sizes × 7 input types in `data/`.

**Mac / Linux:**
```bash
cd src/datasets && python create_datasets.py && cd ../..
```

**Windows:**
```powershell
cd src\datasets; python create_datasets.py; cd ..\..
```

**What gets created:**

| Type | Description |
|------|-------------|
| `random` | Uniformly random integers |
| `sorted` | Already sorted (ascending) |
| `reverse_sorted` | Sorted in reverse (descending) |
| `nearly_sorted` | Sorted with a small number of random swaps |
| `duplicate_heavy` | Only 20 unique values across n elements |
| `all_equal` | Every element is identical |
| `edge_cases` | Empty array, single element |

Sizes generated: `100, 500, 1000, 5000, 10000`

---

### Step 2 — Run Tests

Verifies correctness of all 3 sorting algorithm implementations.

**Mac / Linux:**
```bash
PYTHONPATH=. python tests/test_insertionsort.py
PYTHONPATH=. python tests/test_mergesort.py
PYTHONPATH=. python tests/test_quicksort.py
```

**Windows (PowerShell):**
```powershell
$env:PYTHONPATH="."; python tests\test_insertionsort.py
$env:PYTHONPATH="."; python tests\test_mergesort.py
$env:PYTHONPATH="."; python tests\test_quicksort.py
```

Each test file checks: empty array, single element, duplicates, sorted input,
reverse sorted, negative numbers, all-equal, random input, large input.

---

### Step 3 — Run Benchmark

Runs all selected algorithms across all dataset types and sizes.
Records **time (ms)**, **comparison count**, and **move count** per run.

**Mac / Linux:**
```bash
# All 3 algorithms, all 5 sizes (full research run)
python main.py --algo insertion_sort merge_sort quick_sort --size 100 500 1000 5000 10000

# Single algorithm
python main.py --algo quick_sort --size 1000

# Quick test run (small sizes only)
python main.py --algo insertion_sort merge_sort quick_sort --size 100 500
```

**Windows (PowerShell):**
```powershell
# All 3 algorithms, all 5 sizes (full research run)
python main.py --algo insertion_sort merge_sort quick_sort --size 100 500 1000 5000 10000

# Single algorithm
python main.py --algo quick_sort --size 1000

# Quick test run (small sizes only)
python main.py --algo insertion_sort merge_sort quick_sort --size 100 500
```

**Results are saved to** `results/<algorithm_name>/<size>.json`.

Each result entry looks like:
```json
{
  "algorithm": "quick_sort",
  "type": "all_equal",
  "size": 1000,
  "time_ms": 50.99,
  "comparisons": 499500,
  "moves": 1002996
}
```

---

### All-in-One (Mac / Linux)

```bash
source research_env/bin/activate
cd src/datasets && python create_datasets.py && cd ../..
PYTHONPATH=. python tests/test_insertionsort.py
PYTHONPATH=. python tests/test_mergesort.py
PYTHONPATH=. python tests/test_quicksort.py
python main.py --algo insertion_sort merge_sort quick_sort --size 100 500 1000 5000 10000
```

### All-in-One (Windows PowerShell)

```powershell
research_env\Scripts\activate
cd src\datasets; python create_datasets.py; cd ..\..
$env:PYTHONPATH="."; python tests\test_insertionsort.py
$env:PYTHONPATH="."; python tests\test_mergesort.py
$env:PYTHONPATH="."; python tests\test_quicksort.py
python main.py --algo insertion_sort merge_sort quick_sort --size 100 500 1000 5000 10000
```

---

## Key Findings (Phase 1)

> See `findings.md` for full tables and analysis.

- **QuickSort degrades to O(n²) on uniform/duplicate-heavy data** even with random pivot.
  At n=10,000 on all-equal data: 49,995,000 comparisons, 4,675 ms — **303× slower** than on random data.

- **InsertionSort outperforms QuickSort on structured data.**
  On sorted or all-equal input at n=10,000: ~0.93 ms vs 4,675 ms for QuickSort.

- **MergeSort is the most predictable algorithm.**
  Move count is always exactly `n × ceil(log₂n)` regardless of input type — a reliable fallback.

- **Comparison rate is a strong mid-execution signal.**
  The ratio `comparisons / n` at the 50% checkpoint ranges from 1.0 (InsertionSort on sorted)
  to 4,999.5 (QuickSort or InsertionSort in worst case) — a core feature for the ML model.

---

## Common Issues

**ModuleNotFoundError: No module named 'src'**
```bash
# Mac/Linux
export PYTHONPATH=.

# Windows
$env:PYTHONPATH="."
```

**Virtual environment not found**
Make sure you created it with `python3 -m venv research_env` and activated it before running any scripts.

**Dataset files not found**
Run Step 1 (generate datasets) before running tests or benchmark.
