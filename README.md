# 📊 Adaptive Sorting Research Project

This project implements and benchmarks multiple sorting algorithms:

- QuickSort
- MergeSort
- Insertion Sort

It is designed for **algorithm analysis, performance comparison, and research experimentation** across different dataset sizes and types.

---

---

# ⚙️ Features

- Multiple sorting algorithm implementations
- Edge case handling (empty, single element)
- Dataset generation for different sizes
- Benchmarking (time, comparisons, swaps/moves)
- Modular research design for experimentation

---

# 🚀 Setup Instructions

## 1. Clone Repository

```bash
git clone <your-repo-url>
cd adaptive-sorting-research


2. Create Virtual Environment
Mac / Linux
python3 -m venv research_env
source research_env/bin/activate

Windows (PowerShell)
python -m venv research_env
research_env\Scripts\activate


3. Install Dependencies
pip install -r requirements.txt

📦 Step 1 — Generate Dataset
This project first generates datasets of different sizes.

python src/create_dataset.py

What this does:
Creates random arrays
Uses multiple sizes (100, 500, 1000, 5000, 10000)
Stores them inside data/

🧪 Step 2 — Run Tests (Optional / Manual)

You are not using pytest in this workflow.

Instead, run test files directly:

QuickSort test
python tests/test_quicksort.py

MergeSort test
python tests/test_mergesort.py

Insertion Sort test
python tests/test_insertionsort.py

Each test verifies:

Empty array
Single element
Duplicates
Sorted input
Reverse sorted input


📊 Step 3 — Run Benchmark

This is the main research execution step.

python src/benchmark.py
What benchmark does:

For each:

dataset size
dataset type
sorting algorithm

It records:

⏱ Execution time
🔁 Number of comparisons
🔄 Number of swaps/moves
Output:

Results are stored in:

results/

Example structure:

results/
├── quicksort/
├── mergesort/
└── insertion_sort/

Each file contains:

{
  "size": 1000,
  "time": 0.0023,
  "comparisons": 1200,
  "swaps": 340
}
🧠 Edge Cases

Located in:

data/edge_cases/

Includes:

empty.json → []
single_element.json → [49]

These ensure correctness for minimal inputs.

🪟💻 Cross-Platform Usage
Mac / Linux
python3 src/create_dataset.py
python3 tests/test_quicksort.py
python3 src/benchmark.py
Windows (PowerShell)
python src/create_dataset.py
python tests/test_quicksort.py
python src/benchmark.py
⚠️ Common Issues
1. Module import errors

If Python cannot find modules:

Mac/Linux:
export PYTHONPATH=.
Windows:
set PYTHONPATH=.
```
