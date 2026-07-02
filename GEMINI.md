# Gemini Project Memory — Adaptive Sorting Research

## Research Overview

- **Project**: Mid-Execution Sorting Adaptation via Machine Learning
- **Researcher**: Keepa Maharjan and Nievanik Thapa Shrestha
- **Advisor**: Prof. Vladislav D. Veksler
- **Core Idea**: Run a sorting algorithm, pause at a **50% execution checkpoint**, extract runtime features, and use an ML model (Decision Tree / Random Forest) to decide whether to switch to a better algorithm mid-execution.

---

## Research Pipeline (Proposed Methods)

1. **Algorithm Implementation & Instrumentation** — QuickSort, MergeSort, InsertionSort (from scratch), tracking comparisons, swaps, elapsed time, and progress at checkpoint.
2. **Dataset Construction** — 7 input categories: random, sorted, reverse sorted, nearly sorted, duplicate-heavy, mixed-pattern, real-world numeric.
3. **Switching Cost Analysis** — At 50% checkpoint, compare: continue vs. switch. Build a **Switching Cost Matrix** as ground truth.
4. **Feature Extraction & Labeling** — Feature vector: comparison count, swap count, elapsed time, progress rate, input category. Labels: should switch? which algorithm? magnitude of gain/loss.
5. **ML Model Training** — Decision Tree + Random Forest. Evaluate with accuracy, precision, recall, F1.
6. **Adaptive System Prototype** — Start with one algorithm → checkpoint → ML decision → continue or switch. Benchmark vs. always-continue, TimSort, IntroSort.

---

## Project Structure

```
adaptive-sorting-research/
├── src/
│   ├── algorithms/           ← Clean implementations (no metrics, used by tests)
│   │   ├── insertion_sort.py
│   │   ├── merge_sort.py
│   │   └── quick_sort.py
│   ├── algorithm_metrics/    ← Instrumented versions (track comparisons/moves/time)
│   │   ├── insertion_sort.py
│   │   ├── merge_sort.py
│   │   └── quick_sort.py
│   ├── checkpoint/           ← Instrumented checkpoint versions and runner
│   ├── datasets/
│   │   ├── generator.py      ← 7 input type generators
│   │   └── create_datasets.py ← Saves datasets as JSON to data/
│   └── data_loader.py        ← Loads datasets from data/ (path resolved from file location)
├── data/                     ← Generated dataset JSON files
├── results/                  ← Benchmark output (all 3 algos × all sizes complete)
├── tests/                    ← Manual test files (no pytest, run directly)
├── benchmark.py              ← Uses algorithm_metrics/ — records time_ms, comparisons, moves
├── main.py                   ← CLI entry (--algo, --size flags)
├── findings.md               ← Key empirical findings from Phase 1 benchmarking
└── requirements.txt
```

---

## Bugs Fixed (All Resolved ✅)

| # | File | Bug | Status |
|---|------|-----|--------|
| 1 | `tests/test_insertionsort.py` | Was importing/testing `merge_sort` instead of `insertion_sort` | ✅ Fixed |
| 2 | `src/algorithm_metrics/insertion_sort.py` | Module-level test code ran on every import (no `__main__` guard) | ✅ Fixed |
| 3 | `src/algorithm_metrics/*.py` | Global counters never reset between calls — metrics accumulated | ✅ Fixed |
| 4 | `src/data_loader.py` | Used `Path("data")` (relative) — only worked from project root | ✅ Fixed |
| 5 | `benchmark.py` | Only measured wall-clock time; never used algorithm_metrics | ✅ Fixed |

---

## Conventions & Rules

### General
- Always run scripts from the **project root** (`adaptive-sorting-research/`).
- Use `PYTHONPATH=.` if import errors occur: `export PYTHONPATH=.`
- Virtual environment: `research_env/` (activate before running anything)

### Algorithm Files
- `src/algorithms/` — pure, clean sort functions only. No metrics, no timing, no prints. Used by tests.
- `src/algorithm_metrics/` — instrumented versions. Each call resets all counters at the start. Has `__main__` guard.
- Do NOT modify `src/algorithms/` to add metrics — keep them clean.

### algorithm_metrics Interface (important — used by benchmark)
Each metrics module exposes these globals after calling its sort function:
- `comparison_count` — total element comparisons
- `move_count` — total element writes (1 swap = 2 moves)
- `elapsed_ms` — wall-clock time captured INSIDE the sort function (do NOT add external timer)
- MergeSort uses a public `merge_sort()` wrapper + private `_merge_sort()` recursive helper
  so that the reset only fires at the top-level call, not on every recursive call.

### Benchmark
- `benchmark.py` uses `ALGO_REGISTRY` dict — validation keys for algorithms
- `run_sort_with_checkpoint(algo_name, arr)` is the single call site: runs to 50% checkpoint, then continues to completion alongside all switching combinations
- Results saved to `results/<algorithm_name>/<size>.json`
- Format per entry contains nested dicts for `checkpoint`, `continue`, and `switch_<algo>` paths. Each path includes: `time_ms`, `comparisons`, `moves`, and an isolated `overhead` dictionary (containing setup & merge costs).
- Always pass `arr.copy()` into runner — insertion_sort mutates in place

### Move Count Convention
- 1 move = 1 element write
- A swap (a, b = b, a) = **2 moves** (not 3)
- InsertionSort counts shifts AND the final key placement as moves

### Datasets
- 5 sizes: `100, 500, 1000, 5000, 10000`
- 7 types: `random, sorted, reverse_sorted, nearly_sorted, duplicate_heavy, all_equal, edge_cases`
- **Missing from proposal**: real-world numeric dataset (not yet implemented)
- `edge_cases/` contains: `empty.json`, `single_element.json`
- Generate with: `cd src/datasets && python create_datasets.py && cd ../..`

### Tests
- No pytest — run test files directly with `PYTHONPATH=.`
- Each test file has a `run_test(name, func)` helper that prints `[PASS]` / `[FAIL]`
- Run all: `PYTHONPATH=. python tests/test_insertionsort.py && PYTHONPATH=. python tests/test_mergesort.py && PYTHONPATH=. python tests/test_quicksort.py`

---

## Key Empirical Findings (Phase 1)

> Full details with tables in `findings.md`

### QuickSort — Random Pivot Fails on Uniform Data
- Random pivot protects against sorted/reverse-sorted (O(n log n) maintained ✅)
- Degrades to **O(n²)** on `all_equal` and `duplicate_heavy` data
- At n=10,000: QuickSort all_equal = 49,995,000 comparisons, 4,675 ms vs 161,956 comparisons, 15 ms on random
- Root cause: all elements equal means every partition is maximally unbalanced regardless of pivot choice
- **Research implication**: This is a prime candidate for mid-execution switching — the ML model should detect exploding comparison rate at 50% checkpoint

### InsertionSort — Hidden Strengths on Structured Data
- On sorted/all_equal at n=10,000: 9,999 comparisons, ~0.93 ms → **faster than QuickSort**
- On nearly_sorted at n=10,000: 93,175 comparisons, 7.95 ms → **faster than QuickSort**
- On random/reverse_sorted: degrades to O(n²) as expected
- **Research implication**: Switching TO InsertionSort mid-execution on nearly-sorted data is highly beneficial

### MergeSort — The Predictable Fallback
- Move count is always exactly `n * ceil(log₂(n))` regardless of input type
- At n=10,000: always 133,616 moves across all input types
- Time varies slightly (14.7–19.8 ms at n=10,000) but never catastrophic
- **Research implication**: MergeSort is the "safe" switch target — guarantees O(n log n)

### The Comparison Rate Signal
- `comparisons / n` at checkpoint is a key ML feature candidate
- InsertionSort sorted: ratio = 1.0 (optimal)
- QuickSort all_equal: ratio = 4,999.5 (worst case)
- A rapidly growing comparison rate at 50% is a strong switch signal

---

## What's Missing / Next Steps

- [x] Fix all 5 critical bugs
- [x] algorithm_metrics files clean, consistent, and benchmark-ready
- [x] benchmark.py uses metrics modules, records comparisons + moves
- [x] Full benchmark run complete (all 3 algos × all 5 sizes × all 7 types)
- [x] findings.md created with Phase 1 empirical results
- [x] Build `src/checkpoint/` module — the 50% checkpoint logic
- [x] Design checkpoint: comparison-based pause, return partial array state
- [x] Integrate checkpoint version into benchmark.py (records 50% and 100% metrics)
- [ ] Add real-world numeric dataset to generator
- [x] Build Switching Cost Matrix (run all combos: start algo A, switch to B at 50%)
- [ ] Feature extraction module
- [ ] ML training pipeline (Decision Tree + Random Forest)
- [ ] Adaptive system prototype
- [ ] Baseline comparisons (TimSort, IntroSort)

---

## Important Notes for Gemini

- Do NOT modify `src/algorithms/` to add metrics — keep them clean.
- The checkpoint is defined at **50% of total algorithm steps (comparison-based)**, not 50% of wall time.
- When building the checkpoint system, it needs to work with all 3 algorithms and return the **partial array state** so execution can resume or be handed off.
- The switching cost matrix is the ground truth for ML labels — build it carefully.
- QuickSort uses **random pivot** (randomized quicksort) — this is intentional for fairness.
- InsertionSort mutates in place in `algorithm_metrics/` (no internal copy) — benchmark always passes `arr.copy()`.
- MergeSort in `algorithm_metrics/` has two functions: `merge_sort()` (public wrapper, resets globals) and `_merge_sort()` (recursive, never resets). Do not merge these.
- findings.md is the living empirical record — update it as new experiments are run.
