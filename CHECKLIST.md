# Research Checklist — Mid-Execution Sorting Adaptation via ML

This file tracks the execution progress of the research pipeline for KEEP MAHARJAN's project.

---

## Progress Dashboard

| Phase | Description | Status |
|---|---|---|
| **Phase 1** | Algorithm Implementation & Instrumentation | **Completed** ✅ |
| **Phase 2** | Dataset Construction | **Partially Completed** 🟡 |
| **Phase 3** | Baseline Benchmarking | **Completed** ✅ |
| **Phase 4** | Checkpoint Module | **Completed** ✅ |
| **Phase 5** | Switching Cost Matrix | **Completed** ✅ |
| **Phase 6** | Feature Extraction & Labeling | *Pending* 🔲 |
| **Phase 7** | ML Model Training | *Pending* 🔲 |
| **Phase 8** | Adaptive System Prototype | *Pending* 🔲 |
| **Phase 9** | Write-up & Analysis | *Pending* 🔲 |

---

## Detailed Task Breakdown

### Phase 1 — Algorithm Implementation & Instrumentation ✅
- [x] Implement QuickSort (random pivot, iterative stack-based)
- [x] Implement MergeSort
- [x] Implement InsertionSort
- [x] Instrument all 3 to track comparisons, moves, and elapsed time
- [x] Ensure metric counters reset correctly at the top of each sort call

### Phase 2 — Dataset Construction 🟡
- [x] Implement generator for random arrays
- [x] Implement generator for sorted arrays
- [x] Implement generator for reverse-sorted arrays
- [x] Implement generator for nearly-sorted arrays
- [x] Implement generator for duplicate-heavy arrays
- [x] Implement generator for all-equal arrays
- [x] Implement generator for edge cases (empty, single element)
- [ ] Add real-world numeric dataset (e.g. sensor data or log timestamps) to generator

### Phase 3 — Baseline Benchmarking ✅
- [x] Run baseline benchmarks for all 3 algorithms
- [x] Cover all 5 dataset sizes (100, 500, 1000, 5000, 10000)
- [x] Capture comparison counts and move counts in results

### Phase 4 — Checkpoint Module ✅
- [x] Define O(n) checkpoint for InsertionSort (outer loop `i == n // 2`)
- [x] Define checkpoint for MergeSort (left recursive half completed)
- [x] Define O(n log n) checkpoint for QuickSort (Option B: `comparison_count >= estimated_total / 2`)
- [x] Implement `src/checkpoint/` modules (`runner.py`, `insertion_sort_checkpoint.py`, etc.)
- [x] Support resumption of the starting algorithm
- [x] Support switching/handing off to a different algorithm (all 9 combinations)
- [x] Measure and isolate setup/merge overhead separate from sorting work

### Phase 5 — Switching Cost Matrix ✅
- [x] Expand `benchmark.py` to run continue + switch pathways for every algorithm run
- [x] Evaluate all 9 algorithm transition pathways
- [x] Save complete matrix (checkpoint, continue, switch_A, switch_B, overheads) inside output JSONs

### Phase 6 — Feature Extraction & Labeling 🔲
- [ ] Write a script to load all result JSONs from `results/`
- [ ] Extract mid-execution features at the 50% checkpoint (comparisons, moves, time, size, start algo)
- [ ] Calculate ground truth target labels (true optimal choice, should_switch boolean, gain)
- [ ] Save the consolidated training set as a single `dataset.csv` file

### Phase 7 — ML Model Training 🔲
- [ ] Implement training script using scikit-learn
- [ ] Train Decision Tree Classifier
- [ ] Train Random Forest Classifier
- [ ] Evaluate performance (accuracy, precision, recall, F1-score)
- [ ] Run feature importance analysis

### Phase 8 — Adaptive System Prototype 🔲
- [ ] Build end-to-end adaptive sorting wrapper
- [ ] Integrates: start sort $\rightarrow$ pause $\rightarrow$ feature extraction $\rightarrow$ ML inference $\rightarrow$ continue/switch
- [ ] Benchmark prototype against baselines: always continue, Python's built-in `sorted()` (TimSort), IntroSort

### Phase 9 — Write-up & Analysis 🔲
- [ ] Generate tables comparing ML accuracy and model decisions
- [ ] Complete findings list inside `findings.md`
- [ ] Write final report/paper drafts
