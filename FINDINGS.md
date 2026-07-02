# Research Findings — Adaptive Sorting via Mid-Execution ML

> **Project**: Mid-Execution Sorting Adaptation via Machine Learning
> **Researcher**: Keepa Maharjan | **Advisor**: Prof. Vladislav D. Veksler
> **Status**: Phase 1 Complete — Algorithm benchmarking with full metrics (comparisons, moves, time)

---

## Finding 1 — QuickSort's Random Pivot Fails on Uniform/Duplicate Data

### What we observed
Even with a **random pivot** (which prevents worst-case on sorted/reverse-sorted input),
QuickSort degrades to **O(n²)** on `all_equal` data. The numbers make this undeniable:

| Input Type     | n=1,000 comparisons | n=5,000 comparisons | n=10,000 comparisons | Time (n=10,000) |
|---------------|--------------------|--------------------|---------------------|-----------------|
| random        | 10,775             | 69,093             | 161,956             | 15.4 ms         |
| sorted        | 11,071             | 72,500             | 157,935             | 14.2 ms ✅      |
| reverse_sorted| 10,318             | 74,641             | 150,197             | 13.6 ms ✅      |
| duplicate_heavy| 32,115            | 656,303            | 2,544,250           | 239.7 ms 🟡     |
| all_equal     | **499,500**        | **12,497,500**     | **49,995,000**      | **4,675 ms** 🔴 |

- `499,500` is exactly `n*(n-1)/2` — the mathematical definition of O(n²) worst case.
- Random pivot protects against sorted/reverse-sorted ✅ but **cannot protect against equal elements**.
- At n=10,000: QuickSort on all_equal is **303× slower** than QuickSort on random data.

### Why this happens
When all elements are equal, every element satisfies `arr[j] <= pivot`, so every partition
puts everything on one side. The array is split 0 vs n-1 every single time — maximally unbalanced.
No amount of pivot randomization helps because the elements themselves are the problem.

### Research implication
This is a textbook case where **mid-execution switching would provide massive gains**.
At the 50% checkpoint, the comparison rate would be exploding — the ML model should
detect this signal and switch to MergeSort or InsertionSort immediately.

---

## Finding 2 — InsertionSort Has Hidden Strengths on Structured Data

### What we observed
InsertionSort, commonly dismissed as "slow", outperforms QuickSort dramatically on
structured input types:

| Input Type     | InsertionSort n=10,000 | QuickSort n=10,000 | Winner           |
|---------------|----------------------|-------------------|------------------|
| sorted         | **0.95 ms**, 9,999 cmp | 14.2 ms, 157,935 cmp | InsertionSort 🏆 |
| all_equal      | **0.93 ms**, 9,999 cmp | 4,675 ms, 49,995,000 cmp | InsertionSort 🏆 |
| nearly_sorted  | **7.95 ms**, 93,175 cmp | 15.0 ms, 149,213 cmp | InsertionSort 🏆 |
| reverse_sorted | 4,320 ms 🔴            | 13.6 ms ✅         | QuickSort        |
| random         | 2,090 ms 🔴            | 15.4 ms ✅         | QuickSort        |

### Why this matters
InsertionSort is **O(n)** on sorted and nearly-sorted data — it simply scans through
and makes zero or few swaps. On all_equal data, every comparison immediately breaks
(`arr[j] <= key` is true but `arr[j] > key` is false), so it terminates the inner loop
after a single comparison per element.

### Research implication
A sorting system that starts with QuickSort on what turns out to be nearly-sorted data
is wasting time. If the checkpoint detects a low comparison rate (few inversions),
switching to InsertionSort would finish the job much faster.

---

## Finding 3 — MergeSort is the Most Predictable Algorithm

### What we observed
MergeSort's **move count is constant** for a given array size regardless of input type:

| Input Type     | MergeSort moves (n=10,000) | MergeSort time |
|---------------|--------------------------|----------------|
| random         | 133,616                  | 19.8 ms        |
| sorted         | 133,616                  | 15.3 ms        |
| reverse_sorted | 133,616                  | 15.0 ms        |
| nearly_sorted  | 133,616                  | 15.3 ms        |
| duplicate_heavy| 133,616                  | 17.5 ms        |
| all_equal      | 133,616                  | 14.7 ms        |

Move count is always exactly `n * ceil(log₂(n))` — this is a mathematical property
of MergeSort. Its comparisons vary (sorted data requires fewer merge comparisons since
elements are already in order), but its work in terms of data movement is invariant.

### Research implication
MergeSort is the "safe" fallback algorithm. If the checkpoint is uncertain about what
the data looks like, switching to MergeSort guarantees O(n log n) regardless. This
makes it a strong default candidate in the switching decision.

---

## Finding 4 — The Comparison Rate Signal

### What we observed
At n=10,000, the ratio of comparisons to array size (`comparisons / n`) reveals algorithm behavior:

| Algorithm     | Input Type     | Comparisons/n |
|--------------|---------------|--------------|
| InsertionSort | sorted         | 1.0 (optimal) |
| InsertionSort | all_equal      | 1.0 (optimal) |
| MergeSort     | any            | ~6.5–13.3    |
| QuickSort     | random         | ~15.7        |
| QuickSort     | all_equal      | **4,999.5** (worst case) |
| InsertionSort | reverse_sorted | 4,999.5 (worst case) |

This ratio is a **key feature candidate** for the ML model. A comparison rate that is
growing much faster than expected at the 50% checkpoint is a strong signal to switch.

---

## Finding 5 — Benchmark Configuration (Phase 1 Baseline)

These results were generated with:
- **Datasets**: 5 sizes (100, 500, 1000, 5000, 10000) × 7 input types + 2 edge cases
- **Metrics captured per run**: `time_ms`, `comparisons`, `moves`
- **Pivot strategy**: Randomized QuickSort (random pivot each partition)
- **Move definition**: 1 move = 1 element write (a swap = 2 moves)
- **Results stored in**: `results/<algorithm>/<size>.json`

---

## Open Questions for Research

1. **What is the exact 50% checkpoint for each algorithm?**
   - InsertionSort: when `i == n/2` (outer loop is linear)
   - MergeSort: harder to define — after half the merge calls? After processing n/2 elements?
   - QuickSort: iterative, non-linear — requires comparison-based counting

2. **Does switching cost erode the gains?**
   - Transferring partial array state has overhead. The Switching Cost Matrix must
     account for this to avoid false positives.

3. **Is the comparison rate at 50% predictive of final cost?**
   - This is the core ML hypothesis to test. Early signals may be noisy for small n.

4. **What features are most informative for the ML model?**
   - Candidates: comparison count, move count, elapsed_ms, progress rate (comparisons/n),
     input category (if known), comparison rate acceleration (is it growing faster than expected?)
