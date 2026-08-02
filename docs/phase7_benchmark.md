# Phase 7.5 — End-to-End Benchmark Runner

## Research Purpose
Phase 7.5 establishes the final benchmarking environment to rigorously compare the end-to-end sorting performance of the `adaptive_ml` strategy against standard baseline implementations (`always_continue`, pure `quick_sort`, `merge_sort`, `insertion_sort`, and Python's built-in `timsort`). This runner isolates runtime behavior while enforcing absolute experimental fairness.

## Fairness Guarantees
To guarantee fairness, the benchmark strictly follows these rules:
1. **Single Array Generation**: For any given permutation of `array_size`, `input_type`, and `repetition`, the target array is generated exactly once.
2. **Deep Copy Isolation**: Before passing the array to any strategy, a full, deep copy is made. Strategies *never* execute on a mutated version of the array.
3. **Deterministic Seeding**: `numpy` and `random` are securely seeded before generation.
4. **Uniform Checkpoint Targets**: Adaptive frameworks always respect the identically configured `checkpoint_pct`.

## Evaluated Strategies
- `adaptive_ml`: The fully autonomous runtime controller utilizing the production `RuntimePredictor`.
- `always_continue`: Checkpoint infrastructure enforced, but strictly executes a monolithic fallback upon checkpoint interrupt (demonstrates baseline framework overhead).
- `insertion_sort`, `merge_sort`, `quick_sort`: Pure monolithic baseline algorithms circumventing all checkpoint instrumentation for raw speed comparison.
- `python_timsort`: Native `list.sort()`.

## Output Schema
The output is dumped into `results/phase7/benchmark_raw.csv`. 

### Metadata
- `run_id`, `seed`, `repetition`
- `strategy`, `array_size`, `input_type`, `starting_algorithm`

### Target Metrics
- `total_runtime_ns`: Wall-clock duration of the full strategy lifecycle.
- `comparisons` / `data_movements`: Algorithmic stability metrics.

### Adaptive ML Metrics (For ML runs)
- `requested_action`, `executed_action`, `final_algorithm`
- `prediction_succeeded`, `fallback_used`
- `feature_build_ns`, `inference_ns`, `switch_overhead_ns`

### Correctness
- `is_sorted`: Boolean check determining if output sorting succeeded.
- `output_hash`: MD5 integrity hash confirming the output is identically identical to permutations generated across all other tested strategies.

### Oracle Metadata
For algorithms exceeding size 1, the benchmark calculates the "Oracle action". This represents the theoretical optimal action choice determined dynamically by test-running all checkpoint branches on a sandbox copy and capturing the minimum time footprint.

## Configuration & Reproducibility
Controlled exclusively via `experiments/phase7/benchmark_config.json`, the framework allows scaling out runs. Defaulting to 100 repetitions allows mapping robust confidence intervals during the Phase 8 statistical analysis.

Run manually via:
```bash
./research_env/bin/python experiments/phase7/run_benchmark.py
```
Use `--quick` for integration testing.
