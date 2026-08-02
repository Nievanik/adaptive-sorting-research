# Phase 7.4 — Runtime Overhead Measurement

## Research Purpose
Phase 7.4 strictly focuses on measuring the internal computational overhead introduced by embedding a machine learning pipeline inside an active sorting algorithm. 
It answers the question: *How much time is wasted pausing the sort, extracting features, and querying the ML model?*

> **Limitation Warning**: This phase measures the *cost* of the adaptive ML mechanism. It **does not** by itself establish that the adaptive sorter is faster than baseline sorting strategies overall. The true end-to-end performance comparison against monothilic static baselines is reserved for Phase 7.5.

## Timing Boundaries
- **Model Load Time**: Measured identically and isolated inside `RuntimePredictor.__init__`. Excluded from all per-run sorting timings.
- **Checkpoint Time**: Wall-clock duration elapsed inside the algorithm before yielding the checkpoint snapshot.
- **Feature Build Time**: Strictly bounded around `build_runtime_features()`.
- **Inference Time**: Strictly bounded around `pipeline.predict()`.
- **Switch Overhead**: Measured by the checkpoint framework (`runner.py`) as the excess time transitioning structures (e.g. slicing/merging lists) beyond the raw sorting operations.
- **Execution After Decision**: The duration to finish sorting after the model responds (includes pure sorting and any switch overhead).
- **Total Runtime**: The entire life of `adaptive_sort()`, encapsulating all internal steps.

## Model-Loading Treatment
The ML `Pipeline` is inherently slow to deserialize and construct. 
To replicate a production service environment:
1. `RuntimePredictor()` is initialized exactly once per experiment.
2. The initial `joblib.load()` time is recorded as `model_load_ns` but excluded from per-run metrics.
3. Warmup runs are executed before capturing metrics to ensure the ML framework caches are primed.

## Warmup Methodology
By default, `warmup_runs = 10` forces the JVM or internal Python caching mechanisms used by sklearn and pandas to compile and optimize the prediction path. These dummy iterations use random arrays and their results are discarded.

## Configuration
The experiment uses a JSON-driven configuration file (`experiments/phase7/overhead_config.json`) defining deterministic parameters:
- `random_seed`: Seed for array generation and random sampling (default `42`).
- `warmup_runs`: Number of unmeasured warmups (default `10`).
- `measured_repetitions`: Number of times to repeat every permutation (default `100`).
- `array_sizes`, `input_types`, `starting_algorithms`: Permutation matrix axes.
- `checkpoint_pct`: Target interruption point (default `50.0`).

## Result Schema & Summaries
Two output files are written to `results/phase7/`:
1. **`runtime_overhead_raw.csv`**: A row for every measured run containing exact nanosecond timings, decisions, fallback status, algorithm paths, and exact sizes/distributions.
2. **`runtime_overhead_summary.json`**: An aggregated report grouping all metrics across configurations, computing mean, median, standard deviation, min, max, p95, and p99 percentiles for structural analysis.

## ML Overhead Formula
The primary penalty measured by this experiment is `ml_overhead_ns` and its relative ratio:
```
ml_overhead_ns = feature_build_ns + inference_ns
ml_overhead_pct = (ml_overhead_ns / total_runtime_ns) * 100
```
This precisely isolates the ML intrusion time independently from checkpointing or sorting operations.

## Correctness Checks
To prevent faulty timing measurements from polluting the dataset, every single run executes a rigorous integrity check before appending:
1. `res.is_sorted` must equal True.
2. `res.sorted_values` must be an exact permutation of the original array elements.
3. The original array must remain unmodified (proving the copy policy holds).
If any constraint fails, the runner immediately raises a `RuntimeError` and aborts.

## Reproduction Command
From the root of the repository:
```bash
./research_env/bin/python experiments/phase7/measure_runtime_overhead.py
```
For integration tests or rapid prototyping, use the `--quick` flag which overrides repetitions and sizes to miniature values.
