import json
import pytest
from pathlib import Path
from experiments.phase7.measure_runtime_overhead import run_experiment

def test_default_config_loads_and_runs_quick(tmp_path):
    # We will write a small config to tmp_path just to test the logic
    config = {
        "random_seed": 42,
        "warmup_runs": 0,
        "measured_repetitions": 1,
        "array_sizes": [10],
        "input_types": ["random"],
        "starting_algorithms": ["quick_sort"],
        "checkpoint_pct": 50.0
    }
    config_path = tmp_path / "config.json"
    with config_path.open("w") as f:
        json.dump(config, f)
    
    out_dir = tmp_path / "results"
    num_runs, stats, csv_path, json_path = run_experiment(config_path, out_dir, is_quick=False)
    
    assert num_runs == 1
    assert csv_path.exists()
    assert json_path.exists()
    
    # Check JSON correctness
    with json_path.open("r") as f:
        summary = json.load(f)
    
    assert "overall" in summary
    assert "grouped" in summary
    assert "research_metrics" in summary
    assert summary["overall"]["feature_build_ns"]["count"] == 1

def test_invalid_repetitions_rejected(tmp_path):
    config = {
        "random_seed": 42,
        "warmup_runs": 0,
        "measured_repetitions": 0,
        "array_sizes": [10],
        "input_types": ["random"],
        "starting_algorithms": ["quick_sort"]
    }
    config_path = tmp_path / "config.json"
    with config_path.open("w") as f:
        json.dump(config, f)
        
    with pytest.raises(ValueError, match="measured_repetitions must be > 0"):
        run_experiment(config_path, tmp_path)

def test_invalid_sizes_rejected(tmp_path):
    config = {
        "measured_repetitions": 1,
        "array_sizes": [-5],
        "input_types": ["random"],
        "starting_algorithms": ["quick_sort"]
    }
    config_path = tmp_path / "config.json"
    with config_path.open("w") as f:
        json.dump(config, f)
        
    with pytest.raises(ValueError, match="Invalid array size: -5"):
        run_experiment(config_path, tmp_path)

def test_unsupported_input_type_rejected(tmp_path):
    config = {
        "measured_repetitions": 1,
        "array_sizes": [10],
        "input_types": ["unsupported_type"],
        "starting_algorithms": ["quick_sort"]
    }
    config_path = tmp_path / "config.json"
    with config_path.open("w") as f:
        json.dump(config, f)
        
    with pytest.raises(ValueError, match="Unsupported input_type: unsupported_type"):
        run_experiment(config_path, tmp_path)

def test_unsupported_starting_algorithm_rejected(tmp_path):
    config = {
        "measured_repetitions": 1,
        "array_sizes": [10],
        "input_types": ["random"],
        "starting_algorithms": ["heap_sort"]
    }
    config_path = tmp_path / "config.json"
    with config_path.open("w") as f:
        json.dump(config, f)
        
    with pytest.raises(ValueError, match="Unsupported starting algorithm: heap_sort"):
        run_experiment(config_path, tmp_path)

def test_quick_mode_overrides_config(tmp_path):
    config = {
        "random_seed": 42,
        "warmup_runs": 100,
        "measured_repetitions": 1000,
        "array_sizes": [10, 20, 30, 40],
        "input_types": ["random"],
        "starting_algorithms": ["quick_sort"],
    }
    config_path = tmp_path / "config.json"
    with config_path.open("w") as f:
        json.dump(config, f)
    
    out_dir = tmp_path / "results"
    num_runs, stats, csv_path, json_path = run_experiment(config_path, out_dir, is_quick=True)
    
    # Quick mode forces measured_repetitions to 2, array_sizes to [100, 500]
    # For 1 input type and 1 starting algorithm -> 2 sizes * 1 type * 1 algo * 2 reps = 4 runs
    assert num_runs == 4
