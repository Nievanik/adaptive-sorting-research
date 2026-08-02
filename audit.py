import pandas as pd
import json
import sys

def audit():
    df = pd.read_csv('results/phase7/benchmark_raw.csv')
    print(f"Columns: {list(df.columns)}")
    print(f"Number of rows: {len(df)}")
    print(f"Strategies: {df['strategy'].unique().tolist()}")
    print(f"Input distributions: {df['input_type'].unique().tolist()}")
    print(f"Array sizes: {df['array_size'].unique().tolist()}")
    print(f"Oracle info available: {'oracle_action' in df.columns and not df['oracle_action'].isna().all()}")
    timing_fields = [c for c in df.columns if 'runtime' in c or 'ns' in c or 'ms' in c]
    print(f"Timing fields: {timing_fields}")
    grouping_dims = ['strategy', 'array_size', 'input_type', 'starting_algorithm']
    print(f"Grouping dimensions present: {[g for g in grouping_dims if g in df.columns]}")
    missing_vals = df.isna().sum()
    print(f"Missing values:\n{missing_vals[missing_vals > 0]}")
    
    invalid = df[df['is_sorted'] == False]
    print(f"Invalid/Failed rows (is_sorted=False): {len(invalid)}")

if __name__ == '__main__':
    audit()
