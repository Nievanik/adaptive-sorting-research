"""
conftest.py
-----------
Pytest configuration for ml/tests/.

Adds ml/src to sys.path so that `extract_dataset` and `generate_labels`
can be imported without needing manual sys.path manipulation inside test files.
"""

import sys
from pathlib import Path

# Ensure adaptive-sorting-research root is in sys.path so that 'ml' package can be imported
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

