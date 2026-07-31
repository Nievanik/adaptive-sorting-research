"""
conftest.py
-----------
Pytest configuration for ml/tests/.

Adds ml/src to sys.path so that `extract_dataset` and `generate_labels`
can be imported without needing manual sys.path manipulation inside test files.
"""

import sys
from pathlib import Path

# Ensure ml/src is importable for all tests in this directory
SRC_DIR = Path(__file__).resolve().parents[1] / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))
