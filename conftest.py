"""
conftest.py — project root
Adds the repo root to sys.path so that pytest can import
top-level packages (api, ag_module, ocr_pipeline, detector, qc, etc.)
regardless of which Python interpreter runs pytest.
"""
import sys
import os

# Insert repo root as the first search path entry
sys.path.insert(0, os.path.dirname(__file__))
