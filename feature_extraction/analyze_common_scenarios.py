"""Runs the top-level analyze_common_scenarios.py (this folder used to hold an identical copy).

Kept so both `python analyze_common_scenarios.py` and `python feature_extraction/analyze_common_scenarios.py` work. The script runs from
the repository root, where its relative data/ and outputs/ paths point.
"""
import os
import runpy
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(ROOT)
sys.path.insert(0, ROOT)
runpy.run_path(os.path.join(ROOT, "analyze_common_scenarios.py"), run_name="__main__")
