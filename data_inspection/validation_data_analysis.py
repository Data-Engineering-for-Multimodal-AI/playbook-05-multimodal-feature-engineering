"""Runs the top-level validation_data_analysis.py (this folder used to hold an identical copy).

Kept so both `python validation_data_analysis.py` and `python data_inspection/validation_data_analysis.py` work. The script runs from
the repository root, where its relative data/ and outputs/ paths point.
"""
import os
import runpy
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(ROOT)
sys.path.insert(0, ROOT)
runpy.run_path(os.path.join(ROOT, "validation_data_analysis.py"), run_name="__main__")
