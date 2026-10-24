"""Runs the top-level generate_validation_report.py (this folder used to hold an identical copy).

Kept so both `python generate_validation_report.py` and `python feature_extraction/generate_validation_report.py` work. The script runs from
the repository root, where its relative data/ and outputs/ paths point.
"""
import os
import runpy
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(ROOT)
sys.path.insert(0, ROOT)
runpy.run_path(os.path.join(ROOT, "generate_validation_report.py"), run_name="__main__")
