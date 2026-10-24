"""Runs the top-level waymo_feature_pipeline.py (this folder used to hold an identical copy).

Kept so both `python waymo_feature_pipeline.py` and `python feature_extraction/waymo_feature_pipeline.py` work. The script runs from
the repository root, where its relative data/ and outputs/ paths point.
"""
import os
import runpy
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(ROOT)
sys.path.insert(0, ROOT)
runpy.run_path(os.path.join(ROOT, "waymo_feature_pipeline.py"), run_name="__main__")
