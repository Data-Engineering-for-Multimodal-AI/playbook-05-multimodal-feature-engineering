"""Shared helpers for the Try this experiments.

Builds the fused agent table once with the pipeline's own code
(MultimodalFeaturePipeline in multimodal_feature_pipeline.py) and caches it in
outputs/experiments/, so the three experiments share one pass over the data.
Besides the pipeline's text embedding (env, ego and sur Q&As), it stores a
second embedding that also includes the interaction (int) Q&As, as the first
edition did.

Data: the Waymo data in $WAYMO_DATA_DIR (or data/). Without it, a synthetic
dataset in the Waymo formats (tests/make_fixture.py), and every table says so.
"""

import contextlib
import hashlib
import io
import os
import shutil
import subprocess
import sys

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CACHE_DIR = os.path.join(ROOT, "outputs", "experiments")
SYNTHETIC_DIR = os.path.join(ROOT, "synthetic", "experiments-data")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
os.environ.setdefault("MPLBACKEND", "Agg")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

MOTION_COLS = ["speed", "distance_to_ego"]
COUNT_COLS = ["env_question_count", "ego_question_count", "sur_question_count"]


def data_dir():
    """(folder, is_real). Real when the folder has both datasets, as make run checks."""
    d = os.environ.get("WAYMO_DATA_DIR", os.path.join(ROOT, "data"))
    motion = os.path.join(d, "motion", "tf_example", "validation_interactive")
    reasoning = os.path.join(d, "reasoning", "validation_interactive")
    if os.path.isdir(motion) and any(".tfrecord" in f for f in os.listdir(motion)) and os.path.isdir(reasoning):
        return d, True
    if not os.path.isdir(os.path.join(SYNTHETIC_DIR, "reasoning")):
        subprocess.run([sys.executable, os.path.join(ROOT, "tests", "make_fixture.py"), SYNTHETIC_DIR,
                        "--scenarios", "40"], check=True, stdout=subprocess.DEVNULL)
    return SYNTHETIC_DIR, False


def banner(is_real, folder, df):
    kind = "Waymo data" if is_real else "SYNTHETIC data (tests/make_fixture.py), not real driving data"
    print(f"Data: {kind} in {folder}")
    print(f"{df['scenario_id'].nunique()} scenarios, {len(df)} agents with a label, "
          f"{df['is_yielding'].mean():.1%} yielding\n")


def fused_table():
    """The fused agent table with both text embeddings. Returns (df, text_cols, text_int_cols, is_real, folder)."""
    folder, is_real = data_dir()
    key = hashlib.sha1(os.path.abspath(folder).encode()).hexdigest()[:10]
    cache = os.path.join(CACHE_DIR, f"fused_{key}.pkl")
    if os.path.exists(cache):
        df = pd.read_pickle(cache)
    else:
        df = build(folder)
        os.makedirs(CACHE_DIR, exist_ok=True)
        df.to_pickle(cache)
    text_cols = [c for c in df.columns if c.startswith("text_embedding_") and c != "text_embedding_dim"]
    text_int_cols = [c for c in df.columns if c.startswith("textint_")]
    df = df.dropna(subset=["avg_future_speed", "is_yielding"]).reset_index(drop=True)
    df["is_yielding"] = df["is_yielding"].astype(bool)
    return df, text_cols, text_int_cols, is_real, folder


def build(folder):
    print(f"Building the fused features from {folder} (once; cached in outputs/experiments/)...",
          flush=True)
    sys.path.insert(0, ROOT)
    cwd = os.getcwd()
    scratch = os.path.join(CACHE_DIR, "pipeline-run")
    # The pipeline reuses outputs/validation_analysis/scenario_id_comparison.json from an
    # earlier run, whatever data that run used. Start from a clean folder.
    shutil.rmtree(scratch, ignore_errors=True)
    os.makedirs(scratch)
    os.chdir(scratch)  # the pipeline writes outputs/ relative to the working directory
    try:
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            import multimodal_feature_pipeline as mfp
            p = mfp.MultimodalFeaturePipeline(
                motion_path=os.path.join(folder, "motion", "tf_example", "validation_interactive"),
                reasoning_path=os.path.join(folder, "reasoning", "validation_interactive"),
                output_dir="outputs/multimodal_features", feature_store_dir="outputs/feature_store")
            df = p.process_scenarios()
            if df is None:
                raise SystemExit(f"No scenarios found in both datasets in {folder}.")
            # The first edition's text: the same Q&As plus the interaction ones
            with_int = {}
            for sid, r in p.reasoning_features.items():
                texts = r["qa_texts"]
                current = [t for t in texts if not t.startswith("Interaction:")][:10]
                interaction = [t for t in texts if t.startswith("Interaction:")]
                with_int[sid] = p.text_model.encode(" ".join(current + interaction))
    finally:
        os.chdir(cwd)
    emb = np.stack([with_int[s] for s in df["scenario_id"]])
    extra = pd.DataFrame(emb, columns=[f"textint_{i}" for i in range(emb.shape[1])])
    return pd.concat([df.reset_index(drop=True), extra], axis=1)


def evaluate(df, cols, train_idx, test_idx):
    """The pipeline's model: StandardScaler + LogisticRegression(class_weight='balanced').
    Returns (balanced accuracy, F1 for yielding) on the test rows."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import balanced_accuracy_score, f1_score
    from sklearn.preprocessing import StandardScaler
    y = df["is_yielding"]
    scaler = StandardScaler()
    x_train = scaler.fit_transform(df.loc[train_idx, cols].fillna(0))
    x_test = scaler.transform(df.loc[test_idx, cols].fillna(0))
    model = LogisticRegression(max_iter=1000, class_weight="balanced")
    model.fit(x_train, y.iloc[train_idx])
    pred = model.predict(x_test)
    return (balanced_accuracy_score(y.iloc[test_idx], pred),
            f1_score(y.iloc[test_idx], pred, zero_division=0))


def scenario_split(df, seed=42):
    """The pipeline's split: 30% of the scenarios for testing."""
    from sklearn.model_selection import GroupShuffleSplit
    split = GroupShuffleSplit(n_splits=1, test_size=0.3, random_state=seed)
    return next(split.split(df, df["is_yielding"], df["scenario_id"]))
