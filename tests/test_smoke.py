"""
Smoke test: run every script, in README order, on a small synthetic dataset.

The Waymo data can't be shared, so tests/make_fixture.py writes data in the same
formats. CPU only, no Waymo account. The scripts run in a temporary folder, so
nothing is written to outputs/. The Sentence-BERT model is not downloaded
(HF_HUB_OFFLINE=1): the pipelines fall back to running without text embeddings,
and test_text_embedding_inputs covers the embedding inputs.
Run with `make smoke`.
"""

import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tests"))

from make_fixture import make_scenario, write_fixture  # noqa: E402

README_ORDER = [
    "inspect_waymo_data.py",
    "data_inspection.py",
    "validation_data_analysis.py",
    "extract_validation_ids.py",
    "analyze_common_scenarios.py",
    "generate_validation_report.py",
    "combined_feature_pipeline.py",
    "waymo_feature_pipeline.py",
    "multimodal_feature_pipeline.py",
]


@pytest.fixture(scope="module")
def run_dir(tmp_path_factory):
    data = write_fixture(str(tmp_path_factory.mktemp("data")), scenarios=12, seed=0)
    work = tmp_path_factory.mktemp("work")
    env = {**os.environ, "WAYMO_DATA_DIR": data, "HF_HUB_OFFLINE": "1",
           "TF_CPP_MIN_LOG_LEVEL": "2", "MPLBACKEND": "Agg"}
    logs = {}
    for script in README_ORDER:
        result = subprocess.run([sys.executable, str(ROOT / script)], cwd=work, env=env,
                                capture_output=True, text=True, timeout=600)
        logs[script] = result
    return work, logs


@pytest.mark.parametrize("script", README_ORDER)
def test_script_runs(run_dir, script):
    _, logs = run_dir
    result = logs[script]
    assert result.returncode == 0, result.stdout[-3000:] + result.stderr[-3000:]


def test_common_scenarios_found(run_dir):
    work, _ = run_dir
    comparison = json.loads((work / "outputs/validation_analysis/scenario_id_comparison.json").read_text())
    assert len(comparison["common_ids"]) == 10  # 12 scenarios, 2 without a reasoning file


def test_yield_model_results(run_dir):
    work, logs = run_dir
    out = logs["multimodal_feature_pipeline.py"].stdout
    assert "majority_class_baseline" in out
    results = json.loads((work / "outputs/multimodal_features/yield_model_results.json").read_text())
    assert results["scenarios"] == 10
    assert set(results["test_comparison"]) == {"majority_class_baseline", "motion_only", "motion_plus_reasoning"}
    assert "int_question_count" not in results["feature_importance"]


def test_yield_label_uses_each_agents_own_future():
    """Regression test: future arrays are [agents, 80]; the label must come from that agent's row."""
    import multimodal_feature_pipeline as mfp

    rng = np.random.default_rng(1)
    example, _, brakes = make_scenario(rng, "regression0001", ego_id=100)
    pipeline = mfp.MultimodalFeaturePipeline.__new__(mfp.MultimodalFeaturePipeline)
    agents = pipeline.agent_features_from_example(example, "regression0001")
    assert len(agents) == len(brakes) - 1  # the last agent is padding
    for agent in agents:
        assert agent["is_yielding"] == bool(brakes[agent["agent_id"] - 100])


def test_text_embedding_inputs():
    """Interaction Q&As describe the future, so they must not be embedded."""
    import multimodal_feature_pipeline as mfp

    class RecordingModel:
        def encode(self, text):
            self.text = text
            return np.zeros(4)

    pipeline = mfp.MultimodalFeaturePipeline.__new__(mfp.MultimodalFeaturePipeline)
    pipeline.text_model = RecordingModel()
    pipeline.embed_qa_text({"qa_texts": ["Interaction: Will #0 yield? Yes, #0 will yield.",
                                         "Environment: Is there a light? Yes."]})
    assert "yield" not in pipeline.text_model.text
    assert "Environment" in pipeline.text_model.text
