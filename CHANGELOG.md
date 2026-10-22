# Changelog

## 2026.10 (2026-10-22): 2026 maintenance update

The printed version is tagged `first-edition`. That tag points to the printed code with the Waymo data removed from the repository history (see "Removed").

### Fixed
- **Yield labels** (`multimodal_feature_pipeline.py`): the WOMD future arrays are agent-major (`[agents, 80]`), but the code read them as `[80, agents]`. Each agent's `is_yielding` label mixed other agents' speeds. With the fix, 3.7% of agents are labeled as yielding (776 scenarios). In three runs of the printed command, the old code labeled 72–92% of the test agents as yielding.
- **Leaky model inputs**: the text embedding no longer includes the interaction Q&As (`int_q`/`int_a`), which describe what agents will do in the next 3 seconds, i.e. the thing being predicted. `int_question_count` is no longer a model input.
- **Train/test split by scenario**: agents in the same scenario share its text features, so a random row split put near-copies in both sets.
- **Honest evaluation**: the model is compared with a majority-class baseline and a motion-only model, using accuracy, balanced accuracy and F1. See "Expected output" in the README.
- `waymo_feature_pipeline.py`: only `future/x` was reshaped per agent, so reading the other trajectory arrays failed and the error handlers silently replaced real trajectories with made-up ones. All past and future arrays are now reshaped. Its speed and heading features use past and current state only, its text embedding uses ego and map Q&As instead of interaction Q&As, and it splits by scenario.
- `seaborn` was imported but missing from `requirements.txt`.
- On Linux, PyTorch now comes from the CPU-only index. PyPI's Linux build includes Triton, which crashed `waymo_feature_pipeline.py` and `multimodal_feature_pipeline.py` (segfault) when loaded after TensorFlow, and pulls several GB of CUDA libraries.
- `analyze_common_scenarios.py` exits with an error when there is nothing to analyze (it exited 0).
- `combined_feature_pipeline.py --max_scenarios 50` (printed in `multimodal_pipeline/README.md`) was ignored; it now works.
- Scenario lists are sorted, so "the first 10 scenarios" is the same on every run.
- `multimodal_feature_pipeline.py` reads the motion files once for all scenarios instead of once per scenario.
- README: correct clone URL, Python version and data formats (the reasoning data is JSON, not TFRecords).

### Changed
- The data folder can be set with `WAYMO_DATA_DIR` (default `data`, as before).
- `multimodal_feature_pipeline.py` takes `--max-scenarios` (default 10 as printed; `0` processes all scenarios found in both datasets).
- The scripts in `data_inspection/`, `feature_extraction/` and `multimodal_pipeline/` were identical copies of the root scripts. They are now wrappers that run the root copy, so both paths still work.
- `README_MULTIMODAL.md` and `COMPLETION_SUMMARY.md` are merged into `README.md`. Their claims of "high accuracy" are replaced by measured results.
- `gcp_implementation/` is labeled as an untested sketch. It uses the Vertex AI Feature Store (Legacy) API, which Google deprecated on February 17, 2026.
- The charts that were committed under `outputs/` moved to `docs/img/first-edition/`.

### Removed
- **The Waymo data (`data/reasoning/`, 126,357 files, about 1.1 GB) is removed from the repository and its history.** The Waymo terms allow sharing it only with registered users. `scripts/fetch_data.sh` and `data/README.md` explain how to download it with your own Waymo registration.
- `outputs/validation_analysis/validation_report.md` and `common_scenario_ids.txt` are no longer tracked; the scripts regenerate them.

### Added
- `pyproject.toml`, `uv.lock`, and a pinned `requirements.txt` exported from the lock (Python 3.12+).
- `scripts/fetch_data.sh` (`make data`).
- `tests/make_fixture.py`, a small synthetic dataset in the Waymo formats, and `tests/test_smoke.py` (`make smoke`), which runs every script on it.
- GitHub Actions smoke test on Python 3.12 and 3.13.
- `.env.example`.
