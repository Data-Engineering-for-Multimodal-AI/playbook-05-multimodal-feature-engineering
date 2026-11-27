# Changelog

## v1.1.0 (2026-11-26)

Nothing the book shows has changed.

### Added
- `make show` prints one agent's feature-store entry next to the Q&As it was built from
  (`scripts/show_agent.py`).
- `make check-data DIR=...` and `make load DIR=...`: check a folder of scenarios in the Waymo layout
  and run `multimodal_feature_pipeline.py` on it (`scripts/load_scenarios.py`). `make load` clears
  `outputs/feature_store/` and the cached scenario ID list first. Tests in `tests/test_own_data.py`.
- Three measured experiments: `make try-leakage`, `make try-split`, `make try-text-dims`
  (`experiments/`). They use the Waymo data when it is there and synthetic data otherwise.
- README: numbered Run it steps, time tiers, Try this, Use your own data, What you end up with,
  Before production, What it costs.
- CI: runs the Run it steps on synthetic data with checks on their output, the three experiments,
  the own-data tests, and prints the time to first output in a clean devcontainer.
- `make setup`, `make run` and `make clean`. `make run` runs `multimodal_feature_pipeline.py`. With
  the Waymo data in `data/` (or `$WAYMO_DATA_DIR`), it uses the real data. Without it, it writes
  synthetic data in the Waymo formats with `tests/make_fixture.py` into the git-ignored `synthetic/`
  folder, runs the pipeline there, and says that the output is from synthetic data and how to fetch
  the real data. `make smoke` and `make data` are unchanged.
- `.devcontainer/` so the repo opens in GitHub Codespaces with dependencies installed.
- CI: a weekly scheduled run, a `make run` step, and a job that builds the devcontainer and runs
  `make run` inside it.
- `DATA_SOURCES.md`: the datasets and models this playbook uses, with their licenses and citations.
- `make run` writes `.playbook-done` when it passes (ignored by git). `make card` in the
  organization's `.github` repository reads these files and draws a completion card.

### Changed
- README: "What you'll build" is now "How it works"; "Quick start" is now "Run it the way the
  book does", with its content unchanged. The first-edition note points to it.
- README opening: a title that says what you'll do, three sentences on why, the example and what you'll do, then the data engineering problem and what you'll work with.
- Book page numbers now match the printed book (they were Early Release page numbers).
- README: a header image (light and dark), badges for CI, Python, license and Codespaces, a diagram at
  the top of "How it works", and a footer that links the book, all playbooks, and the previous and
  next playbook.
- README: a Troubleshooting section with three common problems and their fixes.

## 2026.10.1 (2026-10-27)

### Changed
- README, after checking against the book (Early Release, pages 166–173): page numbers; the
  differences table's left column is the first edition.

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
- `multimodal_feature_pipeline.py` processes every scenario found in both datasets by default (the first edition used 10, which leaves no yielding agents in the test set). `--max-scenarios N` processes fewer.
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
