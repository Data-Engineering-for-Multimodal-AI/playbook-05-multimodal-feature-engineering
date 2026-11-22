.PHONY: setup run run-real run-synthetic smoke data clean

# Uniform targets for The Playbooks. The book's commands (gcloud auth login,
# bash scripts/fetch_data.sh, pip install -r requirements.txt, then the scripts in the README's
# Quick start, ending with python multimodal_feature_pipeline.py) still work as printed.

# Folder with the Waymo data, as in the scripts: $WAYMO_DATA_DIR, or data/ by default.
DATA_DIR := $(or $(WAYMO_DATA_DIR),data)
# Synthetic data and the outputs of a run on it (git-ignored). Never real driving data.
SYNTHETIC := synthetic
RUN_ENV := MPLBACKEND=Agg TF_CPP_MIN_LOG_LEVEL=2

# Install the locked dependencies into .venv.
setup:
	uv sync --frozen

# Run the feature pipeline (multimodal_feature_pipeline.py). With the Waymo data in $(DATA_DIR)/,
# it uses the real data. Without it, it uses synthetic data in the Waymo formats and says so.
run: setup
	@if ls "$(DATA_DIR)"/motion/tf_example/validation_interactive/*.tfrecord* >/dev/null 2>&1 \
		&& [ -d "$(DATA_DIR)/reasoning/validation_interactive" ]; then \
		$(MAKE) --no-print-directory run-real; \
	else \
		$(MAKE) --no-print-directory run-synthetic; \
	fi

# The real data: the book's command, writing to outputs/.
run-real:
	@echo "Using the Waymo data in $(DATA_DIR)/."
	WAYMO_DATA_DIR="$(DATA_DIR)" $(RUN_ENV) uv run --frozen python multimodal_feature_pipeline.py

# Synthetic data from tests/make_fixture.py, in $(SYNTHETIC)/data. Outputs go to $(SYNTHETIC)/outputs.
run-synthetic:
	@echo "No Waymo data found in $(DATA_DIR)/. This run uses SYNTHETIC data in the Waymo formats."
	rm -rf "$(SYNTHETIC)"
	uv run --frozen python tests/make_fixture.py "$(SYNTHETIC)/data"
	cd "$(SYNTHETIC)" && WAYMO_DATA_DIR=data $(RUN_ENV) uv run --frozen python "$(CURDIR)/multimodal_feature_pipeline.py"
	@echo ""
	@echo "NOTE: the output above is from SYNTHETIC data (tests/make_fixture.py), not real driving data."
	@echo "Its numbers say nothing about the Waymo data. To run on the real data: register at"
	@echo "https://waymo.com/open, run gcloud auth login, then bash scripts/fetch_data.sh, then make run."

# End-to-end check on synthetic data in the Waymo formats (CPU, no Waymo account).
smoke:
	uv sync --frozen
	uv run --frozen pytest -q tests/test_smoke.py

# Download the real data (needs a waymo.com/open registration; see data/README.md).
data:
	bash scripts/fetch_data.sh

# Remove generated files: the synthetic data, the scripts' outputs and test caches.
# The Waymo data in data/ and outputs/README.md are kept.
clean:
	rm -rf "$(SYNTHETIC)" .pytest_cache common_scenario_ids.txt
	find outputs -mindepth 1 -maxdepth 1 ! -name README.md -exec rm -rf {} +
