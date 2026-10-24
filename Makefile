.PHONY: smoke data

# End-to-end check on synthetic data in the Waymo formats (CPU, no Waymo account).
smoke:
	uv sync --frozen
	uv run --frozen pytest -q tests/test_smoke.py

# Download the real data (needs a waymo.com/open registration; see data/README.md).
data:
	bash scripts/fetch_data.sh
