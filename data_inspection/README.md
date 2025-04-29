# Data Inspection Tools

This directory contains utilities for inspecting and analyzing the Waymo Motion and Reasoning datasets.

## Files

- **inspect_waymo_data.py**: Basic inspection utility to explore the structure and content of both datasets.
- **data_inspection.py**: Detailed inspection tool that provides statistics and summaries of the datasets.
- **validation_data_analysis.py**: Specialized analysis of the validation datasets, focusing on question types, distributions, and patterns.

## Usage

```bash
# Basic dataset inspection
python inspect_waymo_data.py

# Detailed data inspection
python data_inspection.py

# Comprehensive validation dataset analysis
python validation_data_analysis.py
```

## Key Outputs

- Dataset statistics (number of scenarios, agents, etc.)
- Question type distributions and counts
- Visualizations of data patterns
- Summary reports of dataset contents

All analysis results are saved to the `outputs/validation_analysis/` directory. 