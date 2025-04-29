# Feature Extraction Tools

This directory contains tools for extracting, comparing, and analyzing features from the Waymo Motion and Reasoning datasets.

## Files

- **extract_validation_ids.py**: Extracts and compares scenario IDs between motion and reasoning datasets to identify common scenarios.
- **analyze_common_scenarios.py**: Analyzes scenarios that appear in both the motion and reasoning datasets, providing detailed statistics and insights.
- **waymo_feature_pipeline.py**: Primary pipeline for extracting motion features from the Waymo Motion Dataset.
- **generate_validation_report.py**: Generates a comprehensive report summarizing all analysis results and findings.

## Usage

```bash
# Extract and compare scenario IDs
python extract_validation_ids.py

# Analyze common scenarios
python analyze_common_scenarios.py

# Run the motion feature extraction pipeline
python waymo_feature_pipeline.py

# Generate a comprehensive validation report
python generate_validation_report.py
```

## Key Features

- Identification of common scenarios between datasets
- Extraction of motion features (position, velocity, heading, etc.)
- Analysis of agent interactions and behaviors
- Comprehensive reporting of findings and statistics

## Outputs

The feature extraction tools generate several outputs:

- Lists of scenario IDs from both datasets
- Comparison reports of common scenarios
- Extracted motion features in structured formats
- Comprehensive validation reports with statistics and visualizations

All outputs are saved to the `outputs/validation_analysis/` and `outputs/features/` directories. 