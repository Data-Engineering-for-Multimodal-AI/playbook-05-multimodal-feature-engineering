# Waymo Multimodal Feature Engineering Pipeline

This repository contains a comprehensive feature engineering pipeline for extracting, processing, and fusing features from the Waymo Motion and Reasoning datasets. The pipeline enables the creation of multimodal features that combine motion trajectory data with natural language annotations, supporting downstream tasks like yield prediction.

## Overview

The pipeline processes two complementary datasets from Waymo:
1. **Waymo Motion Dataset**: Contains motion trajectories of vehicles, pedestrians, and cyclists
2. **Waymo Reasoning Dataset**: Contains natural language questions and answers about driving scenarios

By aligning and fusing these datasets, the pipeline creates rich multimodal features that capture both physical motion and semantic understanding of driving scenarios.

## Key Features

- **Data Inspection**: Tools for exploring and analyzing the Waymo datasets
- **Common Scenario Analysis**: Identification and processing of scenarios common to both datasets
- **Feature Extraction**: Extraction of motion features and text embeddings
- **Multimodal Fusion**: Alignment and combination of features across modalities
- **Model Training**: Example yield prediction model using the fused features
- **Scalable Implementation**: Both local processing and cloud-based distributed processing options

## Repository Structure

```
pb5-mm-feature-engineering-pipeline/
│
├── data_inspection/                  # Tools for dataset inspection
│   ├── inspect_waymo_data.py         # Basic data inspection utility
│   ├── data_inspection.py            # Detailed inspection for both datasets
│   └── validation_data_analysis.py   # Analysis of validation datasets
│
├── feature_extraction/               # Feature extraction modules
│   ├── extract_validation_ids.py     # Extract scenario IDs from validation sets
│   ├── analyze_common_scenarios.py   # Find common scenarios between datasets
│   ├── waymo_feature_pipeline.py     # Pipeline for motion feature extraction
│   └── generate_validation_report.py # Generate report on validation data
│
├── multimodal_pipeline/              # Multimodal feature processing
│   ├── combined_feature_pipeline.py  # Pipeline combining both datasets
│   └── multimodal_feature_pipeline.py # Complete multimodal feature pipeline
│
├── gcp_implementation/               # Google Cloud implementation
│   ├── dataflow_pipeline.py          # Distributed processing using Dataflow
│   ├── feature_store_setup.py        # Vertex AI Feature Store configuration
│   ├── deploy_endpoint.py            # Model endpoint deployment
│   ├── train_vertex_ai.py            # Model training on Vertex AI
│   ├── launch_job.py                 # Job launch utility
│   └── README.md                     # GCP-specific documentation
│
├── outputs/                          # Output directory for analysis results
│
├── data/                             # Data directory (see Data section below)
│
├── requirements.txt                  # Python dependencies
├── README.md                         # Main documentation
└── COMPLETION_SUMMARY.md             # Summary of pipeline capabilities
```

## Installation

### Prerequisites

- Python 3.8 or higher
- TensorFlow 2.8+
- Access to Waymo Motion and Reasoning datasets

### Setup

1. Clone the repository:
```bash
git clone https://github.com/Data-Engineering-for-Multimodal-AI/pb5-mm-feature-engineering-pipeline.git
cd pb5-mm-feature-engineering-pipeline
```

2. Create and activate a virtual environment:
```bash
python -m venv .venv
source .venv/bin/activate  # Linux/Mac
# or
.venv\Scripts\activate     # Windows
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## Data

This pipeline works with the Waymo Open Motion Dataset and the Waymo Reasoning Dataset. You need to:

1. Download the datasets from the [Waymo Open Dataset website](https://waymo.com/open/)
2. Place the TFRecord files in the appropriate directories:
   - Motion dataset: `data/motion/tf_example/validation_interactive/`
   - Reasoning dataset: `data/reasoning/validation_interactive/`

## Usage

### Local Pipeline

#### Data Inspection

```bash
# Inspect basic dataset properties
python inspect_waymo_data.py

# Generate detailed analysis of validation datasets
python validation_data_analysis.py
```

#### Feature Extraction

```bash
# Extract and analyze common scenarios between datasets
python analyze_common_scenarios.py

# Run the combined feature pipeline
python combined_feature_pipeline.py

# Run the complete multimodal feature pipeline
python multimodal_feature_pipeline.py
```

### Google Cloud Implementation

For processing large datasets, use the Google Cloud implementation:

```bash
# Launch the feature processing pipeline on Dataflow
python gcp_implementation/launch_job.py \
  --project_id=YOUR_PROJECT_ID \
  --region=us-central1 \
  --staging_location=gs://your-bucket/staging \
  --temp_location=gs://your-bucket/temp \
  feature-pipeline \
  --motion_data_path=gs://your-bucket/waymo-motion-data/* \
  --reasoning_data_path=gs://your-bucket/waymo-reasoning-data/* \
  --output_path=gs://your-bucket/output \
  --max_scenarios=1000
```

See the [GCP Implementation README](gcp_implementation/README.md) for more details on cloud-based processing.

## Pipeline Workflow

1. **Data Inspection**: Explore and understand the structure of both datasets
2. **Scenario Identification**: Identify scenarios common to both datasets
3. **Feature Extraction**: Extract motion features and generate text embeddings
4. **Feature Fusion**: Align and combine features from both modalities
5. **Model Training**: Train a yield prediction model using the fused features
6. **Evaluation**: Evaluate model performance and feature importance

## Results and Outputs

The pipeline generates several outputs:
- Scenario ID comparisons between datasets
- Feature analysis reports
- Combined multimodal features
- Model evaluation metrics
- Feature importance visualizations

These outputs are saved to the `outputs/` directory.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This repository is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- Waymo Open Dataset team for providing the datasets
- TensorFlow and PyTorch teams for their ML frameworks 