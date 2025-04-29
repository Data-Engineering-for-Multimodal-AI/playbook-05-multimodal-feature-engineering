# Multimodal Feature Pipeline

This directory contains the core pipelines for extracting and fusing multimodal features from the Waymo Motion and Reasoning datasets.

## Files

- **combined_feature_pipeline.py**: Pipeline that processes both motion and reasoning datasets, extracts features from scenarios common to both, and combines them.
- **multimodal_feature_pipeline.py**: Advanced pipeline that implements multimodal feature extraction, fusion, and model training.

## Usage

```bash
# Run the combined feature pipeline (for common scenarios)
python combined_feature_pipeline.py

# Process a specific number of scenarios
python combined_feature_pipeline.py --max_scenarios 50

# Run the full multimodal feature pipeline
python multimodal_feature_pipeline.py

# Generate yield prediction model
python multimodal_feature_pipeline.py --train_yield_model
```

## Key Features

- **Multimodal Feature Extraction**: Extracts motion features from trajectory data and text embeddings from Q&A annotations.
- **Feature Fusion**: Aligns and combines features from different modalities.
- **Text Embedding**: Uses Sentence-BERT for generating embeddings from natural language Q&A pairs.
- **Early Fusion Layer**: Implements early fusion of motion and text features.
- **Yield Prediction Model**: Trains a model to predict yielding behavior based on multimodal features.

## Input Requirements

These pipelines expect:
- Motion dataset in TFRecord format
- Reasoning dataset in TFRecord format
- Common scenarios between the datasets (identified by analyze_common_scenarios.py)

## Outputs

The multimodal pipelines generate several outputs:

- Combined feature CSV files
- Text embeddings for reasoning data
- Fused multimodal feature vectors
- Trained yield prediction models
- Feature importance visualizations
- Performance metrics and analysis reports

All outputs are saved to the `outputs/combined_features/` and `outputs/multimodal_features/` directories. 