# Multimodal Feature Engineering Pipeline for Autonomous Driving

This project implements a comprehensive multimodal feature engineering pipeline for autonomous driving systems, focusing on the fusion of structured motion data and unstructured language annotations from the Waymo Open Dataset.

## Overview

The pipeline demonstrates how to combine two key modalities in autonomous driving:

1. **Structured Motion Data**: High-frequency telemetry data capturing agent motion over time from Waymo Motion TFRecords
2. **Unstructured Text Annotations**: Natural language descriptions and Q&A-style annotations from WOMD Reasoning JSON files

These modalities are aligned by scenario_id, embedded (text data), and fused to create rich multimodal feature vectors that can support downstream tasks like intent prediction and interaction forecasting.

## Pipeline Architecture

The pipeline consists of the following major components:

### 1. Data Ingestion and Inspection
- Load TFRecord and JSON data from Waymo Motion and Reasoning datasets
- Identify common scenarios between the datasets
- Extract agent-specific sequences: position, velocity, heading, timestamp

### 2. Motion Feature Extraction
- For each agent:
  - Compute position, velocity, heading
  - Derive distance to ego vehicle
  - Determine if agent is yielding based on future trajectory
  - Store features in structured format

### 3. Text Embedding
- Parse JSON files with Q&A annotations
- Extract natural language Q&A pairs
- Use Sentence-BERT (all-mpnet-base-v2) to convert text into 768-dimensional embeddings

### 4. Feature Fusion
- Join structured and unstructured features on scenario_id
- Concatenate the motion vector with the text embeddings
- Add metadata columns: timestamps, modality availability flags

### 5. Feature Store
- Simulates a feature store infrastructure
- Organizes features by scenario and agent ID
- Stores metadata for retrieval

### 6. Yield Prediction Model
- Implements a simple logistic regression model
- Uses fused features to predict if agents will yield
- Evaluates model performance
- Visualizes feature importance and text embeddings

## Implementation

Key files:
- `multimodal_feature_pipeline.py`: Main pipeline implementation
- `outputs/multimodal_features/`: Output directory for fused features
- `outputs/feature_store/`: Simulated feature store
- `outputs/multimodal_features/visualizations/`: Visualizations of embeddings and model results

## Requirements

- TensorFlow
- NumPy
- Pandas
- Sentence-Transformers
- scikit-learn
- UMAP
- Matplotlib
- seaborn

## Usage

```bash
# Install dependencies
pip install tensorflow numpy pandas sentence-transformers scikit-learn umap-learn matplotlib seaborn

# Run the pipeline
python multimodal_feature_pipeline.py
```

## Results

The pipeline demonstrates successful fusion of motion and text data, showing how:

1. Motion features provide the "what is happening" physically
2. Text annotations provide the "why" or the context
3. The combined representation enables better reasoning about agent intent

The yield prediction model achieves high accuracy by leveraging both modalities, with feature importance analysis showing the relative contribution of motion and semantic features.

## Future Work

- Implement more sophisticated fusion techniques (attention mechanisms, cross-modal transformers)
- Integrate with real-time inference systems
- Add uncertainty quantification
- Extend to more complex tasks like trajectory prediction with reasoning 