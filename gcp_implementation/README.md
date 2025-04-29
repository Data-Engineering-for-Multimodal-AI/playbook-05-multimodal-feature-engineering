# Google Cloud Implementation for Waymo Multimodal Feature Pipeline

This directory contains the implementation components for deploying the Waymo multimodal feature pipeline on Google Cloud Platform (GCP). The implementation leverages GCP services such as Dataflow, Vertex AI Feature Store, and other managed services to process Waymo datasets at scale.

## Components

This implementation includes the following components:

- **Data Processing Pipeline (`dataflow_pipeline.py`)**: An Apache Beam pipeline for extracting and processing features from Waymo Motion and Reasoning datasets at scale using Google Cloud Dataflow.

- **Feature Store Setup (`feature_store_setup.py`)**: A utility for creating and configuring a Vertex AI Feature Store to store and serve features extracted from Waymo datasets.

- **Model Endpoint Deployment (`deploy_endpoint.py`)**: A script for deploying trained models as endpoints on Vertex AI for online prediction.

- **Job Launcher (`launch_job.py`)**: A command-line utility for launching Dataflow jobs with appropriate configurations.

## Setup

### Prerequisites

- Google Cloud SDK installed and configured
- Python 3.8 or higher
- Access to Google Cloud Platform with appropriate permissions
- Access to Waymo datasets stored in Google Cloud Storage

### Installation

1. Install required Python dependencies:

```bash
pip install -r requirements.txt
```

2. Configure Google Cloud SDK:

```bash
gcloud auth login
gcloud config set project YOUR_PROJECT_ID
```

## Usage

### Data Processing

To launch the feature processing pipeline on Dataflow:

```bash
python launch_job.py \
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

### Feature Store Setup

To set up the Vertex AI Feature Store:

```bash
python feature_store_setup.py \
  --project_id=YOUR_PROJECT_ID \
  --location=us-central1 \
  --feature_store_id=waymo-multimodal-features \
  --schema_path=gs://your-bucket/output/schema.json
```

To configure batch serving and ingest features:

```bash
python feature_store_setup.py \
  --project_id=YOUR_PROJECT_ID \
  --location=us-central1 \
  --feature_store_id=waymo-multimodal-features \
  --configure_batch_serving \
  --batch_destination=YOUR_PROJECT_ID.waymo_features.combined_features \
  --entity_type=agent
```

### Model Deployment

To deploy a trained model to Vertex AI:

```bash
python deploy_endpoint.py \
  --project_id=YOUR_PROJECT_ID \
  --location=us-central1 \
  --model_id=waymo-yield-prediction-model \
  --endpoint_name=waymo-yield-prediction \
  --machine_type=n1-standard-2 \
  --min_replica_count=1 \
  --max_replica_count=5
```

## Architecture

The GCP implementation follows this overall architecture:

1. **Data Ingestion**: Waymo Motion and Reasoning datasets are read from Google Cloud Storage using Apache Beam and Dataflow.

2. **Feature Extraction**: The Dataflow pipeline extracts motion features from trajectory data and generates text embeddings from Q&A pairs.

3. **Feature Fusion**: Motion and text features are aligned and combined for each scenario/agent.

4. **Feature Storage**: Processed features are stored in Vertex AI Feature Store for efficient retrieval.

5. **Model Training**: Features are used to train models for tasks like yield prediction.

6. **Model Deployment**: Trained models are deployed as endpoints on Vertex AI for online prediction.

## Key Files

- `dataflow_pipeline.py`: Apache Beam pipeline for feature extraction
- `feature_store_setup.py`: Vertex AI Feature Store configuration
- `deploy_endpoint.py`: Model deployment to Vertex AI
- `launch_job.py`: Job launcher utility

## Scaling Considerations

The implementation is designed to scale to the full Waymo dataset:

- Uses Dataflow's auto-scaling capabilities to process large datasets
- Implements efficient TFRecord parsing with appropriate parallelization
- Configures appropriate machine types and worker counts for different jobs
- Handles feature batching for efficient processing
- Includes proper error handling and logging

## Monitoring

Monitoring can be done through the Google Cloud Console:

- Dataflow jobs: https://console.cloud.google.com/dataflow
- Vertex AI Feature Store: https://console.cloud.google.com/vertex-ai/feature-store
- Vertex AI Endpoints: https://console.cloud.google.com/vertex-ai/endpoints

## Troubleshooting

Common issues and solutions:

- **Dataflow job failures**: Check worker logs in Cloud Logging for specific errors
- **Memory issues**: Increase worker machine type or adjust batch size
- **Rate limiting**: Configure appropriate QPS for feature serving
- **Missing features**: Verify feature schema matches extracted features

For additional support, consult the GCP documentation for the respective services.

## References

- [Apache Beam Programming Guide](https://beam.apache.org/documentation/programming-guide/)
- [Vertex AI Feature Store Documentation](https://cloud.google.com/vertex-ai/docs/featurestore)
- [Vertex AI Model Deployment](https://cloud.google.com/vertex-ai/docs/predictions/deploy-model-api)
- [Google Cloud Dataflow Documentation](https://cloud.google.com/dataflow/docs) 