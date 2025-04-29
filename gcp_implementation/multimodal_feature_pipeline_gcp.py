"""
GCP Implementation of the Waymo Multimodal Feature Pipeline 
using Google Cloud Platform services including Vertex AI.

This module provides adapters to connect the feature pipeline to GCP services.
"""

import os
import json
import logging
import argparse
import time
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple

import tensorflow as tf
import pandas as pd
import numpy as np
from google.cloud import aiplatform
from google.cloud import storage
from google.cloud.aiplatform_v1.types import entity_type as entity_type_pb2
from google.cloud.aiplatform_v1.types import feature as feature_pb2
from google.cloud.aiplatform_v1.types import feature_value as feature_value_pb2
from google.cloud.aiplatform_v1.types import featurestore as featurestore_pb2
from google.cloud.aiplatform_v1.types import featurestore_service as featurestore_service_pb2

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Default configuration
DEFAULT_PROJECT_ID = os.environ.get("GOOGLE_CLOUD_PROJECT", "waymo-feature-pipeline")
DEFAULT_LOCATION = "us-central1"
DEFAULT_FEATURE_STORE_ID = "waymo-feature-store"
DEFAULT_ENTITY_TYPE_ID = "agent"
DEFAULT_BUCKET_NAME = "waymo-processed-features"

class VertexAIFeatureStore:
    """Interface to Vertex AI Feature Store."""
    
    def __init__(self, 
                 project_id: str = DEFAULT_PROJECT_ID,
                 location: str = DEFAULT_LOCATION,
                 feature_store_id: str = DEFAULT_FEATURE_STORE_ID,
                 entity_type_id: str = DEFAULT_ENTITY_TYPE_ID):
        """Initialize the Feature Store connector.
        
        Args:
            project_id: GCP project ID
            location: GCP region
            feature_store_id: Feature Store ID
            entity_type_id: Entity type ID (e.g., 'agent')
        """
        self.project_id = project_id
        self.location = location
        self.feature_store_id = feature_store_id
        self.entity_type_id = entity_type_id
        
        # Initialize Vertex AI
        aiplatform.init(project=project_id, location=location)
        
        # Get the full resource name for the feature store
        self.feature_store_name = (
            f"projects/{project_id}/locations/{location}/featurestores/{feature_store_id}"
        )
        self.entity_type_name = f"{self.feature_store_name}/entityTypes/{entity_type_id}"
        
        # Create API clients
        self.fs_client = aiplatform.FeaturestoreServiceClient()
    
    def write_features(self, features_df: pd.DataFrame, entity_id_column: str = "entity_id") -> int:
        """Write features to Vertex AI Feature Store.
        
        Args:
            features_df: DataFrame containing features to write
            entity_id_column: Column name for entity IDs
            
        Returns:
            Number of entities written
        """
        try:
            # Prepare write request
            write_request = featurestore_service_pb2.WriteFeatureValuesRequest()
            write_request.entity_type = self.entity_type_name
            
            # Convert DataFrame to feature values
            entity_count = 0
            
            for _, row in features_df.iterrows():
                entity_id = str(row[entity_id_column])
                
                # Create entity feature values
                entity_feature_value = feature_value_pb2.FeatureValueList()
                entity_feature_value.entity_id = entity_id
                
                # Add feature values
                for col in features_df.columns:
                    if col == entity_id_column:
                        continue
                    
                    value = row[col]
                    if pd.isna(value):
                        continue
                    
                    feature_value = feature_value_pb2.FeatureValue()
                    
                    # Handle different data types
                    if isinstance(value, (int, np.int64, np.int32)):
                        feature_value.int64_value = int(value)
                    elif isinstance(value, (float, np.float64, np.float32)):
                        feature_value.double_value = float(value)
                    elif isinstance(value, (str, bytes)):
                        feature_value.string_value = str(value)
                    elif isinstance(value, bool):
                        feature_value.bool_value = value
                    elif isinstance(value, (list, np.ndarray)) and len(value) > 0:
                        if isinstance(value[0], (int, np.int64, np.int32)):
                            feature_value.int64_list_value.values.extend([int(v) for v in value])
                        elif isinstance(value[0], (float, np.float64, np.float32)):
                            feature_value.double_list_value.values.extend([float(v) for v in value])
                        elif isinstance(value[0], (str, bytes)):
                            feature_value.string_list_value.values.extend([str(v) for v in value])
                        elif isinstance(value[0], bool):
                            feature_value.bool_list_value.values.extend(value)
                        else:
                            logger.warning(f"Unsupported list value type for feature {col}: {type(value[0])}")
                            continue
                    else:
                        logger.warning(f"Unsupported value type for feature {col}: {type(value)}")
                        continue
                    
                    # Add feature value to entity
                    entity_feature_value.feature_values[col].CopyFrom(feature_value)
                
                # Add entity to request
                write_request.entity_feature_values.append(entity_feature_value)
                entity_count += 1
                
                # Write in batches of 100
                if entity_count % 100 == 0:
                    self.fs_client.write_feature_values(request=write_request)
                    write_request = featurestore_service_pb2.WriteFeatureValuesRequest()
                    write_request.entity_type = self.entity_type_name
            
            # Write any remaining entities
            if len(write_request.entity_feature_values) > 0:
                self.fs_client.write_feature_values(request=write_request)
            
            logger.info(f"Successfully wrote {entity_count} entities to Feature Store")
            return entity_count
            
        except Exception as e:
            logger.error(f"Error writing features to Feature Store: {str(e)}")
            raise
    
    def read_features(self, 
                     entity_ids: List[str], 
                     feature_ids: List[str] = None) -> pd.DataFrame:
        """Read features from Vertex AI Feature Store.
        
        Args:
            entity_ids: List of entity IDs to read
            feature_ids: List of feature IDs to read, or None for all features
            
        Returns:
            DataFrame containing features
        """
        try:
            # Prepare read request
            read_request = featurestore_service_pb2.ReadFeatureValuesRequest()
            read_request.entity_type = self.entity_type_name
            read_request.entity_ids.extend(entity_ids)
            
            if feature_ids:
                for feature_id in feature_ids:
                    read_request.feature_selector.id_matcher.ids.append(feature_id)
            
            # Read features
            response = self.fs_client.read_feature_values(request=read_request)
            
            # Convert response to DataFrame
            data = []
            
            for entity_view in response.entity_view:
                entity_id = entity_view.entity.entity_id
                
                # Extract feature values
                row = {"entity_id": entity_id}
                
                for feature_name, feature_value in entity_view.data.items():
                    if feature_value.HasField("int64_value"):
                        row[feature_name] = feature_value.int64_value
                    elif feature_value.HasField("double_value"):
                        row[feature_name] = feature_value.double_value
                    elif feature_value.HasField("string_value"):
                        row[feature_name] = feature_value.string_value
                    elif feature_value.HasField("bool_value"):
                        row[feature_name] = feature_value.bool_value
                    elif feature_value.HasField("int64_list_value"):
                        row[feature_name] = list(feature_value.int64_list_value.values)
                    elif feature_value.HasField("double_list_value"):
                        row[feature_name] = list(feature_value.double_list_value.values)
                    elif feature_value.HasField("string_list_value"):
                        row[feature_name] = list(feature_value.string_list_value.values)
                    elif feature_value.HasField("bool_list_value"):
                        row[feature_name] = list(feature_value.bool_list_value.values)
                
                data.append(row)
            
            return pd.DataFrame(data)
            
        except Exception as e:
            logger.error(f"Error reading features from Feature Store: {str(e)}")
            raise
    
    def batch_serve_features(self, 
                           entity_ids: List[str], 
                           feature_ids: List[str] = None,
                           destination_table_uri: str = None) -> Optional[str]:
        """Batch serve features from Vertex AI Feature Store.
        
        Args:
            entity_ids: List of entity IDs to serve
            feature_ids: List of feature IDs to serve, or None for all features
            destination_table_uri: BigQuery table URI for the output
            
        Returns:
            BigQuery table URI for the output
        """
        try:
            # Prepare batch serve request
            batch_serve_request = featurestore_service_pb2.BatchReadFeatureValuesRequest()
            batch_serve_request.entity_type = self.entity_type_name
            
            # Add entity IDs
            csv_source = featurestore_service_pb2.BatchReadFeatureValuesRequest.EntityIdSource()
            for entity_id in entity_ids:
                csv_source.entity_ids.append(entity_id)
            batch_serve_request.entity_id_source.CopyFrom(csv_source)
            
            # Add feature IDs
            if feature_ids:
                for feature_id in feature_ids:
                    batch_serve_request.feature_selector.id_matcher.ids.append(feature_id)
            
            # Set destination
            if destination_table_uri:
                bigquery_destination = featurestore_service_pb2.BatchReadFeatureValuesRequest.BigQueryDestination()
                bigquery_destination.output_uri = destination_table_uri
                batch_serve_request.destination.bigquery_destination.CopyFrom(bigquery_destination)
            
            # Submit batch serve request
            response = self.fs_client.batch_read_feature_values(request=batch_serve_request)
            
            # Wait for operation to complete
            operation = aiplatform.Operation(response.operation)
            result = operation.result()
            
            logger.info(f"Batch serve operation completed: {operation.operation.name}")
            
            # Return destination URI
            if hasattr(result, 'destination') and hasattr(result.destination, 'bigquery_destination'):
                return result.destination.bigquery_destination.output_uri
            return None
            
        except Exception as e:
            logger.error(f"Error batch serving features from Feature Store: {str(e)}")
            raise

class GCSConnector:
    """Interface to Google Cloud Storage."""
    
    def __init__(self, 
                 project_id: str = DEFAULT_PROJECT_ID,
                 bucket_name: str = DEFAULT_BUCKET_NAME):
        """Initialize the GCS connector.
        
        Args:
            project_id: GCP project ID
            bucket_name: GCS bucket name
        """
        self.project_id = project_id
        self.bucket_name = bucket_name
        
        # Initialize GCS client
        self.storage_client = storage.Client(project=project_id)
        self.bucket = self.storage_client.bucket(bucket_name)
    
    def upload_file(self, local_path: str, gcs_path: str) -> str:
        """Upload a file to GCS.
        
        Args:
            local_path: Local file path
            gcs_path: GCS path (without gs://{bucket_name}/)
            
        Returns:
            GCS URI for the uploaded file
        """
        try:
            blob = self.bucket.blob(gcs_path)
            blob.upload_from_filename(local_path)
            gcs_uri = f"gs://{self.bucket_name}/{gcs_path}"
            logger.info(f"Uploaded {local_path} to {gcs_uri}")
            return gcs_uri
        except Exception as e:
            logger.error(f"Error uploading file to GCS: {str(e)}")
            raise
    
    def download_file(self, gcs_path: str, local_path: str) -> str:
        """Download a file from GCS.
        
        Args:
            gcs_path: GCS path (without gs://{bucket_name}/)
            local_path: Local file path
            
        Returns:
            Local path for the downloaded file
        """
        try:
            # Create directory if it doesn't exist
            local_dir = os.path.dirname(local_path)
            if local_dir and not os.path.exists(local_dir):
                os.makedirs(local_dir)
            
            blob = self.bucket.blob(gcs_path)
            blob.download_to_filename(local_path)
            logger.info(f"Downloaded gs://{self.bucket_name}/{gcs_path} to {local_path}")
            return local_path
        except Exception as e:
            logger.error(f"Error downloading file from GCS: {str(e)}")
            raise
    
    def list_files(self, prefix: str = "", delimiter: str = None) -> List[str]:
        """List files in GCS bucket.
        
        Args:
            prefix: Prefix to filter files
            delimiter: Delimiter for hierarchical listing
            
        Returns:
            List of file paths in the bucket
        """
        try:
            blobs = self.storage_client.list_blobs(
                self.bucket_name, prefix=prefix, delimiter=delimiter
            )
            return [blob.name for blob in blobs]
        except Exception as e:
            logger.error(f"Error listing files in GCS: {str(e)}")
            raise
    
    def file_exists(self, gcs_path: str) -> bool:
        """Check if a file exists in GCS.
        
        Args:
            gcs_path: GCS path (without gs://{bucket_name}/)
            
        Returns:
            True if the file exists, False otherwise
        """
        try:
            blob = self.bucket.blob(gcs_path)
            return blob.exists()
        except Exception as e:
            logger.error(f"Error checking if file exists in GCS: {str(e)}")
            raise

class VertexAIModelManager:
    """Interface to Vertex AI Model Management."""
    
    def __init__(self, 
                 project_id: str = DEFAULT_PROJECT_ID,
                 location: str = DEFAULT_LOCATION):
        """Initialize the Model Manager.
        
        Args:
            project_id: GCP project ID
            location: GCP region
        """
        self.project_id = project_id
        self.location = location
        
        # Initialize Vertex AI
        aiplatform.init(project=project_id, location=location)
    
    def train_model(self, 
                   display_name: str,
                   training_data_uri: str,
                   model_type: str = "tabular-classifier",
                   target_column: str = "is_yielding",
                   budget_milli_node_hours: int = 8000) -> aiplatform.Model:
        """Train a model on Vertex AI.
        
        Args:
            display_name: Display name for the model
            training_data_uri: GCS URI for training data
            model_type: Type of model to train
            target_column: Target column for prediction
            budget_milli_node_hours: Training budget in milli-node-hours
            
        Returns:
            Trained model
        """
        try:
            # Create dataset
            dataset = aiplatform.TabularDataset.create(
                display_name=f"{display_name}-dataset",
                gcs_source=training_data_uri
            )
            
            # Define training job
            if model_type == "tabular-classifier":
                job = aiplatform.AutoMLTabularTrainingJob(
                    display_name=f"{display_name}-training",
                    optimization_objective="minimize-log-loss",
                )
                
                # Launch training job
                model = job.run(
                    dataset=dataset,
                    target_column=target_column,
                    budget_milli_node_hours=budget_milli_node_hours,
                    model_display_name=display_name,
                )
            
            elif model_type == "tabular-regressor":
                job = aiplatform.AutoMLTabularTrainingJob(
                    display_name=f"{display_name}-training",
                    optimization_objective="minimize-rmse",
                )
                
                # Launch training job
                model = job.run(
                    dataset=dataset,
                    target_column=target_column,
                    budget_milli_node_hours=budget_milli_node_hours,
                    model_display_name=display_name,
                )
            
            else:
                raise ValueError(f"Unsupported model type: {model_type}")
            
            logger.info(f"Model training completed: {model.resource_name}")
            return model
            
        except Exception as e:
            logger.error(f"Error training model: {str(e)}")
            raise
    
    def deploy_model(self, 
                    model: aiplatform.Model,
                    machine_type: str = "n1-standard-2",
                    min_replica_count: int = 1,
                    max_replica_count: int = 1) -> aiplatform.Endpoint:
        """Deploy a model to an endpoint.
        
        Args:
            model: Model to deploy
            machine_type: Machine type for the deployment
            min_replica_count: Minimum number of replicas
            max_replica_count: Maximum number of replicas
            
        Returns:
            Deployed endpoint
        """
        try:
            # Create endpoint
            endpoint = aiplatform.Endpoint.create(
                display_name=f"{model.display_name}-endpoint"
            )
            
            # Deploy model to endpoint
            model.deploy(
                endpoint=endpoint,
                machine_type=machine_type,
                min_replica_count=min_replica_count,
                max_replica_count=max_replica_count,
            )
            
            logger.info(f"Model deployed to endpoint: {endpoint.resource_name}")
            return endpoint
            
        except Exception as e:
            logger.error(f"Error deploying model: {str(e)}")
            raise
    
    def predict(self, 
               endpoint: aiplatform.Endpoint,
               instances: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Make predictions using a deployed model.
        
        Args:
            endpoint: Endpoint to query
            instances: Instances to predict
            
        Returns:
            Prediction results
        """
        try:
            # Make prediction
            predictions = endpoint.predict(instances=instances)
            return predictions
            
        except Exception as e:
            logger.error(f"Error making prediction: {str(e)}")
            raise

class MultimodalFeaturePipelineGCP:
    """Main class for the GCP implementation of the Multimodal Feature Pipeline."""
    
    def __init__(self, 
                 project_id: str = DEFAULT_PROJECT_ID,
                 location: str = DEFAULT_LOCATION,
                 bucket_name: str = DEFAULT_BUCKET_NAME,
                 feature_store_id: str = DEFAULT_FEATURE_STORE_ID):
        """Initialize the pipeline.
        
        Args:
            project_id: GCP project ID
            location: GCP region
            bucket_name: GCS bucket name
            feature_store_id: Feature Store ID
        """
        self.project_id = project_id
        self.location = location
        self.bucket_name = bucket_name
        self.feature_store_id = feature_store_id
        
        # Initialize components
        self.feature_store = VertexAIFeatureStore(
            project_id=project_id,
            location=location,
            feature_store_id=feature_store_id
        )
        
        self.gcs = GCSConnector(
            project_id=project_id,
            bucket_name=bucket_name
        )
        
        self.model_manager = VertexAIModelManager(
            project_id=project_id,
            location=location
        )
    
    def run_dataflow_pipeline(self, 
                             motion_path: str,
                             reasoning_path: str,
                             output_path: str,
                             max_scenarios: int = None,
                             job_name: str = None) -> str:
        """Run the Dataflow pipeline for distributed processing.
        
        Args:
            motion_path: GCS path to motion dataset
            reasoning_path: GCS path to reasoning dataset
            output_path: GCS path for output
            max_scenarios: Maximum number of scenarios to process
            job_name: Custom job name
            
        Returns:
            Job ID of the Dataflow job
        """
        try:
            # Generate job name if not provided
            if not job_name:
                job_name = f"waymo-pipeline-{int(time.time())}"
            
            # Build command
            cmd = [
                "python", "-m", "gcp_implementation.dataflow_pipeline",
                f"--project={self.project_id}",
                f"--region={self.location}",
                f"--motion-path={motion_path}",
                f"--reasoning-path={reasoning_path}",
                f"--output-path={output_path}",
                f"--temp-location=gs://{self.bucket_name}/temp",
                "--runner=DataflowRunner",
                f"--job_name={job_name}",
            ]
            
            if max_scenarios:
                cmd.append(f"--max-scenarios={max_scenarios}")
            
            # Run Dataflow job
            import subprocess
            process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                universal_newlines=True
            )
            
            # Get the job ID from the output
            job_id = None
            for line in process.stdout:
                if "job_id" in line.lower():
                    parts = line.strip().split(":")
                    if len(parts) > 1:
                        job_id = parts[1].strip()
                        break
            
            logger.info(f"Started Dataflow job: {job_id}")
            return job_id
            
        except Exception as e:
            logger.error(f"Error running Dataflow pipeline: {str(e)}")
            raise
    
    def load_feature_store(self, features_path: str) -> int:
        """Load features into the Feature Store.
        
        Args:
            features_path: GCS path to feature files
            
        Returns:
            Number of entities loaded
        """
        try:
            # List feature files
            feature_files = self.gcs.list_files(features_path)
            
            # Load and process each file
            total_entities = 0
            
            for file_path in feature_files:
                # Skip non-JSON files
                if not file_path.endswith(".json"):
                    continue
                
                # Download file
                local_path = f"/tmp/{os.path.basename(file_path)}"
                self.gcs.download_file(file_path, local_path)
                
                # Load features
                with open(local_path, "r") as f:
                    features = [json.loads(line) for line in f if line.strip()]
                
                # Convert to DataFrame
                features_df = pd.DataFrame(features)
                
                # Write to Feature Store
                if not features_df.empty:
                    entity_count = self.feature_store.write_features(
                        features_df, entity_id_column="entity_id"
                    )
                    total_entities += entity_count
                
                # Clean up
                os.remove(local_path)
            
            logger.info(f"Loaded {total_entities} entities into Feature Store")
            return total_entities
            
        except Exception as e:
            logger.error(f"Error loading Feature Store: {str(e)}")
            raise
    
    def train_yield_model(self, 
                         agent_features_path: str,
                         model_name: str = "yield-prediction-model") -> Tuple[aiplatform.Model, aiplatform.Endpoint]:
        """Train and deploy a yield prediction model.
        
        Args:
            agent_features_path: GCS path to agent feature files
            model_name: Name for the model
            
        Returns:
            Tuple of (model, endpoint)
        """
        try:
            # Download and prepare training data
            local_csv_path = "/tmp/agent_features_training.csv"
            
            # Convert JSON files to CSV
            agent_features = []
            feature_files = self.gcs.list_files(agent_features_path)
            
            for file_path in feature_files:
                # Skip non-JSON files
                if not file_path.endswith(".json"):
                    continue
                
                # Download file
                local_path = f"/tmp/{os.path.basename(file_path)}"
                self.gcs.download_file(file_path, local_path)
                
                # Load features
                with open(local_path, "r") as f:
                    features = [json.loads(line) for line in f if line.strip()]
                
                agent_features.extend(features)
                
                # Clean up
                os.remove(local_path)
            
            # Convert to DataFrame
            features_df = pd.DataFrame(agent_features)
            
            # Filter out rows without yielding information
            features_df = features_df[~features_df["is_yielding"].isna()]
            
            # Remove text embeddings for model training
            if "text_embedding" in features_df.columns:
                features_df = features_df.drop(columns=["text_embedding"])
            
            # Save to CSV
            features_df.to_csv(local_csv_path, index=False)
            
            # Upload to GCS
            training_data_uri = self.gcs.upload_file(
                local_csv_path, "model_training/agent_features_training.csv"
            )
            
            # Train model
            model = self.model_manager.train_model(
                display_name=model_name,
                training_data_uri=training_data_uri,
                model_type="tabular-classifier",
                target_column="is_yielding",
            )
            
            # Deploy model
            endpoint = self.model_manager.deploy_model(model)
            
            # Clean up
            os.remove(local_csv_path)
            
            logger.info(f"Trained and deployed model: {model.display_name}")
            return model, endpoint
            
        except Exception as e:
            logger.error(f"Error training yield model: {str(e)}")
            raise
    
    def generate_online_features(self, 
                               scenario_id: str, 
                               agent_id: str) -> Dict[str, Any]:
        """Generate online features for a specific agent.
        
        Args:
            scenario_id: Scenario ID
            agent_id: Agent ID
            
        Returns:
            Feature dictionary
        """
        try:
            # Get entity ID
            entity_id = f"{scenario_id}_{agent_id}"
            
            # Read features from Feature Store
            features = self.feature_store.read_features([entity_id])
            
            if features.empty:
                raise ValueError(f"No features found for entity {entity_id}")
            
            # Convert to dictionary
            features_dict = features.iloc[0].to_dict()
            
            return features_dict
            
        except Exception as e:
            logger.error(f"Error generating online features: {str(e)}")
            raise
    
    def predict_yield(self, 
                     endpoint: aiplatform.Endpoint,
                     scenario_id: str, 
                     agent_id: str) -> Dict[str, Any]:
        """Predict if an agent will yield.
        
        Args:
            endpoint: Model endpoint
            scenario_id: Scenario ID
            agent_id: Agent ID
            
        Returns:
            Prediction result
        """
        try:
            # Get features
            features = self.generate_online_features(scenario_id, agent_id)
            
            # Remove text embeddings
            if "text_embedding" in features:
                del features["text_embedding"]
            
            # Make prediction
            prediction = self.model_manager.predict(endpoint, [features])
            
            return {
                "scenario_id": scenario_id,
                "agent_id": agent_id,
                "features": features,
                "prediction": prediction
            }
            
        except Exception as e:
            logger.error(f"Error predicting yield: {str(e)}")
            raise

def main():
    """Main function."""
    parser = argparse.ArgumentParser(description="Waymo Multimodal Feature Pipeline GCP Implementation")
    parser.add_argument("--project", default=DEFAULT_PROJECT_ID, help="GCP project ID")
    parser.add_argument("--location", default=DEFAULT_LOCATION, help="GCP region")
    parser.add_argument("--bucket", default=DEFAULT_BUCKET_NAME, help="GCS bucket name")
    parser.add_argument("--feature-store", default=DEFAULT_FEATURE_STORE_ID, help="Feature Store ID")
    parser.add_argument("--motion-path", required=True, help="GCS path to motion dataset")
    parser.add_argument("--reasoning-path", required=True, help="GCS path to reasoning dataset")
    parser.add_argument("--output-path", required=True, help="GCS path for output")
    parser.add_argument("--max-scenarios", type=int, help="Maximum number of scenarios to process")
    
    args = parser.parse_args()
    
    # Initialize pipeline
    pipeline = MultimodalFeaturePipelineGCP(
        project_id=args.project,
        location=args.location,
        bucket_name=args.bucket,
        feature_store_id=args.feature_store
    )
    
    # Run pipeline
    job_id = pipeline.run_dataflow_pipeline(
        motion_path=args.motion_path,
        reasoning_path=args.reasoning_path,
        output_path=args.output_path,
        max_scenarios=args.max_scenarios
    )
    
    # Wait for pipeline to complete
    logger.info(f"Dataflow job {job_id} started. Check the GCP console for job status.")
    
    # Note: In a production environment, you would typically set up a workflow
    # to wait for the Dataflow job to complete before proceeding with the next steps.
    # For simplicity, we'll just log instructions.
    logger.info(
        "After the Dataflow job completes, run the following commands to load features "
        "into the Feature Store and train a yield prediction model:"
    )
    logger.info(
        f"python -m gcp_implementation.multimodal_feature_pipeline_gcp load-feature-store "
        f"--features-path={args.output_path}/feature_store_entries"
    )
    logger.info(
        f"python -m gcp_implementation.multimodal_feature_pipeline_gcp train-model "
        f"--agent-features-path={args.output_path}/agent_features"
    )

if __name__ == "__main__":
    main() 