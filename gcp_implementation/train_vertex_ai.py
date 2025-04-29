"""
Model training module for Waymo Multimodal Feature Pipeline

This module handles the training of models on Vertex AI using the
features extracted from Waymo motion and reasoning datasets.
"""

import os
import logging
import argparse
import json
import pickle
from datetime import datetime
from typing import Dict, List, Optional, Union, Any

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import xgboost as xgb
import tensorflow as tf
from tensorflow import keras

from google.cloud import aiplatform
from google.cloud import storage

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class ModelTrainer:
    """Class for training ML models on Vertex AI."""
    
    def __init__(
        self,
        project_id: str,
        location: str,
        bucket_name: str,
        features_path: str,
        output_dir: str = "models",
        model_type: str = "random_forest"
    ):
        """Initialize the ModelTrainer.
        
        Args:
            project_id: GCP project ID
            location: GCP region
            bucket_name: GCS bucket for storing models and data
            features_path: Path to the features data
            output_dir: Directory to store model artifacts
            model_type: Type of model to train (random_forest, xgboost, neural_network)
        """
        self.project_id = project_id
        self.location = location
        self.bucket_name = bucket_name
        self.features_path = features_path
        self.output_dir = output_dir
        self.model_type = model_type
        
        # Initialize clients
        self.storage_client = storage.Client(project=project_id)
        
        # Initialize Vertex AI
        aiplatform.init(project=project_id, location=location)
    
    def load_features(self) -> pd.DataFrame:
        """Load features from GCS or local path.
        
        Returns:
            DataFrame containing the features
        """
        logger.info(f"Loading features from {self.features_path}")
        
        if self.features_path.startswith("gs://"):
            # Parse GCS path
            bucket_name = self.features_path.split("/")[2]
            prefix = "/".join(self.features_path.split("/")[3:])
            
            # Get bucket
            bucket = self.storage_client.get_bucket(bucket_name)
            
            # List blobs
            blobs = list(bucket.list_blobs(prefix=prefix))
            
            if not blobs:
                raise ValueError(f"No files found at {self.features_path}")
            
            # Read CSV files
            dfs = []
            for blob in blobs:
                if blob.name.endswith(".csv"):
                    content = blob.download_as_string()
                    df = pd.read_csv(pd.io.common.BytesIO(content))
                    dfs.append(df)
            
            # Combine DataFrames
            if dfs:
                features_df = pd.concat(dfs, ignore_index=True)
            else:
                raise ValueError(f"No CSV files found at {self.features_path}")
        else:
            # Local path
            if os.path.isdir(self.features_path):
                # Directory, read all CSV files
                csv_files = [os.path.join(self.features_path, f) for f in os.listdir(self.features_path) 
                           if f.endswith('.csv')]
                
                if not csv_files:
                    raise ValueError(f"No CSV files found in {self.features_path}")
                
                dfs = [pd.read_csv(f) for f in csv_files]
                features_df = pd.concat(dfs, ignore_index=True)
            else:
                # Single file
                features_df = pd.read_csv(self.features_path)
        
        logger.info(f"Loaded {len(features_df)} records")
        return features_df
    
    def prepare_data(
        self,
        features_df: pd.DataFrame,
        target_col: str = "is_yielding",
        test_size: float = 0.2,
        random_state: int = 42
    ) -> Dict[str, Union[np.ndarray, pd.DataFrame]]:
        """Prepare data for model training.
        
        Args:
            features_df: DataFrame containing features
            target_col: Target column name
            test_size: Proportion of data to use for testing
            random_state: Random seed for reproducibility
            
        Returns:
            Dictionary containing train and test datasets
        """
        logger.info("Preparing data for model training")
        
        # Check if target column exists
        if target_col not in features_df.columns:
            raise ValueError(f"Target column '{target_col}' not found in features DataFrame")
        
        # Drop rows with missing target values
        features_df = features_df.dropna(subset=[target_col])
        
        # Extract target variable
        y = features_df[target_col].astype(int)
        
        # Remove non-feature columns
        exclude_cols = ['scenario_id', 'agent_id', 'ego_agent_id']
        feature_cols = [col for col in features_df.columns if col != target_col and col not in exclude_cols]
        X = features_df[feature_cols]
        
        # Split data into train and test sets
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=y
        )
        
        # Scale numerical features
        numerical_cols = X.select_dtypes(include=['int64', 'float64']).columns
        scaler = StandardScaler()
        X_train[numerical_cols] = scaler.fit_transform(X_train[numerical_cols])
        X_test[numerical_cols] = scaler.transform(X_test[numerical_cols])
        
        # Save feature columns and scaler for inference
        self.feature_cols = feature_cols
        self.scaler = scaler
        
        logger.info(f"Data prepared: {len(X_train)} training samples, {len(X_test)} test samples")
        
        return {
            'X_train': X_train,
            'X_test': X_test,
            'y_train': y_train,
            'y_test': y_test,
            'feature_cols': feature_cols,
            'scaler': scaler
        }
    
    def train_random_forest(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        n_estimators: int = 100,
        max_depth: int = 10,
        random_state: int = 42
    ) -> RandomForestClassifier:
        """Train a Random Forest model.
        
        Args:
            X_train: Training features
            y_train: Training labels
            n_estimators: Number of trees in the forest
            max_depth: Maximum depth of the trees
            random_state: Random seed for reproducibility
            
        Returns:
            Trained Random Forest model
        """
        logger.info("Training Random Forest model")
        
        model = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            random_state=random_state,
            n_jobs=-1
        )
        
        model.fit(X_train, y_train)
        
        logger.info("Random Forest model training complete")
        return model
    
    def train_xgboost(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        n_estimators: int = 100,
        max_depth: int = 6,
        learning_rate: float = 0.1,
        random_state: int = 42
    ) -> xgb.XGBClassifier:
        """Train an XGBoost model.
        
        Args:
            X_train: Training features
            y_train: Training labels
            n_estimators: Number of boosting rounds
            max_depth: Maximum depth of the trees
            learning_rate: Learning rate
            random_state: Random seed for reproducibility
            
        Returns:
            Trained XGBoost model
        """
        logger.info("Training XGBoost model")
        
        model = xgb.XGBClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            learning_rate=learning_rate,
            random_state=random_state,
            use_label_encoder=False,
            eval_metric='logloss'
        )
        
        model.fit(X_train, y_train)
        
        logger.info("XGBoost model training complete")
        return model
    
    def train_neural_network(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        hidden_layers: List[int] = [128, 64, 32],
        dropout_rate: float = 0.2,
        epochs: int = 100,
        batch_size: int = 32,
        random_state: int = 42
    ) -> keras.Model:
        """Train a Neural Network model.
        
        Args:
            X_train: Training features
            y_train: Training labels
            hidden_layers: List of hidden layer sizes
            dropout_rate: Dropout rate for regularization
            epochs: Number of training epochs
            batch_size: Batch size for training
            random_state: Random seed for reproducibility
            
        Returns:
            Trained Neural Network model
        """
        logger.info("Training Neural Network model")
        
        # Set random seeds for reproducibility
        tf.random.set_seed(random_state)
        
        # Input dimension
        input_dim = X_train.shape[1]
        
        # Create model
        model = keras.Sequential()
        
        # Input layer
        model.add(keras.layers.Dense(hidden_layers[0], input_dim=input_dim, activation='relu'))
        model.add(keras.layers.BatchNormalization())
        model.add(keras.layers.Dropout(dropout_rate))
        
        # Hidden layers
        for units in hidden_layers[1:]:
            model.add(keras.layers.Dense(units, activation='relu'))
            model.add(keras.layers.BatchNormalization())
            model.add(keras.layers.Dropout(dropout_rate))
        
        # Output layer
        model.add(keras.layers.Dense(1, activation='sigmoid'))
        
        # Compile model
        model.compile(
            optimizer='adam',
            loss='binary_crossentropy',
            metrics=['accuracy']
        )
        
        # Create validation split
        X_train_nn, X_val, y_train_nn, y_val = train_test_split(
            X_train, y_train, test_size=0.1, random_state=random_state
        )
        
        # Early stopping
        early_stopping = keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=10,
            restore_best_weights=True
        )
        
        # Train model
        model.fit(
            X_train_nn, y_train_nn,
            epochs=epochs,
            batch_size=batch_size,
            validation_data=(X_val, y_val),
            callbacks=[early_stopping],
            verbose=1
        )
        
        logger.info("Neural Network model training complete")
        return model
    
    def evaluate_model(
        self,
        model: Any,
        X_test: pd.DataFrame,
        y_test: pd.Series
    ) -> Dict[str, float]:
        """Evaluate the model.
        
        Args:
            model: Trained model
            X_test: Test features
            y_test: Test labels
            
        Returns:
            Dictionary of evaluation metrics
        """
        logger.info("Evaluating model")
        
        if isinstance(model, keras.Model):
            # Neural Network
            y_pred_proba = model.predict(X_test, verbose=0).flatten()
            y_pred = (y_pred_proba > 0.5).astype(int)
        else:
            # Random Forest or XGBoost
            y_pred = model.predict(X_test)
            y_pred_proba = model.predict_proba(X_test)[:, 1]
        
        # Calculate metrics
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, zero_division=0)
        recall = recall_score(y_test, y_pred, zero_division=0)
        f1 = f1_score(y_test, y_pred, zero_division=0)
        auc = roc_auc_score(y_test, y_pred_proba)
        
        metrics = {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'auc': auc
        }
        
        logger.info(f"Model evaluation results: {metrics}")
        return metrics
    
    def save_model_locally(self, model: Any, prefix: str = "model") -> str:
        """Save model to local filesystem.
        
        Args:
            model: Trained model
            prefix: Prefix for output files
            
        Returns:
            Path to saved model
        """
        # Create output directory if it doesn't exist
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Generate timestamp for model versioning
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        model_name = f"{prefix}_{self.model_type}_{timestamp}"
        model_dir = os.path.join(self.output_dir, model_name)
        os.makedirs(model_dir, exist_ok=True)
        
        # Save model
        if isinstance(model, keras.Model):
            # Neural Network - save in TensorFlow SavedModel format
            model_path = os.path.join(model_dir, "model")
            model.save(model_path, save_format='tf')
        else:
            # Random Forest or XGBoost - save using pickle
            model_path = os.path.join(model_dir, "model.pkl")
            with open(model_path, 'wb') as f:
                pickle.dump(model, f)
        
        # Save feature columns and scaler
        with open(os.path.join(model_dir, "feature_cols.json"), 'w') as f:
            json.dump(self.feature_cols, f)
        
        with open(os.path.join(model_dir, "scaler.pkl"), 'wb') as f:
            pickle.dump(self.scaler, f)
        
        logger.info(f"Model saved locally at {model_dir}")
        return model_dir
    
    def upload_model_to_gcs(self, model_dir: str) -> str:
        """Upload model to GCS.
        
        Args:
            model_dir: Local directory containing model files
            
        Returns:
            GCS URI of the uploaded model
        """
        # Get bucket
        bucket = self.storage_client.get_bucket(self.bucket_name)
        
        # Upload model files
        model_name = os.path.basename(model_dir)
        gcs_prefix = f"models/{model_name}"
        
        for root, _, files in os.walk(model_dir):
            for file in files:
                local_path = os.path.join(root, file)
                relative_path = os.path.relpath(local_path, self.output_dir)
                gcs_path = f"{gcs_prefix}/{os.path.basename(file)}"
                
                blob = bucket.blob(gcs_path)
                blob.upload_from_filename(local_path)
        
        gcs_uri = f"gs://{self.bucket_name}/{gcs_prefix}"
        logger.info(f"Model uploaded to GCS at {gcs_uri}")
        return gcs_uri
    
    def upload_model_to_vertex_ai(
        self,
        gcs_uri: str,
        display_name: str = "waymo_yield_prediction",
        description: str = "Waymo yield prediction model"
    ) -> aiplatform.Model:
        """Upload model to Vertex AI Model Registry.
        
        Args:
            gcs_uri: GCS URI of the model
            display_name: Display name for the model
            description: Description of the model
            
        Returns:
            Vertex AI Model
        """
        # Check if model is TensorFlow or custom
        is_tf_model = os.path.exists(os.path.join(gcs_uri.replace("gs://", ""), "saved_model.pb"))
        
        if is_tf_model:
            # Upload TensorFlow model
            model = aiplatform.Model.upload(
                display_name=display_name,
                artifact_uri=gcs_uri,
                serving_container_image_uri="us-docker.pkg.dev/vertex-ai/prediction/tf2-cpu.2-8:latest",
                description=description
            )
        else:
            # Upload custom model
            model = aiplatform.Model.upload(
                display_name=display_name,
                artifact_uri=gcs_uri,
                serving_container_image_uri="us-docker.pkg.dev/vertex-ai/prediction/sklearn-cpu.1-0:latest",
                description=description
            )
        
        logger.info(f"Model uploaded to Vertex AI Model Registry with ID: {model.resource_name}")
        return model
    
    def run_training_pipeline(self) -> Dict[str, Any]:
        """Run the complete training pipeline.
        
        Returns:
            Dictionary with training results and model information
        """
        # Load features
        features_df = self.load_features()
        
        # Prepare data
        data = self.prepare_data(features_df)
        
        # Train model based on model type
        if self.model_type == "random_forest":
            model = self.train_random_forest(data['X_train'], data['y_train'])
        elif self.model_type == "xgboost":
            model = self.train_xgboost(data['X_train'], data['y_train'])
        elif self.model_type == "neural_network":
            model = self.train_neural_network(data['X_train'], data['y_train'])
        else:
            raise ValueError(f"Unsupported model type: {self.model_type}")
        
        # Evaluate model
        metrics = self.evaluate_model(model, data['X_test'], data['y_test'])
        
        # Save model locally
        model_dir = self.save_model_locally(model)
        
        # Upload model to GCS
        gcs_uri = self.upload_model_to_gcs(model_dir)
        
        # Upload model to Vertex AI (optional)
        vertex_ai_model = self.upload_model_to_vertex_ai(gcs_uri)
        
        # Return results
        results = {
            'metrics': metrics,
            'model_type': self.model_type,
            'model_dir': model_dir,
            'gcs_uri': gcs_uri,
            'vertex_ai_model_id': vertex_ai_model.resource_name
        }
        
        logger.info(f"Training pipeline completed successfully: {results}")
        return results


def run_custom_training_job(
    project_id: str,
    location: str,
    bucket_name: str,
    features_path: str,
    model_type: str,
    display_name: str = "waymo_yield_training_job"
) -> aiplatform.CustomJob:
    """Run a custom training job on Vertex AI.
    
    Args:
        project_id: GCP project ID
        location: GCP region
        bucket_name: GCS bucket for storing models and data
        features_path: Path to the features data
        model_type: Type of model to train
        display_name: Display name for the training job
        
    Returns:
        Vertex AI CustomJob
    """
    # Initialize Vertex AI
    aiplatform.init(project=project_id, location=location)
    
    # Define worker pool spec
    worker_pool_spec = {
        "machine_spec": {
            "machine_type": "n1-standard-4",
            "accelerator_type": "NVIDIA_TESLA_T4",
            "accelerator_count": 1,
        },
        "replica_count": 1,
        "container_spec": {
            "image_uri": "us-docker.pkg.dev/vertex-ai/training/sklearn-cpu.0-23:latest",
            "command": [],
            "args": [
                f"--project={project_id}",
                f"--bucket={bucket_name}",
                f"--features_path={features_path}",
                f"--model_type={model_type}",
            ],
        },
    }
    
    # Create custom job
    job = aiplatform.CustomJob(
        display_name=display_name,
        worker_pool_specs=[worker_pool_spec],
    )
    
    # Run custom job
    job.run()
    
    logger.info(f"Custom training job completed: {job.resource_name}")
    return job


def main():
    """Main function."""
    parser = argparse.ArgumentParser(description="Train models on Vertex AI")
    parser.add_argument("--project", required=True, help="GCP project ID")
    parser.add_argument("--location", default="us-central1", help="GCP region")
    parser.add_argument("--bucket", required=True, help="GCS bucket for storing models and data")
    parser.add_argument("--features_path", required=True, help="Path to features data")
    parser.add_argument("--output_dir", default="models", help="Directory to store model artifacts")
    parser.add_argument("--model_type", default="random_forest", 
                        choices=["random_forest", "xgboost", "neural_network"],
                        help="Type of model to train")
    parser.add_argument("--run_custom_job", action="store_true", 
                        help="Run as a custom training job on Vertex AI")
    
    args = parser.parse_args()
    
    if args.run_custom_job:
        # Run as a custom training job
        run_custom_training_job(
            project_id=args.project,
            location=args.location,
            bucket_name=args.bucket,
            features_path=args.features_path,
            model_type=args.model_type
        )
    else:
        # Run locally
        trainer = ModelTrainer(
            project_id=args.project,
            location=args.location,
            bucket_name=args.bucket,
            features_path=args.features_path,
            output_dir=args.output_dir,
            model_type=args.model_type
        )
        
        trainer.run_training_pipeline()


if __name__ == "__main__":
    main() 