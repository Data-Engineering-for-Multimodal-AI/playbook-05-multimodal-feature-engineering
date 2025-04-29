"""
Vertex AI Feature Store Setup for Waymo Multimodal Features

This script sets up and configures a Vertex AI Feature Store for the Waymo
multimodal feature pipeline, including creating feature store, entity types,
and feature definitions.
"""

import argparse
import json
import logging
import os
import time
from typing import Dict, List, Optional, Union

from google.cloud import aiplatform
from google.cloud.aiplatform import featurestore
from google.cloud.aiplatform.featurestore import EntityType, Feature, Featurestore

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class FeatureStoreManager:
    """Manages the Vertex AI Feature Store operations."""

    def __init__(
        self,
        project_id: str,
        location: str,
        feature_store_id: str,
        schema_path: Optional[str] = None
    ):
        """Initialize the feature store manager.
        
        Args:
            project_id: GCP project ID
            location: GCP region (e.g., us-central1)
            feature_store_id: ID for the feature store
            schema_path: Path to the schema file for feature definitions
        """
        self.project_id = project_id
        self.location = location
        self.feature_store_id = feature_store_id
        self.schema_path = schema_path
        self.feature_schema = None
        
        # Initialize Vertex AI client
        aiplatform.init(project=project_id, location=location)
        
        # Load schema if provided
        if schema_path:
            self._load_schema()
    
    def _load_schema(self):
        """Load the feature schema from the provided path."""
        try:
            with open(self.schema_path, 'r') as f:
                self.feature_schema = json.load(f)
                logger.info(f"Loaded schema with {len(self.feature_schema)} feature definitions")
        except Exception as e:
            logger.error(f"Error loading schema: {e}")
            raise
    
    def create_feature_store(self) -> Featurestore:
        """Create a new feature store or get an existing one.
        
        Returns:
            The feature store object
        """
        try:
            # Check if feature store exists
            try:
                fs = featurestore.Featurestore.get(
                    featurestore_name=self.feature_store_id,
                    project=self.project_id,
                    location=self.location
                )
                logger.info(f"Feature store {self.feature_store_id} already exists")
                return fs
            except Exception:
                # Create new feature store
                logger.info(f"Creating feature store {self.feature_store_id}")
                
                create_fs_op = featurestore.Featurestore.create(
                    featurestore_id=self.feature_store_id,
                    online_store_fixed_node_count=1,
                    project=self.project_id,
                    location=self.location
                )
                
                # Wait for creation to complete
                create_fs_op.result()
                
                # Get the created feature store
                fs = featurestore.Featurestore.get(
                    featurestore_name=self.feature_store_id,
                    project=self.project_id,
                    location=self.location
                )
                
                logger.info(f"Feature store {self.feature_store_id} created successfully")
                return fs
        except Exception as e:
            logger.error(f"Error creating feature store: {e}")
            raise
    
    def create_entity_type(
        self, 
        fs: Featurestore, 
        entity_type_id: str, 
        description: str
    ) -> EntityType:
        """Create a new entity type or get an existing one.
        
        Args:
            fs: Feature store object
            entity_type_id: ID for the entity type
            description: Description of the entity type
            
        Returns:
            The entity type object
        """
        try:
            # Check if entity type exists
            try:
                entity_type = fs.get_entity_type(entity_type_id=entity_type_id)
                logger.info(f"Entity type {entity_type_id} already exists")
                return entity_type
            except Exception:
                # Create new entity type
                logger.info(f"Creating entity type {entity_type_id}")
                
                create_et_op = fs.create_entity_type(
                    entity_type_id=entity_type_id,
                    description=description
                )
                
                # Wait for creation to complete
                create_et_op.result()
                
                # Get the created entity type
                entity_type = fs.get_entity_type(entity_type_id=entity_type_id)
                
                logger.info(f"Entity type {entity_type_id} created successfully")
                return entity_type
        except Exception as e:
            logger.error(f"Error creating entity type: {e}")
            raise
    
    def create_features(
        self, 
        entity_type: EntityType, 
        features_config: Dict[str, Dict[str, str]]
    ) -> List[Feature]:
        """Create features based on the provided configuration.
        
        Args:
            entity_type: Entity type object
            features_config: Dictionary mapping feature names to their configurations
            
        Returns:
            List of created features
        """
        created_features = []
        
        for feature_id, feature_config in features_config.items():
            try:
                # Check feature datatype
                value_type = feature_config.get('type', 'STRING')
                
                # Handle array types
                if '[' in value_type:
                    base_type, array_size = value_type.split('[')
                    array_size = array_size.rstrip(']')
                    
                    if base_type.lower() == 'float':
                        value_type = 'DOUBLE_ARRAY'
                    elif base_type.lower() == 'int':
                        value_type = 'INT64_ARRAY'
                    elif base_type.lower() == 'string':
                        value_type = 'STRING_ARRAY'
                    else:
                        value_type = 'DOUBLE_ARRAY'  # Default for unknown types
                
                # Map schema types to Vertex AI types
                if value_type.lower() == 'float':
                    value_type = 'DOUBLE'
                elif value_type.lower() == 'int':
                    value_type = 'INT64'
                elif value_type.lower() == 'string':
                    value_type = 'STRING'
                elif value_type.lower() == 'bool' or value_type.lower() == 'boolean':
                    value_type = 'BOOL'
                
                # Check if feature exists
                try:
                    feature = entity_type.get_feature(feature_id=feature_id)
                    logger.info(f"Feature {feature_id} already exists")
                except Exception:
                    # Create new feature
                    logger.info(f"Creating feature {feature_id} with type {value_type}")
                    
                    create_feature_op = entity_type.create_feature(
                        feature_id=feature_id,
                        value_type=value_type,
                        description=f"Feature {feature_id}"
                    )
                    
                    # Wait for creation to complete
                    create_feature_op.result()
                    
                    # Get the created feature
                    feature = entity_type.get_feature(feature_id=feature_id)
                    logger.info(f"Feature {feature_id} created successfully")
                
                created_features.append(feature)
            
            except Exception as e:
                logger.error(f"Error creating feature {feature_id}: {e}")
                # Continue with other features
        
        return created_features
    
    def create_feature_store_resources(self):
        """Set up the complete feature store with entity types and features."""
        # Create feature store
        fs = self.create_feature_store()
        
        # Create entity types and features
        if self.feature_schema:
            # Create agent entity type
            agent_entity_type = self.create_entity_type(
                fs=fs,
                entity_type_id="agent",
                description="Agent within a scenario, representing vehicles and other road users"
            )
            
            # Create scenario entity type
            scenario_entity_type = self.create_entity_type(
                fs=fs,
                entity_type_id="scenario",
                description="Traffic scenario from Waymo dataset"
            )
            
            # Create features for agent entity type
            agent_features = {
                k: {"type": v} for k, v in self.feature_schema.items() 
                if k not in ['scenario_id']
            }
            
            # Ensure entity_id is included for agent entity
            agent_features['entity_id'] = {"type": "STRING"}
            
            self.create_features(agent_entity_type, agent_features)
            
            # Create features for scenario entity type
            scenario_features = {
                "scenario_id": {"type": "STRING"},
                "ego_agent_id": {"type": "INT64"},
                "agent_count": {"type": "INT64"},
                "env_questions": {"type": "INT64"},
                "ego_questions": {"type": "INT64"},
                "sur_questions": {"type": "INT64"},
                "int_questions": {"type": "INT64"},
                "text_embedding": {"type": self.feature_schema.get("text_embedding", "DOUBLE_ARRAY")}
            }
            
            self.create_features(scenario_entity_type, scenario_features)
            
            logger.info("Feature store setup completed successfully")
        else:
            logger.warning("No feature schema provided. Only feature store created.")
    
    def configure_batch_serving(
        self,
        entity_type_id: str,
        destination_table: str,
        serving_feature_ids: List[str]
    ):
        """Configure batch serving for the feature store.
        
        Args:
            entity_type_id: ID of the entity type
            destination_table: BigQuery destination table for batch serving
            serving_feature_ids: List of feature IDs to include in serving
        """
        try:
            # Get feature store
            fs = featurestore.Featurestore.get(
                featurestore_name=self.feature_store_id,
                project=self.project_id,
                location=self.location
            )
            
            # Get entity type
            entity_type = fs.get_entity_type(entity_type_id=entity_type_id)
            
            # Define batch serving destination
            batch_serving_op = entity_type.batch_serve_to_bq(
                bq_destination_output_uri=destination_table,
                serving_feature_ids=serving_feature_ids
            )
            
            # Wait for batch serving to complete
            batch_serving_op.result()
            
            logger.info(f"Batch serving configured successfully for {entity_type_id}")
        except Exception as e:
            logger.error(f"Error configuring batch serving: {e}")
            raise
    
    def ingest_features_from_bq(
        self,
        entity_type_id: str,
        source_table: str,
        entity_id_field: str
    ):
        """Ingest features from BigQuery table.
        
        Args:
            entity_type_id: ID of the entity type
            source_table: BigQuery source table containing features
            entity_id_field: Field name in the table that contains entity IDs
        """
        try:
            # Get feature store
            fs = featurestore.Featurestore.get(
                featurestore_name=self.feature_store_id,
                project=self.project_id,
                location=self.location
            )
            
            # Get entity type
            entity_type = fs.get_entity_type(entity_type_id=entity_type_id)
            
            # Ingest features
            ingest_op = entity_type.ingest_from_bq(
                bq_source_uri=source_table,
                entity_id_field=entity_id_field
            )
            
            # Wait for ingestion to complete
            ingest_op.result()
            
            logger.info(f"Feature ingestion completed successfully for {entity_type_id}")
        except Exception as e:
            logger.error(f"Error ingesting features: {e}")
            raise
    
    def get_online_features(
        self,
        entity_type_id: str,
        entity_ids: List[str],
        feature_ids: List[str]
    ) -> Dict[str, List[Union[float, int, str, List]]]:
        """Retrieve online features for the given entity IDs.
        
        Args:
            entity_type_id: ID of the entity type
            entity_ids: List of entity IDs to query
            feature_ids: List of feature IDs to retrieve
            
        Returns:
            Dictionary mapping entity IDs to feature values
        """
        try:
            # Get feature store
            fs = featurestore.Featurestore.get(
                featurestore_name=self.feature_store_id,
                project=self.project_id,
                location=self.location
            )
            
            # Get entity type
            entity_type = fs.get_entity_type(entity_type_id=entity_type_id)
            
            # Read online features
            response = entity_type.read(
                entity_ids=entity_ids,
                feature_ids=feature_ids
            )
            
            # Process response
            results = {}
            for entity_id, feature_values in zip(entity_ids, response):
                results[entity_id] = {}
                for feature_id, value in zip(feature_ids, feature_values):
                    results[entity_id][feature_id] = value
            
            return results
        except Exception as e:
            logger.error(f"Error retrieving online features: {e}")
            raise


def main():
    """Main entry point for feature store setup."""
    parser = argparse.ArgumentParser(
        description="Set up Vertex AI Feature Store for Waymo multimodal features"
    )
    
    parser.add_argument(
        "--project_id",
        required=True,
        help="GCP project ID"
    )
    parser.add_argument(
        "--location",
        default="us-central1",
        help="GCP region for feature store"
    )
    parser.add_argument(
        "--feature_store_id",
        default="waymo_multimodal_features",
        help="ID for the feature store"
    )
    parser.add_argument(
        "--schema_path",
        default=None,
        help="Path to feature schema JSON file"
    )
    parser.add_argument(
        "--configure_batch_serving",
        action="store_true",
        help="Configure batch serving to BigQuery"
    )
    parser.add_argument(
        "--batch_destination",
        default=None,
        help="BigQuery destination table for batch serving (project.dataset.table)"
    )
    parser.add_argument(
        "--ingest_from_bq",
        action="store_true",
        help="Ingest features from BigQuery"
    )
    parser.add_argument(
        "--source_table",
        default=None,
        help="BigQuery source table for ingestion (project.dataset.table)"
    )
    parser.add_argument(
        "--entity_type",
        default="agent",
        help="Entity type to work with (agent or scenario)"
    )
    
    args = parser.parse_args()
    
    # Initialize feature store manager
    manager = FeatureStoreManager(
        project_id=args.project_id,
        location=args.location,
        feature_store_id=args.feature_store_id,
        schema_path=args.schema_path
    )
    
    # Create feature store and resources
    manager.create_feature_store_resources()
    
    # Configure batch serving if requested
    if args.configure_batch_serving and args.batch_destination:
        # Get feature IDs based on entity type
        if args.entity_type == "agent":
            feature_ids = list(manager.feature_schema.keys())
            feature_ids = [f for f in feature_ids if f != "scenario_id"]
        else:
            feature_ids = ["scenario_id", "ego_agent_id", "agent_count", 
                          "env_questions", "ego_questions", "sur_questions", 
                          "int_questions", "text_embedding"]
        
        manager.configure_batch_serving(
            entity_type_id=args.entity_type,
            destination_table=args.batch_destination,
            serving_feature_ids=feature_ids
        )
    
    # Ingest features if requested
    if args.ingest_from_bq and args.source_table:
        entity_id_field = "entity_id" if args.entity_type == "agent" else "scenario_id"
        
        manager.ingest_features_from_bq(
            entity_type_id=args.entity_type,
            source_table=args.source_table,
            entity_id_field=entity_id_field
        )
    

if __name__ == "__main__":
    main() 