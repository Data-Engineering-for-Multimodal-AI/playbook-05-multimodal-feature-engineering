"""
Deployment module for Waymo Multimodal Feature Pipeline

This module handles the deployment of models as endpoints on Vertex AI,
enabling online prediction for the Waymo multimodal features.
"""

import os
import logging
import argparse
import json
import time
from typing import Dict, List, Optional, Union

from google.cloud import aiplatform
from google.cloud.aiplatform.models import Model
from google.cloud.aiplatform.utils import resource_manager
from google.cloud.aiplatform.compat.types import endpoint_service

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class EndpointDeployer:
    """Class for deploying model endpoints on Vertex AI."""
    
    def __init__(
        self,
        project_id: str,
        location: str,
        model_id: Optional[str] = None,
        model_name: Optional[str] = None,
    ):
        """Initialize the EndpointDeployer.
        
        Args:
            project_id: GCP project ID
            location: GCP region
            model_id: Vertex AI model ID (either model_id or model_name must be provided)
            model_name: Vertex AI model display name (either model_id or model_name must be provided)
        """
        if not model_id and not model_name:
            raise ValueError("Either model_id or model_name must be provided")
        
        self.project_id = project_id
        self.location = location
        self.model_id = model_id
        self.model_name = model_name
        
        # Initialize Vertex AI
        aiplatform.init(project=project_id, location=location)
    
    def get_model(self) -> Model:
        """Get the model to deploy.
        
        Returns:
            Vertex AI Model object
        """
        if self.model_id:
            # Get model by ID
            model = aiplatform.Model(model_name=self.model_id)
            logger.info(f"Found model by ID: {model.display_name}")
        else:
            # Get model by display name
            models = aiplatform.Model.list(
                filter=f"display_name={self.model_name}",
                order_by="create_time desc"
            )
            
            if not models:
                raise ValueError(f"No models found with name: {self.model_name}")
            
            # Use the most recently created model
            model = models[0]
            logger.info(f"Found model by name: {model.display_name}, ID: {model.resource_name}")
        
        return model
    
    def create_endpoint(
        self,
        endpoint_name: str,
        description: str = "Waymo yield prediction endpoint"
    ) -> aiplatform.Endpoint:
        """Create a new endpoint.
        
        Args:
            endpoint_name: Display name for the endpoint
            description: Description of the endpoint
            
        Returns:
            Vertex AI Endpoint object
        """
        # Check if endpoint already exists
        endpoints = aiplatform.Endpoint.list(
            filter=f"display_name={endpoint_name}",
            order_by="create_time desc"
        )
        
        if endpoints:
            logger.info(f"Found existing endpoint: {endpoint_name}")
            return endpoints[0]
        
        # Create new endpoint
        endpoint = aiplatform.Endpoint.create(
            display_name=endpoint_name,
            description=description,
            project=self.project_id,
            location=self.location
        )
        
        logger.info(f"Created new endpoint: {endpoint.display_name}, ID: {endpoint.resource_name}")
        return endpoint
    
    def deploy_model(
        self,
        endpoint: aiplatform.Endpoint,
        machine_type: str = "n1-standard-2",
        min_replica_count: int = 1,
        max_replica_count: int = 2,
        traffic_percentage: int = 100,
        deployed_model_display_name: Optional[str] = None,
        accelerator_type: Optional[str] = None,
        accelerator_count: Optional[int] = None
    ) -> aiplatform.Model:
        """Deploy model to an endpoint.
        
        Args:
            endpoint: Vertex AI Endpoint to deploy to
            machine_type: Machine type for deployment
            min_replica_count: Minimum number of replicas
            max_replica_count: Maximum number of replicas
            traffic_percentage: Percentage of traffic to allocate to this model
            deployed_model_display_name: Display name for the deployed model
            accelerator_type: Type of accelerator (e.g., "NVIDIA_TESLA_T4")
            accelerator_count: Number of accelerators
            
        Returns:
            Deployed Vertex AI Model
        """
        # Get model
        model = self.get_model()
        
        # Set deployed model display name
        if not deployed_model_display_name:
            deployed_model_display_name = f"{model.display_name}-deployment"
        
        # Deploy model
        logger.info(f"Deploying model {model.display_name} to endpoint {endpoint.display_name}")
        
        deployment_args = {
            "traffic_percentage": traffic_percentage,
            "machine_type": machine_type,
            "min_replica_count": min_replica_count,
            "max_replica_count": max_replica_count,
            "deployed_model_display_name": deployed_model_display_name,
        }
        
        # Add accelerator if specified
        if accelerator_type and accelerator_count:
            deployment_args["accelerator_type"] = accelerator_type
            deployment_args["accelerator_count"] = accelerator_count
        
        # Deploy model to endpoint
        model.deploy(
            endpoint=endpoint,
            **deployment_args
        )
        
        logger.info(f"Model deployed successfully to endpoint {endpoint.resource_name}")
        return model
    
    def setup_autoscaling(
        self,
        endpoint: aiplatform.Endpoint,
        deployed_model_id: str,
        min_replica_count: int = 1,
        max_replica_count: int = 5,
        cpu_utilization_target: int = 60
    ) -> None:
        """Setup autoscaling for a deployed model.
        
        Args:
            endpoint: Vertex AI Endpoint containing the deployed model
            deployed_model_id: ID of the deployed model
            min_replica_count: Minimum number of replicas
            max_replica_count: Maximum number of replicas
            cpu_utilization_target: Target CPU utilization percentage
        """
        client = aiplatform.gapic.EndpointServiceClient(
            client_options={"api_endpoint": f"{self.location}-aiplatform.googleapis.com"}
        )
        
        # Get endpoint details
        endpoint_name = endpoint.resource_name
        
        # Set up autoscaling
        request = endpoint_service.UpdateEndpointRequest(
            endpoint=endpoint_name,
            deployed_model=endpoint_service.DeployedModel(
                id=deployed_model_id,
                automatic_resources=endpoint_service.AutomaticResources(
                    min_replica_count=min_replica_count,
                    max_replica_count=max_replica_count
                )
            ),
            update_mask={"paths": ["deployed_model.automatic_resources"]}
        )
        
        # Update endpoint
        client.update_endpoint(request=request)
        
        logger.info(f"Autoscaling configured for deployed model {deployed_model_id} on endpoint {endpoint_name}")
    
    def create_traffic_split(
        self,
        endpoint: aiplatform.Endpoint,
        traffic_split: Dict[str, int]
    ) -> None:
        """Configure traffic split between deployed models.
        
        Args:
            endpoint: Vertex AI Endpoint
            traffic_split: Dictionary mapping deployed model IDs to traffic percentages
        """
        # Get endpoint details
        endpoint_name = endpoint.resource_name
        
        # Validate traffic split
        total_percentage = sum(traffic_split.values())
        if total_percentage != 100:
            raise ValueError(f"Traffic split percentages must sum to 100, got {total_percentage}")
        
        # Update traffic split
        endpoint.update(traffic_split=traffic_split)
        
        logger.info(f"Traffic split updated for endpoint {endpoint_name}: {traffic_split}")
    
    def run_deployment(
        self,
        endpoint_name: str,
        machine_type: str = "n1-standard-2",
        min_replica_count: int = 1,
        max_replica_count: int = 2,
        description: str = "Waymo yield prediction endpoint"
    ) -> Dict[str, Union[str, aiplatform.Endpoint, aiplatform.Model]]:
        """Run the complete deployment workflow.
        
        Args:
            endpoint_name: Display name for the endpoint
            machine_type: Machine type for deployment
            min_replica_count: Minimum number of replicas
            max_replica_count: Maximum number of replicas
            description: Description of the endpoint
            
        Returns:
            Dictionary with deployment results
        """
        # Create endpoint
        endpoint = self.create_endpoint(endpoint_name, description)
        
        # Deploy model
        model = self.deploy_model(
            endpoint=endpoint,
            machine_type=machine_type,
            min_replica_count=min_replica_count,
            max_replica_count=max_replica_count
        )
        
        # Wait for deployment to complete
        logger.info("Waiting for deployment to complete...")
        time.sleep(60)  # Wait for deployment to initialize
        
        # Get deployed model ID
        deployed_models = endpoint.list_models()
        if not deployed_models:
            raise ValueError(f"No deployed models found on endpoint {endpoint.resource_name}")
        
        deployed_model_id = deployed_models[0].id
        
        # Setup autoscaling (optional)
        # self.setup_autoscaling(
        #     endpoint=endpoint,
        #     deployed_model_id=deployed_model_id,
        #     min_replica_count=min_replica_count,
        #     max_replica_count=max_replica_count
        # )
        
        # Return results
        results = {
            'endpoint_id': endpoint.resource_name,
            'endpoint_display_name': endpoint.display_name,
            'model_id': model.resource_name,
            'model_display_name': model.display_name,
            'deployed_model_id': deployed_model_id,
            'endpoint': endpoint,
            'model': model
        }
        
        logger.info(f"Deployment completed successfully: {endpoint.display_name}")
        return results
    
    def create_explanation_metadata(self) -> Dict:
        """Create explanation metadata for model explanations.
        
        Returns:
            Dictionary containing explanation metadata
        """
        return {
            "inputs": {
                "ego_speed": {
                    "input_tensor_name": "ego_speed",
                    "input_baselines": [0.0],
                    "modality": "numeric",
                    "encoding": "identity"
                },
                "agent_distance": {
                    "input_tensor_name": "agent_distance",
                    "input_baselines": [0.0],
                    "modality": "numeric",
                    "encoding": "identity"
                },
                "text_embedding": {
                    "input_tensor_name": "text_embedding",
                    "input_baselines": [0.0] * 384,  # Default embedding size
                    "modality": "numeric",
                    "encoding": "identity"
                }
            },
            "outputs": {
                "yield_prediction": {
                    "output_tensor_name": "yield_prediction",
                }
            }
        }
    
    def create_explanation_parameters(self) -> Dict:
        """Create explanation parameters for model explanations.
        
        Returns:
            Dictionary containing explanation parameters
        """
        return {
            "sampled_shapley_attribution": {
                "path_count": 10
            }
        }


def main():
    """Main function."""
    parser = argparse.ArgumentParser(description="Deploy models on Vertex AI")
    parser.add_argument("--project", required=True, help="GCP project ID")
    parser.add_argument("--location", default="us-central1", help="GCP region")
    parser.add_argument("--model-id", help="Vertex AI model ID")
    parser.add_argument("--model-name", help="Vertex AI model display name")
    parser.add_argument("--endpoint-name", required=True, help="Endpoint display name")
    parser.add_argument("--machine-type", default="n1-standard-2", help="Machine type for deployment")
    parser.add_argument("--min-replicas", type=int, default=1, help="Minimum number of replicas")
    parser.add_argument("--max-replicas", type=int, default=2, help="Maximum number of replicas")
    parser.add_argument("--description", default="Waymo yield prediction endpoint", 
                      help="Description of the endpoint")
    
    args = parser.parse_args()
    
    if not args.model_id and not args.model_name:
        parser.error("Either --model-id or --model-name must be provided")
    
    # Initialize deployer
    deployer = EndpointDeployer(
        project_id=args.project,
        location=args.location,
        model_id=args.model_id,
        model_name=args.model_name
    )
    
    # Run deployment
    results = deployer.run_deployment(
        endpoint_name=args.endpoint_name,
        machine_type=args.machine_type,
        min_replica_count=args.min_replicas,
        max_replica_count=args.max_replicas,
        description=args.description
    )
    
    print(f"\nDeployment complete!")
    print(f"Endpoint URL: https://console.cloud.google.com/vertex-ai/endpoints/{results['endpoint_id'].split('/')[-1]}?project={args.project}")
    print(f"Model URL: https://console.cloud.google.com/vertex-ai/models/{results['model_id'].split('/')[-1]}?project={args.project}")


if __name__ == "__main__":
    main() 