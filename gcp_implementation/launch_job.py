#!/usr/bin/env python3
"""
Launch script for Waymo Multimodal Feature ETL Jobs

This script provides a command-line interface for launching Dataflow jobs
for the Waymo multimodal feature extraction pipeline on Google Cloud Platform.
"""

import os
import sys
import json
import time
import logging
import argparse
import datetime
import subprocess
from typing import Dict, List, Optional, Any

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class DataflowJobLauncher:
    """Manages the launching and monitoring of Dataflow jobs."""

    def __init__(
        self,
        project_id: str,
        region: str,
        staging_location: str,
        temp_location: str,
        service_account: Optional[str] = None,
        network: Optional[str] = None,
        subnetwork: Optional[str] = None
    ):
        """Initialize the Dataflow job launcher.
        
        Args:
            project_id: GCP project ID
            region: GCP region for the Dataflow job
            staging_location: GCS location for staging job files
            temp_location: GCS location for temporary job files
            service_account: Service account email to run the job as
            network: VPC network to use for the job
            subnetwork: VPC subnetwork to use for the job
        """
        self.project_id = project_id
        self.region = region
        self.staging_location = staging_location
        self.temp_location = temp_location
        self.service_account = service_account
        self.network = network
        self.subnetwork = subnetwork
        
        # Validate GCS paths
        for path in [staging_location, temp_location]:
            if not path.startswith("gs://"):
                raise ValueError(f"GCS path should start with 'gs://': {path}")
    
    def launch_dataflow_pipeline(
        self,
        pipeline_script: str,
        job_name: str,
        pipeline_args: Dict[str, Any],
        machine_type: str = "n1-standard-4",
        max_workers: int = 10,
        autoscaling_algorithm: str = "THROUGHPUT_BASED",
        disk_size_gb: int = 100,
        worker_disk_type: str = "compute.googleapis.com/projects/PROJECT_ID/zones/ZONE/diskTypes/pd-ssd",
        dataflow_service_options: Optional[List[str]] = None
    ) -> str:
        """Launch a Dataflow pipeline job.
        
        Args:
            pipeline_script: Path to the pipeline script
            job_name: Name for the Dataflow job
            pipeline_args: Pipeline-specific arguments
            machine_type: Compute engine machine type
            max_workers: Maximum number of workers
            autoscaling_algorithm: Algorithm for autoscaling (THROUGHPUT_BASED or NONE)
            disk_size_gb: Size of worker persistent disk in GB
            worker_disk_type: Type of worker persistent disk
            dataflow_service_options: Additional Dataflow service options
            
        Returns:
            Job ID of the launched Dataflow job
        """
        # Construct timestamp for unique job name
        timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        job_name = f"{job_name}-{timestamp}"
        
        # Replace project ID and zone placeholders in disk type
        worker_disk_type = worker_disk_type.replace("PROJECT_ID", self.project_id)
        worker_disk_type = worker_disk_type.replace("ZONE", f"{self.region}-a")
        
        # Construct base command and arguments
        cmd = [sys.executable, pipeline_script]
        
        # Add standard Dataflow arguments
        dataflow_args = [
            f"--project={self.project_id}",
            f"--region={self.region}",
            f"--job_name={job_name}",
            f"--staging_location={self.staging_location}",
            f"--temp_location={self.temp_location}",
            f"--worker_machine_type={machine_type}",
            f"--max_num_workers={max_workers}",
            f"--disk_size_gb={disk_size_gb}",
            f"--worker_disk_type={worker_disk_type}",
            f"--experiments=use_runner_v2",
            f"--runner=DataflowRunner"
        ]
        
        # Add autoscaling algorithm
        if autoscaling_algorithm:
            dataflow_args.append(f"--autoscaling_algorithm={autoscaling_algorithm}")
        
        # Add service account if specified
        if self.service_account:
            dataflow_args.append(f"--service_account_email={self.service_account}")
        
        # Add network configuration if specified
        if self.network:
            dataflow_args.append(f"--network={self.network}")
        if self.subnetwork:
            dataflow_args.append(f"--subnetwork={self.subnetwork}")
        
        # Add Dataflow service options if specified
        if dataflow_service_options:
            for option in dataflow_service_options:
                dataflow_args.append(f"--dataflow_service_option={option}")
        
        # Add pipeline-specific arguments
        pipeline_cmd_args = []
        for key, value in pipeline_args.items():
            if isinstance(value, bool):
                if value:
                    pipeline_cmd_args.append(f"--{key}")
            else:
                pipeline_cmd_args.append(f"--{key}={value}")
        
        # Combine all arguments and construct final command
        full_cmd = cmd + dataflow_args + pipeline_cmd_args
        
        logger.info("Launching Dataflow job with command:")
        logger.info(" ".join(full_cmd))
        
        # Launch the job
        try:
            process = subprocess.Popen(
                full_cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                universal_newlines=True
            )
            
            # Monitor the process output to get the job ID
            job_id = None
            for line in process.stdout:
                logger.info(line.strip())
                if "Submitted job:" in line:
                    job_id = line.strip().split("Submitted job:")[-1].strip()
            
            # Check for errors
            _, stderr = process.communicate()
            if process.returncode != 0:
                logger.error(f"Failed to launch Dataflow job: {stderr}")
                raise RuntimeError(f"Failed to launch Dataflow job: {stderr}")
            
            if job_id:
                logger.info(f"Successfully launched Dataflow job with ID: {job_id}")
                return job_id
            else:
                logger.warning("Job submitted but couldn't extract job ID")
                return job_name  # Return job name as fallback
            
        except Exception as e:
            logger.error(f"Error launching Dataflow job: {str(e)}")
            raise
    
    def launch_waymo_feature_pipeline(
        self,
        motion_data_path: str,
        reasoning_data_path: str,
        output_path: str,
        pipeline_script: str = "gcp_implementation/dataflow_pipeline.py",
        job_name: str = "waymo-feature-pipeline",
        max_scenarios: int = 0,
        embedding_model: str = "all-MiniLM-L6-v2",
        embedding_dim: int = 384,
        batch_size: int = 10,
        machine_type: str = "n1-standard-4",
        max_workers: int = 10
    ) -> str:
        """Launch the Waymo feature pipeline job.
        
        Args:
            motion_data_path: GCS path to the motion dataset
            reasoning_data_path: GCS path to the reasoning dataset
            output_path: GCS path for the output
            pipeline_script: Path to the pipeline script
            job_name: Name for the Dataflow job
            max_scenarios: Maximum number of scenarios to process (0 for all)
            embedding_model: Sentence transformer model to use
            embedding_dim: Dimension of text embeddings
            batch_size: Batch size for feature extraction
            machine_type: Compute engine machine type
            max_workers: Maximum number of workers
            
        Returns:
            Job ID of the launched Dataflow job
        """
        # Validate GCS paths
        for path in [motion_data_path, reasoning_data_path, output_path]:
            if not path.startswith("gs://"):
                raise ValueError(f"GCS path should start with 'gs://': {path}")
        
        # Prepare pipeline arguments
        pipeline_args = {
            "motion_data_path": motion_data_path,
            "reasoning_data_path": reasoning_data_path,
            "output_path": output_path,
            "max_scenarios": max_scenarios,
            "embedding_model": embedding_model,
            "embedding_dim": embedding_dim,
            "batch_size": batch_size
        }
        
        # Launch the pipeline
        return self.launch_dataflow_pipeline(
            pipeline_script=pipeline_script,
            job_name=job_name,
            pipeline_args=pipeline_args,
            machine_type=machine_type,
            max_workers=max_workers
        )
    
    def launch_feature_ingest_job(
        self,
        feature_store_id: str,
        source_path: str,
        entity_type: str = "agent",
        pipeline_script: str = "gcp_implementation/feature_ingest_pipeline.py",
        job_name: str = "waymo-feature-ingest",
        machine_type: str = "n1-standard-2",
        max_workers: int = 5
    ) -> str:
        """Launch a job to ingest features into Vertex AI Feature Store.
        
        Args:
            feature_store_id: ID of the feature store
            source_path: GCS path to the source data
            entity_type: Entity type to ingest (agent or scenario)
            pipeline_script: Path to the pipeline script
            job_name: Name for the Dataflow job
            machine_type: Compute engine machine type
            max_workers: Maximum number of workers
            
        Returns:
            Job ID of the launched Dataflow job
        """
        # Validate GCS path
        if not source_path.startswith("gs://"):
            raise ValueError(f"GCS path should start with 'gs://': {source_path}")
        
        # Prepare pipeline arguments
        pipeline_args = {
            "project": self.project_id,
            "location": self.region,
            "feature_store_id": feature_store_id,
            "source_path": source_path,
            "entity_type": entity_type
        }
        
        # Launch the pipeline
        return self.launch_dataflow_pipeline(
            pipeline_script=pipeline_script,
            job_name=job_name,
            pipeline_args=pipeline_args,
            machine_type=machine_type,
            max_workers=max_workers,
            autoscaling_algorithm="NONE"  # Fixed worker count for feature ingest
        )


def main():
    """Main entry point for launching Dataflow jobs."""
    parser = argparse.ArgumentParser(
        description="Launch Dataflow jobs for Waymo multimodal feature pipeline"
    )
    
    # Common options
    parser.add_argument(
        "--project_id",
        required=True,
        help="GCP project ID"
    )
    parser.add_argument(
        "--region",
        default="us-central1",
        help="GCP region for the Dataflow job"
    )
    parser.add_argument(
        "--staging_location",
        required=True,
        help="GCS location for staging job files"
    )
    parser.add_argument(
        "--temp_location",
        required=True,
        help="GCS location for temporary job files"
    )
    parser.add_argument(
        "--service_account",
        default=None,
        help="Service account email to run the job as"
    )
    parser.add_argument(
        "--network",
        default=None,
        help="VPC network to use for the job"
    )
    parser.add_argument(
        "--subnetwork",
        default=None,
        help="VPC subnetwork to use for the job"
    )
    
    # Job type selection
    subparsers = parser.add_subparsers(dest="job_type", help="Type of job to launch")
    subparsers.required = True
    
    # Feature pipeline job
    feature_parser = subparsers.add_parser(
        "feature-pipeline",
        help="Launch the feature extraction pipeline"
    )
    feature_parser.add_argument(
        "--motion_data_path",
        required=True,
        help="GCS path to the motion dataset"
    )
    feature_parser.add_argument(
        "--reasoning_data_path",
        required=True,
        help="GCS path to the reasoning dataset"
    )
    feature_parser.add_argument(
        "--output_path",
        required=True,
        help="GCS path for the output"
    )
    feature_parser.add_argument(
        "--pipeline_script",
        default="gcp_implementation/dataflow_pipeline.py",
        help="Path to the pipeline script"
    )
    feature_parser.add_argument(
        "--job_name",
        default="waymo-feature-pipeline",
        help="Name for the Dataflow job"
    )
    feature_parser.add_argument(
        "--max_scenarios",
        type=int,
        default=0,
        help="Maximum number of scenarios to process (0 for all)"
    )
    feature_parser.add_argument(
        "--embedding_model",
        default="all-MiniLM-L6-v2",
        help="Sentence transformer model to use"
    )
    feature_parser.add_argument(
        "--embedding_dim",
        type=int,
        default=384,
        help="Dimension of text embeddings"
    )
    feature_parser.add_argument(
        "--batch_size",
        type=int,
        default=10,
        help="Batch size for feature extraction"
    )
    feature_parser.add_argument(
        "--machine_type",
        default="n1-standard-4",
        help="Compute engine machine type"
    )
    feature_parser.add_argument(
        "--max_workers",
        type=int,
        default=10,
        help="Maximum number of workers"
    )
    
    # Feature ingest job
    ingest_parser = subparsers.add_parser(
        "feature-ingest",
        help="Launch the feature ingest job"
    )
    ingest_parser.add_argument(
        "--feature_store_id",
        required=True,
        help="ID of the feature store"
    )
    ingest_parser.add_argument(
        "--source_path",
        required=True,
        help="GCS path to the source data"
    )
    ingest_parser.add_argument(
        "--entity_type",
        default="agent",
        choices=["agent", "scenario"],
        help="Entity type to ingest (agent or scenario)"
    )
    ingest_parser.add_argument(
        "--pipeline_script",
        default="gcp_implementation/feature_ingest_pipeline.py",
        help="Path to the pipeline script"
    )
    ingest_parser.add_argument(
        "--job_name",
        default="waymo-feature-ingest",
        help="Name for the Dataflow job"
    )
    ingest_parser.add_argument(
        "--machine_type",
        default="n1-standard-2",
        help="Compute engine machine type"
    )
    ingest_parser.add_argument(
        "--max_workers",
        type=int,
        default=5,
        help="Maximum number of workers"
    )
    
    # Custom pipeline job
    custom_parser = subparsers.add_parser(
        "custom",
        help="Launch a custom Dataflow pipeline"
    )
    custom_parser.add_argument(
        "--pipeline_script",
        required=True,
        help="Path to the pipeline script"
    )
    custom_parser.add_argument(
        "--job_name",
        required=True,
        help="Name for the Dataflow job"
    )
    custom_parser.add_argument(
        "--args",
        type=json.loads,
        default={},
        help="JSON string of pipeline arguments"
    )
    custom_parser.add_argument(
        "--machine_type",
        default="n1-standard-4",
        help="Compute engine machine type"
    )
    custom_parser.add_argument(
        "--max_workers",
        type=int,
        default=10,
        help="Maximum number of workers"
    )
    custom_parser.add_argument(
        "--autoscaling_algorithm",
        default="THROUGHPUT_BASED",
        choices=["THROUGHPUT_BASED", "NONE"],
        help="Algorithm for autoscaling"
    )
    custom_parser.add_argument(
        "--disk_size_gb",
        type=int,
        default=100,
        help="Size of worker persistent disk in GB"
    )
    
    args = parser.parse_args()
    
    # Initialize the job launcher
    launcher = DataflowJobLauncher(
        project_id=args.project_id,
        region=args.region,
        staging_location=args.staging_location,
        temp_location=args.temp_location,
        service_account=args.service_account,
        network=args.network,
        subnetwork=args.subnetwork
    )
    
    # Launch the appropriate job
    if args.job_type == "feature-pipeline":
        job_id = launcher.launch_waymo_feature_pipeline(
            motion_data_path=args.motion_data_path,
            reasoning_data_path=args.reasoning_data_path,
            output_path=args.output_path,
            pipeline_script=args.pipeline_script,
            job_name=args.job_name,
            max_scenarios=args.max_scenarios,
            embedding_model=args.embedding_model,
            embedding_dim=args.embedding_dim,
            batch_size=args.batch_size,
            machine_type=args.machine_type,
            max_workers=args.max_workers
        )
    
    elif args.job_type == "feature-ingest":
        job_id = launcher.launch_feature_ingest_job(
            feature_store_id=args.feature_store_id,
            source_path=args.source_path,
            entity_type=args.entity_type,
            pipeline_script=args.pipeline_script,
            job_name=args.job_name,
            machine_type=args.machine_type,
            max_workers=args.max_workers
        )
    
    elif args.job_type == "custom":
        job_id = launcher.launch_dataflow_pipeline(
            pipeline_script=args.pipeline_script,
            job_name=args.job_name,
            pipeline_args=args.args,
            machine_type=args.machine_type,
            max_workers=args.max_workers,
            autoscaling_algorithm=args.autoscaling_algorithm,
            disk_size_gb=args.disk_size_gb
        )
    
    print(f"Job ID: {job_id}")
    print(f"Console URL: https://console.cloud.google.com/dataflow/jobs/{args.region}/{job_id}?project={args.project_id}")


if __name__ == "__main__":
    main() 