"""
Dataflow Pipeline for Waymo Multimodal Feature Processing

This module implements a distributed Apache Beam pipeline for processing
Waymo Motion and Reasoning datasets at scale using Google Cloud Dataflow.
"""

import os
import logging
import argparse
import json
import tempfile
from typing import Dict, List, Tuple, Any, Iterable, Optional

import apache_beam as beam
from apache_beam.options.pipeline_options import PipelineOptions, GoogleCloudOptions
from apache_beam.options.pipeline_options import SetupOptions, StandardOptions
from apache_beam.io.gcp.gcsio import GcsIO
from apache_beam.io.tfrecordio import ReadFromTFRecord
from apache_beam.metrics import Metrics, MetricsFilter

import tensorflow as tf
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class WaymoFeatureOptions(PipelineOptions):
    """Custom options for Waymo Feature Pipeline."""

    @classmethod
    def _add_argparse_args(cls, parser):
        # Input/output options
        parser.add_argument(
            "--motion_data_path",
            help="Path to motion dataset (GCS or local)",
            required=True
        )
        parser.add_argument(
            "--reasoning_data_path",
            help="Path to reasoning dataset (GCS or local)",
            required=True
        )
        parser.add_argument(
            "--output_path",
            help="Output path for processed features (GCS or local)",
            required=True
        )
        parser.add_argument(
            "--model_path",
            help="Path to save/load embedding model (GCS or local)",
            default=None
        )
        
        # Processing options
        parser.add_argument(
            "--max_scenarios",
            type=int,
            help="Maximum number of scenarios to process (0 for all)",
            default=0
        )
        parser.add_argument(
            "--embedding_model",
            help="Sentence transformer model to use for text embeddings",
            default="all-MiniLM-L6-v2"
        )
        parser.add_argument(
            "--embedding_dim",
            type=int,
            help="Dimension of text embeddings",
            default=384
        )
        parser.add_argument(
            "--batch_size",
            type=int,
            help="Batch size for feature extraction",
            default=10
        )


class ParseTFRecord(beam.DoFn):
    """Parses TFRecord examples into dictionaries."""
    
    def __init__(self, dataset_type):
        """Initialize the parser.
        
        Args:
            dataset_type: Type of dataset, either 'motion' or 'reasoning'
        """
        self.dataset_type = dataset_type
        self.num_records_processed = Metrics.counter(self.__class__.__name__, 'records_processed')
        self.num_parsing_errors = Metrics.counter(self.__class__.__name__, 'parsing_errors')
    
    def parse_motion_example(self, serialized_example):
        """Parse a motion dataset example.
        
        Args:
            serialized_example: Serialized TFRecord example
            
        Returns:
            Dictionary containing parsed features
        """
        feature_description = {
            'scenario/id': tf.io.FixedLenFeature([], tf.string),
            'state/id': tf.io.FixedLenFeature([1, 128], tf.int64),
            'state/type': tf.io.FixedLenFeature([1, 128], tf.int64),
            'state/is_sdc': tf.io.FixedLenFeature([1, 128], tf.int64),
            'state/tracks_to_predict': tf.io.FixedLenFeature([1, 128], tf.int64),
            'state/current/position': tf.io.FixedLenFeature([1, 128, 2], tf.float32),
            'state/current/velocity': tf.io.FixedLenFeature([1, 128, 2], tf.float32),
            'state/current/heading': tf.io.FixedLenFeature([1, 128, 1], tf.float32),
            'state/current/bbox_yaw': tf.io.FixedLenFeature([1, 128, 1], tf.float32),
            'state/current/valid': tf.io.FixedLenFeature([1, 128], tf.int64)
        }
        
        example = tf.io.parse_single_example(serialized_example, feature_description)
        return {key: value.numpy() for key, value in example.items()}
    
    def parse_reasoning_example(self, serialized_example):
        """Parse a reasoning dataset example.
        
        Args:
            serialized_example: Serialized TFRecord example
            
        Returns:
            Dictionary containing parsed features
        """
        feature_description = {
            'scenario/id': tf.io.FixedLenFeature([], tf.string),
            'sdc/id': tf.io.FixedLenFeature([], tf.int64),
            'questions': tf.io.VarLenFeature(tf.string),
            'answers': tf.io.VarLenFeature(tf.string)
        }
        
        example = tf.io.parse_single_example(serialized_example, feature_description)
        
        # Convert sparse tensors to dense arrays
        questions = tf.sparse.to_dense(example['questions']).numpy()
        answers = tf.sparse.to_dense(example['answers']).numpy()
        
        # Convert byte strings to regular strings
        questions = [q.decode('utf-8') for q in questions]
        answers = [a.decode('utf-8') for a in answers]
        
        return {
            'scenario/id': example['scenario/id'].numpy().decode('utf-8'),
            'sdc/id': example['sdc/id'].numpy(),
            'questions': questions,
            'answers': answers
        }
    
    def process(self, element):
        try:
            if self.dataset_type == 'motion':
                result = self.parse_motion_example(element)
            else:
                result = self.parse_reasoning_example(element)
                
            self.num_records_processed.inc()
            yield result
        except Exception as e:
            self.num_parsing_errors.inc()
            logger.error(f"Error parsing example: {str(e)}")


class ExtractFeatures(beam.DoFn):
    """Extracts features from parsed examples."""
    
    def __init__(self, dataset_type):
        """Initialize the feature extractor.
        
        Args:
            dataset_type: Type of dataset, either 'motion' or 'reasoning'
        """
        self.dataset_type = dataset_type
        self.num_processed = Metrics.counter(self.__class__.__name__, 'scenarios_processed')
        self.num_errors = Metrics.counter(self.__class__.__name__, 'extraction_errors')
    
    def extract_motion_features(self, example):
        """Extract features from a motion example.
        
        Args:
            example: Parsed motion example
            
        Returns:
            Dictionary with extracted features
        """
        scenario_id = example['scenario/id'].decode('utf-8')
        agent_ids = example['state/id'][0]
        agent_types = example['state/type'][0]
        is_sdc = example['state/is_sdc'][0]
        positions = example['state/current/position'][0]
        velocities = example['state/current/velocity'][0]
        headings = example['state/current/heading'][0]
        valid = example['state/current/valid'][0]
        
        # Find ego vehicle index
        ego_index = np.where(is_sdc == 1)[0]
        if len(ego_index) == 0:
            # If no SDC, use first valid agent
            valid_indices = np.where(valid == 1)[0]
            if len(valid_indices) == 0:
                return None
            ego_index = valid_indices[0]
        else:
            ego_index = ego_index[0]
        
        ego_id = agent_ids[ego_index]
        ego_position = positions[ego_index]
        ego_velocity = velocities[ego_index]
        ego_heading = headings[ego_index]
        ego_speed = np.linalg.norm(ego_velocity)
        
        # Extract features for all valid agents
        features = []
        for i in range(len(agent_ids)):
            if valid[i] == 0:
                continue
                
            agent_id = agent_ids[i]
            agent_type = agent_types[i]
            position = positions[i]
            velocity = velocities[i]
            heading = headings[i]
            
            # Calculate relative position to ego
            rel_position = position - ego_position
            distance = np.linalg.norm(rel_position)
            
            # Calculate relative heading
            rel_heading = heading - ego_heading
            
            # Add to features
            features.append({
                'scenario_id': scenario_id,
                'ego_agent_id': int(ego_id),
                'agent_id': int(agent_id),
                'agent_type': int(agent_type),
                'is_ego': int(i == ego_index),
                'ego_x': float(ego_position[0]),
                'ego_y': float(ego_position[1]),
                'ego_speed': float(ego_speed),
                'ego_heading': float(ego_heading[0]),
                'agent_x': float(position[0]),
                'agent_y': float(position[1]),
                'agent_speed': float(np.linalg.norm(velocity)),
                'agent_heading': float(heading[0]),
                'rel_x': float(rel_position[0]),
                'rel_y': float(rel_position[1]),
                'distance': float(distance),
                'rel_heading': float(rel_heading[0])
            })
        
        return features
    
    def extract_reasoning_features(self, example):
        """Extract features from a reasoning example.
        
        Args:
            example: Parsed reasoning example
            
        Returns:
            Dictionary with extracted features
        """
        scenario_id = example['scenario/id']
        ego_agent_id = int(example['sdc/id'])
        questions = example['questions']
        answers = example['answers']
        
        if len(questions) != len(answers):
            logger.warning(f"Mismatch in question and answer counts for scenario {scenario_id}")
            return None
        
        # Categorize questions
        env_questions = []
        ego_questions = []
        sur_questions = []
        int_questions = []
        
        for q, a in zip(questions, answers):
            q_lower = q.lower()
            
            # Environment questions
            if any(kw in q_lower for kw in ['road', 'lane', 'traffic', 'weather', 'location']):
                env_questions.append((q, a))
            # Ego agent questions
            elif any(kw in q_lower for kw in ['ego', 'you', 'your', 'sdc', 'self-driving']):
                ego_questions.append((q, a))
            # Interaction questions
            elif any(kw in q_lower for kw in ['interact', 'yield', 'give way', 'stop for', 'wait for']):
                int_questions.append((q, a))
            # Surrounding agent questions
            else:
                sur_questions.append((q, a))
        
        # Extract behavior patterns from answers
        has_yield = any('yield' in a.lower() for _, a in int_questions)
        has_stop = any('stop' in a.lower() for _, a in int_questions)
        has_continue = any('continue' in a.lower() for _, a in int_questions)
        has_accelerate = any('accelerate' in a.lower() for _, a in int_questions)
        has_decelerate = any('decelerate' in a.lower() for _, a in int_questions)
        has_turn_left = any('turn left' in a.lower() for _, a in int_questions)
        has_turn_right = any('turn right' in a.lower() for _, a in int_questions)
        has_merge = any('merge' in a.lower() for _, a in int_questions)
        has_change_lane = any('change lane' in a.lower() for _, a in int_questions)
        
        # Combine all questions and answers for embedding
        all_qa_pairs = env_questions + ego_questions + sur_questions + int_questions
        qa_text = ' '.join([f"Q: {q} A: {a}" for q, a in all_qa_pairs])
        
        return {
            'scenario_id': scenario_id,
            'ego_agent_id': ego_agent_id,
            'qa_text': qa_text,
            'env_questions': len(env_questions),
            'ego_questions': len(ego_questions),
            'sur_questions': len(sur_questions),
            'int_questions': len(int_questions),
            'has_yield': int(has_yield),
            'has_stop': int(has_stop),
            'has_continue': int(has_continue),
            'has_accelerate': int(has_accelerate),
            'has_decelerate': int(has_decelerate),
            'has_turn_left': int(has_turn_left),
            'has_turn_right': int(has_turn_right),
            'has_merge': int(has_merge),
            'has_change_lane': int(has_change_lane)
        }
    
    def process(self, element):
        try:
            if self.dataset_type == 'motion':
                features = self.extract_motion_features(element)
            else:
                features = self.extract_reasoning_features(element)
                
            if features:
                self.num_processed.inc()
                yield features
        except Exception as e:
            self.num_errors.inc()
            logger.error(f"Error extracting features: {str(e)}")


class GenerateEmbeddings(beam.DoFn):
    """Generates text embeddings for QA pairs."""
    
    def __init__(self, model_name, embedding_dim):
        """Initialize the embedding generator.
        
        Args:
            model_name: Name of the SentenceTransformer model
            embedding_dim: Dimension of the embeddings
        """
        self.model_name = model_name
        self.embedding_dim = embedding_dim
        self.model = None
        self.num_embedded = Metrics.counter(self.__class__.__name__, 'texts_embedded')
    
    def setup(self):
        """Initialize the embedding model."""
        self.model = SentenceTransformer(self.model_name)
    
    def process(self, element):
        if not self.model:
            self.setup()
            
        text = element['qa_text']
        
        # Generate embedding
        embedding = self.model.encode(text)
        
        # Add embedding to element
        element['text_embedding'] = embedding.tolist()
        
        self.num_embedded.inc()
        yield element


class CombineFeatures(beam.DoFn):
    """Combines motion and reasoning features."""
    
    def process(self, element):
        scenario_id, feature_groups = element
        
        if 'motion' not in feature_groups or 'reasoning' not in feature_groups:
            return
        
        motion_features = feature_groups['motion']
        reasoning_features = feature_groups['reasoning']
        
        # Match ego agent IDs
        if motion_features[0]['ego_agent_id'] != reasoning_features['ego_agent_id']:
            logger.warning(f"Ego agent ID mismatch for scenario {scenario_id}")
            return
        
        # Add reasoning features to each motion feature record
        for motion_feat in motion_features:
            combined_feat = motion_feat.copy()
            
            # Add reasoning features
            combined_feat.update({
                'env_questions': reasoning_features['env_questions'],
                'ego_questions': reasoning_features['ego_questions'],
                'sur_questions': reasoning_features['sur_questions'],
                'int_questions': reasoning_features['int_questions'],
                'has_yield': reasoning_features['has_yield'],
                'has_stop': reasoning_features['has_stop'],
                'has_continue': reasoning_features['has_continue'],
                'has_accelerate': reasoning_features['has_accelerate'],
                'has_decelerate': reasoning_features['has_decelerate'],
                'has_turn_left': reasoning_features['has_turn_left'],
                'has_turn_right': reasoning_features['has_turn_right'],
                'has_merge': reasoning_features['has_merge'],
                'has_change_lane': reasoning_features['has_change_lane'],
                'text_embedding': reasoning_features['text_embedding']
            })
            
            yield combined_feat


def run(argv=None):
    """Run the Dataflow pipeline."""
    # Parse pipeline options
    pipeline_options = PipelineOptions(argv)
    waymo_options = pipeline_options.view_as(WaymoFeatureOptions)
    
    # Set up Google Cloud options
    gcp_options = pipeline_options.view_as(GoogleCloudOptions)
    if not gcp_options.project:
        parser.error('Please provide a GCP project ID with --project')
    
    # Set up other options
    pipeline_options.view_as(SetupOptions).save_main_session = True
    pipeline_options.view_as(StandardOptions).runner = 'DataflowRunner'
    
    # Access options
    motion_data_path = waymo_options.motion_data_path
    reasoning_data_path = waymo_options.reasoning_data_path
    output_path = waymo_options.output_path
    max_scenarios = waymo_options.max_scenarios
    embedding_model = waymo_options.embedding_model
    embedding_dim = waymo_options.embedding_dim
    
    # Create and run the pipeline
    with beam.Pipeline(options=pipeline_options) as p:
        # Read and process motion data
        motion_features = (
            p 
            | "Read Motion TFRecords" >> ReadFromTFRecord(
                motion_data_path, coder=beam.coders.BytesCoder())
            | "Parse Motion Examples" >> beam.ParDo(ParseTFRecord('motion'))
            | "Extract Motion Features" >> beam.ParDo(ExtractFeatures('motion'))
        )
        
        # Read and process reasoning data
        reasoning_features = (
            p 
            | "Read Reasoning TFRecords" >> ReadFromTFRecord(
                reasoning_data_path, coder=beam.coders.BytesCoder())
            | "Parse Reasoning Examples" >> beam.ParDo(ParseTFRecord('reasoning'))
            | "Extract Reasoning Features" >> beam.ParDo(ExtractFeatures('reasoning'))
            | "Generate Text Embeddings" >> beam.ParDo(GenerateEmbeddings(embedding_model, embedding_dim))
        )
        
        # Prepare motion features for combining (group by scenario_id)
        motion_by_scenario = (
            motion_features
            | "Key Motion by Scenario" >> beam.Map(lambda x: (x['scenario_id'], x))
            | "Group Motion by Scenario" >> beam.GroupByKey()
            | "Format Motion Groups" >> beam.Map(lambda x: (x[0], {'motion': list(x[1])}))
        )
        
        # Prepare reasoning features for combining
        reasoning_by_scenario = (
            reasoning_features
            | "Key Reasoning by Scenario" >> beam.Map(lambda x: (x['scenario_id'], x))
            | "Format Reasoning Groups" >> beam.Map(lambda x: (x[0], {'reasoning': x[1]}))
        )
        
        # Combine features by scenario_id
        combined_features = (
            (motion_by_scenario, reasoning_by_scenario)
            | "Merge Feature Groups" >> beam.CoGroupByKey()
            | "Combine Features" >> beam.ParDo(CombineFeatures())
        )
        
        # Limit scenarios if specified
        if max_scenarios > 0:
            combined_features = (
                combined_features
                | f"Limit to {max_scenarios} Scenarios" >> beam.transforms.combiners.Sample.FixedSizeGlobally(max_scenarios)
                | "Flatten Limited Sample" >> beam.FlatMap(lambda x: x)
            )
        
        # Write features to output files
        _ = (
            combined_features
            | "Convert to JSON" >> beam.Map(lambda x: json.dumps(x))
            | "Write Features" >> beam.io.WriteToText(
                f"{output_path}/combined_features",
                file_name_suffix=".json",
                num_shards=10
            )
        )
        
        # Write feature metadata
        feature_schema = {
            'scenario_id': 'string',
            'ego_agent_id': 'int',
            'agent_id': 'int',
            'agent_type': 'int',
            'is_ego': 'int',
            'ego_x': 'float',
            'ego_y': 'float',
            'ego_speed': 'float',
            'ego_heading': 'float',
            'agent_x': 'float',
            'agent_y': 'float',
            'agent_speed': 'float',
            'agent_heading': 'float',
            'rel_x': 'float',
            'rel_y': 'float',
            'distance': 'float',
            'rel_heading': 'float',
            'env_questions': 'int',
            'ego_questions': 'int',
            'sur_questions': 'int',
            'int_questions': 'int',
            'has_yield': 'int',
            'has_stop': 'int',
            'has_continue': 'int',
            'has_accelerate': 'int',
            'has_decelerate': 'int',
            'has_turn_left': 'int',
            'has_turn_right': 'int',
            'has_merge': 'int',
            'has_change_lane': 'int',
            'text_embedding': f'float[{embedding_dim}]'
        }
        
        schema_file = f"{output_path}/schema.json"
        with open(schema_file if not schema_file.startswith('gs://') else tempfile.NamedTemporaryFile().name, 'w') as f:
            json.dump(feature_schema, f)
            
        # Upload schema file to GCS if needed
        if schema_file.startswith('gs://'):
            GcsIO().open(schema_file, 'w').write(json.dumps(feature_schema).encode('utf-8'))


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description='Waymo Multimodal Feature Pipeline')
    
    # Parse known args to check if help is needed
    known_args, pipeline_args = parser.parse_known_args()
    
    # Run the pipeline
    run(pipeline_args)


if __name__ == '__main__':
    main() 