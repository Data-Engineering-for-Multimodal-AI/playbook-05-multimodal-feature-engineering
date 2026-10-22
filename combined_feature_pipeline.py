import tensorflow as tf
import numpy as np
import pandas as pd
import json
import os
from tqdm import tqdm
from collections import defaultdict
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple, Any, Optional

# Folder with the Waymo data (see data/README.md). Set WAYMO_DATA_DIR to use another folder.
DATA_DIR = os.environ.get('WAYMO_DATA_DIR', 'data')

# Path definitions
MOTION_VALIDATION_PATH = os.path.join(DATA_DIR, 'motion/tf_example/validation_interactive')
REASONING_VALIDATION_PATH = os.path.join(DATA_DIR, 'reasoning/validation_interactive')
OUTPUT_DIR = 'outputs/combined_features'

# Create output directory if it doesn't exist
os.makedirs(OUTPUT_DIR, exist_ok=True)

class MotionReasoningPipeline:
    """Pipeline for combined processing of Waymo Motion and Reasoning datasets."""
    
    def __init__(self, motion_path=MOTION_VALIDATION_PATH, reasoning_path=REASONING_VALIDATION_PATH, output_dir=OUTPUT_DIR):
        """Initialize the pipeline."""
        self.motion_path = motion_path
        self.reasoning_path = reasoning_path
        self.output_dir = output_dir
        
        # Load common scenario IDs
        self.common_scenario_ids = self.load_common_scenario_ids()
        
        # Initialize data structures
        self.motion_features = {}
        self.reasoning_features = {}
        self.combined_features = {}
    
    def load_common_scenario_ids(self):
        """Load the list of common scenario IDs between motion and reasoning datasets.
        
        If the comparison file doesn't exist, generate it.
        """
        comparison_file = os.path.join('outputs/validation_analysis', 'scenario_id_comparison.json')
        
        if os.path.exists(comparison_file):
            try:
                with open(comparison_file, 'r') as f:
                    comparison = json.load(f)
                common_ids = comparison.get('common_ids', [])
                print(f"Loaded {len(common_ids)} common scenario IDs.")
                return common_ids
            except Exception as e:
                print(f"Error loading scenario ID comparison: {str(e)}")
                
        # If file doesn't exist or couldn't be loaded, generate it
        print("Generating common scenario IDs...")
        motion_ids = self.extract_motion_scenario_ids()
        reasoning_ids = self.extract_reasoning_scenario_ids()
        
        common_ids = list(set(motion_ids).intersection(set(reasoning_ids)))
        print(f"Found {len(common_ids)} common scenario IDs.")
        
        # Save to file for later use
        os.makedirs(os.path.dirname(comparison_file), exist_ok=True)
        with open(comparison_file, 'w') as f:
            json.dump({
                'common_ids': common_ids,
                'motion_only_ids': list(set(motion_ids) - set(reasoning_ids)),
                'reasoning_only_ids': list(set(reasoning_ids) - set(motion_ids))
            }, f, indent=2)
        
        return common_ids
    
    def extract_motion_scenario_ids(self):
        """Extract all scenario IDs from the motion dataset."""
        print(f"Extracting scenario IDs from motion data at {self.motion_path}...")
        
        tfrecord_files = [os.path.join(self.motion_path, f) for f in os.listdir(self.motion_path) 
                         if '.tfrecord' in f]
        
        if not tfrecord_files:
            print("No motion data TFRecord files found.")
            return []
        
        scenario_ids = []
        
        for file_idx, tfrecord_file in enumerate(tfrecord_files):
            print(f"Processing motion file {file_idx+1}/{len(tfrecord_files)}: {os.path.basename(tfrecord_file)}")
            
            raw_dataset = tf.data.TFRecordDataset(tfrecord_file)
            file_scenario_ids = []
            
            for raw_record in tqdm(raw_dataset, desc=f"Extracting IDs from {os.path.basename(tfrecord_file)}"):
                try:
                    example = tf.train.Example()
                    example.ParseFromString(raw_record.numpy())
                    
                    scenario_id = example.features.feature['scenario/id'].bytes_list.value[0].decode('utf-8')
                    file_scenario_ids.append(scenario_id)
                    
                except Exception as e:
                    print(f"Error processing motion record: {str(e)}")
                    continue
            
            print(f"Found {len(file_scenario_ids)} scenario IDs in motion file {file_idx+1}")
            scenario_ids.extend(file_scenario_ids)
        
        # Remove duplicates
        unique_ids = list(dict.fromkeys(scenario_ids))
        print(f"Found {len(unique_ids)} unique scenario IDs in motion data.")
        
        return unique_ids
    
    def extract_reasoning_scenario_ids(self):
        """Extract all scenario IDs from the reasoning dataset."""
        print(f"\nExtracting scenario IDs from reasoning data at {self.reasoning_path}...")
        
        json_files = [os.path.join(self.reasoning_path, f) for f in os.listdir(self.reasoning_path) 
                     if f.endswith('.json')]
        
        if not json_files:
            print("No reasoning data JSON files found.")
            return []
            
        scenario_ids = []
        
        for file_path in tqdm(json_files, desc="Processing reasoning files"):
            try:
                with open(file_path, 'r') as f:
                    data = json.load(f)
                
                scenario_id = data['sid']
                scenario_ids.append(scenario_id)
                
            except Exception as e:
                print(f"Error processing reasoning file {file_path}: {str(e)}")
                continue
        
        # Remove duplicates
        unique_ids = list(dict.fromkeys(scenario_ids))
        print(f"Found {len(unique_ids)} unique scenario IDs in reasoning data.")
        
        return unique_ids
    
    def extract_motion_features(self, scenario_id):
        """Extract features from motion data for a specific scenario."""
        print(f"Extracting motion features for scenario {scenario_id}...")
        
        tfrecord_files = [os.path.join(self.motion_path, f) for f in os.listdir(self.motion_path) 
                         if '.tfrecord' in f]
        
        for tfrecord_file in tfrecord_files:
            raw_dataset = tf.data.TFRecordDataset(tfrecord_file)
            
            for raw_record in tqdm(raw_dataset, desc=f"Searching in {os.path.basename(tfrecord_file)}"):
                try:
                    example = tf.train.Example()
                    example.ParseFromString(raw_record.numpy())
                    
                    record_scenario_id = example.features.feature['scenario/id'].bytes_list.value[0].decode('utf-8')
                    
                    if record_scenario_id == scenario_id:
                        print(f"Found scenario {scenario_id} in {os.path.basename(tfrecord_file)}")
                        
                        # Extract features
                        features = {}
                        features['scenario_id'] = scenario_id
                        
                        # Helper function to extract feature arrays
                        def extract_feature(name, default=None):
                            if name not in example.features.feature:
                                return default
                                
                            feature = example.features.feature[name]
                            
                            # Int features
                            if feature.int64_list.value:
                                return np.array(feature.int64_list.value)
                            
                            # Float features
                            if feature.float_list.value:
                                return np.array(feature.float_list.value)
                                
                            # Bytes features
                            if feature.bytes_list.value:
                                return [v.decode('utf-8') for v in feature.bytes_list.value]
                                
                            return default
                        
                        # Basic scenario features
                        features['agent_ids'] = extract_feature('state/id')
                        features['agent_types'] = extract_feature('state/type')
                        features['is_sdc'] = extract_feature('state/is_sdc')
                        
                        # Current state features
                        features['current_x'] = extract_feature('state/current/x')
                        features['current_y'] = extract_feature('state/current/y')
                        features['current_velocity_x'] = extract_feature('state/current/velocity_x')
                        features['current_velocity_y'] = extract_feature('state/current/velocity_y')
                        features['current_speed'] = extract_feature('state/current/speed')
                        features['current_bbox_yaw'] = extract_feature('state/current/bbox_yaw')
                        features['current_valid'] = extract_feature('state/current/valid')
                        
                        # Find ego agent index
                        ego_index = np.where(features['is_sdc'] == 1)[0][0] if np.any(features['is_sdc'] == 1) else None
                        
                        if ego_index is not None:
                            features['ego_agent_id'] = int(features['agent_ids'][ego_index])
                            features['ego_current_x'] = float(features['current_x'][ego_index])
                            features['ego_current_y'] = float(features['current_y'][ego_index])
                            features['ego_current_speed'] = float(features['current_speed'][ego_index])
                            features['ego_current_heading'] = float(features['current_bbox_yaw'][ego_index])
                        
                        return features
                except Exception as e:
                    print(f"Error processing motion record: {str(e)}")
                    continue
        
        print(f"No features found for scenario {scenario_id} in motion data.")
        return None
    
    def extract_reasoning_features(self, scenario_id):
        """Extract features from reasoning data for a specific scenario."""
        print(f"Extracting reasoning features for scenario {scenario_id}...")
        
        # Look for files matching this scenario ID
        matching_files = [os.path.join(self.reasoning_path, f) 
                         for f in os.listdir(self.reasoning_path)
                         if f.startswith(f"scid_{scenario_id}__")]
        
        if not matching_files:
            print(f"No reasoning files found for scenario {scenario_id}.")
            return None
        
        # Load each matching file
        scenario_features = []
        
        for file_path in matching_files:
            try:
                with open(file_path, 'r') as f:
                    data = json.load(f)
                
                # Extract basic features
                features = {
                    'scenario_id': data['sid'],
                    'ego_agent_id': data['ego'],
                    'source_file': os.path.basename(file_path),
                    'cur_time': data['cur_time'],
                    'future_time': data['future_time'],
                }
                
                # Count questions by category
                features['env_question_count'] = len(data.get('env_q', []))
                features['ego_question_count'] = len(data.get('ego_q', []))
                features['sur_question_count'] = len(data.get('sur_q', []))
                features['int_question_count'] = len(data.get('int_q', []))
                
                # Check for interaction keywords
                interaction_keywords = ['yield', 'stop', 'continue', 'accelerate', 'decelerate', 
                                       'turn left', 'turn right', 'merge', 'change lane']
                
                # Initialize keyword flags
                for keyword in interaction_keywords:
                    features[f'has_{keyword.replace(" ", "_")}'] = False
                
                # Check interaction answers for keywords
                for answer in data.get('int_a', []):
                    answer_lower = answer.lower()
                    for keyword in interaction_keywords:
                        if keyword in answer_lower:
                            features[f'has_{keyword.replace(" ", "_")}'] = True
                
                scenario_features.append(features)
                
            except Exception as e:
                print(f"Error processing reasoning file {file_path}: {str(e)}")
                continue
        
        if not scenario_features:
            print(f"No features extracted from reasoning data for scenario {scenario_id}.")
            return None
        
        # Combine features from all files for this scenario (typically just one file per scenario)
        combined = scenario_features[0]
        combined['file_count'] = len(scenario_features)
        
        return combined
    
    def combine_features(self, motion_features, reasoning_features):
        """Combine features from motion and reasoning data."""
        if not motion_features or not reasoning_features:
            return None
        
        # Check if ego agent IDs match
        motion_ego_id = motion_features.get('ego_agent_id')
        reasoning_ego_id = reasoning_features.get('ego_agent_id')
        
        if motion_ego_id is None or reasoning_ego_id is None:
            print(f"Warning: Missing ego agent ID in one or both datasets.")
            return None
        
        if motion_ego_id != reasoning_ego_id:
            print(f"Warning: Ego agent IDs do not match: Motion={motion_ego_id}, Reasoning={reasoning_ego_id}")
            return None
        
        # Combine features
        combined = {
            'scenario_id': motion_features['scenario_id'],
            'ego_agent_id': motion_ego_id,
            'motion_file': '',  # TFRecord doesn't have a specific file per scenario
            'reasoning_file': reasoning_features.get('source_file', ''),
            
            # Motion features
            'agent_count': len(motion_features['agent_ids']),
            'ego_x': motion_features.get('ego_current_x'),
            'ego_y': motion_features.get('ego_current_y'),
            'ego_speed': motion_features.get('ego_current_speed'),
            'ego_heading': motion_features.get('ego_current_heading'),
            
            # Reasoning features
            'cur_time': reasoning_features.get('cur_time'),
            'future_time': reasoning_features.get('future_time'),
            'env_questions': reasoning_features.get('env_question_count'),
            'ego_questions': reasoning_features.get('ego_question_count'),
            'sur_questions': reasoning_features.get('sur_question_count'),
            'int_questions': reasoning_features.get('int_question_count'),
        }
        
        # Add interaction flags
        for key in reasoning_features:
            if key.startswith('has_'):
                combined[key] = reasoning_features[key]
        
        return combined
    
    def process_scenario(self, scenario_id):
        """Process a single scenario from both datasets and combine features."""
        # Extract features from motion data
        motion_features = self.extract_motion_features(scenario_id)
        if motion_features:
            self.motion_features[scenario_id] = motion_features
        
        # Extract features from reasoning data
        reasoning_features = self.extract_reasoning_features(scenario_id)
        if reasoning_features:
            self.reasoning_features[scenario_id] = reasoning_features
        
        # Combine features if both are available
        if motion_features and reasoning_features:
            combined = self.combine_features(motion_features, reasoning_features)
            if combined:
                self.combined_features[scenario_id] = combined
                return True
        
        return False
    
    def process_common_scenarios(self, max_scenarios=None):
        """Process all common scenarios and extract features."""
        print(f"Processing common scenarios...")
        
        # Limit the number of scenarios if specified
        scenarios_to_process = self.common_scenario_ids[:max_scenarios] if max_scenarios else self.common_scenario_ids
        
        print(f"Will process {len(scenarios_to_process)} scenarios.")
        
        successful = 0
        for i, scenario_id in enumerate(scenarios_to_process):
            print(f"\nProcessing scenario {i+1}/{len(scenarios_to_process)}: {scenario_id}")
            if self.process_scenario(scenario_id):
                successful += 1
        
        print(f"\nProcessed {len(scenarios_to_process)} scenarios, {successful} successful.")
        
        # Convert combined features to DataFrame
        if self.combined_features:
            self.combined_df = pd.DataFrame(list(self.combined_features.values()))
            print(f"Created combined features DataFrame with {len(self.combined_df)} rows.")
            
            # Save to CSV
            output_file = os.path.join(self.output_dir, 'combined_features.csv')
            self.combined_df.to_csv(output_file, index=False)
            print(f"Saved combined features to {output_file}")
            
            return self.combined_df
        else:
            print("No combined features generated.")
            return None
    
    def analyze_combined_features(self):
        """Analyze the combined features dataset."""
        if not hasattr(self, 'combined_df') or self.combined_df.empty:
            print("No combined features to analyze.")
            return
        
        print("\nAnalyzing combined features...")
        df = self.combined_df
        
        # Basic statistics
        print(f"Total scenarios: {len(df)}")
        print(f"Average agent count: {df['agent_count'].mean():.2f}")
        print(f"Average ego speed: {df['ego_speed'].mean():.2f} m/s")
        
        # Question statistics
        print("\nQuestion counts:")
        print(f"  Environment questions: {df['env_questions'].mean():.2f} (mean)")
        print(f"  Ego questions: {df['ego_questions'].mean():.2f} (mean)")
        print(f"  Surrounding questions: {df['sur_questions'].mean():.2f} (mean)")
        print(f"  Interaction questions: {df['int_questions'].mean():.2f} (mean)")
        
        # Interaction pattern statistics
        interaction_cols = [col for col in df.columns if col.startswith('has_')]
        print("\nInteraction patterns:")
        for col in interaction_cols:
            pattern = col.replace('has_', '').replace('_', ' ')
            count = df[col].sum()
            pct = count / len(df) * 100
            print(f"  {pattern}: {count} scenarios ({pct:.1f}%)")
        
        # Generate visualizations
        self.visualize_combined_features(df)
    
    def visualize_combined_features(self, df):
        """Generate visualizations for the combined features."""
        print("\nGenerating visualizations...")
        
        # Create output directory for visualizations
        vis_dir = os.path.join(self.output_dir, 'visualizations')
        os.makedirs(vis_dir, exist_ok=True)
        
        # 1. Ego speed distribution
        plt.figure(figsize=(10, 6))
        plt.hist(df['ego_speed'], bins=20, color='blue', alpha=0.7)
        plt.xlabel('Ego Speed (m/s)')
        plt.ylabel('Frequency')
        plt.title('Distribution of Ego Agent Speeds')
        plt.grid(True, alpha=0.3)
        plt.savefig(os.path.join(vis_dir, 'ego_speed_distribution.png'))
        plt.close()
        
        # 2. Question counts by category
        categories = ['env_questions', 'ego_questions', 'sur_questions', 'int_questions']
        category_labels = ['Environment', 'Ego Agent', 'Surrounding Agents', 'Interaction']
        means = [df[cat].mean() for cat in categories]
        
        plt.figure(figsize=(12, 6))
        bars = plt.bar(category_labels, means, color=['#3498db', '#2ecc71', '#e74c3c', '#f39c12'])
        
        # Add value labels
        for bar in bars:
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                    f'{height:.2f}', ha='center', va='bottom')
        
        plt.title('Average Number of Questions per Category')
        plt.ylabel('Average Number of Questions')
        plt.grid(True, axis='y', alpha=0.3)
        plt.savefig(os.path.join(vis_dir, 'question_category_distribution.png'))
        plt.close()
        
        # 3. Interaction pattern frequency
        interaction_cols = [col for col in df.columns if col.startswith('has_')]
        patterns = [col.replace('has_', '').replace('_', ' ') for col in interaction_cols]
        counts = [df[col].sum() for col in interaction_cols]
        
        # Sort by frequency
        sorted_indices = np.argsort(counts)[::-1]
        patterns = [patterns[i] for i in sorted_indices]
        counts = [counts[i] for i in sorted_indices]
        
        plt.figure(figsize=(14, 8))
        bars = plt.bar(patterns, counts, color='#9b59b6')
        
        # Add value labels
        for bar in bars:
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height + 0.3,
                    f'{height}', ha='center', va='bottom')
        
        plt.title('Frequency of Interaction Patterns')
        plt.ylabel('Number of Scenarios')
        plt.xticks(rotation=45, ha='right')
        plt.grid(True, axis='y', alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(vis_dir, 'interaction_pattern_frequency.png'))
        plt.close()
        
        print(f"Visualizations saved to {vis_dir}")

def main(max_scenarios=10):
    """Main function to run the combined feature pipeline."""
    print("Starting Combined Motion-Reasoning Feature Pipeline")
    
    # Create pipeline
    pipeline = MotionReasoningPipeline()
    
    # Process common scenarios
    combined_df = pipeline.process_common_scenarios(max_scenarios=max_scenarios)
    
    # Analyze combined features
    if combined_df is not None and not combined_df.empty:
        pipeline.analyze_combined_features()
    
    print("Pipeline complete!")

if __name__ == "__main__":
    # Process 10 scenarios by default, change this to process more
    main(max_scenarios=10)