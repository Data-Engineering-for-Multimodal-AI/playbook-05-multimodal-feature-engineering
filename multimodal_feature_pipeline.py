import tensorflow as tf
import numpy as np
import pandas as pd
import json
import os
import matplotlib.pyplot as plt
from tqdm import tqdm
from collections import defaultdict
from sentence_transformers import SentenceTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.preprocessing import StandardScaler
import umap
import seaborn as sns
from typing import Dict, List, Tuple, Any, Optional, Union

# Folder with the Waymo data (see data/README.md). Set WAYMO_DATA_DIR to use another folder.
DATA_DIR = os.environ.get('WAYMO_DATA_DIR', 'data')

# Path definitions
MOTION_VALIDATION_PATH = os.path.join(DATA_DIR, 'motion/tf_example/validation_interactive')
REASONING_VALIDATION_PATH = os.path.join(DATA_DIR, 'reasoning/validation_interactive')
OUTPUT_DIR = 'outputs/multimodal_features'
FEATURE_STORE_DIR = 'outputs/feature_store'

# Create output directories
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(FEATURE_STORE_DIR, exist_ok=True)
os.makedirs(os.path.join(OUTPUT_DIR, 'visualizations'), exist_ok=True)

class MultimodalFeaturePipeline:
    """Pipeline for multimodal feature engineering with Waymo Motion and Reasoning datasets."""
    
    def __init__(self, 
                 motion_path=MOTION_VALIDATION_PATH, 
                 reasoning_path=REASONING_VALIDATION_PATH, 
                 output_dir=OUTPUT_DIR,
                 feature_store_dir=FEATURE_STORE_DIR,
                 use_pretrained_model=True):
        """Initialize the pipeline."""
        self.motion_path = motion_path
        self.reasoning_path = reasoning_path
        self.output_dir = output_dir
        self.feature_store_dir = feature_store_dir
        
        # Load common scenario IDs
        self.common_scenario_ids = self.load_common_scenario_ids()
        
        # Initialize text embedding model
        self.text_model = None
        if use_pretrained_model:
            try:
                print("Loading Sentence-BERT model...")
                self.text_model = SentenceTransformer('all-mpnet-base-v2')
                print("Sentence-BERT model loaded successfully")
            except Exception as e:
                print(f"Error loading Sentence-BERT model: {str(e)}")
                print("Will run without text embeddings")
        
        # Initialize data structures
        self.motion_features = {}
        self.reasoning_features = {}
        self.text_embeddings = {}
        self.fused_features = {}
    
    def load_common_scenario_ids(self):
        """Load the list of common scenario IDs between motion and reasoning datasets."""
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
        """Extract rich motion features from TFRecord data for a specific scenario."""
        print(f"Extracting motion features for scenario {scenario_id}...")
        
        tfrecord_files = [os.path.join(self.motion_path, f) for f in os.listdir(self.motion_path) 
                         if '.tfrecord' in f]
        
        for tfrecord_file in tfrecord_files:
            raw_dataset = tf.data.TFRecordDataset(tfrecord_file)
            
            for i, raw_record in enumerate(tqdm(raw_dataset, desc=f"Searching in {os.path.basename(tfrecord_file)}")):
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
                        
                        # Try to extract trajectory features if available
                        features['future_x'] = extract_feature('state/future/x')
                        features['future_y'] = extract_feature('state/future/y')
                        features['future_velocity_x'] = extract_feature('state/future/velocity_x')
                        features['future_velocity_y'] = extract_feature('state/future/velocity_y')
                        features['future_speed'] = extract_feature('state/future/speed')
                        features['future_valid'] = extract_feature('state/future/valid')
                        
                        # Calculate per-agent features
                        num_agents = len(features['agent_ids'])
                        agent_features = []
                        
                        for agent_idx in range(num_agents):
                            if not features['current_valid'][agent_idx]:
                                continue
                                
                            agent_id = int(features['agent_ids'][agent_idx])
                            
                            agent_data = {
                                'scenario_id': scenario_id,
                                'agent_id': agent_id,
                                'agent_type': int(features['agent_types'][agent_idx]) if 'agent_types' in features else None,
                                'is_sdc': bool(features['is_sdc'][agent_idx]) if 'is_sdc' in features else False,
                                'current_x': float(features['current_x'][agent_idx]),
                                'current_y': float(features['current_y'][agent_idx]),
                                'current_speed': float(features['current_speed'][agent_idx]),
                                'current_heading': float(features['current_bbox_yaw'][agent_idx]) if 'current_bbox_yaw' in features else None,
                            }
                            
                            # Add future trajectory if available
                            if 'future_speed' in features and features['future_speed'] is not None:
                                # Check if we can reshape the array properly
                                if isinstance(features['future_speed'], np.ndarray):
                                    try:
                                        # Calculate average future speed and heading change
                                        future_speeds = features['future_speed'].reshape(-1, num_agents)[:, agent_idx]
                                        agent_data['avg_future_speed'] = float(np.mean(future_speeds[features['future_valid'].reshape(-1, num_agents)[:, agent_idx]]))
                                        
                                        # Check if agent is yielding (significant speed reduction)
                                        agent_data['is_yielding'] = self.is_yielding(future_speeds)
                                    except (ValueError, IndexError) as e:
                                        # If reshaping fails, skip these features
                                        print(f"Error processing future data for agent {agent_id}: {str(e)}")
                            
                            # Calculate distance to ego if this is not the ego
                            if not agent_data['is_sdc']:
                                ego_idx = np.where(features['is_sdc'] == 1)[0]
                                if len(ego_idx) > 0:
                                    ego_idx = ego_idx[0]
                                    ego_x = features['current_x'][ego_idx]
                                    ego_y = features['current_y'][ego_idx]
                                    agent_data['distance_to_ego'] = self.compute_distance(
                                        agent_data['current_x'], 
                                        agent_data['current_y'], 
                                        ego_x, 
                                        ego_y
                                    )
                            
                            agent_features.append(agent_data)
                        
                        return agent_features
                        
                except Exception as e:
                    if i % 100 == 0:  # Only print occasional errors to avoid spam
                        print(f"Error processing motion record: {str(e)}")
                    continue
        
        print(f"No features found for scenario {scenario_id} in motion data.")
        return None
    
    def extract_reasoning_features(self, scenario_id):
        """Extract reasoning features and Q&A text from reasoning data for a specific scenario."""
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
                
                # Extract Q&A text for embedding
                qa_texts = []
                
                # Environment Q&A pairs
                for q, a in zip(data.get('env_q', []), data.get('env_a', [])):
                    qa_texts.append(f"Environment: {q} {a}")
                
                # Ego agent Q&A pairs
                for q, a in zip(data.get('ego_q', []), data.get('ego_a', [])):
                    qa_texts.append(f"Ego: {q} {a}")
                
                # Surrounding agent Q&A pairs
                for q, a in zip(data.get('sur_q', []), data.get('sur_a', [])):
                    qa_texts.append(f"Surrounding: {q} {a}")
                
                # Interaction Q&A pairs
                for q, a in zip(data.get('int_q', []), data.get('int_a', [])):
                    qa_texts.append(f"Interaction: {q} {a}")
                
                # Store Q&A texts
                features['qa_texts'] = qa_texts
                
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
    
    def embed_qa_text(self, reasoning_features):
        """Generate embeddings for Q&A text using Sentence-BERT."""
        if self.text_model is None or not reasoning_features or 'qa_texts' not in reasoning_features:
            return None
            
        qa_texts = reasoning_features['qa_texts']
        if not qa_texts:
            return None
            
        # Select a subset of Q&A pairs (at most 10) to keep embedding size manageable
        selected_texts = qa_texts[:min(10, len(qa_texts))]
        combined_text = " ".join(selected_texts)
        
        # Generate embedding
        try:
            embedding = self.text_model.encode(combined_text)
            return embedding
        except Exception as e:
            print(f"Error generating text embedding: {str(e)}")
            return None
    
    def compute_distance(self, x1, y1, x2, y2):
        """Compute Euclidean distance between two points."""
        return np.sqrt((x2 - x1)**2 + (y2 - y1)**2)
    
    def is_yielding(self, future_speeds, threshold=2.0):
        """Determine if an agent is yielding based on future speed profile.
        
        Args:
            future_speeds: Array of future speeds
            threshold: Speed threshold (m/s) below which agent is considered yielding
            
        Returns:
            True if agent's speed drops below threshold in the future trajectory
        """
        if future_speeds is None or len(future_speeds) == 0:
            return False
            
        # Check if speed drops significantly in the near future
        current_speed = future_speeds[0] if len(future_speeds) > 0 else 0
        min_future_speed = np.min(future_speeds) if len(future_speeds) > 0 else 0
        
        # Agent is yielding if it slows down significantly
        return current_speed > threshold and min_future_speed < threshold
    
    def fuse_features(self, motion_agent_features, reasoning_features, text_embedding):
        """Fuse motion and reasoning features with text embeddings."""
        if not motion_agent_features or not reasoning_features:
            return None
            
        # Find the ego agent in motion features
        ego_agent_feature = None
        for agent_feature in motion_agent_features:
            if agent_feature['is_sdc']:
                ego_agent_feature = agent_feature
                break
                
        if ego_agent_feature is None:
            print(f"No ego agent found in motion features.")
            return None
            
        # Check if ego agent IDs match
        motion_ego_id = ego_agent_feature['agent_id']
        reasoning_ego_id = reasoning_features.get('ego_agent_id')
        
        if motion_ego_id != reasoning_ego_id:
            print(f"Warning: Ego agent IDs do not match: Motion={motion_ego_id}, Reasoning={reasoning_ego_id}")
            return None
            
        # Create fused features for all agents in the scene
        fused_agents = []
        
        for agent_feature in motion_agent_features:
            agent_id = agent_feature['agent_id']
            
            # Base fused features
            fused = {
                'scenario_id': agent_feature['scenario_id'],
                'agent_id': agent_id,
                'is_ego': agent_feature['is_sdc'],
                'agent_type': agent_feature['agent_type'],
                
                # Motion features
                'x': agent_feature['current_x'],
                'y': agent_feature['current_y'],
                'speed': agent_feature['current_speed'],
                'heading': agent_feature['current_heading'],
                
                # Add distance to ego if not ego
                'distance_to_ego': agent_feature.get('distance_to_ego', 0.0) if not agent_feature['is_sdc'] else 0.0,
                
                # Future behavior if available
                'avg_future_speed': agent_feature.get('avg_future_speed', None),
                'is_yielding': agent_feature.get('is_yielding', False),
                
                # Reasoning metadata
                'cur_time': reasoning_features.get('cur_time'),
                'future_time': reasoning_features.get('future_time'),
                'env_question_count': reasoning_features.get('env_question_count', 0),
                'ego_question_count': reasoning_features.get('ego_question_count', 0),
                'sur_question_count': reasoning_features.get('sur_question_count', 0),
                'int_question_count': reasoning_features.get('int_question_count', 0),
                
                # Modality flags
                'has_motion': True,
                'has_reasoning': True,
                'has_text_embedding': text_embedding is not None,
            }
            
            # Add interaction flags if this is the ego agent
            if agent_feature['is_sdc']:
                for key in reasoning_features:
                    if key.startswith('has_'):
                        fused[key] = reasoning_features[key]
            
            # Add text embedding if available
            if text_embedding is not None:
                # Convert embedding to a dictionary with numbered keys
                for i, value in enumerate(text_embedding):
                    fused[f'text_embedding_{i}'] = float(value)
                
                # Also store embedding dimension for reference
                fused['text_embedding_dim'] = len(text_embedding)
            
            fused_agents.append(fused)
        
        return fused_agents
    
    def process_scenario(self, scenario_id):
        """Process a single scenario from both datasets and create fused features."""
        # Extract motion features
        motion_agent_features = self.extract_motion_features(scenario_id)
        if motion_agent_features:
            self.motion_features[scenario_id] = motion_agent_features
        else:
            return False
        
        # Extract reasoning features
        reasoning_features = self.extract_reasoning_features(scenario_id)
        if reasoning_features:
            self.reasoning_features[scenario_id] = reasoning_features
        else:
            return False
        
        # Generate text embedding
        text_embedding = None
        if self.text_model is not None and reasoning_features:
            text_embedding = self.embed_qa_text(reasoning_features)
            if text_embedding is not None:
                self.text_embeddings[scenario_id] = text_embedding
        
        # Fuse features
        fused_agents = self.fuse_features(motion_agent_features, reasoning_features, text_embedding)
        if fused_agents:
            self.fused_features[scenario_id] = fused_agents
            return True
        
        return False
    
    def process_scenarios(self, max_scenarios=None):
        """Process multiple scenarios from both datasets and create fused features."""
        print(f"Processing scenarios...")
        
        # Limit the number of scenarios if specified
        scenarios_to_process = self.common_scenario_ids[:max_scenarios] if max_scenarios else self.common_scenario_ids
        
        print(f"Will process {len(scenarios_to_process)} scenarios.")
        
        successful = 0
        for i, scenario_id in enumerate(scenarios_to_process):
            print(f"\nProcessing scenario {i+1}/{len(scenarios_to_process)}: {scenario_id}")
            if self.process_scenario(scenario_id):
                successful += 1
        
        print(f"\nProcessed {len(scenarios_to_process)} scenarios, {successful} successful.")
        
        # Convert fused features to DataFrame
        if self.fused_features:
            # Flatten the nested structure
            all_agents = []
            for scenario_id, agent_list in self.fused_features.items():
                all_agents.extend(agent_list)
            
            self.fused_df = pd.DataFrame(all_agents)
            print(f"Created fused features DataFrame with {len(self.fused_df)} agent entries.")
            
            # Save to CSV
            output_file = os.path.join(self.output_dir, 'fused_features.csv')
            self.fused_df.to_csv(output_file, index=False)
            print(f"Saved fused features to {output_file}")
            
            return self.fused_df
        else:
            print("No fused features generated.")
            return None
    
    def upload_to_feature_store(self):
        """Simulate uploading features to Vertex AI Feature Store.
        
        In a production system, this would use the Vertex AI Feature Store API.
        For this example, we simply save organized feature files.
        """
        if not hasattr(self, 'fused_df') or self.fused_df.empty:
            print("No fused features to upload to feature store.")
            return
        
        print("\nUploading features to simulated feature store...")
        
        # Create feature store directory structure
        entity_dir = os.path.join(self.feature_store_dir, 'agent_features')
        os.makedirs(entity_dir, exist_ok=True)
        
        # Group by scenario_id and agent_id
        grouped = self.fused_df.groupby(['scenario_id', 'agent_id'])
        
        # Save each agent's features as a separate file
        for (scenario_id, agent_id), group in tqdm(grouped, desc="Uploading to feature store"):
            feature_file = os.path.join(entity_dir, f"{scenario_id}__{agent_id}.json")
            
            # Convert to dictionary
            feature_dict = group.iloc[0].to_dict()
            
            # Extract text embedding if present
            if 'text_embedding_dim' in feature_dict:
                embedding_dim = int(feature_dict['text_embedding_dim'])
                embedding = [feature_dict[f'text_embedding_{i}'] for i in range(embedding_dim)]
                
                # Add embedding as a separate key
                feature_dict['text_embedding'] = embedding
                
                # Remove individual embedding components to clean up
                for i in range(embedding_dim):
                    del feature_dict[f'text_embedding_{i}']
            
            # Save to JSON
            with open(feature_file, 'w') as f:
                json.dump(feature_dict, f, indent=2)
        
        print(f"Uploaded {len(grouped)} agent features to feature store at {entity_dir}")
        
        # Create feature store metadata
        metadata = {
            'entity_type': 'agent',
            'feature_count': len(self.fused_df.columns),
            'agent_count': len(grouped),
            'scenario_count': len(self.fused_df['scenario_id'].unique()),
            'creation_time': pd.Timestamp.now().isoformat(),
            'feature_names': list(self.fused_df.columns)
        }
        
        # Save metadata
        with open(os.path.join(self.feature_store_dir, 'metadata.json'), 'w') as f:
            json.dump(metadata, f, indent=2)
        
        print("Feature store metadata saved.")
    
    def train_yield_prediction_model(self):
        """Train a simple yield prediction model using the fused features."""
        if not hasattr(self, 'fused_df') or self.fused_df.empty:
            print("No fused features available for model training.")
            return None
        
        print("\nTraining yield prediction model...")
        
        # Prepare dataset for yield prediction
        df = self.fused_df.copy()
        
        # Filter for agents that have future trajectory information
        df = df.dropna(subset=['avg_future_speed', 'is_yielding'])
        
        if len(df) == 0:
            print("No samples with valid future trajectory information.")
            return None
            
        print(f"Using {len(df)} samples for yield prediction model.")
        
        # Select features for the model
        feature_cols = ['speed', 'distance_to_ego']
        
        # Add reasoning question counts
        feature_cols.extend(['env_question_count', 'ego_question_count', 
                           'sur_question_count', 'int_question_count'])
        
        # Add text embedding features if available
        if 'text_embedding_dim' in df.columns:
            embedding_dim = df['text_embedding_dim'].iloc[0]
            embedding_cols = [f'text_embedding_{i}' for i in range(int(embedding_dim))]
            filtered_embedding_cols = [col for col in embedding_cols if col in df.columns]
            
            if filtered_embedding_cols:
                print(f"Adding {len(filtered_embedding_cols)} text embedding dimensions to model features.")
                feature_cols.extend(filtered_embedding_cols)
        
        # Prepare features and target
        X = df[feature_cols].fillna(0)
        y = df['is_yielding']
        
        # Split into train and test sets
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
        
        # Scale features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # Train logistic regression model
        model = LogisticRegression(max_iter=1000, class_weight='balanced')
        model.fit(X_train_scaled, y_train)
        
        # Evaluate model
        train_score = model.score(X_train_scaled, y_train)
        test_score = model.score(X_test_scaled, y_test)
        
        print(f"Model training complete.")
        print(f"Train accuracy: {train_score:.4f}")
        print(f"Test accuracy: {test_score:.4f}")
        
        # Detailed classification report
        y_pred = model.predict(X_test_scaled)
        report = classification_report(y_test, y_pred)
        print("\nClassification Report:")
        print(report)
        
        # Save model results
        results = {
            'train_accuracy': train_score,
            'test_accuracy': test_score,
            'feature_importance': dict(zip(feature_cols, model.coef_[0])),
            'classification_report': report
        }
        
        # Save model results to file
        with open(os.path.join(self.output_dir, 'yield_model_results.json'), 'w') as f:
            # Convert numpy values to Python native types
            results_serializable = {
                'train_accuracy': float(train_score),
                'test_accuracy': float(test_score),
                'feature_importance': {k: float(v) for k, v in results['feature_importance'].items()},
                'classification_report': report
            }
            json.dump(results_serializable, f, indent=2)
        
        # Visualize feature importance
        self.visualize_feature_importance(feature_cols, model.coef_[0])
        
        # If text embeddings are present, visualize them
        if filtered_embedding_cols and len(filtered_embedding_cols) > 10:
            self.visualize_text_embeddings(df, filtered_embedding_cols, y)
        
        return model, results
    
    def visualize_feature_importance(self, feature_names, coefficients):
        """Visualize feature importance for the yield prediction model."""
        # Create output directory
        vis_dir = os.path.join(self.output_dir, 'visualizations')
        os.makedirs(vis_dir, exist_ok=True)
        
        # Sort features by absolute importance
        importance = np.abs(coefficients)
        indices = np.argsort(importance)[::-1]
        
        # Select top 15 features if more than 15
        if len(feature_names) > 15:
            indices = indices[:15]
            
        names = [feature_names[i] for i in indices]
        coefs = [coefficients[i] for i in indices]
        
        # Plot feature importance
        plt.figure(figsize=(12, 8))
        colors = ['blue' if c > 0 else 'red' for c in coefs]
        plt.barh(range(len(names)), coefs, color=colors)
        plt.yticks(range(len(names)), names)
        plt.xlabel('Coefficient')
        plt.title('Feature Importance for Yield Prediction')
        plt.grid(axis='x', linestyle='--', alpha=0.6)
        plt.tight_layout()
        plt.savefig(os.path.join(vis_dir, 'feature_importance.png'))
        plt.close()
    
    def visualize_text_embeddings(self, df, embedding_cols, labels):
        """Visualize text embeddings using UMAP dimensionality reduction."""
        try:
            # Create output directory
            vis_dir = os.path.join(self.output_dir, 'visualizations')
            os.makedirs(vis_dir, exist_ok=True)
            
            # Extract embeddings
            embeddings = df[embedding_cols].values
            
            # Apply UMAP dimensionality reduction
            reducer = umap.UMAP(n_components=2, random_state=42)
            embedding_2d = reducer.fit_transform(embeddings)
            
            # Create DataFrame for visualization
            vis_df = pd.DataFrame({
                'umap_1': embedding_2d[:, 0],
                'umap_2': embedding_2d[:, 1],
                'is_yielding': labels.values
            })
            
            # Plot embeddings
            plt.figure(figsize=(10, 8))
            sns.scatterplot(
                data=vis_df, 
                x='umap_1', 
                y='umap_2', 
                hue='is_yielding',
                palette=['#ff7f0e', '#1f77b4'],
                alpha=0.8
            )
            plt.title('UMAP Visualization of Text Embeddings')
            plt.legend(title='Is Yielding')
            plt.tight_layout()
            plt.savefig(os.path.join(vis_dir, 'text_embedding_umap.png'))
            plt.close()
            
            print("Text embedding visualization saved.")
            
        except Exception as e:
            print(f"Error creating text embedding visualization: {str(e)}")

def main(max_scenarios=10):
    """Main function to run the multimodal feature pipeline."""
    print("Starting Multimodal Feature Pipeline")
    
    # Create pipeline
    pipeline = MultimodalFeaturePipeline()
    
    # Process scenarios
    fused_df = pipeline.process_scenarios(max_scenarios=max_scenarios)
    
    # Simulate uploading to feature store
    if fused_df is not None and not fused_df.empty:
        pipeline.upload_to_feature_store()
        
        # Train yield prediction model
        pipeline.train_yield_prediction_model()
    
    print("Pipeline complete!")

if __name__ == "__main__":
    # Process 10 scenarios by default
    main(max_scenarios=10) 