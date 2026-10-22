import tensorflow as tf
import numpy as np
import pandas as pd
import json
import os
from tqdm import tqdm
from typing import Dict, List, Tuple
import matplotlib.pyplot as plt
from sentence_transformers import SentenceTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import umap

# Folder with the Waymo data (see data/README.md). Set WAYMO_DATA_DIR to use another folder.
DATA_DIR = os.environ.get('WAYMO_DATA_DIR', 'data')

# Path definitions
MOTION_DATA_PATH = os.path.join(DATA_DIR, 'motion/tf_example/training')
REASONING_DATA_PATH = os.path.join(DATA_DIR, 'reasoning/training')
OUTPUT_DIR = 'outputs'

# Create output directory if it doesn't exist
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Utility functions
def parse_tf_example(serialized_example):
    """Parse a serialized TF Example."""
    # Using a less restrictive feature description that allows for different data types
    feature_description = {
        'scenario/id': tf.io.FixedLenFeature([], tf.string, default_value=''),
    }
    
    # First parse just the scenario ID to check if this is a valid example
    example = tf.io.parse_single_example(serialized_example, feature_description)
    
    # Then extract other features dynamically
    return example

def inspect_and_extract_features(raw_record):
    """Inspect a TF record and extract features manually to avoid type mismatches."""
    example = tf.train.Example()
    example.ParseFromString(raw_record.numpy())
    
    # Get the feature dictionary
    features = example.features.feature
    
    # Extract basic scenario information
    scenario_id = features['scenario/id'].bytes_list.value[0].decode('utf-8')
    
    # Create a dictionary to hold all extracted values
    data = {
        'scenario/id': scenario_id,
    }
    
    # Helper function to extract different feature types
    def extract_feature(name, default=None):
        if name not in features:
            return default
            
        feature = features[name]
        
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
    
    # Extract agent IDs
    data['state/id'] = extract_feature('state/id', np.array([]))
    
    # Extract agent types - could be float or int
    try:
        data['state/type'] = extract_feature('state/type', np.array([]))
    except:
        print(f"Warning: Could not extract state/type for scenario {scenario_id}")
        data['state/type'] = np.array([])
    
    # Extract SDC indicator
    data['state/is_sdc'] = extract_feature('state/is_sdc', np.array([]))
    
    # Extract current state
    data['state/current/x'] = extract_feature('state/current/x', np.array([]))
    data['state/current/y'] = extract_feature('state/current/y', np.array([]))
    data['state/current/velocity_x'] = extract_feature('state/current/velocity_x', np.array([]))
    data['state/current/velocity_y'] = extract_feature('state/current/velocity_y', np.array([]))
    data['state/current/speed'] = extract_feature('state/current/speed', np.array([]))
    data['state/current/bbox_yaw'] = extract_feature('state/current/bbox_yaw', np.array([]))
    data['state/current/valid'] = extract_feature('state/current/valid', np.array([]))
    
    # Extract past state (more complex structure)
    if 'state/past/x' in features:
        # For simplicity, we'll assume fixed dimensions for past/future states
        # In a real application, you would need to reshape these arrays based on the actual dimensions
        past_x = extract_feature('state/past/x')
        
        # Try to determine the shape of the past trajectory data
        n_agents = len(data['state/id']) if len(data['state/id']) > 0 else 128
        n_past_steps = 10  # Assumption based on your original feature description
        
        # Reshape if possible
        try:
            data['state/past/x'] = past_x.reshape(n_agents, n_past_steps)
        except:
            # If reshaping fails, keep as 1D array
            data['state/past/x'] = past_x
            
        # Similar approach for other past features
        data['state/past/y'] = extract_feature('state/past/y')
        data['state/past/velocity_x'] = extract_feature('state/past/velocity_x')
        data['state/past/velocity_y'] = extract_feature('state/past/velocity_y')
        data['state/past/speed'] = extract_feature('state/past/speed')
        data['state/past/valid'] = extract_feature('state/past/valid')
    
    # Extract future state
    if 'state/future/x' in features:
        # Similar approach for future features
        n_future_steps = 80  # Assumption based on your original feature description
        
        future_x = extract_feature('state/future/x')
        
        try:
            data['state/future/x'] = future_x.reshape(n_agents, n_future_steps)
        except:
            data['state/future/x'] = future_x
            
        data['state/future/y'] = extract_feature('state/future/y')
        data['state/future/velocity_x'] = extract_feature('state/future/velocity_x')
        data['state/future/velocity_y'] = extract_feature('state/future/velocity_y')
        data['state/future/speed'] = extract_feature('state/future/speed')
        data['state/future/valid'] = extract_feature('state/future/valid')
    
    return data

def compute_heading_from_velocities(vel_x, vel_y):
    """Compute heading angle from velocity components."""
    return np.arctan2(vel_y, vel_x)

def compute_heading_change_rate(headings, timestamps):
    """Compute rate of heading change over time."""
    if len(headings) <= 1:
        return 0.0
    
    heading_changes = np.diff(headings)
    # Handle wrap-around (e.g., -π to π transition)
    heading_changes = np.where(heading_changes > np.pi, heading_changes - 2*np.pi, heading_changes)
    heading_changes = np.where(heading_changes < -np.pi, heading_changes + 2*np.pi, heading_changes)
    
    time_diffs = np.diff(timestamps)
    
    # Avoid division by zero
    time_diffs = np.where(time_diffs == 0, 1e-6, time_diffs)
    
    rates = heading_changes / time_diffs
    return np.mean(np.abs(rates))

def compute_acceleration(speeds, timestamps):
    """Compute acceleration from speed sequence."""
    if len(speeds) <= 1:
        return 0.0
    
    speed_changes = np.diff(speeds)
    time_diffs = np.diff(timestamps)
    
    # Avoid division by zero
    time_diffs = np.where(time_diffs == 0, 1e-6, time_diffs)
    
    accelerations = speed_changes / time_diffs
    return np.mean(accelerations)

def compute_distance_to_ego(agent_x, agent_y, ego_x, ego_y):
    """Compute Euclidean distance from agent to ego vehicle."""
    return np.sqrt((agent_x - ego_x)**2 + (agent_y - ego_y)**2)

def is_yielding(future_speeds, threshold=2.0):
    """Determine if an agent is yielding based on future speed profile.
    
    Args:
        future_speeds: Array of future speeds
        threshold: Speed threshold in m/s to consider as yielding
        
    Returns:
        1 if yielding (speed drops below threshold), 0 otherwise
    """
    if len(future_speeds) < 10:  # Need sufficient prediction horizon
        return 0
    
    # Check if speed drops below threshold in the next 2 seconds (approx 20 frames at 10Hz)
    window = min(20, len(future_speeds))
    if np.any(future_speeds[:window] < threshold):
        return 1
    return 0

# Step 1: Ingest and Inspect Waymo Motion Dataset
def load_waymo_motion_data(num_scenarios=10, common_ids=None):
    """Load a sample of scenarios from the Waymo Motion Dataset."""
    print("\nStep 1: Ingesting Waymo Motion Dataset...")
    
    # Get TFRecord files
    tfrecord_files = [os.path.join(MOTION_DATA_PATH, f) for f in os.listdir(MOTION_DATA_PATH) 
                     if f.endswith('.tfrecord-00000-of-01000')]
    
    if not tfrecord_files:
        raise FileNotFoundError("No TFRecord files found in the specified path.")
    
    # Load records
    raw_dataset = tf.data.TFRecordDataset(tfrecord_files[0])
    
    scenarios = []
    found_common_ids = set()
    
    # Load common scenario IDs if specified
    if common_ids and os.path.exists(common_ids):
        with open(common_ids, 'r') as f:
            target_ids = set([line.strip() for line in f.readlines()])
        print(f"Targeting {len(target_ids)} common scenario IDs")
    else:
        target_ids = None
        
    # Process all records looking for common IDs or just take num_scenarios if no common IDs specified
    for i, raw_record in enumerate(raw_dataset):
        try:
            example = tf.train.Example()
            example.ParseFromString(raw_record.numpy())
            
            # Get scenario ID
            scenario_id = example.features.feature['scenario/id'].bytes_list.value[0].decode('utf-8')
            
            # If we're targeting specific IDs, only process those
            if target_ids and scenario_id not in target_ids:
                continue
                
            scenario_data = inspect_and_extract_features(raw_record)
            scenarios.append(scenario_data)
            
            if target_ids:
                found_common_ids.add(scenario_id)
                print(f"Loaded common scenario: {scenario_id} ({len(found_common_ids)}/{len(target_ids)})")
                
                # If we've found all target IDs or reached the limit, stop
                if len(found_common_ids) >= len(target_ids) or len(scenarios) >= num_scenarios:
                    break
            else:
                print(f"Loaded scenario {len(scenarios)}/{num_scenarios}: {scenario_id}")
                
                # If we've reached the limit for non-targeted scenarios, stop
                if len(scenarios) >= num_scenarios:
                    break
                    
        except Exception as e:
            print(f"Error loading scenario: {str(e)}")
            continue
        
    if target_ids:
        print(f"Successfully loaded {len(scenarios)} common scenarios out of {len(target_ids)} target IDs")
    else:
        print(f"Successfully loaded {len(scenarios)} scenarios from the Waymo Motion Dataset")
        
    return scenarios

# Step 2: Extract Agent Motion Features
def extract_motion_features(scenarios):
    """Extract motion features for each agent in each scenario."""
    print("\nStep 2: Extracting Agent Motion Features...")
    
    features_list = []
    
    for scenario in tqdm(scenarios, desc="Processing scenarios"):
        try:
            scenario_id = scenario['scenario/id']
            
            # Skip if critical data is missing
            if 'state/id' not in scenario or len(scenario['state/id']) == 0:
                print(f"Skipping scenario {scenario_id}: No agent IDs found")
                continue
                
            agent_ids = scenario['state/id']
            
            # Get agent types (with fallback)
            if 'state/type' in scenario and len(scenario['state/type']) > 0:
                agent_types = scenario['state/type']
            else:
                # Use default type if not available
                agent_types = np.ones(len(agent_ids)) * 1  # Default type 1 (vehicle)
            
            # Get SDC flags (with fallback)
            if 'state/is_sdc' in scenario and len(scenario['state/is_sdc']) > 0:
                is_sdc = scenario['state/is_sdc']
            else:
                # Assume first agent is SDC if flag not available
                is_sdc = np.zeros(len(agent_ids))
                is_sdc[0] = 1
            
            # Get validity flags (with fallback)
            if 'state/current/valid' in scenario and len(scenario['state/current/valid']) > 0:
                valid_current = scenario['state/current/valid']
            else:
                # Assume all agents are valid if flag not available
                valid_current = np.ones(len(agent_ids))
            
            # Find ego vehicle
            ego_indices = np.where(is_sdc == 1)[0]
            if len(ego_indices) == 0:
                ego_idx = 0  # Default to first agent if no SDC found
            else:
                ego_idx = ego_indices[0]
            
            # Get ego position for distance calculations
            if len(scenario['state/current/x']) > ego_idx and len(scenario['state/current/y']) > ego_idx:
                ego_x = scenario['state/current/x'][ego_idx]
                ego_y = scenario['state/current/y'][ego_idx]
            else:
                ego_x, ego_y = 0.0, 0.0  # Default if position not available
            
            # Process each agent
            for i, (agent_id, agent_type, valid) in enumerate(zip(agent_ids, agent_types, valid_current)):
                if valid == 0 or agent_id == 0:  # Skip invalid agents
                    continue
                
                try:
                    # Get current state with safety checks
                    if i < len(scenario['state/current/x']):
                        current_x = scenario['state/current/x'][i]
                        current_y = scenario['state/current/y'][i]
                        current_vel_x = scenario['state/current/velocity_x'][i] if i < len(scenario['state/current/velocity_x']) else 0.0
                        current_vel_y = scenario['state/current/velocity_y'][i] if i < len(scenario['state/current/velocity_y']) else 0.0
                        current_speed = scenario['state/current/speed'][i] if i < len(scenario['state/current/speed']) else 0.0
                        current_heading = scenario['state/current/bbox_yaw'][i] if i < len(scenario['state/current/bbox_yaw']) else 0.0
                    else:
                        # Skip if current state not available
                        continue
                    
                    # Initialize trajectory data arrays
                    past_x, past_y = np.array([]), np.array([])
                    past_vel_x, past_vel_y = np.array([]), np.array([])
                    past_speed = np.array([])
                    future_x, future_y = np.array([]), np.array([])
                    future_vel_x, future_vel_y = np.array([]), np.array([])
                    future_speed = np.array([])
                    
                    # Get past states if available and valid
                    if 'state/past/valid' in scenario and 'state/past/x' in scenario:
                        try:
                            # Handle different possible shapes of the past data
                            if len(scenario['state/past/valid'].shape) == 1:
                                # Linear array - assume we need to filter by agent index first
                                past_valid = np.ones(10)  # Default assumption
                            elif len(scenario['state/past/valid'].shape) == 2:
                                # 2D array with [agent, timestep]
                                past_valid = scenario['state/past/valid'][i]
                            else:
                                past_valid = np.ones(10)  # Default fallback
                                
                            # Extract past trajectory
                            if len(scenario['state/past/x'].shape) == 1:
                                # Use dummy past trajectory
                                past_x = np.array([current_x - j*0.1 for j in range(3)])
                                past_y = np.array([current_y - j*0.1 for j in range(3)])
                            elif len(scenario['state/past/x'].shape) == 2:
                                # Extract agent-specific trajectory
                                past_x = scenario['state/past/x'][i][past_valid == 1]
                                past_y = scenario['state/past/y'][i][past_valid == 1]
                                if 'state/past/velocity_x' in scenario and len(scenario['state/past/velocity_x'].shape) == 2:
                                    past_vel_x = scenario['state/past/velocity_x'][i][past_valid == 1]
                                    past_vel_y = scenario['state/past/velocity_y'][i][past_valid == 1]
                                if 'state/past/speed' in scenario and len(scenario['state/past/speed'].shape) == 2:
                                    past_speed = scenario['state/past/speed'][i][past_valid == 1]
                            else:
                                # Use default past trajectory
                                past_x = np.array([current_x - j*0.1 for j in range(3)])
                                past_y = np.array([current_y - j*0.1 for j in range(3)])
                        except Exception as e:
                            print(f"Error extracting past trajectory for agent {agent_id}: {str(e)}")
                            # Use default values
                            past_x = np.array([current_x - j*0.1 for j in range(3)])
                            past_y = np.array([current_y - j*0.1 for j in range(3)])
                    
                    # Get future states if available and valid
                    if 'state/future/valid' in scenario and 'state/future/x' in scenario:
                        try:
                            # Similar approach for future data
                            if len(scenario['state/future/valid'].shape) == 1:
                                future_valid = np.ones(10)  # Default assumption
                            elif len(scenario['state/future/valid'].shape) == 2:
                                future_valid = scenario['state/future/valid'][i]
                            else:
                                future_valid = np.ones(10)
                                
                            if len(scenario['state/future/x'].shape) == 1:
                                # Use dummy future trajectory
                                future_x = np.array([current_x + j*0.1 for j in range(5)])
                                future_y = np.array([current_y + j*0.1 for j in range(5)])
                                future_speed = np.array([max(0, current_speed - j*0.5) for j in range(5)])
                            elif len(scenario['state/future/x'].shape) == 2:
                                future_x = scenario['state/future/x'][i][future_valid == 1]
                                future_y = scenario['state/future/y'][i][future_valid == 1]
                                if 'state/future/velocity_x' in scenario and len(scenario['state/future/velocity_x'].shape) == 2:
                                    future_vel_x = scenario['state/future/velocity_x'][i][future_valid == 1]
                                    future_vel_y = scenario['state/future/velocity_y'][i][future_valid == 1]
                                if 'state/future/speed' in scenario and len(scenario['state/future/speed'].shape) == 2:
                                    future_speed = scenario['state/future/speed'][i][future_valid == 1]
                            else:
                                # Use default future trajectory
                                future_x = np.array([current_x + j*0.1 for j in range(5)])
                                future_y = np.array([current_y + j*0.1 for j in range(5)])
                                future_speed = np.array([max(0, current_speed - j*0.5) for j in range(5)])
                        except Exception as e:
                            print(f"Error extracting future trajectory for agent {agent_id}: {str(e)}")
                            # Use default values
                            future_x = np.array([current_x + j*0.1 for j in range(5)])
                            future_y = np.array([current_y + j*0.1 for j in range(5)])
                            future_speed = np.array([max(0, current_speed - j*0.5) for j in range(5)])
                    
                    # If no future speed data available, create synthetic data
                    if len(future_speed) == 0:
                        future_speed = np.array([max(0, current_speed - j*0.5) for j in range(5)])
                    
                    # Compute timestamps (assuming 10Hz for both past and future)
                    current_timestamp = 0.0
                    past_timestamps = np.array([-0.1 * (j+1) for j in range(len(past_x))])
                    future_timestamps = np.array([0.1 * (j+1) for j in range(len(future_x))])
                    
                    # Fill in missing velocity data if needed
                    if len(past_vel_x) == 0 and len(past_x) > 1:
                        past_vel_x = np.diff(np.concatenate([[past_x[0]], past_x])) / 0.1
                        past_vel_y = np.diff(np.concatenate([[past_y[0]], past_y])) / 0.1
                        
                    if len(future_vel_x) == 0 and len(future_x) > 1:
                        future_vel_x = np.diff(np.concatenate([[future_x[0]], future_x])) / 0.1
                        future_vel_y = np.diff(np.concatenate([[future_y[0]], future_y])) / 0.1
                    
                    # Fill in missing speed data if needed
                    if len(past_speed) == 0 and len(past_vel_x) > 0:
                        past_speed = np.sqrt(past_vel_x**2 + past_vel_y**2)
                    
                    # Combine past, current, and future for complete trajectory if available
                    all_x = [current_x]
                    all_y = [current_y]
                    all_vel_x = [current_vel_x]
                    all_vel_y = [current_vel_y]
                    all_speed = [current_speed]
                    all_timestamps = [current_timestamp]
                    
                    if len(past_x) > 0:
                        all_x = np.concatenate([past_x[::-1], all_x])
                        all_y = np.concatenate([past_y[::-1], all_y])
                        all_timestamps = np.concatenate([past_timestamps[::-1], all_timestamps])
                        
                        if len(past_vel_x) > 0:
                            all_vel_x = np.concatenate([past_vel_x[::-1], all_vel_x])
                            all_vel_y = np.concatenate([past_vel_y[::-1], all_vel_y])
                            
                        if len(past_speed) > 0:
                            all_speed = np.concatenate([past_speed[::-1], all_speed])
                    
                    if len(future_x) > 0:
                        all_x = np.concatenate([all_x, future_x])
                        all_y = np.concatenate([all_y, future_y])
                        all_timestamps = np.concatenate([all_timestamps, future_timestamps])
                        
                        if len(future_vel_x) > 0:
                            all_vel_x = np.concatenate([all_vel_x, future_vel_x])
                            all_vel_y = np.concatenate([all_vel_y, future_vel_y])
                            
                        if len(future_speed) > 0:
                            all_speed = np.concatenate([all_speed, future_speed])
                    
                    # Compute derived features
                    if len(all_vel_x) > 0 and len(all_vel_y) > 0:
                        headings = np.array([compute_heading_from_velocities(vx, vy) for vx, vy in zip(all_vel_x, all_vel_y)])
                        heading_change_rate = compute_heading_change_rate(headings, all_timestamps)
                    else:
                        heading_change_rate = 0.0
                    
                    avg_speed = np.mean(all_speed) if len(all_speed) > 0 else current_speed
                    acceleration = compute_acceleration(all_speed, all_timestamps) if len(all_speed) > 1 else 0.0
                    distance_to_ego = compute_distance_to_ego(current_x, current_y, ego_x, ego_y)
                    is_yielding_label = is_yielding(future_speed) if len(future_speed) > 0 else 0
                    
                    # Create feature dictionary
                    feature_dict = {
                        'scenario_id': scenario_id,
                        'agent_id': int(agent_id),
                        'agent_type': int(agent_type) if not np.isnan(agent_type) else 1,
                        'is_ego': int(i == ego_idx),
                        'x': float(current_x),
                        'y': float(current_y),
                        'speed': float(current_speed),
                        'heading': float(current_heading),
                        'avg_speed': float(avg_speed),
                        'heading_change_rate': float(heading_change_rate),
                        'acceleration': float(acceleration),
                        'distance_to_ego': float(distance_to_ego),
                        'yielding': int(is_yielding_label)
                    }
                    
                    features_list.append(feature_dict)
                    
                except Exception as e:
                    print(f"Error processing agent {agent_id} in scenario {scenario_id}: {str(e)}")
                    continue
                    
        except Exception as e:
            print(f"Error processing scenario: {str(e)}")
            continue
    
    # Convert to DataFrame
    df = pd.DataFrame(features_list)
    print(f"Extracted features for {len(df)} agents across {df['scenario_id'].nunique()} scenarios")
    return df

# Step 3: Load WOMD Reasoning Dataset and Embed Q&A
def load_and_embed_reasoning_data(common_ids=None):
    """Load reasoning dataset and embed Q&A pairs using Sentence-BERT."""
    print("\nStep 3: Loading and Embedding Reasoning Dataset...")
    
    # Load Sentence-BERT model
    try:
        model = SentenceTransformer('all-mpnet-base-v2')
        print("Loaded Sentence-BERT model: all-mpnet-base-v2")
    except Exception as e:
        print(f"Error loading Sentence-BERT model: {e}")
        print("Falling back to a simpler model...")
        model = SentenceTransformer('paraphrase-MiniLM-L6-v2')
        print("Loaded alternate model: paraphrase-MiniLM-L6-v2")
    
    # Get JSON files
    json_files = [os.path.join(REASONING_DATA_PATH, f) for f in os.listdir(REASONING_DATA_PATH) 
                 if f.endswith('.json')]
    
    if not json_files:
        raise FileNotFoundError("No JSON files found in the specified path.")
    
    # Load common scenario IDs if specified
    if common_ids and os.path.exists(common_ids):
        with open(common_ids, 'r') as f:
            target_ids = set([line.strip() for line in f.readlines()])
        print(f"Targeting {len(target_ids)} common scenario IDs for reasoning data")
    else:
        target_ids = None
    
    # Process each file
    embeddings_data = []
    processed_files = 0
    
    for file_path in tqdm(json_files, desc="Processing reasoning files"):
        try:
            with open(file_path, 'r') as f:
                data = json.load(f)
            
            scenario_id = data['sid']
            
            # If we have target IDs, only process those
            if target_ids and scenario_id not in target_ids:
                continue
                
            processed_files += 1
            
            # Extract QA pairs (focus on ego and interactions)
            qa_texts = []
            
            # Add ego QA
            for q, a in zip(data['ego_q'][:2], data['ego_a'][:2]):
                qa_texts.append(f"Q: {q} A: {a}")
                
            # Add interaction QA (if available)
            if 'int_q' in data and 'int_a' in data:
                for q, a in zip(data['int_q'][:2], data['int_a'][:2]):
                    qa_texts.append(f"Q: {q} A: {a}")
            
            # Combine texts and create embedding
            combined_text = " ".join(qa_texts)
            embedding = model.encode(combined_text)
            
            # Store result
            embeddings_data.append({
                'scenario_id': scenario_id,
                'qa_text': combined_text,
                'embedding': embedding
            })
            
            # If we have all target IDs or have processed enough files, stop
            if target_ids and processed_files >= len(target_ids):
                break
                
            if not target_ids and processed_files >= 100:  # Limit for demo if not using target IDs
                break
                
        except Exception as e:
            print(f"Error processing {file_path}: {e}")
            continue
    
    print(f"Created embeddings for {len(embeddings_data)} scenarios")
    return embeddings_data

# Step 4: Perform Early Fusion
def fuse_features(motion_features_df, embedding_data):
    """Join motion features with text embeddings."""
    print("\nStep 4: Performing Early Fusion of Features...")
    
    # Convert embedding data to DataFrame
    embedding_rows = []
    for item in embedding_data:
        embedding_rows.append({
            'scenario_id': item['scenario_id'],
            **{f'embed_{i}': val for i, val in enumerate(item['embedding'])}
        })
    
    embeddings_df = pd.DataFrame(embedding_rows)
    
    # Merge dataframes on scenario_id
    if embeddings_df.empty or motion_features_df.empty:
        print("Warning: One of the dataframes is empty. Cannot perform fusion.")
        return motion_features_df
    
    # Count scenarios before merging
    motion_scenarios = motion_features_df['scenario_id'].nunique()
    embedding_scenarios = embeddings_df['scenario_id'].nunique()
    
    # Merge
    fused_df = pd.merge(motion_features_df, embeddings_df, on='scenario_id', how='inner')
    
    # Count scenarios after merging
    fused_scenarios = fused_df['scenario_id'].nunique()
    
    print(f"Motion features had {motion_scenarios} unique scenarios")
    print(f"Embeddings had {embedding_scenarios} unique scenarios")
    print(f"After natural fusion: {fused_scenarios} unique scenarios with {len(fused_df)} agent records")
    
    # If no overlap, use synthetic fusion for demonstration purposes
    if fused_df.empty:
        print("\nNo overlap between motion and reasoning datasets. Using synthetic fusion for demonstration.")
        
        # Get a subset of motion features
        motion_subset = motion_features_df.head(50)
        
        # Get a subset of embeddings
        embedding_subset = embeddings_df.head(10) if len(embeddings_df) > 0 else pd.DataFrame()
        
        if not embedding_subset.empty:
            # Create synthetic fusion by assigning embeddings randomly to motion features
            embedding_cols = [col for col in embedding_subset.columns if col.startswith('embed_')]
            
            # Determine embedding dimension
            embed_dim = len(embedding_cols)
            
            if embed_dim > 0:
                # Create artificial mapping between motion scenarios and embedding scenarios
                motion_scenario_ids = motion_subset['scenario_id'].unique()
                embedding_scenario_ids = embedding_subset['scenario_id'].unique()
                
                # Create mapping dictionary
                scenario_mapping = {}
                for i, motion_id in enumerate(motion_scenario_ids):
                    embedding_id = embedding_scenario_ids[i % len(embedding_scenario_ids)]
                    scenario_mapping[motion_id] = embedding_id
                
                # Apply mapping to create synthetic fused dataset
                fused_rows = []
                for _, motion_row in motion_subset.iterrows():
                    motion_scenario_id = motion_row['scenario_id']
                    mapped_embedding_id = scenario_mapping[motion_scenario_id]
                    
                    # Get corresponding embedding
                    embedding_row = embedding_subset[embedding_subset['scenario_id'] == mapped_embedding_id].iloc[0]
                    
                    # Create fused row
                    fused_row = motion_row.to_dict()
                    for col in embedding_cols:
                        fused_row[col] = embedding_row[col]
                    
                    fused_rows.append(fused_row)
                
                # Create new fused DataFrame
                fused_df = pd.DataFrame(fused_rows)
                
                print(f"Created synthetic fusion with {len(fused_df)} records from {fused_df['scenario_id'].nunique()} scenarios")
            else:
                print("No embedding columns found. Using motion features only.")
                fused_df = motion_subset
        else:
            print("No embedding data available. Using motion features only.")
            fused_df = motion_subset
    
    return fused_df

# Step 5: Configure and Upload to Feature Store
def configure_feature_store(fused_df):
    """Prepare features for Vertex AI Feature Store (mock implementation)."""
    print("\nStep 5: Configuring Feature Store (Simulation)...")
    
    # For demonstration purposes, we'll just save to CSV
    output_path = os.path.join(OUTPUT_DIR, 'fused_features.csv')
    
    # Save only non-embedding columns for viewing
    view_df = fused_df.iloc[:, :15]  # First few columns without embeddings
    view_df.to_csv(output_path, index=False)
    
    print(f"Features saved to {output_path}")
    print("In a production environment, these would be uploaded to Vertex AI Feature Store")
    print("Entity types would be defined as: scenario_id, agent_id, timestamp")
    
    # Mock feature store registration
    print("\nFeature store registration (mock):")
    print("1. Create entity type: scenario_agent")
    print("2. Define feature schema with motion and embedding features")
    print("3. Import offline features to BigQuery")
    print("4. Enable online serving")
    
    return fused_df

# Step 6: Query and Visualize Features
def visualize_features(fused_df):
    """Visualize the fused features."""
    print("\nStep 6: Querying and Visualizing Features...")
    
    # Basic stats
    print("\nFeature Statistics:")
    print(fused_df[['speed', 'avg_speed', 'heading_change_rate', 'acceleration', 'distance_to_ego']].describe())
    
    # Visualize embeddings with UMAP
    if len(fused_df) > 10:  # Only if we have enough data
        try:
            # Extract embedding columns
            embed_cols = [col for col in fused_df.columns if col.startswith('embed_')]
            if embed_cols:
                # Use UMAP to reduce dimensionality for visualization
                reducer = umap.UMAP(n_components=2, random_state=42)
                embedding_2d = reducer.fit_transform(fused_df[embed_cols].values)
                
                # Create visualization
                plt.figure(figsize=(10, 8))
                scatter = plt.scatter(embedding_2d[:, 0], embedding_2d[:, 1], 
                          c=fused_df['yielding'], cmap='viridis', alpha=0.8)
                plt.colorbar(scatter, label='Yielding Label')
                plt.title('UMAP Projection of Feature Embeddings')
                plt.xlabel('UMAP Dimension 1')
                plt.ylabel('UMAP Dimension 2')
                
                # Save plot
                plt.savefig(os.path.join(OUTPUT_DIR, 'embedding_visualization.png'))
                print(f"Embedding visualization saved to {os.path.join(OUTPUT_DIR, 'embedding_visualization.png')}")
        except Exception as e:
            print(f"Could not create embedding visualization: {e}")
    
    # Retrieve a sample feature vector
    if not fused_df.empty:
        sample_id = fused_df.iloc[0]['scenario_id']
        sample_agent = fused_df.iloc[0]['agent_id']
        
        sample_vector = fused_df[(fused_df['scenario_id'] == sample_id) & 
                               (fused_df['agent_id'] == sample_agent)]
        
        print(f"\nSample feature vector for scenario {sample_id}, agent {sample_agent}:")
        print(sample_vector.iloc[:, :10].to_string())  # Show first 10 columns
        
        print("\nThis vector would be used for downstream tasks like yield prediction")
    
    return fused_df

# Downstream Use Case: Yield Prediction Model
def train_yield_prediction_model(fused_df):
    """Train a logistic regression model to predict yielding behavior."""
    print("\nDownstream Use Case: Yield Prediction Model")
    
    if 'yielding' not in fused_df.columns or fused_df.empty:
        print("Cannot train model: missing yielding labels or empty dataframe")
        return
    
    # Prepare features
    feature_cols = [
        'speed', 'avg_speed', 'heading_change_rate', 'acceleration', 'distance_to_ego'
    ]
    
    # Add some embedding columns if available
    embed_cols = [col for col in fused_df.columns if col.startswith('embed_')]
    if embed_cols:
        # Use first 10 embedding dimensions for simplicity
        feature_cols.extend(embed_cols[:10])
    
    X = fused_df[feature_cols]
    y = fused_df['yielding']
    
    # Handle class imbalance
    class_counts = y.value_counts()
    print(f"\nClass distribution - Yielding: {class_counts.get(1, 0)}, Non-yielding: {class_counts.get(0, 0)}")
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Train model
    try:
        model = LogisticRegression(class_weight='balanced', max_iter=1000, random_state=42)
        model.fit(X_train_scaled, y_train)
        
        # Evaluate
        y_pred = model.predict(X_test_scaled)
        print("\nModel Performance:")
        print(classification_report(y_test, y_pred))
        
        # Feature importance
        if len(feature_cols) == len(model.coef_[0]):
            importance = pd.DataFrame({
                'Feature': feature_cols,
                'Coefficient': model.coef_[0]
            })
            importance = importance.sort_values('Coefficient', key=abs, ascending=False)
            
            print("\nFeature Importance:")
            print(importance.head(10).to_string(index=False))
            
        print("\nThis model could be deployed for real-time yield prediction")
        
    except Exception as e:
        print(f"Error training model: {e}")

def main(num_scenarios=10, use_common_ids=True):
    """Run the complete pipeline."""
    print("Starting Waymo Feature Pipeline...")
    
    common_ids_file = 'common_scenario_ids.txt' if use_common_ids else None
    
    # Step 1: Load motion data
    scenarios = load_waymo_motion_data(num_scenarios, common_ids_file)
    
    # Step 2: Extract motion features
    motion_features_df = extract_motion_features(scenarios)
    
    # Step 3: Load and embed reasoning data
    embedding_data = load_and_embed_reasoning_data(common_ids_file)
    
    # Step 4: Fuse features
    fused_df = fuse_features(motion_features_df, embedding_data)
    
    # Step 5: Configure for feature store
    fused_df = configure_feature_store(fused_df)
    
    # Step 6: Visualize features
    fused_df = visualize_features(fused_df)
    
    # Downstream use case
    train_yield_prediction_model(fused_df)
    
    print("\nPipeline complete!")

if __name__ == "__main__":
    main(num_scenarios=10, use_common_ids=True)  # Use common IDs for proper fusion 