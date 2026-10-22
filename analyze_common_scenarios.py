import tensorflow as tf
import os
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from collections import defaultdict, Counter

# Folder with the Waymo data (see data/README.md). Set WAYMO_DATA_DIR to use another folder.
DATA_DIR = os.environ.get('WAYMO_DATA_DIR', 'data')

# Path definitions
MOTION_VALIDATION_PATH = os.path.join(DATA_DIR, 'motion/tf_example/validation_interactive')
REASONING_VALIDATION_PATH = os.path.join(DATA_DIR, 'reasoning/validation_interactive')
OUTPUT_DIR = 'outputs/validation_analysis'

# Create output directory if it doesn't exist
os.makedirs(OUTPUT_DIR, exist_ok=True)

def load_common_scenario_ids():
    """Load the common scenario IDs identified in both datasets."""
    scenario_id_file = os.path.join(OUTPUT_DIR, 'scenario_id_comparison.json')
    
    if not os.path.exists(scenario_id_file):
        print(f"Error: {scenario_id_file} not found. Run extract_validation_ids.py first.")
        return []
        
    with open(scenario_id_file, 'r') as f:
        comparison = json.load(f)
    
    common_ids = comparison.get('common_ids', [])
    print(f"Loaded {len(common_ids)} common scenario IDs.")
    return common_ids

def load_reasoning_data_for_scenarios(scenario_ids):
    """Load the reasoning data for the specified scenario IDs."""
    print(f"\nLoading reasoning data for {len(scenario_ids)} common scenarios...")
    
    reasoning_data = []
    not_found = []
    
    for scenario_id in tqdm(scenario_ids, desc="Loading reasoning data"):
        # Look for files matching this scenario ID
        matching_files = [os.path.join(REASONING_VALIDATION_PATH, f) 
                         for f in os.listdir(REASONING_VALIDATION_PATH)
                         if f.startswith(f"scid_{scenario_id}__")]
        
        if not matching_files:
            not_found.append(scenario_id)
            continue
            
        # Load each matching file
        for file_path in matching_files:
            try:
                with open(file_path, 'r') as f:
                    data = json.load(f)
                    
                # Add source file to track which file this came from
                data['source_file'] = os.path.basename(file_path)
                reasoning_data.append(data)
                
            except Exception as e:
                print(f"Error loading {file_path}: {str(e)}")
                continue
    
    print(f"Successfully loaded {len(reasoning_data)} reasoning files for common scenarios.")
    if not_found:
        print(f"Note: {len(not_found)} common scenario IDs were not found in the reasoning data.")
    
    return reasoning_data

def extract_motion_features_for_scenario(scenario_id, tfrecord_files):
    """Extract basic motion features for a specific scenario ID."""
    print(f"\nExtracting motion features for scenario {scenario_id}...")
    
    for tfrecord_file in tfrecord_files:
        raw_dataset = tf.data.TFRecordDataset(tfrecord_file)
        
        for raw_record in tqdm(raw_dataset, desc=f"Searching in {os.path.basename(tfrecord_file)}"):
            try:
                example = tf.train.Example()
                example.ParseFromString(raw_record.numpy())
                
                # Check if this is the scenario we're looking for
                record_scenario_id = example.features.feature['scenario/id'].bytes_list.value[0].decode('utf-8')
                
                if record_scenario_id == scenario_id:
                    print(f"Found scenario {scenario_id} in {os.path.basename(tfrecord_file)}")
                    
                    # Extract basic features
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
                    
                    # Extract relevant features
                    try:
                        features['agent_ids'] = extract_feature('state/id')
                        features['agent_types'] = extract_feature('state/type')
                        features['is_sdc'] = extract_feature('state/is_sdc')
                        features['x'] = extract_feature('state/current/x')
                        features['y'] = extract_feature('state/current/y')
                        features['speed'] = extract_feature('state/current/speed')
                        features['valid'] = extract_feature('state/current/valid')
                        
                        # Return the extracted features
                        return features
                        
                    except Exception as e:
                        print(f"Error extracting features: {str(e)}")
                        return None
                    
            except Exception as e:
                print(f"Error processing record: {str(e)}")
                continue
    
    print(f"Warning: Scenario {scenario_id} not found in motion data.")
    return None

def analyze_common_scenarios(common_ids, max_scenarios=5):
    """Analyze the common scenarios between motion and reasoning datasets."""
    print(f"\nAnalyzing common scenarios (limited to {max_scenarios} for detailed analysis)...")
    
    # Get TFRecord files
    tfrecord_files = [os.path.join(MOTION_VALIDATION_PATH, f) for f in os.listdir(MOTION_VALIDATION_PATH) 
                     if f.endswith('.tfrecord-00000-of-00150') or '.tfrecord' in f]
    
    # Load reasoning data for all common scenarios
    reasoning_data = load_reasoning_data_for_scenarios(common_ids)
    
    # Group reasoning data by scenario ID
    scenario_to_reasoning = defaultdict(list)
    for data in reasoning_data:
        scenario_to_reasoning[data['sid']].append(data)
    
    # Analyze a subset of common scenarios in detail
    detailed_analysis = []
    
    for i, scenario_id in enumerate(common_ids[:max_scenarios]):
        print(f"\nDetailed analysis for scenario {i+1}/{max_scenarios}: {scenario_id}")
        
        # Get reasoning data for this scenario
        scenario_reasoning = scenario_to_reasoning.get(scenario_id, [])
        
        if not scenario_reasoning:
            print(f"No reasoning data found for scenario {scenario_id}")
            continue
            
        # Extract motion features
        motion_features = extract_motion_features_for_scenario(scenario_id, tfrecord_files)
        
        if not motion_features:
            print(f"No motion features found for scenario {scenario_id}")
            continue
            
        # Analyze the scenario
        analysis = {
            'scenario_id': scenario_id,
            'num_reasoning_files': len(scenario_reasoning),
            'motion': {
                'num_agents': int(len(motion_features['agent_ids'])) if 'agent_ids' in motion_features else 0,
                'ego_agent_id': int(motion_features['agent_ids'][np.where(motion_features['is_sdc'] == 1)[0][0]]) 
                               if 'agent_ids' in motion_features and 'is_sdc' in motion_features and np.any(motion_features['is_sdc'] == 1) else None,
            },
            'reasoning': {
                'ego_agent_ids': [int(data['ego']) for data in scenario_reasoning],
                'question_counts': {
                    'env': sum(len(data.get('env_q', [])) for data in scenario_reasoning),
                    'ego': sum(len(data.get('ego_q', [])) for data in scenario_reasoning),
                    'surrounding': sum(len(data.get('sur_q', [])) for data in scenario_reasoning),
                    'interaction': sum(len(data.get('int_q', [])) for data in scenario_reasoning),
                }
            }
        }
        
        # Check if ego agent IDs match
        if analysis['motion']['ego_agent_id'] is not None:
            matches = [ego_id == analysis['motion']['ego_agent_id'] for ego_id in analysis['reasoning']['ego_agent_ids']]
            analysis['ego_id_match'] = any(matches)
            print(f"Ego agent ID in motion data: {analysis['motion']['ego_agent_id']}")
            print(f"Ego agent IDs in reasoning data: {analysis['reasoning']['ego_agent_ids']}")
            print(f"Match: {analysis['ego_id_match']}")
        else:
            analysis['ego_id_match'] = False
            print("Ego agent ID not found in motion data")
            
        # Add to detailed analysis
        detailed_analysis.append(analysis)
    
    return reasoning_data, detailed_analysis

def analyze_question_distribution(reasoning_data):
    """Analyze the distribution of questions in the reasoning data."""
    print("\nAnalyzing question distribution...")
    
    # Count questions by category
    env_counts = []
    ego_counts = []
    sur_counts = []
    int_counts = []
    
    for data in reasoning_data:
        env_counts.append(len(data.get('env_q', [])))
        ego_counts.append(len(data.get('ego_q', [])))
        sur_counts.append(len(data.get('sur_q', [])))
        int_counts.append(len(data.get('int_q', [])))
    
    # Plot distribution
    plt.figure(figsize=(15, 10))
    
    plt.subplot(2, 2, 1)
    plt.hist(env_counts, bins=15, alpha=0.7, color='blue')
    plt.title('Environment Questions Distribution')
    plt.xlabel('Number of Questions')
    plt.ylabel('Frequency')
    
    plt.subplot(2, 2, 2)
    plt.hist(ego_counts, bins=15, alpha=0.7, color='green')
    plt.title('Ego Agent Questions Distribution')
    plt.xlabel('Number of Questions')
    plt.ylabel('Frequency')
    
    plt.subplot(2, 2, 3)
    plt.hist(sur_counts, bins=15, alpha=0.7, color='red')
    plt.title('Surrounding Agent Questions Distribution')
    plt.xlabel('Number of Questions')
    plt.ylabel('Frequency')
    
    plt.subplot(2, 2, 4)
    plt.hist(int_counts, bins=15, alpha=0.7, color='orange')
    plt.title('Interaction Questions Distribution')
    plt.xlabel('Number of Questions')
    plt.ylabel('Frequency')
    
    plt.tight_layout()
    plt.savefig(f'{OUTPUT_DIR}/question_distribution.png')
    plt.close()
    
    # Calculate statistics and convert numpy types to Python native types
    stats = {
        'environment': {
            'mean': float(np.mean(env_counts)),
            'median': float(np.median(env_counts)),
            'min': int(np.min(env_counts)),
            'max': int(np.max(env_counts)),
        },
        'ego': {
            'mean': float(np.mean(ego_counts)),
            'median': float(np.median(ego_counts)),
            'min': int(np.min(ego_counts)),
            'max': int(np.max(ego_counts)),
        },
        'surrounding': {
            'mean': float(np.mean(sur_counts)),
            'median': float(np.median(sur_counts)),
            'min': int(np.min(sur_counts)),
            'max': int(np.max(sur_counts)),
        },
        'interaction': {
            'mean': float(np.mean(int_counts)),
            'median': float(np.median(int_counts)),
            'min': int(np.min(int_counts)),
            'max': int(np.max(int_counts)),
        }
    }
    
    print("\nQuestion Count Statistics:")
    for category, category_stats in stats.items():
        print(f"  {category.capitalize()} Questions:")
        for stat_name, value in category_stats.items():
            print(f"    {stat_name}: {value:.2f}")
    
    return stats

def main():
    """Main function to analyze common scenarios."""
    # Load common scenario IDs
    common_ids = load_common_scenario_ids()
    
    if not common_ids:
        print("No common scenario IDs found. Exiting.")
        return
    
    # Analyze common scenarios
    reasoning_data, detailed_analysis = analyze_common_scenarios(common_ids, max_scenarios=5)
    
    # Analyze question distribution
    question_stats = analyze_question_distribution(reasoning_data)
    
    # Save results
    print("\nSaving results...")
    
    # Create a custom JSON encoder to handle numpy types
    class NumpyEncoder(json.JSONEncoder):
        def default(self, obj):
            if isinstance(obj, (np.integer, np.int64, np.int32)):
                return int(obj)
            elif isinstance(obj, (np.floating, np.float64, np.float32)):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, np.bool_):
                return bool(obj)
            return super(NumpyEncoder, self).default(obj)
    
    # Save detailed analysis
    with open(f'{OUTPUT_DIR}/detailed_scenario_analysis.json', 'w') as f:
        json.dump(detailed_analysis, f, indent=2, cls=NumpyEncoder)
    
    # Save question statistics
    with open(f'{OUTPUT_DIR}/question_statistics.json', 'w') as f:
        json.dump(question_stats, f, indent=2, cls=NumpyEncoder)
    
    print(f"Results saved to {OUTPUT_DIR}/")

if __name__ == "__main__":
    main() 