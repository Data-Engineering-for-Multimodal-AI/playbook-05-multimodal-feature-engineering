import tensorflow as tf
import os
import json
import pandas as pd
import numpy as np
from tqdm import tqdm

# Path definitions
MOTION_VALIDATION_PATH = 'data/motion/tf_example/validation_interactive'
REASONING_VALIDATION_PATH = 'data/reasoning/validation_interactive'
OUTPUT_DIR = 'outputs/validation_analysis'

# Create output directory if it doesn't exist
os.makedirs(OUTPUT_DIR, exist_ok=True)

def extract_motion_validation_ids(max_examples=None):
    """Extract scenario IDs from Motion validation dataset."""
    print(f"Extracting scenario IDs from Motion validation data at {MOTION_VALIDATION_PATH}...")
    
    # Get TFRecord files
    tfrecord_files = [os.path.join(MOTION_VALIDATION_PATH, f) for f in os.listdir(MOTION_VALIDATION_PATH) 
                     if f.endswith('.tfrecord-00000-of-00150') or '.tfrecord' in f]
    
    if not tfrecord_files:
        print("No TFRecord files found.")
        return []
    
    print(f"Found {len(tfrecord_files)} TFRecord files.")
    
    # Load records
    scenario_ids = []
    agent_ids = []
    
    for file_idx, tfrecord_file in enumerate(tfrecord_files):
        print(f"Processing file {file_idx+1}/{len(tfrecord_files)}: {os.path.basename(tfrecord_file)}")
        
        raw_dataset = tf.data.TFRecordDataset(tfrecord_file)
        file_scenario_ids = []
        file_agent_ids = []
        
        for i, raw_record in enumerate(tqdm(raw_dataset)):
            if max_examples and i >= max_examples:
                break
                
            try:
                example = tf.train.Example()
                example.ParseFromString(raw_record.numpy())
                
                # Get scenario ID
                scenario_id = example.features.feature['scenario/id'].bytes_list.value[0].decode('utf-8')
                file_scenario_ids.append(scenario_id)
                
                # Try to get agent IDs if available
                if 'state/id' in example.features.feature:
                    agent_id_feature = example.features.feature['state/id']
                    
                    if agent_id_feature.int64_list.value:
                        for agent_id in agent_id_feature.int64_list.value:
                            file_agent_ids.append((scenario_id, int(agent_id)))
                
            except Exception as e:
                print(f"Error processing example {i}: {str(e)}")
                continue
            
        print(f"Found {len(file_scenario_ids)} scenario IDs in file {file_idx+1}")
        scenario_ids.extend(file_scenario_ids)
        agent_ids.extend(file_agent_ids)
    
    print(f"Found a total of {len(scenario_ids)} scenario IDs in Motion validation dataset")
    print(f"Found a total of {len(agent_ids)} agent IDs in Motion validation dataset")
    
    # Remove duplicates while preserving order
    unique_scenario_ids = list(dict.fromkeys(scenario_ids))
    
    print(f"Found {len(unique_scenario_ids)} unique scenario IDs")
    
    return unique_scenario_ids, agent_ids

def extract_reasoning_validation_ids():
    """Extract scenario IDs from Reasoning validation dataset."""
    print(f"\nExtracting scenario IDs from Reasoning validation data at {REASONING_VALIDATION_PATH}...")
    
    # Get JSON files
    json_files = [os.path.join(REASONING_VALIDATION_PATH, f) for f in os.listdir(REASONING_VALIDATION_PATH) 
                 if f.endswith('.json')]
    
    if not json_files:
        print("No JSON files found.")
        return []
    
    print(f"Found {len(json_files)} JSON files.")
    
    # Extract scenario IDs and ego agent IDs
    scenario_ids = []
    ego_ids = []
    
    for file_path in tqdm(json_files, desc="Processing files"):
        try:
            with open(file_path, 'r') as f:
                data = json.load(f)
            
            scenario_id = data['sid']
            ego_id = data['ego']
            
            scenario_ids.append(scenario_id)
            ego_ids.append((scenario_id, ego_id))
            
        except Exception as e:
            print(f"Error processing {file_path}: {str(e)}")
            continue
    
    # Remove duplicates while preserving order
    unique_scenario_ids = list(dict.fromkeys(scenario_ids))
    
    print(f"Found {len(unique_scenario_ids)} unique scenario IDs in Reasoning validation dataset")
    print(f"Found {len(ego_ids)} ego agent IDs in Reasoning validation dataset")
    
    return unique_scenario_ids, ego_ids

def compare_scenario_ids(motion_ids, reasoning_ids):
    """Compare scenario IDs between motion and reasoning datasets."""
    motion_set = set(motion_ids)
    reasoning_set = set(reasoning_ids)
    
    # Find common and unique IDs
    common_ids = motion_set.intersection(reasoning_set)
    motion_only_ids = motion_set - reasoning_set
    reasoning_only_ids = reasoning_set - motion_set
    
    print(f"\nComparison results:")
    print(f"  Common IDs: {len(common_ids)}")
    print(f"  Motion-only IDs: {len(motion_only_ids)}")
    print(f"  Reasoning-only IDs: {len(reasoning_only_ids)}")
    
    return {
        'common_ids': list(common_ids),
        'motion_only_ids': list(motion_only_ids),
        'reasoning_only_ids': list(reasoning_only_ids)
    }

def create_agent_mapping(motion_agent_ids, reasoning_ego_ids):
    """Create a mapping between scenario+agent IDs and their presence in datasets."""
    print("\nCreating agent ID mapping...")
    
    # If either dataset has no agent IDs, return empty DataFrame with correct columns
    if not motion_agent_ids or not reasoning_ego_ids:
        print("No agent mappings possible - missing agent IDs from one or both datasets.")
        return pd.DataFrame(columns=['scenario_id', 'agent_id', 'in_motion', 'in_reasoning'])
    
    # Convert to dictionaries for faster lookup
    motion_agent_dict = {}
    for scenario_id, agent_id in motion_agent_ids:
        if scenario_id not in motion_agent_dict:
            motion_agent_dict[scenario_id] = []
        motion_agent_dict[scenario_id].append(agent_id)
    
    reasoning_ego_dict = {}
    for scenario_id, ego_id in reasoning_ego_ids:
        if scenario_id not in reasoning_ego_dict:
            reasoning_ego_dict[scenario_id] = []
        reasoning_ego_dict[scenario_id].append(ego_id)
    
    # Find mappings
    matches = []
    for scenario_id in motion_agent_dict:
        if scenario_id in reasoning_ego_dict:
            motion_agents = set(motion_agent_dict[scenario_id])
            reasoning_egos = set(reasoning_ego_dict[scenario_id])
            
            # Find common agent IDs
            common_agents = motion_agents.intersection(reasoning_egos)
            
            # Add to matches
            for agent_id in common_agents:
                matches.append({
                    'scenario_id': scenario_id,
                    'agent_id': agent_id,
                    'in_motion': True,
                    'in_reasoning': True
                })
            
            # Add motion-only agents
            for agent_id in motion_agents - reasoning_egos:
                matches.append({
                    'scenario_id': scenario_id,
                    'agent_id': agent_id,
                    'in_motion': True,
                    'in_reasoning': False
                })
            
            # Add reasoning-only agents
            for agent_id in reasoning_egos - motion_agents:
                matches.append({
                    'scenario_id': scenario_id,
                    'agent_id': agent_id,
                    'in_motion': False,
                    'in_reasoning': True
                })
    
    print(f"Found {len(matches)} agent mappings")
    
    # Convert to DataFrame for easier analysis
    mapping_df = pd.DataFrame(matches)
    
    # Initialize stat counters
    common_agents_count = 0
    motion_only_agents_count = 0
    reasoning_only_agents_count = 0
    
    # Count statistics if DataFrame is not empty
    if not mapping_df.empty:
        common_agents = mapping_df[(mapping_df['in_motion'] == True) & (mapping_df['in_reasoning'] == True)]
        motion_only_agents = mapping_df[(mapping_df['in_motion'] == True) & (mapping_df['in_reasoning'] == False)]
        reasoning_only_agents = mapping_df[(mapping_df['in_motion'] == False) & (mapping_df['in_reasoning'] == True)]
        
        common_agents_count = len(common_agents)
        motion_only_agents_count = len(motion_only_agents)
        reasoning_only_agents_count = len(reasoning_only_agents)
    
    print(f"  Common agents: {common_agents_count}")
    print(f"  Motion-only agents: {motion_only_agents_count}")
    print(f"  Reasoning-only agents: {reasoning_only_agents_count}")
    
    return mapping_df

def main(max_motion_examples=None):
    """Main function to extract and compare validation IDs."""
    # Extract motion scenario IDs
    motion_scenario_ids, motion_agent_ids = extract_motion_validation_ids(max_examples=max_motion_examples)
    
    # Extract reasoning scenario IDs
    reasoning_scenario_ids, reasoning_ego_ids = extract_reasoning_validation_ids()
    
    if not motion_scenario_ids or not reasoning_scenario_ids:
        print("Failed to extract IDs from one or both datasets. Exiting.")
        return
    
    # Compare scenario IDs
    comparison = compare_scenario_ids(motion_scenario_ids, reasoning_scenario_ids)
    
    # Create agent mapping
    agent_mapping = create_agent_mapping(motion_agent_ids, reasoning_ego_ids)
    
    # Save results
    print("\nSaving results...")
    
    # Save ID comparison
    with open(f'{OUTPUT_DIR}/scenario_id_comparison.json', 'w') as f:
        json.dump(comparison, f, indent=2)
    
    # Save agent mapping as CSV
    if not agent_mapping.empty:
        agent_mapping.to_csv(f'{OUTPUT_DIR}/agent_mapping.csv', index=False)
    else:
        print("No agent mappings to save.")
    
    # Save agent mapping statistics
    stats = {
        'total_mappings': len(agent_mapping),
        'common_agents': len(agent_mapping[(agent_mapping['in_motion'] == True) & (agent_mapping['in_reasoning'] == True)]) if not agent_mapping.empty else 0,
        'motion_only_agents': len(agent_mapping[(agent_mapping['in_motion'] == True) & (agent_mapping['in_reasoning'] == False)]) if not agent_mapping.empty else 0,
        'reasoning_only_agents': len(agent_mapping[(agent_mapping['in_motion'] == False) & (agent_mapping['in_reasoning'] == True)]) if not agent_mapping.empty else 0
    }
    
    with open(f'{OUTPUT_DIR}/agent_mapping_stats.json', 'w') as f:
        json.dump(stats, f, indent=2)
    
    print(f"Results saved to {OUTPUT_DIR}/")

if __name__ == "__main__":
    main(max_motion_examples=1000)  # Limit to 1000 examples for testing, set to None for full dataset 