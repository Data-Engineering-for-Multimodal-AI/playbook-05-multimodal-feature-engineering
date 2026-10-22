import tensorflow as tf
import os
import json
import pandas as pd
import numpy as np
from tqdm import tqdm

# Folder with the Waymo data (see data/README.md). Set WAYMO_DATA_DIR to use another folder.
DATA_DIR = os.environ.get('WAYMO_DATA_DIR', 'data')

# Path definitions
MOTION_DATA_PATH = os.path.join(DATA_DIR, 'motion/tf_example/training')
REASONING_DATA_PATH = os.path.join(DATA_DIR, 'reasoning/training')

def inspect_motion_scenario_ids(max_scenarios=1000):
    """Inspect the scenario IDs in the Motion Dataset."""
    print("Inspecting Motion Dataset Scenario IDs...")
    
    # Get TFRecord files
    tfrecord_files = [os.path.join(MOTION_DATA_PATH, f) for f in os.listdir(MOTION_DATA_PATH) 
                     if f.endswith('.tfrecord-00000-of-01000')]
    
    if not tfrecord_files:
        print("No TFRecord files found.")
        return []
    
    print(f"Found {len(tfrecord_files)} TFRecord files.")
    
    # Load records
    scenario_ids = []
    
    for file_idx, tfrecord_file in enumerate(tfrecord_files):
        print(f"Processing file {file_idx+1}/{len(tfrecord_files)}: {os.path.basename(tfrecord_file)}")
        
        raw_dataset = tf.data.TFRecordDataset(tfrecord_file)
        file_scenario_ids = []
        
        for i, raw_record in enumerate(tqdm(raw_dataset.take(max_scenarios))):
            try:
                example = tf.train.Example()
                example.ParseFromString(raw_record.numpy())
                
                # Get scenario ID
                scenario_id = example.features.feature['scenario/id'].bytes_list.value[0].decode('utf-8')
                file_scenario_ids.append(scenario_id)
                
            except Exception as e:
                print(f"Error processing scenario {i}: {str(e)}")
                continue
            
            if i >= max_scenarios:
                break
        
        print(f"Found {len(file_scenario_ids)} scenario IDs in file {file_idx+1}")
        scenario_ids.extend(file_scenario_ids)
    
    print(f"Found a total of {len(scenario_ids)} scenario IDs in Motion Dataset")
    print(f"First 5 scenario IDs: {scenario_ids[:5]}")
    print(f"Last 5 scenario IDs: {scenario_ids[-5:]}")
    
    return scenario_ids

def inspect_reasoning_scenario_ids():
    """Inspect all scenario IDs in the Reasoning Dataset."""
    print("\nInspecting Reasoning Dataset Scenario IDs...")
    
    # Get JSON files
    json_files = [os.path.join(REASONING_DATA_PATH, f) for f in os.listdir(REASONING_DATA_PATH) 
                 if f.endswith('.json')]
    
    if not json_files:
        print("No JSON files found.")
        return []
    
    print(f"Found {len(json_files)} JSON files.")
    
    # Extract scenario IDs
    scenario_ids = []
    
    for i, file_path in enumerate(tqdm(json_files)):
        try:
            with open(file_path, 'r') as f:
                data = json.load(f)
            
            scenario_id = data['sid']
            scenario_ids.append(scenario_id)
            
        except Exception as e:
            print(f"Error processing file {i+1}/{len(json_files)}: {str(e)}")
            continue
    
    print(f"Found a total of {len(scenario_ids)} scenario IDs in Reasoning Dataset")
    print(f"First 5 scenario IDs: {scenario_ids[:5]}")
    print(f"Last 5 scenario IDs: {scenario_ids[-5:]}")
    
    return scenario_ids

def check_scenario_id_overlap(motion_ids, reasoning_ids):
    """Check if there's any overlap between motion and reasoning scenario IDs."""
    motion_set = set(motion_ids)
    reasoning_set = set(reasoning_ids)
    
    common_ids = motion_set.intersection(reasoning_set)
    
    print(f"\nFound {len(common_ids)} overlapping scenario IDs out of {len(motion_ids)} motion IDs and {len(reasoning_ids)} reasoning IDs")
    
    if common_ids:
        print("First 10 overlapping IDs:")
        for i, sid in enumerate(list(common_ids)[:10]):
            print(f"{i+1}: {sid}")
    
    return common_ids

if __name__ == "__main__":
    motion_ids = inspect_motion_scenario_ids()
    reasoning_ids = inspect_reasoning_scenario_ids()
    
    if motion_ids and reasoning_ids:
        common_ids = check_scenario_id_overlap(motion_ids, reasoning_ids)
        
        # Save overlapping IDs to file if found
        if common_ids:
            with open('common_scenario_ids.txt', 'w') as f:
                for sid in common_ids:
                    f.write(f"{sid}\n")
            print(f"Saved {len(common_ids)} common scenario IDs to common_scenario_ids.txt") 