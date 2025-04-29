import tensorflow as tf
import os
import json
import pandas as pd
import numpy as np
from tqdm import tqdm

# Path definitions
MOTION_DATA_PATH = 'data/motion/tf_example/training'
REASONING_DATA_PATH = 'data/reasoning/training'

def inspect_tfrecord_sample():
    """Load and inspect a small TFRecord sample from the Waymo Motion Dataset"""
    print("Inspecting TFRecord sample from Waymo Motion Dataset...")
    
    # Get first TFRecord file
    tfrecord_files = [os.path.join(MOTION_DATA_PATH, f) for f in os.listdir(MOTION_DATA_PATH) 
                     if f.endswith('.tfrecord-00000-of-01000')]
    
    if not tfrecord_files:
        print("No TFRecord files found in the specified path.")
        return
        
    # Load first record
    raw_dataset = tf.data.TFRecordDataset(tfrecord_files[0])
    
    # Take only first example
    for raw_record in raw_dataset.take(1):
        example = tf.train.Example()
        example.ParseFromString(raw_record.numpy())
        
        # Print feature keys
        feature = example.features.feature
        print("\nFeature keys in TFRecord:")
        for key in feature:
            print(f"- {key}")
            
        # Print more detailed info about a specific feature if needed
        if 'scenario/id' in feature:
            scenario_id = feature['scenario/id'].bytes_list.value[0].decode('utf-8')
            print(f"\nScenario ID: {scenario_id}")

def inspect_reasoning_sample():
    """Load and inspect a sample from the Waymo Reasoning Dataset"""
    print("\nInspecting sample from Waymo Reasoning Dataset...")
    
    # Get first JSON file
    json_files = [os.path.join(REASONING_DATA_PATH, f) for f in os.listdir(REASONING_DATA_PATH) 
                 if f.endswith('.json')]
    
    if not json_files:
        print("No JSON files found in the specified path.")
        return
    
    # Load first JSON file
    with open(json_files[0], 'r') as f:
        data = json.load(f)
    
    # Print structure
    print("\nKeys in Reasoning dataset:")
    for key in data.keys():
        print(f"- {key}")
    
    # Print an example Q&A pair
    if 'ego_q' in data and 'ego_a' in data:
        print("\nExample Q&A pair about ego vehicle:")
        for q, a in zip(data['ego_q'][:2], data['ego_a'][:2]):  # just first 2 pairs
            print(f"Q: {q}")
            print(f"A: {a}")

if __name__ == "__main__":
    inspect_tfrecord_sample()
    inspect_reasoning_sample() 