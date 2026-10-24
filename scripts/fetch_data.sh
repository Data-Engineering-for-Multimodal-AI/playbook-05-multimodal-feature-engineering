#!/usr/bin/env bash
# Download the Waymo data this playbook uses into ./data (or $WAYMO_DATA_DIR).
#
# Before running it:
#   1. Register at https://waymo.com/open and accept the dataset terms with a Google account.
#   2. Install the Google Cloud CLI and sign in with that same account: gcloud auth login
#
# Usage: bash scripts/fetch_data.sh [NUM_SHARDS]
#   NUM_SHARDS  validation_interactive motion shards to fetch (default 10 of 150, about 5 GB).
# The training motion shard (1.1 GB) is only used by inspect_waymo_data.py, data_inspection.py
# and waymo_feature_pipeline.py. Set WITH_TRAINING=0 to skip it.
#
# The data is licensed to you by Waymo. Don't commit it or share it with people who haven't registered.
set -euo pipefail

DATA_DIR="${WAYMO_DATA_DIR:-data}"
NUM_SHARDS="${1:-10}"
WITH_TRAINING="${WITH_TRAINING:-1}"
MOTION="gs://waymo_open_dataset_motion_v_1_3_1/uncompressed/tf_example"
REASONING="gs://waymo_open_dataset_womd_reasoning"

# Sliced (parallel) downloads failed with a hash mismatch on macOS in our test; plain downloads work.
export CLOUDSDK_STORAGE_SLICED_OBJECT_DOWNLOAD_THRESHOLD=0

mkdir -p "$DATA_DIR/motion/tf_example/validation_interactive" "$DATA_DIR/reasoning"

echo "Motion validation_interactive: $NUM_SHARDS shard(s)"
for ((i = 0; i < NUM_SHARDS; i++)); do
  f=$(printf "validation_interactive_tfexample.tfrecord-%05d-of-00150" "$i")
  [ -f "$DATA_DIR/motion/tf_example/validation_interactive/$f" ] || \
    gcloud storage cp "$MOTION/validation_interactive/$f" "$DATA_DIR/motion/tf_example/validation_interactive/"
done

if [ "$WITH_TRAINING" = "1" ]; then
  echo "Motion training: shard 00000"
  mkdir -p "$DATA_DIR/motion/tf_example/training"
  f="training_tfexample.tfrecord-00000-of-01000"
  [ -f "$DATA_DIR/motion/tf_example/training/$f" ] || \
    gcloud storage cp "$MOTION/training/$f" "$DATA_DIR/motion/tf_example/training/"
fi

echo "WOMD-Reasoning annotations"
for split in validation_interactive training; do
  if [ ! -d "$DATA_DIR/reasoning/$split" ]; then
    gcloud storage cp "$REASONING/$split.tar.gz" "$DATA_DIR/reasoning/"
    tar -xzf "$DATA_DIR/reasoning/$split.tar.gz" -C "$DATA_DIR/reasoning"
    rm "$DATA_DIR/reasoning/$split.tar.gz"
  fi
done
gcloud storage cp "$REASONING/LICENSE" "$DATA_DIR/reasoning/LICENSE"

echo "Done. Data is in $DATA_DIR"
