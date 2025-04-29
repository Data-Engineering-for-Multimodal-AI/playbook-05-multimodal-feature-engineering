# Data Directory

This directory should contain the Waymo Motion and Reasoning datasets required for the pipeline. Due to their large size, the actual data files are not included in the repository.

## Data Structure

The expected directory structure is:

```
data/
├── motion/
│   └── tf_example/
│       └── validation_interactive/
│           ├── <tfrecord files for motion data>
│           └── ...
└── reasoning/
    └── validation_interactive/
        ├── <tfrecord files for reasoning data>
        └── ...
```

## How to Obtain the Data

1. Visit the [Waymo Open Dataset website](https://waymo.com/open/)
2. Follow the instructions to download:
   - The Motion Dataset (validation interactive split)
   - The Reasoning Dataset (validation interactive split)
3. Place the downloaded TFRecord files in the appropriate directories shown above

## Data Format

### Motion Dataset

- Format: TFRecord files containing tf.Example protos
- Key features: scenario IDs, agent trajectories, agent types, and motion states
- File pattern: typically named like `validation_interactive.tfrecord-XXXXX-of-YYYYY`

### Reasoning Dataset

- Format: TFRecord files containing tf.Example protos
- Key features: scenario IDs, questions, answers, and ego agent IDs
- File pattern: typically named like `validation_interactive.tfrecord-XXXXX-of-YYYYY`

## Data Processing

Once you have the data in place, you can run the scripts in the `data_inspection` directory to explore and understand the datasets before proceeding with feature extraction and multimodal processing. 