# Data Directory

This playbook uses two Waymo datasets. Neither is included in this repository: the Waymo terms allow sharing them only with people who have registered, so each reader downloads their own copy.

| Dataset | What it is | Where it comes from | License |
|---|---|---|---|
| Waymo Open Motion Dataset (WOMD) v1.3.1, `tf_example` format | Agent trajectories as TFRecord files of `tf.Example` protos | `gs://waymo_open_dataset_motion_v_1_3_1/uncompressed/tf_example/` | [Waymo Dataset License Agreement for Non-Commercial Use](https://waymo.com/open/terms/) |
| WOMD-Reasoning | About 3 million language Q&As about WOMD scenarios, one JSON file per scenario and agent ([paper](https://arxiv.org/abs/2407.04281), [code](https://github.com/yhli123/WOMD-Reasoning)) | `gs://waymo_open_dataset_womd_reasoning/` | Waymo terms above, plus its own `LICENSE` (UC Regents; educational, research and not-for-profit use) |

## How to get the data

1. Register at [waymo.com/open](https://waymo.com/open/) and accept the dataset terms with a Google account.
2. Install the [Google Cloud CLI](https://cloud.google.com/sdk/docs/install) and sign in with the same account: `gcloud auth login`.
3. From the repository root, run:

```bash
bash scripts/fetch_data.sh        # 10 of 150 validation_interactive shards (about 5 GB), 1 training shard (1.1 GB), and the annotations (about 1 GB unpacked)
bash scripts/fetch_data.sh 2      # fewer motion shards (about 1 GB) for a quicker first try
```

If you prefer the browser, the same files are listed at [waymo.com/open/download](https://waymo.com/open/download/). Put them in the layout below.

To keep the data somewhere else, set `WAYMO_DATA_DIR` before running the download script and the pipeline scripts, for example `export WAYMO_DATA_DIR=~/waymo-data`.

## Layout

```
data/
├── motion/
│   └── tf_example/
│       ├── validation_interactive/   # validation_interactive_tfexample.tfrecord-00000-of-00150 ...
│       └── training/                 # training_tfexample.tfrecord-00000-of-01000 (inspection scripts only)
└── reasoning/
    ├── validation_interactive/       # scid_<scenario>__aid_<agent>__atype_<type>.json (11,317 files)
    ├── training/                     # same format (51,854 files)
    └── LICENSE
```

`data/motion/` and `data/reasoning/` are listed in `.gitignore`, so the data can't be committed by accident.

## How the two datasets line up

The reasoning files name the WOMD scenario (`sid`) and the ego agent (`ego`). In the 10 validation_interactive shards that `fetch_data.sh` downloads by default, 776 of the 2,929 motion scenarios have a reasoning file, and in all 776 the reasoning `ego` is the self-driving car (`state/is_sdc`) of the motion record. Checked on 2026-10-22.

## Waymo Dataset Usage

This playbook was made using the Waymo Open Dataset, provided by Waymo LLC under the Waymo Dataset License Agreement for Non-Commercial Use, available at waymo.com/open/terms, and your access and use of such work are governed by the terms and conditions therein.
