# Multimodal Feature Engineering on Waymo Motion and Reasoning Data

**Chapter 5 playbook** from *Data Engineering for Multimodal AI* (O'Reilly, 2026), pp. 164–172.
Book: https://oreil.ly/data-engineering-multimodal

> Want the code exactly as printed? Use the `first-edition` tag. This branch is the 2026 maintenance update; see [Differences from the printed book](#differences-from-the-printed-book). The printed yield model's near-perfect scores came from bugs, which are explained there.

## What you'll build

A local pipeline that aligns two modalities of the same driving scenes and fuses them into per-agent feature vectors:

1. **Structured motion data**: agent trajectories (position, speed, heading; 1 s of past, the current moment, 8 s of future at 10 Hz) from the Waymo Open Motion Dataset (WOMD), stored as TFRecord files of `tf.Example` protos.
2. **Language annotations**: Q&As about each scene from WOMD-Reasoning, one JSON file per scenario, in four groups: map environment (`env`), the ego vehicle's motion (`ego`), surrounding agents' motion (`sur`), and interactions and intentions over the next 3 seconds (`int`).

The stages, as the code implements them:

- **Inspect** both datasets and find the scenarios that appear in both (`inspect_waymo_data.py`, `data_inspection.py`, `validation_data_analysis.py`, `extract_validation_ids.py`, `analyze_common_scenarios.py`, `generate_validation_report.py`).
- **Extract motion features** per agent: current position, speed and heading, and distance to the ego vehicle (the self-driving car). In every matched scenario, the reasoning file's `ego` is the motion record's self-driving car.
- **Embed text** with Sentence-BERT (`all-mpnet-base-v2`, 768 dimensions), from the Q&As that describe the current moment (`env`, `ego`, `sur`). The `int` Q&As describe what agents *will* do, so they are kept in the feature store but are not model inputs.
- **Fuse** (early fusion): join on `scenario_id` and concatenate the motion features, question counts and text embedding into one row per agent (`combined_feature_pipeline.py`, `multimodal_feature_pipeline.py`).
- **Feature store (simulated)**: one JSON file per scenario and agent under `outputs/feature_store/`, plus metadata.
- **Downstream use case**: a logistic-regression model that predicts whether an agent will *yield*, meaning it is above 2 m/s now and drops below 2 m/s at some point in the next 8 s. The split is by scenario, and the model is compared with a majority-class baseline and a motion-only model.

`waymo_feature_pipeline.py` is a second, self-contained version of the same steps on one training shard, with its own yield definition (speed below 2 m/s within the next 2 s).

`gcp_implementation/` sketches how this could map onto Google Cloud. It is untested and optional; see [its README](gcp_implementation/README.md).

## Quick start

You need the Waymo data first (free, but you must register and accept Waymo's terms; see [Data and licenses](#data-and-licenses)):

```bash
gcloud auth login                 # the Google account you registered at waymo.com/open with
bash scripts/fetch_data.sh        # about 7 GB into ./data (or $WAYMO_DATA_DIR)
```

Then, as printed in the book (Python 3.12 or newer):

```bash
pip install -r requirements.txt
python inspect_waymo_data.py
python validation_data_analysis.py
python extract_validation_ids.py          # new: analyze_common_scenarios.py needs its output
python analyze_common_scenarios.py
python combined_feature_pipeline.py
python multimodal_feature_pipeline.py     # 10 scenarios, as printed
```

The printed default of 10 scenarios gives too few yielding agents for a meaningful test set. To train and evaluate on every matched scenario (776 with the default download, about 4 minutes on a laptop CPU):

```bash
python multimodal_feature_pipeline.py --max-scenarios 0
```

Using [uv](https://docs.astral.sh/uv/) instead: `uv sync`, then prefix each command with `uv run`, e.g. `uv run python multimodal_feature_pipeline.py`.

Run the end-to-end check, which uses synthetic data in the Waymo formats and needs no Waymo account: `make smoke`

The subfolder paths from the book (`python multimodal_pipeline/multimodal_feature_pipeline.py` and so on) still work. They run the root scripts.

## Tested on

| Date | OS | Python | Key libraries |
|---|---|---|---|
| 2026-10-22 | macOS 26.5.1 (arm64), CPU only | 3.12.15, 3.13.16 | tensorflow 2.21.0, torch 2.14.1, sentence-transformers 6.2.0, scikit-learn 1.9.1, umap-learn 0.5.12, numpy 2.5.4, pandas 3.1.0 |
| CI, on every push | ubuntu-latest | 3.12, 3.13 | same, from `uv.lock` (smoke test on synthetic data) |

Data: WOMD v1.3.1 `tf_example`, validation_interactive shards 00000–00009 and training shard 00000; WOMD-Reasoning as published in `gs://waymo_open_dataset_womd_reasoning/`.

## Differences from the printed book

Page references are to be confirmed against the printed book.

| What | Printed version | Now | Why |
|---|---|---|---|
| Yield labels | Read the future speed array as `[80, agents]` | Read as `[agents, 80]`, the WOMD layout | Each label mixed several agents' speeds. In three runs of the printed command, 72–92% of test agents were labeled yielding. With correct labels, 3.7% are |
| Text features | Embedding included the interaction Q&As (`int_a`, e.g. "surrounding agent #0 will yield to the ego agent"); `int_question_count` was an input | Only `env`, `ego` and `sur` Q&As; no `int_question_count` | The interaction answers describe the future that the label measures |
| Train/test split | Random split of agent rows | Split by scenario | Agents in one scenario share its text features |
| Reported result | Near-perfect accuracy (94–99% test accuracy in our three runs of the printed code) | Compared with baselines; see [Expected output](#expected-output) | With correct labels and no leakage, the model does not beat the motion-only model |
| `waymo_feature_pipeline.py` | Trajectories replaced by made-up values when an array lookup failed (it always did) | All trajectory arrays read per agent; features from past and current only; scenario split | Its labels and statistics came from the made-up values |
| Data in the repository | 126,357 WOMD-Reasoning files committed | Downloaded by each reader with `scripts/fetch_data.sh` | Waymo's terms allow sharing only with registered users |
| Run order | `analyze_common_scenarios.py` straight after `validation_data_analysis.py` | Run `extract_validation_ids.py` first | `analyze_common_scenarios.py` reads its output |
| Install | Python 3.8+, unpinned, no `seaborn` | Python 3.12+, pinned from `uv.lock`; on Linux, PyTorch comes from the CPU-only index (`requirements.txt` adds it) | `seaborn` is imported; old pins don't install on 3.12+. PyPI's Linux PyTorch build includes Triton, which crashed the scripts when loaded after TensorFlow (seen in CI), and adds several GB of CUDA libraries |
| Scenario selection | "First 10" common scenarios in set order (changed between runs) | Sorted order; `--max-scenarios` | Reproducible runs |
| Subfolder scripts | Copies of the root scripts | Wrappers that run the root scripts | One copy to maintain; both paths work |

The full list is in [CHANGELOG.md](CHANGELOG.md).

## Data and licenses

No data is included in this repository. Each reader downloads their own copy; [`data/README.md`](data/README.md) has the details and layout.

| Data | Source | License |
|---|---|---|
| Waymo Open Motion Dataset v1.3.1 (`tf_example`) | [waymo.com/open](https://waymo.com/open/), `gs://waymo_open_dataset_motion_v_1_3_1/` | [Waymo Dataset License Agreement for Non-Commercial Use](https://waymo.com/open/terms/). Registration required |
| WOMD-Reasoning (Li et al., [arXiv 2407.04281](https://arxiv.org/abs/2407.04281)) | [waymo.com/open](https://waymo.com/open/), `gs://waymo_open_dataset_womd_reasoning/` | Waymo terms above, plus its own LICENSE (UC Regents; educational, research and not-for-profit use) |
| Synthetic test data | Generated by `tests/make_fixture.py` at test time; not real driving data | MIT, like the code |
| `docs/img/first-edition/*.png` | Charts produced by the first-edition scripts | MIT, like the code |

Don't commit the Waymo data or share it with people who haven't registered. `data/motion/` and `data/reasoning/` are in `.gitignore`.

## Expected output

From my run on 2026-10-22: macOS 26.5.1 (arm64), Python 3.12.15, a fresh clone installed with `pip install -r requirements.txt` (67 s), and the default download from `scripts/fetch_data.sh`. Python 3.13.16 gave the same numbers. Progress bars and library warnings are left out.

`python extract_validation_ids.py` (14 s):

```
Found a total of 2929 scenario IDs in Motion validation dataset
Found 11317 unique scenario IDs in Reasoning validation dataset
  Common IDs: 776
  Motion-only IDs: 2153
  Reasoning-only IDs: 10541
```

`python validation_data_analysis.py` (5 s):

```
  total_scenarios: 11317
  total_questions: 512459
```

`python multimodal_feature_pipeline.py` (10 scenarios as printed, 37 s). Only 4 agents yield, and none of them are in the test scenarios, so the scores are meaningless; the script says so:

```
Processed 10 scenarios, 10 successful.
Created fused features DataFrame with 192 agent entries.
Uploaded 192 agent features to feature store at outputs/feature_store/agent_features

Training yield prediction model...
Using 191 agents from 10 scenarios for yield prediction model.
Yielding agents: 4 (2.1%)
Adding 768 text embedding dimensions to model features.
Train: 141 agents from 7 scenarios. Test: 50 agents from 3 scenarios.
WARNING: the test scenarios contain only one class, so the scores below don't measure anything. Process more scenarios, e.g. --max-scenarios 0 for all of them.
```

`python multimodal_feature_pipeline.py --max-scenarios 0` (all 776 matched scenarios, 214 s):

```
Processed 776 scenarios, 776 successful.
Created fused features DataFrame with 20992 agent entries.

Training yield prediction model...
Using 20733 agents from 776 scenarios for yield prediction model.
Yielding agents: 760 (3.7%)
Adding 768 text embedding dimensions to model features.
Train: 14144 agents from 543 scenarios. Test: 6589 agents from 233 scenarios.
Model training complete.
Train accuracy: 0.8072
Test accuracy: 0.7211

Test set comparison:
model                       accuracy  balanced acc  F1 (yielding)
majority_class_baseline        0.965         0.500          0.000
motion_only                    0.783         0.686          0.157
motion_plus_reasoning          0.721         0.547          0.082
```

How to read this: yielding is rare (3.7%), so accuracy is misleading; always predicting "not yielding" scores 0.965. On balanced accuracy and F1, motion features alone beat the fused model: adding 768 scene-level text dimensions to a linear model on this data hurts. That is the honest result of this setup. It is a starting point for the chapter's question of when language features help, not evidence that they do.

`python waymo_feature_pipeline.py` (one training shard, 10 scenarios, 75 s; its label is "below 2 m/s within 2 s", so parked agents count as yielding):

```
Class distribution - Yielding: 189, Non-yielding: 90
Majority-class baseline accuracy on the test set: 0.787

Model Performance:
              precision    recall  f1-score   support

           0       0.88      0.35      0.50        20
           1       0.85      0.99      0.91        74

    accuracy                           0.85        94
```

`make smoke`: `13 passed` in 3 min 16 s (synthetic data; numbers differ from the real data).

The other scripts ran without errors: `inspect_waymo_data.py` 75 s, `data_inspection.py` 17 s, `analyze_common_scenarios.py` 22 s, `generate_validation_report.py` 1 s, `combined_feature_pipeline.py` 23 s.

## Project structure

```
playbook-05-multimodal-feature-engineering/
├── inspect_waymo_data.py           # first look at both datasets (training split)
├── data_inspection.py              # scenario IDs in both datasets (training split)
├── validation_data_analysis.py     # question statistics (validation_interactive)
├── extract_validation_ids.py       # scenario IDs in both datasets (validation_interactive)
├── analyze_common_scenarios.py     # Q&A analysis of the common scenarios
├── generate_validation_report.py   # Markdown report from the analysis outputs
├── combined_feature_pipeline.py    # motion + reasoning features per scenario
├── multimodal_feature_pipeline.py  # text embeddings, fusion, feature store, yield model
├── waymo_feature_pipeline.py       # self-contained variant on one training shard
├── data_inspection/, feature_extraction/, multimodal_pipeline/
│                                   # wrappers that run the root scripts, plus READMEs
├── gcp_implementation/             # untested Google Cloud sketch (optional)
├── scripts/fetch_data.sh           # downloads the Waymo data (needs registration)
├── data/README.md                  # data sources, licenses and layout
├── tests/                          # synthetic Waymo-format data and smoke tests
├── docs/img/first-edition/         # charts from the first-edition run
├── outputs/                        # written by the scripts (git-ignored)
├── pyproject.toml, uv.lock, requirements.txt
└── Makefile                        # make smoke, make data
```

## License

MIT. See [LICENSE](LICENSE). The Waymo data is not covered by this license; see [Data and licenses](#data-and-licenses).

## Waymo Dataset Usage

This playbook was made using the Waymo Open Dataset, provided by Waymo LLC under the Waymo Dataset License Agreement for Non-Commercial Use, available at waymo.com/open/terms, and your access and use of such work are governed by the terms and conditions therein.
