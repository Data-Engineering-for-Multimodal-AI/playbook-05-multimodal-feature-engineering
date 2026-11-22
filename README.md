# Playbook 05: Turn driving data and its explanations into one set of features
**How to build a multimodal feature-engineering pipeline with the Waymo Open Dataset.** Chapter 5 of [*Data Engineering for Multimodal AI*](https://oreil.ly/data-engineering-multimodal) (O'Reilly, 2026), pages 164–171.

A self-driving car's motion data and the text that explains its decisions come in very different forms, and a model needs them as one set of features. The chapter's running example is BayGo, a fictional autonomous vehicle company; in this repo, we use the Waymo Open Dataset. Follow along to extract features from the motion tracks and the question-and-answer annotations, fuse them, and store them in a feature store ready for a model.

**The problem.** Models need numbers, but driving data comes as motion tracks and as written explanations, and the two have to become one feature set.

**What you'll work with.** You'll extract features from each modality, fuse them early into one vector per example, and store the result in a feature store that a model can read from.

> Following the book? Its commands are in [Run it the way the book does](#run-it-the-way-the-book-does) and still work as printed. Want the code exactly as printed? Use the `first-edition` tag. This branch is the 2026 maintenance update; see [Differences from the printed book](#differences-from-the-printed-book). The printed yield model's near-perfect scores came from bugs, which are explained there.

## Run it

You need [uv](https://docs.astral.sh/uv/) and `make`. No Docker, no API keys. The Waymo data is free but needs a registration at [waymo.com/open](https://waymo.com/open/) and the [Google Cloud CLI](https://cloud.google.com/sdk/docs/install) to download it; without it, `make run` uses synthetic data in the same formats. Or open the repo in a GitHub Codespace, where uv and `make` are ready. On Windows without `make`, use a Codespace or the book's steps below.

**1. Get the code.** About 3 MB; the Python environment is 2.4 GB once installed.

```bash
git clone https://github.com/Data-Engineering-for-Multimodal-AI/playbook-05-multimodal-feature-engineering.git
cd playbook-05-multimodal-feature-engineering
```

**2. Run the pipeline on synthetic data.** With no Waymo data in `data/`, this installs the locked dependencies (`uv sync`), writes a small synthetic dataset in the Waymo formats (`python tests/make_fixture.py`) and runs the book's `python multimodal_feature_pipeline.py` on it. The first run downloads the Sentence-BERT model (about 420 MB). On a Mac the first run took 5 min 22 s, most of it creating the environment; a second run took 1 min 7 s.

```bash
make run
```

You should see 10 scenarios fused, a feature store written and a model trained, then a note that the data is synthetic:

```
Processed 10 scenarios, 10 successful.
Created fused features DataFrame with 70 agent entries.
Uploaded 70 agent features to feature store at outputs/feature_store/agent_features
...
NOTE: the output above is from SYNTHETIC data (tests/make_fixture.py), not real driving data.
```

The synthetic numbers only show that the code runs. Its outputs are in `synthetic/outputs/`.

**3. Run it on the Waymo data.** Register at [waymo.com/open](https://waymo.com/open/) and accept the dataset terms, then download about 6.5 GB into `data/` (`bash scripts/fetch_data.sh`; see [data/README.md](data/README.md)) and run again. With the data in place, `make run` runs `python multimodal_feature_pipeline.py` on all 776 scenarios that have both motion and reasoning data. On a Mac that was busy with other work this took 13 min 16 s, 2 min 45 s of it writing the feature store.

```bash
gcloud auth login
bash scripts/fetch_data.sh
make run
```

```
Processed 776 scenarios, 776 successful.
Yielding agents: 760 (3.7%)
...
model                       accuracy  balanced acc  F1 (yielding)
majority_class_baseline        0.965         0.500          0.000
motion_only                    0.783         0.686          0.157
motion_plus_reasoning          0.721         0.547          0.082
```

Adding the text features makes the model worse on balanced accuracy and F1. [Expected output](#expected-output) explains how to read this, and [Try this](#try-this) looks for the reasons. To keep the data elsewhere, set `WAYMO_DATA_DIR` (for example `make run WAYMO_DATA_DIR=~/waymo-data`).

**4. Look at one agent's features.** Prints one entry of the feature store next to the Q&As it was built from. By default it picks the first agent labeled yielding:

```bash
make show
```

```
outputs/feature_store/agent_features/1008aa4114dbc237__3007.json  (20992 entries in the store)

  speed                  2.312948703765869
  distance_to_ego        17.399019241333008
  is_yielding            True
  text_embedding         [768 numbers: 0.0095, -0.0639, ...]
...
  int (interactions (not a model input: they describe the future)), 4 Q&As:
    Q: What interactions will occur between the ego agent and surrounding agent #0?
    A: Surrounding agent #0 will yield to the ego agent because they are both at the intersection with stop signs, ...
```

Every agent in a scenario gets the same text embedding, built from the scene's Q&As. Pick another with `make show SCENARIO=... AGENT=...`.

**5. Clean up.**

```bash
make clean
```

This deletes `synthetic/`, everything in `outputs/` except its README (the feature store and the experiments' cache), and the test cache. The Waymo data in `data/` and the downloaded model stay.

## How much time do you have?

| Time | What to do |
|---|---|
| 15 minutes | [Run it](#run-it), steps 1 and 2, on synthetic data. On a Mac, step 2 took 5 min 22 s the first time. |
| 1 hour | Register for the Waymo data, download it and run steps 3 and 4 (13 min 16 s for step 3 on a busy Mac, plus the download). |
| 1 day | The three [Try this](#try-this) experiments (the first one builds the features in about 16 minutes on a Mac; the others take under 3 minutes), [Use your own data](#use-your-own-data), and [Before production](#before-production). |

## Try this

Each experiment changes one thing and measures the effect. They use the pipeline's own code (`MultimodalFeaturePipeline`) to build the fused features once, cache them in `outputs/experiments/`, and train the pipeline's model (scaled logistic regression with balanced class weights) on the pipeline's split by scenario. They use the Waymo data when it is in `data/` (or `WAYMO_DATA_DIR`) and synthetic data otherwise, and say which. The numbers below are from the Waymo data on a Mac on 2026-11-21. Building the features took 16 min 33 s; each experiment after that took under 3 minutes.

**1. Does the text's "yield" mean the label's?** The interaction Q&As describe what agents will do next, the thing the label measures, so the pipeline leaves them out of the model. This trains the model with the first edition's text (interaction Q&As included), and with one flag per agent: an interaction answer says this agent will yield.

```bash
make try-leakage
```

```
An int answer says 948 agents will yield. 10% of them are labeled yielding (all agents: 3.7%),
and 61% of them are already below 2 m/s, which the label never counts as yielding.
Of the 760 agents labeled yielding, 13% are named in an int answer.

Features                         inputs balanced acc F1 (yielding)
motion only                           2        0.686         0.157
+ text (as the pipeline)            773        0.547         0.082
+ text with interaction Q&As        773        0.556         0.084
+ "will yield" in an int answer       3        0.678         0.146
```

On the Waymo data the interaction answers barely move the scores, because the text and the label mean different things: most agents the answers say will yield are already slow (61% below 2 m/s), and the label needs an agent above 2 m/s that slows below it. On the synthetic data, whose answers are written from the same braking the label measures, the same flag shows what a leak looks like:

```
+ "will yield" in an int answer       3        0.930         0.912
```

Before you use text as features or labels, check that it measures the same thing. Change the feature sets in `experiments/leakage.py` and run it again.

**2. Split by scenario, or by agent?** Every agent in a scenario shares its text embedding and question counts. This trains the same model (motion and text, as the pipeline) with the pipeline's split by scenario and with the first edition's random split of agent rows, over five seeds.

```bash
make try-split
```

```
Split                              balanced acc F1 (yielding)  test agents whose scenario is in training
by scenario (as the pipeline)             0.560         0.091                                        0%
random agent rows (first edition)         0.687         0.164                                      100%
```

With a random split, every test agent has neighbors from its own scene in training, with the same text vector, and the scores rise from 0.560 to 0.687 without the model getting better. Split by the unit that shares features.

**3. How many text dimensions should the fused vector have?** Early fusion puts 2 motion features and 3 question counts next to a 768-dimensional embedding. This shrinks the embedding with PCA, fitted on the training scenarios only, and trains the model again.

```bash
make try-text-dims
```

```
text dimensions inputs text share balanced acc F1 (yielding)
              0      5         0%        0.693         0.157
              2      7        29%        0.693         0.157
              8     13        62%        0.708         0.155
             32     37        86%        0.654         0.125
            128    133        96%        0.618         0.111
            768    773        99%        0.547         0.082
```

Eight text dimensions give the best balanced accuracy, 0.708; all 768 give 0.547, and every step past 8 made both scores worse. The embedding has only 776 distinct values, one per scene, for 20,733 agents. Change `DIMS` in `experiments/text_dims.py` and run it again.

## Use your own data

Put your scenarios in a folder with the Waymo layout from [data/README.md](data/README.md). A subset of the Waymo data, another split renamed to `validation_interactive`, or your own scenes converted to the same formats all work:

```
my_scenarios/
  motion/tf_example/validation_interactive/   *.tfrecord* files of WOMD tf.Example records
  reasoning/validation_interactive/           scid_<scenario>__aid_<agent>__atype_<type>.json
```

Each `tf.Example` needs `scenario/id`, `state/id`, `state/type`, `state/is_sdc` (exactly one self-driving car), `state/current/*` with one value per agent and `state/future/speed` and `state/future/valid` with 80 values per agent. Each JSON file needs `sid`, `ego`, `cur_time`, `future_time` and the `env`, `ego`, `sur` and `int` question and answer lists.

Check it first. This reads every file and writes nothing:

```bash
make check-data DIR=my_scenarios
```

It reports every file the pipeline would crash on or silently drop. On a test folder with mistakes in it:

```
my_scenarios: 6 motion scenarios, 1 with reasoning files, 1 in both

8 problem(s):
  motion/tf_example/validation_interactive/validation_interactive_tfexample.tfrecord-00001-of-00150: can't read past record 0: DataLossError (truncated or not a TFRecord)
  motion/tf_example/validation_interactive/validation_interactive_tfexample.tfrecord-00002-of-00150: Git LFS pointer, not the file
  reasoning/validation_interactive/notes.json: name must look like scid_<scenario>__aid_<agent>__atype_<type>.json; the pipeline finds files by that prefix
  reasoning/validation_interactive/scid_val0000000000000__aid_100__atype_1.json: ego 999 is not the self-driving car (100) of scenario val0000000000000; the pipeline would drop the scenario
  reasoning/validation_interactive/scid_val0000000000001__aid_110__atype_1.json: missing cur_time
  reasoning/validation_interactive/scid_val0000000000001__aid_110__atype_1.json: 6 sur_q questions but 5 sur_a answers
  reasoning/validation_interactive/scid_val0000000000002__aid_120__atype_1.json: not valid JSON (line 1)
  reasoning/validation_interactive/scid_val00000000000ff__aid_130__atype_1.json: file name says scenario val00000000000ff, the file says val0000000000003

Fix or remove these files and run again. Nothing was written.
```

It also catches arrays of the wrong length, JSON that isn't UTF-8, and folders with fewer than two matched scenarios. On the full Waymo download it took 35 s and passed, with a note that 10,541 reasoning files have no motion record (expected, since the download has 10 of 150 motion shards). When everything passes, run the pipeline on it:

```bash
make load DIR=my_scenarios
make show DIR=my_scenarios
```

```
All files passed the checks.

$ WAYMO_DATA_DIR=my_scenarios python multimodal_feature_pipeline.py
...
Uploaded 28 agent features to feature store at outputs/feature_store/agent_features
```

`make load` runs the pipeline with `WAYMO_DATA_DIR` set to your folder and writes to `outputs/`, as with the Waymo data. It first deletes `outputs/feature_store/` and the cached list of scenario IDs, which the pipeline would otherwise mix with the last run's (see [Before production](#before-production)). It took 49 s for four synthetic scenarios on a Mac.

## What you end up with

A feature store of fused motion and language features for your scenarios, a checker for the Waymo formats, and three measurements of how fusion choices change what a model learns.

Skills you'll practice:
- Extracting per-agent features from TFRecords of `tf.Example` protos
- Embedding language annotations with Sentence-BERT
- Joining two modalities on a scenario ID and an ego agent, and fusing them early
- Splitting train and test data by the unit that shares features
- Checking features for leakage from the future
- Validating input files before they reach the pipeline

## Before production

This playbook runs on 776 scenarios and a linear model. Before you build something like it for real:

1. **Check that the text and the label measure the same thing.** The interaction answers say 948 agents will yield; 10% of them meet the label's definition, and 61% are already below 2 m/s ([Try this 1](#try-this)). Write the label's definition down, and test any text feature against it before you use it. *Chapter 5, "Cross-Cutting Considerations in Fusion" (page 153).*
2. **Fuse at the grain of the prediction.** The text embedding describes a scene and is copied to every agent in it, about 27 per scene. A random split of agent rows raised balanced accuracy from 0.560 to 0.687 without a better model ([Try this 2](#try-this)). Keep the split by scenario, and look for agent-level text (the `sur` answers name agents). *Chapter 5, "Why Fusion Is Not a Join" (page 144).*
3. **Balance the modalities.** The 768 text dimensions are 99% of the model's inputs; reduced to 8 they gave 0.708 balanced accuracy against 0.547 ([Try this 3](#try-this)). Reduce or project each modality before concatenating, or fuse later. *Chapter 5, "Fusion Architectures" (page 145).*
4. **Embed the Q&As you mean to.** `embed_qa_text` keeps the first 10 Q&As in the order env, ego, sur. In 246 of the 776 scenarios that leaves no `sur` Q&A at all, and only 1,467 of the 23,575 `sur` Q&As are embedded. Choose the text per agent or per group, on purpose. *Chapter 5, "Extracting Features Across Modalities" (page 139).*
5. **Use a feature store that replaces what it holds.** The simulated store writes one JSON file per agent (20,992 files, 2 min 45 s on a Mac) and never removes old ones: after a second run on 4 scenarios it held 21,020 files while its metadata said 28. The files also contain `NaN`, which strict JSON readers reject. *Chapter 5, "Implementing and Managing Cloud-Based Multimodal Feature Stores" (page 154).*
6. **Key cached results to their input.** `load_common_scenario_ids` reuses `outputs/validation_analysis/scenario_id_comparison.json` from any earlier run. Pointed at other data, the pipeline looks for the old scenario IDs and finds none, as the experiments here did until they started from a clean folder. Store the data's location or a checksum with the cache. *Chapter 8, "Data Versioning and Experiment Tracking for Multimodal AI" (page 276).*
7. **Keep the license with the features.** The Waymo terms allow non-commercial use only and sharing only with registered users, and features derived from the data are covered too. Record the source and terms with every feature set. *Chapter 11, "Compliance Across Modalities" (page 417).*

## What it costs

| Where you run it | Cost |
|---|---|
| Your machine | $0. Everything runs locally: no API keys, no paid services. The Waymo data is free with a registration. About 9.5 GB of disk: 2.4 GB for the Python environment, 420 MB for the model, 6.5 GB for the Waymo data and up to 520 MB for the experiments' cache. |
| A GitHub Codespace (2 cores, 8 GB, the default size; the repo asks for nothing larger) | Estimated $0.18 per hour at GitHub's list price (November 2026). A free personal account includes 120 core-hours a month, about 60 hours on this machine. In CI, `make run` on synthetic data took 84 s in a freshly built devcontainer, including the model download. |

`gcp_implementation/` sketches a Google Cloud version; it is untested, and this README doesn't estimate its cost.

## How it works

The pipeline aligns two modalities of the same driving scenes and fuses them into per-agent feature vectors:

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

## Run it the way the book does

These are the book's steps (pages 164–171; it runs `multimodal_feature_pipeline.py` on page 167), as printed apart from the fixes in [Differences from the printed book](#differences-from-the-printed-book). `make run` runs the last one.

You need the Waymo data first (free, but you must register and accept Waymo's terms; see [Data and licenses](#data-and-licenses)):

```bash
gcloud auth login                 # the Google account you registered at waymo.com/open with
bash scripts/fetch_data.sh        # about 7 GB into ./data (or $WAYMO_DATA_DIR)
```

Then (Python 3.12 or newer; the book, page 167, runs `multimodal_feature_pipeline.py`):

```bash
pip install -r requirements.txt
python inspect_waymo_data.py
python validation_data_analysis.py
python extract_validation_ids.py          # new: analyze_common_scenarios.py needs its output
python analyze_common_scenarios.py
python combined_feature_pipeline.py
python multimodal_feature_pipeline.py     # now uses every matched scenario (see below)
```

`multimodal_feature_pipeline.py` now processes every scenario found in both datasets by default: 776 with the default download, about 4–5 minutes on a laptop CPU. The first edition processed 10, which leaves too few yielding agents to test on. For a quick run like the first edition's, use `--max-scenarios 10`.

Using [uv](https://docs.astral.sh/uv/) instead: `uv sync`, then prefix each command with `uv run`, e.g. `uv run python multimodal_feature_pipeline.py`.

Run the end-to-end check, which uses synthetic data in the Waymo formats and needs no Waymo account: `make smoke` (the first run downloads the public Sentence-BERT models, about 0.5 GB)

The first edition's subfolder paths (`python multimodal_pipeline/multimodal_feature_pipeline.py` and so on) still work. They run the root scripts.

## Tested on

| Date | OS | Python | Key libraries |
|---|---|---|---|
| 2026-10-22 | macOS 26.5.1 (arm64), CPU only | 3.12.15, 3.13.16 | tensorflow 2.21.0, torch 2.14.1, sentence-transformers 6.2.0, scikit-learn 1.9.1, umap-learn 0.5.12, numpy 2.5.4, pandas 3.1.0 |
| CI, on every push | ubuntu-latest | 3.12, 3.13 | same, from `uv.lock` (smoke test on synthetic data) |

Data: WOMD v1.3.1 `tf_example`, validation_interactive shards 00000–00009 and training shard 00000; WOMD-Reasoning as published in `gs://waymo_open_dataset_womd_reasoning/`.

## Differences from the printed book

The "First edition" column is the code at the `first-edition` tag, which the book describes.

| What | First edition | Now | Why |
|---|---|---|---|
| Yield labels | Read the future speed array as `[80, agents]` | Read as `[agents, 80]`, the WOMD layout | Each label mixed several agents' speeds. In three runs of the first-edition command, 72–92% of test agents were labeled yielding. With correct labels, 3.7% are |
| Text features | Embedding included the interaction Q&As (`int_a`, e.g. "surrounding agent #0 will yield to the ego agent"); `int_question_count` was an input | Only `env`, `ego` and `sur` Q&As; no `int_question_count` | The interaction answers describe the future that the label measures |
| Train/test split | Random split of agent rows | Split by scenario | Agents in one scenario share its text features |
| Reported result | Near-perfect accuracy (the book, pages 167 and 169: 100%; 94–99% test accuracy in our three runs of the first-edition code) | Compared with baselines; see [Expected output](#expected-output) | With correct labels and no leakage, the model does not beat the motion-only model |
| `waymo_feature_pipeline.py` | Trajectories replaced by made-up values when an array lookup failed (it always did) | All trajectory arrays read per agent; features from past and current only; scenario split | Its labels and statistics came from the made-up values |
| Data in the repository | 126,357 WOMD-Reasoning files committed | Downloaded by each reader with `scripts/fetch_data.sh` | Waymo's terms allow sharing only with registered users |
| Run order | `analyze_common_scenarios.py` straight after `validation_data_analysis.py` | Run `extract_validation_ids.py` first | `analyze_common_scenarios.py` reads its output |
| Install | Python 3.8+, unpinned, no `seaborn` | Python 3.12+, pinned from `uv.lock`; on Linux, PyTorch comes from the CPU-only index (`requirements.txt` adds it) | `seaborn` is imported; old pins don't install on 3.12+. PyPI's Linux PyTorch build includes Triton, which crashed the scripts when loaded after TensorFlow (seen in CI), and adds several GB of CUDA libraries |
| Scenario selection | 10 common scenarios by default, in set order (changed between runs) | All common scenarios by default, sorted; `--max-scenarios N` for fewer | With 10 scenarios the test set has no yielding agents; sorted order makes runs reproducible |
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

From my run on 2026-10-22: macOS 26.5.1 (arm64), Python 3.12.15, a fresh clone installed with `pip install -r requirements.txt` (67 s), and the default download from `scripts/fetch_data.sh`. Python 3.13.16 gave the same numbers. Progress bars and library warnings are left out. This is the output of this version; it may differ from the output printed in the book, and that's okay.

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

`python multimodal_feature_pipeline.py --max-scenarios 10` (the first edition's default, 33 s). Only 4 agents yield, and none of them are in the test scenarios, so the scores are meaningless; the script says so:

```
Processed 10 scenarios, 10 successful.
Created fused features DataFrame with 192 agent entries.
Uploaded 192 agent features to feature store at outputs/feature_store/agent_features

Training yield prediction model...
Using 191 agents from 10 scenarios for yield prediction model.
Yielding agents: 4 (2.1%)
Adding 768 text embedding dimensions to model features.
Train: 141 agents from 7 scenarios. Test: 50 agents from 3 scenarios.
WARNING: the test scenarios contain only one class, so the scores below don't measure anything. Process more scenarios (the default, --max-scenarios 0, uses all of them).
```

`python multimodal_feature_pipeline.py` (default: all 776 matched scenarios, 263 s):

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
├── scripts/load_scenarios.py       # make check-data / make load: your own scenarios
├── scripts/show_agent.py           # make show: one agent's features and Q&As
├── experiments/                    # make try-leakage, try-split, try-text-dims
├── tests/                          # synthetic Waymo-format data, smoke tests, test_own_data.py
├── docs/img/first-edition/         # charts from the first-edition run
├── outputs/                        # written by the scripts (git-ignored)
├── pyproject.toml, uv.lock, requirements.txt
└── Makefile                        # make run, make smoke, make data and the targets above
```

## License

MIT. See [LICENSE](LICENSE). The Waymo data is not covered by this license; see [Data and licenses](#data-and-licenses).

## Waymo Dataset Usage

This playbook was made using the Waymo Open Dataset, provided by Waymo LLC under the Waymo Dataset License Agreement for Non-Commercial Use, available at waymo.com/open/terms, and your access and use of such work are governed by the terms and conditions therein.
