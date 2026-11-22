"""Try this 2: split by scenario, or by agent? (make try-split)

Every agent in a scenario shares the scenario's text embedding and question
counts. The pipeline splits train and test by scenario, so no scenario is on
both sides; the first edition split agent rows at random. This trains the same
model (motion + text, as the pipeline) both ways, over five random seeds, and
reports the test scores and how many test agents have a scenario that is also
in training.

Change SEEDS or the feature set and run it again.
"""

import numpy as np
from sklearn.model_selection import train_test_split

import common

SEEDS = [0, 1, 2, 3, 4]


def main():
    df, text, _, is_real, folder = common.fused_table()
    common.banner(is_real, folder, df)
    cols = common.MOTION_COLS + common.COUNT_COLS + text
    rows = {"by scenario (as the pipeline)": [], "random agent rows (first edition)": []}
    for seed in SEEDS:
        tr, te = common.scenario_split(df, seed)
        rows["by scenario (as the pipeline)"].append((*common.evaluate(df, cols, tr, te), tr, te))
        tr, te = train_test_split(np.arange(len(df)), test_size=0.3, random_state=seed,
                                  stratify=df["is_yielding"])
        rows["random agent rows (first edition)"].append((*common.evaluate(df, cols, tr, te), tr, te))
    print(f"{'Split':34} {'balanced acc':>12} {'F1 (yielding)':>13} {'test agents whose scenario is in training':>42}")
    for label, runs in rows.items():
        bal = np.mean([r[0] for r in runs])
        f1 = np.mean([r[1] for r in runs])
        shared = np.mean([df["scenario_id"].iloc[te].isin(set(df["scenario_id"].iloc[tr])).mean()
                          for _, _, tr, te in runs])
        print(f"{label:34} {bal:12.3f} {f1:13.3f} {shared:41.0%}")
    print(f"\nMeans over {len(SEEDS)} seeds.")


if __name__ == "__main__":
    main()
