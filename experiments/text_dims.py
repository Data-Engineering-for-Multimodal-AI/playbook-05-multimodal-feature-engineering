"""Try this 3: how many text dimensions should the fused vector have? (make try-text-dims)

Early fusion concatenates 2 motion features, 3 question counts and a
768-dimensional text embedding, so text is 99% of the model's inputs. This
shrinks the embedding with PCA (fitted on the training scenarios only) to
DIMS dimensions and trains the pipeline's model on the same scenario split.

Change DIMS and run it again.
"""

import pandas as pd
from sklearn.decomposition import PCA

import common

DIMS = [0, 2, 8, 32, 128, 768]


def main():
    df, text, _, is_real, folder = common.fused_table()
    common.banner(is_real, folder, df)
    train, test = common.scenario_split(df)
    base = common.MOTION_COLS + common.COUNT_COLS
    print(f"{'text dimensions':>15} {'inputs':>6} {'text share':>10} {'balanced acc':>12} {'F1 (yielding)':>13}")
    for k in DIMS:
        work = df[base + ["is_yielding", "scenario_id"]].copy()
        cols = list(base)
        if k:
            k = min(k, len(text), len(train))
            if k < len(text):
                pca = PCA(n_components=k, random_state=0).fit(df.loc[train, text])
                reduced = pca.transform(df[text])
            else:
                reduced = df[text].to_numpy()
            pcs = pd.DataFrame(reduced, columns=[f"pc{i}" for i in range(k)], index=work.index)
            work = pd.concat([work, pcs], axis=1)
            cols += list(pcs.columns)
        bal, f1 = common.evaluate(work, cols, train, test)
        print(f"{k:15} {len(cols):6} {k / len(cols):10.0%} {bal:12.3f} {f1:13.3f}")


if __name__ == "__main__":
    main()
