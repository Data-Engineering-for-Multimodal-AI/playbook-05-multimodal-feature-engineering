"""Try this 1: does the text's "yield" mean the label's? (make try-leakage)

The interaction Q&As (int_q, int_a) describe what agents will do over the next
few seconds, which is what the yield label measures. The pipeline leaves them
out of the model's inputs; the first edition put them in. This trains the
pipeline's model (same scenario split) with four feature sets:

  motion only                     speed, distance to the ego vehicle
  + text (as the pipeline)        + question counts + embedding of env/ego/sur Q&As
  + text with interaction Q&As    the same, with the int Q&As in the embedding (first edition)
  + "will yield" in an int answer motion + one flag per agent: an int answer says
                                  this agent will yield (agents are named "surrounding
                                  agent #k"; rel_id maps k to the agent ID)

Change the feature sets in main() and run it again.
"""

import json
import os
import re

import common

NUMBER = re.compile(r"#(\d+)")


def yield_flags(df, folder):
    """1.0 for agents an interaction answer says will yield, from the scenario's reasoning file
    (the same file the pipeline reads: the first one listed for the scenario)."""
    reasoning = os.path.join(folder, "reasoning", "validation_interactive")
    listing = os.listdir(reasoning)
    flagged = set()
    for sid in df["scenario_id"].unique():
        files = [f for f in listing if f.startswith(f"scid_{sid}__")]
        if not files:
            continue
        with open(os.path.join(reasoning, files[0])) as fh:
            data = json.load(fh)
        rel = data.get("rel_id", [])
        for answer in data.get("int_a", []):
            for sentence in answer.split(". "):
                if "will yield" not in sentence:
                    continue
                for k in NUMBER.findall(sentence.split("will yield")[0]):
                    if int(k) < len(rel):
                        flagged.add((sid, int(rel[int(k)])))
    return [1.0 if (s, int(a)) in flagged else 0.0 for s, a in zip(df["scenario_id"], df["agent_id"])]


def main():
    df, text, text_int, is_real, folder = common.fused_table()
    common.banner(is_real, folder, df)
    df["int_says_yields"] = yield_flags(df, folder)
    flagged = df["int_says_yields"] == 1
    yielding = df["is_yielding"]
    print(f"An int answer says {int(flagged.sum())} agents will yield. {yielding[flagged].mean():.0%} of them "
          f"are labeled yielding (all agents: {yielding.mean():.1%}),")
    print(f"and {(df.loc[flagged, 'speed'] < 2).mean():.0%} of them are already below 2 m/s, which the label "
          f"never counts as yielding.")
    print(f"Of the {int(yielding.sum())} agents labeled yielding, {df.loc[yielding, 'int_says_yields'].mean():.0%} "
          f"are named in an int answer.\n")
    cases = [
        ("motion only", common.MOTION_COLS),
        ("+ text (as the pipeline)", common.MOTION_COLS + common.COUNT_COLS + text),
        ("+ text with interaction Q&As", common.MOTION_COLS + common.COUNT_COLS + text_int),
        ('+ "will yield" in an int answer', common.MOTION_COLS + ["int_says_yields"]),
    ]
    train, test = common.scenario_split(df)
    print(f"{'Features':32} {'inputs':>6} {'balanced acc':>12} {'F1 (yielding)':>13}")
    for label, cols in cases:
        bal, f1 = common.evaluate(df, cols, train, test)
        print(f"{label:32} {len(cols):6} {bal:12.3f} {f1:13.3f}")


if __name__ == "__main__":
    main()
