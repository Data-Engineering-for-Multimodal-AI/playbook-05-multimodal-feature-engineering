"""Show one agent's entry in the feature store, next to the Q&As it was built from.

    make show                                  # the first yielding agent
    make show SCENARIO=1008aa4114dbc237 AGENT=2998

Reads the feature store the last run wrote (outputs/feature_store/, or
synthetic/outputs/feature_store/ after a run on synthetic data) and the
reasoning file for the agent's scenario. Changes nothing.
"""

import argparse
import json
import math
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
STORES = [("outputs", os.environ.get("WAYMO_DATA_DIR", os.path.join(ROOT, "data"))),
          (os.path.join("synthetic", "outputs"), os.path.join(ROOT, "synthetic", "data"))]
SHOW_QA = 3


def find_store():
    """The feature store the last run wrote (newest metadata.json), and its data folder."""
    found = []
    for out, data in STORES:
        meta = os.path.join(ROOT, out, "feature_store", "metadata.json")
        if os.path.exists(meta):
            found.append((os.path.getmtime(meta), os.path.join(ROOT, out, "feature_store", "agent_features"), data))
    if not found:
        sys.exit("No feature store found. Run make run first.")
    _, store, data = max(found)
    return store, data


def main():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--scenario")
    p.add_argument("--agent")
    args = p.parse_args()
    store, data_dir = find_store()
    names = sorted(os.listdir(store))
    if args.scenario and args.agent:
        name = f"{args.scenario}__{args.agent}.json"
        if name not in names:
            sys.exit(f"{name} is not in {os.path.relpath(store, ROOT)}.")
    else:
        name = None
        for n in names:
            with open(os.path.join(store, n)) as fh:
                if json.load(fh).get("is_yielding") is True:
                    name = n
                    break
        name = name or names[0]
    with open(os.path.join(store, name)) as fh:
        entry = json.load(fh)

    print(f"{os.path.relpath(os.path.join(store, name), ROOT)}  ({len(names)} entries in the store)\n")
    for key, value in entry.items():
        if key == "text_embedding":
            print(f"  {key:22} [{len(value)} numbers: {value[0]:.4f}, {value[1]:.4f}, ...]")
        elif not (isinstance(value, float) and math.isnan(value)):
            print(f"  {key:22} {value}")

    reasoning = os.path.join(data_dir, "reasoning", "validation_interactive")
    files = sorted(f for f in os.listdir(reasoning) if f.startswith(f"scid_{entry['scenario_id']}__")) \
        if os.path.isdir(reasoning) else []
    if not files:
        print(f"\n(No reasoning file for this scenario in {reasoning}.)")
        return
    with open(os.path.join(reasoning, files[0])) as fh:
        qa = json.load(fh)
    print(f"\nFrom reasoning/validation_interactive/{files[0]}:")
    for group, label in (("env", "map"), ("ego", "ego vehicle"), ("sur", "surrounding agents"),
                         ("int", "interactions (not a model input: they describe the future)")):
        pairs = list(zip(qa.get(f"{group}_q", []), qa.get(f"{group}_a", [])))
        print(f"  {group} ({label}), {len(pairs)} Q&As:")
        for q, a in pairs[:SHOW_QA]:
            print(f"    Q: {q}\n    A: {a}")


if __name__ == "__main__":
    main()
