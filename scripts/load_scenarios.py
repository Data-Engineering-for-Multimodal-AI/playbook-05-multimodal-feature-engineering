"""Check a folder of driving scenarios and run the feature pipeline on it.

    make check-data DIR=my_scenarios    # check every file, write nothing
    make load DIR=my_scenarios          # check it, then run multimodal_feature_pipeline.py on it

The folder uses the Waymo layout (data/README.md), so a subset of the Waymo data,
another split, or your own scenes converted to the same formats all work:

    my_scenarios/
      motion/tf_example/validation_interactive/   *.tfrecord* files of WOMD tf.Example records
      reasoning/validation_interactive/           scid_<scenario>__aid_<agent>__atype_<type>.json

Each tf.Example needs scenario/id, state/id, state/type, state/is_sdc (one self-driving
car), state/current/{x,y,speed,bbox_yaw,valid} with one value per agent, and
state/future/{speed,valid} with 80 values per agent (agent-major). Each JSON file
needs sid, ego, cur_time, future_time and the env/ego/sur/int question and
answer lists, with as many answers as questions.

The checks cover what makes the pipeline crash, drop a scenario without saying
why, or mislabel agents: unreadable or truncated TFRecords, Git LFS pointers,
missing or wrongly sized arrays, JSON that doesn't parse or isn't UTF-8,
missing fields, file names that don't match the scenario inside, reasoning
files with no motion record, and an ego that isn't the record's self-driving
car. If any check fails, nothing is written.

`make load` deletes outputs/validation_analysis/scenario_id_comparison.json (the
pipeline reuses that list of scenario IDs from the last run, whatever data it
came from) and outputs/feature_store/ (the pipeline adds to it without removing
the last run's entries), then runs `python multimodal_feature_pipeline.py` with
WAYMO_DATA_DIR set to your folder. Its outputs go to outputs/, as with the
Waymo data.
"""

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
from collections import defaultdict

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MOTION_SUB = os.path.join("motion", "tf_example", "validation_interactive")
REASONING_SUB = os.path.join("reasoning", "validation_interactive")
FUTURE_STEPS = 80
PER_AGENT = ["state/id", "state/type", "state/is_sdc", "state/current/x", "state/current/y",
             "state/current/speed", "state/current/bbox_yaw", "state/current/valid"]
FUTURE = ["state/future/speed", "state/future/valid"]
JSON_FIELDS = {"sid": str, "ego": int, "cur_time": (int, float), "future_time": (int, float)}
QA_GROUPS = ["env", "ego", "sur", "int"]
REASONING_NAME = re.compile(r"scid_(.+?)__aid_(\d+)__atype_(\d+)\.json")


def lfs_or_empty(path):
    with open(path, "rb") as fh:
        head = fh.read(100)
    if not head:
        return "empty file"
    if head.startswith(b"version https://git-lfs"):
        return "Git LFS pointer, not the file"
    return None


def values(feature):
    for kind in ("int64_list", "float_list", "bytes_list"):
        v = getattr(feature, kind).value
        if len(v):
            return list(v)
    return []


def check_record(example):
    """Return (scenario id or None, self-driving car id or None, [problems]) for one tf.Example."""
    f = example.features.feature
    if "scenario/id" not in f or not f["scenario/id"].bytes_list.value:
        return None, None, ["record without scenario/id"]
    sid = f["scenario/id"].bytes_list.value[0].decode("utf-8", "replace")
    missing = [k for k in PER_AGENT + FUTURE if k not in f]
    if missing:
        return sid, None, [f"scenario {sid}: missing {', '.join(missing)}"]
    n = len(values(f["state/id"]))
    problems = []
    for k in PER_AGENT:
        if len(values(f[k])) != n:
            problems.append(f"scenario {sid}: {k} has {len(values(f[k]))} values for {n} agents")
    for k in FUTURE:
        if len(values(f[k])) != n * FUTURE_STEPS:
            problems.append(f"scenario {sid}: {k} has {len(values(f[k]))} values, needs "
                            f"{n} agents x {FUTURE_STEPS} steps = {n * FUTURE_STEPS}")
    sdc = [i for i, v in enumerate(values(f["state/is_sdc"])) if v == 1]
    if len(sdc) != 1:
        problems.append(f"scenario {sid}: {len(sdc)} agents marked state/is_sdc, needs exactly 1")
        return sid, None, problems
    ids = values(f["state/id"])
    return sid, int(ids[sdc[0]]) if sdc[0] < len(ids) else None, problems


def check_tfrecord(path):
    """Return ({scenario id: sdc id}, [problems]) for one TFRecord file."""
    import tensorflow as tf
    problem = lfs_or_empty(path)
    if problem:
        return {}, [problem]
    scenarios, problems = {}, []
    count = 0
    try:
        for raw in tf.data.TFRecordDataset(path):
            count += 1
            example = tf.train.Example()
            try:
                example.ParseFromString(raw.numpy())
            except Exception:
                problems.append(f"record {count} is not a tf.Example")
                continue
            sid, sdc, found = check_record(example)
            problems += found
            if sid is not None:
                if sid in scenarios:
                    problems.append(f"scenario {sid} appears twice")
                scenarios[sid] = sdc
            if len(problems) >= 5:
                problems.append("(stopped after 5 problems in this file)")
                break
    except tf.errors.OpError as e:
        problems.append(f"can't read past record {count}: {e.__class__.__name__} (truncated or not a TFRecord)")
    if count == 0 and not problems:
        problems.append("no records")
    return scenarios, problems


def check_reasoning(path):
    """Return (data or None, [problems]) for one reasoning JSON file."""
    problem = lfs_or_empty(path)
    if problem:
        return None, [problem]
    try:
        with open(path, encoding="utf-8") as fh:
            data = json.load(fh)
    except UnicodeDecodeError:
        return None, ["not UTF-8"]
    except json.JSONDecodeError as e:
        return None, [f"not valid JSON (line {e.lineno})"]
    if not isinstance(data, dict):
        return None, ["not a JSON object"]
    problems = []
    for key, kind in JSON_FIELDS.items():
        if key not in data:
            problems.append(f"missing {key}")
        elif not isinstance(data[key], kind) or isinstance(data[key], bool):
            problems.append(f"{key} should be {'a number' if key != 'sid' else 'a string'}, "
                            f"got {type(data[key]).__name__}")
    for group in QA_GROUPS:
        q, a = data.get(f"{group}_q", []), data.get(f"{group}_a", [])
        if not isinstance(q, list) or not isinstance(a, list):
            problems.append(f"{group}_q and {group}_a must be lists")
        elif len(q) != len(a):
            problems.append(f"{len(q)} {group}_q questions but {len(a)} {group}_a answers")
    if not any(data.get(f"{g}_q") for g in ("env", "ego", "sur")):
        problems.append("no env, ego or sur questions: nothing to embed")
    m = REASONING_NAME.fullmatch(os.path.basename(path))
    if m and "sid" in data and m.group(1) != data["sid"]:
        problems.append(f"file name says scenario {m.group(1)}, the file says {data['sid']}")
    return (data if not problems else None), problems


def scan(folder):
    """Check every file. Returns (summary dict, [problems], [notes])."""
    problems, notes = [], []
    motion_dir = os.path.join(folder, MOTION_SUB)
    reasoning_dir = os.path.join(folder, REASONING_SUB)
    for d in (motion_dir, reasoning_dir):
        if not os.path.isdir(d):
            problems.append(f"{os.path.relpath(d, folder)}/: folder missing (see the layout in "
                            f"scripts/load_scenarios.py)")
    if problems:
        return {}, problems, notes

    motion = {}
    where = {}
    for name in sorted(os.listdir(motion_dir)):
        if name.startswith("."):
            continue
        rel = os.path.join(MOTION_SUB, name)
        if ".tfrecord" not in name:
            problems.append(f"{rel}: not a .tfrecord file; the pipeline would skip it")
            continue
        found, file_problems = check_tfrecord(os.path.join(motion_dir, name))
        problems += [f"{rel}: {p}" for p in file_problems]
        for sid, sdc in found.items():
            if sid in motion:
                problems.append(f"{rel}: scenario {sid} is also in {where[sid]}")
            motion[sid] = sdc
            where[sid] = rel

    reasoning = defaultdict(list)
    no_motion = []
    for name in sorted(os.listdir(reasoning_dir)):
        if name.startswith("."):
            continue
        rel = os.path.join(REASONING_SUB, name)
        if not REASONING_NAME.fullmatch(name):
            problems.append(f"{rel}: name must look like scid_<scenario>__aid_<agent>__atype_<type>.json; "
                            f"the pipeline finds files by that prefix")
            continue
        data, file_problems = check_reasoning(os.path.join(reasoning_dir, name))
        problems += [f"{rel}: {p}" for p in file_problems]
        if data is None:
            continue
        sid = data["sid"]
        reasoning[sid].append(rel)
        if sid not in motion:
            no_motion.append(rel)
        elif motion[sid] is not None and data["ego"] != motion[sid]:
            problems.append(f"{rel}: ego {data['ego']} is not the self-driving car ({motion[sid]}) of "
                            f"scenario {sid}; the pipeline would drop the scenario")
    if no_motion:
        notes.append(f"{len(no_motion)} reasoning file(s) have no motion record, so the pipeline skips them "
                     f"(expected if you have only some motion shards), e.g. {no_motion[0]}")
    for sid, rels in reasoning.items():
        if len(rels) > 1:
            notes.append(f"scenario {sid} has {len(rels)} reasoning files; the pipeline uses the first it lists")

    matched = sorted(set(motion) & set(reasoning))
    if not problems and len(matched) < 2:
        problems.append(f"only {len(matched)} scenario(s) have both motion and reasoning data; "
                        f"the yield model needs at least 2")
    return {"motion": len(motion), "reasoning": len(reasoning), "matched": len(matched)}, problems, notes


def main():
    p = argparse.ArgumentParser(description="Check a folder of scenarios and run the pipeline on it.")
    p.add_argument("--dir", required=True, help="folder in the Waymo layout")
    p.add_argument("--check-only", action="store_true", help="check the files and write nothing")
    args = p.parse_args()
    folder = os.path.abspath(args.dir)
    if not os.path.isdir(folder):
        sys.exit(f"No such folder: {args.dir}")
    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

    summary, problems, notes = scan(folder)
    if summary:
        print(f"{args.dir}: {summary['motion']} motion scenarios, {summary['reasoning']} with reasoning "
              f"files, {summary['matched']} in both")
    for line in notes[:10]:
        print(f"  note: {line}")
    if len(notes) > 10:
        print(f"  note: ... and {len(notes) - 10} more like these")
    if problems:
        print(f"\n{len(problems)} problem(s):")
        for line in problems:
            print(f"  {line}")
        sys.exit("\nFix or remove these files and run again. Nothing was written.")
    print("All files passed the checks.", flush=True)
    if args.check_only:
        return

    stale = os.path.join(ROOT, "outputs", "validation_analysis", "scenario_id_comparison.json")
    if os.path.exists(stale):
        os.remove(stale)
    # The pipeline adds files to the feature store without removing old ones
    shutil.rmtree(os.path.join(ROOT, "outputs", "feature_store"), ignore_errors=True)
    env = dict(os.environ, WAYMO_DATA_DIR=folder, MPLBACKEND="Agg")
    print(f"\n$ WAYMO_DATA_DIR={args.dir} python multimodal_feature_pipeline.py", flush=True)
    sys.exit(subprocess.run([sys.executable, "multimodal_feature_pipeline.py"], cwd=ROOT, env=env).returncode)


if __name__ == "__main__":
    main()
