"""
Write a small synthetic dataset in the same formats as the Waymo data.

The real data can't be shared (see data/README.md), so the smoke tests run on this.
It has the WOMD tf.Example layout (agent-major arrays: 10 past, 1 current and
80 future steps per agent) and WOMD-Reasoning JSON files with the same fields.
None of it is real driving data.

Usage: python tests/make_fixture.py OUT_DIR [--scenarios N] [--seed S]
Then run the scripts with WAYMO_DATA_DIR=OUT_DIR.
"""

import argparse
import json
import os

import numpy as np
import tensorflow as tf

NUM_AGENTS = 8  # real records pad to 128 agents; fewer keeps the fixture small
PAST, FUTURE = 10, 80


def _floats(values):
    return tf.train.Feature(float_list=tf.train.FloatList(value=np.asarray(values, dtype=np.float32).ravel()))


def _ints(values):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=np.asarray(values, dtype=np.int64).ravel()))


def _bytes(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value.encode()]))


def make_scenario(rng, scenario_id, ego_id):
    """One scenario. Agent 0 is the self-driving car; some agents slow to a stop.

    Returns the tf.Example, the reasoning JSON and, per agent, whether it brakes.
    """
    ids = np.arange(ego_id, ego_id + NUM_AGENTS)
    types = rng.choice([1, 1, 1, 2, 3], size=NUM_AGENTS)  # vehicle, pedestrian, cyclist
    valid_now = np.ones(NUM_AGENTS, dtype=int)
    valid_now[-1] = 0  # one padded agent, like the real data

    x0 = rng.uniform(-40, 40, NUM_AGENTS)
    y0 = rng.uniform(-40, 40, NUM_AGENTS)
    heading = rng.uniform(-np.pi, np.pi, NUM_AGENTS)
    speed0 = rng.uniform(0, 15, NUM_AGENTS)

    # Future speed profiles, shape [agents, 80]: about half the moving agents brake to a stop.
    brakes = (rng.random(NUM_AGENTS) < 0.5) & (speed0 > 3)
    t = np.linspace(0, 1, FUTURE)
    future_speed = np.where(brakes[:, None], speed0[:, None] * (1 - t), speed0[:, None] * np.ones(FUTURE))
    future_valid = np.ones((NUM_AGENTS, FUTURE), dtype=int)
    future_valid[-1] = 0

    dt = 0.1
    vx, vy = np.cos(heading), np.sin(heading)
    future_x = x0[:, None] + np.cumsum(future_speed * dt, axis=1) * vx[:, None]
    future_y = y0[:, None] + np.cumsum(future_speed * dt, axis=1) * vy[:, None]
    past_t = np.arange(-PAST, 0) * dt
    past_x = x0[:, None] + speed0[:, None] * past_t * vx[:, None]
    past_y = y0[:, None] + speed0[:, None] * past_t * vy[:, None]

    feature = {
        "scenario/id": _bytes(scenario_id),
        "state/id": _floats(ids),
        "state/type": _floats(types),
        "state/is_sdc": _ints([1] + [0] * (NUM_AGENTS - 1)),
        "state/current/x": _floats(x0),
        "state/current/y": _floats(y0),
        "state/current/speed": _floats(speed0),
        "state/current/velocity_x": _floats(speed0 * vx),
        "state/current/velocity_y": _floats(speed0 * vy),
        "state/current/bbox_yaw": _floats(heading),
        "state/current/valid": _ints(valid_now),
        "state/past/x": _floats(past_x),
        "state/past/y": _floats(past_y),
        "state/past/speed": _floats(np.repeat(speed0[:, None], PAST, axis=1)),
        "state/past/velocity_x": _floats(np.repeat((speed0 * vx)[:, None], PAST, axis=1)),
        "state/past/velocity_y": _floats(np.repeat((speed0 * vy)[:, None], PAST, axis=1)),
        "state/past/valid": _ints(np.ones((NUM_AGENTS, PAST))),
        "state/future/x": _floats(future_x),
        "state/future/y": _floats(future_y),
        "state/future/speed": _floats(future_speed),
        "state/future/velocity_x": _floats(future_speed * vx[:, None]),
        "state/future/velocity_y": _floats(future_speed * vy[:, None]),
        "state/future/valid": _ints(future_valid),
    }
    example = tf.train.Example(features=tf.train.Features(feature=feature))

    others = [int(i) for i in ids[1:-1]]
    reasoning = {
        "sid": scenario_id,
        "ego": int(ego_id),
        "cur_time": 1.0,
        "future_time": 3.0,
        "rel_id": others,
        "rel_qa_id": list(range(len(others))),
        "env_q": ["What type of intersection is in the scene?", "Is there a traffic light?"],
        "env_a": [f"The intersection is a {rng.choice([3, 4])} way intersection.",
                  f"{rng.choice(['Yes', 'No'])}, there is {rng.choice(['a', 'no'])} traffic light."],
        "ego_q": ["What is the ego agent's current speed?", "What is the ego agent's current motion status?"],
        "ego_a": [f"The ego agent's current speed is {speed0[0]:.0f} m/s.",
                  f"The ego agent is {rng.choice(['accelerating', 'decelerating', 'moving at a constant speed'])}."],
        "sur_q": [f"What is the speed of surrounding agent #{k}?" for k in range(len(others))],
        "sur_a": [f"Surrounding agent #{k} is moving at {speed0[k + 1]:.0f} m/s." for k in range(len(others))],
        "int_q": [f"What interaction will occur between the ego agent and surrounding agent #{k}?"
                  for k in range(len(others))],
        "int_a": [f"Surrounding agent #{k} will {'yield to' if brakes[k + 1] else 'not interact with'} the ego agent."
                  for k in range(len(others))],
    }
    return example, reasoning, brakes


def write_fixture(out_dir, scenarios=12, seed=0):
    rng = np.random.default_rng(seed)
    for split, shards in (("validation_interactive", 150), ("training", 1000)):
        motion_dir = os.path.join(out_dir, "motion", "tf_example", split)
        reasoning_dir = os.path.join(out_dir, "reasoning", split)
        os.makedirs(motion_dir, exist_ok=True)
        os.makedirs(reasoning_dir, exist_ok=True)
        path = os.path.join(motion_dir, f"{split}_tfexample.tfrecord-00000-of-{shards:05d}")
        with tf.io.TFRecordWriter(path) as writer:
            for n in range(scenarios):
                scenario_id = f"{split[:3]}{n:013x}"
                ego_id = 100 + 10 * n
                example, reasoning, _ = make_scenario(rng, scenario_id, ego_id)
                writer.write(example.SerializeToString())
                # Two scenarios have motion data but no reasoning file, as in the real data
                if n < scenarios - 2:
                    name = f"scid_{scenario_id}__aid_{ego_id}__atype_1.json"
                    with open(os.path.join(reasoning_dir, name), "w") as f:
                        json.dump(reasoning, f)
    return out_dir


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    parser.add_argument("out_dir")
    parser.add_argument("--scenarios", type=int, default=12)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    print(write_fixture(args.out_dir, args.scenarios, args.seed))
