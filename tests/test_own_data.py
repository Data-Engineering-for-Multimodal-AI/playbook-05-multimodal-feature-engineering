"""Checks for the "Use your own data" loader (scripts/load_scenarios.py), on synthetic
data in the Waymo formats built in tmp_path. No Waymo account, no model download."""

import json
import os
import subprocess
import sys

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "scripts"))
sys.path.insert(0, os.path.join(ROOT, "tests"))
from load_scenarios import scan  # noqa: E402
from make_fixture import write_fixture  # noqa: E402

RD = os.path.join("reasoning", "validation_interactive")
MD = os.path.join("motion", "tf_example", "validation_interactive")
SHARD = "validation_interactive_tfexample.tfrecord-00000-of-00150"


@pytest.fixture(scope="module")
def fixture_dir(tmp_path_factory):
    return str(write_fixture(str(tmp_path_factory.mktemp("good")), scenarios=4))


@pytest.fixture
def folder(fixture_dir, tmp_path):
    import shutil
    dest = tmp_path / "data"
    shutil.copytree(fixture_dir, dest)
    return dest


def reasoning_files(folder):
    return sorted((folder / RD).iterdir())


def edit_json(path, change):
    data = json.loads(path.read_text())
    change(data)
    path.write_text(json.dumps(data))


def test_good_folder(folder):
    summary, problems, notes = scan(str(folder))
    assert problems == [], problems
    assert summary == {"motion": 4, "reasoning": 2, "matched": 2}
    assert notes == []


def test_missing_folders(tmp_path):
    _, problems, _ = scan(str(tmp_path))
    assert len(problems) == 2 and all("folder missing" in p for p in problems)


@pytest.mark.parametrize("change,expected", [
    (lambda d: d.update(ego=999), "is not the self-driving car"),
    (lambda d: d.pop("cur_time"), "missing cur_time"),
    (lambda d: d.update(ego="100"), "ego should be a number"),
    (lambda d: d.update(sur_a=d["sur_a"][:-1]), "sur_q questions but"),
    (lambda d: d.update(env_q=[], env_a=[], ego_q=[], ego_a=[], sur_q=[], sur_a=[]), "nothing to embed"),
    (lambda d: d.update(sid="other"), "file name says scenario"),
])
def test_bad_reasoning_file(folder, change, expected):
    edit_json(reasoning_files(folder)[0], change)
    _, problems, _ = scan(str(folder))
    assert len(problems) == 1 and expected in problems[0], problems


@pytest.mark.parametrize("content,expected", [
    (b"", "empty file"),
    (b"version https://git-lfs.github.com/spec/v1\n", "Git LFS pointer"),
    (b'{"sid": "x",', "not valid JSON"),
    ("{\"sid\": \"café\"}".encode("latin-1"), "not UTF-8"),
])
def test_unreadable_reasoning_file(folder, content, expected):
    reasoning_files(folder)[0].write_bytes(content)
    _, problems, _ = scan(str(folder))
    assert any(expected in p for p in problems), problems


def test_badly_named_reasoning_file(folder):
    (folder / RD / "notes.json").write_text("{}")
    _, problems, _ = scan(str(folder))
    assert len(problems) == 1 and "name must look like scid_" in problems[0]


def test_reasoning_without_motion_is_a_note(folder):
    path = reasoning_files(folder)[0]
    edit_json(path, lambda d: d.update(sid="nomotion"))
    path.rename(folder / RD / "scid_nomotion__aid_100__atype_1.json")
    _, problems, notes = scan(str(folder))
    assert any("1 reasoning file(s) have no motion record" in n and "scid_nomotion__" in n for n in notes)
    assert any("only 1 scenario(s)" in p for p in problems)


def test_truncated_and_pointer_tfrecords(folder):
    shard = (folder / MD / SHARD).read_bytes()
    (folder / MD / SHARD.replace("00000-of", "00001-of")).write_bytes(shard[:3000])
    (folder / MD / SHARD.replace("00000-of", "00002-of")).write_bytes(b"version https://git-lfs.github.com/spec/v1\n")
    (folder / MD / "readme.txt").write_text("x")
    _, problems, _ = scan(str(folder))
    assert any("DataLossError" in p for p in problems), problems
    assert any("Git LFS pointer" in p for p in problems), problems
    assert any("readme.txt: not a .tfrecord file" in p for p in problems), problems


def test_wrong_future_length(folder, tmp_path):
    import tensorflow as tf
    path = folder / MD / SHARD
    records = []
    for raw in tf.data.TFRecordDataset(str(path)):
        ex = tf.train.Example()
        ex.ParseFromString(raw.numpy())
        records.append(ex)
    speeds = list(records[0].features.feature["state/future/speed"].float_list.value)
    del records[0].features.feature["state/future/speed"].float_list.value[:]
    records[0].features.feature["state/future/speed"].float_list.value.extend(speeds[:-8])
    with tf.io.TFRecordWriter(str(path)) as w:
        for ex in records:
            w.write(ex.SerializeToString())
    _, problems, _ = scan(str(folder))
    assert len(problems) == 1 and "state/future/speed has" in problems[0], problems


def test_check_only_writes_nothing(folder):
    stale = os.path.join(ROOT, "outputs", "validation_analysis", "scenario_id_comparison.json")
    before = os.path.getmtime(stale) if os.path.exists(stale) else None
    result = subprocess.run([sys.executable, os.path.join(ROOT, "scripts", "load_scenarios.py"),
                             "--dir", str(folder), "--check-only"], capture_output=True, text=True)
    assert result.returncode == 0, result.stdout + result.stderr
    assert "All files passed the checks." in result.stdout
    assert (os.path.getmtime(stale) if os.path.exists(stale) else None) == before
