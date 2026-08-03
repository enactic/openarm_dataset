# Copyright 2026 Enactic, Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import shutil
from pathlib import Path

import pandas as pd
import pytest

from openarm_dataset.dataset import Dataset

DATASET_DIR = Path(__file__).parent / "fixture" / "dataset_0.4.0_qpos"

ARM_JOINT_COLUMNS = [
    "joint1",
    "joint2",
    "joint3",
    "joint4",
    "joint5",
    "joint6",
    "joint7",
    "gripper",
]
ARM_OBS_KEYS = {
    "arms/left/qpos",
    "arms/left/qvel",
    "arms/left/qtorque",
    "arms/right/qpos",
    "arms/right/qvel",
    "arms/right/qtorque",
}
ARM_ACTION_KEYS = {"arms/left/qpos", "arms/right/qpos"}


@pytest.fixture
def dataset():
    return Dataset(DATASET_DIR)


def test_num_episodes(dataset):
    assert dataset.num_episodes == 2


def test_load_obs(dataset):
    obs = dataset.load_obs(dataset.meta.episodes[0])
    assert set(obs) == ARM_OBS_KEYS | {"lifter/elevation"}
    for key in ARM_OBS_KEYS:
        assert list(obs[key].columns) == ARM_JOINT_COLUMNS
    assert obs["arms/left/qpos"].shape == (745, 8)
    assert obs["arms/right/qpos"].shape == (746, 8)
    assert obs["lifter/elevation"].shape == (745, 1)


def test_load_action(dataset):
    action = dataset.load_action(dataset.meta.episodes[0])
    assert set(action) == ARM_ACTION_KEYS | {"lifter/elevation"}
    assert action["arms/left/qpos"].shape == (90, 8)
    assert list(action["arms/left/qpos"].columns) == ARM_JOINT_COLUMNS
    assert action["lifter/elevation"].shape == (90, 1)


def test_cameras(dataset):
    assert set(dataset.camera_names) == {
        "ceiling",
        "head_left",
        "head_right",
        "wrist_left",
        "wrist_right",
    }
    cameras = dataset.load_cameras(dataset.meta.episodes[0])
    assert cameras["head_left"].num_frames > 0


def test_sample(dataset):
    samples = dataset.sample(hz=30, episode=dataset.meta.episodes[0])
    assert len(samples) > 1
    assert set(samples[0].action) == ARM_ACTION_KEYS | {"lifter/elevation"}
    assert samples[0].action["arms/left/qpos"].shape == (8,)


def test_obs_columns_optional(dataset, tmp_path):
    root = tmp_path / "minimal"
    shutil.copytree(DATASET_DIR, root)
    for episode_id in ("0", "3"):
        for side in ("left", "right"):
            path = (
                root / "episodes" / episode_id / "obs" / "arms" / side / "state.parquet"
            )
            pd.read_parquet(path)[["timestamp", "qpos"]].to_parquet(path)
    minimal = Dataset(root)
    obs = minimal.load_obs(minimal.meta.episodes[0])
    assert set(obs) == {"arms/left/qpos", "arms/right/qpos", "lifter/elevation"}


def test_unknown_state_column_raises(dataset, tmp_path):
    root = tmp_path / "bad"
    shutil.copytree(DATASET_DIR, root)
    path = root / "episodes" / "0" / "obs" / "arms" / "left" / "state.parquet"
    df = pd.read_parquet(path)
    df["tmos"] = df["qpos"]
    df.to_parquet(path)
    bad = Dataset(root)
    with pytest.raises(ValueError, match="tmos"):
        bad.load_obs(bad.meta.episodes[0])


def test_write_round_trip(dataset, tmp_path):
    import yaml

    output = tmp_path / "out"
    dataset.write(output)
    meta = yaml.safe_load((output / "metadata.yaml").read_text())
    assert meta["version"] == "0.4.0"
    assert meta["leader_device_type"] == "OpenArmKER"
    rewritten = Dataset(output)
    action = rewritten.load_action(rewritten.meta.episodes[0])
    assert set(action) == ARM_ACTION_KEYS | {"lifter/elevation"}


def test_missing_embodiment_files_are_skipped(tmp_path):
    # The recorder declares the lifter in metadata.yaml but only writes
    # elevation.parquet when elevation was actually recorded.
    root = tmp_path / "no_lifter"
    shutil.copytree(DATASET_DIR, root)
    for episode_id in ("0", "3"):
        for type_ in ("obs", "action"):
            shutil.rmtree(root / "episodes" / episode_id / type_ / "lifter")
    no_lifter = Dataset(root)
    obs = no_lifter.load_obs(no_lifter.meta.episodes[0])
    assert set(obs) == ARM_OBS_KEYS
    action = no_lifter.load_action(no_lifter.meta.episodes[0])
    assert set(action) == ARM_ACTION_KEYS
    output = tmp_path / "out"
    no_lifter.write(output)
    assert not (output / "episodes" / "0" / "obs" / "lifter").exists()
    assert (
        output / "episodes" / "0" / "action" / "arms" / "left" / "state.parquet"
    ).exists()
