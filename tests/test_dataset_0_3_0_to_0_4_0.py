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

from pathlib import Path

import pyarrow.parquet as pq
import pytest

from openarm_dataset.dataset import Dataset

DATASET_DIR = Path(__file__).parent / "fixture" / "dataset_0.3.0"

JOINT_COLUMNS = [
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

ARM_ACTION_KEYS = {
    "arms/left/qpos",
    "arms/right/qpos",
}


@pytest.fixture
def dataset(tmp_path):
    old_dataset = Dataset(DATASET_DIR)
    new_dataset_dir = tmp_path / "dataset"
    old_dataset.write(new_dataset_dir)
    return Dataset(new_dataset_dir)


def test_metadata_version(dataset):
    assert dataset.meta.version == "0.4.0"


def test_num_episodes(dataset):
    assert dataset.num_episodes == 2


def test_state_parquet_layout(dataset):
    expected_columns = {
        "obs": ["timestamp", "qpos", "qvel", "qtorque"],
        "action": ["timestamp", "qpos"],
    }
    for episode_id in ("0", "3"):
        for type_, columns in expected_columns.items():
            for side in ("left", "right"):
                arm_dir = (
                    dataset.root_path / "episodes" / episode_id / type_ / "arms" / side
                )
                state_path = arm_dir / "state.parquet"
                assert state_path.exists()
                assert not (arm_dir / "qpos.parquet").exists()
                assert pq.read_schema(state_path).names == columns


def test_load_obs(dataset):
    obs = dataset.load_obs(dataset.meta.episodes[0])
    assert set(obs) == ARM_OBS_KEYS | {"lifter/elevation"}
    for key in ARM_OBS_KEYS:
        assert obs[key].index.name == "timestamp"
        assert list(obs[key].columns) == JOINT_COLUMNS
    assert obs["arms/left/qpos"].shape == (745, 8)
    assert obs["arms/left/qvel"].shape == (745, 8)
    assert obs["arms/left/qtorque"].shape == (745, 8)
    assert obs["arms/right/qpos"].shape == (746, 8)
    assert obs["arms/right/qvel"].shape == (746, 8)
    assert obs["arms/right/qtorque"].shape == (746, 8)
    assert obs["lifter/elevation"].index.name == "timestamp"
    assert list(obs["lifter/elevation"].columns) == ["elevation"]
    assert obs["lifter/elevation"].shape == (745, 1)


def test_obs_columns_are_independent(dataset):
    obs = dataset.load_obs(dataset.meta.episodes[0])
    qpos = obs["arms/right/qpos"].iloc[0].to_numpy()
    qvel = obs["arms/right/qvel"].iloc[0].to_numpy()
    qtorque = obs["arms/right/qtorque"].iloc[0].to_numpy()
    assert qvel == pytest.approx(qpos * 0.1, rel=1e-5)
    assert qtorque == pytest.approx(qpos * 0.01, rel=1e-5)


def test_load_all_obs(dataset):
    obs_list = [dataset.load_obs(episode) for episode in dataset.meta.episodes]
    assert len(obs_list) == dataset.num_episodes
    for obs in obs_list:
        for key in ARM_OBS_KEYS | {"lifter/elevation"}:
            assert not obs[key].empty


def test_load_action(dataset):
    action = dataset.load_action(dataset.meta.episodes[0])
    assert set(action) == ARM_ACTION_KEYS | {"lifter/elevation"}
    assert action["arms/left/qpos"].shape == (90, 8)
    assert action["arms/right/qpos"].shape == (90, 8)
    assert list(action["arms/left/qpos"].columns) == JOINT_COLUMNS
    assert list(action["arms/right/qpos"].columns) == JOINT_COLUMNS
    assert action["lifter/elevation"].shape == (90, 1)
    assert list(action["lifter/elevation"].columns) == ["elevation"]


def test_load_all_action(dataset):
    action_list = [dataset.load_action(episode) for episode in dataset.meta.episodes]
    assert len(action_list) == dataset.num_episodes
    for action in action_list:
        for key in ARM_ACTION_KEYS | {"lifter/elevation"}:
            assert not action[key].empty


def test_camera_names(dataset):
    assert set(dataset.camera_names) == {
        "ceiling",
        "head",
        "wrist_left",
        "wrist_right",
    }


def test_load_cameras(dataset):
    cameras = dataset.load_cameras(dataset.meta.episodes[0])
    assert set(cameras) == {
        "ceiling",
        "head",
        "wrist_left",
        "wrist_right",
    }
    assert cameras["ceiling"].num_frames == 3


def test_load_camera(dataset):
    camera_data = dataset.load_camera("ceiling", dataset.meta.episodes[0])
    assert camera_data.num_frames == 3


def test_camera_filter(dataset):
    dataset = Dataset(
        dataset.root_path,
        camera_names=[
            "head",
            "wrist_left",
            "wrist_right",
        ],
    )
    assert set(dataset.camera_names) == {
        "head",
        "wrist_left",
        "wrist_right",
    }
    assert set(dataset.load_cameras(dataset.meta.episodes[0])) == {
        "head",
        "wrist_left",
        "wrist_right",
    }


def test_sample(dataset):
    samples = dataset.sample(hz=30, episode=dataset.meta.episodes[0])
    assert len(samples) > 1
    interval = samples[1].timestamp - samples[0].timestamp
    assert interval == pytest.approx(1 / 30, rel=0.1)
    assert set(samples[0].obs) == ARM_OBS_KEYS | {"lifter/elevation"}
    assert samples[0].obs["arms/left/qpos"].shape == (8,)
    assert set(samples[0].action) == ARM_ACTION_KEYS | {"lifter/elevation"}
    assert samples[0].action["arms/left/qpos"].shape == (8,)
    assert set(samples[0].cameras) == {
        "ceiling",
        "head",
        "wrist_left",
        "wrist_right",
    }
