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


@pytest.fixture
def dataset():
    return Dataset(DATASET_DIR)


def test_num_episodes(dataset):
    assert dataset.num_episodes == 2


def test_load_obs(dataset):
    obs = dataset.load_obs(0)
    assert set(obs) == {
        "arms/left/qpos",
        "arms/left/qvel",
        "arms/left/qtorque",
        "arms/right/qpos",
        "arms/right/qvel",
        "arms/right/qtorque",
    }
    for key in obs:
        assert obs[key].index.name == "timestamp"
        assert list(obs[key].columns) == JOINT_COLUMNS
    assert obs["arms/left/qpos"].shape == (745, 8)
    assert obs["arms/left/qvel"].shape == (745, 8)
    assert obs["arms/left/qtorque"].shape == (745, 8)
    assert obs["arms/right/qpos"].shape == (746, 8)
    assert obs["arms/right/qvel"].shape == (746, 8)
    assert obs["arms/right/qtorque"].shape == (746, 8)


def test_obs_columns_are_independent(dataset):
    obs = dataset.load_obs(0)
    qpos = obs["arms/right/qpos"].iloc[0].to_numpy()
    qvel = obs["arms/right/qvel"].iloc[0].to_numpy()
    qtorque = obs["arms/right/qtorque"].iloc[0].to_numpy()
    # Fixture writes qvel = qpos * 0.1 and qtorque = qpos * 0.01.
    assert qvel == pytest.approx(qpos * 0.1, rel=1e-5)
    assert qtorque == pytest.approx(qpos * 0.01, rel=1e-5)


def test_load_all_obs(dataset):
    obs_list = [dataset.load_obs(i) for i in range(dataset.num_episodes)]
    assert len(obs_list) == dataset.num_episodes
    for obs in obs_list:
        for key in (
            "arms/left/qpos",
            "arms/left/qvel",
            "arms/left/qtorque",
            "arms/right/qpos",
            "arms/right/qvel",
            "arms/right/qtorque",
        ):
            assert not obs[key].empty


def test_load_action(dataset):
    action = dataset.load_action(0)
    assert set(action) == {
        "arms/left/qpos",
        "arms/right/qpos",
    }
    assert action["arms/left/qpos"].shape == (90, 8)
    assert action["arms/right/qpos"].shape == (90, 8)
    assert list(action["arms/left/qpos"].columns) == JOINT_COLUMNS
    assert list(action["arms/right/qpos"].columns) == JOINT_COLUMNS


def test_sample(dataset):
    samples = dataset.sample(hz=30, episode_index=0)
    assert len(samples) > 1
    assert set(samples[0].obs) == {
        "arms/left/qpos",
        "arms/left/qvel",
        "arms/left/qtorque",
        "arms/right/qpos",
        "arms/right/qvel",
        "arms/right/qtorque",
    }
    assert samples[0].obs["arms/left/qpos"].shape == (8,)
    assert samples[0].obs["arms/left/qvel"].shape == (8,)
    assert samples[0].obs["arms/left/qtorque"].shape == (8,)
    assert set(samples[0].action) == {
        "arms/left/qpos",
        "arms/right/qpos",
    }


def test_write_preserves_state_parquet(dataset, tmp_path):
    output = tmp_path / "out"
    dataset.write(output)
    for episode_id in ("0", "3"):
        for side in ("left", "right"):
            assert (
                output
                / "episodes"
                / episode_id
                / "obs"
                / "arms"
                / side
                / "state.parquet"
            ).exists()
            assert not (
                output
                / "episodes"
                / episode_id
                / "obs"
                / "arms"
                / side
                / "qpos.parquet"
            ).exists()
            assert (
                output
                / "episodes"
                / episode_id
                / "action"
                / "arms"
                / side
                / "qpos.parquet"
            ).exists()
    rewritten = Dataset(output)
    obs = rewritten.load_obs(0)
    assert set(obs) == {
        "arms/left/qpos",
        "arms/left/qvel",
        "arms/left/qtorque",
        "arms/right/qpos",
        "arms/right/qvel",
        "arms/right/qtorque",
    }
