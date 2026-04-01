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

DATASET_DIR = Path(__file__).parent / "fixture" / "dataset_0.2.0"


def test_num_episodes():
    dataset = Dataset(DATASET_DIR)
    assert dataset.num_episodes == 2


def test_load_obs():
    dataset = Dataset(DATASET_DIR)
    obs = dataset.load_obs(0)
    assert set(obs) == {
        "arms/left/qpos",
        "arms/right/qpos",
    }
    assert obs["arms/left/qpos"].shape == (745, 8)
    assert obs["arms/right/qpos"].shape == (746, 8)
    assert obs["arms/left/qpos"].index.name == "timestamp"
    assert obs["arms/right/qpos"].index.name == "timestamp"
    assert list(obs["arms/left/qpos"].columns) == [
        "joint1",
        "joint2",
        "joint3",
        "joint4",
        "joint5",
        "joint6",
        "joint7",
        "gripper",
    ]
    assert list(obs["arms/right/qpos"].columns) == [
        "joint1",
        "joint2",
        "joint3",
        "joint4",
        "joint5",
        "joint6",
        "joint7",
        "gripper",
    ]


def test_load_all_obs():
    dataset = Dataset(DATASET_DIR)
    obs_list = [dataset.load_obs(i) for i in range(dataset.num_episodes)]
    assert len(obs_list) == dataset.num_episodes
    for obs in obs_list:
        assert not obs["arms/left/qpos"].empty
        assert not obs["arms/right/qpos"].empty


def test_load_action():
    dataset = Dataset(DATASET_DIR)
    action = dataset.load_action(0)
    assert set(action) == {
        "arms/left/qpos",
        "arms/right/qpos",
    }
    assert action["arms/left/qpos"].shape == (90, 8)
    assert action["arms/right/qpos"].shape == (90, 8)
    assert action["arms/left/qpos"].index.name == "timestamp"
    assert action["arms/right/qpos"].index.name == "timestamp"
    assert list(action["arms/left/qpos"].columns) == [
        "joint1",
        "joint2",
        "joint3",
        "joint4",
        "joint5",
        "joint6",
        "joint7",
        "gripper",
    ]
    assert list(action["arms/right/qpos"].columns) == [
        "joint1",
        "joint2",
        "joint3",
        "joint4",
        "joint5",
        "joint6",
        "joint7",
        "gripper",
    ]


def test_load_all_action():
    dataset = Dataset(DATASET_DIR)
    action_list = [dataset.load_action(i) for i in range(dataset.num_episodes)]
    assert len(action_list) == dataset.num_episodes
    for action in action_list:
        assert not action["arms/left/qpos"].empty
        assert not action["arms/right/qpos"].empty


def test_camera_names():
    dataset = Dataset(DATASET_DIR)
    assert set(dataset.camera_names) == {
        "ceiling",
        "head",
        "wrist_left",
        "wrist_right",
    }


def test_load_cameras():
    dataset = Dataset(DATASET_DIR)
    cameras = dataset.load_cameras(0)
    assert set(cameras) == {
        "ceiling",
        "head",
        "wrist_left",
        "wrist_right",
    }
    assert cameras["ceiling"].num_frames == 3


def test_load_camera():
    dataset = Dataset(DATASET_DIR)
    camera_data = dataset.load_camera("ceiling", 0)
    assert camera_data.num_frames == 3


def test_camera_filter():
    dataset = Dataset(
        DATASET_DIR,
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
    assert set(dataset.load_cameras(0)) == {
        "head",
        "wrist_left",
        "wrist_right",
    }


def test_sample():
    dataset = Dataset(DATASET_DIR)
    samples = dataset.sample(hz=30, episode_index=0)
    assert len(samples) > 1
    interval = samples[1].timestamp - samples[0].timestamp
    assert interval == pytest.approx(1 / 30, rel=0.1)
    assert set(samples[0].obs) == {
        "arms/left/qpos",
        "arms/right/qpos",
    }
    assert samples[0].obs["arms/left/qpos"].shape == (8,)
    assert set(samples[0].action) == {
        "arms/left/qpos",
        "arms/right/qpos",
    }
    assert samples[0].action["arms/left/qpos"].shape == (8,)
    assert set(samples[0].cameras) == {
        "ceiling",
        "head",
        "wrist_left",
        "wrist_right",
    }
    cameras_dir = DATASET_DIR / "episodes" / "0" / "cameras"
    assert samples[0].cameras["ceiling"].path == (
        cameras_dir / "ceiling" / "1772010251629083055.jpeg"
    )
    assert samples[0].cameras["head"].path == (
        cameras_dir / "head" / "1772010251629774985.jpeg"
    )
    assert samples[0].cameras["wrist_left"].path == (
        cameras_dir / "wrist_left" / "1772010251620214727.jpeg"
    )
    assert samples[0].cameras["wrist_right"].path == (
        cameras_dir / "wrist_right" / "1772010251628789283.jpeg"
    )
