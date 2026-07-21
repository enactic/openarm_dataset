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

DATASET_DIR = Path(__file__).parent / "fixture" / "dataset_0.4.0_pose"

POSE_COLUMNS = ["x", "y", "z", "qw", "qx", "qy", "qz", "gripper"]


@pytest.fixture
def dataset():
    return Dataset(DATASET_DIR)


def test_load_action_pose(dataset):
    action = dataset.load_action(dataset.meta.episodes[0])
    assert set(action) == {"arms/left/pose", "arms/right/pose", "lifter/elevation"}
    assert action["arms/left/pose"].shape == (90, 8)
    assert list(action["arms/left/pose"].columns) == POSE_COLUMNS


def test_obs_still_joint_space(dataset):
    obs = dataset.load_obs(dataset.meta.episodes[0])
    assert "arms/left/qpos" in obs
    assert "arms/left/pose" not in obs


def test_sample_pose(dataset):
    samples = dataset.sample(hz=30, episode=dataset.meta.episodes[0])
    assert samples[0].action["arms/left/pose"].shape == (8,)


def test_get_embodiment_attributes(dataset):
    episode = dataset.meta.episodes[0]
    action_attributes = {
        (a["embodiment"].name, a["component"], a["name"])
        for a in dataset.get_embodiment_attributes("action", episode)
    }
    assert action_attributes == {
        ("arms", "left", "pose"),
        ("arms", "right", "pose"),
        ("lifter", None, "elevation"),
    }
    obs_attributes = {
        (a["component"], a["name"])
        for a in dataset.get_embodiment_attributes("obs", episode)
        if a["embodiment"].name == "arms"
    }
    assert obs_attributes == {
        (side, name)
        for side in ("left", "right")
        for name in ("qpos", "qvel", "qtorque")
    }
