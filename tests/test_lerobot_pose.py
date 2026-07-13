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

import json
import shutil
from pathlib import Path

import pandas as pd
import pytest

from openarm_dataset.dataset import Dataset

FIXTURE = Path(__file__).parent / "fixture"

POSE_DIM_NAMES = ["x", "y", "z", "qw", "qx", "qy", "qz", "gripper"]
ARM_JOINTS = [
    "joint1",
    "joint2",
    "joint3",
    "joint4",
    "joint5",
    "joint6",
    "joint7",
    "gripper",
]


def _names(prefix, dims):
    return [f"{prefix}_{d}.pos" for d in dims]


EXPECTED_ACTION_NAMES = (
    _names("right", POSE_DIM_NAMES) + _names("left", POSE_DIM_NAMES) + ["elevation.pos"]
)
EXPECTED_OBS_NAMES = (
    _names("right", ARM_JOINTS) + _names("left", ARM_JOINTS) + ["elevation.pos"]
)


def test_lerobot_v21_pose(tmp_path):
    dataset = Dataset(FIXTURE / "dataset_0.4.0_pose")
    dataset.write(tmp_path / "out", format="lerobot_v2.1", fps=10)
    info = json.loads((tmp_path / "out" / "meta" / "info.json").read_text())
    assert info["features"]["action"]["names"] == EXPECTED_ACTION_NAMES
    assert info["features"]["action"]["shape"] == [17]
    assert info["features"]["observation.state"]["names"] == EXPECTED_OBS_NAMES
    assert info["features"]["observation.state"]["shape"] == [17]
    episode = pd.read_parquet(
        tmp_path / "out" / "data" / "chunk-000" / "episode_000000.parquet"
    )
    assert len(episode["action"].iloc[0]) == 17
    assert len(episode["observation.state"].iloc[0]) == 17


def test_lerobot_v30_pose(tmp_path):
    dataset = Dataset(FIXTURE / "dataset_0.4.0_pose")
    dataset.write(tmp_path / "out", format="lerobot_v3.0", fps=10)
    info = json.loads((tmp_path / "out" / "meta" / "info.json").read_text())
    assert info["features"]["action"]["names"] == EXPECTED_ACTION_NAMES
    assert info["features"]["action"]["shape"] == [17]
    assert info["features"]["observation.state"]["names"] == EXPECTED_OBS_NAMES


def test_gr00t_pose_modality(tmp_path):
    dataset = Dataset(FIXTURE / "dataset_0.4.0_pose")
    dataset.write(tmp_path / "out", format="gr00t", fps=10)
    modality = json.loads((tmp_path / "out" / "meta" / "modality.json").read_text())
    assert modality["action"]["right_arm"] == {"start": 0, "end": 7}
    assert modality["action"]["right_gripper"] == {"start": 7, "end": 8}
    assert modality["action"]["left_arm"] == {"start": 8, "end": 15}
    assert modality["action"]["left_gripper"] == {"start": 15, "end": 16}
    assert modality["action"]["lifter"] == {"start": 16, "end": 17}
    assert modality["state"]["right_arm"] == {"start": 0, "end": 7}
    assert modality["state"]["right_gripper"] == {"start": 7, "end": 8}


def test_ambiguous_action_attributes_raises(tmp_path):
    root = tmp_path / "ambiguous"
    shutil.copytree(FIXTURE / "dataset_0.4.0_pose", root)
    for episode_id in ("0", "3"):
        for side in ("left", "right"):
            path = (
                root
                / "episodes"
                / episode_id
                / "action"
                / "arms"
                / side
                / "state.parquet"
            )
            df = pd.read_parquet(path)
            df["qpos"] = df["pose"]
            df.to_parquet(path)
    dataset = Dataset(root)
    with pytest.raises(ValueError, match="[Aa]mbiguous"):
        dataset.write(tmp_path / "out", format="lerobot_v2.1", fps=10)
