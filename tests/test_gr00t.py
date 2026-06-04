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
import json

from openarm_dataset import Dataset
from openarm_dataset.lerobot_v21 import (
    _collect_keys_and_joint_names,
    _write_modality_json,
)

FIXTURE_DIR = Path(__file__).parent / "fixture"
DATASET_0_3_0_PATH = FIXTURE_DIR / "dataset_0.3.0"
FPS = 30

EXPECTED_RANGES = {
    "right_arm": {"start": 0, "end": 7},
    "right_gripper": {"start": 7, "end": 8},
    "left_arm": {"start": 8, "end": 15},
    "left_gripper": {"start": 15, "end": 16},
    "lifter": {"start": 16, "end": 17},
}

EXPECTED_MODALITY = {
    "state": EXPECTED_RANGES,
    "action": EXPECTED_RANGES,
    "video": {
        "wrist_left": {"original_key": "observation.images.wrist_left"},
        "wrist_right": {"original_key": "observation.images.wrist_right"},
        "ceiling": {"original_key": "observation.images.ceiling"},
        "head": {"original_key": "observation.images.head"},
    },
    "annotation": {
        "human.task_description": {"original_key": "task_index"},
    },
}


def test_write_modality_json(tmp_path):
    dataset = Dataset(DATASET_0_3_0_PATH)
    _write_modality_json(dataset, tmp_path)
    with open(tmp_path / "meta" / "modality.json") as f:
        modality = json.load(f)
    assert modality == EXPECTED_MODALITY


def test_modality_ranges_are_contiguous_and_cover_all_joints(tmp_path):
    dataset = Dataset(DATASET_0_3_0_PATH)
    _, joint_names = _collect_keys_and_joint_names(dataset)
    _write_modality_json(dataset, tmp_path)
    with open(tmp_path / "meta" / "modality.json") as f:
        modality = json.load(f)
    for block in ("state", "action"):
        assert len(modality[block]) == len(EXPECTED_RANGES)
        ranges = list(modality[block].values())
        assert ranges[0]["start"] == 0
        for prev, cur in zip(ranges, ranges[1:]):
            assert cur["start"] == prev["end"]
        assert ranges[-1]["end"] == len(joint_names)
