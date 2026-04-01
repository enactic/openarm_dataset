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

from openarm_dataset.metadata import Metadata

METADATA_PATH = Path(__file__).parent / "fixture" / "dataset_0.1.0" / "metadata.yaml"


def test_version():
    meta = Metadata(METADATA_PATH)
    assert meta.version == "0.1.0"


def test_operator():
    meta = Metadata(METADATA_PATH)
    assert meta.operator == "Tester"


def test_operation_type():
    meta = Metadata(METADATA_PATH)
    assert meta.operation_type == "teleop"


def test_location():
    meta = Metadata(METADATA_PATH)
    assert meta.location == "Test"


def test_tasks():
    meta = Metadata(METADATA_PATH)
    assert meta.tasks == [
        {
            "prompt": "Run test.",
            "description": "Longer task description if need.",
        }
    ]


def episodes():
    meta = Metadata(METADATA_PATH)
    assert meta.episodes == [
        {"id": "0", "success": False, "task_index": 0},
        {"id": "3", "success": True, "task_index": 0},
    ]


def num_episodes():
    meta = Metadata(METADATA_PATH)
    assert meta.num_episodes == 2


def test_equipment():
    meta = Metadata(METADATA_PATH)
    assert meta.equipment.id == "Test"
    assert meta.equipment.version == "1.0"


def test_embodiments():
    meta = Metadata(METADATA_PATH)
    embodiments = meta.equipment.embodiments
    assert set(embodiments) == {"arms"}
    assert embodiments["arms"].id == "OpenArm"


def test_perceptions():
    meta = Metadata(METADATA_PATH)
    perceptions = meta.equipment.perceptions
    assert set(perceptions.cameras) == {
        "ceiling",
        "head",
        "left_wrist",
        "right_wrist",
    }


def test_embodiment():
    meta = Metadata(METADATA_PATH)
    arms = meta.equipment.embodiments["arms"]
    assert arms.name == "arms"
    assert arms.id == "OpenArm"
    assert arms.version == "2.0"
    assert arms.components == ("right", "left")
    assert arms.attributes == ("qpos",)
    assert arms.joints == (
        "joint1",
        "joint2",
        "joint3",
        "joint4",
        "joint5",
        "joint6",
        "joint7",
        "gripper",
    )
