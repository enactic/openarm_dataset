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

import yaml

from openarm_dataset.metadata import Metadata

METADATA_PATH = (
    Path(__file__).parent / "fixture" / "dataset_0.4.0_pose" / "metadata.yaml"
)

ATTRIBUTES = {
    "action": {"arms": {"left": ["pose"], "right": ["pose"]}},
    "obs": {
        "arms": {
            "left": ["qpos", "qvel", "qtorque"],
            "right": ["qpos", "qvel", "qtorque"],
        }
    },
}


def test_attributes_absent():
    meta = Metadata(METADATA_PATH)
    assert meta.attributes == {}


def test_attributes_present(tmp_path):
    with open(METADATA_PATH) as f:
        data = yaml.safe_load(f)
    data["attributes"] = ATTRIBUTES
    path = tmp_path / "metadata.yaml"
    with open(path, "w") as f:
        yaml.safe_dump(data, f)

    meta = Metadata(path)
    assert meta.attributes == ATTRIBUTES
