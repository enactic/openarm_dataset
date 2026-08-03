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

METADATA_PATH = (
    Path(__file__).parent / "fixture" / "dataset_0.4.0_qpos" / "metadata.yaml"
)


def test_leader_device_type():
    meta = Metadata(METADATA_PATH)
    assert meta.leader_device_type == "OpenArmKER"


def test_leader_device_type_missing():
    meta = Metadata(
        Path(__file__).parent / "fixture" / "dataset_0.3.0" / "metadata.yaml"
    )
    assert meta.leader_device_type is None
