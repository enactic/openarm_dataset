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


def test_leader_device():
    meta = Metadata(METADATA_PATH)
    leader = meta.equipment.leader
    assert list(leader) == ["ker"]
    ker = leader["ker"]
    assert ker.kind == "ker"
    assert ker.id == "OpenArmKER"
    assert ker.firmware_version == "1.2.3"
    assert ker.hardware_version == "1.0"


def test_leader_device_missing():
    meta = Metadata(
        Path(__file__).parent / "fixture" / "dataset_0.3.0" / "metadata.yaml"
    )
    assert dict(meta.equipment.leader) == {}


def test_leader_device_without_versions():
    # A device that reports no firmware/hardware version still reads back;
    # only the versions are missing.
    meta = Metadata(METADATA_PATH)
    meta.data["equipment"]["leader"] = {"vr": {"id": "VR-Quest"}}
    vr = meta.equipment.leader["vr"]
    assert vr.id == "VR-Quest"
    assert vr.firmware_version is None
    assert vr.hardware_version is None
