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
import rerun.recording

from openarm_dataset import Dataset


DATASET_PATH = Path(__file__).parent / "fixture" / "dataset_0.2.0"


@pytest.fixture
def rrd_setup(tmp_path):
    dataset = Dataset(DATASET_PATH)
    rrd_path = tmp_path / "output.rrd"
    dataset.to_rrd(rrd_path)
    return dataset, rerun.recording.load_recording(str(rrd_path))


def test_to_rrd_creates_file(tmp_path):
    dataset = Dataset(DATASET_PATH)
    rrd_path = tmp_path / "output.rrd"
    dataset.to_rrd(rrd_path)
    assert rrd_path.exists()


def test_to_rrd_entities(rrd_setup):
    dataset, recording = rrd_setup

    actual = set(recording.schema().entity_paths(include_properties=False))
    expected = set()
    for ep_idx in range(dataset.num_episodes):
        ep_id = dataset.meta.episodes[ep_idx]["id"]
        for name, embodiment in dataset.meta.equipment.embodiments.items():
            for component in embodiment.components:
                for joint in embodiment.joints:
                    expected.add(f"/ep{ep_id}/action/{component}/{joint}")
                    expected.add(f"/ep{ep_id}/obs/{component}/{joint}")
        for name in dataset.meta.equipment.perceptions.cameras:
            expected.add(f"/ep{ep_id}/cameras/{name}")

    assert actual == expected


def test_to_rrd_action_values(rrd_setup):
    dataset, recording = rrd_setup

    chunks_by_path = {c.entity_path: c for c in recording.chunks()}
    for ep_idx in range(dataset.num_episodes):
        ep_id = dataset.meta.episodes[ep_idx]["id"]
        for key, df in dataset.load_action(ep_idx).items():
            side = key.split("/")[1]
            for col in df.columns:
                path = f"/ep{ep_id}/action/{side}/{col}"
                batch = chunks_by_path[path].to_record_batch()
                actual = [v[0] for v in batch["Scalars:scalars"].to_pylist()]
                assert actual == pytest.approx(df[col].tolist()), path


def test_to_rrd_obs_values(rrd_setup):
    dataset, recording = rrd_setup

    chunks_by_path = {c.entity_path: c for c in recording.chunks()}
    for ep_idx in range(dataset.num_episodes):
        ep_id = dataset.meta.episodes[ep_idx]["id"]
        for key, df in dataset.load_obs(ep_idx).items():
            side = key.split("/")[1]
            for col in df.columns:
                path = f"/ep{ep_id}/obs/{side}/{col}"
                batch = chunks_by_path[path].to_record_batch()
                actual = [v[0] for v in batch["Scalars:scalars"].to_pylist()]
                assert actual == pytest.approx(df[col].tolist()), path
