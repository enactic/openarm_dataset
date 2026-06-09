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

import pytest

pytest.importorskip("rerun", minversion="0.33")

from pathlib import Path

import numpy as np
import pyarrow.compute as pc
import rerun as rr

from openarm_dataset import Dataset


DATASET_PATH = Path(__file__).parent / "fixture" / "dataset_0.3.0"


@pytest.fixture(scope="module")
def dataset():
    return Dataset(DATASET_PATH)


@pytest.fixture(scope="module")
def reader(dataset, tmp_path_factory):
    rrd_path = tmp_path_factory.mktemp("rrd") / "output.rrd"
    dataset.write(rrd_path, "rrd")
    assert rrd_path.exists()

    return rr.experimental.RrdReader(rrd_path)


def _column_info(chunk_stream, entity):
    return [
        chunk.to_record_batch().schema.names
        for chunk in chunk_stream
        if chunk.entity_path == entity
    ]


def _to_record_batch(chunk_stream, entity):
    for chunk in chunk_stream:
        if chunk.entity_path == entity and not chunk.is_static:
            return chunk.to_record_batch()


def test_entities(dataset, reader):
    expected = []
    for episode in dataset.meta.episodes:
        for category in ("action", "obs"):
            for attribute in dataset.get_embodiment_attributes(category, episode):
                for joint in attribute["embodiment"].joints:
                    expected.append(
                        "/"
                        + "/".join(
                            [f"ep{episode['id']}", category, attribute["key"], joint]
                        )
                    )
        for name in dataset.camera_names:
            expected.append("/" + "/".join([f"ep{episode['id']}", "camera", name]))

    assert sorted(reader.store().schema().entity_paths()) == sorted(expected)


def test_log_arms_values(dataset, reader):
    recording = reader.store()
    for episode in dataset.meta.episodes:
        samples = dataset.sample(hz=30, episode=episode)
        for category in ("action", "obs"):
            for attribute in dataset.get_embodiment_attributes(category, episode):
                key = attribute["key"]
                joints = attribute["embodiment"].joints
                values = np.array([sample[category][key] for sample in samples])
                for i, joint in enumerate(joints):
                    entity = "/" + "/".join(
                        [f"ep{episode['id']}", category, key, joint]
                    )
                    batch = _to_record_batch(recording.stream(), entity)
                    assert pc.list_flatten(
                        batch.column("Scalars:scalars")
                    ).to_numpy() == pytest.approx(values[:, i]), entity


def test_log_arms_timestamps(dataset, reader):
    recording = reader.store()
    for episode in dataset.meta.episodes:
        samples = dataset.sample(hz=30, episode=episode)
        for category in ("action", "obs"):
            for attribute in dataset.get_embodiment_attributes(category, episode):
                key = attribute["key"]
                for joint in attribute["embodiment"].joints:
                    entity = "/" + "/".join(
                        [f"ep{episode['id']}", category, key, joint]
                    )
                    batch = _to_record_batch(recording.stream(), entity)
                    assert batch.column("timestamp").to_numpy().astype(
                        "int64"
                    ) / 1e9 == pytest.approx(
                        [sample.timestamp for sample in samples], abs=1e-6
                    ), entity


def test_log_cameras(dataset, reader):
    recording = reader.store()
    for episode in dataset.meta.episodes:
        for name in dataset.camera_names:
            entity = "/" + "/".join([f"ep{episode['id']}", "camera", name])
            column_info = _column_info(recording.stream(), entity)
            assert any(
                cols[-1] == "VideoFrameReference:timestamp" for cols in column_info
            )
            assert any(cols[-1] == "AssetVideo:media_type" for cols in column_info)


def test_blueprint_views(dataset, reader):
    blueprint = reader.store(store=reader.blueprints()[0])
    views = {}
    tab_width = []
    for chunk in blueprint.stream():
        batch = chunk.to_record_batch()
        names = batch.schema.names
        if "ViewBlueprint:space_origin" in names:
            entity = batch.column("ViewBlueprint:space_origin").to_pylist()[0][0]
            data_type = batch.column("ViewBlueprint:class_identifier").to_pylist()[0][0]
            views[entity] = data_type
        if "ContainerBlueprint:col_shares" in names:
            tab_width.append(
                batch.column("ContainerBlueprint:col_shares").to_pylist()[0]
            )
    assert tab_width and all(w == pytest.approx([0.3, 0.35, 0.35]) for w in tab_width)

    views_expected = {}
    for episode in dataset.meta.episodes:
        for category in ("action", "obs"):
            for attribute in dataset.get_embodiment_attributes(category, episode):
                entity = "/".join([f"ep{episode['id']}", category, attribute["key"]])
                views_expected[entity] = "TimeSeries"
        for name in dataset.camera_names:
            entity = "/".join([f"ep{episode['id']}", "camera", name])
            views_expected[entity] = "2D"

    assert views == views_expected
