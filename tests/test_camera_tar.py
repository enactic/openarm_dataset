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

import tarfile
from pathlib import Path

import numpy.testing as npt
import pytest

from openarm_dataset.camera import Camera

DIR_CAMERA = (
    Path(__file__).parent
    / "fixture"
    / "dataset_0.3.0"
    / "episodes"
    / "0"
    / "cameras"
    / "ceiling"
)

EXPECTED_TIMESTAMPS = [
    1772010251.619682,
    1772010251.6290832,
    1772010251.6632507,
]


def _make_tar_camera(tmp_path) -> Camera:
    """Pack the dir fixture into cameras/ceiling.tar and return a tar Camera."""
    cameras_dir = tmp_path / "cameras"
    Camera("ceiling", DIR_CAMERA).write(cameras_dir / "ceiling.tar", format="tar")
    # base_path points at the (non-existent) directory; Camera falls back to the
    # sibling .tar, exactly like a real tar-backed dataset on disk.
    return Camera("ceiling", cameras_dir / "ceiling")


@pytest.fixture
def camera(request, tmp_path):
    if request.param == "dir":
        return Camera("ceiling", DIR_CAMERA)
    else:
        return _make_tar_camera(tmp_path)


# Run every read-API assertion against both the directory and the tar backend so
# they are verified to behave identically.
parametrize_backends = pytest.mark.parametrize("camera", ["dir", "tar"], indirect=True)


@parametrize_backends
def test_num_frames(camera):
    assert camera.num_frames == 3


@parametrize_backends
def test_get_frame(camera):
    frame = camera.get_frame(0)
    assert frame.timestamp == pytest.approx(1772010251.619682)
    assert frame.load().shape == (600, 960, 3)


@parametrize_backends
def test_frames(camera):
    frame = next(camera.frames())
    assert frame.timestamp == pytest.approx(1772010251.619682)
    assert frame.load().shape == (600, 960, 3)


@parametrize_backends
def test_load_timestamps(camera):
    npt.assert_allclose(camera.load_timestamps(), EXPECTED_TIMESTAMPS)


def test_tar_frame_load_matches_source(tmp_path):
    tar_camera = _make_tar_camera(tmp_path)
    dir_camera = Camera("ceiling", DIR_CAMERA)
    npt.assert_array_equal(
        tar_camera.get_frame(0).load(), dir_camera.get_frame(0).load()
    )


def test_materialize_extracts_tar_frame(tmp_path):
    tar_camera = _make_tar_camera(tmp_path)
    frame = tar_camera.get_frame(0)
    out_dir = tmp_path / "materialized"
    out_dir.mkdir()
    real_path = frame.materialize(out_dir)
    assert real_path.exists()


def test_materialize_dir_frame_is_zero_copy(tmp_path):
    frame = Camera("ceiling", DIR_CAMERA).get_frame(0)
    out_dir = tmp_path / "materialized"
    out_dir.mkdir()
    real_path = frame.materialize(out_dir)
    # Directory-backed frames return their own path without copying.
    assert real_path == frame.path
    assert list(out_dir.iterdir()) == []


@pytest.mark.parametrize("src_backend", ["dir", "tar"])
@pytest.mark.parametrize("dst_format", ["dir", "tar"])
def test_write_roundtrip_all_combos(tmp_path, src_backend, dst_format):
    if src_backend == "dir":
        src = Camera("ceiling", DIR_CAMERA)
    else:
        src = _make_tar_camera(tmp_path / "src")

    out = tmp_path / "out"
    if dst_format == "tar":
        src.write(out / "ceiling.tar", format="tar")
        result = Camera("ceiling", out / "ceiling")
        assert (out / "ceiling.tar").is_file()
    else:
        src.write(out / "ceiling", format="dir")
        result = Camera("ceiling", out / "ceiling")
        assert (out / "ceiling").is_dir()

    assert result.num_frames == 3
    npt.assert_allclose(result.load_timestamps(), EXPECTED_TIMESTAMPS)


def test_written_tar_is_uncompressed_and_flat(tmp_path):
    out = tmp_path / "ceiling.tar"
    Camera("ceiling", DIR_CAMERA).write(out, format="tar")
    with tarfile.open(out, mode="r:") as tf:  # mode r: requires uncompressed
        names = tf.getnames()
    assert sorted(names) == [
        "1772010251619682157.jpeg",
        "1772010251629083055.jpeg",
        "1772010251663250683.jpeg",
    ]
