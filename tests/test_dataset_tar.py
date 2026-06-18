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

"""End-to-end: write a dataset with tar-packed cameras and read it back."""

from pathlib import Path

import pytest

from openarm_dataset.dataset import Dataset

DATASET_DIR = Path(__file__).parent / "fixture" / "dataset_0.3.0"


@pytest.fixture
def tar_dataset(tmp_path):
    out = tmp_path / "dataset_tar"
    Dataset(DATASET_DIR).write(out, format="openarm", camera_format="tar")
    return Dataset(out)


def test_cameras_written_as_tar(tar_dataset):
    cameras_dir = tar_dataset.root_path / "episodes" / "0" / "cameras"
    # Each camera is a single .tar file, not a directory of JPEGs.
    assert (cameras_dir / "ceiling.tar").is_file()
    assert not (cameras_dir / "ceiling").exists()


def test_load_cameras_from_tar(tar_dataset):
    cameras = tar_dataset.load_cameras(tar_dataset.meta.episodes[0])
    assert set(cameras) == {"ceiling", "head", "wrist_left", "wrist_right"}
    assert cameras["ceiling"].num_frames == 3


def test_frame_load_from_tar(tar_dataset):
    camera = tar_dataset.load_camera("ceiling", tar_dataset.meta.episodes[0])
    frame = camera.get_frame(0)
    assert frame.timestamp == pytest.approx(1772010251.619682)
    assert frame.load().shape == (600, 960, 3)
    # Tar-backed frames expose a synthetic path pointing into the archive.
    assert frame.path.parent.name == "ceiling.tar"


def test_sample_from_tar(tar_dataset):
    samples = tar_dataset.sample(hz=30, episode=tar_dataset.meta.episodes[0])
    assert len(samples) > 1
    assert set(samples[0].cameras) == {
        "ceiling",
        "head",
        "wrist_left",
        "wrist_right",
    }
    assert samples[0].cameras["ceiling"].load().shape == (600, 960, 3)


def test_dir_default_unchanged(tmp_path):
    out = tmp_path / "dataset_dir"
    Dataset(DATASET_DIR).write(out, format="openarm")
    cameras_dir = out / "episodes" / "0" / "cameras"
    # Default output keeps the per-frame JPEG directory layout.
    assert (cameras_dir / "ceiling").is_dir()
    assert not (cameras_dir / "ceiling.tar").exists()


def test_tar_input_roundtrips_to_dir(tmp_path):
    # tar input -> dir output (the extract path) reads back correctly.
    tar_out = tmp_path / "tar"
    Dataset(DATASET_DIR).write(tar_out, format="openarm", camera_format="tar")

    dir_out = tmp_path / "dir"
    Dataset(tar_out).write(dir_out, format="openarm", camera_format="dir")

    camera = Dataset(dir_out).load_camera("ceiling", Dataset(dir_out).meta.episodes[0])
    assert camera.tar_path is None
    assert camera.num_frames == 3
    assert camera.get_frame(0).load().shape == (600, 960, 3)


def test_camera_format_dir():
    assert Dataset(DATASET_DIR).camera_format == "dir"


def test_camera_format_tar(tar_dataset):
    assert tar_dataset.camera_format == "tar"


def test_camera_format_inconsistent_raises(tmp_path):
    out = tmp_path / "mixed"
    Dataset(DATASET_DIR).write(out, format="openarm", camera_format="tar")
    # Turn one camera back into "dir" layout so the dataset mixes both formats.
    (out / "episodes" / "0" / "cameras" / "head").mkdir()
    with pytest.raises(ValueError, match="Inconsistent camera formats"):
        Dataset(out).camera_format
