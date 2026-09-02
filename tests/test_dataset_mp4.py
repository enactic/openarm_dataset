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

"""End-to-end: write a dataset with mp4-packed cameras and read it back."""

from pathlib import Path

import pytest

from openarm_dataset.dataset import Dataset

DATASET_DIR = Path(__file__).parent / "fixture" / "dataset_0.3.0"


@pytest.fixture
def mp4_dataset(tmp_path):
    out = tmp_path / "dataset_mp4"
    Dataset(DATASET_DIR).write(out, format="openarm", camera_format="mp4")
    return Dataset(out)


def test_cameras_written_as_mp4(mp4_dataset):
    cameras_dir = mp4_dataset.root_path / "episodes" / "0" / "cameras"
    # Each camera is a single .mp4 file, not a directory of JPEGs.
    assert (cameras_dir / "ceiling.mp4").is_file()
    assert not (cameras_dir / "ceiling").exists()
    assert not (cameras_dir / "ceiling.tar").exists()


def test_load_cameras_from_mp4(mp4_dataset):
    cameras = mp4_dataset.load_cameras(mp4_dataset.meta.episodes[0])
    assert set(cameras) == {"ceiling", "head", "wrist_left", "wrist_right"}
    assert cameras["ceiling"].num_frames == 3


def test_frame_load_from_mp4(mp4_dataset):
    camera = mp4_dataset.load_camera("ceiling", mp4_dataset.meta.episodes[0])
    frame = camera.get_frame(0)
    assert frame.timestamp == pytest.approx(1772010251.619682)
    assert frame.load().shape == (600, 960, 3)
    # mp4-backed frames expose a synthetic path pointing into the video.
    assert frame.path.parent.name == "ceiling.mp4"


def test_sample_from_mp4(mp4_dataset):
    samples = mp4_dataset.sample(hz=30, episode=mp4_dataset.meta.episodes[0])
    assert len(samples) > 1
    assert set(samples[0].cameras) == {
        "ceiling",
        "head",
        "wrist_left",
        "wrist_right",
    }
    assert samples[0].cameras["ceiling"].load().shape == (600, 960, 3)


def test_mp4_input_roundtrips_to_dir(tmp_path):
    # mp4 input -> dir output (the decode path) reads back correctly.
    mp4_out = tmp_path / "mp4"
    Dataset(DATASET_DIR).write(mp4_out, format="openarm", camera_format="mp4")

    dir_out = tmp_path / "dir"
    Dataset(mp4_out).write(dir_out, format="openarm", camera_format="dir")

    dir_dataset = Dataset(dir_out)
    camera = dir_dataset.load_camera("ceiling", dir_dataset.meta.episodes[0])
    assert camera.format == "dir"
    assert camera.num_frames == 3
    assert camera.get_frame(0).path.name == "1772010251619682157.jpeg"
    assert camera.get_frame(0).load().shape == (600, 960, 3)


def test_mp4_input_roundtrips_to_tar(tmp_path):
    mp4_out = tmp_path / "mp4"
    Dataset(DATASET_DIR).write(mp4_out, format="openarm", camera_format="mp4")

    tar_out = tmp_path / "tar"
    Dataset(mp4_out).write(tar_out, format="openarm", camera_format="tar")

    tar_dataset = Dataset(tar_out)
    assert tar_dataset.camera_format == "tar"
    camera = tar_dataset.load_camera("ceiling", tar_dataset.meta.episodes[0])
    assert camera.num_frames == 3
    assert camera.get_frame(0).load().shape == (600, 960, 3)


def test_camera_format_mp4(mp4_dataset):
    assert mp4_dataset.camera_format == "mp4"


def test_camera_format_inconsistent_raises(tmp_path):
    out = tmp_path / "mixed"
    Dataset(DATASET_DIR).write(out, format="openarm", camera_format="mp4")
    # Turn one camera back into "dir" layout so the dataset mixes both formats.
    (out / "episodes" / "0" / "cameras" / "head").mkdir()
    with pytest.raises(ValueError, match="Inconsistent camera formats"):
        Dataset(out).camera_format
