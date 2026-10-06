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

import shutil
import tarfile
from pathlib import Path

import av
import numpy as np
import numpy.testing as npt
import pytest
from PIL import Image

from openarm_dataset import ffmpeg
from openarm_dataset.camera import Camera
from openarm_dataset.mp4 import TIMESTAMPS_METADATA_KEY

DIR_CAMERA = (
    Path(__file__).parent
    / "fixture"
    / "dataset_0.3.0"
    / "episodes"
    / "0"
    / "cameras"
    / "ceiling"
)

EXPECTED_NAMES = [
    "1772010251619682157.jpeg",
    "1772010251629083055.jpeg",
    "1772010251663250683.jpeg",
]

EXPECTED_TIMESTAMPS = [
    1772010251.619682,
    1772010251.6290832,
    1772010251.6632507,
]


def _make_mp4_camera(tmp_path) -> Camera:
    """Encode the dir fixture into cameras/ceiling.mp4 and return an mp4 Camera."""
    cameras_dir = tmp_path / "cameras"
    Camera("ceiling", DIR_CAMERA).write(cameras_dir / "ceiling.mp4", format="mp4")
    # base_path points at the (non-existent) directory; Camera falls back to the
    # sibling .mp4, exactly like a real mp4-backed dataset on disk.
    return Camera("ceiling", cameras_dir / "ceiling")


def _make_tar_camera(tmp_path) -> Camera:
    cameras_dir = tmp_path / "cameras"
    Camera("ceiling", DIR_CAMERA).write(cameras_dir / "ceiling.tar", format="tar")
    return Camera("ceiling", cameras_dir / "ceiling")


def _make_synthetic_dir_camera(tmp_path, num_frames: int) -> Camera:
    """Create a directory camera of smooth synthetic frames.

    The frame index is encoded in the (flat) red channel so a decoded frame can
    be identified even after lossy video compression; green and blue carry
    gradients so the content compresses like a natural image, unlike the
    noise-like fixture JPEGs.
    """
    camera_dir = tmp_path / "synthetic"
    camera_dir.mkdir()
    start = 1_700_000_000_000_000_000
    for i in range(num_frames):
        image = np.zeros((48, 64, 3), dtype=np.uint8)
        image[..., 0] = i * 3
        image[..., 1] = np.linspace(0, 255, 64, dtype=np.uint8)[np.newaxis, :]
        image[..., 2] = np.linspace(255, 0, 48, dtype=np.uint8)[:, np.newaxis]
        # Uneven spacing: video timestamps must not assume a fixed frame rate.
        timestamp_ns = start + i * 33_000_000 + (i % 5) * 1_000_003
        Image.fromarray(image).save(camera_dir / f"{timestamp_ns}.jpeg", quality=95)
    return Camera("synthetic", camera_dir)


@pytest.fixture
def camera(request, tmp_path):
    if request.param == "dir":
        return Camera("ceiling", DIR_CAMERA)
    else:
        return _make_mp4_camera(tmp_path)


# Run every read-API assertion against both the directory and the mp4 backend so
# they are verified to behave identically.
parametrize_backends = pytest.mark.parametrize("camera", ["dir", "mp4"], indirect=True)


@parametrize_backends
def test_num_frames(camera):
    assert camera.num_frames == 3


@parametrize_backends
def test_get_frame(camera):
    frame = camera.get_frame(0)
    assert frame.timestamp == pytest.approx(1772010251.619682)
    assert frame.load().shape == (600, 960, 3)
    assert frame.load().dtype == np.uint8


@parametrize_backends
def test_frames(camera):
    frame = next(camera.frames())
    assert frame.timestamp == pytest.approx(1772010251.619682)
    assert frame.load().shape == (600, 960, 3)


@parametrize_backends
def test_load_timestamps(camera):
    npt.assert_allclose(camera.load_timestamps(), EXPECTED_TIMESTAMPS)


def test_mp4_camera_format(tmp_path):
    mp4_camera = _make_mp4_camera(tmp_path)
    assert mp4_camera.format == "mp4"
    assert mp4_camera.tar_path is None
    assert mp4_camera.mp4_path == tmp_path / "cameras" / "ceiling.mp4"


def test_mp4_frame_path_is_synthetic(tmp_path):
    mp4_camera = _make_mp4_camera(tmp_path)
    frame = mp4_camera.get_frame(0)
    assert frame.path == tmp_path / "cameras" / "ceiling.mp4" / EXPECTED_NAMES[0]
    assert not frame.path.exists()


def test_mp4_timestamps_are_exact(tmp_path):
    # Nanosecond timestamps survive the round trip bit for bit even though
    # video presentation timestamps only have microsecond resolution.
    mp4_camera = _make_mp4_camera(tmp_path)
    assert [frame.path.name for frame in mp4_camera.frames()] == EXPECTED_NAMES


def test_mp4_frame_load_close_to_source(tmp_path):
    source = _make_synthetic_dir_camera(tmp_path, 10)
    source.write(tmp_path / "synthetic.mp4", format="mp4")
    mp4_camera = Camera("synthetic", tmp_path / "synthetic")
    for index in (0, 5, 9):
        decoded = mp4_camera.get_frame(index).load().astype(np.float64)
        expected = source.get_frame(index).load().astype(np.float64)
        # Lossy H.264, so only require the images to be close on average.
        assert np.abs(decoded - expected).mean() < 3


def test_mp4_frame_open_image(tmp_path):
    mp4_camera = _make_mp4_camera(tmp_path)
    with mp4_camera.get_frame(1).open_image() as image:
        assert image.size == (960, 600)
        assert image.mode == "RGB"


def test_mp4_frame_size_is_positive(tmp_path):
    mp4_camera = _make_mp4_camera(tmp_path)
    assert mp4_camera.get_frame(0).size > 0


def test_random_access_matches_sequential(tmp_path):
    num_frames = 70  # longer than one keyframe interval so seeks are exercised
    source = _make_synthetic_dir_camera(tmp_path, num_frames)
    source.write(tmp_path / "synthetic.mp4", format="mp4")
    mp4_camera = Camera("synthetic", tmp_path / "synthetic")
    assert mp4_camera.num_frames == num_frames

    sequential = [frame.load() for frame in mp4_camera.frames()]
    for index in [69, 5, 40, 30, 31, 29, 0, 68, 1]:
        decoded = mp4_camera.get_frame(index).load()
        # Same decoder output regardless of the access order.
        npt.assert_array_equal(decoded, sequential[index])
        # And it is the right frame: the index is encoded in the red channel.
        assert abs(int(decoded[..., 0].mean()) - index * 3) <= 4


def test_reading_last_frame_closes_video(tmp_path):
    mp4_camera = _make_mp4_camera(tmp_path)
    video = mp4_camera._video
    for frame in mp4_camera.frames():
        frame.load()
    # Sequential readers (e.g. video re-encoding of every episode) must not
    # leave one open file per camera behind.
    assert video._container is None
    # Reading again after the close transparently reopens the file.
    assert mp4_camera.get_frame(0).load().shape == (600, 960, 3)


def test_random_access_timestamps_exact(tmp_path):
    source = _make_synthetic_dir_camera(tmp_path, 70)
    source.write(tmp_path / "synthetic.mp4", format="mp4")
    mp4_camera = Camera("synthetic", tmp_path / "synthetic")
    assert mp4_camera.load_timestamps() == source.load_timestamps()


def test_materialize_extracts_mp4_frame(tmp_path):
    mp4_camera = _make_mp4_camera(tmp_path)
    frame = mp4_camera.get_frame(0)
    out_dir = tmp_path / "materialized"
    out_dir.mkdir()
    real_path = frame.materialize(out_dir)
    assert real_path == out_dir / EXPECTED_NAMES[0]
    with Image.open(real_path) as image:
        assert image.size == (960, 600)


@pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="ffmpeg not installed")
def test_encode_mp4_from_mp4_frames(tmp_path):
    # The lerobot/rrd writers re-encode frames with the ffmpeg CLI through
    # Frame.materialize(); mp4-backed frames must work there too.
    mp4_camera = _make_mp4_camera(tmp_path)
    out = tmp_path / "out.mp4"
    ffmpeg.encode_mp4(list(mp4_camera.frames()), 30, out, verbose=False)
    assert out.stat().st_size > 0


@pytest.mark.parametrize("src_backend", ["dir", "tar", "mp4"])
@pytest.mark.parametrize("dst_format", ["dir", "tar", "mp4"])
def test_write_roundtrip_all_combos(tmp_path, src_backend, dst_format):
    if src_backend == "dir":
        src = Camera("ceiling", DIR_CAMERA)
    elif src_backend == "tar":
        src = _make_tar_camera(tmp_path / "src")
    else:
        src = _make_mp4_camera(tmp_path / "src")

    out = tmp_path / "out"
    if dst_format == "dir":
        src.write(out / "ceiling", format="dir")
        assert (out / "ceiling").is_dir()
    else:
        src.write(out / f"ceiling.{dst_format}", format=dst_format)
        assert (out / f"ceiling.{dst_format}").is_file()
    result = Camera("ceiling", out / "ceiling")

    assert result.format == dst_format
    assert result.num_frames == 3
    assert [frame.path.name for frame in result.frames()] == EXPECTED_NAMES
    assert result.get_frame(2).load().shape == (600, 960, 3)


def test_mp4_to_tar_is_uncompressed_and_flat(tmp_path):
    mp4_camera = _make_mp4_camera(tmp_path)
    out = tmp_path / "out" / "ceiling.tar"
    mp4_camera.write(out, format="tar")
    with tarfile.open(out, mode="r:") as tf:  # mode r: requires uncompressed
        names = tf.getnames()
    assert sorted(names) == EXPECTED_NAMES


def test_written_mp4_is_h264_with_timestamps(tmp_path):
    out = tmp_path / "ceiling.mp4"
    Camera("ceiling", DIR_CAMERA).write(out, format="mp4")
    with av.open(str(out)) as container:
        stream = container.streams.video[0]
        assert stream.codec_context.name == "h264"
        assert stream.frames == 3
        assert (stream.width, stream.height) == (960, 600)
        timestamps = container.metadata[TIMESTAMPS_METADATA_KEY]
    assert timestamps == ",".join(name.removesuffix(".jpeg") for name in EXPECTED_NAMES)


def test_mp4_without_timestamps_metadata_raises(tmp_path):
    out = tmp_path / "cameras" / "ceiling.mp4"
    out.parent.mkdir()
    with av.open(str(out), mode="w") as container:
        stream = container.add_stream("libx264", rate=30)
        stream.width, stream.height = 64, 48
        stream.pix_fmt = "yuv420p"
        frame = av.VideoFrame.from_ndarray(
            np.zeros((48, 64, 3), dtype=np.uint8), format="rgb24"
        )
        for packet in stream.encode(frame):
            container.mux(packet)
        for packet in stream.encode():
            container.mux(packet)
    with pytest.raises(ValueError, match=TIMESTAMPS_METADATA_KEY):
        Camera("ceiling", tmp_path / "cameras" / "ceiling")


def test_empty_camera_writes_no_mp4(tmp_path):
    empty = Camera("missing", tmp_path / "missing")
    assert empty.num_frames == 0
    empty.write(tmp_path / "out" / "missing.mp4", format="mp4")
    assert not (tmp_path / "out" / "missing.mp4").exists()
