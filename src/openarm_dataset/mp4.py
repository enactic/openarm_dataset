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

"""MP4 storage for camera frames.

A camera's frames can be packed into a single ``cameras/<name>.mp4`` H.264
video. The video is self-contained: the exact nanosecond timestamp of every
frame is stored in the container metadata (``openarm_timestamps``, a
comma-separated list in frame order), and the frames' presentation timestamps
carry the same timing at microsecond resolution so the video plays back at
the recorded pace in any player.
"""

import os
from collections.abc import Iterable
from fractions import Fraction
from pathlib import Path

import av
import numpy as np

TIMESTAMPS_METADATA_KEY = "openarm_timestamps"

CODEC = "libx264"
PIX_FMT = "yuv420p"
# Microsecond presentation timestamps; the exact nanosecond timestamps live in
# the container metadata.
TIME_BASE = Fraction(1, 1_000_000)
# Keyframe interval in frames. Random access decodes at most this many frames.
GOP_SIZE = 30
ENCODER_OPTIONS = {"preset": "veryfast", "crf": "23"}


def _timestamps_to_pts(timestamps_ns: list[int]) -> list[int]:
    """Convert nanosecond timestamps to strictly increasing microsecond PTS.

    PTS are relative to the first frame so the video starts at zero.
    """
    pts = []
    previous = -1
    for timestamp_ns in timestamps_ns:
        # Strictly increasing PTS are required by the muxer.
        value = max((timestamp_ns - timestamps_ns[0]) // 1000, previous + 1)
        pts.append(value)
        previous = value
    return pts


def write_mp4(path: os.PathLike, frames: Iterable) -> None:
    """Encode camera frames into an H.264 MP4 file.

    Args:
        path: Output ``.mp4`` path.
        frames: :class:`~openarm_dataset.camera.Frame` objects in timestamp
            order. Nothing is written when there are no frames.

    Raises:
        ValueError: If the frames do not all have the same size.

    """
    frames = list(frames)
    if not frames:
        return
    timestamps_ns = [frame.timestamp_ns for frame in frames]
    pts = _timestamps_to_pts(timestamps_ns)

    with av.open(str(path), mode="w", options={"movflags": "use_metadata_tags"}) as f:
        f.metadata[TIMESTAMPS_METADATA_KEY] = ",".join(map(str, timestamps_ns))
        stream = None
        for frame, frame_pts in zip(frames, pts):
            with frame.open_image() as image:
                video_frame = av.VideoFrame.from_image(image.convert("RGB"))
            if stream is None:
                stream = f.add_stream(CODEC)
                stream.width = video_frame.width
                stream.height = video_frame.height
                stream.pix_fmt = PIX_FMT
                stream.time_base = TIME_BASE
                # The encoder rescales frame PTS into its own time base; leave
                # it at the default frame rate and the PTS get quantized.
                stream.codec_context.time_base = TIME_BASE
                stream.codec_context.gop_size = GOP_SIZE
                stream.options = dict(ENCODER_OPTIONS)
            elif (video_frame.width, video_frame.height) != (
                stream.width,
                stream.height,
            ):
                raise ValueError(
                    f"Frame {frame.path.name} is {video_frame.width}x"
                    f"{video_frame.height} but the video is "
                    f"{stream.width}x{stream.height}: all frames of a camera "
                    "must have the same size to be written as mp4"
                )
            video_frame.pts = frame_pts
            video_frame.time_base = TIME_BASE
            for packet in stream.encode(video_frame):
                f.mux(packet)
        for packet in stream.encode():
            f.mux(packet)


class Mp4Video:
    """Random access reader for a camera ``.mp4`` written by :func:`write_mp4`.

    Frames are decoded lazily. Reading frames in increasing order reuses the
    running decoder; reading backwards (or far ahead) seeks to the nearest
    preceding keyframe first. The file is closed again once the last frame has
    been read, so iterating over many videos does not accumulate open files.
    """

    def __init__(self, path: os.PathLike):
        """Initialize Mp4Video.

        Args:
            path: Path to the ``.mp4`` file.

        Raises:
            ValueError: If the file does not carry the frame timestamps written
                by :func:`write_mp4`, or its frame count does not match them.

        """
        self.path = Path(path)
        self._container = None
        self._stream = None
        self._decoder = None
        # Index of the frame the running decoder yields next.
        self._next_index = 0
        with av.open(str(self.path)) as f:
            timestamps = f.metadata.get(TIMESTAMPS_METADATA_KEY)
            if timestamps is None:
                raise ValueError(
                    f"{self.path} has no {TIMESTAMPS_METADATA_KEY!r} metadata; "
                    "only mp4 files written by openarm_dataset are supported"
                )
            self.timestamps_ns: list[int] = [
                int(timestamp) for timestamp in timestamps.split(",") if timestamp
            ]
            stream = f.streams.video[0]
            if stream.frames and stream.frames != len(self.timestamps_ns):
                raise ValueError(
                    f"{self.path} has {stream.frames} frames but "
                    f"{len(self.timestamps_ns)} timestamps"
                )
            if stream.time_base != TIME_BASE:
                raise ValueError(
                    f"{self.path} has time base {stream.time_base}, expected "
                    f"{TIME_BASE}; only mp4 files written by openarm_dataset "
                    "are supported"
                )
        self._pts = _timestamps_to_pts(self.timestamps_ns)
        self._index_by_pts = {pts: index for index, pts in enumerate(self._pts)}

    @property
    def num_frames(self) -> int:
        """Get number of frames."""
        return len(self.timestamps_ns)

    @property
    def bytes_per_frame(self) -> int:
        """Average encoded size of one frame in bytes."""
        return self.path.stat().st_size // max(self.num_frames, 1)

    def read(self, index: int) -> np.ndarray:
        """Decode the frame at the index.

        Args:
            index: Frame index.

        Returns:
            RGB image array of shape ``(height, width, 3)``.

        """
        if not 0 <= index < self.num_frames:
            raise IndexError(f"Frame index {index} out of range")
        if (
            self._decoder is None
            or index < self._next_index
            or index - self._next_index > GOP_SIZE
        ):
            self._seek(index)
        for video_frame in self._decoder:
            decoded_index = self._index_by_pts.get(video_frame.pts)
            if decoded_index is None:
                raise ValueError(
                    f"{self.path} contains a frame with unexpected pts "
                    f"{video_frame.pts}"
                )
            self._next_index = decoded_index + 1
            if decoded_index == index:
                image = video_frame.to_ndarray(format="rgb24")
                if index == self.num_frames - 1:
                    # A sequential reader is done; release the file handle.
                    self.close()
                return image
            if decoded_index > index:
                break
        raise ValueError(f"Frame {index} not found in {self.path}")

    def _seek(self, index: int):
        if self._container is None:
            self._container = av.open(str(self.path))
            self._stream = self._container.streams.video[0]
            self._stream.thread_type = "AUTO"
        # Land on the last keyframe at or before the target and decode forward.
        self._container.seek(
            self._pts[index], stream=self._stream, backward=True, any_frame=False
        )
        self._decoder = self._container.decode(self._stream)
        self._next_index = index

    def close(self):
        """Close the underlying video file."""
        self._decoder = None
        self._stream = None
        if self._container is not None:
            self._container.close()
            self._container = None

    def __del__(self):
        """Close the video file when garbage collected."""
        self.close()
