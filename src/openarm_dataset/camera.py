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

"""Camera data for OpenArm Dataset."""

import io
import os
import shutil
import tarfile
from pathlib import Path
from collections.abc import Iterator

import numpy as np
from PIL import Image

from .mp4 import Mp4Video, write_mp4

# Quality of JPEGs produced from mp4-backed frames (materialize, dir/tar output).
JPEG_QUALITY = 95


class Frame:
    """An image in camera.

    A frame is backed by a JPEG file on disk, by a member inside a tar archive,
    or by a frame of an mp4 video. For tar- and mp4-backed frames ``path`` is a
    synthetic ``<archive>/<timestamp>.jpeg`` path that locates the image inside
    the archive; it is not a real file, so use :meth:`load` or
    :meth:`open_image` to access the image data.
    """

    def __init__(
        self,
        path: os.PathLike,
        *,
        tar_path: os.PathLike | None = None,
        offset: int | None = None,
        size: int | None = None,
        video: Mp4Video | None = None,
        index: int | None = None,
    ):
        """Initialize Frame.

        Args:
            path: JPEG file path (directory-backed) or synthetic
                ``<archive>/<timestamp>.jpeg`` path (tar- or mp4-backed).
            tar_path: Path to the tar archive, if this frame is tar-backed.
            offset: Byte offset of the image data inside the tar archive.
            size: Size of the image data in bytes inside the tar archive.
            video: The video reader, if this frame is mp4-backed.
            index: Index of this frame in the video, if mp4-backed.

        """
        self.path = Path(path)
        self._tar_path = Path(tar_path) if tar_path is not None else None
        self._offset = offset
        self._size = size
        self._video = video
        self._index = index
        self.timestamp: float = self._get_timestamp()

    def __eq__(self, other):
        """Compare whether the other is the same frame or not."""
        if not isinstance(other, self.__class__):
            return NotImplemented
        return self.path == other.path

    @property
    def timestamp_ns(self) -> int:
        """Unix time of this frame in nanoseconds."""
        return int(self.path.stem)

    @property
    def size(self) -> int:
        """Size of the image in bytes.

        For mp4-backed frames this is the average encoded frame size of the
        video, as individual frames are not stored separately.
        """
        if self._tar_path is not None:
            return self._size
        elif self._video is not None:
            return self._video.bytes_per_frame
        else:
            return self.path.stat().st_size

    def _read_bytes(self) -> bytes:
        """Return the frame as JPEG bytes (encoded on the fly for mp4)."""
        if self._tar_path is not None:
            with open(self._tar_path, "rb") as f:
                f.seek(self._offset)
                return f.read(self._size)
        elif self._video is not None:
            buffer = io.BytesIO()
            with self.open_image() as image:
                image.save(buffer, format="JPEG", quality=JPEG_QUALITY)
            return buffer.getvalue()
        else:
            return self.path.read_bytes()

    def open_image(self) -> Image.Image:
        """Open the image of this frame as a PIL Image.

        Returns:
            PIL Image.

        """
        if self._tar_path is not None:
            return Image.open(io.BytesIO(self._read_bytes()))
        elif self._video is not None:
            return Image.fromarray(self.load())
        else:
            return Image.open(self.path)

    def load(self) -> np.ndarray:
        """Load image of this frame.

        Returns:
            Image array.

        """
        if self._video is not None:
            return self._video.read(self._index)
        with self.open_image() as image:
            return np.array(image)

    def show(self):
        """Show image of this frame."""
        with self.open_image() as image:
            return image.show()

    def materialize(self, temp_dir: os.PathLike) -> Path:
        """Return a real on-disk path to this frame's JPEG.

        Directory-backed frames return their existing path without copying.
        Tar-backed frames are extracted (and mp4-backed frames decoded and
        JPEG-encoded) into ``temp_dir`` under their ``<timestamp>.jpeg`` name
        and that path returned.

        Args:
            temp_dir: Directory to extract tar- or mp4-backed frames into.

        Returns:
            Path to a real JPEG file on disk.

        """
        if self._tar_path is not None or self._video is not None:
            out_path = Path(temp_dir) / self.path.name
            out_path.write_bytes(self._read_bytes())
            return out_path
        else:
            return self.path

    def _get_timestamp(self) -> float:
        return float(self.path.stem) / 1e9


class Camera:
    """Camera for OpenArm Dataset."""

    def __init__(
        self,
        name: str,
        base_path: str | os.PathLike,
    ):
        """Initialize Camera.

        Args:
            name: Camera name.
            base_path: Directory-style path to the camera (e.g.
                ``.../cameras/ceiling``). If that directory does not exist but a
                sibling ``.../cameras/ceiling.tar`` archive or
                ``.../cameras/ceiling.mp4`` video does, the camera is read from
                that file instead.

        """
        self.name: str = name
        self.base_path = Path(base_path)
        self.tar_path: Path | None = None
        self.mp4_path: Path | None = None
        if not self.base_path.is_dir():
            tar_path = self.base_path.with_suffix(".tar")
            mp4_path = self.base_path.with_suffix(".mp4")
            if tar_path.is_file():
                self.tar_path = tar_path
            elif mp4_path.is_file():
                self.mp4_path = mp4_path

        self._video: Mp4Video | None = None
        if self.tar_path is not None:
            self.all_files: list[Path] = []
            self._members: list[tuple[str, int, int]] = self._load_tar_members(
                self.tar_path
            )
        elif self.mp4_path is not None:
            self.all_files = []
            self._members = []
            self._video = Mp4Video(self.mp4_path)
        else:
            self.all_files = (
                sorted(f for f in base_path.iterdir() if f.is_file())
                if base_path.exists()
                else []
            )
            self._members = []

    @staticmethod
    def _load_tar_members(tar_path: Path) -> list[tuple[str, int, int]]:
        members: list[tuple[str, int, int]] = []
        with tarfile.open(tar_path, mode="r:") as tf:
            for m in tf.getmembers():
                if m.isfile():
                    members.append((m.name, m.offset_data, m.size))
        members.sort(key=lambda t: Path(t[0]).name)
        return members

    def _tar_frame(self, name: str, offset: int, size: int) -> Frame:
        return Frame(
            self.tar_path / Path(name).name,
            tar_path=self.tar_path,
            offset=offset,
            size=size,
        )

    def _mp4_frame(self, index: int) -> Frame:
        return Frame(
            self.mp4_path / f"{self._video.timestamps_ns[index]}.jpeg",
            video=self._video,
            index=index,
        )

    @property
    def num_frames(self) -> int:
        """Get number of frames."""
        if self.tar_path is not None:
            return len(self._members)
        elif self.mp4_path is not None:
            return self._video.num_frames
        else:
            return len(self.all_files)

    @property
    def format(self) -> str:
        """Get camera format: "dir", "tar" or "mp4"."""
        if self.tar_path is not None:
            return "tar"
        elif self.mp4_path is not None:
            return "mp4"
        else:
            return "dir"

    def get_frame(self, index: int) -> Frame:
        """Get frame at the index.

        Args:
            index: Index to get.

        Returns:
            Frame at the index.

        """
        if self.tar_path is not None:
            return self._tar_frame(*self._members[index])
        elif self.mp4_path is not None:
            return self._mp4_frame(index)
        else:
            return Frame(self.all_files[index])

    def frames(self) -> Iterator[Frame]:
        """Iterate all frames.

        Returns:
            Iterator of Frame.

        """
        if self.tar_path is not None:
            for member in self._members:
                yield self._tar_frame(*member)
        elif self.mp4_path is not None:
            for index in range(self._video.num_frames):
                yield self._mp4_frame(index)
        else:
            for file in self.all_files:
                yield Frame(file)

    def load_timestamps(self) -> list[float]:
        """Load timestamps.

        Returns:
            List of Unix time.

        """
        return [frame.timestamp for frame in self.frames()]

    def write(self, output: os.PathLike, format):
        """Write this camera's frames to ``output`` in the specified format.

        Converting from mp4 re-encodes the decoded frames as JPEG; converting to
        mp4 is lossy. A camera without frames writes no mp4 file.

        Args:
            output: Destination path. For "dir" format, a directory that must
                not already exist; for "tar" and "mp4" formats, the file to
                write.
            format: Output format: "dir" for directory of JPEGs, "tar" for
                uncompressed tar archive or "mp4" for H.264 video.

        """
        if format == "dir":
            dest_dir = Path(output)
            if self.format == "dir":
                shutil.copytree(self.base_path, dest_dir)
                return
            dest_dir.mkdir(parents=True)
            for frame in self.frames():
                (dest_dir / frame.path.name).write_bytes(frame._read_bytes())

        elif format == "tar":
            dest_tar = Path(output).with_suffix(".tar")
            dest_tar.parent.mkdir(parents=True, exist_ok=True)
            if self.format == "tar":
                shutil.copy2(self.tar_path, dest_tar)
                return
            with tarfile.open(dest_tar, mode="w") as tf:
                if self.format == "dir":
                    for file in self.all_files:
                        tf.add(file, arcname=file.name)
                    return
                for frame in self.frames():
                    data = frame._read_bytes()
                    info = tarfile.TarInfo(frame.path.name)
                    info.size = len(data)
                    info.mtime = int(frame.timestamp)
                    tf.addfile(info, io.BytesIO(data))

        elif format == "mp4":
            dest_mp4 = Path(output).with_suffix(".mp4")
            dest_mp4.parent.mkdir(parents=True, exist_ok=True)
            if self.format == "mp4":
                shutil.copy2(self.mp4_path, dest_mp4)
                return
            write_mp4(dest_mp4, self.frames())
        else:
            raise ValueError(f"Unsupported format: {format}")
