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


class Frame:
    """An image in camera.

    A frame is backed either by a JPEG file on disk or by a member inside a tar
    archive. For tar-backed frames ``path`` is a synthetic ``<archive>/<member>``
    path that locates the image inside the archive; it is not a real file, so use
    :meth:`load` or :meth:`open_image` to access the image data.
    """

    def __init__(
        self,
        path: os.PathLike,
        *,
        tar_path: os.PathLike | None = None,
        offset: int | None = None,
        size: int | None = None,
    ):
        """Initialize Frame.

        Args:
            path: JPEG file path (directory-backed) or synthetic
                ``<archive>/<member>`` path (tar-backed).
            tar_path: Path to the tar archive, if this frame is tar-backed.
            offset: Byte offset of the image data inside the tar archive.
            size: Size of the image data in bytes inside the tar archive.

        """
        self.path = Path(path)
        self._tar_path = Path(tar_path) if tar_path is not None else None
        self._offset = offset
        self._size = size
        self.timestamp: float = self._get_timestamp()

    def __eq__(self, other):
        """Compare whether the other is the same frame or not."""
        if not isinstance(other, self.__class__):
            return NotImplemented
        return self.path == other.path

    @property
    def size(self) -> int:
        """Size of the image in bytes."""
        if self._tar_path is not None:
            return self._size
        else:
            return self.path.stat().st_size

    def _read_bytes(self) -> bytes:
        if self._tar_path is not None:
            with open(self._tar_path, "rb") as f:
                f.seek(self._offset)
                return f.read(self._size)
        else:
            return self.path.read_bytes()

    def open_image(self) -> Image.Image:
        """Open the image of this frame as a PIL Image.

        Returns:
            PIL Image.

        """
        if self._tar_path is not None:
            return Image.open(io.BytesIO(self._read_bytes()))
        else:
            return Image.open(self.path)

    def load(self) -> np.ndarray:
        """Load image of this frame.

        Returns:
            Image array.

        """
        with self.open_image() as image:
            return np.array(image)

    def show(self):
        """Show image of this frame."""
        with self.open_image() as image:
            return image.show()

    def materialize(self, temp_dir: os.PathLike, index: int | None = None) -> Path:
        """Return a real on-disk path to this frame's JPEG.

        Directory-backed frames return their existing path without copying.
        Tar-backed frames are extracted into ``temp_dir`` and that path returned.

        Args:
            temp_dir: Directory to extract tar-backed frames into.
            index: Optional sequence number used to build a collision-free file
                name when extracting (e.g. when frames from several archives are
                materialized into the same directory).

        Returns:
            Path to a real JPEG file on disk.

        """
        if self._tar_path is not None:
            name = f"{index:08d}.jpg" if index is not None else self.path.name
            out_path = Path(temp_dir) / name
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
                sibling ``.../cameras/ceiling.tar`` archive does, the camera is
                read from the archive instead.

        """
        self.name: str = name
        self.base_path = Path(base_path)
        self.tar_path: Path | None = None
        if not self.base_path.is_dir():
            tar_path = self.base_path.with_suffix(".tar")
            if tar_path.is_file():
                self.tar_path = tar_path

        if self.tar_path is not None:
            self.all_files: list[Path] = []
            self._members: list[tuple[str, int, int]] = self._load_tar_members(
                self.tar_path
            )
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

    @property
    def num_frames(self) -> int:
        """Get number of frames."""
        if self.tar_path is not None:
            return len(self._members)
        return len(self.all_files)

    def get_frame(self, index: int) -> Frame:
        """Get frame at the index.

        Args:
            index: Index to get.

        Returns:
            Frame at the index.

        """
        if self.tar_path is not None:
            return self._tar_frame(*self._members[index])
        return Frame(self.all_files[index])

    def frames(self) -> Iterator[Frame]:
        """Iterate all frames.

        Returns:
            Iterator of Frame.

        """
        if self.tar_path is not None:
            for member in self._members:
                yield self._tar_frame(*member)
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

        Args:
            output: Destination path. For "dir" format, a directory that must
                not already exist; for "tar" format, the archive file to write.
            format: Output format, either "dir" for directory of JPEGs or "tar"
                for uncompressed tar archive.

        """
        if format == "dir":
            dest_dir = Path(output)
            if self.tar_path is None:
                shutil.copytree(self.base_path, dest_dir)
                return
            dest_dir.mkdir(parents=True)
            with tarfile.open(self.tar_path, mode="r:") as tf:
                for member in tf.getmembers():
                    if not member.isfile():
                        continue
                    src = tf.extractfile(member)
                    if src is None:
                        continue
                    (dest_dir / Path(member.name).name).write_bytes(src.read())

        elif format == "tar":
            dest_tar = Path(output)
            dest_tar.parent.mkdir(parents=True, exist_ok=True)
            if self.tar_path is not None:
                shutil.copy2(self.tar_path, dest_tar)
                return
            with tarfile.open(dest_tar, mode="w") as tf:
                for file in self.all_files:
                    tf.add(file, arcname=file.name)
        else:
            raise ValueError(f"Unsupported format: {format}")
