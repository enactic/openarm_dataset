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

"""FFmpeg helpers for encoding image frames into a video file."""

import shutil
import subprocess
import tempfile
from pathlib import Path

# config for video encoding
FFMPEG_CODEC = "libx264"
VIDEO_PIX_FMT = "yuv420p"


def _get_ffmpeg_exe() -> str | None:
    """Get the path to a valid ffmpeg executable."""
    # check if ffmpeg is available in the current environment
    exe = shutil.which("ffmpeg")
    if exe and _is_valid_exe(exe):
        return exe
    return None


def _is_valid_exe(exe: str) -> bool:
    """Check if the given executable is a valid ffmpeg."""
    startupinfo = None

    # On Windows, hide the console window when running ffmpeg
    if hasattr(subprocess, "STARTUPINFO"):
        startupinfo = subprocess.STARTUPINFO()
        startupinfo.dwFlags |= subprocess.STARTF_USESHOWWINDOW

    try:
        subprocess.check_call(
            [exe, "-version"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.STDOUT,
            startupinfo=startupinfo,
        )
        return True
    except (OSError, ValueError, subprocess.CalledProcessError):
        return False


def _escape_concat_path(path: Path) -> str:
    return str(path.resolve()).replace("'", "'\\''")


def encode_mp4(frames: list[Path], fps: int, out_mp4: Path, verbose=True):
    """Encode the given image frames into an MP4 file using FFmpeg."""
    if not frames:
        return
    try:
        ffmpeg_exe = _get_ffmpeg_exe()
        if ffmpeg_exe is None:
            raise RuntimeError("FFmpeg executable not found.")
    except RuntimeError as e:
        raise RuntimeError(
            "FFmpeg is required for video encoding but was not found. Please install FFmpeg in your conda environment or ensure it is available in your system PATH."
        ) from e
    with tempfile.TemporaryDirectory() as temp_dir:
        list_path = Path(temp_dir) / "ffmpeg_concat.txt"
        with list_path.open("w") as f_list:
            for f_path in frames:
                f_list.write(f"file '{_escape_concat_path(f_path)}'\n")

        cmd = [
            ffmpeg_exe,  # use the detected ffmpeg executable path
            "-y",
            "-nostdin",
            "-loglevel",
            "warning",
            "-stats",
            "-f",
            "concat",
            "-safe",
            "0",
            "-r",
            str(fps),
            "-i",
            str(list_path),
            "-c:v",
            FFMPEG_CODEC,
            "-preset",
            "veryfast",
            "-pix_fmt",
            VIDEO_PIX_FMT,
            str(out_mp4),
        ]
        subprocess.run(cmd, check=True, capture_output=not verbose)
