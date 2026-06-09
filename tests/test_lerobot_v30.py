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

pytest.importorskip("lerobot")

from pathlib import Path
import json
import shutil
import subprocess
import numpy as np
import packaging.version
import pandas as pd
from PIL import Image
from openarm_dataset import Dataset
from openarm_dataset.lerobot_v21 import _sample_image_indices
import lerobot
from lerobot.datasets.lerobot_dataset import LeRobotDataset

FIXTURE_DIR = Path(__file__).parent / "fixture"

_lerobot_v3 = packaging.version.parse(lerobot.__version__) >= packaging.version.parse(
    "0.5.0"
)
_skip_v30_load = pytest.mark.skipif(
    not _lerobot_v3,
    reason="lerobot < 0.5.0 does not support v3.0 datasets",
)
DATASET_0_3_0_PATH = FIXTURE_DIR / "dataset_0.3.0"
FPS = 30
MEAN_ATOL = 5e-3
STD_ATOL = 1e-2
EXTREMA_SLACK = 1e-2


@pytest.fixture
def lerobot_v30_setup(tmp_path):
    dataset = Dataset(DATASET_0_3_0_PATH)
    dataset.set_smoothing(1.0)
    dataset.write(
        tmp_path,
        format="lerobot_v3.0",
        fps=FPS,
        train_split=0.8,
        success_only=False,
    )
    return dataset, tmp_path


def _numpy_image_stats(paths: list[Path]) -> dict:
    arrays = []
    for p in paths:
        with Image.open(p) as img:
            arrays.append(
                np.asarray(img.convert("RGB"), dtype=np.float64).reshape(-1, 3)
            )
    pixels = np.concatenate(arrays, axis=0)
    return {
        "min": pixels.min(axis=0) / 255.0,
        "max": pixels.max(axis=0) / 255.0,
        "mean": pixels.mean(axis=0) / 255.0,
        "std": pixels.std(axis=0, ddof=0) / 255.0,
        "count": len(paths),
    }


def _flatten_channels(stats_field) -> np.ndarray:
    return np.array([c[0][0] for c in stats_field], dtype=np.float64)


def test_info_json(lerobot_v30_setup):
    _, lerobot_path = lerobot_v30_setup
    info_path = lerobot_path / "meta" / "info.json"
    assert info_path.exists()
    with open(info_path) as f:
        info = json.load(f)
    assert info["codebase_version"] == "v3.0"
    assert info["fps"] == FPS
    assert "data_files_size_in_mb" in info
    assert "video_files_size_in_mb" in info
    assert "total_chunks" not in info
    assert "total_videos" not in info
    assert (
        info["data_path"]
        == "data/chunk-{chunk_index:03d}/file-{file_index:03d}.parquet"
    )
    assert (
        info["video_path"]
        == "videos/{video_key}/chunk-{chunk_index:03d}/file-{file_index:03d}.mp4"
    )
    for key, feat in info["features"].items():
        if feat["dtype"] == "video":
            assert "info" in feat
            assert feat["info"]["video.fps"] == FPS
        else:
            assert feat.get("fps") == FPS, f"feature {key} missing fps"


def test_tasks_parquet(lerobot_v30_setup):
    dataset, lerobot_path = lerobot_v30_setup
    tasks_path = lerobot_path / "meta" / "tasks.parquet"
    assert tasks_path.exists()
    df = pd.read_parquet(tasks_path)
    assert len(df) == len(dataset.meta.tasks)
    assert df.index.name == "task"
    assert "task_index" in df.columns


def test_episodes_parquet(lerobot_v30_setup):
    dataset, lerobot_path = lerobot_v30_setup
    episodes_path = (
        lerobot_path / "meta" / "episodes" / "chunk-000" / "file-000.parquet"
    )
    assert episodes_path.exists()
    df = pd.read_parquet(episodes_path)
    assert len(df) == dataset.meta.num_episodes

    required_cols = [
        "episode_index",
        "tasks",
        "length",
        "data/chunk_index",
        "data/file_index",
        "dataset_from_index",
        "dataset_to_index",
        "meta/episodes/chunk_index",
        "meta/episodes/file_index",
    ]
    for col in required_cols:
        assert col in df.columns, f"Missing column: {col}"

    for cam in dataset.camera_names:
        image_name = f"observation.images.{cam}"
        for suffix in [
            "chunk_index",
            "file_index",
            "from_timestamp",
            "to_timestamp",
        ]:
            col = f"videos/{image_name}/{suffix}"
            assert col in df.columns, f"Missing column: {col}"

    stats_cols = [c for c in df.columns if c.startswith("stats/")]
    assert len(stats_cols) > 0, "No stats columns in episodes parquet"


def test_stats_json(lerobot_v30_setup):
    _, lerobot_path = lerobot_v30_setup
    stats_path = lerobot_path / "meta" / "stats.json"
    assert stats_path.exists()
    with open(stats_path) as f:
        stats = json.load(f)
    assert "action" in stats
    assert "observation.state" in stats
    for key in ("min", "max", "mean", "std", "count"):
        assert key in stats["action"]
        assert key in stats["observation.state"]


def test_packed_data(lerobot_v30_setup):
    dataset, lerobot_path = lerobot_v30_setup
    data_path = lerobot_path / "data" / "chunk-000" / "file-000.parquet"
    assert data_path.exists()

    df = pd.read_parquet(data_path)

    episode_indices = df["episode_index"].unique()
    assert len(episode_indices) == dataset.meta.num_episodes

    sample_episode = dataset.sample(FPS, episode_index=0)
    sample_0_action = np.concatenate(
        [
            sample_episode[0].action["arms/right/qpos"],
            sample_episode[0].action["arms/left/qpos"],
            sample_episode[0].action["lifter/elevation"],
        ]
    )
    lerobot_action = df[df["episode_index"] == 0]["action"].iloc[0]

    assert all(
        abs(lerobot_action[i] - sample_0_action[i]) < 1e-6
        for i in range(len(sample_0_action))
    )

    sample_0_obs = np.concatenate(
        [
            sample_episode[0].obs["arms/right/qpos"],
            sample_episode[0].obs["arms/left/qpos"],
            sample_episode[0].obs["lifter/elevation"],
        ]
    )
    lerobot_obs = df[df["episode_index"] == 0]["observation.state"].iloc[0]

    assert all(
        abs(lerobot_obs[i] - sample_0_obs[i]) < 1e-6 for i in range(len(sample_0_obs))
    )


def test_packed_video(lerobot_v30_setup):
    dataset, lerobot_path = lerobot_v30_setup
    for cam in dataset.camera_names:
        video_path = (
            lerobot_path
            / "videos"
            / f"observation.images.{cam}"
            / "chunk-000"
            / "file-000.mp4"
        )
        assert video_path.exists(), f"Video file for camera {cam} does not exist."


def _ffprobe_pts_ticks(video_path: Path) -> tuple[list[int], int]:
    """Return (sorted PTS in stream time-base ticks, time-base denominator)."""
    tb = subprocess.run(
        [
            "ffprobe",
            "-v",
            "error",
            "-select_streams",
            "v:0",
            "-show_entries",
            "stream=time_base",
            "-of",
            "default=nk=1:nw=1",
            str(video_path),
        ],
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip()
    _, den = tb.split("/")
    out = subprocess.run(
        [
            "ffprobe",
            "-v",
            "error",
            "-select_streams",
            "v:0",
            "-show_entries",
            "frame=pts",
            "-of",
            "csv=p=0",
            str(video_path),
        ],
        capture_output=True,
        text=True,
        check=True,
    ).stdout.split()
    pts = sorted(int(x.rstrip(",")) for x in out if x.strip())
    return pts, int(den)


def test_packed_video_uniform_pts(lerobot_v30_setup):
    """Packed videos must have a strictly uniform i/fps PTS grid.

    Episodes are concatenated into a single video file. If frames drift off the
    exact 1/fps grid (e.g. from per-segment rounding when muxing separately
    encoded clips), LeRobot's ``seek_mode="approximate"`` decoder fetches an
    off-by-one frame and raises ``FrameTimestampError``. A single-pass encode
    keeps every gap at exactly one frame.
    """
    if shutil.which("ffprobe") is None:
        pytest.skip("ffprobe not available")
    dataset, lerobot_path = lerobot_v30_setup
    for cam in dataset.camera_names:
        video_path = (
            lerobot_path
            / "videos"
            / f"observation.images.{cam}"
            / "chunk-000"
            / "file-000.mp4"
        )
        pts, den = _ffprobe_pts_ticks(video_path)
        assert den % FPS == 0, f"{cam}: time-base {den} not divisible by fps {FPS}"
        expected_gap = den // FPS
        gaps = {pts[i + 1] - pts[i] for i in range(len(pts) - 1)}
        assert gaps <= {expected_gap}, (
            f"{cam}: non-uniform PTS gaps {sorted(gaps)} "
            f"(expected only {expected_gap} ticks = 1/{FPS}s)"
        )


def test_video_timestamps_exact(lerobot_v30_setup):
    """from_timestamp/to_timestamp are exact frame-count multiples of 1/fps."""
    dataset, lerobot_path = lerobot_v30_setup
    df = pd.read_parquet(
        lerobot_path / "meta" / "episodes" / "chunk-000" / "file-000.parquet"
    ).sort_values("episode_index")
    for cam in dataset.camera_names:
        image_name = f"observation.images.{cam}"
        from_col = f"videos/{image_name}/from_timestamp"
        to_col = f"videos/{image_name}/to_timestamp"
        for _, row in df.iterrows():
            span = (row[to_col] - row[from_col]) * FPS
            np.testing.assert_allclose(span, row["length"], atol=1e-6)
            # offset must land exactly on the frame grid (no ffprobe-duration drift)
            np.testing.assert_allclose(
                round(row[from_col] * FPS), row[from_col] * FPS, atol=1e-6
            )


@_skip_v30_load
def test_load(lerobot_v30_setup):
    dataset, lerobot_path = lerobot_v30_setup
    lerobot_dataset = LeRobotDataset(repo_id="test/data", root=lerobot_path)

    assert lerobot_dataset.num_episodes == dataset.meta.num_episodes
    assert len(lerobot_dataset.meta.tasks) == len(dataset.meta.tasks)


def test_lifter_info_features(lerobot_v30_setup):
    _, lerobot_path = lerobot_v30_setup
    with open(lerobot_path / "meta" / "info.json") as f:
        info = json.load(f)
    assert info["features"]["action"]["shape"] == [17]
    assert info["features"]["observation.state"]["shape"] == [17]
    assert info["features"]["action"]["names"][-1] == "elevation.pos"
    assert info["features"]["observation.state"]["names"][-1] == "elevation.pos"


@_skip_v30_load
def test_success_only(tmp_path):
    dataset = Dataset(DATASET_0_3_0_PATH)
    dataset.set_smoothing(1.0)
    lerobot_path = tmp_path / "success_only" / "lerobot_v3.0_success_only"
    dataset.write(
        lerobot_path,
        format="lerobot_v3.0",
        fps=FPS,
        train_split=0.8,
        success_only=True,
    )

    episodes_path = (
        lerobot_path / "meta" / "episodes" / "chunk-000" / "file-000.parquet"
    )
    assert episodes_path.exists()
    df_episodes = pd.read_parquet(episodes_path)
    assert len(df_episodes) == 1

    data_path = lerobot_path / "data" / "chunk-000" / "file-000.parquet"
    assert data_path.exists()
    df_data = pd.read_parquet(data_path)
    assert df_data["episode_index"].nunique() == 1

    sample_episode = dataset.sample(FPS, episode_index=1)
    sample_0_action = np.concatenate(
        [
            sample_episode[0].action["arms/right/qpos"],
            sample_episode[0].action["arms/left/qpos"],
            sample_episode[0].action["lifter/elevation"],
        ]
    )
    lerobot_action = df_data["action"].iloc[0]

    assert all(
        abs(lerobot_action[i] - sample_0_action[i]) < 1e-6
        for i in range(len(sample_0_action))
    )

    lerobot_dataset = LeRobotDataset(repo_id="test/data", root=lerobot_path)
    assert lerobot_dataset.num_episodes == 1


def test_episode_image_stats(lerobot_v30_setup):
    dataset, lerobot_path = lerobot_v30_setup
    episodes_path = (
        lerobot_path / "meta" / "episodes" / "chunk-000" / "file-000.parquet"
    )
    df = pd.read_parquet(episodes_path)

    for row_idx in range(len(df)):
        ep_index = int(df.iloc[row_idx]["episode_index"])
        samples = dataset.sample(hz=FPS, episode_index=ep_index)

        for cam in dataset.camera_names:
            image_name = f"observation.images.{cam}"
            paths = [Path(s.cameras[cam].path) for s in samples]
            expected = _numpy_image_stats(paths)

            mean_col = f"stats/{image_name}/mean"
            std_col = f"stats/{image_name}/std"
            min_col = f"stats/{image_name}/min"
            max_col = f"stats/{image_name}/max"
            count_col = f"stats/{image_name}/count"

            saved_mean = _flatten_channels(df.iloc[row_idx][mean_col])
            saved_std = _flatten_channels(df.iloc[row_idx][std_col])
            saved_min = _flatten_channels(df.iloc[row_idx][min_col])
            saved_max = _flatten_channels(df.iloc[row_idx][max_col])

            np.testing.assert_allclose(
                saved_mean,
                expected["mean"],
                atol=MEAN_ATOL,
                err_msg=f"episode {ep_index}, camera {cam}: mean differs",
            )
            np.testing.assert_allclose(
                saved_std,
                expected["std"],
                atol=STD_ATOL,
                err_msg=f"episode {ep_index}, camera {cam}: std differs",
            )
            assert (saved_min >= expected["min"] - 1e-9).all()
            assert (saved_min <= expected["min"] + EXTREMA_SLACK).all()
            assert (saved_max <= expected["max"] + 1e-9).all()
            assert (saved_max >= expected["max"] - EXTREMA_SLACK).all()

            saved_count = df.iloc[row_idx][count_col]
            assert saved_count[0] == len(paths)
