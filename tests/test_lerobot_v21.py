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

from pathlib import Path
import json
import numpy as np
import pandas as pd
import pytest
from PIL import Image
from openarm_dataset import Dataset
from openarm_dataset.lerobot_v21 import _sample_image_indices
from lerobot.datasets.lerobot_dataset import LeRobotDataset

FIXTURE_DIR = Path(__file__).parent / "fixture"
DATASET_0_3_0_PATH = FIXTURE_DIR / "dataset_0.3.0"
FPS = 30
# Image stats are subsampled (lerobot-style), so tolerances are loose vs full-pixel truth.
MEAN_ATOL = 5e-3
STD_ATOL = 1e-2
EXTREMA_SLACK = 1e-2


@pytest.fixture
def lerobot_v21_setup(tmp_path):
    dataset = Dataset(DATASET_0_3_0_PATH)
    dataset.set_smoothing(1.0)
    dataset.write(
        tmp_path,
        format="lerobot_v2.1",
        fps=FPS,
        train_split=0.8,
        success_only=False,
    )
    return dataset, tmp_path


def _numpy_image_stats(paths: list[Path]) -> dict:
    """Reference per-channel stats over RGB pixels, normalized to [0, 1].

    Population std (ddof=0) matches `_describe_images`'s `sumsq/N - mean^2`.
    """
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
        "count": len(_sample_image_indices(len(paths))),
    }


def _flatten_channels(stats_field) -> np.ndarray:
    """Convert `[[[r]], [[g]], [[b]]]` into shape-(3,) array."""
    return np.array([c[0][0] for c in stats_field], dtype=np.float64)


def test_metadata(lerobot_v21_setup):
    dataset, lerobot_path = lerobot_v21_setup
    metadata_path = lerobot_path / "meta"
    ## check info.json
    info_json_path = metadata_path / "info.json"
    assert info_json_path.exists(), "info.json file does not exist."
    with open(info_json_path) as f:
        info = json.load(f)
    assert info["codebase_version"] == "v2.1", (
        "Incorrect codebase version in info.json."
    )
    ## check tasks.jsonl
    tasks_jsonl_path = metadata_path / "tasks.jsonl"
    assert tasks_jsonl_path.exists(), "tasks.jsonl file does not exist."
    with open(tasks_jsonl_path) as f:
        tasks = [json.loads(line) for line in f]
    assert len(tasks) == len(dataset.meta.tasks), (
        "Number of tasks in tasks.jsonl does not match the original dataset."
    )
    ## episodes.jsonl
    episodes_jsonl_path = metadata_path / "episodes.jsonl"
    assert episodes_jsonl_path.exists(), "episodes.jsonl file does not exist."
    with open(episodes_jsonl_path) as f:
        episodes = [json.loads(line) for line in f]
    assert len(episodes) == dataset.meta.num_episodes, (
        "Number of episodes in episodes.jsonl does not match the original dataset."
    )
    ## episodes_stats.jsonl
    episodes_stats_jsonl_path = metadata_path / "episodes_stats.jsonl"
    assert episodes_stats_jsonl_path.exists(), (
        "episodes_stats.jsonl file does not exist."
    )

    with episodes_stats_jsonl_path.open() as f:
        episodes_stats = [json.loads(line) for line in f]

    for ep in episodes_stats:
        episode_index = ep["episode_index"]
        samples = dataset.sample(hz=FPS, episode_index=episode_index)

        for cam in dataset.camera_names:
            key = f"observation.images.{cam}"
            assert key in ep["stats"], (
                f"episode {episode_index}: missing {key} in episodes_stats"
            )

            paths = [Path(s.cameras[cam].path) for s in samples]
            saved = ep["stats"][key]
            expected = _numpy_image_stats(paths)

            saved_min = _flatten_channels(saved["min"])
            saved_max = _flatten_channels(saved["max"])
            saved_mean = _flatten_channels(saved["mean"])
            saved_std = _flatten_channels(saved["std"])

            np.testing.assert_allclose(
                saved_mean,
                expected["mean"],
                atol=MEAN_ATOL,
                err_msg=f"episode {episode_index}, camera {cam}: mean differs",
            )
            np.testing.assert_allclose(
                saved_std,
                expected["std"],
                atol=STD_ATOL,
                err_msg=f"episode {episode_index}, camera {cam}: std differs",
            )
            # Subsampled min can only be ≥ true min; max can only be ≤ true max.
            assert (saved_min >= expected["min"] - 1e-9).all(), (
                f"episode {episode_index}, camera {cam}: subsampled min < true min"
            )
            assert (saved_min <= expected["min"] + EXTREMA_SLACK).all(), (
                f"episode {episode_index}, camera {cam}: subsampled min too far above true min"
            )
            assert (saved_max <= expected["max"] + 1e-9).all(), (
                f"episode {episode_index}, camera {cam}: subsampled max > true max"
            )
            assert (saved_max >= expected["max"] - EXTREMA_SLACK).all(), (
                f"episode {episode_index}, camera {cam}: subsampled max too far below true max"
            )
            assert saved["count"] == [len(_sample_image_indices(len(paths)))], (
                f"episode {episode_index}, camera {cam}: count mismatch"
            )
    assert len(episodes_stats) == dataset.meta.num_episodes, (
        "Number of episodes info in episodes_stats.jsonl does not match the original dataset."
    )


def test_data(lerobot_v21_setup):
    dataset, lerobot_path = lerobot_v21_setup
    data_path = lerobot_path / "data" / "chunk-000" / "episode_000000.parquet"
    assert data_path.exists(), "Data file does not exist."

    df = pd.read_parquet(data_path)

    sample_episode = dataset.sample(30, episode_index=0)
    sample_episode_0_action = sample_episode[
        0
    ].action  # {"arms/right/qpos": array([0.1, 0.2, 0.3]), "arms/left/qpos": array([0.4, 0.5, 0.6])}
    sample_0_action = np.concatenate(
        [
            sample_episode_0_action["arms/right/qpos"],
            sample_episode_0_action["arms/left/qpos"],
        ]
    )
    lerobot_action = df["action"].iloc[0]

    assert all(
        abs(lerobot_action[i] - sample_0_action[i]) < 1e-6
        for i in range(len(sample_0_action))
    ), "Action values in data file do not match the original dataset."

    sample_observation = sample_episode[0].obs
    sample_0_observation = np.concatenate(
        [sample_observation["arms/right/qpos"], sample_observation["arms/left/qpos"]]
    )
    lerobot_observation = df["observation.state"].iloc[0]

    assert all(
        abs(lerobot_observation[i] - sample_0_observation[i]) < 1e-6
        for i in range(len(sample_0_observation))
    ), "Observation values in data file do not match the original dataset."


def test_video(lerobot_v21_setup):
    dataset, lerobot_path = lerobot_v21_setup
    camera_names = dataset.camera_names
    for camera_name in camera_names:
        video_path = (
            lerobot_path
            / "videos"
            / "chunk-000"
            / f"observation.images.{camera_name}"
            / "episode_000000.mp4"
        )
        assert video_path.exists(), (
            f"Video file for camera {camera_name} does not exist."
        )


def test_load(lerobot_v21_setup):
    dataset, lerobot_path = lerobot_v21_setup
    lerobot_dataset = LeRobotDataset(repo_id="test/data", root=lerobot_path)

    # check num episodes
    assert lerobot_dataset.num_episodes == dataset.meta.num_episodes, (
        "Number of episodes in LeRobotDataset does not match the original dataset."
    )

    # check tasks
    assert len(lerobot_dataset.meta.tasks) == len(dataset.meta.tasks), (
        "Number of tasks in LeRobotDataset does not match the original dataset."
    )


def test_success_only(tmp_path):
    dataset = Dataset(DATASET_0_3_0_PATH)
    dataset.set_smoothing(1.0)
    lerobot_path = tmp_path / "success_only" / "lerobot_v2.1_success_only"
    dataset.write(
        lerobot_path, format="lerobot_v2.1", fps=FPS, train_split=0.8, success_only=True
    )

    # check metadata
    with open(lerobot_path / "meta" / "episodes_stats.jsonl") as f:
        episodes_stats = [json.loads(line) for line in f]
    assert len(episodes_stats) == 1  # only id:3 is success

    with open(lerobot_path / "meta" / "episodes.jsonl") as f:
        episodes = [json.loads(line) for line in f]
    assert len(episodes) == 1  # only id:3 is success

    # check data
    data_path = lerobot_path / "data" / "chunk-000" / "episode_000000.parquet"
    assert data_path.exists(), "Data file does not exist."

    df = pd.read_parquet(data_path)

    sample_episode = dataset.sample(
        30, episode_index=1
    )  # episode_index 1 (id:3) is the only success episode
    sample_episode_0_action = sample_episode[0].action
    sample_0_action = np.concatenate(
        [
            sample_episode_0_action["arms/right/qpos"],
            sample_episode_0_action["arms/left/qpos"],
        ]
    )
    lerobot_action = df["action"].iloc[0]

    assert all(
        abs(lerobot_action[i] - sample_0_action[i]) < 1e-6
        for i in range(len(sample_0_action))
    ), "Action values in data file do not match the original dataset."

    ##load with LeRobotDataset and check num episodes
    lerobot_dataset = LeRobotDataset(repo_id="test/data", root=lerobot_path)
    assert lerobot_dataset.num_episodes == 1, (
        "Number of episodes in LeRobotDataset does not match the expected number of success episodes."
    )
