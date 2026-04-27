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
from openarm_dataset import Dataset
from openarm_dataset._lerobot_v21 import (
    _collect_keys_and_joint_names,
    _escape_concat_path,
    collect_downsampled_data,
    write_metadata,
    write_parquet,
    write_videos,
)

FIXTURE_DIR = Path(__file__).parent / "fixture"
DATASET_0_2_0_PATH = FIXTURE_DIR / "dataset_0.2.0"
FPS = 30
TRAIN_SPLIT = 0.8


@pytest.fixture
def lerobotv21_setup(tmp_path):
    dataset = Dataset(DATASET_0_2_0_PATH)
    dataset.set_smoothing(1.0)
    obs_keys, obs_joint_names = _collect_keys_and_joint_names(dataset, "obs")
    action_keys, action_joint_names = _collect_keys_and_joint_names(dataset, "action")
    assert obs_keys == [
        "arms/right/qpos",
        "arms/left/qpos",
    ], "Observation keys do not match expected keys."
    assert action_keys == [
        "arms/right/qpos",
        "arms/left/qpos",
    ], "Action keys do not match expected keys."

    JOINT_NAMES = obs_joint_names  # ["arms/right/qpos", "arms/left/qpos"]
    record = collect_downsampled_data(dataset, FPS, obs_keys, action_keys)
    lerobot_path = Path(tmp_path)
    return dataset, lerobot_path, record, JOINT_NAMES


def test_metadata(lerobotv21_setup):
    dataset, lerobot_path, record, JOINT_NAMES = lerobotv21_setup
    write_metadata(dataset, record, lerobot_path, FPS, TRAIN_SPLIT, JOINT_NAMES)
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
    with open(episodes_stats_jsonl_path) as f:
        episodes_stats = [json.loads(line) for line in f]
    assert len(episodes_stats) == dataset.meta.num_episodes, (
        "Number of episodes info in episodes_stats.jsonl does not match the original dataset."
    )


def test_data(lerobotv21_setup):
    dataset, lerobot_path, record, _ = lerobotv21_setup
    write_parquet(dataset, record, lerobot_path, FPS)

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


def test_video(lerobotv21_setup):
    dataset, lerobot_path, record, _ = lerobotv21_setup
    write_videos(dataset, record, lerobot_path, FPS)

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


def test_convert_dataset(lerobotv21_setup):
    dataset, lerobot_path, record, _ = lerobotv21_setup
    dataset.write(
        lerobot_path, format="lerobot_v2.1", smoothing_cutoff=1.0, train_split=0.8
    )
    # Check metadata 
    metadata_path = lerobot_path / "meta"
    assert metadata_path.exists(), "Metadata directory does not exist."
    info_json_path = metadata_path / "info.json"
    assert info_json_path.exists(), "info.json file does not exist."
    with open(info_json_path) as f:
        info = json.load(f)
    assert info["codebase_version"] == "v2.1", (
        "Incorrect codebase version in info.json."
        )
    
    # Check data files
    data_path = lerobot_path / "data" / "chunk-000" / "episode_000000.parquet"
    assert data_path.exists(), "Data file does not exist."

    # Check video files
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



def test_escape_concat_path_handles_single_quotes(tmp_path):
    frame_path = tmp_path / "single'quote.jpeg"
    frame_path.touch()

    escaped_path = _escape_concat_path(frame_path)

    assert escaped_path == str(frame_path.resolve()).replace("'", "'\\''")
