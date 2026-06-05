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

from openarm_dataset import Dataset

FIXTURE_DIR = Path(__file__).parent / "fixture"
DATASET_0_3_0_PATH = FIXTURE_DIR / "dataset_0.3.0"
FPS = 30

EXPECTED_RANGES = {
    "right_arm": {"start": 0, "end": 7},
    "right_gripper": {"start": 7, "end": 8},
    "left_arm": {"start": 8, "end": 15},
    "left_gripper": {"start": 15, "end": 16},
    "lifter": {"start": 16, "end": 17},
}

EXPECTED_MODALITY = {
    "state": EXPECTED_RANGES,
    "action": EXPECTED_RANGES,
    "video": {
        "wrist_left": {"original_key": "observation.images.wrist_left"},
        "wrist_right": {"original_key": "observation.images.wrist_right"},
        "ceiling": {"original_key": "observation.images.ceiling"},
        "head": {"original_key": "observation.images.head"},
    },
    "annotation": {
        "human.action.task_description": {"original_key": "task_index"},
    },
}


def test_gr00t_write(tmp_path):
    dataset = Dataset(DATASET_0_3_0_PATH)
    dataset.set_smoothing(1.0)
    dataset.write(
        tmp_path, format="gr00t", fps=FPS, train_split=0.8, success_only=False
    )

    # LeRobot v2.1 outputs are still produced
    assert (tmp_path / "meta" / "info.json").exists()
    assert (tmp_path / "data" / "chunk-000" / "episode_000000.parquet").exists()
    for camera_name in dataset.camera_names:
        video_path = (
            tmp_path
            / "videos"
            / "chunk-000"
            / f"observation.images.{camera_name}"
            / "episode_000000.mp4"
        )
        assert video_path.exists()

    # plus the GR00T modality file
    with open(tmp_path / "meta" / "modality.json") as f:
        modality = json.load(f)
    assert modality == EXPECTED_MODALITY

    # test modality ranges are contiguous
    for block in ("state", "action"):
        assert len(modality[block]) == len(EXPECTED_RANGES)
        ranges = list(modality[block].values())
        assert ranges[0]["start"] == 0
        for prev, cur in zip(ranges, ranges[1:]):
            assert cur["start"] == prev["end"]
        assert ranges[-1]["end"] == 17  # total number of joints in the dataset
