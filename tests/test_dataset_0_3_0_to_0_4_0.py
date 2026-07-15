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

import pandas as pd
import pytest
import yaml

from openarm_dataset.dataset import Dataset

DATASET_DIR = Path(__file__).parent / "fixture" / "dataset_0.3.0"

ARM_OBS_KEYS = {
    "arms/left/qpos",
    "arms/left/qvel",
    "arms/left/qtorque",
    "arms/right/qpos",
    "arms/right/qvel",
    "arms/right/qtorque",
}

ARM_ACTION_KEYS = {
    "arms/left/qpos",
    "arms/right/qpos",
}


@pytest.fixture
def dataset():
    return Dataset(DATASET_DIR)


def test_write_converts_to_0_4_0(dataset, tmp_path):
    original_action = dataset.load_action(dataset.meta.episodes[0])
    original_row = original_action["arms/left/qpos"].iloc[0].to_numpy()

    output = tmp_path / "out"
    dataset.write(output)
    meta = yaml.safe_load((output / "metadata.yaml").read_text())
    assert meta["version"] == "0.4.0"
    for episode_id in ("0", "3"):
        for type_ in ("obs", "action"):
            for side in ("left", "right"):
                arm_dir = output / "episodes" / episode_id / type_ / "arms" / side
                assert (arm_dir / "state.parquet").exists()
                assert not (arm_dir / "qpos.parquet").exists()
        action_df = pd.read_parquet(
            output
            / "episodes"
            / episode_id
            / "action"
            / "arms"
            / "left"
            / "state.parquet"
        )
        assert list(action_df.columns) == ["timestamp", "qpos"]
        assert (
            output / "episodes" / episode_id / "action" / "lifter" / "elevation.parquet"
        ).exists()
    rewritten = Dataset(output)
    obs = rewritten.load_obs(rewritten.meta.episodes[0])
    assert set(obs) == ARM_OBS_KEYS | {"lifter/elevation"}
    action = rewritten.load_action(rewritten.meta.episodes[0])
    assert set(action) == ARM_ACTION_KEYS | {"lifter/elevation"}
    assert action["arms/left/qpos"].iloc[0].to_numpy() == pytest.approx(original_row)
