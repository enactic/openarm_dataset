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

import math
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from openarm_dataset.dataset import Dataset
from openarm_dataset.repair import repair_dataset
from openarm_dataset.repair import copy_parquet

DATASET_DIR = Path(__file__).parent / "fixture" / "dataset_0.3.0"
STATE_REL = Path("episodes") / "0" / "obs" / "arms" / "left" / "state.parquet"


def _copy_dataset(tmp_path):
    dst = tmp_path / "dataset"
    copy_parquet(DATASET_DIR, dst)
    return dst


def _inject_whole_frame_null(state_path, frame, column="qpos"):
    df = pd.read_parquet(state_path)
    values = df[column].tolist()
    values[frame] = None
    df[column] = values
    df.to_parquet(state_path)


def _inject_null_inside_list(state_path, frame, joint=0, column="qpos"):
    df = pd.read_parquet(state_path)
    values = df[column].tolist()
    item = list(values[frame])
    item[joint] = None
    values[frame] = item
    df[column] = values
    df.to_parquet(state_path)


def _inject_nan_inside_list(state_path, frame, joint=0, column="qpos"):
    df = pd.read_parquet(state_path)
    values = df[column].tolist()
    item = list(values[frame])
    item[joint] = math.nan
    values[frame] = item
    df[column] = values
    df.to_parquet(state_path)


def _qpos(state_path):
    return pd.read_parquet(state_path)["qpos"].tolist()


def test_repair_nan_inside_list_averages_neighbors(tmp_path):
    dataset = _copy_dataset(tmp_path)
    state_path = dataset / STATE_REL
    before = _qpos(state_path)
    expected = (before[4][0] + before[6][0]) / 2
    _inject_nan_inside_list(state_path, frame=5, joint=0)

    repair_dataset(dataset)

    assert Dataset(dataset).validate()
    assert _qpos(state_path)[5][0] == expected


def test_repair_null_inside_list_averages_neighbors(tmp_path):
    dataset = _copy_dataset(tmp_path)
    state_path = dataset / STATE_REL
    before = _qpos(state_path)
    expected = (before[4][2] + before[6][2]) / 2
    _inject_null_inside_list(state_path, frame=5, joint=2)

    repair_dataset(dataset)

    assert Dataset(dataset).validate()
    assert _qpos(state_path)[5][2] == expected


def test_repair_whole_frame_null_filled(tmp_path):
    dataset = _copy_dataset(tmp_path)
    state_path = dataset / STATE_REL
    before = _qpos(state_path)
    expected = [(before[4][j] + before[6][j]) / 2 for j in range(8)]
    _inject_whole_frame_null(state_path, frame=5)

    repair_dataset(dataset)

    assert Dataset(dataset).validate()
    np.testing.assert_allclose(_qpos(state_path)[5], expected)


def test_repair_consecutive_gaps_left_unrepaired(tmp_path):
    dataset = _copy_dataset(tmp_path)
    state_path = dataset / STATE_REL
    _inject_nan_inside_list(state_path, frame=5, joint=0)
    _inject_nan_inside_list(state_path, frame=6, joint=0)

    errors = []
    repair_dataset(dataset, on_error=errors.append)

    # Two consecutive gaps cannot be averaged; dataset stays invalid.
    assert not Dataset(dataset).validate()
    assert errors


def test_repair_boundary_gap_left_unrepaired(tmp_path):
    dataset = _copy_dataset(tmp_path)
    state_path = dataset / STATE_REL
    _inject_nan_inside_list(state_path, frame=0, joint=0)

    errors = []
    repair_dataset(dataset, on_error=errors.append)

    assert not Dataset(dataset).validate()
    assert errors


def test_repair_output_mode_leaves_input_untouched(tmp_path):
    dataset = _copy_dataset(tmp_path)
    state_path = dataset / STATE_REL
    _inject_nan_inside_list(state_path, frame=5, joint=0)
    output = tmp_path / "repaired"

    repair_dataset(dataset, output)

    # Input is unchanged (still invalid); output is repaired.
    assert not Dataset(dataset).validate()
    assert Dataset(output).validate()


def test_repair_preserves_elevation_dtype(tmp_path):
    dataset = _copy_dataset(tmp_path)
    elevation_path = dataset / "episodes" / "0" / "obs" / "lifter" / "elevation.parquet"
    original_dtype = pd.read_parquet(elevation_path)["value"].tolist()[0].dtype
    _inject_nan_inside_list(elevation_path, frame=5, joint=0, column="value")

    repair_dataset(dataset)

    repaired = pd.read_parquet(elevation_path)["value"].tolist()[5]
    assert repaired.dtype == original_dtype
    assert Dataset(dataset).validate()


def test_repair_cli_in_place(tmp_path):
    dataset = _copy_dataset(tmp_path)
    state_path = dataset / STATE_REL
    _inject_nan_inside_list(state_path, frame=5, joint=0)

    result = subprocess.run(
        [sys.executable, "-m", "openarm_dataset.repair", str(dataset)],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0
    assert Dataset(dataset).validate()


def test_repair_cli_output_flag(tmp_path):
    dataset = _copy_dataset(tmp_path)
    state_path = dataset / STATE_REL
    _inject_nan_inside_list(state_path, frame=5, joint=0)
    output = tmp_path / "repaired"

    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "openarm_dataset.repair",
            str(dataset),
            "-o",
            str(output),
        ],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0
    assert not Dataset(dataset).validate()
    assert Dataset(output).validate()
