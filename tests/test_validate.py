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

import shutil
import subprocess
import sys
from pathlib import Path

import pandas as pd

from openarm_dataset.dataset import Dataset

DATASET_DIR = Path(__file__).parent / "fixture" / "dataset_0.3.0"


def _inject_null_qpos(state_path):
    df = pd.read_parquet(state_path)
    values = df["qpos"].tolist()
    values[0] = None
    df["qpos"] = values
    df.to_parquet(state_path)


def _inject_null_inside_qpos_list(state_path):
    df = pd.read_parquet(state_path)
    values = df["qpos"].tolist()
    first = list(values[0])
    first[0] = None
    values[0] = first
    df["qpos"] = values
    df.to_parquet(state_path)


def test_validate_valid_dataset():
    errors = []
    assert Dataset(DATASET_DIR).validate(on_error=errors.append)
    assert errors == []


def test_validate_invalid_dataset_with_null_qpos(tmp_path):
    shutil.copytree(DATASET_DIR, tmp_path, dirs_exist_ok=True)
    state_path = tmp_path / "episodes" / "0" / "obs" / "arms" / "left" / "state.parquet"
    _inject_null_qpos(state_path)

    errors = []
    assert not Dataset(tmp_path).validate(on_error=errors.append)
    assert errors == ["episodes/0/obs/arms/left/state.parquet: includes null values"]


def test_validate_invalid_dataset_with_null_inside_qpos_list(tmp_path):
    shutil.copytree(DATASET_DIR, tmp_path, dirs_exist_ok=True)
    state_path = tmp_path / "episodes" / "0" / "obs" / "arms" / "left" / "state.parquet"
    _inject_null_inside_qpos_list(state_path)

    errors = []
    assert not Dataset(tmp_path).validate(on_error=errors.append)
    assert errors == ["episodes/0/obs/arms/left/state.parquet: includes null values"]


def test_validate_multiple_invalid_qpos(tmp_path):
    shutil.copytree(DATASET_DIR, tmp_path, dirs_exist_ok=True)
    for side in ("left", "right"):
        _inject_null_qpos(
            tmp_path / "episodes" / "0" / "obs" / "arms" / side / "state.parquet"
        )

    errors = []
    assert not Dataset(tmp_path).validate(on_error=errors.append)
    assert errors == [
        "episodes/0/obs/arms/right/state.parquet: includes null values",
        "episodes/0/obs/arms/left/state.parquet: includes null values",
    ]


def test_validate_multiple_invalid_qpos_with_null_inside_list(tmp_path):
    shutil.copytree(DATASET_DIR, tmp_path, dirs_exist_ok=True)
    for side in ("left", "right"):
        _inject_null_inside_qpos_list(
            tmp_path / "episodes" / "0" / "obs" / "arms" / side / "state.parquet"
        )

    errors = []
    assert not Dataset(tmp_path).validate(on_error=errors.append)
    assert errors == [
        "episodes/0/obs/arms/right/state.parquet: includes null values",
        "episodes/0/obs/arms/left/state.parquet: includes null values",
    ]


def test_validate_cli_valid():
    result = subprocess.run(
        [sys.executable, "-m", "openarm_dataset.validate", str(DATASET_DIR)],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0


def test_validate_cli_invalid(tmp_path):
    shutil.copytree(DATASET_DIR, tmp_path, dirs_exist_ok=True)
    state_path = tmp_path / "episodes" / "0" / "obs" / "arms" / "left" / "state.parquet"
    _inject_null_qpos(state_path)

    result = subprocess.run(
        [sys.executable, "-m", "openarm_dataset.validate", str(tmp_path)],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 1
    assert result.stderr == (
        "episodes/0/obs/arms/left/state.parquet: includes null values\n"
    )
