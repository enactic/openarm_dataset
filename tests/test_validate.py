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

import copy
import math
import shutil
import subprocess
import sys
from pathlib import Path

import pandas as pd
import yaml

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


def _inject_nan_in_qpos_list(state_path):
    """Replace a float value inside qpos with NaN (not null)."""
    df = pd.read_parquet(state_path)
    values = df["qpos"].tolist()
    first = list(values[0])
    first[0] = math.nan
    values[0] = first
    df["qpos"] = values
    df.to_parquet(state_path)


def test_validate_detects_nan_in_qpos(tmp_path):
    shutil.copytree(DATASET_DIR, tmp_path, dirs_exist_ok=True)
    state_path = tmp_path / "episodes" / "0" / "obs" / "arms" / "left" / "state.parquet"
    _inject_nan_in_qpos_list(state_path)

    errors = []
    assert not Dataset(tmp_path).validate(on_error=errors.append)
    assert errors == ["episodes/0/obs/arms/left/state.parquet: includes null values"]


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


POSE_DATASET_DIR = Path(__file__).parent / "fixture" / "dataset_0.4.0_pose"

POSE_ATTRIBUTES = {
    "action": {"arms": {"left": ["pose"], "right": ["pose"]}},
    "obs": {
        "arms": {
            "left": ["qpos", "qvel", "qtorque"],
            "right": ["qpos", "qvel", "qtorque"],
        }
    },
}


def _write_attributes(dataset_path, attributes):
    meta_path = dataset_path / "metadata.yaml"
    with open(meta_path) as f:
        meta = yaml.safe_load(f)
    meta["attributes"] = attributes
    with open(meta_path, "w") as f:
        yaml.safe_dump(meta, f)


def test_validate_attributes_match():
    # The committed fixture declares attributes matching its parquet files.
    errors = []
    assert Dataset(POSE_DATASET_DIR).validate(on_error=errors.append)
    assert errors == []


def test_validate_attributes_mismatch(tmp_path):
    shutil.copytree(POSE_DATASET_DIR, tmp_path, dirs_exist_ok=True)
    attributes = copy.deepcopy(POSE_ATTRIBUTES)
    attributes["action"]["arms"]["left"] = ["qpos"]
    attributes["action"]["arms"]["right"] = ["qpos"]
    _write_attributes(tmp_path, attributes)

    errors = []
    assert not Dataset(tmp_path).validate(on_error=errors.append)
    assert errors == [
        "attributes mismatch for action arms/left: "
        "metadata declares ['qpos'], episodes contain ['pose']",
        "attributes mismatch for action arms/right: "
        "metadata declares ['qpos'], episodes contain ['pose']",
    ]


def test_validate_attributes_order_insensitive(tmp_path):
    shutil.copytree(POSE_DATASET_DIR, tmp_path, dirs_exist_ok=True)
    attributes = copy.deepcopy(POSE_ATTRIBUTES)
    attributes["obs"]["arms"]["left"] = ["qtorque", "qpos", "qvel"]
    _write_attributes(tmp_path, attributes)

    errors = []
    assert Dataset(tmp_path).validate(on_error=errors.append)
    assert errors == []


def test_validate_without_attributes_key_skips_check(tmp_path):
    shutil.copytree(POSE_DATASET_DIR, tmp_path, dirs_exist_ok=True)
    meta_path = tmp_path / "metadata.yaml"
    with open(meta_path) as f:
        meta = yaml.safe_load(f)
    del meta["attributes"]
    with open(meta_path, "w") as f:
        yaml.safe_dump(meta, f)

    errors = []
    assert Dataset(tmp_path).validate(on_error=errors.append)
    assert errors == []
