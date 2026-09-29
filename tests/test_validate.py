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
import shutil
import subprocess
import sys
from pathlib import Path
import pandas as pd
from openarm_dataset.dataset import Dataset

DATASET_DIR = Path(__file__).parent / "fixture" / "dataset_0.4.0_qpos"

# The CLI enables the qpos and duration thresholds by default. Tests that
# only exercise the null check turn them off.
DISABLE_THRESHOLDS = [
    "--qpos-jump-threshold",
    "none",
    "--qpos-absmax",
    "none",
    "--min-duration",
    "none",
]


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


def test_validate_update_metadata(tmp_path):
    shutil.copytree(DATASET_DIR, tmp_path, dirs_exist_ok=True)
    state_path = tmp_path / "episodes" / "0" / "obs" / "arms" / "left" / "state.parquet"
    _inject_null_qpos(state_path)

    assert not Dataset(tmp_path).validate(update_metadata=True)
    assert Dataset(tmp_path).meta.episodes == [
        {"id": "0", "success": False, "task_index": 0, "valid": False},
        {"id": "3", "success": True, "task_index": 0, "valid": True},
    ]


def test_validate_cli_valid(tmp_path):
    shutil.copytree(DATASET_DIR, tmp_path, dirs_exist_ok=True)
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "openarm_dataset.validate",
            str(tmp_path),
            *DISABLE_THRESHOLDS,
        ],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0
    assert Dataset(tmp_path).meta.episodes == [
        {"id": "0", "success": False, "task_index": 0, "valid": True},
        {"id": "3", "success": True, "task_index": 0, "valid": True},
    ]


def test_validate_cli_invalid(tmp_path):
    shutil.copytree(DATASET_DIR, tmp_path, dirs_exist_ok=True)
    state_path = tmp_path / "episodes" / "0" / "obs" / "arms" / "left" / "state.parquet"
    _inject_null_qpos(state_path)

    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "openarm_dataset.validate",
            str(tmp_path),
            *DISABLE_THRESHOLDS,
        ],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 1
    assert result.stderr == (
        "episodes/0/obs/arms/left/state.parquet: includes null values\n"
    )
    assert Dataset(tmp_path).meta.episodes == [
        {"id": "0", "success": False, "task_index": 0, "valid": False},
        {"id": "3", "success": True, "task_index": 0, "valid": True},
    ]


def test_validate_cli_no_update_metadata(tmp_path):
    shutil.copytree(DATASET_DIR, tmp_path, dirs_exist_ok=True)
    metadata_yaml = (tmp_path / "metadata.yaml").read_text()

    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "openarm_dataset.validate",
            str(tmp_path),
            "--no-update-metadata",
            *DISABLE_THRESHOLDS,
        ],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0
    assert (tmp_path / "metadata.yaml").read_text() == metadata_yaml


POSE_DATASET_DIR = Path(__file__).parent / "fixture" / "dataset_0.4.0_pose"


def test_validate_pose_dataset():
    errors = []
    assert Dataset(POSE_DATASET_DIR).validate(on_error=errors.append)
    assert errors == []


def _inject_large_qpos(path, column="qpos", value=100.0):
    df = pd.read_parquet(path)
    values = df[column].tolist()
    first = list(values[0])
    first[0] = value
    values[0] = first
    df[column] = values
    df.to_parquet(path)


def _inject_qpos_jump(path, column="qpos", jump=10.0):
    """Move the last frame away from the previous one to create one jump."""
    df = pd.read_parquet(path)
    values = df[column].tolist()
    last = list(values[-1])
    last[0] = float(values[-2][0]) + jump
    values[-1] = last
    df[column] = values
    df.to_parquet(path)


def test_validate_detects_qpos_absmax(tmp_path):
    shutil.copytree(DATASET_DIR, tmp_path, dirs_exist_ok=True)
    state_path = tmp_path / "episodes" / "0" / "obs" / "arms" / "left" / "state.parquet"
    _inject_large_qpos(state_path)

    errors = []
    assert not Dataset(tmp_path).validate(on_error=errors.append, qpos_absmax=6.28)
    assert errors == [
        "episodes/0/obs/arms/left/state.parquet: qpos absmax=100.0000 > 6.28"
    ]


def test_validate_detects_qpos_jump(tmp_path):
    shutil.copytree(DATASET_DIR, tmp_path, dirs_exist_ok=True)
    state_path = tmp_path / "episodes" / "0" / "obs" / "arms" / "left" / "state.parquet"
    _inject_qpos_jump(state_path)

    errors = []
    assert not Dataset(tmp_path).validate(
        on_error=errors.append, qpos_jump_threshold=1.0
    )
    assert errors == [
        "episodes/0/obs/arms/left/state.parquet: 1 qpos jump(s) > 1.0 rad (max=10.0000)"
    ]


def test_validate_skips_qpos_checks_for_null_file(tmp_path):
    shutil.copytree(DATASET_DIR, tmp_path, dirs_exist_ok=True)
    state_path = tmp_path / "episodes" / "0" / "obs" / "arms" / "left" / "state.parquet"
    _inject_null_qpos(state_path)

    errors = []
    assert not Dataset(tmp_path).validate(
        on_error=errors.append, qpos_absmax=6.28, qpos_jump_threshold=1.0
    )
    assert errors == ["episodes/0/obs/arms/left/state.parquet: includes null values"]


def test_validate_detects_short_episode():
    errors = []
    assert not Dataset(DATASET_DIR).validate(on_error=errors.append, min_duration=2.0)
    assert errors == ["episodes/3: duration=0.81s < 2.0s"]


def test_validate_accepts_clean_dataset():
    errors = []
    assert Dataset(DATASET_DIR).validate(
        on_error=errors.append,
        qpos_absmax=6.28,
        qpos_jump_threshold=1.0,
        min_duration=0.5,
    )
    assert errors == []


def _pose_action_path(root, episode="0", side="left"):
    return root / "episodes" / episode / "action" / "arms" / side / "state.parquet"


def _inject_pose_row(path, row, pose_values):
    """Overwrite one row's 8-dim pose list ([x,y,z, qw,qx,qy,qz, gripper])."""
    df = pd.read_parquet(path)
    values = df["pose"].tolist()
    values[row] = list(pose_values)
    df["pose"] = values
    df.to_parquet(path)


def test_validate_detects_zero_norm_quaternion(tmp_path):
    shutil.copytree(POSE_DATASET_DIR, tmp_path, dirs_exist_ok=True)
    # Zero-norm quaternion (all finite): downstream _pose_to_rot6d divides by
    # the norm and emits NaN for this row, so it must fail validation.
    _inject_pose_row(_pose_action_path(tmp_path), 0, [0.1, 0.2, 0.3, 0, 0, 0, 0, 0.5])

    errors = []
    assert not Dataset(tmp_path).validate(on_error=errors.append)
    assert errors == [
        "episodes/0/action/arms/left/state.parquet: 1 quaternion(s) with zero norm"
    ]


def test_validate_detects_non_finite_quaternion(tmp_path):
    shutil.copytree(POSE_DATASET_DIR, tmp_path, dirs_exist_ok=True)
    # inf in a quaternion component: the null/NaN pass uses pc.is_nan and does
    # not flag inf, so without this check inf would pass and propagate as NaN.
    _inject_pose_row(
        _pose_action_path(tmp_path), 0, [0.1, 0.2, 0.3, math.inf, 0, 0, 0, 0.5]
    )

    errors = []
    assert not Dataset(tmp_path).validate(on_error=errors.append)
    assert errors == [
        "episodes/0/action/arms/left/state.parquet: "
        "includes non-finite quaternion values"
    ]


def test_validate_pose_check_is_scoped_to_quaternion_components(tmp_path):
    shutil.copytree(POSE_DATASET_DIR, tmp_path, dirs_exist_ok=True)
    # Huge position and gripper, but a unit quaternion: the quaternion checks
    # must pass. Confirms the norm/finite checks read indices 3:7 only.
    _inject_pose_row(
        _pose_action_path(tmp_path), 0, [1e6, 1e6, 1e6, 1.0, 0.0, 0.0, 0.0, 1e6]
    )

    errors = []
    assert Dataset(tmp_path).validate(on_error=errors.append)
    assert errors == []


def test_validate_pose_check_skips_null_file(tmp_path):
    shutil.copytree(POSE_DATASET_DIR, tmp_path, dirs_exist_ok=True)
    # A null in the pose list is already reported by the null/NaN pass; the
    # quaternion check must skip that file rather than double-report.
    df = pd.read_parquet(_pose_action_path(tmp_path))
    values = df["pose"].tolist()
    values[0] = None
    df["pose"] = values
    df.to_parquet(_pose_action_path(tmp_path))

    errors = []
    assert not Dataset(tmp_path).validate(on_error=errors.append)
    assert errors == [
        "episodes/0/action/arms/left/state.parquet: includes null values"
    ]


def test_validate_qpos_dataset_unaffected_by_pose_check():
    # A qpos-only dataset has no pose attributes, so the quaternion check is
    # a no-op and the existing qpos-only validation behavior is unchanged.
    errors = []
    assert Dataset(DATASET_DIR).validate(on_error=errors.append)
    assert errors == []


def test_validate_pose_update_metadata(tmp_path):
    shutil.copytree(POSE_DATASET_DIR, tmp_path, dirs_exist_ok=True)
    _inject_pose_row(_pose_action_path(tmp_path), 0, [0.1, 0.2, 0.3, 0, 0, 0, 0, 0.5])

    assert not Dataset(tmp_path).validate(update_metadata=True)
    assert Dataset(tmp_path).meta.episodes == [
        {"id": "0", "success": False, "task_index": 0, "valid": False},
        {"id": "3", "success": True, "task_index": 0, "valid": True},
    ]
