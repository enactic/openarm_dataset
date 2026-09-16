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
import yaml
from openarm_dataset.dataset import Dataset

DATASET_DIR = Path(__file__).parent / "fixture" / "dataset_0.4.0_qpos"

# The CLI enables the qpos, duration and stream thresholds by default. Tests
# that only exercise the null check turn them off.
DISABLE_THRESHOLDS = [
    "--qpos-jump-threshold",
    "none",
    "--qpos-absmax",
    "none",
    "--min-duration",
    "none",
    "--max-stream-desync",
    "none",
    "--max-sample-gap",
    "none",
]


def _inject_null_qpos(state_path):
    df = pd.read_parquet(state_path)
    values = df["qpos"].tolist()
    values[0] = None
    df["qpos"] = values
    df.to_parquet(state_path)


def _truncate_stream(state_path, keep):
    """Cut a stream short, as an arm that stops reporting mid-recording does."""
    df = pd.read_parquet(state_path)
    df.head(keep).to_parquet(state_path, index=False)


def _drop_middle_samples(state_path, start, stop):
    """Drop a run of samples, leaving the stream's span intact but holed."""
    df = pd.read_parquet(state_path)
    pd.concat([df.iloc[:start], df.iloc[stop:]], ignore_index=True).to_parquet(
        state_path, index=False
    )


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


def test_validate_detects_long_episode():
    errors = []
    assert not Dataset(DATASET_DIR).validate(on_error=errors.append, max_duration=0.5)
    assert errors == [
        "episodes/0: duration=2.98s > 0.5s",
        "episodes/3: duration=0.81s > 0.5s",
    ]


def test_validate_max_duration_disabled_by_default():
    errors = []
    assert Dataset(DATASET_DIR).validate(on_error=errors.append, min_duration=0.5)
    assert errors == []


def test_validate_reports_both_duration_bounds():
    errors = []
    assert not Dataset(DATASET_DIR).validate(
        on_error=errors.append, min_duration=0.9, max_duration=1.0
    )
    assert errors == [
        "episodes/0: duration=2.98s > 1.0s",
        "episodes/3: duration=0.81s < 0.9s",
    ]


def test_validate_detects_truncated_stream(tmp_path):
    # Feedback that drops out mid-recording leaves one stream short while the
    # rest run on. Every file is still complete and NaN-free, so only the
    # comparison between streams can see it.
    shutil.copytree(DATASET_DIR, tmp_path, dirs_exist_ok=True)
    _truncate_stream(
        tmp_path / "episodes" / "0" / "obs" / "arms" / "left" / "state.parquet", 250
    )

    errors = []
    assert not Dataset(tmp_path).validate(on_error=errors.append, max_stream_desync=1.0)
    assert errors == [
        "episodes/0: stream durations differ by 1.99s > 1.0s "
        "(obs/arms/left/state.parquet=1.00s, obs/arms/right/state.parquet=2.98s)"
    ]


def test_validate_detects_mid_episode_gap(tmp_path):
    # The other half of the same fault: feedback that comes back leaves the
    # stream the right length overall, with a hole in the middle.
    shutil.copytree(DATASET_DIR, tmp_path, dirs_exist_ok=True)
    _drop_middle_samples(
        tmp_path / "episodes" / "0" / "obs" / "arms" / "left" / "state.parquet",
        100,
        600,
    )

    errors = []
    assert not Dataset(tmp_path).validate(on_error=errors.append, max_sample_gap=1.0)
    assert errors == [
        "episodes/0/obs/arms/left/state.parquet: 2.01s gap between samples > 1.0s"
    ]


def test_validate_gap_survives_a_stream_that_stays_in_sync(tmp_path):
    # Dropping samples from the middle doesn't move either end, so the desync
    # check passes it — the gap check is what has to catch this one.
    shutil.copytree(DATASET_DIR, tmp_path, dirs_exist_ok=True)
    _drop_middle_samples(
        tmp_path / "episodes" / "0" / "obs" / "arms" / "left" / "state.parquet",
        100,
        600,
    )

    errors = []
    assert Dataset(tmp_path).validate(on_error=errors.append, max_stream_desync=1.0)
    assert errors == []


def test_validate_stream_checks_disabled_by_default(tmp_path):
    shutil.copytree(DATASET_DIR, tmp_path, dirs_exist_ok=True)
    _truncate_stream(
        tmp_path / "episodes" / "0" / "obs" / "arms" / "left" / "state.parquet", 250
    )

    errors = []
    assert Dataset(tmp_path).validate(on_error=errors.append)
    assert errors == []


def test_validate_stream_checks_accept_differing_sample_rates():
    # obs runs at 250 Hz and action at ~31 Hz in the fixture, so their spans
    # differ by one slow sample. That is normal, not a desync.
    errors = []
    assert Dataset(DATASET_DIR).validate(
        on_error=errors.append, max_stream_desync=1.0, max_sample_gap=1.0
    )
    assert errors == []


def test_validate_cli_detects_truncated_stream(tmp_path):
    shutil.copytree(DATASET_DIR, tmp_path, dirs_exist_ok=True)
    _truncate_stream(
        tmp_path / "episodes" / "0" / "obs" / "arms" / "left" / "state.parquet", 250
    )
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "openarm_dataset.validate",
            str(tmp_path),
            "--min-duration",
            "none",
        ],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 1
    assert "episodes/0: stream durations differ by 1.99s > 1.0s" in result.stderr

    meta = yaml.safe_load((tmp_path / "metadata.yaml").read_text())
    assert [(ep["id"], ep["valid"]) for ep in meta["episodes"]] == [
        ("0", False),
        ("3", True),
    ]


def test_validate_detects_episode_without_data(tmp_path):
    # A camera-only episode has no parquet for the other checks to read, so
    # without this check it passes as valid.
    shutil.copytree(DATASET_DIR, tmp_path, dirs_exist_ok=True)
    shutil.rmtree(tmp_path / "episodes" / "0" / "obs")
    shutil.rmtree(tmp_path / "episodes" / "0" / "action")

    errors = []
    assert not Dataset(tmp_path).validate(on_error=errors.append)
    assert errors == [
        "episodes/0: no obs data",
        "episodes/0: no action data",
    ]


def test_validate_detects_episode_without_action_data(tmp_path):
    shutil.copytree(DATASET_DIR, tmp_path, dirs_exist_ok=True)
    shutil.rmtree(tmp_path / "episodes" / "0" / "action")

    errors = []
    assert not Dataset(tmp_path).validate(on_error=errors.append)
    assert errors == ["episodes/0: no action data"]


def test_validate_accepts_undeclared_embodiment_without_data(tmp_path):
    # `equipment.embodiments` declares a lifter that plenty of real datasets
    # never record; that is not a missing-data error.
    shutil.copytree(DATASET_DIR, tmp_path, dirs_exist_ok=True)
    for episode_id in ("0", "3"):
        for type_name in ("obs", "action"):
            shutil.rmtree(tmp_path / "episodes" / episode_id / type_name / "lifter")

    errors = []
    assert Dataset(tmp_path).validate(on_error=errors.append)
    assert errors == []


def test_validate_marks_episode_without_data_invalid(tmp_path):
    shutil.copytree(DATASET_DIR, tmp_path, dirs_exist_ok=True)
    shutil.rmtree(tmp_path / "episodes" / "0" / "obs")
    shutil.rmtree(tmp_path / "episodes" / "0" / "action")

    assert not Dataset(tmp_path).validate(update_metadata=True)
    meta = yaml.safe_load((tmp_path / "metadata.yaml").read_text())
    assert [(ep["id"], ep["valid"]) for ep in meta["episodes"]] == [
        ("0", False),
        ("3", True),
    ]


def test_validate_cli_max_duration(tmp_path):
    shutil.copytree(DATASET_DIR, tmp_path, dirs_exist_ok=True)
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "openarm_dataset.validate",
            str(tmp_path),
            "--max-duration",
            "0.9",
        ],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 1
    assert "episodes/0: duration=2.98s > 0.9s" in result.stderr


def test_validate_accepts_clean_dataset():
    errors = []
    assert Dataset(DATASET_DIR).validate(
        on_error=errors.append,
        qpos_absmax=6.28,
        qpos_jump_threshold=1.0,
        min_duration=0.5,
    )
    assert errors == []
