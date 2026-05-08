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

import subprocess
import sys
from pathlib import Path

import pandas as pd

from openarm_dataset.dataset import Dataset

DATASET_DIR = Path(__file__).parent / "fixture" / "dataset_0.2.0"


def _drain(gen):
    """Collect yielded error messages and boolean return value from validate()."""
    errors = []
    try:
        while True:
            errors.append(next(gen))
    except StopIteration as e:
        return errors, e.value


def test_validate_valid_dataset():
    errors, valid = _drain(Dataset(DATASET_DIR).validate())
    assert errors == []
    assert valid is True


def test_validate_invalid_dataset_with_null_qpos(tmp_path):
    import shutil

    shutil.copy(DATASET_DIR / "metadata.yaml", tmp_path / "metadata.yaml")
    src = DATASET_DIR / "episodes" / "0" / "obs" / "arms" / "left" / "qpos.parquet"
    dest_dir = tmp_path / "episodes" / "0" / "obs" / "arms" / "left"
    dest_dir.mkdir(parents=True)
    df = pd.read_parquet(src)
    values = df["value"].tolist()
    values[0] = None
    df["value"] = values
    df.to_parquet(dest_dir / "qpos.parquet")

    errors, valid = _drain(Dataset(tmp_path).validate())
    assert len(errors) == 1
    assert "qpos.parquet" in errors[0]
    assert "null" in errors[0]
    assert valid is False


def test_validate_multiple_invalid_qpos(tmp_path):
    import shutil

    shutil.copy(DATASET_DIR / "metadata.yaml", tmp_path / "metadata.yaml")
    src_dir = DATASET_DIR / "episodes" / "0" / "obs" / "arms"
    for side in ("left", "right"):
        src = src_dir / side / "qpos.parquet"
        dest_dir = tmp_path / "episodes" / "0" / "obs" / "arms" / side
        dest_dir.mkdir(parents=True)
        df = pd.read_parquet(src)
        values = df["value"].tolist()
        values[0] = None
        df["value"] = values
        df.to_parquet(dest_dir / "qpos.parquet")

    errors, valid = _drain(Dataset(tmp_path).validate())
    assert len(errors) == 2
    assert valid is False


def test_validate_cli_valid_dataset():
    result = subprocess.run(
        [sys.executable, "-m", "openarm_dataset.validate", str(DATASET_DIR)],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0
    assert "valid" in result.stdout


def test_validate_cli_invalid_dataset(tmp_path):
    import shutil

    shutil.copy(DATASET_DIR / "metadata.yaml", tmp_path / "metadata.yaml")
    src = DATASET_DIR / "episodes" / "0" / "obs" / "arms" / "left" / "qpos.parquet"
    dest_dir = tmp_path / "episodes" / "0" / "obs" / "arms" / "left"
    dest_dir.mkdir(parents=True)
    df = pd.read_parquet(src)
    df.iloc[0, df.columns.get_loc("value")] = None
    df.to_parquet(dest_dir / "qpos.parquet")

    result = subprocess.run(
        [sys.executable, "-m", "openarm_dataset.validate", str(tmp_path)],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 1
    assert "null" in result.stderr
