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

"""Parallel conversion must produce output identical to the serial path.

These tests do not require ``lerobot``; they operate on the emitted files. The
guarantee is: for any ``jobs``, the data parquet, stats, and metadata are
byte/numeric-identical to a serial (``jobs=1``) run — only the (still valid)
video mp4 bytes may differ, since the ffmpeg thread count varies with ``jobs``.
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from openarm_dataset import Dataset

FIXTURE_DIR = Path(__file__).parent / "fixture"
DATASET_0_3_0_PATH = FIXTURE_DIR / "dataset_0.3.0"
FPS = 30


def _convert(tmp_path: Path, fmt: str, jobs: int) -> Path:
    out = tmp_path / f"{fmt}_jobs{jobs}"
    dataset = Dataset(DATASET_0_3_0_PATH)
    dataset.set_smoothing(1.0)
    dataset.write(
        out,
        format=fmt,
        fps=FPS,
        train_split=0.8,
        success_only=False,
        jobs=jobs,
    )
    return out


def _assert_data_parquet_equal(a: Path, b: Path):
    """Compare every data parquet, including numpy-array list columns."""
    a_files = sorted(p.relative_to(a) for p in a.glob("data/**/*.parquet"))
    b_files = sorted(p.relative_to(b) for p in b.glob("data/**/*.parquet"))
    assert a_files == b_files, "data parquet layout differs"
    assert a_files, "no data parquet files were written"
    for rel in a_files:
        da = pd.read_parquet(a / rel)
        db = pd.read_parquet(b / rel)
        assert list(da.columns) == list(db.columns)
        assert len(da) == len(db)
        for col in da.columns:
            va, vb = da[col].to_list(), db[col].to_list()
            for xa, xb in zip(va, vb):
                np.testing.assert_array_equal(np.asarray(xa), np.asarray(xb))


def _assert_json_equal(a: Path, b: Path, rel: str):
    with (a / rel).open() as f:
        ja = json.load(f)
    with (b / rel).open() as f:
        jb = json.load(f)
    assert ja == jb, f"{rel} differs between serial and parallel runs"


def _assert_videos_valid(out: Path):
    videos = list(out.glob("videos/**/*.mp4"))
    assert videos, "no video files were written"
    for v in videos:
        assert v.stat().st_size > 0, f"empty video {v}"


def test_v21_parallel_matches_serial(tmp_path):
    serial = _convert(tmp_path, "lerobot_v2.1", jobs=1)
    parallel = _convert(tmp_path, "lerobot_v2.1", jobs=4)

    _assert_data_parquet_equal(serial, parallel)
    _assert_json_equal(serial, parallel, "meta/info.json")
    _assert_json_equal(serial, parallel, "meta/stats.json")

    for name in ("episodes.jsonl", "episodes_stats.jsonl", "tasks.jsonl"):
        with (serial / "meta" / name).open() as f:
            sa = [json.loads(line) for line in f]
        with (parallel / "meta" / name).open() as f:
            pa = [json.loads(line) for line in f]
        assert sa == pa, f"meta/{name} differs between serial and parallel runs"

    _assert_videos_valid(parallel)


def test_v30_parallel_matches_serial(tmp_path):
    serial = _convert(tmp_path, "lerobot_v3.0", jobs=1)
    parallel = _convert(tmp_path, "lerobot_v3.0", jobs=4)

    _assert_data_parquet_equal(serial, parallel)
    _assert_json_equal(serial, parallel, "meta/info.json")
    _assert_json_equal(serial, parallel, "meta/stats.json")

    ep_serial = pd.read_parquet(
        serial / "meta" / "episodes" / "chunk-000" / "file-000.parquet"
    )
    ep_parallel = pd.read_parquet(
        parallel / "meta" / "episodes" / "chunk-000" / "file-000.parquet"
    )
    assert list(ep_serial.columns) == list(ep_parallel.columns)
    for col in ep_serial.columns:
        for xa, xb in zip(ep_serial[col].to_list(), ep_parallel[col].to_list()):
            np.testing.assert_array_equal(np.asarray(xa), np.asarray(xb))

    tasks_serial = pd.read_parquet(serial / "meta" / "tasks.parquet")
    tasks_parallel = pd.read_parquet(parallel / "meta" / "tasks.parquet")
    pd.testing.assert_frame_equal(tasks_serial, tasks_parallel)

    _assert_videos_valid(parallel)


@pytest.mark.parametrize("fmt", ["lerobot_v2.1", "lerobot_v3.0", "gr00t"])
def test_default_jobs_runs(tmp_path, fmt):
    """The default (jobs=None => all cores) path converts without error."""
    out = _convert(tmp_path, fmt, jobs=None)
    assert (out / "meta" / "info.json").exists()
    _assert_videos_valid(out)
