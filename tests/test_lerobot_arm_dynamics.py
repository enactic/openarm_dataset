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

"""Arm velocity/torque export and smoothing switch-off."""

from pathlib import Path
import json

import numpy as np
import pandas as pd
import pytest

from openarm_dataset import Dataset

FIXTURE_DIR = Path(__file__).parent / "fixture"
DYNAMICS_DATASET = FIXTURE_DIR / "dataset_0.4.0_qpos"
POSITION_ONLY_DATASET = FIXTURE_DIR / "dataset_0.2.0"
FPS = 30
ARMS = ("right", "left")
JOINTS = [f"joint{i}" for i in range(1, 8)] + ["gripper"]
FORMATS = ("lerobot_v2.1", "lerobot_v3.0")


def _read_data(output: Path) -> pd.DataFrame:
    paths = sorted((output / "data").rglob("*.parquet"))
    return pd.concat([pd.read_parquet(p) for p in paths], ignore_index=True)


def _features(output: Path) -> dict:
    return json.loads((output / "meta/info.json").read_text())["features"]


def _expected(dataset: Dataset, kind: str) -> np.ndarray:
    """Sample the source like the exporter: same timeline, same smoothing."""
    rows = []
    for episode in dataset.meta.episodes:
        for sample in dataset.sample(hz=FPS, episode=episode, state="qpos"):
            rows.append(
                np.concatenate([sample.obs[f"arms/{arm}/{kind}"] for arm in ARMS])
            )
    return np.stack(rows).astype(np.float32)


@pytest.mark.parametrize("format_", FORMATS)
def test_velocity_and_torque_are_exported(tmp_path, format_):
    dataset = Dataset(DYNAMICS_DATASET)
    dataset.write(tmp_path, format=format_, fps=FPS, arm_dynamics=True)

    data = _read_data(tmp_path)
    features = _features(tmp_path)
    columns = list(data.columns)
    assert columns[:4] == [
        "action",
        "observation.state",
        "observation.velocity",
        "observation.torque",
    ]
    assert list(features)[:4] == columns[:4]
    for feature, kind, suffix in (
        ("observation.velocity", "qvel", "velocity"),
        ("observation.torque", "qtorque", "torque"),
    ):
        names = [f"{arm}_{joint}.{suffix}" for arm in ARMS for joint in JOINTS]
        assert features[feature]["names"] == names
        assert features[feature]["shape"] == [16]
        assert features[feature]["dtype"] == "float32"
        if format_ == "lerobot_v3.0":
            assert features[feature]["fps"] == FPS
        else:
            assert "fps" not in features[feature]
        np.testing.assert_array_equal(
            np.stack(data[feature].to_numpy()), _expected(dataset, kind)
        )
        stats = json.loads((tmp_path / "meta/stats.json").read_text())[feature]
        assert stats["count"] == [len(data)]


def test_v21_episode_stats_include_dynamics(tmp_path):
    Dataset(DYNAMICS_DATASET).write(
        tmp_path, format="lerobot_v2.1", fps=FPS, arm_dynamics=True
    )

    lines = (tmp_path / "meta/episodes_stats.jsonl").read_text().splitlines()
    data = _read_data(tmp_path)
    for line in lines:
        record = json.loads(line)
        episode = data[data.episode_index == record["episode_index"]]
        for feature in ("observation.velocity", "observation.torque"):
            values = np.stack(episode[feature].to_numpy())
            np.testing.assert_allclose(
                record["stats"][feature]["mean"], values.mean(axis=0), rtol=1e-6
            )


def test_v30_episode_stats_include_dynamics(tmp_path):
    Dataset(DYNAMICS_DATASET).write(
        tmp_path, format="lerobot_v3.0", fps=FPS, arm_dynamics=True
    )

    episodes = pd.read_parquet(next((tmp_path / "meta/episodes").rglob("*.parquet")))
    data = _read_data(tmp_path)
    for _, row in episodes.iterrows():
        episode = data[data.episode_index == row["episode_index"]]
        for feature in ("observation.velocity", "observation.torque"):
            values = np.stack(episode[feature].to_numpy())
            np.testing.assert_allclose(
                row[f"stats/{feature}/mean"], values.mean(axis=0), rtol=1e-6
            )


@pytest.mark.parametrize("format_", FORMATS)
def test_arm_dynamics_are_off_by_default(tmp_path, format_):
    Dataset(DYNAMICS_DATASET).write(tmp_path, format=format_, fps=FPS)

    assert "observation.velocity" not in _read_data(tmp_path).columns
    assert "observation.torque" not in _features(tmp_path)


def test_position_only_recordings_have_no_dynamics(tmp_path):
    Dataset(POSITION_ONLY_DATASET).write(
        tmp_path, format="lerobot_v2.1", fps=FPS, arm_dynamics=True
    )

    features = _features(tmp_path)
    assert "observation.velocity" not in features
    assert "observation.torque" not in features


def test_gr00t_output_has_no_dynamics(tmp_path):
    Dataset(DYNAMICS_DATASET).write(tmp_path, format="gr00t", fps=FPS)

    assert "observation.velocity" not in _features(tmp_path)


@pytest.mark.parametrize("format_", FORMATS)
def test_zero_smoothing_cutoff_disables_smoothing(tmp_path, format_):
    Dataset(DYNAMICS_DATASET).write(
        tmp_path, format=format_, fps=FPS, smoothing_cutoff=0, arm_dynamics=True
    )

    raw = Dataset(DYNAMICS_DATASET)  # no smoothing set
    data = _read_data(tmp_path)
    np.testing.assert_array_equal(
        np.stack(data["observation.velocity"].to_numpy()), _expected(raw, "qvel")
    )


def test_cli_arm_dynamics(tmp_path, monkeypatch):
    from openarm_dataset import convert

    monkeypatch.setattr(
        "sys.argv",
        [
            "openarm-dataset-convert",
            str(DYNAMICS_DATASET),
            str(tmp_path),
            "--format",
            "lerobot_v3.0",
            "--arm-dynamics",
        ],
    )
    convert.main()
    features = _features(tmp_path)
    assert "observation.velocity" in features
    assert "observation.torque" in features


@pytest.mark.parametrize("format_", ("openarm", "gr00t"))
def test_cli_arm_dynamics_rejected_for_other_formats(tmp_path, monkeypatch, format_):
    from openarm_dataset import convert

    monkeypatch.setattr(
        "sys.argv",
        [
            "openarm-dataset-convert",
            str(DYNAMICS_DATASET),
            str(tmp_path / "out"),
            "--format",
            format_,
            "--arm-dynamics",
        ],
    )
    with pytest.raises(SystemExit):
        convert.main()
