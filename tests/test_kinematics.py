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

import numpy as np
import pyarrow.parquet as pq
import pytest

from openarm_dataset.dataset import Dataset
from openarm_dataset.kinematics import augment, create_engines

FIXTURE_DIR = Path(__file__).parent / "fixture"
POSE_DIR = FIXTURE_DIR / "dataset_0.4.0_pose"
QPOS_DIR = FIXTURE_DIR / "dataset_0.4.0_qpos"

SIDES = ("right", "left")


class FakeEngine:
    """Stands in for SideKinematics without requiring the real solver."""

    def qpos_to_pose(self, qpos):
        return (np.asarray(qpos, dtype=np.float32) + 1.0).astype(np.float32)

    def pose_to_qpos(self, pose, seed_qpos=None):
        qpos = (np.asarray(pose, dtype=np.float32) - 1.0).astype(np.float32)
        return qpos, 0, 0


def run_augment(source, output, engines):
    """Write ``source`` to ``output`` and augment it in place, like main()."""
    Dataset(source).write(output, format="openarm")
    return augment(Dataset(output), engines)


def state_path(root, episode_id, type_, side):
    return (
        Path(root) / "episodes" / episode_id / type_ / "arms" / side / "state.parquet"
    )


def test_augment_adds_missing_attributes_and_metrics(tmp_path):
    output = tmp_path / "out"
    stats = run_augment(POSE_DIR, output, {side: FakeEngine() for side in SIDES})
    dataset = Dataset(output)
    action = dataset.load_action(dataset.meta.episodes[0])
    obs = dataset.load_obs(dataset.meta.episodes[0])
    np.testing.assert_allclose(
        action["arms/right/qpos"].to_numpy(),
        action["arms/right/pose"].to_numpy() - 1.0,
        rtol=1e-6,
    )
    np.testing.assert_allclose(
        obs["arms/right/pose"].to_numpy(),
        obs["arms/right/qpos"].to_numpy() + 1.0,
        rtol=1e-6,
    )
    # 2 episodes x 2 sides for each direction.
    assert stats["pose_added"] == 4
    assert stats["qpos_added"] == 4
    assert len(stats["files"]) == 8
    record = next(
        f
        for f in stats["files"]
        if f["episode"] == "0" and f["type"] == "action" and f["component"] == "right"
    )
    rows = pq.read_table(state_path(output, "0", "action", "right")).num_rows
    assert record == {
        "attribute": "qpos",
        "rows": rows,
        "ik_failures": 0,
        "ik_unconverged": 0,
        "episode": "0",
        "type": "action",
        "name": "arms",
        "component": "right",
    }


# mink's frozen-DOF task carries infinite weights; some BLAS backends emit
# RuntimeWarnings for the resulting inf-times-zero products even though the
# solutions stay finite. Upstream behavior, not ours.
@pytest.mark.filterwarnings("ignore::RuntimeWarning:mink.tasks.task")
def test_real_ik_qpos_reproduces_pose_under_fk(tmp_path):
    # FK first: derive reachable poses from the qpos fixture. The pose
    # fixture's values are synthetic and not reachable, so targets must
    # come from FK.
    fk_output = tmp_path / "fk"
    run_augment(QPOS_DIR, fk_output, create_engines())

    # Strip qpos from the action files, leaving pose-only actions.
    for episode in Dataset(fk_output).meta.episodes:
        for side in SIDES:
            path = state_path(fk_output, episode["id"], "action", side)
            table = pq.read_table(path)
            pq.write_table(table.drop_columns(["qpos"]), path)

    engines = create_engines()
    ik_output = tmp_path / "ik"
    stats = run_augment(fk_output, ik_output, engines)
    assert stats["qpos_added"] == 4
    assert stats["ik_failures"] == 0

    for side in SIDES:
        table = pq.read_table(state_path(ik_output, "0", "action", side))
        pose = np.asarray(table.column("pose").to_pylist(), dtype=np.float32)
        qpos = np.asarray(table.column("qpos").to_pylist(), dtype=np.float32)
        fk_pose = engines[side].qpos_to_pose(qpos)
        np.testing.assert_allclose(fk_pose[:, :3], pose[:, :3], atol=1e-2)
        quat_dot = np.clip(np.abs(np.sum(fk_pose[:, 3:7] * pose[:, 3:7], axis=1)), 0, 1)
        assert (2 * np.arccos(quat_dot) < 5e-3).all()
        np.testing.assert_allclose(qpos[:, 7], pose[:, 7], atol=1e-6)
