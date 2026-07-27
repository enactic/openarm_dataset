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
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from openarm_dataset.dataset import Dataset
from openarm_dataset.kinematics import create_engines

FIXTURE_DIR = Path(__file__).parent / "fixture"
QPOS_DIR = FIXTURE_DIR / "dataset_0.4.0_qpos"

SIDES = ("right", "left")


@pytest.fixture(scope="module")
def engines():
    return create_engines()


def state_path(root, episode_id, type_, side):
    return (
        Path(root) / "episodes" / episode_id / type_ / "arms" / side / "state.parquet"
    )


def make_pose_only_actions(source, output, engines):
    """Write ``source`` to ``output`` with action qpos replaced by FK poses.

    The dataset_0.4.0_pose fixture's pose values are synthetic and not
    reachable, so IK targets must be derived from the qpos fixture via FK.
    """
    Dataset(source).write(output, format="openarm")
    for episode in Dataset(output).meta.episodes:
        for side in SIDES:
            path = state_path(output, episode["id"], "action", side)
            table = pq.read_table(path)
            qpos = np.asarray(table.column("qpos").to_pylist(), dtype=np.float32)
            pose = engines[side].qpos_to_pose(qpos)
            table = table.drop_columns(["qpos"]).append_column(
                "pose", pa.array(list(pose), type=pa.list_(pa.float32()))
            )
            pq.write_table(table.replace_schema_metadata(None), path)


def test_on_the_fly_fk_matches_engine(engines):
    dataset = Dataset(QPOS_DIR, kinematics=engines)
    episode = dataset.meta.episodes[0]
    raw = Dataset(QPOS_DIR).load_action(episode)
    action = dataset.load_action(episode, state="pose")
    for side in SIDES:
        qpos = raw[f"arms/{side}/qpos"].to_numpy(dtype=np.float32)
        np.testing.assert_allclose(
            action[f"arms/{side}/pose"].to_numpy(dtype=np.float32),
            engines[side].qpos_to_pose(qpos),
            rtol=1e-6,
        )


# mink's frozen-DOF task carries infinite weights; some BLAS backends emit
# RuntimeWarnings for the resulting inf-times-zero products even though the
# solutions stay finite. Upstream behavior, not ours.
@pytest.mark.filterwarnings("ignore::RuntimeWarning:mink.tasks.task")
def test_on_the_fly_ik_qpos_reproduces_pose_under_fk(engines, tmp_path):
    output = tmp_path / "pose_only"
    make_pose_only_actions(QPOS_DIR, output, engines)

    dataset = Dataset(output, kinematics=engines)
    action = dataset.load_action(dataset.meta.episodes[0], state="qpos")
    for side in SIDES:
        qpos = action[f"arms/{side}/qpos"].to_numpy(dtype=np.float32)
        table = pq.read_table(state_path(output, "0", "action", side))
        pose = np.asarray(table.column("pose").to_pylist(), dtype=np.float32)
        fk_pose = engines[side].qpos_to_pose(qpos)
        np.testing.assert_allclose(fk_pose[:, :3], pose[:, :3], atol=1e-2)
        quat_dot = np.clip(np.abs(np.sum(fk_pose[:, 3:7] * pose[:, 3:7], axis=1)), 0, 1)
        assert (2 * np.arccos(quat_dot) < 5e-3).all()
        np.testing.assert_allclose(qpos[:, 7], pose[:, 7], atol=1e-6)
