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

"""Add kinematics-derived state attributes to an OpenArm dataset.

0.4.0 ``state.parquet`` files are self-describing and can hold multiple
attribute columns. This module fills in the attributes that can be derived
with `openarm_control <https://github.com/enactic/openarm_control>`_:

- a file with ``qpos`` but no ``pose`` gains ``pose`` via forward kinematics
- a file with ``pose`` but no ``qpos`` gains ``qpos`` via inverse kinematics

Poses are end-effector poses of the configured EE frames expressed in the
MuJoCo model's world frame — the same convention openarm_control uses for
Cartesian teleoperation. Recorded ``pose`` data in a different reference
frame produces wrong (but plausible-looking) ``qpos``. FK/IK run against
the model at the chosen keyframe; time-varying lifter elevation is not
folded into the kinematics.
"""

import argparse
import pathlib

import numpy as np
import openarm_control
import pyarrow as pa
import pyarrow.parquet as pq
from tqdm import tqdm

from .dataset import Dataset
from .metadata import OpenArm

SIDES = ("right", "left")


class SideKinematics:
    """FK/IK for one arm side on top of ``openarm_control.Kinematics``.

    Dataset convention: ``qpos`` = [joint1..joint7, gripper] and ``pose`` =
    [x, y, z, qw, qx, qy, qz, gripper], both float32[8]. The gripper is not
    part of the kinematics and is passed through unchanged.
    """

    def __init__(
        self,
        side: str,
        kinematics,
        home_qpos: np.ndarray,
        pos_tolerance: float = 1e-3,
        ori_tolerance: float = 1e-2,
        max_rounds: int = 20,
        other_home_qpos: np.ndarray | None = None,
    ):
        """Initialize.

        Args:
            side: "right" or "left".
            kinematics: ``openarm_control.Kinematics`` built with IK enabled
                (or a compatible object).
            home_qpos: float32[8] driver values of the model keyframe, used
                as the IK seed when no recorded qpos is available.
            pos_tolerance: IK position convergence tolerance in meters.
            ori_tolerance: IK orientation convergence tolerance in radians.
            max_rounds: Maximum solver rounds per sample. One round runs the
                solver's own ``max_iters`` iterations; extra rounds are only
                used while the solution is outside the tolerances.
            other_home_qpos: float32[8] keyframe driver values of the other
                arm, used to keep it at a valid configuration while solving.
                Defaults to zeros.

        """
        self._side = side
        self._kin = kinematics
        self._home_qpos = np.asarray(home_qpos, dtype=np.float32)
        self._pos_tolerance = pos_tolerance
        self._ori_tolerance = ori_tolerance
        self._max_rounds = max_rounds
        # solve() returns float32[16] = right[8] + left[8].
        self._slice = slice(0, 8) if side == "right" else slice(8, 16)
        self._other_slice = slice(8, 16) if side == "right" else slice(0, 8)
        self._other_home_qpos = (
            np.zeros(8, dtype=np.float32)
            if other_home_qpos is None
            else np.asarray(other_home_qpos, dtype=np.float32)
        )

    def home_qpos(self) -> np.ndarray:
        """Return the float32[8] driver values of the model keyframe."""
        return self._home_qpos.copy()

    def qpos_to_pose(self, qpos: np.ndarray) -> np.ndarray:
        """Compute FK poses ([N, 8]) for driver values ([N, 8])."""
        qpos = np.asarray(qpos, dtype=np.float32)
        pose = np.empty((len(qpos), 8), dtype=np.float32)
        for index, values in enumerate(qpos):
            pose[index, :7] = self._kin.fk(self._side, values)
            pose[index, 7] = values[7]
        return pose

    def pose_to_qpos(
        self, pose: np.ndarray, seed_qpos: np.ndarray | None = None
    ) -> tuple[np.ndarray, int, int]:
        """Compute IK driver values ([N, 8]) for poses ([N, 8]).

        The solver is seeded with ``seed_qpos`` (falling back to the model
        keyframe) and then tracks the pose trajectory row by row, re-solving
        each row until it converges within the tolerances (or ``max_rounds``
        is reached). Rows where IK fails reuse the previous solution.

        Returns:
            Tuple of (qpos, number of failed rows, number of rows that did
            not reach the tolerances within ``max_rounds``).

        """
        pose = np.asarray(pose, dtype=np.float32)
        qpos = np.empty((len(pose), 8), dtype=np.float32)
        failures = 0
        unconverged = 0
        previous = np.asarray(
            self._home_qpos if seed_qpos is None else seed_qpos, dtype=np.float32
        )
        seed16 = np.empty(16, dtype=np.float32)
        seed16[self._slice] = previous
        seed16[self._other_slice] = self._other_home_qpos
        self._kin.sync(seed16)
        for index, values in enumerate(pose):
            target = values[:7]
            self._kin.set_target(self._side, target)
            row = None
            converged = False
            for _ in range(self._max_rounds):
                solved = self._kin.solve()
                if solved is None:
                    break
                row = solved[self._slice].copy()
                if self._converged(row, target):
                    converged = True
                    break
            if row is None:
                failures += 1
                row = previous.copy()
            elif not converged:
                unconverged += 1
            row[7] = values[7]
            qpos[index] = row
            previous = row
        return qpos, failures, unconverged

    def _converged(self, qpos_row: np.ndarray, target: np.ndarray) -> bool:
        pose = self._kin.fk(self._side, qpos_row)
        if np.linalg.norm(pose[:3] - target[:3]) > self._pos_tolerance:
            return False
        quat = target[3:7]
        norm = np.linalg.norm(quat)
        if norm > 0.0:
            quat = quat / norm
        dot = min(1.0, abs(float(np.dot(pose[3:7], quat))))
        return 2.0 * np.arccos(dot) <= self._ori_tolerance


def sides_for_mode(mode: str) -> tuple[str, ...]:
    """Return the sides selected by a --mode value (right/left/bimanual)."""
    return tuple(side for side in SIDES if mode in (side, "bimanual"))


def create_engines(
    args: argparse.Namespace | None = None,
) -> dict[str, SideKinematics]:
    """Build one SideKinematics per side selected by ``args.mode``.

    Args:
        args: Namespace holding openarm_control's common and IK flags
            (``register_common_args`` + ``register_ik_args``). Defaults to
            the openarm_control defaults (bimanual, cell scene, home
            keyframe).

    """
    if args is None:
        parser = argparse.ArgumentParser()
        openarm_control.register_common_args(parser)
        openarm_control.register_ik_args(parser)
        args = parser.parse_args([])
    ik_params = openarm_control.ik_params_from_args(args)
    engines = {}
    for side in sides_for_mode(args.mode):
        # One single-side setup per engine; only override the mode.
        side_args = argparse.Namespace(**vars(args))
        side_args.mode = side
        setup = openarm_control.setup_from_args(side_args)
        home_qpos = {}
        for home_side in SIDES:
            joints, gripper = setup.joint_resolver.get_driver(
                setup.data.qpos, home_side
            )
            home_qpos[home_side] = np.append(joints, gripper).astype(np.float32)
        other_side = "left" if side == "right" else "right"
        engines[side] = SideKinematics(
            side,
            openarm_control.Kinematics(setup, ik_params),
            home_qpos[side],
            other_home_qpos=home_qpos[other_side],
        )
    return engines


def augment(
    dataset: Dataset,
    engines: dict[str, SideKinematics],
) -> dict[str, int]:
    """Add derived qpos/pose attributes to ``dataset`` in place.

    Every OpenArm component ``state.parquet`` in the dataset gains the
    missing derived attribute (pose via FK, qpos via IK). Files that already
    have both attributes, and components without an engine, are left
    untouched.

    Args:
        dataset: Dataset to augment.
        engines: Mapping of component ("right"/"left") to SideKinematics.

    Returns:
        Counts of files that gained pose/qpos and of IK sample failures,
        plus a ``"files"`` list with one per-file metrics record
        (episode/type/name/component/attribute/rows/ik_failures/
        ik_unconverged) for every file that gained an attribute.

    """
    stats = {
        "pose_added": 0,
        "qpos_added": 0,
        "ik_failures": 0,
        "ik_unconverged": 0,
        "files": [],
    }
    for episode in tqdm(dataset.meta.episodes, desc="Adding kinematics", unit="ep"):
        episode_path = dataset.episode_path(episode)
        # obs first so an IK-derived obs qpos can seed the action IK.
        for type_ in ("obs", "action"):
            for name, embodiment in dataset.meta.equipment.embodiments.items():
                if not isinstance(embodiment, OpenArm):
                    continue
                for component in embodiment.components:
                    engine = engines.get(component)
                    if engine is None:
                        continue
                    path = episode_path / type_ / name / component / "state.parquet"
                    if not path.exists():
                        continue
                    record = _augment_state(path, engine, episode_path, name, component)
                    if record is None:
                        continue
                    record.update(
                        episode=episode["id"],
                        type=type_,
                        name=name,
                        component=component,
                    )
                    stats["files"].append(record)
                    stats[f"{record['attribute']}_added"] += 1
                    stats["ik_failures"] += record["ik_failures"]
                    stats["ik_unconverged"] += record["ik_unconverged"]
    return stats


def _augment_state(
    path: pathlib.Path,
    engine: SideKinematics,
    episode_path: pathlib.Path,
    name: str,
    component: str,
) -> dict | None:
    table = pq.read_table(path)
    columns = set(table.schema.names)
    if "qpos" in columns and "pose" not in columns:
        pose = engine.qpos_to_pose(_list_column(table, "qpos"))
        table = _append_list_column(table, "pose", pose)
        record = {
            "attribute": "pose",
            "rows": table.num_rows,
            "ik_failures": 0,
            "ik_unconverged": 0,
        }
    elif "pose" in columns and "qpos" not in columns:
        pose = _list_column(table, "pose")
        seed_qpos = _load_seed_qpos(episode_path, name, component, table)
        qpos, failures, unconverged = engine.pose_to_qpos(pose, seed_qpos=seed_qpos)
        table = _append_list_column(table, "qpos", qpos)
        record = {
            "attribute": "qpos",
            "rows": table.num_rows,
            "ik_failures": failures,
            "ik_unconverged": unconverged,
        }
    else:
        return None
    pq.write_table(table.replace_schema_metadata(None), path)
    return record


def _load_seed_qpos(
    episode_path: pathlib.Path,
    name: str,
    component: str,
    pose_table: pa.Table,
) -> np.ndarray | None:
    """Return the obs qpos row nearest the first pose timestamp, if any."""
    if pose_table.num_rows == 0:
        return None
    obs_path = episode_path / "obs" / name / component / "state.parquet"
    if not obs_path.exists():
        return None
    obs_table = pq.read_table(obs_path)
    if "qpos" not in obs_table.schema.names or obs_table.num_rows == 0:
        return None
    obs_timestamps = obs_table.column("timestamp").to_numpy().astype("int64")
    first_timestamp = pose_table.column("timestamp").to_numpy().astype("int64")[0]
    index = int(np.argmin(np.abs(obs_timestamps - first_timestamp)))
    return np.asarray(obs_table.column("qpos")[index].as_py(), dtype=np.float32)


def _list_column(table: pa.Table, name: str) -> np.ndarray:
    return np.asarray(table.column(name).to_pylist(), dtype=np.float32)


def _append_list_column(table: pa.Table, name: str, values: np.ndarray) -> pa.Table:
    column = pa.array(list(values), type=pa.list_(pa.float32()))
    return table.append_column(name, column)


def main():
    """Add FK/IK-derived qpos/pose attributes to an OpenArm dataset."""
    parser = argparse.ArgumentParser(
        description="Add FK/IK-derived qpos/pose attributes to an OpenArm dataset"
    )
    parser.add_argument(
        "input",
        help="Path of an OpenArm dataset to be augmented",
        type=pathlib.Path,
    )
    parser.add_argument(
        "--output",
        help="Path of the augmented OpenArm dataset (default: overwrite input)",
        type=pathlib.Path,
    )
    parser.add_argument(
        "--camera-format",
        help="How to store camera frames: 'dir' (one JPEG file per frame, "
        "default) or 'tar' (one .tar archive per camera) if output new dataset",
        default="dir",
        choices=["dir", "tar"],
    )
    openarm_control.register_common_args(parser)
    openarm_control.register_ik_args(parser)
    args = parser.parse_args()

    engines = create_engines(args)
    dataset = Dataset(args.input)
    if args.output is not None:
        output = pathlib.Path(args.output)
        dataset.write(output, format="openarm", camera_format=args.camera_format)
        dataset = Dataset(output)

    stats = augment(dataset, engines)
    print(
        f"Added pose to {stats['pose_added']} file(s), "
        f"qpos to {stats['qpos_added']} file(s)."
    )
    ik_files = [f for f in stats["files"] if f["attribute"] == "qpos"]
    if ik_files:
        print("IK metrics per file:")
        for f in ik_files:
            print(
                f"  episode {f['episode']} {f['type']:<6} "
                f"{f['name']}/{f['component']}: {f['rows']} rows, "
                f"{f['ik_failures']} failed, {f['ik_unconverged']} unconverged"
            )
    if stats["ik_failures"]:
        print(
            f"Warning: IK failed on {stats['ik_failures']} sample(s); "
            "the previous solution was reused for those samples."
        )
    if stats["ik_unconverged"]:
        print(
            f"Warning: IK did not reach the convergence tolerances on "
            f"{stats['ik_unconverged']} sample(s); the best-effort solution "
            "was kept for those samples."
        )


if __name__ == "__main__":
    main()
