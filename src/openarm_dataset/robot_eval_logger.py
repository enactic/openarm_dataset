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

"""Convert an OpenArm dataset to the robot_eval_logger on-disk format.

Layout produced, per that project's DATA_FORMAT.md::

    <output>/<eval_id>/
        metadata.json
        traj_0.pkl
        traj_1.pkl
        ...

Each ``traj_{i}.pkl`` is an lz4-frame-compressed pickle of one object whose
attributes carry the episode. The class identity does not matter to the
reader, only the attribute names, dtypes and shapes.

Bimanual grippers
-----------------
The target schema requires a single ``gripper`` field of shape ``(T, 1)``,
but an OpenArm dataset may record two arms, each with its own gripper
(``qpos`` is ``[joint1..joint7, gripper]`` per arm). Nothing in the
metadata designates one arm as canonical -- ``leader``/``follower``
describe teleoperation devices, not which gripper an evaluation means.

So both grippers are always written losslessly as ``<component>_gripper``
extra attributes, which the spec permits and which matches the naming
``lerobot_v21`` already uses for per-component gripper ranges. The
required ``gripper`` field is taken from the only arm when there is one,
and otherwise the caller must name the arm via ``gripper_component``.
Guessing would silently mislabel every converted bimanual dataset.
"""

from __future__ import annotations

import json
import os
import pickle
import random
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from .dataset import Dataset

# robot_eval_logger's metadata.json enumerates the platforms it knows.
ROBOT_TYPE = "openarm"
# We always emit measured joint angles, so this is the state field the
# reader will look for.
CONTROL_MODE = "joint_position"


class Episode:
    """One trajectory, in the attribute layout the reader expects.

    Deliberately a plain container: DATA_FORMAT.md states the class name is
    irrelevant to loading and only the attributes matter, so pickling a
    lightweight object avoids making the reader import anything of ours.
    """

    def __init__(self, **attributes):
        """Set every given attribute on this episode."""
        for name, value in attributes.items():
            setattr(self, name, value)


def _require_lz4():
    """Import lz4.frame, or explain how to get it."""
    try:
        import lz4.frame
    except ModuleNotFoundError as err:  # pragma: no cover - trivial branch
        if err.name in ("lz4", "lz4.frame"):
            raise ModuleNotFoundError(
                "The robot_eval_logger format needs lz4: pip install lz4"
            ) from err
        raise
    return lz4.frame


def _components(keys) -> list[str]:
    """Return the arm components present, in a stable order.

    Sample keys look like ``arms/right/qpos``; the component is the middle
    segment. Sorted so a dataset always converts to the same column order
    rather than one that depends on dict iteration.
    """
    found = []
    for key in keys:
        parts = key.split("/")
        if len(parts) == 3 and parts[0] == "arms":
            if parts[1] not in found:
                found.append(parts[1])
    return sorted(found)


def _split(vector: np.ndarray) -> tuple[np.ndarray, float]:
    """Split one arm's qpos into (joints, gripper).

    Per kinematics.py the per-arm convention is ``[joint1..joint7,
    gripper]``, so the gripper is the trailing element.
    """
    return vector[:-1], float(vector[-1])


def _stack(samples, attribute: str, components: list[str]):
    """Build (joints, grippers) arrays across time for one modality."""
    joints_per_step = []
    grippers_per_step = []
    for sample in samples:
        source = getattr(sample, attribute)
        joints = []
        grippers = {}
        for component in components:
            vector = np.asarray(source[f"arms/{component}/qpos"], dtype=np.float32)
            arm_joints, gripper = _split(vector)
            joints.append(arm_joints)
            grippers[component] = gripper
        joints_per_step.append(np.concatenate(joints))
        grippers_per_step.append(grippers)
    return (
        np.asarray(joints_per_step, dtype=np.float32),
        grippers_per_step,
    )


def _stack_optional(samples, attribute: str, suffix: str, components: list[str]):
    """Stack an optional per-arm modality, or return None if absent.

    The source records ``qvel`` and ``qtorque`` alongside ``qpos``, and the
    target format has optional ``joint_velocity`` and ``joint_effort``
    fields, so passing them through keeps the conversion lossless. Older
    datasets may not carry them, hence the None.
    """
    source = getattr(samples[0], attribute)
    keys = [f"arms/{component}/{suffix}" for component in components]
    if not all(key in source for key in keys):
        return None
    rows = []
    for sample in samples:
        values = getattr(sample, attribute)
        # Drop each arm's trailing gripper element, exactly as
        # joint_position does, so every step-level array shares one D.
        rows.append(
            np.concatenate(
                [np.asarray(values[key], dtype=np.float32)[:-1] for key in keys]
            )
        )
    return np.asarray(rows, dtype=np.float32)


def to_robot_eval_logger(
    dataset: Dataset,
    output: str | os.PathLike,
    fps: int = 30,
    valid_only: bool = False,
    success_only: bool = False,
    gripper_component: str | None = None,
    robot_name: str = "openarm",
    eval_id: int | None = None,
    eval_name: str | None = None,
    location: str | None = None,
    evaluator_name: str | None = None,
) -> None:
    """Write ``dataset`` in the robot_eval_logger format under ``output``.

    Args:
        dataset: Source OpenArm dataset.
        output: Directory to create the ``<eval_id>`` run directory in.
        fps: Sampling rate; also recorded as ``action_frequency_hz``.
        valid_only: Skip episodes marked invalid by ``openarm-dataset-validate``.
        success_only: Skip episodes whose ``success`` flag is false.
        gripper_component: Which arm the required single ``gripper`` field
            refers to. Required when the dataset records more than one arm.
        robot_name: Human-readable robot name for ``metadata.json``.
        eval_id: Run identifier; a random positive integer when omitted.
        eval_name: Optional human-readable name for the run.
        location: Optional physical location.
        evaluator_name: Optional evaluator name.

    """
    lz4_frame = _require_lz4()

    if eval_id is None:
        # The spec asks for "a large positive integer"; the directory name
        # must match this value.
        eval_id = random.randrange(10**15, 10**16)

    run_dir = Path(output) / str(eval_id)
    run_dir.mkdir(parents=True, exist_ok=True)

    tasks = dataset.meta.data.get("tasks") or []
    written = 0

    for episode in dataset.meta.episodes:
        if valid_only and not episode.valid():
            continue
        if success_only and not bool(episode.get("success", False)):
            continue

        samples = dataset.sample(hz=fps, episode=episode, state="qpos")
        if not samples:
            continue

        components = _components(samples[0].obs.keys())
        if not components:
            raise ValueError(
                "No 'arms/<component>/qpos' entries in the sampled observations; "
                "the robot_eval_logger format needs joint positions."
            )
        if gripper_component is None:
            if len(components) > 1:
                raise ValueError(
                    "This dataset records more than one arm "
                    f"({', '.join(components)}), so which one the required "
                    "'gripper' field refers to is ambiguous. Pass "
                    "gripper_component to choose. Both are written in full as "
                    "'<component>_gripper' regardless."
                )
            chosen = components[0]
        else:
            if gripper_component not in components:
                raise ValueError(
                    f"gripper_component {gripper_component!r} is not in this "
                    f"dataset; available: {', '.join(components)}"
                )
            chosen = gripper_component

        joint_position, obs_grippers = _stack(samples, "obs", components)
        action, _ = _stack(samples, "action", components)

        steps = len(samples)
        gripper = np.asarray(
            [[obs_grippers[i][chosen]] for i in range(steps)], dtype=np.float32
        )
        per_component = {
            f"{component}_gripper": np.asarray(
                [[obs_grippers[i][component]] for i in range(steps)],
                dtype=np.float32,
            )
            for component in components
        }

        cameras = {
            name: np.asarray(
                [sample.cameras[name].load() for sample in samples], dtype=np.uint8
            )
            for name in samples[0].cameras
        }

        # Both are required by the target schema. Defaulting them would
        # silently mark every episode failed, or ship an empty instruction,
        # so let a malformed dataset fail loudly instead.
        try:
            success = bool(episode["success"])
            language_command = tasks[int(episode["task_index"])]["prompt"]
        except (KeyError, IndexError, TypeError) as err:
            raise ValueError(
                f"Episode {episode.get('id', '?')} is missing data the "
                f"robot_eval_logger format requires ({err}); "
                "'success' and a resolvable 'task_index' prompt are mandatory."
            ) from err

        timestamps = [sample.timestamp for sample in samples]
        optional = {}
        joint_velocity = _stack_optional(samples, "obs", "qvel", components)
        if joint_velocity is not None:
            optional["joint_velocity"] = joint_velocity
        joint_effort = _stack_optional(samples, "obs", "qtorque", components)
        if joint_effort is not None:
            optional["joint_effort"] = joint_effort

        record = Episode(
            language_command=language_command,
            success=success,
            episode_length=steps,
            duration_seconds=float(timestamps[-1] - timestamps[0]),
            collection_time=datetime.fromtimestamp(
                timestamps[0], tz=timezone.utc
            ).isoformat(),
            obs=cameras,
            action=action,
            joint_position=joint_position,
            gripper=gripper,
            **optional,
            **per_component,
        )

        raw = pickle.dumps(record, protocol=pickle.HIGHEST_PROTOCOL)
        (run_dir / f"traj_{written}.pkl").write_bytes(lz4_frame.compress(raw))
        written += 1

    metadata = {
        "eval_id": eval_id,
        "robot_name": robot_name,
        "robot_type": ROBOT_TYPE,
        "control_mode": CONTROL_MODE,
        "action_frequency_hz": float(fps),
        "time": datetime.now(tz=timezone.utc).isoformat(),
        "location": location,
        "evaluator_name": evaluator_name,
        "eval_name": eval_name,
    }
    with (run_dir / "metadata.json").open("w", encoding="utf-8") as handle:
        json.dump(metadata, handle, ensure_ascii=False, indent=4)
