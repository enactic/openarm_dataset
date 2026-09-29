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

Each ``traj_{i}.pkl`` is an lz4-frame-compressed pickle of one
``types.SimpleNamespace`` whose attributes carry the episode. The spec says
only the attribute names, dtypes and shapes matter, and a standard-library
class lets a reader unpickle it without installing ``openarm_dataset``.

Arms and columns
----------------
Arms come from the dataset's equipment metadata: every embodiment with
``qpos`` and components (OpenArm's ``right`` and ``left``), in the
metadata's order, which is also the order ``lerobot_v21`` uses. A trailing
``gripper`` joint is split off only when the metadata names it.
``joint_position`` (and ``joint_velocity`` / ``joint_effort``) hold the arm
joints without grippers; ``action`` keeps every commanded value, grippers
included (``[right joints..., right gripper, left joints..., left gripper]``),
so it can train or evaluate a policy that drives the grippers. The spec does
not require ``action`` and ``joint_position`` to share a width.

Embodiments without ``qpos`` and components, such as the cell lifter, are
not written: the format has no field for them.

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
import types
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from .dataset import Dataset

# robot_eval_logger's metadata.json enumerates the platforms it knows.
ROBOT_TYPE = "openarm"
# We always emit measured joint angles, so this is the state field the
# reader will look for.
CONTROL_MODE = "joint_position"


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


def _arms(dataset: Dataset) -> list[tuple[str, str, bool]]:
    """Return ``(embodiment, component, has_gripper)`` for every arm, in metadata order.

    An arm is an embodiment that records ``qpos`` per component (OpenArm's
    ``right`` and ``left``). Following ``lerobot_v21``, a gripper is split
    off only when the embodiment's last joint is named ``gripper``.
    """
    arms = []
    for name, embodiment in dataset.meta.equipment.embodiments.items():
        if not embodiment.components or "qpos" not in embodiment.attributes:
            continue  # e.g. the cell lifter: no field for it in this format
        has_gripper = bool(embodiment.joints) and embodiment.joints[-1] == "gripper"
        for component in embodiment.components:
            arms.append((name, component, has_gripper))
    return arms


def _stack(samples, attribute: str, suffix: str, arms, keep_grippers: bool):
    """Stack one per-arm modality across time.

    Returns ``(values, grippers)``: ``values`` concatenates the arms in order,
    without their trailing gripper unless ``keep_grippers``; ``grippers`` maps
    each component with a gripper to its per-step gripper values.
    """
    rows = []
    grippers = {component: [] for _, component, has_gripper in arms if has_gripper}
    for sample in samples:
        source = getattr(sample, attribute)
        row = []
        for name, component, has_gripper in arms:
            vector = np.asarray(
                source[f"{name}/{component}/{suffix}"], dtype=np.float32
            )
            if has_gripper:
                grippers[component].append(vector[-1])
                if not keep_grippers:
                    vector = vector[:-1]
            row.append(vector)
        rows.append(np.concatenate(row))
    return (
        np.asarray(rows, dtype=np.float32),
        {
            component: np.asarray(values, dtype=np.float32).reshape(-1, 1)
            for component, values in grippers.items()
        },
    )


def _stack_optional(samples, attribute: str, suffix: str, arms):
    """Stack an optional per-arm modality, or return None if absent.

    The source records ``qvel`` and ``qtorque`` alongside ``qpos``, and the
    target format has optional ``joint_velocity`` and ``joint_effort``
    fields, so passing them through keeps the conversion lossless. Grippers
    are dropped exactly as in ``joint_position``, so the step-level joint
    arrays share one width. Older datasets may not carry them, hence None.
    """
    source = getattr(samples[0], attribute)
    if not all(f"{name}/{component}/{suffix}" in source for name, component, _ in arms):
        return None
    values, _ = _stack(samples, attribute, suffix, arms, keep_grippers=False)
    return values


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

    tasks = dataset.meta.data.get("tasks") or []
    arms = _arms(dataset)
    if not arms:
        raise ValueError(
            "The equipment metadata declares no arm (an embodiment with 'qpos' "
            "per component); the robot_eval_logger format needs joint positions."
        )
    components = [component for _, component, _ in arms]
    with_gripper = [component for _, component, has_gripper in arms if has_gripper]
    if gripper_component is None:
        if len(with_gripper) > 1:
            raise ValueError(
                "This dataset records more than one arm with a gripper "
                f"({', '.join(with_gripper)}), so which one the required "
                "'gripper' field refers to is ambiguous. Pass "
                "gripper_component to choose. Every gripper is written in full as "
                "'<component>_gripper' regardless."
            )
        if not with_gripper:
            raise ValueError(
                "No arm in the equipment metadata has a gripper joint, but the "
                "robot_eval_logger format requires a 'gripper' field."
            )
        chosen = with_gripper[0]
    else:
        if gripper_component not in components:
            raise ValueError(
                f"gripper_component {gripper_component!r} is not in this "
                f"dataset; available: {', '.join(components)}"
            )
        if gripper_component not in with_gripper:
            raise ValueError(
                f"gripper_component {gripper_component!r} has no gripper joint; "
                f"arms with one: {', '.join(with_gripper) or 'none'}"
            )
        chosen = gripper_component

    # Only after the arguments check out, so a bad call leaves no empty run.
    run_dir = Path(output) / str(eval_id)
    run_dir.mkdir(parents=True, exist_ok=True)
    written = 0

    for episode in dataset.meta.episodes:
        if valid_only and not episode.valid():
            continue
        if success_only and not bool(episode.get("success", False)):
            continue

        samples = dataset.sample(hz=fps, episode=episode, state="qpos")
        if not samples:
            continue

        missing = [
            f"{name}/{component}/qpos"
            for name, component, _ in arms
            if f"{name}/{component}/qpos" not in samples[0].obs
        ]
        if missing:
            raise ValueError(
                f"Episode {episode.get('id', '?')} has no {', '.join(missing)} "
                "in its sampled observations, though the metadata declares them."
            )

        joint_position, obs_grippers = _stack(
            samples, "obs", "qpos", arms, keep_grippers=False
        )
        # Keep the commanded gripper values: without them the output can't
        # train or evaluate a policy that drives the grippers.
        action, _ = _stack(samples, "action", "qpos", arms, keep_grippers=True)

        steps = len(samples)
        gripper = obs_grippers[chosen]
        per_component = {
            f"{component}_gripper": values for component, values in obs_grippers.items()
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
        joint_velocity = _stack_optional(samples, "obs", "qvel", arms)
        if joint_velocity is not None:
            optional["joint_velocity"] = joint_velocity
        joint_effort = _stack_optional(samples, "obs", "qtorque", arms)
        if joint_effort is not None:
            optional["joint_effort"] = joint_effort

        record = types.SimpleNamespace(
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
