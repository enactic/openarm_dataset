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

"""Merge multiple OpenArm datasets."""

from __future__ import annotations

import argparse
import copy
import os
import pathlib
import shutil

import yaml

from .dataset import Dataset


class MergeError(Exception):
    """Raised when datasets cannot be merged."""


def merge_datasets(
    inputs: list[str | os.PathLike],
    output: str | os.PathLike,
    symlink: bool = False,
) -> None:
    """Merge multiple OpenArm datasets into one.

    Episodes are renumbered sequentially starting from 0. Tasks are
    deduplicated by prompt: tasks with identical prompt strings across
    datasets are treated as the same task.

    Args:
        inputs: Paths to input datasets.
        output: Path for the merged output dataset.
        symlink: If True, create symlinks instead of copying episode data.

    Raises:
        MergeError: If fewer than two inputs are given or equipment is
            incompatible.

    """
    if len(inputs) < 2:
        raise MergeError("At least two input datasets are required")

    datasets = [Dataset(p) for p in inputs]
    output = pathlib.Path(output)

    _validate_compatibility(datasets)

    merged_tasks, task_index_maps = _merge_tasks(datasets)
    merged_episodes = _build_merged_episodes(datasets, task_index_maps)

    _write_metadata(datasets[0], merged_tasks, merged_episodes, output)
    _write_episodes(datasets, output, symlink)


def _validate_compatibility(datasets: list[Dataset]) -> None:
    ref = datasets[0]
    ref_embodiments = ref.meta.equipment.embodiments
    ref_cameras = set(ref.meta.equipment.perceptions.cameras)

    for i, ds in enumerate(datasets[1:], 1):
        eq = ds.meta.equipment

        if set(ref_embodiments) != set(eq.embodiments):
            raise MergeError(
                f"Dataset {i}: embodiment mismatch. "
                f"Expected {sorted(ref_embodiments)}, "
                f"got {sorted(eq.embodiments)}"
            )

        for name in ref_embodiments:
            ref_emb = ref_embodiments[name]
            ds_emb = eq.embodiments[name]
            if ref_emb.id != ds_emb.id or ref_emb.version != ds_emb.version:
                raise MergeError(
                    f"Dataset {i}: embodiment '{name}' mismatch. "
                    f"Expected {ref_emb.id} v{ref_emb.version}, "
                    f"got {ds_emb.id} v{ds_emb.version}"
                )

        ds_cameras = set(eq.perceptions.cameras)
        if ref_cameras != ds_cameras:
            raise MergeError(
                f"Dataset {i}: camera mismatch. "
                f"Expected {sorted(ref_cameras)}, got {sorted(ds_cameras)}"
            )


def _merge_tasks(
    datasets: list[Dataset],
) -> tuple[list[dict], list[dict[int, int]]]:
    merged_tasks: list[dict] = []
    prompt_to_index: dict[str, int] = {}
    task_index_maps: list[dict[int, int]] = []

    for ds in datasets:
        ds_map: dict[int, int] = {}
        for orig_idx, task in enumerate(ds.meta.tasks):
            prompt = task["prompt"]
            if prompt in prompt_to_index:
                ds_map[orig_idx] = prompt_to_index[prompt]
            else:
                new_idx = len(merged_tasks)
                merged_tasks.append(copy.deepcopy(task))
                prompt_to_index[prompt] = new_idx
                ds_map[orig_idx] = new_idx
        task_index_maps.append(ds_map)

    return merged_tasks, task_index_maps


def _build_merged_episodes(
    datasets: list[Dataset],
    task_index_maps: list[dict[int, int]],
) -> list[dict]:
    merged_episodes = []
    episode_counter = 0

    for ds_idx, ds in enumerate(datasets):
        for ep in ds.meta.episodes:
            merged_episodes.append(
                {
                    "id": str(episode_counter),
                    "success": ep["success"],
                    "task_index": task_index_maps[ds_idx][ep["task_index"]],
                }
            )
            episode_counter += 1

    return merged_episodes


def _write_metadata(
    ref_dataset: Dataset,
    tasks: list[dict],
    episodes: list[dict],
    output: pathlib.Path,
) -> None:
    equipment = copy.deepcopy(ref_dataset.meta.data.get("equipment", {}))
    if ref_dataset.meta.version is None:
        equipment = ref_dataset.meta._convert_unversioned_equipment()

    data = {
        "version": "0.3.0",
        "location": ref_dataset.meta.location,
        "operator": ref_dataset.meta.operator,
        "operation_type": ref_dataset.meta.operation_type,
        "tasks": tasks,
        "episodes": episodes,
        "equipment": equipment,
        "frequencies": copy.deepcopy(ref_dataset.meta.data.get("frequencies", {})),
    }

    output.mkdir(parents=True, exist_ok=True)
    with open(output / "metadata.yaml", "w") as f:
        yaml.safe_dump(data, f)


def _write_episodes(
    datasets: list[Dataset],
    output: pathlib.Path,
    symlink: bool,
) -> None:
    episode_counter = 0
    for ds in datasets:
        for ep_idx in range(ds.num_episodes):
            src = ds._episode_path(ep_idx)
            dst = output / "episodes" / str(episode_counter)
            if symlink:
                dst.parent.mkdir(parents=True, exist_ok=True)
                dst.symlink_to(src.resolve())
            else:
                shutil.copytree(src, dst)
            episode_counter += 1


def main():
    """CLI entry point for merging datasets."""
    parser = argparse.ArgumentParser(
        description="Merge multiple OpenArm datasets into one"
    )
    parser.add_argument(
        "inputs",
        nargs="+",
        help="Paths of OpenArm datasets to merge",
        type=pathlib.Path,
    )
    parser.add_argument(
        "-o",
        "--output",
        required=True,
        help="Path of merged output dataset",
        type=pathlib.Path,
    )
    parser.add_argument(
        "--symlink",
        action="store_true",
        default=False,
        help="Create symlinks instead of copying episode data",
    )

    args = parser.parse_args()
    merge_datasets(args.inputs, args.output, symlink=args.symlink)


if __name__ == "__main__":
    main()
