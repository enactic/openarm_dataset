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

"""Repair OpenArm dataset.

Fill isolated single-frame gaps (a ``null`` or ``NaN`` in a ``qpos``/``qvel``/
``qtorque``/``value`` array) by averaging the immediately preceding and
following frame values. Gaps spanning two or more consecutive frames, and gaps
at the first or last frame, cannot be averaged and are left untouched.
"""

import argparse
import pathlib
import shutil
import sys
import tempfile

import numpy as np
import pandas as pd

from .dataset import Dataset


def _copy_dataset(input_path: pathlib.Path, output_path: pathlib.Path) -> None:
    """Copy an OpenArm dataset, images are symlinked instead of copied."""
    output_path.mkdir(parents=True, exist_ok=True)

    src_dataset = Dataset(input_path)
    # copy metadata.
    src_dataset.meta.write(pathlib.Path(output_path))

    # save episodes
    for episode in src_dataset.meta.episodes:
        episode_path = src_dataset.episode_path(
            episode
        )  # root / episodes / 0 / {cameras, obs, action}
        for item in episode_path.iterdir():
            if item.is_dir() and item.name == "cameras":
                # Symlink camera images instead of copying.
                target = (
                    pathlib.Path(output_path)
                    / episode_path.relative_to(src_dataset.root_path)
                    / item.name
                )
                target.parent.mkdir(parents=True, exist_ok=True)
                if target.exists():
                    target.unlink()
                target.symlink_to(item.resolve())
            else:
                # Copy everything else (parquet files, metadata, etc.).
                target = (
                    pathlib.Path(output_path)
                    / episode_path.relative_to(src_dataset.root_path)
                    / item.name
                )
                if item.is_dir():
                    shutil.copytree(item, target)
                else:
                    shutil.copy2(item, target)


def repair_dataset(
    input_path: pathlib.Path,
    output_path: pathlib.Path | None = None,
    on_repair=None,
    on_error=None,
) -> None:
    """Repair an OpenArm dataset.

    Args:
        input_path: Path of the dataset to repair.
        output_path: If given, the dataset is copied here and the copy is repaired, leaving the input untouched. If ``None``, the input dataset is repaired in place.
        on_repair: Optional callable invoked with a message string for each
            repaired gap.
        on_error: Optional callable invoked with a message string for each gap
            that could not be repaired.

    """
    if output_path is not None:
        _copy_dataset(input_path, output_path)
        target = output_path
    else:
        target = input_path

    dataset = Dataset(target)
    checked_paths = set()
    for episode_index in range(dataset.num_episodes):
        for type_name in ("obs", "action"):
            for attribute in dataset.get_embodiment_attributes(
                type_name, episode_index
            ):
                path = attribute["path"]
                if path in checked_paths or not path.exists():
                    continue
                checked_paths.add(path)
                repaired, unrepairable = _repair_parquet(path)
                relative = path.relative_to(dataset.root_path)
                if repaired and on_repair is not None:
                    on_repair(f"{relative}: repaired {repaired} value(s)")
                if unrepairable and on_error is not None:
                    on_error(
                        f"{relative}: {unrepairable} value(s) could not be "
                        "repaired (consecutive or boundary gap)"
                    )


def _repair_parquet(path: pathlib.Path) -> tuple[int, int]:
    """Repair isolated single-frame gaps in one parquet file.

    Returns:
        A ``(num_repaired, num_unrepairable)`` tuple.

    """
    df = pd.read_parquet(path)
    total_repaired = 0
    total_unrepairable = 0
    changed = False
    for column in df.columns:
        if column == "timestamp":
            continue
        repaired, unrepairable, new_values = _repair_column(df[column])
        total_repaired += repaired
        total_unrepairable += unrepairable
        if new_values is not None:
            df[column] = new_values
            changed = True
    if changed:
        _write_parquet_atomically(df, path)
    return total_repaired, total_unrepairable


def _write_parquet_atomically(df: pd.DataFrame, path: pathlib.Path) -> None:
    """Write a parquet file without replacing the original until write succeeds."""
    original_mode = path.stat().st_mode & 0o7777
    with tempfile.NamedTemporaryFile(
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".parquet",
    ) as tmp_file:
        tmp_path = pathlib.Path(tmp_file.name)
        df.to_parquet(tmp_path, index=False)
        tmp_path.chmod(original_mode)
        tmp_path.replace(path)


def _repair_column(series: pd.Series) -> tuple[int, int, list | None]:
    """Repair isolated single-frame gaps in one array-valued column.

    Returns:
        ``(num_repaired, num_unrepairable, new_values)`` where ``new_values`` is
        the rebuilt list of per-frame arrays, or ``None`` if nothing changed.

    """
    cells = series.tolist()
    n = len(cells)

    # Find the first present cell to determine array width and element dtype.
    width = None
    dtype = np.float64
    for cell in cells:
        if cell is not None:
            array = np.asarray(cell)
            width = array.shape[0] if array.ndim else 1
            dtype = array.dtype
            break
    if width is None:
        # Every frame is missing; nothing to anchor interpolation on.
        return 0, 0, None

    # Build a (n, width) float matrix; missing values become NaN.
    matrix = np.full((n, width), np.nan, dtype=np.float64)
    for i, cell in enumerate(cells):
        if cell is not None:
            matrix[i] = np.asarray(cell, dtype=np.float64).reshape(width)

    repaired = 0
    unrepairable = 0
    changed_rows = set()
    for j in range(width):
        nan_rows = np.nonzero(np.isnan(matrix[:, j]))[0]
        for i in nan_rows:
            if (
                0 < i < n - 1
                and not np.isnan(matrix[i - 1, j])
                and not np.isnan(matrix[i + 1, j])
            ):
                matrix[i, j] = (matrix[i - 1, j] + matrix[i + 1, j]) / 2
                repaired += 1
                changed_rows.add(int(i))
            else:
                unrepairable += 1

    if not changed_rows:
        return repaired, unrepairable, None

    new_values = list(cells)
    for i in changed_rows:
        new_values[i] = matrix[i].astype(dtype)
    return repaired, unrepairable, new_values


def main():
    """Repair OpenArm dataset."""
    parser = argparse.ArgumentParser(
        description=(
            "Repair an OpenArm dataset by filling isolated single-frame gaps "
            "with the average of the neighboring frames"
        )
    )
    parser.add_argument(
        "input",
        help="Path of an OpenArm dataset to repair",
        type=pathlib.Path,
    )
    parser.add_argument(
        "-o",
        "--output",
        help="Path to write the repaired dataset to. Repairs in place if omitted.",
        type=pathlib.Path,
        default=None,
    )
    args = parser.parse_args()
    repair_dataset(
        args.input,
        args.output,
        on_repair=lambda message: print(message),
        on_error=lambda message: print(message, file=sys.stderr),
    )


if __name__ == "__main__":
    main()
