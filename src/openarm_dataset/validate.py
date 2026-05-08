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

"""Validate OpenArm dataset."""

import argparse
import pathlib
import sys

import pandas as pd


def validate(path: str | pathlib.Path) -> list[str]:
    """Validate an OpenArm dataset.

    Args:
        path: Path to the dataset directory.

    Returns:
        A list of error messages. An empty list means the dataset is valid.

    """
    errors = []
    root = pathlib.Path(path)
    for qpos_path in sorted(root.rglob("qpos.parquet")):
        df = pd.read_parquet(qpos_path)
        if df.isnull().any().any():
            relative = qpos_path.relative_to(root)
            errors.append(f"{relative}: qpos.parquet includes null values")
    return errors


def main():
    """Validate OpenArm dataset."""
    parser = argparse.ArgumentParser(description="Validate OpenArm dataset")
    parser.add_argument(
        "input",
        help="Path of an OpenArm dataset to validate",
        type=pathlib.Path,
    )
    args = parser.parse_args()
    errors = validate(args.input)
    if errors:
        for error in errors:
            print(error, file=sys.stderr)
        sys.exit(1)
    else:
        print("Dataset is valid.")


if __name__ == "__main__":
    main()
