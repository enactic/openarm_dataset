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

import openarm_dataset


def _threshold(value: str) -> float | None:
    """Parse a threshold value, where "none" disables the check."""
    if value.strip().lower() == "none":
        return None
    return float(value)


def main():
    """Validate OpenArm dataset."""
    parser = argparse.ArgumentParser(description="Validate OpenArm dataset")
    parser.add_argument(
        "input",
        help="Path of an OpenArm dataset to validate",
        type=pathlib.Path,
    )
    parser.add_argument(
        "--no-update-metadata",
        help="Do not record per-episode validity ('valid' flag) in the dataset metadata",
        action="store_true",
        default=False,
    )
    parser.add_argument(
        "--qpos-jump-threshold",
        help="Flag qpos frame-to-frame deltas above this value "
        "(default: %(default)s, 'none' to disable)",
        type=_threshold,
        default=1.0,
        metavar="RADIAN",
    )
    parser.add_argument(
        "--qpos-absmax",
        help="Flag qpos values whose absolute value exceeds this threshold "
        "(default: %(default)s, 'none' to disable)",
        type=_threshold,
        default=6.28,
        metavar="RADIAN",
    )
    parser.add_argument(
        "--min-duration",
        help="Flag episodes shorter than this duration "
        "(default: %(default)s, 'none' to disable)",
        type=_threshold,
        default=2.0,
        metavar="SECOND",
    )
    args = parser.parse_args()
    dataset = openarm_dataset.Dataset(args.input)
    valid = dataset.validate(
        on_error=lambda error: print(error, file=sys.stderr),
        update_metadata=not args.no_update_metadata,
        qpos_jump_threshold=args.qpos_jump_threshold,
        qpos_absmax=args.qpos_absmax,
        min_duration=args.min_duration,
    )
    if not valid:
        sys.exit(1)


if __name__ == "__main__":
    main()
