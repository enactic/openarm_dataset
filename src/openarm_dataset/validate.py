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


def main():
    """Validate OpenArm dataset."""
    parser = argparse.ArgumentParser(description="Validate OpenArm dataset")
    parser.add_argument(
        "input",
        help="Path of an OpenArm dataset to validate",
        type=pathlib.Path,
    )
    args = parser.parse_args()
    dataset = openarm_dataset.Dataset(args.input)
    errors = list(dataset.validate())
    if errors:
        for error in errors:
            print(error, file=sys.stderr)
        sys.exit(1)
    else:
        print("Dataset is valid.")


if __name__ == "__main__":
    main()
