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
import pytest

import numpy.testing as npt

from openarm_dataset.camera import Camera

CAMERA_DIR = (
    Path(__file__).parent
    / "fixture"
    / "dataset_0.3.0"
    / "episodes"
    / "0"
    / "cameras"
    / "ceiling"
)


def test_num_frames():
    camera = Camera("ceiling", CAMERA_DIR)
    assert camera.num_frames == 3


def test_get_frame():
    camera = Camera("ceiling", CAMERA_DIR)
    frame = camera.get_frame(0)
    assert frame.timestamp == pytest.approx(1772010251.619682)
    assert frame.load().shape == (600, 960, 3)


def test_frames():
    camera = Camera("ceiling", CAMERA_DIR)
    frames = camera.frames()
    frame = next(frames)
    assert frame.timestamp == pytest.approx(1772010251.619682)
    assert frame.load().shape == (600, 960, 3)
    assert frame.path == CAMERA_DIR / "1772010251619682157.jpeg"


def test_load_timestamps():
    camera = Camera("ceiling", CAMERA_DIR)
    timestamps = camera.load_timestamps()
    npt.assert_allclose(
        timestamps,
        [
            1772010251.619682,
            1772010251.6290832,
            1772010251.6632507,
        ],
    )
