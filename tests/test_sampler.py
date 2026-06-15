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
from types import SimpleNamespace

import numpy as np
import pandas as pd
import random

from openarm_dataset.camera import Camera, Frame
from openarm_dataset.sampler import Sampler


class DummyFrame(Frame):
    def load(self):
        h = 100
        w = 100
        return np.random.randint(0, 256, size=(h, w, 3), dtype=np.uint8)


class DummyCamera(Camera):
    def __init__(self):
        n = 240
        period_ns = 33_000_000  # 30Hz
        start = pd.Timestamp("2026-03-14 00:00:00").value
        self.tar_path = None
        self.all_files = [Path(f"{start + i * period_ns}.jpeg") for i in range(n)]

    def get_frame(self, index: int):
        return Frame(self.all_files[index])


class DummyDataset:
    def __init__(self):
        self.meta = SimpleNamespace(episodes=[{"id": "0"}])
        self._obs = {
            "arms/right/qpos": self._generate_dummy_embodiment_data(2000),
            "arms/left/qpos": self._generate_dummy_embodiment_data(2000),
        }
        self._action = {
            "arms/right/qpos": self._generate_dummy_embodiment_data(2000),
            "arms/left/qpos": self._generate_dummy_embodiment_data(2000),
        }
        self._cameras = {name: DummyCamera() for name in self.camera_names}

    @property
    def camera_names(self):
        return ["ceiling", "head", "left_wrist", "right_wrist"]

    def load_obs(self, *args, **kwargs):
        return self._obs

    def load_action(self, *args, **kwargs):
        return self._action

    def load_camera(self, camera, *args, **kwargs):
        return self._cameras[camera]

    def load_cameras(self, *args, **kwargs):
        return self._cameras

    def _generate_dummy_embodiment_data(self, n):
        columns = [
            "joint1",
            "joint2",
            "joint3",
            "joint4",
            "joint5",
            "joint6",
            "joint7",
            "gripper",
        ]
        index = pd.date_range(
            start="2026-03-14 00:00:00",
            unit="ns",
            periods=n,
            freq="4ms",
        )
        values = np.cumsum(
            np.random.normal(0, 0.01, size=(n, len(columns))),
            axis=0,
        )
        df = pd.DataFrame(
            values,
            index=index,
            columns=columns,
        )
        df.index = df.index.astype("int64") / 1e9
        df.index.name = "timestamp"
        return df


def test_sample():
    dataset = DummyDataset()
    sampler = Sampler()
    samples = list(
        sampler.sample(
            dataset,
            episode=dataset.meta.episodes[0],
            hz=10,
        )
    )

    obs = dataset.load_obs()
    action = dataset.load_action()
    cameras = dataset.load_cameras()
    # Exclude edges for taking candidates safely.
    sample = samples[random.randint(1, len(samples) - 2)]

    for name, position in sample.obs.items():
        idx = obs[name].index.searchsorted(sample.timestamp)
        candidates = [
            obs[name].iloc[idx - 1],
            obs[name].iloc[idx],
            obs[name].iloc[idx + 1],
        ]
        assert any(np.allclose(position, cand) for cand in candidates)

    for name, position in sample.action.items():
        idx = action[name].index.searchsorted(sample.timestamp)
        candidates = [
            action[name].iloc[idx - 1],
            action[name].iloc[idx],
            action[name].iloc[idx + 1],
        ]
        assert any(np.allclose(position, cand) for cand in candidates)

    for name, frame in sample.cameras.items():
        idx = np.searchsorted(
            cameras[name].load_timestamps(),
            sample.timestamp,
        )
        candidates = [
            cameras[name].get_frame(idx - 1),
            cameras[name].get_frame(idx),
            cameras[name].get_frame(idx + 1),
        ]
        assert frame in candidates


# def test_sample_with_load_camera_data_false():
#     dataset = DummyDataset()
#     sampler = Sampler(load_camera_data=False)
#     samples = list(
#         sampler.sample(
#             dataset,
#             episode_index=0,
#             hz=10,
#         )
#     )

#     cameras = dataset.load_cameras()
#     # Exclude edges for taking candidates safely.
#     sample = samples[random.randint(1, len(samples) - 2)]

#     for name, path in sample.cameras.items():
#         idx = np.searchsorted(
#             cameras[name].load_timestamps(),
#             sample.timestamp,
#         )
#         candidates = [
#             cameras[name].all_files[idx - 1],
#             cameras[name].all_files[idx],
#             cameras[name].all_files[idx + 1],
#         ]
#         assert any(path == cand for cand in candidates)
