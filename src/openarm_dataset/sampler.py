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

"""Sampler for OpenArm Dataset."""

from __future__ import annotations
from typing import TYPE_CHECKING
from collections.abc import Iterator, Mapping

import numpy as np
import pandas as pd

if TYPE_CHECKING:
    from .dataset import Dataset
    from .camera import Camera, Frame


class Sample(Mapping):
    """Data structure for sampled data points.

    Attributes:
        timestamp (float): The Unix timestamp of the sample.
        obs (dict): Dictionary of observation data.
        action (dict): Dictionary of action data.
        cameras (dict): Dictionary containing a frame of camera.

    """

    def __init__(
        self,
        timestamp: float,
        obs: dict[str, np.ndarray],
        action: dict[str, np.ndarray],
        cameras: dict[str, Frame],
    ):
        """Initialize Sample."""
        self._data = dict(
            timestamp=timestamp,
            obs=obs,
            action=action,
            cameras=cameras,
        )

    def __getitem__(self, key):
        """Return data for the key."""
        return self._data[key]

    def __iter__(self):
        """Return iterator."""
        return iter(self._data)

    def __len__(self):
        """Return length of keys."""
        return len(self._data)

    def __str__(self):
        """Return a string representation of Sample."""
        return f"Sample(timestamp={self.timestamp})"

    def __repr__(self):
        """Return a string representation of Sample."""
        return str(self)

    @property
    def timestamp(self) -> float:
        """Get timestamp."""
        return self._data["timestamp"]

    @property
    def obs(self) -> dict[str, np.ndarray]:
        """Get obs."""
        return self._data["obs"]

    @property
    def action(self) -> dict[str, np.ndarray]:
        """Get action."""
        return self._data["action"]

    @property
    def cameras(self) -> dict[str, Frame]:
        """Get cameras."""
        return self._data["cameras"]


class Sampler:
    """Sampler for OpenArm Dataset."""

    def sample(
        self,
        dataset: Dataset,
        episode_index: int,
        hz: float,
    ) -> Iterator[Sample]:
        """Sample the all modalities data to the specified hz."""
        obs = dataset.load_obs(episode_index, use_unixtime=True)
        action = dataset.load_action(
            dataset.meta.episodes[episode_index], use_unixtime=True
        )
        cameras = dataset.load_cameras(dataset.meta.episodes[episode_index])

        sampled_times = self._sample_timestamps(hz, obs, action, cameras)

        return self._sample(sampled_times, obs, action, cameras)

    def _sample_timestamps(self, hz, obs, action, cameras) -> np.ndarray:
        """Calculate the common valid time range across all modalities."""
        start_time = 0
        end_time = np.inf

        for data in (obs, action):
            for df in data.values():
                if df.empty:
                    continue
                start_time = max(start_time, df.index[0])
                end_time = min(end_time, df.index[-1])
        for data in cameras.values():
            timestamps = data.load_timestamps()
            if not timestamps:
                continue
            start_time = max(start_time, timestamps[0])
            end_time = min(end_time, timestamps[-1])

        return np.arange(start_time, end_time, 1.0 / hz)

    def _sample(
        self,
        sampled_times: np.ndarray,
        obs: dict[str, pd.DataFrame],
        action: dict[str, pd.DataFrame],
        cameras: dict[str, Camera],
    ) -> Iterator[Sample]:
        original_times = {}
        original_times["obs"] = {}
        for name, df in obs.items():
            original_times["obs"][name] = df.index.to_numpy()
        original_times["action"] = {}
        for name, df in action.items():
            original_times["action"][name] = df.index.to_numpy()
        original_times["cameras"] = {}
        for name, camera in cameras.items():
            original_times["cameras"][name] = np.array(camera.load_timestamps())

        obs_action_values = {}
        obs_action_values["obs"] = {}
        for name, df in obs.items():
            obs_action_values["obs"][name] = df.to_numpy()
        obs_action_values["action"] = {}
        for name, df in action.items():
            obs_action_values["action"][name] = df.to_numpy()

        for sampled_time in sampled_times:
            yield Sample(
                sampled_time,
                *self._search_data(
                    sampled_time,
                    original_times,
                    obs_action_values,
                    cameras,
                ),
            )

    def _search_data(
        self,
        target_time: float,
        original_times: dict[str, dict[str, np.ndarray]],
        obs_action_values: dict[str, dict[str, np.ndarray]],
        cameras: dict[str, Camera],
    ) -> tuple[dict, dict, dict]:
        target_obs_action = {}
        for obs_action in ("obs", "action"):
            target_obs_action[obs_action] = {}
            for name, values in obs_action_values[obs_action].items():
                times = original_times[obs_action][name]
                idx = times.searchsorted(target_time)
                target_obs_action[obs_action][name] = values[idx]

        target_cameras = {}
        for name, camera in cameras.items():
            times = original_times["cameras"][name]
            idx = times.searchsorted(target_time)
            target_cameras[name] = camera.get_frame(idx)

        return target_obs_action["obs"], target_obs_action["action"], target_cameras
