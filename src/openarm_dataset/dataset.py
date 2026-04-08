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

"""OpenArm Dataset."""

from pathlib import Path
import os

import pandas as pd
import scipy.signal as signal

from .camera import Camera
from .metadata import Metadata, Embodiment
from .sampler import Sampler, Sample


class Dataset:
    """OpenArm Dataset."""

    def __init__(
        self,
        path: str | os.PathLike,
        meta: Metadata = None,
        camera_names: list[str] = None,
    ):
        """Initialize Dataset.

        Args:
            path: Path of the dataset.
            meta: Metadata of the dataset. Uses the metadata stored in the
                dataset if None.
            camera_names: Names of the camera to use. Uses all cameras in the
                dataset if None.

        """
        self.root_path = Path(path)
        self.meta = Metadata(self.root_path / "metadata.yaml") if meta is None else meta
        self._camera_names = camera_names
        self._smoothing_cutoff = None

    def set_smoothing(self, cutoff: float):
        """Set smoothing."""
        self._smoothing_cutoff = cutoff

    @property
    def num_episodes(self) -> int:
        """Return number of episodes."""
        return self.meta.num_episodes

    @property
    def camera_names(self) -> list[str]:
        """Return camera names."""
        if self._camera_names is not None:
            return self._camera_names
        return list(self.meta.equipment.perceptions.cameras)

    def _episode_id(self, index: int) -> str:
        return self.meta.episodes[index]["id"]

    def _episode_path(self, episode_index: int = None) -> Path:
        if episode_index is None:
            return self.root_path
        episode_id = self._episode_id(episode_index)
        return self.root_path / "episodes" / episode_id

    def load_obs(
        self,
        episode_index: int,
        use_unixtime: bool = False,
        cutoff: float = None,
    ) -> dict[str, pd.DataFrame]:
        """Load obs data.

        Args:
            episode_index: Episode index to load.
            use_unixtime: If True, the DataFrame index is returned as Unix time
                (float64) instead of datetime64[ns].
            cutoff: If not None, smoothing is applied using this value.

        Returns:
            Dictionary mapping names to DataFrames.

        Example:
            {
                "arms/right/qpos": DataFrame,
                "arms/left/qpos": DataFrame,
            }

        """
        return self._load_obs_or_action(
            "obs",
            episode_index,
            use_unixtime,
            cutoff=cutoff or self._smoothing_cutoff,
        )

    def load_action(
        self,
        episode_index: int,
        use_unixtime: bool = False,
        cutoff: float = None,
    ) -> dict[str, pd.DataFrame]:
        """Load action data.

        Args:
            episode_index: Episode index to load.
            use_unixtime: If True, the DataFrame index is returned as Unix time
                (float64) instead of datetime64[ns].
            cutoff: If not None, smoothing is applied using this value.

        Returns:
            Dictionary mapping names to DataFrames.

        Example:
            {
                "arms/right/qpos": DataFrame,
                "arms/left/qpos": DataFrame,
            }

        """
        return self._load_obs_or_action(
            "action",
            episode_index,
            use_unixtime=use_unixtime,
            cutoff=cutoff or self._smoothing_cutoff,
        )

    def load_cameras(self, episode_index: int) -> dict[str, Camera]:
        """Load all camera data.

        Args:
            episode_index: Episode index to load.

        Returns:
            Dictionary mapping names to Camera.

        Example:
            {
                "ceiling": Camera,
                "head": Camera,
                "wrist_right": Camera,
                "wrist_left": Camera,
            }

        """
        return {
            name: self.load_camera(name, episode_index) for name in self.camera_names
        }

    def load_camera(self, name: str, episode_index: int) -> Camera:
        """Load camera data.

        Args:
            name: Camera name to load.
            episode_index: Episode index to load.

        Returns:
            Camera.

        """
        if name not in self.camera_names:
            raise KeyError(f"Camera {name} not found. Available: {self.camera_names}")
        base_path = self._episode_path(episode_index)
        # Unversioned dataset. This is for backward compatibility.
        if self.meta.version is None:
            path = base_path / f"{name}_image"
            if not path.exists() and name.endswith("_wrist"):
                path = base_path / f"{name.removesuffix('_wrist')}_image"
        else:
            path = base_path / "cameras" / name
        return Camera(name, path)

    def sample(
        self,
        hz: float,
        episode_index: int,
    ) -> list[Sample]:
        """Sample the all modalities data to the specified hz.

        Args:
            episode_index: Episode index to sample.
            hz: Sampling hz.

        Returns:
            List of Sample.

        Example:
            >>> samples = samples(10, 0)
            >>> samples[0].timestamp
            1773446407.1999931
            >>> samples[0].obs
            {
                "arms/right/qpos": np.ndarray,
                'arms/left/qpos': np.ndarray,
            }
            >>> samples[0].action
            {
                "arms/right/qpos": np.ndarray,
                'arms/left/qpos': np.ndarray,
            }
            >>> samples[0].cameras
            {
                "ceiling": Frame,
                "head": Frame,
                "wrist_right": Frame,
                "wrist_left": Frame,
            }

        """
        sampler = Sampler()
        return list(sampler.sample(self, episode_index, hz))

    def _load_obs_or_action(
        self,
        obs_or_action: str,
        episode_index: int,
        use_unixtime: bool = False,
        cutoff: float = None,
    ) -> dict[str, pd.DataFrame]:
        data = {}
        for name, embodiment in self.meta.equipment.embodiments.items():
            # Unversioned dataset.
            # This is for backward compatibility.
            if self.meta.version is None:
                base_path = self._episode_path(episode_index) / obs_or_action
            else:
                base_path = self._episode_path(episode_index) / obs_or_action / name
            if embodiment.components:
                for component in embodiment.components:
                    for attribute in embodiment.attributes:
                        key = f"{name}/{component}/{attribute}"
                        # Unversioned dataset.
                        # This is for backward compatibility.
                        if self.meta.version is None:
                            path = base_path / f"{component}_arm" / f"{attribute}.parquet"
                        else:
                            path = base_path / component / f"{attribute}.parquet"
                        data[key] = self._load_embodiment_data(
                            embodiment,
                            path,
                            use_unixtime=use_unixtime,
                            cutoff=cutoff,
                        )
            else:
                for attribute in embodiment.attributes:
                    key = f"{name}/{attribute}"
                    data[key] = self._load_embodiment_data(
                        embodiment,
                        base_path / f"{attribute}.parquet",
                        use_unixtime=use_unixtime,
                        cutoff=cutoff,
                    )
        return data

    def _load_embodiment_data(
        self,
        embodiment: Embodiment,
        path: str | os.PathLike,
        use_unixtime: bool = False,
        cutoff: float = None,
    ) -> pd.DataFrame:
        df = pd.read_parquet(path)
        # No version and 0.1.0 use "positions"
        if "positions" in df:
            column_name = "positions"
        else:
            column_name = "value"
        df[list(embodiment.joints)] = pd.DataFrame(
            df[column_name].tolist(),
            index=df.index,
        )
        df = df.drop(columns=[column_name])
        if use_unixtime:
            df["timestamp"] = df["timestamp"].astype("int64") / 1e9
        df = df.set_index("timestamp")
        if cutoff is not None:
            df = self._apply_smoothing(df, cutoff=cutoff)
        return df

    def _apply_smoothing(
        self,
        df: pd.DataFrame,
        cutoff: float = 1.0,
        fps: float = 250.0,
    ) -> pd.DataFrame:
        if df.empty or cutoff is None:
            return df
        if len(df) <= 15:
            return df

        nyquist = fps * 0.5
        Wn = cutoff / nyquist
        Wn = min(0.99, max(0.01, Wn))
        b, a = signal.butter(4, Wn, btype="low")

        filtered_values = signal.filtfilt(b, a, df.values, axis=0)
        return pd.DataFrame(filtered_values, index=df.index, columns=df.columns)
