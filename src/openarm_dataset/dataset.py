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

import os
from pathlib import Path
import shutil

import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq
import pandas as pd
import scipy.signal as signal

from .camera import Camera
from .metadata import Metadata
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

    def validate(self, on_error=None) -> bool:
        """Validate this dataset.

        Args:
            on_error: Optional callable that is called with an error message
                string for each validation error found. If ``None``, errors
                are not reported.

        Returns:
            ``True`` if the dataset is valid, ``False`` otherwise.

        """
        valid = True
        checked_paths = set()
        for episode in self.meta.episodes:
            for type_name in ("obs", "action"):
                for attribute in self.get_embodiment_attributes(type_name, episode):
                    path = attribute["path"]
                    if path in checked_paths or not path.exists():
                        continue
                    checked_paths.add(path)
                    file_meta = pq.read_metadata(path)
                    has_null = False
                    for rg_index in range(file_meta.num_row_groups):
                        row_group = file_meta.row_group(rg_index)
                        for col_index in range(row_group.num_columns):
                            col_meta = row_group.column(col_index)
                            col_name = col_meta.path_in_schema.split(".")[0]
                            if col_name == "timestamp":
                                continue
                            stats = col_meta.statistics
                            if (
                                stats is not None
                                and stats.has_null_count
                                and stats.null_count > 0
                            ):
                                has_null = True
                                break
                        if has_null:
                            break
                    if not has_null:
                        table = pq.read_table(path)
                        for col_name in table.schema.names:
                            if col_name == "timestamp":
                                continue
                            col = table.column(col_name)
                            flat = col.combine_chunks().values
                            if (
                                pa.types.is_floating(flat.type)
                                and pc.any(pc.is_nan(flat)).as_py()
                            ):
                                has_null = True
                                break
                    if has_null:
                        if on_error is not None:
                            on_error(
                                f"{path.relative_to(self.root_path)}: "
                                "includes null values"
                            )
                        valid = False
        return valid

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

    def episode_path(self, episode: dict = None) -> Path:
        """Return the path of the episode."""
        if episode is None:
            return self.root_path
        return self.root_path / "episodes" / episode["id"]

    def load_obs(
        self,
        episode: dict,
        use_unixtime: bool = False,
        cutoff: float = None,
    ) -> dict[str, pd.DataFrame]:
        """Load obs data.

        Args:
            episode: Episode to load.
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
        return self._load_embodiment_values(
            "obs",
            episode,
            use_unixtime,
            cutoff=cutoff or self._smoothing_cutoff,
        )

    def load_action(
        self,
        episode: dict,
        use_unixtime: bool = False,
        cutoff: float = None,
    ) -> dict[str, pd.DataFrame]:
        """Load action data.

        Args:
            episode: Episode to load.
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
        return self._load_embodiment_values(
            "action",
            episode,
            use_unixtime=use_unixtime,
            cutoff=cutoff or self._smoothing_cutoff,
        )

    def load_cameras(self, episode: dict) -> dict[str, Camera]:
        """Load all camera data.

        Args:
            episode: Episode to load.

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
        return {name: self.load_camera(name, episode) for name in self.camera_names}

    def load_camera(self, name: str, episode: dict) -> Camera:
        """Load camera data.

        Args:
            name: Camera name to load.
            episode: Episode to load.

        Returns:
            Camera.

        """
        if name not in self.camera_names:
            raise KeyError(f"Camera {name} not found. Available: {self.camera_names}")
        base_path = self.episode_path(episode)
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
        episode: dict,
    ) -> list[Sample]:
        """Sample the all modalities data to the specified hz.

        Args:
            episode: Episode to sample.
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
        return list(sampler.sample(self, episode, hz))

    def get_embodiment_attributes(self, type_: str, episode: dict):
        """Return the list of embodiment attributes for the given type and episode."""
        attributes = []
        for name, embodiment in self.meta.equipment.embodiments.items():
            # Unversioned dataset.
            # This is for backward compatibility.
            if self.meta.version is None:
                base_path = self.episode_path(episode) / type_
            else:
                base_path = self.episode_path(episode) / type_ / name
            if embodiment.components:
                for component in embodiment.components:
                    state_path = base_path / component / "state.parquet"
                    if state_path.exists():
                        for attr_name in ("qpos", "qvel", "qtorque"):
                            attributes.append(
                                {
                                    "key": f"{name}/{component}/{attr_name}",
                                    "embodiment": embodiment,
                                    "component": component,
                                    "name": attr_name,
                                    "path": state_path,
                                }
                            )
                        continue
                    for attribute in embodiment.attributes:
                        key = f"{name}/{component}/{attribute}"
                        # Unversioned dataset.
                        # This is for backward compatibility.
                        if self.meta.version is None:
                            path = (
                                base_path / f"{component}_arm" / f"{attribute}.parquet"
                            )
                        else:
                            path = base_path / component / f"{attribute}.parquet"
                        attributes.append(
                            {
                                "key": key,
                                "embodiment": embodiment,
                                "component": component,
                                "name": attribute,
                                "path": path,
                            }
                        )
            else:
                for attribute in embodiment.attributes:
                    key = f"{name}/{attribute}"
                    attributes.append(
                        {
                            "key": key,
                            "embodiment": embodiment,
                            "component": None,
                            "name": attribute,
                            "path": base_path / f"{attribute}.parquet",
                        }
                    )
        return attributes

    def _load_embodiment_values(
        self,
        type_: str,
        episode: dict,
        use_unixtime: bool = False,
        cutoff: float = None,
    ) -> dict[str, pd.DataFrame]:
        values = {}
        for attribute in self.get_embodiment_attributes(type_, episode):
            values[attribute["key"]] = self._load_embodiment_value(
                attribute,
                use_unixtime=use_unixtime,
                cutoff=cutoff,
            )
        return values

    def _load_embodiment_value(
        self,
        attribute: dict,
        use_unixtime: bool = False,
        cutoff: float = None,
    ) -> pd.DataFrame:
        df = pd.read_parquet(attribute["path"])
        if attribute["path"].name == "state.parquet":
            # 0.3.0 uses state.parquet with qpos/qvel/qtorque columns.
            column_name = attribute["name"]
            drop_columns = [c for c in ("qpos", "qvel", "qtorque") if c in df.columns]
        elif "positions" in df:
            # No version and 0.1.0 use "positions"
            column_name = "positions"
            drop_columns = ["positions"]
        else:
            column_name = "value"
            drop_columns = ["value"]
        df[list(attribute["embodiment"].joints)] = pd.DataFrame(
            df[column_name].tolist(),
            index=df.index,
        )
        df = df.drop(columns=drop_columns)
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

    def _write(self, output: str | os.PathLike):
        """Write this dataset as the latest OpenArm dataset format."""
        output = Path(output)
        self.meta.write(output)
        self._write_data(output)

    def write(self, output: str | os.PathLike, format: str | None = None, **options):
        """Write this dataset in the specified format."""
        if format is None or format == "openarm":
            return self._write(output)
        elif format == "lerobot_v2.1":
            from .lerobot_v21 import to_lerobotv21

            return to_lerobotv21(self, output, **options)
        elif format == "lerobot_v3.0":
            from .lerobot_v30 import to_lerobotv30

            return to_lerobotv30(self, output, **options)
        elif format == "gr00t":
            from .lerobot_v21 import to_gr00t

            return to_gr00t(self, output, **options)
        elif format == "rrd":
            try:
                from .rrd import to_rrd
            except ModuleNotFoundError as err:
                if err.name == "rerun":
                    raise ModuleNotFoundError(
                        "RRD export requires the optional dependency 'rerun-sdk'. Install with `pip install openarm_dataset[rerun]`."
                    ) from err
                raise

            return to_rrd(self, output, **options)
        else:
            raise ValueError(f"Unsupported format: {format}")

    def _write_data(self, output: Path):
        for episode in self.meta.episodes:
            self._write_episode(output, episode)

    def _write_episode(self, output: Path, episode: dict):
        self._write_embodiment_data(output, episode)
        self._write_camera_data(output, episode)

    def _write_embodiment_data(self, output: Path, episode: dict):
        written_state_paths = set()
        for type_ in ["action", "obs"]:
            for attribute in self.get_embodiment_attributes(type_, episode):
                embodiment = attribute["embodiment"]
                component = attribute["component"]
                name = attribute["name"]
                base_path = (
                    output / "episodes" / episode["id"] / type_ / embodiment.name
                )
                # 0.3.0 state.parquet (qpos/qvel/qtorque) is shared across
                # attributes for the same component; copy it once.
                if attribute["path"].name == "state.parquet":
                    if component:
                        new_path = base_path / component / "state.parquet"
                    else:
                        new_path = base_path / "state.parquet"
                    if new_path in written_state_paths:
                        continue
                    new_path.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(attribute["path"], new_path)
                    written_state_paths.add(new_path)
                    continue
                if component:
                    new_path = base_path / component / f"{name}.parquet"
                else:
                    new_path = base_path / f"{name}.parquet"
                new_path.parent.mkdir(parents=True, exist_ok=True)
                df = pd.read_parquet(attribute["path"])
                # No version and 0.1.0 use "positions"
                if "positions" in df:
                    df["value"] = df["positions"]
                    df = df.drop(columns=["positions"])
                    df.to_parquet(new_path)
                else:
                    shutil.copy2(attribute["path"], new_path)

    def _write_camera_data(self, output: os.PathLike, episode: dict):
        base_path = output / "episodes" / episode["id"]
        for name, camera in self.load_cameras(episode).items():
            if self.meta.version is None:
                if name == "left_wrist":
                    name = "wrist_left"
                elif name == "right_wrist":
                    name = "wrist_right"
            new_path = base_path / "cameras" / name
            new_path.parent.mkdir(parents=True, exist_ok=True)
            shutil.copytree(camera.base_path, new_path)
