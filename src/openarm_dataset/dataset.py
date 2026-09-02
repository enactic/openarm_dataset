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
import warnings

import numpy as np
import pyarrow.parquet as pq
import pandas as pd
import scipy.signal as signal

from .camera import Camera
from .metadata import Metadata, OpenArm
from .sampler import Sampler, Sample
from .validator import Validator

# Attributes that may appear as columns in a state.parquet file.
STATE_ATTRIBUTES = ("qpos", "qvel", "qtorque", "pose")

# Recorded attributes that describe an arm configuration.
ARM_STATE_ATTRIBUTES = ("qpos", "pose")

# Representations an arm state can be requested in. "qpos" and "pose" are
# recorded attributes (converted with IK/FK when the other one was recorded);
# "rot6d" is always derived from the pose.
STATE_REPRESENTATIONS = ("qpos", "pose", "rot6d")

POSE_JOINTS = (
    "x",
    "y",
    "z",
    "qw",
    "qx",
    "qy",
    "qz",
    "gripper",
)

# Position, the first two rotation matrix columns. 6D rotation and the gripper.
ROT6D_JOINTS = (
    "x",
    "y",
    "z",
    "r11",
    "r21",
    "r31",
    "r12",
    "r22",
    "r32",
    "gripper",
)


def _pose_to_rot6d(pose: np.ndarray) -> np.ndarray:
    """Convert poses ([N, 8]) to the 6D rotation representation ([N, 10])."""
    pose = np.asarray(pose, dtype=np.float32)
    quat = pose[:, 3:7].astype(np.float64)
    norm = np.linalg.norm(quat, axis=1, keepdims=True)
    # Propagate NaN for zero-norm (corrupt) quaternions instead of masking
    # them with a plausible-looking rotation.
    quat = np.divide(quat, norm, out=np.full_like(quat, np.nan), where=norm > 0)
    w, x, y, z = quat[:, 0], quat[:, 1], quat[:, 2], quat[:, 3]
    rot6d = np.empty((len(pose), 10), dtype=np.float32)
    rot6d[:, :3] = pose[:, :3]  # Position
    rot6d[:, 3] = 1 - 2 * (y * y + z * z)
    rot6d[:, 4] = 2 * (x * y + w * z)
    rot6d[:, 5] = 2 * (x * z - w * y)
    rot6d[:, 6] = 2 * (x * y - w * z)
    rot6d[:, 7] = 1 - 2 * (x * x + z * z)
    rot6d[:, 8] = 2 * (y * z + w * x)
    rot6d[:, 9] = pose[:, 7]  # Gripper
    return rot6d


def _renormalize_orientation(df: pd.DataFrame, attribute: str) -> pd.DataFrame:
    """Project smoothed orientation columns back onto valid rotations.

    Component-wise smoothing leaves quaternions slightly off unit norm and
    rot6d columns slightly non-orthonormal; renormalize (and Gram-Schmidt
    for rot6d) so downstream consumers see valid rotations.
    """
    if attribute == "pose":
        columns = list(POSE_JOINTS[3:7])
        quat = df[columns].to_numpy()
        norm = np.linalg.norm(quat, axis=1, keepdims=True)
        df[columns] = np.divide(quat, norm, out=quat.copy(), where=norm > 0)
    elif attribute == "rot6d":
        a_columns = list(ROT6D_JOINTS[3:6])
        b_columns = list(ROT6D_JOINTS[6:9])
        a = df[a_columns].to_numpy()
        b = df[b_columns].to_numpy()
        norm_a = np.linalg.norm(a, axis=1, keepdims=True)
        a = np.divide(a, norm_a, out=a.copy(), where=norm_a > 0)
        b = b - np.sum(a * b, axis=1, keepdims=True) * a
        norm_b = np.linalg.norm(b, axis=1, keepdims=True)
        b = np.divide(b, norm_b, out=b.copy(), where=norm_b > 0)
        df[a_columns] = a
        df[b_columns] = b
    return df


class Dataset:
    """OpenArm Dataset."""

    def __init__(
        self,
        path: str | os.PathLike,
        meta: Metadata = None,
        camera_names: list[str] = None,
        kinematics: dict = None,
    ):
        """Initialize Dataset.

        Args:
            path: Path of the dataset.
            meta: Metadata of the dataset. Uses the metadata stored in the
                dataset if None.
            camera_names: Names of the camera to use. Uses all cameras in the
                dataset if None.
            kinematics: Mapping of arm component ("right"/"left") to a
                kinematics engine (``kinematics.SideKinematics``) used for
                on-the-fly qpos/pose conversion. Built lazily with
                ``kinematics.create_engines()`` if None.

        """
        self.root_path = Path(path)
        self.meta = Metadata(self.root_path / "metadata.yaml") if meta is None else meta
        self._camera_names = camera_names
        self._smoothing_cutoff = None
        self._kinematics = kinematics

    def set_smoothing(self, cutoff: float):
        """Set smoothing."""
        self._smoothing_cutoff = cutoff

    def validate(
        self,
        on_error=None,
        update_metadata: bool = False,
        qpos_jump_threshold: float | None = None,
        qpos_absmax: float | None = None,
        min_duration: float | None = None,
    ) -> bool:
        """Validate this dataset.

        Args:
            on_error: Optional callable that is called with an error message
                string for each validation error found. If ``None``, errors
                are not reported.
            update_metadata: If ``True``, the validation result is recorded
                in the dataset metadata as a boolean ``valid`` flag per
                episode. Ignored for unversioned datasets.
            qpos_jump_threshold: If set, flag qpos frame-to-frame deltas above
                this value (radians) as abrupt jumps.
            qpos_absmax: If set, flag qpos values whose absolute value
                exceeds this threshold (radians).
            min_duration: If set, flag episodes whose duration is shorter
                than this value (seconds).

        Returns:
            ``True`` if the dataset is valid, ``False`` otherwise.

        """
        # The valid flag write feature is disabled for unversioned datasets.
        if self.meta.version is None:
            update_metadata = False
        validator = Validator(
            self,
            on_error=on_error,
            update_metadata=update_metadata,
            qpos_jump_threshold=qpos_jump_threshold,
            qpos_absmax=qpos_absmax,
            min_duration=min_duration,
        )
        valid = validator.validate()
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

    @property
    def camera_format(self) -> str:
        """Return the camera format ("dir" or "tar") shared by all cameras.

        Every camera in the dataset is expected to use the same format.

        Raises:
            ValueError: If cameras use a mix of "dir" and "tar".

        """
        formats = set()
        for episode in self.meta.episodes:
            for name in self.camera_names:
                formats.add(self.load_camera(name, episode).format)
        if len(formats) > 1:
            raise ValueError(f"Inconsistent camera formats: {sorted(formats)}")
        return formats.pop()

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
        state: str = None,
    ) -> dict[str, pd.DataFrame]:
        """Load obs data.

        Args:
            episode: Episode to load.
            use_unixtime: If True, the DataFrame index is returned as Unix time
                (float64) instead of datetime64[ns].
            cutoff: If not None, smoothing is applied using this value.
            state: If not None, return arm states in this representation
                ("qpos", "pose" or "rot6d"), converting the recorded data
                on the fly (qpos to pose via FK, pose to qpos via IK) when
                the requested representation was not recorded.

        Returns:
            Dictionary mapping names to DataFrames. Keys reflect the
            attributes actually recorded (any of qpos/qvel/qtorque/pose
            per arm component); with ``state``, arm state keys use the
            requested representation instead.

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
            state=state,
        )

    def load_action(
        self,
        episode: dict,
        use_unixtime: bool = False,
        cutoff: float = None,
        state: str = None,
    ) -> dict[str, pd.DataFrame]:
        """Load action data.

        Args:
            episode: Episode to load.
            use_unixtime: If True, the DataFrame index is returned as Unix time
                (float64) instead of datetime64[ns].
            cutoff: If not None, smoothing is applied using this value.
            state: If not None, return arm states in this representation
                ("qpos", "pose" or "rot6d"), converting the recorded data
                on the fly (qpos to pose via FK, pose to qpos via IK) when
                the requested representation was not recorded.

        Returns:
            Dictionary mapping names to DataFrames. Keys reflect the
            attributes actually recorded (qpos for joint-space actions,
            pose for Cartesian actions); with ``state``, arm state keys
            use the requested representation instead.

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
            state=state,
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
        state: str = None,
    ) -> list[Sample]:
        """Sample the all modalities data to the specified hz.

        Args:
            episode: Episode to sample.
            hz: Sampling hz.
            state: If not None, return arm states in this representation
                ("qpos", "pose" or "rot6d"), converted on the fly when the
                requested representation was not recorded.

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
        return list(sampler.sample(self, episode, hz, state=state))

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
                        # state.parquet is self-describing: one list column
                        # per attribute.
                        for attr_name in pq.read_schema(state_path).names:
                            if attr_name == "timestamp":
                                continue
                            if attr_name not in STATE_ATTRIBUTES:
                                raise ValueError(
                                    f"Unknown attribute {attr_name!r} in {state_path}"
                                )
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
                        # The recorder only writes attributes that were
                        # actually recorded.
                        if not path.exists():
                            continue
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
                    path = base_path / f"{attribute}.parquet"
                    # The recorder only writes attributes that were
                    # actually recorded.
                    if not path.exists():
                        continue
                    attributes.append(
                        {
                            "key": key,
                            "embodiment": embodiment,
                            "component": None,
                            "name": attribute,
                            "path": path,
                        }
                    )
        return attributes

    def _load_embodiment_values(
        self,
        type_: str,
        episode: dict,
        use_unixtime: bool = False,
        cutoff: float = None,
        state: str = None,
    ) -> dict[str, pd.DataFrame]:
        if state is not None and state not in STATE_REPRESENTATIONS:
            raise ValueError(
                f"Unknown state representation {state!r}: "
                f"expected one of {STATE_REPRESENTATIONS}"
            )
        values = {}
        arm_states = {}
        for attribute in self.get_embodiment_attributes(type_, episode):
            df = self.load_embodiment_value(attribute, use_unixtime=use_unixtime)
            if (
                state is not None
                and isinstance(attribute["embodiment"], OpenArm)
                and attribute["name"] in ARM_STATE_ATTRIBUTES
            ):
                group = arm_states.setdefault(
                    (attribute["embodiment"].name, attribute["component"]),
                    {"embodiment": attribute["embodiment"], "frames": {}},
                )
                group["frames"][attribute["name"]] = df
            else:
                values[attribute["key"]] = df
        for (name, component), group in arm_states.items():
            values[f"{name}/{component}/{state}"] = self._represent_arm_state(
                group["frames"],
                state,
                group["embodiment"],
                type_,
                episode,
                component,
                use_unixtime=use_unixtime,
            )
        if cutoff is not None:
            values = {
                key: _renormalize_orientation(
                    self._apply_smoothing(df, cutoff=cutoff),
                    key.rsplit("/", 1)[-1],
                )
                for key, df in values.items()
            }
        return values

    def _represent_arm_state(
        self,
        frames: dict[str, pd.DataFrame],
        state: str,
        embodiment,
        type_: str,
        episode: dict,
        component: str,
        use_unixtime: bool = False,
    ) -> pd.DataFrame:
        """Return one arm's state in the requested representation.

        Recorded representations are returned as-is; missing ones are derived
        on the fly (qpos to pose via FK, pose to qpos via IK, and rot6d from
        the pose).
        """
        if state in frames:
            return frames[state]
        if state == "qpos":
            pose = frames["pose"]
            seed_qpos = self._ik_seed(
                episode, embodiment.name, component, pose.index, use_unixtime
            )
            qpos, failures, unconverged = self._arm_engine(component).pose_to_qpos(
                pose.to_numpy(dtype=np.float32), seed_qpos=seed_qpos
            )
            if failures or unconverged:
                warnings.warn(
                    f"IK for {type_} {embodiment.name}/{component} in episode "
                    f"{episode['id']}: {failures} sample(s) failed (previous "
                    f"solution reused), {unconverged} sample(s) did not reach "
                    "the convergence tolerances (best effort kept)."
                )
            return pd.DataFrame(qpos, index=pose.index, columns=list(embodiment.joints))
        # "pose" and "rot6d" need a pose trajectory.
        if "pose" in frames:
            index = frames["pose"].index
            pose = frames["pose"].to_numpy(dtype=np.float32)
        else:
            qpos = frames["qpos"]
            index = qpos.index
            pose = self._arm_engine(component).qpos_to_pose(
                qpos.to_numpy(dtype=np.float32)
            )
        if state == "pose":
            return pd.DataFrame(pose, index=index, columns=list(POSE_JOINTS))
        return pd.DataFrame(
            _pose_to_rot6d(pose), index=index, columns=list(ROT6D_JOINTS)
        )

    def _arm_engine(self, component: str):
        if self._kinematics is None:
            from .kinematics import create_engines

            self._kinematics = create_engines()
        engine = self._kinematics.get(component)
        if engine is None:
            raise ValueError(f"No kinematics engine for arm component {component!r}")
        return engine

    def _ik_seed(
        self,
        episode: dict,
        name: str,
        component: str,
        pose_index: pd.Index,
        use_unixtime: bool,
    ) -> np.ndarray | None:
        """Return the recorded obs qpos row nearest the first pose timestamp."""
        if len(pose_index) == 0:
            return None
        for attribute in self.get_embodiment_attributes("obs", episode):
            if attribute["key"] != f"{name}/{component}/qpos":
                continue
            obs = self.load_embodiment_value(attribute, use_unixtime=use_unixtime)
            if obs.empty:
                return None
            nearest = int(np.abs(obs.index - pose_index[0]).argmin())
            return obs.to_numpy(dtype=np.float32)[nearest]
        return None

    def load_embodiment_value(
        self,
        attribute: dict,
        use_unixtime: bool = False,
    ) -> pd.DataFrame:
        """Load one embodiment attribute as recorded.

        Args:
            attribute: An entry returned by ``get_embodiment_attributes``.
            use_unixtime: If True, the DataFrame index is returned as Unix time
                (float64) instead of datetime64[ns].

        Returns:
            DataFrame indexed by timestamp with one column per joint. No
            smoothing or state conversion is applied.

        """
        df = pd.read_parquet(attribute["path"])
        if attribute["path"].name == "state.parquet":
            column_name = attribute["name"]
            drop_columns = [c for c in df.columns if c != "timestamp"]
        elif "positions" in df:
            # No version and 0.1.0 use "positions"
            column_name = "positions"
            drop_columns = ["positions"]
        else:
            column_name = "value"
            drop_columns = ["value"]
        if attribute["name"] == "pose":
            joints = POSE_JOINTS
        else:
            joints = attribute["embodiment"].joints
        # columns= keeps the joint columns for an empty frame too.
        df[list(joints)] = pd.DataFrame(
            df[column_name].tolist(),
            index=df.index,
            columns=list(joints),
        )
        df = df.drop(columns=drop_columns)
        if use_unixtime:
            df["timestamp"] = df["timestamp"].astype("int64") / 1e9
        df = df.set_index("timestamp")
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

    def _write(
        self,
        output: str | os.PathLike,
        camera_format: str = "dir",
        valid_only: bool = False,
    ):
        """Write this dataset as the latest OpenArm dataset format."""
        output = Path(output)
        self.meta.write(output, valid_only=valid_only)
        self._write_data(output, camera_format=camera_format, valid_only=valid_only)

    def write(self, output: str | os.PathLike, format: str | None = None, **options):
        """Write this dataset in the specified format."""
        if format is None or format == "openarm":
            return self._write(output, **options)
        elif format == "lerobot_v2.1":
            from .lerobot_v21 import to_lerobotv21

            return to_lerobotv21(self, output, **options)
        elif format == "lerobot_v3.0":
            from .lerobot_v30 import to_lerobotv30

            return to_lerobotv30(self, output, **options)
        elif format == "gr00t":
            from .lerobot_v21 import to_gr00t

            return to_gr00t(self, output, **options)
        elif format == "robot_eval_logger":
            from .robot_eval_logger import to_robot_eval_logger

            return to_robot_eval_logger(self, output, **options)
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

    def _write_data(
        self, output: Path, camera_format: str = "dir", valid_only: bool = False
    ):
        for episode in self.meta.episodes:
            if valid_only and not episode.valid():
                continue
            self._write_episode(output, episode, camera_format=camera_format)

    def _write_episode(self, output: Path, episode: dict, camera_format: str = "dir"):
        self._write_embodiment_data(output, episode)
        self._write_camera_data(output, episode, camera_format=camera_format)

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
                    # 0.4.0 stores component data as self-describing
                    # state.parquet. Legacy per-attribute sources are
                    # assumed to have exactly one attribute per component
                    # (currently only "qpos"); a second attribute would
                    # collide on the same state.parquet and silently lose
                    # data, so fail loudly instead of merging columns.
                    new_path = base_path / component / "state.parquet"
                    if new_path in written_state_paths:
                        raise ValueError(
                            f"Cannot convert multiple attributes for component "
                            f"{component!r} into a single {new_path}: attribute "
                            f"{name!r} would overwrite data already written for "
                            f"this component."
                        )
                    new_path.parent.mkdir(parents=True, exist_ok=True)
                    df = pd.read_parquet(attribute["path"])
                    # No version and 0.1.0 use "positions"
                    if "positions" in df:
                        df["value"] = df["positions"]
                        df = df.drop(columns=["positions"])
                    df = df.rename(columns={"value": name})
                    df.to_parquet(new_path)
                    written_state_paths.add(new_path)
                    continue
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

    def _write_camera_data(
        self, output: os.PathLike, episode: dict, camera_format: str = "dir"
    ):
        base_path = output / "episodes" / episode["id"]
        for name, camera in self.load_cameras(episode).items():
            if self.meta.version is None:
                if name == "left_wrist":
                    name = "wrist_left"
                elif name == "right_wrist":
                    name = "wrist_right"
            cameras_dir = base_path / "cameras"
            cameras_dir.mkdir(parents=True, exist_ok=True)
            camera.write(cameras_dir / name, camera_format)
