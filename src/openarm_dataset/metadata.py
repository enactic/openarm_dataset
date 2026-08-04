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

"""Metadata for OpenArm Dataset."""

from __future__ import annotations
from collections.abc import Mapping, MutableMapping
import copy
import os
import pathlib
import json
import yaml


class Episode(MutableMapping):
    """An episode in the dataset metadata.

    A view of an episode dict in Metadata; changes are written through
    to the underlying dict.
    """

    def __init__(self, data: dict):
        """Initialize Episode."""
        self._data = data

    def __getitem__(self, key):
        """Return data for the key."""
        return self._data[key]

    def __setitem__(self, key, value):
        """Set data for the key."""
        self._data[key] = value

    def __delitem__(self, key):
        """Delete data for the key."""
        del self._data[key]

    def __iter__(self):
        """Return iterator."""
        return iter(self._data)

    def __len__(self):
        """Return number of items."""
        return len(self._data)

    def valid(self) -> bool:
        """Return whether this episode is not marked invalid.

        Episodes without a ``valid`` flag are treated as valid.
        """
        return self._data.get("valid", True)


class Metadata:
    """Metadata for OpenArm Dataset."""

    def __init__(self, path: str | os.PathLike):
        """Initialize Metadata."""
        self.path = pathlib.Path(path)
        self.data = self._load_yaml(path)
        # Unversioned dataset. This is for backward compatibility.
        if "meta" in self.data:
            self.data = self.data["meta"]
            episodes_path = os.path.join(os.path.dirname(path), "episodes.jsonl")
            episodes = []
            with open(episodes_path) as f:
                for line in f:
                    episodes.append(json.loads(line))
            self.data["episodes"] = episodes

    def _load_yaml(self, path: str | os.PathLike) -> dict:
        with open(path) as f:
            return yaml.safe_load(f)

    @property
    def version(self) -> str | None:
        """Get version."""
        return self.data.get("version")

    @property
    def operator(self) -> str:
        """Get operator."""
        return self.data.get("operator")

    @property
    def operation_type(self) -> str:
        """Get operation type."""
        return self.data.get("operation_type", "teleop")

    @property
    def location(self) -> str:
        """Get location."""
        return self.data.get("location")

    @property
    def tasks(self) -> list[dict]:
        """Get tasks."""
        return self.data.get("tasks")

    @property
    def episodes(self) -> list[Episode]:
        """Get episodes."""
        return [Episode(episode) for episode in self.data.get("episodes", [])]

    @property
    def num_episodes(self) -> int:
        """Get number of episodes."""
        return len(self.episodes)

    @property
    def equipment(self) -> Equipment:
        """Get equipment."""
        # Unversioned dataset. This is for backward compatibility.
        if self.version is None:
            return Equipment(self._convert_unversioned_equipment())
        else:
            return Equipment(self.data["equipment"])

    @property
    def frequencies(self) -> Frequencies:
        """Get frequencies."""
        return Frequencies(self.data.get("frequencies", {}))

    def _convert_unversioned_equipment(self):
        equipment = copy.deepcopy(self.data["equipment"])
        equipment["id"] = equipment.pop("equipment_id")
        equipment["version"] = equipment.pop("equipment_version")
        # Same key, different field: `leader` here describes the leader arms,
        # not the teleoperation device of v0.4.0's `equipment.leader`.
        openarm_version = equipment["leader"]["arms"]["right_arm"]["hardware_version"]
        equipment["embodiments"] = {
            "arms": {
                "id": "OpenArm",
                "version": openarm_version,
            },
        }
        cameras = {}
        for camera_name in equipment["follower"]["cameras"]:
            cameras[camera_name.removeprefix("cam_")] = {}
        equipment["perceptions"] = {
            "cameras": cameras,
        }
        del equipment["leader"]
        del equipment["follower"]
        return equipment

    def write(self, output: str | os.PathLike, valid_only: bool = False):
        """Write this metadata as the latest OpenArm dataset format.

        Args:
            output: Output directory.
            valid_only: If True, episodes marked invalid (``valid: false``)
                are excluded.

        """
        output = pathlib.Path(output)
        data = copy.deepcopy(self.data)
        latest_version = "0.4.0"
        data["version"] = latest_version
        if valid_only:
            data["episodes"] = [
                episode
                for episode in data.get("episodes", [])
                if Episode(episode).valid()
            ]
        if self.version is None:
            data["equipment"] = self._convert_unversioned_equipment()
        if self.version is None or self.version == "0.1.0":
            cameras = data["equipment"]["perceptions"]["cameras"]
            if "left_wrist" in cameras:
                cameras["wrist_left"] = cameras.pop("left_wrist")
            if "right_wrist" in cameras:
                cameras["wrist_right"] = cameras.pop("right_wrist")
        output.mkdir(parents=True, exist_ok=True)
        with open(output / "metadata.yaml", "w") as f:
            yaml.safe_dump(data, f)


class Equipment:
    """Metadata for equipment."""

    def __init__(self, data: dict):
        """Initialize Equipment."""
        self._data = data
        self.embodiments = Embodiments(self._data["embodiments"])
        self.perceptions = Perceptions(self._data["perceptions"])
        # Often absent, so an empty Leader rather than None: callers iterate
        # without a guard.
        self.leader = Leader(self._data.get("leader") or {})

    @property
    def id(self) -> str:
        """Get id."""
        return self._data["id"]

    @property
    def version(self) -> str:
        """Get version."""
        return self._data["version"]


class Embodiments(Mapping):
    """Metadata for embodiments."""

    def __init__(self, data: dict):
        """Initialize Embodiments."""
        self._data = data
        self.embodiments = {
            name: self._build_embodiment(name, embodiment_data)
            for name, embodiment_data in self._data.items()
        }

    def __getitem__(self, key):
        """Return data for the key."""
        return self.embodiments[key]

    def __iter__(self):
        """Return iterator."""
        return iter(self.embodiments)

    def __len__(self):
        """Return number of Embodiments."""
        return len(self.embodiments)

    def _build_embodiment(self, name: str, data: dict) -> Embodiment:
        id_ = data["id"]
        if id_ == "OpenArm":
            return OpenArm(name, data)
        elif id_ == "OpenArmCellLifter":
            return OpenArmCellLifter(name, data)
        else:
            raise ValueError(f"Invalid embodiment id: {id_}")


class Leader(Mapping):
    """Metadata for the teleoperation leader devices.

    The devices the operator drove, keyed by kind, as written under
    ``equipment.leader``::

        equipment:
          leader:
            ker:
              id: OpenArmKER
              firmware_version: "1.2.3"
              hardware_version: "1.0"

    The input side of a teleoperation session, as distinct from
    ``embodiments`` (the arms driven) and ``operator`` (the person driving).
    Empty when nothing recorded a leader. Unlike ``Embodiments``, an
    unrecognized ``id`` is not an error: a device is a label and its versions.
    """

    def __init__(self, data: dict):
        """Initialize Leader."""
        self._data = data
        self.devices = {
            kind: LeaderDevice(kind, device_data)
            for kind, device_data in self._data.items()
        }

    def __getitem__(self, key):
        """Return the device for the key."""
        return self.devices[key]

    def __iter__(self):
        """Return iterator."""
        return iter(self.devices)

    def __len__(self):
        """Return number of devices."""
        return len(self.devices)


class LeaderDevice:
    """Metadata for one teleoperation leader device."""

    def __init__(self, kind: str, data: dict):
        """Initialize LeaderDevice."""
        self.kind = kind
        self._data = data

    @property
    def id(self) -> str | None:
        """Get id, e.g. ``"OpenArmKER"``."""
        return self._data.get("id")

    @property
    def firmware_version(self) -> str | None:
        """Get firmware version, if the device reported one."""
        return self._data.get("firmware_version")

    @property
    def hardware_version(self) -> str | None:
        """Get hardware version, if the device reported one."""
        return self._data.get("hardware_version")


class Perceptions:
    """Metadata for perceptions."""

    def __init__(self, data: dict):
        """Initialize Perceptions."""
        self._data = data
        self.cameras = {
            name: Camera(name, camera_data)
            for name, camera_data in self._data["cameras"].items()
        }


class Embodiment:
    """Metadata for embodiment."""

    def __init__(self, name: str, data: dict):
        """Initialize Embodiment."""
        self.name = name
        self._data = data
        self.components: tuple[str, ...] = ()
        self.attributes: tuple[str, ...] = ()
        self.joints: tuple[str, ...] = ()

    @property
    def id(self) -> str:
        """Get id."""
        return self._data["id"]

    @property
    def version(self) -> str:
        """Get version."""
        return self._data["version"]


class OpenArm(Embodiment):
    """Metadata for OpenArm as embodiment."""

    def __init__(self, name: str, data: dict):
        """Initialize OpenArm."""
        super().__init__(name, data)
        self.components = ("right", "left")
        self.attributes = ("qpos",)
        self.joints = (
            "joint1",
            "joint2",
            "joint3",
            "joint4",
            "joint5",
            "joint6",
            "joint7",
            "gripper",
        )


class OpenArmCellLifter(Embodiment):
    """Metadata for OpenArm Cell Lifter as embodiment."""

    def __init__(self, name: str, data: dict):
        """Initialize OpenArmCellLifter."""
        super().__init__(name, data)
        self.attributes = ("elevation",)
        self.joints = ("elevation",)


class Camera:
    """Metadata for camera."""

    def __init__(self, name: str, data: dict):
        """Initialize Camera."""
        self.name = name
        self._data = data


class Frequencies:
    """Metadata for frequencies."""

    def __init__(self, data: dict):
        """Initialize Frequencies."""
        self._data = data
        self.action = self._data.get("action", {})
        self.cameras = self._data.get("cameras", {})
        self.obs = self._data.get("obs", {})
