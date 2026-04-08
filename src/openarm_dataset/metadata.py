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
from collections.abc import Mapping
import copy
import os
import json
import yaml


class Metadata:
    """Metadata for OpenArm Dataset."""

    def __init__(self, path: str | os.PathLike):
        """Initialize Metadata."""
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
    def episodes(self) -> list[dict]:
        """Get episodes."""
        return self.data.get("episodes", [])

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
        openarm_version = equipment["leader"]["arms"]["right_arm"]["hardware_version"]
        equipment["embodiments"] = {
            "arms": {
                "id": "OpenArm",
                "version": openarm_version,
            },
        }
        del equipment["leader"]
        del equipment["follower"]
        cameras = {}
        for camera_name in equipment.pop("cameras"):
            cameras[camera_name.removeprefix("cam_")] = {}
        equipment["perceptions"] = {
            "cameras": cameras,
        }
        del equipment["sensors"]
        return equipment


class Equipment:
    """Metadata for equipment."""

    def __init__(self, data: dict):
        """Initialize Equipment."""
        self._data = data
        self.embodiments = Embodiments(self._data["embodiments"])
        self.perceptions = Perceptions(self._data["perceptions"])

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
        elif id_ == "Ball Screw Lifter":
            return BallScrewLifter(name, data)
        else:
            raise ValueError(f"Invalid embodiment id: {id_}")


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


class BallScrewLifter(Embodiment):
    """Metadata for ball screw lifter as embodiment."""

    def __init__(self, name: str, data: dict):
        """Initialize BallScrewLifter."""
        super().__init__(name, data)
        self.attributes = ("qpos",)
        self.joints = ("position",)


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
