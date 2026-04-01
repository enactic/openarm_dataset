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
import os
import yaml


class Metadata:
    """Metadata for OpenArm Dataset."""

    def __init__(self, path: str | os.PathLike):
        """Initialize Metadata."""
        self.data = self._load_yaml(path)

    def _load_yaml(self, path: str | os.PathLike) -> dict:
        with open(path) as f:
            return yaml.safe_load(f)

    @property
    def version(self) -> str:
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
        return Equipment(self.data["equipment"])


class Equipment:
    """Metadata for Equipment."""

    def __init__(self, config: dict):
        """Initialize Equipment."""
        self.config = config
        self.embodiments = Embodiments(self.config["embodiments"])
        self.perceptions = Perceptions(self.config["perceptions"])

    @property
    def id(self) -> str:
        """Get id."""
        return self.config["id"]

    @property
    def version(self) -> str:
        """Get version."""
        return self.config["version"]


class Embodiments(Mapping):
    """Metadata for Embodiments."""

    def __init__(self, config: dict):
        """Initialize Embodiments."""
        self.config = config
        self.embodiments = {
            name: self._build_embodiment(name, embodiment_config)
            for name, embodiment_config in self.config.items()
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

    def _build_embodiment(self, name: str, config: dict) -> Embodiment:
        id_ = config["id"]
        if id_ == "OpenArm":
            return OpenArm(name, config)
        elif id_ == "Ball Screw Lifter":
            return BallScrewLifter(name, config)
        else:
            raise ValueError(f"Invalid embodiment id: {id_}")


class Perceptions:
    """Metadata for Perceptions."""

    def __init__(self, config: dict):
        """Initialize Perceptions."""
        self.config = config
        self.cameras = {
            name: Camera(name, config)
            for name, config in self.config["cameras"].items()
        }


class Embodiment:
    """Metadata for Embodiment."""

    def __init__(self, name: str, config: dict):
        """Initialize Embodiment."""
        self.name = name
        self.config = config
        self.components: tuple[str, ...] = ()
        self.attributes: tuple[str, ...] = ()
        self.joints: tuple[str, ...] = ()

    @property
    def id(self) -> str:
        """Get id."""
        return self.config["id"]

    @property
    def version(self) -> str:
        """Get version."""
        return self.config["version"]


class OpenArm(Embodiment):
    """Metadata for OpenArm as Embodiment."""

    def __init__(self, name: str, config: dict):
        """Initialize OpenArm."""
        super().__init__(name, config)
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
    """Metadata for BallScrewLifter as Embodiment."""

    def __init__(self, name: str, config: dict):
        """Initialize BallScrewLifter."""
        super().__init__(name, config)
        self.attributes = ("qpos",)
        self.joints = ("position",)


class Camera:
    """Metadata for Camera."""

    def __init__(self, name: str, config: dict):
        """Initialize Camera."""
        self.name = name
        self.config = config
