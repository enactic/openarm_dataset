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

"""Convert OpenArm Dataset to rerun.io RRD file."""

import os
from pathlib import Path

import pandas as pd
import numpy as np
import rerun as rr
import rerun.blueprint as rrb

from .camera import Camera
from .dataset import Dataset


def _entity(episode_id: str, *parts: str) -> str:
    return f"ep{episode_id}/{'/'.join(parts)}"


def _build_blueprint(dataset: Dataset) -> rrb.Tabs:
    sides = []
    for embodiment in dataset.meta.equipment.embodiments.values():
        for component in embodiment.components:
            if component not in sides:
                sides.append(component)

    tabs = []
    for episode_index in range(dataset.num_episodes):
        episode_id = dataset.meta.episodes[episode_index]["id"]
        camera_views = [
            rrb.Spatial2DView(
                origin=_entity(episode_id, "cameras", name),
                name=name,
            )
            for name in dataset.camera_names
        ]
        action_views = [
            rrb.TimeSeriesView(
                origin=_entity(episode_id, "action", side),
                name=f"action/{side}",
            )
            for side in sides
        ]
        obs_views = [
            rrb.TimeSeriesView(
                origin=_entity(episode_id, "obs", side),
                name=f"obs/{side}",
            )
            for side in sides
        ]
        tabs.append(
            rrb.Horizontal(
                rrb.Vertical(*camera_views),
                rrb.Vertical(*action_views),
                rrb.Vertical(*obs_views),
                column_shares=[0.3, 0.35, 0.35],
                name=f"ep{episode_id}",
            )
        )

    return rrb.Tabs(*tabs)


def _log_arm_dataframe(rec: rr.RecordingStream, entity: str, df: pd.DataFrame) -> None:
    timestamps = np.array(df.index, dtype="datetime64[ns]")
    # shape: (T, n_joints)
    values = df.to_numpy()
    for i, col in enumerate(df.columns):
        rr.send_columns(
            f"{entity}/{col}",
            indexes=[rr.TimeColumn("timestamp", timestamp=timestamps)],
            columns=rr.Scalars.columns(scalars=values[:, i]),
            recording=rec,
        )


def _log_camera(rec: rr.RecordingStream, entity: str, camera: Camera) -> None:
    for frame in camera.frames():
        rr.set_time(
            "timestamp",
            timestamp=np.datetime64(int(frame.path.stem), "ns"),
            recording=rec,
        )
        rr.log(entity, rr.EncodedImage(path=str(frame.path)), recording=rec)


def _log_episode(rec: rr.RecordingStream, dataset: Dataset, episode_index: int) -> None:
    episode_id = dataset.meta.episodes[episode_index]["id"]
    for category, data in [
        ("action", dataset.load_action(episode_index)),
        ("obs", dataset.load_obs(episode_index)),
    ]:
        for key, df in data.items():
            # key: "arms/right/qpos" → side: "right"
            side = key.split("/")[1]
            entity = _entity(episode_id, category, side)
            _log_arm_dataframe(rec, entity, df)

    for name, camera in dataset.load_cameras(episode_index).items():
        entity = _entity(episode_id, "cameras", name)
        _log_camera(rec, entity, camera)


def _log_episodes(rec: rr.RecordingStream, dataset: Dataset) -> None:
    for episode_index in range(dataset.num_episodes):
        _log_episode(rec, dataset, episode_index)


def to_rrd(
    dataset: Dataset,
    output: str | os.PathLike,
    application_id: str = "openarm_dataset",
) -> Path:
    """Convert OpenArm Dataset to rerun.io RRD file."""
    output_path = Path(output)

    rec = rr.RecordingStream(
        application_id=application_id,
        make_default=False,
    )
    rec.save(str(output_path), default_blueprint=_build_blueprint(dataset))
    _log_episodes(rec, dataset)

    return output_path
