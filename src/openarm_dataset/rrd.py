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

"""Convert OpenArm Dataset to rerun.io RRD file.

* Use `/` to separate entity path parts.
  * See also: https://rerun.io/docs/concepts/logging-and-ingestion/entity-path
* Use a blueprint to set up the visualization layout
  * See also: https://rerun.io/docs/howto/visualization/build-a-blueprint-programmatically
* Use `rr.send_columns()` to record the data in bulk.
  * See also: https://rerun.io/docs/howto/logging-and-ingestion/send-columns
* Record camera images as MP4 (requires `ffmpeg` on PATH). This will make the RRD file size smaller.
  * See also: https://rerun.io/docs/reference/types/archetypes/video_frame_reference
"""

import os
import tempfile
from pathlib import Path

import numpy as np
import rerun as rr
import rerun.blueprint as rrb

from .dataset import Dataset
from .ffmpeg import encode_mp4


def _entity(episode: dict, *parts: str) -> str:
    return "/".join((f"ep{episode['id']}", *parts))


def _time_series_views(
    dataset: Dataset, episode: dict, category: str
) -> list[rrb.TimeSeriesView]:
    return [
        rrb.TimeSeriesView(
            origin=_entity(episode, category, attribute["key"]),
            name=f"{category}/{attribute['key']}",
        )
        for attribute in dataset.get_embodiment_attributes(category, episode)
    ]


def _build_blueprint(dataset: Dataset) -> rrb.Tabs:
    tabs = []
    for episode in dataset.meta.episodes:
        camera_views = [
            rrb.Spatial2DView(
                origin=_entity(episode, "camera", name),
                name=f"camera/{name}",
            )
            for name in dataset.camera_names
        ]
        action_views = _time_series_views(dataset, episode, "action")
        obs_views = _time_series_views(dataset, episode, "obs")
        tabs.append(
            rrb.Horizontal(
                # https://ref.rerun.io/docs/python/0.33.0/blueprint/#rerun.blueprint.Horizontal
                # `Horizontal` splits the tab into side-by-side columns.
                # `column_shares=[0.3, 0.35, 0.35]` sets each column's relative width.
                # cameras (0.3) / action (0.35) / obs (0.35) columns.
                rrb.Vertical(*camera_views),
                rrb.Vertical(*action_views),
                rrb.Vertical(*obs_views),
                column_shares=[0.3, 0.35, 0.35],
                name=f"ep{episode['id']}",
            )
        )
    return rrb.Tabs(*tabs)


def _log_embodiments(
    rec: rr.RecordingStream,
    dataset: Dataset,
    category: str,
    episode: dict,
    samples: list,
    timestamps,
) -> None:
    for attribute in dataset.get_embodiment_attributes(category, episode):
        key = attribute["key"]
        joints = attribute["embodiment"].joints
        values = np.array([sample[category][key] for sample in samples])
        for i, joint in enumerate(joints):
            rr.send_columns(
                _entity(episode, category, key, joint),
                indexes=[rr.TimeColumn("timestamp", timestamp=timestamps)],
                columns=rr.Scalars.columns(scalars=values[:, i]),
                recording=rec,
            )


def _log_cameras(
    rec: rr.RecordingStream,
    dataset: Dataset,
    episode: dict,
    samples: list,
    timestamps,
    fps: int,
) -> None:
    for name in dataset.camera_names:
        entity = _entity(episode, "camera", name)
        with tempfile.TemporaryDirectory() as temp_dir:
            # Encode the sampled frames into an MP4 file and embed it.
            # This keeps the RRD file small.
            video_path = Path(temp_dir) / f"{name}.mp4"
            encode_mp4(
                [sample.cameras[name] for sample in samples],
                fps,
                video_path,
            )

            # Log the encoded video once as a static asset (no timeline).
            video = rr.AssetVideo(path=video_path)
            rr.log(entity, video, static=True, recording=rec)
            # Reference a frame of that video at each timestamp.
            rr.send_columns(
                entity,
                indexes=[rr.TimeColumn("timestamp", timestamp=timestamps)],
                columns=rr.VideoFrameReference.columns_nanos(
                    video.read_frame_timestamps_nanos()
                ),
                recording=rec,
            )


def _log_episodes(rec: rr.RecordingStream, dataset: Dataset, fps: int) -> None:
    for episode in dataset.meta.episodes:
        samples = dataset.sample(hz=fps, episode=episode)
        if not samples:
            continue
        timestamps = [sample.timestamp for sample in samples]
        _log_embodiments(rec, dataset, "action", episode, samples, timestamps)
        _log_embodiments(rec, dataset, "obs", episode, samples, timestamps)
        _log_cameras(rec, dataset, episode, samples, timestamps, fps)


def to_rrd(
    dataset: Dataset,
    output: str | os.PathLike,
    application_id: str = "openarm_dataset",
    fps: int = 30,
) -> None:
    """Convert OpenArm Dataset to rerun.io RRD file."""
    rec = rr.RecordingStream(application_id=application_id)
    rec.save(str(output), default_blueprint=_build_blueprint(dataset))

    _log_episodes(rec, dataset, fps)
