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

"""Validator for OpenArm Dataset."""

from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

from .metadata import Episode


class Validator:
    """Validator for OpenArm Dataset."""

    def __init__(
        self,
        dataset,
        on_error=None,
        update_metadata=False,
        qpos_jump_threshold: float | None = None,
        qpos_absmax: float | None = None,
        min_duration: float | None = None,
        max_duration: float | None = None,
        max_stream_desync: float | None = None,
        max_sample_gap: float | None = None,
    ):
        """Initialize Validator.

        Args:
            dataset: The dataset to validate.
            on_error: Optional callable that is called with an error message
                string for each validation error found. If ``None``, errors
                are not reported.
            update_metadata: If ``True``, the dataset metadata is updated with
                the validation results. If ``False``, the metadata is not updated.
            qpos_jump_threshold: If set, flag qpos frame-to-frame deltas above
                this value (radians) as abrupt jumps.
            qpos_absmax: If set, flag qpos values whose absolute value
                exceeds this threshold (radians).
            min_duration: If set, flag episodes whose duration is shorter
                than this value (seconds).
            max_duration: If set, flag episodes whose duration is longer
                than this value (seconds).
            max_stream_desync: If set, flag episodes whose streams do not all
                cover the same span of time, differing by more than this
                value (seconds).
            max_sample_gap: If set, flag streams that stop recording mid-episode,
                leaving a gap between consecutive samples longer than this
                value (seconds).

        """
        self._dataset = dataset
        self._on_error = on_error
        self._update_metadata = update_metadata
        self._qpos_jump_threshold = qpos_jump_threshold
        self._qpos_absmax = qpos_absmax
        self._min_duration = min_duration
        self._max_duration = max_duration
        self._max_stream_desync = max_stream_desync
        self._max_sample_gap = max_sample_gap

    def validate(self) -> bool:
        """Validate the dataset."""
        valid = True
        for episode in self._dataset.meta.episodes:
            episode_valid = self._validate_episode(episode)
            if self._update_metadata:
                episode["valid"] = episode_valid
            if not episode_valid:
                valid = False
        if self._update_metadata:
            output = self._dataset.meta.path.parent
            self._dataset.meta.write(output)
        return valid

    def _validate_episode(self, episode: Episode) -> bool:
        """Validate the given episode."""
        valid = self._validate_data_presence(episode)
        null_paths = self._collect_null_paths(episode)
        if null_paths:
            valid = False
        # Files with nulls are skipped: their values cannot be compared
        # against a threshold, and they are already reported.
        if not self._validate_qpos(episode, null_paths):
            valid = False
        if not self._validate_duration(episode):
            valid = False
        # Both checks read every stream's timestamps, so they share one read -
        # and neither being asked for means not reading them at all.
        if self._max_stream_desync is not None or self._max_sample_gap is not None:
            streams = self._stream_timestamps(episode)
            if not self._validate_stream_desync(episode, streams):
                valid = False
            if not self._validate_sample_gaps(streams):
                valid = False
        return valid

    def _validate_data_presence(self, episode: Episode) -> bool:
        """Check that the episode recorded any obs and action data at all.

        An episode whose parquet files are missing entirely - a camera-only
        recording, say - has nothing for the other checks to read, so they
        all pass it silently. Presence is checked against what the episode
        itself recorded, not against `equipment.embodiments`: an embodiment
        may be declared and legitimately never recorded.
        """
        valid = True
        for type_name in ("obs", "action"):
            if not self._dataset.get_embodiment_attributes(type_name, episode):
                self._report_error(f"episodes/{episode['id']}: no {type_name} data")
                valid = False
        return valid

    def _report_error(self, message: str):
        if self._on_error is not None:
            self._on_error(message)

    def _relative_path(self, path) -> str:
        return str(path.relative_to(self._dataset.root_path))

    def _collect_null_paths(self, episode: Episode) -> set:
        """Report files that include null values and return their paths."""
        null_paths = set()
        checked_paths = set()
        for type_name in ("obs", "action"):
            for attribute in self._dataset.get_embodiment_attributes(
                type_name, episode
            ):
                path = attribute["path"]
                if path in checked_paths or not path.exists():
                    continue
                checked_paths.add(path)
                if self._has_null(path):
                    self._report_error(
                        f"{self._relative_path(path)}: includes null values"
                    )
                    null_paths.add(path)
        return null_paths

    def _validate_qpos(self, episode: Episode, skipped_paths: set) -> bool:
        """Check qpos values against the absolute and jump thresholds."""
        if self._qpos_absmax is None and self._qpos_jump_threshold is None:
            return True
        valid = True
        for type_name in ("obs", "action"):
            for attribute in self._dataset.get_embodiment_attributes(
                type_name, episode
            ):
                path = attribute["path"]
                if attribute["name"] != "qpos":
                    continue
                if path in skipped_paths or not path.exists():
                    continue
                # Read the recorded values, not the smoothed ones: smoothing
                # is what would hide the anomalies we are looking for.
                values = self._dataset.load_embodiment_value(attribute).to_numpy()
                if self._qpos_absmax is not None and len(values) > 0:
                    absmax = np.abs(values).max()
                    if absmax > self._qpos_absmax:
                        self._report_error(
                            f"{self._relative_path(path)}: "
                            f"qpos absmax={absmax:.4f} > {self._qpos_absmax}"
                        )
                        valid = False
                if self._qpos_jump_threshold is not None and len(values) > 1:
                    deltas = np.abs(np.diff(values, axis=0))
                    count = int(np.count_nonzero(deltas > self._qpos_jump_threshold))
                    if count > 0:
                        self._report_error(
                            f"{self._relative_path(path)}: {count} qpos jump(s) "
                            f"> {self._qpos_jump_threshold} rad "
                            f"(max={deltas.max():.4f})"
                        )
                        valid = False
        return valid

    def _validate_duration(self, episode: Episode) -> bool:
        """Check the episode duration against the duration thresholds."""
        if self._min_duration is None and self._max_duration is None:
            return True
        duration = self._episode_duration(episode)
        if duration is None:
            return True
        valid = True
        if self._min_duration is not None and duration < self._min_duration:
            self._report_error(
                f"episodes/{episode['id']}: "
                f"duration={duration:.2f}s < {self._min_duration}s"
            )
            valid = False
        if self._max_duration is not None and duration > self._max_duration:
            self._report_error(
                f"episodes/{episode['id']}: "
                f"duration={duration:.2f}s > {self._max_duration}s"
            )
            valid = False
        return valid

    def _episode_duration(self, episode: Episode) -> float | None:
        """Return the duration of the longest obs stream in seconds.

        Returns ``None`` if the episode has no obs data to measure.
        """
        durations = []
        checked_paths = set()
        for attribute in self._dataset.get_embodiment_attributes("obs", episode):
            path = attribute["path"]
            if path in checked_paths or not path.exists():
                continue
            checked_paths.add(path)
            timestamps = pq.read_table(path, columns=["timestamp"]).column("timestamp")
            if len(timestamps) < 2:
                durations.append(0.0)
                continue
            timestamps = timestamps.cast(pa.int64())
            durations.append((timestamps[-1].as_py() - timestamps[0].as_py()) / 1e9)
        if not durations:
            return None
        return max(durations)

    def _stream_timestamps(
        self, episode: Episode
    ) -> list[tuple[str, Path, np.ndarray]]:
        """Every stream of the episode as (episode-relative key, path, timestamps).

        One entry per parquet file - a `state.parquet` carries several
        attributes on a single clock, so the file, not the attribute, is the
        stream. Read once here because the two checks below both need it.
        """
        streams = []
        checked_paths = set()
        episode_path = self._dataset.episode_path(episode)
        for type_name in ("obs", "action"):
            for attribute in self._dataset.get_embodiment_attributes(
                type_name, episode
            ):
                path = attribute["path"]
                if path in checked_paths or not path.exists():
                    continue
                checked_paths.add(path)
                timestamps = pq.read_table(path, columns=["timestamp"]).column(
                    "timestamp"
                )
                streams.append(
                    (
                        str(path.relative_to(episode_path)),
                        path,
                        timestamps.cast(pa.int64()).to_numpy(zero_copy_only=False),
                    )
                )
        return streams

    def _validate_stream_desync(
        self, episode: Episode, streams: list[tuple[str, Path, np.ndarray]]
    ) -> bool:
        """Check that every stream of the episode covers the same span of time.

        All of an episode's streams are recorded in one window, so a stream
        that ends early did not finish: the arm dropped its feedback and the
        recorder kept going, leaving cameras and `action` running against an
        `obs` stream that stopped minutes earlier. Nothing else here catches
        it - every file on its own is complete and free of nulls - and the
        duration check makes it worse, since it measures `obs` and so reports
        the truncated stream as a short episode rather than a broken one.
        """
        if self._max_stream_desync is None or len(streams) < 2:
            return True
        spans = [(key, self._span(timestamps)) for key, _, timestamps in streams]
        shortest = min(spans, key=lambda item: item[1])
        longest = max(spans, key=lambda item: item[1])
        desync = longest[1] - shortest[1]
        if desync <= self._max_stream_desync:
            return True
        self._report_error(
            f"episodes/{episode['id']}: stream durations differ by {desync:.2f}s "
            f"> {self._max_stream_desync}s "
            f"({shortest[0]}={shortest[1]:.2f}s, {longest[0]}={longest[1]:.2f}s)"
        )
        return False

    def _validate_sample_gaps(
        self, streams: list[tuple[str, Path, np.ndarray]]
    ) -> bool:
        """Check that no stream stops and resumes mid-episode.

        The other half of the same hardware fault as `_validate_stream_desync`:
        feedback that drops out and comes back leaves the stream the right
        length overall, with a hole in the middle that no threshold on values
        can see.
        """
        if self._max_sample_gap is None:
            return True
        valid = True
        for _, path, timestamps in streams:
            if len(timestamps) < 2:
                continue
            gap = np.diff(timestamps).max() / 1e9
            if gap > self._max_sample_gap:
                self._report_error(
                    f"{self._relative_path(path)}: {gap:.2f}s gap between "
                    f"samples > {self._max_sample_gap}s"
                )
                valid = False
        return valid

    @staticmethod
    def _span(timestamps: np.ndarray) -> float:
        """Seconds from a stream's first sample to its last."""
        if len(timestamps) < 2:
            return 0.0
        return float(timestamps[-1] - timestamps[0]) / 1e9

    def _has_null(self, path) -> bool:
        file_meta = pq.read_metadata(path)
        for rg_index in range(file_meta.num_row_groups):
            row_group = file_meta.row_group(rg_index)
            for col_index in range(row_group.num_columns):
                col_meta = row_group.column(col_index)
                col_name = col_meta.path_in_schema.split(".")[0]
                if col_name == "timestamp":
                    continue
                stats = col_meta.statistics
                if stats is not None and stats.has_null_count and stats.null_count > 0:
                    return True
        # Column statistics don't count NaN as null.
        table = pq.read_table(path)
        for col_name in table.schema.names:
            if col_name == "timestamp":
                continue
            col = table.column(col_name)
            flat = col.combine_chunks().values
            if pa.types.is_floating(flat.type) and pc.any(pc.is_nan(flat)).as_py():
                return True
        return False
