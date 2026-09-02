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

        """
        self._dataset = dataset
        self._on_error = on_error
        self._update_metadata = update_metadata
        self._qpos_jump_threshold = qpos_jump_threshold
        self._qpos_absmax = qpos_absmax
        self._min_duration = min_duration

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
        null_paths = self._collect_null_paths(episode)
        valid = not null_paths
        # Files with nulls are skipped: their values cannot be compared
        # against a threshold, and they are already reported.
        if not self._validate_qpos(episode, null_paths):
            valid = False
        if not self._validate_duration(episode):
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
        """Check the episode duration against the minimum duration."""
        if self._min_duration is None:
            return True
        duration = self._episode_duration(episode)
        if duration is None or duration >= self._min_duration:
            return True
        self._report_error(
            f"episodes/{episode['id']}: "
            f"duration={duration:.2f}s < {self._min_duration}s"
        )
        return False

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
