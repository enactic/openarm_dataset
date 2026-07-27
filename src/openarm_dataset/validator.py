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

import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

from .metadata import Episode


class Validator:
    """Validator for OpenArm Dataset."""

    def __init__(self, dataset, on_error=None, update_metadata=False):
        """Initialize Validator.

        Args:
            dataset: The dataset to validate.
            on_error: Optional callable that is called with an error message
                string for each validation error found. If ``None``, errors
                are not reported.
            update_metadata: If ``True``, the dataset metadata is updated with
                the validation results. If ``False``, the metadata is not updated.

        """
        self._dataset = dataset
        self._on_error = on_error
        self._update_metadata = update_metadata

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
        return self._validate_no_null_values(episode)

    def _report_error(self, message: str):
        if self._on_error is not None:
            self._on_error(message)

    def _validate_no_null_values(self, episode: Episode) -> bool:
        valid = True
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
                        f"{path.relative_to(self._dataset.root_path)}: "
                        "includes null values"
                    )
                    valid = False
        return valid

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
