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

from pathlib import Path


from openarm_dataset import Dataset
from openarm_dataset.rrd import to_rrd

DATASET_PATH = Path(__file__).parent / "fixture" / "dataset_0.2.0"


def test_to_rrd_creates_file(tmp_path):
    dataset = Dataset(DATASET_PATH)
    rrd_path = tmp_path / "output.rrd"
    to_rrd(dataset, rrd_path)
    assert rrd_path.exists()
    assert rrd_path.stat().st_size > 0
