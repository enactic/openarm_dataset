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

import shutil
from pathlib import Path

import pytest
import yaml

from openarm_dataset.merge import MergeError, merge_datasets

FIXTURE_DIR = Path(__file__).parent / "fixture" / "dataset_0.3.0"


@pytest.fixture
def dataset_a(tmp_path):
    dst = tmp_path / "dataset_a"
    shutil.copytree(FIXTURE_DIR, dst)
    return dst


@pytest.fixture
def dataset_b(tmp_path):
    dst = tmp_path / "dataset_b"
    shutil.copytree(FIXTURE_DIR, dst)
    return dst


def _load_meta(path):
    with open(path / "metadata.yaml") as f:
        return yaml.safe_load(f)


def test_merge_two_datasets(dataset_a, dataset_b, tmp_path):
    output = tmp_path / "merged"
    merge_datasets([dataset_a, dataset_b], output)

    meta = _load_meta(output)
    assert meta["version"] == "0.3.0"
    assert len(meta["episodes"]) == 4
    assert [ep["id"] for ep in meta["episodes"]] == ["0", "1", "2", "3"]
    assert len(meta["tasks"]) == 1

    for i in range(4):
        assert (output / "episodes" / str(i)).is_dir()


def test_merge_preserves_metadata_from_first_dataset(dataset_a, dataset_b, tmp_path):
    meta_a = _load_meta(dataset_a)
    output = tmp_path / "merged"
    merge_datasets([dataset_a, dataset_b], output)

    meta = _load_meta(output)
    assert meta["operator"] == meta_a["operator"]
    assert meta["location"] == meta_a["location"]
    assert meta["operation_type"] == meta_a["operation_type"]
    assert meta["equipment"] == meta_a["equipment"]
    assert meta["frequencies"] == meta_a["frequencies"]


def test_merge_episode_data_copied(dataset_a, dataset_b, tmp_path):
    output = tmp_path / "merged"
    merge_datasets([dataset_a, dataset_b], output)

    for i in range(4):
        ep_dir = output / "episodes" / str(i)
        assert not ep_dir.is_symlink()
        assert (ep_dir / "obs").exists()
        assert (ep_dir / "action").exists()
        assert (ep_dir / "cameras").exists()


def test_merge_with_symlink(dataset_a, dataset_b, tmp_path):
    output = tmp_path / "merged"
    merge_datasets([dataset_a, dataset_b], output, symlink=True)

    for i in range(4):
        ep_path = output / "episodes" / str(i)
        assert ep_path.is_symlink()
        assert (ep_path / "obs").exists()


def test_merge_different_tasks(dataset_a, dataset_b, tmp_path):
    meta_path = dataset_b / "metadata.yaml"
    meta = _load_meta(dataset_b)
    meta["tasks"].append(
        {"prompt": "Different task.", "description": "Something else."}
    )
    meta["episodes"][1]["task_index"] = 1
    with open(meta_path, "w") as f:
        yaml.safe_dump(meta, f)

    output = tmp_path / "merged"
    merge_datasets([dataset_a, dataset_b], output)

    merged = _load_meta(output)
    assert len(merged["tasks"]) == 2
    assert merged["tasks"][0]["prompt"] == "Run test."
    assert merged["tasks"][1]["prompt"] == "Different task."
    # dataset_a: eps 0,1 both task_index 0
    assert merged["episodes"][0]["task_index"] == 0
    assert merged["episodes"][1]["task_index"] == 0
    # dataset_b ep0: same prompt -> task_index 0
    assert merged["episodes"][2]["task_index"] == 0
    # dataset_b ep1: different prompt -> task_index 1
    assert merged["episodes"][3]["task_index"] == 1


def test_merge_preserves_success_flag(dataset_a, dataset_b, tmp_path):
    output = tmp_path / "merged"
    merge_datasets([dataset_a, dataset_b], output)

    meta = _load_meta(output)
    meta_a = _load_meta(dataset_a)
    for i, ep in enumerate(meta_a["episodes"]):
        assert meta["episodes"][i]["success"] == ep["success"]


def test_merge_incompatible_version(dataset_a, dataset_b, tmp_path):
    meta = _load_meta(dataset_b)
    meta["version"] = "0.2.0"
    with open(dataset_b / "metadata.yaml", "w") as f:
        yaml.safe_dump(meta, f)

    with pytest.raises(MergeError, match="version mismatch"):
        merge_datasets([dataset_a, dataset_b], tmp_path / "merged")


def test_merge_incompatible_embodiment_version(dataset_a, dataset_b, tmp_path):
    meta = _load_meta(dataset_b)
    meta["equipment"]["embodiments"]["arms"]["version"] = "99.0"
    with open(dataset_b / "metadata.yaml", "w") as f:
        yaml.safe_dump(meta, f)

    with pytest.raises(MergeError, match="equipment.*differs"):
        merge_datasets([dataset_a, dataset_b], tmp_path / "merged")


def test_merge_incompatible_cameras(dataset_a, dataset_b, tmp_path):
    meta = _load_meta(dataset_b)
    meta["equipment"]["perceptions"]["cameras"]["extra_cam"] = {}
    with open(dataset_b / "metadata.yaml", "w") as f:
        yaml.safe_dump(meta, f)

    with pytest.raises(MergeError, match="equipment.*differs"):
        merge_datasets([dataset_a, dataset_b], tmp_path / "merged")


def test_merge_incompatible_lifter_params(dataset_a, dataset_b, tmp_path):
    meta = _load_meta(dataset_b)
    meta["equipment"]["embodiments"]["lifter"]["stroke"] = 999
    with open(dataset_b / "metadata.yaml", "w") as f:
        yaml.safe_dump(meta, f)

    with pytest.raises(MergeError, match="equipment.*differs"):
        merge_datasets([dataset_a, dataset_b], tmp_path / "merged")


def test_merge_incompatible_frequencies(dataset_a, dataset_b, tmp_path):
    meta = _load_meta(dataset_b)
    meta["frequencies"]["obs"]["arms"]["left"] = 500.0
    with open(dataset_b / "metadata.yaml", "w") as f:
        yaml.safe_dump(meta, f)

    with pytest.raises(MergeError, match="frequencies.*differs"):
        merge_datasets([dataset_a, dataset_b], tmp_path / "merged")


def test_merge_rejects_unversioned(tmp_path):
    fixture_unversioned = Path(__file__).parent / "fixture" / "dataset_unversioned"
    ds_a = tmp_path / "ds_a"
    ds_b = tmp_path / "ds_b"
    shutil.copytree(fixture_unversioned, ds_a)
    shutil.copytree(fixture_unversioned, ds_b)

    with pytest.raises(MergeError, match="unversioned"):
        merge_datasets([ds_a, ds_b], tmp_path / "merged")


def test_merge_requires_at_least_two(dataset_a, tmp_path):
    with pytest.raises(MergeError, match="At least two"):
        merge_datasets([dataset_a], tmp_path / "merged")


def test_merge_three_datasets(dataset_a, dataset_b, tmp_path):
    dataset_c = tmp_path / "dataset_c"
    shutil.copytree(FIXTURE_DIR, dataset_c)

    output = tmp_path / "merged"
    merge_datasets([dataset_a, dataset_b, dataset_c], output)

    meta = _load_meta(output)
    assert len(meta["episodes"]) == 6
    assert [ep["id"] for ep in meta["episodes"]] == [
        "0",
        "1",
        "2",
        "3",
        "4",
        "5",
    ]


def test_merged_dataset_is_loadable(dataset_a, dataset_b, tmp_path):
    output = tmp_path / "merged"
    merge_datasets([dataset_a, dataset_b], output)

    from openarm_dataset import Dataset

    ds = Dataset(output)
    assert ds.num_episodes == 4
    assert ds.meta.version == "0.3.0"
    obs = ds.load_obs(ds.meta.episodes[0])
    assert len(obs) > 0
