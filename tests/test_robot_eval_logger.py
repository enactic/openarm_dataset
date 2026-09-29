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

import pytest

pytest.importorskip("lz4")

import json
import pickle
from pathlib import Path

import lz4.frame
import numpy as np

from openarm_dataset import Dataset

DATASET_PATH = Path(__file__).parent / "fixture" / "dataset_0.3.0"
# The fixture records both arms, so it exercises the ambiguous-gripper path.
COMPONENTS = ("left", "right")


@pytest.fixture(scope="module")
def dataset():
    return Dataset(DATASET_PATH)


@pytest.fixture(scope="module")
def run_dir(dataset, tmp_path_factory):
    output = tmp_path_factory.mktemp("rel")
    dataset.write(output, "robot_eval_logger", gripper_component="right")
    runs = list(output.iterdir())
    assert len(runs) == 1, "exactly one <eval_id> run directory per conversion"
    return runs[0]


def _load(path):
    return pickle.loads(lz4.frame.decompress(path.read_bytes()))


def test_bimanual_without_a_choice_raises_rather_than_guessing(dataset, tmp_path):
    """The required gripper field is (T, 1) and this dataset has two arms.

    Picking one silently would mislabel every converted bimanual dataset,
    so the converter must refuse and say what the options are.
    """
    with pytest.raises(ValueError, match="more than one arm"):
        dataset.write(tmp_path / "out", "robot_eval_logger")
    assert not (tmp_path / "out").exists(), "a rejected call must not leave a run"


def test_unknown_gripper_component_raises(dataset, tmp_path):
    with pytest.raises(ValueError, match="not in this dataset"):
        dataset.write(tmp_path / "out", "robot_eval_logger", gripper_component="middle")


def test_run_directory_name_matches_eval_id(run_dir):
    metadata = json.loads((run_dir / "metadata.json").read_text())
    assert str(metadata["eval_id"]) == run_dir.name
    assert metadata["eval_id"] > 0


def test_metadata_declares_openarm_and_joint_position(run_dir):
    metadata = json.loads((run_dir / "metadata.json").read_text())
    assert metadata["robot_type"] == "openarm"
    assert metadata["control_mode"] == "joint_position"
    assert metadata["action_frequency_hz"] > 0
    # Present and ISO 8601-ish rather than exact, since it is the run time.
    assert "T" in metadata["time"]


def test_one_trajectory_file_per_episode_numbered_from_zero(run_dir, dataset):
    trajectories = sorted(run_dir.glob("traj_*.pkl"))
    assert trajectories
    expected = {f"traj_{i}.pkl" for i in range(len(trajectories))}
    assert {p.name for p in trajectories} == expected


def test_step_arrays_agree_on_length_and_dtype(run_dir):
    episode = _load(run_dir / "traj_0.pkl")
    steps = episode.episode_length
    assert steps > 0
    assert episode.joint_position.shape == (steps, episode.joint_position.shape[1])
    assert episode.joint_position.dtype == np.float32
    assert episode.action.dtype == np.float32
    assert len(episode.action) == steps
    assert episode.gripper.shape == (steps, 1)
    assert episode.gripper.dtype == np.float32
    for name, frames in episode.obs.items():
        assert frames.dtype == np.uint8, name
        assert frames.ndim == 4 and frames.shape[-1] == 3, name
        assert len(frames) == steps, name


def test_both_grippers_are_preserved_losslessly(run_dir):
    """The single required field is one arm's; neither arm may be dropped."""
    episode = _load(run_dir / "traj_0.pkl")
    for component in COMPONENTS:
        preserved = getattr(episode, f"{component}_gripper")
        assert preserved.shape == (episode.episode_length, 1)
        assert preserved.dtype == np.float32
    # The required field must be exactly the arm that was asked for.
    np.testing.assert_array_equal(episode.gripper, episode.right_gripper)


def _arm_joints(dataset):
    """(embodiment name, component, joint names) for every arm, in the metadata's order."""
    return [
        (name, component, embodiment.joints)
        for name, embodiment in dataset.meta.equipment.embodiments.items()
        if embodiment.components and "qpos" in embodiment.attributes
        for component in embodiment.components
    ]


def test_joint_position_excludes_the_gripper_column(run_dir, dataset):
    """qpos is [joint1..joint7, gripper] per arm; only the joints belong here."""
    episode = _load(run_dir / "traj_0.pkl")
    expected = sum(len(joints) - 1 for _, _, joints in _arm_joints(dataset))
    assert episode.joint_position.shape == (episode.episode_length, expected)


def test_action_keeps_the_gripper_commands(run_dir, dataset):
    """action is every arm's raw commanded qpos, grippers included, in order."""
    episode = _load(run_dir / "traj_0.pkl")
    arms = _arm_joints(dataset)
    assert episode.action.shape == (
        episode.episode_length,
        sum(len(joints) for _, _, joints in arms),
    )
    samples = dataset.sample(hz=30, episode=dataset.meta.episodes[0], state="qpos")
    expected = np.asarray(
        [
            np.concatenate(
                [
                    np.asarray(
                        sample.action[f"{name}/{component}/qpos"], dtype=np.float32
                    )
                    for name, component, _ in arms
                ]
            )
            for sample in samples
        ],
        dtype=np.float32,
    )
    np.testing.assert_array_equal(episode.action, expected)


def test_optional_modalities_are_carried_through(run_dir):
    """The source records qvel and qtorque; the target has fields for both.

    Dropping them would lose data the format is able to represent.
    """
    episode = _load(run_dir / "traj_0.pkl")
    steps = episode.episode_length
    columns = episode.joint_position.shape[1]
    for name in ("joint_velocity", "joint_effort"):
        values = getattr(episode, name)
        assert values.shape == (steps, columns), name
        assert values.dtype == np.float32, name


def test_required_episode_fields_come_from_the_dataset(run_dir, dataset):
    episode = _load(run_dir / "traj_0.pkl")
    assert isinstance(episode.success, bool)
    prompts = {task["prompt"] for task in dataset.meta.data["tasks"]}
    assert episode.language_command in prompts


def test_success_only_keeps_only_successful_episodes(dataset, tmp_path):
    everything = tmp_path / "all"
    successes = tmp_path / "ok"
    dataset.write(everything, "robot_eval_logger", gripper_component="right")
    dataset.write(
        successes, "robot_eval_logger", gripper_component="right", success_only=True
    )
    n_all = len(list(next(everything.iterdir()).glob("traj_*.pkl")))
    n_ok = len(list(next(successes.iterdir()).glob("traj_*.pkl")))
    episodes = dataset.meta.episodes
    assert n_all == len(episodes)
    assert n_ok == sum(1 for episode in episodes if episode.get("success", False))


def test_eval_options_reach_the_run_directory_and_metadata(dataset, tmp_path):
    dataset.write(
        tmp_path,
        "robot_eval_logger",
        gripper_component="right",
        eval_id=1234567,
        eval_name="nightly",
        location="Tokyo",
        evaluator_name="tester",
    )
    metadata = json.loads((tmp_path / "1234567" / "metadata.json").read_text())
    assert metadata["eval_id"] == 1234567
    assert metadata["eval_name"] == "nightly"
    assert metadata["location"] == "Tokyo"
    assert metadata["evaluator_name"] == "tester"


def test_cli_passes_the_eval_options(tmp_path, monkeypatch):
    from openarm_dataset import convert

    argv = [
        "openarm-dataset-convert",
        str(DATASET_PATH),
        str(tmp_path),
        "--format",
        "robot_eval_logger",
        "--gripper-component",
        "right",
        "--eval-id",
        "42",
        "--eval-name",
        "nightly",
        "--location",
        "Tokyo",
        "--evaluator-name",
        "tester",
    ]
    monkeypatch.setattr("sys.argv", argv)
    convert.main()
    metadata = json.loads((tmp_path / "42" / "metadata.json").read_text())
    assert (metadata["eval_id"], metadata["eval_name"]) == (42, "nightly")
    assert (metadata["location"], metadata["evaluator_name"]) == ("Tokyo", "tester")


def test_cli_rejects_eval_options_for_other_formats(tmp_path, monkeypatch):
    from openarm_dataset import convert

    monkeypatch.setattr(
        "sys.argv",
        [
            "openarm-dataset-convert",
            str(DATASET_PATH),
            str(tmp_path),
            "--eval-id",
            "42",
        ],
    )
    with pytest.raises(SystemExit):
        convert.main()


def test_trajectory_loads_without_openarm_dataset(run_dir, tmp_path):
    """The reader must not need our package to unpickle a trajectory.

    robot_eval_logger's DATA_FORMAT.md says no knowledge of the producing
    codebase is required. The subprocess blocks ``openarm_dataset`` outright
    (installed or not), so a pickled class of ours would fail to load.
    """
    import subprocess
    import sys

    subprocess.run(
        [
            sys.executable,
            "-c",
            "import sys; sys.modules['openarm_dataset'] = None; "
            "import pickle, lz4.frame; "
            "episode = pickle.loads(lz4.frame.decompress(open(sys.argv[1], 'rb').read())); "
            "assert episode.joint_position.ndim == 2",
            str(run_dir / "traj_0.pkl"),
        ],
        check=True,
        cwd=tmp_path,
    )
