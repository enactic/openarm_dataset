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
import pandas as pd
import numpy as np
import subprocess
import tempfile
import json
import shutil
import sys
import os

from .dataset import Dataset

ROBOT_TYPE = "openarm_bimanual"
CHUNK_SIZE = 1000

# config for video encoding
FFMPEG_CODEC = "libx264"
VIDEO_PIX_FMT = "yuv420p"
VIDEO_CODEC = "h264"


def _joint_names_from_attr(attr):
    component = attr["component"]
    joints = attr["embodiment"].joints
    if component is None:
        return [f"{joint}.pos" for joint in joints]
    return [f"{component}_{joint}.pos" for joint in joints]


def _collect_keys_and_joint_names(dataset: Dataset, mode: str):
    keys = []
    joint_names = []

    for attr in dataset._get_embodiment_attributes(mode, 0):
        keys.append(attr["key"])
        joint_names.extend(_joint_names_from_attr(attr))
    return keys, joint_names


def collect_downsampled_data(dataset: Dataset, fps: int, obs_keys, act_keys):
    record = []
    for i in range(dataset.meta.num_episodes):
        samples = dataset.sample(hz=fps, episode_index=i)
        n = len(samples)
        sampled_obs = [
            np.concatenate([s.obs[k] for k in obs_keys], axis=0).astype(np.float32)
            for s in samples
        ]
        sampled_actions = [
            np.concatenate([s.action[k] for k in act_keys], axis=0).astype(np.float32)
            for s in samples
        ]
        sampled_cameras = {
            k: [Path(s.cameras[k].path) for s in samples] for k in dataset.camera_names
        }
        episode_record = (
            i,
            n,
            sampled_obs,
            sampled_actions,
            sampled_cameras,
        )  # (episode_index, num_frames, sampled_obs, sampled_actions, sampled_cameras)
        record.append(episode_record)
    return record


def get_chunk_name(episode_id: int):
    return f"chunk-{episode_id // CHUNK_SIZE:03d}"


def get_imagename_from_key(key: str):
    return f"observation.images.{key}"


def _get_ffmpeg_exe() -> str | None:
    """Get the path to a valid ffmpeg executable."""
    # check if ffmpeg is available in the current environment
    exe = shutil.which("ffmpeg")
    if exe and _is_valid_exe(exe):
        return exe
    return None


def _is_valid_exe(exe: str) -> bool:
    """Check if the given executable is a valid ffmpeg."""
    startupinfo = None
    creationflags = 0

    if sys.platform.startswith("win"):
        startupinfo = subprocess.STARTUPINFO()
        startupinfo.dwFlags |= subprocess.STARTF_USESHOWWINDOW

    try:
        with open(os.devnull, "w") as null:
            subprocess.check_call(
                [exe, "-version"],
                stdout=null,
                stderr=subprocess.STDOUT,
                startupinfo=startupinfo,
                creationflags=creationflags,
            )
        return True
    except (OSError, ValueError, subprocess.CalledProcessError):
        return False


def has_valid_ffmpeg() -> bool:
    """Check if a valid ffmpeg executable is available in the system."""
    exe = _get_ffmpeg_exe()
    if exe is None:
        print(
            "FFmpeg executable not found. Please install ffmpeg in order to encode videos.",
            file=sys.stderr,
        )
        return False
    return True


def _escape_concat_path(path: Path) -> str:
    return str(path.resolve()).replace("'", "'\\''")


def encode_mp4(frames: list[Path], fps: int, out_mp4: Path, verbose=True):
    if not frames:
        return
    try:
        ffmpeg_exe = _get_ffmpeg_exe()
        if ffmpeg_exe is None:
            raise RuntimeError("FFmpeg executable not found.")
    except RuntimeError as e:
        raise RuntimeError(
            "FFmpeg is required for video encoding but was not found. Please install FFmpeg in your conda environment or ensure it is available in your system PATH."
        ) from e
    list_path = None
    try:
        with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f_list:
            list_path = f_list.name
            for f_path in frames:
                f_list.write(f"file '{_escape_concat_path(f_path)}'\n")

        cmd = [
            ffmpeg_exe,  # use the detected ffmpeg executable path
            "-y",
            "-nostdin",
            "-loglevel",
            "warning",
            "-stats",
            "-f",
            "concat",
            "-safe",
            "0",
            "-r",
            str(fps),
            "-i",
            list_path,
            "-c:v",
            FFMPEG_CODEC,
            "-preset",
            "veryfast",
            "-pix_fmt",
            VIDEO_PIX_FMT,
            str(out_mp4),
        ]
        subprocess.run(cmd, check=True, capture_output=not verbose)
    finally:
        if list_path and os.path.exists(list_path):
            os.unlink(list_path)


def _describe_vector(X):
    D = X.shape[1] if X.ndim == 2 else 0
    keys = ("min", "max", "mean", "std", "q01", "q10", "q50", "q90", "q99")

    if X.size == 0 or D == 0:
        return {k: [None] * D for k in keys} | {"count": [0]}

    result = {
        "min": np.nanmin(X, axis=0).astype(float).tolist(),
        "max": np.nanmax(X, axis=0).astype(float).tolist(),
        "mean": np.nanmean(X, axis=0).astype(float).tolist(),
        "std": np.nanstd(X, axis=0).astype(float).tolist(),
        "count": [int(X.shape[0])],
    }

    percentiles = np.nanpercentile(X, [1, 10, 50, 90, 99], axis=0)
    for name, values in zip(("q01", "q10", "q50", "q90", "q99"), percentiles):
        result[name] = values.astype(float).tolist()

    return result


def _describe_scalar(x):
    if x.size == 0:
        return {
            k: [None]
            for k in (
                "min",
                "max",
                "mean",
                "std",
                "q01",
                "q10",
                "q50",
                "q90",
                "q99",
            )
        } | {"count": [0]}

    result = {
        "min": [float(np.nanmin(x))],
        "max": [float(np.nanmax(x))],
        "mean": [float(np.nanmean(x))],
        "std": [float(np.nanstd(x))],
        "count": [int(x.size)],
    }
    result.update(
        {
            name: [float(value)]
            for name, value in zip(
                ("q01", "q10", "q50", "q90", "q99"),
                np.nanpercentile(x, [1, 10, 50, 90, 99]),
            )
        }
    )
    return result


def calc_episode_stats(
    sampled_obs, sampled_actions, out_idx: int, gidx: int, task_index, fps: int, cameras
) -> dict:
    length = len(sampled_obs)
    Act = np.vstack(sampled_actions).astype(np.float32)
    Obs = np.vstack(sampled_obs).astype(np.float32)
    timestamps = np.arange(length, dtype=np.float64) / float(fps)
    stats = {
        "episode_index": out_idx,
        "dataset_from_index": gidx,
        "dataset_to_index": gidx + length,
        "stats": {},
    }
    stats["stats"]["action"] = _describe_vector(Act)
    stats["stats"]["observation.state"] = _describe_vector(Obs)
    stats["stats"]["timestamp"] = _describe_scalar(timestamps)
    stats["stats"]["frame_index"] = _describe_scalar(np.arange(length, dtype=np.int64))
    stats["stats"]["episode_index"] = _describe_scalar(
        np.full(length, out_idx, dtype=np.int64)
    )
    stats["stats"]["index"] = _describe_scalar(
        np.arange(gidx, gidx + length, dtype=np.int64)
    )
    stats["stats"]["task_index"] = _describe_scalar(
        np.full(length, task_index, dtype=np.int64)
    )
    return stats


def write_parquet(dataset, record, output_dir, fps):
    gidx = 0
    for i, n, sampled_obs, sampled_actions, _ in record:
        task_index = int(dataset.meta.episodes[i]["task_index"])
        success = bool(dataset.meta.episodes[i]["success"])
        t_cam = np.arange(n, dtype=np.float64) / float(fps)
        out_idx = i
        df = pd.DataFrame(
            {
                "action": sampled_actions,
                "observation.state": sampled_obs,
                "timestamp": t_cam.astype(np.float64),
                "frame_index": np.arange(n, dtype=np.int64),
                "episode_index": np.full(n, out_idx, dtype=np.int64),
                "index": np.arange(gidx, gidx + n, dtype=np.int64),
                "task_index": np.full(n, task_index, dtype=np.int64),
                "success": np.full(n, success, dtype=np.int64),
                "last_frame_index": np.full(n, n - 1, dtype=np.int64),
            }
        )
        parquet_path = (
            output_dir / "data" / get_chunk_name(i) / f"episode_{i:06d}.parquet"
        )
        parquet_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(parquet_path, index=False)
        gidx += n


def write_videos(dataset, record, output_dir, fps):
    for i, _, _, _, sampled_cameras in record:
        for camera_key in dataset.camera_names:
            video_path = (
                output_dir
                / "videos"
                / get_chunk_name(i)
                / get_imagename_from_key(camera_key)
                / f"episode_{i:06d}.mp4"
            )
            video_path.parent.mkdir(parents=True, exist_ok=True)
            encode_mp4(sampled_cameras[camera_key], fps, video_path)


def write_metadata(dataset, record, output_dir, fps, train_split, joint_names):
    METADATA_DIR = "meta"
    episodes_metadata = []
    episodes_stats = []

    A_all = []
    O_all = []
    timestamp_all = []
    frame_index_all = []
    episode_index_all = []
    task_index_all = []
    index_all = []
    success_all = []
    last_frame_index_all = []

    gidx = 0
    for i, n, sampled_obs, sampled_actions, _ in record:
        # save for overall stats
        A_all.append(sampled_actions)
        O_all.append(sampled_obs)
        timestamp_all.append(np.arange(n, dtype=np.float64) / float(fps))
        frame_index_all.append(np.arange(n, dtype=np.int64))
        episode_index_all.append(np.full(n, i, dtype=np.int64))
        task_index_all.append(
            np.full(n, int(dataset.meta.episodes[i]["task_index"]), dtype=np.int64)
        )
        index_all.append(np.arange(gidx, gidx + n, dtype=np.int64))
        success_all.append(
            np.full(n, bool(dataset.meta.episodes[i]["success"]), dtype=np.int64)
        )
        last_frame_index_all.append(np.full(n, n - 1, dtype=np.int64))

        # episodes metadata and stats
        task_index = int(dataset.meta.episodes[i]["task_index"])
        task_name = dataset.meta.data["tasks"][task_index]["prompt"]
        out_idx = i
        rec = {
            "episode_index": out_idx,
            "task": [task_name],
            "length": len(sampled_obs),
        }
        episodes_metadata.append(rec)

        stats = calc_episode_stats(
            sampled_obs,
            sampled_actions,
            out_idx,
            gidx,
            task_index,
            fps,
            dataset.camera_names,
        )
        episodes_stats.append(stats)
        gidx += len(sampled_obs)
    # save episodes.jsonl
    episodes_metadata_path = output_dir / METADATA_DIR / "episodes.jsonl"
    episodes_metadata_path.parent.mkdir(parents=True, exist_ok=True)
    with episodes_metadata_path.open("w", encoding="utf-8") as f:
        for rec in episodes_metadata:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    # save episodes_stats.jsonl
    episodes_stats_path = output_dir / METADATA_DIR / "episodes_stats.jsonl"
    episodes_stats_path.parent.mkdir(parents=True, exist_ok=True)
    with episodes_stats_path.open("w", encoding="utf-8") as f:
        for stats in episodes_stats:
            f.write(json.dumps(stats, ensure_ascii=False) + "\n")

    # save tasks.jsonl
    tasks = set()
    for episode in dataset.meta.episodes:
        task_index = int(episode["task_index"])
        task_name = dataset.meta.data["tasks"][task_index]["prompt"]
        tasks.add((task_index, task_name))
    tasks = sorted(tasks)
    tasks_path = output_dir / METADATA_DIR / "tasks.jsonl"
    tasks_path.parent.mkdir(parents=True, exist_ok=True)
    with tasks_path.open("w", encoding="utf-8") as f:
        for task_index, task_name in tasks:
            rec = {
                "task_index": task_index,
                "task": task_name,
            }
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    # stats.json
    A_all = (
        np.vstack(A_all) if A_all else np.empty((0, len(joint_names)), dtype=np.float32)
    )
    O_all = (
        np.vstack(O_all) if O_all else np.empty((0, len(joint_names)), dtype=np.float32)
    )
    timestamp_all = (
        np.concatenate(timestamp_all)
        if timestamp_all
        else np.empty((0,), dtype=np.float64)
    )
    frame_index_all = (
        np.concatenate(frame_index_all)
        if frame_index_all
        else np.empty((0,), dtype=np.int64)
    )
    episode_index_all = (
        np.concatenate(episode_index_all)
        if episode_index_all
        else np.empty((0,), dtype=np.int64)
    )
    task_index_all = (
        np.concatenate(task_index_all)
        if task_index_all
        else np.empty((0,), dtype=np.int64)
    )
    index_all = (
        np.concatenate(index_all) if index_all else np.empty((0,), dtype=np.int64)
    )
    success_all = (
        np.concatenate(success_all) if success_all else np.empty((0,), dtype=np.int64)
    )
    last_frame_index_all = (
        np.concatenate(last_frame_index_all)
        if last_frame_index_all
        else np.empty((0,), dtype=np.int64)
    )

    overall_stats = {
        "action": _describe_vector(A_all),
        "observation.state": _describe_vector(O_all),
        "timestamp": _describe_scalar(timestamp_all),
        "frame_index": _describe_scalar(frame_index_all),
        "episode_index": _describe_scalar(episode_index_all),
        "task_index": _describe_scalar(task_index_all),
        "index": _describe_scalar(index_all),
        "success": _describe_scalar(success_all),
        "last_frame_index": _describe_scalar(last_frame_index_all),
    }
    stats_path = output_dir / METADATA_DIR / "stats.json"
    stats_path.parent.mkdir(parents=True, exist_ok=True)
    with stats_path.open("w", encoding="utf-8") as f:
        json.dump(overall_stats, f, ensure_ascii=False, indent=4)

    # info.json
    features = {
        "action": {
            "dtype": "float32",
            "names": joint_names,
            "shape": [len(joint_names)],
        },
        "observation.state": {
            "dtype": "float32",
            "names": joint_names,
            "shape": [len(joint_names)],
        },
        "timestamp": {"dtype": "float64", "shape": [1], "names": None},
        "frame_index": {"dtype": "int64", "shape": [1], "names": None},
        "episode_index": {"dtype": "int64", "shape": [1], "names": None},
        "index": {"dtype": "int64", "shape": [1], "names": None},
        "task_index": {"dtype": "int64", "shape": [1], "names": None},
        "success": {"dtype": "int64", "shape": [1], "names": None},
        "last_frame_index": {"dtype": "int64", "shape": [1], "names": None},
    }
    sample_record = dataset.sample(hz=fps, episode_index=0)[0]
    for cam in dataset.camera_names:
        sample_image = sample_record.cameras[cam].load()
        h, w = sample_image.shape[:2]
        features[f"{get_imagename_from_key(cam)}"] = {
            "dtype": "video",
            "shape": [h, w, 3],
            "names": ["height", "width", "channels"],
            "info": {
                "video.height": h,
                "video.width": w,
                "video.codec": VIDEO_CODEC,
                "video.pix_fmt": VIDEO_PIX_FMT,
                "video.is_depth_map": False,
                "video.fps": fps,
                "video.channels": 3,
                "has_audio": False,
            },
        }
    num_episodes = len(dataset.meta.episodes)
    total_chunks = max((num_episodes - 1) // CHUNK_SIZE + 1, 0) if num_episodes else 0
    train_end = int(num_episodes * train_split)
    splits = {"train": f"0:{train_end}"}
    if train_end < num_episodes:
        splits["val"] = f"{train_end}:{num_episodes}"
    info = {
        "codebase_version": "v2.1",
        "robot_type": ROBOT_TYPE,
        "total_episodes": num_episodes,
        "total_frames": len(index_all),
        "total_tasks": len(set(task_index_all)),
        "total_videos": num_episodes * len(dataset.camera_names),
        "total_chunks": total_chunks,
        "chunks_size": CHUNK_SIZE,
        "fps": fps,
        "splits": splits,
        "data_path": "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet",
        "video_path": "videos/chunk-{episode_chunk:03d}/{video_key}/episode_{episode_index:06d}.mp4",
        "features": features,
    }
    info_path = output_dir / METADATA_DIR / "info.json"
    with info_path.open("w", encoding="utf-8") as f:
        json.dump(info, f, ensure_ascii=False, indent=4)


def to_lerobotv21(
    dataset: Dataset,
    output_dir: str | Path,
    fps: int = 30,
    train_split: float = 0.8,
    smoothing_cutoff: float = 1.0,
) -> None:
    # Validate inputs
    if not (0.0 <= train_split <= 1.0):
        raise ValueError(f"train_split must be between 0 and 1, got {train_split}")
    if fps <= 0:
        raise ValueError(f"fps must be a positive integer, got {fps}")

    # set smoothing cutoff
    dataset.set_smoothing(cutoff=smoothing_cutoff)
    # Create the output directories
    output_dir = Path(output_dir)

    obs_keys, obs_joint_names = _collect_keys_and_joint_names(dataset, "obs")
    action_keys, action_joint_names = _collect_keys_and_joint_names(dataset, "action")

    if obs_joint_names != action_joint_names:
        raise ValueError(
            "Observation joint names and action joint names do not match: "
            f"{obs_joint_names} vs {action_joint_names}"
        )

    joint_names = obs_joint_names

    # collect downsampled data for each episode
    record = collect_downsampled_data(dataset, fps, obs_keys, action_keys)

    # save parquet files for each episode (output_dir/data)
    write_parquet(dataset, record, output_dir, fps)
    # save_videos for each episode (output_dir/videos)
    write_videos(dataset, record, output_dir, fps)
    # episodes metadata and stats
    write_metadata(dataset, record, output_dir, fps, train_split, joint_names)
