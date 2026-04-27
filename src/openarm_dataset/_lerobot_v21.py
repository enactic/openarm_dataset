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

from .dataset import Dataset

ROBOT_TYPE = "openarm_bimanual"
CHUNK_SIZE = 1000

# config for video encoding
FFMPEG_CODEC = "libx264"
VIDEO_PIX_FMT = "yuv420p"
VIDEO_CODEC = "h264"


def _get_joint_names(component, joints):
    if component is None:
        return [f"{joint}.pos" for joint in joints]
    return [f"{component}_{joint}.pos" for joint in joints]


def _collect_keys_and_joint_names(dataset: Dataset):
    keys = []
    joint_names = []
    for name, embodiment in dataset.meta.equipment.embodiments.items():
        if embodiment.components:
            for component in embodiment.components:
                for attribute in embodiment.attributes:
                    key = f"{name}/{component}/{attribute}"
                    keys.append(key)
                    joint_names.extend(_get_joint_names(component, embodiment.joints))
        else:
            for attribute in embodiment.attributes:
                key = f"{name}/{attribute}"
                keys.append(key)
                joint_names.extend(_get_joint_names(None, embodiment.joints))
    return keys, joint_names


def _collect_downsampled_data(dataset: Dataset, fps: int, joint_keys):
    records = []
    for episode_index in range(dataset.meta.num_episodes):
        samples = dataset.sample(hz=fps, episode_index=episode_index)
        num_frames = len(samples)
        sampled_obs = [
            np.concatenate([s.obs[k] for k in joint_keys], axis=0).astype(np.float32)
            for s in samples
        ]
        sampled_actions = [
            np.concatenate([s.action[k] for k in joint_keys], axis=0).astype(np.float32)
            for s in samples
        ]
        sampled_cameras = {
            k: [Path(s.cameras[k].path) for s in samples] for k in dataset.camera_names
        }
        record = (
            episode_index,
            num_frames,
            sampled_obs,
            sampled_actions,
            sampled_cameras,
        )
        records.append(record)
    return records


def _get_chunk_name(episode_id: int):
    return f"chunk-{episode_id // CHUNK_SIZE:03d}"


def _get_image_name_from_key(key: str):
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

    # On Windows, hide the console window when running ffmpeg
    if hasattr(subprocess, "STARTUPINFO"):
        startupinfo = subprocess.STARTUPINFO()
        startupinfo.dwFlags |= subprocess.STARTF_USESHOWWINDOW

    try:
        subprocess.check_call(
            [exe, "-version"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.STDOUT,
            startupinfo=startupinfo,
        )
        return True
    except (OSError, ValueError, subprocess.CalledProcessError):
        return False


def _escape_concat_path(path: Path) -> str:
    return str(path.resolve()).replace("'", "'\\''")


def _encode_mp4(frames: list[Path], fps: int, out_mp4: Path, verbose=True):
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
    with tempfile.TemporaryDirectory() as temp_dir:
        list_path = Path(temp_dir) / "ffmpeg_concat.txt"
        with list_path.open("w") as f_list:
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
            str(list_path),
            "-c:v",
            FFMPEG_CODEC,
            "-preset",
            "veryfast",
            "-pix_fmt",
            VIDEO_PIX_FMT,
            str(out_mp4),
        ]
        subprocess.run(cmd, check=True, capture_output=not verbose)


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


def _calc_episode_stats(
    sampled_obs, sampled_actions, out_idx: int, gidx: int, task_index, fps: int, cameras
) -> dict:
    length = len(sampled_obs)
    actions = np.vstack(sampled_actions).astype(np.float32)
    observations = np.vstack(sampled_obs).astype(np.float32)
    timestamps = np.arange(length, dtype=np.float64) / float(fps)
    stats = {
        "episode_index": out_idx,
        "dataset_from_index": gidx,
        "dataset_to_index": gidx + length,
        "stats": {},
    }
    stats["stats"]["action"] = _describe_vector(actions)
    stats["stats"]["observation.state"] = _describe_vector(observations)
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


def _write_parquet(dataset, records, output_dir, fps):
    gidx = 0
    for episode_index, num_frames, sampled_obs, sampled_actions, _ in records:
        task_index = int(dataset.meta.episodes[episode_index]["task_index"])
        success = bool(dataset.meta.episodes[episode_index]["success"])
        t_cam = np.arange(num_frames, dtype=np.float64) / float(fps)
        df = pd.DataFrame(
            {
                "action": sampled_actions,
                "observation.state": sampled_obs,
                "timestamp": t_cam.astype(np.float64),
                "frame_index": np.arange(num_frames, dtype=np.int64),
                "episode_index": np.full(num_frames, episode_index, dtype=np.int64),
                "index": np.arange(gidx, gidx + num_frames, dtype=np.int64),
                "task_index": np.full(num_frames, task_index, dtype=np.int64),
                "success": np.full(num_frames, success, dtype=np.int64),
                "last_frame_index": np.full(num_frames, num_frames - 1, dtype=np.int64),
            }
        )
        parquet_path = (
            output_dir
            / "data"
            / _get_chunk_name(episode_index)
            / f"episode_{episode_index:06d}.parquet"
        )
        parquet_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(parquet_path, index=False)
        gidx += num_frames


def _write_videos(dataset, records, output_dir, fps):
    for episode_index, _, _, _, sampled_cameras in records:
        for camera_key in dataset.camera_names:
            video_path = (
                output_dir
                / "videos"
                / _get_chunk_name(episode_index)
                / _get_image_name_from_key(camera_key)
                / f"episode_{episode_index:06d}.mp4"
            )
            video_path.parent.mkdir(parents=True, exist_ok=True)
            _encode_mp4(sampled_cameras[camera_key], fps, video_path)


def _write_metadata(dataset, records, output_dir, fps, train_split, joint_names):
    METADATA_DIR = "meta"
    episodes_metadata = []
    episodes_stats = []

    all_actions = []
    all_observations = []
    timestamp_all = []
    frame_index_all = []
    episode_index_all = []
    task_index_all = []
    index_all = []
    success_all = []
    last_frame_index_all = []

    gidx = 0
    for episode_index, num_frames, sampled_obs, sampled_actions, _ in records:
        # save for overall stats
        all_actions.append(sampled_actions)
        all_observations.append(sampled_obs)
        timestamp_all.append(np.arange(num_frames, dtype=np.float64) / float(fps))
        frame_index_all.append(np.arange(num_frames, dtype=np.int64))
        episode_index_all.append(np.full(num_frames, episode_index, dtype=np.int64))
        task_index_all.append(
            np.full(
                num_frames,
                int(dataset.meta.episodes[episode_index]["task_index"]),
                dtype=np.int64,
            )
        )
        index_all.append(np.arange(gidx, gidx + num_frames, dtype=np.int64))
        success_all.append(
            np.full(
                num_frames,
                bool(dataset.meta.episodes[episode_index]["success"]),
                dtype=np.int64,
            )
        )
        last_frame_index_all.append(np.full(num_frames, num_frames - 1, dtype=np.int64))

        # episodes metadata and stats
        task_index = int(dataset.meta.episodes[episode_index]["task_index"])
        task_name = dataset.meta.data["tasks"][task_index]["prompt"]
        rec = {
            "episode_index": episode_index,
            "task": [task_name],
            "length": len(sampled_obs),
        }
        episodes_metadata.append(rec)

        stats = _calc_episode_stats(
            sampled_obs,
            sampled_actions,
            episode_index,
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
    all_actions = (
        np.vstack(all_actions)
        if all_actions
        else np.empty((0, len(joint_names)), dtype=np.float32)
    )
    all_observations = (
        np.vstack(all_observations)
        if all_observations
        else np.empty((0, len(joint_names)), dtype=np.float32)
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
        "action": _describe_vector(all_actions),
        "observation.state": _describe_vector(all_observations),
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
        features[f"{_get_image_name_from_key(cam)}"] = {
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

    # Collect joint keys and names
    joint_keys, joint_names = _collect_keys_and_joint_names(dataset)

    # collect downsampled data for each episode
    records = _collect_downsampled_data(dataset, fps, joint_keys)

    # save parquet files for each episode (output_dir/data)
    _write_parquet(dataset, records, output_dir, fps)
    # save_videos for each episode (output_dir/videos)
    _write_videos(dataset, records, output_dir, fps)
    # episodes metadata and stats
    _write_metadata(dataset, records, output_dir, fps, train_split, joint_names)
