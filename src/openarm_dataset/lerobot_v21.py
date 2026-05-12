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

"""Conversion script for OpenArm Dataset to LeRobot v2.1 format."""

from pathlib import Path
import pandas as pd
import numpy as np
import subprocess
import tempfile
import json
import shutil

from .dataset import Dataset
from PIL import Image

ROBOT_TYPE = "openarm_bimanual"
CHUNK_SIZE = 1000

# config for video encoding
FFMPEG_CODEC = "libx264"
VIDEO_PIX_FMT = "yuv420p"
VIDEO_CODEC = "h264"

# config for image stats estimation
IMAGE_STATS_MIN_SAMPLES = 100
IMAGE_STATS_MAX_SAMPLES = 10_000
IMAGE_STATS_POWER = 0.75
IMAGE_STATS_TARGET_SIZE = 150
IMAGE_STATS_MAX_SIZE_THRESHOLD = 300


def _estimate_num_image_samples(n: int) -> int:
    if n < IMAGE_STATS_MIN_SAMPLES:
        return n
    return max(
        IMAGE_STATS_MIN_SAMPLES, min(int(n**IMAGE_STATS_POWER), IMAGE_STATS_MAX_SAMPLES)
    )


def _sample_image_indices(n: int) -> list[int]:
    if n <= 0:
        return []
    k = _estimate_num_image_samples(n)
    return np.round(np.linspace(0, n - 1, k)).astype(int).tolist()


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


def _collect_downsampled_data(
    dataset: Dataset, fps: int, joint_keys, success_only=False
):
    records = []
    for episode_index in range(dataset.meta.num_episodes):
        success = dataset.meta.episodes[episode_index]["success"]
        if not success and success_only:
            continue
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


def _build_remaps(dataset: Dataset, records):
    """Build remapping dicts from original episode/task indices to contiguous indices.

    When records is a filtered subset of episodes (e.g., success_only=True),
    original indices may be sparse. LeRobot v2.1 expects episode/task indices
    to be contiguous starting from 0. When records contains all episodes the
    returned maps are the identity.
    """
    remap_episode_index = {original: new for new, (original, *_) in enumerate(records)}
    seen = set()
    used_task_indices = []
    for original_episode_index, *_ in records:
        original_task_index = dataset.meta.episodes[original_episode_index][
            "task_index"
        ]
        if original_task_index not in seen:
            seen.add(original_task_index)
            used_task_indices.append(original_task_index)
    used_task_indices.sort()
    remap_task_index = {original: new for new, original in enumerate(used_task_indices)}
    return remap_episode_index, remap_task_index


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


def _describe_images(image_paths: list[Path]):
    """Compute per-channel min/max/mean/std for RGB images.

    subsampling: pick frames at evenly-spaced indices, then for each
    chosen frame integer-stride down to ~150 px on the long side when ≥300 px.
    """
    ch_min = np.full(3, np.inf, dtype=np.float64)
    ch_max = np.full(3, -np.inf, dtype=np.float64)
    ch_sum = np.zeros(3, dtype=np.float64)
    ch_sumsq = np.zeros(3, dtype=np.float64)

    sampled_paths = [image_paths[i] for i in _sample_image_indices(len(image_paths))]

    total_pixels = 0
    for path in sampled_paths:
        with Image.open(path) as img:
            arr = np.asarray(img.convert("RGB"))

        h, w = arr.shape[:2]
        long_side = max(h, w)
        if long_side >= IMAGE_STATS_MAX_SIZE_THRESHOLD:
            factor = max(1, int(long_side / IMAGE_STATS_TARGET_SIZE))
            arr = arr[::factor, ::factor]

        pixels = arr.reshape(-1, 3).astype(np.float64)
        ch_min = np.minimum(ch_min, pixels.min(axis=0))
        ch_max = np.maximum(ch_max, pixels.max(axis=0))
        ch_sum += pixels.sum(axis=0)
        ch_sumsq += np.square(pixels).sum(axis=0)

        total_pixels += pixels.shape[0]

    if total_pixels == 0:
        raise ValueError("No valid images were loaded.")

    mean = ch_sum / total_pixels
    var = ch_sumsq / total_pixels - np.square(mean)
    var = np.maximum(var, 0.0)  # clip negative variance to zero
    std = np.sqrt(var)

    # [0, 255] -> [0, 1]
    scale = 255.0
    stats = {
        "min": [[[float(v / scale)]] for v in ch_min],
        "max": [[[float(v / scale)]] for v in ch_max],
        "mean": [[[float(v / scale)]] for v in mean],
        "std": [[[float(v / scale)]] for v in std],
        "count": [len(sampled_paths)],
    }
    return stats


def _calc_episode_stats(
    sampled_obs,
    sampled_actions,
    episode_index: int,
    gidx: int,
    task_index,
    fps: int,
    cameras,
) -> dict:
    length = len(sampled_obs)
    actions = np.vstack(sampled_actions).astype(np.float32)
    observations = np.vstack(sampled_obs).astype(np.float32)
    timestamps = np.arange(length, dtype=np.float64) / float(fps)
    stats = {
        "episode_index": episode_index,
        "dataset_from_index": gidx,
        "dataset_to_index": gidx + length,
        "stats": {},
    }
    stats["stats"]["action"] = _describe_vector(actions)
    stats["stats"]["observation.state"] = _describe_vector(observations)
    stats["stats"]["timestamp"] = _describe_scalar(timestamps)
    stats["stats"]["frame_index"] = _describe_scalar(np.arange(length, dtype=np.int64))
    stats["stats"]["episode_index"] = _describe_scalar(
        np.full(length, episode_index, dtype=np.int64)
    )
    stats["stats"]["index"] = _describe_scalar(
        np.arange(gidx, gidx + length, dtype=np.int64)
    )
    stats["stats"]["task_index"] = _describe_scalar(
        np.full(length, task_index, dtype=np.int64)
    )
    for cam_key, cam_paths in cameras.items():
        stats["stats"][_get_image_name_from_key(cam_key)] = _describe_images(cam_paths)
    return stats


def _write_parquet(
    dataset, records, output_dir, fps, remap_episode_index, remap_task_index
):
    gidx = 0
    for episode_index, num_frames, sampled_obs, sampled_actions, _ in records:
        lerobot_episode_index = remap_episode_index[episode_index]
        task_index = remap_task_index[
            int(dataset.meta.episodes[episode_index]["task_index"])
        ]
        success = bool(dataset.meta.episodes[episode_index]["success"])
        t_cam = np.arange(num_frames, dtype=np.float64) / float(fps)
        df = pd.DataFrame(
            {
                "action": sampled_actions,
                "observation.state": sampled_obs,
                "timestamp": t_cam.astype(np.float64),
                "frame_index": np.arange(num_frames, dtype=np.int64),
                "episode_index": np.full(
                    num_frames, lerobot_episode_index, dtype=np.int64
                ),
                "index": np.arange(gidx, gidx + num_frames, dtype=np.int64),
                "task_index": np.full(num_frames, task_index, dtype=np.int64),
                "success": np.full(num_frames, success, dtype=np.int64),
                "last_frame_index": np.full(num_frames, num_frames - 1, dtype=np.int64),
            }
        )
        parquet_path = (
            output_dir
            / "data"
            / _get_chunk_name(lerobot_episode_index)
            / f"episode_{lerobot_episode_index:06d}.parquet"
        )
        parquet_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(parquet_path, index=False)
        gidx += num_frames


def _write_videos(dataset, records, output_dir, fps, remap_episode_index):
    for episode_index, _, _, _, sampled_cameras in records:
        lerobot_episode_index = remap_episode_index[episode_index]
        for camera_key in dataset.camera_names:
            video_path = (
                output_dir
                / "videos"
                / _get_chunk_name(lerobot_episode_index)
                / _get_image_name_from_key(camera_key)
                / f"episode_{lerobot_episode_index:06d}.mp4"
            )
            video_path.parent.mkdir(parents=True, exist_ok=True)
            _encode_mp4(sampled_cameras[camera_key], fps, video_path)


def _write_metadata(
    dataset,
    records,
    output_dir,
    fps,
    train_split,
    joint_names,
    remap_episode_index,
    remap_task_index,
):
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
    for (
        episode_index,
        num_frames,
        sampled_obs,
        sampled_actions,
        sampled_cameras,
    ) in records:
        lerobot_episode_index = remap_episode_index[episode_index]
        lerobot_task_index = remap_task_index[
            int(dataset.meta.episodes[episode_index]["task_index"])
        ]
        # save for overall stats
        all_actions.append(sampled_actions)
        all_observations.append(sampled_obs)
        timestamp_all.append(np.arange(num_frames, dtype=np.float64) / float(fps))
        frame_index_all.append(np.arange(num_frames, dtype=np.int64))
        episode_index_all.append(
            np.full(num_frames, lerobot_episode_index, dtype=np.int64)
        )
        task_index_all.append(np.full(num_frames, lerobot_task_index, dtype=np.int64))
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
        task_name = dataset.meta.data["tasks"][
            int(dataset.meta.episodes[episode_index]["task_index"])
        ]["prompt"]
        rec = {
            "episode_index": lerobot_episode_index,
            "tasks": [task_name],
            "length": len(sampled_obs),
        }
        episodes_metadata.append(rec)

        stats = _calc_episode_stats(
            sampled_obs,
            sampled_actions,
            lerobot_episode_index,
            gidx,
            lerobot_task_index,
            fps,
            sampled_cameras,
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

    # save tasks.jsonl using remapped (contiguous) task indices
    tasks_path = output_dir / METADATA_DIR / "tasks.jsonl"
    tasks_path.parent.mkdir(parents=True, exist_ok=True)
    tasks_sorted = sorted(remap_task_index.items(), key=lambda kv: kv[1])
    with tasks_path.open("w", encoding="utf-8") as f:
        for original_task_index, new_task_index in tasks_sorted:
            task_name = dataset.meta.data["tasks"][original_task_index]["prompt"]
            rec = {
                "task_index": new_task_index,
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
    num_episodes = len(records)
    total_chunks = max((num_episodes - 1) // CHUNK_SIZE + 1, 0) if num_episodes else 0
    train_end = round(num_episodes * train_split)
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
    success_only: bool = False,
) -> None:
    """Convert the given dataset to LeRobot v2.1 format and save to the specified output directory."""
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
    records = _collect_downsampled_data(dataset, fps, joint_keys, success_only)

    if not records:
        raise ValueError("No episodes to write.")

    # build remaps from original to contiguous output indices (identity unless filtered)
    remap_episode_index, remap_task_index = _build_remaps(dataset, records)

    # save parquet files for each episode (output_dir/data)
    _write_parquet(
        dataset, records, output_dir, fps, remap_episode_index, remap_task_index
    )
    # save_videos for each episode (output_dir/videos)
    _write_videos(dataset, records, output_dir, fps, remap_episode_index)
    # episodes metadata and stats
    _write_metadata(
        dataset,
        records,
        output_dir,
        fps,
        train_split,
        joint_names,
        remap_episode_index,
        remap_task_index,
    )
