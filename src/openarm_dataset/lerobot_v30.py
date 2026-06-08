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

"""Conversion script for OpenArm Dataset to LeRobot v3.0 format."""

from pathlib import Path

import json
import tempfile
import numpy as np
import pandas as pd
from tqdm import tqdm

from .dataset import Dataset
from .lerobot_v21 import (
    CHUNK_SIZE,
    ROBOT_TYPE,
    VIDEO_CODEC,
    VIDEO_PIX_FMT,
    _collect_downsampled_data,
    _collect_keys_and_joint_names,
    _build_remaps,
    _describe_images,
    _describe_scalar,
    _describe_vector,
    _get_image_name_from_key,
)
from .ffmpeg import encode_mp4

CODEBASE_VERSION = "v3.0"
DATA_FILES_SIZE_IN_MB = 100
VIDEO_FILES_SIZE_IN_MB = 200

DATA_PATH = "data/chunk-{chunk_index:03d}/file-{file_index:03d}.parquet"
VIDEO_PATH = "videos/{video_key}/chunk-{chunk_index:03d}/file-{file_index:03d}.mp4"
EPISODES_PATH = "meta/episodes/chunk-{chunk_index:03d}/file-{file_index:03d}.parquet"
TASKS_PATH = "meta/tasks.parquet"
INFO_PATH = "meta/info.json"
STATS_PATH = "meta/stats.json"


def _update_chunk_file_indices(chunk_idx: int, file_idx: int) -> tuple[int, int]:
    """Advance to the next file index, rolling over to next chunk at CHUNK_SIZE."""
    if file_idx == CHUNK_SIZE - 1:
        return chunk_idx + 1, 0
    return chunk_idx, file_idx + 1


def _get_file_size_in_mb(path: Path) -> float:
    return path.stat().st_size / (1024**2)


def _write_dfs_to_parquet(
    dfs: list[pd.DataFrame], output_dir: Path, chunk_idx: int, file_idx: int
):
    packed = pd.concat(dfs, ignore_index=True)
    out = output_dir / DATA_PATH.format(chunk_index=chunk_idx, file_index=file_idx)
    out.parent.mkdir(parents=True, exist_ok=True)
    packed.to_parquet(out, index=False)


def _write_packed_parquet(
    dataset, records, output_dir, fps, remap_episode_index, remap_task_index
):
    """Write episode data into packed parquet files, splitting by size limit.

    Returns a list of dicts with per-episode data file metadata
    (``data/chunk_index``, ``data/file_index``, ``dataset_from_index``,
    ``dataset_to_index``).
    """
    chunk_idx = 0
    file_idx = 0
    size_in_mb = 0.0
    gidx = 0
    pending_dfs: list[pd.DataFrame] = []
    episodes_data_meta: list[dict] = []

    for episode_index, num_frames, sampled_obs, sampled_actions, _ in tqdm(
        records, desc="Writing data parquet", unit="ep"
    ):
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

        # Estimate this episode's parquet size by writing to a temporary file
        with tempfile.NamedTemporaryFile(suffix=".parquet") as tmp:
            tmp_path = Path(tmp.name)
            df.to_parquet(tmp_path, index=False)
            ep_size_in_mb = _get_file_size_in_mb(tmp_path)

        if size_in_mb + ep_size_in_mb >= DATA_FILES_SIZE_IN_MB and pending_dfs:
            _write_dfs_to_parquet(pending_dfs, output_dir, chunk_idx, file_idx)
            chunk_idx, file_idx = _update_chunk_file_indices(chunk_idx, file_idx)
            size_in_mb = 0.0
            pending_dfs = []

        episodes_data_meta.append(
            {
                "data/chunk_index": chunk_idx,
                "data/file_index": file_idx,
                "dataset_from_index": gidx,
                "dataset_to_index": gidx + num_frames,
            }
        )
        pending_dfs.append(df)
        size_in_mb += ep_size_in_mb
        gidx += num_frames

    if pending_dfs:
        _write_dfs_to_parquet(pending_dfs, output_dir, chunk_idx, file_idx)

    return episodes_data_meta, gidx


def _write_packed_videos(dataset, records, output_dir, fps, remap_episode_index):
    """Encode packed video files, one ffmpeg pass per file (no mp4 concat).

    Each output ``file-XXX.mp4`` is encoded in a single pass directly from the
    raw frames of the episodes assigned to it. Encoding the whole file at once
    (rather than concatenating separately-encoded per-episode clips with
    ``-c copy``) guarantees a strictly uniform ``i / fps`` presentation-timestamp
    grid.
    File sizes track ``VIDEO_FILES_SIZE_IN_MB`` approximately rather than exactly.

    Returns a list of dicts with per-episode video metadata.
    """
    episodes_video_meta: list[dict] = [{} for _ in records]

    for camera_key in dataset.camera_names:
        image_name = _get_image_name_from_key(camera_key)

        # Collect frame lists and their source sizes (no encoding here).
        ep_frame_lists: list[list[Path]] = []
        ep_src_sizes_in_mb: list[float] = []
        for episode_index, num_frames, _, _, sampled_cameras in records:
            frames = sampled_cameras[camera_key]
            if len(frames) != num_frames:
                raise ValueError(
                    f"Camera '{camera_key}' episode {episode_index} has "
                    f"{len(frames)} video frames but the data table has "
                    f"{num_frames} frames; video/data are out of sync."
                )
            ep_frame_lists.append(frames)
            ep_src_sizes_in_mb.append(sum(f.stat().st_size for f in frames) / (1024**2))

        # Calibrate the compression ratio by encoding the first episode once
        # into a temporary file, mirroring the parquet size estimation above.
        compression_ratio = 1.0
        if ep_src_sizes_in_mb and ep_src_sizes_in_mb[0] > 0:
            with tempfile.NamedTemporaryFile(suffix=".mp4") as tmp:
                tmp_mp4 = Path(tmp.name)
                encode_mp4(ep_frame_lists[0], fps, tmp_mp4, verbose=False)
                compression_ratio = _get_file_size_in_mb(tmp_mp4) / ep_src_sizes_in_mb[0]

        # Pack episodes into file-XXX by estimated size. The ratio is refined from each packed file actually written.
        chunk_idx = 0
        file_idx = 0
        src_in_mb = 0.0
        frames_in_file = 0
        pending_frames: list[Path] = []
        ep_assignments: list[tuple[int, int, float, float]] = []

        for idx, frames in enumerate(
            tqdm(ep_frame_lists, desc=f"Encoding videos/{image_name}", unit="ep")
        ):
            est_size_in_mb = (src_in_mb + ep_src_sizes_in_mb[idx]) * compression_ratio

            if est_size_in_mb >= VIDEO_FILES_SIZE_IN_MB and pending_frames:
                out_path = output_dir / VIDEO_PATH.format(
                    video_key=image_name,
                    chunk_index=chunk_idx,
                    file_index=file_idx,
                )
                out_path.parent.mkdir(parents=True, exist_ok=True)
                encode_mp4(pending_frames, fps, out_path, verbose=False)
                if src_in_mb > 0:
                    compression_ratio = _get_file_size_in_mb(out_path) / src_in_mb

                chunk_idx, file_idx = _update_chunk_file_indices(chunk_idx, file_idx)
                src_in_mb = 0.0
                frames_in_file = 0
                pending_frames = []

            from_ts = frames_in_file / float(fps)
            to_ts = (frames_in_file + len(frames)) / float(fps)
            ep_assignments.append((chunk_idx, file_idx, from_ts, to_ts))
            pending_frames.extend(frames)
            src_in_mb += ep_src_sizes_in_mb[idx]
            frames_in_file += len(frames)

        if pending_frames:
            out_path = output_dir / VIDEO_PATH.format(
                video_key=image_name,
                chunk_index=chunk_idx,
                file_index=file_idx,
            )
            out_path.parent.mkdir(parents=True, exist_ok=True)
            encode_mp4(pending_frames, fps, out_path, verbose=False)

        for idx, (c, f, from_ts, to_ts) in enumerate(ep_assignments):
            episodes_video_meta[idx][f"videos/{image_name}/chunk_index"] = c
            episodes_video_meta[idx][f"videos/{image_name}/file_index"] = f
            episodes_video_meta[idx][f"videos/{image_name}/from_timestamp"] = from_ts
            episodes_video_meta[idx][f"videos/{image_name}/to_timestamp"] = to_ts

    return episodes_video_meta


def _calc_episode_stats_numpy(
    sampled_obs,
    sampled_actions,
    episode_index,
    gidx,
    task_index,
    fps,
    cameras,
):
    """Compute per-episode stats as numpy arrays for v3.0 episodes parquet."""
    length = len(sampled_obs)
    actions = np.vstack(sampled_actions).astype(np.float32)
    observations = np.vstack(sampled_obs).astype(np.float32)
    timestamps = np.arange(length, dtype=np.float64) / float(fps)

    stats: dict[str, np.ndarray] = {}

    for key, data in [("action", actions), ("observation.state", observations)]:
        desc = _describe_vector(data)
        for stat_name, value in desc.items():
            stats[f"{key}/{stat_name}"] = np.array(value)

    for key, data in [
        ("timestamp", timestamps),
        ("frame_index", np.arange(length, dtype=np.int64)),
        ("episode_index", np.full(length, episode_index, dtype=np.int64)),
        ("index", np.arange(gidx, gidx + length, dtype=np.int64)),
        ("task_index", np.full(length, task_index, dtype=np.int64)),
    ]:
        desc = _describe_scalar(data)
        for stat_name, value in desc.items():
            stats[f"{key}/{stat_name}"] = np.array(value)

    for cam_key, cam_paths in cameras.items():
        image_name = _get_image_name_from_key(cam_key)
        desc = _describe_images(cam_paths)
        for stat_name, value in desc.items():
            stats[f"{image_name}/{stat_name}"] = np.array(value)

    return stats


def _aggregate_feature_stats(stats_list):
    """Aggregate per-episode stats for a single feature key."""
    means = np.stack([s["mean"] for s in stats_list])
    variances = np.stack([s["std"] ** 2 for s in stats_list])
    counts = np.stack([s["count"] for s in stats_list])
    total_count = counts.sum(axis=0)

    while counts.ndim < means.ndim:
        counts = np.expand_dims(counts, axis=-1)

    weighted_means = means * counts
    total_mean = weighted_means.sum(axis=0) / total_count

    delta_means = means - total_mean
    weighted_variances = (variances + delta_means**2) * counts
    total_variance = weighted_variances.sum(axis=0) / total_count

    result = {
        "min": np.min(np.stack([s["min"] for s in stats_list]), axis=0),
        "max": np.max(np.stack([s["max"] for s in stats_list]), axis=0),
        "mean": total_mean,
        "std": np.sqrt(total_variance),
        "count": total_count,
    }

    quantile_keys = [k for k in stats_list[0] if k.startswith("q") and k[1:].isdigit()]
    for q_key in quantile_keys:
        if all(q_key in s for s in stats_list):
            q_vals = np.stack([s[q_key] for s in stats_list])
            result[q_key] = (q_vals * counts).sum(axis=0) / total_count

    return result


def _aggregate_stats(all_episode_stats):
    """Aggregate per-episode flat stats dicts into overall stats dict.

    Input: list of flat dicts like ``{"action/min": array, ...}``
    Output: nested dict like ``{"action": {"min": array, ...}, ...}``
    """
    if not all_episode_stats:
        return {}

    base_keys: set[str] = set()
    for ep_stats in all_episode_stats:
        for key in ep_stats:
            base_keys.add(key.rsplit("/", 1)[0])

    overall: dict[str, dict] = {}
    for base_key in sorted(base_keys):
        per_ep = []
        for ep_stats in all_episode_stats:
            entry: dict = {}
            for stat_name in ("min", "max", "mean", "std", "count"):
                full_key = f"{base_key}/{stat_name}"
                if full_key in ep_stats:
                    entry[stat_name] = ep_stats[full_key]
            quantile_keys = [
                k
                for k in ep_stats
                if k.startswith(f"{base_key}/q") and k[len(base_key) + 2 :].isdigit()
            ]
            for qk in quantile_keys:
                entry[qk.rsplit("/", 1)[1]] = ep_stats[qk]
            if "min" in entry:
                per_ep.append(entry)
        if per_ep:
            overall[base_key] = _aggregate_feature_stats(per_ep)

    return overall


def _serialize_stats(stats):
    """Convert numpy arrays in stats dict to JSON-serialisable form."""
    result = {}
    for key, value in stats.items():
        if isinstance(value, dict):
            result[key] = {
                k: v.tolist() if isinstance(v, np.ndarray) else v
                for k, v in value.items()
            }
        elif isinstance(value, np.ndarray):
            result[key] = value.tolist()
        else:
            result[key] = value
    return result


def _write_episodes_and_stats(
    dataset,
    records,
    output_dir,
    fps,
    remap_episode_index,
    remap_task_index,
    episodes_data_meta,
    episodes_video_meta,
):
    """Write episodes parquet with metadata + stats, and aggregated stats.json."""
    all_episode_dicts: list[dict] = []
    all_episode_stats: list[dict] = []
    gidx = 0

    for idx, (
        episode_index,
        num_frames,
        sampled_obs,
        sampled_actions,
        sampled_cameras,
    ) in enumerate(tqdm(records, desc="Computing episode stats", unit="ep")):
        lerobot_episode_index = remap_episode_index[episode_index]
        lerobot_task_index = remap_task_index[
            int(dataset.meta.episodes[episode_index]["task_index"])
        ]
        task_name = dataset.meta.data["tasks"][
            int(dataset.meta.episodes[episode_index]["task_index"])
        ]["prompt"]

        ep_dict: dict = {
            "episode_index": lerobot_episode_index,
            "tasks": [task_name],
            "length": num_frames,
            **episodes_data_meta[idx],
            "meta/episodes/chunk_index": 0,
            "meta/episodes/file_index": 0,
        }

        if episodes_video_meta:
            for k, v in episodes_video_meta[idx].items():
                ep_dict[k] = v

        ep_stats = _calc_episode_stats_numpy(
            sampled_obs,
            sampled_actions,
            lerobot_episode_index,
            gidx,
            lerobot_task_index,
            fps,
            sampled_cameras,
        )

        for stat_key, stat_value in ep_stats.items():
            ep_dict[f"stats/{stat_key}"] = stat_value

        all_episode_dicts.append(ep_dict)
        all_episode_stats.append(ep_stats)
        gidx += num_frames

    for ep_dict in all_episode_dicts:
        for k, v in ep_dict.items():
            if isinstance(v, np.ndarray):
                ep_dict[k] = v.tolist()

    episodes_path = output_dir / EPISODES_PATH.format(chunk_index=0, file_index=0)
    episodes_path.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(all_episode_dicts)
    df.to_parquet(episodes_path, index=False)

    overall_stats = _aggregate_stats(all_episode_stats)
    stats_path = output_dir / STATS_PATH
    stats_path.parent.mkdir(parents=True, exist_ok=True)
    with stats_path.open("w", encoding="utf-8") as f:
        json.dump(_serialize_stats(overall_stats), f, ensure_ascii=False, indent=4)


def _write_tasks_parquet(dataset, remap_task_index, output_dir):
    """Write tasks as parquet with ``task`` index and ``task_index`` column."""
    tasks_sorted = sorted(remap_task_index.items(), key=lambda kv: kv[1])
    task_names = []
    task_indices = []
    for original_task_index, new_task_index in tasks_sorted:
        task_name = dataset.meta.data["tasks"][original_task_index]["prompt"]
        task_names.append(task_name)
        task_indices.append(new_task_index)

    df = pd.DataFrame(
        {"task_index": task_indices},
        index=pd.Index(task_names, name="task"),
    )
    tasks_path = output_dir / TASKS_PATH
    tasks_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(tasks_path)


def _write_info_json(
    dataset,
    records,
    output_dir,
    fps,
    train_split,
    joint_names,
    total_frames,
    remap_task_index,
):
    """Write v3.0 info.json."""
    features = {
        "action": {
            "dtype": "float32",
            "names": joint_names,
            "shape": [len(joint_names)],
            "fps": fps,
        },
        "observation.state": {
            "dtype": "float32",
            "names": joint_names,
            "shape": [len(joint_names)],
            "fps": fps,
        },
        "timestamp": {"dtype": "float64", "shape": [1], "names": None, "fps": fps},
        "frame_index": {"dtype": "int64", "shape": [1], "names": None, "fps": fps},
        "episode_index": {"dtype": "int64", "shape": [1], "names": None, "fps": fps},
        "index": {"dtype": "int64", "shape": [1], "names": None, "fps": fps},
        "task_index": {"dtype": "int64", "shape": [1], "names": None, "fps": fps},
        "success": {"dtype": "int64", "shape": [1], "names": None, "fps": fps},
        "last_frame_index": {
            "dtype": "int64",
            "shape": [1],
            "names": None,
            "fps": fps,
        },
    }

    first_episode_index = records[0][0]
    sample_record = dataset.sample(
        hz=fps, episode=dataset.meta.episodes[first_episode_index]
    )[0]
    for cam in dataset.camera_names:
        sample_image = sample_record.cameras[cam].load()
        h, w = sample_image.shape[:2]
        features[_get_image_name_from_key(cam)] = {
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
    train_end = round(num_episodes * train_split)
    splits = {"train": f"0:{train_end}"}
    if train_end < num_episodes:
        splits["val"] = f"{train_end}:{num_episodes}"

    info = {
        "codebase_version": CODEBASE_VERSION,
        "robot_type": ROBOT_TYPE,
        "total_episodes": num_episodes,
        "total_frames": total_frames,
        "total_tasks": len(
            {
                remap_task_index[int(dataset.meta.episodes[ep_idx]["task_index"])]
                for ep_idx, *_ in records
            }
        )
        if records
        else 0,
        "chunks_size": CHUNK_SIZE,
        "data_files_size_in_mb": DATA_FILES_SIZE_IN_MB,
        "video_files_size_in_mb": VIDEO_FILES_SIZE_IN_MB,
        "fps": fps,
        "splits": splits,
        "data_path": DATA_PATH,
        "video_path": VIDEO_PATH,
        "features": features,
    }

    info_path = output_dir / INFO_PATH
    info_path.parent.mkdir(parents=True, exist_ok=True)
    with info_path.open("w", encoding="utf-8") as f:
        json.dump(info, f, ensure_ascii=False, indent=4)


def to_lerobotv30(
    dataset: Dataset,
    output_dir: str | Path,
    fps: int = 30,
    train_split: float = 0.8,
    smoothing_cutoff: float = 1.0,
    success_only: bool = False,
) -> None:
    """Convert the given dataset to LeRobot v3.0 format."""
    if not (0.0 <= train_split <= 1.0):
        raise ValueError(f"train_split must be between 0 and 1, got {train_split}")
    if fps <= 0:
        raise ValueError(f"fps must be a positive integer, got {fps}")

    dataset.set_smoothing(cutoff=smoothing_cutoff)
    output_dir = Path(output_dir)

    joint_keys, joint_names = _collect_keys_and_joint_names(dataset)
    records = _collect_downsampled_data(dataset, fps, joint_keys, success_only)

    if not records:
        raise ValueError("No episodes to write.")

    remap_episode_index, remap_task_index = _build_remaps(dataset, records)

    episodes_data_meta, total_frames = _write_packed_parquet(
        dataset, records, output_dir, fps, remap_episode_index, remap_task_index
    )

    episodes_video_meta = _write_packed_videos(
        dataset, records, output_dir, fps, remap_episode_index
    )

    _write_episodes_and_stats(
        dataset,
        records,
        output_dir,
        fps,
        remap_episode_index,
        remap_task_index,
        episodes_data_meta,
        episodes_video_meta,
    )
    _write_tasks_parquet(dataset, remap_task_index, output_dir)
    _write_info_json(
        dataset,
        records,
        output_dir,
        fps,
        train_split,
        joint_names,
        total_frames,
        remap_task_index,
    )
