# OpenArm Dataset

## Quick start

### Install

```bash
pip install openarm_dataset
```

### Sample usage

Basic:

```python
>>> import openarm_dataset
>>> dataset = openarm_dataset.Dataset("tests/fixture/dataset_0.3.0")
>>> dataset.meta.episodes
[{'id': '0', 'success': False, 'task_index': 0}, {'id': '3', 'success': True, 'task_index': 0}]
>>> dataset.meta.tasks
[{'prompt': 'Run test.', 'description': 'Longer task description if need.'}]
>>> dataset.num_episodes
2
```

Obs/Action:

```python
>>> obs = dataset.load_obs(0)
>>> list(obs.keys())
['arms/right/qpos', 'arms/right/qvel', 'arms/right/qtorque', 'arms/left/qpos', 'arms/left/qvel', 'arms/left/qtorque', 'lifter/elevation']
>>> obs["arms/right/qpos"].shape
(746, 8)
>>> obs["arms/right/qpos"].head(2)
                                 joint1    joint2    joint3    joint4    joint5    joint6    joint7   gripper
timestamp
2026-02-25 09:04:11.614229214 -0.039352  0.989118 -0.051771  0.735691  0.077740 -0.070724  0.079488 -0.124674
2026-02-25 09:04:11.618732974 -0.039352  0.989118 -0.051771  0.735691  0.077740 -0.070724  0.079488 -0.124674

>>> action = dataset.load_action(0, use_unixtime=True)
>>> list(action.keys())
['arms/right/qpos', 'arms/left/qpos', 'lifter/elevation']
>>> action["arms/right/qpos"].shape
(90, 8)
```

The available keys reflect what was actually recorded: arm data may contain any
of `qpos`/`qvel`/`qtorque`/`pose`, and embodiments whose files were not
recorded (e.g. a declared but unused lifter) are simply absent. Datasets
recorded with Cartesian teleoperation expose an 8-dim `pose` action
(position + quaternion + gripper) instead of `qpos`:

```python
>>> dataset = openarm_dataset.Dataset("tests/fixture/dataset_0.4.0_pose")
>>> action = dataset.load_action(0)
>>> list(action.keys())
['arms/right/pose', 'arms/left/pose', 'lifter/elevation']
>>> list(action["arms/right/pose"].columns)
['x', 'y', 'z', 'qw', 'qx', 'qy', 'qz', 'gripper']
```

The dataset itself only ever stores the raw recorded representation. To use a
specific representation, pass `state=` to `load_obs`/`load_action`/`sample`:
recorded data matching the request is returned as-is, and anything else is
converted on the fly with
[openarm_control](https://github.com/enactic/openarm_control) kinematics —
`qpos` to `pose` via FK, `pose` to `qpos` via IK (seeded from the episode's
recorded obs qpos and tracked along the pose trajectory; a warning reports
samples where the solver failed or did not converge). `rot6d`
(position + the first two rotation matrix columns + gripper, 10-dim) is
derived from the pose:

```python
>>> action = dataset.load_action(0, state="qpos")  # IK from the recorded pose
>>> list(action.keys())
['arms/right/qpos', 'arms/left/qpos', 'lifter/elevation']
>>> obs = dataset.load_obs(0, state="rot6d")       # FK, then 6D rotation
>>> list(obs["arms/right/rot6d"].columns)
['x', 'y', 'z', 'r11', 'r21', 'r31', 'r12', 'r22', 'r32', 'gripper']
```

Recorded `pose` data must be end-effector poses in the MuJoCo model's world
frame (the openarm_control Cartesian-teleop convention); the kinematics use
the model at its home keyframe, and time-varying lifter elevation is not
folded in. The IK runs with openarm_control's real-time teleoperation
shaping (bounded per-solve motion, singularity braking, nullspace home
regulation) disabled so that every recorded pose is solved to convergence.
Pass `kinematics=` to `Dataset` to override the engines built with
`openarm_dataset.kinematics.create_engines()` (e.g. for a different scene
or IK tolerances).

Camera:

```python
>>> cameras = dataset.load_cameras(0)
>>> list(cameras.keys())
['wrist_left', 'wrist_right', 'ceiling', 'head']
>>> cam_head = cameras["head"]
>>> cam_head.num_frames
3
>>> cam_head.load_timestamps()
[1772010251.6187909, 1772010251.629775, 1772010251.6634612]
>>> frame = cam_head.get_frame(0)
>>> frame.timestamp
1772010251.6187909
>>> frame.path
PosixPath('.../head/1772010251618790832.jpeg')
>>> frame.load().shape
(600, 960, 3)
>>> for frame in cam_head.frames():
...     pass  # iterate over Frame objects
```

A camera's frames may be stored as individual timestamped JPEG files in a
directory (`episodes/0/cameras/head/<timestamp>.jpeg`), packed into a single
uncompressed tar archive (`episodes/0/cameras/head.tar`), or encoded into a
single H.264 video (`episodes/0/cameras/head.mp4`). Packing keeps the file
count low enough for [Hugging Face Hub's storage
recommendations](https://huggingface.co/docs/hub/storage-limits#recommendations);
tar is lossless, while mp4 is lossy but typically 10–20× smaller. The mp4 is
self-contained: the exact nanosecond timestamp of every frame is stored in the
video's metadata (`openarm_timestamps`), and the frames are also timed so the
video plays back at the recorded pace in any player.
All layouts expose the same API shown above. For tar- and mp4-backed cameras,
`frame.path` is a synthetic `.../head.tar/<timestamp>.jpeg` (or
`.../head.mp4/<timestamp>.jpeg`) path that locates the image inside the
archive — it is not a real file, so use `frame.load()` or `frame.open_image()`
to access the image data. Reading mp4 frames in order reuses one decoder;
random access seeks to the nearest keyframe (every 30 frames) first.

Sampling:

```python
>>> samples = dataset.sample(hz=30, episode_index=0)
>>> samples
[Sample(timestamp=1772010251.6202147), Sample(timestamp=1772010251.653548)]
>>> samples[0].timestamp
1772010251.6202147
>>> samples[0].obs["arms/right/qpos"]
array([-0.0393523 ,  0.9891182 , -0.05177076,  0.7356907 ,  0.07774002,
       -0.07072392,  0.07948788, -0.1246737 ], dtype=float32)
>>> samples[0].action["arms/right/qpos"]
array([ 0.03098021,  0.991799  , -0.16657865,  0.96951085,  0.01440866,
        0.14349142, -0.18980259,  0.08221525], dtype=float32)
>>> {name: frame.load().shape for name, frame in samples[0].cameras.items()}
{'wrist_left': (600, 960, 3), 'wrist_right': (600, 960, 3), 'ceiling': (600, 960, 3), 'head': (600, 960, 3)}
```

## Dataset format (v0.4.0)

The current on-disk format written by `Dataset.write()` (and by the
`dora-openarm-dataset-recorder`):

```
<root>/
  metadata.yaml                      # version: "0.4.0"
  episodes/<id>/
    obs/arms/<side>/state.parquet    # timestamp + any of qpos/qvel/qtorque/pose
    action/arms/<side>/state.parquet # timestamp + qpos or pose
    obs/lifter/elevation.parquet     # timestamp + value (only if recorded)
    action/lifter/elevation.parquet  # timestamp + value (only if recorded)
    cameras/<name>/<timestamp>.jpeg  # or cameras/<name>.tar or cameras/<name>.mp4
```

`state.parquet` is self-describing: its non-timestamp columns (each a list of
floats per frame) define the available attributes. `qpos`/`qvel`/`qtorque`
have one entry per joint; `pose` is 8-dim
(`x, y, z, qw, qx, qy, qz, gripper`). Older formats (0.1.0–0.3.0 and
unversioned) are still read transparently, and `Dataset.write()` always
produces v0.4.0.

## Command-line tools

Validate a dataset:

```bash
openarm-dataset-validate <input> \
    [--no-update-metadata]         # do not record per-episode validity in the metadata
    [--qpos-jump-threshold RADIAN] # default 1.0
    [--qpos-absmax RADIAN]         # default 6.28
    [--min-duration SECOND]        # default 2.0
```

Every episode is checked for `null` and `NaN` values. In addition,
`--qpos-absmax` flags `qpos` values whose absolute value exceeds the
threshold, `--qpos-jump-threshold` flags `qpos` frame-to-frame deltas above
the threshold as abrupt jumps, and `--min-duration` flags episodes shorter
than the given duration. The three thresholds are checked against the
recorded values (smoothing is not applied) and each is disabled by passing
`none`, e.g. `--min-duration none`. Files that include `null` or `NaN` are
reported but not checked against the `qpos` thresholds.

Exits with status `1` if any errors are reported. The result is also recorded
per episode as a boolean `valid` flag in `metadata.yaml` unless
`--no-update-metadata` is given. Recording is not supported for unversioned
datasets. Episodes marked `valid: false` can be excluded from conversion with
`openarm-dataset-convert --valid-only`.

Repair a dataset:

```bash
openarm-dataset-repair <input> \
    [-o <output>]    # write the repaired dataset here; repairs in place if omitted
```

Fills isolated single-frame gaps (a `null` or `NaN` in a `qpos`/`qvel`/
`qtorque`/`pose`/`value` array) by averaging the immediately preceding and following
frame values, per array element. Gaps spanning two or more consecutive frames,
and gaps at the first or last frame, cannot be averaged and are left untouched
with a warning on stderr. The command always exits with status `0`; run
`openarm-dataset-validate` afterwards to confirm the result.

Merge multiple datasets:

```bash
openarm-dataset-merge <input1> <input2> [<input3> ...] \
    -o <output>    \
    [--symlink]    # create symlinks instead of copying episode data
```

All input datasets must have the same version, equipment, and frequencies.
Tasks are deduplicated by prompt: identical prompts are treated as the same
task. Episodes are renumbered sequentially starting from 0.

Convert a dataset:

```bash
openarm-dataset-convert <input> <output> \
    [--format {openarm,lerobot_v2.1,lerobot_v3.0,gr00t}] \
    [--camera-format {dir,tar,mp4}] # default dir (openarm only); tar packs each \
                                # camera into one .tar archive, mp4 encodes it \
                                # into one H.264 video (lossy) \
    [--fps INT]                # default 30 (lerobot/gr00t only) \
    [--smoothing-cutoff FLOAT] # default 1.0 (lerobot/gr00t only) \
    [--train-split FLOAT]      # default 0.8 (lerobot/gr00t only) \
    [--success-only]           # lerobot/gr00t only \
    [--valid-only]             # exclude episodes marked invalid \
    [--state {qpos,pose,rot6d}] # default qpos (lerobot/gr00t only)
```

The `--fps`, `--smoothing-cutoff`, `--train-split`, `--success-only`, and
`--state` flags apply only when `--format lerobot_v2.1`, `--format
lerobot_v3.0`, or `--format gr00t`.
The `--valid-only` flag applies to every output format and excludes episodes
marked `valid: false` by `openarm-dataset-validate`; episodes without the
flag are treated as valid. If no episode has a `valid` flag (i.e. the dataset
has not been validated yet), a warning is printed on stderr and nothing is
excluded.
The `gr00t` format produces a LeRobot v2.1 dataset plus a GR00T-compatible
`meta/modality.json` (see [Isaac-GR00T data preparation](https://github.com/NVIDIA/Isaac-GR00T/blob/main/getting_started/data_preparation.md)).

`--state` selects the arm state representation of the LeRobot output
(default: `qpos`), converting on the fly (see the `state=` API above) when
the dataset stores the other representation. Every arm stream is exported
in this one representation, regardless of what was recorded.
The converted values are only written to the LeRobot output; the OpenArm
dataset itself always keeps the raw recorded data.

Upload a dataset to the Hugging Face Hub:

```bash
openarm-dataset-upload <input> \
    --repo-id <user>/<dataset> \
    [--private]                # create the repo as private if it does not exist
```

The whole dataset directory is uploaded to a
[dataset repository](https://huggingface.co/docs/hub/datasets), creating it if it
does not already exist, and tagged with the dataset version. Cameras stored as
directories of JPEG files are repacked **in place** into one `.tar` archive per
camera before uploading, to stay within [Hugging Face Hub's file-count
recommendations](https://huggingface.co/docs/hub/storage-limits#recommendations).
Repacking is lossless and reversible (`openarm-dataset-convert --camera-format dir`
restores the JPEG-directory layout). Cameras already packed as `.tar` or `.mp4`
are uploaded as they are; only `.tar` cameras are browsable in the Hub's dataset
viewer.

## Development

### Test

```bash
uv sync
uv run pytest
```

## Related links

<!-- - 📚 Read the [documentation](https://docs.openarm.dev/software/dataset/) -->
- 💬 Join the community on [Discord](https://discord.gg/FsZaZ4z3We)
- 📬 Contact us through <openarm@enactic.ai>

## License

Licensed under the Apache License 2.0. See [LICENSE.txt](LICENSE.txt) for details.

Copyright 2026 Enactic, Inc.

## Code of Conduct

All participation in the OpenArm project is governed by our [Code of Conduct](CODE_OF_CONDUCT.md).
