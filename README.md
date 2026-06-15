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

A camera's frames may be stored either as individual timestamped JPEG files in a
directory (`episodes/0/cameras/head/<timestamp>.jpeg`) or packed into a single
uncompressed tar archive (`episodes/0/cameras/head.tar`). Packing keeps the file
count low enough for [Hugging Face Hub's storage
recommendations](https://huggingface.co/docs/hub/storage-limits#recommendations).
Both layouts expose the same API shown above. For tar-backed cameras, `frame.path`
is a synthetic `.../head.tar/<timestamp>.jpeg` path that locates the image inside
the archive — it is not a real file, so use `frame.load()` or `frame.read_bytes()`
to access the image data.

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

## Command-line tools

Validate a dataset:

```bash
openarm-dataset-validate <input>
```

Exits with status `1` if any errors are reported.

Repair a dataset:

```bash
openarm-dataset-repair <input> \
    [-o <output>]    # write the repaired dataset here; repairs in place if omitted
```

Fills isolated single-frame gaps (a `null` or `NaN` in a `qpos`/`qvel`/
`qtorque`/`value` array) by averaging the immediately preceding and following
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
    [--format {openarm,lerobot_v2.1,gr00t}] \
    [--camera-format {dir,tar}] # default dir (openarm only); tar packs each \
                                # camera into one .tar archive \
    [--fps INT]                # default 30 (lerobot/gr00t only) \
    [--smoothing-cutoff FLOAT] # default 1.0 (lerobot/gr00t only) \
    [--train-split FLOAT]      # default 0.8 (lerobot/gr00t only) \
    [--success-only]           # lerobot/gr00t only
```

The `--fps`, `--smoothing-cutoff`, `--train-split`, and `--success-only`
flags apply only when `--format lerobot_v2.1` or `--format gr00t`.
The `gr00t` format produces a LeRobot v2.1 dataset plus a GR00T-compatible
`meta/modality.json` (see [Isaac-GR00T data preparation](https://github.com/NVIDIA/Isaac-GR00T/blob/main/getting_started/data_preparation.md)).

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
