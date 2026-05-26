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
    [--format {openarm,lerobot_v2.1}] \
    [--fps INT]                # default 30 (lerobot only) \
    [--smoothing-cutoff FLOAT] # default 1.0 (lerobot only) \
    [--train-split FLOAT]      # default 0.8 (lerobot only) \
    [--success-only]           # lerobot only
```

The `--fps`, `--smoothing-cutoff`, `--train-split`, and `--success-only`
flags apply only when `--format lerobot_v2.1`.

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
