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
>>> dataset = openarm_dataset.Dataset("tests/data/dataset")
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
>>> obs.keys()
dict_keys(['arms/right_arm/qpos', 'arms/left_arm/qpos'])
>>> obs["arms/right_arm/qpos"]
                                 joint1    joint2    joint3    joint4    joint5    joint6    joint7   gripper
timestamp                                                                                                    
2026-02-25 09:04:11.614229214 -0.039352  0.989118 -0.051771  0.735691  0.077740 -0.070724  0.079488 -0.124674
2026-02-25 09:04:11.618732974 -0.039352  0.989118 -0.051771  0.735691  0.077740 -0.070724  0.079488 -0.124674
...                                 ...       ...       ...       ...       ...       ...       ...       ...
2026-02-25 09:04:14.597666675 -0.296583  0.885962 -0.192270  0.972567  0.194248  0.101626 -0.221057  0.022409

[746 rows x 8 columns]
>>> action = dataset.load_action(0, use_unixtime=True)
>>> action.keys()
dict_keys(['arms/right_arm/qpos', 'arms/left_arm/qpos'])
>>> action["arms/right_arm/qpos"]
                joint1    joint2    joint3    joint4    joint5    joint6    joint7   gripper
timestamp                                                                                   
1.772010e+09 -0.039352  0.989118 -0.051771  0.735691  0.077740 -0.070724  0.079488 -0.124674
1.772010e+09  0.030980  0.991799 -0.166579  0.969511  0.014409  0.143491 -0.189803  0.082215
...                ...       ...       ...       ...       ...       ...       ...       ...
1.772010e+09 -0.007582  1.088525 -0.104895  0.856318  0.134566  0.039683  0.109483 -0.003687

[90 rows x 8 columns]
```

Camera:

```python
>>> cameras = dataset.load_cameras(0)
>>> cameras.keys()
dict_keys(['left_wrist', 'right_wrist', 'ceiling', 'head'])
>>> cam_head = cameras["head"]
>>> cam_head.num_frames
3
>>> cam_head.load_timestamps()
[1772010251.6187909, 1772010251.629775, 1772010251.6634612]
>>> cam_head.load_frame(0)
(1772010251.6187909, array([[[ 75, 144,  90],
        [133, 216, 128],
        ...,
        [132,  54, 199]],

       ...,

       [[ 90, 146, 117],
        [ 98, 134, 122],
        ...,
        [ 89, 162, 155]]], shape=(600, 960, 3), dtype=uint8))
>>> cam_head.iter_frames()
<generator object CameraData.iter_frames at 0x72a24b36fe60>
>>> cam_head.iter_files()
<generator object CameraData.iter_files at 0x72a24b0289e0>
```

Sampling:

```python
>>> samples = dataset.sample(hz=30, episode_index=0)
>>> samples
[Sample(timestamp=1772010251.6202147), Sample(timestamp=1772010251.653548)]
>>> samples[0].timestamp
np.float64(1772010251.6202147)
>>> samples[0].obs
{'arms/right_arm/qpos': array([-0.0393523 ,  0.9891182 , -0.05177076,  0.7356907 ,  0.07774002,
       -0.07072392,  0.07948788, -0.1246737 ], dtype=float32), 'arms/left_arm/qpos': array([-0.1239887 , -1.0022309 , -0.23028165,  1.0189891 , -0.11319982,
        0.0516983 , -0.1742104 , -0.04307283], dtype=float32)}
>>> samples[0].action
{'arms/right_arm/qpos': array([ 0.03098021,  0.991799  , -0.16657865,  0.96951085,  0.01440866,
        0.14349142, -0.18980259,  0.08221525], dtype=float32), 'arms/left_arm/qpos': array([ 0.1032669 , -0.86291695,  0.14351352,  0.9478229 ,  0.18431091,
        0.00171096,  0.03923181,  0.11910774], dtype=float32)}
>>> [(name, image.shape) for name, image in samples[0].cameras.items()]
[('left_wrist', (600, 960, 3)), ('right_wrist', (600, 960, 3)), ('ceiling', (600, 960, 3)), ('head', (600, 960, 3))]

>>> samples = dataset.sample(hz=30, episode_index=0)
>>> samples[0].cameras.items()
dict_items([('left_wrist', PosixPath('1772010251620214727.jpeg')), ('right_wrist', PosixPath('1772010251628789283.jpeg')), ('ceiling', PosixPath('1772010251629083055.jpeg')), ('head', PosixPath('1772010251629774985.jpeg'))])
```

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
