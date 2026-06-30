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

"""Rotation utilities for OpenArm Cartesian pose data (numpy / scipy).

Quaternion convention throughout: xyzw (scipy convention).

Functions
---------
quat_to_rot6d   : [..., 4] xyzw  →  [..., 6]   rot6d
rot6d_to_rotmat : [..., 6]       →  [..., 3, 3] rotation matrix (Gram-Schmidt)
rot6d_to_quat   : [..., 6]       →  [..., 4]    xyzw quaternion
pose_to_vec     : [..., 7]       →  [..., 9]    xyz + rot6d
vec_to_pose     : [..., 9]       →  [..., 7]    xyz + xyzw quat
"""

from __future__ import annotations

import numpy as np
from scipy.spatial.transform import Rotation


def quat_to_rot6d(quat: np.ndarray) -> np.ndarray:
    """[..., 4] xyzw → [..., 6] rot6d (first two columns of rotation matrix)."""
    R = Rotation.from_quat(quat).as_matrix()
    return np.concatenate([R[..., :, 0], R[..., :, 1]], axis=-1)


def rot6d_to_rotmat(rot6d: np.ndarray) -> np.ndarray:
    """[..., 6] → [..., 3, 3] via Gram-Schmidt orthonormalization."""
    a1 = rot6d[..., :3]
    a2 = rot6d[..., 3:]
    b1 = a1 / (np.linalg.norm(a1, axis=-1, keepdims=True) + 1e-8)
    b2 = a2 - (b1 * a2).sum(-1, keepdims=True) * b1
    b2 = b2 / (np.linalg.norm(b2, axis=-1, keepdims=True) + 1e-8)
    b3 = np.cross(b1, b2)
    return np.stack([b1, b2, b3], axis=-1)


def rot6d_to_quat(rot6d: np.ndarray) -> np.ndarray:
    """[..., 6] → [..., 4] xyzw quaternion."""
    return Rotation.from_matrix(rot6d_to_rotmat(rot6d)).as_quat()


def pose_to_vec(pose: np.ndarray) -> np.ndarray:
    """[..., 7] (xyz + xyzw quat) → [..., 9] (xyz + rot6d)."""
    return np.concatenate([pose[..., :3], quat_to_rot6d(pose[..., 3:])], axis=-1)


def vec_to_pose(vec: np.ndarray) -> np.ndarray:
    """[..., 9] (xyz + rot6d) → [..., 7] (xyz + xyzw quat)."""
    return np.concatenate([vec[..., :3], rot6d_to_quat(vec[..., 3:])], axis=-1)
