"""3D Mathematics Library for Physics and Rendering.

This module provides 3D math types for physics simulation and rendering:

- Vec3: 3D vector for positions, velocities, and directions
- Quat: Unit quaternion for 3D rotations
- Mat3: 3x3 rotation/linear transformation matrix
- Mat4: 4x4 affine transformation matrix (rotation + translation + scale)

Example:
    ```mojo
    from math3d import Vec3, Quat, Mat3, Mat4

    # Create a position
    var position = Vec3(1.0, 2.0, 3.0)

    # Create a rotation (45 degrees around Y axis)
    var rotation = Quat.from_axis_angle(Vec3.unit_y(), 0.785)

    # Rotate the position
    var rotated = rotation.rotate_vec(position)

    # Create a full 4x4 transform
    var transform = Mat4.compose(
        Vec3(0.0, 1.0, 0.0),  # translation
        rotation,             # rotation
        Vec3.one(),          # scale
    )
    ```
"""

from .vec3 import Vec3, vec3, dot, cross, normalize, length, distance, lerp
from .quat import Quat, quat_identity, quat_from_axis_angle, slerp
from .mat3 import Mat3, mat3_identity, mat3_rotation_x, mat3_rotation_y, mat3_rotation_z
from .mat4 import (
    Mat4,
    mat4_identity,
    mat4_translation,
    mat4_look_at,
    mat4_perspective,
)
