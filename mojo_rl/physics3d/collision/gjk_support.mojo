"""GJK support functions for all geometry types.

A support function returns the point on a convex shape's surface that is
furthest along a given direction. These are the building blocks of GJK/EPA.

Reference: MuJoCo engine_collision_convex.c lines 162-398
"""

from std.math import sqrt
from ..constants import (
    GEOM_SPHERE,
    GEOM_CAPSULE,
    GEOM_BOX,
    GEOM_CYLINDER,
    GEOM_MESH,
)
from ..kinematics.quat_math import quat_rotate, quat_rotate_inverse


@always_inline
def support_sphere[
    DTYPE: DType,
](
    dir_x: Scalar[DTYPE],
    dir_y: Scalar[DTYPE],
    dir_z: Scalar[DTYPE],
    pos_x: Scalar[DTYPE],
    pos_y: Scalar[DTYPE],
    pos_z: Scalar[DTYPE],
    radius: Scalar[DTYPE],
) -> InlineArray[Scalar[DTYPE], 3]:
    """Support point on sphere: center + radius * dir."""
    var result = InlineArray[Scalar[DTYPE], 3](fill=Scalar[DTYPE](0))
    result[0] = pos_x + radius * dir_x
    result[1] = pos_y + radius * dir_y
    result[2] = pos_z + radius * dir_z
    return result


@always_inline
def support_capsule[
    DTYPE: DType,
](
    dir_x: Scalar[DTYPE],
    dir_y: Scalar[DTYPE],
    dir_z: Scalar[DTYPE],
    pos_x: Scalar[DTYPE],
    pos_y: Scalar[DTYPE],
    pos_z: Scalar[DTYPE],
    qx: Scalar[DTYPE],
    qy: Scalar[DTYPE],
    qz: Scalar[DTYPE],
    qw: Scalar[DTYPE],
    radius: Scalar[DTYPE],
    half_length: Scalar[DTYPE],
) -> InlineArray[Scalar[DTYPE], 3]:
    """Support point on capsule: choose endpoint closest to dir, then sphere support."""
    # Capsule axis in world frame (local z-axis rotated by quaternion)
    var axis = quat_rotate[DTYPE](qx, qy, qz, qw,
        Scalar[DTYPE](0), Scalar[DTYPE](0), Scalar[DTYPE](1))
    var ax = axis[0]
    var ay = axis[1]
    var az = axis[2]

    # Select endpoint: positive or negative half-length based on dot(dir, axis)
    var dot = dir_x * ax + dir_y * ay + dir_z * az
    var sign = Scalar[DTYPE](1) if dot >= 0 else Scalar[DTYPE](-1)

    # Endpoint center
    var ex = pos_x + sign * half_length * ax
    var ey = pos_y + sign * half_length * ay
    var ez = pos_z + sign * half_length * az

    # Sphere support at endpoint
    var result = InlineArray[Scalar[DTYPE], 3](fill=Scalar[DTYPE](0))
    result[0] = ex + radius * dir_x
    result[1] = ey + radius * dir_y
    result[2] = ez + radius * dir_z
    return result


@always_inline
def support_box[
    DTYPE: DType,
](
    dir_x: Scalar[DTYPE],
    dir_y: Scalar[DTYPE],
    dir_z: Scalar[DTYPE],
    pos_x: Scalar[DTYPE],
    pos_y: Scalar[DTYPE],
    pos_z: Scalar[DTYPE],
    qx: Scalar[DTYPE],
    qy: Scalar[DTYPE],
    qz: Scalar[DTYPE],
    qw: Scalar[DTYPE],
    half_x: Scalar[DTYPE],
    half_y: Scalar[DTYPE],
    half_z: Scalar[DTYPE],
) -> InlineArray[Scalar[DTYPE], 3]:
    """Support point on OBB: pick vertex furthest along dir."""
    # Box axes in world frame (columns of rotation matrix from quaternion)
    var ax = quat_rotate[DTYPE](qx, qy, qz, qw,
        Scalar[DTYPE](1), Scalar[DTYPE](0), Scalar[DTYPE](0))
    var ay = quat_rotate[DTYPE](qx, qy, qz, qw,
        Scalar[DTYPE](0), Scalar[DTYPE](1), Scalar[DTYPE](0))
    var az = quat_rotate[DTYPE](qx, qy, qz, qw,
        Scalar[DTYPE](0), Scalar[DTYPE](0), Scalar[DTYPE](1))

    # Sign of projection onto each axis
    var sx = Scalar[DTYPE](1) if (dir_x * ax[0] + dir_y * ax[1] + dir_z * ax[2]) >= 0 else Scalar[DTYPE](-1)
    var sy = Scalar[DTYPE](1) if (dir_x * ay[0] + dir_y * ay[1] + dir_z * ay[2]) >= 0 else Scalar[DTYPE](-1)
    var sz = Scalar[DTYPE](1) if (dir_x * az[0] + dir_y * az[1] + dir_z * az[2]) >= 0 else Scalar[DTYPE](-1)

    var result = InlineArray[Scalar[DTYPE], 3](fill=Scalar[DTYPE](0))
    result[0] = pos_x + sx * half_x * ax[0] + sy * half_y * ay[0] + sz * half_z * az[0]
    result[1] = pos_y + sx * half_x * ax[1] + sy * half_y * ay[1] + sz * half_z * az[1]
    result[2] = pos_z + sx * half_x * ax[2] + sy * half_y * ay[2] + sz * half_z * az[2]
    return result


@always_inline
def support_cylinder[
    DTYPE: DType,
](
    dir_x: Scalar[DTYPE],
    dir_y: Scalar[DTYPE],
    dir_z: Scalar[DTYPE],
    pos_x: Scalar[DTYPE],
    pos_y: Scalar[DTYPE],
    pos_z: Scalar[DTYPE],
    qx: Scalar[DTYPE],
    qy: Scalar[DTYPE],
    qz: Scalar[DTYPE],
    qw: Scalar[DTYPE],
    radius: Scalar[DTYPE],
    half_length: Scalar[DTYPE],
) -> InlineArray[Scalar[DTYPE], 3]:
    """Support point on cylinder: disk rim + endpoint selection."""
    # Cylinder axis in world frame (local z-axis)
    var axis = quat_rotate[DTYPE](qx, qy, qz, qw,
        Scalar[DTYPE](0), Scalar[DTYPE](0), Scalar[DTYPE](1))
    var ax = axis[0]
    var ay = axis[1]
    var az = axis[2]

    # Select endpoint
    var dot_axis = dir_x * ax + dir_y * ay + dir_z * az
    var sign = Scalar[DTYPE](1) if dot_axis >= 0 else Scalar[DTYPE](-1)
    var cx = pos_x + sign * half_length * ax
    var cy = pos_y + sign * half_length * ay
    var cz = pos_z + sign * half_length * az

    # Project dir onto the disk plane (perpendicular to axis)
    var perp_x = dir_x - dot_axis * ax
    var perp_y = dir_y - dot_axis * ay
    var perp_z = dir_z - dot_axis * az
    var perp_len = sqrt(perp_x * perp_x + perp_y * perp_y + perp_z * perp_z)

    var result = InlineArray[Scalar[DTYPE], 3](fill=Scalar[DTYPE](0))
    if perp_len > Scalar[DTYPE](1e-10):
        var inv_perp = radius / perp_len
        result[0] = cx + perp_x * inv_perp
        result[1] = cy + perp_y * inv_perp
        result[2] = cz + perp_z * inv_perp
    else:
        # dir is parallel to axis — any rim point works
        var local_x = quat_rotate[DTYPE](qx, qy, qz, qw,
            Scalar[DTYPE](1), Scalar[DTYPE](0), Scalar[DTYPE](0))
        result[0] = cx + radius * local_x[0]
        result[1] = cy + radius * local_x[1]
        result[2] = cz + radius * local_x[2]
    return result


@always_inline
def support_mesh[
    DTYPE: DType,
](
    dir_x: Scalar[DTYPE],
    dir_y: Scalar[DTYPE],
    dir_z: Scalar[DTYPE],
    pos_x: Scalar[DTYPE],
    pos_y: Scalar[DTYPE],
    pos_z: Scalar[DTYPE],
    qx: Scalar[DTYPE],
    qy: Scalar[DTYPE],
    qz: Scalar[DTYPE],
    qw: Scalar[DTYPE],
    verts: List[Scalar[DTYPE]],
    vert_offset: Int,
    num_verts: Int,
) -> InlineArray[Scalar[DTYPE], 3]:
    """Support point on mesh: exhaustive scan of hull vertices.

    Vertices are stored in local frame. We rotate dir to local frame,
    find max-dot vertex, then transform to world frame. O(n) per call,
    but n is typically 50-200 for robot convex hulls.
    """
    # Rotate direction to local frame
    var local_dir = quat_rotate_inverse[DTYPE](qx, qy, qz, qw,
        dir_x, dir_y, dir_z)
    var ld_x = local_dir[0]
    var ld_y = local_dir[1]
    var ld_z = local_dir[2]

    # Exhaustive scan for max dot product
    var best_dot: Scalar[DTYPE] = -1e30
    var best_idx = 0
    for i in range(num_verts):
        var off = vert_offset + i * 3
        var d = ld_x * verts[off] + ld_y * verts[off + 1] + ld_z * verts[off + 2]
        if d > best_dot:
            best_dot = d
            best_idx = i

    # Transform best vertex to world frame
    var off = vert_offset + best_idx * 3
    var local_pt = quat_rotate[DTYPE](qx, qy, qz, qw,
        verts[off], verts[off + 1], verts[off + 2])

    var result = InlineArray[Scalar[DTYPE], 3](fill=Scalar[DTYPE](0))
    result[0] = pos_x + local_pt[0]
    result[1] = pos_y + local_pt[1]
    result[2] = pos_z + local_pt[2]
    return result
