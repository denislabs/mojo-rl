"""Physics3D v2 collision primitives - Pure collision detection functions.

These are GPU-compatible, stateless collision functions that don't perform I/O.
They compute collision geometry (distance, contact point, normal) for primitive pairs.

Following MuJoCo conventions:
- dist: Signed distance (negative = penetration, positive = gap)
- contact: Midpoint between closest surface points
- normal: Points from body A to body B

Phase 3: sphere-sphere and sphere-plane primitives.
Phase 6: Added tangent basis computation for friction.
"""

from math import sqrt


@always_inline
fn compute_tangent_basis[
    DTYPE: DType
](
    nx: Scalar[DTYPE], ny: Scalar[DTYPE], nz: Scalar[DTYPE],
) -> Tuple[
    Scalar[DTYPE], Scalar[DTYPE], Scalar[DTYPE],  # t1
    Scalar[DTYPE], Scalar[DTYPE], Scalar[DTYPE],  # t2
]:
    """Compute two orthonormal tangent vectors from contact normal.

    Uses Gram-Schmidt: pick axis least parallel to n, cross to get t1,
    cross again to get t2.

    Args:
        nx, ny, nz: Contact normal (should be unit vector).

    Returns:
        Tuple of (t1x, t1y, t1z, t2x, t2y, t2z) - two orthonormal tangent vectors.
    """
    # Find axis least parallel to normal
    var ax: Scalar[DTYPE]
    var ay: Scalar[DTYPE]
    var az: Scalar[DTYPE]

    var abs_nx = abs(nx)
    var abs_ny = abs(ny)
    var abs_nz = abs(nz)

    if abs_nx < abs_ny and abs_nx < abs_nz:
        ax = Scalar[DTYPE](1.0)
        ay = Scalar[DTYPE](0.0)
        az = Scalar[DTYPE](0.0)
    elif abs_ny < abs_nz:
        ax = Scalar[DTYPE](0.0)
        ay = Scalar[DTYPE](1.0)
        az = Scalar[DTYPE](0.0)
    else:
        ax = Scalar[DTYPE](0.0)
        ay = Scalar[DTYPE](0.0)
        az = Scalar[DTYPE](1.0)

    # t1 = normalize(a - (a·n)*n) using Gram-Schmidt
    var dot = ax * nx + ay * ny + az * nz
    var t1x = ax - dot * nx
    var t1y = ay - dot * ny
    var t1z = az - dot * nz
    var t1_len = sqrt(t1x * t1x + t1y * t1y + t1z * t1z)

    # Normalize t1
    if t1_len > Scalar[DTYPE](1e-10):
        t1x = t1x / t1_len
        t1y = t1y / t1_len
        t1z = t1z / t1_len
    else:
        # Degenerate case: use fallback
        t1x = Scalar[DTYPE](1.0)
        t1y = Scalar[DTYPE](0.0)
        t1z = Scalar[DTYPE](0.0)

    # t2 = n × t1 (already normalized since n and t1 are unit vectors)
    var t2x = ny * t1z - nz * t1y
    var t2y = nz * t1x - nx * t1z
    var t2z = nx * t1y - ny * t1x

    return (t1x, t1y, t1z, t2x, t2y, t2z)


@always_inline
fn sphere_sphere[
    DTYPE: DType
](
    pos1_x: Scalar[DTYPE],
    pos1_y: Scalar[DTYPE],
    pos1_z: Scalar[DTYPE],
    radius1: Scalar[DTYPE],
    pos2_x: Scalar[DTYPE],
    pos2_y: Scalar[DTYPE],
    pos2_z: Scalar[DTYPE],
    radius2: Scalar[DTYPE],
) -> Tuple[
    Scalar[DTYPE],
    Scalar[DTYPE],
    Scalar[DTYPE],
    Scalar[DTYPE],
    Scalar[DTYPE],
    Scalar[DTYPE],
    Scalar[DTYPE],
]:
    """Sphere-sphere collision detection (MuJoCo-style).

    Args:
        pos1_x, pos1_y, pos1_z: Center of sphere 1.
        radius1: Radius of sphere 1.
        pos2_x, pos2_y, pos2_z: Center of sphere 2.
        radius2: Radius of sphere 2.

    Returns:
        Tuple of (dist, contact_x, contact_y, contact_z, normal_x, normal_y, normal_z):
        - dist: Signed distance (negative = penetration).
        - contact: Midpoint between surfaces.
        - normal: Unit vector pointing from sphere1 to sphere2.
    """
    # Vector from sphere1 center to sphere2 center
    var dx = pos2_x - pos1_x
    var dy = pos2_y - pos1_y
    var dz = pos2_z - pos1_z
    var center_dist = sqrt(dx * dx + dy * dy + dz * dz)

    # Surface distance (negative = penetrating)
    var dist = center_dist - (radius1 + radius2)

    # Normal vector (from sphere1 to sphere2)
    var nx: Scalar[DTYPE]
    var ny: Scalar[DTYPE]
    var nz: Scalar[DTYPE]

    if center_dist > Scalar[DTYPE](1e-10):
        var inv_dist = Scalar[DTYPE](1.0) / center_dist
        nx = dx * inv_dist
        ny = dy * inv_dist
        nz = dz * inv_dist
    else:
        # Degenerate case: spheres at same position
        # Use arbitrary normal (x-axis)
        nx = Scalar[DTYPE](1.0)
        ny = Scalar[DTYPE](0.0)
        nz = Scalar[DTYPE](0.0)

    # Contact position: midpoint between surface points
    # Surface point on sphere1 = pos1 + normal * radius1
    # Surface point on sphere2 = pos2 - normal * radius2
    # Midpoint = pos1 + normal * (radius1 + dist/2)
    var half_dist = Scalar[DTYPE](0.5) * dist
    var contact_x = pos1_x + nx * (radius1 + half_dist)
    var contact_y = pos1_y + ny * (radius1 + half_dist)
    var contact_z = pos1_z + nz * (radius1 + half_dist)

    return (dist, contact_x, contact_y, contact_z, nx, ny, nz)


@always_inline
fn sphere_plane[
    DTYPE: DType
](
    sphere_x: Scalar[DTYPE],
    sphere_y: Scalar[DTYPE],
    sphere_z: Scalar[DTYPE],
    radius: Scalar[DTYPE],
    ground_z: Scalar[DTYPE],
) -> Tuple[Scalar[DTYPE], Scalar[DTYPE], Scalar[DTYPE], Scalar[DTYPE]]:
    """Sphere-plane collision detection (ground plane with normal = +Z).

    Args:
        sphere_x, sphere_y, sphere_z: Center of the sphere.
        radius: Radius of the sphere.
        ground_z: Z-height of the ground plane.

    Returns:
        Tuple of (dist, contact_x, contact_y, contact_z):
        - dist: Signed distance from sphere surface to plane (negative = penetration).
        - contact: Point on the ground plane directly below sphere center.
    """
    # Distance from sphere surface to ground
    var dist = (sphere_z - ground_z) - radius

    # Contact point is on the ground, directly below sphere center
    var contact_x = sphere_x
    var contact_y = sphere_y
    var contact_z = ground_z

    return (dist, contact_x, contact_y, contact_z)
