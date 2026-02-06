"""Physics3D v2 collision primitives - Pure collision detection functions.

These are GPU-compatible, stateless collision functions that don't perform I/O.
They compute collision geometry (distance, contact point, normal) for primitive pairs.

Following MuJoCo conventions:
- dist: Signed distance (negative = penetration, positive = gap)
- contact: Midpoint between closest surface points
- normal: Points from body A to body B

Phase 3: sphere-sphere and sphere-plane primitives.
Phase 6: Added tangent basis computation for friction.
Phase 8: Added capsule primitives (capsule-plane, capsule-sphere, capsule-capsule).
Phase 9: Added box primitives (box-plane, box-sphere, box-capsule, box-box).
"""

from math import sqrt


# =============================================================================
# Quaternion Helpers (for rotating capsule axis to world frame)
# =============================================================================


@always_inline
fn rotate_vector_by_quat[
    DTYPE: DType
](
    vx: Scalar[DTYPE],
    vy: Scalar[DTYPE],
    vz: Scalar[DTYPE],
    qx: Scalar[DTYPE],
    qy: Scalar[DTYPE],
    qz: Scalar[DTYPE],
    qw: Scalar[DTYPE],
) -> Tuple[Scalar[DTYPE], Scalar[DTYPE], Scalar[DTYPE]]:
    """Rotate a vector by a quaternion.

    Uses the formula: v' = q * v * q^(-1)
    Optimized form avoiding full quaternion multiplication.

    Args:
        vx, vy, vz: Vector to rotate.
        qx, qy, qz, qw: Unit quaternion [x, y, z, w].

    Returns:
        Rotated vector (rx, ry, rz).
    """
    # t = 2 * cross(q.xyz, v)
    var tx = Scalar[DTYPE](2.0) * (qy * vz - qz * vy)
    var ty = Scalar[DTYPE](2.0) * (qz * vx - qx * vz)
    var tz = Scalar[DTYPE](2.0) * (qx * vy - qy * vx)

    # result = v + q.w * t + cross(q.xyz, t)
    var rx = vx + qw * tx + (qy * tz - qz * ty)
    var ry = vy + qw * ty + (qz * tx - qx * tz)
    var rz = vz + qw * tz + (qx * ty - qy * tx)

    return (rx, ry, rz)


@always_inline
fn rotate_vector_by_quat_inverse[
    DTYPE: DType
](
    vx: Scalar[DTYPE],
    vy: Scalar[DTYPE],
    vz: Scalar[DTYPE],
    qx: Scalar[DTYPE],
    qy: Scalar[DTYPE],
    qz: Scalar[DTYPE],
    qw: Scalar[DTYPE],
) -> Tuple[Scalar[DTYPE], Scalar[DTYPE], Scalar[DTYPE]]:
    """Rotate a vector by the inverse (conjugate) of a quaternion.

    For unit quaternions, q^(-1) = q* = (-qx, -qy, -qz, qw).
    This transforms from world frame to local frame.

    Args:
        vx, vy, vz: Vector to rotate.
        qx, qy, qz, qw: Unit quaternion [x, y, z, w].

    Returns:
        Rotated vector (rx, ry, rz) in local frame.
    """
    # Use conjugate: negate the vector part
    return rotate_vector_by_quat(vx, vy, vz, -qx, -qy, -qz, qw)


@always_inline
fn compute_tangent_basis[
    DTYPE: DType
](
    nx: Scalar[DTYPE],
    ny: Scalar[DTYPE],
    nz: Scalar[DTYPE],
) -> Tuple[
    Scalar[DTYPE],
    Scalar[DTYPE],
    Scalar[DTYPE],  # t1
    Scalar[DTYPE],
    Scalar[DTYPE],
    Scalar[DTYPE],  # t2
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


# =============================================================================
# Capsule Collision Primitives (Phase 8)
# =============================================================================


@always_inline
fn capsule_plane[
    DTYPE: DType
](
    # Capsule center and orientation
    cap_x: Scalar[DTYPE],
    cap_y: Scalar[DTYPE],
    cap_z: Scalar[DTYPE],
    cap_qx: Scalar[DTYPE],
    cap_qy: Scalar[DTYPE],
    cap_qz: Scalar[DTYPE],
    cap_qw: Scalar[DTYPE],
    half_length: Scalar[DTYPE],
    radius: Scalar[DTYPE],
    # Plane (horizontal at ground_z)
    ground_z: Scalar[DTYPE],
) -> Tuple[Scalar[DTYPE], Scalar[DTYPE], Scalar[DTYPE], Scalar[DTYPE]]:
    """Capsule-plane collision detection (ground plane with normal = +Z).

    A capsule is a cylinder with hemispherical caps. Its axis is along the local Z-axis.

    Algorithm:
    1. Compute capsule endpoints in world frame by rotating local Z-axis by quaternion
    2. Find lowest endpoint (or both if horizontal)
    3. Compute distance from lowest point's sphere surface to plane

    Args:
        cap_x, cap_y, cap_z: Center of the capsule.
        cap_qx, cap_qy, cap_qz, cap_qw: Quaternion orientation [x, y, z, w].
        half_length: Half-length of the cylindrical part.
        radius: Radius of the capsule.
        ground_z: Z-height of the ground plane.

    Returns:
        Tuple of (dist, contact_x, contact_y, contact_z):
        - dist: Signed distance from capsule surface to plane (negative = penetration).
        - contact: Contact point on the ground plane.
    """
    # Local Z-axis of capsule (before rotation)
    var local_z_x = Scalar[DTYPE](0.0)
    var local_z_y = Scalar[DTYPE](0.0)
    var local_z_z = Scalar[DTYPE](1.0)

    # Rotate local Z-axis to world frame to get capsule axis direction
    var axis = rotate_vector_by_quat(
        local_z_x, local_z_y, local_z_z, cap_qx, cap_qy, cap_qz, cap_qw
    )
    var ax = axis[0]
    var ay = axis[1]
    var az = axis[2]

    # Capsule endpoints in world frame
    # endpoint1 = center + axis * half_length
    # endpoint2 = center - axis * half_length
    var ep1_x = cap_x + ax * half_length
    var ep1_y = cap_y + ay * half_length
    var ep1_z = cap_z + az * half_length

    var ep2_x = cap_x - ax * half_length
    var ep2_y = cap_y - ay * half_length
    var ep2_z = cap_z - az * half_length

    # Find the lowest endpoint (minimum z)
    var lowest_x: Scalar[DTYPE]
    var lowest_y: Scalar[DTYPE]
    var lowest_z: Scalar[DTYPE]

    if ep1_z <= ep2_z:
        lowest_x = ep1_x
        lowest_y = ep1_y
        lowest_z = ep1_z
    else:
        lowest_x = ep2_x
        lowest_y = ep2_y
        lowest_z = ep2_z

    # Signed distance from lowest sphere surface to ground
    var dist = (lowest_z - ground_z) - radius

    # Contact point is on ground, below the lowest endpoint
    var contact_x = lowest_x
    var contact_y = lowest_y
    var contact_z = ground_z

    return (dist, contact_x, contact_y, contact_z)


@always_inline
fn capsule_sphere[
    DTYPE: DType
](
    # Capsule
    cap_x: Scalar[DTYPE],
    cap_y: Scalar[DTYPE],
    cap_z: Scalar[DTYPE],
    cap_qx: Scalar[DTYPE],
    cap_qy: Scalar[DTYPE],
    cap_qz: Scalar[DTYPE],
    cap_qw: Scalar[DTYPE],
    cap_half_len: Scalar[DTYPE],
    cap_radius: Scalar[DTYPE],
    # Sphere
    sph_x: Scalar[DTYPE],
    sph_y: Scalar[DTYPE],
    sph_z: Scalar[DTYPE],
    sph_radius: Scalar[DTYPE],
) -> Tuple[
    Scalar[DTYPE],
    Scalar[DTYPE],
    Scalar[DTYPE],
    Scalar[DTYPE],
    Scalar[DTYPE],
    Scalar[DTYPE],
    Scalar[DTYPE],
]:
    """Capsule-sphere collision detection.

    Algorithm:
    1. Project sphere center onto capsule axis (clamped to segment)
    2. Compute distance from projection point to sphere center
    3. Treat as sphere-sphere collision between the projection point
       (with cap_radius) and the sphere.

    Args:
        cap_x, cap_y, cap_z: Center of the capsule.
        cap_qx, cap_qy, cap_qz, cap_qw: Quaternion orientation [x, y, z, w].
        cap_half_len: Half-length of the cylindrical part.
        cap_radius: Radius of the capsule.
        sph_x, sph_y, sph_z: Center of the sphere.
        sph_radius: Radius of the sphere.

    Returns:
        Tuple of (dist, contact_x, contact_y, contact_z, normal_x, normal_y, normal_z):
        - dist: Signed distance (negative = penetration).
        - contact: Midpoint between surfaces.
        - normal: Unit vector pointing from capsule to sphere.
    """
    # Get capsule axis in world frame
    var axis = rotate_vector_by_quat(
        Scalar[DTYPE](0.0),
        Scalar[DTYPE](0.0),
        Scalar[DTYPE](1.0),
        cap_qx,
        cap_qy,
        cap_qz,
        cap_qw,
    )
    var ax = axis[0]
    var ay = axis[1]
    var az = axis[2]

    # Vector from capsule center to sphere center
    var dx = sph_x - cap_x
    var dy = sph_y - cap_y
    var dz = sph_z - cap_z

    # Project onto capsule axis: t = dot(d, axis), clamped to [-half_len, half_len]
    var t = dx * ax + dy * ay + dz * az

    # Clamp t to the capsule segment
    if t < -cap_half_len:
        t = -cap_half_len
    elif t > cap_half_len:
        t = cap_half_len

    # Closest point on capsule axis to sphere center
    var closest_x = cap_x + ax * t
    var closest_y = cap_y + ay * t
    var closest_z = cap_z + az * t

    # Now treat as sphere-sphere: closest point with cap_radius vs sphere
    return sphere_sphere(
        closest_x,
        closest_y,
        closest_z,
        cap_radius,
        sph_x,
        sph_y,
        sph_z,
        sph_radius,
    )


@always_inline
fn _closest_points_line_segments[
    DTYPE: DType
](
    # Segment 1: p1 to p1 + d1
    p1_x: Scalar[DTYPE],
    p1_y: Scalar[DTYPE],
    p1_z: Scalar[DTYPE],
    d1_x: Scalar[DTYPE],
    d1_y: Scalar[DTYPE],
    d1_z: Scalar[DTYPE],
    # Segment 2: p2 to p2 + d2
    p2_x: Scalar[DTYPE],
    p2_y: Scalar[DTYPE],
    p2_z: Scalar[DTYPE],
    d2_x: Scalar[DTYPE],
    d2_y: Scalar[DTYPE],
    d2_z: Scalar[DTYPE],
) -> Tuple[
    Scalar[DTYPE],
    Scalar[DTYPE],
    Scalar[DTYPE],
    Scalar[DTYPE],
    Scalar[DTYPE],
    Scalar[DTYPE],
]:
    """Find closest points between two line segments.

    Segment 1: P1(s) = p1 + s * d1, s in [0, 1]
    Segment 2: P2(t) = p2 + t * d2, t in [0, 1]

    Uses the algorithm from Real-Time Collision Detection by Christer Ericson.

    Returns:
        Tuple of (c1_x, c1_y, c1_z, c2_x, c2_y, c2_z):
        - c1: Closest point on segment 1
        - c2: Closest point on segment 2
    """
    var EPSILON = Scalar[DTYPE](1e-10)

    # r = p1 - p2
    var r_x = p1_x - p2_x
    var r_y = p1_y - p2_y
    var r_z = p1_z - p2_z

    # Dot products
    var a = d1_x * d1_x + d1_y * d1_y + d1_z * d1_z  # |d1|^2
    var e = d2_x * d2_x + d2_y * d2_y + d2_z * d2_z  # |d2|^2
    var f = d2_x * r_x + d2_y * r_y + d2_z * r_z  # dot(d2, r)

    var s: Scalar[DTYPE]
    var t: Scalar[DTYPE]

    # Check if both segments degenerate into points
    if a <= EPSILON and e <= EPSILON:
        s = Scalar[DTYPE](0.0)
        t = Scalar[DTYPE](0.0)
    elif a <= EPSILON:
        # First segment degenerates into a point
        s = Scalar[DTYPE](0.0)
        t = f / e
        if t < Scalar[DTYPE](0.0):
            t = Scalar[DTYPE](0.0)
        elif t > Scalar[DTYPE](1.0):
            t = Scalar[DTYPE](1.0)
    elif e <= EPSILON:
        # Second segment degenerates into a point
        t = Scalar[DTYPE](0.0)
        var c = d1_x * r_x + d1_y * r_y + d1_z * r_z  # dot(d1, r)
        s = -c / a
        if s < Scalar[DTYPE](0.0):
            s = Scalar[DTYPE](0.0)
        elif s > Scalar[DTYPE](1.0):
            s = Scalar[DTYPE](1.0)
    else:
        # General non-degenerate case
        var b = d1_x * d2_x + d1_y * d2_y + d1_z * d2_z  # dot(d1, d2)
        var c = d1_x * r_x + d1_y * r_y + d1_z * r_z  # dot(d1, r)
        var denom = a * e - b * b

        # If segments not parallel, compute closest point on L1 to L2
        if denom > EPSILON:
            s = (b * f - c * e) / denom
            if s < Scalar[DTYPE](0.0):
                s = Scalar[DTYPE](0.0)
            elif s > Scalar[DTYPE](1.0):
                s = Scalar[DTYPE](1.0)
        else:
            # Parallel segments: pick arbitrary point
            s = Scalar[DTYPE](0.0)

        # Compute point on L2 closest to S1(s)
        t = (b * s + f) / e

        # Clamp t and recompute s if needed
        if t < Scalar[DTYPE](0.0):
            t = Scalar[DTYPE](0.0)
            s = -c / a
            if s < Scalar[DTYPE](0.0):
                s = Scalar[DTYPE](0.0)
            elif s > Scalar[DTYPE](1.0):
                s = Scalar[DTYPE](1.0)
        elif t > Scalar[DTYPE](1.0):
            t = Scalar[DTYPE](1.0)
            s = (b - c) / a
            if s < Scalar[DTYPE](0.0):
                s = Scalar[DTYPE](0.0)
            elif s > Scalar[DTYPE](1.0):
                s = Scalar[DTYPE](1.0)

    # Compute closest points
    var c1_x = p1_x + d1_x * s
    var c1_y = p1_y + d1_y * s
    var c1_z = p1_z + d1_z * s

    var c2_x = p2_x + d2_x * t
    var c2_y = p2_y + d2_y * t
    var c2_z = p2_z + d2_z * t

    return (c1_x, c1_y, c1_z, c2_x, c2_y, c2_z)


@always_inline
fn capsule_capsule[
    DTYPE: DType
](
    # Capsule A
    a_x: Scalar[DTYPE],
    a_y: Scalar[DTYPE],
    a_z: Scalar[DTYPE],
    a_qx: Scalar[DTYPE],
    a_qy: Scalar[DTYPE],
    a_qz: Scalar[DTYPE],
    a_qw: Scalar[DTYPE],
    a_half_len: Scalar[DTYPE],
    a_radius: Scalar[DTYPE],
    # Capsule B
    b_x: Scalar[DTYPE],
    b_y: Scalar[DTYPE],
    b_z: Scalar[DTYPE],
    b_qx: Scalar[DTYPE],
    b_qy: Scalar[DTYPE],
    b_qz: Scalar[DTYPE],
    b_qw: Scalar[DTYPE],
    b_half_len: Scalar[DTYPE],
    b_radius: Scalar[DTYPE],
) -> Tuple[
    Scalar[DTYPE],
    Scalar[DTYPE],
    Scalar[DTYPE],
    Scalar[DTYPE],
    Scalar[DTYPE],
    Scalar[DTYPE],
    Scalar[DTYPE],
]:
    """Capsule-capsule collision detection.

    Algorithm:
    1. Compute capsule axes in world frame
    2. Find closest points between the two line segments (capsule axes)
    3. Treat as sphere-sphere collision between the closest points

    Args:
        a_x, a_y, a_z: Center of capsule A.
        a_qx, a_qy, a_qz, a_qw: Quaternion orientation of capsule A.
        a_half_len: Half-length of capsule A.
        a_radius: Radius of capsule A.
        b_x, b_y, b_z: Center of capsule B.
        b_qx, b_qy, b_qz, b_qw: Quaternion orientation of capsule B.
        b_half_len: Half-length of capsule B.
        b_radius: Radius of capsule B.

    Returns:
        Tuple of (dist, contact_x, contact_y, contact_z, normal_x, normal_y, normal_z):
        - dist: Signed distance (negative = penetration).
        - contact: Midpoint between surfaces.
        - normal: Unit vector pointing from capsule A to capsule B.
    """
    # Get capsule A axis in world frame
    var axis_a = rotate_vector_by_quat(
        Scalar[DTYPE](0.0),
        Scalar[DTYPE](0.0),
        Scalar[DTYPE](1.0),
        a_qx,
        a_qy,
        a_qz,
        a_qw,
    )
    var a_ax = axis_a[0]
    var a_ay = axis_a[1]
    var a_az = axis_a[2]

    # Get capsule B axis in world frame
    var axis_b = rotate_vector_by_quat(
        Scalar[DTYPE](0.0),
        Scalar[DTYPE](0.0),
        Scalar[DTYPE](1.0),
        b_qx,
        b_qy,
        b_qz,
        b_qw,
    )
    var b_ax = axis_b[0]
    var b_ay = axis_b[1]
    var b_az = axis_b[2]

    # Capsule A segment: from (center - axis*half_len) to (center + axis*half_len)
    # P1 = a_center - a_axis * a_half_len
    # d1 = 2 * a_axis * a_half_len
    var p1_x = a_x - a_ax * a_half_len
    var p1_y = a_y - a_ay * a_half_len
    var p1_z = a_z - a_az * a_half_len

    var d1_x = Scalar[DTYPE](2.0) * a_ax * a_half_len
    var d1_y = Scalar[DTYPE](2.0) * a_ay * a_half_len
    var d1_z = Scalar[DTYPE](2.0) * a_az * a_half_len

    # Capsule B segment
    var p2_x = b_x - b_ax * b_half_len
    var p2_y = b_y - b_ay * b_half_len
    var p2_z = b_z - b_az * b_half_len

    var d2_x = Scalar[DTYPE](2.0) * b_ax * b_half_len
    var d2_y = Scalar[DTYPE](2.0) * b_ay * b_half_len
    var d2_z = Scalar[DTYPE](2.0) * b_az * b_half_len

    # Find closest points on the two line segments
    var closest = _closest_points_line_segments(
        p1_x, p1_y, p1_z, d1_x, d1_y, d1_z, p2_x, p2_y, p2_z, d2_x, d2_y, d2_z
    )

    var c1_x = closest[0]
    var c1_y = closest[1]
    var c1_z = closest[2]
    var c2_x = closest[3]
    var c2_y = closest[4]
    var c2_z = closest[5]

    # Treat as sphere-sphere between the closest points
    return sphere_sphere(c1_x, c1_y, c1_z, a_radius, c2_x, c2_y, c2_z, b_radius)


# =============================================================================
# Box Collision Primitives (Phase 9)
# =============================================================================


@always_inline
fn _check_vertex[
    DTYPE: DType
](
    box_x: Scalar[DTYPE],
    box_y: Scalar[DTYPE],
    box_z: Scalar[DTYPE],
    box_qx: Scalar[DTYPE],
    box_qy: Scalar[DTYPE],
    box_qz: Scalar[DTYPE],
    box_qw: Scalar[DTYPE],
    half_x: Scalar[DTYPE],
    half_y: Scalar[DTYPE],
    half_z: Scalar[DTYPE],
    sx: Scalar[DTYPE],
    sy: Scalar[DTYPE],
    sz: Scalar[DTYPE],
    mut min_z: Scalar[DTYPE],
    mut lowest_x: Scalar[DTYPE],
    mut lowest_y: Scalar[DTYPE],
):
    """Check one vertex and update minimum if lower."""
    var lx = sx * half_x
    var ly = sy * half_y
    var lz = sz * half_z
    var rotated = rotate_vector_by_quat(lx, ly, lz, box_qx, box_qy, box_qz, box_qw)
    var vx = box_x + rotated[0]
    var vy = box_y + rotated[1]
    var vz = box_z + rotated[2]
    if vz < min_z:
        min_z = vz
        lowest_x = vx
        lowest_y = vy


@always_inline
fn box_plane[
    DTYPE: DType
](
    # Box center and orientation
    box_x: Scalar[DTYPE],
    box_y: Scalar[DTYPE],
    box_z: Scalar[DTYPE],
    box_qx: Scalar[DTYPE],
    box_qy: Scalar[DTYPE],
    box_qz: Scalar[DTYPE],
    box_qw: Scalar[DTYPE],
    half_x: Scalar[DTYPE],
    half_y: Scalar[DTYPE],
    half_z: Scalar[DTYPE],
    # Plane (horizontal at ground_z)
    ground_z: Scalar[DTYPE],
) -> Tuple[Scalar[DTYPE], Scalar[DTYPE], Scalar[DTYPE], Scalar[DTYPE]]:
    """Box-plane collision detection (ground plane with normal = +Z).

    A box is defined by its half-extents along each local axis.

    Algorithm:
    1. Compute all 8 vertices of the box in world frame
    2. Find the lowest vertex (minimum z)
    3. Compute signed distance from lowest vertex to plane

    Returns:
        Tuple of (dist, contact_x, contact_y, contact_z):
        - dist: Signed distance from box surface to plane (negative = penetration).
        - contact: Contact point on the ground plane below lowest vertex.
    """
    # Compute all 8 box vertices in world frame
    # Local vertex offsets: (+/-hx, +/-hy, +/-hz)
    var min_z = box_z + Scalar[DTYPE](1e10)  # Start with large value
    var lowest_x = box_x
    var lowest_y = box_y

    # Check all 8 corners explicitly (GPU-compatible, no heap allocation)
    var NEG = Scalar[DTYPE](-1.0)
    var POS = Scalar[DTYPE](1.0)

    _check_vertex(box_x, box_y, box_z, box_qx, box_qy, box_qz, box_qw,
                  half_x, half_y, half_z, NEG, NEG, NEG, min_z, lowest_x, lowest_y)
    _check_vertex(box_x, box_y, box_z, box_qx, box_qy, box_qz, box_qw,
                  half_x, half_y, half_z, NEG, NEG, POS, min_z, lowest_x, lowest_y)
    _check_vertex(box_x, box_y, box_z, box_qx, box_qy, box_qz, box_qw,
                  half_x, half_y, half_z, NEG, POS, NEG, min_z, lowest_x, lowest_y)
    _check_vertex(box_x, box_y, box_z, box_qx, box_qy, box_qz, box_qw,
                  half_x, half_y, half_z, NEG, POS, POS, min_z, lowest_x, lowest_y)
    _check_vertex(box_x, box_y, box_z, box_qx, box_qy, box_qz, box_qw,
                  half_x, half_y, half_z, POS, NEG, NEG, min_z, lowest_x, lowest_y)
    _check_vertex(box_x, box_y, box_z, box_qx, box_qy, box_qz, box_qw,
                  half_x, half_y, half_z, POS, NEG, POS, min_z, lowest_x, lowest_y)
    _check_vertex(box_x, box_y, box_z, box_qx, box_qy, box_qz, box_qw,
                  half_x, half_y, half_z, POS, POS, NEG, min_z, lowest_x, lowest_y)
    _check_vertex(box_x, box_y, box_z, box_qx, box_qy, box_qz, box_qw,
                  half_x, half_y, half_z, POS, POS, POS, min_z, lowest_x, lowest_y)

    # Signed distance from lowest vertex to ground
    var dist = min_z - ground_z

    # Contact point is on ground below lowest vertex
    var contact_x = lowest_x
    var contact_y = lowest_y
    var contact_z = ground_z

    return (dist, contact_x, contact_y, contact_z)


@always_inline
fn box_sphere[
    DTYPE: DType
](
    # Box
    box_x: Scalar[DTYPE],
    box_y: Scalar[DTYPE],
    box_z: Scalar[DTYPE],
    box_qx: Scalar[DTYPE],
    box_qy: Scalar[DTYPE],
    box_qz: Scalar[DTYPE],
    box_qw: Scalar[DTYPE],
    half_x: Scalar[DTYPE],
    half_y: Scalar[DTYPE],
    half_z: Scalar[DTYPE],
    # Sphere
    sph_x: Scalar[DTYPE],
    sph_y: Scalar[DTYPE],
    sph_z: Scalar[DTYPE],
    sph_radius: Scalar[DTYPE],
) -> Tuple[
    Scalar[DTYPE],
    Scalar[DTYPE],
    Scalar[DTYPE],
    Scalar[DTYPE],
    Scalar[DTYPE],
    Scalar[DTYPE],
    Scalar[DTYPE],
]:
    """Box-sphere collision detection.

    Algorithm:
    1. Transform sphere center to box's local frame
    2. Clamp sphere center to box bounds to find closest point on box
    3. Compute distance from clamped point to sphere center
    4. Transform result back to world frame

    Returns:
        Tuple of (dist, contact_x, contact_y, contact_z, normal_x, normal_y, normal_z):
        - dist: Signed distance (negative = penetration).
        - contact: Contact point (midpoint between surfaces).
        - normal: Unit vector pointing from box to sphere.
    """
    # Transform sphere center to box's local frame
    var rel_x = sph_x - box_x
    var rel_y = sph_y - box_y
    var rel_z = sph_z - box_z

    var local = rotate_vector_by_quat_inverse(
        rel_x, rel_y, rel_z, box_qx, box_qy, box_qz, box_qw
    )
    var local_x = local[0]
    var local_y = local[1]
    var local_z = local[2]

    # Clamp to box bounds (closest point on box surface in local frame)
    var clamp_x = local_x
    var clamp_y = local_y
    var clamp_z = local_z

    if clamp_x < -half_x:
        clamp_x = -half_x
    elif clamp_x > half_x:
        clamp_x = half_x

    if clamp_y < -half_y:
        clamp_y = -half_y
    elif clamp_y > half_y:
        clamp_y = half_y

    if clamp_z < -half_z:
        clamp_z = -half_z
    elif clamp_z > half_z:
        clamp_z = half_z

    # Vector from closest point to sphere center (in local frame)
    var dx = local_x - clamp_x
    var dy = local_y - clamp_y
    var dz = local_z - clamp_z
    var dist_sq = dx * dx + dy * dy + dz * dz
    var dist_to_center = sqrt(dist_sq)

    # Signed distance (surface to surface)
    var dist = dist_to_center - sph_radius

    # Compute normal (from box to sphere)
    var nx_local: Scalar[DTYPE]
    var ny_local: Scalar[DTYPE]
    var nz_local: Scalar[DTYPE]

    if dist_to_center > Scalar[DTYPE](1e-10):
        var inv_dist = Scalar[DTYPE](1.0) / dist_to_center
        nx_local = dx * inv_dist
        ny_local = dy * inv_dist
        nz_local = dz * inv_dist
    else:
        # Sphere center is inside or on box surface
        # Find which face is closest and use its normal
        var face_dist_x = half_x - abs(local_x)
        var face_dist_y = half_y - abs(local_y)
        var face_dist_z = half_z - abs(local_z)

        if face_dist_x <= face_dist_y and face_dist_x <= face_dist_z:
            # X face is closest
            nx_local = Scalar[DTYPE](1.0) if local_x >= Scalar[DTYPE](0) else Scalar[DTYPE](-1.0)
            ny_local = Scalar[DTYPE](0.0)
            nz_local = Scalar[DTYPE](0.0)
            dist = -face_dist_x - sph_radius
        elif face_dist_y <= face_dist_z:
            # Y face is closest
            nx_local = Scalar[DTYPE](0.0)
            ny_local = Scalar[DTYPE](1.0) if local_y >= Scalar[DTYPE](0) else Scalar[DTYPE](-1.0)
            nz_local = Scalar[DTYPE](0.0)
            dist = -face_dist_y - sph_radius
        else:
            # Z face is closest
            nx_local = Scalar[DTYPE](0.0)
            ny_local = Scalar[DTYPE](0.0)
            nz_local = Scalar[DTYPE](1.0) if local_z >= Scalar[DTYPE](0) else Scalar[DTYPE](-1.0)
            dist = -face_dist_z - sph_radius

    # Transform normal back to world frame
    var normal_world = rotate_vector_by_quat(
        nx_local, ny_local, nz_local, box_qx, box_qy, box_qz, box_qw
    )
    var nx = normal_world[0]
    var ny = normal_world[1]
    var nz = normal_world[2]

    # Transform closest point on box to world frame
    var closest_world = rotate_vector_by_quat(
        clamp_x, clamp_y, clamp_z, box_qx, box_qy, box_qz, box_qw
    )
    var closest_x = box_x + closest_world[0]
    var closest_y = box_y + closest_world[1]
    var closest_z = box_z + closest_world[2]

    # Contact point is midpoint between surfaces
    var half_dist = Scalar[DTYPE](0.5) * dist
    var contact_x = closest_x + nx * half_dist
    var contact_y = closest_y + ny * half_dist
    var contact_z = closest_z + nz * half_dist

    return (dist, contact_x, contact_y, contact_z, nx, ny, nz)


@always_inline
fn box_capsule[
    DTYPE: DType
](
    # Box
    box_x: Scalar[DTYPE],
    box_y: Scalar[DTYPE],
    box_z: Scalar[DTYPE],
    box_qx: Scalar[DTYPE],
    box_qy: Scalar[DTYPE],
    box_qz: Scalar[DTYPE],
    box_qw: Scalar[DTYPE],
    half_x: Scalar[DTYPE],
    half_y: Scalar[DTYPE],
    half_z: Scalar[DTYPE],
    # Capsule
    cap_x: Scalar[DTYPE],
    cap_y: Scalar[DTYPE],
    cap_z: Scalar[DTYPE],
    cap_qx: Scalar[DTYPE],
    cap_qy: Scalar[DTYPE],
    cap_qz: Scalar[DTYPE],
    cap_qw: Scalar[DTYPE],
    cap_half_len: Scalar[DTYPE],
    cap_radius: Scalar[DTYPE],
) -> Tuple[
    Scalar[DTYPE],
    Scalar[DTYPE],
    Scalar[DTYPE],
    Scalar[DTYPE],
    Scalar[DTYPE],
    Scalar[DTYPE],
    Scalar[DTYPE],
]:
    """Box-capsule collision detection.

    Algorithm:
    1. Transform capsule endpoints to box's local frame
    2. Find closest point on capsule segment to box (clamped)
    3. Treat as box-sphere with the closest point as sphere center

    Returns:
        Tuple of (dist, contact_x, contact_y, contact_z, normal_x, normal_y, normal_z):
        - dist: Signed distance (negative = penetration).
        - contact: Contact point (midpoint between surfaces).
        - normal: Unit vector pointing from box to capsule.
    """
    # Get capsule axis in world frame
    var cap_axis = rotate_vector_by_quat(
        Scalar[DTYPE](0.0),
        Scalar[DTYPE](0.0),
        Scalar[DTYPE](1.0),
        cap_qx,
        cap_qy,
        cap_qz,
        cap_qw,
    )
    var ax = cap_axis[0]
    var ay = cap_axis[1]
    var az = cap_axis[2]

    # Capsule endpoints in world frame
    var ep1_x = cap_x + ax * cap_half_len
    var ep1_y = cap_y + ay * cap_half_len
    var ep1_z = cap_z + az * cap_half_len
    var ep2_x = cap_x - ax * cap_half_len
    var ep2_y = cap_y - ay * cap_half_len
    var ep2_z = cap_z - az * cap_half_len

    # Transform capsule segment to box's local frame
    var rel1_x = ep1_x - box_x
    var rel1_y = ep1_y - box_y
    var rel1_z = ep1_z - box_z
    var local1 = rotate_vector_by_quat_inverse(
        rel1_x, rel1_y, rel1_z, box_qx, box_qy, box_qz, box_qw
    )

    var rel2_x = ep2_x - box_x
    var rel2_y = ep2_y - box_y
    var rel2_z = ep2_z - box_z
    var local2 = rotate_vector_by_quat_inverse(
        rel2_x, rel2_y, rel2_z, box_qx, box_qy, box_qz, box_qw
    )

    # Segment direction in local frame
    var seg_dx = local2[0] - local1[0]
    var seg_dy = local2[1] - local1[1]
    var seg_dz = local2[2] - local1[2]

    # Find point on segment closest to box center (origin in local frame)
    # Project origin onto segment: t = -dot(p1, d) / dot(d, d)
    var seg_len_sq = seg_dx * seg_dx + seg_dy * seg_dy + seg_dz * seg_dz
    var t: Scalar[DTYPE]

    if seg_len_sq > Scalar[DTYPE](1e-10):
        t = -(local1[0] * seg_dx + local1[1] * seg_dy + local1[2] * seg_dz) / seg_len_sq
        if t < Scalar[DTYPE](0.0):
            t = Scalar[DTYPE](0.0)
        elif t > Scalar[DTYPE](1.0):
            t = Scalar[DTYPE](1.0)
    else:
        t = Scalar[DTYPE](0.5)

    # Closest point on segment (in local frame)
    var closest_seg_x = local1[0] + t * seg_dx
    var closest_seg_y = local1[1] + t * seg_dy
    var closest_seg_z = local1[2] + t * seg_dz

    # Transform back to world frame for box-sphere test
    var closest_world = rotate_vector_by_quat(
        closest_seg_x, closest_seg_y, closest_seg_z, box_qx, box_qy, box_qz, box_qw
    )
    var closest_x = box_x + closest_world[0]
    var closest_y = box_y + closest_world[1]
    var closest_z = box_z + closest_world[2]

    # Now treat as box-sphere collision
    return box_sphere(
        box_x, box_y, box_z,
        box_qx, box_qy, box_qz, box_qw,
        half_x, half_y, half_z,
        closest_x, closest_y, closest_z,
        cap_radius,
    )


@always_inline
fn _project_box_onto_axis[
    DTYPE: DType
](
    # Box rotation matrix columns (already computed)
    r0_x: Scalar[DTYPE], r0_y: Scalar[DTYPE], r0_z: Scalar[DTYPE],
    r1_x: Scalar[DTYPE], r1_y: Scalar[DTYPE], r1_z: Scalar[DTYPE],
    r2_x: Scalar[DTYPE], r2_y: Scalar[DTYPE], r2_z: Scalar[DTYPE],
    # Box half-extents
    hx: Scalar[DTYPE], hy: Scalar[DTYPE], hz: Scalar[DTYPE],
    # Axis to project onto
    ax: Scalar[DTYPE], ay: Scalar[DTYPE], az: Scalar[DTYPE],
) -> Scalar[DTYPE]:
    """Project a box onto an axis and return the half-width of the projection.

    The half-width is: |dot(r0, axis)|*hx + |dot(r1, axis)|*hy + |dot(r2, axis)|*hz
    where r0, r1, r2 are the box's rotation matrix columns.
    """
    return (
        abs(r0_x * ax + r0_y * ay + r0_z * az) * hx +
        abs(r1_x * ax + r1_y * ay + r1_z * az) * hy +
        abs(r2_x * ax + r2_y * ay + r2_z * az) * hz
    )


@always_inline
fn _test_sat_axis[
    DTYPE: DType
](
    axis_x: Scalar[DTYPE],
    axis_y: Scalar[DTYPE],
    axis_z: Scalar[DTYPE],
    # Translation between box centers
    t_x: Scalar[DTYPE],
    t_y: Scalar[DTYPE],
    t_z: Scalar[DTYPE],
    # Box A rotation columns
    a0_x: Scalar[DTYPE], a0_y: Scalar[DTYPE], a0_z: Scalar[DTYPE],
    a1_x: Scalar[DTYPE], a1_y: Scalar[DTYPE], a1_z: Scalar[DTYPE],
    a2_x: Scalar[DTYPE], a2_y: Scalar[DTYPE], a2_z: Scalar[DTYPE],
    a_hx: Scalar[DTYPE], a_hy: Scalar[DTYPE], a_hz: Scalar[DTYPE],
    # Box B rotation columns
    b0_x: Scalar[DTYPE], b0_y: Scalar[DTYPE], b0_z: Scalar[DTYPE],
    b1_x: Scalar[DTYPE], b1_y: Scalar[DTYPE], b1_z: Scalar[DTYPE],
    b2_x: Scalar[DTYPE], b2_y: Scalar[DTYPE], b2_z: Scalar[DTYPE],
    b_hx: Scalar[DTYPE], b_hy: Scalar[DTYPE], b_hz: Scalar[DTYPE],
    # Output references
    mut min_pen: Scalar[DTYPE],
    mut best_nx: Scalar[DTYPE],
    mut best_ny: Scalar[DTYPE],
    mut best_nz: Scalar[DTYPE],
) -> Bool:
    """Test one SAT axis. Returns True if separated (no collision)."""
    var EPSILON = Scalar[DTYPE](1e-10)

    var axis_len = sqrt(axis_x * axis_x + axis_y * axis_y + axis_z * axis_z)
    if axis_len < EPSILON:
        return False  # Skip degenerate axis

    var inv_len = Scalar[DTYPE](1.0) / axis_len
    var ax = axis_x * inv_len
    var ay = axis_y * inv_len
    var az = axis_z * inv_len

    # Project centers distance onto axis
    var center_dist = t_x * ax + t_y * ay + t_z * az

    # Project both boxes onto axis
    var proj_a = _project_box_onto_axis(
        a0_x, a0_y, a0_z, a1_x, a1_y, a1_z, a2_x, a2_y, a2_z,
        a_hx, a_hy, a_hz, ax, ay, az,
    )
    var proj_b = _project_box_onto_axis(
        b0_x, b0_y, b0_z, b1_x, b1_y, b1_z, b2_x, b2_y, b2_z,
        b_hx, b_hy, b_hz, ax, ay, az,
    )

    # Gap = |center_dist| - (proj_a + proj_b)
    var gap = abs(center_dist) - (proj_a + proj_b)

    if gap > Scalar[DTYPE](0.0):
        return True  # Separating axis found

    # Track minimum penetration
    var penetration = -gap
    if penetration < min_pen:
        min_pen = penetration
        # Normal should point from A to B
        if center_dist >= Scalar[DTYPE](0.0):
            best_nx = ax
            best_ny = ay
            best_nz = az
        else:
            best_nx = -ax
            best_ny = -ay
            best_nz = -az

    return False


@always_inline
fn box_box[
    DTYPE: DType
](
    # Box A
    a_x: Scalar[DTYPE],
    a_y: Scalar[DTYPE],
    a_z: Scalar[DTYPE],
    a_qx: Scalar[DTYPE],
    a_qy: Scalar[DTYPE],
    a_qz: Scalar[DTYPE],
    a_qw: Scalar[DTYPE],
    a_hx: Scalar[DTYPE],
    a_hy: Scalar[DTYPE],
    a_hz: Scalar[DTYPE],
    # Box B
    b_x: Scalar[DTYPE],
    b_y: Scalar[DTYPE],
    b_z: Scalar[DTYPE],
    b_qx: Scalar[DTYPE],
    b_qy: Scalar[DTYPE],
    b_qz: Scalar[DTYPE],
    b_qw: Scalar[DTYPE],
    b_hx: Scalar[DTYPE],
    b_hy: Scalar[DTYPE],
    b_hz: Scalar[DTYPE],
) -> Tuple[
    Scalar[DTYPE],
    Scalar[DTYPE],
    Scalar[DTYPE],
    Scalar[DTYPE],
    Scalar[DTYPE],
    Scalar[DTYPE],
    Scalar[DTYPE],
]:
    """Box-box collision detection using Separating Axis Theorem (SAT).

    Tests 15 potential separating axes:
    - 3 face normals from box A
    - 3 face normals from box B
    - 9 edge-edge cross products

    Returns:
        Tuple of (dist, contact_x, contact_y, contact_z, normal_x, normal_y, normal_z):
        - dist: Signed distance (negative = penetration).
        - contact: Approximate contact point (centroid of overlap region).
        - normal: Unit vector pointing from A to B along minimum penetration axis.
    """
    # Compute rotation matrix columns for box A (local axes in world frame)
    var a0 = rotate_vector_by_quat(Scalar[DTYPE](1), Scalar[DTYPE](0), Scalar[DTYPE](0), a_qx, a_qy, a_qz, a_qw)
    var a1 = rotate_vector_by_quat(Scalar[DTYPE](0), Scalar[DTYPE](1), Scalar[DTYPE](0), a_qx, a_qy, a_qz, a_qw)
    var a2 = rotate_vector_by_quat(Scalar[DTYPE](0), Scalar[DTYPE](0), Scalar[DTYPE](1), a_qx, a_qy, a_qz, a_qw)

    # Compute rotation matrix columns for box B
    var b0 = rotate_vector_by_quat(Scalar[DTYPE](1), Scalar[DTYPE](0), Scalar[DTYPE](0), b_qx, b_qy, b_qz, b_qw)
    var b1 = rotate_vector_by_quat(Scalar[DTYPE](0), Scalar[DTYPE](1), Scalar[DTYPE](0), b_qx, b_qy, b_qz, b_qw)
    var b2 = rotate_vector_by_quat(Scalar[DTYPE](0), Scalar[DTYPE](0), Scalar[DTYPE](1), b_qx, b_qy, b_qz, b_qw)

    # Vector from A center to B center
    var t_x = b_x - a_x
    var t_y = b_y - a_y
    var t_z = b_z - a_z

    # Track minimum penetration axis
    var min_pen = Scalar[DTYPE](1e10)  # Large positive = no contact
    var best_nx = Scalar[DTYPE](0.0)
    var best_ny = Scalar[DTYPE](0.0)
    var best_nz = Scalar[DTYPE](1.0)

    # Separated result for early exit
    var SEPARATED = (Scalar[DTYPE](1.0), a_x, a_y, a_z, Scalar[DTYPE](0), Scalar[DTYPE](0), Scalar[DTYPE](1))

    # Test 15 axes

    # Box A face normals (3 axes)
    if _test_sat_axis(a0[0], a0[1], a0[2], t_x, t_y, t_z,
            a0[0], a0[1], a0[2], a1[0], a1[1], a1[2], a2[0], a2[1], a2[2], a_hx, a_hy, a_hz,
            b0[0], b0[1], b0[2], b1[0], b1[1], b1[2], b2[0], b2[1], b2[2], b_hx, b_hy, b_hz,
            min_pen, best_nx, best_ny, best_nz):
        return SEPARATED
    if _test_sat_axis(a1[0], a1[1], a1[2], t_x, t_y, t_z,
            a0[0], a0[1], a0[2], a1[0], a1[1], a1[2], a2[0], a2[1], a2[2], a_hx, a_hy, a_hz,
            b0[0], b0[1], b0[2], b1[0], b1[1], b1[2], b2[0], b2[1], b2[2], b_hx, b_hy, b_hz,
            min_pen, best_nx, best_ny, best_nz):
        return SEPARATED
    if _test_sat_axis(a2[0], a2[1], a2[2], t_x, t_y, t_z,
            a0[0], a0[1], a0[2], a1[0], a1[1], a1[2], a2[0], a2[1], a2[2], a_hx, a_hy, a_hz,
            b0[0], b0[1], b0[2], b1[0], b1[1], b1[2], b2[0], b2[1], b2[2], b_hx, b_hy, b_hz,
            min_pen, best_nx, best_ny, best_nz):
        return SEPARATED

    # Box B face normals (3 axes)
    if _test_sat_axis(b0[0], b0[1], b0[2], t_x, t_y, t_z,
            a0[0], a0[1], a0[2], a1[0], a1[1], a1[2], a2[0], a2[1], a2[2], a_hx, a_hy, a_hz,
            b0[0], b0[1], b0[2], b1[0], b1[1], b1[2], b2[0], b2[1], b2[2], b_hx, b_hy, b_hz,
            min_pen, best_nx, best_ny, best_nz):
        return SEPARATED
    if _test_sat_axis(b1[0], b1[1], b1[2], t_x, t_y, t_z,
            a0[0], a0[1], a0[2], a1[0], a1[1], a1[2], a2[0], a2[1], a2[2], a_hx, a_hy, a_hz,
            b0[0], b0[1], b0[2], b1[0], b1[1], b1[2], b2[0], b2[1], b2[2], b_hx, b_hy, b_hz,
            min_pen, best_nx, best_ny, best_nz):
        return SEPARATED
    if _test_sat_axis(b2[0], b2[1], b2[2], t_x, t_y, t_z,
            a0[0], a0[1], a0[2], a1[0], a1[1], a1[2], a2[0], a2[1], a2[2], a_hx, a_hy, a_hz,
            b0[0], b0[1], b0[2], b1[0], b1[1], b1[2], b2[0], b2[1], b2[2], b_hx, b_hy, b_hz,
            min_pen, best_nx, best_ny, best_nz):
        return SEPARATED

    # Edge-edge cross products (9 axes)
    # A0 x B0
    var c_x = a0[1] * b0[2] - a0[2] * b0[1]
    var c_y = a0[2] * b0[0] - a0[0] * b0[2]
    var c_z = a0[0] * b0[1] - a0[1] * b0[0]
    if _test_sat_axis(c_x, c_y, c_z, t_x, t_y, t_z,
            a0[0], a0[1], a0[2], a1[0], a1[1], a1[2], a2[0], a2[1], a2[2], a_hx, a_hy, a_hz,
            b0[0], b0[1], b0[2], b1[0], b1[1], b1[2], b2[0], b2[1], b2[2], b_hx, b_hy, b_hz,
            min_pen, best_nx, best_ny, best_nz):
        return SEPARATED

    # A0 x B1
    c_x = a0[1] * b1[2] - a0[2] * b1[1]
    c_y = a0[2] * b1[0] - a0[0] * b1[2]
    c_z = a0[0] * b1[1] - a0[1] * b1[0]
    if _test_sat_axis(c_x, c_y, c_z, t_x, t_y, t_z,
            a0[0], a0[1], a0[2], a1[0], a1[1], a1[2], a2[0], a2[1], a2[2], a_hx, a_hy, a_hz,
            b0[0], b0[1], b0[2], b1[0], b1[1], b1[2], b2[0], b2[1], b2[2], b_hx, b_hy, b_hz,
            min_pen, best_nx, best_ny, best_nz):
        return SEPARATED

    # A0 x B2
    c_x = a0[1] * b2[2] - a0[2] * b2[1]
    c_y = a0[2] * b2[0] - a0[0] * b2[2]
    c_z = a0[0] * b2[1] - a0[1] * b2[0]
    if _test_sat_axis(c_x, c_y, c_z, t_x, t_y, t_z,
            a0[0], a0[1], a0[2], a1[0], a1[1], a1[2], a2[0], a2[1], a2[2], a_hx, a_hy, a_hz,
            b0[0], b0[1], b0[2], b1[0], b1[1], b1[2], b2[0], b2[1], b2[2], b_hx, b_hy, b_hz,
            min_pen, best_nx, best_ny, best_nz):
        return SEPARATED

    # A1 x B0
    c_x = a1[1] * b0[2] - a1[2] * b0[1]
    c_y = a1[2] * b0[0] - a1[0] * b0[2]
    c_z = a1[0] * b0[1] - a1[1] * b0[0]
    if _test_sat_axis(c_x, c_y, c_z, t_x, t_y, t_z,
            a0[0], a0[1], a0[2], a1[0], a1[1], a1[2], a2[0], a2[1], a2[2], a_hx, a_hy, a_hz,
            b0[0], b0[1], b0[2], b1[0], b1[1], b1[2], b2[0], b2[1], b2[2], b_hx, b_hy, b_hz,
            min_pen, best_nx, best_ny, best_nz):
        return SEPARATED

    # A1 x B1
    c_x = a1[1] * b1[2] - a1[2] * b1[1]
    c_y = a1[2] * b1[0] - a1[0] * b1[2]
    c_z = a1[0] * b1[1] - a1[1] * b1[0]
    if _test_sat_axis(c_x, c_y, c_z, t_x, t_y, t_z,
            a0[0], a0[1], a0[2], a1[0], a1[1], a1[2], a2[0], a2[1], a2[2], a_hx, a_hy, a_hz,
            b0[0], b0[1], b0[2], b1[0], b1[1], b1[2], b2[0], b2[1], b2[2], b_hx, b_hy, b_hz,
            min_pen, best_nx, best_ny, best_nz):
        return SEPARATED

    # A1 x B2
    c_x = a1[1] * b2[2] - a1[2] * b2[1]
    c_y = a1[2] * b2[0] - a1[0] * b2[2]
    c_z = a1[0] * b2[1] - a1[1] * b2[0]
    if _test_sat_axis(c_x, c_y, c_z, t_x, t_y, t_z,
            a0[0], a0[1], a0[2], a1[0], a1[1], a1[2], a2[0], a2[1], a2[2], a_hx, a_hy, a_hz,
            b0[0], b0[1], b0[2], b1[0], b1[1], b1[2], b2[0], b2[1], b2[2], b_hx, b_hy, b_hz,
            min_pen, best_nx, best_ny, best_nz):
        return SEPARATED

    # A2 x B0
    c_x = a2[1] * b0[2] - a2[2] * b0[1]
    c_y = a2[2] * b0[0] - a2[0] * b0[2]
    c_z = a2[0] * b0[1] - a2[1] * b0[0]
    if _test_sat_axis(c_x, c_y, c_z, t_x, t_y, t_z,
            a0[0], a0[1], a0[2], a1[0], a1[1], a1[2], a2[0], a2[1], a2[2], a_hx, a_hy, a_hz,
            b0[0], b0[1], b0[2], b1[0], b1[1], b1[2], b2[0], b2[1], b2[2], b_hx, b_hy, b_hz,
            min_pen, best_nx, best_ny, best_nz):
        return SEPARATED

    # A2 x B1
    c_x = a2[1] * b1[2] - a2[2] * b1[1]
    c_y = a2[2] * b1[0] - a2[0] * b1[2]
    c_z = a2[0] * b1[1] - a2[1] * b1[0]
    if _test_sat_axis(c_x, c_y, c_z, t_x, t_y, t_z,
            a0[0], a0[1], a0[2], a1[0], a1[1], a1[2], a2[0], a2[1], a2[2], a_hx, a_hy, a_hz,
            b0[0], b0[1], b0[2], b1[0], b1[1], b1[2], b2[0], b2[1], b2[2], b_hx, b_hy, b_hz,
            min_pen, best_nx, best_ny, best_nz):
        return SEPARATED

    # A2 x B2
    c_x = a2[1] * b2[2] - a2[2] * b2[1]
    c_y = a2[2] * b2[0] - a2[0] * b2[2]
    c_z = a2[0] * b2[1] - a2[1] * b2[0]
    if _test_sat_axis(c_x, c_y, c_z, t_x, t_y, t_z,
            a0[0], a0[1], a0[2], a1[0], a1[1], a1[2], a2[0], a2[1], a2[2], a_hx, a_hy, a_hz,
            b0[0], b0[1], b0[2], b1[0], b1[1], b1[2], b2[0], b2[1], b2[2], b_hx, b_hy, b_hz,
            min_pen, best_nx, best_ny, best_nz):
        return SEPARATED

    # No separating axis found - boxes are colliding
    # Signed distance is negative penetration
    var dist = -min_pen

    # Contact point: approximate as midpoint between centers
    var contact_x = (a_x + b_x) * Scalar[DTYPE](0.5)
    var contact_y = (a_y + b_y) * Scalar[DTYPE](0.5)
    var contact_z = (a_z + b_z) * Scalar[DTYPE](0.5)

    return (dist, contact_x, contact_y, contact_z, best_nx, best_ny, best_nz)
