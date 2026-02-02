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
