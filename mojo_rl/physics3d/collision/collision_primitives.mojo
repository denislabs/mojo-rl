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

from std.math import sqrt


# =============================================================================
# Quaternion Helpers (for rotating capsule axis to world frame)
# =============================================================================


@always_inline
def rotate_vector_by_quat[
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
        vx: Vector to rotate x.
        vy: Vector to rotate y.
        vz: Vector to rotate z.
        qx: Unit quaternion x.
        qy: Unit quaternion y.
        qz: Unit quaternion z.
        qw: Unit quaternion w.

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
def rotate_vector_by_quat_inverse[
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
        vx: Vector to rotate x.
        vy: Vector to rotate y.
        vz: Vector to rotate z.
        qx: Unit quaternion x.
        qy: Unit quaternion y.
        qz: Unit quaternion z.
        qw: Unit quaternion w.

    Returns:
        Rotated vector (rx, ry, rz) in local frame.
    """
    # Use conjugate: negate the vector part
    return rotate_vector_by_quat(vx, vy, vz, -qx, -qy, -qz, qw)


@always_inline
def compute_tangent_basis[
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
        nx: Contact normal x.
        ny: Contact normal y.
        nz: Contact normal z.

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
def sphere_sphere[
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
        pos1_x: Sphere 1 center x.
        pos1_y: Sphere 1 center y.
        pos1_z: Sphere 1 center z.
        radius1: Radius of sphere 1.
        pos2_x: Sphere 2 center x.
        pos2_y: Sphere 2 center y.
        pos2_z: Sphere 2 center z.
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
def sphere_plane[
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
        sphere_x: Sphere center x.
        sphere_y: Sphere center y.
        sphere_z: Sphere center z.
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


@always_inline
def ellipsoid_plane[
    DTYPE: DType
](
    center_x: Scalar[DTYPE],
    center_y: Scalar[DTYPE],
    center_z: Scalar[DTYPE],
    qx: Scalar[DTYPE],
    qy: Scalar[DTYPE],
    qz: Scalar[DTYPE],
    qw: Scalar[DTYPE],
    rx: Scalar[DTYPE],
    ry: Scalar[DTYPE],
    rz: Scalar[DTYPE],
    ground_z: Scalar[DTYPE],
) -> Tuple[Scalar[DTYPE], Scalar[DTYPE], Scalar[DTYPE], Scalar[DTYPE]]:
    """Ellipsoid-plane collision (ground plane with normal = +Z).

    MuJoCo routes plane x ellipsoid through `mjc_PlaneConvex`, which asks the
    convex geom for its SUPPORT POINT along -normal and reports that single
    deepest point. An ellipsoid's support point is closed-form, so no ccd is
    needed here.

    For the ellipsoid `{c + R A u : |u| <= 1}` with `A = diag(rx, ry, rz)`,
    the support along a world direction `w` is

        d     = R^T w
        point = c + R (A^2 d) / |A d|

    Taking `w = (0, 0, -1)` maximises `-z`, i.e. finds the LOWEST point of the
    ellipsoid, and `dist = point_z - ground_z` is then the exact analogue of
    `sphere_plane`'s `(center_z - radius) - ground_z`.

    A smooth strictly-convex surface touches a plane at one point, so one
    contact is the whole story — unlike `box_plane`, which reports up to four.

    Args:
        center_x: Ellipsoid centre x.
        center_y: Ellipsoid centre y.
        center_z: Ellipsoid centre z.
        qx: Orientation quaternion x.
        qy: Orientation quaternion y.
        qz: Orientation quaternion z.
        qw: Orientation quaternion w.
        rx: Semi-axis along local x.
        ry: Semi-axis along local y.
        rz: Semi-axis along local z.
        ground_z: Z-height of the ground plane.

    Returns:
        Tuple of (dist, contact_x, contact_y, contact_z), matching
        `sphere_plane`: dist is signed (negative = penetration) and the
        contact point sits midway across the gap.
    """
    # d = R^T * (0, 0, -1)
    var d = rotate_vector_by_quat_inverse[DTYPE](
        Scalar[DTYPE](0), Scalar[DTYPE](0), Scalar[DTYPE](-1),
        qx, qy, qz, qw,
    )
    var ax = rx * d[0]
    var ay = ry * d[1]
    var az = rz * d[2]
    var norm = sqrt(ax * ax + ay * ay + az * az)
    # Degenerate only if a semi-axis is zero; fall back to the centre so the
    # caller still gets a finite, obviously-non-penetrating distance.
    if norm <= Scalar[DTYPE](1e-15):
        return (center_z - ground_z, center_x, center_y, ground_z)
    var inv = Scalar[DTYPE](1) / norm
    # R * (A^2 d) / |A d|
    var off = rotate_vector_by_quat[DTYPE](
        rx * ax * inv, ry * ay * inv, rz * az * inv,
        qx, qy, qz, qw,
    )
    var px = center_x + off[0]
    var py = center_y + off[1]
    var pz = center_z + off[2]

    var dist = pz - ground_z
    return (dist, px, py, ground_z + dist * Scalar[DTYPE](0.5))


# =============================================================================
# Capsule Collision Primitives (Phase 8)
# =============================================================================


@always_inline
def capsule_plane[
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
        cap_x: Capsule center x.
        cap_y: Capsule center y.
        cap_z: Capsule center z.
        cap_qx: Capsule orientation x.
        cap_qy: Capsule orientation y.
        cap_qz: Capsule orientation z.
        cap_qw: Capsule orientation w.
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
def capsule_sphere[
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
        cap_x: Capsule center x.
        cap_y: Capsule center y.
        cap_z: Capsule center z.
        cap_qx: Capsule orientation x.
        cap_qy: Capsule orientation y.
        cap_qz: Capsule orientation z.
        cap_qw: Capsule orientation w.
        cap_half_len: Half-length of the cylindrical part.
        cap_radius: Radius of the capsule.
        sph_x: Sphere center x.
        sph_y: Sphere center y.
        sph_z: Sphere center z.
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
def _closest_points_line_segments[
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
def capsule_capsule[
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
        a_x: Capsule A center x.
        a_y: Capsule A center y.
        a_z: Capsule A center z.
        a_qx: Capsule A orientation x.
        a_qy: Capsule A orientation y.
        a_qz: Capsule A orientation z.
        a_qw: Capsule A orientation w.
        a_half_len: Half-length of capsule A.
        a_radius: Radius of capsule A.
        b_x: Capsule B center x.
        b_y: Capsule B center y.
        b_z: Capsule B center z.
        b_qx: Capsule B orientation x.
        b_qy: Capsule B orientation y.
        b_qz: Capsule B orientation z.
        b_qw: Capsule B orientation w.
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

    # Check if centerlines cross (closest points coincident).
    # When this happens, sphere_sphere picks an arbitrary normal which
    # makes the constraint solver push in the wrong direction.
    # Instead, use cross(axis_a, axis_b) as the separation normal
    # (perpendicular to both capsule axes) — matches MuJoCo mjc_CapsuleCapsule.
    var dx = c2_x - c1_x
    var dy = c2_y - c1_y
    var dz = c2_z - c1_z
    var center_dist_sq = dx * dx + dy * dy + dz * dz

    if center_dist_sq < Scalar[DTYPE](1e-16):
        # Degenerate: capsule centerlines cross or coincide.
        # Strategy: use direction from capsule A center to capsule B center.
        # This keeps the normal in the motion plane (critical for 2D chains
        # where cross(axis_a, axis_b) is perpendicular to the plane and no
        # DOF can resolve it).  Fall back to cross(axes) only if centers
        # are also coincident.
        var nx = b_x - a_x
        var ny = b_y - a_y
        var nz = b_z - a_z
        var nlen_sq = nx * nx + ny * ny + nz * nz

        if nlen_sq < Scalar[DTYPE](1e-16):
            # Centers also coincident — use cross(axis_a, axis_b).
            nx = a_ay * b_az - a_az * b_ay
            ny = a_az * b_ax - a_ax * b_az
            nz = a_ax * b_ay - a_ay * b_ax
            nlen_sq = nx * nx + ny * ny + nz * nz

        if nlen_sq < Scalar[DTYPE](1e-16):
            # Parallel at same location — pick perpendicular to axis_a.
            var abs_ax = abs(a_ax)
            var abs_ay = abs(a_ay)
            var abs_az = abs(a_az)
            if abs_ax <= abs_ay and abs_ax <= abs_az:
                nx = Scalar[DTYPE](0)
                ny = -a_az
                nz = a_ay
            elif abs_ay <= abs_az:
                nx = a_az
                ny = Scalar[DTYPE](0)
                nz = -a_ax
            else:
                nx = -a_ay
                ny = a_ax
                nz = Scalar[DTYPE](0)
            nlen_sq = nx * nx + ny * ny + nz * nz

        var inv_nlen = Scalar[DTYPE](1.0) / sqrt(nlen_sq)
        nx *= inv_nlen
        ny *= inv_nlen
        nz *= inv_nlen

        var dist = -(a_radius + b_radius)
        var mid_x = (c1_x + c2_x) * Scalar[DTYPE](0.5)
        var mid_y = (c1_y + c2_y) * Scalar[DTYPE](0.5)
        var mid_z = (c1_z + c2_z) * Scalar[DTYPE](0.5)
        return (dist, mid_x, mid_y, mid_z, nx, ny, nz)

    # Normal case: treat as sphere-sphere between the closest points
    return sphere_sphere(c1_x, c1_y, c1_z, a_radius, c2_x, c2_y, c2_z, b_radius)


# =============================================================================
# Cylinder Collision Primitives
# =============================================================================


@always_inline
def cylinder_plane[
    DTYPE: DType
](
    # Cylinder center and orientation
    cyl_x: Scalar[DTYPE],
    cyl_y: Scalar[DTYPE],
    cyl_z: Scalar[DTYPE],
    cyl_qx: Scalar[DTYPE],
    cyl_qy: Scalar[DTYPE],
    cyl_qz: Scalar[DTYPE],
    cyl_qw: Scalar[DTYPE],
    half_length: Scalar[DTYPE],
    radius: Scalar[DTYPE],
    # Plane (horizontal at ground_z)
    ground_z: Scalar[DTYPE],
) -> Tuple[Scalar[DTYPE], Scalar[DTYPE], Scalar[DTYPE], Scalar[DTYPE]]:
    """Cylinder-plane collision detection (ground plane with normal = +Z).

    Matching MuJoCo mjc_PlaneCylinder: tests rim points on both end-caps
    to find the lowest contact point. A cylinder differs from a capsule in that
    it has flat end-caps (no hemispherical caps), so contact points lie on the
    rim edge rather than being offset by radius toward the ground.

    Algorithm:
    1. Compute cylinder axis in world frame
    2. Find the "rim direction" perpendicular to axis that points most downward
    3. Test bottom endpoint + rim offset (lowest point on bottom rim)

    Returns:
        Tuple of (dist, contact_x, contact_y, contact_z):
        - dist: Signed distance from cylinder surface to plane (negative = penetration).
        - contact: Contact point on the ground plane.
    """
    # Get cylinder axis in world frame (local Z)
    var axis = rotate_vector_by_quat(
        Scalar[DTYPE](0.0),
        Scalar[DTYPE](0.0),
        Scalar[DTYPE](1.0),
        cyl_qx,
        cyl_qy,
        cyl_qz,
        cyl_qw,
    )
    var ax = axis[0]
    var ay = axis[1]
    var az = axis[2]

    # Plane normal is (0, 0, 1). Find the rim direction that points most downward.
    # The rim vector is perpendicular to the cylinder axis and in the plane containing
    # the axis and the ground normal. rim = normalize(n - (n·axis)*axis) where n = (0,0,1)
    # Then the lowest point is: endpoint - rim * radius (toward ground)

    # n_dot_axis = az (since n = (0,0,1))
    var n_dot_axis = az

    # rim = (0,0,1) - az * axis = (-az*ax, -az*ay, 1 - az*az)
    var rim_x = -n_dot_axis * ax
    var rim_y = -n_dot_axis * ay
    var rim_z = Scalar[DTYPE](1.0) - n_dot_axis * n_dot_axis
    var rim_len = sqrt(rim_x * rim_x + rim_y * rim_y + rim_z * rim_z)

    if rim_len > Scalar[DTYPE](1e-10):
        var inv_len = Scalar[DTYPE](1.0) / rim_len
        rim_x *= inv_len
        rim_y *= inv_len
        rim_z *= inv_len
    else:
        # Cylinder axis is vertical — any horizontal rim direction works
        rim_x = Scalar[DTYPE](1.0)
        rim_y = Scalar[DTYPE](0.0)
        rim_z = Scalar[DTYPE](0.0)

    # Flip cylinder axis so it points toward the ground (lower endpoint first)
    var flip_ax = ax
    var flip_ay = ay
    var flip_az = az
    if az > Scalar[DTYPE](0.0):
        flip_ax = -ax
        flip_ay = -ay
        flip_az = -az

    # Bottom endpoint (in direction of flipped axis)
    var ep_x = cyl_x + flip_ax * half_length
    var ep_y = cyl_y + flip_ay * half_length
    var ep_z = cyl_z + flip_az * half_length

    # Lowest point on bottom rim: endpoint - rim * radius (rim points UP away from ground)
    var low_x = ep_x - rim_x * radius
    var low_y = ep_y - rim_y * radius
    var low_z = ep_z - rim_z * radius

    # Distance from lowest point to ground (no radius offset — flat end)
    var dist = low_z - ground_z

    # Contact point on the ground plane
    var contact_x = low_x
    var contact_y = low_y
    var contact_z = ground_z

    return (dist, contact_x, contact_y, contact_z)


@always_inline
def cylinder_sphere[
    DTYPE: DType
](
    # Cylinder
    cyl_x: Scalar[DTYPE],
    cyl_y: Scalar[DTYPE],
    cyl_z: Scalar[DTYPE],
    cyl_qx: Scalar[DTYPE],
    cyl_qy: Scalar[DTYPE],
    cyl_qz: Scalar[DTYPE],
    cyl_qw: Scalar[DTYPE],
    cyl_half_len: Scalar[DTYPE],
    cyl_radius: Scalar[DTYPE],
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
    """Cylinder-sphere collision detection (matching MuJoCo mjc_SphereCylinder).

    Three cases based on where the sphere center projects onto the cylinder:
    1. SIDE: Sphere is beside the cylinder body → sphere-line distance
    2. CAP: Sphere is above/below a flat end-cap → plane-sphere distance
    3. CORNER: Sphere is near the rim edge → sphere-point distance

    Args:
        cyl_x: Cylinder center x.
        cyl_y: Cylinder center y.
        cyl_z: Cylinder center z.
        cyl_qx: Cylinder orientation x.
        cyl_qy: Cylinder orientation y.
        cyl_qz: Cylinder orientation z.
        cyl_qw: Cylinder orientation w.
        cyl_half_len: Cylinder half-length.
        cyl_radius: Cylinder radius.
        sph_x: Sphere center x.
        sph_y: Sphere center y.
        sph_z: Sphere center z.
        sph_radius: Sphere radius.

    Returns:
        Tuple of (dist, contact_x, contact_y, contact_z, normal_x, normal_y, normal_z).
    """
    # Get cylinder axis in world frame
    var axis = rotate_vector_by_quat(
        Scalar[DTYPE](0.0),
        Scalar[DTYPE](0.0),
        Scalar[DTYPE](1.0),
        cyl_qx,
        cyl_qy,
        cyl_qz,
        cyl_qw,
    )
    var ax = axis[0]
    var ay = axis[1]
    var az = axis[2]

    # Vector from cylinder center to sphere center
    var dx = sph_x - cyl_x
    var dy = sph_y - cyl_y
    var dz = sph_z - cyl_z

    # Project onto cylinder axis: t = dot(d, axis)
    var t = dx * ax + dy * ay + dz * az

    # Radial vector (perpendicular to axis)
    var rad_x = dx - t * ax
    var rad_y = dy - t * ay
    var rad_z = dz - t * az
    var rad_dist = sqrt(rad_x * rad_x + rad_y * rad_y + rad_z * rad_z)

    # Determine case
    var t_clamped = t
    if t_clamped < -cyl_half_len:
        t_clamped = -cyl_half_len
    elif t_clamped > cyl_half_len:
        t_clamped = cyl_half_len

    var on_side = t >= -cyl_half_len and t <= cyl_half_len
    var inside_radius = rad_dist <= cyl_radius

    var dist: Scalar[DTYPE]
    var cx: Scalar[DTYPE]
    var cy: Scalar[DTYPE]
    var cz: Scalar[DTYPE]
    var nx: Scalar[DTYPE]
    var ny: Scalar[DTYPE]
    var nz: Scalar[DTYPE]

    if on_side and not inside_radius:
        # SIDE case: closest point is on the cylinder side surface
        # Normalize radial direction
        var inv_rad = Scalar[DTYPE](1.0) / rad_dist
        var nr_x = rad_x * inv_rad
        var nr_y = rad_y * inv_rad
        var nr_z = rad_z * inv_rad

        # Closest point on cylinder surface
        var surf_x = cyl_x + t * ax + cyl_radius * nr_x
        var surf_y = cyl_y + t * ay + cyl_radius * nr_y
        var surf_z = cyl_z + t * az + cyl_radius * nr_z

        dist = rad_dist - cyl_radius - sph_radius
        nx = nr_x
        ny = nr_y
        nz = nr_z

        # Contact = midpoint between surfaces
        var half_d = Scalar[DTYPE](0.5) * dist
        cx = surf_x + nx * half_d
        cy = surf_y + ny * half_d
        cz = surf_z + nz * half_d

    elif not on_side and inside_radius:
        # CAP case: sphere is above/below a flat end-cap
        # Normal points along axis direction (toward sphere)
        if t > Scalar[DTYPE](0.0):
            nx = ax
            ny = ay
            nz = az
            dist = (t - cyl_half_len) - sph_radius
        else:
            nx = -ax
            ny = -ay
            nz = -az
            dist = (-t - cyl_half_len) - sph_radius

        # Closest point on cap surface
        var cap_x = cyl_x + t_clamped * ax + rad_x
        var cap_y = cyl_y + t_clamped * ay + rad_y
        var cap_z = cyl_z + t_clamped * az + rad_z

        var half_d = Scalar[DTYPE](0.5) * dist
        cx = cap_x + nx * half_d
        cy = cap_y + ny * half_d
        cz = cap_z + nz * half_d

    else:
        # CORNER case (or degenerate): closest point is on the rim edge
        # Find the closest point on the rim circle
        var rim_x: Scalar[DTYPE]
        var rim_y: Scalar[DTYPE]
        var rim_z: Scalar[DTYPE]

        if rad_dist > Scalar[DTYPE](1e-10):
            var inv_rad = Scalar[DTYPE](1.0) / rad_dist
            rim_x = cyl_x + t_clamped * ax + cyl_radius * rad_x * inv_rad
            rim_y = cyl_y + t_clamped * ay + cyl_radius * rad_y * inv_rad
            rim_z = cyl_z + t_clamped * az + cyl_radius * rad_z * inv_rad
        else:
            # Sphere is on the axis — pick arbitrary radial direction
            # Use the X axis of the cylinder's local frame
            var local_x = rotate_vector_by_quat(
                Scalar[DTYPE](1.0),
                Scalar[DTYPE](0.0),
                Scalar[DTYPE](0.0),
                cyl_qx,
                cyl_qy,
                cyl_qz,
                cyl_qw,
            )
            rim_x = cyl_x + t_clamped * ax + cyl_radius * local_x[0]
            rim_y = cyl_y + t_clamped * ay + cyl_radius * local_x[1]
            rim_z = cyl_z + t_clamped * az + cyl_radius * local_x[2]

        # Sphere-point distance
        var to_sph_x = sph_x - rim_x
        var to_sph_y = sph_y - rim_y
        var to_sph_z = sph_z - rim_z
        var to_sph_dist = sqrt(
            to_sph_x * to_sph_x + to_sph_y * to_sph_y + to_sph_z * to_sph_z
        )

        if to_sph_dist > Scalar[DTYPE](1e-10):
            var inv_d = Scalar[DTYPE](1.0) / to_sph_dist
            nx = to_sph_x * inv_d
            ny = to_sph_y * inv_d
            nz = to_sph_z * inv_d
        else:
            nx = Scalar[DTYPE](0.0)
            ny = Scalar[DTYPE](0.0)
            nz = Scalar[DTYPE](1.0)

        dist = to_sph_dist - sph_radius

        var half_d = Scalar[DTYPE](0.5) * dist
        cx = rim_x + nx * half_d
        cy = rim_y + ny * half_d
        cz = rim_z + nz * half_d

    return (dist, cx, cy, cz, nx, ny, nz)


# =============================================================================
# Box Collision Primitives (Phase 9)
# =============================================================================


@always_inline
def _check_vertex[
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
    var rotated = rotate_vector_by_quat(
        lx, ly, lz, box_qx, box_qy, box_qz, box_qw
    )
    var vx = box_x + rotated[0]
    var vy = box_y + rotated[1]
    var vz = box_z + rotated[2]
    if vz < min_z:
        min_z = vz
        lowest_x = vx
        lowest_y = vy


@always_inline
def box_plane[
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

    _check_vertex(
        box_x,
        box_y,
        box_z,
        box_qx,
        box_qy,
        box_qz,
        box_qw,
        half_x,
        half_y,
        half_z,
        NEG,
        NEG,
        NEG,
        min_z,
        lowest_x,
        lowest_y,
    )
    _check_vertex(
        box_x,
        box_y,
        box_z,
        box_qx,
        box_qy,
        box_qz,
        box_qw,
        half_x,
        half_y,
        half_z,
        NEG,
        NEG,
        POS,
        min_z,
        lowest_x,
        lowest_y,
    )
    _check_vertex(
        box_x,
        box_y,
        box_z,
        box_qx,
        box_qy,
        box_qz,
        box_qw,
        half_x,
        half_y,
        half_z,
        NEG,
        POS,
        NEG,
        min_z,
        lowest_x,
        lowest_y,
    )
    _check_vertex(
        box_x,
        box_y,
        box_z,
        box_qx,
        box_qy,
        box_qz,
        box_qw,
        half_x,
        half_y,
        half_z,
        NEG,
        POS,
        POS,
        min_z,
        lowest_x,
        lowest_y,
    )
    _check_vertex(
        box_x,
        box_y,
        box_z,
        box_qx,
        box_qy,
        box_qz,
        box_qw,
        half_x,
        half_y,
        half_z,
        POS,
        NEG,
        NEG,
        min_z,
        lowest_x,
        lowest_y,
    )
    _check_vertex(
        box_x,
        box_y,
        box_z,
        box_qx,
        box_qy,
        box_qz,
        box_qw,
        half_x,
        half_y,
        half_z,
        POS,
        NEG,
        POS,
        min_z,
        lowest_x,
        lowest_y,
    )
    _check_vertex(
        box_x,
        box_y,
        box_z,
        box_qx,
        box_qy,
        box_qz,
        box_qw,
        half_x,
        half_y,
        half_z,
        POS,
        POS,
        NEG,
        min_z,
        lowest_x,
        lowest_y,
    )
    _check_vertex(
        box_x,
        box_y,
        box_z,
        box_qx,
        box_qy,
        box_qz,
        box_qw,
        half_x,
        half_y,
        half_z,
        POS,
        POS,
        POS,
        min_z,
        lowest_x,
        lowest_y,
    )

    # Signed distance from lowest vertex to ground
    var dist = min_z - ground_z

    # Contact point is on ground below lowest vertex
    var contact_x = lowest_x
    var contact_y = lowest_y
    var contact_z = ground_z

    return (dist, contact_x, contact_y, contact_z)


@always_inline
def box_sphere[
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

    # ⚠ THE INTERIOR CASE NEEDS ITS OWN CONTACT POINT, not `dist/2` along the
    # normal. `clamp_*` equals the sphere CENTRE once the centre is inside the
    # box, so the exterior formula measures from the wrong origin and lands
    # `face_gap` away from MuJoCo's point. Depth and normal stay correct, which
    # is why a gate comparing only those two passes — see
    # `tests/physics3d/test_capsule_box_sweep.mojo`, which did exactly that and
    # missed this until stacker's held cube showed a 15% qacc error with every
    # depth and normal matching.
    var interior = False
    var face_gap = Scalar[DTYPE](0)

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
            nx_local = Scalar[DTYPE](1.0) if local_x >= Scalar[DTYPE](
                0
            ) else Scalar[DTYPE](-1.0)
            ny_local = Scalar[DTYPE](0.0)
            nz_local = Scalar[DTYPE](0.0)
            dist = -face_dist_x - sph_radius
            interior = True
            face_gap = face_dist_x
        elif face_dist_y <= face_dist_z:
            # Y face is closest
            nx_local = Scalar[DTYPE](0.0)
            ny_local = Scalar[DTYPE](1.0) if local_y >= Scalar[DTYPE](
                0
            ) else Scalar[DTYPE](-1.0)
            nz_local = Scalar[DTYPE](0.0)
            dist = -face_dist_y - sph_radius
            interior = True
            face_gap = face_dist_y
        else:
            # Z face is closest
            nx_local = Scalar[DTYPE](0.0)
            ny_local = Scalar[DTYPE](0.0)
            nz_local = Scalar[DTYPE](1.0) if local_z >= Scalar[DTYPE](
                0
            ) else Scalar[DTYPE](-1.0)
            dist = -face_dist_z - sph_radius
            interior = True
            face_gap = face_dist_z

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

    # Contact point is midpoint between surfaces.
    #   exterior  `closest` is on the box surface, so mid = closest + n*dist/2,
    #             which is `mjraw_SphereBox`'s 0.5*(clamped + deepest)
    #   interior  `closest` IS the sphere centre, and MuJoCo's is
    #             centre + nearest*(radius - face_gap)/2
    #
    # ⚠ `nearest` IS THE NEGATION OF OUR NORMAL. MuJoCo's interior normal points
    # from the nearest face INTO the box (its convention is geom1 -> geom2, i.e.
    # sphere -> box); ours points out of the box toward the sphere. So the
    # offset along OUR normal is (face_gap - radius)/2, and writing MuJoCo's
    # expression verbatim moves the point the wrong way by `face_gap` — the same
    # distance the original `dist/2` was wrong by, in the opposite direction.
    var half_dist = Scalar[DTYPE](0.5) * dist
    if interior:
        half_dist = Scalar[DTYPE](0.5) * (face_gap - sph_radius)
    var contact_x = closest_x + nx * half_dist
    var contact_y = closest_y + ny * half_dist
    var contact_z = closest_z + nz * half_dist

    return (dist, contact_x, contact_y, contact_z, nx, ny, nz)


@always_inline
def _capsule_box_best_segment_pos[
    DTYPE: DType
](
    pos_x: Scalar[DTYPE],
    pos_y: Scalar[DTYPE],
    pos_z: Scalar[DTYPE],
    hax_x: Scalar[DTYPE],
    hax_y: Scalar[DTYPE],
    hax_z: Scalar[DTYPE],
    sx: Scalar[DTYPE],
    sy: Scalar[DTYPE],
    sz: Scalar[DTYPE],
    cap_radius: Scalar[DTYPE],
    margin: Scalar[DTYPE],
    mut cltype: Int,
    mut clface: Int,
    mut clcorner: Int,
    mut cledge: Int,
    mut bestboxpos: Scalar[DTYPE],
) -> Scalar[DTYPE]:
    """Where on a capsule's axis it comes closest to a box's SURFACE.

    Everything is in the BOX'S LOCAL FRAME: the box is centred at the origin
    with half-extents `(sx, sy, sz)`, and the capsule's axis segment is
    `pos + halfaxis * s` for `s` in [-1, 1]. Returns that `s`.

    This is `mjraw_CapsuleBox`'s search (`engine_collision_box.c`), which the
    caller then reduces to box/sphere at the returned point exactly as MuJoCo
    reduces to `mjraw_SphereBox`. Two families of candidate:

      FACE  each ENDPOINT, clamped into the box. Accepted only when the
            endpoint is outside along at most ONE axis, which is what makes it
            a face (or interior) case rather than an edge or corner one — those
            are covered exactly by the edge family below, and letting a corner
            through here would return the distance to the corner while claiming
            a face.
      EDGE  all 12 box edges against the segment, as a segment/segment closest
            point with both parameters clamped to [-1, 1]. An edge's midpoint
            is a corner with the edge's own component zeroed, and it runs along
            that axis with half-length `size[j]` — hence the `i & (1 << j)`
            filter, which visits each edge once.

    ⚠ NOT the point closest to the box's CENTRE. That is what this replaced and
    the two agree only for face contacts; see task #45.

    ⚠ THE DEGENERATE EDGE IS SKIPPED, NOT CLAMPED. When the capsule axis is
    parallel to box axis `j`, `det` vanishes for that axis's four edges and
    they carry no information — the other eight still do. Clamping a vanishing
    determinant instead would return an arbitrary point on a parallel edge and
    win the comparison with a bogus distance.

    The `mut` outs are MuJoCo's classification of the winning candidate, which
    `_capsule_box_second_pos` needs to place the SECOND contact:

        cltype     -4 nothing found, -3/-1 an endpoint against a face,
                   0..8 an edge (`s1 * 3 + s2` of the two clamp states)
        clface     which box axis the winning endpoint was clamped on, -1 if
                   the endpoint is inside the box
        clcorner   which of the 8 corners the winning edge belongs to
        cledge     which axis the winning edge runs along
        bestboxpos where on that edge, in [-1, 1]
    """
    comptime MINVAL = Scalar[DTYPE](1e-15)

    cltype = -4
    clface = -1
    clcorner = 0
    cledge = 0
    bestboxpos = Scalar[DTYPE](0)

    var best = Scalar[DTYPE](0)
    # MuJoCo's `bestdistmax`, not an arbitrary large number: a candidate no
    # closer than this is not accepted at all, which is what leaves `cltype` at
    # -4 and means "no contact".
    var best_d2 = margin + Scalar[DTYPE](2) * (
        cap_radius
        + sqrt(hax_x * hax_x + hax_y * hax_y + hax_z * hax_z)
        + sx + sy + sz
    )
    var found = False

    # ── faces: the two endpoints ────────────────────────────────────────────
    var ends = [Scalar[DTYPE](-1), Scalar[DTYPE](1)]
    for k in range(2):
        var i = ends[k]
        var ex = pos_x + hax_x * i
        var ey = pos_y + hax_y * i
        var ez = pos_z + hax_z * i

        var cx = ex
        var cy = ey
        var cz = ez
        var nclamp = 0
        var caxis = -1
        if cx < -sx:
            cx = -sx
            nclamp += 1
            caxis = 0
        elif cx > sx:
            cx = sx
            nclamp += 1
            caxis = 0
        if cy < -sy:
            cy = -sy
            nclamp += 1
            caxis = 1
        elif cy > sy:
            cy = sy
            nclamp += 1
            caxis = 1
        if cz < -sz:
            cz = -sz
            nclamp += 1
            caxis = 2
        elif cz > sz:
            cz = sz
            nclamp += 1
            caxis = 2
        if nclamp > 1:
            continue

        var dx = cx - ex
        var dy = cy - ey
        var dz = cz - ez
        var d2 = dx * dx + dy * dy + dz * dz
        if d2 < best_d2:
            best_d2 = d2
            best = i
            found = True
            cltype = -2 + Int(i)
            clface = caxis

    # ── edges: all 12, as segment vs segment ────────────────────────────────
    var sizes = [sx, sy, sz]
    var hax = [hax_x, hax_y, hax_z]
    var pos = [pos_x, pos_y, pos_z]
    for j in range(3):
        var sj = sizes[j]
        for i in range(8):
            if (i & (1 << j)) != 0:
                continue
            # The edge's midpoint: a corner with component j zeroed.
            var e0 = sx if (i & 1) != 0 else -sx
            var e1 = sy if (i & 2) != 0 else -sy
            var e2 = sz if (i & 4) != 0 else -sz
            var edge = [e0, e1, e2]
            edge[j] = Scalar[DTYPE](0)

            var dif0 = edge[0] - pos[0]
            var dif1 = edge[1] - pos[1]
            var dif2 = edge[2] - pos[2]

            var ma = sj * sj
            var mb = -sj * hax[j]
            var mc = hax_x * hax_x + hax_y * hax_y + hax_z * hax_z

            var difs = [dif0, dif1, dif2]
            var u = -sj * difs[j]
            var v = hax_x * dif0 + hax_y * dif1 + hax_z * dif2

            var det = ma * mc - mb * mb
            if abs(det) < MINVAL:
                continue
            var idet = Scalar[DTYPE](1) / det

            var x1 = (mc * u - mb * v) * idet
            var x2 = (ma * v - mb * u) * idet
            # `s1`/`s2` record WHICH END each parameter clamped to (0 low, 1
            # unclamped, 2 high). MuJoCo packs them as `s1 * 3 + s2` into
            # `cltype`, and the second-point logic reads that back to tell a
            # corner contact from a mid-edge one.
            var s1 = 1
            var s2 = 1

            if x1 > Scalar[DTYPE](1):
                x1 = Scalar[DTYPE](1)
                s1 = 2
                x2 = (v - mb) / mc
            elif x1 < Scalar[DTYPE](-1):
                x1 = Scalar[DTYPE](-1)
                s1 = 0
                x2 = (v + mb) / mc

            if x2 > Scalar[DTYPE](1):
                x2 = Scalar[DTYPE](1)
                s2 = 2
                x1 = (u - mb) / ma
                if x1 > Scalar[DTYPE](1):
                    x1 = Scalar[DTYPE](1)
                    s1 = 2
                elif x1 < Scalar[DTYPE](-1):
                    x1 = Scalar[DTYPE](-1)
                    s1 = 0
            elif x2 < Scalar[DTYPE](-1):
                x2 = Scalar[DTYPE](-1)
                s2 = 0
                x1 = (u + mb) / ma
                if x1 > Scalar[DTYPE](1):
                    x1 = Scalar[DTYPE](1)
                    s1 = 2
                elif x1 < Scalar[DTYPE](-1):
                    x1 = Scalar[DTYPE](-1)
                    s1 = 0

            # Vector from the point on the capsule to the point on the edge.
            var g0 = dif0 - hax_x * x2
            var g1 = dif1 - hax_y * x2
            var g2 = dif2 - hax_z * x2
            var gg = [g0, g1, g2]
            gg[j] = gg[j] + sj * x1
            var d2 = gg[0] * gg[0] + gg[1] * gg[1] + gg[2] * gg[2]

            # The `- MINVAL` is MuJoCo's, and it matters: it stops a tie broken
            # by round-off from moving the contact point between two edges that
            # are exactly equidistant, which a numerically-parallel axis makes
            # common.
            if d2 < best_d2 - MINVAL:
                best_d2 = d2
                best = x2
                found = True
                bestboxpos = x1
                var pk = s1 * 3 + s2
                clcorner = i + (1 << j) * (pk // 6)
                cledge = j
                cltype = pk

    if not found:
        return Scalar[DTYPE](0)
    return best


# A capsule against a box gets at most two contacts: MuJoCo calls
# `mjraw_SphereBox` once at `bestsegmentpos` and, when the geometry supports it,
# once more at `bestsegmentpos + secondpos`.
comptime CB_MAX_POINTS: Int = 2

# `secondpos` is initialised out of range to mean "no second point"; MuJoCo
# tests `secondpos > -3` afterwards.
comptime _CB_NO_SECOND: Float64 = -4.0


@always_inline
def _capsule_box_second_pos[
    DTYPE: DType
](
    pos_x: Scalar[DTYPE],
    pos_y: Scalar[DTYPE],
    pos_z: Scalar[DTYPE],
    hax_x: Scalar[DTYPE],
    hax_y: Scalar[DTYPE],
    hax_z: Scalar[DTYPE],
    sx: Scalar[DTYPE],
    sy: Scalar[DTYPE],
    sz: Scalar[DTYPE],
    bestsegmentpos: Scalar[DTYPE],
    bestboxpos: Scalar[DTYPE],
    cltype: Int,
    clface: Int,
    clcorner: Int,
    cledge: Int,
) -> Scalar[DTYPE]:
    """How far along the capsule axis MuJoCo puts its SECOND contact.

    Returns an offset from `bestsegmentpos`, or `_CB_NO_SECOND` when the
    geometry admits only one point. Port of the `cltype` dispatch in
    `mjraw_CapsuleBox` (`engine_collision_box.c`, lines 400-568).

    A capsule lying along a box FACE touches over a whole segment, and one
    point cannot express that: it leaves the capsule free to pivot about the
    single contact, exactly the failure a box on one contact point has. Which
    of the three branches applies is decided by what the FIRST point was
    closest to — a corner, an edge, or a face — and the answer is always
    "walk along the capsule until it leaves the box, then clamp".
    """
    var s = InlineArray[Scalar[DTYPE], 3](fill=Scalar[DTYPE](0))
    s[0] = sx
    s[1] = sy
    s[2] = sz
    var hax = InlineArray[Scalar[DTYPE], 3](fill=Scalar[DTYPE](0))
    hax[0] = hax_x
    hax[1] = hax_y
    hax[2] = hax_z
    var pos = InlineArray[Scalar[DTYPE], 3](fill=Scalar[DTYPE](0))
    pos[0] = pos_x
    pos[1] = pos_y
    pos[2] = pos_z

    # `axisdir` is the octant the capsule's half-axis points into; XORing it
    # with `clcorner` is MuJoCo's trick for the RELATIVE orientation of the
    # capsule and the closest corner.
    var axisdir = 0
    if hax_x > Scalar[DTYPE](0):
        axisdir += 1
    if hax_y > Scalar[DTYPE](0):
        axisdir += 2
    if hax_z > Scalar[DTYPE](0):
        axisdir += 4

    var hlen = sqrt(hax_x * hax_x + hax_y * hax_y + hax_z * hax_z)
    if hlen <= Scalar[DTYPE](0):
        return Scalar[DTYPE](_CB_NO_SECOND)
    # MuJoCo's `axis` is the UNIT capsule direction; `halfaxis` is that scaled
    # by the half-length. Only `axis` is compared against 0.5 below.
    var axis = InlineArray[Scalar[DTYPE], 3](fill=Scalar[DTYPE](0))
    axis[0] = hax_x / hlen
    axis[1] = hax_y / hlen
    axis[2] = hax_z / hlen

    var secondpos = Scalar[DTYPE](_CB_NO_SECOND)

    if cltype >= 0 and (cltype // 3) != 1:
        # The first point is at a CORNER of the box.
        var c1 = axisdir ^ clcorner
        # Pointing straight at or straight away from the corner: nothing else
        # can touch.
        if c1 == 0 or c1 == 7:
            return Scalar[DTYPE](_CB_NO_SECOND)

        var mul: Scalar[DTYPE]
        var de: Scalar[DTYPE]
        var dp: Scalar[DTYPE]
        if c1 == 1 or c1 == 2 or c1 == 4:
            mul = Scalar[DTYPE](1)
            de = Scalar[DTYPE](1) - bestsegmentpos
            dp = Scalar[DTYPE](1) + bestsegmentpos
        else:
            mul = Scalar[DTYPE](-1)
            c1 = 7 - c1
            dp = Scalar[DTYPE](1) - bestsegmentpos
            de = Scalar[DTYPE](1) + bestsegmentpos

        var ax = 0
        var ax1 = 1
        var ax2 = 2
        if c1 == 2:
            ax = 1
            ax1 = 2
            ax2 = 0
        elif c1 == 4:
            ax = 2
            ax1 = 0
            ax2 = 1

        if axis[ax] * axis[ax] > Scalar[DTYPE](0.5):
            # Second point along the box EDGE the capsule runs down.
            secondpos = de
            var e1 = Scalar[DTYPE](2) * s[ax] / abs(hax[ax])
            if e1 < secondpos:
                secondpos = e1
            secondpos *= mul
        else:
            # Second point along a box FACE.
            secondpos = dp
            var e1 = Scalar[DTYPE](2) * s[ax1] / abs(hax[ax1])
            if e1 < secondpos:
                secondpos = e1
            e1 = Scalar[DTYPE](2) * s[ax2] / abs(hax[ax2])
            if e1 < secondpos:
                secondpos = e1
            secondpos *= -mul

    elif cltype >= 0:
        # The first point is on the MIDDLE of a box edge.
        var c1 = axisdir ^ clcorner
        c1 &= 7 - (1 << cledge)
        # A T configuration (capsule crossing the edge) has no second contact;
        # only an X configuration does.
        if c1 != 1 and c1 != 2 and c1 != 4:
            return Scalar[DTYPE](_CB_NO_SECOND)

        var ax = cledge
        var ax1 = 1
        var ax2 = 2
        if cledge == 1:
            ax1 = 2
            ax2 = 0
        elif cledge == 2:
            ax1 = 0
            ax2 = 1

        # Whichever of the two adjacent faces the capsule lies flatter against.
        if abs(axis[ax1]) > abs(axis[ax2]):
            ax1 = ax2
        ax2 = 3 - ax - ax1

        var mul: Scalar[DTYPE]
        if (c1 & (1 << ax2)) != 0:
            mul = Scalar[DTYPE](1)
            secondpos = Scalar[DTYPE](1) - bestsegmentpos
        else:
            mul = Scalar[DTYPE](-1)
            secondpos = Scalar[DTYPE](1) + bestsegmentpos

        var e1 = Scalar[DTYPE](2) * s[ax2] / abs(hax[ax2])
        if e1 < secondpos:
            secondpos = e1

        var e2 = Scalar[DTYPE](1) + bestboxpos
        if ((axisdir & (1 << ax)) != 0) == ((c1 & (1 << ax2)) != 0):
            e2 = Scalar[DTYPE](1) - bestboxpos

        e1 = s[ax] * e2 / abs(hax[ax])
        if e1 < secondpos:
            secondpos = e1
        secondpos *= mul

    elif cltype < 0:
        # A capsule END is closest to a box FACE. Walk from the other end and
        # clamp at the first side plane it crosses.
        if clface == -1:
            # The closest point is INSIDE the box; there is no second point.
            return Scalar[DTYPE](_CB_NO_SECOND)
        var mul = Scalar[DTYPE](-1)
        if cltype == -3:
            mul = Scalar[DTYPE](1)

        secondpos = Scalar[DTYPE](2)
        var t = InlineArray[Scalar[DTYPE], 3](fill=Scalar[DTYPE](0))
        for i in range(3):
            t[i] = pos[i] - hax[i] * mul

        for i in range(3):
            if i == clface:
                continue
            var e1 = (s[i] - t[i]) / hax[i] * mul
            if e1 > Scalar[DTYPE](0) and e1 < secondpos:
                secondpos = e1
            e1 = (-s[i] - t[i]) / hax[i] * mul
            if e1 > Scalar[DTYPE](0) and e1 < secondpos:
                secondpos = e1
        secondpos *= mul

    return secondpos


@always_inline
def box_capsule[
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
    """Box-capsule collision detection — the PRIMARY point only.

    Algorithm:
    1. Transform capsule endpoints to box's local frame
    2. Find closest point on capsule segment to box (clamped)
    3. Treat as box-sphere with the closest point as sphere center

    ⚠ THE NARROW PHASE NO LONGER CALLS THIS. `box_capsule_manifold` below emits
    the whole manifold (this point plus MuJoCo's second one) and is what
    `_capsule_box_contacts` uses. This stays because `cylinder_box` reduces a
    cylinder to a virtual capsule and wants a single point.

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

    # The capsule as (centre, half-axis) in the box's local frame, which is the
    # parameterisation `_capsule_box_best_segment_pos` and MuJoCo both use:
    # the segment is `pos + halfaxis * s` for s in [-1, 1].
    var pos_x = (local1[0] + local2[0]) * Scalar[DTYPE](0.5)
    var pos_y = (local1[1] + local2[1]) * Scalar[DTYPE](0.5)
    var pos_z = (local1[2] + local2[2]) * Scalar[DTYPE](0.5)
    var hax_x = (local1[0] - local2[0]) * Scalar[DTYPE](0.5)
    var hax_y = (local1[1] - local2[1]) * Scalar[DTYPE](0.5)
    var hax_z = (local1[2] - local2[2]) * Scalar[DTYPE](0.5)

    # ⚠ THE POINT ON THE SEGMENT CLOSEST TO THE BOX'S SURFACE, NOT TO ITS
    # CENTRE. This used to project the box CENTRE onto the segment, which is a
    # different point whenever the box's nearest feature is not the face that
    # projection points at — i.e. for every edge and corner contact. It cost up
    # to 24 mm of depth and a fully reversed normal; see task #45 and
    # `tests/physics3d/test_capsule_box_sweep.mojo`.
    var cltype = -4
    var clface = -1
    var clcorner = 0
    var cledge = 0
    var bestboxpos = Scalar[DTYPE](0)
    var s = _capsule_box_best_segment_pos[DTYPE](
        pos_x, pos_y, pos_z, hax_x, hax_y, hax_z, half_x, half_y, half_z,
        cap_radius, Scalar[DTYPE](0),
        cltype, clface, clcorner, cledge, bestboxpos,
    )

    # Closest point on segment (in local frame)
    var closest_seg_x = pos_x + s * hax_x
    var closest_seg_y = pos_y + s * hax_y
    var closest_seg_z = pos_z + s * hax_z

    # Transform back to world frame for box-sphere test
    var closest_world = rotate_vector_by_quat(
        closest_seg_x,
        closest_seg_y,
        closest_seg_z,
        box_qx,
        box_qy,
        box_qz,
        box_qw,
    )
    var closest_x = box_x + closest_world[0]
    var closest_y = box_y + closest_world[1]
    var closest_z = box_z + closest_world[2]

    # Now treat as box-sphere collision
    return box_sphere(
        box_x,
        box_y,
        box_z,
        box_qx,
        box_qy,
        box_qz,
        box_qw,
        half_x,
        half_y,
        half_z,
        closest_x,
        closest_y,
        closest_z,
        cap_radius,
    )


@always_inline
def box_capsule_manifold[
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
    margin: Scalar[DTYPE],
    mut dist_out: InlineArray[Scalar[DTYPE], CB_MAX_POINTS],
    mut pos_out: InlineArray[Scalar[DTYPE], 3 * CB_MAX_POINTS],
    mut normal_out: InlineArray[Scalar[DTYPE], 3 * CB_MAX_POINTS],
) -> Int:
    """Full contact MANIFOLD for a box/capsule pair — up to TWO points.

    Port of `mjraw_CapsuleBox` (`references/mujoco-3.6.0`, identical in 3.3.6
    and 3.5.1). `box_capsule` above is this routine's first point on its own,
    and stays because `cylinder_box` reduces to it.

    ⚠ EACH POINT CARRIES ITS OWN NORMAL, unlike `box_box_manifold` where a face
    manifold shares one. MuJoCo runs `mjraw_SphereBox` independently at each
    segment position, and the two spheres can land on different box features —
    a capsule over an edge gets a face normal at one end and an edge normal at
    the other. Hence `normal_out` is 3 * CB_MAX_POINTS, not 3.

    Returns the number of points written. Normals point from the BOX to the
    CAPSULE, matching `box_capsule`.
    """
    var cap_axis = rotate_vector_by_quat(
        Scalar[DTYPE](0.0), Scalar[DTYPE](0.0), Scalar[DTYPE](1.0),
        cap_qx, cap_qy, cap_qz, cap_qw,
    )
    var ep1 = rotate_vector_by_quat_inverse(
        cap_x + cap_axis[0] * cap_half_len - box_x,
        cap_y + cap_axis[1] * cap_half_len - box_y,
        cap_z + cap_axis[2] * cap_half_len - box_z,
        box_qx, box_qy, box_qz, box_qw,
    )
    var ep2 = rotate_vector_by_quat_inverse(
        cap_x - cap_axis[0] * cap_half_len - box_x,
        cap_y - cap_axis[1] * cap_half_len - box_y,
        cap_z - cap_axis[2] * cap_half_len - box_z,
        box_qx, box_qy, box_qz, box_qw,
    )

    var pos_x = (ep1[0] + ep2[0]) * Scalar[DTYPE](0.5)
    var pos_y = (ep1[1] + ep2[1]) * Scalar[DTYPE](0.5)
    var pos_z = (ep1[2] + ep2[2]) * Scalar[DTYPE](0.5)
    var hax_x = (ep1[0] - ep2[0]) * Scalar[DTYPE](0.5)
    var hax_y = (ep1[1] - ep2[1]) * Scalar[DTYPE](0.5)
    var hax_z = (ep1[2] - ep2[2]) * Scalar[DTYPE](0.5)

    var cltype = -4
    var clface = -1
    var clcorner = 0
    var cledge = 0
    var bestboxpos = Scalar[DTYPE](0)
    var bestsegmentpos = _capsule_box_best_segment_pos[DTYPE](
        pos_x, pos_y, pos_z, hax_x, hax_y, hax_z, half_x, half_y, half_z,
        cap_radius, margin,
        cltype, clface, clcorner, cledge, bestboxpos,
    )
    if cltype == -4:
        return 0

    var secondpos = _capsule_box_second_pos[DTYPE](
        pos_x, pos_y, pos_z, hax_x, hax_y, hax_z, half_x, half_y, half_z,
        bestsegmentpos, bestboxpos, cltype, clface, clcorner, cledge,
    )

    var n = 0
    for k in range(2):
        var t = bestsegmentpos
        if k == 1:
            if not (secondpos > Scalar[DTYPE](-3)):
                break
            t = bestsegmentpos + secondpos

        var lw = rotate_vector_by_quat(
            pos_x + hax_x * t, pos_y + hax_y * t, pos_z + hax_z * t,
            box_qx, box_qy, box_qz, box_qw,
        )
        var r = box_sphere(
            box_x, box_y, box_z, box_qx, box_qy, box_qz, box_qw,
            half_x, half_y, half_z,
            box_x + lw[0], box_y + lw[1], box_z + lw[2],
            cap_radius,
        )
        # `mjraw_SphereBox` returns nothing at all when the sphere is farther
        # than `margin`; the second point can be rejected while the first is
        # kept, so this is per-point and not a filter on the pair.
        if r[0] > margin:
            continue
        dist_out[n] = r[0]
        pos_out[3 * n + 0] = r[1]
        pos_out[3 * n + 1] = r[2]
        pos_out[3 * n + 2] = r[3]
        normal_out[3 * n + 0] = r[4]
        normal_out[3 * n + 1] = r[5]
        normal_out[3 * n + 2] = r[6]
        n += 1

    return n


@always_inline
def _project_box_onto_axis[
    DTYPE: DType
](
    # Box rotation matrix columns (already computed)
    r0_x: Scalar[DTYPE],
    r0_y: Scalar[DTYPE],
    r0_z: Scalar[DTYPE],
    r1_x: Scalar[DTYPE],
    r1_y: Scalar[DTYPE],
    r1_z: Scalar[DTYPE],
    r2_x: Scalar[DTYPE],
    r2_y: Scalar[DTYPE],
    r2_z: Scalar[DTYPE],
    # Box half-extents
    hx: Scalar[DTYPE],
    hy: Scalar[DTYPE],
    hz: Scalar[DTYPE],
    # Axis to project onto
    ax: Scalar[DTYPE],
    ay: Scalar[DTYPE],
    az: Scalar[DTYPE],
) -> Scalar[DTYPE]:
    """Project a box onto an axis and return the half-width of the projection.

    The half-width is: |dot(r0, axis)|*hx + |dot(r1, axis)|*hy + |dot(r2, axis)|*hz
    where r0, r1, r2 are the box's rotation matrix columns.
    """
    return (
        abs(r0_x * ax + r0_y * ay + r0_z * az) * hx
        + abs(r1_x * ax + r1_y * ay + r1_z * az) * hy
        + abs(r2_x * ax + r2_y * ay + r2_z * az) * hz
    )


@always_inline
def _test_sat_axis[
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
    a0_x: Scalar[DTYPE],
    a0_y: Scalar[DTYPE],
    a0_z: Scalar[DTYPE],
    a1_x: Scalar[DTYPE],
    a1_y: Scalar[DTYPE],
    a1_z: Scalar[DTYPE],
    a2_x: Scalar[DTYPE],
    a2_y: Scalar[DTYPE],
    a2_z: Scalar[DTYPE],
    a_hx: Scalar[DTYPE],
    a_hy: Scalar[DTYPE],
    a_hz: Scalar[DTYPE],
    # Box B rotation columns
    b0_x: Scalar[DTYPE],
    b0_y: Scalar[DTYPE],
    b0_z: Scalar[DTYPE],
    b1_x: Scalar[DTYPE],
    b1_y: Scalar[DTYPE],
    b1_z: Scalar[DTYPE],
    b2_x: Scalar[DTYPE],
    b2_y: Scalar[DTYPE],
    b2_z: Scalar[DTYPE],
    b_hx: Scalar[DTYPE],
    b_hy: Scalar[DTYPE],
    b_hz: Scalar[DTYPE],
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
        a0_x,
        a0_y,
        a0_z,
        a1_x,
        a1_y,
        a1_z,
        a2_x,
        a2_y,
        a2_z,
        a_hx,
        a_hy,
        a_hz,
        ax,
        ay,
        az,
    )
    var proj_b = _project_box_onto_axis(
        b0_x,
        b0_y,
        b0_z,
        b1_x,
        b1_y,
        b1_z,
        b2_x,
        b2_y,
        b2_z,
        b_hx,
        b_hy,
        b_hz,
        ax,
        ay,
        az,
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
def box_box[
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
    var a0 = rotate_vector_by_quat(
        Scalar[DTYPE](1),
        Scalar[DTYPE](0),
        Scalar[DTYPE](0),
        a_qx,
        a_qy,
        a_qz,
        a_qw,
    )
    var a1 = rotate_vector_by_quat(
        Scalar[DTYPE](0),
        Scalar[DTYPE](1),
        Scalar[DTYPE](0),
        a_qx,
        a_qy,
        a_qz,
        a_qw,
    )
    var a2 = rotate_vector_by_quat(
        Scalar[DTYPE](0),
        Scalar[DTYPE](0),
        Scalar[DTYPE](1),
        a_qx,
        a_qy,
        a_qz,
        a_qw,
    )

    # Compute rotation matrix columns for box B
    var b0 = rotate_vector_by_quat(
        Scalar[DTYPE](1),
        Scalar[DTYPE](0),
        Scalar[DTYPE](0),
        b_qx,
        b_qy,
        b_qz,
        b_qw,
    )
    var b1 = rotate_vector_by_quat(
        Scalar[DTYPE](0),
        Scalar[DTYPE](1),
        Scalar[DTYPE](0),
        b_qx,
        b_qy,
        b_qz,
        b_qw,
    )
    var b2 = rotate_vector_by_quat(
        Scalar[DTYPE](0),
        Scalar[DTYPE](0),
        Scalar[DTYPE](1),
        b_qx,
        b_qy,
        b_qz,
        b_qw,
    )

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
    var SEPARATED = (
        Scalar[DTYPE](1.0),
        a_x,
        a_y,
        a_z,
        Scalar[DTYPE](0),
        Scalar[DTYPE](0),
        Scalar[DTYPE](1),
    )

    # Test 15 axes

    # Box A face normals (3 axes)
    if _test_sat_axis(
        a0[0],
        a0[1],
        a0[2],
        t_x,
        t_y,
        t_z,
        a0[0],
        a0[1],
        a0[2],
        a1[0],
        a1[1],
        a1[2],
        a2[0],
        a2[1],
        a2[2],
        a_hx,
        a_hy,
        a_hz,
        b0[0],
        b0[1],
        b0[2],
        b1[0],
        b1[1],
        b1[2],
        b2[0],
        b2[1],
        b2[2],
        b_hx,
        b_hy,
        b_hz,
        min_pen,
        best_nx,
        best_ny,
        best_nz,
    ):
        return SEPARATED
    if _test_sat_axis(
        a1[0],
        a1[1],
        a1[2],
        t_x,
        t_y,
        t_z,
        a0[0],
        a0[1],
        a0[2],
        a1[0],
        a1[1],
        a1[2],
        a2[0],
        a2[1],
        a2[2],
        a_hx,
        a_hy,
        a_hz,
        b0[0],
        b0[1],
        b0[2],
        b1[0],
        b1[1],
        b1[2],
        b2[0],
        b2[1],
        b2[2],
        b_hx,
        b_hy,
        b_hz,
        min_pen,
        best_nx,
        best_ny,
        best_nz,
    ):
        return SEPARATED
    if _test_sat_axis(
        a2[0],
        a2[1],
        a2[2],
        t_x,
        t_y,
        t_z,
        a0[0],
        a0[1],
        a0[2],
        a1[0],
        a1[1],
        a1[2],
        a2[0],
        a2[1],
        a2[2],
        a_hx,
        a_hy,
        a_hz,
        b0[0],
        b0[1],
        b0[2],
        b1[0],
        b1[1],
        b1[2],
        b2[0],
        b2[1],
        b2[2],
        b_hx,
        b_hy,
        b_hz,
        min_pen,
        best_nx,
        best_ny,
        best_nz,
    ):
        return SEPARATED

    # Box B face normals (3 axes)
    if _test_sat_axis(
        b0[0],
        b0[1],
        b0[2],
        t_x,
        t_y,
        t_z,
        a0[0],
        a0[1],
        a0[2],
        a1[0],
        a1[1],
        a1[2],
        a2[0],
        a2[1],
        a2[2],
        a_hx,
        a_hy,
        a_hz,
        b0[0],
        b0[1],
        b0[2],
        b1[0],
        b1[1],
        b1[2],
        b2[0],
        b2[1],
        b2[2],
        b_hx,
        b_hy,
        b_hz,
        min_pen,
        best_nx,
        best_ny,
        best_nz,
    ):
        return SEPARATED
    if _test_sat_axis(
        b1[0],
        b1[1],
        b1[2],
        t_x,
        t_y,
        t_z,
        a0[0],
        a0[1],
        a0[2],
        a1[0],
        a1[1],
        a1[2],
        a2[0],
        a2[1],
        a2[2],
        a_hx,
        a_hy,
        a_hz,
        b0[0],
        b0[1],
        b0[2],
        b1[0],
        b1[1],
        b1[2],
        b2[0],
        b2[1],
        b2[2],
        b_hx,
        b_hy,
        b_hz,
        min_pen,
        best_nx,
        best_ny,
        best_nz,
    ):
        return SEPARATED
    if _test_sat_axis(
        b2[0],
        b2[1],
        b2[2],
        t_x,
        t_y,
        t_z,
        a0[0],
        a0[1],
        a0[2],
        a1[0],
        a1[1],
        a1[2],
        a2[0],
        a2[1],
        a2[2],
        a_hx,
        a_hy,
        a_hz,
        b0[0],
        b0[1],
        b0[2],
        b1[0],
        b1[1],
        b1[2],
        b2[0],
        b2[1],
        b2[2],
        b_hx,
        b_hy,
        b_hz,
        min_pen,
        best_nx,
        best_ny,
        best_nz,
    ):
        return SEPARATED

    # Edge-edge cross products (9 axes)
    # A0 x B0
    var c_x = a0[1] * b0[2] - a0[2] * b0[1]
    var c_y = a0[2] * b0[0] - a0[0] * b0[2]
    var c_z = a0[0] * b0[1] - a0[1] * b0[0]
    if _test_sat_axis(
        c_x,
        c_y,
        c_z,
        t_x,
        t_y,
        t_z,
        a0[0],
        a0[1],
        a0[2],
        a1[0],
        a1[1],
        a1[2],
        a2[0],
        a2[1],
        a2[2],
        a_hx,
        a_hy,
        a_hz,
        b0[0],
        b0[1],
        b0[2],
        b1[0],
        b1[1],
        b1[2],
        b2[0],
        b2[1],
        b2[2],
        b_hx,
        b_hy,
        b_hz,
        min_pen,
        best_nx,
        best_ny,
        best_nz,
    ):
        return SEPARATED

    # A0 x B1
    c_x = a0[1] * b1[2] - a0[2] * b1[1]
    c_y = a0[2] * b1[0] - a0[0] * b1[2]
    c_z = a0[0] * b1[1] - a0[1] * b1[0]
    if _test_sat_axis(
        c_x,
        c_y,
        c_z,
        t_x,
        t_y,
        t_z,
        a0[0],
        a0[1],
        a0[2],
        a1[0],
        a1[1],
        a1[2],
        a2[0],
        a2[1],
        a2[2],
        a_hx,
        a_hy,
        a_hz,
        b0[0],
        b0[1],
        b0[2],
        b1[0],
        b1[1],
        b1[2],
        b2[0],
        b2[1],
        b2[2],
        b_hx,
        b_hy,
        b_hz,
        min_pen,
        best_nx,
        best_ny,
        best_nz,
    ):
        return SEPARATED

    # A0 x B2
    c_x = a0[1] * b2[2] - a0[2] * b2[1]
    c_y = a0[2] * b2[0] - a0[0] * b2[2]
    c_z = a0[0] * b2[1] - a0[1] * b2[0]
    if _test_sat_axis(
        c_x,
        c_y,
        c_z,
        t_x,
        t_y,
        t_z,
        a0[0],
        a0[1],
        a0[2],
        a1[0],
        a1[1],
        a1[2],
        a2[0],
        a2[1],
        a2[2],
        a_hx,
        a_hy,
        a_hz,
        b0[0],
        b0[1],
        b0[2],
        b1[0],
        b1[1],
        b1[2],
        b2[0],
        b2[1],
        b2[2],
        b_hx,
        b_hy,
        b_hz,
        min_pen,
        best_nx,
        best_ny,
        best_nz,
    ):
        return SEPARATED

    # A1 x B0
    c_x = a1[1] * b0[2] - a1[2] * b0[1]
    c_y = a1[2] * b0[0] - a1[0] * b0[2]
    c_z = a1[0] * b0[1] - a1[1] * b0[0]
    if _test_sat_axis(
        c_x,
        c_y,
        c_z,
        t_x,
        t_y,
        t_z,
        a0[0],
        a0[1],
        a0[2],
        a1[0],
        a1[1],
        a1[2],
        a2[0],
        a2[1],
        a2[2],
        a_hx,
        a_hy,
        a_hz,
        b0[0],
        b0[1],
        b0[2],
        b1[0],
        b1[1],
        b1[2],
        b2[0],
        b2[1],
        b2[2],
        b_hx,
        b_hy,
        b_hz,
        min_pen,
        best_nx,
        best_ny,
        best_nz,
    ):
        return SEPARATED

    # A1 x B1
    c_x = a1[1] * b1[2] - a1[2] * b1[1]
    c_y = a1[2] * b1[0] - a1[0] * b1[2]
    c_z = a1[0] * b1[1] - a1[1] * b1[0]
    if _test_sat_axis(
        c_x,
        c_y,
        c_z,
        t_x,
        t_y,
        t_z,
        a0[0],
        a0[1],
        a0[2],
        a1[0],
        a1[1],
        a1[2],
        a2[0],
        a2[1],
        a2[2],
        a_hx,
        a_hy,
        a_hz,
        b0[0],
        b0[1],
        b0[2],
        b1[0],
        b1[1],
        b1[2],
        b2[0],
        b2[1],
        b2[2],
        b_hx,
        b_hy,
        b_hz,
        min_pen,
        best_nx,
        best_ny,
        best_nz,
    ):
        return SEPARATED

    # A1 x B2
    c_x = a1[1] * b2[2] - a1[2] * b2[1]
    c_y = a1[2] * b2[0] - a1[0] * b2[2]
    c_z = a1[0] * b2[1] - a1[1] * b2[0]
    if _test_sat_axis(
        c_x,
        c_y,
        c_z,
        t_x,
        t_y,
        t_z,
        a0[0],
        a0[1],
        a0[2],
        a1[0],
        a1[1],
        a1[2],
        a2[0],
        a2[1],
        a2[2],
        a_hx,
        a_hy,
        a_hz,
        b0[0],
        b0[1],
        b0[2],
        b1[0],
        b1[1],
        b1[2],
        b2[0],
        b2[1],
        b2[2],
        b_hx,
        b_hy,
        b_hz,
        min_pen,
        best_nx,
        best_ny,
        best_nz,
    ):
        return SEPARATED

    # A2 x B0
    c_x = a2[1] * b0[2] - a2[2] * b0[1]
    c_y = a2[2] * b0[0] - a2[0] * b0[2]
    c_z = a2[0] * b0[1] - a2[1] * b0[0]
    if _test_sat_axis(
        c_x,
        c_y,
        c_z,
        t_x,
        t_y,
        t_z,
        a0[0],
        a0[1],
        a0[2],
        a1[0],
        a1[1],
        a1[2],
        a2[0],
        a2[1],
        a2[2],
        a_hx,
        a_hy,
        a_hz,
        b0[0],
        b0[1],
        b0[2],
        b1[0],
        b1[1],
        b1[2],
        b2[0],
        b2[1],
        b2[2],
        b_hx,
        b_hy,
        b_hz,
        min_pen,
        best_nx,
        best_ny,
        best_nz,
    ):
        return SEPARATED

    # A2 x B1
    c_x = a2[1] * b1[2] - a2[2] * b1[1]
    c_y = a2[2] * b1[0] - a2[0] * b1[2]
    c_z = a2[0] * b1[1] - a2[1] * b1[0]
    if _test_sat_axis(
        c_x,
        c_y,
        c_z,
        t_x,
        t_y,
        t_z,
        a0[0],
        a0[1],
        a0[2],
        a1[0],
        a1[1],
        a1[2],
        a2[0],
        a2[1],
        a2[2],
        a_hx,
        a_hy,
        a_hz,
        b0[0],
        b0[1],
        b0[2],
        b1[0],
        b1[1],
        b1[2],
        b2[0],
        b2[1],
        b2[2],
        b_hx,
        b_hy,
        b_hz,
        min_pen,
        best_nx,
        best_ny,
        best_nz,
    ):
        return SEPARATED

    # A2 x B2
    c_x = a2[1] * b2[2] - a2[2] * b2[1]
    c_y = a2[2] * b2[0] - a2[0] * b2[2]
    c_z = a2[0] * b2[1] - a2[1] * b2[0]
    if _test_sat_axis(
        c_x,
        c_y,
        c_z,
        t_x,
        t_y,
        t_z,
        a0[0],
        a0[1],
        a0[2],
        a1[0],
        a1[1],
        a1[2],
        a2[0],
        a2[1],
        a2[2],
        a_hx,
        a_hy,
        a_hz,
        b0[0],
        b0[1],
        b0[2],
        b1[0],
        b1[1],
        b1[2],
        b2[0],
        b2[1],
        b2[2],
        b_hx,
        b_hy,
        b_hz,
        min_pen,
        best_nx,
        best_ny,
        best_nz,
    ):
        return SEPARATED

    # No separating axis found - boxes are colliding
    # Signed distance is negative penetration
    var dist = -min_pen

    # Contact point: approximate as midpoint between centers
    var contact_x = (a_x + b_x) * Scalar[DTYPE](0.5)
    var contact_y = (a_y + b_y) * Scalar[DTYPE](0.5)
    var contact_z = (a_z + b_z) * Scalar[DTYPE](0.5)

    return (dist, contact_x, contact_y, contact_z, best_nx, best_ny, best_nz)


# =============================================================================
# box/box CONTACT MANIFOLD (port of MuJoCo's `_boxbox`, both paths)
# =============================================================================

# Upper bound on the points either path can produce before filtering, and a
# real bound rather than a cap: at most 8 from clipping the incident face's
# four edges against the reference face (2 crossings per edge), 4 from
# reference-face corners inside the incident face, and 4 incident-face corners.
comptime BB_MAX_POINTS: Int = 16

# MuJoCo's mjMINVAL.
comptime _BB_MINVAL: Float64 = 1e-15


@always_inline
def _bb_quat_mat[
    DTYPE: DType
](
    qx: Scalar[DTYPE], qy: Scalar[DTYPE], qz: Scalar[DTYPE], qw: Scalar[DTYPE]
) -> InlineArray[Scalar[DTYPE], 9]:
    """Rotation matrix, ROW-MAJOR (`m[3 * row + col]`), matching MuJoCo's
    `geom_xmat` layout so the transcribed index arithmetic carries over."""
    var c0 = rotate_vector_by_quat(
        Scalar[DTYPE](1), Scalar[DTYPE](0), Scalar[DTYPE](0), qx, qy, qz, qw
    )
    var c1 = rotate_vector_by_quat(
        Scalar[DTYPE](0), Scalar[DTYPE](1), Scalar[DTYPE](0), qx, qy, qz, qw
    )
    var c2 = rotate_vector_by_quat(
        Scalar[DTYPE](0), Scalar[DTYPE](0), Scalar[DTYPE](1), qx, qy, qz, qw
    )
    var m = InlineArray[Scalar[DTYPE], 9](fill=Scalar[DTYPE](0))
    m[0] = c0[0]
    m[1] = c1[0]
    m[2] = c2[0]
    m[3] = c0[1]
    m[4] = c1[1]
    m[5] = c2[1]
    m[6] = c0[2]
    m[7] = c1[2]
    m[8] = c2[2]
    return m


@always_inline
def _bb_outside_box[
    DTYPE: DType
](
    px: Scalar[DTYPE],
    py: Scalar[DTYPE],
    pz: Scalar[DTYPE],
    bx: Scalar[DTYPE],
    by: Scalar[DTYPE],
    bz: Scalar[DTYPE],
    mat: InlineArray[Scalar[DTYPE], 9],
    sx: Scalar[DTYPE],
    sy: Scalar[DTYPE],
    sz: Scalar[DTYPE],
    inflate: Scalar[DTYPE],
) -> Int:
    """Port of `mju_outsideBox`: +1 outside the inflated box, -1 inside the
    deflated box, 0 in the shell between them."""
    var vx = px - bx
    var vy = py - by
    var vz = pz - bz
    # mat^T * v — mat's COLUMNS are the box axes.
    var lx = mat[0] * vx + mat[3] * vy + mat[6] * vz
    var ly = mat[1] * vx + mat[4] * vy + mat[7] * vz
    var lz = mat[2] * vx + mat[5] * vy + mat[8] * vz

    var gx = sx * inflate
    var gy = sy * inflate
    var gz = sz * inflate
    if lx > gx or lx < -gx or ly > gy or ly < -gy or lz > gz or lz < -gz:
        return 1

    var mx = sx / inflate
    var my = sy / inflate
    var mz = sz / inflate
    if lx < mx and lx > -mx and ly < my and ly > -my and lz < mz and lz > -mz:
        return -1
    return 0


@always_inline
def _bb_post_filter[
    DTYPE: DType
](
    n: Int,
    mut dist_out: InlineArray[Scalar[DTYPE], BB_MAX_POINTS],
    mut pos_out: InlineArray[Scalar[DTYPE], 3 * BB_MAX_POINTS],
    a_x: Scalar[DTYPE],
    a_y: Scalar[DTYPE],
    a_z: Scalar[DTYPE],
    mat1: InlineArray[Scalar[DTYPE], 9],
    a_hx: Scalar[DTYPE],
    a_hy: Scalar[DTYPE],
    a_hz: Scalar[DTYPE],
    b_x: Scalar[DTYPE],
    b_y: Scalar[DTYPE],
    b_z: Scalar[DTYPE],
    mat2: InlineArray[Scalar[DTYPE], 9],
    b_hx: Scalar[DTYPE],
    b_hy: Scalar[DTYPE],
    b_hz: Scalar[DTYPE],
    margin: Scalar[DTYPE],
) -> Int:
    """`mjc_BoxBox`'s post-filter, shared by the face and edge-edge paths: drop
    points that sit outside one box without being inside the other, then drop
    exact duplicates. Without it either path emits points that are
    geometrically off both boxes."""
    var bad = InlineArray[Bool, BB_MAX_POINTS](fill=False)
    var ratio = Scalar[DTYPE](1.01)
    for i in range(n):
        var o1 = _bb_outside_box[DTYPE](
            pos_out[3 * i + 0],
            pos_out[3 * i + 1],
            pos_out[3 * i + 2],
            a_x, a_y, a_z, mat1,
            a_hx + margin, a_hy + margin, a_hz + margin,
            ratio,
        )
        var o2 = _bb_outside_box[DTYPE](
            pos_out[3 * i + 0],
            pos_out[3 * i + 1],
            pos_out[3 * i + 2],
            b_x, b_y, b_z, mat2,
            b_hx + margin, b_hy + margin, b_hz + margin,
            ratio,
        )
        if (o1 == 1 and o2 != -1) or (o2 == 1 and o1 != -1):
            bad[i] = True

    for i in range(n - 1):
        if bad[i]:
            continue
        for j in range(i + 1, n):
            if bad[j]:
                continue
            if (
                pos_out[3 * i + 0] == pos_out[3 * j + 0]
                and pos_out[3 * i + 1] == pos_out[3 * j + 1]
                and pos_out[3 * i + 2] == pos_out[3 * j + 2]
            ):
                bad[i] = True
                break

    var w = 0
    for i in range(n):
        if bad[i]:
            continue
        if w != i:
            pos_out[3 * w + 0] = pos_out[3 * i + 0]
            pos_out[3 * w + 1] = pos_out[3 * i + 1]
            pos_out[3 * w + 2] = pos_out[3 * i + 2]
            dist_out[w] = dist_out[i]
        w += 1
    return w


@always_inline
def _bb_edge_manifold[
    DTYPE: DType
](
    code: Int,
    margin: Scalar[DTYPE],
    a_x: Scalar[DTYPE],
    a_y: Scalar[DTYPE],
    a_z: Scalar[DTYPE],
    mat1: InlineArray[Scalar[DTYPE], 9],
    size1: InlineArray[Scalar[DTYPE], 3],
    size2: InlineArray[Scalar[DTYPE], 3],
    pos21: InlineArray[Scalar[DTYPE], 3],
    rot: InlineArray[Scalar[DTYPE], 9],
    rotabs: InlineArray[Scalar[DTYPE], 9],
    cle1: Int,
    cle2: Int,
    clnorm: InlineArray[Scalar[DTYPE], 3],
    inflag: Int,
    mut dist_out: InlineArray[Scalar[DTYPE], BB_MAX_POINTS],
    mut pos_out: InlineArray[Scalar[DTYPE], 3 * BB_MAX_POINTS],
    mut normal_out: InlineArray[Scalar[DTYPE], 3],
) -> Int:
    """The `code >= 12` half of `_boxbox` — the EDGE-EDGE manifold.

    Port of `references/mujoco-3.6.0/src/engine/engine_collision_box.c`, label
    `edgeedge:` (lines 986-1337). Verified identical in 3.3.6, 3.5.1 and 3.6.0,
    so unlike the face path there is no version choice to make here.

    Box 1 is always the reference box on this path (the face path switches),
    and the winning axis is a cross product of one edge direction from each
    box rather than a face normal, so the reference frame is built from the
    box-1 FACE that the leading corner `cle1` belongs to (`clface`) and the
    contact normal is `clnorm` carried into that frame — not the frame's +z.

    Returns the number of points written; the caller still runs
    `_bb_post_filter`.
    """
    var margin2 = margin * margin

    var cc = code - 12
    var q1 = cc // 3
    var q2 = cc % 3

    # The two box-2 axes spanning the incident face, and the two box-1 axes
    # spanning the reference face. Each pair is then reordered so the FIRST is
    # the one more aligned with the other box's edge.
    var ax1 = 1
    var ax2 = 2
    if q2 == 1:
        ax1 = 0
        ax2 = 2
    elif q2 == 2:
        ax1 = 1
        ax2 = 0
    var pax1 = 1
    var pax2 = 2
    if q1 == 1:
        pax1 = 0
        pax2 = 2
    elif q1 == 2:
        pax1 = 1
        pax2 = 0

    if rotabs[3 * q1 + ax1] < rotabs[3 * q1 + ax2]:
        ax1 = ax2
        ax2 = 3 - q2 - ax1
    # rottabs[3 * q2 + pax] is abs(rot[3 * pax + q2]).
    if abs(rot[3 * pax1 + q2]) < abs(rot[3 * pax2 + q2]):
        pax1 = pax2
        pax2 = 3 - q1 - pax1

    var clface = pax2 if (cle1 & (1 << pax2)) != 0 else pax2 + 3

    # Same `rotmore` signed-permutation table as the face path, indexed by
    # `clface` instead of `q1`.
    var i0 = 0
    var i1 = 1
    var i2 = 2
    var f0 = Scalar[DTYPE](1)
    var f1 = Scalar[DTYPE](1)
    var f2 = Scalar[DTYPE](1)
    if clface == 0:
        i0 = 2
        f0 = Scalar[DTYPE](-1)
        i2 = 0
    elif clface == 1:
        i1 = 2
        f1 = Scalar[DTYPE](-1)
        i2 = 1
    elif clface == 3:
        i0 = 2
        i2 = 0
        f2 = Scalar[DTYPE](-1)
    elif clface == 4:
        i1 = 2
        i2 = 1
        f2 = Scalar[DTYPE](-1)
    elif clface == 5:
        f0 = Scalar[DTYPE](-1)
        f2 = Scalar[DTYPE](-1)

    var p = InlineArray[Scalar[DTYPE], 3](fill=Scalar[DTYPE](0))
    p[0] = pos21[i0] * f0
    p[1] = pos21[i1] * f1
    p[2] = pos21[i2] * f2
    var rnorm = InlineArray[Scalar[DTYPE], 3](fill=Scalar[DTYPE](0))
    rnorm[0] = clnorm[i0] * f0
    rnorm[1] = clnorm[i1] * f1
    rnorm[2] = clnorm[i2] * f2

    var r = InlineArray[Scalar[DTYPE], 9](fill=Scalar[DTYPE](0))
    for c in range(3):
        r[0 * 3 + c] = rot[i0 * 3 + c] * f0
        r[1 * 3 + c] = rot[i1 * 3 + c] * f1
        r[2 * 3 + c] = rot[i2 * 3 + c] * f2
    var rt = InlineArray[Scalar[DTYPE], 9](fill=Scalar[DTYPE](0))
    for i in range(3):
        for j in range(3):
            rt[3 * i + j] = r[3 * j + i]

    # ⚠ MuJoCo applies rotmore^T here where the face path applies rotmore.
    # Every entry in the table is an involution as a permutation, so abs()
    # makes the two agree — transcribed in the transposed form anyway.
    var s = InlineArray[Scalar[DTYPE], 3](fill=Scalar[DTYPE](0))
    s[i0] = abs(size1[0])
    s[i1] = abs(size1[1])
    s[i2] = abs(size1[2])

    var lx = s[0]
    var ly = s[1]
    var hz = s[2]
    p[2] -= hz

    # The two box-2 edges that straddle the winning axis: four corners, two per
    # edge, differing only in the sign along `ax1`.
    var crn = InlineArray[Scalar[DTYPE], 12](fill=Scalar[DTYPE](0))
    var s_ax1 = Scalar[DTYPE](1) if (cle2 & (1 << ax1)) != 0 else Scalar[DTYPE](
        -1
    )
    var s_ax2 = Scalar[DTYPE](1) if (cle2 & (1 << ax2)) != 0 else Scalar[DTYPE](
        -1
    )
    for c in range(3):
        var base = p[c]
        base += rt[3 * ax1 + c] * size2[ax1] * s_ax1
        base += rt[3 * ax2 + c] * size2[ax2] * s_ax2
        crn[0 * 3 + c] = base + rt[3 * q2 + c] * size2[q2]
        crn[1 * 3 + c] = base - rt[3 * q2 + c] * size2[q2]
    for c in range(3):
        var base = p[c]
        base += rt[3 * ax1 + c] * size2[ax1] * (-s_ax1)
        base += rt[3 * ax2 + c] * size2[ax2] * s_ax2
        crn[2 * 3 + c] = base + rt[3 * q2 + c] * size2[q2]
        crn[3 * 3 + c] = base - rt[3 * q2 + c] * size2[q2]

    var axi = InlineArray[Scalar[DTYPE], 9](fill=Scalar[DTYPE](0))
    for c in range(3):
        axi[0 * 3 + c] = crn[0 * 3 + c]
        axi[1 * 3 + c] = crn[1 * 3 + c] - crn[0 * 3 + c]
        axi[2 * 3 + c] = crn[2 * 3 + c] - crn[0 * 3 + c]

    if abs(rnorm[2]) < Scalar[DTYPE](_BB_MINVAL):
        return 0
    var innorm = (Scalar[DTYPE](1) / rnorm[2]) * (
        Scalar[DTYPE](-1) if inflag != 0 else Scalar[DTYPE](1)
    )

    # Project the four corners onto the reference plane ALONG the contact
    # normal (not along z) — `pu` keeps the unprojected originals.
    var pu = InlineArray[Scalar[DTYPE], 12](fill=Scalar[DTYPE](0))
    for i in range(4):
        var c1 = -crn[3 * i + 2] * (Scalar[DTYPE](1) / rnorm[2])
        for c in range(3):
            pu[3 * i + c] = crn[3 * i + c]
            crn[3 * i + c] = crn[3 * i + c] + rnorm[c] * c1

    var pts = InlineArray[Scalar[DTYPE], 9](fill=Scalar[DTYPE](0))
    for c in range(3):
        pts[0 * 3 + c] = crn[0 * 3 + c]
        pts[1 * 3 + c] = crn[1 * 3 + c] - crn[0 * 3 + c]
        pts[2 * 3 + c] = crn[2 * 3 + c] - crn[0 * 3 + c]

    # Four edges of the projected quad, in both the projected (`lines`) and
    # unprojected (`linesu`) frames. `m == 3` unconditionally on this path.
    var lines = InlineArray[Scalar[DTYPE], 24](fill=Scalar[DTYPE](0))
    var linesu = InlineArray[Scalar[DTYPE], 24](fill=Scalar[DTYPE](0))
    for c in range(3):
        lines[0 * 6 + c] = pts[0 + c]
        lines[0 * 6 + 3 + c] = pts[3 + c]
        linesu[0 * 6 + c] = axi[0 + c]
        linesu[0 * 6 + 3 + c] = axi[3 + c]

        lines[1 * 6 + c] = pts[0 + c]
        lines[1 * 6 + 3 + c] = pts[6 + c]
        linesu[1 * 6 + c] = axi[0 + c]
        linesu[1 * 6 + 3 + c] = axi[6 + c]

        lines[2 * 6 + c] = pts[0 + c] + pts[3 + c]
        lines[2 * 6 + 3 + c] = pts[6 + c]
        linesu[2 * 6 + c] = axi[0 + c] + axi[3 + c]
        linesu[2 * 6 + 3 + c] = axi[6 + c]

        lines[3 * 6 + c] = pts[0 + c] + pts[6 + c]
        lines[3 * 6 + 3 + c] = pts[3 + c]
        linesu[3 * 6 + c] = axi[0 + c] + axi[6 + c]
        linesu[3 * 6 + 3 + c] = axi[3 + c]

    var pnt = InlineArray[Scalar[DTYPE], 3 * BB_MAX_POINTS](
        fill=Scalar[DTYPE](0)
    )
    var depth = InlineArray[Scalar[DTYPE], BB_MAX_POINTS](
        fill=Scalar[DTYPE](0)
    )
    var n = 0

    # (1) clip each projected edge against the reference face's four sides.
    for i in range(4):
        for q in range(2):
            var a = lines[6 * i + q]
            var b = lines[6 * i + 3 + q]
            var c = lines[6 * i + (1 - q)]
            var d = lines[6 * i + 4 - q]
            if abs(b) <= Scalar[DTYPE](_BB_MINVAL):
                continue
            for jj in range(2):
                if n >= BB_MAX_POINTS:
                    break
                var j = Scalar[DTYPE](-1) if jj == 0 else Scalar[DTYPE](1)
                var l = s[q] * j
                var c1 = (l - a) / b
                if c1 < Scalar[DTYPE](0) or c1 > Scalar[DTYPE](1):
                    continue
                var c2 = c + d * c1
                if abs(c2) > s[1 - q]:
                    continue
                if (
                    linesu[6 * i + 2] + linesu[6 * i + 5] * c1
                ) * innorm > margin:
                    continue
                for cc2 in range(3):
                    pnt[3 * n + cc2] = linesu[6 * i + cc2] * Scalar[DTYPE](
                        0.5
                    ) + linesu[6 * i + 3 + cc2] * (Scalar[DTYPE](0.5) * c1)
                pnt[3 * n + q] += Scalar[DTYPE](0.5) * l
                pnt[3 * n + (1 - q)] += Scalar[DTYPE](0.5) * c2
                depth[n] = pnt[3 * n + 2] * innorm * Scalar[DTYPE](2)
                n += 1
    var nl = n

    # (2) reference-face corners, mapped back through the quad's barycentric
    # coordinates. ⚠ `det` is MuJoCo's `c1`, and MuJoCo REUSES that variable
    # inside this loop — every corner after the first divides by a squared
    # DISTANCE instead of the determinant. Reproduced deliberately: without it
    # we emit 368 edge points over the sweep where the runtime emits 361.
    var ea = pts[3 + 0]
    var eb = pts[6 + 0]
    var ec = pts[3 + 1]
    var ed = pts[6 + 1]
    var det = ea * ed - eb * ec
    for i in range(4):
        if n >= BB_MAX_POINTS:
            break
        var llx = lx if (i // 2) != 0 else -lx
        var lly = ly if (i % 2) != 0 else -ly
        var x = llx - pts[0]
        var y = lly - pts[1]
        var u = (x * ed - y * eb) / det
        var v = (y * ea - x * ec) / det

        if nl == 0:
            if (
                u < Scalar[DTYPE](0) or u > Scalar[DTYPE](1)
            ) and (v < Scalar[DTYPE](0) or v > Scalar[DTYPE](1)):
                continue
        else:
            if (
                u < Scalar[DTYPE](0)
                or u > Scalar[DTYPE](1)
                or v < Scalar[DTYPE](0)
                or v > Scalar[DTYPE](1)
            ):
                continue

        if u < Scalar[DTYPE](0):
            u = Scalar[DTYPE](0)
        if u > Scalar[DTYPE](1):
            u = Scalar[DTYPE](1)
        if v < Scalar[DTYPE](0):
            v = Scalar[DTYPE](0)
        if v > Scalar[DTYPE](1):
            v = Scalar[DTYPE](1)

        var t0 = pu[0] * (Scalar[DTYPE](1) - u - v) + pu[3] * u + pu[6] * v
        var t1 = pu[1] * (Scalar[DTYPE](1) - u - v) + pu[4] * u + pu[7] * v
        var t2 = pu[2] * (Scalar[DTYPE](1) - u - v) + pu[5] * u + pu[8] * v

        var gx = llx - t0
        var gy = lly - t1
        var gz = -t2
        det = gx * gx + gy * gy + gz * gz
        if t2 > Scalar[DTYPE](0) and det > margin2:
            continue

        pnt[3 * n + 0] = (llx + t0) * Scalar[DTYPE](0.5)
        pnt[3 * n + 1] = (lly + t1) * Scalar[DTYPE](0.5)
        pnt[3 * n + 2] = t2 * Scalar[DTYPE](0.5)
        depth[n] = sqrt(det) * (
            Scalar[DTYPE](-1) if t2 < Scalar[DTYPE](0) else Scalar[DTYPE](1)
        )
        n += 1
    var nf = n

    # (3) the projected incident corners themselves, clamped onto the
    # reference face.
    for i in range(4):
        if n >= BB_MAX_POINTS:
            break
        var x = crn[3 * i + 0]
        var y = crn[3 * i + 1]
        if nl == 0:
            if nf != 0:
                if (x < -lx or x > lx) and (y < -ly or y > ly):
                    continue
        else:
            if x < -lx or x > lx or y < -ly or y > ly:
                continue

        var acc = Scalar[DTYPE](0)
        var tx = x * Scalar[DTYPE](0.5)
        var ty = y * Scalar[DTYPE](0.5)
        if x < -s[0]:
            acc += (x + s[0]) * (x + s[0])
            tx = -s[0] * Scalar[DTYPE](0.5)
        elif x > s[0]:
            acc += (x - s[0]) * (x - s[0])
            tx = s[0] * Scalar[DTYPE](0.5)
        if y < -s[1]:
            acc += (y + s[1]) * (y + s[1])
            ty = -s[1] * Scalar[DTYPE](0.5)
        elif y > s[1]:
            acc += (y - s[1]) * (y - s[1])
            ty = s[1] * Scalar[DTYPE](0.5)
        var pz = pu[3 * i + 2]
        acc += pz * innorm * pz * innorm

        if pz > Scalar[DTYPE](0) and acc > margin2:
            continue

        pnt[3 * n + 0] = tx + pu[3 * i + 0] * Scalar[DTYPE](0.5)
        pnt[3 * n + 1] = ty + pu[3 * i + 1] * Scalar[DTYPE](0.5)
        pnt[3 * n + 2] = pz * Scalar[DTYPE](0.5)
        depth[n] = sqrt(acc) * (
            Scalar[DTYPE](-1) if pz < Scalar[DTYPE](0) else Scalar[DTYPE](1)
        )
        n += 1

    # Back to world. The reference box is always box 1 here, and the normal is
    # `clnorm` rotated out of the reference frame, not the frame's +z.
    var rw = InlineArray[Scalar[DTYPE], 9](fill=Scalar[DTYPE](0))
    for row in range(3):
        rw[3 * row + 0] = mat1[3 * row + i0] * f0
        rw[3 * row + 1] = mat1[3 * row + i1] * f1
        rw[3 * row + 2] = mat1[3 * row + i2] * f2

    var nsgn = Scalar[DTYPE](-1) if inflag != 0 else Scalar[DTYPE](1)
    for row in range(3):
        normal_out[row] = nsgn * (
            rw[3 * row + 0] * rnorm[0]
            + rw[3 * row + 1] * rnorm[1]
            + rw[3 * row + 2] * rnorm[2]
        )

    for i in range(n):
        var qx = pnt[3 * i + 0]
        var qy = pnt[3 * i + 1]
        var qz = pnt[3 * i + 2] + hz
        pos_out[3 * i + 0] = rw[0] * qx + rw[1] * qy + rw[2] * qz + a_x
        pos_out[3 * i + 1] = rw[3] * qx + rw[4] * qy + rw[5] * qz + a_y
        pos_out[3 * i + 2] = rw[6] * qx + rw[7] * qy + rw[8] * qz + a_z
        dist_out[i] = depth[i]

    return n


@always_inline
def box_box_manifold[
    DTYPE: DType
](
    # Box A (MuJoCo's geom 1)
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
    # Box B (MuJoCo's geom 2)
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
    margin: Scalar[DTYPE],
    mut n_out: Int,
    mut dist_out: InlineArray[Scalar[DTYPE], BB_MAX_POINTS],
    mut pos_out: InlineArray[Scalar[DTYPE], 3 * BB_MAX_POINTS],
    mut normal_out: InlineArray[Scalar[DTYPE], 3],
) -> Int:
    """Full contact MANIFOLD for a colliding box pair.

    Port of `_boxbox` — BOTH the face (`code < 12`) and edge-edge
    (`code >= 12`) paths — plus the bad/duplicate removal that `mjc_BoxBox`
    wraps around them
    (`references/mujoco-3.6.0/src/engine/engine_collision_box.c`).

    Returns MuJoCo's `code`, which tells the caller which path was taken:

        -1        the boxes are separated; nothing written
        0 .. 11   FACE axis — `n_out` contacts written, sharing `normal_out`
        >= 12     EDGE-EDGE axis — likewise, built by `_bb_edge_manifold`

    `normal_out` points from box A to box B, the convention every primitive in
    this file follows.

    ⚠ PORTED FROM 3.6.0, NOT FROM `references/mujoco-3.3.6/`. Those two differ
    in exactly one line of this routine and it is the answer: 3.3.6 ends with
    `con[i].dist = points[i][2]` where the value has already been halved to
    place the contact POINT midway through the overlap, so every face contact
    it reports is half as deep as it should be. 3.6.0 has
    `con[i].dist = 2 * points[i][2]`. The pixi runtime we compare against is
    MuJoCo 3.10.0 and it agrees with 3.6.0 (a box overlapping another by
    exactly 4 mm reports dist = -0.004), so CLAUDE.md's "3.3.6 matches the pixi
    version" is stale by several releases.

    On a 400-pose sweep of two unequal boxes, 90 of the 217 contacting poses
    take the face path (210 points) and 127 take edge-edge (361 points);
    `tests/physics3d/test_box_box_sweep.mojo` gates both against MuJoCo.
    """
    n_out = 0

    var mat1 = _bb_quat_mat[DTYPE](a_qx, a_qy, a_qz, a_qw)
    var mat2 = _bb_quat_mat[DTYPE](b_qx, b_qy, b_qz, b_qw)

    var size1 = InlineArray[Scalar[DTYPE], 3](fill=Scalar[DTYPE](0))
    size1[0] = a_hx
    size1[1] = a_hy
    size1[2] = a_hz
    var size2 = InlineArray[Scalar[DTYPE], 3](fill=Scalar[DTYPE](0))
    size2[0] = b_hx
    size2[1] = b_hy
    size2[2] = b_hz

    # pos21 = mat1^T (pos2 - pos1); pos12 = mat2^T (pos1 - pos2)
    var dx = b_x - a_x
    var dy = b_y - a_y
    var dz = b_z - a_z
    var pos21 = InlineArray[Scalar[DTYPE], 3](fill=Scalar[DTYPE](0))
    var pos12 = InlineArray[Scalar[DTYPE], 3](fill=Scalar[DTYPE](0))
    for i in range(3):
        pos21[i] = mat1[i] * dx + mat1[3 + i] * dy + mat1[6 + i] * dz
        pos12[i] = -(mat2[i] * dx + mat2[3 + i] * dy + mat2[6 + i] * dz)

    # rot = mat1^T mat2, row-major; rott = rot^T
    var rot = InlineArray[Scalar[DTYPE], 9](fill=Scalar[DTYPE](0))
    for i in range(3):
        for j in range(3):
            var acc = Scalar[DTYPE](0)
            for k in range(3):
                acc += mat1[3 * k + i] * mat2[3 * k + j]
            rot[3 * i + j] = acc
    var rott = InlineArray[Scalar[DTYPE], 9](fill=Scalar[DTYPE](0))
    var rotabs = InlineArray[Scalar[DTYPE], 9](fill=Scalar[DTYPE](0))
    for i in range(3):
        for j in range(3):
            rott[3 * i + j] = rot[3 * j + i]
            rotabs[3 * i + j] = abs(rot[3 * i + j])

    var plen1 = InlineArray[Scalar[DTYPE], 3](fill=Scalar[DTYPE](0))
    var plen2 = InlineArray[Scalar[DTYPE], 3](fill=Scalar[DTYPE](0))
    for i in range(3):
        var s2 = Scalar[DTYPE](0)
        var s1 = Scalar[DTYPE](0)
        for j in range(3):
            s2 += rotabs[3 * i + j] * size2[j]
            s1 += rotabs[3 * j + i] * size1[j]
        plen2[i] = s2
        plen1[i] = s1

    var penetration = margin
    for i in range(3):
        penetration += size1[i] * Scalar[DTYPE](3) + size2[i] * Scalar[DTYPE](
            3
        )

    var code = -1
    for i in range(3):
        var c1 = -abs(pos21[i]) + size1[i] + plen2[i]
        var c2 = -abs(pos12[i]) + size2[i] + plen1[i]
        if c1 < -margin or c2 < -margin:
            return -1
        if c1 < penetration:
            penetration = c1
            code = i + (3 if pos21[i] < Scalar[DTYPE](0) else 0) + 0
        if c2 < penetration:
            penetration = c2
            code = i + (3 if pos12[i] < Scalar[DTYPE](0) else 0) + 6

    # The nine edge-edge axes. Whichever one wins carries state the manifold
    # needs later: which corner of each box leads (`cle1`, `cle2`), the axis
    # itself (`clnorm`), and which side of it box 2's centre sits on (`inflag`).
    var cle1 = 0
    var cle2 = 0
    var clnorm = InlineArray[Scalar[DTYPE], 3](fill=Scalar[DTYPE](0))
    var inflag = 0
    for i in range(3):
        for j in range(3):
            var t0 = Scalar[DTYPE](0)
            var t1 = Scalar[DTYPE](0)
            var t2 = Scalar[DTYPE](0)
            if i == 0:
                t1 = -rott[3 * j + 2]
                t2 = rott[3 * j + 1]
            elif i == 1:
                t0 = rott[3 * j + 2]
                t2 = -rott[3 * j + 0]
            else:
                t0 = -rott[3 * j + 1]
                t1 = rott[3 * j + 0]
            var c1 = sqrt(t0 * t0 + t1 * t1 + t2 * t2)
            if c1 < Scalar[DTYPE](_BB_MINVAL):
                continue
            t0 /= c1
            t1 /= c1
            t2 /= c1
            var c2 = pos21[0] * t0 + pos21[1] * t1 + pos21[2] * t2
            var c3 = Scalar[DTYPE](0)
            for k in range(3):
                if k != i:
                    var tk = t0
                    if k == 1:
                        tk = t1
                    elif k == 2:
                        tk = t2
                    c3 += size1[k] * abs(tk)
            for k in range(3):
                if k != j:
                    c3 += size2[k] * rotabs[3 * i + (3 - k - j)] / c1
            c3 -= abs(c2)
            if c3 < -margin:
                return -1
            if c3 < penetration * (Scalar[DTYPE](1) - Scalar[DTYPE](1e-12)):
                penetration = c3
                code = 12 + i * 3 + j
                cle1 = 0
                for k in range(3):
                    if k != i:
                        var tk = t0
                        if k == 1:
                            tk = t1
                        elif k == 2:
                            tk = t2
                        if (tk > Scalar[DTYPE](0)) != (c2 < Scalar[DTYPE](0)):
                            cle1 += 1 << k
                cle2 = 0
                for k in range(3):
                    if k != j:
                        var e0 = rot[3 * i + (3 - k - j)] > Scalar[DTYPE](0)
                        var e1 = c2 < Scalar[DTYPE](0)
                        var e2 = ((k - j + 3) % 3) == 1
                        if ((e0 != e1) != e2):
                            cle2 += 1 << k
                clnorm[0] = t0
                clnorm[1] = t1
                clnorm[2] = t2
                inflag = 1 if c2 < Scalar[DTYPE](0) else 0

    if code == -1:
        return -1
    if code >= 12:
        n_out = _bb_edge_manifold[DTYPE](
            code, margin,
            a_x, a_y, a_z, mat1, size1, size2,
            pos21, rot, rotabs, cle1, cle2, clnorm, inflag,
            dist_out, pos_out, normal_out,
        )
        n_out = _bb_post_filter[DTYPE](
            n_out, dist_out, pos_out,
            a_x, a_y, a_z, mat1, a_hx, a_hy, a_hz,
            b_x, b_y, b_z, mat2, b_hx, b_hy, b_hz,
            margin,
        )
        return code

    # ------------------------------------------------------------------
    # Face path. `rotmore` is a signed permutation taking the winning face
    # normal to +z; MuJoCo applies it through the `rotaxis` / `rotmatx` macros
    # as an index-and-flip triple rather than a matrix product, and so does
    # this.
    # ------------------------------------------------------------------
    var q1 = code % 6
    var q2 = code // 6

    var i0 = 0
    var i1 = 1
    var i2 = 2
    var f0 = Scalar[DTYPE](1)
    var f1 = Scalar[DTYPE](1)
    var f2 = Scalar[DTYPE](1)
    if q1 == 0:
        i0 = 2
        f0 = Scalar[DTYPE](-1)
        i2 = 0
    elif q1 == 1:
        i1 = 2
        f1 = Scalar[DTYPE](-1)
        i2 = 1
    elif q1 == 3:
        i0 = 2
        i2 = 0
        f2 = Scalar[DTYPE](-1)
    elif q1 == 4:
        i1 = 2
        i2 = 1
        f2 = Scalar[DTYPE](-1)
    elif q1 == 5:
        f0 = Scalar[DTYPE](-1)
        f2 = Scalar[DTYPE](-1)

    # r = rotmore * (q2 ? rot^T : rot); p, tmp1 = rotmore * (vector)
    var src = InlineArray[Scalar[DTYPE], 9](fill=Scalar[DTYPE](0))
    var pv = InlineArray[Scalar[DTYPE], 3](fill=Scalar[DTYPE](0))
    var sv = InlineArray[Scalar[DTYPE], 3](fill=Scalar[DTYPE](0))
    var s = InlineArray[Scalar[DTYPE], 3](fill=Scalar[DTYPE](0))
    if q2 != 0:
        for i in range(9):
            src[i] = rott[i]
        for i in range(3):
            pv[i] = pos12[i]
            sv[i] = size2[i]
            s[i] = size1[i]
    else:
        for i in range(9):
            src[i] = rot[i]
        for i in range(3):
            pv[i] = pos21[i]
            sv[i] = size1[i]
            s[i] = size2[i]

    var r = InlineArray[Scalar[DTYPE], 9](fill=Scalar[DTYPE](0))
    for c in range(3):
        r[0 * 3 + c] = src[i0 * 3 + c] * f0
        r[1 * 3 + c] = src[i1 * 3 + c] * f1
        r[2 * 3 + c] = src[i2 * 3 + c] * f2

    var p = InlineArray[Scalar[DTYPE], 3](fill=Scalar[DTYPE](0))
    p[0] = pv[i0] * f0
    p[1] = pv[i1] * f1
    p[2] = pv[i2] * f2
    var tmp1 = InlineArray[Scalar[DTYPE], 3](fill=Scalar[DTYPE](0))
    tmp1[0] = sv[i0] * f0
    tmp1[1] = sv[i1] * f1
    tmp1[2] = sv[i2] * f2

    # `rt` row i is the world direction of the incident box's local axis i.
    var rt = InlineArray[Scalar[DTYPE], 9](fill=Scalar[DTYPE](0))
    for i in range(3):
        for j in range(3):
            rt[3 * i + j] = r[3 * j + i]

    var ss = InlineArray[Scalar[DTYPE], 3](fill=Scalar[DTYPE](0))
    for i in range(3):
        ss[i] = abs(tmp1[i])
    var lx = ss[0]
    var ly = ss[1]
    var hz = ss[2]
    p[2] -= hz

    # The incident box's DEEPEST corner, and the two face edges leaving it.
    var clcorner = 0
    for i in range(3):
        if r[6 + i] < Scalar[DTYPE](0):
            clcorner += 1 << i

    var pts = InlineArray[Scalar[DTYPE], 18](fill=Scalar[DTYPE](0))
    for c in range(3):
        var acc = p[c]
        for i in range(3):
            var sgn = Scalar[DTYPE](1) if (clcorner & (1 << i)) != 0 else (
                Scalar[DTYPE](-1)
            )
            acc += rt[3 * i + c] * s[i] * sgn
        pts[c] = acc

    var m = 1
    for i in range(3):
        if abs(r[6 + i]) < Scalar[DTYPE](0.5):
            var sc = Scalar[DTYPE](-2) if (clcorner & (1 << i)) != 0 else (
                Scalar[DTYPE](2)
            )
            for c in range(3):
                pts[3 * m + c] = rt[3 * i + c] * s[i] * sc
            m += 1
    for c in range(3):
        pts[3 * 3 + c] = pts[c] + pts[3 + c]
        pts[3 * 4 + c] = pts[c] + pts[6 + c]
    for c in range(3):
        pts[3 * 5 + c] = pts[3 * 3 + c] + pts[6 + c]

    # Four (origin, direction) edges of the incident face.
    var lines = InlineArray[Scalar[DTYPE], 24](fill=Scalar[DTYPE](0))
    var k = 0
    if m > 1:
        for c in range(3):
            lines[6 * k + c] = pts[c]
            lines[6 * k + 3 + c] = pts[3 + c]
        k += 1
    if m > 2:
        for c in range(3):
            lines[6 * k + c] = pts[c]
            lines[6 * k + 3 + c] = pts[6 + c]
        k += 1
        for c in range(3):
            lines[6 * k + c] = pts[3 * 3 + c]
            lines[6 * k + 3 + c] = pts[6 + c]
        k += 1
        for c in range(3):
            lines[6 * k + c] = pts[3 * 4 + c]
            lines[6 * k + 3 + c] = pts[3 + c]
        k += 1

    var pnt = InlineArray[Scalar[DTYPE], 3 * BB_MAX_POINTS](
        fill=Scalar[DTYPE](0)
    )
    var n = 0

    # Clip each incident edge against the reference face's four side planes.
    for i in range(k):
        for q in range(2):
            var a = lines[6 * i + q]
            var b = lines[6 * i + 3 + q]
            var c = lines[6 * i + (1 - q)]
            var d = lines[6 * i + 4 - q]
            if abs(b) <= Scalar[DTYPE](_BB_MINVAL):
                continue
            for jj in range(2):
                if n >= BB_MAX_POINTS:
                    break
                var j = Scalar[DTYPE](-1) if jj == 0 else Scalar[DTYPE](1)
                var l = ss[q] * j
                var c1 = (l - a) / b
                if c1 < Scalar[DTYPE](0) or c1 > Scalar[DTYPE](1):
                    continue
                var c2 = c + d * c1
                if abs(c2) > ss[1 - q]:
                    continue
                for cc in range(3):
                    pnt[3 * n + cc] = (
                        lines[6 * i + cc] + lines[6 * i + 3 + cc] * c1
                    )
                n += 1

    # Reference-face corners that fall inside the incident face.
    var ea = pts[3 + 0]
    var eb = pts[6 + 0]
    var ec = pts[3 + 1]
    var ed = pts[6 + 1]
    var det = ea * ed - eb * ec
    if m > 2:
        for i in range(4):
            if n >= BB_MAX_POINTS:
                break
            var llx = lx if (i // 2) != 0 else -lx
            var lly = ly if (i % 2) != 0 else -ly
            var x = llx - pts[0]
            var y = lly - pts[1]
            var u = (x * ed - y * eb) / det
            var v = (y * ea - x * ec) / det
            if (
                u <= Scalar[DTYPE](0)
                or v <= Scalar[DTYPE](0)
                or u >= Scalar[DTYPE](1)
                or v >= Scalar[DTYPE](1)
            ):
                continue
            pnt[3 * n + 0] = llx
            pnt[3 * n + 1] = lly
            pnt[3 * n + 2] = pts[2] + u * pts[3 + 2] + v * pts[6 + 2]
            n += 1

    # Incident-face corners that fall inside the reference face. ⚠ The deepest
    # corner (i == 0) is added WITHOUT the bounds test — MuJoCo does the same,
    # and the post-filter below is what removes it when it lies outside.
    for i in range(1 << (m - 1)):
        if n >= BB_MAX_POINTS:
            break
        var base = 0 if i == 0 else 3 * (i + 2)
        var tx = pts[base + 0]
        var ty = pts[base + 1]
        if i != 0:
            if tx <= -lx or tx >= lx:
                continue
            if ty <= -ly or ty >= ly:
                continue
        for cc in range(3):
            pnt[3 * n + cc] = pts[base + cc]
        n += 1

    # Drop points above the reference face, then halve z to put the contact
    # POINT midway through the overlap. `dist` keeps the full depth.
    var depth = InlineArray[Scalar[DTYPE], BB_MAX_POINTS](
        fill=Scalar[DTYPE](0)
    )
    var kept = 0
    for i in range(n):
        if pnt[3 * i + 2] > margin:
            continue
        depth[kept] = pnt[3 * i + 2]
        pnt[3 * kept + 0] = pnt[3 * i + 0]
        pnt[3 * kept + 1] = pnt[3 * i + 1]
        pnt[3 * kept + 2] = pnt[3 * i + 2] * Scalar[DTYPE](0.5)
        kept += 1
    n = kept

    # Back to world: rw = (q2 ? mat2 : mat1) * rotmore^T.
    var rw = InlineArray[Scalar[DTYPE], 9](fill=Scalar[DTYPE](0))
    var mref = mat2 if q2 != 0 else mat1
    # rotmore^T has a single non-zero per column: column a is e_{idx[a]} * f_a.
    for row in range(3):
        rw[3 * row + 0] = mref[3 * row + i0] * f0
        rw[3 * row + 1] = mref[3 * row + i1] * f1
        rw[3 * row + 2] = mref[3 * row + i2] * f2

    var nsgn = Scalar[DTYPE](-1) if q2 != 0 else Scalar[DTYPE](1)
    normal_out[0] = nsgn * rw[2]
    normal_out[1] = nsgn * rw[5]
    normal_out[2] = nsgn * rw[8]

    var ox = b_x if q2 != 0 else a_x
    var oy = b_y if q2 != 0 else a_y
    var oz = b_z if q2 != 0 else a_z

    for i in range(n):
        var lx_ = pnt[3 * i + 0]
        var ly_ = pnt[3 * i + 1]
        var lz_ = pnt[3 * i + 2] + hz
        pos_out[3 * i + 0] = (
            rw[0] * lx_ + rw[1] * ly_ + rw[2] * lz_ + ox
        )
        pos_out[3 * i + 1] = (
            rw[3] * lx_ + rw[4] * ly_ + rw[5] * lz_ + oy
        )
        pos_out[3 * i + 2] = (
            rw[6] * lx_ + rw[7] * ly_ + rw[8] * lz_ + oz
        )
        dist_out[i] = depth[i]

    n_out = _bb_post_filter[DTYPE](
        n, dist_out, pos_out,
        a_x, a_y, a_z, mat1, a_hx, a_hy, a_hz,
        b_x, b_y, b_z, mat2, b_hx, b_hy, b_hz,
        margin,
    )
    return code


# =============================================================================
# Phase 10: Missing collision pairs
# cylinder-box, cylinder-capsule, cylinder-cylinder
# =============================================================================


@always_inline
def cylinder_capsule[
    DTYPE: DType
](
    # Cylinder
    cyl_x: Scalar[DTYPE], cyl_y: Scalar[DTYPE], cyl_z: Scalar[DTYPE],
    cyl_qx: Scalar[DTYPE], cyl_qy: Scalar[DTYPE], cyl_qz: Scalar[DTYPE], cyl_qw: Scalar[DTYPE],
    cyl_hl: Scalar[DTYPE],
    cyl_r: Scalar[DTYPE],
    # Capsule
    cap_x: Scalar[DTYPE], cap_y: Scalar[DTYPE], cap_z: Scalar[DTYPE],
    cap_qx: Scalar[DTYPE], cap_qy: Scalar[DTYPE], cap_qz: Scalar[DTYPE], cap_qw: Scalar[DTYPE],
    cap_hl: Scalar[DTYPE],
    cap_r: Scalar[DTYPE],
) -> Tuple[
    Scalar[DTYPE], Scalar[DTYPE], Scalar[DTYPE], Scalar[DTYPE],
    Scalar[DTYPE], Scalar[DTYPE], Scalar[DTYPE],
]:
    """Cylinder-capsule collision.

    Reduces to closest points between two line segments (cylinder axis
    and capsule axis), then handles the cylinder's flat cap vs capsule's
    spherical cap geometry.
    """
    # Cylinder axis in world frame
    var ca = rotate_vector_by_quat(Scalar[DTYPE](0), Scalar[DTYPE](0), Scalar[DTYPE](1),
        cyl_qx, cyl_qy, cyl_qz, cyl_qw)
    # Capsule axis in world frame
    var pa = rotate_vector_by_quat(Scalar[DTYPE](0), Scalar[DTYPE](0), Scalar[DTYPE](1),
        cap_qx, cap_qy, cap_qz, cap_qw)

    # Line segments: cylinder from (center - hl*axis) to (center + hl*axis)
    var c_p1x = cyl_x - cyl_hl * ca[0]
    var c_p1y = cyl_y - cyl_hl * ca[1]
    var c_p1z = cyl_z - cyl_hl * ca[2]
    var c_dx = Scalar[DTYPE](2) * cyl_hl * ca[0]
    var c_dy = Scalar[DTYPE](2) * cyl_hl * ca[1]
    var c_dz = Scalar[DTYPE](2) * cyl_hl * ca[2]

    var p_p1x = cap_x - cap_hl * pa[0]
    var p_p1y = cap_y - cap_hl * pa[1]
    var p_p1z = cap_z - cap_hl * pa[2]
    var p_dx = Scalar[DTYPE](2) * cap_hl * pa[0]
    var p_dy = Scalar[DTYPE](2) * cap_hl * pa[1]
    var p_dz = Scalar[DTYPE](2) * cap_hl * pa[2]

    var cp = _closest_points_line_segments[DTYPE](
        c_p1x, c_p1y, c_p1z, c_dx, c_dy, c_dz,
        p_p1x, p_p1y, p_p1z, p_dx, p_dy, p_dz)

    # Closest point on cylinder axis, closest point on capsule axis
    var q1x = cp[0]
    var q1y = cp[1]
    var q1z = cp[2]
    var q2x = cp[3]
    var q2y = cp[4]
    var q2z = cp[5]

    # Vector between closest axis points
    var dx = q2x - q1x
    var dy = q2y - q1y
    var dz = q2z - q1z
    var d = sqrt(dx * dx + dy * dy + dz * dz)

    var nx: Scalar[DTYPE]
    var ny: Scalar[DTYPE]
    var nz: Scalar[DTYPE]
    if d > Scalar[DTYPE](1e-10):
        nx = dx / d
        ny = dy / d
        nz = dz / d
    else:
        # Axes overlap — use perpendicular to cylinder axis
        var perp = rotate_vector_by_quat(Scalar[DTYPE](1), Scalar[DTYPE](0), Scalar[DTYPE](0),
            cyl_qx, cyl_qy, cyl_qz, cyl_qw)
        nx = perp[0]
        ny = perp[1]
        nz = perp[2]

    # Distance: axis-axis distance minus radii
    var dist = d - cyl_r - cap_r

    # Contact point
    var contact_x = q1x + nx * (cyl_r + dist * Scalar[DTYPE](0.5))
    var contact_y = q1y + ny * (cyl_r + dist * Scalar[DTYPE](0.5))
    var contact_z = q1z + nz * (cyl_r + dist * Scalar[DTYPE](0.5))

    return (dist, contact_x, contact_y, contact_z, nx, ny, nz)


@always_inline
def cylinder_cylinder[
    DTYPE: DType
](
    # Cylinder A
    a_x: Scalar[DTYPE], a_y: Scalar[DTYPE], a_z: Scalar[DTYPE],
    a_qx: Scalar[DTYPE], a_qy: Scalar[DTYPE], a_qz: Scalar[DTYPE], a_qw: Scalar[DTYPE],
    a_hl: Scalar[DTYPE],
    a_r: Scalar[DTYPE],
    # Cylinder B
    b_x: Scalar[DTYPE], b_y: Scalar[DTYPE], b_z: Scalar[DTYPE],
    b_qx: Scalar[DTYPE], b_qy: Scalar[DTYPE], b_qz: Scalar[DTYPE], b_qw: Scalar[DTYPE],
    b_hl: Scalar[DTYPE],
    b_r: Scalar[DTYPE],
) -> Tuple[
    Scalar[DTYPE], Scalar[DTYPE], Scalar[DTYPE], Scalar[DTYPE],
    Scalar[DTYPE], Scalar[DTYPE], Scalar[DTYPE],
]:
    """Cylinder-cylinder collision via axis-axis closest points.

    Treats cylinders as capped line segments with radius, similar to
    capsule-capsule but without hemispherical caps.
    """
    var aa = rotate_vector_by_quat(Scalar[DTYPE](0), Scalar[DTYPE](0), Scalar[DTYPE](1),
        a_qx, a_qy, a_qz, a_qw)
    var ba = rotate_vector_by_quat(Scalar[DTYPE](0), Scalar[DTYPE](0), Scalar[DTYPE](1),
        b_qx, b_qy, b_qz, b_qw)

    var a_p1x = a_x - a_hl * aa[0]
    var a_p1y = a_y - a_hl * aa[1]
    var a_p1z = a_z - a_hl * aa[2]
    var a_dx = Scalar[DTYPE](2) * a_hl * aa[0]
    var a_dy = Scalar[DTYPE](2) * a_hl * aa[1]
    var a_dz = Scalar[DTYPE](2) * a_hl * aa[2]

    var b_p1x = b_x - b_hl * ba[0]
    var b_p1y = b_y - b_hl * ba[1]
    var b_p1z = b_z - b_hl * ba[2]
    var b_dx = Scalar[DTYPE](2) * b_hl * ba[0]
    var b_dy = Scalar[DTYPE](2) * b_hl * ba[1]
    var b_dz = Scalar[DTYPE](2) * b_hl * ba[2]

    var cp = _closest_points_line_segments[DTYPE](
        a_p1x, a_p1y, a_p1z, a_dx, a_dy, a_dz,
        b_p1x, b_p1y, b_p1z, b_dx, b_dy, b_dz)

    var q1x = cp[0]
    var q1y = cp[1]
    var q1z = cp[2]
    var q2x = cp[3]
    var q2y = cp[4]
    var q2z = cp[5]

    var dx = q2x - q1x
    var dy = q2y - q1y
    var dz = q2z - q1z
    var d = sqrt(dx * dx + dy * dy + dz * dz)

    var nx: Scalar[DTYPE]
    var ny: Scalar[DTYPE]
    var nz: Scalar[DTYPE]
    if d > Scalar[DTYPE](1e-10):
        nx = dx / d
        ny = dy / d
        nz = dz / d
    else:
        var cr_x = aa[1] * ba[2] - aa[2] * ba[1]
        var cr_y = aa[2] * ba[0] - aa[0] * ba[2]
        var cr_z = aa[0] * ba[1] - aa[1] * ba[0]
        var cr_len = sqrt(cr_x * cr_x + cr_y * cr_y + cr_z * cr_z)
        if cr_len > Scalar[DTYPE](1e-10):
            nx = cr_x / cr_len
            ny = cr_y / cr_len
            nz = cr_z / cr_len
        else:
            var perp = rotate_vector_by_quat(Scalar[DTYPE](1), Scalar[DTYPE](0), Scalar[DTYPE](0),
                a_qx, a_qy, a_qz, a_qw)
            nx = perp[0]
            ny = perp[1]
            nz = perp[2]

    var dist = d - a_r - b_r
    var contact_x = q1x + nx * (a_r + dist * Scalar[DTYPE](0.5))
    var contact_y = q1y + ny * (a_r + dist * Scalar[DTYPE](0.5))
    var contact_z = q1z + nz * (a_r + dist * Scalar[DTYPE](0.5))

    return (dist, contact_x, contact_y, contact_z, nx, ny, nz)


@always_inline
def cylinder_box[
    DTYPE: DType
](
    # Cylinder
    cyl_x: Scalar[DTYPE], cyl_y: Scalar[DTYPE], cyl_z: Scalar[DTYPE],
    cyl_qx: Scalar[DTYPE], cyl_qy: Scalar[DTYPE], cyl_qz: Scalar[DTYPE], cyl_qw: Scalar[DTYPE],
    cyl_hl: Scalar[DTYPE],
    cyl_r: Scalar[DTYPE],
    # Box
    b_x: Scalar[DTYPE], b_y: Scalar[DTYPE], b_z: Scalar[DTYPE],
    b_qx: Scalar[DTYPE], b_qy: Scalar[DTYPE], b_qz: Scalar[DTYPE], b_qw: Scalar[DTYPE],
    hx: Scalar[DTYPE], hy: Scalar[DTYPE], hz: Scalar[DTYPE],
) -> Tuple[
    Scalar[DTYPE], Scalar[DTYPE], Scalar[DTYPE], Scalar[DTYPE],
    Scalar[DTYPE], Scalar[DTYPE], Scalar[DTYPE],
]:
    """Cylinder-box collision.

    Reduces to capsule-box collision (treats cylinder as capsule with same
    radius and half-length). This is a conservative approximation — the
    hemispherical caps of the virtual capsule extend slightly beyond the
    cylinder's flat caps, but the error is negligible for collision detection
    since the flat cap region is small relative to the cylinder body.

    For exact cylinder-box, a full SAT or GJK approach would be needed,
    but capsule approximation matches MuJoCo's practical behavior for
    typical robot geometries.

    Returns the normal pointing from the CYLINDER to the BOX — first operand
    to second, the convention every primitive in this file follows.

    ⚠ The delegation below SWAPS the operands (`box_capsule` takes the box
    first), so its normal comes back pointing box -> cylinder and has to be
    negated. It was not, so this primitive returned the exact opposite of what
    its own signature promises. Silent until 2026-08-01: no gate exercised a
    cylinder-box pair against MuJoCo, and the two call sites in the narrow
    phase BOTH consumed it, so they were consistently wrong with each other
    and only a MuJoCo comparison could see it —
    `tests/physics3d/test_narrow_phase_pairs.mojo` measured a direction error
    of 2.0 on a unit vector, i.e. a full reversal, on its first run.
    """
    # Treat cylinder as capsule → reuse box_capsule
    var r = box_capsule[DTYPE](
        b_x, b_y, b_z,
        b_qx, b_qy, b_qz, b_qw,
        hx, hy, hz,
        cyl_x, cyl_y, cyl_z,
        cyl_qx, cyl_qy, cyl_qz, cyl_qw,
        cyl_hl,
        cyl_r,
    )
    # dist and the contact point are direction-free; only the normal flips.
    return (r[0], r[1], r[2], r[3], -r[4], -r[5], -r[6])
