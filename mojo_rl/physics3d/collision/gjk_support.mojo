"""GJK support functions for all geometry types.

A support function returns the point on a convex shape's surface that is
furthest along a given direction. These are the building blocks of GJK/EPA.

Reference: MuJoCo engine_collision_convex.c lines 162-398
"""

from std.math import sqrt, abs
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


@always_inline
def _dot3[
    DTYPE: DType
](
    ax: Scalar[DTYPE],
    ay: Scalar[DTYPE],
    az: Scalar[DTYPE],
    bx: Scalar[DTYPE],
    by: Scalar[DTYPE],
    bz: Scalar[DTYPE],
) -> Scalar[DTYPE]:
    return ax * bx + ay * by + az * bz


@always_inline
def _cross3[
    DTYPE: DType
](
    ax: Scalar[DTYPE],
    ay: Scalar[DTYPE],
    az: Scalar[DTYPE],
    bx: Scalar[DTYPE],
    by: Scalar[DTYPE],
    bz: Scalar[DTYPE],
) -> Tuple[Scalar[DTYPE], Scalar[DTYPE], Scalar[DTYPE]]:
    return (ay * bz - az * by, az * bx - ax * bz, ax * by - ay * bx)


def _closest_point_on_simplex[
    DTYPE: DType
](
    mut simplex: InlineArray[Scalar[DTYPE], 36],
    nsimplex: Int,
) -> InlineArray[
    Scalar[DTYPE], 4
]:
    """Find closest point on simplex to origin and reduce simplex.

    Returns [vx, vy, vz, new_nsimplex].
    Removes vertices that don't contribute to the closest feature.
    """
    var result = InlineArray[Scalar[DTYPE], 4](fill=Scalar[DTYPE](0))

    if nsimplex == 1:
        result[0] = simplex[0]
        result[1] = simplex[1]
        result[2] = simplex[2]
        result[3] = Scalar[DTYPE](1)
        return result

    if nsimplex == 2:
        # Line segment: project origin onto segment A→B
        var ax = simplex[0]
        var ay = simplex[1]
        var az = simplex[2]
        var bx = simplex[9]
        var by = simplex[10]
        var bz = simplex[11]
        var abx = bx - ax
        var aby = by - ay
        var abz = bz - az
        var t = -(ax * abx + ay * aby + az * abz) / (
            abx * abx + aby * aby + abz * abz + Scalar[DTYPE](1e-30)
        )
        if t <= Scalar[DTYPE](0):
            result[0] = ax
            result[1] = ay
            result[2] = az
            result[3] = Scalar[DTYPE](1)
            return result
        elif t >= Scalar[DTYPE](1):
            # Keep only B
            for k in range(9):
                simplex[k] = simplex[9 + k]
            result[0] = bx
            result[1] = by
            result[2] = bz
            result[3] = Scalar[DTYPE](1)
            return result
        else:
            result[0] = ax + t * abx
            result[1] = ay + t * aby
            result[2] = az + t * abz
            result[3] = Scalar[DTYPE](2)
            return result

    if nsimplex == 3:
        # Triangle: find closest point to origin using Voronoi regions
        var ax = simplex[0]
        var ay = simplex[1]
        var az = simplex[2]
        var bx = simplex[9]
        var by = simplex[10]
        var bz = simplex[11]
        var cx = simplex[18]
        var cy = simplex[19]
        var cz = simplex[20]

        var abx = bx - ax
        var aby = by - ay
        var abz = bz - az
        var acx = cx - ax
        var acy = cy - ay
        var acz = cz - az

        # Barycentric coordinates of origin projection onto triangle plane
        # Using the method from Real-Time Collision Detection (Ericson)
        var d1 = _dot3[DTYPE](abx, aby, abz, -ax, -ay, -az)
        var d2 = _dot3[DTYPE](acx, acy, acz, -ax, -ay, -az)
        var d3 = _dot3[DTYPE](abx, aby, abz, -bx, -by, -bz)
        var d4 = _dot3[DTYPE](acx, acy, acz, -bx, -by, -bz)
        var d5 = _dot3[DTYPE](abx, aby, abz, -cx, -cy, -cz)
        var d6 = _dot3[DTYPE](acx, acy, acz, -cx, -cy, -cz)

        # Vertex region A
        if d1 <= 0 and d2 <= 0:
            result[0] = ax
            result[1] = ay
            result[2] = az
            # Keep only vertex A
            result[3] = Scalar[DTYPE](1)
            return result

        # Vertex region B
        if d3 >= 0 and d4 <= d3:
            # Keep only vertex B
            for k in range(9):
                simplex[k] = simplex[9 + k]
            result[0] = bx
            result[1] = by
            result[2] = bz
            result[3] = Scalar[DTYPE](1)
            return result

        # Edge region AB
        var vc = d1 * d4 - d3 * d2
        if vc <= 0 and d1 >= 0 and d3 <= 0:
            var v = d1 / (d1 - d3)
            result[0] = ax + v * abx
            result[1] = ay + v * aby
            result[2] = az + v * abz
            # Keep A and B (simplex[0] and simplex[9])
            result[3] = Scalar[DTYPE](2)
            return result

        # Vertex region C
        if d6 >= 0 and d5 <= d6:
            # Keep only vertex C
            for k in range(9):
                simplex[k] = simplex[18 + k]
            result[0] = cx
            result[1] = cy
            result[2] = cz
            result[3] = Scalar[DTYPE](1)
            return result

        # Edge region AC
        var vb = d5 * d2 - d1 * d6
        if vb <= 0 and d2 >= 0 and d6 <= 0:
            var w = d2 / (d2 - d6)
            result[0] = ax + w * acx
            result[1] = ay + w * acy
            result[2] = az + w * acz
            # Keep A and C: move C to slot 1
            for k in range(9):
                simplex[9 + k] = simplex[18 + k]
            result[3] = Scalar[DTYPE](2)
            return result

        # Edge region BC
        var va = d3 * d6 - d5 * d4
        if va <= 0 and (d4 - d3) >= 0 and (d5 - d6) >= 0:
            var w = (d4 - d3) / ((d4 - d3) + (d5 - d6))
            var bcx = cx - bx
            var bcy = cy - by
            var bcz = cz - bz
            result[0] = bx + w * bcx
            result[1] = by + w * bcy
            result[2] = bz + w * bcz
            # Keep B and C: move B to slot 0, C to slot 1
            for k in range(9):
                simplex[k] = simplex[9 + k]
                simplex[9 + k] = simplex[18 + k]
            result[3] = Scalar[DTYPE](2)
            return result

        # Inside triangle: project origin onto plane
        var denom = va + vb + vc
        if denom < Scalar[DTYPE](1e-30):
            result[0] = ax
            result[1] = ay
            result[2] = az
            result[3] = Scalar[DTYPE](1)
            return result
        var v = vb / denom
        var w = vc / denom
        result[0] = ax + abx * v + acx * w
        result[1] = ay + aby * v + acy * w
        result[2] = az + abz * v + acz * w
        result[3] = Scalar[DTYPE](3)
        return result

    # nsimplex == 4: tetrahedron — check if origin is inside or find closest face
    # For each of the 4 faces, check if origin is on the outside
    # Face normals point outward (away from the opposite vertex)
    # Face ABC: normal = (B-A)×(C-A), check sign of dot(normal, D-A)
    # If dot(normal, -A) has OPPOSITE sign to dot(normal, D-A), origin is outside this face

    # Face indices: (A,B,C, opposite D), (A,C,D, opposite B), (A,D,B, opposite C), (B,D,C, opposite A)
    var face_v = InlineArray[Int, 16](fill=0)
    face_v[0] = 0
    face_v[1] = 9
    face_v[2] = 18
    face_v[3] = 27  # ABC, opp D
    face_v[4] = 0
    face_v[5] = 18
    face_v[6] = 27
    face_v[7] = 9   # ACD, opp B
    face_v[8] = 0
    face_v[9] = 27
    face_v[10] = 9
    face_v[11] = 18  # ADB, opp C
    face_v[12] = 9
    face_v[13] = 27
    face_v[14] = 18
    face_v[15] = 0   # BDC, opp A

    var best_dist_sq: Scalar[DTYPE] = 1e30
    var best_face = -1
    var best_vx: Scalar[DTYPE] = 0
    var best_vy: Scalar[DTYPE] = 0
    var best_vz: Scalar[DTYPE] = 0
    var any_usable_face = False

    # Scale for the degeneracy tests below: the largest vertex magnitude.
    var scale: Scalar[DTYPE] = 0
    for q in range(4):
        var m = (
            simplex[q * 9 + 0] * simplex[q * 9 + 0]
            + simplex[q * 9 + 1] * simplex[q * 9 + 1]
            + simplex[q * 9 + 2] * simplex[q * 9 + 2]
        )
        if m > scale:
            scale = m
    scale = sqrt(scale)

    for f in range(4):
        var i0 = face_v[f * 4 + 0]
        var i1 = face_v[f * 4 + 1]
        var i2 = face_v[f * 4 + 2]
        var io = face_v[f * 4 + 3]  # opposite vertex

        var f0x = simplex[i0]
        var f0y = simplex[i0 + 1]
        var f0z = simplex[i0 + 2]
        var f1x = simplex[i1]
        var f1y = simplex[i1 + 1]
        var f1z = simplex[i1 + 2]
        var f2x = simplex[i2]
        var f2y = simplex[i2 + 1]
        var f2z = simplex[i2 + 2]
        var fox = simplex[io]
        var foy = simplex[io + 1]
        var foz = simplex[io + 2]

        # Face normal
        var e1x = f1x - f0x
        var e1y = f1y - f0y
        var e1z = f1z - f0z
        var e2x = f2x - f0x
        var e2y = f2y - f0y
        var e2z = f2z - f0z
        var face_n = _cross3[DTYPE](e1x, e1y, e1z, e2x, e2y, e2z)

        # Sign check: is origin on the same side as the opposite vertex?
        # Both dots are measured against the UNNORMALIZED normal, so divide by
        # |n| to get true heights before testing them against a scale-relative
        # epsilon — otherwise the test's sensitivity rides on the face's area.
        var n_len = sqrt(
            face_n[0] * face_n[0]
            + face_n[1] * face_n[1]
            + face_n[2] * face_n[2]
        )
        if n_len <= Scalar[DTYPE](1e-30):
            # Repeated or collinear vertices — this face says nothing.
            continue
        any_usable_face = True
        var dot_opp = _dot3[DTYPE](face_n[0], face_n[1], face_n[2], fox - f0x, foy - f0y, foz - f0z)
        var dot_origin = _dot3[DTYPE](face_n[0], face_n[1], face_n[2], -f0x, -f0y, -f0z)
        var h_opp = dot_opp / n_len
        var h_origin = dot_origin / n_len
        var flat = Scalar[DTYPE](1e-6) * scale

        # A tetrahedron whose opposite vertex sits IN this face's plane has no
        # interior, so nothing can be enclosed by it. The old test was the bare
        # product `dot_opp * dot_origin < 0`, which reads h_opp == 0 as "not
        # outside" — and a FLAT simplex has h_opp == 0 on all four faces, so no
        # face is ever flagged and the routine reports the origin as ENCLOSED.
        # That is not a rare tie: GJK converges onto a planar facet whenever the
        # closest feature is one (a hull face parallel to a box/cylinder cap),
        # and the caller reads "enclosed" as penetration — a phantom contact
        # between geoms that are centimetres apart, with a depth invented by
        # the EPA fallback.
        var outside = False
        if h_opp > flat:
            outside = h_origin < 0
        elif h_opp < -flat:
            outside = h_origin > 0
        else:
            outside = abs(h_origin) > flat

        if outside:
            # Origin is OUTSIDE this face — closest point is on this triangle
            # Project origin onto this face plane
            var n_dot_n = face_n[0] * face_n[0] + face_n[1] * face_n[1] + face_n[2] * face_n[2]
            if n_dot_n > Scalar[DTYPE](1e-30):
                # Closest point on the face PLANE to the origin is
                # (n.f0 / |n|^2) * n — the same outward convention the
                # nsimplex<=3 branches use (v = origin -> simplex). It used to
                # carry a leading minus, which handed GJK a search direction
                # pointing AWAY from the shape and flipped the reported
                # contact normal on the separated path.
                var d = _dot3[DTYPE](f0x, f0y, f0z, face_n[0], face_n[1], face_n[2]) / n_dot_n
                var proj_x = d * face_n[0]
                var proj_y = d * face_n[1]
                var proj_z = d * face_n[2]
                var d_sq = proj_x * proj_x + proj_y * proj_y + proj_z * proj_z
                if d_sq < best_dist_sq:
                    best_dist_sq = d_sq
                    best_face = f
                    best_vx = proj_x
                    best_vy = proj_y
                    best_vz = proj_z

    if best_face >= 0:
        # Origin is outside at least one face — reduce to that triangle.
        # Stage through a temporary: destination slots OVERLAP the sources.
        # Face ADB is (i0,i1,i2) = (0,27,9), so an in-place copy overwrites
        # slot 1 with D and then reads that slot back as "B", retaining
        # {A,D,D} — a degenerate simplex that stalls the next GJK iteration
        # and makes a penetrating pair report as separated.
        var i0 = face_v[best_face * 4 + 0]
        var i1 = face_v[best_face * 4 + 1]
        var i2 = face_v[best_face * 4 + 2]
        var keep = InlineArray[Scalar[DTYPE], 27](fill=Scalar[DTYPE](0))
        for k in range(9):
            keep[k] = simplex[i0 + k]
            keep[9 + k] = simplex[i1 + k]
            keep[18 + k] = simplex[i2 + k]
        for k in range(27):
            simplex[k] = keep[k]
        result[0] = best_vx
        result[1] = best_vy
        result[2] = best_vz
        result[3] = Scalar[DTYPE](3)
        return result

    if not any_usable_face:
        # Every face was degenerate — the four points are collinear or
        # coincident, so there is no tetrahedron to be inside of. Keep the
        # vertex nearest the origin and let the next iteration rebuild; do NOT
        # fall through to the enclosure verdict, which would be unfounded.
        var near = 0
        var near_sq: Scalar[DTYPE] = 1e30
        for q in range(4):
            var m = (
                simplex[q * 9 + 0] * simplex[q * 9 + 0]
                + simplex[q * 9 + 1] * simplex[q * 9 + 1]
                + simplex[q * 9 + 2] * simplex[q * 9 + 2]
            )
            if m < near_sq:
                near_sq = m
                near = q
        var keep1 = InlineArray[Scalar[DTYPE], 9](fill=Scalar[DTYPE](0))
        for k in range(9):
            keep1[k] = simplex[near * 9 + k]
        for k in range(9):
            simplex[k] = keep1[k]
        result[0] = simplex[0]
        result[1] = simplex[1]
        result[2] = simplex[2]
        result[3] = Scalar[DTYPE](1)
        return result

    # Origin is inside all faces — truly inside tetrahedron
    result[0] = Scalar[DTYPE](0)
    result[1] = Scalar[DTYPE](0)
    result[2] = Scalar[DTYPE](0)
    result[3] = Scalar[DTYPE](4)
    return result
