"""GJK (Gilbert-Johnson-Keerthi) distance algorithm + EPA penetration depth.

Computes the minimum distance between two convex shapes (GJK), and if they
overlap, computes the penetration depth and contact normal (EPA).

Reference: MuJoCo engine_collision_gjk.c (Montanari et al., ToG 2017)
"""

from std.math import sqrt, abs

from .gjk_support import (
    support_sphere,
    support_capsule,
    support_box,
    support_cylinder,
    support_mesh,
)
from ..constants import (
    GEOM_SPHERE,
    GEOM_CAPSULE,
    GEOM_BOX,
    GEOM_CYLINDER,
    GEOM_MESH,
)

# GJK parameters
comptime GJK_MAX_ITERATIONS: Int = 100
comptime GJK_TOLERANCE: Float64 = 1e-10

# EPA parameters
comptime EPA_MAX_ITERATIONS: Int = 64
comptime EPA_MAX_VERTS: Int = 69  # 5 + EPA_MAX_ITERATIONS
comptime EPA_MAX_FACES: Int = 384  # 6 * EPA_MAX_ITERATIONS
comptime EPA_TOLERANCE: Float64 = 1e-8


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


def _support[
    DTYPE: DType
](
    geom_type: Int,
    pos_x: Scalar[DTYPE],
    pos_y: Scalar[DTYPE],
    pos_z: Scalar[DTYPE],
    qx: Scalar[DTYPE],
    qy: Scalar[DTYPE],
    qz: Scalar[DTYPE],
    qw: Scalar[DTYPE],
    radius: Scalar[DTYPE],
    half_length: Scalar[DTYPE],
    half_x: Scalar[DTYPE],
    half_y: Scalar[DTYPE],
    half_z: Scalar[DTYPE],
    mesh_verts: List[Scalar[DTYPE]],
    mesh_vert_offset: Int,
    mesh_num_verts: Int,
    dir_x: Scalar[DTYPE],
    dir_y: Scalar[DTYPE],
    dir_z: Scalar[DTYPE],
) -> InlineArray[Scalar[DTYPE], 3]:
    """Unified support function dispatcher for all geom types."""
    if geom_type == GEOM_SPHERE:
        return support_sphere[DTYPE](
            dir_x, dir_y, dir_z, pos_x, pos_y, pos_z, radius
        )
    elif geom_type == GEOM_CAPSULE:
        return support_capsule[DTYPE](
            dir_x,
            dir_y,
            dir_z,
            pos_x,
            pos_y,
            pos_z,
            qx,
            qy,
            qz,
            qw,
            radius,
            half_length,
        )
    elif geom_type == GEOM_BOX:
        return support_box[DTYPE](
            dir_x,
            dir_y,
            dir_z,
            pos_x,
            pos_y,
            pos_z,
            qx,
            qy,
            qz,
            qw,
            half_x,
            half_y,
            half_z,
        )
    elif geom_type == GEOM_CYLINDER:
        return support_cylinder[DTYPE](
            dir_x,
            dir_y,
            dir_z,
            pos_x,
            pos_y,
            pos_z,
            qx,
            qy,
            qz,
            qw,
            radius,
            half_length,
        )
    elif geom_type == GEOM_MESH:
        return support_mesh[DTYPE](
            dir_x,
            dir_y,
            dir_z,
            pos_x,
            pos_y,
            pos_z,
            qx,
            qy,
            qz,
            qw,
            mesh_verts,
            mesh_vert_offset,
            mesh_num_verts,
        )
    # Fallback: point support (center only)
    var result = InlineArray[Scalar[DTYPE], 3](fill=Scalar[DTYPE](0))
    result[0] = pos_x
    result[1] = pos_y
    result[2] = pos_z
    return result


def _minkowski_support[
    DTYPE: DType
](
    # Geom 1
    type1: Int,
    p1x: Scalar[DTYPE],
    p1y: Scalar[DTYPE],
    p1z: Scalar[DTYPE],
    q1x: Scalar[DTYPE],
    q1y: Scalar[DTYPE],
    q1z: Scalar[DTYPE],
    q1w: Scalar[DTYPE],
    r1: Scalar[DTYPE],
    hl1: Scalar[DTYPE],
    hx1: Scalar[DTYPE],
    hy1: Scalar[DTYPE],
    hz1: Scalar[DTYPE],
    mv1: List[Scalar[DTYPE]],
    mvo1: Int,
    mnv1: Int,
    # Geom 2
    type2: Int,
    p2x: Scalar[DTYPE],
    p2y: Scalar[DTYPE],
    p2z: Scalar[DTYPE],
    q2x: Scalar[DTYPE],
    q2y: Scalar[DTYPE],
    q2z: Scalar[DTYPE],
    q2w: Scalar[DTYPE],
    r2: Scalar[DTYPE],
    hl2: Scalar[DTYPE],
    hx2: Scalar[DTYPE],
    hy2: Scalar[DTYPE],
    hz2: Scalar[DTYPE],
    mv2: List[Scalar[DTYPE]],
    mvo2: Int,
    mnv2: Int,
    # Direction
    dir_x: Scalar[DTYPE],
    dir_y: Scalar[DTYPE],
    dir_z: Scalar[DTYPE],
) -> Tuple[
    Scalar[DTYPE],
    Scalar[DTYPE],
    Scalar[DTYPE],  # Minkowski diff point
    Scalar[DTYPE],
    Scalar[DTYPE],
    Scalar[DTYPE],  # witness on obj1
    Scalar[DTYPE],
    Scalar[DTYPE],
    Scalar[DTYPE],  # witness on obj2
]:
    """Compute support point on Minkowski difference: sup1(dir) - sup2(-dir)."""
    var s1 = _support[DTYPE](
        type1,
        p1x,
        p1y,
        p1z,
        q1x,
        q1y,
        q1z,
        q1w,
        r1,
        hl1,
        hx1,
        hy1,
        hz1,
        mv1,
        mvo1,
        mnv1,
        dir_x,
        dir_y,
        dir_z,
    )
    var s2 = _support[DTYPE](
        type2,
        p2x,
        p2y,
        p2z,
        q2x,
        q2y,
        q2z,
        q2w,
        r2,
        hl2,
        hx2,
        hy2,
        hz2,
        mv2,
        mvo2,
        mnv2,
        -dir_x,
        -dir_y,
        -dir_z,
    )
    return (
        s1[0] - s2[0],
        s1[1] - s2[1],
        s1[2] - s2[2],
        s1[0],
        s1[1],
        s1[2],
        s2[0],
        s2[1],
        s2[2],
    )


def gjk_epa[
    DTYPE: DType
](
    # Geom 1
    type1: Int,
    p1x: Scalar[DTYPE],
    p1y: Scalar[DTYPE],
    p1z: Scalar[DTYPE],
    q1x: Scalar[DTYPE],
    q1y: Scalar[DTYPE],
    q1z: Scalar[DTYPE],
    q1w: Scalar[DTYPE],
    r1: Scalar[DTYPE],
    hl1: Scalar[DTYPE],
    hx1: Scalar[DTYPE],
    hy1: Scalar[DTYPE],
    hz1: Scalar[DTYPE],
    mv1: List[Scalar[DTYPE]],
    mvo1: Int,
    mnv1: Int,
    # Geom 2
    type2: Int,
    p2x: Scalar[DTYPE],
    p2y: Scalar[DTYPE],
    p2z: Scalar[DTYPE],
    q2x: Scalar[DTYPE],
    q2y: Scalar[DTYPE],
    q2z: Scalar[DTYPE],
    q2w: Scalar[DTYPE],
    r2: Scalar[DTYPE],
    hl2: Scalar[DTYPE],
    hx2: Scalar[DTYPE],
    hy2: Scalar[DTYPE],
    hz2: Scalar[DTYPE],
    mv2: List[Scalar[DTYPE]],
    mvo2: Int,
    mnv2: Int,
) -> Tuple[
    Scalar[DTYPE],  # distance (negative = penetration depth)
    Scalar[DTYPE],
    Scalar[DTYPE],
    Scalar[DTYPE],  # contact point
    Scalar[DTYPE],
    Scalar[DTYPE],
    Scalar[DTYPE],  # contact normal (from 2 toward 1)
]:
    """GJK distance + EPA penetration depth between two convex shapes.

    Returns:
        (dist, cx, cy, cz, nx, ny, nz) where:
        - dist > 0: separated, dist is minimum distance
        - dist <= 0: overlapping, |dist| is penetration depth
        - (cx, cy, cz): contact point (midpoint of witness points)
        - (nx, ny, nz): contact normal pointing from obj2 toward obj1.
    """
    # ===== GJK Phase: find minimum distance =====

    # Simplex vertices: each stores Minkowski diff + witness points (9 floats)
    # simplex[i] = [vx, vy, vz, w1x, w1y, w1z, w2x, w2y, w2z]
    var simplex = InlineArray[Scalar[DTYPE], 36](
        fill=Scalar[DTYPE](0)
    )  # 4 vertices × 9
    var nsimplex: Int

    # Initial direction: center1 - center2
    var dx = p1x - p2x
    var dy = p1y - p2y
    var dz = p1z - p2z
    var dlen = sqrt(dx * dx + dy * dy + dz * dz)
    if dlen < Scalar[DTYPE](1e-12):
        dx = Scalar[DTYPE](1)
        dy = Scalar[DTYPE](0)
        dz = Scalar[DTYPE](0)
        dlen = Scalar[DTYPE](1)
    dx /= dlen
    dy /= dlen
    dz /= dlen

    # First support point
    var s = _minkowski_support[DTYPE](
        type1,
        p1x,
        p1y,
        p1z,
        q1x,
        q1y,
        q1z,
        q1w,
        r1,
        hl1,
        hx1,
        hy1,
        hz1,
        mv1,
        mvo1,
        mnv1,
        type2,
        p2x,
        p2y,
        p2z,
        q2x,
        q2y,
        q2z,
        q2w,
        r2,
        hl2,
        hx2,
        hy2,
        hz2,
        mv2,
        mvo2,
        mnv2,
        dx,
        dy,
        dz,
    )
    simplex[0] = s[0]
    simplex[1] = s[1]
    simplex[2] = s[2]
    simplex[3] = s[3]
    simplex[4] = s[4]
    simplex[5] = s[5]
    simplex[6] = s[6]
    simplex[7] = s[7]
    simplex[8] = s[8]
    nsimplex = 1

    # Closest point to origin (initially the single vertex)
    var vx = s[0]
    var vy = s[1]
    var vz = s[2]

    for _ in range(GJK_MAX_ITERATIONS):
        var v_dot_v = vx * vx + vy * vy + vz * vz
        if v_dot_v < Scalar[DTYPE](GJK_TOLERANCE):
            break

        # New search direction: toward origin from closest point
        var inv_vlen = Scalar[DTYPE](1) / sqrt(v_dot_v)
        var ndx = -vx * inv_vlen
        var ndy = -vy * inv_vlen
        var ndz = -vz * inv_vlen

        # Get new support point
        var sn = _minkowski_support[DTYPE](
            type1,
            p1x,
            p1y,
            p1z,
            q1x,
            q1y,
            q1z,
            q1w,
            r1,
            hl1,
            hx1,
            hy1,
            hz1,
            mv1,
            mvo1,
            mnv1,
            type2,
            p2x,
            p2y,
            p2z,
            q2x,
            q2y,
            q2z,
            q2w,
            r2,
            hl2,
            hx2,
            hy2,
            hz2,
            mv2,
            mvo2,
            mnv2,
            ndx,
            ndy,
            ndz,
        )

        # Frank-Wolfe duality gap: if no progress, converged
        var w_dot = sn[0] * ndx + sn[1] * ndy + sn[2] * ndz
        var v_dot = vx * ndx + vy * ndy + vz * ndz
        if w_dot - v_dot < Scalar[DTYPE](GJK_TOLERANCE):
            break

        # Add new vertex to simplex
        var si = nsimplex * 9
        simplex[si + 0] = sn[0]
        simplex[si + 1] = sn[1]
        simplex[si + 2] = sn[2]
        simplex[si + 3] = sn[3]
        simplex[si + 4] = sn[4]
        simplex[si + 5] = sn[5]
        simplex[si + 6] = sn[6]
        simplex[si + 7] = sn[7]
        simplex[si + 8] = sn[8]
        nsimplex += 1

        # Find closest point on simplex to origin + reduce simplex
        var cp = _closest_point_on_simplex[DTYPE](simplex, nsimplex)
        vx = cp[0]
        vy = cp[1]
        vz = cp[2]
        nsimplex = Int(cp[3])

        if nsimplex == 4:
            # Origin is inside tetrahedron → overlap
            vx = Scalar[DTYPE](0)
            vy = Scalar[DTYPE](0)
            vz = Scalar[DTYPE](0)
            break

    var dist = sqrt(vx * vx + vy * vy + vz * vz)

    if dist > Scalar[DTYPE](GJK_TOLERANCE):
        # ===== Separated: compute witness points from simplex =====
        var w1x: Scalar[DTYPE] = 0
        var w1y: Scalar[DTYPE] = 0
        var w1z: Scalar[DTYPE] = 0
        var w2x: Scalar[DTYPE] = 0
        var w2y: Scalar[DTYPE] = 0
        var w2z: Scalar[DTYPE] = 0
        # Use barycentrics from closest point (stored in simplex reduction)
        # For simplicity, use centroid of remaining simplex witness points
        for i in range(nsimplex):
            w1x += simplex[i * 9 + 3]
            w1y += simplex[i * 9 + 4]
            w1z += simplex[i * 9 + 5]
            w2x += simplex[i * 9 + 6]
            w2y += simplex[i * 9 + 7]
            w2z += simplex[i * 9 + 8]
        if nsimplex > 0:
            var inv_n = Scalar[DTYPE](1) / Scalar[DTYPE](nsimplex)
            w1x *= inv_n
            w1y *= inv_n
            w1z *= inv_n
            w2x *= inv_n
            w2y *= inv_n
            w2z *= inv_n

        var cx = (w1x + w2x) * Scalar[DTYPE](0.5)
        var cy = (w1y + w2y) * Scalar[DTYPE](0.5)
        var cz = (w1z + w2z) * Scalar[DTYPE](0.5)
        # Normal from obj2 toward obj1
        var nx = vx / dist
        var ny = vy / dist
        var nz = vz / dist
        return (dist, cx, cy, cz, nx, ny, nz)

    # ===== EPA Phase: compute penetration depth =====
    var epa = _epa[DTYPE](
        type1,
        p1x,
        p1y,
        p1z,
        q1x,
        q1y,
        q1z,
        q1w,
        r1,
        hl1,
        hx1,
        hy1,
        hz1,
        mv1,
        mvo1,
        mnv1,
        type2,
        p2x,
        p2y,
        p2z,
        q2x,
        q2y,
        q2z,
        q2w,
        r2,
        hl2,
        hx2,
        hy2,
        hz2,
        mv2,
        mvo2,
        mnv2,
        simplex,
        nsimplex,
    )
    return epa


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
        var dot_opp = _dot3[DTYPE](face_n[0], face_n[1], face_n[2], fox - f0x, foy - f0y, foz - f0z)
        var dot_origin = _dot3[DTYPE](face_n[0], face_n[1], face_n[2], -f0x, -f0y, -f0z)

        if dot_opp * dot_origin < 0:
            # Origin is OUTSIDE this face — closest point is on this triangle
            # Project origin onto this face plane
            var n_dot_n = face_n[0] * face_n[0] + face_n[1] * face_n[1] + face_n[2] * face_n[2]
            if n_dot_n > Scalar[DTYPE](1e-30):
                var d = _dot3[DTYPE](f0x, f0y, f0z, face_n[0], face_n[1], face_n[2]) / n_dot_n
                var proj_x = -d * face_n[0]
                var proj_y = -d * face_n[1]
                var proj_z = -d * face_n[2]
                var d_sq = proj_x * proj_x + proj_y * proj_y + proj_z * proj_z
                if d_sq < best_dist_sq:
                    best_dist_sq = d_sq
                    best_face = f
                    best_vx = proj_x
                    best_vy = proj_y
                    best_vz = proj_z

    if best_face >= 0:
        # Origin is outside at least one face — reduce to that triangle
        var i0 = face_v[best_face * 4 + 0]
        var i1 = face_v[best_face * 4 + 1]
        var i2 = face_v[best_face * 4 + 2]
        for k in range(9):
            simplex[k] = simplex[i0 + k]
        for k in range(9):
            simplex[9 + k] = simplex[i1 + k]
        for k in range(9):
            simplex[18 + k] = simplex[i2 + k]
        result[0] = best_vx
        result[1] = best_vy
        result[2] = best_vz
        result[3] = Scalar[DTYPE](3)
        return result

    # Origin is inside all faces — truly inside tetrahedron
    result[0] = Scalar[DTYPE](0)
    result[1] = Scalar[DTYPE](0)
    result[2] = Scalar[DTYPE](0)
    result[3] = Scalar[DTYPE](4)
    return result


def _epa[
    DTYPE: DType
](
    # Geom 1
    type1: Int,
    p1x: Scalar[DTYPE],
    p1y: Scalar[DTYPE],
    p1z: Scalar[DTYPE],
    q1x: Scalar[DTYPE],
    q1y: Scalar[DTYPE],
    q1z: Scalar[DTYPE],
    q1w: Scalar[DTYPE],
    r1: Scalar[DTYPE],
    hl1: Scalar[DTYPE],
    hx1: Scalar[DTYPE],
    hy1: Scalar[DTYPE],
    hz1: Scalar[DTYPE],
    mv1: List[Scalar[DTYPE]],
    mvo1: Int,
    mnv1: Int,
    # Geom 2
    type2: Int,
    p2x: Scalar[DTYPE],
    p2y: Scalar[DTYPE],
    p2z: Scalar[DTYPE],
    q2x: Scalar[DTYPE],
    q2y: Scalar[DTYPE],
    q2z: Scalar[DTYPE],
    q2w: Scalar[DTYPE],
    r2: Scalar[DTYPE],
    hl2: Scalar[DTYPE],
    hx2: Scalar[DTYPE],
    hy2: Scalar[DTYPE],
    hz2: Scalar[DTYPE],
    mv2: List[Scalar[DTYPE]],
    mvo2: Int,
    mnv2: Int,
    # GJK terminal simplex
    simplex: InlineArray[Scalar[DTYPE], 36],
    nsimplex: Int,
) -> Tuple[
    Scalar[DTYPE],  # penetration depth (negative)
    Scalar[DTYPE],
    Scalar[DTYPE],
    Scalar[DTYPE],  # contact point
    Scalar[DTYPE],
    Scalar[DTYPE],
    Scalar[DTYPE],  # contact normal
]:
    """EPA: Expanding Polytope Algorithm for penetration depth.

    Called when GJK detects overlap (origin inside Minkowski difference).
    Expands the simplex into a polytope, finding the closest face to the
    origin to determine penetration depth and normal.
    """
    # EPA polytope vertices: [vx, vy, vz, w1x, w1y, w1z, w2x, w2y, w2z] × EPA_MAX_VERTS
    var verts = List[Scalar[DTYPE]](capacity=EPA_MAX_VERTS * 9)
    for _ in range(EPA_MAX_VERTS * 9):
        verts.append(Scalar[DTYPE](0))
    var nverts = 0

    # Copy GJK simplex vertices
    for i in range(nsimplex):
        for k in range(9):
            verts[nverts * 9 + k] = simplex[i * 9 + k]
        nverts += 1

    # If we have fewer than 4 vertices, we can't form a tetrahedron for EPA.
    # Use a fallback: return shallow penetration along center-to-center direction.
    if nverts < 4:
        var fallback_nx = p1x - p2x
        var fallback_ny = p1y - p2y
        var fallback_nz = p1z - p2z
        var fallback_len = sqrt(
            fallback_nx * fallback_nx
            + fallback_ny * fallback_ny
            + fallback_nz * fallback_nz
        )
        if fallback_len < Scalar[DTYPE](1e-10):
            fallback_nx = Scalar[DTYPE](0)
            fallback_ny = Scalar[DTYPE](0)
            fallback_nz = Scalar[DTYPE](1)
        else:
            fallback_nx /= fallback_len
            fallback_ny /= fallback_len
            fallback_nz /= fallback_len
        var cx = (p1x + p2x) * Scalar[DTYPE](0.5)
        var cy = (p1y + p2y) * Scalar[DTYPE](0.5)
        var cz = (p1z + p2z) * Scalar[DTYPE](0.5)
        return (
            Scalar[DTYPE](-0.001),
            cx,
            cy,
            cz,
            fallback_nx,
            fallback_ny,
            fallback_nz,
        )

    # Faces: [v0, v1, v2] × EPA_MAX_FACES (indices into verts)
    var faces = List[Int](capacity=EPA_MAX_FACES * 3)
    for _ in range(EPA_MAX_FACES * 3):
        faces.append(0)
    var nfaces = 0

    # Initialize polytope with tetrahedron (4 faces)
    # Ensure consistent winding: normals point outward (away from centroid)
    var cx = Scalar[DTYPE](0)
    var cy = Scalar[DTYPE](0)
    var cz = Scalar[DTYPE](0)
    for i in range(4):
        cx += verts[i * 9 + 0]
        cy += verts[i * 9 + 1]
        cz += verts[i * 9 + 2]
    cx *= Scalar[DTYPE](0.25)
    cy *= Scalar[DTYPE](0.25)
    cz *= Scalar[DTYPE](0.25)

    # 4 faces of tetrahedron: (0,1,2), (0,3,1), (0,2,3), (1,3,2)
    var face_indices = InlineArray[Int, 12](fill=0)
    face_indices[0] = 0
    face_indices[1] = 1
    face_indices[2] = 2
    face_indices[3] = 0
    face_indices[4] = 3
    face_indices[5] = 1
    face_indices[6] = 0
    face_indices[7] = 2
    face_indices[8] = 3
    face_indices[9] = 1
    face_indices[10] = 3
    face_indices[11] = 2

    for f in range(4):
        var i0 = face_indices[f * 3 + 0]
        var i1 = face_indices[f * 3 + 1]
        var i2 = face_indices[f * 3 + 2]
        # Check winding: face normal should point away from centroid
        var e1x = verts[i1 * 9] - verts[i0 * 9]
        var e1y = verts[i1 * 9 + 1] - verts[i0 * 9 + 1]
        var e1z = verts[i1 * 9 + 2] - verts[i0 * 9 + 2]
        var e2x = verts[i2 * 9] - verts[i0 * 9]
        var e2y = verts[i2 * 9 + 1] - verts[i0 * 9 + 1]
        var e2z = verts[i2 * 9 + 2] - verts[i0 * 9 + 2]
        var face_n = _cross3[DTYPE](e1x, e1y, e1z, e2x, e2y, e2z)
        var to_center_x = cx - verts[i0 * 9]
        var to_center_y = cy - verts[i0 * 9 + 1]
        var to_center_z = cz - verts[i0 * 9 + 2]
        if (
            _dot3[DTYPE](
                face_n[0],
                face_n[1],
                face_n[2],
                to_center_x,
                to_center_y,
                to_center_z,
            )
            > 0
        ):
            # Normal points toward centroid, flip winding
            faces[nfaces * 3 + 0] = i0
            faces[nfaces * 3 + 1] = i2
            faces[nfaces * 3 + 2] = i1
        else:
            faces[nfaces * 3 + 0] = i0
            faces[nfaces * 3 + 1] = i1
            faces[nfaces * 3 + 2] = i2
        nfaces += 1

    # EPA main loop
    var best_dist: Scalar[DTYPE] = 1e30
    var best_nx: Scalar[DTYPE] = 0
    var best_ny: Scalar[DTYPE] = 0
    var best_nz: Scalar[DTYPE] = 1
    var best_face = 0

    for _ in range(EPA_MAX_ITERATIONS):
        if nfaces == 0:
            break

        # Find face closest to origin
        best_dist = Scalar[DTYPE](1e30)
        best_face = 0
        for f in range(nfaces):
            var i0 = faces[f * 3 + 0]
            var i1 = faces[f * 3 + 1]
            var i2 = faces[f * 3 + 2]
            var e1x = verts[i1 * 9] - verts[i0 * 9]
            var e1y = verts[i1 * 9 + 1] - verts[i0 * 9 + 1]
            var e1z = verts[i1 * 9 + 2] - verts[i0 * 9 + 2]
            var e2x = verts[i2 * 9] - verts[i0 * 9]
            var e2y = verts[i2 * 9 + 1] - verts[i0 * 9 + 1]
            var e2z = verts[i2 * 9 + 2] - verts[i0 * 9 + 2]
            var face_n = _cross3[DTYPE](e1x, e1y, e1z, e2x, e2y, e2z)
            var face_n_len = sqrt(
                face_n[0] * face_n[0]
                + face_n[1] * face_n[1]
                + face_n[2] * face_n[2]
            )
            if face_n_len < Scalar[DTYPE](1e-15):
                continue
            var fnx = face_n[0] / face_n_len
            var fny = face_n[1] / face_n_len
            var fnz = face_n[2] / face_n_len
            var d = abs(
                _dot3[DTYPE](
                    verts[i0 * 9],
                    verts[i0 * 9 + 1],
                    verts[i0 * 9 + 2],
                    fnx,
                    fny,
                    fnz,
                )
            )
            if d < best_dist:
                best_dist = d
                best_nx = fnx
                best_ny = fny
                best_nz = fnz
                best_face = f

        # Get new support point along best face normal
        var sn = _minkowski_support[DTYPE](
            type1,
            p1x,
            p1y,
            p1z,
            q1x,
            q1y,
            q1z,
            q1w,
            r1,
            hl1,
            hx1,
            hy1,
            hz1,
            mv1,
            mvo1,
            mnv1,
            type2,
            p2x,
            p2y,
            p2z,
            q2x,
            q2y,
            q2z,
            q2w,
            r2,
            hl2,
            hx2,
            hy2,
            hz2,
            mv2,
            mvo2,
            mnv2,
            best_nx,
            best_ny,
            best_nz,
        )

        # Check if new point expands the polytope significantly
        var new_dist = _dot3[DTYPE](
            sn[0], sn[1], sn[2], best_nx, best_ny, best_nz
        )
        if new_dist - best_dist < Scalar[DTYPE](EPA_TOLERANCE):
            break  # Converged

        if nverts >= EPA_MAX_VERTS:
            break

        # Add new vertex
        verts[nverts * 9 + 0] = sn[0]
        verts[nverts * 9 + 1] = sn[1]
        verts[nverts * 9 + 2] = sn[2]
        verts[nverts * 9 + 3] = sn[3]
        verts[nverts * 9 + 4] = sn[4]
        verts[nverts * 9 + 5] = sn[5]
        verts[nverts * 9 + 6] = sn[6]
        verts[nverts * 9 + 7] = sn[7]
        verts[nverts * 9 + 8] = sn[8]
        var new_vert = nverts
        nverts += 1

        # Remove faces visible from new point and add new faces
        # Simple approach: remove all faces whose normal has positive dot with (new_vert - face_vert)
        var kept_faces = List[Int](capacity=EPA_MAX_FACES * 3)
        var horizon_edges = List[Int](
            capacity=EPA_MAX_FACES * 2
        )  # pairs of vertex indices

        for f in range(nfaces):
            var i0 = faces[f * 3 + 0]
            var i1 = faces[f * 3 + 1]
            var i2 = faces[f * 3 + 2]
            var e1x = verts[i1 * 9] - verts[i0 * 9]
            var e1y = verts[i1 * 9 + 1] - verts[i0 * 9 + 1]
            var e1z = verts[i1 * 9 + 2] - verts[i0 * 9 + 2]
            var e2x = verts[i2 * 9] - verts[i0 * 9]
            var e2y = verts[i2 * 9 + 1] - verts[i0 * 9 + 1]
            var e2z = verts[i2 * 9 + 2] - verts[i0 * 9 + 2]
            var face_n = _cross3[DTYPE](e1x, e1y, e1z, e2x, e2y, e2z)
            var to_new_x = verts[new_vert * 9] - verts[i0 * 9]
            var to_new_y = verts[new_vert * 9 + 1] - verts[i0 * 9 + 1]
            var to_new_z = verts[new_vert * 9 + 2] - verts[i0 * 9 + 2]

            if (
                _dot3[DTYPE](
                    face_n[0],
                    face_n[1],
                    face_n[2],
                    to_new_x,
                    to_new_y,
                    to_new_z,
                )
                > 0
            ):
                # Face visible from new point — remove it, record horizon edges
                # Each edge of this face could be a horizon edge
                var edges = InlineArray[Int, 6](fill=0)
                edges[0] = i0
                edges[1] = i1
                edges[2] = i1
                edges[3] = i2
                edges[4] = i2
                edges[5] = i0
                for e in range(3):
                    horizon_edges.append(edges[e * 2])
                    horizon_edges.append(edges[e * 2 + 1])
            else:
                kept_faces.append(i0)
                kept_faces.append(i1)
                kept_faces.append(i2)

        # Rebuild faces: kept + new faces from horizon edges to new vertex
        nfaces = len(kept_faces) // 3
        for i in range(len(kept_faces)):
            faces[i] = kept_faces[i]

        # Find unique horizon edges (edges that appear exactly once)
        var num_horizon = len(horizon_edges) // 2
        for e in range(num_horizon):
            var ea = horizon_edges[e * 2]
            var eb = horizon_edges[e * 2 + 1]
            # Check if reverse edge also exists (shared between two visible faces)
            var is_shared = False
            for e2 in range(num_horizon):
                if (
                    e2 != e
                    and horizon_edges[e2 * 2] == eb
                    and horizon_edges[e2 * 2 + 1] == ea
                ):
                    is_shared = True
                    break
            if not is_shared and nfaces < EPA_MAX_FACES:
                # This is a true horizon edge — add new face
                faces[nfaces * 3 + 0] = ea
                faces[nfaces * 3 + 1] = eb
                faces[nfaces * 3 + 2] = new_vert
                nfaces += 1

    # Extract contact info from best face
    if nfaces > 0 and best_face < nfaces:
        var i0 = faces[best_face * 3 + 0]
        # Use witness points from best face vertices (average)
        var w1x = verts[i0 * 9 + 3]
        var w1y = verts[i0 * 9 + 4]
        var w1z = verts[i0 * 9 + 5]
        var w2x = verts[i0 * 9 + 6]
        var w2y = verts[i0 * 9 + 7]
        var w2z = verts[i0 * 9 + 8]
        var contact_x = (w1x + w2x) * Scalar[DTYPE](0.5)
        var contact_y = (w1y + w2y) * Scalar[DTYPE](0.5)
        var contact_z = (w1z + w2z) * Scalar[DTYPE](0.5)
        return (
            -best_dist,
            contact_x,
            contact_y,
            contact_z,
            best_nx,
            best_ny,
            best_nz,
        )

    # Fallback
    var fx = (p1x + p2x) * Scalar[DTYPE](0.5)
    var fy = (p1y + p2y) * Scalar[DTYPE](0.5)
    var fz = (p1z + p2z) * Scalar[DTYPE](0.5)
    return (Scalar[DTYPE](-0.001), fx, fy, fz, best_nx, best_ny, best_nz)
