"""GJK/EPA for mesh collision over per-field tensors (migration P4).

Per-field port of gjk_gpu.mojo — arithmetic verbatim. The only change is
the mesh-vertex operand: instead of the flat model slab
(`model[0, mesh_vert_buf_off + i*3 + k]`) the functions take the
`mesh_verts` record tensor (`[NMESH_VERTS, 3]`, see Model) and a
vertex START index (`vert_adr`, MuJoCo `mesh_vertadr`) so reads become
`mesh_verts[vert_adr + i, k]`. Same floats, same iteration order.
"""

from std.math import sqrt, abs
from layout import Layout, LayoutTensor
from .gjk_support import _closest_point_on_simplex
from .gjk_support import (
    support_sphere,
    support_capsule,
    support_box,
    support_cylinder,
)
from ..constants import (
    GEOM_SPHERE,
    GEOM_CAPSULE,
    GEOM_BOX,
    GEOM_CYLINDER,
    GEOM_MESH,
)
from ..kinematics.quat_math import quat_rotate, quat_rotate_inverse

# Reuse CPU GJK parameters (verbatim from gjk_gpu.mojo)
comptime GJK_MAX_ITERATIONS: Int = 100
comptime GJK_TOLERANCE: Float64 = 1e-10
comptime EPA_MAX_ITERATIONS: Int = 64
comptime EPA_MAX_VERTS: Int = 69
comptime EPA_MAX_FACES: Int = 384
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
def _support_mesh[
    DTYPE: DType,
    NMESH_VERTS: Int,
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
    mesh_verts: LayoutTensor[
        DTYPE, Layout.row_major(NMESH_VERTS, 3), MutAnyOrigin
    ],
    vert_adr: Int,
    num_verts: Int,
) -> InlineArray[Scalar[DTYPE], 3]:
    """Support point on mesh reading vertices from the mesh_verts tensor."""
    var local_dir = quat_rotate_inverse[DTYPE](
        qx, qy, qz, qw, dir_x, dir_y, dir_z
    )
    var ld_x = local_dir[0]
    var ld_y = local_dir[1]
    var ld_z = local_dir[2]

    var best_dot: Scalar[DTYPE] = -1e30
    var best_x: Scalar[DTYPE] = 0
    var best_y: Scalar[DTYPE] = 0
    var best_z: Scalar[DTYPE] = 0
    for i in range(num_verts):
        var vx = rebind[Scalar[DTYPE]](mesh_verts[vert_adr + i, 0])
        var vy = rebind[Scalar[DTYPE]](mesh_verts[vert_adr + i, 1])
        var vz = rebind[Scalar[DTYPE]](mesh_verts[vert_adr + i, 2])
        var d = ld_x * vx + ld_y * vy + ld_z * vz
        if d > best_dot:
            best_dot = d
            best_x = vx
            best_y = vy
            best_z = vz

    var world_pt = quat_rotate[DTYPE](qx, qy, qz, qw, best_x, best_y, best_z)
    var result = InlineArray[Scalar[DTYPE], 3](fill=Scalar[DTYPE](0))
    result[0] = pos_x + world_pt[0]
    result[1] = pos_y + world_pt[1]
    result[2] = pos_z + world_pt[2]
    return result


@always_inline
def _support[
    DTYPE: DType,
    NMESH_VERTS: Int,
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
    mesh_verts: LayoutTensor[
        DTYPE, Layout.row_major(NMESH_VERTS, 3), MutAnyOrigin
    ],
    vert_adr: Int,
    mesh_num_verts: Int,
    dir_x: Scalar[DTYPE],
    dir_y: Scalar[DTYPE],
    dir_z: Scalar[DTYPE],
) -> InlineArray[Scalar[DTYPE], 3]:
    """Unified support function — reads mesh verts from the record tensor."""
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
        return _support_mesh[DTYPE, NMESH_VERTS](
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
            vert_adr,
            mesh_num_verts,
        )
    var result = InlineArray[Scalar[DTYPE], 3](fill=Scalar[DTYPE](0))
    result[0] = pos_x
    result[1] = pos_y
    result[2] = pos_z
    return result


@always_inline
def _minkowski_support[
    DTYPE: DType,
    NMESH_VERTS: Int,
](
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
    mesh_verts: LayoutTensor[
        DTYPE, Layout.row_major(NMESH_VERTS, 3), MutAnyOrigin
    ],
    va1: Int,
    mnv1: Int,
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
    va2: Int,
    mnv2: Int,
    dir_x: Scalar[DTYPE],
    dir_y: Scalar[DTYPE],
    dir_z: Scalar[DTYPE],
) -> Tuple[
    Scalar[DTYPE],
    Scalar[DTYPE],
    Scalar[DTYPE],
    Scalar[DTYPE],
    Scalar[DTYPE],
    Scalar[DTYPE],
    Scalar[DTYPE],
    Scalar[DTYPE],
    Scalar[DTYPE],
]:
    var s1 = _support[DTYPE, NMESH_VERTS](
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
        mesh_verts,
        va1,
        mnv1,
        dir_x,
        dir_y,
        dir_z,
    )
    var s2 = _support[DTYPE, NMESH_VERTS](
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
        mesh_verts,
        va2,
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
    DTYPE: DType,
    NMESH_VERTS: Int,
](
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
    mesh_verts: LayoutTensor[
        DTYPE, Layout.row_major(NMESH_VERTS, 3), MutAnyOrigin
    ],
    va1: Int,
    mnv1: Int,
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
    va2: Int,
    mnv2: Int,
) -> Tuple[
    Scalar[DTYPE],
    Scalar[DTYPE],
    Scalar[DTYPE],
    Scalar[DTYPE],
    Scalar[DTYPE],
    Scalar[DTYPE],
    Scalar[DTYPE],
]:
    """GJK distance + EPA penetration depth between two convex shapes
    (verbatim from gjk_epa_gpu; mesh reads via record tensor)."""
    # ===== GJK Phase =====
    var simplex = InlineArray[Scalar[DTYPE], 36](fill=Scalar[DTYPE](0))
    var nsimplex = 0

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

    var s = _minkowski_support[DTYPE, NMESH_VERTS](
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
        mesh_verts,
        va1,
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
        va2,
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

    var vx = s[0]
    var vy = s[1]
    var vz = s[2]

    for _ in range(GJK_MAX_ITERATIONS):
        var v_dot_v = vx * vx + vy * vy + vz * vz
        if v_dot_v < Scalar[DTYPE](GJK_TOLERANCE):
            break

        var inv_vlen = Scalar[DTYPE](1) / sqrt(v_dot_v)
        var ndx = -vx * inv_vlen
        var ndy = -vy * inv_vlen
        var ndz = -vz * inv_vlen

        var sn = _minkowski_support[DTYPE, NMESH_VERTS](
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
            mesh_verts,
            va1,
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
            va2,
            mnv2,
            ndx,
            ndy,
            ndz,
        )

        var w_dot = sn[0] * ndx + sn[1] * ndy + sn[2] * ndz
        var v_dot = vx * ndx + vy * ndy + vz * ndz
        if w_dot - v_dot < Scalar[DTYPE](GJK_TOLERANCE):
            break

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

        # Import and use the CPU closest_point function (works on InlineArray, no List)

        var cp = _closest_point_on_simplex[DTYPE](simplex, nsimplex)
        vx = cp[0]
        vy = cp[1]
        vz = cp[2]
        nsimplex = Int(cp[3])

        if nsimplex == 4:
            vx = Scalar[DTYPE](0)
            vy = Scalar[DTYPE](0)
            vz = Scalar[DTYPE](0)
            break

    var dist = sqrt(vx * vx + vy * vy + vz * vz)

    if dist > Scalar[DTYPE](GJK_TOLERANCE):
        # Separated
        var w1x: Scalar[DTYPE] = 0
        var w1y: Scalar[DTYPE] = 0
        var w1z: Scalar[DTYPE] = 0
        var w2x: Scalar[DTYPE] = 0
        var w2y: Scalar[DTYPE] = 0
        var w2z: Scalar[DTYPE] = 0
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
        var nx = vx / dist
        var ny = vy / dist
        var nz = vz / dist
        return (dist, cx, cy, cz, nx, ny, nz)

    # ===== EPA Phase =====
    # For GPU, use simplified EPA: fallback to center-to-center direction
    # with a shallow penetration estimate. Full EPA with dynamic polytope
    # Lists is not GPU-friendly; the CPU EPA handles the rare deep overlap case.
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

    # Estimate penetration depth from support points along normal
    var s_fwd = _minkowski_support[DTYPE, NMESH_VERTS](
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
        mesh_verts,
        va1,
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
        va2,
        mnv2,
        fallback_nx,
        fallback_ny,
        fallback_nz,
    )
    var pen_depth = _dot3[DTYPE](
        s_fwd[0], s_fwd[1], s_fwd[2], fallback_nx, fallback_ny, fallback_nz
    )
    if pen_depth > Scalar[DTYPE](0):
        pen_depth = -pen_depth  # Make negative for penetration

    var contact_x = (p1x + p2x) * Scalar[DTYPE](0.5)
    var contact_y = (p1y + p2y) * Scalar[DTYPE](0.5)
    var contact_z = (p1z + p2z) * Scalar[DTYPE](0.5)
    return (
        pen_depth,
        contact_x,
        contact_y,
        contact_z,
        fallback_nx,
        fallback_ny,
        fallback_nz,
    )
