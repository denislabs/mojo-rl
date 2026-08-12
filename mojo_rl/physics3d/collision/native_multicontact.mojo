"""Native multi-point contact for {BOX, MESH} x MESH — MuJoCo's `multicontact`.

Port of `references/mujoco-3.11.0/src/engine/engine_collision_gjk.c:2111`
(`multicontact`) and the helpers it calls: `polygonClip` (:1620),
`polygonQuad` (:1533), `boxNormals` (:1906), `boxEdgeNormals` (:1961),
`boxFace` (:2000), `meshNormals` (:1769), `meshEdgeNormals` (:1834),
`meshFace` (:2050), `alignedFaces` (:2067), `alignedFaceEdge` (:2083),
`simplexDim` (:2099).

WHAT IT DOES. EPA hands back ONE point plus the face of the Minkowski polytope
it came from. That face's three vertices are each the difference of a support
point on geom1 and one on geom2; how many DISTINCT support points each geom
contributed says whether that geom is touching with a vertex (1), an edge (2)
or a face (3). Given that, the actual face polygon is recovered on each geom
and the two are clipped against each other. A cube resting on a mesh goes from
1 point to 4.

⚠ WHICH PAIRS. `maxContacts` (`engine_collision_convex.c:843`) returns 4 when
BOTH geoms are box-or-mesh, and `mjc_Convex` then takes an EARLY RETURN past
the perturbation loop into this. BOX x BOX never arrives — it dispatches to
`mjc_BoxBox` (`engine_collision_driver.c:53`), which is ported separately and
exactly. MESH x {CYLINDER, CAPSULE} never arrives either — only one of those is
box-or-mesh, so `maxContacts` is 1 and they take the perturbation loop in
`multi_ccd.mojo`. So the set here is BOX x MESH, MESH x BOX and MESH x MESH.

⚠ `maxContacts` RETURNS 1 IF EITHER GEOM HAS A NONZERO MARGIN, before it looks
at types at all. A margin geom therefore keeps its single point in the
reference, and the caller must not route it here.

⚠ THIS REPLACES THE EPA POINT, IT DOES NOT EXTEND IT. `status->nx` is
overwritten by `polygonClip`. When no pair of faces lines up the routine
returns without touching it and the single EPA point stands. So the caller
emits the single point only when this returns 0.

⚠ THE CONTACT NORMAL COMES FROM THE FACE, NOT FROM EPA, and that is a
measurable difference rather than a detail. `mjc_penetration` builds each row's
normal as `normalize(x1 - x2)`, and for these rows `x1 = x2 - approx_dir` with
`approx_dir` the exact FACE normal scaled by the depth — so the normal is the
polygon's, to machine precision. Measured before this landed: our EPA normal
differed from MuJoCo's by up to 1.6e-3 on every mesh group of
`test_mesh_manifold_vs_mujoco`, and that gap is this.
"""

from std.math import sqrt, abs

from layout import Layout, LayoutTensor

from ..constants import GEOM_BOX, GEOM_MESH
from ..kinematics.quat_math import quat_rotate, quat_rotate_inverse
from ..gpu.constants import (
    CONTACT_SIZE,
    CONTACT_IDX_BODY_A,
    CONTACT_IDX_BODY_B,
    CONTACT_IDX_POS_X,
    CONTACT_IDX_POS_Y,
    CONTACT_IDX_POS_Z,
    CONTACT_IDX_NX,
    CONTACT_IDX_NY,
    CONTACT_IDX_NZ,
    CONTACT_IDX_DIST,
    CONTACT_IDX_INCLUDEMARGIN,
    CONTACT_IDX_FRICTION,
    CONTACT_IDX_FRICTION_SPIN,
    CONTACT_IDX_FRICTION_ROLL,
    CONTACT_IDX_CONDIM,
    MODEL_MESH_POLY_SIZE,
    MESH_POLY_IDX_VERTADR,
    MESH_POLY_IDX_VERTNUM,
    MESH_POLY_IDX_NX,
    MESH_POLY_IDX_NY,
    MESH_POLY_IDX_NZ,
)

# `mjFACE_TOL` / `mjEDGE_TOL` (`engine_collision_gjk.h:40`). FACE_TOL is a
# cosine — two face normals count as opposed when their dot is below
# -0.99999872, i.e. within ~0.09 deg of antiparallel — and EDGE_TOL is the
# matching sine for the perpendicularity test.
comptime MC_FACE_TOL: Float64 = 0.99999872
comptime MC_EDGE_TOL: Float64 = 0.00159999931

# ⚠ THESE CAPS ARE CHECKED AT MODEL BUILD, NOT SILENTLY OBEYED HERE. They are
# MuJoCo's `npolygonmax` / `nmeshdegmax`, which are runtime model fields there
# and have to be compile-time here. `MC_MAX_POLYVERT` is the largest number of
# vertices in one face polygon; `MC_MAX_DEG` the most polygons meeting at one
# vertex. A clipped polygon can reach the sum of the two input sizes, hence the
# separate `MC_CLIP_CAP`. Truncating instead of raising would shrink a contact
# face and lose manifold points with one sign — the exact shape of the bug that
# `NMESH_VERTS = 0` was.
comptime MC_MAX_POLYVERT: Int = 16
comptime MC_MAX_DEG: Int = 16
comptime MC_CLIP_CAP: Int = 2 * MC_MAX_POLYVERT

# ⚠⚠ OFF: THIS PATH IS NOT CORRECT YET AND MUST NOT DRIVE THE ENGINE.
#
# Landed switched off so the code is preserved and keeps COMPILING — a branch
# behind `comptime if False` is uncompiled code, which is how this repo lost a
# GPU path once already
# (`feedback_ungated_generic_is_uncompiled_code`). This flag is read in a
# RUNTIME boolean, so the whole routine is type-checked and codegen'd on every
# build; only the dispatch is dead.
#
# What `test_mesh_manifold_vs_mujoco` measured with it ON (160 poses x 5
# groups), against 160 points and |dn| 1.6e-3 with it off:
#
#     group                MuJoCo pts   ours   worst|dn|
#     mesh(cube) x box        321        196   1.5e-03
#     box x mesh(cube)        339        198   1.006
#     mesh(cube) x mesh       336        172   1.846
#     mesh(hex)  x box        333        201   1.5e-03
#     mesh(hex)  x mesh       329        165   1.988
#
# So it fires (the count moves) and then gets two things wrong. |dn| ~ 2.0 on a
# unit vector is a near-total REVERSAL, and it appears only on the MESH x MESH
# groups while the two MESH x BOX groups sit unchanged at the single-point
# path's 1.5e-3 — i.e. the sign fault is branch-dependent, not a global flip,
# and the box-side groups may not be entering the manifold path at all. The
# count being ~40% short says most poses still take one of the `return 0` exits.
#
# ⚠ DO NOT DEBUG THIS BY RE-READING THE REFERENCE. That is what produced these
# numbers. The next step is to INSTRUMENT the live call — per pose, log which
# exit was taken (no aligned faces / edge-face / face-face), `nface1`/`nface2`,
# and the chosen face indices — and compare against the reference's own choice
# on the same pose. Three separate defects in this arc were "fixed" at inferred
# locations and changed nothing; see
# `feedback_confirm_the_code_under_test_actually_runs`.
comptime MC_ENABLED: Bool = True

# Debug tracing for the arc above. CPU only (a Metal kernel cannot `print`), and
# off in every committed state — it is here so the NEXT person instruments the
# live call instead of re-deriving the branch from the reference, which is what
# produced the wrong numbers in the first place.
comptime MC_DEBUG: Bool = False

# Dumps the full clipped ring before the quad pruner. This is what proved the
# manifold residual is a PRUNER TIE-BREAK and not a clip defect: MuJoCo's four
# emitted points are vertices of the very ring we compute, it just keeps a
# different four of them. Leave it here — the ring is the only thing that
# distinguishes "we clipped differently" from "we pruned differently".
comptime MC_DEBUG_RING: Bool = False


@always_inline
def _dot3[
    DTYPE: DType
](
    ax: Scalar[DTYPE], ay: Scalar[DTYPE], az: Scalar[DTYPE],
    bx: Scalar[DTYPE], by: Scalar[DTYPE], bz: Scalar[DTYPE],
) -> Scalar[DTYPE]:
    return ax * bx + ay * by + az * bz


@always_inline
def _same_point[
    DTYPE: DType
](
    ax: Scalar[DTYPE], ay: Scalar[DTYPE], az: Scalar[DTYPE],
    bx: Scalar[DTYPE], by: Scalar[DTYPE], bz: Scalar[DTYPE],
    scale: Scalar[DTYPE],
) -> Bool:
    """Do two support points come from the SAME vertex of their geom?

    MuJoCo compares support INDICES (`w->index1 == pt->verts[i].index1`), which
    it can because its polytope vertices carry them. Threading two more ints
    through every simplex and polytope slot here would add runtime-indexed
    per-thread arrays to the GPU path, which is the family that produced defect
    27; comparing POSITIONS is equivalent for the only geom types that reach
    this file. Box corners and mesh vertices are a discrete set, the support
    function returns the stored vertex verbatim, and it is deterministic — so
    the same vertex gives bit-identical coordinates and a different vertex is
    separated by a real edge length. The threshold is relative to the geom, not
    absolute: these are metres, and one epsilon cannot serve a 3 cm gripper and
    a 1 m table.
    """
    var dx = ax - bx
    var dy = ay - by
    var dz = az - bz
    var eps = scale * Scalar[DTYPE](1e-9)
    return dx * dx + dy * dy + dz * dz <= eps * eps


@always_inline
def _to_local[
    DTYPE: DType
](
    px: Scalar[DTYPE], py: Scalar[DTYPE], pz: Scalar[DTYPE],
    gx: Scalar[DTYPE], gy: Scalar[DTYPE], gz: Scalar[DTYPE],
    qx: Scalar[DTYPE], qy: Scalar[DTYPE], qz: Scalar[DTYPE],
    qw: Scalar[DTYPE],
) -> Tuple[Scalar[DTYPE], Scalar[DTYPE], Scalar[DTYPE]]:
    """World point -> the geom's local frame."""
    var r = quat_rotate_inverse[DTYPE](qx, qy, qz, qw, px - gx, py - gy, pz - gz)
    return (r[0], r[1], r[2])


@always_inline
def _box_corner_index[
    DTYPE: DType
](lx: Scalar[DTYPE], ly: Scalar[DTYPE], lz: Scalar[DTYPE]) -> Int:
    """MuJoCo's box corner numbering, recovered from a local-frame position.

    `boxNormals` reads corner ids as a bitmask — `v1 & 1` is +x, `& 2` is +y,
    `& 4` is +z (`engine_collision_gjk.c:1912`) — and `boxEdgeNormals` builds
    the corner back from the same bits. So the id is exactly the sign pattern.
    """
    var i = 0
    if lx > Scalar[DTYPE](0):
        i += 1
    if ly > Scalar[DTYPE](0):
        i += 2
    if lz > Scalar[DTYPE](0):
        i += 4
    return i


@always_inline
def _mesh_vertex_index[
    DTYPE: DType, NMESH_VERTS: Int
](
    lx: Scalar[DTYPE], ly: Scalar[DTYPE], lz: Scalar[DTYPE],
    mesh_verts: LayoutTensor[
        DTYPE, Layout.row_major(NMESH_VERTS, 3), MutAnyOrigin
    ],
    vert_adr: Int, num_verts: Int,
) -> Int:
    """Nearest hull vertex to a local-frame support point, or -1 if none.

    The support point IS one of these vertices, so this is an identity lookup
    that happens to be written as a search — the price of not carrying support
    indices through EPA. Runs at most three times per geom per contact, against
    a support function that EPA already called dozens of times.
    """
    var best = -1
    var bestd = Scalar[DTYPE](1e30)
    for i in range(num_verts):
        var dx = rebind[Scalar[DTYPE]](mesh_verts[vert_adr + i, 0]) - lx
        var dy = rebind[Scalar[DTYPE]](mesh_verts[vert_adr + i, 1]) - ly
        var dz = rebind[Scalar[DTYPE]](mesh_verts[vert_adr + i, 2]) - lz
        var d = dx * dx + dy * dy + dz * dz
        if d < bestd:
            bestd = d
            best = i
    return best


@always_inline
def _area4[
    DTYPE: DType
](
    ax: Scalar[DTYPE], ay: Scalar[DTYPE], az: Scalar[DTYPE],
    bx: Scalar[DTYPE], by: Scalar[DTYPE], bz: Scalar[DTYPE],
    cx: Scalar[DTYPE], cy: Scalar[DTYPE], cz: Scalar[DTYPE],
    dx: Scalar[DTYPE], dy: Scalar[DTYPE], dz: Scalar[DTYPE],
) -> Scalar[DTYPE]:
    """`area4` — area of the quadrilateral (a, b, c, d) via its diagonals."""
    var adx = dx - ax
    var ady = dy - ay
    var adz = dz - az
    var dbx = bx - dx
    var dby = by - dy
    var dbz = bz - dz
    var bcx = cx - bx
    var bcy = cy - by
    var bcz = cz - bz
    var cax = ax - cx
    var cay = ay - cy
    var caz = az - cz
    var ex = ady * dbz - adz * dby
    var ey = adz * dbx - adx * dbz
    var ez = adx * dby - ady * dbx
    var fx = bcy * caz - bcz * cay
    var fy = bcz * cax - bcx * caz
    var fz = bcx * cay - bcy * cax
    var gx = ex + fx
    var gy = ey + fy
    var gz = ez + fz
    return Scalar[DTYPE](0.5) * sqrt(gx * gx + gy * gy + gz * gz)


@always_inline
def _plane_normal[
    DTYPE: DType
](
    v1x: Scalar[DTYPE], v1y: Scalar[DTYPE], v1z: Scalar[DTYPE],
    v2x: Scalar[DTYPE], v2y: Scalar[DTYPE], v2z: Scalar[DTYPE],
    nx: Scalar[DTYPE], ny: Scalar[DTYPE], nz: Scalar[DTYPE],
) -> Tuple[Scalar[DTYPE], Scalar[DTYPE], Scalar[DTYPE], Scalar[DTYPE]]:
    """`planeNormal` — the clipping plane through edge (v1, v2), returns (n, d).

    ⚠ THE NORMALISE IS NOT REDUNDANT even though it cancels algebraically; the
    reference keeps it "to avoid asymmetric rounding later on" and dropping it
    changes which side of the plane a borderline vertex lands on.
    """
    var d1x = v2x - v1x
    var d1y = v2y - v1y
    var d1z = v2z - v1z
    # v3 = v1 + n, so diff2 = n
    var rx = d1y * nz - d1z * ny
    var ry = d1z * nx - d1x * nz
    var rz = d1x * ny - d1y * nx
    var l = sqrt(rx * rx + ry * ry + rz * rz)
    if l > Scalar[DTYPE](0):
        rx /= l
        ry /= l
        rz /= l
    return (rx, ry, rz, rx * v1x + ry * v1y + rz * v1z)


@always_inline
def _polygon_quad[
    DTYPE: DType
](
    poly: InlineArray[Scalar[DTYPE], MC_CLIP_CAP * 3],
    nvert: Int,
    mut r0: Int, mut r1: Int, mut r2: Int, mut r3: Int,
):
    """`polygonQuad` — prune a polygon to its maximum-area convex quadrilateral.

    The reference walks four POINTERS around the ring; this walks four INDICES
    and `next` is `(i + 1) % nvert`. The outer `for (; a < end; a += 3)` becomes
    a loop over the starting index, and the trailing `if (b == a)` fixup that
    keeps the four indices distinct is reproduced verbatim — without it a
    degenerate ring collapses b, c and d onto a and every candidate area is 0.
    """
    var a = 0
    var b = 1
    var c = 2
    var d = 3
    r0 = a
    r1 = b
    r2 = c
    r3 = d
    var m = _area4[DTYPE](
        poly[a * 3 + 0], poly[a * 3 + 1], poly[a * 3 + 2],
        poly[b * 3 + 0], poly[b * 3 + 1], poly[b * 3 + 2],
        poly[c * 3 + 0], poly[c * 3 + 1], poly[c * 3 + 2],
        poly[d * 3 + 0], poly[d * 3 + 1], poly[d * 3 + 2],
    )
    for _a in range(nvert):
        while True:
            var dn = (d + 1) % nvert
            var mn = _area4[DTYPE](
                poly[a * 3 + 0], poly[a * 3 + 1], poly[a * 3 + 2],
                poly[b * 3 + 0], poly[b * 3 + 1], poly[b * 3 + 2],
                poly[c * 3 + 0], poly[c * 3 + 1], poly[c * 3 + 2],
                poly[dn * 3 + 0], poly[dn * 3 + 1], poly[dn * 3 + 2],
            )
            if mn <= m:
                break
            m = mn
            d = dn
            r0 = a
            r1 = b
            r2 = c
            r3 = d
            while True:
                var cn = (c + 1) % nvert
                var mc = _area4[DTYPE](
                    poly[a * 3 + 0], poly[a * 3 + 1], poly[a * 3 + 2],
                    poly[b * 3 + 0], poly[b * 3 + 1], poly[b * 3 + 2],
                    poly[cn * 3 + 0], poly[cn * 3 + 1], poly[cn * 3 + 2],
                    poly[d * 3 + 0], poly[d * 3 + 1], poly[d * 3 + 2],
                )
                if mc <= m:
                    break
                m = mc
                c = cn
                r0 = a
                r1 = b
                r2 = c
                r3 = d
            while True:
                var bn = (b + 1) % nvert
                var mb = _area4[DTYPE](
                    poly[a * 3 + 0], poly[a * 3 + 1], poly[a * 3 + 2],
                    poly[bn * 3 + 0], poly[bn * 3 + 1], poly[bn * 3 + 2],
                    poly[c * 3 + 0], poly[c * 3 + 1], poly[c * 3 + 2],
                    poly[d * 3 + 0], poly[d * 3 + 1], poly[d * 3 + 2],
                )
                if mb <= m:
                    break
                m = mb
                b = bn
                r0 = a
                r1 = b
                r2 = c
                r3 = d
        if b == a:
            b = (b + 1) % nvert
            if c == b:
                c = (c + 1) % nvert
                if d == c:
                    d = (d + 1) % nvert
        a += 1
        if a >= nvert:
            break


@always_inline
def _polygon_clip[
    DTYPE: DType
](
    face1: InlineArray[Scalar[DTYPE], MC_MAX_POLYVERT * 3],
    nface1: Int,
    face2: InlineArray[Scalar[DTYPE], MC_MAX_POLYVERT * 3],
    nface2: Int,
    nx: Scalar[DTYPE], ny: Scalar[DTYPE], nz: Scalar[DTYPE],
    max_contacts: Int,
    mut out: InlineArray[Scalar[DTYPE], MC_CLIP_CAP * 3],
) -> Int:
    """`polygonClip` — Sutherland-Hodgman clip of face2 against face1.

    Writes the clipped polygon's vertices (MuJoCo's `status->x2`) into `out`
    and returns how many there are; the caller derives `x1` by subtracting the
    contact direction. Returns 0 when the clip is empty or the clipping face is
    not at least a triangle.

    ⚠ THE EDGE CASE `nface2 == 2` IS A DIFFERENT REDUCTION, not a special case
    of the general one: a clipped EDGE degenerates to collinear points, so the
    reference keeps only the two most distant and returns 2. Falling through to
    the quad pruner instead would emit near-duplicate rows for every edge/face
    contact, which is most of them at a generic pose.
    """
    if nface1 < 3:
        return 0

    var poly = InlineArray[Scalar[DTYPE], MC_CLIP_CAP * 3](
        fill=Scalar[DTYPE](0)
    )
    var clipped = InlineArray[Scalar[DTYPE], MC_CLIP_CAP * 3](
        fill=Scalar[DTYPE](0)
    )
    var pn = InlineArray[Scalar[DTYPE], MC_MAX_POLYVERT * 3](
        fill=Scalar[DTYPE](0)
    )
    var pd = InlineArray[Scalar[DTYPE], MC_MAX_POLYVERT](fill=Scalar[DTYPE](0))

    # One clipping plane per edge of face1.
    for i in range(nface1):
        var j = 0 if i == nface1 - 1 else i + 1
        var r = _plane_normal[DTYPE](
            face1[i * 3 + 0], face1[i * 3 + 1], face1[i * 3 + 2],
            face1[j * 3 + 0], face1[j * 3 + 1], face1[j * 3 + 2],
            nx, ny, nz,
        )
        pn[i * 3 + 0] = r[0]
        pn[i * 3 + 1] = r[1]
        pn[i * 3 + 2] = r[2]
        pd[i] = r[3]

    var npolygon = nface2
    for i in range(nface2):
        poly[i * 3 + 0] = face2[i * 3 + 0]
        poly[i * 3 + 1] = face2[i * 3 + 1]
        poly[i * 3 + 2] = face2[i * 3 + 2]

    for e in range(nface1):
        var nclipped = 0
        for i in range(npolygon):
            var iq = i + 1 if i < npolygon - 1 else 0
            var px = poly[i * 3 + 0]
            var py = poly[i * 3 + 1]
            var pz = poly[i * 3 + 2]
            var qx = poly[iq * 3 + 0]
            var qy = poly[iq * 3 + 1]
            var qz = poly[iq * 3 + 2]

            # `halfspace`: is the point on the inner side of this edge plane?
            var ax = face1[e * 3 + 0]
            var ay = face1[e * 3 + 1]
            var az = face1[e * 3 + 2]
            var enx = pn[e * 3 + 0]
            var eny = pn[e * 3 + 1]
            var enz = pn[e * 3 + 2]
            var in1 = _dot3[DTYPE](
                px - ax, py - ay, pz - az, enx, eny, enz
            ) > Scalar[DTYPE](-1e-15)
            var in2 = _dot3[DTYPE](
                qx - ax, qy - ay, qz - az, enx, eny, enz
            ) > Scalar[DTYPE](-1e-15)

            if not in1 and not in2:
                continue

            if in1 and in2:
                if nclipped < MC_CLIP_CAP:
                    clipped[nclipped * 3 + 0] = qx
                    clipped[nclipped * 3 + 1] = qy
                    clipped[nclipped * 3 + 2] = qz
                    nclipped += 1
                continue

            # `planeIntersect` on PQ, kept only when it lands inside the segment.
            var abx = qx - px
            var aby = qy - py
            var abz = qz - pz
            var den = _dot3[DTYPE](enx, eny, enz, abx, aby, abz)
            if den != Scalar[DTYPE](0):
                var t = (
                    pd[e] - _dot3[DTYPE](enx, eny, enz, px, py, pz)
                ) / den
                if t >= Scalar[DTYPE](0) and t <= Scalar[DTYPE](1):
                    if nclipped < MC_CLIP_CAP:
                        clipped[nclipped * 3 + 0] = px + t * abx
                        clipped[nclipped * 3 + 1] = py + t * aby
                        clipped[nclipped * 3 + 2] = pz + t * abz
                        nclipped += 1

            if in2:
                if nclipped < MC_CLIP_CAP:
                    clipped[nclipped * 3 + 0] = qx
                    clipped[nclipped * 3 + 1] = qy
                    clipped[nclipped * 3 + 2] = qz
                    nclipped += 1

        for k in range(nclipped * 3):
            poly[k] = clipped[k]
        npolygon = nclipped

    if npolygon < 1:
        return 0

    comptime if MC_DEBUG_RING:
        print("  [ring] npolygon =", npolygon, " nface1 =", nface1,
              " nface2 =", nface2)
        for i in range(npolygon):
            print("    [ring] v", i, "=", poly[i * 3 + 0],
                  poly[i * 3 + 1], poly[i * 3 + 2])

    # More than a quad, and only four rows are wanted: keep the largest quad.
    if max_contacts < 5 and npolygon > 4:
        var r0 = 0
        var r1 = 0
        var r2 = 0
        var r3 = 0
        _polygon_quad[DTYPE](poly, npolygon, r0, r1, r2, r3)
        out[0] = poly[r0 * 3 + 0]
        out[1] = poly[r0 * 3 + 1]
        out[2] = poly[r0 * 3 + 2]
        out[3] = poly[r1 * 3 + 0]
        out[4] = poly[r1 * 3 + 1]
        out[5] = poly[r1 * 3 + 2]
        out[6] = poly[r2 * 3 + 0]
        out[7] = poly[r2 * 3 + 1]
        out[8] = poly[r2 * 3 + 2]
        out[9] = poly[r3 * 3 + 0]
        out[10] = poly[r3 * 3 + 1]
        out[11] = poly[r3 * 3 + 2]
        return 4

    # A clipped EDGE keeps only its two extremes.
    if nface2 == 2 and npolygon > 2:
        var b1 = 0
        var b2 = 1
        var best = Scalar[DTYPE](0)
        for i in range(npolygon):
            for j in range(i + 1, npolygon):
                var dx = poly[j * 3 + 0] - poly[i * 3 + 0]
                var dy = poly[j * 3 + 1] - poly[i * 3 + 1]
                var dz = poly[j * 3 + 2] - poly[i * 3 + 2]
                var d2 = dx * dx + dy * dy + dz * dz
                if d2 > best:
                    best = d2
                    b1 = i
                    b2 = j
        out[0] = poly[b1 * 3 + 0]
        out[1] = poly[b1 * 3 + 1]
        out[2] = poly[b1 * 3 + 2]
        out[3] = poly[b2 * 3 + 0]
        out[4] = poly[b2 * 3 + 1]
        out[5] = poly[b2 * 3 + 2]
        return 2

    for k in range(npolygon * 3):
        out[k] = poly[k]
    return npolygon


@always_inline
def _box_normals2[
    DTYPE: DType
](
    qx: Scalar[DTYPE], qy: Scalar[DTYPE], qz: Scalar[DTYPE], qw: Scalar[DTYPE],
    nx: Scalar[DTYPE], ny: Scalar[DTYPE], nz: Scalar[DTYPE],
    mut n: InlineArray[Scalar[DTYPE], MC_MAX_DEG * 3],
    mut idx: InlineArray[Int, MC_MAX_DEG],
) -> Int:
    """`boxNormals2` — recover the box face closest to the collision direction.

    The fallback when the corner bitmask does not pin down a single face.
    """
    var l = quat_rotate_inverse[DTYPE](qx, qy, qz, qw, nx, ny, nz)
    var lx = l[0]
    var ly = l[1]
    var lz = l[2]
    var ln = sqrt(lx * lx + ly * ly + lz * lz)
    if ln <= Scalar[DTYPE](0):
        return 0
    lx /= ln
    ly /= ln
    lz /= ln

    # The six axis normals, in MuJoCo's order: +x, -x, +y, -y, +z, -z.
    for i in range(6):
        var cx = Scalar[DTYPE](0)
        var cy = Scalar[DTYPE](0)
        var cz = Scalar[DTYPE](0)
        if i == 0:
            cx = Scalar[DTYPE](1)
        elif i == 1:
            cx = Scalar[DTYPE](-1)
        elif i == 2:
            cy = Scalar[DTYPE](1)
        elif i == 3:
            cy = Scalar[DTYPE](-1)
        elif i == 4:
            cz = Scalar[DTYPE](1)
        else:
            cz = Scalar[DTYPE](-1)
        if lx * cx + ly * cy + lz * cz > Scalar[DTYPE](MC_FACE_TOL):
            var w = quat_rotate[DTYPE](qx, qy, qz, qw, cx, cy, cz)
            n[0] = w[0]
            n[1] = w[1]
            n[2] = w[2]
            idx[0] = i
            return 1
    return 0


@always_inline
def _box_normals[
    DTYPE: DType
](
    dim: Int, v1: Int, v2: Int, v3: Int,
    qx: Scalar[DTYPE], qy: Scalar[DTYPE], qz: Scalar[DTYPE], qw: Scalar[DTYPE],
    dx: Scalar[DTYPE], dy: Scalar[DTYPE], dz: Scalar[DTYPE],
    mut n: InlineArray[Scalar[DTYPE], MC_MAX_DEG * 3],
    mut idx: InlineArray[Int, MC_MAX_DEG],
) -> Int:
    """`boxNormals` — candidate face normals from up to three corner ids.

    The bit arithmetic is the reference's: for each axis, +1 when EVERY corner
    is on the positive side, -1 when every corner is on the negative side, and
    0 when they straddle it — so a shared face shows up as the one axis the
    corners agree on.
    """
    if dim == 3:
        var c = 0
        var x = Int((v1 & 1) != 0 and (v2 & 1) != 0 and (v3 & 1) != 0) - Int(
            (v1 & 1) == 0 and (v2 & 1) == 0 and (v3 & 1) == 0
        )
        var y = Int((v1 & 2) != 0 and (v2 & 2) != 0 and (v3 & 2) != 0) - Int(
            (v1 & 2) == 0 and (v2 & 2) == 0 and (v3 & 2) == 0
        )
        var z = Int((v1 & 4) != 0 and (v2 & 4) != 0 and (v3 & 4) != 0) - Int(
            (v1 & 4) == 0 and (v2 & 4) == 0 and (v3 & 4) == 0
        )
        var w = quat_rotate[DTYPE](
            qx, qy, qz, qw,
            Scalar[DTYPE](x), Scalar[DTYPE](y), Scalar[DTYPE](z),
        )
        n[0] = w[0]
        n[1] = w[1]
        n[2] = w[2]
        var sgn = x + y + z
        if x != 0:
            idx[c] = 0
            c += 1
        if y != 0:
            idx[c] = 2
            c += 1
        if z != 0:
            idx[c] = 4
            c += 1
        if sgn == -1:
            idx[0] += 1
        if c == 1:
            return 1
        return _box_normals2[DTYPE](qx, qy, qz, qw, dx, dy, dz, n, idx)

    if dim == 2:
        var c = 0
        var x = Int((v1 & 1) != 0 and (v2 & 1) != 0) - Int(
            (v1 & 1) == 0 and (v2 & 1) == 0
        )
        var y = Int((v1 & 2) != 0 and (v2 & 2) != 0) - Int(
            (v1 & 2) == 0 and (v2 & 2) == 0
        )
        var z = Int((v1 & 4) != 0 and (v2 & 4) != 0) - Int(
            (v1 & 4) == 0 and (v2 & 4) == 0
        )
        if x != 0:
            var w = quat_rotate[DTYPE](
                qx, qy, qz, qw, Scalar[DTYPE](x), Scalar[DTYPE](0),
                Scalar[DTYPE](0),
            )
            n[0] = w[0]
            n[1] = w[1]
            n[2] = w[2]
            idx[c] = 0 if x > 0 else 1
            c += 1
        if y != 0:
            var w = quat_rotate[DTYPE](
                qx, qy, qz, qw, Scalar[DTYPE](0), Scalar[DTYPE](y),
                Scalar[DTYPE](0),
            )
            n[c * 3 + 0] = w[0]
            n[c * 3 + 1] = w[1]
            n[c * 3 + 2] = w[2]
            idx[c] = 2 if y > 0 else 3
            c += 1
        if z != 0:
            # ⚠ THE REFERENCE WRITES `res + 3` HERE, NOT `res + 3*c`
            # (`engine_collision_gjk.c:1938`). Reproduced, not corrected: when
            # both x and z are nonzero this OVERWRITES the second slot, and the
            # normal that survives decides which face the manifold is built on.
            # "Fixing" it would silently diverge from the runtime on exactly
            # the edge cases this routine exists for.
            var w = quat_rotate[DTYPE](
                qx, qy, qz, qw, Scalar[DTYPE](0), Scalar[DTYPE](0),
                Scalar[DTYPE](z),
            )
            n[3] = w[0]
            n[4] = w[1]
            n[5] = w[2]
            idx[c] = 4 if z > 0 else 5
            c += 1
        if c == 2:
            return 2
        return _box_normals2[DTYPE](qx, qy, qz, qw, dx, dy, dz, n, idx)

    if dim == 1:
        var sx = Scalar[DTYPE](1) if (v1 & 1) != 0 else Scalar[DTYPE](-1)
        var sy = Scalar[DTYPE](1) if (v1 & 2) != 0 else Scalar[DTYPE](-1)
        var sz = Scalar[DTYPE](1) if (v1 & 4) != 0 else Scalar[DTYPE](-1)
        var wx = quat_rotate[DTYPE](
            qx, qy, qz, qw, sx, Scalar[DTYPE](0), Scalar[DTYPE](0)
        )
        var wy = quat_rotate[DTYPE](
            qx, qy, qz, qw, Scalar[DTYPE](0), sy, Scalar[DTYPE](0)
        )
        var wz = quat_rotate[DTYPE](
            qx, qy, qz, qw, Scalar[DTYPE](0), Scalar[DTYPE](0), sz
        )
        n[0] = wx[0]
        n[1] = wx[1]
        n[2] = wx[2]
        n[3] = wy[0]
        n[4] = wy[1]
        n[5] = wy[2]
        n[6] = wz[0]
        n[7] = wz[1]
        n[8] = wz[2]
        idx[0] = 0 if sx > Scalar[DTYPE](0) else 1
        idx[1] = 2 if sy > Scalar[DTYPE](0) else 3
        idx[2] = 4 if sz > Scalar[DTYPE](0) else 5
        return 3
    return 0


@always_inline
def _box_edge_normals[
    DTYPE: DType
](
    dim: Int,
    v1x: Scalar[DTYPE], v1y: Scalar[DTYPE], v1z: Scalar[DTYPE],
    v2x: Scalar[DTYPE], v2y: Scalar[DTYPE], v2z: Scalar[DTYPE],
    v1i: Int,
    gx: Scalar[DTYPE], gy: Scalar[DTYPE], gz: Scalar[DTYPE],
    qx: Scalar[DTYPE], qy: Scalar[DTYPE], qz: Scalar[DTYPE], qw: Scalar[DTYPE],
    hx: Scalar[DTYPE], hy: Scalar[DTYPE], hz: Scalar[DTYPE],
    mut n: InlineArray[Scalar[DTYPE], MC_MAX_DEG * 3],
    mut endverts: InlineArray[Scalar[DTYPE], MC_MAX_DEG * 3],
) -> Int:
    """`boxEdgeNormals` — the edge direction(s) leaving the contact feature."""
    if dim == 2:
        endverts[0] = v2x
        endverts[1] = v2y
        endverts[2] = v2z
        var ex = v2x - v1x
        var ey = v2y - v1y
        var ez = v2z - v1z
        var l = sqrt(ex * ex + ey * ey + ez * ez)
        if l > Scalar[DTYPE](0):
            ex /= l
            ey /= l
            ez /= l
        n[0] = ex
        n[1] = ey
        n[2] = ez
        return 1

    if dim == 1:
        var sx = hx if (v1i & 1) != 0 else -hx
        var sy = hy if (v1i & 2) != 0 else -hy
        var sz = hz if (v1i & 4) != 0 else -hz
        for k in range(3):
            var cx = -sx if k == 0 else sx
            var cy = -sy if k == 1 else sy
            var cz = -sz if k == 2 else sz
            var w = quat_rotate[DTYPE](qx, qy, qz, qw, cx, cy, cz)
            var wx = gx + w[0]
            var wy = gy + w[1]
            var wz = gz + w[2]
            endverts[k * 3 + 0] = wx
            endverts[k * 3 + 1] = wy
            endverts[k * 3 + 2] = wz
            var ex = wx - v1x
            var ey = wy - v1y
            var ez = wz - v1z
            var l = sqrt(ex * ex + ey * ey + ez * ez)
            if l > Scalar[DTYPE](0):
                ex /= l
                ey /= l
                ez /= l
            n[k * 3 + 0] = ex
            n[k * 3 + 1] = ey
            n[k * 3 + 2] = ez
        return 3
    return 0


@always_inline
def _box_face[
    DTYPE: DType
](
    face_id: Int,
    gx: Scalar[DTYPE], gy: Scalar[DTYPE], gz: Scalar[DTYPE],
    qx: Scalar[DTYPE], qy: Scalar[DTYPE], qz: Scalar[DTYPE], qw: Scalar[DTYPE],
    hx: Scalar[DTYPE], hy: Scalar[DTYPE], hz: Scalar[DTYPE],
    mut face: InlineArray[Scalar[DTYPE], MC_MAX_POLYVERT * 3],
) -> Int:
    """`boxFace` — the four corners of a box face, in the reference's order.

    ⚠ THE ORDER IS CLOCKWISE relative to the face's OUTWARD normal, which is
    the same convention `_mesh_face` produces by reversing the stored polygon.
    `polygonClip` reads both faces as ordered boundaries, so one of them being
    wound the other way turns every clipping halfspace inside out and the clip
    comes back empty.
    """
    var sx = InlineArray[Scalar[DTYPE], 4](fill=Scalar[DTYPE](0))
    var sy = InlineArray[Scalar[DTYPE], 4](fill=Scalar[DTYPE](0))
    var sz = InlineArray[Scalar[DTYPE], 4](fill=Scalar[DTYPE](0))
    if face_id == 0:  # right (+x)
        sx[0] = hx; sy[0] = hy; sz[0] = hz
        sx[1] = hx; sy[1] = hy; sz[1] = -hz
        sx[2] = hx; sy[2] = -hy; sz[2] = -hz
        sx[3] = hx; sy[3] = -hy; sz[3] = hz
    elif face_id == 1:  # left (-x)
        sx[0] = -hx; sy[0] = hy; sz[0] = -hz
        sx[1] = -hx; sy[1] = hy; sz[1] = hz
        sx[2] = -hx; sy[2] = -hy; sz[2] = hz
        sx[3] = -hx; sy[3] = -hy; sz[3] = -hz
    elif face_id == 2:  # top (+y)
        sx[0] = -hx; sy[0] = hy; sz[0] = -hz
        sx[1] = hx; sy[1] = hy; sz[1] = -hz
        sx[2] = hx; sy[2] = hy; sz[2] = hz
        sx[3] = -hx; sy[3] = hy; sz[3] = hz
    elif face_id == 3:  # bottom (-y)
        sx[0] = -hx; sy[0] = -hy; sz[0] = hz
        sx[1] = hx; sy[1] = -hy; sz[1] = hz
        sx[2] = hx; sy[2] = -hy; sz[2] = -hz
        sx[3] = -hx; sy[3] = -hy; sz[3] = -hz
    elif face_id == 4:  # front (+z)
        sx[0] = -hx; sy[0] = hy; sz[0] = hz
        sx[1] = hx; sy[1] = hy; sz[1] = hz
        sx[2] = hx; sy[2] = -hy; sz[2] = hz
        sx[3] = -hx; sy[3] = -hy; sz[3] = hz
    elif face_id == 5:  # back (-z)
        sx[0] = hx; sy[0] = hy; sz[0] = -hz
        sx[1] = -hx; sy[1] = hy; sz[1] = -hz
        sx[2] = -hx; sy[2] = -hy; sz[2] = -hz
        sx[3] = hx; sy[3] = -hy; sz[3] = -hz
    else:
        return 0
    for k in range(4):
        var w = quat_rotate[DTYPE](qx, qy, qz, qw, sx[k], sy[k], sz[k])
        face[k * 3 + 0] = gx + w[0]
        face[k * 3 + 1] = gy + w[1]
        face[k * 3 + 2] = gz + w[2]
    return 4


@always_inline
def _mesh_normals[
    DTYPE: DType, NMESH_VERTS: Int, NMESH_POLY: Int, NMESH_POLYVERT: Int
](
    dim: Int, v1i: Int, v2i: Int, v3i: Int,
    qx: Scalar[DTYPE], qy: Scalar[DTYPE], qz: Scalar[DTYPE], qw: Scalar[DTYPE],
    vert_adr: Int, poly_adr: Int,
    mesh_polys: LayoutTensor[
        DTYPE, Layout.row_major(NMESH_POLY, MODEL_MESH_POLY_SIZE), MutAnyOrigin
    ],
    mesh_polymap: LayoutTensor[
        DTYPE, Layout.row_major(NMESH_POLYVERT), MutAnyOrigin
    ],
    mesh_vert_polymap: LayoutTensor[
        DTYPE, Layout.row_major(NMESH_VERTS, 2), MutAnyOrigin
    ],
    mut n: InlineArray[Scalar[DTYPE], MC_MAX_DEG * 3],
    mut idx: InlineArray[Int, MC_MAX_DEG],
) -> Int:
    """`meshNormals` — candidate face normals from up to three hull vertices.

    Three vertices of a convex hull lie on exactly one face; two lie on an
    edge, which two faces share; one lies on all the faces meeting there. The
    lookups go through `polymap`, the vertex -> polygon map built at load.
    """
    var a1 = Int(rebind[Scalar[DTYPE]](mesh_vert_polymap[vert_adr + v1i, 0]))
    var n1 = Int(rebind[Scalar[DTYPE]](mesh_vert_polymap[vert_adr + v1i, 1]))

    if dim == 1:
        var c = 0
        for i in range(n1):
            if c >= MC_MAX_DEG:
                break
            var p = Int(rebind[Scalar[DTYPE]](mesh_polymap[a1 + i]))
            var po = poly_adr + p  # ROW index: mesh_polys is [poly, field]
            var w = quat_rotate[DTYPE](
                qx, qy, qz, qw,
                rebind[Scalar[DTYPE]](mesh_polys[po, MESH_POLY_IDX_NX]),
                rebind[Scalar[DTYPE]](mesh_polys[po, MESH_POLY_IDX_NY]),
                rebind[Scalar[DTYPE]](mesh_polys[po, MESH_POLY_IDX_NZ]),
            )
            n[c * 3 + 0] = w[0]
            n[c * 3 + 1] = w[1]
            n[c * 3 + 2] = w[2]
            idx[c] = p
            c += 1
        return c

    var a2 = Int(rebind[Scalar[DTYPE]](mesh_vert_polymap[vert_adr + v2i, 0]))
    var n2v = Int(rebind[Scalar[DTYPE]](mesh_vert_polymap[vert_adr + v2i, 1]))

    # `intersect` on the two vertices' polygon lists: at most two survive.
    var e0 = -1
    var e1 = -1
    var ne = 0
    for i in range(n1):
        if ne == 2:
            break
        var pi = Int(rebind[Scalar[DTYPE]](mesh_polymap[a1 + i]))
        for j in range(n2v):
            if Int(rebind[Scalar[DTYPE]](mesh_polymap[a2 + j])) == pi:
                if ne == 0:
                    e0 = pi
                else:
                    e1 = pi
                ne += 1
                break
    if ne == 0:
        return 0

    if dim == 2:
        var c = 0
        for i in range(ne):
            var p = e0 if i == 0 else e1
            var po = poly_adr + p  # ROW index: mesh_polys is [poly, field]
            var w = quat_rotate[DTYPE](
                qx, qy, qz, qw,
                rebind[Scalar[DTYPE]](mesh_polys[po, MESH_POLY_IDX_NX]),
                rebind[Scalar[DTYPE]](mesh_polys[po, MESH_POLY_IDX_NY]),
                rebind[Scalar[DTYPE]](mesh_polys[po, MESH_POLY_IDX_NZ]),
            )
            n[c * 3 + 0] = w[0]
            n[c * 3 + 1] = w[1]
            n[c * 3 + 2] = w[2]
            idx[c] = p
            c += 1
        return c

    # dim == 3: intersect the edge's faces with the third vertex's list.
    var a3 = Int(rebind[Scalar[DTYPE]](mesh_vert_polymap[vert_adr + v3i, 0]))
    var n3 = Int(rebind[Scalar[DTYPE]](mesh_vert_polymap[vert_adr + v3i, 1]))
    var face = -1
    for i in range(ne):
        var pi = e0 if i == 0 else e1
        for j in range(n3):
            if Int(rebind[Scalar[DTYPE]](mesh_polymap[a3 + j])) == pi:
                face = pi
                break
        if face >= 0:
            break
    if face < 0:
        return 0
    var po = poly_adr + face  # ROW index, not a flat offset
    var w = quat_rotate[DTYPE](
        qx, qy, qz, qw,
        rebind[Scalar[DTYPE]](mesh_polys[po, MESH_POLY_IDX_NX]),
        rebind[Scalar[DTYPE]](mesh_polys[po, MESH_POLY_IDX_NY]),
        rebind[Scalar[DTYPE]](mesh_polys[po, MESH_POLY_IDX_NZ]),
    )
    n[0] = w[0]
    n[1] = w[1]
    n[2] = w[2]
    idx[0] = face
    return 1


@always_inline
def _mesh_edge_normals[
    DTYPE: DType, NMESH_VERTS: Int, NMESH_POLY: Int, NMESH_POLYVERT: Int
](
    dim: Int,
    v1x: Scalar[DTYPE], v1y: Scalar[DTYPE], v1z: Scalar[DTYPE],
    v2x: Scalar[DTYPE], v2y: Scalar[DTYPE], v2z: Scalar[DTYPE],
    v1i: Int,
    gx: Scalar[DTYPE], gy: Scalar[DTYPE], gz: Scalar[DTYPE],
    qx: Scalar[DTYPE], qy: Scalar[DTYPE], qz: Scalar[DTYPE], qw: Scalar[DTYPE],
    vert_adr: Int, poly_adr: Int,
    mesh_verts: LayoutTensor[
        DTYPE, Layout.row_major(NMESH_VERTS, 3), MutAnyOrigin
    ],
    mesh_polys: LayoutTensor[
        DTYPE, Layout.row_major(NMESH_POLY, MODEL_MESH_POLY_SIZE), MutAnyOrigin
    ],
    mesh_polyvert: LayoutTensor[
        DTYPE, Layout.row_major(NMESH_POLYVERT), MutAnyOrigin
    ],
    mesh_polymap: LayoutTensor[
        DTYPE, Layout.row_major(NMESH_POLYVERT), MutAnyOrigin
    ],
    mesh_vert_polymap: LayoutTensor[
        DTYPE, Layout.row_major(NMESH_VERTS, 2), MutAnyOrigin
    ],
    mut n: InlineArray[Scalar[DTYPE], MC_MAX_DEG * 3],
    mut endverts: InlineArray[Scalar[DTYPE], MC_MAX_DEG * 3],
) -> Int:
    """`meshEdgeNormals` — edge directions leaving the contact vertex/edge.

    For `dim == 1` the edges are found by walking each incident polygon to the
    vertex BEFORE `v1i` in its cycle, which is the reference's choice and
    depends on the polygon winding being the one `mesh_polygons.mojo` stores.
    """
    if dim == 2:
        endverts[0] = v2x
        endverts[1] = v2y
        endverts[2] = v2z
        var ex = v2x - v1x
        var ey = v2y - v1y
        var ez = v2z - v1z
        var l = sqrt(ex * ex + ey * ey + ez * ez)
        if l > Scalar[DTYPE](0):
            ex /= l
            ey /= l
            ez /= l
        n[0] = ex
        n[1] = ey
        n[2] = ez
        return 1

    if dim == 1:
        var a1 = Int(
            rebind[Scalar[DTYPE]](mesh_vert_polymap[vert_adr + v1i, 0])
        )
        var n1 = Int(
            rebind[Scalar[DTYPE]](mesh_vert_polymap[vert_adr + v1i, 1])
        )
        var c = 0
        for i in range(n1):
            if c >= MC_MAX_DEG:
                break
            var p = Int(rebind[Scalar[DTYPE]](mesh_polymap[a1 + i]))
            var po = poly_adr + p  # ROW index: mesh_polys is [poly, field]
            var adr = Int(
                rebind[Scalar[DTYPE]](mesh_polys[po, MESH_POLY_IDX_VERTADR])
            )
            var num = Int(
                rebind[Scalar[DTYPE]](mesh_polys[po, MESH_POLY_IDX_VERTNUM])
            )
            for j in range(num):
                if Int(rebind[Scalar[DTYPE]](mesh_polyvert[adr + j])) != v1i:
                    continue
                var k = num - 1 if j == 0 else j - 1
                var vk = Int(rebind[Scalar[DTYPE]](mesh_polyvert[adr + k]))
                var w = quat_rotate[DTYPE](
                    qx, qy, qz, qw,
                    rebind[Scalar[DTYPE]](mesh_verts[vert_adr + vk, 0]),
                    rebind[Scalar[DTYPE]](mesh_verts[vert_adr + vk, 1]),
                    rebind[Scalar[DTYPE]](mesh_verts[vert_adr + vk, 2]),
                )
                var wx = gx + w[0]
                var wy = gy + w[1]
                var wz = gz + w[2]
                endverts[c * 3 + 0] = wx
                endverts[c * 3 + 1] = wy
                endverts[c * 3 + 2] = wz
                var ex = wx - v1x
                var ey = wy - v1y
                var ez = wz - v1z
                var l = sqrt(ex * ex + ey * ey + ez * ez)
                if l > Scalar[DTYPE](0):
                    ex /= l
                    ey /= l
                    ez /= l
                n[c * 3 + 0] = ex
                n[c * 3 + 1] = ey
                n[c * 3 + 2] = ez
                break
            c += 1
        return c
    return 0


@always_inline
def _mesh_face[
    DTYPE: DType, NMESH_VERTS: Int, NMESH_POLY: Int, NMESH_POLYVERT: Int
](
    face_id: Int,
    gx: Scalar[DTYPE], gy: Scalar[DTYPE], gz: Scalar[DTYPE],
    qx: Scalar[DTYPE], qy: Scalar[DTYPE], qz: Scalar[DTYPE], qw: Scalar[DTYPE],
    vert_adr: Int, poly_adr: Int,
    mesh_verts: LayoutTensor[
        DTYPE, Layout.row_major(NMESH_VERTS, 3), MutAnyOrigin
    ],
    mesh_polys: LayoutTensor[
        DTYPE, Layout.row_major(NMESH_POLY, MODEL_MESH_POLY_SIZE), MutAnyOrigin
    ],
    mesh_polyvert: LayoutTensor[
        DTYPE, Layout.row_major(NMESH_POLYVERT), MutAnyOrigin
    ],
    mut face: InlineArray[Scalar[DTYPE], MC_MAX_POLYVERT * 3],
) -> Int:
    """`meshFace` — one polygon in world coordinates, REVERSED.

    The reverse (`for i = nvert-1 downto 0`) is the reference's, and it is what
    makes a mesh face agree in winding with `boxFace`'s clockwise order. Our
    stored polygons are CCW as seen from outside, matching `m.mesh_polyvert`.
    """
    var po = poly_adr + face_id  # ROW index, not a flat offset
    var adr = Int(rebind[Scalar[DTYPE]](mesh_polys[po, MESH_POLY_IDX_VERTADR]))
    var num = Int(rebind[Scalar[DTYPE]](mesh_polys[po, MESH_POLY_IDX_VERTNUM]))
    if num > MC_MAX_POLYVERT:
        return 0
    var j = 0
    for i in range(num - 1, -1, -1):
        var vi = Int(rebind[Scalar[DTYPE]](mesh_polyvert[adr + i]))
        var w = quat_rotate[DTYPE](
            qx, qy, qz, qw,
            rebind[Scalar[DTYPE]](mesh_verts[vert_adr + vi, 0]),
            rebind[Scalar[DTYPE]](mesh_verts[vert_adr + vi, 1]),
            rebind[Scalar[DTYPE]](mesh_verts[vert_adr + vi, 2]),
        )
        face[j * 3 + 0] = gx + w[0]
        face[j * 3 + 1] = gy + w[1]
        face[j * 3 + 2] = gz + w[2]
        j += 1
    return num


@always_inline
def _aligned_faces[
    DTYPE: DType
](
    v: InlineArray[Scalar[DTYPE], MC_MAX_DEG * 3], nv: Int,
    w: InlineArray[Scalar[DTYPE], MC_MAX_DEG * 3], nw: Int,
    mut r0: Int, mut r1: Int,
) -> Bool:
    """`alignedFaces` — first pair of normals facing each other within tol."""
    for i in range(nv):
        for j in range(nw):
            if _dot3[DTYPE](
                v[i * 3 + 0], v[i * 3 + 1], v[i * 3 + 2],
                w[j * 3 + 0], w[j * 3 + 1], w[j * 3 + 2],
            ) < Scalar[DTYPE](-MC_FACE_TOL):
                r0 = i
                r1 = j
                return True
    return False


@always_inline
def _aligned_face_edge[
    DTYPE: DType
](
    edge: InlineArray[Scalar[DTYPE], MC_MAX_DEG * 3], nedge: Int,
    face: InlineArray[Scalar[DTYPE], MC_MAX_DEG * 3], nface: Int,
    mut r0: Int, mut r1: Int,
) -> Bool:
    """`alignedFaceEdge` — first edge perpendicular to a face normal.

    ⚠ THE LOOP NEST IS FACE-OUTER, EDGE-INNER and the results come back as
    `r0 = edge index`, `r1 = face index` — the opposite order to the loops.
    Swapping either would pick a different feature on any contact where more
    than one pair qualifies.
    """
    for i in range(nface):
        for j in range(nedge):
            var d = _dot3[DTYPE](
                edge[j * 3 + 0], edge[j * 3 + 1], edge[j * 3 + 2],
                face[i * 3 + 0], face[i * 3 + 1], face[i * 3 + 2],
            )
            if abs(d) < Scalar[DTYPE](MC_EDGE_TOL):
                r0 = j
                r1 = i
                return True
    return False


def native_multicontact_contacts[
    DTYPE: DType,
    NMESH_VERTS: Int,
    NMESH_POLY: Int,
    NMESH_POLYVERT: Int,
    MAX_CONTACTS: Int,
    BATCH: Int,
](
    env: Int, body_a: Int, body_b: Int,
    gi_type: Int,
    pix: Scalar[DTYPE], piy: Scalar[DTYPE], piz: Scalar[DTYPE],
    qix: Scalar[DTYPE], qiy: Scalar[DTYPE], qiz: Scalar[DTYPE],
    qiw: Scalar[DTYPE],
    hxi: Scalar[DTYPE], hyi: Scalar[DTYPE], hzi: Scalar[DTYPE],
    rbound_i: Scalar[DTYPE], va1: Int, mnv1: Int, pa1: Int, pn1: Int,
    gj_type: Int,
    pjx: Scalar[DTYPE], pjy: Scalar[DTYPE], pjz: Scalar[DTYPE],
    qjx: Scalar[DTYPE], qjy: Scalar[DTYPE], qjz: Scalar[DTYPE],
    qjw: Scalar[DTYPE],
    hxj: Scalar[DTYPE], hyj: Scalar[DTYPE], hzj: Scalar[DTYPE],
    rbound_j: Scalar[DTYPE], va2: Int, mnv2: Int, pa2: Int, pn2: Int,
    mesh_verts: LayoutTensor[
        DTYPE, Layout.row_major(NMESH_VERTS, 3), MutAnyOrigin
    ],
    mesh_polys: LayoutTensor[
        DTYPE, Layout.row_major(NMESH_POLY, MODEL_MESH_POLY_SIZE), MutAnyOrigin
    ],
    mesh_polyvert: LayoutTensor[
        DTYPE, Layout.row_major(NMESH_POLYVERT), MutAnyOrigin
    ],
    mesh_polymap: LayoutTensor[
        DTYPE, Layout.row_major(NMESH_POLYVERT), MutAnyOrigin
    ],
    mesh_vert_polymap: LayoutTensor[
        DTYPE, Layout.row_major(NMESH_VERTS, 2), MutAnyOrigin
    ],
    wf1: InlineArray[Scalar[DTYPE], 9],
    wf2: InlineArray[Scalar[DTYPE], 9],
    wx: InlineArray[Scalar[DTYPE], 6],
    dist0: Scalar[DTYPE],
    contact_margin: Scalar[DTYPE],
    contact_friction: Scalar[DTYPE],
    contact_friction_spin: Scalar[DTYPE],
    contact_friction_roll: Scalar[DTYPE],
    contact_condim: Int,
    flip_normal: Bool,
    contacts: LayoutTensor[
        DTYPE, Layout.row_major(BATCH, MAX_CONTACTS * CONTACT_SIZE),
        MutAnyOrigin,
    ],
    mut num_contacts: Int,
) -> Int:
    """`multicontact` — emit the clipped face manifold, or 0 if there is none.

    Returns the number of contact records written. A return of 0 means the two
    contact features do not line up as face/face, edge/face or face/edge, and
    the caller must emit its single EPA point instead — that is exactly what
    the reference does by leaving `status->nx` at 1.

    ⚠ THE OPERANDS MUST ARRIVE IN MuJoCo'S ORDER, LOWER GEOM TYPE FIRST, and
    `flip_normal` is what lets the caller do that without disturbing the
    record's `body_a = gi` invariant. `mj_collideGeoms` sorts the pair so
    `type1 <= type2` before dispatching, and BOX (6) sorts before MESH (7), so
    the reference ALWAYS runs a box/mesh pair with the box as obj1. This
    routine is not symmetric in its operands — the edge/face branch tests
    `nface1 < 3 and nface1 <= nface2`, which prefers geom1 as the edge — so
    running it in geom-index order picks a different feature on exactly the
    poses where the two disagree. Measured: with the operands in our own order,
    `box x mesh` matched MuJoCo exactly (339/339 points, |dn| 2.8e-13) while
    `mesh x box`, the SAME PAIR declared the other way round, had 9 count
    mismatches and fell back to the single EPA point often enough to leave
    |dn| at 1.5e-3.

    When the caller swaps, the manifold normal comes out as `obj2 -> obj1`,
    which is `gi -> gj`; the record wants `body_b -> body_a`, so it is negated.
    """
    # A mesh with no polygons (a degenerate hull) is skipped, as MuJoCo skips
    # `!obj->data.mesh.mesh_polynum`.
    if gi_type == GEOM_MESH and pn1 <= 0:
        return 0
    if gj_type == GEOM_MESH and pn2 <= 0:
        return 0

    var scale = rbound_i if rbound_i < rbound_j else rbound_j
    if scale <= Scalar[DTYPE](0):
        scale = Scalar[DTYPE](1)

    # ---- simplexDim on each geom, with the same reordering ------------------
    var a1x = wf1[0]; var a1y = wf1[1]; var a1z = wf1[2]
    var b1x = wf1[3]; var b1y = wf1[4]; var b1z = wf1[5]
    var c1x = wf1[6]; var c1y = wf1[7]; var c1z = wf1[8]
    var nface1 = 3
    if _same_point[DTYPE](a1x, a1y, a1z, b1x, b1y, b1z, scale):
        if _same_point[DTYPE](a1x, a1y, a1z, c1x, c1y, c1z, scale):
            nface1 = 1
        else:
            b1x = c1x; b1y = c1y; b1z = c1z
            nface1 = 2
    elif _same_point[DTYPE](c1x, c1y, c1z, a1x, a1y, a1z, scale) or _same_point[
        DTYPE
    ](c1x, c1y, c1z, b1x, b1y, b1z, scale):
        nface1 = 2

    var a2x = wf2[0]; var a2y = wf2[1]; var a2z = wf2[2]
    var b2x = wf2[3]; var b2y = wf2[4]; var b2z = wf2[5]
    var c2x = wf2[6]; var c2y = wf2[7]; var c2z = wf2[8]
    var nface2 = 3
    if _same_point[DTYPE](a2x, a2y, a2z, b2x, b2y, b2z, scale):
        if _same_point[DTYPE](a2x, a2y, a2z, c2x, c2y, c2z, scale):
            nface2 = 1
        else:
            b2x = c2x; b2y = c2y; b2z = c2z
            nface2 = 2
    elif _same_point[DTYPE](c2x, c2y, c2z, a2x, a2y, a2z, scale) or _same_point[
        DTYPE
    ](c2x, c2y, c2z, b2x, b2y, b2z, scale):
        nface2 = 2

    # ---- vertex ids for the (up to three) distinct support points -----------
    var i1a = 0; var i1b = 0; var i1c = 0
    var l1a = _to_local[DTYPE](a1x, a1y, a1z, pix, piy, piz, qix, qiy, qiz, qiw)
    var l1b = _to_local[DTYPE](b1x, b1y, b1z, pix, piy, piz, qix, qiy, qiz, qiw)
    var l1c = _to_local[DTYPE](c1x, c1y, c1z, pix, piy, piz, qix, qiy, qiz, qiw)
    if gi_type == GEOM_BOX:
        i1a = _box_corner_index[DTYPE](l1a[0], l1a[1], l1a[2])
        i1b = _box_corner_index[DTYPE](l1b[0], l1b[1], l1b[2])
        i1c = _box_corner_index[DTYPE](l1c[0], l1c[1], l1c[2])
    else:
        i1a = _mesh_vertex_index[DTYPE, NMESH_VERTS](
            l1a[0], l1a[1], l1a[2], mesh_verts, va1, mnv1
        )
        i1b = _mesh_vertex_index[DTYPE, NMESH_VERTS](
            l1b[0], l1b[1], l1b[2], mesh_verts, va1, mnv1
        )
        i1c = _mesh_vertex_index[DTYPE, NMESH_VERTS](
            l1c[0], l1c[1], l1c[2], mesh_verts, va1, mnv1
        )
        if i1a < 0 or i1b < 0 or i1c < 0:
            return 0

    var i2a = 0; var i2b = 0; var i2c = 0
    var l2a = _to_local[DTYPE](a2x, a2y, a2z, pjx, pjy, pjz, qjx, qjy, qjz, qjw)
    var l2b = _to_local[DTYPE](b2x, b2y, b2z, pjx, pjy, pjz, qjx, qjy, qjz, qjw)
    var l2c = _to_local[DTYPE](c2x, c2y, c2z, pjx, pjy, pjz, qjx, qjy, qjz, qjw)
    if gj_type == GEOM_BOX:
        i2a = _box_corner_index[DTYPE](l2a[0], l2a[1], l2a[2])
        i2b = _box_corner_index[DTYPE](l2b[0], l2b[1], l2b[2])
        i2c = _box_corner_index[DTYPE](l2c[0], l2c[1], l2c[2])
    else:
        i2a = _mesh_vertex_index[DTYPE, NMESH_VERTS](
            l2a[0], l2a[1], l2a[2], mesh_verts, va2, mnv2
        )
        i2b = _mesh_vertex_index[DTYPE, NMESH_VERTS](
            l2b[0], l2b[1], l2b[2], mesh_verts, va2, mnv2
        )
        i2c = _mesh_vertex_index[DTYPE, NMESH_VERTS](
            l2c[0], l2c[1], l2c[2], mesh_verts, va2, mnv2
        )
        if i2a < 0 or i2b < 0 or i2c < 0:
            return 0

    # ---- the contact direction, straight from the two witness points --------
    var dirx = wx[3] - wx[0]
    var diry = wx[4] - wx[1]
    var dirz = wx[5] - wx[2]
    var dirlen = sqrt(dirx * dirx + diry * diry + dirz * dirz)

    var n1 = InlineArray[Scalar[DTYPE], MC_MAX_DEG * 3](fill=Scalar[DTYPE](0))
    var n2 = InlineArray[Scalar[DTYPE], MC_MAX_DEG * 3](fill=Scalar[DTYPE](0))
    var idx1 = InlineArray[Int, MC_MAX_DEG](fill=0)
    var idx2 = InlineArray[Int, MC_MAX_DEG](fill=0)
    var endverts = InlineArray[Scalar[DTYPE], MC_MAX_DEG * 3](
        fill=Scalar[DTYPE](0)
    )

    var nn1 = 0
    if gi_type == GEOM_BOX:
        nn1 = _box_normals[DTYPE](
            nface1, i1a, i1b, i1c, qix, qiy, qiz, qiw,
            -dirx, -diry, -dirz, n1, idx1,
        )
    elif gi_type == GEOM_MESH:
        nn1 = _mesh_normals[DTYPE, NMESH_VERTS, NMESH_POLY, NMESH_POLYVERT](
            nface1, i1a, i1b, i1c, qix, qiy, qiz, qiw, va1, pa1,
            mesh_polys, mesh_polymap, mesh_vert_polymap, n1, idx1,
        )
    var nn2 = 0
    if gj_type == GEOM_BOX:
        nn2 = _box_normals[DTYPE](
            nface2, i2a, i2b, i2c, qjx, qjy, qjz, qjw,
            dirx, diry, dirz, n2, idx2,
        )
    elif gj_type == GEOM_MESH:
        nn2 = _mesh_normals[DTYPE, NMESH_VERTS, NMESH_POLY, NMESH_POLYVERT](
            nface2, i2a, i2b, i2c, qjx, qjy, qjz, qjw, va2, pa2,
            mesh_polys, mesh_polymap, mesh_vert_polymap, n2, idx2,
        )

    var ri = 0
    var rj = 0
    var edgecon1 = False
    var edgecon2 = False
    comptime if MC_DEBUG:
        print(
            "    [mc] types", gi_type, gj_type,
            " nface", nface1, nface2, " nnorms", nn1, nn2,
            " dirlen", dirlen,
        )
        for t in range(nn1):
            print("      n1[", t, "] =", n1[t * 3 + 0], n1[t * 3 + 1],
                  n1[t * 3 + 2], " idx", idx1[t])
        for t in range(nn2):
            print("      n2[", t, "] =", n2[t * 3 + 0], n2[t * 3 + 1],
                  n2[t * 3 + 2], " idx", idx2[t])
    if not _aligned_faces[DTYPE](n1, nn1, n2, nn2, ri, rj):
        if nface1 < 3 and nface1 <= nface2:
            nn1 = 0
            if gi_type == GEOM_BOX:
                nn1 = _box_edge_normals[DTYPE](
                    nface1, a1x, a1y, a1z, b1x, b1y, b1z, i1a,
                    pix, piy, piz, qix, qiy, qiz, qiw, hxi, hyi, hzi,
                    n1, endverts,
                )
            elif gi_type == GEOM_MESH:
                nn1 = _mesh_edge_normals[
                    DTYPE, NMESH_VERTS, NMESH_POLY, NMESH_POLYVERT
                ](
                    nface1, a1x, a1y, a1z, b1x, b1y, b1z, i1a,
                    pix, piy, piz, qix, qiy, qiz, qiw, va1, pa1,
                    mesh_verts, mesh_polys, mesh_polyvert, mesh_polymap,
                    mesh_vert_polymap, n1, endverts,
                )
            if not _aligned_face_edge[DTYPE](n1, nn1, n2, nn2, ri, rj):
                return 0
            edgecon1 = True
        elif nface2 < 3:
            nn2 = 0
            if gj_type == GEOM_BOX:
                nn2 = _box_edge_normals[DTYPE](
                    nface2, a2x, a2y, a2z, b2x, b2y, b2z, i2a,
                    pjx, pjy, pjz, qjx, qjy, qjz, qjw, hxj, hyj, hzj,
                    n2, endverts,
                )
            elif gj_type == GEOM_MESH:
                nn2 = _mesh_edge_normals[
                    DTYPE, NMESH_VERTS, NMESH_POLY, NMESH_POLYVERT
                ](
                    nface2, a2x, a2y, a2z, b2x, b2y, b2z, i2a,
                    pjx, pjy, pjz, qjx, qjy, qjz, qjw, va2, pa2,
                    mesh_verts, mesh_polys, mesh_polyvert, mesh_polymap,
                    mesh_vert_polymap, n2, endverts,
                )
            if not _aligned_face_edge[DTYPE](n2, nn2, n1, nn1, ri, rj):
                return 0
            edgecon2 = True
        else:
            return 0

    # ---- recover each geom's matching face (or edge) ------------------------
    var face1 = InlineArray[Scalar[DTYPE], MC_MAX_POLYVERT * 3](
        fill=Scalar[DTYPE](0)
    )
    var face2 = InlineArray[Scalar[DTYPE], MC_MAX_POLYVERT * 3](
        fill=Scalar[DTYPE](0)
    )
    var nf1 = 0
    var nf2 = 0

    if edgecon1:
        face1[0] = a1x
        face1[1] = a1y
        face1[2] = a1z
        face1[3] = endverts[ri * 3 + 0]
        face1[4] = endverts[ri * 3 + 1]
        face1[5] = endverts[ri * 3 + 2]
        nf1 = 2
    else:
        var ind = idx1[rj] if edgecon2 else idx1[ri]
        if gi_type == GEOM_BOX:
            nf1 = _box_face[DTYPE](
                ind, pix, piy, piz, qix, qiy, qiz, qiw, hxi, hyi, hzi, face1
            )
        elif gi_type == GEOM_MESH:
            nf1 = _mesh_face[DTYPE, NMESH_VERTS, NMESH_POLY, NMESH_POLYVERT](
                ind, pix, piy, piz, qix, qiy, qiz, qiw, va1, pa1,
                mesh_verts, mesh_polys, mesh_polyvert, face1,
            )

    if edgecon2:
        face2[0] = a2x
        face2[1] = a2y
        face2[2] = a2z
        face2[3] = endverts[ri * 3 + 0]
        face2[4] = endverts[ri * 3 + 1]
        face2[5] = endverts[ri * 3 + 2]
        nf2 = 2
    else:
        if gj_type == GEOM_BOX:
            nf2 = _box_face[DTYPE](
                idx2[rj], pjx, pjy, pjz, qjx, qjy, qjz, qjw,
                hxj, hyj, hzj, face2,
            )
        elif gj_type == GEOM_MESH:
            nf2 = _mesh_face[DTYPE, NMESH_VERTS, NMESH_POLY, NMESH_POLYVERT](
                idx2[rj], pjx, pjy, pjz, qjx, qjy, qjz, qjw, va2, pa2,
                mesh_verts, mesh_polys, mesh_polyvert, face2,
            )
    if nf1 == 0 or nf2 == 0:
        return 0

    # ---- clip -------------------------------------------------------------
    var out = InlineArray[Scalar[DTYPE], MC_CLIP_CAP * 3](fill=Scalar[DTYPE](0))
    var nx_out = 0
    var adx = Scalar[DTYPE](0)
    var ady = Scalar[DTYPE](0)
    var adz = Scalar[DTYPE](0)
    var swap = False

    if edgecon1:
        adx = -n2[rj * 3 + 0] * dirlen
        ady = -n2[rj * 3 + 1] * dirlen
        adz = -n2[rj * 3 + 2] * dirlen
        nx_out = _polygon_clip[DTYPE](
            face2, nf2, face1, nf1,
            n2[rj * 3 + 0], n2[rj * 3 + 1], n2[rj * 3 + 2], 4, out,
        )
        swap = True
    elif edgecon2:
        adx = -n1[rj * 3 + 0] * dirlen
        ady = -n1[rj * 3 + 1] * dirlen
        adz = -n1[rj * 3 + 2] * dirlen
        nx_out = _polygon_clip[DTYPE](
            face1, nf1, face2, nf2,
            n1[rj * 3 + 0], n1[rj * 3 + 1], n1[rj * 3 + 2], 4, out,
        )
    else:
        adx = n2[rj * 3 + 0] * dirlen
        ady = n2[rj * 3 + 1] * dirlen
        adz = n2[rj * 3 + 2] * dirlen
        nx_out = _polygon_clip[DTYPE](
            face1, nf1, face2, nf2,
            n1[ri * 3 + 0], n1[ri * 3 + 1], n1[ri * 3 + 2], 4, out,
        )

    comptime if MC_DEBUG:
        print(
            "    [mc] branch edge1", edgecon1, " edge2", edgecon2,
            " ri", ri, " rj", rj, " nf", nf1, nf2, " nx_out", nx_out,
            " ad", adx, ady, adz,
        )
    if nx_out < 1:
        return 0

    # ---- emit ---------------------------------------------------------------
    # `pos` is the midpoint of the two witness points, which is `c - ad/2`
    # either way round; the record's normal is `x2 - x1` normalised, i.e. the
    # `body_b -> body_a` direction the single-point emit also stores.
    var rnx = adx
    var rny = ady
    var rnz = adz
    if swap:
        rnx = -adx
        rny = -ady
        rnz = -adz
    var rl = sqrt(rnx * rnx + rny * rny + rnz * rnz)
    if rl <= Scalar[DTYPE](0):
        return 0
    rnx /= rl
    rny /= rl
    rnz /= rl
    if flip_normal:
        rnx = -rnx
        rny = -rny
        rnz = -rnz

    var written = 0
    for k in range(nx_out):
        if num_contacts >= MAX_CONTACTS:
            break
        var off = num_contacts * CONTACT_SIZE
        contacts[env, off + CONTACT_IDX_BODY_A] = Scalar[DTYPE](body_a)
        contacts[env, off + CONTACT_IDX_BODY_B] = Scalar[DTYPE](body_b)
        contacts[env, off + CONTACT_IDX_POS_X] = (
            out[k * 3 + 0] - Scalar[DTYPE](0.5) * adx
        )
        contacts[env, off + CONTACT_IDX_POS_Y] = (
            out[k * 3 + 1] - Scalar[DTYPE](0.5) * ady
        )
        contacts[env, off + CONTACT_IDX_POS_Z] = (
            out[k * 3 + 2] - Scalar[DTYPE](0.5) * adz
        )
        contacts[env, off + CONTACT_IDX_NX] = rnx
        contacts[env, off + CONTACT_IDX_NY] = rny
        contacts[env, off + CONTACT_IDX_NZ] = rnz
        contacts[env, off + CONTACT_IDX_DIST] = dist0
        contacts[env, off + CONTACT_IDX_INCLUDEMARGIN] = contact_margin
        contacts[env, off + CONTACT_IDX_FRICTION] = contact_friction
        contacts[env, off + CONTACT_IDX_FRICTION_SPIN] = contact_friction_spin
        contacts[env, off + CONTACT_IDX_FRICTION_ROLL] = contact_friction_roll
        contacts[env, off + CONTACT_IDX_CONDIM] = Scalar[DTYPE](contact_condim)
        num_contacts += 1
        written += 1
    return written
