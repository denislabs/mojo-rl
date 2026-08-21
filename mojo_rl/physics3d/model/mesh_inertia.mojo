"""Mesh volume, centre of mass and principal axes — MuJoCo's `mjCMesh::Compute`.

`inertia_from_geom.geom_inertia` covers sphere/capsule/cylinder/box/ellipsoid
and returned **(0, 0, 0)** for `GEOM_MESH`, silently. That is fine for a model
whose mesh-bearing bodies carry an explicit `<inertial>` (dog: 162 meshes, every
body with its own inertial) and catastrophic for one that does not (Jaco: every
link is a mesh geom with `mass=` and no `<inertial>` at all, so all 14 of its
non-trivial bodies would have had zero inertia).

## The algorithm, and why it is not the textbook one

MuJoCo's default is **`inertia="legacy"`** (`mjs_defaultMesh`, `user_init.c`;
same in the 3.6.0 and 3.11.0 trees, confirmed by measurement on the 3.10.0
runtime). Legacy differs from the standard divergence-theorem integral in two
ways that both matter:

  1. ⚠ **The apex is `facecen`, not the origin.** Each face contributes a
     pyramid from the AREA-WEIGHTED CENTROID OF FACE CENTRES, so the per-face
     volume is `dot(center - facecen, normal) * area / 3` and the pyramid's
     centroid is `3/4*center + 1/4*facecen`.
  2. ⚠ **Legacy takes `abs()` of every per-face volume.** A closed, correctly
     oriented mesh has all-positive contributions anyway; a mesh with inverted
     or interior faces does not, and the `abs` is what makes legacy tolerant of
     them. It also makes the volume differ from the true enclosed volume.

Measured on Jaco's `base` mesh, all three plausible readings:

    raw triangles, origin apex   com z = 0.08845968   (off by 6.7e-03)
    convex hull,   origin apex   com z = 0.08047642   (off by 1.5e-03)
    LEGACY (this file)           com z = 0.0817722373 (MuJoCo, to 2.8e-17)

So neither "integrate the triangle soup" nor "integrate the hull" reproduces
MuJoCo, and both are wrong by amounts that would look like a plausible small
discrepancy rather than a bug. All 9 Jaco meshes match to **2.8e-17 on the
centre of mass and 4.9e-14 on the principal quaternion**.

⚠ **The faces are the RAW triangles, not the convex hull.** `mjMESH_INERTIA_
CONVEX` is a different, non-default mode that uses `GraphFaces()`. Feeding the
hull here is exactly the 1.5e-03 error above.

## Units

The returned `eigval` is **unitless** — a volume-weighted second moment with
units of length^5. Multiply by density (`mass / volume`) to get a real inertia.
That is MuJoCo's own convention and it is why the same mesh can be reused at
different masses without recomputing anything.

## The frame

`com` and `quat` are the transform MuJoCo BAKES INTO THE MESH: it translates the
vertices by `-com` and rotates them by the conjugate of `quat`, then records the
pair as `mesh_pos` / `mesh_quat` and composes them into every geom that
references the mesh. So a mesh geom's final frame is
`declared_geom_frame o (com, quat)`, and in that frame the mesh's inertia is
diagonal — which is what lets the body-inertia assembly treat a mesh geom
exactly like a box.

⚠ The vertex transform is `v' = R(quat)^T (v - com)`, i.e. rotate by the
CONJUGATE. Verified against `mjModel.mesh_vert`: the conjugate reproduces
MuJoCo's stored bounding box to 2.8e-09 (float32 storage), the non-conjugate is
off by 1.5e-02. Getting this backwards leaves the mesh a plausible-looking
rotated copy of itself.
"""

from std.math import sqrt, abs as math_abs

from .inertia_from_geom import eig3_symmetric, quat_to_mat


struct MeshInertia[DTYPE: DType](Copyable, ImplicitlyCopyable, Movable):
    """`mjCMesh`'s computed frame + unitless principal moments."""

    var volume: Scalar[Self.DTYPE]
    var com_x: Scalar[Self.DTYPE]
    var com_y: Scalar[Self.DTYPE]
    var com_z: Scalar[Self.DTYPE]
    # UNITLESS (length^5). Scale by `mass / volume`.
    var eig0: Scalar[Self.DTYPE]
    var eig1: Scalar[Self.DTYPE]
    var eig2: Scalar[Self.DTYPE]
    # Principal-axis rotation, OUR (x, y, z, w) order — MuJoCo stores (w,x,y,z).
    var qx: Scalar[Self.DTYPE]
    var qy: Scalar[Self.DTYPE]
    var qz: Scalar[Self.DTYPE]
    var qw: Scalar[Self.DTYPE]

    def __init__(out self):
        self.volume = Scalar[Self.DTYPE](0)
        self.com_x = Scalar[Self.DTYPE](0)
        self.com_y = Scalar[Self.DTYPE](0)
        self.com_z = Scalar[Self.DTYPE](0)
        self.eig0 = Scalar[Self.DTYPE](0)
        self.eig1 = Scalar[Self.DTYPE](0)
        self.eig2 = Scalar[Self.DTYPE](0)
        self.qx = Scalar[Self.DTYPE](0)
        self.qy = Scalar[Self.DTYPE](0)
        self.qz = Scalar[Self.DTYPE](0)
        self.qw = Scalar[Self.DTYPE](1)


@always_inline
def _tri_area_center_normal[
    DTYPE: DType
](
    ax: Scalar[DTYPE], ay: Scalar[DTYPE], az: Scalar[DTYPE],
    bx: Scalar[DTYPE], by: Scalar[DTYPE], bz: Scalar[DTYPE],
    cx: Scalar[DTYPE], cy: Scalar[DTYPE], cz: Scalar[DTYPE],
) -> InlineArray[Scalar[DTYPE], 7]:
    """MuJoCo's static `triangle()`: area, centroid, UNIT normal.

    Returns [area, cenx, ceny, cenz, nx, ny, nz]; area 0 (and a zero normal)
    for a degenerate face, which MuJoCo skips by contributing nothing.
    """
    var out = InlineArray[Scalar[DTYPE], 7](fill=Scalar[DTYPE](0))
    out[1] = (ax + bx + cx) / Scalar[DTYPE](3)
    out[2] = (ay + by + cy) / Scalar[DTYPE](3)
    out[3] = (az + bz + cz) / Scalar[DTYPE](3)

    var e1x = bx - ax
    var e1y = by - ay
    var e1z = bz - az
    var e2x = cx - ax
    var e2y = cy - ay
    var e2z = cz - az
    var nx = e1y * e2z - e1z * e2y
    var ny = e1z * e2x - e1x * e2z
    var nz = e1x * e2y - e1y * e2x
    var ln = sqrt(nx * nx + ny * ny + nz * nz)
    # mjMINVAL — a face shorter than this contributes nothing at all.
    if ln >= Scalar[DTYPE](1e-15):
        out[0] = Scalar[DTYPE](0.5) * ln
        out[4] = nx / ln
        out[5] = ny / ln
        out[6] = nz / ln
    return out^


def mesh_legacy_inertia[
    DTYPE: DType
](tri_verts: List[Scalar[DTYPE]], num_tris: Int) -> MeshInertia[DTYPE]:
    """`mjCMesh::Compute` for `inertia="legacy"` — MuJoCo's default.

    `tri_verts` is the RAW triangle soup, 9 scalars per face (three xyz corners
    in winding order) — i.e. exactly what an STL yields before deduplication.
    Deduplicating first is harmless (it changes indices, not geometry) but the
    faces must be ALL of them, not the convex hull's.
    """
    var mi = MeshInertia[DTYPE]()
    if num_tris <= 0:
        return mi

    # ── pass 1: facecen, the area-weighted centroid of face CENTRES ─────────
    var fx = Scalar[DTYPE](0)
    var fy = Scalar[DTYPE](0)
    var fz = Scalar[DTYPE](0)
    var total_area = Scalar[DTYPE](0)
    for t in range(num_tris):
        var o = t * 9
        var r = _tri_area_center_normal[DTYPE](
            tri_verts[o + 0], tri_verts[o + 1], tri_verts[o + 2],
            tri_verts[o + 3], tri_verts[o + 4], tri_verts[o + 5],
            tri_verts[o + 6], tri_verts[o + 7], tri_verts[o + 8],
        )
        fx += r[0] * r[1]
        fy += r[0] * r[2]
        fz += r[0] * r[3]
        total_area += r[0]
    if total_area < Scalar[DTYPE](1e-15):
        return mi
    fx /= total_area
    fy /= total_area
    fz /= total_area

    # ── pass 2: volume + centre of mass, apex at facecen ────────────────────
    var vol_total = Scalar[DTYPE](0)
    var cx = Scalar[DTYPE](0)
    var cy = Scalar[DTYPE](0)
    var cz = Scalar[DTYPE](0)
    for t in range(num_tris):
        var o = t * 9
        var r = _tri_area_center_normal[DTYPE](
            tri_verts[o + 0], tri_verts[o + 1], tri_verts[o + 2],
            tri_verts[o + 3], tri_verts[o + 4], tri_verts[o + 5],
            tri_verts[o + 6], tri_verts[o + 7], tri_verts[o + 8],
        )
        var dot = (
            (r[1] - fx) * r[4] + (r[2] - fy) * r[5] + (r[3] - fz) * r[6]
        )
        # ⚠ LEGACY takes the absolute value per face.
        var v = math_abs(dot * r[0] / Scalar[DTYPE](3))
        vol_total += v
        cx += v * (r[1] * Scalar[DTYPE](0.75) + fx * Scalar[DTYPE](0.25))
        cy += v * (r[2] * Scalar[DTYPE](0.75) + fy * Scalar[DTYPE](0.25))
        cz += v * (r[3] * Scalar[DTYPE](0.75) + fz * Scalar[DTYPE](0.25))
    if vol_total < Scalar[DTYPE](1e-15):
        return mi
    cx /= vol_total
    cy /= vol_total
    cz /= vol_total

    # ── pass 3: products of inertia about the centre of mass ────────────────
    #
    # ⚠ The volume is RECOMPUTED here, with the apex at the (now origin) centre
    # of mass rather than at facecen, and THIS value is the one MuJoCo keeps —
    # `volume_ = total_volume` overwrites the pass-2 result. Measured on Jaco's
    # `base`: pass 2 gives 0.0008078487, pass 3 gives 0.00080755597, and only
    # the latter reproduces MuJoCo's `body_inertia` (density 889.481). Keeping
    # the pass-2 number puts a 3.6e-04 relative error into every density —
    # small enough to look like rounding.
    var p0 = Scalar[DTYPE](0)
    var p1 = Scalar[DTYPE](0)
    var p2 = Scalar[DTYPE](0)
    var p3 = Scalar[DTYPE](0)
    var p4 = Scalar[DTYPE](0)
    var p5 = Scalar[DTYPE](0)
    var vol_recomputed = Scalar[DTYPE](0)
    for t in range(num_tris):
        var o = t * 9
        var dx = tri_verts[o + 0] - cx
        var dy = tri_verts[o + 1] - cy
        var dz = tri_verts[o + 2] - cz
        var ex = tri_verts[o + 3] - cx
        var ey = tri_verts[o + 4] - cy
        var ez = tri_verts[o + 5] - cz
        var gx = tri_verts[o + 6] - cx
        var gy = tri_verts[o + 7] - cy
        var gz = tri_verts[o + 8] - cz
        var r = _tri_area_center_normal[DTYPE](
            dx, dy, dz, ex, ey, ez, gx, gy, gz
        )
        var dot = r[1] * r[4] + r[2] * r[5] + r[3] * r[6]
        var v = math_abs(dot * r[0] / Scalar[DTYPE](3))
        vol_recomputed += v
        var s = v / Scalar[DTYPE](20)

        # k = {00, 11, 22, 01, 02, 12}
        p0 += s * (
            Scalar[DTYPE](2) * (dx * dx + ex * ex + gx * gx)
            + dx * ex + dx * ex + dx * gx + dx * gx + ex * gx + ex * gx
        )
        p1 += s * (
            Scalar[DTYPE](2) * (dy * dy + ey * ey + gy * gy)
            + dy * ey + dy * ey + dy * gy + dy * gy + ey * gy + ey * gy
        )
        p2 += s * (
            Scalar[DTYPE](2) * (dz * dz + ez * ez + gz * gz)
            + dz * ez + dz * ez + dz * gz + dz * gz + ez * gz + ez * gz
        )
        p3 += s * (
            Scalar[DTYPE](2) * (dx * dy + ex * ey + gx * gy)
            + dx * ey + dy * ex + dx * gy + dy * gx + ex * gy + ey * gx
        )
        p4 += s * (
            Scalar[DTYPE](2) * (dx * dz + ex * ez + gx * gz)
            + dx * ez + dz * ex + dx * gz + dz * gx + ex * gz + ez * gx
        )
        p5 += s * (
            Scalar[DTYPE](2) * (dy * dz + ey * ez + gy * gz)
            + dy * ez + dz * ey + dy * gz + dz * gy + ey * gz + ez * gy
        )

    # products of inertia -> moments, in `eig3_symmetric`'s packing
    # [Ixx, Iyy, Izz, Ixy, Ixz, Iyz]
    var full = InlineArray[Scalar[DTYPE], 6](fill=Scalar[DTYPE](0))
    full[0] = p1 + p2
    full[1] = p0 + p2
    full[2] = p0 + p1
    full[3] = -p3
    full[4] = -p4
    full[5] = -p5

    var e = eig3_symmetric[DTYPE](full)

    mi.volume = vol_recomputed
    mi.com_x = cx
    mi.com_y = cy
    mi.com_z = cz
    mi.eig0 = e[0]
    mi.eig1 = e[1]
    mi.eig2 = e[2]
    mi.qx = e[3]
    mi.qy = e[4]
    mi.qz = e[5]
    mi.qw = e[6]
    return mi


def transform_verts_to_principal_frame[
    DTYPE: DType
](
    mut verts: List[Scalar[DTYPE]],
    num_verts: Int,
    mi: MeshInertia[DTYPE],
):
    """Bake `mi`'s frame into the vertices: `v' = R(quat)^T (v - com)`.

    ⚠ The CONJUGATE rotation. Verified against `mjModel.mesh_vert` — the
    conjugate reproduces MuJoCo's stored bounding box to 2.8e-09 (its float32
    storage), the non-conjugate is off by 1.5e-02, which reads as a mesh that
    is merely oriented differently rather than as a bug.

    After this the mesh is centred on its centre of mass with its principal
    axes on x/y/z, which is what makes `geom_inertia` for a mesh a plain
    diagonal and what lets the hull, the polygon normals and `rbound` all be
    built in the same frame MuJoCo uses.
    """
    var mat = InlineArray[Scalar[DTYPE], 9](fill=Scalar[DTYPE](0))
    quat_to_mat[DTYPE](mi.qx, mi.qy, mi.qz, mi.qw, mat)
    for i in range(num_verts):
        var o = i * 3
        var vx = verts[o + 0] - mi.com_x
        var vy = verts[o + 1] - mi.com_y
        var vz = verts[o + 2] - mi.com_z
        # R^T v  — `mat` is row-major, so column k is mat[k], mat[3+k], mat[6+k]
        verts[o + 0] = mat[0] * vx + mat[3] * vy + mat[6] * vz
        verts[o + 1] = mat[1] * vx + mat[4] * vy + mat[7] * vz
        verts[o + 2] = mat[2] * vx + mat[5] * vy + mat[8] * vz


def apply_mesh_ref_transform[
    DTYPE: DType
](
    mut v: List[Scalar[DTYPE]],
    nvert: Int,
    rpx: Float64, rpy: Float64, rpz: Float64,
    rqw: Float64, rqx: Float64, rqy: Float64, rqz: Float64,
    sx: Float64, sy: Float64, sz: Float64,
):
    """`mjCMesh::ApplyTransformations` (user_mesh.cc:1257), in its order.

        v -= refpos
        v  = R(refquat)^T v      <- `mjuu_mulvecmatT`, i.e. the INVERSE turn
        v *= scale

    ⚠⚠ THE ROTATION IS THE QUATERNION'S INVERSE. `refquat="1 -1 0 0"` is a
    -90 deg turn about x and rotates the mesh **+90 deg**; reading it forward
    lands 180 deg away, which on a roughly symmetric part still looks like a
    plausible mesh.

    ⚠ AND IT RUNS BEFORE `scale`, so it cannot be folded in afterwards unless
    the scale happens to be uniform. `scale` is applied here for that reason
    rather than being left to the loader.
    """
    var qn2 = rqw * rqw + rqx * rqx + rqy * rqy + rqz * rqz
    var w = rqw
    var x = rqx
    var y = rqy
    var z = rqz
    if qn2 > 1e-30:
        var inv = 1.0 / sqrt(qn2)
        w *= inv
        x *= inv
        y *= inv
        z *= inv
    else:
        w = 1.0
        x = 0.0
        y = 0.0
        z = 0.0
    # R(q), row-major, then used TRANSPOSED below.
    var m00 = 1.0 - 2.0 * (y * y + z * z)
    var m01 = 2.0 * (x * y - z * w)
    var m02 = 2.0 * (x * z + y * w)
    var m10 = 2.0 * (x * y + z * w)
    var m11 = 1.0 - 2.0 * (x * x + z * z)
    var m12 = 2.0 * (y * z - x * w)
    var m20 = 2.0 * (x * z - y * w)
    var m21 = 2.0 * (y * z + x * w)
    var m22 = 1.0 - 2.0 * (x * x + y * y)
    for i in range(nvert):
        var ax = Float64(v[i * 3 + 0]) - rpx
        var ay = Float64(v[i * 3 + 1]) - rpy
        var az = Float64(v[i * 3 + 2]) - rpz
        # `mjuu_mulvecmatT(res, vec, mat)`: res = M^T vec.
        var bx = m00 * ax + m10 * ay + m20 * az
        var by = m01 * ax + m11 * ay + m21 * az
        var bz = m02 * ax + m12 * ay + m22 * az
        v[i * 3 + 0] = Scalar[DTYPE](bx * sx)
        v[i * 3 + 1] = Scalar[DTYPE](by * sy)
        v[i * 3 + 2] = Scalar[DTYPE](bz * sz)


@always_inline
def mesh_ref_is_identity(
    rpx: Float64, rpy: Float64, rpz: Float64,
    rqw: Float64, rqx: Float64, rqy: Float64, rqz: Float64,
) -> Bool:
    """MuJoCo's own guards: it skips each transform when it is the identity,
    and so does every caller here — which keeps the 84 Menagerie scenes that
    declare neither on the byte-identical path they were already on."""
    return (
        rpx == 0.0 and rpy == 0.0 and rpz == 0.0
        and rqw == 1.0 and rqx == 0.0 and rqy == 0.0 and rqz == 0.0
    )


def mesh_inertia_from_file[
    DTYPE: DType
](
    mesh_filename: String,
    sx: Float64 = 1.0,
    sy: Float64 = 1.0,
    sz: Float64 = 1.0,
    rpx: Float64 = 0.0,
    rpy: Float64 = 0.0,
    rpz: Float64 = 0.0,
    rqw: Float64 = 1.0,
    rqx: Float64 = 0.0,
    rqy: Float64 = 0.0,
    rqz: Float64 = 0.0,
) raises -> MeshInertia[DTYPE]:
    """`mesh_legacy_inertia` straight off an STL path.

    ⚠ Reads the RAW triangle soup — `load_stl` yields three vertices per face
    in winding order, which is exactly legacy's input. Do NOT feed this the
    deduplicated/hulled vertices `load_mesh_hull` builds: the hull is
    `mjMESH_INERTIA_CONVEX`, a different and non-default mode, and on Jaco's
    `base` it moves the centre of mass by 1.5e-03.
    """
    from mojo_rl.render.stl_loader import load_stl

    # ⚠ `refpos`/`refquat` COME BEFORE `scale`, so when either is present the
    # loader is asked for UNSCALED vertices and all three steps are applied
    # here in MuJoCo's order. When both are the identity — 84 of Menagerie's
    # 85 scenes — the loader keeps scaling as it always has, so nothing else
    # moves by even a bit.
    var ident = mesh_ref_is_identity(rpx, rpy, rpz, rqw, rqx, rqy, rqz)
    var mesh_data = load_stl(
        mesh_filename, sx, sy, sz
    ) if ident else load_stl(mesh_filename, 1.0, 1.0, 1.0)
    var n = len(mesh_data.vertices)
    var tris = List[Scalar[DTYPE]]()
    for i in range(n):
        tris.append(Scalar[DTYPE](mesh_data.vertices[i].px))
        tris.append(Scalar[DTYPE](mesh_data.vertices[i].py))
        tris.append(Scalar[DTYPE](mesh_data.vertices[i].pz))
    if not ident:
        apply_mesh_ref_transform[DTYPE](
            tris, n, rpx, rpy, rpz, rqw, rqx, rqy, rqz, sx, sy, sz
        )
    return mesh_legacy_inertia[DTYPE](tris, n // 3)
