"""Linear blend skinning — pose a `SkinData` from body transforms.

    skin_pose(skin, bone_body, xpos, xquat, posed, normals)

⚠ TRANSCRIBED FROM `mjv_updateActiveSkin`
(`references/mujoco-3.6.0/src/engine/engine_vis_visualize.c:3242`), which is
the only definition of what a MuJoCo skin is supposed to look like. Per bone:

    rotate    = xquat[body] * conj(bindquat)
    translate = xpos[body] - rotate * bindpos
    vertex   += weight * (rotate * rest_vertex + translate)

then vertex normals accumulate the UNNORMALIZED face cross products (so each
face contributes in proportion to its area) and are normalized at the end.

⚠ THE BIND POSE IS NOT THE REST POSE OF THE BODY. `bindquat`/`bindpos` describe
where the bone WAS when the vertices were authored; the deformation is the
delta from there to now. Drop the `conj(bindquat)` and the skin still renders —
inside out and rotating with the body instead of with the body's motion — which
is why `test_skinning.mojo` gates the identity case rather than eyeballing it.

SEPARATE FROM THE RENDERER ON PURPOSE. This is arithmetic over plain lists, so
it can be gated on the CPU with no window, no GPU and no model: put every body
at its bind pose and the posed mesh must come back bit-for-bit as the rest mesh.
"""

from std.math import sqrt

from .skn_loader import SkinData


def resolve_skin_bones(
    skin: SkinData, body_names: List[String]
) -> List[Int]:
    """Map each bone to a body index by NAME; -1 for bones with no match.

    ⚠ NAME RESOLUTION, NOT ORDER. A skin's bones are written in whatever order
    the authoring tool emitted and have no relation to body index order — dog's
    57 bones sit in a 61-body model. Matching positionally would produce a
    perfectly smooth animation of the wrong limbs.

    A -1 is not fatal: `skin_pose` skips that bone, so a partially-bound skin
    renders the parts it can. The caller should report the count, because the
    symptom of a bad map is a collapsed region, not an error.
    """
    var out = List[Int]()
    for b in range(len(skin.bones)):
        var found = -1
        for i in range(len(body_names)):
            if body_names[i] == skin.bones[b].body_name:
                found = i
                break
        out.append(found)
    return out^


def skin_pose(
    skin: SkinData,
    bone_body: List[Int],
    xpos: List[Float32],
    xquat: List[Float32],
    mut posed: List[Float32],
    mut normals: List[Float32],
) raises:
    """Deform `skin` into `posed`/`normals` (both 3*nvert, overwritten).

    Args:
        skin: Rest geometry + bones, from `load_skn`.
        bone_body: Body index per bone, from `resolve_skin_bones`; -1 skips.
        xpos: World body positions, 3 per body.
        xquat: World body orientations, 4 per body, (w, x, y, z).
        posed: Output vertex positions, resized if needed.
        normals: Output vertex normals, resized if needed.

    ⚠ CALLED EVERY FRAME. The output lists are arguments rather than return
    values so a caller can keep one pair alive for the life of the window
    instead of allocating 24k vertices' worth of `List` per frame.
    """
    var nv = skin.nvert
    var nb = len(skin.bones)

    if len(bone_body) != nb:
        raise Error(
            "skin_pose: bone_body has " + String(len(bone_body))
            + " entries for " + String(nb) + " bones"
        )

    while len(posed) < 3 * nv:
        posed.append(Float32(0))
    while len(normals) < 3 * nv:
        normals.append(Float32(0))
    for i in range(3 * nv):
        posed[i] = Float32(0)
        normals[i] = Float32(0)

    var n_body = len(xpos) // 3

    for b in range(nb):
        var body = bone_body[b]
        if body < 0 or body >= n_body:
            continue

        # rotate = xquat[body] * conj(bindquat)
        var aw = xquat[4 * body]
        var ax = xquat[4 * body + 1]
        var ay = xquat[4 * body + 2]
        var az = xquat[4 * body + 3]
        # conj of the bind quat — unit quats, so the conjugate IS the inverse.
        var bw = skin.bones[b].bind_quat_w
        var bx = -skin.bones[b].bind_quat_x
        var by = -skin.bones[b].bind_quat_y
        var bz = -skin.bones[b].bind_quat_z

        var qw = aw * bw - ax * bx - ay * by - az * bz
        var qx = aw * bx + ax * bw + ay * bz - az * by
        var qy = aw * by - ax * bz + ay * bw + az * bx
        var qz = aw * bz + ax * by - ay * bx + az * bw

        # Row-major 3x3 from the unit quat.
        var xx = qx * qx
        var yy = qy * qy
        var zz = qz * qz
        var r00 = 1.0 - 2.0 * (yy + zz)
        var r01 = 2.0 * (qx * qy - qw * qz)
        var r02 = 2.0 * (qx * qz + qw * qy)
        var r10 = 2.0 * (qx * qy + qw * qz)
        var r11 = 1.0 - 2.0 * (xx + zz)
        var r12 = 2.0 * (qy * qz - qw * qx)
        var r20 = 2.0 * (qx * qz - qw * qy)
        var r21 = 2.0 * (qy * qz + qw * qx)
        var r22 = 1.0 - 2.0 * (xx + yy)

        # translate = xpos[body] - rotate * bindpos
        var px = skin.bones[b].bind_pos_x
        var py = skin.bones[b].bind_pos_y
        var pz = skin.bones[b].bind_pos_z
        var tx = xpos[3 * body] - (r00 * px + r01 * py + r02 * pz)
        var ty = xpos[3 * body + 1] - (r10 * px + r11 * py + r12 * pz)
        var tz = xpos[3 * body + 2] - (r20 * px + r21 * py + r22 * pz)

        var n_k = len(skin.bones[b].vert_ids)
        for k in range(n_k):
            var vid = Int(skin.bones[b].vert_ids[k])
            var w = skin.bones[b].weights[k]
            var vx = skin.vert[3 * vid]
            var vy = skin.vert[3 * vid + 1]
            var vz = skin.vert[3 * vid + 2]
            posed[3 * vid] += w * (
                r00 * vx + r01 * vy + r02 * vz + tx
            )
            posed[3 * vid + 1] += w * (
                r10 * vx + r11 * vy + r12 * vz + ty
            )
            posed[3 * vid + 2] += w * (
                r20 * vx + r21 * vy + r22 * vz + tz
            )

    # ── vertex normals ───────────────────────────────────────────────────
    # ⚠ THE CROSS PRODUCT IS NOT NORMALIZED BEFORE ACCUMULATION. Its magnitude
    # is twice the triangle's area, so summing the raw vectors weights each
    # face by area — normalizing first would let a sliver of a triangle pull as
    # hard as the large face beside it and crease a smooth surface.
    for fi in range(skin.nface):
        var i0 = Int(skin.face[3 * fi])
        var i1 = Int(skin.face[3 * fi + 1])
        var i2 = Int(skin.face[3 * fi + 2])

        var e1x = posed[3 * i1] - posed[3 * i0]
        var e1y = posed[3 * i1 + 1] - posed[3 * i0 + 1]
        var e1z = posed[3 * i1 + 2] - posed[3 * i0 + 2]
        var e2x = posed[3 * i2] - posed[3 * i0]
        var e2y = posed[3 * i2 + 1] - posed[3 * i0 + 1]
        var e2z = posed[3 * i2 + 2] - posed[3 * i0 + 2]

        var nx = e1y * e2z - e1z * e2y
        var ny = e1z * e2x - e1x * e2z
        var nz = e1x * e2y - e1y * e2x

        normals[3 * i0] += nx
        normals[3 * i0 + 1] += ny
        normals[3 * i0 + 2] += nz
        normals[3 * i1] += nx
        normals[3 * i1 + 1] += ny
        normals[3 * i1 + 2] += nz
        normals[3 * i2] += nx
        normals[3 * i2 + 1] += ny
        normals[3 * i2 + 2] += nz

    for v in range(nv):
        var nx = normals[3 * v]
        var ny = normals[3 * v + 1]
        var nz = normals[3 * v + 2]
        var len_n = sqrt(nx * nx + ny * ny + nz * nz)
        if len_n > Float32(1e-12):
            normals[3 * v] = nx / len_n
            normals[3 * v + 1] = ny / len_n
            normals[3 * v + 2] = nz / len_n
        else:
            # A degenerate fan (every incident face collapsed) has no normal to
            # give. +Z beats a NaN, which would take the whole draw with it.
            normals[3 * v] = Float32(0)
            normals[3 * v + 1] = Float32(0)
            normals[3 * v + 2] = Float32(1)


def bind_pose_transforms(
    skin: SkinData,
    bone_body: List[Int],
    n_body: Int,
    mut xpos: List[Float32],
    mut xquat: List[Float32],
):
    """Fill `xpos`/`xquat` with each bone's own bind pose — the IDENTITY case.

    Posing a skin with this must reproduce the rest mesh exactly: every bone's
    `rotate` becomes `bindquat * conj(bindquat)` = identity and its `translate`
    becomes `bindpos - bindpos` = 0, so each vertex gets back `sum(w) * rest`,
    and the weights sum to 1. That is the whole gate in `test_skinning.mojo`,
    and it is sensitive to the quaternion convention, the conjugate, the matrix
    layout and the bone->body map all at once.
    """
    xpos.clear()
    xquat.clear()
    for _ in range(n_body):
        xpos.append(Float32(0))
        xpos.append(Float32(0))
        xpos.append(Float32(0))
        xquat.append(Float32(1))
        xquat.append(Float32(0))
        xquat.append(Float32(0))
        xquat.append(Float32(0))

    for b in range(len(skin.bones)):
        var body = bone_body[b]
        if body < 0 or body >= n_body:
            continue
        xpos[3 * body] = skin.bones[b].bind_pos_x
        xpos[3 * body + 1] = skin.bones[b].bind_pos_y
        xpos[3 * body + 2] = skin.bones[b].bind_pos_z
        xquat[4 * body] = skin.bones[b].bind_quat_w
        xquat[4 * body + 1] = skin.bones[b].bind_quat_x
        xquat[4 * body + 2] = skin.bones[b].bind_quat_y
        xquat[4 * body + 3] = skin.bones[b].bind_quat_z
