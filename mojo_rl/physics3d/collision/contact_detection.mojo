"""Contact detection over per-field tensors (migration P2, single-source).

Per-field port of `_geom_world_pos_gpu` + `detect_contacts_gpu`
(collision/contact_detection.mojo) — arithmetic verbatim. Reads FK products
(`d.xpos`, `d.xquat`) + geom/body records + model meta + exclude pairs;
writes packed contact records into `d.contacts` and the contact count into
`d.meta` (META_IDX_NUM_CONTACTS).

Operands (10): xpos, xquat (data) + geoms, bodies, mmeta, excludes,
mesh_meta, mesh_verts (model) + contacts, smeta (data outputs). Mesh
collision (plane-mesh vertex scans + GJK/EPA fallback via gjk) is
compiled in only when NMESH_VERTS > 0; zero-mesh models keep today's
branch structure (mesh branches degrade to `continue`)."""

from std.gpu import thread_idx, block_idx, block_dim
from std.gpu.host import DeviceContext
from layout import Layout, LayoutTensor

from std.math import sqrt

from ..kinematics.quat_math import gpu_quat_rotate, gpu_quat_mul
from ..constants import (
    GEOM_SPHERE,
    GEOM_CAPSULE,
    GEOM_BOX,
    GEOM_PLANE,
    GEOM_CYLINDER,
    GEOM_MESH,
    GEOM_ELLIPSOID,
)
from ..fields import Data, Model
from ..gpu.constants import (
    MODEL_BODY_SIZE,
    MODEL_GEOM_SIZE,
    MODEL_META_SIZE,
    METADATA_SIZE,
    MODEL_META_IDX_NEXCLUDE,
    BODY_IDX_PARENT,
    BODY_IDX_WELDID,
    META_IDX_NUM_CONTACTS,
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
    CONTACT_IDX_SOLREF_0,
    CONTACT_IDX_SOLREF_1,
    CONTACT_IDX_SOLIMP_0,
    CONTACT_IDX_SOLIMP_1,
    CONTACT_IDX_SOLIMP_2,
    CONTACT_IDX_SOLIMP_3,
    CONTACT_IDX_SOLIMP_4,
    CONTACT_IDX_FRAME_T1_X,
    CONTACT_IDX_FRAME_T1_Y,
    CONTACT_IDX_FRAME_T1_Z,
    GEOM_IDX_TYPE,
    GEOM_IDX_BODY,
    GEOM_IDX_POS_X,
    GEOM_IDX_POS_Y,
    GEOM_IDX_POS_Z,
    GEOM_IDX_QUAT_X,
    GEOM_IDX_QUAT_Y,
    GEOM_IDX_QUAT_Z,
    GEOM_IDX_QUAT_W,
    GEOM_IDX_RADIUS,
    GEOM_IDX_HALF_LENGTH,
    GEOM_IDX_HALF_X,
    GEOM_IDX_HALF_Y,
    GEOM_IDX_HALF_Z,
    GEOM_IDX_FRICTION,
    GEOM_IDX_CONTYPE,
    GEOM_IDX_CONAFFINITY,
    GEOM_IDX_CONDIM,
    GEOM_IDX_PRIORITY,
    GEOM_IDX_SOLREF_0,
    GEOM_IDX_SOLREF_1,
    GEOM_IDX_SOLIMP_0,
    GEOM_IDX_SOLIMP_1,
    GEOM_IDX_SOLIMP_2,
    GEOM_IDX_SOLIMP_3,
    GEOM_IDX_SOLIMP_4,
    GEOM_IDX_FRICTION_SPIN,
    GEOM_IDX_FRICTION_ROLL,
    GEOM_IDX_RBOUND,
    GEOM_IDX_MARGIN,
    GEOM_IDX_MESH_ID,
    MAX_GPU_MESHES,
    MODEL_MESH_META_SIZE,
)
from .plane_frame import (
    plane_world_normal,
    to_plane_frame,
    from_plane_frame,
    quat_to_plane_frame,
)
from .collision_primitives import (
    sphere_sphere,
    capsule_sphere,
    capsule_capsule,
    box_sphere,
    box_capsule_manifold,
    CB_MAX_POINTS,
    box_box,
    box_box_manifold,
    BB_MAX_POINTS,
    box_plane,
    ellipsoid_plane,
    cylinder_plane,
    cylinder_sphere,
    cylinder_capsule,
    cylinder_cylinder,
    cylinder_box,
)
from .gjk import gjk_epa

comptime CD_TPB: Int = 64


@always_inline
def _geom_world_pos[
    DTYPE: DType,
    NBODY: Int,
    NGEOM: Int,
    BATCH: Int,
](
    env: Int,
    g: Int,
    geoms: LayoutTensor[
        DTYPE, Layout.row_major(NGEOM, MODEL_GEOM_SIZE), MutAnyOrigin
    ],
    xpos: LayoutTensor[
        DTYPE, Layout.row_major(BATCH, NBODY * 3), MutAnyOrigin
    ],
    xquat: LayoutTensor[
        DTYPE, Layout.row_major(BATCH, NBODY * 4), MutAnyOrigin
    ],
    mut out_px: Scalar[DTYPE],
    mut out_py: Scalar[DTYPE],
    mut out_pz: Scalar[DTYPE],
    mut out_qx: Scalar[DTYPE],
    mut out_qy: Scalar[DTYPE],
    mut out_qz: Scalar[DTYPE],
    mut out_qw: Scalar[DTYPE],
):
    """Compute geom world pos/quat (verbatim from _geom_world_pos_gpu)."""
    var body_idx = Int(rebind[Scalar[DTYPE]](geoms[g, GEOM_IDX_BODY]))
    var lx = rebind[Scalar[DTYPE]](geoms[g, GEOM_IDX_POS_X])
    var ly = rebind[Scalar[DTYPE]](geoms[g, GEOM_IDX_POS_Y])
    var lz = rebind[Scalar[DTYPE]](geoms[g, GEOM_IDX_POS_Z])
    var lqx = rebind[Scalar[DTYPE]](geoms[g, GEOM_IDX_QUAT_X])
    var lqy = rebind[Scalar[DTYPE]](geoms[g, GEOM_IDX_QUAT_Y])
    var lqz = rebind[Scalar[DTYPE]](geoms[g, GEOM_IDX_QUAT_Z])
    var lqw = rebind[Scalar[DTYPE]](geoms[g, GEOM_IDX_QUAT_W])
    if body_idx == 0:
        out_px = lx
        out_py = ly
        out_pz = lz
        out_qx = lqx
        out_qy = lqy
        out_qz = lqz
        out_qw = lqw
        return
    var bpx = rebind[Scalar[DTYPE]](xpos[env, body_idx * 3 + 0])
    var bpy = rebind[Scalar[DTYPE]](xpos[env, body_idx * 3 + 1])
    var bpz = rebind[Scalar[DTYPE]](xpos[env, body_idx * 3 + 2])
    var bqx = rebind[Scalar[DTYPE]](xquat[env, body_idx * 4 + 0])
    var bqy = rebind[Scalar[DTYPE]](xquat[env, body_idx * 4 + 1])
    var bqz = rebind[Scalar[DTYPE]](xquat[env, body_idx * 4 + 2])
    var bqw = rebind[Scalar[DTYPE]](xquat[env, body_idx * 4 + 3])
    if (
        lx == Scalar[DTYPE](0)
        and ly == Scalar[DTYPE](0)
        and lz == Scalar[DTYPE](0)
        and lqx == Scalar[DTYPE](0)
        and lqy == Scalar[DTYPE](0)
        and lqz == Scalar[DTYPE](0)
        and lqw == Scalar[DTYPE](1)
    ):
        out_px = bpx
        out_py = bpy
        out_pz = bpz
        out_qx = bqx
        out_qy = bqy
        out_qz = bqz
        out_qw = bqw
        return
    var rotated = gpu_quat_rotate(bqx, bqy, bqz, bqw, lx, ly, lz)
    out_px = bpx + rotated[0]
    out_py = bpy + rotated[1]
    out_pz = bpz + rotated[2]
    var wq = gpu_quat_mul(bqx, bqy, bqz, bqw, lqx, lqy, lqz, lqw)
    out_qx = wq[0]
    out_qy = wq[1]
    out_qz = wq[2]
    out_qw = wq[3]


@always_inline
def _plane_mesh_contacts[
    DTYPE: DType,
    MAX_CONTACTS: Int,
    NGEOM: Int,
    NMESH_VERTS: Int,
    BATCH: Int,
](
    env: Int,
    g: Int,
    g_body: Int,
    p_x: Scalar[DTYPE],
    p_y: Scalar[DTYPE],
    p_z: Scalar[DTYPE],
    q_x: Scalar[DTYPE],
    q_y: Scalar[DTYPE],
    q_z: Scalar[DTYPE],
    q_w: Scalar[DTYPE],
    ground_z: Scalar[DTYPE],
    plp_x: Scalar[DTYPE],
    plp_y: Scalar[DTYPE],
    plp_z: Scalar[DTYPE],
    plq_x: Scalar[DTYPE],
    plq_y: Scalar[DTYPE],
    plq_z: Scalar[DTYPE],
    plq_w: Scalar[DTYPE],
    contact_margin: Scalar[DTYPE],
    contact_friction: Scalar[DTYPE],
    contact_friction_spin: Scalar[DTYPE],
    contact_friction_roll: Scalar[DTYPE],
    contact_condim: Int,
    geoms: LayoutTensor[
        DTYPE, Layout.row_major(NGEOM, MODEL_GEOM_SIZE), MutAnyOrigin
    ],
    mesh_meta: LayoutTensor[
        DTYPE,
        Layout.row_major(MAX_GPU_MESHES, MODEL_MESH_META_SIZE),
        MutAnyOrigin,
    ],
    mesh_verts: LayoutTensor[
        DTYPE, Layout.row_major(NMESH_VERTS, 3), MutAnyOrigin
    ],
    contacts: LayoutTensor[
        DTYPE, Layout.row_major(BATCH, MAX_CONTACTS * CONTACT_SIZE),
        MutAnyOrigin,
    ],
    mut num_contacts: Int,
):
    """Plane-mesh: scan hull vertices below plane (verbatim from the
    detect_contacts_gpu plane-mesh branches; both i/j orientations reduce to
    this after substituting the mesh geom's pose).

    `p_*` / `q_*` are the mesh geom's pose IN THE PLANE'S FRAME and `ground_z`
    is 0 there, so the vertex heights below are heights above the plane
    whatever way the plane faces. `plp_*` / `plq_*` are the plane's own world
    pose, used only to put the contact point and normal back into world —
    see `collision/plane_frame.mojo`."""
    var pn = plane_world_normal[DTYPE](plq_x, plq_y, plq_z, plq_w)
    var m_id = Int(rebind[Scalar[DTYPE]](geoms[g, GEOM_IDX_MESH_ID]))
    if m_id >= 0:
        var pm_vadr = Int(rebind[Scalar[DTYPE]](mesh_meta[m_id, 0]))
        var pm_vnum = Int(rebind[Scalar[DTYPE]](mesh_meta[m_id, 1]))
        for vi in range(pm_vnum):
            if num_contacts >= MAX_CONTACTS:
                break
            var vx = rebind[Scalar[DTYPE]](mesh_verts[pm_vadr + vi, 0])
            var vy = rebind[Scalar[DTYPE]](mesh_verts[pm_vadr + vi, 1])
            var vz = rebind[Scalar[DTYPE]](mesh_verts[pm_vadr + vi, 2])
            var local_pt = gpu_quat_rotate(q_x, q_y, q_z, q_w, vx, vy, vz)
            var wx = p_x + local_pt[0]
            var wy = p_y + local_pt[1]
            var wz = p_z + local_pt[2]
            var dist_v = wz - ground_z
            if dist_v < contact_margin:
                var c_off = num_contacts * CONTACT_SIZE
                contacts[env, c_off + CONTACT_IDX_BODY_A] = Scalar[DTYPE](
                    g_body
                )
                contacts[env, c_off + CONTACT_IDX_BODY_B] = Scalar[DTYPE](0)
                var cw = from_plane_frame[DTYPE](
                    plp_x, plp_y, plp_z, plq_x, plq_y, plq_z, plq_w,
                    wx, wy, ground_z + dist_v * Scalar[DTYPE](0.5),
                )
                contacts[env, c_off + CONTACT_IDX_POS_X] = cw[0]
                contacts[env, c_off + CONTACT_IDX_POS_Y] = cw[1]
                contacts[env, c_off + CONTACT_IDX_POS_Z] = cw[2]
                contacts[env, c_off + CONTACT_IDX_NX] = pn[0]
                contacts[env, c_off + CONTACT_IDX_NY] = pn[1]
                contacts[env, c_off + CONTACT_IDX_NZ] = pn[2]
                contacts[env, c_off + CONTACT_IDX_DIST] = dist_v
                contacts[
                    env, c_off + CONTACT_IDX_INCLUDEMARGIN
                ] = contact_margin
                contacts[
                    env, c_off + CONTACT_IDX_FRICTION
                ] = contact_friction
                contacts[
                    env, c_off + CONTACT_IDX_FRICTION_SPIN
                ] = contact_friction_spin
                contacts[
                    env, c_off + CONTACT_IDX_FRICTION_ROLL
                ] = contact_friction_roll
                contacts[env, c_off + CONTACT_IDX_CONDIM] = Scalar[DTYPE](
                    contact_condim
                )
                num_contacts += 1


@always_inline
@always_inline
def mix_contact_params[
    DTYPE: DType
](
    prio_i: Int,
    condim_i: Int,
    fri_i: Scalar[DTYPE],
    fsp_i: Scalar[DTYPE],
    frl_i: Scalar[DTYPE],
    sr0_i: Scalar[DTYPE],
    sr1_i: Scalar[DTYPE],
    si0_i: Scalar[DTYPE],
    si1_i: Scalar[DTYPE],
    si2_i: Scalar[DTYPE],
    si3_i: Scalar[DTYPE],
    si4_i: Scalar[DTYPE],
    prio_j: Int,
    condim_j: Int,
    fri_j: Scalar[DTYPE],
    fsp_j: Scalar[DTYPE],
    frl_j: Scalar[DTYPE],
    sr0_j: Scalar[DTYPE],
    sr1_j: Scalar[DTYPE],
    si0_j: Scalar[DTYPE],
    si1_j: Scalar[DTYPE],
    si2_j: Scalar[DTYPE],
    si3_j: Scalar[DTYPE],
    si4_j: Scalar[DTYPE],
) -> InlineArray[Scalar[DTYPE], 12]:
    """MuJoCo's contact-parameter mixing. Port of
    `engine_collision_driver.c:1426-1480`.

    Returns `[condim, friction, friction_spin, friction_roll, solref0, solref1,
    solimp0..solimp4]` — index 0 holds condim as a float.

    THE RULE, and every branch of it matters for dm_control's quadruped:

      * **Priorities DIFFER** -> the higher-priority geom supplies condim,
        solref, solimp AND friction, wholesale. No mixing of any kind. This is
        how quadruped's ball (`priority="1"`) forces its own `condim="6"` and
        `solref="-10000 -30"` onto every contact it takes part in, including
        against a floor whose parameters are entirely different.
      * **Priorities EQUAL**:
          - condim   -> max
          - friction -> elementwise max
          - solref   -> elementwise MEAN if BOTH `solref[0] > 0`, otherwise
            elementwise MIN. That second branch is why a DIRECT (negative)
            solref wins over a standard one even at equal priority: it is not
            averaged, it is taken.
          - solimp   -> elementwise mean, with no direct branch at all.

    ⚠ THE SOLREF TEST IS ON COMPONENT [0] OF **BOTH** GEOMS
    (`solref1[0] > 0 && solref2[0] > 0`), not "either is negative". Same
    outcome for the sign combinations MuJoCo's compiler permits, but the
    condition is the source's.

    ⚠ THE MEAN IS A SPECIAL CASE OF A `solmix` WEIGHTING —
    `mix = solmix1/(solmix1 + solmix2)` — which is 0.5 only because every geom
    defaults to `solmix = 1`. `full_parser` REJECTS a non-default `solmix`
    rather than letting it silently degrade to this mean. No suite model sets
    one; a five-point probe against MuJoCo could not have revealed this, and
    did not — the source did.

    ⚠ Until 2026-08-03 the narrow phase applied the equal-priority max rule to
    friction and condim UNCONDITIONALLY and never looked at solref/solimp at
    all, so `priority` was ignored and per-geom solparams were dead data.
    """
    var out = InlineArray[Scalar[DTYPE], 12](fill=Scalar[DTYPE](0))

    if prio_i != prio_j:
        var hi_i = prio_i > prio_j
        out[0] = Scalar[DTYPE](condim_i if hi_i else condim_j)
        out[1] = fri_i if hi_i else fri_j
        out[2] = fsp_i if hi_i else fsp_j
        out[3] = frl_i if hi_i else frl_j
        out[4] = sr0_i if hi_i else sr0_j
        out[5] = sr1_i if hi_i else sr1_j
        out[6] = si0_i if hi_i else si0_j
        out[7] = si1_i if hi_i else si1_j
        out[8] = si2_i if hi_i else si2_j
        out[9] = si3_i if hi_i else si3_j
        out[10] = si4_i if hi_i else si4_j
        return out

    out[0] = Scalar[DTYPE](condim_i if condim_i > condim_j else condim_j)
    out[1] = fri_i if fri_i > fri_j else fri_j
    out[2] = fsp_i if fsp_i > fsp_j else fsp_j
    out[3] = frl_i if frl_i > frl_j else frl_j

    comptime HALF = Scalar[DTYPE](0.5)
    if sr0_i > Scalar[DTYPE](0) and sr0_j > Scalar[DTYPE](0):
        out[4] = HALF * (sr0_i + sr0_j)
        out[5] = HALF * (sr1_i + sr1_j)
    else:
        out[4] = sr0_i if sr0_i < sr0_j else sr0_j
        out[5] = sr1_i if sr1_i < sr1_j else sr1_j

    out[6] = HALF * (si0_i + si0_j)
    out[7] = HALF * (si1_i + si1_j)
    out[8] = HALF * (si2_i + si2_j)
    out[9] = HALF * (si3_i + si3_j)
    out[10] = HALF * (si4_i + si4_j)
    return out


def _plane_cylinder_contacts[
    DTYPE: DType,
    MAX_CONTACTS: Int,
    BATCH: Int,
](
    env: Int,
    g_body: Int,
    p_x: Scalar[DTYPE],
    p_y: Scalar[DTYPE],
    p_z: Scalar[DTYPE],
    q_x: Scalar[DTYPE],
    q_y: Scalar[DTYPE],
    q_z: Scalar[DTYPE],
    q_w: Scalar[DTYPE],
    radius: Scalar[DTYPE],
    half_length: Scalar[DTYPE],
    ground_z: Scalar[DTYPE],
    plp_x: Scalar[DTYPE],
    plp_y: Scalar[DTYPE],
    plp_z: Scalar[DTYPE],
    plq_x: Scalar[DTYPE],
    plq_y: Scalar[DTYPE],
    plq_z: Scalar[DTYPE],
    plq_w: Scalar[DTYPE],
    contact_margin: Scalar[DTYPE],
    contact_friction: Scalar[DTYPE],
    contact_friction_spin: Scalar[DTYPE],
    contact_friction_roll: Scalar[DTYPE],
    contact_condim: Int,
    world_body: Int,
    contacts: LayoutTensor[
        DTYPE, Layout.row_major(BATCH, MAX_CONTACTS * CONTACT_SIZE),
        MutAnyOrigin,
    ],
    mut num_contacts: Int,
):
    """Plane-cylinder: up to FOUR points — two rim, two triangle.

    Port of `mjc_PlaneCylinder` (`engine_collision_primitive.c`). We emitted ONE
    point per pair until 2026-08-03, so a cylinder standing on its flat face had
    no restoring torque and tipped — the same defect class as bug 39 (box/plane
    and box/box), and it survived for the same reason: no test compared a
    plane's contact SET against MuJoCo.

    ⚠ THE ROUTINE IS BEHAVIOURALLY IDENTICAL IN 3.6.0 AND 3.3.6 — checked,
    because `mjc_BoxBox` is NOT (3.3.6 halves the face-path depth), and a
    faithful transcription from the wrong tree there would have been a silent
    factor of two. Here the two differ only in `mji_*` vs `mju_*` inlining and
    whitespace.

    THE POINTS, in MuJoCo's order, which is part of the answer:
      1. the deepest rim point on the NEAR cap — `+vec +axis`. If this one is
         above `margin` the routine returns ZERO contacts outright, without
         testing the others.
      2. the same rim direction on the FAR cap — `+vec -axis`.
      3-4. two triangle points at `±vec1` on the near cap, where
         `vec1 = normalize(cross(vec, axis)) * radius*sqrt(3)/2`, offset by
         `-vec*0.5`. Together with point 1 they are the inscribed triangle that
         gives a flat-resting cylinder its support polygon.

    `p_*` / `q_*` are the cylinder's pose IN THE PLANE'S FRAME, where the plane
    is z = `ground_z` facing +z — so the world normal is (0,0,1) here and
    `dist0` is just `p_z - ground_z`. `plp_*` / `plq_*` put the point and the
    normal back into world (`collision/plane_frame.mojo`).

    ⚠ `axis` IS ALREADY SCALED BY `half_length` WHERE THE CROSS PRODUCT USES
    IT. That is MuJoCo's order (`mju_scl3(axis, axis, size2[1])` runs before the
    first point is built), and it is harmless only because `vec1` is normalized
    straight after — but transcribing it in the other order would change
    nothing until someone removed the normalize.

    Verified before this was written: the routine was transcribed to Python and
    swept against the MuJoCo runtime over 400 random poses — 272 contacting,
    557 points, 0 count mismatches, dist 2.1e-17, position 5.6e-17.
    """
    var pn = plane_world_normal[DTYPE](plq_x, plq_y, plq_z, plq_w)
    comptime MINVAL = Scalar[DTYPE](1e-15)

    # Cylinder axis, flipped so it points TOWARDS the plane.
    var ax0 = gpu_quat_rotate(
        q_x, q_y, q_z, q_w,
        Scalar[DTYPE](0), Scalar[DTYPE](0), Scalar[DTYPE](1),
    )
    var ax = ax0[0]
    var ay = ax0[1]
    var az = ax0[2]
    var prjaxis = az  # dot((0,0,1), axis) in the plane frame
    if prjaxis > Scalar[DTYPE](0):
        ax = -ax
        ay = -ay
        az = -az
        prjaxis = -prjaxis

    var dist0 = p_z - ground_z

    # vec = axis*prjaxis - normal, then rescaled to the cylinder radius. This
    # is the radial direction most steeply into the plane.
    var vx = ax * prjaxis
    var vy = ay * prjaxis
    var vz = az * prjaxis - Scalar[DTYPE](1)
    var len_sqr = vx * vx + vy * vy + vz * vz
    if len_sqr >= MINVAL * MINVAL:
        var scl = radius / sqrt(len_sqr)
        vx = vx * scl
        vy = vy * scl
        vz = vz * scl
    else:
        # Disk parallel to the plane: the radial direction is undefined, so
        # MuJoCo picks the cylinder's own x-axis. This is the branch a
        # flat-resting cylinder takes, i.e. the common case, not a corner one.
        var xa = gpu_quat_rotate(
            q_x, q_y, q_z, q_w,
            Scalar[DTYPE](1), Scalar[DTYPE](0), Scalar[DTYPE](0),
        )
        vx = xa[0] * radius
        vy = xa[1] * radius
        vz = xa[2] * radius

    var prjvec = vz
    ax = ax * half_length
    ay = ay * half_length
    az = az * half_length
    prjaxis = prjaxis * half_length

    # Point 1 — near-cap rim. Its rejection ends the routine.
    var d1 = dist0 + prjaxis + prjvec
    if d1 > contact_margin or num_contacts >= MAX_CONTACTS:
        return
    _emit_plane_contact[DTYPE, MAX_CONTACTS, BATCH](
        env, g_body, p_x + vx + ax, p_y + vy + ay,
        p_z + vz + az - d1 * Scalar[DTYPE](0.5), d1,
        plp_x, plp_y, plp_z, plq_x, plq_y, plq_z, plq_w, pn,
        contact_margin, contact_friction, contact_friction_spin,
        contact_friction_roll, contact_condim, world_body,
        contacts, num_contacts,
    )

    # Point 2 — far-cap rim, same radial direction.
    var d2 = dist0 - prjaxis + prjvec
    if d2 <= contact_margin and num_contacts < MAX_CONTACTS:
        _emit_plane_contact[DTYPE, MAX_CONTACTS, BATCH](
            env, g_body, p_x + vx - ax, p_y + vy - ay,
            p_z + vz - az - d2 * Scalar[DTYPE](0.5), d2,
            plp_x, plp_y, plp_z, plq_x, plq_y, plq_z, plq_w, pn,
            contact_margin, contact_friction, contact_friction_spin,
            contact_friction_roll, contact_condim, world_body,
            contacts, num_contacts,
        )

    # Points 3 and 4 — the triangle on the near cap.
    var prjvec1 = -prjvec * Scalar[DTYPE](0.5)
    var d3 = dist0 + prjaxis + prjvec1
    if d3 <= contact_margin:
        var w1x = vy * az - vz * ay
        var w1y = vz * ax - vx * az
        var w1z = vx * ay - vy * ax
        var wl = sqrt(w1x * w1x + w1y * w1y + w1z * w1z)
        if wl > MINVAL:
            w1x = w1x / wl
            w1y = w1y / wl
            w1z = w1z / wl
        else:
            # `mju_normalize3` rewrites a zero vector as (1,0,0); unreachable
            # here because `vec` is perpendicular to `axis` in both branches
            # above, but transcribed so the degenerate case cannot silently
            # become a zero-length offset.
            w1x = Scalar[DTYPE](1)
            w1y = Scalar[DTYPE](0)
            w1z = Scalar[DTYPE](0)
        var s3 = radius * sqrt(Scalar[DTYPE](3.0)) / Scalar[DTYPE](2)
        w1x = w1x * s3
        w1y = w1y * s3
        w1z = w1z * s3

        var bx = ax - vx * Scalar[DTYPE](0.5)
        var by = ay - vy * Scalar[DTYPE](0.5)
        var bz = az - vz * Scalar[DTYPE](0.5) - d3 * Scalar[DTYPE](0.5)
        if num_contacts < MAX_CONTACTS:
            _emit_plane_contact[DTYPE, MAX_CONTACTS, BATCH](
                env, g_body, p_x + w1x + bx, p_y + w1y + by, p_z + w1z + bz,
                d3, plp_x, plp_y, plp_z, plq_x, plq_y, plq_z, plq_w, pn,
                contact_margin, contact_friction, contact_friction_spin,
                contact_friction_roll, contact_condim, world_body,
                contacts, num_contacts,
            )
        if num_contacts < MAX_CONTACTS:
            _emit_plane_contact[DTYPE, MAX_CONTACTS, BATCH](
                env, g_body, p_x - w1x + bx, p_y - w1y + by, p_z - w1z + bz,
                d3, plp_x, plp_y, plp_z, plq_x, plq_y, plq_z, plq_w, pn,
                contact_margin, contact_friction, contact_friction_spin,
                contact_friction_roll, contact_condim, world_body,
                contacts, num_contacts,
            )


def _emit_plane_contact[
    DTYPE: DType,
    MAX_CONTACTS: Int,
    BATCH: Int,
](
    env: Int,
    g_body: Int,
    lx: Scalar[DTYPE],
    ly: Scalar[DTYPE],
    lz: Scalar[DTYPE],
    dist: Scalar[DTYPE],
    plp_x: Scalar[DTYPE],
    plp_y: Scalar[DTYPE],
    plp_z: Scalar[DTYPE],
    plq_x: Scalar[DTYPE],
    plq_y: Scalar[DTYPE],
    plq_z: Scalar[DTYPE],
    plq_w: Scalar[DTYPE],
    pn: InlineArray[Scalar[DTYPE], 3],
    contact_margin: Scalar[DTYPE],
    contact_friction: Scalar[DTYPE],
    contact_friction_spin: Scalar[DTYPE],
    contact_friction_roll: Scalar[DTYPE],
    contact_condim: Int,
    world_body: Int,
    contacts: LayoutTensor[
        DTYPE, Layout.row_major(BATCH, MAX_CONTACTS * CONTACT_SIZE),
        MutAnyOrigin,
    ],
    mut num_contacts: Int,
):
    """Write one plane contact whose point is given IN THE PLANE FRAME.

    Factored out of `_plane_cylinder_contacts`, which emits four points that
    differ only in position — repeating the twelve-field write four times is how
    a slot gets missed in one copy. ⚠ `world_body` is passed rather than
    hardcoded for the same reason as in `_plane_box_contacts`: the naive path
    writes 0 and the SAP path -1.
    """
    var cw = from_plane_frame[DTYPE](
        plp_x, plp_y, plp_z, plq_x, plq_y, plq_z, plq_w, lx, ly, lz
    )
    var c_off = num_contacts * CONTACT_SIZE
    contacts[env, c_off + CONTACT_IDX_BODY_A] = Scalar[DTYPE](g_body)
    contacts[env, c_off + CONTACT_IDX_BODY_B] = Scalar[DTYPE](world_body)
    contacts[env, c_off + CONTACT_IDX_POS_X] = cw[0]
    contacts[env, c_off + CONTACT_IDX_POS_Y] = cw[1]
    contacts[env, c_off + CONTACT_IDX_POS_Z] = cw[2]
    contacts[env, c_off + CONTACT_IDX_NX] = pn[0]
    contacts[env, c_off + CONTACT_IDX_NY] = pn[1]
    contacts[env, c_off + CONTACT_IDX_NZ] = pn[2]
    contacts[env, c_off + CONTACT_IDX_DIST] = dist
    contacts[env, c_off + CONTACT_IDX_INCLUDEMARGIN] = contact_margin
    contacts[env, c_off + CONTACT_IDX_FRICTION] = contact_friction
    contacts[env, c_off + CONTACT_IDX_FRICTION_SPIN] = contact_friction_spin
    contacts[env, c_off + CONTACT_IDX_FRICTION_ROLL] = contact_friction_roll
    contacts[env, c_off + CONTACT_IDX_CONDIM] = Scalar[DTYPE](contact_condim)
    num_contacts += 1


def _plane_box_contacts[
    DTYPE: DType,
    MAX_CONTACTS: Int,
    BATCH: Int,
](
    env: Int,
    g_body: Int,
    p_x: Scalar[DTYPE],
    p_y: Scalar[DTYPE],
    p_z: Scalar[DTYPE],
    q_x: Scalar[DTYPE],
    q_y: Scalar[DTYPE],
    q_z: Scalar[DTYPE],
    q_w: Scalar[DTYPE],
    hx: Scalar[DTYPE],
    hy: Scalar[DTYPE],
    hz: Scalar[DTYPE],
    ground_z: Scalar[DTYPE],
    plp_x: Scalar[DTYPE],
    plp_y: Scalar[DTYPE],
    plp_z: Scalar[DTYPE],
    plq_x: Scalar[DTYPE],
    plq_y: Scalar[DTYPE],
    plq_z: Scalar[DTYPE],
    plq_w: Scalar[DTYPE],
    contact_margin: Scalar[DTYPE],
    contact_friction: Scalar[DTYPE],
    contact_friction_spin: Scalar[DTYPE],
    contact_friction_roll: Scalar[DTYPE],
    contact_condim: Int,
    world_body: Int,
    contacts: LayoutTensor[
        DTYPE, Layout.row_major(BATCH, MAX_CONTACTS * CONTACT_SIZE),
        MutAnyOrigin,
    ],
    mut num_contacts: Int,
):
    """Plane-box: one contact per box CORNER below the plane, up to four.

    Port of `mjc_PlaneBox` (`engine_collision_primitive.c`). A box resting flat
    on a plane touches it over a whole face, and a single point cannot express
    that: a rigid body supported at ONE point has no restoring torque about it,
    so a cube on a floor pivots instead of resting. That is task #42, and this
    is its box/plane half.

    `p_*` / `q_*` are the box's pose IN THE PLANE'S FRAME, where the plane is
    z = `ground_z` facing +z, so a corner's height above the plane is just its
    z. `plp_*` / `plq_*` are the plane's world pose, used only to put the
    contact point and normal back into world — see `collision/plane_frame.mojo`.

    ⚠ TWO FILTERS, BOTH MuJoCo's, AND THE SECOND IS NOT REDUNDANT. A corner is
    skipped when it is further than `margin` above the plane (obviously) AND
    when its offset from the box CENTRE points along +normal (`ldist > 0`),
    which drops the box's upper four corners even when a deeply sunk box has all
    eight below the plane. Without it a fully submerged box would emit four
    contacts on its TOP face pushing the wrong way.

    ⚠ THE ITERATION ORDER IS PART OF THE ANSWER when more than four corners
    qualify: MuJoCo keeps the first four in `i = 0..7` with x = i&1, y = i&2,
    z = i&4, so this loop matches that order rather than sorting by depth.
    """
    var pn = plane_world_normal[DTYPE](plq_x, plq_y, plq_z, plq_w)
    var cnt = 0
    for i in range(8):
        if num_contacts >= MAX_CONTACTS or cnt >= 4:
            break
        var vx = hx if (i & 1) != 0 else -hx
        var vy = hy if (i & 2) != 0 else -hy
        var vz = hz if (i & 4) != 0 else -hz
        var rel = gpu_quat_rotate(q_x, q_y, q_z, q_w, vx, vy, vz)
        # `ldist` is the corner offset along the plane normal, which in this
        # frame is simply its z component.
        var ldist = rel[2]
        var cdist = (p_z + rel[2]) - ground_z
        if cdist > contact_margin or ldist > Scalar[DTYPE](0):
            continue

        var cw = from_plane_frame[DTYPE](
            plp_x, plp_y, plp_z, plq_x, plq_y, plq_z, plq_w,
            p_x + rel[0],
            p_y + rel[1],
            (p_z + rel[2]) - cdist * Scalar[DTYPE](0.5),
        )
        var c_off = num_contacts * CONTACT_SIZE
        contacts[env, c_off + CONTACT_IDX_BODY_A] = Scalar[DTYPE](g_body)
        # ⚠ THE TWO PATHS DISAGREE ON THE WORLD BODY ID — `detect_contacts`
        # writes 0 and the SAP broadphase writes -1 — so it is passed in rather
        # than hardcoded here, which would silently change one of them.
        contacts[env, c_off + CONTACT_IDX_BODY_B] = Scalar[DTYPE](world_body)
        contacts[env, c_off + CONTACT_IDX_POS_X] = cw[0]
        contacts[env, c_off + CONTACT_IDX_POS_Y] = cw[1]
        contacts[env, c_off + CONTACT_IDX_POS_Z] = cw[2]
        contacts[env, c_off + CONTACT_IDX_NX] = pn[0]
        contacts[env, c_off + CONTACT_IDX_NY] = pn[1]
        contacts[env, c_off + CONTACT_IDX_NZ] = pn[2]
        contacts[env, c_off + CONTACT_IDX_DIST] = cdist
        contacts[env, c_off + CONTACT_IDX_INCLUDEMARGIN] = contact_margin
        contacts[env, c_off + CONTACT_IDX_FRICTION] = contact_friction
        contacts[
            env, c_off + CONTACT_IDX_FRICTION_SPIN
        ] = contact_friction_spin
        contacts[
            env, c_off + CONTACT_IDX_FRICTION_ROLL
        ] = contact_friction_roll
        contacts[env, c_off + CONTACT_IDX_CONDIM] = Scalar[DTYPE](
            contact_condim
        )
        num_contacts += 1
        cnt += 1


@always_inline
def _capsule_box_contacts[
    DTYPE: DType,
    MAX_CONTACTS: Int,
    BATCH: Int,
](
    env: Int,
    body_a: Int,
    body_b: Int,
    box_x: Scalar[DTYPE],
    box_y: Scalar[DTYPE],
    box_z: Scalar[DTYPE],
    box_qx: Scalar[DTYPE],
    box_qy: Scalar[DTYPE],
    box_qz: Scalar[DTYPE],
    box_qw: Scalar[DTYPE],
    box_hx: Scalar[DTYPE],
    box_hy: Scalar[DTYPE],
    box_hz: Scalar[DTYPE],
    cap_x: Scalar[DTYPE],
    cap_y: Scalar[DTYPE],
    cap_z: Scalar[DTYPE],
    cap_qx: Scalar[DTYPE],
    cap_qy: Scalar[DTYPE],
    cap_qz: Scalar[DTYPE],
    cap_qw: Scalar[DTYPE],
    cap_hl: Scalar[DTYPE],
    cap_r: Scalar[DTYPE],
    # -1 when the BOX is geom i, +1 when the CAPSULE is. See below.
    nsgn: Scalar[DTYPE],
    contact_margin: Scalar[DTYPE],
    contact_friction: Scalar[DTYPE],
    contact_friction_spin: Scalar[DTYPE],
    contact_friction_roll: Scalar[DTYPE],
    contact_condim: Int,
    contacts: LayoutTensor[
        DTYPE, Layout.row_major(BATCH, MAX_CONTACTS * CONTACT_SIZE),
        MutAnyOrigin,
    ],
    mut num_contacts: Int,
) -> Int:
    """Capsule/box: up to TWO contacts, MuJoCo's manifold.

    A capsule lying along a box face touches over a segment; one point leaves
    it free to pivot, the same defect a box on one contact point has. See
    `box_capsule_manifold`.

    Returns the number of records written, so the caller knows whether to fall
    back (it never has to — 0 means no contact).

    ⚠ NORMAL SIGN. `box_capsule_manifold` returns box -> capsule. The record's
    convention is `body_b -> body_a` = `gj -> gi`, so when the BOX is geom i
    that is the negation of the manifold normal (`nsgn = -1`) and when the
    CAPSULE is geom i it is the manifold normal itself (`nsgn = +1`). The two
    single-point branches this replaces encoded the same thing as `nx = r[4]`
    versus `nx = -r[4]` followed by the shared emit's unconditional negation.
    """
    var cb_dist = InlineArray[Scalar[DTYPE], CB_MAX_POINTS](
        fill=Scalar[DTYPE](0)
    )
    var cb_pos = InlineArray[Scalar[DTYPE], 3 * CB_MAX_POINTS](
        fill=Scalar[DTYPE](0)
    )
    var cb_n = InlineArray[Scalar[DTYPE], 3 * CB_MAX_POINTS](
        fill=Scalar[DTYPE](0)
    )
    var n_cb = box_capsule_manifold[DTYPE](
        box_x, box_y, box_z, box_qx, box_qy, box_qz, box_qw,
        box_hx, box_hy, box_hz,
        cap_x, cap_y, cap_z, cap_qx, cap_qy, cap_qz, cap_qw,
        cap_hl, cap_r,
        contact_margin,
        cb_dist,
        cb_pos,
        cb_n,
    )

    var written = 0
    for c in range(n_cb):
        if num_contacts >= MAX_CONTACTS:
            break
        if cb_dist[c] >= contact_margin:
            continue
        var c_off = num_contacts * CONTACT_SIZE
        contacts[env, c_off + CONTACT_IDX_BODY_A] = Scalar[DTYPE](body_a)
        contacts[env, c_off + CONTACT_IDX_BODY_B] = Scalar[DTYPE](body_b)
        contacts[env, c_off + CONTACT_IDX_POS_X] = cb_pos[3 * c + 0]
        contacts[env, c_off + CONTACT_IDX_POS_Y] = cb_pos[3 * c + 1]
        contacts[env, c_off + CONTACT_IDX_POS_Z] = cb_pos[3 * c + 2]
        contacts[env, c_off + CONTACT_IDX_NX] = nsgn * cb_n[3 * c + 0]
        contacts[env, c_off + CONTACT_IDX_NY] = nsgn * cb_n[3 * c + 1]
        contacts[env, c_off + CONTACT_IDX_NZ] = nsgn * cb_n[3 * c + 2]
        contacts[env, c_off + CONTACT_IDX_DIST] = cb_dist[c]
        contacts[env, c_off + CONTACT_IDX_INCLUDEMARGIN] = contact_margin
        contacts[env, c_off + CONTACT_IDX_FRICTION] = contact_friction
        contacts[
            env, c_off + CONTACT_IDX_FRICTION_SPIN
        ] = contact_friction_spin
        contacts[
            env, c_off + CONTACT_IDX_FRICTION_ROLL
        ] = contact_friction_roll
        contacts[env, c_off + CONTACT_IDX_CONDIM] = Scalar[DTYPE](
            contact_condim
        )
        num_contacts += 1
        written += 1
    return written


@always_inline
def _box_box_contacts[
    DTYPE: DType,
    MAX_CONTACTS: Int,
    BATCH: Int,
](
    env: Int,
    body_a: Int,
    body_b: Int,
    ai_x: Scalar[DTYPE],
    ai_y: Scalar[DTYPE],
    ai_z: Scalar[DTYPE],
    ai_qx: Scalar[DTYPE],
    ai_qy: Scalar[DTYPE],
    ai_qz: Scalar[DTYPE],
    ai_qw: Scalar[DTYPE],
    ai_hx: Scalar[DTYPE],
    ai_hy: Scalar[DTYPE],
    ai_hz: Scalar[DTYPE],
    bj_x: Scalar[DTYPE],
    bj_y: Scalar[DTYPE],
    bj_z: Scalar[DTYPE],
    bj_qx: Scalar[DTYPE],
    bj_qy: Scalar[DTYPE],
    bj_qz: Scalar[DTYPE],
    bj_qw: Scalar[DTYPE],
    bj_hx: Scalar[DTYPE],
    bj_hy: Scalar[DTYPE],
    bj_hz: Scalar[DTYPE],
    contact_margin: Scalar[DTYPE],
    contact_friction: Scalar[DTYPE],
    contact_friction_spin: Scalar[DTYPE],
    contact_friction_roll: Scalar[DTYPE],
    contact_condim: Int,
    contacts: LayoutTensor[
        DTYPE, Layout.row_major(BATCH, MAX_CONTACTS * CONTACT_SIZE),
        MutAnyOrigin,
    ],
    mut num_contacts: Int,
) -> Int:
    """Box/box: the whole manifold, on both the FACE and EDGE-EDGE axes.

    Returns MuJoCo's `code` so the caller can fall back for the one case this
    does not write: `-1`, separated. On any `code >= 0` the records are written
    here and the caller must NOT emit again.

    A box resting on another touches over a whole face, and one point cannot
    express that — the same reason `_plane_box_contacts` exists. See
    `box_box_manifold` for the port and for why it came from MuJoCo 3.6.0
    rather than from `references/mujoco-3.3.6/`.
    """
    var n_bb = 0
    var bb_dist = InlineArray[Scalar[DTYPE], BB_MAX_POINTS](
        fill=Scalar[DTYPE](0)
    )
    var bb_pos = InlineArray[Scalar[DTYPE], 3 * BB_MAX_POINTS](
        fill=Scalar[DTYPE](0)
    )
    var bb_n = InlineArray[Scalar[DTYPE], 3](fill=Scalar[DTYPE](0))
    var code = box_box_manifold[DTYPE](
        ai_x, ai_y, ai_z, ai_qx, ai_qy, ai_qz, ai_qw, ai_hx, ai_hy, ai_hz,
        bj_x, bj_y, bj_z, bj_qx, bj_qy, bj_qz, bj_qw, bj_hx, bj_hy, bj_hz,
        contact_margin,
        n_bb,
        bb_dist,
        bb_pos,
        bb_n,
    )
    if code < 0:
        return code

    for c in range(n_bb):
        if num_contacts >= MAX_CONTACTS:
            break
        if bb_dist[c] >= contact_margin:
            continue
        var c_off = num_contacts * CONTACT_SIZE
        contacts[env, c_off + CONTACT_IDX_BODY_A] = Scalar[DTYPE](body_a)
        contacts[env, c_off + CONTACT_IDX_BODY_B] = Scalar[DTYPE](body_b)
        contacts[env, c_off + CONTACT_IDX_POS_X] = bb_pos[3 * c + 0]
        contacts[env, c_off + CONTACT_IDX_POS_Y] = bb_pos[3 * c + 1]
        contacts[env, c_off + CONTACT_IDX_POS_Z] = bb_pos[3 * c + 2]
        # `box_box_manifold` returns the normal pointing A -> B; the record's
        # convention is `body_b -> body_a`, which is what the shared emit below
        # gets by negating. Same negation here.
        contacts[env, c_off + CONTACT_IDX_NX] = -bb_n[0]
        contacts[env, c_off + CONTACT_IDX_NY] = -bb_n[1]
        contacts[env, c_off + CONTACT_IDX_NZ] = -bb_n[2]
        contacts[env, c_off + CONTACT_IDX_DIST] = bb_dist[c]
        contacts[env, c_off + CONTACT_IDX_INCLUDEMARGIN] = contact_margin
        contacts[env, c_off + CONTACT_IDX_FRICTION] = contact_friction
        contacts[
            env, c_off + CONTACT_IDX_FRICTION_SPIN
        ] = contact_friction_spin
        contacts[
            env, c_off + CONTACT_IDX_FRICTION_ROLL
        ] = contact_friction_roll
        contacts[env, c_off + CONTACT_IDX_CONDIM] = Scalar[DTYPE](
            contact_condim
        )
        num_contacts += 1
    return code


@always_inline
@always_inline
def _fill_pair_solparams[
    DTYPE: DType,
    MAX_CONTACTS: Int,
    BATCH: Int,
](
    env: Int,
    n0: Int,
    n1: Int,
    mx: InlineArray[Scalar[DTYPE], 12],
    contacts: LayoutTensor[
        DTYPE, Layout.row_major(BATCH, MAX_CONTACTS * CONTACT_SIZE),
        MutAnyOrigin,
    ],
):
    """Stamp a geom pair's mixed solref/solimp onto every contact it emitted.

    The mixed values are constant across all points of one pair, so they are
    written once per pair rather than at each of the nineteen emit sites — a
    narrow-phase branch added later then inherits them instead of silently
    shipping zeros.

    ⚠ CALL THIS AT EVERY EXIT OF THE PAIR LOOP BODY, NOT JUST THE BOTTOM. The
    first version ran only at the bottom, and the two PLANE branches end with
    `continue` — so every plane contact in the engine got solref (0, 0), which
    `solref_spring_damper` then read as the DIRECT form with zero stiffness and
    zero damping. It showed up as `test_contacts_vs_mujoco` failing on hopper
    within one build, but only because that gate exists; a post-pass at the
    bottom of a loop body is safe only when the body has a single exit.
    """
    for c in range(n0, n1):
        var o = c * CONTACT_SIZE
        contacts[env, o + CONTACT_IDX_SOLREF_0] = mx[4]
        contacts[env, o + CONTACT_IDX_SOLREF_1] = mx[5]
        contacts[env, o + CONTACT_IDX_SOLIMP_0] = mx[6]
        contacts[env, o + CONTACT_IDX_SOLIMP_1] = mx[7]
        contacts[env, o + CONTACT_IDX_SOLIMP_2] = mx[8]
        contacts[env, o + CONTACT_IDX_SOLIMP_3] = mx[9]
        contacts[env, o + CONTACT_IDX_SOLIMP_4] = mx[10]


def _detect_contacts_env[
    DTYPE: DType,
    NQ: Int,
    NV: Int,
    NBODY: Int,
    NJOINT: Int,
    MAX_CONTACTS: Int,
    NGEOM: Int,
    NEXCLUDE: Int,
    NMESH_VERTS: Int,
    BATCH: Int,
](
    env: Int,
    xpos: LayoutTensor[
        DTYPE, Layout.row_major(BATCH, NBODY * 3), MutAnyOrigin
    ],
    xquat: LayoutTensor[
        DTYPE, Layout.row_major(BATCH, NBODY * 4), MutAnyOrigin
    ],
    geoms: LayoutTensor[
        DTYPE, Layout.row_major(NGEOM, MODEL_GEOM_SIZE), MutAnyOrigin
    ],
    bodies: LayoutTensor[
        DTYPE, Layout.row_major(NBODY, MODEL_BODY_SIZE), MutAnyOrigin
    ],
    mmeta: LayoutTensor[
        DTYPE, Layout.row_major(MODEL_META_SIZE), MutAnyOrigin
    ],
    excludes: LayoutTensor[
        DTYPE, Layout.row_major(NEXCLUDE, 2), MutAnyOrigin
    ],
    mesh_meta: LayoutTensor[
        DTYPE,
        Layout.row_major(MAX_GPU_MESHES, MODEL_MESH_META_SIZE),
        MutAnyOrigin,
    ],
    mesh_verts: LayoutTensor[
        DTYPE, Layout.row_major(NMESH_VERTS, 3), MutAnyOrigin
    ],
    contacts: LayoutTensor[
        DTYPE, Layout.row_major(BATCH, MAX_CONTACTS * CONTACT_SIZE),
        MutAnyOrigin,
    ],
    smeta: LayoutTensor[
        DTYPE, Layout.row_major(BATCH, METADATA_SIZE), MutAnyOrigin
    ],
):
    """Unified contact detection for one env (verbatim from
    detect_contacts_gpu; mesh branches compiled in iff NMESH_VERTS > 0)."""
    var num_contacts = 0

    for gi in range(NGEOM):
        var gi_type = Int(
            rebind[Scalar[DTYPE]](geoms[gi, GEOM_IDX_TYPE])
        )
        var gi_body = Int(
            rebind[Scalar[DTYPE]](geoms[gi, GEOM_IDX_BODY])
        )
        var gi_contype = Int(
            rebind[Scalar[DTYPE]](geoms[gi, GEOM_IDX_CONTYPE])
        )
        var gi_conaffinity = Int(
            rebind[Scalar[DTYPE]](geoms[gi, GEOM_IDX_CONAFFINITY])
        )
        for gj in range(gi + 1, NGEOM):
            if num_contacts >= MAX_CONTACTS:
                smeta[env, META_IDX_NUM_CONTACTS] = Scalar[DTYPE](
                    num_contacts
                )
                return
            var gj_type = Int(
                rebind[Scalar[DTYPE]](geoms[gj, GEOM_IDX_TYPE])
            )
            var gj_body = Int(
                rebind[Scalar[DTYPE]](geoms[gj, GEOM_IDX_BODY])
            )
            if gi_type == GEOM_PLANE and gj_body == 0:
                continue
            if gj_type == GEOM_PLANE and gi_body == 0:
                continue
            # MuJoCo-style weld body filtering (GPU)
            var weld_i = Int(
                rebind[Scalar[DTYPE]](bodies[gi_body, BODY_IDX_WELDID])
            )
            var weld_j = Int(
                rebind[Scalar[DTYPE]](bodies[gj_body, BODY_IDX_WELDID])
            )
            # Same weld body → filter
            if weld_i == weld_j:
                continue
            # Weld parent check
            if weld_i != 0 and weld_j != 0:
                var wp_i = Int(
                    rebind[Scalar[DTYPE]](bodies[weld_i, BODY_IDX_PARENT])
                )
                var wp_j = Int(
                    rebind[Scalar[DTYPE]](bodies[weld_j, BODY_IDX_PARENT])
                )
                var weld_parent_i = Int(
                    rebind[Scalar[DTYPE]](bodies[wp_i, BODY_IDX_WELDID])
                )
                var weld_parent_j = Int(
                    rebind[Scalar[DTYPE]](bodies[wp_j, BODY_IDX_WELDID])
                )
                if weld_i == weld_parent_j or weld_j == weld_parent_i:
                    continue
                # Check contact exclusion pairs
                var n_ex = Int(
                    rebind[Scalar[DTYPE]](mmeta[MODEL_META_IDX_NEXCLUDE])
                )
                if n_ex > 0:
                    var ba = gi_body if gi_body <= gj_body else gj_body
                    var bb = gj_body if gi_body <= gj_body else gi_body
                    var excluded = False
                    for ex in range(n_ex):
                        var eb1 = Int(
                            rebind[Scalar[DTYPE]](excludes[ex, 0])
                        )
                        var eb2 = Int(
                            rebind[Scalar[DTYPE]](excludes[ex, 1])
                        )
                        if eb1 == ba and eb2 == bb:
                            excluded = True
                            break
                    if excluded:
                        continue
            var gj_contype = Int(
                rebind[Scalar[DTYPE]](geoms[gj, GEOM_IDX_CONTYPE])
            )
            var gj_conaffinity = Int(
                rebind[Scalar[DTYPE]](geoms[gj, GEOM_IDX_CONAFFINITY])
            )
            if (gi_contype & gj_conaffinity) == 0 and (
                gj_contype & gi_conaffinity
            ) == 0:
                continue

            var pi_x: Scalar[DTYPE] = 0
            var pi_y: Scalar[DTYPE] = 0
            var pi_z: Scalar[DTYPE] = 0
            var qi_x: Scalar[DTYPE] = 0
            var qi_y: Scalar[DTYPE] = 0
            var qi_z: Scalar[DTYPE] = 0
            var qi_w: Scalar[DTYPE] = 1
            _geom_world_pos[DTYPE, NBODY, NGEOM, BATCH](
                env,
                gi,
                geoms,
                xpos,
                xquat,
                pi_x,
                pi_y,
                pi_z,
                qi_x,
                qi_y,
                qi_z,
                qi_w,
            )
            var pj_x: Scalar[DTYPE] = 0
            var pj_y: Scalar[DTYPE] = 0
            var pj_z: Scalar[DTYPE] = 0
            var qj_x: Scalar[DTYPE] = 0
            var qj_y: Scalar[DTYPE] = 0
            var qj_z: Scalar[DTYPE] = 0
            var qj_w: Scalar[DTYPE] = 1
            _geom_world_pos[DTYPE, NBODY, NGEOM, BATCH](
                env,
                gj,
                geoms,
                xpos,
                xquat,
                pj_x,
                pj_y,
                pj_z,
                qj_x,
                qj_y,
                qj_z,
                qj_w,
            )

            # Broadphase bounding sphere check (skip for plane geoms — they're infinite)
            if gi_type != GEOM_PLANE and gj_type != GEOM_PLANE:
                var dx = pi_x - pj_x
                var dy = pi_y - pj_y
                var dz = pi_z - pj_z
                var dist_sq = dx * dx + dy * dy + dz * dz
                var ri_bound = rebind[Scalar[DTYPE]](
                    geoms[gi, GEOM_IDX_RBOUND]
                )
                var rj_bound = rebind[Scalar[DTYPE]](
                    geoms[gj, GEOM_IDX_RBOUND]
                )
                var bound = ri_bound + rj_bound
                if dist_sq > bound * bound:
                    continue

            var ri = rebind[Scalar[DTYPE]](geoms[gi, GEOM_IDX_RADIUS])
            var rj = rebind[Scalar[DTYPE]](geoms[gj, GEOM_IDX_RADIUS])
            var hli = rebind[Scalar[DTYPE]](
                geoms[gi, GEOM_IDX_HALF_LENGTH]
            )
            var hlj = rebind[Scalar[DTYPE]](
                geoms[gj, GEOM_IDX_HALF_LENGTH]
            )
            var hxi = rebind[Scalar[DTYPE]](geoms[gi, GEOM_IDX_HALF_X])
            var hyi = rebind[Scalar[DTYPE]](geoms[gi, GEOM_IDX_HALF_Y])
            var hzi = rebind[Scalar[DTYPE]](geoms[gi, GEOM_IDX_HALF_Z])
            var hxj = rebind[Scalar[DTYPE]](geoms[gj, GEOM_IDX_HALF_X])
            var hyj = rebind[Scalar[DTYPE]](geoms[gj, GEOM_IDX_HALF_Y])
            var hzj = rebind[Scalar[DTYPE]](geoms[gj, GEOM_IDX_HALF_Z])
            # Contact parameters: MuJoCo's full rule, PRIORITY FIRST. This
            # used to be an unconditional elementwise max on friction and
            # condim, with solref/solimp not consulted at all — so
            # `<geom priority>` was ignored and the per-geom solparams already
            # in the geom record were dead data. See `mix_contact_params`.
            var _mx = mix_contact_params[DTYPE](
                Int(rebind[Scalar[DTYPE]](geoms[gi, GEOM_IDX_PRIORITY])),
                Int(rebind[Scalar[DTYPE]](geoms[gi, GEOM_IDX_CONDIM])),
                rebind[Scalar[DTYPE]](geoms[gi, GEOM_IDX_FRICTION]),
                rebind[Scalar[DTYPE]](geoms[gi, GEOM_IDX_FRICTION_SPIN]),
                rebind[Scalar[DTYPE]](geoms[gi, GEOM_IDX_FRICTION_ROLL]),
                rebind[Scalar[DTYPE]](geoms[gi, GEOM_IDX_SOLREF_0]),
                rebind[Scalar[DTYPE]](geoms[gi, GEOM_IDX_SOLREF_1]),
                rebind[Scalar[DTYPE]](geoms[gi, GEOM_IDX_SOLIMP_0]),
                rebind[Scalar[DTYPE]](geoms[gi, GEOM_IDX_SOLIMP_1]),
                rebind[Scalar[DTYPE]](geoms[gi, GEOM_IDX_SOLIMP_2]),
                rebind[Scalar[DTYPE]](geoms[gi, GEOM_IDX_SOLIMP_3]),
                rebind[Scalar[DTYPE]](geoms[gi, GEOM_IDX_SOLIMP_4]),
                Int(rebind[Scalar[DTYPE]](geoms[gj, GEOM_IDX_PRIORITY])),
                Int(rebind[Scalar[DTYPE]](geoms[gj, GEOM_IDX_CONDIM])),
                rebind[Scalar[DTYPE]](geoms[gj, GEOM_IDX_FRICTION]),
                rebind[Scalar[DTYPE]](geoms[gj, GEOM_IDX_FRICTION_SPIN]),
                rebind[Scalar[DTYPE]](geoms[gj, GEOM_IDX_FRICTION_ROLL]),
                rebind[Scalar[DTYPE]](geoms[gj, GEOM_IDX_SOLREF_0]),
                rebind[Scalar[DTYPE]](geoms[gj, GEOM_IDX_SOLREF_1]),
                rebind[Scalar[DTYPE]](geoms[gj, GEOM_IDX_SOLIMP_0]),
                rebind[Scalar[DTYPE]](geoms[gj, GEOM_IDX_SOLIMP_1]),
                rebind[Scalar[DTYPE]](geoms[gj, GEOM_IDX_SOLIMP_2]),
                rebind[Scalar[DTYPE]](geoms[gj, GEOM_IDX_SOLIMP_3]),
                rebind[Scalar[DTYPE]](geoms[gj, GEOM_IDX_SOLIMP_4]),
            )
            var contact_condim = Int(_mx[0])
            var contact_friction = _mx[1]
            var contact_friction_spin = _mx[2]
            var contact_friction_roll = _mx[3]
            # First contact slot this PAIR will fill. The mixed solver
            # parameters are constant across every point the pair emits, so
            # they are written once at the bottom of this loop body rather
            # than at each of the nineteen emit sites — which also means a
            # narrow-phase branch added later gets them for free instead of
            # silently shipping zeros.
            var _n0 = num_contacts

            # Margin combination: max of both geoms (MuJoCo convention)
            var margin_gi = rebind[Scalar[DTYPE]](
                geoms[gi, GEOM_IDX_MARGIN]
            )
            var margin_gj = rebind[Scalar[DTYPE]](
                geoms[gj, GEOM_IDX_MARGIN]
            )
            # MuJoCo 3.5+ convention: margin = sum of both geoms
            var contact_margin = margin_gi + margin_gj

            # --- Plane handling ---
            if gi_type == GEOM_PLANE:
                # The plane's FULL pose. This branch used to keep only `pi_z` as a
                # `ground_z` and hardcode the contact normal to (0,0,1), i.e. it modelled
                # every plane as a horizontal floor at the height of its origin. See
                # `collision/plane_frame.mojo`. The other geom is rebased into the PLANE'S
                # FRAME (`fp_*` / `fq_*`), where the plane really is z=0 with normal +z —
                # exactly what every `*_plane` primitive assumes — so the arithmetic below
                # is unchanged and only the contact point and normal map back to world.
                var plp_x = pi_x
                var plp_y = pi_y
                var plp_z = pi_z
                var plq_x = qi_x
                var plq_y = qi_y
                var plq_z = qi_z
                var plq_w = qi_w
                var pn = plane_world_normal[DTYPE](plq_x, plq_y, plq_z, plq_w)
                var lfp = to_plane_frame[DTYPE](
                    plp_x, plp_y, plp_z, plq_x, plq_y, plq_z, plq_w,
                    pj_x, pj_y, pj_z,
                )
                var lfq = quat_to_plane_frame[DTYPE](
                    plq_x, plq_y, plq_z, plq_w, qj_x, qj_y, qj_z, qj_w
                )
                var fp_x = lfp[0]
                var fp_y = lfp[1]
                var fp_z = lfp[2]
                var fq_x = lfq[0]
                var fq_y = lfq[1]
                var fq_z = lfq[2]
                var fq_w = lfq[3]
                var ground_z = Scalar[DTYPE](0)
                if gj_type == GEOM_CAPSULE:
                    # MuJoCo mjc_PlaneCapsule: test BOTH endpoints, up to 2 contacts
                    var axis_w = gpu_quat_rotate(
                        fq_x,
                        fq_y,
                        fq_z,
                        fq_w,
                        Scalar[DTYPE](0),
                        Scalar[DTYPE](0),
                        Scalar[DTYPE](1),
                    )
                    # Endpoint 1: center + half_length * axis
                    # `axis_w` is in the PLANE'S frame (fq_* is the rebased
                    # orientation), which is what the endpoint arithmetic
                    # needs. The FRAME_T1 hint is read in WORLD space, so it
                    # goes back — see collision/contact_frame.mojo for what
                    # that slot is and is not.
                    var axis_wd = gpu_quat_rotate(
                        plq_x, plq_y, plq_z, plq_w,
                        axis_w[0], axis_w[1], axis_w[2],
                    )
                    var e1_x = fp_x + hlj * axis_w[0]
                    var e1_y = fp_y + hlj * axis_w[1]
                    var e1_z = fp_z + hlj * axis_w[2]
                    var dist1 = e1_z - rj - ground_z
                    if dist1 < contact_margin and num_contacts < MAX_CONTACTS:
                        var c_off = num_contacts * CONTACT_SIZE
                        contacts[env, c_off + CONTACT_IDX_BODY_A] = Scalar[
                            DTYPE
                        ](gj_body)
                        contacts[env, c_off + CONTACT_IDX_BODY_B] = Scalar[
                            DTYPE
                        ](0)
                        var cw = from_plane_frame[DTYPE](
                            plp_x, plp_y, plp_z, plq_x, plq_y, plq_z, plq_w,
                            e1_x, e1_y,
                            ground_z + dist1 * Scalar[DTYPE](0.5),
                        )
                        contacts[env, c_off + CONTACT_IDX_POS_X] = cw[0]
                        contacts[env, c_off + CONTACT_IDX_POS_Y] = cw[1]
                        contacts[env, c_off + CONTACT_IDX_POS_Z] = cw[2]
                        contacts[env, c_off + CONTACT_IDX_NX] = pn[0]
                        contacts[env, c_off + CONTACT_IDX_NY] = pn[1]
                        contacts[env, c_off + CONTACT_IDX_NZ] = pn[2]
                        contacts[env, c_off + CONTACT_IDX_DIST] = dist1
                        contacts[
                            env, c_off + CONTACT_IDX_INCLUDEMARGIN
                        ] = contact_margin
                        contacts[
                            env, c_off + CONTACT_IDX_FRICTION
                        ] = contact_friction
                        contacts[
                            env, c_off + CONTACT_IDX_FRICTION_SPIN
                        ] = contact_friction_spin
                        contacts[
                            env, c_off + CONTACT_IDX_FRICTION_ROLL
                        ] = contact_friction_roll
                        contacts[env, c_off + CONTACT_IDX_CONDIM] = Scalar[
                            DTYPE
                        ](contact_condim)
                        contacts[env, c_off + CONTACT_IDX_FRAME_T1_X] = axis_wd[
                            0
                        ]
                        contacts[env, c_off + CONTACT_IDX_FRAME_T1_Y] = axis_wd[
                            1
                        ]
                        contacts[env, c_off + CONTACT_IDX_FRAME_T1_Z] = axis_wd[
                            2
                        ]
                        num_contacts += 1
                    # Endpoint 2: center - half_length * axis
                    var e2_x = fp_x - hlj * axis_w[0]
                    var e2_y = fp_y - hlj * axis_w[1]
                    var e2_z = fp_z - hlj * axis_w[2]
                    var dist2 = e2_z - rj - ground_z
                    if dist2 < contact_margin and num_contacts < MAX_CONTACTS:
                        var c_off = num_contacts * CONTACT_SIZE
                        contacts[env, c_off + CONTACT_IDX_BODY_A] = Scalar[
                            DTYPE
                        ](gj_body)
                        contacts[env, c_off + CONTACT_IDX_BODY_B] = Scalar[
                            DTYPE
                        ](0)
                        var cw = from_plane_frame[DTYPE](
                            plp_x, plp_y, plp_z, plq_x, plq_y, plq_z, plq_w,
                            e2_x, e2_y,
                            ground_z + dist2 * Scalar[DTYPE](0.5),
                        )
                        contacts[env, c_off + CONTACT_IDX_POS_X] = cw[0]
                        contacts[env, c_off + CONTACT_IDX_POS_Y] = cw[1]
                        contacts[env, c_off + CONTACT_IDX_POS_Z] = cw[2]
                        contacts[env, c_off + CONTACT_IDX_NX] = pn[0]
                        contacts[env, c_off + CONTACT_IDX_NY] = pn[1]
                        contacts[env, c_off + CONTACT_IDX_NZ] = pn[2]
                        contacts[env, c_off + CONTACT_IDX_DIST] = dist2
                        contacts[
                            env, c_off + CONTACT_IDX_INCLUDEMARGIN
                        ] = contact_margin
                        contacts[
                            env, c_off + CONTACT_IDX_FRICTION
                        ] = contact_friction
                        contacts[
                            env, c_off + CONTACT_IDX_FRICTION_SPIN
                        ] = contact_friction_spin
                        contacts[
                            env, c_off + CONTACT_IDX_FRICTION_ROLL
                        ] = contact_friction_roll
                        contacts[env, c_off + CONTACT_IDX_CONDIM] = Scalar[
                            DTYPE
                        ](contact_condim)
                        contacts[env, c_off + CONTACT_IDX_FRAME_T1_X] = axis_wd[
                            0
                        ]
                        contacts[env, c_off + CONTACT_IDX_FRAME_T1_Y] = axis_wd[
                            1
                        ]
                        contacts[env, c_off + CONTACT_IDX_FRAME_T1_Z] = axis_wd[
                            2
                        ]
                        num_contacts += 1
                elif gj_type == GEOM_CYLINDER:
                    # Up to FOUR points — two rim, two triangle — not
                    # one. See `_plane_cylinder_contacts`; a cylinder on
                    # its flat face needs a support polygon or it tips.
                    _plane_cylinder_contacts[DTYPE, MAX_CONTACTS, BATCH](
                        env,
                        gj_body,
                        fp_x, fp_y, fp_z,
                        fq_x, fq_y, fq_z, fq_w,
                        rj,
                        hlj,
                        ground_z,
                        plp_x, plp_y, plp_z,
                        plq_x, plq_y, plq_z, plq_w,
                        contact_margin,
                        contact_friction,
                        contact_friction_spin,
                        contact_friction_roll,
                        contact_condim,
                        0,
                        contacts,
                        num_contacts,
                    )
                elif gj_type == GEOM_SPHERE:
                    var dist = fp_z - rj - ground_z
                    if dist < contact_margin and num_contacts < MAX_CONTACTS:
                        var c_off = num_contacts * CONTACT_SIZE
                        contacts[env, c_off + CONTACT_IDX_BODY_A] = Scalar[
                            DTYPE
                        ](gj_body)
                        contacts[env, c_off + CONTACT_IDX_BODY_B] = Scalar[
                            DTYPE
                        ](0)
                        var cw = from_plane_frame[DTYPE](
                            plp_x, plp_y, plp_z, plq_x, plq_y, plq_z, plq_w,
                            fp_x, fp_y,
                            ground_z + dist * Scalar[DTYPE](0.5),
                        )
                        contacts[env, c_off + CONTACT_IDX_POS_X] = cw[0]
                        contacts[env, c_off + CONTACT_IDX_POS_Y] = cw[1]
                        contacts[env, c_off + CONTACT_IDX_POS_Z] = cw[2]
                        contacts[env, c_off + CONTACT_IDX_NX] = pn[0]
                        contacts[env, c_off + CONTACT_IDX_NY] = pn[1]
                        contacts[env, c_off + CONTACT_IDX_NZ] = pn[2]
                        contacts[env, c_off + CONTACT_IDX_DIST] = dist
                        contacts[
                            env, c_off + CONTACT_IDX_INCLUDEMARGIN
                        ] = contact_margin
                        contacts[
                            env, c_off + CONTACT_IDX_FRICTION
                        ] = contact_friction
                        contacts[
                            env, c_off + CONTACT_IDX_FRICTION_SPIN
                        ] = contact_friction_spin
                        contacts[
                            env, c_off + CONTACT_IDX_FRICTION_ROLL
                        ] = contact_friction_roll
                        contacts[env, c_off + CONTACT_IDX_CONDIM] = Scalar[
                            DTYPE
                        ](contact_condim)
                        num_contacts += 1
                elif gj_type == GEOM_ELLIPSOID:
                    # MuJoCo routes plane x ellipsoid through mjc_PlaneConvex,
                    # which reports the single deepest support point. A smooth
                    # strictly-convex surface touches a plane at one point, and
                    # MuJoCo was measured to emit exactly 1 contact over 500
                    # random poses — so, unlike box_plane, there is no second
                    # contact to look for.
                    var ep = ellipsoid_plane[DTYPE](
                        fp_x, fp_y, fp_z,
                        fq_x, fq_y, fq_z, fq_w,
                        hxj, hyj, hzj,
                        ground_z,
                    )
                    var dist = ep[0]
                    if dist < contact_margin and num_contacts < MAX_CONTACTS:
                        var c_off = num_contacts * CONTACT_SIZE
                        contacts[env, c_off + CONTACT_IDX_BODY_A] = Scalar[
                            DTYPE
                        ](gj_body)
                        contacts[env, c_off + CONTACT_IDX_BODY_B] = Scalar[
                            DTYPE
                        ](0)
                        var cw = from_plane_frame[DTYPE](
                            plp_x, plp_y, plp_z, plq_x, plq_y, plq_z, plq_w,
                            ep[1], ep[2], ep[3],
                        )
                        contacts[env, c_off + CONTACT_IDX_POS_X] = cw[0]
                        contacts[env, c_off + CONTACT_IDX_POS_Y] = cw[1]
                        contacts[env, c_off + CONTACT_IDX_POS_Z] = cw[2]
                        contacts[env, c_off + CONTACT_IDX_NX] = pn[0]
                        contacts[env, c_off + CONTACT_IDX_NY] = pn[1]
                        contacts[env, c_off + CONTACT_IDX_NZ] = pn[2]
                        contacts[env, c_off + CONTACT_IDX_DIST] = dist
                        contacts[
                            env, c_off + CONTACT_IDX_INCLUDEMARGIN
                        ] = contact_margin
                        contacts[
                            env, c_off + CONTACT_IDX_FRICTION
                        ] = contact_friction
                        contacts[
                            env, c_off + CONTACT_IDX_FRICTION_SPIN
                        ] = contact_friction_spin
                        contacts[
                            env, c_off + CONTACT_IDX_FRICTION_ROLL
                        ] = contact_friction_roll
                        contacts[env, c_off + CONTACT_IDX_CONDIM] = Scalar[
                            DTYPE
                        ](contact_condim)
                        num_contacts += 1
                elif gj_type == GEOM_BOX:
                    # Up to FOUR corners, not one — see
                    # `_plane_box_contacts` and task #42.
                    _plane_box_contacts[DTYPE, MAX_CONTACTS, BATCH](
                        env,
                        gj_body,
                        fp_x, fp_y, fp_z,
                        fq_x, fq_y, fq_z, fq_w,
                        hxj, hyj, hzj,
                        ground_z,
                        plp_x, plp_y, plp_z,
                        plq_x, plq_y, plq_z, plq_w,
                        contact_margin,
                        contact_friction,
                        contact_friction_spin,
                        contact_friction_roll,
                        contact_condim,
                        0,
                        contacts,
                        num_contacts,
                    )
                elif gj_type == GEOM_MESH:
                    # Plane-mesh: scan hull vertices below plane
                    comptime if NMESH_VERTS > 0:
                        _plane_mesh_contacts[
                            DTYPE, MAX_CONTACTS, NGEOM, NMESH_VERTS, BATCH
                        ](
                            env,
                            gj,
                            gj_body,
                            fp_x,
                            fp_y,
                            fp_z,
                            fq_x,
                            fq_y,
                            fq_z,
                            fq_w,
                            ground_z,
                            plp_x, plp_y, plp_z,
                            plq_x, plq_y, plq_z, plq_w,
                            contact_margin,
                            contact_friction,
                            contact_friction_spin,
                            contact_friction_roll,
                            contact_condim,
                            geoms,
                            mesh_meta,
                            mesh_verts,
                            contacts,
                            num_contacts,
                        )
                _fill_pair_solparams[DTYPE, MAX_CONTACTS, BATCH](
                    env, _n0, num_contacts, _mx, contacts
                )
                continue

            if gj_type == GEOM_PLANE:
                # The plane's FULL pose. This branch used to keep only `pj_z` as a
                # `ground_z` and hardcode the contact normal to (0,0,1), i.e. it modelled
                # every plane as a horizontal floor at the height of its origin. See
                # `collision/plane_frame.mojo`. The other geom is rebased into the PLANE'S
                # FRAME (`fp_*` / `fq_*`), where the plane really is z=0 with normal +z —
                # exactly what every `*_plane` primitive assumes — so the arithmetic below
                # is unchanged and only the contact point and normal map back to world.
                var plp_x = pj_x
                var plp_y = pj_y
                var plp_z = pj_z
                var plq_x = qj_x
                var plq_y = qj_y
                var plq_z = qj_z
                var plq_w = qj_w
                var pn = plane_world_normal[DTYPE](plq_x, plq_y, plq_z, plq_w)
                var lfp = to_plane_frame[DTYPE](
                    plp_x, plp_y, plp_z, plq_x, plq_y, plq_z, plq_w,
                    pi_x, pi_y, pi_z,
                )
                var lfq = quat_to_plane_frame[DTYPE](
                    plq_x, plq_y, plq_z, plq_w, qi_x, qi_y, qi_z, qi_w
                )
                var fp_x = lfp[0]
                var fp_y = lfp[1]
                var fp_z = lfp[2]
                var fq_x = lfq[0]
                var fq_y = lfq[1]
                var fq_z = lfq[2]
                var fq_w = lfq[3]
                var ground_z = Scalar[DTYPE](0)
                if gi_type == GEOM_CAPSULE:
                    # MuJoCo mjc_PlaneCapsule: test BOTH endpoints, up to 2 contacts
                    var axis_w = gpu_quat_rotate(
                        fq_x,
                        fq_y,
                        fq_z,
                        fq_w,
                        Scalar[DTYPE](0),
                        Scalar[DTYPE](0),
                        Scalar[DTYPE](1),
                    )
                    # `axis_w` is in the PLANE'S frame; the FRAME_T1 hint is
                    # read in WORLD space, so it goes back.
                    var axis_wd = gpu_quat_rotate(
                        plq_x, plq_y, plq_z, plq_w,
                        axis_w[0], axis_w[1], axis_w[2],
                    )
                    # Endpoint 1: center + half_length * axis
                    var e1_x = fp_x + hli * axis_w[0]
                    var e1_y = fp_y + hli * axis_w[1]
                    var e1_z = fp_z + hli * axis_w[2]
                    var dist1 = e1_z - ri - ground_z
                    if dist1 < contact_margin and num_contacts < MAX_CONTACTS:
                        var c_off = num_contacts * CONTACT_SIZE
                        contacts[env, c_off + CONTACT_IDX_BODY_A] = Scalar[
                            DTYPE
                        ](gi_body)
                        contacts[env, c_off + CONTACT_IDX_BODY_B] = Scalar[
                            DTYPE
                        ](0)
                        var cw = from_plane_frame[DTYPE](
                            plp_x, plp_y, plp_z, plq_x, plq_y, plq_z, plq_w,
                            e1_x, e1_y,
                            ground_z + dist1 * Scalar[DTYPE](0.5),
                        )
                        contacts[env, c_off + CONTACT_IDX_POS_X] = cw[0]
                        contacts[env, c_off + CONTACT_IDX_POS_Y] = cw[1]
                        contacts[env, c_off + CONTACT_IDX_POS_Z] = cw[2]
                        contacts[env, c_off + CONTACT_IDX_NX] = pn[0]
                        contacts[env, c_off + CONTACT_IDX_NY] = pn[1]
                        contacts[env, c_off + CONTACT_IDX_NZ] = pn[2]
                        contacts[env, c_off + CONTACT_IDX_DIST] = dist1
                        contacts[
                            env, c_off + CONTACT_IDX_INCLUDEMARGIN
                        ] = contact_margin
                        contacts[
                            env, c_off + CONTACT_IDX_FRICTION
                        ] = contact_friction
                        contacts[
                            env, c_off + CONTACT_IDX_FRICTION_SPIN
                        ] = contact_friction_spin
                        contacts[
                            env, c_off + CONTACT_IDX_FRICTION_ROLL
                        ] = contact_friction_roll
                        contacts[env, c_off + CONTACT_IDX_CONDIM] = Scalar[
                            DTYPE
                        ](contact_condim)
                        contacts[env, c_off + CONTACT_IDX_FRAME_T1_X] = axis_wd[
                            0
                        ]
                        contacts[env, c_off + CONTACT_IDX_FRAME_T1_Y] = axis_wd[
                            1
                        ]
                        contacts[env, c_off + CONTACT_IDX_FRAME_T1_Z] = axis_wd[
                            2
                        ]
                        num_contacts += 1
                    # Endpoint 2: center - half_length * axis
                    var e2_x = fp_x - hli * axis_w[0]
                    var e2_y = fp_y - hli * axis_w[1]
                    var e2_z = fp_z - hli * axis_w[2]
                    var dist2 = e2_z - ri - ground_z
                    if dist2 < contact_margin and num_contacts < MAX_CONTACTS:
                        var c_off = num_contacts * CONTACT_SIZE
                        contacts[env, c_off + CONTACT_IDX_BODY_A] = Scalar[
                            DTYPE
                        ](gi_body)
                        contacts[env, c_off + CONTACT_IDX_BODY_B] = Scalar[
                            DTYPE
                        ](0)
                        var cw = from_plane_frame[DTYPE](
                            plp_x, plp_y, plp_z, plq_x, plq_y, plq_z, plq_w,
                            e2_x, e2_y,
                            ground_z + dist2 * Scalar[DTYPE](0.5),
                        )
                        contacts[env, c_off + CONTACT_IDX_POS_X] = cw[0]
                        contacts[env, c_off + CONTACT_IDX_POS_Y] = cw[1]
                        contacts[env, c_off + CONTACT_IDX_POS_Z] = cw[2]
                        contacts[env, c_off + CONTACT_IDX_NX] = pn[0]
                        contacts[env, c_off + CONTACT_IDX_NY] = pn[1]
                        contacts[env, c_off + CONTACT_IDX_NZ] = pn[2]
                        contacts[env, c_off + CONTACT_IDX_DIST] = dist2
                        contacts[
                            env, c_off + CONTACT_IDX_INCLUDEMARGIN
                        ] = contact_margin
                        contacts[
                            env, c_off + CONTACT_IDX_FRICTION
                        ] = contact_friction
                        contacts[
                            env, c_off + CONTACT_IDX_FRICTION_SPIN
                        ] = contact_friction_spin
                        contacts[
                            env, c_off + CONTACT_IDX_FRICTION_ROLL
                        ] = contact_friction_roll
                        contacts[env, c_off + CONTACT_IDX_CONDIM] = Scalar[
                            DTYPE
                        ](contact_condim)
                        contacts[env, c_off + CONTACT_IDX_FRAME_T1_X] = axis_wd[
                            0
                        ]
                        contacts[env, c_off + CONTACT_IDX_FRAME_T1_Y] = axis_wd[
                            1
                        ]
                        contacts[env, c_off + CONTACT_IDX_FRAME_T1_Z] = axis_wd[
                            2
                        ]
                        num_contacts += 1
                elif gi_type == GEOM_CYLINDER:
                    # Up to FOUR points — two rim, two triangle — not
                    # one. See `_plane_cylinder_contacts`; a cylinder on
                    # its flat face needs a support polygon or it tips.
                    _plane_cylinder_contacts[DTYPE, MAX_CONTACTS, BATCH](
                        env,
                        gi_body,
                        fp_x, fp_y, fp_z,
                        fq_x, fq_y, fq_z, fq_w,
                        ri,
                        hli,
                        ground_z,
                        plp_x, plp_y, plp_z,
                        plq_x, plq_y, plq_z, plq_w,
                        contact_margin,
                        contact_friction,
                        contact_friction_spin,
                        contact_friction_roll,
                        contact_condim,
                        0,
                        contacts,
                        num_contacts,
                    )
                elif gi_type == GEOM_SPHERE:
                    var dist = fp_z - ri - ground_z
                    if dist < contact_margin and num_contacts < MAX_CONTACTS:
                        var c_off = num_contacts * CONTACT_SIZE
                        contacts[env, c_off + CONTACT_IDX_BODY_A] = Scalar[
                            DTYPE
                        ](gi_body)
                        contacts[env, c_off + CONTACT_IDX_BODY_B] = Scalar[
                            DTYPE
                        ](0)
                        var cw = from_plane_frame[DTYPE](
                            plp_x, plp_y, plp_z, plq_x, plq_y, plq_z, plq_w,
                            fp_x, fp_y,
                            ground_z + dist * Scalar[DTYPE](0.5),
                        )
                        contacts[env, c_off + CONTACT_IDX_POS_X] = cw[0]
                        contacts[env, c_off + CONTACT_IDX_POS_Y] = cw[1]
                        contacts[env, c_off + CONTACT_IDX_POS_Z] = cw[2]
                        contacts[env, c_off + CONTACT_IDX_NX] = pn[0]
                        contacts[env, c_off + CONTACT_IDX_NY] = pn[1]
                        contacts[env, c_off + CONTACT_IDX_NZ] = pn[2]
                        contacts[env, c_off + CONTACT_IDX_DIST] = dist
                        contacts[
                            env, c_off + CONTACT_IDX_INCLUDEMARGIN
                        ] = contact_margin
                        contacts[
                            env, c_off + CONTACT_IDX_FRICTION
                        ] = contact_friction
                        contacts[
                            env, c_off + CONTACT_IDX_FRICTION_SPIN
                        ] = contact_friction_spin
                        contacts[
                            env, c_off + CONTACT_IDX_FRICTION_ROLL
                        ] = contact_friction_roll
                        contacts[env, c_off + CONTACT_IDX_CONDIM] = Scalar[
                            DTYPE
                        ](contact_condim)
                        num_contacts += 1
                elif gi_type == GEOM_ELLIPSOID:
                    # Mirror of the gj branch above; see it for why one
                    # contact is the whole story.
                    var ep = ellipsoid_plane[DTYPE](
                        fp_x, fp_y, fp_z,
                        fq_x, fq_y, fq_z, fq_w,
                        hxi, hyi, hzi,
                        ground_z,
                    )
                    var dist = ep[0]
                    if dist < contact_margin and num_contacts < MAX_CONTACTS:
                        var c_off = num_contacts * CONTACT_SIZE
                        contacts[env, c_off + CONTACT_IDX_BODY_A] = Scalar[
                            DTYPE
                        ](gi_body)
                        contacts[env, c_off + CONTACT_IDX_BODY_B] = Scalar[
                            DTYPE
                        ](0)
                        var cw = from_plane_frame[DTYPE](
                            plp_x, plp_y, plp_z, plq_x, plq_y, plq_z, plq_w,
                            ep[1], ep[2], ep[3],
                        )
                        contacts[env, c_off + CONTACT_IDX_POS_X] = cw[0]
                        contacts[env, c_off + CONTACT_IDX_POS_Y] = cw[1]
                        contacts[env, c_off + CONTACT_IDX_POS_Z] = cw[2]
                        contacts[env, c_off + CONTACT_IDX_NX] = pn[0]
                        contacts[env, c_off + CONTACT_IDX_NY] = pn[1]
                        contacts[env, c_off + CONTACT_IDX_NZ] = pn[2]
                        contacts[env, c_off + CONTACT_IDX_DIST] = dist
                        contacts[
                            env, c_off + CONTACT_IDX_INCLUDEMARGIN
                        ] = contact_margin
                        contacts[
                            env, c_off + CONTACT_IDX_FRICTION
                        ] = contact_friction
                        contacts[
                            env, c_off + CONTACT_IDX_FRICTION_SPIN
                        ] = contact_friction_spin
                        contacts[
                            env, c_off + CONTACT_IDX_FRICTION_ROLL
                        ] = contact_friction_roll
                        contacts[env, c_off + CONTACT_IDX_CONDIM] = Scalar[
                            DTYPE
                        ](contact_condim)
                        num_contacts += 1
                elif gi_type == GEOM_BOX:
                    # Up to FOUR corners, not one — see
                    # `_plane_box_contacts` and task #42.
                    _plane_box_contacts[DTYPE, MAX_CONTACTS, BATCH](
                        env,
                        gi_body,
                        fp_x, fp_y, fp_z,
                        fq_x, fq_y, fq_z, fq_w,
                        hxi, hyi, hzi,
                        ground_z,
                        plp_x, plp_y, plp_z,
                        plq_x, plq_y, plq_z, plq_w,
                        contact_margin,
                        contact_friction,
                        contact_friction_spin,
                        contact_friction_roll,
                        contact_condim,
                        0,
                        contacts,
                        num_contacts,
                    )
                elif gi_type == GEOM_MESH:
                    comptime if NMESH_VERTS > 0:
                        _plane_mesh_contacts[
                            DTYPE, MAX_CONTACTS, NGEOM, NMESH_VERTS, BATCH
                        ](
                            env,
                            gi,
                            gi_body,
                            fp_x,
                            fp_y,
                            fp_z,
                            fq_x,
                            fq_y,
                            fq_z,
                            fq_w,
                            ground_z,
                            plp_x, plp_y, plp_z,
                            plq_x, plq_y, plq_z, plq_w,
                            contact_margin,
                            contact_friction,
                            contact_friction_spin,
                            contact_friction_roll,
                            contact_condim,
                            geoms,
                            mesh_meta,
                            mesh_verts,
                            contacts,
                            num_contacts,
                        )
                _fill_pair_solparams[DTYPE, MAX_CONTACTS, BATCH](
                    env, _n0, num_contacts, _mx, contacts
                )
                continue

            # --- Non-plane geom pair ---
            var dist: Scalar[DTYPE] = 1.0
            var cx: Scalar[DTYPE] = 0
            var cy: Scalar[DTYPE] = 0
            var cz: Scalar[DTYPE] = 0
            var nx: Scalar[DTYPE] = 0
            var ny: Scalar[DTYPE] = 0
            var nz: Scalar[DTYPE] = 1
            # CONTACT DIRECTION INVARIANT — see the same note in
            # `broadphase_sap.mojo`. Every branch below emits
            # `normal = gi -> gj` with `body_a = gi_body, body_b = gj_body`.
            # The reversed-order branches negate the primitive's normal to get
            # there; they must NOT also swap the bodies, because the double
            # flip lands back on `body_b -> body_a` and desynchronises
            # `jar = aref + J*qacc` (aref is built from the penetration depth
            # and does not flip with the normal).
            var body_a = gi_body
            var body_b = gj_body

            if gi_type == GEOM_SPHERE and gj_type == GEOM_SPHERE:
                var r = sphere_sphere[DTYPE](
                    pi_x, pi_y, pi_z, ri, pj_x, pj_y, pj_z, rj
                )
                dist = r[0]
                cx = r[1]
                cy = r[2]
                cz = r[3]
                nx = r[4]
                ny = r[5]
                nz = r[6]
            elif gi_type == GEOM_CAPSULE and gj_type == GEOM_SPHERE:
                var r = capsule_sphere[DTYPE](
                    pi_x,
                    pi_y,
                    pi_z,
                    qi_x,
                    qi_y,
                    qi_z,
                    qi_w,
                    hli,
                    ri,
                    pj_x,
                    pj_y,
                    pj_z,
                    rj,
                )
                dist = r[0]
                cx = r[1]
                cy = r[2]
                cz = r[3]
                nx = r[4]
                ny = r[5]
                nz = r[6]
            elif gi_type == GEOM_SPHERE and gj_type == GEOM_CAPSULE:
                var r = capsule_sphere[DTYPE](
                    pj_x,
                    pj_y,
                    pj_z,
                    qj_x,
                    qj_y,
                    qj_z,
                    qj_w,
                    hlj,
                    rj,
                    pi_x,
                    pi_y,
                    pi_z,
                    ri,
                )
                dist = r[0]
                cx = r[1]
                cy = r[2]
                cz = r[3]
                nx = -r[4]
                ny = -r[5]
                nz = -r[6]
            elif gi_type == GEOM_CAPSULE and gj_type == GEOM_CAPSULE:
                var r = capsule_capsule[DTYPE](
                    pi_x,
                    pi_y,
                    pi_z,
                    qi_x,
                    qi_y,
                    qi_z,
                    qi_w,
                    hli,
                    ri,
                    pj_x,
                    pj_y,
                    pj_z,
                    qj_x,
                    qj_y,
                    qj_z,
                    qj_w,
                    hlj,
                    rj,
                )
                dist = r[0]
                cx = r[1]
                cy = r[2]
                cz = r[3]
                nx = r[4]
                ny = r[5]
                nz = r[6]
            elif gi_type == GEOM_BOX and gj_type == GEOM_SPHERE:
                var r = box_sphere[DTYPE](
                    pi_x,
                    pi_y,
                    pi_z,
                    qi_x,
                    qi_y,
                    qi_z,
                    qi_w,
                    hxi,
                    hyi,
                    hzi,
                    pj_x,
                    pj_y,
                    pj_z,
                    rj,
                )
                dist = r[0]
                cx = r[1]
                cy = r[2]
                cz = r[3]
                nx = r[4]
                ny = r[5]
                nz = r[6]
            elif gi_type == GEOM_SPHERE and gj_type == GEOM_BOX:
                var r = box_sphere[DTYPE](
                    pj_x,
                    pj_y,
                    pj_z,
                    qj_x,
                    qj_y,
                    qj_z,
                    qj_w,
                    hxj,
                    hyj,
                    hzj,
                    pi_x,
                    pi_y,
                    pi_z,
                    ri,
                )
                dist = r[0]
                cx = r[1]
                cy = r[2]
                cz = r[3]
                nx = -r[4]
                ny = -r[5]
                nz = -r[6]
            elif gi_type == GEOM_BOX and gj_type == GEOM_CAPSULE:
                # A capsule along a box face is a two-point manifold — see
                # `_capsule_box_contacts`, which writes its own records.
                _ = _capsule_box_contacts[DTYPE, MAX_CONTACTS, BATCH](
                    env, gi_body, gj_body,
                    pi_x, pi_y, pi_z, qi_x, qi_y, qi_z, qi_w, hxi, hyi, hzi,
                    pj_x, pj_y, pj_z, qj_x, qj_y, qj_z, qj_w, hlj, rj,
                    Scalar[DTYPE](-1),
                    contact_margin,
                    contact_friction,
                    contact_friction_spin,
                    contact_friction_roll,
                    contact_condim,
                    contacts, num_contacts,
                )
                _fill_pair_solparams[DTYPE, MAX_CONTACTS, BATCH](
                    env, _n0, num_contacts, _mx, contacts
                )
                continue
            elif gi_type == GEOM_CAPSULE and gj_type == GEOM_BOX:
                _ = _capsule_box_contacts[DTYPE, MAX_CONTACTS, BATCH](
                    env, gi_body, gj_body,
                    pj_x, pj_y, pj_z, qj_x, qj_y, qj_z, qj_w, hxj, hyj, hzj,
                    pi_x, pi_y, pi_z, qi_x, qi_y, qi_z, qi_w, hli, ri,
                    Scalar[DTYPE](1),
                    contact_margin,
                    contact_friction,
                    contact_friction_spin,
                    contact_friction_roll,
                    contact_condim,
                    contacts, num_contacts,
                )
                _fill_pair_solparams[DTYPE, MAX_CONTACTS, BATCH](
                    env, _n0, num_contacts, _mx, contacts
                )
                continue
            elif gi_type == GEOM_BOX and gj_type == GEOM_BOX:
                # A box/box contact is a whole manifold, not a point — see
                # `_box_box_contacts`. It writes its own records and this
                # branch is done; only a SEPARATED pair (code -1) falls through
                # to `box_box`, which then rejects it too.
                var code = _box_box_contacts[DTYPE, MAX_CONTACTS, BATCH](
                    env,
                    gi_body,
                    gj_body,
                    pi_x, pi_y, pi_z, qi_x, qi_y, qi_z, qi_w, hxi, hyi, hzi,
                    pj_x, pj_y, pj_z, qj_x, qj_y, qj_z, qj_w, hxj, hyj, hzj,
                    contact_margin,
                    contact_friction,
                    contact_friction_spin,
                    contact_friction_roll,
                    contact_condim,
                    contacts,
                    num_contacts,
                )
                if code >= 0:
                    _fill_pair_solparams[DTYPE, MAX_CONTACTS, BATCH](
                        env, _n0, num_contacts, _mx, contacts
                    )
                    continue
                var r = box_box[DTYPE](
                    pi_x,
                    pi_y,
                    pi_z,
                    qi_x,
                    qi_y,
                    qi_z,
                    qi_w,
                    hxi,
                    hyi,
                    hzi,
                    pj_x,
                    pj_y,
                    pj_z,
                    qj_x,
                    qj_y,
                    qj_z,
                    qj_w,
                    hxj,
                    hyj,
                    hzj,
                )
                dist = r[0]
                cx = r[1]
                cy = r[2]
                cz = r[3]
                nx = r[4]
                ny = r[5]
                nz = r[6]
            elif gi_type == GEOM_CYLINDER and gj_type == GEOM_SPHERE:
                var r = cylinder_sphere[DTYPE](
                    pi_x,
                    pi_y,
                    pi_z,
                    qi_x,
                    qi_y,
                    qi_z,
                    qi_w,
                    hli,
                    ri,
                    pj_x,
                    pj_y,
                    pj_z,
                    rj,
                )
                dist = r[0]
                cx = r[1]
                cy = r[2]
                cz = r[3]
                nx = r[4]
                ny = r[5]
                nz = r[6]
            elif gi_type == GEOM_SPHERE and gj_type == GEOM_CYLINDER:
                var r = cylinder_sphere[DTYPE](
                    pj_x,
                    pj_y,
                    pj_z,
                    qj_x,
                    qj_y,
                    qj_z,
                    qj_w,
                    hlj,
                    rj,
                    pi_x,
                    pi_y,
                    pi_z,
                    ri,
                )
                dist = r[0]
                cx = r[1]
                cy = r[2]
                cz = r[3]
                nx = -r[4]
                ny = -r[5]
                nz = -r[6]
            elif gi_type == GEOM_CYLINDER and gj_type == GEOM_CAPSULE:
                var r = cylinder_capsule[DTYPE](
                    pi_x, pi_y, pi_z, qi_x, qi_y, qi_z, qi_w, hli, ri,
                    pj_x, pj_y, pj_z, qj_x, qj_y, qj_z, qj_w, hlj, rj,
                )
                dist = r[0]
                cx = r[1]
                cy = r[2]
                cz = r[3]
                nx = r[4]
                ny = r[5]
                nz = r[6]
            elif gi_type == GEOM_CAPSULE and gj_type == GEOM_CYLINDER:
                var r = cylinder_capsule[DTYPE](
                    pj_x, pj_y, pj_z, qj_x, qj_y, qj_z, qj_w, hlj, rj,
                    pi_x, pi_y, pi_z, qi_x, qi_y, qi_z, qi_w, hli, ri,
                )
                dist = r[0]
                cx = r[1]
                cy = r[2]
                cz = r[3]
                nx = -r[4]
                ny = -r[5]
                nz = -r[6]
            elif gi_type == GEOM_CYLINDER and gj_type == GEOM_CYLINDER:
                var r = cylinder_cylinder[DTYPE](
                    pi_x, pi_y, pi_z, qi_x, qi_y, qi_z, qi_w, hli, ri,
                    pj_x, pj_y, pj_z, qj_x, qj_y, qj_z, qj_w, hlj, rj,
                )
                dist = r[0]
                cx = r[1]
                cy = r[2]
                cz = r[3]
                nx = r[4]
                ny = r[5]
                nz = r[6]
            elif gi_type == GEOM_CYLINDER and gj_type == GEOM_BOX:
                var r = cylinder_box[DTYPE](
                    pi_x, pi_y, pi_z, qi_x, qi_y, qi_z, qi_w, hli, ri,
                    pj_x, pj_y, pj_z, qj_x, qj_y, qj_z, qj_w, hxj, hyj, hzj,
                )
                dist = r[0]
                cx = r[1]
                cy = r[2]
                cz = r[3]
                nx = r[4]
                ny = r[5]
                nz = r[6]
            elif gi_type == GEOM_BOX and gj_type == GEOM_CYLINDER:
                var r = cylinder_box[DTYPE](
                    pj_x, pj_y, pj_z, qj_x, qj_y, qj_z, qj_w, hlj, rj,
                    pi_x, pi_y, pi_z, qi_x, qi_y, qi_z, qi_w, hxi, hyi, hzi,
                )
                dist = r[0]
                cx = r[1]
                cy = r[2]
                cz = r[3]
                nx = -r[4]
                ny = -r[5]
                nz = -r[6]

            # GJK/EPA fallback for any pair involving a mesh geom
            elif gi_type == GEOM_MESH or gj_type == GEOM_MESH:
                comptime if NMESH_VERTS > 0:
                    # Read mesh IDs from geom data
                    var mi_id = Int(
                        rebind[Scalar[DTYPE]](geoms[gi, GEOM_IDX_MESH_ID])
                    )
                    var mj_id = Int(
                        rebind[Scalar[DTYPE]](geoms[gj, GEOM_IDX_MESH_ID])
                    )
                    # Resolve mesh vertex ranges from mesh_meta records
                    var va1 = 0
                    var mnv1 = 0
                    if mi_id >= 0:
                        va1 = Int(rebind[Scalar[DTYPE]](mesh_meta[mi_id, 0]))
                        mnv1 = Int(rebind[Scalar[DTYPE]](mesh_meta[mi_id, 1]))
                    var va2 = 0
                    var mnv2 = 0
                    if mj_id >= 0:
                        va2 = Int(rebind[Scalar[DTYPE]](mesh_meta[mj_id, 0]))
                        mnv2 = Int(rebind[Scalar[DTYPE]](mesh_meta[mj_id, 1]))
                    var result = gjk_epa[DTYPE, NMESH_VERTS](
                        gi_type,
                        pi_x, pi_y, pi_z, qi_x, qi_y, qi_z, qi_w,
                        ri, hli, hxi, hyi, hzi,
                        mesh_verts, va1, mnv1,
                        gj_type,
                        pj_x, pj_y, pj_z, qj_x, qj_y, qj_z, qj_w,
                        rj, hlj, hxj, hyj, hzj,
                        va2, mnv2,
                    )
                    dist = result[0]
                    cx = result[1]
                    cy = result[2]
                    cz = result[3]
                    nx = result[4]
                    ny = result[5]
                    nz = result[6]
                    body_a = gi_body
                    body_b = gj_body
                else:
                    _fill_pair_solparams[DTYPE, MAX_CONTACTS, BATCH](
                        env, _n0, num_contacts, _mx, contacts
                    )
                    continue

            if dist < contact_margin and num_contacts < MAX_CONTACTS:
                var c_off = num_contacts * CONTACT_SIZE
                contacts[env, c_off + CONTACT_IDX_BODY_A] = Scalar[DTYPE](
                    body_a
                )
                contacts[env, c_off + CONTACT_IDX_BODY_B] = Scalar[DTYPE](
                    body_b
                )
                contacts[env, c_off + CONTACT_IDX_POS_X] = cx
                contacts[env, c_off + CONTACT_IDX_POS_Y] = cy
                contacts[env, c_off + CONTACT_IDX_POS_Z] = cz
                # The record's normal points `body_b -> body_a`. Every branch
                # above computed `gi -> gj` with `body_a = gi`, so it is
                # negated here — UNCONDITIONALLY.
                #
                # ⚠ This used to be `if body_b > 0:`, which skipped the negation
                # whenever the second geom sat on the WORLD body and left those
                # contacts as `a -> b` while every other contact was `b -> a`.
                # Two conventions in one record, selected by a body id. Planes
                # are not affected either way — they have their own loop and
                # never reach this emit — so `body_b == 0` here means a
                # NON-PLANE world geom, which no shipped model currently has.
                # Latent, but it made body labels and normal direction
                # interdependent, and it nearly derailed the bug 35 fix.
                # Measured by `tests/physics3d/test_narrow_phase_pairs.mojo`'s
                # WORLD groups: a full 2.0 reversal on a unit vector.
                nx = -nx
                ny = -ny
                nz = -nz
                contacts[env, c_off + CONTACT_IDX_NX] = nx
                contacts[env, c_off + CONTACT_IDX_NY] = ny
                contacts[env, c_off + CONTACT_IDX_NZ] = nz
                contacts[env, c_off + CONTACT_IDX_DIST] = dist
                contacts[
                    env, c_off + CONTACT_IDX_INCLUDEMARGIN
                ] = contact_margin
                contacts[env, c_off + CONTACT_IDX_FRICTION] = contact_friction
                contacts[
                    env, c_off + CONTACT_IDX_FRICTION_SPIN
                ] = contact_friction_spin
                contacts[
                    env, c_off + CONTACT_IDX_FRICTION_ROLL
                ] = contact_friction_roll
                contacts[env, c_off + CONTACT_IDX_CONDIM] = Scalar[DTYPE](
                    contact_condim
                )
                num_contacts += 1

            _fill_pair_solparams[DTYPE, MAX_CONTACTS, BATCH](
                env, _n0, num_contacts, _mx, contacts
            )

    smeta[env, META_IDX_NUM_CONTACTS] = Scalar[DTYPE](num_contacts)


def _detect_contacts_fields_kernel[
    DTYPE: DType,
    NQ: Int,
    NV: Int,
    NBODY: Int,
    NJOINT: Int,
    MAX_CONTACTS: Int,
    NGEOM: Int,
    NEXCLUDE: Int,
    NMESH_VERTS: Int,
    BATCH: Int,
](
    xpos: LayoutTensor[
        DTYPE, Layout.row_major(BATCH, NBODY * 3), MutAnyOrigin
    ],
    xquat: LayoutTensor[
        DTYPE, Layout.row_major(BATCH, NBODY * 4), MutAnyOrigin
    ],
    geoms: LayoutTensor[
        DTYPE, Layout.row_major(NGEOM, MODEL_GEOM_SIZE), MutAnyOrigin
    ],
    bodies: LayoutTensor[
        DTYPE, Layout.row_major(NBODY, MODEL_BODY_SIZE), MutAnyOrigin
    ],
    mmeta: LayoutTensor[
        DTYPE, Layout.row_major(MODEL_META_SIZE), MutAnyOrigin
    ],
    excludes: LayoutTensor[
        DTYPE, Layout.row_major(NEXCLUDE, 2), MutAnyOrigin
    ],
    mesh_meta: LayoutTensor[
        DTYPE,
        Layout.row_major(MAX_GPU_MESHES, MODEL_MESH_META_SIZE),
        MutAnyOrigin,
    ],
    mesh_verts: LayoutTensor[
        DTYPE, Layout.row_major(NMESH_VERTS, 3), MutAnyOrigin
    ],
    contacts: LayoutTensor[
        DTYPE, Layout.row_major(BATCH, MAX_CONTACTS * CONTACT_SIZE),
        MutAnyOrigin,
    ],
    smeta: LayoutTensor[
        DTYPE, Layout.row_major(BATCH, METADATA_SIZE), MutAnyOrigin
    ],
):
    var env = Int(block_dim.x * block_idx.x + thread_idx.x)
    if env >= BATCH:
        return
    _detect_contacts_env[
        DTYPE, NQ, NV, NBODY, NJOINT, MAX_CONTACTS, NGEOM, NEXCLUDE,
        NMESH_VERTS, BATCH,
    ](
        env, xpos, xquat, geoms, bodies, mmeta, excludes, mesh_meta,
        mesh_verts, contacts, smeta,
    )


def detect_contacts[
    target: StaticString,
    DTYPE: DType,
    NQ: Int,
    NV: Int,
    NBODY: Int,
    NJOINT: Int,
    MAX_CONTACTS: Int,
    NGEOM: Int = 0,
    NEQUALITY: Int = 0,
    NTENDON: Int = 0,
    NSITE: Int = 0,
    NEXCLUDE: Int = 0,
    NMESH_VERTS: Int = 0,
    BATCH: Int = 1,
](
    mut d: Data[DTYPE, NQ, NV, NBODY, MAX_CONTACTS, NSITE, BATCH],
    mut m: Model[
        DTYPE,
        NV,
        NBODY,
        NJOINT,
        NGEOM,
        NEQUALITY,
        NTENDON,
        NSITE,
        NEXCLUDE,
        NMESH_VERTS,
    ],
    ctx: Optional[DeviceContext] = None,
) raises:
    """Unified geom contact detection from FK products, both targets, one
    body. Reads `d.xpos`/`d.xquat` + geom/body/meta/exclude/mesh records;
    writes `d.contacts` + the ncon slot of `d.meta`."""
    comptime L_B3 = Layout.row_major(BATCH, NBODY * 3)
    comptime L_B4 = Layout.row_major(BATCH, NBODY * 4)
    comptime L_GEOM = Layout.row_major(NGEOM, MODEL_GEOM_SIZE)
    comptime L_BODY = Layout.row_major(NBODY, MODEL_BODY_SIZE)
    comptime L_MMETA = Layout.row_major(MODEL_META_SIZE)
    comptime L_EXCLUDE = Layout.row_major(NEXCLUDE, 2)
    comptime L_MESH_META = Layout.row_major(
        MAX_GPU_MESHES, MODEL_MESH_META_SIZE
    )
    comptime L_MESH_VERT = Layout.row_major(NMESH_VERTS, 3)
    comptime L_CONTACTS = Layout.row_major(BATCH, MAX_CONTACTS * CONTACT_SIZE)
    comptime L_SMETA = Layout.row_major(BATCH, METADATA_SIZE)

    comptime if target == "cpu":
        var xpos_v = d.xpos.lt["cpu", L_B3]()
        var xquat_v = d.xquat.lt["cpu", L_B4]()
        var geoms_v = m.geoms.lt["cpu", L_GEOM]()
        var bodies_v = m.bodies.lt["cpu", L_BODY]()
        var mmeta_v = m.meta.lt["cpu", L_MMETA]()
        var excludes_v = m.excludes.lt["cpu", L_EXCLUDE]()
        var mesh_meta_v = m.mesh_meta.lt["cpu", L_MESH_META]()
        var mesh_verts_v = m.mesh_verts.lt["cpu", L_MESH_VERT]()
        var contacts_v = d.contacts.lt["cpu", L_CONTACTS]()
        var smeta_v = d.meta.lt["cpu", L_SMETA]()
        for e in range(BATCH):
            _detect_contacts_env[
                DTYPE, NQ, NV, NBODY, NJOINT, MAX_CONTACTS, NGEOM,
                NEXCLUDE, NMESH_VERTS, BATCH,
            ](
                e, xpos_v, xquat_v, geoms_v, bodies_v, mmeta_v,
                excludes_v, mesh_meta_v, mesh_verts_v, contacts_v, smeta_v,
            )
    else:
        var c = ctx.value()
        comptime BLOCKS = (BATCH + CD_TPB - 1) // CD_TPB
        c.enqueue_function[
            _detect_contacts_fields_kernel[
                DTYPE, NQ, NV, NBODY, NJOINT, MAX_CONTACTS, NGEOM,
                NEXCLUDE, NMESH_VERTS, BATCH,
            ]
        ](
            d.xpos.lt["gpu", L_B3](),
            d.xquat.lt["gpu", L_B4](),
            m.geoms.lt["gpu", L_GEOM](),
            m.bodies.lt["gpu", L_BODY](),
            m.meta.lt["gpu", L_MMETA](),
            m.excludes.lt["gpu", L_EXCLUDE](),
            m.mesh_meta.lt["gpu", L_MESH_META](),
            m.mesh_verts.lt["gpu", L_MESH_VERT](),
            d.contacts.lt["gpu", L_CONTACTS](),
            d.meta.lt["gpu", L_SMETA](),
            grid_dim=(BLOCKS,),
            block_dim=(CD_TPB,),
        )
