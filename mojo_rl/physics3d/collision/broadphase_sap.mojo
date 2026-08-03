"""SAP broadphase contact detection over per-field tensors (migration P4).

Per-field port of `detect_contacts_sap_gpu` (collision/broadphase_sap.mojo)
— arithmetic, iteration order, insertion-sort order and branch structure
verbatim. Reads FK products (`d.xpos`, `d.xquat`) + geom/body records +
model meta + exclude pairs + mesh hulls; writes packed contact records into
`d.contacts` and the contact count into `d.meta` (META_IDX_NUM_CONTACTS).

Operands (10): xpos, xquat (data) + geoms, bodies, mmeta, excludes,
mesh_meta, mesh_verts (model) + contacts, smeta (data outputs). Mesh
collision (plane-mesh vertex scan + GJK/EPA fallback via gjk) is
compiled in only when NMESH_VERTS > 0; zero-mesh models keep the legacy
branch structure (mesh branches degrade to no-emission / `continue`).

NOTE: the legacy SAP kernel's contact conventions differ from
detect_contacts_gpu (plane contacts write BODY_B=-1 instead of 0, no
INCLUDEMARGIN slot, plane-mesh DIST is `dist_v - cm`); this port preserves
the SAP conventions verbatim — bit-exactness is gated against legacy SAP.

`detect_contacts_auto` mirrors `detect_contacts_auto_gpu`:
NGEOM >= SAP_THRESHOLD dispatches to SAP, else to `detect_contacts`.
The fields integrators are NOT rewired to auto here (SAP emission ORDER
differs from O(N^2), which would shift existing bit-exact gates)."""

from std.gpu import thread_idx, block_idx, block_dim
from std.gpu.host import DeviceContext
from layout import Layout, LayoutTensor

from ..kinematics.quat_math import gpu_quat_rotate
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
    CONTACT_IDX_FRICTION,
    CONTACT_IDX_FRICTION_SPIN,
    CONTACT_IDX_FRICTION_ROLL,
    CONTACT_IDX_CONDIM,
    CONTACT_IDX_FRAME_T1_X,
    CONTACT_IDX_FRAME_T1_Y,
    CONTACT_IDX_FRAME_T1_Z,
    GEOM_IDX_TYPE,
    GEOM_IDX_BODY,
    GEOM_IDX_RADIUS,
    GEOM_IDX_HALF_LENGTH,
    GEOM_IDX_HALF_X,
    GEOM_IDX_HALF_Y,
    GEOM_IDX_HALF_Z,
    GEOM_IDX_FRICTION,
    GEOM_IDX_CONTYPE,
    GEOM_IDX_CONAFFINITY,
    GEOM_IDX_CONDIM,
    GEOM_IDX_FRICTION_SPIN,
    GEOM_IDX_FRICTION_ROLL,
    GEOM_IDX_MARGIN,
    GEOM_IDX_MESH_ID,
    MAX_GPU_MESHES,
    MODEL_MESH_META_SIZE,
)
from .collision_primitives import (
    sphere_sphere,
    capsule_sphere,
    capsule_capsule,
    box_sphere,
    box_box,
    box_plane,
    cylinder_plane,
    cylinder_sphere,
    cylinder_capsule,
    cylinder_cylinder,
    cylinder_box,
    ellipsoid_plane,
)
from .plane_frame import (
    plane_world_normal,
    to_plane_frame,
    from_plane_frame,
    quat_to_plane_frame,
)
from .gjk import gjk_epa
from .contact_detection import (
    _plane_box_contacts,
    _plane_cylinder_contacts,
    _box_box_contacts,
    _capsule_box_contacts,
    _geom_world_pos,
    detect_contacts,
)

# SAP broadphase activation threshold + AABB helper (relocated here at the P6
# legacy sunset; formerly imported from the deleted legacy `broadphase_sap`).
comptime SAP_THRESHOLD: Int = 16

comptime SAP_TPB: Int = 64


def _aabb_half_extents[
    DTYPE: DType
](
    geom_type: Int,
    qx: Scalar[DTYPE],
    qy: Scalar[DTYPE],
    qz: Scalar[DTYPE],
    qw: Scalar[DTYPE],
    radius: Scalar[DTYPE],
    half_length: Scalar[DTYPE],
    half_x: Scalar[DTYPE],
    half_y: Scalar[DTYPE],
    half_z: Scalar[DTYPE],
) -> Tuple[Scalar[DTYPE], Scalar[DTYPE], Scalar[DTYPE]]:
    """Return (ex, ey, ez) — the AABB half-extents for one geom in world space.

    The world-space AABB is [center - e, center + e] on each axis.
    Planes are not handled here (they use infinite bounds, handled separately).
    """
    if geom_type == GEOM_SPHERE:
        return (radius, radius, radius)

    if geom_type == GEOM_CAPSULE or geom_type == GEOM_CYLINDER:
        # World-space capsule/cylinder axis = rotate local Z (0,0,1) by quat.
        # Derivation: v' = (2(qx*qz+qy*qw), 2(qy*qz-qx*qw), 1-2(qx²+qy²))
        var two = Scalar[DTYPE](2)
        var ax = two * (qx * qz + qy * qw)
        var ay = two * (qy * qz - qx * qw)
        var az = Scalar[DTYPE](1) - two * (qx * qx + qy * qy)
        return (
            abs(ax) * half_length + radius,
            abs(ay) * half_length + radius,
            abs(az) * half_length + radius,
        )

    if geom_type == GEOM_BOX:
        # Tight AABB via rotation matrix: half_extent[k] = Σ |R[k][j]| * half[j]
        var two = Scalar[DTYPE](2)
        var r00 = Scalar[DTYPE](1) - two * (qy * qy + qz * qz)
        var r01 = two * (qx * qy - qz * qw)
        var r02 = two * (qx * qz + qy * qw)
        var r10 = two * (qx * qy + qz * qw)
        var r11 = Scalar[DTYPE](1) - two * (qx * qx + qz * qz)
        var r12 = two * (qy * qz - qx * qw)
        var r20 = two * (qx * qz - qy * qw)
        var r21 = two * (qy * qz + qx * qw)
        var r22 = Scalar[DTYPE](1) - two * (qx * qx + qy * qy)
        var ex = abs(r00) * half_x + abs(r01) * half_y + abs(r02) * half_z
        var ey = abs(r10) * half_x + abs(r11) * half_y + abs(r12) * half_z
        var ez = abs(r20) * half_x + abs(r21) * half_y + abs(r22) * half_z
        return (ex, ey, ez)

    # Fallback (unknown geom type): use radius as conservative bound
    return (radius, radius, radius)


@always_inline
def _detect_contacts_sap_env[
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
    """AABB/SAP broadphase contact detection for one env (verbatim from
    detect_contacts_sap_gpu; mesh branches compiled in iff NMESH_VERTS > 0).
    """
    var num_contacts = 0

    # ------------------------------------------------------------------
    # 1. Precompute world positions for all NGEOM geoms.
    # ------------------------------------------------------------------
    var wpx = InlineArray[Scalar[DTYPE], NGEOM](uninitialized=True)
    var wpy = InlineArray[Scalar[DTYPE], NGEOM](uninitialized=True)
    var wpz = InlineArray[Scalar[DTYPE], NGEOM](uninitialized=True)
    var wqx = InlineArray[Scalar[DTYPE], NGEOM](uninitialized=True)
    var wqy = InlineArray[Scalar[DTYPE], NGEOM](uninitialized=True)
    var wqz = InlineArray[Scalar[DTYPE], NGEOM](uninitialized=True)
    var wqw = InlineArray[Scalar[DTYPE], NGEOM](uninitialized=True)

    for g in range(NGEOM):
        var px: Scalar[DTYPE] = 0
        var py: Scalar[DTYPE] = 0
        var pz: Scalar[DTYPE] = 0
        var qx: Scalar[DTYPE] = 0
        var qy: Scalar[DTYPE] = 0
        var qz: Scalar[DTYPE] = 0
        var qw: Scalar[DTYPE] = 1
        _geom_world_pos[DTYPE, NBODY, NGEOM, BATCH](
            env, g, geoms, xpos, xquat, px, py, pz, qx, qy, qz, qw
        )
        wpx[g] = px
        wpy[g] = py
        wpz[g] = pz
        wqx[g] = qx
        wqy[g] = qy
        wqz[g] = qz
        wqw[g] = qw

    # ------------------------------------------------------------------
    # 2. Compute AABBs for non-plane geoms.
    # ------------------------------------------------------------------
    var aabb_min_x = InlineArray[Scalar[DTYPE], NGEOM](uninitialized=True)
    var aabb_max_x = InlineArray[Scalar[DTYPE], NGEOM](uninitialized=True)
    var aabb_min_y = InlineArray[Scalar[DTYPE], NGEOM](uninitialized=True)
    var aabb_max_y = InlineArray[Scalar[DTYPE], NGEOM](uninitialized=True)
    var aabb_min_z = InlineArray[Scalar[DTYPE], NGEOM](uninitialized=True)
    var aabb_max_z = InlineArray[Scalar[DTYPE], NGEOM](uninitialized=True)

    for g in range(NGEOM):
        var gt = Int(rebind[Scalar[DTYPE]](geoms[g, GEOM_IDX_TYPE]))
        if gt == GEOM_PLANE:
            continue
        var r = rebind[Scalar[DTYPE]](geoms[g, GEOM_IDX_RADIUS])
        var hl = rebind[Scalar[DTYPE]](geoms[g, GEOM_IDX_HALF_LENGTH])
        var hx = rebind[Scalar[DTYPE]](geoms[g, GEOM_IDX_HALF_X])
        var hy = rebind[Scalar[DTYPE]](geoms[g, GEOM_IDX_HALF_Y])
        var hz = rebind[Scalar[DTYPE]](geoms[g, GEOM_IDX_HALF_Z])
        var he = _aabb_half_extents[DTYPE](
            gt, wqx[g], wqy[g], wqz[g], wqw[g], r, hl, hx, hy, hz
        )
        aabb_min_x[g] = wpx[g] - he[0]
        aabb_max_x[g] = wpx[g] + he[0]
        aabb_min_y[g] = wpy[g] - he[1]
        aabb_max_y[g] = wpy[g] + he[1]
        aabb_min_z[g] = wpz[g] - he[2]
        aabb_max_z[g] = wpz[g] + he[2]

    # ------------------------------------------------------------------
    # 3. Plane vs non-plane pairs.
    # ------------------------------------------------------------------
    for gi in range(NGEOM):
        var gi_type = Int(
            rebind[Scalar[DTYPE]](geoms[gi, GEOM_IDX_TYPE])
        )
        if gi_type != GEOM_PLANE:
            continue
        var gi_body = Int(
            rebind[Scalar[DTYPE]](geoms[gi, GEOM_IDX_BODY])
        )
        var gi_contype = Int(
            rebind[Scalar[DTYPE]](geoms[gi, GEOM_IDX_CONTYPE])
        )
        var gi_conaffinity = Int(
            rebind[Scalar[DTYPE]](geoms[gi, GEOM_IDX_CONAFFINITY])
        )
        # The plane's full pose. This loop used to keep only `wpz[gi]` as a
        # `ground_z` and hardcode the normal to (0,0,1), i.e. it modelled every
        # plane as a horizontal floor at the height of its origin. See
        # `collision/plane_frame.mojo`. Everything below now works in the
        # PLANE'S FRAME — where the plane really is z=0 with normal +z, which
        # is what all the `*_plane` primitives assume — and maps the contact
        # point and normal back to world at the write.
        var plp_x = wpx[gi]
        var plp_y = wpy[gi]
        var plp_z = wpz[gi]
        var plq_x = wqx[gi]
        var plq_y = wqy[gi]
        var plq_z = wqz[gi]
        var plq_w = wqw[gi]
        var pn = plane_world_normal[DTYPE](plq_x, plq_y, plq_z, plq_w)

        for gj in range(NGEOM):
            if num_contacts >= MAX_CONTACTS:
                smeta[env, META_IDX_NUM_CONTACTS] = Scalar[DTYPE](
                    num_contacts
                )
                return
            var gj_type = Int(
                rebind[Scalar[DTYPE]](geoms[gj, GEOM_IDX_TYPE])
            )
            if gj_type == GEOM_PLANE:
                continue
            var gj_body = Int(
                rebind[Scalar[DTYPE]](geoms[gj, GEOM_IDX_BODY])
            )
            if gj_body == 0:
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

            var fi = rebind[Scalar[DTYPE]](geoms[gi, GEOM_IDX_FRICTION])
            var fj = rebind[Scalar[DTYPE]](geoms[gj, GEOM_IDX_FRICTION])
            var cf = fi if fj <= fi else fj
            var fsi = rebind[Scalar[DTYPE]](
                geoms[gi, GEOM_IDX_FRICTION_SPIN]
            )
            var fsj = rebind[Scalar[DTYPE]](
                geoms[gj, GEOM_IDX_FRICTION_SPIN]
            )
            var cfs = fsi if fsj <= fsi else fsj
            var fri = rebind[Scalar[DTYPE]](
                geoms[gi, GEOM_IDX_FRICTION_ROLL]
            )
            var frj = rebind[Scalar[DTYPE]](
                geoms[gj, GEOM_IDX_FRICTION_ROLL]
            )
            var cfr = fri if frj <= fri else frj
            var ci = Int(
                rebind[Scalar[DTYPE]](geoms[gi, GEOM_IDX_CONDIM])
            )
            var cj = Int(
                rebind[Scalar[DTYPE]](geoms[gj, GEOM_IDX_CONDIM])
            )
            var cdim = ci if cj <= ci else cj
            var mgi = rebind[Scalar[DTYPE]](geoms[gi, GEOM_IDX_MARGIN])
            var mgj = rebind[Scalar[DTYPE]](geoms[gj, GEOM_IDX_MARGIN])
            var cm = mgi + mgj  # MuJoCo 3.5+: sum of margins

            # Pose IN THE PLANE'S FRAME, so `ground_z` below is 0 and the
            # branch arithmetic is the same as it always was.
            var lpj = to_plane_frame[DTYPE](
                plp_x, plp_y, plp_z, plq_x, plq_y, plq_z, plq_w,
                wpx[gj], wpy[gj], wpz[gj],
            )
            var lqj = quat_to_plane_frame[DTYPE](
                plq_x, plq_y, plq_z, plq_w,
                wqx[gj], wqy[gj], wqz[gj], wqw[gj],
            )
            var pj_x = lpj[0]
            var pj_y = lpj[1]
            var pj_z = lpj[2]
            var qj_x = lqj[0]
            var qj_y = lqj[1]
            var qj_z = lqj[2]
            var qj_w = lqj[3]
            var ground_z = Scalar[DTYPE](0)
            var rj = rebind[Scalar[DTYPE]](geoms[gj, GEOM_IDX_RADIUS])
            var hlj = rebind[Scalar[DTYPE]](
                geoms[gj, GEOM_IDX_HALF_LENGTH]
            )

            if gj_type == GEOM_SPHERE:
                var dist = pj_z - rj - ground_z
                if dist < cm and num_contacts < MAX_CONTACTS:
                    var c_off = num_contacts * CONTACT_SIZE
                    contacts[env, c_off + CONTACT_IDX_BODY_A] = Scalar[DTYPE](
                        gj_body
                    )
                    contacts[env, c_off + CONTACT_IDX_BODY_B] = Scalar[DTYPE](
                        -1
                    )
                    var cw = from_plane_frame[DTYPE](
                        plp_x, plp_y, plp_z, plq_x, plq_y, plq_z, plq_w,
                        pj_x, pj_y,
                        ground_z + dist * Scalar[DTYPE](0.5),
                    )
                    contacts[env, c_off + CONTACT_IDX_POS_X] = cw[0]
                    contacts[env, c_off + CONTACT_IDX_POS_Y] = cw[1]
                    contacts[env, c_off + CONTACT_IDX_POS_Z] = cw[2]
                    contacts[env, c_off + CONTACT_IDX_NX] = pn[0]
                    contacts[env, c_off + CONTACT_IDX_NY] = pn[1]
                    contacts[env, c_off + CONTACT_IDX_NZ] = pn[2]
                    contacts[env, c_off + CONTACT_IDX_DIST] = dist
                    contacts[env, c_off + CONTACT_IDX_FRICTION] = cf
                    contacts[env, c_off + CONTACT_IDX_FRICTION_SPIN] = cfs
                    contacts[env, c_off + CONTACT_IDX_FRICTION_ROLL] = cfr
                    contacts[env, c_off + CONTACT_IDX_CONDIM] = Scalar[DTYPE](
                        cdim
                    )
                    num_contacts += 1

            elif gj_type == GEOM_CAPSULE:
                var axis_w = gpu_quat_rotate(
                    qj_x,
                    qj_y,
                    qj_z,
                    qj_w,
                    Scalar[DTYPE](0),
                    Scalar[DTYPE](0),
                    Scalar[DTYPE](1),
                )
                # `axis_w` is in the PLANE'S frame (qj_* were rebased above),
                # which is what the endpoint arithmetic below needs. The
                # FRAME_T1 hint written into the record is read in WORLD space,
                # so it goes back — see collision/contact_frame.mojo for what
                # that slot is and is not.
                var axis_wd = gpu_quat_rotate(
                    plq_x, plq_y, plq_z, plq_w,
                    axis_w[0], axis_w[1], axis_w[2],
                )
                var e1_x = pj_x + hlj * axis_w[0]
                var e1_y = pj_y + hlj * axis_w[1]
                var e1_z = pj_z + hlj * axis_w[2]
                var dist1 = e1_z - rj - ground_z
                if dist1 < cm and num_contacts < MAX_CONTACTS:
                    var c_off = num_contacts * CONTACT_SIZE
                    contacts[env, c_off + CONTACT_IDX_BODY_A] = Scalar[DTYPE](
                        gj_body
                    )
                    contacts[env, c_off + CONTACT_IDX_BODY_B] = Scalar[DTYPE](
                        -1
                    )
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
                    contacts[env, c_off + CONTACT_IDX_FRICTION] = cf
                    contacts[env, c_off + CONTACT_IDX_FRICTION_SPIN] = cfs
                    contacts[env, c_off + CONTACT_IDX_FRICTION_ROLL] = cfr
                    contacts[env, c_off + CONTACT_IDX_CONDIM] = Scalar[DTYPE](
                        cdim
                    )
                    contacts[env, c_off + CONTACT_IDX_FRAME_T1_X] = axis_wd[0]
                    contacts[env, c_off + CONTACT_IDX_FRAME_T1_Y] = axis_wd[1]
                    contacts[env, c_off + CONTACT_IDX_FRAME_T1_Z] = axis_wd[2]
                    num_contacts += 1
                var e2_x = pj_x - hlj * axis_w[0]
                var e2_y = pj_y - hlj * axis_w[1]
                var e2_z = pj_z - hlj * axis_w[2]
                var dist2 = e2_z - rj - ground_z
                if dist2 < cm and num_contacts < MAX_CONTACTS:
                    var c_off = num_contacts * CONTACT_SIZE
                    contacts[env, c_off + CONTACT_IDX_BODY_A] = Scalar[DTYPE](
                        gj_body
                    )
                    contacts[env, c_off + CONTACT_IDX_BODY_B] = Scalar[DTYPE](
                        -1
                    )
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
                    contacts[env, c_off + CONTACT_IDX_FRICTION] = cf
                    contacts[env, c_off + CONTACT_IDX_FRICTION_SPIN] = cfs
                    contacts[env, c_off + CONTACT_IDX_FRICTION_ROLL] = cfr
                    contacts[env, c_off + CONTACT_IDX_CONDIM] = Scalar[DTYPE](
                        cdim
                    )
                    contacts[env, c_off + CONTACT_IDX_FRAME_T1_X] = axis_wd[0]
                    contacts[env, c_off + CONTACT_IDX_FRAME_T1_Y] = axis_wd[1]
                    contacts[env, c_off + CONTACT_IDX_FRAME_T1_Z] = axis_wd[2]
                    num_contacts += 1

            elif gj_type == GEOM_CYLINDER:
                # Up to FOUR points — two rim, two triangle — not one.
                # See `_plane_cylinder_contacts` in contact_detection.mojo;
                # shared with the naive path so the two cannot drift, which
                # is exactly how the ellipsoid branch below went missing.
                _plane_cylinder_contacts[DTYPE, MAX_CONTACTS, BATCH](
                    env,
                    gj_body,
                    pj_x, pj_y, pj_z,
                    qj_x, qj_y, qj_z, qj_w,
                    rj,
                    hlj,
                    ground_z,
                    plp_x, plp_y, plp_z,
                    plq_x, plq_y, plq_z, plq_w,
                    cm,
                    cf,
                    cfs,
                    cfr,
                    cdim,
                    -1,
                    contacts,
                    num_contacts,
                )

            elif gj_type == GEOM_ELLIPSOID:
                # ⚠ ADDED 2026-08-03. This branch did not exist, and
                # `broadphase_sap.mojo` contained no mention of ELLIPSOID at
                # all, so every ellipsoid geom was INVISIBLE TO COLLISION in
                # any model that takes the SAP path — `detect_contacts_auto`
                # switches to SAP at NGEOM >= 16, and nothing warns.
                #
                # Shipped and silently wrong at the time of the fix:
                #   quadruped     26 geoms, SAP, ellipsoid = `torso`
                #   humanoid_CMU  50 geoms, SAP, ellipsoids = `lhand`, `rhand`
                # i.e. the quadruped's TORSO never collided with the floor.
                # fish (12 geoms, 7 ellipsoids) and swimmer (7, 1) sit under
                # the threshold and take the naive path, which is why the
                # ellipsoid narrow phase looked exercised.
                #
                # It hid because no test compared a plane's contact SET
                # against MuJoCo — every plane in the suite is an axis-aligned
                # floor and every gate read qacc at poses where the ellipsoid
                # was not touching it. `test_oriented_plane_vs_mujoco` is what
                # found it, and it found it on the AXIS-ALIGNED control rather
                # than the tilted case it was written for.
                #
                # Body id is -1 here where `detect_contacts` writes 0 — the
                # documented split between the two emit paths.
                var hxje = rebind[Scalar[DTYPE]](geoms[gj, GEOM_IDX_HALF_X])
                var hyje = rebind[Scalar[DTYPE]](geoms[gj, GEOM_IDX_HALF_Y])
                var hzje = rebind[Scalar[DTYPE]](geoms[gj, GEOM_IDX_HALF_Z])
                # MuJoCo routes plane x ellipsoid through `mjc_PlaneConvex`,
                # which reports the single deepest support point; a smooth
                # strictly-convex surface meets a plane at one point, so unlike
                # the box there is no second contact to look for.
                var epe = ellipsoid_plane[DTYPE](
                    pj_x, pj_y, pj_z,
                    qj_x, qj_y, qj_z, qj_w,
                    hxje, hyje, hzje,
                    ground_z,
                )
                var diste = epe[0]
                if diste < cm and num_contacts < MAX_CONTACTS:
                    var c_off = num_contacts * CONTACT_SIZE
                    contacts[env, c_off + CONTACT_IDX_BODY_A] = Scalar[DTYPE](
                        gj_body
                    )
                    contacts[env, c_off + CONTACT_IDX_BODY_B] = Scalar[DTYPE](
                        -1
                    )
                    # `ellipsoid_plane` already returns the contact point in
                    # the PLANE frame, including the half-depth offset, so
                    # unlike the sphere branch there is nothing to add here.
                    var cwe = from_plane_frame[DTYPE](
                        plp_x, plp_y, plp_z, plq_x, plq_y, plq_z, plq_w,
                        epe[1], epe[2], epe[3],
                    )
                    contacts[env, c_off + CONTACT_IDX_POS_X] = cwe[0]
                    contacts[env, c_off + CONTACT_IDX_POS_Y] = cwe[1]
                    contacts[env, c_off + CONTACT_IDX_POS_Z] = cwe[2]
                    contacts[env, c_off + CONTACT_IDX_NX] = pn[0]
                    contacts[env, c_off + CONTACT_IDX_NY] = pn[1]
                    contacts[env, c_off + CONTACT_IDX_NZ] = pn[2]
                    contacts[env, c_off + CONTACT_IDX_DIST] = diste
                    contacts[env, c_off + CONTACT_IDX_FRICTION] = cf
                    contacts[env, c_off + CONTACT_IDX_FRICTION_SPIN] = cfs
                    contacts[env, c_off + CONTACT_IDX_FRICTION_ROLL] = cfr
                    contacts[env, c_off + CONTACT_IDX_CONDIM] = Scalar[DTYPE](
                        cdim
                    )
                    num_contacts += 1

            elif gj_type == GEOM_BOX:
                var hxj = rebind[Scalar[DTYPE]](geoms[gj, GEOM_IDX_HALF_X])
                var hyj = rebind[Scalar[DTYPE]](geoms[gj, GEOM_IDX_HALF_Y])
                var hzj = rebind[Scalar[DTYPE]](geoms[gj, GEOM_IDX_HALF_Z])
                # Up to FOUR corners, not one — see `_plane_box_contacts` and
                # task #42. ⚠ This path writes -1 for the world body where
                # `detect_contacts` writes 0, hence the explicit argument.
                _plane_box_contacts[DTYPE, MAX_CONTACTS, BATCH](
                    env,
                    gj_body,
                    pj_x, pj_y, pj_z,
                    qj_x, qj_y, qj_z, qj_w,
                    hxj, hyj, hzj,
                    ground_z,
                    plp_x, plp_y, plp_z,
                    plq_x, plq_y, plq_z, plq_w,
                    cm,
                    cf,
                    cfs,
                    cfr,
                    cdim,
                    -1,
                    contacts,
                    num_contacts,
                )

            elif gj_type == GEOM_MESH:
                # Plane-mesh: scan hull vertices below plane
                comptime if NMESH_VERTS > 0:
                    var mj_id = Int(
                        rebind[Scalar[DTYPE]](geoms[gj, GEOM_IDX_MESH_ID])
                    )
                    if mj_id >= 0:
                        var pm_vadr = Int(
                            rebind[Scalar[DTYPE]](mesh_meta[mj_id, 0])
                        )
                        var pm_vnum = Int(
                            rebind[Scalar[DTYPE]](mesh_meta[mj_id, 1])
                        )
                        for vi in range(pm_vnum):
                            if num_contacts >= MAX_CONTACTS:
                                break
                            var vx = rebind[Scalar[DTYPE]](
                                mesh_verts[pm_vadr + vi, 0]
                            )
                            var vy = rebind[Scalar[DTYPE]](
                                mesh_verts[pm_vadr + vi, 1]
                            )
                            var vz = rebind[Scalar[DTYPE]](
                                mesh_verts[pm_vadr + vi, 2]
                            )
                            var local_pt = gpu_quat_rotate(
                                qj_x, qj_y, qj_z, qj_w, vx, vy, vz
                            )
                            var wx = pj_x + local_pt[0]
                            var wy = pj_y + local_pt[1]
                            var wz = pj_z + local_pt[2]
                            var dist_v = wz - ground_z
                            if dist_v < cm:
                                var c_off = num_contacts * CONTACT_SIZE
                                contacts[
                                    env, c_off + CONTACT_IDX_BODY_A
                                ] = Scalar[DTYPE](gj_body)
                                contacts[
                                    env, c_off + CONTACT_IDX_BODY_B
                                ] = Scalar[DTYPE](-1)
                                var cw = from_plane_frame[DTYPE](
                                    plp_x, plp_y, plp_z,
                                    plq_x, plq_y, plq_z, plq_w,
                                    wx, wy,
                                    ground_z + dist_v * Scalar[DTYPE](0.5),
                                )
                                contacts[
                                    env, c_off + CONTACT_IDX_POS_X
                                ] = cw[0]
                                contacts[
                                    env, c_off + CONTACT_IDX_POS_Y
                                ] = cw[1]
                                contacts[
                                    env, c_off + CONTACT_IDX_POS_Z
                                ] = cw[2]
                                contacts[env, c_off + CONTACT_IDX_NX] = pn[0]
                                contacts[env, c_off + CONTACT_IDX_NY] = pn[1]
                                contacts[env, c_off + CONTACT_IDX_NZ] = pn[2]
                                contacts[
                                    env, c_off + CONTACT_IDX_DIST
                                ] = dist_v - cm
                                contacts[
                                    env, c_off + CONTACT_IDX_FRICTION
                                ] = cf
                                contacts[
                                    env, c_off + CONTACT_IDX_FRICTION_SPIN
                                ] = cfs
                                contacts[
                                    env, c_off + CONTACT_IDX_FRICTION_ROLL
                                ] = cfr
                                contacts[
                                    env, c_off + CONTACT_IDX_CONDIM
                                ] = Scalar[DTYPE](cdim)
                                num_contacts += 1

    # ------------------------------------------------------------------
    # 4. SAP sweep for non-plane pairs.
    # ------------------------------------------------------------------

    # 4a. Build SAP index list.
    var sap_idx = InlineArray[Int, NGEOM](uninitialized=True)
    var sap_n = 0
    for g in range(NGEOM):
        var gt = Int(rebind[Scalar[DTYPE]](geoms[g, GEOM_IDX_TYPE]))
        if gt != GEOM_PLANE:
            sap_idx[sap_n] = g
            sap_n += 1

    # 4b. Insertion sort by aabb_min_x.
    for i in range(1, sap_n):
        var key = sap_idx[i]
        var key_val = aabb_min_x[key]
        var j = i - 1
        while j >= 0 and aabb_min_x[sap_idx[j]] > key_val:
            sap_idx[j + 1] = sap_idx[j]
            j -= 1
        sap_idx[j + 1] = key

    # 4c. Sweep.
    for i in range(sap_n):
        var gi = sap_idx[i]
        var gi_max_x = aabb_max_x[gi]
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
        var pi_x = wpx[gi]
        var pi_y = wpy[gi]
        var pi_z = wpz[gi]
        var qi_x = wqx[gi]
        var qi_y = wqy[gi]
        var qi_z = wqz[gi]
        var qi_w = wqw[gi]
        var ri = rebind[Scalar[DTYPE]](geoms[gi, GEOM_IDX_RADIUS])
        var hli = rebind[Scalar[DTYPE]](geoms[gi, GEOM_IDX_HALF_LENGTH])
        var hxi = rebind[Scalar[DTYPE]](geoms[gi, GEOM_IDX_HALF_X])
        var hyi = rebind[Scalar[DTYPE]](geoms[gi, GEOM_IDX_HALF_Y])
        var hzi = rebind[Scalar[DTYPE]](geoms[gi, GEOM_IDX_HALF_Z])

        for j in range(i + 1, sap_n):
            if num_contacts >= MAX_CONTACTS:
                smeta[env, META_IDX_NUM_CONTACTS] = Scalar[DTYPE](
                    num_contacts
                )
                return
            var gj = sap_idx[j]

            if aabb_min_x[gj] > gi_max_x:
                break

            if (
                aabb_min_y[gj] > aabb_max_y[gi]
                or aabb_min_y[gi] > aabb_max_y[gj]
            ):
                continue
            if (
                aabb_min_z[gj] > aabb_max_z[gi]
                or aabb_min_z[gi] > aabb_max_z[gj]
            ):
                continue

            var gj_type = Int(
                rebind[Scalar[DTYPE]](geoms[gj, GEOM_IDX_TYPE])
            )
            var gj_body = Int(
                rebind[Scalar[DTYPE]](geoms[gj, GEOM_IDX_BODY])
            )
            # MuJoCo-style weld body filtering (GPU SAP)
            var weld_i = Int(
                rebind[Scalar[DTYPE]](bodies[gi_body, BODY_IDX_WELDID])
            )
            var weld_j = Int(
                rebind[Scalar[DTYPE]](bodies[gj_body, BODY_IDX_WELDID])
            )
            if weld_i == weld_j:
                continue
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
                var sap_n_ex = Int(
                    rebind[Scalar[DTYPE]](mmeta[MODEL_META_IDX_NEXCLUDE])
                )
                if sap_n_ex > 0:
                    var ba = gi_body if gi_body <= gj_body else gj_body
                    var bb = gj_body if gi_body <= gj_body else gi_body
                    var excluded = False
                    for ex in range(sap_n_ex):
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

            var fi = rebind[Scalar[DTYPE]](geoms[gi, GEOM_IDX_FRICTION])
            var fj = rebind[Scalar[DTYPE]](geoms[gj, GEOM_IDX_FRICTION])
            var cf = fi if fj <= fi else fj
            var fsi = rebind[Scalar[DTYPE]](
                geoms[gi, GEOM_IDX_FRICTION_SPIN]
            )
            var fsj = rebind[Scalar[DTYPE]](
                geoms[gj, GEOM_IDX_FRICTION_SPIN]
            )
            var cfs = fsi if fsj <= fsi else fsj
            var fri = rebind[Scalar[DTYPE]](
                geoms[gi, GEOM_IDX_FRICTION_ROLL]
            )
            var frj = rebind[Scalar[DTYPE]](
                geoms[gj, GEOM_IDX_FRICTION_ROLL]
            )
            var cfr = fri if frj <= fri else frj
            var ci = Int(
                rebind[Scalar[DTYPE]](geoms[gi, GEOM_IDX_CONDIM])
            )
            var cj_dim = Int(
                rebind[Scalar[DTYPE]](geoms[gj, GEOM_IDX_CONDIM])
            )
            var cdim = ci if cj_dim <= ci else cj_dim
            var mgi = rebind[Scalar[DTYPE]](geoms[gi, GEOM_IDX_MARGIN])
            var mgj = rebind[Scalar[DTYPE]](geoms[gj, GEOM_IDX_MARGIN])
            var cm = mgi + mgj  # MuJoCo 3.5+: sum of margins

            var pj_x = wpx[gj]
            var pj_y = wpy[gj]
            var pj_z = wpz[gj]
            var qj_x = wqx[gj]
            var qj_y = wqy[gj]
            var qj_z = wqz[gj]
            var qj_w = wqw[gj]
            var rj = rebind[Scalar[DTYPE]](geoms[gj, GEOM_IDX_RADIUS])
            var hlj = rebind[Scalar[DTYPE]](
                geoms[gj, GEOM_IDX_HALF_LENGTH]
            )
            var hxj = rebind[Scalar[DTYPE]](geoms[gj, GEOM_IDX_HALF_X])
            var hyj = rebind[Scalar[DTYPE]](geoms[gj, GEOM_IDX_HALF_Y])
            var hzj = rebind[Scalar[DTYPE]](geoms[gj, GEOM_IDX_HALF_Z])

            var dist: Scalar[DTYPE] = 1.0
            var cx: Scalar[DTYPE] = 0
            var cy: Scalar[DTYPE] = 0
            var cz: Scalar[DTYPE] = 0
            var nx: Scalar[DTYPE] = 0
            var ny: Scalar[DTYPE] = 0
            var nz: Scalar[DTYPE] = 1
            # CONTACT DIRECTION INVARIANT — every branch below emits
            # `normal = gi -> gj` with `body_a = gi_body, body_b = gj_body`.
            #
            # The REVERSED-ORDER branches call a primitive written for the
            # other operand order, so they negate the returned normal to get
            # back to gi->gj. They used to ALSO swap body_a/body_b, and that
            # double flip left them emitting `normal = body_b -> body_a` while
            # the ten canonical-order branches emitted `body_a -> body_b`.
            # Either operation alone is correct; both is not.
            #
            # Silent until dm_control manipulator, which is the first model
            # where one physical pair type reaches BOTH orderings — a sphere
            # (the ball) contacting capsules (the fingers), under the SAP
            # broadphase where (gi, gj) comes from the sweep rather than the
            # geom index. `aref` is built from the penetration DEPTH and so
            # does not flip with the normal, so a flipped normal desynchronises
            # `jar = aref + J*qacc`: one contact was self-consistent and the
            # other was not, giving contact forces 9% and 20% below MuJoCo's
            # while every row constant matched to 15 digits.
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
                    cm, cf, cfs, cfr, cdim,
                    contacts, num_contacts,
                )
                continue
            elif gi_type == GEOM_CAPSULE and gj_type == GEOM_BOX:
                _ = _capsule_box_contacts[DTYPE, MAX_CONTACTS, BATCH](
                    env, gi_body, gj_body,
                    pj_x, pj_y, pj_z, qj_x, qj_y, qj_z, qj_w, hxj, hyj, hzj,
                    pi_x, pi_y, pi_z, qi_x, qi_y, qi_z, qi_w, hli, ri,
                    Scalar[DTYPE](1),
                    cm, cf, cfs, cfr, cdim,
                    contacts, num_contacts,
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
                    cm,
                    cf,
                    cfs,
                    cfr,
                    cdim,
                    contacts,
                    num_contacts,
                )
                if code >= 0:
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
                    pj_x, pj_y, pj_z, qj_x, qj_y, qj_z, qj_w, hlj, rj)
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
                    pi_x, pi_y, pi_z, qi_x, qi_y, qi_z, qi_w, hli, ri)
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
                    pj_x, pj_y, pj_z, qj_x, qj_y, qj_z, qj_w, hlj, rj)
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
                    pj_x, pj_y, pj_z, qj_x, qj_y, qj_z, qj_w, hxj, hyj, hzj)
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
                    pi_x, pi_y, pi_z, qi_x, qi_y, qi_z, qi_w, hxi, hyi, hzi)
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
                    continue

            if dist < cm and num_contacts < MAX_CONTACTS:
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
                contacts[env, c_off + CONTACT_IDX_FRICTION] = cf
                contacts[env, c_off + CONTACT_IDX_FRICTION_SPIN] = cfs
                contacts[env, c_off + CONTACT_IDX_FRICTION_ROLL] = cfr
                contacts[env, c_off + CONTACT_IDX_CONDIM] = Scalar[DTYPE](
                    cdim
                )
                num_contacts += 1

    smeta[env, META_IDX_NUM_CONTACTS] = Scalar[DTYPE](num_contacts)


def _detect_contacts_sap_fields_kernel[
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
    _detect_contacts_sap_env[
        DTYPE, NQ, NV, NBODY, NJOINT, MAX_CONTACTS, NGEOM, NEXCLUDE,
        NMESH_VERTS, BATCH,
    ](
        env, xpos, xquat, geoms, bodies, mmeta, excludes, mesh_meta,
        mesh_verts, contacts, smeta,
    )


def detect_contacts_sap[
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
    """AABB/SAP broadphase geom contact detection from FK products, both
    targets, one body. Reads `d.xpos`/`d.xquat` + geom/body/meta/exclude/mesh
    records; writes `d.contacts` + the ncon slot of `d.meta`."""
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
            _detect_contacts_sap_env[
                DTYPE, NQ, NV, NBODY, NJOINT, MAX_CONTACTS, NGEOM,
                NEXCLUDE, NMESH_VERTS, BATCH,
            ](
                e, xpos_v, xquat_v, geoms_v, bodies_v, mmeta_v,
                excludes_v, mesh_meta_v, mesh_verts_v, contacts_v, smeta_v,
            )
    else:
        var c = ctx.value()
        comptime BLOCKS = (BATCH + SAP_TPB - 1) // SAP_TPB
        c.enqueue_function[
            _detect_contacts_sap_fields_kernel[
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
            block_dim=(SAP_TPB,),
        )


def detect_contacts_auto[
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
    """Contact detection with automatic broadphase selection (fields).

    Uses detect_contacts_sap when NGEOM >= SAP_THRESHOLD (default
    16), otherwise falls back to detect_contacts. The branch is
    resolved at compile time. NOTE: SAP contact emission ORDER differs from
    the O(N^2) path — do not swap this into a bit-exact-gated pipeline
    without re-baselining."""

    comptime if NGEOM >= SAP_THRESHOLD:
        detect_contacts_sap[
            target, DTYPE, NQ, NV, NBODY, NJOINT, MAX_CONTACTS, NGEOM,
            NEQUALITY, NTENDON, NSITE, NEXCLUDE, NMESH_VERTS, BATCH,
        ](d, m, ctx)
    else:
        detect_contacts[
            target, DTYPE, NQ, NV, NBODY, NJOINT, MAX_CONTACTS, NGEOM,
            NEQUALITY, NTENDON, NSITE, NEXCLUDE, NMESH_VERTS, BATCH,
        ](d, m, ctx)
