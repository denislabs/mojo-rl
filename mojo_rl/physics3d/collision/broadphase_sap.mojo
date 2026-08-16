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
from max.gpu.host import DeviceContext
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
from ..fields import Data, Model, Dims
from ..gpu.constants import (
    MODEL_BODY_SIZE,
    MODEL_GEOM_SIZE,
    MODEL_META_SIZE,
    METADATA_SIZE,
    MODEL_META_IDX_NEXCLUDE,
    MODEL_META_IDX_NPAIR,
    MODEL_META_IDX_CCD_TOLERANCE,
    MODEL_META_IDX_CCD_ITERATIONS,
    MJ_CCD_TOLERANCE,
    MJ_CCD_ITERATIONS,
    MODEL_PAIR_SIZE,
    PAIR_IDX_GEOM1,
    PAIR_IDX_GEOM2,
    PAIR_IDX_MARGIN,
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
    CONTACT_IDX_FRAME_T1_X,
    CONTACT_IDX_FRAME_T1_Y,
    CONTACT_IDX_FRAME_T1_Z,
    GEOM_IDX_TYPE,
    GEOM_IDX_BODY,
    GEOM_IDX_RADIUS,
    GEOM_IDX_RBOUND,
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
    GEOM_IDX_PRIORITY,
    GEOM_IDX_SOLREF_0,
    GEOM_IDX_SOLREF_1,
    GEOM_IDX_SOLIMP_0,
    GEOM_IDX_SOLIMP_1,
    GEOM_IDX_SOLIMP_2,
    GEOM_IDX_SOLIMP_3,
    GEOM_IDX_SOLIMP_4,
    CONTACT_IDX_SOLREF_0,
    CONTACT_IDX_SOLREF_1,
    CONTACT_IDX_SOLIMP_0,
    CONTACT_IDX_SOLIMP_1,
    CONTACT_IDX_SOLIMP_2,
    CONTACT_IDX_SOLIMP_3,
    CONTACT_IDX_SOLIMP_4,
    GEOM_IDX_MESH_ID,
    MAX_GPU_MESHES,
    MODEL_MESH_META_SIZE,
    MODEL_MESH_POLY_SIZE,
    MESH_META_IDX_POLYADR,
    MESH_META_IDX_POLYNUM,
    mesh_max_poly,
    mesh_max_polyvert,
    mesh_max_edge,
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
from .gjk import gjk_epa, gjk_epa_witness
from .multi_ccd import multi_ccd_pair_supported, multi_ccd_extra_contacts
from .native_multicontact import (
    native_multicontact_contacts,
    MC_ENABLED,
)
from .contact_detection import (
    _plane_mesh_contacts,
    mix_contact_params,
    pair_body_filtered,
    find_predefined_pair,
    pair_params,
    _fill_pair_solparams,
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
    # Appended rather than grouped with NEXCLUDE — see `fields.Model`.
    NPAIR: Int,
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
    pairs: LayoutTensor[
        DTYPE, Layout.row_major(NPAIR, MODEL_PAIR_SIZE), MutAnyOrigin
    ],
    mesh_meta: LayoutTensor[
        DTYPE,
        Layout.row_major(MAX_GPU_MESHES, MODEL_MESH_META_SIZE),
        MutAnyOrigin,
    ],
    mesh_verts: LayoutTensor[
        DTYPE, Layout.row_major(NMESH_VERTS, 3), MutAnyOrigin
    ],
    mesh_polys: LayoutTensor[
        DTYPE,
        Layout.row_major(mesh_max_poly(NMESH_VERTS), MODEL_MESH_POLY_SIZE),
        MutAnyOrigin,
    ],
    mesh_polyvert: LayoutTensor[
        DTYPE, Layout.row_major(mesh_max_polyvert(NMESH_VERTS)), MutAnyOrigin
    ],
    mesh_polymap: LayoutTensor[
        DTYPE, Layout.row_major(mesh_max_polyvert(NMESH_VERTS)), MutAnyOrigin
    ],
    mesh_vert_polymap: LayoutTensor[
        DTYPE, Layout.row_major(NMESH_VERTS, 2), MutAnyOrigin
    ],
    mesh_vert_edgeadr: LayoutTensor[
        DTYPE, Layout.row_major(NMESH_VERTS), MutAnyOrigin
    ],
    mesh_edges: LayoutTensor[
        DTYPE, Layout.row_major(mesh_max_edge(NMESH_VERTS)), MutAnyOrigin
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

    # Inflate by any predefined pair's margin. MuJoCo never subjects a
    # `<contact><pair>` to the broadphase at all — the merge loop collides it
    # whatever the AABBs say — so a pair whose two geoms sit further apart
    # than their extents but closer than its margin has to survive the sweep
    # below. Conservative by construction: a wider AABB only offers the narrow
    # phase more candidates, it never changes a contact.
    #
    # ⚠ ONLY the pair margin. The geoms' own `margin` is NOT folded in here,
    # though `mj_filterSphere` does exactly that and this sweep is the same
    # kind of bound — that is a separate pre-existing divergence, latent while
    # every geom margin in the tree is small next to the geometry, and one
    # that would move dog/quadruped/sawyer if changed. Left alone deliberately
    # rather than smuggled into a commit about pairs.
    var n_pair_aabb = Int(rebind[Scalar[DTYPE]](mmeta[MODEL_META_IDX_NPAIR]))
    # EPA's stopping rule, from model META — see `_detect_contacts_env` for
    # why it is read rather than hardcoded, and why a non-positive value falls
    # back instead of meaning "zero iterations".
    var ccd_tol = rebind[Scalar[DTYPE]](mmeta[MODEL_META_IDX_CCD_TOLERANCE])
    if ccd_tol <= 0:
        ccd_tol = Scalar[DTYPE](MJ_CCD_TOLERANCE)
    var ccd_iter = Int(
        rebind[Scalar[DTYPE]](mmeta[MODEL_META_IDX_CCD_ITERATIONS])
    )
    if ccd_iter < 1:
        ccd_iter = MJ_CCD_ITERATIONS
    if n_pair_aabb > NPAIR:
        n_pair_aabb = NPAIR
    for p in range(n_pair_aabb):
        var pm = rebind[Scalar[DTYPE]](pairs[p, PAIR_IDX_MARGIN])
        if pm <= Scalar[DTYPE](0):
            continue
        for side in range(2):
            var g = Int(
                rebind[Scalar[DTYPE]](
                    pairs[p, PAIR_IDX_GEOM1 if side == 0 else PAIR_IDX_GEOM2]
                )
            )
            if Int(rebind[Scalar[DTYPE]](geoms[g, GEOM_IDX_TYPE])) == (
                GEOM_PLANE
            ):
                continue  # planes have no AABB here
            aabb_min_x[g] -= pm
            aabb_max_x[g] += pm
            aabb_min_y[g] -= pm
            aabb_max_y[g] += pm
            aabb_min_z[g] -= pm
            aabb_max_z[g] += pm

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
            # `<contact><pair>` bypasses every filter below — see the same
            # gate in `_detect_contacts_env`. A plane/geom pair is a normal
            # thing to declare (it is the ONLY form ToddlerBot's scene files
            # use), and the world plane's body is 0, so without this the
            # `gj_body == 0` skip and the weld test would drop it.
            var ipair = find_predefined_pair[DTYPE, NPAIR](
                gi, gj, pairs, mmeta
            )
            if ipair < 0:
                if gj_body == 0:
                    continue
                # DEFECT 24 — this loop had NO body filter. MuJoCo runs the
                # plane path through `filterBodyPair` like every other pair
                # (`engine_collision_driver.c:1277`), which discards on
                # `weldbody1 == weldbody2`; a jointless body welds to the
                # world, so every static geom was colliding with the ground
                # here while the O(N^2) path correctly emitted nothing. See
                # `pair_body_filtered`.
                if pair_body_filtered[DTYPE, NBODY, NEXCLUDE](
                    gi_body, gj_body, bodies, mmeta, excludes
                ):
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

            # MuJoCo's full contact-parameter rule, PRIORITY FIRST — shared
            # with `detect_contacts` so the two paths cannot drift, which is
            # exactly how the SAP ellipsoid branch went missing. A predefined
            # pair supplies its own parameters instead, unmixed.
            var _mx = pair_params[DTYPE, NPAIR](
                ipair, pairs
            ) if ipair >= 0 else mix_contact_params[DTYPE](
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
            var cdim = Int(_mx[0])
            var cf = _mx[1]
            var cfs = _mx[2]
            var cfr = _mx[3]
            var _n0 = num_contacts
            var mgi = rebind[Scalar[DTYPE]](geoms[gi, GEOM_IDX_MARGIN])
            var mgj = rebind[Scalar[DTYPE]](geoms[gj, GEOM_IDX_MARGIN])
            # Sum of the two geoms' margins, or the PAIR's own — never both.
            var cm = mgi + mgj  # MuJoCo 3.5+: sum of margins
            if ipair >= 0:
                cm = rebind[Scalar[DTYPE]](pairs[ipair, PAIR_IDX_MARGIN])

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

            # ── PLANE-SIDE BOUNDING-SPHERE REJECT — MuJoCo's second
            # `mj_filterSphere` arm. In the plane's own frame `pj_z` IS
            # `planeGeomDist`: the signed distance from the plane to the geom
            # centre. If the geom's bounding sphere cannot reach the plane,
            # nothing downstream can produce a contact.
            #
            # ⚠⚠ WITHOUT THIS, A PLANE PAIRED WITH A MESH SCANS EVERY HULL
            # VERTEX, EVERY STEP, FOREVER. `_plane_mesh_contacts` has no early
            # out — it transforms all `pm_vnum` vertices looking for the
            # deepest. SO-ARM101 carries 30 mesh geoms totalling 33 076 hull
            # vertices and a floor its arm never touches, and that scan was
            # 72% of its entire physics step. It is also why the arm-to-arm
            # cost ratio tracked HULL SIZE rather than anything physical.
            #
            #     SO-ARM101   1.86 -> 0.65 ms/env step   ( 539 -> 1544 Hz)
            #     SO-ARM100   1.11 -> 1.04 ms/env step   ( 901 ->  959 Hz)
            #
            # ⚠ THE TWO ARMS SEPARATE HERE, AND THAT IS THE POINT. SO-ARM100
            # barely moves: 2 551 hull vertices is a scan it could afford.
            # SO-ARM101's 33 076 is not, and removing it INVERTS the pair —
            # the arm with 13x the geometry is now the FASTER of the two,
            # because what remains is no longer proportional to hull size.
            # SO-ARM100's residual is elsewhere (its Newton solve is ~25% of
            # its step, against ~0.5% of SO-ARM101's).
            #
            # ⚠ `+ cm` AGAIN, for the same silent reason as the geom-geom arm
            # above: a geom hovering within its margin of the floor is a
            # contact MuJoCo reports.
            var rbound_j_pl = rebind[Scalar[DTYPE]](
                geoms[gj, GEOM_IDX_RBOUND]
            )
            if rbound_j_pl > Scalar[DTYPE](0) and pj_z > cm + rbound_j_pl:
                continue

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
                    contacts[env, c_off + CONTACT_IDX_INCLUDEMARGIN] = cm
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
                    contacts[env, c_off + CONTACT_IDX_INCLUDEMARGIN] = cm
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
                    contacts[env, c_off + CONTACT_IDX_INCLUDEMARGIN] = cm
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
                    contacts[env, c_off + CONTACT_IDX_INCLUDEMARGIN] = cm
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
                # Plane-mesh. Was a verbatim copy of the O(N^2) path's vertex
                # scan, and carried the same defect: one contact per hull
                # vertex, uncapped. Both now go through the single
                # `_plane_mesh_contacts`, so the fix cannot land on one path
                # and miss the other — the duplication is what let a
                # `maxplanemesh` cap be absent from BOTH for as long as it was.
                #
                # ⚠ SAP'S RECORD CONVENTIONS ARE PRESERVED, NOT UNIFIED: this
                # path writes BODY_B = -1, stores `dist - margin` in DIST and
                # has no INCLUDEMARGIN slot (see the module docstring). Those
                # are gated bit-exactly elsewhere, so they are passed as
                # parameters rather than quietly aligned with the other path.
                comptime if NMESH_VERTS > 0:
                    _plane_mesh_contacts[
                        DTYPE, MAX_CONTACTS, NGEOM, NMESH_VERTS, BATCH,
                        -1, True, False,
                    ](
                        env,
                        gj,
                        gj_body,
                        pj_x, pj_y, pj_z,
                        qj_x, qj_y, qj_z, qj_w,
                        ground_z,
                        plp_x, plp_y, plp_z,
                        plq_x, plq_y, plq_z, plq_w,
                        cm,
                        cf,
                        cfs,
                        cfr,
                        cdim,
                        geoms,
                        mesh_meta,
                        mesh_verts,
                        mesh_vert_edgeadr,
                        mesh_edges,
                        contacts,
                        num_contacts,
                    )

            _fill_pair_solparams[DTYPE, MAX_CONTACTS, BATCH](
                env, _n0, num_contacts, _mx, contacts
            )

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
        # Multi-CCD scales its distinctness tolerance by the smaller bounding
        # radius (`mjc_Convex`).
        var rbound_i = rebind[Scalar[DTYPE]](geoms[gi, GEOM_IDX_RBOUND])

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
            # `<contact><pair>` bypasses every filter below — see the same
            # gate in `_detect_contacts_env`. The AABB tests above still
            # apply, which is why the AABBs are inflated by the pair margin
            # where they are built: MuJoCo collides predefined pairs outside
            # the broadphase entirely, so a pair must not be prunable by a
            # bound that ignores its margin.
            var ipair = find_predefined_pair[DTYPE, NPAIR](
                gi, gj, pairs, mmeta
            )
            if ipair < 0:
                # MuJoCo's body-pair filter — weld, weld-parent and exclude.
                # See `pair_body_filtered`; shared with the O(N^2) loop and
                # the plane loop above, which had no body filter at all
                # (defect 24).
                if pair_body_filtered[DTYPE, NBODY, NEXCLUDE](
                    gi_body, gj_body, bodies, mmeta, excludes
                ):
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

            var mgi = rebind[Scalar[DTYPE]](geoms[gi, GEOM_IDX_MARGIN])
            var mgj = rebind[Scalar[DTYPE]](geoms[gj, GEOM_IDX_MARGIN])
            # Sum of the two geoms' margins, or the PAIR's own — never both.
            var cm = mgi + mgj  # MuJoCo 3.5+: sum of margins
            if ipair >= 0:
                cm = rebind[Scalar[DTYPE]](pairs[ipair, PAIR_IDX_MARGIN])

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
            var rbound_j = rebind[Scalar[DTYPE]](geoms[gj, GEOM_IDX_RBOUND])

            # ── BOUNDING-SPHERE REJECT — MuJoCo's `mj_filterSphere` ────────
            # ⚠⚠ THIS PATH RAN WITHOUT IT AND THE O(N^2) PATH DID NOT. MuJoCo
            # applies the test inside `mj_collideGeoms`, which sits DOWNSTREAM
            # of whichever broadphase produced the pair, so it covers every
            # candidate. Ours lived only in `contact_detection.mojo`, so every
            # model big enough to take the SAP branch (`NGEOM >= 16` — which is
            # every interesting one) sent pairs into GJK that MuJoCo rejects
            # with three subtractions. The AABB tests above do NOT subsume it:
            # a sweep overlap on inflated world AABBs is far weaker than the
            # two bounding spheres actually touching.
            #
            # Measured, ms per env step (`FRAME_SKIP=10`), MIN of two
            # interleaved rounds against a pristine worktree of the parent:
            #
            #     SO-ARM100   2.87 -> 1.09   (349 -> 918 Hz)
            #     SO-ARM101   4.77 -> 1.84   (210 -> 544 Hz)
            #
            # MuJoCo steps the same two XMLs at 0.078 and 0.121 ms, so the
            # remaining gap is 14x and 15x, down from 37x and 39x.
            #
            # ⚠ `+ cm` IS LOAD-BEARING, and its absence is silent. A pair
            # separated by more than the two radii but LESS than its margin is
            # a contact MuJoCo reports; drop the term and it vanishes with no
            # error anywhere. This is the same trap the O(N^2) copy documents,
            # which is where the term was missing once before.
            #
            # ⚠ PLANES ARE EXCLUDED BY `rbound > 0`, which is how MuJoCo
            # detects them here too (a plane's `rbound` is 0 because it is
            # unbounded). MuJoCo additionally has a plane-specific arm using
            # `planeGeomDist`; that is NOT implemented here or in the O(N^2)
            # path, so plane pairs fall through to narrow phase exactly as
            # they did before this change.
            if rbound_i > Scalar[DTYPE](0) and rbound_j > Scalar[DTYPE](0):
                var sfx = pi_x - pj_x
                var sfy = pi_y - pj_y
                var sfz = pi_z - pj_z
                var sfb = rbound_i + rbound_j + cm
                if sfx * sfx + sfy * sfy + sfz * sfz > sfb * sfb:
                    continue

            # ⚠⚠ THE CONTACT-PARAMETER MIX RUNS **AFTER** THE SPHERE
            # REJECT, NOT BEFORE, AND THE ORDER IS THE POINT.
            # `mix_contact_params` is ~30 tensor reads plus MuJoCo's
            # priority/max/min rules, and it used to run on every pair that
            # survived the body/contype filters — 65 per step on SO-ARM100,
            # of which the bounding-sphere test then rejects all but 2.
            # Nothing above needs it: the reject reads only the two rbounds
            # and `cm`, and `cm` comes from the geoms' own margins (or the
            # pair's), never from the mix. `_n0` moves with it because it is
            # a snapshot of `num_contacts`, which the reject cannot change.
            #
            # ⚠ THIS IS NOT THE HOIST §5.1 MEASURED AT ZERO. That one tried
            # to compute the per-GEOM decode once per geom; the mix is
            # per-PAIR and hoisting cannot remove it. Deferring past the
            # reject removes 97% of the CALLS.
            # MuJoCo's full contact-parameter rule, PRIORITY FIRST — shared
            # with `detect_contacts` so the two paths cannot drift, which is
            # exactly how the SAP ellipsoid branch went missing. A predefined
            # pair supplies its own parameters instead, unmixed.
            var _mx = pair_params[DTYPE, NPAIR](
                ipair, pairs
            ) if ipair >= 0 else mix_contact_params[DTYPE](
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
            var cdim = Int(_mx[0])
            var cf = _mx[1]
            var cfs = _mx[2]
            var cfr = _mx[3]
            var _n0 = num_contacts

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
            # Mesh vertex ranges, hoisted out of the mesh branch so multi-CCD
            # can re-run the SAME convex query at its perturbed poses. Zero for
            # every non-mesh pair, which is what `gjk_epa` wants there.
            var va1 = 0
            var mnv1 = 0
            var va2 = 0
            var mnv2 = 0

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
                    cm, cf, cfs, cfr, cdim,
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
                    cm,
                    cf,
                    cfs,
                    cfr,
                    cdim,
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

            elif (gi_type == GEOM_CYLINDER and gj_type == GEOM_BOX) or (
                gi_type == GEOM_BOX and gj_type == GEOM_CYLINDER
            ):
                # MuJoCo routes CYLINDER x BOX to `mjc_Convex` — GJK plus EPA
                # (`engine_collision_driver.c:41`), not to a primitive. Ours
                # used `cylinder_box`, which REDUCES THE CYLINDER TO A CAPSULE,
                # so the hemispherical cap dips a full radius below the flat
                # face. Measured against the analytic depth that is an error of
                # exactly -r in EVERY configuration, separated or penetrating:
                # at 1 cm of CLEARANCE it still reported a 4 cm penetration. On
                # sawyer (obj r = 0.02) it manufactured a 2 cm contact at the
                # canonical reset pose, where MuJoCo has none and where all 13
                # Phase 7 manipulation tasks begin.
                #
                # ⚠ THIS RE-ROUTE WAS ATTEMPTED ONCE BEFORE AND REVERTED. It
                # dropped contacts at SHALLOW penetration in the RIM
                # configuration, because GJK handed EPA a 2-simplex that did
                # not enclose the origin. `gjkIntersect` (`4b773bdf`) is what
                # made it viable; without that commit this branch is wrong.
                #
                # One branch for both orderings: `cylinder_box` needed two
                # because the primitive is asymmetric in its operands, but the
                # convex query is symmetric and returns `gi -> gj` either way.
                var r = gjk_epa[DTYPE, NMESH_VERTS](
                    gi_type,
                    pi_x, pi_y, pi_z, qi_x, qi_y, qi_z, qi_w,
                    ri, hli, hxi, hyi, hzi,
                    mesh_verts, mesh_vert_edgeadr, mesh_edges, 0, 0,
                    gj_type,
                    pj_x, pj_y, pj_z, qj_x, qj_y, qj_z, qj_w,
                    rj, hlj, hxj, hyj, hzj,
                    0, 0,
                    ccd_tol, ccd_iter, cm,
                )
                dist = r[0]
                cx = r[1]
                cy = r[2]
                cz = r[3]
                nx = r[4]
                ny = r[5]
                nz = r[6]

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
                    if mi_id >= 0:
                        va1 = Int(rebind[Scalar[DTYPE]](mesh_meta[mi_id, 0]))
                        mnv1 = Int(rebind[Scalar[DTYPE]](mesh_meta[mi_id, 1]))
                    if mj_id >= 0:
                        va2 = Int(rebind[Scalar[DTYPE]](mesh_meta[mj_id, 0]))
                        mnv2 = Int(rebind[Scalar[DTYPE]](mesh_meta[mj_id, 1]))

                    # NATIVE MULTI-CONTACT — the SAME dispatch as
                    # `contact_detection.mojo`. ⚠ THIS FILE IS A SECOND COPY OF
                    # THE NARROW PHASE, and when the manifold path landed there
                    # first the two producers disagreed: an env on the SAP path
                    # got ONE point for a mesh pair where the O(N^2) path gave
                    # four. Same model, different contacts, decided by which
                    # broadphase the config happened to select. See
                    # `feedback_one_field_two_producers`.
                    var mc_pair = (
                        MC_ENABLED
                        and (gi_type == GEOM_MESH or gi_type == GEOM_BOX)
                        and (gj_type == GEOM_MESH or gj_type == GEOM_BOX)
                        and cm <= Scalar[DTYPE](0)
                    )
                    var wf1 = InlineArray[Scalar[DTYPE], 9](
                        fill=Scalar[DTYPE](0)
                    )
                    var wf2 = InlineArray[Scalar[DTYPE], 9](
                        fill=Scalar[DTYPE](0)
                    )
                    var wxx = InlineArray[Scalar[DTYPE], 6](
                        fill=Scalar[DTYPE](0)
                    )
                    var wf_ok = 0
                    var result = gjk_epa_witness[DTYPE, NMESH_VERTS](
                        gi_type,
                        pi_x, pi_y, pi_z, qi_x, qi_y, qi_z, qi_w,
                        ri, hli, hxi, hyi, hzi,
                        mesh_verts, mesh_vert_edgeadr, mesh_edges, va1, mnv1,
                        gj_type,
                        pj_x, pj_y, pj_z, qj_x, qj_y, qj_z, qj_w,
                        rj, hlj, hxj, hyj, hzj,
                        va2, mnv2,
                        wf1, wf2, wxx, wf_ok,
                        ccd_tol, ccd_iter, cm,
                        # Opt in to the cutoff exit: `dist` below is read ONLY
                        # by `if dist < cm`, and everything that consumes the
                        # witness sits inside that branch.
                        cm,
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

                    if (
                        mc_pair
                        and wf_ok == 1
                        and dist < cm
                        and num_contacts < MAX_CONTACTS
                    ):
                        var pa1 = 0
                        var pn1 = 0
                        var pa2 = 0
                        var pn2 = 0
                        if mi_id >= 0:
                            pa1 = Int(rebind[Scalar[DTYPE]](
                                mesh_meta[mi_id, MESH_META_IDX_POLYADR]
                            ))
                            pn1 = Int(rebind[Scalar[DTYPE]](
                                mesh_meta[mi_id, MESH_META_IDX_POLYNUM]
                            ))
                        if mj_id >= 0:
                            pa2 = Int(rebind[Scalar[DTYPE]](
                                mesh_meta[mj_id, MESH_META_IDX_POLYADR]
                            ))
                            pn2 = Int(rebind[Scalar[DTYPE]](
                                mesh_meta[mj_id, MESH_META_IDX_POLYNUM]
                            ))
                        # Operands in MuJoCo's order (lower geom type first),
                        # and the witness pair swaps with them — `dir` is a
                        # SIGNED input to `boxNormals2`, not a magnitude.
                        # ⚠ TYPE ALONE DOES NOT ORDER A PAIR — for EQUAL types
                        # this is false either way and the operand order fell
                        # out of the broadphase, which is exactly how SAP and
                        # the O(N^2) loop came to disagree with each other.
                        # Tie-break on geom index, as `mj_collideGeoms` does.
                        var mc_swap = gi_type > gj_type or (
                            gi_type == gj_type and gi > gj
                        )
                        var wxs = InlineArray[Scalar[DTYPE], 6](
                            fill=Scalar[DTYPE](0)
                        )
                        wxs[0] = wxx[3]
                        wxs[1] = wxx[4]
                        wxs[2] = wxx[5]
                        wxs[3] = wxx[0]
                        wxs[4] = wxx[1]
                        wxs[5] = wxx[2]
                        var mcn = 0
                        if mc_swap:
                            mcn = native_multicontact_contacts[
                                DTYPE, NMESH_VERTS,
                                mesh_max_poly(NMESH_VERTS),
                                mesh_max_polyvert(NMESH_VERTS),
                                MAX_CONTACTS, BATCH,
                            ](
                                env, body_a, body_b,
                                gj_type,
                                pj_x, pj_y, pj_z, qj_x, qj_y, qj_z, qj_w,
                                hxj, hyj, hzj, rbound_j, va2, mnv2, pa2, pn2,
                                gi_type,
                                pi_x, pi_y, pi_z, qi_x, qi_y, qi_z, qi_w,
                                hxi, hyi, hzi, rbound_i, va1, mnv1, pa1, pn1,
                                mesh_verts, mesh_polys, mesh_polyvert,
                                mesh_polymap, mesh_vert_polymap,
                                wf2, wf1, wxs,
                                dist, cm, cf, cfs, cfr, cdim,
                                True,
                                contacts, num_contacts,
                            )
                        else:
                            mcn = native_multicontact_contacts[
                                DTYPE, NMESH_VERTS,
                                mesh_max_poly(NMESH_VERTS),
                                mesh_max_polyvert(NMESH_VERTS),
                                MAX_CONTACTS, BATCH,
                            ](
                                env, body_a, body_b,
                                gi_type,
                                pi_x, pi_y, pi_z, qi_x, qi_y, qi_z, qi_w,
                                hxi, hyi, hzi, rbound_i, va1, mnv1, pa1, pn1,
                                gj_type,
                                pj_x, pj_y, pj_z, qj_x, qj_y, qj_z, qj_w,
                                hxj, hyj, hzj, rbound_j, va2, mnv2, pa2, pn2,
                                mesh_verts, mesh_polys, mesh_polyvert,
                                mesh_polymap, mesh_vert_polymap,
                                wf1, wf2, wxx,
                                dist, cm, cf, cfs, cfr, cdim,
                                False,
                                contacts, num_contacts,
                            )
                        # The manifold REPLACES the single point.
                        if mcn > 0:
                            _fill_pair_solparams[
                                DTYPE, MAX_CONTACTS, BATCH
                            ](env, _n0, num_contacts, _mx, contacts)
                            continue
                else:
                    _fill_pair_solparams[DTYPE, MAX_CONTACTS, BATCH](
                        env, _n0, num_contacts, _mx, contacts
                    )
                    continue

            if dist < cm and num_contacts < MAX_CONTACTS:
                # The `gi -> gj` normal, captured BEFORE the emit negates it in
                # place — see the identical capture in `contact_detection.mojo`.
                var mccd_nx = nx
                var mccd_ny = ny
                var mccd_nz = nz
                var mccd_first = num_contacts
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
                contacts[env, c_off + CONTACT_IDX_INCLUDEMARGIN] = cm
                contacts[env, c_off + CONTACT_IDX_FRICTION] = cf
                contacts[env, c_off + CONTACT_IDX_FRICTION_SPIN] = cfs
                contacts[env, c_off + CONTACT_IDX_FRICTION_ROLL] = cfr
                contacts[env, c_off + CONTACT_IDX_CONDIM] = Scalar[DTYPE](
                    cdim
                )
                num_contacts += 1

                # MULTI-POINT CONVEX CONTACT — defect 21.
                #
                # ⚠⚠ THIS FILE IS THE SECOND NARROW PHASE. `contact_detection`
                # carries the same dispatch and the same emit, and SAP takes
                # over at NGEOM >= SAP_THRESHOLD — so patching only the other
                # one would have left every LARGE model (dog, quadruped: the
                # exact models this was found on) with single-point cylinder
                # contacts while the small-model gate went green. That is the
                # shape of `feedback_sap_path_missing_a_whole_geom_type`, and
                # it is why this hook is duplicated rather than "left for
                # later". The two must move together.
                if multi_ccd_pair_supported(gi_type, gj_type):
                    _ = multi_ccd_extra_contacts[
                        DTYPE, MAX_CONTACTS, BATCH, NMESH_VERTS
                    ](
                        env, body_a, body_b, mccd_first,
                        gi_type,
                        pi_x, pi_y, pi_z, qi_x, qi_y, qi_z, qi_w,
                        ri, hli, hxi, hyi, hzi, rbound_i, va1, mnv1,
                        gj_type,
                        pj_x, pj_y, pj_z, qj_x, qj_y, qj_z, qj_w,
                        rj, hlj, hxj, hyj, hzj, rbound_j, va2, mnv2,
                        mesh_verts,
                        mesh_vert_edgeadr,
                        mesh_edges,
                        cx, cy, cz,
                        mccd_nx, mccd_ny, mccd_nz,
                        dist,
                        cm, cf, cfs, cfr, cdim,
                        contacts, num_contacts,
                        ccd_tol, ccd_iter, cm,
                    )

            _fill_pair_solparams[DTYPE, MAX_CONTACTS, BATCH](
                env, _n0, num_contacts, _mx, contacts
            )

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
    # Appended rather than grouped with NEXCLUDE — see `fields.Model`.
    NPAIR: Int,
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
    pairs: LayoutTensor[
        DTYPE, Layout.row_major(NPAIR, MODEL_PAIR_SIZE), MutAnyOrigin
    ],
    mesh_meta: LayoutTensor[
        DTYPE,
        Layout.row_major(MAX_GPU_MESHES, MODEL_MESH_META_SIZE),
        MutAnyOrigin,
    ],
    mesh_verts: LayoutTensor[
        DTYPE, Layout.row_major(NMESH_VERTS, 3), MutAnyOrigin
    ],
    mesh_polys: LayoutTensor[
        DTYPE,
        Layout.row_major(mesh_max_poly(NMESH_VERTS), MODEL_MESH_POLY_SIZE),
        MutAnyOrigin,
    ],
    mesh_polyvert: LayoutTensor[
        DTYPE, Layout.row_major(mesh_max_polyvert(NMESH_VERTS)), MutAnyOrigin
    ],
    mesh_polymap: LayoutTensor[
        DTYPE, Layout.row_major(mesh_max_polyvert(NMESH_VERTS)), MutAnyOrigin
    ],
    mesh_vert_polymap: LayoutTensor[
        DTYPE, Layout.row_major(NMESH_VERTS, 2), MutAnyOrigin
    ],
    mesh_vert_edgeadr: LayoutTensor[
        DTYPE, Layout.row_major(NMESH_VERTS), MutAnyOrigin
    ],
    mesh_edges: LayoutTensor[
        DTYPE, Layout.row_major(mesh_max_edge(NMESH_VERTS)), MutAnyOrigin
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
        NMESH_VERTS, BATCH, NPAIR,
    ](
        env, xpos, xquat, geoms, bodies, mmeta, excludes, pairs, mesh_meta,
        mesh_verts, mesh_polys, mesh_polyvert, mesh_polymap,
        mesh_vert_polymap, mesh_vert_edgeadr, mesh_edges, contacts, smeta,
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
    NPAIR: Int = 0,
](
    mut d: Data[DTYPE, Dims[nq=NQ, nv=NV, nbody=NBODY, max_contacts=MAX_CONTACTS, nsite=NSITE], BATCH],
    mut m: Model[DTYPE, Dims[nv=NV, nbody=NBODY, njoint=NJOINT, ngeom=NGEOM, nequality=NEQUALITY, ntendon=NTENDON, nsite=NSITE, nexclude=NEXCLUDE, nmesh_verts=NMESH_VERTS, npair=NPAIR]],
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
    comptime L_PAIR = Layout.row_major(NPAIR, MODEL_PAIR_SIZE)
    comptime L_MESH_META = Layout.row_major(
        MAX_GPU_MESHES, MODEL_MESH_META_SIZE
    )
    comptime L_MESH_VERT = Layout.row_major(NMESH_VERTS, 3)
    comptime L_MESH_POLY = Layout.row_major(
        mesh_max_poly(NMESH_VERTS), MODEL_MESH_POLY_SIZE
    )
    comptime L_MESH_POLYVERT = Layout.row_major(mesh_max_polyvert(NMESH_VERTS))
    comptime L_MESH_VPMAP = Layout.row_major(NMESH_VERTS, 2)
    comptime L_MESH_VEADR = Layout.row_major(NMESH_VERTS)
    comptime L_MESH_EDGE = Layout.row_major(mesh_max_edge(NMESH_VERTS))
    comptime L_CONTACTS = Layout.row_major(BATCH, MAX_CONTACTS * CONTACT_SIZE)
    comptime L_SMETA = Layout.row_major(BATCH, METADATA_SIZE)

    comptime if target == "cpu":
        var xpos_v = d.xpos.lt["cpu", L_B3]()
        var xquat_v = d.xquat.lt["cpu", L_B4]()
        var geoms_v = m.geoms.lt["cpu", L_GEOM]()
        var bodies_v = m.bodies.lt["cpu", L_BODY]()
        var mmeta_v = m.meta.lt["cpu", L_MMETA]()
        var excludes_v = m.excludes.lt["cpu", L_EXCLUDE]()
        var pairs_v = m.pairs.lt["cpu", L_PAIR]()
        var mesh_meta_v = m.mesh_meta.lt["cpu", L_MESH_META]()
        var mesh_verts_v = m.mesh_verts.lt["cpu", L_MESH_VERT]()
        var mesh_polys_v = m.mesh_polys.lt["cpu", L_MESH_POLY]()
        var mesh_polyvert_v = m.mesh_polyvert.lt["cpu", L_MESH_POLYVERT]()
        var mesh_polymap_v = m.mesh_polymap.lt["cpu", L_MESH_POLYVERT]()
        var mesh_vert_polymap_v = m.mesh_vert_polymap.lt["cpu", L_MESH_VPMAP]()
        var mesh_vert_edgeadr_v = m.mesh_vert_edgeadr.lt[
            "cpu", L_MESH_VEADR
        ]()
        var mesh_edges_v = m.mesh_edges.lt["cpu", L_MESH_EDGE]()
        var contacts_v = d.contacts.lt["cpu", L_CONTACTS]()
        var smeta_v = d.meta.lt["cpu", L_SMETA]()
        for e in range(BATCH):
            _detect_contacts_sap_env[
                DTYPE, NQ, NV, NBODY, NJOINT, MAX_CONTACTS, NGEOM,
                NEXCLUDE, NMESH_VERTS, BATCH, NPAIR,
            ](
                e, xpos_v, xquat_v, geoms_v, bodies_v, mmeta_v,
                excludes_v, pairs_v, mesh_meta_v, mesh_verts_v, mesh_polys_v,
                mesh_polyvert_v, mesh_polymap_v, mesh_vert_polymap_v,
                mesh_vert_edgeadr_v, mesh_edges_v,
                contacts_v, smeta_v,
            )
    else:
        var c = ctx.value()
        comptime BLOCKS = (BATCH + SAP_TPB - 1) // SAP_TPB
        c.enqueue_function[
            _detect_contacts_sap_fields_kernel[
                DTYPE, NQ, NV, NBODY, NJOINT, MAX_CONTACTS, NGEOM,
                NEXCLUDE, NMESH_VERTS, BATCH, NPAIR,
            ]
        ](
            d.xpos.lt["gpu", L_B3](),
            d.xquat.lt["gpu", L_B4](),
            m.geoms.lt["gpu", L_GEOM](),
            m.bodies.lt["gpu", L_BODY](),
            m.meta.lt["gpu", L_MMETA](),
            m.excludes.lt["gpu", L_EXCLUDE](),
            m.pairs.lt["gpu", L_PAIR](),
            m.mesh_meta.lt["gpu", L_MESH_META](),
            m.mesh_verts.lt["gpu", L_MESH_VERT](),
            m.mesh_polys.lt["gpu", L_MESH_POLY](),
            m.mesh_polyvert.lt["gpu", L_MESH_POLYVERT](),
            m.mesh_polymap.lt["gpu", L_MESH_POLYVERT](),
            m.mesh_vert_polymap.lt["gpu", L_MESH_VPMAP](),
            m.mesh_vert_edgeadr.lt["gpu", L_MESH_VEADR](),
            m.mesh_edges.lt["gpu", L_MESH_EDGE](),
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
    NPAIR: Int = 0,
](
    mut d: Data[DTYPE, Dims[nq=NQ, nv=NV, nbody=NBODY, max_contacts=MAX_CONTACTS, nsite=NSITE], BATCH],
    mut m: Model[DTYPE, Dims[nv=NV, nbody=NBODY, njoint=NJOINT, ngeom=NGEOM, nequality=NEQUALITY, ntendon=NTENDON, nsite=NSITE, nexclude=NEXCLUDE, nmesh_verts=NMESH_VERTS, npair=NPAIR]],
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
            NEQUALITY, NTENDON, NSITE, NEXCLUDE, NMESH_VERTS, BATCH, NPAIR,
        ](d, m, ctx)
    else:
        detect_contacts[
            target, DTYPE, NQ, NV, NBODY, NJOINT, MAX_CONTACTS, NGEOM,
            NEQUALITY, NTENDON, NSITE, NEXCLUDE, NMESH_VERTS, BATCH, NPAIR,
        ](d, m, ctx)
