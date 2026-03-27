"""AABB/SAP broadphase collision detection for physics3d.

Implements sweep-and-prune (SAP) on axis-aligned bounding boxes (AABBs),
reducing the O(N²) broadphase pair test to O(N log N + C) where C is the
number of candidate pairs.

Typical usage — call detect_contacts_auto / detect_contacts_auto_gpu and let
the compile-time dispatch pick the right implementation automatically:

    detect_contacts_auto(model, data)   # SAP when NGEOM >= SAP_THRESHOLD else O(N²)

Algorithm (SAP path)
--------------------
1. Compute world positions for all geoms upfront (O(N)).
2. Compute tight AABB half-extents per geom type (O(N)).
3. Handle plane-vs-non-plane pairs via a direct O(P×N) loop (P = plane count).
4. For non-plane pairs:
   a. Build a SAP index list and insertion-sort by AABB min_x  (O(N log N)).
   b. Sweep with an active-set break condition, check Y/Z overlap  (O(N + C)).
   c. For each candidate pair: body filter → narrowphase dispatch.
"""

from std.math import sqrt, abs
from layout import LayoutTensor, Layout
from ..types import (
    Model,
    Data,
    ConeType,
)
from ..constants import (
    GEOM_SPHERE,
    GEOM_CAPSULE,
    GEOM_BOX,
    GEOM_PLANE,
    GEOM_CYLINDER,
)
from ..kinematics.quat_math import (
    quat_rotate,
    quat_mul,
    gpu_quat_rotate,
    gpu_quat_mul,
)
from ..gpu.constants import (
    BODY_IDX_PARENT,
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
    META_IDX_NUM_CONTACTS,
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
    GEOM_IDX_FRICTION_SPIN,
    GEOM_IDX_FRICTION_ROLL,
    GEOM_IDX_CONTYPE,
    GEOM_IDX_CONAFFINITY,
    GEOM_IDX_CONDIM,
    GEOM_IDX_MARGIN,
    model_body_offset,
    model_geom_offset,
    xpos_offset,
    xquat_offset,
    contacts_offset,
    metadata_offset,
)
from .collision_primitives import (
    sphere_sphere,
    capsule_sphere,
    capsule_capsule,
    box_sphere,
    box_capsule,
    box_box,
    cylinder_plane,
    cylinder_sphere,
)
from .contact_detection import (
    _geom_world_pos,
    _geom_world_pos_gpu,
    detect_contacts,
    detect_contacts_gpu,
)


# =============================================================================
# AABB helper
# =============================================================================


@always_inline
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


# =============================================================================
# CPU: detect_contacts_sap
# =============================================================================


def detect_contacts_sap[
    DTYPE: DType,
    NQ: Int,
    NV: Int,
    NBODY: Int,
    NJOINT: Int,
    MAX_CONTACTS: Int,
    NGEOM: Int,
    MAX_EQUALITY: Int = 0,
    CONE_TYPE: Int = ConeType.ELLIPTIC,
    MAX_TENDON: Int = 0,
    NSITE: Int = 0,
](
    model: Model[
        DTYPE,
        NQ,
        NV,
        NBODY,
        NJOINT,
        MAX_CONTACTS,
        NGEOM,
        MAX_EQUALITY,
        CONE_TYPE,
        MAX_TENDON,
        NSITE,
    ],
    mut data: Data[DTYPE, NQ, NV, NBODY, NJOINT, MAX_CONTACTS, NSITE],
):
    """Contact detection using AABB/SAP broadphase (CPU).

    Drop-in replacement for detect_contacts. Recommended for scenes with 50+
    geoms, where the O(N log N + C) SAP sweep is faster than O(N²) bounding
    sphere checks.
    """
    data.num_contacts = 0

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
        var w = _geom_world_pos(model, data, g)
        wpx[g] = w[0]
        wpy[g] = w[1]
        wpz[g] = w[2]
        wqx[g] = w[3]
        wqy[g] = w[4]
        wqz[g] = w[5]
        wqw[g] = w[6]

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
        if model.geom_type[g] == GEOM_PLANE:
            continue  # planes skip SAP — handled in step 3
        var he = _aabb_half_extents[DTYPE](
            model.geom_type[g],
            wqx[g],
            wqy[g],
            wqz[g],
            wqw[g],
            model.geom_radius[g],
            model.geom_half_length[g],
            model.geom_half_x[g],
            model.geom_half_y[g],
            model.geom_half_z[g],
        )
        aabb_min_x[g] = wpx[g] - he[0]
        aabb_max_x[g] = wpx[g] + he[0]
        aabb_min_y[g] = wpy[g] - he[1]
        aabb_max_y[g] = wpy[g] + he[1]
        aabb_min_z[g] = wpz[g] - he[2]
        aabb_max_z[g] = wpz[g] + he[2]

    # ------------------------------------------------------------------
    # 3. Plane vs non-plane pairs  (O(P × N), usually P=1).
    # ------------------------------------------------------------------
    for gi in range(NGEOM):
        if model.geom_type[gi] != GEOM_PLANE:
            continue
        var gi_body = model.geom_body[gi]
        var gi_contype = model.geom_contype[gi]
        var gi_conaffinity = model.geom_conaffinity[gi]
        var ground_z = wpz[gi]

        for gj in range(NGEOM):
            if data.num_contacts >= MAX_CONTACTS:
                return
            var gj_type = model.geom_type[gj]
            if gj_type == GEOM_PLANE:
                continue
            var gj_body = model.geom_body[gj]
            if gi_body == 0 and gj_body == 0:
                continue
            if gj_body == 0:
                continue  # static geom on world body
            var gj_contype = model.geom_contype[gj]
            var gj_conaffinity = model.geom_conaffinity[gj]
            if (gi_contype & gj_conaffinity) == 0 and (
                gj_contype & gi_conaffinity
            ) == 0:
                continue

            var cf = model.geom_friction[gi]
            if model.geom_friction[gj] > cf:
                cf = model.geom_friction[gj]
            var cfs = model.geom_friction_spin[gi]
            if model.geom_friction_spin[gj] > cfs:
                cfs = model.geom_friction_spin[gj]
            var cfr = model.geom_friction_roll[gi]
            if model.geom_friction_roll[gj] > cfr:
                cfr = model.geom_friction_roll[gj]
            var cdim = model.geom_condim[gi]
            if model.geom_condim[gj] > cdim:
                cdim = model.geom_condim[gj]
            # MuJoCo 3.5+: margin = sum of both geom margins
            var cm = model.geom_margin[gi] + model.geom_margin[gj]

            var pj_x = wpx[gj]
            var pj_y = wpy[gj]
            var pj_z = wpz[gj]
            var qj_x = wqx[gj]
            var qj_y = wqy[gj]
            var qj_z = wqz[gj]
            var qj_w = wqw[gj]
            var rj = model.geom_radius[gj]
            var hlj = model.geom_half_length[gj]

            if gj_type == GEOM_SPHERE:
                var dist = pj_z - rj - ground_z
                if dist < cm and data.num_contacts < MAX_CONTACTS:
                    var idx = data.num_contacts
                    data.contacts[idx].body_a = gj_body
                    data.contacts[idx].body_b = 0
                    data.contacts[idx].pos_x = pj_x
                    data.contacts[idx].pos_y = pj_y
                    data.contacts[idx].pos_z = ground_z + dist * Scalar[DTYPE](
                        0.5
                    )
                    data.contacts[idx].normal_x = Scalar[DTYPE](0)
                    data.contacts[idx].normal_y = Scalar[DTYPE](0)
                    data.contacts[idx].normal_z = Scalar[DTYPE](1)
                    data.contacts[idx].dist = dist - cm
                    data.contacts[idx].friction = cf
                    data.contacts[idx].friction_spin = cfs
                    data.contacts[idx].friction_roll = cfr
                    data.contacts[idx].condim = cdim
                    data.num_contacts += 1

            elif gj_type == GEOM_CAPSULE:
                # Test both endpoints of the capsule (MuJoCo mjc_PlaneCapsule)
                var axis_w = quat_rotate(
                    qj_x,
                    qj_y,
                    qj_z,
                    qj_w,
                    Scalar[DTYPE](0),
                    Scalar[DTYPE](0),
                    Scalar[DTYPE](1),
                )
                var e1_x = pj_x + hlj * axis_w[0]
                var e1_y = pj_y + hlj * axis_w[1]
                var e1_z = pj_z + hlj * axis_w[2]
                var dist1 = e1_z - rj - ground_z
                if dist1 < cm and data.num_contacts < MAX_CONTACTS:
                    var idx = data.num_contacts
                    data.contacts[idx].body_a = gj_body
                    data.contacts[idx].body_b = 0
                    data.contacts[idx].pos_x = e1_x
                    data.contacts[idx].pos_y = e1_y
                    data.contacts[idx].pos_z = ground_z + dist1 * Scalar[DTYPE](
                        0.5
                    )
                    data.contacts[idx].normal_x = Scalar[DTYPE](0)
                    data.contacts[idx].normal_y = Scalar[DTYPE](0)
                    data.contacts[idx].normal_z = Scalar[DTYPE](1)
                    data.contacts[idx].dist = dist1 - cm
                    data.contacts[idx].friction = cf
                    data.contacts[idx].friction_spin = cfs
                    data.contacts[idx].friction_roll = cfr
                    data.contacts[idx].condim = cdim
                    data.contacts[idx].frame_t1_x = axis_w[0]
                    data.contacts[idx].frame_t1_y = axis_w[1]
                    data.contacts[idx].frame_t1_z = axis_w[2]
                    data.num_contacts += 1
                var e2_x = pj_x - hlj * axis_w[0]
                var e2_y = pj_y - hlj * axis_w[1]
                var e2_z = pj_z - hlj * axis_w[2]
                var dist2 = e2_z - rj - ground_z
                if dist2 < cm and data.num_contacts < MAX_CONTACTS:
                    var idx = data.num_contacts
                    data.contacts[idx].body_a = gj_body
                    data.contacts[idx].body_b = 0
                    data.contacts[idx].pos_x = e2_x
                    data.contacts[idx].pos_y = e2_y
                    data.contacts[idx].pos_z = ground_z + dist2 * Scalar[DTYPE](
                        0.5
                    )
                    data.contacts[idx].normal_x = Scalar[DTYPE](0)
                    data.contacts[idx].normal_y = Scalar[DTYPE](0)
                    data.contacts[idx].normal_z = Scalar[DTYPE](1)
                    data.contacts[idx].dist = dist2 - cm
                    data.contacts[idx].friction = cf
                    data.contacts[idx].friction_spin = cfs
                    data.contacts[idx].friction_roll = cfr
                    data.contacts[idx].condim = cdim
                    data.contacts[idx].frame_t1_x = axis_w[0]
                    data.contacts[idx].frame_t1_y = axis_w[1]
                    data.contacts[idx].frame_t1_z = axis_w[2]
                    data.num_contacts += 1

            elif gj_type == GEOM_CYLINDER:
                var cp = cylinder_plane[DTYPE](
                    pj_x,
                    pj_y,
                    pj_z,
                    qj_x,
                    qj_y,
                    qj_z,
                    qj_w,
                    hlj,
                    rj,
                    ground_z,
                )
                var dist = cp[0]
                if dist < cm and data.num_contacts < MAX_CONTACTS:
                    var idx = data.num_contacts
                    data.contacts[idx].body_a = gj_body
                    data.contacts[idx].body_b = 0
                    data.contacts[idx].pos_x = cp[1]
                    data.contacts[idx].pos_y = cp[2]
                    data.contacts[idx].pos_z = cp[3]
                    data.contacts[idx].normal_x = Scalar[DTYPE](0)
                    data.contacts[idx].normal_y = Scalar[DTYPE](0)
                    data.contacts[idx].normal_z = Scalar[DTYPE](1)
                    data.contacts[idx].dist = dist - cm
                    data.contacts[idx].friction = cf
                    data.contacts[idx].friction_spin = cfs
                    data.contacts[idx].friction_roll = cfr
                    data.contacts[idx].condim = cdim
                    data.num_contacts += 1

    # ------------------------------------------------------------------
    # 4. SAP sweep for non-plane pairs.
    # ------------------------------------------------------------------

    # 4a. Build SAP index list (non-plane geoms only).
    var sap_idx = InlineArray[Int, NGEOM](uninitialized=True)
    var sap_n = 0
    for g in range(NGEOM):
        if model.geom_type[g] != GEOM_PLANE:
            sap_idx[sap_n] = g
            sap_n += 1

    # 4b. Insertion sort by aabb_min_x  — O(N²) worst-case but O(N) when
    #     nearly sorted (common for slowly-moving scenes).
    for i in range(1, sap_n):
        var key = sap_idx[i]
        var key_val = aabb_min_x[key]
        var j = i - 1
        while j >= 0 and aabb_min_x[sap_idx[j]] > key_val:
            sap_idx[j + 1] = sap_idx[j]
            j -= 1
        sap_idx[j + 1] = key

    # 4c. Sweep with early-exit on X axis, then check Y and Z.
    for i in range(sap_n):
        var gi = sap_idx[i]
        var gi_max_x = aabb_max_x[gi]
        var gi_type = model.geom_type[gi]
        var gi_body = model.geom_body[gi]
        var gi_contype = model.geom_contype[gi]
        var gi_conaffinity = model.geom_conaffinity[gi]
        var pi_x = wpx[gi]
        var pi_y = wpy[gi]
        var pi_z = wpz[gi]
        var qi_x = wqx[gi]
        var qi_y = wqy[gi]
        var qi_z = wqz[gi]
        var qi_w = wqw[gi]
        var ri = model.geom_radius[gi]
        var hli = model.geom_half_length[gi]
        var hxi = model.geom_half_x[gi]
        var hyi = model.geom_half_y[gi]
        var hzi = model.geom_half_z[gi]

        for j in range(i + 1, sap_n):
            if data.num_contacts >= MAX_CONTACTS:
                return
            var gj = sap_idx[j]

            # X-axis break: all remaining geoms start beyond gi's max_x
            if aabb_min_x[gj] > gi_max_x:
                break

            # Y and Z AABB overlap tests
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

            # AABB overlap confirmed — apply body filter
            var gj_type = model.geom_type[gj]
            var gj_body = model.geom_body[gj]
            if gi_body > 0 and gi_body == gj_body:
                continue
            if gi_body > 0 and gj_body > 0:
                if (
                    model.body_parent[gj_body] == gi_body
                    or model.body_parent[gi_body] == gj_body
                ):
                    continue
            var gj_contype = model.geom_contype[gj]
            var gj_conaffinity = model.geom_conaffinity[gj]
            if (gi_contype & gj_conaffinity) == 0 and (
                gj_contype & gi_conaffinity
            ) == 0:
                continue

            # Friction / condim combination (MuJoCo: max per element)
            var cf = model.geom_friction[gi]
            if model.geom_friction[gj] > cf:
                cf = model.geom_friction[gj]
            var cfs = model.geom_friction_spin[gi]
            if model.geom_friction_spin[gj] > cfs:
                cfs = model.geom_friction_spin[gj]
            var cfr = model.geom_friction_roll[gi]
            if model.geom_friction_roll[gj] > cfr:
                cfr = model.geom_friction_roll[gj]
            var cdim = model.geom_condim[gi]
            if model.geom_condim[gj] > cdim:
                cdim = model.geom_condim[gj]
            # MuJoCo 3.5+: margin = sum of both geom margins
            var cm = model.geom_margin[gi] + model.geom_margin[gj]

            # World positions for gj
            var pj_x = wpx[gj]
            var pj_y = wpy[gj]
            var pj_z = wpz[gj]
            var qj_x = wqx[gj]
            var qj_y = wqy[gj]
            var qj_z = wqz[gj]
            var qj_w = wqw[gj]
            var rj = model.geom_radius[gj]
            var hlj = model.geom_half_length[gj]
            var hxj = model.geom_half_x[gj]
            var hyj = model.geom_half_y[gj]
            var hzj = model.geom_half_z[gj]

            # Narrowphase dispatch (identical to detect_contacts)
            var dist: Scalar[DTYPE] = 1.0
            var cx: Scalar[DTYPE] = 0
            var cy: Scalar[DTYPE] = 0
            var cz: Scalar[DTYPE] = 0
            var nx: Scalar[DTYPE] = 0
            var ny: Scalar[DTYPE] = 0
            var nz: Scalar[DTYPE] = 1
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
                body_a = gj_body
                body_b = gi_body
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
                body_a = gj_body
                body_b = gi_body
            elif gi_type == GEOM_BOX and gj_type == GEOM_CAPSULE:
                var r = box_capsule[DTYPE](
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
            elif gi_type == GEOM_CAPSULE and gj_type == GEOM_BOX:
                var r = box_capsule[DTYPE](
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
                    qi_x,
                    qi_y,
                    qi_z,
                    qi_w,
                    hli,
                    ri,
                )
                dist = r[0]
                cx = r[1]
                cy = r[2]
                cz = r[3]
                nx = -r[4]
                ny = -r[5]
                nz = -r[6]
                body_a = gj_body
                body_b = gi_body
            elif gi_type == GEOM_BOX and gj_type == GEOM_BOX:
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
                body_a = gj_body
                body_b = gi_body

            if dist < cm and data.num_contacts < MAX_CONTACTS:
                var idx = data.num_contacts
                data.contacts[idx].body_a = body_a
                data.contacts[idx].body_b = body_b
                data.contacts[idx].pos_x = cx
                data.contacts[idx].pos_y = cy
                data.contacts[idx].pos_z = cz
                data.contacts[idx].normal_x = nx
                data.contacts[idx].normal_y = ny
                data.contacts[idx].normal_z = nz
                data.contacts[idx].dist = dist - cm
                data.contacts[idx].friction = cf
                data.contacts[idx].friction_spin = cfs
                data.contacts[idx].friction_roll = cfr
                data.contacts[idx].condim = cdim
                data.num_contacts += 1


# =============================================================================
# GPU: detect_contacts_sap_gpu
# =============================================================================


@always_inline
def detect_contacts_sap_gpu[
    DTYPE: DType,
    NQ: Int,
    NV: Int,
    NBODY: Int,
    NJOINT: Int,
    MAX_CONTACTS: Int,
    STATE_SIZE: Int,
    MODEL_SIZE: Int,
    BATCH: Int,
    NGEOM: Int,
](
    env: Int,
    state: LayoutTensor[
        DTYPE, Layout.row_major(BATCH, STATE_SIZE), MutAnyOrigin
    ],
    model: LayoutTensor[DTYPE, Layout.row_major(1, MODEL_SIZE), MutAnyOrigin],
):
    """Contact detection using AABB/SAP broadphase (GPU, one thread per env).

    Drop-in replacement for detect_contacts_gpu. Recommended for scenes with
    50+ geoms.
    """
    var contacts_off = contacts_offset[NQ, NV, NBODY]()
    var meta_off = metadata_offset[NQ, NV, NBODY, MAX_CONTACTS]()
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
        var g_off = model_geom_offset[NBODY, NJOINT](g)
        var px: Scalar[DTYPE] = 0
        var py: Scalar[DTYPE] = 0
        var pz: Scalar[DTYPE] = 0
        var qx: Scalar[DTYPE] = 0
        var qy: Scalar[DTYPE] = 0
        var qz: Scalar[DTYPE] = 0
        var qw: Scalar[DTYPE] = 1
        _geom_world_pos_gpu[
            DTYPE, NQ, NV, NBODY, STATE_SIZE, MODEL_SIZE, BATCH
        ](env, g_off, state, model, px, py, pz, qx, qy, qz, qw)
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
        var g_off = model_geom_offset[NBODY, NJOINT](g)
        var gt = Int(rebind[Scalar[DTYPE]](model[0, g_off + GEOM_IDX_TYPE]))
        if gt == GEOM_PLANE:
            continue
        var r = rebind[Scalar[DTYPE]](model[0, g_off + GEOM_IDX_RADIUS])
        var hl = rebind[Scalar[DTYPE]](model[0, g_off + GEOM_IDX_HALF_LENGTH])
        var hx = rebind[Scalar[DTYPE]](model[0, g_off + GEOM_IDX_HALF_X])
        var hy = rebind[Scalar[DTYPE]](model[0, g_off + GEOM_IDX_HALF_Y])
        var hz = rebind[Scalar[DTYPE]](model[0, g_off + GEOM_IDX_HALF_Z])
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
        var gi_off = model_geom_offset[NBODY, NJOINT](gi)
        var gi_type = Int(
            rebind[Scalar[DTYPE]](model[0, gi_off + GEOM_IDX_TYPE])
        )
        if gi_type != GEOM_PLANE:
            continue
        var gi_body = Int(
            rebind[Scalar[DTYPE]](model[0, gi_off + GEOM_IDX_BODY])
        )
        var gi_contype = Int(
            rebind[Scalar[DTYPE]](model[0, gi_off + GEOM_IDX_CONTYPE])
        )
        var gi_conaffinity = Int(
            rebind[Scalar[DTYPE]](model[0, gi_off + GEOM_IDX_CONAFFINITY])
        )
        var ground_z = wpz[gi]

        for gj in range(NGEOM):
            if num_contacts >= MAX_CONTACTS:
                state[env, meta_off + META_IDX_NUM_CONTACTS] = Scalar[DTYPE](
                    num_contacts
                )
                return
            var gj_off = model_geom_offset[NBODY, NJOINT](gj)
            var gj_type = Int(
                rebind[Scalar[DTYPE]](model[0, gj_off + GEOM_IDX_TYPE])
            )
            if gj_type == GEOM_PLANE:
                continue
            var gj_body = Int(
                rebind[Scalar[DTYPE]](model[0, gj_off + GEOM_IDX_BODY])
            )
            if gj_body == 0:
                continue
            var gj_contype = Int(
                rebind[Scalar[DTYPE]](model[0, gj_off + GEOM_IDX_CONTYPE])
            )
            var gj_conaffinity = Int(
                rebind[Scalar[DTYPE]](model[0, gj_off + GEOM_IDX_CONAFFINITY])
            )
            if (gi_contype & gj_conaffinity) == 0 and (
                gj_contype & gi_conaffinity
            ) == 0:
                continue

            var fi = rebind[Scalar[DTYPE]](model[0, gi_off + GEOM_IDX_FRICTION])
            var fj = rebind[Scalar[DTYPE]](model[0, gj_off + GEOM_IDX_FRICTION])
            var cf = fi if fj <= fi else fj
            var fsi = rebind[Scalar[DTYPE]](
                model[0, gi_off + GEOM_IDX_FRICTION_SPIN]
            )
            var fsj = rebind[Scalar[DTYPE]](
                model[0, gj_off + GEOM_IDX_FRICTION_SPIN]
            )
            var cfs = fsi if fsj <= fsi else fsj
            var fri = rebind[Scalar[DTYPE]](
                model[0, gi_off + GEOM_IDX_FRICTION_ROLL]
            )
            var frj = rebind[Scalar[DTYPE]](
                model[0, gj_off + GEOM_IDX_FRICTION_ROLL]
            )
            var cfr = fri if frj <= fri else frj
            var ci = Int(
                rebind[Scalar[DTYPE]](model[0, gi_off + GEOM_IDX_CONDIM])
            )
            var cj = Int(
                rebind[Scalar[DTYPE]](model[0, gj_off + GEOM_IDX_CONDIM])
            )
            var cdim = ci if cj <= ci else cj
            var mgi = rebind[Scalar[DTYPE]](model[0, gi_off + GEOM_IDX_MARGIN])
            var mgj = rebind[Scalar[DTYPE]](model[0, gj_off + GEOM_IDX_MARGIN])
            var cm = mgi + mgj  # MuJoCo 3.5+: sum of margins

            var pj_x = wpx[gj]
            var pj_y = wpy[gj]
            var pj_z = wpz[gj]
            var qj_x = wqx[gj]
            var qj_y = wqy[gj]
            var qj_z = wqz[gj]
            var qj_w = wqw[gj]
            var rj = rebind[Scalar[DTYPE]](model[0, gj_off + GEOM_IDX_RADIUS])
            var hlj = rebind[Scalar[DTYPE]](
                model[0, gj_off + GEOM_IDX_HALF_LENGTH]
            )

            if gj_type == GEOM_SPHERE:
                var dist = pj_z - rj - ground_z
                if dist < cm and num_contacts < MAX_CONTACTS:
                    var c_off = contacts_off + num_contacts * CONTACT_SIZE
                    state[env, c_off + CONTACT_IDX_BODY_A] = Scalar[DTYPE](
                        gj_body
                    )
                    state[env, c_off + CONTACT_IDX_BODY_B] = Scalar[DTYPE](-1)
                    state[env, c_off + CONTACT_IDX_POS_X] = pj_x
                    state[env, c_off + CONTACT_IDX_POS_Y] = pj_y
                    state[
                        env, c_off + CONTACT_IDX_POS_Z
                    ] = ground_z + dist * Scalar[DTYPE](0.5)
                    state[env, c_off + CONTACT_IDX_NX] = Scalar[DTYPE](0)
                    state[env, c_off + CONTACT_IDX_NY] = Scalar[DTYPE](0)
                    state[env, c_off + CONTACT_IDX_NZ] = Scalar[DTYPE](1)
                    state[env, c_off + CONTACT_IDX_DIST] = dist
                    state[env, c_off + CONTACT_IDX_FRICTION] = cf
                    state[env, c_off + CONTACT_IDX_FRICTION_SPIN] = cfs
                    state[env, c_off + CONTACT_IDX_FRICTION_ROLL] = cfr
                    state[env, c_off + CONTACT_IDX_CONDIM] = Scalar[DTYPE](cdim)
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
                var e1_x = pj_x + hlj * axis_w[0]
                var e1_y = pj_y + hlj * axis_w[1]
                var e1_z = pj_z + hlj * axis_w[2]
                var dist1 = e1_z - rj - ground_z
                if dist1 < cm and num_contacts < MAX_CONTACTS:
                    var c_off = contacts_off + num_contacts * CONTACT_SIZE
                    state[env, c_off + CONTACT_IDX_BODY_A] = Scalar[DTYPE](
                        gj_body
                    )
                    state[env, c_off + CONTACT_IDX_BODY_B] = Scalar[DTYPE](-1)
                    state[env, c_off + CONTACT_IDX_POS_X] = e1_x
                    state[env, c_off + CONTACT_IDX_POS_Y] = e1_y
                    state[
                        env, c_off + CONTACT_IDX_POS_Z
                    ] = ground_z + dist1 * Scalar[DTYPE](0.5)
                    state[env, c_off + CONTACT_IDX_NX] = Scalar[DTYPE](0)
                    state[env, c_off + CONTACT_IDX_NY] = Scalar[DTYPE](0)
                    state[env, c_off + CONTACT_IDX_NZ] = Scalar[DTYPE](1)
                    state[env, c_off + CONTACT_IDX_DIST] = dist1
                    state[env, c_off + CONTACT_IDX_FRICTION] = cf
                    state[env, c_off + CONTACT_IDX_FRICTION_SPIN] = cfs
                    state[env, c_off + CONTACT_IDX_FRICTION_ROLL] = cfr
                    state[env, c_off + CONTACT_IDX_CONDIM] = Scalar[DTYPE](cdim)
                    state[env, c_off + CONTACT_IDX_FRAME_T1_X] = axis_w[0]
                    state[env, c_off + CONTACT_IDX_FRAME_T1_Y] = axis_w[1]
                    state[env, c_off + CONTACT_IDX_FRAME_T1_Z] = axis_w[2]
                    num_contacts += 1
                var e2_x = pj_x - hlj * axis_w[0]
                var e2_y = pj_y - hlj * axis_w[1]
                var e2_z = pj_z - hlj * axis_w[2]
                var dist2 = e2_z - rj - ground_z
                if dist2 < cm and num_contacts < MAX_CONTACTS:
                    var c_off = contacts_off + num_contacts * CONTACT_SIZE
                    state[env, c_off + CONTACT_IDX_BODY_A] = Scalar[DTYPE](
                        gj_body
                    )
                    state[env, c_off + CONTACT_IDX_BODY_B] = Scalar[DTYPE](-1)
                    state[env, c_off + CONTACT_IDX_POS_X] = e2_x
                    state[env, c_off + CONTACT_IDX_POS_Y] = e2_y
                    state[
                        env, c_off + CONTACT_IDX_POS_Z
                    ] = ground_z + dist2 * Scalar[DTYPE](0.5)
                    state[env, c_off + CONTACT_IDX_NX] = Scalar[DTYPE](0)
                    state[env, c_off + CONTACT_IDX_NY] = Scalar[DTYPE](0)
                    state[env, c_off + CONTACT_IDX_NZ] = Scalar[DTYPE](1)
                    state[env, c_off + CONTACT_IDX_DIST] = dist2
                    state[env, c_off + CONTACT_IDX_FRICTION] = cf
                    state[env, c_off + CONTACT_IDX_FRICTION_SPIN] = cfs
                    state[env, c_off + CONTACT_IDX_FRICTION_ROLL] = cfr
                    state[env, c_off + CONTACT_IDX_CONDIM] = Scalar[DTYPE](cdim)
                    state[env, c_off + CONTACT_IDX_FRAME_T1_X] = axis_w[0]
                    state[env, c_off + CONTACT_IDX_FRAME_T1_Y] = axis_w[1]
                    state[env, c_off + CONTACT_IDX_FRAME_T1_Z] = axis_w[2]
                    num_contacts += 1

            elif gj_type == GEOM_CYLINDER:
                var cp = cylinder_plane[DTYPE](
                    pj_x,
                    pj_y,
                    pj_z,
                    qj_x,
                    qj_y,
                    qj_z,
                    qj_w,
                    hlj,
                    rj,
                    ground_z,
                )
                var dist = cp[0]
                if dist < cm and num_contacts < MAX_CONTACTS:
                    var c_off = contacts_off + num_contacts * CONTACT_SIZE
                    state[env, c_off + CONTACT_IDX_BODY_A] = Scalar[DTYPE](
                        gj_body
                    )
                    state[env, c_off + CONTACT_IDX_BODY_B] = Scalar[DTYPE](-1)
                    state[env, c_off + CONTACT_IDX_POS_X] = cp[1]
                    state[env, c_off + CONTACT_IDX_POS_Y] = cp[2]
                    state[env, c_off + CONTACT_IDX_POS_Z] = cp[3]
                    state[env, c_off + CONTACT_IDX_NX] = Scalar[DTYPE](0)
                    state[env, c_off + CONTACT_IDX_NY] = Scalar[DTYPE](0)
                    state[env, c_off + CONTACT_IDX_NZ] = Scalar[DTYPE](1)
                    state[env, c_off + CONTACT_IDX_DIST] = dist
                    state[env, c_off + CONTACT_IDX_FRICTION] = cf
                    state[env, c_off + CONTACT_IDX_FRICTION_SPIN] = cfs
                    state[env, c_off + CONTACT_IDX_FRICTION_ROLL] = cfr
                    state[env, c_off + CONTACT_IDX_CONDIM] = Scalar[DTYPE](cdim)
                    num_contacts += 1

    # ------------------------------------------------------------------
    # 4. SAP sweep for non-plane pairs.
    # ------------------------------------------------------------------

    # 4a. Build SAP index list.
    var sap_idx = InlineArray[Int, NGEOM](uninitialized=True)
    var sap_n = 0
    for g in range(NGEOM):
        var g_off = model_geom_offset[NBODY, NJOINT](g)
        var gt = Int(rebind[Scalar[DTYPE]](model[0, g_off + GEOM_IDX_TYPE]))
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
        var gi_off = model_geom_offset[NBODY, NJOINT](gi)
        var gi_max_x = aabb_max_x[gi]
        var gi_type = Int(
            rebind[Scalar[DTYPE]](model[0, gi_off + GEOM_IDX_TYPE])
        )
        var gi_body = Int(
            rebind[Scalar[DTYPE]](model[0, gi_off + GEOM_IDX_BODY])
        )
        var gi_contype = Int(
            rebind[Scalar[DTYPE]](model[0, gi_off + GEOM_IDX_CONTYPE])
        )
        var gi_conaffinity = Int(
            rebind[Scalar[DTYPE]](model[0, gi_off + GEOM_IDX_CONAFFINITY])
        )
        var pi_x = wpx[gi]
        var pi_y = wpy[gi]
        var pi_z = wpz[gi]
        var qi_x = wqx[gi]
        var qi_y = wqy[gi]
        var qi_z = wqz[gi]
        var qi_w = wqw[gi]
        var ri = rebind[Scalar[DTYPE]](model[0, gi_off + GEOM_IDX_RADIUS])
        var hli = rebind[Scalar[DTYPE]](model[0, gi_off + GEOM_IDX_HALF_LENGTH])
        var hxi = rebind[Scalar[DTYPE]](model[0, gi_off + GEOM_IDX_HALF_X])
        var hyi = rebind[Scalar[DTYPE]](model[0, gi_off + GEOM_IDX_HALF_Y])
        var hzi = rebind[Scalar[DTYPE]](model[0, gi_off + GEOM_IDX_HALF_Z])

        for j in range(i + 1, sap_n):
            if num_contacts >= MAX_CONTACTS:
                state[env, meta_off + META_IDX_NUM_CONTACTS] = Scalar[DTYPE](
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

            var gj_off = model_geom_offset[NBODY, NJOINT](gj)
            var gj_type = Int(
                rebind[Scalar[DTYPE]](model[0, gj_off + GEOM_IDX_TYPE])
            )
            var gj_body = Int(
                rebind[Scalar[DTYPE]](model[0, gj_off + GEOM_IDX_BODY])
            )
            if gi_body > 0 and gi_body == gj_body:
                continue
            if gi_body > 0 and gj_body > 0:
                var bi_off = model_body_offset(gi_body)
                var bj_off = model_body_offset(gj_body)
                var pi = Int(
                    rebind[Scalar[DTYPE]](model[0, bi_off + BODY_IDX_PARENT])
                )
                var pj = Int(
                    rebind[Scalar[DTYPE]](model[0, bj_off + BODY_IDX_PARENT])
                )
                if pj == gi_body or pi == gj_body:
                    continue
            var gj_contype = Int(
                rebind[Scalar[DTYPE]](model[0, gj_off + GEOM_IDX_CONTYPE])
            )
            var gj_conaffinity = Int(
                rebind[Scalar[DTYPE]](model[0, gj_off + GEOM_IDX_CONAFFINITY])
            )
            if (gi_contype & gj_conaffinity) == 0 and (
                gj_contype & gi_conaffinity
            ) == 0:
                continue

            var fi = rebind[Scalar[DTYPE]](model[0, gi_off + GEOM_IDX_FRICTION])
            var fj = rebind[Scalar[DTYPE]](model[0, gj_off + GEOM_IDX_FRICTION])
            var cf = fi if fj <= fi else fj
            var fsi = rebind[Scalar[DTYPE]](
                model[0, gi_off + GEOM_IDX_FRICTION_SPIN]
            )
            var fsj = rebind[Scalar[DTYPE]](
                model[0, gj_off + GEOM_IDX_FRICTION_SPIN]
            )
            var cfs = fsi if fsj <= fsi else fsj
            var fri = rebind[Scalar[DTYPE]](
                model[0, gi_off + GEOM_IDX_FRICTION_ROLL]
            )
            var frj = rebind[Scalar[DTYPE]](
                model[0, gj_off + GEOM_IDX_FRICTION_ROLL]
            )
            var cfr = fri if frj <= fri else frj
            var ci = Int(
                rebind[Scalar[DTYPE]](model[0, gi_off + GEOM_IDX_CONDIM])
            )
            var cj_dim = Int(
                rebind[Scalar[DTYPE]](model[0, gj_off + GEOM_IDX_CONDIM])
            )
            var cdim = ci if cj_dim <= ci else cj_dim
            var mgi = rebind[Scalar[DTYPE]](model[0, gi_off + GEOM_IDX_MARGIN])
            var mgj = rebind[Scalar[DTYPE]](model[0, gj_off + GEOM_IDX_MARGIN])
            var cm = mgi + mgj  # MuJoCo 3.5+: sum of margins

            var pj_x = wpx[gj]
            var pj_y = wpy[gj]
            var pj_z = wpz[gj]
            var qj_x = wqx[gj]
            var qj_y = wqy[gj]
            var qj_z = wqz[gj]
            var qj_w = wqw[gj]
            var rj = rebind[Scalar[DTYPE]](model[0, gj_off + GEOM_IDX_RADIUS])
            var hlj = rebind[Scalar[DTYPE]](
                model[0, gj_off + GEOM_IDX_HALF_LENGTH]
            )
            var hxj = rebind[Scalar[DTYPE]](model[0, gj_off + GEOM_IDX_HALF_X])
            var hyj = rebind[Scalar[DTYPE]](model[0, gj_off + GEOM_IDX_HALF_Y])
            var hzj = rebind[Scalar[DTYPE]](model[0, gj_off + GEOM_IDX_HALF_Z])

            var dist: Scalar[DTYPE] = 1.0
            var cx: Scalar[DTYPE] = 0
            var cy: Scalar[DTYPE] = 0
            var cz: Scalar[DTYPE] = 0
            var nx: Scalar[DTYPE] = 0
            var ny: Scalar[DTYPE] = 0
            var nz: Scalar[DTYPE] = 1
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
                body_a = gj_body
                body_b = gi_body
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
                body_a = gj_body
                body_b = gi_body
            elif gi_type == GEOM_BOX and gj_type == GEOM_CAPSULE:
                var r = box_capsule[DTYPE](
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
            elif gi_type == GEOM_CAPSULE and gj_type == GEOM_BOX:
                var r = box_capsule[DTYPE](
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
                    qi_x,
                    qi_y,
                    qi_z,
                    qi_w,
                    hli,
                    ri,
                )
                dist = r[0]
                cx = r[1]
                cy = r[2]
                cz = r[3]
                nx = -r[4]
                ny = -r[5]
                nz = -r[6]
                body_a = gj_body
                body_b = gi_body
            elif gi_type == GEOM_BOX and gj_type == GEOM_BOX:
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
                body_a = gj_body
                body_b = gi_body

            if dist < cm and num_contacts < MAX_CONTACTS:
                var c_off = contacts_off + num_contacts * CONTACT_SIZE
                state[env, c_off + CONTACT_IDX_BODY_A] = Scalar[DTYPE](body_a)
                state[env, c_off + CONTACT_IDX_BODY_B] = Scalar[DTYPE](body_b)
                state[env, c_off + CONTACT_IDX_POS_X] = cx
                state[env, c_off + CONTACT_IDX_POS_Y] = cy
                state[env, c_off + CONTACT_IDX_POS_Z] = cz
                state[env, c_off + CONTACT_IDX_NX] = nx
                state[env, c_off + CONTACT_IDX_NY] = ny
                state[env, c_off + CONTACT_IDX_NZ] = nz
                state[env, c_off + CONTACT_IDX_DIST] = dist
                state[env, c_off + CONTACT_IDX_FRICTION] = cf
                state[env, c_off + CONTACT_IDX_FRICTION_SPIN] = cfs
                state[env, c_off + CONTACT_IDX_FRICTION_ROLL] = cfr
                state[env, c_off + CONTACT_IDX_CONDIM] = Scalar[DTYPE](cdim)
                num_contacts += 1

    state[env, meta_off + META_IDX_NUM_CONTACTS] = Scalar[DTYPE](num_contacts)


# =============================================================================
# Compile-time auto-dispatch
# =============================================================================

# Geom count at which the SAP broadphase becomes faster than the O(N²) bounding
# sphere check.  Below this threshold detect_contacts is used; at or above it
# detect_contacts_sap is used.  Both branches are dead-code-eliminated at
# compile time via comptime if, so there is zero runtime overhead.
comptime SAP_THRESHOLD: Int = 16


def detect_contacts_auto[
    DTYPE: DType,
    NQ: Int,
    NV: Int,
    NBODY: Int,
    NJOINT: Int,
    MAX_CONTACTS: Int,
    NGEOM: Int,
    MAX_EQUALITY: Int = 0,
    CONE_TYPE: Int = ConeType.ELLIPTIC,
    MAX_TENDON: Int = 0,
    NSITE: Int = 0,
](
    model: Model[
        DTYPE,
        NQ,
        NV,
        NBODY,
        NJOINT,
        MAX_CONTACTS,
        NGEOM,
        MAX_EQUALITY,
        CONE_TYPE,
        MAX_TENDON,
        NSITE,
    ],
    mut data: Data[DTYPE, NQ, NV, NBODY, NJOINT, MAX_CONTACTS, NSITE],
):
    """Contact detection with automatic broadphase selection (CPU).

    Uses detect_contacts_sap when NGEOM >= SAP_THRESHOLD (default 16),
    otherwise falls back to detect_contacts.  The branch is resolved at
    compile time — only one code path is emitted per instantiation.
    """

    comptime if NGEOM >= SAP_THRESHOLD:
        detect_contacts_sap[
            DTYPE,
            NQ,
            NV,
            NBODY,
            NJOINT,
            MAX_CONTACTS,
            NGEOM,
            MAX_EQUALITY,
            CONE_TYPE,
            MAX_TENDON,
            NSITE,
        ](model, data)
    else:
        detect_contacts[
            DTYPE,
            NQ,
            NV,
            NBODY,
            NJOINT,
            MAX_CONTACTS,
            NGEOM,
            MAX_EQUALITY,
            CONE_TYPE,
            MAX_TENDON,
            NSITE,
        ](model, data)


@always_inline
def detect_contacts_auto_gpu[
    DTYPE: DType,
    NQ: Int,
    NV: Int,
    NBODY: Int,
    NJOINT: Int,
    MAX_CONTACTS: Int,
    STATE_SIZE: Int,
    MODEL_SIZE: Int,
    BATCH: Int,
    NGEOM: Int,
](
    env: Int,
    state: LayoutTensor[
        DTYPE, Layout.row_major(BATCH, STATE_SIZE), MutAnyOrigin
    ],
    model: LayoutTensor[DTYPE, Layout.row_major(1, MODEL_SIZE), MutAnyOrigin],
):
    """Contact detection with automatic broadphase selection (GPU).

    Uses detect_contacts_sap_gpu when NGEOM >= SAP_THRESHOLD (default 16),
    otherwise falls back to detect_contacts_gpu.  The branch is resolved at
    compile time.
    """

    comptime if NGEOM >= SAP_THRESHOLD:
        detect_contacts_sap_gpu[
            DTYPE,
            NQ,
            NV,
            NBODY,
            NJOINT,
            MAX_CONTACTS,
            STATE_SIZE,
            MODEL_SIZE,
            BATCH,
            NGEOM,
        ](env, state, model)
    else:
        detect_contacts_gpu[
            DTYPE,
            NQ,
            NV,
            NBODY,
            NJOINT,
            MAX_CONTACTS,
            STATE_SIZE,
            MODEL_SIZE,
            BATCH,
            NGEOM,
        ](env, state, model)
