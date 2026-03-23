"""Contact detection for physics engine.

Provides unified geom-based contact detection and quaternion normalization.
"""

from std.math import sqrt
from layout import LayoutTensor, Layout
from ..types import (
    Model,
    Data,
    ConeType,
)
from ..joint_types import JNT_HINGE, JNT_SLIDE, JNT_BALL, JNT_FREE

from ..kinematics.quat_math import (
    quat_normalize,
    quat_rotate,
    quat_mul,
    gpu_quat_rotate,
    gpu_quat_normalize,
    gpu_quat_mul,
)
from ..constants import (
    GEOM_SPHERE,
    GEOM_CAPSULE,
    GEOM_BOX,
    GEOM_PLANE,
    GEOM_CYLINDER,
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
    JOINT_IDX_TYPE,
    JOINT_IDX_QPOS_ADR,
    META_IDX_NUM_CONTACTS,
    MODEL_META_IDX_NJOINT,
    MODEL_GEOM_SIZE,
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
    GEOM_IDX_FRICTION_SPIN,
    GEOM_IDX_FRICTION_ROLL,
    GEOM_IDX_RBOUND,
    GEOM_IDX_MARGIN,
    model_body_offset,
    model_joint_offset,
    model_geom_offset,
    model_metadata_offset,
    qpos_offset,
    xpos_offset,
    xquat_offset,
    contacts_offset,
    metadata_offset,
)
from ..collision.collision_primitives import (
    sphere_sphere,
    capsule_sphere,
    capsule_capsule,
    box_sphere,
    box_capsule,
    box_box,
    cylinder_plane,
    cylinder_sphere,
)


def normalize_qpos_quaternions[
    DTYPE: DType,
    NQ: Int,
    NV: Int,
    NBODY: Int,
    NJOINT: Int,
    MAX_CONTACTS: Int,
    NGEOM: Int = 0,
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
    """Normalize quaternions in qpos for BALL and FREE joints."""
    for j in range(model.num_joints):
        var joint = model.joints[j]
        var qpos_adr = joint.qpos_adr

        if joint.jnt_type == JNT_FREE:
            var qx = data.qpos[qpos_adr + 3]
            var qy = data.qpos[qpos_adr + 4]
            var qz = data.qpos[qpos_adr + 5]
            var qw = data.qpos[qpos_adr + 6]

            var normalized = quat_normalize(qx, qy, qz, qw)
            data.qpos[qpos_adr + 3] = normalized[0]
            data.qpos[qpos_adr + 4] = normalized[1]
            data.qpos[qpos_adr + 5] = normalized[2]
            data.qpos[qpos_adr + 6] = normalized[3]

        elif joint.jnt_type == JNT_BALL:
            var qx = data.qpos[qpos_adr + 0]
            var qy = data.qpos[qpos_adr + 1]
            var qz = data.qpos[qpos_adr + 2]
            var qw = data.qpos[qpos_adr + 3]

            var normalized = quat_normalize(qx, qy, qz, qw)
            data.qpos[qpos_adr + 0] = normalized[0]
            data.qpos[qpos_adr + 1] = normalized[1]
            data.qpos[qpos_adr + 2] = normalized[2]
            data.qpos[qpos_adr + 3] = normalized[3]


@always_inline
def normalize_qpos_quaternions_gpu[
    DTYPE: DType,
    NQ: Int,
    NV: Int,
    NBODY: Int,
    NJOINT: Int,
    MAX_CONTACTS: Int,
    STATE_SIZE: Int,
    MODEL_SIZE: Int,
    BATCH: Int,
](
    env: Int,
    state: LayoutTensor[
        DTYPE, Layout.row_major(BATCH, STATE_SIZE), MutAnyOrigin
    ],
    model: LayoutTensor[DTYPE, Layout.row_major(1, MODEL_SIZE), MutAnyOrigin],
):
    """Normalize quaternions in qpos for BALL and FREE joints."""
    var qpos_off = qpos_offset[NQ, NV]()

    var model_meta_off = model_metadata_offset[NBODY, NJOINT]()
    var num_joints = Int(
        rebind[Scalar[DTYPE]](model[0, model_meta_off + MODEL_META_IDX_NJOINT])
    )

    for j in range(num_joints):
        var joint_off = model_joint_offset[NBODY](j)
        var jnt_type = Int(
            rebind[Scalar[DTYPE]](model[0, joint_off + JOINT_IDX_TYPE])
        )
        var qpos_adr = Int(
            rebind[Scalar[DTYPE]](model[0, joint_off + JOINT_IDX_QPOS_ADR])
        )

        if jnt_type == JNT_FREE:
            var qx = rebind[Scalar[DTYPE]](state[env, qpos_off + qpos_adr + 3])
            var qy = rebind[Scalar[DTYPE]](state[env, qpos_off + qpos_adr + 4])
            var qz = rebind[Scalar[DTYPE]](state[env, qpos_off + qpos_adr + 5])
            var qw = rebind[Scalar[DTYPE]](state[env, qpos_off + qpos_adr + 6])

            var normalized = gpu_quat_normalize(qx, qy, qz, qw)
            state[env, qpos_off + qpos_adr + 3] = normalized[0]
            state[env, qpos_off + qpos_adr + 4] = normalized[1]
            state[env, qpos_off + qpos_adr + 5] = normalized[2]
            state[env, qpos_off + qpos_adr + 6] = normalized[3]

        elif jnt_type == JNT_BALL:
            var qx = rebind[Scalar[DTYPE]](state[env, qpos_off + qpos_adr + 0])
            var qy = rebind[Scalar[DTYPE]](state[env, qpos_off + qpos_adr + 1])
            var qz = rebind[Scalar[DTYPE]](state[env, qpos_off + qpos_adr + 2])
            var qw = rebind[Scalar[DTYPE]](state[env, qpos_off + qpos_adr + 3])

            var normalized = gpu_quat_normalize(qx, qy, qz, qw)
            state[env, qpos_off + qpos_adr + 0] = normalized[0]
            state[env, qpos_off + qpos_adr + 1] = normalized[1]
            state[env, qpos_off + qpos_adr + 2] = normalized[2]
            state[env, qpos_off + qpos_adr + 3] = normalized[3]


# =============================================================================
# Unified Contact Detection (CPU)
# =============================================================================


def _geom_world_pos[
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
    data: Data[DTYPE, NQ, NV, NBODY, NJOINT, MAX_CONTACTS, NSITE],
    g: Int,
) -> Tuple[
    Scalar[DTYPE],
    Scalar[DTYPE],
    Scalar[DTYPE],
    Scalar[DTYPE],
    Scalar[DTYPE],
    Scalar[DTYPE],
    Scalar[DTYPE],
]:
    """Compute world position and orientation for geom g."""
    var body_idx = model.geom_body[g]
    var lx = model.geom_pos[g * 3 + 0]
    var ly = model.geom_pos[g * 3 + 1]
    var lz = model.geom_pos[g * 3 + 2]
    var lqx = model.geom_quat[g * 4 + 0]
    var lqy = model.geom_quat[g * 4 + 1]
    var lqz = model.geom_quat[g * 4 + 2]
    var lqw = model.geom_quat[g * 4 + 3]

    if body_idx == 0:
        return (lx, ly, lz, lqx, lqy, lqz, lqw)

    var bpx = data.xpos[body_idx * 3 + 0]
    var bpy = data.xpos[body_idx * 3 + 1]
    var bpz = data.xpos[body_idx * 3 + 2]
    var bqx = data.xquat[body_idx * 4 + 0]
    var bqy = data.xquat[body_idx * 4 + 1]
    var bqz = data.xquat[body_idx * 4 + 2]
    var bqw = data.xquat[body_idx * 4 + 3]

    var is_identity = (
        lx == Scalar[DTYPE](0)
        and ly == Scalar[DTYPE](0)
        and lz == Scalar[DTYPE](0)
        and lqx == Scalar[DTYPE](0)
        and lqy == Scalar[DTYPE](0)
        and lqz == Scalar[DTYPE](0)
        and lqw == Scalar[DTYPE](1)
    )
    if is_identity:
        return (bpx, bpy, bpz, bqx, bqy, bqz, bqw)

    var rotated = quat_rotate(bqx, bqy, bqz, bqw, lx, ly, lz)
    var wpx = bpx + rotated[0]
    var wpy = bpy + rotated[1]
    var wpz = bpz + rotated[2]
    var wq = quat_mul(bqx, bqy, bqz, bqw, lqx, lqy, lqz, lqw)
    return (wpx, wpy, wpz, wq[0], wq[1], wq[2], wq[3])


def detect_contacts[
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
    """Unified contact detection using geom arrays."""
    data.num_contacts = 0
    for gi in range(NGEOM):
        var gi_type = model.geom_type[gi]
        var gi_body = model.geom_body[gi]
        var gi_contype = model.geom_contype[gi]
        var gi_conaffinity = model.geom_conaffinity[gi]
        for gj in range(gi + 1, NGEOM):
            if data.num_contacts >= MAX_CONTACTS:
                return
            var gj_type = model.geom_type[gj]
            var gj_body = model.geom_body[gj]
            if gi_type == GEOM_PLANE and gj_body == 0:
                continue
            if gj_type == GEOM_PLANE and gi_body == 0:
                continue
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

            var wi = _geom_world_pos(model, data, gi)
            var wj = _geom_world_pos(model, data, gj)
            var pi_x = wi[0]
            var pi_y = wi[1]
            var pi_z = wi[2]
            var qi_x = wi[3]
            var qi_y = wi[4]
            var qi_z = wi[5]
            var qi_w = wi[6]
            var pj_x = wj[0]
            var pj_y = wj[1]
            var pj_z = wj[2]
            var qj_x = wj[3]
            var qj_y = wj[4]
            var qj_z = wj[5]
            var qj_w = wj[6]

            # Broadphase bounding sphere check (skip for plane geoms — they're infinite)
            if gi_type != GEOM_PLANE and gj_type != GEOM_PLANE:
                var dx = pi_x - pj_x
                var dy = pi_y - pj_y
                var dz = pi_z - pj_z
                var dist_sq = dx * dx + dy * dy + dz * dz
                var bound = model.geom_rbound[gi] + model.geom_rbound[gj]
                if dist_sq > bound * bound:
                    continue

            var ri = model.geom_radius[gi]
            var rj = model.geom_radius[gj]
            var hli = model.geom_half_length[gi]
            var hlj = model.geom_half_length[gj]
            var hxi = model.geom_half_x[gi]
            var hyi = model.geom_half_y[gi]
            var hzi = model.geom_half_z[gi]
            var hxj = model.geom_half_x[gj]
            var hyj = model.geom_half_y[gj]
            var hzj = model.geom_half_z[gj]
            # Friction combination: max per element (MuJoCo convention)
            var contact_friction = model.geom_friction[gi]
            if model.geom_friction[gj] > contact_friction:
                contact_friction = model.geom_friction[gj]
            var contact_friction_spin = model.geom_friction_spin[gi]
            if model.geom_friction_spin[gj] > contact_friction_spin:
                contact_friction_spin = model.geom_friction_spin[gj]
            var contact_friction_roll = model.geom_friction_roll[gi]
            if model.geom_friction_roll[gj] > contact_friction_roll:
                contact_friction_roll = model.geom_friction_roll[gj]
            # Condim: max of both geoms
            var contact_condim = model.geom_condim[gi]
            if model.geom_condim[gj] > contact_condim:
                contact_condim = model.geom_condim[gj]
            # Contact margin: max of both geoms (MuJoCo convention)
            var contact_margin = model.geom_margin[gi]
            if model.geom_margin[gj] > contact_margin:
                contact_margin = model.geom_margin[gj]

            # --- Plane vs body-attached geom ---
            if gi_type == GEOM_PLANE:
                var ground_z = pi_z
                if gj_type == GEOM_CAPSULE:
                    # MuJoCo mjc_PlaneCapsule: test BOTH endpoints, up to 2 contacts
                    var axis_w = quat_rotate(
                        qj_x,
                        qj_y,
                        qj_z,
                        qj_w,
                        Scalar[DTYPE](0),
                        Scalar[DTYPE](0),
                        Scalar[DTYPE](1),
                    )
                    # Endpoint 1: center + half_length * axis
                    var e1_x = pj_x + hlj * axis_w[0]
                    var e1_y = pj_y + hlj * axis_w[1]
                    var e1_z = pj_z + hlj * axis_w[2]
                    var dist1 = e1_z - rj - ground_z
                    if (
                        dist1 < contact_margin
                        and data.num_contacts < MAX_CONTACTS
                    ):
                        var idx = data.num_contacts
                        data.contacts[idx].body_a = gj_body
                        data.contacts[idx].body_b = 0
                        data.contacts[idx].pos_x = e1_x
                        data.contacts[idx].pos_y = e1_y
                        data.contacts[idx].pos_z = ground_z + dist1 * Scalar[
                            DTYPE
                        ](0.5)
                        data.contacts[idx].normal_x = Scalar[DTYPE](0)
                        data.contacts[idx].normal_y = Scalar[DTYPE](0)
                        data.contacts[idx].normal_z = Scalar[DTYPE](1)
                        data.contacts[idx].dist = dist1 - contact_margin
                        data.contacts[idx].friction = contact_friction
                        data.contacts[idx].friction_spin = contact_friction_spin
                        data.contacts[idx].friction_roll = contact_friction_roll
                        data.contacts[idx].condim = contact_condim
                        data.contacts[idx].frame_t1_x = axis_w[0]
                        data.contacts[idx].frame_t1_y = axis_w[1]
                        data.contacts[idx].frame_t1_z = axis_w[2]
                        data.num_contacts += 1
                    # Endpoint 2: center - half_length * axis
                    var e2_x = pj_x - hlj * axis_w[0]
                    var e2_y = pj_y - hlj * axis_w[1]
                    var e2_z = pj_z - hlj * axis_w[2]
                    var dist2 = e2_z - rj - ground_z
                    if (
                        dist2 < contact_margin
                        and data.num_contacts < MAX_CONTACTS
                    ):
                        var idx = data.num_contacts
                        data.contacts[idx].body_a = gj_body
                        data.contacts[idx].body_b = 0
                        data.contacts[idx].pos_x = e2_x
                        data.contacts[idx].pos_y = e2_y
                        data.contacts[idx].pos_z = ground_z + dist2 * Scalar[
                            DTYPE
                        ](0.5)
                        data.contacts[idx].normal_x = Scalar[DTYPE](0)
                        data.contacts[idx].normal_y = Scalar[DTYPE](0)
                        data.contacts[idx].normal_z = Scalar[DTYPE](1)
                        data.contacts[idx].dist = dist2 - contact_margin
                        data.contacts[idx].friction = contact_friction
                        data.contacts[idx].friction_spin = contact_friction_spin
                        data.contacts[idx].friction_roll = contact_friction_roll
                        data.contacts[idx].condim = contact_condim
                        data.contacts[idx].frame_t1_x = axis_w[0]
                        data.contacts[idx].frame_t1_y = axis_w[1]
                        data.contacts[idx].frame_t1_z = axis_w[2]
                        data.num_contacts += 1
                elif gj_type == GEOM_CYLINDER:
                    # Cylinder-plane: single contact at lowest rim point
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
                    if (
                        dist < contact_margin
                        and data.num_contacts < MAX_CONTACTS
                    ):
                        var idx = data.num_contacts
                        data.contacts[idx].body_a = gj_body
                        data.contacts[idx].body_b = 0
                        data.contacts[idx].pos_x = cp[1]
                        data.contacts[idx].pos_y = cp[2]
                        data.contacts[idx].pos_z = cp[3]
                        data.contacts[idx].normal_x = Scalar[DTYPE](0)
                        data.contacts[idx].normal_y = Scalar[DTYPE](0)
                        data.contacts[idx].normal_z = Scalar[DTYPE](1)
                        data.contacts[idx].dist = dist - contact_margin
                        data.contacts[idx].friction = contact_friction
                        data.contacts[idx].friction_spin = contact_friction_spin
                        data.contacts[idx].friction_roll = contact_friction_roll
                        data.contacts[idx].condim = contact_condim
                        data.num_contacts += 1
                elif gj_type == GEOM_SPHERE:
                    var dist = pj_z - rj - ground_z
                    if (
                        dist < contact_margin
                        and data.num_contacts < MAX_CONTACTS
                    ):
                        var idx = data.num_contacts
                        data.contacts[idx].body_a = gj_body
                        data.contacts[idx].body_b = 0
                        data.contacts[idx].pos_x = pj_x
                        data.contacts[idx].pos_y = pj_y
                        data.contacts[idx].pos_z = ground_z + dist * Scalar[
                            DTYPE
                        ](0.5)
                        data.contacts[idx].normal_x = Scalar[DTYPE](0)
                        data.contacts[idx].normal_y = Scalar[DTYPE](0)
                        data.contacts[idx].normal_z = Scalar[DTYPE](1)
                        data.contacts[idx].dist = dist - contact_margin
                        data.contacts[idx].friction = contact_friction
                        data.contacts[idx].friction_spin = contact_friction_spin
                        data.contacts[idx].friction_roll = contact_friction_roll
                        data.contacts[idx].condim = contact_condim
                        data.num_contacts += 1
                continue

            if gj_type == GEOM_PLANE:
                var ground_z = pj_z
                if gi_type == GEOM_CAPSULE:
                    # MuJoCo mjc_PlaneCapsule: test BOTH endpoints, up to 2 contacts
                    var axis_w = quat_rotate(
                        qi_x,
                        qi_y,
                        qi_z,
                        qi_w,
                        Scalar[DTYPE](0),
                        Scalar[DTYPE](0),
                        Scalar[DTYPE](1),
                    )
                    # Endpoint 1
                    var e1_x = pi_x + hli * axis_w[0]
                    var e1_y = pi_y + hli * axis_w[1]
                    var e1_z = pi_z + hli * axis_w[2]
                    var dist1 = e1_z - ri - ground_z
                    if (
                        dist1 < contact_margin
                        and data.num_contacts < MAX_CONTACTS
                    ):
                        var idx = data.num_contacts
                        data.contacts[idx].body_a = gi_body
                        data.contacts[idx].body_b = 0
                        data.contacts[idx].pos_x = e1_x
                        data.contacts[idx].pos_y = e1_y
                        data.contacts[idx].pos_z = ground_z + dist1 * Scalar[
                            DTYPE
                        ](0.5)
                        data.contacts[idx].normal_x = Scalar[DTYPE](0)
                        data.contacts[idx].normal_y = Scalar[DTYPE](0)
                        data.contacts[idx].normal_z = Scalar[DTYPE](1)
                        data.contacts[idx].dist = dist1 - contact_margin
                        data.contacts[idx].friction = contact_friction
                        data.contacts[idx].friction_spin = contact_friction_spin
                        data.contacts[idx].friction_roll = contact_friction_roll
                        data.contacts[idx].condim = contact_condim
                        data.contacts[idx].frame_t1_x = axis_w[0]
                        data.contacts[idx].frame_t1_y = axis_w[1]
                        data.contacts[idx].frame_t1_z = axis_w[2]
                        data.num_contacts += 1
                    # Endpoint 2
                    var e2_x = pi_x - hli * axis_w[0]
                    var e2_y = pi_y - hli * axis_w[1]
                    var e2_z = pi_z - hli * axis_w[2]
                    var dist2 = e2_z - ri - ground_z
                    if (
                        dist2 < contact_margin
                        and data.num_contacts < MAX_CONTACTS
                    ):
                        var idx = data.num_contacts
                        data.contacts[idx].body_a = gi_body
                        data.contacts[idx].body_b = 0
                        data.contacts[idx].pos_x = e2_x
                        data.contacts[idx].pos_y = e2_y
                        data.contacts[idx].pos_z = ground_z + dist2 * Scalar[
                            DTYPE
                        ](0.5)
                        data.contacts[idx].normal_x = Scalar[DTYPE](0)
                        data.contacts[idx].normal_y = Scalar[DTYPE](0)
                        data.contacts[idx].normal_z = Scalar[DTYPE](1)
                        data.contacts[idx].dist = dist2 - contact_margin
                        data.contacts[idx].friction = contact_friction
                        data.contacts[idx].friction_spin = contact_friction_spin
                        data.contacts[idx].friction_roll = contact_friction_roll
                        data.contacts[idx].condim = contact_condim
                        data.contacts[idx].frame_t1_x = axis_w[0]
                        data.contacts[idx].frame_t1_y = axis_w[1]
                        data.contacts[idx].frame_t1_z = axis_w[2]
                        data.num_contacts += 1
                elif gi_type == GEOM_CYLINDER:
                    # Cylinder-plane: single contact at lowest rim point
                    var cp = cylinder_plane[DTYPE](
                        pi_x,
                        pi_y,
                        pi_z,
                        qi_x,
                        qi_y,
                        qi_z,
                        qi_w,
                        hli,
                        ri,
                        ground_z,
                    )
                    var dist = cp[0]
                    if (
                        dist < contact_margin
                        and data.num_contacts < MAX_CONTACTS
                    ):
                        var idx = data.num_contacts
                        data.contacts[idx].body_a = gi_body
                        data.contacts[idx].body_b = 0
                        data.contacts[idx].pos_x = cp[1]
                        data.contacts[idx].pos_y = cp[2]
                        data.contacts[idx].pos_z = cp[3]
                        data.contacts[idx].normal_x = Scalar[DTYPE](0)
                        data.contacts[idx].normal_y = Scalar[DTYPE](0)
                        data.contacts[idx].normal_z = Scalar[DTYPE](1)
                        data.contacts[idx].dist = dist - contact_margin
                        data.contacts[idx].friction = contact_friction
                        data.contacts[idx].friction_spin = contact_friction_spin
                        data.contacts[idx].friction_roll = contact_friction_roll
                        data.contacts[idx].condim = contact_condim
                        data.num_contacts += 1
                elif gi_type == GEOM_SPHERE:
                    var dist = pi_z - ri - ground_z
                    if (
                        dist < contact_margin
                        and data.num_contacts < MAX_CONTACTS
                    ):
                        var idx = data.num_contacts
                        data.contacts[idx].body_a = gi_body
                        data.contacts[idx].body_b = 0
                        data.contacts[idx].pos_x = pi_x
                        data.contacts[idx].pos_y = pi_y
                        data.contacts[idx].pos_z = ground_z + dist * Scalar[
                            DTYPE
                        ](0.5)
                        data.contacts[idx].normal_x = Scalar[DTYPE](0)
                        data.contacts[idx].normal_y = Scalar[DTYPE](0)
                        data.contacts[idx].normal_z = Scalar[DTYPE](1)
                        data.contacts[idx].dist = dist - contact_margin
                        data.contacts[idx].friction = contact_friction
                        data.contacts[idx].friction_spin = contact_friction_spin
                        data.contacts[idx].friction_roll = contact_friction_roll
                        data.contacts[idx].condim = contact_condim
                        data.num_contacts += 1
                continue

            # --- Non-plane geom pair ---
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

            if dist < contact_margin and data.num_contacts < MAX_CONTACTS:
                var idx = data.num_contacts
                data.contacts[idx].body_a = body_a
                data.contacts[idx].body_b = body_b
                data.contacts[idx].pos_x = cx
                data.contacts[idx].pos_y = cy
                data.contacts[idx].pos_z = cz
                data.contacts[idx].normal_x = nx
                data.contacts[idx].normal_y = ny
                data.contacts[idx].normal_z = nz
                data.contacts[idx].dist = dist - contact_margin
                data.contacts[idx].friction = contact_friction
                data.contacts[idx].friction_spin = contact_friction_spin
                data.contacts[idx].friction_roll = contact_friction_roll
                data.contacts[idx].condim = contact_condim
                data.num_contacts += 1


# =============================================================================
# Unified Contact Detection (GPU)
# =============================================================================


@always_inline
def _geom_world_pos_gpu[
    DTYPE: DType,
    NQ: Int,
    NV: Int,
    NBODY: Int,
    STATE_SIZE: Int,
    MODEL_SIZE: Int,
    BATCH: Int,
](
    env: Int,
    g_off: Int,
    state: LayoutTensor[
        DTYPE, Layout.row_major(BATCH, STATE_SIZE), MutAnyOrigin
    ],
    model: LayoutTensor[DTYPE, Layout.row_major(1, MODEL_SIZE), MutAnyOrigin],
    mut out_px: Scalar[DTYPE],
    mut out_py: Scalar[DTYPE],
    mut out_pz: Scalar[DTYPE],
    mut out_qx: Scalar[DTYPE],
    mut out_qy: Scalar[DTYPE],
    mut out_qz: Scalar[DTYPE],
    mut out_qw: Scalar[DTYPE],
):
    """Compute geom world pos/quat on GPU."""
    var body_idx = Int(rebind[Scalar[DTYPE]](model[0, g_off + GEOM_IDX_BODY]))
    var lx = rebind[Scalar[DTYPE]](model[0, g_off + GEOM_IDX_POS_X])
    var ly = rebind[Scalar[DTYPE]](model[0, g_off + GEOM_IDX_POS_Y])
    var lz = rebind[Scalar[DTYPE]](model[0, g_off + GEOM_IDX_POS_Z])
    var lqx = rebind[Scalar[DTYPE]](model[0, g_off + GEOM_IDX_QUAT_X])
    var lqy = rebind[Scalar[DTYPE]](model[0, g_off + GEOM_IDX_QUAT_Y])
    var lqz = rebind[Scalar[DTYPE]](model[0, g_off + GEOM_IDX_QUAT_Z])
    var lqw = rebind[Scalar[DTYPE]](model[0, g_off + GEOM_IDX_QUAT_W])
    if body_idx == 0:
        out_px = lx
        out_py = ly
        out_pz = lz
        out_qx = lqx
        out_qy = lqy
        out_qz = lqz
        out_qw = lqw
        return
    var xpos_off = xpos_offset[NQ, NV, NBODY]()
    var xquat_off = xquat_offset[NQ, NV, NBODY]()
    var bpx = rebind[Scalar[DTYPE]](state[env, xpos_off + body_idx * 3 + 0])
    var bpy = rebind[Scalar[DTYPE]](state[env, xpos_off + body_idx * 3 + 1])
    var bpz = rebind[Scalar[DTYPE]](state[env, xpos_off + body_idx * 3 + 2])
    var bqx = rebind[Scalar[DTYPE]](state[env, xquat_off + body_idx * 4 + 0])
    var bqy = rebind[Scalar[DTYPE]](state[env, xquat_off + body_idx * 4 + 1])
    var bqz = rebind[Scalar[DTYPE]](state[env, xquat_off + body_idx * 4 + 2])
    var bqw = rebind[Scalar[DTYPE]](state[env, xquat_off + body_idx * 4 + 3])
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
def detect_contacts_gpu[
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
    """Unified contact detection on GPU using geom buffer section."""
    from ..collision.collision_primitives import (
        sphere_sphere,
        capsule_sphere,
        capsule_capsule,
        box_sphere,
        box_capsule,
        box_box,
        cylinder_plane,
        cylinder_sphere,
    )

    var contacts_off = contacts_offset[NQ, NV, NBODY]()
    var meta_off = metadata_offset[NQ, NV, NBODY, MAX_CONTACTS]()
    var num_contacts = 0

    for gi in range(NGEOM):
        var gi_off = model_geom_offset[NBODY, NJOINT](gi)
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
        for gj in range(gi + 1, NGEOM):
            if num_contacts >= MAX_CONTACTS:
                state[env, meta_off + META_IDX_NUM_CONTACTS] = Scalar[DTYPE](
                    num_contacts
                )
                return
            var gj_off = model_geom_offset[NBODY, NJOINT](gj)
            var gj_type = Int(
                rebind[Scalar[DTYPE]](model[0, gj_off + GEOM_IDX_TYPE])
            )
            var gj_body = Int(
                rebind[Scalar[DTYPE]](model[0, gj_off + GEOM_IDX_BODY])
            )
            if gi_type == GEOM_PLANE and gj_body == 0:
                continue
            if gj_type == GEOM_PLANE and gi_body == 0:
                continue
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

            var pi_x: Scalar[DTYPE] = 0
            var pi_y: Scalar[DTYPE] = 0
            var pi_z: Scalar[DTYPE] = 0
            var qi_x: Scalar[DTYPE] = 0
            var qi_y: Scalar[DTYPE] = 0
            var qi_z: Scalar[DTYPE] = 0
            var qi_w: Scalar[DTYPE] = 1
            _geom_world_pos_gpu[
                DTYPE, NQ, NV, NBODY, STATE_SIZE, MODEL_SIZE, BATCH
            ](
                env,
                gi_off,
                state,
                model,
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
            _geom_world_pos_gpu[
                DTYPE, NQ, NV, NBODY, STATE_SIZE, MODEL_SIZE, BATCH
            ](
                env,
                gj_off,
                state,
                model,
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
                    model[0, gi_off + GEOM_IDX_RBOUND]
                )
                var rj_bound = rebind[Scalar[DTYPE]](
                    model[0, gj_off + GEOM_IDX_RBOUND]
                )
                var bound = ri_bound + rj_bound
                if dist_sq > bound * bound:
                    continue

            var ri = rebind[Scalar[DTYPE]](model[0, gi_off + GEOM_IDX_RADIUS])
            var rj = rebind[Scalar[DTYPE]](model[0, gj_off + GEOM_IDX_RADIUS])
            var hli = rebind[Scalar[DTYPE]](
                model[0, gi_off + GEOM_IDX_HALF_LENGTH]
            )
            var hlj = rebind[Scalar[DTYPE]](
                model[0, gj_off + GEOM_IDX_HALF_LENGTH]
            )
            var hxi = rebind[Scalar[DTYPE]](model[0, gi_off + GEOM_IDX_HALF_X])
            var hyi = rebind[Scalar[DTYPE]](model[0, gi_off + GEOM_IDX_HALF_Y])
            var hzi = rebind[Scalar[DTYPE]](model[0, gi_off + GEOM_IDX_HALF_Z])
            var hxj = rebind[Scalar[DTYPE]](model[0, gj_off + GEOM_IDX_HALF_X])
            var hyj = rebind[Scalar[DTYPE]](model[0, gj_off + GEOM_IDX_HALF_Y])
            var hzj = rebind[Scalar[DTYPE]](model[0, gj_off + GEOM_IDX_HALF_Z])
            # Friction combination: max per element (MuJoCo convention)
            var fi = rebind[Scalar[DTYPE]](model[0, gi_off + GEOM_IDX_FRICTION])
            var fj = rebind[Scalar[DTYPE]](model[0, gj_off + GEOM_IDX_FRICTION])
            var contact_friction = fi
            if fj > fi:
                contact_friction = fj
            var fsi = rebind[Scalar[DTYPE]](
                model[0, gi_off + GEOM_IDX_FRICTION_SPIN]
            )
            var fsj = rebind[Scalar[DTYPE]](
                model[0, gj_off + GEOM_IDX_FRICTION_SPIN]
            )
            var contact_friction_spin = fsi
            if fsj > fsi:
                contact_friction_spin = fsj
            var fri = rebind[Scalar[DTYPE]](
                model[0, gi_off + GEOM_IDX_FRICTION_ROLL]
            )
            var frj = rebind[Scalar[DTYPE]](
                model[0, gj_off + GEOM_IDX_FRICTION_ROLL]
            )
            var contact_friction_roll = fri
            if frj > fri:
                contact_friction_roll = frj
            var ci = Int(
                rebind[Scalar[DTYPE]](model[0, gi_off + GEOM_IDX_CONDIM])
            )
            var cj = Int(
                rebind[Scalar[DTYPE]](model[0, gj_off + GEOM_IDX_CONDIM])
            )
            var contact_condim = ci
            if cj > ci:
                contact_condim = cj

            # Margin combination: max of both geoms (MuJoCo convention)
            var margin_gi = rebind[Scalar[DTYPE]](
                model[0, gi_off + GEOM_IDX_MARGIN]
            )
            var margin_gj = rebind[Scalar[DTYPE]](
                model[0, gj_off + GEOM_IDX_MARGIN]
            )
            var contact_margin = margin_gi
            if margin_gj > margin_gi:
                contact_margin = margin_gj

            # --- Plane handling ---
            if gi_type == GEOM_PLANE:
                var ground_z = pi_z
                if gj_type == GEOM_CAPSULE:
                    # MuJoCo mjc_PlaneCapsule: test BOTH endpoints, up to 2 contacts
                    var axis_w = gpu_quat_rotate(
                        qj_x,
                        qj_y,
                        qj_z,
                        qj_w,
                        Scalar[DTYPE](0),
                        Scalar[DTYPE](0),
                        Scalar[DTYPE](1),
                    )
                    # Endpoint 1: center + half_length * axis
                    var e1_x = pj_x + hlj * axis_w[0]
                    var e1_y = pj_y + hlj * axis_w[1]
                    var e1_z = pj_z + hlj * axis_w[2]
                    var dist1 = e1_z - rj - ground_z
                    if dist1 < contact_margin and num_contacts < MAX_CONTACTS:
                        var c_off = contacts_off + num_contacts * CONTACT_SIZE
                        state[env, c_off + CONTACT_IDX_BODY_A] = Scalar[DTYPE](
                            gj_body
                        )
                        state[env, c_off + CONTACT_IDX_BODY_B] = Scalar[DTYPE](
                            0
                        )
                        state[env, c_off + CONTACT_IDX_POS_X] = e1_x
                        state[env, c_off + CONTACT_IDX_POS_Y] = e1_y
                        state[
                            env, c_off + CONTACT_IDX_POS_Z
                        ] = ground_z + dist1 * Scalar[DTYPE](0.5)
                        state[env, c_off + CONTACT_IDX_NX] = Scalar[DTYPE](0)
                        state[env, c_off + CONTACT_IDX_NY] = Scalar[DTYPE](0)
                        state[env, c_off + CONTACT_IDX_NZ] = Scalar[DTYPE](1)
                        state[env, c_off + CONTACT_IDX_DIST] = (
                            dist1 - contact_margin
                        )
                        state[
                            env, c_off + CONTACT_IDX_FRICTION
                        ] = contact_friction
                        state[
                            env, c_off + CONTACT_IDX_FRICTION_SPIN
                        ] = contact_friction_spin
                        state[
                            env, c_off + CONTACT_IDX_FRICTION_ROLL
                        ] = contact_friction_roll
                        state[env, c_off + CONTACT_IDX_CONDIM] = Scalar[DTYPE](
                            contact_condim
                        )
                        state[env, c_off + CONTACT_IDX_FRAME_T1_X] = axis_w[0]
                        state[env, c_off + CONTACT_IDX_FRAME_T1_Y] = axis_w[1]
                        state[env, c_off + CONTACT_IDX_FRAME_T1_Z] = axis_w[2]
                        num_contacts += 1
                    # Endpoint 2: center - half_length * axis
                    var e2_x = pj_x - hlj * axis_w[0]
                    var e2_y = pj_y - hlj * axis_w[1]
                    var e2_z = pj_z - hlj * axis_w[2]
                    var dist2 = e2_z - rj - ground_z
                    if dist2 < contact_margin and num_contacts < MAX_CONTACTS:
                        var c_off = contacts_off + num_contacts * CONTACT_SIZE
                        state[env, c_off + CONTACT_IDX_BODY_A] = Scalar[DTYPE](
                            gj_body
                        )
                        state[env, c_off + CONTACT_IDX_BODY_B] = Scalar[DTYPE](
                            0
                        )
                        state[env, c_off + CONTACT_IDX_POS_X] = e2_x
                        state[env, c_off + CONTACT_IDX_POS_Y] = e2_y
                        state[
                            env, c_off + CONTACT_IDX_POS_Z
                        ] = ground_z + dist2 * Scalar[DTYPE](0.5)
                        state[env, c_off + CONTACT_IDX_NX] = Scalar[DTYPE](0)
                        state[env, c_off + CONTACT_IDX_NY] = Scalar[DTYPE](0)
                        state[env, c_off + CONTACT_IDX_NZ] = Scalar[DTYPE](1)
                        state[env, c_off + CONTACT_IDX_DIST] = (
                            dist2 - contact_margin
                        )
                        state[
                            env, c_off + CONTACT_IDX_FRICTION
                        ] = contact_friction
                        state[
                            env, c_off + CONTACT_IDX_FRICTION_SPIN
                        ] = contact_friction_spin
                        state[
                            env, c_off + CONTACT_IDX_FRICTION_ROLL
                        ] = contact_friction_roll
                        state[env, c_off + CONTACT_IDX_CONDIM] = Scalar[DTYPE](
                            contact_condim
                        )
                        state[env, c_off + CONTACT_IDX_FRAME_T1_X] = axis_w[0]
                        state[env, c_off + CONTACT_IDX_FRAME_T1_Y] = axis_w[1]
                        state[env, c_off + CONTACT_IDX_FRAME_T1_Z] = axis_w[2]
                        num_contacts += 1
                elif gj_type == GEOM_CYLINDER:
                    # Cylinder-plane: single contact at lowest rim point
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
                    if dist < contact_margin and num_contacts < MAX_CONTACTS:
                        var c_off = contacts_off + num_contacts * CONTACT_SIZE
                        state[env, c_off + CONTACT_IDX_BODY_A] = Scalar[DTYPE](
                            gj_body
                        )
                        state[env, c_off + CONTACT_IDX_BODY_B] = Scalar[DTYPE](
                            0
                        )
                        state[env, c_off + CONTACT_IDX_POS_X] = cp[1]
                        state[env, c_off + CONTACT_IDX_POS_Y] = cp[2]
                        state[env, c_off + CONTACT_IDX_POS_Z] = cp[3]
                        state[env, c_off + CONTACT_IDX_NX] = Scalar[DTYPE](0)
                        state[env, c_off + CONTACT_IDX_NY] = Scalar[DTYPE](0)
                        state[env, c_off + CONTACT_IDX_NZ] = Scalar[DTYPE](1)
                        state[env, c_off + CONTACT_IDX_DIST] = (
                            dist - contact_margin
                        )
                        state[
                            env, c_off + CONTACT_IDX_FRICTION
                        ] = contact_friction
                        state[
                            env, c_off + CONTACT_IDX_FRICTION_SPIN
                        ] = contact_friction_spin
                        state[
                            env, c_off + CONTACT_IDX_FRICTION_ROLL
                        ] = contact_friction_roll
                        state[env, c_off + CONTACT_IDX_CONDIM] = Scalar[DTYPE](
                            contact_condim
                        )
                        num_contacts += 1
                elif gj_type == GEOM_SPHERE:
                    var dist = pj_z - rj - ground_z
                    if dist < contact_margin and num_contacts < MAX_CONTACTS:
                        var c_off = contacts_off + num_contacts * CONTACT_SIZE
                        state[env, c_off + CONTACT_IDX_BODY_A] = Scalar[DTYPE](
                            gj_body
                        )
                        state[env, c_off + CONTACT_IDX_BODY_B] = Scalar[DTYPE](
                            0
                        )
                        state[env, c_off + CONTACT_IDX_POS_X] = pj_x
                        state[env, c_off + CONTACT_IDX_POS_Y] = pj_y
                        state[
                            env, c_off + CONTACT_IDX_POS_Z
                        ] = ground_z + dist * Scalar[DTYPE](0.5)
                        state[env, c_off + CONTACT_IDX_NX] = Scalar[DTYPE](0)
                        state[env, c_off + CONTACT_IDX_NY] = Scalar[DTYPE](0)
                        state[env, c_off + CONTACT_IDX_NZ] = Scalar[DTYPE](1)
                        state[env, c_off + CONTACT_IDX_DIST] = (
                            dist - contact_margin
                        )
                        state[
                            env, c_off + CONTACT_IDX_FRICTION
                        ] = contact_friction
                        state[
                            env, c_off + CONTACT_IDX_FRICTION_SPIN
                        ] = contact_friction_spin
                        state[
                            env, c_off + CONTACT_IDX_FRICTION_ROLL
                        ] = contact_friction_roll
                        state[env, c_off + CONTACT_IDX_CONDIM] = Scalar[DTYPE](
                            contact_condim
                        )
                        num_contacts += 1
                continue

            if gj_type == GEOM_PLANE:
                var ground_z = pj_z
                if gi_type == GEOM_CAPSULE:
                    # MuJoCo mjc_PlaneCapsule: test BOTH endpoints, up to 2 contacts
                    var axis_w = gpu_quat_rotate(
                        qi_x,
                        qi_y,
                        qi_z,
                        qi_w,
                        Scalar[DTYPE](0),
                        Scalar[DTYPE](0),
                        Scalar[DTYPE](1),
                    )
                    # Endpoint 1: center + half_length * axis
                    var e1_x = pi_x + hli * axis_w[0]
                    var e1_y = pi_y + hli * axis_w[1]
                    var e1_z = pi_z + hli * axis_w[2]
                    var dist1 = e1_z - ri - ground_z
                    if dist1 < contact_margin and num_contacts < MAX_CONTACTS:
                        var c_off = contacts_off + num_contacts * CONTACT_SIZE
                        state[env, c_off + CONTACT_IDX_BODY_A] = Scalar[DTYPE](
                            gi_body
                        )
                        state[env, c_off + CONTACT_IDX_BODY_B] = Scalar[DTYPE](
                            0
                        )
                        state[env, c_off + CONTACT_IDX_POS_X] = e1_x
                        state[env, c_off + CONTACT_IDX_POS_Y] = e1_y
                        state[
                            env, c_off + CONTACT_IDX_POS_Z
                        ] = ground_z + dist1 * Scalar[DTYPE](0.5)
                        state[env, c_off + CONTACT_IDX_NX] = Scalar[DTYPE](0)
                        state[env, c_off + CONTACT_IDX_NY] = Scalar[DTYPE](0)
                        state[env, c_off + CONTACT_IDX_NZ] = Scalar[DTYPE](1)
                        state[env, c_off + CONTACT_IDX_DIST] = (
                            dist1 - contact_margin
                        )
                        state[
                            env, c_off + CONTACT_IDX_FRICTION
                        ] = contact_friction
                        state[
                            env, c_off + CONTACT_IDX_FRICTION_SPIN
                        ] = contact_friction_spin
                        state[
                            env, c_off + CONTACT_IDX_FRICTION_ROLL
                        ] = contact_friction_roll
                        state[env, c_off + CONTACT_IDX_CONDIM] = Scalar[DTYPE](
                            contact_condim
                        )
                        state[env, c_off + CONTACT_IDX_FRAME_T1_X] = axis_w[0]
                        state[env, c_off + CONTACT_IDX_FRAME_T1_Y] = axis_w[1]
                        state[env, c_off + CONTACT_IDX_FRAME_T1_Z] = axis_w[2]
                        num_contacts += 1
                    # Endpoint 2: center - half_length * axis
                    var e2_x = pi_x - hli * axis_w[0]
                    var e2_y = pi_y - hli * axis_w[1]
                    var e2_z = pi_z - hli * axis_w[2]
                    var dist2 = e2_z - ri - ground_z
                    if dist2 < contact_margin and num_contacts < MAX_CONTACTS:
                        var c_off = contacts_off + num_contacts * CONTACT_SIZE
                        state[env, c_off + CONTACT_IDX_BODY_A] = Scalar[DTYPE](
                            gi_body
                        )
                        state[env, c_off + CONTACT_IDX_BODY_B] = Scalar[DTYPE](
                            0
                        )
                        state[env, c_off + CONTACT_IDX_POS_X] = e2_x
                        state[env, c_off + CONTACT_IDX_POS_Y] = e2_y
                        state[
                            env, c_off + CONTACT_IDX_POS_Z
                        ] = ground_z + dist2 * Scalar[DTYPE](0.5)
                        state[env, c_off + CONTACT_IDX_NX] = Scalar[DTYPE](0)
                        state[env, c_off + CONTACT_IDX_NY] = Scalar[DTYPE](0)
                        state[env, c_off + CONTACT_IDX_NZ] = Scalar[DTYPE](1)
                        state[env, c_off + CONTACT_IDX_DIST] = (
                            dist2 - contact_margin
                        )
                        state[
                            env, c_off + CONTACT_IDX_FRICTION
                        ] = contact_friction
                        state[
                            env, c_off + CONTACT_IDX_FRICTION_SPIN
                        ] = contact_friction_spin
                        state[
                            env, c_off + CONTACT_IDX_FRICTION_ROLL
                        ] = contact_friction_roll
                        state[env, c_off + CONTACT_IDX_CONDIM] = Scalar[DTYPE](
                            contact_condim
                        )
                        state[env, c_off + CONTACT_IDX_FRAME_T1_X] = axis_w[0]
                        state[env, c_off + CONTACT_IDX_FRAME_T1_Y] = axis_w[1]
                        state[env, c_off + CONTACT_IDX_FRAME_T1_Z] = axis_w[2]
                        num_contacts += 1
                elif gi_type == GEOM_CYLINDER:
                    # Cylinder-plane: single contact at lowest rim point
                    var cp = cylinder_plane[DTYPE](
                        pi_x,
                        pi_y,
                        pi_z,
                        qi_x,
                        qi_y,
                        qi_z,
                        qi_w,
                        hli,
                        ri,
                        ground_z,
                    )
                    var dist = cp[0]
                    if dist < contact_margin and num_contacts < MAX_CONTACTS:
                        var c_off = contacts_off + num_contacts * CONTACT_SIZE
                        state[env, c_off + CONTACT_IDX_BODY_A] = Scalar[DTYPE](
                            gi_body
                        )
                        state[env, c_off + CONTACT_IDX_BODY_B] = Scalar[DTYPE](
                            0
                        )
                        state[env, c_off + CONTACT_IDX_POS_X] = cp[1]
                        state[env, c_off + CONTACT_IDX_POS_Y] = cp[2]
                        state[env, c_off + CONTACT_IDX_POS_Z] = cp[3]
                        state[env, c_off + CONTACT_IDX_NX] = Scalar[DTYPE](0)
                        state[env, c_off + CONTACT_IDX_NY] = Scalar[DTYPE](0)
                        state[env, c_off + CONTACT_IDX_NZ] = Scalar[DTYPE](1)
                        state[env, c_off + CONTACT_IDX_DIST] = (
                            dist - contact_margin
                        )
                        state[
                            env, c_off + CONTACT_IDX_FRICTION
                        ] = contact_friction
                        state[
                            env, c_off + CONTACT_IDX_FRICTION_SPIN
                        ] = contact_friction_spin
                        state[
                            env, c_off + CONTACT_IDX_FRICTION_ROLL
                        ] = contact_friction_roll
                        state[env, c_off + CONTACT_IDX_CONDIM] = Scalar[DTYPE](
                            contact_condim
                        )
                        num_contacts += 1
                elif gi_type == GEOM_SPHERE:
                    var dist = pi_z - ri - ground_z
                    if dist < contact_margin and num_contacts < MAX_CONTACTS:
                        var c_off = contacts_off + num_contacts * CONTACT_SIZE
                        state[env, c_off + CONTACT_IDX_BODY_A] = Scalar[DTYPE](
                            gi_body
                        )
                        state[env, c_off + CONTACT_IDX_BODY_B] = Scalar[DTYPE](
                            0
                        )
                        state[env, c_off + CONTACT_IDX_POS_X] = pi_x
                        state[env, c_off + CONTACT_IDX_POS_Y] = pi_y
                        state[
                            env, c_off + CONTACT_IDX_POS_Z
                        ] = ground_z + dist * Scalar[DTYPE](0.5)
                        state[env, c_off + CONTACT_IDX_NX] = Scalar[DTYPE](0)
                        state[env, c_off + CONTACT_IDX_NY] = Scalar[DTYPE](0)
                        state[env, c_off + CONTACT_IDX_NZ] = Scalar[DTYPE](1)
                        state[env, c_off + CONTACT_IDX_DIST] = (
                            dist - contact_margin
                        )
                        state[
                            env, c_off + CONTACT_IDX_FRICTION
                        ] = contact_friction
                        state[
                            env, c_off + CONTACT_IDX_FRICTION_SPIN
                        ] = contact_friction_spin
                        state[
                            env, c_off + CONTACT_IDX_FRICTION_ROLL
                        ] = contact_friction_roll
                        state[env, c_off + CONTACT_IDX_CONDIM] = Scalar[DTYPE](
                            contact_condim
                        )
                        num_contacts += 1
                continue

            # --- Non-plane geom pair ---
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

            if dist < contact_margin and num_contacts < MAX_CONTACTS:
                var c_off = contacts_off + num_contacts * CONTACT_SIZE
                state[env, c_off + CONTACT_IDX_BODY_A] = Scalar[DTYPE](body_a)
                state[env, c_off + CONTACT_IDX_BODY_B] = Scalar[DTYPE](body_b)
                state[env, c_off + CONTACT_IDX_POS_X] = cx
                state[env, c_off + CONTACT_IDX_POS_Y] = cy
                state[env, c_off + CONTACT_IDX_POS_Z] = cz
                state[env, c_off + CONTACT_IDX_NX] = nx
                state[env, c_off + CONTACT_IDX_NY] = ny
                state[env, c_off + CONTACT_IDX_NZ] = nz
                state[env, c_off + CONTACT_IDX_DIST] = dist - contact_margin
                state[env, c_off + CONTACT_IDX_FRICTION] = contact_friction
                state[
                    env, c_off + CONTACT_IDX_FRICTION_SPIN
                ] = contact_friction_spin
                state[
                    env, c_off + CONTACT_IDX_FRICTION_ROLL
                ] = contact_friction_roll
                state[env, c_off + CONTACT_IDX_CONDIM] = Scalar[DTYPE](
                    contact_condim
                )
                num_contacts += 1

    state[env, meta_off + META_IDX_NUM_CONTACTS] = Scalar[DTYPE](num_contacts)
