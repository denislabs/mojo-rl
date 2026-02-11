"""Contact detection for physics engine.

Provides ground contact detection, body-body contact detection,
and quaternion normalization utilities.
"""

from math import sqrt
from layout import LayoutTensor, Layout
from ..types import (
    Model,
    Data,
)
from ..joint_types import JNT_HINGE, JNT_SLIDE, JNT_BALL, JNT_FREE

from ..kinematics.quat_math import (
    quat_normalize,
    quat_rotate,
    gpu_quat_rotate,
    gpu_quat_normalize,
)
from ..constants import GEOM_SPHERE, GEOM_CAPSULE, GEOM_BOX
from ..gpu.constants import (
    BODY_IDX_RADIUS,
    BODY_IDX_HALF_LENGTH,
    BODY_IDX_HALF_X,
    BODY_IDX_HALF_Y,
    BODY_IDX_HALF_Z,
    BODY_IDX_PARENT,
    BODY_IDX_GEOM_TYPE,
    BODY_IDX_CONTYPE,
    BODY_IDX_CONAFFINITY,
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
    JOINT_IDX_TYPE,
    JOINT_IDX_QPOS_ADR,
    META_IDX_NUM_CONTACTS,
    MODEL_META_IDX_GROUND_Z,
    MODEL_META_IDX_GROUND_CONTYPE,
    MODEL_META_IDX_GROUND_CONAFFINITY,
    MODEL_META_IDX_NJOINT,
    model_body_offset,
    model_joint_offset,
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
)


fn normalize_qpos_quaternions[
    DTYPE: DType,
    NQ: Int,
    NV: Int,
    NBODY: Int,
    NJOINT: Int,
    MAX_CONTACTS: Int,
](
    model: Model[DTYPE, NQ, NV, NBODY, NJOINT, MAX_CONTACTS],
    mut data: Data[DTYPE, NQ, NV, NBODY, NJOINT, MAX_CONTACTS],
) where DTYPE.is_floating_point():
    """Normalize quaternions in qpos for BALL and FREE joints."""
    for j in range(model.num_joints):
        var joint = model.joints[j]
        var qpos_adr = joint.qpos_adr

        if joint.jnt_type == JNT_FREE:
            # Quaternion at qpos_adr + 3..6
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
            # Quaternion at qpos_adr + 0..3
            var qx = data.qpos[qpos_adr + 0]
            var qy = data.qpos[qpos_adr + 1]
            var qz = data.qpos[qpos_adr + 2]
            var qw = data.qpos[qpos_adr + 3]

            var normalized = quat_normalize(qx, qy, qz, qw)
            data.qpos[qpos_adr + 0] = normalized[0]
            data.qpos[qpos_adr + 1] = normalized[1]
            data.qpos[qpos_adr + 2] = normalized[2]
            data.qpos[qpos_adr + 3] = normalized[3]


fn detect_ground_contacts[
    DTYPE: DType,
    NQ: Int,
    NV: Int,
    NBODY: Int,
    NJOINT: Int,
    MAX_CONTACTS: Int,
](
    model: Model[DTYPE, NQ, NV, NBODY, NJOINT, MAX_CONTACTS],
    mut data: Data[DTYPE, NQ, NV, NBODY, NJOINT, MAX_CONTACTS],
):
    """Detect contacts between bodies and ground plane.

    For capsules, checks both endpoints (center ± half_length along axis).
    The capsule axis is determined by the body's world orientation.
    """
    data.num_contacts = 0
    var ground_z = model.ground_z
    var ground_contype = model.ground_contype
    var ground_conaffinity = model.ground_conaffinity

    for body in range(NBODY):
        # MuJoCo contype/conaffinity filtering for ground
        if (model.body_contype[body] & ground_conaffinity) == 0 and (
            ground_contype & model.body_conaffinity[body]
        ) == 0:
            continue

        var px = data.xpos[body * 3 + 0]
        var py = data.xpos[body * 3 + 1]
        var pz = data.xpos[body * 3 + 2]
        var radius = model.body_radius[body]
        var half_length = model.body_half_length[body]

        # Get body orientation
        var qx = data.xquat[body * 4 + 0]
        var qy = data.xquat[body * 4 + 1]
        var qz = data.xquat[body * 4 + 2]
        var qw = data.xquat[body * 4 + 3]

        # Capsule axis in local frame is (0, 0, 1) - along Z
        # Transform to world frame
        var axis_world = quat_rotate(
            qx, qy, qz, qw, Scalar[DTYPE](0), Scalar[DTYPE](0), Scalar[DTYPE](1)
        )
        var axis_x = axis_world[0]
        var axis_y = axis_world[1]
        var axis_z = axis_world[2]

        # For spheres (half_length = 0), just check center - radius
        if half_length <= Scalar[DTYPE](0.0001):
            var dist = pz - radius - ground_z
            if dist < Scalar[DTYPE](0):
                if data.num_contacts < MAX_CONTACTS:
                    var idx = data.num_contacts
                    data.contacts[idx].body_a = body
                    data.contacts[idx].body_b = -1  # Ground
                    data.contacts[idx].pos_x = px
                    data.contacts[idx].pos_y = py
                    data.contacts[idx].pos_z = ground_z
                    data.contacts[idx].normal_x = Scalar[DTYPE](0)
                    data.contacts[idx].normal_y = Scalar[DTYPE](0)
                    data.contacts[idx].normal_z = Scalar[DTYPE](1)
                    data.contacts[idx].dist = dist
                    data.num_contacts += 1
        else:
            # Capsule: check both endpoints
            # Endpoint 1: center + half_length * axis
            var e1_x = px + half_length * axis_x
            var e1_y = py + half_length * axis_y
            var e1_z = pz + half_length * axis_z
            var dist1 = e1_z - radius - ground_z

            # Endpoint 2: center - half_length * axis
            var e2_x = px - half_length * axis_x
            var e2_y = py - half_length * axis_y
            var e2_z = pz - half_length * axis_z
            var dist2 = e2_z - radius - ground_z

            # Check endpoint 1
            if dist1 < Scalar[DTYPE](0):
                if data.num_contacts < MAX_CONTACTS:
                    var idx = data.num_contacts
                    data.contacts[idx].body_a = body
                    data.contacts[idx].body_b = -1  # Ground
                    data.contacts[idx].pos_x = e1_x
                    data.contacts[idx].pos_y = e1_y
                    data.contacts[idx].pos_z = ground_z
                    data.contacts[idx].normal_x = Scalar[DTYPE](0)
                    data.contacts[idx].normal_y = Scalar[DTYPE](0)
                    data.contacts[idx].normal_z = Scalar[DTYPE](1)
                    data.contacts[idx].dist = dist1
                    data.num_contacts += 1

            # Check endpoint 2
            if dist2 < Scalar[DTYPE](0):
                if data.num_contacts < MAX_CONTACTS:
                    var idx = data.num_contacts
                    data.contacts[idx].body_a = body
                    data.contacts[idx].body_b = -1  # Ground
                    data.contacts[idx].pos_x = e2_x
                    data.contacts[idx].pos_y = e2_y
                    data.contacts[idx].pos_z = ground_z
                    data.contacts[idx].normal_x = Scalar[DTYPE](0)
                    data.contacts[idx].normal_y = Scalar[DTYPE](0)
                    data.contacts[idx].normal_z = Scalar[DTYPE](1)
                    data.contacts[idx].dist = dist2
                    data.num_contacts += 1


fn detect_body_body_contacts[
    DTYPE: DType,
    NQ: Int,
    NV: Int,
    NBODY: Int,
    NJOINT: Int,
    MAX_CONTACTS: Int,
](
    model: Model[DTYPE, NQ, NV, NBODY, NJOINT, MAX_CONTACTS],
    mut data: Data[DTYPE, NQ, NV, NBODY, NJOINT, MAX_CONTACTS],
):
    """Detect body-body contacts and append to existing contact list.

    Does NOT reset num_contacts — appends to contacts from ground detection.
    O(N^2) pair iteration, skipping parent-child pairs.
    Dispatches to collision primitives based on geometry types.
    """

    for i in range(NBODY):
        for j in range(i + 1, NBODY):
            # Skip parent-child pairs (connected bodies)
            if model.body_parent[j] == i or model.body_parent[i] == j:
                continue

            # MuJoCo contype/conaffinity filtering:
            # collide if (contype_i & conaffinity_j) || (contype_j & conaffinity_i)
            if (model.body_contype[i] & model.body_conaffinity[j]) == 0 and (
                model.body_contype[j] & model.body_conaffinity[i]
            ) == 0:
                continue

            # Skip if already at max contacts
            if data.num_contacts >= MAX_CONTACTS:
                return

            var gi = model.body_geom_type[i]
            var gj = model.body_geom_type[j]

            # Get positions
            var pi_x = data.xpos[i * 3 + 0]
            var pi_y = data.xpos[i * 3 + 1]
            var pi_z = data.xpos[i * 3 + 2]
            var pj_x = data.xpos[j * 3 + 0]
            var pj_y = data.xpos[j * 3 + 1]
            var pj_z = data.xpos[j * 3 + 2]

            # Get quaternions
            var qi_x = data.xquat[i * 4 + 0]
            var qi_y = data.xquat[i * 4 + 1]
            var qi_z = data.xquat[i * 4 + 2]
            var qi_w = data.xquat[i * 4 + 3]
            var qj_x = data.xquat[j * 4 + 0]
            var qj_y = data.xquat[j * 4 + 1]
            var qj_z = data.xquat[j * 4 + 2]
            var qj_w = data.xquat[j * 4 + 3]

            var dist: Scalar[DTYPE] = 1.0  # Default: no contact
            var cx: Scalar[DTYPE] = 0
            var cy: Scalar[DTYPE] = 0
            var cz: Scalar[DTYPE] = 0
            var nx: Scalar[DTYPE] = 0
            var ny: Scalar[DTYPE] = 0
            var nz: Scalar[DTYPE] = 1
            var swap = False  # Whether we swapped body order for the primitive

            # Dispatch based on geometry pair
            if gi == GEOM_SPHERE and gj == GEOM_SPHERE:
                var result = sphere_sphere[DTYPE](
                    pi_x,
                    pi_y,
                    pi_z,
                    model.body_radius[i],
                    pj_x,
                    pj_y,
                    pj_z,
                    model.body_radius[j],
                )
                dist = result[0]
                cx = result[1]
                cy = result[2]
                cz = result[3]
                nx = result[4]
                ny = result[5]
                nz = result[6]

            elif gi == GEOM_CAPSULE and gj == GEOM_SPHERE:
                var result = capsule_sphere[DTYPE](
                    pi_x,
                    pi_y,
                    pi_z,
                    qi_x,
                    qi_y,
                    qi_z,
                    qi_w,
                    model.body_half_length[i],
                    model.body_radius[i],
                    pj_x,
                    pj_y,
                    pj_z,
                    model.body_radius[j],
                )
                dist = result[0]
                cx = result[1]
                cy = result[2]
                cz = result[3]
                nx = result[4]
                ny = result[5]
                nz = result[6]

            elif gi == GEOM_SPHERE and gj == GEOM_CAPSULE:
                # Swap: call capsule_sphere with j as capsule, negate normal
                var result = capsule_sphere[DTYPE](
                    pj_x,
                    pj_y,
                    pj_z,
                    qj_x,
                    qj_y,
                    qj_z,
                    qj_w,
                    model.body_half_length[j],
                    model.body_radius[j],
                    pi_x,
                    pi_y,
                    pi_z,
                    model.body_radius[i],
                )
                dist = result[0]
                cx = result[1]
                cy = result[2]
                cz = result[3]
                nx = -result[4]
                ny = -result[5]
                nz = -result[6]
                swap = True

            elif gi == GEOM_CAPSULE and gj == GEOM_CAPSULE:
                var result = capsule_capsule[DTYPE](
                    pi_x,
                    pi_y,
                    pi_z,
                    qi_x,
                    qi_y,
                    qi_z,
                    qi_w,
                    model.body_half_length[i],
                    model.body_radius[i],
                    pj_x,
                    pj_y,
                    pj_z,
                    qj_x,
                    qj_y,
                    qj_z,
                    qj_w,
                    model.body_half_length[j],
                    model.body_radius[j],
                )
                dist = result[0]
                cx = result[1]
                cy = result[2]
                cz = result[3]
                nx = result[4]
                ny = result[5]
                nz = result[6]

            elif gi == GEOM_BOX and gj == GEOM_SPHERE:
                var result = box_sphere[DTYPE](
                    pi_x,
                    pi_y,
                    pi_z,
                    qi_x,
                    qi_y,
                    qi_z,
                    qi_w,
                    model.body_half_x[i],
                    model.body_half_y[i],
                    model.body_half_z[i],
                    pj_x,
                    pj_y,
                    pj_z,
                    model.body_radius[j],
                )
                dist = result[0]
                cx = result[1]
                cy = result[2]
                cz = result[3]
                nx = result[4]
                ny = result[5]
                nz = result[6]

            elif gi == GEOM_SPHERE and gj == GEOM_BOX:
                # Swap: call box_sphere with j as box, negate normal
                var result = box_sphere[DTYPE](
                    pj_x,
                    pj_y,
                    pj_z,
                    qj_x,
                    qj_y,
                    qj_z,
                    qj_w,
                    model.body_half_x[j],
                    model.body_half_y[j],
                    model.body_half_z[j],
                    pi_x,
                    pi_y,
                    pi_z,
                    model.body_radius[i],
                )
                dist = result[0]
                cx = result[1]
                cy = result[2]
                cz = result[3]
                nx = -result[4]
                ny = -result[5]
                nz = -result[6]
                swap = True

            elif gi == GEOM_BOX and gj == GEOM_CAPSULE:
                var result = box_capsule[DTYPE](
                    pi_x,
                    pi_y,
                    pi_z,
                    qi_x,
                    qi_y,
                    qi_z,
                    qi_w,
                    model.body_half_x[i],
                    model.body_half_y[i],
                    model.body_half_z[i],
                    pj_x,
                    pj_y,
                    pj_z,
                    qj_x,
                    qj_y,
                    qj_z,
                    qj_w,
                    model.body_half_length[j],
                    model.body_radius[j],
                )
                dist = result[0]
                cx = result[1]
                cy = result[2]
                cz = result[3]
                nx = result[4]
                ny = result[5]
                nz = result[6]

            elif gi == GEOM_CAPSULE and gj == GEOM_BOX:
                # Swap: call box_capsule with j as box, negate normal
                var result = box_capsule[DTYPE](
                    pj_x,
                    pj_y,
                    pj_z,
                    qj_x,
                    qj_y,
                    qj_z,
                    qj_w,
                    model.body_half_x[j],
                    model.body_half_y[j],
                    model.body_half_z[j],
                    pi_x,
                    pi_y,
                    pi_z,
                    qi_x,
                    qi_y,
                    qi_z,
                    qi_w,
                    model.body_half_length[i],
                    model.body_radius[i],
                )
                dist = result[0]
                cx = result[1]
                cy = result[2]
                cz = result[3]
                nx = -result[4]
                ny = -result[5]
                nz = -result[6]
                swap = True

            # Store contact if penetrating
            if dist < Scalar[DTYPE](0) and data.num_contacts < MAX_CONTACTS:
                var idx = data.num_contacts
                if swap:
                    data.contacts[idx].body_a = j
                    data.contacts[idx].body_b = i
                else:
                    data.contacts[idx].body_a = i
                    data.contacts[idx].body_b = j
                data.contacts[idx].pos_x = cx
                data.contacts[idx].pos_y = cy
                data.contacts[idx].pos_z = cz
                data.contacts[idx].normal_x = nx
                data.contacts[idx].normal_y = ny
                data.contacts[idx].normal_z = nz
                data.contacts[idx].dist = dist
                data.num_contacts += 1


# =============================================================================
# Ground Contact Detection Kernel
# =============================================================================


@always_inline
fn detect_ground_contacts_gpu[
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
    """Detect contacts between bodies and ground plane.

    For capsules, checks both endpoints (center ± half_length along axis).
    The capsule axis is determined by the body's world orientation.
    """
    var xpos_off = xpos_offset[NQ, NV, NBODY]()
    var xquat_off = xquat_offset[NQ, NV, NBODY]()
    var contacts_off = contacts_offset[NQ, NV, NBODY]()
    var meta_off = metadata_offset[NQ, NV, NBODY, MAX_CONTACTS]()

    var model_meta_off = model_metadata_offset[NBODY, NJOINT]()
    var ground_z = rebind[Scalar[DTYPE]](
        model[0, model_meta_off + MODEL_META_IDX_GROUND_Z]
    )
    var ground_contype = Int(
        rebind[Scalar[DTYPE]](
            model[0, model_meta_off + MODEL_META_IDX_GROUND_CONTYPE]
        )
    )
    var ground_conaffinity = Int(
        rebind[Scalar[DTYPE]](
            model[0, model_meta_off + MODEL_META_IDX_GROUND_CONAFFINITY]
        )
    )

    var num_contacts = 0

    for body in range(NBODY):
        var body_off = model_body_offset(body)

        # MuJoCo contype/conaffinity filtering for ground
        var body_contype = Int(
            rebind[Scalar[DTYPE]](model[0, body_off + BODY_IDX_CONTYPE])
        )
        var body_conaffinity = Int(
            rebind[Scalar[DTYPE]](model[0, body_off + BODY_IDX_CONAFFINITY])
        )
        if (body_contype & ground_conaffinity) == 0 and (
            ground_contype & body_conaffinity
        ) == 0:
            continue

        var radius = rebind[Scalar[DTYPE]](model[0, body_off + BODY_IDX_RADIUS])
        var half_length = rebind[Scalar[DTYPE]](
            model[0, body_off + BODY_IDX_HALF_LENGTH]
        )

        var px = rebind[Scalar[DTYPE]](state[env, xpos_off + body * 3 + 0])
        var py = rebind[Scalar[DTYPE]](state[env, xpos_off + body * 3 + 1])
        var pz = rebind[Scalar[DTYPE]](state[env, xpos_off + body * 3 + 2])

        # Get body orientation
        var qx = rebind[Scalar[DTYPE]](state[env, xquat_off + body * 4 + 0])
        var qy = rebind[Scalar[DTYPE]](state[env, xquat_off + body * 4 + 1])
        var qz = rebind[Scalar[DTYPE]](state[env, xquat_off + body * 4 + 2])
        var qw = rebind[Scalar[DTYPE]](state[env, xquat_off + body * 4 + 3])

        # Capsule axis in local frame is (0, 0, 1) - along Z
        # Transform to world frame
        var axis_world = gpu_quat_rotate(
            qx, qy, qz, qw, Scalar[DTYPE](0), Scalar[DTYPE](0), Scalar[DTYPE](1)
        )
        var axis_x = axis_world[0]
        var axis_y = axis_world[1]
        var axis_z = axis_world[2]

        # For spheres (half_length = 0), just check center - radius
        if half_length <= Scalar[DTYPE](0.0001):
            var dist = pz - radius - ground_z
            if dist < Scalar[DTYPE](0) and num_contacts < MAX_CONTACTS:
                var c_off = contacts_off + num_contacts * CONTACT_SIZE
                state[env, c_off + CONTACT_IDX_BODY_A] = Scalar[DTYPE](body)
                state[env, c_off + CONTACT_IDX_BODY_B] = Scalar[DTYPE](-1)
                state[env, c_off + CONTACT_IDX_POS_X] = px
                state[env, c_off + CONTACT_IDX_POS_Y] = py
                state[env, c_off + CONTACT_IDX_POS_Z] = ground_z
                state[env, c_off + CONTACT_IDX_NX] = Scalar[DTYPE](0)
                state[env, c_off + CONTACT_IDX_NY] = Scalar[DTYPE](0)
                state[env, c_off + CONTACT_IDX_NZ] = Scalar[DTYPE](1)
                state[env, c_off + CONTACT_IDX_DIST] = dist
                num_contacts += 1
        else:
            # Capsule: check both endpoints
            # Endpoint 1: center + half_length * axis
            var e1_x = px + half_length * axis_x
            var e1_y = py + half_length * axis_y
            var e1_z = pz + half_length * axis_z
            var dist1 = e1_z - radius - ground_z

            # Endpoint 2: center - half_length * axis
            var e2_x = px - half_length * axis_x
            var e2_y = py - half_length * axis_y
            var e2_z = pz - half_length * axis_z
            var dist2 = e2_z - radius - ground_z

            # Check endpoint 1
            if dist1 < Scalar[DTYPE](0) and num_contacts < MAX_CONTACTS:
                var c_off = contacts_off + num_contacts * CONTACT_SIZE
                state[env, c_off + CONTACT_IDX_BODY_A] = Scalar[DTYPE](body)
                state[env, c_off + CONTACT_IDX_BODY_B] = Scalar[DTYPE](-1)
                state[env, c_off + CONTACT_IDX_POS_X] = e1_x
                state[env, c_off + CONTACT_IDX_POS_Y] = e1_y
                state[env, c_off + CONTACT_IDX_POS_Z] = ground_z
                state[env, c_off + CONTACT_IDX_NX] = Scalar[DTYPE](0)
                state[env, c_off + CONTACT_IDX_NY] = Scalar[DTYPE](0)
                state[env, c_off + CONTACT_IDX_NZ] = Scalar[DTYPE](1)
                state[env, c_off + CONTACT_IDX_DIST] = dist1
                num_contacts += 1

            # Check endpoint 2
            if dist2 < Scalar[DTYPE](0) and num_contacts < MAX_CONTACTS:
                var c_off = contacts_off + num_contacts * CONTACT_SIZE
                state[env, c_off + CONTACT_IDX_BODY_A] = Scalar[DTYPE](body)
                state[env, c_off + CONTACT_IDX_BODY_B] = Scalar[DTYPE](-1)
                state[env, c_off + CONTACT_IDX_POS_X] = e2_x
                state[env, c_off + CONTACT_IDX_POS_Y] = e2_y
                state[env, c_off + CONTACT_IDX_POS_Z] = ground_z
                state[env, c_off + CONTACT_IDX_NX] = Scalar[DTYPE](0)
                state[env, c_off + CONTACT_IDX_NY] = Scalar[DTYPE](0)
                state[env, c_off + CONTACT_IDX_NZ] = Scalar[DTYPE](1)
                state[env, c_off + CONTACT_IDX_DIST] = dist2
                num_contacts += 1

    state[env, meta_off + META_IDX_NUM_CONTACTS] = Scalar[DTYPE](num_contacts)


# =============================================================================
# Body-Body Contact Detection Kernel
# =============================================================================


@always_inline
fn detect_body_body_contacts_gpu[
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
    """Detect body-body contacts and append to existing contact list.

    Reads current num_contacts (set by ground detection) and appends.
    O(N^2) pair iteration, skipping parent-child pairs.
    """
    from ..constants import GEOM_SPHERE, GEOM_CAPSULE, GEOM_BOX
    from ..collision.collision_primitives import (
        sphere_sphere,
        capsule_sphere,
        capsule_capsule,
        box_sphere,
        box_capsule,
    )

    var xpos_off = xpos_offset[NQ, NV, NBODY]()
    var xquat_off = xquat_offset[NQ, NV, NBODY]()
    var contacts_off = contacts_offset[NQ, NV, NBODY]()
    var meta_off = metadata_offset[NQ, NV, NBODY, MAX_CONTACTS]()

    var num_contacts = Int(
        rebind[Scalar[DTYPE]](state[env, meta_off + META_IDX_NUM_CONTACTS])
    )

    for i in range(NBODY):
        for j in range(i + 1, NBODY):
            if num_contacts >= MAX_CONTACTS:
                state[env, meta_off + META_IDX_NUM_CONTACTS] = Scalar[DTYPE](
                    num_contacts
                )
                return

            # Skip parent-child pairs
            var body_off_i = model_body_offset(i)
            var body_off_j = model_body_offset(j)
            var parent_i = Int(
                rebind[Scalar[DTYPE]](model[0, body_off_i + BODY_IDX_PARENT])
            )
            var parent_j = Int(
                rebind[Scalar[DTYPE]](model[0, body_off_j + BODY_IDX_PARENT])
            )
            if parent_j == i or parent_i == j:
                continue

            # MuJoCo contype/conaffinity filtering
            var contype_i = Int(
                rebind[Scalar[DTYPE]](model[0, body_off_i + BODY_IDX_CONTYPE])
            )
            var conaffinity_i = Int(
                rebind[Scalar[DTYPE]](
                    model[0, body_off_i + BODY_IDX_CONAFFINITY]
                )
            )
            var contype_j = Int(
                rebind[Scalar[DTYPE]](model[0, body_off_j + BODY_IDX_CONTYPE])
            )
            var conaffinity_j = Int(
                rebind[Scalar[DTYPE]](
                    model[0, body_off_j + BODY_IDX_CONAFFINITY]
                )
            )
            if (contype_i & conaffinity_j) == 0 and (
                contype_j & conaffinity_i
            ) == 0:
                continue

            var gi = Int(
                rebind[Scalar[DTYPE]](model[0, body_off_i + BODY_IDX_GEOM_TYPE])
            )
            var gj = Int(
                rebind[Scalar[DTYPE]](model[0, body_off_j + BODY_IDX_GEOM_TYPE])
            )

            # Get positions
            var pi_x = rebind[Scalar[DTYPE]](state[env, xpos_off + i * 3 + 0])
            var pi_y = rebind[Scalar[DTYPE]](state[env, xpos_off + i * 3 + 1])
            var pi_z = rebind[Scalar[DTYPE]](state[env, xpos_off + i * 3 + 2])
            var pj_x = rebind[Scalar[DTYPE]](state[env, xpos_off + j * 3 + 0])
            var pj_y = rebind[Scalar[DTYPE]](state[env, xpos_off + j * 3 + 1])
            var pj_z = rebind[Scalar[DTYPE]](state[env, xpos_off + j * 3 + 2])

            # Get quaternions
            var qi_x = rebind[Scalar[DTYPE]](state[env, xquat_off + i * 4 + 0])
            var qi_y = rebind[Scalar[DTYPE]](state[env, xquat_off + i * 4 + 1])
            var qi_z = rebind[Scalar[DTYPE]](state[env, xquat_off + i * 4 + 2])
            var qi_w = rebind[Scalar[DTYPE]](state[env, xquat_off + i * 4 + 3])
            var qj_x = rebind[Scalar[DTYPE]](state[env, xquat_off + j * 4 + 0])
            var qj_y = rebind[Scalar[DTYPE]](state[env, xquat_off + j * 4 + 1])
            var qj_z = rebind[Scalar[DTYPE]](state[env, xquat_off + j * 4 + 2])
            var qj_w = rebind[Scalar[DTYPE]](state[env, xquat_off + j * 4 + 3])

            # Get geometry parameters
            var ri = rebind[Scalar[DTYPE]](
                model[0, body_off_i + BODY_IDX_RADIUS]
            )
            var rj = rebind[Scalar[DTYPE]](
                model[0, body_off_j + BODY_IDX_RADIUS]
            )
            var hli = rebind[Scalar[DTYPE]](
                model[0, body_off_i + BODY_IDX_HALF_LENGTH]
            )
            var hlj = rebind[Scalar[DTYPE]](
                model[0, body_off_j + BODY_IDX_HALF_LENGTH]
            )
            var hxi = rebind[Scalar[DTYPE]](
                model[0, body_off_i + BODY_IDX_HALF_X]
            )
            var hyi = rebind[Scalar[DTYPE]](
                model[0, body_off_i + BODY_IDX_HALF_Y]
            )
            var hzi = rebind[Scalar[DTYPE]](
                model[0, body_off_i + BODY_IDX_HALF_Z]
            )
            var hxj = rebind[Scalar[DTYPE]](
                model[0, body_off_j + BODY_IDX_HALF_X]
            )
            var hyj = rebind[Scalar[DTYPE]](
                model[0, body_off_j + BODY_IDX_HALF_Y]
            )
            var hzj = rebind[Scalar[DTYPE]](
                model[0, body_off_j + BODY_IDX_HALF_Z]
            )

            var dist: Scalar[DTYPE] = 1.0
            var cx: Scalar[DTYPE] = 0
            var cy: Scalar[DTYPE] = 0
            var cz: Scalar[DTYPE] = 0
            var nx: Scalar[DTYPE] = 0
            var ny: Scalar[DTYPE] = 0
            var nz: Scalar[DTYPE] = 1
            var body_a = i
            var body_b = j

            # Dispatch based on geometry pair
            if gi == GEOM_SPHERE and gj == GEOM_SPHERE:
                var result = sphere_sphere[DTYPE](
                    pi_x,
                    pi_y,
                    pi_z,
                    ri,
                    pj_x,
                    pj_y,
                    pj_z,
                    rj,
                )
                dist = result[0]
                cx = result[1]
                cy = result[2]
                cz = result[3]
                nx = result[4]
                ny = result[5]
                nz = result[6]

            elif gi == GEOM_CAPSULE and gj == GEOM_SPHERE:
                var result = capsule_sphere[DTYPE](
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
                dist = result[0]
                cx = result[1]
                cy = result[2]
                cz = result[3]
                nx = result[4]
                ny = result[5]
                nz = result[6]

            elif gi == GEOM_SPHERE and gj == GEOM_CAPSULE:
                var result = capsule_sphere[DTYPE](
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
                dist = result[0]
                cx = result[1]
                cy = result[2]
                cz = result[3]
                nx = -result[4]
                ny = -result[5]
                nz = -result[6]
                body_a = j
                body_b = i

            elif gi == GEOM_CAPSULE and gj == GEOM_CAPSULE:
                var result = capsule_capsule[DTYPE](
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
                dist = result[0]
                cx = result[1]
                cy = result[2]
                cz = result[3]
                nx = result[4]
                ny = result[5]
                nz = result[6]

            elif gi == GEOM_BOX and gj == GEOM_SPHERE:
                var result = box_sphere[DTYPE](
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
                dist = result[0]
                cx = result[1]
                cy = result[2]
                cz = result[3]
                nx = result[4]
                ny = result[5]
                nz = result[6]

            elif gi == GEOM_SPHERE and gj == GEOM_BOX:
                var result = box_sphere[DTYPE](
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
                dist = result[0]
                cx = result[1]
                cy = result[2]
                cz = result[3]
                nx = -result[4]
                ny = -result[5]
                nz = -result[6]
                body_a = j
                body_b = i

            elif gi == GEOM_BOX and gj == GEOM_CAPSULE:
                var result = box_capsule[DTYPE](
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
                dist = result[0]
                cx = result[1]
                cy = result[2]
                cz = result[3]
                nx = result[4]
                ny = result[5]
                nz = result[6]

            elif gi == GEOM_CAPSULE and gj == GEOM_BOX:
                var result = box_capsule[DTYPE](
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
                dist = result[0]
                cx = result[1]
                cy = result[2]
                cz = result[3]
                nx = -result[4]
                ny = -result[5]
                nz = -result[6]
                body_a = j
                body_b = i

            # Store contact if penetrating
            if dist < Scalar[DTYPE](0) and num_contacts < MAX_CONTACTS:
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
                num_contacts += 1

    state[env, meta_off + META_IDX_NUM_CONTACTS] = Scalar[DTYPE](num_contacts)


# =============================================================================
# Normalize Quaternions Kernel
# =============================================================================


@always_inline
fn normalize_qpos_quaternions_gpu[
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
