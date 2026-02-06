"""GPU kernels for Generalized Coordinates (GC) physics engine.

This module contains the main step kernel and integration-specific kernels.
Component kernels are colocated with their CPU counterparts:
- Forward kinematics: kinematics/forward_kinematics.mojo
- Body velocities: kinematics/forward_kinematics.mojo
- Quaternion math: kinematics/quat_math.mojo
- Mass matrix: dynamics/mass_matrix.mojo
- Bias forces: dynamics/bias_forces.mojo
"""

from math import sqrt
from layout import LayoutTensor, Layout

# Import GPU component functions from their colocated modules
from ..kinematics.quat_math import (
    gpu_quat_mul,
    gpu_quat_rotate,
    gpu_axis_angle_to_quat,
    gpu_quat_normalize,
)
from ..kinematics.forward_kinematics import (
    forward_kinematics_gpu,
    compute_body_velocities_gpu,
)
from ..dynamics.mass_matrix import compute_mass_matrix_diagonal_gpu
from ..dynamics.bias_forces import compute_bias_forces_gpu
from ..dynamics.jacobian import compute_cdof_gpu, compute_contact_jacobian_row_gpu
from ..solver.gc_pgs_solver import GcPGSSolver

from .constants import (
    TPB,
    gc_qpos_offset,
    gc_qvel_offset,
    gc_qacc_offset,
    gc_qfrc_offset,
    gc_xpos_offset,
    gc_xquat_offset,
    gc_xvel_offset,
    gc_xangvel_offset,
    gc_contacts_offset,
    gc_contact_offset,
    gc_metadata_offset,
    gc_state_size,
    gc_model_body_offset,
    gc_model_joint_offset,
    gc_model_metadata_offset,
    gc_model_size,
    GC_MODEL_BODY_SIZE,
    GC_MODEL_JOINT_SIZE,
    GC_MODEL_META_SIZE,
    GC_CONTACT_SIZE,
    GC_BODY_IDX_MASS,
    GC_BODY_IDX_INV_MASS,
    GC_BODY_IDX_IXX,
    GC_BODY_IDX_IYY,
    GC_BODY_IDX_IZZ,
    GC_BODY_IDX_POS_X,
    GC_BODY_IDX_POS_Y,
    GC_BODY_IDX_POS_Z,
    GC_BODY_IDX_QUAT_X,
    GC_BODY_IDX_QUAT_Y,
    GC_BODY_IDX_QUAT_Z,
    GC_BODY_IDX_QUAT_W,
    GC_BODY_IDX_PARENT,
    GC_BODY_IDX_RADIUS,
    GC_BODY_IDX_HALF_LENGTH,
    GC_JOINT_IDX_TYPE,
    GC_JOINT_IDX_BODY_ID,
    GC_JOINT_IDX_QPOS_ADR,
    GC_JOINT_IDX_DOF_ADR,
    GC_JOINT_IDX_POS_X,
    GC_JOINT_IDX_POS_Y,
    GC_JOINT_IDX_POS_Z,
    GC_JOINT_IDX_AXIS_X,
    GC_JOINT_IDX_AXIS_Y,
    GC_JOINT_IDX_AXIS_Z,
    GC_MODEL_META_IDX_NBODY,
    GC_MODEL_META_IDX_NJOINT,
    GC_MODEL_META_IDX_GRAVITY_Z,
    GC_MODEL_META_IDX_TIMESTEP,
    GC_MODEL_META_IDX_GROUND_Z,
    GC_MODEL_META_IDX_FRICTION,
    GC_CONTACT_IDX_BODY_A,
    GC_CONTACT_IDX_BODY_B,
    GC_CONTACT_IDX_POS_X,
    GC_CONTACT_IDX_POS_Y,
    GC_CONTACT_IDX_POS_Z,
    GC_CONTACT_IDX_NX,
    GC_CONTACT_IDX_NY,
    GC_CONTACT_IDX_NZ,
    GC_CONTACT_IDX_DIST,
    GC_META_IDX_NUM_CONTACTS,
    GC_JNT_FREE,
    GC_JNT_BALL,
    GC_JNT_SLIDE,
    GC_JNT_HINGE,
    GC_JOINT_IDX_RANGE_MIN,
    GC_JOINT_IDX_RANGE_MAX,
)


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
    var xpos_off = gc_xpos_offset[NQ, NV, NBODY]()
    var xquat_off = gc_xquat_offset[NQ, NV, NBODY]()
    var contacts_off = gc_contacts_offset[NQ, NV, NBODY]()
    var meta_off = gc_metadata_offset[NQ, NV, NBODY, MAX_CONTACTS]()

    var model_meta_off = gc_model_metadata_offset[NBODY, NJOINT]()
    var ground_z = rebind[Scalar[DTYPE]](model[0, model_meta_off + GC_MODEL_META_IDX_GROUND_Z])

    var num_contacts = 0

    for body in range(NBODY):
        var body_off = gc_model_body_offset(body)
        var radius = rebind[Scalar[DTYPE]](model[0, body_off + GC_BODY_IDX_RADIUS])
        var half_length = rebind[Scalar[DTYPE]](model[0, body_off + GC_BODY_IDX_HALF_LENGTH])

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
        var axis_world = gpu_quat_rotate(qx, qy, qz, qw,
            Scalar[DTYPE](0), Scalar[DTYPE](0), Scalar[DTYPE](1))
        var axis_x = axis_world[0]
        var axis_y = axis_world[1]
        var axis_z = axis_world[2]

        # For spheres (half_length = 0), just check center - radius
        if half_length <= Scalar[DTYPE](0.0001):
            var dist = pz - radius - ground_z
            if dist < Scalar[DTYPE](0) and num_contacts < MAX_CONTACTS:
                var c_off = contacts_off + num_contacts * GC_CONTACT_SIZE
                state[env, c_off + GC_CONTACT_IDX_BODY_A] = Scalar[DTYPE](body)
                state[env, c_off + GC_CONTACT_IDX_BODY_B] = Scalar[DTYPE](-1)
                state[env, c_off + GC_CONTACT_IDX_POS_X] = px
                state[env, c_off + GC_CONTACT_IDX_POS_Y] = py
                state[env, c_off + GC_CONTACT_IDX_POS_Z] = ground_z
                state[env, c_off + GC_CONTACT_IDX_NX] = Scalar[DTYPE](0)
                state[env, c_off + GC_CONTACT_IDX_NY] = Scalar[DTYPE](0)
                state[env, c_off + GC_CONTACT_IDX_NZ] = Scalar[DTYPE](1)
                state[env, c_off + GC_CONTACT_IDX_DIST] = dist
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
                var c_off = contacts_off + num_contacts * GC_CONTACT_SIZE
                state[env, c_off + GC_CONTACT_IDX_BODY_A] = Scalar[DTYPE](body)
                state[env, c_off + GC_CONTACT_IDX_BODY_B] = Scalar[DTYPE](-1)
                state[env, c_off + GC_CONTACT_IDX_POS_X] = e1_x
                state[env, c_off + GC_CONTACT_IDX_POS_Y] = e1_y
                state[env, c_off + GC_CONTACT_IDX_POS_Z] = ground_z
                state[env, c_off + GC_CONTACT_IDX_NX] = Scalar[DTYPE](0)
                state[env, c_off + GC_CONTACT_IDX_NY] = Scalar[DTYPE](0)
                state[env, c_off + GC_CONTACT_IDX_NZ] = Scalar[DTYPE](1)
                state[env, c_off + GC_CONTACT_IDX_DIST] = dist1
                num_contacts += 1

            # Check endpoint 2
            if dist2 < Scalar[DTYPE](0) and num_contacts < MAX_CONTACTS:
                var c_off = contacts_off + num_contacts * GC_CONTACT_SIZE
                state[env, c_off + GC_CONTACT_IDX_BODY_A] = Scalar[DTYPE](body)
                state[env, c_off + GC_CONTACT_IDX_BODY_B] = Scalar[DTYPE](-1)
                state[env, c_off + GC_CONTACT_IDX_POS_X] = e2_x
                state[env, c_off + GC_CONTACT_IDX_POS_Y] = e2_y
                state[env, c_off + GC_CONTACT_IDX_POS_Z] = ground_z
                state[env, c_off + GC_CONTACT_IDX_NX] = Scalar[DTYPE](0)
                state[env, c_off + GC_CONTACT_IDX_NY] = Scalar[DTYPE](0)
                state[env, c_off + GC_CONTACT_IDX_NZ] = Scalar[DTYPE](1)
                state[env, c_off + GC_CONTACT_IDX_DIST] = dist2
                num_contacts += 1

    state[env, meta_off + GC_META_IDX_NUM_CONTACTS] = Scalar[DTYPE](num_contacts)


# =============================================================================
# Contact Forces Kernel
# =============================================================================


@always_inline
fn compute_contact_forces_gpu[
    DTYPE: DType,
    NQ: Int,
    NV: Int,
    NBODY: Int,
    NJOINT: Int,
    MAX_CONTACTS: Int,
    STATE_SIZE: Int,
    MODEL_SIZE: Int,
    V_SIZE: Int,
    BATCH: Int,
](
    env: Int,
    state: LayoutTensor[
        DTYPE, Layout.row_major(BATCH, STATE_SIZE), MutAnyOrigin
    ],
    model: LayoutTensor[DTYPE, Layout.row_major(1, MODEL_SIZE), MutAnyOrigin],
    mut qfrc_contact: InlineArray[Scalar[DTYPE], V_SIZE],
):
    """Compute joint-space contact forces with friction."""
    var xpos_off = gc_xpos_offset[NQ, NV, NBODY]()
    var xquat_off = gc_xquat_offset[NQ, NV, NBODY]()
    var xvel_off = gc_xvel_offset[NQ, NV, NBODY]()
    var contacts_off = gc_contacts_offset[NQ, NV, NBODY]()
    var meta_off = gc_metadata_offset[NQ, NV, NBODY, MAX_CONTACTS]()

    var model_meta_off = gc_model_metadata_offset[NBODY, NJOINT]()
    var num_joints = Int(rebind[Scalar[DTYPE]](model[0, model_meta_off + GC_MODEL_META_IDX_NJOINT]))
    var num_contacts = Int(rebind[Scalar[DTYPE]](state[env, meta_off + GC_META_IDX_NUM_CONTACTS]))
    var friction_coef = rebind[Scalar[DTYPE]](model[0, model_meta_off + GC_MODEL_META_IDX_FRICTION])

    # Initialize to zero
    for i in range(NV):
        qfrc_contact[i] = Scalar[DTYPE](0)

    var stiffness: Scalar[DTYPE] = 5000.0
    var damping: Scalar[DTYPE] = 100.0

    for c in range(num_contacts):
        var c_off = contacts_off + c * GC_CONTACT_SIZE
        var body = Int(rebind[Scalar[DTYPE]](state[env, c_off + GC_CONTACT_IDX_BODY_A]))
        var dist = rebind[Scalar[DTYPE]](state[env, c_off + GC_CONTACT_IDX_DIST])
        var nx = rebind[Scalar[DTYPE]](state[env, c_off + GC_CONTACT_IDX_NX])
        var ny = rebind[Scalar[DTYPE]](state[env, c_off + GC_CONTACT_IDX_NY])
        var nz = rebind[Scalar[DTYPE]](state[env, c_off + GC_CONTACT_IDX_NZ])
        var cpx = rebind[Scalar[DTYPE]](state[env, c_off + GC_CONTACT_IDX_POS_X])
        var cpy = rebind[Scalar[DTYPE]](state[env, c_off + GC_CONTACT_IDX_POS_Y])
        var cpz = rebind[Scalar[DTYPE]](state[env, c_off + GC_CONTACT_IDX_POS_Z])

        if dist >= Scalar[DTYPE](0):
            continue

        var depth = -dist

        # Body velocity at contact
        var vx = rebind[Scalar[DTYPE]](state[env, xvel_off + body * 3 + 0])
        var vy = rebind[Scalar[DTYPE]](state[env, xvel_off + body * 3 + 1])
        var vz = rebind[Scalar[DTYPE]](state[env, xvel_off + body * 3 + 2])

        # Normal force (spring-damper)
        var normal_force = stiffness * depth - damping * vz
        if normal_force < Scalar[DTYPE](0):
            normal_force = Scalar[DTYPE](0)

        # Tangential velocity (in XY plane for ground contact)
        var v_tangent_mag = sqrt(vx * vx + vy * vy)

        # Coulomb friction force (opposes tangential velocity)
        var max_friction = friction_coef * normal_force
        var friction_x: Scalar[DTYPE] = Scalar[DTYPE](0)
        var friction_y: Scalar[DTYPE] = Scalar[DTYPE](0)

        if v_tangent_mag > Scalar[DTYPE](1e-6):
            friction_x = -max_friction * (vx / v_tangent_mag)
            friction_y = -max_friction * (vy / v_tangent_mag)

        # Total contact force in world frame
        var total_fx = friction_x
        var total_fy = friction_y
        var total_fz = normal_force

        # Project to joint space
        # Note: A joint affects a body if the body is the joint's body OR a descendant
        for j in range(num_joints):
            var joint_off = gc_model_joint_offset[NBODY](j)
            var jnt_type = Int(rebind[Scalar[DTYPE]](model[0, joint_off + GC_JOINT_IDX_TYPE]))
            var joint_body = Int(rebind[Scalar[DTYPE]](model[0, joint_off + GC_JOINT_IDX_BODY_ID]))
            var dof_adr = Int(rebind[Scalar[DTYPE]](model[0, joint_off + GC_JOINT_IDX_DOF_ADR]))

            # Check if this joint affects the contacted body
            # (body is the joint's body OR a descendant of the joint's body)
            var joint_affects = False
            if body == joint_body:
                joint_affects = True
            else:
                # Check if body is a descendant of joint_body
                var current = body
                while current >= 0:
                    var current_body_off = gc_model_body_offset(current)
                    var current_parent = Int(rebind[Scalar[DTYPE]](model[0, current_body_off + GC_BODY_IDX_PARENT]))
                    if current_parent == joint_body:
                        joint_affects = True
                        break
                    current = current_parent

            if not joint_affects:
                continue

            var body_off = gc_model_body_offset(joint_body)
            var parent = Int(rebind[Scalar[DTYPE]](model[0, body_off + GC_BODY_IDX_PARENT]))

            var jpos_x = rebind[Scalar[DTYPE]](model[0, joint_off + GC_JOINT_IDX_POS_X])
            var jpos_y = rebind[Scalar[DTYPE]](model[0, joint_off + GC_JOINT_IDX_POS_Y])
            var jpos_z = rebind[Scalar[DTYPE]](model[0, joint_off + GC_JOINT_IDX_POS_Z])
            var axis_x = rebind[Scalar[DTYPE]](model[0, joint_off + GC_JOINT_IDX_AXIS_X])
            var axis_y = rebind[Scalar[DTYPE]](model[0, joint_off + GC_JOINT_IDX_AXIS_Y])
            var axis_z = rebind[Scalar[DTYPE]](model[0, joint_off + GC_JOINT_IDX_AXIS_Z])

            if jnt_type == GC_JNT_HINGE:
                var jpos_world_x = jpos_x
                var jpos_world_y = jpos_y
                var jpos_world_z = jpos_z

                if parent >= 0:
                    var ppx = rebind[Scalar[DTYPE]](state[env, xpos_off + parent * 3 + 0])
                    var ppy = rebind[Scalar[DTYPE]](state[env, xpos_off + parent * 3 + 1])
                    var ppz = rebind[Scalar[DTYPE]](state[env, xpos_off + parent * 3 + 2])
                    var pqx = rebind[Scalar[DTYPE]](state[env, xquat_off + parent * 4 + 0])
                    var pqy = rebind[Scalar[DTYPE]](state[env, xquat_off + parent * 4 + 1])
                    var pqz = rebind[Scalar[DTYPE]](state[env, xquat_off + parent * 4 + 2])
                    var pqw = rebind[Scalar[DTYPE]](state[env, xquat_off + parent * 4 + 3])

                    var rotated = gpu_quat_rotate(pqx, pqy, pqz, pqw, jpos_x, jpos_y, jpos_z)
                    jpos_world_x = ppx + rotated[0]
                    jpos_world_y = ppy + rotated[1]
                    jpos_world_z = ppz + rotated[2]

                var rx = cpx - jpos_world_x
                var ry = cpy - jpos_world_y
                var rz = cpz - jpos_world_z

                # Torque = r x F (with friction)
                var tau_x = ry * total_fz - rz * total_fy
                var tau_y = rz * total_fx - rx * total_fz
                var tau_z = rx * total_fy - ry * total_fx

                var axis_world_x = axis_x
                var axis_world_y = axis_y
                var axis_world_z = axis_z
                if parent >= 0:
                    var pqx = rebind[Scalar[DTYPE]](state[env, xquat_off + parent * 4 + 0])
                    var pqy = rebind[Scalar[DTYPE]](state[env, xquat_off + parent * 4 + 1])
                    var pqz = rebind[Scalar[DTYPE]](state[env, xquat_off + parent * 4 + 2])
                    var pqw = rebind[Scalar[DTYPE]](state[env, xquat_off + parent * 4 + 3])
                    var rotated = gpu_quat_rotate(pqx, pqy, pqz, pqw, axis_x, axis_y, axis_z)
                    axis_world_x = rotated[0]
                    axis_world_y = rotated[1]
                    axis_world_z = rotated[2]

                var tau_joint = tau_x * axis_world_x + tau_y * axis_world_y + tau_z * axis_world_z
                qfrc_contact[dof_adr] = qfrc_contact[dof_adr] + tau_joint

            elif jnt_type == GC_JNT_SLIDE:
                var axis_world_x = axis_x
                var axis_world_y = axis_y
                var axis_world_z = axis_z
                if parent >= 0:
                    var pqx = rebind[Scalar[DTYPE]](state[env, xquat_off + parent * 4 + 0])
                    var pqy = rebind[Scalar[DTYPE]](state[env, xquat_off + parent * 4 + 1])
                    var pqz = rebind[Scalar[DTYPE]](state[env, xquat_off + parent * 4 + 2])
                    var pqw = rebind[Scalar[DTYPE]](state[env, xquat_off + parent * 4 + 3])
                    var rotated = gpu_quat_rotate(pqx, pqy, pqz, pqw, axis_x, axis_y, axis_z)
                    axis_world_x = rotated[0]
                    axis_world_y = rotated[1]
                    axis_world_z = rotated[2]

                # Project total force (with friction) onto axis
                var f_joint = total_fx * axis_world_x + total_fy * axis_world_y + total_fz * axis_world_z
                qfrc_contact[dof_adr] = qfrc_contact[dof_adr] + f_joint

            elif jnt_type == GC_JNT_FREE:
                qfrc_contact[dof_adr + 0] = qfrc_contact[dof_adr + 0] + total_fx
                qfrc_contact[dof_adr + 1] = qfrc_contact[dof_adr + 1] + total_fy
                qfrc_contact[dof_adr + 2] = qfrc_contact[dof_adr + 2] + total_fz


# =============================================================================
# Integration Kernel
# =============================================================================


@always_inline
fn integrate_gc_gpu[
    DTYPE: DType,
    NQ: Int,
    NV: Int,
    NBODY: Int,
    NJOINT: Int,
    MAX_CONTACTS: Int,
    STATE_SIZE: Int,
    MODEL_SIZE: Int,
    V_SIZE: Int,
    BATCH: Int,
](
    env: Int,
    state: LayoutTensor[
        DTYPE, Layout.row_major(BATCH, STATE_SIZE), MutAnyOrigin
    ],
    model: LayoutTensor[DTYPE, Layout.row_major(1, MODEL_SIZE), MutAnyOrigin],
    M_diag: InlineArray[Scalar[DTYPE], V_SIZE],
    bias: InlineArray[Scalar[DTYPE], V_SIZE],
    qfrc_contact: InlineArray[Scalar[DTYPE], V_SIZE],
):
    """Integrate qvel and qpos."""
    var qpos_off = gc_qpos_offset[NQ, NV]()
    var qvel_off = gc_qvel_offset[NQ, NV]()
    var qacc_off = gc_qacc_offset[NQ, NV]()
    var qfrc_off = gc_qfrc_offset[NQ, NV]()

    var model_meta_off = gc_model_metadata_offset[NBODY, NJOINT]()
    var dt = rebind[Scalar[DTYPE]](model[0, model_meta_off + GC_MODEL_META_IDX_TIMESTEP])

    # Solve M * qacc = qfrc + qfrc_contact - bias
    for i in range(NV):
        var f_net = rebind[Scalar[DTYPE]](state[env, qfrc_off + i]) + qfrc_contact[i] - bias[i]
        var m_ii = M_diag[i]
        var qacc: Scalar[DTYPE] = 0
        if m_ii > Scalar[DTYPE](1e-10):
            qacc = f_net / m_ii
        state[env, qacc_off + i] = qacc

    # Integrate: qvel += qacc * dt
    for i in range(NV):
        var qvel = rebind[Scalar[DTYPE]](state[env, qvel_off + i])
        var qacc = rebind[Scalar[DTYPE]](state[env, qacc_off + i])
        state[env, qvel_off + i] = qvel + qacc * dt

    # Integrate: qpos += qvel * dt (for simple joints)
    for i in range(NQ):
        if i < NV:
            var qpos = rebind[Scalar[DTYPE]](state[env, qpos_off + i])
            var qvel = rebind[Scalar[DTYPE]](state[env, qvel_off + i])
            state[env, qpos_off + i] = qpos + qvel * dt


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
    var qpos_off = gc_qpos_offset[NQ, NV]()

    var model_meta_off = gc_model_metadata_offset[NBODY, NJOINT]()
    var num_joints = Int(rebind[Scalar[DTYPE]](model[0, model_meta_off + GC_MODEL_META_IDX_NJOINT]))

    for j in range(num_joints):
        var joint_off = gc_model_joint_offset[NBODY](j)
        var jnt_type = Int(rebind[Scalar[DTYPE]](model[0, joint_off + GC_JOINT_IDX_TYPE]))
        var qpos_adr = Int(rebind[Scalar[DTYPE]](model[0, joint_off + GC_JOINT_IDX_QPOS_ADR]))

        if jnt_type == GC_JNT_FREE:
            var qx = rebind[Scalar[DTYPE]](state[env, qpos_off + qpos_adr + 3])
            var qy = rebind[Scalar[DTYPE]](state[env, qpos_off + qpos_adr + 4])
            var qz = rebind[Scalar[DTYPE]](state[env, qpos_off + qpos_adr + 5])
            var qw = rebind[Scalar[DTYPE]](state[env, qpos_off + qpos_adr + 6])

            var normalized = gpu_quat_normalize(qx, qy, qz, qw)
            state[env, qpos_off + qpos_adr + 3] = normalized[0]
            state[env, qpos_off + qpos_adr + 4] = normalized[1]
            state[env, qpos_off + qpos_adr + 5] = normalized[2]
            state[env, qpos_off + qpos_adr + 6] = normalized[3]

        elif jnt_type == GC_JNT_BALL:
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
# Joint Limit Enforcement Kernel
# =============================================================================


@always_inline
fn enforce_joint_limits_gpu[
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
    """Enforce joint position limits for HINGE and SLIDE joints.

    When a joint exceeds its limit:
    1. Clamp position to the limit
    2. Zero velocity if moving further into the limit
    """
    var qpos_off = gc_qpos_offset[NQ, NV]()
    var qvel_off = gc_qvel_offset[NQ, NV]()

    var model_meta_off = gc_model_metadata_offset[NBODY, NJOINT]()
    var num_joints = Int(rebind[Scalar[DTYPE]](model[0, model_meta_off + GC_MODEL_META_IDX_NJOINT]))

    for j in range(num_joints):
        var joint_off = gc_model_joint_offset[NBODY](j)
        var jnt_type = Int(rebind[Scalar[DTYPE]](model[0, joint_off + GC_JOINT_IDX_TYPE]))

        # Only enforce limits for HINGE and SLIDE joints
        if jnt_type == GC_JNT_HINGE or jnt_type == GC_JNT_SLIDE:
            var qpos_adr = Int(rebind[Scalar[DTYPE]](model[0, joint_off + GC_JOINT_IDX_QPOS_ADR]))
            var dof_adr = Int(rebind[Scalar[DTYPE]](model[0, joint_off + GC_JOINT_IDX_DOF_ADR]))
            var range_min = rebind[Scalar[DTYPE]](model[0, joint_off + GC_JOINT_IDX_RANGE_MIN])
            var range_max = rebind[Scalar[DTYPE]](model[0, joint_off + GC_JOINT_IDX_RANGE_MAX])

            var pos = rebind[Scalar[DTYPE]](state[env, qpos_off + qpos_adr])
            var vel = rebind[Scalar[DTYPE]](state[env, qvel_off + dof_adr])

            # Check lower limit
            if pos < range_min:
                state[env, qpos_off + qpos_adr] = range_min
                # Zero velocity if moving into the limit
                if vel < Scalar[DTYPE](0):
                    state[env, qvel_off + dof_adr] = Scalar[DTYPE](0)

            # Check upper limit
            elif pos > range_max:
                state[env, qpos_off + qpos_adr] = range_max
                # Zero velocity if moving into the limit
                if vel > Scalar[DTYPE](0):
                    state[env, qvel_off + dof_adr] = Scalar[DTYPE](0)


# =============================================================================
# Complete Step Kernel
# =============================================================================


fn _max_one[n: Int]() -> Int:
    """Helper to ensure V_SIZE is at least 1."""
    if n > 0:
        return n
    return 1


@always_inline
fn step_gc_kernel[
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
    """Complete GC physics step for one environment.

    Pipeline:
    1. Forward kinematics
    2. Compute body velocities
    3. Detect ground contacts
    4. Compute mass matrix diagonal
    5. Compute bias forces
    6. Compute contact forces
    7. Integrate (solve and update qvel, qpos)
    8. Normalize quaternions
    """
    comptime V_SIZE = _max_one[NV]()

    var M_diag = InlineArray[Scalar[DTYPE], V_SIZE](uninitialized=True)
    var bias = InlineArray[Scalar[DTYPE], V_SIZE](uninitialized=True)
    var qfrc_contact = InlineArray[Scalar[DTYPE], V_SIZE](uninitialized=True)

    for i in range(V_SIZE):
        M_diag[i] = Scalar[DTYPE](0)
        bias[i] = Scalar[DTYPE](0)
        qfrc_contact[i] = Scalar[DTYPE](0)

    # 1. Forward kinematics
    forward_kinematics_gpu[
        DTYPE, NQ, NV, NBODY, NJOINT, MAX_CONTACTS, STATE_SIZE, MODEL_SIZE, BATCH
    ](env, state, model)

    # 2. Compute body velocities
    compute_body_velocities_gpu[
        DTYPE, NQ, NV, NBODY, NJOINT, MAX_CONTACTS, STATE_SIZE, MODEL_SIZE, BATCH
    ](env, state, model)

    # 3. Detect ground contacts
    detect_ground_contacts_gpu[
        DTYPE, NQ, NV, NBODY, NJOINT, MAX_CONTACTS, STATE_SIZE, MODEL_SIZE, BATCH
    ](env, state, model)

    # 4. Compute mass matrix diagonal
    compute_mass_matrix_diagonal_gpu[
        DTYPE, NQ, NV, NBODY, NJOINT, MAX_CONTACTS, STATE_SIZE, MODEL_SIZE, V_SIZE, BATCH
    ](env, state, model, M_diag)

    # 5. Compute bias forces
    compute_bias_forces_gpu[
        DTYPE, NQ, NV, NBODY, NJOINT, MAX_CONTACTS, STATE_SIZE, MODEL_SIZE, V_SIZE, BATCH
    ](env, state, model, bias)

    # 6. Compute contact forces
    compute_contact_forces_gpu[
        DTYPE, NQ, NV, NBODY, NJOINT, MAX_CONTACTS, STATE_SIZE, MODEL_SIZE, V_SIZE, BATCH
    ](env, state, model, qfrc_contact)

    # 7. Integrate
    integrate_gc_gpu[
        DTYPE, NQ, NV, NBODY, NJOINT, MAX_CONTACTS, STATE_SIZE, MODEL_SIZE, V_SIZE, BATCH
    ](env, state, model, M_diag, bias, qfrc_contact)

    # 8. Normalize quaternions
    normalize_qpos_quaternions_gpu[
        DTYPE, NQ, NV, NBODY, NJOINT, MAX_CONTACTS, STATE_SIZE, MODEL_SIZE, BATCH
    ](env, state, model)

    # 9. Enforce joint limits
    enforce_joint_limits_gpu[
        DTYPE, NQ, NV, NBODY, NJOINT, MAX_CONTACTS, STATE_SIZE, MODEL_SIZE, BATCH
    ](env, state, model)


# =============================================================================
# Constraint-Based Step Kernel (replaces penalty springs with PGS)
# =============================================================================


@always_inline
fn step_gc_constraint_kernel[
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
    """Complete GC physics step with constraint-based contact solving.

    Pipeline:
    1. Forward kinematics (qpos -> xpos, xquat)
    2. Compute body velocities (qvel -> xvel, xangvel)
    3. Detect ground contacts
    4. Compute mass matrix diagonal
    5. Compute bias forces
    6. Compute cdof (spatial motion axes per DOF)
    7. Compute unconstrained acceleration and predict velocity
    8. PGS constraint solve (modifies predicted velocity)
    9. Write back constrained velocity, integrate position
    10. Normalize quaternions
    11. Enforce joint limits
    """
    comptime V_SIZE = _max_one[NV]()
    comptime CDOF_SIZE = _max_one[NV * 6]()

    var M_diag = InlineArray[Scalar[DTYPE], V_SIZE](uninitialized=True)
    var bias = InlineArray[Scalar[DTYPE], V_SIZE](uninitialized=True)
    var cdof = InlineArray[Scalar[DTYPE], CDOF_SIZE](uninitialized=True)

    for i in range(V_SIZE):
        M_diag[i] = Scalar[DTYPE](0)
        bias[i] = Scalar[DTYPE](0)
    for i in range(CDOF_SIZE):
        cdof[i] = Scalar[DTYPE](0)

    # 1. Forward kinematics
    forward_kinematics_gpu[
        DTYPE, NQ, NV, NBODY, NJOINT, MAX_CONTACTS, STATE_SIZE, MODEL_SIZE, BATCH
    ](env, state, model)

    # 2. Compute body velocities
    compute_body_velocities_gpu[
        DTYPE, NQ, NV, NBODY, NJOINT, MAX_CONTACTS, STATE_SIZE, MODEL_SIZE, BATCH
    ](env, state, model)

    # 3. Detect ground contacts
    detect_ground_contacts_gpu[
        DTYPE, NQ, NV, NBODY, NJOINT, MAX_CONTACTS, STATE_SIZE, MODEL_SIZE, BATCH
    ](env, state, model)

    # 4. Compute mass matrix diagonal
    compute_mass_matrix_diagonal_gpu[
        DTYPE, NQ, NV, NBODY, NJOINT, MAX_CONTACTS, STATE_SIZE, MODEL_SIZE, V_SIZE, BATCH
    ](env, state, model, M_diag)

    # 5. Compute bias forces
    compute_bias_forces_gpu[
        DTYPE, NQ, NV, NBODY, NJOINT, MAX_CONTACTS, STATE_SIZE, MODEL_SIZE, V_SIZE, BATCH
    ](env, state, model, bias)

    # 6. Compute cdof (spatial motion axes per DOF)
    compute_cdof_gpu[
        DTYPE, NQ, NV, NBODY, NJOINT, MAX_CONTACTS, STATE_SIZE, MODEL_SIZE, CDOF_SIZE, BATCH
    ](env, state, model, cdof)

    # 7. Compute unconstrained acceleration and predict velocity
    var qvel_off = gc_qvel_offset[NQ, NV]()
    var qacc_off = gc_qacc_offset[NQ, NV]()
    var qfrc_off = gc_qfrc_offset[NQ, NV]()
    var model_meta_off = gc_model_metadata_offset[NBODY, NJOINT]()
    var dt = rebind[Scalar[DTYPE]](model[0, model_meta_off + GC_MODEL_META_IDX_TIMESTEP])

    var qvel_pred = InlineArray[Scalar[DTYPE], V_SIZE](uninitialized=True)
    for i in range(NV):
        var qfrc = rebind[Scalar[DTYPE]](state[env, qfrc_off + i])
        var f_net = qfrc - bias[i]
        var qacc: Scalar[DTYPE] = 0
        if M_diag[i] > Scalar[DTYPE](1e-10):
            qacc = f_net / M_diag[i]
        state[env, qacc_off + i] = qacc

        var qvel = rebind[Scalar[DTYPE]](state[env, qvel_off + i])
        qvel_pred[i] = qvel + qacc * dt

    # 8. PGS constraint solve (modifies qvel_pred in-place)
    GcPGSSolver.solve_gpu[
        DTYPE, NQ, NV, NBODY, NJOINT, MAX_CONTACTS,
        STATE_SIZE, MODEL_SIZE, V_SIZE, CDOF_SIZE, BATCH,
    ](env, state, model, M_diag, cdof, qvel_pred, dt)

    # 9. Write back constrained velocity and integrate position
    var qpos_off = gc_qpos_offset[NQ, NV]()
    for i in range(NV):
        state[env, qvel_off + i] = qvel_pred[i]

    for i in range(NQ):
        if i < NV:
            var qpos = rebind[Scalar[DTYPE]](state[env, qpos_off + i])
            var qvel = rebind[Scalar[DTYPE]](state[env, qvel_off + i])
            state[env, qpos_off + i] = qpos + qvel * dt

    # 10. Normalize quaternions
    normalize_qpos_quaternions_gpu[
        DTYPE, NQ, NV, NBODY, NJOINT, MAX_CONTACTS, STATE_SIZE, MODEL_SIZE, BATCH
    ](env, state, model)

    # 11. Enforce joint limits
    enforce_joint_limits_gpu[
        DTYPE, NQ, NV, NBODY, NJOINT, MAX_CONTACTS, STATE_SIZE, MODEL_SIZE, BATCH
    ](env, state, model)
