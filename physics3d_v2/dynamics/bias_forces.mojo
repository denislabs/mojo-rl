"""Bias forces computation for Generalized Coordinates engine.

Computes the bias forces b(q, qvel) = C(q, qvel) + g(q) where:
- C(q, qvel): Coriolis and centrifugal forces
- g(q): Gravitational forces

For simple HINGE chains (pendulums), the gravity term dominates:
- bias[i] = m * g * L * sin(theta) for each joint

Reference: Featherstone, "Rigid Body Dynamics Algorithms"
"""

from math import sin, cos
from layout import LayoutTensor, Layout

from ..types import ModelGC, DataGC
from ..joint_types import JNT_HINGE, JNT_SLIDE, JNT_BALL, JNT_FREE
from ..kinematics.quat_math import quat_rotate, gpu_quat_rotate
from ..gpu.constants import (
    gc_xpos_offset,
    gc_xquat_offset,
    gc_model_body_offset,
    gc_model_joint_offset,
    gc_model_metadata_offset,
    GC_BODY_IDX_PARENT,
    GC_BODY_IDX_MASS,
    GC_JOINT_IDX_TYPE,
    GC_JOINT_IDX_BODY_ID,
    GC_JOINT_IDX_DOF_ADR,
    GC_JOINT_IDX_POS_X,
    GC_JOINT_IDX_POS_Y,
    GC_JOINT_IDX_POS_Z,
    GC_JOINT_IDX_AXIS_X,
    GC_JOINT_IDX_AXIS_Y,
    GC_JOINT_IDX_AXIS_Z,
    GC_MODEL_META_IDX_NJOINT,
    GC_MODEL_META_IDX_GRAVITY_Z,
    GC_JNT_FREE,
    GC_JNT_BALL,
    GC_JNT_SLIDE,
    GC_JNT_HINGE,
)


# =============================================================================
# Bias Forces for HINGE-only Chains
# =============================================================================


fn compute_bias_forces[
    DTYPE: DType,
    NQ: Int,
    NV: Int,
    NBODY: Int,
    NJOINT: Int,
    MAX_CONTACTS: Int,
    V_SIZE: Int,
](
    model: ModelGC[DTYPE, NQ, NV, NBODY, NJOINT, MAX_CONTACTS],
    data: DataGC[DTYPE, NQ, NV, NBODY, NJOINT, MAX_CONTACTS],
    mut bias: InlineArray[Scalar[DTYPE], V_SIZE],
):
    """Compute bias forces b(q, qvel) = C(q, qvel) + g(q).

    For each joint, computes the gravitational torque/force and Coriolis terms.
    For HINGE joints in simple chains, this is primarily gravitational torque.

    Args:
        model: Static model configuration with gravity.
        data: Current state (qpos, qvel, xpos, xquat from FK).
        bias: Output bias force vector (NV elements).

    Note: These forces oppose motion, so qacc = M^-1 * (qfrc - bias).
    """
    # Initialize to zero
    for i in range(NV):
        bias[i] = Scalar[DTYPE](0)

    # Get gravity
    var gx = model.gravity[0]
    var gy = model.gravity[1]
    var gz = model.gravity[2]

    # For each joint, compute gravitational contribution
    for j in range(model.num_joints):
        var joint = model.joints[j]
        var body = joint.body_id
        var dof_idx = joint.dof_adr

        if joint.jnt_type == JNT_HINGE:
            # Gravitational torque = r x (m * g)
            # where r is from joint axis to body CoM

            # Get joint position in world frame
            var parent = model.body_parent[body]
            var joint_pos_x = joint.pos_x
            var joint_pos_y = joint.pos_y
            var joint_pos_z = joint.pos_z

            var jpos_world_x = joint_pos_x
            var jpos_world_y = joint_pos_y
            var jpos_world_z = joint_pos_z

            if parent >= 0:
                var parent_px = data.xpos[parent * 3 + 0]
                var parent_py = data.xpos[parent * 3 + 1]
                var parent_pz = data.xpos[parent * 3 + 2]
                var parent_qx = data.xquat[parent * 4 + 0]
                var parent_qy = data.xquat[parent * 4 + 1]
                var parent_qz = data.xquat[parent * 4 + 2]
                var parent_qw = data.xquat[parent * 4 + 3]

                var rotated = quat_rotate(
                    parent_qx, parent_qy, parent_qz, parent_qw,
                    joint_pos_x, joint_pos_y, joint_pos_z
                )
                jpos_world_x = parent_px + rotated[0]
                jpos_world_y = parent_py + rotated[1]
                jpos_world_z = parent_pz + rotated[2]

            # Get joint axis in world frame
            var axis_x = joint.axis_x
            var axis_y = joint.axis_y
            var axis_z = joint.axis_z

            if parent >= 0:
                var parent_qx = data.xquat[parent * 4 + 0]
                var parent_qy = data.xquat[parent * 4 + 1]
                var parent_qz = data.xquat[parent * 4 + 2]
                var parent_qw = data.xquat[parent * 4 + 3]
                var axis_world = quat_rotate(
                    parent_qx, parent_qy, parent_qz, parent_qw,
                    axis_x, axis_y, axis_z
                )
                axis_x = axis_world[0]
                axis_y = axis_world[1]
                axis_z = axis_world[2]

            # Compute gravitational torque from body and all descendants
            var tau_gravity = Scalar[DTYPE](0)

            # Body contribution
            var body_px = data.xpos[body * 3 + 0]
            var body_py = data.xpos[body * 3 + 1]
            var body_pz = data.xpos[body * 3 + 2]
            var mass = model.body_mass[body]

            # Vector from joint to body CoM
            var r_x = body_px - jpos_world_x
            var r_y = body_py - jpos_world_y
            var r_z = body_pz - jpos_world_z

            # Gravity force
            var fg_x = mass * gx
            var fg_y = mass * gy
            var fg_z = mass * gz

            # Torque = r x F
            var tau_x = r_y * fg_z - r_z * fg_y
            var tau_y = r_z * fg_x - r_x * fg_z
            var tau_z = r_x * fg_y - r_y * fg_x

            # Project onto joint axis
            tau_gravity = tau_gravity + (tau_x * axis_x + tau_y * axis_y + tau_z * axis_z)

            # Add contributions from descendant bodies
            for desc_body in range(body + 1, NBODY):
                if _is_descendant(model, desc_body, body):
                    var desc_px = data.xpos[desc_body * 3 + 0]
                    var desc_py = data.xpos[desc_body * 3 + 1]
                    var desc_pz = data.xpos[desc_body * 3 + 2]
                    var desc_mass = model.body_mass[desc_body]

                    var desc_r_x = desc_px - jpos_world_x
                    var desc_r_y = desc_py - jpos_world_y
                    var desc_r_z = desc_pz - jpos_world_z

                    var desc_fg_x = desc_mass * gx
                    var desc_fg_y = desc_mass * gy
                    var desc_fg_z = desc_mass * gz

                    var desc_tau_x = desc_r_y * desc_fg_z - desc_r_z * desc_fg_y
                    var desc_tau_y = desc_r_z * desc_fg_x - desc_r_x * desc_fg_z
                    var desc_tau_z = desc_r_x * desc_fg_y - desc_r_y * desc_fg_x

                    tau_gravity = tau_gravity + (
                        desc_tau_x * axis_x + desc_tau_y * axis_y + desc_tau_z * axis_z
                    )

            # Store bias force (note: sign convention - bias opposes motion)
            bias[dof_idx] = -tau_gravity

        elif joint.jnt_type == JNT_SLIDE:
            # Gravitational force along slide axis

            # Get joint axis in world frame
            var parent = model.body_parent[body]
            var axis_x = joint.axis_x
            var axis_y = joint.axis_y
            var axis_z = joint.axis_z

            if parent >= 0:
                var parent_qx = data.xquat[parent * 4 + 0]
                var parent_qy = data.xquat[parent * 4 + 1]
                var parent_qz = data.xquat[parent * 4 + 2]
                var parent_qw = data.xquat[parent * 4 + 3]
                var axis_world = quat_rotate(
                    parent_qx, parent_qy, parent_qz, parent_qw,
                    axis_x, axis_y, axis_z
                )
                axis_x = axis_world[0]
                axis_y = axis_world[1]
                axis_z = axis_world[2]

            # Total mass of body and descendants
            var total_mass = model.body_mass[body]
            for desc_body in range(body + 1, NBODY):
                if _is_descendant(model, desc_body, body):
                    total_mass = total_mass + model.body_mass[desc_body]

            # Gravity force component along axis
            var fg_dot_axis = total_mass * (gx * axis_x + gy * axis_y + gz * axis_z)

            bias[dof_idx] = -fg_dot_axis

        elif joint.jnt_type == JNT_FREE:
            # FREE joint: direct gravity forces and torques
            var total_mass = model.body_mass[body]

            for desc_body in range(body + 1, NBODY):
                if _is_descendant(model, desc_body, body):
                    total_mass = total_mass + model.body_mass[desc_body]

            # Linear gravity force
            bias[dof_idx + 0] = -total_mass * gx
            bias[dof_idx + 1] = -total_mass * gy
            bias[dof_idx + 2] = -total_mass * gz

            # Angular gravity torque (usually zero for symmetric bodies at CoM)
            bias[dof_idx + 3] = Scalar[DTYPE](0)
            bias[dof_idx + 4] = Scalar[DTYPE](0)
            bias[dof_idx + 5] = Scalar[DTYPE](0)

        elif joint.jnt_type == JNT_BALL:
            # BALL joint: gravity doesn't directly create torque at CoM
            # (but would if CoM offset from joint)
            bias[dof_idx + 0] = Scalar[DTYPE](0)
            bias[dof_idx + 1] = Scalar[DTYPE](0)
            bias[dof_idx + 2] = Scalar[DTYPE](0)


# =============================================================================
# Helper Functions
# =============================================================================


fn _is_descendant[
    DTYPE: DType,
    NQ: Int,
    NV: Int,
    NBODY: Int,
    NJOINT: Int,
    MAX_CONTACTS: Int,
](
    model: ModelGC[DTYPE, NQ, NV, NBODY, NJOINT, MAX_CONTACTS],
    body: Int,
    ancestor: Int,
) -> Bool:
    """Check if body is a descendant of ancestor in the kinematic tree."""
    var current = body
    while current >= 0:
        if model.body_parent[current] == ancestor:
            return True
        current = model.body_parent[current]
    return False


# =============================================================================
# Coriolis Forces (for higher-order accuracy)
# =============================================================================


fn compute_coriolis_forces[
    DTYPE: DType,
    NQ: Int,
    NV: Int,
    NBODY: Int,
    NJOINT: Int,
    MAX_CONTACTS: Int,
](
    model: ModelGC[DTYPE, NQ, NV, NBODY, NJOINT, MAX_CONTACTS],
    data: DataGC[DTYPE, NQ, NV, NBODY, NJOINT, MAX_CONTACTS],
    mut coriolis: InlineArray[Scalar[DTYPE], NV],
):
    """Compute Coriolis and centrifugal forces.

    For simple HINGE chains at low velocities, these are often negligible
    compared to gravity. This is a placeholder for more accurate dynamics.

    Args:
        model: Static model configuration.
        data: Current state with qvel.
        coriolis: Output Coriolis force vector.
    """
    # Initialize to zero (simplified - ignore Coriolis for now)
    for i in range(NV):
        coriolis[i] = Scalar[DTYPE](0)

    # TODO: Implement Coriolis terms for higher accuracy
    # C(q, qvel) involves velocity-dependent coupling between joints
    # For a double pendulum:
    # C[0] = -m2*l1*l2*sin(q1-q0)*qvel[1]^2 - m2*l2*(l1*sin(q1-q0)*qvel[0]*qvel[1])
    # C[1] = m2*l1*l2*sin(q1-q0)*qvel[0]^2


# =============================================================================
# GPU Bias Forces Kernel
# =============================================================================


@always_inline
fn compute_bias_forces_gpu[
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
    mut bias: InlineArray[Scalar[DTYPE], V_SIZE],
):
    """Compute bias forces: C(q, qvel) + g(q) (GPU version).

    For simple systems (HINGE-only chains), this is primarily gravity torques.
    """
    var xpos_off = gc_xpos_offset[NQ, NV, NBODY]()
    var xquat_off = gc_xquat_offset[NQ, NV, NBODY]()

    var model_meta_off = gc_model_metadata_offset[NBODY, NJOINT]()
    var num_joints = Int(rebind[Scalar[DTYPE]](model[0, model_meta_off + GC_MODEL_META_IDX_NJOINT]))
    var gravity_z = rebind[Scalar[DTYPE]](model[0, model_meta_off + GC_MODEL_META_IDX_GRAVITY_Z])

    # Initialize bias to zero
    for i in range(NV):
        bias[i] = Scalar[DTYPE](0)

    # Compute gravity torques for each joint
    for j in range(num_joints):
        var joint_off = gc_model_joint_offset[NBODY](j)

        var jnt_type = Int(rebind[Scalar[DTYPE]](model[0, joint_off + GC_JOINT_IDX_TYPE]))
        var body_id = Int(rebind[Scalar[DTYPE]](model[0, joint_off + GC_JOINT_IDX_BODY_ID]))
        var dof_adr = Int(rebind[Scalar[DTYPE]](model[0, joint_off + GC_JOINT_IDX_DOF_ADR]))

        var body_off = gc_model_body_offset(body_id)
        var parent = Int(rebind[Scalar[DTYPE]](model[0, body_off + GC_BODY_IDX_PARENT]))
        var mass = rebind[Scalar[DTYPE]](model[0, body_off + GC_BODY_IDX_MASS])

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

            var body_px = rebind[Scalar[DTYPE]](state[env, xpos_off + body_id * 3 + 0])
            var body_py = rebind[Scalar[DTYPE]](state[env, xpos_off + body_id * 3 + 1])
            var body_pz = rebind[Scalar[DTYPE]](state[env, xpos_off + body_id * 3 + 2])

            var rx = body_px - jpos_world_x
            var ry = body_py - jpos_world_y
            _ = body_pz - jpos_world_z  # rz not needed for torque calculation

            var fz = mass * gravity_z
            var tau_x = ry * fz
            var tau_y = -rx * fz

            var tau_joint = tau_x * axis_world_x + tau_y * axis_world_y
            bias[dof_adr] = bias[dof_adr] - tau_joint

        elif jnt_type == GC_JNT_SLIDE:
            var axis_world_z = axis_z
            if parent >= 0:
                var pqx = rebind[Scalar[DTYPE]](state[env, xquat_off + parent * 4 + 0])
                var pqy = rebind[Scalar[DTYPE]](state[env, xquat_off + parent * 4 + 1])
                var pqz = rebind[Scalar[DTYPE]](state[env, xquat_off + parent * 4 + 2])
                var pqw = rebind[Scalar[DTYPE]](state[env, xquat_off + parent * 4 + 3])
                var rotated = gpu_quat_rotate(pqx, pqy, pqz, pqw, axis_x, axis_y, axis_z)
                axis_world_z = rotated[2]

            var f_gravity = mass * gravity_z * axis_world_z
            bias[dof_adr] = bias[dof_adr] - f_gravity

        elif jnt_type == GC_JNT_FREE:
            bias[dof_adr + 2] = bias[dof_adr + 2] - mass * gravity_z
