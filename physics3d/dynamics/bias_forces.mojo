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

from ..types import Model, Data, _max_one
from ..joint_types import JNT_HINGE, JNT_SLIDE, JNT_BALL, JNT_FREE
from ..kinematics.quat_math import quat_rotate, gpu_quat_rotate
from ..gpu.constants import (
    xpos_offset,
    xquat_offset,
    xvel_offset,
    xangvel_offset,
    qvel_offset,
    model_body_offset,
    model_joint_offset,
    model_metadata_offset,
    ws_cdof_offset,
    ws_bias_offset,
    BODY_IDX_PARENT,
    BODY_IDX_MASS,
    BODY_IDX_IXX,
    BODY_IDX_IYY,
    BODY_IDX_IZZ,
    JOINT_IDX_TYPE,
    JOINT_IDX_BODY_ID,
    JOINT_IDX_DOF_ADR,
    JOINT_IDX_POS_X,
    JOINT_IDX_POS_Y,
    JOINT_IDX_POS_Z,
    JOINT_IDX_AXIS_X,
    JOINT_IDX_AXIS_Y,
    JOINT_IDX_AXIS_Z,
    MODEL_META_IDX_NJOINT,
    MODEL_META_IDX_GRAVITY_X,
    MODEL_META_IDX_GRAVITY_Y,
    MODEL_META_IDX_GRAVITY_Z,
)

from ..joint_types import (
    JNT_FREE,
    JNT_BALL,
    JNT_SLIDE,
    JNT_HINGE,
)


# =============================================================================
# GPU Helper: Check if body is descendant
# =============================================================================


@always_inline
fn _is_descendant_gpu[
    DTYPE: DType,
    NBODY: Int,
    MODEL_SIZE: Int,
](
    model: LayoutTensor[DTYPE, Layout.row_major(1, MODEL_SIZE), MutAnyOrigin],
    body: Int,
    ancestor: Int,
) -> Bool:
    """GPU-compatible check if body is a descendant of ancestor."""
    var current = body
    while current >= 0:
        var body_off = model_body_offset(current)
        var parent = Int(
            rebind[Scalar[DTYPE]](model[0, body_off + BODY_IDX_PARENT])
        )
        if parent == ancestor:
            return True
        current = parent
    return False


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
    NGEOM: Int = 0,
](
    model: Model[DTYPE, NQ, NV, NBODY, NJOINT, MAX_CONTACTS, NGEOM],
    data: Data[DTYPE, NQ, NV, NBODY, NJOINT, MAX_CONTACTS],
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
                    parent_qx,
                    parent_qy,
                    parent_qz,
                    parent_qw,
                    joint_pos_x,
                    joint_pos_y,
                    joint_pos_z,
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
                    parent_qx,
                    parent_qy,
                    parent_qz,
                    parent_qw,
                    axis_x,
                    axis_y,
                    axis_z,
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
            tau_gravity = tau_gravity + (
                tau_x * axis_x + tau_y * axis_y + tau_z * axis_z
            )

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
                        desc_tau_x * axis_x
                        + desc_tau_y * axis_y
                        + desc_tau_z * axis_z
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
                    parent_qx,
                    parent_qy,
                    parent_qz,
                    parent_qw,
                    axis_x,
                    axis_y,
                    axis_z,
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
            var fg_dot_axis = total_mass * (
                gx * axis_x + gy * axis_y + gz * axis_z
            )

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
    NGEOM: Int = 0,
](
    model: Model[DTYPE, NQ, NV, NBODY, NJOINT, MAX_CONTACTS, NGEOM],
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
    NGEOM: Int = 0,
](
    model: Model[DTYPE, NQ, NV, NBODY, NJOINT, MAX_CONTACTS, NGEOM],
    data: Data[DTYPE, NQ, NV, NBODY, NJOINT, MAX_CONTACTS],
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
    var xpos_off = xpos_offset[NQ, NV, NBODY]()
    var xquat_off = xquat_offset[NQ, NV, NBODY]()

    var model_meta_off = model_metadata_offset[NBODY, NJOINT]()
    var num_joints = Int(
        rebind[Scalar[DTYPE]](model[0, model_meta_off + MODEL_META_IDX_NJOINT])
    )
    var gravity_z = rebind[Scalar[DTYPE]](
        model[0, model_meta_off + MODEL_META_IDX_GRAVITY_Z]
    )

    # Initialize bias to zero
    for i in range(NV):
        bias[i] = Scalar[DTYPE](0)

    # Compute gravity torques for each joint
    for j in range(num_joints):
        var joint_off = model_joint_offset[NBODY](j)

        var jnt_type = Int(
            rebind[Scalar[DTYPE]](model[0, joint_off + JOINT_IDX_TYPE])
        )
        var body_id = Int(
            rebind[Scalar[DTYPE]](model[0, joint_off + JOINT_IDX_BODY_ID])
        )
        var dof_adr = Int(
            rebind[Scalar[DTYPE]](model[0, joint_off + JOINT_IDX_DOF_ADR])
        )

        var body_off = model_body_offset(body_id)
        var parent = Int(
            rebind[Scalar[DTYPE]](model[0, body_off + BODY_IDX_PARENT])
        )
        var mass = rebind[Scalar[DTYPE]](model[0, body_off + BODY_IDX_MASS])

        var jpos_x = rebind[Scalar[DTYPE]](
            model[0, joint_off + JOINT_IDX_POS_X]
        )
        var jpos_y = rebind[Scalar[DTYPE]](
            model[0, joint_off + JOINT_IDX_POS_Y]
        )
        var jpos_z = rebind[Scalar[DTYPE]](
            model[0, joint_off + JOINT_IDX_POS_Z]
        )
        var axis_x = rebind[Scalar[DTYPE]](
            model[0, joint_off + JOINT_IDX_AXIS_X]
        )
        var axis_y = rebind[Scalar[DTYPE]](
            model[0, joint_off + JOINT_IDX_AXIS_Y]
        )
        var axis_z = rebind[Scalar[DTYPE]](
            model[0, joint_off + JOINT_IDX_AXIS_Z]
        )

        if jnt_type == JNT_HINGE:
            var jpos_world_x = jpos_x
            var jpos_world_y = jpos_y
            var jpos_world_z = jpos_z

            if parent >= 0:
                var ppx = rebind[Scalar[DTYPE]](
                    state[env, xpos_off + parent * 3 + 0]
                )
                var ppy = rebind[Scalar[DTYPE]](
                    state[env, xpos_off + parent * 3 + 1]
                )
                var ppz = rebind[Scalar[DTYPE]](
                    state[env, xpos_off + parent * 3 + 2]
                )
                var pqx = rebind[Scalar[DTYPE]](
                    state[env, xquat_off + parent * 4 + 0]
                )
                var pqy = rebind[Scalar[DTYPE]](
                    state[env, xquat_off + parent * 4 + 1]
                )
                var pqz = rebind[Scalar[DTYPE]](
                    state[env, xquat_off + parent * 4 + 2]
                )
                var pqw = rebind[Scalar[DTYPE]](
                    state[env, xquat_off + parent * 4 + 3]
                )

                var rotated = gpu_quat_rotate(
                    pqx, pqy, pqz, pqw, jpos_x, jpos_y, jpos_z
                )
                jpos_world_x = ppx + rotated[0]
                jpos_world_y = ppy + rotated[1]
                jpos_world_z = ppz + rotated[2]

            var axis_world_x = axis_x
            var axis_world_y = axis_y
            var axis_world_z = axis_z
            if parent >= 0:
                var pqx = rebind[Scalar[DTYPE]](
                    state[env, xquat_off + parent * 4 + 0]
                )
                var pqy = rebind[Scalar[DTYPE]](
                    state[env, xquat_off + parent * 4 + 1]
                )
                var pqz = rebind[Scalar[DTYPE]](
                    state[env, xquat_off + parent * 4 + 2]
                )
                var pqw = rebind[Scalar[DTYPE]](
                    state[env, xquat_off + parent * 4 + 3]
                )
                var rotated = gpu_quat_rotate(
                    pqx, pqy, pqz, pqw, axis_x, axis_y, axis_z
                )
                axis_world_x = rotated[0]
                axis_world_y = rotated[1]
                axis_world_z = rotated[2]

            # Compute gravitational torque from body and all descendants (matching CPU)
            var tau_gravity: Scalar[DTYPE] = 0

            # Body contribution
            var body_px = rebind[Scalar[DTYPE]](
                state[env, xpos_off + body_id * 3 + 0]
            )
            var body_py = rebind[Scalar[DTYPE]](
                state[env, xpos_off + body_id * 3 + 1]
            )
            var body_pz = rebind[Scalar[DTYPE]](
                state[env, xpos_off + body_id * 3 + 2]
            )

            var rx = body_px - jpos_world_x
            var ry = body_py - jpos_world_y
            var rz = body_pz - jpos_world_z

            # Gravity force (only z component for gravity_z)
            var fz = mass * gravity_z
            # Torque = r x F (with F = [0, 0, fz])
            var tau_x = ry * fz
            var tau_y = -rx * fz
            # tau_z = 0 (rx*0 - ry*0)

            # Project onto joint axis
            tau_gravity = tau_gravity + (
                tau_x * axis_world_x + tau_y * axis_world_y
            )

            # Add contributions from descendant bodies
            for desc_body in range(body_id + 1, NBODY):
                if _is_descendant_gpu[DTYPE, NBODY, MODEL_SIZE](
                    model, desc_body, body_id
                ):
                    var desc_body_off = model_body_offset(desc_body)
                    var desc_mass = rebind[Scalar[DTYPE]](
                        model[0, desc_body_off + BODY_IDX_MASS]
                    )

                    var desc_px = rebind[Scalar[DTYPE]](
                        state[env, xpos_off + desc_body * 3 + 0]
                    )
                    var desc_py = rebind[Scalar[DTYPE]](
                        state[env, xpos_off + desc_body * 3 + 1]
                    )
                    var desc_pz = rebind[Scalar[DTYPE]](
                        state[env, xpos_off + desc_body * 3 + 2]
                    )

                    var desc_rx = desc_px - jpos_world_x
                    var desc_ry = desc_py - jpos_world_y
                    _ = (
                        desc_pz - jpos_world_z
                    )  # desc_rz not needed for torque calc

                    var desc_fz = desc_mass * gravity_z
                    var desc_tau_x = desc_ry * desc_fz
                    var desc_tau_y = -desc_rx * desc_fz

                    tau_gravity = tau_gravity + (
                        desc_tau_x * axis_world_x + desc_tau_y * axis_world_y
                    )

            bias[dof_adr] = bias[dof_adr] - tau_gravity

        elif jnt_type == JNT_SLIDE:
            var axis_world_x = axis_x
            var axis_world_y = axis_y
            var axis_world_z = axis_z
            if parent >= 0:
                var pqx = rebind[Scalar[DTYPE]](
                    state[env, xquat_off + parent * 4 + 0]
                )
                var pqy = rebind[Scalar[DTYPE]](
                    state[env, xquat_off + parent * 4 + 1]
                )
                var pqz = rebind[Scalar[DTYPE]](
                    state[env, xquat_off + parent * 4 + 2]
                )
                var pqw = rebind[Scalar[DTYPE]](
                    state[env, xquat_off + parent * 4 + 3]
                )
                var rotated = gpu_quat_rotate(
                    pqx, pqy, pqz, pqw, axis_x, axis_y, axis_z
                )
                axis_world_x = rotated[0]
                axis_world_y = rotated[1]
                axis_world_z = rotated[2]

            # Accumulate total mass from body and ALL descendants (matching CPU)
            var total_mass = mass
            for desc_body in range(body_id + 1, NBODY):
                if _is_descendant_gpu[DTYPE, NBODY, MODEL_SIZE](
                    model, desc_body, body_id
                ):
                    var desc_body_off = model_body_offset(desc_body)
                    var desc_mass = rebind[Scalar[DTYPE]](
                        model[0, desc_body_off + BODY_IDX_MASS]
                    )
                    total_mass = total_mass + desc_mass

            # Gravity force component along axis (gravity is [0, 0, gravity_z])
            var f_gravity = total_mass * gravity_z * axis_world_z
            bias[dof_adr] = bias[dof_adr] - f_gravity

        elif jnt_type == JNT_FREE:
            # Accumulate total mass from body and ALL descendants
            var total_mass = mass
            for desc_body in range(body_id + 1, NBODY):
                if _is_descendant_gpu[DTYPE, NBODY, MODEL_SIZE](
                    model, desc_body, body_id
                ):
                    var desc_body_off = model_body_offset(desc_body)
                    var desc_mass = rebind[Scalar[DTYPE]](
                        model[0, desc_body_off + BODY_IDX_MASS]
                    )
                    total_mass = total_mass + desc_mass
            bias[dof_adr + 2] = bias[dof_adr + 2] - total_mass * gravity_z


# =============================================================================
# Full RNE Bias Forces (Gravity + Coriolis + Centrifugal) - CPU
# =============================================================================


fn compute_bias_forces_rne[
    DTYPE: DType,
    NQ: Int,
    NV: Int,
    NBODY: Int,
    NJOINT: Int,
    MAX_CONTACTS: Int,
    V_SIZE: Int,
    CDOF_SIZE: Int,
    NGEOM: Int = 0,
](
    model: Model[DTYPE, NQ, NV, NBODY, NJOINT, MAX_CONTACTS, NGEOM],
    data: Data[DTYPE, NQ, NV, NBODY, NJOINT, MAX_CONTACTS],
    cdof: InlineArray[Scalar[DTYPE], CDOF_SIZE],
    mut bias: InlineArray[Scalar[DTYPE], V_SIZE],
):
    """Compute bias forces using full Recursive Newton-Euler Algorithm.

    Computes b(q, qvel) = C(q, qvel)*qvel + g(q) including:
    - Gravitational forces/torques
    - Coriolis forces (velocity-dependent coupling between joints)
    - Centrifugal forces (velocity-dependent self-terms)

    Algorithm (MuJoCo-style, world frame):
    1. Compute world-frame inertia tensors per body
    2. Forward pass: propagate spatial accelerations (gravity + cdof_dot*qvel)
    3. Compute spatial forces: cfrc = I*cacc + cvel x* (I*cvel)
    4. Backward pass: accumulate forces to parents (with moment transfer)
    5. Project to joint space: bias[d] = cdof[d] . cfrc[body_of_dof[d]]

    Reference: Featherstone "Rigid Body Dynamics Algorithms", Chapter 5
    Reference: MuJoCo engine_core_smooth.c mj_rne()

    Args:
        model: Static model configuration.
        data: Current state (xpos, xquat, xvel, xangvel from FK + body velocities).
        cdof: Spatial motion axes per DOF (6*NV elements, from compute_cdof).
        bias: Output bias force vector (NV elements).
    """
    # Initialize output
    for i in range(NV):
        bias[i] = Scalar[DTYPE](0)

    # Get gravity
    var gx = model.gravity[0]
    var gy = model.gravity[1]
    var gz = model.gravity[2]

    # Per-body spatial acceleration [angular(3), linear(3)]
    comptime BODY6_SIZE = _max_one[NBODY * 6]()
    var cacc = InlineArray[Scalar[DTYPE], BODY6_SIZE](uninitialized=True)
    for i in range(BODY6_SIZE):
        cacc[i] = Scalar[DTYPE](0)

    # Per-body spatial force [torque(3), force(3)]
    var cfrc = InlineArray[Scalar[DTYPE], BODY6_SIZE](uninitialized=True)
    for i in range(BODY6_SIZE):
        cfrc[i] = Scalar[DTYPE](0)

    # Per-body world-frame inertia tensor (symmetric: Ixx, Iyy, Izz, Ixy, Ixz, Iyz)
    var I_world = InlineArray[Scalar[DTYPE], BODY6_SIZE](uninitialized=True)
    for i in range(BODY6_SIZE):
        I_world[i] = Scalar[DTYPE](0)

    # =========================================================================
    # Step 0: Compute world-frame inertia tensors for each body
    # =========================================================================
    for b in range(NBODY):
        var Ixx_local = model.body_inertia[b * 3 + 0]
        var Iyy_local = model.body_inertia[b * 3 + 1]
        var Izz_local = model.body_inertia[b * 3 + 2]

        var qx = data.xquat[b * 4 + 0]
        var qy = data.xquat[b * 4 + 1]
        var qz = data.xquat[b * 4 + 2]
        var qw = data.xquat[b * 4 + 3]

        # Rotation matrix from quaternion
        var r00 = Scalar[DTYPE](1) - Scalar[DTYPE](2) * (qy * qy + qz * qz)
        var r10 = Scalar[DTYPE](2) * (qx * qy + qw * qz)
        var r20 = Scalar[DTYPE](2) * (qx * qz - qw * qy)
        var r01 = Scalar[DTYPE](2) * (qx * qy - qw * qz)
        var r11 = Scalar[DTYPE](1) - Scalar[DTYPE](2) * (qx * qx + qz * qz)
        var r21 = Scalar[DTYPE](2) * (qy * qz + qw * qx)
        var r02 = Scalar[DTYPE](2) * (qx * qz + qw * qy)
        var r12 = Scalar[DTYPE](2) * (qy * qz - qw * qx)
        var r22 = Scalar[DTYPE](1) - Scalar[DTYPE](2) * (qx * qx + qy * qy)

        # I_world = R @ diag(Ixx, Iyy, Izz) @ R^T
        I_world[b * 6 + 0] = (
            Ixx_local * r00 * r00
            + Iyy_local * r01 * r01
            + Izz_local * r02 * r02
        )  # Ixx
        I_world[b * 6 + 1] = (
            Ixx_local * r10 * r10
            + Iyy_local * r11 * r11
            + Izz_local * r12 * r12
        )  # Iyy
        I_world[b * 6 + 2] = (
            Ixx_local * r20 * r20
            + Iyy_local * r21 * r21
            + Izz_local * r22 * r22
        )  # Izz
        I_world[b * 6 + 3] = (
            Ixx_local * r00 * r10
            + Iyy_local * r01 * r11
            + Izz_local * r02 * r12
        )  # Ixy
        I_world[b * 6 + 4] = (
            Ixx_local * r00 * r20
            + Iyy_local * r01 * r21
            + Izz_local * r02 * r22
        )  # Ixz
        I_world[b * 6 + 5] = (
            Ixx_local * r10 * r20
            + Iyy_local * r11 * r21
            + Izz_local * r12 * r22
        )  # Iyz

    # =========================================================================
    # Step 1: Forward pass - compute spatial accelerations (root to leaves)
    #   cacc[body] = cacc[parent] + sum(cdof_dot[d] * qvel[d])
    #   where cdof_dot[d] = cvel[parent] x_motion cdof[d]
    # =========================================================================
    for b in range(NBODY):
        var parent = model.body_parent[b]

        if parent < 0:
            # Root body: gravity as fictitious acceleration
            # cacc = [0, 0, 0, -gx, -gy, -gz]
            cacc[b * 6 + 3] = -gx
            cacc[b * 6 + 4] = -gy
            cacc[b * 6 + 5] = -gz
        else:
            # Copy parent's acceleration
            for k in range(6):
                cacc[b * 6 + k] = cacc[parent * 6 + k]

        # Get parent velocity (zero for root bodies)
        var wp_x: Scalar[DTYPE] = 0
        var wp_y: Scalar[DTYPE] = 0
        var wp_z: Scalar[DTYPE] = 0
        var vp_x: Scalar[DTYPE] = 0
        var vp_y: Scalar[DTYPE] = 0
        var vp_z: Scalar[DTYPE] = 0
        if parent >= 0:
            wp_x = data.xangvel[parent * 3 + 0]
            wp_y = data.xangvel[parent * 3 + 1]
            wp_z = data.xangvel[parent * 3 + 2]
            vp_x = data.xvel[parent * 3 + 0]
            vp_y = data.xvel[parent * 3 + 1]
            vp_z = data.xvel[parent * 3 + 2]

        # Add velocity-dependent acceleration for each DOF of this body
        # cdof_dot[d] = cvel[parent] x_motion cdof[d]
        # (MuJoCo: engine_core_smooth.c mj_comVel, crossMotion)
        for j in range(model.num_joints):
            var joint = model.joints[j]
            if joint.body_id != b:
                continue

            var dof_adr = joint.dof_adr
            var num_dof = 1
            if joint.jnt_type == JNT_FREE:
                num_dof = 6
            elif joint.jnt_type == JNT_BALL:
                num_dof = 3

            for d in range(num_dof):
                var dof = dof_adr + d
                var qdot = data.qvel[dof]

                # cdof components
                var s_ang_x = cdof[dof * 6 + 0]
                var s_ang_y = cdof[dof * 6 + 1]
                var s_ang_z = cdof[dof * 6 + 2]
                var s_lin_x = cdof[dof * 6 + 3]
                var s_lin_y = cdof[dof * 6 + 4]
                var s_lin_z = cdof[dof * 6 + 5]

                # Spatial motion cross: cvel_parent x_m cdof
                # cdot_ang = w_p x s_ang
                var cdot_ang_x = wp_y * s_ang_z - wp_z * s_ang_y
                var cdot_ang_y = wp_z * s_ang_x - wp_x * s_ang_z
                var cdot_ang_z = wp_x * s_ang_y - wp_y * s_ang_x

                # cdot_lin = w_p x s_lin + v_p x s_ang
                var cdot_lin_x = (wp_y * s_lin_z - wp_z * s_lin_y) + (
                    vp_y * s_ang_z - vp_z * s_ang_y
                )
                var cdot_lin_y = (wp_z * s_lin_x - wp_x * s_lin_z) + (
                    vp_z * s_ang_x - vp_x * s_ang_z
                )
                var cdot_lin_z = (wp_x * s_lin_y - wp_y * s_lin_x) + (
                    vp_x * s_ang_y - vp_y * s_ang_x
                )

                # Accumulate: cacc += cdof_dot * qvel
                cacc[b * 6 + 0] = cacc[b * 6 + 0] + cdot_ang_x * qdot
                cacc[b * 6 + 1] = cacc[b * 6 + 1] + cdot_ang_y * qdot
                cacc[b * 6 + 2] = cacc[b * 6 + 2] + cdot_ang_z * qdot
                cacc[b * 6 + 3] = cacc[b * 6 + 3] + cdot_lin_x * qdot
                cacc[b * 6 + 4] = cacc[b * 6 + 4] + cdot_lin_y * qdot
                cacc[b * 6 + 5] = cacc[b * 6 + 5] + cdot_lin_z * qdot

    # =========================================================================
    # Step 2: Compute spatial forces per body
    #   cfrc = I * cacc + cvel x* (I * cvel)
    #
    #   At CoM (offset=0):
    #   - I*v: angular = I_world @ w, linear = mass * v
    #   - v x* (I*v): angular = w x (I_world*w), linear = w x (mass*v)
    # =========================================================================
    for b in range(NBODY):
        var mass = model.body_mass[b]

        # Body velocities
        var wx = data.xangvel[b * 3 + 0]
        var wy = data.xangvel[b * 3 + 1]
        var wz = data.xangvel[b * 3 + 2]
        var vx = data.xvel[b * 3 + 0]
        var vy = data.xvel[b * 3 + 1]
        var vz = data.xvel[b * 3 + 2]

        # Spatial acceleration
        var a_ang_x = cacc[b * 6 + 0]
        var a_ang_y = cacc[b * 6 + 1]
        var a_ang_z = cacc[b * 6 + 2]
        var a_lin_x = cacc[b * 6 + 3]
        var a_lin_y = cacc[b * 6 + 4]
        var a_lin_z = cacc[b * 6 + 5]

        # World-frame inertia tensor (symmetric)
        var Ixx = I_world[b * 6 + 0]
        var Iyy = I_world[b * 6 + 1]
        var Izz = I_world[b * 6 + 2]
        var Ixy = I_world[b * 6 + 3]
        var Ixz = I_world[b * 6 + 4]
        var Iyz = I_world[b * 6 + 5]

        # I * cacc (at CoM, offset=0)
        var Ia_ang_x = Ixx * a_ang_x + Ixy * a_ang_y + Ixz * a_ang_z
        var Ia_ang_y = Ixy * a_ang_x + Iyy * a_ang_y + Iyz * a_ang_z
        var Ia_ang_z = Ixz * a_ang_x + Iyz * a_ang_y + Izz * a_ang_z
        var Ia_lin_x = mass * a_lin_x
        var Ia_lin_y = mass * a_lin_y
        var Ia_lin_z = mass * a_lin_z

        # I * cvel
        var Iw_x = Ixx * wx + Ixy * wy + Ixz * wz
        var Iw_y = Ixy * wx + Iyy * wy + Iyz * wz
        var Iw_z = Ixz * wx + Iyz * wy + Izz * wz

        # cvel x* (I * cvel) = [w x (I*w), w x (m*v)]
        # (v x (m*v) = 0 since v x v = 0)
        var xf_ang_x = wy * Iw_z - wz * Iw_y
        var xf_ang_y = wz * Iw_x - wx * Iw_z
        var xf_ang_z = wx * Iw_y - wy * Iw_x

        var xf_lin_x = wy * (mass * vz) - wz * (mass * vy)
        var xf_lin_y = wz * (mass * vx) - wx * (mass * vz)
        var xf_lin_z = wx * (mass * vy) - wy * (mass * vx)

        # cfrc = I*cacc + cvel x* (I*cvel)
        cfrc[b * 6 + 0] = Ia_ang_x + xf_ang_x
        cfrc[b * 6 + 1] = Ia_ang_y + xf_ang_y
        cfrc[b * 6 + 2] = Ia_ang_z + xf_ang_z
        cfrc[b * 6 + 3] = Ia_lin_x + xf_lin_x
        cfrc[b * 6 + 4] = Ia_lin_y + xf_lin_y
        cfrc[b * 6 + 5] = Ia_lin_z + xf_lin_z

    # =========================================================================
    # Step 3: Backward pass - accumulate forces to parents
    #   When transferring force wrench from child CoM to parent CoM:
    #   tau_parent += tau_child + r x f_child
    #   f_parent += f_child
    #   where r = xpos[child] - xpos[parent]
    # =========================================================================
    for b in range(NBODY - 1, 0, -1):
        var parent = model.body_parent[b]
        if parent < 0:
            continue

        # Offset from parent CoM to child CoM
        var rx = data.xpos[b * 3 + 0] - data.xpos[parent * 3 + 0]
        var ry = data.xpos[b * 3 + 1] - data.xpos[parent * 3 + 1]
        var rz = data.xpos[b * 3 + 2] - data.xpos[parent * 3 + 2]

        # Child force wrench
        var child_tau_x = cfrc[b * 6 + 0]
        var child_tau_y = cfrc[b * 6 + 1]
        var child_tau_z = cfrc[b * 6 + 2]
        var child_f_x = cfrc[b * 6 + 3]
        var child_f_y = cfrc[b * 6 + 4]
        var child_f_z = cfrc[b * 6 + 5]

        # Transfer: tau_parent += tau_child + r x f_child
        cfrc[parent * 6 + 0] = (
            cfrc[parent * 6 + 0]
            + child_tau_x
            + (ry * child_f_z - rz * child_f_y)
        )
        cfrc[parent * 6 + 1] = (
            cfrc[parent * 6 + 1]
            + child_tau_y
            + (rz * child_f_x - rx * child_f_z)
        )
        cfrc[parent * 6 + 2] = (
            cfrc[parent * 6 + 2]
            + child_tau_z
            + (rx * child_f_y - ry * child_f_x)
        )
        # Transfer: f_parent += f_child
        cfrc[parent * 6 + 3] = cfrc[parent * 6 + 3] + child_f_x
        cfrc[parent * 6 + 4] = cfrc[parent * 6 + 4] + child_f_y
        cfrc[parent * 6 + 5] = cfrc[parent * 6 + 5] + child_f_z

    # =========================================================================
    # Step 4: Project to joint space
    #   bias[d] = cdof[d] . cfrc[body_of_dof[d]]
    #   6D dot product: angular . torque + linear . force
    # =========================================================================
    for j in range(model.num_joints):
        var joint = model.joints[j]
        var body = joint.body_id
        var dof_adr = joint.dof_adr
        var num_dof = 1
        if joint.jnt_type == JNT_FREE:
            num_dof = 6
        elif joint.jnt_type == JNT_BALL:
            num_dof = 3

        for d in range(num_dof):
            var dof = dof_adr + d
            bias[dof] = Scalar[DTYPE](0)
            for k in range(6):
                bias[dof] = bias[dof] + cdof[dof * 6 + k] * cfrc[body * 6 + k]


# =============================================================================
# Full RNE Bias Forces (Gravity + Coriolis + Centrifugal) - GPU
# =============================================================================


@always_inline
fn compute_bias_forces_rne_gpu[
    DTYPE: DType,
    NQ: Int,
    NV: Int,
    NBODY: Int,
    NJOINT: Int,
    MAX_CONTACTS: Int,
    STATE_SIZE: Int,
    MODEL_SIZE: Int,
    BATCH: Int,
    WS_SIZE: Int,
](
    env: Int,
    state: LayoutTensor[
        DTYPE, Layout.row_major(BATCH, STATE_SIZE), MutAnyOrigin
    ],
    model: LayoutTensor[DTYPE, Layout.row_major(1, MODEL_SIZE), MutAnyOrigin],
    workspace: LayoutTensor[
        DTYPE, Layout.row_major(BATCH, WS_SIZE), MutAnyOrigin
    ],
):
    """Compute bias forces using full RNE algorithm (GPU version).

    Same algorithm as compute_bias_forces_rne but reads from GPU LayoutTensors.
    Computes b(q, qvel) = C(q, qvel)*qvel + g(q).
    Reads cdof from workspace, writes bias to workspace.
    """
    # Derive workspace pointers
    comptime cdof_idx = ws_cdof_offset()
    comptime bias_idx = ws_bias_offset[NV, NBODY]()

    # Initialize output
    for i in range(NV):
        workspace[env, bias_idx + i] = 0

    # Get gravity from model metadata
    var model_meta_off = model_metadata_offset[NBODY, NJOINT]()
    var gx = rebind[Scalar[DTYPE]](
        model[0, model_meta_off + MODEL_META_IDX_GRAVITY_X]
    )
    var gy = rebind[Scalar[DTYPE]](
        model[0, model_meta_off + MODEL_META_IDX_GRAVITY_Y]
    )
    var gz = rebind[Scalar[DTYPE]](
        model[0, model_meta_off + MODEL_META_IDX_GRAVITY_Z]
    )

    # State buffer offsets
    var xpos_off = xpos_offset[NQ, NV, NBODY]()
    var xquat_off = xquat_offset[NQ, NV, NBODY]()
    var xvel_off = xvel_offset[NQ, NV, NBODY]()
    var xangvel_off = xangvel_offset[NQ, NV, NBODY]()
    var qvel_off = qvel_offset[NQ, NV]()

    # Per-body arrays: spatial acceleration, force, world-frame inertia
    comptime BODY6_SIZE = _max_one[NBODY * 6]()
    var cacc = InlineArray[Scalar[DTYPE], BODY6_SIZE](uninitialized=True)
    for i in range(BODY6_SIZE):
        cacc[i] = Scalar[DTYPE](0)
    var cfrc = InlineArray[Scalar[DTYPE], BODY6_SIZE](uninitialized=True)
    for i in range(BODY6_SIZE):
        cfrc[i] = Scalar[DTYPE](0)
    var I_world = InlineArray[Scalar[DTYPE], BODY6_SIZE](uninitialized=True)
    for i in range(BODY6_SIZE):
        I_world[i] = Scalar[DTYPE](0)

    # =========================================================================
    # Step 0: Compute world-frame inertia tensors
    # =========================================================================
    for b in range(NBODY):
        var body_off = model_body_offset(b)
        var Ixx_local = rebind[Scalar[DTYPE]](model[0, body_off + BODY_IDX_IXX])
        var Iyy_local = rebind[Scalar[DTYPE]](model[0, body_off + BODY_IDX_IYY])
        var Izz_local = rebind[Scalar[DTYPE]](model[0, body_off + BODY_IDX_IZZ])

        var qx = rebind[Scalar[DTYPE]](state[env, xquat_off + b * 4 + 0])
        var qy = rebind[Scalar[DTYPE]](state[env, xquat_off + b * 4 + 1])
        var qz = rebind[Scalar[DTYPE]](state[env, xquat_off + b * 4 + 2])
        var qw = rebind[Scalar[DTYPE]](state[env, xquat_off + b * 4 + 3])

        # Rotation matrix from quaternion
        var r00 = Scalar[DTYPE](1) - Scalar[DTYPE](2) * (qy * qy + qz * qz)
        var r10 = Scalar[DTYPE](2) * (qx * qy + qw * qz)
        var r20 = Scalar[DTYPE](2) * (qx * qz - qw * qy)
        var r01 = Scalar[DTYPE](2) * (qx * qy - qw * qz)
        var r11 = Scalar[DTYPE](1) - Scalar[DTYPE](2) * (qx * qx + qz * qz)
        var r21 = Scalar[DTYPE](2) * (qy * qz + qw * qx)
        var r02 = Scalar[DTYPE](2) * (qx * qz + qw * qy)
        var r12 = Scalar[DTYPE](2) * (qy * qz - qw * qx)
        var r22 = Scalar[DTYPE](1) - Scalar[DTYPE](2) * (qx * qx + qy * qy)

        # I_world = R @ diag(Ixx, Iyy, Izz) @ R^T
        I_world[b * 6 + 0] = (
            Ixx_local * r00 * r00
            + Iyy_local * r01 * r01
            + Izz_local * r02 * r02
        )  # Ixx
        I_world[b * 6 + 1] = (
            Ixx_local * r10 * r10
            + Iyy_local * r11 * r11
            + Izz_local * r12 * r12
        )  # Iyy
        I_world[b * 6 + 2] = (
            Ixx_local * r20 * r20
            + Iyy_local * r21 * r21
            + Izz_local * r22 * r22
        )  # Izz
        I_world[b * 6 + 3] = (
            Ixx_local * r00 * r10
            + Iyy_local * r01 * r11
            + Izz_local * r02 * r12
        )  # Ixy
        I_world[b * 6 + 4] = (
            Ixx_local * r00 * r20
            + Iyy_local * r01 * r21
            + Izz_local * r02 * r22
        )  # Ixz
        I_world[b * 6 + 5] = (
            Ixx_local * r10 * r20
            + Iyy_local * r11 * r21
            + Izz_local * r12 * r22
        )  # Iyz

    # =========================================================================
    # Step 1: Forward pass - spatial accelerations (root to leaves)
    # =========================================================================
    for b in range(NBODY):
        var body_off = model_body_offset(b)
        var parent = Int(
            rebind[Scalar[DTYPE]](model[0, body_off + BODY_IDX_PARENT])
        )

        if parent < 0:
            # Root body: gravity as fictitious acceleration
            cacc[b * 6 + 3] = -gx
            cacc[b * 6 + 4] = -gy
            cacc[b * 6 + 5] = -gz
        else:
            for k in range(6):
                cacc[b * 6 + k] = cacc[parent * 6 + k]

        # Get parent velocity
        var wp_x: Scalar[DTYPE] = 0
        var wp_y: Scalar[DTYPE] = 0
        var wp_z: Scalar[DTYPE] = 0
        var vp_x: Scalar[DTYPE] = 0
        var vp_y: Scalar[DTYPE] = 0
        var vp_z: Scalar[DTYPE] = 0
        if parent >= 0:
            wp_x = rebind[Scalar[DTYPE]](
                state[env, xangvel_off + parent * 3 + 0]
            )
            wp_y = rebind[Scalar[DTYPE]](
                state[env, xangvel_off + parent * 3 + 1]
            )
            wp_z = rebind[Scalar[DTYPE]](
                state[env, xangvel_off + parent * 3 + 2]
            )
            vp_x = rebind[Scalar[DTYPE]](state[env, xvel_off + parent * 3 + 0])
            vp_y = rebind[Scalar[DTYPE]](state[env, xvel_off + parent * 3 + 1])
            vp_z = rebind[Scalar[DTYPE]](state[env, xvel_off + parent * 3 + 2])

        # Add cdof_dot * qvel for each DOF of this body
        for j in range(NJOINT):
            var joint_off = model_joint_offset[NBODY](j)
            var jnt_body = Int(
                rebind[Scalar[DTYPE]](model[0, joint_off + JOINT_IDX_BODY_ID])
            )
            if jnt_body != b:
                continue

            var jnt_type = Int(
                rebind[Scalar[DTYPE]](model[0, joint_off + JOINT_IDX_TYPE])
            )
            var dof_adr = Int(
                rebind[Scalar[DTYPE]](model[0, joint_off + JOINT_IDX_DOF_ADR])
            )
            var num_dof = 1
            if jnt_type == JNT_FREE:
                num_dof = 6
            elif jnt_type == JNT_BALL:
                num_dof = 3

            for d in range(num_dof):
                var dof = dof_adr + d
                var qdot = rebind[Scalar[DTYPE]](state[env, qvel_off + dof])

                # cdof components (read from workspace)
                var s_ang_x = workspace[env, cdof_idx + dof * 6 + 0]
                var s_ang_y = workspace[env, cdof_idx + dof * 6 + 1]
                var s_ang_z = workspace[env, cdof_idx + dof * 6 + 2]
                var s_lin_x = workspace[env, cdof_idx + dof * 6 + 3]
                var s_lin_y = workspace[env, cdof_idx + dof * 6 + 4]
                var s_lin_z = workspace[env, cdof_idx + dof * 6 + 5]

                # Spatial motion cross: cvel_parent x_m cdof
                var cdot_ang_x = wp_y * s_ang_z - wp_z * s_ang_y
                var cdot_ang_y = wp_z * s_ang_x - wp_x * s_ang_z
                var cdot_ang_z = wp_x * s_ang_y - wp_y * s_ang_x

                var cdot_lin_x = (wp_y * s_lin_z - wp_z * s_lin_y) + (
                    vp_y * s_ang_z - vp_z * s_ang_y
                )
                var cdot_lin_y = (wp_z * s_lin_x - wp_x * s_lin_z) + (
                    vp_z * s_ang_x - vp_x * s_ang_z
                )
                var cdot_lin_z = (wp_x * s_lin_y - wp_y * s_lin_x) + (
                    vp_x * s_ang_y - vp_y * s_ang_x
                )

                # Accumulate: cacc += cdof_dot * qvel
                cacc[b * 6 + 0] = cacc[b * 6 + 0] + rebind[Scalar[DTYPE]](
                    cdot_ang_x
                ) * rebind[Scalar[DTYPE]](qdot)
                cacc[b * 6 + 1] = cacc[b * 6 + 1] + rebind[Scalar[DTYPE]](
                    cdot_ang_y
                ) * rebind[Scalar[DTYPE]](qdot)
                cacc[b * 6 + 2] = cacc[b * 6 + 2] + rebind[Scalar[DTYPE]](
                    cdot_ang_z
                ) * rebind[Scalar[DTYPE]](qdot)
                cacc[b * 6 + 3] = cacc[b * 6 + 3] + rebind[Scalar[DTYPE]](
                    cdot_lin_x
                ) * rebind[Scalar[DTYPE]](qdot)
                cacc[b * 6 + 4] = cacc[b * 6 + 4] + rebind[Scalar[DTYPE]](
                    cdot_lin_y
                ) * rebind[Scalar[DTYPE]](qdot)
                cacc[b * 6 + 5] = cacc[b * 6 + 5] + rebind[Scalar[DTYPE]](
                    cdot_lin_z
                ) * rebind[Scalar[DTYPE]](qdot)

    # =========================================================================
    # Step 2: Compute spatial forces per body
    #   cfrc = I * cacc + cvel x* (I * cvel)
    # =========================================================================
    for b in range(NBODY):
        var body_off = model_body_offset(b)
        var mass = rebind[Scalar[DTYPE]](model[0, body_off + BODY_IDX_MASS])

        # Body velocities from state buffer
        var wx = rebind[Scalar[DTYPE]](state[env, xangvel_off + b * 3 + 0])
        var wy = rebind[Scalar[DTYPE]](state[env, xangvel_off + b * 3 + 1])
        var wz = rebind[Scalar[DTYPE]](state[env, xangvel_off + b * 3 + 2])
        var vx = rebind[Scalar[DTYPE]](state[env, xvel_off + b * 3 + 0])
        var vy = rebind[Scalar[DTYPE]](state[env, xvel_off + b * 3 + 1])
        var vz = rebind[Scalar[DTYPE]](state[env, xvel_off + b * 3 + 2])

        # Spatial acceleration
        var a_ang_x = cacc[b * 6 + 0]
        var a_ang_y = cacc[b * 6 + 1]
        var a_ang_z = cacc[b * 6 + 2]
        var a_lin_x = cacc[b * 6 + 3]
        var a_lin_y = cacc[b * 6 + 4]
        var a_lin_z = cacc[b * 6 + 5]

        # World-frame inertia (symmetric)
        var Ixx = I_world[b * 6 + 0]
        var Iyy = I_world[b * 6 + 1]
        var Izz = I_world[b * 6 + 2]
        var Ixy = I_world[b * 6 + 3]
        var Ixz = I_world[b * 6 + 4]
        var Iyz = I_world[b * 6 + 5]

        # I * cacc
        var Ia_ang_x = Ixx * a_ang_x + Ixy * a_ang_y + Ixz * a_ang_z
        var Ia_ang_y = Ixy * a_ang_x + Iyy * a_ang_y + Iyz * a_ang_z
        var Ia_ang_z = Ixz * a_ang_x + Iyz * a_ang_y + Izz * a_ang_z
        var Ia_lin_x = mass * a_lin_x
        var Ia_lin_y = mass * a_lin_y
        var Ia_lin_z = mass * a_lin_z

        # I * cvel
        var Iw_x = Ixx * wx + Ixy * wy + Ixz * wz
        var Iw_y = Ixy * wx + Iyy * wy + Iyz * wz
        var Iw_z = Ixz * wx + Iyz * wy + Izz * wz

        # cvel x* (I * cvel) = [w x (I*w), w x (m*v)]
        var xf_ang_x = wy * Iw_z - wz * Iw_y
        var xf_ang_y = wz * Iw_x - wx * Iw_z
        var xf_ang_z = wx * Iw_y - wy * Iw_x
        var xf_lin_x = wy * (mass * vz) - wz * (mass * vy)
        var xf_lin_y = wz * (mass * vx) - wx * (mass * vz)
        var xf_lin_z = wx * (mass * vy) - wy * (mass * vx)

        # cfrc = I*cacc + cvel x* (I*cvel)
        cfrc[b * 6 + 0] = Ia_ang_x + xf_ang_x
        cfrc[b * 6 + 1] = Ia_ang_y + xf_ang_y
        cfrc[b * 6 + 2] = Ia_ang_z + xf_ang_z
        cfrc[b * 6 + 3] = Ia_lin_x + xf_lin_x
        cfrc[b * 6 + 4] = Ia_lin_y + xf_lin_y
        cfrc[b * 6 + 5] = Ia_lin_z + xf_lin_z

    # =========================================================================
    # Step 3: Backward pass - accumulate forces to parents
    # =========================================================================
    for b in range(NBODY - 1, 0, -1):
        var body_off = model_body_offset(b)
        var parent = Int(
            rebind[Scalar[DTYPE]](model[0, body_off + BODY_IDX_PARENT])
        )
        if parent < 0:
            continue

        # Offset from parent CoM to child CoM
        var rx = rebind[Scalar[DTYPE]](
            state[env, xpos_off + b * 3 + 0]
        ) - rebind[Scalar[DTYPE]](state[env, xpos_off + parent * 3 + 0])
        var ry = rebind[Scalar[DTYPE]](
            state[env, xpos_off + b * 3 + 1]
        ) - rebind[Scalar[DTYPE]](state[env, xpos_off + parent * 3 + 1])
        var rz = rebind[Scalar[DTYPE]](
            state[env, xpos_off + b * 3 + 2]
        ) - rebind[Scalar[DTYPE]](state[env, xpos_off + parent * 3 + 2])

        var child_tau_x = cfrc[b * 6 + 0]
        var child_tau_y = cfrc[b * 6 + 1]
        var child_tau_z = cfrc[b * 6 + 2]
        var child_f_x = cfrc[b * 6 + 3]
        var child_f_y = cfrc[b * 6 + 4]
        var child_f_z = cfrc[b * 6 + 5]

        # Transfer: tau_parent += tau_child + r x f_child
        cfrc[parent * 6 + 0] = (
            cfrc[parent * 6 + 0]
            + child_tau_x
            + (ry * child_f_z - rz * child_f_y)
        )
        cfrc[parent * 6 + 1] = (
            cfrc[parent * 6 + 1]
            + child_tau_y
            + (rz * child_f_x - rx * child_f_z)
        )
        cfrc[parent * 6 + 2] = (
            cfrc[parent * 6 + 2]
            + child_tau_z
            + (rx * child_f_y - ry * child_f_x)
        )
        # Transfer: f_parent += f_child
        cfrc[parent * 6 + 3] = cfrc[parent * 6 + 3] + child_f_x
        cfrc[parent * 6 + 4] = cfrc[parent * 6 + 4] + child_f_y
        cfrc[parent * 6 + 5] = cfrc[parent * 6 + 5] + child_f_z

    # =========================================================================
    # Step 4: Project to joint space
    #   bias[d] = cdof[d] . cfrc[body_of_dof[d]]
    # =========================================================================
    for j in range(NJOINT):
        var joint_off = model_joint_offset[NBODY](j)
        var jnt_type = Int(
            rebind[Scalar[DTYPE]](model[0, joint_off + JOINT_IDX_TYPE])
        )
        var body = Int(
            rebind[Scalar[DTYPE]](model[0, joint_off + JOINT_IDX_BODY_ID])
        )
        var dof_adr = Int(
            rebind[Scalar[DTYPE]](model[0, joint_off + JOINT_IDX_DOF_ADR])
        )
        var num_dof = 1
        if jnt_type == JNT_FREE:
            num_dof = 6
        elif jnt_type == JNT_BALL:
            num_dof = 3

        for d in range(num_dof):
            var dof = dof_adr + d
            workspace[env, bias_idx + dof] = 0
            for k in range(6):
                workspace[env, bias_idx + dof] = (
                    workspace[env, bias_idx + dof]
                    + workspace[env, cdof_idx + dof * 6 + k]
                    * cfrc[body * 6 + k]
                )
