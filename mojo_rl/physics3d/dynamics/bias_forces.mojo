"""Bias forces computation for Generalized Coordinates engine.

Computes the bias forces b(q, qvel) = C(q, qvel) + g(q) where:
- C(q, qvel): Coriolis and centrifugal forces
- g(q): Gravitational forces

For simple HINGE chains (pendulums), the gravity term dominates:
- bias[i] = m * g * L * sin(theta) for each joint

Reference: Featherstone, "Rigid Body Dynamics Algorithms"
"""

from std.math import sin, cos
from layout import LayoutTensor, Layout

from ..types import Model, Data, _max_one, ConeType
from ..joint_types import JNT_HINGE, JNT_SLIDE, JNT_BALL, JNT_FREE
from ..kinematics.quat_math import (
    quat_rotate,
    quat_mul,
    gpu_quat_rotate,
    gpu_quat_mul,
)
from ..gpu.constants import (
    xpos_offset,
    xquat_offset,
    xipos_offset,
    qvel_offset,
    model_body_offset,
    model_joint_offset,
    model_metadata_offset,
    ws_cdof_offset,
    ws_crb_offset,
    ws_bias_offset,
    BODY_IDX_PARENT,
    BODY_IDX_MASS,
    BODY_IDX_IXX,
    BODY_IDX_IYY,
    BODY_IDX_IZZ,
    BODY_IDX_IQUAT_X,
    BODY_IDX_IQUAT_Y,
    BODY_IDX_IQUAT_Z,
    BODY_IDX_IQUAT_W,
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
def _is_descendant_gpu[
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
    while current > 0:
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


def compute_bias_forces[
    DTYPE: DType,
    NQ: Int,
    NV: Int,
    NBODY: Int,
    NJOINT: Int,
    MAX_CONTACTS: Int,
    V_SIZE: Int,
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
    data: Data[DTYPE, NQ, NV, NBODY, NJOINT, MAX_CONTACTS, NSITE],
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

            # Body contribution (use xipos = CoM world position)
            var body_px = data.xipos[body * 3 + 0]
            var body_py = data.xipos[body * 3 + 1]
            var body_pz = data.xipos[body * 3 + 2]
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
                    var desc_px = data.xipos[desc_body * 3 + 0]
                    var desc_py = data.xipos[desc_body * 3 + 1]
                    var desc_pz = data.xipos[desc_body * 3 + 2]
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


def _is_descendant[
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
    body: Int,
    ancestor: Int,
) -> Bool:
    """Check if body is a descendant of ancestor in the kinematic tree."""
    var current = body
    while current > 0:
        if model.body_parent[current] == ancestor:
            return True
        current = model.body_parent[current]
    return False


# =============================================================================
# Coriolis Forces (for higher-order accuracy)
# =============================================================================


def compute_coriolis_forces[
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
    data: Data[DTYPE, NQ, NV, NBODY, NJOINT, MAX_CONTACTS, NSITE],
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
def compute_bias_forces_gpu[
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

            # Body contribution (use xipos = CoM world position)
            var xipos_off = xipos_offset[NQ, NV, NBODY]()
            var body_px = rebind[Scalar[DTYPE]](
                state[env, xipos_off + body_id * 3 + 0]
            )
            var body_py = rebind[Scalar[DTYPE]](
                state[env, xipos_off + body_id * 3 + 1]
            )
            var body_pz = rebind[Scalar[DTYPE]](
                state[env, xipos_off + body_id * 3 + 2]
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
                        state[env, xipos_off + desc_body * 3 + 0]
                    )
                    var desc_py = rebind[Scalar[DTYPE]](
                        state[env, xipos_off + desc_body * 3 + 1]
                    )
                    var desc_pz = rebind[Scalar[DTYPE]](
                        state[env, xipos_off + desc_body * 3 + 2]
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


def compute_bias_forces_rne[
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
    data: Data[DTYPE, NQ, NV, NBODY, NJOINT, MAX_CONTACTS, NSITE],
    cdof: List[Scalar[DTYPE]],
    mut bias: List[Scalar[DTYPE]],
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
       When subtree_com is provided, cfrc is shifted from xipos to
       subtree_com[rootid] before projection (matching cdof reference).

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

    # Per-body spatial velocity [angular(3), linear(3)] at CoM
    comptime BODY6_SIZE = _max_one[NBODY * 6]()
    var cvel = InlineArray[Scalar[DTYPE], BODY6_SIZE](uninitialized=True)
    for i in range(BODY6_SIZE):
        cvel[i] = Scalar[DTYPE](0)

    # Per-body spatial acceleration [angular(3), linear(3)]
    var cacc = InlineArray[Scalar[DTYPE], BODY6_SIZE](uninitialized=True)
    for i in range(BODY6_SIZE):
        cacc[i] = Scalar[DTYPE](0)

    # Per-body spatial force [torque(3), force(3)]
    var cfrc = InlineArray[Scalar[DTYPE], BODY6_SIZE](uninitialized=True)
    for i in range(BODY6_SIZE):
        cfrc[i] = Scalar[DTYPE](0)

    # Per-body cinert (MuJoCo format: Ixx,Iyy,Izz,Ixy,Ixz,Iyz,mcx,mcy,mcz,mass)
    # Inertia at subtree_com[rootid] reference, including parallel axis shift.
    comptime CINERT_SIZE = _max_one[NBODY * 10]()
    var cinert = InlineArray[Scalar[DTYPE], CINERT_SIZE](uninitialized=True)
    for i in range(CINERT_SIZE):
        cinert[i] = Scalar[DTYPE](0)

    # =========================================================================
    # Step 0: Compute cinert — spatial inertia at subtree_com (MuJoCo mj_inertCom)
    # cinert[b*10+0..5] = rotated inertia + parallel axis shift
    # cinert[b*10+6..8] = mass * (xipos - subtree_com)
    # cinert[b*10+9] = mass
    # =========================================================================
    for b in range(NBODY):
        var mass = model.body_mass[b]
        var Ixx_local = model.body_inertia[b * 3 + 0]
        var Iyy_local = model.body_inertia[b * 3 + 1]
        var Izz_local = model.body_inertia[b * 3 + 2]

        # Compose xquat with body_iquat for inertia rotation
        var bqx = data.xquat[b * 4 + 0]
        var bqy = data.xquat[b * 4 + 1]
        var bqz = data.xquat[b * 4 + 2]
        var bqw = data.xquat[b * 4 + 3]
        var iqx = model.body_iquat[b * 4 + 0]
        var iqy = model.body_iquat[b * 4 + 1]
        var iqz = model.body_iquat[b * 4 + 2]
        var iqw = model.body_iquat[b * 4 + 3]
        var iq = quat_mul(bqx, bqy, bqz, bqw, iqx, iqy, iqz, iqw)
        var qx = iq[0]
        var qy = iq[1]
        var qz = iq[2]
        var qw = iq[3]

        # Rotation matrix from quaternion (ximat)
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
        cinert[b * 10 + 0] = (
            Ixx_local * r00 * r00 + Iyy_local * r01 * r01 + Izz_local * r02 * r02
        )  # Ixx
        cinert[b * 10 + 1] = (
            Ixx_local * r10 * r10 + Iyy_local * r11 * r11 + Izz_local * r12 * r12
        )  # Iyy
        cinert[b * 10 + 2] = (
            Ixx_local * r20 * r20 + Iyy_local * r21 * r21 + Izz_local * r22 * r22
        )  # Izz
        cinert[b * 10 + 3] = (
            Ixx_local * r00 * r10 + Iyy_local * r01 * r11 + Izz_local * r02 * r12
        )  # Ixy
        cinert[b * 10 + 4] = (
            Ixx_local * r00 * r20 + Iyy_local * r01 * r21 + Izz_local * r02 * r22
        )  # Ixz
        cinert[b * 10 + 5] = (
            Ixx_local * r10 * r20 + Iyy_local * r11 * r21 + Izz_local * r12 * r22
        )  # Iyz

        # Parallel axis theorem: shift from xipos to subtree_com[rootid]
        # dif = xipos - subtree_com[rootid]
        # I += mass * (dif^2*I_3 - dif⊗dif)
        if data.has_subtree_com:
            var root = model.body_rootid[b]
            var dx = data.xipos[b * 3 + 0] - data.subtree_com[root * 3 + 0]
            var dy = data.xipos[b * 3 + 1] - data.subtree_com[root * 3 + 1]
            var dz = data.xipos[b * 3 + 2] - data.subtree_com[root * 3 + 2]
            cinert[b * 10 + 0] = cinert[b * 10 + 0] + mass * (dy * dy + dz * dz)
            cinert[b * 10 + 1] = cinert[b * 10 + 1] + mass * (dx * dx + dz * dz)
            cinert[b * 10 + 2] = cinert[b * 10 + 2] + mass * (dx * dx + dy * dy)
            cinert[b * 10 + 3] = cinert[b * 10 + 3] - mass * dx * dy
            cinert[b * 10 + 4] = cinert[b * 10 + 4] - mass * dx * dz
            cinert[b * 10 + 5] = cinert[b * 10 + 5] - mass * dy * dz
            cinert[b * 10 + 6] = mass * dx
            cinert[b * 10 + 7] = mass * dy
            cinert[b * 10 + 8] = mass * dz
        cinert[b * 10 + 9] = mass

    # =========================================================================
    # Step 1: Forward pass - compute cvel and cacc (root to leaves)
    #   MuJoCo-style: cvel is accumulated progressively per joint.
    #   For each body:
    #     cvel = cvel[parent]
    #     For each joint d of this body:
    #       cdof_dot = cvel x_motion cdof[d]
    #       cacc += cdof_dot * qvel[d]
    #       cvel += cdof[d] * qvel[d]   (update BEFORE next joint)
    #
    #   Reference: MuJoCo engine_core_smooth.c mj_comVel()
    # =========================================================================
    # Skip worldbody at 0 (no joints, cvel=0, cacc=0)
    for b in range(1, NBODY):
        var parent = model.body_parent[b]

        # Initialize cvel from parent — simple copy (MuJoCo mj_comVel).
        # No moment arm transfer because all bodies share the same
        # subtree_com reference point.
        var cv_wx = cvel[parent * 6 + 0]
        var cv_wy = cvel[parent * 6 + 1]
        var cv_wz = cvel[parent * 6 + 2]
        var cv_vx = cvel[parent * 6 + 3]
        var cv_vy = cvel[parent * 6 + 4]
        var cv_vz = cvel[parent * 6 + 5]

        if parent == 0:
            # Root body (parent is worldbody): gravity as fictitious acceleration
            cacc[b * 6 + 0] = Scalar[DTYPE](0)
            cacc[b * 6 + 1] = Scalar[DTYPE](0)
            cacc[b * 6 + 2] = Scalar[DTYPE](0)
            cacc[b * 6 + 3] = -gx
            cacc[b * 6 + 4] = -gy
            cacc[b * 6 + 5] = -gz
        else:
            # Copy parent's acceleration
            for k in range(6):
                cacc[b * 6 + k] = cacc[parent * 6 + k]

        # Process each joint of this body
        for j in range(model.num_joints):
            var joint = model.joints[j]
            if joint.body_id != b:
                continue

            var dof_adr = joint.dof_adr

            if joint.jnt_type == JNT_FREE:
                # FREE joint: translation DOFs have cdof_dot = 0
                # First 3 DOFs (translation): cdof_dot = 0, just update cvel
                for d in range(3):
                    var dof = dof_adr + d
                    var qdot = data.qvel[dof]
                    cv_wx = cv_wx + cdof[dof * 6 + 0] * qdot
                    cv_wy = cv_wy + cdof[dof * 6 + 1] * qdot
                    cv_wz = cv_wz + cdof[dof * 6 + 2] * qdot
                    cv_vx = cv_vx + cdof[dof * 6 + 3] * qdot
                    cv_vy = cv_vy + cdof[dof * 6 + 4] * qdot
                    cv_vz = cv_vz + cdof[dof * 6 + 5] * qdot

                # Next 3 DOFs (rotation): compute cdof_dot with current cvel
                for d in range(3, 6):
                    var dof = dof_adr + d
                    var qdot = data.qvel[dof]
                    var s_ang_x = cdof[dof * 6 + 0]
                    var s_ang_y = cdof[dof * 6 + 1]
                    var s_ang_z = cdof[dof * 6 + 2]
                    var s_lin_x = cdof[dof * 6 + 3]
                    var s_lin_y = cdof[dof * 6 + 4]
                    var s_lin_z = cdof[dof * 6 + 5]

                    # cdof_dot = cvel x_motion cdof
                    var cdot_ang_x = cv_wy * s_ang_z - cv_wz * s_ang_y
                    var cdot_ang_y = cv_wz * s_ang_x - cv_wx * s_ang_z
                    var cdot_ang_z = cv_wx * s_ang_y - cv_wy * s_ang_x
                    var cdot_lin_x = (cv_wy * s_lin_z - cv_wz * s_lin_y) + (
                        cv_vy * s_ang_z - cv_vz * s_ang_y
                    )
                    var cdot_lin_y = (cv_wz * s_lin_x - cv_wx * s_lin_z) + (
                        cv_vz * s_ang_x - cv_vx * s_ang_z
                    )
                    var cdot_lin_z = (cv_wx * s_lin_y - cv_wy * s_lin_x) + (
                        cv_vx * s_ang_y - cv_vy * s_ang_x
                    )

                    cacc[b * 6 + 0] = cacc[b * 6 + 0] + cdot_ang_x * qdot
                    cacc[b * 6 + 1] = cacc[b * 6 + 1] + cdot_ang_y * qdot
                    cacc[b * 6 + 2] = cacc[b * 6 + 2] + cdot_ang_z * qdot
                    cacc[b * 6 + 3] = cacc[b * 6 + 3] + cdot_lin_x * qdot
                    cacc[b * 6 + 4] = cacc[b * 6 + 4] + cdot_lin_y * qdot
                    cacc[b * 6 + 5] = cacc[b * 6 + 5] + cdot_lin_z * qdot

                    # Update cvel
                    cv_wx = cv_wx + s_ang_x * qdot
                    cv_wy = cv_wy + s_ang_y * qdot
                    cv_wz = cv_wz + s_ang_z * qdot
                    cv_vx = cv_vx + s_lin_x * qdot
                    cv_vy = cv_vy + s_lin_y * qdot
                    cv_vz = cv_vz + s_lin_z * qdot

            elif joint.jnt_type == JNT_BALL:
                # BALL: compute all 3 cdof_dots using current cvel, then update
                for d in range(3):
                    var dof = dof_adr + d
                    var qdot = data.qvel[dof]
                    var s_ang_x = cdof[dof * 6 + 0]
                    var s_ang_y = cdof[dof * 6 + 1]
                    var s_ang_z = cdof[dof * 6 + 2]
                    var s_lin_x = cdof[dof * 6 + 3]
                    var s_lin_y = cdof[dof * 6 + 4]
                    var s_lin_z = cdof[dof * 6 + 5]

                    var cdot_ang_x = cv_wy * s_ang_z - cv_wz * s_ang_y
                    var cdot_ang_y = cv_wz * s_ang_x - cv_wx * s_ang_z
                    var cdot_ang_z = cv_wx * s_ang_y - cv_wy * s_ang_x
                    var cdot_lin_x = (cv_wy * s_lin_z - cv_wz * s_lin_y) + (
                        cv_vy * s_ang_z - cv_vz * s_ang_y
                    )
                    var cdot_lin_y = (cv_wz * s_lin_x - cv_wx * s_lin_z) + (
                        cv_vz * s_ang_x - cv_vx * s_ang_z
                    )
                    var cdot_lin_z = (cv_wx * s_lin_y - cv_wy * s_lin_x) + (
                        cv_vx * s_ang_y - cv_vy * s_ang_x
                    )

                    cacc[b * 6 + 0] = cacc[b * 6 + 0] + cdot_ang_x * qdot
                    cacc[b * 6 + 1] = cacc[b * 6 + 1] + cdot_ang_y * qdot
                    cacc[b * 6 + 2] = cacc[b * 6 + 2] + cdot_ang_z * qdot
                    cacc[b * 6 + 3] = cacc[b * 6 + 3] + cdot_lin_x * qdot
                    cacc[b * 6 + 4] = cacc[b * 6 + 4] + cdot_lin_y * qdot
                    cacc[b * 6 + 5] = cacc[b * 6 + 5] + cdot_lin_z * qdot

                # Update cvel after all 3 DOFs
                for d in range(3):
                    var dof = dof_adr + d
                    var qdot = data.qvel[dof]
                    cv_wx = cv_wx + cdof[dof * 6 + 0] * qdot
                    cv_wy = cv_wy + cdof[dof * 6 + 1] * qdot
                    cv_wz = cv_wz + cdof[dof * 6 + 2] * qdot
                    cv_vx = cv_vx + cdof[dof * 6 + 3] * qdot
                    cv_vy = cv_vy + cdof[dof * 6 + 4] * qdot
                    cv_vz = cv_vz + cdof[dof * 6 + 5] * qdot

            else:
                # HINGE or SLIDE (1 DOF)
                var dof = dof_adr
                var qdot = data.qvel[dof]
                var s_ang_x = cdof[dof * 6 + 0]
                var s_ang_y = cdof[dof * 6 + 1]
                var s_ang_z = cdof[dof * 6 + 2]
                var s_lin_x = cdof[dof * 6 + 3]
                var s_lin_y = cdof[dof * 6 + 4]
                var s_lin_z = cdof[dof * 6 + 5]

                # cdof_dot = cvel x_motion cdof
                var cdot_ang_x = cv_wy * s_ang_z - cv_wz * s_ang_y
                var cdot_ang_y = cv_wz * s_ang_x - cv_wx * s_ang_z
                var cdot_ang_z = cv_wx * s_ang_y - cv_wy * s_ang_x
                var cdot_lin_x = (cv_wy * s_lin_z - cv_wz * s_lin_y) + (
                    cv_vy * s_ang_z - cv_vz * s_ang_y
                )
                var cdot_lin_y = (cv_wz * s_lin_x - cv_wx * s_lin_z) + (
                    cv_vz * s_ang_x - cv_vx * s_ang_z
                )
                var cdot_lin_z = (cv_wx * s_lin_y - cv_wy * s_lin_x) + (
                    cv_vx * s_ang_y - cv_vy * s_ang_x
                )

                cacc[b * 6 + 0] = cacc[b * 6 + 0] + cdot_ang_x * qdot
                cacc[b * 6 + 1] = cacc[b * 6 + 1] + cdot_ang_y * qdot
                cacc[b * 6 + 2] = cacc[b * 6 + 2] + cdot_ang_z * qdot
                cacc[b * 6 + 3] = cacc[b * 6 + 3] + cdot_lin_x * qdot
                cacc[b * 6 + 4] = cacc[b * 6 + 4] + cdot_lin_y * qdot
                cacc[b * 6 + 5] = cacc[b * 6 + 5] + cdot_lin_z * qdot

                # Update cvel (crossMotion(cdof,cdof)=0 so order doesn't matter
                # for single DOF, but we update for consistency)
                cv_wx = cv_wx + s_ang_x * qdot
                cv_wy = cv_wy + s_ang_y * qdot
                cv_wz = cv_wz + s_ang_z * qdot
                cv_vx = cv_vx + s_lin_x * qdot
                cv_vy = cv_vy + s_lin_y * qdot
                cv_vz = cv_vz + s_lin_z * qdot

        # Store final cvel for this body (used by children and Step 2)
        cvel[b * 6 + 0] = cv_wx
        cvel[b * 6 + 1] = cv_wy
        cvel[b * 6 + 2] = cv_wz
        cvel[b * 6 + 3] = cv_vx
        cvel[b * 6 + 4] = cv_vy
        cvel[b * 6 + 5] = cv_vz

    # =========================================================================
    # Step 2: Compute spatial forces per body using cinert (MuJoCo mj_rne)
    #   cfrc = cinert * cacc + cvel x* (cinert * cvel)
    #   cinert is 10-element spatial inertia at subtree_com[rootid]
    #   mju_mulInertVec: [I -mc×; mc× m*I] * [ω; v]
    # =========================================================================
    for b in range(NBODY):
        # Read cinert
        var ci0 = cinert[b * 10 + 0]  # Ixx
        var ci1 = cinert[b * 10 + 1]  # Iyy
        var ci2 = cinert[b * 10 + 2]  # Izz
        var ci3 = cinert[b * 10 + 3]  # Ixy
        var ci4 = cinert[b * 10 + 4]  # Ixz
        var ci5 = cinert[b * 10 + 5]  # Iyz
        var ci6 = cinert[b * 10 + 6]  # mcx = m*dx
        var ci7 = cinert[b * 10 + 7]  # mcy = m*dy
        var ci8 = cinert[b * 10 + 8]  # mcz = m*dz
        var ci9 = cinert[b * 10 + 9]  # mass

        # cvel and cacc
        var wx = cvel[b * 6 + 0]
        var wy = cvel[b * 6 + 1]
        var wz = cvel[b * 6 + 2]
        var vx = cvel[b * 6 + 3]
        var vy = cvel[b * 6 + 4]
        var vz = cvel[b * 6 + 5]
        var ax = cacc[b * 6 + 0]
        var ay = cacc[b * 6 + 1]
        var az = cacc[b * 6 + 2]
        var alx = cacc[b * 6 + 3]
        var aly = cacc[b * 6 + 4]
        var alz = cacc[b * 6 + 5]

        # cinert * cacc (MuJoCo mju_mulInertVec)
        var Ia0 = ci0*ax + ci3*ay + ci4*az - ci8*aly + ci7*alz
        var Ia1 = ci3*ax + ci1*ay + ci5*az + ci8*alx - ci6*alz
        var Ia2 = ci4*ax + ci5*ay + ci2*az - ci7*alx + ci6*aly
        var Ia3 = ci8*ay - ci7*az + ci9*alx
        var Ia4 = ci6*az - ci8*ax + ci9*aly
        var Ia5 = ci7*ax - ci6*ay + ci9*alz

        # cinert * cvel
        var Iv0 = ci0*wx + ci3*wy + ci4*wz - ci8*vy + ci7*vz
        var Iv1 = ci3*wx + ci1*wy + ci5*wz + ci8*vx - ci6*vz
        var Iv2 = ci4*wx + ci5*wy + ci2*wz - ci7*vx + ci6*vy
        var Iv3 = ci8*wy - ci7*wz + ci9*vx
        var Iv4 = ci6*wz - ci8*wx + ci9*vy
        var Iv5 = ci7*wx - ci6*wy + ci9*vz

        # cvel x* (cinert * cvel) — spatial force cross product
        # MuJoCo mju_crossForce: [ω×τ + v×f; ω×f]
        var xf0 = wy*Iv2 - wz*Iv1 + vy*Iv5 - vz*Iv4
        var xf1 = wz*Iv0 - wx*Iv2 + vz*Iv3 - vx*Iv5
        var xf2 = wx*Iv1 - wy*Iv0 + vx*Iv4 - vy*Iv3
        var xf3 = wy*Iv5 - wz*Iv4
        var xf4 = wz*Iv3 - wx*Iv5
        var xf5 = wx*Iv4 - wy*Iv3

        # cfrc = cinert*cacc + cvel x* (cinert*cvel)
        cfrc[b * 6 + 0] = Ia0 + xf0
        cfrc[b * 6 + 1] = Ia1 + xf1
        cfrc[b * 6 + 2] = Ia2 + xf2
        cfrc[b * 6 + 3] = Ia3 + xf3
        cfrc[b * 6 + 4] = Ia4 + xf4
        cfrc[b * 6 + 5] = Ia5 + xf5

    # =========================================================================
    # Step 3: Backward pass - accumulate forces to parents
    #   Simple addition — no moment arm transfer needed because all cfrc
    #   are at the same reference point (subtree_com[rootid]).
    #   MuJoCo mj_rne: mju_addTo(cfrc_body[parent], cfrc_body[child], 6)
    # =========================================================================
    for b in range(NBODY - 1, 0, -1):
        var parent = model.body_parent[b]
        if parent > 0:
            cfrc[parent * 6 + 0] = cfrc[parent * 6 + 0] + cfrc[b * 6 + 0]
            cfrc[parent * 6 + 1] = cfrc[parent * 6 + 1] + cfrc[b * 6 + 1]
            cfrc[parent * 6 + 2] = cfrc[parent * 6 + 2] + cfrc[b * 6 + 2]
            cfrc[parent * 6 + 3] = cfrc[parent * 6 + 3] + cfrc[b * 6 + 3]
            cfrc[parent * 6 + 4] = cfrc[parent * 6 + 4] + cfrc[b * 6 + 4]
            cfrc[parent * 6 + 5] = cfrc[parent * 6 + 5] + cfrc[b * 6 + 5]

    # =========================================================================
    # Step 4: Project to joint space
    #   bias[d] = cdof[d] . cfrc[body_of_dof[d]]
    #   6D dot product: angular . torque + linear . force
    # =========================================================================
    # Step 4: Project to joint space — direct cdof . cfrc (MuJoCo mj_rne)
    # Both cdof and cfrc are now at subtree_com[rootid], so no shift needed.
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
def compute_bias_forces_rne_gpu[
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

    Same algorithm as compute_bias_forces_rne but reads from std.gpu LayoutTensors.
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
    var xquat_off = xquat_offset[NQ, NV, NBODY]()
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

        # Compose xquat with body_iquat for inertia rotation
        var bqx = rebind[Scalar[DTYPE]](state[env, xquat_off + b * 4 + 0])
        var bqy = rebind[Scalar[DTYPE]](state[env, xquat_off + b * 4 + 1])
        var bqz = rebind[Scalar[DTYPE]](state[env, xquat_off + b * 4 + 2])
        var bqw = rebind[Scalar[DTYPE]](state[env, xquat_off + b * 4 + 3])
        var iqx = rebind[Scalar[DTYPE]](model[0, body_off + BODY_IDX_IQUAT_X])
        var iqy = rebind[Scalar[DTYPE]](model[0, body_off + BODY_IDX_IQUAT_Y])
        var iqz = rebind[Scalar[DTYPE]](model[0, body_off + BODY_IDX_IQUAT_Z])
        var iqw = rebind[Scalar[DTYPE]](model[0, body_off + BODY_IDX_IQUAT_W])
        var iq = gpu_quat_mul(bqx, bqy, bqz, bqw, iqx, iqy, iqz, iqw)
        var qx = iq[0]
        var qy = iq[1]
        var qz = iq[2]
        var qw = iq[3]

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

    # Per-body spatial velocity stored in workspace at crb offset (reused, not needed during RNE)
    # cvel uses NBODY*6 slots, crb region has NBODY*10 slots — fits easily
    comptime cvel_idx = ws_crb_offset[NV]()
    for i in range(NBODY * 6):
        workspace[env, cvel_idx + i] = 0

    # =========================================================================
    # Step 1: Forward pass - compute cvel and cacc (root to leaves)
    #   MuJoCo-style: cvel is accumulated progressively per joint.
    #   Matches CPU compute_bias_forces_rne.
    # =========================================================================
    var xipos_off = xipos_offset[NQ, NV, NBODY]()

    # Skip worldbody at 0 (no joints, cvel=0, cacc=0)
    for b in range(1, NBODY):
        var body_off = model_body_offset(b)
        var parent = Int(
            rebind[Scalar[DTYPE]](model[0, body_off + BODY_IDX_PARENT])
        )

        # Initialize cvel from parent, transferred to this body's CoM
        var cv_wx = rebind[Scalar[DTYPE]](
            workspace[env, cvel_idx + parent * 6 + 0]
        )
        var cv_wy = rebind[Scalar[DTYPE]](
            workspace[env, cvel_idx + parent * 6 + 1]
        )
        var cv_wz = rebind[Scalar[DTYPE]](
            workspace[env, cvel_idx + parent * 6 + 2]
        )
        var cv_vx = rebind[Scalar[DTYPE]](
            workspace[env, cvel_idx + parent * 6 + 3]
        )
        var cv_vy = rebind[Scalar[DTYPE]](
            workspace[env, cvel_idx + parent * 6 + 4]
        )
        var cv_vz = rebind[Scalar[DTYPE]](
            workspace[env, cvel_idx + parent * 6 + 5]
        )
        if parent > 0:
            # Transfer linear velocity from parent CoM to this body's CoM
            var rx = rebind[Scalar[DTYPE]](
                state[env, xipos_off + b * 3 + 0]
            ) - rebind[Scalar[DTYPE]](state[env, xipos_off + parent * 3 + 0])
            var ry = rebind[Scalar[DTYPE]](
                state[env, xipos_off + b * 3 + 1]
            ) - rebind[Scalar[DTYPE]](state[env, xipos_off + parent * 3 + 1])
            var rz = rebind[Scalar[DTYPE]](
                state[env, xipos_off + b * 3 + 2]
            ) - rebind[Scalar[DTYPE]](state[env, xipos_off + parent * 3 + 2])
            cv_vx = cv_vx + (cv_wy * rz - cv_wz * ry)
            cv_vy = cv_vy + (cv_wz * rx - cv_wx * rz)
            cv_vz = cv_vz + (cv_wx * ry - cv_wy * rx)

        if parent == 0:
            # Root body (parent is worldbody): gravity as fictitious acceleration
            cacc[b * 6 + 0] = Scalar[DTYPE](0)
            cacc[b * 6 + 1] = Scalar[DTYPE](0)
            cacc[b * 6 + 2] = Scalar[DTYPE](0)
            cacc[b * 6 + 3] = -gx
            cacc[b * 6 + 4] = -gy
            cacc[b * 6 + 5] = -gz
        else:
            for k in range(6):
                cacc[b * 6 + k] = cacc[parent * 6 + k]

        # Process each joint of this body
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

            if jnt_type == JNT_FREE:
                # Translation DOFs: cdof_dot = 0, just update cvel
                for d in range(3):
                    var dof = dof_adr + d
                    var qdot = rebind[Scalar[DTYPE]](state[env, qvel_off + dof])
                    cv_wx = (
                        cv_wx
                        + rebind[Scalar[DTYPE]](
                            workspace[env, cdof_idx + dof * 6 + 0]
                        )
                        * qdot
                    )
                    cv_wy = (
                        cv_wy
                        + rebind[Scalar[DTYPE]](
                            workspace[env, cdof_idx + dof * 6 + 1]
                        )
                        * qdot
                    )
                    cv_wz = (
                        cv_wz
                        + rebind[Scalar[DTYPE]](
                            workspace[env, cdof_idx + dof * 6 + 2]
                        )
                        * qdot
                    )
                    cv_vx = (
                        cv_vx
                        + rebind[Scalar[DTYPE]](
                            workspace[env, cdof_idx + dof * 6 + 3]
                        )
                        * qdot
                    )
                    cv_vy = (
                        cv_vy
                        + rebind[Scalar[DTYPE]](
                            workspace[env, cdof_idx + dof * 6 + 4]
                        )
                        * qdot
                    )
                    cv_vz = (
                        cv_vz
                        + rebind[Scalar[DTYPE]](
                            workspace[env, cdof_idx + dof * 6 + 5]
                        )
                        * qdot
                    )

                # Rotation DOFs: compute cdof_dot with current cvel
                for d in range(3, 6):
                    var dof = dof_adr + d
                    var qdot = rebind[Scalar[DTYPE]](state[env, qvel_off + dof])
                    var s_ang_x = rebind[Scalar[DTYPE]](
                        workspace[env, cdof_idx + dof * 6 + 0]
                    )
                    var s_ang_y = rebind[Scalar[DTYPE]](
                        workspace[env, cdof_idx + dof * 6 + 1]
                    )
                    var s_ang_z = rebind[Scalar[DTYPE]](
                        workspace[env, cdof_idx + dof * 6 + 2]
                    )
                    var s_lin_x = rebind[Scalar[DTYPE]](
                        workspace[env, cdof_idx + dof * 6 + 3]
                    )
                    var s_lin_y = rebind[Scalar[DTYPE]](
                        workspace[env, cdof_idx + dof * 6 + 4]
                    )
                    var s_lin_z = rebind[Scalar[DTYPE]](
                        workspace[env, cdof_idx + dof * 6 + 5]
                    )

                    var cdot_ang_x = cv_wy * s_ang_z - cv_wz * s_ang_y
                    var cdot_ang_y = cv_wz * s_ang_x - cv_wx * s_ang_z
                    var cdot_ang_z = cv_wx * s_ang_y - cv_wy * s_ang_x
                    var cdot_lin_x = (cv_wy * s_lin_z - cv_wz * s_lin_y) + (
                        cv_vy * s_ang_z - cv_vz * s_ang_y
                    )
                    var cdot_lin_y = (cv_wz * s_lin_x - cv_wx * s_lin_z) + (
                        cv_vz * s_ang_x - cv_vx * s_ang_z
                    )
                    var cdot_lin_z = (cv_wx * s_lin_y - cv_wy * s_lin_x) + (
                        cv_vx * s_ang_y - cv_vy * s_ang_x
                    )

                    cacc[b * 6 + 0] = cacc[b * 6 + 0] + cdot_ang_x * qdot
                    cacc[b * 6 + 1] = cacc[b * 6 + 1] + cdot_ang_y * qdot
                    cacc[b * 6 + 2] = cacc[b * 6 + 2] + cdot_ang_z * qdot
                    cacc[b * 6 + 3] = cacc[b * 6 + 3] + cdot_lin_x * qdot
                    cacc[b * 6 + 4] = cacc[b * 6 + 4] + cdot_lin_y * qdot
                    cacc[b * 6 + 5] = cacc[b * 6 + 5] + cdot_lin_z * qdot

                    cv_wx = cv_wx + s_ang_x * qdot
                    cv_wy = cv_wy + s_ang_y * qdot
                    cv_wz = cv_wz + s_ang_z * qdot
                    cv_vx = cv_vx + s_lin_x * qdot
                    cv_vy = cv_vy + s_lin_y * qdot
                    cv_vz = cv_vz + s_lin_z * qdot

            elif jnt_type == JNT_BALL:
                # BALL: compute all 3 cdof_dots using current cvel, then update
                for d in range(3):
                    var dof = dof_adr + d
                    var qdot = rebind[Scalar[DTYPE]](state[env, qvel_off + dof])
                    var s_ang_x = rebind[Scalar[DTYPE]](
                        workspace[env, cdof_idx + dof * 6 + 0]
                    )
                    var s_ang_y = rebind[Scalar[DTYPE]](
                        workspace[env, cdof_idx + dof * 6 + 1]
                    )
                    var s_ang_z = rebind[Scalar[DTYPE]](
                        workspace[env, cdof_idx + dof * 6 + 2]
                    )
                    var s_lin_x = rebind[Scalar[DTYPE]](
                        workspace[env, cdof_idx + dof * 6 + 3]
                    )
                    var s_lin_y = rebind[Scalar[DTYPE]](
                        workspace[env, cdof_idx + dof * 6 + 4]
                    )
                    var s_lin_z = rebind[Scalar[DTYPE]](
                        workspace[env, cdof_idx + dof * 6 + 5]
                    )

                    var cdot_ang_x = cv_wy * s_ang_z - cv_wz * s_ang_y
                    var cdot_ang_y = cv_wz * s_ang_x - cv_wx * s_ang_z
                    var cdot_ang_z = cv_wx * s_ang_y - cv_wy * s_ang_x
                    var cdot_lin_x = (cv_wy * s_lin_z - cv_wz * s_lin_y) + (
                        cv_vy * s_ang_z - cv_vz * s_ang_y
                    )
                    var cdot_lin_y = (cv_wz * s_lin_x - cv_wx * s_lin_z) + (
                        cv_vz * s_ang_x - cv_vx * s_ang_z
                    )
                    var cdot_lin_z = (cv_wx * s_lin_y - cv_wy * s_lin_x) + (
                        cv_vx * s_ang_y - cv_vy * s_ang_x
                    )

                    cacc[b * 6 + 0] = cacc[b * 6 + 0] + cdot_ang_x * qdot
                    cacc[b * 6 + 1] = cacc[b * 6 + 1] + cdot_ang_y * qdot
                    cacc[b * 6 + 2] = cacc[b * 6 + 2] + cdot_ang_z * qdot
                    cacc[b * 6 + 3] = cacc[b * 6 + 3] + cdot_lin_x * qdot
                    cacc[b * 6 + 4] = cacc[b * 6 + 4] + cdot_lin_y * qdot
                    cacc[b * 6 + 5] = cacc[b * 6 + 5] + cdot_lin_z * qdot

                # Update cvel after all 3 DOFs
                for d in range(3):
                    var dof = dof_adr + d
                    var qdot = rebind[Scalar[DTYPE]](state[env, qvel_off + dof])
                    cv_wx = (
                        cv_wx
                        + rebind[Scalar[DTYPE]](
                            workspace[env, cdof_idx + dof * 6 + 0]
                        )
                        * qdot
                    )
                    cv_wy = (
                        cv_wy
                        + rebind[Scalar[DTYPE]](
                            workspace[env, cdof_idx + dof * 6 + 1]
                        )
                        * qdot
                    )
                    cv_wz = (
                        cv_wz
                        + rebind[Scalar[DTYPE]](
                            workspace[env, cdof_idx + dof * 6 + 2]
                        )
                        * qdot
                    )
                    cv_vx = (
                        cv_vx
                        + rebind[Scalar[DTYPE]](
                            workspace[env, cdof_idx + dof * 6 + 3]
                        )
                        * qdot
                    )
                    cv_vy = (
                        cv_vy
                        + rebind[Scalar[DTYPE]](
                            workspace[env, cdof_idx + dof * 6 + 4]
                        )
                        * qdot
                    )
                    cv_vz = (
                        cv_vz
                        + rebind[Scalar[DTYPE]](
                            workspace[env, cdof_idx + dof * 6 + 5]
                        )
                        * qdot
                    )

            else:
                # HINGE or SLIDE (1 DOF)
                var dof = dof_adr
                var qdot = rebind[Scalar[DTYPE]](state[env, qvel_off + dof])
                var s_ang_x = rebind[Scalar[DTYPE]](
                    workspace[env, cdof_idx + dof * 6 + 0]
                )
                var s_ang_y = rebind[Scalar[DTYPE]](
                    workspace[env, cdof_idx + dof * 6 + 1]
                )
                var s_ang_z = rebind[Scalar[DTYPE]](
                    workspace[env, cdof_idx + dof * 6 + 2]
                )
                var s_lin_x = rebind[Scalar[DTYPE]](
                    workspace[env, cdof_idx + dof * 6 + 3]
                )
                var s_lin_y = rebind[Scalar[DTYPE]](
                    workspace[env, cdof_idx + dof * 6 + 4]
                )
                var s_lin_z = rebind[Scalar[DTYPE]](
                    workspace[env, cdof_idx + dof * 6 + 5]
                )

                var cdot_ang_x = cv_wy * s_ang_z - cv_wz * s_ang_y
                var cdot_ang_y = cv_wz * s_ang_x - cv_wx * s_ang_z
                var cdot_ang_z = cv_wx * s_ang_y - cv_wy * s_ang_x
                var cdot_lin_x = (cv_wy * s_lin_z - cv_wz * s_lin_y) + (
                    cv_vy * s_ang_z - cv_vz * s_ang_y
                )
                var cdot_lin_y = (cv_wz * s_lin_x - cv_wx * s_lin_z) + (
                    cv_vz * s_ang_x - cv_vx * s_ang_z
                )
                var cdot_lin_z = (cv_wx * s_lin_y - cv_wy * s_lin_x) + (
                    cv_vx * s_ang_y - cv_vy * s_ang_x
                )

                cacc[b * 6 + 0] = cacc[b * 6 + 0] + cdot_ang_x * qdot
                cacc[b * 6 + 1] = cacc[b * 6 + 1] + cdot_ang_y * qdot
                cacc[b * 6 + 2] = cacc[b * 6 + 2] + cdot_ang_z * qdot
                cacc[b * 6 + 3] = cacc[b * 6 + 3] + cdot_lin_x * qdot
                cacc[b * 6 + 4] = cacc[b * 6 + 4] + cdot_lin_y * qdot
                cacc[b * 6 + 5] = cacc[b * 6 + 5] + cdot_lin_z * qdot

                cv_wx = cv_wx + s_ang_x * qdot
                cv_wy = cv_wy + s_ang_y * qdot
                cv_wz = cv_wz + s_ang_z * qdot
                cv_vx = cv_vx + s_lin_x * qdot
                cv_vy = cv_vy + s_lin_y * qdot
                cv_vz = cv_vz + s_lin_z * qdot

        # Store final cvel for this body (in workspace crb region)
        workspace[env, cvel_idx + b * 6 + 0] = cv_wx
        workspace[env, cvel_idx + b * 6 + 1] = cv_wy
        workspace[env, cvel_idx + b * 6 + 2] = cv_wz
        workspace[env, cvel_idx + b * 6 + 3] = cv_vx
        workspace[env, cvel_idx + b * 6 + 4] = cv_vy
        workspace[env, cvel_idx + b * 6 + 5] = cv_vz

    # =========================================================================
    # Step 2: Compute spatial forces per body
    #   cfrc = I * cacc + cvel x* (I * cvel)
    #   Using accumulated cvel from workspace
    # =========================================================================
    for b in range(NBODY):
        var body_off = model_body_offset(b)
        var mass = rebind[Scalar[DTYPE]](model[0, body_off + BODY_IDX_MASS])

        # Body velocities from accumulated cvel (stored in workspace)
        var wx = rebind[Scalar[DTYPE]](workspace[env, cvel_idx + b * 6 + 0])
        var wy = rebind[Scalar[DTYPE]](workspace[env, cvel_idx + b * 6 + 1])
        var wz = rebind[Scalar[DTYPE]](workspace[env, cvel_idx + b * 6 + 2])
        var vx = rebind[Scalar[DTYPE]](workspace[env, cvel_idx + b * 6 + 3])
        var vy = rebind[Scalar[DTYPE]](workspace[env, cvel_idx + b * 6 + 4])
        var vz = rebind[Scalar[DTYPE]](workspace[env, cvel_idx + b * 6 + 5])

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

        # Offset from parent CoM to child CoM (use xipos)
        var xipos_off = xipos_offset[NQ, NV, NBODY]()
        var rx = rebind[Scalar[DTYPE]](
            state[env, xipos_off + b * 3 + 0]
        ) - rebind[Scalar[DTYPE]](state[env, xipos_off + parent * 3 + 0])
        var ry = rebind[Scalar[DTYPE]](
            state[env, xipos_off + b * 3 + 1]
        ) - rebind[Scalar[DTYPE]](state[env, xipos_off + parent * 3 + 1])
        var rz = rebind[Scalar[DTYPE]](
            state[env, xipos_off + b * 3 + 2]
        ) - rebind[Scalar[DTYPE]](state[env, xipos_off + parent * 3 + 2])

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
