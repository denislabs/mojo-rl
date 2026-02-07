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
from ..dynamics.mass_matrix import (
    compute_mass_matrix_diagonal_gpu,
    compute_mass_matrix_full_gpu,
    ldl_factor_gpu,
    ldl_solve_gpu,
    compute_M_inv_from_ldl_gpu,
)
from ..dynamics.bias_forces import (
    compute_bias_forces_gpu,
    compute_bias_forces_rne_gpu,
)
from ..dynamics.jacobian import (
    compute_cdof_gpu,
    compute_contact_jacobian_row_gpu,
    compute_composite_inertia_gpu,
)
from ..solver.pgs_solver import PGSSolver
from ..traits.solver import ConstraintSolver

from .constants import (
    TPB,
    qpos_offset,
    qvel_offset,
    qacc_offset,
    qfrc_offset,
    xpos_offset,
    xquat_offset,
    xvel_offset,
    xangvel_offset,
    contacts_offset,
    contact_offset,
    metadata_offset,
    state_size,
    model_body_offset,
    model_joint_offset,
    model_metadata_offset,
    model_size,
    MODEL_BODY_SIZE,
    MODEL_JOINT_SIZE,
    MODEL_META_SIZE,
    CONTACT_SIZE,
    BODY_IDX_MASS,
    BODY_IDX_INV_MASS,
    BODY_IDX_IXX,
    BODY_IDX_IYY,
    BODY_IDX_IZZ,
    BODY_IDX_POS_X,
    BODY_IDX_POS_Y,
    BODY_IDX_POS_Z,
    BODY_IDX_QUAT_X,
    BODY_IDX_QUAT_Y,
    BODY_IDX_QUAT_Z,
    BODY_IDX_QUAT_W,
    BODY_IDX_PARENT,
    BODY_IDX_RADIUS,
    BODY_IDX_HALF_LENGTH,
    BODY_IDX_GEOM_TYPE,
    BODY_IDX_HALF_X,
    BODY_IDX_HALF_Y,
    BODY_IDX_HALF_Z,
    JOINT_IDX_TYPE,
    JOINT_IDX_BODY_ID,
    JOINT_IDX_QPOS_ADR,
    JOINT_IDX_DOF_ADR,
    JOINT_IDX_POS_X,
    JOINT_IDX_POS_Y,
    JOINT_IDX_POS_Z,
    JOINT_IDX_AXIS_X,
    JOINT_IDX_AXIS_Y,
    JOINT_IDX_AXIS_Z,
    MODEL_META_IDX_NBODY,
    MODEL_META_IDX_NJOINT,
    MODEL_META_IDX_GRAVITY_Z,
    MODEL_META_IDX_TIMESTEP,
    MODEL_META_IDX_GROUND_Z,
    MODEL_META_IDX_FRICTION,
    CONTACT_IDX_BODY_A,
    CONTACT_IDX_BODY_B,
    CONTACT_IDX_POS_X,
    CONTACT_IDX_POS_Y,
    CONTACT_IDX_POS_Z,
    CONTACT_IDX_NX,
    CONTACT_IDX_NY,
    CONTACT_IDX_NZ,
    CONTACT_IDX_DIST,
    META_IDX_NUM_CONTACTS,
    JNT_FREE,
    JNT_BALL,
    JNT_SLIDE,
    JNT_HINGE,
    JOINT_IDX_RANGE_MIN,
    JOINT_IDX_RANGE_MAX,
    JOINT_IDX_ARMATURE,
    JOINT_IDX_DAMPING,
    JOINT_IDX_STIFFNESS,
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
    var xpos_off = xpos_offset[NQ, NV, NBODY]()
    var xquat_off = xquat_offset[NQ, NV, NBODY]()
    var contacts_off = contacts_offset[NQ, NV, NBODY]()
    var meta_off = metadata_offset[NQ, NV, NBODY, MAX_CONTACTS]()

    var model_meta_off = model_metadata_offset[NBODY, NJOINT]()
    var ground_z = rebind[Scalar[DTYPE]](
        model[0, model_meta_off + MODEL_META_IDX_GROUND_Z]
    )

    var num_contacts = 0

    for body in range(NBODY):
        var body_off = model_body_offset(body)
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
    var qpos_off = qpos_offset[NQ, NV]()
    var qvel_off = qvel_offset[NQ, NV]()
    var qacc_off = qacc_offset[NQ, NV]()
    var qfrc_off = qfrc_offset[NQ, NV]()

    var model_meta_off = model_metadata_offset[NBODY, NJOINT]()
    var dt = rebind[Scalar[DTYPE]](
        model[0, model_meta_off + MODEL_META_IDX_TIMESTEP]
    )

    # Solve M * qacc = qfrc + qfrc_contact - bias
    for i in range(NV):
        var f_net = (
            rebind[Scalar[DTYPE]](state[env, qfrc_off + i])
            + qfrc_contact[i]
            - bias[i]
        )
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
# Constraint-Based Step Kernel (parametrized by solver type)
# =============================================================================


@always_inline
fn step_constraint_kernel_with_solver[
    DTYPE: DType,
    NQ: Int,
    NV: Int,
    NBODY: Int,
    NJOINT: Int,
    MAX_CONTACTS: Int,
    STATE_SIZE: Int,
    MODEL_SIZE: Int,
    BATCH: Int,
    SOLVER: ConstraintSolver,
](
    env: Int,
    state: LayoutTensor[
        DTYPE, Layout.row_major(BATCH, STATE_SIZE), MutAnyOrigin
    ],
    model: LayoutTensor[DTYPE, Layout.row_major(1, MODEL_SIZE), MutAnyOrigin],
):
    """Complete GC physics step with configurable constraint solver.

    Pipeline:
    1. Forward kinematics (qpos -> xpos, xquat)
    2. Compute body velocities (qvel -> xvel, xangvel)
    3. Detect ground contacts
    4. Compute cdof (spatial motion axes per DOF)
    5. Compute composite rigid body inertia (CRBA)
    6. Compute full mass matrix M(q)
    7. LDL factorize M, compute M_inv
    8. Compute bias forces
    9. Compute unconstrained acceleration via LDL solve
    10. Predict velocity
    11. Constraint solve using SOLVER with full M_inv
    12. Write back constrained velocity, integrate position
    13. Normalize quaternions
    14. Enforce joint limits
    """
    comptime V_SIZE = _max_one[NV]()
    comptime M_SIZE = _max_one[NV * NV]()
    comptime CDOF_SIZE = _max_one[NV * 6]()
    comptime CRB_SIZE = _max_one[NBODY * 10]()

    var bias = InlineArray[Scalar[DTYPE], V_SIZE](uninitialized=True)
    var cdof = InlineArray[Scalar[DTYPE], CDOF_SIZE](uninitialized=True)

    for i in range(V_SIZE):
        bias[i] = Scalar[DTYPE](0)
    for i in range(CDOF_SIZE):
        cdof[i] = Scalar[DTYPE](0)

    # 1. Forward kinematics
    forward_kinematics_gpu[
        DTYPE,
        NQ,
        NV,
        NBODY,
        NJOINT,
        MAX_CONTACTS,
        STATE_SIZE,
        MODEL_SIZE,
        BATCH,
    ](env, state, model)

    # 2. Compute body velocities
    compute_body_velocities_gpu[
        DTYPE,
        NQ,
        NV,
        NBODY,
        NJOINT,
        MAX_CONTACTS,
        STATE_SIZE,
        MODEL_SIZE,
        BATCH,
    ](env, state, model)

    # 3. Detect ground contacts + body-body contacts
    detect_ground_contacts_gpu[
        DTYPE,
        NQ,
        NV,
        NBODY,
        NJOINT,
        MAX_CONTACTS,
        STATE_SIZE,
        MODEL_SIZE,
        BATCH,
    ](env, state, model)
    detect_body_body_contacts_gpu[
        DTYPE,
        NQ,
        NV,
        NBODY,
        NJOINT,
        MAX_CONTACTS,
        STATE_SIZE,
        MODEL_SIZE,
        BATCH,
    ](env, state, model)

    # 4. Compute cdof (spatial motion axes per DOF)
    compute_cdof_gpu[
        DTYPE,
        NQ,
        NV,
        NBODY,
        NJOINT,
        MAX_CONTACTS,
        STATE_SIZE,
        MODEL_SIZE,
        CDOF_SIZE,
        BATCH,
    ](env, state, model, cdof)

    # 5. Compute composite rigid body inertia
    var crb = InlineArray[Scalar[DTYPE], CRB_SIZE](uninitialized=True)
    for i in range(CRB_SIZE):
        crb[i] = Scalar[DTYPE](0)
    compute_composite_inertia_gpu[
        DTYPE,
        NQ,
        NV,
        NBODY,
        NJOINT,
        MAX_CONTACTS,
        STATE_SIZE,
        MODEL_SIZE,
        CRB_SIZE,
        BATCH,
    ](env, state, model, crb)

    # 6. Compute full mass matrix using CRBA
    var M = InlineArray[Scalar[DTYPE], M_SIZE](uninitialized=True)
    for i in range(M_SIZE):
        M[i] = Scalar[DTYPE](0)
    compute_mass_matrix_full_gpu[
        DTYPE,
        NQ,
        NV,
        NBODY,
        NJOINT,
        MAX_CONTACTS,
        STATE_SIZE,
        MODEL_SIZE,
        M_SIZE,
        CDOF_SIZE,
        CRB_SIZE,
        BATCH,
    ](env, state, model, cdof, crb, M)

    # 6b. Add armature + implicit damping to mass matrix diagonal
    # MuJoCo implicitfast: M_eff[i,i] += armature[i] + dt * damping[i]
    var model_meta_off_arm = model_metadata_offset[NBODY, NJOINT]()
    var dt_arm = rebind[Scalar[DTYPE]](
        model[0, model_meta_off_arm + MODEL_META_IDX_TIMESTEP]
    )
    for j in range(NJOINT):
        var joint_off = model_joint_offset[NBODY](j)
        var jnt_type = Int(
            rebind[Scalar[DTYPE]](model[0, joint_off + JOINT_IDX_TYPE])
        )
        var dof_adr = Int(
            rebind[Scalar[DTYPE]](model[0, joint_off + JOINT_IDX_DOF_ADR])
        )
        var arm = rebind[Scalar[DTYPE]](
            model[0, joint_off + JOINT_IDX_ARMATURE]
        )
        var damp = rebind[Scalar[DTYPE]](
            model[0, joint_off + JOINT_IDX_DAMPING]
        )
        var diag_add = arm + dt_arm * damp
        if jnt_type == JNT_FREE:
            for d in range(6):
                M[(dof_adr + d) * NV + (dof_adr + d)] = (
                    M[(dof_adr + d) * NV + (dof_adr + d)] + diag_add
                )
        elif jnt_type == JNT_BALL:
            for d in range(3):
                M[(dof_adr + d) * NV + (dof_adr + d)] = (
                    M[(dof_adr + d) * NV + (dof_adr + d)] + diag_add
                )
        else:
            M[dof_adr * NV + dof_adr] = M[dof_adr * NV + dof_adr] + diag_add

    # 7. LDL factorize and compute M_inv
    var L = InlineArray[Scalar[DTYPE], M_SIZE](uninitialized=True)
    var D = InlineArray[Scalar[DTYPE], V_SIZE](uninitialized=True)
    ldl_factor_gpu[DTYPE, NV, M_SIZE, V_SIZE](M, L, D)

    var M_inv = InlineArray[Scalar[DTYPE], M_SIZE](uninitialized=True)
    for i in range(M_SIZE):
        M_inv[i] = Scalar[DTYPE](0)
    compute_M_inv_from_ldl_gpu[DTYPE, NV, M_SIZE, V_SIZE](L, D, M_inv)

    # 8. Compute bias forces (full RNE: gravity + Coriolis + centrifugal)
    compute_bias_forces_rne_gpu[
        DTYPE,
        NQ,
        NV,
        NBODY,
        NJOINT,
        MAX_CONTACTS,
        STATE_SIZE,
        MODEL_SIZE,
        V_SIZE,
        CDOF_SIZE,
        BATCH,
    ](env, state, model, cdof, bias)

    # 9. Compute unconstrained acceleration via LDL solve
    var qvel_off = qvel_offset[NQ, NV]()
    var qacc_off = qacc_offset[NQ, NV]()
    var qfrc_off = qfrc_offset[NQ, NV]()
    var model_meta_off = model_metadata_offset[NBODY, NJOINT]()
    var dt = rebind[Scalar[DTYPE]](
        model[0, model_meta_off + MODEL_META_IDX_TIMESTEP]
    )

    var f_net = InlineArray[Scalar[DTYPE], V_SIZE](uninitialized=True)
    for i in range(NV):
        var qfrc = rebind[Scalar[DTYPE]](state[env, qfrc_off + i])
        f_net[i] = qfrc - bias[i]

    # 8b. Apply passive joint forces: stiffness only
    # Damping is handled implicitly via M_eff (step 6b).
    var qpos_off_stiff = qpos_offset[NQ, NV]()
    for j in range(NJOINT):
        var joint_off = model_joint_offset[NBODY](j)
        var jnt_type = Int(
            rebind[Scalar[DTYPE]](model[0, joint_off + JOINT_IDX_TYPE])
        )
        var dof_adr = Int(
            rebind[Scalar[DTYPE]](model[0, joint_off + JOINT_IDX_DOF_ADR])
        )
        var qpos_adr = Int(
            rebind[Scalar[DTYPE]](model[0, joint_off + JOINT_IDX_QPOS_ADR])
        )
        var stiff = rebind[Scalar[DTYPE]](
            model[0, joint_off + JOINT_IDX_STIFFNESS]
        )
        if stiff > Scalar[DTYPE](0):
            if jnt_type == JNT_FREE:
                for d in range(6):
                    var qpos_d = rebind[Scalar[DTYPE]](
                        state[env, qpos_off_stiff + qpos_adr + d]
                    )
                    f_net[dof_adr + d] = f_net[dof_adr + d] - stiff * qpos_d
            elif jnt_type == JNT_BALL:
                for d in range(3):
                    var qpos_d = rebind[Scalar[DTYPE]](
                        state[env, qpos_off_stiff + qpos_adr + d]
                    )
                    f_net[dof_adr + d] = f_net[dof_adr + d] - stiff * qpos_d
            else:
                # Hinge/slide: f = -stiffness * qpos
                var qpos_d = rebind[Scalar[DTYPE]](
                    state[env, qpos_off_stiff + qpos_adr]
                )
                f_net[dof_adr] = f_net[dof_adr] - stiff * qpos_d

    var qacc = InlineArray[Scalar[DTYPE], V_SIZE](uninitialized=True)
    for i in range(NV):
        qacc[i] = Scalar[DTYPE](0)
    ldl_solve_gpu[DTYPE, NV, M_SIZE, V_SIZE](L, D, f_net, qacc)

    for i in range(NV):
        state[env, qacc_off + i] = qacc[i]

    # 10. Predict velocity
    var qvel_pred = InlineArray[Scalar[DTYPE], V_SIZE](uninitialized=True)
    for i in range(NV):
        var qvel = rebind[Scalar[DTYPE]](state[env, qvel_off + i])
        qvel_pred[i] = qvel + qacc[i] * dt

    # 11. Constraint solve using parametrized solver with full M_inv
    SOLVER.solve_gpu[
        DTYPE,
        NQ,
        NV,
        NBODY,
        NJOINT,
        MAX_CONTACTS,
        STATE_SIZE,
        MODEL_SIZE,
        V_SIZE,
        M_SIZE,
        CDOF_SIZE,
        BATCH,
    ](env, state, model, M_inv, cdof, qvel_pred, dt)

    # 9. Write back constrained velocity and integrate position
    var qpos_off = qpos_offset[NQ, NV]()
    for i in range(NV):
        state[env, qvel_off + i] = qvel_pred[i]

    # 9b. Clamp velocities to prevent divergence
    # MuJoCo uses ~10-50 depending on model; 20 is reasonable for walking robots
    comptime MAX_QVEL: Scalar[DTYPE] = 20.0
    for i in range(NV):
        var v = rebind[Scalar[DTYPE]](state[env, qvel_off + i])
        if v > MAX_QVEL:
            state[env, qvel_off + i] = MAX_QVEL
        elif v < -MAX_QVEL:
            state[env, qvel_off + i] = -MAX_QVEL

    for i in range(NQ):
        if i < NV:
            var qpos = rebind[Scalar[DTYPE]](state[env, qpos_off + i])
            var qvel = rebind[Scalar[DTYPE]](state[env, qvel_off + i])
            state[env, qpos_off + i] = qpos + qvel * dt

    # 10. Normalize quaternions
    normalize_qpos_quaternions_gpu[
        DTYPE,
        NQ,
        NV,
        NBODY,
        NJOINT,
        MAX_CONTACTS,
        STATE_SIZE,
        MODEL_SIZE,
        BATCH,
    ](env, state, model)

    # 11. Joint limits now enforced as constraints inside the solver
    # (no post-step clamping needed)


# Backward-compatible alias: uses PGS solver by default
@always_inline
fn step_constraint_kernel[
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
    """Complete GC physics step with PGS constraint solving (default)."""
    step_constraint_kernel_with_solver[
        DTYPE,
        NQ,
        NV,
        NBODY,
        NJOINT,
        MAX_CONTACTS,
        STATE_SIZE,
        MODEL_SIZE,
        BATCH,
        PGSSolver,
    ](env, state, model)
