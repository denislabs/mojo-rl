"""CPU/GPU Phase-by-Phase Integrator Comparison Test.

Runs the EulerIntegrator pipeline on CPU and GPU separately, capturing
all intermediate values, then compares phase-by-phase to pinpoint any
divergence between CPU and GPU code paths.

Uses a Hopper model (4 bodies, 6 joints, NQ=6, NV=6, MAX_CONTACTS=10).

Run with:
    cd mojo-rl
    pixi run -e apple mojo run physics3d/tests/test_integrator_phases_cpu_gpu.mojo
"""

from builtin.math import abs
from gpu.host import DeviceContext, DeviceBuffer, HostBuffer
from gpu import thread_idx, block_idx, block_dim
from layout import Layout, LayoutTensor

from physics3d.types import Model, Data, _max_one, compute_capsule_inertia
from physics3d.constants import GEOM_CAPSULE
from physics3d.joint_types import JNT_HINGE, JNT_SLIDE

# CPU functions
from physics3d.kinematics.forward_kinematics import (
    forward_kinematics,
    compute_body_velocities,
    forward_kinematics_gpu,
    compute_body_velocities_gpu,
)
from physics3d.dynamics.mass_matrix import (
    compute_mass_matrix_full,
    compute_mass_matrix_full_gpu,
    ldl_factor,
    ldl_factor_gpu,
    ldl_solve,
    ldl_solve_gpu,
    compute_M_inv_from_ldl,
    compute_M_inv_from_ldl_gpu,
)
from physics3d.dynamics.bias_forces import (
    compute_bias_forces_rne,
    compute_bias_forces_rne_gpu,
)
from physics3d.dynamics.jacobian import (
    compute_cdof,
    compute_cdof_gpu,
    compute_composite_inertia,
    compute_composite_inertia_gpu,
)
from physics3d.collision.contact_detection import (
    detect_ground_contacts,
    detect_ground_contacts_gpu,
    detect_body_body_contacts,
    detect_body_body_contacts_gpu,
    normalize_qpos_quaternions,
    normalize_qpos_quaternions_gpu,
)
from physics3d.solver.pgs_solver import PGSSolver

from physics3d.gpu.constants import (
    TPB,
    state_size,
    model_size,
    qpos_offset,
    qvel_offset,
    qacc_offset,
    qfrc_offset,
    xpos_offset,
    xquat_offset,
    xvel_offset,
    xangvel_offset,
    contacts_offset,
    metadata_offset,
    CONTACT_SIZE,
    META_IDX_NUM_CONTACTS,
    model_body_offset,
    model_joint_offset,
    model_metadata_offset,
    JOINT_IDX_TYPE,
    JOINT_IDX_BODY_ID,
    JOINT_IDX_QPOS_ADR,
    JOINT_IDX_DOF_ADR,
    JOINT_IDX_AXIS_X,
    JOINT_IDX_AXIS_Y,
    JOINT_IDX_AXIS_Z,
    JOINT_IDX_POS_X,
    JOINT_IDX_POS_Y,
    JOINT_IDX_POS_Z,
    JOINT_IDX_TAU_LIMIT,
    JOINT_IDX_RANGE_MIN,
    JOINT_IDX_RANGE_MAX,
    JOINT_IDX_ARMATURE,
    JOINT_IDX_DAMPING,
    JOINT_IDX_STIFFNESS,
    MODEL_META_IDX_NBODY,
    MODEL_META_IDX_NJOINT,
    MODEL_META_IDX_GRAVITY_X,
    MODEL_META_IDX_GRAVITY_Y,
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
    CONTACT_IDX_IMPULSE_N,
    CONTACT_IDX_IMPULSE_T1,
    CONTACT_IDX_IMPULSE_T2,
)
from physics3d.gpu.buffer_utils import (
    copy_model_to_buffer,
    copy_data_to_buffer,
)


# =============================================================================
# Hopper model dimensions
# =============================================================================

comptime DTYPE = DType.float32
comptime NQ: Int = 6
comptime NV: Int = 6
comptime NBODY: Int = 4
comptime NJOINT: Int = 6
comptime MAX_CONTACTS: Int = 10
comptime BATCH: Int = 1

comptime STATE_SIZE = state_size[NQ, NV, NBODY, MAX_CONTACTS]()
comptime MODEL_SIZE = model_size[NBODY, NJOINT]()
comptime V_SIZE: Int = 6
comptime M_SIZE: Int = 36  # NV*NV
comptime CDOF_SIZE: Int = 36  # NV*6
comptime CRB_SIZE: Int = 40  # NBODY*10

# Debug output buffer: intermediates that only live in local vars
# Layout: cdof(36) + crb(40) + M(36) + M_arm(36) + L(36) + D(6) + M_inv(36)
#         + bias(6) + f_net(6) + qacc_pre(6) + qvel_pred(6) + qvel_post(6)
comptime DBG_CDOF_OFF: Int = 0
comptime DBG_CRB_OFF: Int = 36
comptime DBG_M_OFF: Int = 76
comptime DBG_M_ARM_OFF: Int = 112
comptime DBG_L_OFF: Int = 148
comptime DBG_D_OFF: Int = 184
comptime DBG_M_INV_OFF: Int = 190
comptime DBG_BIAS_OFF: Int = 226
comptime DBG_FNET_OFF: Int = 232
comptime DBG_QACC_OFF: Int = 238
comptime DBG_QVEL_PRED_OFF: Int = 244
comptime DBG_QVEL_POST_OFF: Int = 250
comptime DBG_SIZE: Int = 256

comptime TOL: Float32 = 1e-4


# =============================================================================
# Helper: max absolute error between arrays
# =============================================================================

fn max_abs_error(
    a: UnsafePointer[Scalar[DTYPE]],
    b: UnsafePointer[Scalar[DTYPE]],
    n: Int,
) -> Float64:
    var max_err = Float64(0)
    for i in range(n):
        var diff = abs(Float64(a[i]) - Float64(b[i]))
        if diff > max_err:
            max_err = diff
    return max_err


fn max_abs_error_ia_ptr[SIZE: Int](
    a: InlineArray[Scalar[DTYPE], SIZE],
    b: UnsafePointer[Scalar[DTYPE]],
    n: Int,
) -> Float64:
    var max_err = Float64(0)
    for i in range(n):
        var diff = abs(Float64(a[i]) - Float64(b[i]))
        if diff > max_err:
            max_err = diff
    return max_err


fn report(phase: String, max_err: Float64) -> Bool:
    var status = String("PASS") if max_err < Float64(TOL) else String("FAIL")
    print(phase, " max_err=", max_err, " ", status)
    return max_err < Float64(TOL)


# create_hopper_model is inlined into main() because Model is not Movable


# =============================================================================
# GPU instrumented kernel: runs each phase and dumps intermediates
# =============================================================================

@always_inline
fn instrumented_kernel(
    state: LayoutTensor[DTYPE, Layout.row_major(BATCH, STATE_SIZE), MutAnyOrigin],
    model: LayoutTensor[DTYPE, Layout.row_major(1, MODEL_SIZE), MutAnyOrigin],
    debug: LayoutTensor[DTYPE, Layout.row_major(BATCH, DBG_SIZE), MutAnyOrigin],
):
    var env = Int(block_dim.x * block_idx.x + thread_idx.x)
    if env >= BATCH:
        return

    # --- Phase 1: Forward Kinematics ---
    forward_kinematics_gpu[
        DTYPE, NQ, NV, NBODY, NJOINT, MAX_CONTACTS, STATE_SIZE, MODEL_SIZE, BATCH,
    ](env, state, model)

    # --- Phase 2: Body Velocities ---
    compute_body_velocities_gpu[
        DTYPE, NQ, NV, NBODY, NJOINT, MAX_CONTACTS, STATE_SIZE, MODEL_SIZE, BATCH,
    ](env, state, model)

    # --- Phase 3: Contact Detection ---
    detect_ground_contacts_gpu[
        DTYPE, NQ, NV, NBODY, NJOINT, MAX_CONTACTS, STATE_SIZE, MODEL_SIZE, BATCH,
    ](env, state, model)
    detect_body_body_contacts_gpu[
        DTYPE, NQ, NV, NBODY, NJOINT, MAX_CONTACTS, STATE_SIZE, MODEL_SIZE, BATCH,
    ](env, state, model)

    # --- Phase 4: CDOF ---
    var cdof = InlineArray[Scalar[DTYPE], CDOF_SIZE](uninitialized=True)
    for i in range(CDOF_SIZE):
        cdof[i] = Scalar[DTYPE](0)
    compute_cdof_gpu[
        DTYPE, NQ, NV, NBODY, NJOINT, MAX_CONTACTS, STATE_SIZE, MODEL_SIZE, CDOF_SIZE, BATCH,
    ](env, state, model, cdof)

    # Dump cdof
    for i in range(CDOF_SIZE):
        debug[env, DBG_CDOF_OFF + i] = cdof[i]

    # --- Phase 5: Composite Inertia ---
    var crb = InlineArray[Scalar[DTYPE], CRB_SIZE](uninitialized=True)
    for i in range(CRB_SIZE):
        crb[i] = Scalar[DTYPE](0)
    compute_composite_inertia_gpu[
        DTYPE, NQ, NV, NBODY, NJOINT, MAX_CONTACTS, STATE_SIZE, MODEL_SIZE, CRB_SIZE, BATCH,
    ](env, state, model, crb)

    # Dump crb
    for i in range(CRB_SIZE):
        debug[env, DBG_CRB_OFF + i] = crb[i]

    # --- Phase 6: Mass Matrix (CRBA) ---
    var M = InlineArray[Scalar[DTYPE], M_SIZE](uninitialized=True)
    for i in range(M_SIZE):
        M[i] = Scalar[DTYPE](0)
    compute_mass_matrix_full_gpu[
        DTYPE, NQ, NV, NBODY, NJOINT, MAX_CONTACTS, STATE_SIZE, MODEL_SIZE,
        M_SIZE, CDOF_SIZE, CRB_SIZE, BATCH,
    ](env, state, model, cdof, crb, M)

    # Dump M (before armature)
    for i in range(M_SIZE):
        debug[env, DBG_M_OFF + i] = M[i]

    # --- Phase 6b: Armature + implicit damping ---
    var model_meta_off = model_metadata_offset[NBODY, NJOINT]()
    var dt = rebind[Scalar[DTYPE]](model[0, model_meta_off + MODEL_META_IDX_TIMESTEP])
    for j in range(NJOINT):
        var joint_off = model_joint_offset[NBODY](j)
        var jnt_type = Int(rebind[Scalar[DTYPE]](model[0, joint_off + JOINT_IDX_TYPE]))
        var dof_adr = Int(rebind[Scalar[DTYPE]](model[0, joint_off + JOINT_IDX_DOF_ADR]))
        var arm = rebind[Scalar[DTYPE]](model[0, joint_off + JOINT_IDX_ARMATURE])
        var damp = rebind[Scalar[DTYPE]](model[0, joint_off + JOINT_IDX_DAMPING])
        var diag_add = arm + dt * damp
        if jnt_type == JNT_SLIDE or jnt_type == JNT_HINGE:
            M[dof_adr * NV + dof_adr] = M[dof_adr * NV + dof_adr] + diag_add

    # Dump M after armature
    for i in range(M_SIZE):
        debug[env, DBG_M_ARM_OFF + i] = M[i]

    # --- Phase 7: LDL Factorization ---
    var L = InlineArray[Scalar[DTYPE], M_SIZE](uninitialized=True)
    var D = InlineArray[Scalar[DTYPE], V_SIZE](uninitialized=True)
    ldl_factor_gpu[DTYPE, NV, M_SIZE, V_SIZE](M, L, D)

    # Dump L and D
    for i in range(M_SIZE):
        debug[env, DBG_L_OFF + i] = L[i]
    for i in range(V_SIZE):
        debug[env, DBG_D_OFF + i] = D[i]

    # --- Phase 7b: M_inv ---
    var M_inv = InlineArray[Scalar[DTYPE], M_SIZE](uninitialized=True)
    for i in range(M_SIZE):
        M_inv[i] = Scalar[DTYPE](0)
    compute_M_inv_from_ldl_gpu[DTYPE, NV, M_SIZE, V_SIZE](L, D, M_inv)

    for i in range(M_SIZE):
        debug[env, DBG_M_INV_OFF + i] = M_inv[i]

    # --- Phase 8: Bias Forces (RNE) ---
    var bias = InlineArray[Scalar[DTYPE], V_SIZE](uninitialized=True)
    for i in range(V_SIZE):
        bias[i] = Scalar[DTYPE](0)
    compute_bias_forces_rne_gpu[
        DTYPE, NQ, NV, NBODY, NJOINT, MAX_CONTACTS, STATE_SIZE, MODEL_SIZE,
        V_SIZE, CDOF_SIZE, BATCH,
    ](env, state, model, cdof, bias)

    for i in range(V_SIZE):
        debug[env, DBG_BIAS_OFF + i] = bias[i]

    # --- Phase 9: Net Forces + Stiffness ---
    var qfrc_off = qfrc_offset[NQ, NV]()
    var qpos_off_stiff = qpos_offset[NQ, NV]()
    var f_net = InlineArray[Scalar[DTYPE], V_SIZE](uninitialized=True)
    for i in range(NV):
        var qfrc = rebind[Scalar[DTYPE]](state[env, qfrc_off + i])
        f_net[i] = qfrc - bias[i]

    for j in range(NJOINT):
        var joint_off = model_joint_offset[NBODY](j)
        var jnt_type = Int(rebind[Scalar[DTYPE]](model[0, joint_off + JOINT_IDX_TYPE]))
        var dof_adr = Int(rebind[Scalar[DTYPE]](model[0, joint_off + JOINT_IDX_DOF_ADR]))
        var qpos_adr = Int(rebind[Scalar[DTYPE]](model[0, joint_off + JOINT_IDX_QPOS_ADR]))
        var stiff = rebind[Scalar[DTYPE]](model[0, joint_off + JOINT_IDX_STIFFNESS])
        if stiff > Scalar[DTYPE](0):
            if jnt_type == JNT_SLIDE or jnt_type == JNT_HINGE:
                var qpos_d = rebind[Scalar[DTYPE]](state[env, qpos_off_stiff + qpos_adr])
                f_net[dof_adr] = f_net[dof_adr] - stiff * qpos_d

    for i in range(V_SIZE):
        debug[env, DBG_FNET_OFF + i] = f_net[i]

    # --- Phase 10: Unconstrained Accel (LDL solve) ---
    var qacc = InlineArray[Scalar[DTYPE], V_SIZE](uninitialized=True)
    for i in range(NV):
        qacc[i] = Scalar[DTYPE](0)
    ldl_solve_gpu[DTYPE, NV, M_SIZE, V_SIZE](L, D, f_net, qacc)

    var qacc_off = qacc_offset[NQ, NV]()
    for i in range(NV):
        state[env, qacc_off + i] = qacc[i]

    for i in range(V_SIZE):
        debug[env, DBG_QACC_OFF + i] = qacc[i]

    # --- Phase 11: Predicted Velocity ---
    var qvel_off = qvel_offset[NQ, NV]()
    var qvel_pred = InlineArray[Scalar[DTYPE], V_SIZE](uninitialized=True)
    for i in range(NV):
        var qvel = rebind[Scalar[DTYPE]](state[env, qvel_off + i])
        qvel_pred[i] = qvel + qacc[i] * dt

    for i in range(V_SIZE):
        debug[env, DBG_QVEL_PRED_OFF + i] = qvel_pred[i]

    # --- Phase 12: Constraint Solve (PGS) ---
    PGSSolver.solve_gpu[
        DTYPE, NQ, NV, NBODY, NJOINT, MAX_CONTACTS, STATE_SIZE, MODEL_SIZE,
        V_SIZE, M_SIZE, CDOF_SIZE, BATCH,
    ](env, state, model, M_inv, cdof, qvel_pred, dt)

    for i in range(V_SIZE):
        debug[env, DBG_QVEL_POST_OFF + i] = qvel_pred[i]

    # --- Phase 13: Write back + integrate ---
    var qpos_off = qpos_offset[NQ, NV]()
    for i in range(NV):
        var old_qvel = rebind[Scalar[DTYPE]](state[env, qvel_off + i])
        state[env, qacc_off + i] = (qvel_pred[i] - old_qvel) / dt
        state[env, qvel_off + i] = qvel_pred[i]

    # Clamp velocities
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

    normalize_qpos_quaternions_gpu[
        DTYPE, NQ, NV, NBODY, NJOINT, MAX_CONTACTS, STATE_SIZE, MODEL_SIZE, BATCH,
    ](env, state, model)


# =============================================================================
# Main
# =============================================================================

fn main() raises:
    print("=" * 60)
    print("Physics3D Integrator CPU vs GPU Phase Comparison Test")
    print("=" * 60)
    print("Model: Hopper (NBODY=4, NJOINT=6, NQ=6, NV=6)")

    # Initial state with perturbation to exercise all code paths
    var init_qpos = InlineArray[Scalar[DTYPE], NQ](uninitialized=True)
    init_qpos[0] = Scalar[DTYPE](0.0)     # rootx
    init_qpos[1] = Scalar[DTYPE](1.05)    # rootz (low enough for foot contacts)
    init_qpos[2] = Scalar[DTYPE](0.1)     # rooty
    init_qpos[3] = Scalar[DTYPE](-0.2)    # thigh
    init_qpos[4] = Scalar[DTYPE](-0.1)    # leg
    init_qpos[5] = Scalar[DTYPE](0.05)    # foot

    var init_qvel = InlineArray[Scalar[DTYPE], NV](uninitialized=True)
    init_qvel[0] = Scalar[DTYPE](0.5)     # rootx vel
    init_qvel[1] = Scalar[DTYPE](-0.3)    # rootz vel
    init_qvel[2] = Scalar[DTYPE](0.1)     # rooty vel
    init_qvel[3] = Scalar[DTYPE](0.2)     # thigh vel
    init_qvel[4] = Scalar[DTYPE](-0.1)    # leg vel
    init_qvel[5] = Scalar[DTYPE](0.05)    # foot vel

    var init_qfrc = InlineArray[Scalar[DTYPE], NV](uninitialized=True)
    init_qfrc[0] = Scalar[DTYPE](0.0)     # rootx (not actuated)
    init_qfrc[1] = Scalar[DTYPE](0.0)     # rootz (not actuated)
    init_qfrc[2] = Scalar[DTYPE](0.0)     # rooty (not actuated)
    init_qfrc[3] = Scalar[DTYPE](50.0)    # thigh torque
    init_qfrc[4] = Scalar[DTYPE](-30.0)   # leg torque
    init_qfrc[5] = Scalar[DTYPE](20.0)    # foot torque

    print("Initial state: qpos=[0, 1.05, 0.1, -0.2, -0.1, 0.05]")
    print("               qvel=[0.5, -0.3, 0.1, 0.2, -0.1, 0.05]")
    print("               qfrc=[0, 0, 0, 50, -30, 20]")
    print()

    # =====================================================================
    # Create CPU model and data (inline, Model is not Movable)
    # =====================================================================
    var torso_mass = Scalar[DTYPE](3.53429174)
    var torso_radius = Scalar[DTYPE](0.05)
    var torso_half_length = Scalar[DTYPE](0.2)
    var thigh_mass = Scalar[DTYPE](3.92699082)
    var thigh_radius = Scalar[DTYPE](0.05)
    var thigh_half_length = Scalar[DTYPE](0.225)
    var leg_mass = Scalar[DTYPE](2.71433605)
    var leg_radius = Scalar[DTYPE](0.04)
    var leg_half_length = Scalar[DTYPE](0.25)
    var foot_mass = Scalar[DTYPE](5.0893801)
    var foot_radius = Scalar[DTYPE](0.06)
    var foot_half_length = Scalar[DTYPE](0.195)

    var model = Model[DTYPE, NQ, NV, NBODY, NJOINT, MAX_CONTACTS](
        gravity_z=Scalar[DTYPE](-9.81),
        timestep=Scalar[DTYPE](0.002),
        ground_z=Scalar[DTYPE](0.0),
        friction=Scalar[DTYPE](0.5),
    )

    # Body 0: Torso
    var torso_inertia = compute_capsule_inertia(torso_mass, torso_radius, torso_half_length)
    model.set_body(0, mass=torso_mass, inertia=torso_inertia, radius=torso_radius)
    model.set_body_parent(0, -1)
    model.body_geom_type[0] = GEOM_CAPSULE
    model.body_half_length[0] = torso_half_length

    # Body 1: Thigh
    var thigh_inertia = compute_capsule_inertia(thigh_mass, thigh_radius, thigh_half_length)
    model.set_body(1, mass=thigh_mass, inertia=thigh_inertia, radius=thigh_radius)
    model.set_body_parent(1, 0)
    model.body_geom_type[1] = GEOM_CAPSULE
    model.body_half_length[1] = thigh_half_length
    model.set_body_local_frame(
        1, pos=(Scalar[DTYPE](0.0), Scalar[DTYPE](0.0), -(torso_half_length + thigh_half_length)),
    )

    # Body 2: Leg
    var leg_inertia = compute_capsule_inertia(leg_mass, leg_radius, leg_half_length)
    model.set_body(2, mass=leg_mass, inertia=leg_inertia, radius=leg_radius)
    model.set_body_parent(2, 1)
    model.body_geom_type[2] = GEOM_CAPSULE
    model.body_half_length[2] = leg_half_length
    model.set_body_local_frame(
        2, pos=(Scalar[DTYPE](0.0), Scalar[DTYPE](0.0), -(thigh_half_length + leg_half_length)),
    )

    # Body 3: Foot (horizontal capsule, 90 deg rotation around Y)
    var foot_inertia = compute_capsule_inertia(foot_mass, foot_radius, foot_half_length)
    model.set_body(3, mass=foot_mass, inertia=foot_inertia, radius=foot_radius)
    model.set_body_parent(3, 2)
    model.body_geom_type[3] = GEOM_CAPSULE
    model.body_half_length[3] = foot_half_length
    model.set_body_local_frame(
        3, pos=(Scalar[DTYPE](0.0), Scalar[DTYPE](0.0), -leg_half_length),
        quat=(Scalar[DTYPE](0.0), Scalar[DTYPE](0.70710678), Scalar[DTYPE](0.0), Scalar[DTYPE](0.70710678)),
    )

    # Joint 0: rootx (slide X, body 0)
    _ = model.add_slide_joint(
        body_id=0,
        pos=(Scalar[DTYPE](0.0), Scalar[DTYPE](0.0), Scalar[DTYPE](0.0)),
        axis=(Scalar[DTYPE](1.0), Scalar[DTYPE](0.0), Scalar[DTYPE](0.0)),
        force_limit=Scalar[DTYPE](0.0),
    )
    # Joint 1: rootz (slide Z, body 0)
    _ = model.add_slide_joint(
        body_id=0,
        pos=(Scalar[DTYPE](0.0), Scalar[DTYPE](0.0), Scalar[DTYPE](0.0)),
        axis=(Scalar[DTYPE](0.0), Scalar[DTYPE](0.0), Scalar[DTYPE](1.0)),
        force_limit=Scalar[DTYPE](0.0),
    )
    # Joint 2: rooty (hinge Y, body 0)
    _ = model.add_hinge_joint(
        body_id=0,
        pos=(Scalar[DTYPE](0.0), Scalar[DTYPE](0.0), Scalar[DTYPE](0.0)),
        axis=(Scalar[DTYPE](0.0), Scalar[DTYPE](1.0), Scalar[DTYPE](0.0)),
        tau_limit=Scalar[DTYPE](0.0),
    )
    # Joint 3: thigh (hinge Y, body 1)
    _ = model.add_hinge_joint(
        body_id=1,
        pos=(Scalar[DTYPE](0.0), Scalar[DTYPE](0.0), -torso_half_length),
        axis=(Scalar[DTYPE](0.0), Scalar[DTYPE](1.0), Scalar[DTYPE](0.0)),
        tau_limit=Scalar[DTYPE](200.0),
        range_min=Scalar[DTYPE](-2.618), range_max=Scalar[DTYPE](0.0),
        armature=Scalar[DTYPE](1.0), damping=Scalar[DTYPE](1.0),
    )
    # Joint 4: leg (hinge Y, body 2)
    _ = model.add_hinge_joint(
        body_id=2,
        pos=(Scalar[DTYPE](0.0), Scalar[DTYPE](0.0), -thigh_half_length),
        axis=(Scalar[DTYPE](0.0), Scalar[DTYPE](1.0), Scalar[DTYPE](0.0)),
        tau_limit=Scalar[DTYPE](200.0),
        range_min=Scalar[DTYPE](-2.618), range_max=Scalar[DTYPE](0.0),
        armature=Scalar[DTYPE](1.0), damping=Scalar[DTYPE](1.0),
    )
    # Joint 5: foot (hinge Y, body 3)
    _ = model.add_hinge_joint(
        body_id=3,
        pos=(Scalar[DTYPE](0.0), Scalar[DTYPE](0.0), -leg_half_length),
        axis=(Scalar[DTYPE](0.0), Scalar[DTYPE](1.0), Scalar[DTYPE](0.0)),
        tau_limit=Scalar[DTYPE](200.0),
        range_min=Scalar[DTYPE](-0.785), range_max=Scalar[DTYPE](0.785),
        armature=Scalar[DTYPE](1.0), damping=Scalar[DTYPE](1.0),
    )

    var data = Data[DTYPE, NQ, NV, NBODY, NJOINT, MAX_CONTACTS]()

    for i in range(NQ):
        data.qpos[i] = init_qpos[i]
    for i in range(NV):
        data.qvel[i] = init_qvel[i]
    for i in range(NV):
        data.qfrc[i] = init_qfrc[i]

    # =====================================================================
    # CPU PIPELINE: run each phase manually, capture intermediates
    # =====================================================================

    # Phase 1: Forward kinematics
    forward_kinematics(model, data)

    # Phase 2: Body velocities
    compute_body_velocities(model, data)

    # Phase 3: Contact detection
    detect_ground_contacts(model, data)
    detect_body_body_contacts(model, data)
    var cpu_num_contacts = data.num_contacts

    # Phase 4: CDOF
    var cpu_cdof = InlineArray[Scalar[DTYPE], CDOF_SIZE](uninitialized=True)
    for i in range(CDOF_SIZE):
        cpu_cdof[i] = Scalar[DTYPE](0)
    compute_cdof[DTYPE, NQ, NV, NBODY, NJOINT, MAX_CONTACTS, CDOF_SIZE](
        model, data, cpu_cdof
    )

    # Phase 5: Composite Inertia
    var cpu_crb = InlineArray[Scalar[DTYPE], CRB_SIZE](uninitialized=True)
    for i in range(CRB_SIZE):
        cpu_crb[i] = Scalar[DTYPE](0)
    compute_composite_inertia[DTYPE, NQ, NV, NBODY, NJOINT, MAX_CONTACTS, CRB_SIZE](
        model, data, cpu_crb
    )

    # Phase 6: Mass Matrix (CRBA)
    var cpu_M = InlineArray[Scalar[DTYPE], M_SIZE](uninitialized=True)
    for i in range(M_SIZE):
        cpu_M[i] = Scalar[DTYPE](0)
    compute_mass_matrix_full[
        DTYPE, NQ, NV, NBODY, NJOINT, MAX_CONTACTS, M_SIZE, CDOF_SIZE, CRB_SIZE,
    ](model, data, cpu_cdof, cpu_crb, cpu_M)

    # Save copy before armature
    var cpu_M_pre_arm = InlineArray[Scalar[DTYPE], M_SIZE](uninitialized=True)
    for i in range(M_SIZE):
        cpu_M_pre_arm[i] = cpu_M[i]

    # Phase 6b: Armature + implicit damping
    var dt = model.timestep
    for j in range(model.num_joints):
        var joint = model.joints[j]
        var dof_adr = joint.dof_adr
        var arm = joint.armature
        var damp = joint.damping
        var diag_add = arm + dt * damp
        if joint.jnt_type == JNT_SLIDE or joint.jnt_type == JNT_HINGE:
            cpu_M[dof_adr * NV + dof_adr] = cpu_M[dof_adr * NV + dof_adr] + diag_add

    # Phase 7: LDL Factorization
    var cpu_L = InlineArray[Scalar[DTYPE], M_SIZE](uninitialized=True)
    var cpu_D = InlineArray[Scalar[DTYPE], V_SIZE](uninitialized=True)
    ldl_factor[DTYPE, NV, M_SIZE, V_SIZE](cpu_M, cpu_L, cpu_D)

    # Phase 7b: M_inv
    var cpu_M_inv = InlineArray[Scalar[DTYPE], M_SIZE](uninitialized=True)
    for i in range(M_SIZE):
        cpu_M_inv[i] = Scalar[DTYPE](0)
    compute_M_inv_from_ldl[DTYPE, NV, M_SIZE, V_SIZE](cpu_L, cpu_D, cpu_M_inv)

    # Phase 8: Bias Forces (RNE)
    var cpu_bias = InlineArray[Scalar[DTYPE], V_SIZE](uninitialized=True)
    for i in range(V_SIZE):
        cpu_bias[i] = Scalar[DTYPE](0)
    compute_bias_forces_rne[
        DTYPE, NQ, NV, NBODY, NJOINT, MAX_CONTACTS, V_SIZE, CDOF_SIZE,
    ](model, data, cpu_cdof, cpu_bias)

    # Phase 9: Net Forces + Stiffness
    var cpu_f_net = InlineArray[Scalar[DTYPE], V_SIZE](uninitialized=True)
    for i in range(NV):
        cpu_f_net[i] = data.qfrc[i] - cpu_bias[i]

    for j in range(model.num_joints):
        var joint = model.joints[j]
        var dof_adr = joint.dof_adr
        var qpos_adr = joint.qpos_adr
        var stiff = joint.stiffness
        if stiff > Scalar[DTYPE](0):
            if joint.jnt_type == JNT_SLIDE or joint.jnt_type == JNT_HINGE:
                cpu_f_net[dof_adr] = cpu_f_net[dof_adr] - stiff * data.qpos[qpos_adr]

    # Phase 10: Unconstrained Accel
    var cpu_qacc = InlineArray[Scalar[DTYPE], V_SIZE](uninitialized=True)
    for i in range(NV):
        cpu_qacc[i] = Scalar[DTYPE](0)
    ldl_solve[DTYPE, NV, M_SIZE, V_SIZE](cpu_L, cpu_D, cpu_f_net, cpu_qacc)

    # Phase 11: Predicted Velocity
    var cpu_qvel_pred = InlineArray[Scalar[DTYPE], V_SIZE](uninitialized=True)
    for i in range(NV):
        cpu_qvel_pred[i] = data.qvel[i] + cpu_qacc[i] * dt

    # Phase 12: Constraint Solve (PGS)
    PGSSolver.solve[
        DTYPE, NQ, NV, NBODY, NJOINT, MAX_CONTACTS, V_SIZE, M_SIZE, CDOF_SIZE,
    ](model, data, cpu_M_inv, cpu_cdof, cpu_qvel_pred, dt)

    # Save post-constraint velocity
    var cpu_qvel_post = InlineArray[Scalar[DTYPE], V_SIZE](uninitialized=True)
    for i in range(NV):
        cpu_qvel_post[i] = cpu_qvel_pred[i]

    # Phase 13: Write back + integrate
    for i in range(NV):
        data.qacc[i] = (cpu_qvel_pred[i] - data.qvel[i]) / dt
        data.qvel[i] = cpu_qvel_pred[i]

    comptime MAX_QVEL: Scalar[DTYPE] = 20.0
    for i in range(NV):
        if data.qvel[i] > MAX_QVEL:
            data.qvel[i] = MAX_QVEL
        elif data.qvel[i] < -MAX_QVEL:
            data.qvel[i] = -MAX_QVEL

    for i in range(NQ):
        if i < NV:
            data.qpos[i] = data.qpos[i] + data.qvel[i] * dt

    normalize_qpos_quaternions(model, data)

    # =====================================================================
    # GPU PIPELINE: set up buffers, launch instrumented kernel
    # =====================================================================
    var ctx = DeviceContext()

    # Create host buffers
    var state_host = ctx.enqueue_create_host_buffer[DTYPE](BATCH * STATE_SIZE)
    var model_host = ctx.enqueue_create_host_buffer[DTYPE](MODEL_SIZE)
    var debug_host = ctx.enqueue_create_host_buffer[DTYPE](BATCH * DBG_SIZE)

    for i in range(BATCH * STATE_SIZE):
        state_host[i] = Scalar[DTYPE](0)
    for i in range(MODEL_SIZE):
        model_host[i] = Scalar[DTYPE](0)
    for i in range(BATCH * DBG_SIZE):
        debug_host[i] = Scalar[DTYPE](0)

    # Copy model to host buffer using the existing helper
    copy_model_to_buffer[DTYPE, NQ, NV, NBODY, NJOINT, MAX_CONTACTS](model, model_host)

    # Copy initial state to host buffer (env_idx=0)
    # We need a fresh Data with the initial state (before CPU pipeline modified it)
    var data_gpu = Data[DTYPE, NQ, NV, NBODY, NJOINT, MAX_CONTACTS]()
    for i in range(NQ):
        data_gpu.qpos[i] = init_qpos[i]
    for i in range(NV):
        data_gpu.qvel[i] = init_qvel[i]
    for i in range(NV):
        data_gpu.qfrc[i] = init_qfrc[i]

    copy_data_to_buffer[DTYPE, NQ, NV, NBODY, NJOINT, MAX_CONTACTS](data_gpu, state_host, 0)

    # Create GPU buffers and copy
    var state_buf = ctx.enqueue_create_buffer[DTYPE](BATCH * STATE_SIZE)
    var model_buf = ctx.enqueue_create_buffer[DTYPE](MODEL_SIZE)
    var debug_buf = ctx.enqueue_create_buffer[DTYPE](BATCH * DBG_SIZE)

    ctx.enqueue_copy(state_buf, state_host.unsafe_ptr())
    ctx.enqueue_copy(model_buf, model_host.unsafe_ptr())
    ctx.enqueue_copy(debug_buf, debug_host.unsafe_ptr())
    ctx.synchronize()

    # Launch instrumented kernel
    var st = LayoutTensor[DTYPE, Layout.row_major(BATCH, STATE_SIZE), MutAnyOrigin](
        state_buf.unsafe_ptr()
    )
    var md = LayoutTensor[DTYPE, Layout.row_major(1, MODEL_SIZE), MutAnyOrigin](
        model_buf.unsafe_ptr()
    )
    var dbg = LayoutTensor[DTYPE, Layout.row_major(BATCH, DBG_SIZE), MutAnyOrigin](
        debug_buf.unsafe_ptr()
    )

    ctx.enqueue_function[instrumented_kernel, instrumented_kernel](
        st, md, dbg,
        grid_dim=(1,),
        block_dim=(1,),
    )
    ctx.synchronize()

    # Copy results back to host
    ctx.enqueue_copy(state_host.unsafe_ptr(), state_buf)
    ctx.enqueue_copy(debug_host.unsafe_ptr(), debug_buf)
    ctx.synchronize()

    # =====================================================================
    # COMPARISON: Phase-by-phase
    # =====================================================================
    var gpu_state = state_host.unsafe_ptr()
    var gpu_dbg = debug_host.unsafe_ptr()
    var all_passed = True
    print()

    # --- Phase 1: Forward Kinematics (xpos, xquat) ---
    var xp_off = xpos_offset[NQ, NV, NBODY]()
    var xq_off = xquat_offset[NQ, NV, NBODY]()
    var xpos_err = Float64(0)
    for i in range(NBODY * 3):
        var diff = abs(Float64(data.xpos[i]) - Float64(gpu_state[xp_off + i]))
        if diff > xpos_err:
            xpos_err = diff
    var xquat_err = Float64(0)
    for i in range(NBODY * 4):
        var diff = abs(Float64(data.xquat[i]) - Float64(gpu_state[xq_off + i]))
        if diff > xquat_err:
            xquat_err = diff
    var fk_err = xpos_err if xpos_err > xquat_err else xquat_err
    all_passed = report("Phase  1: Forward Kinematics     ", fk_err) and all_passed

    # --- Phase 2: Body Velocities (xvel, xangvel) ---
    var xv_off = xvel_offset[NQ, NV, NBODY]()
    var xa_off = xangvel_offset[NQ, NV, NBODY]()
    var xvel_err = Float64(0)
    for i in range(NBODY * 3):
        var diff = abs(Float64(data.xvel[i]) - Float64(gpu_state[xv_off + i]))
        if diff > xvel_err:
            xvel_err = diff
    var xangvel_err = Float64(0)
    for i in range(NBODY * 3):
        var diff = abs(Float64(data.xangvel[i]) - Float64(gpu_state[xa_off + i]))
        if diff > xangvel_err:
            xangvel_err = diff
    var bv_err = xvel_err if xvel_err > xangvel_err else xangvel_err
    all_passed = report("Phase  2: Body Velocities        ", bv_err) and all_passed

    # --- Phase 3: Contact Detection ---
    var meta_off = metadata_offset[NQ, NV, NBODY, MAX_CONTACTS]()
    var gpu_num_contacts = Int(gpu_state[meta_off + META_IDX_NUM_CONTACTS])
    var contacts_match = cpu_num_contacts == gpu_num_contacts
    print(
        "Phase  3: Contact Detection       contacts: CPU=",
        cpu_num_contacts,
        " GPU=",
        gpu_num_contacts,
        " ",
        "PASS" if contacts_match else "FAIL",
    )
    if not contacts_match:
        all_passed = False

    # Compare contact data (if same count)
    if contacts_match and cpu_num_contacts > 0:
        var ct_off = contacts_offset[NQ, NV, NBODY]()
        var ct_err = Float64(0)
        for c in range(cpu_num_contacts):
            # Compare pos, normal, dist
            var cpu_c = data.contacts[c]
            var gpu_c_base = ct_off + c * CONTACT_SIZE
            var fields = InlineArray[Float64, 7](uninitialized=True)
            fields[0] = abs(Float64(cpu_c.pos_x) - Float64(gpu_state[gpu_c_base + CONTACT_IDX_POS_X]))
            fields[1] = abs(Float64(cpu_c.pos_y) - Float64(gpu_state[gpu_c_base + CONTACT_IDX_POS_Y]))
            fields[2] = abs(Float64(cpu_c.pos_z) - Float64(gpu_state[gpu_c_base + CONTACT_IDX_POS_Z]))
            fields[3] = abs(Float64(cpu_c.normal_x) - Float64(gpu_state[gpu_c_base + CONTACT_IDX_NX]))
            fields[4] = abs(Float64(cpu_c.normal_y) - Float64(gpu_state[gpu_c_base + CONTACT_IDX_NY]))
            fields[5] = abs(Float64(cpu_c.normal_z) - Float64(gpu_state[gpu_c_base + CONTACT_IDX_NZ]))
            fields[6] = abs(Float64(cpu_c.dist) - Float64(gpu_state[gpu_c_base + CONTACT_IDX_DIST]))
            for f in range(7):
                if fields[f] > ct_err:
                    ct_err = fields[f]
        all_passed = report("Phase  3: Contact Data           ", ct_err) and all_passed

    # --- Phase 4: CDOF ---
    var cdof_err = max_abs_error_ia_ptr[CDOF_SIZE](cpu_cdof, gpu_dbg + DBG_CDOF_OFF, CDOF_SIZE)
    all_passed = report("Phase  4: CDOF                   ", cdof_err) and all_passed

    # --- Phase 5: Composite Inertia ---
    var crb_err = max_abs_error_ia_ptr[CRB_SIZE](cpu_crb, gpu_dbg + DBG_CRB_OFF, CRB_SIZE)
    all_passed = report("Phase  5: Composite Inertia      ", crb_err) and all_passed

    # --- Phase 6: Mass Matrix (CRBA) ---
    var m_err = max_abs_error_ia_ptr[M_SIZE](cpu_M_pre_arm, gpu_dbg + DBG_M_OFF, M_SIZE)
    all_passed = report("Phase  6: Mass Matrix (CRBA)     ", m_err) and all_passed

    # --- Phase 6b: Mass Matrix (+ armature) ---
    var m_arm_err = max_abs_error_ia_ptr[M_SIZE](cpu_M, gpu_dbg + DBG_M_ARM_OFF, M_SIZE)
    all_passed = report("Phase 6b: Mass Matrix (+armature)", m_arm_err) and all_passed

    # --- Phase 7: LDL Factorization ---
    var l_err = max_abs_error_ia_ptr[M_SIZE](cpu_L, gpu_dbg + DBG_L_OFF, M_SIZE)
    var d_err = max_abs_error_ia_ptr[V_SIZE](cpu_D, gpu_dbg + DBG_D_OFF, V_SIZE)
    var ldl_err = l_err if l_err > d_err else d_err
    all_passed = report("Phase  7: LDL Factorization      ", ldl_err) and all_passed

    # --- Phase 7b: M_inv ---
    var minv_err = max_abs_error_ia_ptr[M_SIZE](cpu_M_inv, gpu_dbg + DBG_M_INV_OFF, M_SIZE)
    all_passed = report("Phase 7b: M_inv                  ", minv_err) and all_passed

    # --- Phase 8: Bias Forces (RNE) ---
    var bias_err = max_abs_error_ia_ptr[V_SIZE](cpu_bias, gpu_dbg + DBG_BIAS_OFF, V_SIZE)
    all_passed = report("Phase  8: Bias Forces (RNE)      ", bias_err) and all_passed

    # --- Phase 9: Net Forces ---
    var fnet_err = max_abs_error_ia_ptr[V_SIZE](cpu_f_net, gpu_dbg + DBG_FNET_OFF, V_SIZE)
    all_passed = report("Phase  9: Net Forces             ", fnet_err) and all_passed

    # --- Phase 10: Unconstrained Accel ---
    var qacc_err = max_abs_error_ia_ptr[V_SIZE](cpu_qacc, gpu_dbg + DBG_QACC_OFF, V_SIZE)
    all_passed = report("Phase 10: Unconstrained Accel    ", qacc_err) and all_passed

    # --- Phase 11: Predicted Velocity ---
    # cpu_qvel_pred was modified by PGS solve, use pre-solve values
    # We need to reconstruct pre-solve qvel_pred from qacc
    var cpu_qvel_pred_pre = InlineArray[Scalar[DTYPE], V_SIZE](uninitialized=True)
    for i in range(NV):
        cpu_qvel_pred_pre[i] = init_qvel[i] + cpu_qacc[i] * dt
    var qvp_err = max_abs_error_ia_ptr[V_SIZE](cpu_qvel_pred_pre, gpu_dbg + DBG_QVEL_PRED_OFF, V_SIZE)
    all_passed = report("Phase 11: Predicted Velocity     ", qvp_err) and all_passed

    # --- Phase 12: Constrained Velocity ---
    var qvpost_err = max_abs_error_ia_ptr[V_SIZE](cpu_qvel_post, gpu_dbg + DBG_QVEL_POST_OFF, V_SIZE)
    all_passed = report("Phase 12: Constrained Velocity   ", qvpost_err) and all_passed

    # --- Phase 13: Final State (qpos, qvel, qacc separately) ---
    var qp_off = qpos_offset[NQ, NV]()
    var qv_off = qvel_offset[NQ, NV]()
    var qa_off = qacc_offset[NQ, NV]()
    var qpos_final_err = Float64(0)
    for i in range(NQ):
        var diff = abs(Float64(data.qpos[i]) - Float64(gpu_state[qp_off + i]))
        if diff > qpos_final_err:
            qpos_final_err = diff
    var qvel_final_err = Float64(0)
    for i in range(NV):
        var diff = abs(Float64(data.qvel[i]) - Float64(gpu_state[qv_off + i]))
        if diff > qvel_final_err:
            qvel_final_err = diff
    var qacc_final_err = Float64(0)
    for i in range(NV):
        var diff = abs(Float64(data.qacc[i]) - Float64(gpu_state[qa_off + i]))
        if diff > qacc_final_err:
            qacc_final_err = diff
    all_passed = report("Phase 13: Final qpos             ", qpos_final_err) and all_passed
    all_passed = report("Phase 13: Final qvel             ", qvel_final_err) and all_passed
    # qacc = (qvel_post - qvel_old) / dt: dividing by dt=0.002 amplifies errors ~500x
    # Use relaxed tolerance proportional to 1/dt amplification
    var qacc_status = String("PASS") if qacc_final_err < Float64(TOL) * Float64(500) else String("FAIL")
    print("Phase 13: Final qacc              max_err=", qacc_final_err, " ", qacc_status)
    if qacc_final_err >= Float64(TOL) * Float64(500):
        all_passed = False

    print()
    if all_passed:
        print("ALL PHASES PASSED")
    else:
        print("SOME PHASES FAILED")
