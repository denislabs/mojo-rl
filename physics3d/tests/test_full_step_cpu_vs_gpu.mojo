"""Test Full Step (no contacts): CPU vs GPU.

Compares qpos/qvel after running N physics steps on CPU vs GPU for the
HalfCheetah model. Uses high rootz to avoid ground contacts so we test the
dynamics pipeline only (FK → M → bias → qacc → implicit damping → integrate).

The CPU uses EulerIntegrator[PrimalNewtonSolver].step() (float32).
The GPU uses EulerIntegrator[PrimalNewtonSolver].step_gpu() (float32).

Run with:
    cd mojo-rl && pixi run -e apple mojo run physics3d/tests/test_full_step_cpu_vs_gpu.mojo
"""

from math import abs
from collections import InlineArray
from gpu.host import DeviceContext, DeviceBuffer, HostBuffer
from layout import Layout, LayoutTensor

from physics3d.types import Model, Data, _max_one, ConeType
from physics3d.integrator.euler_integrator import EulerIntegrator
from physics3d.solver import PrimalNewtonSolver
from physics3d.kinematics.forward_kinematics import forward_kinematics
from physics3d.dynamics.mass_matrix import compute_body_invweight0
from physics3d.gpu.constants import (
    state_size,
    model_size,
    qpos_offset,
    qvel_offset,
    qfrc_offset,
    integrator_workspace_size,
)
from physics3d.gpu.buffer_utils import (
    create_state_buffer,
    create_model_buffer,
    copy_model_to_buffer,
    copy_data_to_buffer,
)
from envs.half_cheetah.half_cheetah_def import (
    HalfCheetahModel,
    HalfCheetahBodies,
    HalfCheetahJoints,
    HalfCheetahGeoms,
    HalfCheetahActuators,
    HalfCheetahParams,
)


# =============================================================================
# Constants
# =============================================================================

comptime DTYPE = DType.float32
comptime NQ = HalfCheetahModel.NQ  # 9
comptime NV = HalfCheetahModel.NV  # 9
comptime NBODY = HalfCheetahModel.NBODY
comptime NJOINT = HalfCheetahModel.NJOINT
comptime NGEOM = HalfCheetahModel.NGEOM
comptime MAX_CONTACTS = HalfCheetahParams[DTYPE].MAX_CONTACTS
comptime ACTION_DIM = HalfCheetahParams[DTYPE].ACTION_DIM
comptime BATCH = 1

comptime STATE_SIZE = state_size[NQ, NV, NBODY, MAX_CONTACTS]()
comptime MODEL_SIZE = model_size[NBODY, NJOINT]()
comptime WS_SIZE = integrator_workspace_size[NV, NBODY]() + NV * NV + PrimalNewtonSolver.solver_workspace_size[NV, MAX_CONTACTS]()

# Tolerances (float32 accumulates errors through full pipeline + integration)
comptime QPOS_ABS_TOL: Float64 = 1e-3
comptime QPOS_REL_TOL: Float64 = 1e-2
comptime QVEL_ABS_TOL: Float64 = 1e-2
comptime QVEL_REL_TOL: Float64 = 1e-2


# =============================================================================
# Main
# =============================================================================


fn compare_step(
    test_name: String,
    qpos_init: InlineArray[Float64, NQ],
    qvel_init: InlineArray[Float64, NV],
    actions: InlineArray[Float64, ACTION_DIM],
    num_steps: Int,
    ctx: DeviceContext,
    model_buf: DeviceBuffer[DTYPE],
    mut state_host: HostBuffer[DTYPE],
    mut state_buf: DeviceBuffer[DTYPE],
    mut workspace_buf: DeviceBuffer[DTYPE],
    mut ws_host: HostBuffer[DTYPE],
) raises -> Bool:
    """Run num_steps physics steps on CPU and GPU, compare final qpos/qvel."""
    print("--- Test:", test_name, "(", num_steps, "steps) ---")

    # === CPU pipeline ===
    var model_cpu = Model[
        DTYPE, NQ, NV, NBODY, NJOINT, MAX_CONTACTS, NGEOM, 0, ConeType.ELLIPTIC
    ](
        gravity_z=Scalar[DTYPE](-9.81),
        timestep=Scalar[DTYPE](0.01),
    )
    HalfCheetahBodies.setup_model(model_cpu)
    HalfCheetahJoints.setup_model(model_cpu)
    HalfCheetahGeoms.setup_model(model_cpu)

    # Set solver params
    comptime P = HalfCheetahParams[DTYPE]
    model_cpu.solref_contact[0] = P.SOLREF_CONTACT_0
    model_cpu.solref_contact[1] = P.SOLREF_CONTACT_1
    model_cpu.solimp_contact[0] = P.SOLIMP_CONTACT_0
    model_cpu.solimp_contact[1] = P.SOLIMP_CONTACT_1
    model_cpu.solimp_contact[2] = P.SOLIMP_CONTACT_2
    model_cpu.solref_limit[0] = P.SOLREF_LIMIT_0
    model_cpu.solref_limit[1] = P.SOLREF_LIMIT_1
    model_cpu.solimp_limit[0] = P.SOLIMP_LIMIT_0
    model_cpu.solimp_limit[1] = P.SOLIMP_LIMIT_1
    model_cpu.solimp_limit[2] = P.SOLIMP_LIMIT_2

    var data_cpu = Data[DTYPE, NQ, NV, NBODY, NJOINT, MAX_CONTACTS]()

    # Compute body_invweight0 at reference pose
    forward_kinematics(model_cpu, data_cpu)
    compute_body_invweight0[DTYPE, NQ, NV, NBODY, NJOINT, MAX_CONTACTS, NGEOM](
        model_cpu, data_cpu
    )

    # Set initial state
    for i in range(NQ):
        data_cpu.qpos[i] = Scalar[DTYPE](qpos_init[i])
    for i in range(NV):
        data_cpu.qvel[i] = Scalar[DTYPE](qvel_init[i])

    var action_list = List[Float64]()
    for i in range(ACTION_DIM):
        action_list.append(actions[i])

    # Run CPU steps
    for _ in range(num_steps):
        for i in range(NV):
            data_cpu.qfrc[i] = Scalar[DTYPE](0)
        HalfCheetahActuators.apply_actions[
            DTYPE, NQ, NV, NBODY, NJOINT, MAX_CONTACTS
        ](data_cpu, action_list)
        EulerIntegrator[SOLVER=PrimalNewtonSolver].step[NGEOM=NGEOM](
            model_cpu, data_cpu
        )

    # === GPU pipeline ===
    # Create model for GPU (need to rebuild with same params)
    var model_gpu = Model[
        DTYPE, NQ, NV, NBODY, NJOINT, MAX_CONTACTS, NGEOM, 0, ConeType.ELLIPTIC
    ](
        gravity_z=Scalar[DTYPE](-9.81),
        timestep=Scalar[DTYPE](0.01),
    )
    HalfCheetahBodies.setup_model(model_gpu)
    HalfCheetahJoints.setup_model(model_gpu)
    HalfCheetahGeoms.setup_model(model_gpu)
    model_gpu.solref_contact[0] = P.SOLREF_CONTACT_0
    model_gpu.solref_contact[1] = P.SOLREF_CONTACT_1
    model_gpu.solimp_contact[0] = P.SOLIMP_CONTACT_0
    model_gpu.solimp_contact[1] = P.SOLIMP_CONTACT_1
    model_gpu.solimp_contact[2] = P.SOLIMP_CONTACT_2
    model_gpu.solref_limit[0] = P.SOLREF_LIMIT_0
    model_gpu.solref_limit[1] = P.SOLREF_LIMIT_1
    model_gpu.solimp_limit[0] = P.SOLIMP_LIMIT_0
    model_gpu.solimp_limit[1] = P.SOLIMP_LIMIT_1
    model_gpu.solimp_limit[2] = P.SOLIMP_LIMIT_2

    # Compute invweight0 at ref pose
    var data_ref = Data[DTYPE, NQ, NV, NBODY, NJOINT, MAX_CONTACTS]()
    forward_kinematics(model_gpu, data_ref)
    compute_body_invweight0[DTYPE, NQ, NV, NBODY, NJOINT, MAX_CONTACTS, NGEOM](
        model_gpu, data_ref
    )

    var model_host = create_model_buffer[DTYPE, NBODY, NJOINT](ctx)
    copy_model_to_buffer[DTYPE, NQ, NV, NBODY, NJOINT, MAX_CONTACTS](
        model_gpu, model_host
    )
    var model_buf_local = ctx.enqueue_create_buffer[DTYPE](MODEL_SIZE)
    ctx.enqueue_copy(model_buf_local, model_host.unsafe_ptr())

    # For each GPU step, we need to set qfrc and then step
    # Set initial state in state buffer
    for i in range(BATCH * STATE_SIZE):
        state_host[i] = Scalar[DTYPE](0)
    for i in range(NQ):
        state_host[qpos_offset[NQ, NV]() + i] = Scalar[DTYPE](qpos_init[i])
    for i in range(NV):
        state_host[qvel_offset[NQ, NV]() + i] = Scalar[DTYPE](qvel_init[i])

    # Apply actions to get qfrc (use a temp data)
    var data_temp = Data[DTYPE, NQ, NV, NBODY, NJOINT, MAX_CONTACTS]()
    HalfCheetahActuators.apply_actions[
        DTYPE, NQ, NV, NBODY, NJOINT, MAX_CONTACTS
    ](data_temp, action_list)
    for i in range(NV):
        state_host[qfrc_offset[NQ, NV]() + i] = data_temp.qfrc[i]

    ctx.enqueue_copy(state_buf, state_host.unsafe_ptr())

    # Zero workspace
    for i in range(BATCH * WS_SIZE):
        ws_host[i] = Scalar[DTYPE](0)
    ctx.enqueue_copy(workspace_buf, ws_host.unsafe_ptr())
    ctx.synchronize()

    # Run GPU steps
    for step in range(num_steps):
        if step > 0:
            # For multi-step: read back state, re-apply actions, re-upload
            ctx.enqueue_copy(state_host.unsafe_ptr(), state_buf)
            ctx.synchronize()
            # Re-apply actions (qfrc) for next step
            for i in range(NV):
                state_host[qfrc_offset[NQ, NV]() + i] = data_temp.qfrc[i]
            ctx.enqueue_copy(state_buf, state_host.unsafe_ptr())
            # Re-zero workspace
            for i in range(BATCH * WS_SIZE):
                ws_host[i] = Scalar[DTYPE](0)
            ctx.enqueue_copy(workspace_buf, ws_host.unsafe_ptr())
            ctx.synchronize()

        EulerIntegrator[SOLVER=PrimalNewtonSolver].step_gpu[
            DTYPE, NQ, NV, NBODY, NJOINT, MAX_CONTACTS, BATCH,
            NGEOM=0, CONE_TYPE=ConeType.ELLIPTIC,
        ](
            ctx, state_buf, model_buf_local, workspace_buf,
            dt=Scalar[DTYPE](0.01),
            gravity_z=Scalar[DTYPE](-9.81),
            ground_z=Scalar[DTYPE](0.0),
        )
        ctx.synchronize()

    # Read back GPU results
    ctx.enqueue_copy(state_host.unsafe_ptr(), state_buf)
    ctx.synchronize()

    # === Compare qpos ===
    var qpos_pass = True
    var qpos_max_abs: Float64 = 0.0
    var qpos_max_rel: Float64 = 0.0
    var qpos_fails = 0

    for i in range(NQ):
        var cpu_val = Float64(data_cpu.qpos[i])
        var gpu_val = Float64(state_host[qpos_offset[NQ, NV]() + i])
        var abs_err = abs(cpu_val - gpu_val)
        var ref_mag = abs(cpu_val)
        var rel_err: Float64 = 0.0
        if ref_mag > 1e-10:
            rel_err = abs_err / ref_mag

        if abs_err > qpos_max_abs:
            qpos_max_abs = abs_err
        if rel_err > qpos_max_rel:
            qpos_max_rel = rel_err

        var ok = abs_err < QPOS_ABS_TOL or rel_err < QPOS_REL_TOL
        if not ok:
            if qpos_fails < 5:
                print(
                    "  FAIL qpos[", i, "]",
                    " cpu=", cpu_val,
                    " gpu=", gpu_val,
                    " abs=", abs_err,
                    " rel=", rel_err,
                )
            qpos_fails += 1
            qpos_pass = False

    # === Compare qvel ===
    var qvel_pass = True
    var qvel_max_abs: Float64 = 0.0
    var qvel_max_rel: Float64 = 0.0
    var qvel_fails = 0

    for i in range(NV):
        var cpu_val = Float64(data_cpu.qvel[i])
        var gpu_val = Float64(state_host[qvel_offset[NQ, NV]() + i])
        var abs_err = abs(cpu_val - gpu_val)
        var ref_mag = abs(cpu_val)
        var rel_err: Float64 = 0.0
        if ref_mag > 1e-10:
            rel_err = abs_err / ref_mag

        if abs_err > qvel_max_abs:
            qvel_max_abs = abs_err
        if rel_err > qvel_max_rel:
            qvel_max_rel = rel_err

        var ok = abs_err < QVEL_ABS_TOL or rel_err < QVEL_REL_TOL
        if not ok:
            if qvel_fails < 5:
                print(
                    "  FAIL qvel[", i, "]",
                    " cpu=", cpu_val,
                    " gpu=", gpu_val,
                    " abs=", abs_err,
                    " rel=", rel_err,
                )
            qvel_fails += 1
            qvel_pass = False

    var all_pass = qpos_pass and qvel_pass
    if all_pass:
        print(
            "  ALL OK  qpos(abs=", qpos_max_abs, " rel=", qpos_max_rel,
            ") qvel(abs=", qvel_max_abs, " rel=", qvel_max_rel, ")",
        )
    else:
        print(
            "  FAILED  qpos:", qpos_fails, "fails (abs=", qpos_max_abs,
            " rel=", qpos_max_rel, ")",
            " qvel:", qvel_fails, "fails (abs=", qvel_max_abs,
            " rel=", qvel_max_rel, ")",
        )

    # Print values
    print("  CPU qpos:", end="")
    for i in range(NQ):
        print(" ", Float64(data_cpu.qpos[i]), end="")
    print()
    print("  GPU qpos:", end="")
    for i in range(NQ):
        print(" ", Float64(state_host[qpos_offset[NQ, NV]() + i]), end="")
    print()
    print("  CPU qvel:", end="")
    for i in range(NV):
        print(" ", Float64(data_cpu.qvel[i]), end="")
    print()
    print("  GPU qvel:", end="")
    for i in range(NV):
        print(" ", Float64(state_host[qvel_offset[NQ, NV]() + i]), end="")
    print()
    print("  CPU contacts:", Int(data_cpu.num_contacts))

    return all_pass


fn main() raises:
    print("=" * 60)
    print("Full Step (no contacts): CPU vs GPU")
    print("=" * 60)
    print("Model: HalfCheetah (NQ=9, NV=9)")
    print("Integrator: Euler + PrimalNewtonSolver")
    print("Cone: elliptic")
    print("Precision: float32")
    print("Tolerances: qpos abs=", QPOS_ABS_TOL, " rel=", QPOS_REL_TOL)
    print("            qvel abs=", QVEL_ABS_TOL, " rel=", QVEL_REL_TOL)
    print()

    # Initialize GPU
    var ctx = DeviceContext()
    print("GPU device initialized")

    # Pre-allocate GPU buffers (reused across tests)
    var state_host = create_state_buffer[
        DTYPE, NQ, NV, NBODY, MAX_CONTACTS, BATCH
    ](ctx)
    var state_buf = ctx.enqueue_create_buffer[DTYPE](BATCH * STATE_SIZE)
    var workspace_buf = ctx.enqueue_create_buffer[DTYPE](BATCH * WS_SIZE)
    var ws_host = ctx.enqueue_create_host_buffer[DTYPE](BATCH * WS_SIZE)
    var model_buf = ctx.enqueue_create_buffer[DTYPE](MODEL_SIZE)  # placeholder
    print("GPU buffers allocated")
    print()

    var num_pass = 0
    var num_fail = 0

    # --- Config 1: Free fall (no contacts) ---
    var qpos1 = InlineArray[Float64, NQ](fill=0.0)
    qpos1[1] = 1.5  # rootz high enough for free fall
    var qvel1 = InlineArray[Float64, NV](fill=0.0)
    var act1 = InlineArray[Float64, ACTION_DIM](fill=0.0)
    if compare_step(
        "Free fall (1 step)", qpos1, qvel1, act1, 1,
        ctx, model_buf, state_host, state_buf, workspace_buf, ws_host,
    ):
        num_pass += 1
    else:
        num_fail += 1
    print()

    # --- Config 2: Free fall with actions ---
    var act2 = InlineArray[Float64, ACTION_DIM](fill=0.0)
    act2[0] = 0.5  # bthigh
    act2[1] = -0.3  # bshin
    act2[2] = 0.2  # bfoot
    act2[3] = 0.5  # fthigh
    act2[4] = -0.3  # fshin
    act2[5] = 0.1  # ffoot
    if compare_step(
        "Free fall + actions (1 step)", qpos1, qvel1, act2, 1,
        ctx, model_buf, state_host, state_buf, workspace_buf, ws_host,
    ):
        num_pass += 1
    else:
        num_fail += 1
    print()

    # --- Config 3: Moving robot, high up ---
    var qpos3 = InlineArray[Float64, NQ](fill=0.0)
    qpos3[1] = 1.5  # high
    qpos3[2] = 0.1  # slight pitch
    qpos3[3] = -0.3  # bthigh bent
    qpos3[6] = 0.4  # fthigh bent
    var qvel3 = InlineArray[Float64, NV](fill=0.0)
    qvel3[0] = 2.0  # rootx vel
    qvel3[2] = 0.5  # rooty vel
    qvel3[3] = -1.0  # bthigh vel
    qvel3[6] = 1.2  # fthigh vel
    var act3 = InlineArray[Float64, ACTION_DIM](fill=0.0)
    act3[0] = 1.0
    act3[1] = -0.5
    act3[3] = 1.0
    act3[4] = -0.5
    if compare_step(
        "Moving + actions (1 step)", qpos3, qvel3, act3, 1,
        ctx, model_buf, state_host, state_buf, workspace_buf, ws_host,
    ):
        num_pass += 1
    else:
        num_fail += 1
    print()

    # --- Config 4: Free fall 10 steps ---
    if compare_step(
        "Free fall (10 steps)", qpos1, qvel1, act1, 10,
        ctx, model_buf, state_host, state_buf, workspace_buf, ws_host,
    ):
        num_pass += 1
    else:
        num_fail += 1
    print()

    # --- Config 5: Moving + actions 10 steps ---
    if compare_step(
        "Moving + actions (10 steps)", qpos3, qvel3, act3, 10,
        ctx, model_buf, state_host, state_buf, workspace_buf, ws_host,
    ):
        num_pass += 1
    else:
        num_fail += 1
    print()

    # --- Config 6: Extreme velocities ---
    var qpos6 = InlineArray[Float64, NQ](fill=0.0)
    qpos6[1] = 2.0  # very high
    qpos6[3] = -0.52
    qpos6[6] = -1.0
    var qvel6 = InlineArray[Float64, NV](fill=0.0)
    qvel6[0] = 5.0
    qvel6[1] = -2.0
    qvel6[2] = 3.0
    qvel6[3] = -5.0
    qvel6[4] = 5.0
    qvel6[5] = -3.0
    qvel6[6] = 5.0
    qvel6[7] = -5.0
    qvel6[8] = 3.0
    if compare_step(
        "Extreme velocities (1 step)", qpos6, qvel6, act1, 1,
        ctx, model_buf, state_host, state_buf, workspace_buf, ws_host,
    ):
        num_pass += 1
    else:
        num_fail += 1
    print()

    print("=" * 60)
    print(
        "Results:",
        num_pass,
        "passed,",
        num_fail,
        "failed out of",
        num_pass + num_fail,
    )
    if num_fail == 0:
        print("ALL TESTS PASSED")
    else:
        print("SOME TESTS FAILED")
    print("=" * 60)
