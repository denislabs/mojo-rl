"""Test ImplicitFast + PGS: CPU vs GPU.

Compares qpos/qvel after running physics steps using ImplicitFastIntegrator[PGSSolver]
(= DefaultIntegrator) on CPU vs GPU for the HalfCheetah model.
This is the integrator combo used by Hopper.

Tests both no-contact (free flight) and with-contact (ground contact) configurations.

Run with:
    cd mojo-rl && pixi run -e apple mojo run physics3d/tests/test_implicit_fast_pgs_cpu_vs_gpu.mojo
"""

from math import abs
from collections import InlineArray
from gpu.host import DeviceContext, DeviceBuffer, HostBuffer
from layout import Layout, LayoutTensor

from physics3d.types import Model, Data, _max_one, ConeType
from physics3d.integrator.implicit_fast_integrator import ImplicitFastIntegrator
from physics3d.solver.pgs_solver import PGSSolver
from physics3d.kinematics.forward_kinematics import forward_kinematics
from physics3d.dynamics.mass_matrix import compute_body_invweight0
from physics3d.gpu.constants import (
    state_size,
    model_size_with_invweight,
    qpos_offset,
    qvel_offset,
    qfrc_offset,
    integrator_workspace_size,
)
from physics3d.gpu.buffer_utils import (
    create_state_buffer,
    copy_model_to_buffer,
    copy_geoms_to_buffer,
    copy_invweight0_to_buffer,
)
from envs.half_cheetah.half_cheetah_def import (
    HalfCheetahModel,
    HalfCheetahBodies,
    HalfCheetahJoints,
    HalfCheetahGeoms,
    HalfCheetahActuators,
    HalfCheetahParams,
    HalfCheetahDefaults,
)


# =============================================================================
# Constants
# =============================================================================

comptime DTYPE = DType.float32
comptime NQ = HalfCheetahModel.NQ
comptime NV = HalfCheetahModel.NV
comptime NBODY = HalfCheetahModel.NBODY
comptime NJOINT = HalfCheetahModel.NJOINT
comptime NGEOM = HalfCheetahModel.NGEOM
comptime MAX_CONTACTS = HalfCheetahParams[DTYPE].MAX_CONTACTS
comptime ACTION_DIM = HalfCheetahParams[DTYPE].ACTION_DIM
comptime BATCH = 1

comptime STATE_SIZE = state_size[NQ, NV, NBODY, MAX_CONTACTS]()
comptime MODEL_SIZE = model_size_with_invweight[NBODY, NJOINT, NV, NGEOM]()
comptime WS_SIZE = integrator_workspace_size[NV, NBODY]() + NV * NV + PGSSolver.solver_workspace_size[NV, MAX_CONTACTS]()

# Tolerances — PGS is iterative with limited iterations, so CPU vs GPU may differ more
comptime QPOS_ABS_TOL: Float64 = 5e-2
comptime QPOS_REL_TOL: Float64 = 3e-1
comptime QVEL_ABS_TOL: Float64 = 1.0
comptime QVEL_REL_TOL: Float64 = 5e-1

comptime Integrator = ImplicitFastIntegrator[SOLVER=PGSSolver]


# =============================================================================
# Compare helper
# =============================================================================


fn compare_step(
    test_name: String,
    qpos_init: InlineArray[Float64, NQ],
    qvel_init: InlineArray[Float64, NV],
    actions: InlineArray[Float64, ACTION_DIM],
    num_steps: Int,
    ctx: DeviceContext,
    mut model_buf: DeviceBuffer[DTYPE],
    mut state_host: HostBuffer[DTYPE],
    mut state_buf: DeviceBuffer[DTYPE],
    mut workspace_buf: DeviceBuffer[DTYPE],
    mut ws_host: HostBuffer[DTYPE],
) raises -> Bool:
    print("--- Test:", test_name, "(", num_steps, "steps) ---")

    # === CPU pipeline ===
    var model_cpu = Model[
        DTYPE, NQ, NV, NBODY, NJOINT, MAX_CONTACTS, NGEOM, 0, ConeType.ELLIPTIC
    ](
        gravity_z=Scalar[DTYPE](-9.81),
        timestep=Scalar[DTYPE](0.01),
    )
    HalfCheetahModel.setup_solver_params[Defaults=HalfCheetahDefaults](model_cpu)

    HalfCheetahBodies.setup_model(model_cpu)

    HalfCheetahJoints.setup_model[Defaults=HalfCheetahDefaults](model_cpu)

    HalfCheetahGeoms.setup_model[Defaults=HalfCheetahDefaults](model_cpu)

    var data_cpu = Data[DTYPE, NQ, NV, NBODY, NJOINT, MAX_CONTACTS]()

    forward_kinematics(model_cpu, data_cpu)
    compute_body_invweight0[DTYPE, NQ, NV, NBODY, NJOINT, MAX_CONTACTS, NGEOM](
        model_cpu, data_cpu
    )

    for i in range(NQ):
        data_cpu.qpos[i] = Scalar[DTYPE](qpos_init[i])
    for i in range(NV):
        data_cpu.qvel[i] = Scalar[DTYPE](qvel_init[i])

    var action_list = List[Float64]()
    for i in range(ACTION_DIM):
        action_list.append(actions[i])

    for _ in range(num_steps):
        for i in range(NV):
            data_cpu.qfrc[i] = Scalar[DTYPE](0)
        HalfCheetahActuators.apply_actions[
            DTYPE, NQ, NV, NBODY, NJOINT, MAX_CONTACTS
        ](data_cpu, action_list)
        Integrator.step[NGEOM=NGEOM](model_cpu, data_cpu)

    # === GPU pipeline ===
    for i in range(BATCH * STATE_SIZE):
        state_host[i] = Scalar[DTYPE](0)
    for i in range(NQ):
        state_host[qpos_offset[NQ, NV]() + i] = Scalar[DTYPE](qpos_init[i])
    for i in range(NV):
        state_host[qvel_offset[NQ, NV]() + i] = Scalar[DTYPE](qvel_init[i])

    var data_temp = Data[DTYPE, NQ, NV, NBODY, NJOINT, MAX_CONTACTS]()
    HalfCheetahActuators.apply_actions[
        DTYPE, NQ, NV, NBODY, NJOINT, MAX_CONTACTS
    ](data_temp, action_list)
    for i in range(NV):
        state_host[qfrc_offset[NQ, NV]() + i] = data_temp.qfrc[i]

    ctx.enqueue_copy(state_buf, state_host.unsafe_ptr())
    for i in range(BATCH * WS_SIZE):
        ws_host[i] = Scalar[DTYPE](0)
    ctx.enqueue_copy(workspace_buf, ws_host.unsafe_ptr())
    ctx.synchronize()

    for step in range(num_steps):
        if step > 0:
            ctx.enqueue_copy(state_host.unsafe_ptr(), state_buf)
            ctx.synchronize()
            for i in range(NV):
                state_host[qfrc_offset[NQ, NV]() + i] = data_temp.qfrc[i]
            ctx.enqueue_copy(state_buf, state_host.unsafe_ptr())
            for i in range(BATCH * WS_SIZE):
                ws_host[i] = Scalar[DTYPE](0)
            ctx.enqueue_copy(workspace_buf, ws_host.unsafe_ptr())
            ctx.synchronize()

        Integrator.step_gpu[
            DTYPE, NQ, NV, NBODY, NJOINT, MAX_CONTACTS, BATCH,
            NGEOM=NGEOM, CONE_TYPE=ConeType.ELLIPTIC,
        ](
            ctx, state_buf, model_buf, workspace_buf,
            dt=Scalar[DTYPE](0.01),
            gravity_z=Scalar[DTYPE](-9.81),
            ground_z=Scalar[DTYPE](0.0),
        )
        ctx.synchronize()

    ctx.enqueue_copy(state_host.unsafe_ptr(), state_buf)
    ctx.synchronize()

    # === Compare ===
    var qpos_pass = True
    var qpos_max_abs: Float64 = 0.0
    var qvel_pass = True
    var qvel_max_abs: Float64 = 0.0
    var qpos_fails = 0
    var qvel_fails = 0

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
        var ok = abs_err < QPOS_ABS_TOL or rel_err < QPOS_REL_TOL
        if not ok:
            if qpos_fails < 5:
                print(
                    "  FAIL qpos[", i, "]",
                    " cpu=", cpu_val, " gpu=", gpu_val,
                    " abs=", abs_err, " rel=", rel_err,
                )
            qpos_fails += 1
            qpos_pass = False

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
        var ok = abs_err < QVEL_ABS_TOL or rel_err < QVEL_REL_TOL
        if not ok:
            if qvel_fails < 5:
                print(
                    "  FAIL qvel[", i, "]",
                    " cpu=", cpu_val, " gpu=", gpu_val,
                    " abs=", abs_err, " rel=", rel_err,
                )
            qvel_fails += 1
            qvel_pass = False

    var all_pass = qpos_pass and qvel_pass
    if all_pass:
        print(
            "  ALL OK  qpos_max_abs=", qpos_max_abs,
            " qvel_max_abs=", qvel_max_abs,
        )
    else:
        print(
            "  FAILED  qpos:", qpos_fails, "fails (max_abs=", qpos_max_abs, ")",
            " qvel:", qvel_fails, "fails (max_abs=", qvel_max_abs, ")",
        )
    print("  CPU contacts:", Int(data_cpu.num_contacts))

    return all_pass


fn main() raises:
    print("=" * 60)
    print("ImplicitFast + PGS: CPU vs GPU")
    print("=" * 60)
    print("Model: HalfCheetah (NQ=9, NV=9, NGEOM=", NGEOM, ")")
    print("Integrator: ImplicitFast + PGSSolver (elliptic)")
    print("Precision: float32")
    print()

    var ctx = DeviceContext()

    # Create GPU model
    var model_gpu = Model[
        DTYPE, NQ, NV, NBODY, NJOINT, MAX_CONTACTS, NGEOM, 0, ConeType.ELLIPTIC
    ](
        gravity_z=Scalar[DTYPE](-9.81),
        timestep=Scalar[DTYPE](0.01),
    )
    HalfCheetahModel.setup_solver_params[Defaults=HalfCheetahDefaults](model_gpu)

    HalfCheetahBodies.setup_model(model_gpu)

    HalfCheetahJoints.setup_model[Defaults=HalfCheetahDefaults](model_gpu)

    HalfCheetahGeoms.setup_model[Defaults=HalfCheetahDefaults](model_gpu)

    var data_ref = Data[DTYPE, NQ, NV, NBODY, NJOINT, MAX_CONTACTS]()
    forward_kinematics(model_gpu, data_ref)
    compute_body_invweight0[DTYPE, NQ, NV, NBODY, NJOINT, MAX_CONTACTS, NGEOM](
        model_gpu, data_ref
    )

    var model_host = ctx.enqueue_create_host_buffer[DTYPE](MODEL_SIZE)
    for i in range(MODEL_SIZE):
        model_host[i] = Scalar[DTYPE](0)
    copy_model_to_buffer(model_gpu, model_host)
    copy_geoms_to_buffer(model_gpu, model_host)
    copy_invweight0_to_buffer(model_gpu, model_host)
    var model_buf = ctx.enqueue_create_buffer[DTYPE](MODEL_SIZE)
    ctx.enqueue_copy(model_buf, model_host.unsafe_ptr())
    ctx.synchronize()

    var state_host = create_state_buffer[DTYPE, NQ, NV, NBODY, MAX_CONTACTS, BATCH](ctx)
    var state_buf = ctx.enqueue_create_buffer[DTYPE](BATCH * STATE_SIZE)
    var workspace_buf = ctx.enqueue_create_buffer[DTYPE](BATCH * WS_SIZE)
    var ws_host = ctx.enqueue_create_host_buffer[DTYPE](BATCH * WS_SIZE)
    print("GPU ready")
    print()

    var num_pass = 0
    var num_fail = 0

    var zero_act = InlineArray[Float64, ACTION_DIM](fill=0.0)
    var zero_vel = InlineArray[Float64, NV](fill=0.0)
    var actions = InlineArray[Float64, ACTION_DIM](fill=0.0)
    actions[0] = 0.5; actions[1] = -0.3; actions[2] = 0.2
    actions[3] = 0.5; actions[4] = -0.3; actions[5] = 0.1

    # --- No contact configs ---
    var qpos_high = InlineArray[Float64, NQ](fill=0.0)
    qpos_high[1] = 1.5

    if compare_step("Free fall (1 step)", qpos_high, zero_vel, zero_act, 1,
        ctx, model_buf, state_host, state_buf, workspace_buf, ws_host):
        num_pass += 1
    else:
        num_fail += 1
    print()

    if compare_step("Free fall + actions (1 step)", qpos_high, zero_vel, actions, 1,
        ctx, model_buf, state_host, state_buf, workspace_buf, ws_host):
        num_pass += 1
    else:
        num_fail += 1
    print()

    if compare_step("Free fall (10 steps)", qpos_high, zero_vel, zero_act, 10,
        ctx, model_buf, state_host, state_buf, workspace_buf, ws_host):
        num_pass += 1
    else:
        num_fail += 1
    print()

    # --- Contact configs ---
    var qpos_low = InlineArray[Float64, NQ](fill=0.0)
    qpos_low[1] = -0.2

    if compare_step("Ground contact (1 step)", qpos_low, zero_vel, zero_act, 1,
        ctx, model_buf, state_host, state_buf, workspace_buf, ws_host):
        num_pass += 1
    else:
        num_fail += 1
    print()

    if compare_step("Ground contact + actions (1 step)", qpos_low, zero_vel, actions, 1,
        ctx, model_buf, state_host, state_buf, workspace_buf, ws_host):
        num_pass += 1
    else:
        num_fail += 1
    print()

    var qpos_deep = InlineArray[Float64, NQ](fill=0.0)
    qpos_deep[1] = -0.5
    if compare_step("Deep penetration (1 step)", qpos_deep, zero_vel, zero_act, 1,
        ctx, model_buf, state_host, state_buf, workspace_buf, ws_host):
        num_pass += 1
    else:
        num_fail += 1
    print()

    if compare_step("Ground contact (5 steps)", qpos_low, zero_vel, zero_act, 5,
        ctx, model_buf, state_host, state_buf, workspace_buf, ws_host):
        num_pass += 1
    else:
        num_fail += 1
    print()

    if compare_step("Ground contact + actions (5 steps)", qpos_low, zero_vel, actions, 5,
        ctx, model_buf, state_host, state_buf, workspace_buf, ws_host):
        num_pass += 1
    else:
        num_fail += 1
    print()

    print("=" * 60)
    print("Results:", num_pass, "passed,", num_fail, "failed out of", num_pass + num_fail)
    if num_fail == 0:
        print("ALL TESTS PASSED")
    else:
        print("SOME TESTS FAILED")
    print("=" * 60)
