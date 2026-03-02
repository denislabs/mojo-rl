"""Test Full Physics Step (no contacts): CPU vs GPU for Swimmer.

Compares qpos/qvel after running N RK4 physics steps on CPU vs GPU for the
Swimmer model. Uses various configurations to test the full pipeline.

The Swimmer has contype=0 on all geoms so there are no contacts — this is
a pure dynamics test (FK + mass matrix + bias forces + actuators + RK4 integration).

CPU uses RK4Integrator[NewtonSolver].step() (float32).
GPU uses RK4Integrator[NewtonSolver].step_gpu() (float32).

Run with:
    cd mojo-rl && pixi run -e apple mojo run physics3d/tests/test_swimmer_full_step_cpu_vs_gpu.mojo
"""

from testing import assert_true
from math import abs
from collections import InlineArray
from gpu.host import DeviceContext, DeviceBuffer, HostBuffer
from layout import Layout, LayoutTensor

from physics3d.types import Model, Data, ConeType
from physics3d.integrator.rk4_integrator import RK4Integrator
from physics3d.solver import NewtonSolver
from physics3d.kinematics.forward_kinematics import forward_kinematics
from physics3d.gpu.constants import (
    state_size,
    model_size_with_invweight,
    qpos_offset,
    qvel_offset,
    qfrc_offset,
    integrator_workspace_size,
    rk4_extra_workspace_size,
)
from physics3d.gpu.buffer_utils import create_state_buffer
from envs.swimmer.swimmer_xml import SwimmerModel


# =============================================================================
# Constants
# =============================================================================

comptime DTYPE = DType.float32
comptime NQ = SwimmerModel.NQ          # 5
comptime NV = SwimmerModel.NV          # 5
comptime NBODY = SwimmerModel.NBODY    # 4
comptime NJOINT = SwimmerModel.NJOINT  # 5
comptime NGEOM = SwimmerModel.NGEOM
comptime MAX_CONTACTS = SwimmerModel.MAX_CONTACTS  # 5
comptime ACTION_DIM = SwimmerModel.ACTION_DIM      # 2
comptime NSITE = SwimmerModel.NSITE   # 0
comptime BATCH = 1

comptime STATE_SIZE = state_size[NQ, NV, NBODY, MAX_CONTACTS, NSITE]()
comptime MODEL_SIZE = model_size_with_invweight[NBODY, NJOINT, NV, NGEOM]()
comptime SOLVER_WS = NewtonSolver.solver_workspace_size[NV, MAX_CONTACTS]()
comptime WS_SIZE = integrator_workspace_size[NV, NBODY]() + NV * NV + SOLVER_WS + rk4_extra_workspace_size[NQ, NV]()

# Tolerances (float32, no contacts → tight match expected)
comptime QPOS_ABS_TOL: Float64 = 3e-2
comptime QPOS_REL_TOL: Float64 = 3e-2
comptime QVEL_ABS_TOL: Float64 = 5e-1
comptime QVEL_REL_TOL: Float64 = 5e-1


# =============================================================================
# Comparison helper
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
) raises:
    """Run num_steps RK4 physics steps on CPU and GPU, compare final qpos/qvel."""
    print("--- Test:", test_name, "(", num_steps, "steps) ---")

    # === CPU pipeline ===
    var model_cpu = Model[
        DTYPE, NQ, NV, NBODY, NJOINT, MAX_CONTACTS, NGEOM,
        SwimmerModel.MAX_EQUALITY, SwimmerModel.CONE_TYPE,
        SwimmerModel.MAX_TENDON, SwimmerModel.NSITE,
    ]()
    var data_cpu = Data[DTYPE, NQ, NV, NBODY, NJOINT, MAX_CONTACTS, SwimmerModel.NSITE]()
    SwimmerModel.setup_model_and_data(model_cpu, data_cpu)

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
        SwimmerModel.apply_actions(data_cpu, action_list)
        RK4Integrator[SOLVER=NewtonSolver].step[NGEOM=NGEOM](model_cpu, data_cpu)

    # === GPU pipeline ===
    var model_buf_local = ctx.enqueue_create_buffer[DTYPE](MODEL_SIZE)
    SwimmerModel.init_model_gpu(ctx, model_buf_local)

    for i in range(BATCH * STATE_SIZE):
        state_host[i] = Scalar[DTYPE](0)
    for i in range(NQ):
        state_host[qpos_offset[NQ, NV]() + i] = Scalar[DTYPE](qpos_init[i])
    for i in range(NV):
        state_host[qvel_offset[NQ, NV]() + i] = Scalar[DTYPE](qvel_init[i])

    # Apply actions to get qfrc
    var data_temp = Data[DTYPE, NQ, NV, NBODY, NJOINT, MAX_CONTACTS, SwimmerModel.NSITE]()
    SwimmerModel.apply_actions(data_temp, action_list)
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

        RK4Integrator[SOLVER=NewtonSolver].step_gpu[
            DTYPE, NQ, NV, NBODY, NJOINT, MAX_CONTACTS, BATCH,
            NGEOM=0, CONE_TYPE=SwimmerModel.CONE_TYPE,
        ](ctx, state_buf, model_buf_local, workspace_buf)
        ctx.synchronize()

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
                print("  FAIL qpos[", i, "] cpu=", cpu_val, " gpu=", gpu_val, " abs=", abs_err, " rel=", rel_err)
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
                print("  FAIL qvel[", i, "] cpu=", cpu_val, " gpu=", gpu_val, " abs=", abs_err, " rel=", rel_err)
            qvel_fails += 1
            qvel_pass = False

    var all_pass = qpos_pass and qvel_pass
    if all_pass:
        print("  ALL OK  qpos(abs=", qpos_max_abs, " rel=", qpos_max_rel, ") qvel(abs=", qvel_max_abs, " rel=", qvel_max_rel, ")")
    else:
        print("  FAILED  qpos:", qpos_fails, "fails (abs=", qpos_max_abs, " rel=", qpos_max_rel, ")",
              " qvel:", qvel_fails, "fails (abs=", qvel_max_abs, " rel=", qvel_max_rel, ")")

    print("  CPU qpos:", end="")
    for i in range(NQ):
        print(" ", Float64(data_cpu.qpos[i]), end="")
    print()
    print("  GPU qpos:", end="")
    for i in range(NQ):
        print(" ", Float64(state_host[qpos_offset[NQ, NV]() + i]), end="")
    print()
    print("  CPU contacts:", Int(data_cpu.num_contacts))

    assert_true(all_pass, "CPU vs GPU mismatch for: " + test_name)


# =============================================================================
# Test cases (same configs as test_swimmer_full_step_vs_mujoco.mojo)
# =============================================================================


fn test_zero_state_1_step() raises:
    var ctx = DeviceContext()
    var state_host = create_state_buffer[DTYPE, NQ, NV, NBODY, MAX_CONTACTS, NSITE, BATCH](ctx)
    var state_buf = ctx.enqueue_create_buffer[DTYPE](BATCH * STATE_SIZE)
    var workspace_buf = ctx.enqueue_create_buffer[DTYPE](BATCH * WS_SIZE)
    var ws_host = ctx.enqueue_create_host_buffer[DTYPE](BATCH * WS_SIZE)
    var model_buf = ctx.enqueue_create_buffer[DTYPE](MODEL_SIZE)

    var qpos1 = InlineArray[Float64, NQ](fill=0.0)
    var qvel1 = InlineArray[Float64, NV](fill=0.0)
    var act1 = InlineArray[Float64, ACTION_DIM](fill=0.0)
    compare_step("Zero state (1 step)", qpos1, qvel1, act1, 1, ctx, model_buf, state_host, state_buf, workspace_buf, ws_host)
    print()


fn test_bent_joints_1_step() raises:
    var ctx = DeviceContext()
    var state_host = create_state_buffer[DTYPE, NQ, NV, NBODY, MAX_CONTACTS, NSITE, BATCH](ctx)
    var state_buf = ctx.enqueue_create_buffer[DTYPE](BATCH * STATE_SIZE)
    var workspace_buf = ctx.enqueue_create_buffer[DTYPE](BATCH * WS_SIZE)
    var ws_host = ctx.enqueue_create_host_buffer[DTYPE](BATCH * WS_SIZE)
    var model_buf = ctx.enqueue_create_buffer[DTYPE](MODEL_SIZE)

    var qpos2 = InlineArray[Float64, NQ](fill=0.0)
    qpos2[3] = 0.5   # motor1_rot
    qpos2[4] = -0.3  # motor2_rot
    var qvel2 = InlineArray[Float64, NV](fill=0.0)
    var act2 = InlineArray[Float64, ACTION_DIM](fill=0.0)
    compare_step("Bent joints (motor1=0.5, motor2=-0.3, 1 step)", qpos2, qvel2, act2, 1, ctx, model_buf, state_host, state_buf, workspace_buf, ws_host)
    print()


fn test_motor_actions_1_step() raises:
    var ctx = DeviceContext()
    var state_host = create_state_buffer[DTYPE, NQ, NV, NBODY, MAX_CONTACTS, NSITE, BATCH](ctx)
    var state_buf = ctx.enqueue_create_buffer[DTYPE](BATCH * STATE_SIZE)
    var workspace_buf = ctx.enqueue_create_buffer[DTYPE](BATCH * WS_SIZE)
    var ws_host = ctx.enqueue_create_host_buffer[DTYPE](BATCH * WS_SIZE)
    var model_buf = ctx.enqueue_create_buffer[DTYPE](MODEL_SIZE)

    var qpos3 = InlineArray[Float64, NQ](fill=0.0)
    qpos3[3] = 0.3   # motor1_rot
    qpos3[4] = -0.2  # motor2_rot
    var qvel3 = InlineArray[Float64, NV](fill=0.0)
    var act3 = InlineArray[Float64, ACTION_DIM](fill=0.0)
    act3[0] = 0.8    # motor1 (gear=150)
    act3[1] = -0.6   # motor2
    compare_step("Motor actions (ctrl=[0.8,-0.6], 1 step)", qpos3, qvel3, act3, 1, ctx, model_buf, state_host, state_buf, workspace_buf, ws_host)
    print()


fn test_moving_with_actions_1_step() raises:
    var ctx = DeviceContext()
    var state_host = create_state_buffer[DTYPE, NQ, NV, NBODY, MAX_CONTACTS, NSITE, BATCH](ctx)
    var state_buf = ctx.enqueue_create_buffer[DTYPE](BATCH * STATE_SIZE)
    var workspace_buf = ctx.enqueue_create_buffer[DTYPE](BATCH * WS_SIZE)
    var ws_host = ctx.enqueue_create_host_buffer[DTYPE](BATCH * WS_SIZE)
    var model_buf = ctx.enqueue_create_buffer[DTYPE](MODEL_SIZE)

    # Swimmer moving (nonzero slider velocities) + joints bent + motor actions
    var qpos4 = InlineArray[Float64, NQ](fill=0.0)
    qpos4[0] = 1.0   # slider1 (x)
    qpos4[1] = 0.5   # slider2 (y)
    qpos4[2] = 0.8   # free_body_rot (yaw)
    qpos4[3] = 0.4   # motor1_rot
    qpos4[4] = -0.3  # motor2_rot
    var qvel4 = InlineArray[Float64, NV](fill=0.0)
    qvel4[0] = 1.5   # x velocity
    qvel4[1] = -0.5  # y velocity
    qvel4[2] = 0.3   # yaw rate
    qvel4[3] = 1.0   # motor1 rate
    qvel4[4] = -0.8  # motor2 rate
    var act4 = InlineArray[Float64, ACTION_DIM](fill=0.0)
    act4[0] = 0.5
    act4[1] = -0.4
    compare_step("Moving + actions (1 step)", qpos4, qvel4, act4, 1, ctx, model_buf, state_host, state_buf, workspace_buf, ws_host)
    print()


fn test_zero_state_10_steps() raises:
    var ctx = DeviceContext()
    var state_host = create_state_buffer[DTYPE, NQ, NV, NBODY, MAX_CONTACTS, NSITE, BATCH](ctx)
    var state_buf = ctx.enqueue_create_buffer[DTYPE](BATCH * STATE_SIZE)
    var workspace_buf = ctx.enqueue_create_buffer[DTYPE](BATCH * WS_SIZE)
    var ws_host = ctx.enqueue_create_host_buffer[DTYPE](BATCH * WS_SIZE)
    var model_buf = ctx.enqueue_create_buffer[DTYPE](MODEL_SIZE)

    var qpos1 = InlineArray[Float64, NQ](fill=0.0)
    var qvel1 = InlineArray[Float64, NV](fill=0.0)
    var act5 = InlineArray[Float64, ACTION_DIM](fill=0.0)
    act5[0] = 0.6
    act5[1] = -0.4
    compare_step("Motor actions (10 steps)", qpos1, qvel1, act5, 10, ctx, model_buf, state_host, state_buf, workspace_buf, ws_host)
    print()


fn main() raises:
    print("=" * 60)
    print("Full Step Validation: CPU vs GPU — Swimmer (RK4, no contacts)")
    print("=" * 60)
    print("Model: Swimmer (NQ=5, NV=5, contype=0)")
    print("Integrator: RK4 + Newton solver")
    print("Precision: float32")
    print("Tolerances: qpos abs=", QPOS_ABS_TOL, " qvel abs=", QVEL_ABS_TOL)
    print()

    test_zero_state_1_step()
    test_bent_joints_1_step()
    test_motor_actions_1_step()
    test_moving_with_actions_1_step()
    test_zero_state_10_steps()
    print("All Swimmer full step CPU vs GPU tests passed.")
