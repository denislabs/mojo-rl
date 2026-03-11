"""Test Full Step (no contacts): CPU vs GPU.

Compares qpos/qvel after running N physics steps on CPU vs GPU for the
HalfCheetah model. Uses high rootz to avoid ground contacts so we test the
dynamics pipeline only (FK → M → bias → qacc → implicit damping → integrate).

The CPU uses EulerIntegrator[NewtonSolver].step() (float32).
The GPU uses EulerIntegrator[NewtonSolver].step_gpu() (float32).

Run with:
    cd mojo-rl && pixi run -e apple mojo run physics3d/tests/test_full_step_cpu_vs_gpu.mojo
"""

from std.testing import assert_true, TestSuite
from std.math import abs
from std.collections import InlineArray
from std.gpu.host import DeviceContext, DeviceBuffer, HostBuffer
from layout import Layout, LayoutTensor

from mojo_rl.physics3d.types import Model, Data, ConeType
from mojo_rl.physics3d.integrator.euler_integrator import EulerIntegrator
from mojo_rl.physics3d.solver import NewtonSolver
from mojo_rl.physics3d.gpu.constants import (
    state_size,
    model_size_with_invweight,
    qpos_offset,
    qvel_offset,
    qfrc_offset,
    integrator_workspace_size,
)
from mojo_rl.physics3d.gpu.buffer_utils import (
    create_state_buffer,
    copy_data_to_buffer,
)
from mojo_rl.envs.half_cheetah.half_cheetah_xml import HalfCheetahModel
from mojo_rl.envs.half_cheetah.half_cheetah_config import HalfCheetahConfig


# =============================================================================
# Constants
# =============================================================================

comptime DTYPE = DType.float32
comptime NQ = HalfCheetahModel.NQ  # 9
comptime NV = HalfCheetahModel.NV  # 9
comptime NBODY = HalfCheetahModel.NBODY
comptime NJOINT = HalfCheetahModel.NJOINT
comptime NGEOM = HalfCheetahModel.NGEOM
comptime MAX_CONTACTS = HalfCheetahConfig.MAX_CONTACTS
comptime ACTION_DIM = HalfCheetahConfig.ACTION_DIM
comptime BATCH = 1

comptime STATE_SIZE = state_size[NQ, NV, NBODY, MAX_CONTACTS]()
comptime MODEL_SIZE = model_size_with_invweight[NBODY, NJOINT, NV, NGEOM]()
comptime WS_SIZE = integrator_workspace_size[
    NV, NBODY
]() + NV * NV + NewtonSolver.solver_workspace_size[NV, MAX_CONTACTS]()

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
) raises:
    """Run num_steps physics steps on CPU and GPU, compare final qpos/qvel."""
    print("--- Test:", test_name, "(", num_steps, "steps) ---")

    # === CPU pipeline ===
    var model_cpu = Model[
        DTYPE,
        NQ,
        NV,
        NBODY,
        NJOINT,
        MAX_CONTACTS,
        NGEOM,
        HalfCheetahModel.MAX_EQUALITY,
        HalfCheetahModel.CONE_TYPE,
        HalfCheetahModel.MAX_TENDON,
        HalfCheetahModel.NSITE,
    ]()
    var data_cpu = Data[
        DTYPE, NQ, NV, NBODY, NJOINT, MAX_CONTACTS, HalfCheetahModel.NSITE
    ]()
    HalfCheetahModel.setup_model_and_data(model_cpu, data_cpu)

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
        HalfCheetahModel.apply_actions(data_cpu, action_list)
        EulerIntegrator[SOLVER=NewtonSolver].step[NGEOM=NGEOM](
            model_cpu, data_cpu
        )

    # === GPU pipeline ===
    var model_buf_local = ctx.enqueue_create_buffer[DTYPE](MODEL_SIZE)
    HalfCheetahModel.init_model_gpu(ctx, model_buf_local)

    # For each GPU step, we need to set qfrc and then step
    # Set initial state in state buffer
    for i in range(BATCH * STATE_SIZE):
        state_host[i] = Scalar[DTYPE](0)
    for i in range(NQ):
        state_host[qpos_offset[NQ, NV]() + i] = Scalar[DTYPE](qpos_init[i])
    for i in range(NV):
        state_host[qvel_offset[NQ, NV]() + i] = Scalar[DTYPE](qvel_init[i])

    # Apply actions to get qfrc (use a temp data)
    var data_temp = Data[
        DTYPE, NQ, NV, NBODY, NJOINT, MAX_CONTACTS, HalfCheetahModel.NSITE
    ]()
    HalfCheetahModel.apply_actions(data_temp, action_list)
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

        EulerIntegrator[SOLVER=NewtonSolver].step_gpu[
            DTYPE,
            NQ,
            NV,
            NBODY,
            NJOINT,
            MAX_CONTACTS,
            BATCH,
            NGEOM=0,
            CONE_TYPE=HalfCheetahModel.CONE_TYPE,
        ](
            ctx,
            state_buf,
            model_buf_local,
            workspace_buf,
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
                    "  FAIL qpos[",
                    i,
                    "]",
                    " cpu=",
                    cpu_val,
                    " gpu=",
                    gpu_val,
                    " abs=",
                    abs_err,
                    " rel=",
                    rel_err,
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
                    "  FAIL qvel[",
                    i,
                    "]",
                    " cpu=",
                    cpu_val,
                    " gpu=",
                    gpu_val,
                    " abs=",
                    abs_err,
                    " rel=",
                    rel_err,
                )
            qvel_fails += 1
            qvel_pass = False

    var all_pass = qpos_pass and qvel_pass
    if all_pass:
        print(
            "  ALL OK  qpos(abs=",
            qpos_max_abs,
            " rel=",
            qpos_max_rel,
            ") qvel(abs=",
            qvel_max_abs,
            " rel=",
            qvel_max_rel,
            ")",
        )
    else:
        print(
            "  FAILED  qpos:",
            qpos_fails,
            "fails (abs=",
            qpos_max_abs,
            " rel=",
            qpos_max_rel,
            ")",
            " qvel:",
            qvel_fails,
            "fails (abs=",
            qvel_max_abs,
            " rel=",
            qvel_max_rel,
            ")",
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

    assert_true(all_pass, "CPU vs GPU mismatch for: " + test_name)


fn test_free_fall_1_step() raises:
    print("=" * 60)
    print("Full Step (no contacts): CPU vs GPU")
    print("=" * 60)
    print("Model: HalfCheetah (NQ=9, NV=9)")
    print("Integrator: Euler + NewtonSolver")
    print("Cone: pyramidal")
    print("Precision: float32")
    print("Tolerances: qpos abs=", QPOS_ABS_TOL, " rel=", QPOS_REL_TOL)
    print("            qvel abs=", QVEL_ABS_TOL, " rel=", QVEL_REL_TOL)
    print()

    var ctx = DeviceContext()
    var state_host = create_state_buffer[
        DTYPE, NQ, NV, NBODY, MAX_CONTACTS, HalfCheetahModel.NSITE, BATCH
    ](ctx)
    var state_buf = ctx.enqueue_create_buffer[DTYPE](BATCH * STATE_SIZE)
    var workspace_buf = ctx.enqueue_create_buffer[DTYPE](BATCH * WS_SIZE)
    var ws_host = ctx.enqueue_create_host_buffer[DTYPE](BATCH * WS_SIZE)
    var model_buf = ctx.enqueue_create_buffer[DTYPE](MODEL_SIZE)

    var qpos = InlineArray[Float64, NQ](fill=0.0)
    qpos[1] = 1.5  # rootz high enough for free fall
    var qvel = InlineArray[Float64, NV](fill=0.0)
    var act = InlineArray[Float64, ACTION_DIM](fill=0.0)
    compare_step(
        "Free fall (1 step)",
        qpos,
        qvel,
        act,
        1,
        ctx,
        model_buf,
        state_host,
        state_buf,
        workspace_buf,
        ws_host,
    )
    print()


fn test_free_fall_with_actions_1_step() raises:
    var ctx = DeviceContext()
    var state_host = create_state_buffer[
        DTYPE, NQ, NV, NBODY, MAX_CONTACTS, HalfCheetahModel.NSITE, BATCH
    ](ctx)
    var state_buf = ctx.enqueue_create_buffer[DTYPE](BATCH * STATE_SIZE)
    var workspace_buf = ctx.enqueue_create_buffer[DTYPE](BATCH * WS_SIZE)
    var ws_host = ctx.enqueue_create_host_buffer[DTYPE](BATCH * WS_SIZE)
    var model_buf = ctx.enqueue_create_buffer[DTYPE](MODEL_SIZE)

    var qpos = InlineArray[Float64, NQ](fill=0.0)
    qpos[1] = 1.5  # rootz high enough for free fall
    var qvel = InlineArray[Float64, NV](fill=0.0)
    var act = InlineArray[Float64, ACTION_DIM](fill=0.0)
    act[0] = 0.5  # bthigh
    act[1] = -0.3  # bshin
    act[2] = 0.2  # bfoot
    act[3] = 0.5  # fthigh
    act[4] = -0.3  # fshin
    act[5] = 0.1  # ffoot
    compare_step(
        "Free fall + actions (1 step)",
        qpos,
        qvel,
        act,
        1,
        ctx,
        model_buf,
        state_host,
        state_buf,
        workspace_buf,
        ws_host,
    )
    print()


fn test_moving_with_actions_1_step() raises:
    var ctx = DeviceContext()
    var state_host = create_state_buffer[
        DTYPE, NQ, NV, NBODY, MAX_CONTACTS, HalfCheetahModel.NSITE, BATCH
    ](ctx)
    var state_buf = ctx.enqueue_create_buffer[DTYPE](BATCH * STATE_SIZE)
    var workspace_buf = ctx.enqueue_create_buffer[DTYPE](BATCH * WS_SIZE)
    var ws_host = ctx.enqueue_create_host_buffer[DTYPE](BATCH * WS_SIZE)
    var model_buf = ctx.enqueue_create_buffer[DTYPE](MODEL_SIZE)

    var qpos = InlineArray[Float64, NQ](fill=0.0)
    qpos[1] = 1.5  # high
    qpos[2] = 0.1  # slight pitch
    qpos[3] = -0.3  # bthigh bent
    qpos[6] = 0.4  # fthigh bent
    var qvel = InlineArray[Float64, NV](fill=0.0)
    qvel[0] = 2.0  # rootx vel
    qvel[2] = 0.5  # rooty vel
    qvel[3] = -1.0  # bthigh vel
    qvel[6] = 1.2  # fthigh vel
    var act = InlineArray[Float64, ACTION_DIM](fill=0.0)
    act[0] = 1.0
    act[1] = -0.5
    act[3] = 1.0
    act[4] = -0.5
    compare_step(
        "Moving + actions (1 step)",
        qpos,
        qvel,
        act,
        1,
        ctx,
        model_buf,
        state_host,
        state_buf,
        workspace_buf,
        ws_host,
    )
    print()


fn test_free_fall_10_steps() raises:
    var ctx = DeviceContext()
    var state_host = create_state_buffer[
        DTYPE, NQ, NV, NBODY, MAX_CONTACTS, HalfCheetahModel.NSITE, BATCH
    ](ctx)
    var state_buf = ctx.enqueue_create_buffer[DTYPE](BATCH * STATE_SIZE)
    var workspace_buf = ctx.enqueue_create_buffer[DTYPE](BATCH * WS_SIZE)
    var ws_host = ctx.enqueue_create_host_buffer[DTYPE](BATCH * WS_SIZE)
    var model_buf = ctx.enqueue_create_buffer[DTYPE](MODEL_SIZE)

    var qpos = InlineArray[Float64, NQ](fill=0.0)
    qpos[1] = 1.5  # rootz high enough for free fall
    var qvel = InlineArray[Float64, NV](fill=0.0)
    var act = InlineArray[Float64, ACTION_DIM](fill=0.0)
    compare_step(
        "Free fall (10 steps)",
        qpos,
        qvel,
        act,
        10,
        ctx,
        model_buf,
        state_host,
        state_buf,
        workspace_buf,
        ws_host,
    )
    print()


fn test_moving_with_actions_10_steps() raises:
    var ctx = DeviceContext()
    var state_host = create_state_buffer[
        DTYPE, NQ, NV, NBODY, MAX_CONTACTS, HalfCheetahModel.NSITE, BATCH
    ](ctx)
    var state_buf = ctx.enqueue_create_buffer[DTYPE](BATCH * STATE_SIZE)
    var workspace_buf = ctx.enqueue_create_buffer[DTYPE](BATCH * WS_SIZE)
    var ws_host = ctx.enqueue_create_host_buffer[DTYPE](BATCH * WS_SIZE)
    var model_buf = ctx.enqueue_create_buffer[DTYPE](MODEL_SIZE)

    var qpos = InlineArray[Float64, NQ](fill=0.0)
    qpos[1] = 1.5  # high
    qpos[2] = 0.1  # slight pitch
    qpos[3] = -0.3  # bthigh bent
    qpos[6] = 0.4  # fthigh bent
    var qvel = InlineArray[Float64, NV](fill=0.0)
    qvel[0] = 2.0  # rootx vel
    qvel[2] = 0.5  # rooty vel
    qvel[3] = -1.0  # bthigh vel
    qvel[6] = 1.2  # fthigh vel
    var act = InlineArray[Float64, ACTION_DIM](fill=0.0)
    act[0] = 1.0
    act[1] = -0.5
    act[3] = 1.0
    act[4] = -0.5
    compare_step(
        "Moving + actions (10 steps)",
        qpos,
        qvel,
        act,
        10,
        ctx,
        model_buf,
        state_host,
        state_buf,
        workspace_buf,
        ws_host,
    )
    print()


fn test_extreme_velocities_1_step() raises:
    var ctx = DeviceContext()
    var state_host = create_state_buffer[
        DTYPE, NQ, NV, NBODY, MAX_CONTACTS, HalfCheetahModel.NSITE, BATCH
    ](ctx)
    var state_buf = ctx.enqueue_create_buffer[DTYPE](BATCH * STATE_SIZE)
    var workspace_buf = ctx.enqueue_create_buffer[DTYPE](BATCH * WS_SIZE)
    var ws_host = ctx.enqueue_create_host_buffer[DTYPE](BATCH * WS_SIZE)
    var model_buf = ctx.enqueue_create_buffer[DTYPE](MODEL_SIZE)

    var qpos = InlineArray[Float64, NQ](fill=0.0)
    qpos[1] = 2.0  # very high
    qpos[3] = -0.52
    qpos[6] = -1.0
    var qvel = InlineArray[Float64, NV](fill=0.0)
    qvel[0] = 5.0
    qvel[1] = -2.0
    qvel[2] = 3.0
    qvel[3] = -5.0
    qvel[4] = 5.0
    qvel[5] = -3.0
    qvel[6] = 5.0
    qvel[7] = -5.0
    qvel[8] = 3.0
    var act = InlineArray[Float64, ACTION_DIM](fill=0.0)
    compare_step(
        "Extreme velocities (1 step)",
        qpos,
        qvel,
        act,
        1,
        ctx,
        model_buf,
        state_host,
        state_buf,
        workspace_buf,
        ws_host,
    )
    print()


fn main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
