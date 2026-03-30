"""Diagnostic: RK4 with offset workspace pointer vs fresh allocation.

Tests whether the workspace sub-pointer causes NaN in RK4.
Uses hardcoded correct sizes (not from ENV config) to avoid coupling.

Run with:
    pixi run -e apple mojo run -I . examples/inverted_pendulum/debug_inverted_pendulum_gpu.mojo
"""

from std.random import seed
from std.gpu.host import DeviceContext, DeviceBuffer

from mojo_rl.envs.inverted_pendulum import InvertedPendulum
from mojo_rl.envs.inverted_pendulum.inverted_pendulum_xml import (
    InvertedPendulumModel,
)
from mojo_rl.physics3d.integrator import RK4Integrator
from mojo_rl.physics3d.solver import NewtonSolver
from mojo_rl.physics3d.gpu.constants import (
    qpos_offset,
    qvel_offset,
    qacc_offset,
    qfrc_offset,
    state_size,
    model_size_with_invweight,
    integrator_workspace_size,
    rk4_extra_workspace_size,
)
from mojo_rl.nn import dtype as gpu_dtype

comptime N_ENVS = 4
comptime NQ = InvertedPendulumModel.NQ
comptime NV = InvertedPendulumModel.NV
comptime NBODY = InvertedPendulumModel.NBODY
comptime NJOINT = InvertedPendulumModel.NJOINT
comptime NGEOM = InvertedPendulumModel.NGEOM
comptime MAX_CONTACTS = InvertedPendulumModel.MAX_CONTACTS

comptime STATE_SIZE = state_size[NQ, NV, NBODY, MAX_CONTACTS]()
comptime MODEL_SIZE = model_size_with_invweight[NBODY, NJOINT, NV, NGEOM]()
comptime SOLVER_WS = NewtonSolver.solver_workspace_size[NV, MAX_CONTACTS]()
comptime RK4_EXTRA = rk4_extra_workspace_size[NQ, NV]()
comptime WS_SIZE = (
    integrator_workspace_size[NV, NBODY]() + NV * NV + SOLVER_WS + RK4_EXTRA
)

comptime QPOS_OFF = qpos_offset[NQ, NV]()
comptime QVEL_OFF = qvel_offset[NQ, NV]()
comptime QACC_OFF = qacc_offset[NQ, NV]()
comptime QFRC_OFF = qfrc_offset[NQ, NV]()

comptime ENV = InvertedPendulum[gpu_dtype, TERMINATE_ON_UNHEALTHY=True]


def run_rk4_test(
    ctx: DeviceContext,
    label: String,
    mut model_buf: DeviceBuffer[gpu_dtype],
    mut ws_buf: DeviceBuffer[gpu_dtype],
) raises:
    """Reset env, set qfrc=50, run 1 RK4 step, print qacc."""
    var states_buf = ctx.enqueue_create_buffer[gpu_dtype](N_ENVS * STATE_SIZE)
    ENV.reset_kernel_gpu[N_ENVS, STATE_SIZE](ctx, states_buf, rng_seed=0)

    var h = ctx.enqueue_create_host_buffer[gpu_dtype](N_ENVS * STATE_SIZE)
    ctx.enqueue_copy(h, states_buf)
    ctx.synchronize()
    for e in range(N_ENVS):
        h[e * STATE_SIZE + QFRC_OFF] = Scalar[gpu_dtype](50.0)
    ctx.enqueue_copy(states_buf, h)
    ctx.synchronize()

    RK4Integrator[SOLVER=NewtonSolver].step_gpu[
        gpu_dtype,
        NQ,
        NV,
        NBODY,
        NJOINT,
        MAX_CONTACTS,
        N_ENVS,
        NGEOM,
        STEP_THREADS=NV,
    ](ctx, states_buf, model_buf, ws_buf)

    ctx.enqueue_copy(h, states_buf)
    ctx.synchronize()
    var a0 = Float64(h[QACC_OFF])
    var a1 = Float64(h[QACC_OFF + 1])
    var v0 = Float64(h[QVEL_OFF])
    var v1 = Float64(h[QVEL_OFF + 1])
    print(
        " ",
        label,
        "| qacc=[",
        a0,
        ",",
        a1,
        "] qvel=[",
        v0,
        ",",
        v1,
        "]",
    )
    if a0 != a0:
        print("  >>> NaN! <<<")


def main() raises:
    seed(42)
    print("=" * 70)
    print("RK4 offset-pointer workspace diagnostic")
    print("=" * 70)
    print(
        "MODEL_SIZE=",
        MODEL_SIZE,
        " WS_SIZE=",
        WS_SIZE,
        " RK4_EXTRA=",
        RK4_EXTRA,
    )
    print()

    with DeviceContext() as ctx:
        # Fresh model
        var fresh_model = ctx.enqueue_create_buffer[gpu_dtype](MODEL_SIZE)
        InvertedPendulumModel.init_model_gpu(ctx, fresh_model)

        # ====== Test 1: Fresh model + fresh workspace ======
        var ws1 = ctx.enqueue_create_buffer[gpu_dtype](N_ENVS * WS_SIZE)
        run_rk4_test(ctx, "Fresh model + fresh WS", fresh_model, ws1)

        # ====== Test 2: Fresh model + offset workspace ======
        # Simulate the step_kernel_gpu pattern: model at start, workspace after
        var combined_size = MODEL_SIZE + N_ENVS * WS_SIZE
        var combined = ctx.enqueue_create_buffer[gpu_dtype](combined_size)

        # Copy fresh model data to beginning of combined buffer
        var model_host = ctx.enqueue_create_host_buffer[gpu_dtype](MODEL_SIZE)
        ctx.enqueue_copy(model_host, fresh_model)
        ctx.synchronize()

        var combined_host = ctx.enqueue_create_host_buffer[gpu_dtype](
            combined_size
        )
        for i in range(MODEL_SIZE):
            combined_host[i] = model_host[i]
        for i in range(N_ENVS * WS_SIZE):
            combined_host[MODEL_SIZE + i] = Scalar[gpu_dtype](0)
        ctx.enqueue_copy(combined, combined_host)
        ctx.synchronize()

        # Create sub-pointer views (EXACTLY like step_kernel_gpu does)
        var sub_model = DeviceBuffer[gpu_dtype](
            ctx,
            combined.unsafe_ptr(),
            MODEL_SIZE,
            owning=False,
        )
        var sub_ws = DeviceBuffer[gpu_dtype](
            ctx,
            combined.unsafe_ptr() + MODEL_SIZE,
            N_ENVS * WS_SIZE,
            owning=False,
        )
        run_rk4_test(ctx, "Sub-ptr model + sub-ptr WS", sub_model, sub_ws)

        # ====== Test 3: Fresh model + offset workspace (offset only) ======
        var ws_only_combined = ctx.enqueue_create_buffer[gpu_dtype](
            MODEL_SIZE + N_ENVS * WS_SIZE
        )
        # Don't init model, just use offset for workspace
        var sub_ws2 = DeviceBuffer[gpu_dtype](
            ctx,
            ws_only_combined.unsafe_ptr() + MODEL_SIZE,
            N_ENVS * WS_SIZE,
            owning=False,
        )
        run_rk4_test(
            ctx, "Fresh model + offset-only WS", fresh_model, sub_ws2
        )

        # ====== Test 4: init_step_workspace (actual training path) ======
        var ws_total = MODEL_SIZE + N_ENVS * WS_SIZE
        var training_ws = ctx.enqueue_create_buffer[gpu_dtype](ws_total)
        ENV.init_step_workspace_gpu[N_ENVS](ctx, training_ws)
        ctx.synchronize()

        var train_model = DeviceBuffer[gpu_dtype](
            ctx,
            training_ws.unsafe_ptr(),
            MODEL_SIZE,
            owning=False,
        )
        var train_ws = DeviceBuffer[gpu_dtype](
            ctx,
            training_ws.unsafe_ptr() + MODEL_SIZE,
            N_ENVS * WS_SIZE,
            owning=False,
        )
        run_rk4_test(
            ctx, "init_step_workspace model+WS", train_model, train_ws
        )

    print()
    print(">>> Done <<<")
