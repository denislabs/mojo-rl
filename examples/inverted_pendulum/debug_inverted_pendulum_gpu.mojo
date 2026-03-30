"""Diagnostic: compare model buffer in workspace vs fresh allocation.

Root cause investigation: RK4 produces NaN when using workspace_ptr model
but works with a fresh model buffer.

Run with:
    pixi run -e apple mojo run -I . examples/inverted_pendulum/debug_inverted_pendulum_gpu.mojo
"""

from std.random import seed
from std.math import abs
from std.gpu.host import DeviceContext, DeviceBuffer
from layout import Layout, LayoutTensor

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

comptime ENV = InvertedPendulum[gpu_dtype, TERMINATE_ON_UNHEALTHY=True]
comptime N_ENVS = 4
comptime STATE_SIZE = ENV.STATE_SIZE
comptime NQ = ENV.NQ
comptime NV = ENV.NV
comptime NBODY = ENV.NUM_BODIES
comptime NJOINT = ENV.NUM_JOINTS
comptime NGEOM = ENV.NGEOM
comptime MAX_CONTACTS = ENV.MAX_CONTACTS
comptime ACTION_DIM = ENV.ACTION_DIM

comptime MODEL_SIZE = model_size_with_invweight[NBODY, NJOINT, NV, NGEOM]()
comptime SOLVER_WS = NewtonSolver.solver_workspace_size[NV, MAX_CONTACTS]()
comptime WS_SIZE = (
    integrator_workspace_size[NV, NBODY]()
    + NV * NV
    + SOLVER_WS
    + rk4_extra_workspace_size[NQ, NV]()
)
comptime QPOS_OFF = qpos_offset[NQ, NV]()
comptime QVEL_OFF = qvel_offset[NQ, NV]()
comptime QACC_OFF = qacc_offset[NQ, NV]()
comptime QFRC_OFF = qfrc_offset[NQ, NV]()


def main() raises:
    seed(42)
    print("=" * 70)
    print("Model buffer comparison: workspace vs fresh")
    print("=" * 70)
    print("MODEL_SIZE =", MODEL_SIZE)
    print("WS_SIZE =", WS_SIZE)
    print("STEP_WS_SHARED =", ENV.STEP_WS_SHARED)
    print("STEP_WS_PER_ENV =", ENV.STEP_WS_PER_ENV)
    print()

    with DeviceContext() as ctx:
        # ====== A: Fresh model buffer (known working) ======
        var fresh_buf = ctx.enqueue_create_buffer[gpu_dtype](MODEL_SIZE)
        InvertedPendulumModel.init_model_gpu(ctx, fresh_buf)

        var fresh_host = ctx.enqueue_create_host_buffer[gpu_dtype](MODEL_SIZE)
        ctx.enqueue_copy(fresh_host, fresh_buf)
        ctx.synchronize()

        # ====== B: Workspace-based model buffer (step_kernel_gpu path) ======
        var ws_total = ENV.STEP_WS_SHARED + N_ENVS * ENV.STEP_WS_PER_ENV
        var workspace_buf = ctx.enqueue_create_buffer[gpu_dtype](ws_total)
        ENV.init_step_workspace_gpu[N_ENVS](ctx, workspace_buf)
        ctx.synchronize()

        # Read back the model portion of workspace
        var ws_host = ctx.enqueue_create_host_buffer[gpu_dtype](ws_total)
        ctx.enqueue_copy(ws_host, workspace_buf)
        ctx.synchronize()

        # ====== Compare byte-by-byte ======
        print("Comparing model buffers (fresh vs workspace):")
        var mismatches = 0
        var first_mismatch = -1
        for i in range(MODEL_SIZE):
            var fresh_val = Float64(fresh_host[i])
            var ws_val = Float64(ws_host[i])
            if fresh_val != ws_val:
                if mismatches < 10:
                    print(
                        "  MISMATCH [",
                        i,
                        "]: fresh=",
                        fresh_val,
                        " ws=",
                        ws_val,
                    )
                if first_mismatch == -1:
                    first_mismatch = i
                mismatches += 1

        if mismatches == 0:
            print("  ALL MATCH - model data is identical")
        else:
            print("  TOTAL MISMATCHES:", mismatches, "/ ", MODEL_SIZE)
            print("  First mismatch at index:", first_mismatch)
        print()

        # ====== C: Test RK4 with workspace model ======
        print("Testing RK4 with workspace-based model buffer:")

        # Create state buffer and reset
        var states_buf = ctx.enqueue_create_buffer[gpu_dtype](
            N_ENVS * STATE_SIZE
        )
        ENV.reset_kernel_gpu[N_ENVS, STATE_SIZE](ctx, states_buf, rng_seed=0)

        # Set qfrc manually
        var h = ctx.enqueue_create_host_buffer[gpu_dtype](N_ENVS * STATE_SIZE)
        ctx.enqueue_copy(h, states_buf)
        ctx.synchronize()
        for e in range(N_ENVS):
            h[e * STATE_SIZE + QFRC_OFF] = Scalar[gpu_dtype](50.0)
        ctx.enqueue_copy(states_buf, h)

        # Extract workspace model as DeviceBuffer (same as step_kernel_gpu does)
        var ws_model_buf = DeviceBuffer[gpu_dtype](
            ctx,
            workspace_buf.unsafe_ptr(),
            MODEL_SIZE,
            owning=False,
        )
        var ws_per_env_buf = DeviceBuffer[gpu_dtype](
            ctx,
            workspace_buf.unsafe_ptr() + MODEL_SIZE,
            N_ENVS * WS_SIZE,
            owning=False,
        )

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
        ](ctx, states_buf, ws_model_buf, ws_per_env_buf)

        ctx.enqueue_copy(h, states_buf)
        ctx.synchronize()
        print(
            "  qacc=[",
            Float64(h[QACC_OFF]),
            ",",
            Float64(h[QACC_OFF + 1]),
            "] qvel=[",
            Float64(h[QVEL_OFF]),
            ",",
            Float64(h[QVEL_OFF + 1]),
            "]",
        )
        var ws_qacc = Float64(h[QACC_OFF])
        if ws_qacc != ws_qacc:
            print("  >>> NaN with workspace model! <<<")
        else:
            print("  >>> OK with workspace model <<<")
        print()

        # ====== D: Test RK4 with fresh model (same workspace) ======
        print("Testing RK4 with fresh model buffer (same per-env workspace):")

        ENV.reset_kernel_gpu[N_ENVS, STATE_SIZE](ctx, states_buf, rng_seed=0)
        ctx.enqueue_copy(h, states_buf)
        ctx.synchronize()
        for e in range(N_ENVS):
            h[e * STATE_SIZE + QFRC_OFF] = Scalar[gpu_dtype](50.0)
        ctx.enqueue_copy(states_buf, h)

        # Fresh workspace for the integrator
        var fresh_ws = ctx.enqueue_create_buffer[gpu_dtype](N_ENVS * WS_SIZE)

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
        ](ctx, states_buf, fresh_buf, fresh_ws)

        ctx.enqueue_copy(h, states_buf)
        ctx.synchronize()
        print(
            "  qacc=[",
            Float64(h[QACC_OFF]),
            ",",
            Float64(h[QACC_OFF + 1]),
            "] qvel=[",
            Float64(h[QVEL_OFF]),
            ",",
            Float64(h[QVEL_OFF + 1]),
            "]",
        )
        var fresh_qacc = Float64(h[QACC_OFF])
        if fresh_qacc != fresh_qacc:
            print("  >>> NaN with fresh model! <<<")
        else:
            print("  >>> OK with fresh model <<<")

    print()
    print(">>> Diagnostic complete <<<")
