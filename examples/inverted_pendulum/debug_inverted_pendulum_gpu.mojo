"""Diagnostic test for InvertedPendulum GPU physics.

Tests whether the RK4 integrator actually updates state.
Also tests Euler integrator for comparison.

Run with:
    pixi run -e apple mojo run -I . examples/inverted_pendulum/debug_inverted_pendulum_gpu.mojo
"""

from std.random import seed
from std.gpu.host import DeviceContext, DeviceBuffer
from layout import Layout, LayoutTensor

from mojo_rl.envs.inverted_pendulum import InvertedPendulum
from mojo_rl.envs.inverted_pendulum.inverted_pendulum_xml import (
    InvertedPendulumModel,
)
from mojo_rl.physics3d.integrator import RK4Integrator
from mojo_rl.physics3d.integrator.euler_integrator import EulerIntegrator
from mojo_rl.physics3d.solver import NewtonSolver
from mojo_rl.physics3d.gpu.constants import (
    metadata_offset,
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

comptime META_OFF = metadata_offset[NQ, NV, NBODY, MAX_CONTACTS]()
comptime QPOS_OFF = qpos_offset[NQ, NV]()
comptime QVEL_OFF = qvel_offset[NQ, NV]()
comptime QACC_OFF = qacc_offset[NQ, NV]()
comptime QFRC_OFF = qfrc_offset[NQ, NV]()

comptime MODEL_SIZE = model_size_with_invweight[NBODY, NJOINT, NV, NGEOM]()


def print_env_state(
    host: type_of(DeviceContext().enqueue_create_host_buffer[gpu_dtype](1)),
    env: Int,
    label: String,
):
    var base = env * STATE_SIZE
    print(
        "  ",
        label,
        " | qpos=[",
        Float64(host[base + QPOS_OFF + 0]),
        ",",
        Float64(host[base + QPOS_OFF + 1]),
        "] qvel=[",
        Float64(host[base + QVEL_OFF + 0]),
        ",",
        Float64(host[base + QVEL_OFF + 1]),
        "] qacc=[",
        Float64(host[base + QACC_OFF + 0]),
        ",",
        Float64(host[base + QACC_OFF + 1]),
        "] qfrc=[",
        Float64(host[base + QFRC_OFF + 0]),
        ",",
        Float64(host[base + QFRC_OFF + 1]),
        "]",
    )


def main() raises:
    seed(42)
    print("=" * 60)
    print("InvertedPendulum GPU Physics Diagnostic")
    print("=" * 60)
    print()
    print("NQ=", NQ, "NV=", NV, "NBODY=", NBODY, "NJOINT=", NJOINT)
    print("STATE_SIZE=", STATE_SIZE, "MODEL_SIZE=", MODEL_SIZE)
    print()

    with DeviceContext() as ctx:
        var states_buf = ctx.enqueue_create_buffer[gpu_dtype](
            N_ENVS * STATE_SIZE
        )
        var host_states = ctx.enqueue_create_host_buffer[gpu_dtype](
            N_ENVS * STATE_SIZE
        )

        # =====================================================
        # Test 1: RK4 Integrator (what InvertedPendulum uses)
        # =====================================================
        print("=" * 60)
        print("TEST 1: RK4 Integrator")
        print("=" * 60)

        # Reset
        ENV.reset_kernel_gpu[N_ENVS, STATE_SIZE](ctx, states_buf, rng_seed=0)

        # Apply constant force to qfrc directly
        var host_tmp = ctx.enqueue_create_host_buffer[gpu_dtype](
            N_ENVS * STATE_SIZE
        )
        ctx.enqueue_copy(host_tmp, states_buf)
        ctx.synchronize()
        for e in range(N_ENVS):
            # Set qfrc[0] = 50.0 (gear=100 * action=0.5)
            host_tmp[e * STATE_SIZE + QFRC_OFF + 0] = Scalar[gpu_dtype](50.0)
            host_tmp[e * STATE_SIZE + QFRC_OFF + 1] = Scalar[gpu_dtype](0.0)
        ctx.enqueue_copy(states_buf, host_tmp)

        ctx.enqueue_copy(host_states, states_buf)
        ctx.synchronize()
        print_env_state(host_states, 0, "Before RK4")

        # Create model + workspace for RK4
        comptime SOLVER_WS = NewtonSolver.solver_workspace_size[
            NV, MAX_CONTACTS
        ]()
        comptime WS_SIZE = (
            integrator_workspace_size[NV, NBODY]()
            + NV * NV
            + SOLVER_WS
            + rk4_extra_workspace_size[NQ, NV]()
        )

        var model_buf = ctx.enqueue_create_buffer[gpu_dtype](MODEL_SIZE)
        InvertedPendulumModel.init_model_gpu(ctx, model_buf)
        var ws_buf = ctx.enqueue_create_buffer[gpu_dtype](N_ENVS * WS_SIZE)

        # Run 1 RK4 step
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

        ctx.enqueue_copy(host_states, states_buf)
        ctx.synchronize()
        print_env_state(host_states, 0, "After RK4 (STEP_THREADS=NV)")

        # Check if state changed
        var rk4_qpos0 = Float64(host_states[QPOS_OFF + 0])
        var rk4_qpos1 = Float64(host_states[QPOS_OFF + 1])
        var rk4_qvel0 = Float64(host_states[QVEL_OFF + 0])
        var rk4_qacc0 = Float64(host_states[QACC_OFF + 0])

        if rk4_qacc0 != rk4_qacc0:
            print("  >>> QACC IS NaN! Forward dynamics produced NaN <<<")
        elif rk4_qacc0 == 0.0 and rk4_qvel0 == 0.0:
            print("  >>> STATE FROZEN: zero acceleration and velocity <<<")
        else:
            print("  >>> RK4 physics updating normally <<<")
        print()

        # =====================================================
        # Test 1b: RK4 with STEP_THREADS=1
        # =====================================================
        print("=" * 60)
        print("TEST 1b: RK4 Integrator (STEP_THREADS=1)")
        print("=" * 60)

        # Re-reset
        ENV.reset_kernel_gpu[N_ENVS, STATE_SIZE](ctx, states_buf, rng_seed=0)
        ctx.enqueue_copy(host_tmp, states_buf)
        ctx.synchronize()
        for e in range(N_ENVS):
            host_tmp[e * STATE_SIZE + QFRC_OFF + 0] = Scalar[gpu_dtype](50.0)
            host_tmp[e * STATE_SIZE + QFRC_OFF + 1] = Scalar[gpu_dtype](0.0)
        ctx.enqueue_copy(states_buf, host_tmp)

        ctx.enqueue_copy(host_states, states_buf)
        ctx.synchronize()
        print_env_state(host_states, 0, "Before RK4 (ST=1)")

        var ws_buf2 = ctx.enqueue_create_buffer[gpu_dtype](N_ENVS * WS_SIZE)
        RK4Integrator[SOLVER=NewtonSolver].step_gpu[
            gpu_dtype,
            NQ,
            NV,
            NBODY,
            NJOINT,
            MAX_CONTACTS,
            N_ENVS,
            NGEOM,
            STEP_THREADS=1,
        ](ctx, states_buf, model_buf, ws_buf2)

        ctx.enqueue_copy(host_states, states_buf)
        ctx.synchronize()
        print_env_state(host_states, 0, "After RK4 (ST=1)")

        var rk4_st1_qacc0 = Float64(host_states[QACC_OFF + 0])
        var rk4_st1_qvel0 = Float64(host_states[QVEL_OFF + 0])
        if rk4_st1_qacc0 != rk4_st1_qacc0:
            print("  >>> QACC IS NaN! <<<")
        elif rk4_st1_qacc0 == 0.0 and rk4_st1_qvel0 == 0.0:
            print("  >>> STATE FROZEN <<<")
        else:
            print("  >>> RK4 (ST=1) physics updating normally <<<")
        print()

        # =====================================================
        # Test 2: Euler Integrator (for comparison)
        # =====================================================
        print("=" * 60)
        print("TEST 2: Euler Integrator")
        print("=" * 60)

        # Re-reset
        ENV.reset_kernel_gpu[N_ENVS, STATE_SIZE](ctx, states_buf, rng_seed=0)
        ctx.enqueue_copy(host_tmp, states_buf)
        ctx.synchronize()
        for e in range(N_ENVS):
            host_tmp[e * STATE_SIZE + QFRC_OFF + 0] = Scalar[gpu_dtype](50.0)
            host_tmp[e * STATE_SIZE + QFRC_OFF + 1] = Scalar[gpu_dtype](0.0)
        ctx.enqueue_copy(states_buf, host_tmp)

        ctx.enqueue_copy(host_states, states_buf)
        ctx.synchronize()
        print_env_state(host_states, 0, "Before Euler")

        # Euler needs different workspace size (no RK4 extra)
        comptime EULER_WS_SIZE = (
            integrator_workspace_size[NV, NBODY]() + NV * NV + SOLVER_WS
        )
        var euler_ws_buf = ctx.enqueue_create_buffer[gpu_dtype](
            N_ENVS * EULER_WS_SIZE
        )

        EulerIntegrator[SOLVER=NewtonSolver].step_gpu[
            gpu_dtype,
            NQ,
            NV,
            NBODY,
            NJOINT,
            MAX_CONTACTS,
            N_ENVS,
            NGEOM,
            STEP_THREADS=NV,
        ](ctx, states_buf, model_buf, euler_ws_buf)

        ctx.enqueue_copy(host_states, states_buf)
        ctx.synchronize()
        print_env_state(host_states, 0, "After Euler")

        var euler_qacc0 = Float64(host_states[QACC_OFF + 0])
        var euler_qvel0 = Float64(host_states[QVEL_OFF + 0])
        if euler_qacc0 != euler_qacc0:
            print("  >>> QACC IS NaN! <<<")
        elif euler_qacc0 == 0.0 and euler_qvel0 == 0.0:
            print("  >>> STATE FROZEN <<<")
        else:
            print("  >>> Euler physics updating normally <<<")

    print()
    print(">>> Diagnostic complete <<<")
