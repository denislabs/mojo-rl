"""Diagnostic: step-by-step isolation of InvertedPendulum GPU physics freeze.

Calls each piece of step_kernel_gpu separately to find where state change disappears.

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
from mojo_rl.physics3d.gpu.cfrc_ext_gpu import compute_cfrc_ext_gpu
from mojo_rl.physics3d.gpu.cvel_gpu import compute_cvel_gpu
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
comptime MAX_EQUALITY = ENV.MAX_EQUALITY
comptime CONE_TYPE = ENV.CONE_TYPE
comptime ACTION_DIM = ENV.ACTION_DIM
comptime OBS_DIM = ENV.OBS_DIM
comptime NSITE = ENV.NSITE

comptime META_OFF = metadata_offset[NQ, NV, NBODY, MAX_CONTACTS]()
comptime QPOS_OFF = qpos_offset[NQ, NV]()
comptime QVEL_OFF = qvel_offset[NQ, NV]()
comptime QACC_OFF = qacc_offset[NQ, NV]()
comptime QFRC_OFF = qfrc_offset[NQ, NV]()

comptime MODEL_SIZE = model_size_with_invweight[NBODY, NJOINT, NV, NGEOM]()
comptime SOLVER_WS = NewtonSolver.solver_workspace_size[NV, MAX_CONTACTS]()
comptime WS_SIZE = (
    integrator_workspace_size[NV, NBODY]()
    + NV * NV
    + SOLVER_WS
    + rk4_extra_workspace_size[NQ, NV]()
)


def dump(
    ctx: DeviceContext,
    states_buf: DeviceBuffer[gpu_dtype],
    label: String,
) raises:
    var h = ctx.enqueue_create_host_buffer[gpu_dtype](N_ENVS * STATE_SIZE)
    ctx.enqueue_copy(h, states_buf)
    ctx.synchronize()
    var b = 0 * STATE_SIZE  # env 0
    print(
        "  ",
        label,
        "| qpos=[",
        Float64(h[b + QPOS_OFF]),
        ",",
        Float64(h[b + QPOS_OFF + 1]),
        "] qvel=[",
        Float64(h[b + QVEL_OFF]),
        ",",
        Float64(h[b + QVEL_OFF + 1]),
        "] qacc=[",
        Float64(h[b + QACC_OFF]),
        ",",
        Float64(h[b + QACC_OFF + 1]),
        "] qfrc=[",
        Float64(h[b + QFRC_OFF]),
        ",",
        Float64(h[b + QFRC_OFF + 1]),
        "]",
    )


def main() raises:
    seed(42)
    print("=" * 70)
    print("Step-by-step isolation of InvertedPendulum GPU physics freeze")
    print("=" * 70)
    print()

    with DeviceContext() as ctx:
        var states_buf = ctx.enqueue_create_buffer[gpu_dtype](
            N_ENVS * STATE_SIZE
        )
        var actions_buf = ctx.enqueue_create_buffer[gpu_dtype](
            N_ENVS * ACTION_DIM
        )
        var rewards_buf = ctx.enqueue_create_buffer[gpu_dtype](N_ENVS)
        var dones_buf = ctx.enqueue_create_buffer[gpu_dtype](N_ENVS)
        var terminated_buf = ctx.enqueue_create_buffer[gpu_dtype](N_ENVS)
        var obs_buf = ctx.enqueue_create_buffer[gpu_dtype](N_ENVS * OBS_DIM)

        # Set actions = 0.5
        var host_act = ctx.enqueue_create_host_buffer[gpu_dtype](
            N_ENVS * ACTION_DIM
        )
        for i in range(N_ENVS):
            host_act[i] = Scalar[gpu_dtype](0.5)
        ctx.enqueue_copy(actions_buf, host_act)

        var model_buf = ctx.enqueue_create_buffer[gpu_dtype](MODEL_SIZE)
        InvertedPendulumModel.init_model_gpu(ctx, model_buf)
        var ws_buf = ctx.enqueue_create_buffer[gpu_dtype](N_ENVS * WS_SIZE)

        # ===== TEST A: Full step_kernel_gpu =====
        print("TEST A: Full step_kernel_gpu (what training uses)")
        print("-" * 70)
        ENV.reset_kernel_gpu[N_ENVS, STATE_SIZE](ctx, states_buf, rng_seed=0)
        dump(ctx, states_buf, "After reset")

        # Workspace for step_kernel_gpu
        var full_ws_size = ENV.STEP_WS_SHARED + N_ENVS * ENV.STEP_WS_PER_ENV
        if full_ws_size == 0:
            full_ws_size = 1
        var full_ws_buf = ctx.enqueue_create_buffer[gpu_dtype](full_ws_size)
        ENV.init_step_workspace_gpu[N_ENVS](ctx, full_ws_buf)

        ENV.step_kernel_gpu[N_ENVS, STATE_SIZE, OBS_DIM, ACTION_DIM](
            ctx,
            states_buf,
            actions_buf,
            rewards_buf,
            dones_buf,
            terminated_buf,
            obs_buf,
            rng_seed=1,
            workspace_ptr=full_ws_buf.unsafe_ptr(),
        )
        dump(ctx, states_buf, "After step_kernel_gpu")
        print()

        # ===== TEST B: Step-by-step =====
        print("TEST B: Manual step-by-step (same operations)")
        print("-" * 70)
        ENV.reset_kernel_gpu[N_ENVS, STATE_SIZE](ctx, states_buf, rng_seed=0)
        dump(ctx, states_buf, "After reset")

        # B1. Pre-step
        ENV._pre_step_gpu[N_ENVS, STATE_SIZE](ctx, states_buf)
        dump(ctx, states_buf, "After pre_step")

        # B2. Apply actions
        InvertedPendulumModel.apply_actions_kernel_gpu[
            gpu_dtype, N_ENVS, STATE_SIZE, ACTION_DIM
        ](ctx, states_buf, actions_buf)
        dump(ctx, states_buf, "After apply_actions")

        # B3. Physics substep 1
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
        dump(ctx, states_buf, "After RK4 substep 1")

        # B4. Physics substep 2
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
            STEP_THREADS=NV,
        ](ctx, states_buf, model_buf, ws_buf2)
        dump(ctx, states_buf, "After RK4 substep 2")

        # B5. cfrc_ext + cvel
        compute_cfrc_ext_gpu[
            gpu_dtype,
            N_ENVS,
            STATE_SIZE,
            MODEL_SIZE,
            NQ,
            NV,
            NBODY,
            MAX_CONTACTS,
            NSITE,
        ](ctx, states_buf, model_buf)
        dump(ctx, states_buf, "After cfrc_ext")

        compute_cvel_gpu[
            gpu_dtype,
            N_ENVS,
            STATE_SIZE,
            NQ,
            NV,
            NBODY,
            MAX_CONTACTS,
            NSITE,
        ](ctx, states_buf)
        dump(ctx, states_buf, "After cvel")

        # B6. Extract obs/rewards/dones
        ENV._extract_obs_rewards_dones_gpu[
            N_ENVS,
            STATE_SIZE,
            MODEL_SIZE,
            OBS_DIM,
            1000,
        ](
            ctx,
            states_buf,
            model_buf,
            actions_buf,
            rewards_buf,
            dones_buf,
            terminated_buf,
            obs_buf,
        )
        dump(ctx, states_buf, "After extract")

    print()
    print(">>> Compare TEST A vs TEST B to find the divergence point <<<")
