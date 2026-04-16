"""IDP Diagnostic: CPU step vs step_kernel_gpu — full pipeline comparison.

Uses step_kernel_gpu (the same code path as evaluate_gpu) with 1 env,
feeding the same action each step. Compares obs, rewards, dones.

This isolates whether the issue is in step_kernel_gpu's action application,
obs extraction, reward computation, or physics — vs the raw RK4 GPU path.

Run with:
    pixi run -e apple mojo run -I . examples/inverted_double_pendulum/idp_step_kernel_diagnostic.mojo
    pixi run -e nvidia mojo run -I . examples/inverted_double_pendulum/idp_step_kernel_diagnostic.mojo
"""

from std.random import seed
from std.math import abs, sin, cos, tanh
from std.gpu.host import DeviceContext, DeviceBuffer, HostBuffer
from layout import Layout, LayoutTensor

from mojo_rl.physics3d.types import Model, Data
from mojo_rl.physics3d.kinematics import forward_kinematics
from mojo_rl.physics3d.solver import NewtonSolver
from mojo_rl.physics3d.integrator.rk4_integrator import RK4Integrator
from mojo_rl.physics3d.gpu.constants import (
    state_size,
    model_size_with_invweight,
    integrator_workspace_size,
    rk4_extra_workspace_size,
    qpos_offset,
    qvel_offset,
    qfrc_offset,
    metadata_offset,
    META_IDX_STEP_COUNT,
    META_IDX_PREV_X,
)
from mojo_rl.physics3d.gpu.buffer_utils import create_state_buffer
from mojo_rl.envs.inverted_double_pendulum import InvertedDoublePendulum
from mojo_rl.envs.inverted_double_pendulum.inverted_double_pendulum_xml import (
    InvertedDoublePendulumModel,
)
from mojo_rl.envs.inverted_double_pendulum.inverted_double_pendulum_config import (
    InvertedDoublePendulumConfig,
)
from mojo_rl.deep_agents.core.agents import DeepSACAgent
from mojo_rl.deep_agents.core.configs.offpolicy_config import SACConfig
from mojo_rl.nn.training import Network
from mojo_rl.nn.constants import dtype as nn_dtype
from mojo_rl.core.logger import NoOpLogger
from mojo_rl.core.cont_action import ContAction


# =============================================================================
# Constants
# =============================================================================

comptime OBS_DIM = 9
comptime ACTION_DIM = 1
comptime HIDDEN_DIM = 128
comptime BUFFER_CAPACITY = 1_000_000
comptime BATCH_SIZE = 256
comptime MAX_N_ENVS = 1
comptime NQ = InvertedDoublePendulumModel.NQ
comptime NV = InvertedDoublePendulumModel.NV
comptime NBODY = InvertedDoublePendulumModel.NBODY
comptime NJOINT = InvertedDoublePendulumModel.NJOINT
comptime NGEOM = InvertedDoublePendulumModel.NGEOM
comptime NSITE = InvertedDoublePendulumModel.NSITE
comptime MAX_CONTACTS = InvertedDoublePendulumModel.MAX_CONTACTS
comptime FRAME_SKIP = 5

comptime DTYPE = DType.float32
comptime GPU_BATCH = 1
comptime STATE_SIZE = state_size[NQ, NV, NBODY, MAX_CONTACTS, NSITE]()

comptime MAX_ENV_STEPS = 200
comptime POLE_LEN: Float64 = 0.6

comptime ActorModel = SACConfig[OBS_DIM, ACTION_DIM, HIDDEN_DIM].ActorModel
comptime ActorOpt = SACConfig[OBS_DIM, ACTION_DIM, HIDDEN_DIM].ActorOpt

comptime AgentType = DeepSACAgent[
    OBS_DIM, ACTION_DIM, HIDDEN_DIM, BUFFER_CAPACITY, BATCH_SIZE,
    0.0003, 0.001, 0, NoOpLogger, MAX_N_ENVS,
]


def get_cpu_obs_from_env(
    cpu_env: InvertedDoublePendulum[DTYPE, TERMINATE_ON_UNHEALTHY=True],
) -> InlineArray[Scalar[nn_dtype], OBS_DIM]:
    """9D custom obs from CPU env accessors."""
    var obs = InlineArray[Scalar[nn_dtype], OBS_DIM](fill=Scalar[nn_dtype](0))
    var q0 = Float64(cpu_env.get_qpos(0))
    var q1 = Float64(cpu_env.get_qpos(1))
    var q2 = Float64(cpu_env.get_qpos(2))
    obs[0] = Scalar[nn_dtype](q0)
    obs[1] = Scalar[nn_dtype](sin(q1))
    obs[2] = Scalar[nn_dtype](sin(q2))
    obs[3] = Scalar[nn_dtype](cos(q1))
    obs[4] = Scalar[nn_dtype](cos(q2))
    for i in range(3):
        var v = Float64(cpu_env.get_qvel(i))
        if v > 10.0:
            v = 10.0
        elif v < -10.0:
            v = -10.0
        obs[5 + i] = Scalar[nn_dtype](v)
    obs[8] = Scalar[nn_dtype](0.0)
    return obs^


def get_greedy_action(
    agent: AgentType,
    obs: InlineArray[Scalar[nn_dtype], OBS_DIM],
) -> Float64:
    """Deterministic action: tanh(mean) from actor network."""
    var obs_local = obs
    var obs_t = LayoutTensor[
        nn_dtype, Layout.row_major(1, OBS_DIM), MutAnyOrigin
    ](obs_local.unsafe_ptr())
    comptime ACTOR_OUT = ActorModel.OUT_DIM
    var out_arr = InlineArray[Scalar[nn_dtype], ACTOR_OUT](uninitialized=True)
    var out_t = LayoutTensor[
        nn_dtype, Layout.row_major(1, ACTOR_OUT), MutAnyOrigin
    ](out_arr.unsafe_ptr())
    comptime PS = ActorModel.PARAM_SIZE
    var p = LayoutTensor[nn_dtype, Layout.row_major(PS), MutAnyOrigin](
        agent.state.actor.online.params
    )
    Network[ActorModel, ActorOpt].forward[1](obs_t, out_t, p)
    return tanh(Float64(out_arr[0])) * agent.action_scale


def main() raises:
    seed(42)
    print("=" * 110)
    print("IDP Diagnostic: CPU step vs step_kernel_gpu (full pipeline)")
    print("Same action each step — compares obs, rewards, dones")
    print("=" * 110)

    var agent = AgentType(
        gamma=0.99, tau=0.005, action_scale=1.0,
        alpha=0.2, auto_alpha=False, target_entropy=-1.0,
    )
    agent.load_checkpoint("sac_inverted_double_pendulum.ckpt")
    print("Loaded checkpoint")

    # === CPU env (f32) ===
    var cpu_env = InvertedDoublePendulum[DTYPE, TERMINATE_ON_UNHEALTHY=True]()
    _ = cpu_env.reset()

    # === GPU env via step_kernel_gpu ===
    var ctx = DeviceContext()

    # GPU buffers
    var gpu_state_buf = ctx.enqueue_create_buffer[DTYPE](GPU_BATCH * STATE_SIZE)
    var gpu_obs_buf = ctx.enqueue_create_buffer[DTYPE](GPU_BATCH * OBS_DIM)
    var gpu_obs_host = ctx.enqueue_create_host_buffer[DTYPE](GPU_BATCH * OBS_DIM)
    var gpu_rewards_buf = ctx.enqueue_create_buffer[DTYPE](GPU_BATCH)
    var gpu_dones_buf = ctx.enqueue_create_buffer[DTYPE](GPU_BATCH)
    var gpu_terminated_buf = ctx.enqueue_create_buffer[DTYPE](GPU_BATCH)
    var gpu_actions_buf = ctx.enqueue_create_buffer[DTYPE](GPU_BATCH * ACTION_DIM)
    var gpu_actions_host = ctx.enqueue_create_host_buffer[DTYPE](GPU_BATCH * ACTION_DIM)
    var gpu_rewards_host = ctx.enqueue_create_host_buffer[DTYPE](GPU_BATCH)
    var gpu_dones_host = ctx.enqueue_create_host_buffer[DTYPE](GPU_BATCH)
    var gpu_state_host = create_state_buffer[
        DTYPE, NQ, NV, NBODY, MAX_CONTACTS, NSITE, GPU_BATCH
    ](ctx)

    # Step workspace (model + per-env workspace)
    comptime ENV_WS_SHARED = InvertedDoublePendulum[DTYPE].STEP_WS_SHARED
    comptime ENV_WS_PER = InvertedDoublePendulum[DTYPE].STEP_WS_PER_ENV
    comptime TOTAL_WS = ENV_WS_SHARED + GPU_BATCH * ENV_WS_PER
    var env_ws_buf = ctx.enqueue_create_buffer[DTYPE](TOTAL_WS)
    InvertedDoublePendulum[DTYPE].init_step_workspace_gpu[GPU_BATCH](ctx, env_ws_buf)
    ctx.synchronize()

    # Sync CPU initial state → GPU (instead of using reset_kernel_gpu)
    # This ensures both start from the EXACT same state
    for i in range(GPU_BATCH * STATE_SIZE):
        gpu_state_host[i] = Scalar[DTYPE](0)
    for i in range(NQ):
        gpu_state_host[qpos_offset[NQ, NV]() + i] = cpu_env.get_qpos(i)
    for i in range(NV):
        gpu_state_host[qvel_offset[NQ, NV]() + i] = cpu_env.get_qvel(i)
    # Set step counter to 0 and prev_x
    comptime META_OFF = metadata_offset[NQ, NV, NBODY, MAX_CONTACTS]()
    gpu_state_host[META_OFF + META_IDX_STEP_COUNT] = Scalar[DTYPE](0)
    gpu_state_host[META_OFF + META_IDX_PREV_X] = cpu_env.get_qpos(0)
    ctx.enqueue_copy(gpu_state_buf, gpu_state_host.unsafe_ptr())
    ctx.synchronize()

    # Extract initial GPU obs via the kernel (tests obs extraction)
    InvertedDoublePendulum[DTYPE].extract_obs_kernel_gpu[
        GPU_BATCH, STATE_SIZE, OBS_DIM
    ](ctx, gpu_state_buf, gpu_obs_buf)
    ctx.synchronize()

    # Read back to compare
    ctx.enqueue_copy(gpu_state_host.unsafe_ptr(), gpu_state_buf)
    ctx.enqueue_copy(gpu_obs_host.unsafe_ptr(), gpu_obs_buf)
    ctx.synchronize()

    print()
    print("Initial CPU qpos:", end="")
    for i in range(NQ):
        print(" " + String(Float64(cpu_env.get_qpos(i)))[byte=:12], end="")
    print()
    print("Initial GPU qpos:", end="")
    for i in range(NQ):
        print(" " + String(Float64(gpu_state_host[qpos_offset[NQ, NV]() + i]))[byte=:12], end="")
    print()

    # Compare initial obs
    print("Initial CPU obs:", end="")
    var init_cpu_obs = get_cpu_obs_from_env(cpu_env)
    for i in range(OBS_DIM):
        print(" " + String(Float64(init_cpu_obs[i]))[byte=:10], end="")
    print()
    print("Initial GPU obs:", end="")
    for i in range(OBS_DIM):
        print(" " + String(Float64(gpu_obs_host[i]))[byte=:10], end="")
    print()

    # Check initial obs match
    var init_obs_err: Float64 = 0.0
    for i in range(OBS_DIM):
        var err = abs(Float64(init_cpu_obs[i]) - Float64(gpu_obs_host[i]))
        if err > init_obs_err:
            init_obs_err = err
    print("Initial obs max error: " + String(init_obs_err))
    print()

    var cpu_total: Float64 = 0.0
    var gpu_total: Float64 = 0.0

    print(
        "Step | action   | cpu_rew  | gpu_rew  | rew_err  "
        "| obs_err          | cpu_done | gpu_done"
    )
    print("-" * 110)

    for step in range(MAX_ENV_STEPS):
        # Get CPU obs and compute action
        var cpu_obs = get_cpu_obs_from_env(cpu_env)
        var action = get_greedy_action(agent, cpu_obs)

        # === CPU step ===
        var act_data = InlineArray[Float64, InvertedDoublePendulumModel.ACTION_DIM](fill=action)
        var cont_action = ContAction[InvertedDoublePendulumModel.ACTION_DIM](act_data)
        var cpu_step_result = cpu_env.step(cont_action)
        var cpu_reward = Float64(cpu_step_result[1])
        var cpu_done = cpu_step_result[2]
        cpu_total += cpu_reward

        # === GPU step via step_kernel_gpu (same action) ===
        gpu_actions_host[0] = Scalar[DTYPE](action)
        ctx.enqueue_copy(gpu_actions_buf, gpu_actions_host.unsafe_ptr())

        InvertedDoublePendulum[DTYPE].step_kernel_gpu[
            GPU_BATCH, STATE_SIZE, OBS_DIM, ACTION_DIM,
        ](
            ctx, gpu_state_buf, gpu_actions_buf, gpu_rewards_buf,
            gpu_dones_buf, gpu_terminated_buf, gpu_obs_buf,
            rng_seed=UInt64(step + 1),
            workspace_ptr=env_ws_buf.unsafe_ptr(),
        )
        ctx.synchronize()

        ctx.enqueue_copy(gpu_rewards_host.unsafe_ptr(), gpu_rewards_buf)
        ctx.enqueue_copy(gpu_dones_host.unsafe_ptr(), gpu_dones_buf)
        ctx.enqueue_copy(gpu_obs_host.unsafe_ptr(), gpu_obs_buf)
        ctx.enqueue_copy(gpu_state_host.unsafe_ptr(), gpu_state_buf)
        ctx.synchronize()

        var gpu_reward = Float64(gpu_rewards_host[0])
        var gpu_done = Float64(gpu_dones_host[0]) > 0.5
        gpu_total += gpu_reward

        # Compare obs
        var cpu_obs_next = get_cpu_obs_from_env(cpu_env)
        var obs_err: Float64 = 0.0
        for i in range(OBS_DIM):
            var err = abs(Float64(cpu_obs_next[i]) - Float64(gpu_obs_host[i]))
            if err > obs_err:
                obs_err = err

        var rew_err = abs(cpu_reward - gpu_reward)

        var should_print = step % 5 == 0 or obs_err > 0.01 or rew_err > 0.1
        if should_print:
            print(
                String(step)[byte=:4]
                + " | "
                + String(action)[byte=:8]
                + " | "
                + String(cpu_reward)[byte=:8]
                + " | "
                + String(gpu_reward)[byte=:8]
                + " | "
                + String(rew_err)[byte=:8]
                + " | "
                + String(obs_err)[byte=:16]
                + " | "
                + String(cpu_done)
                + "    | "
                + String(gpu_done)
            )

        if obs_err > 0.01 or rew_err > 0.5:
            print("  >>> DIVERGENCE at step " + String(step) + " <<<")
            print("    CPU obs:", end="")
            for i in range(OBS_DIM):
                print(" " + String(Float64(cpu_obs_next[i]))[byte=:12], end="")
            print()
            print("    GPU obs:", end="")
            for i in range(OBS_DIM):
                print(" " + String(Float64(gpu_obs_host[i]))[byte=:12], end="")
            print()
            # Also dump qpos
            print("    CPU qpos:", end="")
            for i in range(NQ):
                print(" " + String(Float64(cpu_env.get_qpos(i)))[byte=:14], end="")
            print()
            print("    GPU qpos:", end="")
            for i in range(NQ):
                print(
                    " " + String(Float64(gpu_state_host[qpos_offset[NQ, NV]() + i]))[byte=:14],
                    end="",
                )
            print()

        if cpu_done or gpu_done:
            if cpu_done:
                print(">>> CPU done at step " + String(step) + " <<<")
            if gpu_done:
                print(">>> GPU done at step " + String(step) + " <<<")
            break

    print("-" * 110)
    print("CPU total reward: " + String(cpu_total)[byte=:12])
    print("GPU total reward: " + String(gpu_total)[byte=:12])
    print("Reward gap:       " + String(abs(cpu_total - gpu_total))[byte=:12])
    print()
    if abs(cpu_total - gpu_total) < 5.0:
        print("MATCH: step_kernel_gpu produces same results as CPU step")
    else:
        print(
            "DIVERGED: step_kernel_gpu differs from CPU — issue is in"
            " action application, obs extraction, reward, or physics"
        )

    cpu_env.close()
