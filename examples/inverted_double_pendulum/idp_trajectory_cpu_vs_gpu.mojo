"""Compare InvertedDoublePendulum trajectory: CPU (f32) vs GPU (f32).

Both run INDEPENDENTLY with their own observations driving action selection
(no sync between steps). This shows what each backend truly experiences,
matching what evaluate/evaluate_gpu do.

Run with:
    pixi run -e apple mojo run -I . examples/inverted_double_pendulum/idp_trajectory_cpu_vs_gpu.mojo
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
from mojo_rl.deep_agents.core.utils import obs_to_inline
from mojo_rl.nn.training import Network
from mojo_rl.nn.constants import dtype as nn_dtype
from mojo_rl.core.logger import NoOpLogger


# =============================================================================
# Constants
# =============================================================================

comptime OBS_DIM = 9
comptime ACTION_DIM = 1
comptime HIDDEN_DIM = 128
comptime BUFFER_CAPACITY = 1_000_000
comptime BATCH_SIZE = 256
comptime MAX_N_ENVS = 4
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
comptime MODEL_SIZE = model_size_with_invweight[
    NBODY, NJOINT, NV, NGEOM,
    NEQUALITY=InvertedDoublePendulumModel.MAX_EQUALITY,
    NTENDON=InvertedDoublePendulumModel.MAX_TENDON,
    NSITE=InvertedDoublePendulumModel.NSITE,
]()

comptime MAX_ENV_STEPS = 50
comptime POLE_LEN: Float64 = 0.6

comptime ActorModel = SACConfig[OBS_DIM, ACTION_DIM, HIDDEN_DIM].ActorModel
comptime ActorOpt = SACConfig[OBS_DIM, ACTION_DIM, HIDDEN_DIM].ActorOpt

comptime AgentType = DeepSACAgent[
    OBS_DIM, ACTION_DIM, HIDDEN_DIM, BUFFER_CAPACITY, BATCH_SIZE,
    0.0003, 0.001, 0, NoOpLogger, MAX_N_ENVS,
]


def get_cpu_obs(
    data: Data[DTYPE, NQ, NV, NBODY, NJOINT, MAX_CONTACTS, NSITE],
) -> InlineArray[Scalar[nn_dtype], OBS_DIM]:
    """9D custom obs from CPU data."""
    var obs = InlineArray[Scalar[nn_dtype], OBS_DIM](fill=Scalar[nn_dtype](0))
    obs[0] = Scalar[nn_dtype](data.qpos[0])
    obs[1] = Scalar[nn_dtype](sin(Float64(data.qpos[1])))
    obs[2] = Scalar[nn_dtype](sin(Float64(data.qpos[2])))
    obs[3] = Scalar[nn_dtype](cos(Float64(data.qpos[1])))
    obs[4] = Scalar[nn_dtype](cos(Float64(data.qpos[2])))
    for i in range(3):
        var v = Float64(data.qvel[i])
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
    print("=" * 100)
    print("InvertedDoublePendulum: CPU vs GPU Independent Trajectories")
    print("=" * 100)

    var agent = AgentType(
        gamma=0.99, tau=0.005, action_scale=1.0,
        alpha=0.2, auto_alpha=False, target_entropy=-1.0,
    )
    agent.load_checkpoint("sac_inverted_double_pendulum.ckpt")
    print("Loaded checkpoint")
    print()

    # === CPU setup ===
    var cpu_model = Model[
        DTYPE, NQ, NV, NBODY, NJOINT, MAX_CONTACTS, NGEOM,
        InvertedDoublePendulumModel.MAX_EQUALITY,
        InvertedDoublePendulumModel.CONE_TYPE,
        InvertedDoublePendulumModel.MAX_TENDON,
        InvertedDoublePendulumModel.NSITE,
    ]()
    var cpu_data = Data[DTYPE, NQ, NV, NBODY, NJOINT, MAX_CONTACTS, NSITE]()
    InvertedDoublePendulumModel.setup_model_and_data[DTYPE](cpu_model, cpu_data)

    # === GPU setup ===
    var ctx = DeviceContext()
    var gpu_state_host = create_state_buffer[
        DTYPE, NQ, NV, NBODY, MAX_CONTACTS, NSITE, GPU_BATCH
    ](ctx)
    var gpu_state_buf = ctx.enqueue_create_buffer[DTYPE](GPU_BATCH * STATE_SIZE)

    # Sync initial state CPU → GPU
    for i in range(GPU_BATCH * STATE_SIZE):
        gpu_state_host[i] = Scalar[DTYPE](0)
    for i in range(NQ):
        gpu_state_host[qpos_offset[NQ, NV]() + i] = cpu_data.qpos[i]
    for i in range(NV):
        gpu_state_host[qvel_offset[NQ, NV]() + i] = cpu_data.qvel[i]
    ctx.enqueue_copy(gpu_state_buf, gpu_state_host.unsafe_ptr())
    ctx.synchronize()

    # GPU env buffers
    var gpu_obs_buf = ctx.enqueue_create_buffer[DTYPE](GPU_BATCH * OBS_DIM)
    var gpu_obs_host = ctx.enqueue_create_host_buffer[DTYPE](GPU_BATCH * OBS_DIM)
    var gpu_rewards_buf = ctx.enqueue_create_buffer[DTYPE](GPU_BATCH)
    var gpu_dones_buf = ctx.enqueue_create_buffer[DTYPE](GPU_BATCH)
    var gpu_terminated_buf = ctx.enqueue_create_buffer[DTYPE](GPU_BATCH)
    var gpu_actions_buf = ctx.enqueue_create_buffer[DTYPE](GPU_BATCH * ACTION_DIM)
    var gpu_actions_host = ctx.enqueue_create_host_buffer[DTYPE](GPU_BATCH * ACTION_DIM)
    var gpu_rewards_host = ctx.enqueue_create_host_buffer[DTYPE](GPU_BATCH)
    var gpu_dones_host = ctx.enqueue_create_host_buffer[DTYPE](GPU_BATCH)

    comptime ENV_WS_SIZE = MODEL_SIZE + GPU_BATCH * (
        integrator_workspace_size[NV, NBODY]() + NV * NV
        + NewtonSolver.solver_workspace_size[NV, MAX_CONTACTS]()
        + rk4_extra_workspace_size[NQ, NV]()
    )
    var env_ws_buf = ctx.enqueue_create_buffer[DTYPE](ENV_WS_SIZE)
    InvertedDoublePendulum[DTYPE].init_step_workspace_gpu[GPU_BATCH](
        ctx, env_ws_buf
    )
    ctx.synchronize()

    # Initial GPU obs extraction
    InvertedDoublePendulum[DTYPE].extract_obs_kernel_gpu[
        GPU_BATCH, STATE_SIZE, OBS_DIM
    ](ctx, gpu_state_buf, gpu_obs_buf)
    ctx.synchronize()

    var cpu_total: Float64 = 0.0
    var gpu_total: Float64 = 0.0

    print(
        "Step | CPU_act  | GPU_act  | act_err  | CPU_rew  | GPU_rew "
        "| qpos_err         | obs_err          | cpu_done | gpu_done"
    )
    print("-" * 120)

    for step in range(MAX_ENV_STEPS):
        # === CPU: get obs → action ===
        var cpu_obs = get_cpu_obs(cpu_data)
        var cpu_action = get_greedy_action(agent, cpu_obs)

        # === GPU: get obs → action (using GPU-extracted obs) ===
        ctx.enqueue_copy(gpu_obs_host.unsafe_ptr(), gpu_obs_buf)
        ctx.synchronize()
        var gpu_obs = InlineArray[Scalar[nn_dtype], OBS_DIM](fill=Scalar[nn_dtype](0))
        for i in range(OBS_DIM):
            gpu_obs[i] = Scalar[nn_dtype](gpu_obs_host[i])
        var gpu_action = get_greedy_action(agent, gpu_obs)

        var act_err = abs(cpu_action - gpu_action)

        # === CPU step ===
        for i in range(NV):
            cpu_data.qfrc[i] = Scalar[DTYPE](0)
        var ctrl = cpu_action
        if ctrl > Float64(InvertedDoublePendulumModel._acd.motor_ctrl_max[0]):
            ctrl = Float64(InvertedDoublePendulumModel._acd.motor_ctrl_max[0])
        elif ctrl < Float64(InvertedDoublePendulumModel._acd.motor_ctrl_min[0]):
            ctrl = Float64(InvertedDoublePendulumModel._acd.motor_ctrl_min[0])
        cpu_data.qfrc[InvertedDoublePendulumModel._acd.motor_dof_adr[0]] = Scalar[DTYPE](
            InvertedDoublePendulumModel._acd.motor_gears[0] * ctrl
        )
        for _ in range(FRAME_SKIP):
            RK4Integrator[SOLVER=NewtonSolver].step[NGEOM=NGEOM](cpu_model, cpu_data)
        forward_kinematics(cpu_model, cpu_data)

        # CPU reward/done
        var q0 = Float64(cpu_data.qpos[0])
        var q1 = Float64(cpu_data.qpos[1])
        var q2 = Float64(cpu_data.qpos[2])
        var z_tip = POLE_LEN * cos(q1) + POLE_LEN * cos(q1 + q2)
        var cpu_terminated = z_tip <= 1.0
        var x_tip = q0 + POLE_LEN * sin(q1) + POLE_LEN * sin(q1 + q2)
        var dist_p = 0.01 * x_tip * x_tip + (z_tip - 2.0) * (z_tip - 2.0)
        var vel_p = 1e-3 * Float64(cpu_data.qvel[1]) ** 2 + 5e-3 * Float64(cpu_data.qvel[2]) ** 2
        var cpu_alive = 0.0 if cpu_terminated else 10.0
        var cpu_reward = cpu_alive - dist_p - vel_p
        cpu_total += cpu_reward

        # === GPU step ===
        gpu_actions_host[0] = Scalar[DTYPE](gpu_action)
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
        ctx.enqueue_copy(gpu_state_host.unsafe_ptr(), gpu_state_buf)
        ctx.enqueue_copy(gpu_obs_host.unsafe_ptr(), gpu_obs_buf)
        ctx.synchronize()

        var gpu_reward = Float64(gpu_rewards_host[0])
        var gpu_done = Float64(gpu_dones_host[0]) > 0.5
        gpu_total += gpu_reward

        # Compare qpos
        var qpos_err: Float64 = 0.0
        for i in range(NQ):
            var err = abs(
                Float64(cpu_data.qpos[i])
                - Float64(gpu_state_host[qpos_offset[NQ, NV]() + i])
            )
            if err > qpos_err:
                qpos_err = err

        # Compare obs
        var obs_err: Float64 = 0.0
        var cpu_obs_next = get_cpu_obs(cpu_data)
        for i in range(OBS_DIM):
            var err = abs(Float64(cpu_obs_next[i]) - Float64(gpu_obs_host[i]))
            if err > obs_err:
                obs_err = err

        print(
            String(step)[byte=:4]
            + " | "
            + String(cpu_action)[byte=:8]
            + " | "
            + String(gpu_action)[byte=:8]
            + " | "
            + String(act_err)[byte=:8]
            + " | "
            + String(cpu_reward)[byte=:8]
            + " | "
            + String(gpu_reward)[byte=:7]
            + " | "
            + String(qpos_err)[byte=:16]
            + " | "
            + String(obs_err)[byte=:16]
            + " | "
            + String(cpu_terminated)
            + "    | "
            + String(gpu_done)
        )

        # Dump obs if large mismatch
        if obs_err > 0.01 or act_err > 0.01:
            print("    CPU obs:", end="")
            for i in range(OBS_DIM):
                print(" " + String(Float64(cpu_obs_next[i]))[byte=:10], end="")
            print()
            print("    GPU obs:", end="")
            for i in range(OBS_DIM):
                print(" " + String(Float64(gpu_obs_host[i]))[byte=:10], end="")
            print()

        if cpu_terminated or gpu_done:
            if cpu_terminated:
                print(">>> CPU terminated at step " + String(step) + " <<<")
            if gpu_done:
                print(">>> GPU terminated at step " + String(step) + " <<<")
            break

    print("-" * 120)
    print("CPU total reward: " + String(cpu_total)[byte=:10])
    print("GPU total reward: " + String(gpu_total)[byte=:10])
