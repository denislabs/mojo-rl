"""Compare Hopper trajectory: CPU (f64) vs GPU (f32) vs MuJoCo step-by-step.

Loads a checkpoint, runs one episode with deterministic actions, and at each
env step compares qpos/qvel across all three backends driven by the SAME actions
from the SAME initial state.

Run with:
    pixi run -e apple mojo run -I . examples/hopper/sac_hopper_trajectory_compare.mojo
"""

from std.random import seed
from std.math import abs
from std.python import Python, PythonObject
from std.gpu.host import DeviceContext, DeviceBuffer, HostBuffer

from mojo_rl.deep_agents.core.agents import DeepSACAgent
from mojo_rl.core.logger import NoOpLogger
from mojo_rl.envs.hopper import Hopper, HopperConfig
from mojo_rl.envs.hopper.hopper_xml import HopperModel
from mojo_rl.physics3d.types import Model, Data
from mojo_rl.physics3d.kinematics import forward_kinematics
from mojo_rl.physics3d.solver import NewtonSolver
from mojo_rl.physics3d.gpu.constants import (
    state_size,
    model_size_with_invweight,
    integrator_workspace_size,
    rk4_extra_workspace_size,
    qpos_offset,
    qvel_offset,
    qfrc_offset,
    qacc_offset,
)
from mojo_rl.physics3d.gpu.buffer_utils import create_state_buffer
from mojo_rl.physics3d.integrator.rk4_integrator import RK4Integrator
from mojo_rl.physics3d.solver import NewtonSolver

from layout import Layout, LayoutTensor
from std.math import tanh
from mojo_rl.nn.constants import dtype as nn_dtype
from mojo_rl.deep_agents.core.configs.offpolicy_config import SACConfig
from mojo_rl.deep_agents.core.utils import obs_to_inline
from mojo_rl.nn.training import Network


# =============================================================================
# Constants
# =============================================================================

comptime OBS_DIM = HopperConfig.OBS_DIM
comptime ACTION_DIM = HopperConfig.ACTION_DIM
comptime HIDDEN_DIM = 256
comptime BUFFER_CAPACITY = 1_000_000
comptime BATCH_SIZE = 256
comptime MAX_N_ENVS = 4
comptime NQ = HopperModel.NQ
comptime NV = HopperModel.NV
comptime NBODY = HopperModel.NBODY
comptime NJOINT = HopperModel.NJOINT
comptime NGEOM = HopperModel.NGEOM
comptime NSITE = HopperModel.NSITE
comptime FRAME_SKIP = HopperConfig.FRAME_SKIP  # 4

comptime MAX_STEPS = 500

# GPU constants
comptime GPU_DTYPE = DType.float32
comptime GPU_BATCH = 1
comptime MAX_CONTACTS = HopperConfig.MAX_CONTACTS
comptime STATE_SIZE = state_size[NQ, NV, NBODY, MAX_CONTACTS, NSITE]()
comptime MODEL_SIZE = model_size_with_invweight[NBODY, NJOINT, NV, NGEOM]()
comptime SOLVER_WS = NewtonSolver.solver_workspace_size[NV, MAX_CONTACTS]()
comptime WS_SIZE = integrator_workspace_size[
    NV, NBODY
]() + NV * NV + SOLVER_WS + rk4_extra_workspace_size[NQ, NV]()

comptime ActorModel = SACConfig[OBS_DIM, ACTION_DIM].ActorModel
comptime ActorOpt = SACConfig[OBS_DIM, ACTION_DIM].ActorOpt

comptime AgentType = DeepSACAgent[
    OBS_DIM,
    ACTION_DIM,
    HIDDEN_DIM,
    BUFFER_CAPACITY,
    BATCH_SIZE,
    0.0003,
    0.0003,
    0,
    NoOpLogger,
    MAX_N_ENVS,
]


def _get_greedy_action(
    agent: AgentType,
    obs: List[Float64],
) -> List[Float64]:
    """Get deterministic action: tanh(mean) from actor network."""
    var obs_arr = obs_to_inline[OBS_DIM, DType.float64](obs)
    var obs_f32 = InlineArray[Scalar[nn_dtype], OBS_DIM](uninitialized=True)
    for i in range(OBS_DIM):
        obs_f32[i] = Scalar[nn_dtype](obs_arr[i])
    var obs_t = LayoutTensor[
        nn_dtype, Layout.row_major(1, OBS_DIM), MutAnyOrigin
    ](obs_f32.unsafe_ptr())

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

    var result = List[Float64](capacity=ACTION_DIM)
    for i in range(ACTION_DIM):
        var mean = Float64(out_arr[i])
        var a = tanh(mean) * agent.action_scale
        result.append(a)
    return result^


def main() raises:
    seed(42)
    print("=" * 70)
    print("Hopper Trajectory: CPU (f64) vs GPU (f32) vs MuJoCo")
    print("=" * 70)

    # Load trained agent
    var agent = AgentType(
        gamma=0.99,
        tau=0.005,
        action_scale=1.0,
        alpha=0.2,
        auto_alpha=False,
        target_entropy=-3.0,
    )
    agent.load_checkpoint("sac_hopper_400.ckpt")
    print("Loaded checkpoint: sac_hopper_400.ckpt")

    # === CPU environment (f64) ===
    var env = Hopper[DType.float64, TERMINATE_ON_UNHEALTHY=True]()
    _ = env.reset()

    # === GPU environment (f32) — single env ===
    var ctx = DeviceContext()
    var gpu_state_host = create_state_buffer[
        GPU_DTYPE, NQ, NV, NBODY, MAX_CONTACTS, NSITE, GPU_BATCH
    ](ctx)
    var gpu_state_buf = ctx.enqueue_create_buffer[GPU_DTYPE](
        GPU_BATCH * STATE_SIZE
    )
    var gpu_model_buf = ctx.enqueue_create_buffer[GPU_DTYPE](MODEL_SIZE)
    HopperModel.init_model_gpu(ctx, gpu_model_buf)
    var gpu_ws_buf = ctx.enqueue_create_buffer[GPU_DTYPE](GPU_BATCH * WS_SIZE)
    var gpu_actions_host = ctx.enqueue_create_host_buffer[GPU_DTYPE](
        GPU_BATCH * ACTION_DIM
    )
    var gpu_actions_buf = ctx.enqueue_create_buffer[GPU_DTYPE](
        GPU_BATCH * ACTION_DIM
    )
    ctx.synchronize()

    # Sync initial state from CPU env to GPU buffer
    for i in range(GPU_BATCH * STATE_SIZE):
        gpu_state_host[i] = Scalar[GPU_DTYPE](0)
    for i in range(NQ):
        gpu_state_host[qpos_offset[NQ, NV]() + i] = Scalar[GPU_DTYPE](
            Float64(env.get_qpos(i))
        )
    for i in range(NV):
        gpu_state_host[qvel_offset[NQ, NV]() + i] = Scalar[GPU_DTYPE](
            Float64(env.get_qvel(i))
        )
    ctx.enqueue_copy(gpu_state_buf, gpu_state_host.unsafe_ptr())
    ctx.synchronize()

    # === MuJoCo environment ===
    var mujoco = Python.import_module("mujoco")
    var xml_path = (
        "./references/Gymnasium-main/gymnasium/envs/mujoco/assets/hopper.xml"
    )
    var mj_model = mujoco.MjModel.from_xml_path(xml_path)
    mj_model.opt.integrator = 1  # RK4
    mj_model.opt.solver = 2  # Newton
    var mj_data = mujoco.MjData(mj_model)

    # Sync initial state to MuJoCo
    for i in range(NQ):
        mj_data.qpos[i] = env.get_qpos(i)
    for i in range(NV):
        mj_data.qvel[i] = env.get_qvel(i)
    mujoco.mj_forward(mj_model, mj_data)

    print()
    print(
        "Step | CPU_Rew  | MJ_Rew   | CPU-MJ_qpos | CPU-MJ_qvel"
        " | CPU-GPU_qpos | CPU-GPU_qvel | GPU_z  | GPU_angle"
    )
    print("-" * 120)

    var cpu_total: Float64 = 0.0
    var mj_total: Float64 = 0.0
    var gpu_total: Float64 = 0.0

    for step in range(MAX_STEPS):
        # --- Get obs from CPU env, select action ---
        var obs_raw = List[Scalar[DType.float64]](capacity=OBS_DIM)
        for i in range(OBS_DIM):
            obs_raw.append(Scalar[DType.float64](0))
        for i in range(5):
            obs_raw[i] = Scalar[DType.float64](env.get_qpos(i + 1))
        for i in range(6):
            var v = env.get_qvel(i)
            if Float64(v) > 10.0:
                v = Scalar[DType.float64](10.0)
            elif Float64(v) < -10.0:
                v = Scalar[DType.float64](-10.0)
            obs_raw[5 + i] = v
        var obs_f64 = List[Float64](capacity=OBS_DIM)
        for i in range(OBS_DIM):
            obs_f64.append(Float64(obs_raw[i]))

        var action = _get_greedy_action(agent, obs_f64)

        # === Step CPU env ===
        var result = env.step_continuous_vec(action)
        var cpu_reward = Float64(result[1])
        var cpu_done = result[2]
        cpu_total += cpu_reward

        # === Step GPU env (manual: apply actions + frame_skip RK4 steps) ===
        # Read back state to host, apply actions to qfrc, copy back
        ctx.enqueue_copy(gpu_state_host.unsafe_ptr(), gpu_state_buf)
        ctx.synchronize()

        # Apply per-motor clamped actions to qfrc (matching apply_actions)
        for i in range(ACTION_DIM):
            var ctrl = action[i]
            if ctrl > HopperModel._acd.motor_ctrl_max[i]:
                ctrl = HopperModel._acd.motor_ctrl_max[i]
            elif ctrl < HopperModel._acd.motor_ctrl_min[i]:
                ctrl = HopperModel._acd.motor_ctrl_min[i]
            var dof = HopperModel._acd.motor_dof_adr[i]
            gpu_state_host[qfrc_offset[NQ, NV]() + dof] = Scalar[GPU_DTYPE](
                HopperModel._acd.motor_gears[i] * ctrl
            )
        ctx.enqueue_copy(gpu_state_buf, gpu_state_host.unsafe_ptr())
        ctx.synchronize()

        # Run FRAME_SKIP RK4 steps (matching hopper_config.physics_substep_gpu)
        for _ in range(FRAME_SKIP):
            RK4Integrator[SOLVER=NewtonSolver].step_gpu[
                GPU_DTYPE,
                NQ,
                NV,
                NBODY,
                NJOINT,
                MAX_CONTACTS,
                GPU_BATCH,
                NGEOM=NGEOM,
                CONE_TYPE=HopperModel.CONE_TYPE,
                STEP_THREADS=NV,
            ](ctx, gpu_state_buf, gpu_model_buf, gpu_ws_buf)
        ctx.synchronize()

        # Read back GPU state
        ctx.enqueue_copy(gpu_state_host.unsafe_ptr(), gpu_state_buf)
        ctx.synchronize()

        # === Step MuJoCo ===
        for i in range(ACTION_DIM):
            mj_data.ctrl[i] = action[i]
        for _ in range(FRAME_SKIP):
            mujoco.mj_step(mj_model, mj_data)

        # MuJoCo reward
        var mj_xvel = Float64(py=mj_data.qvel[0])
        var mj_z = Float64(py=mj_data.qpos[1])
        var mj_angle = Float64(py=mj_data.qpos[2])
        var mj_healthy = True
        if mj_z < 0.7 or mj_angle > 0.2 or mj_angle < -0.2:
            mj_healthy = False
        for k in range(2, NQ):
            var qp = Float64(py=mj_data.qpos[k])
            if qp <= -100.0 or qp >= 100.0:
                mj_healthy = False
        for k in range(NV):
            var qv = Float64(py=mj_data.qvel[k])
            if qv <= -100.0 or qv >= 100.0:
                mj_healthy = False
        var mj_healthy_reward: Float64 = 1.0 if mj_healthy else 0.0
        var mj_ctrl_cost: Float64 = 0.0
        for i in range(ACTION_DIM):
            mj_ctrl_cost += action[i] * action[i]
        mj_ctrl_cost *= 0.001
        var mj_reward = mj_xvel + mj_healthy_reward - mj_ctrl_cost
        mj_total += mj_reward
        var mj_terminated = not mj_healthy

        # === Compare ===
        var cpu_mj_qpos_err: Float64 = 0.0
        var cpu_mj_qvel_err: Float64 = 0.0
        var cpu_gpu_qpos_err: Float64 = 0.0
        var cpu_gpu_qvel_err: Float64 = 0.0

        for i in range(NQ):
            var cpu_q = Float64(env.get_qpos(i))
            var mj_q = Float64(py=mj_data.qpos[i])
            var gpu_q = Float64(gpu_state_host[qpos_offset[NQ, NV]() + i])
            var err_mj = abs(cpu_q - mj_q)
            var err_gpu = abs(cpu_q - gpu_q)
            if err_mj > cpu_mj_qpos_err:
                cpu_mj_qpos_err = err_mj
            if err_gpu > cpu_gpu_qpos_err:
                cpu_gpu_qpos_err = err_gpu

        for i in range(NV):
            var cpu_v = Float64(env.get_qvel(i))
            var mj_v = Float64(py=mj_data.qvel[i])
            var gpu_v = Float64(gpu_state_host[qvel_offset[NQ, NV]() + i])
            var err_mj = abs(cpu_v - mj_v)
            var err_gpu = abs(cpu_v - gpu_v)
            if err_mj > cpu_mj_qvel_err:
                cpu_mj_qvel_err = err_mj
            if err_gpu > cpu_gpu_qvel_err:
                cpu_gpu_qvel_err = err_gpu

        var gpu_z = Float64(gpu_state_host[qpos_offset[NQ, NV]() + 1])
        var gpu_angle = Float64(gpu_state_host[qpos_offset[NQ, NV]() + 2])
        _ = Float64(env.get_qpos(1))
        _ = Float64(env.get_qpos(2))

        # Print every 5 steps or when GPU error is large
        if step % 5 == 0 or cpu_gpu_qpos_err > 0.01 or cpu_gpu_qvel_err > 0.1:
            print(
                String(step)[byte=:4]
                + " | "
                + String(cpu_reward)[byte=:8]
                + " | "
                + String(mj_reward)[byte=:8]
                + " | "
                + String(cpu_mj_qpos_err)[byte=:24]
                + " | "
                + String(cpu_mj_qvel_err)[byte=:24]
                + " | "
                + String(cpu_gpu_qpos_err)[byte=:24]
                + " | "
                + String(cpu_gpu_qvel_err)[byte=:24]
                + " | "
                + String(gpu_z)[byte=:6]
                + " | "
                + String(gpu_angle)[byte=:9]
            )

        if cpu_done:
            print("  >>> CPU env terminated at step " + String(step) + " <<<")
            break

        if mj_terminated:
            print(
                "  >>> MuJoCo terminated at step "
                + String(step)
                + " (z="
                + String(mj_z)[byte=:6]
                + " angle="
                + String(mj_angle)[byte=:6]
                + ") <<<"
            )
            break

    print("-" * 120)
    print("CPU total reward:    " + String(cpu_total)[byte=:10])
    print("MuJoCo total reward: " + String(mj_total)[byte=:10])
    print()

    env.close()
