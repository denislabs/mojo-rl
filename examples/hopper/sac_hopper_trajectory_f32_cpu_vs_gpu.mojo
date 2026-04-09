"""Compare Hopper trajectory: CPU (f32) vs GPU (f32) — same dtype, same actions.

If CPU f32 and GPU f32 match, the divergence in the f64 vs f32 comparison
is purely from float32 precision, not a GPU-specific bug.

Run with:
    pixi run -e apple mojo run -I . examples/hopper/sac_hopper_trajectory_f32_cpu_vs_gpu.mojo
"""

from std.random import seed
from std.math import abs, tanh
from std.python import Python
from std.gpu.host import DeviceContext, DeviceBuffer, HostBuffer

from mojo_rl.deep_agents.core.agents import DeepSACAgent
from mojo_rl.core.logger import NoOpLogger
from mojo_rl.envs.hopper import Hopper, HopperConfig
from mojo_rl.envs.hopper.hopper_xml import HopperModel
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

from layout import Layout, LayoutTensor
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
comptime FRAME_SKIP = HopperConfig.FRAME_SKIP

comptime MAX_STEPS = 1000

# Both CPU and GPU use float32
comptime DTYPE = DType.float32
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
    print("Hopper Trajectory: CPU (f32) vs GPU (f32)")
    print("Both use the SAME dtype — proves GPU divergence is (or isn't)")
    print("from float32 precision vs a GPU-specific bug.")
    print("=" * 70)

    var agent = AgentType(
        gamma=0.99,
        tau=0.005,
        action_scale=1.0,
        alpha=0.2,
        auto_alpha=False,
        target_entropy=-3.0,
    )
    agent.load_checkpoint("sac_hopper_1000.ckpt")
    print("Loaded checkpoint: sac_hopper_1000.ckpt")

    # === CPU environment (f32!) ===
    var cpu_env = Hopper[DTYPE, TERMINATE_ON_UNHEALTHY=True]()
    _ = cpu_env.reset()

    # === CPU physics objects (f32) for manual stepping ===
    var cpu_model = Model[
        DTYPE,
        NQ,
        NV,
        NBODY,
        NJOINT,
        MAX_CONTACTS,
        NGEOM,
        HopperModel.MAX_EQUALITY,
        HopperModel.CONE_TYPE,
        HopperModel.MAX_TENDON,
        HopperModel.NSITE,
    ]()
    var cpu_data = Data[
        DTYPE, NQ, NV, NBODY, NJOINT, MAX_CONTACTS, HopperModel.NSITE
    ]()
    HopperModel.setup_model_and_data(cpu_model, cpu_data)

    # Copy initial state from env to manual CPU data
    for i in range(NQ):
        cpu_data.qpos[i] = cpu_env.get_qpos(i)
    for i in range(NV):
        cpu_data.qvel[i] = cpu_env.get_qvel(i)
    forward_kinematics(cpu_model, cpu_data)

    # === GPU setup (f32) ===
    var ctx = DeviceContext()
    var gpu_state_host = create_state_buffer[
        DTYPE, NQ, NV, NBODY, MAX_CONTACTS, NSITE, GPU_BATCH
    ](ctx)
    var gpu_state_buf = ctx.enqueue_create_buffer[DTYPE](GPU_BATCH * STATE_SIZE)
    var gpu_model_buf = ctx.enqueue_create_buffer[DTYPE](MODEL_SIZE)
    HopperModel.init_model_gpu(ctx, gpu_model_buf)
    var gpu_ws_buf = ctx.enqueue_create_buffer[DTYPE](GPU_BATCH * WS_SIZE)
    ctx.synchronize()

    # Sync initial state to GPU
    for i in range(GPU_BATCH * STATE_SIZE):
        gpu_state_host[i] = Scalar[DTYPE](0)
    for i in range(NQ):
        gpu_state_host[qpos_offset[NQ, NV]() + i] = cpu_data.qpos[i]
    for i in range(NV):
        gpu_state_host[qvel_offset[NQ, NV]() + i] = cpu_data.qvel[i]
    ctx.enqueue_copy(gpu_state_buf, gpu_state_host.unsafe_ptr())
    ctx.synchronize()

    print()
    print(
        "Step | CPU32_qpos_max | CPU32_qvel_max | CPU-GPU_qpos     "
        "    | CPU-GPU_qvel         | CPU32_z | GPU_z   | CPU32_ang | GPU_ang"
    )
    print("-" * 130)

    for step in range(MAX_STEPS):
        # Get obs from CPU f32 data (with velocity clipping)
        var obs = List[Float64](capacity=OBS_DIM)
        for k in range(1, 6):
            obs.append(Float64(cpu_data.qpos[k]))
        for k in range(6):
            var v = Float64(cpu_data.qvel[k])
            if v > 10.0:
                v = 10.0
            elif v < -10.0:
                v = -10.0
            obs.append(v)

        var action = _get_greedy_action(agent, obs)

        # === Step CPU f32 (manual, matching GPU exactly) ===
        for i in range(NV):
            cpu_data.qfrc[i] = Scalar[DTYPE](0)
        for i in range(ACTION_DIM):
            var ctrl = action[i]
            if ctrl > HopperModel._acd.motor_ctrl_max[i]:
                ctrl = HopperModel._acd.motor_ctrl_max[i]
            elif ctrl < HopperModel._acd.motor_ctrl_min[i]:
                ctrl = HopperModel._acd.motor_ctrl_min[i]
            var dof = HopperModel._acd.motor_dof_adr[i]
            cpu_data.qfrc[dof] = Scalar[DTYPE](
                HopperModel._acd.motor_gears[i] * ctrl
            )

        for _ in range(FRAME_SKIP):
            RK4Integrator[SOLVER=NewtonSolver].step[NGEOM=NGEOM](
                cpu_model, cpu_data
            )

        # === Step GPU f32 ===
        ctx.enqueue_copy(gpu_state_host.unsafe_ptr(), gpu_state_buf)
        ctx.synchronize()

        for i in range(ACTION_DIM):
            var ctrl = action[i]
            if ctrl > HopperModel._acd.motor_ctrl_max[i]:
                ctrl = HopperModel._acd.motor_ctrl_max[i]
            elif ctrl < HopperModel._acd.motor_ctrl_min[i]:
                ctrl = HopperModel._acd.motor_ctrl_min[i]
            var dof = HopperModel._acd.motor_dof_adr[i]
            gpu_state_host[qfrc_offset[NQ, NV]() + dof] = Scalar[DTYPE](
                HopperModel._acd.motor_gears[i] * ctrl
            )
        ctx.enqueue_copy(gpu_state_buf, gpu_state_host.unsafe_ptr())
        ctx.synchronize()

        for _ in range(FRAME_SKIP):
            RK4Integrator[SOLVER=NewtonSolver].step_gpu[
                DTYPE,
                NQ,
                NV,
                NBODY,
                NJOINT,
                MAX_CONTACTS,
                GPU_BATCH,
                NGEOM=NGEOM,
                CONE_TYPE=HopperModel.CONE_TYPE,
                STEP_THREADS=1,
            ](ctx, gpu_state_buf, gpu_model_buf, gpu_ws_buf)
        ctx.synchronize()

        # Read back GPU
        ctx.enqueue_copy(gpu_state_host.unsafe_ptr(), gpu_state_buf)
        ctx.synchronize()

        # === Compare CPU f32 vs GPU f32 ===
        var qpos_err: Float64 = 0.0
        var qvel_err: Float64 = 0.0
        var cpu_qpos_max: Float64 = 0.0
        var cpu_qvel_max: Float64 = 0.0

        for i in range(NQ):
            var cpu_q = Float64(cpu_data.qpos[i])
            var gpu_q = Float64(gpu_state_host[qpos_offset[NQ, NV]() + i])
            var err = abs(cpu_q - gpu_q)
            if err > qpos_err:
                qpos_err = err
            if abs(cpu_q) > cpu_qpos_max:
                cpu_qpos_max = abs(cpu_q)

        for i in range(NV):
            var cpu_v = Float64(cpu_data.qvel[i])
            var gpu_v = Float64(gpu_state_host[qvel_offset[NQ, NV]() + i])
            var err = abs(cpu_v - gpu_v)
            if err > qvel_err:
                qvel_err = err
            if abs(cpu_v) > cpu_qvel_max:
                cpu_qvel_max = abs(cpu_v)

        var cpu_z = Float64(cpu_data.qpos[1])
        var cpu_angle = Float64(cpu_data.qpos[2])
        var gpu_z = Float64(gpu_state_host[qpos_offset[NQ, NV]() + 1])
        var gpu_angle = Float64(gpu_state_host[qpos_offset[NQ, NV]() + 2])

        # or qpos_err > 0.01 or qvel_err > 0.1
        if step % 10 == 0:
            print(
                String(step)[byte=:4]
                + " | "
                + String(cpu_qpos_max)[byte=:14]
                + " | "
                + String(cpu_qvel_max)[byte=:14]
                + " | "
                + String(qpos_err)[byte=:25]
                + " | "
                + String(qvel_err)[byte=:25]
                + " | "
                + String(cpu_z)[byte=:7]
                + " | "
                + String(gpu_z)[byte=:7]
                + " | "
                + String(cpu_angle)[byte=:9]
                + " | "
                + String(gpu_angle)[byte=:9]
            )

    print("-" * 130)
    print(
        "If CPU-GPU errors are near zero → GPU divergence is purely float32 vs"
        " float64."
    )
    print(
        "If CPU-GPU errors are large → there's a GPU-specific bug beyond"
        " precision."
    )

    cpu_env.close()
