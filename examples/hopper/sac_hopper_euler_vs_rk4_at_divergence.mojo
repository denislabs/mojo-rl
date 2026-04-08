"""Diagnostic: At the exact state where RK4 CPU-GPU diverges (substep 173),
run a single EULER step on both CPU and GPU to isolate whether the issue
is in the per-stage dynamics or the RK4 multi-stage interaction.

Also runs a single RK4 step for comparison.

Run with:
    pixi run -e apple mojo run -I . examples/hopper/sac_hopper_euler_vs_rk4_at_divergence.mojo
"""

from std.random import seed
from std.math import abs, tanh
from std.gpu.host import DeviceContext, DeviceBuffer, HostBuffer

from mojo_rl.deep_agents.core.agents import DeepSACAgent
from mojo_rl.core.logger import NoOpLogger
from mojo_rl.envs.hopper import Hopper, HopperConfig
from mojo_rl.envs.hopper.hopper_xml import HopperModel
from mojo_rl.physics3d.types import Model, Data
from mojo_rl.physics3d.kinematics import forward_kinematics
from mojo_rl.physics3d.solver import NewtonSolver
from mojo_rl.physics3d.integrator.rk4_integrator import RK4Integrator
from mojo_rl.physics3d.integrator.euler_integrator import EulerIntegrator
from mojo_rl.physics3d.gpu.constants import (
    state_size, model_size_with_invweight,
    integrator_workspace_size, rk4_extra_workspace_size,
    qpos_offset, qvel_offset, qfrc_offset, qacc_offset,
    contacts_offset, metadata_offset,
    CONTACT_SIZE, META_IDX_NUM_CONTACTS,
)
from mojo_rl.physics3d.gpu.buffer_utils import create_state_buffer

from layout import Layout, LayoutTensor
from mojo_rl.nn.constants import dtype as nn_dtype
from mojo_rl.deep_agents.core.configs.offpolicy_config import SACConfig
from mojo_rl.deep_agents.core.utils import obs_to_inline
from mojo_rl.nn.training import Network


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
comptime MAX_CONTACTS = HopperConfig.MAX_CONTACTS

comptime DTYPE = DType.float32
comptime GPU_BATCH = 1
comptime STATE_SIZE = state_size[NQ, NV, NBODY, MAX_CONTACTS, NSITE]()
comptime MODEL_SIZE = model_size_with_invweight[NBODY, NJOINT, NV, NGEOM]()
comptime SOLVER_WS = NewtonSolver.solver_workspace_size[NV, MAX_CONTACTS]()
comptime RK4_WS = integrator_workspace_size[
    NV, NBODY
]() + NV * NV + SOLVER_WS + rk4_extra_workspace_size[NQ, NV]()
comptime EULER_WS = integrator_workspace_size[
    NV, NBODY
]() + NV * NV + SOLVER_WS

# Reproduce the state at substep 172 (just before divergence)
comptime TARGET_SUBSTEP = 172

comptime ActorModel = SACConfig[OBS_DIM, ACTION_DIM].ActorModel
comptime ActorOpt = SACConfig[OBS_DIM, ACTION_DIM].ActorOpt
comptime AgentType = DeepSACAgent[
    OBS_DIM, ACTION_DIM, HIDDEN_DIM, BUFFER_CAPACITY, BATCH_SIZE,
    0.0003, 0.0003, 0, NoOpLogger, MAX_N_ENVS,
]


def _get_greedy_action(
    agent: AgentType, obs: List[Float64],
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
        result.append(tanh(Float64(out_arr[i])) * agent.action_scale)
    return result^


def _compare_state(
    label: String,
    cpu_data: Data[DTYPE, NQ, NV, NBODY, NJOINT, MAX_CONTACTS, NSITE],
    gpu_host: HostBuffer[DTYPE],
):
    print("  " + label + ":")
    var max_qpos_err: Float64 = 0.0
    var max_qvel_err: Float64 = 0.0
    var max_qacc_err: Float64 = 0.0
    for i in range(NQ):
        var cpu_q = Float64(cpu_data.qpos[i])
        var gpu_q = Float64(gpu_host[qpos_offset[NQ, NV]() + i])
        var e = abs(cpu_q - gpu_q)
        if e > max_qpos_err:
            max_qpos_err = e
    for i in range(NV):
        var cpu_v = Float64(cpu_data.qvel[i])
        var gpu_v = Float64(gpu_host[qvel_offset[NQ, NV]() + i])
        var e = abs(cpu_v - gpu_v)
        if e > max_qvel_err:
            max_qvel_err = e
    for i in range(NV):
        var cpu_a = Float64(cpu_data.qacc[i])
        var gpu_a = Float64(gpu_host[qacc_offset[NQ, NV]() + i])
        var e = abs(cpu_a - gpu_a)
        if e > max_qacc_err:
            max_qacc_err = e

    comptime META_OFF = metadata_offset[NQ, NV, NBODY, MAX_CONTACTS]()
    var cpu_ncon = Int(cpu_data.num_contacts)
    var gpu_ncon = Int(gpu_host[META_OFF + META_IDX_NUM_CONTACTS])

    print(
        "    qpos_err=" + String(max_qpos_err)
        + "  qvel_err=" + String(max_qvel_err)
        + "  qacc_err=" + String(max_qacc_err)
        + "  ncon: cpu=" + String(cpu_ncon)
        + " gpu=" + String(gpu_ncon)
    )
    # Print per-DOF qvel comparison
    for i in range(NV):
        var cv = Float64(cpu_data.qvel[i])
        var gv = Float64(gpu_host[qvel_offset[NQ, NV]() + i])
        if abs(cv - gv) > 0.001:
            print(
                "    qvel[" + String(i) + "] cpu="
                + String(cv) + " gpu=" + String(gv)
                + " err=" + String(abs(cv - gv))
            )
    for i in range(NV):
        var ca = Float64(cpu_data.qacc[i])
        var ga = Float64(gpu_host[qacc_offset[NQ, NV]() + i])
        if abs(ca - ga) > 0.1:
            print(
                "    qacc[" + String(i) + "] cpu="
                + String(ca) + " gpu=" + String(ga)
                + " err=" + String(abs(ca - ga))
            )


def main() raises:
    seed(42)
    print("=" * 70)
    print("Euler vs RK4 at divergence point (substep 173)")
    print("=" * 70)

    var agent = AgentType(
        gamma=0.99, tau=0.005, action_scale=1.0,
        alpha=0.2, auto_alpha=False, target_entropy=-3.0,
    )
    agent.load_checkpoint("sac_hopper_1000.ckpt")

    # CPU setup
    var cpu_env = Hopper[DTYPE, TERMINATE_ON_UNHEALTHY=True]()
    _ = cpu_env.reset()
    var cpu_model = Model[
        DTYPE, NQ, NV, NBODY, NJOINT, MAX_CONTACTS, NGEOM,
        HopperModel.MAX_EQUALITY, HopperModel.CONE_TYPE,
        HopperModel.MAX_TENDON, HopperModel.NSITE,
    ]()
    var cpu_data = Data[
        DTYPE, NQ, NV, NBODY, NJOINT, MAX_CONTACTS, HopperModel.NSITE
    ]()
    HopperModel.setup_model_and_data(cpu_model, cpu_data)
    for i in range(NQ):
        cpu_data.qpos[i] = cpu_env.get_qpos(i)
    for i in range(NV):
        cpu_data.qvel[i] = cpu_env.get_qvel(i)
    forward_kinematics(cpu_model, cpu_data)

    # Run CPU to substep TARGET_SUBSTEP using RK4 (reproduce the state)
    print("Running CPU to substep " + String(TARGET_SUBSTEP) + "...")
    var substep = 0
    for env_step in range(TARGET_SUBSTEP // FRAME_SKIP + 1):
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

        for i in range(NV):
            cpu_data.qfrc[i] = Scalar[DTYPE](0)
        for i in range(ACTION_DIM):
            var ctrl = action[i]
            if ctrl > HopperModel._acd.motor_ctrl_max[i]:
                ctrl = HopperModel._acd.motor_ctrl_max[i]
            elif ctrl < HopperModel._acd.motor_ctrl_min[i]:
                ctrl = HopperModel._acd.motor_ctrl_min[i]
            cpu_data.qfrc[HopperModel._acd.motor_dof_adr[i]] = Scalar[DTYPE](
                HopperModel._acd.motor_gears[i] * ctrl
            )

        for sub in range(FRAME_SKIP):
            if substep >= TARGET_SUBSTEP:
                break
            RK4Integrator[SOLVER=NewtonSolver].step[NGEOM=NGEOM](
                cpu_model, cpu_data
            )
            substep += 1
        if substep >= TARGET_SUBSTEP:
            break

    # Now cpu_data is at substep TARGET_SUBSTEP, with qfrc set for current env step
    print("CPU state at substep " + String(TARGET_SUBSTEP) + ":")
    print("  qpos:", end="")
    for i in range(NQ):
        print(" " + String(Float64(cpu_data.qpos[i]))[byte=:12], end="")
    print()
    print("  qvel:", end="")
    for i in range(NV):
        print(" " + String(Float64(cpu_data.qvel[i]))[byte=:12], end="")
    print()
    print("  ncon:", Int(cpu_data.num_contacts))

    # GPU setup
    var ctx = DeviceContext()
    var gpu_state_host = create_state_buffer[
        DTYPE, NQ, NV, NBODY, MAX_CONTACTS, NSITE, GPU_BATCH
    ](ctx)
    var gpu_rk4_buf = ctx.enqueue_create_buffer[DTYPE](GPU_BATCH * STATE_SIZE)
    var gpu_euler_buf = ctx.enqueue_create_buffer[DTYPE](GPU_BATCH * STATE_SIZE)
    var gpu_model_buf = ctx.enqueue_create_buffer[DTYPE](MODEL_SIZE)
    HopperModel.init_model_gpu(ctx, gpu_model_buf)
    var gpu_rk4_ws = ctx.enqueue_create_buffer[DTYPE](GPU_BATCH * RK4_WS)
    var gpu_euler_ws = ctx.enqueue_create_buffer[DTYPE](GPU_BATCH * EULER_WS)
    ctx.synchronize()

    # Copy CPU state to both GPU buffers
    for i in range(GPU_BATCH * STATE_SIZE):
        gpu_state_host[i] = Scalar[DTYPE](0)
    for i in range(NQ):
        gpu_state_host[qpos_offset[NQ, NV]() + i] = cpu_data.qpos[i]
    for i in range(NV):
        gpu_state_host[qvel_offset[NQ, NV]() + i] = cpu_data.qvel[i]
        gpu_state_host[qacc_offset[NQ, NV]() + i] = cpu_data.qacc[i]
        gpu_state_host[qfrc_offset[NQ, NV]() + i] = cpu_data.qfrc[i]
    ctx.enqueue_copy(gpu_rk4_buf, gpu_state_host.unsafe_ptr())
    ctx.enqueue_copy(gpu_euler_buf, gpu_state_host.unsafe_ptr())
    ctx.synchronize()

    # === Test 1: Single EULER step CPU vs GPU ===
    print()
    print("=" * 50)
    print("Test 1: Single EULER step from same state")
    print("=" * 50)

    # CPU Euler
    var cpu_euler = Data[
        DTYPE, NQ, NV, NBODY, NJOINT, MAX_CONTACTS, HopperModel.NSITE
    ]()
    # Copy state
    for i in range(NQ):
        cpu_euler.qpos[i] = cpu_data.qpos[i]
    for i in range(NV):
        cpu_euler.qvel[i] = cpu_data.qvel[i]
        cpu_euler.qacc[i] = cpu_data.qacc[i]
        cpu_euler.qfrc[i] = cpu_data.qfrc[i]

    var cpu_euler_model = Model[
        DTYPE, NQ, NV, NBODY, NJOINT, MAX_CONTACTS, NGEOM,
        HopperModel.MAX_EQUALITY, HopperModel.CONE_TYPE,
        HopperModel.MAX_TENDON, HopperModel.NSITE,
    ]()
    HopperModel.setup_model_and_data(cpu_euler_model, cpu_euler)
    # Override with our state
    for i in range(NQ):
        cpu_euler.qpos[i] = cpu_data.qpos[i]
    for i in range(NV):
        cpu_euler.qvel[i] = cpu_data.qvel[i]
        cpu_euler.qacc[i] = cpu_data.qacc[i]
        cpu_euler.qfrc[i] = cpu_data.qfrc[i]
    forward_kinematics(cpu_euler_model, cpu_euler)

    EulerIntegrator[SOLVER=NewtonSolver].step[NGEOM=NGEOM](
        cpu_euler_model, cpu_euler
    )

    # GPU Euler
    EulerIntegrator[SOLVER=NewtonSolver].step_gpu[
        DTYPE, NQ, NV, NBODY, NJOINT, MAX_CONTACTS, GPU_BATCH,
        NGEOM=NGEOM, CONE_TYPE=HopperModel.CONE_TYPE,
    ](ctx, gpu_euler_buf, gpu_model_buf, gpu_euler_ws)
    ctx.synchronize()

    ctx.enqueue_copy(gpu_state_host.unsafe_ptr(), gpu_euler_buf)
    ctx.synchronize()

    _compare_state("EULER CPU vs GPU", cpu_euler, gpu_state_host)

    # === Test 2: Single RK4 step CPU vs GPU ===
    print()
    print("=" * 50)
    print("Test 2: Single RK4 step from same state")
    print("=" * 50)

    # CPU RK4
    var cpu_rk4 = Data[
        DTYPE, NQ, NV, NBODY, NJOINT, MAX_CONTACTS, HopperModel.NSITE
    ]()
    var cpu_rk4_model = Model[
        DTYPE, NQ, NV, NBODY, NJOINT, MAX_CONTACTS, NGEOM,
        HopperModel.MAX_EQUALITY, HopperModel.CONE_TYPE,
        HopperModel.MAX_TENDON, HopperModel.NSITE,
    ]()
    HopperModel.setup_model_and_data(cpu_rk4_model, cpu_rk4)
    for i in range(NQ):
        cpu_rk4.qpos[i] = cpu_data.qpos[i]
    for i in range(NV):
        cpu_rk4.qvel[i] = cpu_data.qvel[i]
        cpu_rk4.qacc[i] = cpu_data.qacc[i]
        cpu_rk4.qfrc[i] = cpu_data.qfrc[i]
    forward_kinematics(cpu_rk4_model, cpu_rk4)

    RK4Integrator[SOLVER=NewtonSolver].step[NGEOM=NGEOM](
        cpu_rk4_model, cpu_rk4
    )

    # GPU RK4
    RK4Integrator[SOLVER=NewtonSolver].step_gpu[
        DTYPE, NQ, NV, NBODY, NJOINT, MAX_CONTACTS, GPU_BATCH,
        NGEOM=NGEOM, CONE_TYPE=HopperModel.CONE_TYPE,
    ](ctx, gpu_rk4_buf, gpu_model_buf, gpu_rk4_ws)
    ctx.synchronize()

    ctx.enqueue_copy(gpu_state_host.unsafe_ptr(), gpu_rk4_buf)
    ctx.synchronize()

    _compare_state("RK4 CPU vs GPU", cpu_rk4, gpu_state_host)

    print()
    print("If EULER matches but RK4 diverges → issue is in RK4 multi-stage interaction")
    print("If EULER also diverges → issue is in single-step dynamics/solver")

    cpu_env.close()
