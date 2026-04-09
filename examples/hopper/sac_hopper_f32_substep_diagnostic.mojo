"""Diagnostic: CPU f32 vs GPU f32 per-substep — NO compounding.

After each RK4 substep, copies CPU state to GPU so GPU always starts
from the exact same state as CPU. This isolates per-step GPU error
from compounding. If errors stay tiny (~1e-7), the GPU implementation
is correct and previous divergence was purely from error accumulation.

Run with:
    pixi run -e apple mojo run -I . examples/hopper/sac_hopper_f32_substep_diagnostic.mojo
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
from mojo_rl.physics3d.gpu.constants import (
    state_size,
    model_size_with_invweight,
    integrator_workspace_size,
    rk4_extra_workspace_size,
    qpos_offset,
    qvel_offset,
    qfrc_offset,
    qacc_offset,
    contacts_offset,
    metadata_offset,
    CONTACT_SIZE,
    CONTACT_IDX_BODY_A,
    CONTACT_IDX_BODY_B,
    CONTACT_IDX_DIST,
    CONTACT_IDX_POS_X,
    CONTACT_IDX_POS_Y,
    CONTACT_IDX_POS_Z,
    CONTACT_IDX_FORCE_N,
    META_IDX_NUM_CONTACTS,
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
comptime WS_SIZE = integrator_workspace_size[
    NV, NBODY
]() + NV * NV + SOLVER_WS + rk4_extra_workspace_size[NQ, NV]()

# How many env steps to run (each = FRAME_SKIP substeps)
comptime MAX_ENV_STEPS = 200

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
        result.append(tanh(Float64(out_arr[i])) * agent.action_scale)
    return result^


def main() raises:
    seed(42)
    print("=" * 70)
    print("Hopper f32 Substep Diagnostic: CPU vs GPU per-RK4-step")
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
    print("Loaded checkpoint")

    # === CPU f32 setup ===
    var cpu_env = Hopper[DTYPE, TERMINATE_ON_UNHEALTHY=True]()
    _ = cpu_env.reset()

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

    for i in range(NQ):
        cpu_data.qpos[i] = cpu_env.get_qpos(i)
    for i in range(NV):
        cpu_data.qvel[i] = cpu_env.get_qvel(i)
    forward_kinematics(cpu_model, cpu_data)

    # === GPU f32 setup ===
    var ctx = DeviceContext()
    var gpu_state_host = create_state_buffer[
        DTYPE, NQ, NV, NBODY, MAX_CONTACTS, NSITE, GPU_BATCH
    ](ctx)
    var gpu_state_buf = ctx.enqueue_create_buffer[DTYPE](GPU_BATCH * STATE_SIZE)
    var gpu_model_buf = ctx.enqueue_create_buffer[DTYPE](MODEL_SIZE)
    HopperModel.init_model_gpu(ctx, gpu_model_buf)
    var gpu_ws_buf = ctx.enqueue_create_buffer[DTYPE](GPU_BATCH * WS_SIZE)
    ctx.synchronize()

    # Sync initial state
    for i in range(GPU_BATCH * STATE_SIZE):
        gpu_state_host[i] = Scalar[DTYPE](0)
    for i in range(NQ):
        gpu_state_host[qpos_offset[NQ, NV]() + i] = cpu_data.qpos[i]
    for i in range(NV):
        gpu_state_host[qvel_offset[NQ, NV]() + i] = cpu_data.qvel[i]
    ctx.enqueue_copy(gpu_state_buf, gpu_state_host.unsafe_ptr())
    ctx.synchronize()

    # Offsets
    comptime CONTACTS_OFF = contacts_offset[NQ, NV, NBODY]()
    comptime META_OFF = metadata_offset[NQ, NV, NBODY, MAX_CONTACTS]()

    var first_diverge = -1
    var substep_count = 0

    for env_step in range(MAX_ENV_STEPS):
        # Get obs from CPU state
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

        # Apply actions to CPU
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

        # Apply actions to GPU
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

        # Step FRAME_SKIP substeps, comparing after EACH one
        for sub in range(FRAME_SKIP):
            # Sync CPU qfrc to GPU before each substep (GPU reads qfrc
            # from state buffer; CPU qfrc is already set from actions above)
            ctx.enqueue_copy(gpu_state_host.unsafe_ptr(), gpu_state_buf)
            ctx.synchronize()
            for i in range(NV):
                gpu_state_host[qfrc_offset[NQ, NV]() + i] = cpu_data.qfrc[i]
            ctx.enqueue_copy(gpu_state_buf, gpu_state_host.unsafe_ptr())

            # Zero GPU workspace to prevent stale warm-start from affecting solver
            var ws_host = ctx.enqueue_create_host_buffer[DTYPE](
                GPU_BATCH * WS_SIZE
            )
            for i in range(GPU_BATCH * WS_SIZE):
                ws_host[i] = Scalar[DTYPE](0)
            ctx.enqueue_copy(gpu_ws_buf, ws_host.unsafe_ptr())
            ctx.synchronize()

            # CPU: one RK4 step
            RK4Integrator[SOLVER=NewtonSolver].step[NGEOM=NGEOM](
                cpu_model, cpu_data
            )

            # GPU: one RK4 step
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
            ](ctx, gpu_state_buf, gpu_model_buf, gpu_ws_buf)
            ctx.synchronize()

            # Read GPU state
            ctx.enqueue_copy(gpu_state_host.unsafe_ptr(), gpu_state_buf)
            ctx.synchronize()

            # Compare
            var qpos_err: Float64 = 0.0
            var qvel_err: Float64 = 0.0
            var qacc_err: Float64 = 0.0
            for i in range(NQ):
                var err = abs(
                    Float64(cpu_data.qpos[i])
                    - Float64(gpu_state_host[qpos_offset[NQ, NV]() + i])
                )
                if err > qpos_err:
                    qpos_err = err
            for i in range(NV):
                var err = abs(
                    Float64(cpu_data.qvel[i])
                    - Float64(gpu_state_host[qvel_offset[NQ, NV]() + i])
                )
                if err > qvel_err:
                    qvel_err = err
            for i in range(NV):
                var err = abs(
                    Float64(cpu_data.qacc[i])
                    - Float64(gpu_state_host[qacc_offset[NQ, NV]() + i])
                )
                if err > qacc_err:
                    qacc_err = err

            # Contact count
            var cpu_ncon = Int(cpu_data.num_contacts)
            var gpu_ncon = Int(gpu_state_host[META_OFF + META_IDX_NUM_CONTACTS])

            var contact_mismatch = cpu_ncon != gpu_ncon

            # Print if: every 20 substeps, or contact mismatch, or large error
            var should_print = (
                substep_count % 20 == 0
                or contact_mismatch
                or qpos_err > 1e-4
                or qvel_err > 1e-3
                or qacc_err > 1e-2
            )

            if should_print:
                var tag = "  "
                if contact_mismatch:
                    tag = "C!"
                elif qvel_err > 0.01:
                    tag = "V!"
                elif qpos_err > 1e-4:
                    tag = "P!"

                print(
                    tag
                    + " step="
                    + String(env_step)
                    + "."
                    + String(sub)
                    + " (sub="
                    + String(substep_count)
                    + ")"
                    + " | ncon: cpu="
                    + String(cpu_ncon)
                    + " gpu="
                    + String(gpu_ncon)
                    + " | qpos_err="
                    + String(qpos_err)[byte=:24]
                    + " | qvel_err="
                    + String(qvel_err)[byte=:24]
                    + " | qacc_err="
                    + String(qacc_err)[byte=:24]
                )

                # Print contact details if mismatch
                if contact_mismatch:
                    print("    CPU contacts:")
                    for c in range(cpu_ncon):
                        print(
                            "      ["
                            + String(c)
                            + "]"
                            + " body="
                            + String(Int(cpu_data.contacts[c].body_a))
                            + "-"
                            + String(Int(cpu_data.contacts[c].body_b))
                            + " dist="
                            + String(Float64(cpu_data.contacts[c].dist))[
                                byte=:12
                            ]
                        )
                    print("    GPU contacts:")
                    for c in range(gpu_ncon):
                        var c_off = CONTACTS_OFF + c * CONTACT_SIZE
                        print(
                            "      ["
                            + String(c)
                            + "]"
                            + " body="
                            + String(
                                Int(gpu_state_host[c_off + CONTACT_IDX_BODY_A])
                            )
                            + "-"
                            + String(
                                Int(gpu_state_host[c_off + CONTACT_IDX_BODY_B])
                            )
                            + " dist="
                            + String(
                                Float64(
                                    gpu_state_host[c_off + CONTACT_IDX_DIST]
                                )
                            )[byte=:12]
                        )

                if first_diverge < 0 and (contact_mismatch or qpos_err > 1e-4):
                    first_diverge = substep_count
                    print(
                        "  >>> FIRST SIGNIFICANT DIVERGENCE at substep "
                        + String(substep_count)
                        + " <<<"
                    )
                    # Print full state comparison
                    print("    CPU qpos:", end="")
                    for i in range(NQ):
                        print(
                            " " + String(Float64(cpu_data.qpos[i]))[byte=:12],
                            end="",
                        )
                    print()
                    print("    GPU qpos:", end="")
                    for i in range(NQ):
                        print(
                            " "
                            + String(
                                Float64(
                                    gpu_state_host[qpos_offset[NQ, NV]() + i]
                                )
                            )[byte=:12],
                            end="",
                        )
                    print()
                    print("    CPU qvel:", end="")
                    for i in range(NV):
                        print(
                            " " + String(Float64(cpu_data.qvel[i]))[byte=:12],
                            end="",
                        )
                    print()
                    print("    GPU qvel:", end="")
                    for i in range(NV):
                        print(
                            " "
                            + String(
                                Float64(
                                    gpu_state_host[qvel_offset[NQ, NV]() + i]
                                )
                            )[byte=:12],
                            end="",
                        )
                    print()

            # === Sync CPU state → GPU: zero ENTIRE buffer, set only qpos/qvel/qacc/qfrc ===
            # This matches exactly what the fresh Euler/RK4 test does.
            for i in range(GPU_BATCH * STATE_SIZE):
                gpu_state_host[i] = Scalar[DTYPE](0)
            for i in range(NQ):
                gpu_state_host[qpos_offset[NQ, NV]() + i] = cpu_data.qpos[i]
            for i in range(NV):
                gpu_state_host[qvel_offset[NQ, NV]() + i] = cpu_data.qvel[i]
                gpu_state_host[qacc_offset[NQ, NV]() + i] = cpu_data.qacc[i]
                gpu_state_host[qfrc_offset[NQ, NV]() + i] = cpu_data.qfrc[i]
            ctx.enqueue_copy(gpu_state_buf, gpu_state_host.unsafe_ptr())
            ctx.synchronize()

            substep_count += 1

    print()
    print("Total substeps: " + String(substep_count))
    if first_diverge >= 0:
        print("First divergence at substep: " + String(first_diverge))
    else:
        print("No significant divergence detected!")

    cpu_env.close()
