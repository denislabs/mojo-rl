"""Diagnostic: InvertedDoublePendulum CPU vs GPU step-by-step.

Runs identical actions on CPU (f32) and GPU (f32), comparing:
- qpos/qvel after each physics step
- Observations (custom sin/cos encoding)
- Rewards and termination
- Raw state buffer values

This isolates whether the GPU env produces different results from CPU,
which would explain SAC Q-value divergence on this env but not others.

Run with:
    pixi run -e apple mojo run -I . examples/inverted_double_pendulum/idp_cpu_vs_gpu_diagnostic.mojo
    pixi run -e nvidia mojo run -I . examples/inverted_double_pendulum/idp_cpu_vs_gpu_diagnostic.mojo
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
# Legacy engine kept here on purpose: this CPU-vs-GPU diagnostic exercises the
# legacy static GPU kernels (step_kernel_gpu); the fields facade has none. Dies
# with the legacy engine at P6.
from mojo_rl.envs.phyics3d_env import Phyics3dEnv
from mojo_rl.nn.core.ptr import mptr
from mojo_rl.envs.inverted_double_pendulum.inverted_double_pendulum_xml import (
    InvertedDoublePendulumModel,
)
from mojo_rl.envs.inverted_double_pendulum.inverted_double_pendulum_config import (
    InvertedDoublePendulumConfig,
)


# =============================================================================
# Constants
# =============================================================================

comptime DTYPE = DType.float32
comptime NQ = InvertedDoublePendulumModel.NQ  # 3
comptime NV = InvertedDoublePendulumModel.NV  # 3
comptime NBODY = InvertedDoublePendulumModel.NBODY  # 4
comptime NJOINT = InvertedDoublePendulumModel.NJOINT  # 3
comptime NGEOM = InvertedDoublePendulumModel.NGEOM
comptime NSITE = InvertedDoublePendulumModel.NSITE  # 1
comptime MAX_CONTACTS = InvertedDoublePendulumModel.MAX_CONTACTS
comptime ACTION_DIM = InvertedDoublePendulumModel.ACTION_DIM  # 1
comptime OBS_DIM = 9
comptime FRAME_SKIP = 5

comptime GPU_BATCH = 1
comptime STATE_SIZE = state_size[NQ, NV, NBODY, MAX_CONTACTS, NSITE]()
comptime MODEL_SIZE = model_size_with_invweight[
    NBODY,
    NJOINT,
    NV,
    NGEOM,
    NEQUALITY=InvertedDoublePendulumModel.MAX_EQUALITY,
    NTENDON=InvertedDoublePendulumModel.MAX_TENDON,
    NSITE=InvertedDoublePendulumModel.NSITE,
]()
comptime SOLVER_WS = NewtonSolver.solver_workspace_size[NV, MAX_CONTACTS]()
comptime WS_SIZE = integrator_workspace_size[
    NV, NBODY
]() + NV * NV + SOLVER_WS + rk4_extra_workspace_size[NQ, NV]()

comptime MAX_ENV_STEPS = 100
comptime POLE_LEN: Float64 = 0.6


def compute_cpu_obs(
    data: Data[DTYPE, NQ, NV, NBODY, NJOINT, MAX_CONTACTS, NSITE],
) -> InlineArray[Float64, OBS_DIM]:
    """Compute the custom 9D observation from CPU data (matching GPU kernel)."""
    var obs = InlineArray[Float64, OBS_DIM](fill=0.0)
    obs[0] = Float64(data.qpos[0])  # cart_x
    obs[1] = sin(Float64(data.qpos[1]))  # sin(q1)
    obs[2] = sin(Float64(data.qpos[2]))  # sin(q2)
    obs[3] = cos(Float64(data.qpos[1]))  # cos(q1)
    obs[4] = cos(Float64(data.qpos[2]))  # cos(q2)
    for i in range(3):
        var v = Float64(data.qvel[i])
        if v > 10.0:
            v = 10.0
        elif v < -10.0:
            v = -10.0
        obs[5 + i] = v
    obs[8] = 0.0
    return obs^


def compute_cpu_reward_done(
    data: Data[DTYPE, NQ, NV, NBODY, NJOINT, MAX_CONTACTS, NSITE],
) -> Tuple[Float64, Bool]:
    """Compute reward and termination (matching GPU kernel)."""
    var q0 = Float64(data.qpos[0])
    var q1 = Float64(data.qpos[1])
    var q2 = Float64(data.qpos[2])

    var x_tip = q0 + POLE_LEN * sin(q1) + POLE_LEN * sin(q1 + q2)
    var z_tip = POLE_LEN * cos(q1) + POLE_LEN * cos(q1 + q2)

    var terminated = z_tip <= 1.0

    var dist_penalty = 0.01 * x_tip * x_tip + (z_tip - 2.0) * (z_tip - 2.0)
    var v1 = Float64(data.qvel[1])
    var v2 = Float64(data.qvel[2])
    var vel_penalty = 1e-3 * v1 * v1 + 5e-3 * v2 * v2

    var alive_bonus = 0.0 if terminated else 10.0
    var reward = alive_bonus - dist_penalty - vel_penalty

    return (reward, terminated)


def main() raises:
    seed(42)
    print("=" * 80)
    print("InvertedDoublePendulum: CPU (f32) vs GPU (f32) Step-by-Step")
    print("=" * 80)
    print(
        "STATE_SIZE:",
        STATE_SIZE,
        "MODEL_SIZE:",
        MODEL_SIZE,
        "WS_SIZE:",
        WS_SIZE,
    )
    print("NQ:", NQ, "NV:", NV, "NBODY:", NBODY, "NSITE:", NSITE)
    print(
        "OBS_DIM:",
        OBS_DIM,
        "ACTION_DIM:",
        ACTION_DIM,
        "FRAME_SKIP:",
        FRAME_SKIP,
    )
    print()

    # === CPU setup ===
    var cpu_model = Model[
        DTYPE,
        NQ,
        NV,
        NBODY,
        NJOINT,
        MAX_CONTACTS,
        NGEOM,
        InvertedDoublePendulumModel.MAX_EQUALITY,
        InvertedDoublePendulumModel.CONE_TYPE,
        InvertedDoublePendulumModel.MAX_TENDON,
        InvertedDoublePendulumModel.NSITE,
    ]()
    var cpu_data = Data[
        DTYPE,
        NQ,
        NV,
        NBODY,
        NJOINT,
        MAX_CONTACTS,
        NSITE,
    ]()
    InvertedDoublePendulumModel.setup_model_and_data[DTYPE](cpu_model, cpu_data)

    # === GPU setup ===
    var ctx = DeviceContext()
    var gpu_state_host = create_state_buffer[
        DTYPE, NQ, NV, NBODY, MAX_CONTACTS, NSITE, GPU_BATCH
    ](ctx)
    var gpu_state_buf = ctx.enqueue_create_buffer[DTYPE](GPU_BATCH * STATE_SIZE)
    var gpu_model_buf = ctx.enqueue_create_buffer[DTYPE](MODEL_SIZE)
    InvertedDoublePendulumModel.init_model_gpu(ctx, gpu_model_buf)
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

    # === GPU obs/reward extraction buffers ===
    var gpu_obs_buf = ctx.enqueue_create_buffer[DTYPE](GPU_BATCH * OBS_DIM)
    var gpu_obs_host = ctx.enqueue_create_host_buffer[DTYPE](
        GPU_BATCH * OBS_DIM
    )
    var gpu_rewards_buf = ctx.enqueue_create_buffer[DTYPE](GPU_BATCH)
    var gpu_dones_buf = ctx.enqueue_create_buffer[DTYPE](GPU_BATCH)
    var gpu_terminated_buf = ctx.enqueue_create_buffer[DTYPE](GPU_BATCH)
    var gpu_actions_buf = ctx.enqueue_create_buffer[DTYPE](
        GPU_BATCH * ACTION_DIM
    )
    var gpu_actions_host = ctx.enqueue_create_host_buffer[DTYPE](
        GPU_BATCH * ACTION_DIM
    )
    var gpu_rewards_host = ctx.enqueue_create_host_buffer[DTYPE](GPU_BATCH)
    var gpu_dones_host = ctx.enqueue_create_host_buffer[DTYPE](GPU_BATCH)
    var gpu_terminated_host = ctx.enqueue_create_host_buffer[DTYPE](GPU_BATCH)

    # Pre-allocate env workspace for step_kernel_gpu
    comptime ENV_WS_SIZE = MODEL_SIZE + GPU_BATCH * WS_SIZE
    var env_ws_buf = ctx.enqueue_create_buffer[DTYPE](ENV_WS_SIZE)
    # Initialize model into workspace
    Phyics3dEnv[
        InvertedDoublePendulumModel, InvertedDoublePendulumConfig, DTYPE
    ].init_step_workspace_gpu[GPU_BATCH](ctx, env_ws_buf)
    ctx.synchronize()

    # Predefined actions: alternate between small positive and negative
    var actions = InlineArray[Float64, MAX_ENV_STEPS](fill=0.0)
    for i in range(MAX_ENV_STEPS):
        if i % 3 == 0:
            actions[i] = 0.3
        elif i % 3 == 1:
            actions[i] = -0.2
        else:
            actions[i] = 0.0

    print(
        "Step | qpos_err         | qvel_err         | obs_err          "
        "| rew_cpu  | rew_gpu  | rew_err     | done_c | done_g"
    )
    print("-" * 120)

    var max_qpos_err: Float64 = 0.0
    var max_qvel_err: Float64 = 0.0
    var max_obs_err: Float64 = 0.0
    var max_rew_err: Float64 = 0.0

    for step in range(MAX_ENV_STEPS):
        var action = actions[step]

        # === CPU step ===
        for i in range(NV):
            cpu_data.qfrc[i] = Scalar[DTYPE](0)
        var ctrl = action
        if ctrl > Float64(InvertedDoublePendulumModel._acd.motor_ctrl_max[0]):
            ctrl = Float64(InvertedDoublePendulumModel._acd.motor_ctrl_max[0])
        elif ctrl < Float64(InvertedDoublePendulumModel._acd.motor_ctrl_min[0]):
            ctrl = Float64(InvertedDoublePendulumModel._acd.motor_ctrl_min[0])
        var dof = InvertedDoublePendulumModel._acd.motor_dof_adr[0]
        cpu_data.qfrc[dof] = Scalar[DTYPE](
            InvertedDoublePendulumModel._acd.motor_gears[0] * ctrl
        )

        for _ in range(FRAME_SKIP):
            RK4Integrator[SOLVER=NewtonSolver].step[NGEOM=NGEOM](
                cpu_model, cpu_data
            )
        forward_kinematics(cpu_model, cpu_data)

        var cpu_obs = compute_cpu_obs(cpu_data)
        var cpu_rwd = compute_cpu_reward_done(cpu_data)
        var cpu_reward = cpu_rwd[0]
        var cpu_done = cpu_rwd[1]

        # === GPU step via step_kernel_gpu ===
        gpu_actions_host[0] = Scalar[DTYPE](action)
        ctx.enqueue_copy(gpu_actions_buf, gpu_actions_host.unsafe_ptr())

        Phyics3dEnv[
            InvertedDoublePendulumModel, InvertedDoublePendulumConfig, DTYPE
        ].step_kernel_gpu[
            GPU_BATCH,
            STATE_SIZE,
            OBS_DIM,
            ACTION_DIM,
        ](
            ctx,
            gpu_state_buf,
            gpu_actions_buf,
            gpu_rewards_buf,
            gpu_dones_buf,
            gpu_terminated_buf,
            gpu_obs_buf,
            rng_seed=UInt64(step + 1),
            workspace_ptr=mptr(env_ws_buf.unsafe_ptr()),
        )
        ctx.synchronize()

        # Read back GPU results
        ctx.enqueue_copy(gpu_obs_host.unsafe_ptr(), gpu_obs_buf)
        ctx.enqueue_copy(gpu_rewards_host.unsafe_ptr(), gpu_rewards_buf)
        ctx.enqueue_copy(gpu_dones_host.unsafe_ptr(), gpu_dones_buf)
        ctx.enqueue_copy(gpu_terminated_host.unsafe_ptr(), gpu_terminated_buf)
        ctx.enqueue_copy(gpu_state_host.unsafe_ptr(), gpu_state_buf)
        ctx.synchronize()

        # === Compare qpos/qvel ===
        var qpos_err: Float64 = 0.0
        var qvel_err: Float64 = 0.0
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

        # === Compare observations ===
        var obs_err: Float64 = 0.0
        for i in range(OBS_DIM):
            var cpu_o = cpu_obs[i]
            var gpu_o = Float64(gpu_obs_host[i])
            var err = abs(cpu_o - gpu_o)
            if err > obs_err:
                obs_err = err

        # === Compare reward ===
        var gpu_reward = Float64(gpu_rewards_host[0])
        var rew_err = abs(cpu_reward - gpu_reward)

        var gpu_done = Float64(gpu_dones_host[0]) > 0.5

        if qpos_err > max_qpos_err:
            max_qpos_err = qpos_err
        if qvel_err > max_qvel_err:
            max_qvel_err = qvel_err
        if obs_err > max_obs_err:
            max_obs_err = obs_err
        if rew_err > max_rew_err:
            max_rew_err = rew_err

        # Print every 5 steps or when error is large
        var large_err = (
            qpos_err > 1e-4
            or qvel_err > 1e-3
            or obs_err > 1e-4
            or rew_err > 0.01
        )
        if step % 5 == 0 or large_err:
            var tag = "  " if not large_err else "!!"
            print(
                tag
                + String(step)[byte=:4]
                + " | "
                + String(qpos_err)[byte=:24]
                + " | "
                + String(qvel_err)[byte=:24]
                + " | "
                + String(obs_err)[byte=:24]
                + " | "
                + String(cpu_reward)[byte=:24]
                + " | "
                + String(gpu_reward)[byte=:24]
                + " | "
                + String(rew_err)[byte=:24]
                + " | "
                + String(cpu_done)
                + "  | "
                + String(gpu_done)
            )

            # If obs error is large, dump individual obs values
            if obs_err > 1e-3:
                print("    CPU obs:", end="")
                for i in range(OBS_DIM):
                    print(" " + String(cpu_obs[i])[byte=:10], end="")
                print()
                print("    GPU obs:", end="")
                for i in range(OBS_DIM):
                    print(
                        " " + String(Float64(gpu_obs_host[i]))[byte=:10], end=""
                    )
                print()

        if cpu_done or gpu_done:
            print(">>> Episode ended at step " + String(step) + " <<<")
            break

        # Sync CPU → GPU for next step (so they don't diverge from each other)
        for i in range(GPU_BATCH * STATE_SIZE):
            gpu_state_host[i] = Scalar[DTYPE](0)
        for i in range(NQ):
            gpu_state_host[qpos_offset[NQ, NV]() + i] = cpu_data.qpos[i]
        for i in range(NV):
            gpu_state_host[qvel_offset[NQ, NV]() + i] = cpu_data.qvel[i]
        ctx.enqueue_copy(gpu_state_buf, gpu_state_host.unsafe_ptr())
        ctx.synchronize()

    print("-" * 120)
    print(
        "Max errors: qpos="
        + String(max_qpos_err)[byte=:16]
        + " qvel="
        + String(max_qvel_err)[byte=:16]
        + " obs="
        + String(max_obs_err)[byte=:16]
        + " rew="
        + String(max_rew_err)[byte=:11]
    )
    print()
    if max_obs_err > 0.01:
        print("!!! SIGNIFICANT OBS MISMATCH — GPU obs differ from expected !!!")
    elif max_qpos_err > 1e-3:
        print(
            "!!! SIGNIFICANT PHYSICS MISMATCH — GPU dynamics differ from"
            " CPU !!!"
        )
    else:
        print("CPU and GPU match well — env is not the problem.")
