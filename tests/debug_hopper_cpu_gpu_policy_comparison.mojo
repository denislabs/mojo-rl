"""Debug script to compare CPU and GPU Hopper with trained policy.

This script runs the SAME trained policy on both CPU and GPU environments
starting from identical initial states to pinpoint where they diverge.

Key diagnostics:
1. Same initial qpos/qvel (no reset noise)
2. Same policy actions at each step
3. Step-by-step comparison of state, reward, and done

Run with:
    pixi run -e apple mojo run tests/debug_hopper_cpu_gpu_policy_comparison.mojo
"""

from random import seed

from gpu.host import DeviceContext, DeviceBuffer
from layout import Layout, LayoutTensor

from deep_agents.ppo import DeepPPOContinuousAgent
from envs.hopper import Hopper
from envs.hopper.hopper_def import HopperConstantsGPU
from deep_rl import dtype as gpu_dtype
from physics3d.gpu.constants import (
    qpos_offset,
    qvel_offset,
    qacc_offset,
    qfrc_offset,
)


# =============================================================================
# Constants (must match training configuration)
# =============================================================================

comptime OBS_DIM = HopperConstantsGPU.OBS_DIM  # 11
comptime ACTION_DIM = HopperConstantsGPU.ACTION_DIM  # 3
comptime HIDDEN_DIM = 256
comptime ROLLOUT_LEN = 512
comptime N_ENVS = 256
comptime GPU_MINIBATCH_SIZE = 2048

comptime dtype = DType.float32
comptime NQ = 6
comptime NV = 6


fn copy_cpu_state_to_gpu[
    STATE_SIZE: Int,
](
    ctx: DeviceContext,
    mut states_buf: DeviceBuffer[gpu_dtype],
    cpu_qpos: InlineArray[Scalar[DType.float64], 6],
    cpu_qvel: InlineArray[Scalar[DType.float64], 6],
) raises:
    """Copy CPU state to GPU state buffer (single environment at index 0)."""
    comptime QPOS_OFF = qpos_offset[NQ, NV]()
    comptime QVEL_OFF = qvel_offset[NQ, NV]()
    comptime QACC_OFF = qacc_offset[NQ, NV]()
    comptime QFRC_OFF = qfrc_offset[NQ, NV]()

    var state_host = List[Scalar[gpu_dtype]](capacity=STATE_SIZE)
    for _ in range(STATE_SIZE):
        state_host.append(Scalar[gpu_dtype](0.0))

    # Copy qpos
    for i in range(6):
        state_host[QPOS_OFF + i] = Scalar[gpu_dtype](cpu_qpos[i])

    # Copy qvel
    for i in range(6):
        state_host[QVEL_OFF + i] = Scalar[gpu_dtype](cpu_qvel[i])

    # Zero qacc and qfrc
    for i in range(6):
        state_host[QACC_OFF + i] = Scalar[gpu_dtype](0.0)
        state_host[QFRC_OFF + i] = Scalar[gpu_dtype](0.0)

    ctx.enqueue_copy(states_buf, state_host.unsafe_ptr())
    ctx.synchronize()


fn extract_gpu_state[
    STATE_SIZE: Int,
](
    ctx: DeviceContext,
    states_buf: DeviceBuffer[gpu_dtype],
) raises -> Tuple[
    InlineArray[Scalar[gpu_dtype], 6],
    InlineArray[Scalar[gpu_dtype], 6],
]:
    """Extract qpos and qvel from GPU state buffer."""
    comptime QPOS_OFF = qpos_offset[NQ, NV]()
    comptime QVEL_OFF = qvel_offset[NQ, NV]()

    var state_host = List[Scalar[gpu_dtype]](capacity=STATE_SIZE)
    for _ in range(STATE_SIZE):
        state_host.append(Scalar[gpu_dtype](0.0))

    ctx.enqueue_copy(state_host.unsafe_ptr(), states_buf)
    ctx.synchronize()

    var qpos = InlineArray[Scalar[gpu_dtype], 6](uninitialized=True)
    var qvel = InlineArray[Scalar[gpu_dtype], 6](uninitialized=True)

    for i in range(6):
        qpos[i] = state_host[QPOS_OFF + i]
        qvel[i] = state_host[QVEL_OFF + i]

    return (qpos^, qvel^)


fn main() raises:
    seed(42)
    print("=" * 80)
    print("DEBUG: CPU vs GPU Policy Comparison for Hopper")
    print("=" * 80)
    print()

    # =========================================================================
    # Load trained agent
    # =========================================================================

    var agent = DeepPPOContinuousAgent[
        obs_dim=OBS_DIM,
        action_dim=ACTION_DIM,
        hidden_dim=HIDDEN_DIM,
        rollout_len=ROLLOUT_LEN,
        n_envs=N_ENVS,
        gpu_minibatch_size=GPU_MINIBATCH_SIZE,
        clip_value=True,
    ](
        gamma=0.99,
        gae_lambda=0.95,
        clip_epsilon=0.2,
        actor_lr=0.0003,
        critic_lr=0.0003,
        entropy_coef=0.0,
        value_loss_coef=0.5,
        num_epochs=10,
        target_kl=0.0,
        max_grad_norm=0.5,
        anneal_lr=False,
    )

    print("Loading checkpoint...")
    try:
        agent.load_checkpoint("ppo_hopper.ckpt")
        print("Checkpoint loaded successfully!")
    except:
        print("Error loading checkpoint!")
        print("Make sure you have trained the agent first.")
        return

    print()

    # =========================================================================
    # Create environments
    # =========================================================================

    var cpu_env = Hopper[DType.float64](
        torque_limit=200.0,
        min_height=0.7,
        max_pitch=0.2,
        max_steps=1000,
        timestep=0.002,
        friction=0.5,
    )

    var ctx = DeviceContext()

    comptime STATE_SIZE = Hopper[DType.float64].STATE_SIZE
    comptime BATCH_SIZE = 1

    # Create GPU buffers
    var states_buf = ctx.enqueue_create_buffer[gpu_dtype](
        BATCH_SIZE * STATE_SIZE
    )
    var actions_buf = ctx.enqueue_create_buffer[gpu_dtype](
        BATCH_SIZE * ACTION_DIM
    )
    var rewards_buf = ctx.enqueue_create_buffer[gpu_dtype](BATCH_SIZE)
    var dones_buf = ctx.enqueue_create_buffer[gpu_dtype](BATCH_SIZE)
    var obs_buf = ctx.enqueue_create_buffer[gpu_dtype](BATCH_SIZE * OBS_DIM)

    # =========================================================================
    # Reset CPU environment with NO noise (deterministic initial state)
    # =========================================================================

    _ = cpu_env.reset()

    # Override with deterministic state (no reset noise)
    cpu_env.data.qpos[0] = 0.0  # rootx
    cpu_env.data.qpos[1] = 1.25  # rootz (initial height)
    cpu_env.data.qpos[2] = 0.0  # rooty (pitch)
    cpu_env.data.qpos[3] = 0.0  # thigh
    cpu_env.data.qpos[4] = 0.0  # leg
    cpu_env.data.qpos[5] = 0.0  # foot

    for i in range(6):
        cpu_env.data.qvel[i] = 0.0
        cpu_env.data.qacc[i] = 0.0
        cpu_env.data.qfrc[i] = 0.0

    cpu_env.current_step = 0

    # Run forward kinematics to update xpos/xquat
    from physics3d.kinematics.forward_kinematics import forward_kinematics

    forward_kinematics(cpu_env.model, cpu_env.data)
    cpu_env._update_cached_state()

    var cpu_qpos = cpu_env.get_qpos()
    var cpu_qvel = cpu_env.get_qvel()

    print("Initial CPU state:")
    print(
        "  qpos:",
        cpu_qpos[0],
        cpu_qpos[1],
        cpu_qpos[2],
        cpu_qpos[3],
        cpu_qpos[4],
        cpu_qpos[5],
    )
    print(
        "  qvel:",
        cpu_qvel[0],
        cpu_qvel[1],
        cpu_qvel[2],
        cpu_qvel[3],
        cpu_qvel[4],
        cpu_qvel[5],
    )

    # =========================================================================
    # Copy CPU state to GPU
    # =========================================================================

    copy_cpu_state_to_gpu[STATE_SIZE](ctx, states_buf, cpu_qpos, cpu_qvel)

    # Run forward kinematics on GPU to match CPU
    comptime MODEL_SIZE = Hopper[DType.float64].STATE_SIZE  # Approximate
    from physics3d.gpu.constants import model_size

    comptime ACTUAL_MODEL_SIZE = model_size[4, 6]()  # 4 bodies, 6 joints
    var model_buf = ctx.enqueue_create_buffer[gpu_dtype](ACTUAL_MODEL_SIZE)
    Hopper[DType.float64]._init_model_gpu(ctx, model_buf)

    # Run FK on GPU
    from physics3d.kinematics.forward_kinematics import (
        forward_kinematics_gpu,
    )
    from layout import Layout, LayoutTensor
    from gpu import thread_idx, block_idx, block_dim

    var states = LayoutTensor[
        gpu_dtype,
        Layout.row_major(BATCH_SIZE, STATE_SIZE),
        MutAnyOrigin,
    ](states_buf.unsafe_ptr())
    var model = LayoutTensor[
        gpu_dtype, Layout.row_major(1, ACTUAL_MODEL_SIZE), MutAnyOrigin
    ](model_buf.unsafe_ptr())

    @always_inline
    fn fk_wrapper(
        states: LayoutTensor[
            gpu_dtype, Layout.row_major(BATCH_SIZE, STATE_SIZE), MutAnyOrigin
        ],
        model: LayoutTensor[
            gpu_dtype, Layout.row_major(1, ACTUAL_MODEL_SIZE), MutAnyOrigin
        ],
    ):
        forward_kinematics_gpu[
            gpu_dtype,
            NQ,
            NV,
            4,
            6,
            10,
            STATE_SIZE,
            ACTUAL_MODEL_SIZE,
            BATCH_SIZE,
        ](0, states, model)

    ctx.enqueue_function[fk_wrapper, fk_wrapper](
        states,
        model,
        grid_dim=(1,),
        block_dim=(1,),
    )
    ctx.synchronize()

    var gpu_state = extract_gpu_state[STATE_SIZE](ctx, states_buf)
    var gpu_qpos = gpu_state[0].copy()
    var gpu_qvel = gpu_state[1].copy()

    print()
    print("Initial GPU state (after copy from CPU):")
    print(
        "  qpos:",
        gpu_qpos[0],
        gpu_qpos[1],
        gpu_qpos[2],
        gpu_qpos[3],
        gpu_qpos[4],
        gpu_qpos[5],
    )
    print(
        "  qvel:",
        gpu_qvel[0],
        gpu_qvel[1],
        gpu_qvel[2],
        gpu_qvel[3],
        gpu_qvel[4],
        gpu_qvel[5],
    )

    # =========================================================================
    # Run comparison: same policy, same initial state
    # =========================================================================

    print()
    print("=" * 80)
    print("Step-by-step comparison with trained policy")
    print("=" * 80)
    print()
    print(
        "Step | CPU z_pos | GPU z_pos | Diff    | CPU x_vel | GPU x_vel | CPU R"
        "   | GPU R   | CPU done | GPU done"
    )
    print("-" * 120)

    var cpu_total_reward: Float64 = 0.0
    var gpu_total_reward: Float64 = 0.0
    var cpu_done = False
    var gpu_done = False
    var first_divergence_step = -1

    comptime MAX_STEPS = 200  # Run for 200 steps or until done

    for step in range(MAX_STEPS):
        if cpu_done and gpu_done:
            break

        # Get CPU observation
        var cpu_obs_list = cpu_env.get_obs_list()
        var cpu_obs = InlineArray[Scalar[gpu_dtype], OBS_DIM](
            uninitialized=True
        )
        for i in range(OBS_DIM):
            cpu_obs[i] = Scalar[gpu_dtype](cpu_obs_list[i])

        # Get action from policy (deterministic)
        var action_result = agent.select_action(cpu_obs, training=False)
        var actions = action_result[0].copy()

        # Clip actions to [-1, 1]
        var action_list_cpu = List[Scalar[DType.float64]]()
        var action_host = List[Scalar[gpu_dtype]](capacity=ACTION_DIM)
        for j in range(ACTION_DIM):
            var action_val = Float64(actions[j])
            if action_val > 1.0:
                action_val = 1.0
            elif action_val < -1.0:
                action_val = -1.0
            action_list_cpu.append(Scalar[DType.float64](action_val))
            action_host.append(Scalar[gpu_dtype](action_val))

        # Step CPU environment
        var cpu_result = cpu_env.step_continuous_vec(action_list_cpu)
        var cpu_reward = Float64(cpu_result[1])
        cpu_done = cpu_result[2]

        # Copy actions to GPU and step
        ctx.enqueue_copy(actions_buf, action_host.unsafe_ptr())
        ctx.synchronize()

        if not gpu_done:
            Hopper[DType.float64].step_kernel_gpu[
                BATCH_SIZE, STATE_SIZE, OBS_DIM, ACTION_DIM
            ](
                ctx,
                states_buf,
                actions_buf,
                rewards_buf,
                dones_buf,
                obs_buf,
            )
            ctx.synchronize()

        # Get GPU reward and done
        var reward_host = List[Scalar[gpu_dtype]](capacity=1)
        reward_host.append(Scalar[gpu_dtype](0.0))
        var done_host = List[Scalar[gpu_dtype]](capacity=1)
        done_host.append(Scalar[gpu_dtype](0.0))

        ctx.enqueue_copy(reward_host.unsafe_ptr(), rewards_buf)
        ctx.enqueue_copy(done_host.unsafe_ptr(), dones_buf)
        ctx.synchronize()

        var gpu_reward = Float64(reward_host[0])
        gpu_done = done_host[0] > 0.5

        # Get states for comparison
        cpu_qpos = cpu_env.get_qpos()
        cpu_qvel = cpu_env.get_qvel()
        gpu_state = extract_gpu_state[STATE_SIZE](ctx, states_buf)
        gpu_qpos = gpu_state[0].copy()
        gpu_qvel = gpu_state[1].copy()

        if not cpu_done:
            cpu_total_reward += cpu_reward
        if not gpu_done:
            gpu_total_reward += gpu_reward

        # Compute differences
        var z_diff = abs(Float64(cpu_qpos[1]) - Float64(gpu_qpos[1]))

        # Track first significant divergence
        if first_divergence_step < 0 and z_diff > 0.01:
            first_divergence_step = step

        # Print every 10 steps or on significant events
        if step % 10 == 0 or cpu_done or gpu_done or z_diff > 0.05:
            print(
                String(step).rjust(4),
                "|",
                String(Float64(cpu_qpos[1]))[:9].ljust(9),
                "|",
                String(Float64(gpu_qpos[1]))[:9].ljust(9),
                "|",
                String(z_diff)[:7].ljust(7),
                "|",
                String(Float64(cpu_qvel[0]))[:9].ljust(9),
                "|",
                String(Float64(gpu_qvel[0]))[:9].ljust(9),
                "|",
                String(cpu_reward)[:7].ljust(7),
                "|",
                String(gpu_reward)[:7].ljust(7),
                "|",
                cpu_done,
                "|",
                gpu_done,
            )

    print()
    print("=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print()
    print("CPU total reward:", cpu_total_reward)
    print("GPU total reward:", gpu_total_reward)
    print("Reward difference:", gpu_total_reward - cpu_total_reward)
    print()
    print(
        "CPU ended at step:",
        String(cpu_env.current_step) if cpu_done else "still running",
    )
    print("GPU ended:", "Yes" if gpu_done else "No")
    print()

    if first_divergence_step >= 0:
        print(
            "First significant divergence (z_diff > 0.01) at step:",
            first_divergence_step,
        )
    else:
        print("No significant divergence detected")

    # Final state comparison
    print()
    print("Final state comparison:")
    print(
        "  CPU qpos:",
        cpu_qpos[0],
        cpu_qpos[1],
        cpu_qpos[2],
        cpu_qpos[3],
        cpu_qpos[4],
        cpu_qpos[5],
    )
    print(
        "  GPU qpos:",
        gpu_qpos[0],
        gpu_qpos[1],
        gpu_qpos[2],
        gpu_qpos[3],
        gpu_qpos[4],
        gpu_qpos[5],
    )
    print()
    print(
        "  CPU qvel:",
        cpu_qvel[0],
        cpu_qvel[1],
        cpu_qvel[2],
        cpu_qvel[3],
        cpu_qvel[4],
        cpu_qvel[5],
    )
    print(
        "  GPU qvel:",
        gpu_qvel[0],
        gpu_qvel[1],
        gpu_qvel[2],
        gpu_qvel[3],
        gpu_qvel[4],
        gpu_qvel[5],
    )

    print()
    print("=" * 80)
    print("DIAGNOSIS")
    print("=" * 80)

    if cpu_done and not gpu_done:
        print("ISSUE: CPU terminated early while GPU kept running")
        print("  - CPU z_pos at termination:", cpu_qpos[1])
        print("  - GPU z_pos still:", gpu_qpos[1])
        print(
            "  This suggests different physics behavior causing CPU hopper to"
            " fall"
        )
    elif gpu_done and not cpu_done:
        print("ISSUE: GPU terminated early while CPU kept running")
        print("  This is unusual - GPU usually more stable")
    elif cpu_total_reward < gpu_total_reward - 50:
        print("ISSUE: CPU reward significantly lower than GPU")
        print("  This could be due to:")
        print("  1. Different physics causing different velocities")
        print("  2. Different contact handling")
        print("  3. Numerical precision differences accumulating")
    else:
        print("CPU and GPU appear to track reasonably well")

    print()
    print(">>> Comparison completed <<<")
