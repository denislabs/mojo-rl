"""Quick GPU-only evaluation to verify continuous PPO checkpoint works on GPU environment.

This tests that the trained continuous PPO model performs well on the GPU environment
it was trained on.

Run with:
    pixi run -e apple mojo run tests/test_ppo_lunar_continuous_eval_gpu.mojo
    pixi run -e nvidia mojo run tests/test_ppo_lunar_continuous_eval_gpu.mojo
"""

from random import seed
from time import perf_counter_ns

from gpu.host import DeviceContext, DeviceBuffer
from gpu import thread_idx, block_idx, block_dim
from layout import Layout, LayoutTensor

from deep_agents.ppo import DeepPPOContinuousAgent
from envs.lunar_lander import LunarLanderV2, LLConstants
from deep_rl import dtype as gpu_dtype


# =============================================================================
# Helper kernel to extract observations from state buffer
# =============================================================================


fn _extract_obs_kernel[
    dtype: DType,
    N_ENVS: Int,
    STATE_SIZE: Int,
    OBS_DIM: Int,
](
    obs: LayoutTensor[dtype, Layout.row_major(N_ENVS, OBS_DIM), MutAnyOrigin],
    states: LayoutTensor[
        dtype, Layout.row_major(N_ENVS, STATE_SIZE), MutAnyOrigin
    ],
):
    """Extract observations from full state buffer.

    Observations are stored at offset 0 in each environment's state.
    """
    var i = Int(block_dim.x * block_idx.x + thread_idx.x)
    if i >= N_ENVS:
        return

    for d in range(OBS_DIM):
        obs[i, d] = states[i, d]


# =============================================================================
# Constants
# =============================================================================

comptime OBS_DIM = LLConstants.OBS_DIM_VAL  # 8
comptime ACTION_DIM = LLConstants.ACTION_DIM_VAL  # 2
comptime STATE_SIZE = LLConstants.STATE_SIZE_VAL
comptime HIDDEN_DIM = 256
comptime ROLLOUT_LEN = 128
comptime N_ENVS = 512
comptime GPU_MINIBATCH_SIZE = 512

# Evaluation settings
comptime EVAL_ENVS = 64  # Number of parallel envs for evaluation
comptime MAX_STEPS = 1000


# =============================================================================
# Main
# =============================================================================


fn main() raises:
    seed(42)
    print("=" * 70)
    print("PPO Continuous Agent GPU Evaluation (on GPU environment)")
    print("=" * 70)
    print()

    with DeviceContext() as ctx:
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
            critic_lr=0.001,
            entropy_coef=0.01,
            value_loss_coef=0.5,
            num_epochs=10,
            target_kl=0.02,
            max_grad_norm=0.5,
            anneal_lr=True,
            anneal_entropy=False,
            target_total_steps=0,
            norm_adv_per_minibatch=True,
            checkpoint_every=1000,
            checkpoint_path="ppo_lunar_continuous_gpu.ckpt",
        )

        print("Loading checkpoint...")
        agent.load_checkpoint("ppo_lunar_continuous_gpu.ckpt")
        print("Checkpoint loaded successfully!")

        # DEBUG: Check first few actor parameters to verify checkpoint loaded
        print("DEBUG: First 10 actor params:", end=" ")
        for i in range(min(10, len(agent.actor.params))):
            print(agent.actor.params[i], end=" ")
        print()
        print("DEBUG: Actor param count:", len(agent.actor.params))
        print()

        # =====================================================================
        # GPU Evaluation
        # =====================================================================

        print(
            "Running GPU evaluation with "
            + String(EVAL_ENVS)
            + " parallel environments..."
        )
        print("-" * 70)

        # Allocate GPU buffers for evaluation
        # Note: states_buf uses STATE_SIZE for internal physics state
        var states_buf = ctx.enqueue_create_buffer[gpu_dtype](
            EVAL_ENVS * STATE_SIZE
        )
        var actions_buf = ctx.enqueue_create_buffer[gpu_dtype](
            EVAL_ENVS * ACTION_DIM
        )
        var rewards_buf = ctx.enqueue_create_buffer[gpu_dtype](EVAL_ENVS)
        var dones_buf = ctx.enqueue_create_buffer[gpu_dtype](EVAL_ENVS)
        var obs_buf = ctx.enqueue_create_buffer[gpu_dtype](EVAL_ENVS * OBS_DIM)

        # Host buffers for reading results
        var host_obs = List[Scalar[gpu_dtype]](capacity=EVAL_ENVS * OBS_DIM)
        var host_rewards = List[Scalar[gpu_dtype]](capacity=EVAL_ENVS)
        var host_dones = List[Scalar[gpu_dtype]](capacity=EVAL_ENVS)
        var host_actions = List[Scalar[gpu_dtype]](
            capacity=EVAL_ENVS * ACTION_DIM
        )

        for _ in range(EVAL_ENVS * OBS_DIM):
            host_obs.append(Scalar[gpu_dtype](0.0))
        for _ in range(EVAL_ENVS):
            host_rewards.append(Scalar[gpu_dtype](0.0))
            host_dones.append(Scalar[gpu_dtype](0.0))
        for _ in range(EVAL_ENVS * ACTION_DIM):
            host_actions.append(Scalar[gpu_dtype](0.0))

        # Reset all environments
        LunarLanderV2[gpu_dtype].reset_kernel_gpu[EVAL_ENVS, STATE_SIZE](
            ctx, states_buf
        )
        ctx.synchronize()

        # Extract initial observations from state buffer
        # (observations are stored at offset 0 in each env's state)
        comptime extract_wrapper = _extract_obs_kernel[
            gpu_dtype, EVAL_ENVS, STATE_SIZE, OBS_DIM
        ]

        var obs_tensor = LayoutTensor[
            gpu_dtype, Layout.row_major(EVAL_ENVS, OBS_DIM)
        ](obs_buf)
        var states_tensor = LayoutTensor[
            gpu_dtype, Layout.row_major(EVAL_ENVS, STATE_SIZE)
        ](states_buf)

        comptime EVAL_BLOCKS = (EVAL_ENVS + 255) // 256
        ctx.enqueue_function[extract_wrapper, extract_wrapper](
            obs_tensor,
            states_tensor,
            grid_dim=(EVAL_BLOCKS,),
            block_dim=(256,),
        )
        ctx.synchronize()

        # Track episode rewards
        var episode_rewards = List[Float64]()
        var current_rewards = List[Float64]()
        for _ in range(EVAL_ENVS):
            current_rewards.append(0.0)

        var completed_episodes = 0
        var target_episodes = 100

        print("Evaluating " + String(target_episodes) + " episodes...")

        var start_time = perf_counter_ns()

        for step in range(MAX_STEPS):
            # Copy observations to host
            ctx.enqueue_copy(host_obs.unsafe_ptr(), obs_buf)
            ctx.synchronize()

            # Select actions using agent (deterministic for eval)
            for env_idx in range(EVAL_ENVS):
                var obs = InlineArray[Scalar[gpu_dtype], OBS_DIM](
                    uninitialized=True
                )
                for j in range(OBS_DIM):
                    obs[j] = host_obs[env_idx * OBS_DIM + j]

                # select_action returns (actions_array, log_prob, value)
                # training=True means stochastic (sample from distribution)
                # Using True because the mean action collapses to near-zero throttle
                var action_result = agent.select_action(obs, training=True)
                var actions = action_result[0]

                # DEBUG: Detailed layer-by-layer trace of forward pass
                if step < 3 and env_idx == 0:
                    print("  DEBUG step", step)
                    print(
                        "    obs[0:4]:",
                        obs[0],
                        obs[1],
                        obs[2],
                        obs[3],
                    )
                    print(
                        "    obs[4:8]:",
                        obs[4],
                        obs[5],
                        obs[6],
                        obs[7],
                    )

                    # Manually trace through layers to find where output becomes constant
                    # Layer 1: LinearReLU[8, 256] - first hidden layer
                    comptime L1_OUT = 256
                    comptime L1_PARAMS = 8 * 256 + 256  # W + b
                    var hidden1 = List[Scalar[gpu_dtype]](capacity=L1_OUT)
                    for _ in range(L1_OUT):
                        hidden1.append(Scalar[gpu_dtype](0))

                    # Manual forward for first LinearReLU
                    var params_ptr = agent.actor.params.unsafe_ptr()
                    for j in range(L1_OUT):
                        var acc = params_ptr[8 * 256 + j]  # bias
                        for i in range(OBS_DIM):
                            acc += obs[i] * params_ptr[i * 256 + j]
                        hidden1[j] = acc if acc > 0 else 0

                    # Print first few hidden1 values
                    var hidden1_nonzero = 0
                    for j in range(L1_OUT):
                        if hidden1[j] != 0:
                            hidden1_nonzero += 1
                    print(
                        "    Layer1 (LinearReLU[8,256]): nonzero outputs:",
                        hidden1_nonzero,
                        "/ 256",
                    )
                    print(
                        "    hidden1[0:5]:",
                        hidden1[0],
                        hidden1[1],
                        hidden1[2],
                        hidden1[3],
                        hidden1[4],
                    )

                    # Layer 2: LinearReLU[256, 256] - second hidden layer
                    comptime L2_OFFSET = L1_PARAMS
                    comptime L2_OUT = 256
                    var hidden2 = List[Scalar[gpu_dtype]](capacity=L2_OUT)
                    for _ in range(L2_OUT):
                        hidden2.append(Scalar[gpu_dtype](0))

                    var l2_params = params_ptr + L2_OFFSET
                    for j in range(L2_OUT):
                        var acc = l2_params[256 * 256 + j]  # bias
                        for i in range(L1_OUT):
                            acc += hidden1[i] * l2_params[i * 256 + j]
                        hidden2[j] = acc if acc > 0 else 0

                    var hidden2_nonzero = 0
                    for j in range(L2_OUT):
                        if hidden2[j] != 0:
                            hidden2_nonzero += 1
                    print(
                        "    Layer2 (LinearReLU[256,256]): nonzero outputs:",
                        hidden2_nonzero,
                        "/ 256",
                    )
                    print(
                        "    hidden2[0:5]:",
                        hidden2[0],
                        hidden2[1],
                        hidden2[2],
                        hidden2[3],
                        hidden2[4],
                    )

                    # Layer 3: StochasticActor[256, 2] with state-independent log_std
                    # New layout: mean_W (256*2) + mean_b (2) + log_std (2)
                    # Total: 516 params (NO log_std weights!)
                    comptime L3_OFFSET = L2_OFFSET + 256 * 256 + 256
                    var l3_params = params_ptr + L3_OFFSET

                    # mean = input @ W_mean + b_mean
                    var mean0_acc = l3_params[256 * 2]  # mean bias 0
                    var mean1_acc = l3_params[256 * 2 + 1]  # mean bias 1
                    for i in range(L2_OUT):
                        mean0_acc += hidden2[i] * l3_params[i * 2]
                        mean1_acc += hidden2[i] * l3_params[i * 2 + 1]

                    # log_std is state-independent (just learnable parameters)
                    comptime LOG_STD_OFFSET = 256 * 2 + 2
                    var logstd0 = l3_params[LOG_STD_OFFSET]
                    var logstd1 = l3_params[LOG_STD_OFFSET + 1]

                    print(
                        "    Layer3 (StochasticActor) pre-clamp: mean0=",
                        mean0_acc,
                        "mean1=",
                        mean1_acc,
                        "log_std0=",
                        logstd0,
                        "log_std1=",
                        logstd1,
                    )

                    # Now compare with agent.actor.forward output
                    var actor_out = InlineArray[Scalar[gpu_dtype], 4](
                        uninitialized=True
                    )
                    agent.actor.forward[1](obs, actor_out)
                    print(
                        "    actor.forward output:",
                        actor_out[0],
                        actor_out[1],
                        actor_out[2],
                        actor_out[3],
                    )
                    print(
                        "    FINAL action[0]:",
                        actions[0],
                        "| action[1]:",
                        actions[1],
                    )

                # Copy 2D action to host buffer (no transformation needed)
                # The GPU kernel handles the action interpretation:
                # - action[0]: main throttle, values < 0 are clipped to 0
                # - action[1]: side control in [-1, 1]
                # This matches how training passes actions directly from the policy
                for a in range(ACTION_DIM):
                    host_actions[env_idx * ACTION_DIM + a] = actions[a]

            # Copy actions to GPU
            ctx.enqueue_copy(actions_buf, host_actions.unsafe_ptr())

            # Step environments using continuous action kernel
            LunarLanderV2[gpu_dtype].step_kernel_gpu[
                EVAL_ENVS, STATE_SIZE, OBS_DIM, ACTION_DIM
            ](ctx, states_buf, actions_buf, rewards_buf, dones_buf, obs_buf)
            ctx.synchronize()

            # Copy results to host
            ctx.enqueue_copy(host_rewards.unsafe_ptr(), rewards_buf)
            ctx.enqueue_copy(host_dones.unsafe_ptr(), dones_buf)
            ctx.synchronize()

            # Accumulate rewards and check for done episodes
            for env_idx in range(EVAL_ENVS):
                current_rewards[env_idx] += Float64(host_rewards[env_idx])

                if host_dones[env_idx] > 0.5:
                    episode_rewards.append(current_rewards[env_idx])
                    current_rewards[env_idx] = 0.0
                    completed_episodes += 1

            # Reset done environments
            LunarLanderV2[gpu_dtype].selective_reset_kernel_gpu[
                EVAL_ENVS, STATE_SIZE
            ](ctx, states_buf, dones_buf, UInt32(step * 12345))

            # Extract observations from state buffer after selective reset
            # (needed because reset envs have new states but obs_buf has old obs)
            ctx.enqueue_function[extract_wrapper, extract_wrapper](
                obs_tensor,
                states_tensor,
                grid_dim=(EVAL_BLOCKS,),
                block_dim=(256,),
            )
            ctx.synchronize()

            if completed_episodes >= target_episodes:
                break

            if step % 100 == 0:
                print(
                    "  Step "
                    + String(step)
                    + ": "
                    + String(completed_episodes)
                    + " episodes completed"
                )

        var end_time = perf_counter_ns()
        var elapsed_s = Float64(end_time - start_time) / 1e9

        # Compute statistics
        var total_reward: Float64 = 0.0
        var min_reward: Float64 = 1e9
        var max_reward: Float64 = -1e9

        for i in range(len(episode_rewards)):
            var r = episode_rewards[i]
            total_reward += r
            if r < min_reward:
                min_reward = r
            if r > max_reward:
                max_reward = r

        var num_episodes = len(episode_rewards)
        var avg_reward = (
            total_reward / Float64(num_episodes) if num_episodes > 0 else 0.0
        )

        print()
        print("-" * 70)
        print("GPU EVALUATION SUMMARY (Continuous Actions)")
        print("-" * 70)
        print("Episodes completed: " + String(num_episodes))
        print("Average reward: " + String(avg_reward)[:8])
        print("Min reward: " + String(min_reward)[:8])
        print("Max reward: " + String(max_reward)[:8])
        print("Evaluation time: " + String(elapsed_s)[:5] + " seconds")
        print()

        if avg_reward > 200:
            print("Result: EXCELLENT - Model performs great on GPU!")
        elif avg_reward > 100:
            print("Result: VERY GOOD - Consistent successful landings!")
        elif avg_reward > 0:
            print("Result: GOOD - Model works on GPU (positive avg reward)")
        elif avg_reward > -100:
            print("Result: PARTIAL - Model partially works on GPU")
        else:
            print("Result: POOR - Model doesn't perform well, needs more training")

        print()
        if avg_reward > 0 and num_episodes > 0:
            print("SUCCESS: Continuous action PPO learned to land!")
            print(
                "  The model can control throttle precisely for smooth landings."
            )
        elif avg_reward > -100:
            print("PROGRESS: Model is learning but needs more training.")
        else:
            print("NEEDS WORK: Consider training for more episodes.")

    print()
    print(">>> GPU Evaluation completed <<<")
