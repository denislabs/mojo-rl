"""Quick GPU-only evaluation to verify checkpoint works on GPU environment.

This tests that the trained model performs well on the same GPU environment
it was trained on, to confirm the CPU evaluation failure is due to physics mismatch.

Run with:
    pixi run -e apple mojo run tests/test_ppo_lunar_gpu_eval_gpu.mojo
"""

from random import seed
from time import perf_counter_ns

from gpu.host import DeviceContext, DeviceBuffer
from layout import Layout, LayoutTensor

from deep_agents.ppo import DeepPPOAgent
from envs.lunar_lander_gpu import LunarLanderGPU, gpu_dtype


# =============================================================================
# Constants
# =============================================================================

comptime OBS_DIM = 8
comptime NUM_ACTIONS = 4
comptime HIDDEN_DIM = 300
comptime ROLLOUT_LEN = 128
comptime N_ENVS = 256
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
    print("PPO Agent GPU Evaluation (on GPU environment)")
    print("=" * 70)
    print()

    with DeviceContext() as ctx:
        var agent = DeepPPOAgent[
            OBS_DIM,
            NUM_ACTIONS,
            HIDDEN_DIM,
            ROLLOUT_LEN,
            N_ENVS,
            GPU_MINIBATCH_SIZE,
        ](
            gamma=0.99,
            gae_lambda=0.95,
            clip_epsilon=0.2,
            actor_lr=0.0003,
            critic_lr=0.0003,
            entropy_coef=0.01,
            value_loss_coef=0.5,
            num_epochs=10,
            target_kl=0.02,
            max_grad_norm=0.5,
            anneal_lr=True,
            anneal_entropy=False,
            target_total_steps=0,
            clip_value=True,
            norm_adv_per_minibatch=True,
            checkpoint_every=1000,
            checkpoint_path="ppo_lunar_gpu.ckpt",
        )

        print("Loading checkpoint...")
        agent.load_checkpoint("ppo_lunar_gpu.ckpt")
        print("Checkpoint loaded successfully!")
        print()

        # =====================================================================
        # GPU Evaluation
        # =====================================================================

        print("Running GPU evaluation with " + String(EVAL_ENVS) + " parallel environments...")
        print("-" * 70)

        # Allocate GPU buffers for evaluation
        var states_buf = ctx.enqueue_create_buffer[gpu_dtype](EVAL_ENVS * OBS_DIM)
        var actions_buf = ctx.enqueue_create_buffer[gpu_dtype](EVAL_ENVS)
        var rewards_buf = ctx.enqueue_create_buffer[gpu_dtype](EVAL_ENVS)
        var dones_buf = ctx.enqueue_create_buffer[gpu_dtype](EVAL_ENVS)

        # Host buffers for reading results
        var host_states = List[Scalar[gpu_dtype]](capacity=EVAL_ENVS * OBS_DIM)
        var host_rewards = List[Scalar[gpu_dtype]](capacity=EVAL_ENVS)
        var host_dones = List[Scalar[gpu_dtype]](capacity=EVAL_ENVS)
        var host_actions = List[Scalar[gpu_dtype]](capacity=EVAL_ENVS)

        for _ in range(EVAL_ENVS * OBS_DIM):
            host_states.append(Scalar[gpu_dtype](0.0))
        for _ in range(EVAL_ENVS):
            host_rewards.append(Scalar[gpu_dtype](0.0))
            host_dones.append(Scalar[gpu_dtype](0.0))
            host_actions.append(Scalar[gpu_dtype](0.0))

        # Reset all environments
        LunarLanderGPU.reset_kernel_gpu[EVAL_ENVS, OBS_DIM](ctx, states_buf)
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
            # Copy states to host
            ctx.enqueue_copy(host_states.unsafe_ptr(), states_buf)
            ctx.synchronize()

            # Select actions using agent (greedy)
            for env_idx in range(EVAL_ENVS):
                var obs = InlineArray[Scalar[gpu_dtype], OBS_DIM](uninitialized=True)
                for j in range(OBS_DIM):
                    obs[j] = host_states[env_idx * OBS_DIM + j]

                var action_result = agent.select_action(obs, training=False)
                host_actions[env_idx] = Scalar[gpu_dtype](action_result[0])

            # Copy actions to GPU
            ctx.enqueue_copy(actions_buf, host_actions.unsafe_ptr())

            # Step environments
            LunarLanderGPU.step_kernel_gpu[EVAL_ENVS, OBS_DIM](
                ctx, states_buf, actions_buf, rewards_buf, dones_buf
            )
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
            LunarLanderGPU.selective_reset_kernel_gpu[EVAL_ENVS, OBS_DIM](
                ctx, states_buf, dones_buf, UInt64(step * 12345)
            )
            ctx.synchronize()

            if completed_episodes >= target_episodes:
                break

            if step % 100 == 0:
                print("  Step " + String(step) + ": " + String(completed_episodes) + " episodes completed")

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
        var avg_reward = total_reward / Float64(num_episodes) if num_episodes > 0 else 0.0

        print()
        print("-" * 70)
        print("GPU EVALUATION SUMMARY")
        print("-" * 70)
        print("Episodes completed: " + String(num_episodes))
        print("Average reward: " + String(avg_reward)[:8])
        print("Min reward: " + String(min_reward)[:8])
        print("Max reward: " + String(max_reward)[:8])
        print("Evaluation time: " + String(elapsed_s)[:5] + " seconds")
        print()

        if avg_reward > 200:
            print("Result: EXCELLENT - Model performs great on GPU!")
        elif avg_reward > 0:
            print("Result: GOOD - Model works on GPU (positive avg reward)")
        elif avg_reward > -100:
            print("Result: PARTIAL - Model partially works on GPU")
        else:
            print("Result: POOR - Model doesn't perform well even on GPU")

        print()
        if avg_reward > 0 and num_episodes > 0:
            print("CONCLUSION: Physics mismatch confirmed!")
            print("  The model works on GPU but fails on CPU due to different physics.")
        elif avg_reward < -100:
            print("CONCLUSION: Model may need more training or checkpoint issue.")

    print()
    print(">>> GPU Evaluation completed <<<")
