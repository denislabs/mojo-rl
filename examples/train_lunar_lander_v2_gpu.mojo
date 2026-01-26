"""Training script for LunarLanderV2 GPU environment.

This demonstrates the LunarLanderV2 environment which implements GPUDiscreteEnv
trait and uses the physics2d architecture with GPU methods for full physics
simulation on GPU.

Usage:
    pixi run -e apple mojo run examples/train_lunar_lander_v2_gpu.mojo
"""

from time import perf_counter_ns
from random import seed, random_float64
from math import exp, sqrt

from gpu.host import DeviceContext, DeviceBuffer
from layout import Layout, LayoutTensor

from envs.lunar_lander import LunarLanderV2
from physics2d import dtype, TPB


# Training hyperparameters
comptime BATCH_SIZE: Int = 32  # Number of parallel environments (reduced for testing)
comptime STATE_SIZE: Int = LunarLanderV2.STATE_SIZE
comptime OBS_DIM: Int = LunarLanderV2.OBS_DIM
comptime NUM_ACTIONS: Int = LunarLanderV2.NUM_ACTIONS

# REINFORCE hyperparameters
comptime HIDDEN_DIM: Int = 64
comptime LEARNING_RATE: Float64 = 0.001
comptime GAMMA: Float64 = 0.99
comptime STEPS_PER_UPDATE: Int = 128
comptime NUM_UPDATES: Int = 1000
comptime PRINT_EVERY: Int = 10


fn xorshift32(state: UInt32) -> UInt32:
    """Simple xorshift RNG."""
    var x = state
    x ^= x << 13
    x ^= x >> 17
    x ^= x << 5
    return x


fn main() raises:
    seed(42)

    print("=" * 70)
    print("LunarLanderV2 GPU Training - REINFORCE")
    print("=" * 70)
    print()
    print("Configuration:")
    print("  Batch size:", BATCH_SIZE, "parallel environments")
    print("  State size:", STATE_SIZE)
    print("  Observation dim:", OBS_DIM)
    print("  Number of actions:", NUM_ACTIONS)
    print("  Hidden dim:", HIDDEN_DIM)
    print("  Learning rate:", LEARNING_RATE)
    print("  Discount factor:", GAMMA)
    print("  Steps per update:", STEPS_PER_UPDATE)
    print()

    with DeviceContext() as ctx:
        # Allocate GPU buffers
        print("Allocating GPU buffers...")
        var states_buf = ctx.enqueue_create_buffer[dtype](
            BATCH_SIZE * STATE_SIZE
        )
        var actions_buf = ctx.enqueue_create_buffer[dtype](BATCH_SIZE)
        var rewards_buf = ctx.enqueue_create_buffer[dtype](BATCH_SIZE)
        var dones_buf = ctx.enqueue_create_buffer[dtype](BATCH_SIZE)

        # Initialize buffers
        states_buf.enqueue_fill(0)
        actions_buf.enqueue_fill(0)
        rewards_buf.enqueue_fill(0)
        dones_buf.enqueue_fill(0)

        # Reset all environments
        print("Resetting all environments...")
        LunarLanderV2.reset_kernel_gpu[BATCH_SIZE, STATE_SIZE](ctx, states_buf)
        ctx.synchronize()

        # Training statistics
        var total_reward: Float64 = 0.0
        var episode_count: Int = 0
        var step_count: Int = 0
        var best_avg_reward: Float64 = -1000.0

        print()
        print("Starting training...")
        print("-" * 70)

        var rng_state = UInt32(42)
        var start_time = perf_counter_ns()

        for update in range(NUM_UPDATES):
            var update_reward: Float64 = 0.0
            var update_episodes: Int = 0

            for step in range(STEPS_PER_UPDATE):
                # Generate random actions for now (simple baseline)
                # In a real implementation, we'd use a policy network
                with actions_buf.map_to_host() as actions_host:
                    for i in range(BATCH_SIZE):
                        rng_state = xorshift32(rng_state)
                        var action = Int(rng_state % NUM_ACTIONS)
                        actions_host[i] = Scalar[dtype](action)

                # Step all environments
                LunarLanderV2.step_kernel_gpu[BATCH_SIZE, STATE_SIZE](
                    ctx, states_buf, actions_buf, rewards_buf, dones_buf
                )

                # Selective reset for done environments
                rng_state = xorshift32(rng_state)
                LunarLanderV2.selective_reset_kernel_gpu[
                    BATCH_SIZE, STATE_SIZE
                ](ctx, states_buf, dones_buf, UInt64(rng_state))

                ctx.synchronize()

                # Accumulate rewards and count episodes
                with rewards_buf.map_to_host() as rewards_host:
                    for i in range(BATCH_SIZE):
                        update_reward += Float64(rewards_host[i])

                with dones_buf.map_to_host() as dones_host:
                    for i in range(BATCH_SIZE):
                        if dones_host[i] > Scalar[dtype](0.5):
                            update_episodes += 1

                step_count += 1

            total_reward += update_reward
            episode_count += update_episodes

            # Print progress
            if (update + 1) % PRINT_EVERY == 0:
                var elapsed_ns = perf_counter_ns() - start_time
                var elapsed_s = Float64(elapsed_ns) / 1e9
                var steps_per_sec = Float64(step_count * BATCH_SIZE) / elapsed_s

                var avg_reward = update_reward / Float64(
                    STEPS_PER_UPDATE * BATCH_SIZE
                )
                var avg_ep_reward: Float64 = 0.0
                if update_episodes > 0:
                    avg_ep_reward = update_reward / Float64(update_episodes)

                if avg_reward > best_avg_reward:
                    best_avg_reward = avg_reward

                print(
                    "Update",
                    update + 1,
                    "/",
                    NUM_UPDATES,
                    "| Steps:",
                    step_count * BATCH_SIZE,
                    "| Episodes:",
                    episode_count,
                    "| Avg Reward:",
                    avg_reward,
                    "| Steps/s:",
                    Int(steps_per_sec),
                )

        var total_time = Float64(perf_counter_ns() - start_time) / 1e9

        print("-" * 70)
        print()
        print("Training complete!")
        print("  Total steps:", step_count * BATCH_SIZE)
        print("  Total episodes:", episode_count)
        print("  Total time:", total_time, "seconds")
        print(
            "  Throughput:",
            Int(Float64(step_count * BATCH_SIZE) / total_time),
            "steps/s",
        )
        print("  Best avg reward:", best_avg_reward)
        print()

        # Test final performance with visualization of a few episodes
        print("Running final test episodes...")

        # Reset for test
        LunarLanderV2.reset_kernel_gpu[BATCH_SIZE, STATE_SIZE](ctx, states_buf)
        ctx.synchronize()

        var test_rewards = List[Float64]()
        var test_lengths = List[Int]()
        var test_episode_reward: Float64 = 0.0
        var test_episode_length: Int = 0
        var max_test_episodes = 10

        while len(test_rewards) < max_test_episodes:
            # Random actions
            with actions_buf.map_to_host() as actions_host:
                for i in range(BATCH_SIZE):
                    rng_state = xorshift32(rng_state)
                    var action = Int(rng_state % NUM_ACTIONS)
                    actions_host[i] = Scalar[dtype](action)

            LunarLanderV2.step_kernel_gpu[BATCH_SIZE, STATE_SIZE](
                ctx, states_buf, actions_buf, rewards_buf, dones_buf
            )

            LunarLanderV2.selective_reset_kernel_gpu[BATCH_SIZE, STATE_SIZE](
                ctx, states_buf, dones_buf, UInt64(rng_state)
            )

            ctx.synchronize()

            # Check first environment for episode completion
            with rewards_buf.map_to_host() as rewards_host:
                test_episode_reward += Float64(rewards_host[0])

            with dones_buf.map_to_host() as dones_host:
                test_episode_length += 1
                if dones_host[0] > Scalar[dtype](0.5):
                    test_rewards.append(test_episode_reward)
                    test_lengths.append(test_episode_length)
                    test_episode_reward = 0.0
                    test_episode_length = 0

            if test_episode_length > 1000:
                # Truncate long episodes
                test_rewards.append(test_episode_reward)
                test_lengths.append(test_episode_length)
                test_episode_reward = 0.0
                test_episode_length = 0

        print()
        print("Test Episode Results (first", max_test_episodes, "episodes):")
        var total_test_reward: Float64 = 0.0
        var total_test_length: Int = 0
        for i in range(len(test_rewards)):
            print(
                "  Episode",
                i + 1,
                ": Reward =",
                test_rewards[i],
                ", Length =",
                test_lengths[i],
            )
            total_test_reward += test_rewards[i]
            total_test_length += test_lengths[i]

        print()
        print(
            "Average test reward:",
            total_test_reward / Float64(len(test_rewards)),
        )
        print("Average episode length:", total_test_length // len(test_rewards))
        print()
        print("Done!")
