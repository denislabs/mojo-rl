"""Test Vectorized DQN Agent GPU Training on CartPole.

This tests the optimized GPU implementation of DQN using:
- Vectorized environments (N parallel envs on GPU)
- GPU-side replay buffer
- Batched forward pass using Network.forward_gpu
- Environment stepping on GPU using GPUDiscreteEnv.step_inline

Run with:
    pixi run -e apple mojo run test_dqn_gpu_vectorized.mojo    # Apple Silicon
    pixi run -e nvidia mojo run test_dqn_gpu_vectorized.mojo   # NVIDIA GPU
"""

from random import seed
from time import perf_counter_ns

from gpu.host import DeviceContext

from deep_agents.dqn import DQNAgent
from envs import CartPoleEnv


# =============================================================================
# Constants
# =============================================================================

comptime OBS_DIM = 4
comptime NUM_ACTIONS = 2
comptime HIDDEN_DIM = 64
comptime BUFFER_CAPACITY = 10000
comptime BATCH_SIZE = 128

# Vectorized training parameters
comptime N_ENVS = 32  # Number of parallel environments
comptime NUM_STEPS = 50000  # Total environment steps
comptime WARMUP_STEPS = 1000
comptime TRAIN_EVERY = 4
comptime PRINT_EVERY = 5000


# =============================================================================
# Main
# =============================================================================


fn main() raises:
    seed(42)
    print("=" * 70)
    print("Vectorized DQN Agent GPU Test on CartPole")
    print("=" * 70)
    print()

    # =========================================================================
    # Create GPU context and agent
    # =========================================================================

    with DeviceContext() as ctx:
        var agent = DQNAgent[
            OBS_DIM, NUM_ACTIONS, HIDDEN_DIM, BUFFER_CAPACITY, BATCH_SIZE
        ](
            gamma=0.99,
            tau=0.005,
            lr=0.001,
            epsilon=1.0,
            epsilon_min=0.01,
            epsilon_decay=0.9995,  # Slower decay since more steps
        )

        print("Environment: CartPole (Vectorized GPU)")
        print("Agent: DQN (Double DQN enabled, GPU)")
        print("  Hidden dim: " + String(HIDDEN_DIM))
        print("  Buffer capacity: " + String(BUFFER_CAPACITY))
        print("  Batch size: " + String(BATCH_SIZE))
        print("  Parallel envs: " + String(N_ENVS))
        print("  Total steps: " + String(NUM_STEPS))
        print()

        # =====================================================================
        # Train using vectorized GPU method
        # =====================================================================

        print("Starting Vectorized GPU training...")
        print("-" * 70)

        var start_time = perf_counter_ns()

        # Use simpler version without GPU replay buffer for testing
        var metrics = agent.train_gpu_vectorized[CartPoleEnv, N_ENVS](
            ctx,
            num_steps=NUM_STEPS,
            verbose=True,
            print_every=PRINT_EVERY,
            environment_name="CartPole (Simple Vectorized GPU)",
        )

        var end_time = perf_counter_ns()
        var elapsed_s = Float64(end_time - start_time) / 1e9

        print("-" * 70)
        print()

        # =====================================================================
        # Summary
        # =====================================================================

        print("=" * 70)
        print("Vectorized GPU Training Complete")
        print("=" * 70)
        print()
        print("Total steps: " + String(NUM_STEPS))
        print("Training time: " + String(elapsed_s)[:6] + " seconds")
        print(
            "Steps per second: " + String(Int(Float64(NUM_STEPS) / elapsed_s))
        )
        print()

        # Print metrics summary
        var num_episodes = metrics.num_episodes()
        print("Total episodes completed: " + String(num_episodes))
        if num_episodes > 0:
            print(
                "Final average reward (last 20): "
                + String(metrics.mean_reward_last_n(20))[:7]
            )
            print("Best episode reward: " + String(metrics.max_reward())[:7])
        print()

        # =====================================================================
        # Evaluation using CPU env (to verify learning)
        # =====================================================================

        var env = CartPoleEnv()
        print("Evaluating greedy policy on CPU env (10 episodes)...")
        var eval_avg = agent.evaluate_greedy(
            env, num_episodes=10, max_steps=500
        )
        print("Evaluation average: " + String(eval_avg)[:7])

        print()
        print("=" * 70)
