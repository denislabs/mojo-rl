"""Test DQN Agent GPU Training on CartPole.

This tests the GPU implementation of DQN using:
- Network wrapper GPU methods (forward_gpu, backward_gpu, update_gpu)
- CPU environment interaction + GPU batch training
- Double DQN with experience replay

Run with:
    pixi run -e apple mojo run test_dqn_gpu.mojo    # Apple Silicon
    pixi run -e nvidia mojo run test_dqn_gpu.mojo   # NVIDIA GPU
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
comptime BATCH_SIZE = 128  # Larger batch for better GPU utilization

comptime NUM_EPISODES = 500  # More episodes for better training
comptime MAX_STEPS = 500
comptime WARMUP_STEPS = 500
comptime TRAIN_EVERY = 1  # Train every iteration for faster learning
comptime SYNC_EVERY = 10  # Sync GPU params to CPU every N episodes


# =============================================================================
# Main
# =============================================================================


fn main() raises:
    seed(42)
    print("=" * 70)
    print("DQN Agent GPU Test on CartPole")
    print("=" * 70)
    print()

    # =========================================================================
    # Create GPU context, environment and agent
    # =========================================================================

    with DeviceContext() as ctx:
        var env = CartPoleEnv()
        var agent = DQNAgent[
            OBS_DIM, NUM_ACTIONS, HIDDEN_DIM, BUFFER_CAPACITY, BATCH_SIZE
        ](
            gamma=0.99,
            tau=0.005,
            lr=0.001,
            epsilon=1.0,
            epsilon_min=0.01,
            epsilon_decay=0.995,
        )

        print("Environment: CartPole")
        print("Agent: DQN (Double DQN enabled, GPU)")
        print("  Hidden dim: " + String(HIDDEN_DIM))
        print("  Buffer capacity: " + String(BUFFER_CAPACITY))
        print("  Batch size: " + String(BATCH_SIZE))
        print("  Sync every: " + String(SYNC_EVERY) + " episodes")
        print()

        # =====================================================================
        # Train using the train_gpu() method
        # =====================================================================

        print("Starting GPU training...")
        print("-" * 70)

        var start_time = perf_counter_ns()

        var metrics = agent.train_gpu_full(
            ctx,
            env,
            num_episodes=NUM_EPISODES,
            max_steps_per_episode=MAX_STEPS,
            warmup_steps=WARMUP_STEPS,
            train_every=TRAIN_EVERY,
            sync_every=SYNC_EVERY,
            verbose=True,
            print_every=10,
            environment_name="CartPole (GPU)",
        )

        var end_time = perf_counter_ns()
        var elapsed_s = Float64(end_time - start_time) / 1e9

        print("-" * 70)
        print()

        # =====================================================================
        # Summary
        # =====================================================================

        print("=" * 70)
        print("GPU Training Complete")
        print("=" * 70)
        print()
        print("Total episodes: " + String(NUM_EPISODES))
        print("Total train steps: " + String(agent.get_train_steps()))
        print("Training time: " + String(elapsed_s)[:6] + " seconds")
        print()

        # Print metrics summary
        print(
            "Final average reward (last 20 episodes): "
            + String(metrics.mean_reward_last_n(20))[:7]
        )
        print("Best episode reward: " + String(metrics.max_reward())[:7])
        print()

        # =====================================================================
        # Evaluation (greedy policy, on CPU with GPU-trained params)
        # =====================================================================

        print("Evaluating greedy policy (10 episodes)...")
        var eval_avg = agent.evaluate_greedy(
            env, num_episodes=10, max_steps=MAX_STEPS
        )
        print("Evaluation average: " + String(eval_avg)[:7])

        print()
        print("Evaluating with current epsilon (10 episodes)...")
        var eval_eps_avg = agent.evaluate(
            env, num_episodes=10, max_steps=MAX_STEPS
        )
        print(
            "Evaluation average (epsilon="
            + String(agent.get_epsilon())[:5]
            + "): "
            + String(eval_eps_avg)[:7]
        )

        print()
        print("=" * 70)
