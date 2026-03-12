"""Test DQN Agent GPU Training on CartPole.

This tests the GPU implementation of DQN using:
- Network wrapper GPU methods (forward_gpu, backward_gpu, update_gpu)
- CPU environment interaction + GPU batch training
- Double DQN with experience replay

Run with:
    pixi run -e apple mojo run -I . test_dqn_gpu.mojo    # Apple Silicon
    pixi run -e nvidia mojo run -I . test_dqn_gpu.mojo   # NVIDIA GPU
"""

from std.random import seed
from std.time import perf_counter_ns

from std.gpu.host import DeviceContext

from mojo_rl.deep_agents.dqn import DQNAgent
from mojo_rl.envs import CartPoleEnv


# =============================================================================
# Constants
# =============================================================================

comptime OBS_DIM = 4
comptime NUM_ACTIONS = 2
comptime HIDDEN_DIM = 120
comptime BUFFER_CAPACITY = 100_000
comptime BATCH_SIZE = 128  # Training batch size for gradient updates
comptime N_ENVS = 256  # Parallel environments for GPU collection

comptime NUM_STEPS = 100_000  # Total env transitions
comptime MAX_STEPS = 500
comptime WARMUP_STEPS = 10_000  # Fill buffer before training
comptime SYNC_EVERY = 10_000  # Sync GPU params to CPU every N transitions


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
        var env = CartPoleEnv[DType.float32]()
        var agent = DQNAgent[
            OBS_DIM,
            NUM_ACTIONS,
            HIDDEN_DIM,
            BUFFER_CAPACITY,
            BATCH_SIZE,
            N_ENVS,
            lr=2.5e-4,
        ](
            gamma=0.99,
            tau=0.005,
            epsilon_min=0.05,
        )

        print("Environment: CartPole")
        print("Agent: DQN (Double DQN enabled, GPU)")
        print("  Hidden dim: " + String(HIDDEN_DIM))
        print("  Buffer capacity: " + String(BUFFER_CAPACITY))
        print("  Batch size (training): " + String(BATCH_SIZE))
        print("  N envs (parallel): " + String(N_ENVS))
        print("  Sync every: " + String(SYNC_EVERY) + " episodes")
        print()

        # =====================================================================
        # Train using the train_gpu() method
        # =====================================================================

        print("Starting GPU training...")
        print("-" * 70)

        var start_time = perf_counter_ns()

        var metrics = agent.train_gpu[CartPoleEnv[DType.float32]](
            ctx,
            num_steps=NUM_STEPS,
            warmup_steps=WARMUP_STEPS,
            gradient_steps=128,
            sync_every=SYNC_EVERY,
            verbose=True,
            print_every=10_000,
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
        print("Total steps: " + String(NUM_STEPS))
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
        var eval_avg = agent.evaluate(
            env, num_episodes=10, max_steps_per_episode=MAX_STEPS, greedy=True
        )
        print("Evaluation average: " + String(eval_avg)[:7])

        print()
        print("Evaluating with current epsilon (10 episodes)...")
        var eval_eps_avg = agent.evaluate(
            env, num_episodes=10, max_steps_per_episode=MAX_STEPS, greedy=False
        )
        print(
            "Evaluation average (epsilon="
            + String(agent.get_epsilon())[:5]
            + "): "
            + String(eval_eps_avg)[:7]
        )

        print()
        print("=" * 70)
