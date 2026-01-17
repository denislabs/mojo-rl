"""Test PPO Agent GPU Training on CartPole.

This tests the GPU implementation of PPO using:
- Parallel environments on GPU
- Rollout buffer for on-policy data
- GAE (Generalized Advantage Estimation) computed on CPU
- Actor-Critic training with clipped surrogate objective

Run with:
    pixi run -e apple mojo run tests/test_ppo_gpu.mojo    # Apple Silicon
    pixi run -e nvidia mojo run tests/test_ppo_gpu.mojo   # NVIDIA GPU
"""

from random import seed
from time import perf_counter_ns

from gpu.host import DeviceContext

from deep_agents.ppo import DeepPPOAgent
from envs import CartPoleEnv


# =============================================================================
# Constants
# =============================================================================

comptime OBS_DIM = 4
comptime NUM_ACTIONS = 2
comptime HIDDEN_DIM = 64
comptime ROLLOUT_LEN = 128  # Steps per rollout per environment
comptime N_ENVS = 256  # Smaller for testing (use 1024 for full training)
comptime GPU_MINIBATCH_SIZE = 512  # Larger minibatch = fewer kernel launches

comptime NUM_EPISODES = 17_500  # More episodes to reach convergence


# =============================================================================
# Main
# =============================================================================


fn main() raises:
    seed(42)
    print("=" * 70)
    print("PPO Agent GPU Test on CartPole")
    print("=" * 70)
    print()

    # =========================================================================
    # Create GPU context and agent
    # =========================================================================

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
            clip_epsilon=0.2,  # Standard PPO clipping
            actor_lr=0.0003,
            critic_lr=0.001,  # Higher critic LR for faster value fitting
            entropy_coef=0.01,
            value_loss_coef=0.5,
            num_epochs=4,  # Standard PPO epochs
            minibatch_size=GPU_MINIBATCH_SIZE,
            normalize_advantages=True,
            # Advanced hyperparameters
            target_kl=0.02,  # KL threshold for early epoch stopping (set to 0 to disable)
            max_grad_norm=0.5,  # Gradient clipping
            anneal_lr=True,  # Linear LR decay
            anneal_entropy=False,  # Keep exploration constant
            target_total_steps=0,  # Auto-calculate based on num_episodes
        )

        print("Environment: CartPole (GPU)")
        print("Agent: PPO (GPU)")
        print("  Hidden dim: " + String(HIDDEN_DIM))
        print("  Rollout length: " + String(ROLLOUT_LEN))
        print("  N envs (parallel): " + String(N_ENVS))
        print("  Minibatch size: " + String(GPU_MINIBATCH_SIZE))
        print(
            "  Total transitions per rollout: " + String(ROLLOUT_LEN * N_ENVS)
        )
        print("  Advanced features:")
        print("    - LR annealing: enabled")
        print("    - KL early stopping: target_kl=0.02")
        print("    - Gradient clipping: max_grad_norm=0.5")
        print()

        # =====================================================================
        # Train using the train_gpu() method
        # =====================================================================

        print("Starting GPU training...")
        print("-" * 70)

        var start_time = perf_counter_ns()

        var metrics = agent.train_gpu[CartPoleEnv](
            ctx,
            num_episodes=NUM_EPISODES,
            verbose=True,
            print_every=50,
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
        var env = CartPoleEnv[DType.float64]()
        var eval_avg = agent.evaluate(
            env, num_episodes=10, max_steps=500, verbose=False
        )
        print("Evaluation average: " + String(eval_avg)[:7])

        agent.evaluate(
            env, num_episodes=1, max_steps=500, verbose=False, render=True
        )

        print()
        print("=" * 70)
