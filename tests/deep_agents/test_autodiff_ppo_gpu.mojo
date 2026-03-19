"""Test Autodiff PPO Agent GPU Training on CartPole.

Same as test_ppo_gpu.mojo but using AutodiffPPOConfig which uses a composed
autodiff graph (CategoricalLogProb → Ratio → ClipSurrogate) for the actor
gradient instead of the manual PolicyGradient kernel.

Run with:
    pixi run -e apple mojo run -I . tests/deep_agents/test_autodiff_ppo_gpu.mojo
    pixi run -e nvidia mojo run -I . tests/deep_agents/test_autodiff_ppo_gpu.mojo
"""

from std.random import seed
from std.time import perf_counter_ns
from std.gpu.host import DeviceContext

from mojo_rl.deep_agents.core.agents import (
    GenericOnPolicyAgent,
    AutodiffPPOConfig,
)
from mojo_rl.envs import CartPoleEnv


# =============================================================================
# Constants
# =============================================================================

comptime OBS_DIM = 4
comptime NUM_ACTIONS = 2
comptime HIDDEN_DIM = 64
comptime ROLLOUT_LEN = 128
comptime N_ENVS = 256
comptime GPU_MINIBATCH_SIZE = 4096

comptime NUM_UPDATES = 1_000


# =============================================================================
# Main
# =============================================================================


fn main() raises:
    seed(42)
    print("=" * 70)
    print("Autodiff PPO Agent GPU Test on CartPole")
    print("=" * 70)
    print()

    with DeviceContext() as ctx:
        var agent = GenericOnPolicyAgent[
            AutodiffPPOConfig[OBS_DIM, NUM_ACTIONS, HIDDEN_DIM, ROLLOUT_LEN],
            N_ENVS,
            GPU_MINIBATCH_SIZE,
        ](
            gamma=0.99,
            gae_lambda=0.95,
            clip_epsilon=0.2,
            entropy_coef=0.01,
            value_loss_coef=0.5,
            num_epochs=4,
            minibatch_size=GPU_MINIBATCH_SIZE,
            normalize_advantages=True,
            target_kl=0.02,
            max_grad_norm=0.5,
        )

        print("Environment: CartPole (GPU)")
        print("Agent: Autodiff PPO (GPU)")
        print("  Actor gradient: CategoricalLogProb → Ratio → ClipSurrogate")
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
        # Train
        # =====================================================================

        print("Starting GPU training...")
        print("-" * 70)

        var start_time = perf_counter_ns()

        var metrics = agent.train_gpu[CartPoleEnv[DType.float32]](
            ctx,
            num_updates=NUM_UPDATES,
            verbose=True,
            print_every=50,
        )

        var end_time = perf_counter_ns()
        var elapsed_s = Float64(end_time - start_time) / 1e9

        print("-" * 70)
        print()

        print("=" * 70)
        print("Autodiff PPO GPU Training Complete")
        print("=" * 70)
        print()
        print("Total updates: " + String(NUM_UPDATES))
        print("Training time: " + String(elapsed_s)[:6] + " seconds")
        print()

        print(
            "Final average reward (last 20 episodes): "
            + String(metrics.mean_reward_last_n(20))[:7]
        )
        print("Best episode reward: " + String(metrics.max_reward())[:7])
        print()

        # =====================================================================
        # Evaluation
        # =====================================================================

        print("Evaluating greedy policy (10 episodes)...")
        var env = CartPoleEnv[DType.float64]()
        var eval_avg = agent.evaluate(
            env, num_episodes=10, max_steps_per_episode=500, verbose=False
        )
        print("Evaluation average: " + String(eval_avg)[:7])

        print()
        print("=" * 70)
