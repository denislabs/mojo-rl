"""Test PPO Continuous Agent GPU Training on LunarLander.

This tests the GPU implementation of PPO with continuous actions using the
LunarLanderV2 environment with:
- Parallel environments on GPU
- Full rigid body physics with joints
- 2D continuous action space (main throttle + side control)
- Reward shaping for faster learning

Action space (matching Gymnasium LunarLander-v3 continuous):
- action[0]: main engine throttle (0.0 to 1.0)
- action[1]: side engine control (-1.0 to 1.0)
            negative = left engine, positive = right engine
            magnitude > 0.5 activates the engine

Run with:
    pixi run -e apple mojo run tests/test_ppo_lunar_continuous_gpu.mojo    # Apple Silicon
    pixi run -e nvidia mojo run tests/test_ppo_lunar_continuous_gpu.mojo   # NVIDIA GPU
"""

from random import seed
from time import perf_counter_ns

from gpu.host import DeviceContext

from deep_agents.ppo import DeepPPOContinuousAgent
from envs.lunar_lander import LunarLanderV2, LLConstants


# =============================================================================
# Constants
# =============================================================================

# LunarLander: 8D observation, 2D continuous action
comptime OBS_DIM = LLConstants.OBS_DIM_VAL  # 8: [x, y, vx, vy, angle, ang_vel, left_leg, right_leg]
comptime ACTION_DIM = LLConstants.ACTION_DIM_VAL  # 2: [main_throttle, side_control]

# Network architecture (larger for LunarLander - more complex than Pendulum)
comptime HIDDEN_DIM = 256

# GPU training parameters
comptime ROLLOUT_LEN = 128  # Steps per rollout per environment
comptime N_ENVS = 512  # Parallel environments
comptime GPU_MINIBATCH_SIZE = 512  # Minibatch size for PPO updates

# Training duration
comptime NUM_EPISODES = 25_000  # LunarLander needs more episodes than Pendulum

comptime dtype = DType.float32


# =============================================================================
# Main
# =============================================================================


fn main() raises:
    seed(42)
    print("=" * 70)
    print("PPO Continuous Agent GPU Test on LunarLander")
    print("=" * 70)
    print()

    # =========================================================================
    # Create GPU context and agent
    # =========================================================================

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
            gamma=0.99,  # Standard discount
            gae_lambda=0.95,  # Standard GAE lambda
            clip_epsilon=0.2,  # Standard clipping
            actor_lr=0.0003,  # Standard learning rate
            critic_lr=0.001,  # Higher critic LR for faster value learning
            entropy_coef=0.01,  # Entropy for exploration
            value_loss_coef=0.5,
            num_epochs=10,  # More epochs for LunarLander
            # Advanced hyperparameters
            target_kl=0.02,  # KL early stopping
            max_grad_norm=0.5,
            anneal_lr=True,  # Enable LR annealing
            anneal_entropy=False,
            target_total_steps=0,  # Auto-calculate
            norm_adv_per_minibatch=True,
            checkpoint_every=1000,
            checkpoint_path="ppo_lunar_continuous_gpu.ckpt",
            # Action scaling: PPO outputs [-1, 1], we need [0, 1] for main and [-1, 1] for side
            # The environment handles this internally via step_continuous_vec
        )

        print("Environment: LunarLander Continuous (GPU)")
        print("Agent: PPO Continuous (GPU)")
        print("  Observation dim: " + String(OBS_DIM))
        print("  Action dim: " + String(ACTION_DIM))
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
        print("LunarLander Continuous specifics:")
        print(
            "  - 8D observations: [x, y, vx, vy, angle, ang_vel, left_leg,"
            " right_leg]"
        )
        print("  - 2D continuous actions:")
        print("      action[0]: main engine throttle (0.0 to 1.0)")
        print("      action[1]: side engine control (-1.0 to 1.0)")
        print("                 negative = left, positive = right")
        print("  - Reward shaping: distance + velocity + angle penalties")
        print("  - Landing bonus: +100, Crash penalty: -100")
        print("  - Fuel costs: proportional to throttle")
        print()
        print("Expected rewards:")
        print("  - Random policy: ~-200 to -400")
        print("  - Learning policy: > -100")
        print("  - Good policy: > 0")
        print("  - Successful landing: > 100")
        print()

        # =====================================================================
        # Train using the train_gpu() method
        # =====================================================================

        print("Starting GPU training...")
        print("-" * 70)

        var start_time = perf_counter_ns()

        try:
            var metrics = agent.train_gpu[LunarLanderV2[dtype]](
                ctx,
                num_episodes=NUM_EPISODES,
                verbose=True,
                print_every=1,
            )

            var end_time = perf_counter_ns()
            var elapsed_s = Float64(end_time - start_time) / 1e9

            print("-" * 70)
            print()
            print(">>> train_gpu returned successfully! <<<")

            # =================================================================
            # Summary
            # =================================================================

            print("=" * 70)
            print("GPU Training Complete")
            print("=" * 70)
            print()
            print("Total episodes: " + String(NUM_EPISODES))
            print("Training time: " + String(elapsed_s)[:6] + " seconds")
            print(
                "Episodes/second: "
                + String(Float64(NUM_EPISODES) / elapsed_s)[:7]
            )
            print()

            # Print metrics summary
            print(
                "Final average reward (last 100 episodes): "
                + String(metrics.mean_reward_last_n(100))[:8]
            )
            print("Best episode reward: " + String(metrics.max_reward())[:8])
            print()

            # Check for successful training
            var final_avg = metrics.mean_reward_last_n(100)
            if final_avg > 100.0:
                print(
                    "EXCELLENT: Agent consistently lands successfully!"
                    " (avg reward > 100)"
                )
            elif final_avg > 0.0:
                print("SUCCESS: Agent learned to land! (avg reward > 0)")
            elif final_avg > -100.0:
                print(
                    "GOOD PROGRESS: Agent is learning to control"
                    " (avg reward > -100)"
                )
            elif final_avg > -200.0:
                print(
                    "LEARNING: Agent improving but needs more training"
                    " (avg reward > -200)"
                )
            else:
                print("EARLY STAGE: Agent still exploring (avg reward < -200)")

            print()
            print("=" * 70)

        except e:
            print("!!! EXCEPTION CAUGHT !!!")
            print("Error:", e)
            print("!!! END EXCEPTION !!!")

    print(">>> main() completed normally <<<")
