"""Test PPO Agent GPU Training on LunarLander.

This tests the GPU implementation of PPO using the simplified GPU-compatible
LunarLander environment with:
- Parallel environments on GPU
- Simplified rigid body physics
- Wind/turbulence effects
- Reward shaping for faster learning

Run with:
    pixi run -e apple mojo run tests/test_ppo_lunar_gpu.mojo    # Apple Silicon
    pixi run -e nvidia mojo run tests/test_ppo_lunar_gpu.mojo   # NVIDIA GPU
"""

from random import seed
from time import perf_counter_ns

from gpu.host import DeviceContext

from deep_agents.ppo import DeepPPOAgent
from envs.lunar_lander import LunarLanderV2


# =============================================================================
# Constants
# =============================================================================

# LunarLander: 8D observation, 4 discrete actions
comptime OBS_DIM = 8
comptime NUM_ACTIONS = 4

# Network architecture
comptime HIDDEN_DIM = 300

# GPU training parameters
comptime ROLLOUT_LEN = 128  # Steps per rollout per environment
comptime N_ENVS = 512  # Parallel environments
comptime GPU_MINIBATCH_SIZE = 512  # Minibatch size for PPO updates

# Training duration
comptime NUM_EPISODES = 50_000  # More episodes for LunarLander (harder than CartPole)

comptime dtype = DType.float32

# =============================================================================
# Main
# =============================================================================


fn main() raises:
    seed(42)
    print("=" * 70)
    print("PPO Agent GPU Test on LunarLander")
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
            clip_epsilon=0.2,
            actor_lr=0.0003,
            critic_lr=0.0003,
            entropy_coef=0.01,
            value_loss_coef=0.5,
            num_epochs=10,
            # Advanced hyperparameters
            target_kl=0.02,
            max_grad_norm=0.5,
            anneal_lr=True,
            anneal_entropy=False,
            target_total_steps=0,  # Auto-calculate+
            clip_value=True,
            norm_adv_per_minibatch=True,
            checkpoint_every=1000,
            checkpoint_path="ppo_lunar_gpu_v2.ckpt",
        )

        # agent.load_checkpoint("ppo_lunar_gpu_v2.ckpt")

        print("Environment: LunarLander (GPU)")
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
        print("LunarLander specifics:")
        print(
            "  - 8D observations: [x, y, vx, vy, angle, ang_vel, left_leg,"
            " right_leg]"
        )
        print("  - 4 actions: nop, left, main, right")
        print("  - Wind effects: enabled")
        print(
            "  - Reward shaping: distance + velocity + angle penalties, leg"
            " contact bonus"
        )
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
                + String(metrics.mean_reward_last_n(100))[:7]
            )
            print("Best episode reward: " + String(metrics.max_reward())[:7])
            print()

            # Check for successful training
            var final_avg = metrics.mean_reward_last_n(100)
            if final_avg > 0.0:
                print("SUCCESS: Agent learned to land! (avg reward > 0)")
            elif final_avg > -100.0:
                print(
                    "PARTIAL: Agent is learning but not consistently landing"
                    " (avg reward > -100)"
                )
            else:
                print(
                    "NEEDS MORE TRAINING: Agent still crashing frequently (avg"
                    " reward < -100)"
                )

            print()
            print("=" * 70)

        except e:
            print("!!! EXCEPTION CAUGHT !!!")
            print("Error:", e)
            print("!!! END EXCEPTION !!!")

    print(">>> main() completed normally <<<")
