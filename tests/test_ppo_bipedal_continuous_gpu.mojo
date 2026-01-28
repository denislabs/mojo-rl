"""Test PPO Continuous Agent GPU Training on BipedalWalker.

This tests the GPU implementation of PPO with continuous actions using the
BipedalWalkerV2 environment with:
- Parallel environments on GPU
- Full rigid body physics with motor-enabled joints
- 4D continuous action space (hip and knee torques)
- 24D observation (hull state + leg states + lidar)

Action space (matching Gymnasium BipedalWalker-v3 continuous):
- action[0]: hip1 torque (-1.0 to 1.0)
- action[1]: knee1 torque (-1.0 to 1.0)
- action[2]: hip2 torque (-1.0 to 1.0)
- action[3]: knee2 torque (-1.0 to 1.0)

Run with:
    pixi run -e apple mojo run tests/test_ppo_bipedal_continuous_gpu.mojo    # Apple Silicon
    pixi run -e nvidia mojo run tests/test_ppo_bipedal_continuous_gpu.mojo   # NVIDIA GPU
"""

from random import seed
from time import perf_counter_ns

from gpu.host import DeviceContext

from deep_agents.ppo import DeepPPOContinuousAgent
from envs.bipedal_walker import BipedalWalkerV2, BWConstants


# =============================================================================
# Constants
# =============================================================================

# BipedalWalker: 24D observation, 4D continuous action
comptime OBS_DIM = BWConstants.OBS_DIM_VAL  # 24: hull state + leg states + lidar
comptime ACTION_DIM = BWConstants.ACTION_DIM_VAL  # 4: [hip1, knee1, hip2, knee2]

# Network architecture (larger for BipedalWalker - harder task than LunarLander)
comptime HIDDEN_DIM = 512

# GPU training parameters
comptime ROLLOUT_LEN = 256  # Steps per rollout per environment
comptime N_ENVS = 512  # Parallel environments
comptime GPU_MINIBATCH_SIZE = 2048  # Minibatch size for PPO updates

# Training duration (BipedalWalker needs more episodes than LunarLander)
comptime NUM_EPISODES = 100_000

comptime dtype = DType.float32


# =============================================================================
# Main
# =============================================================================


fn main() raises:
    seed(42)
    print("=" * 70)
    print("PPO Continuous Agent GPU Test on BipedalWalker")
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
            actor_lr=0.0001,  # Standard learning rate
            critic_lr=0.001,  # Higher critic LR for faster value learning
            entropy_coef=0.1,  # Higher entropy for exploration in hard task
            value_loss_coef=0.5,
            num_epochs=5,  # More epochs for BipedalWalker
            # Advanced hyperparameters
            target_kl=0.0,  # KL early stopping
            max_grad_norm=0.5,
            anneal_lr=False,  # Disabled - causes late-training collapse
            anneal_entropy=False,
            target_total_steps=0,  # Auto-calculate
            norm_adv_per_minibatch=True,
            checkpoint_every=1_000,
            checkpoint_path="ppo_bipedal_continuous_gpu.ckpt",
            # Reward normalization (CleanRL-style)
            normalize_rewards=True,
            # Observation noise for robustness (domain randomization)
            obs_noise_std=0.01,
        )

        # agent.load_checkpoint("ppo_bipedal_continuous_gpu.ckpt")

        print("Environment: BipedalWalker Continuous (GPU)")
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
        print("    - LR annealing: disabled (prevents late collapse)")
        print("    - KL early stopping: target_kl=0.1")
        print("    - Gradient clipping: max_grad_norm=0.5")
        print("    - Reward normalization: enabled (CleanRL-style)")
        print("    - Higher entropy: 0.02 (for exploration)")
        print()
        print("BipedalWalker specifics:")
        print("  - 24D observations: [hull_angle, hull_ang_vel, vel_x, vel_y,")
        print(
            "                       hip1_angle, hip1_speed, knee1_angle,"
            " knee1_speed, leg1_contact,"
        )
        print(
            "                       hip2_angle, hip2_speed, knee2_angle,"
            " knee2_speed, leg2_contact,"
        )
        print("                       lidar[0-9]]")
        print("  - 4D continuous actions:")
        print("      action[0]: hip1 torque (-1.0 to 1.0)")
        print("      action[1]: knee1 torque (-1.0 to 1.0)")
        print("      action[2]: hip2 torque (-1.0 to 1.0)")
        print("      action[3]: knee2 torque (-1.0 to 1.0)")
        print("  - Reward shaping: forward progress + angle penalty")
        print("  - Termination: hull contacts ground")
        print()
        print("Expected rewards:")
        print("  - Random policy: ~-100 to -300")
        print("  - Learning policy: > -50")
        print("  - Good policy: > 100")
        print("  - Walking well: > 200")
        print()

        # =====================================================================
        # Train using the train_gpu() method
        # =====================================================================

        print("Starting GPU training...")
        print("-" * 70)

        var start_time = perf_counter_ns()

        try:
            var metrics = agent.train_gpu[BipedalWalkerV2[dtype]](
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
            if final_avg > 200.0:
                print(
                    "EXCELLENT: Agent is walking consistently!"
                    " (avg reward > 200)"
                )
            elif final_avg > 100.0:
                print("SUCCESS: Agent learned to walk! (avg reward > 100)")
            elif final_avg > 0.0:
                print(
                    "GOOD PROGRESS: Agent is learning to balance"
                    " (avg reward > 0)"
                )
            elif final_avg > -50.0:
                print(
                    "LEARNING: Agent improving but needs more training"
                    " (avg reward > -50)"
                )
            else:
                print("EARLY STAGE: Agent still exploring (avg reward < -50)")

            print()
            print("=" * 70)

        except e:
            print("!!! EXCEPTION CAUGHT !!!")
            print("Error:", e)
            print("!!! END EXCEPTION !!!")

    print(">>> main() completed normally <<<")
