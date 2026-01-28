"""Test PPO Continuous Agent GPU Training on HalfCheetah V2.

This tests the GPU implementation of PPO with continuous actions using the
HalfCheetahPlanarV2 environment with:
- Parallel environments on GPU
- Full rigid body physics with motor-enabled joints (physics2d modular)
- 6D continuous action space (back/front leg joint torques)
- 17D observation (torso state + joint angles + velocities)

Action space (6D continuous):
- action[0]: back thigh (hip) torque (-1.0 to 1.0)
- action[1]: back shin (knee) torque (-1.0 to 1.0)
- action[2]: back foot (ankle) torque (-1.0 to 1.0)
- action[3]: front thigh (hip) torque (-1.0 to 1.0)
- action[4]: front shin (knee) torque (-1.0 to 1.0)
- action[5]: front foot (ankle) torque (-1.0 to 1.0)

Run with:
    pixi run -e apple mojo run tests/test_ppo_half_cheetah_continuous_gpu.mojo    # Apple Silicon
    pixi run -e nvidia mojo run tests/test_ppo_half_cheetah_continuous_gpu.mojo   # NVIDIA GPU
"""

from random import seed
from time import perf_counter_ns

from gpu.host import DeviceContext

from deep_agents.ppo import DeepPPOContinuousAgent
from envs.half_cheetah import HalfCheetahPlanarV2, HCConstants


# =============================================================================
# Constants
# =============================================================================

# HalfCheetah: 17D observation, 6D continuous action
comptime OBS_DIM = HCConstants.OBS_DIM_VAL  # 17: torso state + joint angles/velocities
comptime ACTION_DIM = HCConstants.ACTION_DIM_VAL  # 6: [bthigh, bshin, bfoot, fthigh, fshin, ffoot]

# Network architecture (scaled for GPU)
comptime HIDDEN_DIM = 256  # Larger network for GPU efficiency

# GPU training parameters (GPU-optimized with CleanRL ratios)
# CleanRL uses 2048 steps * 1 env = 2048 batch, 32 minibatches = 64 per minibatch
# Scaled for GPU: 256 envs * 512 steps = 131072 batch, ~64 minibatches = 2048 per minibatch
comptime ROLLOUT_LEN = 512  # Longer than before for better GAE
comptime N_ENVS = 256  # Good GPU parallelism
comptime GPU_MINIBATCH_SIZE = 2048  # Efficient GPU batch size

# Training duration (HalfCheetah needs many episodes to learn running)
comptime NUM_EPISODES = 50_000

comptime dtype = DType.float32


# =============================================================================
# Main
# =============================================================================


fn main() raises:
    seed(42)
    print("=" * 70)
    print("PPO Continuous Agent GPU Test on HalfCheetah V2")
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
            actor_lr=0.0003,  # CleanRL: 3e-4 (same for actor and critic)
            critic_lr=0.0003,  # CleanRL: 3e-4 (same as actor!)
            entropy_coef=0.0,  # CleanRL: 0 for MuJoCo (CRITICAL!)
            value_loss_coef=0.5,
            num_epochs=10,  # CleanRL default (was 4)
            # Advanced hyperparameters
            target_kl=0.0,  # KL early stopping disabled
            max_grad_norm=0.5,
            anneal_lr=True,  # CleanRL uses LR annealing
            anneal_entropy=False,
            target_total_steps=0,  # Auto-calculate
            norm_adv_per_minibatch=True,
            checkpoint_every=1_000,
            checkpoint_path="ppo_half_cheetah_cleanrl.ckpt",
            # Reward normalization (CleanRL-style)
            normalize_rewards=True,
            # No observation noise (CleanRL doesn't use this)
            obs_noise_std=0.0,
        )

        # Checkpoint disabled - architecture changed to CleanRL-style
        # agent.load_checkpoint("ppo_half_cheetah_cleanrl.ckpt")

        print("Environment: HalfCheetah V2 Continuous (GPU)")
        print(
            "Agent: PPO Continuous (GPU) - CleanRL hyperparams, GPU-optimized"
        )
        print("  Observation dim: " + String(OBS_DIM))
        print("  Action dim: " + String(ACTION_DIM))
        print("  Hidden dim: " + String(HIDDEN_DIM))
        print("  Rollout length: " + String(ROLLOUT_LEN))
        print("  N envs (parallel): " + String(N_ENVS))
        print("  Minibatch size: " + String(GPU_MINIBATCH_SIZE))
        print(
            "  Total transitions per rollout: " + String(ROLLOUT_LEN * N_ENVS)
        )
        print("  Key CleanRL hyperparameters (GPU-scaled):")
        print("    - Learning rate: 3e-4 (same for actor & critic)")
        print("    - Entropy coef: 0.0 (CRITICAL for MuJoCo)")
        print("    - Update epochs: 10")
        print("    - LR annealing: enabled")
        print("    - Gradient clipping: max_grad_norm=0.5")
        print("    - Reward normalization: enabled")
        print()
        print("HalfCheetah V2 specifics:")
        print("  - 17D observations: [torso_z, torso_angle,")
        print("                       bthigh_angle, bshin_angle, bfoot_angle,")
        print("                       fthigh_angle, fshin_angle, ffoot_angle,")
        print("                       vel_x, vel_z, torso_omega,")
        print("                       bthigh_omega, bshin_omega, bfoot_omega,")
        print("                       fthigh_omega, fshin_omega, ffoot_omega]")
        print("  - 6D continuous actions:")
        print("      action[0]: back thigh (hip) torque (-1.0 to 1.0)")
        print("      action[1]: back shin (knee) torque (-1.0 to 1.0)")
        print("      action[2]: back foot (ankle) torque (-1.0 to 1.0)")
        print("      action[3]: front thigh (hip) torque (-1.0 to 1.0)")
        print("      action[4]: front shin (knee) torque (-1.0 to 1.0)")
        print("      action[5]: front foot (ankle) torque (-1.0 to 1.0)")
        print("  - Reward: forward_velocity - ctrl_cost")
        print("  - No termination on falling (runs for MAX_STEPS)")
        print()
        print("Expected rewards:")
        print("  - Random policy: ~-100 to -200")
        print("  - Learning policy: > 0")
        print("  - Good policy: > 500")
        print("  - Running well: > 1000")
        print()

        # =====================================================================
        # Train using the train_gpu() method
        # =====================================================================

        print("Starting GPU training...")
        print("-" * 70)

        var start_time = perf_counter_ns()

        try:
            var metrics = agent.train_gpu[HalfCheetahPlanarV2[dtype]](
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
            if final_avg > 1000.0:
                print("EXCELLENT: Agent is running fast! (avg reward > 1000)")
            elif final_avg > 500.0:
                print("SUCCESS: Agent learned to run! (avg reward > 500)")
            elif final_avg > 100.0:
                print(
                    "GOOD PROGRESS: Agent is learning locomotion"
                    " (avg reward > 100)"
                )
            elif final_avg > 0.0:
                print(
                    "LEARNING: Agent improving but needs more training"
                    " (avg reward > 0)"
                )
            else:
                print("EARLY STAGE: Agent still exploring (avg reward < 0)")

            print()
            print("=" * 70)

        except e:
            print("!!! EXCEPTION CAUGHT !!!")
            print("Error:", e)
            print("!!! END EXCEPTION !!!")

    print(">>> main() completed normally <<<")
