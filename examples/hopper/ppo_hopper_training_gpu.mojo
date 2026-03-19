"""Test PPO Continuous Agent GPU Training on Hopper.

This tests the GPU implementation of PPO with continuous actions using the
Hopper environment with:
- Parallel environments on GPU
- Generalized Coordinates physics with SemiImplicitEulerIntegrator
- 3D continuous action space (joint torques)
- 11D observation (matching MuJoCo Hopper)

Action space (3D continuous):
- action[0]: thigh torque (-1.0 to 1.0)
- action[1]: leg torque (-1.0 to 1.0)
- action[2]: foot torque (-1.0 to 1.0)

Run with:
    pixi run -e apple mojo run -I . tests/test_ppo_hopper_continuous_gpu.mojo    # Apple Silicon
    pixi run -e nvidia mojo run -I . tests/test_ppo_hopper_continuous_gpu.mojo   # NVIDIA GPU
"""

from std.random import seed
from std.time import perf_counter_ns

from std.gpu.host import DeviceContext

from mojo_rl.deep_agents.core.agents import DeepPPOContinuousAgent
from mojo_rl.envs.hopper import Hopper, HopperCurriculum, HopperConfig


# =============================================================================
# Constants
# =============================================================================

# Hopper: 11D observation, 3D continuous action
comptime OBS_DIM = HopperConfig.OBS_DIM  # 11
comptime ACTION_DIM = HopperConfig.ACTION_DIM  # 3

# Network architecture (scaled for GPU)
comptime HIDDEN_DIM = 256  # Larger network for GPU efficiency

# GPU training parameters (GPU-optimized with CleanRL ratios)
comptime ROLLOUT_LEN = 512  # Longer rollouts for better GAE
comptime N_ENVS = 256  # Good GPU parallelism
comptime GPU_MINIBATCH_SIZE = 2048  # Efficient GPU batch size

# Training duration
comptime NUM_UPDATES = 500

comptime dtype = DType.float32


# =============================================================================
# Main
# =============================================================================


fn main() raises:
    seed(42)
    print("=" * 70)
    print("PPO Continuous Agent GPU Test on Hopper")
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
            actor_lr=0.0003,  # CleanRL: 3e-4
            critic_lr=0.0003,  # CleanRL: 3e-4
        ](
            clip_value=True,
            gamma=0.99,  # Standard discount
            gae_lambda=0.95,  # Standard GAE lambda
            clip_epsilon=0.2,  # Standard clipping
            entropy_coef=0.02,  # Small entropy for exploration (was 0.0)
            value_loss_coef=0.5,
            num_epochs=10,  # CleanRL default
            target_kl=0.015,  # KL early stopping
            max_grad_norm=0.5,
            norm_adv_per_minibatch=True,
            checkpoint_every=10,
            checkpoint_path="ppo_hopper.ckpt",
        )

        # agent.load_checkpoint("ppo_hopper.ckpt")

        print("Environment: Hopper Continuous (GPU)")
        print("Agent: PPO Continuous (GPU) - CleanRL hyperparams")
        print("  Observation dim: " + String(OBS_DIM))
        print("  Action dim: " + String(ACTION_DIM))
        print("  Hidden dim: " + String(HIDDEN_DIM))
        print("  Rollout length: " + String(ROLLOUT_LEN))
        print("  N envs (parallel): " + String(N_ENVS))
        print("  Minibatch size: " + String(GPU_MINIBATCH_SIZE))
        print(
            "  Total transitions per rollout: " + String(ROLLOUT_LEN * N_ENVS)
        )
        print("  Key hyperparameters:")
        print("    - Learning rate: 3e-4 (same for actor & critic)")
        print("    - Entropy coef: 0.01 (for exploration, annealed to 0)")
        print("    - Update epochs: 10")
        print("    - LR annealing: enabled")
        print("    - Entropy annealing: enabled")
        print("    - Gradient clipping: max_grad_norm=0.5")
        print("    - Reward normalization: enabled")
        print("    - Reset noise: enabled (±0.005 on qpos/qvel)")
        print()
        print("Hopper specifics:")
        print("  - Generalized Coordinates physics (MuJoCo-style)")
        print("  - SemiImplicitEulerIntegrator (symplectic, energy-conserving)")
        print("  - 4-body articulated hopper (torso, thigh, leg, foot)")
        print("  - 6 DOF: rootx (slide), rootz (slide), rooty (hinge),")
        print("           thigh (hinge), leg (hinge), foot (hinge)")
        print("  - 11D observations: [rootz, rooty, thigh, leg, foot,")
        print("                       vx, vz, omega_y, omega_thigh,")
        print("                       omega_leg, omega_foot]")
        print("  - 3D continuous actions (thigh, leg, foot torques)")
        print("  - Reward: forward_velocity + alive_bonus - ctrl_cost")
        print("  - Terminates on: torso_z < 0.7 or |pitch| > 0.2 rad")
        print()
        print("Expected rewards:")
        print("  - Random policy: ~-500 to -100")
        print("  - Learning policy: > 0")
        print("  - Good policy: > 500")
        print("  - Hopping well: > 1000")
        print()

        # =====================================================================
        # Train using the train_gpu() method
        # =====================================================================

        print("Starting GPU training...")
        print("-" * 70)

        var start_time = perf_counter_ns()

        try:
            var metrics = agent.train_gpu[
                Hopper[dtype, TERMINATE_ON_UNHEALTHY=True], HopperCurriculum
            ](
                ctx,
                num_updates=NUM_UPDATES,
                verbose=True,
                print_every=10,
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
            print("Total updates: " + String(NUM_UPDATES))
            print("Training time: " + String(elapsed_s)[:6] + " seconds")
            print(
                "Updates/second: "
                + String(Float64(NUM_UPDATES) / elapsed_s)[:7]
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
                print("EXCELLENT: Agent is hopping well! (avg reward > 1000)")
            elif final_avg > 500.0:
                print("SUCCESS: Agent learned to hop! (avg reward > 500)")
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
