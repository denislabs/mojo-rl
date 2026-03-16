"""Test PPO Continuous Agent GPU Training on Ant.

This tests the GPU implementation of PPO with continuous actions using the
Ant environment with:
- Parallel environments on GPU
- Generalized Coordinates (GC) physics engine (MuJoCo-style)
- 8D continuous action space (joint torques)
- 27D observation (qpos + qvel excluding x,y)

Action space (8D continuous):
- action[0]: hip_4 torque (-1.0 to 1.0) * gear=150
- action[1]: ankle_4 torque (-1.0 to 1.0) * gear=150
- action[2]: hip_1 torque (-1.0 to 1.0) * gear=150
- action[3]: ankle_1 torque (-1.0 to 1.0) * gear=150
- action[4]: hip_2 torque (-1.0 to 1.0) * gear=150
- action[5]: ankle_2 torque (-1.0 to 1.0) * gear=150
- action[6]: hip_3 torque (-1.0 to 1.0) * gear=150
- action[7]: ankle_3 torque (-1.0 to 1.0) * gear=150

Run with:
    pixi run -e apple mojo run -I . tests/test_ppo_ant_continuous_gpu.mojo    # Apple Silicon
    pixi run -e nvidia mojo run -I . tests/test_ppo_ant_continuous_gpu.mojo   # NVIDIA GPU
"""

from std.random import seed
from std.time import perf_counter_ns

from std.gpu.host import DeviceContext

from mojo_rl.deep_agents.core.generic import DeepPPOContinuousAgent
from mojo_rl.envs.ant import (
    Ant,
    AntConfig,
    AntCurriculum,
)


# =============================================================================
# Constants
# =============================================================================

# Ant: 27D observation, 8D continuous action
comptime OBS_DIM = AntConfig.OBS_DIM  # 27
comptime ACTION_DIM = AntConfig.ACTION_DIM  # 8

# Network architecture (scaled for GPU)
comptime HIDDEN_DIM = 256  # Larger network for GPU efficiency

# GPU training parameters (GPU-optimized with CleanRL ratios)
comptime ROLLOUT_LEN = 512  # Longer rollouts for better GAE
comptime N_ENVS = 256  # Good GPU parallelism
comptime GPU_MINIBATCH_SIZE = 2048  # Efficient GPU batch size

# Training duration
comptime NUM_EPISODES = 1_024

comptime dtype = DType.float32


# =============================================================================
# Main
# =============================================================================


fn main() raises:
    seed(42)
    print("=" * 70)
    print("PPO Continuous Agent GPU Test on Ant")
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
            entropy_coef=0.0,  # CleanRL: 0 for MuJoCo
            value_loss_coef=0.5,
            num_epochs=10,  # CleanRL default
            target_kl=0.0,  # KL early stopping disabled
            max_grad_norm=0.5,
            anneal_lr=True,  # CleanRL uses LR annealing
            anneal_entropy=False,
            target_total_steps=0,  # Auto-calculate
            norm_adv_per_minibatch=True,
            checkpoint_every=1_000,
            checkpoint_path="ppo_ant.ckpt",
            normalize_rewards=True,
            obs_noise_std=0.0,
        )

        # agent.load_checkpoint("ppo_ant.ckpt")

        print("Environment: Ant Continuous (GPU)")
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
        print("    - Entropy coef: 0.0 (MuJoCo standard)")
        print("    - Update epochs: 10")
        print("    - LR annealing: enabled")
        print("    - Gradient clipping: max_grad_norm=0.5")
        print("    - Reward normalization: enabled")
        print()
        print("Ant specifics:")
        print("  - Generalized Coordinates (GC) physics engine")
        print("  - MuJoCo-style joint-space dynamics")
        print("  - RK4 integration (matching MuJoCo ant.xml)")
        print("  - 14 bodies: torso + 4 legs (welded_leg + aux + ankle)")
        print("  - 9 joints: 1 free root + 8 hinges (4 hip + 4 ankle)")
        print("  - 27D observations: [z_pos, quat(4), hinge_qpos(8),")
        print(
            "                       vel_xyz(3), angvel_xyz(3), hinge_qvel(8)]"
        )
        print("  - 8D continuous actions (joint torques with gear=150)")
        print("  - Reward: forward_velocity + healthy(1.0) - ctrl_cost(0.5)")
        print("  - Termination: z not in [0.2, 1.0]")
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
            var metrics = agent.train_gpu[
                Ant[dtype, TERMINATE_ON_UNHEALTHY=False],
                # AntCurriculum,
            ](
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
