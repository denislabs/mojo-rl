"""Test PPO Continuous Agent GPU Training on HalfCheetah.

This tests the GPU implementation of PPO with continuous actions using the
HalfCheetah environment with:
- Parallel environments on GPU
- Generalized Coordinates (GC) physics engine (MuJoCo-style)
- 6D continuous action space (joint torques)
- 17D observation (qpos + qvel excluding rootx and head)

Action space (6D continuous):
- action[0]: back thigh (hip) torque (-1.0 to 1.0) * gear=120
- action[1]: back shin (knee) torque (-1.0 to 1.0) * gear=90
- action[2]: back foot (ankle) torque (-1.0 to 1.0) * gear=60
- action[3]: front thigh (hip) torque (-1.0 to 1.0) * gear=120
- action[4]: front shin (knee) torque (-1.0 to 1.0) * gear=60
- action[5]: front foot (ankle) torque (-1.0 to 1.0) * gear=30

Run with:
    pixi run -e apple mojo run -I . tests/test_ppo_half_cheetah_continuous_gpu.mojo    # Apple Silicon
    pixi run -e nvidia mojo run -I . tests/test_ppo_half_cheetah_continuous_gpu.mojo   # NVIDIA GPU
"""

from std.random import seed
from std.time import perf_counter_ns
from std.memory import UnsafePointer

from std.gpu.host import DeviceContext

from mojo_rl.core.dotenv import load_dotenv
from mojo_rl.deep_agents.core.generic import DeepPPOContinuousAgent
from mojo_rl.envs.half_cheetah import (
    HalfCheetah,
    HalfCheetahConfig,
    HalfCheetahCurriculum,
)
from mojo_rl.core.logger import RemoteLogger


# =============================================================================
# Constants
# =============================================================================

# HalfCheetah: 17D observation, 6D continuous action
comptime OBS_DIM = HalfCheetahConfig.OBS_DIM  # 17
comptime ACTION_DIM = HalfCheetahConfig.ACTION_DIM  # 6

# Network architecture (scaled for GPU)
comptime HIDDEN_DIM = 256  # Larger network for GPU efficiency

# GPU training parameters (GPU-optimized with CleanRL ratios)
comptime ROLLOUT_LEN = 512  # Longer rollouts for better GAE
comptime N_ENVS = 256  # Good GPU parallelism
comptime GPU_MINIBATCH_SIZE = 2048  # Efficient GPU batch size

# Training duration
comptime NUM_EPISODES = 50_000

comptime dtype = DType.float32


# =============================================================================
# Main
# =============================================================================


fn main() raises:
    seed(42)
    print("=" * 70)
    print("PPO Continuous Agent GPU Test on HalfCheetah")
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
            L=RemoteLogger,
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
            checkpoint_every=10,
            checkpoint_path="ppo_half_cheetah.ckpt",
            normalize_rewards=True,
            obs_noise_std=0.0,
        )

        # agent.load_checkpoint("ppo_half_cheetah.ckpt")

        print("Environment: HalfCheetah Continuous (GPU)")
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
        print("HalfCheetah specifics:")
        print("  - Generalized Coordinates (GC) physics engine")
        print("  - MuJoCo-style joint-space dynamics")
        print("  - Semi-implicit Euler integration (symplectic)")
        print("  - 8 bodies: torso, 2 legs (thigh+shin+foot), head")
        print("  - 10 joints: 3 root DOFs + 6 actuated + 1 fixed head")
        print("  - 17D observations: [z_pos, y_angle,")
        print("                       bthigh, bshin, bfoot,")
        print("                       fthigh, fshin, ffoot,")
        print("                       vel_x, vel_z, y_angvel,")
        print("                       bthigh_vel, bshin_vel, bfoot_vel,")
        print("                       fthigh_vel, fshin_vel, ffoot_vel]")
        print("  - 6D continuous actions (joint torques with gear ratios)")
        print("  - Reward: forward_velocity - ctrl_cost - angle_penalty")
        print("  - Anti-flip: angle penalty + unhealthy termination")
        print("  - Curriculum: max_pitch 3.0 → 1.0 rad")
        print()
        print("Expected rewards:")
        print("  - Random policy: ~-100 to -200")
        print("  - Learning policy: > 0")
        print("  - Good policy: > 500")
        print("  - Running well: > 1000")
        print()

        # =====================================================================
        # Setup logger — posts to RL Monitor
        # =====================================================================

        var env_vars = load_dotenv()
        var api_key = env_vars.get("RL_MONITOR_API_KEY", "")
        var url = env_vars.get("RL_MONITOR_URL", "")

        var logger = RemoteLogger(
            server_url=url,
            run_name="PPO HalfCheetah GPU",
            buffer_size=64,
            api_key=api_key,
        )
        logger.set_config("agent", "PPO Continuous")
        logger.set_config("env", "HalfCheetah")
        logger.set_config("hidden_dim", String(HIDDEN_DIM))
        logger.set_config("actor_lr", "3e-4")
        logger.set_config("critic_lr", "3e-4")
        logger.set_config("gamma", "0.99")
        logger.set_config("rollout_len", String(ROLLOUT_LEN))
        logger.set_config("n_envs", String(N_ENVS))
        logger.set_config("minibatch_size", String(GPU_MINIBATCH_SIZE))

        # =====================================================================
        # Train using the train_gpu() method
        # =====================================================================

        print("Starting GPU training...")
        print("-" * 70)

        var start_time = perf_counter_ns()

        try:
            var metrics = agent.train_gpu[
                HalfCheetah[dtype, TERMINATE_ON_UNHEALTHY=False],
            ](
                ctx,
                num_episodes=NUM_EPISODES,
                verbose=True,
                print_every=10,
                logger=UnsafePointer(to=logger),
                diag_every=10,
            )

            var end_time = perf_counter_ns()
            var elapsed_s = Float64(end_time - start_time) / 1e9

            logger.close()

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
