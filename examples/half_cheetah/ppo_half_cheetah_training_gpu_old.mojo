"""PPO Continuous Agent GPU Training on HalfCheetah (old non-generic agent).

Uses the original DeepPPOContinuousAgent from mojo_rl.deep_agents.ppo
instead of the generic composable agent.

Run with:
    pixi run -e apple mojo run -I . examples/half_cheetah/ppo_half_cheetah_training_gpu_old.mojo
    pixi run -e nvidia mojo run -I . examples/half_cheetah/ppo_half_cheetah_training_gpu_old.mojo
"""

from std.random import seed
from std.time import perf_counter_ns
from std.memory import UnsafePointer

from std.gpu.host import DeviceContext

from mojo_rl.core.dotenv import load_dotenv
from mojo_rl.deep_agents.ppo import DeepPPOContinuousAgent
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
comptime NUM_UPDATES = 500

comptime dtype = DType.float32


# =============================================================================
# Main
# =============================================================================


fn main() raises:
    seed(42)
    print("=" * 70)
    print("PPO Continuous Agent GPU Test on HalfCheetah (Old Agent)")
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
            checkpoint_path="ppo_half_cheetah_old.ckpt",
            normalize_rewards=True,
            obs_noise_std=0.0,
        )

        print("Environment: HalfCheetah Continuous (GPU)")
        print("Agent: PPO Continuous OLD (GPU) - CleanRL hyperparams")
        print("  Observation dim: " + String(OBS_DIM))
        print("  Action dim: " + String(ACTION_DIM))
        print("  Hidden dim: " + String(HIDDEN_DIM))
        print("  Rollout length: " + String(ROLLOUT_LEN))
        print("  N envs (parallel): " + String(N_ENVS))
        print("  Minibatch size: " + String(GPU_MINIBATCH_SIZE))
        print(
            "  Total transitions per rollout: " + String(ROLLOUT_LEN * N_ENVS)
        )
        print()

        # =====================================================================
        # Setup logger — posts to RL Monitor
        # =====================================================================

        var env_vars = load_dotenv()
        var api_key = env_vars.get("RL_MONITOR_API_KEY", "")
        var url = env_vars.get("RL_MONITOR_URL", "")

        var logger = RemoteLogger(
            server_url=url,
            run_name="PPO HalfCheetah GPU (Old)",
            buffer_size=64,
            api_key=api_key,
        )
        logger.set_config("agent", "PPO Continuous (Old)")
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
                num_episodes=NUM_UPDATES,
                verbose=True,
                print_every=10,
                logger=UnsafePointer(to=logger),
                diag_every=1,
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
