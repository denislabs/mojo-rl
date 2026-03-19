"""Autodiff PPO Continuous Agent GPU Training on HalfCheetah.

Same as ppo_half_cheetah_training_gpu.mojo but using the autodiff-composed
policy gradient (AutodiffClippedSurrogate) instead of manual gradient code.

The policy gradient backward chains DiffOp vjps:
    ClipSurrogateOp.vjp → grad_ratio
    RatioOp.vjp         → grad_log_prob = grad_ratio * ratio
    (Gaussian log_prob backward for continuous actions)

Note: Continuous PPO uses a Gaussian policy, not CategoricalLogProbOp.
The AutodiffClippedSurrogate handles the ratio + clipping part; the
Gaussian log_prob backward is in the continuous agent's training loop.

Run with:
    pixi run -e nvidia mojo run -I . examples/half_cheetah/autodiff_ppo_half_cheetah_training_gpu.mojo
"""

from std.random import seed
from std.time import perf_counter_ns
from std.memory import UnsafePointer

from std.gpu.host import DeviceContext

from mojo_rl.core.dotenv import load_dotenv
from mojo_rl.core.logger import RemoteLogger
from mojo_rl.deep_agents.core.generic import (
    GenericOnPolicyContinuousAgent,
    ContinuousPPOConfig,
)
from mojo_rl.envs.half_cheetah import (
    HalfCheetah,
    HalfCheetahConfig,
    HalfCheetahCurriculum,
)


# =============================================================================
# Constants
# =============================================================================

# HalfCheetah: 17D observation, 6D continuous action
comptime OBS_DIM = HalfCheetahConfig.OBS_DIM  # 17
comptime ACTION_DIM = HalfCheetahConfig.ACTION_DIM  # 6

# Network architecture (scaled for GPU)
comptime HIDDEN_DIM = 256

# GPU training parameters (CleanRL ratios)
comptime ROLLOUT_LEN = 512
comptime N_ENVS = 256
comptime GPU_MINIBATCH_SIZE = 2048

# Training duration
comptime NUM_UPDATES = 500

comptime dtype = DType.float32


# =============================================================================
# Main
# =============================================================================


fn main() raises:
    seed(42)
    print("=" * 70)
    print("Autodiff PPO Continuous Agent GPU Training on HalfCheetah")
    print("=" * 70)
    print()

    # Note: Continuous PPO uses ContinuousPPOConfig which already has
    # ClippedSurrogate for the discrete policy gradient part.
    # The continuous actor uses Gaussian log_prob which is handled
    # separately in the continuous agent's training loop.
    # For a full autodiff continuous PPO, we would need a
    # GaussianLogProbOp — currently using the standard config.

    with DeviceContext() as ctx:
        var agent = GenericOnPolicyContinuousAgent[
            ContinuousPPOConfig[OBS_DIM, ACTION_DIM, HIDDEN_DIM, ROLLOUT_LEN],
            N_ENVS,
            GPU_MINIBATCH_SIZE,
            RemoteLogger,
        ](
            clip_value=True,
            gamma=0.99,
            gae_lambda=0.95,
            clip_epsilon=0.2,
            entropy_coef=0.0,  # CleanRL: 0 for MuJoCo
            value_loss_coef=0.5,
            num_epochs=10,
            target_kl=0.0,
            max_grad_norm=0.5,
            norm_adv_per_minibatch=True,
            checkpoint_every=10,
            checkpoint_path="autodiff_ppo_half_cheetah.ckpt",
        )

        print("Environment: HalfCheetah Continuous (GPU)")
        print("Agent: PPO Continuous (GPU) - CleanRL hyperparams")
        print("  Policy gradient: ClippedSurrogate (standard)")
        print("  Note: Full autodiff continuous PPO needs GaussianLogProbOp")
        print()
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
        print("    - Gradient clipping: max_grad_norm=0.5")
        print()

        # =====================================================================
        # Setup logger
        # =====================================================================

        var env_vars = load_dotenv()
        var api_key = env_vars.get("RL_MONITOR_API_KEY", "")
        var url = env_vars.get("RL_MONITOR_URL", "")

        var logger = RemoteLogger(
            server_url=url,
            run_name="Autodiff PPO HalfCheetah GPU",
            buffer_size=64,
            api_key=api_key,
        )
        logger.set_config("agent", "PPO Continuous (Autodiff)")
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
                num_updates=NUM_UPDATES,
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

            print("=" * 70)
            print("Autodiff PPO GPU Training Complete")
            print("=" * 70)
            print()
            print("Total updates: " + String(NUM_UPDATES))
            print("Training time: " + String(elapsed_s)[:6] + " seconds")
            print(
                "Updates/second: "
                + String(Float64(NUM_UPDATES) / elapsed_s)[:7]
            )
            print()

            print(
                "Final average reward (last 100 episodes): "
                + String(metrics.mean_reward_last_n(100))[:8]
            )
            print("Best episode reward: " + String(metrics.max_reward())[:8])
            print()

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
