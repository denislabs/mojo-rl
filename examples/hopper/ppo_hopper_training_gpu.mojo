"""PPO Continuous Agent GPU Training on Hopper.

Diagnostic test: PPO on Hopper to isolate whether the reward collapse
seen with SAC is due to the physics engine or the SAC algorithm.
PPO is on-policy (no replay buffer, no Q-overestimation) so it should
not suffer from the same failure mode.

Run with:
    pixi run -e apple mojo run -I . examples/hopper/ppo_hopper_training_gpu.mojo
    pixi run -e nvidia mojo run -I . examples/hopper/ppo_hopper_training_gpu.mojo
"""

from std.random import seed
from std.time import perf_counter_ns
from std.memory import UnsafePointer

from std.gpu.host import DeviceContext

from mojo_rl.core.dotenv import load_dotenv
from mojo_rl.deep_agents.core.agents import DeepPPOContinuousAgent
from mojo_rl.envs.hopper import Hopper, HopperConfig
from mojo_rl.core.logger import RemoteLogger


# =============================================================================
# Constants
# =============================================================================

# Hopper: 11D observation, 3D continuous action
comptime OBS_DIM = HopperConfig.OBS_DIM  # 11
comptime ACTION_DIM = HopperConfig.ACTION_DIM  # 3

# Network architecture
comptime HIDDEN_DIM = 256

# GPU training parameters (CleanRL-style)
comptime ROLLOUT_LEN = 2048  # CleanRL default for MuJoCo
comptime N_ENVS = 64  # Parallel environments
comptime GPU_MINIBATCH_SIZE = 2048  # Efficient GPU batch size

# Training duration (~2M transitions)
comptime NUM_UPDATES = 500

comptime dtype = DType.float32


# =============================================================================
# Main
# =============================================================================


def main() raises:
    seed(42)
    print("=" * 70)
    print("PPO Continuous Agent GPU Training on Hopper")
    print("=" * 70)
    print()

    with DeviceContext() as ctx:
        var agent = DeepPPOContinuousAgent[
            obs_dim=OBS_DIM,
            action_dim=ACTION_DIM,
            hidden_dim=HIDDEN_DIM,
            rollout_len=ROLLOUT_LEN,
            n_envs=N_ENVS,
            gpu_minibatch_size=GPU_MINIBATCH_SIZE,
            actor_lr=0.0003,
            critic_lr=0.0003,
            L=RemoteLogger,
        ](
            clip_value=True,
            gamma=0.99,
            gae_lambda=0.95,
            clip_epsilon=0.2,
            entropy_coef=0.0,
            value_loss_coef=0.5,
            num_epochs=10,
            target_kl=0.0,
            max_grad_norm=0.5,
            norm_adv_per_minibatch=True,
            checkpoint_every=10,
            checkpoint_path="ppo_hopper.ckpt",
        )

        print("Environment: Hopper Continuous (GPU)")
        print("Agent: PPO Continuous (GPU) - CleanRL hyperparams")
        print("  Observation dim: " + String(OBS_DIM))
        print("  Action dim: " + String(ACTION_DIM))
        print("  Hidden dim: " + String(HIDDEN_DIM))
        print("  Rollout length: " + String(ROLLOUT_LEN))
        print("  N envs (parallel): " + String(N_ENVS))
        print("  Minibatch size: " + String(GPU_MINIBATCH_SIZE))
        print(
            "  Total transitions per update: " + String(ROLLOUT_LEN * N_ENVS)
        )
        print()

        # =====================================================================
        # Setup logger
        # =====================================================================

        var env_vars = load_dotenv()
        var api_key = env_vars.get("RL_MONITOR_API_KEY", "")
        var url = env_vars.get("RL_MONITOR_URL", "")

        var logger = RemoteLogger(
            server_url=url,
            run_name="PPO Hopper GPU",
            buffer_size=64,
            api_key=api_key,
        )
        logger.set_config("agent", "PPO Continuous")
        logger.set_config("env", "Hopper")
        logger.set_config("hidden_dim", String(HIDDEN_DIM))
        logger.set_config("rollout_len", String(ROLLOUT_LEN))
        logger.set_config("n_envs", String(N_ENVS))

        # =====================================================================
        # Train
        # =====================================================================

        print("Starting GPU training...")
        print("-" * 70)

        var start_time = perf_counter_ns()

        try:
            var metrics = agent.train_gpu[
                Hopper[dtype, TERMINATE_ON_UNHEALTHY=True],
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
            print("=" * 70)
            print("GPU Training Complete")
            print("=" * 70)
            print()
            print("Total updates: " + String(NUM_UPDATES))
            print("Training time: " + String(elapsed_s)[byte=:6] + " seconds")
            print()
            print(
                "Final average reward (last 100 episodes): "
                + String(metrics.mean_reward_last_n(100))[byte=:8]
            )
            print(
                "Best episode reward: "
                + String(metrics.max_reward())[byte=:8]
            )
            print()

            var final_avg = metrics.mean_reward_last_n(100)
            if final_avg > 2000.0:
                print("EXCELLENT: Hopping fast! (avg > 2000)")
            elif final_avg > 1000.0:
                print("SUCCESS: Learned to hop! (avg > 1000)")
            elif final_avg > 500.0:
                print("GOOD PROGRESS: Learning locomotion (avg > 500)")
            else:
                print("STILL LEARNING: avg reward = " + String(final_avg)[byte=:8])

            print("=" * 70)

        except e:
            print("!!! EXCEPTION: " + String(e) + " !!!")

    print(">>> main() completed <<<")
