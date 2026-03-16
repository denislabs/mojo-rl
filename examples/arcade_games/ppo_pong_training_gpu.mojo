"""PPO GPU Training on Pong.

Trains a PPO agent on the native Pong environment using GPU-batched
parallel environments. Pong has 3 discrete actions (NOOP, UP, DOWN) and
6D observations (ball_xy, ball_vxy, paddle_y, cpu_paddle_y — all normalized).

Each PPO update collects rollout_len * n_envs = 256 * 128 = 32768 transitions,
then performs num_epochs mini-batch updates on them.

Run with:
    pixi run -e apple mojo run -I . examples/arcade_games/ppo_pong_training_gpu.mojo    # Apple Silicon
    pixi run -e nvidia mojo run -I . examples/arcade_games/ppo_pong_training_gpu.mojo   # NVIDIA GPU
"""

from std.random import seed
from std.time import perf_counter_ns
from std.memory import UnsafePointer

from std.gpu.host import DeviceContext

from mojo_rl.core.dotenv import load_dotenv
from mojo_rl.deep_agents.ppo import DeepPPOAgent
from mojo_rl.envs.arcade_games.pong import PongEnv
from mojo_rl.core.logger import RemoteLogger


# =============================================================================
# Constants
# =============================================================================

# Pong: 6D observation, 3 discrete actions
comptime OBS_DIM = PongEnv[DType.float64].OBS_DIM  # 6
comptime NUM_ACTIONS = PongEnv[DType.float64].NUM_ACTIONS  # 3

# Network architecture
comptime HIDDEN_DIM = 128

# PPO hyperparameters
comptime ROLLOUT_LEN = 256  # Steps per env per update
comptime N_ENVS = 256  # Parallel environments on GPU
comptime GPU_MINIBATCH_SIZE = 2048

# Training duration
# Each update = ROLLOUT_LEN * N_ENVS = 32768 transitions
# 200 updates = ~6.5M total transitions
comptime NUM_UPDATES = 10_000

comptime dtype = DType.float32


# =============================================================================
# Main
# =============================================================================


fn main() raises:
    seed(42)
    print("=" * 70)
    print("PPO GPU Training on Pong")
    print("=" * 70)
    print()

    # =========================================================================
    # Create GPU context and agent
    # =========================================================================

    with DeviceContext() as ctx:
        var agent = DeepPPOAgent[
            obs_dim=OBS_DIM,
            num_actions=NUM_ACTIONS,
            hidden_dim=HIDDEN_DIM,
            rollout_len=ROLLOUT_LEN,
            n_envs=N_ENVS,
            gpu_minibatch_size=GPU_MINIBATCH_SIZE,
            actor_lr=0.0003,
            critic_lr=0.001,
            L=RemoteLogger,
        ](
            gamma=0.99,
            gae_lambda=0.95,
            clip_epsilon=0.2,
            entropy_coef=0.01,
            value_loss_coef=0.5,
            num_epochs=4,
            target_kl=0.015,
            max_grad_norm=0.5,
            anneal_lr=True,
            anneal_entropy=False,
            target_total_steps=0,
            clip_value=True,
            norm_adv_per_minibatch=True,
            checkpoint_every=20,
            checkpoint_path="ppo_pong.ckpt",
        )

        var transitions_per_update = ROLLOUT_LEN * N_ENVS
        var total_transitions = transitions_per_update * NUM_UPDATES

        print("Environment: Pong (GPU-batched)")
        print("Agent: PPO (GPU)")
        print("  Observation dim:", OBS_DIM)
        print("  Actions:", NUM_ACTIONS, "(NOOP, UP, DOWN)")
        print("  Hidden dim:", HIDDEN_DIM)
        print("  Rollout length:", ROLLOUT_LEN)
        print("  N envs (parallel):", N_ENVS)
        print("  Minibatch size:", GPU_MINIBATCH_SIZE)
        print("  Transitions per update:", transitions_per_update)
        print("  Total updates:", NUM_UPDATES)
        print("  Total transitions:", total_transitions)
        print()
        print("  Key hyperparameters:")
        print("    - Actor LR: 3e-4, Critic LR: 1e-3")
        print("    - Entropy coef: 0.01")
        print("    - Update epochs: 4")
        print("    - GAE lambda: 0.95")
        print("    - Clip epsilon: 0.2")
        print("    - LR annealing: enabled")
        print("    - Gradient clipping: max_grad_norm=0.5")
        print()
        print("Pong specifics:")
        print("  - Ball + 2 paddles, 160x210 play area")
        print("  - 6D obs: ball_x/y, ball_vx/vy, paddle_y, cpu_y (normalized)")
        print("  - Reward: +1 score point, -1 opponent scores")
        print("  - Score to 21 wins the game")
        print()
        print("Expected rewards:")
        print("  - Random policy: ~-21 (CPU wins almost every point)")
        print("  - Learning policy: > -10")
        print("  - Good policy: > 0 (beating CPU)")
        print("  - Strong policy: > 10")
        print()

        # =====================================================================
        # Setup logger — posts to RL Monitor
        # =====================================================================

        var env_vars = load_dotenv()
        var api_key = env_vars.get("RL_MONITOR_API_KEY", "")
        var url = env_vars.get("RL_MONITOR_URL", "")

        var logger = RemoteLogger(
            server_url=url,
            run_name="PPO Pong GPU",
            buffer_size=64,
            api_key=api_key,
        )
        logger.set_config("agent", "PPO")
        logger.set_config("env", "Pong")
        logger.set_config("hidden_dim", String(HIDDEN_DIM))
        logger.set_config("actor_lr", "3e-4")
        logger.set_config("critic_lr", "1e-3")
        logger.set_config("gamma", "0.99")
        logger.set_config("rollout_len", String(ROLLOUT_LEN))
        logger.set_config("n_envs", String(N_ENVS))
        logger.set_config("minibatch_size", String(GPU_MINIBATCH_SIZE))

        # =====================================================================
        # Train
        # =====================================================================

        print("Starting GPU training...")
        print("-" * 70)

        var start_time = perf_counter_ns()

        try:
            var metrics = agent.train_gpu[PongEnv[dtype]](
                ctx,
                num_updates=NUM_UPDATES,
                verbose=True,
                print_every=100,
                logger=UnsafePointer(to=logger),
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
            print("Total updates:", NUM_UPDATES)
            print("Total transitions:", total_transitions)
            print("Training time:", String(elapsed_s)[:6], "seconds")
            print(
                "Transitions/second:",
                String(Float64(total_transitions) / elapsed_s)[:9],
            )
            print()

            var final_avg = metrics.mean_reward_last_n(100)
            print(
                "Final average reward (last 100 episodes):",
                String(final_avg)[:8],
            )
            print("Best episode reward:", String(metrics.max_reward())[:8])
            print()

            if final_avg > 10.0:
                print("EXCELLENT: Agent dominates CPU! (avg reward > 10)")
            elif final_avg > 0.0:
                print("SUCCESS: Agent beats CPU! (avg reward > 0)")
            elif final_avg > -10.0:
                print("GOOD PROGRESS: Agent is competitive (avg reward > -10)")
            elif final_avg > -15.0:
                print("LEARNING: Agent improving (avg reward > -15)")
            else:
                print("EARLY STAGE: Agent still exploring (avg reward < -15)")

            print()
            print("=" * 70)

        except e:
            print("!!! EXCEPTION CAUGHT !!!")
            print("Error:", e)
            print("!!! END EXCEPTION !!!")

    print(">>> main() completed normally <<<")
