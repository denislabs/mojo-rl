"""DQN GPU Training on Pong.

Trains a Double DQN agent on the native Pong environment using GPU-batched
parallel environments. Pong has 3 discrete actions (NOOP, UP, DOWN) and
6D observations (ball_xy, ball_vxy, paddle_y, cpu_paddle_y — all normalized).

Run with:
    pixi run -e apple mojo run -I . examples/arcade_games/dqn_pong_training_gpu.mojo    # Apple Silicon
    pixi run -e nvidia mojo run -I . examples/arcade_games/dqn_pong_training_gpu.mojo   # NVIDIA GPU
"""

from std.random import seed
from std.time import perf_counter_ns
from std.memory import UnsafePointer

from std.gpu.host import DeviceContext

from mojo_rl.core.dotenv import load_dotenv
from mojo_rl.deep_agents.core.agents import DQNAgent
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

# DQN hyperparameters
comptime BUFFER_CAPACITY = 100_000
comptime BATCH_SIZE = 256
comptime N_ENVS = 256  # Parallel environments on GPU

# Training duration — Pong episodes are long (~2000-5000 steps each)
# so we need many total transitions to see enough episodes
comptime NUM_STEPS = 50_000  # Total env transitions

comptime dtype = DType.float32


# =============================================================================
# Main
# =============================================================================


fn main() raises:
    seed(42)
    print("=" * 70)
    print("DQN (Double) GPU Training on Pong")
    print("=" * 70)
    print()

    # =========================================================================
    # Create GPU context and agent
    # =========================================================================

    with DeviceContext() as ctx:
        var agent = DQNAgent[
            obs_dim=OBS_DIM,
            num_actions=NUM_ACTIONS,
            hidden_dim=HIDDEN_DIM,
            buffer_capacity=BUFFER_CAPACITY,
            batch_size=BATCH_SIZE,
            n_envs=N_ENVS,
            lr=0.0005,
            L=RemoteLogger,
        ](
            gamma=0.99,
            tau=0.005,
            epsilon=1.0,
            epsilon_min=0.02,
            epsilon_decay=0.9995,
            checkpoint_every=50,
            checkpoint_path="dqn_pong.ckpt",
        )

        print("Environment: Pong (GPU-batched)")
        print("Agent: Double DQN (GPU)")
        print("  Observation dim:", OBS_DIM)
        print("  Actions:", NUM_ACTIONS, "(NOOP, UP, DOWN)")
        print("  Hidden dim:", HIDDEN_DIM)
        print("  N envs (parallel):", N_ENVS)
        print("  Buffer capacity:", BUFFER_CAPACITY)
        print("  Batch size:", BATCH_SIZE)
        print("  Learning rate: 5e-4")
        print("  Epsilon: 1.0 → 0.02 (decay=0.9995)")
        print("  Tau (soft update): 0.005")
        print("  Double DQN: enabled")
        print("  Total transitions:", NUM_STEPS)
        print()
        print("Pong specifics:")
        print("  - Ball + 2 paddles, 160×210 play area")
        print("  - 6D obs: ball_x/y, ball_vx/vy, paddle_y, cpu_y (normalized)")
        print("  - Score to 21 wins the game")
        print("  - CPU opponent tracks ball with slight delay")
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
            run_name="DQN Pong GPU",
            buffer_size=64,
            api_key=api_key,
        )
        logger.set_config("agent", "Double DQN")
        logger.set_config("env", "Pong")
        logger.set_config("hidden_dim", String(HIDDEN_DIM))
        logger.set_config("lr", "5e-4")
        logger.set_config("gamma", "0.99")
        logger.set_config("batch_size", String(BATCH_SIZE))
        logger.set_config("n_envs", String(N_ENVS))
        logger.set_config("buffer_capacity", String(BUFFER_CAPACITY))

        # =====================================================================
        # Train
        # =====================================================================

        print("Starting GPU training...")
        print("-" * 70)

        var start_time = perf_counter_ns()

        try:
            var metrics = agent.train_gpu[PongEnv[dtype],](
                ctx,
                num_steps=NUM_STEPS,
                warmup_steps=5000,
                gradient_steps=0,  # 1:1 replay ratio
                sync_every=10_000,
                verbose=True,
                print_every=100_000,
                environment_name="Pong",
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
            print("Total transitions:", NUM_STEPS)
            print("Training time:", String(elapsed_s)[:6], "seconds")
            print(
                "Transitions/second:",
                String(Float64(NUM_STEPS) / elapsed_s)[:9],
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
