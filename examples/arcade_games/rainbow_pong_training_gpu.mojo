"""Rainbow DQN GPU Training on Pong.

Trains a Rainbow DQN agent (C51 + Double DQN + PER + Dueling + Noisy + N-step)
on the native Pong environment using GPU-batched parallel environments.

Pong has 3 discrete actions (NOOP, UP, DOWN) and 6D observations
(ball_xy, ball_vxy, paddle_y, cpu_paddle_y — all normalized).

Run with:
    pixi run -e apple mojo run -I . examples/arcade_games/rainbow_pong_training_gpu.mojo    # Apple Silicon
    pixi run -e nvidia mojo run -I . examples/arcade_games/rainbow_pong_training_gpu.mojo   # NVIDIA GPU
"""

from std.random import seed
from std.time import perf_counter_ns
from std.memory import UnsafePointer

from std.gpu.host import DeviceContext

from mojo_rl.core.dotenv import load_dotenv
from mojo_rl.deep_agents.core.agents import RainbowConfig, GenericRainbowAgent
from mojo_rl.envs.arcade_games.pong import PongEnv
from mojo_rl.core.logger import RemoteLogger


# =============================================================================
# Constants
# =============================================================================

# Pong: 6D observation, 3 discrete actions
comptime OBS_DIM = PongEnv[DType.float64].OBS_DIM  # 6
comptime NUM_ACTIONS = PongEnv[DType.float64].NUM_ACTIONS  # 3

# Rainbow hyperparameters
comptime HIDDEN_DIM = 128
comptime STREAM_HIDDEN = 128
comptime NUM_ATOMS = 51
comptime N_STEP = 3
comptime BUFFER_CAPACITY = 1_000_000
comptime BATCH_SIZE = 64
comptime N_ENVS = 256

# Training duration
comptime NUM_STEPS = 2_000_000

comptime dtype = DType.float32


# =============================================================================
# Main
# =============================================================================


def main() raises:
    seed(42)
    print("=" * 70)
    print("Rainbow DQN GPU Training on Pong")
    print("=" * 70)
    print()

    with DeviceContext() as ctx:
        var agent = GenericRainbowAgent[
            RainbowConfig[
                OBS_DIM,
                NUM_ACTIONS,
                NUM_ATOMS,  # 51 atoms
                -21.0,  # v_min (Pong score range)
                21.0,  # v_max
                HIDDEN_DIM,
                STREAM_HIDDEN,
                N_STEP,  # 3-step returns
                BUFFER_CAPACITY,
                BATCH_SIZE,
                6.25e-5,  # lr (Rainbow paper default)
            ],
            N_ENVS,
            RemoteLogger,
        ](
            gamma=0.99,
            tau=0.005,
            target_update_freq=500,
            alpha=0.5,
            beta=0.4,
            beta_frames=100_000,
        )

        print("Environment: Pong (GPU-batched)")
        print("Agent: Rainbow DQN (GPU)")
        print("  Components: C51 + Double DQN + PER + Dueling + Noisy + 3-step")
        print("  Observation dim:", OBS_DIM)
        print("  Actions:", NUM_ACTIONS, "(NOOP, UP, DOWN)")
        print("  Hidden dim:", HIDDEN_DIM)
        print("  Dueling stream hidden:", STREAM_HIDDEN)
        print("  Atoms:", NUM_ATOMS, "support [-21, 21]")
        print("  N-step:", N_STEP)
        print("  N envs (parallel):", N_ENVS)
        print("  Buffer capacity:", BUFFER_CAPACITY)
        print("  Batch size:", BATCH_SIZE)
        print("  Learning rate: 6.25e-5")
        print("  PER: alpha=0.5, beta=0.4→1.0")
        print("  Noisy networks: factorized Gaussian (no epsilon-greedy)")
        print("  Tau (soft update): 0.005")
        print("  Total transitions:", NUM_STEPS)
        print()
        print("Expected rewards:")
        print("  - Random policy: ~-21 (CPU wins almost every point)")
        print("  - Learning policy: > -10")
        print("  - Good policy: > 0 (beating CPU)")
        print("  - Strong policy: > 10")
        print()

        # =====================================================================
        # Setup logger
        # =====================================================================

        var env_vars = load_dotenv()
        var api_key = env_vars.get("RL_MONITOR_API_KEY", "")
        var url = env_vars.get("RL_MONITOR_URL", "")

        var logger = RemoteLogger(
            server_url=url,
            run_name="Rainbow Pong GPU",
            buffer_size=64,
            api_key=api_key,
        )
        logger.set_config("agent", "Rainbow DQN")
        logger.set_config("env", "Pong")
        logger.set_config("hidden_dim", String(HIDDEN_DIM))
        logger.set_config("lr", "6.25e-5")
        logger.set_config("gamma", "0.99")
        logger.set_config("batch_size", String(BATCH_SIZE))
        logger.set_config("n_envs", String(N_ENVS))
        logger.set_config("buffer_capacity", String(BUFFER_CAPACITY))
        logger.set_config("n_step", String(N_STEP))
        logger.set_config("num_atoms", String(NUM_ATOMS))

        # =====================================================================
        # Train
        # =====================================================================

        print("Starting GPU training...")
        print("-" * 70)

        var start_time = perf_counter_ns()

        try:
            var metrics = agent.train_gpu[PongEnv[dtype]](
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

            # =================================================================
            # Summary
            # =================================================================

            print("=" * 70)
            print("Rainbow GPU Training Complete")
            print("=" * 70)
            print()
            print("Total transitions:", NUM_STEPS)
            print("Training time:", String(elapsed_s)[byte=:6], "seconds")
            print(
                "Transitions/second:",
                String(Float64(NUM_STEPS) / elapsed_s)[byte=:9],
            )
            print()

            var final_avg = metrics.mean_reward_last_n(100)
            print(
                "Final average reward (last 100 episodes):",
                String(final_avg)[byte=:8],
            )
            print("Best episode reward:", String(metrics.max_reward())[byte=:8])
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
