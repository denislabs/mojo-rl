"""Rainbow DQN CNN GPU Training on Pong with Pixel Observations.

Trains a Rainbow DQN CNN agent on the native Pong environment using
pixel observations (4×84×84 stacked grayscale frames).

Rainbow components: C51 + Double DQN + PER + Dueling + Noisy Networks + N-step.

Architecture (Nature DQN CNN backbone + dueling noisy distributional heads):
  Conv2D[4→32, 8×8, stride=4] → ReLU →
  Conv2D[32→64, 4×4, stride=2] → ReLU →
  Conv2D[64→64, 3×3, stride=1] → ReLU →
  Flatten → NoisyDense[3136→512] → ReLU →
  Parallel[
    NoisyDense[512→51],          # V distribution (51 atoms)
    NoisyDense[512→3*51],        # A distributions (3 actions × 51 atoms)
  ]

Run with:
    pixi run -e apple mojo run -I . examples/arcade_games/rainbow_pong_pixel_training_gpu.mojo
    pixi run -e nvidia mojo run -I . examples/arcade_games/rainbow_pong_pixel_training_gpu.mojo
"""

from std.random import seed
from std.time import perf_counter_ns
from std.memory import UnsafePointer

from std.gpu.host import DeviceContext

from mojo_rl.core.dotenv import load_dotenv
from mojo_rl.deep_agents.core.agents import (
    RainbowCNNConfig,
    GenericRainbowAgent,
)
from mojo_rl.envs.arcade_games.pong import PongPixelEnv
from mojo_rl.core.logger import RemoteLogger


# =============================================================================
# Constants
# =============================================================================

# Pong: 3 discrete actions, pixel observations (4×84×84)
comptime NUM_ACTIONS = 3  # NOOP, UP, DOWN
comptime NUM_ATOMS = 51
comptime N_STEP = 3

# Host-memory buffer allows large capacity without GPU OOM
comptime BUFFER_CAPACITY = 100_000
comptime BATCH_SIZE = 32
comptime N_ENVS = 64  # Fewer envs — each needs pixel workspace

# Training duration
comptime NUM_STEPS = 2_000_000

comptime dtype = DType.float32


# =============================================================================
# Main
# =============================================================================


def main() raises:
    seed(42)
    print("=" * 70)
    print("Rainbow DQN CNN GPU Training on Pong — Pixel Observations")
    print("=" * 70)
    print()

    with DeviceContext() as ctx:
        var agent = GenericRainbowAgent[
            RainbowCNNConfig[
                NUM_ACTIONS,
                NUM_ATOMS,
                -21.0,  # v_min (Pong score range)
                21.0,  # v_max
                N_STEP,
                BUFFER_CAPACITY,
                BATCH_SIZE,
                6.25e-5,  # lr
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

        print("Environment: Pong (GPU-batched, Pixel)")
        print("Agent: Rainbow DQN CNN (GPU)")
        print("  Components: C51 + Double DQN + PER + Dueling + Noisy + N-step")
        print("  Observation: 4 × 84 × 84 = 28224 (pixel frames)")
        print("  Actions:", NUM_ACTIONS, "(NOOP, UP, DOWN)")
        print("  Network: Nature CNN + Dueling Noisy Distributional Heads")
        print("    Conv1: 4→32, 8×8, stride 4  → 32×20×20")
        print("    Conv2: 32→64, 4×4, stride 2 → 64×9×9")
        print("    Conv3: 64→64, 3×3, stride 1 → 64×7×7")
        print("    NoisyDense: 3136→512")
        print("    Parallel[NoisyDense 512→51, NoisyDense 512→153]")
        print("  Atoms:", NUM_ATOMS, "support [-21, 21]")
        print("  N-step:", N_STEP)
        print("  N envs (parallel):", N_ENVS)
        print("  Buffer capacity:", BUFFER_CAPACITY)
        print("  Batch size:", BATCH_SIZE)
        print("  Learning rate: 6.25e-5")
        print("  PER: alpha=0.5, beta=0.4→1.0")
        print("  Noisy networks: factorized Gaussian (no epsilon-greedy)")
        print("  Total transitions:", NUM_STEPS)
        print()
        print("Note: Pixel-based Rainbow is slower than clean obs due to:")
        print("  - Per-env GPU rendering (160×210 framebuffer)")
        print("  - Frame resize (84×84) and 4-frame stacking")
        print("  - Larger replay buffer entries (56K floats per transition)")
        print("  - CNN forward/backward + distributional loss")
        print()

        # =====================================================================
        # Setup logger
        # =====================================================================

        var env_vars = load_dotenv()
        var api_key = env_vars.get("RL_MONITOR_API_KEY", "")
        var url = env_vars.get("RL_MONITOR_URL", "")

        var logger = RemoteLogger(
            server_url=url,
            run_name="Rainbow Pong Pixel GPU",
            buffer_size=64,
            api_key=api_key,
        )
        logger.set_config("agent", "Rainbow DQN CNN")
        logger.set_config("env", "Pong (Pixel)")
        logger.set_config("obs", "4x84x84")
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
            var metrics = agent.train_gpu[PongPixelEnv[dtype]](
                ctx,
                num_steps=NUM_STEPS,
                warmup_steps=5000,
                gradient_steps=8,
                sync_every=10_000,
                verbose=True,
                print_every=100_000,
                environment_name="Pong (Rainbow Pixel)",
                logger=UnsafePointer(to=logger),
                diag_every=1000,
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
            print("Rainbow CNN GPU Training Complete")
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
