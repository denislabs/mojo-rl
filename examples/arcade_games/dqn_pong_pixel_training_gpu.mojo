"""DQN CNN GPU Training on Pong with Pixel Observations.

Trains a Double DQN CNN agent on the native Pong environment using
pixel observations (4×84×84 stacked grayscale frames).

The Nature DQN architecture processes raw pixels:
  Conv2D[4→32, 8×8, stride=4] → ReLU →
  Conv2D[32→64, 4×4, stride=2] → ReLU →
  Conv2D[64→64, 3×3, stride=1] → ReLU →
  Flatten → Dense[3136→512] → ReLU →
  Dense[512→3]

Run with:
    pixi run -e apple mojo run -I . examples/arcade_games/dqn_pong_pixel_training_gpu.mojo
    pixi run -e nvidia mojo run -I . examples/arcade_games/dqn_pong_pixel_training_gpu.mojo
"""

from std.random import seed
from std.time import perf_counter_ns

from std.gpu.host import DeviceContext

from mojo_rl.deep_agents.dqn_cnn import DQNCNNAgent
from mojo_rl.envs.arcade_games.pong import PongPixelEnv


# =============================================================================
# Constants
# =============================================================================

# Pong: 3 discrete actions, pixel observations (4×84×84)
comptime NUM_ACTIONS = 3  # NOOP, UP, DOWN

# DQN CNN hyperparameters — smaller buffers/batch due to large obs size
comptime BUFFER_CAPACITY = 10_000
comptime BATCH_SIZE = 256
comptime N_ENVS = 64  # Fewer envs — each needs pixel workspace

# Training duration
comptime NUM_STEPS = 2_000_000  # Total env transitions

comptime dtype = DType.float32


# =============================================================================
# Main
# =============================================================================


fn main() raises:
    seed(42)
    print("=" * 70)
    print("DQN CNN (Double) GPU Training on Pong — Pixel Observations")
    print("=" * 70)
    print()

    with DeviceContext() as ctx:
        var agent = DQNCNNAgent[
            num_actions=NUM_ACTIONS,
            buffer_capacity=BUFFER_CAPACITY,
            batch_size=BATCH_SIZE,
            n_envs=N_ENVS,
            double_dqn=True,
            lr=0.00025,
        ](
            gamma=0.99,
            tau=0.005,
            epsilon=1.0,
            epsilon_min=0.02,
            epsilon_decay=0.9998,
            checkpoint_every=50,
            checkpoint_path="dqn_cnn_pong.ckpt",
        )

        print("Environment: Pong (GPU-batched, Pixel)")
        print("Agent: Double DQN CNN (GPU)")
        print("  Observation: 4 × 84 × 84 = 28224 (pixel frames)")
        print("  Actions:", NUM_ACTIONS, "(NOOP, UP, DOWN)")
        print("  Network: Nature DQN CNN")
        print("    Conv1: 4→32, 8×8, stride 4  → 32×20×20")
        print("    Conv2: 32→64, 4×4, stride 2 → 64×9×9")
        print("    Conv3: 64→64, 3×3, stride 1 → 64×7×7")
        print("    Dense: 3136→512→3")
        print("  N envs (parallel):", N_ENVS)
        print("  Buffer capacity:", BUFFER_CAPACITY)
        print("  Batch size:", BATCH_SIZE)
        print("  Learning rate: 2.5e-4")
        print("  Epsilon: 1.0 → 0.02 (decay=0.9998)")
        print("  Tau (soft update): 0.005")
        print("  Double DQN: enabled")
        print("  Total transitions:", NUM_STEPS)
        print()
        print("Note: Pixel-based training is slower than clean obs due to:")
        print("  - Per-env GPU rendering (160×210 framebuffer)")
        print("  - Frame resize (84×84) and 4-frame stacking")
        print("  - Larger replay buffer entries (56K floats per transition)")
        print("  - CNN forward/backward (1.7M parameters)")
        print()

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
                environment_name="Pong (Pixel)",
            )

            var end_time = perf_counter_ns()
            var elapsed_s = Float64(end_time - start_time) / 1e9

            print("-" * 70)
            print()

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
