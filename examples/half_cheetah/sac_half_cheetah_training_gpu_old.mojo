"""SAC Agent GPU Training on HalfCheetah (OLD custom agent for comparison).

Run with:
    pixi run -e apple mojo run -I . examples/half_cheetah/sac_half_cheetah_training_gpu_old.mojo
    pixi run -e nvidia mojo run -I . examples/half_cheetah/sac_half_cheetah_training_gpu_old.mojo
"""

from std.random import seed
from std.time import perf_counter_ns

from std.gpu.host import DeviceContext

from mojo_rl.deep_agents.sac import DeepSACAgent
from mojo_rl.envs.half_cheetah import (
    HalfCheetah,
    HalfCheetahConfig,
)


# =============================================================================
# Constants
# =============================================================================

# HalfCheetah: 17D observation, 6D continuous action
comptime OBS_DIM = HalfCheetahConfig.OBS_DIM  # 17
comptime ACTION_DIM = HalfCheetahConfig.ACTION_DIM  # 6

# Network architecture
comptime HIDDEN_DIM = 256

# Off-policy GPU training parameters
comptime BUFFER_CAPACITY = 1_000_000
comptime BATCH_SIZE = 256
comptime MAX_N_ENVS = 32

# Training duration (off-policy uses steps, not episodes)
comptime NUM_STEPS = 2_000_000
comptime WARMUP_STEPS = 10_000

comptime dtype = DType.float32


# =============================================================================
# Main
# =============================================================================


fn main() raises:
    seed(42)
    print("=" * 70)
    print("SAC Agent GPU Training on HalfCheetah (OLD custom agent)")
    print("=" * 70)
    print()

    with DeviceContext() as ctx:
        var agent = DeepSACAgent[
            obs_dim=OBS_DIM,
            action_dim=ACTION_DIM,
            hidden_dim=HIDDEN_DIM,
            buffer_capacity=BUFFER_CAPACITY,
            batch_size=BATCH_SIZE,
            actor_lr=0.0003,
            critic_lr=0.001,
            max_n_envs=MAX_N_ENVS,
        ](
            gamma=0.99,
            tau=0.005,
            action_scale=1.0,
            alpha=0.2,
            auto_alpha=True,
            alpha_lr=0.001,
            checkpoint_every=100_000,
            checkpoint_path="sac_half_cheetah_old.ckpt",
        )

        print("Environment: HalfCheetah Continuous (GPU)")
        print("Agent: SAC OLD (custom implementation)")
        print("  Observation dim: " + String(OBS_DIM))
        print("  Action dim: " + String(ACTION_DIM))
        print("  Hidden dim: " + String(HIDDEN_DIM))
        print("  Buffer capacity: " + String(BUFFER_CAPACITY))
        print("  Batch size: " + String(BATCH_SIZE))
        print("  Max parallel envs: " + String(MAX_N_ENVS))
        print()

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
                num_steps=NUM_STEPS,
                warmup_steps=WARMUP_STEPS,
                verbose=True,
                print_every=50_000,
                environment_name="HalfCheetah",
            )

            var end_time = perf_counter_ns()
            var elapsed_s = Float64(end_time - start_time) / 1e9

            print("-" * 70)
            print()
            print(">>> train_gpu returned successfully! <<<")

            print("=" * 70)
            print("GPU Training Complete")
            print("=" * 70)
            print()
            print("Total steps: " + String(NUM_STEPS))
            print("Training time: " + String(elapsed_s)[:6] + " seconds")
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
