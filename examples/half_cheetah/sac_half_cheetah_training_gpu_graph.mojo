"""SAC Agent GPU Training on HalfCheetah — CUDA Graph accelerated.

Same as sac_half_cheetah_training_gpu.mojo but with CUDA graph capture
for the training step. Uses fixed alpha (no auto-tuning) since
alpha tuning requires ctx.synchronize() which is not capturable.

The CUDA graph captures the full do_gpu_train_step (108+ kernels)
and replays it, eliminating per-kernel launch overhead (~2.9x speedup
on the training step).

Run with:
    pixi run -e nvidia mojo run -I . examples/half_cheetah/sac_half_cheetah_training_gpu_graph.mojo
"""

from std.random import seed
from std.time import perf_counter_ns
from std.memory import UnsafePointer

from std.gpu.host import DeviceContext

from mojo_rl.deep_agents.core.agents import DeepSACAgent
from mojo_rl.envs.half_cheetah import (
    HalfCheetah,
    HalfCheetahConfig,
)


# =============================================================================
# Constants — same as sac_half_cheetah_training_gpu.mojo
# =============================================================================

comptime OBS_DIM = HalfCheetahConfig.OBS_DIM  # 17
comptime ACTION_DIM = HalfCheetahConfig.ACTION_DIM  # 6
comptime HIDDEN_DIM = 256
comptime BUFFER_CAPACITY = 1_000_000
comptime BATCH_SIZE = 256
comptime MAX_N_ENVS = 32

comptime NUM_STEPS = 600_000
comptime WARMUP_STEPS = 10_000

comptime dtype = DType.float32


def main() raises:
    seed(42)
    print("=" * 70)
    print("SAC Agent GPU Training on HalfCheetah (CUDA Graph)")
    print("=" * 70)
    print()

    with DeviceContext() as ctx:
        # Fixed alpha (no auto-tuning) — required for CUDA graph capture
        # since alpha tuning does ctx.synchronize() inside the train step
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
            auto_alpha=True,  # GPU-side alpha tuning (graph compatible)
            alpha_lr=0.001,
            target_entropy=-1.0,
        )

        print("Environment: HalfCheetah Continuous (GPU)")
        print("Agent: SAC (auto alpha, CUDA Graph)")
        print("  Observation dim: " + String(OBS_DIM))
        print("  Action dim: " + String(ACTION_DIM))
        print("  Hidden dim: " + String(HIDDEN_DIM))
        print("  Buffer capacity: " + String(BUFFER_CAPACITY))
        print("  Batch size: " + String(BATCH_SIZE))
        print("  Max parallel envs: " + String(MAX_N_ENVS))
        print()

        print("Starting GPU training...")
        print("-" * 70)

        var start_time = perf_counter_ns()

        try:
            var metrics = agent.train_gpu[
                HalfCheetah[dtype, TERMINATE_ON_UNHEALTHY=False],
                USE_CUDA_GRAPH=True,
            ](
                ctx,
                num_steps=NUM_STEPS,
                warmup_steps=WARMUP_STEPS,
                verbose=True,
                print_every=50_000,
            )

            var end_time = perf_counter_ns()
            var elapsed_s = Float64(end_time - start_time) / 1e9

            print("-" * 70)
            print()
            print("=" * 70)
            print("GPU Training Complete")
            print("=" * 70)
            print()
            print("Total steps: " + String(NUM_STEPS))
            print(
                "Training time: "
                + String(elapsed_s)[byte=:6]
                + " seconds"
            )
            print()

            var final_avg = metrics.mean_reward_last_n(100)
            print(
                "Final average reward (last 100 episodes): "
                + String(final_avg)[byte=:8]
            )
            print(
                "Best episode reward: "
                + String(metrics.max_reward())[byte=:8]
            )
            print()

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
