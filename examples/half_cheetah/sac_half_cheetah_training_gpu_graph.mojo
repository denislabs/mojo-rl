"""SAC Agent GPU Training on HalfCheetah — CUDA Graph accelerated.

Same as sac_half_cheetah_training_gpu.mojo but with CUDA graph capture
for the training step. All dynamic state (RNG, alpha, buffer size) lives
on GPU, so graph replay produces correct training.

The CUDA graph captures _gpu_train_kernels (216 nodes) and replays it,
eliminating per-kernel launch overhead (~1.5x end-to-end speedup).

Run with:
    pixi run -e nvidia mojo run -I . examples/half_cheetah/sac_half_cheetah_training_gpu_graph.mojo
"""

from std.random import seed
from std.time import perf_counter_ns
from std.memory import UnsafePointer

from std.gpu.host import DeviceContext

from mojo_rl.core.dotenv import load_dotenv
from mojo_rl.core.logger import RemoteLogger
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
        var agent = DeepSACAgent[
            obs_dim=OBS_DIM,
            action_dim=ACTION_DIM,
            hidden_dim=HIDDEN_DIM,
            buffer_capacity=BUFFER_CAPACITY,
            batch_size=BATCH_SIZE,
            actor_lr=0.0003,
            critic_lr=0.001,
            L=RemoteLogger,
            max_n_envs=MAX_N_ENVS,
        ](
            gamma=0.99,
            tau=0.005,
            action_scale=1.0,
            alpha=0.2,
            auto_alpha=True,
            alpha_lr=0.001,
            target_entropy=-1.0,
            checkpoint_every=100_000,
            checkpoint_path="sac_half_cheetah_graph.ckpt",
        )

        print("Environment: HalfCheetah Continuous (GPU)")
        print("Agent: SAC (auto alpha, CUDA Graph)")
        print("  Observation dim: " + String(OBS_DIM))
        print("  Action dim: " + String(ACTION_DIM))
        print("  Hidden dim: " + String(HIDDEN_DIM))
        print("  Buffer capacity: " + String(BUFFER_CAPACITY))
        print("  Batch size: " + String(BATCH_SIZE))
        print("  Max parallel envs: " + String(MAX_N_ENVS))
        print("  Key hyperparameters:")
        print("    - Actor LR: 3e-4")
        print("    - Critic LR: 1e-3 (CleanRL default)")
        print("    - Alpha LR: 1e-3 (CleanRL default)")
        print("    - Tau (soft update): 0.005")
        print("    - Initial alpha: 0.2 (auto-tuned on GPU)")
        print("    - Target entropy: -" + String(ACTION_DIM))
        print("    - Warmup steps: " + String(WARMUP_STEPS))
        print()

        # =====================================================================
        # Setup logger
        # =====================================================================

        var env_vars = load_dotenv()
        var api_key = env_vars.get("RL_MONITOR_API_KEY", "")
        var url = env_vars.get("RL_MONITOR_URL", "")

        var logger = RemoteLogger(
            server_url=url,
            run_name="SAC HalfCheetah GPU Graph",
            buffer_size=64,
            api_key=api_key,
        )
        logger.set_config("agent", "SAC Generic (CUDA Graph)")
        logger.set_config("env", "HalfCheetah")
        logger.set_config("hidden_dim", String(HIDDEN_DIM))
        logger.set_config("actor_lr", "3e-4")
        logger.set_config("critic_lr", "1e-3")
        logger.set_config("alpha_lr", "1e-3")
        logger.set_config("batch_size", String(BATCH_SIZE))
        logger.set_config("buffer_capacity", String(BUFFER_CAPACITY))
        logger.set_config("cuda_graph", "true")

        # =====================================================================
        # Train using the train_gpu() method with CUDA Graph
        # =====================================================================

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
                logger=UnsafePointer(to=logger),
                diag_every=1_000,
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
            print("GPU Training Complete (CUDA Graph)")
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
