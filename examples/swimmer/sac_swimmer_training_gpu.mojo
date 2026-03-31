"""SAC Agent GPU Training on Swimmer.

This trains the SAC (Soft Actor-Critic) agent on the Swimmer environment
using GPU-accelerated off-policy training with:
- Parallel environments on GPU
- Generalized Coordinates (GC) physics engine (MuJoCo-style)
- 2D continuous action space (joint torques)
- 8D observation (qpos + qvel excluding rootx/rooty)

Run with:
    pixi run -e apple mojo run -I . examples/swimmer/sac_swimmer_training_gpu.mojo    # Apple Silicon
    pixi run -e nvidia mojo run -I . examples/swimmer/sac_swimmer_training_gpu.mojo   # NVIDIA GPU
"""

from std.random import seed
from std.time import perf_counter_ns
from std.memory import UnsafePointer

from std.gpu.host import DeviceContext

from mojo_rl.core.dotenv import load_dotenv
from mojo_rl.core.logger import RemoteLogger
from mojo_rl.deep_agents.core.agents import DeepSACAgent
from mojo_rl.envs.swimmer import Swimmer


# =============================================================================
# Constants
# =============================================================================

# Swimmer: 8D observation, 2D continuous action
comptime OBS_DIM = 8  # qpos[2:5] + qvel[0:5]
comptime ACTION_DIM = 2  # 2 rotational motors

# Network architecture (smaller for simple env)
comptime HIDDEN_DIM = 128

# Off-policy GPU training parameters
comptime BUFFER_CAPACITY = 300_000
comptime BATCH_SIZE = 64
comptime MAX_N_ENVS = 4

# Training duration (Swimmer is simpler)
comptime NUM_STEPS = 500_000
comptime WARMUP_STEPS = 5_000

comptime dtype = DType.float32


# =============================================================================
# Main
# =============================================================================


def main() raises:
    seed(42)
    print("=" * 70)
    print("SAC Agent GPU Training on Swimmer")
    print("=" * 70)
    print()

    # =========================================================================
    # Create GPU context and agent
    # =========================================================================

    with DeviceContext() as ctx:
        var agent = DeepSACAgent[
            obs_dim=OBS_DIM,
            action_dim=ACTION_DIM,
            hidden_dim=HIDDEN_DIM,
            buffer_capacity=BUFFER_CAPACITY,
            batch_size=BATCH_SIZE,
            actor_lr=0.0003,
            critic_lr=0.0003,
            L=RemoteLogger,
            max_n_envs=MAX_N_ENVS,
        ](
            gamma=0.99,
            tau=0.005,
            action_scale=1.0,
            alpha=0.2,
            auto_alpha=True,
            alpha_lr=0.0003,
            target_entropy=-1.0,
            checkpoint_every=100_000,
            checkpoint_path="sac_swimmer.ckpt",
        )

        print("Environment: Swimmer Continuous (GPU)")
        print("Agent: SAC (Soft Actor-Critic)")
        print("  Observation dim: " + String(OBS_DIM))
        print("  Action dim: " + String(ACTION_DIM))
        print("  Hidden dim: " + String(HIDDEN_DIM))
        print("  Buffer capacity: " + String(BUFFER_CAPACITY))
        print("  Batch size: " + String(BATCH_SIZE))
        print("  Max parallel envs: " + String(MAX_N_ENVS))
        print("  Key hyperparameters:")
        print("    - Actor LR: 3e-4")
        print("    - Critic LR: 1e-3")
        print("    - Alpha LR: 1e-3")
        print("    - Tau (soft update): 0.005")
        print("    - Initial alpha: 0.2 (auto-tuned)")
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
            run_name="SAC Swimmer GPU",
            buffer_size=64,
            api_key=api_key,
        )
        logger.set_config("agent", "SAC")
        logger.set_config("env", "Swimmer")
        logger.set_config("hidden_dim", String(HIDDEN_DIM))
        logger.set_config("actor_lr", "3e-4")
        logger.set_config("critic_lr", "1e-3")
        logger.set_config("alpha_lr", "1e-3")
        logger.set_config("batch_size", String(BATCH_SIZE))
        logger.set_config("buffer_capacity", String(BUFFER_CAPACITY))

        # =====================================================================
        # Train using the train_gpu() method
        # =====================================================================

        print("Starting GPU training...")
        print("-" * 70)

        var start_time = perf_counter_ns()

        try:
            var metrics = agent.train_gpu[
                Swimmer[dtype, TERMINATE_ON_UNHEALTHY=False],
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
            print("GPU Training Complete")
            print("=" * 70)
            print()
            print("Total steps: " + String(NUM_STEPS))
            print("Training time: " + String(elapsed_s)[byte=:6] + " seconds")
            print()

            print(
                "Final average reward (last 100 episodes): "
                + String(metrics.mean_reward_last_n(100))[byte=:8]
            )
            print(
                "Best episode reward: " + String(metrics.max_reward())[byte=:8]
            )
            print()

            var final_avg = metrics.mean_reward_last_n(100)
            if final_avg > 300.0:
                print("EXCELLENT: Swimmer is moving fast! (avg reward > 300)")
            elif final_avg > 100.0:
                print("SUCCESS: Swimmer learned to swim! (avg reward > 100)")
            elif final_avg > 30.0:
                print("GOOD PROGRESS: Swimmer is learning (avg reward > 30)")
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
