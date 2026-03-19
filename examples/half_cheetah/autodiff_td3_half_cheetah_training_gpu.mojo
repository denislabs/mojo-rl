"""Autodiff TD3 Agent GPU Training on HalfCheetah.

Same as td3_half_cheetah_training_gpu.mojo but using the autodiff-composed
actor loss (AutodiffTD3Config) instead of manual backward code (TD3Config).

The actor loss graph is expressed as a composed Model type:
    obs → SkipConcat[Actor] → [obs, action]
        → DualPath[Critic1, Critic2] → [Q1, Q2]
        → Min → min_Q → Negate → -min_Q

Forward and backward are fully automatic — no manual gradient stitching.

Run with:
    pixi run -e nvidia mojo run -I . examples/half_cheetah/autodiff_td3_half_cheetah_training_gpu.mojo
"""

from std.random import seed
from std.time import perf_counter_ns
from std.memory import UnsafePointer

from std.gpu.host import DeviceContext

from mojo_rl.core.dotenv import load_dotenv
from mojo_rl.core.logger import RemoteLogger
from mojo_rl.deep_agents.core.agents import (
    GenericOffPolicyAgent,
    AutodiffTD3Config,
)
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

# Training duration (total env transitions across all parallel envs)
comptime NUM_STEPS = 3_000_000
comptime WARMUP_STEPS = 10_000

comptime dtype = DType.float32


# =============================================================================
# Main
# =============================================================================


fn main() raises:
    seed(42)
    print("=" * 70)
    print("Autodiff TD3 Agent GPU Training on HalfCheetah")
    print("=" * 70)
    print()

    with DeviceContext() as ctx:
        comptime Config = AutodiffTD3Config[
            OBS_DIM,
            ACTION_DIM,
            HIDDEN_DIM,
            BUFFER_CAPACITY,
            BATCH_SIZE,
            actor_lr=0.0003,
            critic_lr=0.0003,
        ]

        var agent = GenericOffPolicyAgent[
            Config, L=RemoteLogger, max_n_envs=MAX_N_ENVS
        ](
            gamma=0.99,
            tau=0.005,
            action_scale=1.0,
            noise_std=0.1,
            policy_delay=2,
            target_noise_std=0.2,
            target_noise_clip=0.5,
            checkpoint_every=500_000,
            checkpoint_path="autodiff_td3_half_cheetah.ckpt",
        )

        print("Environment: HalfCheetah Continuous (GPU)")
        print("Agent: Autodiff TD3 (composed autodiff graph)")
        print("  Actor loss: AutodiffTD3Loss")
        print("    Graph: obs → SkipConcat[Actor]")
        print("         → DualPath[C1, C2] → Min → Negate")
        print("    Backward: AUTOMATIC (no manual gradient code)")
        print()
        print("  Observation dim: " + String(OBS_DIM))
        print("  Action dim: " + String(ACTION_DIM))
        print("  Hidden dim: " + String(HIDDEN_DIM))
        print("  Buffer capacity: " + String(BUFFER_CAPACITY))
        print("  Batch size: " + String(BATCH_SIZE))
        print("  Max parallel envs: " + String(MAX_N_ENVS))
        print("  Key hyperparameters:")
        print("    - Actor LR: 3e-4")
        print("    - Critic LR: 3e-4")
        print("    - Tau (soft update): 0.005")
        print("    - Exploration noise: 0.1 (decaying)")
        print("    - Policy delay: 2")
        print("    - Target noise: 0.2 (clip 0.5)")
        print("    - Warmup transitions: " + String(WARMUP_STEPS))
        print()

        # =====================================================================
        # Setup logger
        # =====================================================================

        var env_vars = load_dotenv()
        var api_key = env_vars.get("RL_MONITOR_API_KEY", "")
        var url = env_vars.get("RL_MONITOR_URL", "")

        var logger = RemoteLogger(
            server_url=url,
            run_name="Autodiff TD3 HalfCheetah GPU",
            buffer_size=64,
            api_key=api_key,
        )
        logger.set_config("agent", "Autodiff TD3")
        logger.set_config("env", "HalfCheetah")
        logger.set_config("actor_loss", "AutodiffTD3Loss")
        logger.set_config("hidden_dim", String(HIDDEN_DIM))
        logger.set_config("actor_lr", "3e-4")
        logger.set_config("critic_lr", "3e-4")
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
                HalfCheetah[dtype, TERMINATE_ON_UNHEALTHY=False],
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

            print("=" * 70)
            print("Autodiff TD3 GPU Training Complete")
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
