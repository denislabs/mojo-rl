"""MBPO Agent GPU Training on Ant.

This trains the MBPO (Model-Based Policy Optimization) agent on the Ant
environment using GPU-accelerated training with:
- GPU environment stepping (with early termination on unhealthy state)
- GPU SAC gradient updates
- GPU dynamics ensemble training (7 networks, 5 elites)
- GPU branched model rollouts from real states

Ant terminates early when z-position leaves [0.2, 1.0]. The dynamics model
does not predict termination, so AntTerminate is plugged in so that model-
generated rollouts see the same termination criterion as the env.

Run with:
    pixi run -e apple mojo run -I . examples/ant/mbpo_ant_training_gpu.mojo    # Apple Silicon
    pixi run -e nvidia mojo run -I . examples/ant/mbpo_ant_training_gpu.mojo   # NVIDIA GPU
"""

from std.random import seed
from std.time import perf_counter_ns
from std.memory import UnsafePointer

from std.gpu.host import DeviceContext

from mojo_rl.core.dotenv import load_dotenv
from mojo_rl.core.logger import RemoteLogger
from mojo_rl.deep_agents import MBPOAgent, MBPOSACAgent
from mojo_rl.deep_agents.core.configs.mbpo_config import DefaultMBPOConfig
from mojo_rl.deep_agents.core.strategies.termination import AntTerminate
from mojo_rl.envs.ant import Ant, AntConfig


# =============================================================================
# Constants
# =============================================================================

# Ant: 27D observation, 8D continuous action
comptime OBS_DIM = AntConfig.OBS_DIM  # 27
comptime ACTION_DIM = AntConfig.ACTION_DIM  # 8

# SAC network architecture
comptime HIDDEN_DIM = 256

# Buffer sizes
comptime BUFFER_CAPACITY = 1_000_000  # Real buffer
comptime SYNTH_CAPACITY = 400_000  # Synthetic buffer
comptime BATCH_SIZE = 128

# Dynamics ensemble
comptime NUM_ENSEMBLE = 7
comptime NUM_ELITES = 5
comptime DYN_HIDDEN = 200

# Training duration (Ant needs more steps due to 8D action space)
comptime NUM_STEPS = 500_000
comptime WARMUP_STEPS = 5_000

comptime dtype = DType.float32

# MBPO config — AntTerminate so branched model rollouts respect Ant's
# early-termination criterion (height < 0.2 or > 1.0).
comptime MBPOAntConfig = DefaultMBPOConfig[
    OBS_DIM,
    ACTION_DIM,
    HIDDEN_DIM,
    BUFFER_CAPACITY,
    SYNTH_CAPACITY,
    BATCH_SIZE,
    NUM_ENSEMBLE,
    NUM_ELITES,
    DYN_HIDDEN,
    0.0003,  # actor_lr
    0.0003,  # critic_lr (MBPO paper; 1e-3 over-shoots at high UTD)
    0.001,  # model_lr
    AntTerminate,
]


# =============================================================================
# Main
# =============================================================================


def main() raises:
    seed(42)
    print("=" * 70)
    print("MBPO Agent GPU Training on Ant")
    print("=" * 70)
    print()

    # =========================================================================
    # Create GPU context and agent
    # =========================================================================

    with DeviceContext() as ctx:
        var agent = MBPOAgent[
            MBPOAntConfig,
            RemoteLogger,
        ](
            gamma=0.99,
            tau=0.005,
            action_scale=1.0,
            alpha=0.2,
            auto_alpha=True,
            alpha_lr=0.0003,
            target_entropy=-8.0,  # -ACTION_DIM
            model_train_freq=250,
            rollout_min_length=1,
            rollout_max_length=25,  # MBPO paper uses schedule 1→25 on Ant
            rollout_min_epoch=20,
            rollout_max_epoch=100,
            num_rollouts_per_step=100_000,
            real_ratio=0.05,
            sac_updates_per_step=20,  # MBPO paper n_train_repeat=20 for Ant
            checkpoint_every=50_000,
            checkpoint_path="mbpo_ant.ckpt",
            diag_every=500,  # log critic_loss, mean_q, mean_target, ... every 500 SAC updates
        )

        print("Environment: Ant Continuous (GPU)")
        print("Agent: MBPO (Model-Based Policy Optimization)")
        print("  Observation dim: " + String(OBS_DIM))
        print("  Action dim: " + String(ACTION_DIM))
        print("  SAC hidden dim: " + String(HIDDEN_DIM))
        print("  Dynamics hidden dim: " + String(DYN_HIDDEN))
        print("  Buffer capacity (real): " + String(BUFFER_CAPACITY))
        print("  Buffer capacity (synth): " + String(SYNTH_CAPACITY))
        print("  Batch size: " + String(BATCH_SIZE))
        print(
            "  Ensemble: "
            + String(NUM_ENSEMBLE)
            + " models, "
            + String(NUM_ELITES)
            + " elites"
        )
        print("  Key hyperparameters:")
        print("    - Actor LR: 3e-4")
        print("    - Critic LR: 3e-4")
        print("    - Model LR: 1e-3")
        print("    - Alpha LR: 3e-4 (auto-tuned)")
        print("    - Tau: 0.005")
        print("    - Model train freq: 250 steps")
        print("    - Rollout length schedule: 1 -> 25 (Ant)")
        print("    - Real ratio: 0.05 (95% synthetic)")
        print("    - SAC updates per step: 20")
        print("    - Termination: AntTerminate")
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
            run_name="MBPO Ant GPU",
            buffer_size=64,
            api_key=api_key,
        )
        logger.set_config("agent", "MBPO")
        logger.set_config("env", "Ant")
        logger.set_config("hidden_dim", String(HIDDEN_DIM))
        logger.set_config("dyn_hidden", String(DYN_HIDDEN))
        logger.set_config("ensemble_size", String(NUM_ENSEMBLE))
        logger.set_config("actor_lr", "3e-4")
        logger.set_config("critic_lr", "3e-4")
        logger.set_config("model_lr", "1e-3")
        logger.set_config("batch_size", String(BATCH_SIZE))
        logger.set_config("model_train_freq", "250")
        logger.set_config("rollout_length_min", "1")
        logger.set_config("rollout_length_max", "25")
        logger.set_config("real_ratio", "0.05")
        logger.set_config("sac_updates_per_step", "20")

        # =====================================================================
        # Train
        # =====================================================================

        print("Starting GPU training...")
        print("-" * 70)

        var start_time = perf_counter_ns()

        try:
            var metrics = agent.train_gpu[
                Ant[dtype, TERMINATE_ON_UNHEALTHY=True],
                USE_CUDA_GRAPH=False,
            ](
                ctx,
                num_steps=NUM_STEPS,
                warmup_steps=WARMUP_STEPS,
                verbose=True,
                print_every=10_000,
                environment_name="Ant",
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
            print("MBPO GPU Training Complete")
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
            if final_avg > 5000.0:
                print("EXCELLENT: Ant is running fast! (avg reward > 5000)")
            elif final_avg > 3000.0:
                print("SUCCESS: Ant learned to walk! (avg reward > 3000)")
            elif final_avg > 1000.0:
                print(
                    "GOOD PROGRESS: Ant is learning locomotion"
                    " (avg reward > 1000)"
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
