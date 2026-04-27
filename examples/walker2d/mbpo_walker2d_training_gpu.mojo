"""MBPO Agent GPU Training on Walker2d.

This trains the MBPO (Model-Based Policy Optimization) agent on the Walker2d
environment using GPU-accelerated training with:
- GPU environment stepping (with early termination on unhealthy state)
- GPU SAC gradient updates
- GPU dynamics ensemble training (7 networks, 5 elites)
- GPU branched model rollouts from real states

Walker2d terminates early when torso height leaves [0.8, 2.0] or |torso angle|
> 1.0. The dynamics model does not predict termination, so model rollouts use
NeverTerminate (matching HalfCheetah / Swimmer) — the paper keeps k=1 on
Walker2d so the mismatch has negligible impact.

Run with:
    pixi run -e apple mojo run -I . examples/walker2d/mbpo_walker2d_training_gpu.mojo    # Apple Silicon
    pixi run -e nvidia mojo run -I . examples/walker2d/mbpo_walker2d_training_gpu.mojo   # NVIDIA GPU
"""

from std.random import seed
from std.time import perf_counter_ns
from std.memory import UnsafePointer

from std.gpu.host import DeviceContext

from mojo_rl.core.dotenv import load_dotenv
from mojo_rl.core.logger import RemoteLogger
from mojo_rl.deep_agents import MBPOAgent, MBPOSACAgent
from mojo_rl.deep_agents.core.configs.mbpo_config import DefaultMBPOConfig
from mojo_rl.deep_agents.core.strategies.termination import NeverTerminate
from mojo_rl.envs.walker2d import Walker2d


# =============================================================================
# Constants
# =============================================================================

# Walker2d: 17D observation, 6D continuous action
comptime OBS_DIM = 17  # qpos[1:9] + qvel[0:9]
comptime ACTION_DIM = 6  # thigh, leg, foot x 2 legs

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

# Training duration
comptime NUM_STEPS = 300_000
comptime WARMUP_STEPS = 5_000

comptime dtype = DType.float32

# MBPO config — Walker2d uses k=1 rollouts per MBPO paper, so NeverTerminate
# matches the codebase convention (see NeverTerminate docstring).
comptime MBPOWalker2dConfig = DefaultMBPOConfig[
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
    NeverTerminate,
]


# =============================================================================
# Main
# =============================================================================


def main() raises:
    seed(42)
    print("=" * 70)
    print("MBPO Agent GPU Training on Walker2d")
    print("=" * 70)
    print()

    # =========================================================================
    # Create GPU context and agent
    # =========================================================================

    with DeviceContext() as ctx:
        var agent = MBPOAgent[
            MBPOWalker2dConfig,
            RemoteLogger,
        ](
            gamma=0.99,
            tau=0.005,
            action_scale=1.0,
            alpha=0.2,
            auto_alpha=True,
            alpha_lr=0.0003,
            target_entropy=-6.0,  # -ACTION_DIM
            model_train_freq=250,
            rollout_min_length=1,
            rollout_max_length=1,  # MBPO paper uses k=1 for Walker2d
            rollout_min_epoch=20,
            rollout_max_epoch=150,
            num_rollouts_per_step=100_000,
            real_ratio=0.05,
            sac_updates_per_step=40,  # MBPO paper n_train_repeat=40 for Walker2d
            checkpoint_every=50_000,
            checkpoint_path="mbpo_walker2d.ckpt",
            diag_every=500,  # log critic_loss, mean_q, mean_target, ... every 500 SAC updates
        )

        print("Environment: Walker2d Continuous (GPU)")
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
        print("    - Rollout length: 1 (Walker2d)")
        print("    - Real ratio: 0.05 (95% synthetic)")
        print("    - SAC updates per step: 40")
        print("    - Termination: NeverTerminate")
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
            run_name="MBPO Walker2d GPU",
            buffer_size=64,
            api_key=api_key,
        )
        logger.set_config("agent", "MBPO")
        logger.set_config("env", "Walker2d")
        logger.set_config("hidden_dim", String(HIDDEN_DIM))
        logger.set_config("dyn_hidden", String(DYN_HIDDEN))
        logger.set_config("ensemble_size", String(NUM_ENSEMBLE))
        logger.set_config("actor_lr", "3e-4")
        logger.set_config("critic_lr", "3e-4")
        logger.set_config("model_lr", "1e-3")
        logger.set_config("batch_size", String(BATCH_SIZE))
        logger.set_config("model_train_freq", "250")
        logger.set_config("rollout_length", "1")
        logger.set_config("real_ratio", "0.05")
        logger.set_config("sac_updates_per_step", "40")

        # =====================================================================
        # Train
        # =====================================================================

        print("Starting GPU training...")
        print("-" * 70)

        var start_time = perf_counter_ns()

        try:
            var metrics = agent.train_gpu[
                Walker2d[dtype, TERMINATE_ON_UNHEALTHY=True],
                USE_CUDA_GRAPH=False,
            ](
                ctx,
                num_steps=NUM_STEPS,
                warmup_steps=WARMUP_STEPS,
                verbose=True,
                print_every=10_000,
                environment_name="Walker2d",
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
            if final_avg > 4000.0:
                print("EXCELLENT: Walker is running fast! (avg reward > 4000)")
            elif final_avg > 2000.0:
                print("SUCCESS: Walker learned to walk! (avg reward > 2000)")
            elif final_avg > 500.0:
                print(
                    "GOOD PROGRESS: Walker is learning locomotion"
                    " (avg reward > 500)"
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
