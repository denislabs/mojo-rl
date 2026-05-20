"""MBPO Agent GPU Training on HalfCheetah.

This trains the MBPO (Model-Based Policy Optimization) agent on the HalfCheetah
environment using GPU-accelerated training with:
- GPU environment stepping
- GPU SAC gradient updates (20 per env step)
- GPU dynamics ensemble training (7 networks, 5 elites)
- GPU model rollouts (400 branched rollouts from real states)

MBPO achieves SAC-level asymptotic performance with ~10x fewer real env samples
by augmenting real data with model-generated synthetic rollouts.

Run with:
    pixi run -e apple mojo run -I . examples/half_cheetah/mbpo_half_cheetah_training_gpu.mojo    # Apple Silicon
    pixi run -e nvidia mojo run -I . examples/half_cheetah/mbpo_half_cheetah_training_gpu.mojo   # NVIDIA GPU
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

# SAC network architecture
comptime HIDDEN_DIM = 256

# Buffer sizes
comptime BUFFER_CAPACITY = 1_000_000  # Real buffer (match SAC)
comptime SYNTH_CAPACITY = 400_000  # Synthetic buffer
comptime BATCH_SIZE = 128

# Dynamics ensemble
comptime NUM_ENSEMBLE = 7
comptime NUM_ELITES = 5
comptime DYN_HIDDEN = 200

# Training duration
comptime NUM_STEPS = 300_000  # MBPO needs ~10x fewer steps than SAC
comptime WARMUP_STEPS = 5_000

comptime dtype = DType.float32

# MBPO config
comptime MBPOHalfCheetahConfig = DefaultMBPOConfig[
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
    NeverTerminate,  # HalfCheetah has no termination
]


# =============================================================================
# Main
# =============================================================================


def main() raises:
    seed(42)
    print("=" * 70)
    print("MBPO Agent GPU Training on HalfCheetah")
    print("=" * 70)
    print()

    # =========================================================================
    # Diagnostic: STATE_SIZE per submodel
    #
    # Hypothesis: after the state-buffer NN refactor, some MBPO submodel might
    # report STATE_SIZE > 0 unexpectedly. mbpo_agent.mojo passes a NULL pointer
    # for the critic's `state` view assuming "critic is stateless"; if any
    # composed layer now declares state, that NULL becomes a live garbage read.
    #
    # Expected (no bug): all four numbers below print 0.
    # If any line prints > 0, we've found state coming in unexpectedly.
    # =========================================================================
    print("--- STATE_SIZE diagnostic ---")
    print("ActorModel.STATE_SIZE     =", MBPOHalfCheetahConfig.ActorModel.STATE_SIZE)
    print("CriticModel.STATE_SIZE    =", MBPOHalfCheetahConfig.CriticModel.STATE_SIZE)
    print("DynamicsModel.STATE_SIZE  =", MBPOHalfCheetahConfig.DynamicsModel.STATE_SIZE)
    # Also break down each LinearSwish + final Linear inside DynamicsModel —
    # if the Sequential sum is 0 but a child is non-zero, that's its own bug.
    print(
        "  DynamicsModel.model_types[0].STATE_SIZE (LinearSwish[23,200]) =",
        MBPOHalfCheetahConfig.DynamicsModel.model_types[0].STATE_SIZE,
    )
    print(
        "  DynamicsModel.model_types[1].STATE_SIZE (LinearSwish[200,200]) =",
        MBPOHalfCheetahConfig.DynamicsModel.model_types[1].STATE_SIZE,
    )
    print(
        "  DynamicsModel.model_types[4].STATE_SIZE (Linear[200,36])      =",
        MBPOHalfCheetahConfig.DynamicsModel.model_types[4].STATE_SIZE,
    )
    print()

    # =========================================================================
    # Create GPU context and agent
    # =========================================================================

    with DeviceContext() as ctx:
        var agent = MBPOAgent[
            MBPOHalfCheetahConfig,
            RemoteLogger,
        ](
            gamma=0.99,
            tau=0.005,
            action_scale=1.0,
            alpha=0.2,
            auto_alpha=True,
            alpha_lr=0.0003,
            target_entropy=-3.0,  # Reference uses -3 (not -ACTION_DIM)
            model_train_freq=250,
            rollout_min_length=1,
            rollout_max_length=1,  # HalfCheetah uses k=1 per MBPO paper
            rollout_min_epoch=20,
            rollout_max_epoch=150,
            num_rollouts_per_step=100_000,  # Reference: 100K per dynamics training
            real_ratio=0.05,
            sac_updates_per_step=40,  # Reference: n_train_repeat=40
            # ERE (Emphasizing Recent Experience) on both real + synth buffers.
            # Not paper-faithful, but closes the Q-explosion gap at high UTD
            # (low TRAIN_N_ENVS). Safe to keep on at TRAIN_N_ENVS=32 too.
            # use_ere=True,
            # ere_eta=0.996,
            checkpoint_every=50_000,
            checkpoint_path="mbpo_half_cheetah.ckpt",
            diag_every=500,  # log critic_loss, mean_q, mean_target, ... every 500 SAC updates
        )

        print("Environment: HalfCheetah Continuous (GPU)")
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
        print("    - Rollout length: 1 (HalfCheetah)")
        print("    - Rollouts per step: 400")
        print("    - Real ratio: 0.05 (95% synthetic)")
        print("    - SAC updates per step: 20")
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
            run_name="MBPO HalfCheetah GPU",
            buffer_size=64,
            api_key=api_key,
        )
        logger.set_config("agent", "MBPO")
        logger.set_config("env", "HalfCheetah")
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
        logger.set_config("sac_updates_per_step", "20")

        # =====================================================================
        # Train
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
                print_every=10_000,
                environment_name="HalfCheetah",
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
