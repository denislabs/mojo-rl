"""REDQ-OFE Agent GPU Training on Humanoid.

REDQ + OFENet (Ota et al., ICML 2020): 8-layer DenseNet-style feature
extractor (total_units=240, num_layers=8 → per_unit=30, matching the
paper's Humanoid.gin) trained via auxiliary next-state-prediction loss.

Paper-faithful REDQ: N=10 critics, subset-min target (M=2), UTD=20,
policy update delay=20. Per the paper, OFE gives the largest gains on
high-dim envs like Humanoid.

Humanoid terminates early on fall (handled by the env when
TERMINATE_ON_UNHEALTHY=True).

Run with:
    pixi run -e nvidia mojo run -I . examples/humanoid/redq_ofe_humanoid_training_gpu.mojo
    pixi run -e apple mojo run -I . examples/humanoid/redq_ofe_humanoid_training_gpu.mojo
"""

from std.random import seed
from std.time import perf_counter_ns
from std.memory import UnsafePointer

from std.gpu.host import DeviceContext

from mojo_rl.core.dotenv import load_dotenv
from mojo_rl.core.logger import RemoteLogger
from mojo_rl.deep_agents.redq_ofe import (
    DefaultREDQOFEConfig8,
    REDQOFEAgent,
)
from mojo_rl.deep_agents.redq import REDQ_TARGET_MIN
from mojo_rl.envs.humanoid import Humanoid


# =============================================================================
# Constants
# =============================================================================

comptime OBS_DIM = 45  # qpos[2:24] + qvel[0:23]
comptime ACTION_DIM = 17  # 17 motors for all joints

# REDQ paper configuration
comptime HIDDEN_DIM = 256
comptime BUFFER_CAPACITY = 1_000_000
comptime BATCH_SIZE = 256
comptime NUM_ENSEMBLE = 10
comptime NUM_MIN = 2
comptime UTD_RATIO = 20
comptime POLICY_DELAY = 20

comptime N_ENVS = 1

# Training duration (Humanoid is high-D; REDQ is sample-efficient but
# each update is expensive at UTD=20, so use 500K steps as a budget)
comptime NUM_STEPS = 500_000
comptime WARMUP_STEPS = 10_000

comptime dtype = DType.float32

# 8-layer OFE (Ota et al. Humanoid.gin): total_units=240, num_layers=8 → per_unit=30.
comptime REDQOFEHumanoidConfig = DefaultREDQOFEConfig8[
    OBS_DIM,
    ACTION_DIM,
    HIDDEN_DIM,
    BUFFER_CAPACITY,
    BATCH_SIZE,
    NUM_ENSEMBLE,
    NUM_MIN,
    UTD_RATIO,
    POLICY_DELAY,
    REDQ_TARGET_MIN,
    0.0003,  # actor_lr
    0.0003,  # critic_lr
    0.0003,  # ofe_lr (aux Adam)
    240,  # OFE_TOTAL_UNITS
    0.4,  # action_scale (match SAC Humanoid)
]


def main() raises:
    seed(42)
    print("=" * 70)
    print("REDQ-OFE Agent GPU Training on Humanoid")
    print("=" * 70)
    print()

    with DeviceContext() as ctx:
        var agent = REDQOFEAgent[REDQOFEHumanoidConfig, max_n_envs=N_ENVS](
            gamma=0.99,
            tau=0.005,
            action_scale=0.4,
            auto_alpha=True,
            alpha=0.2,
            alpha_lr=0.0003,
            target_entropy=-17,  # -ACTION_DIM
            max_grad_norm=0.0,
            checkpoint_every=100_000,
            checkpoint_path="redq_ofe_humanoid.ckpt",
            diag_every=1_000,
        )

        # To resume from a previous run, uncomment:
        # agent.load_checkpoint("redq_ofe_humanoid.ckpt")

        print("Environment: Humanoid Continuous (GPU)")
        print("Agent: REDQ-OFE (REDQ + OFENet feature extractor)")
        print("  Observation dim: " + String(OBS_DIM))
        print("  Action dim: " + String(ACTION_DIM))
        print("  Hidden dim: " + String(HIDDEN_DIM))
        print("  Buffer capacity: " + String(BUFFER_CAPACITY))
        print("  Batch size: " + String(BATCH_SIZE))
        print("  N_ENSEMBLE: " + String(NUM_ENSEMBLE))
        print("  N_MIN (subset-min): " + String(NUM_MIN))
        print("  UTD ratio: " + String(UTD_RATIO))
        print("  Policy delay: " + String(POLICY_DELAY))
        print("  Parallel envs: " + String(N_ENVS))
        print("  Warmup steps: " + String(WARMUP_STEPS))
        print("  OFE total_units: 240")
        print("  OFE num_layers: 8")
        print("  OFE per_unit: 30")
        print("  phi_s dim: " + String(REDQOFEHumanoidConfig.PHI_S_DIM))
        print("  phi_sa dim: " + String(REDQOFEHumanoidConfig.PHI_SA_DIM))
        print("  Key hyperparameters:")
        print("    - Actor LR: 3e-4")
        print("    - Critic LR: 3e-4")
        print("    - Alpha LR: 3e-4")
        print("    - OFE (aux) LR: 3e-4")
        print("    - Action scale: 0.4")
        print("    - Tau (soft update): 0.005")
        print("    - Initial alpha: 0.2 (auto-tuned)")
        print("    - Target entropy: -17")
        print()

        # =====================================================================
        # Setup remote logger
        # =====================================================================

        var env_vars = load_dotenv()
        var api_key = env_vars.get("RL_MONITOR_API_KEY", "")
        var url = env_vars.get("RL_MONITOR_URL", "")

        var logger = RemoteLogger(
            server_url=url,
            run_name="REDQ-OFE Humanoid GPU",
            buffer_size=64,
            api_key=api_key,
        )
        logger.set_config("agent", "REDQ-OFE")
        logger.set_config("env", "Humanoid")
        logger.set_config("hidden_dim", String(HIDDEN_DIM))
        logger.set_config("actor_lr", "3e-4")
        logger.set_config("critic_lr", "3e-4")
        logger.set_config("alpha_lr", "3e-4")
        logger.set_config("ofe_lr", "3e-4")
        logger.set_config("ofe_num_layers", "8")
        logger.set_config("ofe_total_units", "240")
        logger.set_config("batch_size", String(BATCH_SIZE))
        logger.set_config("buffer_capacity", String(BUFFER_CAPACITY))
        logger.set_config("num_ensemble", String(NUM_ENSEMBLE))
        logger.set_config("num_min", String(NUM_MIN))
        logger.set_config("utd_ratio", String(UTD_RATIO))
        logger.set_config("policy_delay", String(POLICY_DELAY))

        print("Starting REDQ-OFE training...")
        print("-" * 70)

        var start_time = perf_counter_ns()

        try:
            var metrics = agent.train_gpu[
                Humanoid[dtype, TERMINATE_ON_UNHEALTHY=True],
                RemoteLogger,
            ](
                ctx,
                num_steps=NUM_STEPS,
                warmup_steps=WARMUP_STEPS,
                verbose=True,
                print_every=10_000,
                environment_name="Humanoid",
                logger=UnsafePointer(to=logger),
                diag_every=1_000,
            )

            var end_time = perf_counter_ns()
            var elapsed_s = Float64(end_time - start_time) / 1e9

            logger.close()

            print("-" * 70)
            print()
            print(">>> train_gpu returned successfully! <<<")
            print()
            print("=" * 70)
            print("REDQ-OFE Training Complete")
            print("=" * 70)
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
                print("EXCELLENT: Humanoid is running! (avg reward > 5000)")
            elif final_avg > 2000.0:
                print("SUCCESS: Humanoid learned to walk! (avg reward > 2000)")
            elif final_avg > 500.0:
                print("GOOD PROGRESS: Humanoid is learning (avg > 500)")
            elif final_avg > 0.0:
                print("LEARNING: Agent improving but needs more training")
            else:
                print("EARLY STAGE: Agent still exploring (avg reward < 0)")
            print("=" * 70)

        except e:
            print("!!! EXCEPTION CAUGHT !!!")
            print("Error:", e)
            print("!!! END EXCEPTION !!!")

    print(">>> main() completed normally <<<")
