"""REDQ-OFE Agent GPU Training on Ant.

REDQ + OFENet (Ota et al., ICML 2020): 8-layer DenseNet-style feature
extractor (total_units=240, num_layers=8 → per_unit=30, matching the
paper's Ant.gin) trained via auxiliary next-state-prediction loss.

Paper-faithful REDQ: N=10 critics, subset-min target (M=2), UTD=20,
policy update delay=20. Single parallel env; each env transition
triggers UTD_RATIO gradient updates.

Ant terminates early when z-position leaves [0.2, 1.0] (handled by the
env when TERMINATE_ON_UNHEALTHY=True).

Run with:
    pixi run -e nvidia mojo run -I . examples/ant/redq_ofe_ant_training_gpu.mojo
    pixi run -e apple mojo run -I . examples/ant/redq_ofe_ant_training_gpu.mojo
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
from mojo_rl.envs.ant import Ant, AntConfig


# =============================================================================
# Constants
# =============================================================================

comptime OBS_DIM = AntConfig.OBS_DIM  # 27
comptime ACTION_DIM = AntConfig.ACTION_DIM  # 8

# REDQ paper configuration
comptime HIDDEN_DIM = 256
comptime BUFFER_CAPACITY = 1_000_000
comptime BATCH_SIZE = 256
comptime NUM_ENSEMBLE = 10
comptime NUM_MIN = 2
comptime UTD_RATIO = 20
comptime POLICY_DELAY = 20

comptime N_ENVS = 1

# Training duration (Ant needs more steps due to 8D action space)
comptime NUM_STEPS = 500_000
comptime WARMUP_STEPS = 5_000

comptime dtype = DType.float32

# 8-layer OFE (Ota et al. Ant.gin): total_units=240, num_layers=8 → per_unit=30.
comptime REDQOFEAntConfig = DefaultREDQOFEConfig8[
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
    240,     # OFE_TOTAL_UNITS
    1.0,     # action_scale
]


def main() raises:
    seed(42)
    print("=" * 70)
    print("REDQ-OFE Agent GPU Training on Ant")
    print("=" * 70)
    print()

    with DeviceContext() as ctx:
        var agent = REDQOFEAgent[REDQOFEAntConfig, max_n_envs=N_ENVS](
            gamma=0.99,
            tau=0.005,
            action_scale=1.0,
            auto_alpha=True,
            alpha=0.2,
            alpha_lr=0.0003,
            target_entropy=-8,  # -ACTION_DIM
            max_grad_norm=0.0,
            checkpoint_every=50_000,
            checkpoint_path="redq_ofe_ant.ckpt",
            diag_every=1_000,
            # OFENet pretraining: run this many aux_train_step calls on the
            # random-policy buffer right after env-collection warmup ends,
            # before the first RL update. Mirrors `references/OFENet-main/
            # teflon/tool/eager_main.py:256-261`.
            aux_warmup_steps=10_000,
        )

        # To resume from a previous run, uncomment:
        # agent.load_checkpoint("redq_ofe_ant.ckpt")

        print("Environment: Ant Continuous (GPU)")
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
        print("  phi_s dim: " + String(REDQOFEAntConfig.PHI_S_DIM))
        print("  phi_sa dim: " + String(REDQOFEAntConfig.PHI_SA_DIM))
        print("  Key hyperparameters:")
        print("    - Actor LR: 3e-4")
        print("    - Critic LR: 3e-4")
        print("    - Alpha LR: 3e-4")
        print("    - OFE (aux) LR: 3e-4")
        print("    - Tau (soft update): 0.005")
        print("    - Initial alpha: 0.2 (auto-tuned)")
        print("    - Target entropy: -8")
        print()

        # =====================================================================
        # Setup remote logger
        # =====================================================================

        var env_vars = load_dotenv()
        var api_key = env_vars.get("RL_MONITOR_API_KEY", "")
        var url = env_vars.get("RL_MONITOR_URL", "")

        var logger = RemoteLogger(
            server_url=url,
            run_name="REDQ-OFE Ant GPU",
            buffer_size=64,
            api_key=api_key,
        )
        logger.set_config("agent", "REDQ-OFE")
        logger.set_config("env", "Ant")
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
                Ant[dtype, TERMINATE_ON_UNHEALTHY=True],
                RemoteLogger,
            ](
                ctx,
                num_steps=NUM_STEPS,
                warmup_steps=WARMUP_STEPS,
                verbose=True,
                print_every=10_000,
                environment_name="Ant",
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
                print("EXCELLENT: Ant is running fast! (avg reward > 5000)")
            elif final_avg > 3000.0:
                print("SUCCESS: Ant learned to walk! (avg reward > 3000)")
            elif final_avg > 1000.0:
                print("GOOD PROGRESS: Ant is learning locomotion (avg > 1000)")
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
