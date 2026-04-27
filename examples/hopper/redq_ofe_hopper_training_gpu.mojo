"""REDQ-OFE Agent GPU Training on Hopper.

REDQ + OFENet (Ota et al., ICML 2020): DenseNet-style feature extractor
(total_units=240, num_layers=6 per paper's Hopper.gin config) trained
jointly via auxiliary next-state-prediction loss.

Paper-faithful REDQ: N=10 critics, subset-min target (M=2), UTD=20,
policy update delay=20. Single parallel env (paper setup).

Hopper terminates early when |angle| > 0.2 or height < 0.7 (handled by
the env when TERMINATE_ON_UNHEALTHY=True).

Run with:
    pixi run -e apple mojo run -I . examples/hopper/redq_ofe_hopper_training_gpu.mojo
    pixi run -e nvidia mojo run -I . examples/hopper/redq_ofe_hopper_training_gpu.mojo
"""

from std.random import seed
from std.time import perf_counter_ns
from std.memory import UnsafePointer

from std.gpu.host import DeviceContext

from mojo_rl.core.dotenv import load_dotenv
from mojo_rl.core.logger import RemoteLogger
from mojo_rl.deep_agents.redq_ofe import (
    DefaultREDQOFEConfig6,
)
from mojo_rl.deep_agents.redq import REDQ_TARGET_MIN
from mojo_rl.deep_agents.redq_ofe.redq_ofe import REDQOFEAgent
from mojo_rl.envs.hopper import Hopper, HopperConfig


# =============================================================================
# Constants
# =============================================================================

comptime OBS_DIM = HopperConfig.OBS_DIM  # 11
comptime ACTION_DIM = HopperConfig.ACTION_DIM  # 3

# REDQ paper configuration
comptime HIDDEN_DIM = 256
comptime BUFFER_CAPACITY = 1_000_000
comptime BATCH_SIZE = 256
comptime NUM_ENSEMBLE = 10
comptime NUM_MIN = 2
comptime UTD_RATIO = 20
comptime POLICY_DELAY = 20

# Single-env collection (paper setup); loop still runs UTD_RATIO
# gradient updates per transition.
comptime N_ENVS = 1

# Training duration
comptime NUM_STEPS = 300_000
comptime WARMUP_STEPS = 5_000

comptime dtype = DType.float32

comptime REDQOFEHopperConfig = DefaultREDQOFEConfig6[
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
    0.0003,  # ofe_lr (aux Adam) — matches HalfCheetah's working config
    240,     # OFE_TOTAL_UNITS (paper's Hopper.gin)
    1.0,     # action_scale
]


def main() raises:
    seed(42)
    print("=" * 70)
    print("REDQ-OFE Agent GPU Training on Hopper")
    print("=" * 70)
    print()

    with DeviceContext() as ctx:
        var agent = REDQOFEAgent[REDQOFEHopperConfig, max_n_envs=N_ENVS](
            gamma=0.99,
            tau=0.005,
            action_scale=1.0,
            auto_alpha=True,
            alpha=0.2,
            alpha_lr=0.0003,
            target_entropy=-3,  # -ACTION_DIM (matches vanilla REDQ Hopper)
            max_grad_norm=0.0,  # paper does not clip
            checkpoint_every=50_000,
            checkpoint_path="redq_ofe_hopper.ckpt",
            diag_every=1_000,
            disable_aux=False,
            # OFENet pretraining: run this many aux_train_step calls on the
            # random-policy buffer right after env-collection warmup ends,
            # before the first RL update. Mirrors `references/OFENet-main/
            # teflon/tool/eager_main.py:256-261`.
            aux_warmup_steps=10_000,
        )

        # To resume from a previous run, uncomment:
        # agent.load_checkpoint("redq_ofe_hopper.ckpt")

        print("Environment: Hopper Continuous (GPU)")
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
        print("  OFE num_layers: 6")
        print("  OFE per_unit: 40")
        print("  phi_s dim: " + String(REDQOFEHopperConfig.PHI_S_DIM))
        print("  phi_sa dim: " + String(REDQOFEHopperConfig.PHI_SA_DIM))
        print("  Key hyperparameters:")
        print("    - Actor LR: 3e-4")
        print("    - Critic LR: 3e-4")
        print("    - Alpha LR: 3e-4")
        print("    - OFE (aux) LR: 3e-4")
        print("    - Tau (soft update): 0.005")
        print("    - Initial alpha: 0.2 (auto-tuned)")
        print("    - Target entropy: -3")
        print()

        # =====================================================================
        # Setup remote logger
        # =====================================================================

        var env_vars = load_dotenv()
        var api_key = env_vars.get("RL_MONITOR_API_KEY", "")
        var url = env_vars.get("RL_MONITOR_URL", "")

        var logger = RemoteLogger(
            server_url=url,
            run_name="REDQ-OFE Hopper GPU",
            buffer_size=64,
            api_key=api_key,
        )
        logger.set_config("agent", "REDQ-OFE")
        logger.set_config("env", "Hopper")
        logger.set_config("hidden_dim", String(HIDDEN_DIM))
        logger.set_config("actor_lr", "3e-4")
        logger.set_config("critic_lr", "3e-4")
        logger.set_config("alpha_lr", "3e-4")
        logger.set_config("ofe_lr", "3e-4")
        logger.set_config("ofe_num_layers", "6")
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
                Hopper[dtype, TERMINATE_ON_UNHEALTHY=True],
                RemoteLogger,
            ](
                ctx,
                num_steps=NUM_STEPS,
                warmup_steps=WARMUP_STEPS,
                verbose=True,
                print_every=10_000,
                environment_name="Hopper",
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
            if final_avg > 3000.0:
                print("EXCELLENT: Agent is hopping fast! (avg reward > 3000)")
            elif final_avg > 1500.0:
                print("SUCCESS: Agent learned to hop! (avg reward > 1500)")
            elif final_avg > 500.0:
                print("GOOD PROGRESS: Agent is learning locomotion (avg > 500)")
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
