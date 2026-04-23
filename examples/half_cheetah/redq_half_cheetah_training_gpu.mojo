"""REDQ Agent GPU Training on HalfCheetah.

Paper-faithful REDQ: N=10 critics, subset-min target (M=2), UTD=20,
policy update delay=20. Single parallel env (paper setup); each env
transition triggers UTD_RATIO gradient updates.

Run with:
    pixi run -e apple mojo run -I . examples/half_cheetah/redq_half_cheetah_training_gpu.mojo
    pixi run -e nvidia mojo run -I . examples/half_cheetah/redq_half_cheetah_training_gpu.mojo
"""

from std.random import seed
from std.time import perf_counter_ns
from std.memory import UnsafePointer

from std.gpu.host import DeviceContext

from mojo_rl.core.dotenv import load_dotenv
from mojo_rl.core.logger import RemoteLogger
from mojo_rl.deep_agents.core.configs.redq_config import (
    DefaultREDQConfig,
    REDQ_TARGET_MIN,
)
from mojo_rl.deep_agents.core.agents.redq_agent import REDQAgent
from mojo_rl.deep_agents.core.training.redq_train import run_redq_train_gpu
from mojo_rl.envs.half_cheetah import (
    HalfCheetah,
    HalfCheetahConfig,
)


# =============================================================================
# Constants
# =============================================================================

comptime OBS_DIM = HalfCheetahConfig.OBS_DIM  # 17
comptime ACTION_DIM = HalfCheetahConfig.ACTION_DIM  # 6

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

comptime REDQHalfCheetahConfig = DefaultREDQConfig[
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
    1.0,  # action_scale
]


def main() raises:
    seed(42)
    print("=" * 70)
    print("REDQ Agent GPU Training on HalfCheetah")
    print("=" * 70)
    print()

    with DeviceContext() as ctx:
        var agent = REDQAgent[REDQHalfCheetahConfig, max_n_envs=N_ENVS](
            gamma=0.99,
            tau=0.005,
            action_scale=1.0,
            auto_alpha=True,
            alpha=0.2,
            alpha_lr=0.0003,
            target_entropy=-Float64(ACTION_DIM),
            max_grad_norm=0.0,  # paper does not clip
            checkpoint_every=50_000,
            checkpoint_path="redq_half_cheetah.ckpt",
        )

        # To resume from a previous run, uncomment:
        # agent.load_checkpoint("redq_half_cheetah.ckpt")

        print("Environment: HalfCheetah Continuous (GPU)")
        print("Agent: REDQ (Randomized Ensembled Double Q-learning)")
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
        print("  Key hyperparameters:")
        print("    - Actor LR: 3e-4")
        print("    - Critic LR: 3e-4")
        print("    - Alpha LR: 3e-4")
        print("    - Tau (soft update): 0.005")
        print("    - Initial alpha: 0.2 (auto-tuned)")
        print("    - Target entropy: -" + String(ACTION_DIM))
        print()

        # =====================================================================
        # Setup remote logger
        # =====================================================================

        var env_vars = load_dotenv()
        var api_key = env_vars.get("RL_MONITOR_API_KEY", "")
        var url = env_vars.get("RL_MONITOR_URL", "")

        var logger = RemoteLogger(
            server_url=url,
            run_name="REDQ HalfCheetah GPU",
            buffer_size=64,
            api_key=api_key,
        )
        logger.set_config("agent", "REDQ")
        logger.set_config("env", "HalfCheetah")
        logger.set_config("hidden_dim", String(HIDDEN_DIM))
        logger.set_config("actor_lr", "3e-4")
        logger.set_config("critic_lr", "3e-4")
        logger.set_config("alpha_lr", "3e-4")
        logger.set_config("batch_size", String(BATCH_SIZE))
        logger.set_config("buffer_capacity", String(BUFFER_CAPACITY))
        logger.set_config("num_ensemble", String(NUM_ENSEMBLE))
        logger.set_config("num_min", String(NUM_MIN))
        logger.set_config("utd_ratio", String(UTD_RATIO))
        logger.set_config("policy_delay", String(POLICY_DELAY))

        print("Starting REDQ training...")
        print("-" * 70)

        var start_time = perf_counter_ns()

        try:
            var metrics = run_redq_train_gpu[
                HalfCheetah[dtype, TERMINATE_ON_UNHEALTHY=False],
                REDQHalfCheetahConfig,
                RemoteLogger,
                N_ENVS,
            ](
                agent,
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
            print(">>> run_redq_train_gpu returned successfully! <<<")
            print()
            print("=" * 70)
            print("REDQ Training Complete")
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
            if final_avg > 1000.0:
                print("EXCELLENT: Agent is running fast! (avg reward > 1000)")
            elif final_avg > 500.0:
                print("SUCCESS: Agent learned to run! (avg reward > 500)")
            elif final_avg > 100.0:
                print("GOOD PROGRESS: Agent is learning locomotion (avg > 100)")
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
