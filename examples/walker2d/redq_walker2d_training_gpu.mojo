"""REDQ Agent GPU Training on Walker2d.

Paper-faithful REDQ: N=10 critics, subset-min target (M=2), UTD=20,
policy update delay=20. Single parallel env (paper setup); each env
transition triggers UTD_RATIO gradient updates.

Walker2d terminates early when torso height leaves [0.8, 2.0] or |torso
angle| > 1.0 (handled by the env when TERMINATE_ON_UNHEALTHY=True).

Run with:
    pixi run -e apple mojo run -I . examples/walker2d/redq_walker2d_training_gpu.mojo
    pixi run -e nvidia mojo run -I . examples/walker2d/redq_walker2d_training_gpu.mojo
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
from mojo_rl.envs.walker2d import Walker2d


# =============================================================================
# Constants
# =============================================================================

comptime OBS_DIM = 17  # qpos[1:9] + qvel[0:9]
comptime ACTION_DIM = 6  # thigh, leg, foot x 2 legs

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

comptime REDQWalker2dConfig = DefaultREDQConfig[
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
    print("REDQ Agent GPU Training on Walker2d")
    print("=" * 70)
    print()

    with DeviceContext() as ctx:
        var agent = REDQAgent[REDQWalker2dConfig, max_n_envs=N_ENVS](
            gamma=0.99,
            tau=0.005,
            action_scale=1.0,
            auto_alpha=True,
            alpha=0.2,
            alpha_lr=0.0003,
            target_entropy=-6,  # -ACTION_DIM
            max_grad_norm=0.0,  # paper does not clip
            checkpoint_every=50_000,
            checkpoint_path="redq_walker2d.ckpt",
            diag_every=1_000,  # per-train-step metrics (critic_loss, mean_q, ...)
        )

        # To resume from a previous run, uncomment:
        # agent.load_checkpoint("redq_walker2d.ckpt")

        print("Environment: Walker2d Continuous (GPU)")
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
        print("    - Target entropy: -6")
        print()

        # =====================================================================
        # Setup remote logger
        # =====================================================================

        var env_vars = load_dotenv()
        var api_key = env_vars.get("RL_MONITOR_API_KEY", "")
        var url = env_vars.get("RL_MONITOR_URL", "")

        var logger = RemoteLogger(
            server_url=url,
            run_name="REDQ Walker2d GPU",
            buffer_size=64,
            api_key=api_key,
        )
        logger.set_config("agent", "REDQ")
        logger.set_config("env", "Walker2d")
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
            var metrics = agent.train_gpu[
                Walker2d[dtype, TERMINATE_ON_UNHEALTHY=True],
                RemoteLogger,
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
            if final_avg > 4000.0:
                print("EXCELLENT: Walker is running fast! (avg reward > 4000)")
            elif final_avg > 2000.0:
                print("SUCCESS: Walker learned to walk! (avg reward > 2000)")
            elif final_avg > 500.0:
                print("GOOD PROGRESS: Walker is learning locomotion (avg > 500)")
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
