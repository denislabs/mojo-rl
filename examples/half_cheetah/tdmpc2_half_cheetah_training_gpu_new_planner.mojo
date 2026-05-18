"""TD-MPC2 Agent GPU Training on HalfCheetah — NEW PLANNER variant.

Identical to ``tdmpc2_half_cheetah_training_gpu.mojo`` except the
``train_gpu`` call passes ``use_new_mppi_planner=True``, swapping the
legacy ``plan_gpu_batched`` for the new Phase-2 planner-package path
(``MPPIGPUBatched`` + ``TDMPC2RolloutCallback``).

Use this for A/B parity validation against the legacy run:

    pixi run -e apple mojo run -I . \\
        examples/half_cheetah/tdmpc2_half_cheetah_training_gpu.mojo
    pixi run -e apple mojo run -I . \\
        examples/half_cheetah/tdmpc2_half_cheetah_training_gpu_new_planner.mojo

Same seed (42), same hyperparameters, same env. Compare episode-reward
curves — they should track each other within float32 / seed variance.
The new path uses slightly different kernel sequencing (no
sample_actions+build_za fusion; the reward decode is split between
callback and planner), so byte-exact parity is **not** expected — the
floating-point noise model differs slightly. Learning-curve parity
within seed variance is the real criterion.

Run with:
    pixi run -e apple mojo run -I . examples/half_cheetah/tdmpc2_half_cheetah_training_gpu_new_planner.mojo   # Apple Silicon
    pixi run -e nvidia mojo run -I . examples/half_cheetah/tdmpc2_half_cheetah_training_gpu_new_planner.mojo  # NVIDIA GPU
"""

from std.random import seed
from std.time import perf_counter_ns
from std.memory import UnsafePointer

from std.gpu.host import DeviceContext

from mojo_rl.core.dotenv import load_dotenv
from mojo_rl.core.logger import RemoteLogger
from mojo_rl.deep_agents.tdmpc2 import TDMPC2Agent
from mojo_rl.envs.half_cheetah import (
    HalfCheetah,
    HalfCheetahConfig,
)


# =============================================================================
# Constants — IDENTICAL to tdmpc2_half_cheetah_training_gpu.mojo so the
# A/B comparison is apples-to-apples.
# =============================================================================

# HalfCheetah: 17D observation, 6D continuous action
comptime OBS_DIM = HalfCheetahConfig.OBS_DIM  # 17
comptime ACTION_DIM = HalfCheetahConfig.ACTION_DIM  # 6

# Network architecture (TD-MPC2 5M config — official default)
comptime LATENT_DIM = 512
comptime MLP_DIM = 512
comptime ENC_DIM = 256

# Distributional RL
comptime NUM_BINS = 101

# Q-ensemble size
comptime NUM_Q = 5

# Planning horizon (used for sequence replay sampling and consistency loss)
comptime HORIZON = 3

# MPPI parameters
comptime NUM_SAMPLES = 512
comptime NUM_PI_TRAJS = 24
comptime NUM_ITERATIONS = 6

# Replay buffer and batch
comptime BATCH_SIZE = 256
comptime BUFFER_CAPACITY = 1_000_000

# Value range for distributional RL
comptime V_MIN = -10.0
comptime V_MAX = 10.0

# Number of parallel GPU environments
comptime N_ENVS = 1

# Training duration
comptime NUM_EPISODES = 2_000

comptime dtype = DType.float32  # GPU training uses float32


# =============================================================================
# Main
# =============================================================================


def main() raises:
    seed(42)
    print("=" * 70)
    print("TD-MPC2 Agent GPU Training on HalfCheetah [NEW PLANNER]")
    print("=" * 70)
    print()

    with DeviceContext() as ctx:
        # =====================================================================
        # Setup logger — distinct run_name so the A/B is comparable
        # in the logger UI without overwriting the legacy run.
        # =====================================================================

        var env_vars = load_dotenv()
        var api_key = env_vars.get("RL_MONITOR_API_KEY", "")
        var url = env_vars.get("RL_MONITOR_URL", "")

        var logger = RemoteLogger(
            server_url=url,
            run_name="TD-MPC2 HalfCheetah GPU [new planner]",
            buffer_size=64,
            api_key=api_key,
        )
        logger.set_config("agent", "TD-MPC2")
        logger.set_config("env", "HalfCheetah")
        logger.set_config("planner", "MPPIGPUBatched (Phase 2)")
        logger.set_config("latent_dim", String(LATENT_DIM))
        logger.set_config("mlp_dim", String(MLP_DIM))
        logger.set_config("num_bins", String(NUM_BINS))
        logger.set_config("num_q", String(NUM_Q))
        logger.set_config("horizon", String(HORIZON))
        logger.set_config("batch_size", String(BATCH_SIZE))
        logger.set_config("n_envs", String(N_ENVS))
        logger.set_config("wm_lr", "3e-4")
        logger.set_config("pi_lr", "3e-4")
        logger.set_config("gamma", "0.995 (dynamic)")

        var agent = TDMPC2Agent[
            obs_dim=OBS_DIM,
            action_dim=ACTION_DIM,
            latent_dim=LATENT_DIM,
            mlp_dim=MLP_DIM,
            enc_dim=ENC_DIM,
            num_bins=NUM_BINS,
            num_q=NUM_Q,
            horizon=HORIZON,
            batch_size=BATCH_SIZE,
            buffer_capacity=BUFFER_CAPACITY,
            num_samples=NUM_SAMPLES,
            num_pi_trajs=NUM_PI_TRAJS,
            num_iterations=NUM_ITERATIONS,
            v_min=V_MIN,
            v_max=V_MAX,
            L=RemoteLogger,
        ](
            episode_length=HalfCheetahConfig.MAX_STEPS,  # dynamic gamma: 0.995
            rho=0.5,
            tau=0.01,
            consistency_coef=20.0,
            reward_coef=0.1,
            value_coef=0.1,
            terminal_coef=1.0,
            entropy_coef=5e-4,
            temperature=0.5,
            action_scale=1.0,
            warmup_steps=5_000,
            wm_lr=3e-4,
            enc_lr_scale=0.3,
            pi_lr=3e-4,
            diag_every=500,
        )
        agent.logger = UnsafePointer(to=logger)

        print("Environment: HalfCheetah Continuous (GPU)")
        print("Agent: TD-MPC2 (GPU) — using NEW planner package")
        print("  Observation dim: " + String(OBS_DIM))
        print("  Action dim: " + String(ACTION_DIM))
        print("  Latent dim: " + String(LATENT_DIM))
        print("  MLP dim: " + String(MLP_DIM))
        print("  Encoder dim: " + String(ENC_DIM))
        print("  Num bins (distributional RL): " + String(NUM_BINS))
        print("  Q-ensemble size: " + String(NUM_Q))
        print()
        print("Planner:")
        print("  MPPIGPUBatched + TDMPC2RolloutCallback (Phase 2)")
        print("  Same algorithmic recipe as legacy plan_gpu_batched;")
        print("  kernel boundary split between planner + callback.")
        print()
        print("GPU Data Collection:")
        print("  Parallel environments: " + String(N_ENVS))
        print("  Horizon: " + String(HORIZON))
        print("  Exploration: policy + Gaussian noise (GPU-parallelizable)")
        print("  Warmup: random actions (tdmpc2_random_actions_kernel)")
        print()
        print("Training:")
        print("  Episodes: " + String(NUM_EPISODES))
        print("  Batch size: " + String(BATCH_SIZE))
        print("  Buffer capacity: " + String(BUFFER_CAPACITY))
        print(
            "  Per-env buffer: "
            + String(max(BATCH_SIZE + HORIZON + 2, BUFFER_CAPACITY // N_ENVS))
        )
        print(
            "  Updates per step: "
            + String(N_ENVS)
            + " (= N_ENVS, gives UTD=1 per transition matching reference)"
        )
        print("  Warmup steps: 5000 (random actions before training)")
        print("  World model LR: 3e-4")
        print("  Encoder LR scale: 0.3 (enc_lr = 9e-5)")
        print("  Policy LR: 3e-4")
        print()
        print("A/B comparison:")
        print(
            "  Compare against tdmpc2_half_cheetah_training_gpu.mojo"
            " (legacy plan_gpu_batched) with identical seed(42)."
        )
        print("  Expected: episode-reward curves within seed variance.")
        print()

        # =====================================================================
        # Train using train_gpu() with the new planner enabled
        # =====================================================================

        print("Starting GPU training [new planner]...")
        print("-" * 70)

        var start_time = perf_counter_ns()

        try:
            var metrics = agent.train_gpu[
                HalfCheetah[dtype, TERMINATE_ON_UNHEALTHY=False],
                n_envs=N_ENVS,
            ](
                ctx,
                num_episodes=NUM_EPISODES,
                verbose=True,
                use_mppi=True,
                updates_per_step=N_ENVS,
                use_new_mppi_planner=True,
            )

            var end_time = perf_counter_ns()
            var elapsed_s = Float64(end_time - start_time) / 1e9

            logger.close()

            print("-" * 70)
            print()
            print(">>> train_gpu() returned successfully! <<<")

            print("=" * 70)
            print("GPU Training Complete [new planner]")
            print("=" * 70)
            print()
            print("Total episodes: " + String(NUM_EPISODES))
            print("Training time: " + String(elapsed_s)[byte=:6] + " seconds")
            print(
                "Episodes/second: "
                + String(Float64(NUM_EPISODES) / elapsed_s)[byte=:7]
            )
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
