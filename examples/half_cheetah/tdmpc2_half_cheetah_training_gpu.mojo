"""TD-MPC2 Agent GPU Training on HalfCheetah.

Train TDMPC2 on HalfCheetah using GPU-based world model learning with
N_ENVS parallel GPU environments for data collection:
- World model: encoder, dynamics, reward, termination, policy, Q-ensemble
- Policy-based exploration (encoder + policy + noise) for GPU data collection
- Distributional RL with 101 bins (two-hot targets)
- 5-network Q-ensemble with soft-updated target networks
- N_ENVS independent sequence replay buffers (one per environment)
- All network training on GPU (forward, backward, Adam update)

Action space (6D continuous):
- action[0]: back thigh (hip) torque (-1.0 to 1.0) * gear=120
- action[1]: back shin (knee) torque (-1.0 to 1.0) * gear=90
- action[2]: back foot (ankle) torque (-1.0 to 1.0) * gear=60
- action[3]: front thigh (hip) torque (-1.0 to 1.0) * gear=120
- action[4]: front shin (knee) torque (-1.0 to 1.0) * gear=60
- action[5]: front foot (ankle) torque (-1.0 to 1.0) * gear=30

Run with:
    pixi run -e apple mojo run -I . examples/half_cheetah/tdmpc2_half_cheetah_training_gpu.mojo   # Apple Silicon
    pixi run -e nvidia mojo run -I . examples/half_cheetah/tdmpc2_half_cheetah_training_gpu.mojo  # NVIDIA GPU
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
# Constants
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

# MPPI parameters (used by CPU train(), not GPU train_gpu())
comptime NUM_SAMPLES = 512
comptime NUM_PI_TRAJS = 24
comptime NUM_ITERATIONS = 6

# Replay buffer and batch
comptime BATCH_SIZE = 256
comptime BUFFER_CAPACITY = 100_000

# Value range for distributional RL
comptime V_MIN = -10.0
comptime V_MAX = 10.0

# Number of parallel GPU environments
comptime N_ENVS = 32

# Training duration
comptime NUM_EPISODES = 1_000

comptime dtype = DType.float32  # GPU training uses float32


# =============================================================================
# Main
# =============================================================================


fn main() raises:
    seed(42)
    print("=" * 70)
    print("TD-MPC2 Agent GPU Training on HalfCheetah")
    print("=" * 70)
    print()

    # =========================================================================
    # Create GPU context and agent
    # =========================================================================

    with DeviceContext() as ctx:
        # =====================================================================
        # Setup logger
        # =====================================================================

        var env_vars = load_dotenv()
        var api_key = env_vars.get("RL_MONITOR_API_KEY", "")
        var url = env_vars.get("RL_MONITOR_URL", "")

        var logger = RemoteLogger(
            server_url=url,
            run_name="TD-MPC2 HalfCheetah GPU",
            buffer_size=64,
            api_key=api_key,
        )
        logger.set_config("agent", "TD-MPC2")
        logger.set_config("env", "HalfCheetah")
        logger.set_config("latent_dim", String(LATENT_DIM))
        logger.set_config("mlp_dim", String(MLP_DIM))
        logger.set_config("num_bins", String(NUM_BINS))
        logger.set_config("num_q", String(NUM_Q))
        logger.set_config("horizon", String(HORIZON))
        logger.set_config("batch_size", String(BATCH_SIZE))
        logger.set_config("n_envs", String(N_ENVS))
        logger.set_config("wm_lr", "3e-4")
        logger.set_config("pi_lr", "3e-4")
        logger.set_config("gamma", "0.99")

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
            gamma=0.99,
            rho=0.5,
            tau=0.01,
            consistency_coef=20.0,
            reward_coef=0.1,  # Reference default
            value_coef=0.1,
            terminal_coef=1.0,
            entropy_coef=1e-4,  # Reference default
            temperature=0.5,
            action_scale=1.0,
            warmup_steps=5_000,
            wm_lr=3e-4,
            enc_lr_scale=0.3,
            pi_lr=3e-4,
            logger=UnsafePointer(to=logger),
            diag_every=50,
        )

        print("Environment: HalfCheetah Continuous (GPU)")
        print("Agent: TD-MPC2 (GPU)")
        print("  Observation dim: " + String(OBS_DIM))
        print("  Action dim: " + String(ACTION_DIM))
        print("  Latent dim: " + String(LATENT_DIM))
        print("  MLP dim: " + String(MLP_DIM))
        print("  Encoder dim: " + String(ENC_DIM))
        print("  Num bins (distributional RL): " + String(NUM_BINS))
        print("  Q-ensemble size: " + String(NUM_Q))
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
        print("  Warmup steps: 5000 (random actions before training)")
        print("  World model LR: 3e-4")
        print("  Encoder LR scale: 0.3 (enc_lr = 9e-5)")
        print("  Policy LR: 3e-4")
        print()
        print("Key hyperparameters:")
        print("  - gamma: 0.99 (discount)")
        print("  - rho: 0.5 (temporal weight decay per horizon step)")
        print("  - tau: 0.01 (target Q soft update)")
        print("  - consistency_coef: 20.0 (latent consistency loss weight)")
        print("  - reward_coef: 0.1 (distributional reward loss weight)")
        print("  - value_coef: 0.1 (distributional Q loss weight)")
        print("  - entropy_coef: 1e-4 (policy entropy regularization)")
        print()
        print("HalfCheetah specifics:")
        print("  - Generalized Coordinates (GC) physics engine")
        print("  - MuJoCo-style joint-space dynamics")
        print("  - 8 bodies: torso, 2 legs (thigh+shin+foot), head")
        print("  - 17D observations: joint positions + velocities")
        print("  - 6D continuous actions (joint torques with gear ratios)")
        print("  - Reward: forward_velocity - ctrl_cost - angle_penalty")
        print()
        print("Expected rewards:")
        print("  - Warmup (random): ~-100 to -200")
        print("  - Early learning: > 0")
        print("  - Good policy: > 500")
        print("  - Running well: > 1000")
        print()

        # =====================================================================
        # Train using train_gpu()
        # =====================================================================

        print("Starting GPU training...")
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
            )

            var end_time = perf_counter_ns()
            var elapsed_s = Float64(end_time - start_time) / 1e9

            logger.close()

            print("-" * 70)
            print()
            print(">>> train_gpu() returned successfully! <<<")

            # =================================================================
            # Summary
            # =================================================================

            print("=" * 70)
            print("GPU Training Complete")
            print("=" * 70)
            print()
            print("Total episodes: " + String(NUM_EPISODES))
            print("Training time: " + String(elapsed_s)[:6] + " seconds")
            print(
                "Episodes/second: "
                + String(Float64(NUM_EPISODES) / elapsed_s)[:7]
            )
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
