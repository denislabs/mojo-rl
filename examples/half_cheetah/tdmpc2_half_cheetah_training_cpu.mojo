"""TD-MPC2 Agent CPU Training on HalfCheetah.

Train TDMPC2 on HalfCheetah using CPU-based world model learning and
MPPI planning in latent space:
- World model: encoder, dynamics, reward, termination, policy, Q-ensemble
- MPPI planning with horizon H=3, 512 samples, 6 iterations
- Distributional RL with 101 bins (two-hot targets)
- 5-network Q-ensemble for robust value estimation
- Replay buffer with sequence sampling for temporal consistency

Action space (6D continuous):
- action[0]: back thigh (hip) torque (-1.0 to 1.0) * gear=120
- action[1]: back shin (knee) torque (-1.0 to 1.0) * gear=90
- action[2]: back foot (ankle) torque (-1.0 to 1.0) * gear=60
- action[3]: front thigh (hip) torque (-1.0 to 1.0) * gear=120
- action[4]: front shin (knee) torque (-1.0 to 1.0) * gear=60
- action[5]: front foot (ankle) torque (-1.0 to 1.0) * gear=30

Run with:
    pixi run mojo run examples/half_cheetah/tdmpc2_half_cheetah_training_cpu.mojo
"""

from random import seed
from time import perf_counter_ns

from deep_agents.tdmpc2 import TDMPC2Agent
from envs.half_cheetah import (
    HalfCheetah,
    HalfCheetahConfig,
)


# =============================================================================
# Constants
# =============================================================================

# HalfCheetah: 17D observation, 6D continuous action
comptime OBS_DIM = HalfCheetahConfig.OBS_DIM  # 17
comptime ACTION_DIM = HalfCheetahConfig.ACTION_DIM  # 6

# Network architecture (TD-MPC2 defaults)
comptime LATENT_DIM = 256
comptime MLP_DIM = 256

# Distributional RL
comptime NUM_BINS = 101

# Q-ensemble size
comptime NUM_Q = 5

# MPPI planning parameters
comptime HORIZON = 3  # Planning horizon
comptime NUM_SAMPLES = 512  # MPPI candidate trajectories
comptime NUM_PI_TRAJS = 24  # Policy-seeded trajectories
comptime NUM_ITERATIONS = 6  # MPPI optimization iterations

# Replay buffer and batch
comptime BATCH_SIZE = 256
comptime BUFFER_CAPACITY = 100_000

# Value range for distributional RL
comptime V_MIN = -10.0
comptime V_MAX = 10.0

# Training duration
comptime NUM_EPISODES = 1_000

comptime dtype = DType.float64  # Physics precision


# =============================================================================
# Main
# =============================================================================


fn main() raises:
    seed(42)
    print("=" * 70)
    print("TD-MPC2 Agent CPU Training on HalfCheetah")
    print("=" * 70)
    print()

    # =========================================================================
    # Create agent and environment
    # =========================================================================

    var agent = TDMPC2Agent[
        obs_dim=OBS_DIM,
        action_dim=ACTION_DIM,
        latent_dim=LATENT_DIM,
        mlp_dim=MLP_DIM,
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
    ](
        gamma=0.99,
        rho=0.5,
        tau=0.01,
        consistency_coef=2.0,
        reward_coef=0.5,
        value_coef=0.1,
        terminal_coef=1.0,
        entropy_coef=1e-4,
        temperature=0.5,
        action_scale=1.0,
        warmup_steps=5_000,
        wm_lr=3e-4,
        enc_lr_scale=0.3,
        pi_lr=3e-4,
    )

    var env = HalfCheetah[dtype, TERMINATE_ON_UNHEALTHY=False]()

    print("Environment: HalfCheetah Continuous (CPU)")
    print("Agent: TD-MPC2 (CPU)")
    print("  Observation dim: " + String(OBS_DIM))
    print("  Action dim: " + String(ACTION_DIM))
    print("  Latent dim: " + String(LATENT_DIM))
    print("  MLP dim: " + String(MLP_DIM))
    print("  Num bins (distributional RL): " + String(NUM_BINS))
    print("  Q-ensemble size: " + String(NUM_Q))
    print()
    print("MPPI Planning:")
    print("  Horizon: " + String(HORIZON))
    print("  Num samples: " + String(NUM_SAMPLES))
    print("  Num policy trajs: " + String(NUM_PI_TRAJS))
    print("  Num iterations: " + String(NUM_ITERATIONS))
    print()
    print("Training:")
    print("  Episodes: " + String(NUM_EPISODES))
    print("  Batch size: " + String(BATCH_SIZE))
    print("  Buffer capacity: " + String(BUFFER_CAPACITY))
    print("  Warmup steps: 5000 (random actions before training)")
    print("  World model LR: 3e-4")
    print("  Encoder LR scale: 0.3 (enc_lr = 9e-5)")
    print("  Policy LR: 3e-4")
    print()
    print("Key hyperparameters:")
    print("  - gamma: 0.99 (discount)")
    print("  - rho: 0.5 (temporal weight decay)")
    print("  - tau: 0.01 (target Q soft update)")
    print("  - consistency_coef: 2.0")
    print("  - reward_coef: 0.5")
    print("  - value_coef: 0.1")
    print("  - temperature: 0.5 (MPPI softmax)")
    print()
    print("Expected rewards:")
    print("  - Warmup (random): ~-100 to -200")
    print("  - Early learning: > 0")
    print("  - Good policy: > 500")
    print("  - Running well: > 1000")
    print()

    # =========================================================================
    # Train
    # =========================================================================

    print("Starting CPU training...")
    print("-" * 70)

    var start_time = perf_counter_ns()

    var metrics = agent.train(
        env,
        num_episodes=NUM_EPISODES,
        updates_per_step=1,
    )

    var end_time = perf_counter_ns()
    var elapsed_s = Float64(end_time - start_time) / 1e9

    print("-" * 70)
    print()
    print(">>> train() returned successfully! <<<")

    # =========================================================================
    # Summary
    # =========================================================================

    print("=" * 70)
    print("Training Complete")
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
            "GOOD PROGRESS: Agent is learning locomotion (avg reward > 100)"
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

    print(">>> main() completed normally <<<")
