"""SAC Agent CPU Training on HalfCheetah.

This trains the SAC (Soft Actor-Critic) agent on the HalfCheetah environment
using single-env CPU training with:
- Generalized Coordinates (GC) physics engine (MuJoCo-style)
- 6D continuous action space (joint torques)
- 17D observation (qpos + qvel excluding rootx and head)

SAC key features:
- Maximum entropy RL (reward + alpha * entropy)
- Stochastic Gaussian policy (reparameterization trick)
- Twin Q-networks (min of Q1, Q2 reduces overestimation)
- Automatic entropy temperature (alpha) tuning
- No target actor (only critic targets)

Run with:
    pixi run mojo run -I . examples/half_cheetah/sac_half_cheetah_training_cpu.mojo
"""

from std.random import seed
from std.time import perf_counter_ns

from mojo_rl.deep_agents.core.agents import DeepSACAgent
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

# Network architecture
comptime HIDDEN_DIM = 256

# Off-policy CPU training parameters
comptime BUFFER_CAPACITY = 100_000
comptime BATCH_SIZE = 64

# Training duration
comptime NUM_STEPS = 500_000
comptime MAX_STEPS_PER_EPISODE = 1000
comptime WARMUP_STEPS = 1_000

comptime dtype = DType.float64


# =============================================================================
# Main
# =============================================================================


def main() raises:
    seed(42)
    print("=" * 70)
    print("SAC Agent CPU Training on HalfCheetah")
    print("=" * 70)
    print()

    # =========================================================================
    # Create environment and agent
    # =========================================================================

    var env = HalfCheetah[dtype, TERMINATE_ON_UNHEALTHY=False]()

    var agent = DeepSACAgent[
        obs_dim=OBS_DIM,
        action_dim=ACTION_DIM,
        hidden_dim=HIDDEN_DIM,
        buffer_capacity=BUFFER_CAPACITY,
        batch_size=BATCH_SIZE,
        actor_lr=0.0003,
        critic_lr=0.0003,
    ](
        gamma=0.99,
        tau=0.005,
        action_scale=1.0,
        alpha=0.2,
        auto_alpha=True,
        alpha_lr=0.0003,
    )

    print("Environment: HalfCheetah Continuous (CPU)")
    print("Agent: SAC (Soft Actor-Critic)")
    print("  Observation dim: " + String(OBS_DIM))
    print("  Action dim: " + String(ACTION_DIM))
    print("  Hidden dim: " + String(HIDDEN_DIM))
    print("  Buffer capacity: " + String(BUFFER_CAPACITY))
    print("  Batch size: " + String(BATCH_SIZE))
    print("  Total steps: " + String(NUM_STEPS))
    print("  Max steps/episode: " + String(MAX_STEPS_PER_EPISODE))
    print("  Warmup steps: " + String(WARMUP_STEPS))
    print()

    # =========================================================================
    # Train
    # =========================================================================

    print("Starting CPU training...")
    print("-" * 70)

    var start_time = perf_counter_ns()

    var metrics = agent.train(
        env,
        num_steps=NUM_STEPS,
        max_steps_per_episode=MAX_STEPS_PER_EPISODE,
        warmup_steps=WARMUP_STEPS,
        train_every=1,
        verbose=True,
        print_every=10_000,
        environment_name="HalfCheetah",
    )

    var end_time = perf_counter_ns()
    var elapsed_s = Float64(end_time - start_time) / 1e9

    # =========================================================================
    # Summary
    # =========================================================================

    print("-" * 70)
    print()
    print("=" * 70)
    print("CPU Training Complete")
    print("=" * 70)
    print()
    print("Total steps: " + String(NUM_STEPS))
    print("Training time: " + String(elapsed_s)[byte=:6] + " seconds")
    print()

    print(
        "Final average reward (last 100 episodes): "
        + String(metrics.mean_reward_last_n(100))[byte=:8]
    )
    print("Best episode reward: " + String(metrics.max_reward())[byte=:8])
    print()

    var final_avg = metrics.mean_reward_last_n(100)
    if final_avg > 1000.0:
        print("EXCELLENT: Agent is running fast! (avg reward > 1000)")
    elif final_avg > 500.0:
        print("SUCCESS: Agent learned to run! (avg reward > 500)")
    elif final_avg > 100.0:
        print("GOOD PROGRESS: Agent is learning locomotion (avg reward > 100)")
    elif final_avg > 0.0:
        print(
            "LEARNING: Agent improving but needs more training (avg reward > 0)"
        )
    else:
        print("EARLY STAGE: Agent still exploring (avg reward < 0)")

    print()
    print("=" * 70)
