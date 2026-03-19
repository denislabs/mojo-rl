"""TD3 Agent CPU Training on HalfCheetah.

This trains the TD3 (Twin Delayed DDPG) agent on the HalfCheetah environment
using single-env CPU training with:
- Generalized Coordinates (GC) physics engine (MuJoCo-style)
- 6D continuous action space (joint torques)
- 17D observation (qpos + qvel excluding rootx and head)

TD3 key features:
- Twin Q-networks (min of Q1, Q2 reduces overestimation)
- Delayed policy updates (actor updated every 2 critic updates)
- Target policy smoothing (clipped noise on target actions)

Run with:
    pixi run mojo run -I . examples/half_cheetah/td3_half_cheetah_training_cpu.mojo
"""

from std.random import seed
from std.time import perf_counter_ns

from mojo_rl.deep_agents.core.agents import DeepTD3Agent
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
comptime HIDDEN_DIM = 64

# Off-policy CPU training parameters
comptime BUFFER_CAPACITY = 100_000
comptime BATCH_SIZE = 64

# Training duration
comptime NUM_EPISODES = 1000
comptime MAX_STEPS_PER_EPISODE = 1000
comptime WARMUP_STEPS = 1_000

comptime dtype = DType.float64


# =============================================================================
# Main
# =============================================================================


fn main() raises:
    seed(42)
    print("=" * 70)
    print("TD3 Agent CPU Training on HalfCheetah")
    print("=" * 70)
    print()

    # =========================================================================
    # Create environment and agent
    # =========================================================================

    var env = HalfCheetah[dtype, TERMINATE_ON_UNHEALTHY=False]()

    var agent = DeepTD3Agent[
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
        noise_std=0.1,
        noise_std_min=0.01,
        noise_decay=0.995,
        policy_delay=2,
        target_noise_std=0.2,
        target_noise_clip=0.5,
    )

    print("Environment: HalfCheetah Continuous (CPU)")
    print("Agent: TD3 (Twin Delayed DDPG)")
    print("  Observation dim: " + String(OBS_DIM))
    print("  Action dim: " + String(ACTION_DIM))
    print("  Hidden dim: " + String(HIDDEN_DIM))
    print("  Buffer capacity: " + String(BUFFER_CAPACITY))
    print("  Batch size: " + String(BATCH_SIZE))
    print("  Episodes: " + String(NUM_EPISODES))
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
        num_episodes=NUM_EPISODES,
        max_steps_per_episode=MAX_STEPS_PER_EPISODE,
        warmup_steps=WARMUP_STEPS,
        train_every=1,
        verbose=True,
        print_every=10,
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
    print("Total episodes: " + String(NUM_EPISODES))
    print("Training time: " + String(elapsed_s)[:6] + " seconds")
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
        print("GOOD PROGRESS: Agent is learning locomotion (avg reward > 100)")
    elif final_avg > 0.0:
        print(
            "LEARNING: Agent improving but needs more training (avg reward > 0)"
        )
    else:
        print("EARLY STAGE: Agent still exploring (avg reward < 0)")

    print()
    print("=" * 70)
