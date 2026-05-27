"""SAC Agent CPU Training on BipedalWalker.

Trains SAC on the native BipedalWalker environment:
- 24D observation: hull state + joint states + lidar
- 4D continuous action: hip and knee torques in [-1, 1]

BipedalWalker is solved when average reward > 300 over 100 episodes.

Run with:
    pixi run mojo run -I . examples/sac_bipedal_walker_training_cpu.mojo
"""

from std.random import seed
from std.time import perf_counter_ns

from mojo_rl.deep_agents.core.agents import DeepSACAgent
from mojo_rl.envs.bipedal_walker import BipedalWalker, BWConstants


comptime OBS_DIM = BWConstants.OBS_DIM_VAL  # 24
comptime ACTION_DIM = BWConstants.ACTION_DIM_VAL  # 4
comptime HIDDEN_DIM = 256
comptime BUFFER_CAPACITY = 100_000
comptime BATCH_SIZE = 256
comptime NUM_STEPS = 500_000
comptime MAX_STEPS_PER_EPISODE = 1600
comptime WARMUP_STEPS = 10_000
comptime dtype = DType.float32


def main() raises:
    seed(42)
    print("=" * 70)
    print("SAC Agent CPU Training on BipedalWalker")
    print("=" * 70)
    print()

    var env = BipedalWalker[dtype](seed=42)

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
        target_entropy=-4.0,
        use_ere=True,
        ere_eta=0.996,
    )

    print("Environment: BipedalWalker (CPU)")
    print("Agent: SAC (Soft Actor-Critic)")
    print("  Observation dim:", OBS_DIM)
    print("  Action dim:", ACTION_DIM)
    print("  Hidden dim:", HIDDEN_DIM)
    print("  Buffer capacity:", BUFFER_CAPACITY)
    print("  Batch size:", BATCH_SIZE)
    print("  Total steps:", NUM_STEPS)
    print("  Max steps/episode:", MAX_STEPS_PER_EPISODE)
    print("  Warmup steps:", WARMUP_STEPS)
    print("  ERE: enabled (eta=0.996)")
    print()
    print("Expected rewards:")
    print("  - Random policy: ~-100")
    print("  - Learning policy: > 0")
    print("  - Good walking: > 200")
    print("  - Solved: > 300")
    print()

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
        environment_name="BipedalWalker",
    )

    var elapsed_s = Float64(perf_counter_ns() - start_time) / 1e9

    print("-" * 70)
    print()
    print("=" * 70)
    print("BipedalWalker CPU Training Complete")
    print("=" * 70)
    print()
    print("Total steps:", NUM_STEPS)
    print("Training time:", String(elapsed_s)[byte=:6], "seconds")
    print()

    var final_avg = metrics.mean_reward_last_n(100)
    print(
        "Final average reward (last 100 episodes):",
        String(final_avg)[byte=:10],
    )
    print("Best episode reward:", String(metrics.max_reward())[byte=:10])
    print()

    if final_avg > 300.0:
        print("SOLVED: Average reward > 300!")
    elif final_avg > 200.0:
        print("EXCELLENT: Agent walking well (avg > 200)")
    elif final_avg > 100.0:
        print("GOOD: Agent learning to walk (avg > 100)")
    elif final_avg > 0.0:
        print("LEARNING: Agent improving (avg > 0)")
    else:
        print("EARLY STAGE: Agent still exploring (avg < 0)")

    print()
    print("=" * 70)
