"""SAC Agent CPU Training on LunarLander (Continuous).

Trains SAC on the native LunarLander environment using continuous actions:
- 8D observation: [x, y, vx, vy, theta, omega, left_contact, right_contact]
- 2D continuous action: [main_throttle, side_throttle] in [-1, 1]

LunarLander is solved when average reward > 200 over 100 episodes.

Run with:
    pixi run mojo run -I . examples/sac_lunar_lander_training_cpu.mojo
"""

from std.random import seed
from std.time import perf_counter_ns

from mojo_rl.deep_agents.core.agents import DeepSACAgent
from mojo_rl.envs.lunar_lander import LunarLander


comptime OBS_DIM = 8
comptime ACTION_DIM = 2
comptime HIDDEN_DIM = 128
comptime BUFFER_CAPACITY = 100_000
comptime BATCH_SIZE = 64
comptime NUM_STEPS = 500_000
comptime MAX_STEPS_PER_EPISODE = 1000
comptime WARMUP_STEPS = 5_000
comptime dtype = DType.float32


def main() raises:
    seed(42)
    print("=" * 70)
    print("SAC Agent CPU Training on LunarLander (Continuous)")
    print("=" * 70)
    print()

    var env = LunarLander[dtype]()

    var agent = DeepSACAgent[
        obs_dim=OBS_DIM,
        action_dim=ACTION_DIM,
        hidden_dim=HIDDEN_DIM,
        buffer_capacity=BUFFER_CAPACITY,
        batch_size=BATCH_SIZE,
        actor_lr=0.0003,
        critic_lr=0.001,
    ](
        gamma=0.99,
        tau=0.005,
        action_scale=1.0,
        alpha=0.2,
        auto_alpha=True,
        alpha_lr=0.0003,
        target_entropy=-2.0,
        use_ere=True,
        ere_eta=0.996,
    )

    print("Environment: LunarLander Continuous (CPU)")
    print("Agent: SAC (Soft Actor-Critic)")
    print("  Observation dim:", OBS_DIM)
    print("  Action dim:", ACTION_DIM)
    print("  Hidden dim:", HIDDEN_DIM)
    print("  Buffer capacity:", BUFFER_CAPACITY)
    print("  Batch size:", BATCH_SIZE)
    print("  Total steps:", NUM_STEPS)
    print("  Max steps/episode:", MAX_STEPS_PER_EPISODE)
    print("  Warmup steps:", WARMUP_STEPS)
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
        environment_name="LunarLander",
    )

    var end_time = perf_counter_ns()
    var elapsed_s = Float64(end_time - start_time) / 1e9

    print("-" * 70)
    print()
    print("=" * 70)
    print("LunarLander CPU Training Complete")
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

    if final_avg > 200.0:
        print("SOLVED: Average reward > 200!")
    elif final_avg > 100.0:
        print("GOOD: Agent landing consistently (avg > 100)")
    elif final_avg > 0.0:
        print("LEARNING: Agent improving (avg > 0)")
    else:
        print("EARLY STAGE: Agent still exploring (avg < 0)")

    print()
    print("=" * 70)
