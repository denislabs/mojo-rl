"""PPO Continuous Agent CPU Training on LunarLander.

Trains PPO with continuous actions on a single LunarLander environment (CPU).
Faster than GPU on Apple Silicon due to kernel launch overhead.

- 8D observation: [x, y, vx, vy, angle, ang_vel, left_leg, right_leg]
- 2D continuous action: [main_throttle, side_control] in [-1, 1]

Run with:
    pixi run mojo run -I . examples/lunar_lander/ppo_lunar_continuous_cpu.mojo
"""

from std.random import seed
from std.time import perf_counter_ns

from mojo_rl.deep_agents.core.agents import DeepPPOContinuousAgent
from mojo_rl.envs.lunar_lander import LunarLander, LLConstants


comptime OBS_DIM = LLConstants.OBS_DIM_VAL  # 8
comptime ACTION_DIM = LLConstants.ACTION_DIM_VAL  # 2
comptime HIDDEN_DIM = 256
comptime ROLLOUT_LEN = 128
comptime NUM_UPDATES = 10_000
comptime dtype = DType.float32


def main() raises:
    seed(42)
    print("=" * 70)
    print("PPO Continuous Agent CPU Training on LunarLander")
    print("=" * 70)
    print()

    var env = LunarLander[dtype]()

    var agent = DeepPPOContinuousAgent[
        obs_dim=OBS_DIM,
        action_dim=ACTION_DIM,
        hidden_dim=HIDDEN_DIM,
        rollout_len=ROLLOUT_LEN,
        actor_lr=0.0003,
        critic_lr=0.001,
    ](
        gamma=0.99,
        gae_lambda=0.95,
        clip_epsilon=0.2,
        entropy_coef=0.01,
        value_loss_coef=0.5,
        num_epochs=10,
        target_kl=0.1,
        max_grad_norm=0.5,
        clip_value=True,
        norm_adv_per_minibatch=True,
    )

    print("Environment: LunarLander Continuous (CPU)")
    print("Agent: PPO Continuous")
    print("  Observation dim:", OBS_DIM)
    print("  Action dim:", ACTION_DIM)
    print("  Hidden dim:", HIDDEN_DIM)
    print("  Rollout length:", ROLLOUT_LEN)
    print("  Num updates:", NUM_UPDATES)
    print("  Total transitions:", ROLLOUT_LEN * NUM_UPDATES)
    print()
    print("Expected rewards:")
    print("  - Random policy: ~-200 to -400")
    print("  - Learning policy: > -100")
    print("  - Good policy: > 0")
    print("  - Successful landing: > 100")
    print()

    print("Starting CPU training...")
    print("-" * 70)

    var start_time = perf_counter_ns()

    var metrics = agent.train(
        env,
        num_updates=NUM_UPDATES,
        verbose=True,
        print_every=100,
        environment_name="LunarLander",
    )

    var elapsed_s = Float64(perf_counter_ns() - start_time) / 1e9

    print("-" * 70)
    print()
    print("=" * 70)
    print("LunarLander CPU Training Complete")
    print("=" * 70)
    print()
    print("Total updates:", NUM_UPDATES)
    print("Total transitions:", ROLLOUT_LEN * NUM_UPDATES)
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
        print("EXCELLENT: Agent landing consistently (avg > 100)")
    elif final_avg > 0.0:
        print("GOOD: Agent learned to land (avg > 0)")
    elif final_avg > -100.0:
        print("LEARNING: Agent improving (avg > -100)")
    else:
        print("EARLY STAGE: Agent still exploring (avg < -100)")

    print()
    print("=" * 70)
