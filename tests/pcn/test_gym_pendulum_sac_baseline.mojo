"""Gymnasium Pendulum-v1 SAC — Phase-2 baseline (raw 3D obs).

Identical to `test_pendulum_sac_baseline.mojo` but uses `GymPendulumEnv`
(Python wrapper around `gymnasium.make('Pendulum-v1')`) instead of our
native `PendulumEnv`. Diagnostic to rule out an env-specific bug in the
native implementation as a cause of the Phase-2 SAC results.

Same SAC hyperparameters: hidden=64, buffer 50K, batch 64, γ=0.99, τ=0.005,
action_scale=2.0, auto_alpha=True, 200 episodes × 200 steps.

Run:
    pixi run mojo run -I . tests/pcn/test_gym_pendulum_sac_baseline.mojo
"""

from std.time import perf_counter_ns

from mojo_rl.envs.gymnasium.gymnasium_classic_control import GymPendulumEnv
from mojo_rl.deep_agents.core.agents import DeepSACAgent


comptime NUM_EPISODES = 200
comptime MAX_STEPS = 200
comptime WARMUP_STEPS = 1000
comptime PRINT_EVERY = 20


def main() raises:
    print("=" * 60)
    print("Gymnasium Pendulum-v1 SAC — Phase-2 baseline (raw 3D obs)")
    print("=" * 60)
    print("  Env        : GymPendulumEnv (gymnasium.make('Pendulum-v1'))")
    print("  Obs        : raw [cos θ, sin θ, θ_dot] (3D)")
    print("  Action     : 1D continuous torque ∈ [-2, 2]")
    print("  SAC arch   : hidden=64, twin Q, auto-α")
    print("  Episodes   :", NUM_EPISODES, "  steps/ep:", MAX_STEPS)

    var env = GymPendulumEnv(render_mode="")

    var agent = DeepSACAgent[
        obs_dim=3,
        action_dim=1,
        hidden_dim=64,
        buffer_capacity=50000,
        batch_size=64,
        actor_lr=0.0003,
        critic_lr=0.0003,
    ](
        gamma=0.99,
        tau=0.005,
        action_scale=2.0,
        alpha=0.1,
        auto_alpha=True,
        alpha_lr=0.0001,
    )

    print("\n  --- training ---")
    var t0 = perf_counter_ns()
    var metrics = agent.train(
        env,
        num_episodes=NUM_EPISODES,
        max_steps_per_episode=MAX_STEPS,
        warmup_steps=WARMUP_STEPS,
        train_every=1,
        verbose=True,
        print_every=PRINT_EVERY,
        environment_name="GymPendulum (baseline)",
    )
    var train_t = Float64(perf_counter_ns() - t0) / 1e9

    print("\n  === per-episode returns (CSV: ep,return,steps) ===")
    var rewards = metrics.get_rewards()
    var steps = metrics.get_steps()
    for i in range(len(rewards)):
        print("  CSV:", i, ",", rewards[i], ",", steps[i])

    print("\n  === Gym Pendulum baseline summary ===")
    print("  Train wall :", train_t, "s")
    print("  Final α    :", String(agent.alpha)[byte=:6])
    print("  Last-20 avg:", metrics.mean_reward_last_n(20))
    print("=== Done ===")
