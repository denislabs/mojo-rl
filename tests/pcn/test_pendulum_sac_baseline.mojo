"""Pendulum SAC — Phase-2 baseline (raw 3D observation, no encoder).

Trains SAC on the standard Pendulum environment with the raw `[cos θ, sin θ,
θ_dot]` observation. Logs the per-episode return so this can be compared
to the PCN-encoded and MLP-encoded variants.

Same SAC hyperparameters as `examples/pendulum_deep_sac.mojo`: hidden=64,
buffer 50K, batch 64, γ=0.99, τ=0.005, action_scale=2.0. We override
`auto_alpha=True` (vs the example's False) so all three Phase-2 variants
use the same standard SAC recipe.

See `docs/PCN_MBRL_PLAN.md` Phase 2 for context. Comparison only — no
pass/fail threshold; the writeup uses the per-episode return curve.

Run:
    pixi run mojo run -I . tests/pcn/test_pendulum_sac_baseline.mojo
"""

from std.time import perf_counter_ns

from mojo_rl.envs import PendulumEnv
from mojo_rl.deep_agents.core.agents import DeepSACAgent


# Shared SAC hyperparameters — identical across the three Phase-2 variants.
comptime NUM_STEPS = 40_000
comptime MAX_STEPS = 200
comptime WARMUP_STEPS = 1000
comptime PRINT_EVERY = 20
comptime EVAL_EPISODES = 10


def main() raises:
    print("=" * 60)
    print("Pendulum SAC — Phase-2 baseline (raw 3D obs)")
    print("=" * 60)
    print("  Obs        : raw [cos θ, sin θ, θ_dot] (3D)")
    print("  Action     : 1D continuous torque ∈ [-2, 2]")
    print("  SAC arch   : hidden=64, twin Q, auto-α")
    print("  Steps      :", NUM_STEPS, "  max_steps/ep:", MAX_STEPS)
    print("  Warmup     :", WARMUP_STEPS)

    var env = PendulumEnv[DType.float64]()

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
        auto_alpha=True,  # Standard SAC: auto-tuned entropy temperature.
        alpha_lr=0.0001,
    )

    print("\n  --- training ---")
    var t0 = perf_counter_ns()
    var metrics = agent.train(
        env,
        num_steps=NUM_STEPS,
        max_steps_per_episode=MAX_STEPS,
        warmup_steps=WARMUP_STEPS,
        train_every=1,
        verbose=True,
        print_every=PRINT_EVERY,
        environment_name="Pendulum (baseline)",
    )
    var train_t = Float64(perf_counter_ns() - t0) / 1e9

    # Eval pass (deterministic policy).
    print("\n  --- eval (deterministic policy) ---")
    var eval_reward = agent.evaluate(
        env,
        num_episodes=EVAL_EPISODES,
        max_steps_per_episode=MAX_STEPS,
    )

    # Dump per-episode return curve in CSV form so it can be diffed against
    # PCN- and MLP-encoded variants.
    print("\n  === per-episode returns (CSV: ep,return,steps) ===")
    var rewards = metrics.get_rewards()
    var steps = metrics.get_steps()
    for i in range(len(rewards)):
        print("  CSV:", i, ",", rewards[i], ",", steps[i])

    print("\n  === Phase-2 baseline summary ===")
    print("  Train wall :", train_t, "s")
    print("  Final α    :", String(agent.alpha)[byte=:6])
    print("  Eval avg   :", eval_reward, " (", EVAL_EPISODES, " eps)")
    print("  Last-20 avg:", metrics.mean_reward_last_n(20))
    print("=== Done ===")
