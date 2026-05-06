"""Phase-2 mixed-value-target wiring test.

Verifies that `train_step` / `compute_loss_components` actually use
`MixedValueTarget[T_FRESH, T_STALE].compute(sve, td, age)` for the L_V
target — not just SVE — by forcing the per-transition `step_at_write`
stamps to two extreme age regimes on the same buffer and confirming L_V
differs.

Procedure:
  1. Roll out CartPole to fill the replay buffer.
  2. With all `step_at_write[i] = train_step_count` (age = 0) → blend
     yields pure SVE.
  3. With all `step_at_write[i] = 0` and a high `train_step_count`
     (age ≫ T_STALE) → blend yields pure n-step TD.
  4. Compute L_V under both regimes (with the same RNG seed for
     sampling) and assert they differ — that's the smoking gun the
     wiring is active.

If both produce the same L_V the wiring is dead and the agent is
silently still using stored MCTS root values regardless of staleness.
"""

from std.math import abs
from std.random import seed
from mojo_rl.deep_agents.efficient_zero_v2 import (
    EZV2DiscreteMLPConfig,
    GenericEfficientZeroV2Agent,
    VALUE_TARGET_MIXED,
)
from mojo_rl.envs.cartpole import CartPoleEnv
from mojo_rl.nn.constants import dtype


def _expect(
    cond: Bool,
    label: String,
    mut passed: Int,
    mut total: Int,
):
    total += 1
    if cond:
        print("PASS:", label)
        passed += 1
    else:
        print("FAIL:", label)


def main():
    print("=== EZ-V2 mixed-value-target wiring test ===")
    var passed = 0
    var total = 0

    comptime Config = EZV2DiscreteMLPConfig[
        OBS=4,
        ACT=2,
        LATENT=32,
        HIDDEN=32,
        PROJ=64,
        PRED_BOTTLENECK=32,
        BINS=21,
        BS=8,
        K_UNROLL=3,
        N_TD=5,
        SIMS=8,
        NODES=32,
        K_GUMBEL=2,
        # Tight thresholds so it's easy to flip between fresh/stale
        # regimes by forcing step_at_write.
        T_FRESH=10,
        T_STALE=20,
        # Test specifically validates the MixedValueTarget blend.
        VALUE_TARGET_MODE=VALUE_TARGET_MIXED,
    ]

    seed(2026)
    var agent = GenericEfficientZeroV2Agent[Config](
        gamma=0.99, v_min=-10.0, v_max=10.0, temperature=1.0,
    )
    var env = CartPoleEnv[DType.float32]()

    # Roll out enough to fill replay buffer.
    var num_episodes = 50
    var max_steps_per_ep = 60
    for _ep in range(num_episodes):
        var obs = env.reset_obs_list()
        for _step in range(max_steps_per_ep):
            var result = agent.select_action(obs, training=True)
            var action = result[0]
            var policy = result[1]
            var root_value = result[2]
            var step_result = env.step_obs(action)
            var next_obs = step_result[0].copy()
            var reward = Float64(step_result[1])
            var done = step_result[2]
            agent.store_transition(
                obs, action, reward, policy, root_value, done
            )
            obs = next_obs^
            if done:
                break
    print()
    print("--- Replay state ---")
    print("    buf size           =", agent.state.buffer.size)
    print("    train_step_count   =", agent.train_step_count)
    print("    Config.t_fresh     =", Config.t_fresh)
    print("    Config.t_stale     =", Config.t_stale)

    var n = agent.state.buffer.size

    # ── Regime A: every transition fresh (age = 0) ───────────────────────
    # Set step_at_write[i] = train_step_count for all valid indices.
    # train_step_count is currently 0 (we haven't called train_step), so
    # this leaves the buffer in its initial all-zero state — but we'll
    # set it explicitly for clarity.
    for i in range(n):
        agent.state.step_at_write[i] = Scalar[DType.uint32](
            agent.train_step_count
        )

    seed(99)
    var fresh = agent.compute_loss_components()
    var L_V_fresh = fresh[2]

    # ── Regime B: every transition stale (age ≫ T_STALE) ─────────────────
    # Bump train_step_count to a value past Config.t_stale and clear
    # step_at_write so age = train_step_count - 0 ≫ t_stale.
    agent.train_step_count = 1000
    for i in range(n):
        agent.state.step_at_write[i] = Scalar[DType.uint32](0)

    seed(99)
    var stale = agent.compute_loss_components()
    var L_V_stale = stale[2]

    # Restore agent state (in case anything downstream depends on it).
    agent.train_step_count = 0
    for i in range(n):
        agent.state.step_at_write[i] = Scalar[DType.uint32](0)

    print()
    print("--- L_V under two age regimes ---")
    print("    fresh (age=0, → pure SVE)         L_V =", L_V_fresh)
    print("    stale (age=1000 ≫ t_stale=20,    L_V =", L_V_stale)
    print("           → pure n-step TD)")
    var diff = L_V_fresh - L_V_stale
    if diff < 0.0:
        diff = -diff
    print("    |fresh - stale|                    =", diff)

    _expect(
        L_V_fresh != L_V_stale,
        "L_V differs between fresh-age and stale-age regimes",
        passed, total,
    )
    # Sanity: both should still be finite + ≥ 0 (cross-entropy).
    _expect(
        L_V_fresh >= -1e-6 and L_V_stale >= -1e-6,
        "both L_V values are non-negative (cross-entropy convention)",
        passed, total,
    )
    # The two values should be meaningfully apart, not just float noise.
    _expect(
        diff > 0.01,
        "fresh vs stale L_V differ by more than float noise (≥ 0.01)",
        passed, total,
    )

    print()
    print("=== Result:", passed, "/", total, "tests passed ===")
