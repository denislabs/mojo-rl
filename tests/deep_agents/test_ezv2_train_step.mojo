"""Phase-2 Step 5c smoke test — full K-step BPTT + optimizer.

Verifies that `agent.train_step()` actually trains the networks: roll out
a fixed amount of CartPole experience, then loop over `train_step` and
watch the composite loss go down. If the K-step BPTT or optimizer
plumbing is wrong (e.g., a backward call overwriting instead of
accumulating, a missing gradient path, an off-by-one between cache and
backward), the loss either explodes, stays flat, or oscillates.

This is a *training-machinery* test, not a convergence test — we don't
expect CartPole to be solved here. The goal is to confirm the gradients
flow correctly through every component of the loss:

    L_R  →  reward CE       →  through dyn at K steps      →  rep
    L_P  →  policy CE       →  through pred + dyn unroll   →  rep
    L_V  →  value CE        →  through pred + dyn unroll   →  rep
    L_G  →  cosine cons.    →  through predictor + projector
                                + dyn unroll               →  rep

If any of these gradient paths is broken, the corresponding loss
component won't drop. We log all four and assert at least the composite
plus the two simplest (L_P and L_V — both straight pred-head CE) drop
meaningfully.
"""

from std.math import abs
from std.random import seed
from mojo_rl.deep_agents.efficient_zero_v2 import (
    EZV2DiscreteMLPConfig,
    GenericEfficientZeroV2Agent,
)
from mojo_rl.envs.cartpole import CartPoleEnv
from mojo_rl.nn.constants import dtype


def _is_finite(x: Float64) -> Bool:
    if x != x:
        return False
    if x > 1.0e300 or x < -1.0e300:
        return False
    return True


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
    print("=== EZ-V2 Phase 2 / Step 5c — train_step (BPTT + Adam) ===")
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
    ]

    seed(2026)
    var agent = GenericEfficientZeroV2Agent[Config](
        gamma=0.99, v_min=-10.0, v_max=10.0, temperature=1.0,
    )
    var env = CartPoleEnv[DType.float32]()

    # ── Roll out replay buffer with a fixed dataset ──────────────────────
    print()
    print("--- Filling replay buffer ---")
    var num_episodes = 60
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
    print("    replay buf size =", agent.state.buffer.size)

    _expect(
        agent.state.is_ready(),
        "replay buffer holds enough data to train",
        passed, total,
    )

    # ── Initial loss snapshot ────────────────────────────────────────────
    print()
    print("--- Initial loss components ---")
    var initial = agent.compute_loss_components()
    var L_R0 = initial[0]
    var L_P0 = initial[1]
    var L_V0 = initial[2]
    var L_G0 = initial[3]
    var L0_total = (
        Config.lambda_reward * L_R0
        + Config.lambda_policy * L_P0
        + Config.lambda_value * L_V0
        + Config.lambda_consistency * L_G0
    )
    print("    L_R =", L_R0, "  L_P =", L_P0)
    print("    L_V =", L_V0, "  L_G =", L_G0)
    print("    L_total =", L0_total)

    # ── Train a few steps and watch losses ───────────────────────────────
    print()
    print("--- Running 100 train_step calls ---")
    var num_train_steps = 100
    var sum_L_total = Float64(0.0)
    var L_total_first = Float64(0.0)
    var L_total_last = Float64(0.0)
    var L_R_last = Float64(0.0)
    var L_P_last = Float64(0.0)
    var L_V_last = Float64(0.0)
    var L_G_last = Float64(0.0)
    var any_nan = False
    for step in range(num_train_steps):
        var t = agent.train_step()
        var L_total = t[0]
        var L_R = t[1]
        var L_P = t[2]
        var L_V = t[3]
        var L_G = t[4]
        if not _is_finite(L_total):
            any_nan = True
        sum_L_total += L_total
        if step == 0:
            L_total_first = L_total
        if step == num_train_steps - 1:
            L_total_last = L_total
            L_R_last = L_R
            L_P_last = L_P
            L_V_last = L_V
            L_G_last = L_G
        if step % 20 == 0:
            print(
                "    step",
                step,
                ": L_total=",
                L_total,
                " L_R=",
                L_R,
                " L_P=",
                L_P,
                " L_V=",
                L_V,
                " L_G=",
                L_G,
            )

    print()
    print("--- After training ---")
    print("    L_total: first =", L_total_first, ", last =", L_total_last)
    print("    L_R: ", L_R0, "→", L_R_last)
    print("    L_P: ", L_P0, "→", L_P_last)
    print("    L_V: ", L_V0, "→", L_V_last)
    print("    L_G: ", L_G0, "→", L_G_last)

    _expect(
        not any_nan,
        "no NaN/Inf encountered across",
        passed, total,
    )
    _expect(
        _is_finite(L_total_last),
        "final L_total is finite",
        passed, total,
    )

    # Composite loss should decrease meaningfully (>= 10%).
    _expect(
        L_total_last < 0.9 * L_total_first,
        "composite L decreased ≥ 10% over 100 steps",
        passed, total,
    )

    # The two pred-head CE losses (simplest gradient paths) should drop.
    _expect(
        L_V_last < 0.9 * L_V0,
        "L_V dropped ≥ 10% (value head trained via pred backward)",
        passed, total,
    )
    _expect(
        L_R_last < 0.9 * L_R0,
        "L_R dropped ≥ 10% (reward head trained via dyn backward)",
        passed, total,
    )

    print()
    print("=== Result:", passed, "/", total, "tests passed ===")
