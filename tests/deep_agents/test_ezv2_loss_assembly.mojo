"""Phase-2 Step 5b smoke test — forward-only K-step loss assembly.

Drives `agent.compute_loss_components()` after a CartPole rollout fills
the replay buffer. Verifies:

  1. All four loss components (L_R, L_P, L_V, L_G) are finite.
  2. Each is in its respective sane bound:
        L_R, L_P, L_V ≥ 0          (cross-entropy is non-negative)
        L_G ∈ [-1, 1]              (negative cosine is bounded)
  3. The composite L = λ_R·L_R + λ_P·L_P + λ_V·L_V + λ_G·L_G is finite.
  4. compute_loss_components is deterministic-up-to-sampling: a second
     call with the same RNG state produces identical results.

Step 5c will add the backward pass + optimizer; at that point we'll
re-test the same agent and watch the losses *decrease* over training
steps. For now this is a forward-only sanity check that the full K-step
plumbing — sample + rep + K×dyn + (K+1)×pred + K×projector + K×predictor
+ K×rep on obs branch + K×projector on obs branch — runs end-to-end
without numerical issues on real replay data.
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
    print("=== EZ-V2 Phase 2 / Step 5b — forward-only loss assembly ===")
    var passed = 0
    var total = 0

    # Tiny config tuned for fast smoke test.
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

    # ── Roll out ~50 episodes to fill replay buffer ──────────────────────
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
    print("--- After rollout ---")
    print("    replay buf size =", agent.state.buffer.size)
    print("    state.is_ready  =", agent.state.is_ready())

    # ── 1. Forward-only loss assembly ────────────────────────────────────
    print()
    print("--- compute_loss_components ---")
    var losses = agent.compute_loss_components()
    var L_R = losses[0]
    var L_P = losses[1]
    var L_V = losses[2]
    var L_G = losses[3]

    print("    L_R (reward CE)       =", L_R)
    print("    L_P (policy CE)       =", L_P)
    print("    L_V (value CE)        =", L_V)
    print("    L_G (consistency cos) =", L_G)

    _expect(_is_finite(L_R), "L_R is finite", passed, total)
    _expect(_is_finite(L_P), "L_P is finite", passed, total)
    _expect(_is_finite(L_V), "L_V is finite", passed, total)
    _expect(_is_finite(L_G), "L_G is finite", passed, total)

    # Cross-entropy losses must be ≥ 0 (with float-noise tolerance).
    _expect(L_R >= -1e-6, "L_R ≥ 0 (cross-entropy non-negative)", passed, total)
    _expect(L_P >= -1e-6, "L_P ≥ 0", passed, total)
    _expect(L_V >= -1e-6, "L_V ≥ 0", passed, total)

    # Cosine consistency loss is bounded in [-1, 1].
    _expect(
        L_G >= -1.0 - 1e-6 and L_G <= 1.0 + 1e-6,
        "L_G ∈ [-1, 1]",
        passed, total,
    )

    # ── 2. Composite L is finite under paper-Eq.-3 weights. ──────────────
    var L_total = (
        Config.lambda_reward * L_R
        + Config.lambda_policy * L_P
        + Config.lambda_value * L_V
        + Config.lambda_consistency * L_G
    )
    print(
        "    L_total = ",
        Config.lambda_reward, "·L_R + ",
        Config.lambda_policy, "·L_P + ",
        Config.lambda_value, "·L_V + ",
        Config.lambda_consistency, "·L_G = ",
        L_total,
    )
    _expect(_is_finite(L_total), "composite L is finite", passed, total)

    # ── 3. Deterministic given fixed RNG state. ──────────────────────────
    seed(99)
    var l1 = agent.compute_loss_components()
    seed(99)
    var l2 = agent.compute_loss_components()

    def _delta(a: Float64, b: Float64) -> Float64:
        var d = a - b
        return d if d >= 0.0 else -d

    var max_diff = _delta(l1[0], l2[0])
    var d_p = _delta(l1[1], l2[1])
    if d_p > max_diff:
        max_diff = d_p
    var d_v = _delta(l1[2], l2[2])
    if d_v > max_diff:
        max_diff = d_v
    var d_g = _delta(l1[3], l2[3])
    if d_g > max_diff:
        max_diff = d_g
    print("    determinism: max |l1[i] - l2[i]| =", max_diff)
    _expect(
        max_diff < 1e-9,
        "compute_loss_components is deterministic given fixed RNG seed",
        passed, total,
    )

    print()
    print("=== Result:", passed, "/", total, "tests passed ===")
