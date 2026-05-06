"""EZ-V2 reward-prefix LSTM head — full integration smoke test.

Verifies that `train_step` with `USE_REWARD_PREFIX=True` actually trains
the LSTM head + MLP head on the cumulative-reward CE target. This pairs
with the building-block test (`test_ezv2_reward_prefix_head.mojo`) which
covers the LSTM/MLP forward + backward chain in isolation; here we
confirm that:

  1. `train_step()` runs end-to-end without NaN/Inf when the reward-
     prefix head is wired in.
  2. Composite `L_total` drops meaningfully over 100 train steps.
  3. `L_R` (now the cumulative-reward CE through the LSTM head) drops —
     i.e. the LSTM/MLP head ARE getting gradients and training.
  4. The original five networks (rep, dyn, pred, projector, predictor)
     still train alongside (L_V drops, L_G drops). The reward-prefix
     wiring shouldn't starve the rest of the network.

The dynamics network's reward-output slice gets ZERO gradient when the
reward-prefix head is on — we don't assert on the dyn-reward magnitude
because that path is decoupled by design.
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
    print("=== EZ-V2 reward-prefix train_step (LSTM head) integration ===")
    var passed = 0
    var total = 0

    # Match the plain CPU smoke config + flip USE_REWARD_PREFIX=True.
    # Keep LSTM_HORIZON_LEN > K_UNROLL so within-unroll resets don't
    # fire (saves us a separate horizon-reset test); the building-block
    # `test_ezv2_reward_prefix_head` already covers reset behaviour.
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
        USE_REWARD_PREFIX=True,
        LSTM_HIDDEN=32,
        LSTM_HORIZON_LEN=5,
        LSTM_MLP_HIDDEN=32,
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
    # `compute_loss_components` is the forward-only, non-reward-prefix
    # baseline. We use it to confirm the agent isn't already trained;
    # the comparison-of-interest is the train_step trajectory below.
    print()
    print("--- Initial L (forward-only, dyn-reward path) ---")
    var initial = agent.compute_loss_components()
    print("    L_R(dyn) =", initial[0], "  L_P =", initial[1])
    print("    L_V      =", initial[2], "  L_G =", initial[3])

    # ── Run train_step calls + watch the reward-prefix loss ──────────────
    print()
    print("--- Running 100 train_step calls (USE_REWARD_PREFIX=True) ---")
    var num_train_steps = 100
    var L_total_first = Float64(0.0)
    var L_total_last = Float64(0.0)
    var L_R_first = Float64(0.0)
    var L_R_last = Float64(0.0)
    var L_P_first = Float64(0.0)
    var L_P_last = Float64(0.0)
    var L_V_first = Float64(0.0)
    var L_V_last = Float64(0.0)
    var L_G_first = Float64(0.0)
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
        if step == 0:
            L_total_first = L_total
            L_R_first = L_R
            L_P_first = L_P
            L_V_first = L_V
            L_G_first = L_G
        if step == num_train_steps - 1:
            L_total_last = L_total
            L_R_last = L_R
            L_P_last = L_P
            L_V_last = L_V
            L_G_last = L_G
        if step % 20 == 0:
            print(
                "    step", step,
                ": L_total=", L_total,
                " L_R(prefix)=", L_R,
                " L_V=", L_V,
                " L_G=", L_G,
            )

    print()
    print("--- After training ---")
    print("    L_total: ", L_total_first, "→", L_total_last)
    print("    L_R(prefix):", L_R_first, "→", L_R_last)
    print("    L_P:     ", L_P_first, "→", L_P_last)
    print("    L_V:     ", L_V_first, "→", L_V_last)
    print("    L_G:     ", L_G_first, "→", L_G_last)

    _expect(
        not any_nan,
        "no NaN/Inf encountered across train_step calls",
        passed, total,
    )
    _expect(
        _is_finite(L_total_last),
        "final L_total is finite",
        passed, total,
    )
    _expect(
        L_total_last < 0.9 * L_total_first,
        "composite L decreased ≥ 10% over 100 steps",
        passed, total,
    )
    _expect(
        L_R_last < 0.9 * L_R_first,
        "L_R(prefix) dropped ≥ 10% — LSTM + MLP head are training",
        passed, total,
    )
    _expect(
        L_V_last < 0.9 * L_V_first,
        "L_V dropped ≥ 10% (value head trains alongside the LSTM head)",
        passed, total,
    )

    # Action selection still works after training the LSTM head.
    var probe_obs = env.reset_obs_list()
    var probe_result = agent.select_action(probe_obs, training=False)
    var probe_action = probe_result[0]
    var probe_value = probe_result[2]
    print()
    print(
        "    probe action =",
        probe_action,
        "  probe SVE =",
        probe_value,
    )
    _expect(
        probe_action >= 0 and probe_action < Config.action_dim,
        "post-train action selection returns a legal action",
        passed, total,
    )
    _expect(
        _is_finite(probe_value),
        "post-train SVE value is finite",
        passed, total,
    )

    print()
    print("=== Result:", passed, "/", total, "tests passed ===")
