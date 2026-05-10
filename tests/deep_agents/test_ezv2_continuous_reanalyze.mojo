"""Smoke test for `GenericEZV2ContinuousAgent.reanalyze` (added 2026-05-10).

Mirrors `test_ezv2_reanalyze.mojo` (discrete) but drives the continuous
agent: pre-fills replay state with canned transitions, snapshots the
stored mcts_policies/mcts_values/step_at_write at a few indices,
calls `reanalyze(num_samples=N)`, and verifies:

    (a) returns N (number refreshed) when buffer is ready,
    (b) at least some `step_at_write` entries get bumped to current
        train_step_count,
    (c) refreshed mcts_values are finite,
    (d) refreshed mcts_policies are within [-MAX_ACTION, +MAX_ACTION] per
        dim (the search returns squashed actions),
    (e) reanalyze on an empty buffer returns 0 (early-out path).
"""

from std.random import seed, random_float64
from std.gpu.host import DeviceContext
from mojo_rl.deep_agents.efficient_zero_v2 import (
    EZV2ContinuousMLPConfig,
    EZV2DiscreteConfig,
    EZV2DiscreteCPUState,
    GenericEZV2ContinuousAgent,
)
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


def _fill_canned_transitions[
    Config: EZV2DiscreteConfig,
](
    mut state: EZV2DiscreteCPUState[Config],
    num_transitions: Int,
    max_action: Float64,
    episode_len: Int = 50,
) raises:
    for i in range(num_transitions):
        var obs_arr = InlineArray[Scalar[dtype], Config.obs_dim](
            uninitialized=True
        )
        for d in range(Config.obs_dim):
            obs_arr[d] = Scalar[dtype](random_float64(-1.0, 1.0))
        var act_arr = InlineArray[Scalar[dtype], Config.action_dim](
            uninitialized=True
        )
        for d in range(Config.action_dim):
            act_arr[d] = Scalar[dtype](
                random_float64(-0.9, 0.9) * max_action
            )
        var reward = Scalar[dtype](random_float64(-1.0, 1.0))
        var done = (i + 1) % episode_len == 0
        state.buffer.add(obs_arr, act_arr, reward, done)

        var slot = (state.buffer.ptr - 1 + 50000) % 50000
        for d in range(Config.action_dim):
            state.mcts_policies[slot * Config.action_dim + d] = act_arr[d]
        # Use a sentinel value so we can detect post-reanalyze
        # changes via comparison.
        state.mcts_values[slot] = Scalar[dtype](-999.0)
        state.step_at_write[slot] = Scalar[DType.uint32](0)
        state.priorities[slot] = Scalar[dtype](1.0)


def main() raises:
    print("=== EZ-V2 continuous reanalyze smoke ===")
    var passed = 0
    var total = 0

    comptime Config = EZV2ContinuousMLPConfig[
        OBS=4,
        ACT_DIM=2,
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
        K_ROOT=4,
        K_NON_ROOT=2,
        MAX_ACTION=1.0,
    ]

    seed(2026)
    var agent = GenericEZV2ContinuousAgent[Config](
        gamma=0.99, v_min=-10.0, v_max=10.0, temperature=1.0,
    )

    # ── Empty buffer → reanalyze returns 0 ────────────────────────────────
    print()
    print("--- reanalyze on empty buffer ---")
    var n_empty = agent.reanalyze(num_samples=4)
    _expect(
        n_empty == 0,
        "reanalyze on empty buffer returns 0 (early-out)",
        passed, total,
    )

    # ── Fill replay + reanalyze ───────────────────────────────────────────
    print()
    print("--- Filling replay state (200 canned transitions) ---")
    _fill_canned_transitions[Config](
        agent.state,
        num_transitions=200,
        max_action=1.0,
        episode_len=50,
    )
    print("    buffer.size =", agent.state.buffer.size)
    _expect(
        agent.state.is_ready(),
        "replay state ready after fill",
        passed, total,
    )

    # Bump train_step_count to a value distinct from the canned 0 so we
    # can detect step_at_write updates by reanalyze.
    agent.train_step_count = 7
    var fresh_step = UInt32(agent.train_step_count)

    # Snapshot all mcts_values (sentinel = -999.0) so we can count how
    # many indices got rewritten by reanalyze.
    var initial_mcts_values = List[Float64]()
    for i in range(50000):
        initial_mcts_values.append(Float64(agent.state.mcts_values[i]))

    print()
    print("--- reanalyze(num_samples=16) ---")
    var num_samples = 16
    var n_refreshed = agent.reanalyze(num_samples=num_samples)
    print("    returned n_refreshed =", n_refreshed)
    _expect(
        n_refreshed == num_samples,
        "reanalyze returns num_samples on ready buffer",
        passed, total,
    )

    # ── Verify side-effects ──────────────────────────────────────────────
    var n_step_bumped = 0
    var n_value_changed = 0
    var any_value_nonfinite = False
    var any_policy_oor = False
    for i in range(50000):
        if Float64(agent.state.step_at_write[i]) >= Float64(fresh_step):
            n_step_bumped += 1
        var v = Float64(agent.state.mcts_values[i])
        if v != initial_mcts_values[i]:
            n_value_changed += 1
            if not _is_finite(v):
                any_value_nonfinite = True
        # Policy values for refreshed indices must stay inside the
        # squashed-action range (search returns MAX_ACTION · tanh(u)).
        if v != initial_mcts_values[i]:
            for d in range(Config.action_dim):
                var p = Float64(
                    agent.state.mcts_policies[
                        i * Config.action_dim + d
                    ]
                )
                # Allow exact ±1.0 boundary (fp32 tanh saturation).
                if p > 1.0 + 1e-5 or p < -1.0 - 1e-5:
                    any_policy_oor = True

    print("    indices with step_at_write bumped =", n_step_bumped)
    print("    indices with mcts_values changed   =", n_value_changed)

    _expect(
        n_step_bumped >= 1,
        "at least one step_at_write got bumped to fresh_step",
        passed, total,
    )
    _expect(
        n_value_changed >= 1,
        "at least one mcts_values entry changed (sentinel -999 → SVE)",
        passed, total,
    )
    _expect(
        not any_value_nonfinite,
        "all refreshed mcts_values are finite",
        passed, total,
    )
    _expect(
        not any_policy_oor,
        "refreshed mcts_policies stay in [-MAX, +MAX] per dim",
        passed, total,
    )

    print()
    print("=== Result:", passed, "/", total, "tests passed ===")
