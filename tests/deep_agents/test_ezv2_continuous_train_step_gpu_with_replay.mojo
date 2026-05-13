"""Smoke test for `GenericEZV2ContinuousAgent.train_step_gpu_with_replay`.

Mirrors `test_ezv2_train_step_gpu_with_replay.mojo` (the discrete version)
but drives the continuous agent: pre-populates the replay state with
canned continuous transitions, builds a `EZV2GPUReplayBuffer`, syncs from
CPU, and runs 100 GPU-sampling train steps.

Verifies (over 100 calls):
    (a) no NaN/Inf loss across the run,
    (b) at least one train_step_gpu_with_replay call fired,
    (c) all four loss components finite at end,
    (d) composite L decreases ≥ 5%.
"""

from std.random import seed, random_float64
from std.gpu.host import DeviceContext
from mojo_rl.deep_agents.efficient_zero_v2 import (
    EZV2ContinuousMLPConfig,
    EZV2DiscreteConfig,
    EZV2DiscreteCPUState,
    EZV2GPUStateBase,
    GenericEZV2ContinuousAgent,
)
from mojo_rl.deep_agents.efficient_zero_v2.gpu_replay import (
    EZV2GPUReplayBuffer,
)
from mojo_rl.nn.constants import dtype


# Helper parameterized over Config so Mojo nightly's comptime-alias
# unification reuses the agent struct's binding context. Same trick as
# `test_ezv2_continuous_train_step_gpu.mojo`.
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
        state.mcts_values[slot] = Scalar[dtype](
            random_float64(-0.5, 0.5)
        )
        state.step_at_write[slot] = Scalar[DType.uint32](0)
        state.priorities[slot] = Scalar[dtype](1.0)
        # Phase 1 (sum-tree PER): keep tree in sync with raw priorities.
        state.on_flush_write(slot)


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


def main() raises:
    print(
        "=== EZ-V2 continuous train_step_gpu_with_replay smoke ==="
    )
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
    var ctx = DeviceContext()

    # ── Pre-populate CPU replay state ──────────────────────────────────
    print()
    print("--- Filling CPU replay state (200 canned transitions) ---")
    _fill_canned_transitions[Config](
        agent.state,
        num_transitions=200,
        max_action=1.0,
        episode_len=50,
    )
    print("    buffer.size =", agent.state.buffer.size)
    _expect(
        agent.state.is_ready(),
        "CPU replay buffer ready",
        passed, total,
    )

    # ── Build GPU state + GPU replay buffer + sync ─────────────────────
    print()
    print("--- Building GPU state + GPU replay; uploading ---")
    var gpu = EZV2GPUStateBase[Config](ctx)
    gpu.upload_from(agent.state, ctx)
    var gpu_replay = EZV2GPUReplayBuffer[
        50000, Config.obs_dim, Config.action_dim
    ](ctx)
    gpu_replay.upload_from_cpu(agent.state, ctx)
    gpu_replay.max_priority = agent.max_priority
    ctx.synchronize()
    print("    GPU replay size =", gpu_replay.size)

    # ── Run train_step_gpu_with_replay ─────────────────────────────────
    print()
    print("--- Running 100 train_step_gpu_with_replay calls ---")
    var num_train_steps = 100
    var L_total_first = Float64(0.0)
    var L_total_last = Float64(0.0)
    var L_R_last = Float64(0.0)
    var L_P_last = Float64(0.0)
    var L_V_last = Float64(0.0)
    var L_G_last = Float64(0.0)
    var any_nan = False
    var any_call_fired = False
    for step in range(num_train_steps):
        var t = agent.train_step_gpu_with_replay(
            gpu, gpu_replay, ctx, UInt32(7 + step)
        )
        var L_total = t[0]
        var L_R = t[1]
        var L_P = t[2]
        var L_V = t[3]
        var L_G = t[4]
        if not _is_finite(L_total):
            any_nan = True
        if (
            L_total != Float64(0.0)
            or L_R != Float64(0.0)
            or L_P != Float64(0.0)
            or L_V != Float64(0.0)
        ):
            any_call_fired = True
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
                "    step", step,
                ": L_total=", L_total,
                " L_R=", L_R, " L_P=", L_P,
                " L_V=", L_V, " L_G=", L_G,
            )

    print()
    print("--- After GPU-sampling continuous training ---")
    print("    L_total: first =", L_total_first, ", last =", L_total_last)
    print("    L_R last =", L_R_last)
    print("    L_P last =", L_P_last)
    print("    L_V last =", L_V_last)
    print("    L_G last =", L_G_last)

    _expect(
        not any_nan,
        "no NaN/Inf encountered across 100 calls",
        passed, total,
    )
    _expect(
        any_call_fired,
        "at least one train_step_gpu_with_replay call returned non-zero losses",
        passed, total,
    )
    _expect(
        _is_finite(L_R_last)
        and _is_finite(L_P_last)
        and _is_finite(L_V_last)
        and _is_finite(L_G_last),
        "all four loss components finite at end",
        passed, total,
    )
    _expect(
        L_total_last < 0.95 * L_total_first,
        "composite L decreased ≥ 5% (training does something)",
        passed, total,
    )

    print()
    print("=== Result:", passed, "/", total, "tests passed ===")
