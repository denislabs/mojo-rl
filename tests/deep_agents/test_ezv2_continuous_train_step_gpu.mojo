"""Phase 3.3 smoke: end-to-end continuous training step on GPU.

Validates that the entire training pipeline plumbs through cleanly when
`Config.ActSpace = ContinuousActionSpace`. We don't have a sampled-MCTS
agent yet (Phase 3.5), so we bypass the acting side: pre-populate the
replay state directly with canned continuous transitions, then drive the
existing `agent.train_step_gpu` with the continuous config.

This works because:
  • `EZV2DiscreteCPUState[Config]` and `EZV2GPUStateBase[Config]` are
    action-space-agnostic — they only manipulate raw `[ACT_DIM]`-wide
    float buffers; the discrete-vs-continuous semantics live entirely
    in `Config.ActSpace`'s policy-loss kernel hook.
  • `GenericEfficientZeroV2Agent`'s training methods touch the same raw
    buffers — only the acting-side methods (`select_action`,
    `store_transition`) hard-code discrete action types.
  • Default `value_target_mode = VALUE_TARGET_SEARCH` skips the host-side
    target-net value decode, which is the only path that bakes in the
    discrete pred-output layout (offset = `ACT`, would be wrong for
    continuous's `2*ACT_DIM`). Continuous SARSA/MIXED is a follow-up.

Verifies (over 100 train_step_gpu calls):
    (a) no NaN/Inf loss across the run,
    (b) all four loss components (L_R, L_P, L_V, L_G) are finite,
    (c) composite L decreases ≥ 5% (smoke: shows training does *something*).

ACT_DIM = 2 (Pendulum-shaped). Synthetic data: random obs, random
actions in (-MAX, MAX), random rewards. The "policy target" stored in
mcts_policies is the raw chosen action vector — exactly what the
continuous policy-loss kernel consumes.
"""

from std.random import seed, random_float64
from std.gpu.host import DeviceContext
from mojo_rl.deep_agents.efficient_zero_v2 import (
    EZV2ContinuousMLPConfig,
    EZV2DiscreteConfig,
    EZV2DiscreteCPUState,
    EZV2GPUStateBase,
    GenericEfficientZeroV2Agent,
)
from mojo_rl.nn.constants import dtype


# Wrapper parameterized over `Config` — Mojo nightly's comptime-alias
# unification reuses the agent struct's binding context here, which lets
# `buffer.add()` accept InlineArrays sized by `Config.obs_dim` /
# `Config.action_dim`. The outer `main()` can't construct those
# InlineArrays directly because its `Config.obs_dim` is a fresh comptime
# binding distinct from the buffer's parameter even when both evaluate
# to the same Int.
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
        # Policy target = chosen action vector (Eq. 8 simple-best-action).
        for d in range(Config.action_dim):
            state.mcts_policies[slot * Config.action_dim + d] = act_arr[d]
        state.mcts_values[slot] = Scalar[dtype](
            random_float64(-0.5, 0.5)
        )
        state.step_at_write[slot] = Scalar[DType.uint32](0)
        state.priorities[slot] = Scalar[dtype](1.0)


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
    print("=== EZ-V2 continuous train_step_gpu smoke (Phase 3.3) ===")
    var passed = 0
    var total = 0

    comptime OBS = 4
    comptime ACT_DIM = 2
    comptime LATENT = 32
    comptime HIDDEN = 32
    comptime BINS = 21
    comptime BS = 8
    comptime K = 3
    comptime N_TD = 5
    comptime SIMS = 8
    comptime NODES = 32
    comptime K_ROOT = 4
    comptime K_NON_ROOT = 2
    comptime MAX_ACTION = 1.0

    comptime Config = EZV2ContinuousMLPConfig[
        OBS=OBS,
        ACT_DIM=ACT_DIM,
        LATENT=LATENT,
        HIDDEN=HIDDEN,
        PROJ=64,
        PRED_BOTTLENECK=32,
        BINS=BINS,
        BS=BS,
        K_UNROLL=K,
        N_TD=N_TD,
        SIMS=SIMS,
        NODES=NODES,
        K_ROOT=K_ROOT,
        K_NON_ROOT=K_NON_ROOT,
        MAX_ACTION=MAX_ACTION,
    ]

    seed(2026)
    var agent = GenericEfficientZeroV2Agent[Config](
        gamma=0.99, v_min=-10.0, v_max=10.0, temperature=1.0,
    )
    var ctx = DeviceContext()

    # ── Fill replay state with canned continuous transitions ────────────
    # The agent's `select_action` / `store_transition` hard-code discrete
    # action types, so we bypass them and write directly into the state's
    # raw buffers. Layout matches what `train_step_gpu`'s host sampler
    # reads:
    #   state.buffer.obs[idx * OBS + d]
    #   state.buffer.actions[idx * ACT_DIM + d]
    #   state.buffer.rewards[idx]
    #   state.buffer.dones[idx]
    #   state.priorities[idx]
    #   state.mcts_policies[idx * ACT_DIM + d]   — chosen-action target
    #   state.mcts_values[idx]                   — root value (SVE)
    #   state.step_at_write[idx]                 — train-step age
    print()
    print("--- Filling replay state (200 canned transitions) ---")
    _fill_canned_transitions[Config](
        agent.state,
        num_transitions=200,
        max_action=MAX_ACTION,
        episode_len=50,
    )
    print("    buffer.size =", agent.state.buffer.size)
    print("    buffer.ptr  =", agent.state.buffer.ptr)
    _expect(
        agent.state.is_ready(),
        "agent.state ready to train",
        passed, total,
    )

    # ── Build GPU state + upload network params ──────────────────────────
    print()
    print("--- Building GPU state + uploading params ---")
    var gpu = EZV2GPUStateBase[Config](ctx)
    gpu.upload_from(agent.state, ctx)
    ctx.synchronize()

    # ── Run 100 train_step_gpu calls ────────────────────────────────────
    print()
    print("--- Running 100 train_step_gpu calls (continuous config) ---")
    var num_train_steps = 100
    var L_total_first = Float64(0.0)
    var L_total_last = Float64(0.0)
    var L_R_last = Float64(0.0)
    var L_P_last = Float64(0.0)
    var L_V_last = Float64(0.0)
    var L_G_last = Float64(0.0)
    var any_nan = False
    for step in range(num_train_steps):
        var t = agent.train_step_gpu(gpu, ctx)
        var L_total = t[0]
        var L_R = t[1]
        var L_P = t[2]
        var L_V = t[3]
        var L_G = t[4]
        if not _is_finite(L_total):
            any_nan = True
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
                " L_R=", L_R,
                " L_P=", L_P,
                " L_V=", L_V,
                " L_G=", L_G,
            )

    print()
    print("--- After continuous training ---")
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
        _is_finite(L_R_last)
        and _is_finite(L_P_last)
        and _is_finite(L_V_last)
        and _is_finite(L_G_last),
        "all four loss components finite at end",
        passed, total,
    )
    _expect(
        _is_finite(L_total_first) and _is_finite(L_total_last),
        "first + last L_total finite",
        passed, total,
    )
    _expect(
        L_total_last < 0.95 * L_total_first,
        "composite L decreased ≥ 5% (training does something)",
        passed, total,
    )

    print()
    print("=== Result:", passed, "/", total, "tests passed ===")
