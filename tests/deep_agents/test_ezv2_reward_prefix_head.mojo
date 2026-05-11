"""EZ-V2 reward-prefix head — building-block test.

Verifies the **forward shape + structure** of the reward-prefix head from
EfficientZero V1 (kept in EZ-V2, paper App. G):

    (h[k], c[k]) = LSTMCell.step_forward(z_dyn[k], h[k-1], c[k-1])
    reward_prefix_logits[k] = RewardPrefixHeadMLP(h[k])

Hidden state `h, c` resets to zero every `lstm_horizon_len = 5` unroll
steps to bound BPTT depth.

Coverage:
  1. Output shape is `(BATCH, BINS)` and finite at every step.
  2. The LSTM is actually doing work — `h[k]` differs meaningfully from
     `h[k-1]`. (If they were equal, the cell weights would be zero or
     forgetting trivially.)
  3. The MLP output also differs across steps (the LSTM's evolving
     hidden state translates into evolving reward-prefix logits).
  4. Resetting `h_prev`, `c_prev` to zero produces a different `h[k]`
     than the post-rollout state — the reset is a clean wipe, not a
     phantom carryover.
  5. Backward through LSTMCell + MLP is wired and produces finite
     gradients. (LSTMCell's own gradcheck lives in `tests/nn/test_lstm.mojo`
     — here we just confirm the chain doesn't blow up.)

Wiring this head into the K-step training loop (replacing the per-step
reward target with cumulative rewards + LSTM BPTT + horizon resets) is
deferred — needs changes to the value-loss assembly and the state's
hidden-state buffers. This file ships the building block.
"""

from std.math import abs
from std.random import seed
from layout import Layout, LayoutTensor
from mojo_rl.nn.constants import dtype
from mojo_rl.nn.model import LSTMCell, Model
from mojo_rl.nn.optimizer import Adam
from mojo_rl.nn.initializer import Xavier, Kaiming
from mojo_rl.nn.training import Network, NetworkState
from mojo_rl.deep_agents.efficient_zero_v2 import RewardPrefixHeadMLP


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
    print("=== EZ-V2 reward-prefix head (LSTM + MLP) test ===")
    var passed = 0
    var total = 0

    comptime LATENT = 32
    comptime LSTM_HIDDEN = 64
    comptime MLP_HIDDEN = 64
    comptime BINS = 21
    comptime BATCH = 4
    comptime LSTM_HORIZON = 5

    comptime LSTM = LSTMCell[LATENT, LSTM_HIDDEN]
    comptime MLP = RewardPrefixHeadMLP[LSTM_HIDDEN, MLP_HIDDEN, BINS]

    # Sanity on resolved comptime dims.
    print()
    print("--- Resolved dims ---")
    print("    LSTMCell    PARAM_SIZE =", LSTM.PARAM_SIZE)
    print("    LSTMCell    CACHE_SIZE =", LSTM.CACHE_SIZE)
    print("    HeadMLP     IN_DIM     =", MLP.IN_DIM)
    print("    HeadMLP     OUT_DIM    =", MLP.OUT_DIM)
    print("    HeadMLP     PARAM_SIZE =", MLP.PARAM_SIZE)
    _expect(
        MLP.IN_DIM == LSTM_HIDDEN and MLP.OUT_DIM == BINS,
        "MLP dims line up: IN=LSTM_HIDDEN, OUT=BINS",
        passed, total,
    )

    seed(2026)

    # ── Allocate LSTM params ─────────────────────────────────────────────
    var lstm_params_arr = InlineArray[
        Scalar[dtype], LSTM.PARAM_SIZE
    ](uninitialized=True)
    var lstm_params = LayoutTensor[
        dtype, Layout.row_major(LSTM.PARAM_SIZE), MutAnyOrigin
    ](lstm_params_arr.unsafe_ptr())
    LSTM.initialize_params[Xavier[]](lstm_params)

    # ── Allocate MLP NetworkState ────────────────────────────────────────
    var mlp = NetworkState[MLP, Adam[]]()
    mlp.initialize[Kaiming[]]()

    # ── Allocate per-step buffers ────────────────────────────────────────
    var z_arr = InlineArray[
        Scalar[dtype], BATCH * LATENT
    ](uninitialized=True)
    var h_prev_arr = InlineArray[
        Scalar[dtype], BATCH * LSTM_HIDDEN
    ](uninitialized=True)
    var c_prev_arr = InlineArray[
        Scalar[dtype], BATCH * LSTM_HIDDEN
    ](uninitialized=True)
    var h_arr = InlineArray[
        Scalar[dtype], BATCH * LSTM_HIDDEN
    ](uninitialized=True)
    var c_arr = InlineArray[
        Scalar[dtype], BATCH * LSTM_HIDDEN
    ](uninitialized=True)
    var logits_arr = InlineArray[
        Scalar[dtype], BATCH * BINS
    ](uninitialized=True)

    # Reset h_prev, c_prev = 0 (start of horizon).
    for i in range(BATCH * LSTM_HIDDEN):
        h_prev_arr[i] = Scalar[dtype](0.0)
        c_prev_arr[i] = Scalar[dtype](0.0)

    # Snapshots for the across-step comparison.
    var h_snapshots = InlineArray[
        Scalar[dtype], 6 * BATCH * LSTM_HIDDEN
    ](uninitialized=True)
    var logits_snapshots = InlineArray[
        Scalar[dtype], 6 * BATCH * BINS
    ](uninitialized=True)

    # ── Run 6 unroll steps with horizon reset between step 4 and step 5
    #     (i.e. step 5 starts from zeroed h, c). ─────────────────────────
    var any_nonfinite = False
    var max_logit_abs = Float64(0.0)
    print()
    print("--- 6-step rollout (horizon reset before step 5) ---")
    for k in range(6):
        # Fill z deterministically (different per step + per sample).
        for i in range(BATCH * LATENT):
            z_arr[i] = Scalar[dtype](
                0.13 * Float64((i + k * 7) % 11) - 0.4
            )

        # Reset h_prev, c_prev at horizon boundary (k = 5 here).
        if k == LSTM_HORIZON:
            for i in range(BATCH * LSTM_HIDDEN):
                h_prev_arr[i] = Scalar[dtype](0.0)
                c_prev_arr[i] = Scalar[dtype](0.0)

        # LSTM step (no cache — we're just testing forward shape).
        var z_t = LayoutTensor[
            dtype, Layout.row_major(BATCH, LATENT), MutAnyOrigin
        ](z_arr.unsafe_ptr())
        var h_prev_t = LayoutTensor[
            dtype, Layout.row_major(BATCH, LSTM_HIDDEN), MutAnyOrigin
        ](h_prev_arr.unsafe_ptr())
        var c_prev_t = LayoutTensor[
            dtype, Layout.row_major(BATCH, LSTM_HIDDEN), MutAnyOrigin
        ](c_prev_arr.unsafe_ptr())
        var h_t = LayoutTensor[
            dtype, Layout.row_major(BATCH, LSTM_HIDDEN), MutAnyOrigin
        ](h_arr.unsafe_ptr())
        var c_t = LayoutTensor[
            dtype, Layout.row_major(BATCH, LSTM_HIDDEN), MutAnyOrigin
        ](c_arr.unsafe_ptr())
        LSTM.step_forward_no_cache[BATCH](
            z_t, h_prev_t, c_prev_t, lstm_params, h_t, c_t
        )

        # MLP forward on h_t → reward_prefix_logits.
        var mlp_in_t = LayoutTensor[
            dtype, Layout.row_major(BATCH, MLP.IN_DIM), MutAnyOrigin
        ](h_arr.unsafe_ptr())
        var mlp_out_t = LayoutTensor[
            dtype, Layout.row_major(BATCH, MLP.OUT_DIM), MutAnyOrigin
        ](logits_arr.unsafe_ptr())
        var mlp_params_t = LayoutTensor[
            dtype, Layout.row_major(MLP.PARAM_SIZE), MutAnyOrigin
        ](mlp.params)
        var mlp_state_t = LayoutTensor[
            dtype, Layout.row_major(MLP.STATE_SIZE), MutAnyOrigin
        ](mlp.model_state)
        Network[MLP, Adam[]].forward[BATCH](
            mlp_in_t, mlp_out_t, mlp_params_t, mlp_state_t
        )

        # Validate output.
        for i in range(BATCH * BINS):
            var v = Float64(logits_arr[i])
            if not _is_finite(v):
                any_nonfinite = True
            var av = v if v >= 0.0 else -v
            if av > max_logit_abs:
                max_logit_abs = av

        # Snapshot h and logits.
        for i in range(BATCH * LSTM_HIDDEN):
            h_snapshots[k * BATCH * LSTM_HIDDEN + i] = h_arr[i]
        for i in range(BATCH * BINS):
            logits_snapshots[k * BATCH * BINS + i] = logits_arr[i]

        # Roll h, c forward for next step.
        for i in range(BATCH * LSTM_HIDDEN):
            h_prev_arr[i] = h_arr[i]
            c_prev_arr[i] = c_arr[i]

        print(
            "    step", k,
            " max|logit| =", max_logit_abs,
            "  finite =", not any_nonfinite,
        )

    # ── 1. Output shape + finite-ness ────────────────────────────────────
    _expect(
        not any_nonfinite,
        "all reward-prefix logits finite across 6 unroll steps",
        passed, total,
    )
    _expect(
        max_logit_abs < 1e3,
        "logit magnitudes bounded (< 1e3)",
        passed, total,
    )

    # ── 2. h[k] ≠ h[k−1] within horizon ──────────────────────────────────
    # Compare snapshots across consecutive within-horizon steps.
    var max_h_step_diff = Float64(0.0)
    for k in range(1, LSTM_HORIZON):
        for i in range(BATCH * LSTM_HIDDEN):
            var prev_v = Float64(
                h_snapshots[(k - 1) * BATCH * LSTM_HIDDEN + i]
            )
            var v = Float64(h_snapshots[k * BATCH * LSTM_HIDDEN + i])
            var d = v - prev_v
            if d < 0:
                d = -d
            if d > max_h_step_diff:
                max_h_step_diff = d
    print()
    print("    max |h[k] − h[k−1]| within horizon =", max_h_step_diff)
    _expect(
        max_h_step_diff > 1e-3,
        "LSTM is doing work — h[k] meaningfully differs from h[k−1]",
        passed, total,
    )

    # ── 3. logits[k] ≠ logits[k−1] within horizon ────────────────────────
    var max_logit_step_diff = Float64(0.0)
    for k in range(1, LSTM_HORIZON):
        for i in range(BATCH * BINS):
            var prev_v = Float64(
                logits_snapshots[(k - 1) * BATCH * BINS + i]
            )
            var v = Float64(logits_snapshots[k * BATCH * BINS + i])
            var d = v - prev_v
            if d < 0:
                d = -d
            if d > max_logit_step_diff:
                max_logit_step_diff = d
    print(
        "    max |logits[k] − logits[k−1]| within horizon =",
        max_logit_step_diff,
    )
    _expect(
        max_logit_step_diff > 1e-3,
        "Reward-prefix logits evolve over the horizon",
        passed, total,
    )

    # ── 4. After-reset h[5] differs from h[4] (the reset is a real wipe). ─
    var max_reset_diff = Float64(0.0)
    for i in range(BATCH * LSTM_HIDDEN):
        var v_before = Float64(
            h_snapshots[4 * BATCH * LSTM_HIDDEN + i]
        )
        var v_after = Float64(h_snapshots[5 * BATCH * LSTM_HIDDEN + i])
        var d = v_before - v_after
        if d < 0:
            d = -d
        if d > max_reset_diff:
            max_reset_diff = d
    print("    |h[4] − h[5]| (across reset)              =", max_reset_diff)
    _expect(
        max_reset_diff > 1e-3,
        "Horizon reset visibly changes the LSTM hidden state at step 5",
        passed, total,
    )

    # ── 5. Backward through LSTMCell + MLP ───────────────────────────────
    # Rerun the last step with cache enabled, then call backward through
    # both the MLP and LSTMCell. We're only checking that the chain
    # produces finite gradients — the LSTMCell's own analytical gradcheck
    # lives in `tests/nn/test_lstm.mojo`.
    print()
    print("--- Backward chain (LSTM cache + MLP backward) ---")
    var lstm_cache_arr = InlineArray[
        Scalar[dtype], BATCH * LSTM.CACHE_SIZE
    ](uninitialized=True)
    var mlp_cache_arr = InlineArray[
        Scalar[dtype], BATCH * MLP.CACHE_SIZE
    ](uninitialized=True)
    for i in range(BATCH * LSTM_HIDDEN):
        h_prev_arr[i] = Scalar[dtype](0.0)
        c_prev_arr[i] = Scalar[dtype](0.0)

    var z_t2 = LayoutTensor[
        dtype, Layout.row_major(BATCH, LATENT), MutAnyOrigin
    ](z_arr.unsafe_ptr())
    var h_prev_t2 = LayoutTensor[
        dtype, Layout.row_major(BATCH, LSTM_HIDDEN), MutAnyOrigin
    ](h_prev_arr.unsafe_ptr())
    var c_prev_t2 = LayoutTensor[
        dtype, Layout.row_major(BATCH, LSTM_HIDDEN), MutAnyOrigin
    ](c_prev_arr.unsafe_ptr())
    var h_t2 = LayoutTensor[
        dtype, Layout.row_major(BATCH, LSTM_HIDDEN), MutAnyOrigin
    ](h_arr.unsafe_ptr())
    var c_t2 = LayoutTensor[
        dtype, Layout.row_major(BATCH, LSTM_HIDDEN), MutAnyOrigin
    ](c_arr.unsafe_ptr())
    var lstm_cache_t = LayoutTensor[
        dtype, Layout.row_major(BATCH, LSTM.CACHE_SIZE), MutAnyOrigin
    ](lstm_cache_arr.unsafe_ptr())
    LSTM.step_forward[BATCH](
        z_t2, h_prev_t2, c_prev_t2, lstm_params, h_t2, c_t2, lstm_cache_t
    )

    var mlp_in_t2 = LayoutTensor[
        dtype, Layout.row_major(BATCH, MLP.IN_DIM), MutAnyOrigin
    ](h_arr.unsafe_ptr())
    var mlp_out_t2 = LayoutTensor[
        dtype, Layout.row_major(BATCH, MLP.OUT_DIM), MutAnyOrigin
    ](logits_arr.unsafe_ptr())
    var mlp_params_t2 = LayoutTensor[
        dtype, Layout.row_major(MLP.PARAM_SIZE), MutAnyOrigin
    ](mlp.params)
    var mlp_state_t2 = LayoutTensor[
        dtype, Layout.row_major(MLP.STATE_SIZE), MutAnyOrigin
    ](mlp.model_state)
    var mlp_cache_t = LayoutTensor[
        dtype, Layout.row_major(BATCH, MLP.CACHE_SIZE), MutAnyOrigin
    ](mlp_cache_arr.unsafe_ptr())
    Network[MLP, Adam[]].forward_with_cache[BATCH](
        mlp_in_t2, mlp_out_t2, mlp_params_t2, mlp_state_t2, mlp_cache_t
    )

    # Upstream gradient on the logits (e.g., from cross-entropy) — use ones.
    var grad_logits_arr = InlineArray[
        Scalar[dtype], BATCH * BINS
    ](uninitialized=True)
    for i in range(BATCH * BINS):
        grad_logits_arr[i] = Scalar[dtype](1.0)
    var grad_logits_t = LayoutTensor[
        dtype, Layout.row_major(BATCH, BINS), MutAnyOrigin
    ](grad_logits_arr.unsafe_ptr())

    # MLP backward → grad on h_t.
    var grad_h_arr = InlineArray[
        Scalar[dtype], BATCH * LSTM_HIDDEN
    ](uninitialized=True)
    for i in range(BATCH * LSTM_HIDDEN):
        grad_h_arr[i] = Scalar[dtype](0.0)
    var grad_h_t = LayoutTensor[
        dtype, Layout.row_major(BATCH, LSTM_HIDDEN), MutAnyOrigin
    ](grad_h_arr.unsafe_ptr())
    var mlp_grads_t = LayoutTensor[
        dtype, Layout.row_major(MLP.PARAM_SIZE), MutAnyOrigin
    ](mlp.grads)
    Network[MLP, Adam[]].backward[BATCH](
        grad_logits_t,
        grad_h_t,
        mlp_params_t2,
        mlp_state_t2,
        mlp_cache_t,
        mlp_grads_t,
    )

    # LSTMCell backward → grads on x, h_prev, c_prev, lstm params.
    var grad_x_arr = InlineArray[
        Scalar[dtype], BATCH * LATENT
    ](uninitialized=True)
    var grad_h_prev_arr = InlineArray[
        Scalar[dtype], BATCH * LSTM_HIDDEN
    ](uninitialized=True)
    var grad_c_prev_arr = InlineArray[
        Scalar[dtype], BATCH * LSTM_HIDDEN
    ](uninitialized=True)
    var grad_c_t_arr = InlineArray[
        Scalar[dtype], BATCH * LSTM_HIDDEN
    ](uninitialized=True)
    for i in range(BATCH * LATENT):
        grad_x_arr[i] = Scalar[dtype](0.0)
    for i in range(BATCH * LSTM_HIDDEN):
        grad_h_prev_arr[i] = Scalar[dtype](0.0)
        grad_c_prev_arr[i] = Scalar[dtype](0.0)
        grad_c_t_arr[i] = Scalar[dtype](0.0)
    var lstm_grads_arr = InlineArray[
        Scalar[dtype], LSTM.PARAM_SIZE
    ](uninitialized=True)
    for i in range(LSTM.PARAM_SIZE):
        lstm_grads_arr[i] = Scalar[dtype](0.0)
    var grad_x_t = LayoutTensor[
        dtype, Layout.row_major(BATCH, LATENT), MutAnyOrigin
    ](grad_x_arr.unsafe_ptr())
    var grad_h_prev_t = LayoutTensor[
        dtype, Layout.row_major(BATCH, LSTM_HIDDEN), MutAnyOrigin
    ](grad_h_prev_arr.unsafe_ptr())
    var grad_c_prev_t = LayoutTensor[
        dtype, Layout.row_major(BATCH, LSTM_HIDDEN), MutAnyOrigin
    ](grad_c_prev_arr.unsafe_ptr())
    var grad_c_t_t = LayoutTensor[
        dtype, Layout.row_major(BATCH, LSTM_HIDDEN), MutAnyOrigin
    ](grad_c_t_arr.unsafe_ptr())
    var lstm_grads_t = LayoutTensor[
        dtype, Layout.row_major(LSTM.PARAM_SIZE), MutAnyOrigin
    ](lstm_grads_arr.unsafe_ptr())
    LSTM.step_backward[BATCH](
        grad_h_t,            # dh — from MLP backward
        grad_c_t_t,          # dc — zero (no later time step)
        z_t2,                # x at step k
        h_prev_t2,           # h_{k-1}
        c_prev_t2,           # c_{k-1}
        lstm_params,
        lstm_cache_t,
        grad_x_t,
        grad_h_prev_t,
        grad_c_prev_t,
        lstm_grads_t,
    )

    # Sanity: all grads finite.
    var any_grad_nonfinite = False
    var max_grad_x = Float64(0.0)
    var max_grad_lstm_param = Float64(0.0)
    var max_grad_mlp_param = Float64(0.0)
    for i in range(BATCH * LATENT):
        var v = Float64(grad_x_arr[i])
        if not _is_finite(v):
            any_grad_nonfinite = True
        var av = v if v >= 0.0 else -v
        if av > max_grad_x:
            max_grad_x = av
    for i in range(LSTM.PARAM_SIZE):
        var v = Float64(lstm_grads_arr[i])
        if not _is_finite(v):
            any_grad_nonfinite = True
        var av = v if v >= 0.0 else -v
        if av > max_grad_lstm_param:
            max_grad_lstm_param = av
    for i in range(MLP.PARAM_SIZE):
        var v = Float64(mlp.grads[i])
        if not _is_finite(v):
            any_grad_nonfinite = True
        var av = v if v >= 0.0 else -v
        if av > max_grad_mlp_param:
            max_grad_mlp_param = av

    print(
        "    max |grad_x|             =", max_grad_x,
    )
    print(
        "    max |grad_lstm_params|   =", max_grad_lstm_param,
    )
    print(
        "    max |grad_mlp_params|    =", max_grad_mlp_param,
    )
    _expect(
        not any_grad_nonfinite,
        "all grads finite across the LSTM + MLP backward chain",
        passed, total,
    )
    _expect(
        max_grad_lstm_param > 1e-6 and max_grad_mlp_param > 1e-6,
        "non-trivial gradients flow into both LSTM and MLP params",
        passed, total,
    )

    print()
    print("=== Result:", passed, "/", total, "tests passed ===")
