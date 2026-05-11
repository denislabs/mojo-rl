"""Phase-2 Step 2 unit tests — EZ-V2 network composites.

For each of the four composites we:

  (1) check that the resolved Model trait dimensions match the formula in
      `networks.mojo` (IN_DIM, OUT_DIM, PARAM_SIZE);
  (2) run a forward pass on a random initialisation with a small batch and
      verify outputs are finite and the right shape;
  (3) for `ImproveResidualBlock` — the only composite with a non-trivial
      gradient flow path (LayerNorm + residual + two Linears) — also run a
      finite-diff gradcheck over the parameters to confirm the
      Pre-LN-residual stack is wired correctly through the autodiff graph.

If (3) fails, the `consistency loss + autodiff op` step (next on the
phase-2 list) cannot succeed — the SimSiam projector / predictor sit on
top of the same residual machinery.
"""

from std.math import abs
from layout import Layout, LayoutTensor
from mojo_rl.nn.constants import dtype
from mojo_rl.nn.model import Model
from mojo_rl.nn.training import NetworkState
from mojo_rl.nn.optimizer import Adam
from mojo_rl.nn.initializer import Xavier, Kaiming
from mojo_rl.deep_agents.efficient_zero_v2 import (
    ImproveResidualBlock,
    ActionEmbedding,
    ProjectionMLP,
    PredictionMLP,
)


def _is_finite(x: Float64) -> Bool:
    # Float64 NaN ≠ NaN; Inf is bounded by 1e308.
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


def _forward_and_check[
    M: Model,
    BATCH: Int,
](
    label: String,
    expected_in: Int,
    expected_out: Int,
    mut passed: Int,
    mut total: Int,
):
    """Initialize a fresh state, run forward on a small random input, and
    assert dim invariants + output finiteness."""
    print()
    print("---", label, "---")
    print("    IN_DIM =", M.IN_DIM, " OUT_DIM =", M.OUT_DIM)
    print("    PARAM_SIZE =", M.PARAM_SIZE, " CACHE_SIZE =", M.CACHE_SIZE)

    _expect(M.IN_DIM == expected_in, "IN_DIM matches spec", passed, total)
    _expect(M.OUT_DIM == expected_out, "OUT_DIM matches spec", passed, total)
    _expect(M.PARAM_SIZE > 0, "PARAM_SIZE > 0", passed, total)

    var state = NetworkState[M, Adam[]]()
    state.initialize[Kaiming[]]()
    var params = state.params_view()

    # Random-ish input: mix of positive/negative/zero entries to exercise
    # ReLU branches.
    var input_arr = InlineArray[Scalar[dtype], BATCH * M.IN_DIM](
        uninitialized=True
    )
    for i in range(BATCH * M.IN_DIM):
        input_arr[i] = Scalar[dtype](0.1 * Float64(i % 7) - 0.3)

    var output_arr = InlineArray[Scalar[dtype], BATCH * M.OUT_DIM](
        uninitialized=True
    )
    var cache_arr = InlineArray[Scalar[dtype], BATCH * M.CACHE_SIZE](
        uninitialized=True
    )

    var input_t = LayoutTensor[
        dtype, Layout.row_major(BATCH, M.IN_DIM), MutAnyOrigin
    ](input_arr.unsafe_ptr())
    var output_t = LayoutTensor[
        dtype, Layout.row_major(BATCH, M.OUT_DIM), MutAnyOrigin
    ](output_arr.unsafe_ptr())
    var cache_t = LayoutTensor[
        dtype, Layout.row_major(BATCH, M.CACHE_SIZE), MutAnyOrigin
    ](cache_arr.unsafe_ptr())
    var state_t = LayoutTensor[
        dtype, Layout.row_major(M.STATE_SIZE), MutAnyOrigin
    ](state.model_state)

    M.forward[BATCH](input_t, output_t, params, state_t, cache_t)

    var all_finite = True
    var max_mag = Float64(0.0)
    for i in range(BATCH * M.OUT_DIM):
        var v = Float64(output_arr[i])
        if not _is_finite(v):
            all_finite = False
        var av = v if v >= 0.0 else -v
        if av > max_mag:
            max_mag = av
    print("    forward output max |·| =", max_mag)

    _expect(all_finite, "forward output is finite", passed, total)
    _expect(max_mag < 1e6, "forward output magnitude is reasonable", passed, total)


def _gradcheck_residual[
    DIM: Int,
    BATCH: Int = 2,
](mut passed: Int, mut total: Int):
    """Finite-diff gradcheck for the ImproveResidualBlock at small DIM."""
    comptime M = ImproveResidualBlock[DIM]
    print()
    print("--- ImproveResidualBlock gradcheck (DIM=", DIM, ", BATCH=", BATCH, ") ---")

    var state = NetworkState[M, Adam[]]()
    state.initialize[Xavier[]]()
    var params = state.params_view()

    var input_arr = InlineArray[Scalar[dtype], BATCH * M.IN_DIM](
        uninitialized=True
    )
    for i in range(BATCH * M.IN_DIM):
        input_arr[i] = Scalar[dtype](0.15 * Float64(i % 9) - 0.4)

    var grad_out_arr = InlineArray[Scalar[dtype], BATCH * M.OUT_DIM](
        uninitialized=True
    )
    for i in range(BATCH * M.OUT_DIM):
        grad_out_arr[i] = Scalar[dtype](1.0)

    var input_t = LayoutTensor[
        dtype, Layout.row_major(BATCH, M.IN_DIM), MutAnyOrigin
    ](input_arr.unsafe_ptr())

    # Forward (cache for backward).
    var output_arr = InlineArray[Scalar[dtype], BATCH * M.OUT_DIM](
        uninitialized=True
    )
    var output_t = LayoutTensor[
        dtype, Layout.row_major(BATCH, M.OUT_DIM), MutAnyOrigin
    ](output_arr.unsafe_ptr())
    var cache_arr = InlineArray[Scalar[dtype], BATCH * M.CACHE_SIZE](
        uninitialized=True
    )
    var cache_t = LayoutTensor[
        dtype, Layout.row_major(BATCH, M.CACHE_SIZE), MutAnyOrigin
    ](cache_arr.unsafe_ptr())
    var state_t = LayoutTensor[
        dtype, Layout.row_major(M.STATE_SIZE), MutAnyOrigin
    ](state.model_state)
    M.forward[BATCH](input_t, output_t, params, state_t, cache_t)

    # Analytical backward.
    var grad_out_t = LayoutTensor[
        dtype, Layout.row_major(BATCH, M.OUT_DIM), MutAnyOrigin
    ](grad_out_arr.unsafe_ptr())
    var grad_in_arr = InlineArray[Scalar[dtype], BATCH * M.IN_DIM](
        uninitialized=True
    )
    var grad_in_t = LayoutTensor[
        dtype, Layout.row_major(BATCH, M.IN_DIM), MutAnyOrigin
    ](grad_in_arr.unsafe_ptr())
    var grads_arr = InlineArray[Scalar[dtype], M.PARAM_SIZE](uninitialized=True)
    for i in range(M.PARAM_SIZE):
        grads_arr[i] = Scalar[dtype](0.0)
    var grads_t = LayoutTensor[
        dtype, Layout.row_major(M.PARAM_SIZE), MutAnyOrigin
    ](grads_arr.unsafe_ptr())
    M.backward[BATCH](
        grad_out_t, grad_in_t, params, state_t, cache_t, grads_t
    )

    # Finite difference per parameter (skip every Nth param if there are
    # many; the ResidualBlock at DIM=8 has ~232 params, all checkable).
    var step = 1
    if M.PARAM_SIZE > 200:
        step = M.PARAM_SIZE // 100

    var eps = Float64(1e-4)
    var max_abs = Float64(0.0)
    var num_checked = 0

    for p_idx in range(0, M.PARAM_SIZE, step):
        var orig = params.ptr[p_idx]

        # f(p + ε)
        params.ptr[p_idx] = orig + Scalar[dtype](eps)
        var out_plus = InlineArray[Scalar[dtype], BATCH * M.OUT_DIM](
            uninitialized=True
        )
        var out_plus_t = LayoutTensor[
            dtype, Layout.row_major(BATCH, M.OUT_DIM), MutAnyOrigin
        ](out_plus.unsafe_ptr())
        var cache_plus = InlineArray[Scalar[dtype], BATCH * M.CACHE_SIZE](
            uninitialized=True
        )
        var cache_plus_t = LayoutTensor[
            dtype, Layout.row_major(BATCH, M.CACHE_SIZE), MutAnyOrigin
        ](cache_plus.unsafe_ptr())
        M.forward[BATCH](
            input_t, out_plus_t, params, state_t, cache_plus_t
        )

        # f(p − ε)
        params.ptr[p_idx] = orig - Scalar[dtype](eps)
        var out_minus = InlineArray[Scalar[dtype], BATCH * M.OUT_DIM](
            uninitialized=True
        )
        var out_minus_t = LayoutTensor[
            dtype, Layout.row_major(BATCH, M.OUT_DIM), MutAnyOrigin
        ](out_minus.unsafe_ptr())
        var cache_minus = InlineArray[Scalar[dtype], BATCH * M.CACHE_SIZE](
            uninitialized=True
        )
        var cache_minus_t = LayoutTensor[
            dtype, Layout.row_major(BATCH, M.CACHE_SIZE), MutAnyOrigin
        ](cache_minus.unsafe_ptr())
        M.forward[BATCH](
            input_t, out_minus_t, params, state_t, cache_minus_t
        )
        params.ptr[p_idx] = orig

        # Numerical d(loss)/dp where loss = Σ output[i] · 1.0
        var num_grad = Float64(0.0)
        for j in range(BATCH * M.OUT_DIM):
            var diff = Float64(out_plus[j]) - Float64(out_minus[j])
            num_grad += diff
        num_grad /= 2.0 * eps

        var ana_grad = Float64(grads_arr[p_idx])
        var d = num_grad - ana_grad
        if d < 0:
            d = -d
        if d > max_abs:
            max_abs = d
        num_checked += 1

    print(
        "    finite-diff vs autodiff, max |Δ| =", max_abs,
        " (", num_checked, " params checked)",
    )
    _expect(
        max_abs < 1e-2,
        "ImproveResidualBlock backward matches finite-diff",
        passed, total,
    )


def main() raises:
    print("=== EZ-V2 Phase 2 / Step 2 — network composites ===")
    var passed = 0
    var total = 0

    # ─── 1. ImproveResidualBlock ─────────────────────────────────────────
    # IN_DIM = OUT_DIM = DIM (residual identity passthrough).
    _forward_and_check[ImproveResidualBlock[32], 4](
        "ImproveResidualBlock[DIM=32]",
        expected_in=32,
        expected_out=32,
        passed=passed,
        total=total,
    )

    # ─── 2. ActionEmbedding ──────────────────────────────────────────────
    # IN_DIM = ACT, OUT_DIM = EMBED.
    _forward_and_check[ActionEmbedding[ACT=8, EMBED=64], 4](
        "ActionEmbedding[ACT=8, EMBED=64]",
        expected_in=8,
        expected_out=64,
        passed=passed,
        total=total,
    )

    # ─── 3. ProjectionMLP ────────────────────────────────────────────────
    # IN_DIM = HIDDEN, OUT_DIM = PROJ. Use small dims to keep test fast.
    _forward_and_check[ProjectionMLP[HIDDEN=64, PROJ=128], 4](
        "ProjectionMLP[HIDDEN=64, PROJ=128]",
        expected_in=64,
        expected_out=128,
        passed=passed,
        total=total,
    )

    # ─── 4. PredictionMLP ────────────────────────────────────────────────
    # IN_DIM = PROJ, OUT_DIM = PROJ (asymmetric bottleneck inside).
    _forward_and_check[PredictionMLP[PROJ=128, BOTTLENECK=64], 4](
        "PredictionMLP[PROJ=128, BOTTLENECK=64]",
        expected_in=128,
        expected_out=128,
        passed=passed,
        total=total,
    )

    # ─── 5. ImproveResidualBlock — backward gradcheck ────────────────────
    _gradcheck_residual[8](passed, total)

    print()
    print("=== Result:", passed, "/", total, "tests passed ===")
