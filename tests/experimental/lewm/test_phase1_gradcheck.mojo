"""Phase 1 gradcheck — LeWM building blocks (CPU).

Tests Modulate, Gate, and SIGRegOp via finite-difference on inputs.

Each test sets `grad_output = ones` then compares analytical `grad_input`
(from `vjp`) against `(L(input + eps * e_i) - L(input - eps * e_i)) / (2*eps)`
where `L = sum_i grad_output_i * output_i`.

Run:
    pixi run mojo run -I . tests/experimental/lewm/test_phase1_gradcheck.mojo
"""

from mojo_rl.nn.constants import dtype
from mojo_rl.nn.autodiff.primitives import ModulateOp, GateOp, SIGRegOp
from layout import Layout, LayoutTensor
from std.math import abs


# =============================================================================
# Modulate
# =============================================================================

def test_modulate_gradcheck() raises:
    """Verify ModulateOp gradients against finite differences."""
    comptime BATCH = 3
    comptime DIM = 4
    comptime IN = 3 * DIM     # x | scale | shift
    comptime CACHE = 2 * DIM

    var input_arr = InlineArray[Scalar[dtype], BATCH * IN](uninitialized=True)
    # Spread values across [-0.6, 0.6]
    for i in range(BATCH * IN):
        input_arr[i] = Scalar[dtype](0.13 * Float64(i % 11) - 0.55)

    var grad_out = InlineArray[Scalar[dtype], BATCH * DIM](uninitialized=True)
    for i in range(BATCH * DIM):
        grad_out[i] = Scalar[dtype](1.0)

    var output_arr = InlineArray[Scalar[dtype], BATCH * DIM](uninitialized=True)
    var cache_arr = InlineArray[Scalar[dtype], BATCH * CACHE](uninitialized=True)
    var params_arr = InlineArray[Scalar[dtype], 1](uninitialized=True)

    var input_t = LayoutTensor[
        dtype, Layout.row_major(BATCH, IN), MutAnyOrigin
    ](input_arr.unsafe_ptr())
    var output_t = LayoutTensor[
        dtype, Layout.row_major(BATCH, DIM), MutAnyOrigin
    ](output_arr.unsafe_ptr())
    var cache_t = LayoutTensor[
        dtype, Layout.row_major(BATCH, CACHE), MutAnyOrigin
    ](cache_arr.unsafe_ptr())
    var params_t = LayoutTensor[
        dtype, Layout.row_major(0), MutAnyOrigin
    ](params_arr.unsafe_ptr())

    # Analytical
    ModulateOp[DIM].eval[BATCH](input_t, output_t, params_t, cache_t)

    var grad_out_t = LayoutTensor[
        dtype, Layout.row_major(BATCH, DIM), MutAnyOrigin
    ](grad_out.unsafe_ptr())
    var grad_in = InlineArray[Scalar[dtype], BATCH * IN](uninitialized=True)
    var grad_in_t = LayoutTensor[
        dtype, Layout.row_major(BATCH, IN), MutAnyOrigin
    ](grad_in.unsafe_ptr())
    var gp_arr = InlineArray[Scalar[dtype], 1](uninitialized=True)
    var gp_t = LayoutTensor[
        dtype, Layout.row_major(0), MutAnyOrigin
    ](gp_arr.unsafe_ptr())
    ModulateOp[DIM].vjp[BATCH](
        grad_out_t, grad_in_t, params_t, cache_t, gp_t
    )

    # FD on inputs
    var max_abs = Float64(0.0)
    var eps = Float64(1e-4)
    for idx in range(BATCH * IN):
        var orig = input_arr[idx]
        input_arr[idx] = orig + Scalar[dtype](eps)
        ModulateOp[DIM].eval[BATCH](input_t, output_t, params_t, cache_t)
        var f_plus = Float64(0.0)
        for j in range(BATCH * DIM):
            f_plus += Float64(output_arr[j]) * Float64(grad_out[j])

        input_arr[idx] = orig - Scalar[dtype](eps)
        ModulateOp[DIM].eval[BATCH](input_t, output_t, params_t, cache_t)
        var f_minus = Float64(0.0)
        for j in range(BATCH * DIM):
            f_minus += Float64(output_arr[j]) * Float64(grad_out[j])

        input_arr[idx] = orig

        var num_g = (f_plus - f_minus) / (2.0 * eps)
        var ana_g = Float64(grad_in[idx])
        var err = abs(ana_g - num_g)
        if err > max_abs:
            max_abs = err

    if max_abs < 1e-3:
        print("  [PASS] ModulateOp gradcheck: max_abs_err =", max_abs)
    else:
        print("  [FAIL] ModulateOp gradcheck: max_abs_err =", max_abs)


# =============================================================================
# Gate
# =============================================================================

def test_gate_gradcheck() raises:
    """Verify GateOp gradients against finite differences."""
    comptime BATCH = 3
    comptime DIM = 4
    comptime IN = 3 * DIM     # x | gate | branch
    comptime CACHE = 2 * DIM

    var input_arr = InlineArray[Scalar[dtype], BATCH * IN](uninitialized=True)
    for i in range(BATCH * IN):
        input_arr[i] = Scalar[dtype](0.17 * Float64(i % 9) - 0.45)

    var grad_out = InlineArray[Scalar[dtype], BATCH * DIM](uninitialized=True)
    for i in range(BATCH * DIM):
        grad_out[i] = Scalar[dtype](1.0)

    var output_arr = InlineArray[Scalar[dtype], BATCH * DIM](uninitialized=True)
    var cache_arr = InlineArray[Scalar[dtype], BATCH * CACHE](uninitialized=True)
    var params_arr = InlineArray[Scalar[dtype], 1](uninitialized=True)

    var input_t = LayoutTensor[
        dtype, Layout.row_major(BATCH, IN), MutAnyOrigin
    ](input_arr.unsafe_ptr())
    var output_t = LayoutTensor[
        dtype, Layout.row_major(BATCH, DIM), MutAnyOrigin
    ](output_arr.unsafe_ptr())
    var cache_t = LayoutTensor[
        dtype, Layout.row_major(BATCH, CACHE), MutAnyOrigin
    ](cache_arr.unsafe_ptr())
    var params_t = LayoutTensor[
        dtype, Layout.row_major(0), MutAnyOrigin
    ](params_arr.unsafe_ptr())

    GateOp[DIM].eval[BATCH](input_t, output_t, params_t, cache_t)

    var grad_out_t = LayoutTensor[
        dtype, Layout.row_major(BATCH, DIM), MutAnyOrigin
    ](grad_out.unsafe_ptr())
    var grad_in = InlineArray[Scalar[dtype], BATCH * IN](uninitialized=True)
    var grad_in_t = LayoutTensor[
        dtype, Layout.row_major(BATCH, IN), MutAnyOrigin
    ](grad_in.unsafe_ptr())
    var gp_arr = InlineArray[Scalar[dtype], 1](uninitialized=True)
    var gp_t = LayoutTensor[
        dtype, Layout.row_major(0), MutAnyOrigin
    ](gp_arr.unsafe_ptr())
    GateOp[DIM].vjp[BATCH](
        grad_out_t, grad_in_t, params_t, cache_t, gp_t
    )

    var max_abs = Float64(0.0)
    var eps = Float64(1e-4)
    for idx in range(BATCH * IN):
        var orig = input_arr[idx]
        input_arr[idx] = orig + Scalar[dtype](eps)
        GateOp[DIM].eval[BATCH](input_t, output_t, params_t, cache_t)
        var f_plus = Float64(0.0)
        for j in range(BATCH * DIM):
            f_plus += Float64(output_arr[j]) * Float64(grad_out[j])

        input_arr[idx] = orig - Scalar[dtype](eps)
        GateOp[DIM].eval[BATCH](input_t, output_t, params_t, cache_t)
        var f_minus = Float64(0.0)
        for j in range(BATCH * DIM):
            f_minus += Float64(output_arr[j]) * Float64(grad_out[j])

        input_arr[idx] = orig

        var num_g = (f_plus - f_minus) / (2.0 * eps)
        var ana_g = Float64(grad_in[idx])
        var err = abs(ana_g - num_g)
        if err > max_abs:
            max_abs = err

    if max_abs < 1e-3:
        print("  [PASS] GateOp gradcheck: max_abs_err =", max_abs)
    else:
        print("  [FAIL] GateOp gradcheck: max_abs_err =", max_abs)


# =============================================================================
# SIGReg
# =============================================================================

def test_sigreg_gradcheck() raises:
    """Verify SIGRegOp gradients against finite differences.

    Toy config: dim=8, seq_len=2, num_proj=4, knots=5, batch=4.
    Total input elements per sample = 16, params checked = 64.
    """
    comptime BATCH = 4
    comptime DIM = 8
    comptime SEQ = 2
    comptime PROJ = 4
    comptime KNOTS = 5
    comptime IN = SEQ * DIM
    comptime CACHE = SEQ * PROJ

    var input_arr = InlineArray[Scalar[dtype], BATCH * IN](uninitialized=True)
    # Use values broadly distributed around 0 so projections are non-trivial.
    for i in range(BATCH * IN):
        input_arr[i] = Scalar[dtype](0.31 * Float64(i % 13) - 1.85)

    var grad_out = InlineArray[Scalar[dtype], BATCH](uninitialized=True)
    # Standard loss seed: 1/BATCH per output slot. Sum = 1.
    for b in range(BATCH):
        grad_out[b] = Scalar[dtype](1.0 / Float64(BATCH))

    var output_arr = InlineArray[Scalar[dtype], BATCH](uninitialized=True)
    var cache_arr = InlineArray[Scalar[dtype], BATCH * CACHE](uninitialized=True)
    var params_arr = InlineArray[Scalar[dtype], 1](uninitialized=True)

    var input_t = LayoutTensor[
        dtype, Layout.row_major(BATCH, IN), MutAnyOrigin
    ](input_arr.unsafe_ptr())
    var output_t = LayoutTensor[
        dtype, Layout.row_major(BATCH, 1), MutAnyOrigin
    ](output_arr.unsafe_ptr())
    var cache_t = LayoutTensor[
        dtype, Layout.row_major(BATCH, CACHE), MutAnyOrigin
    ](cache_arr.unsafe_ptr())
    var params_t = LayoutTensor[
        dtype, Layout.row_major(0), MutAnyOrigin
    ](params_arr.unsafe_ptr())

    # Analytical
    SIGRegOp[DIM, SEQ, PROJ, KNOTS].eval[BATCH](
        input_t, output_t, params_t, cache_t
    )

    var grad_out_t = LayoutTensor[
        dtype, Layout.row_major(BATCH, 1), MutAnyOrigin
    ](grad_out.unsafe_ptr())
    var grad_in = InlineArray[Scalar[dtype], BATCH * IN](uninitialized=True)
    var grad_in_t = LayoutTensor[
        dtype, Layout.row_major(BATCH, IN), MutAnyOrigin
    ](grad_in.unsafe_ptr())
    var gp_arr = InlineArray[Scalar[dtype], 1](uninitialized=True)
    var gp_t = LayoutTensor[
        dtype, Layout.row_major(0), MutAnyOrigin
    ](gp_arr.unsafe_ptr())
    SIGRegOp[DIM, SEQ, PROJ, KNOTS].vjp[BATCH](
        grad_out_t, grad_in_t, params_t, cache_t, gp_t
    )

    # Print statistic value for sanity
    print("    SIGReg statistic =", Float64(output_arr[0]))

    # FD on inputs. Larger eps since cos/sin can be smooth but the statistic
    # combines many terms — 1e-3 gives a better signal-to-noise ratio.
    var max_abs = Float64(0.0)
    var max_rel = Float64(0.0)
    var eps = Float64(1e-3)
    var num_checked = 0
    for idx in range(BATCH * IN):
        var orig = input_arr[idx]
        input_arr[idx] = orig + Scalar[dtype](eps)
        SIGRegOp[DIM, SEQ, PROJ, KNOTS].eval[BATCH](
            input_t, output_t, params_t, cache_t
        )
        var f_plus = Float64(0.0)
        for j in range(BATCH):
            f_plus += Float64(output_arr[j]) * Float64(grad_out[j])

        input_arr[idx] = orig - Scalar[dtype](eps)
        SIGRegOp[DIM, SEQ, PROJ, KNOTS].eval[BATCH](
            input_t, output_t, params_t, cache_t
        )
        var f_minus = Float64(0.0)
        for j in range(BATCH):
            f_minus += Float64(output_arr[j]) * Float64(grad_out[j])

        input_arr[idx] = orig

        var num_g = (f_plus - f_minus) / (2.0 * eps)
        var ana_g = Float64(grad_in[idx])
        var err = abs(ana_g - num_g)
        var denom = abs(ana_g) + abs(num_g)
        var rel = Float64(0.0)
        if denom > 1e-6:
            rel = err / denom

        if err > max_abs:
            max_abs = err
        if rel > max_rel:
            max_rel = rel
        num_checked += 1

    # SIGReg is smooth; analytical and FD should match closely. The statistic
    # is small (~1e-2 to 1e0 for this input range) so allow up to 1e-3 abs or
    # 1e-2 relative — comparable tolerances to other primitive gradchecks.
    if max_abs < 1e-3 or max_rel < 1e-2:
        print(
            "  [PASS] SIGRegOp gradcheck: max_abs_err =",
            max_abs,
            "max_rel_err =",
            max_rel,
            "(",
            num_checked,
            "checked)",
        )
    else:
        print(
            "  [FAIL] SIGRegOp gradcheck: max_abs_err =",
            max_abs,
            "max_rel_err =",
            max_rel,
        )


def main() raises:
    print("=== LeWM Phase 1 Gradient Checks (CPU) ===")
    print()
    print("--- ModulateOp ---")
    test_modulate_gradcheck()
    print()
    print("--- GateOp ---")
    test_gate_gradcheck()
    print()
    print("--- SIGRegOp ---")
    test_sigreg_gradcheck()
    print()
    print("=== Phase 1 gradcheck done ===")
