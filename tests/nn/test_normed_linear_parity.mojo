"""NormedLinear vs Sequential[Linear, LayerNorm, Mish] parity test.

NormedLinear is supposed to compute y = Mish(LayerNorm(Wx + b)) — the same
function as the equivalent Sequential composition. This test verifies:

1. INIT: gamma slice = 1.0, beta slice = 0, b slice = 0, W slice = init values
2. FORWARD parity: NormedLinear(x, params) == Sequential(x, equivalent_params)
3. BACKWARD parity: param grads match Sequential's accumulated grads
4. NUMERIC GRADCHECK: backward param grads match finite-difference gradients

A clean pass on (1)+(2) means NormedLinear's forward is correct.
A clean pass on (3) means NormedLinear's backward chain (Mish→LN→Linear) is correct.
A clean pass on (4) means the gradient computation itself is mathematically right.

If all four pass and TD-MPC2 still NaNs, the bug is somewhere else
(training-loop interaction, BPTT topology, etc.).

Usage:
    pixi run mojo run -I . tests/nn/test_normed_linear_parity.mojo
"""

from std.gpu.host import DeviceContext
from std.math import abs, sqrt
from std.memory import alloc, memset
from layout import Layout, LayoutTensor
from mojo_rl.nn.constants import dtype
from mojo_rl.nn.training import NetworkState, GPUNetworkState
from mojo_rl.nn.optimizer import Adam
from mojo_rl.nn.initializer import Kaiming
from mojo_rl.nn.model import (
    Model,
    Sequential,
    Linear,
    LayerNorm,
    Mish,
    NormedLinear,
    SimNorm,
)


# =============================================================================
# Helpers
# =============================================================================


def fill_deterministic[
    SIZE: Int
](mut t: LayoutTensor[dtype, Layout.row_major(SIZE), MutAnyOrigin], salt: Int):
    """Fill a 1D tensor with a deterministic pattern."""
    for i in range(SIZE):
        # Mix index + salt to spread values over a reasonable range
        var v = Float64((i * 7919 + salt * 31337) % 997) / 997.0 - 0.5
        t[i] = Scalar[dtype](v * 0.5)


def fill_input[
    BATCH: Int, DIM: Int
](
    mut t: LayoutTensor[dtype, Layout.row_major(BATCH, DIM), MutAnyOrigin],
    salt: Int,
):
    """Fill 2D input tensor — values in roughly N(0, 1)."""
    for b in range(BATCH):
        for i in range(DIM):
            var v = Float64(((b * 1009 + i * 53 + salt * 71) % 997)) / 997.0
            # Map to [-1, 1] so input has spread
            t[b, i] = Scalar[dtype](2.0 * v - 1.0)


def max_abs_diff_1d[
    N: Int
](
    a: LayoutTensor[dtype, Layout.row_major(N), MutAnyOrigin],
    b: LayoutTensor[dtype, Layout.row_major(N), MutAnyOrigin],
) -> Float64:
    var max_d: Float64 = 0.0
    for i in range(N):
        var d = abs(Float64(a[i][0]) - Float64(b[i][0]))
        if d > max_d:
            max_d = d
    return max_d


def max_abs_diff_2d[
    M: Int, N: Int
](
    a: LayoutTensor[dtype, Layout.row_major(M, N), MutAnyOrigin],
    b: LayoutTensor[dtype, Layout.row_major(M, N), MutAnyOrigin],
) -> Float64:
    var max_d: Float64 = 0.0
    for i in range(M):
        for j in range(N):
            var d = abs(Float64(a[i, j][0]) - Float64(b[i, j][0]))
            if d > max_d:
                max_d = d
    return max_d


# =============================================================================
# Test 1: Init is what we expect
# =============================================================================


def test_init():
    print("\n[Test 1] NormedLinear init: gamma=1, beta=0, b=0, W=Kaiming")
    print("-" * 60)

    comptime IN = 8
    comptime OUT = 4
    comptime PS = IN * OUT + 3 * OUT  # W + b + gamma + beta

    var state = NetworkState[NormedLinear[IN, OUT], Adam[]]()
    state.initialize[Kaiming[]]()

    var p = state.params_view()

    # Offsets: W at 0..IN*OUT, b at IN*OUT..IN*OUT+OUT, gamma at +OUT, beta at +OUT
    comptime W_END = IN * OUT
    comptime B_END = IN * OUT + OUT
    comptime G_END = IN * OUT + 2 * OUT
    comptime BETA_END = IN * OUT + 3 * OUT

    # Check gamma == 1
    var gamma_ok = True
    var gamma_max_dev: Float64 = 0.0
    for i in range(W_END + OUT, G_END):
        var v = Float64(p[i][0])
        var dev = abs(v - 1.0)
        if dev > gamma_max_dev:
            gamma_max_dev = dev
        if dev > 1e-6:
            gamma_ok = False
    print("  gamma == 1.0:", gamma_ok, " max_dev =", gamma_max_dev)

    # Check beta == 0
    var beta_ok = True
    var beta_max: Float64 = 0.0
    for i in range(G_END, BETA_END):
        var v = abs(Float64(p[i][0]))
        if v > beta_max:
            beta_max = v
        if v > 1e-6:
            beta_ok = False
    print("  beta  == 0.0:", beta_ok, " max =", beta_max)

    # Check b == 0
    var b_ok = True
    var b_max: Float64 = 0.0
    for i in range(W_END, B_END):
        var v = abs(Float64(p[i][0]))
        if v > b_max:
            b_max = v
        if v > 1e-6:
            b_ok = False
    print("  b     == 0.0:", b_ok, " max =", b_max)

    # Check W is non-zero (has some values)
    var w_nonzero = False
    for i in range(0, W_END):
        if abs(Float64(p[i][0])) > 1e-6:
            w_nonzero = True
            break
    print("  W non-zero (Kaiming) :", w_nonzero)

    if gamma_ok and beta_ok and b_ok and w_nonzero:
        print("  PASS")
    else:
        print("  FAIL")


# =============================================================================
# Test 2: Forward parity vs Sequential[Linear, LayerNorm, Mish]
# =============================================================================


def test_forward_parity():
    print("\n[Test 2] Forward parity: NormedLinear vs Linear+LayerNorm+Mish")
    print("-" * 60)

    comptime IN = 8
    comptime OUT = 4
    comptime BS = 3

    comptime NL = NormedLinear[IN, OUT]
    comptime SEQ = Sequential[Linear[IN, OUT], LayerNorm[OUT], Mish[OUT]]

    # ── Initialize NormedLinear with deterministic params ──
    var nl_state = NetworkState[NL, Adam[]]()
    nl_state.initialize[Kaiming[]]()  # gamma=1, beta=0, b=0, W=Kaiming
    var nl_p = nl_state.params_view()

    # ── Manually build matching params for Sequential ──
    # Sequential param layout: [Linear params (W + b) | LayerNorm params (gamma+beta) | Mish params (none)]
    var seq_state = NetworkState[SEQ, Adam[]]()
    seq_state.initialize[Kaiming[]]()
    var seq_p = seq_state.params_view()

    # Copy NL's W to SEQ's W (Linear's W is the first IN*OUT values)
    for i in range(IN * OUT):
        seq_p[i] = nl_p[i]
    # SEQ's bias starts at IN*OUT (Linear bias) — copy NL's b (also at IN*OUT)
    for i in range(OUT):
        seq_p[IN * OUT + i] = nl_p[IN * OUT + i]
    # SEQ's LayerNorm gamma starts at IN*OUT + OUT — copy NL's gamma
    for i in range(OUT):
        seq_p[IN * OUT + OUT + i] = nl_p[IN * OUT + OUT + i]
    # SEQ's LayerNorm beta starts at IN*OUT + 2*OUT — copy NL's beta
    for i in range(OUT):
        seq_p[IN * OUT + 2 * OUT + i] = nl_p[IN * OUT + 2 * OUT + i]

    # ── Build input ──
    var input_ptr = alloc[Scalar[dtype]](BS * IN)
    var input_t = LayoutTensor[dtype, Layout.row_major(BS, IN), MutAnyOrigin](
        input_ptr
    )
    fill_input[BS, IN](input_t, salt=1)

    # ── NormedLinear forward ──
    var nl_out_ptr = alloc[Scalar[dtype]](BS * OUT)
    var nl_cache_ptr = alloc[Scalar[dtype]](BS * NL.CACHE_SIZE)
    var nl_out = LayoutTensor[dtype, Layout.row_major(BS, OUT), MutAnyOrigin](
        nl_out_ptr
    )
    var nl_cache = LayoutTensor[
        dtype, Layout.row_major(BS, NL.CACHE_SIZE), MutAnyOrigin
    ](nl_cache_ptr)
    NL.forward[BS](input_t, nl_out, nl_p, nl_state.model_state_view(), nl_cache)

    # ── Sequential forward ──
    var seq_out_ptr = alloc[Scalar[dtype]](BS * OUT)
    var seq_cache_ptr = alloc[Scalar[dtype]](BS * SEQ.CACHE_SIZE)
    var seq_out = LayoutTensor[dtype, Layout.row_major(BS, OUT), MutAnyOrigin](
        seq_out_ptr
    )
    var seq_cache = LayoutTensor[
        dtype, Layout.row_major(BS, SEQ.CACHE_SIZE), MutAnyOrigin
    ](seq_cache_ptr)
    SEQ.forward[BS](
        input_t, seq_out, seq_p, seq_state.model_state_view(), seq_cache
    )

    # ── Compare ──
    var max_diff = max_abs_diff_2d[BS, OUT](nl_out, seq_out)
    print("  max |NormedLinear - Sequential| forward output =", max_diff)
    if max_diff < 1e-5:
        print("  PASS")
    else:
        print("  FAIL")
        # Print sample values for debugging
        for b in range(BS):
            for j in range(OUT):
                print(
                    "    [",
                    b,
                    ",",
                    j,
                    "] NL=",
                    Float64(nl_out[b, j][0]),
                    " SEQ=",
                    Float64(seq_out[b, j][0]),
                )


# =============================================================================
# Test 3: Backward parity vs Sequential
# =============================================================================


def test_backward_parity():
    print("\n[Test 3] Backward parity: param grads match Sequential")
    print("-" * 60)

    comptime IN = 8
    comptime OUT = 4
    comptime BS = 3

    comptime NL = NormedLinear[IN, OUT]
    comptime SEQ = Sequential[Linear[IN, OUT], LayerNorm[OUT], Mish[OUT]]

    # Initialize and sync params (same as test 2)
    var nl_state = NetworkState[NL, Adam[]]()
    nl_state.initialize[Kaiming[]]()
    var seq_state = NetworkState[SEQ, Adam[]]()
    seq_state.initialize[Kaiming[]]()

    var nl_p = nl_state.params_view()
    var seq_p = seq_state.params_view()
    for i in range(IN * OUT + 3 * OUT):
        seq_p[i] = nl_p[i]

    # Input + grad_output
    var input_ptr = alloc[Scalar[dtype]](BS * IN)
    var input_t = LayoutTensor[dtype, Layout.row_major(BS, IN), MutAnyOrigin](
        input_ptr
    )
    fill_input[BS, IN](input_t, salt=2)

    var grad_out_ptr = alloc[Scalar[dtype]](BS * OUT)
    var grad_out = LayoutTensor[dtype, Layout.row_major(BS, OUT), MutAnyOrigin](
        grad_out_ptr
    )
    fill_input[BS, OUT](grad_out, salt=3)

    # NL.backward mutates grad_out in-place (Mish chain stored there).
    # Snapshot BEFORE running NL.backward so SEQ gets the original.
    var grad_out_seq_ptr = alloc[Scalar[dtype]](BS * OUT)
    var grad_out_seq = LayoutTensor[
        dtype, Layout.row_major(BS, OUT), MutAnyOrigin
    ](grad_out_seq_ptr)
    for b in range(BS):
        for j in range(OUT):
            grad_out_seq[b, j] = grad_out[b, j]

    # ── NormedLinear forward+backward ──
    var nl_out_ptr = alloc[Scalar[dtype]](BS * OUT)
    var nl_cache_ptr = alloc[Scalar[dtype]](BS * NL.CACHE_SIZE)
    var nl_grad_in_ptr = alloc[Scalar[dtype]](BS * IN)
    var nl_out = LayoutTensor[dtype, Layout.row_major(BS, OUT), MutAnyOrigin](
        nl_out_ptr
    )
    var nl_cache = LayoutTensor[
        dtype, Layout.row_major(BS, NL.CACHE_SIZE), MutAnyOrigin
    ](nl_cache_ptr)
    var nl_grad_in = LayoutTensor[
        dtype, Layout.row_major(BS, IN), MutAnyOrigin
    ](nl_grad_in_ptr)

    NL.forward[BS](input_t, nl_out, nl_p, nl_state.model_state_view(), nl_cache)

    # Zero param grads, then call backward (CPU backward accumulates)
    var nl_g = nl_state.grads_view()
    for i in range(IN * OUT + 3 * OUT):
        nl_g[i] = 0
    NL.backward[BS](
        grad_out,
        nl_grad_in,
        nl_p,
        nl_state.model_state_view(),
        nl_cache,
        nl_g,
    )

    # ── Sequential forward+backward ──
    var seq_out_ptr = alloc[Scalar[dtype]](BS * OUT)
    var seq_cache_ptr = alloc[Scalar[dtype]](BS * SEQ.CACHE_SIZE)
    var seq_grad_in_ptr = alloc[Scalar[dtype]](BS * IN)
    var seq_out = LayoutTensor[dtype, Layout.row_major(BS, OUT), MutAnyOrigin](
        seq_out_ptr
    )
    var seq_cache = LayoutTensor[
        dtype, Layout.row_major(BS, SEQ.CACHE_SIZE), MutAnyOrigin
    ](seq_cache_ptr)
    var seq_grad_in = LayoutTensor[
        dtype, Layout.row_major(BS, IN), MutAnyOrigin
    ](seq_grad_in_ptr)

    SEQ.forward[BS](
        input_t, seq_out, seq_p, seq_state.model_state_view(), seq_cache
    )

    var seq_g = seq_state.grads_view()
    for i in range(IN * OUT + 3 * OUT):
        seq_g[i] = 0

    SEQ.backward[BS](
        grad_out_seq,
        seq_grad_in,
        seq_p,
        seq_state.model_state_view(),
        seq_cache,
        seq_g,
    )

    # ── Compare grad_input ──
    var grad_in_diff = max_abs_diff_2d[BS, IN](nl_grad_in, seq_grad_in)
    print("  max |grad_input| diff =", grad_in_diff)

    # ── Compare param grads ──
    var grad_p_max: Float64 = 0.0
    for i in range(IN * OUT + 3 * OUT):
        var d = abs(Float64(nl_g[i][0]) - Float64(seq_g[i][0]))
        if d > grad_p_max:
            grad_p_max = d
    print("  max |grad_params| diff =", grad_p_max)

    if grad_in_diff < 1e-4 and grad_p_max < 1e-4:
        print("  PASS")
    else:
        print("  FAIL")
        # Per-section breakdown
        var dW: Float64 = 0.0
        for i in range(0, IN * OUT):
            var d = abs(Float64(nl_g[i][0]) - Float64(seq_g[i][0]))
            if d > dW:
                dW = d
        var db: Float64 = 0.0
        for i in range(IN * OUT, IN * OUT + OUT):
            var d = abs(Float64(nl_g[i][0]) - Float64(seq_g[i][0]))
            if d > db:
                db = d
        var dgamma: Float64 = 0.0
        for i in range(IN * OUT + OUT, IN * OUT + 2 * OUT):
            var d = abs(Float64(nl_g[i][0]) - Float64(seq_g[i][0]))
            if d > dgamma:
                dgamma = d
        var dbeta: Float64 = 0.0
        for i in range(IN * OUT + 2 * OUT, IN * OUT + 3 * OUT):
            var d = abs(Float64(nl_g[i][0]) - Float64(seq_g[i][0]))
            if d > dbeta:
                dbeta = d
        print(
            "    per-section: dW =",
            dW,
            " db =",
            db,
            " dgamma =",
            dgamma,
            " dbeta =",
            dbeta,
        )


# =============================================================================
# Test 4: Numeric gradient check on params
# =============================================================================


def test_gradcheck():
    print("\n[Test 4] Numeric gradcheck: NormedLinear param grads vs FD")
    print("-" * 60)

    comptime IN = 4
    comptime OUT = 3
    comptime BS = 2
    comptime PS = IN * OUT + 3 * OUT

    comptime NL = NormedLinear[IN, OUT]

    var state = NetworkState[NL, Adam[]]()
    state.initialize[Kaiming[]]()
    var p = state.params_view()
    var s = state.model_state_view()

    var input_ptr = alloc[Scalar[dtype]](BS * IN)
    var input_t = LayoutTensor[dtype, Layout.row_major(BS, IN), MutAnyOrigin](
        input_ptr
    )
    fill_input[BS, IN](input_t, salt=4)

    # Build a fixed grad_output so the loss is L = sum(grad_output * output)
    # so that dL/d(output) = grad_output, and dL/d(params) is exactly what
    # backward computes.
    var grad_out_ptr = alloc[Scalar[dtype]](BS * OUT)
    var grad_out = LayoutTensor[dtype, Layout.row_major(BS, OUT), MutAnyOrigin](
        grad_out_ptr
    )
    fill_input[BS, OUT](grad_out, salt=5)

    # Analytical grad
    var out_ptr = alloc[Scalar[dtype]](BS * OUT)
    var out_t = LayoutTensor[dtype, Layout.row_major(BS, OUT), MutAnyOrigin](
        out_ptr
    )
    var cache_ptr = alloc[Scalar[dtype]](BS * NL.CACHE_SIZE)
    var cache_t = LayoutTensor[
        dtype, Layout.row_major(BS, NL.CACHE_SIZE), MutAnyOrigin
    ](cache_ptr)
    NL.forward[BS](input_t, out_t, p, s, cache_t)

    var grad_in_ptr = alloc[Scalar[dtype]](BS * IN)
    var grad_in = LayoutTensor[dtype, Layout.row_major(BS, IN), MutAnyOrigin](
        grad_in_ptr
    )
    var g_view = state.grads_view()
    for i in range(PS):
        g_view[i] = 0
    var grad_out_copy_ptr = alloc[Scalar[dtype]](BS * OUT)
    var grad_out_copy = LayoutTensor[
        dtype, Layout.row_major(BS, OUT), MutAnyOrigin
    ](grad_out_copy_ptr)
    for b in range(BS):
        for j in range(OUT):
            grad_out_copy[b, j] = grad_out[b, j]
    NL.backward[BS](grad_out_copy, grad_in, p, s, cache_t, g_view)

    # Save analytical grad
    var analytical = alloc[Scalar[dtype]](PS)
    for i in range(PS):
        analytical[i] = g_view[i][0]

    # Numeric FD grad
    comptime EPS: Float64 = 1e-3
    var max_rel_err: Float64 = 0.0
    var max_abs_err: Float64 = 0.0
    var worst_idx = 0
    for i in range(PS):
        var orig = Float64(p[i][0])
        # f(p + eps)
        p[i] = Scalar[dtype](orig + EPS)
        NL.forward[BS](input_t, out_t, p, s, cache_t)
        var L_plus: Float64 = 0.0
        for b in range(BS):
            for j in range(OUT):
                L_plus += Float64(out_t[b, j][0]) * Float64(grad_out[b, j][0])
        # f(p - eps)
        p[i] = Scalar[dtype](orig - EPS)
        NL.forward[BS](input_t, out_t, p, s, cache_t)
        var L_minus: Float64 = 0.0
        for b in range(BS):
            for j in range(OUT):
                L_minus += Float64(out_t[b, j][0]) * Float64(grad_out[b, j][0])
        # Restore
        p[i] = Scalar[dtype](orig)

        var fd_grad = (L_plus - L_minus) / (2.0 * EPS)
        var ana_grad = Float64(analytical[i][0])
        var abs_err = abs(fd_grad - ana_grad)
        var denom = abs(fd_grad) + abs(ana_grad) + 1e-8
        var rel_err = abs_err / denom
        if abs_err > max_abs_err:
            max_abs_err = abs_err
        if rel_err > max_rel_err:
            max_rel_err = rel_err
            worst_idx = i

    print("  max abs error =", max_abs_err)
    print("  max rel error =", max_rel_err, " (idx", worst_idx, ")")
    if max_rel_err < 5e-3:
        print("  PASS")
    else:
        print("  FAIL")


# =============================================================================
# Main
# =============================================================================


def test_sequential_gradcheck():
    """Numeric gradcheck on Sequential[Linear, LayerNorm, Mish]. If
    NormedLinear passes but this fails, the bug is in the Sequential
    chain (intermediate buffers, layer backward, or composition).
    """
    print("\n[Test 5] Numeric gradcheck on Sequential[Linear, LayerNorm, Mish]")
    print("-" * 60)

    comptime IN = 4
    comptime OUT = 3
    comptime BS = 2

    comptime SEQ = Sequential[Linear[IN, OUT], LayerNorm[OUT], Mish[OUT]]
    comptime PS = SEQ.PARAM_SIZE

    var state = NetworkState[SEQ, Adam[]]()
    state.initialize[Kaiming[]]()
    var p = state.params_view()
    var s = state.model_state_view()

    # Force gamma=1, beta=0, b=0 (same as NormedLinear init) so we test the
    # exact same network as Test 4
    for i in range(IN * OUT, IN * OUT + OUT):  # b
        p[i] = 0
    for i in range(IN * OUT + OUT, IN * OUT + 2 * OUT):  # gamma
        p[i] = 1
    for i in range(IN * OUT + 2 * OUT, IN * OUT + 3 * OUT):  # beta
        p[i] = 0

    var input_ptr = alloc[Scalar[dtype]](BS * IN)
    var input_t = LayoutTensor[dtype, Layout.row_major(BS, IN), MutAnyOrigin](
        input_ptr
    )
    fill_input[BS, IN](input_t, salt=4)

    var grad_out_ptr = alloc[Scalar[dtype]](BS * OUT)
    var grad_out = LayoutTensor[dtype, Layout.row_major(BS, OUT), MutAnyOrigin](
        grad_out_ptr
    )
    fill_input[BS, OUT](grad_out, salt=5)

    var out_ptr = alloc[Scalar[dtype]](BS * OUT)
    var out_t = LayoutTensor[dtype, Layout.row_major(BS, OUT), MutAnyOrigin](
        out_ptr
    )
    var cache_ptr = alloc[Scalar[dtype]](BS * SEQ.CACHE_SIZE)
    var cache_t = LayoutTensor[
        dtype, Layout.row_major(BS, SEQ.CACHE_SIZE), MutAnyOrigin
    ](cache_ptr)
    SEQ.forward[BS](input_t, out_t, p, s, cache_t)

    var grad_in_ptr = alloc[Scalar[dtype]](BS * IN)
    var grad_in = LayoutTensor[dtype, Layout.row_major(BS, IN), MutAnyOrigin](
        grad_in_ptr
    )
    var g_view = state.grads_view()
    for i in range(PS):
        g_view[i] = 0

    var grad_out_copy_ptr = alloc[Scalar[dtype]](BS * OUT)
    var grad_out_copy = LayoutTensor[
        dtype, Layout.row_major(BS, OUT), MutAnyOrigin
    ](grad_out_copy_ptr)
    for b in range(BS):
        for j in range(OUT):
            grad_out_copy[b, j] = grad_out[b, j]
    SEQ.backward[BS](grad_out_copy, grad_in, p, s, cache_t, g_view)

    var analytical = alloc[Scalar[dtype]](PS)
    for i in range(PS):
        analytical[i] = g_view[i][0]

    comptime EPS: Float64 = 1e-3
    var max_rel_err: Float64 = 0.0
    var max_abs_err: Float64 = 0.0
    var worst_idx = 0
    for i in range(PS):
        var orig = Float64(p[i][0])
        p[i] = Scalar[dtype](orig + EPS)
        SEQ.forward[BS](input_t, out_t, p, s, cache_t)
        var L_plus: Float64 = 0.0
        for b in range(BS):
            for j in range(OUT):
                L_plus += Float64(out_t[b, j][0]) * Float64(grad_out[b, j][0])
        p[i] = Scalar[dtype](orig - EPS)
        SEQ.forward[BS](input_t, out_t, p, s, cache_t)
        var L_minus: Float64 = 0.0
        for b in range(BS):
            for j in range(OUT):
                L_minus += Float64(out_t[b, j][0]) * Float64(grad_out[b, j][0])
        p[i] = Scalar[dtype](orig)

        var fd_grad = (L_plus - L_minus) / (2.0 * EPS)
        var ana_grad = Float64(analytical[i][0])
        var abs_err = abs(fd_grad - ana_grad)
        var denom = abs(fd_grad) + abs(ana_grad) + 1e-8
        var rel_err = abs_err / denom
        if abs_err > max_abs_err:
            max_abs_err = abs_err
        if rel_err > max_rel_err:
            max_rel_err = rel_err
            worst_idx = i

    print("  max abs error =", max_abs_err)
    print("  max rel error =", max_rel_err, " (idx", worst_idx, ")")
    if max_rel_err < 5e-3:
        print("  PASS — Sequential is correct, NormedLinear bug?")
    else:
        print("  FAIL — Sequential[Linear,LN,Mish] backward chain has a bug")
        # Per-section breakdown
        var ranges = List[Tuple[Int, Int, String]]()
        ranges.append((0, IN * OUT, String("W (Linear)")))
        ranges.append((IN * OUT, IN * OUT + OUT, String("b (Linear)")))
        ranges.append(
            (IN * OUT + OUT, IN * OUT + 2 * OUT, String("gamma (LayerNorm)"))
        )
        ranges.append(
            (
                IN * OUT + 2 * OUT,
                IN * OUT + 3 * OUT,
                String("beta (LayerNorm)"),
            )
        )
        for r in ranges:
            var section_max: Float64 = 0.0
            var section_idx = 0
            for i in range(r[0], r[1]):
                var orig = Float64(p[i][0])
                p[i] = Scalar[dtype](orig + EPS)
                SEQ.forward[BS](input_t, out_t, p, s, cache_t)
                var L_plus: Float64 = 0.0
                for b in range(BS):
                    for j in range(OUT):
                        L_plus += Float64(out_t[b, j][0]) * Float64(
                            grad_out[b, j][0]
                        )
                p[i] = Scalar[dtype](orig - EPS)
                SEQ.forward[BS](input_t, out_t, p, s, cache_t)
                var L_minus: Float64 = 0.0
                for b in range(BS):
                    for j in range(OUT):
                        L_minus += Float64(out_t[b, j][0]) * Float64(
                            grad_out[b, j][0]
                        )
                p[i] = Scalar[dtype](orig)
                var fd_grad = (L_plus - L_minus) / (2.0 * EPS)
                var ana_grad = Float64(analytical[i][0])
                var rel_err = abs(fd_grad - ana_grad) / (
                    abs(fd_grad) + abs(ana_grad) + 1e-8
                )
                if rel_err > section_max:
                    section_max = rel_err
                    section_idx = i
            print(
                "    ",
                r[2],
                ": max rel err =",
                section_max,
                " at idx",
                section_idx,
            )


def test_cpu_vs_gpu_normed_linear() raises:
    """Verify NormedLinear's GPU forward+backward match its CPU counterpart.

    If CPU passes gradcheck (Test 4) and CPU == GPU, then GPU is also correct.
    If CPU != GPU, the bug is in a GPU kernel.
    """
    print("\n[Test 6] CPU vs GPU parity for NormedLinear")
    print("-" * 60)

    comptime IN = 16
    comptime OUT = 8
    comptime BS = 4

    comptime NL = NormedLinear[IN, OUT]
    comptime PS = NL.PARAM_SIZE
    comptime CS = NL.CACHE_SIZE
    comptime WS = NL.WORKSPACE_SIZE_PER_SAMPLE

    with DeviceContext() as ctx:
        var cpu_state = NetworkState[NL, Adam[]]()
        cpu_state.initialize[Kaiming[]]()

        var gpu = GPUNetworkState[NL, Adam[], dtype](ctx)
        gpu.upload_from(cpu_state, ctx)

        # ── Build deterministic input ──
        var input_ptr = alloc[Scalar[dtype]](BS * IN)
        for i in range(BS * IN):
            (input_ptr + i)[] = Scalar[dtype](
                Float64(((i * 53) % 197)) / 197.0 * 2.0 - 1.0
            )

        # ── CPU forward + backward ──
        var cpu_out_ptr = alloc[Scalar[dtype]](BS * OUT)
        var cpu_cache_ptr = alloc[Scalar[dtype]](BS * CS if CS > 0 else 1)
        var cpu_grad_in_ptr = alloc[Scalar[dtype]](BS * IN)
        var cpu_input_t = LayoutTensor[
            dtype, Layout.row_major(BS, IN), MutAnyOrigin
        ](input_ptr)
        var cpu_out_t = LayoutTensor[
            dtype, Layout.row_major(BS, OUT), MutAnyOrigin
        ](cpu_out_ptr)
        var cpu_cache_t = LayoutTensor[
            dtype, Layout.row_major(BS, CS), MutAnyOrigin
        ](cpu_cache_ptr)
        var cpu_grad_in_t = LayoutTensor[
            dtype, Layout.row_major(BS, IN), MutAnyOrigin
        ](cpu_grad_in_ptr)

        NL.forward[BS](
            cpu_input_t,
            cpu_out_t,
            cpu_state.params_view(),
            cpu_state.model_state_view(),
            cpu_cache_t,
        )

        # grad_out filled deterministically
        var grad_out_ptr = alloc[Scalar[dtype]](BS * OUT)
        var grad_out_t = LayoutTensor[
            dtype, Layout.row_major(BS, OUT), MutAnyOrigin
        ](grad_out_ptr)
        for i in range(BS * OUT):
            (grad_out_ptr + i)[] = Scalar[dtype](
                Float64(((i * 41 + 17) % 113)) / 113.0 - 0.5
            )

        # Snapshot grad_out for GPU (CPU.backward mutates grad_out)
        var grad_out_gpu_host = ctx.enqueue_create_host_buffer[dtype](BS * OUT)
        for i in range(BS * OUT):
            grad_out_gpu_host[i] = (grad_out_ptr + i)[]

        var cpu_g = cpu_state.grads_view()
        for i in range(PS):
            cpu_g[i] = 0
        NL.backward[BS](
            grad_out_t,
            cpu_grad_in_t,
            cpu_state.params_view(),
            cpu_state.model_state_view(),
            cpu_cache_t,
            cpu_g,
        )

        # ── GPU forward + backward ──
        var gpu_input = ctx.enqueue_create_buffer[dtype](BS * IN)
        var gpu_input_host = ctx.enqueue_create_host_buffer[dtype](BS * IN)
        for i in range(BS * IN):
            gpu_input_host[i] = (input_ptr + i)[]
        ctx.enqueue_copy(gpu_input, gpu_input_host)
        var gpu_input_t = LayoutTensor[
            dtype, Layout.row_major(BS, IN), MutAnyOrigin
        ](gpu_input.unsafe_ptr())

        var gpu_out = ctx.enqueue_create_buffer[dtype](BS * OUT)
        var gpu_out_t = LayoutTensor[
            dtype, Layout.row_major(BS, OUT), MutAnyOrigin
        ](gpu_out.unsafe_ptr())

        var gpu_cache = ctx.enqueue_create_buffer[dtype](BS * CS)
        var gpu_cache_t = LayoutTensor[
            dtype, Layout.row_major(BS, CS), MutAnyOrigin
        ](gpu_cache.unsafe_ptr())

        var gpu_ws_size = BS * WS if WS > 0 else 1
        var gpu_ws = ctx.enqueue_create_buffer[dtype](gpu_ws_size)

        NL.forward_gpu[BS](
            ctx,
            gpu_out_t,
            gpu_input_t,
            gpu.params_view(),
            gpu.model_state_view(),
            gpu_cache_t,
            gpu_ws,
        )

        # GPU grad_out
        var gpu_grad_out = ctx.enqueue_create_buffer[dtype](BS * OUT)
        ctx.enqueue_copy(gpu_grad_out, grad_out_gpu_host)
        var gpu_grad_out_t = LayoutTensor[
            dtype, Layout.row_major(BS, OUT), MutAnyOrigin
        ](gpu_grad_out.unsafe_ptr())

        var gpu_grad_in = ctx.enqueue_create_buffer[dtype](BS * IN)
        var gpu_grad_in_t = LayoutTensor[
            dtype, Layout.row_major(BS, IN), MutAnyOrigin
        ](gpu_grad_in.unsafe_ptr())

        # Zero GPU grads
        var gpu_g = gpu.grads_view()
        var zero_host = ctx.enqueue_create_host_buffer[dtype](PS)
        for i in range(PS):
            zero_host[i] = 0
        ctx.enqueue_copy(gpu.grads_buf, zero_host)

        NL.backward_gpu[BS](
            ctx,
            gpu_grad_in_t,
            gpu_grad_out_t,
            gpu.params_view(),
            gpu.model_state_view(),
            gpu_cache_t,
            gpu_g,
            gpu_ws,
        )
        ctx.synchronize()

        # ── Compare ──
        # Forward
        var gpu_out_host = ctx.enqueue_create_host_buffer[dtype](BS * OUT)
        ctx.enqueue_copy(gpu_out_host, gpu_out)
        ctx.synchronize()
        var fwd_diff: Float64 = 0.0
        for i in range(BS * OUT):
            var d = abs(Float64(cpu_out_ptr[i][0]) - Float64(gpu_out_host[i]))
            if d > fwd_diff:
                fwd_diff = d
        print("  forward max diff =", fwd_diff)

        # grad_input
        var gpu_grad_in_host = ctx.enqueue_create_host_buffer[dtype](BS * IN)
        ctx.enqueue_copy(gpu_grad_in_host, gpu_grad_in)
        ctx.synchronize()
        var gi_diff: Float64 = 0.0
        for i in range(BS * IN):
            var d = abs(
                Float64(cpu_grad_in_ptr[i][0]) - Float64(gpu_grad_in_host[i])
            )
            if d > gi_diff:
                gi_diff = d
        print("  grad_input max diff =", gi_diff)

        # grad_params
        var gpu_grads_host = ctx.enqueue_create_host_buffer[dtype](PS)
        ctx.enqueue_copy(gpu_grads_host, gpu.grads_buf)
        ctx.synchronize()

        var gp_diff: Float64 = 0.0
        var dW: Float64 = 0.0
        var db: Float64 = 0.0
        var dgamma: Float64 = 0.0
        var dbeta: Float64 = 0.0
        for i in range(PS):
            var d = abs(Float64(cpu_g[i][0]) - Float64(gpu_grads_host[i]))
            if d > gp_diff:
                gp_diff = d
            if i < IN * OUT:
                if d > dW:
                    dW = d
            elif i < IN * OUT + OUT:
                if d > db:
                    db = d
            elif i < IN * OUT + 2 * OUT:
                if d > dgamma:
                    dgamma = d
            else:
                if d > dbeta:
                    dbeta = d
        print(
            "  grad_params max diff =",
            gp_diff,
            "(dW=",
            dW,
            "db=",
            db,
            "dgamma=",
            dgamma,
            "dbeta=",
            dbeta,
            ")",
        )

        if fwd_diff < 1e-4 and gi_diff < 1e-3 and gp_diff < 1e-3:
            print("  PASS")
        else:
            print("  FAIL — GPU NormedLinear diverges from CPU")


def test_dynamics_arch_gradcheck():
    """Numeric gradcheck on TD-MPC2 dynamics architecture:
    Sequential[NormedLinear, NormedLinear, Linear, LayerNorm, SimNorm].
    """
    print(
        "\n[Test 7] Dynamics arch gradcheck (NL → NL → Linear → LN → SimNorm)"
    )
    print("-" * 60)

    comptime IN = 8
    comptime MLP = 8
    comptime LATENT = 8
    comptime SIMPLEX = 4
    comptime BS = 2

    comptime DYN = Sequential[
        NormedLinear[IN, MLP],
        NormedLinear[MLP, LATENT],
        Linear[LATENT, LATENT],
        LayerNorm[LATENT],
        SimNorm[LATENT, SIMPLEX],
    ]
    comptime PS = DYN.PARAM_SIZE

    var state = NetworkState[DYN, Adam[]]()
    state.initialize[Kaiming[]]()
    var p = state.params_view()
    var s = state.model_state_view()

    var input_ptr = alloc[Scalar[dtype]](BS * IN)
    var input_t = LayoutTensor[dtype, Layout.row_major(BS, IN), MutAnyOrigin](
        input_ptr
    )
    fill_input[BS, IN](input_t, salt=10)

    var grad_out_ptr = alloc[Scalar[dtype]](BS * LATENT)
    var grad_out = LayoutTensor[
        dtype, Layout.row_major(BS, LATENT), MutAnyOrigin
    ](grad_out_ptr)
    fill_input[BS, LATENT](grad_out, salt=11)

    var out_ptr = alloc[Scalar[dtype]](BS * LATENT)
    var out_t = LayoutTensor[dtype, Layout.row_major(BS, LATENT), MutAnyOrigin](
        out_ptr
    )
    var cache_ptr = alloc[Scalar[dtype]](BS * DYN.CACHE_SIZE)
    var cache_t = LayoutTensor[
        dtype, Layout.row_major(BS, DYN.CACHE_SIZE), MutAnyOrigin
    ](cache_ptr)
    DYN.forward[BS](input_t, out_t, p, s, cache_t)

    var grad_in_ptr = alloc[Scalar[dtype]](BS * IN)
    var grad_in = LayoutTensor[dtype, Layout.row_major(BS, IN), MutAnyOrigin](
        grad_in_ptr
    )
    var g_view = state.grads_view()
    for i in range(PS):
        g_view[i] = 0

    var grad_out_copy_ptr = alloc[Scalar[dtype]](BS * LATENT)
    var grad_out_copy = LayoutTensor[
        dtype, Layout.row_major(BS, LATENT), MutAnyOrigin
    ](grad_out_copy_ptr)
    for b in range(BS):
        for j in range(LATENT):
            grad_out_copy[b, j] = grad_out[b, j]
    DYN.backward[BS](grad_out_copy, grad_in, p, s, cache_t, g_view)

    # FD check
    var analytical = alloc[Scalar[dtype]](PS)
    for i in range(PS):
        analytical[i] = g_view[i][0]

    # See `gradcheck_sequential` and `test_normed_linear_fd_eps_sweep.mojo`
    # for why 1e-2 is the right FP32 FD step for deep chains.
    comptime EPS: Float64 = 1e-2
    var max_rel_err: Float64 = 0.0
    var max_abs_err: Float64 = 0.0
    var worst_idx = 0
    for i in range(PS):
        var orig = Float64(p[i][0])
        p[i] = Scalar[dtype](orig + EPS)
        DYN.forward[BS](input_t, out_t, p, s, cache_t)
        var L_plus: Float64 = 0.0
        for b in range(BS):
            for j in range(LATENT):
                L_plus += Float64(out_t[b, j][0]) * Float64(grad_out[b, j][0])
        p[i] = Scalar[dtype](orig - EPS)
        DYN.forward[BS](input_t, out_t, p, s, cache_t)
        var L_minus: Float64 = 0.0
        for b in range(BS):
            for j in range(LATENT):
                L_minus += Float64(out_t[b, j][0]) * Float64(grad_out[b, j][0])
        p[i] = Scalar[dtype](orig)

        var fd_grad = (L_plus - L_minus) / (2.0 * EPS)
        var ana_grad = Float64(analytical[i][0])
        var abs_err = abs(fd_grad - ana_grad)
        var denom = abs(fd_grad) + abs(ana_grad) + 1e-8
        var rel_err = abs_err / denom
        if abs_err > max_abs_err:
            max_abs_err = abs_err
        if rel_err > max_rel_err:
            max_rel_err = rel_err
            worst_idx = i

    print("  max abs error =", max_abs_err)
    print("  max rel error =", max_rel_err, " (idx", worst_idx, ")")
    # Match the OR criterion used by `gradcheck_sequential`: tiny absolute
    # errors (below the FP32 FD precision floor) are fine even if relative
    # error exceeds 1% — that just means the gradient sits near zero and
    # FP roundoff dominates.
    if max_rel_err < 1e-2 or max_abs_err < 5e-4:
        print("  PASS")
    else:
        print("  FAIL — bug somewhere in the dynamics chain")


def gradcheck_sequential[
    M: Model, IN: Int, OUT: Int, BS: Int = 2
](name: String):
    """Generic FD gradcheck on any Sequential composition."""
    print("\n[gradcheck]", name)
    print("-" * 60)
    comptime PS = M.PARAM_SIZE
    comptime IN_FIX = M.IN_DIM
    comptime OUT_FIX = M.OUT_DIM
    comptime CS = M.CACHE_SIZE

    var state = NetworkState[M, Adam[]]()
    state.initialize[Kaiming[]]()
    var p = state.params_view()
    var s = state.model_state_view()

    var input_ptr = alloc[Scalar[dtype]](BS * IN_FIX)
    var input_t = LayoutTensor[
        dtype, Layout.row_major(BS, IN_FIX), MutAnyOrigin
    ](input_ptr)
    fill_input[BS, IN_FIX](input_t, salt=10)

    var grad_out_ptr = alloc[Scalar[dtype]](BS * OUT_FIX)
    var grad_out = LayoutTensor[
        dtype, Layout.row_major(BS, OUT_FIX), MutAnyOrigin
    ](grad_out_ptr)
    fill_input[BS, OUT_FIX](grad_out, salt=11)

    var out_ptr = alloc[Scalar[dtype]](BS * OUT_FIX)
    var out_t = LayoutTensor[
        dtype, Layout.row_major(BS, OUT_FIX), MutAnyOrigin
    ](out_ptr)
    var cache_ptr = alloc[Scalar[dtype]](BS * CS)
    var cache_t = LayoutTensor[dtype, Layout.row_major(BS, CS), MutAnyOrigin](
        cache_ptr
    )
    M.forward[BS](input_t, out_t, p, s, cache_t)

    var grad_in_ptr = alloc[Scalar[dtype]](BS * IN_FIX)
    var grad_in = LayoutTensor[
        dtype, Layout.row_major(BS, IN_FIX), MutAnyOrigin
    ](grad_in_ptr)
    var g_view = state.grads_view()
    for i in range(PS):
        g_view[i] = 0
    var grad_out_copy_ptr = alloc[Scalar[dtype]](BS * OUT_FIX)
    var grad_out_copy = LayoutTensor[
        dtype, Layout.row_major(BS, OUT_FIX), MutAnyOrigin
    ](grad_out_copy_ptr)
    for b in range(BS):
        for j in range(OUT_FIX):
            grad_out_copy[b, j] = grad_out[b, j]
    M.backward[BS](grad_out_copy, grad_in, p, s, cache_t, g_view)

    var analytical = alloc[Scalar[dtype]](PS)
    for i in range(PS):
        analytical[i] = g_view[i][0]

    # Central-difference FD with FP32 forwards has a U-shaped error curve:
    # truncation `~ eps² · ∂³L/∂p³` dominates at large eps, FP roundoff
    # `~ machine_eps / eps` dominates at small eps. The sweet spot is around
    # √(fp32_machine_eps) ≈ 3e-4 for shallow nets, but **drifts up to ~1e-2
    # for deep chains** (NL → NL → Linear / + LN), where roundoff in
    # (out_plus - out_minus) compounds through multiple matmul+LN passes.
    # 1e-2 was empirically validated as the global sweet spot across all
    # chains in `tests/nn/test_normed_linear_fd_eps_sweep.mojo`.
    comptime EPS: Float64 = 1e-2
    var max_rel_err: Float64 = 0.0
    var max_abs_err: Float64 = 0.0
    var worst_idx = 0
    for i in range(PS):
        var orig = Float64(p[i][0])
        p[i] = Scalar[dtype](orig + EPS)
        M.forward[BS](input_t, out_t, p, s, cache_t)
        var L_plus: Float64 = 0.0
        for b in range(BS):
            for j in range(OUT_FIX):
                L_plus += Float64(out_t[b, j][0]) * Float64(grad_out[b, j][0])
        p[i] = Scalar[dtype](orig - EPS)
        M.forward[BS](input_t, out_t, p, s, cache_t)
        var L_minus: Float64 = 0.0
        for b in range(BS):
            for j in range(OUT_FIX):
                L_minus += Float64(out_t[b, j][0]) * Float64(grad_out[b, j][0])
        p[i] = Scalar[dtype](orig)
        var fd_grad = (L_plus - L_minus) / (2.0 * EPS)
        var ana_grad = Float64(analytical[i][0])
        var abs_err = abs(fd_grad - ana_grad)
        var rel_err = abs_err / (abs(fd_grad) + abs(ana_grad) + 1e-8)
        if abs_err > max_abs_err:
            max_abs_err = abs_err
        if rel_err > max_rel_err:
            max_rel_err = rel_err
            worst_idx = i

    # Re-run FD at worst_idx to print analytical vs FD for context
    var orig = Float64(p[worst_idx][0])
    p[worst_idx] = Scalar[dtype](orig + EPS)
    M.forward[BS](input_t, out_t, p, s, cache_t)
    var Lp: Float64 = 0.0
    for b in range(BS):
        for j in range(OUT_FIX):
            Lp += Float64(out_t[b, j][0]) * Float64(grad_out[b, j][0])
    p[worst_idx] = Scalar[dtype](orig - EPS)
    M.forward[BS](input_t, out_t, p, s, cache_t)
    var Lm: Float64 = 0.0
    for b in range(BS):
        for j in range(OUT_FIX):
            Lm += Float64(out_t[b, j][0]) * Float64(grad_out[b, j][0])
    p[worst_idx] = Scalar[dtype](orig)
    var fd_at_worst = (Lp - Lm) / (2.0 * EPS)
    var ana_at_worst = Float64(analytical[worst_idx][0])
    print(
        "  max abs error =",
        max_abs_err,
        " max rel =",
        max_rel_err,
        " (idx",
        worst_idx,
        ", fd=",
        fd_at_worst,
        ", ana=",
        ana_at_worst,
        ")",
    )
    # PASS if either rel err is low OR abs err is below FD noise floor (~5e-4 for float32)
    if max_rel_err < 1e-2 or max_abs_err < 5e-4:
        print("  PASS")
    else:
        print("  FAIL")


def test_narrowing_dynamics():
    """Add layers one at a time to find which addition breaks gradcheck."""
    print("\n[Test 8] Narrowing the dynamics-chain bug")
    print("-" * 60)

    comptime D = 8
    comptime SIMPLEX = 4

    # Sanity: single NL via Sequential[NL]
    gradcheck_sequential[Sequential[NormedLinear[D, D]], D, D]("Seq[NL]")

    # 2 NormedLinears stacked, BS=2 (default)
    gradcheck_sequential[
        Sequential[NormedLinear[D, D], NormedLinear[D, D]], D, D
    ]("Seq[NL, NL] BS=2")

    # 2 NormedLinears stacked, BS=1 (rule out batch-stride layout bug)
    gradcheck_sequential[
        Sequential[NormedLinear[D, D], NormedLinear[D, D]], D, D, BS=1
    ]("Seq[NL, NL] BS=1")

    # NL stacked with Linear (no LN, no Mish in between)
    gradcheck_sequential[Sequential[NormedLinear[D, D], Linear[D, D]], D, D](
        "Seq[NL, Linear]"
    )

    # Linear → NL
    gradcheck_sequential[Sequential[Linear[D, D], NormedLinear[D, D]], D, D](
        "Seq[Linear, NL]"
    )

    # Add Linear
    gradcheck_sequential[
        Sequential[NormedLinear[D, D], NormedLinear[D, D], Linear[D, D]], D, D
    ]("Seq[NL, NL, Linear]")

    # Add LayerNorm
    gradcheck_sequential[
        Sequential[
            NormedLinear[D, D],
            NormedLinear[D, D],
            Linear[D, D],
            LayerNorm[D],
        ],
        D,
        D,
    ]("Seq[NL, NL, Linear, LN]")

    # Add SimNorm (full dynamics)
    gradcheck_sequential[
        Sequential[
            NormedLinear[D, D],
            NormedLinear[D, D],
            Linear[D, D],
            LayerNorm[D],
            SimNorm[D, SIMPLEX],
        ],
        D,
        D,
    ]("Seq[NL, NL, Linear, LN, SimNorm] (full dynamics)")

    # Just the tail: Linear → LN → SimNorm
    gradcheck_sequential[
        Sequential[Linear[D, D], LayerNorm[D], SimNorm[D, SIMPLEX]], D, D
    ]("Seq[Linear, LN, SimNorm] (tail only)")


def main() raises:
    print("=" * 60)
    print("NormedLinear Layer Tests")
    print("=" * 60)

    test_init()
    test_forward_parity()
    test_backward_parity()
    test_gradcheck()
    test_sequential_gradcheck()
    test_cpu_vs_gpu_normed_linear()
    test_dynamics_arch_gradcheck()
    test_narrowing_dynamics()

    print()
    print("=" * 60)
    print("Done.")
    print("=" * 60)
