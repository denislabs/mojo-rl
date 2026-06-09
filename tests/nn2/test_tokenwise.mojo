"""Tokenwise[SEQ_LEN, Inner] — CPU oracle test (Wave B).

Validates the reshape-and-delegate plumbing end-to-end by wrapping a
Linear and comparing forward + grad_input + grad_param against a manual
per-token Linear oracle. Linear[IN,OUT] weight is laid out (IN, OUT):
    out[o] = sum_i x[i]*W[i,o] + bias[o].

Docs: docs/NN2_TRANSFORMER_PORT.md Phase 1 Wave B.
"""

from std.memory import alloc
from std.math import abs
from std.testing import assert_true
from layout import TileTensor, row_major

from mojo_rl.nn2.constants import DT
from mojo_rl.nn2.initializer import Zero
from mojo_rl.nn2.primitives.linear import Linear
from mojo_rl.nn2.combinators.tokenwise import Tokenwise


comptime TOL: Float64 = 1e-4


def _alloc(n: Int) -> UnsafePointer[Scalar[DT], MutAnyOrigin]:
    return rebind[UnsafePointer[Scalar[DT], MutAnyOrigin]](
        alloc[Scalar[DT]](n)
    )


def test_tokenwise_linear() raises:
    print("test_tokenwise_linear ...")
    comptime BATCH = 2
    comptime SEQ = 3
    comptime IN = 4
    comptime OUT = 5
    comptime IN_N = BATCH * SEQ * IN
    comptime OUT_N = BATCH * SEQ * OUT
    comptime W_N = IN * OUT

    var tw = Tokenwise[SEQ, Linear[IN, OUT]].make[target="cpu", INIT=Zero]()
    # Deterministic weights/bias on the (shared) inner Linear.
    var w = tw.inner.weight.value_unsafe_ptr_cpu()   # [IN, OUT]
    var b = tw.inner.bias.value_unsafe_ptr_cpu()     # [OUT]
    for i in range(W_N):
        w[i] = Scalar[DT](0.05 * Float64(i) - 0.3)
    for o in range(OUT):
        b[o] = Scalar[DT](0.1 * Float64(o) - 0.2)

    var x = _alloc(IN_N)
    var y = _alloc(OUT_N)
    var go = _alloc(OUT_N)
    var gi = _alloc(IN_N)
    for i in range(IN_N):
        x[i] = Scalar[DT](0.13 * Float64(i) - 0.5)
    for i in range(OUT_N):
        go[i] = Scalar[DT](0.07 * Float64(i) - 0.25)

    var x_t = TileTensor(x, row_major[BATCH, SEQ * IN]())
    var y_t = TileTensor(y, row_major[BATCH, SEQ * OUT]())
    tw.forward["cpu", BATCH](x_t, output=y_t)

    # Forward oracle: per-token Linear.
    var fwd_err: Float64 = 0.0
    for bt in range(BATCH * SEQ):
        for o in range(OUT):
            var acc: Float64 = Float64(b[o])
            for i in range(IN):
                acc += Float64(x[bt * IN + i]) * Float64(w[i * OUT + o])
            var d = abs(Float64(y[bt * OUT + o]) - acc)
            if d > fwd_err:
                fwd_err = d
    print("   forward max err =", fwd_err)
    assert_true(fwd_err < 1e-5, "Tokenwise[Linear] forward vs per-token oracle")

    # Backward.
    tw.zero_grad["cpu"]()
    var go_t = TileTensor(go, row_major[BATCH, SEQ * OUT]())
    var gi_t = TileTensor(gi, row_major[BATCH, SEQ * IN]())
    tw.vjp["cpu", BATCH](go_t, gi_t)

    # grad_input oracle: grad_in[bt,i] = sum_o go[bt,o]*W[i,o].
    var gi_err: Float64 = 0.0
    for bt in range(BATCH * SEQ):
        for i in range(IN):
            var acc: Float64 = 0.0
            for o in range(OUT):
                acc += Float64(go[bt * OUT + o]) * Float64(w[i * OUT + o])
            var d = abs(Float64(gi[bt * IN + i]) - acc)
            if d > gi_err:
                gi_err = d
    print("   grad_input max err =", gi_err)
    assert_true(gi_err < TOL, "Tokenwise[Linear] grad_input vs oracle")

    # grad_weight oracle: gW[i,o] = sum_bt x[bt,i]*go[bt,o]  (shared weights).
    var gw = tw.inner.weight.grad_unsafe_ptr_cpu()
    var gw_err: Float64 = 0.0
    for i in range(IN):
        for o in range(OUT):
            var acc: Float64 = 0.0
            for bt in range(BATCH * SEQ):
                acc += Float64(x[bt * IN + i]) * Float64(go[bt * OUT + o])
            var d = abs(Float64(gw[i * OUT + o]) - acc)
            if d > gw_err:
                gw_err = d
    print("   grad_weight max err =", gw_err)
    assert_true(gw_err < TOL, "Tokenwise[Linear] grad_weight vs oracle")

    # grad_bias oracle: gB[o] = sum_bt go[bt,o].
    var gb = tw.inner.bias.grad_unsafe_ptr_cpu()
    var gb_err: Float64 = 0.0
    for o in range(OUT):
        var acc: Float64 = 0.0
        for bt in range(BATCH * SEQ):
            acc += Float64(go[bt * OUT + o])
        var d = abs(Float64(gb[o]) - acc)
        if d > gb_err:
            gb_err = d
    print("   grad_bias max err =", gb_err)
    assert_true(gb_err < TOL, "Tokenwise[Linear] grad_bias vs oracle")
    print("  ok")


def main() raises:
    print("=" * 70)
    print("Tokenwise[SEQ, Linear] CPU oracle (Wave B)")
    print("=" * 70)
    test_tokenwise_linear()
    print("=" * 70)
    print("ALL PASSED")
    print("=" * 70)
