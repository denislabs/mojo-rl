"""TiedLinear (weight-tied LM head) — CPU correctness + tie invariants.

  1. forward / grad-input / grad-weight match closed-form references
     (out = x@Wᵀ, dx = dout@W, dW += doutᵀ@x), accumulating into the
     *borrowed* grad buffer.
  2. A `TiedLinear` exposes ZERO owned params to reflection — so the
     optimizer collects the shared weight exactly once (via the source
     leaf), never double-counting it (`AdamW.total_size == 0`).
  3. The grad buffer ACCUMULATES across two vjp calls (the property that
     lets the embedding + head both write into one shared buffer).

Run:
    pixi run mojo run -I . tests/nn/test_tied_linear.mojo
"""

from std.random import seed, random_float64
from std.testing import assert_true, assert_almost_equal
from layout import TileTensor, row_major

from mojo_rl.nn.constants import DT
from mojo_rl.nn.primitives.tied_linear import TiedLinear
from mojo_rl.nn.optimizer import AdamW
from mojo_rl.nn.initializer import Zero
from mojo_rl.nn.core.module import mptr

comptime IN = 5    # EMBED
comptime OUT = 7   # VOCAB
comptime B = 4


def _fill_rand(mut l: List[Scalar[DT]]):
    for i in range(len(l)):
        l[i] = Scalar[DT](random_float64(-1.0, 1.0))


def test_tied_linear_math() raises:
    print("test_tied_linear_math ...", end=" ")
    seed(7)
    var W = List[Scalar[DT]](length=OUT * IN, fill=0.0)
    var gW = List[Scalar[DT]](length=OUT * IN, fill=0.0)
    var x = List[Scalar[DT]](length=B * IN, fill=0.0)
    var dout = List[Scalar[DT]](length=B * OUT, fill=0.0)
    _fill_rand(W)
    _fill_rand(x)
    _fill_rand(dout)

    var head = TiedLinear[IN, OUT].make[target="cpu", INIT=Zero]()
    head.tie_to(mptr(W.unsafe_ptr()), mptr(gW.unsafe_ptr()))

    # forward: out[b,v] = Σ_e x[b,e]·W[v,e]
    var out = List[Scalar[DT]](length=B * OUT, fill=0.0)
    var x_tt = TileTensor(mptr(x.unsafe_ptr()), row_major[B, IN]())
    var out_tt = TileTensor(mptr(out.unsafe_ptr()), row_major[B, OUT]())
    head.forward["cpu", B](x_tt, output=out_tt)
    for b in range(B):
        for v in range(OUT):
            var expv: Float64 = 0.0
            for e in range(IN):
                expv += Float64(x[b * IN + e]) * Float64(W[v * IN + e])
            assert_almost_equal(
                Float64(out[b * OUT + v]), expv, atol=1e-4,
                msg="forward mismatch",
            )

    # vjp: grad-input dx[b,e] = Σ_v dout[b,v]·W[v,e]; grad-weight accumulate.
    var dx = List[Scalar[DT]](length=B * IN, fill=0.0)
    var dout_tt = TileTensor(mptr(dout.unsafe_ptr()), row_major[B, OUT]())
    var dx_tt = TileTensor(mptr(dx.unsafe_ptr()), row_major[B, IN]())
    head.vjp["cpu", B](dout_tt, dx_tt)
    for b in range(B):
        for e in range(IN):
            var expv: Float64 = 0.0
            for v in range(OUT):
                expv += Float64(dout[b * OUT + v]) * Float64(W[v * IN + e])
            assert_almost_equal(
                Float64(dx[b * IN + e]), expv, atol=1e-4,
                msg="grad-input mismatch",
            )
    for v in range(OUT):
        for e in range(IN):
            var expv: Float64 = 0.0
            for b in range(B):
                expv += Float64(dout[b * OUT + v]) * Float64(x[b * IN + e])
            assert_almost_equal(
                Float64(gW[v * IN + e]), expv, atol=1e-4,
                msg="grad-weight mismatch",
            )
    print("PASS")


def test_tied_linear_has_no_owned_params() raises:
    # The borrowed weight must be INVISIBLE to reflection so the optimizer
    # collects the shared weight only once (via the source leaf). An AdamW
    # built over a bare TiedLinear therefore sees zero parameters.
    print("test_tied_linear_has_no_owned_params ...", end=" ")
    var head = TiedLinear[IN, OUT].make[target="cpu", INIT=Zero]()
    var opt = AdamW.make["cpu", M = TiedLinear[IN, OUT]](head)
    assert_true(
        opt.total_size == 0,
        "TiedLinear must expose 0 params (got " + String(opt.total_size) + ")",
    )
    print("PASS")


def test_tied_grad_accumulates() raises:
    # Two vjp calls must ACCUMULATE into the shared grad (the property that
    # lets the embedding + head both write one buffer in a backward pass).
    print("test_tied_grad_accumulates ...", end=" ")
    seed(11)
    var W = List[Scalar[DT]](length=OUT * IN, fill=0.0)
    var gW = List[Scalar[DT]](length=OUT * IN, fill=0.0)
    var x = List[Scalar[DT]](length=B * IN, fill=0.0)
    var dout = List[Scalar[DT]](length=B * OUT, fill=0.0)
    _fill_rand(W)
    _fill_rand(x)
    _fill_rand(dout)
    var head = TiedLinear[IN, OUT].make[target="cpu", INIT=Zero]()
    head.tie_to(mptr(W.unsafe_ptr()), mptr(gW.unsafe_ptr()))
    var x_tt = TileTensor(mptr(x.unsafe_ptr()), row_major[B, IN]())
    var out = List[Scalar[DT]](length=B * OUT, fill=0.0)
    var out_tt = TileTensor(mptr(out.unsafe_ptr()), row_major[B, OUT]())
    var dx = List[Scalar[DT]](length=B * IN, fill=0.0)
    var dout_tt = TileTensor(mptr(dout.unsafe_ptr()), row_major[B, OUT]())
    var dx_tt = TileTensor(mptr(dx.unsafe_ptr()), row_major[B, IN]())

    head.forward["cpu", B](x_tt, output=out_tt)
    head.vjp["cpu", B](dout_tt, dx_tt)
    var after_one = Float64(gW[0])
    head.vjp["cpu", B](dout_tt, dx_tt)  # accumulate again
    var after_two = Float64(gW[0])
    assert_almost_equal(
        after_two, 2.0 * after_one, atol=1e-4,
        msg="grad did not accumulate across two vjp calls",
    )
    print("PASS")


def main() raises:
    print("=" * 60)
    print("TiedLinear tests")
    print("=" * 60)
    test_tied_linear_math()
    test_tied_linear_has_no_owned_params()
    test_tied_grad_accumulates()
    print("=" * 60)
    print("ALL PASSED")
    print("=" * 60)
