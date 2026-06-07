"""Unit coverage for three previously-untested nn2 primitives (audit L6).

  * ZeroLinear[IN, OUT]  — weight AND bias start at zero (AdaLN-zero
    identity at init), so forward on any input is all-zeros until grads
    ramp it up.
  * MSEPerSample[DIM]    — per-row mean squared error (ARITY=2, OUT=1):
    out[b] = mean_i (a-b)^2; backward grad_a = (2/DIM)·go·(a-b),
    grad_b = -grad_a.
  * BranchConcat[*B]     — fan-out then column-concat: output =
    [branch0(x) | branch1(x) | …]; verified against the analytic
    relu/tanh of the input.

Run: `pixi run mojo run -I . tests/nn2/test_untested_primitives_cpu.mojo`
"""

from std.math import tanh as math_tanh
from std.memory import alloc
from std.testing import assert_true
from layout import TileTensor, row_major

from mojo_rl.nn2.constants import DT
from mojo_rl.nn2.core.tensor_pack import TensorPack
from mojo_rl.nn2.initializer import Zero, Xavier
from mojo_rl.nn2.primitives.zero_linear import ZeroLinear
from mojo_rl.nn2.primitives.mse_per_sample import MSEPerSample
from mojo_rl.nn2.primitives.relu import ReLU
from mojo_rl.nn2.primitives.tanh import Tanh
from mojo_rl.nn2.combinators.branch_concat import BranchConcat


def _absf(x: Scalar[DT]) -> Scalar[DT]:
    return x if x >= Scalar[DT](0) else -x


def test_zero_linear() raises:
    print("--- ZeroLinear: zero output at init ---")
    comptime IN = 4
    comptime OUT = 3
    comptime BATCH = 2
    # INIT=Xavier on purpose: ZeroLinear must override it back to zero.
    var zl = ZeroLinear[IN, OUT].make[target="cpu", INIT=Xavier]()

    var xin: UnsafePointer[Scalar[DT], MutAnyOrigin] = alloc[Scalar[DT]](
        BATCH * IN
    )
    var yout: UnsafePointer[Scalar[DT], MutAnyOrigin] = alloc[Scalar[DT]](
        BATCH * OUT
    )
    for i in range(BATCH * IN):
        xin[i] = Scalar[DT](0.5 * Float64(i) - 1.0)
    for i in range(BATCH * OUT):
        yout[i] = Scalar[DT](999.0)  # poison

    var x_t = TileTensor(xin, row_major[BATCH, IN]())
    var y_t = TileTensor(yout, row_major[BATCH, OUT]())
    zl.forward["cpu", BATCH](x_t, output=y_t)

    var max_abs: Scalar[DT] = 0.0
    for i in range(BATCH * OUT):
        var a = _absf(yout[i])
        if a > max_abs:
            max_abs = a
    print("  max |out| =", max_abs)
    assert_true(
        max_abs == Scalar[DT](0.0),
        "ZeroLinear forward must be exactly zero at init",
    )
    xin.free(); yout.free()
    print("  ok")


def test_mse_per_sample() raises:
    print("--- MSEPerSample: forward value + backward grads ---")
    comptime DIM = 3
    comptime BATCH = 2
    var m = MSEPerSample[DIM].make[target="cpu", INIT=Zero]()

    var a: UnsafePointer[Scalar[DT], MutAnyOrigin] = alloc[Scalar[DT]](
        BATCH * DIM
    )
    var b: UnsafePointer[Scalar[DT], MutAnyOrigin] = alloc[Scalar[DT]](
        BATCH * DIM
    )
    var out: UnsafePointer[Scalar[DT], MutAnyOrigin] = alloc[Scalar[DT]](BATCH)
    # row0: a=(1,2,3) b=(0,0,0) -> (1+4+9)/3 = 14/3
    # row1: a=(2,2,2) b=(1,0,-1)-> (1+4+9)/3 = 14/3
    a[0] = 1; a[1] = 2; a[2] = 3
    a[3] = 2; a[4] = 2; a[5] = 2
    b[0] = 0; b[1] = 0; b[2] = 0
    b[3] = 1; b[4] = 0; b[5] = -1

    var a_t = TileTensor(a, row_major[BATCH, DIM]())
    var b_t = TileTensor(b, row_major[BATCH, DIM]())
    var o_t = TileTensor(out, row_major[BATCH, 1]())
    m.forward["cpu", BATCH](
            TensorPack[2].of(a_t, b_t), output=o_t,
        )

    comptime expect = Scalar[DT](14.0 / 3.0)
    for r in range(BATCH):
        print("  out[", r, "] =", out[r], " expect", expect)
        assert_true(
            _absf(out[r] - expect) < Scalar[DT](1e-5),
            "MSEPerSample forward value mismatch",
        )

    # Backward with go = 1 per row: grad_a = (2/DIM)*(a-b), grad_b = -grad_a.
    var go: UnsafePointer[Scalar[DT], MutAnyOrigin] = alloc[Scalar[DT]](BATCH)
    var ga: UnsafePointer[Scalar[DT], MutAnyOrigin] = alloc[Scalar[DT]](
        BATCH * DIM
    )
    var gb: UnsafePointer[Scalar[DT], MutAnyOrigin] = alloc[Scalar[DT]](
        BATCH * DIM
    )
    for r in range(BATCH):
        go[r] = 1.0
    var go_t = TileTensor(go, row_major[BATCH, 1]())
    var ga_t = TileTensor(ga, row_major[BATCH, DIM]())
    var gb_t = TileTensor(gb, row_major[BATCH, DIM]())
    m.vjp["cpu", BATCH](go_t, TensorPack[2].of(ga_t, gb_t))

    comptime c = Scalar[DT](2.0 / Float64(DIM))
    var max_err: Scalar[DT] = 0.0
    for r in range(BATCH):
        for i in range(DIM):
            var diff = a[r * DIM + i] - b[r * DIM + i]
            var want_a = c * diff
            max_err = max(max_err, _absf(ga[r * DIM + i] - want_a))
            max_err = max(max_err, _absf(gb[r * DIM + i] - (-want_a)))
    print("  max |grad - analytic| =", max_err)
    assert_true(
        max_err < Scalar[DT](1e-6),
        "MSEPerSample backward grads must match the analytic form",
    )
    a.free(); b.free(); out.free(); go.free(); ga.free(); gb.free()
    print("  ok")


def test_branch_concat() raises:
    print("--- BranchConcat: [relu(x) | tanh(x)] ---")
    comptime DIM = 3
    comptime BATCH = 2
    comptime OUT = 2 * DIM
    var bc = BranchConcat[ReLU[DIM], Tanh[DIM]].make[
        target="cpu", INIT=Zero
    ]()

    var xin: UnsafePointer[Scalar[DT], MutAnyOrigin] = alloc[Scalar[DT]](
        BATCH * DIM
    )
    var yout: UnsafePointer[Scalar[DT], MutAnyOrigin] = alloc[Scalar[DT]](
        BATCH * OUT
    )
    # Mix of signs so relu actually clips.
    xin[0] = -1.0; xin[1] = 0.5; xin[2] = 2.0
    xin[3] = 1.5; xin[4] = -0.3; xin[5] = 0.0

    var x_t = TileTensor(xin, row_major[BATCH, DIM]())
    var y_t = TileTensor(yout, row_major[BATCH, OUT]())
    bc.forward["cpu", BATCH](x_t, output=y_t)

    var max_err: Scalar[DT] = 0.0
    for r in range(BATCH):
        for i in range(DIM):
            var v = xin[r * DIM + i]
            var want_relu = v if v >= Scalar[DT](0) else Scalar[DT](0)
            var want_tanh = math_tanh(v)
            # cols [0:DIM] = relu branch, [DIM:2DIM] = tanh branch
            max_err = max(max_err, _absf(yout[r * OUT + i] - want_relu))
            max_err = max(
                max_err, _absf(yout[r * OUT + DIM + i] - want_tanh)
            )
    print("  max |out - [relu|tanh]| =", max_err)
    assert_true(
        max_err < Scalar[DT](1e-6),
        "BranchConcat must column-concat the per-branch outputs",
    )
    xin.free(); yout.free()
    print("  ok")


def main() raises:
    print("=" * 70)
    print("nn2 untested-primitive coverage (L6)")
    print("=" * 70)
    test_zero_linear()
    test_mse_per_sample()
    test_branch_concat()
    print("=" * 70)
    print("ALL PASSED")
    print("=" * 70)
