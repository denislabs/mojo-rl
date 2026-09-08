"""`bce_logits_*_t` — BCE-with-logits value and cotangent.

  [1] closed form: `softplus(l) − y·l` equals `−y log σ(l) − (1−y) log(1−σ(l))`
      evaluated in Float64 on a grid, and stays FINITE at |l| = 60 where the
      two-step form is `−log(0)`.
  [2] the cotangent is the derivative: central finite differences of the
      per-row loss, on the same grid.
  [3] the per-row-label entry point agrees with the constant-label one.

Run:
    pixi run mojo run -I . tests/nn/test_bce_logits.mojo
"""

from std.math import abs, exp, log
from std.testing import assert_true

from mojo_rl.nn.constants import DT
from mojo_rl.nn.core.tensor import Tensor
from mojo_rl.nn.loss.bce_logits import bce_logits_const_t, bce_logits_rows_t


comptime N = 13


def _grid() raises -> Tensor:
    var t = Tensor.alloc(N)
    var vals: List[Float64] = [-60.0, -8.0, -3.0, -1.0, -0.3, -0.01, 0.0, 0.01, 0.3, 1.0, 3.0, 8.0, 60.0]
    for i in range(N):
        t.data[i] = Scalar[DT](vals[i])
    return t^


def _ref(l: Float64, y: Float64) -> Float64:
    # −y log σ − (1−y) log(1−σ), σ in Float64; only trusted where σ is not 0/1
    var s = 1.0 / (1.0 + exp(-l))
    return -y * log(s) - (1.0 - y) * log(1.0 - s)


def main() raises:
    print("=" * 70)
    print("bce_logits — value and cotangent")
    print("=" * 70)

    print("[1] closed form on a grid, both labels, finite at |l| = 60 ...")
    var l = _grid()
    var cot = Tensor()
    var rows = Tensor()
    var worst = Float64(0)
    for lab in range(2):
        var y = Float64(lab)
        bce_logits_const_t["cpu", N](l, y, 1.0, cot, rows, None)
        for i in range(N):
            var li = Float64(l.data[i])
            var got = Float64(rows.data[i])
            assert_true(got == got and got < 1e6, "non-finite loss at l=" + String(li))
            if abs(li) < 20.0:
                # ⚠ ABSOLUTE band, scaled by |l|: at l = 8, y = 1 the value is
                # `(8 + 3.4e-4) − 8` in fp32, and a relative band on 3.4e-4
                # sits under the cancellation floor (one ulp of 8 is 4.8e-7).
                var want = _ref(li, y)
                var d = abs(got - want) / (1.0 + abs(li))
                if d > worst:
                    worst = d
            else:
                # softplus(60) = 60 + O(1e-26); softplus(−60) = O(1e-26)
                var want = (li if li > 0.0 else 0.0) - y * li
                assert_true(abs(got - want) < 1e-4, "tail value wrong at l=" + String(li))
    print("      worst |got − want| / (1 + |l|) vs Float64 closed form:", worst)
    assert_true(worst < 2e-6, "closed form mismatch")

    print("[2] cotangent == d(loss)/d(logit), central differences ...")
    var h = Float64(1e-2)
    var worst2 = Float64(0)
    for lab in range(2):
        var y = Float64(lab)
        var scale = 0.37
        bce_logits_const_t["cpu", N](l, y, scale, cot, rows, None)
        var lp = Tensor.alloc(N)
        var lm = Tensor.alloc(N)
        for i in range(N):
            lp.data[i] = Scalar[DT](Float64(l.data[i]) + h)
            lm.data[i] = Scalar[DT](Float64(l.data[i]) - h)
        var cp = Tensor()
        var rp = Tensor()
        var cm = Tensor()
        var rm = Tensor()
        bce_logits_const_t["cpu", N](lp, y, 1.0, cp, rp, None)
        bce_logits_const_t["cpu", N](lm, y, 1.0, cm, rm, None)
        for i in range(N):
            if abs(Float64(l.data[i])) > 20.0:
                continue
            var fd = (Float64(rp.data[i]) - Float64(rm.data[i])) / (2.0 * h) * scale
            var d = abs(fd - Float64(cot.data[i]))
            if d > worst2:
                worst2 = d
    print("      worst |fd − cot|:", worst2)
    assert_true(worst2 < 2e-3, "cotangent disagrees with the finite difference")

    print("[3] per-row labels == constant label ...")
    var labels = Tensor.alloc(N)
    for i in range(N):
        labels.data[i] = Scalar[DT](1.0)
    var cot_r = Tensor()
    var rows_r = Tensor()
    bce_logits_rows_t["cpu", N](l, labels, 0.5, cot_r, rows_r, None)
    bce_logits_const_t["cpu", N](l, 1.0, 0.5, cot, rows, None)
    for i in range(N):
        assert_true(cot_r.data[i] == cot.data[i], "rows/const cotangent differ")
        assert_true(rows_r.data[i] == rows.data[i], "rows/const loss differ")
    print("\n[PASS] bce_logits")
