"""CPU tests for Slice[IN, START, END]. Phase 10E primitive.

Verifies:
  - Forward extracts cols [START:END]
  - Backward writes grad_output into [START:END] of grad_input and ZEROS
    the rest (critical for CG v2 scatter-add semantics when two Slice
    nodes share a predecessor — e.g. action + log_prob both slice from
    rsample output)
  - FD gradcheck on the slice-range entries (rest are dead code so no FD)
"""

from std.math import abs as fabs
from std.memory import alloc
from std.testing import assert_almost_equal, assert_true, assert_equal
from layout import TileTensor, row_major

from mojo_rl.nn2.constants import DT
from mojo_rl.nn2.primitives.slice import Slice
from mojo_rl.nn2.initializer import Zero


def test_slice_forward_backward() raises:
    comptime IN_DIM = 5
    comptime START = 1
    comptime END = 4
    comptime OUT_DIM = END - START  # 3
    comptime BATCH = 2

    var in_buf = alloc[Scalar[DT]](BATCH * IN_DIM)
    var out_buf = alloc[Scalar[DT]](BATCH * OUT_DIM)
    var go_buf = alloc[Scalar[DT]](BATCH * OUT_DIM)
    var gi_buf = alloc[Scalar[DT]](BATCH * IN_DIM)

    for i in range(BATCH * IN_DIM):
        in_buf[i] = Scalar[DT](Float32(i) * 0.21 - 0.6)
    for i in range(BATCH * OUT_DIM):
        go_buf[i] = Scalar[DT](Float32(i) * 0.13 + 0.1)
    # Pre-fill grad_input with sentinel — backward must overwrite ALL cols.
    for i in range(BATCH * IN_DIM):
        gi_buf[i] = Scalar[DT](999.0)

    var op = Slice[IN_DIM, START, END].make[target="cpu", INIT=Zero]()

    var in_t = TileTensor(in_buf, row_major[BATCH, IN_DIM]())
    var out_t = TileTensor(out_buf, row_major[BATCH, OUT_DIM]())
    op.forward["cpu", BATCH](in_t, out_t)
    for b in range(BATCH):
        for j in range(OUT_DIM):
            assert_almost_equal(
                out_buf[b * OUT_DIM + j],
                in_buf[b * IN_DIM + START + j],
                atol=1e-7,
            )

    var go_t = TileTensor(go_buf, row_major[BATCH, OUT_DIM]())
    var gi_t = TileTensor(gi_buf, row_major[BATCH, IN_DIM]())
    op.backward["cpu", BATCH](go_t, gi_t)
    # Zero outside [START, END), grad_output inside.
    for b in range(BATCH):
        for k in range(IN_DIM):
            if k < START or k >= END:
                assert_almost_equal(gi_buf[b * IN_DIM + k], 0.0, atol=1e-7)
            else:
                var j = k - START
                assert_almost_equal(
                    gi_buf[b * IN_DIM + k],
                    go_buf[b * OUT_DIM + j],
                    atol=1e-7,
                )

    # FD gradcheck on the slice-range entries.
    var eps: Scalar[DT] = 1e-3
    var max_rel: Scalar[DT] = 0.0
    for b in range(BATCH):
        for k in range(START, END):
            var idx = b * IN_DIM + k
            var orig = in_buf[idx]
            in_buf[idx] = orig + eps
            op.forward["cpu", BATCH](in_t, out_t)
            var L_plus: Scalar[DT] = 0.0
            for kk in range(BATCH * OUT_DIM):
                L_plus += go_buf[kk] * out_buf[kk]
            in_buf[idx] = orig - eps
            op.forward["cpu", BATCH](in_t, out_t)
            var L_minus: Scalar[DT] = 0.0
            for kk in range(BATCH * OUT_DIM):
                L_minus += go_buf[kk] * out_buf[kk]
            in_buf[idx] = orig
            var num = (L_plus - L_minus) / (Scalar[DT](2.0) * eps)
            var ana = gi_buf[idx]
            var ae = fabs(num - ana)
            var denom = fabs(num) + fabs(ana) + Scalar[DT](1e-6)
            var rel = ae / denom
            if rel > max_rel:
                max_rel = rel
    print("  Slice FD max_rel=", max_rel)
    assert_true(max_rel < Scalar[DT](1e-3), "Slice FD too loose")

    in_buf.free(); out_buf.free(); go_buf.free(); gi_buf.free()
    print("  test_slice PASSED")


def main() raises:
    print("=" * 70)
    print("nn2 Phase 10E — Slice primitive CPU tests")
    print("=" * 70)
    test_slice_forward_backward()
    print("=" * 70)
    print("ALL PASSED")
    print("=" * 70)
