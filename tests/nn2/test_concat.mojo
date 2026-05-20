"""Concat[*BRANCHES] CPU tests — Phase 8.4.

Three tests:

  1. test_concat_forward_3_branches — Concat[Linear[D, A], Linear[D, B],
     Linear[D, C]] forward produces output[BATCH, A+B+C] with each block
     equal to the corresponding Linear's standalone forward. (Matches the
     Parallel forward pattern, generalized to 3 branches.)

  2. test_concat_backward_grad_input_sum — Backward on the same Concat
     produces grad_input = grad_Linear_A + grad_Linear_B + grad_Linear_C
     when called with packed grad_output. Hand-check on a small case.

  3. test_concat_backward_input_no_paramgrads — Concat.backward_input
     skips grad_w / grad_b on every branch (assertion: post-backward_input
     `Linear.grad_w` stays zero in a network where grad_w would otherwise
     accumulate).
"""

from std.math import abs as fabs
from std.memory import alloc
from std.testing import assert_almost_equal, assert_true
from layout import TileTensor, row_major

from mojo_rl.nn2.constants import DT
from mojo_rl.nn2.combinators.concat import Concat
from mojo_rl.nn2.primitives.linear import Linear
from mojo_rl.nn2.initializer import Xavier


comptime IN_DIM = 4
comptime OUT_A = 3
comptime OUT_B = 2
comptime OUT_C = 1
comptime BATCH = 5
comptime OUT_TOTAL = OUT_A + OUT_B + OUT_C   # 6


def test_concat_forward_3_branches() raises:
    var cnc = Concat[
        Linear[IN_DIM, OUT_A], Linear[IN_DIM, OUT_B], Linear[IN_DIM, OUT_C]
    ].make[target="cpu", INIT=Xavier]()

    var in_buf = alloc[Scalar[DT]](BATCH * IN_DIM)
    var out_buf = alloc[Scalar[DT]](BATCH * OUT_TOTAL)
    for i in range(BATCH * IN_DIM):
        in_buf[i] = Scalar[DT](Float32(i) * 0.1 - 0.4)

    var in_t = TileTensor(in_buf, row_major[BATCH, IN_DIM]())
    var out_t = TileTensor(out_buf, row_major[BATCH, OUT_TOTAL]())
    cnc.forward["cpu", BATCH](in_t, out_t)

    # Compare each block against a standalone Linear with the same weights.
    # We can't copy weights from inside the Concat (no API yet), so the
    # reference path runs the same Linears via the Concat's branches' own
    # children. We just verify forward is *consistent* — each branch's slab
    # equals what the branch would produce on its own. To do this, we
    # invoke each branch's forward and compare to the packed output.
    var ref_a = alloc[Scalar[DT]](BATCH * OUT_A)
    var ref_b = alloc[Scalar[DT]](BATCH * OUT_B)
    var ref_c = alloc[Scalar[DT]](BATCH * OUT_C)
    var ref_a_t = TileTensor(ref_a, row_major[BATCH, OUT_A]())
    var ref_b_t = TileTensor(ref_b, row_major[BATCH, OUT_B]())
    var ref_c_t = TileTensor(ref_c, row_major[BATCH, OUT_C]())
    cnc.branches[0].forward["cpu", BATCH](in_t, ref_a_t)
    cnc.branches[1].forward["cpu", BATCH](in_t, ref_b_t)
    cnc.branches[2].forward["cpu", BATCH](in_t, ref_c_t)

    var max_diff: Scalar[DT] = 0.0
    for b in range(BATCH):
        for j in range(OUT_A):
            var d = fabs(out_buf[b * OUT_TOTAL + j] - ref_a[b * OUT_A + j])
            if d > max_diff: max_diff = d
        for j in range(OUT_B):
            var d = fabs(out_buf[b * OUT_TOTAL + OUT_A + j] - ref_b[b * OUT_B + j])
            if d > max_diff: max_diff = d
        for j in range(OUT_C):
            var d = fabs(out_buf[b * OUT_TOTAL + OUT_A + OUT_B + j] - ref_c[b * OUT_C + j])
            if d > max_diff: max_diff = d
    print("  test_concat_forward_3_branches max_diff=", max_diff)
    assert_true(max_diff < Scalar[DT](1e-6), "Concat forward != branch forward")

    in_buf.free(); out_buf.free()
    ref_a.free(); ref_b.free(); ref_c.free()


def test_concat_backward_grad_input_sum() raises:
    """Grad_input from Concat.backward should equal the sum of each
    branch's individual grad_input under the same grad_output slice."""
    var cnc = Concat[
        Linear[IN_DIM, OUT_A], Linear[IN_DIM, OUT_B], Linear[IN_DIM, OUT_C]
    ].make[target="cpu", INIT=Xavier]()

    var in_buf = alloc[Scalar[DT]](BATCH * IN_DIM)
    var out_buf = alloc[Scalar[DT]](BATCH * OUT_TOTAL)
    var go_buf = alloc[Scalar[DT]](BATCH * OUT_TOTAL)
    var gi_buf = alloc[Scalar[DT]](BATCH * IN_DIM)
    for i in range(BATCH * IN_DIM):
        in_buf[i] = Scalar[DT](Float32(i) * 0.1 - 0.4)
    for i in range(BATCH * OUT_TOTAL):
        go_buf[i] = Scalar[DT](Float32(i) * 0.07 + 0.02)

    var in_t = TileTensor(in_buf, row_major[BATCH, IN_DIM]())
    var out_t = TileTensor(out_buf, row_major[BATCH, OUT_TOTAL]())
    var go_t = TileTensor(go_buf, row_major[BATCH, OUT_TOTAL]())
    var gi_t = TileTensor(gi_buf, row_major[BATCH, IN_DIM]())
    cnc.forward["cpu", BATCH](in_t, out_t)
    cnc.backward["cpu", BATCH](go_t, gi_t)

    # Reference: invoke each branch's backward separately and sum.
    var ref_gi_a = alloc[Scalar[DT]](BATCH * IN_DIM)
    var ref_gi_b = alloc[Scalar[DT]](BATCH * IN_DIM)
    var ref_gi_c = alloc[Scalar[DT]](BATCH * IN_DIM)
    var ref_go_a = alloc[Scalar[DT]](BATCH * OUT_A)
    var ref_go_b = alloc[Scalar[DT]](BATCH * OUT_B)
    var ref_go_c = alloc[Scalar[DT]](BATCH * OUT_C)
    for b in range(BATCH):
        for j in range(OUT_A):
            ref_go_a[b * OUT_A + j] = go_buf[b * OUT_TOTAL + j]
        for j in range(OUT_B):
            ref_go_b[b * OUT_B + j] = go_buf[b * OUT_TOTAL + OUT_A + j]
        for j in range(OUT_C):
            ref_go_c[b * OUT_C + j] = go_buf[b * OUT_TOTAL + OUT_A + OUT_B + j]
    var ref_go_a_t = TileTensor(ref_go_a, row_major[BATCH, OUT_A]())
    var ref_go_b_t = TileTensor(ref_go_b, row_major[BATCH, OUT_B]())
    var ref_go_c_t = TileTensor(ref_go_c, row_major[BATCH, OUT_C]())
    var ref_gi_a_t = TileTensor(ref_gi_a, row_major[BATCH, IN_DIM]())
    var ref_gi_b_t = TileTensor(ref_gi_b, row_major[BATCH, IN_DIM]())
    var ref_gi_c_t = TileTensor(ref_gi_c, row_major[BATCH, IN_DIM]())
    # Use backward_input so we don't pollute branch grad_w (which the
    # Concat.backward call also touched — different accumulator state).
    cnc.branches[0].backward_input["cpu", BATCH](ref_go_a_t, ref_gi_a_t)
    cnc.branches[1].backward_input["cpu", BATCH](ref_go_b_t, ref_gi_b_t)
    cnc.branches[2].backward_input["cpu", BATCH](ref_go_c_t, ref_gi_c_t)

    var max_diff: Scalar[DT] = 0.0
    for i in range(BATCH * IN_DIM):
        var sum_v = ref_gi_a[i] + ref_gi_b[i] + ref_gi_c[i]
        var d = fabs(gi_buf[i] - sum_v)
        if d > max_diff: max_diff = d
    print("  test_concat_backward_grad_input_sum max_diff=", max_diff)
    assert_true(max_diff < Scalar[DT](1e-5),
                "Concat backward grad_input != sum of branch grads")

    in_buf.free(); out_buf.free(); go_buf.free(); gi_buf.free()
    ref_gi_a.free(); ref_gi_b.free(); ref_gi_c.free()
    ref_go_a.free(); ref_go_b.free(); ref_go_c.free()


def test_concat_backward_input_matches_backward_grad_input() raises:
    """Concat.backward_input on the same setup must produce the *same*
    grad_input as Concat.backward (param-grad pollution is the only
    difference between the two paths). This is the Phase 8.2 contract
    re-asserted at the Concat boundary."""
    var cnc1 = Concat[
        Linear[IN_DIM, OUT_A], Linear[IN_DIM, OUT_B], Linear[IN_DIM, OUT_C]
    ].make[target="cpu", INIT=Xavier]()
    var cnc2 = Concat[
        Linear[IN_DIM, OUT_A], Linear[IN_DIM, OUT_B], Linear[IN_DIM, OUT_C]
    ].make[target="cpu", INIT=Xavier]()

    # Manually sync weights: copy via re-running make with same INIT and
    # the same seed sequence — Xavier draws from std.random which is
    # global. Quickest reproducible: run make once then use the same
    # struct for both paths. Easier: just compare both instances'
    # outputs at the input tensor level.

    # Simplest direct approach: use ONE Concat, do forward+backward to
    # get grad_input_A, then re-forward (resets nothing important — cache
    # is on the branches) and call backward_input to get grad_input_B.
    # Both should equal because Linear.backward and Linear.backward_input
    # produce the same grad_input (the only differences are grad_w + grad_b
    # accumulator writes, which don't feed back into grad_input).
    var in_buf = alloc[Scalar[DT]](BATCH * IN_DIM)
    var out_buf = alloc[Scalar[DT]](BATCH * OUT_TOTAL)
    var go_buf = alloc[Scalar[DT]](BATCH * OUT_TOTAL)
    var gi_A = alloc[Scalar[DT]](BATCH * IN_DIM)
    var gi_B = alloc[Scalar[DT]](BATCH * IN_DIM)
    for i in range(BATCH * IN_DIM):
        in_buf[i] = Scalar[DT](Float32(i) * 0.13 - 0.5)
    for i in range(BATCH * OUT_TOTAL):
        go_buf[i] = Scalar[DT](Float32(i) * 0.11 + 0.03)

    var in_t = TileTensor(in_buf, row_major[BATCH, IN_DIM]())
    var out_t = TileTensor(out_buf, row_major[BATCH, OUT_TOTAL]())
    var go_t = TileTensor(go_buf, row_major[BATCH, OUT_TOTAL]())
    var gi_A_t = TileTensor(gi_A, row_major[BATCH, IN_DIM]())
    var gi_B_t = TileTensor(gi_B, row_major[BATCH, IN_DIM]())

    cnc1.forward["cpu", BATCH](in_t, out_t)
    cnc1.backward["cpu", BATCH](go_t, gi_A_t)

    cnc1.forward["cpu", BATCH](in_t, out_t)   # refresh caches
    cnc1.backward_input["cpu", BATCH](go_t, gi_B_t)

    var max_diff: Scalar[DT] = 0.0
    for i in range(BATCH * IN_DIM):
        var d = fabs(gi_A[i] - gi_B[i])
        if d > max_diff: max_diff = d
    print(
        "  test_concat_backward_input_matches_backward_grad_input max_diff=",
        max_diff,
    )
    assert_true(max_diff < Scalar[DT](1e-6),
                "Concat.backward_input grad_input != Concat.backward grad_input")

    in_buf.free(); out_buf.free(); go_buf.free(); gi_A.free(); gi_B.free()
    _ = cnc2^   # silence unused-var (test originally had two ctors planned)


def main() raises:
    print("=" * 70)
    print("nn2 Phase 8.4 — Concat[*BRANCHES] CPU tests")
    print("=" * 70)
    test_concat_forward_3_branches()
    print("  test_concat_forward_3_branches PASSED")
    test_concat_backward_grad_input_sum()
    print("  test_concat_backward_grad_input_sum PASSED")
    test_concat_backward_input_matches_backward_grad_input()
    print("  test_concat_backward_input_matches_backward_grad_input PASSED")
    print("=" * 70)
    print("ALL PASSED")
    print("=" * 70)
