"""ScaledDotProductAttention causality test.

A causal attention's output at query position i must depend ONLY on K/V at
positions ≤ i (and Q at i). If perturbing a FUTURE token's K/V (position p)
changes an EARLIER output (i < p), causality is violated — a leak that makes
teacher-forced next-token prediction trivially easy (val loss collapses) while
autoregressive generation degenerates. This is the prime suspect for nn2 GPT's
val 0.48 / 86% top-1 / garbage-generation triad.

Input layout per sample: [Q(seq·dim) | K(seq·dim) | V(seq·dim)].
We perturb K and V at position p = SEQ-1 and assert outputs at i < p are
unchanged (bit-identical). CPU path; causal=True.
"""

from std.memory import alloc
from std.math import abs
from std.testing import assert_true
from layout import TileTensor, row_major

from mojo_rl.nn2.constants import DT
from mojo_rl.nn2.initializer import Zero
from mojo_rl.nn2.primitives.attention import ScaledDotProductAttention


def _alloc(n: Int) -> UnsafePointer[Scalar[DT], MutAnyOrigin]:
    return rebind[UnsafePointer[Scalar[DT], MutAnyOrigin]](
        alloc[Scalar[DT]](n)
    )


def main() raises:
    print("=" * 70)
    print("ScaledDotProductAttention causality (perturb future K/V)")
    print("=" * 70)
    comptime DIM = 4
    comptime N_HEADS = 2
    comptime SEQ = 5
    comptime BATCH = 1
    comptime IN_N = BATCH * SEQ * DIM * 3
    comptime OUT_N = BATCH * SEQ * DIM
    comptime KOFF = SEQ * DIM
    comptime VOFF = 2 * SEQ * DIM

    var op = ScaledDotProductAttention[
        DIM, N_HEADS, SEQ, True
    ].make[target="cpu", INIT=Zero]()

    var x = _alloc(IN_N)
    var y1 = _alloc(OUT_N)
    var y2 = _alloc(OUT_N)
    for i in range(IN_N):
        x[i] = Scalar[DT](0.31 * Float64(i % 7) - 1.0 + 0.05 * Float64(i))

    var x_t = TileTensor(x, row_major[BATCH, SEQ * DIM * 3]())
    var y1_t = TileTensor(y1, row_major[BATCH, SEQ * DIM]())
    op.forward["cpu", BATCH](x_t, output=y1_t)

    # Perturb K and V at the LAST position p = SEQ-1 (future for all i < SEQ-1).
    comptime p = SEQ - 1
    for d in range(DIM):
        x[KOFF + p * DIM + d] = x[KOFF + p * DIM + d] + Scalar[DT](3.3)
        x[VOFF + p * DIM + d] = x[VOFF + p * DIM + d] - Scalar[DT](2.7)
    var y2_t = TileTensor(y2, row_major[BATCH, SEQ * DIM]())
    op.forward["cpu", BATCH](x_t, output=y2_t)

    # Outputs at i < p must be bit-identical; output at p may change.
    var max_earlier: Float64 = 0.0
    for i in range(p):  # i = 0 .. p-1
        for d in range(DIM):
            var diff = abs(Float64(y1[i * DIM + d]) - Float64(y2[i * DIM + d]))
            if diff > max_earlier:
                max_earlier = diff
    var changed_last: Float64 = 0.0
    for d in range(DIM):
        var diff = abs(
            Float64(y1[p * DIM + d]) - Float64(y2[p * DIM + d])
        )
        if diff > changed_last:
            changed_last = diff

    print("   max change at earlier positions i<", p, " =", max_earlier)
    print("   change at position", p, " (expected > 0) =", changed_last)
    assert_true(
        max_earlier == 0.0,
        "CAUSALITY LEAK: future K/V at pos "
        + String(p) + " changed an earlier output",
    )
    assert_true(
        changed_last > 0.0,
        "sanity: perturbing pos p's own K/V should change output[p]",
    )
    print("  ok — causal (no future leak)")
    print("=" * 70)
    print("ALL PASSED")
    print("=" * 70)
