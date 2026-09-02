"""Segmented Cholesky == dense Cholesky, BIT FOR BIT. PN2b's gate.

`chol_factor_seg` / `chol_solve_seg` factor and solve one diagonal sub-block
`[s0, s1)` of an `nv x nv` system. PN2 runs them once per segment of
`build_dof_segments`, so that `H = M + sum D*J^T J` — which is block-diagonal
over the kinematic trees, merged by whichever trees a constraint row couples —
costs `sum(size^3)` instead of `nv^3`. P0 measured the dense factorisation at
70% of GPU time on `so101_park_k9`.

⚠⚠ THE CLAIM IS BIT-EXACTNESS, SO THE TEST IS `!=` AND NOT A TOLERANCE. Every
entry a segment skips is `L[i*nv+k] * L[j*nv+k]` with `k` outside the block,
where `L` is exactly `0` — zeroed once by the caller and never written, since
no segment owns those columns. Dropping exact zeros from a sequential
accumulation returns the identical bit pattern. If that argument were wrong the
error would be ~1e-16 and a tolerance would hide it.

FOUR ARMS:
  A  the delegation: `chol_factor_inline` == `chol_factor_seg` over `[0, nv)`,
     and likewise for solve. Pins the refactor itself.
  B  a genuinely block-diagonal `H`: per-segment == dense, bit for bit, for
     both the factor and the solve.
  C  ⚠ THE NEGATIVE CONTROL. The same `H` with the off-block corner FILLED must
     make the two DISAGREE. Without it, arm B passes on a harness where
     segmentation does nothing — "identical" over a comparison that cannot
     differ looks exactly like a pass.
  D  ⚠ THE SECOND NEGATIVE CONTROL, on the other side: segmenting a matrix that
     really is block-diagonal must still SOLVE it — `H*x == b` — so arm B
     cannot pass by both paths being equally wrong.

Run: pixi run mojo run -I . tests/physics3d/test_cholesky_segmented.mojo
"""

from std.math import sqrt, abs
from mojo_rl.physics3d.fields.scratch import Scratch
from mojo_rl.physics3d.solver.cholesky import (
    chol_factor_inline, chol_factor_seg,
    chol_solve_inline, chol_solve_seg,
)

comptime DT = DType.float64
comptime NV = 12
comptime SPLIT = 5            # blocks [0,5) and [5,12) — deliberately uneven
comptime M_CAP = NV * NV
comptime V_CAP = NV


struct Tally:
    var checks: Int
    var fails: Int

    def __init__(out self):
        self.checks = 0
        self.fails = 0

    def truth(mut self, ok: Bool, msg: String):
        self.checks += 1
        if ok:
            print("  ok:", msg)
        else:
            self.fails += 1
            print("  FAIL:", msg)


def _spd(mut H: Scratch[Scalar[DT], M_CAP], coupled: Bool):
    """A symmetric positive-definite `H`, block-diagonal over [0,SPLIT) and
    [SPLIT,NV) unless `coupled`, in which case one off-block pair is filled."""
    for i in range(NV * NV):
        H[i] = Scalar[DT](0)
    for i in range(NV):
        for j in range(NV):
            var same = (i < SPLIT) == (j < SPLIT)
            if not same:
                continue
            # A deterministic, well-conditioned SPD pattern.
            var v = Float64(1) / Float64(1 + (i - j) * (i - j))
            H[i * NV + j] = Scalar[DT](v)
        H[i * NV + i] = Scalar[DT](4.0 + 0.25 * Float64(i))
    if coupled:
        # ⚠ SYMMETRIC, and small enough to keep H positive definite. The point
        # is only that the block structure is now a LIE.
        H[2 * NV + 9] = Scalar[DT](0.75)
        H[9 * NV + 2] = Scalar[DT](0.75)


def _zero(mut L: Scratch[Scalar[DT], M_CAP]):
    for i in range(NV * NV):
        L[i] = Scalar[DT](0)


def _bits_differ(
    a: Scratch[Scalar[DT], M_CAP], b: Scratch[Scalar[DT], M_CAP], n: Int
) -> Int:
    var d = 0
    for i in range(n):
        if a[i] != b[i]:
            d += 1
    return d


def _vbits_differ(
    a: Scratch[Scalar[DT], V_CAP], b: Scratch[Scalar[DT], V_CAP], n: Int
) -> Int:
    var d = 0
    for i in range(n):
        if a[i] != b[i]:
            d += 1
    return d


def main() raises:
    var t = Tally()
    print("=== segmented Cholesky vs dense, bit for bit (PN2b) ===")

    var H = Scratch[Scalar[DT], M_CAP](NV * NV, uninitialized=Scalar[DT](0))
    var b = Scratch[Scalar[DT], V_CAP](NV, uninitialized=Scalar[DT](0))
    for i in range(NV):
        b[i] = Scalar[DT](1.0 + 0.5 * Float64(i))

    # ── A: the delegation is the same code ────────────────────────────────
    print("--- A: chol_factor_inline == chol_factor_seg[0, nv) ---")
    _spd(H, False)
    var La = Scratch[Scalar[DT], M_CAP](NV * NV, uninitialized=Scalar[DT](0))
    var Lb = Scratch[Scalar[DT], M_CAP](NV * NV, uninitialized=Scalar[DT](0))
    var oka = chol_factor_inline[DT, M_CAP](H, La, NV)
    _zero(Lb)
    var okb = chol_factor_seg[DT, M_CAP](H, Lb, NV, 0, NV)
    t.truth(oka == okb, String("rank flag agrees (", oka, ")"))
    t.truth(_bits_differ(La, Lb, NV * NV) == 0,
            String("L identical over ", NV * NV, " entries (",
                   _bits_differ(La, Lb, NV * NV), " differ)"))
    var xa = Scratch[Scalar[DT], V_CAP](NV, uninitialized=Scalar[DT](0))
    var xb = Scratch[Scalar[DT], V_CAP](NV, uninitialized=Scalar[DT](0))
    chol_solve_inline[DT, M_CAP, V_CAP](La, b, xa, NV)
    chol_solve_seg[DT, M_CAP, V_CAP](Lb, b, xb, NV, 0, NV)
    t.truth(_vbits_differ(xa, xb, NV) == 0,
            String("x identical over ", NV, " entries (",
                   _vbits_differ(xa, xb, NV), " differ)"))

    # ── B: block-diagonal H — per-segment == dense ───────────────────────
    print("--- B: block-diagonal H, two segments vs one dense factor ---")
    _spd(H, False)
    var Ld = Scratch[Scalar[DT], M_CAP](NV * NV, uninitialized=Scalar[DT](0))
    _ = chol_factor_inline[DT, M_CAP](H, Ld, NV)
    var Ls = Scratch[Scalar[DT], M_CAP](NV * NV, uninitialized=Scalar[DT](0))
    _zero(Ls)
    _ = chol_factor_seg[DT, M_CAP](H, Ls, NV, 0, SPLIT)
    _ = chol_factor_seg[DT, M_CAP](H, Ls, NV, SPLIT, NV)
    var dif = _bits_differ(Ld, Ls, NV * NV)
    t.truth(dif == 0,
            String("L identical over ", NV * NV, " entries (", dif,
                   " differ) — the skipped terms really were exact zeros"))
    var xd = Scratch[Scalar[DT], V_CAP](NV, uninitialized=Scalar[DT](0))
    var xs = Scratch[Scalar[DT], V_CAP](NV, uninitialized=Scalar[DT](0))
    chol_solve_inline[DT, M_CAP, V_CAP](Ld, b, xd, NV)
    chol_solve_seg[DT, M_CAP, V_CAP](Ls, b, xs, NV, 0, SPLIT)
    chol_solve_seg[DT, M_CAP, V_CAP](Ls, b, xs, NV, SPLIT, NV)
    t.truth(_vbits_differ(xd, xs, NV) == 0,
            String("x identical over ", NV, " entries (",
                   _vbits_differ(xd, xs, NV), " differ)"))

    # ── C: the negative control — a COUPLED H must disagree ──────────────
    print("--- C: coupled H — the two MUST differ (negative control) ---")
    _spd(H, True)
    var Lcd = Scratch[Scalar[DT], M_CAP](NV * NV, uninitialized=Scalar[DT](0))
    _ = chol_factor_inline[DT, M_CAP](H, Lcd, NV)
    var Lcs = Scratch[Scalar[DT], M_CAP](NV * NV, uninitialized=Scalar[DT](0))
    _zero(Lcs)
    _ = chol_factor_seg[DT, M_CAP](H, Lcs, NV, 0, SPLIT)
    _ = chol_factor_seg[DT, M_CAP](H, Lcs, NV, SPLIT, NV)
    var cdif = _bits_differ(Lcd, Lcs, NV * NV)
    t.truth(cdif > 0,
            String("L differs in ", cdif, " entries — segmentation is not a"
                   " no-op, and arm B's agreement means something"))

    # ── D: the second control — the segmented solve is a real solve ──────
    print("--- D: H*x == b on the block-diagonal case ---")
    _spd(H, False)
    var worst = Float64(0)
    for i in range(NV):
        var r = Float64(0)
        for j in range(NV):
            r += Float64(H[i * NV + j]) * Float64(xs[j])
        var e = abs(r - Float64(b[i]))
        if e > worst:
            worst = e
    t.truth(worst < 1e-12,
            String("worst |H*x - b| = ", worst, " (segmented solve is a"
                   " solve, not just equal to the dense one)"))

    print("===", t.checks - t.fails, "/", t.checks, "passed ===")
    if t.fails != 0:
        raise Error("test_cholesky_segmented: " + String(t.fails) + " failed")
