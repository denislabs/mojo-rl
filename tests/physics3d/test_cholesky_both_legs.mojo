"""`cholesky` gives the same factorization on the stack leg and the heap leg.

WHY THIS TEST EXISTS
====================

2b.2 turns cholesky's `NV` — a comptime parameter every call site bound to
`D.CAP_NV` — into a runtime `nv`, and its `M_SIZE`/`V_SIZE` into `M_CAP`/
`V_CAP` that size a `Scratch` and nothing else.

⚠ NO GATE IN THE TREE TODAY CAN SEE A MISTAKE IN THAT SPLIT. On the static
leg `CAP_NV == NV`, so a cap left behind where a stride was meant is *the same
integer*, and the 124-file suite would stay green. The mistake only appears
when the two differ, and the only place they differ is the dynamic leg — which
until now nothing but `test_dyn_dims_ldl` exercised at all.

So this gate runs the SAME source lines twice:

    static   M_CAP = NV*NV, V_CAP = NV   ->  Scratch picks InlineArray
    dynamic  M_CAP = 0,     V_CAP = 0    ->  Scratch picks List

and requires them to agree. A cap used as a stride collapses rows on the
dynamic arm (`i * 0 + j`) and a cap used as a loop bound iterates zero times,
so either one shows up here as a numerical disagreement rather than as
silence.

⚠ (C) IS THE CLAIM THAT CANNOT PASS BY ACCIDENT — the same argument
`test_dyn_dims_ldl` makes. Claims (A) "the legs agree" and (B) "the heap leg
compiles" would BOTH hold if the compiler quietly specialised the dynamic arm
on a constant. So the dynamic arm below runs nv=6 and nv=11 through ONE
`M_CAP=0` instantiation, and their answers must differ from each other while
each matches its own static counterpart.

Tolerance: f64, and the two arms run identical source, so agreement should be
near-exact. The gate is 1e-12 and the worst error is PRINTED — a hard zero
across every shape would suggest the arms are not actually distinct.

Run: pixi run mojo run -I . tests/physics3d/test_cholesky_both_legs.mojo
"""

from std.math import sqrt
from mojo_rl.physics3d.fields.scratch import Scratch
from mojo_rl.physics3d.solver.cholesky import (
    chol_factor_inline,
    chol_solve_inline,
    chol_rank1_update,
)

comptime DT = DType.float64


struct Tally(Movable):
    var checks: Int
    var fails: Int

    def __init__(out self):
        self.checks = 0
        self.fails = 0

    def truth(mut self, ok: Bool, what: String):
        self.checks += 1
        if not ok:
            self.fails += 1
            print("  FAIL", what)

    def close(mut self, got: Float64, want: Float64, tol: Float64, what: String):
        self.checks += 1
        var e = got - want
        if e < 0:
            e = -e
        if not (e <= tol):
            self.fails += 1
            print("  FAIL", what, "got", got, "want", want, "err", e)


def spd(nv: Int, i: Int, j: Int) -> Float64:
    """A deterministic SPD matrix: diagonally dominant, size-dependent.

    Size-dependent on purpose — if H did not vary with nv, the (C) check
    below could pass with the two dynamic runs agreeing for the wrong reason.
    """
    if i == j:
        return Float64(nv) * 4.0 + Float64(i) * 0.5 + 3.0
    var d = Float64(i - j)
    if d < 0:
        d = -d
    return 1.0 / (1.0 + d) * 0.75


def rhs(nv: Int, i: Int) -> Float64:
    return 1.0 + Float64(i) * 0.25 - Float64(nv) * 0.05


def solve[M_CAP: Int, V_CAP: Int](nv: Int) -> List[Float64]:
    """Factor and solve H x = b. ONE body; the caps choose the container."""
    var H = Scratch[Scalar[DT], M_CAP](nv * nv, uninitialized=Scalar[DT](0))
    var L = Scratch[Scalar[DT], M_CAP](nv * nv, uninitialized=Scalar[DT](0))
    for i in range(nv):
        for j in range(nv):
            H[i * nv + j] = Scalar[DT](spd(nv, i, j))

    var ok = chol_factor_inline[DT, M_CAP](H, L, nv)
    if not ok:
        print("  !! chol_factor_inline reported rank-deficient at nv =", nv)

    var b = Scratch[Scalar[DT], V_CAP](nv, uninitialized=Scalar[DT](0))
    var x = Scratch[Scalar[DT], V_CAP](nv, uninitialized=Scalar[DT](0))
    for i in range(nv):
        b[i] = Scalar[DT](rhs(nv, i))
    chol_solve_inline[DT, M_CAP, V_CAP](L, b, x, nv)

    # Return L and x together so a stride bug in EITHER routine is visible.
    var out = List[Float64]()
    for i in range(nv * nv):
        out.append(Float64(L[i]))
    for i in range(nv):
        out.append(Float64(x[i]))
    return out^


def rank1[M_CAP: Int, V_CAP: Int](nv: Int) -> List[Float64]:
    """`chol_rank1_update` on one leg; same split of cap vs stride."""
    var L = Scratch[Scalar[DT], M_CAP](nv * nv, uninitialized=Scalar[DT](0))
    var H = Scratch[Scalar[DT], M_CAP](nv * nv, uninitialized=Scalar[DT](0))
    for i in range(nv):
        for j in range(nv):
            H[i * nv + j] = Scalar[DT](spd(nv, i, j))
    _ = chol_factor_inline[DT, M_CAP](H, L, nv)

    var v = Scratch[Scalar[DT], V_CAP](nv, uninitialized=Scalar[DT](0))
    for i in range(nv):
        v[i] = Scalar[DT](0.1 + Float64(i) * 0.05)
    chol_rank1_update[DT, M_CAP, V_CAP](L, v, Scalar[DT](1.0), nv)

    var out = List[Float64]()
    for i in range(nv * nv):
        out.append(Float64(L[i]))
    return out^


def worst(a: List[Float64], b: List[Float64]) -> Float64:
    if len(a) != len(b):
        print("  !! length mismatch", len(a), len(b))
        return 1e30
    var w = 0.0
    for i in range(len(a)):
        var e = a[i] - b[i]
        if e < 0:
            e = -e
        if e > w:
            w = e
    return w


def residual(x: List[Float64], nv: Int, off: Int) -> Float64:
    """Max |H x - b|, computed from `spd`/`rhs` directly.

    An independent check: if BOTH legs shared a wrong stride they would agree
    with each other and this would still catch it.
    """
    var w = 0.0
    for i in range(nv):
        var s = 0.0
        for j in range(nv):
            s += spd(nv, i, j) * x[off + j]
        var e = s - rhs(nv, i)
        if e < 0:
            e = -e
        if e > w:
            w = e
    return w


def main():
    print("=== cholesky: static leg vs heap leg ===")
    var t = Tally()
    var tol = 1e-12

    # ---- VACUITY GUARD: the two arms really are two different containers --
    # Everything below compares a "static" arm against a "dynamic" one, and
    # the headline result is an exact 0.0. That is also what a test would
    # print if BOTH arms were the same leg. `STATIC` is the flag that picks
    # the container, so assert it rather than infer it from the agreement.
    t.truth(Scratch[Scalar[DT], 36].STATIC, "cap 36 selects the stack leg")
    t.truth(not Scratch[Scalar[DT], 0].STATIC, "cap 0 selects the heap leg")

    # ---- (A) the two legs agree, across a spread of sizes ----------------
    # Each static arm needs its own comptime caps; the dynamic arm reuses ONE.
    var worst_all = 0.0

    var s6 = solve[6 * 6, 6](6)
    var d6 = solve[0, 0](6)
    var w6 = worst(s6, d6)
    t.close(w6, 0.0, tol, "nv=6 legs agree")
    if w6 > worst_all:
        worst_all = w6

    var s11 = solve[11 * 11, 11](11)
    var d11 = solve[0, 0](11)
    var w11 = worst(s11, d11)
    t.close(w11, 0.0, tol, "nv=11 legs agree")
    if w11 > worst_all:
        worst_all = w11

    var s23 = solve[23 * 23, 23](23)
    var d23 = solve[0, 0](23)
    var w23 = worst(s23, d23)
    t.close(w23, 0.0, tol, "nv=23 legs agree")
    if w23 > worst_all:
        worst_all = w23

    var sr = rank1[9 * 9, 9](9)
    var dr = rank1[0, 0](9)
    var wr = worst(sr, dr)
    t.close(wr, 0.0, tol, "nv=9 rank1 legs agree")
    if wr > worst_all:
        worst_all = wr

    print("  worst |static - dynamic| =", worst_all)

    # ---- the answers are right, not merely equal -------------------------
    # Two identical wrong strides agree with each other. Solve residual is
    # computed from `spd`/`rhs` with independent index arithmetic.
    t.close(residual(s6, 6, 6 * 6), 0.0, 1e-10, "nv=6 static residual")
    t.close(residual(d6, 6, 6 * 6), 0.0, 1e-10, "nv=6 dynamic residual")
    t.close(residual(s23, 23, 23 * 23), 0.0, 1e-10, "nv=23 static residual")
    t.close(residual(d23, 23, 23 * 23), 0.0, 1e-10, "nv=23 dynamic residual")

    # ---- (C) ONE dynamic instantiation serves DIFFERENT sizes ------------
    # This is the claim that cannot be faked by the compiler specialising the
    # "dynamic" arm on a constant: if it had, these would not differ.
    t.truth(len(d6) != len(d11), "one M_CAP=0 body returns different sizes")
    t.truth(len(d6) == 6 * 6 + 6, "dynamic nv=6 produced a full nv*nv + nv")
    t.truth(len(d23) == 23 * 23 + 23, "dynamic nv=23 produced a full nv*nv + nv")
    var differ = False
    for i in range(6 * 6):
        if d6[i] != d11[i]:
            differ = True
    t.truth(differ, "dynamic nv=6 and nv=11 give DIFFERENT factors")

    # ---- NEGATIVE CONTROL -------------------------------------------------
    # Every assertion above is "two numbers are close". If `close` were broken
    # the run would report a clean sweep of nothing. Plant the two failures
    # this test exists to catch — a collapsed stride and a truncated bound —
    # and require both to be caught.
    var probe = Tally()
    # a stride bug: row 1 of the nv=6 factor read as if the stride were 5
    probe.close(s6[1 * 6 + 0], s6[1 * 5 + 0], tol, "planted: stride 6 vs 5")
    # a bound bug: a zero-length answer must not compare equal to a real one
    var empty = List[Float64]()
    probe.close(worst(s6, empty), 0.0, tol, "planted: truncated result")
    if probe.fails != 2:
        print("!! THE CHECKER DOES NOT FAIL ON WRONG INPUT — run is VOID")
        t.fails += 1
    else:
        print("  negative control: 2/2 planted errors caught")

    print("checks:", t.checks, " failures:", t.fails)
    if t.fails == 0:
        print("test_cholesky_both_legs: ALL PASS")
    else:
        print("test_cholesky_both_legs: FAILED")
