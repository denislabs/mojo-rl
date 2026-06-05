"""SIGReg semantics — does the statistic actually penalize collapse?

SIGReg is the JEPA anti-collapse term. Minimizing it should drive the
projected embeddings toward N(0, I). So the FORWARD statistic must score:
  collapsed (var≈0)  >  unit-Gaussian (var≈1)   and   large (var≈4) > unit.
If a collapsed batch scores LOWER than (or equal to) a unit-Gaussian one,
then minimizing SIGReg does NOT oppose collapse — which would explain the
real-Pong λ-sweep (var_min fell as λ rose). This test was the gap: SIGReg
was only fd-gradchecked (gradient ↔ its own forward), never checked for
the actual anti-collapse DIRECTION.

Run:  pixi run mojo run -I . tests/nn2/test_sigreg_semantics.mojo
"""

from std.memory import alloc
from std.math import sqrt, log, cos, sin, pi
from std.testing import assert_true
from layout import TileTensor, row_major

from mojo_rl.nn2.constants import DT
from mojo_rl.nn2.initializer import Kaiming
from mojo_rl.nn2.primitives.sigreg import SIGReg


def _a(n: Int) -> UnsafePointer[Scalar[DT], MutAnyOrigin]:
    return alloc[Scalar[DT]](n)


comptime D = 8
comptime T = 2
comptime P = 64
comptime K = 5
comptime B = 64
comptime N = B * T * D


# deterministic approx-N(0,1) via Box–Muller over an LCG
def _fill_gauss(ptr: UnsafePointer[Scalar[DT], MutAnyOrigin], n: Int,
               std: Float64, seed: UInt64):
    var s = seed
    var i = 0
    while i < n:
        s = s * 6364136223846793005 + 1442695040888963407
        var u1 = (Float64((s >> 11) & 0xFFFFFFFFFFFFF) + 1.0) / Float64(
            1 << 52
        )
        s = s * 6364136223846793005 + 1442695040888963407
        var u2 = Float64((s >> 11) & 0xFFFFFFFFFFFFF) / Float64(1 << 52)
        var r = sqrt(-2.0 * log(u1))
        ptr[i] = Scalar[DT](std * r * cos(2.0 * pi * u2))
        if i + 1 < n:
            ptr[i + 1] = Scalar[DT](std * r * sin(2.0 * pi * u2))
        i += 2


def _stat(mut sig: SIGReg[D, T, P, K],
          inp: UnsafePointer[Scalar[DT], MutAnyOrigin]) raises -> Float64:
    var out = _a(B)
    var in_t = TileTensor(inp, row_major[B, T * D]())
    var out_t = TileTensor(out, row_major[B, 1]())
    sig.forward["cpu", B](in_t, output=out_t)
    var v = Float64(out[0])
    out.free()
    return v


def main() raises:
    print("=" * 70)
    print("SIGReg semantics — collapse must score HIGHER than unit-Gaussian")
    print("=" * 70)
    var sig = SIGReg[D, T, P, K].make["cpu", Kaiming]()

    var collapsed = _a(N)
    var unit = _a(N)
    var large = _a(N)
    _fill_gauss(collapsed, N, 0.01, 11)   # var ≈ 1e-4 (collapsed)
    _fill_gauss(unit, N, 1.0, 11)         # var ≈ 1 (target)
    _fill_gauss(large, N, 2.0, 11)        # var ≈ 4 (over-spread)

    # same instance ⇒ same random projection A across the three calls
    var s_collapsed = _stat(sig, collapsed)
    var s_unit = _stat(sig, unit)
    var s_large = _stat(sig, large)

    print("   stat(collapsed var≈0)  =", s_collapsed)
    print("   stat(unit      var≈1)  =", s_unit)
    print("   stat(large     var≈4)  =", s_large)

    assert_true(s_collapsed > s_unit,
                "collapsed must score HIGHER than unit-Gaussian "
                "(else minimizing SIGReg does not oppose collapse)")
    assert_true(s_large > s_unit,
                "over-spread must score higher than unit-Gaussian")

    collapsed.free(); unit.free(); large.free()
    _ = sig^
    print("=" * 70)
    print("ALL PASSED — SIGReg direction is correct")
    print("=" * 70)
