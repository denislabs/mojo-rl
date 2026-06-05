"""Finite-difference gradcheck for the LeWM nn2 primitives (Phase A).

Covers `LayerNormNoAffine`, `Modulate`, `Gate` on CPU:
  - forward correctness vs closed form,
  - vjp vs central finite differences on every input (within 1e-2 rel).

Loss = sum_{b,i} w[b,i]·y[b,i]  ⇒  analytic grad_input = vjp(w);
numeric grad via central difference on each input element.
"""

from std.memory import alloc
from std.testing import assert_true
from layout import TileTensor, row_major

from mojo_rl.nn2.constants import DT
from mojo_rl.nn2.initializer import Kaiming
from mojo_rl.nn2.primitives.layer_norm_no_affine import LayerNormNoAffine
from mojo_rl.nn2.primitives.modulate import Modulate
from mojo_rl.nn2.primitives.gate import Gate


comptime EPS: Scalar[DT] = 1e-3
comptime RTOL: Scalar[DT] = 1e-2


def _a(n: Int) -> UnsafePointer[Scalar[DT], MutAnyOrigin]:
    return alloc[Scalar[DT]](n)


def _det(i: Int, scale: Float64) -> Scalar[DT]:
    # Deterministic pseudo-values in roughly [-1, 1]·scale.
    var v = (Float64((i * 2654435761) % 1000) / 500.0) - 1.0
    return Scalar[DT](v * scale)


comptime ATOL: Scalar[DT] = 3e-4


def _rel_ok(a: Scalar[DT], b: Scalar[DT]) -> Bool:
    # Combined abs+rel: float32 central differences suffer cancellation
    # noise (~L0·eps/2EPS) that swamps a pure relative test on
    # small-magnitude gradients. Absolute floor covers those.
    var ad = (a - b).__abs__()
    if ad < ATOL:
        return True
    var denom = a.__abs__() + b.__abs__() + Scalar[DT](1e-4)
    return (ad / denom) < RTOL


def _wdot(w: UnsafePointer[Scalar[DT], MutAnyOrigin],
          y: UnsafePointer[Scalar[DT], MutAnyOrigin], n: Int) -> Scalar[DT]:
    var s: Scalar[DT] = 0.0
    for j in range(n):
        s += w[j] * y[j]
    return s


# ──────────────────────────────────────────────────────────────────────
# LayerNormNoAffine
# ──────────────────────────────────────────────────────────────────────
def test_layer_norm_no_affine() raises:
    print("test_layer_norm_no_affine ...")
    comptime BATCH = 3
    comptime DIM = 8
    comptime N = BATCH * DIM

    var x = _a(N)
    var y = _a(N)
    var w = _a(N)
    var gx = _a(N)
    for k in range(N):
        x[k] = _det(k + 1, 2.0)
        w[k] = _det(k + 7, 1.0)

    var m = LayerNormNoAffine[DIM].make[target="cpu", INIT=Kaiming]()
    var x_t = TileTensor(x, row_major[BATCH, DIM]())
    var y_t = TileTensor(y, row_major[BATCH, DIM]())
    m.forward["cpu", BATCH](x_t, output=y_t)

    for b in range(BATCH):
        var s: Scalar[DT] = 0.0
        var ss: Scalar[DT] = 0.0
        for i in range(DIM):
            s += y[b * DIM + i]
            ss += y[b * DIM + i] * y[b * DIM + i]
        assert_true(s.__abs__() < Scalar[DT](1e-3), "LNNA row mean ~ 0")
        assert_true(
            (ss / Scalar[DT](DIM) - Scalar[DT](1.0)).__abs__()
            < Scalar[DT](1e-2),
            "LNNA row normalized variance ~ 1",
        )

    var w_t = TileTensor(w, row_major[BATCH, DIM]())
    var gx_t = TileTensor(gx, row_major[BATCH, DIM]())
    m.vjp["cpu", BATCH](w_t, gx_t)

    for k in range(N):
        var saved = x[k]
        x[k] = saved + EPS
        m.forward["cpu", BATCH](x_t, output=y_t)
        var lp = _wdot(w, y, N)
        x[k] = saved - EPS
        m.forward["cpu", BATCH](x_t, output=y_t)
        var lm = _wdot(w, y, N)
        x[k] = saved
        var num = (lp - lm) / (Scalar[DT](2.0) * EPS)
        assert_true(_rel_ok(gx[k], num), "LNNA grad_x fd mismatch")

    x.free(); y.free(); w.free(); gx.free()
    print("  ok")


# ──────────────────────────────────────────────────────────────────────
# Modulate:  y = x*(1+scale) + shift
# ──────────────────────────────────────────────────────────────────────
def test_modulate() raises:
    print("test_modulate ...")
    comptime BATCH = 3
    comptime DIM = 6
    comptime N = BATCH * DIM

    var x = _a(N)
    var sc = _a(N)
    var sh = _a(N)
    var y = _a(N)
    var w = _a(N)
    var gx = _a(N)
    var gs = _a(N)
    var gsh = _a(N)
    for k in range(N):
        x[k] = _det(k + 1, 1.5)
        sc[k] = _det(k + 5, 0.8)
        sh[k] = _det(k + 9, 0.5)
        w[k] = _det(k + 13, 1.0)

    var m = Modulate[DIM].make[target="cpu", INIT=Kaiming]()
    var x_t = TileTensor(x, row_major[BATCH, DIM]())
    var sc_t = TileTensor(sc, row_major[BATCH, DIM]())
    var sh_t = TileTensor(sh, row_major[BATCH, DIM]())
    var y_t = TileTensor(y, row_major[BATCH, DIM]())
    m.forward["cpu", BATCH](x_t, sc_t, sh_t, output=y_t)

    for k in range(N):
        var want = x[k] * (Scalar[DT](1.0) + sc[k]) + sh[k]
        assert_true((y[k] - want).__abs__() < Scalar[DT](1e-5),
                    "Modulate forward")

    var w_t = TileTensor(w, row_major[BATCH, DIM]())
    var gx_t = TileTensor(gx, row_major[BATCH, DIM]())
    var gs_t = TileTensor(gs, row_major[BATCH, DIM]())
    var gsh_t = TileTensor(gsh, row_major[BATCH, DIM]())
    m.vjp["cpu", BATCH](w_t, gx_t, gs_t, gsh_t)

    # fd over x, scale, shift
    for which in range(3):
        var p = x if which == 0 else (sc if which == 1 else sh)
        var ga = gx if which == 0 else (gs if which == 1 else gsh)
        for k in range(N):
            var saved = p[k]
            p[k] = saved + EPS
            m.forward["cpu", BATCH](x_t, sc_t, sh_t, output=y_t)
            var lp = _wdot(w, y, N)
            p[k] = saved - EPS
            m.forward["cpu", BATCH](x_t, sc_t, sh_t, output=y_t)
            var lm = _wdot(w, y, N)
            p[k] = saved
            var num = (lp - lm) / (Scalar[DT](2.0) * EPS)
            assert_true(_rel_ok(ga[k], num), "Modulate grad fd mismatch")

    x.free(); sc.free(); sh.free(); y.free(); w.free()
    gx.free(); gs.free(); gsh.free()
    print("  ok")


# ──────────────────────────────────────────────────────────────────────
# Gate:  y = x + gate*branch
# ──────────────────────────────────────────────────────────────────────
def test_gate() raises:
    print("test_gate ...")
    comptime BATCH = 3
    comptime DIM = 6
    comptime N = BATCH * DIM

    var x = _a(N)
    var g = _a(N)
    var br = _a(N)
    var y = _a(N)
    var w = _a(N)
    var gx = _a(N)
    var gg = _a(N)
    var gbr = _a(N)
    for k in range(N):
        x[k] = _det(k + 1, 1.5)
        g[k] = _det(k + 5, 0.8)
        br[k] = _det(k + 9, 1.2)
        w[k] = _det(k + 13, 1.0)

    var m = Gate[DIM].make[target="cpu", INIT=Kaiming]()
    var x_t = TileTensor(x, row_major[BATCH, DIM]())
    var g_t = TileTensor(g, row_major[BATCH, DIM]())
    var br_t = TileTensor(br, row_major[BATCH, DIM]())
    var y_t = TileTensor(y, row_major[BATCH, DIM]())
    m.forward["cpu", BATCH](x_t, g_t, br_t, output=y_t)

    for k in range(N):
        var want = x[k] + g[k] * br[k]
        assert_true((y[k] - want).__abs__() < Scalar[DT](1e-5),
                    "Gate forward")

    var w_t = TileTensor(w, row_major[BATCH, DIM]())
    var gx_t = TileTensor(gx, row_major[BATCH, DIM]())
    var gg_t = TileTensor(gg, row_major[BATCH, DIM]())
    var gbr_t = TileTensor(gbr, row_major[BATCH, DIM]())
    m.vjp["cpu", BATCH](w_t, gx_t, gg_t, gbr_t)

    for which in range(3):
        var p = x if which == 0 else (g if which == 1 else br)
        var ga = gx if which == 0 else (gg if which == 1 else gbr)
        for k in range(N):
            var saved = p[k]
            p[k] = saved + EPS
            m.forward["cpu", BATCH](x_t, g_t, br_t, output=y_t)
            var lp = _wdot(w, y, N)
            p[k] = saved - EPS
            m.forward["cpu", BATCH](x_t, g_t, br_t, output=y_t)
            var lm = _wdot(w, y, N)
            p[k] = saved
            var num = (lp - lm) / (Scalar[DT](2.0) * EPS)
            assert_true(_rel_ok(ga[k], num), "Gate grad fd mismatch")

    x.free(); g.free(); br.free(); y.free(); w.free()
    gx.free(); gg.free(); gbr.free()
    print("  ok")


def main() raises:
    print("=" * 70)
    print("LeWM nn2 primitives — finite-difference gradcheck (Phase A)")
    print("=" * 70)
    test_layer_norm_no_affine()
    test_modulate()
    test_gate()
    print("=" * 70)
    print("ALL PASSED")
    print("=" * 70)
