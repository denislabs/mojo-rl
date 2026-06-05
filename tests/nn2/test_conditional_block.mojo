"""ConditionalTransformerBlock — Phase B (CPU).

1. Zero-init identity: with AdaLN-zero, block(x, c) == x **bitwise**.
2. After randomizing internal params, fd-gradcheck on x and c (validates
   the whole internal graph forward+backward through the Module wrapper).
"""

from std.memory import alloc
from std.gpu.memory import AddressSpace
from std.testing import assert_true
from layout import TileTensor, row_major

from mojo_rl.nn2.constants import DT
from mojo_rl.nn2.initializer import Kaiming
from mojo_rl.nn2.core import ParamVisitor
from mojo_rl.nn2.primitives.conditional_transformer_block import (
    ConditionalTransformerBlock,
)


comptime EMB = 4
comptime HEADS = 2
comptime H = 3
comptime FF = 8
comptime BATCH = 2
comptime SEQ = H * EMB
comptime N = BATCH * SEQ


def _a(n: Int) -> UnsafePointer[Scalar[DT], MutAnyOrigin]:
    return alloc[Scalar[DT]](n)


def _det(i: Int, scale: Float64) -> Scalar[DT]:
    var v = (Float64((i * 2654435761) % 1000) / 500.0) - 1.0
    return Scalar[DT](v * scale)


# Fills every parameter with small deterministic values so the block is
# non-trivial (zero-init params would make the gradcheck degenerate).
struct FillParams(ParamVisitor):
    var idx: Int

    def __init__(out self):
        self.idx = 0

    def visit(
        mut self, name: String,
        param: TileTensor[
            dtype=DT, address_space=AddressSpace.GENERIC, element_size=1, ...
        ],
        grad: TileTensor[
            dtype=DT, address_space=AddressSpace.GENERIC, element_size=1, ...
        ],
        n_elems: Int, apply_decay: Bool,
    ) raises:
        var p = rebind[UnsafePointer[Scalar[DT], MutAnyOrigin]](param.ptr)
        for i in range(n_elems):
            p[i] = _det(self.idx + i + 1, 0.3)
        self.idx += n_elems


def test_identity_at_init() raises:
    print("test_identity_at_init ...")
    var blk = ConditionalTransformerBlock[
        EMB, HEADS, H, FF
    ].make[target="cpu", INIT=Kaiming]()

    var x = _a(N); var c = _a(N); var y = _a(N)
    for k in range(N):
        x[k] = _det(k + 1, 1.0)
        c[k] = _det(k + 50, 1.0)
    var x_t = TileTensor(x, row_major[BATCH, SEQ]())
    var c_t = TileTensor(c, row_major[BATCH, SEQ]())
    var y_t = TileTensor(y, row_major[BATCH, SEQ]())
    blk.forward["cpu", BATCH](x_t, c_t, output=y_t)

    var maxd: Scalar[DT] = 0.0
    for k in range(N):
        var d = (y[k] - x[k]).__abs__()
        if d > maxd:
            maxd = d
    print("   max|block(x,c) - x| =", maxd)
    assert_true(maxd == Scalar[DT](0.0),
                "AdaLN-zero block must be bitwise identity at init")
    x.free(); c.free(); y.free()
    _ = blk^
    print("  ok")


def test_gradcheck_randomized() raises:
    print("test_gradcheck_randomized ...")
    var blk = ConditionalTransformerBlock[
        EMB, HEADS, H, FF
    ].make[target="cpu", INIT=Kaiming]()
    var filler = FillParams()
    blk.for_each_param["cpu", FillParams]("blk", filler)
    print("   randomized", filler.idx, "param elements")

    var x = _a(N); var c = _a(N); var y = _a(N); var w = _a(N)
    var gx = _a(N); var gc = _a(N)
    for k in range(N):
        x[k] = _det(k + 1, 1.0)
        c[k] = _det(k + 50, 1.0)
        w[k] = _det(k + 99, 1.0)
    var x_t = TileTensor(x, row_major[BATCH, SEQ]())
    var c_t = TileTensor(c, row_major[BATCH, SEQ]())
    var y_t = TileTensor(y, row_major[BATCH, SEQ]())
    blk.forward["cpu", BATCH](x_t, c_t, output=y_t)

    var w_t = TileTensor(w, row_major[BATCH, SEQ]())
    var gx_t = TileTensor(gx, row_major[BATCH, SEQ]())
    var gc_t = TileTensor(gc, row_major[BATCH, SEQ]())
    blk.vjp["cpu", BATCH](w_t, gx_t, gc_t)

    comptime EPS = Scalar[DT](1e-3)
    var bad = 0
    for which in range(2):
        var p = x if which == 0 else c
        var ga = gx if which == 0 else gc
        for k in range(N):
            var saved = p[k]
            p[k] = saved + EPS
            blk.forward["cpu", BATCH](x_t, c_t, output=y_t)
            var lp: Scalar[DT] = 0.0
            for j in range(N):
                lp += w[j] * y[j]
            p[k] = saved - EPS
            blk.forward["cpu", BATCH](x_t, c_t, output=y_t)
            var lm: Scalar[DT] = 0.0
            for j in range(N):
                lm += w[j] * y[j]
            p[k] = saved
            var num = (lp - lm) / (Scalar[DT](2.0) * EPS)
            var ad = (ga[k] - num).__abs__()
            var ok = ad < Scalar[DT](5e-4) or (
                ad / (ga[k].__abs__() + num.__abs__() + Scalar[DT](1e-4))
            ) < Scalar[DT](3e-2)
            if not ok:
                bad += 1
                if bad <= 3:
                    print("   mismatch which=", which, "k=", k,
                          "analytic=", ga[k], "num=", num)
    assert_true(bad == 0, "ConditionalTransformerBlock gradcheck failed")
    x.free(); c.free(); y.free(); w.free(); gx.free(); gc.free()
    _ = blk^
    print("  ok")


def main() raises:
    print("=" * 70)
    print("ConditionalTransformerBlock (Phase B, CPU)")
    print("=" * 70)
    test_identity_at_init()
    test_gradcheck_randomized()
    print("=" * 70)
    print("ALL PASSED")
    print("=" * 70)
