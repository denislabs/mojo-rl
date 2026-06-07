"""RepeatConditional[DEPTH, ConditionalTransformerBlock] — Phase B (CPU).

The LeWM AR-predictor stack. Validates:
1. Zero-init: a DEPTH-stack of AdaLN-zero blocks is **bitwise identity**.
2. After randomizing all params, fd-gradcheck on x and c — exercises the
   x-chain + grad_c accumulation across DEPTH layers.
"""

from std.memory import alloc
from std.gpu.memory import AddressSpace
from std.testing import assert_true
from layout import TileTensor, row_major

from mojo_rl.nn2.constants import DT
from mojo_rl.nn2.core.tensor_pack import TensorPack
from mojo_rl.nn2.initializer import Kaiming
from mojo_rl.nn2.core import ParamVisitor
from mojo_rl.nn2.combinators import RepeatConditional
from mojo_rl.nn2.primitives.conditional_transformer_block import (
    ConditionalTransformerBlock,
)


comptime EMB = 4
comptime HEADS = 2
comptime H = 3
comptime FF = 8
comptime DEPTH = 3
comptime BATCH = 2
comptime SEQ = H * EMB
comptime N = BATCH * SEQ

comptime Block = ConditionalTransformerBlock[EMB, HEADS, H, FF]
comptime Stack = RepeatConditional[DEPTH, Block]


def _a(n: Int) -> UnsafePointer[Scalar[DT], MutAnyOrigin]:
    return alloc[Scalar[DT]](n)


def _det(i: Int, scale: Float64) -> Scalar[DT]:
    var v = (Float64((i * 2654435761) % 1000) / 500.0) - 1.0
    return Scalar[DT](v * scale)


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
            p[i] = _det(self.idx + i + 1, 0.25)
        self.idx += n_elems


def test_stack_identity() raises:
    print("test_stack_identity ...")
    var stk = Stack.make[target="cpu", INIT=Kaiming]()
    var x = _a(N); var c = _a(N); var y = _a(N)
    for k in range(N):
        x[k] = _det(k + 1, 1.0)
        c[k] = _det(k + 50, 1.0)
    var x_t = TileTensor(x, row_major[BATCH, SEQ]())
    var c_t = TileTensor(c, row_major[BATCH, SEQ]())
    var y_t = TileTensor(y, row_major[BATCH, SEQ]())
    stk.forward["cpu", BATCH](
            TensorPack[2].of(x_t, c_t), output=y_t,
        )
    var maxd: Scalar[DT] = 0.0
    for k in range(N):
        var d = (y[k] - x[k]).__abs__()
        if d > maxd:
            maxd = d
    print("   max|stack(x,c) - x| =", maxd)
    assert_true(maxd == Scalar[DT](0.0),
                "DEPTH-stacked AdaLN-zero must be bitwise identity")
    x.free(); c.free(); y.free()
    _ = stk^
    print("  ok")


def test_stack_gradcheck() raises:
    print("test_stack_gradcheck ...")
    var stk = Stack.make[target="cpu", INIT=Kaiming]()
    var filler = FillParams()
    stk.for_each_param["cpu", FillParams]("stk", filler)
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
    stk.forward["cpu", BATCH](
            TensorPack[2].of(x_t, c_t), output=y_t,
        )

    var w_t = TileTensor(w, row_major[BATCH, SEQ]())
    var gx_t = TileTensor(gx, row_major[BATCH, SEQ]())
    var gc_t = TileTensor(gc, row_major[BATCH, SEQ]())
    stk.vjp["cpu", BATCH](w_t, TensorPack[2].of(gx_t, gc_t))

    comptime EPS = Scalar[DT](1e-3)
    var bad = 0
    for which in range(2):
        var p = x if which == 0 else c
        var ga = gx if which == 0 else gc
        for k in range(N):
            var saved = p[k]
            p[k] = saved + EPS
            stk.forward["cpu", BATCH](
            TensorPack[2].of(x_t, c_t), output=y_t,
        )
            var lp: Scalar[DT] = 0.0
            for j in range(N):
                lp += w[j] * y[j]
            p[k] = saved - EPS
            stk.forward["cpu", BATCH](
            TensorPack[2].of(x_t, c_t), output=y_t,
        )
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
    assert_true(bad == 0, "RepeatConditional gradcheck failed")
    x.free(); c.free(); y.free(); w.free(); gx.free(); gc.free()
    _ = stk^
    print("  ok")


def main() raises:
    print("=" * 70)
    print("RepeatConditional[DEPTH, ConditionalTransformerBlock] (Phase B, CPU)")
    print("=" * 70)
    test_stack_identity()
    test_stack_gradcheck()
    print("=" * 70)
    print("ALL PASSED")
    print("=" * 70)
