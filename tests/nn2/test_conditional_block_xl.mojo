"""ConditionalTransformerBlock with EXPANDED (XL) predictor attention.

The paper LeWM predictor uses head_dim decoupled from emb (16 heads × 64 =
1024 inner > emb 192). This test exercises that path on nn2 via the new
`HEAD_DIM` param with inner = HEADS·HEAD_DIM > EMB. AdaLN-zero init makes the
block the identity regardless of attention width, so:
  forward(x, c) == x   (bitwise) ;  vjp(w): grad_x == w, grad_c == 0
which end-to-end exercises the expanded MultiHeadAttentionXL QKV/SDPA/out
projections (Linear[EMB, 3·HEADS·HEAD_DIM] → SDPA@inner → Linear[inner, EMB]),
forward + backward, on CPU and GPU.

Run:  pixi run -e apple mojo run -I . tests/nn2/test_conditional_block_xl.mojo
"""

from std.memory import alloc
from std.gpu.host import DeviceContext, DeviceBuffer
from std.testing import assert_true
from layout import TileTensor, row_major

from mojo_rl.nn2.constants import DT
from mojo_rl.nn2.core.tensor_pack import TensorPack
from mojo_rl.nn2.initializer import Kaiming
from mojo_rl.nn2.combinators import RepeatConditional
from mojo_rl.nn2.primitives.conditional_transformer_block import (
    ConditionalTransformerBlock,
)


comptime EMB = 4
comptime HEADS = 2
comptime HEAD_DIM = 4          # inner = HEADS·HEAD_DIM = 8 > EMB=4 (expanded)
comptime H = 3
comptime FF = 8
comptime DEPTH = 2
comptime BATCH = 2
comptime SEQ = H * EMB
comptime N = BATCH * SEQ

comptime Stack = RepeatConditional[
    DEPTH, ConditionalTransformerBlock[EMB, HEADS, H, FF, HEAD_DIM]
]


def _p(b: DeviceBuffer[DT]) -> UnsafePointer[Scalar[DT], MutAnyOrigin]:
    return rebind[UnsafePointer[Scalar[DT], MutAnyOrigin]](b.unsafe_ptr())


def _det(i: Int) -> Scalar[DT]:
    return Scalar[DT]((Float64((i * 2654435761) % 1000) / 500.0) - 1.0)


def _a(n: Int) -> UnsafePointer[Scalar[DT], MutAnyOrigin]:
    return alloc[Scalar[DT]](n)


def test_cpu() raises:
    print("test_cpu (XL, inner=", HEADS * HEAD_DIM, "> emb=", EMB, ") ...")
    var stk = Stack.make[target="cpu", INIT=Kaiming]()
    var x = _a(N); var c = _a(N); var y = _a(N)
    for k in range(N):
        x[k] = _det(k + 1); c[k] = _det(k + 50)
    var x_t = TileTensor(x, row_major[BATCH, SEQ]())
    var c_t = TileTensor(c, row_major[BATCH, SEQ]())
    var y_t = TileTensor(y, row_major[BATCH, SEQ]())
    stk.forward["cpu", BATCH](TensorPack[2].of(x_t, c_t), output=y_t)
    var maxd: Scalar[DT] = 0.0
    for k in range(N):
        var d = (y[k] - x[k]).__abs__()
        if d > maxd:
            maxd = d
    print("   max|stack(x,c) - x| =", maxd)
    assert_true(maxd < Scalar[DT](1e-6), "XL block identity at init (cpu)")

    var w = _a(N); var gx = _a(N); var gc = _a(N)
    for k in range(N):
        w[k] = _det(k + 99)
    var w_t = TileTensor(w, row_major[BATCH, SEQ]())
    var gx_t = TileTensor(gx, row_major[BATCH, SEQ]())
    var gc_t = TileTensor(gc, row_major[BATCH, SEQ]())
    stk.vjp["cpu", BATCH](w_t, TensorPack[2].of(gx_t, gc_t))
    var mgx: Scalar[DT] = 0.0; var mgc: Scalar[DT] = 0.0
    for k in range(N):
        var dgx = (gx[k] - w[k]).__abs__()
        if dgx > mgx:
            mgx = dgx
        if gc[k].__abs__() > mgc:
            mgc = gc[k].__abs__()
    print("   max|grad_x - w| =", mgx, " max|grad_c| =", mgc)
    assert_true(mgx < Scalar[DT](1e-6), "grad_x == grad_out (cpu)")
    assert_true(mgc < Scalar[DT](1e-6), "grad_c == 0 (cpu)")
    x.free(); c.free(); y.free(); w.free(); gx.free(); gc.free()
    _ = stk^
    print("  ok")


def test_gpu() raises:
    print("test_gpu (XL) ...")
    var ctx = DeviceContext()
    var stk = Stack.make[target="gpu", INIT=Kaiming](ctx)
    var x_d = ctx.enqueue_create_buffer[DT](N)
    var c_d = ctx.enqueue_create_buffer[DT](N)
    var y_d = ctx.enqueue_create_buffer[DT](N)
    var w_d = ctx.enqueue_create_buffer[DT](N)
    var gx_d = ctx.enqueue_create_buffer[DT](N)
    var gc_d = ctx.enqueue_create_buffer[DT](N)
    var xh = ctx.enqueue_create_host_buffer[DT](N)
    var wh = ctx.enqueue_create_host_buffer[DT](N)
    var oh = ctx.enqueue_create_host_buffer[DT](N)
    var gxh = ctx.enqueue_create_host_buffer[DT](N)
    var gch = ctx.enqueue_create_host_buffer[DT](N)
    ctx.synchronize()
    for k in range(N):
        xh.unsafe_ptr()[k] = _det(k + 1)
        wh.unsafe_ptr()[k] = _det(k + 99)
    ctx.enqueue_copy(x_d, xh); ctx.enqueue_copy(w_d, wh)
    var ch = ctx.enqueue_create_host_buffer[DT](N)
    for k in range(N):
        ch.unsafe_ptr()[k] = _det(k + 50)
    ctx.enqueue_copy(c_d, ch)
    ctx.synchronize()

    var x_t = TileTensor(_p(x_d), row_major[BATCH, SEQ]())
    var c_t = TileTensor(_p(c_d), row_major[BATCH, SEQ]())
    var y_t = TileTensor(_p(y_d), row_major[BATCH, SEQ]())
    stk.forward["gpu", BATCH](TensorPack[2].of(x_t, c_t), output=y_t)
    ctx.enqueue_copy(oh, y_d); ctx.synchronize()
    var maxd: Scalar[DT] = 0.0
    for k in range(N):
        var d = (oh.unsafe_ptr()[k] - xh.unsafe_ptr()[k]).__abs__()
        if d > maxd:
            maxd = d
    print("   max|stack(x,c) - x| =", maxd)
    assert_true(maxd < Scalar[DT](1e-6), "XL block identity at init (gpu)")

    var w_t = TileTensor(_p(w_d), row_major[BATCH, SEQ]())
    var gx_t = TileTensor(_p(gx_d), row_major[BATCH, SEQ]())
    var gc_t = TileTensor(_p(gc_d), row_major[BATCH, SEQ]())
    stk.vjp["gpu", BATCH](w_t, TensorPack[2].of(gx_t, gc_t))
    ctx.enqueue_copy(gxh, gx_d); ctx.enqueue_copy(gch, gc_d); ctx.synchronize()
    var mgx: Scalar[DT] = 0.0; var mgc: Scalar[DT] = 0.0
    for k in range(N):
        var dgx = (gxh.unsafe_ptr()[k] - wh.unsafe_ptr()[k]).__abs__()
        if dgx > mgx:
            mgx = dgx
        if gch.unsafe_ptr()[k].__abs__() > mgc:
            mgc = gch.unsafe_ptr()[k].__abs__()
    print("   max|grad_x - w| =", mgx, " max|grad_c| =", mgc)
    assert_true(mgx < Scalar[DT](1e-6), "grad_x == grad_out (gpu)")
    assert_true(mgc < Scalar[DT](1e-6), "grad_c == 0 (gpu)")
    _ = stk^
    print("  ok")


def main() raises:
    print("=" * 70)
    print("ConditionalTransformerBlock — expanded (XL) attention")
    print("=" * 70)
    test_cpu()
    test_gpu()
    print("=" * 70)
    print("ALL PASSED")
    print("=" * 70)
