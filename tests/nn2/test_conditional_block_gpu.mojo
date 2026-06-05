"""ConditionalTransformerBlock + RepeatConditional GPU smoke (Phase B).

At AdaLN-zero init the block/stack is the identity, so:
  forward(x, c) == x          (bitwise)
  vjp(w):  grad_x == w,  grad_c == 0
which end-to-end exercises the GPU paths of Modulate/Gate/LN-no-affine/MHA/
FFN, the block's grad-copy kernel, and RepeatConditional's zero/accum kernels.

Run:  pixi run -e apple mojo run -I . tests/nn2/test_conditional_block_gpu.mojo
"""

from std.gpu.host import DeviceContext, DeviceBuffer
from std.testing import assert_true
from layout import TileTensor, row_major

from mojo_rl.nn2.constants import DT
from mojo_rl.nn2.initializer import Kaiming
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


def _p(b: DeviceBuffer[DT]) -> UnsafePointer[Scalar[DT], MutAnyOrigin]:
    return rebind[UnsafePointer[Scalar[DT], MutAnyOrigin]](b.unsafe_ptr())


def _det(i: Int, scale: Float64) -> Scalar[DT]:
    var v = (Float64((i * 2654435761) % 1000) / 500.0) - 1.0
    return Scalar[DT](v * scale)


def test_stack_gpu_identity_and_grad() raises:
    print("test_stack_gpu_identity_and_grad ...")
    var ctx = DeviceContext()
    var stk = RepeatConditional[
        DEPTH, ConditionalTransformerBlock[EMB, HEADS, H, FF]
    ].make[target="gpu", INIT=Kaiming](ctx)

    var x_d = ctx.enqueue_create_buffer[DT](N)
    var c_d = ctx.enqueue_create_buffer[DT](N)
    var y_d = ctx.enqueue_create_buffer[DT](N)
    var w_d = ctx.enqueue_create_buffer[DT](N)
    var gx_d = ctx.enqueue_create_buffer[DT](N)
    var gc_d = ctx.enqueue_create_buffer[DT](N)
    var xh = ctx.enqueue_create_host_buffer[DT](N)
    var ch = ctx.enqueue_create_host_buffer[DT](N)
    var wh = ctx.enqueue_create_host_buffer[DT](N)
    var oh = ctx.enqueue_create_host_buffer[DT](N)
    ctx.synchronize()

    for k in range(N):
        xh.unsafe_ptr()[k] = _det(k + 1, 1.0)
        ch.unsafe_ptr()[k] = _det(k + 50, 1.0)
        wh.unsafe_ptr()[k] = _det(k + 99, 1.0)
    ctx.enqueue_copy(x_d, xh)
    ctx.enqueue_copy(c_d, ch)
    ctx.enqueue_copy(w_d, wh)
    ctx.synchronize()

    var x_t = TileTensor(_p(x_d), row_major[BATCH, SEQ]())
    var c_t = TileTensor(_p(c_d), row_major[BATCH, SEQ]())
    var y_t = TileTensor(_p(y_d), row_major[BATCH, SEQ]())
    stk.forward["gpu", BATCH](x_t, c_t, output=y_t)
    ctx.enqueue_copy(oh, y_d)
    ctx.synchronize()
    var maxd: Scalar[DT] = 0.0
    for k in range(N):
        var d = (oh.unsafe_ptr()[k] - xh.unsafe_ptr()[k]).__abs__()
        if d > maxd:
            maxd = d
    print("   max|stack(x,c) - x| =", maxd)
    assert_true(maxd < Scalar[DT](1e-6), "GPU stack identity at init")

    var w_t = TileTensor(_p(w_d), row_major[BATCH, SEQ]())
    var gx_t = TileTensor(_p(gx_d), row_major[BATCH, SEQ]())
    var gc_t = TileTensor(_p(gc_d), row_major[BATCH, SEQ]())
    stk.vjp["gpu", BATCH](w_t, gx_t, gc_t)
    var gxh = ctx.enqueue_create_host_buffer[DT](N)
    var gch = ctx.enqueue_create_host_buffer[DT](N)
    ctx.enqueue_copy(gxh, gx_d)
    ctx.enqueue_copy(gch, gc_d)
    ctx.synchronize()

    var max_gx: Scalar[DT] = 0.0
    var max_gc: Scalar[DT] = 0.0
    for k in range(N):
        var dgx = (gxh.unsafe_ptr()[k] - wh.unsafe_ptr()[k]).__abs__()
        if dgx > max_gx:
            max_gx = dgx
        var agc = gch.unsafe_ptr()[k].__abs__()
        if agc > max_gc:
            max_gc = agc
    print("   max|grad_x - w| =", max_gx, "  max|grad_c| =", max_gc)
    assert_true(max_gx < Scalar[DT](1e-6), "grad_x must equal grad_out at init")
    assert_true(max_gc < Scalar[DT](1e-6), "grad_c must be 0 at init")
    _ = stk^
    print("  ok")


def main() raises:
    print("=" * 70)
    print("ConditionalTransformerBlock / RepeatConditional GPU smoke (Phase B)")
    print("=" * 70)
    test_stack_gpu_identity_and_grad()
    print("=" * 70)
    print("ALL PASSED")
    print("=" * 70)
