"""Linear GPU row-independence — does output row i depend on input row j≠i?

A matmul (BATCH, IN) @ (IN, OUT) must produce each output row from only its
own input row. If perturbing input row p changes an earlier output row, the
GPU matmul is bleeding across batch rows — which, since the GPT runs every
per-token Linear as a (SEQ, D) batch via Tokenwise, leaks future tokens into
earlier positions (the nn GPT val-0.48 / teacher-forcing-too-good bug).

Tested with a NON-tile-aligned batch (5) and an aligned one (8).
"""

from std.gpu.host import DeviceContext, DeviceBuffer
from std.math import abs
from std.testing import assert_true
from layout import TileTensor, row_major

from mojo_rl.nn.constants import DT
from mojo_rl.nn.initializer import Kaiming
from mojo_rl.nn.primitives.linear import Linear


def _mao(b: DeviceBuffer[DT]) -> UnsafePointer[Scalar[DT], MutAnyOrigin]:
    return rebind[UnsafePointer[Scalar[DT], MutAnyOrigin]](b.unsafe_ptr())


def _run[BATCH: Int, IN: Int, OUT: Int](ctx: DeviceContext, name: String) raises:
    print(name, " (BATCH=", BATCH, ") ...")
    comptime IN_N = BATCH * IN
    comptime OUT_N = BATCH * OUT
    var op = Linear[IN, OUT].make[target="gpu", INIT=Kaiming](ctx)

    var xh = ctx.enqueue_create_host_buffer[DT](IN_N)
    var y1h = ctx.enqueue_create_host_buffer[DT](OUT_N)
    var y2h = ctx.enqueue_create_host_buffer[DT](OUT_N)
    ctx.synchronize()
    for i in range(IN_N):
        xh.unsafe_ptr()[i] = Scalar[DT](0.2 * Float64(i % 5) - 0.4)

    var xd = ctx.enqueue_create_buffer[DT](IN_N)
    var y1d = ctx.enqueue_create_buffer[DT](OUT_N)
    var y2d = ctx.enqueue_create_buffer[DT](OUT_N)
    ctx.enqueue_copy(xd, xh)
    ctx.synchronize()

    var x_t = TileTensor(_mao(xd), row_major[BATCH, IN]())
    var y1_t = TileTensor(_mao(y1d), row_major[BATCH, OUT]())
    op.forward["gpu", BATCH](x_t, output=y1_t)

    # Perturb the LAST input row only.
    comptime p = BATCH - 1
    for k in range(IN):
        xh.unsafe_ptr()[p * IN + k] += Scalar[DT](5.0)
    ctx.enqueue_copy(xd, xh)
    ctx.synchronize()
    var y2_t = TileTensor(_mao(y2d), row_major[BATCH, OUT]())
    op.forward["gpu", BATCH](x_t, output=y2_t)

    ctx.enqueue_copy(y1h, y1d)
    ctx.enqueue_copy(y2h, y2d)
    ctx.synchronize()

    var max_earlier: Float64 = 0.0
    for r in range(p):
        for j in range(OUT):
            var d = abs(
                Float64(y1h.unsafe_ptr()[r * OUT + j])
                - Float64(y2h.unsafe_ptr()[r * OUT + j])
            )
            if d > max_earlier:
                max_earlier = d
    var changed_p: Float64 = 0.0
    for j in range(OUT):
        var d = abs(
            Float64(y1h.unsafe_ptr()[p * OUT + j])
            - Float64(y2h.unsafe_ptr()[p * OUT + j])
        )
        if d > changed_p:
            changed_p = d
    print("   max change at rows r<", p, " =", max_earlier, " (row p changed by", changed_p, ")")
    assert_true(
        max_earlier == 0.0,
        name + ": ROW BLEED — perturbing input row " + String(p)
        + " changed an earlier output row",
    )
    print("  ok — rows independent")


def main() raises:
    print("=" * 70)
    print("Linear GPU row-independence")
    print("=" * 70)
    var ctx = DeviceContext()
    _run[5, 4, 4](ctx, "M5_nonaligned")
    _run[8, 4, 4](ctx, "M8_aligned")
    _run[6, 16, 8](ctx, "M6_wider")
    print("=" * 70)
    print("ALL PASSED")
    print("=" * 70)
