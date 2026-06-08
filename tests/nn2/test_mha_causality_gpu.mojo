"""MultiHeadAttention composite causality (QKV-proj → SDPA → out-proj).

Isolated SDPA is causal, but the full GPT leaks. Suspect: the QKV projection
emits token-major [tok: q|k|v] while SDPA reads qkv-major [all-q|all-k|all-v],
so SDPA's position axis is scrambled and the causal mask hits the wrong axis.

Test: feed the MHA composite a (1, SEQ*DIM) residual stream, perturb a future
TOKEN's input (position p), and assert earlier token outputs are unchanged.
A causal MHA must leave tokens < p untouched.
"""

from std.gpu.host import DeviceContext, DeviceBuffer
from std.math import abs
from std.testing import assert_true
from layout import TileTensor, row_major

from mojo_rl.nn2.constants import DT
from mojo_rl.nn2.initializer import Kaiming
from mojo_rl.nn2.models.transformer import MultiHeadAttention


def _mao(b: DeviceBuffer[DT]) -> UnsafePointer[Scalar[DT], MutAnyOrigin]:
    return rebind[UnsafePointer[Scalar[DT], MutAnyOrigin]](b.unsafe_ptr())


def main() raises:
    print("=" * 70)
    print("MultiHeadAttention composite causality")
    print("=" * 70)
    comptime DIM = 8
    comptime N_HEADS = 2
    comptime SEQ = 5
    comptime BATCH = 1
    comptime N = BATCH * SEQ * DIM
    comptime p = SEQ - 1  # perturb last token (future for all i<p)

    var op = MultiHeadAttention[DIM, N_HEADS, SEQ, True].make[
        target="gpu", INIT=Kaiming
    ](ctx=DeviceContext())
    var ctx = DeviceContext()

    var xh = ctx.enqueue_create_host_buffer[DT](N)
    var y1h = ctx.enqueue_create_host_buffer[DT](N)
    var y2h = ctx.enqueue_create_host_buffer[DT](N)
    ctx.synchronize()
    for i in range(N):
        xh.unsafe_ptr()[i] = Scalar[DT](0.2 * Float64(i % 6) - 0.5)

    var xd = ctx.enqueue_create_buffer[DT](N)
    var y1d = ctx.enqueue_create_buffer[DT](N)
    var y2d = ctx.enqueue_create_buffer[DT](N)
    ctx.enqueue_copy(xd, xh)
    ctx.synchronize()
    var x_t = TileTensor(_mao(xd), row_major[BATCH, SEQ * DIM]())
    var y1_t = TileTensor(_mao(y1d), row_major[BATCH, SEQ * DIM]())
    op.forward["gpu", BATCH](x_t, output=y1_t)

    for d in range(DIM):
        xh.unsafe_ptr()[p * DIM + d] += Scalar[DT](4.0)
    ctx.enqueue_copy(xd, xh)
    ctx.synchronize()
    var y2_t = TileTensor(_mao(y2d), row_major[BATCH, SEQ * DIM]())
    op.forward["gpu", BATCH](x_t, output=y2_t)

    ctx.enqueue_copy(y1h, y1d)
    ctx.enqueue_copy(y2h, y2d)
    ctx.synchronize()

    var max_earlier: Float64 = 0.0
    for i in range(p):
        for d in range(DIM):
            var diff = abs(
                Float64(y1h.unsafe_ptr()[i * DIM + d])
                - Float64(y2h.unsafe_ptr()[i * DIM + d])
            )
            if diff > max_earlier:
                max_earlier = diff
    print("   max change at earlier tokens i<", p, " =", max_earlier)
    assert_true(
        max_earlier == 0.0,
        "MHA CAUSALITY LEAK — a future token changed an earlier token's output",
    )
    print("  ok — MHA causal")
    print("=" * 70)
    print("ALL PASSED")
    print("=" * 70)
