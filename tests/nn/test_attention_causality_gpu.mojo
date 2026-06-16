"""ScaledDotProductAttention GPU causality test (custom + bmm paths).

The CPU causality test passes, but the full-model GPU forward leaks future
tokens into earlier outputs (generation-consistency test). All parity tests
compare identical full inputs, so a GPU-only causality bug slips through them.
This perturbs a FUTURE token's K/V on the GPU attention directly and asserts
earlier outputs are unchanged — for BOTH the custom and bmm GPU paths.

    pixi run -e apple  mojo run -I . tests/nn/test_attention_causality_gpu.mojo
    pixi run -e nvidia mojo run -I . tests/nn/test_attention_causality_gpu.mojo
"""

from std.gpu.host import DeviceContext, DeviceBuffer
from std.math import abs
from std.testing import assert_true
from layout import TileTensor, row_major

from mojo_rl.nn.constants import DT
from mojo_rl.nn.initializer import Zero
from mojo_rl.nn.primitives.attention import ScaledDotProductAttention


def _mao(b: DeviceBuffer[DT]) -> UnsafePointer[Scalar[DT], MutAnyOrigin]:
    return rebind[UnsafePointer[Scalar[DT], MutAnyOrigin]](b.unsafe_ptr())


def _run[USE_MAX: Bool](ctx: DeviceContext, name: String) raises:
    print(name, "...")
    comptime DIM = 4
    comptime N_HEADS = 2
    comptime SEQ = 5
    comptime BATCH = 1
    comptime IN_N = BATCH * SEQ * DIM * 3
    comptime OUT_N = BATCH * SEQ * DIM
    comptime KOFF = SEQ * DIM
    comptime VOFF = 2 * SEQ * DIM
    comptime p = SEQ - 1  # perturb the LAST position (future for all i<p)

    var op = ScaledDotProductAttention[
        DIM, N_HEADS, SEQ, True, USE_MAX
    ].make[target="gpu", INIT=Zero](ctx)

    var xh = ctx.enqueue_create_host_buffer[DT](IN_N)
    var y1h = ctx.enqueue_create_host_buffer[DT](OUT_N)
    var y2h = ctx.enqueue_create_host_buffer[DT](OUT_N)
    ctx.synchronize()
    for i in range(IN_N):
        xh.unsafe_ptr()[i] = Scalar[DT](0.31 * Float64(i % 7) - 1.0 + 0.05 * Float64(i))

    var xd = ctx.enqueue_create_buffer[DT](IN_N)
    var y1d = ctx.enqueue_create_buffer[DT](OUT_N)
    var y2d = ctx.enqueue_create_buffer[DT](OUT_N)
    ctx.enqueue_copy(xd, xh)
    ctx.synchronize()

    var x_t = TileTensor(_mao(xd), row_major[BATCH, SEQ * DIM * 3]())
    var y1_t = TileTensor(_mao(y1d), row_major[BATCH, SEQ * DIM]())
    op.forward["gpu", BATCH](x_t, output=y1_t)

    # Perturb K and V at the last position p (future for all i < p).
    for d in range(DIM):
        xh.unsafe_ptr()[KOFF + p * DIM + d] += Scalar[DT](3.3)
        xh.unsafe_ptr()[VOFF + p * DIM + d] -= Scalar[DT](2.7)
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
    print("   max change at earlier positions i<", p, " =", max_earlier)
    assert_true(
        max_earlier == 0.0,
        name + ": GPU CAUSALITY LEAK — future K/V changed an earlier output",
    )
    print("  ok — causal")


def main() raises:
    print("=" * 70)
    print("Attention GPU causality (custom + bmm)")
    print("=" * 70)
    var ctx = DeviceContext()
    _run[False](ctx, "custom_gpu")
    _run[True](ctx, "bmm_gpu")
    print("=" * 70)
    print("ALL PASSED")
    print("=" * 70)
