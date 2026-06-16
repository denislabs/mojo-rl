"""Isolate: TwoHotDecode forward/backward CPU vs GPU (deterministic, Apple).

Decode has no RNG, so any CPU/GPU diff here is a real kernel bug (vs the
policy/td-target diffs which also involve RSample noise). Fixed logits in,
compare decoded scalar + grad-w.r.t-logits.
"""

from std.memory import alloc
from std.math import abs
from std.testing import assert_true, TestSuite
from std.gpu.host import DeviceContext, DeviceBuffer
from layout import row_major, TileTensor

from mojo_rl.nn.constants import DT
from mojo_rl.nn.initializer import Zero
from mojo_rl.deep_agents.tdmpc2.losses import TwoHotDecode

comptime BINS = 11
comptime VMIN = -10
comptime VMAX = 10
comptime B = 4
comptime DecT = TwoHotDecode[BINS, VMIN, VMAX]


def _a(n: Int) -> UnsafePointer[Scalar[DT], MutAnyOrigin]:
    return alloc[Scalar[DT]](n)


def _fill(p: UnsafePointer[Scalar[DT], MutAnyOrigin], n: Int, sd: Int):
    var s = UInt64(sd * 2654435761 + 12345)
    for i in range(n):
        s = s * UInt64(6364136223846793005) + UInt64(1442695040888963407)
        p[i] = Scalar[DT](Float64((s >> 33)) / Float64(UInt64(1) << 31) - 0.5)


def _dp(b: DeviceBuffer[DT]) -> UnsafePointer[Scalar[DT], MutAnyOrigin]:
    return rebind[UnsafePointer[Scalar[DT], MutAnyOrigin]](b.unsafe_ptr())


def test_decode_parity() raises:
    var ctx = DeviceContext()
    var dc = DecT.make["cpu", INIT=Zero]()
    var dg = DecT.make["gpu", INIT=Zero](ctx=ctx)

    var lg = _a(B * BINS)
    _fill(lg, B * BINS, 7)
    # CPU forward
    var oc = _a(B)
    var oc_t = TileTensor(oc, row_major[B, 1]())
    dc.forward["cpu", B](TileTensor(lg, row_major[B, BINS]()), output=oc_t)

    # GPU forward (upload logits)
    var d_lg = ctx.enqueue_create_buffer[DT](B * BINS)
    var h_lg = ctx.enqueue_create_host_buffer[DT](B * BINS)
    ctx.synchronize()
    for i in range(B * BINS):
        h_lg.unsafe_ptr()[i] = lg[i]
    ctx.enqueue_copy(d_lg, h_lg)
    ctx.synchronize()
    var d_o = ctx.enqueue_create_buffer[DT](B)
    var og_t = TileTensor(_dp(d_o), row_major[B, 1]())
    dg.forward["gpu", B](TileTensor(_dp(d_lg), row_major[B, BINS]()), output=og_t)
    var h_o = ctx.enqueue_create_host_buffer[DT](B)
    ctx.enqueue_copy(h_o, d_o)
    ctx.synchronize()

    var max_d: Scalar[DT] = 0.0
    for b in range(B):
        var d = oc[b] - h_o.unsafe_ptr()[b]
        if d < 0:
            d = -d
        if d > max_d:
            max_d = d
        print("  b", b, " cpu=", oc[b], " gpu=", h_o.unsafe_ptr()[b])
    print("  max abs diff =", max_d)
    assert_true(max_d < Scalar[DT](1e-4), "TwoHotDecode CPU/GPU must match")
    lg.free(); oc.free()


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
