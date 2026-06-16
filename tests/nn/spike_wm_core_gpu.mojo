"""PR5c Step 5 GPU — WMCoreGraph composes + runs on Metal (forward+vjp).

The 5 custom ops are CPU↔GPU validated in isolation (spike_rssm_ops_gpu).
This confirms they compose INSIDE the real `ComputeGraph` on GPU: every
node's gpu path dispatches and the graph's GPU forward/backward wiring
(scatter-add / copy / zero kernels) executes. Gate: forward output finite,
vjp runs without error, forward still finite afterwards. Output
[B, 2+DETER+SC] = [dyn, rep, nd(passthrough), stoch_new(passthrough)].

Run: `pixi run -e apple mojo run -I . tests/nn/spike_wm_core_gpu.mojo`
"""

from std.testing import assert_true
from std.gpu.host import DeviceContext, DeviceBuffer
from layout import TileTensor, row_major

from mojo_rl.nn.constants import DT
from mojo_rl.nn.initializer import Kaiming
from mojo_rl.deep_agents.dreamerv3.wm import WMCoreGraph

comptime B = 2
comptime DETER = 16
comptime H = 12
comptime STOCH = 3
comptime CLASSES = 5
comptime BLOCKS = 4
comptime ACT = 2
comptime TOKEN = 8
comptime SC = STOCH * CLASSES
comptime CARRY = 2 + DETER + SC


@always_inline
def _p(buf: DeviceBuffer[DT]) -> UnsafePointer[Scalar[DT], MutAnyOrigin]:
    return rebind[UnsafePointer[Scalar[DT], MutAnyOrigin]](buf.unsafe_ptr())


def _pseudo_h(ctx: DeviceContext, dev: DeviceBuffer[DT], n: Int, seed: Int) raises:
    var h = ctx.enqueue_create_host_buffer[DT](n)
    ctx.synchronize()
    var s = UInt64(seed * 2654435761 + 12345)
    for i in range(n):
        s = s * UInt64(6364136223846793005) + UInt64(1442695040888963407)
        h.unsafe_ptr()[i] = Scalar[DT](
            (Float64((s >> 33)) / Float64(UInt64(1) << 31)) - 1.0
        )
    ctx.enqueue_copy(dev, h)
    ctx.synchronize()


def _fill(ctx: DeviceContext, dev: DeviceBuffer[DT], n: Int, v: Scalar[DT]) raises:
    var h = ctx.enqueue_create_host_buffer[DT](n)
    ctx.synchronize()
    for i in range(n):
        h.unsafe_ptr()[i] = v
    ctx.enqueue_copy(dev, h)
    ctx.synchronize()


def _d2h_finite(ctx: DeviceContext, dev: DeviceBuffer[DT], n: Int) raises -> Bool:
    var h = ctx.enqueue_create_host_buffer[DT](n)
    ctx.synchronize()
    ctx.enqueue_copy(h, dev)
    ctx.synchronize()
    for k in range(n):
        var x = h.unsafe_ptr()[k]
        if not (x == x):
            return False
        var ax = x if x >= 0 else -x
        if ax > Scalar[DT](1e30):
            return False
    return True


def main() raises:
    print("=" * 70)
    print("PR5c Step 5 GPU — WMCoreGraph forward+vjp on Metal")
    print("=" * 70)
    var ctx = DeviceContext()
    var g = WMCoreGraph[DETER, H, STOCH, CLASSES, BLOCKS, ACT, TOKEN].make[
        "gpu", INIT=Kaiming
    ](ctx=ctx)

    var deterd = ctx.enqueue_create_buffer[DT](B * DETER)
    var stochd = ctx.enqueue_create_buffer[DT](B * SC)
    var actd = ctx.enqueue_create_buffer[DT](B * ACT)
    var tokd = ctx.enqueue_create_buffer[DT](B * TOKEN)
    _pseudo_h(ctx, deterd, B * DETER, 1)
    _pseudo_h(ctx, stochd, B * SC, 2)
    _pseudo_h(ctx, actd, B * ACT, 3)
    _pseudo_h(ctx, tokd, B * TOKEN, 4)
    g.set_input["deter", B](TileTensor(_p(deterd), row_major[B, DETER]()))
    g.set_input["stoch", B](TileTensor(_p(stochd), row_major[B, SC]()))
    g.set_input["action", B](TileTensor(_p(actd), row_major[B, ACT]()))
    g.set_input["tokens", B](TileTensor(_p(tokd), row_major[B, TOKEN]()))

    var outd = ctx.enqueue_create_buffer[DT](B * CARRY)
    var out_t = TileTensor(_p(outd), row_major[B, CARRY]())
    g.forward["gpu", B](out_t)
    ctx.synchronize()
    assert_true(_d2h_finite(ctx, outd, B * CARRY), "gpu forward finite")
    print("  forward finite (all nodes' gpu paths dispatched)")

    var seedd = ctx.enqueue_create_buffer[DT](B * CARRY)
    _fill(ctx, seedd, B * CARRY, Scalar[DT](1.0))
    var seed_t = TileTensor(_p(seedd), row_major[B, CARRY]())
    g.vjp["gpu", B](seed_t)
    ctx.synchronize()
    print("  vjp ran (gpu backward wiring + custom-op vjp kernels OK)")

    # graph still consistent post-vjp: re-run forward, still finite.
    var out2 = ctx.enqueue_create_buffer[DT](B * CARRY)
    var out2_t = TileTensor(_p(out2), row_major[B, CARRY]())
    g.forward["gpu", B](out2_t)
    ctx.synchronize()
    assert_true(_d2h_finite(ctx, out2, B * CARRY), "gpu re-forward finite")
    print("  re-forward finite")
    print("=" * 70)
    print("PASSED — WMCoreGraph composes + runs forward+vjp on GPU")
    print("=" * 70)
