"""PR5c Step 5 GPU — CPU↔GPU parity for the WM loss ops.

SymlogMSELoss / BinaryLoss / TwoHotLoss forward+vjp, CPU vs Metal, ≤1e-4.
Completes the custom-op GPU set (with spike_rssm_ops_gpu's 5 RSSM ops).

Run: `pixi run -e apple mojo run -I . tests/nn/spike_loss_ops_gpu.mojo`
"""

from std.memory import alloc
from std.testing import assert_true
from std.gpu.host import DeviceContext, DeviceBuffer
from layout import TileTensor, row_major

from mojo_rl.nn.constants import DT
from mojo_rl.nn.core.tensor_pack import TensorPack
from mojo_rl.nn.initializer import Zero
from mojo_rl.deep_agents.dreamerv3.wm_loss_ops import (
    SymlogMSELoss, TwoHotLoss, BinaryLoss,
)


@always_inline
def _p(buf: DeviceBuffer[DT]) -> UnsafePointer[Scalar[DT], MutAnyOrigin]:
    return rebind[UnsafePointer[Scalar[DT], MutAnyOrigin]](buf.unsafe_ptr())


@always_inline
def _a(n: Int) -> UnsafePointer[Scalar[DT], MutAnyOrigin]:
    return alloc[Scalar[DT]](n)


def _pseudo(p: UnsafePointer[Scalar[DT], MutAnyOrigin], n: Int, seed: Int):
    var s = UInt64(seed * 2654435761 + 12345)
    for i in range(n):
        s = s * UInt64(6364136223846793005) + UInt64(1442695040888963407)
        p[i] = Scalar[DT]((Float64((s >> 33)) / Float64(UInt64(1) << 31)) - 1.0)


def _h2d(ctx: DeviceContext, dev: DeviceBuffer[DT],
         src: UnsafePointer[Scalar[DT], MutAnyOrigin], n: Int) raises:
    var h = ctx.enqueue_create_host_buffer[DT](n)
    ctx.synchronize()
    for k in range(n):
        h.unsafe_ptr()[k] = src[k]
    ctx.enqueue_copy(dev, h)
    ctx.synchronize()


def _d2h(ctx: DeviceContext, dev: DeviceBuffer[DT], n: Int) raises -> List[Scalar[DT]]:
    var h = ctx.enqueue_create_host_buffer[DT](n)
    ctx.synchronize()
    ctx.enqueue_copy(h, dev)
    ctx.synchronize()
    var out = List[Scalar[DT]]()
    for k in range(n):
        out.append(h.unsafe_ptr()[k])
    return out^


def _diff(got: List[Scalar[DT]], exp_: UnsafePointer[Scalar[DT], MutAnyOrigin]) -> Scalar[DT]:
    var m: Scalar[DT] = 0.0
    for i in range(len(got)):
        var d = got[i] - exp_[i]
        var ad = d if d >= 0 else -d
        if ad > m:
            m = ad
    return m


def test_symmse() raises:
    print("SymlogMSELoss CPU↔GPU ...")
    comptime B = 4
    comptime OBS = 4
    comptime N = B * OBS
    var pred = _a(N)
    var tgt = _a(N)
    var go = _a(B)
    _pseudo(pred, N, 1)
    _pseudo(tgt, N, 2)
    _pseudo(go, B, 3)

    var cpu = SymlogMSELoss[OBS].make["cpu", INIT=Zero]()
    var co = _a(B)
    var cgp = _a(N)
    var cgt = _a(N)
    var pt = TileTensor(pred, row_major[B, OBS]())
    var tt = TileTensor(tgt, row_major[B, OBS]())
    var cot = TileTensor(co, row_major[B, 1]())
    cpu.forward["cpu", B](
            TensorPack[2].of(pt, tt), output=cot,
        )
    var got = TileTensor(go, row_major[B, 1]())
    var cgpt = TileTensor(cgp, row_major[B, OBS]())
    var cgtt = TileTensor(cgt, row_major[B, OBS]())
    cpu.vjp["cpu", B](got, TensorPack[2].of(cgpt, cgtt))

    var ctx = DeviceContext()
    var gpu = SymlogMSELoss[OBS].make["gpu", INIT=Zero](ctx=ctx)
    var pd = ctx.enqueue_create_buffer[DT](N)
    var td = ctx.enqueue_create_buffer[DT](N)
    var od = ctx.enqueue_create_buffer[DT](B)
    var god = ctx.enqueue_create_buffer[DT](B)
    var gpd = ctx.enqueue_create_buffer[DT](N)
    var gtd = ctx.enqueue_create_buffer[DT](N)
    _h2d(ctx, pd, pred, N)
    _h2d(ctx, td, tgt, N)
    _h2d(ctx, god, go, B)
    var pdt = TileTensor(_p(pd), row_major[B, OBS]())
    var tdt = TileTensor(_p(td), row_major[B, OBS]())
    var odt = TileTensor(_p(od), row_major[B, 1]())
    gpu.forward["gpu", B](
            TensorPack[2].of(pdt, tdt), output=odt,
        )
    var godt = TileTensor(_p(god), row_major[B, 1]())
    var gpdt = TileTensor(_p(gpd), row_major[B, OBS]())
    var gtdt = TileTensor(_p(gtd), row_major[B, OBS]())
    gpu.vjp["gpu", B](godt, TensorPack[2].of(gpdt, gtdt))
    ctx.synchronize()

    var dfo = _diff(_d2h(ctx, od, B), co)
    var dgp = _diff(_d2h(ctx, gpd, N), cgp)
    print("  out =", dfo, " g_pred =", dgp)
    assert_true(dfo < Scalar[DT](1e-4), "SymlogMSE fwd parity")
    assert_true(dgp < Scalar[DT](1e-4), "SymlogMSE bwd parity")
    _ = pd; _ = td
    print("  ok")


def test_binary() raises:
    print("BinaryLoss CPU↔GPU ...")
    comptime B = 4
    var lo = _a(B)
    var tg = _a(B)
    var go = _a(B)
    _pseudo(lo, B, 4)
    for i in range(B):
        tg[i] = Scalar[DT](1.0) if (i % 2 == 0) else Scalar[DT](0.0)
    _pseudo(go, B, 6)

    var cpu = BinaryLoss.make["cpu", INIT=Zero]()
    var co = _a(B)
    var cgl = _a(B)
    var cgt = _a(B)
    var lot = TileTensor(lo, row_major[B, 1]())
    var tgt = TileTensor(tg, row_major[B, 1]())
    var cot = TileTensor(co, row_major[B, 1]())
    cpu.forward["cpu", B](
            TensorPack[2].of(lot, tgt), output=cot,
        )
    var got = TileTensor(go, row_major[B, 1]())
    var cglt = TileTensor(cgl, row_major[B, 1]())
    var cgtt = TileTensor(cgt, row_major[B, 1]())
    cpu.vjp["cpu", B](got, TensorPack[2].of(cglt, cgtt))

    var ctx = DeviceContext()
    var gpu = BinaryLoss.make["gpu", INIT=Zero](ctx=ctx)
    var lod = ctx.enqueue_create_buffer[DT](B)
    var tgd = ctx.enqueue_create_buffer[DT](B)
    var od = ctx.enqueue_create_buffer[DT](B)
    var god = ctx.enqueue_create_buffer[DT](B)
    var gld = ctx.enqueue_create_buffer[DT](B)
    var gtd = ctx.enqueue_create_buffer[DT](B)
    _h2d(ctx, lod, lo, B)
    _h2d(ctx, tgd, tg, B)
    _h2d(ctx, god, go, B)
    var lodt = TileTensor(_p(lod), row_major[B, 1]())
    var tgdt = TileTensor(_p(tgd), row_major[B, 1]())
    var odt = TileTensor(_p(od), row_major[B, 1]())
    gpu.forward["gpu", B](
            TensorPack[2].of(lodt, tgdt), output=odt,
        )
    var godt = TileTensor(_p(god), row_major[B, 1]())
    var gldt = TileTensor(_p(gld), row_major[B, 1]())
    var gtdt = TileTensor(_p(gtd), row_major[B, 1]())
    gpu.vjp["gpu", B](godt, TensorPack[2].of(gldt, gtdt))
    ctx.synchronize()

    var dfo = _diff(_d2h(ctx, od, B), co)
    var dgl = _diff(_d2h(ctx, gld, B), cgl)
    print("  out =", dfo, " g_logit =", dgl)
    assert_true(dfo < Scalar[DT](1e-4), "Binary fwd parity")
    assert_true(dgl < Scalar[DT](1e-4), "Binary bwd parity")
    _ = lod
    print("  ok")


def test_twohot() raises:
    print("TwoHotLoss CPU↔GPU ...")
    comptime B = 4
    comptime BINS = 7
    var lg = _a(B * BINS)
    var tg = _a(B)
    var go = _a(B)
    _pseudo(lg, B * BINS, 7)
    _pseudo(tg, B, 8)
    _pseudo(go, B, 9)

    var cpu = TwoHotLoss[BINS].make["cpu", INIT=Zero]()
    var co = _a(B)
    var cgl = _a(B * BINS)
    var cgt = _a(B)
    # variadic inputs need uniform DIMS[0]=BINS layout
    var lgt = TileTensor(lg, row_major[B, BINS]())
    var tgt = TileTensor(tg, row_major[B, BINS]())
    var cot = TileTensor(co, row_major[B, 1]())
    cpu.forward["cpu", B](
            TensorPack[2].of(lgt, tgt), output=cot,
        )
    var got = TileTensor(go, row_major[B, 1]())
    var cglt = TileTensor(cgl, row_major[B, BINS]())
    var cgtt = TileTensor(cgt, row_major[B, BINS]())
    cpu.vjp["cpu", B](got, TensorPack[2].of(cglt, cgtt))

    var ctx = DeviceContext()
    var gpu = TwoHotLoss[BINS].make["gpu", INIT=Zero](ctx=ctx)
    var lgd = ctx.enqueue_create_buffer[DT](B * BINS)
    var tgd = ctx.enqueue_create_buffer[DT](B)
    var od = ctx.enqueue_create_buffer[DT](B)
    var god = ctx.enqueue_create_buffer[DT](B)
    var gld = ctx.enqueue_create_buffer[DT](B * BINS)
    var gtd = ctx.enqueue_create_buffer[DT](B)
    _h2d(ctx, lgd, lg, B * BINS)
    _h2d(ctx, tgd, tg, B)
    _h2d(ctx, god, go, B)
    var lgdt = TileTensor(_p(lgd), row_major[B, BINS]())
    var tgdt = TileTensor(_p(tgd), row_major[B, BINS]())
    var odt = TileTensor(_p(od), row_major[B, 1]())
    gpu.forward["gpu", B](
            TensorPack[2].of(lgdt, tgdt), output=odt,
        )
    var godt = TileTensor(_p(god), row_major[B, 1]())
    var gldt = TileTensor(_p(gld), row_major[B, BINS]())
    var gtdt = TileTensor(_p(gtd), row_major[B, BINS]())
    gpu.vjp["gpu", B](godt, TensorPack[2].of(gldt, gtdt))
    ctx.synchronize()

    var dfo = _diff(_d2h(ctx, od, B), co)
    var dgl = _diff(_d2h(ctx, gld, B * BINS), cgl)
    print("  out =", dfo, " g_logits =", dgl)
    assert_true(dfo < Scalar[DT](1e-4), "TwoHot fwd parity")
    assert_true(dgl < Scalar[DT](1e-4), "TwoHot bwd parity")
    _ = lgd
    print("  ok")


def main() raises:
    print("=" * 70)
    print("PR5c Step 5 GPU — WM loss ops CPU↔GPU parity")
    print("=" * 70)
    test_symmse()
    test_binary()
    test_twohot()
    print("=" * 70)
    print("ALL PASSED")
    print("=" * 70)
