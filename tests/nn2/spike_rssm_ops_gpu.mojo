"""PR5c Step 5 — CPU↔GPU parity for the custom RSSM ops.

Each op runs forward+vjp on CPU and on GPU (Metal, via H2D/D2H) over the
same pseudo-random inputs; asserts max-abs diff ≤1e-4. No jax fixture —
the math is identical to the validated CPU path, so CPU is the oracle.

Run: `pixi run -e apple mojo run -I . tests/nn2/spike_rssm_ops_gpu.mojo`
"""

from std.memory import alloc
from std.testing import assert_true
from std.gpu.host import DeviceContext, DeviceBuffer
from layout import TileTensor, row_major

from mojo_rl.nn2.constants import DT
from mojo_rl.nn2.initializer import Zero
from mojo_rl.deep_agents2.dreamerv3.rssm_ops import (
    ActionSquash, BlockGroupAssemble, GRUGate, StraightThroughSample,
)
from mojo_rl.deep_agents2.dreamerv3.onehot_kl import OneHotKLLoss


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


def test_action_squash() raises:
    print("ActionSquash CPU↔GPU ...")
    comptime B = 4
    comptime ACT = 3
    comptime N = B * ACT
    var x = _a(N)
    var go = _a(N)
    _pseudo(x, N, 1)
    _pseudo(go, N, 2)
    # scale x up so some |x|>1 exercises the max(1,|x|) branch
    for i in range(N):
        x[i] = x[i] * Scalar[DT](2.0)

    # CPU
    var cpu = ActionSquash[ACT].make["cpu", INIT=Zero]()
    var co = _a(N)
    var cgi = _a(N)
    var xt = TileTensor(x, row_major[B, ACT]())
    var cot = TileTensor(co, row_major[B, ACT]())
    cpu.forward["cpu", B](xt, output=cot)
    var got = TileTensor(go, row_major[B, ACT]())
    var cgit = TileTensor(cgi, row_major[B, ACT]())
    cpu.vjp["cpu", B](got, cgit)

    # GPU
    var ctx = DeviceContext()
    var gpu = ActionSquash[ACT].make["gpu", INIT=Zero](ctx=ctx)
    var xd = ctx.enqueue_create_buffer[DT](N)
    var od = ctx.enqueue_create_buffer[DT](N)
    var god = ctx.enqueue_create_buffer[DT](N)
    var gid = ctx.enqueue_create_buffer[DT](N)
    _h2d(ctx, xd, x, N)
    _h2d(ctx, god, go, N)
    var xdt = TileTensor(_p(xd), row_major[B, ACT]())
    var odt = TileTensor(_p(od), row_major[B, ACT]())
    gpu.forward["gpu", B](xdt, output=odt)
    var godt = TileTensor(_p(god), row_major[B, ACT]())
    var gidt = TileTensor(_p(gid), row_major[B, ACT]())
    gpu.vjp["gpu", B](godt, gidt)
    ctx.synchronize()

    var df = _diff(_d2h(ctx, od, N), co)
    var db = _diff(_d2h(ctx, gid, N), cgi)
    print("  fwd diff =", df, " bwd diff =", db)
    assert_true(df < Scalar[DT](1e-4), "ActionSquash fwd parity")
    assert_true(db < Scalar[DT](1e-4), "ActionSquash bwd parity")
    _ = xd
    print("  ok")
    x.free(); go.free(); co.free(); cgi.free()


def test_gru_gate() raises:
    print("GRUGate CPU↔GPU ...")
    comptime B = 4
    comptime D = 16
    comptime BLK = 4
    comptime GRU = 3 * D
    var gru = _a(B * GRU)
    var deter = _a(B * D)
    var go = _a(B * D)
    _pseudo(gru, B * GRU, 3)
    _pseudo(deter, B * D, 4)
    _pseudo(go, B * D, 5)

    var cpu = GRUGate[D, BLK].make["cpu", INIT=Zero]()
    var cnd = _a(B * D)
    var grut = TileTensor(gru, row_major[B, GRU]())
    var dett = TileTensor(deter, row_major[B, GRU]())
    var cndt = TileTensor(cnd, row_major[B, D]())
    cpu.forward["cpu", B](grut, dett, output=cndt)
    var cgg = _a(B * GRU)
    var cgd = _a(B * D)
    var got = TileTensor(go, row_major[B, D]())
    var cggt = TileTensor(cgg, row_major[B, GRU]())
    var cgdt = TileTensor(cgd, row_major[B, GRU]())
    cpu.vjp["cpu", B](got, cggt, cgdt)

    var ctx = DeviceContext()
    var gpu = GRUGate[D, BLK].make["gpu", INIT=Zero](ctx=ctx)
    var grud = ctx.enqueue_create_buffer[DT](B * GRU)
    var deterd = ctx.enqueue_create_buffer[DT](B * D)
    var god = ctx.enqueue_create_buffer[DT](B * D)
    var ndd = ctx.enqueue_create_buffer[DT](B * D)
    var ggd = ctx.enqueue_create_buffer[DT](B * GRU)
    var gdd = ctx.enqueue_create_buffer[DT](B * D)
    _h2d(ctx, grud, gru, B * GRU)
    _h2d(ctx, deterd, deter, B * D)
    _h2d(ctx, god, go, B * D)
    var grudt = TileTensor(_p(grud), row_major[B, GRU]())
    var deterdt = TileTensor(_p(deterd), row_major[B, GRU]())
    var nddt = TileTensor(_p(ndd), row_major[B, D]())
    gpu.forward["gpu", B](grudt, deterdt, output=nddt)
    var godt = TileTensor(_p(god), row_major[B, D]())
    var ggdt = TileTensor(_p(ggd), row_major[B, GRU]())
    var gddt = TileTensor(_p(gdd), row_major[B, GRU]())
    gpu.vjp["gpu", B](godt, ggdt, gddt)
    ctx.synchronize()

    var dnd = _diff(_d2h(ctx, ndd, B * D), cnd)
    var dgg = _diff(_d2h(ctx, ggd, B * GRU), cgg)
    var dgd = _diff(_d2h(ctx, gdd, B * D), cgd)
    print("  nd =", dnd, " g_gru =", dgg, " g_deter =", dgd)
    assert_true(dnd < Scalar[DT](1e-4), "GRUGate fwd parity")
    assert_true(dgg < Scalar[DT](1e-4), "GRUGate g_gru parity")
    assert_true(dgd < Scalar[DT](1e-4), "GRUGate g_deter parity")
    _ = grud; _ = deterd
    print("  ok")


def test_bga() raises:
    print("BlockGroupAssemble CPU↔GPU ...")
    comptime B = 4
    comptime D = 16
    comptime H = 12
    comptime BLK = 4
    comptime OUT = D + 3 * H * BLK
    var deter = _a(B * D)
    var x0 = _a(B * H)
    var x1 = _a(B * H)
    var x2 = _a(B * H)
    var go = _a(B * OUT)
    _pseudo(deter, B * D, 6)
    _pseudo(x0, B * H, 7)
    _pseudo(x1, B * H, 8)
    _pseudo(x2, B * H, 9)
    _pseudo(go, B * OUT, 10)

    var cpu = BlockGroupAssemble[D, H, BLK].make["cpu", INIT=Zero]()
    var co = _a(B * OUT)
    var dt = TileTensor(deter, row_major[B, D]())
    var x0t = TileTensor(x0, row_major[B, D]())
    var x1t = TileTensor(x1, row_major[B, D]())
    var x2t = TileTensor(x2, row_major[B, D]())
    var cot = TileTensor(co, row_major[B, OUT]())
    cpu.forward["cpu", B](dt, x0t, x1t, x2t, output=cot)
    var cgd = _a(B * D)
    var cg0 = _a(B * H)
    var cg1 = _a(B * H)
    var cg2 = _a(B * H)
    var got = TileTensor(go, row_major[B, OUT]())
    var cgdt = TileTensor(cgd, row_major[B, D]())
    var cg0t = TileTensor(cg0, row_major[B, D]())
    var cg1t = TileTensor(cg1, row_major[B, D]())
    var cg2t = TileTensor(cg2, row_major[B, D]())
    cpu.vjp["cpu", B](got, cgdt, cg0t, cg1t, cg2t)

    var ctx = DeviceContext()
    var gpu = BlockGroupAssemble[D, H, BLK].make["gpu", INIT=Zero](ctx=ctx)
    var dd = ctx.enqueue_create_buffer[DT](B * D)
    var x0d = ctx.enqueue_create_buffer[DT](B * H)
    var x1d = ctx.enqueue_create_buffer[DT](B * H)
    var x2d = ctx.enqueue_create_buffer[DT](B * H)
    var od = ctx.enqueue_create_buffer[DT](B * OUT)
    var god = ctx.enqueue_create_buffer[DT](B * OUT)
    var gdd = ctx.enqueue_create_buffer[DT](B * D)
    var g0d = ctx.enqueue_create_buffer[DT](B * H)
    var g1d = ctx.enqueue_create_buffer[DT](B * H)
    var g2d = ctx.enqueue_create_buffer[DT](B * H)
    _h2d(ctx, dd, deter, B * D)
    _h2d(ctx, x0d, x0, B * H)
    _h2d(ctx, x1d, x1, B * H)
    _h2d(ctx, x2d, x2, B * H)
    _h2d(ctx, god, go, B * OUT)
    var ddt = TileTensor(_p(dd), row_major[B, D]())
    var x0dt = TileTensor(_p(x0d), row_major[B, D]())
    var x1dt = TileTensor(_p(x1d), row_major[B, D]())
    var x2dt = TileTensor(_p(x2d), row_major[B, D]())
    var odt = TileTensor(_p(od), row_major[B, OUT]())
    gpu.forward["gpu", B](ddt, x0dt, x1dt, x2dt, output=odt)
    var godt = TileTensor(_p(god), row_major[B, OUT]())
    var gddt = TileTensor(_p(gdd), row_major[B, D]())
    var g0dt = TileTensor(_p(g0d), row_major[B, D]())
    var g1dt = TileTensor(_p(g1d), row_major[B, D]())
    var g2dt = TileTensor(_p(g2d), row_major[B, D]())
    gpu.vjp["gpu", B](godt, gddt, g0dt, g1dt, g2dt)
    ctx.synchronize()

    var dfo = _diff(_d2h(ctx, od, B * OUT), co)
    var dgdv = _diff(_d2h(ctx, gdd, B * D), cgd)
    var dg0 = _diff(_d2h(ctx, g0d, B * H), cg0)
    print("  out =", dfo, " g_deter =", dgdv, " g_x0 =", dg0)
    assert_true(dfo < Scalar[DT](1e-4), "BGA fwd parity")
    assert_true(dgdv < Scalar[DT](1e-4), "BGA g_deter parity")
    assert_true(dg0 < Scalar[DT](1e-4), "BGA g_x0 parity")
    print("  ok")


def test_st_sample() raises:
    print("StraightThroughSample CPU↔GPU ...")
    comptime B = 4
    comptime STOCH = 3
    comptime CLASSES = 5
    comptime SC = STOCH * CLASSES
    comptime N = B * SC
    var z = _a(N)
    var go = _a(N)
    _pseudo(z, N, 11)
    _pseudo(go, N, 12)

    var cpu = StraightThroughSample[STOCH, CLASSES].make["cpu", INIT=Zero]()
    var co = _a(N)
    var cgz = _a(N)
    var zt = TileTensor(z, row_major[B, SC]())
    var cot = TileTensor(co, row_major[B, SC]())
    cpu.forward["cpu", B](zt, output=cot)
    var got = TileTensor(go, row_major[B, SC]())
    var cgzt = TileTensor(cgz, row_major[B, SC]())
    cpu.vjp["cpu", B](got, cgzt)

    var ctx = DeviceContext()
    var gpu = StraightThroughSample[STOCH, CLASSES].make["gpu", INIT=Zero](ctx=ctx)
    var zd = ctx.enqueue_create_buffer[DT](N)
    var od = ctx.enqueue_create_buffer[DT](N)
    var god = ctx.enqueue_create_buffer[DT](N)
    var gzd = ctx.enqueue_create_buffer[DT](N)
    _h2d(ctx, zd, z, N)
    _h2d(ctx, god, go, N)
    var zdt = TileTensor(_p(zd), row_major[B, SC]())
    var odt = TileTensor(_p(od), row_major[B, SC]())
    gpu.forward["gpu", B](zdt, output=odt)
    var godt = TileTensor(_p(god), row_major[B, SC]())
    var gzdt = TileTensor(_p(gzd), row_major[B, SC]())
    gpu.vjp["gpu", B](godt, gzdt)
    ctx.synchronize()

    var dfo = _diff(_d2h(ctx, od, N), co)
    var dgz = _diff(_d2h(ctx, gzd, N), cgz)
    print("  onehot =", dfo, " grad_z =", dgz)
    assert_true(dfo < Scalar[DT](1e-4), "ST fwd parity")
    assert_true(dgz < Scalar[DT](1e-4), "ST bwd parity")
    print("  ok")


def test_onehot_kl() raises:
    print("OneHotKLLoss CPU↔GPU ...")
    comptime B = 4
    comptime STOCH = 3
    comptime CLASSES = 5
    comptime SC = STOCH * CLASSES
    comptime N = B * SC
    var post = _a(N)
    var prior = _a(N)
    var go = _a(B * 2)
    _pseudo(post, N, 13)
    _pseudo(prior, N, 14)
    _pseudo(go, B * 2, 15)

    var cpu = OneHotKLLoss[STOCH, CLASSES].make["cpu", INIT=Zero]()
    var co = _a(B * 2)
    var postt = TileTensor(post, row_major[B, SC]())
    var priort = TileTensor(prior, row_major[B, SC]())
    var cot = TileTensor(co, row_major[B, 2]())
    cpu.forward["cpu", B](postt, priort, output=cot)
    var cgp = _a(N)
    var cgpr = _a(N)
    var got = TileTensor(go, row_major[B, 2]())
    var cgpt = TileTensor(cgp, row_major[B, SC]())
    var cgprt = TileTensor(cgpr, row_major[B, SC]())
    cpu.vjp["cpu", B](got, cgpt, cgprt)

    var ctx = DeviceContext()
    var gpu = OneHotKLLoss[STOCH, CLASSES].make["gpu", INIT=Zero](ctx=ctx)
    var postd = ctx.enqueue_create_buffer[DT](N)
    var priord = ctx.enqueue_create_buffer[DT](N)
    var od = ctx.enqueue_create_buffer[DT](B * 2)
    var god = ctx.enqueue_create_buffer[DT](B * 2)
    var gpd = ctx.enqueue_create_buffer[DT](N)
    var gprd = ctx.enqueue_create_buffer[DT](N)
    _h2d(ctx, postd, post, N)
    _h2d(ctx, priord, prior, N)
    _h2d(ctx, god, go, B * 2)
    var postdt = TileTensor(_p(postd), row_major[B, SC]())
    var priordt = TileTensor(_p(priord), row_major[B, SC]())
    var odt = TileTensor(_p(od), row_major[B, 2]())
    gpu.forward["gpu", B](postdt, priordt, output=odt)
    var godt = TileTensor(_p(god), row_major[B, 2]())
    var gpdt = TileTensor(_p(gpd), row_major[B, SC]())
    var gprdt = TileTensor(_p(gprd), row_major[B, SC]())
    gpu.vjp["gpu", B](godt, gpdt, gprdt)
    ctx.synchronize()

    var dfo = _diff(_d2h(ctx, od, B * 2), co)
    var dgp = _diff(_d2h(ctx, gpd, N), cgp)
    var dgpr = _diff(_d2h(ctx, gprd, N), cgpr)
    print("  out =", dfo, " g_post =", dgp, " g_prior =", dgpr)
    assert_true(dfo < Scalar[DT](1e-4), "OneHotKL fwd parity")
    assert_true(dgp < Scalar[DT](1e-4), "OneHotKL g_post parity")
    assert_true(dgpr < Scalar[DT](1e-4), "OneHotKL g_prior parity")
    print("  ok")


def main() raises:
    print("=" * 70)
    print("PR5c Step 5 — custom RSSM ops CPU↔GPU parity")
    print("=" * 70)
    test_action_squash()
    test_gru_gate()
    test_bga()
    test_st_sample()
    test_onehot_kl()
    print("=" * 70)
    print("ALL PASSED")
    print("=" * 70)
