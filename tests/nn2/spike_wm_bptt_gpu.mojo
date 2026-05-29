"""PR5c Step 5 GPU — WM-BPTT scan trains on Metal (trainer orchestration).

The GPU analog of `spike_wm_bptt`: the T-step WM-BPTT (encode → core scan
with carry threading → head losses → recompute-in-backward → seed assembly →
per-module DreamerOpt step) entirely on GPU, using device buffers + the
GPU graphs/ops. Time-major device layouts make per-t slices contiguous
offsets; only 2 marshalling kernels (contiguous copy + seed assembly).
Gate: total WM loss DECREASES over N steps (no jax fixture — orchestration).

Run: `pixi run -e apple mojo run -I . tests/nn2/spike_wm_bptt_gpu.mojo`
"""

from std.gpu import global_idx
from std.gpu.host import DeviceContext, DeviceBuffer
from std.testing import assert_true
from layout import Layout, LayoutTensor, TileTensor, row_major

from mojo_rl.nn2.constants import DT, TPB
from mojo_rl.nn2.initializer import Kaiming
from mojo_rl.nn2.optimizer.dreamer_opt import DreamerOpt
from mojo_rl.deep_agents2.dreamerv3.wm import (
    WMCoreGraph, DecLossGraph, RewLossGraph, ConLossGraph,
)
from mojo_rl.deep_agents2.dreamerv3.nets import DreamerEncoder

comptime B = 2
comptime T = 3
comptime OBS = 3
comptime ACT = 1
comptime DETER = 16
comptime H = 12
comptime STOCH = 3
comptime CLASSES = 5
comptime BLOCKS = 4
comptime TOKEN = 8
comptime DEC_U = 8
comptime HU = 8
comptime BINS = 7
comptime SC = STOCH * CLASSES
comptime CARRY = 2 + DETER + SC
comptime DYN = Scalar[DT](1.0)
comptime REP = Scalar[DT](0.1)

comptime Enc = DreamerEncoder[OBS, TOKEN]
comptime Core = WMCoreGraph[DETER, H, STOCH, CLASSES, BLOCKS, ACT, TOKEN]
comptime Dec = DecLossGraph[SC, DETER, OBS, DEC_U]
comptime Rew = RewLossGraph[DETER, SC, HU, BINS]
comptime Con = ConLossGraph[DETER, SC, HU]


@always_inline
def _p(b: DeviceBuffer[DT]) -> UnsafePointer[Scalar[DT], MutAnyOrigin]:
    return rebind[UnsafePointer[Scalar[DT], MutAnyOrigin]](b.unsafe_ptr())


@always_inline
def _lt[N: Int](p: UnsafePointer[Scalar[DT], MutAnyOrigin]) -> LayoutTensor[DT, Layout.row_major(N), MutAnyOrigin]:
    return LayoutTensor[DT, Layout.row_major(N), MutAnyOrigin](p)


def _copyk[N: Int](
    src: LayoutTensor[DT, Layout.row_major(N), MutAnyOrigin],
    dst: LayoutTensor[DT, Layout.row_major(N), MutAnyOrigin],
):
    var i = Int(global_idx.x)
    if i < N:
        dst[i] = rebind[Scalar[DT]](src[i])


def _seed_asm[B_: Int, CARRY_: Int, D_: Int, SC_: Int](
    seed: LayoutTensor[DT, Layout.row_major(B_ * CARRY_), MutAnyOrigin],
    gcd: LayoutTensor[DT, Layout.row_major(B_ * D_), MutAnyOrigin],
    gcs: LayoutTensor[DT, Layout.row_major(B_ * SC_), MutAnyOrigin],
    dnd: LayoutTensor[DT, Layout.row_major(B_ * D_), MutAnyOrigin],
    rnd: LayoutTensor[DT, Layout.row_major(B_ * D_), MutAnyOrigin],
    cnd: LayoutTensor[DT, Layout.row_major(B_ * D_), MutAnyOrigin],
    dsn: LayoutTensor[DT, Layout.row_major(B_ * SC_), MutAnyOrigin],
    rsn: LayoutTensor[DT, Layout.row_major(B_ * SC_), MutAnyOrigin],
    csn: LayoutTensor[DT, Layout.row_major(B_ * SC_), MutAnyOrigin],
    dyn: Scalar[DT],
    rep: Scalar[DT],
):
    var b = Int(global_idx.x)
    if b < B_:
        seed[b * CARRY_] = dyn
        seed[b * CARRY_ + 1] = rep
        for k in range(D_):
            seed[b * CARRY_ + 2 + k] = (
                rebind[Scalar[DT]](gcd[b * D_ + k])
                + rebind[Scalar[DT]](dnd[b * D_ + k])
                + rebind[Scalar[DT]](rnd[b * D_ + k])
                + rebind[Scalar[DT]](cnd[b * D_ + k])
            )
        for k in range(SC_):
            seed[b * CARRY_ + 2 + D_ + k] = (
                rebind[Scalar[DT]](gcs[b * SC_ + k])
                + rebind[Scalar[DT]](dsn[b * SC_ + k])
                + rebind[Scalar[DT]](rsn[b * SC_ + k])
                + rebind[Scalar[DT]](csn[b * SC_ + k])
            )


def _pseudo_h(ctx: DeviceContext, dev: DeviceBuffer[DT], n: Int, seed: Int) raises:
    var h = ctx.enqueue_create_host_buffer[DT](n)
    ctx.synchronize()
    var s = UInt64(seed * 2654435761 + 12345)
    for i in range(n):
        s = s * UInt64(6364136223846793005) + UInt64(1442695040888963407)
        h.unsafe_ptr()[i] = Scalar[DT]((Float64((s >> 33)) / Float64(UInt64(1) << 31)) - 1.0)
    ctx.enqueue_copy(dev, h)
    ctx.synchronize()


def _fill(ctx: DeviceContext, dev: DeviceBuffer[DT], n: Int, v: Scalar[DT]) raises:
    var h = ctx.enqueue_create_host_buffer[DT](n)
    ctx.synchronize()
    for i in range(n):
        h.unsafe_ptr()[i] = v
    ctx.enqueue_copy(dev, h)
    ctx.synchronize()


def _d2h_sum(ctx: DeviceContext, dev: DeviceBuffer[DT], n: Int) raises -> Scalar[DT]:
    var h = ctx.enqueue_create_host_buffer[DT](n)
    ctx.synchronize()
    ctx.enqueue_copy(h, dev)
    ctx.synchronize()
    var s: Scalar[DT] = 0.0
    for k in range(n):
        s += h.unsafe_ptr()[k]
    return s


def main() raises:
    print("=" * 70)
    print("PR5c Step 5 GPU — WM-BPTT scan trains on Metal")
    print("=" * 70)
    var ctx = DeviceContext()
    var enc = Enc.make["gpu", INIT=Kaiming](ctx=ctx)
    var core = Core.make["gpu", INIT=Kaiming](ctx=ctx)
    var dec = Dec.make["gpu", INIT=Kaiming](ctx=ctx)
    var rew = Rew.make["gpu", INIT=Kaiming](ctx=ctx)
    var con = Con.make["gpu", INIT=Kaiming](ctx=ctx)
    var oe = DreamerOpt.make["gpu", Enc](enc, ctx=ctx)
    var ocore = DreamerOpt.make_graph["gpu"](core, ctx=ctx)
    var odec = DreamerOpt.make_graph["gpu"](dec, ctx=ctx)
    var orew = DreamerOpt.make_graph["gpu"](rew, ctx=ctx)
    var ocon = DreamerOpt.make_graph["gpu"](con, ctx=ctx)
    var lr = Scalar[DT](3e-3)
    oe.lr = lr; ocore.lr = lr; odec.lr = lr; orew.lr = lr; ocon.lr = lr

    # time-major device buffers (per-t slice = contiguous offset)
    var obs = ctx.enqueue_create_buffer[DT](T * B * OBS)
    var act = ctx.enqueue_create_buffer[DT](T * B * ACT)
    var rewt = ctx.enqueue_create_buffer[DT](T * B)
    var cont = ctx.enqueue_create_buffer[DT](T * B)
    _pseudo_h(ctx, obs, T * B * OBS, 1)
    _pseudo_h(ctx, act, T * B * ACT, 2)
    _pseudo_h(ctx, rewt, T * B, 3)
    _fill(ctx, cont, T * B, Scalar[DT](1.0))

    var cdeter = ctx.enqueue_create_buffer[DT]((T + 1) * B * DETER)
    var cstoch = ctx.enqueue_create_buffer[DT]((T + 1) * B * SC)
    var toks = ctx.enqueue_create_buffer[DT](T * B * TOKEN)
    var outbuf = ctx.enqueue_create_buffer[DT](B * CARRY)
    var dl = ctx.enqueue_create_buffer[DT](B)
    var seed = ctx.enqueue_create_buffer[DT](B * CARRY)
    var gcd = ctx.enqueue_create_buffer[DT](B * DETER)
    var gcs = ctx.enqueue_create_buffer[DT](B * SC)
    var ones1 = ctx.enqueue_create_buffer[DT](B)
    var gobs = ctx.enqueue_create_buffer[DT](B * OBS)
    var tokscr = ctx.enqueue_create_buffer[DT](B * TOKEN)
    _fill(ctx, ones1, B, Scalar[DT](1.0))

    comptime CD = B * DETER
    comptime CS = B * SC
    comptime nbD = (CD + TPB - 1) // TPB
    comptime nbS = (CS + TPB - 1) // TPB
    comptime nbB = (B + TPB - 1) // TPB

    # reusable named IO l-values (forward output / vjp grad_inputs require them)
    var out_t = TileTensor(_p(outbuf), row_major[B, CARRY]())
    var dl_t = TileTensor(_p(dl), row_major[B, 1]())
    var seed_t = TileTensor(_p(seed), row_major[B, CARRY]())
    var ones_t = TileTensor(_p(ones1), row_major[B, 1]())
    var tokscr_t = TileTensor(_p(tokscr), row_major[B, TOKEN]())
    var gobs_t = TileTensor(_p(gobs), row_major[B, OBS]())
    comptime ckND = _copyk[CD]
    comptime ckSC = _copyk[CS]
    comptime ksa = _seed_asm[B, CARRY, DETER, SC]

    var first: Scalar[DT] = 0.0
    var last: Scalar[DT] = 0.0
    comptime ITERS = 30
    for it in range(ITERS):
        # ── forward scan ──
        _fill(ctx, cdeter, (T + 1) * B * DETER, Scalar[DT](0.0))   # zero all slots; slot 0 = carry_0
        _fill(ctx, cstoch, (T + 1) * B * SC, Scalar[DT](0.0))
        var total: Scalar[DT] = 0.0
        for t in range(T):
            var obt = _p(obs) + t * B * OBS
            var tkt = _p(toks) + t * B * TOKEN
            var tkt_t = TileTensor(tkt, row_major[B, TOKEN]())
            enc.forward["gpu", B](TileTensor(obt, row_major[B, OBS]()), output=tkt_t)
            var dt = _p(cdeter) + t * B * DETER
            var st = _p(cstoch) + t * B * SC
            core.set_input["deter", B](TileTensor(dt, row_major[B, DETER]()))
            core.set_input["stoch", B](TileTensor(st, row_major[B, SC]()))
            core.set_input["action", B](TileTensor(_p(act) + t * B * ACT, row_major[B, ACT]()))
            core.set_input["tokens", B](TileTensor(tkt, row_major[B, TOKEN]()))
            core.forward["gpu", B](out_t)
            var ndt = _p(cdeter) + (t + 1) * B * DETER
            var snt = _p(cstoch) + (t + 1) * B * SC
            ctx.enqueue_function[ckND](_lt[CD](core.node_out_ptr["nd"]()), _lt[CD](ndt), grid_dim=nbD, block_dim=TPB)
            ctx.enqueue_function[ckSC](_lt[CS](core.node_out_ptr["stoch_new"]()), _lt[CS](snt), grid_dim=nbS, block_dim=TPB)
            dec.set_input["stoch_new", B](TileTensor(snt, row_major[B, SC]()))
            dec.set_input["nd", B](TileTensor(ndt, row_major[B, DETER]()))
            dec.set_input["rtgt", B](TileTensor(obt, row_major[B, OBS]()))
            dec.forward["gpu", B](dl_t)
            ctx.synchronize()
            total += _d2h_sum(ctx, dl, B)
            rew.set_input["nd", B](TileTensor(ndt, row_major[B, DETER]()))
            rew.set_input["stoch_new", B](TileTensor(snt, row_major[B, SC]()))
            rew.set_input["rtgt", B](TileTensor(_p(rewt) + t * B, row_major[B, 1]()))
            rew.forward["gpu", B](dl_t)
            ctx.synchronize()
            total += _d2h_sum(ctx, dl, B)
            con.set_input["nd", B](TileTensor(ndt, row_major[B, DETER]()))
            con.set_input["stoch_new", B](TileTensor(snt, row_major[B, SC]()))
            con.set_input["ctgt", B](TileTensor(_p(cont) + t * B, row_major[B, 1]()))
            con.forward["gpu", B](dl_t)
            ctx.synchronize()
            total += _d2h_sum(ctx, dl, B)

        # ── backward scan ──
        oe.zero_grad["gpu", Enc](enc)
        ocore.zero_grad_graph["gpu"](core)
        odec.zero_grad_graph["gpu"](dec)
        orew.zero_grad_graph["gpu"](rew)
        ocon.zero_grad_graph["gpu"](con)
        _fill(ctx, gcd, B * DETER, Scalar[DT](0.0))
        _fill(ctx, gcs, B * SC, Scalar[DT](0.0))
        for rev in range(T):
            var t = T - 1 - rev
            var dt = _p(cdeter) + t * B * DETER
            var st = _p(cstoch) + t * B * SC
            var ndt = _p(cdeter) + (t + 1) * B * DETER
            var snt = _p(cstoch) + (t + 1) * B * SC
            var obt = _p(obs) + t * B * OBS
            dec.set_input["stoch_new", B](TileTensor(snt, row_major[B, SC]()))
            dec.set_input["nd", B](TileTensor(ndt, row_major[B, DETER]()))
            dec.set_input["rtgt", B](TileTensor(obt, row_major[B, OBS]()))
            dec.forward["gpu", B](dl_t)
            dec.vjp["gpu", B](ones_t)
            rew.set_input["nd", B](TileTensor(ndt, row_major[B, DETER]()))
            rew.set_input["stoch_new", B](TileTensor(snt, row_major[B, SC]()))
            rew.set_input["rtgt", B](TileTensor(_p(rewt) + t * B, row_major[B, 1]()))
            rew.forward["gpu", B](dl_t)
            rew.vjp["gpu", B](ones_t)
            con.set_input["nd", B](TileTensor(ndt, row_major[B, DETER]()))
            con.set_input["stoch_new", B](TileTensor(snt, row_major[B, SC]()))
            con.set_input["ctgt", B](TileTensor(_p(cont) + t * B, row_major[B, 1]()))
            con.forward["gpu", B](dl_t)
            con.vjp["gpu", B](ones_t)
            ctx.enqueue_function[ksa](
                _lt[B * CARRY](_p(seed)),
                _lt[CD](_p(gcd)), _lt[CS](_p(gcs)),
                _lt[CD](dec.grad_input_ptr["nd"]()), _lt[CD](rew.grad_input_ptr["nd"]()), _lt[CD](con.grad_input_ptr["nd"]()),
                _lt[CS](dec.grad_input_ptr["stoch_new"]()), _lt[CS](rew.grad_input_ptr["stoch_new"]()), _lt[CS](con.grad_input_ptr["stoch_new"]()),
                DYN, REP, grid_dim=nbB, block_dim=TPB,
            )
            core.set_input["deter", B](TileTensor(dt, row_major[B, DETER]()))
            core.set_input["stoch", B](TileTensor(st, row_major[B, SC]()))
            core.set_input["action", B](TileTensor(_p(act) + t * B * ACT, row_major[B, ACT]()))
            core.set_input["tokens", B](TileTensor(_p(toks) + t * B * TOKEN, row_major[B, TOKEN]()))
            core.forward["gpu", B](out_t)
            core.vjp["gpu", B](seed_t)
            ctx.enqueue_function[ckND](_lt[CD](core.grad_input_ptr["deter"]()), _lt[CD](_p(gcd)), grid_dim=nbD, block_dim=TPB)
            ctx.enqueue_function[ckSC](_lt[CS](core.grad_input_ptr["stoch"]()), _lt[CS](_p(gcs)), grid_dim=nbS, block_dim=TPB)
            enc.forward["gpu", B](TileTensor(obt, row_major[B, OBS]()), output=tokscr_t)
            var gtok_t = TileTensor(core.grad_input_ptr["tokens"](), row_major[B, TOKEN]())
            enc.vjp["gpu", B](gtok_t, gobs_t)
        oe.step["gpu", Enc](enc)
        ocore.step_graph["gpu"](core)
        odec.step_graph["gpu"](dec)
        orew.step_graph["gpu"](rew)
        ocon.step_graph["gpu"](con)
        ctx.synchronize()
        if it == 0:
            first = total
            print("  iter 0   total WM loss =", total)
        if it == ITERS - 1:
            last = total
            print("  iter", ITERS - 1, " total WM loss =", total)
        assert_true(total == total, "loss finite")

    print("  decrease:", first, "->", last)
    assert_true(last < first, "GPU WM-BPTT loss must decrease")
    print("=" * 70)
    print("PASSED — WM-BPTT scan trains on GPU (loss decreases)")
    print("=" * 70)
