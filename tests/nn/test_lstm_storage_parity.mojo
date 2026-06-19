"""LSTM legacy ↔ storage parity.

Drives the storage LSTM cell (one step) + a short LSTMSeq (T=3) and compares to
the LEGACY LSTM with identical weights / inputs / initial state:
  - CPU: max|Δ| < 1e-6 on outputs, state, and param grads after a backward.
  - GPU: storage-GPU vs storage-CPU, TOL ~2e-5 (same weights/inputs).

Both legacy and storage use the Deterministic `(i%7-3)*0.1` init (legacy
`nn.initializer.Deterministic` ↔ storage `core.initializer.Deterministic`), so
the slabs are bit-identical and the forward/backward should match to fp noise.

Run:
  rm -f mojo_rl.mojoc && pixi run mojo run -I . tests/nn/test_lstm_storage_parity.mojo
  rm -f mojo_rl.mojoc && pixi run -e apple mojo run -I . tests/nn/test_lstm_storage_parity.mojo
"""

from std.memory import alloc
from std.testing import assert_true
from std.gpu.host import DeviceContext
from layout import TileTensor, row_major

from mojo_rl.nn.constants import DT
from mojo_rl.nn.primitives.lstm_cell import LSTMCell as LegacyCell
from mojo_rl.nn.primitives.lstm_seq import LSTMSeq as LegacySeq
from mojo_rl.nn.initializer import Zero as LegacyZero
from mojo_rl.nn.core.module import mptr

from mojo_rl.nn.storage.core.tensor import Tensor
from mojo_rl.nn.storage.core.tensor_refs import TensorRefs
from mojo_rl.nn.storage.core.initializer import Deterministic
from mojo_rl.nn.storage.primitives.lstm_cell import LSTMCell
from mojo_rl.nn.storage.primitives.lstm_seq import LSTMSeq


comptime BATCH = 2
comptime IN_ = 3
comptime H = 4
comptime V = 4
comptime SEQ = 3


# ──────────────────────────────────────────────────────────────────────
# Cell: storage CPU ↔ legacy CPU (one step forward + backward).
# ──────────────────────────────────────────────────────────────────────


def test_cell_cpu_parity() raises:
    print("test_cell_cpu_parity ...", end=" ")
    comptime Lcell = LegacyCell[IN_, H]
    comptime Scell = LSTMCell[IN_, H]
    var leg = Lcell.make[target="cpu", INIT=LegacyZero]()
    var sto = Scell.make[target="cpu", INIT=Deterministic]()
    # Make legacy weights bit-identical to storage Deterministic (i%7-3)*0.1.
    for i in range(Scell.W_IH_SIZE):
        leg.W_ih.val.cpu[i] = Scalar[DT]((i % 7) - 3) * 0.1
    for i in range(Scell.W_HH_SIZE):
        leg.W_hh.val.cpu[i] = Scalar[DT]((i % 7) - 3) * 0.1

    # Inputs (deterministic).
    var x = List[Scalar[DT]](length=BATCH * IN_, fill=0.0)
    var hp = List[Scalar[DT]](length=BATCH * H, fill=0.0)
    var cp = List[Scalar[DT]](length=BATCH * H, fill=0.0)
    var goh = List[Scalar[DT]](length=BATCH * H, fill=0.0)
    var goc = List[Scalar[DT]](length=BATCH * H, fill=0.0)
    for i in range(BATCH * IN_):
        x[i] = Scalar[DT](-0.3 + 0.17 * Float64(i))
    for i in range(BATCH * H):
        hp[i] = Scalar[DT](0.1 - 0.09 * Float64(i))
        cp[i] = Scalar[DT](-0.2 + 0.05 * Float64(i))
        goh[i] = Scalar[DT](0.4 + 0.1 * Float64(i))
        goc[i] = Scalar[DT](0.2 - 0.06 * Float64(i))

    # ----- legacy forward+backward (TileTensor views over alloc'd ptrs) -----
    var lx = _aa(BATCH * IN_)
    var lhp = _aa(BATCH * H)
    var lcp = _aa(BATCH * H)
    var lgoh = _aa(BATCH * H)
    var lgoc = _aa(BATCH * H)
    for i in range(BATCH * IN_):
        lx[i] = x[i]
    for i in range(BATCH * H):
        lhp[i] = hp[i]; lcp[i] = cp[i]; lgoh[i] = goh[i]; lgoc[i] = goc[i]
    var lht = _aa(BATCH * H)
    var lct = _aa(BATCH * H)
    var lcache = _aa(BATCH * Lcell.CACHE_SIZE)
    var ldx = _aa(BATCH * IN_)
    var ldhp = _aa(BATCH * H)
    var ldcp = _aa(BATCH * H)
    var lx_t = TileTensor(lx, row_major[BATCH, IN_]())
    var lhp_t = TileTensor(lhp, row_major[BATCH, H]())
    var lcp_t = TileTensor(lcp, row_major[BATCH, H]())
    var lht_t = TileTensor(lht, row_major[BATCH, H]())
    var lct_t = TileTensor(lct, row_major[BATCH, H]())
    var lcache_t = TileTensor(lcache, row_major[BATCH, Lcell.CACHE_SIZE]())
    var lgoh_t = TileTensor(lgoh, row_major[BATCH, H]())
    var lgoc_t = TileTensor(lgoc, row_major[BATCH, H]())
    var ldx_t = TileTensor(ldx, row_major[BATCH, IN_]())
    var ldhp_t = TileTensor(ldhp, row_major[BATCH, H]())
    var ldcp_t = TileTensor(ldcp, row_major[BATCH, H]())
    leg.zero_grad["cpu"]()
    leg.step_forward["cpu", BATCH](lx_t, lhp_t, lcp_t, lht_t, lct_t, lcache_t)
    leg.step_backward["cpu", BATCH](
        lgoh_t, lgoc_t, lx_t, lhp_t, lcp_t, lcache_t, ldx_t, ldhp_t, ldcp_t
    )

    # ----- storage forward+backward (Tensor storage cells) -----
    # Merged recurrent state: sh / sc are 2·BATCH·H (prev=slab 0, out=slab 1).
    var sx = _tt(x)
    var sh = Tensor.alloc(2 * BATCH * H)
    var sc = Tensor.alloc(2 * BATCH * H)
    for i in range(BATCH * H):
        sh.data[i] = hp[i]
        sc.data[i] = cp[i]
    var shp = _tt(hp)  # separate prev/c for step_backward (read-only there)
    var scp = _tt(cp)
    var sgoh = _tt(goh)
    var sgoc = _tt(goc)
    var scache = Tensor()
    var sdx = Tensor()
    var sdhp = Tensor()
    var sdcp = Tensor()
    sto.zero_grad["cpu"](None)
    sto.step_forward["cpu", BATCH](
        sx, sh, sc, scache,
        None, 0, 0, 0, BATCH * H, BATCH * H, 0,
    )
    # step_backward reads h_prev/c_prev (slab 0) — pass the prev Tensors.
    sto.step_backward["cpu", BATCH](
        sgoh, sgoc, sx, shp, scp, scache, sdx, sdhp, sdcp
    )

    var m: Scalar[DT] = 0.0
    m = max(m, _cmp_off(lht, sh, BATCH * H, BATCH * H))
    m = max(m, _cmp_off(lct, sc, BATCH * H, BATCH * H))
    m = max(m, _cmp_ptr(ldx, sdx, BATCH * IN_))
    m = max(m, _cmp_ptr(ldhp, sdhp, BATCH * H))
    m = max(m, _cmp_ptr(ldcp, sdcp, BATCH * H))
    # param grads
    m = max(m, _cmp_lists(leg.W_ih.grd.cpu, sto.W_ih.grd.data, Scell.W_IH_SIZE))
    m = max(m, _cmp_lists(leg.W_hh.grd.cpu, sto.W_hh.grd.data, Scell.W_HH_SIZE))
    m = max(m, _cmp_lists(leg.b.grd.cpu, sto.b.grd.data, Scell.B_SIZE))

    lht.free(); lct.free(); lcache.free(); ldx.free(); ldhp.free(); ldcp.free()
    lx.free(); lhp.free(); lcp.free(); lgoh.free(); lgoc.free()
    print("max|Δ| =", m)
    assert_true(m < 1e-6, "cell CPU parity failed: " + String(m))
    print("PASS")


# ──────────────────────────────────────────────────────────────────────
# Seq: storage CPU ↔ legacy CPU (forward + vjp).
# ──────────────────────────────────────────────────────────────────────


def test_seq_cpu_parity() raises:
    print("test_seq_cpu_parity ...", end=" ")
    comptime Lseq = LegacySeq[V, H, SEQ]
    comptime Sseq = LSTMSeq[V, H, SEQ]
    comptime IN = SEQ * V
    comptime OUT = SEQ * H
    var leg = Lseq.make[target="cpu", INIT=LegacyZero]()
    var sto = Sseq.make[target="cpu", INIT=Deterministic]()
    for i in range(Sseq.Cell.W_IH_SIZE):
        leg.cell.W_ih.val.cpu[i] = Scalar[DT]((i % 7) - 3) * 0.1
    for i in range(Sseq.Cell.W_HH_SIZE):
        leg.cell.W_hh.val.cpu[i] = Scalar[DT]((i % 7) - 3) * 0.1

    var inp = List[Scalar[DT]](length=BATCH * IN, fill=0.0)
    var go = List[Scalar[DT]](length=BATCH * OUT, fill=0.0)
    for i in range(BATCH * IN):
        inp[i] = Scalar[DT](-0.5 + 0.07 * Float64(i))
    for i in range(BATCH * OUT):
        go[i] = Scalar[DT](0.3 - 0.04 * Float64(i))

    # ----- legacy forward+vjp (TileTensor surface, raw allocs) -----
    var linp = _aa(BATCH * IN)
    var lgo = _aa(BATCH * OUT)
    var lout = _aa(BATCH * OUT)
    var lgin = _aa(BATCH * IN)
    for i in range(BATCH * IN):
        linp[i] = inp[i]
    for i in range(BATCH * OUT):
        lgo[i] = go[i]
    var lin_tt = TileTensor(linp, row_major[BATCH, IN]())
    var lout_tt = TileTensor(lout, row_major[BATCH, OUT]())
    var lgin_tt = TileTensor(lgin, row_major[BATCH, IN]())
    var lgo_tt = TileTensor(lgo, row_major[BATCH, OUT]())
    leg.forward["cpu", BATCH](lin_tt, output=lout_tt)
    leg.zero_grad["cpu"]()
    leg.vjp["cpu", BATCH](lgo_tt, lgin_tt)

    # ----- storage forward+vjp (Tensor + TensorRefs surface) -----
    var sin = _tt(inp)
    var sgo = _tt(go)
    var sout = Tensor()
    var sgin = Tensor.alloc(BATCH * IN)
    sto.forward["cpu", BATCH](TensorRefs[1](sin), sout)
    sto.zero_grad["cpu"](None)
    sto.vjp["cpu", BATCH](
        TensorRefs[1](sin),
        sgo,
        TensorRefs[1](sgin),
    )

    var m: Scalar[DT] = 0.0
    m = max(m, _cmp_ptr(lout, sout, BATCH * OUT))
    m = max(m, _cmp_ptr(lgin, sgin, BATCH * IN))
    m = max(m, _cmp_lists(leg.cell.W_ih.grd.cpu, sto.cell.W_ih.grd.data, Sseq.Cell.W_IH_SIZE))
    m = max(m, _cmp_lists(leg.cell.W_hh.grd.cpu, sto.cell.W_hh.grd.data, Sseq.Cell.W_HH_SIZE))
    m = max(m, _cmp_lists(leg.cell.b.grd.cpu, sto.cell.b.grd.data, Sseq.Cell.B_SIZE))
    linp.free(); lgo.free(); lout.free(); lgin.free()
    print("max|Δ| =", m)
    assert_true(m < 1e-6, "seq CPU parity failed: " + String(m))
    print("PASS")


# ──────────────────────────────────────────────────────────────────────
# GPU: storage GPU vs storage CPU (cell step + seq), TOL 2e-5.
# ──────────────────────────────────────────────────────────────────────


def test_cell_gpu_parity() raises:
    print("test_cell_gpu_parity ...", end=" ")
    try:
        var ctx = DeviceContext()
        comptime Scell = LSTMCell[IN_, H]
        var cpu = Scell.make[target="cpu", INIT=Deterministic]()
        var gpu = Scell.make[target="gpu", INIT=Deterministic](ctx)

        var x = List[Scalar[DT]](length=BATCH * IN_, fill=0.0)
        var hp = List[Scalar[DT]](length=BATCH * H, fill=0.0)
        var cp = List[Scalar[DT]](length=BATCH * H, fill=0.0)
        var goh = List[Scalar[DT]](length=BATCH * H, fill=0.0)
        var goc = List[Scalar[DT]](length=BATCH * H, fill=0.0)
        for i in range(BATCH * IN_):
            x[i] = Scalar[DT](-0.3 + 0.17 * Float64(i))
        for i in range(BATCH * H):
            hp[i] = Scalar[DT](0.1 - 0.09 * Float64(i))
            cp[i] = Scalar[DT](-0.2 + 0.05 * Float64(i))
            goh[i] = Scalar[DT](0.4 + 0.1 * Float64(i))
            goc[i] = Scalar[DT](0.2 - 0.06 * Float64(i))

        # CPU reference (merged h/c slabs).
        var cx = _tt(x); var chp = _tt(hp); var ccp = _tt(cp)
        var ch = Tensor.alloc(2 * BATCH * H); var cc_ = Tensor.alloc(2 * BATCH * H)
        for i in range(BATCH * H):
            ch.data[i] = hp[i]; cc_.data[i] = cp[i]
        var cgoh = _tt(goh); var cgoc = _tt(goc)
        var ccache = Tensor()
        var cdx = Tensor(); var cdhp = Tensor(); var cdcp = Tensor()
        cpu.zero_grad["cpu"](None)
        cpu.step_forward["cpu", BATCH](
            cx, ch, cc_, ccache, None, 0, 0, 0, BATCH * H, BATCH * H, 0
        )
        cpu.step_backward["cpu", BATCH](
            cgoh, cgoc, cx, chp, ccp, ccache, cdx, cdhp, cdcp
        )

        # GPU (merged h/c slabs).
        var gx = _tt_gpu(ctx, x); var ghp = _tt_gpu(ctx, hp)
        var gcp = _tt_gpu(ctx, cp)
        var gh = Tensor.alloc(2 * BATCH * H); var gc_ = Tensor.alloc(2 * BATCH * H)
        for i in range(BATCH * H):
            gh.data[i] = hp[i]; gc_.data[i] = cp[i]
        gh.upload(ctx); gc_.upload(ctx)
        var ggoh = _tt_gpu(ctx, goh); var ggoc = _tt_gpu(ctx, goc)
        var gcache = Tensor()
        var gdx = Tensor(); var gdhp = Tensor(); var gdcp = Tensor()
        gpu.zero_grad["gpu"](ctx)
        gpu.step_forward["gpu", BATCH](
            gx, gh, gc_, gcache, ctx, 0, 0, 0, BATCH * H, BATCH * H, 0
        )
        gpu.step_backward["gpu", BATCH](
            ggoh, ggoc, gx, ghp, gcp, gcache, gdx, gdhp, gdcp, ctx
        )
        ctx.synchronize()

        var m: Scalar[DT] = 0.0
        m = max(m, _cmp_off_dev(ctx, ch, gh, BATCH * H, BATCH * H))
        m = max(m, _cmp_off_dev(ctx, cc_, gc_, BATCH * H, BATCH * H))
        m = max(m, _cmp_dev(ctx, cdx, gdx, BATCH * IN_))
        m = max(m, _cmp_dev(ctx, cdhp, gdhp, BATCH * H))
        m = max(m, _cmp_dev(ctx, cdcp, gdcp, BATCH * H))
        m = max(m, _cmp_param_dev(ctx, cpu.W_ih.grd, gpu.W_ih.grd, Scell.W_IH_SIZE))
        m = max(m, _cmp_param_dev(ctx, cpu.W_hh.grd, gpu.W_hh.grd, Scell.W_HH_SIZE))
        m = max(m, _cmp_param_dev(ctx, cpu.b.grd, gpu.b.grd, Scell.B_SIZE))
        print("max|Δ| =", m)
        assert_true(m < 2e-5, "cell GPU parity failed: " + String(m))
        print("PASS")
    except e:
        print("SKIP (no GPU):", e)


def test_seq_gpu_parity() raises:
    print("test_seq_gpu_parity ...", end=" ")
    try:
        var ctx = DeviceContext()
        comptime Sseq = LSTMSeq[V, H, SEQ]
        comptime IN = SEQ * V
        comptime OUT = SEQ * H
        var cpu = Sseq.make[target="cpu", INIT=Deterministic]()
        var gpu = Sseq.make[target="gpu", INIT=Deterministic](ctx)

        var inp = List[Scalar[DT]](length=BATCH * IN, fill=0.0)
        var go = List[Scalar[DT]](length=BATCH * OUT, fill=0.0)
        for i in range(BATCH * IN):
            inp[i] = Scalar[DT](-0.5 + 0.07 * Float64(i))
        for i in range(BATCH * OUT):
            go[i] = Scalar[DT](0.3 - 0.04 * Float64(i))

        # CPU.
        var cin = _tt(inp); var cgo = _tt(go)
        var cout = Tensor(); var cgin = Tensor.alloc(BATCH * IN)
        cpu.forward["cpu", BATCH](TensorRefs[1](cin), cout)
        cpu.zero_grad["cpu"](None)
        cpu.vjp["cpu", BATCH](
            TensorRefs[1](cin), cgo,
            TensorRefs[1](cgin),
        )

        # GPU.
        var gin_in = _tt_gpu(ctx, inp); var ggo = _tt_gpu(ctx, go)
        var gout = Tensor(); var ggin = Tensor.alloc_gpu(ctx, BATCH * IN)
        gpu.forward["gpu", BATCH](
            TensorRefs[1](gin_in), gout, ctx
        )
        gpu.zero_grad["gpu"](ctx)
        gpu.vjp["gpu", BATCH](
            TensorRefs[1](gin_in), ggo,
            TensorRefs[1](ggin), ctx,
        )
        ctx.synchronize()

        var m: Scalar[DT] = 0.0
        m = max(m, _cmp_dev(ctx, cout, gout, BATCH * OUT))
        m = max(m, _cmp_dev(ctx, cgin, ggin, BATCH * IN))
        m = max(m, _cmp_param_dev(ctx, cpu.cell.W_ih.grd, gpu.cell.W_ih.grd, Sseq.Cell.W_IH_SIZE))
        m = max(m, _cmp_param_dev(ctx, cpu.cell.W_hh.grd, gpu.cell.W_hh.grd, Sseq.Cell.W_HH_SIZE))
        m = max(m, _cmp_param_dev(ctx, cpu.cell.b.grd, gpu.cell.b.grd, Sseq.Cell.B_SIZE))
        print("max|Δ| =", m)
        assert_true(m < 2e-5, "seq GPU parity failed: " + String(m))
        print("PASS")
    except e:
        print("SKIP (no GPU):", e)


# ──────────────────────────────────────────────────────────────────────
# helpers
# ──────────────────────────────────────────────────────────────────────


def _aa(n: Int) -> UnsafePointer[Scalar[DT], MutAnyOrigin]:
    return alloc[Scalar[DT]](n)


def _tt(src: List[Scalar[DT]]) raises -> Tensor:
    var t = Tensor.alloc(len(src))
    for i in range(len(src)):
        t.data[i] = src[i]
    return t^


def _tt_gpu(ctx: DeviceContext, src: List[Scalar[DT]]) raises -> Tensor:
    var t = Tensor.alloc(len(src))
    for i in range(len(src)):
        t.data[i] = src[i]
    t.upload(ctx)
    return t^


def _cmp_ptr(
    a: UnsafePointer[Scalar[DT], MutAnyOrigin], mut b: Tensor, n: Int
) -> Scalar[DT]:
    var m: Scalar[DT] = 0.0
    for i in range(n):
        var d = a[i] - b.data[i]
        m = max(m, d if d >= 0 else -d)
    return m


def _cmp_off(
    a: UnsafePointer[Scalar[DT], MutAnyOrigin], mut b: Tensor, n: Int, boff: Int
) -> Scalar[DT]:
    var m: Scalar[DT] = 0.0
    for i in range(n):
        var d = a[i] - b.data[boff + i]
        m = max(m, d if d >= 0 else -d)
    return m


def _cmp_off_dev(
    ctx: DeviceContext, mut cpu: Tensor, mut gpu: Tensor, n: Int, off: Int
) raises -> Scalar[DT]:
    gpu.download(ctx)
    var m: Scalar[DT] = 0.0
    for i in range(n):
        var d = cpu.data[off + i] - gpu.data[off + i]
        m = max(m, d if d >= 0 else -d)
    return m


def _cmp_lists(a: List[Scalar[DT]], b: List[Scalar[DT]], n: Int) -> Scalar[DT]:
    var m: Scalar[DT] = 0.0
    for i in range(n):
        var d = a[i] - b[i]
        m = max(m, d if d >= 0 else -d)
    return m


def _cmp_dev(ctx: DeviceContext, mut cpu: Tensor, mut gpu: Tensor, n: Int) raises -> Scalar[DT]:
    gpu.download(ctx)
    var m: Scalar[DT] = 0.0
    for i in range(n):
        var d = cpu.data[i] - gpu.data[i]
        m = max(m, d if d >= 0 else -d)
    return m


def _cmp_param_dev(ctx: DeviceContext, mut cpu: Tensor, mut gpu: Tensor, n: Int) raises -> Scalar[DT]:
    gpu.download(ctx)
    var m: Scalar[DT] = 0.0
    for i in range(n):
        var d = cpu.data[i] - gpu.data[i]
        m = max(m, d if d >= 0 else -d)
    return m


def main() raises:
    print("=" * 60)
    print("LSTM storage parity tests")
    print("=" * 60)
    test_cell_cpu_parity()
    test_seq_cpu_parity()
    test_cell_gpu_parity()
    test_seq_gpu_parity()
    print("=" * 60)
    print("ALL PASSED")
    print("=" * 60)
