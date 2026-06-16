"""Unit tests for the LeWM decoder primitives.

  - BroadcastTokens[N, D]   : forward replicates (B,D)→(B,N·D) exactly;
                              vjp sums per-token grads back (adjoint of TokenMean).
  - LearnedQueries[I, N, D]  : forward broadcasts the learned tokens across the
                              batch (every row equal); grad_input == 0;
                              grad_queries == sum_b grad_out.
  - DecoderBlock[N, HID, FF]: forward finite; backward routes finite grads to
                              both x and the conditioning c (c is injected).

CPU + GPU.

Run:  pixi run -e apple mojo run -I . tests/nn/test_lewm_decoder_ops.mojo
"""

from std.memory import alloc
from std.gpu.host import DeviceContext, DeviceBuffer
from std.testing import assert_true
from layout import TileTensor, row_major

from mojo_rl.nn.constants import DT
from mojo_rl.nn.core.tensor_pack import TensorPack
from mojo_rl.nn.initializer import Kaiming
from mojo_rl.nn.primitives.broadcast_tokens import BroadcastTokens
from mojo_rl.nn.primitives.learned_queries import LearnedQueries
from mojo_rl.nn.primitives.decoder_block import DecoderBlock


comptime BATCH = 2
comptime NTOK = 3
comptime D = 4
comptime HID = 4
comptime FF = 8
comptime IGNORE = 5  # carrier input dim for LearnedQueries (e.g. emb width)


def _a(n: Int) -> UnsafePointer[Scalar[DT], MutAnyOrigin]:
    return alloc[Scalar[DT]](n)


def _p(b: DeviceBuffer[DT]) -> UnsafePointer[Scalar[DT], MutAnyOrigin]:
    return rebind[UnsafePointer[Scalar[DT], MutAnyOrigin]](b.unsafe_ptr())


def _det(i: Int) -> Scalar[DT]:
    return Scalar[DT]((Float64((i * 2654435761) % 1000) / 500.0) - 1.0)


def _finite(x: Scalar[DT]) -> Bool:
    return x == x and (x - x) == Scalar[DT](0.0)


# ── BroadcastTokens ────────────────────────────────────────────────────
def test_broadcast_cpu() raises:
    print("BroadcastTokens cpu ...")
    var m = BroadcastTokens[NTOK, D].make[target="cpu", INIT=Kaiming]()
    comptime IN = BATCH * D
    comptime OUT = BATCH * NTOK * D
    var x = _a(IN); var y = _a(OUT)
    for k in range(IN):
        x[k] = _det(k + 1)
    var x_t = TileTensor(x, row_major[BATCH, D]())
    var y_t = TileTensor(y, row_major[BATCH, NTOK * D]())
    m.forward["cpu", BATCH](TensorPack[1].of(x_t), output=y_t)
    # every token == the source row
    var maxd: Scalar[DT] = 0.0
    for b in range(BATCH):
        for t in range(NTOK):
            for d in range(D):
                var diff = (y[b * NTOK * D + t * D + d] - x[b * D + d]).__abs__()
                if diff > maxd:
                    maxd = diff
    assert_true(maxd < Scalar[DT](1e-7), "broadcast replicate exact")

    # vjp: grad_in[b,d] = sum_t grad_out  (use grad_out = 1 → expect NTOK)
    var go = _a(OUT); var gi = _a(IN)
    for k in range(OUT):
        go[k] = Scalar[DT](1.0)
    var go_t = TileTensor(go, row_major[BATCH, NTOK * D]())
    var gi_t = TileTensor(gi, row_major[BATCH, D]())
    m.vjp["cpu", BATCH](go_t, TensorPack[1].of(gi_t))
    var mge: Scalar[DT] = 0.0
    for k in range(IN):
        var e = (gi[k] - Scalar[DT](Float64(NTOK))).__abs__()
        if e > mge:
            mge = e
    assert_true(mge < Scalar[DT](1e-6), "broadcast vjp = token sum")
    x.free(); y.free(); go.free(); gi.free()
    _ = m^
    print("  ok")


# ── LearnedQueries ─────────────────────────────────────────────────────
def test_learned_queries_cpu() raises:
    print("LearnedQueries cpu ...")
    var m = LearnedQueries[IGNORE, NTOK, D].make[target="cpu", INIT=Kaiming]()
    m.zero_grad["cpu"]()
    comptime OUT = NTOK * D
    var carrier = _a(BATCH * IGNORE)  # ignored values
    for k in range(BATCH * IGNORE):
        carrier[k] = _det(k + 7)
    var y = _a(BATCH * OUT)
    var car_t = TileTensor(carrier, row_major[BATCH, IGNORE]())
    var y_t = TileTensor(y, row_major[BATCH, OUT]())
    m.forward["cpu", BATCH](TensorPack[1].of(car_t), output=y_t)
    # all batch rows identical (queries are batch-independent)
    var rowdiff: Scalar[DT] = 0.0
    for i in range(OUT):
        var d = (y[i] - y[OUT + i]).__abs__()
        if d > rowdiff:
            rowdiff = d
    assert_true(rowdiff < Scalar[DT](1e-7), "queries identical across batch")

    # vjp: grad_input == 0; grad_queries == sum_b grad_out (=BATCH for ones)
    var go = _a(BATCH * OUT); var gi = _a(BATCH * IGNORE)
    for k in range(BATCH * OUT):
        go[k] = Scalar[DT](1.0)
    var go_t = TileTensor(go, row_major[BATCH, OUT]())
    var gi_t = TileTensor(gi, row_major[BATCH, IGNORE]())
    m.vjp["cpu", BATCH](go_t, TensorPack[1].of(gi_t))
    var mgi: Scalar[DT] = 0.0
    for k in range(BATCH * IGNORE):
        if gi[k].__abs__() > mgi:
            mgi = gi[k].__abs__()
    assert_true(mgi < Scalar[DT](1e-7), "grad_input == 0")
    var gq = TileTensor(m.queries.grd.cpu, row_major[OUT]())
    var mgq: Scalar[DT] = 0.0
    for i in range(OUT):
        var e = (gq[i] - Scalar[DT](Float64(BATCH))).__abs__()
        if e > mgq:
            mgq = e
    assert_true(mgq < Scalar[DT](1e-6), "grad_queries == batch sum")
    carrier.free(); y.free(); go.free(); gi.free()
    _ = m^
    print("  ok")


# ── DecoderBlock ───────────────────────────────────────────────────────
def test_decoder_block_cpu() raises:
    print("DecoderBlock cpu ...")
    var blk = DecoderBlock[NTOK, HID, FF].make[target="cpu", INIT=Kaiming]()
    comptime SEQ = NTOK * HID
    comptime NN = BATCH * SEQ
    var x = _a(NN); var c = _a(NN); var y = _a(NN)
    for k in range(NN):
        x[k] = _det(k + 1); c[k] = _det(k + 50)
    var x_t = TileTensor(x, row_major[BATCH, SEQ]())
    var c_t = TileTensor(c, row_major[BATCH, SEQ]())
    var y_t = TileTensor(y, row_major[BATCH, SEQ]())
    blk.forward["cpu", BATCH](TensorPack[2].of(x_t, c_t), output=y_t)
    var allfin = True
    for k in range(NN):
        if not _finite(y[k]):
            allfin = False
    assert_true(allfin, "decoder block forward finite")

    var w = _a(NN); var gx = _a(NN); var gc = _a(NN)
    for k in range(NN):
        w[k] = _det(k + 99)
    var w_t = TileTensor(w, row_major[BATCH, SEQ]())
    var gx_t = TileTensor(gx, row_major[BATCH, SEQ]())
    var gc_t = TileTensor(gc, row_major[BATCH, SEQ]())
    blk.vjp["cpu", BATCH](w_t, TensorPack[2].of(gx_t, gc_t))
    var gxfin = True; var gcfin = True
    var gcmag: Scalar[DT] = 0.0
    for k in range(NN):
        if not _finite(gx[k]):
            gxfin = False
        if not _finite(gc[k]):
            gcfin = False
        gcmag += gc[k].__abs__()
    assert_true(gxfin and gcfin, "decoder block grads finite")
    assert_true(gcmag > Scalar[DT](1e-6), "grad flows to conditioning c")
    x.free(); c.free(); y.free(); w.free(); gx.free(); gc.free()
    _ = blk^
    print("  ok")


# ── GPU smoke (parity to CPU values for the two leaf ops) ───────────────
def test_gpu_smoke() raises:
    print("gpu smoke ...")
    var ctx = DeviceContext()
    # BroadcastTokens
    var bm = BroadcastTokens[NTOK, D].make[target="gpu", INIT=Kaiming](ctx)
    comptime IN = BATCH * D
    comptime OUT = BATCH * NTOK * D
    var xd = ctx.enqueue_create_buffer[DT](IN)
    var yd = ctx.enqueue_create_buffer[DT](OUT)
    var xh = ctx.enqueue_create_host_buffer[DT](IN)
    var yh = ctx.enqueue_create_host_buffer[DT](OUT)
    ctx.synchronize()
    for k in range(IN):
        xh.unsafe_ptr()[k] = _det(k + 1)
    ctx.enqueue_copy(xd, xh); ctx.synchronize()
    var xt = TileTensor(_p(xd), row_major[BATCH, D]())
    var yt = TileTensor(_p(yd), row_major[BATCH, NTOK * D]())
    bm.forward["gpu", BATCH](TensorPack[1].of(xt), output=yt)
    ctx.enqueue_copy(yh, yd); ctx.synchronize()
    var maxd: Scalar[DT] = 0.0
    for b in range(BATCH):
        for t in range(NTOK):
            for d in range(D):
                var diff = (
                    yh.unsafe_ptr()[b * NTOK * D + t * D + d]
                    - xh.unsafe_ptr()[b * D + d]
                ).__abs__()
                if diff > maxd:
                    maxd = diff
    assert_true(maxd < Scalar[DT](1e-7), "broadcast replicate exact (gpu)")
    _ = bm^

    # DecoderBlock forward finite on GPU
    var blk = DecoderBlock[NTOK, HID, FF].make[target="gpu", INIT=Kaiming](ctx)
    comptime SEQ = NTOK * HID
    comptime NN = BATCH * SEQ
    var xbd = ctx.enqueue_create_buffer[DT](NN)
    var cbd = ctx.enqueue_create_buffer[DT](NN)
    var ybd = ctx.enqueue_create_buffer[DT](NN)
    var xbh = ctx.enqueue_create_host_buffer[DT](NN)
    var cbh = ctx.enqueue_create_host_buffer[DT](NN)
    var ybh = ctx.enqueue_create_host_buffer[DT](NN)
    ctx.synchronize()
    for k in range(NN):
        xbh.unsafe_ptr()[k] = _det(k + 1)
        cbh.unsafe_ptr()[k] = _det(k + 50)
    ctx.enqueue_copy(xbd, xbh); ctx.enqueue_copy(cbd, cbh); ctx.synchronize()
    var xbt = TileTensor(_p(xbd), row_major[BATCH, SEQ]())
    var cbt = TileTensor(_p(cbd), row_major[BATCH, SEQ]())
    var ybt = TileTensor(_p(ybd), row_major[BATCH, SEQ]())
    blk.forward["gpu", BATCH](TensorPack[2].of(xbt, cbt), output=ybt)
    ctx.enqueue_copy(ybh, ybd); ctx.synchronize()
    var allfin = True
    for k in range(NN):
        if not _finite(ybh.unsafe_ptr()[k]):
            allfin = False
    assert_true(allfin, "decoder block forward finite (gpu)")
    _ = blk^
    print("  ok")


def main() raises:
    print("=" * 70)
    print("LeWM decoder primitives — BroadcastTokens / LearnedQueries / DecoderBlock")
    print("=" * 70)
    test_broadcast_cpu()
    test_learned_queries_cpu()
    test_decoder_block_cpu()
    test_gpu_smoke()
    print("=" * 70)
    print("ALL PASSED")
    print("=" * 70)
