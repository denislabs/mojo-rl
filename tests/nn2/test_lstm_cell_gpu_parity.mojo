"""LSTMCell GPU↔CPU parity (Apple/CUDA).

Builds a CPU and a GPU cell with identical params, runs step_forward +
step_backward on both, and checks h_t / c_t / cache / dx / dh_prev /
dc_prev and the param grads (dW_ih, dW_hh, db) match within fp32 noise.
"""

from std.memory import alloc
from std.random import seed
from std.testing import assert_true
from std.gpu.host import DeviceContext, DeviceBuffer
from layout import TileTensor, row_major

from mojo_rl.nn2.constants import DT
from mojo_rl.nn2.primitives.lstm_cell import LSTMCell
from mojo_rl.nn2.initializer import Xavier


comptime BATCH = 3
comptime IN_ = 5
comptime H = 6
comptime Cell = LSTMCell[IN_, H]
comptime ATOL: Scalar[DT] = 5e-5


def _abs(v: Scalar[DT]) -> Scalar[DT]:
    return v if v >= 0 else -v


def _upload(
    ctx: DeviceContext, src: UnsafePointer[Scalar[DT], MutAnyOrigin], n: Int
) raises -> DeviceBuffer[DT]:
    var hb = ctx.enqueue_create_host_buffer[DT](n)
    ctx.synchronize()
    for i in range(n):
        hb.unsafe_ptr()[i] = src[i]
    var d = ctx.enqueue_create_buffer[DT](n)
    ctx.enqueue_copy(d, hb)
    ctx.synchronize()
    return d^


def _download(
    ctx: DeviceContext, d: DeviceBuffer[DT], dst: UnsafePointer[Scalar[DT], MutAnyOrigin], n: Int
) raises:
    var hb = ctx.enqueue_create_host_buffer[DT](n)
    ctx.enqueue_copy(hb, d)
    ctx.synchronize()
    for i in range(n):
        dst[i] = hb.unsafe_ptr()[i]


def main() raises:
    print("test_lstm_cell_gpu_parity ...")
    seed(42)
    var ctx = DeviceContext()
    var cpu = Cell.make[target="cpu", INIT=Xavier]()
    var gpu = Cell.make[target="gpu", INIT=Xavier](ctx)

    # Force identical params: copy CPU param values into the GPU buffers.
    var wih_h = ctx.enqueue_create_host_buffer[DT](Cell.W_IH_SIZE)
    var whh_h = ctx.enqueue_create_host_buffer[DT](Cell.W_HH_SIZE)
    var b_h = ctx.enqueue_create_host_buffer[DT](Cell.B_SIZE)
    ctx.synchronize()
    for i in range(Cell.W_IH_SIZE):
        wih_h.unsafe_ptr()[i] = cpu.W_ih.val.cpu[i]
    for i in range(Cell.W_HH_SIZE):
        whh_h.unsafe_ptr()[i] = cpu.W_hh.val.cpu[i]
    for i in range(Cell.B_SIZE):
        b_h.unsafe_ptr()[i] = cpu.b.val.cpu[i]
    ctx.enqueue_copy(gpu.W_ih.val.dev.value(), wih_h)
    ctx.enqueue_copy(gpu.W_hh.val.dev.value(), whh_h)
    ctx.enqueue_copy(gpu.b.val.dev.value(), b_h)
    ctx.synchronize()

    # Inputs (host).
    var x: UnsafePointer[Scalar[DT], MutAnyOrigin] = alloc[Scalar[DT]](BATCH * IN_)
    var hp: UnsafePointer[Scalar[DT], MutAnyOrigin] = alloc[Scalar[DT]](BATCH * H)
    var cp: UnsafePointer[Scalar[DT], MutAnyOrigin] = alloc[Scalar[DT]](BATCH * H)
    var dh: UnsafePointer[Scalar[DT], MutAnyOrigin] = alloc[Scalar[DT]](BATCH * H)
    var dc: UnsafePointer[Scalar[DT], MutAnyOrigin] = alloc[Scalar[DT]](BATCH * H)
    for i in range(BATCH * IN_):
        x[i] = Scalar[DT](-0.3 + 0.07 * Float64(i))
    for i in range(BATCH * H):
        hp[i] = Scalar[DT](0.1 - 0.03 * Float64(i))
        cp[i] = Scalar[DT](-0.2 + 0.05 * Float64(i))
        dh[i] = Scalar[DT](0.4 + 0.02 * Float64(i))
        dc[i] = Scalar[DT](0.15 - 0.01 * Float64(i))

    # ---- CPU forward + backward ----
    var ht_c: UnsafePointer[Scalar[DT], MutAnyOrigin] = alloc[Scalar[DT]](BATCH * H)
    var ct_c: UnsafePointer[Scalar[DT], MutAnyOrigin] = alloc[Scalar[DT]](BATCH * H)
    var cache_c: UnsafePointer[Scalar[DT], MutAnyOrigin] = alloc[Scalar[DT]](BATCH * Cell.CACHE_SIZE)
    var dx_c: UnsafePointer[Scalar[DT], MutAnyOrigin] = alloc[Scalar[DT]](BATCH * IN_)
    var dhp_c: UnsafePointer[Scalar[DT], MutAnyOrigin] = alloc[Scalar[DT]](BATCH * H)
    var dcp_c: UnsafePointer[Scalar[DT], MutAnyOrigin] = alloc[Scalar[DT]](BATCH * H)
    var x_ct = TileTensor(x, row_major[BATCH, IN_]())
    var hp_ct = TileTensor(hp, row_major[BATCH, H]())
    var cp_ct = TileTensor(cp, row_major[BATCH, H]())
    var dh_ct = TileTensor(dh, row_major[BATCH, H]())
    var dc_ct = TileTensor(dc, row_major[BATCH, H]())
    var ht_ct = TileTensor(ht_c, row_major[BATCH, H]())
    var ct_ct = TileTensor(ct_c, row_major[BATCH, H]())
    var cache_ct = TileTensor(cache_c, row_major[BATCH, Cell.CACHE_SIZE]())
    var dx_ct = TileTensor(dx_c, row_major[BATCH, IN_]())
    var dhp_ct = TileTensor(dhp_c, row_major[BATCH, H]())
    var dcp_ct = TileTensor(dcp_c, row_major[BATCH, H]())
    cpu.zero_grad["cpu"]()
    cpu.step_forward["cpu", BATCH](x_ct, hp_ct, cp_ct, ht_ct, ct_ct, cache_ct)
    cpu.step_backward["cpu", BATCH](
        dh_ct, dc_ct, x_ct, hp_ct, cp_ct, cache_ct, dx_ct, dhp_ct, dcp_ct
    )

    # ---- GPU forward + backward ----
    var xd = _upload(ctx, x, BATCH * IN_)
    var hpd = _upload(ctx, hp, BATCH * H)
    var cpd = _upload(ctx, cp, BATCH * H)
    var dhd = _upload(ctx, dh, BATCH * H)
    var dcd = _upload(ctx, dc, BATCH * H)
    var htd = ctx.enqueue_create_buffer[DT](BATCH * H)
    var ctd = ctx.enqueue_create_buffer[DT](BATCH * H)
    var cached = ctx.enqueue_create_buffer[DT](BATCH * Cell.CACHE_SIZE)
    var dxd = ctx.enqueue_create_buffer[DT](BATCH * IN_)
    var dhpd = ctx.enqueue_create_buffer[DT](BATCH * H)
    var dcpd = ctx.enqueue_create_buffer[DT](BATCH * H)
    ctx.synchronize()

    # Build TileTensors over MutAnyOrigin pointers (device-buffer-backed
    # tensors otherwise carry the buffer's origin, which the step methods'
    # mut args reject).
    var xp: UnsafePointer[Scalar[DT], MutAnyOrigin] = xd.unsafe_ptr()
    var hpp: UnsafePointer[Scalar[DT], MutAnyOrigin] = hpd.unsafe_ptr()
    var cpp: UnsafePointer[Scalar[DT], MutAnyOrigin] = cpd.unsafe_ptr()
    var dhp_p: UnsafePointer[Scalar[DT], MutAnyOrigin] = dhd.unsafe_ptr()
    var dcp_p: UnsafePointer[Scalar[DT], MutAnyOrigin] = dcd.unsafe_ptr()
    var htp: UnsafePointer[Scalar[DT], MutAnyOrigin] = htd.unsafe_ptr()
    var ctp: UnsafePointer[Scalar[DT], MutAnyOrigin] = ctd.unsafe_ptr()
    var ccp: UnsafePointer[Scalar[DT], MutAnyOrigin] = cached.unsafe_ptr()
    var dxp: UnsafePointer[Scalar[DT], MutAnyOrigin] = dxd.unsafe_ptr()
    var dhpp: UnsafePointer[Scalar[DT], MutAnyOrigin] = dhpd.unsafe_ptr()
    var dcpp: UnsafePointer[Scalar[DT], MutAnyOrigin] = dcpd.unsafe_ptr()
    var x_gt = TileTensor(xp, row_major[BATCH, IN_]())
    var hp_gt = TileTensor(hpp, row_major[BATCH, H]())
    var cp_gt = TileTensor(cpp, row_major[BATCH, H]())
    var dh_gt = TileTensor(dhp_p, row_major[BATCH, H]())
    var dc_gt = TileTensor(dcp_p, row_major[BATCH, H]())
    var ht_gt = TileTensor(htp, row_major[BATCH, H]())
    var ct_gt = TileTensor(ctp, row_major[BATCH, H]())
    var cache_gt = TileTensor(ccp, row_major[BATCH, Cell.CACHE_SIZE]())
    var dx_gt = TileTensor(dxp, row_major[BATCH, IN_]())
    var dhp_gt = TileTensor(dhpp, row_major[BATCH, H]())
    var dcp_gt = TileTensor(dcpp, row_major[BATCH, H]())
    gpu.zero_grad["gpu"]()
    gpu.step_forward["gpu", BATCH](x_gt, hp_gt, cp_gt, ht_gt, ct_gt, cache_gt)
    gpu.step_backward["gpu", BATCH](
        dh_gt, dc_gt, x_gt, hp_gt, cp_gt, cache_gt, dx_gt, dhp_gt, dcp_gt
    )
    ctx.synchronize()

    # Download GPU results.
    var ht_g: UnsafePointer[Scalar[DT], MutAnyOrigin] = alloc[Scalar[DT]](BATCH * H)
    var ct_g: UnsafePointer[Scalar[DT], MutAnyOrigin] = alloc[Scalar[DT]](BATCH * H)
    var cache_g: UnsafePointer[Scalar[DT], MutAnyOrigin] = alloc[Scalar[DT]](BATCH * Cell.CACHE_SIZE)
    var dx_g: UnsafePointer[Scalar[DT], MutAnyOrigin] = alloc[Scalar[DT]](BATCH * IN_)
    var dhp_g: UnsafePointer[Scalar[DT], MutAnyOrigin] = alloc[Scalar[DT]](BATCH * H)
    var dcp_g: UnsafePointer[Scalar[DT], MutAnyOrigin] = alloc[Scalar[DT]](BATCH * H)
    var dwih_g: UnsafePointer[Scalar[DT], MutAnyOrigin] = alloc[Scalar[DT]](Cell.W_IH_SIZE)
    var dwhh_g: UnsafePointer[Scalar[DT], MutAnyOrigin] = alloc[Scalar[DT]](Cell.W_HH_SIZE)
    var db_g: UnsafePointer[Scalar[DT], MutAnyOrigin] = alloc[Scalar[DT]](Cell.B_SIZE)
    _download(ctx, htd, ht_g, BATCH * H)
    _download(ctx, ctd, ct_g, BATCH * H)
    _download(ctx, cached, cache_g, BATCH * Cell.CACHE_SIZE)
    _download(ctx, dxd, dx_g, BATCH * IN_)
    _download(ctx, dhpd, dhp_g, BATCH * H)
    _download(ctx, dcpd, dcp_g, BATCH * H)
    _download(ctx, gpu.W_ih.grd.dev.value(), dwih_g, Cell.W_IH_SIZE)
    _download(ctx, gpu.W_hh.grd.dev.value(), dwhh_g, Cell.W_HH_SIZE)
    _download(ctx, gpu.b.grd.dev.value(), db_g, Cell.B_SIZE)

    # ---- Compare ----
    var md: Scalar[DT] = 0.0
    for i in range(BATCH * H):
        md = max(md, _abs(ht_c[i] - ht_g[i]))
        md = max(md, _abs(ct_c[i] - ct_g[i]))
        md = max(md, _abs(dhp_c[i] - dhp_g[i]))
        md = max(md, _abs(dcp_c[i] - dcp_g[i]))
    for i in range(BATCH * Cell.CACHE_SIZE):
        md = max(md, _abs(cache_c[i] - cache_g[i]))
    for i in range(BATCH * IN_):
        md = max(md, _abs(dx_c[i] - dx_g[i]))
    var mg: Scalar[DT] = 0.0
    for i in range(Cell.W_IH_SIZE):
        mg = max(mg, _abs(cpu.W_ih.grd.cpu[i] - dwih_g[i]))
    for i in range(Cell.W_HH_SIZE):
        mg = max(mg, _abs(cpu.W_hh.grd.cpu[i] - dwhh_g[i]))
    for i in range(Cell.B_SIZE):
        mg = max(mg, _abs(cpu.b.grd.cpu[i] - db_g[i]))

    print("  max|fwd/state diff| =", md, "  max|param-grad diff| =", mg)
    assert_true(md < ATOL, "LSTM GPU forward/state/input-grad parity failed")
    assert_true(mg < ATOL, "LSTM GPU param-grad parity failed")
    print("  ok")
