"""BatchNorm2D + MaxPool2D — NCHW vs NHWC parity (perf + correctness).

De-risk spike for the channels_last (NHWC) conv-stack refactor. The forward
im2col NHWC win (1.65× on EZv2 rep48, neutral elsewhere) is already proven in
`bench_conv2d_im2col_col2im_gpu.mojo`; the OPEN question is whether the OTHER
spatial ops hold NVIDIA parity once channels-last makes the channel the inner
(contiguous) axis. The two that matter are BatchNorm2D (a per-channel reduction
over N×H×W) and MaxPool2D (a per-window reduction). This bench answers GO/NO-GO.

Key access-pattern facts (why each layout coalesces when written correctly):
  • BN reduction is over (N, spatial) PER CHANNEL.
    - NCHW: spatial is contiguous within a channel → block-per-channel, threads
      stride spatial → coalesced (the current conv2d/batch_norm_2d design).
    - NHWC: channel is contiguous, spatial is strided → the reduction must be
      TRANSPOSED: thread-per-channel, loop over rows; consecutive threads read
      consecutive channels = one coalesced transaction per row. (cuDNN's NHWC BN
      pattern.) A naive NCHW-style port (block-per-channel, stride-C reads) would
      be uncoalesced — so we DON'T do that; we test the correct transpose.
  • MaxPool is thread-per-output-element in both; map consecutive threads to the
    contiguous output dim (W in NCHW, C in NHWC) → coalesced either way.

Both are MEMORY-BOUND (stream the activation once), so if both are written
coalesced the layouts should land at ~parity. This bench measures whether that
holds on real silicon.

Verify: one logical tensor [N,C,H,W], laid out as NCHW and NHWC; check the two
layouts agree (BN per-channel mean/inv_std + normalized output; pooled output).
Forward only — backward shares the access structure, so forward parity implies it.

Run (NVIDIA = perf truth):
    pixi run -e nvidia mojo run -I . benchmarks/bench_bn_pool_nhwc_parity_gpu.mojo
Run (Apple = parity only):
    pixi run -e apple  mojo run -I . benchmarks/bench_bn_pool_nhwc_parity_gpu.mojo
"""

from std.math import sqrt
from std.gpu import global_idx, thread_idx, block_idx, block_dim
from std.gpu.primitives import block
from std.gpu.host import DeviceContext, DeviceBuffer
from std.time import perf_counter_ns
from layout import Layout, LayoutTensor

comptime DT = DType.float32
comptime TPB = 128        # block-per-channel reduction width (NCHW) / flat launches
comptime NCHUNK = 512     # NHWC partial-reduction row chunks (parallelism source)
comptime EPS = 1e-5


# ══════════════════════════════════════════════════════════════════════════
# BatchNorm2D — NCHW (current design): block per channel, threads stride spatial
# ══════════════════════════════════════════════════════════════════════════
# stats: one block per channel reduces Σx, Σx² over (N, spatial) → mean, inv_std.
# Reads input[b, c*SP + s] — spatial contiguous within a channel → coalesced.
def _bn_stats_nchw[
    N: Int, C: Int, SP: Int, FLAT: Int,
](
    input: LayoutTensor[DT, Layout.row_major(N, FLAT), MutAnyOrigin],
    mean_out: LayoutTensor[DT, Layout.row_major(C), MutAnyOrigin],
    inv_std_out: LayoutTensor[DT, Layout.row_major(C), MutAnyOrigin],
):
    var c = Int(block_idx.x)
    if c >= C:
        return
    var t = Int(thread_idx.x)
    var c_off = c * SP
    var my_sum: Scalar[DT] = 0.0
    var my_sumsq: Scalar[DT] = 0.0
    for b in range(N):
        var s = t
        while s < SP:
            var x = rebind[Scalar[DT]](input[b, c_off + s])
            my_sum += x
            my_sumsq += x * x
            s += TPB
    var bsum = block.sum[block_size=TPB, broadcast=False](val=my_sum)
    var bsq = block.sum[block_size=TPB, broadcast=False](val=my_sumsq)
    if t == 0:
        var inv_n = Scalar[DT](1.0) / Scalar[DT](Float32(N * SP))
        var mean = bsum[0] * inv_n
        var var_ = bsq[0] * inv_n - mean * mean
        if var_ < Scalar[DT](0.0):
            var_ = Scalar[DT](0.0)
        mean_out[c] = mean
        inv_std_out[c] = Scalar[DT](1.0) / sqrt(var_ + Scalar[DT](EPS))


# ── REAL NCHW design: G-grouped partial + finalize (matches batch_norm_2d.mojo) ─
# The strawman `_bn_stats_nchw` above launches grid=C (32–64 blocks) → grossly
# under-utilizes the GPU, making NHWC look 3–5× faster than it really is. The
# PRODUCTION BatchNorm2D splits the batch into G groups → grid = C*G (≈2–4k
# blocks), coalesced AND well-parallelized. THIS is the honest NCHW baseline to
# compare NHWC against.
def _bn_stats_nchw_grouped[
    N: Int, C: Int, SP: Int, FLAT: Int, G: Int,
](
    input: LayoutTensor[DT, Layout.row_major(N, FLAT), MutAnyOrigin],
    partial_sum: LayoutTensor[DT, Layout.row_major(C * G), MutAnyOrigin],
    partial_sumsq: LayoutTensor[DT, Layout.row_major(C * G), MutAnyOrigin],
):
    var blk = Int(block_idx.x)
    var c = blk // G
    var g = blk % G
    if c >= C:
        return
    var t = Int(thread_idx.x)
    var bpb = (N + G - 1) // G
    var b0 = g * bpb
    var b1 = b0 + bpb
    if b1 > N:
        b1 = N
    var c_off = c * SP
    var my_sum: Scalar[DT] = 0.0
    var my_sumsq: Scalar[DT] = 0.0
    for b in range(b0, b1):
        var s = t
        while s < SP:
            var x = rebind[Scalar[DT]](input[b, c_off + s])
            my_sum += x
            my_sumsq += x * x
            s += TPB
    var bsum = block.sum[block_size=TPB, broadcast=False](val=my_sum)
    var bsq = block.sum[block_size=TPB, broadcast=False](val=my_sumsq)
    if t == 0:
        partial_sum[c * G + g] = bsum[0]
        partial_sumsq[c * G + g] = bsq[0]


def _bn_finalize_nchw_grouped[
    C: Int, G: Int, RTOT: Int,
](
    partial_sum: LayoutTensor[DT, Layout.row_major(C * G), MutAnyOrigin],
    partial_sumsq: LayoutTensor[DT, Layout.row_major(C * G), MutAnyOrigin],
    mean_out: LayoutTensor[DT, Layout.row_major(C), MutAnyOrigin],
    inv_std_out: LayoutTensor[DT, Layout.row_major(C), MutAnyOrigin],
):
    var c = Int(global_idx.x)
    if c >= C:
        return
    var s: Scalar[DT] = 0.0
    var sq: Scalar[DT] = 0.0
    for g in range(G):
        s += rebind[Scalar[DT]](partial_sum[c * G + g])
        sq += rebind[Scalar[DT]](partial_sumsq[c * G + g])
    var inv_n = Scalar[DT](1.0) / Scalar[DT](Float32(RTOT))
    var mean = s * inv_n
    var var_ = sq * inv_n - mean * mean
    if var_ < Scalar[DT](0.0):
        var_ = Scalar[DT](0.0)
    mean_out[c] = mean
    inv_std_out[c] = Scalar[DT](1.0) / sqrt(var_ + Scalar[DT](EPS))


# normalize: block per channel, threads stride spatial → coalesced read+write.
def _bn_norm_nchw[
    N: Int, C: Int, SP: Int, FLAT: Int,
](
    input: LayoutTensor[DT, Layout.row_major(N, FLAT), MutAnyOrigin],
    output: LayoutTensor[DT, Layout.row_major(N, FLAT), MutAnyOrigin],
    gamma: LayoutTensor[DT, Layout.row_major(C), MutAnyOrigin],
    beta: LayoutTensor[DT, Layout.row_major(C), MutAnyOrigin],
    mean_in: LayoutTensor[DT, Layout.row_major(C), MutAnyOrigin],
    inv_std_in: LayoutTensor[DT, Layout.row_major(C), MutAnyOrigin],
):
    var c = Int(block_idx.x)
    if c >= C:
        return
    var t = Int(thread_idx.x)
    var c_off = c * SP
    var mean = rebind[Scalar[DT]](mean_in[c])
    var inv_std = rebind[Scalar[DT]](inv_std_in[c])
    var gm = rebind[Scalar[DT]](gamma[c])
    var bt = rebind[Scalar[DT]](beta[c])
    for b in range(N):
        var s = t
        while s < SP:
            var off = c_off + s
            var xh = (rebind[Scalar[DT]](input[b, off]) - mean) * inv_std
            output[b, off] = gm * xh + bt
            s += TPB


# G-grouped normalize (matches the real BN2D: grid=C*G, batch split into G groups).
def _bn_norm_nchw_grouped[
    N: Int, C: Int, SP: Int, FLAT: Int, G: Int,
](
    input: LayoutTensor[DT, Layout.row_major(N, FLAT), MutAnyOrigin],
    output: LayoutTensor[DT, Layout.row_major(N, FLAT), MutAnyOrigin],
    gamma: LayoutTensor[DT, Layout.row_major(C), MutAnyOrigin],
    beta: LayoutTensor[DT, Layout.row_major(C), MutAnyOrigin],
    mean_in: LayoutTensor[DT, Layout.row_major(C), MutAnyOrigin],
    inv_std_in: LayoutTensor[DT, Layout.row_major(C), MutAnyOrigin],
):
    var blk = Int(block_idx.x)
    var c = blk // G
    var g = blk % G
    if c >= C:
        return
    var t = Int(thread_idx.x)
    var bpb = (N + G - 1) // G
    var b0 = g * bpb
    var b1 = b0 + bpb
    if b1 > N:
        b1 = N
    var c_off = c * SP
    var mean = rebind[Scalar[DT]](mean_in[c])
    var inv_std = rebind[Scalar[DT]](inv_std_in[c])
    var gm = rebind[Scalar[DT]](gamma[c])
    var bt = rebind[Scalar[DT]](beta[c])
    for b in range(b0, b1):
        var s = t
        while s < SP:
            var off = c_off + s
            var xh = (rebind[Scalar[DT]](input[b, off]) - mean) * inv_std
            output[b, off] = gm * xh + bt
            s += TPB


# ══════════════════════════════════════════════════════════════════════════
# BatchNorm2D — NHWC (channels-last): TRANSPOSED reduction — thread per channel
# ══════════════════════════════════════════════════════════════════════════
# partial: grid = NCHUNK row-chunks, block_dim = C (one thread per channel). Row
# r = (b*SP+s) is a contiguous C-vector at input[r*C ..]. Thread c accumulates its
# channel over the chunk's rows; consecutive threads (c, c+1) → consecutive
# addresses → one coalesced transaction per row (cuDNN NHWC-BN pattern). A
# production version would tile multiple warps/rows per block for occupancy; this
# 1-warp-per-block (C=32/64) form is the honest baseline — saturated at device
# level by NCHUNK blocks.
def _bn_stats_nhwc_partial[
    C: Int, R: Int,
](
    input: LayoutTensor[DT, Layout.row_major(R * C), MutAnyOrigin],
    partial_sum: LayoutTensor[DT, Layout.row_major(NCHUNK * C), MutAnyOrigin],
    partial_sumsq: LayoutTensor[DT, Layout.row_major(NCHUNK * C), MutAnyOrigin],
):
    var chunk = Int(block_idx.x)
    var c = Int(thread_idx.x)
    if c >= C:
        return
    comptime rpc = (R + NCHUNK - 1) // NCHUNK
    var r0 = chunk * rpc
    if r0 >= R:
        partial_sum[chunk * C + c] = Scalar[DT](0.0)
        partial_sumsq[chunk * C + c] = Scalar[DT](0.0)
        return
    var r1 = r0 + rpc
    if r1 > R:
        r1 = R
    var my_sum: Scalar[DT] = 0.0
    var my_sumsq: Scalar[DT] = 0.0
    for r in range(r0, r1):
        var x = rebind[Scalar[DT]](input[r * C + c])
        my_sum += x
        my_sumsq += x * x
    partial_sum[chunk * C + c] = my_sum
    partial_sumsq[chunk * C + c] = my_sumsq


# finalize: one thread per channel, sum the NCHUNK partials → mean, inv_std.
def _bn_finalize_nhwc[
    C: Int, RTOT: Int,
](
    partial_sum: LayoutTensor[DT, Layout.row_major(NCHUNK * C), MutAnyOrigin],
    partial_sumsq: LayoutTensor[DT, Layout.row_major(NCHUNK * C), MutAnyOrigin],
    mean_out: LayoutTensor[DT, Layout.row_major(C), MutAnyOrigin],
    inv_std_out: LayoutTensor[DT, Layout.row_major(C), MutAnyOrigin],
):
    var c = Int(global_idx.x)
    if c >= C:
        return
    var s: Scalar[DT] = 0.0
    var sq: Scalar[DT] = 0.0
    for k in range(NCHUNK):
        s += rebind[Scalar[DT]](partial_sum[k * C + c])
        sq += rebind[Scalar[DT]](partial_sumsq[k * C + c])
    var inv_n = Scalar[DT](1.0) / Scalar[DT](Float32(RTOT))
    var mean = s * inv_n
    var var_ = sq * inv_n - mean * mean
    if var_ < Scalar[DT](0.0):
        var_ = Scalar[DT](0.0)
    mean_out[c] = mean
    inv_std_out[c] = Scalar[DT](1.0) / sqrt(var_ + Scalar[DT](EPS))


# normalize: flat thread per element (R*C); c = idx % C → coalesced read+write.
def _bn_norm_nhwc[
    C: Int, RC: Int,
](
    input: LayoutTensor[DT, Layout.row_major(RC), MutAnyOrigin],
    output: LayoutTensor[DT, Layout.row_major(RC), MutAnyOrigin],
    gamma: LayoutTensor[DT, Layout.row_major(C), MutAnyOrigin],
    beta: LayoutTensor[DT, Layout.row_major(C), MutAnyOrigin],
    mean_in: LayoutTensor[DT, Layout.row_major(C), MutAnyOrigin],
    inv_std_in: LayoutTensor[DT, Layout.row_major(C), MutAnyOrigin],
):
    var idx = Int(global_idx.x)
    if idx >= RC:
        return
    var c = idx % C
    var xh = (rebind[Scalar[DT]](input[idx]) - rebind[Scalar[DT]](mean_in[c])) \
        * rebind[Scalar[DT]](inv_std_in[c])
    output[idx] = rebind[Scalar[DT]](gamma[c]) * xh + rebind[Scalar[DT]](beta[c])


# ══════════════════════════════════════════════════════════════════════════
# MaxPool2D — thread per output element (coalesced over the contiguous out dim)
# ══════════════════════════════════════════════════════════════════════════
# NCHW: out_pos = c*OSP + oh*OW + ow → consecutive threads = consecutive ow.
def _maxpool_nchw[
    N: Int, C: Int, K: Int, S: Int, P: Int, H: Int, W: Int,
    OH: Int, OW: Int, IN_FLAT: Int, OUT_FLAT: Int,
](
    input: LayoutTensor[DT, Layout.row_major(N, IN_FLAT), MutAnyOrigin],
    output: LayoutTensor[DT, Layout.row_major(N, OUT_FLAT), MutAnyOrigin],
):
    var idx = Int(global_idx.x)
    if idx >= N * OUT_FLAT:
        return
    var b = idx // OUT_FLAT
    var out_pos = idx % OUT_FLAT
    var osp = OH * OW
    var c = out_pos // osp
    var rem = out_pos % osp
    var oh = rem // OW
    var ow = rem % OW
    var c_off = c * H * W
    var best: Scalar[DT] = -3.0e38
    for kh in range(K):
        var ih = oh * S + kh - P
        if ih < 0 or ih >= H:
            continue
        for kw in range(K):
            var iw = ow * S + kw - P
            if iw < 0 or iw >= W:
                continue
            var v = rebind[Scalar[DT]](input[b, c_off + ih * W + iw])
            if v > best:
                best = v
    output[b, out_pos] = best


# NHWC: out_pos = (oh*OW+ow)*C + c → consecutive threads = consecutive c.
def _maxpool_nhwc[
    N: Int, C: Int, K: Int, S: Int, P: Int, H: Int, W: Int,
    OH: Int, OW: Int, IN_FLAT: Int, OUT_FLAT: Int,
](
    input: LayoutTensor[DT, Layout.row_major(N, IN_FLAT), MutAnyOrigin],
    output: LayoutTensor[DT, Layout.row_major(N, OUT_FLAT), MutAnyOrigin],
):
    var idx = Int(global_idx.x)
    if idx >= N * OUT_FLAT:
        return
    var b = idx // OUT_FLAT
    var out_pos = idx % OUT_FLAT
    var c = out_pos % C
    var sp = out_pos // C
    var oh = sp // OW
    var ow = sp % OW
    var best: Scalar[DT] = -3.0e38
    for kh in range(K):
        var ih = oh * S + kh - P
        if ih < 0 or ih >= H:
            continue
        for kw in range(K):
            var iw = ow * S + kw - P
            if iw < 0 or iw >= W:
                continue
            var v = rebind[Scalar[DT]](input[b, (ih * W + iw) * C + c])
            if v > best:
                best = v
    output[b, out_pos] = best


# ══════════════════════════════════════════════════════════════════════════
# host helpers — fill one logical tensor into NCHW + NHWC buffers
# ══════════════════════════════════════════════════════════════════════════
def _fill_layouts[
    N: Int, C: Int, H: Int, W: Int, FLAT: Int,
](
    ctx: DeviceContext,
    nchw: DeviceBuffer[DT],
    nhwc: DeviceBuffer[DT],
) raises:
    var SP = H * W
    with nchw.map_to_host() as hn:
        with nhwc.map_to_host() as hh:
            for n in range(N):
                for c in range(C):
                    for s in range(SP):
                        # distinct, bounded value per logical (n,c,s)
                        var v = Scalar[DT](
                            Float64((((n * C + c) * SP + s) % 1009) + 1) * 0.03
                            - 15.0
                        )
                        hn[n * FLAT + c * SP + s] = v          # NCHW
                        hh[(n * SP + s) * C + c] = v           # NHWC


# ══════════════════════════════════════════════════════════════════════════
# BatchNorm: run both layouts, verify agreement, time each pipeline
# ══════════════════════════════════════════════════════════════════════════
def run_bn[
    N: Int, C: Int, H: Int, W: Int, WARMUP: Int, ITERS: Int,
](ctx: DeviceContext, label: StaticString) raises:
    comptime SP = H * W
    comptime FLAT = C * SP
    comptime R = N * SP
    comptime RC = R * C
    print(label, " BN  N=", N, " C=", C, " ", H, "x", W, " | R=", R)

    var x_nchw = ctx.enqueue_create_buffer[DT](N * FLAT)
    var x_nhwc = ctx.enqueue_create_buffer[DT](RC)
    _fill_layouts[N, C, H, W, FLAT](ctx, x_nchw, x_nhwc)
    var out_nchw = ctx.enqueue_create_buffer[DT](N * FLAT)
    var out_nhwc = ctx.enqueue_create_buffer[DT](RC)
    var gamma = ctx.enqueue_create_buffer[DT](C)
    var beta = ctx.enqueue_create_buffer[DT](C)
    _ = gamma.enqueue_fill(Scalar[DT](1.3))
    _ = beta.enqueue_fill(Scalar[DT](-0.4))
    var mean_a = ctx.enqueue_create_buffer[DT](C)
    var istd_a = ctx.enqueue_create_buffer[DT](C)
    var mean_b = ctx.enqueue_create_buffer[DT](C)
    var istd_b = ctx.enqueue_create_buffer[DT](C)
    var ps = ctx.enqueue_create_buffer[DT](NCHUNK * C)
    var psq = ctx.enqueue_create_buffer[DT](NCHUNK * C)

    var gl = LayoutTensor[DT, Layout.row_major(C), MutAnyOrigin](gamma)
    var bl = LayoutTensor[DT, Layout.row_major(C), MutAnyOrigin](beta)

    # ---- NCHW pipeline ----
    var xn = LayoutTensor[DT, Layout.row_major(N, FLAT), MutAnyOrigin](x_nchw)
    var on = LayoutTensor[DT, Layout.row_major(N, FLAT), MutAnyOrigin](out_nchw)
    var ma = LayoutTensor[DT, Layout.row_major(C), MutAnyOrigin](mean_a)
    var ia = LayoutTensor[DT, Layout.row_major(C), MutAnyOrigin](istd_a)
    ctx.enqueue_function[_bn_stats_nchw[N, C, SP, FLAT]](
        xn, ma, ia, grid_dim=C, block_dim=TPB)
    ctx.enqueue_function[_bn_norm_nchw[N, C, SP, FLAT]](
        xn, on, gl, bl, ma, ia, grid_dim=C, block_dim=TPB)

    # ---- NHWC pipeline ----
    var xh = LayoutTensor[DT, Layout.row_major(RC), MutAnyOrigin](x_nhwc)
    var oh = LayoutTensor[DT, Layout.row_major(RC), MutAnyOrigin](out_nhwc)
    var mb = LayoutTensor[DT, Layout.row_major(C), MutAnyOrigin](mean_b)
    var ib = LayoutTensor[DT, Layout.row_major(C), MutAnyOrigin](istd_b)
    var psl = LayoutTensor[DT, Layout.row_major(NCHUNK * C), MutAnyOrigin](ps)
    var pql = LayoutTensor[DT, Layout.row_major(NCHUNK * C), MutAnyOrigin](psq)
    var xhr = LayoutTensor[DT, Layout.row_major(R * C), MutAnyOrigin](x_nhwc)
    comptime nb_norm = (RC + TPB - 1) // TPB
    ctx.enqueue_function[_bn_stats_nhwc_partial[C, R]](
        xhr, psl, pql, grid_dim=NCHUNK, block_dim=C)
    ctx.enqueue_function[_bn_finalize_nhwc[C, R]](
        psl, pql, mb, ib, grid_dim=(C + TPB - 1) // TPB, block_dim=TPB)
    ctx.enqueue_function[_bn_norm_nhwc[C, RC]](
        xh, oh, gl, bl, mb, ib, grid_dim=nb_norm, block_dim=TPB)
    ctx.synchronize()

    # ---- verify: per-channel stats + normalized outputs agree ----
    var stat_bad = 0
    with mean_a.map_to_host() as hma:
        with mean_b.map_to_host() as hmb:
            with istd_a.map_to_host() as hia:
                with istd_b.map_to_host() as hib:
                    for c in range(C):
                        if abs(Float64(hma[c] - hmb[c])) > 1e-2:
                            stat_bad += 1
                        if abs(Float64(hia[c] - hib[c])) > 1e-2:
                            stat_bad += 1
    var out_bad = 0
    with out_nchw.map_to_host() as hon:
        with out_nhwc.map_to_host() as hoh:
            for n in range(N):
                for c in range(C):
                    for s in range(SP):
                        var a = hon[n * FLAT + c * SP + s]
                        var bb = hoh[(n * SP + s) * C + c]
                        if abs(Float64(a - bb)) > 1e-2:
                            out_bad += 1
    print("  verify: stat_mismatch=", stat_bad, " out_mismatch=", out_bad)

    # ---- time NCHW ----
    comptime for _ in range(WARMUP):
        ctx.enqueue_function[_bn_stats_nchw[N, C, SP, FLAT]](
            xn, ma, ia, grid_dim=C, block_dim=TPB)
        ctx.enqueue_function[_bn_norm_nchw[N, C, SP, FLAT]](
            xn, on, gl, bl, ma, ia, grid_dim=C, block_dim=TPB)
    ctx.synchronize()
    var t0 = perf_counter_ns()
    comptime for _ in range(ITERS):
        ctx.enqueue_function[_bn_stats_nchw[N, C, SP, FLAT]](
            xn, ma, ia, grid_dim=C, block_dim=TPB)
        ctx.enqueue_function[_bn_norm_nchw[N, C, SP, FLAT]](
            xn, on, gl, bl, ma, ia, grid_dim=C, block_dim=TPB)
    ctx.synchronize()
    var us_nchw = Float64(perf_counter_ns() - t0) / Float64(ITERS) / 1000.0

    # ---- time NHWC ----
    comptime for _ in range(WARMUP):
        ctx.enqueue_function[_bn_stats_nhwc_partial[C, R]](
            xhr, psl, pql, grid_dim=NCHUNK, block_dim=C)
        ctx.enqueue_function[_bn_finalize_nhwc[C, R]](
            psl, pql, mb, ib, grid_dim=(C + TPB - 1) // TPB, block_dim=TPB)
        ctx.enqueue_function[_bn_norm_nhwc[C, RC]](
            xh, oh, gl, bl, mb, ib, grid_dim=nb_norm, block_dim=TPB)
    ctx.synchronize()
    var t1 = perf_counter_ns()
    comptime for _ in range(ITERS):
        ctx.enqueue_function[_bn_stats_nhwc_partial[C, R]](
            xhr, psl, pql, grid_dim=NCHUNK, block_dim=C)
        ctx.enqueue_function[_bn_finalize_nhwc[C, R]](
            psl, pql, mb, ib, grid_dim=(C + TPB - 1) // TPB, block_dim=TPB)
        ctx.enqueue_function[_bn_norm_nhwc[C, RC]](
            xh, oh, gl, bl, mb, ib, grid_dim=nb_norm, block_dim=TPB)
    ctx.synchronize()
    var us_nhwc = Float64(perf_counter_ns() - t1) / Float64(ITERS) / 1000.0

    # ---- time NCHW-grouped: the REAL BatchNorm2D design (grid=C*G) ----
    comptime G = N if N < 64 else 64
    comptime lcg = Layout.row_major(C * G)
    var psg = LayoutTensor[DT, lcg, MutAnyOrigin](ps)   # reuse ps/psq (C*G<=NCHUNK*C)
    var pqg = LayoutTensor[DT, lcg, MutAnyOrigin](psq)
    comptime nb_fin = (C + TPB - 1) // TPB
    comptime for _ in range(WARMUP):
        ctx.enqueue_function[_bn_stats_nchw_grouped[N, C, SP, FLAT, G]](
            xn, psg, pqg, grid_dim=C * G, block_dim=TPB)
        ctx.enqueue_function[_bn_finalize_nchw_grouped[C, G, R]](
            psg, pqg, ma, ia, grid_dim=nb_fin, block_dim=TPB)
        ctx.enqueue_function[_bn_norm_nchw_grouped[N, C, SP, FLAT, G]](
            xn, on, gl, bl, ma, ia, grid_dim=C * G, block_dim=TPB)
    ctx.synchronize()
    var t2 = perf_counter_ns()
    comptime for _ in range(ITERS):
        ctx.enqueue_function[_bn_stats_nchw_grouped[N, C, SP, FLAT, G]](
            xn, psg, pqg, grid_dim=C * G, block_dim=TPB)
        ctx.enqueue_function[_bn_finalize_nchw_grouped[C, G, R]](
            psg, pqg, ma, ia, grid_dim=nb_fin, block_dim=TPB)
        ctx.enqueue_function[_bn_norm_nchw_grouped[N, C, SP, FLAT, G]](
            xn, on, gl, bl, ma, ia, grid_dim=C * G, block_dim=TPB)
    ctx.synchronize()
    var us_nchw_g = Float64(perf_counter_ns() - t2) / Float64(ITERS) / 1000.0

    var gb = 2.0 * Float64(RC) * 4.0 / 1e9   # stream in+out once (norm pass)
    print("  BN NCHW strawman(grid=C):", us_nchw, "us ",
          gb / (us_nchw / 1e6) / 1e3, "TB/s")
    print("  BN NCHW real(G-grouped): ", us_nchw_g, "us ",
          gb / (us_nchw_g / 1e6) / 1e3, "TB/s  <- HONEST baseline")
    print("  BN NHWC transposed:      ", us_nhwc, "us ",
          gb / (us_nhwc / 1e6) / 1e3, "TB/s")
    print("  >> NHWC/realNCHW =", us_nhwc / us_nchw_g,
          "x  (NHWC/strawman =", us_nhwc / us_nchw, "x)")


# ══════════════════════════════════════════════════════════════════════════
# MaxPool: run both layouts, verify agreement, time each
# ══════════════════════════════════════════════════════════════════════════
def run_pool[
    N: Int, C: Int, K: Int, S: Int, P: Int, H: Int, W: Int,
    WARMUP: Int, ITERS: Int,
](ctx: DeviceContext, label: StaticString) raises:
    comptime OH = (H + 2 * P - K) // S + 1
    comptime OW = (W + 2 * P - K) // S + 1
    comptime IN_FLAT = C * H * W
    comptime OUT_FLAT = C * OH * OW
    comptime TOT = N * OUT_FLAT
    print(label, " POOL N=", N, " C=", C, " ", H, "x", W, " K=", K, " S=", S,
          " -> ", OH, "x", OW)

    var x_nchw = ctx.enqueue_create_buffer[DT](N * IN_FLAT)
    var x_nhwc = ctx.enqueue_create_buffer[DT](N * IN_FLAT)
    _fill_layouts[N, C, H, W, IN_FLAT](ctx, x_nchw, x_nhwc)
    var o_nchw = ctx.enqueue_create_buffer[DT](N * OUT_FLAT)
    var o_nhwc = ctx.enqueue_create_buffer[DT](N * OUT_FLAT)

    var xn = LayoutTensor[DT, Layout.row_major(N, IN_FLAT), MutAnyOrigin](x_nchw)
    var onn = LayoutTensor[DT, Layout.row_major(N, OUT_FLAT), MutAnyOrigin](o_nchw)
    var xh = LayoutTensor[DT, Layout.row_major(N, IN_FLAT), MutAnyOrigin](x_nhwc)
    var ohh = LayoutTensor[DT, Layout.row_major(N, OUT_FLAT), MutAnyOrigin](o_nhwc)
    comptime nb = (TOT + TPB - 1) // TPB

    ctx.enqueue_function[
        _maxpool_nchw[N, C, K, S, P, H, W, OH, OW, IN_FLAT, OUT_FLAT]
    ](xn, onn, grid_dim=nb, block_dim=TPB)
    ctx.enqueue_function[
        _maxpool_nhwc[N, C, K, S, P, H, W, OH, OW, IN_FLAT, OUT_FLAT]
    ](xh, ohh, grid_dim=nb, block_dim=TPB)
    ctx.synchronize()

    # verify: pooled outputs agree per logical (n,c,oh,ow)
    var osp = OH * OW
    var bad = 0
    with o_nchw.map_to_host() as hn:
        with o_nhwc.map_to_host() as hh:
            for n in range(N):
                for c in range(C):
                    for s in range(osp):
                        var a = hn[n * OUT_FLAT + c * osp + s]
                        var bb = hh[n * OUT_FLAT + s * C + c]
                        if abs(Float64(a - bb)) > 1e-4:
                            bad += 1
    print("  verify: out_mismatch=", bad)

    comptime for _ in range(WARMUP):
        ctx.enqueue_function[
            _maxpool_nchw[N, C, K, S, P, H, W, OH, OW, IN_FLAT, OUT_FLAT]
        ](xn, onn, grid_dim=nb, block_dim=TPB)
    ctx.synchronize()
    var t0 = perf_counter_ns()
    comptime for _ in range(ITERS):
        ctx.enqueue_function[
            _maxpool_nchw[N, C, K, S, P, H, W, OH, OW, IN_FLAT, OUT_FLAT]
        ](xn, onn, grid_dim=nb, block_dim=TPB)
    ctx.synchronize()
    var us_n = Float64(perf_counter_ns() - t0) / Float64(ITERS) / 1000.0

    comptime for _ in range(WARMUP):
        ctx.enqueue_function[
            _maxpool_nhwc[N, C, K, S, P, H, W, OH, OW, IN_FLAT, OUT_FLAT]
        ](xh, ohh, grid_dim=nb, block_dim=TPB)
    ctx.synchronize()
    var t1 = perf_counter_ns()
    comptime for _ in range(ITERS):
        ctx.enqueue_function[
            _maxpool_nhwc[N, C, K, S, P, H, W, OH, OW, IN_FLAT, OUT_FLAT]
        ](xh, ohh, grid_dim=nb, block_dim=TPB)
    ctx.synchronize()
    var us_h = Float64(perf_counter_ns() - t1) / Float64(ITERS) / 1000.0

    print("  POOL NCHW: ", us_n, "us   POOL NHWC: ", us_h,
          "us   | NHWC/NCHW =", us_h / us_n, "x")


def main() raises:
    var ctx = DeviceContext()
    print("BN2D + MaxPool2D NCHW-vs-NHWC parity [fp32]")
    print("=" * 70)
    # EZv2-Atari rep-net hot shapes (where the conv NHWC win lives).
    run_bn[64, 32, 48, 48, 5, 50](ctx, "EZv2 rep48")
    run_bn[64, 64, 24, 24, 5, 50](ctx, "EZv2 rep24")
    run_bn[256, 64, 6, 7, 5, 50](ctx, "C4 res")
    print("-" * 70)
    # Pool: typical 2x2 s2 downsamples on Atari/DQN-scale maps.
    run_pool[64, 32, 2, 2, 0, 48, 48, 5, 50](ctx, "rep48")
    run_pool[64, 64, 2, 2, 0, 24, 24, 5, 50](ctx, "rep24")
    run_pool[64, 32, 3, 2, 0, 84, 84, 5, 50](ctx, "atari84")
    print("=" * 70)
    print("GO if: all mismatches=0 AND NHWC/NCHW ~<=1.2x on the hot shapes.")
    print("Perf truth = NVIDIA; Apple is parity-only.")
