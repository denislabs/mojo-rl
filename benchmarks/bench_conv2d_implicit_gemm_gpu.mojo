"""O5 spike: fused implicit-GEMM Conv2D forward (no materialized `col`).

The Conv2D forward is `out[BS,OC] = im2col(x)[BS,COL] @ Wᵀ[OC,COL] + bias`, today
done as THREE ops: `_im2col_kernel` (materialize col[BS,COL]) → `max_matmul` →
scatter/bias. On NVIDIA im2col is bandwidth-bound (~2 TB/s), so the only lever is
removing the `col` buffer traffic entirely. O5 = an implicit GEMM: a tiled GEMM
whose A operand (col) is GATHERED from `input` on the fly (via the O1 shape-static
index table) inside the tile loop, so col is never written or re-read.

This spike measures the crossover honestly:
  baseline = _im2col_kernel + max_matmul[transpose_b] + bias-add   (3 ops)
  O5       = _implicit_gemm_fwd                                    (1 fused kernel)

Expectation: the hand-rolled TILE×TILE SIMT GEMM will NOT match max_matmul's
tensor-core throughput; O5's only saving is the col write (~BS·COL·4 B) + a launch.
So O5-by-hand likely WINS only when max_matmul is on a slow small-N path, and
LOSES where max_matmul shines → conclusion informs whether O5 needs Modular's
structured/`max` conv kernels rather than a hand-roll. Correctness is verified on
a small shape vs a CPU im2col+matmul reference.

Run (NVIDIA = perf truth):
    pixi run -e nvidia mojo run -I . benchmarks/bench_conv2d_implicit_gemm_gpu.mojo
Run (Apple = parity + signal):
    pixi run -e apple  mojo run -I . benchmarks/bench_conv2d_implicit_gemm_gpu.mojo
"""

from std.gpu import global_idx, thread_idx, block_idx, block_dim, barrier
from std.gpu.memory import AddressSpace
from std.gpu.host import DeviceContext, DeviceBuffer
from std.time import perf_counter_ns
from layout import Layout, LayoutTensor, TileTensor, row_major
from linalg.matmul import matmul as max_matmul

from mojo_rl.nn.primitives.conv2d import _im2col_cpu, _im2col_kernel

comptime DT = DType.float32
comptime IT = DType.int32
comptime TPB = 128
comptime TILE = 16


# ── O5: fused implicit-GEMM forward ───────────────────────────────────────
# C[m,n] = Σ_k A[m,k]·W[n,k] + bias[n], m∈[0,BS) n∈[0,OC) k∈[0,COL).
# A[m,k] = gather(input): m→(b,s), k→ck, off=table[s*COL+ck] (−1=pad). W direct.
def _implicit_gemm_fwd[
    IC: Int, K: Int, S: Int, P: Int, H: Int, W: Int, OH: Int, OW: Int,
    IN_FLAT: Int, COL: Int, SO: Int, BS: Int, OC: Int,
](
    input: LayoutTensor[DT, Layout.row_major(BS // SO * IN_FLAT), MutAnyOrigin],
    gather: LayoutTensor[IT, Layout.row_major(SO * COL), MutAnyOrigin],
    weight: LayoutTensor[DT, Layout.row_major(OC, COL), MutAnyOrigin],
    bias: LayoutTensor[DT, Layout.row_major(OC), MutAnyOrigin],
    output: LayoutTensor[DT, Layout.row_major(BS, OC), MutAnyOrigin],
):
    var a_sh = LayoutTensor[
        DT, Layout.row_major(TILE, TILE), MutAnyOrigin,
        address_space=AddressSpace.SHARED,
    ].stack_allocation()
    var b_sh = LayoutTensor[
        DT, Layout.row_major(TILE, TILE), MutAnyOrigin,
        address_space=AddressSpace.SHARED,
    ].stack_allocation()

    var ty = Int(thread_idx.y)
    var tx = Int(thread_idx.x)
    var row = Int(block_idx.y) * TILE + ty  # m (bs)
    var col = Int(block_idx.x) * TILE + tx  # n (oc)
    var acc: Scalar[DT] = 0

    comptime nkt = (COL + TILE - 1) // TILE
    for kt in range(nkt):
        # A-tile: a_sh[ty,tx] = A[row, kt*TILE+tx] = gather(input)
        var k_a = kt * TILE + tx
        var av: Scalar[DT] = 0
        if row < BS and k_a < COL:
            var b = row // SO
            var s = row % SO
            var off = Int(rebind[Scalar[IT]](gather[s * COL + k_a]))
            if off >= 0:
                av = rebind[Scalar[DT]](input[b * IN_FLAT + off])
        a_sh[ty, tx] = av
        # B-tile: b_sh[ty,tx] = Wᵀ[kt*TILE+ty, col] = W[col, kt*TILE+ty]
        var k_b = kt * TILE + ty
        var bv: Scalar[DT] = 0
        if col < OC and k_b < COL:
            bv = rebind[Scalar[DT]](weight[col, k_b])
        b_sh[ty, tx] = bv
        barrier()
        for e in range(TILE):
            acc += rebind[Scalar[DT]](a_sh[ty, e]) * rebind[Scalar[DT]](
                b_sh[e, tx]
            )
        barrier()

    if row < BS and col < OC:
        output[row, col] = acc + rebind[Scalar[DT]](bias[col])


# ── baseline helpers ──────────────────────────────────────────────────────
def _bias_add[BS: Int, OC: Int](
    output: LayoutTensor[DT, Layout.row_major(BS, OC), MutAnyOrigin],
    bias: LayoutTensor[DT, Layout.row_major(OC), MutAnyOrigin],
):
    var idx = Int(global_idx.x)
    if idx >= BS * OC:
        return
    var row = idx // OC
    var oc = idx % OC
    output[row, oc] = rebind[Scalar[DT]](output[row, oc]) + rebind[
        Scalar[DT]
    ](bias[oc])


def _build_gather[
    IC: Int, K: Int, S: Int, P: Int, H: Int, W: Int, OH: Int, OW: Int,
    COL: Int, SO: Int,
](ctx: DeviceContext) raises -> DeviceBuffer[IT]:
    var g = ctx.enqueue_create_buffer[IT](SO * COL)
    with g.map_to_host() as hg:
        for s in range(SO):
            var oh = s // OW
            var ow = s % OW
            for ck in range(COL):
                var ic = ck // (K * K)
                var rem = ck % (K * K)
                var kh = rem // K
                var kw = rem % K
                var ih = oh * S + kh - P
                var iw = ow * S + kw - P
                var v: Int = -1
                if ih >= 0 and ih < H and iw >= 0 and iw < W:
                    v = ic * H * W + ih * W + iw
                hg[s * COL + ck] = Scalar[IT](v)
    return g


# ── correctness (small shape) ─────────────────────────────────────────────
def _verify[
    BATCH: Int, IC: Int, K: Int, S: Int, P: Int, H: Int, W: Int,
    OH: Int, OW: Int, IN_FLAT: Int, COL: Int, SO: Int, BS: Int, OC: Int,
](ctx: DeviceContext) raises -> Float64:
    var inp = ctx.enqueue_create_buffer[DT](BATCH * IN_FLAT)
    var wbuf = ctx.enqueue_create_buffer[DT](OC * COL)
    var bbuf = ctx.enqueue_create_buffer[DT](OC)
    var obuf = ctx.enqueue_create_buffer[DT](BS * OC)

    var in_host = List[Scalar[DT]](length=BATCH * IN_FLAT, fill=Scalar[DT](0))
    var w_host = List[Scalar[DT]](length=OC * COL, fill=Scalar[DT](0))
    var b_host = List[Scalar[DT]](length=OC, fill=Scalar[DT](0))
    with inp.map_to_host() as hi:
        for i in range(BATCH * IN_FLAT):
            var v = Scalar[DT](Float64((i % 97) - 48) * 0.05)
            hi[i] = v
            in_host[i] = v
    with wbuf.map_to_host() as hw:
        for i in range(OC * COL):
            var v = Scalar[DT](Float64((i % 53) - 26) * 0.03)
            hw[i] = v
            w_host[i] = v
    with bbuf.map_to_host() as hb:
        for i in range(OC):
            var v = Scalar[DT](Float64(i) * 0.01)
            hb[i] = v
            b_host[i] = v
    _ = obuf.enqueue_fill(Scalar[DT](-999.0))

    var g = _build_gather[IC, K, S, P, H, W, OH, OW, COL, SO](ctx)
    var inl = LayoutTensor[DT, Layout.row_major(BATCH * IN_FLAT), MutAnyOrigin](inp)
    var gl = LayoutTensor[IT, Layout.row_major(SO * COL), MutAnyOrigin](g)
    var wl = LayoutTensor[DT, Layout.row_major(OC, COL), MutAnyOrigin](wbuf)
    var bl = LayoutTensor[DT, Layout.row_major(OC), MutAnyOrigin](bbuf)
    var ol = LayoutTensor[DT, Layout.row_major(BS, OC), MutAnyOrigin](obuf)
    comptime gx = (OC + TILE - 1) // TILE
    comptime gy = (BS + TILE - 1) // TILE
    ctx.enqueue_function[
        _implicit_gemm_fwd[IC, K, S, P, H, W, OH, OW, IN_FLAT, COL, SO, BS, OC]
    ](inl, gl, wl, bl, ol, grid_dim=(gx, gy), block_dim=(TILE, TILE))
    ctx.synchronize()

    # CPU ref: im2col per sample → out[bs,oc] = Σ_ck col[s,ck]·W[oc,ck] + bias
    var col_s = List[Scalar[DT]](length=SO * COL, fill=Scalar[DT](0))
    var max_abs = Float64(0)
    with obuf.map_to_host() as ho:
        for b in range(BATCH):
            _im2col_cpu[IC, K, S, P, H, W, OH, OW](in_host, b * IN_FLAT, col_s)
            for s in range(SO):
                for oc in range(OC):
                    var acc = Scalar[DT](0)
                    for ck in range(COL):
                        acc += col_s[s * COL + ck] * w_host[oc * COL + ck]
                    acc += b_host[oc]
                    var got = ho[(b * SO + s) * OC + oc]
                    var d = Float64(got - acc)
                    if d < 0:
                        d = -d
                    if d > max_abs:
                        max_abs = d
    return max_abs


# ── timing ────────────────────────────────────────────────────────────────
def _time_baseline[
    BATCH: Int, IC: Int, K: Int, S: Int, P: Int, H: Int, W: Int,
    OH: Int, OW: Int, IN_FLAT: Int, COL: Int, SO: Int, BS: Int, OC: Int,
    WARMUP: Int, ITERS: Int,
](ctx: DeviceContext) raises -> Float64:
    var inp = ctx.enqueue_create_buffer[DT](BATCH * IN_FLAT)
    var colb = ctx.enqueue_create_buffer[DT](BS * COL)
    var wbuf = ctx.enqueue_create_buffer[DT](OC * COL)
    var bbuf = ctx.enqueue_create_buffer[DT](OC)
    var obuf = ctx.enqueue_create_buffer[DT](BS * OC)
    _ = inp.enqueue_fill(Scalar[DT](0.01))
    _ = wbuf.enqueue_fill(Scalar[DT](0.01))
    _ = bbuf.enqueue_fill(Scalar[DT](0.0))
    var inl = LayoutTensor[DT, Layout.row_major(BATCH, IN_FLAT), MutAnyOrigin](inp)
    var coll = LayoutTensor[DT, Layout.row_major(BS, COL), MutAnyOrigin](colb)
    var col_tt = TileTensor(colb, row_major[BS, COL]())
    var w_tt = TileTensor(wbuf, row_major[OC, COL]())
    var outp_tt = TileTensor(obuf, row_major[BS, OC]())
    var bl = LayoutTensor[DT, Layout.row_major(OC), MutAnyOrigin](bbuf)
    var ol = LayoutTensor[DT, Layout.row_major(BS, OC), MutAnyOrigin](obuf)
    comptime nb_col = (BS * COL + TPB - 1) // TPB
    comptime nb_bias = (BS * OC + TPB - 1) // TPB

    @parameter
    def one() raises:
        ctx.enqueue_function[
            _im2col_kernel[
                BATCH, IC, K, S, P, H, W, OH, OW, IN_FLAT, COL, SO, BS
            ]
        ](inl, coll, grid_dim=nb_col, block_dim=TPB)
        max_matmul[transpose_b=True, target="gpu"](outp_tt, col_tt, w_tt, ctx)
        ctx.enqueue_function[_bias_add[BS, OC]](
            ol, bl, grid_dim=nb_bias, block_dim=TPB
        )

    comptime for _ in range(WARMUP):
        one()
    ctx.synchronize()
    var t0 = perf_counter_ns()
    comptime for _ in range(ITERS):
        one()
    ctx.synchronize()
    return Float64(perf_counter_ns() - t0) / Float64(ITERS) / 1000.0


def _time_o5[
    BATCH: Int, IC: Int, K: Int, S: Int, P: Int, H: Int, W: Int,
    OH: Int, OW: Int, IN_FLAT: Int, COL: Int, SO: Int, BS: Int, OC: Int,
    WARMUP: Int, ITERS: Int,
](ctx: DeviceContext) raises -> Float64:
    var inp = ctx.enqueue_create_buffer[DT](BATCH * IN_FLAT)
    var wbuf = ctx.enqueue_create_buffer[DT](OC * COL)
    var bbuf = ctx.enqueue_create_buffer[DT](OC)
    var obuf = ctx.enqueue_create_buffer[DT](BS * OC)
    _ = inp.enqueue_fill(Scalar[DT](0.01))
    _ = wbuf.enqueue_fill(Scalar[DT](0.01))
    _ = bbuf.enqueue_fill(Scalar[DT](0.0))
    var g = _build_gather[IC, K, S, P, H, W, OH, OW, COL, SO](ctx)
    var inl = LayoutTensor[DT, Layout.row_major(BATCH * IN_FLAT), MutAnyOrigin](inp)
    var gl = LayoutTensor[IT, Layout.row_major(SO * COL), MutAnyOrigin](g)
    var wl = LayoutTensor[DT, Layout.row_major(OC, COL), MutAnyOrigin](wbuf)
    var bl = LayoutTensor[DT, Layout.row_major(OC), MutAnyOrigin](bbuf)
    var ol = LayoutTensor[DT, Layout.row_major(BS, OC), MutAnyOrigin](obuf)
    comptime gx = (OC + TILE - 1) // TILE
    comptime gy = (BS + TILE - 1) // TILE

    @parameter
    def one() raises:
        ctx.enqueue_function[
            _implicit_gemm_fwd[
                IC, K, S, P, H, W, OH, OW, IN_FLAT, COL, SO, BS, OC
            ]
        ](inl, gl, wl, bl, ol, grid_dim=(gx, gy), block_dim=(TILE, TILE))

    comptime for _ in range(WARMUP):
        one()
    ctx.synchronize()
    var t0 = perf_counter_ns()
    comptime for _ in range(ITERS):
        one()
    ctx.synchronize()
    return Float64(perf_counter_ns() - t0) / Float64(ITERS) / 1000.0


def run_shape[
    BATCH: Int, IC: Int, K: Int, S: Int, P: Int, H: Int, W: Int, OC: Int,
    WARMUP: Int, ITERS: Int,
](ctx: DeviceContext, label: StaticString) raises:
    comptime OH = (H + 2 * P - K) // S + 1
    comptime OW = (W + 2 * P - K) // S + 1
    comptime IN_FLAT = IC * H * W
    comptime COL = IC * K * K
    comptime SO = OH * OW
    comptime BS = BATCH * SO
    var base = _time_baseline[
        BATCH, IC, K, S, P, H, W, OH, OW, IN_FLAT, COL, SO, BS, OC, WARMUP, ITERS
    ](ctx)
    var o5 = _time_o5[
        BATCH, IC, K, S, P, H, W, OH, OW, IN_FLAT, COL, SO, BS, OC, WARMUP, ITERS
    ](ctx)
    print(
        "  ", label, " BS=", BS, " OC=", OC, " COL=", COL,
        " | baseline(im2col+mm+bias)=", base, "us  O5(fused)=", o5,
        "us  speedup=", base / o5,
    )


def main() raises:
    var ctx = DeviceContext()
    print("O5 spike: fused implicit-GEMM Conv2D forward [fp32]")
    print("=" * 70)
    # correctness on a small shape (B=2, IC=8, 3x3 s1 p1, 6x7, OC=16)
    var err = _verify[2, 8, 3, 1, 1, 6, 7, 6, 7, 8 * 6 * 7, 8 * 9, 42, 84, 16](ctx)
    print("  correctness: max|Δ| vs CPU im2col+matmul =", err)
    print("-" * 70)
    # C4 res tower (hot): IC=64, 3x3 s1 p1, 6x7, OC=64
    run_shape[256, 64, 3, 1, 1, 6, 7, 64, 5, 100](ctx, "C4 res    ")
    # EZv2 deep block spatial 6x6: IC=64, 3x3 s1 p1, OC=64
    run_shape[256, 64, 3, 1, 1, 6, 6, 64, 5, 100](ctx, "EZ deep6  ")
    # Atari mid: IC=32, 4x4 s2 p0, 20x20, OC=64
    run_shape[64, 32, 4, 2, 0, 20, 20, 64, 5, 50](ctx, "Atari mid ")
    print("=" * 70)
    print("speedup>1 → fused beats im2col+max_matmul (col-write + launch saved).")
    print("speedup<1 → hand SIMT GEMM loses to max_matmul tensor-core path;")
    print("            O5 then needs Modular structured/max conv, not a hand-roll.")
