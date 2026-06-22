"""grad_w backward variants + microbench driver (transpose_a headroom probe).

TEMPORARY / exploratory: this module exists so the heavy `max_matmul` shape
instantiations live inside the cached `mojo_rl` package (`.mojoc`) instead of a
benchmark file (which is recompiled every run). The benchmark just calls the
concrete `run_grad_w_bench(ctx)` driver below.

It quantifies whether a native `max_matmul` `transpose_a` is worth chasing for
the backward grad_w = xᵀ @ go  (x = [B, IN], go = [B, OUT], dW = [IN, OUT]).
`max_matmul` has no `transpose_a` (GPU asserts `not transpose_a`), so the
storage primitives materialise xᵀ with a transpose kernel, run the GEMM, then
accumulate into the param grad with a third kernel. A native `transpose_a`
would collapse all three into one GEMM. Variants timed per shape:

    floor      : max_matmul alone over a pre-transposed cacheT (transpose_a==free)
    Tnaive     : naive one-thread/elem transpose (linear.mojo:_transpose_kernel)
    Ttiled     : 32x8 BLOCK_ROWS shared-mem transpose (B1', transpose_2d.mojo)
    accum      : gw += dW kernel alone
    full_naive : Tnaive + GEMM + accum   (== current production chain)
    full_tiled : Ttiled + GEMM + accum   (drop-in: reuse the B1' transpose)
    fused      : Ttiled + GEMM-with-accumulate-epilogue (no dW_tmp, no accum kernel)

Read-out: (full_* − floor) = headroom a native transpose_a would remove;
fused ~= floor = the in-house fix recovers it WITHOUT transpose_a.
"""

from std.gpu import global_idx, thread_idx, block_idx, barrier
from std.gpu.memory import AddressSpace
from std.gpu.host import DeviceContext
from std.time import perf_counter_ns
from std.utils.index import IndexList
from layout import Layout, LayoutTensor, TileTensor, row_major
from linalg.matmul import matmul as max_matmul

from mojo_rl.nn.constants import DT

comptime TPB = 256
comptime TILE = 32
comptime BR = 8           # 32x8 block, 4 elems/thread (B1')


# ── naive transpose: one thread/elem, strided read (linear.mojo verbatim) ──
def _t_naive[
    ROWS: Int, COLS: Int
](
    src: LayoutTensor[DT, Layout.row_major(ROWS, COLS), MutAnyOrigin],
    dst: LayoutTensor[DT, Layout.row_major(COLS, ROWS), MutAnyOrigin],
):
    var gid = Int(global_idx.x)
    if gid < ROWS * COLS:
        dst[gid % COLS, gid // COLS] = rebind[Scalar[DT]](
            src[gid // COLS, gid % COLS]
        )


# ── tiled transpose: 32x8 BLOCK_ROWS shared-mem tile (B1') ─────────────────
def _t_tiled[
    ROWS: Int, COLS: Int
](
    src: LayoutTensor[DT, Layout.row_major(ROWS, COLS), MutAnyOrigin],
    dst: LayoutTensor[DT, Layout.row_major(COLS, ROWS), MutAnyOrigin],
):
    var tile = LayoutTensor[
        DT,
        Layout.row_major(TILE, TILE + 1),   # +1 pad → no bank conflicts
        MutAnyOrigin,
        address_space=AddressSpace.SHARED,
    ].stack_allocation()

    var cy = Int(block_idx.y) * TILE        # tile origin in ROWS
    var cx = Int(block_idx.x) * TILE        # tile origin in COLS
    var tx = Int(thread_idx.x)              # [0, TILE)
    var ty = Int(thread_idx.y)              # [0, BR)

    var c = cx + tx
    comptime for r in range(0, TILE, BR):
        var rr = cy + ty + r
        if rr < ROWS and c < COLS:
            tile[ty + r, tx] = rebind[Scalar[DT]](src[rr, c])
    barrier()

    var r2 = cy + tx                        # dst col (coalesced, stride 1)
    comptime for r in range(0, TILE, BR):
        var c2 = cx + ty + r                # dst row
        if r2 < ROWS and c2 < COLS:
            dst[c2, r2] = rebind[Scalar[DT]](tile[tx, ty + r])


# ── accumulate: gw += dW (linear.mojo:_accum_kernel verbatim) ──────────────
def _accum[N: Int](
    dst: LayoutTensor[DT, Layout.row_major(N), MutAnyOrigin],
    src: LayoutTensor[DT, Layout.row_major(N), MutAnyOrigin],
):
    var i = Int(global_idx.x)
    if i < N:
        dst[i] += src[i]


def _time[
    B: Int, IN: Int, OUT: Int, MODE: Int, WARMUP: Int, ITERS: Int
](ctx: DeviceContext, label: StaticString) raises:
    var xb = ctx.enqueue_create_buffer[DT](B * IN)
    var cTb = ctx.enqueue_create_buffer[DT](IN * B)
    var gob = ctx.enqueue_create_buffer[DT](B * OUT)
    var dWb = ctx.enqueue_create_buffer[DT](IN * OUT)
    var gwb = ctx.enqueue_create_buffer[DT](IN * OUT)
    _ = xb.enqueue_fill(Scalar[DT](0.01))
    _ = cTb.enqueue_fill(Scalar[DT](0.01))
    _ = gob.enqueue_fill(Scalar[DT](0.01))
    _ = dWb.enqueue_fill(Scalar[DT](0.0))
    _ = gwb.enqueue_fill(Scalar[DT](0.0))

    var xl = LayoutTensor[DT, Layout.row_major(B, IN), MutAnyOrigin](xb)
    var cTl = LayoutTensor[DT, Layout.row_major(IN, B), MutAnyOrigin](cTb)
    var gwl = LayoutTensor[DT, Layout.row_major(IN * OUT), MutAnyOrigin](gwb)
    var dWl = LayoutTensor[DT, Layout.row_major(IN * OUT), MutAnyOrigin](dWb)

    var cT_v = TileTensor(cTb, row_major[IN, B]())
    var go_v = TileTensor(gob, row_major[B, OUT]())
    var dW_v = TileTensor(dWb, row_major[IN, OUT]())

    # accumulate-into-gw epilogue: GEMM hands each result element to this
    # lambda instead of storing to dW → fold gw += val, no temp, no 3rd kernel.
    var gw2d = LayoutTensor[DT, Layout.row_major(IN, OUT), MutAnyOrigin](gwb)

    @parameter
    @always_inline
    def accum_ep[
        dtype: DType, width: SIMDSize, *, alignment: Int = 1
    ](idx: IndexList[2], val: SIMD[dtype, width]):
        var cur = gw2d.load[width=width](idx[0], idx[1])
        gw2d.store[width=width](
            idx[0], idx[1], cur + rebind[SIMD[DT, width]](val)
        )

    comptime nb_t = (B * IN + TPB - 1) // TPB         # naive transpose grid
    comptime nb_a = (IN * OUT + TPB - 1) // TPB        # accum grid
    comptime gx = (B + TILE - 1) // TILE               # tiled: cols = B
    comptime gy = (IN + TILE - 1) // TILE              # tiled: rows = IN

    @parameter
    @always_inline
    def run() raises:
        comptime if MODE == 0:                         # floor: GEMM only
            max_matmul[target="gpu"](dW_v, cT_v, go_v, ctx)
        elif MODE == 1:                                # naive transpose only
            ctx.enqueue_function[_t_naive[B, IN]](
                xl, cTl, grid_dim=nb_t, block_dim=TPB
            )
        elif MODE == 2:                                # tiled transpose only
            ctx.enqueue_function[_t_tiled[B, IN]](
                xl, cTl, grid_dim=(gx, gy), block_dim=(TILE, BR)
            )
        elif MODE == 3:                                # accum only
            ctx.enqueue_function[_accum[IN * OUT]](
                gwl, dWl, grid_dim=nb_a, block_dim=TPB
            )
        elif MODE == 4:                                # full naive chain
            ctx.enqueue_function[_t_naive[B, IN]](
                xl, cTl, grid_dim=nb_t, block_dim=TPB
            )
            max_matmul[target="gpu"](dW_v, cT_v, go_v, ctx)
            ctx.enqueue_function[_accum[IN * OUT]](
                gwl, dWl, grid_dim=nb_a, block_dim=TPB
            )
        elif MODE == 5:                                # full tiled chain
            ctx.enqueue_function[_t_tiled[B, IN]](
                xl, cTl, grid_dim=(gx, gy), block_dim=(TILE, BR)
            )
            max_matmul[target="gpu"](dW_v, cT_v, go_v, ctx)
            ctx.enqueue_function[_accum[IN * OUT]](
                gwl, dWl, grid_dim=nb_a, block_dim=TPB
            )
        else:                                          # fused: tiled + epilogue
            ctx.enqueue_function[_t_tiled[B, IN]](
                xl, cTl, grid_dim=(gx, gy), block_dim=(TILE, BR)
            )
            max_matmul[target="gpu", elementwise_lambda_fn=accum_ep](
                dW_v, cT_v, go_v, ctx
            )

    comptime for _ in range(WARMUP):
        run()
    ctx.synchronize()
    var t0 = perf_counter_ns()
    comptime for _ in range(ITERS):
        run()
    ctx.synchronize()
    var us = Float64(perf_counter_ns() - t0) / Float64(ITERS) / 1000.0

    var gflop = 2.0 * Float64(B) * Float64(IN) * Float64(OUT) / 1e9
    print("    ", label, us, "us/iter ", gflop / (us / 1e6) / 1e3, "TFLOP/s")


def _ab[B: Int, IN: Int, OUT: Int, WARMUP: Int, ITERS: Int](
    ctx: DeviceContext
) raises:
    print("  B=", B, " IN=", IN, " OUT=", OUT)
    _time[B, IN, OUT, 1, WARMUP, ITERS](ctx, "Tnaive    ")
    _time[B, IN, OUT, 2, WARMUP, ITERS](ctx, "Ttiled    ")
    _time[B, IN, OUT, 3, WARMUP, ITERS](ctx, "accum     ")
    _time[B, IN, OUT, 0, WARMUP, ITERS](ctx, "floor GEMM")
    _time[B, IN, OUT, 4, WARMUP, ITERS](ctx, "full_naive")
    _time[B, IN, OUT, 5, WARMUP, ITERS](ctx, "full_tiled")
    _time[B, IN, OUT, 6, WARMUP, ITERS](ctx, "fused     ")


def run_grad_w_bench(ctx: DeviceContext) raises:
    """Concrete driver (all max_matmul instantiations live here, so they cache
    in the package .mojoc instead of recompiling per benchmark run)."""
    print("grad_w backward — transpose+GEMM+accum vs floor [fp32]")
    print("=" * 62)
    print("(full_* - floor) = headroom a native transpose_a would remove;")
    print(" fused ~= floor  = the in-house fix recovers it WITHOUT transpose_a.")
    print("-" * 62)
    # SAC / TD3 / DDPG MLP (hidden 256), square grad_w
    _ab[256, 256, 256, 10, 200](ctx)
    # larger batch (GPU-batched off-policy)
    _ab[1024, 256, 256, 10, 200](ctx)
    # Dreamer-ish hidden 512
    _ab[1024, 512, 512, 10, 100](ctx)
    # skinny grad_w: large B·IN, small OUT (transpose most likely to surface)
    _ab[4096, 256, 64, 10, 100](ctx)
    _ab[4096, 512, 32, 10, 100](ctx)
    print("=" * 62)
