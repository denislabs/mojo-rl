"""Phase-0 go/no-go bench: cached-weight bf16 vs fp32 on a wide GEMM.

Workload = layer-1 forward GEMM [B,IN] @ [IN,OUT], B=256, IN=784, OUT=4096 —
the shape the legacy reported bf16 plateauing on (~0.98x on NVIDIA).

Variants (per-iter steady-state, R timed iters after warmup):
  A  fp32 GEMM                                   (baseline)
  B  bf16: cast x+W EVERY iter, GEMM, cast out   (legacy per-forward recast)
  C  bf16: cast W ONCE, cast x+out per iter, GEMM (cached-weight = the fix)

The W cast is IN*OUT = 3.2M elems — recasting it per-forward (B) is what ate the
legacy speedup; C pays it once. On NVIDIA (tensor cores) C should beat A if the
bf16-GEMM win exceeds the residual x/out cast tax. On Apple there is no bf16 GEMM
speedup, so all bf16 variants are >= A — this run measures the CAST TAX + parity,
not the end-to-end win (that verdict is NVIDIA-only).
"""

from std.time import perf_counter_ns
from std.gpu import global_idx
from std.gpu.host import DeviceContext
from layout import Layout, LayoutTensor, TileTensor, row_major
from linalg.matmul import matmul as max_matmul

from mojo_rl.nn.core.tensor import Tensor, TensorImpl

comptime BF16 = DType.bfloat16
comptime B = 256
comptime IN = 784
comptime OUT = 4096
comptime R = 200       # timed iters
comptime WARM = 20


def _f2b[N: Int](
    src: LayoutTensor[DType.float32, Layout.row_major(N), MutAnyOrigin],
    dst: LayoutTensor[BF16, Layout.row_major(N), MutAnyOrigin],
):
    var i = Int(global_idx.x)
    if i < N:
        dst[i] = src[i].cast[BF16]()


def _b2f[N: Int](
    src: LayoutTensor[BF16, Layout.row_major(N), MutAnyOrigin],
    dst: LayoutTensor[DType.float32, Layout.row_major(N), MutAnyOrigin],
):
    var i = Int(global_idx.x)
    if i < N:
        dst[i] = src[i].cast[DType.float32]()


def _filled_f32(ctx: DeviceContext, n: Int) raises -> Tensor:
    var t = Tensor.alloc(n)
    for i in range(n):
        t.data[i] = Scalar[DType.float32](0.01 * Float32((i % 7) - 3))
    t.upload(ctx)
    return t^


def main() raises:
    var ctx = DeviceContext()
    print(
        "wide GEMM [", B, ",", IN, "] @ [", IN, ",", OUT, "]  R =", R,
        " (W cast =", IN * OUT, "elems)",
    )

    var x = _filled_f32(ctx, B * IN)
    var w = _filled_f32(ctx, IN * OUT)
    var out = Tensor()
    out.ensure_gpu(ctx, B * OUT)
    var x_bf = TensorImpl[BF16]()
    x_bf.ensure_gpu(ctx, B * IN)
    var w_bf = TensorImpl[BF16]()
    w_bf.ensure_gpu(ctx, IN * OUT)
    var o_bf = TensorImpl[BF16]()
    o_bf.ensure_gpu(ctx, B * OUT)

    comptime GX = (B * IN + 255) // 256
    comptime GW = (IN * OUT + 255) // 256
    comptime GO = (B * OUT + 255) // 256

    var xv32 = TileTensor(x.dev.value(), row_major[B, IN]())
    var wv32 = TileTensor(w.dev.value(), row_major[IN, OUT]())
    var ov32 = TileTensor(out.dev.value(), row_major[B, OUT]())
    var xvb = TileTensor(x_bf.dev.value(), row_major[B, IN]())
    var wvb = TileTensor(w_bf.dev.value(), row_major[IN, OUT]())
    var ovb = TileTensor(o_bf.dev.value(), row_major[B, OUT]())

    # ---- A: fp32 ----
    for _ in range(WARM):
        max_matmul[target="gpu"](ov32, xv32, wv32, ctx)
    ctx.synchronize()
    var a0 = perf_counter_ns()
    for _ in range(R):
        max_matmul[target="gpu"](ov32, xv32, wv32, ctx)
    ctx.synchronize()
    var a_us = Float64(perf_counter_ns() - a0) / Float64(R) / 1000.0

    # ---- B: bf16, recast W every iter (legacy) ----
    for _ in range(WARM):
        ctx.enqueue_function[_f2b[B * IN]](
            x.lt["gpu", Layout.row_major(B * IN)](),
            x_bf.lt["gpu", Layout.row_major(B * IN)](),
            grid_dim=GX, block_dim=256)
        ctx.enqueue_function[_f2b[IN * OUT]](
            w.lt["gpu", Layout.row_major(IN * OUT)](),
            w_bf.lt["gpu", Layout.row_major(IN * OUT)](),
            grid_dim=GW, block_dim=256)
        max_matmul[target="gpu"](ovb, xvb, wvb, ctx)
        ctx.enqueue_function[_b2f[B * OUT]](
            o_bf.lt["gpu", Layout.row_major(B * OUT)](),
            out.lt["gpu", Layout.row_major(B * OUT)](),
            grid_dim=GO, block_dim=256)
    ctx.synchronize()
    var b0 = perf_counter_ns()
    for _ in range(R):
        ctx.enqueue_function[_f2b[B * IN]](
            x.lt["gpu", Layout.row_major(B * IN)](),
            x_bf.lt["gpu", Layout.row_major(B * IN)](),
            grid_dim=GX, block_dim=256)
        ctx.enqueue_function[_f2b[IN * OUT]](
            w.lt["gpu", Layout.row_major(IN * OUT)](),
            w_bf.lt["gpu", Layout.row_major(IN * OUT)](),
            grid_dim=GW, block_dim=256)
        max_matmul[target="gpu"](ovb, xvb, wvb, ctx)
        ctx.enqueue_function[_b2f[B * OUT]](
            o_bf.lt["gpu", Layout.row_major(B * OUT)](),
            out.lt["gpu", Layout.row_major(B * OUT)](),
            grid_dim=GO, block_dim=256)
    ctx.synchronize()
    var b_us = Float64(perf_counter_ns() - b0) / Float64(R) / 1000.0

    # ---- C: bf16, cache W cast ONCE (the fix) ----
    ctx.enqueue_function[_f2b[IN * OUT]](
        w.lt["gpu", Layout.row_major(IN * OUT)](),
        w_bf.lt["gpu", Layout.row_major(IN * OUT)](),
        grid_dim=GW, block_dim=256)
    for _ in range(WARM):
        ctx.enqueue_function[_f2b[B * IN]](
            x.lt["gpu", Layout.row_major(B * IN)](),
            x_bf.lt["gpu", Layout.row_major(B * IN)](),
            grid_dim=GX, block_dim=256)
        max_matmul[target="gpu"](ovb, xvb, wvb, ctx)
        ctx.enqueue_function[_b2f[B * OUT]](
            o_bf.lt["gpu", Layout.row_major(B * OUT)](),
            out.lt["gpu", Layout.row_major(B * OUT)](),
            grid_dim=GO, block_dim=256)
    ctx.synchronize()
    var c0 = perf_counter_ns()
    for _ in range(R):
        ctx.enqueue_function[_f2b[B * IN]](
            x.lt["gpu", Layout.row_major(B * IN)](),
            x_bf.lt["gpu", Layout.row_major(B * IN)](),
            grid_dim=GX, block_dim=256)
        max_matmul[target="gpu"](ovb, xvb, wvb, ctx)
        ctx.enqueue_function[_b2f[B * OUT]](
            o_bf.lt["gpu", Layout.row_major(B * OUT)](),
            out.lt["gpu", Layout.row_major(B * OUT)](),
            grid_dim=GO, block_dim=256)
    ctx.synchronize()
    var c_us = Float64(perf_counter_ns() - c0) / Float64(R) / 1000.0

    # ---- isolated W-cast cost (the per-forward tax in B) ----
    for _ in range(WARM):
        ctx.enqueue_function[_f2b[IN * OUT]](
            w.lt["gpu", Layout.row_major(IN * OUT)](),
            w_bf.lt["gpu", Layout.row_major(IN * OUT)](),
            grid_dim=GW, block_dim=256)
    ctx.synchronize()
    var wc0 = perf_counter_ns()
    for _ in range(R):
        ctx.enqueue_function[_f2b[IN * OUT]](
            w.lt["gpu", Layout.row_major(IN * OUT)](),
            w_bf.lt["gpu", Layout.row_major(IN * OUT)](),
            grid_dim=GW, block_dim=256)
    ctx.synchronize()
    var wc_us = Float64(perf_counter_ns() - wc0) / Float64(R) / 1000.0

    print("A  fp32 GEMM                : ", a_us, "us/iter")
    print("B  bf16 recast-W-every-iter : ", b_us, "us/iter  (", b_us / a_us, "x vs A)")
    print("C  bf16 cached-W            : ", c_us, "us/iter  (", c_us / a_us, "x vs A)")
    print("   isolated W-cast kernel   : ", wc_us, "us/iter (the tax C removes)")
