"""Microbench: is the EZv2 rep-ResNet conv GEMM on a slow `max_matmul` path?

The GPU Conv2D forward runs `outp[BS, OC] = col[BS, COL] @ Wᵀ` via
`max_matmul[transpose_b=True]` (conv2d.mojo:675). The GEMM dims are:

    M = BS = BATCH * SPATIAL_OUT   (huge)
    N = OC                         (32 or 64 — SUB-128 on every rep layer)
    K = COL = IC * K * K           (108 or 576)

`max_matmul`'s fast multistage GPU kernel generally wants the output's N
128-aligned; OC=32/64 may fall to a slower fallback. This bench times each
real rep-ResNet shape at the natural OC, then again with OC padded to 128.

Interpretation:
  - If padded-OC (2x the FLOPs) is FASTER or ~equal → unpadded N=OC is on a
    bad path; padding OC to a multiple of 128 is a real fp32 lever
    (independent of AMP), for both rep.forward and reana rep.
  - If padded-OC is ~2x SLOWER (scales with FLOPs) → the kernel is already
    fine at N=64; no shape win, AMP is the only lever. Don't pad.

Run (NVIDIA):
    pixi run -e nvidia mojo run -I . benchmarks/bench_conv_gemm_shapes_gpu.mojo
"""

from std.gpu.host import DeviceContext
from std.time import perf_counter_ns
from layout import TileTensor, row_major
from linalg.matmul import matmul as max_matmul

comptime DT = DType.float32

# (BS, OC, COL) for each rep-ResNet conv layer at BATCH=256.
# stem:       OC=32, spatial 48x48=2304 → BS=256*2304
# downsample: OC=64, spatial 24x24=576  → BS=256*576
# deep 12x12: OC=64, spatial 12x12=144  → BS=256*144
# deep 6x6:   OC=64, spatial 6x6=36     → BS=256*36


def bench_one[
    BS: Int, OC: Int, COL: Int, WARMUP: Int, ITERS: Int
](ctx: DeviceContext, label: StaticString) raises:
    # outp[BS, OC] = col[BS, COL] @ W[OC, COL]ᵀ  (transpose_b=True)
    var col = ctx.enqueue_create_buffer[DT](BS * COL)
    var w = ctx.enqueue_create_buffer[DT](OC * COL)
    var outp = ctx.enqueue_create_buffer[DT](BS * OC)
    _ = col.enqueue_fill(Scalar[DT](0.01))
    _ = w.enqueue_fill(Scalar[DT](0.01))
    _ = outp.enqueue_fill(Scalar[DT](0.0))

    var col_tt = TileTensor(col, row_major[BS, COL]())
    var w_tt = TileTensor(w, row_major[OC, COL]())
    var outp_tt = TileTensor(outp, row_major[BS, OC]())

    comptime for _ in range(WARMUP):
        max_matmul[transpose_b=True, target="gpu"](outp_tt, col_tt, w_tt, ctx)
    ctx.synchronize()

    var t0 = perf_counter_ns()

    comptime for _ in range(ITERS):
        max_matmul[transpose_b=True, target="gpu"](outp_tt, col_tt, w_tt, ctx)
    ctx.synchronize()
    var t1 = perf_counter_ns()

    var us = Float64(t1 - t0) / Float64(ITERS) / 1000.0
    # FLOPs = 2 * M * N * K
    var gflop = 2.0 * Float64(BS) * Float64(OC) * Float64(COL) / 1e9
    var tflops = gflop / (us / 1e6) / 1e3
    print(
        "  ", label, " M(BS)=", BS, " N(OC)=", OC, " K(COL)=", COL,
        " | ", us, " us/iter  ", tflops, " TFLOP/s",
    )


def main() raises:
    var ctx = DeviceContext()
    print("conv-GEMM shape bench (max_matmul[transpose_b=True], fp32)")
    print("=" * 64)

    print("stem (spatial 2304):")
    bench_one[256 * 2304, 32, 108, 5, 50](ctx, "  natural OC=32 ")
    bench_one[256 * 2304, 128, 108, 5, 50](ctx, "  padded  OC=128")

    print("downsample (spatial 576):")
    bench_one[256 * 576, 64, 576, 5, 50](ctx, "  natural OC=64 ")
    bench_one[256 * 576, 128, 576, 5, 50](ctx, "  padded  OC=128")

    print("deep block (spatial 144):")
    bench_one[256 * 144, 64, 576, 5, 50](ctx, "  natural OC=64 ")
    bench_one[256 * 144, 128, 576, 5, 50](ctx, "  padded  OC=128")

    print("deep block (spatial 36):")
    bench_one[256 * 36, 64, 576, 5, 100](ctx, "  natural OC=64 ")
    bench_one[256 * 36, 128, 576, 5, 100](ctx, "  padded  OC=128")

    print("=" * 64)
    print("If padded-OC=128 is faster/equal despite 2x FLOPs → N=64 is on a")
    print("slow path; padding OC to mult-of-128 is a real fp32 lever.")
    print("If padded is ~2x slower → kernel is fine at N=64; AMP is the lever.")
