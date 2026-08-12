"""Does `linalg.matmul` allocate a device buffer PER CALL?

    pixi run -e nvidia nsys profile --stats=true mojo run -I . \
        benchmarks/bench_matmul_alloc_probe.mojo

## Why

An nsys capture of `examples/dm_control/tdmpc2_dm_walker_profile_gpu.mojo` on
an RTX 5090 showed:

    cuMemFree_v2    2.59 s   36.7%   21,464 calls   avg 121 us
    cuMemAlloc_v2   2.13 s   30.1%   21,464 calls   avg  99 us

4.72 s — 67% of all CUDA API time, and MORE than the 3.10 s of GPU kernel time
in the same run. The counts are exactly equal, so something creates and
destroys a buffer per use. It is not the repo's own allocation sites: every
`alloc_gpu` / `enqueue_create_buffer` under `tdmpc2/`, `planners/trajectory/`
and `nn/` sits in a constructor, and `Tensor.ensure_gpu` only ever grows
(`if not self.dev or self.n < n`), so per-call scratch does not reallocate.

That leaves the GEMM. This file calls `max_matmul` `REPS` times on
PREALLOCATED buffers and does nothing else in the loop — no Tensor, no Module,
no agent. So every `cuMemAlloc_v2` nsys reports beyond the handful at startup
belongs to `linalg.matmul`.

## Reading it

Compare `cuMemAlloc_v2`'s `Num Calls` against `REPS` (printed at the end):

    ~= one loop's calls  ->  that shape allocates per call.
    ~= a handful (< 50)  ->  REFUTED: the GEMM is not the source. Fall back to
                             a backtrace capture of the full profile:

        nsys profile --trace=cuda,osrt --cudabacktrace=memory:10000 \
             --sample=cpu --backtrace=dwarf -o alloc --force-overwrite=true \
             mojo run -I . examples/dm_control/tdmpc2_dm_walker_profile_gpu.mojo

    then open `alloc.nsys-rep` in the Nsight GUI and read the backtrace on any
    `cuMemAlloc_v2` row. (`--cudabacktrace` needs `--sample`, and it is slow —
    shrink COLLECT_STEPS/TRAIN_STEPS in that script before using it.)

## MEASURED, RTX 5090, 2026-08-12

    cuMemAlloc_v2  2016 calls    cuMemFree_v2  2016 calls
    cuMemsetD8Async 2010 calls   [CUDA memset] 67.4 GB, avg 33.5 MB

2016 = 2010 + 6 startup, and 2010 is ONE loop, not both (4020). The kernel
summary names it:

    N=512 planner  -> multistage_gemm x2010,  26.3 us/call,  NO allocation
    N=101 two-hot  -> cutlass ...nn_align1 x2010
                      + splitKreduce_kernel x2010, 217 us/call

cuBLAS picks a SPLIT-K path for the narrow N, and split-K needs a workspace:
allocated, memset and freed EVERY call. 2144*101*32 splits*4 B = 27.7 MB
against the measured 33.5 MB median, and the kernel is literally
`splitKreduce_kernel<(int)32, ...>`. The allocation is most of the 217 us.

So this is NOT a MAX bug to report upstream — split-K genuinely needs scratch.
It is our SHAPE: N = BINS = 101, an unaligned narrow OUTPUT width, in the
reward head, all five Q heads and termination. It also explains why
`[268x512]@[512x101]` measured 312 us against `[512x128]`'s 24.8 us in
`bench_matmul_k_alignment.mojo` — that was this, not kernel selection.
"""

from std.time import perf_counter_ns
from max.gpu.host import DeviceContext
from layout import TileTensor, row_major
from linalg.matmul import matmul as max_matmul

comptime DT = DType.float32

# The walker planner's dominant GEMM: BATCH_TOTAL = N_ENVS(8) * (256 + 12).
comptime M = 2144
comptime K = 544          # (LATENT 512 + ACT 6) padded to a multiple of 32
comptime N = 512
comptime REPS = 2000

# A second shape: the two-hot head, whose narrow N triggers split-K on NVIDIA
# (14,500 `splitKreduce_kernel` instances appeared in the walker capture).
comptime M2 = 2144
comptime K2 = 512
comptime N2 = 101
comptime REPS2 = 2000

# Third: the SAME shape with N padded to a multiple of 32. On an RTX 5090 the
# N=101 loop was measured at 2010 allocations for 2010 calls — one workspace
# alloc + a 33.5 MB memset + a free PER CALL, because cuBLAS picks a split-K
# path (`splitKreduce_kernel<32, ...>`, and 2144*101*32*4 B = 27.7 MB matches
# the measured 33.5 MB median). N=128 should take a single-kernel path with no
# workspace at all. This loop is the PROOF that padding N fixes it, and must be
# run before anyone writes the padding into `Linear`.
comptime M3 = 2144
comptime K3 = 512
comptime N3 = 128
comptime REPS3 = 2000


def run[M_: Int, K_: Int, N_: Int](
    ctx: DeviceContext, reps: Int, label: String
) raises:
    var a = ctx.enqueue_create_buffer[DT](M_ * K_)
    var b = ctx.enqueue_create_buffer[DT](K_ * N_)
    var c = ctx.enqueue_create_buffer[DT](M_ * N_)
    a.enqueue_fill(Scalar[DT](0.01))
    b.enqueue_fill(Scalar[DT](0.01))
    ctx.synchronize()

    var av = TileTensor(a, row_major[M_, K_]())
    var bv = TileTensor(b, row_major[K_, N_]())
    var cv = TileTensor(c, row_major[M_, N_]())

    for _ in range(10):                      # warm up / one-off allocations
        max_matmul[target="gpu"](cv, av, bv, ctx)
    ctx.synchronize()

    var t0 = perf_counter_ns()
    for _ in range(reps):                    # <- ONLY max_matmul in here
        max_matmul[target="gpu"](cv, av, bv, ctx)
    ctx.synchronize()
    var t1 = perf_counter_ns()

    print(
        "  ", label, ": [", M_, "x", K_, "]@[", K_, "x", N_, "]  x", reps,
        " calls   ", Float64(t1 - t0) / 1000.0 / Float64(reps), " us/call",
        sep="",
    )


def main() raises:
    var ctx = DeviceContext()
    print("=" * 70)
    print("max_matmul per-call allocation probe —", ctx.name())
    print("=" * 70)
    run[M, K, N](ctx, REPS, "planner GEMM  N=512")
    run[M2, K2, N2](ctx, REPS2, "two-hot head  N=101")
    run[M3, K3, N3](ctx, REPS3, "two-hot PADDED N=128")
    print("=" * 70)
    print("  max_matmul calls per loop:", REPS, "(+10 warm-up each)")
    print()
    print("  Read `cuMemAlloc_v2 / Num Calls` from the CUDA API Summary:")
    print("    ~2010  -> ONLY the N=101 loop allocates. Padding N to a")
    print("              multiple of 32 removes the split-K workspace, and")
    print("              the fix belongs in Linear alongside the K padding.")
    print("    ~4020  -> N=128 allocates too; padding N is NOT the fix and")
    print("              the workspace is not split-K-specific.")
    print("    ~6030  -> all three allocate; it is per-call regardless of")
    print("              shape, i.e. a MAX-level workspace-cache issue.")
    print()
    print("  Cross-check with `splitKreduce_kernel` in the kernel summary —")
    print("  its instance count should equal the allocating loop's calls.")
    print("=" * 70)
