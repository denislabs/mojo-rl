"""How WIDE does N have to be before cuBLAS stops allocating?

    pixi run -e nvidia nsys profile --trace=cuda -o alloc_thresh \\
        pixi run -e nvidia mojo run -I . \\
        benchmarks/bench_matmul_alloc_threshold.mojo 2>&1 | tee alloc_thresh_run.txt
    nsys stats --report cuda_gpu_kern_sum --report cuda_api_sum \\
        alloc_thresh.nsys-rep > alloc_thresh_stats.txt 2>&1

## The measurement this exists to finish

`bench_matmul_alloc_act_shapes.mojo` on an RTX 5090, decoded against the kernel
summary (every count matched to the unit):

    [ 992 x  256] @ [ 256 x  256]   16.23 us   multistage        -
    [2592 x  256] @ [ 256 x  256]   16.45 us   multistage        -
    [2592 x  256] @ [ 256 x 1024]   18.10 us   multistage        -
    [2592 x 1024] @ [1024 x  256]   50.15 us   multistage        -
    [2560 x  512] @ [ 512 x  256]   27.47 us   multistage        -
    [  16 x   32] @ [  32 x  256]  297.29 us   cutlass + splitK  ALLOCATES
    [  32 x   32] @ [  32 x  256]  295.73 us   cutlass + splitK  ALLOCATES
    [ 960 x  256] @ [ 256 x   32]  291.60 us   cutlass           ALLOCATES
    [ 960 x  256] @ [ 256 x  256]   14.87 us   multistage        -
    [  16 x  256] @ [ 256 x   64]  362.32 us   cutlass + splitK  ALLOCATES
    [307200 x 160] @ [160 x   64]  416.93 us   cutlass           ALLOCATES
    [64 x 307200] @ [307200 x 160] 482.97 us   cutlass + splitK  ALLOCATES

**The action head, same M and same K, 8x the FLOPs, 19.6x FASTER at N=256 than
at N=32.** 291.60 -> 14.87 us. The 277 us difference is allocator: one
`cuMemAlloc` + one `cuMemsetD8Async` + one `cuMemFree` per call
(`cuMemsetD8Async` came back at 592, exactly the cutlass launch count).

⚠ **This refutes the N-padding rule as `Linear` currently implements it.**
`Linear.N_PAD` rounds `OUT_` up to a multiple of `PAD_TO = 32`, and **N = 32 is
a multiple of 32 and still allocates**. TD-MPC2's N=101 -> 128 worked because
128 is WIDE ENOUGH, not because it is aligned. So `ahat` (`Linear[256, 6]`,
padded 6 -> 32) and `latent_proj` (N = 64) are BOTH still on the allocating
path today, and the comment in `linear.mojo` claiming otherwise is wrong for
small `OUT_`.

⚠ **Padding M does nothing.** 16 -> 32 moved 297.29 to 295.73 us. Whatever
cuBLAS is keying on, it is not the M alignment, so the tiny-M sites
(`qenc`/`prop`/`lattok`, M = BATCH = 16) need a different answer from the
narrow-N sites — most likely not going through `max_matmul` at all, the way
`conv2d`'s `_fwd_oc1_matvec_kernel` already sidesteps it at OC == 1.

## What this file answers, and why it must run BEFORE any constant changes

Three questions, none of which the sweep above can answer:

  A. Where between N=64 and N=256 does the allocating path stop? The new
     `N_PAD` floor is that number, and guessing it wrong either leaves the
     allocation in place (too low) or burns FLOPs for nothing (too high).
  B. Can a tiny M ever escape by itself? [16 x 32] @ [32 x 256] allocates with
     N ALREADY at 256, so widening N is clearly not the lever there. Sweeping M
     at fixed N says whether there is a row count that escapes, or whether
     these shapes need to leave `max_matmul` entirely.
  C. Does the conv stem's forward escape at OC = 128? Its N is `OC_ = 64` and
     `conv2d` deliberately does NOT pad that axis — the decision is recorded in
     `Conv2D`'s comment block as "OC is always aligned", which the data above
     shows was the wrong criterion.

⚠ Do not calibrate any of this on Apple. Metal has no split-K path and no
equivalent workspace, and a threshold read off an M1 would be a fiction.

## Reading it

Distinct rep counts again, so the kernel summary decodes directly: find
`reps + 5` in `Instances`, read across to the kernel name.
`multistage_gemm_kernel` = free. `cutlass::Kernel2<...>` = allocates, whether or
not a `splitKreduce_kernel` accompanies it. Cross-check `cuMemAlloc_v2` against
(sum of allocating reps + 5 each) + 3 per shape for the buffers.
"""

from std.time import perf_counter_ns
from max.gpu.host import DeviceContext
from layout import TileTensor, row_major
from linalg.matmul import matmul as max_matmul

comptime DT = DType.float32


def run[
    M_: Int, K_: Int, N_: Int
](ctx: DeviceContext, reps: Int, label: String) raises:
    """`reps` calls on preallocated buffers, nothing else in the loop. The 5
    warm-up calls absorb cuBLAS's one-off per-shape allocations, which would
    otherwise make every shape look like it allocates."""
    var a = ctx.enqueue_create_buffer[DT](M_ * K_)
    var b = ctx.enqueue_create_buffer[DT](K_ * N_)
    var c = ctx.enqueue_create_buffer[DT](M_ * N_)
    a.enqueue_fill(Scalar[DT](0.01))
    b.enqueue_fill(Scalar[DT](0.01))
    ctx.synchronize()

    var av = TileTensor(a, row_major[M_, K_]())
    var bv = TileTensor(b, row_major[K_, N_]())
    var cv = TileTensor(c, row_major[M_, N_]())

    for _ in range(5):
        max_matmul[target="gpu"](cv, av, bv, ctx)
    ctx.synchronize()

    var t0 = perf_counter_ns()
    for _ in range(reps):
        max_matmul[target="gpu"](cv, av, bv, ctx)
    ctx.synchronize()
    var t1 = perf_counter_ns()

    print(
        "  x", reps, "  [", M_, " x ", K_, "] @ [", K_, " x ", N_, "]   ",
        Float64(t1 - t0) / 1000.0 / Float64(reps), " us/call   ", label,
        sep="",
    )


def main() raises:
    var ctx = DeviceContext()
    print("=" * 78)
    print("Where does the allocating cuBLAS path start? —", ctx.name())
    print("=" * 78)

    # ── A. N sweep at the action head's M and K ──────────────────────────
    # N=32 allocates and N=256 does not; the new `Linear.N_PAD` floor is
    # whichever of these is the first to come back `multistage`.
    print("A. N sweep, M=960 K=256 (the `ahat` shape)")
    run[960, 256, 32](ctx, 101, "N=32   (today's N_PAD for OUT_=6)")
    run[960, 256, 48](ctx, 103, "N=48")
    run[960, 256, 64](ctx, 107, "N=64   (latent_proj's real N)")
    run[960, 256, 96](ctx, 109, "N=96")
    run[960, 256, 128](ctx, 113, "N=128  (TD-MPC2's working pad)")
    run[960, 256, 160](ctx, 127, "N=160")
    run[960, 256, 192](ctx, 131, "N=192")
    run[960, 256, 256](ctx, 137, "N=256  (known free)")

    # ── B. M sweep — can a tiny M escape at all? ─────────────────────────
    # N is ALREADY 256 here, so if these all allocate, widening is not the
    # lever and these sites have to leave `max_matmul`.
    print("B. M sweep, K=32 N=256 (the `qenc`/`prop`/`lattok` shape)")
    run[16, 32, 256](ctx, 139, "M=16   (BATCH)")
    run[64, 32, 256](ctx, 149, "M=64")
    run[256, 32, 256](ctx, 151, "M=256")
    run[1024, 32, 256](ctx, 157, "M=1024")

    # ── C. the conv stem's forward, at OC and at a padded OC ─────────────
    # `Conv2D` pads COL but deliberately not OC. If N=128 comes back free,
    # that decision needs revisiting and `_scatter_bias_kernel` gets the
    # slice-back the way `Linear`'s bias add already does.
    print("C. conv stem forward, N = OC")
    run[307200, 160, 64](ctx, 11, "OC=64  (real)")
    run[307200, 160, 128](ctx, 13, "OC=128 (padded)")

    print("=" * 78)
    print("  reps + 5 warm-up = the `Instances` count to look for.")
    print("    multistage_gemm_kernel...  -> free")
    print("    cutlass::Kernel2<...>      -> allocates (splitK or not)")
    print()
    print("  A's first `multistage` row is the N_PAD floor.")
    print("  If every row of B allocates, tiny-M needs its own kernel, not a pad.")
    print("  If C's OC=128 is free, Conv2D's OC axis needs padding after all.")
    print("=" * 78)
