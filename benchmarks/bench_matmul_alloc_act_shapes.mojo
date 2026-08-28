"""WHICH of ACT's GEMM shapes make cuBLAS allocate a workspace?

    pixi run -e nvidia nsys profile --stats=true mojo run -I . \\
        benchmarks/bench_matmul_alloc_act_shapes.mojo > act_shapes.txt 2>&1

The shape-sweep companion to `bench_matmul_alloc_probe.mojo` — same method
(preallocated buffers, nothing but `max_matmul` in the loop, so every
allocation nsys reports is the GEMM's), applied to the shapes ACT actually
issues instead of TD-MPC2's three. Read that file first; it carries the
original measurement and the reasoning.

## What this is for

After the grouped-arena work landed, allocation is the ENTIRE remaining item in
`docs/GPU_STEP_PERF.md`: 4.24 s of CUDA API time against 3.38 s of kernel time.
A `--cudabacktrace` capture could not name the call sites (nsys could not
configure CPU sampling on that box), so `tools/nsys_alloc_sites.py` attributed
each allocation to the kernel launched next instead. The answer was blunt:

    cuMemAlloc_v2, 7,789 calls, by following kernel
      #1  1170 (15.0%)  multistage_gemm_split_k_kernel ..._7853db89
      #2  1146 (14.7%)  cutlass ..._s1688gemm_64x64_16x6_nn_align4
      #3   799 (10.3%)  mojo_rl_deep_agents_loss_seed...   <- ONE-OFF, see below
      #4   625 ( 8.0%)  cutlass ..._s1688gemm_128x128_32x3_tn_align4
      #5   518 ( 6.7%)  cutlass ..._s1688gemm_64x64_16x6_tn_align4
      ... 157 distinct followers, the rest cutlass/split-K

`cutlass` fired 2,995 times in that run and `split_k` 2,910 — 5,905 together,
which is ~76% of the 7,789 allocations. **One allocation per cutlass or
split-K launch.** Meanwhile `multistage_gemm` (the non-split-K path) fired
12,860 times and does not appear as an allocation follower AT ALL.

⚠ **Alignment is NOT the criterion, and this corrects how the padding work was
framed.** Three of the top five allocators are `_align4` — already padded.
`docs/GPU_STEP_PERF.md` hedged this correctly ("cuBLAS allocates whenever it
picks a *cutlass* kernel; only about half of those also split-K") and the data
now says so emphatically. Padding N from 101 to 128 fixed the TD-MPC2 head
because it moved that shape onto `multistage_gemm`, not because 128 is aligned.
**The question is which PATH cuBLAS picks, and that is what this file measures.**

⚠ #3 is not a per-step site. `seed_grad_inv_batch` runs ONCE, at `make`, so it
is simply the first kernel in the trace and every allocation made during model
construction is attributed to it. 799 one-off allocations at startup, correctly
bucketed and correctly ignored. Nothing in the repo allocates per step: the
`Tensor.upload`-in-a-hot-loop class of bug has no instances left here.

## How to read the output

Every shape gets a DISTINCT rep count, so the kernel summary decomposes without
any subset-sum guessing: find each rep count in the `Instances` column and read
across to the kernel name.

    multistage_gemm_kernel...        -> NO workspace. This shape is free.
    multistage_gemm_split_k_kernel   -> allocates, memsets and frees PER CALL.
    cutlass::Kernel2<...>            -> allocates PER CALL (aligned or not).

Then cross-check `cuMemAlloc_v2 / Num Calls` in the API summary against the sum
of the allocating shapes' reps.

A shape that lands on `multistage_gemm` costs nothing and needs no change. A
shape that lands on cutlass/split-K is a candidate — and the fix is to find a
nearby shape that does not, which is why several entries below are deliberately
paired with a padded or reshaped variant.

## The shapes

`examples/so101/act_so101_profile_gpu.mojo` at BATCH=16, DIM=256, FF=1024,
ENC=4, DEC=1, K=60, 2 cameras at 240x320 (ResNet18 -> 8x10 per camera, so 160
image tokens + latent + proprio = 162 memory tokens; the CVAE encoder sees
cls + qpos + 60 actions = 62). M is `BATCH * tokens`; `Tokenwise[S, Linear]`
issues ONE GEMM at M = BATCH*S.

⚠ These are the FORWARD orientations only, plus the two backward shapes for the
conv stem. `vjp` is unaligned on the opposite axes from `forward` (grad_w has
N = OUT_, grad_input has K = OUT_ and N = IN_), so a full answer needs the
backward orientations too — add them once the forward picture is known, rather
than tripling the sweep up front.
"""

from std.time import perf_counter_ns
from max.gpu.host import DeviceContext
from layout import TileTensor, row_major
from linalg.matmul import matmul as max_matmul

comptime DT = DType.float32

# Distinct rep counts — the whole decoding trick. Keep them distinct, and keep
# them SMALL for the big shapes (the conv stem GEMM is 5.8 GFLOP a call).
comptime R_ATTN_ENC = 101      # [ 992 x  256] @ [ 256 x  256]
comptime R_ATTN_MEM = 103      # [2592 x  256] @ [ 256 x  256]
comptime R_FFN_UP = 107        # [2592 x  256] @ [ 256 x 1024]
comptime R_FFN_DOWN = 109      # [2592 x 1024] @ [1024 x  256]
comptime R_SRC = 113           # [2560 x  512] @ [ 512 x  256]
comptime R_TINY_M = 127        # [  16 x   32] @ [  32 x  256]   qenc/prop/lattok
comptime R_TINY_M_PAD = 131    # [  32 x   32] @ [  32 x  256]   same, M padded
comptime R_HEAD_N32 = 137      # [ 960 x  256] @ [ 256 x   32]   ahat (N padded)
comptime R_HEAD_N256 = 139     # [ 960 x  256] @ [ 256 x  256]   ahat, N widened
comptime R_LATINFO = 149       # [  16 x  256] @ [ 256 x   64]   latent_proj
comptime R_STEM_FWD = 11       # [307200 x 160] @ [160 x   64]   conv stem fwd
comptime R_STEM_DW = 7         # [   64 x 307200] @ [307200 x 160] conv stem dW


def run[
    M_: Int, K_: Int, N_: Int
](ctx: DeviceContext, reps: Int, label: String) raises:
    """`reps` calls of `max_matmul` on preallocated buffers and NOTHING else,
    so every allocation nsys attributes to this process during the loop is the
    GEMM's. Lifted verbatim in shape from `bench_matmul_alloc_probe.mojo`."""
    var a = ctx.enqueue_create_buffer[DT](M_ * K_)
    var b = ctx.enqueue_create_buffer[DT](K_ * N_)
    var c = ctx.enqueue_create_buffer[DT](M_ * N_)
    a.enqueue_fill(Scalar[DT](0.01))
    b.enqueue_fill(Scalar[DT](0.01))
    ctx.synchronize()

    var av = TileTensor(a, row_major[M_, K_]())
    var bv = TileTensor(b, row_major[K_, N_]())
    var cv = TileTensor(c, row_major[M_, N_]())

    # ⚠ The warm-up is NOT decoration: cuBLAS does one-off allocations on its
    # first call for a shape (heuristic tables, module loads). Without it those
    # land in the measured window and every shape looks like it allocates.
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
    print("ACT GEMM shapes — which pick an ALLOCATING cuBLAS path?")
    print("device:", ctx.name())
    print("=" * 78)

    run[992, 256, 256](ctx, R_ATTN_ENC, "cvae-enc attn proj (SEQ 62)")
    run[2592, 256, 256](ctx, R_ATTN_MEM, "encoder attn proj (SEQ 162)")
    run[2592, 256, 1024](ctx, R_FFN_UP, "ffn up")
    run[2592, 1024, 256](ctx, R_FFN_DOWN, "ffn down")
    run[2560, 512, 256](ctx, R_SRC, "input_proj (backbone 512 -> 256)")
    run[16, 32, 256](ctx, R_TINY_M, "qenc/prop/lattok  <- TINY M=16")
    run[32, 32, 256](ctx, R_TINY_M_PAD, "  ... same, M padded to 32")
    run[960, 256, 32](ctx, R_HEAD_N32, "ahat action head  <- N=32")
    run[960, 256, 256](ctx, R_HEAD_N256, "  ... same, N widened to 256")
    run[16, 256, 64](ctx, R_LATINFO, "latent_proj  <- TINY M, narrow N")
    run[307200, 160, 64](ctx, R_STEM_FWD, "conv stem forward (im2col)")
    run[64, 307200, 160](ctx, R_STEM_DW, "conv stem dW  <- huge K")

    print("=" * 78)
    print("  Reps are DISTINCT so the kernel summary decodes directly:")
    print("    find each count in `Instances`, read across to the kernel name.")
    print("      multistage_gemm_kernel...       -> free, no workspace")
    print("      multistage_gemm_split_k_kernel  -> allocates PER CALL")
    print("      cutlass::Kernel2<...>           -> allocates PER CALL")
    print("    (+5 warm-up calls per shape, excluded from the printed timing")
    print("     but PRESENT in the nsys counts — expect reps+5 instances.)")
    print()
    print("  Then check `cuMemAlloc_v2 / Num Calls` against the sum of the")
    print("  allocating shapes' reps. A shape on multistage needs no change;")
    print("  a shape on cutlass/split-K is a candidate, and its PAIRED variant")
    print("  above says whether a nearby shape escapes the allocating path.")
    print("=" * 78)
