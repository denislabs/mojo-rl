"""Two follow-ups the ACT profile left open: MAX's OWN split-K, and `bmm`.

    nsys profile --trace=cuda -o splitk_bmm \\
        pixi run -e nvidia mojo run -I . \\
        benchmarks/bench_matmul_alloc_splitk_and_bmm.mojo 2>&1 | tee splitk_bmm_run.txt
    nsys stats --report cuda_gpu_kern_sum --report cuda_api_sum \\
        splitk_bmm.nsys-rep > splitk_bmm_stats.txt 2>&1

## F — does `multistage_gemm_split_k` allocate too?

Padding K and N to MAX's dispatch gate (`09ff31fa`) moved 2,020 cutlass
launches onto the free path and cut allocator time 4.24 s -> 1.88 s. But 5,848
alloc/free pairs remain and only 975 cutlass launches are left to explain them.
The counts point at MAX's OWN split-K kernel, which barely moved:

    cutlass              2,995 -> 975     (the pad)
    multistage_split_k   2,910 -> 2,975   (untouched)
    startup one-offs       799 -> 799
                         -----    -----
                         6,704    4,749   vs 7,789 / 5,848 actual

That is circumstantial — the residual (~1,090) is identical in both runs, so
*something* constant is unaccounted for either way. **F settles it**: shapes
that PASS the gate (so no vendor fallback is possible) but are skinny enough in
M and N against a long K that `select_config` partitions the reduction. If
`cuMemAlloc_v2` tracks these reps, MAX's own split-K allocates per call too,
and the workspace-reuse ask in `docs/MODULAR_MATMUL_ALLOC_REPORT.md` is about
MAX's kernels rather than only about the cuBLAS fallback — a materially
stronger and more actionable report.

⚠ These shapes are `Linear`'s dW GEMM (`grad_w = xᵀ @ go`, so M = IN_,
K = BATCH*tokens, N = OUT_), which is where a long K comes from in this repo.

## G — `bmm` has the SAME gate and a WORSE fallback

`linalg/bmm.mojo:815`:

    comptime multistage_gemm_cond = (
        c_n % 128 == 0 and a_k % 32 == 0 and a_k >= 128
    )

Identical to `matmul`'s, minus the H100/AMD escapes. But the `else` branch is
not cuBLAS — it is `naive_batched_matmul_kernel` at `BLOCK_DIM = 16`. So bmm
never allocates, and instead silently drops to a kernel with no tensor cores.
That is `naive_batched_matmul`: **5,100 launches, 189.7 ms, 5.5% of kernel
time** in the ACT profile.

⚠ **Every multi-head attention in this repo fails that gate structurally**,
because both attention GEMMs put `HEAD_DIM` on a gated axis:

    q @ kᵀ    a_k = HEAD_DIM = 32   -> fails `a_k >= 128`
              c_n = KV_LEN  = 162   -> fails `c_n % 128`
    attn @ v  a_k = KV_LEN  = 162   -> fails `a_k % 32`
              c_n = HEAD_DIM = 32   -> fails `c_n % 128`

Passing would need `HEAD_DIM >= 128` AND `HEAD_DIM % 128 == 0`, i.e. exactly
128/256/... — DIM 1024 over 8 heads. ACT is 256/8 = 32, and no transformer here
is close. This is not a shape we drifted into; it is the standard one.

G measures what that costs and whether padding can buy it back. Padding is NOT
free here the way it was for `Linear`:

  * `attn @ v`'s `c_n = HEAD_DIM` -> 128 is SAFE and exact — the extra output
    columns are computed against zero V columns, come out 0, and are sliced.
  * `q @ kᵀ`'s `a_k = HEAD_DIM` -> 128 is SAFE and exact — zero-padded q/k
    contribute exactly 0 to every score.
  * `q @ kᵀ`'s `c_n = KV_LEN` -> 256 is **NOT** safe on its own. Padded key
    columns score 0, and `exp(0) = 1` is a perfectly good softmax weight, so
    the padded positions would steal probability mass. They must be masked to
    -inf first. The primitive already has a mask path, so it is reachable —
    but it is a correctness change, not a shape change, and it is why this
    file only MEASURES.

The rows below therefore bracket the decision: the real shape, then each axis
padded in isolation, then both. If the fully padded arm is not comfortably
faster than the naive one, the masking work is not worth starting.
"""

from std.time import perf_counter_ns
from max.gpu.host import DeviceContext
from layout import TileTensor, row_major
from linalg.matmul import matmul as max_matmul
from linalg.bmm import batched_matmul

comptime DT = DType.float32


def run[
    M_: Int, K_: Int, N_: Int
](ctx: DeviceContext, reps: Int, label: String) raises:
    """Plain `matmul` on preallocated buffers, nothing else in the loop."""
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


def runb[
    BH_: Int, M_: Int, K_: Int, N_: Int, TB: Bool = False
](ctx: DeviceContext, reps: Int, label: String) raises:
    """`batched_matmul` on preallocated buffers. `TB` mirrors the `q @ kᵀ`
    call, which passes `transpose_b=True` — `b` is then `[BH, N, K]`."""
    var a = ctx.enqueue_create_buffer[DT](BH_ * M_ * K_)
    var b = ctx.enqueue_create_buffer[DT](BH_ * K_ * N_)
    var c = ctx.enqueue_create_buffer[DT](BH_ * M_ * N_)
    a.enqueue_fill(Scalar[DT](0.01))
    b.enqueue_fill(Scalar[DT](0.01))
    ctx.synchronize()
    var av = TileTensor(a, row_major[BH_, M_, K_]())
    var cv = TileTensor(c, row_major[BH_, M_, N_]())
    var t0: Int
    var t1: Int
    comptime if TB:
        var bv = TileTensor(b, row_major[BH_, N_, K_]())
        for _ in range(5):
            batched_matmul[transpose_b=True, target="gpu"](cv, av, bv, context=ctx)
        ctx.synchronize()
        t0 = perf_counter_ns()
        for _ in range(reps):
            batched_matmul[transpose_b=True, target="gpu"](cv, av, bv, context=ctx)
        ctx.synchronize()
        t1 = perf_counter_ns()
    else:
        var bv = TileTensor(b, row_major[BH_, K_, N_]())
        for _ in range(5):
            batched_matmul[target="gpu"](cv, av, bv, context=ctx)
        ctx.synchronize()
        t0 = perf_counter_ns()
        for _ in range(reps):
            batched_matmul[target="gpu"](cv, av, bv, context=ctx)
        ctx.synchronize()
        t1 = perf_counter_ns()
    print(
        "  x", reps, "  BH=", BH_, " [", M_, " x ", K_, "] @ [", K_, " x ",
        N_, "]   ", Float64(t1 - t0) / 1000.0 / Float64(reps),
        " us/call   ", label,
        sep="",
    )


def main() raises:
    var ctx = DeviceContext()
    print("=" * 78)
    print("F: does MAX's OWN split-K allocate?   G: what does bmm's naive"
          " fallback cost?")
    print("device:", ctx.name())
    print("=" * 78)

    # ── F. shapes that PASS the gate but should still partition K ────────
    # `Linear`'s dW GEMM: M = IN_, K = BATCH*tokens, N = OUT_. All three pass
    # `n % 128 == 0 and k % 32 == 0 and k >= 128`, so a vendor fallback is
    # impossible and any allocation is MAX's own kernel.
    print("F. long-K, skinny-MN — all PASS the gate")
    run[256, 2592, 256](ctx, 101, "ACT attn-proj dW")
    run[256, 2592, 1024](ctx, 103, "ACT ffn-up dW")
    run[1024, 2592, 256](ctx, 107, "ACT ffn-down dW")
    run[128, 32768, 128](ctx, 109, "deliberately extreme K")
    run[2592, 256, 256](ctx, 113, "same M*N, SHORT K (control — expect no split-K)")

    # ── G. bmm: ACT's attention, then each axis padded ───────────────────
    # BH = BATCH * N_HEADS = 16 * 8 = 128. HEAD_DIM = 32, KV_LEN = 162.
    print("G. bmm, ACT cross-attention (BH=128, HEAD_DIM=32, Q=60, KV=162)")
    runb[128, 60, 32, 162, True](ctx, 127, "q@k^T   REAL     (naive)")
    runb[128, 60, 128, 162, True](ctx, 131, "q@k^T   a_k->128 (c_n still bad)")
    runb[128, 60, 128, 256, True](ctx, 137, "q@k^T   BOTH padded  <- needs -inf mask")
    runb[128, 60, 162, 32](ctx, 139, "attn@v  REAL     (naive)")
    runb[128, 60, 162, 128](ctx, 149, "attn@v  c_n->128 (a_k still bad)")
    runb[128, 60, 192, 128](ctx, 151, "attn@v  BOTH padded  <- k-pad needs the mask")

    print("=" * 78)
    print("  reps + 5 warm-up = the `Instances` count to look for.")
    print()
    print("  F: if cuMemAlloc_v2 ~= the sum of F's reps, MAX's own split-K")
    print("     allocates per call and the Modular report widens beyond the")
    print("     vendor fallback. If it is ~15 (just the buffers), the")
    print("     remaining ACT allocations are something else and the")
    print("     neighbour attribution needs redoing with real backtraces.")
    print("     Cross-check `multistage_gemm_split_k` instances in the")
    print("     kernel summary against the same reps.")
    print()
    print("  G: compare the REAL row against BOTH-padded. Padding q@k^T is")
    print("     ~6.3x the FLOPs (a_k 32->128, c_n 162->256), so the naive")
    print("     kernel has to be losing by more than that to justify the")
    print("     softmax masking work. Anything less and workstream C stays")
    print("     closed.")
    print("=" * 78)
