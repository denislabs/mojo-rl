"""Sweep the real ACT dW shapes: which split K, and what does owning the
workspace buy on each?

`bench_splitk_persistent_workspace.mojo` proved the technique on ONE shape
(1.39x, bit-identical, captures and replays). This asks whether the win holds
across the shapes ACT actually issues, and turns "1.39x on a GEMM" into "N ms
per training step".

WHERE THE SHAPES COME FROM
--------------------------
`Linear`'s GPU vjp issues its weight gradient as

    max_matmul(dW[K_PAD, N_PAD], cacheT[K_PAD, B], go[B, N_PAD])   (linear.mojo:922)

so a dW GEMM is  M = in_features, N = out_features, K = B, where B is the
FLATTENED row count = batch x tokens. Everything below follows from
`examples/so101/act_so101_profile_gpu.mojo` (BATCH=16, DIM=256, FF=1024, K=60,
N_ENC=4, N_DEC=1, 2 cameras at 240x320) and `deep_agents/act/layers.mojo`:

    transformer encoder   162 tokens (2 x 8x10 ResNet18 features + latent + qpos)
                          B = 16 * 162 = 2592
    CVAE encoder           62 tokens (cls + qpos + 60 actions)
                          B = 16 *  62 =  992
    decoder self/cross-q   60 queries          B = 16 *  60 =  960
    decoder cross k/v     162 memory tokens    B = 16 * 162 = 2592

⚠ That derivation is not free-floating: B=2592, B=992 and B=960 each reproduce
a shape MODULAR_MATMUL_ALLOC_REPORT.md measured independently
(`[256 x 2592] @ [2592 x 256]`, `[992 x 256] @ [256 x 256]`,
`[960 x 256] @ [256 x 32]`). Rows marked MEASURED appear in that report
verbatim; rows marked DERIVED follow from the config and the layer list.

⚠ The AUTHORITATIVE list is the training step itself. To check this table is
complete rather than merely correct:

    pixi run -e nvidia mojo run -D LOGGING_LEVEL=INFO -I . \
        examples/so101/act_so101_profile_gpu.mojo 2>&1 \
      | grep -A6 'MATMUL GPU execution started' \
      | grep -E 'MxNxK|K partitions' | paste - - | sort | uniq -c | sort -rn

`q`, `k` and `v` are SEPARATE Linears here, not a fused `Linear[DIM, 3*DIM]`
(layers.mojo:15 says so explicitly), which is why the count column below has
four 256x256 attention GEMMs per layer rather than one wide one.

Run (NVIDIA only):

    pixi run -e nvidia mojo run -I . benchmarks/bench_splitk_act_dw_sweep.mojo
"""

from std.math import ceildiv, align_up
from std.sys import has_nvidia_gpu_accelerator
from std.time import perf_counter_ns

from max.gpu.host import DeviceContext, DeviceBuffer, FuncAttribute
from layout import Layout, LayoutTensor, TileTensor, RuntimeLayout, UNKNOWN_VALUE
from layout import row_major, Coord
from std.utils.index import Index

from linalg.matmul import matmul as max_matmul
from linalg.utils_gpu import MatmulConfig, MatmulKernels, select_config
from linalg.matmul.gpu import multistage_gemm_split_k_kernel, split_k_reduce


comptime DT = DType.float32
comptime WARMUP = 5
comptime REPS = 30


struct SplitKWorkspace[dtype: DType](Movable):
    """A device buffer reused across split-K GEMMs. See
    `bench_splitk_persistent_workspace.mojo` for the why."""

    var buf: DeviceBuffer[Self.dtype]
    var capacity: Int

    def __init__(out self, ctx: DeviceContext, capacity: Int) raises:
        self.buf = ctx.enqueue_create_buffer[Self.dtype](capacity)
        self.capacity = capacity
        ctx.synchronize()


def splitk_gemm[
    c_type: DType,
    a_type: DType,
    b_type: DType,
    ws_type: DType, //,
    *,
    transpose_b: Bool,
    config: MatmulConfig[a_type, b_type, c_type, transpose_b],
](
    c: TileTensor[mut=True, c_type, ...],
    a: TileTensor[mut=False, a_type, ...],
    b: TileTensor[mut=False, b_type, ...],
    num_partitions: Int,
    mut ws: SplitKWorkspace[ws_type],
    ctx: DeviceContext,
) raises:
    """`multistage_gemm`'s split-K branch on a caller-owned workspace.
    Mirrors matmul/gpu/__init__.mojo:1840-1915 without its per-call
    `enqueue_create_buffer` / `_ = work_space_data^` pair."""
    # The workspace dtype is `config.split_k_reduction_type` -- taken as an
    # INFERRED parameter rather than written into the signature, because the
    # compiler will not fold `config.split_k_reduction_type` when the caller
    # holds the workspace in a variable typed elsewhere. The assert restores
    # exactly the guarantee the signature would have given.
    comptime assert ws_type == config.split_k_reduction_type, (
        "workspace dtype must equal config.split_k_reduction_type"
    )
    var tensor_c = c.to_layout_tensor()
    var tensor_a = a.to_layout_tensor()
    var tensor_b = b.to_layout_tensor()
    var M = tensor_c.dim[0]()
    var N = tensor_c.dim[1]()

    if num_partitions * M * N > ws.capacity:
        raise Error("SplitKWorkspace too small for this GEMM")

    # ⚠ PARTITION COUNT SAFETY. `multistage_gemm_split_k_kernel`'s NVIDIA path
    # splits K with `LayoutTensor.split[axis, split_alignment=BK]`, which is
    # (layout_tensor.mojo:3870):
    #
    #     part = align_up(K // P, BK)                     <- FLOOR div, then align
    #     size_of_partition_i = min(part, K - i * part)
    #     ptr_i               = base + i * part * stride
    #
    # so when `(P-1) * part >= K` the last block's pointer starts PAST the end
    # of A and B and its size goes NEGATIVE. Measured on the 5090 at
    # K=2592, BK=16: P<=12 is fine (11*224 = 2464 < 2592) and P=16 faults
    # (15*176 = 2640 > 2592, last size -48) with CUDA_ERROR_ILLEGAL_ADDRESS.
    #
    # There are TWO ways to get this wrong, and only one of them is loud:
    #
    #   OVERRUN       (P-1) * part >= K   the last block starts past the end of
    #                                     A and B with a NEGATIVE size
    #                                     -> CUDA_ERROR_ILLEGAL_ADDRESS
    #   UNDERCOVERAGE  P * part < K       `part` is floor-then-align, so the P
    #                                     partitions can fail to span K. The
    #                                     tail is never accumulated and the
    #                                     GEMM SILENTLY RETURNS A WRONG ANSWER
    #
    # Both measured on the 5090 at K=2592, BK=16. P=16 overruns
    # (15*176 = 2640 > 2592). P=40 undercovers: part = align_up(64,16) = 64 and
    # 40*64 = 2560, so 32 of 2592 contraction elements are dropped -- and the
    # observed relative error was 0.012344, against 32/2592 = 0.012346. The
    # error IS the fraction of K dropped, to five digits.
    #
    # ⚠ MAX'S OWN GUARD COVERS NEITHER. `select_config` breaks on `K < P * bk`
    # (2592 < 256 is false at P=16, so it passes) and is saved only by the
    # SEPARATE `min_k_partition = 1024` test capping P at 2 here for unrelated
    # reasons. Any caller that chooses P itself -- us, or MAX's own
    # `TUNE_NUM_K_PARTITIONS` autotune define -- can reach both.
    var K_dim = Int(tensor_a.dim[1]())
    comptime BK = config.block_tile_shape[2]
    var part = align_up(K_dim // num_partitions, BK) if num_partitions > 0 else 0
    if num_partitions < 1:
        raise Error("num_k_partitions must be >= 1")
    if (num_partitions - 1) * part >= K_dim:
        raise Error(
            "num_k_partitions OVERRUNS K: the first P-1 partitions of"
            " align_up(K//P, BK) already cover K, so the last block would read"
            " past the operands (this one crashes loudly)"
        )
    if num_partitions * part < K_dim:
        raise Error(
            "num_k_partitions UNDERCOVERS K: P*align_up(K//P, BK) < K, so the"
            " tail of the contraction is never accumulated and the GEMM"
            " silently returns a wrong answer"
        )


    comptime static_N = tensor_c.layout.shape[1].value()
    comptime ws_layout = Layout.row_major(UNKNOWN_VALUE, UNKNOWN_VALUE, static_N)
    var ws_rt_layout = RuntimeLayout[ws_layout].row_major(
        Index(num_partitions, M, N)
    )
    var ws_lt = LayoutTensor[ws_type, ws_layout, MutAnyOrigin](
        ws.buf, ws_rt_layout
    )

    comptime kern = multistage_gemm_split_k_kernel[
        c_type, tensor_c.layout,
        a_type, tensor_a.layout,
        b_type, tensor_b.layout,
        ws_type, ws_lt.layout,
        transpose_b, config, None,
    ]

    ctx.enqueue_function[kern](
        tensor_c, tensor_a, tensor_b, ws_lt, Int32(num_partitions),
        grid_dim=(
            ceildiv(N, config.block_tile_shape[1]),
            ceildiv(M, config.block_tile_shape[0]),
            num_partitions,
        ),
        block_dim=config.block_dim(),
        shared_mem_bytes=config.shared_mem_usage(),
        func_attribute=FuncAttribute.MAX_DYNAMIC_SHARED_SIZE_BYTES(
            UInt32(config.shared_mem_usage())
        ),
    )

    var ws_tt = TileTensor(ws.buf, row_major(Coord(num_partitions, M, N)))
    split_k_reduce(c, ws_tt, ctx)


def sweep_one[
    M: Int, K: Int, N: Int, COUNT: Int, LABEL: StaticString
](ctx: DeviceContext, mut ws: SplitKWorkspace[DT], mut total_saved_us: Float64) raises:
    """One row of the sweep. Prints the dispatch verdict, and A/B only when
    the shape actually partitions K -- a shape that does not split takes
    plain `multistage_gemm` today and has nothing to win."""
    var picked = select_config[DT, DT, DT, False](M, N, K, ctx)
    var P = picked.num_k_partitions

    if P <= 1:
        print(
            "  ", LABEL, " [", M, "x", K, "] @ [", K, "x", N, "]",
            "  x", COUNT, "  ->  no split (P=1), unchanged", sep="",
        )
        return

    var ab = ctx.enqueue_create_buffer[DT](M * K)
    var bb = ctx.enqueue_create_buffer[DT](K * N)
    var c_ref = ctx.enqueue_create_buffer[DT](M * N)
    var c_ours = ctx.enqueue_create_buffer[DT](M * N)
    ab.enqueue_fill(Float32(0.01))
    bb.enqueue_fill(Float32(0.02))
    c_ref.enqueue_fill(Float32(0.0))
    c_ours.enqueue_fill(Float32(0.0))
    ctx.synchronize()

    var av = TileTensor(ab, row_major[M, K]())
    var bv = TileTensor(bb, row_major[K, N]())
    var cref = TileTensor(c_ref, row_major[M, N]())
    var cours = TileTensor(c_ours, row_major[M, N]())

    # ---- arm A: linalg.matmul (allocates its workspace per call) ----------
    for _ in range(WARMUP):
        max_matmul[target="gpu"](cref, av, bv, ctx)
    ctx.synchronize()
    var t0 = perf_counter_ns()
    for _ in range(REPS):
        max_matmul[target="gpu"](cref, av, bv, ctx)
    ctx.synchronize()
    var t1 = perf_counter_ns()

    # ---- arm B: ours, on the persistent workspace ------------------------
    # `cfg` supplies the grid and the shared-memory request, so it must be the
    # tile `select_config` chose. MAX picks between three; `_256x128_3` is
    # A100-gated, so two branches cover every NVIDIA part we run on. Getting
    # this wrong is a launch-geometry mismatch, not a slowdown -- hence the
    # raise rather than a fallback.
    comptime kernels = MatmulKernels[DT, DT, DT, False]()
    var t2: Int
    var t3: Int
    if picked == kernels.ampere_256x64_4:
        comptime cfg = kernels.ampere_256x64_4
        for _ in range(WARMUP):
            splitk_gemm[transpose_b=False, config=cfg](cours, av, bv, P, ws, ctx)
        ctx.synchronize()
        t2 = perf_counter_ns()
        for _ in range(REPS):
            splitk_gemm[transpose_b=False, config=cfg](cours, av, bv, P, ws, ctx)
        ctx.synchronize()
        t3 = perf_counter_ns()
    elif picked == kernels.ampere_128x128_4:
        comptime cfg = kernels.ampere_128x128_4
        for _ in range(WARMUP):
            splitk_gemm[transpose_b=False, config=cfg](cours, av, bv, P, ws, ctx)
        ctx.synchronize()
        t2 = perf_counter_ns()
        for _ in range(REPS):
            splitk_gemm[transpose_b=False, config=cfg](cours, av, bv, P, ws, ctx)
        ctx.synchronize()
        t3 = perf_counter_ns()
    else:
        raise Error(
            "select_config picked a tile this sweep does not instantiate"
            " (likely ampere_256x128_3, which is A100-gated)"
        )

    var us_a = Float64(t1 - t0) / 1000.0 / Float64(REPS)
    var us_b = Float64(t3 - t2) / 1000.0 / Float64(REPS)

    var worst = Float64(0.0)
    with c_ref.map_to_host() as hr:
        with c_ours.map_to_host() as ho:
            for i in range(M * N):
                var d = abs(Float64(hr[i]) - Float64(ho[i]))
                if d > worst:
                    worst = d

    var saved = (us_a - us_b) * Float64(COUNT)
    total_saved_us += saved

    print(
        "  ", LABEL, " [", M, "x", K, "] @ [", K, "x", N, "]",
        "  x", COUNT, "  P=", P,
        "  A ", us_a, "us  B ", us_b, "us  ", us_a / us_b, "x",
        "  |A-B| ", worst,
        "  saves ", saved, "us/step",
        sep="",
    )

    # Keep every operand alive past the last launch. A view does not own its
    # buffer, and Mojo destroys at last use -- see bench_splitk_persistent_
    # workspace.mojo's keep-alive note, which cost one CUDA_ERROR_ILLEGAL_ADDRESS
    # to learn.
    _ = ab^
    _ = bb^
    _ = c_ref^
    _ = c_ours^


def choose_partitions(
    M: Int, N: Int, K: Int, BM: Int, BN: Int, BK: Int, sm_count: Int,
    max_p: Int = 48,
) -> Int:
    """Pick `num_k_partitions` to fill the machine once, and no more.

    The sweep on a 5090 (170 SMs) puts the knee exactly at the wave boundary.
    Blocks launched is `tiles * P` where `tiles = ceildiv(M,BM)*ceildiv(N,BN)`:

        [256 x 2592] @ [2592 x  256]   4 tiles   P=24 ->  96 blocks   11.23 us
        [256 x 2592] @ [2592 x 1024]  16 tiles   P= 8 -> 128 blocks   20.79 us
                                                 P=12 -> 192 blocks   28.75 us

    128 blocks fit in one wave on 170 SMs and 192 do not, and the 192-block
    point is 38% SLOWER than the 128-block one despite doing the same work in
    more parallel pieces. So: maximise P subject to `tiles * P <= sm_count`.

    ⚠ LEGALITY IS NOT MONOTONE IN P, so this scans instead of breaking. A P is
    usable only if it BOTH avoids the overrun and covers K (see `splitk_gemm`):

        (P - 1) * part <  K        no block starts past the end
        P       * part >= K        the partitions actually span K

    At K=2592, BK=16 that leaves 1..15, 17, 18, 21, 24, 27, 33, 41. P=16 and 20
    overrun; P=23 and 40 UNDERCOVER, which does not crash -- it silently drops
    the tail of the contraction. A loop that stopped at the first failure would
    return 15 and miss the best point by a wide margin.

    Returns 1 when nothing better is available, which callers should read as
    "do not use split-K for this shape".
    """
    var tiles = ceildiv(M, BM) * ceildiv(N, BN)
    var best = 1
    for P in range(2, max_p + 1):
        if tiles * P > sm_count:
            continue
        var part = align_up(K // P, BK)
        if (P - 1) * part >= K:
            continue                      # overruns: would fault
        if P * part < K:
            continue                      # undercovers: would be WRONG
        if P > best:
            best = P
    return best


def sweep_partitions[
    M: Int, K: Int, N: Int, LABEL: StaticString
](ctx: DeviceContext, mut ws: SplitKWorkspace[DT]) raises:
    """Sweep `num_k_partitions` past what `select_config` would choose.

    `select_config` caps P by `min_k_partition = 1024`: it will not cut K into
    pieces smaller than 1024, so at K=2592 it stops at P=2. That is a heuristic
    in a HOST-SIDE CHOOSER, not a constraint of the kernel -- the kernel takes
    the partition count as a runtime argument and MAX's own guard is only
    `K >= P * BK` (BK=16 here, so P=8 needs K>=128).

    Now that we launch the kernel ourselves, the cap is ours to pick. It is
    worth picking deliberately, because at these shapes the grid is tiny:
    [256 x 2592] @ [2592 x 256] with BM=BN=128 is 2x2 tiles, so P=2 puts
    EIGHT blocks on a 170-SM card. That is why the P=2 timings barely move
    between N=256 and N=1024 -- 4x the work fits in the same latency because
    the machine was idle either way. More partitions is more blocks.

    ⚠ Higher P is not free and not exact. Each partition is a separate fp32
    accumulation that the reduce then sums, so |A - B| stops being 0 as P
    grows -- that is arithmetic, not a bug, but it is a reason to choose P
    on evidence rather than maximising it.
    """
    comptime kernels = MatmulKernels[DT, DT, DT, False]()
    comptime cfg = kernels.ampere_128x128_4
    var picked = select_config[DT, DT, DT, False](M, N, K, ctx)
    if not (picked == cfg):
        print("  ", LABEL, ": select_config chose another tile; skipping", sep="")
        return

    var ab = ctx.enqueue_create_buffer[DT](M * K)
    var bb = ctx.enqueue_create_buffer[DT](K * N)
    var c_ref = ctx.enqueue_create_buffer[DT](M * N)
    var c_ours = ctx.enqueue_create_buffer[DT](M * N)
    ab.enqueue_fill(Float32(0.01))
    bb.enqueue_fill(Float32(0.02))
    ctx.synchronize()

    var av = TileTensor(ab, row_major[M, K]())
    var bv = TileTensor(bb, row_major[K, N]())
    var cref = TileTensor(c_ref, row_major[M, N]())
    var cours = TileTensor(c_ours, row_major[M, N]())

    for _ in range(WARMUP):
        max_matmul[target="gpu"](cref, av, bv, ctx)
    ctx.synchronize()

    var tiles = ceildiv(M, 128) * ceildiv(N, 128)
    comptime sm_count = ctx.default_device_info.sm_count
    var want = choose_partitions(
        M, N, K,
        cfg.block_tile_shape[0], cfg.block_tile_shape[1],
        cfg.block_tile_shape[2],
        sm_count,
    )
    print(
        "  ", LABEL, " [", M, "x", K, "] @ [", K, "x", N, "]  ",
        tiles, " tiles, ", sm_count, " SMs",
        "  select_config P=", picked.num_k_partitions,
        "  chooser P=", want, " (", tiles * want, " blocks)", sep="",
    )

    # Candidate partition counts. The overrun rule (see splitk_gemm) is
    # `(P-1) * align_up(K//P, BK) < K`; it is fully comptime here, so P values
    # that would fault are never instantiated rather than raised on.
    comptime PS = [2, 3, 4, 5, 6, 8, 10, 12, 16, 20, 23, 24, 33, 40, 41]
    comptime BK = cfg.block_tile_shape[2]
    comptime for i in range(len(PS)):
        comptime P = PS[i]
        comptime PART = ((K // P) + BK - 1) // BK * BK
        comptime NO_OVERRUN = (P - 1) * PART < K
        comptime COVERS = P * PART >= K
        comptime FITS = NO_OVERRUN and COVERS
        comptime if not NO_OVERRUN:
            print(
                "      P=", P, "  SKIPPED (overrun): (P-1)*align_up(K//P,BK) = ",
                (P - 1) * PART, " >= K=", K,
                " -> the last block would read past the operands", sep="",
            )
        comptime if NO_OVERRUN and not COVERS:
            print(
                "      P=", P, "  SKIPPED (undercovers): P*align_up(K//P,BK) = ",
                P * PART, " < K=", K, " -> ", K - P * PART,
                " contraction elements silently dropped, rel err would be ",
                Float64(K - P * PART) / Float64(K), sep="",
            )
        comptime if FITS:
            if P * M * N <= ws.capacity:
                for _ in range(WARMUP):
                    splitk_gemm[transpose_b=False, config=cfg](
                        cours, av, bv, P, ws, ctx
                    )
                ctx.synchronize()
                var ta = perf_counter_ns()
                for _ in range(REPS):
                    splitk_gemm[transpose_b=False, config=cfg](
                        cours, av, bv, P, ws, ctx
                    )
                ctx.synchronize()
                var tb = perf_counter_ns()

                # Report RELATIVE error beside the absolute one. Each
                # partition is its own fp32 accumulation, so the difference
                # grows with P; 7e-7 means nothing until you know the entries
                # are ~0.5, at which point it is ~12 ulps and fine.
                var worst = Float64(0.0)
                var mag = Float64(0.0)
                with c_ref.map_to_host() as hr:
                    with c_ours.map_to_host() as ho:
                        for j in range(M * N):
                            var r = abs(Float64(hr[j]))
                            if r > mag:
                                mag = r
                            var d = abs(r - abs(Float64(ho[j])))
                            if d > worst:
                                worst = d

                print(
                    "      P=", P, "  blocks=", tiles * P,
                    "  ", Float64(tb - ta) / 1000.0 / Float64(REPS), "us",
                    "  |A-B| ", worst,
                    "  rel ", (worst / mag) if mag > 0.0 else 0.0, sep="",
                )

    _ = ab^
    _ = bb^
    _ = c_ref^
    _ = c_ours^


def main() raises:
    comptime if not has_nvidia_gpu_accelerator():
        print("NVIDIA only -- build with `pixi run -e nvidia`.")
    else:
        with DeviceContext() as ctx:
            # 8 partitions is select_config's ceiling; the widest M*N below is
            # 1024*256. Sized once, before anything that would be captured.
            var ws = SplitKWorkspace[DT](ctx, 8 * 1024 * 1024)

            var total = Float64(0.0)
            print("ACT dW shapes, BATCH=16 DIM=256 FF=1024 K=60 ENC=4 DEC=1")
            print("dW GEMM is [in_features x B] @ [B x out_features], B = batch*tokens")
            print()
            print("-- transformer encoder, B = 16*162 = 2592 --------------------")
            # 4 layers x {q,k,v,ao} = 16, plus the decoder's cross-attention
            # k and v, which run over the 162 MEMORY tokens, not the 60 queries.
            sweep_one[256, 2592, 256, 18, "enc attn qkv/out + dec cross k,v"](ctx, ws, total)
            sweep_one[256, 2592, 1024, 4, "enc ff1                        "](ctx, ws, total)
            sweep_one[1024, 2592, 256, 4, "enc ff2                        "](ctx, ws, total)

            print()
            print("-- controls: these should NOT split (B < 2048) ---------------")
            sweep_one[256, 992, 256, 16, "CVAE enc attn, B=16*62=992     "](ctx, ws, total)
            sweep_one[256, 960, 256, 6, "decoder self/cross-q, B=960    "](ctx, ws, total)
            sweep_one[256, 960, 1024, 1, "decoder ff1                    "](ctx, ws, total)

            print()
            print("-- conv dW family (M=out_ch, N=col, K=images*OH*OW) ----------")
            # MEASURED in MODULAR_MATMUL_ALLOC_REPORT.md Measurement 4.
            sweep_one[128, 32768, 128, 1, "measured in the report         "](ctx, ws, total)
            # ResNet18 stem dW, N padded 160 -> 256 to clear `n % 128`.
            sweep_one[64, 307200, 256, 2, "ResNet18 stem dW (N padded)    "](ctx, ws, total)

            print()
            print("TOTAL saved per training step (listed shapes only):", total, "us")
            print(
                "  = ", total / 1000.0, " ms/step; over 1000 steps, ",
                total / 1000.0, " s", sep="",
            )
            print()
            print(
                "⚠ Counts are from the layer list, not from a run. Confirm with"
                " the LOGGING_LEVEL=INFO command in this file's docstring"
                " before quoting the total."
            )

            print()
            print("-- P sweep: select_config's cap is a heuristic, not a limit --")
            print("   (grid is tiny at these shapes; more partitions = more SMs)")
            sweep_partitions[256, 2592, 256, "enc attn      "](ctx, ws)
            sweep_partitions[256, 2592, 1024, "enc ff1       "](ctx, ws)
            sweep_partitions[1024, 2592, 256, "enc ff2       "](ctx, ws)
            _ = ws^
