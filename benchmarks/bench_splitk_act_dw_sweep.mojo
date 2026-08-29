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

from std.math import ceildiv
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
                "  = ", total / 1000.0, " ms/step; at 1000 steps, ",
                total / 1e6, " s", sep="",
            )
            print()
            print(
                "⚠ Counts are from the layer list, not from a run. Confirm with"
                " the LOGGING_LEVEL=INFO command in this file's docstring"
                " before quoting the total."
            )
            _ = ws^
