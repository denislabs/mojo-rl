"""Split-K GEMM with a CALLER-OWNED workspace — the alloc-free, capturable variant.

Context: `docs/MODULAR_MATMUL_ALLOC_REPORT.md` Measurements 4 and 5.
`linalg.matmul.gpu.multistage_gemm` allocates its split-K reduction
workspace on every call (matmul/gpu/__init__.mojo:1845) and frees it at
the end of the call. That costs ~250 us of cuMemAlloc/cuMemFree per GEMM
and makes the GEMM impossible to put inside a CUDA-graph capture.

The report asked Modular to reuse that workspace. This file shows we do
not have to wait for them: every piece `multistage_gemm` uses on that
branch is exported from the SHIPPED `linalg` package, so we can inline
the same ~40 lines against a buffer we own.

  from linalg.matmul.gpu import multistage_gemm_split_k_kernel, split_k_reduce
  from linalg.utils_gpu   import MatmulConfig, MatmulKernels, select_config

Verified importable against MAX 26.5.0 / Mojo 1.0.0 (ed45d567).

  arm A   linalg.matmul                      allocates (baseline)
  arm B   splitk_gemm + SplitKWorkspace      no allocation
  arm C   arm B inside a CUDA-graph capture  the thing Measurement 5 blocked

Run (NVIDIA only — the multistage kernel does not build for Metal):

    pixi run -e nvidia mojo run -I . benchmarks/bench_splitk_persistent_workspace.mojo
    pixi run -e nvidia mojo run -D LOGGING_LEVEL=INFO -I . \
        benchmarks/bench_splitk_persistent_workspace.mojo

`-D LOGGING_LEVEL=INFO` makes MAX name the path it took and print
"K partitions: N" for each call. It is a COMPILE-TIME define; the
same-named environment variable does nothing (verified).
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

from mojo_rl.cuda import CUDAGraph


struct SplitKWorkspace[dtype: DType](Movable):
    """A device buffer reused across split-K GEMMs.

    Sized once for the largest `num_k_partitions * M * N` any call will
    ask for, then handed to every launch. Because it is an ordinary
    long-lived buffer, a capture region containing the GEMM never sees an
    allocation — which is the whole point.
    """

    var buf: DeviceBuffer[Self.dtype]
    var capacity: Int

    def __init__(out self, ctx: DeviceContext, capacity: Int) raises:
        self.buf = ctx.enqueue_create_buffer[Self.dtype](capacity)
        self.capacity = capacity
        ctx.synchronize()


def splitk_gemm[
    c_type: DType,
    a_type: DType,
    b_type: DType, //,
    *,
    transpose_b: Bool,
    config: MatmulConfig[a_type, b_type, c_type, transpose_b],
](
    c: TileTensor[mut=True, c_type, ...],
    a: TileTensor[mut=False, a_type, ...],
    b: TileTensor[mut=False, b_type, ...],
    num_partitions: Int,
    mut ws: SplitKWorkspace[config.split_k_reduction_type],
    ctx: DeviceContext,
) raises:
    """`multistage_gemm`'s split-K branch, verbatim, but on `ws.buf`.

    Mirrors matmul/gpu/__init__.mojo:1840-1915 with the
    `ctx.enqueue_create_buffer` / `_ = work_space_data^` pair removed.

    On NVIDIA the comptime `config` supplies the TILE SHAPE only: the
    partition count is a runtime kernel argument (the kernel slices the
    workspace by `block_idx.z * M * N`), which is why MAX itself passes
    the static `kernels.ampere_*` config alongside a runtime config whose
    `num_k_partitions` differs. `MatmulConfig.__eq__` compares only
    `block_tile_shape` and `num_pipeline_stages`, confirming that split.
    """
    var tensor_c = c.to_layout_tensor()
    var tensor_a = a.to_layout_tensor()
    var tensor_b = b.to_layout_tensor()
    var M = tensor_c.dim[0]()
    var N = tensor_c.dim[1]()

    if num_partitions * M * N > ws.capacity:
        raise Error("SplitKWorkspace too small for this GEMM")

    comptime ws_type = config.split_k_reduction_type
    comptime static_N = tensor_c.layout.shape[1].value()
    comptime ws_layout = Layout.row_major(UNKNOWN_VALUE, UNKNOWN_VALUE, static_N)
    var ws_rt_layout = RuntimeLayout[ws_layout].row_major(
        Index(num_partitions, M, N)
    )
    var ws_lt = LayoutTensor[ws_type, ws_layout, MutAnyOrigin](
        ws.buf, ws_rt_layout
    )

    comptime kern = multistage_gemm_split_k_kernel[
        c_type,
        tensor_c.layout,
        a_type,
        tensor_a.layout,
        b_type,
        tensor_b.layout,
        ws_type,
        ws_lt.layout,
        transpose_b,
        config,
        None,
    ]

    ctx.enqueue_function[kern](
        tensor_c,
        tensor_a,
        tensor_b,
        ws_lt,
        Int32(num_partitions),
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


def main() raises:
    comptime if not has_nvidia_gpu_accelerator():
        print(
            "NVIDIA only — the multistage kernel has no Metal codegen."
            " Build with `pixi run -e nvidia`."
        )
    else:
        # An ACT dW shape: grad_w = x^T @ go, K = batch x tokens.
        # `[256 x 2592] @ [2592 x 256]` measured 90.98 us + an alloc/free
        # pair in MODULAR_MATMUL_ALLOC_REPORT.md Measurement 4.
        comptime M = 256
        comptime K = 2592
        comptime N = 256
        comptime DT = DType.float32
        comptime REPS = 101

        with DeviceContext() as ctx:
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

            var picked = select_config[DT, DT, DT, False](M, N, K, ctx)
            print(
                "select_config: block",
                picked.block_tile_shape,
                " stages",
                picked.num_pipeline_stages,
                " k_partitions",
                picked.num_k_partitions,
            )
            if picked.num_k_partitions <= 1:
                print(
                    "!! this shape does NOT split K here — pick another shape,"
                    " the A/B below is vacuous."
                )

            comptime kernels = MatmulKernels[DT, DT, DT, False]()
            comptime cfg = kernels.ampere_128x128_4
            var ws = SplitKWorkspace[cfg.split_k_reduction_type](
                ctx, 8 * M * N
            )

            # ---- arm A: MAX's matmul (allocates per call) ----------------
            for _ in range(5):
                max_matmul[target="gpu"](cref, av, bv, ctx)
            ctx.synchronize()
            var t0 = perf_counter_ns()
            for _ in range(REPS):
                max_matmul[target="gpu"](cref, av, bv, ctx)
            ctx.synchronize()
            var t1 = perf_counter_ns()

            # ---- arm B: ours (no allocation) ------------------------------
            for _ in range(5):
                splitk_gemm[transpose_b=False, config=cfg](
                    cours, av, bv, picked.num_k_partitions, ws, ctx
                )
            ctx.synchronize()
            var t2 = perf_counter_ns()
            for _ in range(REPS):
                splitk_gemm[transpose_b=False, config=cfg](
                    cours, av, bv, picked.num_k_partitions, ws, ctx
                )
            ctx.synchronize()
            var t3 = perf_counter_ns()

            print(
                "A linalg.matmul     ",
                Float64(t1 - t0) / 1000.0 / Float64(REPS),
                "us/call",
            )
            print(
                "B persistent split-K",
                Float64(t3 - t2) / 1000.0 / Float64(REPS),
                "us/call",
            )

            # ---- correctness: B must equal A ------------------------------
            var worst = Float64(0.0)
            with c_ref.map_to_host() as hr:
                with c_ours.map_to_host() as ho:
                    for i in range(M * N):
                        var d = abs(Float64(hr[i]) - Float64(ho[i]))
                        if d > worst:
                            worst = d
            print("max |A - B| =", worst, " (expect ~1e-4 at fp32/K=2592)")

            # ---- arm C: the same GEMM inside a capture --------------------
            # Measurement 5 aborted the PROCESS here (MAX's error path calls
            # cuStreamSynchronize on a captured stream). If arm B is really
            # allocation-free this now survives and reports a node count.
            print("--- capture attempt (arm C) ---")
            var g = CUDAGraph(ctx)
            if g.is_disabled():
                print("CUDAGraph disabled — no usable stream; skipping arm C.")
            else:
                g.begin_capture()
                splitk_gemm[transpose_b=False, config=cfg](
                    cours, av, bv, picked.num_k_partitions, ws, ctx
                )
                g.end_capture()
                print("captured", g.num_nodes(), "nodes")
                g.replay_on_mojo_stream()
                ctx.synchronize()
                print("replay ok")
