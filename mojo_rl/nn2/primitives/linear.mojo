"""Linear[IN, OUT] — affine `y = x·W + b`. The lighthouse leaf.

State:
  * `ts: TargetStorage` carries `target_tag` + (optional) `ctx`; the
    method-level `assert_tag_for` polices misuse.
  * `weight: Param["weight", True, IN*OUT]` + `bias: Param["bias", False, OUT]`
    — `True`/`False` indicates whether AdamW weight-decay applies. The
    `for_each_param` / `zero_grad` bodies are one call each into the
    reflection-walked `_auto` helpers.
  * `_cached_input_ptr` aliases the orchestrator's input slab (no copy).
    Sequential / ComputeGraph guarantee that slab stays live until
    backward completes.
  * `amp: LinearAMPState[IN, OUT]` owns the bf16 scratch + cast helpers
    when `POLICY.compute_dtype == bf16`. The bf16 weight is re-cast on
    every fwd/bwd: a `w_dirty` flag was tried, but no caller flipped it
    after Adam updates so the cache went stale at step 1. Re-casting
    costs `IN*OUT` scalar ops vs `BATCH*IN*OUT` for the matmul itself —
    negligible.
  * `backward[mode]` collapses backward + backward_input; the
    `mode="input_only"` shortcut skips param grads (used by SAC actor
    loss propagating through the critic).

**BACKWARD-ORDER INVARIANT (critical)**: param grads run BEFORE
input grad. `_cached_input_ptr` aliases the orchestrator's input slab
— the same slab that `grad_input` writes into. If `grad_input` ran
first, it would clobber the cache before `grad_w` could read it. So
the order is `grad_b → grad_w → grad_input`, and Sequential's
backward walks children in the matching reverse-order.
"""

from std.math import ceildiv
from std.memory import alloc
from std.sys import CompilationTarget
from std.gpu import global_idx, thread_idx, block_idx
from std.gpu.host import DeviceContext, DeviceBuffer
from std.gpu.memory import AddressSpace
from std.gpu.primitives import block
from layout import Layout, LayoutTensor, TileTensor, row_major
from linalg.matmul import matmul as max_matmul
from linalg.matmul.cpu.apple_accelerate import (
    get_cblas_f32_function,
    _CBLASOrder,
    _CBLASTranspose,
)

from ..constants import DT, CPU_SIMD_W, TPB
from ..core import (
    Initializer,
    AMPPolicy,
    NoAMP,
    Param,
    for_each_param_auto,
    zero_grad_auto,
    cast_fp32_to_bf16,
    cast_bf16_to_fp32,
    LinearAMPState,
    ParamVisitor,
)
from ..core.module import Module, typed_view, typed_view_mut
from ..core.target_storage import (
    require_ctx,
    TargetStorage,
    assert_tag_for,
    ensure_gpu_buffer,
)


# ──────────────────────────────────────────────────────────────────────
# GPU kernels — module-level so enqueue_function can bind them.
# Lifted from v1; `_cache_input_kernel` removed (no input copy anymore).
# ──────────────────────────────────────────────────────────────────────


def _bias_add_kernel[
    BATCH: Int,
    OUT: Int,
](
    output: LayoutTensor[DT, Layout.row_major(BATCH, OUT), MutAnyOrigin],
    bias: LayoutTensor[DT, Layout.row_major(OUT), MutAnyOrigin],
):
    var idx = Int(global_idx.x)
    var total = BATCH * OUT
    if idx < total:
        var b = idx // OUT
        var j = idx % OUT
        output[b, j] = rebind[Scalar[DT]](output[b, j]) + rebind[Scalar[DT]](bias[j])


def _grad_bias_reduce_kernel[
    BATCH: Int,
    OUT: Int,
](
    grad_output: LayoutTensor[DT, Layout.row_major(BATCH, OUT), MutAnyOrigin],
    grad_b: LayoutTensor[DT, Layout.row_major(OUT), MutAnyOrigin],
):
    """grad_b[col] += Σ_b grad_output[b, col]. ONE BLOCK per output column,
    `TPB` threads striding over BATCH + a `block.sum` reduction → full
    occupancy (OUT·TPB thread-slots) vs the old one-thread-per-column
    serial-BATCH-loop kernel (only OUT threads). Mirrors LayerNorm's
    grad_beta reduction. Launch: grid_dim=OUT, block_dim=TPB."""
    var col = Int(block_idx.x)
    var t = Int(thread_idx.x)
    if col >= OUT:
        return
    var my_s: Scalar[DT] = 0.0
    var bi = t
    while bi < BATCH:
        my_s += rebind[Scalar[DT]](grad_output[bi, col])
        bi += TPB
    var total = block.sum[block_size=TPB, broadcast=False](val=my_s)
    if t == 0:
        grad_b[col] = rebind[Scalar[DT]](grad_b[col]) + total[0]


def _transpose_kernel[ROWS: Int, COLS: Int](
    src: LayoutTensor[DT, Layout.row_major(ROWS, COLS), MutAnyOrigin],
    dst: LayoutTensor[DT, Layout.row_major(COLS, ROWS), MutAnyOrigin],
):
    """dst[c,r] = src[r,c]. One thread per source element. Used to materialize
    cacheᵀ[IN,BATCH] so grad_w = cacheᵀ @ grad_output runs through `max_matmul`
    (tensor cores) — `max_matmul` rejects `transpose_a`. Shared by Linear,
    LinearAct, and NoisyLinear."""
    var idx = Int(global_idx.x)
    if idx < ROWS * COLS:
        var r = idx // COLS
        var c = idx % COLS
        dst[c, r] = rebind[Scalar[DT]](src[r, c])


def _accum_kernel[N: Int](
    dst: LayoutTensor[DT, Layout.row_major(N), MutAnyOrigin],
    src: LayoutTensor[DT, Layout.row_major(N), MutAnyOrigin],
):
    """dst[i] += src[i]. Accumulates a `max_matmul` dW result into grad_w,
    preserving the `vjp` accumulate contract."""
    var idx = Int(global_idx.x)
    if idx < N:
        dst[idx] = rebind[Scalar[DT]](dst[idx]) + rebind[Scalar[DT]](src[idx])


# ──────────────────────────────────────────────────────────────────────
# Linear.
# ──────────────────────────────────────────────────────────────────────


struct Linear[IN: Int, OUT: Int](Module):
    comptime ARITY: Int = 1
    comptime IN_DIMS = InlineArray[Int, 1](fill=Self.IN)
    comptime OUT_DIM = Self.OUT
    comptime W_SIZE = Self.IN * Self.OUT
    comptime B_SIZE = Self.OUT

    @staticmethod
    def display_label() -> String:
        return String("Linear")

    # Parameters — visible to reflection, walked by for_each_param_auto.
    var weight: Param["weight", True,  Self.W_SIZE]
    var bias:   Param["bias",   False, Self.B_SIZE]

    # Forward-time pointer alias of the orchestrator's input slab.
    # Backward reads from this directly; no copy at forward time.
    var _cached_input_ptr: Optional[UnsafePointer[Scalar[DT], MutAnyOrigin]]

    # AMP scratch (lazy-allocated on first bf16 call).
    var amp: LinearAMPState[Self.IN, Self.OUT]

    # Backward grad_w temporaries (GPU) — mirror LinearAct. `cacheT_dev` holds
    # cacheᵀ[IN, BATCH] (lazy, BATCH-sized); `dW_tmp_dev` holds [IN, OUT] = W_SIZE
    # (fixed). Let grad_w run via transpose + max_matmul (tensor cores) instead
    # of the old naive serial kernel. Lazy alloc is CUDA-graph-capture-safe
    # (allocated on the pre-capture settle call, reused on every replay).
    var cacheT_dev: Optional[DeviceBuffer[DT]]
    var cacheT_n: Int
    var dW_tmp_dev: Optional[DeviceBuffer[DT]]

    var ts: TargetStorage

    # ----- Defaultable -----------------------------------------------------

    def __init__(out self):
        self.weight = Param["weight", True,  Self.W_SIZE]()
        self.bias   = Param["bias",   False, Self.B_SIZE]()
        self._cached_input_ptr = None
        self.amp = LinearAMPState[Self.IN, Self.OUT].make()
        self.cacheT_dev = None
        self.cacheT_n = 0
        self.dW_tmp_dev = None
        self.ts = TargetStorage.make_uninit()

    # ----- Factories -------------------------------------------------------

    @staticmethod
    def make[
        target: StaticString, INIT: Initializer
    ](
        ctx: Optional[DeviceContext] = None,
    ) raises -> Self:
        """Unified CPU/GPU factory. `ctx=None` on CPU; required on GPU."""
        comptime assert target == "cpu" or target == "gpu", (
            "Linear: target must be 'cpu' or 'gpu'"
        )
        var lin = Self()
        comptime if target == "cpu":
            lin.weight = Param["weight", True,  Self.W_SIZE].make_cpu()
            lin.bias   = Param["bias",   False, Self.B_SIZE].make_cpu()
            INIT.init_weight(
                lin.weight.value_unsafe_ptr_cpu(),
                Self.W_SIZE, Self.IN, Self.OUT,
            )
            INIT.init_bias(lin.bias.value_unsafe_ptr_cpu(), Self.B_SIZE)
            lin.ts = TargetStorage.make_cpu()
        else:
            var ctx_v = require_ctx["Linear.make[target='gpu']"](ctx)
            lin.weight = Param["weight", True,  Self.W_SIZE].make_gpu(ctx_v)
            lin.bias   = Param["bias",   False, Self.B_SIZE].make_gpu(ctx_v)
            # Init weights/biases on host via INIT, then upload.
            var w_host = ctx_v.enqueue_create_host_buffer[DT](Self.W_SIZE)
            var b_host = ctx_v.enqueue_create_host_buffer[DT](Self.B_SIZE)
            ctx_v.synchronize()
            INIT.init_weight(w_host.unsafe_ptr(), Self.W_SIZE, Self.IN, Self.OUT)
            INIT.init_bias(b_host.unsafe_ptr(), Self.B_SIZE)
            ctx_v.enqueue_copy(lin.weight.val.dev.value(), w_host)
            ctx_v.enqueue_copy(lin.bias.val.dev.value(),   b_host)
            ctx_v.synchronize()
            # Fixed [IN, OUT] dW scratch for the max_matmul grad_w path; cacheT
            # stays None (lazily sized to BATCH on first backward).
            lin.dW_tmp_dev = ctx_v.enqueue_create_buffer[DT](Self.W_SIZE)
            lin.ts = TargetStorage.make_gpu(ctx_v)
        return lin^

    # ----- Forward ---------------------------------------------------------

    def forward[
        target: StaticString,
        BATCH: Int,
        POLICY: AMPPolicy = NoAMP,
    ](
        mut self,
        var *inputs: TileTensor[
            dtype=DT, address_space=AddressSpace.GENERIC,
            element_size=1, origin=MutAnyOrigin, ...,
        ],
        mut output: TileTensor[
            mut=True, dtype=DT, address_space=AddressSpace.GENERIC,
            element_size=1, origin=MutAnyOrigin, ...,
        ],
    ) raises:
        assert_tag_for["Linear", target](self.ts.target_tag)
        var input_v = typed_view[BATCH, Self.IN](inputs[0])
        var output_v = typed_view_mut[BATCH, Self.OUT](output)

        var in_p = rebind[UnsafePointer[Scalar[DT], MutAnyOrigin]](input_v.ptr)
        var out_p = rebind[UnsafePointer[Scalar[DT], MutAnyOrigin]](output_v.ptr)

        # Save pointer alias for backward — NO copy.
        self._cached_input_ptr = in_p

        comptime if target == "cpu":
            comptime if POLICY.compute_dtype == DT:
                var w_tt = TileTensor(
                    self.weight.val.cpu, row_major[Self.IN, Self.OUT](),
                )
                max_matmul[target="cpu"](output_v, input_v, w_tt, None)
            else:
                comptime assert POLICY.compute_dtype == DType.bfloat16, (
                    "Linear CPU supports only fp32 and bf16 compute_dtype"
                )
                self.amp.ensure_cpu(BATCH)

                var w_bf16_p = rebind[
                    UnsafePointer[Scalar[DType.bfloat16], MutAnyOrigin]
                ](self.amp.w_bf16_cpu.unsafe_ptr())
                cast_fp32_to_bf16[target="cpu", N=Self.W_SIZE](
                    self.weight.value_unsafe_ptr_cpu(), w_bf16_p,
                )
                var in_bf16_p = rebind[
                    UnsafePointer[Scalar[DType.bfloat16], MutAnyOrigin]
                ](self.amp.in_bf16_cpu.unsafe_ptr())
                cast_fp32_to_bf16[target="cpu", N = BATCH * Self.IN](
                    in_p, in_bf16_p,
                )
                var in_bf16_tt = TileTensor(
                    self.amp.in_bf16_cpu, row_major[BATCH, Self.IN](),
                )
                var w_bf16_tt = TileTensor(
                    self.amp.w_bf16_cpu, row_major[Self.IN, Self.OUT](),
                )
                var ou_bf16_tt = TileTensor(
                    self.amp.ou_bf16_cpu, row_major[BATCH, Self.OUT](),
                )
                max_matmul[target="cpu"](
                    ou_bf16_tt, in_bf16_tt, w_bf16_tt, None,
                )
                var ou_bf16_p = rebind[
                    UnsafePointer[Scalar[DType.bfloat16], MutAnyOrigin]
                ](self.amp.ou_bf16_cpu.unsafe_ptr())
                cast_bf16_to_fp32[target="cpu", N = BATCH * Self.OUT](
                    ou_bf16_p, out_p,
                )

            # Bias-add. fp32 regardless of POLICY.
            var b_ptr = self.bias.value_unsafe_ptr_cpu()
            for bi in range(BATCH):
                var row_off = bi * Self.OUT
                var ij = 0
                while ij + CPU_SIMD_W <= Self.OUT:
                    var ov = out_p.load[width=CPU_SIMD_W](row_off + ij)
                    var bv = b_ptr.load[width=CPU_SIMD_W](ij)
                    out_p.store(row_off + ij, ov + bv)
                    ij += CPU_SIMD_W
                while ij < Self.OUT:
                    out_p[row_off + ij] = out_p[row_off + ij] + b_ptr[ij]
                    ij += 1
        else:
            var ctx = self.ts.ctx.value()

            comptime if POLICY.compute_dtype == DT:
                var weight_tt = TileTensor(
                    self.weight.val.dev.value(),
                    row_major[Self.IN, Self.OUT](),
                )
                max_matmul[target="gpu"](output_v, input_v, weight_tt, ctx)
            else:
                comptime assert POLICY.compute_dtype == DType.bfloat16, (
                    "Linear supports only fp32 and bf16 compute_dtype"
                )
                self.amp.ensure_gpu(BATCH, ctx)

                var w_bf16_p = rebind[
                    UnsafePointer[Scalar[DType.bfloat16], MutAnyOrigin]
                ](self.amp.w_bf16_dev.value().unsafe_ptr())
                cast_fp32_to_bf16[target="gpu", N=Self.W_SIZE](
                    rebind[UnsafePointer[Scalar[DT], MutAnyOrigin]](
                        self.weight.val.dev.value().unsafe_ptr()
                    ),
                    w_bf16_p,
                    ctx,
                )

                var in_bf16_p = rebind[
                    UnsafePointer[Scalar[DType.bfloat16], MutAnyOrigin]
                ](self.amp.in_bf16_dev.value().unsafe_ptr())
                cast_fp32_to_bf16[target="gpu", N = BATCH * Self.IN](
                    in_p, in_bf16_p, ctx,
                )
                var in_bf16_tt = TileTensor(
                    self.amp.in_bf16_dev.value(),
                    row_major[BATCH, Self.IN](),
                )
                var w_bf16_tt = TileTensor(
                    self.amp.w_bf16_dev.value(),
                    row_major[Self.IN, Self.OUT](),
                )
                var ou_bf16_tt = TileTensor(
                    self.amp.ou_bf16_dev.value(),
                    row_major[BATCH, Self.OUT](),
                )
                max_matmul[target="gpu"](
                    ou_bf16_tt, in_bf16_tt, w_bf16_tt, ctx,
                )
                var ou_bf16_p = rebind[
                    UnsafePointer[Scalar[DType.bfloat16], MutAnyOrigin]
                ](self.amp.ou_bf16_dev.value().unsafe_ptr())
                cast_bf16_to_fp32[target="gpu", N = BATCH * Self.OUT](
                    ou_bf16_p, out_p, ctx,
                )

            # Bias add (fp32, both branches).
            comptime out_layout = Layout.row_major(BATCH, Self.OUT)
            comptime bias_layout = Layout.row_major(Self.OUT)
            var output_lt = LayoutTensor[DT, out_layout, MutAnyOrigin](out_p)
            var bias_lt = LayoutTensor[DT, bias_layout, MutAnyOrigin](
                self.bias.val.dev.value()
            )
            comptime n_blocks_ba = (BATCH * Self.OUT + TPB - 1) // TPB
            comptime ba_kernel = _bias_add_kernel[BATCH, Self.OUT]
            ctx.enqueue_function[ba_kernel](
                output_lt, bias_lt,
                grid_dim=n_blocks_ba, block_dim=TPB,
            )

    # ----- Backward --------------------------------------------------------

    def vjp[
        target: StaticString,
        BATCH: Int,
        POLICY: AMPPolicy = NoAMP,
        mode: StaticString = "all",
    ](
        mut self,
        grad_output: TileTensor[
            dtype=DT, address_space=AddressSpace.GENERIC,
            element_size=1, origin=MutAnyOrigin, ...,
        ],
        mut *grad_inputs: TileTensor[
            mut=True, dtype=DT, address_space=AddressSpace.GENERIC,
            element_size=1, origin=MutAnyOrigin, ...,
        ],
    ) raises:
        """Combined backward — single source of truth that fixes the
        param-before-input order by calling the two phases in sequence
        (S7, 2026-06-07). Direct callers (ComputeGraph, non-Sequential
        combinators, tests) keep using this; `Sequential` calls the two
        phases directly so the order is enforced by the orchestrator."""
        comptime assert (
            mode == "all" or mode == "input_only"
        ), "mode must be 'all' or 'input_only'"
        self.vjp_param_grads[target, BATCH, POLICY=POLICY, mode=mode](
            grad_output
        )
        self.vjp_grad_input[target, BATCH, POLICY=POLICY, mode=mode](
            grad_output, *grad_inputs
        )

    def vjp_param_grads[
        target: StaticString,
        BATCH: Int,
        POLICY: AMPPolicy = NoAMP,
        mode: StaticString = "all",
    ](
        mut self,
        grad_output: TileTensor[
            dtype=DT, address_space=AddressSpace.GENERIC,
            element_size=1, origin=MutAnyOrigin, ...,
        ],
    ) raises:
        """Phase 1 (S7): grad_b + grad_w. Reads grad_output + the cached
        forward input (`_cached_input_ptr`); writes bias/weight grads.
        Skipped entirely under `mode == "input_only"`. MUST run before
        `vjp_grad_input` — grad_input may clobber the cache slab."""
        comptime assert (
            mode == "all" or mode == "input_only"
        ), "mode must be 'all' or 'input_only'"
        comptime if mode == "all":
            assert_tag_for["Linear", target](self.ts.target_tag)
            var grad_output_v = typed_view[BATCH, Self.OUT](grad_output)
            var go_p = rebind[UnsafePointer[Scalar[DT], MutAnyOrigin]](grad_output_v.ptr)

            comptime if target == "cpu":
                # ── (1) grad_b += column-sum(grad_output) ───
                var gb_ptr = self.bias.grad_unsafe_ptr_cpu()
                for bi in range(BATCH):
                    var row_off = bi * Self.OUT
                    var gj = 0
                    while gj + CPU_SIMD_W <= Self.OUT:
                        var gbv = gb_ptr.load[width=CPU_SIMD_W](gj)
                        var gov = go_p.load[width=CPU_SIMD_W](row_off + gj)
                        gb_ptr.store(gj, gbv + gov)
                        gj += CPU_SIMD_W
                    while gj < Self.OUT:
                        gb_ptr[gj] = gb_ptr[gj] + go_p[row_off + gj]
                        gj += 1

                # ── (2) grad_w += cache^T @ grad_output ─────
                # Reads cache via _cached_input_ptr — MUST come before
                # grad_input write since grad_input may alias the cache slab.
                var gw_ptr = self.weight.grad_unsafe_ptr_cpu()
                var cache_ptr = self._cached_input_ptr.value()
                comptime if CompilationTarget.is_macos() and DT == DType.float32:
                    # Apple Accelerate: dW += cache.T @ grad_output in one
                    # cblas_sgemm call (transpose_a, beta=1). No temp alloc.
                    var cblas_gemm = get_cblas_f32_function()
                    cblas_gemm(
                        _CBLASOrder.ROW_MAJOR,
                        _CBLASTranspose.TRANSPOSE,
                        _CBLASTranspose.NO_TRANSPOSE,
                        Int32(Self.IN),
                        Int32(Self.OUT),
                        Int32(BATCH),
                        Float32(1.0),
                        rebind[UnsafePointer[Float32, ImmutAnyOrigin]](
                            cache_ptr
                        ),
                        Int32(Self.IN),
                        rebind[UnsafePointer[Float32, ImmutAnyOrigin]](go_p),
                        Int32(Self.OUT),
                        Float32(1.0),
                        rebind[UnsafePointer[Float32, MutAnyOrigin]](gw_ptr),
                        Int32(Self.OUT),
                    )
                else:
                    comptime gw_n = Self.IN * Self.OUT
                    var cT_buf: UnsafePointer[
                        Scalar[DT], MutAnyOrigin
                    ] = alloc[Scalar[DT]](BATCH * Self.IN)
                    var dW_tmp_buf: UnsafePointer[
                        Scalar[DT], MutAnyOrigin
                    ] = alloc[Scalar[DT]](gw_n)
                    for bi in range(BATCH):
                        for i in range(Self.IN):
                            cT_buf[i * BATCH + bi] = cache_ptr[
                                bi * Self.IN + i
                            ]
                    var cT_tt = TileTensor(
                        cT_buf, row_major[Self.IN, BATCH](),
                    )
                    var dW_tmp_tt = TileTensor(
                        dW_tmp_buf, row_major[Self.IN, Self.OUT](),
                    )
                    max_matmul[target="cpu"](
                        dW_tmp_tt, cT_tt, grad_output_v, None,
                    )
                    var dw_i = 0
                    while dw_i + CPU_SIMD_W <= gw_n:
                        var gw_v = gw_ptr.load[width=CPU_SIMD_W](dw_i)
                        var dt_v = dW_tmp_buf.load[width=CPU_SIMD_W](dw_i)
                        gw_ptr.store(dw_i, gw_v + dt_v)
                        dw_i += CPU_SIMD_W
                    while dw_i < gw_n:
                        gw_ptr[dw_i] = gw_ptr[dw_i] + dW_tmp_buf[dw_i]
                        dw_i += 1
                    dW_tmp_buf.free()
                    cT_buf.free()
            else:
                var ctx = self.ts.ctx.value()

                # ── (1) grad_b += column-sum(grad_output) — block-per-column
                #        reduction (full occupancy), replaces the serial kernel.
                comptime gb_layout = Layout.row_major(Self.OUT)
                var go_lt = LayoutTensor[
                    DT, Layout.row_major(BATCH, Self.OUT), MutAnyOrigin
                ](go_p)
                var gb_lt = LayoutTensor[DT, gb_layout, MutAnyOrigin](
                    self.bias.grd.dev.value()
                )
                comptime gb_kernel = _grad_bias_reduce_kernel[BATCH, Self.OUT]
                ctx.enqueue_function[gb_kernel](
                    go_lt, gb_lt,
                    grid_dim=Self.OUT, block_dim=TPB,
                )

                # ── (2) grad_w += cacheᵀ @ grad_output — transpose + max_matmul
                #        (tensor cores) + accumulate, replaces the naive serial
                #        kernel. Reads cache; must precede the grad_input matmul
                #        (which may alias the cache slab).
                ensure_gpu_buffer(
                    self.cacheT_dev, self.cacheT_n, BATCH * Self.IN, ctx,
                )
                var cache_lt = LayoutTensor[
                    DT, Layout.row_major(BATCH, Self.IN), MutAnyOrigin
                ](self._cached_input_ptr.value())
                var cacheT_lt = LayoutTensor[
                    DT, Layout.row_major(Self.IN, BATCH), MutAnyOrigin
                ](self.cacheT_dev.value())
                comptime n_blocks_t = (BATCH * Self.IN + TPB - 1) // TPB
                comptime t_kernel = _transpose_kernel[BATCH, Self.IN]
                ctx.enqueue_function[t_kernel](
                    cache_lt, cacheT_lt,
                    grid_dim=n_blocks_t, block_dim=TPB,
                )
                var cacheT_tt = TileTensor(
                    self.cacheT_dev.value(), row_major[Self.IN, BATCH](),
                )
                var dW_tmp_tt = TileTensor(
                    self.dW_tmp_dev.value(), row_major[Self.IN, Self.OUT](),
                )
                max_matmul[target="gpu"](
                    dW_tmp_tt, cacheT_tt, grad_output_v, ctx,
                )
                comptime gw_layout = Layout.row_major(Self.W_SIZE)
                var gw_lt = LayoutTensor[DT, gw_layout, MutAnyOrigin](
                    self.weight.grd.dev.value()
                )
                var dW_tmp_lt = LayoutTensor[DT, gw_layout, MutAnyOrigin](
                    self.dW_tmp_dev.value()
                )
                comptime n_blocks_acc = (Self.W_SIZE + TPB - 1) // TPB
                comptime acc_kernel = _accum_kernel[Self.W_SIZE]
                ctx.enqueue_function[acc_kernel](
                    gw_lt, dW_tmp_lt,
                    grid_dim=n_blocks_acc, block_dim=TPB,
                )

    def vjp_grad_input[
        target: StaticString,
        BATCH: Int,
        POLICY: AMPPolicy = NoAMP,
        mode: StaticString = "all",
    ](
        mut self,
        grad_output: TileTensor[
            dtype=DT, address_space=AddressSpace.GENERIC,
            element_size=1, origin=MutAnyOrigin, ...,
        ],
        mut *grad_inputs: TileTensor[
            mut=True, dtype=DT, address_space=AddressSpace.GENERIC,
            element_size=1, origin=MutAnyOrigin, ...,
        ],
    ) raises:
        """Phase 2 (S7): grad_input = grad_output @ Wᵀ. May alias the
        cache slab — safe because the orchestrator (or `vjp`) ran
        `vjp_param_grads` first. Runs regardless of `mode`."""
        comptime assert (
            mode == "all" or mode == "input_only"
        ), "mode must be 'all' or 'input_only'"
        assert_tag_for["Linear", target](self.ts.target_tag)
        var grad_output_v = typed_view[BATCH, Self.OUT](grad_output)
        var grad_input_v = typed_view_mut[BATCH, Self.IN](grad_inputs[0])

        var go_p = rebind[UnsafePointer[Scalar[DT], MutAnyOrigin]](grad_output_v.ptr)
        var gi_p = rebind[UnsafePointer[Scalar[DT], MutAnyOrigin]](grad_input_v.ptr)

        comptime if target == "cpu":
            # ── (3) grad_input = grad_output @ W^T (always) ────────────
            comptime if POLICY.compute_dtype == DT:
                var w_tt = TileTensor(
                    self.weight.val.cpu, row_major[Self.IN, Self.OUT](),
                )
                max_matmul[transpose_b=True, target="cpu"](
                    grad_input_v, grad_output_v, w_tt, None,
                )
            else:
                comptime assert POLICY.compute_dtype == DType.bfloat16, (
                    "Linear CPU supports only fp32 and bf16 compute_dtype"
                )
                self.amp.ensure_cpu(BATCH)

                var w_bf16_p = rebind[
                    UnsafePointer[Scalar[DType.bfloat16], MutAnyOrigin]
                ](self.amp.w_bf16_cpu.unsafe_ptr())
                cast_fp32_to_bf16[target="cpu", N=Self.W_SIZE](
                    self.weight.value_unsafe_ptr_cpu(), w_bf16_p,
                )

                var go_bf16_p = rebind[
                    UnsafePointer[Scalar[DType.bfloat16], MutAnyOrigin]
                ](self.amp.ou_bf16_cpu.unsafe_ptr())
                cast_fp32_to_bf16[target="cpu", N = BATCH * Self.OUT](
                    go_p, go_bf16_p,
                )

                var go_bf16_tt = TileTensor(
                    self.amp.ou_bf16_cpu, row_major[BATCH, Self.OUT](),
                )
                var w_bf16_tt = TileTensor(
                    self.amp.w_bf16_cpu, row_major[Self.IN, Self.OUT](),
                )
                var gi_bf16_tt = TileTensor(
                    self.amp.in_bf16_cpu, row_major[BATCH, Self.IN](),
                )
                max_matmul[transpose_b=True, target="cpu"](
                    gi_bf16_tt, go_bf16_tt, w_bf16_tt, None,
                )

                var gi_bf16_p = rebind[
                    UnsafePointer[Scalar[DType.bfloat16], MutAnyOrigin]
                ](self.amp.in_bf16_cpu.unsafe_ptr())
                cast_bf16_to_fp32[target="cpu", N = BATCH * Self.IN](
                    gi_bf16_p, gi_p,
                )
        else:
            var ctx = self.ts.ctx.value()

            # ── (3) grad_input = grad_output @ W^T ─────────────────────
            comptime if POLICY.compute_dtype == DT:
                var weight_tt = TileTensor(
                    self.weight.val.dev.value(),
                    row_major[Self.IN, Self.OUT](),
                )
                max_matmul[transpose_b=True, target="gpu"](
                    grad_input_v, grad_output_v, weight_tt, ctx,
                )
            else:
                comptime assert POLICY.compute_dtype == DType.bfloat16, (
                    "Linear supports only fp32 and bf16 compute_dtype"
                )
                self.amp.ensure_gpu(BATCH, ctx)

                var w_bf16_p = rebind[
                    UnsafePointer[Scalar[DType.bfloat16], MutAnyOrigin]
                ](self.amp.w_bf16_dev.value().unsafe_ptr())
                cast_fp32_to_bf16[target="gpu", N=Self.W_SIZE](
                    rebind[UnsafePointer[Scalar[DT], MutAnyOrigin]](
                        self.weight.val.dev.value().unsafe_ptr()
                    ),
                    w_bf16_p, ctx,
                )

                var go_bf16_p = rebind[
                    UnsafePointer[Scalar[DType.bfloat16], MutAnyOrigin]
                ](self.amp.ou_bf16_dev.value().unsafe_ptr())
                cast_fp32_to_bf16[target="gpu", N = BATCH * Self.OUT](
                    go_p, go_bf16_p, ctx,
                )

                var go_bf16_tt = TileTensor(
                    self.amp.ou_bf16_dev.value(),
                    row_major[BATCH, Self.OUT](),
                )
                var w_bf16_tt = TileTensor(
                    self.amp.w_bf16_dev.value(),
                    row_major[Self.IN, Self.OUT](),
                )
                var gi_bf16_tt = TileTensor(
                    self.amp.in_bf16_dev.value(),
                    row_major[BATCH, Self.IN](),
                )
                max_matmul[transpose_b=True, target="gpu"](
                    gi_bf16_tt, go_bf16_tt, w_bf16_tt, ctx,
                )

                var gi_bf16_p = rebind[
                    UnsafePointer[Scalar[DType.bfloat16], MutAnyOrigin]
                ](self.amp.in_bf16_dev.value().unsafe_ptr())
                cast_bf16_to_fp32[target="gpu", N = BATCH * Self.IN](
                    gi_bf16_p, gi_p, ctx,
                )

    # ----- Param / grad walkers (reflection-derived) ----------------------

    def for_each_param[
        target: StaticString,
        V: ParamVisitor,
    ](mut self, prefix: String, mut visitor: V) raises:
        """Auto-derived via reflection — finds every `Param[NAME, DECAY, SIZE]`
        field and dispatches the visitor."""
        assert_tag_for["Linear", target](self.ts.target_tag)
        for_each_param_auto[Self, V, target](self, prefix, visitor)

    def zero_grad[target: StaticString](mut self) raises:
        """Auto-derived via reflection — clears every Param's grad."""
        assert_tag_for["Linear", target](self.ts.target_tag)
        zero_grad_auto[Self, target](self)

