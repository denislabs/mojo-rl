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
from std.gpu import global_idx
from std.gpu.host import DeviceContext, DeviceBuffer
from std.gpu.memory import AddressSpace
from layout import Layout, LayoutTensor, TileTensor, row_major
from linalg.matmul import matmul as max_matmul

from ..constants import DT, CPU_SIMD_W
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
from ..core.target_storage import TargetStorage, assert_tag_for


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


def _grad_w_accum_kernel[
    BATCH: Int,
    IN: Int,
    OUT: Int,
](
    cache: LayoutTensor[DT, Layout.row_major(BATCH, IN), MutAnyOrigin],
    grad_output: LayoutTensor[DT, Layout.row_major(BATCH, OUT), MutAnyOrigin],
    grad_w: LayoutTensor[DT, Layout.row_major(IN, OUT), MutAnyOrigin],
):
    var idx = Int(global_idx.x)
    var total = IN * OUT
    if idx < total:
        var i = idx // OUT
        var j = idx % OUT
        var s: Scalar[DT] = 0.0
        for b in range(BATCH):
            s += rebind[Scalar[DT]](cache[b, i]) * rebind[Scalar[DT]](grad_output[b, j])
        grad_w[i, j] = rebind[Scalar[DT]](grad_w[i, j]) + s


def _grad_bias_kernel[
    BATCH: Int,
    OUT: Int,
](
    grad_output: LayoutTensor[DT, Layout.row_major(BATCH, OUT), MutAnyOrigin],
    grad_b: LayoutTensor[DT, Layout.row_major(OUT), MutAnyOrigin],
):
    var j = Int(global_idx.x)
    if j < OUT:
        var s: Scalar[DT] = 0.0
        for b in range(BATCH):
            s += rebind[Scalar[DT]](grad_output[b, j])
        grad_b[j] = rebind[Scalar[DT]](grad_b[j]) + s


# ──────────────────────────────────────────────────────────────────────
# Linear.
# ──────────────────────────────────────────────────────────────────────


struct Linear[IN: Int, OUT: Int](Module):
    comptime ARITY: Int = 1
    comptime IN_DIM = Self.IN
    comptime IN1_DIM: Int = 0
    comptime IN2_DIM: Int = 0
    comptime OUT_DIM = Self.OUT
    comptime W_SIZE = Self.IN * Self.OUT
    comptime B_SIZE = Self.OUT

    # Parameters — visible to reflection, walked by for_each_param_auto.
    var weight: Param["weight", True,  Self.W_SIZE]
    var bias:   Param["bias",   False, Self.B_SIZE]

    # Forward-time pointer alias of the orchestrator's input slab.
    # Backward reads from this directly; no copy at forward time.
    var _cached_input_ptr: UnsafePointer[Scalar[DT], MutAnyOrigin]

    # AMP scratch (lazy-allocated on first bf16 call).
    var amp: LinearAMPState[Self.IN, Self.OUT]

    var ts: TargetStorage

    # ----- Defaultable -----------------------------------------------------

    def __init__(out self):
        self.weight = Param["weight", True,  Self.W_SIZE]()
        self.bias   = Param["bias",   False, Self.B_SIZE]()
        self._cached_input_ptr = UnsafePointer[
            Scalar[DT], MutAnyOrigin,
        ](unsafe_from_address=0)
        self.amp = LinearAMPState[Self.IN, Self.OUT].make()
        self.ts = TargetStorage.make_uninit()

    # ----- Factories -------------------------------------------------------

    @staticmethod
    def make[target: StaticString, INIT: Initializer]() raises -> Self:
        comptime assert target == "cpu", (
            "Linear.make[target='gpu', INIT] requires a DeviceContext"
        )
        var lin = Self()
        lin.weight = Param["weight", True,  Self.W_SIZE].make_cpu()
        lin.bias   = Param["bias",   False, Self.B_SIZE].make_cpu()
        INIT.init_weight(
            lin.weight.value_unsafe_ptr_cpu(),
            Self.W_SIZE, Self.IN, Self.OUT,
        )
        INIT.init_bias(lin.bias.value_unsafe_ptr_cpu(), Self.B_SIZE)
        lin.ts = TargetStorage.make_cpu()
        return lin^

    @staticmethod
    def make[
        target: StaticString, INIT: Initializer
    ](ctx: DeviceContext) raises -> Self:
        comptime assert target == "gpu", (
            "Linear.make[target='cpu', INIT](ctx) — drop ctx for CPU"
        )
        var lin = Self()
        lin.weight = Param["weight", True,  Self.W_SIZE].make_gpu(ctx)
        lin.bias   = Param["bias",   False, Self.B_SIZE].make_gpu(ctx)
        # Init weights/biases on host via INIT, then upload.
        var w_host = ctx.enqueue_create_host_buffer[DT](Self.W_SIZE)
        var b_host = ctx.enqueue_create_host_buffer[DT](Self.B_SIZE)
        ctx.synchronize()
        INIT.init_weight(w_host.unsafe_ptr(), Self.W_SIZE, Self.IN, Self.OUT)
        INIT.init_bias(b_host.unsafe_ptr(), Self.B_SIZE)
        ctx.enqueue_copy(lin.weight.value_dev.value(), w_host)
        ctx.enqueue_copy(lin.bias.value_dev.value(),   b_host)
        ctx.synchronize()
        lin.ts = TargetStorage.make_gpu(ctx)
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
                    self.weight.value, row_major[Self.IN, Self.OUT](),
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
                    self.weight.value_dev.value(),
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
                        self.weight.value_dev.value().unsafe_ptr()
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
                self.bias.value_dev.value()
            )
            comptime TPB = 128
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
        comptime assert (
            mode == "all" or mode == "input_only"
        ), "mode must be 'all' or 'input_only'"
        assert_tag_for["Linear", target](self.ts.target_tag)
        var grad_output_v = typed_view[BATCH, Self.OUT](grad_output)
        var grad_input_v = typed_view_mut[BATCH, Self.IN](grad_inputs[0])

        var go_p = rebind[UnsafePointer[Scalar[DT], MutAnyOrigin]](grad_output_v.ptr)
        var gi_p = rebind[UnsafePointer[Scalar[DT], MutAnyOrigin]](grad_input_v.ptr)

        comptime if target == "cpu":
            # ── (1) grad_b += column-sum(grad_output) (mode=all only) ───
            comptime if mode == "all":
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

            # ── (2) grad_w += cache^T @ grad_output (mode=all only) ─────
            # Reads cache via _cached_input_ptr — MUST come before
            # grad_input write since grad_input may alias the cache slab.
            comptime if mode == "all":
                comptime cache_n = BATCH * Self.IN
                comptime gw_n = Self.IN * Self.OUT
                var cT_buf: UnsafePointer[Scalar[DT], MutAnyOrigin] = alloc[
                    Scalar[DT]
                ](cache_n)
                var dW_tmp_buf: UnsafePointer[Scalar[DT], MutAnyOrigin] = alloc[
                    Scalar[DT]
                ](gw_n)
                var cache_ptr = self._cached_input_ptr
                for bi in range(BATCH):
                    for i in range(Self.IN):
                        cT_buf[i * BATCH + bi] = cache_ptr[bi * Self.IN + i]
                var cT_tt = TileTensor(cT_buf, row_major[Self.IN, BATCH]())
                var dW_tmp_tt = TileTensor(
                    dW_tmp_buf, row_major[Self.IN, Self.OUT](),
                )
                max_matmul[target="cpu"](
                    dW_tmp_tt, cT_tt, grad_output_v, None,
                )
                var gw_ptr = self.weight.grad_unsafe_ptr_cpu()
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

            # ── (3) grad_input = grad_output @ W^T (always) ────────────
            # May alias the cache slab — safe now (1) and (2) are done.
            comptime if POLICY.compute_dtype == DT:
                var w_tt = TileTensor(
                    self.weight.value, row_major[Self.IN, Self.OUT](),
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
            comptime TPB = 128

            # ── (1) grad_b ─────────────────────────────────────────────
            comptime if mode == "all":
                comptime go_layout = Layout.row_major(BATCH, Self.OUT)
                comptime gb_layout = Layout.row_major(Self.OUT)
                var go_lt = LayoutTensor[DT, go_layout, MutAnyOrigin](go_p)
                var gb_lt = LayoutTensor[DT, gb_layout, MutAnyOrigin](
                    self.bias.grad_dev.value()
                )
                comptime n_blocks_gb = (Self.OUT + TPB - 1) // TPB
                comptime gb_kernel = _grad_bias_kernel[BATCH, Self.OUT]
                ctx.enqueue_function[gb_kernel](
                    go_lt, gb_lt,
                    grid_dim=n_blocks_gb, block_dim=TPB,
                )

            # ── (2) grad_w ─────────────────────────────────────────────
            comptime if mode == "all":
                comptime cache_layout = Layout.row_major(BATCH, Self.IN)
                comptime go_layout2 = Layout.row_major(BATCH, Self.OUT)
                comptime gw_layout = Layout.row_major(Self.IN, Self.OUT)
                var cache_lt = LayoutTensor[DT, cache_layout, MutAnyOrigin](
                    self._cached_input_ptr
                )
                var go_lt2 = LayoutTensor[DT, go_layout2, MutAnyOrigin](go_p)
                var gw_lt = LayoutTensor[DT, gw_layout, MutAnyOrigin](
                    self.weight.grad_dev.value()
                )
                comptime n_blocks_gw = (Self.W_SIZE + TPB - 1) // TPB
                comptime gw_kernel = _grad_w_accum_kernel[
                    BATCH, Self.IN, Self.OUT
                ]
                ctx.enqueue_function[gw_kernel](
                    cache_lt, go_lt2, gw_lt,
                    grid_dim=n_blocks_gw, block_dim=TPB,
                )

            # ── (3) grad_input = grad_output @ W^T ─────────────────────
            comptime if POLICY.compute_dtype == DT:
                var weight_tt = TileTensor(
                    self.weight.value_dev.value(),
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
                        self.weight.value_dev.value().unsafe_ptr()
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

