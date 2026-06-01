"""LinearAct[IN, OUT, OP: ElementOp] — fused matmul + bias + activation.

Single Module that replaces `Sequential[Linear[IN, OUT], Elementwise[OUT, OP]]`,
fusing the bias-add and activation into one GPU kernel on forward and the
activation derivative + grad_b reduction into one kernel on backward. Saves
2 kernel launches per training step vs the unfused chain.

Cache layout (one slab per direction):
  - `_cached_input_ptr` — aliases orchestrator's input slab (same as `Linear`).
    Used by the grad_W path. The orchestrator (Sequential / ComputeGraph)
    keeps the slab alive across the backward.
  - `act_cache` / `act_cache_dev` — owned `[BATCH, OUT]` buffer. Holds the
    pre-activation `z` when `OP.owns_cache=False` (ReLU, Mish, …) or the
    post-activation `y` when `OP.owns_cache=True` (Tanh, Sigmoid, …). The
    pre-activation is intermediate and has no upstream owner; unlike
    `Elementwise[OP]`, LinearAct must always own this cache.

Use via one-line aliases:
    alias LinearReLU[IN, OUT]    = LinearAct[IN, OUT, ReLUOp]
    alias LinearTanh[IN, OUT]    = LinearAct[IN, OUT, TanhOp]
    alias LinearSigmoid[IN, OUT] = LinearAct[IN, OUT, SigmoidOp]
    alias LinearMish[IN, OUT]    = LinearAct[IN, OUT, MishOp]
    alias LinearSwish[IN, OUT]   = LinearAct[IN, OUT, SwishOp]

BACKWARD-ORDER INVARIANT (inherited from Linear): the in-place rewrite of
`grad_output` with `OP.backward(c, go)` runs first, then grad_W reads
`_cached_input_ptr` (must run before grad_input clobbers that slab), then
grad_input via `max_matmul[transpose_b=True]`.

POLICY: fp32 only for v1 — bf16/AMP path is rejected at compile time
inside forward/vjp. Add AMP later if a benchmark calls for it.
"""

from std.memory import alloc
from std.gpu import global_idx
from std.gpu.host import DeviceContext, DeviceBuffer
from std.gpu.memory import AddressSpace
from layout import Layout, LayoutTensor, TileTensor, row_major
from linalg.matmul import matmul as max_matmul

from ..constants import DT, CPU_SIMD_W, TPB
from ..core import (
    Initializer,
    AMPPolicy,
    NoAMP,
    Param,
    for_each_param_auto,
    zero_grad_auto,
    ParamVisitor,
)
from ..core.element_op import ElementOp
from ..core.module import Module, typed_view, typed_view_mut
from ..core.target_storage import (
    TargetStorage,
    assert_tag_for,
    ensure_cpu_buffer,
    ensure_gpu_buffer,
)
from .linear import _grad_w_accum_kernel


# ──────────────────────────────────────────────────────────────────────
# GPU kernels — module-level so enqueue_function can bind them.
# ──────────────────────────────────────────────────────────────────────


def _linear_act_fwd_epilogue_kernel[
    BATCH: Int,
    OUT: Int,
    OP: ElementOp,
](
    output: LayoutTensor[DT, Layout.row_major(BATCH, OUT), MutAnyOrigin],
    bias: LayoutTensor[DT, Layout.row_major(OUT), MutAnyOrigin],
    cache: LayoutTensor[DT, Layout.row_major(BATCH, OUT), MutAnyOrigin],
):
    """Fused epilogue: output[b,o] = OP.forward(matmul[b,o] + bias[o]).

    Cache layout (comptime-branched on `OP.owns_cache`):
      - owns_cache = True  → cache[b,o] = y (post-activation)
      - owns_cache = False → cache[b,o] = z (pre-activation = matmul + bias)
    """
    var idx = Int(global_idx.x)
    var total = BATCH * OUT
    if idx < total:
        var b = idx // OUT
        var j = idx % OUT
        var z = rebind[Scalar[DT]](output[b, j]) + rebind[Scalar[DT]](bias[j])
        var y = OP.forward_scalar(z)
        output[b, j] = y
        comptime if OP.owns_cache:
            cache[b, j] = y
        else:
            cache[b, j] = z


def _linear_act_bwd_fused_kernel[
    BATCH: Int,
    OUT: Int,
    OP: ElementOp,
    ACCUM_GRAD_B: Bool,
](
    grad_output: LayoutTensor[DT, Layout.row_major(BATCH, OUT), MutAnyOrigin],
    cache: LayoutTensor[DT, Layout.row_major(BATCH, OUT), MutAnyOrigin],
    grad_b: LayoutTensor[DT, Layout.row_major(OUT), MutAnyOrigin],
):
    """Per-column j: rewrite grad_output[b,j] in place with the activation
    derivative AND reduce-sum across b into grad_b[j] in a single pass.

    Layout: one thread per output column (grid = ceil(OUT/TPB)). The
    inner loop walks BATCH serially per thread. For typical (BATCH<=256,
    OUT<=512) this is faster than launching a separate grad_b reduction.

    When `ACCUM_GRAD_B=False` (mode='input_only'), the grad_b write is
    elided at comptime — the in-place rewrite of grad_output still happens
    so the downstream `max_matmul[transpose_b=True]` produces correct
    grad_input.
    """
    var j = Int(global_idx.x)
    if j < OUT:
        var s: Scalar[DT] = 0.0
        for b in range(BATCH):
            var c = rebind[Scalar[DT]](cache[b, j])
            var go = rebind[Scalar[DT]](grad_output[b, j])
            var gpre = OP.backward_scalar(c, go)
            grad_output[b, j] = gpre
            comptime if ACCUM_GRAD_B:
                s = s + gpre
        comptime if ACCUM_GRAD_B:
            grad_b[j] = rebind[Scalar[DT]](grad_b[j]) + s


# ──────────────────────────────────────────────────────────────────────
# LinearAct.
# ──────────────────────────────────────────────────────────────────────


struct LinearAct[IN: Int, OUT: Int, OP: ElementOp](Module):
    comptime ARITY: Int = 1
    comptime IN_DIMS = InlineArray[Int, 1](fill=Self.IN)
    comptime OUT_DIM = Self.OUT
    comptime W_SIZE = Self.IN * Self.OUT
    comptime B_SIZE = Self.OUT

    @staticmethod
    def display_label() -> String:
        return String("LinearAct(") + Self.OP.display_label() + String(")")

    # Parameters — same layout as Linear so reflection walkers find them.
    var weight: Param["weight", True,  Self.W_SIZE]
    var bias:   Param["bias",   False, Self.B_SIZE]

    # Aliased input slab — used by grad_W on backward (same as Linear).
    var _cached_input_ptr: Optional[UnsafePointer[Scalar[DT], MutAnyOrigin]]

    # Owned activation cache — stores z (if !owns_cache) or y (if owns_cache).
    # Lazy-grown to [BATCH * OUT] on first forward.
    var act_cache_cpu: List[Scalar[DT]]
    var act_cache_dev: Optional[DeviceBuffer[DT]]
    var act_cache_n: Int

    var ts: TargetStorage

    # ----- Defaultable -----------------------------------------------------

    def __init__(out self):
        self.weight = Param["weight", True,  Self.W_SIZE]()
        self.bias   = Param["bias",   False, Self.B_SIZE]()
        self._cached_input_ptr = None
        self.act_cache_cpu = List[Scalar[DT]]()
        self.act_cache_dev = None
        self.act_cache_n = 0
        self.ts = TargetStorage.make_uninit()

    # ----- Factories -------------------------------------------------------

    @staticmethod
    def make[
        target: StaticString, INIT: Initializer
    ](
        ctx: Optional[DeviceContext] = None,
    ) raises -> Self:
        """Unified CPU/GPU factory. Mirrors `Linear.make` (same param
        init: weights via `INIT.init_weight`, bias via `INIT.init_bias`)."""
        comptime assert target == "cpu" or target == "gpu", (
            "LinearAct: target must be 'cpu' or 'gpu'"
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
            if not ctx:
                raise Error("LinearAct.make[target='gpu']: ctx required")
            var ctx_v = ctx.value()
            lin.weight = Param["weight", True,  Self.W_SIZE].make_gpu(ctx_v)
            lin.bias   = Param["bias",   False, Self.B_SIZE].make_gpu(ctx_v)
            var w_host = ctx_v.enqueue_create_host_buffer[DT](Self.W_SIZE)
            var b_host = ctx_v.enqueue_create_host_buffer[DT](Self.B_SIZE)
            ctx_v.synchronize()
            INIT.init_weight(w_host.unsafe_ptr(), Self.W_SIZE, Self.IN, Self.OUT)
            INIT.init_bias(b_host.unsafe_ptr(), Self.B_SIZE)
            ctx_v.enqueue_copy(lin.weight.value_dev.value(), w_host)
            ctx_v.enqueue_copy(lin.bias.value_dev.value(),   b_host)
            ctx_v.synchronize()
            # Placeholder dev buffer for act_cache; grown lazily on first fwd.
            lin.act_cache_dev = ctx_v.enqueue_create_buffer[DT](1)
            lin.act_cache_n = 0
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
        comptime assert POLICY.compute_dtype == DT, (
            "LinearAct: bf16/AMP not yet supported; use Sequential[Linear, "
            "Elementwise[OP]] for the bf16 path"
        )
        assert_tag_for["LinearAct", target](self.ts.target_tag)
        var input_v = typed_view[BATCH, Self.IN](inputs[0])
        var output_v = typed_view_mut[BATCH, Self.OUT](output)

        var in_p = rebind[UnsafePointer[Scalar[DT], MutAnyOrigin]](input_v.ptr)
        var out_p = rebind[UnsafePointer[Scalar[DT], MutAnyOrigin]](output_v.ptr)

        # Save the orchestrator input slab pointer for grad_W on backward.
        self._cached_input_ptr = in_p

        comptime if target == "cpu":
            # (1) max_matmul: output = input @ W
            var w_tt = TileTensor(
                self.weight.value, row_major[Self.IN, Self.OUT](),
            )
            max_matmul[target="cpu"](output_v, input_v, w_tt, None)

            # (2) Fused bias-add + activation + cache write. SIMD inner loop.
            comptime N = BATCH * Self.OUT
            ensure_cpu_buffer(self.act_cache_cpu, N)
            var b_ptr = self.bias.value_unsafe_ptr_cpu()
            var cache_p = self.act_cache_cpu.unsafe_ptr()
            for bi in range(BATCH):
                var row_off = bi * Self.OUT
                var ij = 0
                while ij + CPU_SIMD_W <= Self.OUT:
                    var ov = out_p.load[width=CPU_SIMD_W](row_off + ij)
                    var bv = b_ptr.load[width=CPU_SIMD_W](ij)
                    var z = ov + bv
                    var y = Self.OP.forward_simd[CPU_SIMD_W](z)
                    out_p.store(row_off + ij, y)
                    comptime if Self.OP.owns_cache:
                        cache_p.store(row_off + ij, y)
                    else:
                        cache_p.store(row_off + ij, z)
                    ij += CPU_SIMD_W
                while ij < Self.OUT:
                    var z_s = out_p[row_off + ij] + b_ptr[ij]
                    var y_s = Self.OP.forward_scalar(z_s)
                    out_p[row_off + ij] = y_s
                    comptime if Self.OP.owns_cache:
                        cache_p[row_off + ij] = y_s
                    else:
                        cache_p[row_off + ij] = z_s
                    ij += 1
        else:
            var ctx = self.ts.ctx.value()

            # (1) max_matmul: output = input @ W
            var weight_tt = TileTensor(
                self.weight.value_dev.value(),
                row_major[Self.IN, Self.OUT](),
            )
            max_matmul[target="gpu"](output_v, input_v, weight_tt, ctx)

            # (2) Fused bias-add + activation + cache write (one launch).
            ensure_gpu_buffer(
                self.act_cache_dev, self.act_cache_n,
                BATCH * Self.OUT, ctx,
            )
            comptime out_layout = Layout.row_major(BATCH, Self.OUT)
            comptime bias_layout = Layout.row_major(Self.OUT)
            var output_lt = LayoutTensor[DT, out_layout, MutAnyOrigin](out_p)
            var bias_lt = LayoutTensor[DT, bias_layout, MutAnyOrigin](
                self.bias.value_dev.value()
            )
            var cache_lt = LayoutTensor[DT, out_layout, MutAnyOrigin](
                self.act_cache_dev.value()
            )
            comptime n_blocks = (BATCH * Self.OUT + TPB - 1) // TPB
            comptime fwd_kernel = _linear_act_fwd_epilogue_kernel[
                BATCH, Self.OUT, Self.OP,
            ]
            ctx.enqueue_function[fwd_kernel](
                output_lt, bias_lt, cache_lt,
                grid_dim=n_blocks, block_dim=TPB,
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
        comptime assert POLICY.compute_dtype == DT, (
            "LinearAct.vjp: bf16/AMP not yet supported"
        )
        assert_tag_for["LinearAct", target](self.ts.target_tag)
        var grad_output_v = typed_view[BATCH, Self.OUT](grad_output)
        var grad_input_v = typed_view_mut[BATCH, Self.IN](grad_inputs[0])

        var go_p = rebind[UnsafePointer[Scalar[DT], MutAnyOrigin]](grad_output_v.ptr)
        var gi_p = rebind[UnsafePointer[Scalar[DT], MutAnyOrigin]](grad_input_v.ptr)

        comptime if target == "cpu":
            # (1) In-place rewrite: grad_output[b,j] = OP.backward(cache[b,j], go).
            #     SIMD loop over BATCH*OUT.
            comptime N = BATCH * Self.OUT
            var cache_p = self.act_cache_cpu.unsafe_ptr()
            var k = 0
            while k + CPU_SIMD_W <= N:
                var c = cache_p.load[width=CPU_SIMD_W](k)
                var g = go_p.load[width=CPU_SIMD_W](k)
                go_p.store(k, Self.OP.backward_simd[CPU_SIMD_W](c, g))
                k += CPU_SIMD_W
            while k < N:
                go_p[k] = Self.OP.backward_scalar(cache_p[k], go_p[k])
                k += 1

            # (2) grad_b = column-sum(rewritten grad_output) (mode=all only).
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

            # (3) grad_w += cache_input^T @ grad_pre_act  (mode=all only).
            #     Must run BEFORE grad_input write (aliases input slab).
            comptime if mode == "all":
                comptime gw_n = Self.IN * Self.OUT
                var gw_ptr = self.weight.grad_unsafe_ptr_cpu()
                var cache_in_p = self._cached_input_ptr.value()
                var cT_buf: UnsafePointer[
                    Scalar[DT], MutAnyOrigin
                ] = alloc[Scalar[DT]](BATCH * Self.IN)
                var dW_tmp_buf: UnsafePointer[
                    Scalar[DT], MutAnyOrigin
                ] = alloc[Scalar[DT]](gw_n)
                for bi in range(BATCH):
                    for i in range(Self.IN):
                        cT_buf[i * BATCH + bi] = cache_in_p[
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

            # (4) grad_input = grad_pre_act @ W^T (always).
            var w_tt = TileTensor(
                self.weight.value, row_major[Self.IN, Self.OUT](),
            )
            max_matmul[transpose_b=True, target="cpu"](
                grad_input_v, grad_output_v, w_tt, None,
            )
        else:
            var ctx = self.ts.ctx.value()

            # (1+2) Fused: rewrite grad_output in-place with activation
            #       derivative AND reduce-sum into grad_b (when mode=all).
            comptime go_layout = Layout.row_major(BATCH, Self.OUT)
            comptime gb_layout = Layout.row_major(Self.OUT)
            var go_lt = LayoutTensor[DT, go_layout, MutAnyOrigin](go_p)
            var cache_lt = LayoutTensor[DT, go_layout, MutAnyOrigin](
                self.act_cache_dev.value()
            )
            var gb_lt = LayoutTensor[DT, gb_layout, MutAnyOrigin](
                self.bias.grad_dev.value()
            )
            comptime n_blocks_bwd = (Self.OUT + TPB - 1) // TPB
            comptime accum = mode == "all"
            comptime bwd_kernel = _linear_act_bwd_fused_kernel[
                BATCH, Self.OUT, Self.OP, accum,
            ]
            ctx.enqueue_function[bwd_kernel](
                go_lt, cache_lt, gb_lt,
                grid_dim=n_blocks_bwd, block_dim=TPB,
            )

            # (3) grad_w += cache_input^T @ grad_pre_act (mode=all only).
            #     Reads rewritten grad_output (now grad_pre_act).
            comptime if mode == "all":
                comptime cache_layout = Layout.row_major(BATCH, Self.IN)
                comptime go_layout2 = Layout.row_major(BATCH, Self.OUT)
                comptime gw_layout = Layout.row_major(Self.IN, Self.OUT)
                var cache_in_lt = LayoutTensor[DT, cache_layout, MutAnyOrigin](
                    self._cached_input_ptr.value()
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
                    cache_in_lt, go_lt2, gw_lt,
                    grid_dim=n_blocks_gw, block_dim=TPB,
                )

            # (4) grad_input = grad_pre_act @ W^T (always).
            var weight_tt = TileTensor(
                self.weight.value_dev.value(),
                row_major[Self.IN, Self.OUT](),
            )
            max_matmul[transpose_b=True, target="gpu"](
                grad_input_v, grad_output_v, weight_tt, ctx,
            )

    # ----- Param / grad walkers (reflection-derived) ----------------------

    def for_each_param[
        target: StaticString,
        V: ParamVisitor,
    ](mut self, prefix: String, mut visitor: V) raises:
        assert_tag_for["LinearAct", target](self.ts.target_tag)
        for_each_param_auto[Self, V, target](self, prefix, visitor)

    def zero_grad[target: StaticString](mut self) raises:
        assert_tag_for["LinearAct", target](self.ts.target_tag)
        zero_grad_auto[Self, target](self)
