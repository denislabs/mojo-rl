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

POLICY: fp32 compute only. A bf16/AMP `POLICY` is accepted but treated as a
no-op (forward/vjp always compute in `DT`) so fused layers can compose into
AMP-compiled trainers (e.g. the SAC trainer instantiates a bf16 branch
unconditionally). Outputs are fp32 regardless — same as `Linear` — so mixing
with bf16-compute `Linear`s in one Sequential is safe. For a real bf16 compute
path use Sequential[Linear, Elementwise[OP]]. Add true AMP later if a
benchmark calls for it.
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
from .linear import _grad_bias_reduce_kernel, _transpose_kernel, _accum_kernel


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


def _act_deriv_rewrite_kernel[BATCH: Int, OUT: Int, OP: ElementOp](
    grad_output: LayoutTensor[DT, Layout.row_major(BATCH, OUT), MutAnyOrigin],
    cache: LayoutTensor[DT, Layout.row_major(BATCH, OUT), MutAnyOrigin],
):
    """In-place activation-derivative rewrite: grad_output[b,o] ←
    OP.backward(cache[b,o], grad_output[b,o]). One thread per ELEMENT
    (grid = ceil(BATCH·OUT/TPB)) — full GPU occupancy, replacing the old
    one-thread-per-column serial-BATCH-loop kernel (which launched only OUT
    threads and dominated the SAC backward on large GPUs). grad_b and grad_w
    are now separate, properly-parallel passes (a sum reduction + a
    `max_matmul`), so this kernel does only the per-element rewrite the
    downstream matmuls consume."""
    var idx = Int(global_idx.x)
    if idx < BATCH * OUT:
        var b = idx // OUT
        var o = idx % OUT
        var c = rebind[Scalar[DT]](cache[b, o])
        var go = rebind[Scalar[DT]](grad_output[b, o])
        grad_output[b, o] = OP.backward_scalar(c, go)


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

    # Backward grad_w temporaries (GPU). `cacheT_dev` holds the transposed
    # input cacheᵀ[IN, BATCH] (lazy, BATCH-sized — mirrors `act_cache_dev` so
    # it allocates on the pre-capture settle call and is reused on every
    # CUDA-graph replay). `dW_tmp_dev` holds the [IN, OUT] result of
    # `cacheᵀ @ grad_output` (W_SIZE, fixed) before it is accumulated into
    # grad_w. Lets grad_w run through `max_matmul` (tensor cores) instead of
    # the serial `_grad_w_accum_kernel`.
    var cacheT_dev: Optional[DeviceBuffer[DT]]
    var cacheT_n: Int
    var dW_tmp_dev: Optional[DeviceBuffer[DT]]

    var ts: TargetStorage

    # ----- Defaultable -----------------------------------------------------

    def __init__(out self):
        self.weight = Param["weight", True,  Self.W_SIZE]()
        self.bias   = Param["bias",   False, Self.B_SIZE]()
        self._cached_input_ptr = None
        self.act_cache_cpu = List[Scalar[DT]]()
        self.act_cache_dev = None
        self.act_cache_n = 0
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
            # Fixed-size [IN, OUT] dW scratch for the max_matmul grad_w path.
            # cacheT stays None (lazily sized to BATCH on first backward).
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
        # AMP no-op: this fused layer always computes in `DT` (fp32). A bf16
        # `POLICY` is accepted but IGNORED — the matmul + activation epilogue
        # below run in fp32 — so fused layers can compose into AMP-compiled
        # trainers (the SAC trainer instantiates a bf16 branch unconditionally,
        # even when bf16 is disabled at runtime). Outputs are fp32 regardless,
        # matching `Linear` (whose bias-add is fp32 under any policy), so mixing
        # this layer with bf16-compute `Linear`s in one Sequential is safe. For
        # an actual bf16 compute path use Sequential[Linear, Elementwise[OP]].
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
        # AMP no-op (see `forward`): a bf16 `POLICY` is accepted but ignored;
        # the backward matmuls run in fp32. Lets fused layers live inside an
        # AMP-compiled trainer; fp32 at runtime when bf16 is off.
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

            comptime go_layout = Layout.row_major(BATCH, Self.OUT)
            var go_lt = LayoutTensor[DT, go_layout, MutAnyOrigin](go_p)
            var cache_lt = LayoutTensor[DT, go_layout, MutAnyOrigin](
                self.act_cache_dev.value()
            )

            # (1) In-place activation-derivative rewrite: grad_output ←
            #     OP.backward(cache, grad_output). Per-ELEMENT (BATCH·OUT
            #     threads) → full occupancy. Replaces the old serial
            #     one-thread-per-column fused kernel.
            comptime n_elems = BATCH * Self.OUT
            comptime n_blocks_rw = (n_elems + TPB - 1) // TPB
            comptime rw_kernel = _act_deriv_rewrite_kernel[
                BATCH, Self.OUT, Self.OP
            ]
            ctx.enqueue_function[rw_kernel](
                go_lt, cache_lt,
                grid_dim=n_blocks_rw, block_dim=TPB,
            )

            # (2) grad_b += column-sum(grad_pre_act)  (mode=all only). Sum-only
            #     reduction (no activation/writes) — far lighter than folding it
            #     into the serial rewrite. Reuses Linear's `_grad_bias_kernel`.
            comptime if mode == "all":
                comptime gb_layout = Layout.row_major(Self.OUT)
                var gb_lt = LayoutTensor[DT, gb_layout, MutAnyOrigin](
                    self.bias.grad_dev.value()
                )
                comptime gb_kernel = _grad_bias_reduce_kernel[BATCH, Self.OUT]
                ctx.enqueue_function[gb_kernel](
                    go_lt, gb_lt,
                    grid_dim=Self.OUT, block_dim=TPB,
                )

            # (3) grad_w += cache_inputᵀ @ grad_pre_act  (mode=all only), via
            #     transpose + max_matmul (tensor cores) + accumulate — replaces
            #     the naive serial `_grad_w_accum_kernel`. Reads the rewritten
            #     grad_output (now grad_pre_act).
            comptime if mode == "all":
                # 3a. cacheTᵀ[IN, BATCH] = transpose(cache_input[BATCH, IN]).
                #     Lazy-size cacheT to BATCH (capture-safe: first/settle call
                #     allocates, replays reuse).
                ensure_gpu_buffer(
                    self.cacheT_dev, self.cacheT_n, BATCH * Self.IN, ctx,
                )
                comptime cin_layout = Layout.row_major(BATCH, Self.IN)
                comptime cinT_layout = Layout.row_major(Self.IN, BATCH)
                var cin_lt = LayoutTensor[DT, cin_layout, MutAnyOrigin](
                    self._cached_input_ptr.value()
                )
                var cinT_lt = LayoutTensor[DT, cinT_layout, MutAnyOrigin](
                    self.cacheT_dev.value()
                )
                comptime n_blocks_t = (BATCH * Self.IN + TPB - 1) // TPB
                comptime t_kernel = _transpose_kernel[BATCH, Self.IN]
                ctx.enqueue_function[t_kernel](
                    cin_lt, cinT_lt,
                    grid_dim=n_blocks_t, block_dim=TPB,
                )
                # 3b. dW_tmp[IN, OUT] = cacheTᵀ @ grad_pre_act  (max_matmul).
                var cinT_tt = TileTensor(
                    self.cacheT_dev.value(), row_major[Self.IN, BATCH](),
                )
                var dW_tmp_tt = TileTensor(
                    self.dW_tmp_dev.value(), row_major[Self.IN, Self.OUT](),
                )
                max_matmul[target="gpu"](
                    dW_tmp_tt, cinT_tt, grad_output_v, ctx,
                )
                # 3c. grad_w += dW_tmp.
                comptime gw_layout = Layout.row_major(Self.W_SIZE)
                var gw_lt = LayoutTensor[DT, gw_layout, MutAnyOrigin](
                    self.weight.grad_dev.value()
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
