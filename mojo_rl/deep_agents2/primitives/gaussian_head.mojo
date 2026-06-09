"""GaussianHead[IN, ACT] — CleanRL-style PPO actor head.

CleanRL-style PPO actor head: state-dependent mean (Linear) + a single
learnable state-independent log_std vector. See v1 docstring for the
algorithmic notes; this file only changes the storage / scaffolding:

  * `ts: TargetStorage` replaces the per-leaf tag/inference/ctx triplet.
  * `weight: Param["weight", True,  IN*ACT]` +
    `bias:   Param["bias",   False, ACT]` +
    `log_std: Param["log_std", False, ACT]` replace the six lists +
    six device buffers.
  * `_cached_input_ptr` pointer alias replaces the COPIED input cache
    (same trick as `Linear`).
  * `backward[mode]` collapses v1's `backward` + `backward_input`.
  * `for_each_param` / `zero_grad` are one-liners delegating to
    `for_each_param_auto` / `zero_grad_auto`.
  * Phase 10A buffer surface dropped.

**BACKWARD-ORDER INVARIANT**: same as Linear. Because `_cached_input_ptr`
aliases the orchestrator's input slab, and grad_input writes into that
same slab, `grad_w` (which reads the cache) MUST come before
`grad_input`. v1's order (grad_input → grad_w → grad_b → grad_log_std)
is reversed to (grad_b → grad_log_std → grad_w → grad_input).
"""

from std.gpu import global_idx
from std.gpu.host import DeviceContext, DeviceBuffer
from std.gpu.memory import AddressSpace
from layout import Layout, LayoutTensor, TileTensor, row_major

from mojo_rl.nn2.constants import DT, TPB
from mojo_rl.nn2.core import (
    Initializer,
    AMPPolicy,
    NoAMP,
    Param,
    ParamVisitor,
    for_each_param_auto,
    zero_grad_auto,
)
from mojo_rl.nn2.core.module import Module, typed_view, typed_view_mut, mptr
from mojo_rl.nn2.core.tensor_pack import TensorPack
from mojo_rl.nn2.core.target_storage import require_ctx, TargetStorage, assert_tag_for
from mojo_rl.nn2.core.target_tag import TARGET_GPU


comptime LOG_STD_MIN: Scalar[DT] = -5.0
comptime LOG_STD_MAX: Scalar[DT] = 2.0


# ──────────────────────────────────────────────────────────────────────
# GPU kernels — copied from v1; `_cache_input_kernel` removed (no copy).
# ──────────────────────────────────────────────────────────────────────


def _gauss_head_forward_kernel[BATCH: Int, IN: Int, ACT: Int](
    input: LayoutTensor[DT, Layout.row_major(BATCH, IN), MutAnyOrigin],
    weight: LayoutTensor[DT, Layout.row_major(IN, ACT), MutAnyOrigin],
    bias: LayoutTensor[DT, Layout.row_major(ACT), MutAnyOrigin],
    log_std: LayoutTensor[DT, Layout.row_major(ACT), MutAnyOrigin],
    output: LayoutTensor[DT, Layout.row_major(BATCH, 2 * ACT), MutAnyOrigin],
):
    var idx = Int(global_idx.x)
    var total = BATCH * 2 * ACT
    if idx < total:
        var b = idx // (2 * ACT)
        var j = idx % (2 * ACT)
        if j < ACT:
            var acc = rebind[Scalar[DT]](bias[j])
            for i in range(IN):
                acc += rebind[Scalar[DT]](input[b, i]) * rebind[Scalar[DT]](weight[i, j])
            output[b, j] = acc
        else:
            var k = j - ACT
            var v = rebind[Scalar[DT]](log_std[k])
            if v < LOG_STD_MIN:
                v = LOG_STD_MIN
            elif v > LOG_STD_MAX:
                v = LOG_STD_MAX
            output[b, j] = v


def _gauss_head_grad_input_kernel[BATCH: Int, IN: Int, ACT: Int](
    grad_output: LayoutTensor[DT, Layout.row_major(BATCH, 2 * ACT), MutAnyOrigin],
    weight: LayoutTensor[DT, Layout.row_major(IN, ACT), MutAnyOrigin],
    grad_input: LayoutTensor[DT, Layout.row_major(BATCH, IN), MutAnyOrigin],
):
    var idx = Int(global_idx.x)
    var total = BATCH * IN
    if idx < total:
        var b = idx // IN
        var i = idx % IN
        var acc: Scalar[DT] = 0.0
        for j in range(ACT):
            acc += rebind[Scalar[DT]](grad_output[b, j]) * rebind[Scalar[DT]](weight[i, j])
        grad_input[b, i] = acc


def _gauss_head_grad_w_kernel[BATCH: Int, IN: Int, ACT: Int](
    cache: LayoutTensor[DT, Layout.row_major(BATCH, IN), MutAnyOrigin],
    grad_output: LayoutTensor[DT, Layout.row_major(BATCH, 2 * ACT), MutAnyOrigin],
    grad_w: LayoutTensor[DT, Layout.row_major(IN, ACT), MutAnyOrigin],
):
    var idx = Int(global_idx.x)
    var total = IN * ACT
    if idx < total:
        var i = idx // ACT
        var j = idx % ACT
        var s: Scalar[DT] = 0.0
        for b in range(BATCH):
            s += rebind[Scalar[DT]](cache[b, i]) * rebind[Scalar[DT]](grad_output[b, j])
        grad_w[i, j] = rebind[Scalar[DT]](grad_w[i, j]) + s


def _gauss_head_grad_b_kernel[BATCH: Int, ACT: Int](
    grad_output: LayoutTensor[DT, Layout.row_major(BATCH, 2 * ACT), MutAnyOrigin],
    grad_b: LayoutTensor[DT, Layout.row_major(ACT), MutAnyOrigin],
):
    var j = Int(global_idx.x)
    if j < ACT:
        var s: Scalar[DT] = 0.0
        for b in range(BATCH):
            s += rebind[Scalar[DT]](grad_output[b, j])
        grad_b[j] = rebind[Scalar[DT]](grad_b[j]) + s


def _gauss_head_grad_ls_kernel[BATCH: Int, ACT: Int](
    grad_output: LayoutTensor[DT, Layout.row_major(BATCH, 2 * ACT), MutAnyOrigin],
    grad_ls: LayoutTensor[DT, Layout.row_major(ACT), MutAnyOrigin],
):
    var j = Int(global_idx.x)
    if j < ACT:
        var s: Scalar[DT] = 0.0
        for b in range(BATCH):
            s += rebind[Scalar[DT]](grad_output[b, ACT + j])
        grad_ls[j] = rebind[Scalar[DT]](grad_ls[j]) + s


# ──────────────────────────────────────────────────────────────────────
# GaussianHead.
# ──────────────────────────────────────────────────────────────────────


struct GaussianHead[IN: Int, ACT: Int](Module):
    comptime ARITY: Int = 1
    comptime IN_DIMS = InlineArray[Int, 1](fill=Self.IN)
    comptime OUT_DIM = 2 * Self.ACT
    comptime W_SIZE = Self.IN * Self.ACT
    comptime B_SIZE = Self.ACT
    comptime LS_SIZE = Self.ACT
    comptime DEFAULT_LOG_STD_INIT: Scalar[DT] = 0.0

    var weight:  Param["weight",  True,  Self.W_SIZE]
    var bias:    Param["bias",    False, Self.B_SIZE]
    var log_std: Param["log_std", False, Self.LS_SIZE]

    var _cached_input_ptr: Optional[UnsafePointer[Scalar[DT], MutAnyOrigin]]

    var ts: TargetStorage

    def __init__(out self):
        self.weight  = Param["weight",  True,  Self.W_SIZE]()
        self.bias    = Param["bias",    False, Self.B_SIZE]()
        self.log_std = Param["log_std", False, Self.LS_SIZE]()
        self._cached_input_ptr = None
        self.ts = TargetStorage.make_uninit()

    @staticmethod
    def make[
        target: StaticString, INIT: Initializer
    ](
        ctx: Optional[DeviceContext] = None,
    ) raises -> Self:
        """Unified CPU/GPU factory. `ctx=None` on CPU; required on GPU."""
        comptime assert target == "cpu" or target == "gpu", (
            "GaussianHead: target must be 'cpu' or 'gpu'"
        )
        var h = Self()
        comptime if target == "cpu":
            h.weight  = Param["weight",  True,  Self.W_SIZE].make_cpu()
            h.bias    = Param["bias",    False, Self.B_SIZE].make_cpu()
            h.log_std = Param["log_std", False, Self.LS_SIZE].make_cpu()
            INIT.init_weight(
                h.weight.value_unsafe_ptr_cpu(), Self.W_SIZE, Self.IN, Self.ACT,
            )
            INIT.init_bias(h.bias.value_unsafe_ptr_cpu(), Self.B_SIZE)
            var ls_ptr = h.log_std.value_unsafe_ptr_cpu()
            for k in range(Self.LS_SIZE):
                ls_ptr[k] = Self.DEFAULT_LOG_STD_INIT
            h.ts = TargetStorage.make_cpu()
        else:
            var ctx_v = require_ctx["GaussianHead.make[target='gpu']"](ctx)
            h.weight  = Param["weight",  True,  Self.W_SIZE].make_gpu(ctx_v)
            h.bias    = Param["bias",    False, Self.B_SIZE].make_gpu(ctx_v)
            h.log_std = Param["log_std", False, Self.LS_SIZE].make_gpu(ctx_v)
            h.log_std.val.dev.value().enqueue_fill(Self.DEFAULT_LOG_STD_INIT)
            # Init weights/bias on host then upload.
            var w_host = ctx_v.enqueue_create_host_buffer[DT](Self.W_SIZE)
            var b_host = ctx_v.enqueue_create_host_buffer[DT](Self.B_SIZE)
            ctx_v.synchronize()
            INIT.init_weight(w_host.unsafe_ptr(), Self.W_SIZE, Self.IN, Self.ACT)
            INIT.init_bias(b_host.unsafe_ptr(), Self.B_SIZE)
            ctx_v.enqueue_copy(h.weight.val.dev.value(), w_host)
            ctx_v.enqueue_copy(h.bias.val.dev.value(),   b_host)
            ctx_v.synchronize()
            h.ts = TargetStorage.make_gpu(ctx_v)
        return h^

    def set_log_std_init(mut self, value: Scalar[DT]) raises:
        """Override the default log_std initialization. Call after make."""
        if self.ts.target_tag == TARGET_GPU:
            self.log_std.val.dev.value().enqueue_fill(value)
        else:
            var ls_ptr = self.log_std.value_unsafe_ptr_cpu()
            for k in range(Self.LS_SIZE):
                ls_ptr[k] = value

    # ----- Forward ---------------------------------------------------------

    def forward[
        target: StaticString,
        BATCH: Int,
        POLICY: AMPPolicy = NoAMP,
    ](
        mut self,
        inputs: TensorPack[Self.ARITY],
        mut output: TileTensor[
            mut=True, dtype=DT, address_space=AddressSpace.GENERIC,
            element_size=1, origin=MutAnyOrigin, ...,
        ],
    ) raises:
        assert_tag_for["GaussianHead", target](self.ts.target_tag)
        var input = inputs.tile[0, BATCH, Self.IN_DIMS[0]]()
        var output_v = typed_view_mut[BATCH, Self.OUT_DIM](output)

        var in_p = mptr(input.ptr)
        self._cached_input_ptr = in_p

        comptime if target == "cpu":
            var w = TileTensor(self.weight.val.cpu, row_major[Self.IN, Self.ACT]())
            var b = TileTensor(self.bias.val.cpu,   row_major[Self.ACT]())
            var ls = TileTensor(self.log_std.val.cpu, row_major[Self.ACT]())
            for bi in range(BATCH):
                # mu = input @ W + b
                for j in range(Self.ACT):
                    var acc = b[j]
                    for i in range(Self.IN):
                        acc += input[bi, i] * w[i, j]
                    output_v[bi, j] = acc
                # log_std broadcast + clamp
                for j in range(Self.ACT):
                    var v = ls[j]
                    if v < LOG_STD_MIN:
                        v = LOG_STD_MIN
                    elif v > LOG_STD_MAX:
                        v = LOG_STD_MAX
                    output_v[bi, Self.ACT + j] = v
        else:
            var ctx = self.ts.ctx.value()
            var out_p_w = mptr(output_v.ptr)

            comptime in_layout = Layout.row_major(BATCH, Self.IN)
            comptime out_layout = Layout.row_major(BATCH, 2 * Self.ACT)
            comptime w_layout = Layout.row_major(Self.IN, Self.ACT)
            comptime b_layout = Layout.row_major(Self.ACT)
            var input_lt = LayoutTensor[DT, in_layout, MutAnyOrigin](in_p)
            var w_lt = LayoutTensor[DT, w_layout, MutAnyOrigin](
                self.weight.val.dev.value()
            )
            var b_lt = LayoutTensor[DT, b_layout, MutAnyOrigin](
                self.bias.val.dev.value()
            )
            var ls_lt = LayoutTensor[DT, b_layout, MutAnyOrigin](
                self.log_std.val.dev.value()
            )
            var output_lt = LayoutTensor[DT, out_layout, MutAnyOrigin](out_p_w)

            comptime n_blocks_fwd = (BATCH * 2 * Self.ACT + TPB - 1) // TPB
            comptime fwd_kernel = _gauss_head_forward_kernel[
                BATCH, Self.IN, Self.ACT
            ]
            ctx.enqueue_function[fwd_kernel](
                input_lt, w_lt, b_lt, ls_lt, output_lt,
                grid_dim=n_blocks_fwd, block_dim=TPB,
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
        grad_inputs: TensorPack[Self.ARITY],
    ) raises:
        comptime assert (
            mode == "all" or mode == "input_only"
        ), "mode must be 'all' or 'input_only'"
        assert_tag_for["GaussianHead", target](self.ts.target_tag)
        var grad_output_v = typed_view[BATCH, Self.OUT_DIM](grad_output)
        var grad_input_v = grad_inputs.tile[0, BATCH, Self.IN_DIMS[0]]()

        comptime if target == "cpu":
            var w = TileTensor(self.weight.val.cpu, row_major[Self.IN, Self.ACT]())

            # ── (1) grad_b, grad_log_std (mode=all) ────────────────────
            comptime if mode == "all":
                var gb = TileTensor(self.bias.grd.cpu,    row_major[Self.ACT]())
                var gls = TileTensor(self.log_std.grd.cpu, row_major[Self.ACT]())
                for j in range(Self.ACT):
                    var acc_b: Scalar[DT] = 0.0
                    var acc_l: Scalar[DT] = 0.0
                    for bi in range(BATCH):
                        acc_b += grad_output_v[bi, j]
                        acc_l += grad_output_v[bi, Self.ACT + j]
                    gb[j]  = gb[j]  + acc_b
                    gls[j] = gls[j] + acc_l

            # ── (2) grad_w (mode=all). Reads cache via _cached_input_ptr ─
            comptime if mode == "all":
                var gw = TileTensor(self.weight.grd.cpu, row_major[Self.IN, Self.ACT]())
                var c_ptr = self._cached_input_ptr.value()
                for i in range(Self.IN):
                    for j in range(Self.ACT):
                        var acc: Scalar[DT] = 0.0
                        for bi in range(BATCH):
                            acc += c_ptr[bi * Self.IN + i] * grad_output_v[bi, j]
                        gw[i, j] = gw[i, j] + acc

            # ── (3) grad_input = grad_mu @ W^T (always) ────────────────
            for bi in range(BATCH):
                for i in range(Self.IN):
                    var acc: Scalar[DT] = 0.0
                    for j in range(Self.ACT):
                        acc += grad_output_v[bi, j] * w[i, j]
                    grad_input_v[bi, i] = acc
        else:
            var ctx = self.ts.ctx.value()
            var go_p_w = mptr(grad_output_v.ptr)
            var gi_p_w = mptr(grad_input_v.ptr)

            comptime go_layout = Layout.row_major(BATCH, 2 * Self.ACT)
            comptime gi_layout = Layout.row_major(BATCH, Self.IN)
            comptime w_layout = Layout.row_major(Self.IN, Self.ACT)
            comptime b_layout = Layout.row_major(Self.ACT)
            comptime cache_layout = Layout.row_major(BATCH, Self.IN)

            var go_lt = LayoutTensor[DT, go_layout, MutAnyOrigin](go_p_w)
            var gi_lt = LayoutTensor[DT, gi_layout, MutAnyOrigin](gi_p_w)
            var w_lt = LayoutTensor[DT, w_layout, MutAnyOrigin](
                self.weight.val.dev.value()
            )

            # (1) grad_b, grad_log_std
            comptime if mode == "all":
                var gb_lt = LayoutTensor[DT, b_layout, MutAnyOrigin](
                    self.bias.grd.dev.value()
                )
                var gls_lt = LayoutTensor[DT, b_layout, MutAnyOrigin](
                    self.log_std.grd.dev.value()
                )
                comptime n_blocks_gb = (Self.ACT + TPB - 1) // TPB
                comptime gb_kernel = _gauss_head_grad_b_kernel[BATCH, Self.ACT]
                ctx.enqueue_function[gb_kernel](
                    go_lt, gb_lt,
                    grid_dim=n_blocks_gb, block_dim=TPB,
                )
                comptime gls_kernel = _gauss_head_grad_ls_kernel[BATCH, Self.ACT]
                ctx.enqueue_function[gls_kernel](
                    go_lt, gls_lt,
                    grid_dim=n_blocks_gb, block_dim=TPB,
                )

            # (2) grad_w
            comptime if mode == "all":
                var cache_lt = LayoutTensor[DT, cache_layout, MutAnyOrigin](
                    self._cached_input_ptr.value()
                )
                var gw_lt = LayoutTensor[DT, w_layout, MutAnyOrigin](
                    self.weight.grd.dev.value()
                )
                comptime n_blocks_gw = (Self.W_SIZE + TPB - 1) // TPB
                comptime gw_kernel = _gauss_head_grad_w_kernel[
                    BATCH, Self.IN, Self.ACT
                ]
                ctx.enqueue_function[gw_kernel](
                    cache_lt, go_lt, gw_lt,
                    grid_dim=n_blocks_gw, block_dim=TPB,
                )

            # (3) grad_input
            comptime n_blocks_gi = (BATCH * Self.IN + TPB - 1) // TPB
            comptime gi_kernel = _gauss_head_grad_input_kernel[
                BATCH, Self.IN, Self.ACT
            ]
            ctx.enqueue_function[gi_kernel](
                go_lt, w_lt, gi_lt,
                grid_dim=n_blocks_gi, block_dim=TPB,
            )

    # ----- Walkers ---------------------------------------------------------

    def for_each_param[
        target: StaticString,
        V: ParamVisitor,
    ](mut self, prefix: String, mut visitor: V) raises:
        assert_tag_for["GaussianHead", target](self.ts.target_tag)
        for_each_param_auto[Self, V, target](self, prefix, visitor)

    def zero_grad[target: StaticString](mut self) raises:
        assert_tag_for["GaussianHead", target](self.ts.target_tag)
        zero_grad_auto[Self, target](self)
