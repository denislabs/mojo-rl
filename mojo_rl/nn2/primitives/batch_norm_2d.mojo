"""BatchNorm2D[C, H, W, MOMENTUM, EPSILON] — per-channel BN for spatial inputs.

Phase 5 of `nn2/PORTING_PLAN.md`. Mirrors `batch_norm_1d.mojo`'s
surface — γ/β as `Param`s with `decay=False`, running_mean/var as
decay-exempt zero-grad `Param`s too (M1 — they ride the `for_each_param`
walk into the v2 checkpoint envelope; the optimizer visits them but BN
backward never writes their grad, so they stay BIT-EXACT and evolve only
via the forward EMA), per-instance `training: Bool`, `cache_is_training`
flag.

The only structural difference vs BN1D is the reduction axis: stats
are reduced over batch *and* spatial position (H·W), giving
`N_eff = BATCH · H · W` samples per channel. Forward and backward are
otherwise the standard BN formulas, applied per channel.

Comptime shape: input `[BATCH, C, H, W]` flattened to `[BATCH, C·H·W]`;
output is the same shape. Used after every `Conv2D` in a CNN trunk
(NatureDQN doesn't use it, but ResNet-style trunks do).

GPU layout: one block per channel, threads stride over BATCH·SPATIAL
samples and reduce via `block.sum[block_size=BN2D_TPB]`.
"""

from std.math import sqrt
from std.gpu import thread_idx, block_idx
from std.gpu.primitives import block
from std.gpu.host import DeviceContext, DeviceBuffer
from std.gpu.memory import AddressSpace
from layout import Layout, LayoutTensor, TileTensor, row_major

from ..constants import DT
from ..core import (
    Initializer,
    AMPPolicy,
    NoAMP,
    Param,
    ParamVisitor,
    for_each_param_auto,
    zero_grad_auto,
)
from ..core.module import Module, typed_view, typed_view_mut
from ..core.target_storage import (
    require_ctx,
    TargetStorage,
    assert_tag_for,
    ensure_cpu_buffer,
)


comptime BN2D_DEFAULT_EPS: Float64 = 1e-5
comptime BN2D_DEFAULT_MOM: Float64 = 0.1
comptime BN2D_TPB: Int = 128


# ──────────────────────────────────────────────────────────────────────
# GPU kernels — one block per channel, threads stride over the joint
# (batch, spatial) sample axis. Flat storage layout `[BATCH, C, SPATIAL]`
# is consumed via explicit address arithmetic so the LayoutTensor stays
# at a single 1-D shape we can index directly.
# ──────────────────────────────────────────────────────────────────────


def _bn2d_forward_train_kernel[
    BATCH: Int, C: Int, SPATIAL: Int, FLAT: Int,
    EPSILON: Float64, MOMENTUM: Float64,
](
    input: LayoutTensor[DT, Layout.row_major(BATCH, FLAT), MutAnyOrigin],
    output: LayoutTensor[DT, Layout.row_major(BATCH, FLAT), MutAnyOrigin],
    gamma: LayoutTensor[DT, Layout.row_major(C), MutAnyOrigin],
    beta:  LayoutTensor[DT, Layout.row_major(C), MutAnyOrigin],
    running_mean: LayoutTensor[DT, Layout.row_major(C), MutAnyOrigin],
    running_var:  LayoutTensor[DT, Layout.row_major(C), MutAnyOrigin],
    cache_xhat: LayoutTensor[DT, Layout.row_major(BATCH, FLAT), MutAnyOrigin],
    cache_inv_std: LayoutTensor[DT, Layout.row_major(C), MutAnyOrigin],
):
    var c = Int(block_idx.x)
    var t = Int(thread_idx.x)
    if c >= C:
        return

    var n_eff = BATCH * SPATIAL
    var inv_n: Scalar[DT] = 1.0 / Scalar[DT](Float32(n_eff))
    var eps = Scalar[DT](EPSILON)
    var mom = Scalar[DT](MOMENTUM)
    var one_m = Scalar[DT](1.0) - mom
    var c_off = c * SPATIAL

    # Three reduction passes (mean, var, xhat/scatter) all step (b, s)
    # through the per-channel slab as nested loops — avoids a `% SPATIAL`
    # / `// SPATIAL` per element. Each thread handles
    # SPATIAL // BN2D_TPB samples per batch (+1 for the tail).
    var my_sum: Scalar[DT] = 0.0
    for b in range(BATCH):
        var s = t
        while s < SPATIAL:
            my_sum += rebind[Scalar[DT]](input[b, c_off + s])
            s += BN2D_TPB
    var mean = (
        block.sum[block_size=BN2D_TPB, broadcast=True](val=my_sum) * inv_n
    )

    var my_var: Scalar[DT] = 0.0
    for b in range(BATCH):
        var s = t
        while s < SPATIAL:
            var d = rebind[Scalar[DT]](input[b, c_off + s]) - mean
            my_var += d * d
            s += BN2D_TPB
    var var_ = (
        block.sum[block_size=BN2D_TPB, broadcast=True](val=my_var) * inv_n
    )

    var inv_std: Scalar[DT] = 1.0 / sqrt(var_ + eps)
    if t == 0:
        cache_inv_std[c] = inv_std
        var rm = rebind[Scalar[DT]](running_mean[c])
        var rv = rebind[Scalar[DT]](running_var[c])
        running_mean[c] = one_m * rm + mom * mean
        running_var[c]  = one_m * rv + mom * var_

    var g = rebind[Scalar[DT]](gamma[c])
    var bt = rebind[Scalar[DT]](beta[c])
    for b in range(BATCH):
        var s = t
        while s < SPATIAL:
            var off = c_off + s
            var x = rebind[Scalar[DT]](input[b, off])
            var xh = (x - mean) * inv_std
            cache_xhat[b, off] = xh
            output[b, off] = g * xh + bt
            s += BN2D_TPB


def _bn2d_forward_eval_kernel[
    BATCH: Int, C: Int, SPATIAL: Int, FLAT: Int, EPSILON: Float64,
](
    input: LayoutTensor[DT, Layout.row_major(BATCH, FLAT), MutAnyOrigin],
    output: LayoutTensor[DT, Layout.row_major(BATCH, FLAT), MutAnyOrigin],
    gamma: LayoutTensor[DT, Layout.row_major(C), MutAnyOrigin],
    beta:  LayoutTensor[DT, Layout.row_major(C), MutAnyOrigin],
    running_mean: LayoutTensor[DT, Layout.row_major(C), MutAnyOrigin],
    running_var:  LayoutTensor[DT, Layout.row_major(C), MutAnyOrigin],
):
    var c = Int(block_idx.x)
    var t = Int(thread_idx.x)
    if c >= C:
        return
    var eps = Scalar[DT](EPSILON)
    var rm = rebind[Scalar[DT]](running_mean[c])
    var rv = rebind[Scalar[DT]](running_var[c])
    var inv_std: Scalar[DT] = 1.0 / sqrt(rv + eps)
    var g = rebind[Scalar[DT]](gamma[c])
    var bt = rebind[Scalar[DT]](beta[c])
    var c_off = c * SPATIAL
    for b in range(BATCH):
        var s = t
        while s < SPATIAL:
            var off = c_off + s
            var x = rebind[Scalar[DT]](input[b, off])
            output[b, off] = g * (x - rm) * inv_std + bt
            s += BN2D_TPB


def _bn2d_backward_kernel[
    BATCH: Int, C: Int, SPATIAL: Int, FLAT: Int,
](
    grad_output: LayoutTensor[DT, Layout.row_major(BATCH, FLAT), MutAnyOrigin],
    gamma: LayoutTensor[DT, Layout.row_major(C), MutAnyOrigin],
    cache_xhat: LayoutTensor[DT, Layout.row_major(BATCH, FLAT), MutAnyOrigin],
    cache_inv_std: LayoutTensor[DT, Layout.row_major(C), MutAnyOrigin],
    grad_input: LayoutTensor[DT, Layout.row_major(BATCH, FLAT), MutAnyOrigin],
    grad_gamma: LayoutTensor[DT, Layout.row_major(C), MutAnyOrigin],
    grad_beta:  LayoutTensor[DT, Layout.row_major(C), MutAnyOrigin],
):
    var c = Int(block_idx.x)
    var t = Int(thread_idx.x)
    if c >= C:
        return
    var n_eff = BATCH * SPATIAL
    var inv_n: Scalar[DT] = 1.0 / Scalar[DT](Float32(n_eff))
    var g = rebind[Scalar[DT]](gamma[c])
    var inv_std = rebind[Scalar[DT]](cache_inv_std[c])
    var c_off = c * SPATIAL

    # Same nested-(b, s) traversal as the forward — avoids per-element
    # `% SPATIAL` / `// SPATIAL` divisions.
    var my_sum_dxhat: Scalar[DT] = 0.0
    var my_sum_dxhat_xhat: Scalar[DT] = 0.0
    var my_dgamma: Scalar[DT] = 0.0
    var my_dbeta:  Scalar[DT] = 0.0
    for b in range(BATCH):
        var s = t
        while s < SPATIAL:
            var off = c_off + s
            var dy = rebind[Scalar[DT]](grad_output[b, off])
            var xh = rebind[Scalar[DT]](cache_xhat[b, off])
            var dxhat = dy * g
            my_sum_dxhat += dxhat
            my_sum_dxhat_xhat += dxhat * xh
            my_dgamma += dy * xh
            my_dbeta  += dy
            s += BN2D_TPB
    var sum_dxhat = block.sum[block_size=BN2D_TPB, broadcast=True](
        val=my_sum_dxhat
    )
    var sum_dxhat_xhat = block.sum[block_size=BN2D_TPB, broadcast=True](
        val=my_sum_dxhat_xhat
    )
    var d_gamma_tot = block.sum[block_size=BN2D_TPB, broadcast=False](
        val=my_dgamma
    )
    var d_beta_tot = block.sum[block_size=BN2D_TPB, broadcast=False](
        val=my_dbeta
    )
    if t == 0:
        grad_gamma[c] = (
            rebind[Scalar[DT]](grad_gamma[c]) + d_gamma_tot[0]
        )
        grad_beta[c] = (
            rebind[Scalar[DT]](grad_beta[c]) + d_beta_tot[0]
        )

    var m1 = sum_dxhat * inv_n
    var m2 = sum_dxhat_xhat * inv_n
    for b in range(BATCH):
        var s = t
        while s < SPATIAL:
            var off = c_off + s
            var dy = rebind[Scalar[DT]](grad_output[b, off])
            var xh = rebind[Scalar[DT]](cache_xhat[b, off])
            var dxhat = dy * g
            grad_input[b, off] = inv_std * (dxhat - m1 - xh * m2)
            s += BN2D_TPB


struct BatchNorm2D[
    C: Int, H: Int, W: Int,
    MOMENTUM: Float64 = BN2D_DEFAULT_MOM,
    EPSILON: Float64 = BN2D_DEFAULT_EPS,
](Module):
    comptime ARITY: Int = 1
    comptime FLAT_DIM: Int = Self.C * Self.H * Self.W
    comptime IN_DIMS = InlineArray[Int, 1](fill=Self.FLAT_DIM)
    comptime OUT_DIM = Self.FLAT_DIM
    comptime SPATIAL: Int = Self.H * Self.W

    var gamma: Param["gamma", False, Self.C]
    var beta:  Param["beta",  False, Self.C]
    # Running stats — decay-exempt, zero-grad Params (M1); walked by
    # for_each_param into the v2 checkpoint, never moved by the optimizer.
    var running_mean: Param["running_mean", False, Self.C]
    var running_var:  Param["running_var",  False, Self.C]
    var cache_xhat: List[Scalar[DT]]     # [BATCH, C, H, W] flat
    var cache_inv_std: List[Scalar[DT]]  # [C]
    var cache_xhat_dev: Optional[DeviceBuffer[DT]]
    var cache_inv_std_dev: Optional[DeviceBuffer[DT]]
    var cache_n_batch: Int
    var cache_is_training: Bool
    var training: Bool
    var ts: TargetStorage

    def __init__(out self):
        self.gamma = Param["gamma", False, Self.C]()
        self.beta  = Param["beta",  False, Self.C]()
        self.running_mean = Param["running_mean", False, Self.C]()
        self.running_var  = Param["running_var",  False, Self.C]()
        self.cache_xhat = List[Scalar[DT]]()
        self.cache_inv_std = List[Scalar[DT]]()
        self.cache_xhat_dev = None
        self.cache_inv_std_dev = None
        self.cache_n_batch = 0
        self.cache_is_training = False
        self.training = True
        self.ts = TargetStorage.make_uninit()

    @staticmethod
    def make[
        target: StaticString, INIT: Initializer
    ](
        ctx: Optional[DeviceContext] = None,
    ) raises -> Self:
        comptime assert target == "cpu" or target == "gpu", (
            "BatchNorm2D: target must be 'cpu' or 'gpu'"
        )
        comptime assert Self.C > 0 and Self.H > 0 and Self.W > 0, (
            "BatchNorm2D: C, H, W must all be > 0"
        )
        comptime assert Self.MOMENTUM > 0.0 and Self.MOMENTUM <= 1.0, (
            "BatchNorm2D: MOMENTUM must be in (0, 1]"
        )
        var bn = Self()
        comptime if target == "cpu":
            bn.gamma = Param["gamma", False, Self.C].make_cpu()
            bn.beta  = Param["beta",  False, Self.C].make_cpu()
            var g_ptr = bn.gamma.value_unsafe_ptr_cpu()
            for k in range(Self.C):
                g_ptr[k] = Scalar[DT](1.0)
            bn.running_mean = Param["running_mean", False, Self.C].make_cpu()
            bn.running_var  = Param["running_var",  False, Self.C].make_cpu()
            # make_cpu zero-fills value → running_mean already 0; set var←1.
            var rv_ptr = bn.running_var.value_unsafe_ptr_cpu()
            for k in range(Self.C):
                rv_ptr[k] = Scalar[DT](1.0)
            bn.ts = TargetStorage.make_cpu()
        else:
            var ctx_v = require_ctx["BatchNorm2D.make[target='gpu']"](ctx)
            bn.gamma = Param["gamma", False, Self.C].make_gpu(ctx_v)
            bn.beta  = Param["beta",  False, Self.C].make_gpu(ctx_v)
            bn.gamma.value_dev.value().enqueue_fill(1.0)
            bn.beta.value_dev.value().enqueue_fill(0.0)
            bn.running_mean = Param["running_mean", False, Self.C].make_gpu(
                ctx_v
            )
            bn.running_var = Param["running_var", False, Self.C].make_gpu(
                ctx_v
            )
            bn.running_mean.value_dev.value().enqueue_fill(0.0)
            bn.running_var.value_dev.value().enqueue_fill(1.0)
            bn.cache_xhat_dev    = ctx_v.enqueue_create_buffer[DT](1)
            bn.cache_inv_std_dev = ctx_v.enqueue_create_buffer[DT](Self.C)
            bn.cache_n_batch = 0
            bn.ts = TargetStorage.make_gpu(ctx_v)
        return bn^

    def _ensure_cache_gpu(mut self, batch: Int) raises:
        if self.cache_n_batch < batch:
            var ctx = self.ts.ctx.value()
            self.cache_xhat_dev = ctx.enqueue_create_buffer[DT](
                batch * Self.FLAT_DIM
            )
            self.cache_n_batch = batch

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
        assert_tag_for["BatchNorm2D", target](self.ts.target_tag)
        var input = typed_view[BATCH, Self.IN_DIMS[0]](inputs[0])
        var output_v = typed_view_mut[BATCH, Self.OUT_DIM](output)

        comptime if target == "cpu":
            var in_p = rebind[UnsafePointer[Scalar[DT], MutAnyOrigin]](
                input.ptr
            )
            var out_p = rebind[UnsafePointer[Scalar[DT], MutAnyOrigin]](
                output_v.ptr
            )
            var g_p = self.gamma.value_unsafe_ptr_cpu()
            var b_p = self.bias_unsafe_ptr_cpu()
            var rm_v = TileTensor(self.running_mean.value, row_major[Self.C]())
            var rv_v = TileTensor(self.running_var.value,  row_major[Self.C]())
            var eps = Scalar[DT](Self.EPSILON)
            var n_eff = Scalar[DT](Float64(BATCH * Self.SPATIAL))
            var inv_n = Scalar[DT](1.0) / n_eff
            if self.training:
                ensure_cpu_buffer(
                    self.cache_xhat, BATCH * Self.FLAT_DIM,
                )
                ensure_cpu_buffer(self.cache_inv_std, Self.C)
                var xhat_p = self.cache_xhat.unsafe_ptr()
                var inv_v = TileTensor(
                    self.cache_inv_std, row_major[Self.C](),
                )
                var mom = Scalar[DT](Self.MOMENTUM)
                var one_m = Scalar[DT](1.0) - mom
                for c in range(Self.C):
                    var g = g_p[c]
                    var bt = b_p[c]
                    var mean: Scalar[DT] = 0.0
                    for b in range(BATCH):
                        var base = (
                            b * Self.FLAT_DIM + c * Self.SPATIAL
                        )
                        for s in range(Self.SPATIAL):
                            mean += in_p[base + s]
                    mean *= inv_n
                    var var_: Scalar[DT] = 0.0
                    for b in range(BATCH):
                        var base = (
                            b * Self.FLAT_DIM + c * Self.SPATIAL
                        )
                        for s in range(Self.SPATIAL):
                            var d = in_p[base + s] - mean
                            var_ += d * d
                    var_ *= inv_n
                    var inv_std = Scalar[DT](1.0) / sqrt(var_ + eps)
                    inv_v[c] = inv_std
                    for b in range(BATCH):
                        var base = (
                            b * Self.FLAT_DIM + c * Self.SPATIAL
                        )
                        for s in range(Self.SPATIAL):
                            var xh = (in_p[base + s] - mean) * inv_std
                            xhat_p[base + s] = xh
                            out_p[base + s] = g * xh + bt
                    rm_v[c] = one_m * rm_v[c] + mom * mean
                    rv_v[c] = one_m * rv_v[c] + mom * var_
                self.cache_is_training = True
            else:
                for c in range(Self.C):
                    var rm = rm_v[c]
                    var rv = rv_v[c]
                    var inv_std = Scalar[DT](1.0) / sqrt(rv + eps)
                    var g = g_p[c]
                    var bt = b_p[c]
                    for b in range(BATCH):
                        var base = (
                            b * Self.FLAT_DIM + c * Self.SPATIAL
                        )
                        for s in range(Self.SPATIAL):
                            out_p[base + s] = (
                                g * (in_p[base + s] - rm) * inv_std + bt
                            )
        else:
            comptime layout_2d = Layout.row_major(BATCH, Self.FLAT_DIM)
            comptime layout_c  = Layout.row_major(Self.C)
            var in_p_w  = rebind[UnsafePointer[Scalar[DT], MutAnyOrigin]](
                input.ptr
            )
            var out_p_w = rebind[UnsafePointer[Scalar[DT], MutAnyOrigin]](
                output_v.ptr
            )
            var in_lt  = LayoutTensor[DT, layout_2d, MutAnyOrigin](in_p_w)
            var out_lt = LayoutTensor[DT, layout_2d, MutAnyOrigin](out_p_w)
            var g_lt = LayoutTensor[DT, layout_c, MutAnyOrigin](
                self.gamma.value_dev.value()
            )
            var b_lt = LayoutTensor[DT, layout_c, MutAnyOrigin](
                self.beta.value_dev.value()
            )
            var rm_lt = LayoutTensor[DT, layout_c, MutAnyOrigin](
                self.running_mean.value_dev.value()
            )
            var rv_lt = LayoutTensor[DT, layout_c, MutAnyOrigin](
                self.running_var.value_dev.value()
            )
            var ctx = self.ts.ctx.value()
            if self.training:
                self._ensure_cache_gpu(BATCH)
                var xh_lt = LayoutTensor[DT, layout_2d, MutAnyOrigin](
                    self.cache_xhat_dev.value()
                )
                var is_lt = LayoutTensor[DT, layout_c, MutAnyOrigin](
                    self.cache_inv_std_dev.value()
                )
                comptime fkernel = _bn2d_forward_train_kernel[
                    BATCH, Self.C, Self.SPATIAL, Self.FLAT_DIM,
                    Self.EPSILON, Self.MOMENTUM,
                ]
                ctx.enqueue_function[fkernel](
                    in_lt, out_lt, g_lt, b_lt, rm_lt, rv_lt,
                    xh_lt, is_lt,
                    grid_dim=Self.C, block_dim=BN2D_TPB,
                )
                self.cache_is_training = True
            else:
                comptime ekernel = _bn2d_forward_eval_kernel[
                    BATCH, Self.C, Self.SPATIAL, Self.FLAT_DIM,
                    Self.EPSILON,
                ]
                ctx.enqueue_function[ekernel](
                    in_lt, out_lt, g_lt, b_lt, rm_lt, rv_lt,
                    grid_dim=Self.C, block_dim=BN2D_TPB,
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
        assert_tag_for["BatchNorm2D", target](self.ts.target_tag)
        if not self.cache_is_training:
            raise Error(
                "BatchNorm2D.vjp: training-mode cache not populated."
                " Call forward(training=True) before vjp."
            )
        var grad_output_v = typed_view[BATCH, Self.OUT_DIM](grad_output)
        var grad_input_v = typed_view_mut[BATCH, Self.IN_DIMS[0]](
            grad_inputs[0]
        )

        comptime if target == "cpu":
            var go_p = rebind[UnsafePointer[Scalar[DT], MutAnyOrigin]](
                grad_output_v.ptr
            )
            var gi_p = rebind[UnsafePointer[Scalar[DT], MutAnyOrigin]](
                grad_input_v.ptr
            )
            var g_p = self.gamma.value_unsafe_ptr_cpu()
            var dg_p = self.gamma.grad_unsafe_ptr_cpu()
            var db_p = self.beta.grad_unsafe_ptr_cpu()
            var xhat_p = self.cache_xhat.unsafe_ptr()
            var inv_v = TileTensor(
                self.cache_inv_std, row_major[Self.C](),
            )
            var n_eff = Scalar[DT](Float64(BATCH * Self.SPATIAL))
            var inv_n = Scalar[DT](1.0) / n_eff
            for c in range(Self.C):
                var g = g_p[c]
                var inv_std = inv_v[c]
                var sum_dxhat: Scalar[DT] = 0.0
                var sum_dxhat_xhat: Scalar[DT] = 0.0
                var d_gamma: Scalar[DT] = 0.0
                var d_beta:  Scalar[DT] = 0.0
                for b in range(BATCH):
                    var base = b * Self.FLAT_DIM + c * Self.SPATIAL
                    for s in range(Self.SPATIAL):
                        var dy = go_p[base + s]
                        var xh = xhat_p[base + s]
                        var dxhat = dy * g
                        sum_dxhat += dxhat
                        sum_dxhat_xhat += dxhat * xh
                        d_gamma += dy * xh
                        d_beta  += dy
                var m1 = sum_dxhat * inv_n
                var m2 = sum_dxhat_xhat * inv_n
                for b in range(BATCH):
                    var base = b * Self.FLAT_DIM + c * Self.SPATIAL
                    for s in range(Self.SPATIAL):
                        var dy = go_p[base + s]
                        var xh = xhat_p[base + s]
                        var dxhat = dy * g
                        gi_p[base + s] = inv_std * (
                            dxhat - m1 - xh * m2
                        )
                comptime if mode == "all":
                    dg_p[c] += d_gamma
                    db_p[c] += d_beta
        else:
            comptime layout_2d = Layout.row_major(BATCH, Self.FLAT_DIM)
            comptime layout_c  = Layout.row_major(Self.C)
            var go_p = rebind[UnsafePointer[Scalar[DT], MutAnyOrigin]](
                grad_output_v.ptr
            )
            var gi_p = rebind[UnsafePointer[Scalar[DT], MutAnyOrigin]](
                grad_input_v.ptr
            )
            var go_lt = LayoutTensor[DT, layout_2d, MutAnyOrigin](go_p)
            var gi_lt = LayoutTensor[DT, layout_2d, MutAnyOrigin](gi_p)
            var g_lt = LayoutTensor[DT, layout_c, MutAnyOrigin](
                self.gamma.value_dev.value()
            )
            var xh_lt = LayoutTensor[DT, layout_2d, MutAnyOrigin](
                self.cache_xhat_dev.value()
            )
            var is_lt = LayoutTensor[DT, layout_c, MutAnyOrigin](
                self.cache_inv_std_dev.value()
            )
            var dg_lt = LayoutTensor[DT, layout_c, MutAnyOrigin](
                self.gamma.grad_dev.value()
            )
            var db_lt = LayoutTensor[DT, layout_c, MutAnyOrigin](
                self.beta.grad_dev.value()
            )
            comptime kernel = _bn2d_backward_kernel[
                BATCH, Self.C, Self.SPATIAL, Self.FLAT_DIM,
            ]
            self.ts.ctx.value().enqueue_function[kernel](
                go_lt, g_lt, xh_lt, is_lt, gi_lt, dg_lt, db_lt,
                grid_dim=Self.C, block_dim=BN2D_TPB,
            )

    # Inline helper so we don't read `beta.value` through `value_unsafe_ptr_cpu`
    # twice (one less symbol to update if Param's API shifts).
    def bias_unsafe_ptr_cpu(
        mut self,
    ) -> UnsafePointer[Scalar[DT], MutAnyOrigin]:
        return self.beta.value_unsafe_ptr_cpu()

    # ----- Walkers ---------------------------------------------------------

    def for_each_param[
        target: StaticString,
        V: ParamVisitor,
    ](mut self, prefix: String, mut visitor: V) raises:
        assert_tag_for["BatchNorm2D", target](self.ts.target_tag)
        for_each_param_auto[Self, V, target](self, prefix, visitor)

    def zero_grad[target: StaticString](mut self) raises:
        assert_tag_for["BatchNorm2D", target](self.ts.target_tag)
        zero_grad_auto[Self, target](self)

    def set_attr[ATTR: StaticString](mut self, value: Scalar[DT]):
        comptime if ATTR == "training":
            self.training = value > Scalar[DT](0.5)
