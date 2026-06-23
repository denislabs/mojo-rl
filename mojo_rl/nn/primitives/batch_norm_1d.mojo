"""BatchNorm1D[DIM, MOMENTUM, EPSILON] — per-feature batch norm (storage surface).

Transformed from legacy `nn.primitives.BatchNorm1D` (surface-only change; the
math, GPU kernels, finite-check, train/eval split, and cache discipline are
carried over verbatim). This is the first leaf with **running State**:

  - `gamma` / `beta` are `Param` (decay=False) — walked by `for_each_param`,
    optimized like any weight.
  - `running_mean` / `running_var` are plain owned `Tensor`s (the storage
    equivalent of legacy's decay-exempt `State`): they evolve ONLY via the
    forward EMA, are NOT visited by `for_each_param` (the optimizer never
    touches them), and will ride the checkpoint via a State walker in Stage 4.
  - `training: Bool` is a runtime field (default True), flipped via
    `set_training(...)`. Train caches x̂ + inv_std for backward; eval uses the
    running stats and leaves the cache untouched. `cache_is_training` guards
    against vjp-after-eval (a stale cache).

Backward (training cache):
    dx̂ = dy·γ ; m1 = mean_b(dx̂) ; m2 = mean_b(dx̂·x̂)
    dx = inv_std·(dx̂ - m1 - x̂·m2) ; dγ += Σ_b dy·x̂ ; dβ += Σ_b dy
"""

from std.math import sqrt
from std.gpu import thread_idx, block_idx
from std.gpu.primitives import block
from std.gpu.host import DeviceContext
from layout import Layout, LayoutTensor, TileTensor, row_major

from mojo_rl.nn.constants import DT
from ..core.tensor import Tensor
from ..core.tensor_refs import TensorRefs
from ..core.module import Module
from ..core.param import Param, ParamVisitor
from ..core.state import State
from ..core.initializer import Initializer
from ..core.amp import AMPPolicy, NoAMP


comptime BN_DEFAULT_EPS: Float64 = 1e-5
comptime BN_DEFAULT_MOM: Float64 = 0.1
comptime BN_TPB: Int = 128


# ── GPU kernels (verbatim from legacy; args MutAnyOrigin = GPU ABI) ─────
def _bn1d_forward_train_kernel[
    BATCH: Int,
    DIM: Int,
    EPSILON: Float64,
    MOMENTUM: Float64,
](
    input: LayoutTensor[DT, Layout.row_major(BATCH, DIM), MutAnyOrigin],
    output: LayoutTensor[DT, Layout.row_major(BATCH, DIM), MutAnyOrigin],
    gamma: LayoutTensor[DT, Layout.row_major(DIM), MutAnyOrigin],
    beta: LayoutTensor[DT, Layout.row_major(DIM), MutAnyOrigin],
    running_mean: LayoutTensor[DT, Layout.row_major(DIM), MutAnyOrigin],
    running_var: LayoutTensor[DT, Layout.row_major(DIM), MutAnyOrigin],
    cache_xhat: LayoutTensor[DT, Layout.row_major(BATCH, DIM), MutAnyOrigin],
    cache_inv_std: LayoutTensor[DT, Layout.row_major(DIM), MutAnyOrigin],
):
    var f = Int(block_idx.x)
    var t = Int(thread_idx.x)
    if f >= DIM:
        return
    var inv_n: Scalar[DT] = 1.0 / Scalar[DT](Float32(BATCH))
    var eps = Scalar[DT](EPSILON)
    var mom = Scalar[DT](MOMENTUM)
    var one_m = Scalar[DT](1.0) - mom
    var my_sum: input.element_type = 0.0
    var b = t
    while b < BATCH:
        my_sum += input[b, f]
        b += BN_TPB
    var mean = block.sum[block_size=BN_TPB, broadcast=True](val=my_sum) * inv_n
    var my_var: input.element_type = 0.0
    b = t
    while b < BATCH:
        var d = input[b, f] - mean
        my_var += d * d
        b += BN_TPB
    var var_ = block.sum[block_size=BN_TPB, broadcast=True](val=my_var) * inv_n
    var inv_std: input.element_type = 1.0 / sqrt(var_ + eps)
    if t == 0:
        cache_inv_std[f] = inv_std
        if (mean - mean == 0.0) and (var_ - var_ == 0.0):
            var rm = running_mean[f]
            var rv = running_var[f]
            running_mean[f] = one_m * rm + mom * mean
            running_var[f] = one_m * rv + mom * var_
    var g = gamma[f]
    var bt = beta[f]
    b = t
    while b < BATCH:
        var x = input[b, f]
        var xh = (x - mean) * inv_std
        cache_xhat[b, f] = xh
        output[b, f] = g * xh + bt
        b += BN_TPB


def _bn1d_forward_eval_kernel[
    BATCH: Int,
    DIM: Int,
    EPSILON: Float64,
](
    input: LayoutTensor[DT, Layout.row_major(BATCH, DIM), MutAnyOrigin],
    output: LayoutTensor[DT, Layout.row_major(BATCH, DIM), MutAnyOrigin],
    gamma: LayoutTensor[DT, Layout.row_major(DIM), MutAnyOrigin],
    beta: LayoutTensor[DT, Layout.row_major(DIM), MutAnyOrigin],
    running_mean: LayoutTensor[DT, Layout.row_major(DIM), MutAnyOrigin],
    running_var: LayoutTensor[DT, Layout.row_major(DIM), MutAnyOrigin],
):
    var f = Int(block_idx.x)
    var t = Int(thread_idx.x)
    if f >= DIM:
        return
    var eps = Scalar[DT](EPSILON)
    var rm = running_mean[f]
    var rv = running_var[f]
    var inv_std: input.element_type = 1.0 / sqrt(rv + eps)
    var g = gamma[f]
    var bt = beta[f]
    var b = t
    while b < BATCH:
        var x = input[b, f]
        output[b, f] = g * (x - rm) * inv_std + bt
        b += BN_TPB


def _bn1d_backward_kernel[
    BATCH: Int,
    DIM: Int,
](
    grad_output: LayoutTensor[DT, Layout.row_major(BATCH, DIM), MutAnyOrigin],
    gamma: LayoutTensor[DT, Layout.row_major(DIM), MutAnyOrigin],
    cache_xhat: LayoutTensor[DT, Layout.row_major(BATCH, DIM), MutAnyOrigin],
    cache_inv_std: LayoutTensor[DT, Layout.row_major(DIM), MutAnyOrigin],
    grad_input: LayoutTensor[DT, Layout.row_major(BATCH, DIM), MutAnyOrigin],
    grad_gamma: LayoutTensor[DT, Layout.row_major(DIM), MutAnyOrigin],
    grad_beta: LayoutTensor[DT, Layout.row_major(DIM), MutAnyOrigin],
):
    var f = Int(block_idx.x)
    var t = Int(thread_idx.x)
    if f >= DIM:
        return
    var inv_n: Scalar[DT] = 1.0 / Scalar[DT](Float32(BATCH))
    var g = gamma[f]
    var inv_std = cache_inv_std[f]
    var my_sum_dxhat: grad_output.element_type = 0.0
    var my_sum_dxhat_xhat: grad_output.element_type = 0.0
    var my_dgamma: grad_output.element_type = 0.0
    var my_dbeta: grad_output.element_type = 0.0
    var b = t
    while b < BATCH:
        var dy = grad_output[b, f]
        var xh = cache_xhat[b, f]
        var dxhat = dy * g
        my_sum_dxhat += dxhat
        my_sum_dxhat_xhat += dxhat * xh
        my_dgamma += dy * xh
        my_dbeta += dy
        b += BN_TPB
    var sum_dxhat = block.sum[block_size=BN_TPB, broadcast=True](
        val=my_sum_dxhat
    )
    var sum_dxhat_xhat = block.sum[block_size=BN_TPB, broadcast=True](
        val=my_sum_dxhat_xhat
    )
    var d_gamma_tot = block.sum[block_size=BN_TPB, broadcast=False](
        val=my_dgamma
    )
    var d_beta_tot = block.sum[block_size=BN_TPB, broadcast=False](val=my_dbeta)
    if t == 0:
        grad_gamma[f] = grad_gamma[f] + d_gamma_tot[0]
        grad_beta[f] = grad_beta[f] + d_beta_tot[0]
    var m1 = sum_dxhat * inv_n
    var m2 = sum_dxhat_xhat * inv_n
    b = t
    while b < BATCH:
        var dy = grad_output[b, f]
        var xh = cache_xhat[b, f]
        var dxhat = dy * g
        grad_input[b, f] = inv_std * (dxhat - m1 - xh * m2)
        b += BN_TPB


struct BatchNorm1D[
    DIM_: Int,
    MOMENTUM: Float64 = BN_DEFAULT_MOM,
    EPSILON: Float64 = BN_DEFAULT_EPS,
](Module):
    comptime ARITY = 1
    comptime IN_DIMS = InlineArray[Int, 1](fill=Self.DIM_)
    comptime OUT_DIM = Self.DIM_

    var gamma: Param["gamma", False, Self.DIM_]
    var beta: Param["beta", False, Self.DIM_]
    # Running stats (State): EMA-updated in forward, never optimized.
    var running_mean: State["running_mean", Self.DIM_]
    var running_var: State["running_var", Self.DIM_]
    # Training cache (owned storage — sound; not a back-pointer).
    var cache_xhat: Tensor  # [BATCH, DIM]
    var cache_inv_std: Tensor  # [DIM]
    var cache_is_training: Bool
    var training: Bool

    def __init__(out self):
        self.gamma = Param["gamma", False, Self.DIM_]()
        self.beta = Param["beta", False, Self.DIM_]()
        self.running_mean = State["running_mean", Self.DIM_]()
        self.running_var = State["running_var", Self.DIM_]()
        self.cache_xhat = Tensor()
        self.cache_inv_std = Tensor()
        self.cache_is_training = False
        self.training = True

    @staticmethod
    def make[
        target: StaticString, INIT: Initializer
    ](ctx: Optional[DeviceContext] = None) raises -> Self:
        var bn = Self()
        bn.gamma = Param["gamma", False, Self.DIM_].make[target](ctx)
        bn.beta = Param["beta", False, Self.DIM_].make[target](ctx)
        for k in range(Self.DIM_):
            bn.gamma.val.data[k] = Scalar[DT](1.0)  # γ←1, β←0
        bn.running_mean = State["running_mean", Self.DIM_].make[target](ctx)  # ←0
        bn.running_var = State["running_var", Self.DIM_].make[target](ctx)
        for k in range(Self.DIM_):
            bn.running_var.t.data[k] = Scalar[DT](1.0)  # σ²_run←1
        comptime if target != "cpu":
            var dctx = ctx.value()
            bn.gamma.val.upload(dctx)
            bn.beta.val.upload(dctx)  # zeros → device
            bn.running_mean.t.upload(dctx)
            bn.running_var.t.upload(dctx)
        return bn^

    def set_training(mut self, v: Bool):
        self.training = v

    def set_attr[ATTR: StaticString](mut self, value: Scalar[DT]):
        """Named-attr hook so a parent combinator can toggle BN train/eval via
        `net.set_attr["training"](1.0/0.0)`. `value != 0` → training."""
        comptime if ATTR == "training":
            self.training = value != Scalar[DT](0.0)

    def forward[
        target: StaticString, B: Int, o: MutOrigin, POLICY: AMPPolicy = NoAMP
    ](
        mut self,
        inputs: TensorRefs[1, o],
        mut out: Tensor,
        ctx: Optional[DeviceContext] = None,
    ) raises:
        ref in0 = inputs[0]
        var eps = Scalar[DT](Self.EPSILON)
        comptime if target == "cpu":
            out.ensure(B * Self.DIM_)
            var input = TileTensor(in0.data, row_major[B, Self.DIM_]())
            var output_v = TileTensor(out.data, row_major[B, Self.DIM_]())
            var gamma_v = TileTensor(
                self.gamma.val.data, row_major[Self.DIM_]()
            )
            var beta_v = TileTensor(self.beta.val.data, row_major[Self.DIM_]())
            var rm_v = TileTensor(
                self.running_mean.t.data, row_major[Self.DIM_]()
            )
            var rv_v = TileTensor(self.running_var.t.data, row_major[Self.DIM_]())
            if self.training:
                self.cache_xhat.ensure(B * Self.DIM_)
                self.cache_inv_std.ensure(Self.DIM_)
                var xhat_v = TileTensor(
                    self.cache_xhat.data, row_major[B, Self.DIM_]()
                )
                var inv_v = TileTensor(
                    self.cache_inv_std.data, row_major[Self.DIM_]()
                )
                var inv_n = Scalar[DT](1.0) / Scalar[DT](Float64(B))
                var mom = Scalar[DT](Self.MOMENTUM)
                var one_m = Scalar[DT](1.0) - mom
                for f in range(Self.DIM_):
                    var mean: Scalar[DT] = 0.0
                    for b in range(B):
                        mean += input[b, f]
                    mean *= inv_n
                    var var_: Scalar[DT] = 0.0
                    for b in range(B):
                        var diff = input[b, f] - mean
                        var_ += diff * diff
                    var_ *= inv_n
                    var inv_std = Scalar[DT](1.0) / sqrt(var_ + eps)
                    inv_v[f] = inv_std
                    var g = gamma_v[f]
                    var bt = beta_v[f]
                    for b in range(B):
                        var xh = (input[b, f] - mean) * inv_std
                        xhat_v[b, f] = xh
                        output_v[b, f] = g * xh + bt
                    rm_v[f] = one_m * rm_v[f] + mom * mean
                    rv_v[f] = one_m * rv_v[f] + mom * var_
                self.cache_is_training = True
            else:
                for f in range(Self.DIM_):
                    var inv_std = Scalar[DT](1.0) / sqrt(rv_v[f] + eps)
                    var g = gamma_v[f]
                    var bt = beta_v[f]
                    var rm = rm_v[f]
                    for b in range(B):
                        output_v[b, f] = g * (input[b, f] - rm) * inv_std + bt
        else:
            var c = ctx.value()
            out.ensure_gpu(c, B * Self.DIM_)
            comptime l2d = Layout.row_major(B, Self.DIM_)
            comptime ld = Layout.row_major(Self.DIM_)
            if self.training:
                self.cache_xhat.ensure_gpu(c, B * Self.DIM_)
                self.cache_inv_std.ensure_gpu(c, Self.DIM_)
                c.enqueue_function[
                    _bn1d_forward_train_kernel[
                        B,
                        Self.DIM_,
                        Self.EPSILON,
                        Self.MOMENTUM,
                    ]
                ](
                    in0.lt["gpu", l2d](),
                    out.lt["gpu", l2d](),
                    self.gamma.val.lt["gpu", ld](),
                    self.beta.val.lt["gpu", ld](),
                    self.running_mean.t.lt["gpu", ld](),
                    self.running_var.t.lt["gpu", ld](),
                    self.cache_xhat.lt["gpu", l2d](),
                    self.cache_inv_std.lt["gpu", ld](),
                    grid_dim=Self.DIM_,
                    block_dim=BN_TPB,
                )
                self.cache_is_training = True
            else:
                c.enqueue_function[
                    _bn1d_forward_eval_kernel[
                        B,
                        Self.DIM_,
                        Self.EPSILON,
                    ]
                ](
                    in0.lt["gpu", l2d](),
                    out.lt["gpu", l2d](),
                    self.gamma.val.lt["gpu", ld](),
                    self.beta.val.lt["gpu", ld](),
                    self.running_mean.t.lt["gpu", ld](),
                    self.running_var.t.lt["gpu", ld](),
                    grid_dim=Self.DIM_,
                    block_dim=BN_TPB,
                )

    def vjp[
        target: StaticString,
        B: Int,
        ofi: MutOrigin,
        ogi: MutOrigin,
        POLICY: AMPPolicy = NoAMP,
    ](
        mut self,
        forward_input: TensorRefs[1, ofi],
        mut grad_output: Tensor,
        grad_inputs: TensorRefs[1, ogi],
        ctx: Optional[DeviceContext] = None,
    ) raises:
        if not self.cache_is_training:
            raise Error(
                "BatchNorm1D.vjp: training-mode cache not populated. Call"
                " forward with training=True before vjp."
            )
        ref gin = grad_inputs[0]
        comptime if target == "cpu":
            gin.ensure(B * Self.DIM_)
            var go_v = TileTensor(grad_output.data, row_major[B, Self.DIM_]())
            var gi_v = TileTensor(gin.data, row_major[B, Self.DIM_]())
            var gamma_v = TileTensor(
                self.gamma.val.data, row_major[Self.DIM_]()
            )
            var dgamma_v = TileTensor(
                self.gamma.grd.data, row_major[Self.DIM_]()
            )
            var dbeta_v = TileTensor(self.beta.grd.data, row_major[Self.DIM_]())
            var xhat_v = TileTensor(
                self.cache_xhat.data, row_major[B, Self.DIM_]()
            )
            var inv_v = TileTensor(
                self.cache_inv_std.data, row_major[Self.DIM_]()
            )
            var inv_n = Scalar[DT](1.0) / Scalar[DT](Float64(B))
            for f in range(Self.DIM_):
                var g = gamma_v[f]
                var inv_std = inv_v[f]
                var sum_dxhat: Scalar[DT] = 0.0
                var sum_dxhat_xhat: Scalar[DT] = 0.0
                var d_gamma: Scalar[DT] = 0.0
                var d_beta: Scalar[DT] = 0.0
                for b in range(B):
                    var dy = go_v[b, f]
                    var xh = xhat_v[b, f]
                    var dxhat = dy * g
                    sum_dxhat += dxhat
                    sum_dxhat_xhat += dxhat * xh
                    d_gamma += dy * xh
                    d_beta += dy
                var m1 = sum_dxhat * inv_n
                var m2 = sum_dxhat_xhat * inv_n
                for b in range(B):
                    var dy = go_v[b, f]
                    var xh = xhat_v[b, f]
                    var dxhat = dy * g
                    gi_v[b, f] = inv_std * (dxhat - m1 - xh * m2)
                dgamma_v[f] += d_gamma
                dbeta_v[f] += d_beta
        else:
            var c = ctx.value()
            gin.ensure_gpu(c, B * Self.DIM_)
            comptime l2d = Layout.row_major(B, Self.DIM_)
            comptime ld = Layout.row_major(Self.DIM_)
            c.enqueue_function[_bn1d_backward_kernel[B, Self.DIM_]](
                grad_output.lt["gpu", l2d](),
                self.gamma.val.lt["gpu", ld](),
                self.cache_xhat.lt["gpu", l2d](),
                self.cache_inv_std.lt["gpu", ld](),
                gin.lt["gpu", l2d](),
                self.gamma.grd.lt["gpu", ld](),
                self.beta.grd.lt["gpu", ld](),
                grid_dim=Self.DIM_,
                block_dim=BN_TPB,
            )

    # for_each_param / zero_grad inherit the Module reflection defaults
    # (core/walkers.mojo auto-discovers the Param fields).
