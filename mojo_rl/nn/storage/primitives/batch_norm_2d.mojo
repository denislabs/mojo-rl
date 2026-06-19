"""BatchNorm2D[C, H, W, MOMENTUM, EPSILON] — per-channel BN for spatial inputs.

Transformed from legacy `nn.primitives.BatchNorm2D` (surface-only change). The
per-channel reduction over batch×spatial, the multi-block GPU reduction
(partial → finalize → scatter, the Σx/Σx² one-pass variance form), the finite-
guarded EMA, and the train/eval split are all carried over verbatim. Same State
treatment as BatchNorm1D: γ/β are Param (optimized); running_mean/var are
owned `Tensor`s evolved only by the forward EMA (not optimized).
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


comptime BN2D_DEFAULT_EPS: Float64 = 1e-5
comptime BN2D_DEFAULT_MOM: Float64 = 0.1
comptime BN2D_TPB: Int = 128
comptime BN2D_RBLOCKS: Int = 64


# ── GPU kernels (verbatim from legacy; args MutAnyOrigin = GPU ABI) ─────
def _bn2d_partial_stats_kernel[
    BATCH: Int,
    C: Int,
    SPATIAL: Int,
    FLAT: Int,
    G: Int,
](
    input: LayoutTensor[DT, Layout.row_major(BATCH, FLAT), MutAnyOrigin],
    partial_sum: LayoutTensor[DT, Layout.row_major(C * G), MutAnyOrigin],
    partial_sumsq: LayoutTensor[DT, Layout.row_major(C * G), MutAnyOrigin],
):
    var blk = Int(block_idx.x)
    var c = blk // G
    var g = blk % G
    if c >= C:
        return
    var t = Int(thread_idx.x)
    var bpb = (BATCH + G - 1) // G
    var b0 = g * bpb
    var b1 = b0 + bpb
    if b1 > BATCH:
        b1 = BATCH
    var c_off = c * SPATIAL
    var my_sum: input.element_type = 0.0
    var my_sumsq: input.element_type = 0.0
    for b in range(b0, b1):
        var s = t
        while s < SPATIAL:
            var x = input[b, c_off + s]
            my_sum += x
            my_sumsq += x * x
            s += BN2D_TPB
    var bsum = block.sum[block_size=BN2D_TPB, broadcast=False](val=my_sum)
    var bsq = block.sum[block_size=BN2D_TPB, broadcast=False](val=my_sumsq)
    if t == 0:
        partial_sum[c * G + g] = bsum[0]
        partial_sumsq[c * G + g] = bsq[0]


def _bn2d_finalize_stats_kernel[
    BATCH: Int,
    C: Int,
    SPATIAL: Int,
    G: Int,
    EPSILON: Float64,
    MOMENTUM: Float64,
](
    partial_sum: LayoutTensor[DT, Layout.row_major(C * G), MutAnyOrigin],
    partial_sumsq: LayoutTensor[DT, Layout.row_major(C * G), MutAnyOrigin],
    running_mean: LayoutTensor[DT, Layout.row_major(C), MutAnyOrigin],
    running_var: LayoutTensor[DT, Layout.row_major(C), MutAnyOrigin],
    cache_mean: LayoutTensor[DT, Layout.row_major(C), MutAnyOrigin],
    cache_inv_std: LayoutTensor[DT, Layout.row_major(C), MutAnyOrigin],
):
    var c = Int(block_idx.x)
    if c >= C:
        return
    if Int(thread_idx.x) != 0:
        return
    var s: partial_sum.element_type = 0.0
    var sq: partial_sumsq.element_type = 0.0
    for g in range(G):
        s += partial_sum[c * G + g]
        sq += partial_sumsq[c * G + g]
    var inv_n: Scalar[DT] = 1.0 / Scalar[DT](Float32(BATCH * SPATIAL))
    var mean = s * inv_n
    var var_ = sq * inv_n - mean * mean
    if var_ < Scalar[DT](0.0):
        var_ = Scalar[DT](0.0)
    var eps: partial_sum.element_type = Scalar[DT](EPSILON)
    var inv_std: partial_sum.element_type = 1.0 / sqrt(var_ + eps)
    cache_mean[c] = mean
    cache_inv_std[c] = inv_std
    if (mean - mean == Scalar[DT](0.0)) and (var_ - var_ == Scalar[DT](0.0)):
        var mom = Scalar[DT](MOMENTUM)
        var one_m = Scalar[DT](1.0) - mom
        running_mean[c] = one_m * running_mean[c] + mom * mean
        running_var[c] = one_m * running_var[c] + mom * var_


def _bn2d_normalize_kernel[
    BATCH: Int,
    C: Int,
    SPATIAL: Int,
    FLAT: Int,
    G: Int,
](
    input: LayoutTensor[DT, Layout.row_major(BATCH, FLAT), MutAnyOrigin],
    output: LayoutTensor[DT, Layout.row_major(BATCH, FLAT), MutAnyOrigin],
    gamma: LayoutTensor[DT, Layout.row_major(C), MutAnyOrigin],
    beta: LayoutTensor[DT, Layout.row_major(C), MutAnyOrigin],
    cache_mean: LayoutTensor[DT, Layout.row_major(C), MutAnyOrigin],
    cache_inv_std: LayoutTensor[DT, Layout.row_major(C), MutAnyOrigin],
    cache_xhat: LayoutTensor[DT, Layout.row_major(BATCH, FLAT), MutAnyOrigin],
):
    var blk = Int(block_idx.x)
    var c = blk // G
    var g = blk % G
    if c >= C:
        return
    var t = Int(thread_idx.x)
    var bpb = (BATCH + G - 1) // G
    var b0 = g * bpb
    var b1 = b0 + bpb
    if b1 > BATCH:
        b1 = BATCH
    var c_off = c * SPATIAL
    var mean = cache_mean[c]
    var inv_std = cache_inv_std[c]
    var gm = gamma[c]
    var bt = beta[c]
    for b in range(b0, b1):
        var s = t
        while s < SPATIAL:
            var off = c_off + s
            var xh = (input[b, off] - mean) * inv_std
            cache_xhat[b, off] = xh
            output[b, off] = gm * xh + bt
            s += BN2D_TPB


def _bn2d_forward_eval_kernel[
    BATCH: Int,
    C: Int,
    SPATIAL: Int,
    FLAT: Int,
    EPSILON: Float64,
](
    input: LayoutTensor[DT, Layout.row_major(BATCH, FLAT), MutAnyOrigin],
    output: LayoutTensor[DT, Layout.row_major(BATCH, FLAT), MutAnyOrigin],
    gamma: LayoutTensor[DT, Layout.row_major(C), MutAnyOrigin],
    beta: LayoutTensor[DT, Layout.row_major(C), MutAnyOrigin],
    running_mean: LayoutTensor[DT, Layout.row_major(C), MutAnyOrigin],
    running_var: LayoutTensor[DT, Layout.row_major(C), MutAnyOrigin],
):
    var c = Int(block_idx.x)
    var t = Int(thread_idx.x)
    if c >= C:
        return
    var eps = Scalar[DT](EPSILON)
    var rm = running_mean[c]
    var rv = running_var[c]
    var inv_std: input.element_type = 1.0 / sqrt(rv + eps)
    var g = gamma[c]
    var bt = beta[c]
    var c_off = c * SPATIAL
    for b in range(BATCH):
        var s = t
        while s < SPATIAL:
            var off = c_off + s
            var x = input[b, off]
            output[b, off] = g * (x - rm) * inv_std + bt
            s += BN2D_TPB


def _bn2d_bwd_partial_kernel[
    BATCH: Int,
    C: Int,
    SPATIAL: Int,
    FLAT: Int,
    G: Int,
](
    grad_output: LayoutTensor[DT, Layout.row_major(BATCH, FLAT), MutAnyOrigin],
    gamma: LayoutTensor[DT, Layout.row_major(C), MutAnyOrigin],
    cache_xhat: LayoutTensor[DT, Layout.row_major(BATCH, FLAT), MutAnyOrigin],
    p_dxhat: LayoutTensor[DT, Layout.row_major(C * G), MutAnyOrigin],
    p_dxhat_xhat: LayoutTensor[DT, Layout.row_major(C * G), MutAnyOrigin],
    p_dgamma: LayoutTensor[DT, Layout.row_major(C * G), MutAnyOrigin],
    p_dbeta: LayoutTensor[DT, Layout.row_major(C * G), MutAnyOrigin],
):
    var blk = Int(block_idx.x)
    var c = blk // G
    var g = blk % G
    if c >= C:
        return
    var t = Int(thread_idx.x)
    var bpb = (BATCH + G - 1) // G
    var b0 = g * bpb
    var b1 = b0 + bpb
    if b1 > BATCH:
        b1 = BATCH
    var gm = gamma[c]
    var c_off = c * SPATIAL
    var s_dxhat: grad_output.element_type = 0.0
    var s_dxx: grad_output.element_type = 0.0
    var s_dg: grad_output.element_type = 0.0
    var s_db: grad_output.element_type = 0.0
    for b in range(b0, b1):
        var s = t
        while s < SPATIAL:
            var off = c_off + s
            var dy = grad_output[b, off]
            var xh = cache_xhat[b, off]
            var dxhat = dy * gm
            s_dxhat += dxhat
            s_dxx += dxhat * xh
            s_dg += dy * xh
            s_db += dy
            s += BN2D_TPB
    var a = block.sum[block_size=BN2D_TPB, broadcast=False](val=s_dxhat)
    var bb = block.sum[block_size=BN2D_TPB, broadcast=False](val=s_dxx)
    var cc = block.sum[block_size=BN2D_TPB, broadcast=False](val=s_dg)
    var dd = block.sum[block_size=BN2D_TPB, broadcast=False](val=s_db)
    if t == 0:
        p_dxhat[c * G + g] = a[0]
        p_dxhat_xhat[c * G + g] = bb[0]
        p_dgamma[c * G + g] = cc[0]
        p_dbeta[c * G + g] = dd[0]


def _bn2d_bwd_finalize_kernel[
    BATCH: Int,
    C: Int,
    SPATIAL: Int,
    G: Int,
    mode: StaticString,
](
    p_dxhat: LayoutTensor[DT, Layout.row_major(C * G), MutAnyOrigin],
    p_dxhat_xhat: LayoutTensor[DT, Layout.row_major(C * G), MutAnyOrigin],
    p_dgamma: LayoutTensor[DT, Layout.row_major(C * G), MutAnyOrigin],
    p_dbeta: LayoutTensor[DT, Layout.row_major(C * G), MutAnyOrigin],
    m1_out: LayoutTensor[DT, Layout.row_major(C), MutAnyOrigin],
    m2_out: LayoutTensor[DT, Layout.row_major(C), MutAnyOrigin],
    grad_gamma: LayoutTensor[DT, Layout.row_major(C), MutAnyOrigin],
    grad_beta: LayoutTensor[DT, Layout.row_major(C), MutAnyOrigin],
):
    var c = Int(block_idx.x)
    if c >= C:
        return
    if Int(thread_idx.x) != 0:
        return
    var sa: p_dxhat.element_type = 0.0
    var sb: p_dxhat_xhat.element_type = 0.0
    var sg: p_dgamma.element_type = 0.0
    var sd: p_dbeta.element_type = 0.0
    for g in range(G):
        sa += p_dxhat[c * G + g]
        sb += p_dxhat_xhat[c * G + g]
        sg += p_dgamma[c * G + g]
        sd += p_dbeta[c * G + g]
    var inv_n: Scalar[DT] = 1.0 / Scalar[DT](BATCH * SPATIAL)
    m1_out[c] = sa * inv_n
    m2_out[c] = sb * inv_n
    comptime if mode == "all":
        grad_gamma[c] += sg
        grad_beta[c] += sd


def _bn2d_bwd_scatter_kernel[
    BATCH: Int,
    C: Int,
    SPATIAL: Int,
    FLAT: Int,
    G: Int,
](
    grad_output: LayoutTensor[DT, Layout.row_major(BATCH, FLAT), MutAnyOrigin],
    gamma: LayoutTensor[DT, Layout.row_major(C), MutAnyOrigin],
    cache_xhat: LayoutTensor[DT, Layout.row_major(BATCH, FLAT), MutAnyOrigin],
    cache_inv_std: LayoutTensor[DT, Layout.row_major(C), MutAnyOrigin],
    m1: LayoutTensor[DT, Layout.row_major(C), MutAnyOrigin],
    m2: LayoutTensor[DT, Layout.row_major(C), MutAnyOrigin],
    grad_input: LayoutTensor[DT, Layout.row_major(BATCH, FLAT), MutAnyOrigin],
):
    var blk = Int(block_idx.x)
    var c = blk // G
    var g = blk % G
    if c >= C:
        return
    var t = Int(thread_idx.x)
    var bpb = (BATCH + G - 1) // G
    var b0 = g * bpb
    var b1 = b0 + bpb
    if b1 > BATCH:
        b1 = BATCH
    var gm = gamma[c]
    var inv_std = cache_inv_std[c]
    var mm1 = m1[c]
    var mm2 = m2[c]
    var c_off = c * SPATIAL
    for b in range(b0, b1):
        var s = t
        while s < SPATIAL:
            var off = c_off + s
            var dy = grad_output[b, off]
            var xh = cache_xhat[b, off]
            var dxhat = dy * gm
            grad_input[b, off] = inv_std * (dxhat - mm1 - xh * mm2)
            s += BN2D_TPB


struct BatchNorm2D[
    C_: Int,
    H_: Int,
    W_: Int,
    MOMENTUM: Float64 = BN2D_DEFAULT_MOM,
    EPSILON: Float64 = BN2D_DEFAULT_EPS,
](Module):
    comptime ARITY = 1
    comptime FLAT_DIM = Self.C_ * Self.H_ * Self.W_
    comptime IN_DIMS = InlineArray[Int, 1](fill=Self.FLAT_DIM)
    comptime OUT_DIM = Self.FLAT_DIM
    comptime SPATIAL = Self.H_ * Self.W_

    var gamma: Param["gamma", False, Self.C_]
    var beta: Param["beta", False, Self.C_]
    var running_mean: State["running_mean", Self.C_]  # [C] State
    var running_var: State["running_var", Self.C_]  # [C] State
    var cache_xhat: Tensor  # [BATCH, FLAT]
    var cache_inv_std: Tensor  # [C]
    var cache_mean: Tensor  # [C] (GPU multiblock normalize)
    # Multi-block reduction scratch ([C·RBLOCKS] or [C]).
    var bn_psum: Tensor
    var bn_psumsq: Tensor
    var bn_pdg: Tensor
    var bn_pdb: Tensor
    var bn_m1: Tensor
    var bn_m2: Tensor
    var cache_is_training: Bool
    var training: Bool

    def __init__(out self):
        self.gamma = Param["gamma", False, Self.C_]()
        self.beta = Param["beta", False, Self.C_]()
        self.running_mean = State["running_mean", Self.C_]()
        self.running_var = State["running_var", Self.C_]()
        self.cache_xhat = Tensor()
        self.cache_inv_std = Tensor()
        self.cache_mean = Tensor()
        self.bn_psum = Tensor()
        self.bn_psumsq = Tensor()
        self.bn_pdg = Tensor()
        self.bn_pdb = Tensor()
        self.bn_m1 = Tensor()
        self.bn_m2 = Tensor()
        self.cache_is_training = False
        self.training = True

    @staticmethod
    def make[
        target: StaticString, INIT: Initializer
    ](ctx: Optional[DeviceContext] = None) raises -> Self:
        var bn = Self()
        bn.gamma = Param["gamma", False, Self.C_].make[target](ctx)
        bn.beta = Param["beta", False, Self.C_].make[target](ctx)
        for k in range(Self.C_):
            bn.gamma.val.data[k] = Scalar[DT](1.0)
        bn.running_mean = State["running_mean", Self.C_].make[target](ctx)
        bn.running_var = State["running_var", Self.C_].make[target](ctx)
        for k in range(Self.C_):
            bn.running_var.t.data[k] = Scalar[DT](1.0)
        comptime if target != "cpu":
            var c = ctx.value()
            bn.gamma.val.upload(c)
            bn.beta.val.upload(c)
            bn.running_mean.t.upload(c)
            bn.running_var.t.upload(c)
            # Multi-block scratch + channel caches.
            comptime PR = Self.C_ * BN2D_RBLOCKS
            bn.cache_inv_std.ensure_gpu(c, Self.C_)
            bn.cache_mean.ensure_gpu(c, Self.C_)
            bn.bn_psum.ensure_gpu(c, PR)
            bn.bn_psumsq.ensure_gpu(c, PR)
            bn.bn_pdg.ensure_gpu(c, PR)
            bn.bn_pdb.ensure_gpu(c, PR)
            bn.bn_m1.ensure_gpu(c, Self.C_)
            bn.bn_m2.ensure_gpu(c, Self.C_)
        return bn^

    def set_training(mut self, v: Bool):
        self.training = v

    def forward[
        target: StaticString, B: Int, o: MutOrigin
    ](
        mut self,
        inputs: TensorRefs[1, o],
        mut out: Tensor,
        ctx: Optional[DeviceContext] = None,
    ) raises:
        ref in0 = inputs[0]
        var eps = Scalar[DT](Self.EPSILON)
        comptime if target == "cpu":
            out.ensure(B * Self.FLAT_DIM)
            var in_p = in0.data.unsafe_ptr()
            var out_p = out.data.unsafe_ptr()
            var g_p = self.gamma.val.data.unsafe_ptr()
            var b_p = self.beta.val.data.unsafe_ptr()
            var rm_v = TileTensor(self.running_mean.t.data, row_major[Self.C_]())
            var rv_v = TileTensor(self.running_var.t.data, row_major[Self.C_]())
            var inv_n = Scalar[DT](1.0) / Scalar[DT](Float64(B * Self.SPATIAL))
            if self.training:
                self.cache_xhat.ensure(B * Self.FLAT_DIM)
                self.cache_inv_std.ensure(Self.C_)
                var xhat_p = self.cache_xhat.data.unsafe_ptr()
                var inv_v = TileTensor(
                    self.cache_inv_std.data, row_major[Self.C_]()
                )
                var mom = Scalar[DT](Self.MOMENTUM)
                var one_m = Scalar[DT](1.0) - mom
                for c in range(Self.C_):
                    var g = g_p[c]
                    var bt = b_p[c]
                    var mean: Scalar[DT] = 0.0
                    for b in range(B):
                        var base = b * Self.FLAT_DIM + c * Self.SPATIAL
                        for s in range(Self.SPATIAL):
                            mean += in_p[base + s]
                    mean *= inv_n
                    var var_: Scalar[DT] = 0.0
                    for b in range(B):
                        var base = b * Self.FLAT_DIM + c * Self.SPATIAL
                        for s in range(Self.SPATIAL):
                            var d = in_p[base + s] - mean
                            var_ += d * d
                    var_ *= inv_n
                    var inv_std = Scalar[DT](1.0) / sqrt(var_ + eps)
                    inv_v[c] = inv_std
                    for b in range(B):
                        var base = b * Self.FLAT_DIM + c * Self.SPATIAL
                        for s in range(Self.SPATIAL):
                            var xh = (in_p[base + s] - mean) * inv_std
                            xhat_p[base + s] = xh
                            out_p[base + s] = g * xh + bt
                    rm_v[c] = one_m * rm_v[c] + mom * mean
                    rv_v[c] = one_m * rv_v[c] + mom * var_
                self.cache_is_training = True
            else:
                for c in range(Self.C_):
                    var rm = rm_v[c]
                    var rv = rv_v[c]
                    var inv_std = Scalar[DT](1.0) / sqrt(rv + eps)
                    var g = g_p[c]
                    var bt = b_p[c]
                    for b in range(B):
                        var base = b * Self.FLAT_DIM + c * Self.SPATIAL
                        for s in range(Self.SPATIAL):
                            out_p[base + s] = (
                                g * (in_p[base + s] - rm) * inv_std + bt
                            )
        else:
            var c = ctx.value()
            out.ensure_gpu(c, B * Self.FLAT_DIM)
            comptime l2d = Layout.row_major(B, Self.FLAT_DIM)
            comptime lc = Layout.row_major(Self.C_)
            if self.training:
                self.cache_xhat.ensure_gpu(c, B * Self.FLAT_DIM)
                comptime G = B if B < BN2D_RBLOCKS else BN2D_RBLOCKS
                comptime lpr = Layout.row_major(Self.C_ * G)
                # Pass 1: partial Σx, Σx².
                c.enqueue_function[
                    _bn2d_partial_stats_kernel[
                        B,
                        Self.C_,
                        Self.SPATIAL,
                        Self.FLAT_DIM,
                        G,
                    ]
                ](
                    in0.lt["gpu", l2d](),
                    self.bn_psum.lt["gpu", lpr](),
                    self.bn_psumsq.lt["gpu", lpr](),
                    grid_dim=Self.C_ * G,
                    block_dim=BN2D_TPB,
                )
                # Pass 2: mean/var/inv_std + EMA.
                c.enqueue_function[
                    _bn2d_finalize_stats_kernel[
                        B,
                        Self.C_,
                        Self.SPATIAL,
                        G,
                        Self.EPSILON,
                        Self.MOMENTUM,
                    ]
                ](
                    self.bn_psum.lt["gpu", lpr](),
                    self.bn_psumsq.lt["gpu", lpr](),
                    self.running_mean.t.lt["gpu", lc](),
                    self.running_var.t.lt["gpu", lc](),
                    self.cache_mean.lt["gpu", lc](),
                    self.cache_inv_std.lt["gpu", lc](),
                    grid_dim=Self.C_,
                    block_dim=1,
                )
                # Pass 3: normalize + cache x̂.
                c.enqueue_function[
                    _bn2d_normalize_kernel[
                        B,
                        Self.C_,
                        Self.SPATIAL,
                        Self.FLAT_DIM,
                        G,
                    ]
                ](
                    in0.lt["gpu", l2d](),
                    out.lt["gpu", l2d](),
                    self.gamma.val.lt["gpu", lc](),
                    self.beta.val.lt["gpu", lc](),
                    self.cache_mean.lt["gpu", lc](),
                    self.cache_inv_std.lt["gpu", lc](),
                    self.cache_xhat.lt["gpu", l2d](),
                    grid_dim=Self.C_ * G,
                    block_dim=BN2D_TPB,
                )
                self.cache_is_training = True
            else:
                c.enqueue_function[
                    _bn2d_forward_eval_kernel[
                        B,
                        Self.C_,
                        Self.SPATIAL,
                        Self.FLAT_DIM,
                        Self.EPSILON,
                    ]
                ](
                    in0.lt["gpu", l2d](),
                    out.lt["gpu", l2d](),
                    self.gamma.val.lt["gpu", lc](),
                    self.beta.val.lt["gpu", lc](),
                    self.running_mean.t.lt["gpu", lc](),
                    self.running_var.t.lt["gpu", lc](),
                    grid_dim=Self.C_,
                    block_dim=BN2D_TPB,
                )

    def vjp[
        target: StaticString, B: Int, ofi: MutOrigin, ogi: MutOrigin
    ](
        mut self,
        forward_input: TensorRefs[1, ofi],
        mut grad_output: Tensor,
        grad_inputs: TensorRefs[1, ogi],
        ctx: Optional[DeviceContext] = None,
    ) raises:
        if not self.cache_is_training:
            raise Error(
                "BatchNorm2D.vjp: training-mode cache not populated. Call"
                " forward with training=True before vjp."
            )
        ref gin = grad_inputs[0]
        comptime if target == "cpu":
            gin.ensure(B * Self.FLAT_DIM)
            var go_p = grad_output.data.unsafe_ptr()
            var gi_p = gin.data.unsafe_ptr()
            var g_p = self.gamma.val.data.unsafe_ptr()
            var dg_p = self.gamma.grd.data.unsafe_ptr()
            var db_p = self.beta.grd.data.unsafe_ptr()
            var xhat_p = self.cache_xhat.data.unsafe_ptr()
            var inv_v = TileTensor(
                self.cache_inv_std.data, row_major[Self.C_]()
            )
            var inv_n = Scalar[DT](1.0) / Scalar[DT](Float64(B * Self.SPATIAL))
            for c in range(Self.C_):
                var g = g_p[c]
                var inv_std = inv_v[c]
                var sum_dxhat: Scalar[DT] = 0.0
                var sum_dxhat_xhat: Scalar[DT] = 0.0
                var d_gamma: Scalar[DT] = 0.0
                var d_beta: Scalar[DT] = 0.0
                for b in range(B):
                    var base = b * Self.FLAT_DIM + c * Self.SPATIAL
                    for s in range(Self.SPATIAL):
                        var dy = go_p[base + s]
                        var xh = xhat_p[base + s]
                        var dxhat = dy * g
                        sum_dxhat += dxhat
                        sum_dxhat_xhat += dxhat * xh
                        d_gamma += dy * xh
                        d_beta += dy
                var m1 = sum_dxhat * inv_n
                var m2 = sum_dxhat_xhat * inv_n
                for b in range(B):
                    var base = b * Self.FLAT_DIM + c * Self.SPATIAL
                    for s in range(Self.SPATIAL):
                        var dy = go_p[base + s]
                        var xh = xhat_p[base + s]
                        var dxhat = dy * g
                        gi_p[base + s] = inv_std * (dxhat - m1 - xh * m2)
                dg_p[c] += d_gamma
                db_p[c] += d_beta
        else:
            var c = ctx.value()
            gin.ensure_gpu(c, B * Self.FLAT_DIM)
            comptime l2d = Layout.row_major(B, Self.FLAT_DIM)
            comptime lc = Layout.row_major(Self.C_)
            comptime G = B if B < BN2D_RBLOCKS else BN2D_RBLOCKS
            comptime lpr = Layout.row_major(Self.C_ * G)
            # Pass 1: partial Σdx̂, Σdx̂·x̂, Σdγ, Σdβ (reuse psum/psumsq).
            c.enqueue_function[
                _bn2d_bwd_partial_kernel[
                    B,
                    Self.C_,
                    Self.SPATIAL,
                    Self.FLAT_DIM,
                    G,
                ]
            ](
                grad_output.lt["gpu", l2d](),
                self.gamma.val.lt["gpu", lc](),
                self.cache_xhat.lt["gpu", l2d](),
                self.bn_psum.lt["gpu", lpr](),
                self.bn_psumsq.lt["gpu", lpr](),
                self.bn_pdg.lt["gpu", lpr](),
                self.bn_pdb.lt["gpu", lpr](),
                grid_dim=Self.C_ * G,
                block_dim=BN2D_TPB,
            )
            # Pass 2: m1, m2 + accumulate grad_gamma/beta (mode=all).
            c.enqueue_function[
                _bn2d_bwd_finalize_kernel[
                    B,
                    Self.C_,
                    Self.SPATIAL,
                    G,
                    "all",
                ]
            ](
                self.bn_psum.lt["gpu", lpr](),
                self.bn_psumsq.lt["gpu", lpr](),
                self.bn_pdg.lt["gpu", lpr](),
                self.bn_pdb.lt["gpu", lpr](),
                self.bn_m1.lt["gpu", lc](),
                self.bn_m2.lt["gpu", lc](),
                self.gamma.grd.lt["gpu", lc](),
                self.beta.grd.lt["gpu", lc](),
                grid_dim=Self.C_,
                block_dim=1,
            )
            # Pass 3: grad_input scatter.
            c.enqueue_function[
                _bn2d_bwd_scatter_kernel[
                    B,
                    Self.C_,
                    Self.SPATIAL,
                    Self.FLAT_DIM,
                    G,
                ]
            ](
                grad_output.lt["gpu", l2d](),
                self.gamma.val.lt["gpu", lc](),
                self.cache_xhat.lt["gpu", l2d](),
                self.cache_inv_std.lt["gpu", lc](),
                self.bn_m1.lt["gpu", lc](),
                self.bn_m2.lt["gpu", lc](),
                gin.lt["gpu", l2d](),
                grid_dim=Self.C_ * G,
                block_dim=BN2D_TPB,
            )

    # for_each_param / zero_grad inherit the Module reflection defaults
    # (core/walkers.mojo auto-discovers the Param fields).
