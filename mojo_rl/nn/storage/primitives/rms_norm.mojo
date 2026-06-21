"""RMSNorm[DIM] — root-mean-square normalization (storage surface).

Transformed from legacy `nn.primitives.RMSNorm` (surface-only change). No mean
subtraction, no β — one `Param` (γ, decay=False, init 1). The x·inv_rms + inv_rms
cache is leaf-owned. CPU SIMD feature-axis loops + 3 GPU kernels (forward /
backward-dx 1-block-per-row / backward-dgamma 1-block-per-col) carried verbatim.

Backward (cache):  R = Σ_d go·γ·n ; dx = inv_rms·(go·γ - n·R/DIM) ; dγ += Σ_b go·n
"""

from std.math import sqrt
from std.gpu import thread_idx, block_idx
from std.gpu.primitives import block
from std.gpu.host import DeviceContext
from std.utils import Index
from std.utils.numerics import get_accum_type
from layout import Layout, LayoutTensor, TileTensor, row_major

from mojo_rl.nn.constants import DT, CPU_SIMD_W
from ..core.tensor import Tensor
from ..core.tensor_refs import TensorRefs
from ..core.module import Module
from ..core.param import Param, ParamVisitor
from ..core.initializer import Initializer
from ..core.amp import AMPPolicy, NoAMP


comptime RMS_EPS: Scalar[DT] = 1e-4
comptime RMS_TPB: Int = 128
# Reductions/normalization run in the accumulation dtype (f32 for bf16 inputs;
# identity for DT=f32). VEC-wide vectorized global loads when DIM % VEC == 0.
comptime RMS_ACC = get_accum_type[DT]()


# ── GPU kernels (vectorized + accum_type; block-per-row, threads cooperate
#    over the feature axis with VEC-wide strided loads) ──────────────────
def _rms_norm_forward_kernel[
    BATCH: Int,
    DIM: Int,
](
    input: LayoutTensor[DT, Layout.row_major(BATCH, DIM), MutAnyOrigin],
    output: LayoutTensor[DT, Layout.row_major(BATCH, DIM), MutAnyOrigin],
    gamma: LayoutTensor[DT, Layout.row_major(DIM), MutAnyOrigin],
    cache_norm: LayoutTensor[DT, Layout.row_major(BATCH, DIM), MutAnyOrigin],
    cache_inv_rms: LayoutTensor[DT, Layout.row_major(BATCH), MutAnyOrigin],
):
    var b = Int(block_idx.x)
    var t = Int(thread_idx.x)
    if b >= BATCH:
        return
    comptime VEC = 4 if DIM % 4 == 0 else 1
    var inv_dim = Scalar[RMS_ACC](1.0) / Scalar[RMS_ACC](DIM)
    var my_sumsq = Scalar[RMS_ACC](0)
    var idx = t * VEC
    while idx < DIM:
        var x = input.load[width=VEC](b, idx).cast[RMS_ACC]()
        my_sumsq += (x * x).reduce_add()
        idx += RMS_TPB * VEC
    var mean2 = (
        block.sum[block_size=RMS_TPB, broadcast=True](val=my_sumsq) * inv_dim
    )
    var inv_rms = Scalar[RMS_ACC](1.0) / sqrt(mean2 + RMS_EPS.cast[RMS_ACC]())
    if t == 0:
        cache_inv_rms[b] = inv_rms.cast[DT]()
    idx = t * VEC
    while idx < DIM:
        var x = input.load[width=VEC](b, idx).cast[RMS_ACC]()
        var n = x * inv_rms
        cache_norm.store[width=VEC](b, idx, n.cast[DT]())
        var g = gamma.load[width=VEC](Index(idx)).cast[RMS_ACC]()
        output.store[width=VEC](b, idx, (n * g).cast[DT]())
        idx += RMS_TPB * VEC


def _rms_norm_backward_dx_kernel[
    BATCH: Int,
    DIM: Int,
](
    grad_output: LayoutTensor[DT, Layout.row_major(BATCH, DIM), MutAnyOrigin],
    gamma: LayoutTensor[DT, Layout.row_major(DIM), MutAnyOrigin],
    cache_norm: LayoutTensor[DT, Layout.row_major(BATCH, DIM), MutAnyOrigin],
    cache_inv_rms: LayoutTensor[DT, Layout.row_major(BATCH), MutAnyOrigin],
    grad_input: LayoutTensor[DT, Layout.row_major(BATCH, DIM), MutAnyOrigin],
):
    var b = Int(block_idx.x)
    var t = Int(thread_idx.x)
    if b >= BATCH:
        return
    comptime VEC = 4 if DIM % 4 == 0 else 1
    var inv_dim = Scalar[RMS_ACC](1.0) / Scalar[RMS_ACC](DIM)
    var inv_rms = rebind[Scalar[DT]](cache_inv_rms[b]).cast[RMS_ACC]()
    var my_r = Scalar[RMS_ACC](0)
    var idx = t * VEC
    while idx < DIM:
        var go = grad_output.load[width=VEC](b, idx).cast[RMS_ACC]()
        var gm = gamma.load[width=VEC](Index(idx)).cast[RMS_ACC]()
        var n = cache_norm.load[width=VEC](b, idx).cast[RMS_ACC]()
        my_r += (go * gm * n).reduce_add()
        idx += RMS_TPB * VEC
    var R = block.sum[block_size=RMS_TPB, broadcast=True](val=my_r)
    idx = t * VEC
    while idx < DIM:
        var go = grad_output.load[width=VEC](b, idx).cast[RMS_ACC]()
        var gm = gamma.load[width=VEC](Index(idx)).cast[RMS_ACC]()
        var n = cache_norm.load[width=VEC](b, idx).cast[RMS_ACC]()
        grad_input.store[width=VEC](
            b, idx, (inv_rms * (go * gm - n * R * inv_dim)).cast[DT]()
        )
        idx += RMS_TPB * VEC


def _rms_norm_backward_dgamma_kernel[
    BATCH: Int,
    DIM: Int,
](
    grad_output: LayoutTensor[DT, Layout.row_major(BATCH, DIM), MutAnyOrigin],
    cache_norm: LayoutTensor[DT, Layout.row_major(BATCH, DIM), MutAnyOrigin],
    grad_gamma: LayoutTensor[DT, Layout.row_major(DIM), MutAnyOrigin],
):
    var col = Int(block_idx.x)
    var t = Int(thread_idx.x)
    if col >= DIM:
        return
    var my_dg: Scalar[DT] = 0.0
    var bi = t
    while bi < BATCH:
        var go = rebind[Scalar[DT]](grad_output[bi, col])
        var n = rebind[Scalar[DT]](cache_norm[bi, col])
        my_dg += go * n
        bi += RMS_TPB
    var total_dg = block.sum[block_size=RMS_TPB, broadcast=False](val=my_dg)
    if t == 0:
        grad_gamma[col] = rebind[Scalar[DT]](grad_gamma[col]) + total_dg[0]


struct RMSNorm[DIM_: Int](Module):
    comptime ARITY = 1
    comptime IN_DIMS = InlineArray[Int, 1](fill=Self.DIM_)
    comptime OUT_DIM = Self.DIM_

    var gamma: Param["gamma", False, Self.DIM_]
    var cache_norm: Tensor  # [BATCH, DIM]
    var cache_inv_rms: Tensor  # [BATCH]

    def __init__(out self):
        self.gamma = Param["gamma", False, Self.DIM_]()
        self.cache_norm = Tensor()
        self.cache_inv_rms = Tensor()

    @staticmethod
    def make[
        target: StaticString, INIT: Initializer
    ](ctx: Optional[DeviceContext] = None) raises -> Self:
        var rn = Self()
        rn.gamma = Param["gamma", False, Self.DIM_].make[target](ctx)
        for k in range(Self.DIM_):
            rn.gamma.val.data[k] = Scalar[DT](1.0)
        comptime if target != "cpu":
            rn.gamma.val.upload(ctx.value())
        return rn^

    def forward[
        target: StaticString, B: Int, o: MutOrigin, POLICY: AMPPolicy = NoAMP
    ](
        mut self,
        inputs: TensorRefs[1, o],
        mut out: Tensor,
        ctx: Optional[DeviceContext] = None,
    ) raises:
        ref in0 = inputs[0]
        comptime if target == "cpu":
            out.ensure(B * Self.DIM_)
            self.cache_norm.ensure(B * Self.DIM_)
            self.cache_inv_rms.ensure(B)
            var inv_v = TileTensor(self.cache_inv_rms.data, row_major[B]())
            var in_p = in0.data.unsafe_ptr()
            var out_p = out.data.unsafe_ptr()
            var g_p = self.gamma.val.data.unsafe_ptr()
            var nm_p = self.cache_norm.data.unsafe_ptr()
            comptime W = CPU_SIMD_W
            var inv_dim: Scalar[DT] = 1.0 / Float32(Self.DIM_)
            for b in range(B):
                var row = b * Self.DIM_
                var acc = SIMD[DT, W](0)
                var d = 0
                while d + W <= Self.DIM_:
                    var x = in_p.load[width=W](row + d)
                    acc += x * x
                    d += W
                var sumsq = acc.reduce_add()
                while d < Self.DIM_:
                    var x = in_p[row + d]
                    sumsq += x * x
                    d += 1
                var mean2 = sumsq * inv_dim
                var inv_rms = Scalar[DT](1.0) / sqrt(mean2 + RMS_EPS)
                inv_v[b] = inv_rms
                var irv = SIMD[DT, W](inv_rms)
                d = 0
                while d + W <= Self.DIM_:
                    var n = in_p.load[width=W](row + d) * irv
                    nm_p.store(row + d, n)
                    out_p.store(row + d, n * g_p.load[width=W](d))
                    d += W
                while d < Self.DIM_:
                    var n = in_p[row + d] * inv_rms
                    nm_p[row + d] = n
                    out_p[row + d] = n * g_p[d]
                    d += 1
        else:
            var c = ctx.value()
            out.ensure_gpu(c, B * Self.DIM_)
            self.cache_norm.ensure_gpu(c, B * Self.DIM_)
            self.cache_inv_rms.ensure_gpu(c, B)
            comptime l2d = Layout.row_major(B, Self.DIM_)
            comptime lb = Layout.row_major(B)
            comptime ld = Layout.row_major(Self.DIM_)
            c.enqueue_function[_rms_norm_forward_kernel[B, Self.DIM_]](
                in0.lt["gpu", l2d](),
                out.lt["gpu", l2d](),
                self.gamma.val.lt["gpu", ld](),
                self.cache_norm.lt["gpu", l2d](),
                self.cache_inv_rms.lt["gpu", lb](),
                grid_dim=B,
                block_dim=RMS_TPB,
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
        ref gin = grad_inputs[0]
        comptime if target == "cpu":
            gin.ensure(B * Self.DIM_)
            var inv_v = TileTensor(self.cache_inv_rms.data, row_major[B]())
            var go_p = grad_output.data.unsafe_ptr()
            var gi_p = gin.data.unsafe_ptr()
            var g_p = self.gamma.val.data.unsafe_ptr()
            var gg_p = self.gamma.grd.data.unsafe_ptr()
            var nm_p = self.cache_norm.data.unsafe_ptr()
            comptime W = CPU_SIMD_W
            var inv_dim: Scalar[DT] = 1.0 / Float32(Self.DIM_)
            for b in range(B):
                var row = b * Self.DIM_
                var inv_rms = inv_v[b]
                var acc = SIMD[DT, W](0)
                var d = 0
                while d + W <= Self.DIM_:
                    acc += (
                        go_p.load[width=W](row + d)
                        * g_p.load[width=W](d)
                        * nm_p.load[width=W](row + d)
                    )
                    d += W
                var R = acc.reduce_add()
                while d < Self.DIM_:
                    R += go_p[row + d] * g_p[d] * nm_p[row + d]
                    d += 1
                var irv = SIMD[DT, W](inv_rms)
                var rscaled = SIMD[DT, W](R * inv_dim)
                d = 0
                while d + W <= Self.DIM_:
                    var go = go_p.load[width=W](row + d)
                    var n = nm_p.load[width=W](row + d)
                    gi_p.store(
                        row + d, irv * (go * g_p.load[width=W](d) - n * rscaled)
                    )
                    d += W
                while d < Self.DIM_:
                    var go = go_p[row + d]
                    gi_p[row + d] = inv_rms * (
                        go * g_p[d] - nm_p[row + d] * R * inv_dim
                    )
                    d += 1
                # dγ += go·n  (accumulated across the batch)
                d = 0
                while d + W <= Self.DIM_:
                    gg_p.store(
                        d,
                        gg_p.load[width=W](d)
                        + go_p.load[width=W](row + d)
                        * nm_p.load[width=W](row + d),
                    )
                    d += W
                while d < Self.DIM_:
                    gg_p[d] = gg_p[d] + go_p[row + d] * nm_p[row + d]
                    d += 1
        else:
            var c = ctx.value()
            gin.ensure_gpu(c, B * Self.DIM_)
            comptime l2d = Layout.row_major(B, Self.DIM_)
            comptime lb = Layout.row_major(B)
            comptime ld = Layout.row_major(Self.DIM_)
            c.enqueue_function[_rms_norm_backward_dx_kernel[B, Self.DIM_]](
                grad_output.lt["gpu", l2d](),
                self.gamma.val.lt["gpu", ld](),
                self.cache_norm.lt["gpu", l2d](),
                self.cache_inv_rms.lt["gpu", lb](),
                gin.lt["gpu", l2d](),
                grid_dim=B,
                block_dim=RMS_TPB,
            )
            c.enqueue_function[_rms_norm_backward_dgamma_kernel[B, Self.DIM_]](
                grad_output.lt["gpu", l2d](),
                self.cache_norm.lt["gpu", l2d](),
                self.gamma.grd.lt["gpu", ld](),
                grid_dim=Self.DIM_,
                block_dim=RMS_TPB,
            )

    # for_each_param / zero_grad inherit the Module reflection defaults
    # (core/walkers.mojo auto-discovers the Param fields).
