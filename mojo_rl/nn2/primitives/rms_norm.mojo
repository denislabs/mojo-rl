"""RMSNorm[DIM] — root-mean-square layer normalization (no mean, no β).

Matches the DreamerV3 reference `embodied/jax/nets.py:Norm` with
`impl='rms'` (the default `norm: rms` across RSSM / Encoder / Decoder):

    mean2 = mean_d(x²)                      # per row, last axis
    y     = x · (rsqrt(mean2 + eps) · γ)    # γ per-feature, eps = 1e-4

Structurally a LayerNorm with the mean subtraction and the β shift
removed. One Param (`gamma`, decay=False, init 1).

Cache (leaf-owned, output-style): `cache_norm` = x·inv_rms  [BATCH, DIM]
and `cache_inv_rms` [BATCH]. Backward:

    R          = Σ_d go[b,d]·γ[d]·n[b,d]            (per row)
    grad_x[b,e]= inv_rms[b]·( go[b,e]·γ[e] − n[b,e]·R/DIM )
    grad_γ[d] += Σ_b go[b,d]·n[b,d]

eps = 1e-4 (reference `Norm.eps` default for the plain `rms` impl).
Stats run in DT (force_fp32-style) — bf16 RMS is unstable.
"""

from std.math import sqrt
from std.gpu import thread_idx, block_idx
from std.gpu.primitives import block
from std.gpu.host import DeviceContext, DeviceBuffer
from std.gpu.memory import AddressSpace
from layout import Layout, LayoutTensor, TileTensor, row_major

from ..constants import DT, CPU_SIMD_W
from ..core import (
    Initializer,
    AMPPolicy,
    NoAMP,
    Param,
    Cache,
    ParamVisitor,
    for_each_param_auto,
    zero_grad_auto,
)
from ..core.module import Module, typed_view, typed_view_mut, mptr
from ..core.tensor_pack import TensorPack
from ..core.target_storage import (
    require_ctx,
    TargetStorage,
    assert_tag_for,
)


comptime RMS_EPS: Scalar[DT] = 1e-4
comptime RMS_TPB: Int = 128


# ──────────────────────────────────────────────────────────────────────
# GPU kernels.
# ──────────────────────────────────────────────────────────────────────


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

    var inv_dim: Scalar[DT] = 1.0 / Float32(DIM)
    var my_sumsq: Scalar[DT] = 0.0
    var idx = t
    while idx < DIM:
        var x = rebind[Scalar[DT]](input[b, idx])
        my_sumsq += x * x
        idx += RMS_TPB
    var mean2 = (
        block.sum[block_size=RMS_TPB, broadcast=True](val=my_sumsq) * inv_dim
    )
    var inv_rms: Scalar[DT] = 1.0 / sqrt(mean2 + RMS_EPS)
    if t == 0:
        cache_inv_rms[b] = inv_rms

    idx = t
    while idx < DIM:
        var x = rebind[Scalar[DT]](input[b, idx])
        var n = x * inv_rms
        cache_norm[b, idx] = n
        output[b, idx] = n * rebind[Scalar[DT]](gamma[idx])
        idx += RMS_TPB


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

    var inv_dim: Scalar[DT] = 1.0 / Float32(DIM)
    var inv_rms = rebind[Scalar[DT]](cache_inv_rms[b])

    var my_r: Scalar[DT] = 0.0
    var idx = t
    while idx < DIM:
        var go = rebind[Scalar[DT]](grad_output[b, idx])
        var gm = rebind[Scalar[DT]](gamma[idx])
        var n = rebind[Scalar[DT]](cache_norm[b, idx])
        my_r += go * gm * n
        idx += RMS_TPB
    var R = block.sum[block_size=RMS_TPB, broadcast=True](val=my_r)

    idx = t
    while idx < DIM:
        var go = rebind[Scalar[DT]](grad_output[b, idx])
        var gm = rebind[Scalar[DT]](gamma[idx])
        var n = rebind[Scalar[DT]](cache_norm[b, idx])
        grad_input[b, idx] = inv_rms * (go * gm - n * R * inv_dim)
        idx += RMS_TPB


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


# ──────────────────────────────────────────────────────────────────────
# RMSNorm.
# ──────────────────────────────────────────────────────────────────────


struct RMSNorm[DIM: Int](Module):
    comptime ARITY: Int = 1
    comptime IN_DIMS = InlineArray[Int, 1](fill=Self.DIM)
    comptime OUT_DIM = Self.DIM

    @staticmethod
    def display_label() -> String:
        return String("RMSNorm")

    var gamma: Param["gamma", False, Self.DIM]

    var cache_norm: Cache["cache_norm"]
    var cache_inv_rms: Cache["cache_inv_rms"]

    var ts: TargetStorage

    def __init__(out self):
        self.gamma = Param["gamma", False, Self.DIM]()
        self.cache_norm = Cache["cache_norm"]()
        self.cache_inv_rms = Cache["cache_inv_rms"]()
        self.ts = TargetStorage.make_uninit()

    @staticmethod
    def make[
        target: StaticString, INIT: Initializer
    ](
        ctx: Optional[DeviceContext] = None,
    ) raises -> Self:
        """Unified CPU/GPU factory. `ctx=None` on CPU; required on GPU.
        γ initialised to 1 (reference scale init); INIT accepted for
        trait conformance but ignored."""
        comptime assert target == "cpu" or target == "gpu", (
            "RMSNorm: target must be 'cpu' or 'gpu'"
        )
        var rn = Self()
        comptime if target == "cpu":
            rn.gamma = Param["gamma", False, Self.DIM].make_cpu()
            var g_ptr = rn.gamma.value_unsafe_ptr_cpu()
            for k in range(Self.DIM):
                g_ptr[k] = Scalar[DT](1.0)
            rn.ts = TargetStorage.make_cpu()
        else:
            var ctx_v = require_ctx["RMSNorm.make[target='gpu']"](ctx)
            rn.gamma = Param["gamma", False, Self.DIM].make_gpu(ctx_v)
            rn.gamma.val.dev.value().enqueue_fill(1.0)
            rn.ts = TargetStorage.make_gpu(ctx_v)
        return rn^

    def _ensure_cache_gpu(mut self, batch: Int) raises:
        var ctx = self.ts.ctx.value()
        self.cache_norm.ensure_gpu(ctx, batch * Self.DIM)
        self.cache_inv_rms.ensure_gpu(ctx, batch)

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
        assert_tag_for["RMSNorm", target](self.ts.target_tag)
        var input = inputs.tile[0, BATCH, Self.IN_DIMS[0]]()
        var output_v = typed_view_mut[BATCH, Self.OUT_DIM](output)

        comptime if target == "cpu":
            self.cache_norm.ensure_cpu(BATCH * Self.DIM)
            self.cache_inv_rms.ensure_cpu(BATCH)
            var inv_v = TileTensor(self.cache_inv_rms.cpu, row_major[BATCH]())
            # SIMD-vectorized over the feature axis (C3).
            var in_p  = input.ptr
            var out_p = output_v.ptr
            var g_p   = self.gamma.value_unsafe_ptr_cpu()
            var nm_p  = mptr(self.cache_norm.cpu.unsafe_ptr())
            comptime W = CPU_SIMD_W
            var inv_dim: Scalar[DT] = 1.0 / Float32(Self.DIM)
            for b in range(BATCH):
                var row = b * Self.DIM
                # mean2 = Σ x² / DIM
                var acc = SIMD[DT, W](0)
                var d = 0
                while d + W <= Self.DIM:
                    var x = in_p.load[width=W](row + d)
                    acc += x * x
                    d += W
                var sumsq = acc.reduce_add()
                while d < Self.DIM:
                    var x = in_p[row + d]
                    sumsq += x * x
                    d += 1
                var mean2 = sumsq * inv_dim
                var inv_rms = Scalar[DT](1.0) / sqrt(mean2 + RMS_EPS)
                inv_v[b] = inv_rms
                # n = x·inv_rms ; out = n·γ
                var irv = SIMD[DT, W](inv_rms)
                d = 0
                while d + W <= Self.DIM:
                    var n = in_p.load[width=W](row + d) * irv
                    nm_p.store(row + d, n)
                    out_p.store(row + d, n * g_p.load[width=W](d))
                    d += W
                while d < Self.DIM:
                    var n = in_p[row + d] * inv_rms
                    nm_p[row + d] = n
                    out_p[row + d] = n * g_p[d]
                    d += 1
        else:
            self._ensure_cache_gpu(BATCH)
            comptime layout_2d = Layout.row_major(BATCH, Self.DIM)
            comptime layout_b = Layout.row_major(BATCH)
            comptime layout_d = Layout.row_major(Self.DIM)
            var in_p_w = input.ptr
            var out_p_w = output_v.ptr
            var in_lt = LayoutTensor[DT, layout_2d, MutAnyOrigin](in_p_w)
            var out_lt = LayoutTensor[DT, layout_2d, MutAnyOrigin](out_p_w)
            var g_lt = LayoutTensor[DT, layout_d, MutAnyOrigin](
                self.gamma.val.dev.value()
            )
            var nm_lt = LayoutTensor[DT, layout_2d, MutAnyOrigin](
                self.cache_norm.dev.value()
            )
            var ir_lt = LayoutTensor[DT, layout_b, MutAnyOrigin](
                self.cache_inv_rms.dev.value()
            )
            comptime kernel = _rms_norm_forward_kernel[BATCH, Self.DIM]
            self.ts.ctx.value().enqueue_function[kernel](
                in_lt, out_lt, g_lt, nm_lt, ir_lt,
                grid_dim=BATCH,
                block_dim=RMS_TPB,
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
        assert_tag_for["RMSNorm", target](self.ts.target_tag)
        var grad_output_v = typed_view[BATCH, Self.OUT_DIM](grad_output)
        var grad_input_v = grad_inputs.tile[0, BATCH, Self.IN_DIMS[0]]()

        comptime if target == "cpu":
            var inv_v = TileTensor(self.cache_inv_rms.cpu, row_major[BATCH]())
            # SIMD-vectorized over the feature axis (C3).
            var go_p = grad_output_v.ptr
            var gi_p = grad_input_v.ptr
            var g_p  = self.gamma.value_unsafe_ptr_cpu()
            var gg_p = self.gamma.grad_unsafe_ptr_cpu()
            var nm_p = mptr(self.cache_norm.cpu.unsafe_ptr())
            comptime W = CPU_SIMD_W
            var inv_dim: Scalar[DT] = 1.0 / Float32(Self.DIM)
            for b in range(BATCH):
                var row = b * Self.DIM
                var inv_rms = inv_v[b]
                # R = Σ go·γ·n
                var acc = SIMD[DT, W](0)
                var d = 0
                while d + W <= Self.DIM:
                    acc += (
                        go_p.load[width=W](row + d)
                        * g_p.load[width=W](d)
                        * nm_p.load[width=W](row + d)
                    )
                    d += W
                var R = acc.reduce_add()
                while d < Self.DIM:
                    R += go_p[row + d] * g_p[d] * nm_p[row + d]
                    d += 1
                # grad_input = inv_rms·(go·γ - n·R/DIM)
                var irv = SIMD[DT, W](inv_rms)
                var rscaled = SIMD[DT, W](R * inv_dim)
                d = 0
                while d + W <= Self.DIM:
                    var go = go_p.load[width=W](row + d)
                    var n  = nm_p.load[width=W](row + d)
                    gi_p.store(
                        row + d,
                        irv * (go * g_p.load[width=W](d) - n * rscaled),
                    )
                    d += W
                while d < Self.DIM:
                    var go = go_p[row + d]
                    gi_p[row + d] = inv_rms * (
                        go * g_p[d] - nm_p[row + d] * R * inv_dim
                    )
                    d += 1
                comptime if mode == "all":
                    # dγ += go·n  (accumulated across the batch)
                    d = 0
                    while d + W <= Self.DIM:
                        gg_p.store(
                            d,
                            gg_p.load[width=W](d)
                            + go_p.load[width=W](row + d)
                            * nm_p.load[width=W](row + d),
                        )
                        d += W
                    while d < Self.DIM:
                        gg_p[d] = gg_p[d] + go_p[row + d] * nm_p[row + d]
                        d += 1
        else:
            var ctx = self.ts.ctx.value()
            comptime layout_2d = Layout.row_major(BATCH, Self.DIM)
            comptime layout_b = Layout.row_major(BATCH)
            comptime layout_d = Layout.row_major(Self.DIM)
            var go_p_w = grad_output_v.ptr
            var gi_p_w = grad_input_v.ptr
            var go_lt = LayoutTensor[DT, layout_2d, MutAnyOrigin](go_p_w)
            var gi_lt = LayoutTensor[DT, layout_2d, MutAnyOrigin](gi_p_w)
            var g_lt = LayoutTensor[DT, layout_d, MutAnyOrigin](
                self.gamma.val.dev.value()
            )
            var nm_lt = LayoutTensor[DT, layout_2d, MutAnyOrigin](
                self.cache_norm.dev.value()
            )
            var ir_lt = LayoutTensor[DT, layout_b, MutAnyOrigin](
                self.cache_inv_rms.dev.value()
            )
            comptime dx_kernel = _rms_norm_backward_dx_kernel[BATCH, Self.DIM]
            ctx.enqueue_function[dx_kernel](
                go_lt, g_lt, nm_lt, ir_lt, gi_lt,
                grid_dim=BATCH,
                block_dim=RMS_TPB,
            )
            comptime if mode == "all":
                var gg_lt = LayoutTensor[DT, layout_d, MutAnyOrigin](
                    self.gamma.grd.dev.value()
                )
                comptime dg_kernel = _rms_norm_backward_dgamma_kernel[
                    BATCH, Self.DIM
                ]
                ctx.enqueue_function[dg_kernel](
                    go_lt, nm_lt, gg_lt,
                    grid_dim=Self.DIM,
                    block_dim=RMS_TPB,
                )

    # ----- Walkers ---------------------------------------------------------

    def for_each_param[
        target: StaticString,
        V: ParamVisitor,
    ](mut self, prefix: String, mut visitor: V) raises:
        assert_tag_for["RMSNorm", target](self.ts.target_tag)
        for_each_param_auto[Self, V, target](self, prefix, visitor)

    def zero_grad[target: StaticString](mut self) raises:
        assert_tag_for["RMSNorm", target](self.ts.target_tag)
        zero_grad_auto[Self, target](self)
