"""LayerNorm[DIM].

State:

  * `ts: TargetStorage` carries `target_tag` + (optional) `ctx`.
  * `gamma: Param["gamma", False, DIM]` + `beta: Param["beta", False, DIM]`
    — weight-decay-exempt (γ/β shouldn't decay).
  * `for_each_param` / `zero_grad` delegate to the reflection-walked
    `_auto` helpers.

Cache stays leaf-owned (output-caching, no input aliasing): `cache_xhat`
[BATCH, DIM] and `cache_inv_std` [BATCH] live on the leaf. Backward
order is grad_input → dgamma/dbeta — safe because cache is in its own
buffer, not the input slab.

AMP: `force_fp32_input = True` — LayerNorm ignores POLICY and always
runs in DT. Stats are numerically unstable in bf16.

Init: γ=1, β=0 (universal LayerNorm init); the supplied `INIT` is
accepted for trait conformance but ignored — `Param.make_*` zero-fills,
and we post-fill γ to 1.
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


comptime LN_EPS: Scalar[DT] = 1e-5
comptime LN_TPB: Int = 128


# ──────────────────────────────────────────────────────────────────────
# GPU kernels — copied from v1 verbatim. Kept here so LayerNorm is
# self-contained; v1 can be deleted after Phase F.
# ──────────────────────────────────────────────────────────────────────


def _layer_norm_forward_kernel[
    BATCH: Int,
    DIM: Int,
](
    input: LayoutTensor[DT, Layout.row_major(BATCH, DIM), MutAnyOrigin],
    output: LayoutTensor[DT, Layout.row_major(BATCH, DIM), MutAnyOrigin],
    gamma: LayoutTensor[DT, Layout.row_major(DIM), MutAnyOrigin],
    beta: LayoutTensor[DT, Layout.row_major(DIM), MutAnyOrigin],
    cache_xhat: LayoutTensor[DT, Layout.row_major(BATCH, DIM), MutAnyOrigin],
    cache_inv_std: LayoutTensor[DT, Layout.row_major(BATCH), MutAnyOrigin],
):
    var b = Int(block_idx.x)
    var t = Int(thread_idx.x)

    if b >= BATCH:
        return

    var inv_dim: Scalar[DT] = 1.0 / Float32(DIM)

    var my_sum: Scalar[DT] = 0.0
    var idx = t
    while idx < DIM:
        my_sum += rebind[Scalar[DT]](input[b, idx])
        idx += LN_TPB
    var mean_val = (
        block.sum[block_size=LN_TPB, broadcast=True](val=my_sum) * inv_dim
    )

    var my_var: Scalar[DT] = 0.0
    idx = t
    while idx < DIM:
        var diff = rebind[Scalar[DT]](input[b, idx]) - mean_val
        my_var += diff * diff
        idx += LN_TPB
    var var_val = (
        block.sum[block_size=LN_TPB, broadcast=True](val=my_var) * inv_dim
    )

    var inv_std: Scalar[DT] = 1.0 / sqrt(var_val + LN_EPS)
    if t == 0:
        cache_inv_std[b] = inv_std

    idx = t
    while idx < DIM:
        var x = rebind[Scalar[DT]](input[b, idx])
        var x_hat = (x - mean_val) * inv_std
        cache_xhat[b, idx] = x_hat
        var g_d = rebind[Scalar[DT]](gamma[idx])
        var bt_d = rebind[Scalar[DT]](beta[idx])
        output[b, idx] = g_d * x_hat + bt_d
        idx += LN_TPB


def _layer_norm_backward_dx_kernel[
    BATCH: Int,
    DIM: Int,
](
    grad_output: LayoutTensor[DT, Layout.row_major(BATCH, DIM), MutAnyOrigin],
    gamma: LayoutTensor[DT, Layout.row_major(DIM), MutAnyOrigin],
    cache_xhat: LayoutTensor[DT, Layout.row_major(BATCH, DIM), MutAnyOrigin],
    cache_inv_std: LayoutTensor[DT, Layout.row_major(BATCH), MutAnyOrigin],
    grad_input: LayoutTensor[DT, Layout.row_major(BATCH, DIM), MutAnyOrigin],
):
    var b = Int(block_idx.x)
    var t = Int(thread_idx.x)

    if b >= BATCH:
        return

    var inv_dim: Scalar[DT] = 1.0 / Float32(DIM)
    var inv_std = rebind[Scalar[DT]](cache_inv_std[b])

    var my_g: Scalar[DT] = 0.0
    var my_g_xhat: Scalar[DT] = 0.0
    var idx = t
    while idx < DIM:
        var go = rebind[Scalar[DT]](grad_output[b, idx])
        var gm = rebind[Scalar[DT]](gamma[idx])
        var xh = rebind[Scalar[DT]](cache_xhat[b, idx])
        var g  = go * gm
        my_g      += g
        my_g_xhat += g * xh
        idx += LN_TPB
    var mean_g = (
        block.sum[block_size=LN_TPB, broadcast=True](val=my_g) * inv_dim
    )
    var mean_g_xhat = (
        block.sum[block_size=LN_TPB, broadcast=True](val=my_g_xhat) * inv_dim
    )

    idx = t
    while idx < DIM:
        var go = rebind[Scalar[DT]](grad_output[b, idx])
        var gm = rebind[Scalar[DT]](gamma[idx])
        var xh = rebind[Scalar[DT]](cache_xhat[b, idx])
        var g  = go * gm
        grad_input[b, idx] = inv_std * (g - mean_g - xh * mean_g_xhat)
        idx += LN_TPB


def _layer_norm_backward_dparams_kernel[
    BATCH: Int,
    DIM: Int,
](
    grad_output: LayoutTensor[DT, Layout.row_major(BATCH, DIM), MutAnyOrigin],
    cache_xhat: LayoutTensor[DT, Layout.row_major(BATCH, DIM), MutAnyOrigin],
    grad_gamma: LayoutTensor[DT, Layout.row_major(DIM), MutAnyOrigin],
    grad_beta:  LayoutTensor[DT, Layout.row_major(DIM), MutAnyOrigin],
):
    var col = Int(block_idx.x)
    var t = Int(thread_idx.x)

    if col >= DIM:
        return

    var my_dg: Scalar[DT] = 0.0
    var my_db: Scalar[DT] = 0.0
    var bi = t
    while bi < BATCH:
        var go = rebind[Scalar[DT]](grad_output[bi, col])
        var xh = rebind[Scalar[DT]](cache_xhat[bi, col])
        my_dg += go * xh
        my_db += go
        bi += LN_TPB

    var total_dg = block.sum[block_size=LN_TPB, broadcast=False](val=my_dg)
    var total_db = block.sum[block_size=LN_TPB, broadcast=False](val=my_db)
    if t == 0:
        grad_gamma[col] = rebind[Scalar[DT]](grad_gamma[col]) + total_dg[0]
        grad_beta[col]  = rebind[Scalar[DT]](grad_beta[col])  + total_db[0]


# ──────────────────────────────────────────────────────────────────────
# LayerNorm.
# ──────────────────────────────────────────────────────────────────────


struct LayerNorm[DIM: Int](Module):
    comptime ARITY: Int = 1
    comptime IN_DIMS = InlineArray[Int, 1](fill=Self.DIM)
    comptime OUT_DIM = Self.DIM

    # Params (decay=False for both; PyTorch + canonical AdamW recipe).
    var gamma: Param["gamma", False, Self.DIM]
    var beta:  Param["beta",  False, Self.DIM]

    # Cache (leaf-owned, output-caching).
    # S5 dynamic Cache role — lazy-grown at forward (was List + Optional
    # DeviceBuffer ×2 + a shared cache_n_batch capacity Int).
    var cache_xhat: Cache["ln_xhat"]                # [BATCH, DIM]
    var cache_inv_std: Cache["ln_inv_std"]          # [BATCH]

    var ts: TargetStorage

    def __init__(out self):
        self.gamma = Param["gamma", False, Self.DIM]()
        self.beta  = Param["beta",  False, Self.DIM]()
        self.cache_xhat = Cache["ln_xhat"]()
        self.cache_inv_std = Cache["ln_inv_std"]()
        self.ts = TargetStorage.make_uninit()

    @staticmethod
    def make[
        target: StaticString, INIT: Initializer
    ](
        ctx: Optional[DeviceContext] = None,
    ) raises -> Self:
        """Unified CPU/GPU factory. `ctx=None` on CPU; required on GPU."""
        comptime assert target == "cpu" or target == "gpu", (
            "LayerNorm: target must be 'cpu' or 'gpu'"
        )
        var ln = Self()
        comptime if target == "cpu":
            ln.gamma = Param["gamma", False, Self.DIM].make_cpu()
            ln.beta  = Param["beta",  False, Self.DIM].make_cpu()
            # Universal LayerNorm init: γ=1, β=0. Param.make_cpu zero-filled;
            # post-fill gamma to 1.
            var g_ptr = ln.gamma.value_unsafe_ptr_cpu()
            for k in range(Self.DIM):
                g_ptr[k] = Scalar[DT](1.0)
            ln.ts = TargetStorage.make_cpu()
        else:
            var ctx_v = require_ctx["LayerNorm.make[target='gpu']"](ctx)
            ln.gamma = Param["gamma", False, Self.DIM].make_gpu(ctx_v)
            ln.beta  = Param["beta",  False, Self.DIM].make_gpu(ctx_v)
            ln.gamma.val.dev.value().enqueue_fill(1.0)
            ln.beta.val.dev.value().enqueue_fill(0.0)
            # cache is lazy (S5 Cache) — grown at forward via ensure_gpu.
            ln.ts = TargetStorage.make_gpu(ctx_v)
        return ln^

    def _ensure_cache_gpu(mut self, batch: Int) raises:
        var ctx = self.ts.ctx.value()
        self.cache_xhat.ensure_gpu(ctx, batch * Self.DIM)
        self.cache_inv_std.ensure_gpu(ctx, batch)

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
        assert_tag_for["LayerNorm", target](self.ts.target_tag)
        var input = inputs.tile[0, BATCH, Self.IN_DIMS[0]]()
        var output_v = typed_view_mut[BATCH, Self.OUT_DIM](output)

        comptime if target == "cpu":
            self.cache_xhat.ensure_cpu(BATCH * Self.DIM)
            self.cache_inv_std.ensure_cpu(BATCH)
            var inv_v   = TileTensor(
                self.cache_inv_std.cpu, row_major[BATCH](),
            )
            # SIMD-vectorized over the feature axis (C3). Reductions use a
            # width-W accumulator + reduce_add (tree order — within CPU↔GPU
            # parity tolerance, not bit-identical to the old scalar loop).
            var in_p   = input.ptr
            var out_p  = output_v.ptr
            var g_p    = self.gamma.value_unsafe_ptr_cpu()
            var b_p    = self.beta.value_unsafe_ptr_cpu()
            var xh_p   = mptr(self.cache_xhat.cpu.unsafe_ptr())
            comptime W = CPU_SIMD_W
            var inv_dim: Scalar[DT] = 1.0 / Float32(Self.DIM)
            for b in range(BATCH):
                var row = b * Self.DIM
                # mean = Σ x / DIM
                var acc = SIMD[DT, W](0)
                var d = 0
                while d + W <= Self.DIM:
                    acc += in_p.load[width=W](row + d)
                    d += W
                var s = acc.reduce_add()
                while d < Self.DIM:
                    s += in_p[row + d]
                    d += 1
                var mean = s * inv_dim
                var meanv = SIMD[DT, W](mean)
                # var = Σ (x-mean)² / DIM
                var vacc = SIMD[DT, W](0)
                d = 0
                while d + W <= Self.DIM:
                    var diff = in_p.load[width=W](row + d) - meanv
                    vacc += diff * diff
                    d += W
                var sv = vacc.reduce_add()
                while d < Self.DIM:
                    var diff = in_p[row + d] - mean
                    sv += diff * diff
                    d += 1
                var var_v = sv * inv_dim
                var inv_std = Scalar[DT](1.0) / sqrt(var_v + LN_EPS)
                inv_v[b] = inv_std
                # x_hat = (x-mean)·inv_std ; out = γ·x_hat + β
                var isv = SIMD[DT, W](inv_std)
                d = 0
                while d + W <= Self.DIM:
                    var xh = (in_p.load[width=W](row + d) - meanv) * isv
                    xh_p.store(row + d, xh)
                    out_p.store(
                        row + d,
                        g_p.load[width=W](d) * xh + b_p.load[width=W](d),
                    )
                    d += W
                while d < Self.DIM:
                    var xh = (in_p[row + d] - mean) * inv_std
                    xh_p[row + d] = xh
                    out_p[row + d] = g_p[d] * xh + b_p[d]
                    d += 1
        else:
            self._ensure_cache_gpu(BATCH)
            comptime layout_2d = Layout.row_major(BATCH, Self.DIM)
            comptime layout_b  = Layout.row_major(BATCH)
            comptime layout_d  = Layout.row_major(Self.DIM)
            var in_p_w  = input.ptr
            var out_p_w = output_v.ptr
            var in_lt  = LayoutTensor[DT, layout_2d, MutAnyOrigin](in_p_w)
            var out_lt = LayoutTensor[DT, layout_2d, MutAnyOrigin](out_p_w)
            var g_lt   = LayoutTensor[DT, layout_d, MutAnyOrigin](
                self.gamma.val.dev.value()
            )
            var b_lt   = LayoutTensor[DT, layout_d, MutAnyOrigin](
                self.beta.val.dev.value()
            )
            var xh_lt  = LayoutTensor[DT, layout_2d, MutAnyOrigin](
                self.cache_xhat.dev.value()
            )
            var is_lt  = LayoutTensor[DT, layout_b, MutAnyOrigin](
                self.cache_inv_std.dev.value()
            )
            comptime kernel = _layer_norm_forward_kernel[BATCH, Self.DIM]
            self.ts.ctx.value().enqueue_function[kernel](
                in_lt, out_lt, g_lt, b_lt, xh_lt, is_lt,
                grid_dim=BATCH,
                block_dim=LN_TPB,
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
        assert_tag_for["LayerNorm", target](self.ts.target_tag)
        var grad_output_v = typed_view[BATCH, Self.OUT_DIM](grad_output)
        var grad_input_v = grad_inputs.tile[0, BATCH, Self.IN_DIMS[0]]()

        comptime if target == "cpu":
            var inv_v         = TileTensor(
                self.cache_inv_std.cpu, row_major[BATCH](),
            )
            # SIMD-vectorized over the feature axis (C3).
            var go_p = grad_output_v.ptr
            var gi_p = grad_input_v.ptr
            var g_p  = self.gamma.value_unsafe_ptr_cpu()
            var gg_p = self.gamma.grad_unsafe_ptr_cpu()
            var gb_p = self.beta.grad_unsafe_ptr_cpu()
            var xh_p = mptr(self.cache_xhat.cpu.unsafe_ptr())
            comptime W = CPU_SIMD_W
            var inv_dim: Scalar[DT] = 1.0 / Float32(Self.DIM)
            for b in range(BATCH):
                var row = b * Self.DIM
                var inv_std = inv_v[b]
                # mean_g, mean_g_xhat — two fused reductions
                var acc_g  = SIMD[DT, W](0)
                var acc_gx = SIMD[DT, W](0)
                var d = 0
                while d + W <= Self.DIM:
                    var g = go_p.load[width=W](row + d) * g_p.load[width=W](d)
                    acc_g  += g
                    acc_gx += g * xh_p.load[width=W](row + d)
                    d += W
                var sum_g      = acc_g.reduce_add()
                var sum_g_xhat = acc_gx.reduce_add()
                while d < Self.DIM:
                    var g = go_p[row + d] * g_p[d]
                    sum_g      += g
                    sum_g_xhat += g * xh_p[row + d]
                    d += 1
                var mean_g      = sum_g      * inv_dim
                var mean_g_xhat = sum_g_xhat * inv_dim
                # grad_input = inv_std·(g - mean_g - x̂·mean_g_xhat)
                var mg  = SIMD[DT, W](mean_g)
                var mgx = SIMD[DT, W](mean_g_xhat)
                var isv = SIMD[DT, W](inv_std)
                d = 0
                while d + W <= Self.DIM:
                    var g  = go_p.load[width=W](row + d) * g_p.load[width=W](d)
                    var xh = xh_p.load[width=W](row + d)
                    gi_p.store(row + d, isv * (g - mg - xh * mgx))
                    d += W
                while d < Self.DIM:
                    var g  = go_p[row + d] * g_p[d]
                    var xh = xh_p[row + d]
                    gi_p[row + d] = inv_std * (g - mean_g - xh * mean_g_xhat)
                    d += 1
                comptime if mode == "all":
                    # dγ += go·x̂ ; dβ += go  (accumulated across the batch)
                    d = 0
                    while d + W <= Self.DIM:
                        var go = go_p.load[width=W](row + d)
                        gg_p.store(
                            d,
                            gg_p.load[width=W](d)
                            + go * xh_p.load[width=W](row + d),
                        )
                        gb_p.store(d, gb_p.load[width=W](d) + go)
                        d += W
                    while d < Self.DIM:
                        gg_p[d] = gg_p[d] + go_p[row + d] * xh_p[row + d]
                        gb_p[d] = gb_p[d] + go_p[row + d]
                        d += 1
        else:
            var ctx = self.ts.ctx.value()
            comptime layout_2d = Layout.row_major(BATCH, Self.DIM)
            comptime layout_b  = Layout.row_major(BATCH)
            comptime layout_d  = Layout.row_major(Self.DIM)
            var go_p_w = grad_output_v.ptr
            var gi_p_w = grad_input_v.ptr
            var go_lt = LayoutTensor[DT, layout_2d, MutAnyOrigin](go_p_w)
            var gi_lt = LayoutTensor[DT, layout_2d, MutAnyOrigin](gi_p_w)
            var g_lt  = LayoutTensor[DT, layout_d, MutAnyOrigin](
                self.gamma.val.dev.value()
            )
            var xh_lt = LayoutTensor[DT, layout_2d, MutAnyOrigin](
                self.cache_xhat.dev.value()
            )
            var is_lt = LayoutTensor[DT, layout_b, MutAnyOrigin](
                self.cache_inv_std.dev.value()
            )
            # dx kernel: one block per sample.
            comptime dx_kernel = _layer_norm_backward_dx_kernel[BATCH, Self.DIM]
            ctx.enqueue_function[dx_kernel](
                go_lt, g_lt, xh_lt, is_lt, gi_lt,
                grid_dim=BATCH,
                block_dim=LN_TPB,
            )
            comptime if mode == "all":
                var gg_lt = LayoutTensor[DT, layout_d, MutAnyOrigin](
                    self.gamma.grd.dev.value()
                )
                var gb_lt = LayoutTensor[DT, layout_d, MutAnyOrigin](
                    self.beta.grd.dev.value()
                )
                comptime dp_kernel = _layer_norm_backward_dparams_kernel[
                    BATCH, Self.DIM
                ]
                ctx.enqueue_function[dp_kernel](
                    go_lt, xh_lt, gg_lt, gb_lt,
                    grid_dim=Self.DIM,
                    block_dim=LN_TPB,
                )

    # ----- Walkers ---------------------------------------------------------

    def for_each_param[
        target: StaticString,
        V: ParamVisitor,
    ](mut self, prefix: String, mut visitor: V) raises:
        assert_tag_for["LayerNorm", target](self.ts.target_tag)
        for_each_param_auto[Self, V, target](self, prefix, visitor)

    def zero_grad[target: StaticString](mut self) raises:
        assert_tag_for["LayerNorm", target](self.ts.target_tag)
        zero_grad_auto[Self, target](self)
