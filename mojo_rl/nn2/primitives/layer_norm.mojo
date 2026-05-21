"""LayerNorm[DIM] — retrofit (Phase B).

Same algorithm + kernels as v1 (`layer_norm.mojo`); only the
parameter/storage scaffolding changes:

  * `ts: TargetStorage` replaces `_target_tag` / `_inference` / `ctx`.
  * `gamma: Param["gamma", False, DIM]` + `beta: Param["beta", False, DIM]`
    replace the four lists + four device buffers.
  * `for_each_param` / `zero_grad` are one-liners delegating to
    `for_each_param_auto` / `zero_grad_auto`.
  * `backward[mode]` collapses v1's `backward` + `backward_input`.
  * Phase 10A buffer surface dropped — orchestrators own slabs.

Cache stays leaf-owned (output-caching, no aliasing): `cache_xhat`
[BATCH, DIM] and `cache_inv_std` [BATCH] live on the leaf and are
read in backward. Backward order doesn't matter for aliasing since
the cache is in dedicated buffers — kept v1's order (grad_input,
then dgamma/dbeta) for minimal behavioral change.

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
from ..core.module import Module
from ..core.target_storage import (
    TargetStorage,
    assert_tag_for,
    ensure_cpu_buffer,
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
    comptime IN_DIM = Self.DIM
    comptime OUT_DIM = Self.DIM

    # Params (decay=False for both; PyTorch + canonical AdamW recipe).
    var gamma: Param["gamma", False, Self.DIM]
    var beta:  Param["beta",  False, Self.DIM]

    # Cache (leaf-owned, output-caching).
    var cache_xhat: List[Scalar[DT]]                # [BATCH, DIM]
    var cache_inv_std: List[Scalar[DT]]             # [BATCH]
    var cache_xhat_dev: Optional[DeviceBuffer[DT]]
    var cache_inv_std_dev: Optional[DeviceBuffer[DT]]
    var cache_n_batch: Int

    var ts: TargetStorage

    def __init__(out self):
        self.gamma = Param["gamma", False, Self.DIM]()
        self.beta  = Param["beta",  False, Self.DIM]()
        self.cache_xhat = List[Scalar[DT]]()
        self.cache_inv_std = List[Scalar[DT]]()
        self.cache_xhat_dev = None
        self.cache_inv_std_dev = None
        self.cache_n_batch = 0
        self.ts = TargetStorage.make_uninit()

    @staticmethod
    def make[target: StaticString, INIT: Initializer]() raises -> Self:
        comptime assert target == "cpu", (
            "LayerNorm.make[target='gpu', INIT] requires a DeviceContext"
        )
        var ln = Self()
        ln.gamma = Param["gamma", False, Self.DIM].make_cpu()
        ln.beta  = Param["beta",  False, Self.DIM].make_cpu()
        # Universal LayerNorm init: γ=1, β=0. Param.make_cpu zero-filled;
        # post-fill gamma to 1.
        var g_ptr = ln.gamma.value_unsafe_ptr_cpu()
        for k in range(Self.DIM):
            g_ptr[k] = Scalar[DT](1.0)
        ln.ts = TargetStorage.make_cpu()
        return ln^

    @staticmethod
    def make[
        target: StaticString, INIT: Initializer
    ](ctx: DeviceContext) raises -> Self:
        comptime assert target == "gpu", (
            "LayerNorm.make[target='cpu', INIT](ctx) — drop ctx for CPU"
        )
        var ln = Self()
        ln.gamma = Param["gamma", False, Self.DIM].make_gpu(ctx)
        ln.beta  = Param["beta",  False, Self.DIM].make_gpu(ctx)
        ln.gamma.value_dev.value().enqueue_fill(1.0)
        ln.beta.value_dev.value().enqueue_fill(0.0)
        # Tiny placeholder cache buffers — actual sizes set on first forward.
        ln.cache_xhat_dev    = ctx.enqueue_create_buffer[DT](1)
        ln.cache_inv_std_dev = ctx.enqueue_create_buffer[DT](1)
        ln.cache_n_batch = 0
        ln.ts = TargetStorage.make_gpu(ctx)
        return ln^

    def _ensure_cache_gpu(mut self, batch: Int) raises:
        if self.cache_n_batch < batch:
            var ctx = self.ts.ctx.value()
            self.cache_xhat_dev = ctx.enqueue_create_buffer[DT](
                batch * Self.DIM
            )
            self.cache_inv_std_dev = ctx.enqueue_create_buffer[DT](batch)
            self.cache_n_batch = batch

    # ----- Forward ---------------------------------------------------------

    def forward[
        target: StaticString,
        BATCH: Int,
        POLICY: AMPPolicy = NoAMP,
    ](
        mut self,
        input: TileTensor[
            dtype=DT, address_space=AddressSpace.GENERIC, element_size=1, ...
        ],
        mut output: TileTensor[
            mut=True, dtype=DT, address_space=AddressSpace.GENERIC,
            element_size=1, ...,
        ],
    ) raises:
        comptime assert input.flat_rank == 2, "input must be rank-2 [BATCH, DIM]"
        comptime assert (
            output.flat_rank == 2
        ), "output must be rank-2 [BATCH, DIM]"
        assert_tag_for["LayerNorm", target](self.ts.target_tag)

        comptime if target == "cpu":
            ensure_cpu_buffer(self.cache_xhat,    BATCH * Self.DIM)
            ensure_cpu_buffer(self.cache_inv_std, BATCH)
            var gamma_v = TileTensor(self.gamma.value, row_major[Self.DIM]())
            var beta_v  = TileTensor(self.beta.value,  row_major[Self.DIM]())
            var xhat_v  = TileTensor(
                self.cache_xhat, row_major[BATCH, Self.DIM](),
            )
            var inv_v   = TileTensor(
                self.cache_inv_std, row_major[BATCH](),
            )
            var inv_dim: Scalar[DT] = 1.0 / Float32(Self.DIM)
            for b in range(BATCH):
                var s: Scalar[DT] = 0.0
                for d in range(Self.DIM):
                    s += input[b, d]
                var mean = s * inv_dim
                var sv: Scalar[DT] = 0.0
                for d in range(Self.DIM):
                    var diff = input[b, d] - mean
                    sv += diff * diff
                var var_v = sv * inv_dim
                var inv_std = Scalar[DT](1.0) / sqrt(var_v + LN_EPS)
                inv_v[b] = inv_std
                for d in range(Self.DIM):
                    var xh = (input[b, d] - mean) * inv_std
                    xhat_v[b, d] = xh
                    output[b, d] = gamma_v[d] * xh + beta_v[d]
        else:
            self._ensure_cache_gpu(BATCH)
            comptime layout_2d = Layout.row_major(BATCH, Self.DIM)
            comptime layout_b  = Layout.row_major(BATCH)
            comptime layout_d  = Layout.row_major(Self.DIM)
            var in_p_w  = rebind[UnsafePointer[Scalar[DT], MutAnyOrigin]](input.ptr)
            var out_p_w = rebind[UnsafePointer[Scalar[DT], MutAnyOrigin]](output.ptr)
            var in_lt  = LayoutTensor[DT, layout_2d, MutAnyOrigin](in_p_w)
            var out_lt = LayoutTensor[DT, layout_2d, MutAnyOrigin](out_p_w)
            var g_lt   = LayoutTensor[DT, layout_d, MutAnyOrigin](
                self.gamma.value_dev.value()
            )
            var b_lt   = LayoutTensor[DT, layout_d, MutAnyOrigin](
                self.beta.value_dev.value()
            )
            var xh_lt  = LayoutTensor[DT, layout_2d, MutAnyOrigin](
                self.cache_xhat_dev.value()
            )
            var is_lt  = LayoutTensor[DT, layout_b, MutAnyOrigin](
                self.cache_inv_std_dev.value()
            )
            comptime kernel = _layer_norm_forward_kernel[BATCH, Self.DIM]
            self.ts.ctx.value().enqueue_function[kernel](
                in_lt, out_lt, g_lt, b_lt, xh_lt, is_lt,
                grid_dim=BATCH,
                block_dim=LN_TPB,
            )

    # ----- Backward --------------------------------------------------------

    def backward[
        target: StaticString,
        BATCH: Int,
        POLICY: AMPPolicy = NoAMP,
        mode: StaticString = "all",
    ](
        mut self,
        grad_output: TileTensor[
            dtype=DT, address_space=AddressSpace.GENERIC, element_size=1, ...
        ],
        mut grad_input: TileTensor[
            mut=True, dtype=DT, address_space=AddressSpace.GENERIC,
            element_size=1, ...,
        ],
    ) raises:
        comptime assert grad_output.flat_rank == 2, "grad_output must be rank-2"
        comptime assert grad_input.flat_rank == 2, "grad_input must be rank-2"
        comptime assert (
            mode == "all" or mode == "input_only"
        ), "mode must be 'all' or 'input_only'"
        assert_tag_for["LayerNorm", target](self.ts.target_tag)

        comptime if target == "cpu":
            var gamma_v       = TileTensor(self.gamma.value, row_major[Self.DIM]())
            var grad_gamma_v  = TileTensor(self.gamma.grad,  row_major[Self.DIM]())
            var grad_beta_v   = TileTensor(self.beta.grad,   row_major[Self.DIM]())
            var xhat_v        = TileTensor(
                self.cache_xhat, row_major[BATCH, Self.DIM](),
            )
            var inv_v         = TileTensor(
                self.cache_inv_std, row_major[BATCH](),
            )
            var inv_dim: Scalar[DT] = 1.0 / Float32(Self.DIM)
            for b in range(BATCH):
                var inv_std = inv_v[b]
                var sum_g: Scalar[DT] = 0.0
                var sum_g_xhat: Scalar[DT] = 0.0
                for d in range(Self.DIM):
                    var g = grad_output[b, d] * gamma_v[d]
                    sum_g       += g
                    sum_g_xhat  += g * xhat_v[b, d]
                var mean_g       = sum_g       * inv_dim
                var mean_g_xhat  = sum_g_xhat  * inv_dim
                for d in range(Self.DIM):
                    var g  = grad_output[b, d] * gamma_v[d]
                    var xh = xhat_v[b, d]
                    grad_input[b, d] = inv_std * (g - mean_g - xh * mean_g_xhat)
                comptime if mode == "all":
                    for d in range(Self.DIM):
                        grad_gamma_v[d] += grad_output[b, d] * xhat_v[b, d]
                        grad_beta_v[d]  += grad_output[b, d]
        else:
            var ctx = self.ts.ctx.value()
            comptime layout_2d = Layout.row_major(BATCH, Self.DIM)
            comptime layout_b  = Layout.row_major(BATCH)
            comptime layout_d  = Layout.row_major(Self.DIM)
            var go_p_w = rebind[UnsafePointer[Scalar[DT], MutAnyOrigin]](grad_output.ptr)
            var gi_p_w = rebind[UnsafePointer[Scalar[DT], MutAnyOrigin]](grad_input.ptr)
            var go_lt = LayoutTensor[DT, layout_2d, MutAnyOrigin](go_p_w)
            var gi_lt = LayoutTensor[DT, layout_2d, MutAnyOrigin](gi_p_w)
            var g_lt  = LayoutTensor[DT, layout_d, MutAnyOrigin](
                self.gamma.value_dev.value()
            )
            var xh_lt = LayoutTensor[DT, layout_2d, MutAnyOrigin](
                self.cache_xhat_dev.value()
            )
            var is_lt = LayoutTensor[DT, layout_b, MutAnyOrigin](
                self.cache_inv_std_dev.value()
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
                    self.gamma.grad_dev.value()
                )
                var gb_lt = LayoutTensor[DT, layout_d, MutAnyOrigin](
                    self.beta.grad_dev.value()
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
