"""LayerNormNoAffine[DIM] — LayerNorm without learnable scale/bias.

    y = (x - mean) / sqrt(var + eps)

No γ/β params (PARAM-free). AdaLN-zero supplies the affine externally via
a `Modulate` step, so the normalization itself must be affine-free. This
is the nn port of the legacy `nn/.../layer_norm_no_affine.mojo`, reusing
the LayerNorm GPU reduction kernels with the gamma/beta terms dropped.

Cache (leaf-owned, output-caching — no input aliasing):
  * `cache_xhat`    [BATCH, DIM]  normalized x (= output)
  * `cache_inv_std` [BATCH]       1/sqrt(var+eps)

Backward (no gamma): grad_in = inv_std·(g − mean(g) − x̂·mean(g·x̂)),
with g = grad_out directly.

AMP: always runs in DT (stats unstable in bf16). `INIT` accepted for
trait conformance, ignored (no params).
"""

from std.math import sqrt
from std.gpu import thread_idx, block_idx
from std.gpu.primitives import block
from std.gpu.host import DeviceContext, DeviceBuffer
from std.gpu.memory import AddressSpace
from layout import Layout, LayoutTensor, TileTensor, row_major

from ..constants import DT
from ..core import Initializer, AMPPolicy, NoAMP, Cache
from ..core.module import Module, typed_view, typed_view_mut
from ..core.tensor_pack import TensorPack
from ..core.target_storage import (
    require_ctx,
    TargetStorage,
    assert_tag_for,
)


comptime LNNA_EPS: Scalar[DT] = 1e-6
comptime LNNA_TPB: Int = 128


# ──────────────────────────────────────────────────────────────────────
# GPU kernels — LayerNorm reductions with gamma/beta dropped.
# ──────────────────────────────────────────────────────────────────────


def _lnna_forward_kernel[
    BATCH: Int, DIM: Int,
](
    input: LayoutTensor[DT, Layout.row_major(BATCH, DIM), MutAnyOrigin],
    output: LayoutTensor[DT, Layout.row_major(BATCH, DIM), MutAnyOrigin],
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
        idx += LNNA_TPB
    var mean_val = (
        block.sum[block_size=LNNA_TPB, broadcast=True](val=my_sum) * inv_dim
    )

    var my_var: Scalar[DT] = 0.0
    idx = t
    while idx < DIM:
        var diff = rebind[Scalar[DT]](input[b, idx]) - mean_val
        my_var += diff * diff
        idx += LNNA_TPB
    var var_val = (
        block.sum[block_size=LNNA_TPB, broadcast=True](val=my_var) * inv_dim
    )

    var inv_std: Scalar[DT] = 1.0 / sqrt(var_val + LNNA_EPS)
    if t == 0:
        cache_inv_std[b] = inv_std

    idx = t
    while idx < DIM:
        var x = rebind[Scalar[DT]](input[b, idx])
        var x_hat = (x - mean_val) * inv_std
        cache_xhat[b, idx] = x_hat
        output[b, idx] = x_hat
        idx += LNNA_TPB


def _lnna_backward_kernel[
    BATCH: Int, DIM: Int,
](
    grad_output: LayoutTensor[DT, Layout.row_major(BATCH, DIM), MutAnyOrigin],
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
        var g = rebind[Scalar[DT]](grad_output[b, idx])
        var xh = rebind[Scalar[DT]](cache_xhat[b, idx])
        my_g += g
        my_g_xhat += g * xh
        idx += LNNA_TPB
    var mean_g = (
        block.sum[block_size=LNNA_TPB, broadcast=True](val=my_g) * inv_dim
    )
    var mean_g_xhat = (
        block.sum[block_size=LNNA_TPB, broadcast=True](val=my_g_xhat) * inv_dim
    )

    idx = t
    while idx < DIM:
        var g = rebind[Scalar[DT]](grad_output[b, idx])
        var xh = rebind[Scalar[DT]](cache_xhat[b, idx])
        grad_input[b, idx] = inv_std * (g - mean_g - xh * mean_g_xhat)
        idx += LNNA_TPB


# ──────────────────────────────────────────────────────────────────────
# LayerNormNoAffine.
# ──────────────────────────────────────────────────────────────────────


struct LayerNormNoAffine[DIM: Int](Module):
    comptime ARITY: Int = 1
    comptime IN_DIMS = InlineArray[Int, 1](fill=Self.DIM)
    comptime OUT_DIM = Self.DIM

    @staticmethod
    def display_label() -> String:
        return String("LayerNormNoAffine")

    var cache_xhat: Cache["cache_xhat"]
    var cache_inv_std: Cache["cache_inv_std"]
    var ts: TargetStorage

    def __init__(out self):
        self.cache_xhat = Cache["cache_xhat"]()
        self.cache_inv_std = Cache["cache_inv_std"]()
        self.ts = TargetStorage.make_uninit()

    @staticmethod
    def make[
        target: StaticString, INIT: Initializer
    ](ctx: Optional[DeviceContext] = None) raises -> Self:
        comptime assert target == "cpu" or target == "gpu", (
            "LayerNormNoAffine: target must be 'cpu' or 'gpu'"
        )
        var ln = Self()
        comptime if target == "cpu":
            ln.ts = TargetStorage.make_cpu()
        else:
            var ctx_v = require_ctx["LayerNormNoAffine.make[target='gpu']"](ctx)
            ln.ts = TargetStorage.make_gpu(ctx_v)
        return ln^

    def _ensure_cache_gpu(mut self, batch: Int) raises:
        var ctx = self.ts.ctx.value()
        self.cache_xhat.ensure_gpu(ctx, batch * Self.DIM)
        self.cache_inv_std.ensure_gpu(ctx, batch)

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
        assert_tag_for["LayerNormNoAffine", target](self.ts.target_tag)
        var input = inputs.tile[0, BATCH, Self.IN_DIMS[0]]()
        var output_v = typed_view_mut[BATCH, Self.OUT_DIM](output)

        comptime if target == "cpu":
            self.cache_xhat.ensure_cpu(BATCH * Self.DIM)
            self.cache_inv_std.ensure_cpu(BATCH)
            var xhat_v = TileTensor(
                self.cache_xhat.cpu, row_major[BATCH, Self.DIM]()
            )
            var inv_v = TileTensor(self.cache_inv_std.cpu, row_major[BATCH]())
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
                var inv_std = Scalar[DT](1.0) / sqrt(var_v + LNNA_EPS)
                inv_v[b] = inv_std
                for d in range(Self.DIM):
                    var xh = (input[b, d] - mean) * inv_std
                    xhat_v[b, d] = xh
                    output_v[b, d] = xh
        else:
            self._ensure_cache_gpu(BATCH)
            comptime layout_2d = Layout.row_major(BATCH, Self.DIM)
            comptime layout_b = Layout.row_major(BATCH)
            var in_p = input.ptr
            var out_p = output_v.ptr
            var in_lt = LayoutTensor[DT, layout_2d, MutAnyOrigin](in_p)
            var out_lt = LayoutTensor[DT, layout_2d, MutAnyOrigin](out_p)
            var xh_lt = LayoutTensor[DT, layout_2d, MutAnyOrigin](
                self.cache_xhat.dev.value()
            )
            var is_lt = LayoutTensor[DT, layout_b, MutAnyOrigin](
                self.cache_inv_std.dev.value()
            )
            comptime kernel = _lnna_forward_kernel[BATCH, Self.DIM]
            self.ts.ctx.value().enqueue_function[kernel](
                in_lt, out_lt, xh_lt, is_lt,
                grid_dim=BATCH, block_dim=LNNA_TPB,
            )

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
        assert_tag_for["LayerNormNoAffine", target](self.ts.target_tag)
        var grad_output_v = typed_view[BATCH, Self.OUT_DIM](grad_output)
        var grad_input_v = grad_inputs.tile[0, BATCH, Self.IN_DIMS[0]]()

        comptime if target == "cpu":
            var xhat_v = TileTensor(
                self.cache_xhat.cpu, row_major[BATCH, Self.DIM]()
            )
            var inv_v = TileTensor(self.cache_inv_std.cpu, row_major[BATCH]())
            var inv_dim: Scalar[DT] = 1.0 / Float32(Self.DIM)
            for b in range(BATCH):
                var inv_std = inv_v[b]
                var sum_g: Scalar[DT] = 0.0
                var sum_g_xhat: Scalar[DT] = 0.0
                for d in range(Self.DIM):
                    var g = grad_output_v[b, d]
                    sum_g += g
                    sum_g_xhat += g * xhat_v[b, d]
                var mean_g = sum_g * inv_dim
                var mean_g_xhat = sum_g_xhat * inv_dim
                for d in range(Self.DIM):
                    var g = grad_output_v[b, d]
                    var xh = xhat_v[b, d]
                    grad_input_v[b, d] = (
                        inv_std * (g - mean_g - xh * mean_g_xhat)
                    )
        else:
            var ctx = self.ts.ctx.value()
            comptime layout_2d = Layout.row_major(BATCH, Self.DIM)
            comptime layout_b = Layout.row_major(BATCH)
            var go_p = grad_output_v.ptr
            var gi_p = grad_input_v.ptr
            var go_lt = LayoutTensor[DT, layout_2d, MutAnyOrigin](go_p)
            var gi_lt = LayoutTensor[DT, layout_2d, MutAnyOrigin](gi_p)
            var xh_lt = LayoutTensor[DT, layout_2d, MutAnyOrigin](
                self.cache_xhat.dev.value()
            )
            var is_lt = LayoutTensor[DT, layout_b, MutAnyOrigin](
                self.cache_inv_std.dev.value()
            )
            comptime kernel = _lnna_backward_kernel[BATCH, Self.DIM]
            ctx.enqueue_function[kernel](
                go_lt, xh_lt, is_lt, gi_lt,
                grid_dim=BATCH, block_dim=LNNA_TPB,
            )
