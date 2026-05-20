"""LayerNorm[DIM] — per-sample layer normalization. Phase 5.4.

Forward:
    mean = (1/D) Σ_d x[b, d]
    var  = (1/D) Σ_d (x[b, d] - mean)^2
    inv_std = 1 / sqrt(var + EPS)
    x_hat = (x - mean) * inv_std
    y[b, d] = gamma[d] * x_hat[b, d] + beta[d]

Backward (per-sample, D-dim, derived from the affine + normalize chain):
    g           = grad_output * gamma          (per-element)
    mean_g      = (1/D) Σ_d g
    mean_g_xhat = (1/D) Σ_d g * x_hat
    grad_input  = inv_std * (g - mean_g - x_hat * mean_g_xhat)
    grad_gamma  += grad_output * x_hat         (reduced over batch)
    grad_beta   += grad_output                  (reduced over batch)

AMP: `force_fp32_input = True` per AMPPolicy contract — LayerNorm
ignores POLICY and always runs in DT. Stats become numerically
unstable in bf16, and Phase 3 ships only fp32 + bf16 anyway.

apply_decay: False for both γ and β. Layer-local convention; AdamW
sees `apply_decay=False` and skips the decay term — same trick that
keeps biases out of decay on Linear.

Initialization: γ=1, β=0 (the only universal LayerNorm init). INIT
is accepted for trait conformance but ignored.

ε: hardcoded to 1e-5 (matches PyTorch + Phase 3 reference). If a use
case appears, promote to a comptime param later.

Cache: x_hat (BATCH × DIM) + inv_std (BATCH × 1). Two separate
buffers — simpler than packing into a (BATCH × (DIM+1)) blob.
"""

from std.math import sqrt
from std.gpu import thread_idx, block_idx
from std.gpu.primitives import block
from std.gpu.host import DeviceContext, DeviceBuffer
from layout import Layout, LayoutTensor, TileTensor, TensorLayout, row_major

from ..constants import DT
from ..core import (
    Module,
    ParamVisitor,
    Initializer,
    AMPPolicy,
    NoAMP,
    TARGET_UNINIT,
    TARGET_CPU,
    TARGET_GPU,
    target_tag_for,
)


comptime LN_EPS: Scalar[DT] = 1e-5
comptime LN_TPB: Int = 128


# ──────────────────────────────────────────────────────────────────────────
# GPU kernels.
# ──────────────────────────────────────────────────────────────────────────


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
    """Grid: (BATCH,) Block: (LN_TPB,). One block normalizes one sample."""
    var b = Int(block_idx.x)
    var t = Int(thread_idx.x)

    if b >= BATCH:
        return

    var inv_dim: Scalar[DT] = 1.0 / Float32(DIM)

    # Phase 1: mean via block-reduce.
    var my_sum: Scalar[DT] = 0.0
    var idx = t
    while idx < DIM:
        my_sum += rebind[Scalar[DT]](input[b, idx])
        idx += LN_TPB
    var mean_val = (
        block.sum[block_size=LN_TPB, broadcast=True](val=my_sum) * inv_dim
    )

    # Phase 2: variance via block-reduce.
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

    # Phase 3: normalize, scale, shift.
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
    """Grid: (BATCH,) Block: (LN_TPB,). Computes dx only (γ, β handled
    by a separate column-reduction kernel)."""
    var b = Int(block_idx.x)
    var t = Int(thread_idx.x)

    if b >= BATCH:
        return

    var inv_dim: Scalar[DT] = 1.0 / Float32(DIM)
    var inv_std = rebind[Scalar[DT]](cache_inv_std[b])

    # Phase 1: mean(g) and mean(g * x_hat) where g = grad_output * gamma.
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

    # Phase 2: write grad_input.
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
    """Grid: (DIM,) Block: (LN_TPB,). Each block reduces one column
    over the batch dimension, accumulating into grad_γ/grad_β
    (caller must zero them first if a fresh accumulation is wanted —
    matches Linear's grad-accumulate convention)."""
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


# ──────────────────────────────────────────────────────────────────────────
# LayerNorm — method-level target.
# ──────────────────────────────────────────────────────────────────────────


struct LayerNorm[DIM: Int](Module):
    comptime IN_DIM = Self.DIM
    comptime OUT_DIM = Self.DIM

    # CPU storage.
    var gamma: List[Scalar[DT]]
    var beta: List[Scalar[DT]]
    var grad_gamma: List[Scalar[DT]]
    var grad_beta: List[Scalar[DT]]
    var cache_xhat: List[Scalar[DT]]
    var cache_inv_std: List[Scalar[DT]]

    # GPU storage.
    var gamma_dev: Optional[DeviceBuffer[DT]]
    var beta_dev: Optional[DeviceBuffer[DT]]
    var grad_gamma_dev: Optional[DeviceBuffer[DT]]
    var grad_beta_dev: Optional[DeviceBuffer[DT]]
    var cache_xhat_dev: Optional[DeviceBuffer[DT]]
    var cache_inv_std_dev: Optional[DeviceBuffer[DT]]
    var cache_n_batch: Int
    var ctx: Optional[DeviceContext]

    var _target_tag: Int8
    var _inference: Bool

    def __init__(out self):
        self.gamma = List[Scalar[DT]]()
        self.beta = List[Scalar[DT]]()
        self.grad_gamma = List[Scalar[DT]]()
        self.grad_beta = List[Scalar[DT]]()
        self.cache_xhat = List[Scalar[DT]]()
        self.cache_inv_std = List[Scalar[DT]]()
        self.gamma_dev = None
        self.beta_dev = None
        self.grad_gamma_dev = None
        self.grad_beta_dev = None
        self.cache_xhat_dev = None
        self.cache_inv_std_dev = None
        self.cache_n_batch = 0
        self.ctx = None
        self._target_tag = TARGET_UNINIT
        self._inference = False

    @staticmethod
    def make[target: StaticString, INIT: Initializer]() raises -> Self:
        """CPU factory. γ=1, β=0 (universal LayerNorm init); INIT ignored."""
        comptime assert (
            target == "cpu"
        ), "LayerNorm.make[target='gpu', INIT] requires a DeviceContext"
        var ln = Self()
        ln.gamma      = List[Scalar[DT]](length=Self.DIM, fill=1.0)
        ln.beta       = List[Scalar[DT]](length=Self.DIM, fill=0.0)
        ln.grad_gamma = List[Scalar[DT]](length=Self.DIM, fill=0.0)
        ln.grad_beta  = List[Scalar[DT]](length=Self.DIM, fill=0.0)
        ln._target_tag = TARGET_CPU
        return ln^

    @staticmethod
    def make[
        target: StaticString, INIT: Initializer
    ](ctx: DeviceContext) raises -> Self:
        """GPU factory."""
        comptime assert (
            target == "gpu"
        ), "LayerNorm.make[target='cpu', INIT](ctx) — drop ctx for CPU"
        var ln = Self()
        var gamma_dev = ctx.enqueue_create_buffer[DT](Self.DIM)
        var beta_dev  = ctx.enqueue_create_buffer[DT](Self.DIM)
        var gg_dev    = ctx.enqueue_create_buffer[DT](Self.DIM)
        var gb_dev    = ctx.enqueue_create_buffer[DT](Self.DIM)
        var cxh_dev   = ctx.enqueue_create_buffer[DT](1)
        var cis_dev   = ctx.enqueue_create_buffer[DT](1)
        gamma_dev.enqueue_fill(1.0)
        beta_dev.enqueue_fill(0.0)
        gg_dev.enqueue_fill(0.0)
        gb_dev.enqueue_fill(0.0)
        ln.gamma_dev = gamma_dev^
        ln.beta_dev = beta_dev^
        ln.grad_gamma_dev = gg_dev^
        ln.grad_beta_dev = gb_dev^
        ln.cache_xhat_dev = cxh_dev^
        ln.cache_inv_std_dev = cis_dev^
        ln.cache_n_batch = 0
        ln.ctx = ctx
        ln._target_tag = TARGET_GPU
        return ln^

    def _assert_tag[target: StaticString](self) raises:
        comptime expected = target_tag_for[target]()
        if self._target_tag != expected:
            raise Error(
                "LayerNorm: method called with [target='"
                + String(target)
                + "'] but module was make'd for a different target "
                + "(tag=" + String(Int(self._target_tag)) + ")"
            )

    def _ensure_cache_cpu(mut self, batch: Int):
        var needed_xhat = batch * Self.DIM
        if len(self.cache_xhat) < needed_xhat:
            self.cache_xhat.resize(needed_xhat, 0.0)
        if len(self.cache_inv_std) < batch:
            self.cache_inv_std.resize(batch, 0.0)

    def _ensure_cache_gpu(mut self, batch: Int) raises:
        if self.cache_n_batch < batch:
            self.cache_xhat_dev = self.ctx.value().enqueue_create_buffer[DT](
                batch * Self.DIM
            )
            self.cache_inv_std_dev = self.ctx.value().enqueue_create_buffer[DT](
                batch
            )
            self.cache_n_batch = batch

    def forward[
        target: StaticString,
        BATCH: Int,
        LIN: TensorLayout,
        LOUT: TensorLayout,
        OIN: MutOrigin,
        OOUT: MutOrigin,
        POLICY: AMPPolicy = NoAMP,
    ](
        mut self,
        input: TileTensor[DT, LIN, OIN],
        mut output: TileTensor[DT, LOUT, OOUT],
    ) raises:
        # LayerNorm contract: force_fp32_input=True. POLICY ignored.
        comptime assert (
            input.flat_rank == 2
        ), "input must be rank-2 [BATCH, DIM]"
        comptime assert (
            output.flat_rank == 2
        ), "output must be rank-2 [BATCH, DIM]"
        self._assert_tag[target]()

        comptime if target == "cpu":
            self._ensure_cache_cpu(BATCH)
            var gamma_v = TileTensor(self.gamma, row_major[Self.DIM]())
            var beta_v  = TileTensor(self.beta,  row_major[Self.DIM]())
            var xhat_v  = TileTensor(self.cache_xhat, row_major[BATCH, Self.DIM]())
            var inv_v   = TileTensor(self.cache_inv_std, row_major[BATCH]())
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
            var input_w  = rebind[TileTensor[DT, LIN, MutAnyOrigin]](input)
            var output_w = rebind[TileTensor[DT, LOUT, MutAnyOrigin]](output)
            var in_lt  = LayoutTensor[DT, layout_2d, MutAnyOrigin](input_w.ptr)
            var out_lt = LayoutTensor[DT, layout_2d, MutAnyOrigin](output_w.ptr)
            var g_lt   = LayoutTensor[DT, layout_d, MutAnyOrigin](self.gamma_dev.value())
            var b_lt   = LayoutTensor[DT, layout_d, MutAnyOrigin](self.beta_dev.value())
            var xh_lt  = LayoutTensor[DT, layout_2d, MutAnyOrigin](self.cache_xhat_dev.value())
            var is_lt  = LayoutTensor[DT, layout_b, MutAnyOrigin](self.cache_inv_std_dev.value())
            comptime kernel = _layer_norm_forward_kernel[BATCH, Self.DIM]
            self.ctx.value().enqueue_function[kernel](
                in_lt, out_lt, g_lt, b_lt, xh_lt, is_lt,
                grid_dim=BATCH,
                block_dim=LN_TPB,
            )

    def backward[
        target: StaticString,
        BATCH: Int,
        LGO: TensorLayout,
        LGI: TensorLayout,
        OGO: MutOrigin,
        OGI: MutOrigin,
        POLICY: AMPPolicy = NoAMP,
    ](
        mut self,
        grad_output: TileTensor[DT, LGO, OGO],
        mut grad_input: TileTensor[DT, LGI, OGI],
    ) raises:
        comptime assert grad_output.flat_rank == 2, "grad_output must be rank-2"
        comptime assert grad_input.flat_rank == 2, "grad_input must be rank-2"
        self._assert_tag[target]()

        comptime if target == "cpu":
            var gamma_v = TileTensor(self.gamma, row_major[Self.DIM]())
            var grad_gamma_v = TileTensor(self.grad_gamma, row_major[Self.DIM]())
            var grad_beta_v  = TileTensor(self.grad_beta,  row_major[Self.DIM]())
            var xhat_v       = TileTensor(self.cache_xhat, row_major[BATCH, Self.DIM]())
            var inv_v        = TileTensor(self.cache_inv_std, row_major[BATCH]())
            var inv_dim: Scalar[DT] = 1.0 / Float32(Self.DIM)
            for b in range(BATCH):
                var inv_std = inv_v[b]
                # mean(g) and mean(g * x_hat).
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
                # Accumulate dgamma / dbeta over batch.
                for d in range(Self.DIM):
                    grad_gamma_v[d] += grad_output[b, d] * xhat_v[b, d]
                    grad_beta_v[d]  += grad_output[b, d]
        else:
            comptime layout_2d = Layout.row_major(BATCH, Self.DIM)
            comptime layout_b  = Layout.row_major(BATCH)
            comptime layout_d  = Layout.row_major(Self.DIM)
            var grad_output_w = rebind[TileTensor[DT, LGO, MutAnyOrigin]](
                grad_output
            )
            var grad_input_w  = rebind[TileTensor[DT, LGI, MutAnyOrigin]](
                grad_input
            )
            var go_lt = LayoutTensor[DT, layout_2d, MutAnyOrigin](grad_output_w.ptr)
            var gi_lt = LayoutTensor[DT, layout_2d, MutAnyOrigin](grad_input_w.ptr)
            var g_lt  = LayoutTensor[DT, layout_d, MutAnyOrigin](self.gamma_dev.value())
            var xh_lt = LayoutTensor[DT, layout_2d, MutAnyOrigin](self.cache_xhat_dev.value())
            var is_lt = LayoutTensor[DT, layout_b, MutAnyOrigin](self.cache_inv_std_dev.value())
            var gg_lt = LayoutTensor[DT, layout_d, MutAnyOrigin](self.grad_gamma_dev.value())
            var gb_lt = LayoutTensor[DT, layout_d, MutAnyOrigin](self.grad_beta_dev.value())

            # dx kernel: one block per sample.
            comptime dx_kernel = _layer_norm_backward_dx_kernel[BATCH, Self.DIM]
            self.ctx.value().enqueue_function[dx_kernel](
                go_lt, g_lt, xh_lt, is_lt, gi_lt,
                grid_dim=BATCH,
                block_dim=LN_TPB,
            )
            # dgamma/dbeta kernel: one block per column.
            comptime dp_kernel = _layer_norm_backward_dparams_kernel[BATCH, Self.DIM]
            self.ctx.value().enqueue_function[dp_kernel](
                go_lt, xh_lt, gg_lt, gb_lt,
                grid_dim=Self.DIM,
                block_dim=LN_TPB,
            )

    def backward_input[
        target: StaticString,
        BATCH: Int,
        LGO: TensorLayout,
        LGI: TensorLayout,
        OGO: MutOrigin,
        OGI: MutOrigin,
        POLICY: AMPPolicy = NoAMP,
    ](
        mut self,
        grad_output: TileTensor[DT, LGO, OGO],
        mut grad_input: TileTensor[DT, LGI, OGI],
    ) raises:
        # Phase 8.2: grad_input only; skip dgamma/dbeta accumulation.
        comptime assert grad_output.flat_rank == 2, "grad_output must be rank-2"
        comptime assert grad_input.flat_rank == 2, "grad_input must be rank-2"
        self._assert_tag[target]()

        comptime if target == "cpu":
            var gamma_v = TileTensor(self.gamma, row_major[Self.DIM]())
            var xhat_v  = TileTensor(self.cache_xhat, row_major[BATCH, Self.DIM]())
            var inv_v   = TileTensor(self.cache_inv_std, row_major[BATCH]())
            var inv_dim: Scalar[DT] = 1.0 / Float32(Self.DIM)
            for b in range(BATCH):
                var inv_std = inv_v[b]
                var sum_g: Scalar[DT] = 0.0
                var sum_g_xhat: Scalar[DT] = 0.0
                for d in range(Self.DIM):
                    var g = grad_output[b, d] * gamma_v[d]
                    sum_g      += g
                    sum_g_xhat += g * xhat_v[b, d]
                var mean_g      = sum_g * inv_dim
                var mean_g_xhat = sum_g_xhat * inv_dim
                for d in range(Self.DIM):
                    var g  = grad_output[b, d] * gamma_v[d]
                    var xh = xhat_v[b, d]
                    grad_input[b, d] = inv_std * (g - mean_g - xh * mean_g_xhat)
        else:
            comptime layout_2d = Layout.row_major(BATCH, Self.DIM)
            comptime layout_b  = Layout.row_major(BATCH)
            comptime layout_d  = Layout.row_major(Self.DIM)
            var grad_output_w = rebind[TileTensor[DT, LGO, MutAnyOrigin]](
                grad_output
            )
            var grad_input_w  = rebind[TileTensor[DT, LGI, MutAnyOrigin]](
                grad_input
            )
            var go_lt = LayoutTensor[DT, layout_2d, MutAnyOrigin](grad_output_w.ptr)
            var gi_lt = LayoutTensor[DT, layout_2d, MutAnyOrigin](grad_input_w.ptr)
            var g_lt  = LayoutTensor[DT, layout_d, MutAnyOrigin](self.gamma_dev.value())
            var xh_lt = LayoutTensor[DT, layout_2d, MutAnyOrigin](self.cache_xhat_dev.value())
            var is_lt = LayoutTensor[DT, layout_b, MutAnyOrigin](self.cache_inv_std_dev.value())
            comptime dx_kernel = _layer_norm_backward_dx_kernel[BATCH, Self.DIM]
            self.ctx.value().enqueue_function[dx_kernel](
                go_lt, g_lt, xh_lt, is_lt, gi_lt,
                grid_dim=BATCH,
                block_dim=LN_TPB,
            )

    def for_each_param[
        target: StaticString,
        V: ParamVisitor,
    ](mut self, prefix: String, mut visitor: V,) raises:
        self._assert_tag[target]()
        var sep = "." if prefix.byte_length() > 0 else ""
        # Decay convention: γ and β are both excluded from weight decay
        # (PyTorch + canonical AdamW transformer recipes).
        comptime if target == "cpu":
            var g_view  = TileTensor(self.gamma, row_major[Self.DIM]())
            var gg_view = TileTensor(self.grad_gamma, row_major[Self.DIM]())
            var b_view  = TileTensor(self.beta, row_major[Self.DIM]())
            var gb_view = TileTensor(self.grad_beta, row_major[Self.DIM]())
            visitor.visit(prefix + sep + "gamma", g_view, gg_view, Self.DIM, False)
            visitor.visit(prefix + sep + "beta",  b_view, gb_view, Self.DIM, False)
        else:
            var g_view  = TileTensor(self.gamma_dev.value(),      row_major[Self.DIM]())
            var gg_view = TileTensor(self.grad_gamma_dev.value(), row_major[Self.DIM]())
            var b_view  = TileTensor(self.beta_dev.value(),       row_major[Self.DIM]())
            var gb_view = TileTensor(self.grad_beta_dev.value(),  row_major[Self.DIM]())
            visitor.visit(prefix + sep + "gamma", g_view, gg_view, Self.DIM, False)
            visitor.visit(prefix + sep + "beta",  b_view, gb_view, Self.DIM, False)

    def set_inference(mut self, value: Bool):
        # LayerNorm has no train/eval split (no running stats in this
        # version — by design; BN-style running stats land separately).
        self._inference = value
