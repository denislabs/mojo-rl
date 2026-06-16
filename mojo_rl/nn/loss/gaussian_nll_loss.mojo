"""GaussianNLLLoss[DIM] — diagonal Gaussian negative-log-likelihood.

Phase I.1.a. The probabilistic head used by MBPO's dynamics ensemble:
each member predicts `(µ, logvar)` per output dim, and the training
target is to maximise the log-likelihood of the observed `(reward, Δobs)`
under that diagonal-Gaussian distribution.

Conforms to nn's `Loss` trait, so `Trainer[..., LOSS=GaussianNLLLoss[DIM]]`
slots in next to MSELoss / SoftCrossEntropyLoss.

Tensor conventions:

  - **logits**  shape `BATCH × (2*DIM)` — first `DIM` columns are
    means, next `DIM` columns are raw logvars (pre-clamp).
  - **targets** shape `BATCH × DIM` — observed values whose likelihood
    we maximise.
  - **grad_logits** shape `BATCH × (2*DIM)` — first `DIM` rows of
    output are `d_loss/d_µ`, next `DIM` are `d_loss/d_raw_logvar`.

Math (per-row, summed over DIM, then averaged over BATCH):

  σ²       = exp(clamp(raw_logvar, [LOGVAR_MIN, LOGVAR_MAX]))
  loss_row = Σᵢ ½·(yᵢ - µᵢ)² · σ⁻²ᵢ + ½·clamped_logvarᵢ
  loss     = (1/BATCH) · Σ_row loss_row

  d_loss/d_µ_i             = (µᵢ - yᵢ) · σ⁻²ᵢ / BATCH
  d_loss/d_raw_logvar_i    = [ ½ - ½·(yᵢ - µᵢ)²·σ⁻²ᵢ ] / BATCH   (in-clamp)
  d_loss/d_raw_logvar_i    = 0                                       (clamped)

Default logvar bounds `[-10, -2]` match MBPO reference
(`deep_agents/core/agents/mbpo_agent.mojo:182-183` — CPU fixed-bounds
path; the GPU production agent treats them as learnable parameters,
which is deferred to a future I.1.* phase).

GPU paths follow the MSELoss pattern: per-batch-row forward kernel
(each thread computes its row's contribution to the loss + populates
the per-row caches), then a host-side sum + divide; per-element vjp
kernel (each thread writes one [µᵢ, raw_lvᵢ] grad pair).
"""

from std.math import exp
from std.gpu import global_idx
from std.gpu.host import DeviceContext, DeviceBuffer, HostBuffer
from std.gpu.memory import AddressSpace
from layout import Layout, LayoutTensor, TileTensor

from ..constants import DT, CPU_SIMD_W, TPB
from ..core.module import mptr
from ..core import Loss, AMPPolicy, NoAMP
from ..core.target_storage import (
    TargetStorage, assert_tag_for, ensure_gpu_buffer,
)


# ──────────────────────────────────────────────────────────────────────
# GPU kernels — one thread per batch row for forward (populates per-row
# caches + partial_loss), one thread per element for vjp.
# ──────────────────────────────────────────────────────────────────────


def _gauss_nll_forward_kernel[
    DIM: Int, BATCH: Int,
    LV_MIN: Float64, LV_MAX: Float64,
](
    logits: LayoutTensor[DT, Layout.row_major(BATCH, 2 * DIM), MutAnyOrigin],
    targets: LayoutTensor[DT, Layout.row_major(BATCH, DIM), MutAnyOrigin],
    cache_diff: LayoutTensor[DT, Layout.row_major(BATCH, DIM), MutAnyOrigin],
    cache_inv_var: LayoutTensor[
        DT, Layout.row_major(BATCH, DIM), MutAnyOrigin
    ],
    cache_in_clamp: LayoutTensor[
        DT, Layout.row_major(BATCH, DIM), MutAnyOrigin
    ],
    partial_loss: LayoutTensor[DT, Layout.row_major(BATCH), MutAnyOrigin],
):
    var b = Int(global_idx.x)
    if b >= BATCH:
        return
    var lv_min = Scalar[DT](LV_MIN)
    var lv_max = Scalar[DT](LV_MAX)
    var row_total: Scalar[DT] = 0.0
    for i in range(DIM):
        var mu = rebind[Scalar[DT]](logits[b, i])
        var raw_lv = rebind[Scalar[DT]](logits[b, DIM + i])
        var y = rebind[Scalar[DT]](targets[b, i])
        var in_clamp = Scalar[DT](1.0)
        var lv = raw_lv
        if lv > lv_max:
            lv = lv_max
            in_clamp = Scalar[DT](0.0)
        elif lv < lv_min:
            lv = lv_min
            in_clamp = Scalar[DT](0.0)
        var inv_var = exp(-lv)
        var d = mu - y
        cache_diff[b, i] = d
        cache_inv_var[b, i] = inv_var
        cache_in_clamp[b, i] = in_clamp
        row_total += Scalar[DT](0.5) * d * d * inv_var + Scalar[DT](0.5) * lv
    partial_loss[b] = row_total


def _gauss_nll_vjp_kernel[DIM: Int, BATCH: Int](
    cache_diff: LayoutTensor[DT, Layout.row_major(BATCH, DIM), MutAnyOrigin],
    cache_inv_var: LayoutTensor[
        DT, Layout.row_major(BATCH, DIM), MutAnyOrigin
    ],
    cache_in_clamp: LayoutTensor[
        DT, Layout.row_major(BATCH, DIM), MutAnyOrigin
    ],
    grad_logits: LayoutTensor[
        DT, Layout.row_major(BATCH, 2 * DIM), MutAnyOrigin
    ],
):
    var idx = Int(global_idx.x)
    var total = BATCH * DIM
    if idx >= total:
        return
    var b = idx // DIM
    var i = idx % DIM
    var inv_b: Scalar[DT] = 1.0 / Scalar[DT](BATCH)
    var d = rebind[Scalar[DT]](cache_diff[b, i])
    var inv_v = rebind[Scalar[DT]](cache_inv_var[b, i])
    var ic = rebind[Scalar[DT]](cache_in_clamp[b, i])
    # d_loss/d_µ = (µ-y) · σ⁻² / BATCH.
    grad_logits[b, i] = d * inv_v * inv_b
    # d_loss/d_raw_lv = (½ - ½·d²·σ⁻²) / BATCH · in_clamp.
    grad_logits[b, DIM + i] = (
        Scalar[DT](0.5) - Scalar[DT](0.5) * d * d * inv_v
    ) * inv_b * ic


struct GaussianNLLLoss[
    DIM: Int,
    LOGVAR_MIN: Float64 = -10.0,
    LOGVAR_MAX: Float64 = -2.0,
](Loss):
    """Diagonal Gaussian NLL with clamped logvar bounds.

    `OUT_DIM = 2*DIM` so consumers that pre-allocate grad_logits via
    the `Loss.OUT_DIM` trait member size the buffer correctly."""

    comptime OUT_DIM: Int = 2 * Self.DIM

    # CPU caches.
    var cache_diff: List[Scalar[DT]]      # (µ - y) per element
    var cache_inv_var: List[Scalar[DT]]   # exp(-clamped_logvar) per element
    var cache_in_clamp: List[Scalar[DT]]  # 1.0 if in clamp range else 0.0

    # GPU caches — same semantics as CPU caches, plus per-row partial
    # loss + host buffer for the host-side sum + divide reduction.
    var cache_diff_dev: Optional[DeviceBuffer[DT]]
    var cache_diff_dev_n: Int
    var cache_inv_var_dev: Optional[DeviceBuffer[DT]]
    var cache_inv_var_dev_n: Int
    var cache_in_clamp_dev: Optional[DeviceBuffer[DT]]
    var cache_in_clamp_dev_n: Int
    var partial_loss_dev: Optional[DeviceBuffer[DT]]
    var partial_loss_host: Optional[HostBuffer[DT]]
    var partial_loss_n: Int

    var ts: TargetStorage

    def __init__(out self):
        self.cache_diff = List[Scalar[DT]]()
        self.cache_inv_var = List[Scalar[DT]]()
        self.cache_in_clamp = List[Scalar[DT]]()
        self.cache_diff_dev = None
        self.cache_diff_dev_n = 0
        self.cache_inv_var_dev = None
        self.cache_inv_var_dev_n = 0
        self.cache_in_clamp_dev = None
        self.cache_in_clamp_dev_n = 0
        self.partial_loss_dev = None
        self.partial_loss_host = None
        self.partial_loss_n = 0
        self.ts = TargetStorage.make_uninit()

    @staticmethod
    def make[target: StaticString](
        ctx: Optional[DeviceContext] = None,
    ) raises -> Self:
        """Unified CPU/GPU factory. `ctx=None` on CPU; required on GPU."""
        comptime assert target == "cpu" or target == "gpu", (
            "GaussianNLLLoss: target must be 'cpu' or 'gpu'"
        )
        var loss = Self()
        loss.ts = TargetStorage.make[target](ctx=ctx)
        comptime if target == "gpu":
            if not ctx:
                raise Error(
                    "GaussianNLLLoss.make[target='gpu']: ctx required"
                )
            # Eager 1-element placeholders so Optional unwrap is always
            # safe; lazy-grown to real size on first forward call.
            var ctx_v = ctx.value()
            loss.cache_diff_dev = ctx_v.enqueue_create_buffer[DT](1)
            loss.cache_inv_var_dev = ctx_v.enqueue_create_buffer[DT](1)
            loss.cache_in_clamp_dev = ctx_v.enqueue_create_buffer[DT](1)
            loss.partial_loss_dev = ctx_v.enqueue_create_buffer[DT](1)
            loss.partial_loss_host = ctx_v.enqueue_create_host_buffer[DT](1)
        return loss^

    def _ensure_cpu(mut self, batch: Int):
        var need = batch * Self.DIM
        if len(self.cache_diff) < need:
            self.cache_diff.resize(need, Scalar[DT](0.0))
            self.cache_inv_var.resize(need, Scalar[DT](0.0))
            self.cache_in_clamp.resize(need, Scalar[DT](0.0))

    def _ensure_buffers_gpu(mut self, batch: Int) raises:
        var ctx = self.ts.ctx.value()
        var c_needed = batch * Self.DIM
        ensure_gpu_buffer(
            self.cache_diff_dev, self.cache_diff_dev_n, c_needed, ctx,
        )
        ensure_gpu_buffer(
            self.cache_inv_var_dev, self.cache_inv_var_dev_n, c_needed, ctx,
        )
        ensure_gpu_buffer(
            self.cache_in_clamp_dev, self.cache_in_clamp_dev_n, c_needed, ctx,
        )
        if self.partial_loss_n < batch:
            self.partial_loss_dev = ctx.enqueue_create_buffer[DT](batch)
            self.partial_loss_host = ctx.enqueue_create_host_buffer[DT](batch)
            self.partial_loss_n = batch

    def forward[
        target: StaticString,
        BATCH: Int,
        POLICY: AMPPolicy = NoAMP,
    ](
        mut self,
        logits: TileTensor[
            dtype=DT, address_space=AddressSpace.GENERIC, element_size=1, ...
        ],
        targets: TileTensor[
            dtype=DT, address_space=AddressSpace.GENERIC, element_size=1, ...
        ],
    ) raises -> Scalar[DT]:
        comptime assert logits.flat_rank == 2, "logits rank-2"
        comptime assert targets.flat_rank == 2, "targets rank-2"
        assert_tag_for["GaussianNLLLoss", target](self.ts.target_tag)

        comptime if target == "cpu":
            self._ensure_cpu(BATCH)
            var lp = mptr(logits.ptr)
            var tp = mptr(targets.ptr)
            var diff_p = self.cache_diff.unsafe_ptr()
            var ivar_p = self.cache_inv_var.unsafe_ptr()
            var clamp_p = self.cache_in_clamp.unsafe_ptr()
            var lv_min = Scalar[DT](Self.LOGVAR_MIN)
            var lv_max = Scalar[DT](Self.LOGVAR_MAX)
            var total = Scalar[DT](0.0)
            for b in range(BATCH):
                # logits layout: [µ₀..µ_{DIM-1}, lv₀..lv_{DIM-1}]  per row.
                var lo = b * (2 * Self.DIM)
                var to = b * Self.DIM
                var co = b * Self.DIM
                for i in range(Self.DIM):
                    var mu = lp[lo + i]
                    var raw_lv = lp[lo + Self.DIM + i]
                    var y = tp[to + i]
                    var in_clamp = Scalar[DT](1.0)
                    var lv = raw_lv
                    if lv > lv_max:
                        lv = lv_max
                        in_clamp = Scalar[DT](0.0)
                    elif lv < lv_min:
                        lv = lv_min
                        in_clamp = Scalar[DT](0.0)
                    var inv_var = exp(-lv)
                    var d = mu - y
                    diff_p[co + i] = d
                    ivar_p[co + i] = inv_var
                    clamp_p[co + i] = in_clamp
                    total += Scalar[DT](0.5) * d * d * inv_var + Scalar[DT](0.5) * lv
            return total / Scalar[DT](BATCH)
        else:
            self._ensure_buffers_gpu(BATCH)
            var ctx = self.ts.ctx.value()
            comptime mat_in = Layout.row_major(BATCH, 2 * Self.DIM)
            comptime mat_out = Layout.row_major(BATCH, Self.DIM)
            comptime row_layout = Layout.row_major(BATCH)
            var lp = mptr(logits.ptr)
            var tp = mptr(targets.ptr)
            var logits_lt = LayoutTensor[DT, mat_in, MutAnyOrigin](lp)
            var targets_lt = LayoutTensor[DT, mat_out, MutAnyOrigin](tp)
            var diff_lt = LayoutTensor[DT, mat_out, MutAnyOrigin](
                self.cache_diff_dev.value(),
            )
            var ivar_lt = LayoutTensor[DT, mat_out, MutAnyOrigin](
                self.cache_inv_var_dev.value(),
            )
            var clamp_lt = LayoutTensor[DT, mat_out, MutAnyOrigin](
                self.cache_in_clamp_dev.value(),
            )
            var partial_lt = LayoutTensor[DT, row_layout, MutAnyOrigin](
                self.partial_loss_dev.value(),
            )
            comptime n_blocks = (BATCH + TPB - 1) // TPB
            comptime fwd_kernel = _gauss_nll_forward_kernel[
                Self.DIM, BATCH, Self.LOGVAR_MIN, Self.LOGVAR_MAX,
            ]
            ctx.enqueue_function[fwd_kernel](
                logits_lt, targets_lt,
                diff_lt, ivar_lt, clamp_lt,
                partial_lt,
                grid_dim=n_blocks, block_dim=TPB,
            )
            ctx.enqueue_copy(
                self.partial_loss_host.value(), self.partial_loss_dev.value(),
            )
            ctx.synchronize()
            var hp = self.partial_loss_host.value().unsafe_ptr()
            var total: Scalar[DT] = 0.0
            for b in range(BATCH):
                total += hp[b]
            return total / Scalar[DT](BATCH)

    def vjp[
        target: StaticString,
        BATCH: Int,
        POLICY: AMPPolicy = NoAMP,
    ](
        mut self,
        targets: TileTensor[
            dtype=DT, address_space=AddressSpace.GENERIC, element_size=1, ...
        ],
        mut grad_logits: TileTensor[
            mut=True, dtype=DT, address_space=AddressSpace.GENERIC,
            element_size=1, ...,
        ],
    ) raises:
        comptime assert targets.flat_rank == 2, "targets rank-2"
        comptime assert grad_logits.flat_rank == 2, "grad_logits rank-2"
        assert_tag_for["GaussianNLLLoss", target](self.ts.target_tag)

        comptime if target == "cpu":
            var gp = mptr(grad_logits.ptr)
            var diff_p = self.cache_diff.unsafe_ptr()
            var ivar_p = self.cache_inv_var.unsafe_ptr()
            var clamp_p = self.cache_in_clamp.unsafe_ptr()
            var inv_b = Scalar[DT](1.0) / Scalar[DT](BATCH)
            for b in range(BATCH):
                var go = b * (2 * Self.DIM)
                var co = b * Self.DIM
                for i in range(Self.DIM):
                    var d = diff_p[co + i]            # (µ - y)
                    var inv_v = ivar_p[co + i]        # exp(-lv_clamped)
                    var ic = clamp_p[co + i]          # 1 in-clamp, 0 clamped
                    # d_loss/d_µ = (µ-y) · σ⁻² / BATCH.
                    gp[go + i] = d * inv_v * inv_b
                    # d_loss/d_raw_lv = (½ - ½·d²·σ⁻²) / BATCH · in_clamp.
                    var d_lv = (
                        Scalar[DT](0.5)
                        - Scalar[DT](0.5) * d * d * inv_v
                    ) * inv_b * ic
                    gp[go + Self.DIM + i] = d_lv
        else:
            var ctx = self.ts.ctx.value()
            comptime mat_out = Layout.row_major(BATCH, Self.DIM)
            comptime mat_grad = Layout.row_major(BATCH, 2 * Self.DIM)
            var gp = mptr(grad_logits.ptr)
            var grad_lt = LayoutTensor[DT, mat_grad, MutAnyOrigin](gp)
            var diff_lt = LayoutTensor[DT, mat_out, MutAnyOrigin](
                self.cache_diff_dev.value(),
            )
            var ivar_lt = LayoutTensor[DT, mat_out, MutAnyOrigin](
                self.cache_inv_var_dev.value(),
            )
            var clamp_lt = LayoutTensor[DT, mat_out, MutAnyOrigin](
                self.cache_in_clamp_dev.value(),
            )
            comptime total = BATCH * Self.DIM
            comptime n_blocks = (total + TPB - 1) // TPB
            comptime vjp_kernel = _gauss_nll_vjp_kernel[Self.DIM, BATCH]
            ctx.enqueue_function[vjp_kernel](
                diff_lt, ivar_lt, clamp_lt, grad_lt,
                grid_dim=n_blocks, block_dim=TPB,
            )
