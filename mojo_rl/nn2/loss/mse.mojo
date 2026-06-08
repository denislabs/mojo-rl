"""MSELoss[OUT_DIM] — 0.5 * mean((logits - targets)^2), conforms to `Loss` trait.

Used by PPO critic for value-fn regression: logits = V(s) (BATCH × 1),
targets = returns (BATCH × 1). The 0.5 factor matches CleanRL convention
so the gradient is clean: d_L / d_logits = (logits - targets) / BATCH.

For multi-output regression (OUT_DIM > 1), entries are treated as
independent and averaged across the batch (no per-dim averaging — same
convention as PyTorch `MSELoss(reduction='mean')` which also averages
over both batch and feature dims, but here we divide by BATCH only to
match the PPO critic gradient).

AMP: POLICY accepted but ignored (loss math is fp32-only).
"""

from std.gpu import global_idx, thread_idx
from std.gpu.primitives import block
from std.gpu.host import DeviceContext, DeviceBuffer, HostBuffer
from std.gpu.memory import AddressSpace
from layout import Layout, LayoutTensor, TileTensor, row_major

from ..constants import DT, CPU_SIMD_W, TPB, TPB_REDUCE
from ..core.module import mptr
from ..core import Loss, AMPPolicy, NoAMP
from ..core.target_storage import require_ctx, TargetStorage, assert_tag_for


# ──────────────────────────────────────────────────────────────────────────
# GPU kernels.
# ──────────────────────────────────────────────────────────────────────────


def _mse_forward_kernel[BATCH: Int, DIM: Int](
    logits: LayoutTensor[DT, Layout.row_major(BATCH, DIM), MutAnyOrigin],
    targets: LayoutTensor[DT, Layout.row_major(BATCH, DIM), MutAnyOrigin],
    partial_loss: LayoutTensor[DT, Layout.row_major(BATCH), MutAnyOrigin],
):
    var b = Int(global_idx.x)
    if b < BATCH:
        var s: Scalar[DT] = 0.0
        for j in range(DIM):
            var d = rebind[Scalar[DT]](logits[b, j]) - rebind[Scalar[DT]](targets[b, j])
            s += Scalar[DT](0.5) * d * d
        partial_loss[b] = s


def _mse_reduce_add_kernel[BATCH: Int](
    partial_loss: UnsafePointer[Scalar[DT], MutAnyOrigin],
    acc: UnsafePointer[Scalar[DT], MutAnyOrigin],
):
    """Single-block `block.sum` reduce of `partial_loss[0..BATCH]`, then
    thread 0 adds `total/BATCH` into `acc[0]` and `+1` into `acc[1]`. No D2H
    — the mean loss stays device-resident for CUDA-graph capture; the host
    reads `acc` once per `diag_every` flush. Launch with `grid_dim=1,
    block_dim=TPB_REDUCE`. Mirrors grad_clip's `_sum_sq_partial_kernel`.
    Reduction order differs from the CPU left-to-right sweep (~1e-5 in fp32)
    but the loss scalar feeds only metrics, never the gradient, so training
    stays bit-identical."""
    var t = Int(thread_idx.x)
    var my_sum: Scalar[DT] = 0.0
    var k = t
    while k < BATCH:
        my_sum += partial_loss[k]
        k += TPB_REDUCE
    var total = block.sum[block_size=TPB_REDUCE, broadcast=False](val=my_sum)
    if t == 0:
        acc[0] = acc[0] + total[0] / Scalar[DT](BATCH)
        acc[1] = acc[1] + Scalar[DT](1.0)


def _mse_backward_kernel[BATCH: Int, DIM: Int](
    logits: LayoutTensor[DT, Layout.row_major(BATCH, DIM), MutAnyOrigin],
    targets: LayoutTensor[DT, Layout.row_major(BATCH, DIM), MutAnyOrigin],
    grad_logits: LayoutTensor[DT, Layout.row_major(BATCH, DIM), MutAnyOrigin],
):
    var idx = Int(global_idx.x)
    var total = BATCH * DIM
    if idx < total:
        var b = idx // DIM
        var j = idx % DIM
        var inv_batch: Scalar[DT] = 1.0 / Scalar[DT](BATCH)
        grad_logits[b, j] = (
            rebind[Scalar[DT]](logits[b, j])
            - rebind[Scalar[DT]](targets[b, j])
        ) * inv_batch


# ──────────────────────────────────────────────────────────────────────────
# MSELoss — needs `logits` cached on backward (subtract targets from cached
# logits). The trait sig has `backward(targets, grad_logits)` — `logits`
# isn't a backward arg — so we cache logits on forward and re-use.
# ──────────────────────────────────────────────────────────────────────────


struct MSELoss[DIM: Int](Loss):
    comptime OUT_DIM = Self.DIM

    # CPU
    var cache_logits: List[Scalar[DT]]
    # GPU
    var cache_logits_dev: Optional[DeviceBuffer[DT]]
    var cache_logits_dev_n: Int
    var partial_loss_dev: Optional[DeviceBuffer[DT]]
    var partial_loss_host: Optional[HostBuffer[DT]]
    var partial_loss_n: Int

    # Slice 3 — device-resident loss accumulator for CUDA-graph capture.
    # `forward_accumulate` reduces on device and adds (mean_loss, +1) into
    # this [2]-buffer ([sum_of_means, count]) with NO per-step D2H. `read_accum`
    # D2Hs once at flush cadence; `reset_accum` zeroes it. CPU mirror:
    # `_acc_sum`/`_acc_n` host scalars.
    var loss_acc_dev: Optional[DeviceBuffer[DT]]
    var _acc_sum: Scalar[DT]
    var _acc_n: Int

    var ts: TargetStorage

    def __init__(out self):
        self.cache_logits = List[Scalar[DT]]()
        self.cache_logits_dev = None
        self.cache_logits_dev_n = 0
        self.partial_loss_dev = None
        self.partial_loss_host = None
        self.partial_loss_n = 0
        self.loss_acc_dev = None
        self._acc_sum = Scalar[DT](0.0)
        self._acc_n = 0
        self.ts = TargetStorage.make_uninit()

    @staticmethod
    def make[target: StaticString](
        ctx: Optional[DeviceContext] = None,
    ) raises -> Self:
        """Unified CPU/GPU factory. `ctx=None` on CPU; required on GPU."""
        comptime assert target == "cpu" or target == "gpu", (
            "MSELoss: target must be 'cpu' or 'gpu'"
        )
        var loss = Self()
        loss.ts = TargetStorage.make[target](ctx=ctx)
        comptime if target == "gpu":
            var ctx_v = require_ctx["MSELoss.make[target='gpu']"](ctx)
            loss.cache_logits_dev = ctx_v.enqueue_create_buffer[DT](1)
            loss.partial_loss_dev = ctx_v.enqueue_create_buffer[DT](1)
            loss.partial_loss_host = ctx_v.enqueue_create_host_buffer[DT](1)
            var acc_real = ctx_v.enqueue_create_buffer[DT](2)
            acc_real.enqueue_fill(0.0)
            loss.loss_acc_dev = acc_real^
        return loss^

    def _ensure_cache_cpu(mut self, batch: Int):
        var needed = batch * Self.DIM
        if len(self.cache_logits) < needed:
            self.cache_logits.resize(needed, 0.0)

    def _ensure_buffers_gpu(mut self, batch: Int) raises:
        var ctx = self.ts.ctx.value()
        var c_needed = batch * Self.DIM
        if self.cache_logits_dev_n < c_needed:
            self.cache_logits_dev = ctx.enqueue_create_buffer[DT](c_needed)
            self.cache_logits_dev_n = c_needed
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
        assert_tag_for["MSELoss", target](self.ts.target_tag)

        comptime if target == "cpu":
            # SIMD path. Flat sweep over BATCH * DIM — sum is associative in
            # fp32 modulo rounding, so row structure doesn't matter (the
            # 1/BATCH normalization dwarfs the rounding delta).
            self._ensure_cache_cpu(BATCH)
            var lp = mptr(logits.ptr)
            var tp = mptr(targets.ptr)
            var cp = self.cache_logits.unsafe_ptr()
            comptime N = BATCH * Self.DIM
            var acc_v = SIMD[DT, CPU_SIMD_W](0)
            var half_v = SIMD[DT, CPU_SIMD_W](0.5)
            var k = 0
            while k + CPU_SIMD_W <= N:
                var l = lp.load[width=CPU_SIMD_W](k)
                var t = tp.load[width=CPU_SIMD_W](k)
                cp.store(k, l)
                var d = l - t
                acc_v += half_v * d * d
                k += CPU_SIMD_W
            var total: Scalar[DT] = acc_v.reduce_add()
            while k < N:
                cp[k] = lp[k]
                var d = lp[k] - tp[k]
                total += Scalar[DT](0.5) * d * d
                k += 1
            return total / Scalar[DT](BATCH)
        else:
            self._ensure_buffers_gpu(BATCH)
            var ctx = self.ts.ctx.value()
            comptime mat_layout = Layout.row_major(BATCH, Self.DIM)
            comptime row_layout = Layout.row_major(BATCH)
            var lp_w = mptr(logits.ptr)
            var tp_w = mptr(targets.ptr)
            var logits_lt = LayoutTensor[DT, mat_layout, MutAnyOrigin](lp_w)
            var targets_lt = LayoutTensor[DT, mat_layout, MutAnyOrigin](tp_w)
            var cache_lt = LayoutTensor[DT, mat_layout, MutAnyOrigin](
                self.cache_logits_dev.value()
            )
            var partial_lt = LayoutTensor[DT, row_layout, MutAnyOrigin](
                self.partial_loss_dev.value()
            )
            # Copy logits to cache so backward kernel can reference them
            # without depending on the caller keeping logits buffer alive.
            ctx.enqueue_copy(self.cache_logits_dev.value(), lp_w)
            comptime TPB = TPB_REDUCE
            comptime n_blocks = (BATCH + TPB - 1) // TPB
            comptime kernel = _mse_forward_kernel[BATCH, Self.DIM]
            ctx.enqueue_function[kernel](
                logits_lt, targets_lt, partial_lt,
                grid_dim=n_blocks, block_dim=TPB,
            )
            ctx.enqueue_copy(
                self.partial_loss_host.value(), self.partial_loss_dev.value()
            )
            ctx.synchronize()
            var total: Scalar[DT] = 0.0
            var hp = self.partial_loss_host.value().unsafe_ptr()
            for b in range(BATCH):
                total += hp[b]
            return total / Scalar[DT](BATCH)

    # ──────────────────────────────────────────────────────────────────
    # Slice 3 — device-resident accumulate path (CUDA-graph capturable).
    # `forward_accumulate` computes the same mean loss as `forward` but adds
    # it into the on-device `loss_acc_dev` instead of returning a host scalar
    # — NO per-step D2H. Caller reads it via `read_accum` at flush cadence.
    # The vjp (called separately after this) re-uses the cached logits, same
    # as after a plain `forward`.
    # ──────────────────────────────────────────────────────────────────
    def forward_accumulate[
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
    ) raises:
        comptime assert logits.flat_rank == 2, "logits rank-2"
        comptime assert targets.flat_rank == 2, "targets rank-2"
        assert_tag_for["MSELoss", target](self.ts.target_tag)

        comptime if target == "cpu":
            # CPU is not a capture target — reuse `forward` exactly, then
            # accumulate the host scalar.
            var L = self.forward[target, BATCH, POLICY](logits, targets)
            self._acc_sum += L
            self._acc_n += 1
        else:
            self._ensure_buffers_gpu(BATCH)
            var ctx = self.ts.ctx.value()
            comptime mat_layout = Layout.row_major(BATCH, Self.DIM)
            comptime row_layout = Layout.row_major(BATCH)
            var lp_w = mptr(logits.ptr)
            var tp_w = mptr(targets.ptr)
            var logits_lt = LayoutTensor[DT, mat_layout, MutAnyOrigin](lp_w)
            var targets_lt = LayoutTensor[DT, mat_layout, MutAnyOrigin](tp_w)
            var partial_lt = LayoutTensor[DT, row_layout, MutAnyOrigin](
                self.partial_loss_dev.value()
            )
            # Cache logits for the subsequent vjp (matches `forward`).
            ctx.enqueue_copy(self.cache_logits_dev.value(), lp_w)
            comptime TPB = TPB_REDUCE
            comptime n_blocks = (BATCH + TPB - 1) // TPB
            comptime fwd_k = _mse_forward_kernel[BATCH, Self.DIM]
            ctx.enqueue_function[fwd_k](
                logits_lt, targets_lt, partial_lt,
                grid_dim=n_blocks, block_dim=TPB,
            )
            # Device reduce + accumulate — no D2H.
            comptime red_k = _mse_reduce_add_kernel[BATCH]
            ctx.enqueue_function[red_k](
                self.partial_loss_dev.value().unsafe_ptr(),
                self.loss_acc_dev.value().unsafe_ptr(),
                grid_dim=1, block_dim=TPB,
            )

    def reset_accum[target: StaticString](mut self) raises:
        """Zero the (sum, count) accumulator. Call once per `diag_every`
        window after `read_accum`. Outside the capture region."""
        comptime if target == "cpu":
            self._acc_sum = Scalar[DT](0.0)
            self._acc_n = 0
        else:
            self.loss_acc_dev.value().enqueue_fill(0.0)

    def read_accum[target: StaticString](mut self) raises -> Scalar[DT]:
        """Return the mean accumulated loss (sum / count) over the window.
        GPU path D2Hs the [2]-buffer once here — flush cadence only, NOT in
        the per-step hot loop. Returns 0 if count == 0."""
        comptime if target == "cpu":
            if self._acc_n == 0:
                return Scalar[DT](0.0)
            return self._acc_sum / Scalar[DT](self._acc_n)
        else:
            var ctx = self.ts.ctx.value()
            var h = ctx.enqueue_create_host_buffer[DT](2)
            ctx.enqueue_copy(h, self.loss_acc_dev.value())
            ctx.synchronize()
            var s = h.unsafe_ptr()[0]
            var n = h.unsafe_ptr()[1]
            if n == Scalar[DT](0.0):
                return Scalar[DT](0.0)
            return s / n

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
        assert_tag_for["MSELoss", target](self.ts.target_tag)

        comptime if target == "cpu":
            # SIMD path.
            var cp = self.cache_logits.unsafe_ptr()
            var tp = mptr(targets.ptr)
            var gp = mptr(grad_logits.ptr)
            var inv_batch_v = SIMD[DT, CPU_SIMD_W](1.0 / Scalar[DT](BATCH))
            comptime N = BATCH * Self.DIM
            var k = 0
            while k + CPU_SIMD_W <= N:
                var c = cp.load[width=CPU_SIMD_W](k)
                var t = tp.load[width=CPU_SIMD_W](k)
                gp.store(k, (c - t) * inv_batch_v)
                k += CPU_SIMD_W
            var inv_batch: Scalar[DT] = 1.0 / Scalar[DT](BATCH)
            while k < N:
                gp[k] = (cp[k] - tp[k]) * inv_batch
                k += 1
        else:
            var ctx = self.ts.ctx.value()
            comptime mat_layout = Layout.row_major(BATCH, Self.DIM)
            var tp_w = mptr(targets.ptr)
            var gp_w = mptr(grad_logits.ptr)
            var cache_lt = LayoutTensor[DT, mat_layout, MutAnyOrigin](
                self.cache_logits_dev.value()
            )
            var targets_lt = LayoutTensor[DT, mat_layout, MutAnyOrigin](tp_w)
            var grad_lt = LayoutTensor[DT, mat_layout, MutAnyOrigin](gp_w)
            comptime n_blocks = (BATCH * Self.DIM + TPB - 1) // TPB
            comptime kernel = _mse_backward_kernel[BATCH, Self.DIM]
            ctx.enqueue_function[kernel](
                cache_lt, targets_lt, grad_lt,
                grid_dim=n_blocks, block_dim=TPB,
            )
