"""MSELoss[OUT_DIM] — 0.5 * mean((logits - targets)^2), conforms to `Loss` trait.

Used by PPO critic for value-fn regression: logits = V(s) (BATCH × 1),
targets = returns (BATCH × 1). The 0.5 factor matches CleanRL convention
so the gradient is clean: d_L / d_logits = (logits - targets) / BATCH.

For multi-output regression (OUT_DIM > 1), entries are treated as
independent and averaged across the batch (no per-dim averaging — same
convention as PyTorch `MSELoss(reduction='mean')` which also averages
over both batch and feature dims, but here we divide by BATCH only to
match the PPO critic gradient).

Phase 6.3. AMP: POLICY accepted but ignored (loss math is fp32-only).
"""

from std.gpu import global_idx
from std.gpu.host import DeviceContext, DeviceBuffer, HostBuffer
from layout import Layout, LayoutTensor, TileTensor, TensorLayout, row_major

from ..constants import DT, CPU_SIMD_W
from ..core import (
    Loss, AMPPolicy, NoAMP,
    TARGET_UNINIT, TARGET_CPU, TARGET_GPU, target_tag_for,
)


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
    var ctx: Optional[DeviceContext]

    var _target_tag: Int8

    def __init__(out self):
        self.cache_logits = List[Scalar[DT]]()
        self.cache_logits_dev = None
        self.cache_logits_dev_n = 0
        self.partial_loss_dev = None
        self.partial_loss_host = None
        self.partial_loss_n = 0
        self.ctx = None
        self._target_tag = TARGET_UNINIT

    @staticmethod
    def make[target: StaticString]() raises -> Self:
        comptime assert target == "cpu", (
            "MSELoss.make[target='gpu'] requires a DeviceContext"
        )
        var loss = Self()
        loss._target_tag = TARGET_CPU
        return loss^

    @staticmethod
    def make[target: StaticString](ctx: DeviceContext) raises -> Self:
        comptime assert target == "gpu", (
            "MSELoss.make[target='cpu'](ctx) — drop ctx for CPU"
        )
        var loss = Self()
        loss.cache_logits_dev = ctx.enqueue_create_buffer[DT](1)
        loss.partial_loss_dev = ctx.enqueue_create_buffer[DT](1)
        loss.partial_loss_host = ctx.enqueue_create_host_buffer[DT](1)
        loss.ctx = ctx
        loss._target_tag = TARGET_GPU
        return loss^

    def _assert_tag[target: StaticString](self) raises:
        comptime expected = target_tag_for[target]()
        if self._target_tag != expected:
            raise Error(
                "MSELoss: method called with [target='" + String(target)
                + "'] but loss was make'd for a different target "
                + "(tag=" + String(Int(self._target_tag)) + ")"
            )

    def _ensure_cache_cpu(mut self, batch: Int):
        var needed = batch * Self.DIM
        if len(self.cache_logits) < needed:
            self.cache_logits.resize(needed, 0.0)

    def _ensure_buffers_gpu(mut self, batch: Int) raises:
        var ctx = self.ctx.value()
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
        LL: TensorLayout,
        LT: TensorLayout,
        OL: MutOrigin,
        OT: MutOrigin,
        POLICY: AMPPolicy = NoAMP,
    ](
        mut self,
        logits: TileTensor[DT, LL, OL],
        targets: TileTensor[DT, LT, OT],
    ) raises -> Scalar[DT]:
        comptime assert logits.flat_rank == 2, "logits rank-2"
        comptime assert targets.flat_rank == 2, "targets rank-2"
        self._assert_tag[target]()

        comptime if target == "cpu":
            # Phase 8.0: SIMD path. Flat sweep over BATCH * DIM since the
            # reduction doesn't care about row structure (sum is associative
            # in fp32 modulo rounding — bit-changes vs scalar are negligible
            # against the 1/BATCH normalization).
            self._ensure_cache_cpu(BATCH)
            var logits_w = rebind[TileTensor[DT, LL, MutAnyOrigin]](logits)
            var targets_w = rebind[TileTensor[DT, LT, MutAnyOrigin]](targets)
            var lp = logits_w.ptr
            var tp = targets_w.ptr
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
            var ctx = self.ctx.value()
            comptime mat_layout = Layout.row_major(BATCH, Self.DIM)
            comptime row_layout = Layout.row_major(BATCH)
            var logits_w = rebind[TileTensor[DT, LL, MutAnyOrigin]](logits)
            var targets_w = rebind[TileTensor[DT, LT, MutAnyOrigin]](targets)
            var logits_lt = LayoutTensor[DT, mat_layout, MutAnyOrigin](logits_w.ptr)
            var targets_lt = LayoutTensor[DT, mat_layout, MutAnyOrigin](targets_w.ptr)
            var cache_lt = LayoutTensor[DT, mat_layout, MutAnyOrigin](
                self.cache_logits_dev.value()
            )
            var partial_lt = LayoutTensor[DT, row_layout, MutAnyOrigin](
                self.partial_loss_dev.value()
            )
            # Copy logits to cache so backward kernel can reference them
            # without depending on the caller keeping logits buffer alive.
            ctx.enqueue_copy(self.cache_logits_dev.value(), logits_w.ptr)
            comptime TPB = 64
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

    def backward[
        target: StaticString,
        BATCH: Int,
        LT: TensorLayout,
        LG: TensorLayout,
        OT: MutOrigin,
        OG: MutOrigin,
        POLICY: AMPPolicy = NoAMP,
    ](
        mut self,
        targets: TileTensor[DT, LT, OT],
        mut grad_logits: TileTensor[DT, LG, OG],
    ) raises:
        comptime assert targets.flat_rank == 2, "targets rank-2"
        comptime assert grad_logits.flat_rank == 2, "grad_logits rank-2"
        self._assert_tag[target]()

        comptime if target == "cpu":
            # Phase 8.0: SIMD path.
            var targets_w = rebind[TileTensor[DT, LT, MutAnyOrigin]](targets)
            var grad_w = rebind[TileTensor[DT, LG, MutAnyOrigin]](grad_logits)
            var cp = self.cache_logits.unsafe_ptr()
            var tp = targets_w.ptr
            var gp = grad_w.ptr
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
            var ctx = self.ctx.value()
            comptime mat_layout = Layout.row_major(BATCH, Self.DIM)
            var targets_w = rebind[TileTensor[DT, LT, MutAnyOrigin]](targets)
            var grad_w = rebind[TileTensor[DT, LG, MutAnyOrigin]](grad_logits)
            var cache_lt = LayoutTensor[DT, mat_layout, MutAnyOrigin](
                self.cache_logits_dev.value()
            )
            var targets_lt = LayoutTensor[DT, mat_layout, MutAnyOrigin](targets_w.ptr)
            var grad_lt = LayoutTensor[DT, mat_layout, MutAnyOrigin](grad_w.ptr)
            comptime TPB = 128
            comptime n_blocks = (BATCH * Self.DIM + TPB - 1) // TPB
            comptime kernel = _mse_backward_kernel[BATCH, Self.DIM]
            ctx.enqueue_function[kernel](
                cache_lt, targets_lt, grad_lt,
                grid_dim=n_blocks, block_dim=TPB,
            )
