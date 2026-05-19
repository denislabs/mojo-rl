"""CrossEntropyLoss[N_CLASSES] — softmax + cross-entropy, numerically stable.

Phase 2.4: target is a comptime method param. The loss is Defaultable;
`make[target]` populates the matching scratch buffers and stamps the tag.
"""

from std.math import exp, log
from std.gpu import global_idx
from std.gpu.host import DeviceContext, DeviceBuffer, HostBuffer
from layout import Layout, LayoutTensor, TileTensor, TensorLayout, row_major

from ..constants import DT
from ..core import (
    Loss, TARGET_UNINIT, TARGET_CPU, TARGET_GPU, target_tag_for,
)


# ──────────────────────────────────────────────────────────────────────────
# GPU kernels.
# ──────────────────────────────────────────────────────────────────────────

def _ce_forward_kernel[
    BATCH: Int, N_CLASSES: Int,
](
    logits: LayoutTensor[DT, Layout.row_major(BATCH, N_CLASSES), MutAnyOrigin],
    targets: LayoutTensor[DT, Layout.row_major(BATCH, N_CLASSES), MutAnyOrigin],
    softmax: LayoutTensor[DT, Layout.row_major(BATCH, N_CLASSES), MutAnyOrigin],
    partial_loss: LayoutTensor[DT, Layout.row_major(BATCH), MutAnyOrigin],
):
    var b = Int(global_idx.x)
    if b < BATCH:
        var m = rebind[Scalar[DT]](logits[b, 0])
        for c in range(1, N_CLASSES):
            var v = rebind[Scalar[DT]](logits[b, c])
            if v > m:
                m = v
        var sum_exp: Scalar[DT] = 0.0
        for c in range(N_CLASSES):
            sum_exp += exp(rebind[Scalar[DT]](logits[b, c]) - m)
        var lse = m + log(sum_exp)
        var sample_loss: Scalar[DT] = 0.0
        for c in range(N_CLASSES):
            var x = rebind[Scalar[DT]](logits[b, c])
            softmax[b, c] = exp(x - lse)
            sample_loss += -rebind[Scalar[DT]](targets[b, c]) * (x - lse)
        partial_loss[b] = sample_loss


def _ce_backward_kernel[
    BATCH: Int, N_CLASSES: Int,
](
    softmax: LayoutTensor[DT, Layout.row_major(BATCH, N_CLASSES), MutAnyOrigin],
    targets: LayoutTensor[DT, Layout.row_major(BATCH, N_CLASSES), MutAnyOrigin],
    grad_logits: LayoutTensor[DT, Layout.row_major(BATCH, N_CLASSES), MutAnyOrigin],
):
    var idx = Int(global_idx.x)
    var total = BATCH * N_CLASSES
    if idx < total:
        var b = idx // N_CLASSES
        var c = idx % N_CLASSES
        var inv_batch: Scalar[DT] = 1.0 / Scalar[DT](BATCH)
        var sm = rebind[Scalar[DT]](softmax[b, c])
        var tg = rebind[Scalar[DT]](targets[b, c])
        grad_logits[b, c] = (sm - tg) * inv_batch


# ──────────────────────────────────────────────────────────────────────────
# CrossEntropyLoss — method-level target.
# ──────────────────────────────────────────────────────────────────────────

struct CrossEntropyLoss[N_CLASSES: Int](Loss):
    comptime OUT_DIM = Self.N_CLASSES

    # CPU
    var softmax: List[Scalar[DT]]
    # GPU
    var softmax_dev: Optional[DeviceBuffer[DT]]
    var softmax_dev_n: Int
    var partial_loss_dev: Optional[DeviceBuffer[DT]]
    var partial_loss_host: Optional[HostBuffer[DT]]
    var partial_loss_n: Int
    var ctx: Optional[DeviceContext]

    var _target_tag: Int8

    def __init__(out self):
        self.softmax = List[Scalar[DT]]()
        self.softmax_dev = None
        self.softmax_dev_n = 0
        self.partial_loss_dev = None
        self.partial_loss_host = None
        self.partial_loss_n = 0
        self.ctx = None
        self._target_tag = TARGET_UNINIT

    @staticmethod
    def make[target: StaticString]() raises -> Self:
        """CPU factory."""
        comptime assert target == "cpu", (
            "CrossEntropyLoss.make[target='gpu'] requires a DeviceContext"
        )
        var loss = Self()
        loss._target_tag = TARGET_CPU
        return loss^

    @staticmethod
    def make[target: StaticString](ctx: DeviceContext) raises -> Self:
        """GPU factory."""
        comptime assert target == "gpu", (
            "CrossEntropyLoss.make[target='cpu'](ctx) — drop ctx for CPU"
        )
        var loss = Self()
        loss.softmax_dev = ctx.enqueue_create_buffer[DT](1)
        loss.partial_loss_dev = ctx.enqueue_create_buffer[DT](1)
        loss.partial_loss_host = ctx.enqueue_create_host_buffer[DT](1)
        loss.ctx = ctx
        loss._target_tag = TARGET_GPU
        return loss^

    def _assert_tag[target: StaticString](self) raises:
        comptime expected = target_tag_for[target]()
        if self._target_tag != expected:
            raise Error(
                "CrossEntropyLoss: method called with [target='" + String(target)
                + "'] but loss was make'd for a different target "
                + "(tag=" + String(Int(self._target_tag)) + ")"
            )

    def _ensure_cache_cpu(mut self, batch: Int):
        var needed = batch * Self.N_CLASSES
        if len(self.softmax) < needed:
            self.softmax.resize(needed, 0.0)

    def _ensure_buffers_gpu(mut self, batch: Int) raises:
        var ctx = self.ctx.value()
        var sm_needed = batch * Self.N_CLASSES
        if self.softmax_dev_n < sm_needed:
            self.softmax_dev = ctx.enqueue_create_buffer[DT](sm_needed)
            self.softmax_dev_n = sm_needed
        if self.partial_loss_n < batch:
            self.partial_loss_dev  = ctx.enqueue_create_buffer[DT](batch)
            self.partial_loss_host = ctx.enqueue_create_host_buffer[DT](batch)
            self.partial_loss_n = batch

    def forward[
        target: StaticString,
        BATCH: Int,
        LL: TensorLayout,
        LT: TensorLayout,
        OL: MutOrigin,
        OT: MutOrigin,
    ](
        mut self,
        logits: TileTensor[DT, LL, OL],
        targets: TileTensor[DT, LT, OT],
    ) raises -> Scalar[DT]:
        comptime assert logits.flat_rank  == 2, "logits must be rank-2"
        comptime assert targets.flat_rank == 2, "targets must be rank-2"
        self._assert_tag[target]()

        comptime if target == "cpu":
            self._ensure_cache_cpu(BATCH)
            var softmax = TileTensor(self.softmax, row_major[BATCH, Self.N_CLASSES]())
            var total_loss: Scalar[DT] = 0.0
            for b in range(BATCH):
                var m = logits[b, 0]
                for c in range(1, Self.N_CLASSES):
                    if logits[b, c] > m:
                        m = logits[b, c]
                var sum_exp: Scalar[DT] = 0.0
                for c in range(Self.N_CLASSES):
                    sum_exp += exp(logits[b, c] - m)
                var lse = m + log(sum_exp)
                for c in range(Self.N_CLASSES):
                    softmax[b, c] = exp(logits[b, c] - lse)
                    total_loss += -targets[b, c] * (logits[b, c] - lse)
            return total_loss / Scalar[DT](BATCH)
        else:
            self._ensure_buffers_gpu(BATCH)
            var ctx = self.ctx.value()
            comptime mat_layout = Layout.row_major(BATCH, Self.N_CLASSES)
            comptime row_layout = Layout.row_major(BATCH)
            var logits_w  = rebind[TileTensor[DT, LL, MutAnyOrigin]](logits)
            var targets_w = rebind[TileTensor[DT, LT, MutAnyOrigin]](targets)
            var logits_lt  = LayoutTensor[DT, mat_layout, MutAnyOrigin](logits_w.ptr)
            var targets_lt = LayoutTensor[DT, mat_layout, MutAnyOrigin](targets_w.ptr)
            var softmax_lt = LayoutTensor[DT, mat_layout, MutAnyOrigin](self.softmax_dev.value())
            var partial_lt = LayoutTensor[DT, row_layout, MutAnyOrigin](self.partial_loss_dev.value())
            comptime TPB = 64
            comptime n_blocks = (BATCH + TPB - 1) // TPB
            comptime kernel = _ce_forward_kernel[BATCH, Self.N_CLASSES]
            ctx.enqueue_function[kernel](
                logits_lt, targets_lt, softmax_lt, partial_lt,
                grid_dim=n_blocks, block_dim=TPB,
            )
            ctx.enqueue_copy(self.partial_loss_host.value(), self.partial_loss_dev.value())
            ctx.synchronize()
            var total: Scalar[DT] = 0.0
            var host_ptr = self.partial_loss_host.value().unsafe_ptr()
            for b in range(BATCH):
                total += host_ptr[b]
            return total / Scalar[DT](BATCH)

    def backward[
        target: StaticString,
        BATCH: Int,
        LT: TensorLayout,
        LG: TensorLayout,
        OT: MutOrigin,
        OG: MutOrigin,
    ](
        mut self,
        targets: TileTensor[DT, LT, OT],
        mut grad_logits: TileTensor[DT, LG, OG],
    ) raises:
        comptime assert targets.flat_rank     == 2, "targets must be rank-2"
        comptime assert grad_logits.flat_rank == 2, "grad_logits must be rank-2"
        self._assert_tag[target]()

        comptime if target == "cpu":
            var softmax = TileTensor(self.softmax, row_major[BATCH, Self.N_CLASSES]())
            var inv_batch: Scalar[DT] = 1.0 / Scalar[DT](BATCH)
            for b in range(BATCH):
                for c in range(Self.N_CLASSES):
                    grad_logits[b, c] = (softmax[b, c] - targets[b, c]) * inv_batch
        else:
            var ctx = self.ctx.value()
            comptime mat_layout = Layout.row_major(BATCH, Self.N_CLASSES)
            var targets_w     = rebind[TileTensor[DT, LT, MutAnyOrigin]](targets)
            var grad_logits_w = rebind[TileTensor[DT, LG, MutAnyOrigin]](grad_logits)
            var softmax_lt = LayoutTensor[DT, mat_layout, MutAnyOrigin](self.softmax_dev.value())
            var targets_lt = LayoutTensor[DT, mat_layout, MutAnyOrigin](targets_w.ptr)
            var grad_lt    = LayoutTensor[DT, mat_layout, MutAnyOrigin](grad_logits_w.ptr)
            comptime TPB = 128
            comptime n_blocks = (BATCH * Self.N_CLASSES + TPB - 1) // TPB
            comptime kernel = _ce_backward_kernel[BATCH, Self.N_CLASSES]
            ctx.enqueue_function[kernel](
                softmax_lt, targets_lt, grad_lt, grid_dim=n_blocks, block_dim=TPB,
            )
