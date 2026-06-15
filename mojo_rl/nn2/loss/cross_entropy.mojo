"""CrossEntropyLoss[N_CLASSES] — softmax + cross-entropy, numerically stable.

`target` is a comptime method param. The loss is Defaultable;
`make[target]` populates the matching scratch buffers and stamps the tag.
"""

from std.math import exp, log
from std.gpu import global_idx, thread_idx
from std.gpu.primitives import block
from std.gpu.host import DeviceContext, DeviceBuffer, HostBuffer
from std.gpu.memory import AddressSpace
from layout import Layout, LayoutTensor, TileTensor, row_major

from ..constants import DT, TPB, TPB_REDUCE
from ..core.module import mptr
from ..core import Loss, AMPPolicy, NoAMP
from ..core.target_storage import require_ctx, TargetStorage, assert_tag_for


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


# Slice 7 — device-resident loss accumulator for CUDA-graph capture.
# `_ce_reduce_add_kernel` single-block-reduces `partial_loss[0..BATCH]` and
# thread 0 folds `total/BATCH` into `acc[0]` and `+1` into `acc[1]` — NO D2H,
# so `forward_accumulate` is capturable (mirrors `_mse_reduce_add_kernel`).
def _ce_reduce_add_kernel[BATCH: Int](
    partial_loss: UnsafePointer[Scalar[DT], MutAnyOrigin],
    acc: UnsafePointer[Scalar[DT], MutAnyOrigin],
):
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

    # Slice 7 — device-resident (sum_of_means, count) accumulator. Hot on the
    # CUDA-graph capture path: `forward_accumulate` folds the mean loss in with
    # no D2H; `read_accum` D2Hs once at flush; `reset_accum` zeroes it. CPU
    # mirror: `_acc_sum` / `_acc_n` host scalars.
    var loss_acc_dev: Optional[DeviceBuffer[DT]]
    var _acc_sum: Scalar[DT]
    var _acc_n: Int

    var ts: TargetStorage

    def __init__(out self):
        self.softmax = List[Scalar[DT]]()
        self.softmax_dev = None
        self.softmax_dev_n = 0
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
            "CrossEntropyLoss: target must be 'cpu' or 'gpu'"
        )
        var loss = Self()
        loss.ts = TargetStorage.make[target](ctx=ctx)
        comptime if target == "gpu":
            var ctx_v = require_ctx["CrossEntropyLoss.make[target='gpu']"](ctx)
            loss.softmax_dev = ctx_v.enqueue_create_buffer[DT](1)
            loss.partial_loss_dev = ctx_v.enqueue_create_buffer[DT](1)
            loss.partial_loss_host = ctx_v.enqueue_create_host_buffer[DT](1)
            var acc_real = ctx_v.enqueue_create_buffer[DT](2)
            acc_real.enqueue_fill(0.0)
            loss.loss_acc_dev = acc_real^
        return loss^

    def _ensure_cache_cpu(mut self, batch: Int):
        var needed = batch * Self.N_CLASSES
        if len(self.softmax) < needed:
            self.softmax.resize(needed, 0.0)

    def _ensure_buffers_gpu(mut self, batch: Int) raises:
        var ctx = self.ts.ctx.value()
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
        # CrossEntropy is `force_fp32_input=True` per the AMP doc — softmax
        # + log/exp need fp32 dynamic range. POLICY is accepted for trait
        # conformance but ignored.
        comptime assert logits.flat_rank  == 2, "logits must be rank-2"
        comptime assert targets.flat_rank == 2, "targets must be rank-2"
        assert_tag_for["CrossEntropyLoss", target](self.ts.target_tag)

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
            var ctx = self.ts.ctx.value()
            comptime mat_layout = Layout.row_major(BATCH, Self.N_CLASSES)
            comptime row_layout = Layout.row_major(BATCH)
            var lp_w = mptr(logits.ptr)
            var tp_w = mptr(targets.ptr)
            var logits_lt  = LayoutTensor[DT, mat_layout, MutAnyOrigin](lp_w)
            var targets_lt = LayoutTensor[DT, mat_layout, MutAnyOrigin](tp_w)
            var softmax_lt = LayoutTensor[DT, mat_layout, MutAnyOrigin](self.softmax_dev.value())
            var partial_lt = LayoutTensor[DT, row_layout, MutAnyOrigin](self.partial_loss_dev.value())
            comptime TPB = TPB_REDUCE
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

    def forward_capture[
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
        # CUDA-graph-capturable forward: enqueue ONLY `_ce_forward_kernel`
        # (which fills `softmax_dev`, the cache `vjp` reads) and skip the
        # partial-loss D2H copy + `ctx.synchronize()` that `forward` does to
        # return the scalar. No host work → capturable. The scalar loss is
        # not produced here (see Loss.forward_capture). POLICY ignored as in
        # `forward` (softmax/log/exp stay fp32). CPU: fall back to `forward`.
        comptime assert logits.flat_rank == 2, "logits must be rank-2"
        comptime assert targets.flat_rank == 2, "targets must be rank-2"
        assert_tag_for["CrossEntropyLoss", target](self.ts.target_tag)
        comptime if target == "cpu":
            _ = self.forward[target, BATCH, POLICY](logits, targets)
        else:
            self._ensure_buffers_gpu(BATCH)
            var ctx = self.ts.ctx.value()
            comptime mat_layout = Layout.row_major(BATCH, Self.N_CLASSES)
            comptime row_layout = Layout.row_major(BATCH)
            var lp_w = mptr(logits.ptr)
            var tp_w = mptr(targets.ptr)
            var logits_lt = LayoutTensor[DT, mat_layout, MutAnyOrigin](lp_w)
            var targets_lt = LayoutTensor[DT, mat_layout, MutAnyOrigin](tp_w)
            var softmax_lt = LayoutTensor[DT, mat_layout, MutAnyOrigin](self.softmax_dev.value())
            var partial_lt = LayoutTensor[DT, row_layout, MutAnyOrigin](self.partial_loss_dev.value())
            comptime TPB = TPB_REDUCE
            comptime n_blocks = (BATCH + TPB - 1) // TPB
            comptime kernel = _ce_forward_kernel[BATCH, Self.N_CLASSES]
            ctx.enqueue_function[kernel](
                logits_lt, targets_lt, softmax_lt, partial_lt,
                grid_dim=n_blocks, block_dim=TPB,
            )

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
        comptime assert targets.flat_rank     == 2, "targets must be rank-2"
        comptime assert grad_logits.flat_rank == 2, "grad_logits must be rank-2"
        assert_tag_for["CrossEntropyLoss", target](self.ts.target_tag)

        comptime if target == "cpu":
            var softmax = TileTensor(self.softmax, row_major[BATCH, Self.N_CLASSES]())
            var inv_batch: Scalar[DT] = 1.0 / Scalar[DT](BATCH)
            for b in range(BATCH):
                for c in range(Self.N_CLASSES):
                    grad_logits[b, c] = (softmax[b, c] - targets[b, c]) * inv_batch
        else:
            var ctx = self.ts.ctx.value()
            comptime mat_layout = Layout.row_major(BATCH, Self.N_CLASSES)
            var tp_w = mptr(targets.ptr)
            var gp_w = mptr(grad_logits.ptr)
            var softmax_lt = LayoutTensor[DT, mat_layout, MutAnyOrigin](self.softmax_dev.value())
            var targets_lt = LayoutTensor[DT, mat_layout, MutAnyOrigin](tp_w)
            var grad_lt    = LayoutTensor[DT, mat_layout, MutAnyOrigin](gp_w)
            comptime n_blocks = (BATCH * Self.N_CLASSES + TPB - 1) // TPB
            comptime kernel = _ce_backward_kernel[BATCH, Self.N_CLASSES]
            ctx.enqueue_function[kernel](
                softmax_lt, targets_lt, grad_lt, grid_dim=n_blocks, block_dim=TPB,
            )

    # ──────────────────────────────────────────────────────────────────
    # Slice 7 — CUDA-graph-capturable forward that folds the mean loss into
    # the device accumulator instead of returning a host scalar. Same
    # `_ce_forward_kernel` as `forward`/`forward_capture` (fills `softmax_dev`
    # the vjp reads), plus a device reduce-add into `loss_acc_dev` — NO D2H.
    # Caller reads it via `read_accum` at flush cadence.
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
        comptime assert logits.flat_rank == 2, "logits must be rank-2"
        comptime assert targets.flat_rank == 2, "targets must be rank-2"
        assert_tag_for["CrossEntropyLoss", target](self.ts.target_tag)

        comptime if target == "cpu":
            # CPU is not a capture target — reuse `forward`, accumulate scalar.
            var L = self.forward[target, BATCH, POLICY](logits, targets)
            self._acc_sum += L
            self._acc_n += 1
        else:
            self._ensure_buffers_gpu(BATCH)
            var ctx = self.ts.ctx.value()
            comptime mat_layout = Layout.row_major(BATCH, Self.N_CLASSES)
            comptime row_layout = Layout.row_major(BATCH)
            var lp_w = mptr(logits.ptr)
            var tp_w = mptr(targets.ptr)
            var logits_lt = LayoutTensor[DT, mat_layout, MutAnyOrigin](lp_w)
            var targets_lt = LayoutTensor[DT, mat_layout, MutAnyOrigin](tp_w)
            var softmax_lt = LayoutTensor[DT, mat_layout, MutAnyOrigin](self.softmax_dev.value())
            var partial_lt = LayoutTensor[DT, row_layout, MutAnyOrigin](self.partial_loss_dev.value())
            comptime TPB = TPB_REDUCE
            comptime n_blocks = (BATCH + TPB - 1) // TPB
            comptime kernel = _ce_forward_kernel[BATCH, Self.N_CLASSES]
            ctx.enqueue_function[kernel](
                logits_lt, targets_lt, softmax_lt, partial_lt,
                grid_dim=n_blocks, block_dim=TPB,
            )
            # Device reduce + accumulate — no D2H.
            comptime red_k = _ce_reduce_add_kernel[BATCH]
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
