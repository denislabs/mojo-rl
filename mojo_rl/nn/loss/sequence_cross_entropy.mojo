"""SequenceCrossEntropyLoss[SEQ_LEN, VOCAB] — per-token softmax + CE.

For language-model / sequence heads whose output is `SEQ_LEN * VOCAB`
logits per sample (e.g. nn `GPT`). The plain `CrossEntropyLoss[OUT_DIM]`
would softmax over the *whole* flattened `SEQ_LEN*VOCAB` row jointly,
which is wrong; this loss treats the `(BATCH, SEQ_LEN*VOCAB)` slab as
`(BATCH*SEQ_LEN, VOCAB)` and applies an independent softmax + cross-
entropy at each token position (the same row-reshape trick as
`Tokenwise`), averaging over all `BATCH*SEQ_LEN` positions.

  OUT_DIM = SEQ_LEN * VOCAB  (so Trainer's `LOSS.OUT_DIM == NET.OUT_DIM`
                              holds against a `GPT[...]` whose OUT_DIM is
                              SEQ_LEN * VOCAB).

Targets are per-token one-hots laid out the same as logits. Loss is the
mean per-token CE; grad_logits[t] = (softmax[t] - target[t]) / (B*SEQ).
Numerically stable (log-sum-exp), fp32. Reuses the `CrossEntropyLoss`
kernels with effective batch `BATCH*SEQ_LEN`.
"""

from std.math import exp, log
from std.gpu.host import DeviceContext, DeviceBuffer, HostBuffer
from std.gpu.memory import AddressSpace
from layout import Layout, LayoutTensor, TileTensor, row_major

from ..constants import DT, TPB, TPB_REDUCE
from ..core.module import mptr
from ..core import Loss, AMPPolicy, NoAMP
from ..core.target_storage import TargetStorage, assert_tag_for
from .cross_entropy import _ce_forward_kernel, _ce_backward_kernel


struct SequenceCrossEntropyLoss[SEQ_LEN: Int, VOCAB: Int](Loss):
    comptime OUT_DIM = Self.SEQ_LEN * Self.VOCAB

    # CPU
    var softmax: List[Scalar[DT]]
    # GPU
    var softmax_dev: Optional[DeviceBuffer[DT]]
    var softmax_dev_n: Int
    var partial_loss_dev: Optional[DeviceBuffer[DT]]
    var partial_loss_host: Optional[HostBuffer[DT]]
    var partial_loss_n: Int

    var ts: TargetStorage

    def __init__(out self):
        self.softmax = List[Scalar[DT]]()
        self.softmax_dev = None
        self.softmax_dev_n = 0
        self.partial_loss_dev = None
        self.partial_loss_host = None
        self.partial_loss_n = 0
        self.ts = TargetStorage.make_uninit()

    @staticmethod
    def make[target: StaticString](
        ctx: Optional[DeviceContext] = None,
    ) raises -> Self:
        comptime assert target == "cpu" or target == "gpu", (
            "SequenceCrossEntropyLoss: target must be 'cpu' or 'gpu'"
        )
        var loss = Self()
        loss.ts = TargetStorage.make[target](ctx=ctx)
        comptime if target == "gpu":
            if not ctx:
                raise Error(
                    "SequenceCrossEntropyLoss.make[target='gpu']: ctx required"
                )
            var ctx_v = ctx.value()
            loss.softmax_dev = ctx_v.enqueue_create_buffer[DT](1)
            loss.partial_loss_dev = ctx_v.enqueue_create_buffer[DT](1)
            loss.partial_loss_host = ctx_v.enqueue_create_host_buffer[DT](1)
        return loss^

    def _ensure_cache_cpu(mut self, bt: Int):
        var needed = bt * Self.VOCAB
        if len(self.softmax) < needed:
            self.softmax.resize(needed, 0.0)

    def _ensure_buffers_gpu(mut self, bt: Int) raises:
        var ctx = self.ts.ctx.value()
        var sm_needed = bt * Self.VOCAB
        if self.softmax_dev_n < sm_needed:
            self.softmax_dev = ctx.enqueue_create_buffer[DT](sm_needed)
            self.softmax_dev_n = sm_needed
        if self.partial_loss_n < bt:
            self.partial_loss_dev = ctx.enqueue_create_buffer[DT](bt)
            self.partial_loss_host = ctx.enqueue_create_host_buffer[DT](bt)
            self.partial_loss_n = bt

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
        comptime assert logits.flat_rank == 2, "logits must be rank-2"
        comptime assert targets.flat_rank == 2, "targets must be rank-2"
        assert_tag_for["SequenceCrossEntropyLoss", target](self.ts.target_tag)
        comptime BT = BATCH * Self.SEQ_LEN

        comptime if target == "cpu":
            self._ensure_cache_cpu(BT)
            var lp = mptr(logits.ptr)
            var tp = mptr(targets.ptr)
            var sm = TileTensor(self.softmax, row_major[BT, Self.VOCAB]())
            var total: Scalar[DT] = 0.0
            for r in range(BT):
                var base = r * Self.VOCAB
                var m = lp[base]
                for c in range(1, Self.VOCAB):
                    if lp[base + c] > m:
                        m = lp[base + c]
                var sum_exp: Scalar[DT] = 0.0
                for c in range(Self.VOCAB):
                    sum_exp += exp(lp[base + c] - m)
                var lse = m + log(sum_exp)
                for c in range(Self.VOCAB):
                    sm[r, c] = exp(lp[base + c] - lse)
                    total += -tp[base + c] * (lp[base + c] - lse)
            return total / Scalar[DT](BT)
        else:
            self._ensure_buffers_gpu(BT)
            var ctx = self.ts.ctx.value()
            comptime mat = Layout.row_major(BT, Self.VOCAB)
            comptime rowl = Layout.row_major(BT)
            var lp_w = mptr(logits.ptr)
            var tp_w = mptr(targets.ptr)
            var logits_lt = LayoutTensor[DT, mat, MutAnyOrigin](lp_w)
            var targets_lt = LayoutTensor[DT, mat, MutAnyOrigin](tp_w)
            var softmax_lt = LayoutTensor[DT, mat, MutAnyOrigin](
                self.softmax_dev.value()
            )
            var partial_lt = LayoutTensor[DT, rowl, MutAnyOrigin](
                self.partial_loss_dev.value()
            )
            comptime RTPB = TPB_REDUCE
            comptime n_blocks = (BT + RTPB - 1) // RTPB
            comptime kernel = _ce_forward_kernel[BT, Self.VOCAB]
            ctx.enqueue_function[kernel](
                logits_lt, targets_lt, softmax_lt, partial_lt,
                grid_dim=n_blocks, block_dim=RTPB,
            )
            ctx.enqueue_copy(
                self.partial_loss_host.value(), self.partial_loss_dev.value()
            )
            ctx.synchronize()
            var total: Scalar[DT] = 0.0
            var host_ptr = self.partial_loss_host.value().unsafe_ptr()
            for r in range(BT):
                total += host_ptr[r]
            return total / Scalar[DT](BT)

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
        comptime assert targets.flat_rank == 2, "targets must be rank-2"
        comptime assert grad_logits.flat_rank == 2, "grad_logits must be rank-2"
        assert_tag_for["SequenceCrossEntropyLoss", target](self.ts.target_tag)
        comptime BT = BATCH * Self.SEQ_LEN

        comptime if target == "cpu":
            var tp = mptr(targets.ptr)
            var gp = mptr(grad_logits.ptr)
            var sm = TileTensor(self.softmax, row_major[BT, Self.VOCAB]())
            var inv: Scalar[DT] = 1.0 / Scalar[DT](BT)
            for r in range(BT):
                var base = r * Self.VOCAB
                for c in range(Self.VOCAB):
                    gp[base + c] = (sm[r, c] - tp[base + c]) * inv
        else:
            var ctx = self.ts.ctx.value()
            comptime mat = Layout.row_major(BT, Self.VOCAB)
            var tp_w = mptr(targets.ptr)
            var gp_w = mptr(grad_logits.ptr)
            var softmax_lt = LayoutTensor[DT, mat, MutAnyOrigin](
                self.softmax_dev.value()
            )
            var targets_lt = LayoutTensor[DT, mat, MutAnyOrigin](tp_w)
            var grad_lt = LayoutTensor[DT, mat, MutAnyOrigin](gp_w)
            comptime n_blocks = (BT * Self.VOCAB + TPB - 1) // TPB
            comptime kernel = _ce_backward_kernel[BT, Self.VOCAB]
            ctx.enqueue_function[kernel](
                softmax_lt, targets_lt, grad_lt,
                grid_dim=n_blocks, block_dim=TPB,
            )
