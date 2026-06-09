"""AlphaZero loss op — graph-Module wrapper (ARITY=2) for the AZ training loss.

One node captures the whole AlphaZero objective so the ComputeGraph routes its
gradient straight back into the prediction net's logits/value:

    inputs (pred[B, ACT+1], target[B, ACT+1]) → loss[B, 1]

where ``pred = [policy_logits(ACT) | raw_value(1)]`` (the net output) and
``target = [mcts_policy(ACT) | z(1)]`` (the packed self-play target). The loss is

    loss_b = −Σ_a π_a · log_softmax(logits)_a   +   (tanh(raw_value) − z)²

i.e. soft cross-entropy of the policy against the MCTS visit-count distribution
(``trainer.py`` / ``Connect4NNet.py:91``) plus value MSE on the **tanh-squashed**
value head (AlphaZero value ∈ [-1,1]). The value squash lives here (not in the
net) so the same raw value head feeds the MCTS expand kernel's ``VALUE_SQUASH``.

Gradients (target detached, ``grad_target = 0``):
  * policy: ``grad_logits_a = up · (softmax(logits)_a − π_a)``  (Σπ = 1)
  * value:  ``grad_raw = up · 2·(tanh(raw) − z)·(1 − tanh(raw)²)``

No trainable params (inherits the no-op param walkers). Mirrors the DreamerV3
``wm_loss_ops.mojo`` op pattern (cache input ptrs in forward; write both
grad_inputs in vjp); CPU + GPU.
"""

from std.math import exp, log, tanh
from std.gpu.host import DeviceContext, DeviceBuffer
from std.gpu import global_idx
from std.gpu.memory import AddressSpace
from layout import Layout, LayoutTensor, TileTensor

from mojo_rl.nn2.constants import DT, TPB
from mojo_rl.nn2.core import Initializer, AMPPolicy, NoAMP
from mojo_rl.nn2.core.module import Module, typed_view, typed_view_mut, mptr
from mojo_rl.nn2.core.tensor_pack import TensorPack
from mojo_rl.nn2.core.target_storage import TargetStorage, assert_tag_for


@always_inline
def _dlt[N: Int](
    p: UnsafePointer[Scalar[DT], MutAnyOrigin]
) -> LayoutTensor[DT, Layout.row_major(N), MutAnyOrigin]:
    return LayoutTensor[DT, Layout.row_major(N), MutAnyOrigin](p)


# ── GPU kernels (one thread per batch row). ───────────────────────────────
def _az_loss_fwd_kernel[B: Int, ACT: Int](
    pred: LayoutTensor[DT, Layout.row_major(B * (ACT + 1)), MutAnyOrigin],
    tgt: LayoutTensor[DT, Layout.row_major(B * (ACT + 1)), MutAnyOrigin],
    o: LayoutTensor[DT, Layout.row_major(B), MutAnyOrigin],
):
    var b = Int(global_idx.x)
    if b < B:
        var base = b * (ACT + 1)
        var zmax = rebind[Scalar[DT]](pred[base])
        for c in range(1, ACT):
            var v = rebind[Scalar[DT]](pred[base + c])
            if v > zmax:
                zmax = v
        var ssum: Scalar[DT] = 0.0
        for c in range(ACT):
            ssum += exp(rebind[Scalar[DT]](pred[base + c]) - zmax)
        var lse = zmax + log(ssum)
        var ce: Scalar[DT] = 0.0
        for c in range(ACT):
            ce += rebind[Scalar[DT]](tgt[base + c]) * (
                rebind[Scalar[DT]](pred[base + c]) - lse
            )
        var tv = tanh(rebind[Scalar[DT]](pred[base + ACT]))
        var d = tv - rebind[Scalar[DT]](tgt[base + ACT])
        o[b] = -ce + d * d


def _az_loss_bwd_kernel[B: Int, ACT: Int](
    go: LayoutTensor[DT, Layout.row_major(B), MutAnyOrigin],
    pred: LayoutTensor[DT, Layout.row_major(B * (ACT + 1)), MutAnyOrigin],
    tgt: LayoutTensor[DT, Layout.row_major(B * (ACT + 1)), MutAnyOrigin],
    gp: LayoutTensor[DT, Layout.row_major(B * (ACT + 1)), MutAnyOrigin],
    gt: LayoutTensor[DT, Layout.row_major(B * (ACT + 1)), MutAnyOrigin],
):
    var b = Int(global_idx.x)
    if b < B:
        var base = b * (ACT + 1)
        var up = rebind[Scalar[DT]](go[b])
        var zmax = rebind[Scalar[DT]](pred[base])
        for c in range(1, ACT):
            var v = rebind[Scalar[DT]](pred[base + c])
            if v > zmax:
                zmax = v
        var ssum: Scalar[DT] = 0.0
        for c in range(ACT):
            ssum += exp(rebind[Scalar[DT]](pred[base + c]) - zmax)
        var inv = Scalar[DT](1.0) / ssum
        for c in range(ACT):
            var sm = exp(rebind[Scalar[DT]](pred[base + c]) - zmax) * inv
            gp[base + c] = up * (sm - rebind[Scalar[DT]](tgt[base + c]))
            gt[base + c] = 0.0
        var tv = tanh(rebind[Scalar[DT]](pred[base + ACT]))
        var d = tv - rebind[Scalar[DT]](tgt[base + ACT])
        gp[base + ACT] = up * Scalar[DT](2.0) * d * (Scalar[DT](1.0) - tv * tv)
        gt[base + ACT] = 0.0


struct AZLossOp[ACT: Int](Module):
    comptime ARITY: Int = 2
    comptime IN_DIMS = InlineArray[Int, 2](fill=Self.ACT + 1)
    comptime OUT_DIM = 1

    @staticmethod
    def display_label() -> String:
        return String("AZLoss")

    var _pred_ptr: Optional[UnsafePointer[Scalar[DT], MutAnyOrigin]]
    var _tgt_ptr: Optional[UnsafePointer[Scalar[DT], MutAnyOrigin]]
    var ts: TargetStorage

    def __init__(out self):
        self._pred_ptr = None
        self._tgt_ptr = None
        self.ts = TargetStorage.make_uninit()

    @staticmethod
    def make[
        target: StaticString, INIT: Initializer
    ](ctx: Optional[DeviceContext] = None) raises -> Self:
        comptime assert target == "cpu" or target == "gpu", (
            "AZLossOp: target must be 'cpu' or 'gpu'"
        )
        var s = Self()
        comptime if target == "cpu":
            s.ts = TargetStorage.make_cpu()
        else:
            if not ctx:
                raise Error("AZLossOp.make[gpu]: ctx required")
            s.ts = TargetStorage.make_gpu(ctx.value())
        return s^

    def forward[
        target: StaticString, BATCH: Int, POLICY: AMPPolicy = NoAMP
    ](
        mut self,
        inputs: TensorPack[Self.ARITY],
        mut output: TileTensor[
            mut=True, dtype=DT, address_space=AddressSpace.GENERIC,
            element_size=1, origin=MutAnyOrigin, ...,
        ],
    ) raises:
        assert_tag_for["AZLossOp", target](self.ts.target_tag)
        comptime W = Self.ACT + 1
        var pred = mptr(inputs.tile[0, BATCH, W]().ptr)
        var tgt = mptr(inputs.tile[1, BATCH, W]().ptr)
        self._pred_ptr = pred
        self._tgt_ptr = tgt
        var o = typed_view_mut[BATCH, 1](output).ptr
        comptime if target == "cpu":
            for b in range(BATCH):
                var base = b * W
                var zmax = pred[base]
                for c in range(1, Self.ACT):
                    if pred[base + c] > zmax:
                        zmax = pred[base + c]
                var ssum: Scalar[DT] = 0.0
                for c in range(Self.ACT):
                    ssum += exp(pred[base + c] - zmax)
                var lse = zmax + log(ssum)
                var ce: Scalar[DT] = 0.0
                for c in range(Self.ACT):
                    ce += tgt[base + c] * (pred[base + c] - lse)
                var tv = tanh(pred[base + Self.ACT])
                var d = tv - tgt[base + Self.ACT]
                o[b] = -ce + d * d
        else:
            var op = rebind[UnsafePointer[Scalar[DT], MutAnyOrigin]](o)
            comptime nb = (BATCH + TPB - 1) // TPB
            comptime kf = _az_loss_fwd_kernel[BATCH, Self.ACT]
            self.ts.ctx.value().enqueue_function[kf](
                _dlt[BATCH * W](pred), _dlt[BATCH * W](tgt),
                _dlt[BATCH](op), grid_dim=nb, block_dim=TPB,
            )

    def vjp[
        target: StaticString, BATCH: Int, POLICY: AMPPolicy = NoAMP,
        mode: StaticString = "all",
    ](
        mut self,
        grad_output: TileTensor[
            dtype=DT, address_space=AddressSpace.GENERIC,
            element_size=1, origin=MutAnyOrigin, ...,
        ],
        grad_inputs: TensorPack[Self.ARITY],
    ) raises:
        comptime W = Self.ACT + 1
        var go = typed_view[BATCH, 1](grad_output).ptr
        var g_pred = grad_inputs.tile[0, BATCH, W]().ptr
        var g_tgt = grad_inputs.tile[1, BATCH, W]().ptr
        var pred = self._pred_ptr.value()
        var tgt = self._tgt_ptr.value()
        comptime if target == "cpu":
            for b in range(BATCH):
                var base = b * W
                var up = go[b]
                var zmax = pred[base]
                for c in range(1, Self.ACT):
                    if pred[base + c] > zmax:
                        zmax = pred[base + c]
                var ssum: Scalar[DT] = 0.0
                for c in range(Self.ACT):
                    ssum += exp(pred[base + c] - zmax)
                var inv = Scalar[DT](1.0) / ssum
                for c in range(Self.ACT):
                    var sm = exp(pred[base + c] - zmax) * inv
                    g_pred[base + c] = up * (sm - tgt[base + c])
                    g_tgt[base + c] = 0.0
                var tv = tanh(pred[base + Self.ACT])
                var d = tv - tgt[base + Self.ACT]
                g_pred[base + Self.ACT] = (
                    up * Scalar[DT](2.0) * d * (Scalar[DT](1.0) - tv * tv)
                )
                g_tgt[base + Self.ACT] = 0.0
        else:
            var gop = rebind[UnsafePointer[Scalar[DT], MutAnyOrigin]](go)
            var gpp = rebind[UnsafePointer[Scalar[DT], MutAnyOrigin]](g_pred)
            var gtp = rebind[UnsafePointer[Scalar[DT], MutAnyOrigin]](g_tgt)
            comptime nb = (BATCH + TPB - 1) // TPB
            comptime kb = _az_loss_bwd_kernel[BATCH, Self.ACT]
            self.ts.ctx.value().enqueue_function[kb](
                _dlt[BATCH](gop), _dlt[BATCH * W](pred), _dlt[BATCH * W](tgt),
                _dlt[BATCH * W](gpp), _dlt[BATCH * W](gtp),
                grid_dim=nb, block_dim=TPB,
            )
