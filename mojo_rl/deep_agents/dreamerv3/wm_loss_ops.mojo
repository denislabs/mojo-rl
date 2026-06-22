"""World-model loss ops — storage ComputeGraph nodes (Module ABI).

The WM-loss `ComputeGraph` needs the recon / reward / cont losses to attach
as graph nodes so the framework routes their gradient to the upstream
logits/preds automatically (the same role `OneHotKLLoss` plays for dyn/rep).
Each op:

  * `SymlogMSELoss[OBS]`  (recon) — inputs (pred, target) → [B,1];
        loss = Σ_o (pred − symlog(target))² ; grad = 2·(pred − symlog(t))
        to `pred` only (target detached).
  * `TwoHotLoss[BINS]`    (reward) — inputs (logits, target) → [B,1];
        twohot cross-entropy against symexp-twohot bins (value space, so the
        raw target is bracketed directly — NO symlog). Delegates to the
        reusable bin-agnostic two-hot CE in `nn.storage.loss.two_hot`
        (shared with TD-MPC2), called with `SYMLOG=False`; bins owned by the
        op (symexp grid from `make`).
  * `BinaryLoss`          (cont)  — inputs (logit[1], target[1]) → [B,1];
        loss = softplus(x) − t·x ; grad = sigmoid(x) − target.

STORAGE migration: the legacy `TensorPack`/`TileTensor`/`mptr`/`TargetStorage`
ABI is gone — these are plain storage `Module`s (forward over `TensorRefs` →
`Tensor`, vjp recomputed from `forward_input`, no cached pointers). The two-hot
math now lives once in `loss/two_hot.mojo`; only SymlogMSE / Binary keep private
kernels here. No trainable params → inherit the no-op `for_each_param`/
`zero_grad`.
"""

from std.math import exp, log, log1p
from std.gpu import global_idx
from std.gpu.host import DeviceContext, DeviceBuffer
from layout import Layout, LayoutTensor

from mojo_rl.nn.constants import DT, TPB
from mojo_rl.nn.storage.core.tensor import Tensor
from mojo_rl.nn.storage.core.tensor_refs import TensorRefs
from mojo_rl.nn.storage.core.module import Module
from mojo_rl.nn.storage.core.initializer import Initializer
from mojo_rl.nn.storage.core.amp import AMPPolicy, NoAMP

from mojo_rl.nn.storage.loss.two_hot import (
    two_hot_ce_loss_batch,
    two_hot_ce_backward_batch,
    two_hot_ce_fwd_kernel,
    two_hot_ce_bwd_kernel,
)
from .twohot import symexp_twohot_bins, DREAMER_REWARD_GRID_LO


@always_inline
def _symlog(x: Scalar[DT]) -> Scalar[DT]:
    var s = Scalar[DT](1.0) if x >= Scalar[DT](0.0) else Scalar[DT](-1.0)
    var a = x if x >= Scalar[DT](0.0) else -x
    return s * log1p(a)


@always_inline
def _symk(x: Scalar[DT]) -> Scalar[DT]:
    var s = Scalar[DT](1.0) if x >= Scalar[DT](0.0) else Scalar[DT](-1.0)
    var a = x if x >= Scalar[DT](0.0) else -x
    return s * log(Scalar[DT](1.0) + a)


@always_inline
def _sigmoid(x: Scalar[DT]) -> Scalar[DT]:
    return Scalar[DT](1.0) / (Scalar[DT](1.0) + exp(-x))


# ──────────────────────────────────────────────────────────────────────
# SymlogMSELoss[OBS] — recon head. inputs (pred[B,OBS], target[B,OBS]).
# loss[b] = Σ_k (pred − symlog(tgt))² ; grad_pred = 2·(pred − symlog(tgt))·go ;
# grad_tgt = 0 (target detached).
# ──────────────────────────────────────────────────────────────────────


def _symmse_fwd_kernel[B: Int, OBS: Int](
    pred: LayoutTensor[DT, Layout.row_major(B * OBS), MutAnyOrigin],
    tgt: LayoutTensor[DT, Layout.row_major(B * OBS), MutAnyOrigin],
    o: LayoutTensor[DT, Layout.row_major(B), MutAnyOrigin],
):
    var b = Int(global_idx.x)
    if b < B:
        var s: Scalar[DT] = 0.0
        for k in range(OBS):
            var d = rebind[Scalar[DT]](pred[b * OBS + k]) - _symk(
                rebind[Scalar[DT]](tgt[b * OBS + k])
            )
            s += d * d
        o[b] = s


def _symmse_bwd_kernel[B: Int, OBS: Int](
    go: LayoutTensor[DT, Layout.row_major(B), MutAnyOrigin],
    pred: LayoutTensor[DT, Layout.row_major(B * OBS), MutAnyOrigin],
    tgt: LayoutTensor[DT, Layout.row_major(B * OBS), MutAnyOrigin],
    gp: LayoutTensor[DT, Layout.row_major(B * OBS), MutAnyOrigin],
    gt: LayoutTensor[DT, Layout.row_major(B * OBS), MutAnyOrigin],
):
    var b = Int(global_idx.x)
    if b < B:
        var up = rebind[Scalar[DT]](go[b])
        for k in range(OBS):
            var idx = b * OBS + k
            gp[idx] = up * Scalar[DT](2.0) * (
                rebind[Scalar[DT]](pred[idx]) - _symk(rebind[Scalar[DT]](tgt[idx]))
            )
            gt[idx] = 0.0


struct SymlogMSELoss[OBS: Int](Module):
    comptime ARITY: Int = 2
    comptime IN_DIMS = InlineArray[Int, 2](fill=Self.OBS)
    comptime OUT_DIM = 1

    @staticmethod
    def display_label() -> String:
        return String("SymlogMSE")

    def __init__(out self):
        pass

    @staticmethod
    def make[
        target: StaticString, INIT: Initializer
    ](ctx: Optional[DeviceContext] = None) raises -> Self:
        return Self()

    def forward[
        target: StaticString, B: Int, o: MutOrigin, POLICY: AMPPolicy = NoAMP
    ](
        mut self,
        inputs: TensorRefs[Self.ARITY, o],
        mut out: Tensor,
        ctx: Optional[DeviceContext] = None,
    ) raises:
        ref pred = inputs[0]
        ref tgt = inputs[1]
        comptime if target == "cpu":
            out.ensure(B)
            for b in range(B):
                var s: Scalar[DT] = 0.0
                for k in range(Self.OBS):
                    var d = pred.data[b * Self.OBS + k] - _symlog(
                        tgt.data[b * Self.OBS + k]
                    )
                    s += d * d
                out.data[b] = s
        else:
            var c = ctx.value()
            out.ensure_gpu(c, B)
            comptime nb = (B + TPB - 1) // TPB
            c.enqueue_function[_symmse_fwd_kernel[B, Self.OBS]](
                pred.lt["gpu", Layout.row_major(B * Self.OBS)](),
                tgt.lt["gpu", Layout.row_major(B * Self.OBS)](),
                out.lt["gpu", Layout.row_major(B)](),
                grid_dim=nb, block_dim=TPB,
            )

    def vjp[
        target: StaticString, B: Int, ofi: MutOrigin, ogi: MutOrigin,
        POLICY: AMPPolicy = NoAMP,
    ](
        mut self,
        forward_input: TensorRefs[Self.ARITY, ofi],
        mut grad_output: Tensor,
        grad_inputs: TensorRefs[Self.ARITY, ogi],
        ctx: Optional[DeviceContext] = None,
    ) raises:
        ref pred = forward_input[0]
        ref tgt = forward_input[1]
        ref gp = grad_inputs[0]
        ref gt = grad_inputs[1]
        comptime if target == "cpu":
            gp.ensure(B * Self.OBS)
            gt.ensure(B * Self.OBS)
            for b in range(B):
                var up = grad_output.data[b]
                for k in range(Self.OBS):
                    var idx = b * Self.OBS + k
                    gp.data[idx] = up * Scalar[DT](2.0) * (
                        pred.data[idx] - _symlog(tgt.data[idx])
                    )
                    gt.data[idx] = 0.0
        else:
            var c = ctx.value()
            gp.ensure_gpu(c, B * Self.OBS)
            gt.ensure_gpu(c, B * Self.OBS)
            comptime nb = (B + TPB - 1) // TPB
            c.enqueue_function[_symmse_bwd_kernel[B, Self.OBS]](
                grad_output.lt["gpu", Layout.row_major(B)](),
                pred.lt["gpu", Layout.row_major(B * Self.OBS)](),
                tgt.lt["gpu", Layout.row_major(B * Self.OBS)](),
                gp.lt["gpu", Layout.row_major(B * Self.OBS)](),
                gt.lt["gpu", Layout.row_major(B * Self.OBS)](),
                grid_dim=nb, block_dim=TPB,
            )


# ──────────────────────────────────────────────────────────────────────
# TwoHotLoss[BINS] — reward head. inputs (logits[B,BINS], target[B,1]).
# Bins owned by the op (symexp_twohot grid; value space). The two-hot CE math
# is delegated to the reusable `two_hot_ce_*` helpers with SYMLOG=False (the
# bins already live in value space, so the raw target is bracketed directly —
# matching the legacy `twohot_loss`/`twohot_loss_backward`).
# ──────────────────────────────────────────────────────────────────────


struct TwoHotLoss[BINS: Int](Module):
    comptime ARITY: Int = 2
    comptime IN_DIMS = Self._mk_in_dims()
    comptime OUT_DIM = 1

    @staticmethod
    def display_label() -> String:
        return String("TwoHot")

    @staticmethod
    def _mk_in_dims() -> InlineArray[Int, 2]:
        var d = InlineArray[Int, 2](fill=1)
        d[0] = Self.BINS
        return d

    var bins: Tensor  # [BINS] symexp_twohot grid; host + device

    def __init__(out self):
        self.bins = Tensor()

    @staticmethod
    def make[
        target: StaticString, INIT: Initializer
    ](ctx: Optional[DeviceContext] = None) raises -> Self:
        comptime assert target == "cpu" or target == "gpu", (
            "TwoHotLoss: target must be 'cpu' or 'gpu'"
        )
        var m = Self()
        m.bins = Tensor.alloc(Self.BINS)
        # This grid MUST match the grid the reward is read back on in
        # imagination / imag_loss (DreamerV3Trainer.bins). Both now read the
        # SAME `DREAMER_REWARD_GRID_LO` constant (S4) so they can't diverge — a
        # past -9-vs-(-20)-default split made the head learn the right bin INDEX
        # but decode it on the wrong value grid → predictions ~5× off, poisoning
        # imagined returns. The narrow grid also keeps bin values bounded
        # (≈8102) so `Σ softmax·bins` stays CPU↔GPU bit-stable.
        symexp_twohot_bins[Self.BINS](
            m.bins.data.unsafe_ptr(),
            lo=Scalar[DT](DREAMER_REWARD_GRID_LO),
        )
        comptime if target == "gpu":
            if not ctx:
                raise Error("TwoHotLoss.make[gpu]: ctx required")
            m.bins.upload(ctx.value())
        return m^

    def forward[
        target: StaticString, B: Int, o: MutOrigin, POLICY: AMPPolicy = NoAMP
    ](
        mut self,
        inputs: TensorRefs[Self.ARITY, o],
        mut out: Tensor,
        ctx: Optional[DeviceContext] = None,
    ) raises:
        comptime if target == "cpu":
            two_hot_ce_loss_batch[B, Self.BINS, False](inputs, self.bins, out)
        else:
            var c = ctx.value()
            out.ensure_gpu(c, B)
            comptime nb = (B + TPB - 1) // TPB
            c.enqueue_function[two_hot_ce_fwd_kernel[B, Self.BINS, False]](
                inputs[0].lt["gpu", Layout.row_major(B * Self.BINS)](),
                inputs[1].lt["gpu", Layout.row_major(B)](),
                self.bins.lt["gpu", Layout.row_major(Self.BINS)](),
                out.lt["gpu", Layout.row_major(B)](),
                grid_dim=nb, block_dim=TPB,
            )

    def vjp[
        target: StaticString, B: Int, ofi: MutOrigin, ogi: MutOrigin,
        POLICY: AMPPolicy = NoAMP,
    ](
        mut self,
        forward_input: TensorRefs[Self.ARITY, ofi],
        mut grad_output: Tensor,
        grad_inputs: TensorRefs[Self.ARITY, ogi],
        ctx: Optional[DeviceContext] = None,
    ) raises:
        comptime if target == "cpu":
            two_hot_ce_backward_batch[B, Self.BINS, False](
                forward_input, self.bins, grad_output, grad_inputs
            )
        else:
            var c = ctx.value()
            ref g_lg = grad_inputs[0]
            ref g_tgt = grad_inputs[1]
            g_lg.ensure_gpu(c, B * Self.BINS)
            g_tgt.ensure_gpu(c, B)
            g_tgt.dev.value().enqueue_fill(Scalar[DT](0))
            comptime nb = (B + TPB - 1) // TPB
            c.enqueue_function[two_hot_ce_bwd_kernel[B, Self.BINS, False]](
                grad_output.lt["gpu", Layout.row_major(B)](),
                forward_input[0].lt["gpu", Layout.row_major(B * Self.BINS)](),
                forward_input[1].lt["gpu", Layout.row_major(B)](),
                self.bins.lt["gpu", Layout.row_major(Self.BINS)](),
                g_lg.lt["gpu", Layout.row_major(B * Self.BINS)](),
                grid_dim=nb, block_dim=TPB,
            )


# ──────────────────────────────────────────────────────────────────────
# BinaryLoss — cont head. inputs (logit[B,1], target[B,1]).
# loss = softplus(x) − t·x ; grad = sigmoid(x) − target.
# ──────────────────────────────────────────────────────────────────────


def _binary_fwd_kernel[B: Int](
    lo: LayoutTensor[DT, Layout.row_major(B), MutAnyOrigin],
    tg: LayoutTensor[DT, Layout.row_major(B), MutAnyOrigin],
    o: LayoutTensor[DT, Layout.row_major(B), MutAnyOrigin],
):
    var b = Int(global_idx.x)
    if b < B:
        var x = rebind[Scalar[DT]](lo[b])
        var ax = x if x >= Scalar[DT](0.0) else -x
        var sp = (x if x >= Scalar[DT](0.0) else Scalar[DT](0.0)) + log(
            Scalar[DT](1.0) + exp(-ax)
        )
        o[b] = sp - rebind[Scalar[DT]](tg[b]) * x


def _binary_bwd_kernel[B: Int](
    go: LayoutTensor[DT, Layout.row_major(B), MutAnyOrigin],
    lo: LayoutTensor[DT, Layout.row_major(B), MutAnyOrigin],
    tg: LayoutTensor[DT, Layout.row_major(B), MutAnyOrigin],
    gl: LayoutTensor[DT, Layout.row_major(B), MutAnyOrigin],
    gt: LayoutTensor[DT, Layout.row_major(B), MutAnyOrigin],
):
    var b = Int(global_idx.x)
    if b < B:
        var x = rebind[Scalar[DT]](lo[b])
        var sig = Scalar[DT](1.0) / (Scalar[DT](1.0) + exp(-x))
        gl[b] = rebind[Scalar[DT]](go[b]) * (sig - rebind[Scalar[DT]](tg[b]))
        gt[b] = 0.0


struct BinaryLoss(Module):
    comptime ARITY: Int = 2
    comptime IN_DIMS = InlineArray[Int, 2](fill=1)
    comptime OUT_DIM = 1

    @staticmethod
    def display_label() -> String:
        return String("Binary")

    def __init__(out self):
        pass

    @staticmethod
    def make[
        target: StaticString, INIT: Initializer
    ](ctx: Optional[DeviceContext] = None) raises -> Self:
        return Self()

    def forward[
        target: StaticString, B: Int, o: MutOrigin, POLICY: AMPPolicy = NoAMP
    ](
        mut self,
        inputs: TensorRefs[Self.ARITY, o],
        mut out: Tensor,
        ctx: Optional[DeviceContext] = None,
    ) raises:
        ref lo = inputs[0]
        ref tgt = inputs[1]
        comptime if target == "cpu":
            out.ensure(B)
            for b in range(B):
                var x = lo.data[b]
                # softplus stable: max(x,0) + log(1+exp(-|x|))
                var ax = x if x >= Scalar[DT](0.0) else -x
                var sp = (x if x >= Scalar[DT](0.0) else Scalar[DT](0.0)) + log(
                    Scalar[DT](1.0) + exp(-ax)
                )
                out.data[b] = sp - tgt.data[b] * x
        else:
            var c = ctx.value()
            out.ensure_gpu(c, B)
            comptime nb = (B + TPB - 1) // TPB
            c.enqueue_function[_binary_fwd_kernel[B]](
                lo.lt["gpu", Layout.row_major(B)](),
                tgt.lt["gpu", Layout.row_major(B)](),
                out.lt["gpu", Layout.row_major(B)](),
                grid_dim=nb, block_dim=TPB,
            )

    def vjp[
        target: StaticString, B: Int, ofi: MutOrigin, ogi: MutOrigin,
        POLICY: AMPPolicy = NoAMP,
    ](
        mut self,
        forward_input: TensorRefs[Self.ARITY, ofi],
        mut grad_output: Tensor,
        grad_inputs: TensorRefs[Self.ARITY, ogi],
        ctx: Optional[DeviceContext] = None,
    ) raises:
        ref lo = forward_input[0]
        ref tgt = forward_input[1]
        ref gl = grad_inputs[0]
        ref gt = grad_inputs[1]
        comptime if target == "cpu":
            gl.ensure(B)
            gt.ensure(B)
            for b in range(B):
                gl.data[b] = grad_output.data[b] * (
                    _sigmoid(lo.data[b]) - tgt.data[b]
                )
                gt.data[b] = 0.0
        else:
            var c = ctx.value()
            gl.ensure_gpu(c, B)
            gt.ensure_gpu(c, B)
            comptime nb = (B + TPB - 1) // TPB
            c.enqueue_function[_binary_bwd_kernel[B]](
                grad_output.lt["gpu", Layout.row_major(B)](),
                lo.lt["gpu", Layout.row_major(B)](),
                tgt.lt["gpu", Layout.row_major(B)](),
                gl.lt["gpu", Layout.row_major(B)](),
                gt.lt["gpu", Layout.row_major(B)](),
                grid_dim=nb, block_dim=TPB,
            )
