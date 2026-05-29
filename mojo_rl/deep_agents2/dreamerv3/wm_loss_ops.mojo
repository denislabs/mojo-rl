"""World-model loss ops — graph-Module wrappers (ARITY=2) for the WM heads.

The full WM-loss `ComputeGraph` (PR5c Step 1) needs the recon / reward /
cont losses to attach as graph nodes so the framework routes their
gradient to the upstream logits/preds automatically (the same role
`OneHotKLLoss` plays for dyn/rep). Each op:

  * `SymlogMSELoss[OBS]`  (recon) — inputs (pred, target) → [B,1];
        loss = Σ_o (pred − symlog(target))² ; grad = 2·(pred − symlog(t))
        to `pred` only (target detached).
  * `TwoHotLoss[BINS]`    (reward) — inputs (logits, target) → [B,1];
        twohot cross-entropy; grad via `twohot_loss_backward`
        (= softmax − twohot(target)); bins owned by the op.
  * `BinaryLoss`          (cont)  — inputs (logit[1], target[1]) → [B,1];
        loss = softplus(x) − t·x ; grad = sigmoid(x) − target.

All three: no trainable params (inherit the no-op `for_each_param`/
`zero_grad`); cache the two input pointers in `forward`; write BOTH
grad_inputs in `vjp` (target grad = 0). The math is identical to the
manually-seeded cotangents validated in `spike_dreamer_nets.mojo` /
PR5b; here the op produces the cotangent and `Sequential.vjp` does the
rest. CPU-only at landing (GPU is PR5c Step 5, mirrors the other custom
ops' straightforward port).
"""

from std.math import exp, log, log1p
from std.gpu.host import DeviceContext
from std.gpu.memory import AddressSpace
from layout import TileTensor

from mojo_rl.nn2.constants import DT
from mojo_rl.nn2.core import Initializer, AMPPolicy, NoAMP
from mojo_rl.nn2.core.module import Module, typed_view, typed_view_mut
from mojo_rl.nn2.core.target_storage import TargetStorage, assert_tag_for
from .twohot import twohot_loss, twohot_loss_backward, symexp_twohot_bins


@always_inline
def _symlog(x: Scalar[DT]) -> Scalar[DT]:
    var s = Scalar[DT](1.0) if x >= Scalar[DT](0.0) else Scalar[DT](-1.0)
    var a = x if x >= Scalar[DT](0.0) else -x
    return s * log1p(a)


@always_inline
def _sigmoid(x: Scalar[DT]) -> Scalar[DT]:
    return Scalar[DT](1.0) / (Scalar[DT](1.0) + exp(-x))


# ──────────────────────────────────────────────────────────────────────
# SymlogMSELoss[OBS] — recon head. inputs (pred[B,OBS], target[B,OBS]).
# ──────────────────────────────────────────────────────────────────────


struct SymlogMSELoss[OBS: Int](Module):
    comptime ARITY: Int = 2
    comptime IN_DIMS = InlineArray[Int, 2](fill=Self.OBS)
    comptime OUT_DIM = 1

    var _pred_ptr: UnsafePointer[Scalar[DT], MutAnyOrigin]
    var _target_ptr: UnsafePointer[Scalar[DT], MutAnyOrigin]
    var ts: TargetStorage

    def __init__(out self):
        self._pred_ptr = UnsafePointer[Scalar[DT], MutAnyOrigin](
            unsafe_from_address=0
        )
        self._target_ptr = UnsafePointer[Scalar[DT], MutAnyOrigin](
            unsafe_from_address=0
        )
        self.ts = TargetStorage.make_uninit()

    @staticmethod
    def make[
        target: StaticString, INIT: Initializer
    ](ctx: Optional[DeviceContext] = None) raises -> Self:
        comptime assert target == "cpu", "SymlogMSELoss: PR5c CPU-only"
        var s = Self()
        s.ts = TargetStorage.make_cpu()
        return s^

    def forward[
        target: StaticString, BATCH: Int, POLICY: AMPPolicy = NoAMP
    ](
        mut self,
        var *inputs: TileTensor[
            dtype=DT, address_space=AddressSpace.GENERIC,
            element_size=1, origin=MutAnyOrigin, ...,
        ],
        mut output: TileTensor[
            mut=True, dtype=DT, address_space=AddressSpace.GENERIC,
            element_size=1, origin=MutAnyOrigin, ...,
        ],
    ) raises:
        assert_tag_for["SymlogMSELoss", target](self.ts.target_tag)
        var pred = rebind[UnsafePointer[Scalar[DT], MutAnyOrigin]](
            typed_view[BATCH, Self.OBS](inputs[0]).ptr
        )
        var tgt = rebind[UnsafePointer[Scalar[DT], MutAnyOrigin]](
            typed_view[BATCH, Self.OBS](inputs[1]).ptr
        )
        self._pred_ptr = pred
        self._target_ptr = tgt
        var o = typed_view_mut[BATCH, 1](output).ptr
        for b in range(BATCH):
            var s: Scalar[DT] = 0.0
            for k in range(Self.OBS):
                var d = pred[b * Self.OBS + k] - _symlog(tgt[b * Self.OBS + k])
                s += d * d
            o[b] = s

    def vjp[
        target: StaticString, BATCH: Int, POLICY: AMPPolicy = NoAMP,
        mode: StaticString = "all",
    ](
        mut self,
        grad_output: TileTensor[
            dtype=DT, address_space=AddressSpace.GENERIC,
            element_size=1, origin=MutAnyOrigin, ...,
        ],
        mut *grad_inputs: TileTensor[
            mut=True, dtype=DT, address_space=AddressSpace.GENERIC,
            element_size=1, origin=MutAnyOrigin, ...,
        ],
    ) raises:
        var go = typed_view[BATCH, 1](grad_output).ptr
        var g_pred = typed_view_mut[BATCH, Self.OBS](grad_inputs[0]).ptr
        var g_tgt = typed_view_mut[BATCH, Self.OBS](grad_inputs[1]).ptr
        var pred = self._pred_ptr
        var tgt = self._target_ptr
        for b in range(BATCH):
            var up = go[b]
            for k in range(Self.OBS):
                var idx = b * Self.OBS + k
                g_pred[idx] = up * Scalar[DT](2.0) * (
                    pred[idx] - _symlog(tgt[idx])
                )
                g_tgt[idx] = 0.0


# ──────────────────────────────────────────────────────────────────────
# TwoHotLoss[BINS] — reward head. inputs (logits[B,BINS], target[B,1]).
# Bins owned by the op (symexp_twohot by default; overridable in tests).
# ──────────────────────────────────────────────────────────────────────


struct TwoHotLoss[BINS: Int](Module):
    comptime ARITY: Int = 2
    comptime IN_DIMS = Self._mk_in_dims()
    comptime OUT_DIM = 1

    @staticmethod
    def _mk_in_dims() -> InlineArray[Int, 2]:
        var d = InlineArray[Int, 2](fill=1)
        d[0] = Self.BINS
        return d

    var bins: List[Scalar[DT]]
    var _logits_ptr: UnsafePointer[Scalar[DT], MutAnyOrigin]
    var _target_ptr: UnsafePointer[Scalar[DT], MutAnyOrigin]
    var ts: TargetStorage

    def __init__(out self):
        self.bins = List[Scalar[DT]]()
        self._logits_ptr = UnsafePointer[Scalar[DT], MutAnyOrigin](
            unsafe_from_address=0
        )
        self._target_ptr = UnsafePointer[Scalar[DT], MutAnyOrigin](
            unsafe_from_address=0
        )
        self.ts = TargetStorage.make_uninit()

    @staticmethod
    def make[
        target: StaticString, INIT: Initializer
    ](ctx: Optional[DeviceContext] = None) raises -> Self:
        comptime assert target == "cpu", "TwoHotLoss: PR5c CPU-only"
        var m = Self()
        m.bins = List[Scalar[DT]](length=Self.BINS, fill=Scalar[DT](0.0))
        symexp_twohot_bins[Self.BINS](
            rebind[UnsafePointer[Scalar[DT], MutAnyOrigin]](
                m.bins.unsafe_ptr()
            )
        )
        m.ts = TargetStorage.make_cpu()
        return m^

    def bins_unsafe_ptr(mut self) -> UnsafePointer[Scalar[DT], MutAnyOrigin]:
        """Mutable handle to the bin grid (tests overwrite with the
        fixture's bins; production uses the symexp grid from `make`)."""
        return rebind[UnsafePointer[Scalar[DT], MutAnyOrigin]](
            self.bins.unsafe_ptr()
        )

    def forward[
        target: StaticString, BATCH: Int, POLICY: AMPPolicy = NoAMP
    ](
        mut self,
        var *inputs: TileTensor[
            dtype=DT, address_space=AddressSpace.GENERIC,
            element_size=1, origin=MutAnyOrigin, ...,
        ],
        mut output: TileTensor[
            mut=True, dtype=DT, address_space=AddressSpace.GENERIC,
            element_size=1, origin=MutAnyOrigin, ...,
        ],
    ) raises:
        assert_tag_for["TwoHotLoss", target](self.ts.target_tag)
        var lg = rebind[UnsafePointer[Scalar[DT], MutAnyOrigin]](
            typed_view[BATCH, Self.BINS](inputs[0]).ptr
        )
        var tgt = rebind[UnsafePointer[Scalar[DT], MutAnyOrigin]](
            typed_view[BATCH, 1](inputs[1]).ptr
        )
        self._logits_ptr = lg
        self._target_ptr = tgt
        var bins = self.bins_unsafe_ptr()
        var o = typed_view_mut[BATCH, 1](output).ptr
        for b in range(BATCH):
            o[b] = twohot_loss[Self.BINS](lg, b * Self.BINS, bins, tgt[b])

    def vjp[
        target: StaticString, BATCH: Int, POLICY: AMPPolicy = NoAMP,
        mode: StaticString = "all",
    ](
        mut self,
        grad_output: TileTensor[
            dtype=DT, address_space=AddressSpace.GENERIC,
            element_size=1, origin=MutAnyOrigin, ...,
        ],
        mut *grad_inputs: TileTensor[
            mut=True, dtype=DT, address_space=AddressSpace.GENERIC,
            element_size=1, origin=MutAnyOrigin, ...,
        ],
    ) raises:
        var go = typed_view[BATCH, 1](grad_output).ptr
        var g_lg = typed_view_mut[BATCH, Self.BINS](grad_inputs[0]).ptr
        var g_tgt = typed_view_mut[BATCH, 1](grad_inputs[1]).ptr
        var lg = self._logits_ptr
        var tgt = self._target_ptr
        var bins = self.bins_unsafe_ptr()
        for i in range(BATCH * Self.BINS):
            g_lg[i] = 0.0
        for b in range(BATCH):
            twohot_loss_backward[Self.BINS](
                lg, b * Self.BINS, bins, tgt[b], go[b], g_lg
            )
            g_tgt[b] = 0.0


# ──────────────────────────────────────────────────────────────────────
# BinaryLoss — cont head. inputs (logit[B,1], target[B,1]).
# loss = softplus(x) − t·x ; grad = sigmoid(x) − target.
# ──────────────────────────────────────────────────────────────────────


struct BinaryLoss(Module):
    comptime ARITY: Int = 2
    comptime IN_DIMS = InlineArray[Int, 2](fill=1)
    comptime OUT_DIM = 1

    var _logit_ptr: UnsafePointer[Scalar[DT], MutAnyOrigin]
    var _target_ptr: UnsafePointer[Scalar[DT], MutAnyOrigin]
    var ts: TargetStorage

    def __init__(out self):
        self._logit_ptr = UnsafePointer[Scalar[DT], MutAnyOrigin](
            unsafe_from_address=0
        )
        self._target_ptr = UnsafePointer[Scalar[DT], MutAnyOrigin](
            unsafe_from_address=0
        )
        self.ts = TargetStorage.make_uninit()

    @staticmethod
    def make[
        target: StaticString, INIT: Initializer
    ](ctx: Optional[DeviceContext] = None) raises -> Self:
        comptime assert target == "cpu", "BinaryLoss: PR5c CPU-only"
        var s = Self()
        s.ts = TargetStorage.make_cpu()
        return s^

    def forward[
        target: StaticString, BATCH: Int, POLICY: AMPPolicy = NoAMP
    ](
        mut self,
        var *inputs: TileTensor[
            dtype=DT, address_space=AddressSpace.GENERIC,
            element_size=1, origin=MutAnyOrigin, ...,
        ],
        mut output: TileTensor[
            mut=True, dtype=DT, address_space=AddressSpace.GENERIC,
            element_size=1, origin=MutAnyOrigin, ...,
        ],
    ) raises:
        assert_tag_for["BinaryLoss", target](self.ts.target_tag)
        var lo = rebind[UnsafePointer[Scalar[DT], MutAnyOrigin]](
            typed_view[BATCH, 1](inputs[0]).ptr
        )
        var tgt = rebind[UnsafePointer[Scalar[DT], MutAnyOrigin]](
            typed_view[BATCH, 1](inputs[1]).ptr
        )
        self._logit_ptr = lo
        self._target_ptr = tgt
        var o = typed_view_mut[BATCH, 1](output).ptr
        for b in range(BATCH):
            var x = lo[b]
            # softplus(x) numerically stable: max(x,0) + log(1+exp(-|x|))
            var ax = x if x >= Scalar[DT](0.0) else -x
            var sp = (x if x >= Scalar[DT](0.0) else Scalar[DT](0.0)) + log(
                Scalar[DT](1.0) + exp(-ax)
            )
            o[b] = sp - tgt[b] * x

    def vjp[
        target: StaticString, BATCH: Int, POLICY: AMPPolicy = NoAMP,
        mode: StaticString = "all",
    ](
        mut self,
        grad_output: TileTensor[
            dtype=DT, address_space=AddressSpace.GENERIC,
            element_size=1, origin=MutAnyOrigin, ...,
        ],
        mut *grad_inputs: TileTensor[
            mut=True, dtype=DT, address_space=AddressSpace.GENERIC,
            element_size=1, origin=MutAnyOrigin, ...,
        ],
    ) raises:
        var go = typed_view[BATCH, 1](grad_output).ptr
        var g_lo = typed_view_mut[BATCH, 1](grad_inputs[0]).ptr
        var g_tgt = typed_view_mut[BATCH, 1](grad_inputs[1]).ptr
        var lo = self._logit_ptr
        var tgt = self._target_ptr
        for b in range(BATCH):
            g_lo[b] = go[b] * (_sigmoid(lo[b]) - tgt[b])
            g_tgt[b] = 0.0
