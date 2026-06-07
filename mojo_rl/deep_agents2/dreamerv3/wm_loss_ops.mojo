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
from std.gpu import global_idx
from std.gpu.host import DeviceContext, DeviceBuffer
from std.gpu.memory import AddressSpace
from layout import Layout, LayoutTensor, TileTensor

from mojo_rl.nn2.constants import DT, TPB
from mojo_rl.nn2.core import Initializer, AMPPolicy, NoAMP
from mojo_rl.nn2.core.module import Module, typed_view, typed_view_mut
from mojo_rl.nn2.core.tensor_pack import TensorPack
from mojo_rl.nn2.core.target_storage import require_ctx, TargetStorage, assert_tag_for
from .twohot import (
    twohot_loss,
    twohot_loss_backward,
    symexp_twohot_bins,
    DREAMER_REWARD_GRID_LO,
)


@always_inline
def _symlog(x: Scalar[DT]) -> Scalar[DT]:
    var s = Scalar[DT](1.0) if x >= Scalar[DT](0.0) else Scalar[DT](-1.0)
    var a = x if x >= Scalar[DT](0.0) else -x
    return s * log1p(a)


@always_inline
def _sigmoid(x: Scalar[DT]) -> Scalar[DT]:
    return Scalar[DT](1.0) / (Scalar[DT](1.0) + exp(-x))


@always_inline
def _dlt[N: Int](
    p: UnsafePointer[Scalar[DT], MutAnyOrigin]
) -> LayoutTensor[DT, Layout.row_major(N), MutAnyOrigin]:
    return LayoutTensor[DT, Layout.row_major(N), MutAnyOrigin](p)


@always_inline
def _symk(x: Scalar[DT]) -> Scalar[DT]:
    var s = Scalar[DT](1.0) if x >= Scalar[DT](0.0) else Scalar[DT](-1.0)
    var a = x if x >= Scalar[DT](0.0) else -x
    return s * log(Scalar[DT](1.0) + a)


# ── GPU loss-op kernels (one thread per batch row). ────────────────────
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


# twohot: place target between its 2 nearest bins (inv-distance weights),
# CE against log_softmax(logits). One thread per row over BINS.
def _twohot_fwd_kernel[B: Int, BINS: Int](
    lg: LayoutTensor[DT, Layout.row_major(B * BINS), MutAnyOrigin],
    tg: LayoutTensor[DT, Layout.row_major(B), MutAnyOrigin],
    bins: LayoutTensor[DT, Layout.row_major(BINS), MutAnyOrigin],
    o: LayoutTensor[DT, Layout.row_major(B), MutAnyOrigin],
):
    var b = Int(global_idx.x)
    if b < B:
        var base = b * BINS
        var target = rebind[Scalar[DT]](tg[b])
        var n_le = 0
        for c in range(BINS):
            if rebind[Scalar[DT]](bins[c]) <= target:
                n_le += 1
        var below = n_le - 1
        var above = n_le
        if below < 0:
            below = 0
        if below > BINS - 1:
            below = BINS - 1
        if above < 0:
            above = 0
        if above > BINS - 1:
            above = BINS - 1
        var w_below: Scalar[DT]
        var w_above: Scalar[DT]
        if below == above:
            w_below = Scalar[DT](0.5)
            w_above = Scalar[DT](0.5)
        else:
            var db = rebind[Scalar[DT]](bins[below]) - target
            var da = rebind[Scalar[DT]](bins[above]) - target
            db = db if db >= Scalar[DT](0.0) else -db
            da = da if da >= Scalar[DT](0.0) else -da
            var tot = db + da
            w_below = da / tot
            w_above = db / tot
        var zmax = rebind[Scalar[DT]](lg[base])
        for c in range(1, BINS):
            var v = rebind[Scalar[DT]](lg[base + c])
            if v > zmax:
                zmax = v
        var ssum: Scalar[DT] = 0.0
        for c in range(BINS):
            ssum += exp(rebind[Scalar[DT]](lg[base + c]) - zmax)
        var lse = zmax + log(ssum)
        var lp_b = rebind[Scalar[DT]](lg[base + below]) - lse
        var lp_a = rebind[Scalar[DT]](lg[base + above]) - lse
        o[b] = -(w_below * lp_b + w_above * lp_a)


def _twohot_bwd_kernel[B: Int, BINS: Int](
    go: LayoutTensor[DT, Layout.row_major(B), MutAnyOrigin],
    lg: LayoutTensor[DT, Layout.row_major(B * BINS), MutAnyOrigin],
    tg: LayoutTensor[DT, Layout.row_major(B), MutAnyOrigin],
    bins: LayoutTensor[DT, Layout.row_major(BINS), MutAnyOrigin],
    glg: LayoutTensor[DT, Layout.row_major(B * BINS), MutAnyOrigin],
    gt: LayoutTensor[DT, Layout.row_major(B), MutAnyOrigin],
):
    var b = Int(global_idx.x)
    if b < B:
        var base = b * BINS
        var target = rebind[Scalar[DT]](tg[b])
        var up = rebind[Scalar[DT]](go[b])
        var n_le = 0
        for c in range(BINS):
            if rebind[Scalar[DT]](bins[c]) <= target:
                n_le += 1
        var below = n_le - 1
        var above = n_le
        if below < 0:
            below = 0
        if below > BINS - 1:
            below = BINS - 1
        if above < 0:
            above = 0
        if above > BINS - 1:
            above = BINS - 1
        var w_below: Scalar[DT]
        var w_above: Scalar[DT]
        if below == above:
            w_below = Scalar[DT](0.5)
            w_above = Scalar[DT](0.5)
        else:
            var db = rebind[Scalar[DT]](bins[below]) - target
            var da = rebind[Scalar[DT]](bins[above]) - target
            db = db if db >= Scalar[DT](0.0) else -db
            da = da if da >= Scalar[DT](0.0) else -da
            var tot = db + da
            w_below = da / tot
            w_above = db / tot
        var zmax = rebind[Scalar[DT]](lg[base])
        for c in range(1, BINS):
            var v = rebind[Scalar[DT]](lg[base + c])
            if v > zmax:
                zmax = v
        var ssum: Scalar[DT] = 0.0
        for c in range(BINS):
            ssum += exp(rebind[Scalar[DT]](lg[base + c]) - zmax)
        var inv = Scalar[DT](1.0) / ssum
        for c in range(BINS):
            glg[base + c] = up * (exp(rebind[Scalar[DT]](lg[base + c]) - zmax) * inv)
        glg[base + below] = rebind[Scalar[DT]](glg[base + below]) - up * w_below
        glg[base + above] = rebind[Scalar[DT]](glg[base + above]) - up * w_above
        gt[b] = 0.0


# ──────────────────────────────────────────────────────────────────────
# SymlogMSELoss[OBS] — recon head. inputs (pred[B,OBS], target[B,OBS]).
# ──────────────────────────────────────────────────────────────────────


struct SymlogMSELoss[OBS: Int](Module):
    comptime ARITY: Int = 2
    comptime IN_DIMS = InlineArray[Int, 2](fill=Self.OBS)
    comptime OUT_DIM = 1

    @staticmethod
    def display_label() -> String:
        return String("SymlogMSE")

    var _pred_ptr: Optional[UnsafePointer[Scalar[DT], MutAnyOrigin]]
    var _target_ptr: Optional[UnsafePointer[Scalar[DT], MutAnyOrigin]]
    var ts: TargetStorage

    def __init__(out self):
        self._pred_ptr = None
        self._target_ptr = None
        self.ts = TargetStorage.make_uninit()

    @staticmethod
    def make[
        target: StaticString, INIT: Initializer
    ](ctx: Optional[DeviceContext] = None) raises -> Self:
        comptime assert target == "cpu" or target == "gpu", (
            "SymlogMSELoss: target must be 'cpu' or 'gpu'"
        )
        var s = Self()
        comptime if target == "cpu":
            s.ts = TargetStorage.make_cpu()
        else:
            if not ctx:
                raise Error("SymlogMSELoss.make[gpu]: ctx required")
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
        assert_tag_for["SymlogMSELoss", target](self.ts.target_tag)
        var pred = rebind[UnsafePointer[Scalar[DT], MutAnyOrigin]](
            inputs.tile[0, BATCH, Self.OBS]().ptr
        )
        var tgt = rebind[UnsafePointer[Scalar[DT], MutAnyOrigin]](
            inputs.tile[1, BATCH, Self.OBS]().ptr
        )
        self._pred_ptr = pred
        self._target_ptr = tgt
        var o = typed_view_mut[BATCH, 1](output).ptr
        comptime if target == "cpu":
            for b in range(BATCH):
                var s: Scalar[DT] = 0.0
                for k in range(Self.OBS):
                    var d = pred[b * Self.OBS + k] - _symlog(tgt[b * Self.OBS + k])
                    s += d * d
                o[b] = s
        else:
            var op = rebind[UnsafePointer[Scalar[DT], MutAnyOrigin]](o)
            comptime nb = (BATCH + TPB - 1) // TPB
            comptime kf = _symmse_fwd_kernel[BATCH, Self.OBS]
            self.ts.ctx.value().enqueue_function[kf](
                _dlt[BATCH * Self.OBS](pred), _dlt[BATCH * Self.OBS](tgt),
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
        var go = typed_view[BATCH, 1](grad_output).ptr
        var g_pred = grad_inputs.tile[0, BATCH, Self.OBS]().ptr
        var g_tgt = grad_inputs.tile[1, BATCH, Self.OBS]().ptr
        var pred = self._pred_ptr.value()
        var tgt = self._target_ptr.value()
        comptime if target == "cpu":
            for b in range(BATCH):
                var up = go[b]
                for k in range(Self.OBS):
                    var idx = b * Self.OBS + k
                    g_pred[idx] = up * Scalar[DT](2.0) * (
                        pred[idx] - _symlog(tgt[idx])
                    )
                    g_tgt[idx] = 0.0
        else:
            var gop = rebind[UnsafePointer[Scalar[DT], MutAnyOrigin]](go)
            var gpp = rebind[UnsafePointer[Scalar[DT], MutAnyOrigin]](g_pred)
            var gtp = rebind[UnsafePointer[Scalar[DT], MutAnyOrigin]](g_tgt)
            comptime nb = (BATCH + TPB - 1) // TPB
            comptime kb = _symmse_bwd_kernel[BATCH, Self.OBS]
            self.ts.ctx.value().enqueue_function[kb](
                _dlt[BATCH](gop), _dlt[BATCH * Self.OBS](pred),
                _dlt[BATCH * Self.OBS](tgt), _dlt[BATCH * Self.OBS](gpp),
                _dlt[BATCH * Self.OBS](gtp), grid_dim=nb, block_dim=TPB,
            )


# ──────────────────────────────────────────────────────────────────────
# TwoHotLoss[BINS] — reward head. inputs (logits[B,BINS], target[B,1]).
# Bins owned by the op (symexp_twohot by default; overridable in tests).
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

    var bins: List[Scalar[DT]]
    var _bins_dev: Optional[DeviceBuffer[DT]]
    var _logits_ptr: Optional[UnsafePointer[Scalar[DT], MutAnyOrigin]]
    var _target_ptr: Optional[UnsafePointer[Scalar[DT], MutAnyOrigin]]
    var ts: TargetStorage

    def __init__(out self):
        self.bins = List[Scalar[DT]]()
        self._bins_dev = None
        self._logits_ptr = None
        self._target_ptr = None
        self.ts = TargetStorage.make_uninit()

    @staticmethod
    def make[
        target: StaticString, INIT: Initializer
    ](ctx: Optional[DeviceContext] = None) raises -> Self:
        comptime assert target == "cpu" or target == "gpu", (
            "TwoHotLoss: target must be 'cpu' or 'gpu'"
        )
        var m = Self()
        m.bins = List[Scalar[DT]](length=Self.BINS, fill=Scalar[DT](0.0))
        # This grid MUST match the grid the reward is read back on in
        # imagination / imag_loss (DreamerV3Trainer.bins). Both now read the
        # SAME `DREAMER_REWARD_GRID_LO` constant (S4) so they can't diverge — a
        # past -9-vs-(-20)-default split made the head learn the right bin INDEX
        # but decode it on the wrong value grid → predictions ~5× off, poisoning
        # imagined returns. The narrow grid also keeps bin values bounded
        # (≈8102) so `Σ softmax·bins` stays CPU↔GPU bit-stable.
        symexp_twohot_bins[Self.BINS](
            rebind[UnsafePointer[Scalar[DT], MutAnyOrigin]](
                m.bins.unsafe_ptr()
            ),
            lo=Scalar[DT](DREAMER_REWARD_GRID_LO),
        )
        comptime if target == "cpu":
            m.ts = TargetStorage.make_cpu()
        else:
            var c = require_ctx["TwoHotLoss.make[gpu]"](ctx)
            var bd = c.enqueue_create_buffer[DT](Self.BINS)
            var hb = c.enqueue_create_host_buffer[DT](Self.BINS)
            c.synchronize()
            for i in range(Self.BINS):
                hb.unsafe_ptr()[i] = m.bins[i]
            c.enqueue_copy(bd, hb)
            c.synchronize()
            m._bins_dev = bd^
            m.ts = TargetStorage.make_gpu(c)
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
        inputs: TensorPack[Self.ARITY],
        mut output: TileTensor[
            mut=True, dtype=DT, address_space=AddressSpace.GENERIC,
            element_size=1, origin=MutAnyOrigin, ...,
        ],
    ) raises:
        assert_tag_for["TwoHotLoss", target](self.ts.target_tag)
        var lg = rebind[UnsafePointer[Scalar[DT], MutAnyOrigin]](
            inputs.tile[0, BATCH, Self.BINS]().ptr
        )
        var tgt = rebind[UnsafePointer[Scalar[DT], MutAnyOrigin]](
            inputs.tile[1, BATCH, 1]().ptr
        )
        self._logits_ptr = lg
        self._target_ptr = tgt
        var o = typed_view_mut[BATCH, 1](output).ptr
        comptime if target == "cpu":
            var bins = self.bins_unsafe_ptr()
            for b in range(BATCH):
                o[b] = twohot_loss[Self.BINS](lg, b * Self.BINS, bins, tgt[b])
        else:
            var binsd = rebind[UnsafePointer[Scalar[DT], MutAnyOrigin]](
                self._bins_dev.value().unsafe_ptr()
            )
            var op = rebind[UnsafePointer[Scalar[DT], MutAnyOrigin]](o)
            comptime nb = (BATCH + TPB - 1) // TPB
            comptime kf = _twohot_fwd_kernel[BATCH, Self.BINS]
            self.ts.ctx.value().enqueue_function[kf](
                _dlt[BATCH * Self.BINS](lg), _dlt[BATCH](tgt),
                _dlt[Self.BINS](binsd), _dlt[BATCH](op),
                grid_dim=nb, block_dim=TPB,
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
        var go = typed_view[BATCH, 1](grad_output).ptr
        var g_lg = grad_inputs.tile[0, BATCH, Self.BINS]().ptr
        var g_tgt = grad_inputs.tile[1, BATCH, 1]().ptr
        var lg = self._logits_ptr.value()
        var tgt = self._target_ptr.value()
        comptime if target == "cpu":
            var bins = self.bins_unsafe_ptr()
            for i in range(BATCH * Self.BINS):
                g_lg[i] = 0.0
            for b in range(BATCH):
                twohot_loss_backward[Self.BINS](
                    lg, b * Self.BINS, bins, tgt[b], go[b], g_lg
                )
                g_tgt[b] = 0.0
        else:
            var binsd = rebind[UnsafePointer[Scalar[DT], MutAnyOrigin]](
                self._bins_dev.value().unsafe_ptr()
            )
            var gop = rebind[UnsafePointer[Scalar[DT], MutAnyOrigin]](go)
            var glp = rebind[UnsafePointer[Scalar[DT], MutAnyOrigin]](g_lg)
            var gtp = rebind[UnsafePointer[Scalar[DT], MutAnyOrigin]](g_tgt)
            comptime nb = (BATCH + TPB - 1) // TPB
            comptime kb = _twohot_bwd_kernel[BATCH, Self.BINS]
            self.ts.ctx.value().enqueue_function[kb](
                _dlt[BATCH](gop), _dlt[BATCH * Self.BINS](lg), _dlt[BATCH](tgt),
                _dlt[Self.BINS](binsd), _dlt[BATCH * Self.BINS](glp),
                _dlt[BATCH](gtp), grid_dim=nb, block_dim=TPB,
            )


# ──────────────────────────────────────────────────────────────────────
# BinaryLoss — cont head. inputs (logit[B,1], target[B,1]).
# loss = softplus(x) − t·x ; grad = sigmoid(x) − target.
# ──────────────────────────────────────────────────────────────────────


struct BinaryLoss(Module):
    comptime ARITY: Int = 2
    comptime IN_DIMS = InlineArray[Int, 2](fill=1)
    comptime OUT_DIM = 1

    @staticmethod
    def display_label() -> String:
        return String("Binary")

    var _logit_ptr: Optional[UnsafePointer[Scalar[DT], MutAnyOrigin]]
    var _target_ptr: Optional[UnsafePointer[Scalar[DT], MutAnyOrigin]]
    var ts: TargetStorage

    def __init__(out self):
        self._logit_ptr = None
        self._target_ptr = None
        self.ts = TargetStorage.make_uninit()

    @staticmethod
    def make[
        target: StaticString, INIT: Initializer
    ](ctx: Optional[DeviceContext] = None) raises -> Self:
        comptime assert target == "cpu" or target == "gpu", (
            "BinaryLoss: target must be 'cpu' or 'gpu'"
        )
        var s = Self()
        comptime if target == "cpu":
            s.ts = TargetStorage.make_cpu()
        else:
            if not ctx:
                raise Error("BinaryLoss.make[gpu]: ctx required")
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
        assert_tag_for["BinaryLoss", target](self.ts.target_tag)
        var lo = rebind[UnsafePointer[Scalar[DT], MutAnyOrigin]](
            inputs.tile[0, BATCH, 1]().ptr
        )
        var tgt = rebind[UnsafePointer[Scalar[DT], MutAnyOrigin]](
            inputs.tile[1, BATCH, 1]().ptr
        )
        self._logit_ptr = lo
        self._target_ptr = tgt
        var o = typed_view_mut[BATCH, 1](output).ptr
        comptime if target == "cpu":
            for b in range(BATCH):
                var x = lo[b]
                # softplus stable: max(x,0) + log(1+exp(-|x|))
                var ax = x if x >= Scalar[DT](0.0) else -x
                var sp = (x if x >= Scalar[DT](0.0) else Scalar[DT](0.0)) + log(
                    Scalar[DT](1.0) + exp(-ax)
                )
                o[b] = sp - tgt[b] * x
        else:
            var op = rebind[UnsafePointer[Scalar[DT], MutAnyOrigin]](o)
            comptime nb = (BATCH + TPB - 1) // TPB
            comptime kf = _binary_fwd_kernel[BATCH]
            self.ts.ctx.value().enqueue_function[kf](
                _dlt[BATCH](lo), _dlt[BATCH](tgt), _dlt[BATCH](op),
                grid_dim=nb, block_dim=TPB,
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
        var go = typed_view[BATCH, 1](grad_output).ptr
        var g_lo = grad_inputs.tile[0, BATCH, 1]().ptr
        var g_tgt = grad_inputs.tile[1, BATCH, 1]().ptr
        var lo = self._logit_ptr.value()
        var tgt = self._target_ptr.value()
        comptime if target == "cpu":
            for b in range(BATCH):
                g_lo[b] = go[b] * (_sigmoid(lo[b]) - tgt[b])
                g_tgt[b] = 0.0
        else:
            var gop = rebind[UnsafePointer[Scalar[DT], MutAnyOrigin]](go)
            var glp = rebind[UnsafePointer[Scalar[DT], MutAnyOrigin]](g_lo)
            var gtp = rebind[UnsafePointer[Scalar[DT], MutAnyOrigin]](g_tgt)
            comptime nb = (BATCH + TPB - 1) // TPB
            comptime kb = _binary_bwd_kernel[BATCH]
            self.ts.ctx.value().enqueue_function[kb](
                _dlt[BATCH](gop), _dlt[BATCH](lo), _dlt[BATCH](tgt),
                _dlt[BATCH](glp), _dlt[BATCH](gtp), grid_dim=nb, block_dim=TPB,
            )
