"""TD-MPC2 loss ops — graph-Module wrappers (ARITY=2).

Two ops the world-model `ComputeGraph` attaches as nodes so the framework
routes their gradient back to the upstream predictions automatically:

  * `MSELossPlain[DIM]`  (latent consistency) — inputs (pred, target) → [B,1];
        loss = Σ_k (pred − target)² ; grad = 2·(pred − target) to `pred` only
        (target detached). NO symlog (unlike DreamerV3's `SymlogMSELoss`):
        TD-MPC2's consistency loss is a plain MSE in SimNorm-latent space.

  * `TDMPC2TwoHotLoss[BINS, VMIN, VMAX]`  (reward + value) — inputs
        (logits[B,BINS], target[B,1]) → [B,1]; two-hot cross-entropy with
        **linear bins in symlog space** (`linspace(VMIN, VMAX, BINS)`) and the
        target **symlog-compressed inside the op** — i.e. CE against
        `two_hot(symlog(target))`, matching reference `math.soft_ce` /
        `math.two_hot` (`references/tdmpc2-main/tdmpc2/common/math.py`).
        This differs from DreamerV3's `TwoHotLoss` (symexp bins, raw target).

Both: no trainable params (inherit the no-op `for_each_param`/`zero_grad`);
cache the two input pointers in `forward`; write BOTH grad_inputs in `vjp`
(target grad = 0). The CE math reuses the bin-agnostic `twohot_loss` /
`twohot_loss_backward` helpers from the DreamerV3 port (they take a target
scalar + an arbitrary bin grid), fed the symlog'd target + linear bins.
"""

from std.math import exp, log, log1p
from std.gpu.host import DeviceContext, DeviceBuffer
from std.gpu import global_idx
from std.gpu.memory import AddressSpace
from layout import Layout, LayoutTensor, TileTensor

from mojo_rl.nn2.constants import DT, TPB
from mojo_rl.nn2.core import Initializer, AMPPolicy, NoAMP
from mojo_rl.nn2.core.module import Module, typed_view, typed_view_mut
from mojo_rl.nn2.core.target_storage import TargetStorage, assert_tag_for
from mojo_rl.deep_agents2.dreamerv3.twohot import (
    twohot_loss,
    twohot_loss_backward,
)


@always_inline
def _symlog(x: Scalar[DT]) -> Scalar[DT]:
    var s = Scalar[DT](1.0) if x >= Scalar[DT](0.0) else Scalar[DT](-1.0)
    var a = x if x >= Scalar[DT](0.0) else -x
    return s * log1p(a)


@always_inline
def _dlt[N: Int](
    p: UnsafePointer[Scalar[DT], MutAnyOrigin]
) -> LayoutTensor[DT, Layout.row_major(N), MutAnyOrigin]:
    return LayoutTensor[DT, Layout.row_major(N), MutAnyOrigin](p)


def _linspace_bins[
    BINS: Int
](
    bins_out: UnsafePointer[Scalar[DT], MutAnyOrigin],
    lo: Scalar[DT],
    hi: Scalar[DT],
):
    """bins[i] = lo + (hi − lo)·i/(BINS−1) — reference `torch.linspace`."""
    var step = (hi - lo) / Scalar[DT](BINS - 1)
    for i in range(BINS):
        bins_out[i] = lo + step * Scalar[DT](i)


# ── GPU kernels (one thread per batch row). ────────────────────────────
def _mse_fwd_kernel[B: Int, DIM: Int](
    pred: LayoutTensor[DT, Layout.row_major(B * DIM), MutAnyOrigin],
    tgt: LayoutTensor[DT, Layout.row_major(B * DIM), MutAnyOrigin],
    o: LayoutTensor[DT, Layout.row_major(B), MutAnyOrigin],
):
    var b = Int(global_idx.x)
    if b < B:
        var s: Scalar[DT] = 0.0
        for k in range(DIM):
            var d = rebind[Scalar[DT]](pred[b * DIM + k]) - rebind[Scalar[DT]](
                tgt[b * DIM + k]
            )
            s += d * d
        o[b] = s


def _mse_bwd_kernel[B: Int, DIM: Int](
    go: LayoutTensor[DT, Layout.row_major(B), MutAnyOrigin],
    pred: LayoutTensor[DT, Layout.row_major(B * DIM), MutAnyOrigin],
    tgt: LayoutTensor[DT, Layout.row_major(B * DIM), MutAnyOrigin],
    gp: LayoutTensor[DT, Layout.row_major(B * DIM), MutAnyOrigin],
    gt: LayoutTensor[DT, Layout.row_major(B * DIM), MutAnyOrigin],
):
    var b = Int(global_idx.x)
    if b < B:
        var up = rebind[Scalar[DT]](go[b])
        for k in range(DIM):
            var idx = b * DIM + k
            gp[idx] = up * Scalar[DT](2.0) * (
                rebind[Scalar[DT]](pred[idx]) - rebind[Scalar[DT]](tgt[idx])
            )
            gt[idx] = 0.0


def _th_fwd_kernel[B: Int, BINS: Int](
    lg: LayoutTensor[DT, Layout.row_major(B * BINS), MutAnyOrigin],
    tg: LayoutTensor[DT, Layout.row_major(B), MutAnyOrigin],
    bins: LayoutTensor[DT, Layout.row_major(BINS), MutAnyOrigin],
    o: LayoutTensor[DT, Layout.row_major(B), MutAnyOrigin],
):
    var b = Int(global_idx.x)
    if b < B:
        var base = b * BINS
        var raw = rebind[Scalar[DT]](tg[b])
        var sgn = Scalar[DT](1.0) if raw >= Scalar[DT](0.0) else Scalar[DT](-1.0)
        var av = raw if raw >= Scalar[DT](0.0) else -raw
        var target = sgn * log(Scalar[DT](1.0) + av)
        # generic bin search (mirrors CPU twohot_loss → GPU==CPU).
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


def _th_bwd_kernel[B: Int, BINS: Int](
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
        var raw = rebind[Scalar[DT]](tg[b])
        var sgn = Scalar[DT](1.0) if raw >= Scalar[DT](0.0) else Scalar[DT](-1.0)
        var av = raw if raw >= Scalar[DT](0.0) else -raw
        var target = sgn * log(Scalar[DT](1.0) + av)
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
# MSELossPlain[DIM] — latent consistency. inputs (pred[B,DIM], target[B,DIM]).
# ──────────────────────────────────────────────────────────────────────


struct MSELossPlain[DIM: Int](Module):
    comptime ARITY: Int = 2
    comptime IN_DIMS = InlineArray[Int, 2](fill=Self.DIM)
    comptime OUT_DIM = 1

    @staticmethod
    def display_label() -> String:
        return String("MSEPlain")

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
            "MSELossPlain: target must be 'cpu' or 'gpu'"
        )
        var s = Self()
        comptime if target == "cpu":
            s.ts = TargetStorage.make_cpu()
        else:
            if not ctx:
                raise Error("MSELossPlain.make[gpu]: ctx required")
            s.ts = TargetStorage.make_gpu(ctx.value())
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
        assert_tag_for["MSELossPlain", target](self.ts.target_tag)
        var pred = rebind[UnsafePointer[Scalar[DT], MutAnyOrigin]](
            typed_view[BATCH, Self.DIM](inputs[0]).ptr
        )
        var tgt = rebind[UnsafePointer[Scalar[DT], MutAnyOrigin]](
            typed_view[BATCH, Self.DIM](inputs[1]).ptr
        )
        self._pred_ptr = pred
        self._target_ptr = tgt
        var o = typed_view_mut[BATCH, 1](output).ptr
        comptime if target == "cpu":
            for b in range(BATCH):
                var s: Scalar[DT] = 0.0
                for k in range(Self.DIM):
                    var d = pred[b * Self.DIM + k] - tgt[b * Self.DIM + k]
                    s += d * d
                o[b] = s
        else:
            var op = rebind[UnsafePointer[Scalar[DT], MutAnyOrigin]](o)
            comptime nb = (BATCH + TPB - 1) // TPB
            comptime kf = _mse_fwd_kernel[BATCH, Self.DIM]
            self.ts.ctx.value().enqueue_function[kf](
                _dlt[BATCH * Self.DIM](pred), _dlt[BATCH * Self.DIM](tgt),
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
        mut *grad_inputs: TileTensor[
            mut=True, dtype=DT, address_space=AddressSpace.GENERIC,
            element_size=1, origin=MutAnyOrigin, ...,
        ],
    ) raises:
        var go = typed_view[BATCH, 1](grad_output).ptr
        var g_pred = typed_view_mut[BATCH, Self.DIM](grad_inputs[0]).ptr
        var g_tgt = typed_view_mut[BATCH, Self.DIM](grad_inputs[1]).ptr
        var pred = self._pred_ptr.value()
        var tgt = self._target_ptr.value()
        comptime if target == "cpu":
            for b in range(BATCH):
                var up = go[b]
                for k in range(Self.DIM):
                    var idx = b * Self.DIM + k
                    g_pred[idx] = up * Scalar[DT](2.0) * (pred[idx] - tgt[idx])
                    g_tgt[idx] = 0.0
        else:
            var gop = rebind[UnsafePointer[Scalar[DT], MutAnyOrigin]](go)
            var gpp = rebind[UnsafePointer[Scalar[DT], MutAnyOrigin]](g_pred)
            var gtp = rebind[UnsafePointer[Scalar[DT], MutAnyOrigin]](g_tgt)
            comptime nb = (BATCH + TPB - 1) // TPB
            comptime kb = _mse_bwd_kernel[BATCH, Self.DIM]
            self.ts.ctx.value().enqueue_function[kb](
                _dlt[BATCH](gop), _dlt[BATCH * Self.DIM](pred),
                _dlt[BATCH * Self.DIM](tgt), _dlt[BATCH * Self.DIM](gpp),
                _dlt[BATCH * Self.DIM](gtp), grid_dim=nb, block_dim=TPB,
            )


# ──────────────────────────────────────────────────────────────────────
# TDMPC2TwoHotLoss[BINS, VMIN, VMAX] — reward + value heads.
# inputs (logits[B,BINS], target[B,1]); linear bins in [VMIN,VMAX] (symlog
# space); target symlog'd inside. Bins owned by the op.
# ──────────────────────────────────────────────────────────────────────


struct TDMPC2TwoHotLoss[BINS: Int, VMIN: Int, VMAX: Int](Module):
    comptime ARITY: Int = 2
    comptime IN_DIMS = Self._mk_in_dims()
    comptime OUT_DIM = 1

    @staticmethod
    def display_label() -> String:
        return String("TDMPC2TwoHot")

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
            "TDMPC2TwoHotLoss: target must be 'cpu' or 'gpu'"
        )
        var m = Self()
        m.bins = List[Scalar[DT]](length=Self.BINS, fill=Scalar[DT](0.0))
        _linspace_bins[Self.BINS](
            rebind[UnsafePointer[Scalar[DT], MutAnyOrigin]](
                m.bins.unsafe_ptr()
            ),
            lo=Scalar[DT](Self.VMIN),
            hi=Scalar[DT](Self.VMAX),
        )
        comptime if target == "cpu":
            m.ts = TargetStorage.make_cpu()
        else:
            if not ctx:
                raise Error("TDMPC2TwoHotLoss.make[gpu]: ctx required")
            var c = ctx.value()
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
        assert_tag_for["TDMPC2TwoHotLoss", target](self.ts.target_tag)
        var lg = rebind[UnsafePointer[Scalar[DT], MutAnyOrigin]](
            typed_view[BATCH, Self.BINS](inputs[0]).ptr
        )
        var tgt = rebind[UnsafePointer[Scalar[DT], MutAnyOrigin]](
            typed_view[BATCH, 1](inputs[1]).ptr
        )
        self._logits_ptr = lg
        self._target_ptr = tgt
        var o = typed_view_mut[BATCH, 1](output).ptr
        comptime if target == "cpu":
            var bins = self.bins_unsafe_ptr()
            for b in range(BATCH):
                o[b] = twohot_loss[Self.BINS](
                    lg, b * Self.BINS, bins, _symlog(tgt[b])
                )
        else:
            var binsd = rebind[UnsafePointer[Scalar[DT], MutAnyOrigin]](
                self._bins_dev.value().unsafe_ptr()
            )
            var op = rebind[UnsafePointer[Scalar[DT], MutAnyOrigin]](o)
            comptime nb = (BATCH + TPB - 1) // TPB
            comptime kf = _th_fwd_kernel[BATCH, Self.BINS]
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
        mut *grad_inputs: TileTensor[
            mut=True, dtype=DT, address_space=AddressSpace.GENERIC,
            element_size=1, origin=MutAnyOrigin, ...,
        ],
    ) raises:
        var go = typed_view[BATCH, 1](grad_output).ptr
        var g_lg = typed_view_mut[BATCH, Self.BINS](grad_inputs[0]).ptr
        var g_tgt = typed_view_mut[BATCH, 1](grad_inputs[1]).ptr
        var lg = self._logits_ptr.value()
        var tgt = self._target_ptr.value()
        comptime if target == "cpu":
            var bins = self.bins_unsafe_ptr()
            for i in range(BATCH * Self.BINS):
                g_lg[i] = 0.0
            for b in range(BATCH):
                twohot_loss_backward[Self.BINS](
                    lg, b * Self.BINS, bins, _symlog(tgt[b]), go[b], g_lg
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
            comptime kb = _th_bwd_kernel[BATCH, Self.BINS]
            self.ts.ctx.value().enqueue_function[kb](
                _dlt[BATCH](gop), _dlt[BATCH * Self.BINS](lg), _dlt[BATCH](tgt),
                _dlt[Self.BINS](binsd), _dlt[BATCH * Self.BINS](glp),
                _dlt[BATCH](gtp), grid_dim=nb, block_dim=TPB,
            )
