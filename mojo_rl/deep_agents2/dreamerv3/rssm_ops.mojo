"""RSSM custom leaf ops — the only non-composable pieces of the RSSM `_core`.

SPIKE (design redesign): everything else in `_core` is existing nn2 modules
(`Sequential[Linear, RMSNorm, GELU]`, `BlockLinear`) wired by a
`ComputeGraph`. These three ops carry the algorithm-specific math (validated
in PR5b's `core_vjp`); the graph handles all grad routing/accumulation.

  * `ActionSquash[ACT]`         — a = action / sg(max(1, |action|))
  * `BlockGroupAssemble[D,H,G]` — per block g: [deter[g·dpb:+dpb], x0, x1, x2]
  * `GRUGate[D,G]`              — block-interleaved GRU gates + mix with deter

CPU-only for the spike (GPU is a straightforward kernel port if the design
holds). Each follows the nn2 leaf pattern (Concat template): ARITY / IN_DIMS
/ OUT_DIM, `make[target,INIT]`, `forward`, `vjp`; no params → inherits the
no-op `for_each_param`/`zero_grad`.
"""

from std.math import tanh, exp
from std.gpu.host import DeviceContext
from std.gpu.memory import AddressSpace
from layout import TileTensor

from mojo_rl.nn2.constants import DT
from mojo_rl.nn2.core import Initializer, AMPPolicy, NoAMP
from mojo_rl.nn2.core.module import Module, typed_view, typed_view_mut
from mojo_rl.nn2.core.target_storage import TargetStorage, assert_tag_for


@always_inline
def _sig(x: Scalar[DT]) -> Scalar[DT]:
    return Scalar[DT](1.0) / (Scalar[DT](1.0) + exp(-x))


# ──────────────────────────────────────────────────────────────────────
# ActionSquash[ACT] — elementwise a = x / max(1, |x|) (denom stop-grad).
# ──────────────────────────────────────────────────────────────────────


struct ActionSquash[ACT: Int](Module):
    comptime ARITY: Int = 1
    comptime IN_DIMS = InlineArray[Int, 1](fill=Self.ACT)
    comptime OUT_DIM = Self.ACT

    var _cached_input_ptr: UnsafePointer[Scalar[DT], MutAnyOrigin]
    var ts: TargetStorage

    def __init__(out self):
        self._cached_input_ptr = UnsafePointer[Scalar[DT], MutAnyOrigin](
            unsafe_from_address=0
        )
        self.ts = TargetStorage.make_uninit()

    @staticmethod
    def make[
        target: StaticString, INIT: Initializer
    ](ctx: Optional[DeviceContext] = None) raises -> Self:
        comptime assert target == "cpu", "ActionSquash: spike CPU-only"
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
        assert_tag_for["ActionSquash", target](self.ts.target_tag)
        var iv = typed_view[BATCH, Self.ACT](inputs[0])
        var ov = typed_view_mut[BATCH, Self.ACT](output)
        var ip = rebind[UnsafePointer[Scalar[DT], MutAnyOrigin]](iv.ptr)
        self._cached_input_ptr = ip
        for i in range(BATCH * Self.ACT):
            var v = ip[i]
            var av = v if v >= Scalar[DT](0.0) else -v
            var denom = av if av > Scalar[DT](1.0) else Scalar[DT](1.0)
            ov.ptr[i] = v / denom

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
        var gov = typed_view[BATCH, Self.ACT](grad_output)
        var giv = typed_view_mut[BATCH, Self.ACT](grad_inputs[0])
        var xp = self._cached_input_ptr
        for i in range(BATCH * Self.ACT):
            var v = xp[i]
            var av = v if v >= Scalar[DT](0.0) else -v
            var denom = av if av > Scalar[DT](1.0) else Scalar[DT](1.0)
            giv.ptr[i] = gov.ptr[i] / denom


# ──────────────────────────────────────────────────────────────────────
# BlockGroupAssemble[DETER, H, BLOCKS] — build the dynhid input.
#   per block g: [ deter[g·dpb : +dpb] , x0 , x1 , x2 ]   (x* shared/repeated)
# ──────────────────────────────────────────────────────────────────────


struct BlockGroupAssemble[DETER: Int, H: Int, BLOCKS: Int](Module):
    comptime DPB = Self.DETER // Self.BLOCKS
    comptime PER_GROUP = Self.DPB + 3 * Self.H
    comptime ARITY: Int = 4
    comptime IN_DIMS = Self._mk_in_dims()
    comptime OUT_DIM = Self.BLOCKS * Self.PER_GROUP   # = DETER + 3·H·BLOCKS

    @staticmethod
    def _mk_in_dims() -> InlineArray[Int, 4]:
        var d = InlineArray[Int, 4](fill=Self.H)
        d[0] = Self.DETER
        return d

    var ts: TargetStorage

    def __init__(out self):
        self.ts = TargetStorage.make_uninit()

    @staticmethod
    def make[
        target: StaticString, INIT: Initializer
    ](ctx: Optional[DeviceContext] = None) raises -> Self:
        comptime assert target == "cpu", "BlockGroupAssemble: spike CPU-only"
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
        assert_tag_for["BlockGroupAssemble", target](self.ts.target_tag)
        comptime D = Self.DETER
        comptime HH = Self.H
        comptime g = Self.BLOCKS
        comptime dpb = Self.DPB
        comptime pg = Self.PER_GROUP
        var deter = typed_view[BATCH, D](inputs[0]).ptr
        var x0 = typed_view[BATCH, HH](inputs[1]).ptr
        var x1 = typed_view[BATCH, HH](inputs[2]).ptr
        var x2 = typed_view[BATCH, HH](inputs[3]).ptr
        var o = typed_view_mut[BATCH, Self.OUT_DIM](output).ptr
        for b in range(BATCH):
            for grp in range(g):
                var off = b * Self.OUT_DIM + grp * pg
                for k in range(dpb):
                    o[off + k] = deter[b * D + grp * dpb + k]
                for k in range(HH):
                    o[off + dpb + k] = x0[b * HH + k]
                for k in range(HH):
                    o[off + dpb + HH + k] = x1[b * HH + k]
                for k in range(HH):
                    o[off + dpb + 2 * HH + k] = x2[b * HH + k]

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
        comptime D = Self.DETER
        comptime HH = Self.H
        comptime g = Self.BLOCKS
        comptime dpb = Self.DPB
        comptime pg = Self.PER_GROUP
        var go = typed_view[BATCH, Self.OUT_DIM](grad_output).ptr
        var g_deter = typed_view_mut[BATCH, D](grad_inputs[0]).ptr
        var g_x0 = typed_view_mut[BATCH, HH](grad_inputs[1]).ptr
        var g_x1 = typed_view_mut[BATCH, HH](grad_inputs[2]).ptr
        var g_x2 = typed_view_mut[BATCH, HH](grad_inputs[3]).ptr
        for i in range(BATCH * HH):
            g_x0[i] = 0.0
            g_x1[i] = 0.0
            g_x2[i] = 0.0
        for b in range(BATCH):
            for grp in range(g):
                var off = b * Self.OUT_DIM + grp * pg
                for k in range(dpb):
                    g_deter[b * D + grp * dpb + k] = go[off + k]
                for k in range(HH):
                    g_x0[b * HH + k] += go[off + dpb + k]
                    g_x1[b * HH + k] += go[off + dpb + HH + k]
                    g_x2[b * HH + k] += go[off + dpb + 2 * HH + k]


# ──────────────────────────────────────────────────────────────────────
# GRUGate[DETER, BLOCKS] — block-interleaved GRU gates + mix with deter.
#   in0 = gru [B, 3·DETER]   in1 = deter [B, DETER]   → new_deter [B, DETER]
# ──────────────────────────────────────────────────────────────────────


struct GRUGate[DETER: Int, BLOCKS: Int](Module):
    comptime DPB = Self.DETER // Self.BLOCKS
    comptime OPB = 3 * Self.DPB
    comptime GRU_DIM = 3 * Self.DETER
    comptime ARITY: Int = 2
    comptime IN_DIMS = Self._mk_in_dims()
    comptime OUT_DIM = Self.DETER

    @staticmethod
    def _mk_in_dims() -> InlineArray[Int, 2]:
        var d = InlineArray[Int, 2](fill=Self.DETER)
        d[0] = Self.GRU_DIM
        return d

    var _gru_ptr: UnsafePointer[Scalar[DT], MutAnyOrigin]
    var _deter_ptr: UnsafePointer[Scalar[DT], MutAnyOrigin]
    var ts: TargetStorage

    def __init__(out self):
        self._gru_ptr = UnsafePointer[Scalar[DT], MutAnyOrigin](
            unsafe_from_address=0
        )
        self._deter_ptr = UnsafePointer[Scalar[DT], MutAnyOrigin](
            unsafe_from_address=0
        )
        self.ts = TargetStorage.make_uninit()

    @staticmethod
    def make[
        target: StaticString, INIT: Initializer
    ](ctx: Optional[DeviceContext] = None) raises -> Self:
        comptime assert target == "cpu", "GRUGate: spike CPU-only"
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
        assert_tag_for["GRUGate", target](self.ts.target_tag)
        comptime D = Self.DETER
        comptime g = Self.BLOCKS
        comptime dpb = Self.DPB
        comptime opb = Self.OPB
        var gru = rebind[UnsafePointer[Scalar[DT], MutAnyOrigin]](
            typed_view[BATCH, Self.GRU_DIM](inputs[0]).ptr
        )
        var deter = rebind[UnsafePointer[Scalar[DT], MutAnyOrigin]](
            typed_view[BATCH, D](inputs[1]).ptr
        )
        self._gru_ptr = gru
        self._deter_ptr = deter
        var o = typed_view_mut[BATCH, D](output).ptr
        for b in range(BATCH):
            for grp in range(g):
                var gb = b * Self.GRU_DIM + grp * opb
                var dbase = b * D + grp * dpb
                for k in range(dpb):
                    var reset = _sig(gru[gb + k])
                    var cand = tanh(reset * gru[gb + dpb + k])
                    var upd = _sig(gru[gb + 2 * dpb + k] - Scalar[DT](1.0))
                    o[dbase + k] = upd * cand + (
                        Scalar[DT](1.0) - upd
                    ) * deter[dbase + k]

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
        comptime D = Self.DETER
        comptime g = Self.BLOCKS
        comptime dpb = Self.DPB
        comptime opb = Self.OPB
        var go = typed_view[BATCH, D](grad_output).ptr
        var g_gru = typed_view_mut[BATCH, Self.GRU_DIM](grad_inputs[0]).ptr
        var g_deter = typed_view_mut[BATCH, D](grad_inputs[1]).ptr
        var gru = self._gru_ptr
        var deter = self._deter_ptr
        for i in range(BATCH * Self.GRU_DIM):
            g_gru[i] = 0.0
        for b in range(BATCH):
            for grp in range(g):
                var gb = b * Self.GRU_DIM + grp * opb
                var dbase = b * D + grp * dpb
                for k in range(dpb):
                    var r_pre = gru[gb + k]
                    var c_pre = gru[gb + dpb + k]
                    var u_pre = gru[gb + 2 * dpb + k] - Scalar[DT](1.0)
                    var reset = _sig(r_pre)
                    var cand = tanh(reset * c_pre)
                    var upd = _sig(u_pre)
                    var d_prev = deter[dbase + k]
                    var gov = go[dbase + k]
                    var d_update = gov * (cand - d_prev)
                    var d_cand = gov * upd
                    g_deter[dbase + k] = gov * (Scalar[DT](1.0) - upd)
                    var d_m = d_cand * (Scalar[DT](1.0) - cand * cand)
                    g_gru[gb + k] = d_m * c_pre * reset * (
                        Scalar[DT](1.0) - reset
                    )
                    g_gru[gb + dpb + k] = d_m * reset
                    g_gru[gb + 2 * dpb + k] = d_update * upd * (
                        Scalar[DT](1.0) - upd
                    )


# ──────────────────────────────────────────────────────────────────────
# StraightThroughSample[STOCH, CLASSES] — one-hot categorical sample with
# straight-through gradient.  logits[B, S·C] → onehot[B, S·C].
#   forward : value = onehot(sample)            (per (b,s) over CLASSES)
#   backward: grad_z = (1-u)·softmax_vjp(z)      (INDEPENDENT of the sample —
#             the index is stop-grad'd; grad flows through the unimix probs)
#
# Forward currently takes the argmax (deterministic placeholder); the trainer
# wires a PhiloxRandom categorical sample. The backward — the trainable path —
# is exact and validated (st_fixture).
# ──────────────────────────────────────────────────────────────────────


struct StraightThroughSample[STOCH: Int, CLASSES: Int](Module):
    comptime SC = Self.STOCH * Self.CLASSES
    comptime ARITY: Int = 1
    comptime IN_DIMS = InlineArray[Int, 1](fill=Self.SC)
    comptime OUT_DIM = Self.SC

    var unimix: Scalar[DT]
    var _sm: List[Scalar[DT]]   # cached softmax(z), [B·S·C]
    var _n: Int
    var ts: TargetStorage

    def __init__(out self):
        self.unimix = Scalar[DT](0.01)
        self._sm = List[Scalar[DT]]()
        self._n = 0
        self.ts = TargetStorage.make_uninit()

    @staticmethod
    def make[
        target: StaticString, INIT: Initializer
    ](ctx: Optional[DeviceContext] = None) raises -> Self:
        comptime assert target == "cpu", "StraightThroughSample: spike CPU-only"
        var s = Self()
        s.ts = TargetStorage.make_cpu()
        return s^

    def _ensure(mut self, n: Int):
        if self._n < n:
            self._sm = List[Scalar[DT]](length=n, fill=Scalar[DT](0.0))
            self._n = n

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
        assert_tag_for["StraightThroughSample", target](self.ts.target_tag)
        comptime C = Self.CLASSES
        var z = typed_view[BATCH, Self.SC](inputs[0]).ptr
        var o = typed_view_mut[BATCH, Self.SC](output).ptr
        self._ensure(BATCH * Self.SC)
        var sm = rebind[UnsafePointer[Scalar[DT], MutAnyOrigin]](
            self._sm.unsafe_ptr()
        )
        for b in range(BATCH):
            for s in range(Self.STOCH):
                var base = (b * Self.STOCH + s) * C
                var zmax = z[base]
                var amax = 0
                for c in range(1, C):
                    if z[base + c] > zmax:
                        zmax = z[base + c]
                        amax = c
                var ssum: Scalar[DT] = 0.0
                for c in range(C):
                    var e = exp(z[base + c] - zmax)
                    sm[base + c] = e
                    ssum += e
                var inv = Scalar[DT](1.0) / ssum
                for c in range(C):
                    sm[base + c] = sm[base + c] * inv
                    o[base + c] = (
                        Scalar[DT](1.0) if c == amax else Scalar[DT](0.0)
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
        comptime C = Self.CLASSES
        var go = typed_view[BATCH, Self.SC](grad_output).ptr
        var gz = typed_view_mut[BATCH, Self.SC](grad_inputs[0]).ptr
        var sm = rebind[UnsafePointer[Scalar[DT], MutAnyOrigin]](
            self._sm.unsafe_ptr()
        )
        var one_m_u = Scalar[DT](1.0) - self.unimix
        for b in range(BATCH):
            for s in range(Self.STOCH):
                var base = (b * Self.STOCH + s) * C
                var dot: Scalar[DT] = 0.0
                for c in range(C):
                    dot += go[base + c] * sm[base + c]
                for c in range(C):
                    gz[base + c] = one_m_u * sm[base + c] * (go[base + c] - dot)
