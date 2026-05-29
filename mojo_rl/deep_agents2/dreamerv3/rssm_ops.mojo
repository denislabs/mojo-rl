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
from std.gpu import global_idx
from std.gpu.host import DeviceContext, DeviceBuffer
from std.gpu.memory import AddressSpace
from layout import Layout, LayoutTensor, TileTensor

from mojo_rl.nn2.constants import DT, TPB
from mojo_rl.nn2.core import Initializer, AMPPolicy, NoAMP
from mojo_rl.nn2.core.module import Module, typed_view, typed_view_mut
from mojo_rl.nn2.core.target_storage import TargetStorage, assert_tag_for


@always_inline
def _sig(x: Scalar[DT]) -> Scalar[DT]:
    return Scalar[DT](1.0) / (Scalar[DT](1.0) + exp(-x))


@always_inline
def _dev_lt[N: Int](
    p: UnsafePointer[Scalar[DT], MutAnyOrigin]
) -> LayoutTensor[DT, Layout.row_major(N), MutAnyOrigin]:
    """Wrap a flat device pointer as a row-major [N] LayoutTensor for kernels."""
    return LayoutTensor[DT, Layout.row_major(N), MutAnyOrigin](p)


# ──────────────────────────────────────────────────────────────────────
# GPU kernels for the custom RSSM ops (PR5c Step 5). Top-level fns so
# `enqueue_function` can bind them. Each mirrors the op's CPU body. Math
# is identical to the CPU path → CPU↔GPU parity ≤1e-4 (per-op spikes).
# ──────────────────────────────────────────────────────────────────────


def _asq_fwd_kernel[N: Int](
    x: LayoutTensor[DT, Layout.row_major(N), MutAnyOrigin],
    o: LayoutTensor[DT, Layout.row_major(N), MutAnyOrigin],
):
    var i = Int(global_idx.x)
    if i < N:
        var v = rebind[Scalar[DT]](x[i])
        var av = v if v >= Scalar[DT](0.0) else -v
        var denom = av if av > Scalar[DT](1.0) else Scalar[DT](1.0)
        o[i] = v / denom


def _asq_bwd_kernel[N: Int](
    x: LayoutTensor[DT, Layout.row_major(N), MutAnyOrigin],
    go: LayoutTensor[DT, Layout.row_major(N), MutAnyOrigin],
    gi: LayoutTensor[DT, Layout.row_major(N), MutAnyOrigin],
):
    var i = Int(global_idx.x)
    if i < N:
        var v = rebind[Scalar[DT]](x[i])
        var av = v if v >= Scalar[DT](0.0) else -v
        var denom = av if av > Scalar[DT](1.0) else Scalar[DT](1.0)
        gi[i] = rebind[Scalar[DT]](go[i]) / denom


@always_inline
def _sigk(x: Scalar[DT]) -> Scalar[DT]:
    return Scalar[DT](1.0) / (Scalar[DT](1.0) + exp(-x))


# ── GRUGate: N=B·D, D=DETER, DPB=DETER//BLOCKS. gru [B,3D], deter/nd [B,D];
#    block grp uses gru lanes [grp·3·DPB : +3·DPB]. ──────────────────────
def _gru_fwd_kernel[N: Int, D: Int, DPB: Int](
    gru: LayoutTensor[DT, Layout.row_major(N * 3), MutAnyOrigin],
    deter: LayoutTensor[DT, Layout.row_major(N), MutAnyOrigin],
    nd: LayoutTensor[DT, Layout.row_major(N), MutAnyOrigin],
):
    var i = Int(global_idx.x)
    if i < N:
        var b = i // D
        var d = i % D
        var grp = d // DPB
        var k = d % DPB
        var gb = b * (3 * D) + grp * (3 * DPB) + k
        var reset = _sigk(rebind[Scalar[DT]](gru[gb]))
        var cand = tanh(reset * rebind[Scalar[DT]](gru[gb + DPB]))
        var upd = _sigk(rebind[Scalar[DT]](gru[gb + 2 * DPB]) - Scalar[DT](1.0))
        nd[i] = upd * cand + (Scalar[DT](1.0) - upd) * rebind[Scalar[DT]](deter[i])


def _gru_bwd_kernel[N: Int, D: Int, DPB: Int](
    gru: LayoutTensor[DT, Layout.row_major(N * 3), MutAnyOrigin],
    deter: LayoutTensor[DT, Layout.row_major(N), MutAnyOrigin],
    go: LayoutTensor[DT, Layout.row_major(N), MutAnyOrigin],
    g_gru: LayoutTensor[DT, Layout.row_major(N * 3), MutAnyOrigin],
    g_deter: LayoutTensor[DT, Layout.row_major(N), MutAnyOrigin],
):
    var i = Int(global_idx.x)
    if i < N:
        var b = i // D
        var d = i % D
        var grp = d // DPB
        var k = d % DPB
        var gb = b * (3 * D) + grp * (3 * DPB) + k
        var c_pre = rebind[Scalar[DT]](gru[gb + DPB])
        var reset = _sigk(rebind[Scalar[DT]](gru[gb]))
        var cand = tanh(reset * c_pre)
        var upd = _sigk(rebind[Scalar[DT]](gru[gb + 2 * DPB]) - Scalar[DT](1.0))
        var d_prev = rebind[Scalar[DT]](deter[i])
        var gov = rebind[Scalar[DT]](go[i])
        var d_update = gov * (cand - d_prev)
        var d_cand = gov * upd
        g_deter[i] = gov * (Scalar[DT](1.0) - upd)
        var d_m = d_cand * (Scalar[DT](1.0) - cand * cand)
        g_gru[gb] = d_m * c_pre * reset * (Scalar[DT](1.0) - reset)
        g_gru[gb + DPB] = d_m * reset
        g_gru[gb + 2 * DPB] = d_update * upd * (Scalar[DT](1.0) - upd)


# ── BlockGroupAssemble: OUT=BLOCKS·PG, PG=DPB+3H. fwd one thread/out-elem;
#    bwd split — g_deter per (B·D), g_x* per (B·H) reducing over groups. ──
def _bga_fwd_kernel[
    NO: Int, OUT: Int, D: Int, H: Int, DPB: Int, PG: Int
](
    deter: LayoutTensor[DT, Layout.row_major(NO), MutAnyOrigin],
    x0: LayoutTensor[DT, Layout.row_major(NO), MutAnyOrigin],
    x1: LayoutTensor[DT, Layout.row_major(NO), MutAnyOrigin],
    x2: LayoutTensor[DT, Layout.row_major(NO), MutAnyOrigin],
    o: LayoutTensor[DT, Layout.row_major(NO), MutAnyOrigin],
):
    var i = Int(global_idx.x)
    if i < NO:
        var b = i // OUT
        var ob = i % OUT
        var grp = ob // PG
        var local = ob % PG
        if local < DPB:
            o[i] = rebind[Scalar[DT]](deter[b * D + grp * DPB + local])
        elif local < DPB + H:
            o[i] = rebind[Scalar[DT]](x0[b * H + (local - DPB)])
        elif local < DPB + 2 * H:
            o[i] = rebind[Scalar[DT]](x1[b * H + (local - DPB - H)])
        else:
            o[i] = rebind[Scalar[DT]](x2[b * H + (local - DPB - 2 * H)])


def _bga_bwd_deter_kernel[
    ND: Int, GON: Int, OUT: Int, D: Int, DPB: Int, PG: Int
](
    go: LayoutTensor[DT, Layout.row_major(GON), MutAnyOrigin],
    g_deter: LayoutTensor[DT, Layout.row_major(ND), MutAnyOrigin],
):
    var i = Int(global_idx.x)
    if i < ND:
        var b = i // D
        var d = i % D
        var grp = d // DPB
        var k = d % DPB
        g_deter[i] = rebind[Scalar[DT]](go[b * OUT + grp * PG + k])


def _bga_bwd_x_kernel[
    NH: Int, GON: Int, OUT: Int, H: Int, DPB: Int, PG: Int, BLOCKS: Int
](
    go: LayoutTensor[DT, Layout.row_major(GON), MutAnyOrigin],
    gx0: LayoutTensor[DT, Layout.row_major(NH), MutAnyOrigin],
    gx1: LayoutTensor[DT, Layout.row_major(NH), MutAnyOrigin],
    gx2: LayoutTensor[DT, Layout.row_major(NH), MutAnyOrigin],
):
    var i = Int(global_idx.x)
    if i < NH:
        var b = i // H
        var h = i % H
        var s0: Scalar[DT] = 0.0
        var s1: Scalar[DT] = 0.0
        var s2: Scalar[DT] = 0.0
        for grp in range(BLOCKS):
            var base = b * OUT + grp * PG + DPB
            s0 += rebind[Scalar[DT]](go[base + h])
            s1 += rebind[Scalar[DT]](go[base + H + h])
            s2 += rebind[Scalar[DT]](go[base + 2 * H + h])
        gx0[i] = s0
        gx1[i] = s1
        gx2[i] = s2


# ── StraightThroughSample: per (b,s) group of C lanes — softmax (cached) +
#    one-hot argmax (fwd); grad_z = (1-u)·sm·(go − Σ go·sm) (bwd). ────────
def _st_fwd_kernel[NG: Int, C: Int](
    z: LayoutTensor[DT, Layout.row_major(NG * C), MutAnyOrigin],
    sm: LayoutTensor[DT, Layout.row_major(NG * C), MutAnyOrigin],
    o: LayoutTensor[DT, Layout.row_major(NG * C), MutAnyOrigin],
):
    var gg = Int(global_idx.x)
    if gg < NG:
        var base = gg * C
        var zmax = rebind[Scalar[DT]](z[base])
        var amax = 0
        for c in range(1, C):
            var zc = rebind[Scalar[DT]](z[base + c])
            if zc > zmax:
                zmax = zc
                amax = c
        var ssum: Scalar[DT] = 0.0
        for c in range(C):
            var e = exp(rebind[Scalar[DT]](z[base + c]) - zmax)
            sm[base + c] = e
            ssum += e
        var inv = Scalar[DT](1.0) / ssum
        for c in range(C):
            sm[base + c] = rebind[Scalar[DT]](sm[base + c]) * inv
            o[base + c] = Scalar[DT](1.0) if c == amax else Scalar[DT](0.0)


def _st_bwd_kernel[NG: Int, C: Int](
    sm: LayoutTensor[DT, Layout.row_major(NG * C), MutAnyOrigin],
    go: LayoutTensor[DT, Layout.row_major(NG * C), MutAnyOrigin],
    gz: LayoutTensor[DT, Layout.row_major(NG * C), MutAnyOrigin],
    one_m_u: Scalar[DT],
):
    var gg = Int(global_idx.x)
    if gg < NG:
        var base = gg * C
        var dot: Scalar[DT] = 0.0
        for c in range(C):
            dot += rebind[Scalar[DT]](go[base + c]) * rebind[Scalar[DT]](sm[base + c])
        for c in range(C):
            gz[base + c] = one_m_u * rebind[Scalar[DT]](sm[base + c]) * (
                rebind[Scalar[DT]](go[base + c]) - dot
            )


struct ActionSquash[ACT: Int](Module):
    comptime ARITY: Int = 1
    comptime IN_DIMS = InlineArray[Int, 1](fill=Self.ACT)
    comptime OUT_DIM = Self.ACT

    @staticmethod
    def display_label() -> String:
        return String("ActionSquash")

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
        comptime assert target == "cpu" or target == "gpu", (
            "ActionSquash: target must be 'cpu' or 'gpu'"
        )
        var s = Self()
        comptime if target == "cpu":
            s.ts = TargetStorage.make_cpu()
        else:
            if not ctx:
                raise Error("ActionSquash.make[gpu]: ctx required")
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
        assert_tag_for["ActionSquash", target](self.ts.target_tag)
        var iv = typed_view[BATCH, Self.ACT](inputs[0])
        var ov = typed_view_mut[BATCH, Self.ACT](output)
        var ip = rebind[UnsafePointer[Scalar[DT], MutAnyOrigin]](iv.ptr)
        self._cached_input_ptr = ip
        comptime if target == "cpu":
            for i in range(BATCH * Self.ACT):
                var v = ip[i]
                var av = v if v >= Scalar[DT](0.0) else -v
                var denom = av if av > Scalar[DT](1.0) else Scalar[DT](1.0)
                ov.ptr[i] = v / denom
        else:
            comptime N = BATCH * Self.ACT
            var op = rebind[UnsafePointer[Scalar[DT], MutAnyOrigin]](ov.ptr)
            comptime nb = (N + TPB - 1) // TPB
            comptime kf = _asq_fwd_kernel[N]
            self.ts.ctx.value().enqueue_function[kf](
                _dev_lt[N](ip), _dev_lt[N](op), grid_dim=nb, block_dim=TPB
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
        var gov = typed_view[BATCH, Self.ACT](grad_output)
        var giv = typed_view_mut[BATCH, Self.ACT](grad_inputs[0])
        var xp = self._cached_input_ptr
        comptime if target == "cpu":
            for i in range(BATCH * Self.ACT):
                var v = xp[i]
                var av = v if v >= Scalar[DT](0.0) else -v
                var denom = av if av > Scalar[DT](1.0) else Scalar[DT](1.0)
                giv.ptr[i] = gov.ptr[i] / denom
        else:
            comptime N = BATCH * Self.ACT
            var gop = rebind[UnsafePointer[Scalar[DT], MutAnyOrigin]](gov.ptr)
            var gip = rebind[UnsafePointer[Scalar[DT], MutAnyOrigin]](giv.ptr)
            comptime nb = (N + TPB - 1) // TPB
            comptime kb = _asq_bwd_kernel[N]
            self.ts.ctx.value().enqueue_function[kb](
                _dev_lt[N](xp), _dev_lt[N](gop), _dev_lt[N](gip),
                grid_dim=nb, block_dim=TPB,
            )


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
    def display_label() -> String:
        return String("BlockGroupAssemble")

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
        comptime assert target == "cpu" or target == "gpu", (
            "BlockGroupAssemble: target must be 'cpu' or 'gpu'"
        )
        var s = Self()
        comptime if target == "cpu":
            s.ts = TargetStorage.make_cpu()
        else:
            if not ctx:
                raise Error("BlockGroupAssemble.make[gpu]: ctx required")
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
        comptime if target == "cpu":
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
        else:
            comptime NO = BATCH * Self.OUT_DIM
            comptime nb = (NO + TPB - 1) // TPB
            comptime kf = _bga_fwd_kernel[NO, Self.OUT_DIM, D, HH, dpb, pg]
            var dp = rebind[UnsafePointer[Scalar[DT], MutAnyOrigin]](deter)
            var x0p = rebind[UnsafePointer[Scalar[DT], MutAnyOrigin]](x0)
            var x1p = rebind[UnsafePointer[Scalar[DT], MutAnyOrigin]](x1)
            var x2p = rebind[UnsafePointer[Scalar[DT], MutAnyOrigin]](x2)
            var op = rebind[UnsafePointer[Scalar[DT], MutAnyOrigin]](o)
            self.ts.ctx.value().enqueue_function[kf](
                _dev_lt[NO](dp), _dev_lt[NO](x0p), _dev_lt[NO](x1p),
                _dev_lt[NO](x2p), _dev_lt[NO](op), grid_dim=nb, block_dim=TPB,
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
        comptime if target == "cpu":
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
        else:
            comptime GON = BATCH * Self.OUT_DIM
            comptime ND = BATCH * D
            comptime NH = BATCH * HH
            var gop = rebind[UnsafePointer[Scalar[DT], MutAnyOrigin]](go)
            var gdp = rebind[UnsafePointer[Scalar[DT], MutAnyOrigin]](g_deter)
            var g0p = rebind[UnsafePointer[Scalar[DT], MutAnyOrigin]](g_x0)
            var g1p = rebind[UnsafePointer[Scalar[DT], MutAnyOrigin]](g_x1)
            var g2p = rebind[UnsafePointer[Scalar[DT], MutAnyOrigin]](g_x2)
            comptime nbd = (ND + TPB - 1) // TPB
            comptime kd = _bga_bwd_deter_kernel[ND, GON, Self.OUT_DIM, D, dpb, pg]
            self.ts.ctx.value().enqueue_function[kd](
                _dev_lt[GON](gop), _dev_lt[ND](gdp), grid_dim=nbd, block_dim=TPB,
            )
            comptime nbh = (NH + TPB - 1) // TPB
            comptime kx = _bga_bwd_x_kernel[NH, GON, Self.OUT_DIM, HH, dpb, pg, g]
            self.ts.ctx.value().enqueue_function[kx](
                _dev_lt[GON](gop), _dev_lt[NH](g0p), _dev_lt[NH](g1p),
                _dev_lt[NH](g2p), grid_dim=nbh, block_dim=TPB,
            )


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
    def display_label() -> String:
        return String("GRUGate")

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
        comptime assert target == "cpu" or target == "gpu", (
            "GRUGate: target must be 'cpu' or 'gpu'"
        )
        var s = Self()
        comptime if target == "cpu":
            s.ts = TargetStorage.make_cpu()
        else:
            if not ctx:
                raise Error("GRUGate.make[gpu]: ctx required")
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
        comptime if target == "cpu":
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
        else:
            comptime N = BATCH * D
            var op = rebind[UnsafePointer[Scalar[DT], MutAnyOrigin]](o)
            comptime nb = (N + TPB - 1) // TPB
            comptime kf = _gru_fwd_kernel[N, D, dpb]
            self.ts.ctx.value().enqueue_function[kf](
                _dev_lt[N * 3](gru), _dev_lt[N](deter), _dev_lt[N](op),
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
        comptime D = Self.DETER
        comptime g = Self.BLOCKS
        comptime dpb = Self.DPB
        comptime opb = Self.OPB
        var go = typed_view[BATCH, D](grad_output).ptr
        var g_gru = typed_view_mut[BATCH, Self.GRU_DIM](grad_inputs[0]).ptr
        var g_deter = typed_view_mut[BATCH, D](grad_inputs[1]).ptr
        var gru = self._gru_ptr
        var deter = self._deter_ptr
        comptime if target == "cpu":
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
        else:
            comptime N = BATCH * D
            var gop = rebind[UnsafePointer[Scalar[DT], MutAnyOrigin]](go)
            comptime nb = (N + TPB - 1) // TPB
            comptime kb = _gru_bwd_kernel[N, D, dpb]
            self.ts.ctx.value().enqueue_function[kb](
                _dev_lt[N * 3](gru), _dev_lt[N](deter), _dev_lt[N](gop),
                _dev_lt[N * 3](g_gru), _dev_lt[N](g_deter),
                grid_dim=nb, block_dim=TPB,
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

    @staticmethod
    def display_label() -> String:
        return String("STSample")

    var unimix: Scalar[DT]
    var _sm: List[Scalar[DT]]   # cached softmax(z), [B·S·C] (CPU)
    var _n: Int
    var _sm_dev: Optional[DeviceBuffer[DT]]   # cached softmax(z) (GPU)
    var _sm_dev_n: Int
    var ts: TargetStorage

    def __init__(out self):
        self.unimix = Scalar[DT](0.01)
        self._sm = List[Scalar[DT]]()
        self._n = 0
        self._sm_dev = None
        self._sm_dev_n = 0
        self.ts = TargetStorage.make_uninit()

    @staticmethod
    def make[
        target: StaticString, INIT: Initializer
    ](ctx: Optional[DeviceContext] = None) raises -> Self:
        comptime assert target == "cpu" or target == "gpu", (
            "StraightThroughSample: target must be 'cpu' or 'gpu'"
        )
        var s = Self()
        comptime if target == "cpu":
            s.ts = TargetStorage.make_cpu()
        else:
            if not ctx:
                raise Error("StraightThroughSample.make[gpu]: ctx required")
            s.ts = TargetStorage.make_gpu(ctx.value())
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
        comptime if target == "cpu":
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
        else:
            comptime NN = BATCH * Self.SC
            comptime NG = BATCH * Self.STOCH
            var ctx = self.ts.ctx.value()
            if (not self._sm_dev) or self._sm_dev_n < NN:
                self._sm_dev = ctx.enqueue_create_buffer[DT](NN)
                self._sm_dev_n = NN
            var smp = rebind[UnsafePointer[Scalar[DT], MutAnyOrigin]](
                self._sm_dev.value().unsafe_ptr()
            )
            var zp = rebind[UnsafePointer[Scalar[DT], MutAnyOrigin]](z)
            var op = rebind[UnsafePointer[Scalar[DT], MutAnyOrigin]](o)
            comptime nb = (NG + TPB - 1) // TPB
            comptime kf = _st_fwd_kernel[NG, C]
            ctx.enqueue_function[kf](
                _dev_lt[NN](zp), _dev_lt[NN](smp), _dev_lt[NN](op),
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
        comptime C = Self.CLASSES
        var go = typed_view[BATCH, Self.SC](grad_output).ptr
        var gz = typed_view_mut[BATCH, Self.SC](grad_inputs[0]).ptr
        var one_m_u = Scalar[DT](1.0) - self.unimix
        comptime if target == "cpu":
            var sm = rebind[UnsafePointer[Scalar[DT], MutAnyOrigin]](
                self._sm.unsafe_ptr()
            )
            for b in range(BATCH):
                for s in range(Self.STOCH):
                    var base = (b * Self.STOCH + s) * C
                    var dot: Scalar[DT] = 0.0
                    for c in range(C):
                        dot += go[base + c] * sm[base + c]
                    for c in range(C):
                        gz[base + c] = one_m_u * sm[base + c] * (go[base + c] - dot)
        else:
            comptime NN = BATCH * Self.SC
            comptime NG = BATCH * Self.STOCH
            var ctx = self.ts.ctx.value()
            var smp = rebind[UnsafePointer[Scalar[DT], MutAnyOrigin]](
                self._sm_dev.value().unsafe_ptr()
            )
            var gop = rebind[UnsafePointer[Scalar[DT], MutAnyOrigin]](go)
            var gzp = rebind[UnsafePointer[Scalar[DT], MutAnyOrigin]](gz)
            comptime nb = (NG + TPB - 1) // TPB
            comptime kb = _st_bwd_kernel[NG, C]
            ctx.enqueue_function[kb](
                _dev_lt[NN](smp), _dev_lt[NN](gop), _dev_lt[NN](gzp),
                one_m_u, grid_dim=nb, block_dim=TPB,
            )
