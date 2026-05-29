"""RSSM — DreamerV3 recurrent state-space model (forward + WM loss).

Port of `references/dreamerv3-main/dreamerv3/rssm.py:RSSM`, composing the
validated nn2 primitives (`Linear`, `BlockLinear`, `RMSNorm` — PR1) with
the algorithm-specific glue (action squash, block-group interleaving, the
BlockLinear GRU gates) and the categorical KL (`OneHotKL` — PR2).

PR4 SCOPE = **forward + loss-value only**. Backward / training is PR5
(the trainer hand-wires the WM-loss graph, like the legacy world models).
The methods here are validated to ≤1e-4 against the *actual* reference
(`tests/nn2/dreamerv3/fixtures/extract_pr4.py`, COMPUTE_DTYPE forced f32).

Layer counts are pinned to the v1 / `size1m` defaults — `dynlayers=1`,
`imglayers=2`, `obslayers=1` — matching the Pendulum lighthouse config.
Other depths are a PR5+ generalization (would need a comptime loop over a
List of sub-layers).

`_core` dataflow (the bug-prone part), with `g = BLOCKS`, `dpb = DETER/g`:

    a      = action / max(1, |action|)            # ELEMENTWISE squash
    x0     = gelu(rms(dynin0(deter)))   [B,H]
    x1     = gelu(rms(dynin1(stoch)))   [B,H]      # stoch flattened to S·C
    x2     = gelu(rms(dynin2(a)))       [B,H]
    # per block g: [ deter[g·dpb : g·dpb+dpb] , x0 , x1 , x2 ]  → [B, DETER+3·H·g]
    h      = gelu(rms(dynhid0(blockconcat)))   [B,DETER]   (BlockLinear)
    gru    = dyngru(h)                          [B,3·DETER] (BlockLinear)
    # de-interleave gru: block g spans 3·dpb lanes = [reset|cand|update] each dpb
    reset  = sigmoid(reset);  cand = tanh(reset·cand);  update = sigmoid(update-1)
    deter' = update·cand + (1-update)·deter

The interleaving is the einops `flat2group/group2flat` round-trip made
explicit: BlockLinear keeps block `g`'s outputs contiguous, and the gate
split happens *within* each block's `3·dpb` output lanes.
"""

from std.math import tanh, exp, sqrt, log1p
from std.memory import alloc
from layout import TileTensor, row_major

from mojo_rl.nn2.constants import DT
from mojo_rl.nn2.primitives.linear import Linear
from mojo_rl.nn2.primitives.block_linear import BlockLinear
from mojo_rl.nn2.primitives.rms_norm import RMSNorm
from mojo_rl.nn2.initializer import Zero

from .onehot_kl import OneHotKL


# ── shared forward drivers (build named output l-values) ────────────────


def lin_fwd[
    IN: Int, OUT: Int, B: Int
](
    mut lin: Linear[IN, OUT],
    inp: UnsafePointer[Scalar[DT], MutAnyOrigin],
    outp: UnsafePointer[Scalar[DT], MutAnyOrigin],
) raises:
    var ot = TileTensor(outp, row_major[B, OUT]())
    lin.forward["cpu", B](TileTensor(inp, row_major[B, IN]()), output=ot)


def bl_fwd[
    IN: Int, OUT: Int, BLK: Int, B: Int
](
    mut bl: BlockLinear[IN, OUT, BLK],
    inp: UnsafePointer[Scalar[DT], MutAnyOrigin],
    outp: UnsafePointer[Scalar[DT], MutAnyOrigin],
) raises:
    var ot = TileTensor(outp, row_major[B, OUT]())
    bl.forward["cpu", B](TileTensor(inp, row_major[B, IN]()), output=ot)


def rms_fwd[
    DIM: Int, B: Int
](
    mut rn: RMSNorm[DIM],
    inp: UnsafePointer[Scalar[DT], MutAnyOrigin],
    outp: UnsafePointer[Scalar[DT], MutAnyOrigin],
) raises:
    var ot = TileTensor(outp, row_major[B, DIM]())
    rn.forward["cpu", B](TileTensor(inp, row_major[B, DIM]()), output=ot)


# ── shared vjp drivers (build named grad-input l-values for the variadic
#    `mut *grad_inputs`; zero the leaf grads first so each call is fresh) ──


def lin_vjp[
    IN: Int, OUT: Int, B: Int
](
    mut lin: Linear[IN, OUT],
    go: UnsafePointer[Scalar[DT], MutAnyOrigin],
    gi: UnsafePointer[Scalar[DT], MutAnyOrigin],
) raises:
    lin.zero_grad["cpu"]()
    var gi_tt = TileTensor(gi, row_major[B, IN]())
    lin.vjp["cpu", B](TileTensor(go, row_major[B, OUT]()), gi_tt)


def bl_vjp[
    IN: Int, OUT: Int, BLK: Int, B: Int
](
    mut bl: BlockLinear[IN, OUT, BLK],
    go: UnsafePointer[Scalar[DT], MutAnyOrigin],
    gi: UnsafePointer[Scalar[DT], MutAnyOrigin],
) raises:
    bl.zero_grad["cpu"]()
    var gi_tt = TileTensor(gi, row_major[B, IN]())
    bl.vjp["cpu", B](TileTensor(go, row_major[B, OUT]()), gi_tt)


def rms_vjp[
    DIM: Int, B: Int
](
    mut rn: RMSNorm[DIM],
    go: UnsafePointer[Scalar[DT], MutAnyOrigin],
    gi: UnsafePointer[Scalar[DT], MutAnyOrigin],
) raises:
    rn.zero_grad["cpu"]()
    var gi_tt = TileTensor(gi, row_major[B, DIM]())
    rn.vjp["cpu", B](TileTensor(go, row_major[B, DIM]()), gi_tt)


# ── scalar activation helpers (match GELUOp tanh-approx + jax) ──────────


@always_inline
def _gelu_scalar(x: Scalar[DT]) -> Scalar[DT]:
    comptime C = Scalar[DT](0.7978845608028654)  # sqrt(2/pi)
    var u = C * (x + Scalar[DT](0.044715) * x * x * x)
    return Scalar[DT](0.5) * x * (Scalar[DT](1.0) + tanh(u))


@always_inline
def _sigmoid_scalar(x: Scalar[DT]) -> Scalar[DT]:
    return Scalar[DT](1.0) / (Scalar[DT](1.0) + exp(-x))


@always_inline
def _gelu_buf(p: UnsafePointer[Scalar[DT], MutAnyOrigin], n: Int):
    for i in range(n):
        p[i] = _gelu_scalar(p[i])


@always_inline
def _gelu_grad(x: Scalar[DT]) -> Scalar[DT]:
    """d/dx of the tanh-approx GELU (matches GELUOp.backward, PR1)."""
    comptime C = Scalar[DT](0.7978845608028654)
    var u = C * (x + Scalar[DT](0.044715) * x * x * x)
    var t = tanh(u)
    var dudx = C * (Scalar[DT](1.0) + Scalar[DT](0.134145) * x * x)
    return Scalar[DT](0.5) * (Scalar[DT](1.0) + t) + Scalar[DT](
        0.5
    ) * x * (Scalar[DT](1.0) - t * t) * dudx


@always_inline
def _symlog_grad(x: Scalar[DT]) -> Scalar[DT]:
    """d/dx of symlog = 1/(1+|x|)."""
    var a = x if x >= Scalar[DT](0.0) else -x
    return Scalar[DT](1.0) / (Scalar[DT](1.0) + a)


@always_inline
def _symlog_scalar(x: Scalar[DT]) -> Scalar[DT]:
    var s = Scalar[DT](1.0) if x >= Scalar[DT](0.0) else Scalar[DT](-1.0)
    var a = x if x >= Scalar[DT](0.0) else -x
    return s * log1p(a)


struct RSSM[
    DETER: Int,
    HIDDEN: Int,
    STOCH: Int,
    CLASSES: Int,
    BLOCKS: Int,
    ACT: Int,
    TOKEN: Int,
](Movable & ImplicitlyDestructible):
    """RSSM with `dynlayers=1, imglayers=2, obslayers=1` (v1 defaults)."""

    comptime SC = Self.STOCH * Self.CLASSES         # stoch flat dim
    comptime DPB = Self.DETER // Self.BLOCKS         # deter per block
    comptime DHIN = Self.DETER + 3 * Self.HIDDEN * Self.BLOCKS  # dynhid in
    comptime GRU_OUT = 3 * Self.DETER
    comptime OBSIN = Self.DETER + Self.TOKEN

    # ── sub-modules ──────────────────────────────────────────────────
    var dynin0: Linear[Self.DETER, Self.HIDDEN]
    var dynin0n: RMSNorm[Self.HIDDEN]
    var dynin1: Linear[Self.SC, Self.HIDDEN]
    var dynin1n: RMSNorm[Self.HIDDEN]
    var dynin2: Linear[Self.ACT, Self.HIDDEN]
    var dynin2n: RMSNorm[Self.HIDDEN]
    var dynhid0: BlockLinear[Self.DHIN, Self.DETER, Self.BLOCKS]
    var dynhid0n: RMSNorm[Self.DETER]
    var dyngru: BlockLinear[Self.DETER, Self.GRU_OUT, Self.BLOCKS]
    var prior0: Linear[Self.DETER, Self.HIDDEN]
    var prior0n: RMSNorm[Self.HIDDEN]
    var prior1: Linear[Self.HIDDEN, Self.HIDDEN]
    var prior1n: RMSNorm[Self.HIDDEN]
    var priorlogit: Linear[Self.HIDDEN, Self.SC]
    var obs0: Linear[Self.OBSIN, Self.HIDDEN]
    var obs0n: RMSNorm[Self.HIDDEN]
    var obslogit: Linear[Self.HIDDEN, Self.SC]

    var kl: OneHotKL[Self.STOCH, Self.CLASSES]

    def __init__(out self):
        self.dynin0 = Linear[Self.DETER, Self.HIDDEN]()
        self.dynin0n = RMSNorm[Self.HIDDEN]()
        self.dynin1 = Linear[Self.SC, Self.HIDDEN]()
        self.dynin1n = RMSNorm[Self.HIDDEN]()
        self.dynin2 = Linear[Self.ACT, Self.HIDDEN]()
        self.dynin2n = RMSNorm[Self.HIDDEN]()
        self.dynhid0 = BlockLinear[Self.DHIN, Self.DETER, Self.BLOCKS]()
        self.dynhid0n = RMSNorm[Self.DETER]()
        self.dyngru = BlockLinear[Self.DETER, Self.GRU_OUT, Self.BLOCKS]()
        self.prior0 = Linear[Self.DETER, Self.HIDDEN]()
        self.prior0n = RMSNorm[Self.HIDDEN]()
        self.prior1 = Linear[Self.HIDDEN, Self.HIDDEN]()
        self.prior1n = RMSNorm[Self.HIDDEN]()
        self.priorlogit = Linear[Self.HIDDEN, Self.SC]()
        self.obs0 = Linear[Self.OBSIN, Self.HIDDEN]()
        self.obs0n = RMSNorm[Self.HIDDEN]()
        self.obslogit = Linear[Self.HIDDEN, Self.SC]()
        self.kl = OneHotKL[Self.STOCH, Self.CLASSES]()

    @staticmethod
    def make[
        target: StaticString
    ](
        unimix: Scalar[DT] = Scalar[DT](0.01),
        free_nats: Scalar[DT] = Scalar[DT](1.0),
    ) raises -> Self:
        comptime assert target == "cpu", "RSSM: PR4 is CPU-only (forward)"
        var m = Self()
        m.dynin0 = Linear[Self.DETER, Self.HIDDEN].make[target, Zero]()
        m.dynin0n = RMSNorm[Self.HIDDEN].make[target, Zero]()
        m.dynin1 = Linear[Self.SC, Self.HIDDEN].make[target, Zero]()
        m.dynin1n = RMSNorm[Self.HIDDEN].make[target, Zero]()
        m.dynin2 = Linear[Self.ACT, Self.HIDDEN].make[target, Zero]()
        m.dynin2n = RMSNorm[Self.HIDDEN].make[target, Zero]()
        m.dynhid0 = BlockLinear[
            Self.DHIN, Self.DETER, Self.BLOCKS
        ].make[target, Zero]()
        m.dynhid0n = RMSNorm[Self.DETER].make[target, Zero]()
        m.dyngru = BlockLinear[
            Self.DETER, Self.GRU_OUT, Self.BLOCKS
        ].make[target, Zero]()
        m.prior0 = Linear[Self.DETER, Self.HIDDEN].make[target, Zero]()
        m.prior0n = RMSNorm[Self.HIDDEN].make[target, Zero]()
        m.prior1 = Linear[Self.HIDDEN, Self.HIDDEN].make[target, Zero]()
        m.prior1n = RMSNorm[Self.HIDDEN].make[target, Zero]()
        m.priorlogit = Linear[Self.HIDDEN, Self.SC].make[target, Zero]()
        m.obs0 = Linear[Self.OBSIN, Self.HIDDEN].make[target, Zero]()
        m.obs0n = RMSNorm[Self.HIDDEN].make[target, Zero]()
        m.obslogit = Linear[Self.HIDDEN, Self.SC].make[target, Zero]()
        m.kl = OneHotKL[Self.STOCH, Self.CLASSES].make(unimix, free_nats)
        return m^

    # ── primitive-driving helpers (CPU) ──────────────────────────────

    @staticmethod
    def _lin[
        IN: Int, OUT: Int, B: Int
    ](
        mut lin: Linear[IN, OUT],
        inp: UnsafePointer[Scalar[DT], MutAnyOrigin],
        outp: UnsafePointer[Scalar[DT], MutAnyOrigin],
    ) raises:
        var it = TileTensor(inp, row_major[B, IN]())
        var ot = TileTensor(outp, row_major[B, OUT]())
        lin.forward["cpu", B](it, output=ot)

    @staticmethod
    def _bl[
        IN: Int, OUT: Int, BLK: Int, B: Int
    ](
        mut bl: BlockLinear[IN, OUT, BLK],
        inp: UnsafePointer[Scalar[DT], MutAnyOrigin],
        outp: UnsafePointer[Scalar[DT], MutAnyOrigin],
    ) raises:
        var it = TileTensor(inp, row_major[B, IN]())
        var ot = TileTensor(outp, row_major[B, OUT]())
        bl.forward["cpu", B](it, output=ot)

    @staticmethod
    def _rms[
        DIM: Int, B: Int
    ](
        mut rn: RMSNorm[DIM],
        inp: UnsafePointer[Scalar[DT], MutAnyOrigin],
        outp: UnsafePointer[Scalar[DT], MutAnyOrigin],
    ) raises:
        var it = TileTensor(inp, row_major[B, DIM]())
        var ot = TileTensor(outp, row_major[B, DIM]())
        rn.forward["cpu", B](it, output=ot)

    # ── _core: (deter, stoch, action) → new_deter ────────────────────

    def core[
        B: Int
    ](
        mut self,
        deter: UnsafePointer[Scalar[DT], MutAnyOrigin],     # [B, DETER]
        stoch: UnsafePointer[Scalar[DT], MutAnyOrigin],     # [B, SC]
        action: UnsafePointer[Scalar[DT], MutAnyOrigin],    # [B, ACT]
        out_deter: UnsafePointer[Scalar[DT], MutAnyOrigin],  # [B, DETER]
    ) raises:
        comptime H = Self.HIDDEN
        comptime D = Self.DETER
        comptime g = Self.BLOCKS
        comptime dpb = Self.DPB

        # action squash (elementwise): a = action / max(1, |action|)
        var a = alloc[Scalar[DT]](B * Self.ACT)
        for i in range(B * Self.ACT):
            var v = action[i]
            var av = v if v >= Scalar[DT](0.0) else -v
            var denom = av if av > Scalar[DT](1.0) else Scalar[DT](1.0)
            a[i] = v / denom

        # x0/x1/x2 = gelu(rms(lin(.)))
        var x0 = alloc[Scalar[DT]](B * H)
        var t0 = alloc[Scalar[DT]](B * H)
        Self._lin[Self.DETER, H, B](self.dynin0, deter, t0)
        Self._rms[H, B](self.dynin0n, t0, x0)
        _gelu_buf(x0, B * H)

        var x1 = alloc[Scalar[DT]](B * H)
        var t1 = alloc[Scalar[DT]](B * H)
        Self._lin[Self.SC, H, B](self.dynin1, stoch, t1)
        Self._rms[H, B](self.dynin1n, t1, x1)
        _gelu_buf(x1, B * H)

        var x2 = alloc[Scalar[DT]](B * H)
        var t2 = alloc[Scalar[DT]](B * H)
        Self._lin[Self.ACT, H, B](self.dynin2, a, t2)
        Self._rms[H, B](self.dynin2n, t2, x2)
        _gelu_buf(x2, B * H)

        # block-interleaved dynhid input [B, DHIN]:
        #   group g = [ deter[g·dpb : g·dpb+dpb] , x0[b] , x1[b] , x2[b] ]
        var dhin = alloc[Scalar[DT]](B * Self.DHIN)
        comptime per_group = dpb + 3 * H
        for b in range(B):
            var gbase = b * Self.DHIN
            for grp in range(g):
                var off = gbase + grp * per_group
                for k in range(dpb):
                    dhin[off + k] = deter[b * D + grp * dpb + k]
                for k in range(H):
                    dhin[off + dpb + k] = x0[b * H + k]
                for k in range(H):
                    dhin[off + dpb + H + k] = x1[b * H + k]
                for k in range(H):
                    dhin[off + dpb + 2 * H + k] = x2[b * H + k]

        # h = gelu(rms(dynhid0(dhin)))   [B, DETER]
        var hraw = alloc[Scalar[DT]](B * D)
        var hn = alloc[Scalar[DT]](B * D)
        Self._bl[Self.DHIN, D, g, B](self.dynhid0, dhin, hraw)
        Self._rms[D, B](self.dynhid0n, hraw, hn)
        _gelu_buf(hn, B * D)

        # gru = dyngru(h)   [B, 3·DETER], block-interleaved
        var gru = alloc[Scalar[DT]](B * Self.GRU_OUT)
        Self._bl[D, Self.GRU_OUT, g, B](self.dyngru, hn, gru)

        # de-interleave gates within each block's 3·dpb lanes, apply GRU
        comptime opb = 3 * dpb
        for b in range(B):
            for grp in range(g):
                var go = b * Self.GRU_OUT + grp * opb
                var dbase = b * D + grp * dpb
                for k in range(dpb):
                    var r = _sigmoid_scalar(gru[go + k])
                    var c = tanh(r * gru[go + dpb + k])
                    var u = _sigmoid_scalar(gru[go + 2 * dpb + k] - Scalar[DT](1.0))
                    var d_prev = deter[dbase + k]
                    out_deter[dbase + k] = u * c + (Scalar[DT](1.0) - u) * d_prev

        a.free(); x0.free(); t0.free(); x1.free(); t1.free(); x2.free()
        t2.free(); dhin.free(); hraw.free(); hn.free(); gru.free()

    def core_vjp[
        B: Int
    ](
        mut self,
        deter: UnsafePointer[Scalar[DT], MutAnyOrigin],     # [B, DETER]
        stoch: UnsafePointer[Scalar[DT], MutAnyOrigin],     # [B, SC]
        action: UnsafePointer[Scalar[DT], MutAnyOrigin],    # [B, ACT]
        grad_out: UnsafePointer[Scalar[DT], MutAnyOrigin],  # [B, DETER]
        grad_deter: UnsafePointer[Scalar[DT], MutAnyOrigin],   # [B, DETER]
        grad_stoch: UnsafePointer[Scalar[DT], MutAnyOrigin],   # [B, SC]
        grad_action: UnsafePointer[Scalar[DT], MutAnyOrigin],  # [B, ACT]
    ) raises:
        """Backward of `core` (recompute). grad_deter receives three paths
        (GRU mix bypass + dynhid deter-slice + dynin0), grad_stoch from
        dynin1, grad_action from dynin2 (through the sg'd squash). Param
        grads accumulate in the dyn* leaves (zeroed by the vjp helpers)."""
        comptime H = Self.HIDDEN
        comptime D = Self.DETER
        comptime g = Self.BLOCKS
        comptime dpb = Self.DPB
        comptime opb = 3 * dpb
        comptime per_group = dpb + 3 * H

        # ── recompute forward, retaining caches ──────────────────────────
        var a: UnsafePointer[Scalar[DT], MutAnyOrigin] = alloc[Scalar[DT]](
            B * Self.ACT
        )
        for i in range(B * Self.ACT):
            var v = action[i]
            var av = v if v >= Scalar[DT](0.0) else -v
            var denom = av if av > Scalar[DT](1.0) else Scalar[DT](1.0)
            a[i] = v / denom
        var n0d: UnsafePointer[Scalar[DT], MutAnyOrigin] = alloc[Scalar[DT]](B * H)
        var x0: UnsafePointer[Scalar[DT], MutAnyOrigin] = alloc[Scalar[DT]](B * H)
        var n1d: UnsafePointer[Scalar[DT], MutAnyOrigin] = alloc[Scalar[DT]](B * H)
        var x1: UnsafePointer[Scalar[DT], MutAnyOrigin] = alloc[Scalar[DT]](B * H)
        var n2d: UnsafePointer[Scalar[DT], MutAnyOrigin] = alloc[Scalar[DT]](B * H)
        var x2: UnsafePointer[Scalar[DT], MutAnyOrigin] = alloc[Scalar[DT]](B * H)
        var tmp: UnsafePointer[Scalar[DT], MutAnyOrigin] = alloc[Scalar[DT]](B * H)
        Self._lin[Self.DETER, H, B](self.dynin0, deter, tmp)
        Self._rms[H, B](self.dynin0n, tmp, n0d)
        for i in range(B * H):
            x0[i] = n0d[i]
        _gelu_buf(x0, B * H)
        var tmp1: UnsafePointer[Scalar[DT], MutAnyOrigin] = alloc[Scalar[DT]](B * H)
        Self._lin[Self.SC, H, B](self.dynin1, stoch, tmp1)
        Self._rms[H, B](self.dynin1n, tmp1, n1d)
        for i in range(B * H):
            x1[i] = n1d[i]
        _gelu_buf(x1, B * H)
        var tmp2: UnsafePointer[Scalar[DT], MutAnyOrigin] = alloc[Scalar[DT]](B * H)
        Self._lin[Self.ACT, H, B](self.dynin2, a, tmp2)
        Self._rms[H, B](self.dynin2n, tmp2, n2d)
        for i in range(B * H):
            x2[i] = n2d[i]
        _gelu_buf(x2, B * H)

        var dhin: UnsafePointer[Scalar[DT], MutAnyOrigin] = alloc[Scalar[DT]](
            B * Self.DHIN
        )
        for b in range(B):
            var gbase = b * Self.DHIN
            for grp in range(g):
                var off = gbase + grp * per_group
                for k in range(dpb):
                    dhin[off + k] = deter[b * D + grp * dpb + k]
                for k in range(H):
                    dhin[off + dpb + k] = x0[b * H + k]
                for k in range(H):
                    dhin[off + dpb + H + k] = x1[b * H + k]
                for k in range(H):
                    dhin[off + dpb + 2 * H + k] = x2[b * H + k]
        var hn: UnsafePointer[Scalar[DT], MutAnyOrigin] = alloc[Scalar[DT]](B * D)
        var h: UnsafePointer[Scalar[DT], MutAnyOrigin] = alloc[Scalar[DT]](B * D)
        var hrawb: UnsafePointer[Scalar[DT], MutAnyOrigin] = alloc[Scalar[DT]](B * D)
        Self._bl[Self.DHIN, D, g, B](self.dynhid0, dhin, hrawb)
        Self._rms[D, B](self.dynhid0n, hrawb, hn)
        for i in range(B * D):
            h[i] = hn[i]
        _gelu_buf(h, B * D)
        var gru: UnsafePointer[Scalar[DT], MutAnyOrigin] = alloc[Scalar[DT]](
            B * Self.GRU_OUT
        )
        Self._bl[D, Self.GRU_OUT, g, B](self.dyngru, h, gru)

        # ── backward ─────────────────────────────────────────────────────
        for i in range(B * D):
            grad_deter[i] = 0.0
        for i in range(B * Self.SC):
            grad_stoch[i] = 0.0
        for i in range(B * Self.ACT):
            grad_action[i] = 0.0

        # step 6: GRU gates → grad_gru + grad_deter (mix bypass)
        var grad_gru: UnsafePointer[Scalar[DT], MutAnyOrigin] = alloc[Scalar[DT]](
            B * Self.GRU_OUT
        )
        for i in range(B * Self.GRU_OUT):
            grad_gru[i] = 0.0
        for b in range(B):
            for grp in range(g):
                var gb = b * Self.GRU_OUT + grp * opb
                var dbase = b * D + grp * dpb
                for k in range(dpb):
                    var r_pre = gru[gb + k]
                    var c_pre = gru[gb + dpb + k]
                    var u_pre = gru[gb + 2 * dpb + k] - Scalar[DT](1.0)
                    var reset = _sigmoid_scalar(r_pre)
                    var m = reset * c_pre
                    var cand = tanh(m)
                    var update = _sigmoid_scalar(u_pre)
                    var d_prev = deter[dbase + k]
                    var go = grad_out[dbase + k]
                    var d_update = go * (cand - d_prev)
                    var d_cand = go * update
                    grad_deter[dbase + k] += go * (Scalar[DT](1.0) - update)
                    var d_m = d_cand * (Scalar[DT](1.0) - cand * cand)
                    var d_reset = d_m * c_pre
                    var d_cpre = d_m * reset
                    var d_rpre = d_reset * reset * (Scalar[DT](1.0) - reset)
                    var d_upre = d_update * update * (Scalar[DT](1.0) - update)
                    grad_gru[gb + k] += d_rpre
                    grad_gru[gb + dpb + k] += d_cpre
                    grad_gru[gb + 2 * dpb + k] += d_upre

        # step 5: dyngru.vjp → grad_h
        var grad_h: UnsafePointer[Scalar[DT], MutAnyOrigin] = alloc[Scalar[DT]](
            B * D
        )
        bl_vjp[D, Self.GRU_OUT, g, B](self.dyngru, grad_gru, grad_h)

        # step 4: gelu' → dynhid0n.vjp → dynhid0.vjp → d_dhin
        for i in range(B * D):
            grad_h[i] = grad_h[i] * _gelu_grad(hn[i])
        var d_hraw: UnsafePointer[Scalar[DT], MutAnyOrigin] = alloc[Scalar[DT]](
            B * D
        )
        rms_vjp[D, B](self.dynhid0n, grad_h, d_hraw)
        var d_dhin: UnsafePointer[Scalar[DT], MutAnyOrigin] = alloc[Scalar[DT]](
            B * Self.DHIN
        )
        bl_vjp[Self.DHIN, D, g, B](self.dynhid0, d_hraw, d_dhin)

        # step 3: de-interleave → grad_deter (slice) + d_x0/x1/x2 (sum blocks)
        var d_x0: UnsafePointer[Scalar[DT], MutAnyOrigin] = alloc[Scalar[DT]](B * H)
        var d_x1: UnsafePointer[Scalar[DT], MutAnyOrigin] = alloc[Scalar[DT]](B * H)
        var d_x2: UnsafePointer[Scalar[DT], MutAnyOrigin] = alloc[Scalar[DT]](B * H)
        for i in range(B * H):
            d_x0[i] = 0.0
            d_x1[i] = 0.0
            d_x2[i] = 0.0
        for b in range(B):
            for grp in range(g):
                var off = b * Self.DHIN + grp * per_group
                for k in range(dpb):
                    grad_deter[b * D + grp * dpb + k] += d_dhin[off + k]
                for k in range(H):
                    d_x0[b * H + k] += d_dhin[off + dpb + k]
                    d_x1[b * H + k] += d_dhin[off + dpb + H + k]
                    d_x2[b * H + k] += d_dhin[off + dpb + 2 * H + k]

        # step 2: x0 → deter (accumulate), x1 → stoch, x2 → squashed action
        for i in range(B * H):
            d_x0[i] = d_x0[i] * _gelu_grad(n0d[i])
        var d_a0: UnsafePointer[Scalar[DT], MutAnyOrigin] = alloc[Scalar[DT]](B * H)
        rms_vjp[H, B](self.dynin0n, d_x0, d_a0)
        var d_deter0: UnsafePointer[Scalar[DT], MutAnyOrigin] = alloc[Scalar[DT]](
            B * D
        )
        lin_vjp[Self.DETER, H, B](self.dynin0, d_a0, d_deter0)
        for i in range(B * D):
            grad_deter[i] += d_deter0[i]

        for i in range(B * H):
            d_x1[i] = d_x1[i] * _gelu_grad(n1d[i])
        var d_a1b: UnsafePointer[Scalar[DT], MutAnyOrigin] = alloc[Scalar[DT]](B * H)
        rms_vjp[H, B](self.dynin1n, d_x1, d_a1b)
        lin_vjp[Self.SC, H, B](self.dynin1, d_a1b, grad_stoch)

        for i in range(B * H):
            d_x2[i] = d_x2[i] * _gelu_grad(n2d[i])
        var d_a2b: UnsafePointer[Scalar[DT], MutAnyOrigin] = alloc[Scalar[DT]](B * H)
        rms_vjp[H, B](self.dynin2n, d_x2, d_a2b)
        var d_asq: UnsafePointer[Scalar[DT], MutAnyOrigin] = alloc[Scalar[DT]](
            B * Self.ACT
        )
        lin_vjp[Self.ACT, H, B](self.dynin2, d_a2b, d_asq)

        # step 1: squash backward (denom is sg → constant)
        for i in range(B * Self.ACT):
            var v = action[i]
            var av = v if v >= Scalar[DT](0.0) else -v
            var denom = av if av > Scalar[DT](1.0) else Scalar[DT](1.0)
            grad_action[i] = d_asq[i] / denom

        a.free(); n0d.free(); x0.free(); n1d.free(); x1.free(); n2d.free()
        x2.free(); tmp.free(); tmp1.free(); tmp2.free(); dhin.free()
        hn.free(); h.free(); hrawb.free(); gru.free(); grad_gru.free()
        grad_h.free(); d_hraw.free(); d_dhin.free(); d_x0.free(); d_x1.free()
        d_x2.free(); d_a0.free(); d_deter0.free(); d_a1b.free(); d_a2b.free()
        d_asq.free()

    # ── _prior: deter → logit [B, SC] ────────────────────────────────

    def prior[
        B: Int
    ](
        mut self,
        deter: UnsafePointer[Scalar[DT], MutAnyOrigin],     # [B, DETER]
        out_logit: UnsafePointer[Scalar[DT], MutAnyOrigin],  # [B, SC]
    ) raises:
        comptime H = Self.HIDDEN
        var a0 = alloc[Scalar[DT]](B * H)
        var n0 = alloc[Scalar[DT]](B * H)
        Self._lin[Self.DETER, H, B](self.prior0, deter, a0)
        Self._rms[H, B](self.prior0n, a0, n0)
        _gelu_buf(n0, B * H)
        var a1 = alloc[Scalar[DT]](B * H)
        var n1 = alloc[Scalar[DT]](B * H)
        Self._lin[H, H, B](self.prior1, n0, a1)
        Self._rms[H, B](self.prior1n, a1, n1)
        _gelu_buf(n1, B * H)
        Self._lin[H, Self.SC, B](self.priorlogit, n1, out_logit)
        a0.free(); n0.free(); a1.free(); n1.free()

    def prior_vjp[
        B: Int
    ](
        mut self,
        deter: UnsafePointer[Scalar[DT], MutAnyOrigin],       # [B, DETER]
        grad_logit: UnsafePointer[Scalar[DT], MutAnyOrigin],  # [B, SC]
        grad_deter: UnsafePointer[Scalar[DT], MutAnyOrigin],  # [B, DETER]
    ) raises:
        """Backward of `prior` (recompute). Param grads accumulate in the
        prior leaves (zeroed by the vjp helpers)."""
        comptime H = Self.HIDDEN
        var a0: UnsafePointer[Scalar[DT], MutAnyOrigin] = alloc[Scalar[DT]](B * H)
        var n0: UnsafePointer[Scalar[DT], MutAnyOrigin] = alloc[Scalar[DT]](B * H)
        var g0: UnsafePointer[Scalar[DT], MutAnyOrigin] = alloc[Scalar[DT]](B * H)
        var a1: UnsafePointer[Scalar[DT], MutAnyOrigin] = alloc[Scalar[DT]](B * H)
        var n1: UnsafePointer[Scalar[DT], MutAnyOrigin] = alloc[Scalar[DT]](B * H)
        var g1: UnsafePointer[Scalar[DT], MutAnyOrigin] = alloc[Scalar[DT]](B * H)
        var lg: UnsafePointer[Scalar[DT], MutAnyOrigin] = alloc[Scalar[DT]](
            B * Self.SC
        )
        lin_fwd[Self.DETER, H, B](self.prior0, deter, a0)
        rms_fwd[H, B](self.prior0n, a0, n0)
        for i in range(B * H):
            g0[i] = n0[i]
        _gelu_buf(g0, B * H)
        lin_fwd[H, H, B](self.prior1, g0, a1)
        rms_fwd[H, B](self.prior1n, a1, n1)
        for i in range(B * H):
            g1[i] = n1[i]
        _gelu_buf(g1, B * H)
        lin_fwd[H, Self.SC, B](self.priorlogit, g1, lg)  # refresh head cache

        var d_g1: UnsafePointer[Scalar[DT], MutAnyOrigin] = alloc[Scalar[DT]](B * H)
        lin_vjp[H, Self.SC, B](self.priorlogit, grad_logit, d_g1)
        for i in range(B * H):
            d_g1[i] = d_g1[i] * _gelu_grad(n1[i])
        var d_a1: UnsafePointer[Scalar[DT], MutAnyOrigin] = alloc[Scalar[DT]](B * H)
        rms_vjp[H, B](self.prior1n, d_g1, d_a1)
        var d_g0: UnsafePointer[Scalar[DT], MutAnyOrigin] = alloc[Scalar[DT]](B * H)
        lin_vjp[H, H, B](self.prior1, d_a1, d_g0)
        for i in range(B * H):
            d_g0[i] = d_g0[i] * _gelu_grad(n0[i])
        var d_a0: UnsafePointer[Scalar[DT], MutAnyOrigin] = alloc[Scalar[DT]](B * H)
        rms_vjp[H, B](self.prior0n, d_g0, d_a0)
        lin_vjp[Self.DETER, H, B](self.prior0, d_a0, grad_deter)

        a0.free(); n0.free(); g0.free(); a1.free(); n1.free(); g1.free()
        lg.free(); d_g1.free(); d_a1.free(); d_g0.free(); d_a0.free()

    # ── observe (single step, reset=False): → (new_deter, obslogit) ──

    def observe[
        B: Int
    ](
        mut self,
        deter: UnsafePointer[Scalar[DT], MutAnyOrigin],     # [B, DETER]
        stoch: UnsafePointer[Scalar[DT], MutAnyOrigin],     # [B, SC]
        action: UnsafePointer[Scalar[DT], MutAnyOrigin],    # [B, ACT]
        tokens: UnsafePointer[Scalar[DT], MutAnyOrigin],    # [B, TOKEN]
        out_deter: UnsafePointer[Scalar[DT], MutAnyOrigin],   # [B, DETER]
        out_logit: UnsafePointer[Scalar[DT], MutAnyOrigin],   # [B, SC]
    ) raises:
        comptime H = Self.HIDDEN
        comptime D = Self.DETER
        self.core[B](deter, stoch, action, out_deter)
        # x = concat([new_deter, tokens])  [B, DETER+TOKEN]
        var x = alloc[Scalar[DT]](B * Self.OBSIN)
        for b in range(B):
            for k in range(D):
                x[b * Self.OBSIN + k] = out_deter[b * D + k]
            for k in range(Self.TOKEN):
                x[b * Self.OBSIN + D + k] = tokens[b * Self.TOKEN + k]
        var o0 = alloc[Scalar[DT]](B * H)
        var on0 = alloc[Scalar[DT]](B * H)
        Self._lin[Self.OBSIN, H, B](self.obs0, x, o0)
        Self._rms[H, B](self.obs0n, o0, on0)
        _gelu_buf(on0, B * H)
        Self._lin[H, Self.SC, B](self.obslogit, on0, out_logit)
        x.free(); o0.free(); on0.free()

    def observe_vjp[
        B: Int
    ](
        mut self,
        deter: UnsafePointer[Scalar[DT], MutAnyOrigin],     # [B, DETER]
        stoch: UnsafePointer[Scalar[DT], MutAnyOrigin],     # [B, SC]
        action: UnsafePointer[Scalar[DT], MutAnyOrigin],    # [B, ACT]
        tokens: UnsafePointer[Scalar[DT], MutAnyOrigin],    # [B, TOKEN]
        grad_logit: UnsafePointer[Scalar[DT], MutAnyOrigin],     # [B, SC]
        grad_new_deter: UnsafePointer[Scalar[DT], MutAnyOrigin],  # [B, DETER]
        grad_deter: UnsafePointer[Scalar[DT], MutAnyOrigin],   # [B, DETER]
        grad_stoch: UnsafePointer[Scalar[DT], MutAnyOrigin],   # [B, SC]
        grad_action: UnsafePointer[Scalar[DT], MutAnyOrigin],  # [B, ACT]
        grad_tokens: UnsafePointer[Scalar[DT], MutAnyOrigin],  # [B, TOKEN]
    ) raises:
        """Backward of `observe`. `grad_logit` is upstream on the obslogit;
        `grad_new_deter` is extra upstream on the new deter (from prior /
        decoder / heads). The obs-head backward produces grad on the deter,
        which is summed with `grad_new_deter` and fed to `core_vjp`. Obs-head
        params accumulate in obs0/obslogit; dyn* in core_vjp."""
        comptime H = Self.HIDDEN
        comptime D = Self.DETER
        # recompute new_deter (obs path needs it as input)
        var new_deter: UnsafePointer[Scalar[DT], MutAnyOrigin] = alloc[
            Scalar[DT]
        ](B * D)
        self.core[B](deter, stoch, action, new_deter)
        # recompute obs path, retaining caches
        var x: UnsafePointer[Scalar[DT], MutAnyOrigin] = alloc[Scalar[DT]](
            B * Self.OBSIN
        )
        for b in range(B):
            for k in range(D):
                x[b * Self.OBSIN + k] = new_deter[b * D + k]
            for k in range(Self.TOKEN):
                x[b * Self.OBSIN + D + k] = tokens[b * Self.TOKEN + k]
        var o0: UnsafePointer[Scalar[DT], MutAnyOrigin] = alloc[Scalar[DT]](B * H)
        var on0: UnsafePointer[Scalar[DT], MutAnyOrigin] = alloc[Scalar[DT]](B * H)
        var g: UnsafePointer[Scalar[DT], MutAnyOrigin] = alloc[Scalar[DT]](B * H)
        var lg: UnsafePointer[Scalar[DT], MutAnyOrigin] = alloc[Scalar[DT]](
            B * Self.SC
        )
        lin_fwd[Self.OBSIN, H, B](self.obs0, x, o0)
        rms_fwd[H, B](self.obs0n, o0, on0)
        for i in range(B * H):
            g[i] = on0[i]
        _gelu_buf(g, B * H)
        lin_fwd[H, Self.SC, B](self.obslogit, g, lg)  # refresh head cache

        # obs-head backward → grad on x → split deter/tokens
        var d_g: UnsafePointer[Scalar[DT], MutAnyOrigin] = alloc[Scalar[DT]](B * H)
        lin_vjp[H, Self.SC, B](self.obslogit, grad_logit, d_g)
        for i in range(B * H):
            d_g[i] = d_g[i] * _gelu_grad(on0[i])
        var d_o0: UnsafePointer[Scalar[DT], MutAnyOrigin] = alloc[Scalar[DT]](B * H)
        rms_vjp[H, B](self.obs0n, d_g, d_o0)
        var d_x: UnsafePointer[Scalar[DT], MutAnyOrigin] = alloc[Scalar[DT]](
            B * Self.OBSIN
        )
        lin_vjp[Self.OBSIN, H, B](self.obs0, d_o0, d_x)

        # total grad on new_deter = obs-path + extra upstream
        var g_nd: UnsafePointer[Scalar[DT], MutAnyOrigin] = alloc[Scalar[DT]](B * D)
        for b in range(B):
            for k in range(D):
                g_nd[b * D + k] = (
                    d_x[b * Self.OBSIN + k] + grad_new_deter[b * D + k]
                )
            for k in range(Self.TOKEN):
                grad_tokens[b * Self.TOKEN + k] = d_x[b * Self.OBSIN + D + k]

        self.core_vjp[B](
            deter, stoch, action, g_nd, grad_deter, grad_stoch, grad_action
        )
        new_deter.free(); x.free(); o0.free(); on0.free(); g.free(); lg.free()
        d_g.free(); d_o0.free(); d_x.free(); g_nd.free()

    # ── loss: (post=obslogit, prior=prior(new_deter)) → dyn/rep [B] ──

    def loss[
        B: Int
    ](
        mut self,
        deter: UnsafePointer[Scalar[DT], MutAnyOrigin],     # [B, DETER]
        stoch: UnsafePointer[Scalar[DT], MutAnyOrigin],     # [B, SC]
        action: UnsafePointer[Scalar[DT], MutAnyOrigin],    # [B, ACT]
        tokens: UnsafePointer[Scalar[DT], MutAnyOrigin],    # [B, TOKEN]
        mut dyn_out: UnsafePointer[Scalar[DT], MutAnyOrigin],    # [B]
        mut rep_out: UnsafePointer[Scalar[DT], MutAnyOrigin],    # [B]
    ) raises:
        var new_deter = alloc[Scalar[DT]](B * Self.DETER)
        var post = alloc[Scalar[DT]](B * Self.SC)
        var prior_l = alloc[Scalar[DT]](B * Self.SC)
        self.observe[B](deter, stoch, action, tokens, new_deter, post)
        self.prior[B](new_deter, prior_l)
        self.kl.forward[B](post, prior_l, dyn_out, rep_out)
        new_deter.free(); post.free(); prior_l.free()

    def loss_vjp[
        B: Int
    ](
        mut self,
        deter: UnsafePointer[Scalar[DT], MutAnyOrigin],     # [B, DETER]
        stoch: UnsafePointer[Scalar[DT], MutAnyOrigin],     # [B, SC]
        action: UnsafePointer[Scalar[DT], MutAnyOrigin],    # [B, ACT]
        tokens: UnsafePointer[Scalar[DT], MutAnyOrigin],    # [B, TOKEN]
        d_dyn: UnsafePointer[Scalar[DT], MutAnyOrigin],     # [B] cotangent
        d_rep: UnsafePointer[Scalar[DT], MutAnyOrigin],     # [B] cotangent
        grad_deter: UnsafePointer[Scalar[DT], MutAnyOrigin],   # [B, DETER]
        grad_stoch: UnsafePointer[Scalar[DT], MutAnyOrigin],   # [B, SC]
        grad_action: UnsafePointer[Scalar[DT], MutAnyOrigin],  # [B, ACT]
        grad_tokens: UnsafePointer[Scalar[DT], MutAnyOrigin],  # [B, TOKEN]
    ) raises:
        """Full WM dyn/rep loss backward: OneHotKL.backward → {prior_vjp,
        observe_vjp}. Accumulates grads in every RSSM leaf (obs/prior/dyn).
        Validated end-to-end vs jax.vjp of the deterministic dyn+rep loss."""
        var new_deter: UnsafePointer[Scalar[DT], MutAnyOrigin] = alloc[
            Scalar[DT]
        ](B * Self.DETER)
        var post: UnsafePointer[Scalar[DT], MutAnyOrigin] = alloc[Scalar[DT]](
            B * Self.SC
        )
        var prior_l: UnsafePointer[Scalar[DT], MutAnyOrigin] = alloc[Scalar[DT]](
            B * Self.SC
        )
        var dyn_tmp: UnsafePointer[Scalar[DT], MutAnyOrigin] = alloc[Scalar[DT]](B)
        var rep_tmp: UnsafePointer[Scalar[DT], MutAnyOrigin] = alloc[Scalar[DT]](B)
        self.observe[B](deter, stoch, action, tokens, new_deter, post)
        self.prior[B](new_deter, prior_l)
        # forward KL to populate the OneHotKL cache + active gate.
        self.kl.forward[B](post, prior_l, dyn_tmp, rep_tmp)

        var grad_post: UnsafePointer[Scalar[DT], MutAnyOrigin] = alloc[Scalar[DT]](
            B * Self.SC
        )
        var grad_prior: UnsafePointer[Scalar[DT], MutAnyOrigin] = alloc[
            Scalar[DT]
        ](B * Self.SC)
        self.kl.backward[B](d_dyn, d_rep, grad_post, grad_prior)

        # prior path: grad on prior logit → prior params + grad on new_deter
        var g_nd_prior: UnsafePointer[Scalar[DT], MutAnyOrigin] = alloc[
            Scalar[DT]
        ](B * Self.DETER)
        self.prior_vjp[B](new_deter, grad_prior, g_nd_prior)

        # obs path (+ core): grad on obslogit + extra grad on new_deter
        self.observe_vjp[B](
            deter, stoch, action, tokens, grad_post, g_nd_prior,
            grad_deter, grad_stoch, grad_action, grad_tokens,
        )

        new_deter.free(); post.free(); prior_l.free(); dyn_tmp.free()
        rep_tmp.free(); grad_post.free(); grad_prior.free(); g_nd_prior.free()
