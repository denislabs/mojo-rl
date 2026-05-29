"""DreamerV3 trainer blocks — SAC-style composable make/step units.

Mirrors `deep_agents2/sac/` + `training/blocks/`: each block is a concrete
struct with `make[target](ctx)` + `step[target](mut state, mut <modules/opts>)`.
`target: StaticString` is comptime ("cpu"/"gpu"); `ctx: Optional[DeviceContext]`
is threaded at runtime via `DreamerState`. CPU/GPU split inside each step is a
`comptime if target == "cpu": ... else: ...`.

v1 lands the CPU branch (the validated `spike_wm_bptt` / `spike_wm_imag_ac`
logic). The GPU branch is gated until PR5c Step 5 adds GPU kernels to the 5
custom RSSM ops (ActionSquash / BlockGroupAssemble / GRUGate / OneHotKLLoss /
StraightThroughSample) — everything else (Linear/BlockLinear/RMSNorm/
Sequential/ComputeGraph/DreamerOpt) is already GPU. The block structure means
that swap is localized to each `else:` branch; the trainer/composition + the
target/ctx plumbing are GPU-ready now.

Shared state (`DreamerState`) holds the cross-block buffers (sampled batch,
RSSM carries, imagination noise) + inter-block scalars + ctx. v1 buffers are
plain CPU allocs sized once in `make`; the GPU upgrade swaps them for
`Scratch[..]` device buffers (same `target_ptr[target]` access shape SAC uses).
"""

from std.memory import alloc
from std.math import tanh, exp
from layout import TileTensor, row_major
from std.gpu.host import DeviceContext

from mojo_rl.nn2.constants import DT
from mojo_rl.nn2.primitives.ops.swish_op import SwishOp
from mojo_rl.deep_agents2.dreamerv3.twohot import twohot_pred
from mojo_rl.deep_agents2.dreamerv3.dists import bounded_std
from mojo_rl.deep_agents2.dreamerv3.normalize import PercentileNormalize
from mojo_rl.deep_agents2.dreamerv3.imag_loss import (
    imag_loss_cpu, imag_loss_backward,
)
from mojo_rl.deep_agents2.dreamerv3.param_sync import (
    collect_params, apply_params,
)
from mojo_rl.deep_agents2.dreamerv3.polyak import polyak_module
from mojo_rl.deep_agents2.dreamerv3.wm import (
    WMCoreGraph, WMImagineGraph, DecLossGraph, RewLossGraph, ConLossGraph,
)
from mojo_rl.deep_agents2.dreamerv3.nets import (
    DreamerEncoder, DreamerValue, DreamerPolicy,
)
from mojo_rl.nn2.optimizer.dreamer_opt import DreamerOpt


@always_inline
def _alloc(n: Int) -> UnsafePointer[Scalar[DT], MutAnyOrigin]:
    return alloc[Scalar[DT]](n)


# ──────────────────────────────────────────────────────────────────────
# DreamerState — cross-block shared buffers + ctx + inter-block scalars.
# v1 CPU allocs (sized once); GPU upgrade → Scratch device buffers.
# ──────────────────────────────────────────────────────────────────────


@fieldwise_init
struct DreamerState[
    OBS: Int, ACT: Int, DETER: Int, SC: Int, TOKEN: Int,
    B: Int, T: Int, T_IMAG: Int,
](Movable & ImplicitlyDestructible):
    var ctx: Optional[DeviceContext]
    # sampled batch (filled by the trainer from replay)
    var mb_obs: UnsafePointer[Scalar[DT], MutAnyOrigin]   # [B,T+1,OBS]
    var mb_act: UnsafePointer[Scalar[DT], MutAnyOrigin]   # [B,T,ACT]
    var mb_rew: UnsafePointer[Scalar[DT], MutAnyOrigin]   # [B,T]
    var mb_dne: UnsafePointer[Scalar[DT], MutAnyOrigin]   # [B,T]
    # RSSM carries (WMStep fills; ACStep reads the final one)
    var cdeter: UnsafePointer[Scalar[DT], MutAnyOrigin]   # [(T+1)*B*DETER]
    var cstoch: UnsafePointer[Scalar[DT], MutAnyOrigin]   # [(T+1)*B*SC]
    var toks: UnsafePointer[Scalar[DT], MutAnyOrigin]     # [T*B*TOKEN]
    var noise: UnsafePointer[Scalar[DT], MutAnyOrigin]    # [T_IMAG*B*ACT]
    var last_wm_loss: Scalar[DT]
    var last_ac_loss: Scalar[DT]

    @staticmethod
    def make[target: StaticString](ctx: Optional[DeviceContext] = None) raises -> Self:
        comptime assert target == "cpu", (
            "DreamerState: GPU buffers land in PR5c Step 5 (Scratch upgrade)"
        )
        return Self(
            ctx=ctx,
            mb_obs=_alloc(Self.B * (Self.T + 1) * Self.OBS),
            mb_act=_alloc(Self.B * Self.T * Self.ACT),
            mb_rew=_alloc(Self.B * Self.T),
            mb_dne=_alloc(Self.B * Self.T),
            cdeter=_alloc((Self.T + 1) * Self.B * Self.DETER),
            cstoch=_alloc((Self.T + 1) * Self.B * Self.SC),
            toks=_alloc(Self.T * Self.B * Self.TOKEN),
            noise=_alloc(Self.T_IMAG * Self.B * Self.ACT),
            last_wm_loss=Scalar[DT](0.0),
            last_ac_loss=Scalar[DT](0.0),
        )


# ──────────────────────────────────────────────────────────────────────
# WMStep — WM-BPTT over one sampled length-T window. Trains enc/core/dec/
# rew/con; fills state.cdeter / cstoch with the posterior carry sequence.
# ──────────────────────────────────────────────────────────────────────


@fieldwise_init
struct WMStep[
    OBS: Int, ACT: Int, DETER: Int, H: Int, STOCH: Int, CLASSES: Int,
    BLOCKS: Int, TOKEN: Int, DEC_U: Int, HU: Int, BINS: Int, B: Int, T: Int,
](Movable & ImplicitlyDestructible):
    comptime SC = Self.STOCH * Self.CLASSES
    comptime CARRY = 2 + Self.DETER + Self.SC
    comptime EncT = DreamerEncoder[Self.OBS, Self.TOKEN, SwishOp]
    comptime CoreT = WMCoreGraph[
        Self.DETER, Self.H, Self.STOCH, Self.CLASSES, Self.BLOCKS, Self.ACT,
        Self.TOKEN, SwishOp,
    ]
    comptime DecT = DecLossGraph[Self.SC, Self.DETER, Self.OBS, Self.DEC_U, SwishOp]
    comptime RewT = RewLossGraph[Self.DETER, Self.SC, Self.HU, Self.BINS, SwishOp]
    comptime ConT = ConLossGraph[Self.DETER, Self.SC, Self.HU, SwishOp]
    comptime StateT = DreamerState[
        Self.OBS, Self.ACT, Self.DETER, Self.SC, Self.TOKEN, Self.B, Self.T, 1,
    ]
    var dyn_scale: Scalar[DT]
    var rep_scale: Scalar[DT]

    @staticmethod
    def make[target: StaticString](ctx: Optional[DeviceContext] = None) raises -> Self:
        return Self(dyn_scale=Scalar[DT](1.0), rep_scale=Scalar[DT](0.1))

    def step[
        target: StaticString, T_IMAG: Int,
    ](
        mut self,
        mut st: DreamerState[Self.OBS, Self.ACT, Self.DETER, Self.SC, Self.TOKEN, Self.B, Self.T, T_IMAG],
        mut enc: Self.EncT,
        mut core: Self.CoreT,
        mut dec: Self.DecT,
        mut rew: Self.RewT,
        mut con: Self.ConT,
        mut oe: DreamerOpt,
        mut ocore: DreamerOpt,
        mut odec: DreamerOpt,
        mut orew: DreamerOpt,
        mut ocon: DreamerOpt,
    ) raises:
        comptime assert target == "cpu", (
            "WMStep: GPU branch lands in PR5c Step 5"
        )
        comptime D = Self.DETER
        comptime SCl = Self.SC
        comptime TOK = Self.TOKEN
        comptime CARRYl = Self.CARRY
        comptime OBSD = Self.OBS
        comptime ACTD = Self.ACT
        var DYN = self.dyn_scale
        var REP = self.rep_scale
        var obs = st.mb_obs
        var act = st.mb_act
        var rew_t = st.mb_rew
        var dne = st.mb_dne
        var cdeter = st.cdeter
        var cstoch = st.cstoch
        var toks = st.toks
        # encode tokens
        for t in range(Self.T):
            var ob = _alloc(Self.B * OBSD)
            for b in range(Self.B):
                for k in range(OBSD):
                    ob[b * OBSD + k] = obs[(b * (Self.T + 1) + t) * OBSD + k]
            var tk = toks + t * Self.B * TOK
            var tkt = TileTensor(tk, row_major[Self.B, TOK]())
            enc.forward[target, Self.B](
                TileTensor(ob, row_major[Self.B, OBSD]()), output=tkt
            )
            ob.free()
        for i in range(Self.B * D):
            cdeter[i] = 0.0
        for i in range(Self.B * SCl):
            cstoch[i] = 0.0
        var total: Scalar[DT] = 0.0
        var outbuf = _alloc(Self.B * CARRYl)
        var dl = _alloc(Self.B)
        for t in range(Self.T):
            var dtp = cdeter + t * Self.B * D
            var stp = cstoch + t * Self.B * SCl
            var at = _alloc(Self.B * ACTD)
            for b in range(Self.B):
                for k in range(ACTD):
                    at[b * ACTD + k] = act[(b * Self.T + t) * ACTD + k]
            core.set_input["deter", Self.B](TileTensor(dtp, row_major[Self.B, D]()))
            core.set_input["stoch", Self.B](TileTensor(stp, row_major[Self.B, SCl]()))
            core.set_input["action", Self.B](TileTensor(at, row_major[Self.B, ACTD]()))
            core.set_input["tokens", Self.B](TileTensor(toks + t * Self.B * TOK, row_major[Self.B, TOK]()))
            var ot = TileTensor(outbuf, row_major[Self.B, CARRYl]())
            core.forward[target, Self.B](ot)
            var ndn = cdeter + (t + 1) * Self.B * D
            var snn = cstoch + (t + 1) * Self.B * SCl
            for b in range(Self.B):
                for k in range(D):
                    ndn[b * D + k] = outbuf[b * CARRYl + 2 + k]
                for k in range(SCl):
                    snn[b * SCl + k] = outbuf[b * CARRYl + 2 + D + k]
                total += DYN * outbuf[b * CARRYl + 0] + REP * outbuf[b * CARRYl + 1]
            var rtg = _alloc(Self.B * OBSD)
            for b in range(Self.B):
                for k in range(OBSD):
                    rtg[b * OBSD + k] = obs[(b * (Self.T + 1) + t) * OBSD + k]
            var rwt = _alloc(Self.B)
            var cnt = _alloc(Self.B)
            for b in range(Self.B):
                rwt[b] = rew_t[b * Self.T + t]
                cnt[b] = Scalar[DT](1.0) - dne[b * Self.T + t]
            dec.set_input["stoch_new", Self.B](TileTensor(snn, row_major[Self.B, SCl]()))
            dec.set_input["nd", Self.B](TileTensor(ndn, row_major[Self.B, D]()))
            dec.set_input["rtgt", Self.B](TileTensor(rtg, row_major[Self.B, OBSD]()))
            var dlt = TileTensor(dl, row_major[Self.B, 1]())
            dec.forward[target, Self.B](dlt)
            for b in range(Self.B):
                total += dl[b]
            rew.set_input["nd", Self.B](TileTensor(ndn, row_major[Self.B, D]()))
            rew.set_input["stoch_new", Self.B](TileTensor(snn, row_major[Self.B, SCl]()))
            rew.set_input["rtgt", Self.B](TileTensor(rwt, row_major[Self.B, 1]()))
            var rlt = TileTensor(dl, row_major[Self.B, 1]())
            rew.forward[target, Self.B](rlt)
            for b in range(Self.B):
                total += dl[b]
            con.set_input["nd", Self.B](TileTensor(ndn, row_major[Self.B, D]()))
            con.set_input["stoch_new", Self.B](TileTensor(snn, row_major[Self.B, SCl]()))
            con.set_input["ctgt", Self.B](TileTensor(cnt, row_major[Self.B, 1]()))
            var clt = TileTensor(dl, row_major[Self.B, 1]())
            con.forward[target, Self.B](clt)
            for b in range(Self.B):
                total += dl[b]
            at.free(); rtg.free(); rwt.free(); cnt.free()

        oe.zero_grad[target, Self.EncT](enc)
        ocore.zero_grad_graph[target](core)
        odec.zero_grad_graph[target](dec)
        orew.zero_grad_graph[target](rew)
        ocon.zero_grad_graph[target](con)
        var gcd = _alloc(Self.B * D)
        var gcs = _alloc(Self.B * SCl)
        for i in range(Self.B * D):
            gcd[i] = 0.0
        for i in range(Self.B * SCl):
            gcs[i] = 0.0
        var ones1 = _alloc(Self.B)
        for b in range(Self.B):
            ones1[b] = 1.0
        var seed = _alloc(Self.B * CARRYl)
        var scratch = _alloc(Self.B * CARRYl)
        var dl1 = _alloc(Self.B)
        for rev in range(Self.T):
            var t = Self.T - 1 - rev
            var dtp = cdeter + t * Self.B * D
            var stp = cstoch + t * Self.B * SCl
            var ndn = cdeter + (t + 1) * Self.B * D
            var snn = cstoch + (t + 1) * Self.B * SCl
            var at = _alloc(Self.B * ACTD)
            for b in range(Self.B):
                for k in range(ACTD):
                    at[b * ACTD + k] = act[(b * Self.T + t) * ACTD + k]
            var rtg = _alloc(Self.B * OBSD)
            for b in range(Self.B):
                for k in range(OBSD):
                    rtg[b * OBSD + k] = obs[(b * (Self.T + 1) + t) * OBSD + k]
            var rwt = _alloc(Self.B)
            var cnt = _alloc(Self.B)
            for b in range(Self.B):
                rwt[b] = rew_t[b * Self.T + t]
                cnt[b] = Scalar[DT](1.0) - dne[b * Self.T + t]
            dec.set_input["stoch_new", Self.B](TileTensor(snn, row_major[Self.B, SCl]()))
            dec.set_input["nd", Self.B](TileTensor(ndn, row_major[Self.B, D]()))
            dec.set_input["rtgt", Self.B](TileTensor(rtg, row_major[Self.B, OBSD]()))
            var dlt = TileTensor(dl1, row_major[Self.B, 1]())
            dec.forward[target, Self.B](dlt)
            var ds = TileTensor(ones1, row_major[Self.B, 1]())
            dec.vjp[target, Self.B](ds)
            rew.set_input["nd", Self.B](TileTensor(ndn, row_major[Self.B, D]()))
            rew.set_input["stoch_new", Self.B](TileTensor(snn, row_major[Self.B, SCl]()))
            rew.set_input["rtgt", Self.B](TileTensor(rwt, row_major[Self.B, 1]()))
            var rlt = TileTensor(dl1, row_major[Self.B, 1]())
            rew.forward[target, Self.B](rlt)
            var rs = TileTensor(ones1, row_major[Self.B, 1]())
            rew.vjp[target, Self.B](rs)
            con.set_input["nd", Self.B](TileTensor(ndn, row_major[Self.B, D]()))
            con.set_input["stoch_new", Self.B](TileTensor(snn, row_major[Self.B, SCl]()))
            con.set_input["ctgt", Self.B](TileTensor(cnt, row_major[Self.B, 1]()))
            var clt = TileTensor(dl1, row_major[Self.B, 1]())
            con.forward[target, Self.B](clt)
            var cs = TileTensor(ones1, row_major[Self.B, 1]())
            con.vjp[target, Self.B](cs)
            var dnd = dec.grad_input_ptr["nd"]()
            var dsn = dec.grad_input_ptr["stoch_new"]()
            var rnd = rew.grad_input_ptr["nd"]()
            var rsn = rew.grad_input_ptr["stoch_new"]()
            var cnd = con.grad_input_ptr["nd"]()
            var csn = con.grad_input_ptr["stoch_new"]()
            for b in range(Self.B):
                seed[b * CARRYl + 0] = DYN
                seed[b * CARRYl + 1] = REP
                for k in range(D):
                    seed[b * CARRYl + 2 + k] = (
                        gcd[b * D + k] + dnd[b * D + k] + rnd[b * D + k]
                        + cnd[b * D + k]
                    )
                for k in range(SCl):
                    seed[b * CARRYl + 2 + D + k] = (
                        gcs[b * SCl + k] + dsn[b * SCl + k] + rsn[b * SCl + k]
                        + csn[b * SCl + k]
                    )
            core.set_input["deter", Self.B](TileTensor(dtp, row_major[Self.B, D]()))
            core.set_input["stoch", Self.B](TileTensor(stp, row_major[Self.B, SCl]()))
            core.set_input["action", Self.B](TileTensor(at, row_major[Self.B, ACTD]()))
            core.set_input["tokens", Self.B](TileTensor(toks + t * Self.B * TOK, row_major[Self.B, TOK]()))
            var sct = TileTensor(scratch, row_major[Self.B, CARRYl]())
            core.forward[target, Self.B](sct)
            var seedt = TileTensor(seed, row_major[Self.B, CARRYl]())
            core.vjp[target, Self.B](seedt)
            var gdt = core.grad_input_ptr["deter"]()
            var gst = core.grad_input_ptr["stoch"]()
            for i in range(Self.B * D):
                gcd[i] = gdt[i]
            for i in range(Self.B * SCl):
                gcs[i] = gst[i]
            var gtok = core.grad_input_ptr["tokens"]()
            var ob = _alloc(Self.B * OBSD)
            for b in range(Self.B):
                for k in range(OBSD):
                    ob[b * OBSD + k] = obs[(b * (Self.T + 1) + t) * OBSD + k]
            var tkscr = _alloc(Self.B * TOK)
            var tkt = TileTensor(tkscr, row_major[Self.B, TOK]())
            enc.forward[target, Self.B](TileTensor(ob, row_major[Self.B, OBSD]()), output=tkt)
            var gobs = _alloc(Self.B * OBSD)
            var gobst = TileTensor(gobs, row_major[Self.B, OBSD]())
            enc.vjp[target, Self.B](TileTensor(gtok, row_major[Self.B, TOK]()), gobst)
            at.free(); rtg.free(); rwt.free(); cnt.free(); ob.free()
            tkscr.free(); gobs.free()
        oe.step[target, Self.EncT](enc)
        ocore.step_graph[target](core)
        odec.step_graph[target](dec)
        orew.step_graph[target](rew)
        ocon.step_graph[target](con)
        outbuf.free(); dl.free(); gcd.free(); gcs.free(); ones1.free()
        seed.free(); scratch.free(); dl1.free()
        st.last_wm_loss = total


# ──────────────────────────────────────────────────────────────────────
# ParamSyncStep — copy core/prior params WMCoreGraph → WMImagineGraph.
# ──────────────────────────────────────────────────────────────────────


@fieldwise_init
struct ParamSyncStep[
    DETER: Int, H: Int, STOCH: Int, CLASSES: Int, BLOCKS: Int, ACT: Int,
    TOKEN: Int,
](Movable & ImplicitlyDestructible):
    comptime CoreT = WMCoreGraph[
        Self.DETER, Self.H, Self.STOCH, Self.CLASSES, Self.BLOCKS, Self.ACT,
        Self.TOKEN, SwishOp,
    ]
    comptime ImagT = WMImagineGraph[
        Self.DETER, Self.H, Self.STOCH, Self.CLASSES, Self.BLOCKS, Self.ACT, SwishOp,
    ]

    @staticmethod
    def make[target: StaticString](ctx: Optional[DeviceContext] = None) raises -> Self:
        return Self()

    def step[target: StaticString](
        mut self, mut core: Self.CoreT, mut imagine: Self.ImagT,
    ) raises:
        comptime assert target == "cpu", "ParamSyncStep: GPU = Step 5"
        var names = List[String]()
        var ptrs = List[UnsafePointer[Scalar[DT], MutAnyOrigin]]()
        var lens = List[Int]()
        collect_params[target](core, names, ptrs, lens)
        apply_params[target](imagine, names, ptrs, lens)
        _ = names^; _ = ptrs^; _ = lens^


# ──────────────────────────────────────────────────────────────────────
# ACStep — imagination rollout + actor-critic loss. Trains value/policy;
# Polyak-updates slowvalue. Reads the start carry from state.cdeter[T].
# ──────────────────────────────────────────────────────────────────────


@fieldwise_init
struct ACStep[
    OBS: Int, ACT: Int, DETER: Int, H: Int, STOCH: Int, CLASSES: Int,
    BLOCKS: Int, TOKEN: Int, HU: Int, VU: Int, PU: Int, BINS: Int,
    B: Int, T: Int, T_IMAG: Int,
](Movable & ImplicitlyDestructible):
    comptime SC = Self.STOCH * Self.CLASSES
    comptime FEAT = Self.DETER + Self.SC
    comptime ImagT = WMImagineGraph[
        Self.DETER, Self.H, Self.STOCH, Self.CLASSES, Self.BLOCKS, Self.ACT, SwishOp,
    ]
    comptime ValT = DreamerValue[Self.FEAT, Self.VU, Self.BINS, SwishOp]
    comptime PolT = DreamerPolicy[Self.FEAT, Self.PU, Self.ACT, SwishOp]
    comptime RewT = RewLossGraph[Self.DETER, Self.SC, Self.HU, Self.BINS, SwishOp]
    comptime ConT = ConLossGraph[Self.DETER, Self.SC, Self.HU, SwishOp]
    var minstd: Scalar[DT]
    var maxstd: Scalar[DT]
    var lam: Scalar[DT]
    var actent: Scalar[DT]
    var slowreg: Scalar[DT]
    var slow_rate: Scalar[DT]

    @staticmethod
    def make[target: StaticString](ctx: Optional[DeviceContext] = None) raises -> Self:
        return Self(
            minstd=Scalar[DT](0.1), maxstd=Scalar[DT](1.0), lam=Scalar[DT](0.95),
            actent=Scalar[DT](3e-4), slowreg=Scalar[DT](1.0),
            slow_rate=Scalar[DT](0.02),
        )

    def step[target: StaticString](
        mut self,
        mut st: DreamerState[Self.OBS, Self.ACT, Self.DETER, Self.SC, Self.TOKEN, Self.B, Self.T, Self.T_IMAG],
        mut imagine: Self.ImagT,
        mut value: Self.ValT,
        mut slowvalue: Self.ValT,
        mut policy: Self.PolT,
        mut rew: Self.RewT,
        mut con: Self.ConT,
        mut oval: DreamerOpt,
        mut opol: DreamerOpt,
        mut retnorm: PercentileNormalize,
        bins: UnsafePointer[Scalar[DT], MutAnyOrigin],
    ) raises:
        comptime assert target == "cpu", "ACStep: GPU branch lands in Step 5"
        comptime D = Self.DETER
        comptime SCl = Self.SC
        comptime FEATl = Self.FEAT
        comptime ACTD = Self.ACT
        comptime BINSl = Self.BINS
        comptime TI = Self.T_IMAG
        var MINSTD = self.minstd
        var MAXSTD = self.maxstd
        var deter0 = st.cdeter + Self.T * Self.B * D
        var stoch0 = st.cstoch + Self.T * Self.B * SCl
        var noise = st.noise
        var feats = _alloc(Self.B * TI * FEATl)
        var acts = _alloc(Self.B * TI * ACTD)
        var pmean = _alloc(Self.B * TI * ACTD)
        var pstd = _alloc(Self.B * TI * ACTD)
        var vlog = _alloc(Self.B * TI * BINSl)
        var svlog = _alloc(Self.B * TI * BINSl)
        var rewv = _alloc(Self.B * TI)
        var conv = _alloc(Self.B * TI)
        var cd = _alloc(Self.B * D)
        var cs = _alloc(Self.B * SCl)
        for i in range(Self.B * D):
            cd[i] = deter0[i]
        for i in range(Self.B * SCl):
            cs[i] = stoch0[i]
        var fb = _alloc(Self.B * FEATl)
        var pb = _alloc(Self.B * 2 * ACTD)
        var vb = _alloc(Self.B * BINSl)
        var svb = _alloc(Self.B * BINSl)
        var dummy1 = _alloc(Self.B * 1)
        for t in range(TI):
            for b in range(Self.B):
                for k in range(D):
                    fb[b * FEATl + k] = cd[b * D + k]
                for k in range(SCl):
                    fb[b * FEATl + D + k] = cs[b * SCl + k]
                for k in range(FEATl):
                    feats[(b * TI + t) * FEATl + k] = fb[b * FEATl + k]
            var ft = TileTensor(fb, row_major[Self.B, FEATl]())
            var pt = TileTensor(pb, row_major[Self.B, 2 * ACTD]())
            policy.forward[target, Self.B](ft, output=pt)
            for b in range(Self.B):
                for a in range(ACTD):
                    var mr = pb[b * 2 * ACTD + a]
                    var sr = pb[b * 2 * ACTD + ACTD + a]
                    pmean[(b * TI + t) * ACTD + a] = mr
                    pstd[(b * TI + t) * ACTD + a] = sr
                    var z = noise[(t * Self.B + b) * ACTD + a]
                    acts[(b * TI + t) * ACTD + a] = (
                        tanh(mr) + bounded_std(sr, MINSTD, MAXSTD) * z
                    )
            var ft2 = TileTensor(fb, row_major[Self.B, FEATl]())
            var vt = TileTensor(vb, row_major[Self.B, BINSl]())
            value.forward[target, Self.B](ft2, output=vt)
            var ft3 = TileTensor(fb, row_major[Self.B, FEATl]())
            var svt = TileTensor(svb, row_major[Self.B, BINSl]())
            slowvalue.forward[target, Self.B](ft3, output=svt)
            for b in range(Self.B):
                for c in range(BINSl):
                    vlog[(b * TI + t) * BINSl + c] = vb[b * BINSl + c]
                    svlog[(b * TI + t) * BINSl + c] = svb[b * BINSl + c]
            rew.set_input["nd", Self.B](TileTensor(cd, row_major[Self.B, D]()))
            rew.set_input["stoch_new", Self.B](TileTensor(cs, row_major[Self.B, SCl]()))
            rew.set_input["rtgt", Self.B](TileTensor(dummy1, row_major[Self.B, 1]()))
            var rlt = TileTensor(dummy1, row_major[Self.B, 1]())
            rew.forward[target, Self.B](rlt)
            var rew_logits = rew.node_out_ptr["rew"]()
            con.set_input["nd", Self.B](TileTensor(cd, row_major[Self.B, D]()))
            con.set_input["stoch_new", Self.B](TileTensor(cs, row_major[Self.B, SCl]()))
            con.set_input["ctgt", Self.B](TileTensor(dummy1, row_major[Self.B, 1]()))
            var clt = TileTensor(dummy1, row_major[Self.B, 1]())
            con.forward[target, Self.B](clt)
            var con_logit = con.node_out_ptr["con"]()
            for b in range(Self.B):
                rewv[b * TI + t] = twohot_pred[BINSl](rew_logits, b * BINSl, bins)
                conv[b * TI + t] = Scalar[DT](1.0) / (
                    Scalar[DT](1.0) + exp(-con_logit[b])
                )
            var at = _alloc(Self.B * ACTD)
            for b in range(Self.B):
                for a in range(ACTD):
                    at[b * ACTD + a] = acts[(b * TI + t) * ACTD + a]
            imagine.set_input["deter", Self.B](TileTensor(cd, row_major[Self.B, D]()))
            imagine.set_input["stoch", Self.B](TileTensor(cs, row_major[Self.B, SCl]()))
            imagine.set_input["action", Self.B](TileTensor(at, row_major[Self.B, ACTD]()))
            var fo = TileTensor(fb, row_major[Self.B, FEATl]())
            imagine.forward[target, Self.B](fo)
            var nd = imagine.node_out_ptr["nd"]()
            var sn = imagine.node_out_ptr["stoch_new"]()
            for i in range(Self.B * D):
                cd[i] = nd[i]
            for i in range(Self.B * SCl):
                cs[i] = sn[i]
            at.free()

        comptime TM1 = TI - 1
        var pol_loss = _alloc(Self.B * TM1)
        var val_loss = _alloc(Self.B * TM1)
        var ret = _alloc(Self.B * TM1)
        imag_loss_cpu[Self.B, TI, ACTD, BINSl](
            acts, rewv, conv, vlog, svlog, pmean, pstd, bins,
            MINSTD, MAXSTD, self.lam, self.actent, self.slowreg,
            retnorm, pol_loss, val_loss, ret,
        )
        var total: Scalar[DT] = 0.0
        for i in range(Self.B * TM1):
            total += pol_loss[i] + val_loss[i]
        var rscale = retnorm.stats()[1]
        var d_pol = _alloc(Self.B * TM1)
        var d_val = _alloc(Self.B * TM1)
        for i in range(Self.B * TM1):
            d_pol[i] = 1.0
            d_val[i] = 1.0
        var g_vlog = _alloc(Self.B * TI * BINSl)
        var g_pmean = _alloc(Self.B * TI * ACTD)
        var g_pstd = _alloc(Self.B * TI * ACTD)
        imag_loss_backward[Self.B, TI, ACTD, BINSl](
            acts, rewv, conv, vlog, svlog, pmean, pstd, bins,
            MINSTD, MAXSTD, self.lam, self.actent, self.slowreg,
            rscale, d_pol, d_val, g_vlog, g_pmean, g_pstd,
        )
        oval.zero_grad[target, Self.ValT](value)
        opol.zero_grad[target, Self.PolT](policy)
        var gfeat = _alloc(Self.B * FEATl)
        var vscr = _alloc(Self.B * BINSl)
        var pscr = _alloc(Self.B * 2 * ACTD)
        var polg = _alloc(Self.B * 2 * ACTD)
        for t in range(TI):
            var ftt = _alloc(Self.B * FEATl)
            for b in range(Self.B):
                for k in range(FEATl):
                    ftt[b * FEATl + k] = feats[(b * TI + t) * FEATl + k]
            var fvt = TileTensor(ftt, row_major[Self.B, FEATl]())
            var vot = TileTensor(vscr, row_major[Self.B, BINSl]())
            value.forward[target, Self.B](fvt, output=vot)
            var gv = _alloc(Self.B * BINSl)
            for b in range(Self.B):
                for c in range(BINSl):
                    gv[b * BINSl + c] = g_vlog[(b * TI + t) * BINSl + c]
            var gvt = TileTensor(gv, row_major[Self.B, BINSl]())
            var gft = TileTensor(gfeat, row_major[Self.B, FEATl]())
            value.vjp[target, Self.B](gvt, gft)
            var fpt = TileTensor(ftt, row_major[Self.B, FEATl]())
            var pot = TileTensor(pscr, row_major[Self.B, 2 * ACTD]())
            policy.forward[target, Self.B](fpt, output=pot)
            for b in range(Self.B):
                for a in range(ACTD):
                    polg[b * 2 * ACTD + a] = g_pmean[(b * TI + t) * ACTD + a]
                    polg[b * 2 * ACTD + ACTD + a] = g_pstd[(b * TI + t) * ACTD + a]
            var pgt = TileTensor(polg, row_major[Self.B, 2 * ACTD]())
            var gft2 = TileTensor(gfeat, row_major[Self.B, FEATl]())
            policy.vjp[target, Self.B](pgt, gft2)
            gv.free(); ftt.free()
        oval.step[target, Self.ValT](value)
        opol.step[target, Self.PolT](policy)
        polyak_module[target, Self.ValT](value, slowvalue, self.slow_rate)

        feats.free(); acts.free(); pmean.free(); pstd.free(); vlog.free()
        svlog.free(); rewv.free(); conv.free(); cd.free(); cs.free()
        fb.free(); pb.free(); vb.free(); svb.free(); dummy1.free()
        pol_loss.free(); val_loss.free(); ret.free()
        d_pol.free(); d_val.free(); g_vlog.free(); g_pmean.free()
        g_pstd.free(); gfeat.free(); vscr.free(); pscr.free(); polg.free()
        st.last_ac_loss = total
