"""DreamerV3Trainer — assembles the validated PR5c spikes into a trainer.

CPU v1. Composes:
  * encoder (`DreamerEncoder` Sequential)
  * `WMCoreGraph` (RSSM dyn/rep + carry passthrough)  — WM training substrate
  * `DecLossGraph` / `RewLossGraph` / `ConLossGraph`   — head losses (recon/rew/con)
  * `DreamerValue` (value + Polyak slowvalue) + `DreamerPolicy`
  * `WMImagineGraph` (param-synced mirror of core/prior) — imagination rollout
  * one `DreamerOpt` per trainable module (via the `*_graph` overloads)
  * `SequenceReplay[OBS, ACT, CAP]` + `PercentileNormalize` retnorm

`train_step` = WM-BPTT over a sampled length-T window (carry threaded through
the passthrough columns; recompute-in-backward) → sync core/prior → imagination
rollout from the final posterior carry → AC loss (`imag_loss`) → value/policy
step → Polyak slowvalue. Every mechanism here is validated standalone:
`spike_wm_bptt` (WM↓), `spike_wm_imag_ac` (AC↓), `spike_param_sync` (0.0).

v1 simplifications (documented; lighthouse/tuning is Step 7):
  - data collection uses uniform-random actions (no running-carry policy
    action yet); the WM still learns the env, the AC learns a policy in
    imagination. Proper carry-tracked action selection is a follow-up.
  - imagination starts from the WM scan's final posterior carry (B traj),
    not every state in the batch.
  - reward/cont predictions for imagined states come from the WM-trained
    Rew/ConLossGraph heads (forward with a dummy target, read the logits).
  - act = GELU (Step 5 swaps to SiLU for reference-matching convergence).
"""

from std.memory import alloc
from std.math import tanh, exp
from std.random import random_float64
from std.gpu.memory import AddressSpace
from layout import TileTensor, row_major

from mojo_rl.nn2.constants import DT
from mojo_rl.nn2.initializer import Kaiming
from mojo_rl.nn2.core import ParamVisitor
from mojo_rl.nn2.core.module import Module
from mojo_rl.nn2.optimizer.dreamer_opt import DreamerOpt
from mojo_rl.deep_agents2.data.sequence_replay import SequenceReplay
from mojo_rl.deep_agents2.dreamerv3.wm import (
    WMCoreGraph, WMImagineGraph, DecLossGraph, RewLossGraph, ConLossGraph,
)
from mojo_rl.deep_agents2.dreamerv3.nets import (
    DreamerEncoder, DreamerValue, DreamerPolicy,
)
from mojo_rl.deep_agents2.dreamerv3.twohot import twohot_pred, symexp_twohot_bins
from mojo_rl.deep_agents2.dreamerv3.dists import bounded_std
from mojo_rl.deep_agents2.dreamerv3.normalize import PercentileNormalize
from mojo_rl.deep_agents2.dreamerv3.imag_loss import (
    imag_loss_cpu, imag_loss_backward,
)
from mojo_rl.deep_agents2.dreamerv3.param_sync import (
    collect_params, apply_params,
)


@always_inline
def _alloc(n: Int) -> UnsafePointer[Scalar[DT], MutAnyOrigin]:
    return alloc[Scalar[DT]](n)


@fieldwise_init
struct DreamerV3Trainer[
    OBS: Int, ACT: Int, DETER: Int, H: Int, STOCH: Int, CLASSES: Int,
    BLOCKS: Int, TOKEN: Int, DEC_U: Int, HU: Int, VU: Int, PU: Int,
    BINS: Int, B: Int, T: Int, T_IMAG: Int, CAP: Int,
](Movable & ImplicitlyDestructible):
    comptime SC = Self.STOCH * Self.CLASSES
    comptime FEAT = Self.DETER + Self.SC
    comptime CARRY = 2 + Self.DETER + Self.SC

    comptime EncT = DreamerEncoder[Self.OBS, Self.TOKEN]
    comptime CoreT = WMCoreGraph[
        Self.DETER, Self.H, Self.STOCH, Self.CLASSES, Self.BLOCKS, Self.ACT,
        Self.TOKEN,
    ]
    comptime DecT = DecLossGraph[Self.SC, Self.DETER, Self.OBS, Self.DEC_U]
    comptime RewT = RewLossGraph[Self.DETER, Self.SC, Self.HU, Self.BINS]
    comptime ConT = ConLossGraph[Self.DETER, Self.SC, Self.HU]
    comptime ValT = DreamerValue[Self.FEAT, Self.VU, Self.BINS]
    comptime PolT = DreamerPolicy[Self.FEAT, Self.PU, Self.ACT]
    comptime ImagT = WMImagineGraph[
        Self.DETER, Self.H, Self.STOCH, Self.CLASSES, Self.BLOCKS, Self.ACT,
    ]
    comptime RepT = SequenceReplay[Self.OBS, Self.ACT, Self.CAP]

    var enc: Self.EncT
    var core: Self.CoreT
    var dec: Self.DecT
    var rew: Self.RewT
    var con: Self.ConT
    var value: Self.ValT
    var slowvalue: Self.ValT
    var policy: Self.PolT
    var imagine: Self.ImagT

    var oe: DreamerOpt
    var ocore: DreamerOpt
    var odec: DreamerOpt
    var orew: DreamerOpt
    var ocon: DreamerOpt
    var oval: DreamerOpt
    var opol: DreamerOpt

    var replay: Self.RepT
    var retnorm: PercentileNormalize
    var bins: List[Scalar[DT]]
    var slow_rate: Scalar[DT]
    var learning_starts: Int
    var train_steps: Int
    var last_wm_loss: Scalar[DT]
    var last_ac_loss: Scalar[DT]

    @staticmethod
    def make(
        lr: Scalar[DT] = Scalar[DT](4e-5),
        learning_starts: Int = 200,
    ) raises -> Self:
        var enc = Self.EncT.make["cpu", INIT=Kaiming]()
        var core = Self.CoreT.make["cpu", INIT=Kaiming]()
        var dec = Self.DecT.make["cpu", INIT=Kaiming]()
        var rew = Self.RewT.make["cpu", INIT=Kaiming]()
        var con = Self.ConT.make["cpu", INIT=Kaiming]()
        var value = Self.ValT.make["cpu", INIT=Kaiming]()
        var slowvalue = Self.ValT.make["cpu", INIT=Kaiming]()
        var policy = Self.PolT.make["cpu", INIT=Kaiming]()
        var imagine = Self.ImagT.make["cpu", INIT=Kaiming]()

        var oe = DreamerOpt.make["cpu", Self.EncT](enc)
        var ocore = DreamerOpt.make_graph["cpu"](core)
        var odec = DreamerOpt.make_graph["cpu"](dec)
        var orew = DreamerOpt.make_graph["cpu"](rew)
        var ocon = DreamerOpt.make_graph["cpu"](con)
        var oval = DreamerOpt.make["cpu", Self.ValT](value)
        var opol = DreamerOpt.make["cpu", Self.PolT](policy)
        oe.lr = lr; ocore.lr = lr; odec.lr = lr; orew.lr = lr
        ocon.lr = lr; oval.lr = lr; opol.lr = lr

        var bins = List[Scalar[DT]](length=Self.BINS, fill=Scalar[DT](0.0))
        symexp_twohot_bins[Self.BINS](
            rebind[UnsafePointer[Scalar[DT], MutAnyOrigin]](bins.unsafe_ptr())
        )
        var retnorm = PercentileNormalize.make(
            String("perc"), Scalar[DT](0.01), Scalar[DT](5.0),
            Scalar[DT](95.0), Scalar[DT](1.0), False,
        )

        return Self(
            enc=enc^, core=core^, dec=dec^, rew=rew^, con=con^, value=value^,
            slowvalue=slowvalue^, policy=policy^, imagine=imagine^,
            oe=oe^, ocore=ocore^, odec=odec^, orew=orew^, ocon=ocon^,
            oval=oval^, opol=opol^,
            replay=Self.RepT.new(), retnorm=retnorm^, bins=bins^,
            slow_rate=Scalar[DT](0.02), learning_starts=learning_starts,
            train_steps=0, last_wm_loss=0.0, last_ac_loss=0.0,
        )

    def record(
        mut self,
        obs: UnsafePointer[Scalar[DT], MutAnyOrigin],
        act: UnsafePointer[Scalar[DT], MutAnyOrigin],
        reward: Scalar[DT],
        done: Scalar[DT],
    ):
        self.replay.record(obs, act, reward, done)

    def can_train(self) -> Bool:
        return self.replay.size >= Self.T + 1

    # ── WM-BPTT over one sampled window; fills final carry; returns WM loss
    def _wm_step(
        mut self,
        obs: UnsafePointer[Scalar[DT], MutAnyOrigin],   # [B,T+1,OBS]
        act: UnsafePointer[Scalar[DT], MutAnyOrigin],   # [B,T,ACT]
        rew: UnsafePointer[Scalar[DT], MutAnyOrigin],   # [B,T]
        dne: UnsafePointer[Scalar[DT], MutAnyOrigin],   # [B,T]
        cdeter: UnsafePointer[Scalar[DT], MutAnyOrigin],  # [(T+1)*B*DETER]
        cstoch: UnsafePointer[Scalar[DT], MutAnyOrigin],  # [(T+1)*B*SCv]
        toks: UnsafePointer[Scalar[DT], MutAnyOrigin],    # [T*B*TOKEN]
    ) raises -> Scalar[DT]:
        comptime D = Self.DETER
        comptime SCv = Self.SC
        comptime TOK = Self.TOKEN
        comptime CARRYv = Self.CARRY
        comptime OBSD = Self.OBS
        comptime ACTD = Self.ACT
        comptime DYN = Scalar[DT](1.0)
        comptime REP = Scalar[DT](0.1)
        # encode tokens
        for t in range(Self.T):
            var ob = _alloc(Self.B * OBSD)
            for b in range(Self.B):
                for k in range(OBSD):
                    ob[b * OBSD + k] = obs[(b * (Self.T + 1) + t) * OBSD + k]
            var tk = toks + t * Self.B * TOK
            var tkt = TileTensor(tk, row_major[Self.B, TOK]())
            self.enc.forward["cpu", Self.B](
                TileTensor(ob, row_major[Self.B, OBSD]()), output=tkt
            )
            ob.free()
        # forward scan
        for i in range(Self.B * D):
            cdeter[i] = 0.0
        for i in range(Self.B * SCv):
            cstoch[i] = 0.0
        var total: Scalar[DT] = 0.0
        var outbuf = _alloc(Self.B * CARRYv)
        var dl = _alloc(Self.B)
        for t in range(Self.T):
            var dtp = cdeter + t * Self.B * D
            var stp = cstoch + t * Self.B * SCv
            var at = _alloc(Self.B * ACTD)
            for b in range(Self.B):
                for k in range(ACTD):
                    at[b * ACTD + k] = act[(b * Self.T + t) * ACTD + k]
            self.core.set_input["deter", Self.B](TileTensor(dtp, row_major[Self.B, D]()))
            self.core.set_input["stoch", Self.B](TileTensor(stp, row_major[Self.B, SCv]()))
            self.core.set_input["action", Self.B](TileTensor(at, row_major[Self.B, ACTD]()))
            self.core.set_input["tokens", Self.B](TileTensor(toks + t * Self.B * TOK, row_major[Self.B, TOK]()))
            var ot = TileTensor(outbuf, row_major[Self.B, CARRYv]())
            self.core.forward["cpu", Self.B](ot)
            var ndn = cdeter + (t + 1) * Self.B * D
            var snn = cstoch + (t + 1) * Self.B * SCv
            for b in range(Self.B):
                for k in range(D):
                    ndn[b * D + k] = outbuf[b * CARRYv + 2 + k]
                for k in range(SCv):
                    snn[b * SCv + k] = outbuf[b * CARRYv + 2 + D + k]
                total += DYN * outbuf[b * CARRYv + 0] + REP * outbuf[b * CARRYv + 1]
            # head forwards
            var rtg = _alloc(Self.B * OBSD)
            for b in range(Self.B):
                for k in range(OBSD):
                    rtg[b * OBSD + k] = obs[(b * (Self.T + 1) + t) * OBSD + k]
            var rwt = _alloc(Self.B)
            var cnt = _alloc(Self.B)
            for b in range(Self.B):
                rwt[b] = rew[b * Self.T + t]
                cnt[b] = Scalar[DT](1.0) - dne[b * Self.T + t]
            self.dec.set_input["stoch_new", Self.B](TileTensor(snn, row_major[Self.B, SCv]()))
            self.dec.set_input["nd", Self.B](TileTensor(ndn, row_major[Self.B, D]()))
            self.dec.set_input["rtgt", Self.B](TileTensor(rtg, row_major[Self.B, OBSD]()))
            var dlt = TileTensor(dl, row_major[Self.B, 1]())
            self.dec.forward["cpu", Self.B](dlt)
            for b in range(Self.B):
                total += dl[b]
            self.rew.set_input["nd", Self.B](TileTensor(ndn, row_major[Self.B, D]()))
            self.rew.set_input["stoch_new", Self.B](TileTensor(snn, row_major[Self.B, SCv]()))
            self.rew.set_input["rtgt", Self.B](TileTensor(rwt, row_major[Self.B, 1]()))
            var rlt = TileTensor(dl, row_major[Self.B, 1]())
            self.rew.forward["cpu", Self.B](rlt)
            for b in range(Self.B):
                total += dl[b]
            self.con.set_input["nd", Self.B](TileTensor(ndn, row_major[Self.B, D]()))
            self.con.set_input["stoch_new", Self.B](TileTensor(snn, row_major[Self.B, SCv]()))
            self.con.set_input["ctgt", Self.B](TileTensor(cnt, row_major[Self.B, 1]()))
            var clt = TileTensor(dl, row_major[Self.B, 1]())
            self.con.forward["cpu", Self.B](clt)
            for b in range(Self.B):
                total += dl[b]
            at.free(); rtg.free(); rwt.free(); cnt.free()

        # backward scan
        self.oe.zero_grad["cpu", Self.EncT](self.enc)
        self.ocore.zero_grad_graph["cpu"](self.core)
        self.odec.zero_grad_graph["cpu"](self.dec)
        self.orew.zero_grad_graph["cpu"](self.rew)
        self.ocon.zero_grad_graph["cpu"](self.con)
        var gcd = _alloc(Self.B * D)
        var gcs = _alloc(Self.B * SCv)
        for i in range(Self.B * D):
            gcd[i] = 0.0
        for i in range(Self.B * SCv):
            gcs[i] = 0.0
        var ones1 = _alloc(Self.B)
        for b in range(Self.B):
            ones1[b] = 1.0
        var seed = _alloc(Self.B * CARRYv)
        var scratch = _alloc(Self.B * CARRYv)
        var dl1 = _alloc(Self.B)
        for rev in range(Self.T):
            var t = Self.T - 1 - rev
            var dtp = cdeter + t * Self.B * D
            var stp = cstoch + t * Self.B * SCv
            var ndn = cdeter + (t + 1) * Self.B * D
            var snn = cstoch + (t + 1) * Self.B * SCv
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
                rwt[b] = rew[b * Self.T + t]
                cnt[b] = Scalar[DT](1.0) - dne[b * Self.T + t]
            # head backward (recompute + vjp ones)
            self.dec.set_input["stoch_new", Self.B](TileTensor(snn, row_major[Self.B, SCv]()))
            self.dec.set_input["nd", Self.B](TileTensor(ndn, row_major[Self.B, D]()))
            self.dec.set_input["rtgt", Self.B](TileTensor(rtg, row_major[Self.B, OBSD]()))
            var dlt = TileTensor(dl1, row_major[Self.B, 1]())
            self.dec.forward["cpu", Self.B](dlt)
            var ds = TileTensor(ones1, row_major[Self.B, 1]())
            self.dec.vjp["cpu", Self.B](ds)
            self.rew.set_input["nd", Self.B](TileTensor(ndn, row_major[Self.B, D]()))
            self.rew.set_input["stoch_new", Self.B](TileTensor(snn, row_major[Self.B, SCv]()))
            self.rew.set_input["rtgt", Self.B](TileTensor(rwt, row_major[Self.B, 1]()))
            var rlt = TileTensor(dl1, row_major[Self.B, 1]())
            self.rew.forward["cpu", Self.B](rlt)
            var rs = TileTensor(ones1, row_major[Self.B, 1]())
            self.rew.vjp["cpu", Self.B](rs)
            self.con.set_input["nd", Self.B](TileTensor(ndn, row_major[Self.B, D]()))
            self.con.set_input["stoch_new", Self.B](TileTensor(snn, row_major[Self.B, SCv]()))
            self.con.set_input["ctgt", Self.B](TileTensor(cnt, row_major[Self.B, 1]()))
            var clt = TileTensor(dl1, row_major[Self.B, 1]())
            self.con.forward["cpu", Self.B](clt)
            var cs = TileTensor(ones1, row_major[Self.B, 1]())
            self.con.vjp["cpu", Self.B](cs)
            var dnd = self.dec.grad_input_ptr["nd"]()
            var dsn = self.dec.grad_input_ptr["stoch_new"]()
            var rnd = self.rew.grad_input_ptr["nd"]()
            var rsn = self.rew.grad_input_ptr["stoch_new"]()
            var cnd = self.con.grad_input_ptr["nd"]()
            var csn = self.con.grad_input_ptr["stoch_new"]()
            for b in range(Self.B):
                seed[b * CARRYv + 0] = DYN
                seed[b * CARRYv + 1] = REP
                for k in range(D):
                    seed[b * CARRYv + 2 + k] = (
                        gcd[b * D + k] + dnd[b * D + k] + rnd[b * D + k]
                        + cnd[b * D + k]
                    )
                for k in range(SCv):
                    seed[b * CARRYv + 2 + D + k] = (
                        gcs[b * SCv + k] + dsn[b * SCv + k] + rsn[b * SCv + k]
                        + csn[b * SCv + k]
                    )
            self.core.set_input["deter", Self.B](TileTensor(dtp, row_major[Self.B, D]()))
            self.core.set_input["stoch", Self.B](TileTensor(stp, row_major[Self.B, SCv]()))
            self.core.set_input["action", Self.B](TileTensor(at, row_major[Self.B, ACTD]()))
            self.core.set_input["tokens", Self.B](TileTensor(toks + t * Self.B * TOK, row_major[Self.B, TOK]()))
            var sct = TileTensor(scratch, row_major[Self.B, CARRYv]())
            self.core.forward["cpu", Self.B](sct)
            var seedt = TileTensor(seed, row_major[Self.B, CARRYv]())
            self.core.vjp["cpu", Self.B](seedt)
            var gdt = self.core.grad_input_ptr["deter"]()
            var gst = self.core.grad_input_ptr["stoch"]()
            for i in range(Self.B * D):
                gcd[i] = gdt[i]
            for i in range(Self.B * SCv):
                gcs[i] = gst[i]
            var gtok = self.core.grad_input_ptr["tokens"]()
            var ob = _alloc(Self.B * OBSD)
            for b in range(Self.B):
                for k in range(OBSD):
                    ob[b * OBSD + k] = obs[(b * (Self.T + 1) + t) * OBSD + k]
            var tkscr = _alloc(Self.B * TOK)
            var tkt = TileTensor(tkscr, row_major[Self.B, TOK]())
            self.enc.forward["cpu", Self.B](TileTensor(ob, row_major[Self.B, OBSD]()), output=tkt)
            var gobs = _alloc(Self.B * OBSD)
            var gobst = TileTensor(gobs, row_major[Self.B, OBSD]())
            self.enc.vjp["cpu", Self.B](TileTensor(gtok, row_major[Self.B, TOK]()), gobst)
            at.free(); rtg.free(); rwt.free(); cnt.free(); ob.free()
            tkscr.free(); gobs.free()
        self.oe.step["cpu", Self.EncT](self.enc)
        self.ocore.step_graph["cpu"](self.core)
        self.odec.step_graph["cpu"](self.dec)
        self.orew.step_graph["cpu"](self.rew)
        self.ocon.step_graph["cpu"](self.con)
        outbuf.free(); dl.free(); gcd.free(); gcs.free(); ones1.free()
        seed.free(); scratch.free(); dl1.free()
        return total

    # ── imagination rollout + AC loss; returns AC loss ──────────────────
    def _ac_step(
        mut self,
        deter0: UnsafePointer[Scalar[DT], MutAnyOrigin],  # [B,DETER]
        stoch0: UnsafePointer[Scalar[DT], MutAnyOrigin],  # [B,SCv]
        noise: UnsafePointer[Scalar[DT], MutAnyOrigin],   # [T_IMAG*B*ACT]
    ) raises -> Scalar[DT]:
        comptime D = Self.DETER
        comptime SCv = Self.SC
        comptime FEATv = Self.FEAT
        comptime ACTD = Self.ACT
        comptime BINSv = Self.BINS
        comptime TI = Self.T_IMAG
        comptime MINSTD = Scalar[DT](0.1)
        comptime MAXSTD = Scalar[DT](1.0)
        var bins = rebind[UnsafePointer[Scalar[DT], MutAnyOrigin]](
            self.bins.unsafe_ptr()
        )
        var feats = _alloc(Self.B * TI * FEATv)
        var acts = _alloc(Self.B * TI * ACTD)
        var pmean = _alloc(Self.B * TI * ACTD)
        var pstd = _alloc(Self.B * TI * ACTD)
        var vlog = _alloc(Self.B * TI * BINSv)
        var svlog = _alloc(Self.B * TI * BINSv)
        var rewv = _alloc(Self.B * TI)
        var conv = _alloc(Self.B * TI)
        var cd = _alloc(Self.B * D)
        var cs = _alloc(Self.B * SCv)
        for i in range(Self.B * D):
            cd[i] = deter0[i]
        for i in range(Self.B * SCv):
            cs[i] = stoch0[i]
        var fb = _alloc(Self.B * FEATv)
        var pb = _alloc(Self.B * 2 * ACTD)
        var vb = _alloc(Self.B * BINSv)
        var svb = _alloc(Self.B * BINSv)
        var rb = _alloc(Self.B * BINSv)
        var cb = _alloc(Self.B * 1)
        var dummy1 = _alloc(Self.B * 1)
        for t in range(TI):
            for b in range(Self.B):
                for k in range(D):
                    fb[b * FEATv + k] = cd[b * D + k]
                for k in range(SCv):
                    fb[b * FEATv + D + k] = cs[b * SCv + k]
                for k in range(FEATv):
                    feats[(b * TI + t) * FEATv + k] = fb[b * FEATv + k]
            var ft = TileTensor(fb, row_major[Self.B, FEATv]())
            var pt = TileTensor(pb, row_major[Self.B, 2 * ACTD]())
            self.policy.forward["cpu", Self.B](ft, output=pt)
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
            var ft2 = TileTensor(fb, row_major[Self.B, FEATv]())
            var vt = TileTensor(vb, row_major[Self.B, BINSv]())
            self.value.forward["cpu", Self.B](ft2, output=vt)
            var ft3 = TileTensor(fb, row_major[Self.B, FEATv]())
            var svt = TileTensor(svb, row_major[Self.B, BINSv]())
            self.slowvalue.forward["cpu", Self.B](ft3, output=svt)
            for b in range(Self.B):
                for c in range(BINSv):
                    vlog[(b * TI + t) * BINSv + c] = vb[b * BINSv + c]
                    svlog[(b * TI + t) * BINSv + c] = svb[b * BINSv + c]
            # reward/cont preds from WM-trained heads (read logits node)
            self.rew.set_input["nd", Self.B](TileTensor(cd, row_major[Self.B, D]()))
            self.rew.set_input["stoch_new", Self.B](TileTensor(cs, row_major[Self.B, SCv]()))
            self.rew.set_input["rtgt", Self.B](TileTensor(dummy1, row_major[Self.B, 1]()))
            var rlt = TileTensor(dummy1, row_major[Self.B, 1]())
            self.rew.forward["cpu", Self.B](rlt)
            var rew_logits = self.rew.node_out_ptr["rew"]()
            self.con.set_input["nd", Self.B](TileTensor(cd, row_major[Self.B, D]()))
            self.con.set_input["stoch_new", Self.B](TileTensor(cs, row_major[Self.B, SCv]()))
            self.con.set_input["ctgt", Self.B](TileTensor(dummy1, row_major[Self.B, 1]()))
            var clt = TileTensor(dummy1, row_major[Self.B, 1]())
            self.con.forward["cpu", Self.B](clt)
            var con_logit = self.con.node_out_ptr["con"]()
            for b in range(Self.B):
                rewv[b * TI + t] = twohot_pred[BINSv](rew_logits, b * BINSv, bins)
                conv[b * TI + t] = Scalar[DT](1.0) / (
                    Scalar[DT](1.0) + exp(-con_logit[b])
                )
            # imagine_step → carry_{t+1}
            var at = _alloc(Self.B * ACTD)
            for b in range(Self.B):
                for a in range(ACTD):
                    at[b * ACTD + a] = acts[(b * TI + t) * ACTD + a]
            self.imagine.set_input["deter", Self.B](TileTensor(cd, row_major[Self.B, D]()))
            self.imagine.set_input["stoch", Self.B](TileTensor(cs, row_major[Self.B, SCv]()))
            self.imagine.set_input["action", Self.B](TileTensor(at, row_major[Self.B, ACTD]()))
            var fo = TileTensor(fb, row_major[Self.B, FEATv]())
            self.imagine.forward["cpu", Self.B](fo)
            var nd = self.imagine.node_out_ptr["nd"]()
            var sn = self.imagine.node_out_ptr["stoch_new"]()
            for i in range(Self.B * D):
                cd[i] = nd[i]
            for i in range(Self.B * SCv):
                cs[i] = sn[i]
            at.free()

        comptime TM1 = TI - 1
        var pol_loss = _alloc(Self.B * TM1)
        var val_loss = _alloc(Self.B * TM1)
        var ret = _alloc(Self.B * TM1)
        imag_loss_cpu[Self.B, TI, ACTD, BINSv](
            acts, rewv, conv, vlog, svlog, pmean, pstd, bins,
            MINSTD, MAXSTD, Scalar[DT](0.95), Scalar[DT](3e-4), Scalar[DT](1.0),
            self.retnorm, pol_loss, val_loss, ret,
        )
        var total: Scalar[DT] = 0.0
        for i in range(Self.B * TM1):
            total += pol_loss[i] + val_loss[i]
        var rscale = self.retnorm.stats()[1]
        var d_pol = _alloc(Self.B * TM1)
        var d_val = _alloc(Self.B * TM1)
        for i in range(Self.B * TM1):
            d_pol[i] = 1.0
            d_val[i] = 1.0
        var g_vlog = _alloc(Self.B * TI * BINSv)
        var g_pmean = _alloc(Self.B * TI * ACTD)
        var g_pstd = _alloc(Self.B * TI * ACTD)
        imag_loss_backward[Self.B, TI, ACTD, BINSv](
            acts, rewv, conv, vlog, svlog, pmean, pstd, bins,
            MINSTD, MAXSTD, Scalar[DT](0.95), Scalar[DT](3e-4), Scalar[DT](1.0),
            rscale, d_pol, d_val, g_vlog, g_pmean, g_pstd,
        )
        self.oval.zero_grad["cpu", Self.ValT](self.value)
        self.opol.zero_grad["cpu", Self.PolT](self.policy)
        var gfeat = _alloc(Self.B * FEATv)
        var vscr = _alloc(Self.B * BINSv)
        var pscr = _alloc(Self.B * 2 * ACTD)
        var polg = _alloc(Self.B * 2 * ACTD)
        for t in range(TI):
            var ftt = _alloc(Self.B * FEATv)
            for b in range(Self.B):
                for k in range(FEATv):
                    ftt[b * FEATv + k] = feats[(b * TI + t) * FEATv + k]
            var fvt = TileTensor(ftt, row_major[Self.B, FEATv]())
            var vot = TileTensor(vscr, row_major[Self.B, BINSv]())
            self.value.forward["cpu", Self.B](fvt, output=vot)
            var gv = _alloc(Self.B * BINSv)
            for b in range(Self.B):
                for c in range(BINSv):
                    gv[b * BINSv + c] = g_vlog[(b * TI + t) * BINSv + c]
            var gvt = TileTensor(gv, row_major[Self.B, BINSv]())
            var gft = TileTensor(gfeat, row_major[Self.B, FEATv]())
            self.value.vjp["cpu", Self.B](gvt, gft)
            var fpt = TileTensor(ftt, row_major[Self.B, FEATv]())
            var pot = TileTensor(pscr, row_major[Self.B, 2 * ACTD]())
            self.policy.forward["cpu", Self.B](fpt, output=pot)
            for b in range(Self.B):
                for a in range(ACTD):
                    polg[b * 2 * ACTD + a] = g_pmean[(b * TI + t) * ACTD + a]
                    polg[b * 2 * ACTD + ACTD + a] = g_pstd[(b * TI + t) * ACTD + a]
            var pgt = TileTensor(polg, row_major[Self.B, 2 * ACTD]())
            var gft2 = TileTensor(gfeat, row_major[Self.B, FEATv]())
            self.policy.vjp["cpu", Self.B](pgt, gft2)
            gv.free(); ftt.free()
        self.oval.step["cpu", Self.ValT](self.value)
        self.opol.step["cpu", Self.PolT](self.policy)
        # Polyak slowvalue ← value (same module type → identical walk order)
        _polyak_value[Self.ValT](self.value, self.slowvalue, self.slow_rate)

        feats.free(); acts.free(); pmean.free(); pstd.free(); vlog.free()
        svlog.free(); rewv.free(); conv.free(); cd.free(); cs.free()
        fb.free(); pb.free(); vb.free(); svb.free(); rb.free(); cb.free()
        dummy1.free(); pol_loss.free(); val_loss.free(); ret.free()
        d_pol.free(); d_val.free(); g_vlog.free(); g_pmean.free()
        g_pstd.free(); gfeat.free(); vscr.free(); pscr.free(); polg.free()
        return total

    def train_step(mut self) raises -> Bool:
        if not self.can_train():
            return False
        var obs = _alloc(Self.B * (Self.T + 1) * Self.OBS)
        var act = _alloc(Self.B * Self.T * Self.ACT)
        var rew = _alloc(Self.B * Self.T)
        var dne = _alloc(Self.B * Self.T)
        self.replay.sample_batch[Self.B, Self.T](obs, act, rew, dne)
        var cdeter = _alloc((Self.T + 1) * Self.B * Self.DETER)
        var cstoch = _alloc((Self.T + 1) * Self.B * Self.SC)
        var toks = _alloc(Self.T * Self.B * Self.TOKEN)
        self.last_wm_loss = self._wm_step(obs, act, rew, dne, cdeter, cstoch, toks)
        # sync core/prior → imagine
        var names = List[String]()
        var ptrs = List[UnsafePointer[Scalar[DT], MutAnyOrigin]]()
        var lens = List[Int]()
        collect_params["cpu"](self.core, names, ptrs, lens)
        apply_params["cpu"](self.imagine, names, ptrs, lens)
        # imagination from final posterior carry
        var deter0 = cdeter + Self.T * Self.B * Self.DETER
        var stoch0 = cstoch + Self.T * Self.B * Self.SC
        var noise = _alloc(Self.T_IMAG * Self.B * Self.ACT)
        for i in range(Self.T_IMAG * Self.B * Self.ACT):
            noise[i] = Scalar[DT](random_float64() * 2.0 - 1.0)
        self.last_ac_loss = self._ac_step(deter0, stoch0, noise)
        self.train_steps += 1
        obs.free(); act.free(); rew.free(); dne.free()
        cdeter.free(); cstoch.free(); toks.free(); noise.free()
        _ = names^; _ = ptrs^; _ = lens^
        return True


# ── Polyak slow-value sync (module-level: avoids self-aliasing) ─────────
# src and dst share the SAME module type → for_each_param visits params in
# identical order, so an index-keyed collect-then-mix is exact.


@fieldwise_init
struct _PolyakCollect(ParamVisitor):
    var ptrs: UnsafePointer[
        List[UnsafePointer[Scalar[DT], MutAnyOrigin]], MutAnyOrigin
    ]

    def visit(
        mut self, name: String,
        param: TileTensor[dtype=DT, address_space=AddressSpace.GENERIC, element_size=1, ...],
        grad: TileTensor[dtype=DT, address_space=AddressSpace.GENERIC, element_size=1, ...],
        n_elems: Int, apply_decay: Bool,
    ) raises:
        self.ptrs[].append(
            rebind[UnsafePointer[Scalar[DT], MutAnyOrigin]](param.ptr)
        )


@fieldwise_init
struct _PolyakMix(ParamVisitor):
    var ptrs: UnsafePointer[
        List[UnsafePointer[Scalar[DT], MutAnyOrigin]], MutAnyOrigin
    ]
    var rate: Scalar[DT]
    var idx: Int

    def visit(
        mut self, name: String,
        param: TileTensor[dtype=DT, address_space=AddressSpace.GENERIC, element_size=1, ...],
        grad: TileTensor[dtype=DT, address_space=AddressSpace.GENERIC, element_size=1, ...],
        n_elems: Int, apply_decay: Bool,
    ) raises:
        var dp = rebind[UnsafePointer[Scalar[DT], MutAnyOrigin]](param.ptr)
        var sp = self.ptrs[][self.idx]
        var keep = Scalar[DT](1.0) - self.rate
        for k in range(n_elems):
            dp[k] = keep * dp[k] + self.rate * sp[k]
        self.idx += 1


def _polyak_value[V: Module](mut src: V, mut dst: V, rate: Scalar[DT]) raises:
    var sp = List[UnsafePointer[Scalar[DT], MutAnyOrigin]]()
    var c = _PolyakCollect(ptrs=UnsafePointer(to=sp))
    src.for_each_param["cpu", _PolyakCollect](String(""), c)
    var m = _PolyakMix(ptrs=UnsafePointer(to=sp), rate=rate, idx=0)
    dst.for_each_param["cpu", _PolyakMix](String(""), m)
    _ = sp^
