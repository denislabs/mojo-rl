"""DreamerV3Trainer — SAC-style block-composed trainer (CPU; GPU-ready).

Composes the `blocks.mojo` units (`WMStep` / `ParamSyncStep` / `ACStep`)
over a shared `DreamerState`, mirroring `deep_agents/sac/trainer.mojo`:
`train_target: StaticString` is comptime, `ctx: Optional[DeviceContext]` is
threaded through `make`/`step`. v1 instantiates `train_target="cpu"` (the
custom RSSM ops are CPU-only until PR5c Step 5 adds their GPU kernels; the
composition + plumbing are already GPU-shaped, so the GPU enable is localized
to each block's `comptime if target == "gpu"` branch + the ops' kernels).

`train_step` = sample a length-T window → `WMStep` (WM-BPTT) → `ParamSyncStep`
(core/prior → imagine) → `ACStep` (imagination AC + Polyak slowvalue).

Validated CPU: `test_dreamerv3_trainer_smoke.mojo` (synthetic) +
`test_dreamerv3_pendulum_smoke.mojo` (real Pendulum env loop).

Activation = SiLU (size1m/dmc config `act: silu`) — production type members
pass `SwishOp`; the validation spikes keep the GELU default for fixture parity.

v1 simplifications: random-action data collection (running-carry policy
action lives on `DreamerV3Agent`); imagination from the final posterior carry.
"""

from std.random import random_float64
from std.math import exp
from std.gpu.host import DeviceContext

from mojo_rl.nn.constants import DT
from mojo_rl.nn.storage.core.initializer import Kaiming
from mojo_rl.nn.storage.core.tensor import Tensor
from mojo_rl.nn.storage.core.tensor_refs import TensorRefs
from mojo_rl.nn.primitives.ops.swish_op import SwishOp
from mojo_rl.nn.storage.optimizer.dreamer_opt import DreamerOpt
from mojo_rl.nn.storage.optimizer.schedules import LinearWarmupSchedule
from mojo_rl.deep_agents.data.sequence_replay import SequenceReplay
from mojo_rl.deep_agents.dreamerv3.wm import (
    WMCoreGraph, WMImagineGraph, DecLossGraph, RewLossGraph, ConLossGraph,
)
from mojo_rl.deep_agents.dreamerv3.nets import (
    DreamerEncoder, DreamerValue, DreamerPolicyHead,
)
from mojo_rl.deep_agents.dreamerv3.twohot import (
    symexp_twohot_bins,
    twohot_pred,
    DREAMER_REWARD_GRID_LO,
)
from mojo_rl.deep_agents.dreamerv3.normalize import PercentileNormalize
from mojo_rl.deep_agents.dreamerv3.zero_init import (
    scale_output_module, scale_output_graph,
)
from mojo_rl.deep_agents.dreamerv3.blocks import (
    DreamerState, WMStep, ParamSyncStep, ACStep,
)


@always_inline
def _hp(mut t: Tensor) -> UnsafePointer[Scalar[DT], MutAnyOrigin]:
    """Sanctioned host-pointer view of a storage Tensor's CPU `data` — for the
    raw-pointer helpers (`twohot_pred`, sample_batch). CPU only."""
    return rebind[UnsafePointer[Scalar[DT], MutAnyOrigin]](t.data.unsafe_ptr())


@always_inline
def _symexp(x: Scalar[DT]) -> Scalar[DT]:
    """Inverse of symlog: sign(x)·(exp(|x|)−1). Decoder targets are symlog(obs),
    so symexp maps a decoder output back to raw observation space."""
    var s = Scalar[DT](1.0) if x >= Scalar[DT](0.0) else Scalar[DT](-1.0)
    var a = x if x >= Scalar[DT](0.0) else -x
    return s * (exp(a) - Scalar[DT](1.0))


@fieldwise_init
struct DreamerV3Trainer[
    train_target: StaticString,
    OBS: Int, ACT: Int, DETER: Int, H: Int, STOCH: Int, CLASSES: Int,
    BLOCKS: Int, TOKEN: Int, DEC_U: Int, HU: Int, VU: Int, PU: Int,
    BINS: Int, B: Int, T: Int, T_IMAG: Int, CAP: Int, DISCRETE: Bool = False,
](Movable & ImplicitlyDeletable):
    comptime SC = Self.STOCH * Self.CLASSES
    comptime FEAT = Self.DETER + Self.SC

    comptime EncT = DreamerEncoder[Self.OBS, Self.TOKEN, SwishOp]
    comptime CoreT = WMCoreGraph[
        Self.DETER, Self.H, Self.STOCH, Self.CLASSES, Self.BLOCKS, Self.ACT,
        Self.TOKEN, SwishOp,
    ]
    comptime DecT = DecLossGraph[Self.SC, Self.DETER, Self.OBS, Self.DEC_U, SwishOp]
    comptime RewT = RewLossGraph[Self.DETER, Self.SC, Self.HU, Self.BINS, SwishOp]
    comptime ConT = ConLossGraph[Self.DETER, Self.SC, Self.HU, SwishOp]
    comptime ValT = DreamerValue[Self.FEAT, Self.VU, Self.BINS, SwishOp]
    # discrete (categorical) actor → ACT logits; continuous → 2·ACT (mean,std)
    comptime PolT = DreamerPolicyHead[
        Self.FEAT, Self.PU, Self.ACT, Self.DISCRETE, SwishOp
    ]
    comptime ImagT = WMImagineGraph[
        Self.DETER, Self.H, Self.STOCH, Self.CLASSES, Self.BLOCKS, Self.ACT, SwishOp,
    ]
    comptime RepT = SequenceReplay[Self.OBS, Self.ACT, Self.CAP]
    comptime StateT = DreamerState[
        Self.OBS, Self.ACT, Self.DETER, Self.SC, Self.TOKEN, Self.B, Self.T,
        Self.T_IMAG,
    ]
    comptime WMBlk = WMStep[
        Self.OBS, Self.ACT, Self.DETER, Self.H, Self.STOCH, Self.CLASSES,
        Self.BLOCKS, Self.TOKEN, Self.DEC_U, Self.HU, Self.BINS, Self.B, Self.T,
    ]
    comptime SyncBlk = ParamSyncStep[
        Self.DETER, Self.H, Self.STOCH, Self.CLASSES, Self.BLOCKS, Self.ACT,
        Self.TOKEN,
    ]
    comptime ACBlk = ACStep[
        Self.OBS, Self.ACT, Self.DETER, Self.H, Self.STOCH, Self.CLASSES,
        Self.BLOCKS, Self.TOKEN, Self.HU, Self.VU, Self.PU, Self.BINS, Self.B,
        Self.T, Self.T_IMAG, Self.DISCRETE,
    ]

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

    var wm_blk: Self.WMBlk
    var sync_blk: Self.SyncBlk
    var ac_blk: Self.ACBlk

    var replay: Self.RepT
    var retnorm: PercentileNormalize
    var bins: List[Scalar[DT]]
    var state: Self.StateT
    var ctx: Optional[DeviceContext]
    var learning_starts: Int
    var train_steps: Int
    var warmup: LinearWarmupSchedule   # reference LR ramp 0→lr over warmup_steps

    @staticmethod
    def make(
        ctx: Optional[DeviceContext] = None,
        lr: Scalar[DT] = Scalar[DT](4e-5),
        learning_starts: Int = 200,
        warmup_steps: Int = 1000,
        out_init_scale: Scalar[DT] = Scalar[DT](0.0),
        actent: Scalar[DT] = Scalar[DT](3e-4),
        slowtar: Bool = False,
    ) raises -> Self:
        comptime assert (
            Self.train_target == "cpu" or Self.train_target == "gpu"
        ), "DreamerV3Trainer: train_target must be 'cpu' or 'gpu'"
        var enc = Self.EncT.make[Self.train_target, INIT=Kaiming](ctx=ctx)
        var core = Self.CoreT.make[Self.train_target, INIT=Kaiming](ctx=ctx)
        var dec = Self.DecT.make[Self.train_target, INIT=Kaiming](ctx=ctx)
        var rew = Self.RewT.make[Self.train_target, INIT=Kaiming](ctx=ctx)
        var con = Self.ConT.make[Self.train_target, INIT=Kaiming](ctx=ctx)
        var value = Self.ValT.make[Self.train_target, INIT=Kaiming](ctx=ctx)
        var slowvalue = Self.ValT.make[Self.train_target, INIT=Kaiming](ctx=ctx)
        var policy = Self.PolT.make[Self.train_target, INIT=Kaiming](ctx=ctx)
        var imagine = Self.ImagT.make[Self.train_target, INIT=Kaiming](ctx=ctx)

        # Finding 4 (paper p.6): scale the reward-predictor and critic OUTPUT
        # layers toward ~0 at init. Otherwise large/biased initial reward+value
        # predictions make the imagined λ-returns optimistic (even positive on
        # negative-reward tasks like Pendulum), and the actor optimizes a reward
        # landscape that doesn't exist. `out_init_scale=0.0` == the paper's hard
        # zero-init (best for negative rewards). A small nonzero scale keeps a
        # little Kaiming optimism, which empirically helps POSITIVE-reward tasks
        # (CartPole) explore / solve faster without the full-Kaiming blow-up.
        # The output Linear is Sequential child 3 (`nets.mojo` pins head MLP
        # depth to 1); inside the reward ComputeGraph the head is node `rew`.
        # slowvalue is scaled too — the value loss regularizes value TOWARD
        # slowvalue (slowreg=1), so a non-neutral slowvalue would pull it back.
        scale_output_graph[Self.train_target](
            rew, String("rew.3.weight"), String("rew.3.bias"), out_init_scale, ctx
        )
        scale_output_module[Self.train_target, Self.ValT](
            value, String("3.weight"), String("3.bias"), out_init_scale, ctx
        )
        scale_output_module[Self.train_target, Self.ValT](
            slowvalue, String("3.weight"), String("3.bias"), out_init_scale, ctx
        )

        # Storage DreamerOpt is driven INSIDE the blocks (graph.for_each_param /
        # opt.step[target, M]); the trainer just constructs them with the lr.
        var oe = DreamerOpt(lr=lr)
        var ocore = DreamerOpt(lr=lr)
        var odec = DreamerOpt(lr=lr)
        var orew = DreamerOpt(lr=lr)
        var ocon = DreamerOpt(lr=lr)
        var oval = DreamerOpt(lr=lr)
        var opol = DreamerOpt(lr=lr)

        var bins = List[Scalar[DT]](length=Self.BINS, fill=Scalar[DT](0.0))
        # Reward/value twohot bins, narrowed (max bin ≈ 8102 vs the reference's
        # symexp(20) ≈ 4.85e8). Two reasons for the narrow grid:
        #   (1) huge bins amplify float noise in `Σ softmax·bins` (the e-tail
        #       over a 4.85e8 bin), which breaks CPU↔GPU AC parity;
        #   (2) they turn off-distribution head errors into 1e5–1e6 predictions.
        # This grid MUST equal the grid the reward head is TRAINED on
        # (`TwoHotLoss.make` in wm_loss_ops.mojo). Both now read the SAME
        # `DREAMER_REWARD_GRID_LO` constant (S4) so they can no longer diverge —
        # a past -9-vs-(-20)-default split decoded reward ~5× small, starving
        # imagined returns.
        symexp_twohot_bins[Self.BINS](
            rebind[UnsafePointer[Scalar[DT], MutAnyOrigin]](bins.unsafe_ptr()),
            lo=Scalar[DT](DREAMER_REWARD_GRID_LO),
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
            wm_blk=Self.WMBlk.make[Self.train_target](ctx=ctx),
            sync_blk=Self.SyncBlk.make[Self.train_target](ctx=ctx),
            ac_blk=Self.ACBlk.make[Self.train_target](
                ctx=ctx, actent=actent, slowtar=slowtar
            ),
            # Replay stays host-resident on both targets; the GPU WMStep
            # uploads the sampled batch per-step (so make["cpu"] always).
            replay=Self.RepT.make["cpu"](ctx=ctx),
            retnorm=retnorm^,
            bins=bins^,
            state=Self.StateT.make[Self.train_target](ctx=ctx),
            ctx=ctx,
            learning_starts=learning_starts,
            train_steps=0,
            warmup=LinearWarmupSchedule.make(lr, warmup_steps),
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

    def last_wm_loss(self) -> Scalar[DT]:
        return self.state.last_wm_loss

    def last_ac_loss(self) -> Scalar[DT]:
        return self.state.last_ac_loss

    def dbg_real_rew(self) -> Scalar[DT]:
        return self.state.dbg_real_rew

    def dbg_rew_pred(self) -> Scalar[DT]:
        return self.state.dbg_rew_pred

    def dbg_ret_mean(self) -> Scalar[DT]:
        return self.state.dbg_ret_mean

    def dbg_ret_std(self) -> Scalar[DT]:
        return self.state.dbg_ret_std

    def dbg_pmean_abs(self) -> Scalar[DT]:
        return self.state.dbg_pmean_abs

    def dbg_val_mean(self) -> Scalar[DT]:
        return self.state.dbg_val_mean

    def dbg_pstd(self) -> Scalar[DT]:
        return self.state.dbg_pstd

    def dbg_rscale(self) -> Scalar[DT]:
        return self.state.dbg_rscale

    def train_step(mut self) raises -> Bool:
        if not self.can_train():
            return False
        # reference LR warmup: ramp 0→lr over warmup_steps (all modules).
        var clr = self.warmup.lr_at(self.train_steps)
        self.oe.lr = clr; self.ocore.lr = clr; self.odec.lr = clr
        self.orew.lr = clr; self.ocon.lr = clr; self.oval.lr = clr
        self.opol.lr = clr
        # sample a length-T window into the shared state batch buffers (storage
        # Tensors; sample_batch takes raw host pointers → rebind .data ptr).
        self.replay.sample_batch[Self.B, Self.T](
            _hp(self.state.mb_obs), _hp(self.state.mb_act),
            _hp(self.state.mb_rew), _hp(self.state.mb_dne),
        )
        var rr: Scalar[DT] = 0.0
        for i in range(Self.B * Self.T):
            rr += self.state.mb_rew.data[i]
        self.state.dbg_real_rew = rr / Scalar[DT](Self.B * Self.T)
        # imagination sampling noise: NS = T*B starts × T_IMAG steps × ACT.
        # Pre-filled here (both targets) so _ac_cpu and _ac_gpu read the SAME
        # noise[(t*NS+b)*ACT+a] → CPU↔GPU bit-match.
        for i in range(Self.T_IMAG * Self.T * Self.B * Self.ACT):
            self.state.noise.data[i] = Scalar[DT](random_float64() * 2.0 - 1.0)
        # WM-BPTT → fills state.cdeter / cstoch + state.last_wm_loss
        self.wm_blk.step[Self.train_target, Self.T_IMAG](
            self.state, self.enc, self.core, self.dec, self.rew, self.con,
            self.oe, self.ocore, self.odec, self.orew, self.ocon,
        )
        # core/prior → imagine mirror
        self.sync_blk.step[Self.train_target](
            self.core, self.imagine, ctx=self.ctx
        )
        # imagination AC + Polyak → state.last_ac_loss
        self.ac_blk.step[Self.train_target](
            self.state, self.imagine, self.value, self.slowvalue, self.policy,
            self.rew, self.con, self.oval, self.opol, self.retnorm,
            rebind[UnsafePointer[Scalar[DT], MutAnyOrigin]](
                self.bins.unsafe_ptr()
            ),
        )
        self.train_steps += 1
        return True

    def openloop_report(
        mut self,
        real_obs: UnsafePointer[Scalar[DT], MutAnyOrigin],   # [(ctx+hor+1)*OBS]
        real_act: UnsafePointer[Scalar[DT], MutAnyOrigin],   # [(ctx+hor)*ACT] NORMALIZED [-1,1]
        real_rew: UnsafePointer[Scalar[DT], MutAnyOrigin],   # [ctx+hor]
        ctx_len: Int,
        hor: Int,
        out_ol_obs: UnsafePointer[Scalar[DT], MutAnyOrigin],  # [hor] open-loop obs MSE (raw space)
        out_tf_obs: UnsafePointer[Scalar[DT], MutAnyOrigin],  # [hor] teacher-forced obs MSE
        out_ol_rew: UnsafePointer[Scalar[DT], MutAnyOrigin],  # [hor] open-loop |reward err|
        out_tf_rew: UnsafePointer[Scalar[DT], MutAnyOrigin],  # [hor] teacher-forced |reward err|
    ) raises:
        """Open-loop WM-accuracy probe (Finding-4 follow-up diagnostic).

        Builds the posterior belief by OBSERVING `ctx_len` real steps, then rolls
        `hor` steps forward TWO ways and compares decoded predictions to reality:
          * open-loop  — `imagine` (PRIOR dynamics, no obs) fed the real recorded
            actions. Tests whether multi-step dynamics stay accurate.
          * teacher-forced — `core` re-observes each real obs (POSTERIOR). The 1-step
            recon floor (decoder + posterior quality).
        If `out_ol_obs` grows steeply with horizon while `out_tf_obs` stays small,
        the world model's *dynamics* are the bottleneck (capacity / latent
        resolution) — not a code bug, not the decoder. Reward `out_ol_rew` shows
        whether the predicted reward tracks reality along the rollout.

        Uses `imagine` (synced from `core` every train_step) for the prior path, so
        call AFTER training. CPU-only — run with a CPU agent; the WM forward is
        bit-identical CPU↔GPU (parity test), so the diagnosis transfers to GPU runs.
        """
        comptime assert Self.train_target == "cpu", (
            "openloop_report is CPU-only — run the diagnostic with a CPU agent"
        )
        comptime D = Self.DETER
        comptime SCl = Self.SC
        comptime TOK = Self.TOKEN
        comptime OBSD = Self.OBS
        comptime ACTD = Self.ACT
        comptime FEATl = Self.FEAT
        comptime BINSl = Self.BINS
        comptime CARRY = 2 + D + SCl
        var bins = rebind[UnsafePointer[Scalar[DT], MutAnyOrigin]](
            self.bins.unsafe_ptr()
        )

        # Belief carry (host Tensors, reused as set_input sources).
        var bd = Tensor.alloc(D)
        var bs = Tensor.alloc(SCl)
        var tok = Tensor.alloc(TOK)
        var pa = Tensor.alloc(ACTD)
        var obt = Tensor.alloc(OBSD)       # obs frame staging (enc input)
        var dummyO = Tensor.alloc(OBSD)    # recon target (unused)
        var dummy1 = Tensor.alloc(1)       # reward target (unused)
        # reusable mutable output Tensors (forward writes the loss/feat slot).
        var carry_t = Tensor.alloc(CARRY)
        var feat_t = Tensor.alloc(FEATl)
        var loss_t = Tensor.alloc(1)

        # ── context: observe ctx_len real steps (prev-action convention) ──
        for t in range(ctx_len):
            if t == 0:
                for k in range(ACTD):
                    pa.data[k] = 0.0
            else:
                for k in range(ACTD):
                    pa.data[k] = real_act[(t - 1) * ACTD + k]
            for k in range(OBSD):
                obt.data[k] = real_obs[t * OBSD + k]
            self.enc.forward["cpu", 1](TensorRefs[1](obt), tok, None)
            self.core.set_input["deter", 1](bd, None)
            self.core.set_input["stoch", 1](bs, None)
            self.core.set_input["action", 1](pa, None)
            self.core.set_input["tokens", 1](tok, None)
            self.core.forward[1, "cpu"](carry_t, None)
            ref nd = self.core.node_output["nd"]()
            ref sn = self.core.node_output["stoch_new"]()
            for k in range(D):
                bd.data[k] = nd.data[k]
            for k in range(SCl):
                bs.data[k] = sn.data[k]

        # open-loop + teacher-forced belief chains, both seeded from `bd/bs`.
        var old = Tensor.alloc(D); var ols = Tensor.alloc(SCl)
        var tfd = Tensor.alloc(D); var tfs = Tensor.alloc(SCl)
        # staging Tensors for the per-step posterior carry fed back into graphs.
        var ond_t = Tensor.alloc(D); var osn_t = Tensor.alloc(SCl)
        var tnd_t = Tensor.alloc(D); var tsn_t = Tensor.alloc(SCl)
        for k in range(D):
            old.data[k] = bd.data[k]; tfd.data[k] = bd.data[k]
        for k in range(SCl):
            ols.data[k] = bs.data[k]; tfs.data[k] = bs.data[k]

        for h in range(hor):
            var idx = ctx_len - 1 + h
            for k in range(ACTD):
                pa.data[k] = real_act[idx * ACTD + k]
            # ── open-loop: prior dynamics (imagine), no observation ──
            self.imagine.set_input["deter", 1](old, None)
            self.imagine.set_input["stoch", 1](ols, None)
            self.imagine.set_input["action", 1](pa, None)
            self.imagine.forward[1, "cpu"](feat_t, None)
            ref ond = self.imagine.node_output["nd"]()
            ref osn = self.imagine.node_output["stoch_new"]()
            for k in range(D):
                ond_t.data[k] = ond.data[k]
            for k in range(SCl):
                osn_t.data[k] = osn.data[k]
            self.dec.set_input["stoch_new", 1](osn_t, None)
            self.dec.set_input["nd", 1](ond_t, None)
            self.dec.set_input["rtgt", 1](dummyO, None)
            self.dec.forward[1, "cpu"](loss_t, None)
            ref pred = self.dec.node_output["dec"]()
            var ms: Scalar[DT] = 0.0
            for k in range(OBSD):
                var dv = _symexp(pred.data[k]) - real_obs[(idx + 1) * OBSD + k]
                ms += dv * dv
            out_ol_obs[h] = ms / Scalar[DT](OBSD)
            self.rew.set_input["nd", 1](ond_t, None)
            self.rew.set_input["stoch_new", 1](osn_t, None)
            self.rew.set_input["rtgt", 1](dummy1, None)
            self.rew.forward[1, "cpu"](loss_t, None)
            ref rl = self.rew.node_output["rew"]()
            var re = twohot_pred[BINSl](_hp(rl), 0, bins) - real_rew[idx]
            out_ol_rew[h] = re if re >= Scalar[DT](0.0) else -re
            for k in range(D):
                old.data[k] = ond_t.data[k]
            for k in range(SCl):
                ols.data[k] = osn_t.data[k]

            # ── teacher-forced: re-observe the real obs (posterior path) ──
            for k in range(OBSD):
                obt.data[k] = real_obs[(idx + 1) * OBSD + k]
            self.enc.forward["cpu", 1](TensorRefs[1](obt), tok, None)
            self.core.set_input["deter", 1](tfd, None)
            self.core.set_input["stoch", 1](tfs, None)
            self.core.set_input["action", 1](pa, None)
            self.core.set_input["tokens", 1](tok, None)
            self.core.forward[1, "cpu"](carry_t, None)
            ref tnd = self.core.node_output["nd"]()
            ref tsn = self.core.node_output["stoch_new"]()
            for k in range(D):
                tnd_t.data[k] = tnd.data[k]
            for k in range(SCl):
                tsn_t.data[k] = tsn.data[k]
            self.dec.set_input["stoch_new", 1](tsn_t, None)
            self.dec.set_input["nd", 1](tnd_t, None)
            self.dec.set_input["rtgt", 1](dummyO, None)
            self.dec.forward[1, "cpu"](loss_t, None)
            ref tpred = self.dec.node_output["dec"]()
            var tms: Scalar[DT] = 0.0
            for k in range(OBSD):
                var dv = _symexp(tpred.data[k]) - real_obs[(idx + 1) * OBSD + k]
                tms += dv * dv
            out_tf_obs[h] = tms / Scalar[DT](OBSD)
            # teacher-forced reward: head prediction on the REAL posterior state
            # vs real reward — isolates head calibration from dynamics drift.
            self.rew.set_input["nd", 1](tnd_t, None)
            self.rew.set_input["stoch_new", 1](tsn_t, None)
            self.rew.set_input["rtgt", 1](dummy1, None)
            self.rew.forward[1, "cpu"](loss_t, None)
            ref trl = self.rew.node_output["rew"]()
            var tre = twohot_pred[BINSl](_hp(trl), 0, bins) - real_rew[idx]
            out_tf_rew[h] = tre if tre >= Scalar[DT](0.0) else -tre
            for k in range(D):
                tfd.data[k] = tnd_t.data[k]
            for k in range(SCl):
                tfs.data[k] = tsn_t.data[k]
