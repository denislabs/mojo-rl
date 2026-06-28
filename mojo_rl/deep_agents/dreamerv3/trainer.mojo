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
from mojo_rl.nn.core.initializer import Kaiming
from mojo_rl.nn.core.tensor import Tensor
from mojo_rl.nn.core.tensor_refs import TensorRefs
from mojo_rl.nn.core.checkpoint import (
    CheckpointWriter, CheckpointReader, _split_lines,
)
from mojo_rl.nn.primitives.ops.swish_op import SwishOp
from mojo_rl.nn.optimizer.dreamer_opt import DreamerOpt
from mojo_rl.nn.optimizer.schedules import LinearWarmupSchedule
from mojo_rl.deep_agents.data.any_sequence_replay import AnySequenceReplay
from mojo_rl.cuda import CUDAGraph, maybe_capture_replay
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
    # Sequence replay backend, selected by `train_target` via the carry-both
    # `AnySequenceReplay` shim (Mojo has no type-ternary). GPU resolves to the
    # device-resident `GPUSequenceReplay` (device circular storage + device
    # Philox window draws → no per-step host B×T sampling, capture-safe); CPU
    # to the host `SequenceReplay`.
    comptime RepT = AnySequenceReplay[
        Self.train_target, Self.OBS, Self.ACT, Self.CAP
    ]
    comptime StateT = DreamerState[
        Self.OBS, Self.ACT, Self.DETER, Self.SC, Self.TOKEN, Self.B, Self.T,
        Self.T_IMAG,
    ]
    comptime WMBlk = WMStep[
        Self.OBS, Self.ACT, Self.DETER, Self.H, Self.STOCH, Self.CLASSES,
        Self.BLOCKS, Self.TOKEN, Self.DEC_U, Self.HU, Self.BINS, Self.B, Self.T,
        Self.DISCRETE,
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
    # Lazily-captured discrete-GPU train-step graph (Stage 3 P5; `None` until the
    # first capture). Moved into a disjoint local for the `maybe_capture_replay`
    # call (mbpo idiom) so the capture closure can borrow `self` mutably.
    var _train_graph: Optional[CUDAGraph]

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
            # GPU: device-resident GPUSequenceReplay (sampled straight into the
            # WM device buffers); CPU: host SequenceReplay. The shim picks the
            # backend off `train_target`; the GPU side requires `ctx`.
            replay=Self.RepT.make(ctx=ctx),
            retnorm=retnorm^,
            bins=bins^,
            state=Self.StateT.make[Self.train_target](ctx=ctx),
            ctx=ctx,
            learning_starts=learning_starts,
            train_steps=0,
            warmup=LinearWarmupSchedule.make(lr, warmup_steps),
            _train_graph=None,
        )

    def record(
        mut self,
        obs: UnsafePointer[Scalar[DT], MutAnyOrigin],
        act: UnsafePointer[Scalar[DT], MutAnyOrigin],
        reward: Scalar[DT],
        done: Scalar[DT],
    ) raises:
        self.replay.record(obs, act, reward, done)

    def record_terminal(
        mut self, obs: UnsafePointer[Scalar[DT], MutAnyOrigin]
    ) raises:
        """Store a genuine terminal observation (call right after `record(done=1)`).
        Lets the WM continue head learn `latent(terminal)→0`."""
        self.replay.record_terminal(obs)

    def can_train(self) -> Bool:
        return self.replay.count() >= Self.T + 1

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

    def dbg_con_mean(self) -> Scalar[DT]:
        return self.state.dbg_con_mean

    def dbg_con_min(self) -> Scalar[DT]:
        return self.state.dbg_con_min

    def dbg_val_std(self) -> Scalar[DT]:
        return self.state.dbg_val_std

    def dbg_feat_std(self) -> Scalar[DT]:
        return self.state.dbg_feat_std

    def dbg_dyn_kl(self) -> Scalar[DT]:
        return self.state.dbg_dyn_kl

    def dbg_rep_kl(self) -> Scalar[DT]:
        return self.state.dbg_rep_kl

    def dbg_obs_loss(self) -> Scalar[DT]:
        return self.state.dbg_obs_loss

    def dbg_rew_loss(self) -> Scalar[DT]:
        return self.state.dbg_rew_loss

    def dbg_con_loss(self) -> Scalar[DT]:
        return self.state.dbg_con_loss

    def dbg_pol_loss(self) -> Scalar[DT]:
        return self.state.dbg_pol_loss

    def dbg_val_loss(self) -> Scalar[DT]:
        return self.state.dbg_val_loss

    def train_steps_done(self) -> Int:
        return self.train_steps

    def train_step(mut self, want_diag: Bool = True) raises -> Bool:
        if not self.can_train():
            return False
        self.train_prologue(want_diag)
        self._device_step(want_diag)
        self.train_steps += 1
        return True

    def train_prologue(mut self, want_diag: Bool) raises:
        """Eager per-step work that is NOT part of the captured device-kernel
        sequence (Stage 3 P5): LR refresh + minibatch draw + imagination-noise
        fill/upload. Must run before `_device_step` / `train_device_kernels`
        each step so the (fixed) device input buffers — `wm_blk.mb*_d`,
        `ac_blk.noise_d`, `state.d_*` — hold this step's fresh data before a
        graph replay reads them."""
        # reference LR warmup: ramp 0→lr over warmup_steps (all modules).
        var clr = self.warmup.lr_at(self.train_steps)
        self.oe.lr = clr; self.ocore.lr = clr; self.odec.lr = clr
        self.orew.lr = clr; self.ocon.lr = clr; self.oval.lr = clr
        self.opol.lr = clr
        self._draw_minibatch(want_diag)
        self._fill_noise()

    def _draw_minibatch(mut self, want_diag: Bool) raises:
        """Sample one length-T window into the minibatch buffers the WM/AC
        blocks consume.

        CPU: host `SequenceReplay.sample_batch_fst` → `state.mb_*` (raw host
        pointers).

        GPU: device `GPUSequenceReplay.sample_batch_fst_dev` straight into the
        WM device buffers (`wm_blk.mb*_d`) — no host obs/act gather, no upload
        (the windows are drawn with the buffer's device Philox RNG). rew/dne:
          * discrete (device-resident AC, the capture target): copied
            device→device into the shared `state.d_rew/d_cont`, which the AC
            reads directly — no D2H/H2D round-trip. `dbg_real_rew` is then the
            only host readout, and it's `want_diag`-gated → the non-diag draw is
            host-free.
          * continuous (host-side repval): pulled to `state.mb_rew/mb_dne`."""
        comptime if Self.train_target == "gpu":
            var sctx = self.ctx.value()
            self.replay.sample_batch_fst_dev[Self.B, Self.T](
                sctx,
                self.wm_blk.mbobs_d.dev.value(),
                self.wm_blk.mbact_d.dev.value(),
                self.wm_blk.mbrew_d.dev.value(),
                self.wm_blk.mbdne_d.dev.value(),
                self.wm_blk.mbfst_d.dev.value(),
            )
            comptime if Self.DISCRETE:
                # rew/dne stay on device (st.d_rew / st.d_cont carries dne) for
                # the device-resident AC; only dbg pulls them, gated.
                sctx.enqueue_copy(
                    self.state.d_rew.dev.value(),
                    self.wm_blk.mbrew_d.dev.value(),
                )
                sctx.enqueue_copy(
                    self.state.d_cont.dev.value(),
                    self.wm_blk.mbdne_d.dev.value(),
                )
                if want_diag:
                    self.wm_blk.mbrew_d.download(sctx)
                    var rr: Scalar[DT] = 0.0
                    for i in range(Self.B * Self.T):
                        rr += self.wm_blk.mbrew_d.data[i]
                    self.state.dbg_real_rew = rr / Scalar[DT](Self.B * Self.T)
            else:
                self.wm_blk.mbrew_d.download(sctx)
                self.wm_blk.mbdne_d.download(sctx)
                for i in range(Self.B * Self.T):
                    self.state.mb_rew.data[i] = self.wm_blk.mbrew_d.data[i]
                    self.state.mb_dne.data[i] = self.wm_blk.mbdne_d.data[i]
                var rr: Scalar[DT] = 0.0
                for i in range(Self.B * Self.T):
                    rr += self.state.mb_rew.data[i]
                self.state.dbg_real_rew = rr / Scalar[DT](Self.B * Self.T)
        else:
            self.replay.sample_batch_fst[Self.B, Self.T](
                _hp(self.state.mb_obs), _hp(self.state.mb_act),
                _hp(self.state.mb_rew), _hp(self.state.mb_dne),
                _hp(self.state.mb_fst),
            )
            var rr: Scalar[DT] = 0.0
            for i in range(Self.B * Self.T):
                rr += self.state.mb_rew.data[i]
            self.state.dbg_real_rew = rr / Scalar[DT](Self.B * Self.T)

    def load_minibatch(
        mut self,
        obs: UnsafePointer[Scalar[DT], MutAnyOrigin],   # [B*(T+1)*OBS]
        act: UnsafePointer[Scalar[DT], MutAnyOrigin],   # [B*T*ACT]
        rew: UnsafePointer[Scalar[DT], MutAnyOrigin],   # [B*T]
        dne: UnsafePointer[Scalar[DT], MutAnyOrigin],   # [B*T]
        fst: UnsafePointer[Scalar[DT], MutAnyOrigin],   # [B*(T+1)]
    ) raises:
        """Load a FIXED minibatch into the WM/AC buffers, bypassing the replay
        draw. The CPU↔GPU parity gate uses this to feed both backends a
        byte-identical window — their samplers now genuinely differ (host RNG
        vs device Philox), so `train_step`'s own draw can no longer be compared
        directly. Same buffer layout as `_draw_minibatch` fills."""
        comptime OBSD = Self.OBS
        comptime ACTD = Self.ACT
        for i in range(Self.B * Self.T):
            self.state.mb_rew.data[i] = rew[i]
            self.state.mb_dne.data[i] = dne[i]
        comptime if Self.train_target == "gpu":
            var sctx = self.ctx.value()
            for i in range(Self.B * (Self.T + 1) * OBSD):
                self.wm_blk.mbobs_d.data[i] = obs[i]
            for i in range(Self.B * Self.T * ACTD):
                self.wm_blk.mbact_d.data[i] = act[i]
            for i in range(Self.B * Self.T):
                self.wm_blk.mbrew_d.data[i] = rew[i]
                self.wm_blk.mbdne_d.data[i] = dne[i]
            for i in range(Self.B * (Self.T + 1)):
                self.wm_blk.mbfst_d.data[i] = fst[i]
            self.wm_blk.mbobs_d.upload(sctx)
            self.wm_blk.mbact_d.upload(sctx)
            self.wm_blk.mbrew_d.upload(sctx)
            self.wm_blk.mbdne_d.upload(sctx)
            self.wm_blk.mbfst_d.upload(sctx)
            comptime if Self.DISCRETE:
                # mirror _draw_minibatch: hand rew/dne to the device-resident AC
                # via the shared state buffers (st.d_cont carries dne).
                sctx.enqueue_copy(
                    self.state.d_rew.dev.value(),
                    self.wm_blk.mbrew_d.dev.value(),
                )
                sctx.enqueue_copy(
                    self.state.d_cont.dev.value(),
                    self.wm_blk.mbdne_d.dev.value(),
                )
        else:
            for i in range(Self.B * (Self.T + 1) * OBSD):
                self.state.mb_obs.data[i] = obs[i]
            for i in range(Self.B * Self.T * ACTD):
                self.state.mb_act.data[i] = act[i]
            for i in range(Self.B * (Self.T + 1)):
                self.state.mb_fst.data[i] = fst[i]
        var rr: Scalar[DT] = 0.0
        for i in range(Self.B * Self.T):
            rr += self.state.mb_rew.data[i]
        self.state.dbg_real_rew = rr / Scalar[DT](Self.B * Self.T)

    def _fill_noise(mut self) raises:
        """Imagination sampling noise (eager prologue): NS = T*B starts ×
        T_IMAG steps × ACT. Host-filled so _ac_cpu and the discrete GPU AC read
        the SAME noise[(t*NS+b)*ACT+a] → CPU↔GPU bit-match. On discrete GPU the
        noise is uploaded to `ac_blk.noise_d` here (P4) so the captured AC reads
        a pre-filled device buffer — out of the captured region."""
        for i in range(Self.T_IMAG * Self.T * Self.B * Self.ACT):
            self.state.noise.data[i] = Scalar[DT](random_float64() * 2.0 - 1.0)
        comptime if Self.train_target == "gpu" and Self.DISCRETE:
            var nctx = self.ctx.value()
            for i in range(Self.T_IMAG * Self.T * Self.B * Self.ACT):
                self.ac_blk.noise_d.data[i] = self.state.noise.data[i]
            self.ac_blk.noise_d.upload(nctx)

    def _device_step(mut self, want_diag: Bool) raises:
        """WM-BPTT → ParamSync → imagination AC over the CURRENTLY-LOADED
        minibatch + noise (filled by `train_prologue` / `load_minibatch`). This
        is the pure compute sequence; on discrete GPU with `want_diag=False` it
        is fully device-resident + sync-free (Stage 3 P1–P4) → the body the
        CUDA-graph capture closure replays. Does NOT advance `train_steps` (the
        caller does, once per logical update, via `train_step` /
        `note_train_update`)."""
        # WM-BPTT → fills the carry (device for discrete, host for continuous)
        # + state.last_wm_loss/dbg (want_diag-gated; the optimizer steps run
        # regardless, so non-diag steps still train — just no loss readout).
        self.wm_blk.step[Self.train_target, Self.T_IMAG](
            self.state, self.enc, self.core, self.dec, self.rew, self.con,
            self.oe, self.ocore, self.odec, self.orew, self.ocon,
            want_diag=want_diag,
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
            want_diag,
        )

    def _run_minibatch(mut self, want_diag: Bool) raises:
        """Noise fill + device step over an ALREADY-LOADED minibatch, advancing
        the step counter. `train_step` uses `train_prologue` + `_device_step`
        directly; this wrapper is the entry the CPU↔GPU parity gate drives after
        `load_minibatch` (no draw, no LR refresh — just noise + compute)."""
        self._fill_noise()
        self._device_step(want_diag)
        self.train_steps += 1

    # ─── CUDA-graph capture surface (Stage 3 P5) ───────────────────────────
    #
    # The discrete-GPU non-diag step is sync/D2H-free (P1–P4), so its WM+AC
    # device-kernel sequence is capturable. `train_device_kernels` is that
    # sequence with NO host work and NO counter advance — the body of the
    # driver's capture closure. The eager prologue (sample + noise, which read
    # host RNG / use host ring indices) is NOT captured: the caller runs
    # `train_prologue` each step before replay so the fixed device input
    # buffers are refreshed. Mirrors `SACAgent.train_device_kernels`.
    def train_device_kernels(mut self) raises:
        comptime assert Self.train_target == "gpu" and Self.DISCRETE, (
            "train_device_kernels is the discrete-GPU CUDA-graph capture path"
        )
        self._device_step(want_diag=False)

    def note_train_update(mut self):
        """Advance one logical update's host counter under graph replay (the
        device work is replayed by the captured graph; loss/dbg scalars are
        only refreshed on the eager want_diag steps)."""
        self.train_steps += 1

    def learning_starts_count(self) -> Int:
        return self.learning_starts

    def train_step_captured(mut self, want_diag: Bool = False) raises -> Bool:
        """CUDA-graph train step for the discrete-GPU path. `want_diag` steps
        take the eager `train_step` (the diagnostic readout — D2H + host sum —
        can't be captured); non-diag steps run the eager prologue (sample +
        noise → fixed device buffers) then capture-once / replay the WM+AC
        device-kernel sequence (`train_device_kernels`). Host counter advances
        via `note_train_update`. On non-NVIDIA `maybe_capture_replay` runs the
        closure directly → identical to `train_step(want_diag=False)` (the
        Apple transparency path; real capture/replay = NVIDIA).

        The `_train_graph` field is moved into a disjoint local for the capture
        call (mbpo idiom) so the closure can borrow `self` mutably without
        overlapping a mut borrow of the field."""
        comptime assert Self.train_target == "gpu" and Self.DISCRETE, (
            "train_step_captured is the discrete-GPU CUDA-graph path"
        )
        if not self.can_train():
            return False
        if want_diag:
            return self.train_step(want_diag=True)
        var ctx = self.ctx.value()
        # Eager prologue: refresh the FIXED device input buffers this step.
        self.train_prologue(want_diag=False)
        var g = self._train_graph^
        self._train_graph = None

        def _captured() capturing raises -> None:
            self.train_device_kernels()

        maybe_capture_replay[_captured](g, ctx)
        self._train_graph = g^
        self.note_train_update()
        return True

    # ─── Checkpoint (ONE file: the whole world model + actor-critic) ───────
    def save_state(mut self, path: String) raises:
        """Write the full DreamerV3 network set into a SINGLE `nn-ckpt v2` file,
        sections name-prefixed per module. `imagine` is NOT saved — it is a
        read-only mirror of `core` re-synced every train_step (and unused by the
        greedy/inference path). Optimizer moments are NOT persisted (resume
        re-warms), matching the SAC checkpoint convention."""
        var w = CheckpointWriter(save_moments=False)
        w.mode = 0
        self.enc.for_each_param[Self.train_target](w, self.ctx, "enc")
        self.core.for_each_param[Self.train_target](w, self.ctx, "core")
        self.dec.for_each_param[Self.train_target](w, self.ctx, "dec")
        self.rew.for_each_param[Self.train_target](w, self.ctx, "rew")
        self.con.for_each_param[Self.train_target](w, self.ctx, "con")
        self.value.for_each_param[Self.train_target](w, self.ctx, "value")
        self.slowvalue.for_each_param[Self.train_target](w, self.ctx, "slowvalue")
        self.policy.for_each_param[Self.train_target](w, self.ctx, "policy")
        w.mode = 1
        self.enc.for_each_state[Self.train_target](w, self.ctx, "enc")
        self.core.for_each_state[Self.train_target](w, self.ctx, "core")
        self.dec.for_each_state[Self.train_target](w, self.ctx, "dec")
        self.rew.for_each_state[Self.train_target](w, self.ctx, "rew")
        self.con.for_each_state[Self.train_target](w, self.ctx, "con")
        self.value.for_each_state[Self.train_target](w, self.ctx, "value")
        self.slowvalue.for_each_state[Self.train_target](w, self.ctx, "slowvalue")
        self.policy.for_each_state[Self.train_target](w, self.ctx, "policy")
        with open(path, "w") as f:
            f.write(w.content)

    def load_state(mut self, path: String) raises:
        """Restore the full network set from the single envelope (same walk
        order as `save_state`). `imagine` is re-synced from `core` on the next
        `train_step`."""
        var content: String
        with open(path, "r") as f:
            content = String(f.read())
        var lines = _split_lines(content)
        var body = List[String]()
        for li in range(len(lines)):
            if lines[li].startswith("storage-ckpt"):
                continue
            body.append(lines[li])
        var r = CheckpointReader(body^)
        r.mode = 0
        self.enc.for_each_param[Self.train_target](r, self.ctx, "enc")
        self.core.for_each_param[Self.train_target](r, self.ctx, "core")
        self.dec.for_each_param[Self.train_target](r, self.ctx, "dec")
        self.rew.for_each_param[Self.train_target](r, self.ctx, "rew")
        self.con.for_each_param[Self.train_target](r, self.ctx, "con")
        self.value.for_each_param[Self.train_target](r, self.ctx, "value")
        self.slowvalue.for_each_param[Self.train_target](r, self.ctx, "slowvalue")
        self.policy.for_each_param[Self.train_target](r, self.ctx, "policy")
        r.mode = 1
        self.enc.for_each_state[Self.train_target](r, self.ctx, "enc")
        self.core.for_each_state[Self.train_target](r, self.ctx, "core")
        self.dec.for_each_state[Self.train_target](r, self.ctx, "dec")
        self.rew.for_each_state[Self.train_target](r, self.ctx, "rew")
        self.con.for_each_state[Self.train_target](r, self.ctx, "con")
        self.value.for_each_state[Self.train_target](r, self.ctx, "value")
        self.slowvalue.for_each_state[Self.train_target](r, self.ctx, "slowvalue")
        self.policy.for_each_state[Self.train_target](r, self.ctx, "policy")

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

    def openloop_trace(
        mut self,
        real_obs: UnsafePointer[Scalar[DT], MutAnyOrigin],   # [(ctx+hor+1)*OBS]
        real_act: UnsafePointer[Scalar[DT], MutAnyOrigin],   # [(ctx+hor)*ACT]
        ctx_len: Int,
        hor: Int,
        out_ol_obs: UnsafePointer[Scalar[DT], MutAnyOrigin],  # [hor*OBS] open-loop decoded obs (raw space)
        out_tf_obs: UnsafePointer[Scalar[DT], MutAnyOrigin],  # [hor*OBS] teacher-forced decoded obs (raw space)
        out_ol_con: UnsafePointer[Scalar[DT], MutAnyOrigin],  # [hor] open-loop continue prob = sigmoid(con logit)
        out_tf_con: UnsafePointer[Scalar[DT], MutAnyOrigin],  # [hor] teacher-forced continue prob
    ) raises:
        """Per-component open-loop WM trace (CartPole fidelity diagnostic).

        Same belief/rollout machinery as `openloop_report`, but instead of an
        aggregate obs MSE it writes the RAW decoded observation vectors (so the
        caller can read individual components, e.g. CartPole's pole angle) plus
        the CONTINUE-head probability along both the open-loop (prior, imagined)
        and teacher-forced (posterior) rollouts. If the open-loop pole angle
        stops tracking the real fall and `ol_con` stays ≈1, imagination does not
        reproduce termination → the model-exploitation gap is a dynamics-fidelity
        problem. CPU-only.
        """
        comptime assert Self.train_target == "cpu", (
            "openloop_trace is CPU-only — run the diagnostic with a CPU agent"
        )
        comptime D = Self.DETER
        comptime SCl = Self.SC
        comptime TOK = Self.TOKEN
        comptime OBSD = Self.OBS
        comptime ACTD = Self.ACT
        comptime FEATl = Self.FEAT
        comptime CARRY = 2 + D + SCl

        var bd = Tensor.alloc(D)
        var bs = Tensor.alloc(SCl)
        var tok = Tensor.alloc(TOK)
        var pa = Tensor.alloc(ACTD)
        var obt = Tensor.alloc(OBSD)
        var dummyO = Tensor.alloc(OBSD)
        var dummy1 = Tensor.alloc(1)
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

        var old = Tensor.alloc(D); var ols = Tensor.alloc(SCl)
        var tfd = Tensor.alloc(D); var tfs = Tensor.alloc(SCl)
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
            for k in range(OBSD):
                out_ol_obs[h * OBSD + k] = _symexp(pred.data[k])
            self.con.set_input["nd", 1](ond_t, None)
            self.con.set_input["stoch_new", 1](osn_t, None)
            self.con.set_input["ctgt", 1](dummy1, None)
            self.con.forward[1, "cpu"](loss_t, None)
            ref ocon = self.con.node_output["con"]()
            out_ol_con[h] = Scalar[DT](1.0) / (Scalar[DT](1.0) + exp(-ocon.data[0]))
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
            for k in range(OBSD):
                out_tf_obs[h * OBSD + k] = _symexp(tpred.data[k])
            self.con.set_input["nd", 1](tnd_t, None)
            self.con.set_input["stoch_new", 1](tsn_t, None)
            self.con.set_input["ctgt", 1](dummy1, None)
            self.con.forward[1, "cpu"](loss_t, None)
            ref tcon = self.con.node_output["con"]()
            out_tf_con[h] = Scalar[DT](1.0) / (Scalar[DT](1.0) + exp(-tcon.data[0]))
            for k in range(D):
                tfd.data[k] = tnd_t.data[k]
            for k in range(SCl):
                tfs.data[k] = tsn_t.data[k]
