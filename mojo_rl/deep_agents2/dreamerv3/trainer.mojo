"""DreamerV3Trainer — SAC-style block-composed trainer (CPU; GPU-ready).

Composes the `blocks.mojo` units (`WMStep` / `ParamSyncStep` / `ACStep`)
over a shared `DreamerState`, mirroring `deep_agents2/sac/trainer.mojo`:
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
from std.gpu.host import DeviceContext

from mojo_rl.nn2.constants import DT
from mojo_rl.nn2.initializer import Kaiming
from mojo_rl.nn2.primitives.ops.swish_op import SwishOp
from mojo_rl.nn2.optimizer.dreamer_opt import DreamerOpt
from mojo_rl.deep_agents2.data.sequence_replay import SequenceReplay
from mojo_rl.deep_agents2.dreamerv3.wm import (
    WMCoreGraph, WMImagineGraph, DecLossGraph, RewLossGraph, ConLossGraph,
)
from mojo_rl.deep_agents2.dreamerv3.nets import (
    DreamerEncoder, DreamerValue, DreamerPolicy,
)
from mojo_rl.deep_agents2.dreamerv3.twohot import symexp_twohot_bins
from mojo_rl.deep_agents2.dreamerv3.normalize import PercentileNormalize
from mojo_rl.deep_agents2.dreamerv3.blocks import (
    DreamerState, WMStep, ParamSyncStep, ACStep,
)


@fieldwise_init
struct DreamerV3Trainer[
    train_target: StaticString,
    OBS: Int, ACT: Int, DETER: Int, H: Int, STOCH: Int, CLASSES: Int,
    BLOCKS: Int, TOKEN: Int, DEC_U: Int, HU: Int, VU: Int, PU: Int,
    BINS: Int, B: Int, T: Int, T_IMAG: Int, CAP: Int,
](Movable & ImplicitlyDestructible):
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
    comptime PolT = DreamerPolicy[Self.FEAT, Self.PU, Self.ACT, SwishOp]
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
        Self.T, Self.T_IMAG,
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

    @staticmethod
    def make(
        ctx: Optional[DeviceContext] = None,
        lr: Scalar[DT] = Scalar[DT](4e-5),
        learning_starts: Int = 200,
    ) raises -> Self:
        comptime assert Self.train_target == "cpu", (
            "DreamerV3Trainer: GPU lands in PR5c Step 5 (custom ops CPU-only);"
            " composition is already GPU-shaped via train_target/ctx."
        )
        var enc = Self.EncT.make[Self.train_target, INIT=Kaiming](ctx=ctx)
        var core = Self.CoreT.make[Self.train_target, INIT=Kaiming](ctx=ctx)
        var dec = Self.DecT.make[Self.train_target, INIT=Kaiming](ctx=ctx)
        var rew = Self.RewT.make[Self.train_target, INIT=Kaiming](ctx=ctx)
        var con = Self.ConT.make[Self.train_target, INIT=Kaiming](ctx=ctx)
        var value = Self.ValT.make[Self.train_target, INIT=Kaiming](ctx=ctx)
        var slowvalue = Self.ValT.make[Self.train_target, INIT=Kaiming](ctx=ctx)
        var policy = Self.PolT.make[Self.train_target, INIT=Kaiming](ctx=ctx)
        var imagine = Self.ImagT.make[Self.train_target, INIT=Kaiming](ctx=ctx)

        var oe = DreamerOpt.make[Self.train_target, Self.EncT](enc, ctx=ctx)
        var ocore = DreamerOpt.make_graph[Self.train_target](core, ctx=ctx)
        var odec = DreamerOpt.make_graph[Self.train_target](dec, ctx=ctx)
        var orew = DreamerOpt.make_graph[Self.train_target](rew, ctx=ctx)
        var ocon = DreamerOpt.make_graph[Self.train_target](con, ctx=ctx)
        var oval = DreamerOpt.make[Self.train_target, Self.ValT](value, ctx=ctx)
        var opol = DreamerOpt.make[Self.train_target, Self.PolT](policy, ctx=ctx)
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
            wm_blk=Self.WMBlk.make[Self.train_target](ctx=ctx),
            sync_blk=Self.SyncBlk.make[Self.train_target](ctx=ctx),
            ac_blk=Self.ACBlk.make[Self.train_target](ctx=ctx),
            replay=Self.RepT.make[Self.train_target](ctx=ctx),
            retnorm=retnorm^,
            bins=bins^,
            state=Self.StateT.make[Self.train_target](ctx=ctx),
            ctx=ctx,
            learning_starts=learning_starts,
            train_steps=0,
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

    def train_step(mut self) raises -> Bool:
        if not self.can_train():
            return False
        # sample a length-T window into the shared state batch buffers
        self.replay.sample_batch[Self.B, Self.T](
            self.state.mb_obs, self.state.mb_act, self.state.mb_rew,
            self.state.mb_dne,
        )
        for i in range(Self.T_IMAG * Self.B * Self.ACT):
            self.state.noise[i] = Scalar[DT](random_float64() * 2.0 - 1.0)
        # WM-BPTT → fills state.cdeter / cstoch + state.last_wm_loss
        self.wm_blk.step[Self.train_target, Self.T_IMAG](
            self.state, self.enc, self.core, self.dec, self.rew, self.con,
            self.oe, self.ocore, self.odec, self.orew, self.ocon,
        )
        # core/prior → imagine mirror
        self.sync_blk.step[Self.train_target](self.core, self.imagine)
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
