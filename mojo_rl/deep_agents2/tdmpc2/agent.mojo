"""TD-MPC2 agent (deep_agents2, CPU + GPU, MPC-off) — Pendulum lighthouse.

Single `target`-generic struct owning the world model (encoder, dynamics,
reward, Q ensemble online + target, policy) + their optimizers + the WM
ComputeGraph + the training blocks (WMStep BPTT, PolicyStep, TDTargetStep)
+ a SequenceReplay (host). `target` ("cpu"/"gpu") is comptime; `ctx` is
threaded for GPU.

Acting is MPC-off: `a = π(encode(obs))` (reference `cfg.mpc=False`). MPPI
planning is deferred to the GPU batched planner (P4+). See
docs/TDMPC2_DEEP_AGENTS2_PORT.md.

train_step: sample length-T window (host) → transpose to t-major → TD
targets (stop-grad) → WM BPTT → policy update on encoded latents → Polyak.
Replay stays host; GPU blocks upload/download internally (correctness-first;
a GPUSequenceReplay would remove the per-step transfers later).
"""

from std.memory import alloc
from std.math import tanh
from std.random import random_float64

from mojo_rl.nn2.constants import DT
from mojo_rl.nn2.initializer import Kaiming, Zero
from mojo_rl.nn2.optimizer.adam import Adam
from layout import Layout, LayoutTensor, TileTensor, row_major
from std.gpu.host import DeviceContext, DeviceBuffer

from mojo_rl.deep_agents2.primitives.rsample import RSample
from mojo_rl.deep_agents2.dreamerv3.polyak import polyak_module
from mojo_rl.deep_agents2.data.sequence_replay import SequenceReplay

from .nets import (
    TDMPC2Encoder, TDMPC2Dynamics, TDMPC2Reward, TDMPC2QNet, TDMPC2Policy,
)
from .wm_graph import TDMPC2WMGraph, NQ
from .wm_step import WMStep
from .policy_step import PolicyStep
from .td_target_step import TDTargetStep


@always_inline
def _alloc(n: Int) -> UnsafePointer[Scalar[DT], MutAnyOrigin]:
    return alloc[Scalar[DT]](n)


@always_inline
def _dp(b: DeviceBuffer[DT]) -> UnsafePointer[Scalar[DT], MutAnyOrigin]:
    return rebind[UnsafePointer[Scalar[DT], MutAnyOrigin]](b.unsafe_ptr())


def _upload(
    ctx: DeviceContext, src: UnsafePointer[Scalar[DT], MutAnyOrigin], n: Int
) raises -> DeviceBuffer[DT]:
    var d = ctx.enqueue_create_buffer[DT](n)
    var h = ctx.enqueue_create_host_buffer[DT](n)
    ctx.synchronize()
    for i in range(n):
        h.unsafe_ptr()[i] = src[i]
    ctx.enqueue_copy(d, h)
    ctx.synchronize()
    return d^


@fieldwise_init
struct TDMPC2Agent[
    target: StaticString,
    OBS: Int,
    ENC: Int,
    ACT: Int,
    LATENT: Int,
    MLP: Int,
    BINS: Int,
    SN: Int,
    VMIN: Int,
    VMAX: Int,
    B: Int,
    H: Int,
    CAP: Int,
](Movable & ImplicitlyDestructible):
    comptime EncT = TDMPC2Encoder[Self.OBS, Self.ENC, Self.LATENT, Self.SN]
    comptime DynT = TDMPC2Dynamics[Self.LATENT, Self.ACT, Self.MLP, Self.SN]
    comptime RewT = TDMPC2Reward[Self.LATENT, Self.ACT, Self.MLP, Self.BINS]
    comptime QNetT = TDMPC2QNet[Self.LATENT, Self.ACT, Self.MLP, Self.BINS]
    comptime PolicyT = TDMPC2Policy[Self.LATENT, Self.ACT, Self.MLP]
    comptime GraphT = TDMPC2WMGraph[
        Self.LATENT, Self.ACT, Self.MLP, Self.BINS, Self.SN, Self.VMIN, Self.VMAX
    ]
    comptime PB = (Self.H + 1) * Self.B
    comptime WMStepT = WMStep[
        Self.OBS, Self.ENC, Self.ACT, Self.LATENT, Self.MLP, Self.BINS,
        Self.SN, Self.VMIN, Self.VMAX, Self.B, Self.H,
    ]
    comptime PolStepT = PolicyStep[
        Self.LATENT, Self.ACT, Self.MLP, Self.BINS, Self.VMIN, Self.VMAX, Self.PB,
    ]
    comptime TDStepT = TDTargetStep[
        Self.OBS, Self.ENC, Self.ACT, Self.LATENT, Self.MLP, Self.BINS,
        Self.SN, Self.VMIN, Self.VMAX, Self.B, Self.H,
    ]

    var encoder: Self.EncT
    var dynamics: Self.DynT
    var reward: Self.RewT
    var q: List[Self.QNetT]
    var qt: List[Self.QNetT]
    var policy: Self.PolicyT

    var enc_opt: Adam
    var dyn_opt: Adam
    var rew_opt: Adam
    var q_opt: List[Adam]
    var pi_opt: Adam

    var wm_graph: Self.GraphT
    var wm_step: Self.WMStepT
    var pol_step: Self.PolStepT
    var td_step: Self.TDStepT
    var act_rsample: RSample[Self.ACT]
    var replay: SequenceReplay[Self.OBS, Self.ACT, Self.CAP]

    var gamma: Scalar[DT]
    var tau: Scalar[DT]
    var action_scale: Scalar[DT]
    var learning_starts: Int
    var step_count: Int
    var _last_wm: Scalar[DT]
    var _last_pi: Scalar[DT]
    var ctx: Optional[DeviceContext]

    @staticmethod
    def make(
        lr: Scalar[DT] = Scalar[DT](3e-4),
        gamma: Scalar[DT] = Scalar[DT](0.99),
        tau: Scalar[DT] = Scalar[DT](0.01),
        action_scale: Scalar[DT] = Scalar[DT](1.0),
        learning_starts: Int = 1000,
        enc_lr_scale: Scalar[DT] = Scalar[DT](0.3),
        ctx: Optional[DeviceContext] = None,
    ) raises -> Self:
        comptime tg = Self.target
        var enc = Self.EncT.make[tg, INIT=Kaiming](ctx=ctx)
        var dyn = Self.DynT.make[tg, INIT=Kaiming](ctx=ctx)
        var rew = Self.RewT.make[tg, INIT=Kaiming](ctx=ctx)
        var pol = Self.PolicyT.make[tg, INIT=Kaiming](ctx=ctx)

        var q = List[Self.QNetT]()
        var qt = List[Self.QNetT]()
        var q_opt = List[Adam]()
        for _ in range(NQ):
            var qn = Self.QNetT.make[tg, INIT=Kaiming](ctx=ctx)
            var qtn = Self.QNetT.make[tg, INIT=Kaiming](ctx=ctx)
            var qo = Adam.make[tg, Self.QNetT](qn, ctx=ctx)
            qo.lr = lr
            q.append(qn^)
            qt.append(qtn^)
            q_opt.append(qo^)
        for i in range(NQ):
            polyak_module[tg, Self.QNetT](q[i], qt[i], Scalar[DT](1.0), ctx=ctx)

        var enc_opt = Adam.make[tg, Self.EncT](enc, ctx=ctx)
        enc_opt.lr = lr * enc_lr_scale
        var dyn_opt = Adam.make[tg, Self.DynT](dyn, ctx=ctx)
        dyn_opt.lr = lr
        var rew_opt = Adam.make[tg, Self.RewT](rew, ctx=ctx)
        rew_opt.lr = lr
        var pi_opt = Adam.make[tg, Self.PolicyT](pol, ctx=ctx)
        pi_opt.lr = lr
        pi_opt.eps = Scalar[DT](1e-5)

        var ar = RSample[Self.ACT].make[tg, INIT=Zero](ctx=ctx)
        ar.action_scale = action_scale

        return Self(
            encoder=enc^, dynamics=dyn^, reward=rew^, q=q^, qt=qt^, policy=pol^,
            enc_opt=enc_opt^, dyn_opt=dyn_opt^, rew_opt=rew_opt^, q_opt=q_opt^,
            pi_opt=pi_opt^,
            wm_graph=Self.GraphT.make[tg, INIT=Kaiming](ctx=ctx),
            wm_step=Self.WMStepT.make[tg](ctx=ctx),
            pol_step=Self.PolStepT.make[tg](ctx=ctx),
            td_step=Self.TDStepT.make[tg](ctx=ctx),
            act_rsample=ar^,
            replay=SequenceReplay[Self.OBS, Self.ACT, Self.CAP].new(),
            gamma=gamma, tau=tau, action_scale=action_scale,
            learning_starts=learning_starts, step_count=0,
            _last_wm=Scalar[DT](0.0), _last_pi=Scalar[DT](0.0), ctx=ctx,
        )

    # ── acting (MPC-off): a = π(encode(obs)) ───────────────────────────
    def select_action(
        mut self,
        obs: UnsafePointer[Scalar[DT], MutAnyOrigin],
        act_out: UnsafePointer[Scalar[DT], MutAnyOrigin],
        explore: Bool = True,
    ) raises:
        comptime tg = Self.target
        comptime A = Self.ACT
        comptime LAT = Self.LATENT
        comptime if tg == "cpu":
            var z = _alloc(LAT)
            var z_t = TileTensor(z, row_major[1, LAT]())
            self.encoder.forward[tg, 1](
                TileTensor(obs, row_major[1, Self.OBS]()), output=z_t,
            )
            var pio = _alloc(2 * A)
            var pio_t = TileTensor(pio, row_major[1, 2 * A]())
            self.policy.forward[tg, 1](z_t, output=pio_t)
            if explore:
                var alp = _alloc(A + 1)
                var alp_t = TileTensor(alp, row_major[1, A + 1]())
                self.act_rsample.forward[tg, 1](pio_t, output=alp_t)
                for j in range(A):
                    act_out[j] = alp[j]
                alp.free()
            else:
                for j in range(A):
                    act_out[j] = tanh(pio[j]) * self.action_scale
            z.free(); pio.free()
        else:
            var ctx = self.ctx.value()
            var d_obs = _upload(ctx, obs, Self.OBS)
            var d_z = ctx.enqueue_create_buffer[DT](LAT)
            var z_t = TileTensor(_dp(d_z), row_major[1, LAT]())
            self.encoder.forward[tg, 1](
                TileTensor(_dp(d_obs), row_major[1, Self.OBS]()), output=z_t,
            )
            var d_pio = ctx.enqueue_create_buffer[DT](2 * A)
            var pio_t = TileTensor(_dp(d_pio), row_major[1, 2 * A]())
            self.policy.forward[tg, 1](z_t, output=pio_t)
            if explore:
                var d_alp = ctx.enqueue_create_buffer[DT](A + 1)
                var alp_t = TileTensor(_dp(d_alp), row_major[1, A + 1]())
                self.act_rsample.forward[tg, 1](pio_t, output=alp_t)
                var h = ctx.enqueue_create_host_buffer[DT](A + 1)
                ctx.enqueue_copy(h, d_alp)
                ctx.synchronize()
                for j in range(A):
                    act_out[j] = h.unsafe_ptr()[j]
            else:
                var h = ctx.enqueue_create_host_buffer[DT](2 * A)
                ctx.enqueue_copy(h, d_pio)
                ctx.synchronize()
                for j in range(A):
                    act_out[j] = tanh(h.unsafe_ptr()[j]) * self.action_scale

    def select_greedy_action(
        mut self,
        obs: UnsafePointer[Scalar[DT], MutAnyOrigin],
        act_out: UnsafePointer[Scalar[DT], MutAnyOrigin],
    ) raises:
        self.select_action(obs, act_out, explore=False)

    def record(
        mut self,
        obs: UnsafePointer[Scalar[DT], MutAnyOrigin],
        act: UnsafePointer[Scalar[DT], MutAnyOrigin],
        reward: Scalar[DT],
        done: Scalar[DT],
    ):
        self.replay.record(obs, act, reward, done)

    def last_wm_loss(self) -> Scalar[DT]:
        return self._last_wm

    def last_pi_loss(self) -> Scalar[DT]:
        return self._last_pi

    def train_step(mut self) raises -> Bool:
        self.step_count += 1
        if not self.replay.can_sample[Self.H]():
            return False
        if self.replay.count() < self.learning_starts:
            return False

        comptime tg = Self.target
        comptime OBSD = Self.OBS
        comptime ACTD = Self.ACT
        comptime LAT = Self.LATENT
        comptime HH = Self.H
        comptime BB = Self.B

        # ── sample (b-major) + transpose to t-major (host) ─────────────
        var ob = _alloc(BB * (HH + 1) * OBSD)
        var ab = _alloc(BB * HH * ACTD)
        var rb = _alloc(BB * HH)
        var db = _alloc(BB * HH)
        self.replay.sample_batch[BB, HH](ob, ab, rb, db)

        var ot = _alloc((HH + 1) * BB * OBSD)
        var at = _alloc(HH * BB * ACTD)
        var rt = _alloc(HH * BB)
        var dt = _alloc(HH * BB)
        for b in range(BB):
            for t in range(HH + 1):
                for i in range(OBSD):
                    ot[(t * BB + b) * OBSD + i] = ob[
                        (b * (HH + 1) + t) * OBSD + i
                    ]
            for t in range(HH):
                for j in range(ACTD):
                    at[(t * BB + b) * ACTD + j] = ab[(b * HH + t) * ACTD + j]
                rt[t * BB + b] = rb[b * HH + t]
                dt[t * BB + b] = db[b * HH + t]

        var td = _alloc(HH * BB)
        var ta = Int(random_float64() * Float64(NQ))
        if ta >= NQ:
            ta = NQ - 1
        var tb = (ta + 1) % NQ
        var pa = Int(random_float64() * Float64(NQ))
        if pa >= NQ:
            pa = NQ - 1
        var pb = (pa + 1) % NQ

        comptime if tg == "cpu":
            self.td_step.step[tg](
                self.encoder, self.policy, self.qt, ta, tb,
                ot, rt, dt, td, self.gamma,
            )
            self._last_wm = self.wm_step.step[tg](
                self.wm_graph, self.encoder, self.dynamics, self.reward, self.q,
                self.enc_opt, self.dyn_opt, self.rew_opt, self.q_opt,
                ot, at, rt, td,
            )
            var zpol = _alloc(Self.PB * LAT)
            var zpol_t = TileTensor(zpol, row_major[Self.PB, LAT]())
            self.encoder.forward[tg, Self.PB](
                TileTensor(ot, row_major[Self.PB, OBSD]()), output=zpol_t,
            )
            self._last_pi = self.pol_step.step[tg](
                self.policy, self.q, pa, pb, self.pi_opt, zpol,
            )
            zpol.free()
            for i in range(NQ):
                polyak_module[tg, Self.QNetT](
                    self.q[i], self.qt[i], self.tau
                )
        else:
            var ctx = self.ctx.value()
            self.td_step.step[tg](
                self.encoder, self.policy, self.qt, ta, tb,
                ot, rt, dt, td, self.gamma, ctx=ctx,
            )
            self._last_wm = self.wm_step.step[tg](
                self.wm_graph, self.encoder, self.dynamics, self.reward, self.q,
                self.enc_opt, self.dyn_opt, self.rew_opt, self.q_opt,
                ot, at, rt, td, ctx=ctx,
            )
            var d_ot = _upload(ctx, ot, Self.PB * OBSD)
            var d_zpol = ctx.enqueue_create_buffer[DT](Self.PB * LAT)
            var zpol_t = TileTensor(_dp(d_zpol), row_major[Self.PB, LAT]())
            self.encoder.forward[tg, Self.PB](
                TileTensor(_dp(d_ot), row_major[Self.PB, OBSD]()), output=zpol_t,
            )
            self._last_pi = self.pol_step.step[tg](
                self.policy, self.q, pa, pb, self.pi_opt, _dp(d_zpol), ctx=ctx,
            )
            for i in range(NQ):
                polyak_module[tg, Self.QNetT](
                    self.q[i], self.qt[i], self.tau, ctx=ctx
                )

        ob.free(); ab.free(); rb.free(); db.free()
        ot.free(); at.free(); rt.free(); dt.free()
        td.free()
        return True
