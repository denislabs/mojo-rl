"""TD-MPC2 agent (deep_agents2, CPU, MPC-off) — Pendulum lighthouse.

Single struct owning the world model (encoder, dynamics, reward, Q ensemble
online + target, policy) + their optimizers + the WM ComputeGraph + the
training blocks (WMStep BPTT, PolicyStep, TDTargetStep) + a SequenceReplay.

Acting is MPC-off: `a = π(encode(obs))` (reference `cfg.mpc=False`). MPPI
planning is deferred to P3.5/P4 (the per-sample CPU planner is too slow for
a training run; the GPU batched planner is the production path). See
docs/TDMPC2_DEEP_AGENTS2_PORT.md.

train_step: sample length-T sequence → transpose to t-major → TD targets
(stop-grad) → WM BPTT update → policy update on encoded latents → Polyak
target-Q. Mirrors `DreamerV3Agent`'s make/select_action/record/train_step
surface for the existing Pendulum smoke harness.
"""

from std.memory import alloc
from std.math import tanh
from std.random import random_float64

from mojo_rl.nn2.constants import DT
from mojo_rl.nn2.initializer import Kaiming, Zero
from mojo_rl.nn2.optimizer.adam import Adam
from layout import TileTensor, row_major

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


@fieldwise_init
struct TDMPC2Agent[
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
    comptime PB = (Self.H + 1) * Self.B   # policy batch = all latents in window
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

    # ── world-model modules ────────────────────────────────────────────
    var encoder: Self.EncT
    var dynamics: Self.DynT
    var reward: Self.RewT
    var q: List[Self.QNetT]
    var qt: List[Self.QNetT]
    var policy: Self.PolicyT

    # ── optimizers ─────────────────────────────────────────────────────
    var enc_opt: Adam
    var dyn_opt: Adam
    var rew_opt: Adam
    var q_opt: List[Adam]
    var pi_opt: Adam

    # ── graph + blocks ─────────────────────────────────────────────────
    var wm_graph: Self.GraphT
    var wm_step: Self.WMStepT
    var pol_step: Self.PolStepT
    var td_step: Self.TDStepT
    var act_rsample: RSample[Self.ACT]
    var replay: SequenceReplay[Self.OBS, Self.ACT, Self.CAP]

    # ── hyperparams + diagnostics ──────────────────────────────────────
    var gamma: Scalar[DT]
    var tau: Scalar[DT]
    var action_scale: Scalar[DT]
    var learning_starts: Int
    var step_count: Int
    var _last_wm: Scalar[DT]
    var _last_pi: Scalar[DT]
    var _pair: Int

    @staticmethod
    def make(
        lr: Scalar[DT] = Scalar[DT](3e-4),
        gamma: Scalar[DT] = Scalar[DT](0.99),
        tau: Scalar[DT] = Scalar[DT](0.01),
        action_scale: Scalar[DT] = Scalar[DT](1.0),
        learning_starts: Int = 1000,
        enc_lr_scale: Scalar[DT] = Scalar[DT](0.3),
    ) raises -> Self:
        var enc = Self.EncT.make["cpu", INIT=Kaiming]()
        var dyn = Self.DynT.make["cpu", INIT=Kaiming]()
        var rew = Self.RewT.make["cpu", INIT=Kaiming]()
        var pol = Self.PolicyT.make["cpu", INIT=Kaiming]()

        var q = List[Self.QNetT]()
        var qt = List[Self.QNetT]()
        var q_opt = List[Adam]()
        for _ in range(NQ):
            var qn = Self.QNetT.make["cpu", INIT=Kaiming]()
            var qtn = Self.QNetT.make["cpu", INIT=Kaiming]()
            var qo = Adam.make["cpu", Self.QNetT](qn)
            qo.lr = lr
            q.append(qn^)
            qt.append(qtn^)
            q_opt.append(qo^)
        # hard-copy online → target (rate=1.0).
        for i in range(NQ):
            polyak_module["cpu", Self.QNetT](q[i], qt[i], Scalar[DT](1.0))

        var enc_opt = Adam.make["cpu", Self.EncT](enc)
        enc_opt.lr = lr * enc_lr_scale
        var dyn_opt = Adam.make["cpu", Self.DynT](dyn)
        dyn_opt.lr = lr
        var rew_opt = Adam.make["cpu", Self.RewT](rew)
        rew_opt.lr = lr
        var pi_opt = Adam.make["cpu", Self.PolicyT](pol)
        pi_opt.lr = lr
        pi_opt.eps = Scalar[DT](1e-5)   # reference pi_optim eps

        var ar = RSample[Self.ACT].make["cpu", INIT=Zero]()
        ar.action_scale = action_scale

        return Self(
            encoder=enc^, dynamics=dyn^, reward=rew^, q=q^, qt=qt^, policy=pol^,
            enc_opt=enc_opt^, dyn_opt=dyn_opt^, rew_opt=rew_opt^, q_opt=q_opt^,
            pi_opt=pi_opt^,
            wm_graph=Self.GraphT.make["cpu", INIT=Kaiming](),
            wm_step=Self.WMStepT.make["cpu"](),
            pol_step=Self.PolStepT.make["cpu"](),
            td_step=Self.TDStepT.make["cpu"](),
            act_rsample=ar^,
            replay=SequenceReplay[Self.OBS, Self.ACT, Self.CAP].new(),
            gamma=gamma, tau=tau, action_scale=action_scale,
            learning_starts=learning_starts, step_count=0,
            _last_wm=Scalar[DT](0.0), _last_pi=Scalar[DT](0.0), _pair=0,
        )

    # ── acting (MPC-off): a = π(encode(obs)) ───────────────────────────
    def _encode_one(
        mut self, obs: UnsafePointer[Scalar[DT], MutAnyOrigin],
        z: UnsafePointer[Scalar[DT], MutAnyOrigin],
    ) raises:
        var z_t = TileTensor(z, row_major[1, Self.LATENT]())
        self.encoder.forward["cpu", 1](
            TileTensor(obs, row_major[1, Self.OBS]()), output=z_t,
        )

    def select_action(
        mut self,
        obs: UnsafePointer[Scalar[DT], MutAnyOrigin],
        act_out: UnsafePointer[Scalar[DT], MutAnyOrigin],
        explore: Bool = True,
    ) raises:
        var z = _alloc(Self.LATENT)
        self._encode_one(obs, z)
        var pio = _alloc(2 * Self.ACT)
        var pio_t = TileTensor(pio, row_major[1, 2 * Self.ACT]())
        self.policy.forward["cpu", 1](
            TileTensor(z, row_major[1, Self.LATENT]()), output=pio_t,
        )
        if explore:
            var alp = _alloc(Self.ACT + 1)
            var alp_t = TileTensor(alp, row_major[1, Self.ACT + 1]())
            self.act_rsample.forward["cpu", 1](pio_t, output=alp_t)
            for j in range(Self.ACT):
                act_out[j] = alp[j]
            alp.free()
        else:
            for j in range(Self.ACT):
                var m = tanh(pio[j]) * self.action_scale
                act_out[j] = m
        z.free(); pio.free()

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

        comptime OBSD = Self.OBS
        comptime ACTD = Self.ACT
        comptime LAT = Self.LATENT
        comptime HH = Self.H
        comptime BB = Self.B

        # ── sample (b-major) ───────────────────────────────────────────
        var ob = _alloc(BB * (HH + 1) * OBSD)
        var ab = _alloc(BB * HH * ACTD)
        var rb = _alloc(BB * HH)
        var db = _alloc(BB * HH)
        self.replay.sample_batch[BB, HH](ob, ab, rb, db)

        # ── transpose to t-major (each step's slice contiguous) ────────
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

        # ── TD targets (stop-grad) — min of 2 random target-Q ──────────
        var td = _alloc(HH * BB)
        var ta = Int(random_float64() * Float64(NQ))
        if ta >= NQ:
            ta = NQ - 1
        var tb = (ta + 1) % NQ
        self.td_step.step["cpu"](
            self.encoder, self.policy, self.qt, ta, tb,
            ot, rt, dt, td, self.gamma,
        )

        # ── WM BPTT update ─────────────────────────────────────────────
        var wm_loss = self.wm_step.step["cpu"](
            self.wm_graph, self.encoder, self.dynamics, self.reward, self.q,
            self.enc_opt, self.dyn_opt, self.rew_opt, self.q_opt,
            ot, at, rt, td,
        )
        self._last_wm = wm_loss

        # ── policy update on encoded latents z = encode(obs[t]) ────────
        var zpol = _alloc(Self.PB * LAT)
        var zpol_t = TileTensor(zpol, row_major[Self.PB, LAT]())
        self.encoder.forward["cpu", Self.PB](
            TileTensor(ot, row_major[Self.PB, OBSD]()), output=zpol_t,
        )
        var pa = Int(random_float64() * Float64(NQ))
        if pa >= NQ:
            pa = NQ - 1
        var pb = (pa + 1) % NQ
        var pi_loss = self.pol_step.step["cpu"](
            self.policy, self.q, pa, pb, self.pi_opt, zpol,
        )
        self._last_pi = pi_loss

        # ── Polyak target-Q ────────────────────────────────────────────
        for i in range(NQ):
            polyak_module["cpu", Self.QNetT](self.q[i], self.qt[i], self.tau)

        ob.free(); ab.free(); rb.free(); db.free()
        ot.free(); at.free(); rt.free(); dt.free()
        td.free(); zpol.free()
        return True
