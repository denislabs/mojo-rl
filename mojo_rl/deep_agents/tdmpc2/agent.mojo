"""TD-MPC2 agent (deep_agents, storage framework, CPU + GPU, MPC-off).

Single `target`-generic struct owning the world model (encoder, dynamics,
reward, Q ensemble online + target, policy) + their optimizers + the WM
ComputeGraph + the training blocks (WMStep BPTT, PolicyStep, TDTargetStep)
+ a SequenceReplay (host). `target` ("cpu"/"gpu") is comptime; `ctx` is
threaded for GPU.

Storage migration (Stage 5): the 5 online Q heads, 5 target Q heads, and 5
Q optimizers are DISTINCT FIELDS (q0..q4 / qt0..qt4 / qo0..qo4; NQ fixed = 5).
Storage threads externals into ONE forward/vjp call (two `mut` subscripts of
one List can't alias). The WM step threads all 5 online Q as distinct args;
the random PAIR steps (policy: online (pa,pb); td: target (ta,tb)) use a
comptime-unrolled guarded dispatch so two DISTINCT fields are threaded.

Acting is MPC-off: `a = π(encode(obs))` (reference `cfg.mpc=False`).

train_step: sample length-T window (host) → transpose to t-major → TD
targets (stop-grad) → WM BPTT → policy update on encoded latents → Polyak.
Replay stays host; the steps upload/download internally via storage Tensors.
"""

from std.math import tanh
from std.random import random_float64
from std.time import perf_counter_ns
from layout import Layout, LayoutTensor
from std.gpu.host import DeviceContext, DeviceBuffer

from mojo_rl.nn.constants import DT
from mojo_rl.nn.core.tensor import Tensor
from mojo_rl.nn.core.tensor_refs import TensorRefs
from mojo_rl.nn.core.initializer import Kaiming, Zero
from mojo_rl.nn.optimizer.adam import Adam
from mojo_rl.nn.primitives.rsample import RSample
from mojo_rl.nn.core.checkpoint import (
    CheckpointWriter, CheckpointReader, _split_lines,
)

from mojo_rl.deep_agents.data.sequence_replay import SequenceReplay
from mojo_rl.planners.trajectory.mppi import MPPIGPUBatched
from mojo_rl.core.logger import Logger, NoOpLogger
from mojo_rl.core.env_traits import BoxContinuousActionEnv
from .callback import TDMPC2RolloutCallbackGPU
from .metrics import TDMPC2Metrics

from .nets import (
    TDMPC2Encoder, TDMPC2Dynamics, TDMPC2Reward, TDMPC2QNet, TDMPC2Policy,
    TDMPC2Termination,
)
from .wm_graph import TDMPC2WMGraph, NQ
from .wm_step import WMStep
from .policy_step import PolicyStep
from .td_target_step import TDTargetStep


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
    NUM_SAMPLES: Int = 512,
    NUM_PI_TRAJS: Int = 24,
    NUM_ELITES: Int = 64,
    NUM_ITERS: Int = 6,
    QP: Float64 = 0.0,
](Movable & ImplicitlyDeletable):
    comptime EncT = TDMPC2Encoder[Self.OBS, Self.ENC, Self.LATENT, Self.SN]
    comptime DynT = TDMPC2Dynamics[Self.LATENT, Self.ACT, Self.MLP, Self.SN]
    comptime RewT = TDMPC2Reward[Self.LATENT, Self.ACT, Self.MLP, Self.BINS]
    comptime QNetT = TDMPC2QNet[Self.LATENT, Self.ACT, Self.MLP, Self.BINS, Self.QP]
    comptime TermT = TDMPC2Termination[Self.LATENT, Self.ACT, Self.MLP]
    comptime PolicyT = TDMPC2Policy[Self.LATENT, Self.ACT, Self.MLP]
    comptime GraphT = TDMPC2WMGraph[
        Self.LATENT, Self.ACT, Self.MLP, Self.BINS, Self.SN, Self.VMIN,
        Self.VMAX, Self.QP,
    ]
    comptime PB = (Self.H + 1) * Self.B
    comptime WMStepT = WMStep[
        Self.OBS, Self.ENC, Self.ACT, Self.LATENT, Self.MLP, Self.BINS,
        Self.SN, Self.VMIN, Self.VMAX, Self.B, Self.H, Self.QP,
    ]
    comptime PolStepT = PolicyStep[
        Self.LATENT, Self.ACT, Self.MLP, Self.BINS, Self.VMIN, Self.VMAX,
        Self.PB, Self.QP,
    ]
    comptime TDStepT = TDTargetStep[
        Self.OBS, Self.ENC, Self.ACT, Self.LATENT, Self.MLP, Self.BINS,
        Self.SN, Self.VMIN, Self.VMAX, Self.B, Self.H, Self.QP,
    ]
    comptime MPC_BT = Self.NUM_SAMPLES + Self.NUM_PI_TRAJS
    comptime PlannerT = MPPIGPUBatched[
        Self.LATENT, Self.ACT, Self.H, Self.NUM_SAMPLES, Self.NUM_PI_TRAJS,
        Self.NUM_ELITES, Self.NUM_ITERS, 1,
    ]
    comptime MpcCB = TDMPC2RolloutCallbackGPU[
        Self.ACT, Self.LATENT, Self.MLP, Self.BINS, Self.SN, Self.VMIN,
        Self.VMAX, NQ, Self.MPC_BT, Self.QP,
    ]

    var encoder: Self.EncT
    var dynamics: Self.DynT
    var reward: Self.RewT
    # 5 online Q heads (distinct fields; threaded as externals).
    var q0: Self.QNetT
    var q1: Self.QNetT
    var q2: Self.QNetT
    var q3: Self.QNetT
    var q4: Self.QNetT
    # 5 target Q heads.
    var qt0: Self.QNetT
    var qt1: Self.QNetT
    var qt2: Self.QNetT
    var qt3: Self.QNetT
    var qt4: Self.QNetT
    var policy: Self.PolicyT
    var termination: Self.TermT

    var enc_opt: Adam
    var dyn_opt: Adam
    var rew_opt: Adam
    # 5 Q optimizers.
    var qo0: Adam
    var qo1: Adam
    var qo2: Adam
    var qo3: Adam
    var qo4: Adam
    var pi_opt: Adam
    var term_opt: Adam

    var wm_graph: Self.GraphT
    var wm_step: Self.WMStepT
    var pol_step: Self.PolStepT
    var td_step: Self.TDStepT
    var act_rsample: RSample[Self.ACT]
    var replay: SequenceReplay[Self.OBS, Self.ACT, Self.CAP]

    var gamma: Scalar[DT]
    var tau: Scalar[DT]
    var bce_coef: Scalar[DT]
    var action_scale: Scalar[DT]
    var learning_starts: Int
    var step_count: Int
    var _last_wm: Scalar[DT]
    var _last_pi: Scalar[DT]
    var _last_cons: Scalar[DT]
    var _last_rew: Scalar[DT]
    var _last_val: Scalar[DT]
    var _last_term: Scalar[DT]
    var _cons_acc: Scalar[DT]
    var _rew_acc: Scalar[DT]
    var _val_acc: Scalar[DT]
    var _term_acc: Scalar[DT]
    var _pi_acc: Scalar[DT]
    var _q_mean_acc: Scalar[DT]
    var _q_min_last: Scalar[DT]
    var _q_max_last: Scalar[DT]
    var _td_mean_acc: Scalar[DT]
    var _td_min_last: Scalar[DT]
    var _td_max_last: Scalar[DT]
    var _n_diag: Int
    var ctx: Optional[DeviceContext]
    var planner: Optional[Self.PlannerT]
    var temperature: Scalar[DT]

    # ── comptime accessors: distinct online / target Q field by index ──────
    def get_q[i: Int](mut self) -> ref[MutAnyOrigin] Self.QNetT:
        comptime if i == 0:
            return self.q0
        elif i == 1:
            return self.q1
        elif i == 2:
            return self.q2
        elif i == 3:
            return self.q3
        else:
            return self.q4

    def get_qt[i: Int](mut self) -> ref[MutAnyOrigin] Self.QNetT:
        comptime if i == 0:
            return self.qt0
        elif i == 1:
            return self.qt1
        elif i == 2:
            return self.qt2
        elif i == 3:
            return self.qt3
        else:
            return self.qt4

    @staticmethod
    def make(
        lr: Scalar[DT] = Scalar[DT](3e-4),
        gamma: Scalar[DT] = Scalar[DT](0.99),
        tau: Scalar[DT] = Scalar[DT](0.01),
        action_scale: Scalar[DT] = Scalar[DT](1.0),
        learning_starts: Int = 1000,
        enc_lr_scale: Scalar[DT] = Scalar[DT](0.3),
        temperature: Scalar[DT] = Scalar[DT](0.5),
        bce_coef: Scalar[DT] = Scalar[DT](0.0),
        ctx: Optional[DeviceContext] = None,
    ) raises -> Self:
        comptime tg = Self.target
        var enc = Self.EncT.make[tg, INIT=Kaiming](ctx=ctx)
        var dyn = Self.DynT.make[tg, INIT=Kaiming](ctx=ctx)
        var rew = Self.RewT.make[tg, INIT=Kaiming](ctx=ctx)
        var pol = Self.PolicyT.make[tg, INIT=Kaiming](ctx=ctx)

        var q0 = Self.QNetT.make[tg, INIT=Kaiming](ctx=ctx)
        var q1 = Self.QNetT.make[tg, INIT=Kaiming](ctx=ctx)
        var q2 = Self.QNetT.make[tg, INIT=Kaiming](ctx=ctx)
        var q3 = Self.QNetT.make[tg, INIT=Kaiming](ctx=ctx)
        var q4 = Self.QNetT.make[tg, INIT=Kaiming](ctx=ctx)
        var qt0 = Self.QNetT.make[tg, INIT=Kaiming](ctx=ctx)
        var qt1 = Self.QNetT.make[tg, INIT=Kaiming](ctx=ctx)
        var qt2 = Self.QNetT.make[tg, INIT=Kaiming](ctx=ctx)
        var qt3 = Self.QNetT.make[tg, INIT=Kaiming](ctx=ctx)
        var qt4 = Self.QNetT.make[tg, INIT=Kaiming](ctx=ctx)

        var qo0 = Adam(lr=lr)
        var qo1 = Adam(lr=lr)
        var qo2 = Adam(lr=lr)
        var qo3 = Adam(lr=lr)
        var qo4 = Adam(lr=lr)
        comptime if tg == "gpu":
            qo0.adopt[tg, Self.QNetT](q0, ctx)
            qo1.adopt[tg, Self.QNetT](q1, ctx)
            qo2.adopt[tg, Self.QNetT](q2, ctx)
            qo3.adopt[tg, Self.QNetT](q3, ctx)
            qo4.adopt[tg, Self.QNetT](q4, ctx)

        # hard-copy online → target (tau = 1.0).
        qt0.polyak_from[tg](q0, Scalar[DT](1.0), ctx)
        qt1.polyak_from[tg](q1, Scalar[DT](1.0), ctx)
        qt2.polyak_from[tg](q2, Scalar[DT](1.0), ctx)
        qt3.polyak_from[tg](q3, Scalar[DT](1.0), ctx)
        qt4.polyak_from[tg](q4, Scalar[DT](1.0), ctx)

        # Q-dropout (item D): target Q nets eval (no masking) when QP>0.
        comptime if Self.QP > 0.0:
            qt0.set_attr["training"](Scalar[DT](0.0))
            qt1.set_attr["training"](Scalar[DT](0.0))
            qt2.set_attr["training"](Scalar[DT](0.0))
            qt3.set_attr["training"](Scalar[DT](0.0))
            qt4.set_attr["training"](Scalar[DT](0.0))

        # Termination head (item B): Zero-init at bce_coef=0 (no RNG draw → other
        # nets bit-identical), Kaiming when episodic. Built last.
        var term: Self.TermT
        if bce_coef > Scalar[DT](0.0):
            term = Self.TermT.make[tg, INIT=Kaiming](ctx=ctx)
        else:
            term = Self.TermT.make[tg, INIT=Zero](ctx=ctx)

        var enc_opt = Adam(lr=lr * enc_lr_scale)
        var dyn_opt = Adam(lr=lr)
        var rew_opt = Adam(lr=lr)
        var pi_opt = Adam(lr=lr)
        pi_opt.eps = Scalar[DT](1e-5)
        var term_opt = Adam(lr=lr)
        comptime if tg == "gpu":
            enc_opt.adopt[tg, Self.EncT](enc, ctx)
            dyn_opt.adopt[tg, Self.DynT](dyn, ctx)
            rew_opt.adopt[tg, Self.RewT](rew, ctx)
            pi_opt.adopt[tg, Self.PolicyT](pol, ctx)
            term_opt.adopt[tg, Self.TermT](term, ctx)

        var ar = RSample[Self.ACT].make[tg, INIT=Zero](ctx=ctx)
        ar.action_scale = action_scale

        var planner: Optional[Self.PlannerT] = None
        comptime if tg == "gpu":
            planner = Self.PlannerT(ctx.value())

        return Self(
            encoder=enc^, dynamics=dyn^, reward=rew^,
            q0=q0^, q1=q1^, q2=q2^, q3=q3^, q4=q4^,
            qt0=qt0^, qt1=qt1^, qt2=qt2^, qt3=qt3^, qt4=qt4^,
            policy=pol^, termination=term^,
            enc_opt=enc_opt^, dyn_opt=dyn_opt^, rew_opt=rew_opt^,
            qo0=qo0^, qo1=qo1^, qo2=qo2^, qo3=qo3^, qo4=qo4^,
            pi_opt=pi_opt^, term_opt=term_opt^,
            wm_graph=Self.GraphT.make[tg, INIT=Kaiming](ctx=ctx),
            wm_step=Self.WMStepT.make[tg](ctx=ctx, termination_coef=bce_coef),
            pol_step=Self.PolStepT.make[tg](ctx=ctx),
            td_step=Self.TDStepT.make[tg](ctx=ctx),
            act_rsample=ar^,
            replay=SequenceReplay[Self.OBS, Self.ACT, Self.CAP].new(),
            gamma=gamma, tau=tau, bce_coef=bce_coef, action_scale=action_scale,
            learning_starts=learning_starts, step_count=0,
            _last_wm=Scalar[DT](0.0), _last_pi=Scalar[DT](0.0),
            _last_cons=Scalar[DT](0.0), _last_rew=Scalar[DT](0.0),
            _last_val=Scalar[DT](0.0), _last_term=Scalar[DT](0.0),
            _cons_acc=Scalar[DT](0.0),
            _rew_acc=Scalar[DT](0.0), _val_acc=Scalar[DT](0.0),
            _term_acc=Scalar[DT](0.0),
            _pi_acc=Scalar[DT](0.0),
            _q_mean_acc=Scalar[DT](0.0), _q_min_last=Scalar[DT](0.0),
            _q_max_last=Scalar[DT](0.0), _td_mean_acc=Scalar[DT](0.0),
            _td_min_last=Scalar[DT](0.0), _td_max_last=Scalar[DT](0.0),
            _n_diag=0,
            ctx=ctx,
            planner=planner^, temperature=temperature,
        )

    # ── acting (MPC-off): a = π(encode(obs)) ───────────────────────────
    def select_action(
        mut self,
        ref obs: List[Scalar[DT]],
        mut act_out: List[Scalar[DT]],
        explore: Bool = True,
    ) raises:
        comptime tg = Self.target
        comptime A = Self.ACT
        comptime LAT = Self.LATENT
        var ctx = self.ctx
        # stage obs into a Tensor, upload on GPU.
        var ob = Tensor.alloc(Self.OBS)
        for d in range(Self.OBS):
            ob.data[d] = obs[d]
        comptime if tg == "gpu":
            ob.upload(ctx.value())
        var z = Tensor.make[tg](LAT, ctx)
        self.encoder.forward[tg, 1](TensorRefs[1](ob), z, ctx)
        var pio = Tensor.make[tg](2 * A, ctx)
        self.policy.forward[tg, 1](TensorRefs[1](z), pio, ctx)
        if explore:
            var alp = Tensor.make[tg](A + 1, ctx)
            self.act_rsample.forward[tg, 1](TensorRefs[1](pio), alp, ctx)
            comptime if tg == "gpu":
                alp.download(ctx.value())
            for j in range(A):
                act_out[j] = alp.data[j]
        else:
            comptime if tg == "gpu":
                pio.download(ctx.value())
            for j in range(A):
                act_out[j] = tanh(pio.data[j]) * self.action_scale

    def select_greedy_action(
        mut self,
        ref obs: List[Scalar[DT]],
        mut act_out: List[Scalar[DT]],
    ) raises:
        self.select_action(obs, act_out, explore=False)

    def mpc_start_episode(mut self) raises:
        comptime if Self.target == "gpu":
            self.planner.value().start_episode(0)

    def select_action_mpc(
        mut self,
        ref obs: List[Scalar[DT]],
        mut act_out: List[Scalar[DT]],
        explore: Bool = True,
    ) raises:
        """MPC acting: plan in latent space via MPPIGPUBatched (single env).
        GPU only."""
        comptime assert Self.target == "gpu", (
            "select_action_mpc requires target='gpu' (CPU MPPI is eval-only)"
        )
        comptime A = Self.ACT
        comptime LAT = Self.LATENT
        var ctx = self.ctx.value()

        var ob = Tensor.alloc(Self.OBS)
        for d in range(Self.OBS):
            ob.data[d] = obs[d]
        ob.upload(ctx)
        var z0 = Tensor.alloc_gpu(ctx, LAT)
        self.encoder.forward[Self.target, 1](
            TensorRefs[1](ob), z0, Optional(ctx)
        )

        # transient callback over self's modules (target Q for the bootstrap).
        var cb = Self.MpcCB.make(
            self.dynamics, self.reward, self.policy,
            self.qt0, self.qt1, self.qt2, self.qt3, self.qt4,
            self.action_scale, ctx,
        )

        var d_out = ctx.enqueue_create_buffer[DT](A)
        var z0_lt = z0.lt["gpu", Layout.row_major(1, LAT)]()
        var out_lt = LayoutTensor[DT, Layout.row_major(1 * A), MutAnyOrigin](
            d_out
        )
        self.planner.value().plan_gpu[Self.MpcCB](
            ctx, cb, z0_lt, out_lt,
            gamma=Float64(self.gamma),
            temperature=Float64(self.temperature),
            action_scale=Float64(self.action_scale),
            deterministic=not explore,
        )
        var h = ctx.enqueue_create_host_buffer[DT](A)
        ctx.enqueue_copy(h, d_out)
        ctx.synchronize()
        for j in range(A):
            act_out[j] = h.unsafe_ptr()[j]

    def record(
        mut self,
        ref obs: List[Scalar[DT]],
        ref act: List[Scalar[DT]],
        reward: Scalar[DT],
        done: Scalar[DT],
    ) raises:
        self.replay.record(
            rebind[UnsafePointer[Scalar[DT], MutAnyOrigin]](
                UnsafePointer(to=obs[0])
            ),
            rebind[UnsafePointer[Scalar[DT], MutAnyOrigin]](
                UnsafePointer(to=act[0])
            ),
            reward, done,
        )

    def last_wm_loss(self) -> Scalar[DT]:
        return self._last_wm

    def last_pi_loss(self) -> Scalar[DT]:
        return self._last_pi

    def last_consistency_loss(self) -> Scalar[DT]:
        return self._last_cons

    def last_reward_loss(self) -> Scalar[DT]:
        return self._last_rew

    def last_value_loss(self) -> Scalar[DT]:
        return self._last_val

    def last_termination_loss(self) -> Scalar[DT]:
        return self._last_term

    def pi_scale(self) -> Scalar[DT]:
        return self.pol_step.scale.value

    def flush_metrics[
        L: Logger
    ](
        mut self,
        logger: Optional[UnsafePointer[L, MutAnyOrigin]],
        step: Int,
    ) raises -> TDMPC2Metrics:
        var n = self._n_diag if self._n_diag > 0 else 1
        var inv = Scalar[DT](1.0) / Scalar[DT](n)
        var m = TDMPC2Metrics(
            consistency_loss=self._cons_acc * inv,
            reward_loss=self._rew_acc * inv,
            value_loss=self._val_acc * inv,
            termination_loss=self._term_acc * inv,
            wm_loss=(
                self._cons_acc + self._rew_acc + self._val_acc + self._term_acc
            ) * inv,
            pi_loss=self._pi_acc * inv,
            pi_scale=self.pol_step.scale.value,
            q_mean=self._q_mean_acc * inv,
            q_min=self._q_min_last,
            q_max=self._q_max_last,
            td_target_mean=self._td_mean_acc * inv,
            td_target_min=self._td_min_last,
            td_target_max=self._td_max_last,
        )
        if Bool(logger):
            var lg = logger.value()
            lg[].log_scalar("consistency_loss", Float64(m.consistency_loss), step)
            lg[].log_scalar("reward_loss", Float64(m.reward_loss), step)
            lg[].log_scalar("value_loss", Float64(m.value_loss), step)
            lg[].log_scalar("termination_loss", Float64(m.termination_loss), step)
            lg[].log_scalar("wm_loss", Float64(m.wm_loss), step)
            lg[].log_scalar("policy_loss", Float64(m.pi_loss), step)
            lg[].log_scalar("pi_scale", Float64(m.pi_scale), step)
            lg[].log_scalar("q_mean", Float64(m.q_mean), step)
            lg[].log_scalar("q_min", Float64(m.q_min), step)
            lg[].log_scalar("q_max", Float64(m.q_max), step)
            lg[].log_scalar("td_target_mean", Float64(m.td_target_mean), step)
            lg[].log_scalar("td_target_min", Float64(m.td_target_min), step)
            lg[].log_scalar("td_target_max", Float64(m.td_target_max), step)
        self._cons_acc = Scalar[DT](0.0)
        self._rew_acc = Scalar[DT](0.0)
        self._val_acc = Scalar[DT](0.0)
        self._term_acc = Scalar[DT](0.0)
        self._pi_acc = Scalar[DT](0.0)
        self._q_mean_acc = Scalar[DT](0.0)
        self._td_mean_acc = Scalar[DT](0.0)
        self._n_diag = 0
        return m^

    def flush_metrics_through_logger[
        L: Logger
    ](
        mut self,
        logger: Optional[UnsafePointer[L, MutAnyOrigin]],
        step: Int,
    ) raises:
        _ = self.flush_metrics[L](logger, step)

    # ── Checkpointing (storage one-file v2 envelope) ───────────────────
    def save_state(mut self, path: String) raises:
        """Save every world-model module + the Q ensemble (online + target) +
        policy + termination into a SINGLE storage-ckpt envelope. running_scale
        + optimizer moments are NOT persisted (resume re-warms)."""
        comptime tg = Self.target
        var w = CheckpointWriter(save_moments=False)
        w.mode = 0
        self.encoder.for_each_param[tg](w, self.ctx, "encoder")
        self.dynamics.for_each_param[tg](w, self.ctx, "dynamics")
        self.reward.for_each_param[tg](w, self.ctx, "reward")
        self.policy.for_each_param[tg](w, self.ctx, "policy")
        self.q0.for_each_param[tg](w, self.ctx, "q0")
        self.q1.for_each_param[tg](w, self.ctx, "q1")
        self.q2.for_each_param[tg](w, self.ctx, "q2")
        self.q3.for_each_param[tg](w, self.ctx, "q3")
        self.q4.for_each_param[tg](w, self.ctx, "q4")
        self.qt0.for_each_param[tg](w, self.ctx, "qt0")
        self.qt1.for_each_param[tg](w, self.ctx, "qt1")
        self.qt2.for_each_param[tg](w, self.ctx, "qt2")
        self.qt3.for_each_param[tg](w, self.ctx, "qt3")
        self.qt4.for_each_param[tg](w, self.ctx, "qt4")
        self.termination.for_each_param[tg](w, self.ctx, "termination")
        w.mode = 1
        self.encoder.for_each_state[tg](w, self.ctx, "encoder")
        self.dynamics.for_each_state[tg](w, self.ctx, "dynamics")
        self.reward.for_each_state[tg](w, self.ctx, "reward")
        self.policy.for_each_state[tg](w, self.ctx, "policy")
        self.q0.for_each_state[tg](w, self.ctx, "q0")
        self.q1.for_each_state[tg](w, self.ctx, "q1")
        self.q2.for_each_state[tg](w, self.ctx, "q2")
        self.q3.for_each_state[tg](w, self.ctx, "q3")
        self.q4.for_each_state[tg](w, self.ctx, "q4")
        self.qt0.for_each_state[tg](w, self.ctx, "qt0")
        self.qt1.for_each_state[tg](w, self.ctx, "qt1")
        self.qt2.for_each_state[tg](w, self.ctx, "qt2")
        self.qt3.for_each_state[tg](w, self.ctx, "qt3")
        self.qt4.for_each_state[tg](w, self.ctx, "qt4")
        self.termination.for_each_state[tg](w, self.ctx, "termination")
        with open(path, "w") as f:
            f.write(w.content)

    def load_state(mut self, path: String) raises:
        """Inverse of `save_state` (online + target Q both restored)."""
        comptime tg = Self.target
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
        self.encoder.for_each_param[tg](r, self.ctx, "encoder")
        self.dynamics.for_each_param[tg](r, self.ctx, "dynamics")
        self.reward.for_each_param[tg](r, self.ctx, "reward")
        self.policy.for_each_param[tg](r, self.ctx, "policy")
        self.q0.for_each_param[tg](r, self.ctx, "q0")
        self.q1.for_each_param[tg](r, self.ctx, "q1")
        self.q2.for_each_param[tg](r, self.ctx, "q2")
        self.q3.for_each_param[tg](r, self.ctx, "q3")
        self.q4.for_each_param[tg](r, self.ctx, "q4")
        self.qt0.for_each_param[tg](r, self.ctx, "qt0")
        self.qt1.for_each_param[tg](r, self.ctx, "qt1")
        self.qt2.for_each_param[tg](r, self.ctx, "qt2")
        self.qt3.for_each_param[tg](r, self.ctx, "qt3")
        self.qt4.for_each_param[tg](r, self.ctx, "qt4")
        self.termination.for_each_param[tg](r, self.ctx, "termination")
        r.mode = 1
        self.encoder.for_each_state[tg](r, self.ctx, "encoder")
        self.dynamics.for_each_state[tg](r, self.ctx, "dynamics")
        self.reward.for_each_state[tg](r, self.ctx, "reward")
        self.policy.for_each_state[tg](r, self.ctx, "policy")
        self.q0.for_each_state[tg](r, self.ctx, "q0")
        self.q1.for_each_state[tg](r, self.ctx, "q1")
        self.q2.for_each_state[tg](r, self.ctx, "q2")
        self.q3.for_each_state[tg](r, self.ctx, "q3")
        self.q4.for_each_state[tg](r, self.ctx, "q4")
        self.qt0.for_each_state[tg](r, self.ctx, "qt0")
        self.qt1.for_each_state[tg](r, self.ctx, "qt1")
        self.qt2.for_each_state[tg](r, self.ctx, "qt2")
        self.qt3.for_each_state[tg](r, self.ctx, "qt3")
        self.qt4.for_each_state[tg](r, self.ctx, "qt4")
        self.termination.for_each_state[tg](r, self.ctx, "termination")

    # ── td-target dispatch: thread the random target pair as DISTINCT fields ─
    def _td_dispatch(
        mut self,
        a: Int, b: Int,
        mut obs: Tensor, mut rew: Tensor, mut done: Tensor, mut td: Tensor,
        gamma: Scalar[DT],
        ctx: Optional[DeviceContext],
    ) raises:
        comptime tg = Self.target
        comptime for i in range(NQ):
            comptime for j in range(NQ):
                comptime if i < j:
                    if (a == i and b == j) or (a == j and b == i):
                        self.td_step.step[tg](
                            self.encoder, self.policy,
                            self.get_qt[i](), self.get_qt[j](),
                            obs, rew, done, td, gamma, ctx,
                        )

    def _policy_dispatch(
        mut self,
        a: Int, b: Int,
        mut zpol: Tensor,
    ) raises -> Scalar[DT]:
        comptime tg = Self.target
        comptime for i in range(NQ):
            comptime for j in range(NQ):
                comptime if i < j:
                    if (a == i and b == j) or (a == j and b == i):
                        return self.pol_step.step[tg](
                            self.policy, self.get_q[i](), self.get_q[j](),
                            self.pi_opt, zpol, self.ctx,
                        )
        return Scalar[DT](0.0)

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
        var ob = List[Scalar[DT]](length=BB * (HH + 1) * OBSD, fill=0)
        var ab = List[Scalar[DT]](length=BB * HH * ACTD, fill=0)
        var rb = List[Scalar[DT]](length=BB * HH, fill=0)
        var dbf = List[Scalar[DT]](length=BB * HH, fill=0)
        self.replay.sample_batch[BB, HH](
            rebind[UnsafePointer[Scalar[DT], MutAnyOrigin]](
                UnsafePointer(to=ob[0])
            ),
            rebind[UnsafePointer[Scalar[DT], MutAnyOrigin]](
                UnsafePointer(to=ab[0])
            ),
            rebind[UnsafePointer[Scalar[DT], MutAnyOrigin]](
                UnsafePointer(to=rb[0])
            ),
            rebind[UnsafePointer[Scalar[DT], MutAnyOrigin]](
                UnsafePointer(to=dbf[0])
            ),
        )

        # t-major input Tensors.
        var ot = Tensor.alloc((HH + 1) * BB * OBSD)
        var at = Tensor.alloc(HH * BB * ACTD)
        var rt = Tensor.alloc(HH * BB)
        var dt = Tensor.alloc(HH * BB)
        var td = Tensor.alloc(HH * BB)
        for b in range(BB):
            for t in range(HH + 1):
                for i in range(OBSD):
                    ot.data[(t * BB + b) * OBSD + i] = ob[
                        (b * (HH + 1) + t) * OBSD + i
                    ]
            for t in range(HH):
                for j in range(ACTD):
                    at.data[(t * BB + b) * ACTD + j] = ab[(b * HH + t) * ACTD + j]
                rt.data[t * BB + b] = rb[b * HH + t]
                dt.data[t * BB + b] = dbf[b * HH + t]

        var ta = Int(random_float64() * Float64(NQ))
        if ta >= NQ:
            ta = NQ - 1
        var tb = (ta + 1) % NQ
        var pa = Int(random_float64() * Float64(NQ))
        if pa >= NQ:
            pa = NQ - 1
        var pb = (pa + 1) % NQ

        var ctx = self.ctx
        comptime if tg == "gpu":
            ot.upload(ctx.value())
            at.upload(ctx.value())
            rt.upload(ctx.value())
            dt.upload(ctx.value())
            td.upload(ctx.value())

        # ── TD targets (stop-grad) ─────────────────────────────────────
        self._td_dispatch(ta, tb, ot, rt, dt, td, self.gamma, ctx)

        # ── WM BPTT ─────────────────────────────────────────────────────
        var wl = self.wm_step.step[tg](
            self.wm_graph, self.encoder, self.dynamics, self.reward,
            self.q0, self.q1, self.q2, self.q3, self.q4, self.termination,
            self.enc_opt, self.dyn_opt, self.rew_opt,
            self.qo0, self.qo1, self.qo2, self.qo3, self.qo4, self.term_opt,
            ot, at, rt, td, dt, ctx,
        )
        self._last_cons = wl.consistency
        self._last_rew = wl.reward
        self._last_val = wl.value
        self._last_term = wl.termination
        self._last_wm = wl.total()

        # ── policy update on encoded latents ───────────────────────────
        var zpol = Tensor.make[tg](Self.PB * LAT, ctx)
        var obs_pb = Tensor.alloc(Self.PB * OBSD)
        for i in range(Self.PB * OBSD):
            obs_pb.data[i] = ot.data[i]
        comptime if tg == "gpu":
            obs_pb.upload(ctx.value())
        self.encoder.forward[tg, Self.PB](TensorRefs[1](obs_pb), zpol, ctx)
        self._last_pi = self._policy_dispatch(pa, pb, zpol)

        # ── Polyak (target ← online) ────────────────────────────────────
        self.qt0.polyak_from[tg](self.q0, self.tau, ctx)
        self.qt1.polyak_from[tg](self.q1, self.tau, ctx)
        self.qt2.polyak_from[tg](self.q2, self.tau, ctx)
        self.qt3.polyak_from[tg](self.q3, self.tau, ctx)
        self.qt4.polyak_from[tg](self.q4, self.tau, ctx)

        # ── TD-target stats over the [H*B] targets (host on both paths) ──
        comptime if tg == "gpu":
            td.download(ctx.value())
        var td_sum: Scalar[DT] = 0.0
        var td_mn = td.data[0]
        var td_mx = td.data[0]
        for i in range(HH * BB):
            var v = td.data[i]
            td_sum += v
            if v < td_mn:
                td_mn = v
            if v > td_mx:
                td_mx = v

        self._cons_acc += self._last_cons
        self._rew_acc += self._last_rew
        self._val_acc += self._last_val
        self._term_acc += self._last_term
        self._pi_acc += self._last_pi
        self._q_mean_acc += self.pol_step.q_mean
        self._q_min_last = self.pol_step.q_min
        self._q_max_last = self.pol_step.q_max
        self._td_mean_acc += td_sum / Scalar[DT](HH * BB)
        self._td_min_last = td_mn
        self._td_max_last = td_mx
        self._n_diag += 1
        return True

    # ── one-call training / eval drivers (single-env facade) ───────────────
    # TD-MPC2 acts single-env (the MPPI planner + world-model BPTT are
    # per-env), so unlike SAC there is no batched driver — these methods
    # internalize the collect → record → train_step loop (+ warmup, periodic
    # eval, logging, checkpoint) so examples don't hand-roll it.

    def evaluate[
        E: BoxContinuousActionEnv,
        USE_MPC: Bool = False,
    ](
        mut self,
        mut env: E,
        *,
        episodes: Int = 2,
        max_steps: Int = 1_000,
    ) raises -> Scalar[DT]:
        """Deterministic eval → mean episode return.

        `USE_MPC=False` (default) acts greedily via `a = π(encode(obs))`;
        `USE_MPC=True` plans with MPPI (`select_action_mpc`, GPU only — a
        comptime assert in that path enforces `target == "gpu"`)."""
        comptime OBSD = Self.OBS
        comptime ACTD = Self.ACT
        var obs_l = List[Scalar[DT]](length=OBSD, fill=Scalar[DT](0.0))
        var act_l = List[Scalar[DT]](length=ACTD, fill=Scalar[DT](0.0))
        var total: Scalar[DT] = 0.0
        for _ep in range(episodes):
            var obs = env.reset_obs_list()
            comptime if USE_MPC:
                self.mpc_start_episode()
            for _s in range(max_steps):
                for d in range(OBSD):
                    obs_l[d] = Scalar[DT](obs[d])
                comptime if USE_MPC:
                    self.select_action_mpc(obs_l, act_l, explore=False)
                else:
                    self.select_greedy_action(obs_l, act_l)
                var env_action = List[Scalar[E.dtype]](capacity=ACTD)
                for j in range(ACTD):
                    env_action.append(Scalar[E.dtype](act_l[j]))
                var r = env.step_continuous_vec[E.dtype](env_action)
                total += Scalar[DT](r[1])
                obs = r[0].copy()
                if r[2]:
                    break
        return total / Scalar[DT](episodes if episodes > 0 else 1)

    def train[
        E: BoxContinuousActionEnv,
        L: Logger = NoOpLogger,
        EE: BoxContinuousActionEnv = E,
        USE_MPC: Bool = False,
    ](
        mut self,
        mut env: E,
        total_timesteps: Int,
        *,
        train_every: Int = 1,
        print_every: Int = 20_000,
        verbose: Bool = True,
        logger: Optional[UnsafePointer[L, MutAnyOrigin]] = None,
        diag_every: Int = 0,
        checkpoint_path: String = "",
        checkpoint_every: Int = 0,
        eval_env: Optional[UnsafePointer[EE, MutAnyOrigin]] = None,
        eval_every: Int = 0,
        eval_episodes: Int = 2,
        eval_max_steps: Int = 1_000,
    ) raises -> Scalar[DT]:
        """Single-env TD-MPC2 training driver → best eval return.

        One env step + (after warmup) one `train_step` per iteration:
          * `step < learning_starts` → uniform random actions in [-1, 1]
            (the `learning_starts` passed at construction);
          * else → `USE_MPC ? select_action_mpc : select_action` (explore).

        Bootstrapping: records `done = was_terminated()` (NATURAL termination
        only) so the value bootstrap continues across truncation and drops on
        a real terminal — truncation-only envs (e.g. HalfCheetah with
        `TERMINATE_ON_UNHEALTHY=False`) record `done = 0` throughout.

        Optional streams (all off by default):
          * `logger` + `diag_every > 0` → drain the full TD-MPC2 metric bundle
            (consistency/reward/value/wm/policy losses, q/td stats) every
            `diag_every` env-steps, plus an `avg_reward` training signal;
          * `checkpoint_every > 0` + `checkpoint_path` → `save_state` cadence
            (+ once at the end);
          * `eval_env` (ISOLATED env ptr) + `eval_every > 0` → periodic
            DETERMINISTIC eval logged as `eval/mean_return` (the deployable
            signal; pass `USE_MPC=True` to eval the planner)."""
        comptime OBSD = Self.OBS
        comptime ACTD = Self.ACT

        var obs_l = List[Scalar[DT]](length=OBSD, fill=Scalar[DT](0.0))
        var act_l = List[Scalar[DT]](length=ACTD, fill=Scalar[DT](0.0))
        var obs = env.reset_obs_list()

        # Ring buffer of the last 100 completed-episode returns (for the
        # `avg_reward` stream + progress prints). Avoids List slicing/pop.
        var window = List[Scalar[DT]](length=100, fill=Scalar[DT](0.0))
        var w_idx = 0
        var w_cnt = 0
        var cur_ret: Scalar[DT] = 0.0
        var best: Scalar[DT] = Scalar[DT](-1.0e30)
        var t_start = perf_counter_ns()

        comptime if USE_MPC:
            self.mpc_start_episode()

        for step in range(total_timesteps):
            for d in range(OBSD):
                obs_l[d] = Scalar[DT](obs[d])

            if step < self.learning_starts:
                for j in range(ACTD):
                    act_l[j] = Scalar[DT](random_float64() * 2.0 - 1.0)
            else:
                comptime if USE_MPC:
                    self.select_action_mpc(obs_l, act_l, explore=True)
                else:
                    self.select_action(obs_l, act_l, explore=True)

            var env_action = List[Scalar[E.dtype]](capacity=ACTD)
            for j in range(ACTD):
                env_action.append(Scalar[E.dtype](act_l[j]))
            var res = env.step_continuous_vec[E.dtype](env_action)
            var reward = Scalar[DT](res[1])
            var done = res[2]
            # Replay stores NATURAL termination only (truncation keeps the
            # bootstrap). `was_terminated()` returns terminated-not-truncated.
            var term: Scalar[DT] = 1.0 if env.was_terminated() else 0.0
            self.record(obs_l, act_l, reward, term)
            cur_ret += reward
            obs = res[0].copy()

            if done:
                obs = env.reset_obs_list()
                window[w_idx] = cur_ret
                w_idx = (w_idx + 1) % 100
                if w_cnt < 100:
                    w_cnt += 1
                cur_ret = 0.0
                comptime if USE_MPC:
                    self.mpc_start_episode()

            if step >= self.learning_starts and step % train_every == 0:
                _ = self.train_step()

            if diag_every > 0 and step > 0 and step % diag_every == 0:
                self.flush_metrics_through_logger[L](logger, step)
                if Bool(logger):
                    var lg = logger.value()
                    if w_cnt > 0:
                        var s: Scalar[DT] = 0.0
                        for k in range(w_cnt):
                            s += window[k]
                        lg[].log_scalar(
                            "avg_reward",
                            Float64(s / Scalar[DT](w_cnt)),
                            step,
                        )
                    lg[].flush()

            if (
                checkpoint_every > 0
                and step > 0
                and step % checkpoint_every == 0
                and checkpoint_path.byte_length() > 0
            ):
                self.save_state(checkpoint_path)

            var do_eval = (
                eval_every > 0 and step > 0 and step % eval_every == 0
                and Bool(eval_env)
            )
            if do_eval:
                var ep = eval_env.value()
                var ret = self.evaluate[EE, USE_MPC](
                    ep[], episodes=eval_episodes, max_steps=eval_max_steps
                )
                if ret > best:
                    best = ret
                if Bool(logger):
                    var lg = logger.value()
                    lg[].log_scalar("eval/mean_return", Float64(ret), step)
                    lg[].log_scalar("eval/best_return", Float64(best), step)
                if verbose:
                    var elapsed = (
                        Float64(perf_counter_ns() - t_start) / 1e9
                    )
                    print(
                        "  step", step, " eval_return=", ret, " best=", best,
                        " wm=", self.last_wm_loss(),
                        " pi=", self.last_pi_loss(),
                        " (", elapsed, "s )",
                    )
            elif verbose and print_every > 0 and step > 0 and (
                step % print_every == 0
            ):
                var elapsed = Float64(perf_counter_ns() - t_start) / 1e9
                var mean_ret: Scalar[DT] = 0.0
                if w_cnt > 0:
                    for k in range(w_cnt):
                        mean_ret += window[k]
                    mean_ret /= Scalar[DT](w_cnt)
                print(
                    "  step", step, " mean_ret(100)=", mean_ret,
                    " wm=", self.last_wm_loss(),
                    " pi=", self.last_pi_loss(), " (", elapsed, "s )",
                )

        if checkpoint_every > 0 and checkpoint_path.byte_length() > 0:
            self.save_state(checkpoint_path)
        return best
