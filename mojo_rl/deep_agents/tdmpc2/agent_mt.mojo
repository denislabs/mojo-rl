"""TD-MPC2 multi-task agent (deep_agents, storage framework, CPU + GPU, MPC-off).

A parallel struct to `TDMPC2Agent` for the multi-task setting: one agent trained
over a SET of envs (heterogeneous obs/action dims, padded to `MAX_OBS`/`MAX_ACT`),
conditioned on a learned per-task embedding. The single-task `TDMPC2Agent` and all
its blocks are left untouched (bit-identical by construction); everything here
uses the `*MT` variants + the storage `TaskEmbedding` table.

Storage migration (mirrors `agent.mojo`): the 5 online Q heads, 5 target Q heads,
and 5 Q optimizers are DISTINCT FIELDS (q0..q4 / qt0..qt4 / qo0..qo4; NQ=5).
Storage threads externals into ONE forward/vjp call (two `mut` List subscripts
can't alias). The WM step threads all 5 online Q as distinct args; the random
PAIR steps (policy: online (pa,pb); td: target (ta,tb)) use a comptime-unrolled
guarded dispatch so two DISTINCT fields are threaded. Adam via ctor+adopt+step;
polyak via `Module.polyak_from`; storage CheckpointWriter/Reader (task_emb body
appended last); action/record entry points take `List[Scalar[DT]]`.

Acting is MPC-off (`a = π(encode([obs|task_emb]))`); the MPPI planner is NOT built
for the multi-task path (deferred).

train_step: sample length-T windows + per-window task id (host) → t-major →
gather task embeddings → TD targets (stop-grad) → WM BPTT (accumulates embedding
grad sites 1+2) → policy update on encoded latents (site 3) → ONE embedding Adam
step → Polyak. The embedding table's `zero_grad`/`step` bracket the whole step.
"""

from std.math import tanh
from std.random import random_float64
from layout import Layout, LayoutTensor
from max.gpu.host import DeviceContext

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
from mojo_rl.core.logger import Logger
from .metrics import TDMPC2Metrics

from .nets_mt import (
    TDMPC2EncoderMT, TDMPC2DynamicsMT, TDMPC2RewardMT, TDMPC2QNetMT,
    TDMPC2PolicyMT, TDMPC2TerminationMT,
)
from .wm_graph import NQ
from .wm_graph_mt import TDMPC2WMGraphMT
from .wm_step_mt import WMStepMT
from .policy_step_mt import PolicyStepMT
from .td_target_step_mt import TDTargetStepMT
from .task_embedding import TaskEmbedding


@fieldwise_init
struct TDMPC2MultiTaskAgent[
    target: StaticString,
    MAX_OBS: Int,
    ENC: Int,
    MAX_ACT: Int,
    LATENT: Int,
    MLP: Int,
    BINS: Int,
    SN: Int,
    VMIN: Int,
    VMAX: Int,
    B: Int,
    H: Int,
    CAP: Int,
    NUM_TASKS: Int,
    TASK_EMB: Int,
    QP: Float64 = 0.0,
](Movable & Deinitable):
    comptime EncT = TDMPC2EncoderMT[
        Self.MAX_OBS, Self.ENC, Self.LATENT, Self.SN, Self.TASK_EMB
    ]
    comptime DynT = TDMPC2DynamicsMT[
        Self.LATENT, Self.MAX_ACT, Self.MLP, Self.SN, Self.TASK_EMB
    ]
    comptime RewT = TDMPC2RewardMT[
        Self.LATENT, Self.MAX_ACT, Self.MLP, Self.BINS, Self.TASK_EMB
    ]
    comptime QNetT = TDMPC2QNetMT[
        Self.LATENT, Self.MAX_ACT, Self.MLP, Self.BINS, Self.TASK_EMB, Self.QP
    ]
    comptime TermT = TDMPC2TerminationMT[
        Self.LATENT, Self.MAX_ACT, Self.MLP, Self.TASK_EMB
    ]
    comptime PolicyT = TDMPC2PolicyMT[
        Self.LATENT, Self.MAX_ACT, Self.MLP, Self.TASK_EMB
    ]
    comptime GraphT = TDMPC2WMGraphMT[
        Self.LATENT, Self.MAX_ACT, Self.MLP, Self.BINS, Self.SN, Self.VMIN,
        Self.VMAX, Self.TASK_EMB, Self.QP,
    ]
    comptime EmbT = TaskEmbedding[Self.NUM_TASKS, Self.TASK_EMB]
    comptime PB = (Self.H + 1) * Self.B
    comptime AOBS = Self.MAX_OBS + Self.TASK_EMB
    comptime PIN = Self.LATENT + Self.TASK_EMB
    comptime WMStepT = WMStepMT[
        Self.MAX_OBS, Self.ENC, Self.MAX_ACT, Self.LATENT, Self.MLP, Self.BINS,
        Self.SN, Self.VMIN, Self.VMAX, Self.B, Self.H, Self.NUM_TASKS,
        Self.TASK_EMB, Self.QP,
    ]
    comptime PolStepT = PolicyStepMT[
        Self.LATENT, Self.MAX_ACT, Self.MLP, Self.BINS, Self.VMIN, Self.VMAX,
        Self.PB, Self.NUM_TASKS, Self.TASK_EMB, Self.QP,
    ]
    comptime TDStepT = TDTargetStepMT[
        Self.MAX_OBS, Self.ENC, Self.MAX_ACT, Self.LATENT, Self.MLP, Self.BINS,
        Self.SN, Self.VMIN, Self.VMAX, Self.B, Self.H, Self.NUM_TASKS,
        Self.TASK_EMB, Self.QP,
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
    var task_emb: Self.EmbT

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
    var act_rsample: RSample[Self.MAX_ACT]
    var replay: SequenceReplay[Self.MAX_OBS, Self.MAX_ACT, Self.CAP]

    # Per-task action mask [NUM_TASKS, MAX_ACT] (1=active, 0=unused), applied at
    # acting time (recorded actions are masked by the env wrapper). Default 1s.
    var action_mask: List[Scalar[DT]]
    # Acting task id (set by the driver before select_action / record).
    var cur_task: Int

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

        # Termination head (item B): Zero-init at bce_coef=0 (no RNG draw),
        # Kaiming when episodic. Built before the table (RNG discipline).
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

        var ar = RSample[Self.MAX_ACT].make[tg, INIT=Zero](ctx=ctx)
        ar.action_scale = action_scale

        # Task embedding built LAST (RNG discipline) — its random init draws from
        # the global RNG, so building it after every net keeps the nets' init
        # stream unperturbed (matches the term-head ordering convention).
        var te = Self.EmbT.make[tg](ctx=ctx, lr=lr)

        var amask = List[Scalar[DT]](
            length=Self.NUM_TASKS * Self.MAX_ACT, fill=Scalar[DT](1.0)
        )

        return Self(
            encoder=enc^, dynamics=dyn^, reward=rew^,
            q0=q0^, q1=q1^, q2=q2^, q3=q3^, q4=q4^,
            qt0=qt0^, qt1=qt1^, qt2=qt2^, qt3=qt3^, qt4=qt4^,
            policy=pol^, termination=term^, task_emb=te^,
            enc_opt=enc_opt^, dyn_opt=dyn_opt^, rew_opt=rew_opt^,
            qo0=qo0^, qo1=qo1^, qo2=qo2^, qo3=qo3^, qo4=qo4^,
            pi_opt=pi_opt^, term_opt=term_opt^,
            wm_graph=Self.GraphT.make[tg, INIT=Kaiming](ctx=ctx),
            wm_step=Self.WMStepT.make[tg](ctx=ctx, termination_coef=bce_coef),
            pol_step=Self.PolStepT.make[tg](ctx=ctx),
            td_step=Self.TDStepT.make[tg](ctx=ctx),
            act_rsample=ar^,
            replay=SequenceReplay[
                Self.MAX_OBS, Self.MAX_ACT, Self.CAP
            ].new(),
            action_mask=amask^, cur_task=0,
            gamma=gamma, tau=tau, bce_coef=bce_coef, action_scale=action_scale,
            learning_starts=learning_starts, step_count=0,
            _last_wm=Scalar[DT](0.0), _last_pi=Scalar[DT](0.0),
            _last_cons=Scalar[DT](0.0), _last_rew=Scalar[DT](0.0),
            _last_val=Scalar[DT](0.0), _last_term=Scalar[DT](0.0),
            _cons_acc=Scalar[DT](0.0), _rew_acc=Scalar[DT](0.0),
            _val_acc=Scalar[DT](0.0), _term_acc=Scalar[DT](0.0),
            _pi_acc=Scalar[DT](0.0),
            _q_mean_acc=Scalar[DT](0.0), _q_min_last=Scalar[DT](0.0),
            _q_max_last=Scalar[DT](0.0), _td_mean_acc=Scalar[DT](0.0),
            _td_min_last=Scalar[DT](0.0), _td_max_last=Scalar[DT](0.0),
            _n_diag=0, ctx=ctx, temperature=temperature,
        )

    # ── task / mask plumbing ───────────────────────────────────────────
    def set_task(mut self, t: Int):
        self.cur_task = t

    def task(self) -> Int:
        return self.cur_task

    def set_action_mask(mut self, t: Int, ref mask: List[Scalar[DT]]):
        for j in range(Self.MAX_ACT):
            self.action_mask[t * Self.MAX_ACT + j] = mask[j]

    # ── acting (MPC-off): a = π(encode([obs|task_emb])) ─────────────────
    def select_action(
        mut self,
        ref obs: List[Scalar[DT]],
        mut act_out: List[Scalar[DT]],
        explore: Bool = True,
    ) raises:
        comptime tg = Self.target
        comptime A = Self.MAX_ACT
        comptime LAT = Self.LATENT
        comptime MO = Self.MAX_OBS
        comptime EMB = Self.TASK_EMB
        comptime AOBS = Self.AOBS
        comptime PIN = Self.PIN
        var ctx = self.ctx

        # gather this task's embedding (1 row).
        var tids = Tensor.alloc(1)
        tids.data[0] = Scalar[DT](self.cur_task)
        comptime if tg == "gpu":
            tids.upload(ctx.value())
        var tem = Tensor.make[tg](EMB, ctx)
        self.task_emb.gather[tg, 1](tids, tem, ctx)
        comptime if tg == "gpu":
            tem.download(ctx.value())

        # ein = [obs | tem]
        var ein = Tensor.alloc(AOBS)
        for i in range(MO):
            ein.data[i] = obs[i]
        for e in range(EMB):
            ein.data[MO + e] = tem.data[e]
        comptime if tg == "gpu":
            ein.upload(ctx.value())
        var z = Tensor.make[tg](LAT, ctx)
        self.encoder.forward[tg, 1](TensorRefs[1](ein), z, ctx)
        comptime if tg == "gpu":
            z.download(ctx.value())

        # pin = [z | tem]
        var pin = Tensor.alloc(PIN)
        for k in range(LAT):
            pin.data[k] = z.data[k]
        for e in range(EMB):
            pin.data[LAT + e] = tem.data[e]
        comptime if tg == "gpu":
            pin.upload(ctx.value())
        var pio = Tensor.make[tg](2 * A, ctx)
        self.policy.forward[tg, 1](TensorRefs[1](pin), pio, ctx)
        if explore:
            var alp = Tensor.make[tg](A + 1, ctx)
            self.act_rsample.forward[tg, 1](TensorRefs[1](pio), alp, ctx)
            comptime if tg == "gpu":
                alp.download(ctx.value())
            for j in range(A):
                act_out[j] = alp.data[j] * self.action_mask[
                    self.cur_task * A + j
                ]
        else:
            comptime if tg == "gpu":
                pio.download(ctx.value())
            for j in range(A):
                act_out[j] = tanh(pio.data[j]) * self.action_scale * (
                    self.action_mask[self.cur_task * A + j]
                )

    def select_greedy_action(
        mut self,
        ref obs: List[Scalar[DT]],
        mut act_out: List[Scalar[DT]],
    ) raises:
        self.select_action(obs, act_out, explore=False)

    def record(
        mut self,
        ref obs: List[Scalar[DT]],
        ref act: List[Scalar[DT]],
        reward: Scalar[DT],
        done: Scalar[DT],
    ) raises:
        self.replay.record_task(
            rebind[Pointer[Scalar[DT], MutAnyOrigin]](
                Pointer(to=obs[0])
            ),
            rebind[Pointer[Scalar[DT], MutAnyOrigin]](
                Pointer(to=act[0])
            ),
            reward, done, self.cur_task,
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
        logger: Optional[Pointer[L, MutAnyOrigin]],
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
        logger: Optional[Pointer[L, MutAnyOrigin]],
        step: Int,
    ) raises:
        _ = self.flush_metrics[L](logger, step)

    # ── Checkpointing (storage v2 envelope; table appended LAST) ────────
    def save_state(mut self, path: String) raises:
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
        # Task embedding table appended last (param + Adam moments).
        self.task_emb.save_body(w.content, "task_emb")
        with open(path, "w") as f:
            f.write(w.content)

    def load_state(mut self, path: String) raises:
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
        # Trailing task-embedding body, read positionally from the reader cursor.
        self.task_emb.load_body(r.lines, r.cur, "task_emb")

    # ── td-target dispatch: thread the random target pair as DISTINCT fields ─
    def _td_dispatch(
        mut self,
        a: Int, b: Int,
        mut task_ids: Tensor,
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
                            self.task_emb, task_ids,
                            obs, rew, done, td, gamma, ctx,
                        )

    def _policy_dispatch(
        mut self,
        a: Int, b: Int,
        mut zpol: Tensor,
        mut task_ids: Tensor,
    ) raises -> Scalar[DT]:
        comptime tg = Self.target
        comptime for i in range(NQ):
            comptime for j in range(NQ):
                comptime if i < j:
                    if (a == i and b == j) or (a == j and b == i):
                        return self.pol_step.step[tg](
                            self.policy, self.get_q[i](), self.get_q[j](),
                            self.pi_opt, self.task_emb, zpol, task_ids, self.ctx,
                        )
        return Scalar[DT](0.0)

    def train_step(mut self) raises -> Bool:
        self.step_count += 1
        if not self.replay.can_sample[Self.H]():
            return False
        if self.replay.count() < self.learning_starts:
            return False

        comptime tg = Self.target
        comptime MO = Self.MAX_OBS
        comptime A = Self.MAX_ACT
        comptime LAT = Self.LATENT
        comptime EMB = Self.TASK_EMB
        comptime AOBS = Self.AOBS
        comptime HH = Self.H
        comptime BB = Self.B
        comptime PBP = Self.PB

        # ── sample (b-major) + per-window task ids ──────────────────────
        var ob = List[Scalar[DT]](length=BB * (HH + 1) * MO, fill=0)
        var ab = List[Scalar[DT]](length=BB * HH * A, fill=0)
        var rb = List[Scalar[DT]](length=BB * HH, fill=0)
        var dbf = List[Scalar[DT]](length=BB * HH, fill=0)
        var tk = List[Scalar[DT]](length=BB, fill=0)
        self.replay.sample_batch_task[BB, HH](
            rebind[Pointer[Scalar[DT], MutAnyOrigin]](
                Pointer(to=ob[0])
            ),
            rebind[Pointer[Scalar[DT], MutAnyOrigin]](
                Pointer(to=ab[0])
            ),
            rebind[Pointer[Scalar[DT], MutAnyOrigin]](
                Pointer(to=rb[0])
            ),
            rebind[Pointer[Scalar[DT], MutAnyOrigin]](
                Pointer(to=dbf[0])
            ),
            rebind[Pointer[Scalar[DT], MutAnyOrigin]](
                Pointer(to=tk[0])
            ),
        )

        # ── t-major input Tensors ───────────────────────────────────────
        var ot = Tensor.alloc((HH + 1) * BB * MO)
        var at = Tensor.alloc(HH * BB * A)
        var rt = Tensor.alloc(HH * BB)
        var dt = Tensor.alloc(HH * BB)
        var td = Tensor.alloc(HH * BB)
        var tids = Tensor.alloc(BB)            # [B] per-window task id
        var tids_pb = Tensor.alloc(PBP)        # [PB] per-row task id
        for b in range(BB):
            tids.data[b] = tk[b]
            for t in range(HH + 1):
                for i in range(MO):
                    ot.data[(t * BB + b) * MO + i] = ob[
                        (b * (HH + 1) + t) * MO + i
                    ]
            for t in range(HH):
                for j in range(A):
                    at.data[(t * BB + b) * A + j] = ab[(b * HH + t) * A + j]
                rt.data[t * BB + b] = rb[b * HH + t]
                dt.data[t * BB + b] = dbf[b * HH + t]
        for t in range(HH + 1):
            for b in range(BB):
                tids_pb.data[t * BB + b] = tk[b]

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
            tids.upload(ctx.value())
            tids_pb.upload(ctx.value())

        # bracket the embedding grad accumulation around the whole step.
        self.task_emb.zero_grad[tg]()

        # ── TD targets (stop-grad) ──────────────────────────────────────
        self._td_dispatch(ta, tb, tids, ot, rt, dt, td, self.gamma, ctx)

        # ── WM BPTT (accumulates embedding grad sites 1+2) ──────────────
        var wl = self.wm_step.step[tg](
            self.wm_graph, self.encoder, self.dynamics, self.reward,
            self.q0, self.q1, self.q2, self.q3, self.q4, self.termination,
            self.task_emb,
            self.enc_opt, self.dyn_opt, self.rew_opt,
            self.qo0, self.qo1, self.qo2, self.qo3, self.qo4, self.term_opt,
            tids, ot, at, rt, td, dt, ctx,
        )
        self._last_cons = wl.consistency
        self._last_rew = wl.reward
        self._last_val = wl.value
        self._last_term = wl.termination
        self._last_wm = wl.total()

        # ── policy update on encoded [obs|task_emb] latents (site 3) ────
        var tem_pb = Tensor.make[tg](PBP * EMB, ctx)
        self.task_emb.gather[tg, PBP](tids_pb, tem_pb, ctx)
        comptime if tg == "gpu":
            tem_pb.download(ctx.value())
        var oaug = Tensor.alloc(PBP * AOBS)
        for row in range(PBP):
            for i in range(MO):
                oaug.data[row * AOBS + i] = ot.data[row * MO + i]
            for e in range(EMB):
                oaug.data[row * AOBS + MO + e] = tem_pb.data[row * EMB + e]
        comptime if tg == "gpu":
            oaug.upload(ctx.value())
        var zpol = Tensor.make[tg](PBP * LAT, ctx)
        self.encoder.forward[tg, PBP](TensorRefs[1](oaug), zpol, ctx)
        self._last_pi = self._policy_dispatch(pa, pb, zpol, tids_pb)

        # ── ONE embedding Adam step (sites 1+2+3 accumulated) ───────────
        self.task_emb.step[tg]()

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
