"""TD-MPC2 multi-task agent (deep_agents, CPU + GPU, MPC-off) — item C, §14.3.

A parallel struct to `TDMPC2Agent` for the multi-task setting: one agent trained
over a SET of envs (heterogeneous obs/action dims, padded to `MAX_OBS`/`MAX_ACT`),
conditioned on a learned per-task embedding. The single-task `TDMPC2Agent` and all
its blocks are left untouched (bit-identical by construction); everything here
uses the `*MT` variants + the `TaskEmbedding` table.

Acting is MPC-off (`a = π(encode([obs|task_emb]))`); the MPPI planner is NOT
built for the multi-task path (deferred — the 2-task lighthouse acts MPC-off,
like the HalfCheetah/Hopper examples).

train_step: sample length-T windows + per-window task id (host) → t-major →
gather task embeddings → TD targets (stop-grad) → WM BPTT (accumulates embedding
grad sites 1+2) → policy update on encoded latents (site 3) → ONE embedding Adam
step → Polyak. The embedding table's `zero_grad`/`step` bracket the whole step.
"""

from std.memory import alloc
from std.math import tanh
from std.random import random_float64

from mojo_rl.nn.constants import DT
from mojo_rl.nn.core.module import mptr
from mojo_rl.nn.initializer import Kaiming, Zero
from mojo_rl.nn.optimizer.adam import Adam
from layout import Layout, LayoutTensor, TileTensor, row_major
from std.gpu.host import DeviceContext, DeviceBuffer

from mojo_rl.deep_agents.primitives.rsample import RSample
from mojo_rl.deep_agents.dreamerv3.polyak import polyak_module
from mojo_rl.deep_agents.data.sequence_replay import SequenceReplay
from mojo_rl.nn.core.checkpoint import (
    save_state_v2_body, load_state_v2_body,
    save_state_v2_body_gpu, load_state_v2_body_gpu,
)
from mojo_rl.deep_agents.core.checkpoint_helpers import (
    save_optimizer_v2_body, load_optimizer_v2_body,
    save_optimizer_v2_body_gpu, load_optimizer_v2_body_gpu,
    split_lines_v2, read_file_v2, expect_v2_header,
)
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
from .td_target_step_mt import _cat2_k
from .task_embedding import TaskEmbedding


@always_inline
def _alloc(n: Int) -> UnsafePointer[Scalar[DT], MutAnyOrigin]:
    return alloc[Scalar[DT]](n)


@always_inline
def _dp(b: DeviceBuffer[DT]) -> UnsafePointer[Scalar[DT], MutAnyOrigin]:
    return mptr(b.unsafe_ptr())


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
](Movable & ImplicitlyDeletable):
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
    var q: List[Self.QNetT]
    var qt: List[Self.QNetT]
    var policy: Self.PolicyT
    var termination: Self.TermT
    var task_emb: Self.EmbT

    var enc_opt: Adam
    var dyn_opt: Adam
    var rew_opt: Adam
    var q_opt: List[Adam]
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
    var action_mask: UnsafePointer[Scalar[DT], MutAnyOrigin]
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

        comptime if Self.QP > 0.0:
            for i in range(NQ):
                qt[i].set_attr["training"](Scalar[DT](0.0))

        var term: Self.TermT
        if bce_coef > Scalar[DT](0.0):
            term = Self.TermT.make[tg, INIT=Kaiming](ctx=ctx)
        else:
            term = Self.TermT.make[tg, INIT=Zero](ctx=ctx)

        var enc_opt = Adam.make[tg, Self.EncT](enc, ctx=ctx)
        enc_opt.lr = lr * enc_lr_scale
        var dyn_opt = Adam.make[tg, Self.DynT](dyn, ctx=ctx)
        dyn_opt.lr = lr
        var rew_opt = Adam.make[tg, Self.RewT](rew, ctx=ctx)
        rew_opt.lr = lr
        var pi_opt = Adam.make[tg, Self.PolicyT](pol, ctx=ctx)
        pi_opt.lr = lr
        pi_opt.eps = Scalar[DT](1e-5)
        var term_opt = Adam.make[tg, Self.TermT](term, ctx=ctx)
        term_opt.lr = lr

        var ar = RSample[Self.MAX_ACT].make[tg, INIT=Zero](ctx=ctx)
        ar.action_scale = action_scale

        # Task embedding built LAST (RNG discipline) — its random init draws from
        # the global RNG, so building it after every net keeps the nets' init
        # stream unperturbed (matches the term-head ordering convention).
        var te = Self.EmbT.make[tg](ctx=ctx, lr=lr)

        var amask = alloc[Scalar[DT]](Self.NUM_TASKS * Self.MAX_ACT)
        for i in range(Self.NUM_TASKS * Self.MAX_ACT):
            amask[i] = Scalar[DT](1.0)

        return Self(
            encoder=enc^, dynamics=dyn^, reward=rew^, q=q^, qt=qt^, policy=pol^,
            termination=term^, task_emb=te^,
            enc_opt=enc_opt^, dyn_opt=dyn_opt^, rew_opt=rew_opt^, q_opt=q_opt^,
            pi_opt=pi_opt^, term_opt=term_opt^,
            wm_graph=Self.GraphT.make[tg, INIT=Kaiming](ctx=ctx),
            wm_step=Self.WMStepT.make[tg](ctx=ctx, termination_coef=bce_coef),
            pol_step=Self.PolStepT.make[tg](ctx=ctx),
            td_step=Self.TDStepT.make[tg](ctx=ctx),
            act_rsample=ar^,
            replay=SequenceReplay[Self.MAX_OBS, Self.MAX_ACT, Self.CAP].new(),
            action_mask=amask, cur_task=0,
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

    def set_action_mask(
        mut self, t: Int, mask: UnsafePointer[Scalar[DT], MutAnyOrigin]
    ):
        for j in range(Self.MAX_ACT):
            self.action_mask[t * Self.MAX_ACT + j] = mask[j]

    # ── acting (MPC-off): a = π(encode([obs|task_emb])) ─────────────────
    def select_action(
        mut self,
        obs: UnsafePointer[Scalar[DT], MutAnyOrigin],
        act_out: UnsafePointer[Scalar[DT], MutAnyOrigin],
        explore: Bool = True,
    ) raises:
        comptime tg = Self.target
        comptime A = Self.MAX_ACT
        comptime LAT = Self.LATENT
        comptime MO = Self.MAX_OBS
        comptime EMB = Self.TASK_EMB
        comptime AOBS = Self.AOBS
        comptime PIN = Self.PIN
        var tid = Scalar[DT](self.cur_task)
        comptime if tg == "cpu":
            var tem = _alloc(EMB)
            self.task_emb.gather[tg, 1](UnsafePointer(to=tid), tem)
            var ein = _alloc(AOBS)
            for i in range(MO):
                ein[i] = obs[i]
            for e in range(EMB):
                ein[MO + e] = tem[e]
            var z = _alloc(LAT)
            var z_t = TileTensor(z, row_major[1, LAT]())
            self.encoder.forward[tg, 1](
                TileTensor(ein, row_major[1, AOBS]()), output=z_t,
            )
            var pin = _alloc(PIN)
            for k in range(LAT):
                pin[k] = z[k]
            for e in range(EMB):
                pin[LAT + e] = tem[e]
            var pio = _alloc(2 * A)
            var pio_t = TileTensor(pio, row_major[1, 2 * A]())
            self.policy.forward[tg, 1](
                TileTensor(pin, row_major[1, PIN]()), output=pio_t,
            )
            if explore:
                var alp = _alloc(A + 1)
                var alp_t = TileTensor(alp, row_major[1, A + 1]())
                self.act_rsample.forward[tg, 1](pio_t, output=alp_t)
                for j in range(A):
                    act_out[j] = alp[j] * self.action_mask[self.cur_task * A + j]
                alp.free()
            else:
                for j in range(A):
                    act_out[j] = tanh(pio[j]) * self.action_scale * (
                        self.action_mask[self.cur_task * A + j]
                    )
            tem.free(); ein.free(); z.free(); pin.free(); pio.free()
        else:
            var ctx = self.ctx.value()
            var d_tid = _upload(ctx, UnsafePointer(to=tid), 1)
            var d_tem = ctx.enqueue_create_buffer[DT](EMB)
            self.task_emb.gather[tg, 1](_dp(d_tid), _dp(d_tem), ctx=ctx)
            var d_obs = _upload(ctx, obs, MO)
            var d_ein = ctx.enqueue_create_buffer[DT](AOBS)
            comptime ein_k = _cat2_k[1, MO, EMB]
            ctx.enqueue_function[ein_k](
                _lt1[MO](_dp(d_obs)), _lt1[EMB](_dp(d_tem)),
                _lt1[AOBS](_dp(d_ein)), grid_dim=1, block_dim=AOBS,
            )
            var d_z = ctx.enqueue_create_buffer[DT](LAT)
            var z_t = TileTensor(_dp(d_z), row_major[1, LAT]())
            self.encoder.forward[tg, 1](
                TileTensor(_dp(d_ein), row_major[1, AOBS]()), output=z_t,
            )
            var d_pin = ctx.enqueue_create_buffer[DT](PIN)
            comptime pin_k = _cat2_k[1, LAT, EMB]
            ctx.enqueue_function[pin_k](
                _lt1[LAT](_dp(d_z)), _lt1[EMB](_dp(d_tem)),
                _lt1[PIN](_dp(d_pin)), grid_dim=1, block_dim=PIN,
            )
            var d_pio = ctx.enqueue_create_buffer[DT](2 * A)
            var pio_t = TileTensor(_dp(d_pio), row_major[1, 2 * A]())
            self.policy.forward[tg, 1](
                TileTensor(_dp(d_pin), row_major[1, PIN]()), output=pio_t,
            )
            if explore:
                var d_alp = ctx.enqueue_create_buffer[DT](A + 1)
                var alp_t = TileTensor(_dp(d_alp), row_major[1, A + 1]())
                self.act_rsample.forward[tg, 1](pio_t, output=alp_t)
                var h = ctx.enqueue_create_host_buffer[DT](A + 1)
                ctx.enqueue_copy(h, d_alp)
                ctx.synchronize()
                for j in range(A):
                    act_out[j] = h.unsafe_ptr()[j] * self.action_mask[
                        self.cur_task * A + j
                    ]
            else:
                var h = ctx.enqueue_create_host_buffer[DT](2 * A)
                ctx.enqueue_copy(h, d_pio)
                ctx.synchronize()
                for j in range(A):
                    act_out[j] = tanh(h.unsafe_ptr()[j]) * self.action_scale * (
                        self.action_mask[self.cur_task * A + j]
                    )

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
        self.replay.record_task(obs, act, reward, done, self.cur_task)

    def last_wm_loss(self) -> Scalar[DT]:
        return self._last_wm

    def last_pi_loss(self) -> Scalar[DT]:
        return self._last_pi

    def last_termination_loss(self) -> Scalar[DT]:
        return self._last_term

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

    # ── Checkpointing (table appended LAST) ─────────────────────────────
    def save_state(mut self, path: String) raises:
        comptime tg = Self.target
        var body = String("")
        comptime if tg == "cpu":
            save_state_v2_body(self.encoder, body, "encoder")
            save_state_v2_body(self.dynamics, body, "dynamics")
            save_state_v2_body(self.reward, body, "reward")
            save_state_v2_body(self.policy, body, "policy")
            for i in range(NQ):
                save_state_v2_body(self.q[i], body, "q" + String(i))
                save_state_v2_body(self.qt[i], body, "qt" + String(i))
            save_optimizer_v2_body(self.enc_opt, body, "enc_opt")
            save_optimizer_v2_body(self.dyn_opt, body, "dyn_opt")
            save_optimizer_v2_body(self.rew_opt, body, "rew_opt")
            save_optimizer_v2_body(self.pi_opt, body, "pi_opt")
            for i in range(NQ):
                save_optimizer_v2_body(self.q_opt[i], body, "q_opt" + String(i))
            save_state_v2_body(self.termination, body, "termination")
            save_optimizer_v2_body(self.term_opt, body, "term_opt")
        else:
            var c = self.ctx.value()
            save_state_v2_body_gpu(self.encoder, body, "encoder", c)
            save_state_v2_body_gpu(self.dynamics, body, "dynamics", c)
            save_state_v2_body_gpu(self.reward, body, "reward", c)
            save_state_v2_body_gpu(self.policy, body, "policy", c)
            for i in range(NQ):
                save_state_v2_body_gpu(self.q[i], body, "q" + String(i), c)
                save_state_v2_body_gpu(self.qt[i], body, "qt" + String(i), c)
            save_optimizer_v2_body_gpu(self.enc_opt, body, "enc_opt")
            save_optimizer_v2_body_gpu(self.dyn_opt, body, "dyn_opt")
            save_optimizer_v2_body_gpu(self.rew_opt, body, "rew_opt")
            save_optimizer_v2_body_gpu(self.pi_opt, body, "pi_opt")
            for i in range(NQ):
                save_optimizer_v2_body_gpu(self.q_opt[i], body, "q_opt" + String(i))
            save_state_v2_body_gpu(self.termination, body, "termination", c)
            save_optimizer_v2_body_gpu(self.term_opt, body, "term_opt")
        # Task embedding table appended last (param + Adam moments).
        self.task_emb.save_body(body, "task_emb")
        var content = String("nn-ckpt v2\n") + body
        with open(path, "w") as f:
            f.write(content)

    def load_state(mut self, path: String) raises:
        comptime tg = Self.target
        var content = read_file_v2(path)
        var lines = split_lines_v2(content)
        expect_v2_header(lines)
        var idx: Int = 1
        comptime if tg == "cpu":
            load_state_v2_body(self.encoder, lines, idx, "encoder")
            load_state_v2_body(self.dynamics, lines, idx, "dynamics")
            load_state_v2_body(self.reward, lines, idx, "reward")
            load_state_v2_body(self.policy, lines, idx, "policy")
            for i in range(NQ):
                load_state_v2_body(self.q[i], lines, idx, "q" + String(i))
                load_state_v2_body(self.qt[i], lines, idx, "qt" + String(i))
            load_optimizer_v2_body(self.enc_opt, lines, idx, "enc_opt")
            load_optimizer_v2_body(self.dyn_opt, lines, idx, "dyn_opt")
            load_optimizer_v2_body(self.rew_opt, lines, idx, "rew_opt")
            load_optimizer_v2_body(self.pi_opt, lines, idx, "pi_opt")
            for i in range(NQ):
                load_optimizer_v2_body(self.q_opt[i], lines, idx, "q_opt" + String(i))
            load_state_v2_body(self.termination, lines, idx, "termination")
            load_optimizer_v2_body(self.term_opt, lines, idx, "term_opt")
        else:
            var c = self.ctx.value()
            load_state_v2_body_gpu(self.encoder, lines, idx, "encoder", c)
            load_state_v2_body_gpu(self.dynamics, lines, idx, "dynamics", c)
            load_state_v2_body_gpu(self.reward, lines, idx, "reward", c)
            load_state_v2_body_gpu(self.policy, lines, idx, "policy", c)
            for i in range(NQ):
                load_state_v2_body_gpu(self.q[i], lines, idx, "q" + String(i), c)
                load_state_v2_body_gpu(self.qt[i], lines, idx, "qt" + String(i), c)
            load_optimizer_v2_body_gpu(self.enc_opt, lines, idx, "enc_opt")
            load_optimizer_v2_body_gpu(self.dyn_opt, lines, idx, "dyn_opt")
            load_optimizer_v2_body_gpu(self.rew_opt, lines, idx, "rew_opt")
            load_optimizer_v2_body_gpu(self.pi_opt, lines, idx, "pi_opt")
            for i in range(NQ):
                load_optimizer_v2_body_gpu(self.q_opt[i], lines, idx, "q_opt" + String(i))
            load_state_v2_body_gpu(self.termination, lines, idx, "termination", c)
            load_optimizer_v2_body_gpu(self.term_opt, lines, idx, "term_opt")
        self.task_emb.load_body(lines, idx, "task_emb")

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

        # sample (b-major) + per-window task ids.
        var ob = _alloc(BB * (HH + 1) * MO)
        var ab = _alloc(BB * HH * A)
        var rb = _alloc(BB * HH)
        var db = _alloc(BB * HH)
        var tk = _alloc(BB)
        self.replay.sample_batch_task[BB, HH](ob, ab, rb, db, tk)

        # transpose to t-major.
        var ot = _alloc((HH + 1) * BB * MO)
        var at = _alloc(HH * BB * A)
        var rt = _alloc(HH * BB)
        var dt = _alloc(HH * BB)
        for b in range(BB):
            for t in range(HH + 1):
                for i in range(MO):
                    ot[(t * BB + b) * MO + i] = ob[(b * (HH + 1) + t) * MO + i]
            for t in range(HH):
                for j in range(A):
                    at[(t * BB + b) * A + j] = ab[(b * HH + t) * A + j]
                rt[t * BB + b] = rb[b * HH + t]
                dt[t * BB + b] = db[b * HH + t]

        # per-row task ids for the PB-batched policy encode/graph.
        var tk_pb = _alloc(PBP)
        for t in range(HH + 1):
            for b in range(BB):
                tk_pb[t * BB + b] = tk[b]

        var td = _alloc(HH * BB)
        var ta = Int(random_float64() * Float64(NQ))
        if ta >= NQ:
            ta = NQ - 1
        var tb = (ta + 1) % NQ
        var pa = Int(random_float64() * Float64(NQ))
        if pa >= NQ:
            pa = NQ - 1
        var pb = (pa + 1) % NQ

        # bracket the embedding grad accumulation around the whole step.
        self.task_emb.zero_grad[tg]()

        comptime if tg == "cpu":
            self.td_step.step[tg](
                self.encoder, self.policy, self.qt, self.task_emb, ta, tb,
                ot, rt, dt, td, tk, self.gamma,
            )
            var wl = self.wm_step.step[tg](
                self.wm_graph, self.encoder, self.dynamics, self.reward, self.q,
                self.termination, self.task_emb,
                self.enc_opt, self.dyn_opt, self.rew_opt, self.q_opt,
                self.term_opt,
                ot, at, rt, td, dt, tk,
            )
            self._last_cons = wl.consistency
            self._last_rew = wl.reward
            self._last_val = wl.value
            self._last_term = wl.termination
            self._last_wm = wl.total()
            # policy encode on [ot|task_emb] (PB-batched).
            var tem_pb = _alloc(PBP * EMB)
            self.task_emb.gather[tg, PBP](tk_pb, tem_pb)
            var oaug = _alloc(PBP * AOBS)
            for row in range(PBP):
                for i in range(MO):
                    oaug[row * AOBS + i] = ot[row * MO + i]
                for e in range(EMB):
                    oaug[row * AOBS + MO + e] = tem_pb[row * EMB + e]
            var zpol = _alloc(PBP * LAT)
            var zpol_t = TileTensor(zpol, row_major[PBP, LAT]())
            self.encoder.forward[tg, PBP](
                TileTensor(oaug, row_major[PBP, AOBS]()), output=zpol_t,
            )
            self._last_pi = self.pol_step.step[tg](
                self.policy, self.q, pa, pb, self.pi_opt, self.task_emb,
                zpol, tk_pb,
            )
            zpol.free(); tem_pb.free(); oaug.free()
            self.task_emb.step[tg]()
            for i in range(NQ):
                polyak_module[tg, Self.QNetT](self.q[i], self.qt[i], self.tau)
        else:
            var ctx = self.ctx.value()
            self.td_step.step[tg](
                self.encoder, self.policy, self.qt, self.task_emb, ta, tb,
                ot, rt, dt, td, tk, self.gamma, ctx=ctx,
            )
            var wl = self.wm_step.step[tg](
                self.wm_graph, self.encoder, self.dynamics, self.reward, self.q,
                self.termination, self.task_emb,
                self.enc_opt, self.dyn_opt, self.rew_opt, self.q_opt,
                self.term_opt,
                ot, at, rt, td, dt, tk, ctx=ctx,
            )
            self._last_cons = wl.consistency
            self._last_rew = wl.reward
            self._last_val = wl.value
            self._last_term = wl.termination
            self._last_wm = wl.total()
            var d_ot = _upload(ctx, ot, PBP * MO)
            var d_tkpb = _upload(ctx, tk_pb, PBP)
            var d_tempb = ctx.enqueue_create_buffer[DT](PBP * EMB)
            self.task_emb.gather[tg, PBP](_dp(d_tkpb), _dp(d_tempb), ctx=ctx)
            var d_oaug = ctx.enqueue_create_buffer[DT](PBP * AOBS)
            comptime oaug_k = _cat2_k[PBP, MO, EMB]
            comptime nb_oa = (PBP * AOBS + 255) // 256
            ctx.enqueue_function[oaug_k](
                _ltn[PBP * MO](_dp(d_ot)), _ltn[PBP * EMB](_dp(d_tempb)),
                _ltn[PBP * AOBS](_dp(d_oaug)), grid_dim=nb_oa, block_dim=256,
            )
            var d_zpol = ctx.enqueue_create_buffer[DT](PBP * LAT)
            var zpol_t = TileTensor(_dp(d_zpol), row_major[PBP, LAT]())
            self.encoder.forward[tg, PBP](
                TileTensor(_dp(d_oaug), row_major[PBP, AOBS]()), output=zpol_t,
            )
            self._last_pi = self.pol_step.step[tg](
                self.policy, self.q, pa, pb, self.pi_opt, self.task_emb,
                _dp(d_zpol), _dp(d_tkpb), ctx=ctx,
            )
            self.task_emb.step[tg]()
            for i in range(NQ):
                polyak_module[tg, Self.QNetT](
                    self.q[i], self.qt[i], self.tau, ctx=ctx
                )

        var td_sum: Scalar[DT] = 0.0
        var td_mn = td[0]
        var td_mx = td[0]
        for i in range(HH * BB):
            var v = td[i]
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

        ob.free(); ab.free(); rb.free(); db.free(); tk.free()
        ot.free(); at.free(); rt.free(); dt.free(); td.free(); tk_pb.free()
        return True


@always_inline
def _lt1[N: Int](
    p: UnsafePointer[Scalar[DT], MutAnyOrigin]
) -> LayoutTensor[DT, Layout.row_major(N), MutAnyOrigin]:
    return LayoutTensor[DT, Layout.row_major(N), MutAnyOrigin](p)


@always_inline
def _ltn[N: Int](
    p: UnsafePointer[Scalar[DT], MutAnyOrigin]
) -> LayoutTensor[DT, Layout.row_major(N), MutAnyOrigin]:
    return LayoutTensor[DT, Layout.row_major(N), MutAnyOrigin](p)
