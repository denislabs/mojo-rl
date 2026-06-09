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
from mojo_rl.nn2.core.module import mptr
from mojo_rl.nn2.initializer import Kaiming, Zero
from mojo_rl.nn2.optimizer.adam import Adam
from layout import Layout, LayoutTensor, TileTensor, row_major
from std.gpu.host import DeviceContext, DeviceBuffer

from mojo_rl.deep_agents2.primitives.rsample import RSample
from mojo_rl.deep_agents2.dreamerv3.polyak import polyak_module
from mojo_rl.deep_agents2.data.sequence_replay import SequenceReplay
from mojo_rl.planners.trajectory.mppi import MPPIGPUBatched
from mojo_rl.nn2.core.checkpoint import (
    save_state_v2_body, load_state_v2_body,
    save_state_v2_body_gpu, load_state_v2_body_gpu,
)
from mojo_rl.deep_agents2.core.checkpoint_helpers import (
    save_optimizer_v2_body, load_optimizer_v2_body,
    save_optimizer_v2_body_gpu, load_optimizer_v2_body_gpu,
    split_lines_v2, read_file_v2, expect_v2_header,
)
from mojo_rl.core.logger import Logger
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
    # MPC (MPPIGPUBatched) planning config — defaults match the reference.
    # Used only by select_action_mpc (GPU). Existing instantiations that omit
    # these get the reference values.
    NUM_SAMPLES: Int = 512,
    NUM_PI_TRAJS: Int = 24,
    NUM_ELITES: Int = 64,
    NUM_ITERS: Int = 6,
    # Q-trunk dropout prob (item D, §14.4). 0.0 = always-on no-op (bit-identical
    # default); >0 enables the experimental Q-net dropout (see nets.mojo caveats).
    QP: Float64 = 0.0,
](Movable & ImplicitlyDestructible):
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
    # MPC: single-env (N_ENVS=1) batched planner + its rollout callback.
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
    var q: List[Self.QNetT]
    var qt: List[Self.QNetT]
    var policy: Self.PolicyT
    # Termination head (item B). Always present; trains only when bce_coef > 0.
    var termination: Self.TermT

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
    var act_rsample: RSample[Self.ACT]
    var replay: SequenceReplay[Self.OBS, Self.ACT, Self.CAP]

    var gamma: Scalar[DT]
    var tau: Scalar[DT]
    # Termination BCE coefficient (item B): 0 = non-episodic (bit-identical);
    # >0 trains the termination head on episodic envs (Hopper/Walker/Humanoid).
    var bce_coef: Scalar[DT]
    var action_scale: Scalar[DT]
    var learning_starts: Int
    var step_count: Int
    var _last_wm: Scalar[DT]
    var _last_pi: Scalar[DT]
    # per-component last + diag-window accumulators (drained by flush_metrics).
    var _last_cons: Scalar[DT]
    var _last_rew: Scalar[DT]
    var _last_val: Scalar[DT]
    var _last_term: Scalar[DT]
    var _cons_acc: Scalar[DT]
    var _rew_acc: Scalar[DT]
    var _val_acc: Scalar[DT]
    var _term_acc: Scalar[DT]
    var _pi_acc: Scalar[DT]
    # Q + TD-target diagnostics (means window-averaged; min/max last-step).
    var _q_mean_acc: Scalar[DT]
    var _q_min_last: Scalar[DT]
    var _q_max_last: Scalar[DT]
    var _td_mean_acc: Scalar[DT]
    var _td_min_last: Scalar[DT]
    var _td_max_last: Scalar[DT]
    var _n_diag: Int
    var ctx: Optional[DeviceContext]
    # MPC planner (persistent warm-start; None on CPU). The rollout callback
    # is built transiently in select_action_mpc (it points at self's modules,
    # valid only during the call — never stored, to avoid self-pointer hazards).
    var planner: Optional[Self.PlannerT]
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

        # Q-dropout (item D): target-Q nets are used only for stop-grad targets
        # (td_target) + the MPPI terminal bootstrap (callback) — both eval
        # contexts, so force their Dropout to eval mode (no masking). Online Q
        # keeps training=True (drops in the WM/policy training graphs, matching
        # the reference, which keeps the model in train mode through update()).
        # Skipped entirely at QP=0.0 to leave the bit-identical default untouched.
        comptime if Self.QP > 0.0:
            for i in range(NQ):
                qt[i].set_attr["training"](Scalar[DT](0.0))

        # Termination head (item B). RNG-discipline for a *truly* bit-identical
        # off-path: Kaiming init draws from the global RNG (initializers.mojo),
        # which would shift the downstream warmup/exploration stream. So at
        # bce_coef=0 (non-episodic) build it with Zero init — no RNG draw, no
        # gradient, fully inert → the encoder/dynamics/reward/Q/policy AND the
        # rollout RNG are bit-identical to the pre-item-B agent. Only when
        # bce_coef>0 (an episodic run, where HalfCheetah parity is moot) do we
        # Kaiming-init so the head can actually learn state-dependent
        # termination. Built last so it never perturbs the other nets' init.
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

        var ar = RSample[Self.ACT].make[tg, INIT=Zero](ctx=ctx)
        ar.action_scale = action_scale

        var planner: Optional[Self.PlannerT] = None
        comptime if tg == "gpu":
            planner = Self.PlannerT(ctx.value())

        return Self(
            encoder=enc^, dynamics=dyn^, reward=rew^, q=q^, qt=qt^, policy=pol^,
            termination=term^,
            enc_opt=enc_opt^, dyn_opt=dyn_opt^, rew_opt=rew_opt^, q_opt=q_opt^,
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

    def mpc_start_episode(mut self) raises:
        """Reset the MPC planner's warm-start state (call on env reset)."""
        comptime if Self.target == "gpu":
            self.planner.value().start_episode(0)

    def select_action_mpc(
        mut self,
        obs: UnsafePointer[Scalar[DT], MutAnyOrigin],
        act_out: UnsafePointer[Scalar[DT], MutAnyOrigin],
        explore: Bool = True,
    ) raises:
        """MPC acting: plan in latent space via MPPIGPUBatched (single env).
        GPU only — the per-sample CPU planner is too slow for acting."""
        comptime assert Self.target == "gpu", (
            "select_action_mpc requires target='gpu' (CPU MPPI is eval-only)"
        )
        comptime A = Self.ACT
        comptime LAT = Self.LATENT
        var ctx = self.ctx.value()

        # encode obs → z0 [1, LATENT] on device
        var d_obs = _upload(ctx, obs, Self.OBS)
        var d_z0 = ctx.enqueue_create_buffer[DT](LAT)
        var z0_t = TileTensor(_dp(d_z0), row_major[1, LAT]())
        self.encoder.forward[Self.target, 1](
            TileTensor(_dp(d_obs), row_major[1, Self.OBS]()), output=z0_t,
        )

        # transient callback over self's modules (uses TARGET Q for the
        # terminal bootstrap; never stored → no self-pointer hazard).
        var cb = Self.MpcCB.make(
            self.dynamics, self.reward, self.policy, self.qt,
            self.action_scale, ctx,
        )

        var d_out = ctx.enqueue_create_buffer[DT](A)
        var z0_lt = LayoutTensor[DT, Layout.row_major(1, LAT), MutAnyOrigin](
            _dp(d_z0)
        )
        var out_lt = LayoutTensor[DT, Layout.row_major(1 * A), MutAnyOrigin](
            _dp(d_out)
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

    # ── Metrics: drain the diag-window accumulators into a bundle and, if a
    #    logger is wired, stream one log_scalar per field (driver cadence). ──
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
            # Names follow the dashboard's KNOWN_GROUPS conventions:
            #   reward_loss → World Model Losses; value_loss → Critic Loss;
            #   wm_loss → Loss; policy_loss → Policy Loss; pi_scale → Policy
            #   Scale. consistency_loss is TD-MPC2-specific (ungrouped).
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
        # reset the chunk accumulators
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

    # ── Checkpointing (one-file nn2-ckpt v2 envelope) ──────────────────
    def save_state(mut self, path: String) raises:
        """Save every world-model module (encoder/dynamics/reward + online
        & target Q ensemble + policy) and its optimizer. running_scale is
        NOT saved (re-converges via its EMA in ~100 steps). Overwrites
        `path`."""
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
            # Termination head (item B) — appended LAST so pre-item-B
            # checkpoints (which lack it) still load via the guard below.
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
                save_optimizer_v2_body_gpu(
                    self.q_opt[i], body, "q_opt" + String(i)
                )
            save_state_v2_body_gpu(self.termination, body, "termination", c)
            save_optimizer_v2_body_gpu(self.term_opt, body, "term_opt")
        var content = String("nn2-ckpt v2\n") + body
        with open(path, "w") as f:
            f.write(content)

    def load_state(mut self, path: String) raises:
        """Inverse of `save_state` (online + target Q both restored)."""
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
                load_optimizer_v2_body(
                    self.q_opt[i], lines, idx, "q_opt" + String(i)
                )
            # Termination head — present only in item-B-era checkpoints; older
            # files end here, so guard on remaining lines (term stays at init,
            # which is fine when bce_coef=0).
            if idx < len(lines):
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
                load_state_v2_body_gpu(
                    self.qt[i], lines, idx, "qt" + String(i), c
                )
            load_optimizer_v2_body_gpu(self.enc_opt, lines, idx, "enc_opt")
            load_optimizer_v2_body_gpu(self.dyn_opt, lines, idx, "dyn_opt")
            load_optimizer_v2_body_gpu(self.rew_opt, lines, idx, "rew_opt")
            load_optimizer_v2_body_gpu(self.pi_opt, lines, idx, "pi_opt")
            for i in range(NQ):
                load_optimizer_v2_body_gpu(
                    self.q_opt[i], lines, idx, "q_opt" + String(i)
                )
            if idx < len(lines):
                load_state_v2_body_gpu(
                    self.termination, lines, idx, "termination", c
                )
                load_optimizer_v2_body_gpu(self.term_opt, lines, idx, "term_opt")

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
            var wl = self.wm_step.step[tg](
                self.wm_graph, self.encoder, self.dynamics, self.reward, self.q,
                self.termination,
                self.enc_opt, self.dyn_opt, self.rew_opt, self.q_opt,
                self.term_opt,
                ot, at, rt, td, dt,
            )
            self._last_cons = wl.consistency
            self._last_rew = wl.reward
            self._last_val = wl.value
            self._last_term = wl.termination
            self._last_wm = wl.total()
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
            var wl = self.wm_step.step[tg](
                self.wm_graph, self.encoder, self.dynamics, self.reward, self.q,
                self.termination,
                self.enc_opt, self.dyn_opt, self.rew_opt, self.q_opt,
                self.term_opt,
                ot, at, rt, td, dt, ctx=ctx,
            )
            self._last_cons = wl.consistency
            self._last_rew = wl.reward
            self._last_val = wl.value
            self._last_term = wl.termination
            self._last_wm = wl.total()
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

        # TD-target stats over the [H*B] targets (host in both paths).
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

        # diag-window accumulation (drained by flush_metrics).
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

        ob.free(); ab.free(); rb.free(); db.free()
        ot.free(); at.free(); rt.free(); dt.free()
        td.free()
        return True
