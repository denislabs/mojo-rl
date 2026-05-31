"""REDQTrainer — Phase R.3 (CPU).

Randomized Ensembled Double Q-learning (Chen et al., ICLR 2021).
Mirrors `SACTrainer`'s shape — same `[train_target, SAMPLE, ACTOR,
CRITIC]` family of comptime params plus `[N, N_MIN, UTD, POLICY_DELAY,
Q_MODE]` for the ensemble knobs — and runs a UTD inner critic loop
per `train_step` call:

    train_step(step_idx):                  # outer = 1 env step
        sample once (gates warmup)
        for inner = 0..UTD-1:
            resample subset                # Fisher-Yates of (N choose N_MIN)
            target-y     (EnsembleTargetYBlock)
            critic       (EnsembleCriticStep)
            polyak       (EnsemblePolyakStep — every inner step, paper-faithful)
            if (_inner_count % POLICY_DELAY) == 0:
                actor    (EnsembleActorStep — mean over N online critics)
                alpha    (AlphaUpdateStep — unchanged from SAC)

The actor/α delayed cadence and the per-inner polyak match the
paper-faithful schedule from `deep_agents/redq/redq.mojo`.

R.3 is CPU-only and ships the OffPolicyAgent trait surface
(select_action_batched / select_greedy_action / record /
record_batch_cpu / train_step / mean_return / add_complete_return)
so the existing CPU off-policy driver can train it unchanged. GPU,
config-driven presets, and the one-file v2 checkpoint follow up in
R.5 / a separate slice.
"""

from std.math import exp as fexp, tanh as ftanh
from std.random import random_float64
from std.time import perf_counter_ns
from std.gpu.host import DeviceContext
from layout import TileTensor, row_major

from mojo_rl.core.logger import Logger, NoOpLogger
from mojo_rl.nn2.constants import DT
from mojo_rl.nn2.core.amp import AMPPolicy, NoAMP
from mojo_rl.nn2.core.module import Module
from mojo_rl.nn2.core.scratch import Scratch
from mojo_rl.nn2.core.scratch_walkers import init_scratch_auto
from mojo_rl.nn2.core.metric import LogScalar
from mojo_rl.nn2.core.log_bundle import log_bundle
from mojo_rl.nn2.initializer import Xavier
from mojo_rl.nn2.optimizer.adam import Adam
from mojo_rl.nn2.optimizer.scalar_adam import ScalarAdam
from mojo_rl.nn2.training.timer import Timer

from ..training.episode_tracker import EpisodeTracker
from ..training.trainer_block import TrainerState
from ..training.driver_offpolicy import OffPolicyAgent
from ..training.blocks import SampleBlock
from ..sac.blocks.alpha_update_step import AlphaUpdateStep

from .ensemble import CriticEnsemble
from .ensemble_target_y_block import EnsembleTargetYBlock
from .blocks.ensemble_critic_step import EnsembleCriticStep
from .blocks.ensemble_actor_step import EnsembleActorStep
from .blocks.ensemble_polyak_step import EnsemblePolyakStep
from .metrics import REDQMetrics


struct REDQTrainer[
    train_target: StaticString,
    SAMPLE: SampleBlock,
    ACTOR: Module,
    CRITIC: Module,
    N: Int,
    N_MIN: Int,
    UTD: Int,
    POLICY_DELAY: Int,
    Q_MODE: Int,
](OffPolicyAgent):
    """REDQ Trainer. Dims (OBS / ACT / BATCH) derived from SAMPLE,
    so the user specifies them ONCE on the sample block type.

    `N` total online/target critics; `N_MIN` random subset size for
    MIN-mode TD target; `UTD` inner critic updates per env step;
    `POLICY_DELAY` actor + α update every K-th inner critic update;
    `Q_MODE` ∈ {0=MIN, 1=AVE} (REM is GPU-only — out of R.3 scope).
    """

    comptime OBS_DIM: Int = Self.SAMPLE.OBS
    comptime ACT_DIM: Int = Self.SAMPLE.ACT
    comptime BATCH: Int = Self.SAMPLE.BATCH

    # OffPolicyAgent trait aliases.
    comptime AGENT_OBS_DIM: Int = Self.OBS_DIM
    comptime AGENT_ACT_DIM: Int = Self.ACT_DIM
    comptime AGENT_TRAIN_TARGET: StaticString = Self.train_target

    # Timer section indices. Order matches `add_section` calls in `make`.
    comptime _T_SAMPLE = 0
    comptime _T_TARGET_Y = 1
    comptime _T_CRITIC = 2
    comptime _T_ACTOR = 3
    comptime _T_ALPHA = 4
    comptime _T_POLYAK = 5
    comptime _T_DIAG = 6

    # ─── Owned state ─────────────────────────────────────────────────
    var actor: Self.ACTOR
    var ensemble: CriticEnsemble[Self.CRITIC, Self.N]
    var actor_opt: Adam
    var alpha_opt: ScalarAdam

    var sample_blk: Self.SAMPLE
    var target_y_blk: EnsembleTargetYBlock[
        Self.ACTOR, Self.CRITIC, Self.N, Self.BATCH,
        Self.OBS_DIM, Self.ACT_DIM, Self.N_MIN, Self.Q_MODE,
    ]
    var critic_blk: EnsembleCriticStep[
        Self.CRITIC, Self.N, Self.OBS_DIM, Self.ACT_DIM, Self.BATCH,
    ]
    var actor_blk: EnsembleActorStep[
        Self.ACTOR, Self.CRITIC, Self.N,
        Self.OBS_DIM, Self.ACT_DIM, Self.BATCH,
    ]
    var alpha_blk: AlphaUpdateStep[
        Self.OBS_DIM, Self.ACT_DIM, Self.BATCH,
    ]
    var polyak_blk: EnsemblePolyakStep[
        Self.CRITIC, Self.N, Self.OBS_DIM, Self.ACT_DIM, Self.BATCH,
    ]

    var state: TrainerState[Self.OBS_DIM, Self.ACT_DIM, Self.BATCH]
    var tracker: EpisodeTracker
    var ctx: Optional[DeviceContext]

    # Single-env action-selection scratch (mirror SAC's _ob1 / _ao1 / _alp1).
    var _ob1: Scratch["ob1", Self.OBS_DIM, True]
    var _ao1: Scratch["ao1", 2 * Self.ACT_DIM, True]
    var _alp1: Scratch["alp1", Self.ACT_DIM + 1, True]

    var action_scale: Scalar[DT]
    var learning_starts: Int

    # UTD bookkeeping. `_inner_count` is the cumulative inner critic
    # update counter modulo POLICY_DELAY drives the actor cadence.
    var _inner_count: Int

    # Metric accumulators (drained on flush).
    var _actor_L_accum: Scalar[DT]
    var _critic_L_accum: Scalar[DT]
    var _alpha_accum: Scalar[DT]
    var _q_accum: Scalar[DT]
    var _target_accum: Scalar[DT]
    var _reward_accum: Scalar[DT]
    var _done_accum: Scalar[DT]
    var _abs_action_accum: Scalar[DT]
    var _update_count: Int          # inner steps this chunk
    var _actor_update_count: Int    # actor steps this chunk
    var _total_train_steps: Int     # cumulative inner steps (never reset)

    var timer: Timer

    def __init__(out self):
        self.actor = Self.ACTOR()
        self.ensemble = CriticEnsemble[Self.CRITIC, Self.N]()
        self.actor_opt = Adam()
        self.alpha_opt = ScalarAdam(
            value=0.0, m=0.0, v=0.0, t=0,
            lr=0.0003, beta1=0.9, beta2=0.999, eps=1e-8,
        )
        self.sample_blk = Self.SAMPLE()
        self.target_y_blk = EnsembleTargetYBlock[
            Self.ACTOR, Self.CRITIC, Self.N, Self.BATCH,
            Self.OBS_DIM, Self.ACT_DIM, Self.N_MIN, Self.Q_MODE,
        ]()
        self.critic_blk = EnsembleCriticStep[
            Self.CRITIC, Self.N, Self.OBS_DIM, Self.ACT_DIM, Self.BATCH,
        ]()
        self.actor_blk = EnsembleActorStep[
            Self.ACTOR, Self.CRITIC, Self.N,
            Self.OBS_DIM, Self.ACT_DIM, Self.BATCH,
        ]()
        self.alpha_blk = AlphaUpdateStep[
            Self.OBS_DIM, Self.ACT_DIM, Self.BATCH,
        ]()
        self.polyak_blk = EnsemblePolyakStep[
            Self.CRITIC, Self.N, Self.OBS_DIM, Self.ACT_DIM, Self.BATCH,
        ]()
        self.state = TrainerState[
            Self.OBS_DIM, Self.ACT_DIM, Self.BATCH,
        ]()
        self.tracker = EpisodeTracker(
            window=List[Scalar[DT]](),
            window_size=0,
            idx=0,
            current_return=Scalar[DT](0.0),
            ep_count=0,
        )
        self.ctx = None
        self._ob1 = Scratch["ob1", Self.OBS_DIM, True]()
        self._ao1 = Scratch["ao1", 2 * Self.ACT_DIM, True]()
        self._alp1 = Scratch["alp1", Self.ACT_DIM + 1, True]()
        self.action_scale = Scalar[DT](1.0)
        self.learning_starts = 1_000
        self._inner_count = 0
        self._actor_L_accum = Scalar[DT](0.0)
        self._critic_L_accum = Scalar[DT](0.0)
        self._alpha_accum = Scalar[DT](0.0)
        self._q_accum = Scalar[DT](0.0)
        self._target_accum = Scalar[DT](0.0)
        self._reward_accum = Scalar[DT](0.0)
        self._done_accum = Scalar[DT](0.0)
        self._abs_action_accum = Scalar[DT](0.0)
        self._update_count = 0
        self._actor_update_count = 0
        self._total_train_steps = 0
        self.timer = Timer.new()

    @staticmethod
    def make(
        ctx: Optional[DeviceContext] = None,
        actor_lr: Scalar[DT] = Scalar[DT](3e-4),
        critic_lr: Scalar[DT] = Scalar[DT](3e-4),
        alpha_lr: Scalar[DT] = Scalar[DT](3e-4),
        gamma: Scalar[DT] = Scalar[DT](0.99),
        tau: Scalar[DT] = Scalar[DT](0.005),
        action_scale: Scalar[DT] = Scalar[DT](1.0),
        init_alpha: Scalar[DT] = Scalar[DT](0.2),
        target_entropy: Scalar[DT] = Scalar[DT](-1.0),
        learning_starts: Int = 1_000,
        window_size: Int = 10,
        initial_episode_fill: Scalar[DT] = Scalar[DT](-1250.0),
        max_grad_norm: Scalar[DT] = Scalar[DT](0.0),
    ) raises -> Self:
        comptime assert Self.train_target == "cpu", (
            "REDQTrainer: R.3 supports CPU only — GPU follow-up."
        )
        comptime assert Self.N >= 2, "REDQ: N must be ≥ 2"
        comptime assert Self.N_MIN >= 1, "REDQ: N_MIN must be ≥ 1"
        comptime assert Self.N_MIN <= Self.N, "REDQ: N_MIN must be ≤ N"
        comptime assert Self.UTD >= 1, "REDQ: UTD must be ≥ 1"
        comptime assert Self.POLICY_DELAY >= 1, (
            "REDQ: POLICY_DELAY must be ≥ 1"
        )

        var t = Self()
        t.ctx = ctx

        t.actor = Self.ACTOR.make[Self.train_target, Xavier](ctx=ctx)
        t.ensemble = CriticEnsemble[Self.CRITIC, Self.N].make[
            Self.train_target, Xavier,
        ](ctx=ctx)
        t.actor_opt = Adam.make[Self.train_target, M=Self.ACTOR](
            t.actor, ctx=ctx,
        )
        t.alpha_opt = ScalarAdam.new(fexp_to_log(init_alpha), alpha_lr)
        # Apply LR to all N critic Adams (defaults already set inside
        # CriticEnsemble.make; this is the explicit user-tunable knob).
        for i in range(Self.N):
            t.ensemble.opts[i].lr = critic_lr
            t.ensemble.opts[i].max_grad_norm = max_grad_norm

        t.actor_opt.lr = actor_lr
        t.actor_opt.max_grad_norm = max_grad_norm

        t.target_y_blk = EnsembleTargetYBlock[
            Self.ACTOR, Self.CRITIC, Self.N, Self.BATCH,
            Self.OBS_DIM, Self.ACT_DIM, Self.N_MIN, Self.Q_MODE,
        ].make[Self.train_target](
            action_scale=action_scale, gamma=gamma, ctx=ctx,
        )
        t.critic_blk = EnsembleCriticStep[
            Self.CRITIC, Self.N, Self.OBS_DIM, Self.ACT_DIM, Self.BATCH,
        ].make[Self.train_target](ctx=ctx)
        t.actor_blk = EnsembleActorStep[
            Self.ACTOR, Self.CRITIC, Self.N,
            Self.OBS_DIM, Self.ACT_DIM, Self.BATCH,
        ].make[Self.train_target](
            action_scale=action_scale, ctx=ctx,
        )
        t.alpha_blk = AlphaUpdateStep[
            Self.OBS_DIM, Self.ACT_DIM, Self.BATCH,
        ].make(target_entropy=target_entropy)
        t.polyak_blk = EnsemblePolyakStep[
            Self.CRITIC, Self.N, Self.OBS_DIM, Self.ACT_DIM, Self.BATCH,
        ].make(tau=tau)

        t.state = TrainerState[
            Self.OBS_DIM, Self.ACT_DIM, Self.BATCH,
        ].make[Self.train_target](ctx=ctx)

        t.tracker = EpisodeTracker.new(
            window_size=window_size, initial_fill=initial_episode_fill,
        )

        init_scratch_auto[Self, target=Self.train_target](t, ctx)

        t.action_scale = action_scale
        t.learning_starts = learning_starts

        # Sample block lifecycle.
        t.sample_blk.setup(learning_starts, ctx=ctx)

        # Timer sections (order must match the `_T_*` aliases).
        t.timer.add_section("sample")
        t.timer.add_section("target_y")
        t.timer.add_section("critic")
        t.timer.add_section("actor")
        t.timer.add_section("alpha")
        t.timer.add_section("polyak")
        t.timer.add_section("diag")
        return t^

    # ────────────────────────────────────────────────────────────────
    # Inner step body — one (target_y + critic + polyak + maybe actor).
    # ────────────────────────────────────────────────────────────────

    def _run_inner_step[POLICY: AMPPolicy = NoAMP](mut self) raises:
        """Single UTD inner step. Assumes `self.state` already has a
        fresh minibatch loaded by `sample_blk.step` above."""
        self._inner_count += 1

        # Subset resample (Fisher-Yates over {0..N-1}, length N_MIN).
        # MODE=AVE ignores it but the call is harmless.
        self.target_y_blk.resample_subset_idxs()

        # Target Y. CPU bakes α as a host scalar; GPU would wire a
        # device buffer once at make-time (R.5).
        var t_ty = perf_counter_ns()
        var alpha_val = fexp(self.alpha_opt.value)
        self.state.alpha = alpha_val
        self.target_y_blk.step["cpu", POLICY](
            self.actor,
            self.ensemble,
            self.state.mb_sp.cpu_ptr(),
            self.state.mb_r.cpu_ptr(),
            self.state.mb_d.cpu_ptr(),
            alpha_val,
            self.state.mb_y.cpu_ptr(),
        )
        self.timer.accumulate(Self._T_TARGET_Y, t_ty)

        # Critic update (N forward+vjp+Adam.step against shared y).
        var t_crit = perf_counter_ns()
        self.critic_blk.step["cpu", POLICY](self.state, self.ensemble)
        self.timer.accumulate(Self._T_CRITIC, t_crit)

        # Polyak ALL N targets every inner step (paper-faithful).
        var t_pol = perf_counter_ns()
        self.polyak_blk.step["cpu"](self.state, self.ensemble)
        self.timer.accumulate(Self._T_POLYAK, t_pol)

        # Actor + α every POLICY_DELAY inner critic updates. First fire
        # at _inner_count == POLICY_DELAY, then 2·POLICY_DELAY, …
        # Matches legacy `(critic_update_count % POL_DELAY == 0)`.
        if self._inner_count % Self.POLICY_DELAY == 0:
            var t_act = perf_counter_ns()
            self.actor_blk.step["cpu", POLICY](
                self.state, self.actor, self.actor_opt, self.ensemble,
            )
            self.timer.accumulate(Self._T_ACTOR, t_act)

            var t_alp = perf_counter_ns()
            self.alpha_blk.step["cpu"](self.state, self.alpha_opt)
            self.timer.accumulate(Self._T_ALPHA, t_alp)

            self._actor_L_accum += self.state.actor_loss
            self._alpha_accum += fexp(self.alpha_opt.value)
            self._actor_update_count += 1

        # Per-batch diagnostic accumulators (cheap; CPU-only).
        self._accumulate_diag()

        self._critic_L_accum += self.state.critic_loss
        self._update_count += 1
        self._total_train_steps += 1

    def _accumulate_diag(mut self):
        """Per-inner-step host-walk over the batch scratches:
        mean_q (last critic's Q), mean_target, mean_reward, mean_done,
        mean_abs_action. Matches the SAC `_T_DIAG` walk shape."""
        var t_diag = perf_counter_ns()
        var inv_b: Scalar[DT] = Scalar[DT](1.0) / Scalar[DT](Self.BATCH)
        var y_p = self.state.mb_y.cpu_ptr()
        var r_p = self.state.mb_r.cpu_ptr()
        var d_p = self.state.mb_d.cpu_ptr()
        var a_p = self.state.mb_a.cpu_ptr()
        # Last-critic Q (one CriticUpdateBlock reused across N critics —
        # `_mb_q` carries the last iter's `Q_{N-1}(s, a)`).
        var q_p = self.critic_blk.member_step._mb_q.cpu_ptr()
        var sum_y: Scalar[DT] = 0.0
        var sum_r: Scalar[DT] = 0.0
        var sum_d: Scalar[DT] = 0.0
        var sum_q: Scalar[DT] = 0.0
        for i in range(Self.BATCH):
            sum_y += y_p[i]
            sum_r += r_p[i]
            sum_d += d_p[i]
            sum_q += q_p[i]
        var sum_a: Scalar[DT] = 0.0
        for i in range(Self.BATCH * Self.ACT_DIM):
            var av = a_p[i]
            sum_a += av if av >= Scalar[DT](0.0) else -av
        self._q_accum += sum_q * inv_b
        self._target_accum += sum_y * inv_b
        self._reward_accum += sum_r * inv_b
        self._done_accum += sum_d * inv_b
        self._abs_action_accum += sum_a * (
            Scalar[DT](1.0) / Scalar[DT](Self.BATCH * Self.ACT_DIM)
        )
        self.timer.accumulate(Self._T_DIAG, t_diag)

    # ────────────────────────────────────────────────────────────────
    # train_step — outer (one env step). Runs UTD inner critic updates.
    # ────────────────────────────────────────────────────────────────

    def train_step(mut self, step_idx: Int) raises -> Bool:
        """One outer training tick. Samples + runs UTD inner critic
        updates. Returns True if any inner update ran (False during
        warmup / when buffer too small). The trainer's
        `_total_train_steps` increments by UTD when this returns True."""
        self.state.step_idx = step_idx
        self.state.did_step = True

        # Sample #1 — gates warmup and buffer-readiness.
        var t_s0 = perf_counter_ns()
        self.sample_blk.step(self.state)
        self.timer.accumulate(Self._T_SAMPLE, t_s0)
        if not self.state.did_step:
            return False

        self._run_inner_step[NoAMP]()

        # Inner steps 2..UTD. Re-sample each time.
        for _ in range(Self.UTD - 1):
            var t_s = perf_counter_ns()
            self.state.did_step = True
            self.sample_blk.step(self.state)
            self.timer.accumulate(Self._T_SAMPLE, t_s)
            if not self.state.did_step:
                break  # buffer drained mid-iter (single-env: never)
            self._run_inner_step[NoAMP]()

        return True

    # ────────────────────────────────────────────────────────────────
    # Action-selection surface (OffPolicyAgent trait).
    # ────────────────────────────────────────────────────────────────

    def select_action_batched[
        N_ENVS: Int,
    ](
        mut self,
        obs_ptr: UnsafePointer[Scalar[DT], MutAnyOrigin],
        action_ptr: UnsafePointer[Scalar[DT], MutAnyOrigin],
        ao_scratch_ptr: UnsafePointer[Scalar[DT], MutAnyOrigin],
        alp_scratch_ptr: UnsafePointer[Scalar[DT], MutAnyOrigin],
        step_idx: Int,
    ) raises:
        """Single batched entry. CPU only for R.3."""
        comptime assert N_ENVS > 0, "N_ENVS must be > 0"
        comptime ACT = Self.ACT_DIM
        comptime OBS = Self.OBS_DIM

        if step_idx < self.learning_starts:
            # Uniform random in [-scale, scale].
            for i in range(N_ENVS * ACT):
                var u = Scalar[DT](2.0 * random_float64() - 1.0)
                action_ptr[i] = u * self.action_scale
            return

        # actor.forward(s) → ao [N_ENVS, 2*ACT]
        var obs_t = TileTensor(obs_ptr, row_major[N_ENVS, OBS]())
        var ao_t = TileTensor(ao_scratch_ptr, row_major[N_ENVS, 2 * ACT]())
        var alp_t = TileTensor(alp_scratch_ptr, row_major[N_ENVS, ACT + 1]())
        self.actor.forward["cpu", N_ENVS](obs_t, output=ao_t)
        self.actor_blk.inner.rsample.forward["cpu", N_ENVS](
            ao_t, output=alp_t,
        )
        # Clamp action.
        for env in range(N_ENVS):
            var src = alp_scratch_ptr + env * (ACT + 1)
            var dst = action_ptr + env * ACT
            for j in range(ACT):
                var a = src[j]
                if a > self.action_scale:
                    a = self.action_scale
                elif a < -self.action_scale:
                    a = -self.action_scale
                dst[j] = a

    def select_action(
        mut self,
        ref obs: List[Scalar[DT]],
        mut action_out: List[Scalar[DT]],
        step_idx: Int,
    ) raises:
        """Host-list single-env wrapper (smoke-test/eval-loop callers)."""
        var ob1_p = self._ob1.cpu_ptr()
        for d in range(Self.OBS_DIM):
            ob1_p[d] = obs[d]
        self.select_action_batched[1](
            ob1_p,
            self._alp1.cpu_ptr(),  # action_ptr — alias safe at N_ENVS=1
            self._ao1.cpu_ptr(),
            self._alp1.cpu_ptr(),
            step_idx,
        )
        var alp_p = self._alp1.cpu_ptr()
        for j in range(Self.ACT_DIM):
            action_out[j] = alp_p[j]

    def select_greedy_action(
        mut self,
        ref obs: List[Scalar[DT]],
        mut action_out: List[Scalar[DT]],
    ) raises:
        """Deterministic eval: `action = tanh(actor.forward(s).mean) ·
        action_scale`, clamped."""
        var ob1_p = self._ob1.cpu_ptr()
        var ao1_p = self._ao1.cpu_ptr()
        for d in range(Self.OBS_DIM):
            ob1_p[d] = obs[d]
        var ob1_t = TileTensor(ob1_p, row_major[1, Self.OBS_DIM]())
        var ao1_t = TileTensor(ao1_p, row_major[1, 2 * Self.ACT_DIM]())
        self.actor.forward["cpu", 1](ob1_t, output=ao1_t)
        for j in range(Self.ACT_DIM):
            var mean = ao1_p[j]
            var a = ftanh(mean) * self.action_scale
            if a > self.action_scale:
                a = self.action_scale
            elif a < -self.action_scale:
                a = -self.action_scale
            action_out[j] = a

    # ────────────────────────────────────────────────────────────────
    # Replay-push surface.
    # ────────────────────────────────────────────────────────────────

    def record(
        mut self,
        ref obs: List[Scalar[DT]],
        ref action: List[Scalar[DT]],
        reward: Scalar[DT],
        ref next_obs: List[Scalar[DT]],
        done: Scalar[DT],
    ) raises:
        self.tracker.add_reward(reward)
        self.sample_blk.add(
            obs, action, reward, next_obs, done, ctx=self.ctx,
        )

    def record_batch_cpu[
        N_ENVS: Int,
    ](
        mut self,
        prev_obs_ptr: UnsafePointer[Scalar[DT], MutAnyOrigin],
        action_ptr: UnsafePointer[Scalar[DT], MutAnyOrigin],
        reward_ptr: UnsafePointer[Scalar[DT], MutAnyOrigin],
        next_obs_ptr: UnsafePointer[Scalar[DT], MutAnyOrigin],
        done_ptr: UnsafePointer[Scalar[DT], MutAnyOrigin],
    ) raises:
        """Per-lane SAC-style replay push without tracker.add_reward
        (the driver manages per-env return accumulators)."""
        comptime OBS = Self.OBS_DIM
        comptime ACT = Self.ACT_DIM
        var obs_lane = List[Scalar[DT]](length=OBS, fill=Scalar[DT](0.0))
        var act_lane = List[Scalar[DT]](length=ACT, fill=Scalar[DT](0.0))
        var nxt_lane = List[Scalar[DT]](length=OBS, fill=Scalar[DT](0.0))
        for env_idx in range(N_ENVS):
            for d in range(OBS):
                obs_lane[d] = prev_obs_ptr[env_idx * OBS + d]
                nxt_lane[d] = next_obs_ptr[env_idx * OBS + d]
            for j in range(ACT):
                act_lane[j] = action_ptr[env_idx * ACT + j]
            self.sample_blk.add(
                obs_lane, act_lane, reward_ptr[env_idx], nxt_lane,
                done_ptr[env_idx], ctx=self.ctx,
            )

    # ────────────────────────────────────────────────────────────────
    # Tracker passthroughs.
    # ────────────────────────────────────────────────────────────────

    def end_episode(mut self):
        self.tracker.end_episode()

    def mean_return(self) -> Scalar[DT]:
        return self.tracker.mean_return()

    def ep_count(self) -> Int:
        return self.tracker.ep_count

    def add_complete_return(mut self, ret: Scalar[DT]):
        self.tracker.add_complete_return(ret)

    # ────────────────────────────────────────────────────────────────
    # Metrics surface.
    # ────────────────────────────────────────────────────────────────

    def total_train_steps(self) -> Int:
        """Cumulative INNER train steps. One env-step contributes UTD
        of these when past warmup. Used as the `train_steps` metric."""
        return self._total_train_steps

    def flush_metrics[
        L: Logger = NoOpLogger,
    ](
        mut self,
        logger: Optional[UnsafePointer[L, MutAnyOrigin]] = None,
        step: Int = 0,
    ) raises -> REDQMetrics:
        """Drain inner-step accumulators into a REDQMetrics bundle.
        Critic_loss / mean_* are averaged over `_update_count` inner
        steps; actor_loss / alpha are averaged over `_actor_update_count`
        actor steps (smaller — actor fires every POLICY_DELAY inner
        steps)."""
        var n = self._update_count if self._update_count > 0 else 1
        var inv = Scalar[DT](1.0) / Scalar[DT](n)
        var na = self._actor_update_count if self._actor_update_count > 0 else 1
        var inv_a = Scalar[DT](1.0) / Scalar[DT](na)

        var bundle = REDQMetrics(
            actor_loss=LogScalar[DT](self._actor_L_accum * inv_a),
            critic_loss=LogScalar[DT](self._critic_L_accum * inv),
            alpha=LogScalar[DT](self._alpha_accum * inv_a),
            mean_q=LogScalar[DT](self._q_accum * inv),
            mean_target=LogScalar[DT](self._target_accum * inv),
            mean_reward=LogScalar[DT](self._reward_accum * inv),
            mean_next_q=LogScalar[DT](Scalar[DT](0.0)),
            mean_done=LogScalar[DT](self._done_accum * inv),
            mean_abs_action=LogScalar[DT](self._abs_action_accum * inv),
            train_steps=LogScalar[DT](Scalar[DT](self._total_train_steps)),
            n_updates=LogScalar[DT](Scalar[DT](self._update_count)),
        )
        self._actor_L_accum = Scalar[DT](0.0)
        self._critic_L_accum = Scalar[DT](0.0)
        self._alpha_accum = Scalar[DT](0.0)
        self._q_accum = Scalar[DT](0.0)
        self._target_accum = Scalar[DT](0.0)
        self._reward_accum = Scalar[DT](0.0)
        self._done_accum = Scalar[DT](0.0)
        self._abs_action_accum = Scalar[DT](0.0)
        self._update_count = 0
        self._actor_update_count = 0
        if Bool(logger):
            log_bundle(logger.value()[], bundle, step)
        return bundle^

    def flush_metrics_through_logger[L: Logger](
        mut self,
        logger: Optional[UnsafePointer[L, MutAnyOrigin]],
        step: Int,
    ) raises:
        """Trait-uniform passthrough — drains the bundle via
        `flush_metrics` (which logs to `logger`) and discards the
        typed return."""
        _ = self.flush_metrics[L](logger, step)

    def flush_timer_log(mut self) -> String:
        var report = self.timer.format_report()
        self.timer.reset()
        return report


# ────────────────────────────────────────────────────────────────────
# Helpers
# ────────────────────────────────────────────────────────────────────


def fexp_to_log(alpha: Scalar[DT]) -> Scalar[DT]:
    """log(α) — for seeding ScalarAdam.value (which holds log_α). Wrapper
    around std.math.log; named to avoid clashing with the `flog` import
    used elsewhere in the module."""
    from std.math import log as _flog
    return _flog(alpha)
