"""PPODiscreteTrainer — block-composed categorical PPO trainer.

Discrete sibling of `ppo/trainer.mojo`. Same 6-block composition over
`OnPolicyState`, but the policy is a softmax categorical over
`N_ACTIONS` logits instead of a diagonal Gaussian:

  PPODiscreteActStep      — per env-step: actor.forward → logits +
                            categorical sample + critic.forward
  PPORecordStep           — per env-step: push cached → rollout buffer
  PPOGAEStep              — per rollout:  bootstrap + per-env GAE
  PPOMinibatchGatherStep  — per epoch:    Fisher-Yates shuffle
                            per minibatch: gather + normalise mb_adv
  PPODiscreteActorTrainStep — per minibatch: categorical clipped surrogate
  PPOCriticTrainStep      — per minibatch: critic MSE update

The GAE / gather / record / critic blocks are SHARED verbatim with the
continuous trainer (they are policy-agnostic and only need ACT=1 — the
action slot holds a discrete index). The act-step and actor-loss are
the only discrete-specific blocks.

Single-env (host-list) surface only — conforms to
`OnPolicyDiscreteAgent`, consumed by `run_onpolicy_discrete_train`.
The batched discrete on-policy driver is deferred. Dual-target
(CPU/GPU via `train_target`): per-step actor/critic forwards run on
device (the act-step H2Ds obs and D2Hs logits/value); rollout buffers
live host-only; the K-epoch minibatch is H2D-uploaded before each
train step (same hybrid scheme as the continuous trainer).
"""

from std.gpu import global_idx
from std.gpu.host import DeviceContext, DeviceBuffer
from std.math import exp as fexp, log as flog
from std.memory import alloc
from std.time import perf_counter_ns
from layout import TileTensor, row_major

from mojo_rl.core.logger import Logger, NoOpLogger
from mojo_rl.nn2.constants import DT, TPB, TPB_REDUCE
from ..training.device_mean_accum import DeviceMeanAccum
from ..ppo.trainer import _ppo_ev_kernel
from mojo_rl.nn2.core import Module
from mojo_rl.nn2.core.checkpoint import (
    save_state_v2_body, load_state_v2_body,
    save_state_v2_body_gpu, load_state_v2_body_gpu,
)
from mojo_rl.nn2.core.log_bundle import log_bundle
from mojo_rl.nn2.core.metric import LogScalar
from mojo_rl.nn2.initializer import Xavier
from mojo_rl.nn2.optimizer.adam import Adam
from mojo_rl.nn2.training.timer import Timer
from ..core.checkpoint_helpers import (
    save_optimizer_v2_body, load_optimizer_v2_body,
    save_optimizer_v2_body_gpu, load_optimizer_v2_body_gpu,
    split_lines_v2, read_file_v2, expect_v2_header,
)
from ..training.episode_tracker import EpisodeTracker
from ..training.onpolicy_state import OnPolicyState
from ..training.driver_onpolicy_discrete import OnPolicyDiscreteAgent
from ..ppo.blocks.record_step import PPORecordStep
from ..ppo.blocks.gae_step import PPOGAEStep
from ..ppo.blocks.minibatch_gather_step import PPOMinibatchGatherStep
from ..ppo.blocks.critic_train_step import PPOCriticTrainStep
from ..ppo.metrics import PPOMetrics
from .blocks.act_step import PPODiscreteActStep
from .blocks.actor_train_step import PPODiscreteActorTrainStep


comptime _DIAG_LOG_PROB_DIFF_MAX: Scalar[DT] = 20.0


# ──────────────────────────────────────────────────────────────────────────
# GPU diag kernel — one thread per minibatch row; mirrors the CPU walk in
# `_accumulate_diag` (log-softmax entropy + log-prob at the taken index +
# Schulman-2020 KL + clip indicator). Writes per-sample ent / kl / clip into
# three `[MB]` buffers; explained variance reuses `ppo.trainer._ppo_ev_kernel`.
# ──────────────────────────────────────────────────────────────────────────
def _ppo_discrete_diag_kernel[MB: Int, N: Int](
    logits: UnsafePointer[Scalar[DT], MutAnyOrigin],
    act: UnsafePointer[Scalar[DT], MutAnyOrigin],
    olp: UnsafePointer[Scalar[DT], MutAnyOrigin],
    clip_eps: Scalar[DT],
    ent_out: UnsafePointer[Scalar[DT], MutAnyOrigin],
    kl_out: UnsafePointer[Scalar[DT], MutAnyOrigin],
    clip_out: UnsafePointer[Scalar[DT], MutAnyOrigin],
):
    var b = Int(global_idx.x)
    if b >= MB:
        return
    var base = b * N
    var max_l = logits[base]
    for j in range(1, N):
        if logits[base + j] > max_l:
            max_l = logits[base + j]
    var sum_exp: Scalar[DT] = 0.0
    for j in range(N):
        sum_exp += fexp(logits[base + j] - max_l)
    var log_sum = flog(sum_exp)
    var ent: Scalar[DT] = 0.0
    for j in range(N):
        var lp_j = (logits[base + j] - max_l) - log_sum
        var p_j = fexp(lp_j)
        ent += -p_j * lp_j
    var a_idx = Int(act[b])
    var nlp = (logits[base + a_idx] - max_l) - log_sum
    var diff = nlp - olp[b]
    if diff > _DIAG_LOG_PROB_DIFF_MAX:
        diff = _DIAG_LOG_PROB_DIFF_MAX
    elif diff < -_DIAG_LOG_PROB_DIFF_MAX:
        diff = -_DIAG_LOG_PROB_DIFF_MAX
    var ratio = fexp(diff)
    kl_out[b] = (ratio - Scalar[DT](1.0)) - diff
    var dev = ratio - Scalar[DT](1.0)
    if dev < Scalar[DT](0.0):
        dev = -dev
    clip_out[b] = Scalar[DT](1.0) if dev > clip_eps else Scalar[DT](0.0)
    ent_out[b] = ent


struct PPODiscreteTrainer[
    train_target: StaticString,
    ACTOR: Module,
    CRITIC: Module,
    OBS_DIM: Int,
    N_ACTIONS: Int,
    ROLLOUT_LEN: Int,
    MINIBATCH: Int,
    N_EPOCHS: Int,
    N_ENVS: Int = 1,
](OnPolicyDiscreteAgent):
    """CleanRL-style categorical PPO trainer. Single-env (host-list)
    surface; `N_ENVS` sizes the per-step scratches (defaults to 1)."""

    # OnPolicyDiscreteAgent trait-visible comptime aliases.
    comptime AGENT_TRAIN_TARGET = Self.train_target
    comptime AGENT_OBS_DIM      = Self.OBS_DIM
    comptime AGENT_NUM_ACTIONS  = Self.N_ACTIONS

    comptime N_MINIBATCHES = (Self.ROLLOUT_LEN * Self.N_ENVS) // Self.MINIBATCH

    comptime _T_GAE = 0
    comptime _T_UPDATE = 1
    comptime _T_DIAG = 2

    # ── Networks + optimisers ────────────────────────────────────────
    var actor: Self.ACTOR
    var critic: Self.CRITIC
    var actor_opt: Adam
    var critic_opt: Adam

    # ── Blocks ───────────────────────────────────────────────────────
    var act_step: PPODiscreteActStep[
        Self.OBS_DIM, Self.N_ACTIONS, Self.N_ENVS, Self.ACTOR, Self.CRITIC,
    ]
    var record_step: PPORecordStep[Self.OBS_DIM, 1, Self.ROLLOUT_LEN]
    var gae_step: PPOGAEStep[Self.OBS_DIM, Self.ROLLOUT_LEN, Self.CRITIC]
    var gather_step: PPOMinibatchGatherStep[
        Self.OBS_DIM, 1, Self.ROLLOUT_LEN, Self.MINIBATCH,
    ]
    var actor_train: PPODiscreteActorTrainStep[
        Self.OBS_DIM, Self.MINIBATCH, Self.ACTOR,
    ]
    var critic_train: PPOCriticTrainStep[
        Self.OBS_DIM, Self.MINIBATCH, Self.CRITIC,
    ]

    # ── State (ACT=1: the action slot holds a discrete index) ─────────
    var state: OnPolicyState[
        Self.OBS_DIM, 1, Self.ROLLOUT_LEN, Self.MINIBATCH, Self.N_ENVS,
    ]

    # Host-side staging for the N=1 host-list wrapper paths.
    var _obs1: Optional[UnsafePointer[Scalar[DT], MutAnyOrigin]]
    var _act1: Optional[UnsafePointer[Scalar[DT], MutAnyOrigin]]
    var _rew1: Optional[UnsafePointer[Scalar[DT], MutAnyOrigin]]
    var _done1: Optional[UnsafePointer[Scalar[DT], MutAnyOrigin]]
    var _nobs1: Optional[UnsafePointer[Scalar[DT], MutAnyOrigin]]

    # ── Hyperparameters ──────────────────────────────────────────────
    var gamma: Scalar[DT]
    var gae_lambda: Scalar[DT]
    var clip_eps: Scalar[DT]
    var entropy_coef: Scalar[DT]

    # ── Episode tracker ──────────────────────────────────────────────
    var tracker: EpisodeTracker
    var _ep_returns: Optional[UnsafePointer[Scalar[DT], MutAnyOrigin]]  # N_ENVS

    # ── Train-step accumulators ──────────────────────────────────────
    var _actor_L_accum: Scalar[DT]
    var _critic_L_accum: Scalar[DT]
    var _entropy_accum: Scalar[DT]
    var _kl_accum: Scalar[DT]
    var _clip_accum: Scalar[DT]
    var _ev_accum: Scalar[DT]
    # Host scratch for the diag actor forward (MINIBATCH * N_ACTIONS).
    var _diag_logits: Optional[UnsafePointer[Scalar[DT], MutAnyOrigin]]
    # GPU diag: device-resident mirrors + device scratch the diag kernels use.
    var _entropy_mean_dev: DeviceMeanAccum
    var _kl_mean_dev: DeviceMeanAccum
    var _clip_mean_dev: DeviceMeanAccum
    var _ev_mean_dev: DeviceMeanAccum
    var _diag_logits_dev: Optional[DeviceBuffer[DT]]
    var _diag_ent_dev: Optional[DeviceBuffer[DT]]
    var _diag_kl_dev: Optional[DeviceBuffer[DT]]
    var _diag_clip_dev: Optional[DeviceBuffer[DT]]
    var _diag_ev_dev: Optional[DeviceBuffer[DT]]
    var _update_count: Int
    var _total_train_steps: Int

    var timer: Timer
    var ctx: Optional[DeviceContext]

    def __init__(out self):
        comptime assert (
            Self.train_target == "cpu" or Self.train_target == "gpu"
        ), "PPODiscreteTrainer: train_target must be 'cpu' or 'gpu'"
        comptime assert Self.ACTOR.IN_DIMS[0] == Self.OBS_DIM, (
            "PPODiscreteTrainer: ACTOR.IN_DIM must equal OBS_DIM"
        )
        comptime assert Self.ACTOR.OUT_DIM == Self.N_ACTIONS, (
            "PPODiscreteTrainer: ACTOR.OUT_DIM must equal N_ACTIONS"
        )
        comptime assert Self.CRITIC.IN_DIMS[0] == Self.OBS_DIM, (
            "PPODiscreteTrainer: CRITIC.IN_DIM must equal OBS_DIM"
        )
        comptime assert Self.CRITIC.OUT_DIM == 1, (
            "PPODiscreteTrainer: CRITIC.OUT_DIM must equal 1"
        )
        comptime assert (Self.ROLLOUT_LEN * Self.N_ENVS) % Self.MINIBATCH == 0, (
            "PPODiscreteTrainer: ROLLOUT_LEN * N_ENVS must be divisible by MINIBATCH"
        )
        comptime assert Self.N_ENVS >= 1, "PPODiscreteTrainer: N_ENVS must be >= 1"
        self.actor = Self.ACTOR()
        self.critic = Self.CRITIC()
        self.actor_opt = Adam()
        self.critic_opt = Adam()
        self.act_step = PPODiscreteActStep[
            Self.OBS_DIM, Self.N_ACTIONS, Self.N_ENVS, Self.ACTOR, Self.CRITIC,
        ]()
        self.record_step = PPORecordStep[Self.OBS_DIM, 1, Self.ROLLOUT_LEN]()
        self.gae_step = PPOGAEStep[
            Self.OBS_DIM, Self.ROLLOUT_LEN, Self.CRITIC,
        ]()
        self.gather_step = PPOMinibatchGatherStep[
            Self.OBS_DIM, 1, Self.ROLLOUT_LEN, Self.MINIBATCH,
        ]()
        self.actor_train = PPODiscreteActorTrainStep[
            Self.OBS_DIM, Self.MINIBATCH, Self.ACTOR,
        ]()
        self.critic_train = PPOCriticTrainStep[
            Self.OBS_DIM, Self.MINIBATCH, Self.CRITIC,
        ]()
        self.state = OnPolicyState[
            Self.OBS_DIM, 1, Self.ROLLOUT_LEN, Self.MINIBATCH, Self.N_ENVS,
        ]()
        self._obs1  = None
        self._act1  = None
        self._rew1  = None
        self._done1 = None
        self._nobs1 = None
        self.gamma = Scalar[DT](0.99)
        self.gae_lambda = Scalar[DT](0.95)
        self.clip_eps = Scalar[DT](0.2)
        self.entropy_coef = Scalar[DT](0.0)
        self.tracker = EpisodeTracker.new(
            window_size=10, initial_fill=Scalar[DT](0.0),
        )
        self._ep_returns = None
        self._actor_L_accum = Scalar[DT](0.0)
        self._critic_L_accum = Scalar[DT](0.0)
        self._entropy_accum = Scalar[DT](0.0)
        self._kl_accum = Scalar[DT](0.0)
        self._clip_accum = Scalar[DT](0.0)
        self._ev_accum = Scalar[DT](0.0)
        self._diag_logits = None
        self._entropy_mean_dev = DeviceMeanAccum()
        self._kl_mean_dev = DeviceMeanAccum()
        self._clip_mean_dev = DeviceMeanAccum()
        self._ev_mean_dev = DeviceMeanAccum()
        self._diag_logits_dev = None
        self._diag_ent_dev = None
        self._diag_kl_dev = None
        self._diag_clip_dev = None
        self._diag_ev_dev = None
        self._update_count = 0
        self._total_train_steps = 0
        self.timer = Timer.new()
        self.ctx = None

    @staticmethod
    def make(
        actor_lr: Scalar[DT] = Scalar[DT](3e-4),
        critic_lr: Scalar[DT] = Scalar[DT](1e-3),
        gamma: Scalar[DT] = Scalar[DT](0.99),
        gae_lambda: Scalar[DT] = Scalar[DT](0.95),
        clip_eps: Scalar[DT] = Scalar[DT](0.2),
        entropy_coef: Scalar[DT] = Scalar[DT](0.01),
        window_size: Int = 10,
        initial_episode_fill: Scalar[DT] = Scalar[DT](0.0),
        max_grad_norm: Scalar[DT] = Scalar[DT](0.0),
        ctx: Optional[DeviceContext] = None,
    ) raises -> Self:
        comptime assert (
            Self.train_target == "cpu" or Self.train_target == "gpu"
        ), "PPODiscreteTrainer.make: train_target must be 'cpu' or 'gpu'"
        comptime if Self.train_target == "gpu":
            if not ctx:
                raise Error(
                    "PPODiscreteTrainer.make[train_target='gpu']: ctx required"
                )
        var t = Self()
        t.ctx = ctx
        t.actor = Self.ACTOR.make[target=Self.train_target, INIT=Xavier](
            ctx=ctx
        )
        t.critic = Self.CRITIC.make[target=Self.train_target, INIT=Xavier](
            ctx=ctx
        )
        t.actor_opt = Adam.make[target=Self.train_target, M=Self.ACTOR](
            t.actor, ctx=ctx,
        )
        t.actor_opt.lr = actor_lr
        t.actor_opt.max_grad_norm = max_grad_norm
        t.critic_opt = Adam.make[target=Self.train_target, M=Self.CRITIC](
            t.critic, ctx=ctx,
        )
        t.critic_opt.lr = critic_lr
        t.critic_opt.max_grad_norm = max_grad_norm
        t.act_step = PPODiscreteActStep[
            Self.OBS_DIM, Self.N_ACTIONS, Self.N_ENVS, Self.ACTOR, Self.CRITIC,
        ].make[Self.train_target](ctx=ctx)
        t.record_step = PPORecordStep[
            Self.OBS_DIM, 1, Self.ROLLOUT_LEN,
        ].make[Self.train_target](ctx=ctx)
        t.gae_step = PPOGAEStep[
            Self.OBS_DIM, Self.ROLLOUT_LEN, Self.CRITIC,
        ].make[Self.train_target](ctx=ctx)
        t.gather_step = PPOMinibatchGatherStep[
            Self.OBS_DIM, 1, Self.ROLLOUT_LEN, Self.MINIBATCH,
        ].make[Self.train_target](ctx=ctx)
        t.actor_train = PPODiscreteActorTrainStep[
            Self.OBS_DIM, Self.MINIBATCH, Self.ACTOR,
        ].make[Self.train_target](
            ctx=ctx, clip_eps=clip_eps, entropy_coef=entropy_coef,
        )
        t.critic_train = PPOCriticTrainStep[
            Self.OBS_DIM, Self.MINIBATCH, Self.CRITIC,
        ].make[Self.train_target](ctx=ctx)
        t.state = OnPolicyState[
            Self.OBS_DIM, 1, Self.ROLLOUT_LEN, Self.MINIBATCH, Self.N_ENVS,
        ].make[Self.train_target](ctx=ctx)
        t._obs1  = alloc[Scalar[DT]](Self.OBS_DIM)
        t._act1  = alloc[Scalar[DT]](1)
        t._rew1  = alloc[Scalar[DT]](1)
        t._done1 = alloc[Scalar[DT]](1)
        t._nobs1 = alloc[Scalar[DT]](Self.OBS_DIM)
        var ep_returns_p = alloc[Scalar[DT]](Self.N_ENVS)
        for e in range(Self.N_ENVS):
            ep_returns_p[e] = Scalar[DT](0.0)
        t._ep_returns = ep_returns_p
        t._diag_logits = alloc[Scalar[DT]](Self.MINIBATCH * Self.N_ACTIONS)

        comptime if Self.train_target == "gpu":
            var c = ctx.value()
            t._diag_logits_dev = c.enqueue_create_buffer[DT](
                Self.MINIBATCH * Self.N_ACTIONS
            )
            t._diag_ent_dev = c.enqueue_create_buffer[DT](Self.MINIBATCH)
            t._diag_kl_dev = c.enqueue_create_buffer[DT](Self.MINIBATCH)
            t._diag_clip_dev = c.enqueue_create_buffer[DT](Self.MINIBATCH)
            t._diag_ev_dev = c.enqueue_create_buffer[DT](1)
            t._entropy_mean_dev = DeviceMeanAccum.make["gpu"](ctx=ctx)
            t._kl_mean_dev = DeviceMeanAccum.make["gpu"](ctx=ctx)
            t._clip_mean_dev = DeviceMeanAccum.make["gpu"](ctx=ctx)
            t._ev_mean_dev = DeviceMeanAccum.make["gpu"](ctx=ctx)

        t.gamma = gamma
        t.gae_lambda = gae_lambda
        t.clip_eps = clip_eps
        t.entropy_coef = entropy_coef
        t.tracker = EpisodeTracker.new(
            window_size=window_size, initial_fill=initial_episode_fill,
        )
        t.timer.add_section("gae")
        t.timer.add_section("update")
        t.timer.add_section("diag")
        return t^

    # ──────────────────────────────────────────────────────────────────
    # OnPolicyDiscreteAgent surface (single-env host-list)
    # ──────────────────────────────────────────────────────────────────

    def select_action(
        mut self,
        ref obs: List[Scalar[DT]],
        step_idx: Int,
    ) raises -> Int:
        comptime assert Self.N_ENVS == 1, (
            "PPODiscreteTrainer.select_action: host-list wrapper only "
            "valid at N_ENVS=1"
        )
        _ = step_idx
        var obs_p = self._obs1.value()
        var act_p = self._act1.value()
        for d in range(Self.OBS_DIM):
            obs_p[d] = obs[d]
        self.act_step.step[
            Self.train_target, Self.ROLLOUT_LEN, Self.MINIBATCH, Self.N_ENVS,
        ](self.state, self.actor, self.critic, obs_p, act_p)
        return Int(act_p[0])

    def select_greedy_action(
        mut self,
        ref obs: List[Scalar[DT]],
    ) raises -> Int:
        return self.act_step.step_greedy_n1[
            Self.train_target, Self.ROLLOUT_LEN, Self.MINIBATCH, Self.N_ENVS,
        ](self.state, self.actor, obs)

    def record_transition(
        mut self,
        ref obs: List[Scalar[DT]],
        action_idx: Int,
        reward: Scalar[DT],
        ref next_obs: List[Scalar[DT]],
        done: Scalar[DT],
    ) raises:
        comptime assert Self.N_ENVS == 1, (
            "PPODiscreteTrainer.record_transition: host-list wrapper only "
            "valid at N_ENVS=1"
        )
        # The action index was cached by select_action; the passed
        # action_idx is ignored (cached sample is authoritative — mirrors
        # the continuous trainer ignoring the env-ready action).
        _ = action_idx
        var obs_p = self._obs1.value()
        var nobs_p = self._nobs1.value()
        var rew_p = self._rew1.value()
        var done_p = self._done1.value()
        for d in range(Self.OBS_DIM):
            obs_p[d]  = obs[d]
            nobs_p[d] = next_obs[d]
        rew_p[0]  = reward
        done_p[0] = done
        self.record_step.step[
            Self.train_target, Self.MINIBATCH, Self.N_ENVS,
        ](self.state, obs_p, rew_p, nobs_p, done_p)
        self.tracker.add_reward(reward)

    def mark_terminal(mut self) raises:
        comptime assert Self.N_ENVS == 1, (
            "PPODiscreteTrainer.mark_terminal: host-list wrapper only "
            "valid at N_ENVS=1"
        )
        self.record_step.mark_terminal[
            Self.train_target, Self.MINIBATCH, Self.N_ENVS,
        ](self.state, 0)

    def end_episode(mut self):
        self.tracker.end_episode()

    def train_step(mut self, step_idx: Int) raises -> Bool:
        _ = step_idx
        if self.state.rollout_idx < Self.ROLLOUT_LEN:
            return False

        var t_gae = perf_counter_ns()
        self.gae_step.step[
            Self.train_target, 1, Self.MINIBATCH, Self.N_ENVS,
        ](self.state, self.critic, self.gamma, self.gae_lambda)
        self.timer.accumulate(Self._T_GAE, t_gae)

        var t_upd = perf_counter_ns()
        self.gather_step.reset_indices[Self.train_target, Self.N_ENVS](
            self.state
        )
        for _epoch in range(Self.N_EPOCHS):
            self.gather_step.shuffle_epoch[Self.train_target, Self.N_ENVS](
                self.state
            )
            for mb in range(Self.N_MINIBATCHES):
                self.gather_step.gather[Self.train_target, Self.N_ENVS](
                    self.state, mb
                )
                var aL = self.actor_train.step[
                    Self.train_target, Self.ROLLOUT_LEN, Self.N_ENVS,
                ](self.state, self.actor, self.actor_opt)
                var cL = self.critic_train.step[
                    Self.train_target, 1, Self.ROLLOUT_LEN, Self.N_ENVS,
                ](self.state, self.critic, self.critic_opt)
                self._actor_L_accum += aL
                self._critic_L_accum += cL
                self._update_count += 1
                self._total_train_steps += 1
                var t_diag = perf_counter_ns()
                comptime if Self.train_target == "cpu":
                    self._accumulate_diag()
                else:
                    self._accumulate_diag_gpu()
                self.timer.accumulate(Self._T_DIAG, t_diag)
        self.timer.accumulate(Self._T_UPDATE, t_upd)

        self.record_step.reset_rollout[
            Self.train_target, Self.MINIBATCH, Self.N_ENVS,
        ](self.state)
        return True

    def _accumulate_diag(mut self) raises:
        """CPU-only per-minibatch categorical-PPO diagnostics: entropy,
        Schulman-2020 approx_kl, clip_fraction, explained variance.
        Re-runs the (post-update) actor over the minibatch obs and reads
        the critic's value/return scratches."""
        comptime N = Self.N_ACTIONS
        comptime MB = Self.MINIBATCH
        var act_p = self.state.mb_act.target_ptr["cpu"]()   # action indices
        var olp_p = self.state.mb_olp.target_ptr["cpu"]()
        var v_p   = self.state.mb_v.target_ptr["cpu"]()
        var ret_p = self.state.mb_ret.target_ptr["cpu"]()
        var obs_p = self.state.mb_obs.target_ptr["cpu"]()

        var obs_t = TileTensor(obs_p, row_major[MB, Self.OBS_DIM]())
        var lg = self._diag_logits.value()
        var lg_t  = TileTensor(lg, row_major[MB, N]())
        self.actor.forward["cpu", MB](obs_t, output=lg_t)

        var ent_sum = Scalar[DT](0.0)
        var kl_sum = Scalar[DT](0.0)
        var clip_sum = Scalar[DT](0.0)
        for b in range(MB):
            var base = b * N
            var max_l = lg[base]
            for j in range(1, N):
                if lg[base + j] > max_l:
                    max_l = lg[base + j]
            var sum_exp = Scalar[DT](0.0)
            for j in range(N):
                sum_exp += fexp(lg[base + j] - max_l)
            var log_sum = flog(sum_exp)
            var ent = Scalar[DT](0.0)
            for j in range(N):
                var lp_j = (lg[base + j] - max_l) - log_sum
                var p_j = fexp(lp_j)
                ent += -p_j * lp_j
            var a_idx = Int(act_p[b])
            var nlp = (lg[base + a_idx] - max_l) - log_sum
            var diff = nlp - olp_p[b]
            if diff > _DIAG_LOG_PROB_DIFF_MAX:
                diff = _DIAG_LOG_PROB_DIFF_MAX
            elif diff < -_DIAG_LOG_PROB_DIFF_MAX:
                diff = -_DIAG_LOG_PROB_DIFF_MAX
            var ratio = fexp(diff)
            kl_sum += (ratio - Scalar[DT](1.0)) - diff
            var dev = ratio - Scalar[DT](1.0)
            if dev < Scalar[DT](0.0):
                dev = -dev
            if dev > self.clip_eps:
                clip_sum += Scalar[DT](1.0)
            ent_sum += ent

        var inv_mb = Scalar[DT](1.0) / Scalar[DT](MB)
        self._entropy_accum += ent_sum * inv_mb
        self._kl_accum += kl_sum * inv_mb
        self._clip_accum += clip_sum * inv_mb

        # Explained variance (CleanRL): 1 - Var(ret - v) / Var(ret).
        var mean_ret = Scalar[DT](0.0)
        var mean_res = Scalar[DT](0.0)
        for b in range(MB):
            mean_ret += ret_p[b]
            mean_res += ret_p[b] - v_p[b]
        mean_ret *= inv_mb
        mean_res *= inv_mb
        var var_ret = Scalar[DT](0.0)
        var var_res = Scalar[DT](0.0)
        for b in range(MB):
            var dr = ret_p[b] - mean_ret
            var rr = (ret_p[b] - v_p[b]) - mean_res
            var_ret += dr * dr
            var_res += rr * rr
        var ev = Scalar[DT](0.0)
        if var_ret > Scalar[DT](1e-8):
            ev = Scalar[DT](1.0) - var_res / var_ret
        self._ev_accum += ev

    def _accumulate_diag_gpu(mut self) raises:
        """GPU mirror of `_accumulate_diag`: re-run the (post-update) actor on
        the minibatch obs on device, derive categorical entropy / KL / clip
        via `_ppo_discrete_diag_kernel` and explained variance via the shared
        `_ppo_ev_kernel`, then fold them into the device-resident running
        means. No per-minibatch D2H."""
        comptime N = Self.N_ACTIONS
        comptime MB = Self.MINIBATCH
        var ctx = self.ctx.value()

        var obs_p = self.state.mb_obs.target_ptr["gpu"]()
        var obs_t = TileTensor(obs_p, row_major[MB, Self.OBS_DIM]())
        var lg_p = rebind[UnsafePointer[Scalar[DT], MutAnyOrigin]](
            self._diag_logits_dev.value().unsafe_ptr()
        )
        var lg_t = TileTensor(lg_p, row_major[MB, N]())
        self.actor.forward["gpu", MB](obs_t, output=lg_t)

        var act_p = self.state.mb_act.target_ptr["gpu"]()
        var olp_p = self.state.mb_olp.target_ptr["gpu"]()
        var v_p = self.state.mb_v.target_ptr["gpu"]()
        var ret_p = self.state.mb_ret.target_ptr["gpu"]()
        var ent_p = self._diag_ent_dev.value().unsafe_ptr()
        var kl_p = self._diag_kl_dev.value().unsafe_ptr()
        var clip_p = self._diag_clip_dev.value().unsafe_ptr()
        var ev_p = self._diag_ev_dev.value().unsafe_ptr()

        comptime n_blocks = (MB + TPB - 1) // TPB
        comptime per_k = _ppo_discrete_diag_kernel[MB, N]
        ctx.enqueue_function[per_k](
            lg_p, act_p, olp_p, self.clip_eps, ent_p, kl_p, clip_p,
            grid_dim=n_blocks, block_dim=TPB,
        )
        self._entropy_mean_dev.accumulate_gpu[MB](ent_p)
        self._kl_mean_dev.accumulate_gpu[MB](kl_p)
        self._clip_mean_dev.accumulate_gpu[MB](clip_p)

        comptime ev_k = _ppo_ev_kernel[MB]
        ctx.enqueue_function[ev_k](
            ret_p, v_p, ev_p, grid_dim=1, block_dim=TPB_REDUCE,
        )
        self._ev_mean_dev.accumulate_gpu[1](ev_p)

    def mean_return(self) -> Scalar[DT]:
        return self.tracker.mean_return()

    def ep_count(self) -> Int:
        return self.tracker.ep_count

    # ─── Logging surface ─────────────────────────────────────────────

    def total_train_steps(self) -> Int:
        return self._total_train_steps

    def flush_metrics[
        L: Logger = NoOpLogger
    ](
        mut self,
        logger: Optional[UnsafePointer[L, MutAnyOrigin]] = None,
        step: Int = 0,
    ) raises -> PPOMetrics:
        """Drain accumulators into a PPOMetrics bundle (reused verbatim —
        `entropy` is the categorical entropy here). Resets per-chunk
        accumulators; `_total_train_steps` is NOT reset."""
        var n = self._update_count if self._update_count > 0 else 1
        var inv = Scalar[DT](1.0) / Scalar[DT](n)
        # entropy / approx_kl / clip_fraction / explained_variance are
        # device-resident on GPU (derived by `_accumulate_diag_gpu`), host
        # scalars on CPU.
        var entropy_mean: Scalar[DT]
        var kl_mean: Scalar[DT]
        var clip_mean: Scalar[DT]
        var ev_mean: Scalar[DT]
        comptime if Self.train_target == "gpu":
            entropy_mean = self._entropy_mean_dev.read["gpu"]()
            kl_mean = self._kl_mean_dev.read["gpu"]()
            clip_mean = self._clip_mean_dev.read["gpu"]()
            ev_mean = self._ev_mean_dev.read["gpu"]()
        else:
            entropy_mean = self._entropy_accum * inv
            kl_mean = self._kl_accum * inv
            clip_mean = self._clip_accum * inv
            ev_mean = self._ev_accum * inv
        var bundle = PPOMetrics(
            actor_loss=LogScalar[DT](self._actor_L_accum * inv),
            critic_loss=LogScalar[DT](self._critic_L_accum * inv),
            train_steps=LogScalar[DT](Scalar[DT](self._total_train_steps)),
            n_updates=LogScalar[DT](Scalar[DT](self._update_count)),
            entropy=LogScalar[DT](entropy_mean),
            approx_kl=LogScalar[DT](kl_mean),
            clip_fraction=LogScalar[DT](clip_mean),
            explained_variance=LogScalar[DT](ev_mean),
        )
        self._actor_L_accum = Scalar[DT](0.0)
        self._critic_L_accum = Scalar[DT](0.0)
        self._entropy_accum = Scalar[DT](0.0)
        self._kl_accum = Scalar[DT](0.0)
        self._clip_accum = Scalar[DT](0.0)
        self._ev_accum = Scalar[DT](0.0)
        self._update_count = 0
        comptime if Self.train_target == "gpu":
            self._entropy_mean_dev.reset["gpu"]()
            self._kl_mean_dev.reset["gpu"]()
            self._clip_mean_dev.reset["gpu"]()
            self._ev_mean_dev.reset["gpu"]()
        if Bool(logger):
            log_bundle(logger.value()[], bundle, step)
        return bundle^

    def flush_metrics_through_logger[L: Logger](
        mut self,
        logger: Optional[UnsafePointer[L, MutAnyOrigin]],
        step: Int,
    ) raises:
        _ = self.flush_metrics[L](logger, step)

    def save_state(mut self, path: String) raises:
        """One-file v2 checkpoint: actor / critic / actor_opt / critic_opt."""
        var body = String("")
        comptime if Self.train_target == "cpu":
            save_state_v2_body(self.actor, body, "actor")
            save_state_v2_body(self.critic, body, "critic")
            save_optimizer_v2_body(self.actor_opt, body, "actor_opt")
            save_optimizer_v2_body(self.critic_opt, body, "critic_opt")
        else:
            var c = self.ctx.value()
            save_state_v2_body_gpu(self.actor, body, "actor", c)
            save_state_v2_body_gpu(self.critic, body, "critic", c)
            save_optimizer_v2_body_gpu(self.actor_opt, body, "actor_opt")
            save_optimizer_v2_body_gpu(self.critic_opt, body, "critic_opt")
        var content = String("nn2-ckpt v2\n") + body
        with open(path, "w") as f:
            f.write(content)

    def load_state(mut self, path: String) raises:
        var content = read_file_v2(path)
        var lines = split_lines_v2(content)
        expect_v2_header(lines)
        var idx: Int = 1
        comptime if Self.train_target == "cpu":
            load_state_v2_body(self.actor, lines, idx, "actor")
            load_state_v2_body(self.critic, lines, idx, "critic")
            load_optimizer_v2_body(self.actor_opt, lines, idx, "actor_opt")
            load_optimizer_v2_body(self.critic_opt, lines, idx, "critic_opt")
        else:
            var c = self.ctx.value()
            load_state_v2_body_gpu(self.actor, lines, idx, "actor", c)
            load_state_v2_body_gpu(self.critic, lines, idx, "critic", c)
            load_optimizer_v2_body_gpu(self.actor_opt, lines, idx, "actor_opt")
            load_optimizer_v2_body_gpu(self.critic_opt, lines, idx, "critic_opt")

    def flush_timer_log(mut self) -> String:
        var report = self.timer.format_report()
        self.timer.reset()
        return report
