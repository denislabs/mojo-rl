"""PPOTrainer — block-composed PPO continuous trainer (CleanRL-style).

Composes 6 ref-based step blocks via `OnPolicyState`:

  PPOActStep              — per env-step: actor.forward + sample + critic.forward
  PPORecordStep           — per env-step: push cached → rollout buffer
  PPOGAEStep              — per rollout: bootstrap + per-env GAE
  PPOMinibatchGatherStep  — per epoch:    Fisher-Yates shuffle
                            per minibatch: gather + normalise mb_adv
  PPOActorTrainStep       — per minibatch: actor PPO clipped surrogate update
  PPOCriticTrainStep      — per minibatch: critic MSE update

Dual-target (CPU/GPU via `train_target` struct comptime) × N_ENVS-
parametric (default 1). Single-env (N_ENVS=1) users get a host-list
`OnPolicyAgent` surface (select_action / record_transition / etc.)
consumed by `run_onpolicy_train`. Multi-env (N_ENVS>=1) users get
the pointer-based `OnPolicyAgentBatched` surface consumed by
`run_onpolicy_train_batched` over a `BatchedEnv` adapter.

GPU train_target is a hybrid: per-step actor/critic forwards run on
device (H2D obs + D2H ao/v inside PPOActStep); rollout buffers live
host-only; the K-epoch minibatch is H2D-uploaded into device-side
mb_* scratches before each PPOActorTrainStep / PPOCriticTrainStep.
"""

from std.gpu.host import DeviceContext
from std.memory import alloc
from std.time import perf_counter_ns

from mojo_rl.core.logger import Logger, NoOpLogger
from mojo_rl.nn2.constants import DT
from mojo_rl.nn2.core import Module
from mojo_rl.nn2.core.checkpoint import (
    save_state_v2_body, load_state_v2_body,
    save_state_v2_body_gpu, load_state_v2_body_gpu,
)
from mojo_rl.nn2.core.log_bundle import log_bundle
from mojo_rl.nn2.core.metric import LogScalar
from mojo_rl.nn2.combinators.sequential import Sequential
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
from ..training.driver_onpolicy import OnPolicyAgent, OnPolicyAgentBatched
from .blocks.act_step import PPOActStep
from .blocks.record_step import PPORecordStep
from .blocks.gae_step import PPOGAEStep
from .blocks.minibatch_gather_step import PPOMinibatchGatherStep
from .blocks.actor_train_step import PPOActorTrainStep
from .blocks.critic_train_step import PPOCriticTrainStep
from .metrics import PPOMetrics


struct PPOTrainer[
    train_target: StaticString,
    ACTOR: Module,
    CRITIC: Module,
    OBS_DIM: Int,
    ACT_DIM: Int,
    ROLLOUT_LEN: Int,
    MINIBATCH: Int,
    N_EPOCHS: Int,
    N_ENVS: Int = 1,
](OnPolicyAgent & OnPolicyAgentBatched):
    """CleanRL-style PPO continuous trainer. N_ENVS defaults to 1 for
    single-env consumers (host-list select_action / record_transition
    surface); N_ENVS > 1 uses the pointer-based batched methods
    consumed by `run_onpolicy_train_batched`."""

    # OnPolicyAgentBatched trait-visible comptime aliases.
    comptime AGENT_TRAIN_TARGET = Self.train_target
    comptime AGENT_OBS_DIM      = Self.OBS_DIM
    comptime AGENT_ACT_DIM      = Self.ACT_DIM
    comptime AGENT_N_ENVS       = Self.N_ENVS

    comptime N_MINIBATCHES = (Self.ROLLOUT_LEN * Self.N_ENVS) // Self.MINIBATCH

    # Timer section indices — order matches `add_section` calls in `make`.
    # PPO's train_step body is dominated by the K-epoch SGD loop (single
    # section: `update`). GAE bootstrap + per-env backward is its own
    # `gae` section. Sample / target_y / polyak don't apply to on-policy.
    comptime _T_GAE = 0
    comptime _T_UPDATE = 1

    # ── Networks + optimisers ────────────────────────────────────────
    var actor: Self.ACTOR
    var critic: Self.CRITIC
    var actor_opt: Adam
    var critic_opt: Adam

    # ── Blocks ───────────────────────────────────────────────────────
    var act_step: PPOActStep[Self.OBS_DIM, Self.ACT_DIM, Self.ACTOR, Self.CRITIC]
    var record_step: PPORecordStep[Self.OBS_DIM, Self.ACT_DIM, Self.ROLLOUT_LEN]
    var gae_step: PPOGAEStep[Self.OBS_DIM, Self.ROLLOUT_LEN, Self.CRITIC]
    var gather_step: PPOMinibatchGatherStep[
        Self.OBS_DIM, Self.ACT_DIM, Self.ROLLOUT_LEN, Self.MINIBATCH,
    ]
    var actor_train: PPOActorTrainStep[
        Self.OBS_DIM, Self.ACT_DIM, Self.MINIBATCH, Self.ACTOR,
    ]
    var critic_train: PPOCriticTrainStep[
        Self.OBS_DIM, Self.MINIBATCH, Self.CRITIC,
    ]

    # ── State ────────────────────────────────────────────────────────
    var state: OnPolicyState[
        Self.OBS_DIM, Self.ACT_DIM, Self.ROLLOUT_LEN, Self.MINIBATCH,
        Self.N_ENVS,
    ]

    # Host-side staging for the N=1 host-list wrapper paths (so they
    # don't allocate per call).
    var _obs1: UnsafePointer[Scalar[DT], MutAnyOrigin]
    var _act1: UnsafePointer[Scalar[DT], MutAnyOrigin]
    var _rew1: UnsafePointer[Scalar[DT], MutAnyOrigin]
    var _done1: UnsafePointer[Scalar[DT], MutAnyOrigin]
    var _nobs1: UnsafePointer[Scalar[DT], MutAnyOrigin]

    # ── Hyperparameters ──────────────────────────────────────────────
    var gamma: Scalar[DT]
    var gae_lambda: Scalar[DT]
    var clip_eps: Scalar[DT]
    var entropy_coef: Scalar[DT]
    var action_scale: Scalar[DT]

    # ── Episode tracker (per-env running-return + completed-return window) ─
    var tracker: EpisodeTracker
    var _ep_returns: UnsafePointer[Scalar[DT], MutAnyOrigin]  # N_ENVS

    # ── Train-step accumulators (summed across all minibatch updates) ────
    var _actor_L_accum: Scalar[DT]
    var _critic_L_accum: Scalar[DT]
    var _update_count: Int
    # Never reset by `flush_*` — emitted as `train_steps` so the
    # downstream monitor can plot cumulative minibatch updates.
    var _total_train_steps: Int

    var timer: Timer

    # Device context (GPU only; None on CPU). Threaded from `make` so the
    # GPU checkpoint path can stage device buffers → host.
    var ctx: Optional[DeviceContext]

    def __init__(out self):
        comptime assert (
            Self.train_target == "cpu" or Self.train_target == "gpu"
        ), "PPOTrainer: train_target must be 'cpu' or 'gpu'"
        comptime assert Self.ACTOR.IN_DIMS[0] == Self.OBS_DIM, (
            "PPOTrainer: ACTOR.IN_DIM must equal OBS_DIM"
        )
        comptime assert Self.ACTOR.OUT_DIM == 2 * Self.ACT_DIM, (
            "PPOTrainer: ACTOR.OUT_DIM must equal 2 * ACT_DIM"
        )
        comptime assert Self.CRITIC.IN_DIMS[0] == Self.OBS_DIM, (
            "PPOTrainer: CRITIC.IN_DIM must equal OBS_DIM"
        )
        comptime assert Self.CRITIC.OUT_DIM == 1, (
            "PPOTrainer: CRITIC.OUT_DIM must equal 1"
        )
        comptime assert (Self.ROLLOUT_LEN * Self.N_ENVS) % Self.MINIBATCH == 0, (
            "PPOTrainer: ROLLOUT_LEN * N_ENVS must be divisible by MINIBATCH"
        )
        comptime assert Self.N_ENVS >= 1, "PPOTrainer: N_ENVS must be >= 1"
        self.actor = Self.ACTOR()
        self.critic = Self.CRITIC()
        self.actor_opt = Adam()
        self.critic_opt = Adam()
        self.act_step = PPOActStep[
            Self.OBS_DIM, Self.ACT_DIM, Self.ACTOR, Self.CRITIC,
        ]()
        self.record_step = PPORecordStep[
            Self.OBS_DIM, Self.ACT_DIM, Self.ROLLOUT_LEN,
        ]()
        self.gae_step = PPOGAEStep[
            Self.OBS_DIM, Self.ROLLOUT_LEN, Self.CRITIC,
        ]()
        self.gather_step = PPOMinibatchGatherStep[
            Self.OBS_DIM, Self.ACT_DIM, Self.ROLLOUT_LEN, Self.MINIBATCH,
        ]()
        self.actor_train = PPOActorTrainStep[
            Self.OBS_DIM, Self.ACT_DIM, Self.MINIBATCH, Self.ACTOR,
        ]()
        self.critic_train = PPOCriticTrainStep[
            Self.OBS_DIM, Self.MINIBATCH, Self.CRITIC,
        ]()
        self.state = OnPolicyState[
            Self.OBS_DIM, Self.ACT_DIM, Self.ROLLOUT_LEN, Self.MINIBATCH,
            Self.N_ENVS,
        ]()
        var null_p = UnsafePointer[Scalar[DT], MutAnyOrigin](
            unsafe_from_address=0
        )
        self._obs1  = null_p
        self._act1  = null_p
        self._rew1  = null_p
        self._done1 = null_p
        self._nobs1 = null_p
        self.gamma = Scalar[DT](0.99)
        self.gae_lambda = Scalar[DT](0.95)
        self.clip_eps = Scalar[DT](0.2)
        self.entropy_coef = Scalar[DT](0.0)
        self.action_scale = Scalar[DT](1.0)
        self.tracker = EpisodeTracker.new(
            window_size=10, initial_fill=Scalar[DT](-1600.0),
        )
        self._ep_returns = null_p
        self._actor_L_accum = Scalar[DT](0.0)
        self._critic_L_accum = Scalar[DT](0.0)
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
        entropy_coef: Scalar[DT] = Scalar[DT](0.0),
        action_scale: Scalar[DT] = Scalar[DT](1.0),
        log_std_init: Scalar[DT] = Scalar[DT](-0.5),
        window_size: Int = 10,
        initial_episode_fill: Scalar[DT] = Scalar[DT](-1600.0),
        # Canonical PPO uses max_grad_norm=0.5 (Schulman 2017 + most
        # implementations). Default 0 keeps bit-identity for callers
        # that previously trained unclipped. Wired to both optimizers
        # below — separate from `clip_eps`, which is the policy-ratio
        # surrogate clip, not the gradient-norm clip.
        max_grad_norm: Scalar[DT] = Scalar[DT](0.0),
        ctx: Optional[DeviceContext] = None,
    ) raises -> Self:
        comptime assert (
            Self.train_target == "cpu" or Self.train_target == "gpu"
        ), "PPOTrainer.make: train_target must be 'cpu' or 'gpu'"
        comptime if Self.train_target == "gpu":
            if not ctx:
                raise Error(
                    "PPOTrainer.make[train_target='gpu']: ctx required"
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
        t.act_step = PPOActStep[
            Self.OBS_DIM, Self.ACT_DIM, Self.ACTOR, Self.CRITIC,
        ].make[Self.train_target](ctx=ctx)
        t.record_step = PPORecordStep[
            Self.OBS_DIM, Self.ACT_DIM, Self.ROLLOUT_LEN,
        ].make[Self.train_target](ctx=ctx)
        t.gae_step = PPOGAEStep[
            Self.OBS_DIM, Self.ROLLOUT_LEN, Self.CRITIC,
        ].make[Self.train_target](ctx=ctx)
        t.gather_step = PPOMinibatchGatherStep[
            Self.OBS_DIM, Self.ACT_DIM, Self.ROLLOUT_LEN, Self.MINIBATCH,
        ].make[Self.train_target](ctx=ctx)
        t.actor_train = PPOActorTrainStep[
            Self.OBS_DIM, Self.ACT_DIM, Self.MINIBATCH, Self.ACTOR,
        ].make[Self.train_target](
            ctx=ctx, clip_eps=clip_eps, entropy_coef=entropy_coef,
        )
        t.critic_train = PPOCriticTrainStep[
            Self.OBS_DIM, Self.MINIBATCH, Self.CRITIC,
        ].make[Self.train_target](ctx=ctx)
        t.state = OnPolicyState[
            Self.OBS_DIM, Self.ACT_DIM, Self.ROLLOUT_LEN, Self.MINIBATCH,
            Self.N_ENVS,
        ].make[Self.train_target](ctx=ctx)
        t._obs1  = alloc[Scalar[DT]](Self.OBS_DIM)
        t._act1  = alloc[Scalar[DT]](Self.ACT_DIM)
        t._rew1  = alloc[Scalar[DT]](1)
        t._done1 = alloc[Scalar[DT]](1)
        t._nobs1 = alloc[Scalar[DT]](Self.OBS_DIM)
        t._ep_returns = alloc[Scalar[DT]](Self.N_ENVS)
        for e in range(Self.N_ENVS):
            t._ep_returns[e] = Scalar[DT](0.0)
        t.gamma = gamma
        t.gae_lambda = gae_lambda
        t.clip_eps = clip_eps
        t.entropy_coef = entropy_coef
        t.action_scale = action_scale
        # log_std_init is the caller's responsibility (reaching into the
        # actor's GaussianHead.log_std vector — see the example for the
        # idiom). Kept here for forward-compat / docs.
        _ = log_std_init
        t.tracker = EpisodeTracker.new(
            window_size=window_size, initial_fill=initial_episode_fill,
        )

        # Timer sections — index order MUST match the `_T_*` comptime
        # constants above.
        t.timer.add_section("gae")
        t.timer.add_section("update")
        return t^

    # ──────────────────────────────────────────────────────────────────
    # OnPolicyAgent surface
    # ──────────────────────────────────────────────────────────────────

    def select_action(
        mut self,
        ref obs: List[Scalar[DT]],
        mut action_out: List[Scalar[DT]],
        step_idx: Int,
    ) raises:
        """N=1 host-list wrapper — only valid when Self.N_ENVS == 1.
        Stages obs into _obs1, delegates to `select_action_batched`
        (which is N_ENVS=Self.N_ENVS-wide), then copies _act1 out."""
        comptime assert Self.N_ENVS == 1, (
            "PPOTrainer.select_action: host-list wrapper only valid "
            "at N_ENVS=1; use select_action_batched for N_ENVS>1"
        )
        for d in range(Self.OBS_DIM):
            self._obs1[d] = obs[d]
        self.select_action_batched(self._obs1, self._act1, step_idx)
        for j in range(Self.ACT_DIM):
            action_out[j] = self._act1[j]

    def select_action_batched(
        mut self,
        obs_ptr: UnsafePointer[Scalar[DT], MutAnyOrigin],
        action_ptr: UnsafePointer[Scalar[DT], MutAnyOrigin],
        step_idx: Int,
    ) raises:
        """N_ENVS-wide action selection. Reads N_ENVS*OBS from obs_ptr,
        writes N_ENVS*ACT into action_ptr, caches per-env sample /
        log_prob / value into state for the next record."""
        _ = step_idx
        self.act_step.step[
            Self.train_target, Self.ROLLOUT_LEN, Self.MINIBATCH, Self.N_ENVS,
        ](
            self.state, self.actor, self.critic,
            obs_ptr, action_ptr, self.action_scale,
        )

    def select_greedy_action(
        mut self,
        ref obs: List[Scalar[DT]],
        mut action_out: List[Scalar[DT]],
    ) raises:
        """Single-env greedy eval — always BATCH=1 even when state is
        sized for N_ENVS > 1 (eval bypasses the rollout buffer)."""
        self.act_step.step_greedy_n1[
            Self.train_target, Self.ROLLOUT_LEN, Self.MINIBATCH, Self.N_ENVS,
        ](self.state, self.actor, obs, action_out, self.action_scale)

    def record_transition(
        mut self,
        ref obs: List[Scalar[DT]],
        ref action: List[Scalar[DT]],
        reward: Scalar[DT],
        ref next_obs: List[Scalar[DT]],
        done: Scalar[DT],
    ) raises:
        """N=1 host-list wrapper. Only valid when Self.N_ENVS == 1.
        Bypasses `record_batch_cpu` to keep the legacy tracker pattern
        (per-step add_reward + driver-driven end_episode) and stay
        bit-identical to the pre-N_ENVS PPOTrainer at single-env."""
        comptime assert Self.N_ENVS == 1, (
            "PPOTrainer.record_transition: host-list wrapper only "
            "valid at N_ENVS=1; use record_batch_cpu for N_ENVS>1"
        )
        _ = action  # env-ready action ignored (cached unbounded used)
        for d in range(Self.OBS_DIM):
            self._obs1[d]  = obs[d]
            self._nobs1[d] = next_obs[d]
        self._rew1[0]  = reward
        self._done1[0] = done
        self.record_step.step[
            Self.train_target, Self.MINIBATCH, Self.N_ENVS,
        ](
            self.state, self._obs1, self._rew1, self._nobs1, self._done1,
        )
        self.tracker.add_reward(reward)

    def record_batch_cpu(
        mut self,
        obs_ptr: UnsafePointer[Scalar[DT], MutAnyOrigin],
        reward_ptr: UnsafePointer[Scalar[DT], MutAnyOrigin],
        next_obs_ptr: UnsafePointer[Scalar[DT], MutAnyOrigin],
        done_ptr: UnsafePointer[Scalar[DT], MutAnyOrigin],
    ) raises:
        """N_ENVS-wide transition record. Maintains a per-env running
        return sum (_ep_returns[e]); when done[e] is set, pushes the
        completed return into the EpisodeTracker via the same
        add_reward + end_episode pattern used by the N=1 wrapper."""
        self.record_step.step[
            Self.train_target, Self.MINIBATCH, Self.N_ENVS,
        ](self.state, obs_ptr, reward_ptr, next_obs_ptr, done_ptr)
        for e in range(Self.N_ENVS):
            self._ep_returns[e] += reward_ptr[e]
            if done_ptr[e] > Scalar[DT](0.5):
                # Push a single completed-episode return into the tracker
                # window using its add_reward + end_episode contract.
                self.tracker.add_reward(self._ep_returns[e])
                self.tracker.end_episode()
                self._ep_returns[e] = Scalar[DT](0.0)

    def mark_terminal(mut self) raises:
        """N=1 host-list wrapper — env 0 terminal."""
        comptime assert Self.N_ENVS == 1, (
            "PPOTrainer.mark_terminal: host-list wrapper only valid "
            "at N_ENVS=1; pass env_idx via mark_terminal_env"
        )
        self.mark_terminal_env(0)

    def mark_terminal_env(mut self, env_idx: Int) raises:
        self.record_step.mark_terminal[
            Self.train_target, Self.MINIBATCH, Self.N_ENVS,
        ](self.state, env_idx)

    def end_episode(mut self):
        self.tracker.end_episode()

    def train_step(mut self, step_idx: Int) raises -> Bool:
        _ = step_idx
        if self.state.rollout_idx < Self.ROLLOUT_LEN:
            return False

        # ── GAE: bootstrap V(s_T) per env + per-env backward pass.
        var t_gae = perf_counter_ns()
        self.gae_step.step[
            Self.train_target, Self.ACT_DIM, Self.MINIBATCH, Self.N_ENVS,
        ](self.state, self.critic, self.gamma, self.gae_lambda)
        self.timer.accumulate(Self._T_GAE, t_gae)

        # ── K-epoch minibatch SGD. Reset indices ONCE per rollout
        # (matches legacy ordering for bit-identity); epoch shuffles
        # operate on whatever state the previous epoch left behind.
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
                    Self.train_target, Self.ACT_DIM, Self.ROLLOUT_LEN,
                    Self.N_ENVS,
                ](self.state, self.critic, self.critic_opt)
                self._actor_L_accum += aL
                self._critic_L_accum += cL
                self._update_count += 1
                self._total_train_steps += 1
        self.timer.accumulate(Self._T_UPDATE, t_upd)

        # ── Reset rollout cursor + clear term buf.
        self.record_step.reset_rollout[
            Self.train_target, Self.MINIBATCH, Self.N_ENVS,
        ](self.state)
        return True

    def mean_return(self) -> Scalar[DT]:
        return self.tracker.mean_return()

    def ep_count(self) -> Int:
        return self.tracker.ep_count

    # ─── Logging surface (parity with SACTrainer) ────────────────────────

    def flush_train_log(
        mut self,
    ) -> Tuple[Scalar[DT], Scalar[DT], Int]:
        """Return (mean_actor_loss, mean_critic_loss, n_updates) since
        last flush. `n_updates` counts minibatch updates across the
        K-epoch SGD inside one train_step. Resets accumulators."""
        var n = self._update_count if self._update_count > 0 else 1
        var inv = Scalar[DT](1.0) / Scalar[DT](n)
        var out = (
            self._actor_L_accum * inv,
            self._critic_L_accum * inv,
            self._update_count,
        )
        self._actor_L_accum = Scalar[DT](0.0)
        self._critic_L_accum = Scalar[DT](0.0)
        self._update_count = 0
        return out

    def total_train_steps(self) -> Int:
        """Cumulative minibatch updates since trainer was made. Not reset
        by `flush_*`."""
        return self._total_train_steps

    def flush_metrics[
        L: Logger = NoOpLogger
    ](
        mut self,
        logger: Optional[UnsafePointer[L, MutAnyOrigin]] = None,
        step: Int = 0,
    ) raises -> PPOMetrics:
        """Drain accumulators into a PPOMetrics bundle. If a logger
        pointer is wired, also emit one log_scalar per metric field.
        Resets per-chunk accumulators on every call; the cumulative
        `_total_train_steps` counter is NOT reset."""
        var n = self._update_count if self._update_count > 0 else 1
        var inv = Scalar[DT](1.0) / Scalar[DT](n)
        var bundle = PPOMetrics(
            actor_loss=LogScalar[DT](self._actor_L_accum * inv),
            critic_loss=LogScalar[DT](self._critic_L_accum * inv),
            train_steps=LogScalar[DT](Scalar[DT](self._total_train_steps)),
            n_updates=LogScalar[DT](Scalar[DT](self._update_count)),
        )
        self._actor_L_accum = Scalar[DT](0.0)
        self._critic_L_accum = Scalar[DT](0.0)
        self._update_count = 0
        if Bool(logger):
            log_bundle(logger.value()[], bundle, step)
        return bundle^

    # ─── Trait-uniform cadence hooks (consumed by the driver) ─────────

    def flush_metrics_through_logger[L: Logger](
        mut self,
        logger: Optional[UnsafePointer[L, MutAnyOrigin]],
        step: Int,
    ) raises:
        """Trait-uniform passthrough: drains the PPO metric accumulators
        through `flush_metrics` and discards the typed bundle. The
        driver calls this at the user's `diag_every` cadence so no
        chunking is needed."""
        _ = self.flush_metrics[L](logger, step)

    def save_state(mut self, path: String) raises:
        """One-file v2 checkpoint of every PPO module + optimizer.
        Sections: `actor.*`, `critic.*`, `actor_opt.*`, `critic_opt.*`.
        Overwrites `path`. CPU-only; GPU save/load would need device→host
        sync first."""
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
        """Inverse of `save_state`. PPO has no target nets, so no
        hard-copy step is needed. On GPU the device params + Adam moments
        are restored via host staging (byte-identical on-disk format)."""
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
        """Per-section wall-time report (one line per sub-step:
        gae / update) and reset the accumulators. PPO's train_step only
        fires at rollout-length boundaries, so per-section costs are
        amortised across many env steps."""
        var report = self.timer.format_report()
        self.timer.reset()
        return report
