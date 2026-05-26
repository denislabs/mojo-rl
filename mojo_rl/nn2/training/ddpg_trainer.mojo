"""DDPGTrainer — end-to-end DDPG continuous trainer for nn2 (Block E-4).

DDPG (Lillicrap et al., 2016) is the deterministic-policy off-policy
continuous-control algorithm SAC and TD3 evolved from. Compared to SAC:

  - Deterministic actor (single ACT_DIM output, no log_std head)
  - Single Q critic (vs SAC's twin)
  - Target actor + target critic (Polyak-updated)
  - Exploration via additive Gaussian noise (not sampled from policy)
  - No entropy temperature

Composes the Block E abstractions:

  - `OptimizerBundle[Adam, Adam]` for (actor_opt, critic_opt)
  - `ActionSamplingBlock` for warmup + deterministic-with-noise selection
  - `DDPGActorLoss` (DPG block) + `CriticUpdateBlock` (MSE) + `DDPGTargetYBlock`
  - `OnlineTargetPair[ACTOR]` + `OnlineTargetPair[CRITIC]` for the
    Polyak-updated target networks
  - `CPUReplay[OBS, ACT, CAP]` for transition storage

CPU only. GPU port is mechanical (all sub-blocks have GPU paths
already except `DDPGActorLoss` + `DDPGTargetYBlock`, which are
deferred for the same reason SAC's were initially: CPU smoke gates
correctness first).

Pendulum truncation gotcha: `done` is the step-200 time-limit (not a
real terminal). DDPGTargetYBlock hard-codes `nonterm=1.0`. See
`feedback_ppo_pendulum_timelimit_gae`.
"""

from std.math import exp as fexp
from std.memory import alloc
from layout import TileTensor, row_major

from ..constants import DT
from ..core import Module
from ..core.online_target_pair import OnlineTargetPair
from ..initializer import Xavier
from ..optimizer.adam import Adam
from ..optimizer.optimizer_bundle import OptimizerBundle
from ..loss.critic_update_block import CriticUpdateBlock
from ..loss.ddpg_actor_loss import DDPGActorLoss
from ..data.cpu_replay import CPUReplay
from mojo_rl.core.logger import Logger, NoOpLogger
from ..core.log_bundle import log_bundle
from ..core.metric import LogScalar
from .action_sampling_block import ActionSamplingBlock
from .ddpg_config import DDPGConfig
from .ddpg_metrics import DDPGMetrics
from .ddpg_target_y_block import DDPGTargetYBlock
from .driver_cpu import OffPolicyTrainable
from .episode_tracker import EpisodeTracker


struct DDPGTrainer[
    ACTOR: Module,
    CRITIC: Module,
    OBS_DIM: Int,
    ACT_DIM: Int,
    BATCH: Int,
    REPLAY_CAPACITY: Int,
](OffPolicyTrainable):
    comptime SA_DIM = Self.OBS_DIM + Self.ACT_DIM

    # ─── Networks ─────────────────────────────────────────────────────
    var actor_pair: OnlineTargetPair[Self.ACTOR]
    var critic_pair: OnlineTargetPair[Self.CRITIC]

    # ─── Optimizers (bundled — Block E-1) ─────────────────────────────
    # opts.items[0] = actor optimizer
    # opts.items[1] = critic optimizer
    var opts: OptimizerBundle[Adam, Adam]

    # ─── Loss blocks ──────────────────────────────────────────────────
    var actor_loss: DDPGActorLoss[Self.ACTOR, Self.CRITIC, Self.BATCH]
    var critic_block: CriticUpdateBlock[Self.CRITIC, Self.BATCH, Self.SA_DIM]
    var target_y_block: DDPGTargetYBlock[
        Self.ACTOR, Self.CRITIC, Self.BATCH, Self.OBS_DIM, Self.ACT_DIM
    ]

    # ─── Env-interaction (Block E-2) ──────────────────────────────────
    var policy_head: ActionSamplingBlock[
        Self.ACTOR, Self.OBS_DIM, Self.ACT_DIM, Self.ACT_DIM
    ]

    # ─── Replay + tracker ─────────────────────────────────────────────
    var buf: CPUReplay[Self.OBS_DIM, Self.ACT_DIM, Self.REPLAY_CAPACITY]
    var tracker: EpisodeTracker

    # ─── Minibatch raw scratch (replay samples + concat + y) ──────────
    var _mb_s: UnsafePointer[Scalar[DT], MutAnyOrigin]    # [BATCH, OBS]
    var _mb_a: UnsafePointer[Scalar[DT], MutAnyOrigin]    # [BATCH, ACT]
    var _mb_r: UnsafePointer[Scalar[DT], MutAnyOrigin]    # [BATCH]
    var _mb_sp: UnsafePointer[Scalar[DT], MutAnyOrigin]   # [BATCH, OBS]
    var _mb_d: UnsafePointer[Scalar[DT], MutAnyOrigin]    # [BATCH]
    var _mb_y: UnsafePointer[Scalar[DT], MutAnyOrigin]    # [BATCH]
    var _mb_sa: UnsafePointer[Scalar[DT], MutAnyOrigin]   # [BATCH, SA] (critic input)

    # ─── Hyperparameters ──────────────────────────────────────────────
    var gamma: Scalar[DT]
    var tau: Scalar[DT]
    var action_scale: Scalar[DT]
    var noise_scale: Scalar[DT]
    var learning_starts: Int

    # ─── Logging accumulators ─────────────────────────────────────────
    var _actor_L_accum: Scalar[DT]
    var _critic_L_accum: Scalar[DT]
    var _update_count: Int

    def __init__(out self):
        self.actor_pair = OnlineTargetPair[Self.ACTOR]()
        self.critic_pair = OnlineTargetPair[Self.CRITIC]()
        self.opts = OptimizerBundle[Adam, Adam]()
        self.actor_loss = DDPGActorLoss[
            Self.ACTOR, Self.CRITIC, Self.BATCH
        ]()
        self.critic_block = CriticUpdateBlock[
            Self.CRITIC, Self.BATCH, Self.SA_DIM
        ]()
        self.target_y_block = DDPGTargetYBlock[
            Self.ACTOR, Self.CRITIC, Self.BATCH, Self.OBS_DIM, Self.ACT_DIM
        ]()
        self.policy_head = ActionSamplingBlock[
            Self.ACTOR, Self.OBS_DIM, Self.ACT_DIM, Self.ACT_DIM
        ]()
        self.buf = CPUReplay[
            Self.OBS_DIM, Self.ACT_DIM, Self.REPLAY_CAPACITY
        ](
            obs=UnsafePointer[Scalar[DT], MutAnyOrigin](unsafe_from_address=0),
            act=UnsafePointer[Scalar[DT], MutAnyOrigin](unsafe_from_address=0),
            rew=UnsafePointer[Scalar[DT], MutAnyOrigin](unsafe_from_address=0),
            nxt=UnsafePointer[Scalar[DT], MutAnyOrigin](unsafe_from_address=0),
            dne=UnsafePointer[Scalar[DT], MutAnyOrigin](unsafe_from_address=0),
            size=0, pos=0,
        )
        self.tracker = EpisodeTracker(
            window=List[Scalar[DT]](),
            window_size=0, idx=0,
            current_return=Scalar[DT](0.0), ep_count=0,
        )
        var null_p = UnsafePointer[Scalar[DT], MutAnyOrigin](unsafe_from_address=0)
        self._mb_s = null_p
        self._mb_a = null_p
        self._mb_r = null_p
        self._mb_sp = null_p
        self._mb_d = null_p
        self._mb_y = null_p
        self._mb_sa = null_p
        self.gamma = Scalar[DT](0.99)
        self.tau = Scalar[DT](0.005)
        self.action_scale = Scalar[DT](1.0)
        self.noise_scale = Scalar[DT](0.1)
        self.learning_starts = 1_000
        self._actor_L_accum = Scalar[DT](0.0)
        self._critic_L_accum = Scalar[DT](0.0)
        self._update_count = 0

    @staticmethod
    def make[target: StaticString](config: DDPGConfig) raises -> Self:
        """Phase A.4 — Config-driven factory. Forwards to the keyword path."""
        return Self.make[target](
            actor_lr=config.actor_lr.v,
            critic_lr=config.critic_lr.v,
            gamma=config.gamma.v,
            tau=config.tau.v,
            action_scale=config.action_scale.v,
            noise_scale=config.noise_scale.v,
            learning_starts=config.learning_starts.v,
            window_size=config.window_size.v,
            initial_episode_fill=config.initial_episode_fill.v,
            max_grad_norm=config.max_grad_norm.v,
        )

    @staticmethod
    def make[target: StaticString](
        actor_lr: Scalar[DT] = Scalar[DT](1e-4),
        critic_lr: Scalar[DT] = Scalar[DT](1e-3),
        gamma: Scalar[DT] = Scalar[DT](0.99),
        tau: Scalar[DT] = Scalar[DT](0.005),
        action_scale: Scalar[DT] = Scalar[DT](1.0),
        noise_scale: Scalar[DT] = Scalar[DT](0.1),
        learning_starts: Int = 1_000,
        window_size: Int = 10,
        initial_episode_fill: Scalar[DT] = Scalar[DT](-1250.0),
        max_grad_norm: Scalar[DT] = Scalar[DT](0.0),
    ) raises -> Self:
        comptime assert target == "cpu", "DDPGTrainer: CPU only"
        var t = Self()
        t.actor_pair = OnlineTargetPair[Self.ACTOR].make[
            target="cpu", INIT=Xavier
        ]()
        t.critic_pair = OnlineTargetPair[Self.CRITIC].make[
            target="cpu", INIT=Xavier
        ]()

        # OptimizerBundle (E1): actor + critic Adam.
        t.opts = OptimizerBundle[Adam, Adam].make_default["cpu"]()
        t.opts.items[0] = Adam.make[target="cpu", M=Self.ACTOR](t.actor_pair.online)
        t.opts.items[0].lr = actor_lr
        t.opts.items[0].max_grad_norm = max_grad_norm
        t.opts.items[1] = Adam.make[target="cpu", M=Self.CRITIC](
            t.critic_pair.online
        )
        t.opts.items[1].lr = critic_lr
        t.opts.items[1].max_grad_norm = max_grad_norm

        t.actor_loss = DDPGActorLoss[
            Self.ACTOR, Self.CRITIC, Self.BATCH
        ].make["cpu"]()
        t.critic_block = CriticUpdateBlock[
            Self.CRITIC, Self.BATCH, Self.SA_DIM
        ].make["cpu"]()
        t.target_y_block = DDPGTargetYBlock[
            Self.ACTOR, Self.CRITIC, Self.BATCH, Self.OBS_DIM, Self.ACT_DIM
        ].make["cpu"](action_scale=action_scale, gamma=gamma)
        t.policy_head = ActionSamplingBlock[
            Self.ACTOR, Self.OBS_DIM, Self.ACT_DIM, Self.ACT_DIM
        ].make["cpu"]()
        t.buf = CPUReplay[
            Self.OBS_DIM, Self.ACT_DIM, Self.REPLAY_CAPACITY
        ].new()
        t.tracker = EpisodeTracker.new(
            window_size=window_size, initial_fill=initial_episode_fill
        )
        t._mb_s = alloc[Scalar[DT]](Self.BATCH * Self.OBS_DIM)
        t._mb_a = alloc[Scalar[DT]](Self.BATCH * Self.ACT_DIM)
        t._mb_r = alloc[Scalar[DT]](Self.BATCH)
        t._mb_sp = alloc[Scalar[DT]](Self.BATCH * Self.OBS_DIM)
        t._mb_d = alloc[Scalar[DT]](Self.BATCH)
        t._mb_y = alloc[Scalar[DT]](Self.BATCH)
        t._mb_sa = alloc[Scalar[DT]](Self.BATCH * Self.SA_DIM)

        t.gamma = gamma
        t.tau = tau
        t.action_scale = action_scale
        t.noise_scale = noise_scale
        t.learning_starts = learning_starts
        return t^

    # ─── Env-interaction API ──────────────────────────────────────────

    def select_action(
        mut self,
        ref obs: List[Scalar[DT]],
        mut action_out: List[Scalar[DT]],
        step_idx: Int,
    ) raises:
        self.policy_head.select_deterministic_with_noise["cpu"](
            self.actor_pair.online, obs, action_out,
            step_idx=step_idx, learning_starts=self.learning_starts,
            action_scale=self.action_scale, noise_scale=self.noise_scale,
        )

    def select_greedy_action(
        mut self,
        ref obs: List[Scalar[DT]],
        mut action_out: List[Scalar[DT]],
    ) raises:
        """Phase B.2 — deterministic greedy action for eval.

        DDPG's actor is already deterministic (one ACT_DIM output);
        eval just forwards through the online actor without adding the
        Gaussian exploration noise that `select_action` injects. Uses
        `ActionSamplingBlock.select_deterministic` with `step_idx`
        forced past the warmup window."""
        self.policy_head.select_deterministic["cpu"](
            self.actor_pair.online, obs, action_out,
            step_idx=1, learning_starts=0,
            action_scale=self.action_scale,
        )

    def record(
        mut self,
        ref obs: List[Scalar[DT]],
        ref action: List[Scalar[DT]],
        reward: Scalar[DT],
        ref next_obs: List[Scalar[DT]],
        done: Scalar[DT],
    ) raises:
        var obs_p = rebind[UnsafePointer[Scalar[DT], MutAnyOrigin]](
            obs.unsafe_ptr()
        )
        var act_p = rebind[UnsafePointer[Scalar[DT], MutAnyOrigin]](
            action.unsafe_ptr()
        )
        var nxt_p = rebind[UnsafePointer[Scalar[DT], MutAnyOrigin]](
            next_obs.unsafe_ptr()
        )
        self.buf.add(obs_p, act_p, reward, nxt_p, done)
        self.tracker.add_reward(reward)

    def end_episode(mut self):
        self.tracker.end_episode()

    # ─── Training step ────────────────────────────────────────────────

    def train_step(mut self, step_idx: Int) raises -> Bool:
        if step_idx < self.learning_starts:
            return False
        if self.buf.size < Self.BATCH:
            return False
        self.buf.sample(
            Self.BATCH, self._mb_s, self._mb_a, self._mb_r, self._mb_sp, self._mb_d,
        )

        # Target-y compute.
        self.target_y_block.step["cpu"](
            self.actor_pair.target_net,
            self.critic_pair.target_net,
            self._mb_sp, self._mb_r, self._mb_y,
        )

        # Critic update — MSE against y.
        var crit_loss = self._critic_update()

        # Actor update — DPG against the (now-updated) critic.
        var actor_loss_val = self.actor_loss.forward_backward["cpu", OPT=Adam](
            self.actor_pair.online, self.opts.items[0],
            self.critic_pair.online,
            self._mb_s,
        )

        # Polyak both pairs.
        self.actor_pair.polyak_step["cpu"](self.tau)
        self.critic_pair.polyak_step["cpu"](self.tau)

        self._actor_L_accum += actor_loss_val
        self._critic_L_accum += crit_loss
        self._update_count += 1
        return True

    def _critic_update(mut self) raises -> Scalar[DT]:
        # Build sa = concat(s, a) into _mb_sa.
        for b in range(Self.BATCH):
            for d in range(Self.OBS_DIM):
                self._mb_sa[b * Self.SA_DIM + d] = self._mb_s[b * Self.OBS_DIM + d]
            for j in range(Self.ACT_DIM):
                self._mb_sa[b * Self.SA_DIM + Self.OBS_DIM + j] = (
                    self._mb_a[b * Self.ACT_DIM + j]
                )
        var sa_t = TileTensor(self._mb_sa, row_major[Self.BATCH, Self.SA_DIM]())
        var y_t = TileTensor(self._mb_y, row_major[Self.BATCH, 1]())
        return self.critic_block.step["cpu"](
            self.critic_pair.online, self.opts.items[1], sa_t, y_t,
        )

    # ─── Logging accessors ────────────────────────────────────────────

    def mean_return(self) -> Scalar[DT]:
        return self.tracker.mean_return()

    def ep_count(self) -> Int:
        return self.tracker.ep_count

    def flush_train_log(mut self) -> Tuple[Scalar[DT], Scalar[DT], Int]:
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

    def flush_metrics[L: Logger = NoOpLogger](
        mut self,
        logger: Optional[UnsafePointer[L, MutAnyOrigin]] = None,
        step: Int = 0,
    ) raises -> DDPGMetrics:
        """Phase A.5 — Structured-logging variant of `flush_train_log`.
        Builds a DDPGMetrics bundle, optionally emits to a Logger, then
        resets accumulators. See `SACTrainer.flush_metrics` for design
        notes (zero-overhead short-circuit on NoOpLogger)."""
        var n = self._update_count if self._update_count > 0 else 1
        var inv = Scalar[DT](1.0) / Scalar[DT](n)
        var bundle = DDPGMetrics(
            actor_loss=LogScalar[DT](self._actor_L_accum * inv),
            critic_loss=LogScalar[DT](self._critic_L_accum * inv),
            n_updates=LogScalar[DT](Scalar[DT](self._update_count)),
        )
        self._actor_L_accum = Scalar[DT](0.0)
        self._critic_L_accum = Scalar[DT](0.0)
        self._update_count = 0
        if Bool(logger):
            log_bundle(logger.value()[], bundle, step)
        return bundle^
