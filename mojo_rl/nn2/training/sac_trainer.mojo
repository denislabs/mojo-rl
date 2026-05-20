"""SACTrainer — end-to-end SAC continuous trainer for nn2.

Phase 9B. The user-facing trainer for SAC continuous. Owns *everything*
the algorithm needs: networks, optimizers, loss-blocks, replay, tracker,
all forward/backward scratch, logging accumulators, and hyperparameters.

Exposed surface (user-visible API):
    SACTrainer[ACTOR, CRITIC, OBS_DIM, ACT_DIM, BATCH, REPLAY_CAPACITY].make["cpu"](...)
        builds the whole pipeline.
    trainer.select_action(obs_ptr, action_ptr_out, step_idx)
        warmup (uniform) vs policy (squashed-Gaussian) action, in-place.
    trainer.record(obs_ptr, action_ptr, reward, next_obs_ptr, done)
        push transition to replay, accumulate episode reward.
    trainer.end_episode()
        roll episode return into the tracker window.
    trainer.train_step(step_idx) -> Bool
        one full off-policy update if past warmup; no-op otherwise.
        Returns True when a training step actually ran.
    trainer.mean_return() / .ep_count() / .last_actor_loss() / .last_critic_loss()
        logging accessors.

Algorithm (CleanRL-style continuous SAC):
    target y       = r + γ·(min Q1_t(s', a'), Q2_t(s', a') − α·log_prob(a'|s'))
                     where a' = squashed-Gaussian sample from online actor π(·|s')
                     (NO target actor — SAC samples a' from the online policy).
    critic_i loss  = MSE(Q_i(s, a_buffer), y)            (i=1, 2; independent updates)
    actor loss     = mean_b ( α·log_prob(a|s) − min(Q1(s, a), Q2(s, a)) )
                     where (a, log_prob) = squashed-Gaussian rsample of π(·|s)
                     The composed-form chain is in `SACActorLoss` (Phase 9A).
    α update       = ScalarAdam.step( -( mean_b log_prob + target_entropy ) )
    target update  = Polyak τ-soft-update on both critic pairs.

Pendulum truncation gotcha: `done` from the env is the step-200 time-limit
truncation, NOT a real terminal. The target-y compute hard-codes
`nonterm = 1.0` — the value still bootstraps past `done`. See
`feedback_ppo_pendulum_timelimit_gae`. Real-terminal envs (LunarLander,
Hopper, ...) will eventually need a `gymnasium-style terminated vs
truncated` split through this code path; the current API does not yet
expose it.

CPU only in Phase 9B. The GPU path lands when the first GPU SAC env (e.g.
HalfCheetah on physics3d) ships through nn2.

Method boundaries are split to dodge the Mojo nightly inline-call-
explosion trap (memory: `feedback_mojo_function_inline_call_explosion` —
~20 sequential def-raises calls/function ceiling). `train_step` is a
6-call orchestrator over `_train_compute_target_y`, `_train_critic_update`,
`_train_actor_update`, `_train_alpha_update`, `_train_polyak`. Each
sub-method stays under the threshold.
"""

from std.math import exp as fexp, log as flog
from std.memory import alloc
from std.random import random_float64
from layout import TileTensor, TensorLayout, row_major

from ..constants import DT
from ..core import Module
from ..core.online_target_pair import OnlineTargetPair
from ..initializer import Xavier
from ..optimizer.adam import Adam
from ..optimizer.scalar_adam import ScalarAdam
from ..loss.sac_actor_loss_cg import SACActorLossCG as SACActorLoss, SACActorLossOut
from ..loss.critic_update_block import TwinCriticUpdateBlock
from ..data.cpu_replay import CPUReplay
from ..random.box_muller import box_muller_normal
from .episode_tracker import EpisodeTracker
from .target_y_block import TargetYBlock


struct SACTrainer[
    ACTOR: Module,
    CRITIC: Module,
    OBS_DIM: Int,
    ACT_DIM: Int,
    BATCH: Int,
    REPLAY_CAPACITY: Int,
](Movable & ImplicitlyDestructible):
    comptime SA_DIM = Self.OBS_DIM + Self.ACT_DIM

    # ─── Networks ─────────────────────────────────────────────────────
    var actor: Self.ACTOR
    var pair1: OnlineTargetPair[Self.CRITIC]
    var pair2: OnlineTargetPair[Self.CRITIC]

    # ─── Optimizers ───────────────────────────────────────────────────
    var actor_opt: Adam
    var critic1_opt: Adam
    var critic2_opt: Adam
    var alpha_opt: ScalarAdam

    # ─── Loss objects ─────────────────────────────────────────────────
    var actor_loss: SACActorLoss[Self.ACTOR, Self.CRITIC, Self.BATCH]
    var twin_critic_block: TwinCriticUpdateBlock[
        Self.CRITIC, Self.BATCH, Self.OBS_DIM, Self.ACT_DIM
    ]
    var target_y_block: TargetYBlock[
        Self.ACTOR, Self.CRITIC, Self.BATCH, Self.OBS_DIM, Self.ACT_DIM
    ]

    # ─── Replay + tracker ─────────────────────────────────────────────
    var buf: CPUReplay[Self.OBS_DIM, Self.ACT_DIM, Self.REPLAY_CAPACITY]
    var tracker: EpisodeTracker

    # ─── Single-step scratch (env interaction) ────────────────────────
    var _ob1: UnsafePointer[Scalar[DT], MutAnyOrigin]            # [OBS_DIM]
    var _ao1: UnsafePointer[Scalar[DT], MutAnyOrigin]            # [2*ACT_DIM]
    var _alp1: UnsafePointer[Scalar[DT], MutAnyOrigin]           # [ACT_DIM+1]

    # ─── Minibatch scratch (training) — only raw replay samples ──────
    # Target-y compute scratch is owned by `target_y_block` (Phase 10F).
    # Critic update scratch is owned by `twin_critic_block` (Phase 10F).
    var _mb_s: UnsafePointer[Scalar[DT], MutAnyOrigin]           # [BATCH, OBS]
    var _mb_a: UnsafePointer[Scalar[DT], MutAnyOrigin]           # [BATCH, ACT]
    var _mb_r: UnsafePointer[Scalar[DT], MutAnyOrigin]           # [BATCH]
    var _mb_sp: UnsafePointer[Scalar[DT], MutAnyOrigin]          # [BATCH, OBS]
    var _mb_d: UnsafePointer[Scalar[DT], MutAnyOrigin]           # [BATCH]
    var _mb_y: UnsafePointer[Scalar[DT], MutAnyOrigin]           # [BATCH, 1]

    # ─── Hyperparameters ──────────────────────────────────────────────
    var gamma: Scalar[DT]
    var tau: Scalar[DT]
    var action_scale: Scalar[DT]
    var target_entropy: Scalar[DT]
    var learning_starts: Int

    # ─── Logging accumulators ─────────────────────────────────────────
    var _actor_L_accum: Scalar[DT]
    var _critic_L_accum: Scalar[DT]
    var _alpha_accum: Scalar[DT]
    var _update_count: Int

    def __init__(out self):
        self.actor = Self.ACTOR()
        self.pair1 = OnlineTargetPair[Self.CRITIC]()
        self.pair2 = OnlineTargetPair[Self.CRITIC]()
        self.actor_opt = Adam()
        self.critic1_opt = Adam()
        self.critic2_opt = Adam()
        self.alpha_opt = ScalarAdam(
            value=0.0, m=0.0, v=0.0, t=0,
            lr=0.0003, beta1=0.9, beta2=0.999, eps=1e-8,
        )
        self.actor_loss = SACActorLoss[
            Self.ACTOR, Self.CRITIC, Self.BATCH
        ]()
        self.twin_critic_block = TwinCriticUpdateBlock[
            Self.CRITIC, Self.BATCH, Self.OBS_DIM, Self.ACT_DIM
        ]()
        self.target_y_block = TargetYBlock[
            Self.ACTOR, Self.CRITIC, Self.BATCH, Self.OBS_DIM, Self.ACT_DIM
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
        self._ob1 = null_p
        self._ao1 = null_p
        self._alp1 = null_p
        self._mb_s = null_p
        self._mb_a = null_p
        self._mb_r = null_p
        self._mb_sp = null_p
        self._mb_d = null_p
        self._mb_y = null_p
        self.gamma = Scalar[DT](0.99)
        self.tau = Scalar[DT](0.005)
        self.action_scale = Scalar[DT](1.0)
        self.target_entropy = Scalar[DT](-1.0)
        self.learning_starts = 1_000
        self._actor_L_accum = Scalar[DT](0.0)
        self._critic_L_accum = Scalar[DT](0.0)
        self._alpha_accum = Scalar[DT](0.0)
        self._update_count = 0

    @staticmethod
    def make[target: StaticString](
        actor_lr: Scalar[DT] = Scalar[DT](3e-4),
        critic_lr: Scalar[DT] = Scalar[DT](1e-3),
        alpha_lr: Scalar[DT] = Scalar[DT](3e-4),
        gamma: Scalar[DT] = Scalar[DT](0.99),
        tau: Scalar[DT] = Scalar[DT](0.005),
        action_scale: Scalar[DT] = Scalar[DT](1.0),
        init_alpha: Scalar[DT] = Scalar[DT](0.2),
        target_entropy: Scalar[DT] = Scalar[DT](-1.0),
        learning_starts: Int = 1_000,
        window_size: Int = 10,
        initial_episode_fill: Scalar[DT] = Scalar[DT](-1250.0),
    ) raises -> Self:
        """CPU factory. Builds nets via Xavier init, 3 Adams + 1 ScalarAdam,
        the Phase 9A actor-loss block, MSE loss, replay buffer, and tracker.
        Allocates all training scratch up front (no per-step allocation)."""
        comptime assert target == "cpu", (
            "SACTrainer.make[target='gpu'] not yet implemented (Phase 9B CPU only)"
        )
        var t = Self()
        t.actor = Self.ACTOR.make[target="cpu", INIT=Xavier]()
        t.pair1 = OnlineTargetPair[Self.CRITIC].make[
            target="cpu", INIT=Xavier
        ]()
        t.pair2 = OnlineTargetPair[Self.CRITIC].make[
            target="cpu", INIT=Xavier
        ]()
        t.actor_opt = Adam.make[target="cpu", M=Self.ACTOR](t.actor, lr=actor_lr)
        t.critic1_opt = Adam.make[target="cpu", M=Self.CRITIC](
            t.pair1.online, lr=critic_lr
        )
        t.critic2_opt = Adam.make[target="cpu", M=Self.CRITIC](
            t.pair2.online, lr=critic_lr
        )
        t.alpha_opt = ScalarAdam.new(flog(init_alpha), alpha_lr)
        t.actor_loss = SACActorLoss[
            Self.ACTOR, Self.CRITIC, Self.BATCH
        ].make["cpu"](action_scale=action_scale)
        t.twin_critic_block = TwinCriticUpdateBlock[
            Self.CRITIC, Self.BATCH, Self.OBS_DIM, Self.ACT_DIM
        ].make["cpu"]()
        t.target_y_block = TargetYBlock[
            Self.ACTOR, Self.CRITIC, Self.BATCH, Self.OBS_DIM, Self.ACT_DIM
        ].make["cpu"](action_scale=action_scale, gamma=gamma)
        t.buf = CPUReplay[
            Self.OBS_DIM, Self.ACT_DIM, Self.REPLAY_CAPACITY
        ].new()
        t.tracker = EpisodeTracker.new(
            window_size=window_size, initial_fill=initial_episode_fill
        )

        # Allocate all training scratch.
        t._ob1 = alloc[Scalar[DT]](Self.OBS_DIM)
        t._ao1 = alloc[Scalar[DT]](2 * Self.ACT_DIM)
        t._alp1 = alloc[Scalar[DT]](Self.ACT_DIM + 1)
        t._mb_s = alloc[Scalar[DT]](Self.BATCH * Self.OBS_DIM)
        t._mb_a = alloc[Scalar[DT]](Self.BATCH * Self.ACT_DIM)
        t._mb_r = alloc[Scalar[DT]](Self.BATCH)
        t._mb_sp = alloc[Scalar[DT]](Self.BATCH * Self.OBS_DIM)
        t._mb_d = alloc[Scalar[DT]](Self.BATCH)
        t._mb_y = alloc[Scalar[DT]](Self.BATCH)

        t.gamma = gamma
        t.tau = tau
        t.action_scale = action_scale
        t.target_entropy = target_entropy
        t.learning_starts = learning_starts
        return t^

    # ──────────────────────────────────────────────────────────────────
    # Env-interaction API
    # ──────────────────────────────────────────────────────────────────

    def select_action(
        mut self,
        obs: UnsafePointer[Scalar[DT], MutAnyOrigin],
        action_out: UnsafePointer[Scalar[DT], MutAnyOrigin],
        step_idx: Int,
    ) raises:
        """Sample action into `action_out` ([ACT_DIM]).

        Warmup (step_idx < learning_starts): uniform on [-action_scale, +action_scale].
        Else: actor.forward + rsample.forward at BATCH=1, write the
        squashed-Gaussian sample (no log_prob extracted here).

        Output is clamped to ±action_scale (the env's valid range).
        """
        if step_idx < self.learning_starts:
            for j in range(Self.ACT_DIM):
                var u = Scalar[DT](2.0 * random_float64() - 1.0)
                action_out[j] = u * self.action_scale
        else:
            for d in range(Self.OBS_DIM):
                self._ob1[d] = obs[d]
            var ob1_t = TileTensor(self._ob1, row_major[1, Self.OBS_DIM]())
            var ao1_t = TileTensor(self._ao1, row_major[1, 2 * Self.ACT_DIM]())
            self.actor.forward["cpu", 1](ob1_t, ao1_t)
            var alp1_t = TileTensor(self._alp1, row_major[1, Self.ACT_DIM + 1]())
            self.actor_loss.rsample.forward["cpu", 1](ao1_t, alp1_t)
            for j in range(Self.ACT_DIM):
                var a = self._alp1[j]
                if a > self.action_scale:
                    a = self.action_scale
                elif a < -self.action_scale:
                    a = -self.action_scale
                action_out[j] = a

    def record(
        mut self,
        obs: UnsafePointer[Scalar[DT], MutAnyOrigin],
        action: UnsafePointer[Scalar[DT], MutAnyOrigin],
        reward: Scalar[DT],
        next_obs: UnsafePointer[Scalar[DT], MutAnyOrigin],
        done: Scalar[DT],
    ):
        """Push (s, a, r, s', done) into replay; accumulate the episode reward."""
        self.buf.add(obs, action, reward, next_obs, done)
        self.tracker.add_reward(reward)

    def end_episode(mut self):
        """Roll the current episode return into the tracker window."""
        self.tracker.end_episode()

    # ──────────────────────────────────────────────────────────────────
    # Training step + sub-steps. Each sub-step stays under the Mojo
    # ~20-sequential-def-raises inline-explosion threshold.
    # ──────────────────────────────────────────────────────────────────

    def train_step(mut self, step_idx: Int) raises -> Bool:
        """One full off-policy SAC update if past warmup.

        Returns True if a training step actually ran, False if the call
        was skipped (warmup or under-filled buffer).
        """
        if step_idx < self.learning_starts:
            return False
        if self.buf.size < Self.BATCH:
            return False
        self.buf.sample(
            Self.BATCH, self._mb_s, self._mb_a, self._mb_r, self._mb_sp, self._mb_d
        )
        var alpha = fexp(self.alpha_opt.value)
        self._train_compute_target_y(alpha)
        var crit_loss = self._train_critic_update()
        var actor_res = self._train_actor_update(alpha)
        self._train_alpha_update(actor_res.log_prob_mean)
        self._train_polyak()

        self._actor_L_accum += actor_res.loss
        self._critic_L_accum += crit_loss
        self._alpha_accum += fexp(self.alpha_opt.value)
        self._update_count += 1
        return True

    def _train_compute_target_y(mut self, alpha: Scalar[DT]) raises:
        """Delegate to `target_y_block` (Phase 10F): computes
        y = r + γ·(min Q_target(s', a') − α·log_prob(a'|s')) in-place
        into `self._mb_y`. All target-y scratch lives in the block."""
        self.target_y_block.step["cpu"](
            self.actor, self.pair1.target_net, self.pair2.target_net,
            self._mb_sp, self._mb_r, alpha, self._mb_y,
        )

    def _train_critic_update(mut self) raises -> Scalar[DT]:
        """Twin-critic MSE step against shared target `mb_y`. Returns
        the sum of both critic losses (for logging). All scratch lives
        in `twin_critic_block` (Phase 10F)."""
        var mb_y_t = TileTensor(self._mb_y, row_major[Self.BATCH, 1]())
        return self.twin_critic_block.step["cpu"](
            self.pair1.online, self.critic1_opt,
            self.pair2.online, self.critic2_opt,
            self._mb_s, self._mb_a, mb_y_t,
        )

    def _train_actor_update(mut self, alpha: Scalar[DT]) raises -> SACActorLossOut:
        """One Phase 9A actor-loss-block call — does the full composed-
        form chain (rsample → twin critics frozen → ElemMin → Scale α →
        Sub → mean), plus actor.zero_grad/step."""
        return self.actor_loss.forward_backward["cpu", OPT=Adam](
            self.actor, self.actor_opt, self.pair1.online, self.pair2.online,
            self._mb_s, alpha,
        )

    def _train_alpha_update(mut self, log_prob_mean: Scalar[DT]):
        """Auto-tuned entropy temperature: minimize α·(−(H_target + H_curr)).
        ScalarAdam grad = -(log_prob_mean + target_entropy)."""
        self.alpha_opt.step(-(log_prob_mean + self.target_entropy))

    def _train_polyak(mut self) raises:
        """Polyak τ-soft-update both critic target nets."""
        self.pair1.polyak_step["cpu"](self.tau)
        self.pair2.polyak_step["cpu"](self.tau)

    # ──────────────────────────────────────────────────────────────────
    # Logging accessors
    # ──────────────────────────────────────────────────────────────────

    def mean_return(self) -> Scalar[DT]:
        return self.tracker.mean_return()

    def ep_count(self) -> Int:
        return self.tracker.ep_count

    def flush_train_log(mut self) -> Tuple[Scalar[DT], Scalar[DT], Scalar[DT], Int]:
        """Return (mean_actor_loss, mean_critic_loss, mean_alpha, n_updates)
        accumulated since the last flush. Resets accumulators."""
        var n = self._update_count if self._update_count > 0 else 1
        var inv = Scalar[DT](1.0) / Scalar[DT](n)
        var out = (
            self._actor_L_accum * inv,
            self._critic_L_accum * inv,
            self._alpha_accum * inv,
            self._update_count,
        )
        self._actor_L_accum = Scalar[DT](0.0)
        self._critic_L_accum = Scalar[DT](0.0)
        self._alpha_accum = Scalar[DT](0.0)
        self._update_count = 0
        return out


