"""TD3Trainer — end-to-end TD3 continuous trainer for nn2 (Block E-4).

TD3 (Fujimoto et al., 2018) hardens DDPG with three additions:

  1. **Twin critics + min** — target-y uses min(Q1, Q2) to fight Q
     overestimation. (SAC borrowed this.)
  2. **Target-policy smoothing** — adds clipped Gaussian noise to the
     target actor's action before evaluating target Qs. Smooths critic
     peaks → less brittle policy.
  3. **Delayed actor updates** — only update actor + target nets every
     `policy_delay` critic steps. Lets critic stabilize before policy
     pushes against it.

Composes the Block E abstractions:

  - `OptimizerBundle[Adam, Adam, Adam]` for (actor_opt, c1_opt, c2_opt)
  - `ActionSamplingBlock` for warmup + deterministic-with-noise selection
  - `DDPGActorLoss` (DPG block — identical math to DDPG, uses critic1)
  - `TwinCriticUpdateBlock` (MSE on both critics against shared y)
  - `TD3TargetYBlock` (clipped noise + twin-critic min)
  - `OnlineTargetPair[ACTOR]` + 2× `OnlineTargetPair[CRITIC]`

CPU only.

Pendulum truncation gotcha: `done` is the step-200 time-limit. Target-y
hard-codes `nonterm=1.0`. See `feedback_ppo_pendulum_timelimit_gae`.
"""

from std.memory import alloc
from layout import TileTensor, row_major

from ..constants import DT
from ..core import Module
from ..core.online_target_pair import OnlineTargetPair
from ..initializer import Xavier
from ..optimizer.adam import Adam
from ..optimizer.optimizer_bundle import OptimizerBundle
from ..loss.critic_update_block import TwinCriticUpdateBlock
from ..loss.ddpg_actor_loss import DDPGActorLoss
from ..data.cpu_replay import CPUReplay
from .action_sampling_block import ActionSamplingBlock
from .episode_tracker import EpisodeTracker
from .td3_target_y_block import TD3TargetYBlock


struct TD3Trainer[
    ACTOR: Module,
    CRITIC: Module,
    OBS_DIM: Int,
    ACT_DIM: Int,
    BATCH: Int,
    REPLAY_CAPACITY: Int,
](Movable & ImplicitlyDestructible):
    comptime SA_DIM = Self.OBS_DIM + Self.ACT_DIM

    # ─── Networks ─────────────────────────────────────────────────────
    var actor_pair: OnlineTargetPair[Self.ACTOR]
    var pair1: OnlineTargetPair[Self.CRITIC]
    var pair2: OnlineTargetPair[Self.CRITIC]

    # ─── Optimizers ───────────────────────────────────────────────────
    # Actor optimizer lives in `opts.items[0]` (OptimizerBundle). Critic
    # optimizers are kept as separate fields because Mojo nightly's
    # aliasing analyzer rejects two `mut bundle.items[i]` / `mut
    # bundle.items[j]` arguments passed to the same call (e.g.
    # `TwinCriticUpdateBlock.step` takes both critic opts in one call).
    # A bundle of N=1 is degenerate but preserves the field-grouping
    # affordance for the actor side. DreamerV3 will face the same
    # constraint and likely keep its 2-3 simultaneously-passed
    # optimizers as bare fields.
    var opts: OptimizerBundle[Adam]
    var critic1_opt: Adam
    var critic2_opt: Adam

    # ─── Loss blocks ──────────────────────────────────────────────────
    var actor_loss: DDPGActorLoss[Self.ACTOR, Self.CRITIC, Self.BATCH]
    var twin_critic_block: TwinCriticUpdateBlock[
        Self.CRITIC, Self.BATCH, Self.OBS_DIM, Self.ACT_DIM
    ]
    var target_y_block: TD3TargetYBlock[
        Self.ACTOR, Self.CRITIC, Self.BATCH, Self.OBS_DIM, Self.ACT_DIM
    ]

    # ─── Env-interaction (Block E-2) ──────────────────────────────────
    var policy_head: ActionSamplingBlock[
        Self.ACTOR, Self.OBS_DIM, Self.ACT_DIM, Self.ACT_DIM
    ]

    # ─── Replay + tracker ─────────────────────────────────────────────
    var buf: CPUReplay[Self.OBS_DIM, Self.ACT_DIM, Self.REPLAY_CAPACITY]
    var tracker: EpisodeTracker

    # ─── Minibatch raw scratch ────────────────────────────────────────
    var _mb_s: UnsafePointer[Scalar[DT], MutAnyOrigin]
    var _mb_a: UnsafePointer[Scalar[DT], MutAnyOrigin]
    var _mb_r: UnsafePointer[Scalar[DT], MutAnyOrigin]
    var _mb_sp: UnsafePointer[Scalar[DT], MutAnyOrigin]
    var _mb_d: UnsafePointer[Scalar[DT], MutAnyOrigin]
    var _mb_y: UnsafePointer[Scalar[DT], MutAnyOrigin]

    # ─── Hyperparameters ──────────────────────────────────────────────
    var gamma: Scalar[DT]
    var tau: Scalar[DT]
    var action_scale: Scalar[DT]
    var exploration_noise: Scalar[DT]
    var policy_delay: Int
    var learning_starts: Int
    var _critic_steps_since_actor: Int

    # ─── Logging accumulators ─────────────────────────────────────────
    var _actor_L_accum: Scalar[DT]
    var _critic_L_accum: Scalar[DT]
    var _actor_updates: Int
    var _critic_updates: Int

    def __init__(out self):
        self.actor_pair = OnlineTargetPair[Self.ACTOR]()
        self.pair1 = OnlineTargetPair[Self.CRITIC]()
        self.pair2 = OnlineTargetPair[Self.CRITIC]()
        self.opts = OptimizerBundle[Adam]()
        self.critic1_opt = Adam()
        self.critic2_opt = Adam()
        self.actor_loss = DDPGActorLoss[
            Self.ACTOR, Self.CRITIC, Self.BATCH
        ]()
        self.twin_critic_block = TwinCriticUpdateBlock[
            Self.CRITIC, Self.BATCH, Self.OBS_DIM, Self.ACT_DIM
        ]()
        self.target_y_block = TD3TargetYBlock[
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
        self.gamma = Scalar[DT](0.99)
        self.tau = Scalar[DT](0.005)
        self.action_scale = Scalar[DT](1.0)
        self.exploration_noise = Scalar[DT](0.1)
        self.policy_delay = 2
        self.learning_starts = 1_000
        self._critic_steps_since_actor = 0
        self._actor_L_accum = Scalar[DT](0.0)
        self._critic_L_accum = Scalar[DT](0.0)
        self._actor_updates = 0
        self._critic_updates = 0

    @staticmethod
    def make[target: StaticString](
        actor_lr: Scalar[DT] = Scalar[DT](3e-4),
        critic_lr: Scalar[DT] = Scalar[DT](3e-4),
        gamma: Scalar[DT] = Scalar[DT](0.99),
        tau: Scalar[DT] = Scalar[DT](0.005),
        action_scale: Scalar[DT] = Scalar[DT](1.0),
        exploration_noise: Scalar[DT] = Scalar[DT](0.1),
        target_policy_noise: Scalar[DT] = Scalar[DT](0.2),
        target_noise_clip: Scalar[DT] = Scalar[DT](0.5),
        policy_delay: Int = 2,
        learning_starts: Int = 1_000,
        window_size: Int = 10,
        initial_episode_fill: Scalar[DT] = Scalar[DT](-1250.0),
    ) raises -> Self:
        comptime assert target == "cpu", "TD3Trainer: CPU only"
        var t = Self()
        t.actor_pair = OnlineTargetPair[Self.ACTOR].make[
            target="cpu", INIT=Xavier
        ]()
        t.pair1 = OnlineTargetPair[Self.CRITIC].make[
            target="cpu", INIT=Xavier
        ]()
        t.pair2 = OnlineTargetPair[Self.CRITIC].make[
            target="cpu", INIT=Xavier
        ]()

        t.opts = OptimizerBundle[Adam].make_default["cpu"]()
        t.opts.items[0] = Adam.make[target="cpu", M=Self.ACTOR](t.actor_pair.online)
        t.opts.items[0].lr = actor_lr
        t.critic1_opt = Adam.make[target="cpu", M=Self.CRITIC](t.pair1.online)
        t.critic1_opt.lr = critic_lr
        t.critic2_opt = Adam.make[target="cpu", M=Self.CRITIC](t.pair2.online)
        t.critic2_opt.lr = critic_lr

        t.actor_loss = DDPGActorLoss[
            Self.ACTOR, Self.CRITIC, Self.BATCH
        ].make["cpu"]()
        t.twin_critic_block = TwinCriticUpdateBlock[
            Self.CRITIC, Self.BATCH, Self.OBS_DIM, Self.ACT_DIM
        ].make["cpu"]()
        t.target_y_block = TD3TargetYBlock[
            Self.ACTOR, Self.CRITIC, Self.BATCH, Self.OBS_DIM, Self.ACT_DIM
        ].make["cpu"](
            action_scale=action_scale, gamma=gamma,
            noise_std=target_policy_noise, noise_clip=target_noise_clip,
        )
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

        t.gamma = gamma
        t.tau = tau
        t.action_scale = action_scale
        t.exploration_noise = exploration_noise
        t.policy_delay = policy_delay
        t.learning_starts = learning_starts
        return t^

    # ─── Env-interaction API ──────────────────────────────────────────

    def select_action(
        mut self,
        obs: UnsafePointer[Scalar[DT], MutAnyOrigin],
        action_out: UnsafePointer[Scalar[DT], MutAnyOrigin],
        step_idx: Int,
    ) raises:
        self.policy_head.select_deterministic_with_noise["cpu"](
            self.actor_pair.online, obs, action_out,
            step_idx=step_idx, learning_starts=self.learning_starts,
            action_scale=self.action_scale, noise_scale=self.exploration_noise,
        )

    def record(
        mut self,
        obs: UnsafePointer[Scalar[DT], MutAnyOrigin],
        action: UnsafePointer[Scalar[DT], MutAnyOrigin],
        reward: Scalar[DT],
        next_obs: UnsafePointer[Scalar[DT], MutAnyOrigin],
        done: Scalar[DT],
    ):
        self.buf.add(obs, action, reward, next_obs, done)
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

        # 1. Target y: clipped-noise smoothed + min over twin target critics.
        self.target_y_block.step["cpu"](
            self.actor_pair.target_net,
            self.pair1.target_net,
            self.pair2.target_net,
            self._mb_sp, self._mb_r, self._mb_y,
        )

        # 2. Twin critic update against shared y.
        var mb_y_t = TileTensor(self._mb_y, row_major[Self.BATCH, 1]())
        var crit_loss = self.twin_critic_block.step["cpu"](
            self.pair1.online, self.critic1_opt,
            self.pair2.online, self.critic2_opt,
            self._mb_s, self._mb_a, mb_y_t,
        )
        self._critic_L_accum += crit_loss
        self._critic_updates += 1
        self._critic_steps_since_actor += 1

        # 3. Delayed actor + target update.
        if self._critic_steps_since_actor >= self.policy_delay:
            var actor_loss_val = self.actor_loss.forward_backward["cpu", OPT=Adam](
                self.actor_pair.online, self.opts.items[0],
                self.pair1.online, self._mb_s,
            )
            self.actor_pair.polyak_step["cpu"](self.tau)
            self.pair1.polyak_step["cpu"](self.tau)
            self.pair2.polyak_step["cpu"](self.tau)
            self._actor_L_accum += actor_loss_val
            self._actor_updates += 1
            self._critic_steps_since_actor = 0

        return True

    # ─── Logging accessors ────────────────────────────────────────────

    def mean_return(self) -> Scalar[DT]:
        return self.tracker.mean_return()

    def ep_count(self) -> Int:
        return self.tracker.ep_count

    def flush_train_log(mut self) -> Tuple[Scalar[DT], Scalar[DT], Int, Int]:
        var n_a = self._actor_updates if self._actor_updates > 0 else 1
        var n_c = self._critic_updates if self._critic_updates > 0 else 1
        var out = (
            self._actor_L_accum / Scalar[DT](n_a),
            self._critic_L_accum / Scalar[DT](n_c),
            self._actor_updates,
            self._critic_updates,
        )
        self._actor_L_accum = Scalar[DT](0.0)
        self._critic_L_accum = Scalar[DT](0.0)
        self._actor_updates = 0
        self._critic_updates = 0
        return out
