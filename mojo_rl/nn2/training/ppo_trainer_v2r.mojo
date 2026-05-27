"""PPOTrainerV2R — V2R ref-based block PPO trainer.

N_ENVS=1 in P.1/P.2; CPU bit-identical to legacy in P.1, GPU lifted in
P.2 (hybrid: per-step actor/critic forwards on device, rollout buffers
on host, K-epoch minibatch H2D-uploaded to device before train).
P.3 extends to N_ENVS multi-env via BatchedEnv.

Composes 6 step blocks via ref-based calls, holds `OnPolicyState` for
the shared per-step + per-rollout buffers:

  PPOActStep              — per env-step: actor.forward + sample + critic.forward
  PPORecordStep           — per env-step: push cached → rollout buffer
  PPOGAEStep              — per rollout: bootstrap + compute_gae
  PPOMinibatchGatherStep  — per epoch:    Fisher-Yates shuffle
                            per minibatch: gather + normalise mb_adv
  PPOActorTrainStep       — per minibatch: actor PPO clipped surrogate update
  PPOCriticTrainStep      — per minibatch: critic MSE update

Conforms to `OnPolicyAgent` (host-list select_action /
record_transition surface) so the existing `run_onpolicy_train`
driver works unmodified. Bit-identical to the legacy `PPOTrainer` at
CPU N_ENVS=1 by construction — same ops, same order, same data.
"""

from std.gpu.host import DeviceContext
from std.memory import alloc

from ..constants import DT
from ..core import Module
from ..combinators.sequential import Sequential
from ..initializer import Xavier
from ..optimizer.adam import Adam
from .episode_tracker import EpisodeTracker
from .onpolicy_state import OnPolicyState
from .driver_onpolicy import OnPolicyAgent
from .blocks import (
    PPOActStep,
    PPORecordStep,
    PPOGAEStep,
    PPOMinibatchGatherStep,
    PPOActorTrainStep,
    PPOCriticTrainStep,
)


struct PPOTrainerV2R[
    train_target: StaticString,
    ACTOR: Module,
    CRITIC: Module,
    OBS_DIM: Int,
    ACT_DIM: Int,
    ROLLOUT_LEN: Int,
    MINIBATCH: Int,
    N_EPOCHS: Int,
    N_ENVS: Int = 1,
](OnPolicyAgent):
    """V2R CleanRL-style PPO continuous trainer. N_ENVS defaults to 1
    so existing single-env consumers (host-list select_action / record_
    transition surface) stay bit-identical without code changes.
    N_ENVS > 1 is reached via the batched trainer methods consumed by
    `run_onpolicy_train_batched` (P.3 driver)."""

    comptime N_MINIBATCHES = (Self.ROLLOUT_LEN * Self.N_ENVS) // Self.MINIBATCH

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

    def __init__(out self):
        comptime assert (
            Self.train_target == "cpu" or Self.train_target == "gpu"
        ), "PPOTrainerV2R: train_target must be 'cpu' or 'gpu'"
        comptime assert Self.ACTOR.IN_DIMS[0] == Self.OBS_DIM, (
            "PPOTrainerV2R: ACTOR.IN_DIM must equal OBS_DIM"
        )
        comptime assert Self.ACTOR.OUT_DIM == 2 * Self.ACT_DIM, (
            "PPOTrainerV2R: ACTOR.OUT_DIM must equal 2 * ACT_DIM"
        )
        comptime assert Self.CRITIC.IN_DIMS[0] == Self.OBS_DIM, (
            "PPOTrainerV2R: CRITIC.IN_DIM must equal OBS_DIM"
        )
        comptime assert Self.CRITIC.OUT_DIM == 1, (
            "PPOTrainerV2R: CRITIC.OUT_DIM must equal 1"
        )
        comptime assert (Self.ROLLOUT_LEN * Self.N_ENVS) % Self.MINIBATCH == 0, (
            "PPOTrainerV2R: ROLLOUT_LEN * N_ENVS must be divisible by MINIBATCH"
        )
        comptime assert Self.N_ENVS >= 1, "PPOTrainerV2R: N_ENVS must be >= 1"
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
        ctx: Optional[DeviceContext] = None,
    ) raises -> Self:
        comptime assert (
            Self.train_target == "cpu" or Self.train_target == "gpu"
        ), "PPOTrainerV2R.make: train_target must be 'cpu' or 'gpu'"
        comptime if Self.train_target == "gpu":
            if not ctx:
                raise Error(
                    "PPOTrainerV2R.make[train_target='gpu']: ctx required"
                )
        var t = Self()
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
        t.critic_opt = Adam.make[target=Self.train_target, M=Self.CRITIC](
            t.critic, ctx=ctx,
        )
        t.critic_opt.lr = critic_lr
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
            "PPOTrainerV2R.select_action: host-list wrapper only valid "
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
            "PPOTrainerV2R.record_transition: host-list wrapper only "
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
            "PPOTrainerV2R.mark_terminal: host-list wrapper only valid "
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
        self.gae_step.step[
            Self.train_target, Self.ACT_DIM, Self.MINIBATCH, Self.N_ENVS,
        ](self.state, self.critic, self.gamma, self.gae_lambda)

        # ── K-epoch minibatch SGD. Reset indices ONCE per rollout
        # (matches legacy ordering for bit-identity); epoch shuffles
        # operate on whatever state the previous epoch left behind.
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
                _ = self.actor_train.step[
                    Self.train_target, Self.ROLLOUT_LEN, Self.N_ENVS,
                ](self.state, self.actor, self.actor_opt)
                _ = self.critic_train.step[
                    Self.train_target, Self.ACT_DIM, Self.ROLLOUT_LEN,
                    Self.N_ENVS,
                ](self.state, self.critic, self.critic_opt)

        # ── Reset rollout cursor + clear term buf.
        self.record_step.reset_rollout[
            Self.train_target, Self.MINIBATCH, Self.N_ENVS,
        ](self.state)
        return True

    def mean_return(self) -> Scalar[DT]:
        return self.tracker.mean_return()

    def ep_count(self) -> Int:
        return self.tracker.ep_count
