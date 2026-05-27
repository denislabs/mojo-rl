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
](OnPolicyAgent):
    """V2R CleanRL-style PPO continuous trainer. CPU N_ENVS=1 in P.1."""

    comptime N_MINIBATCHES = Self.ROLLOUT_LEN // Self.MINIBATCH

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
    ]

    # ── Hyperparameters ──────────────────────────────────────────────
    var gamma: Scalar[DT]
    var gae_lambda: Scalar[DT]
    var clip_eps: Scalar[DT]
    var entropy_coef: Scalar[DT]
    var action_scale: Scalar[DT]

    # ── Episode tracker ──────────────────────────────────────────────
    var tracker: EpisodeTracker

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
        comptime assert Self.ROLLOUT_LEN % Self.MINIBATCH == 0, (
            "PPOTrainerV2R: ROLLOUT_LEN must be divisible by MINIBATCH"
        )
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
        ]()
        self.gamma = Scalar[DT](0.99)
        self.gae_lambda = Scalar[DT](0.95)
        self.clip_eps = Scalar[DT](0.2)
        self.entropy_coef = Scalar[DT](0.0)
        self.action_scale = Scalar[DT](1.0)
        self.tracker = EpisodeTracker.new(
            window_size=10, initial_fill=Scalar[DT](-1600.0),
        )

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
        ].make[Self.train_target](ctx=ctx)
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
        _ = step_idx
        self.act_step.step[Self.train_target, Self.ROLLOUT_LEN, Self.MINIBATCH](
            self.state, self.actor, self.critic, obs, action_out, self.action_scale,
        )

    def select_greedy_action(
        mut self,
        ref obs: List[Scalar[DT]],
        mut action_out: List[Scalar[DT]],
    ) raises:
        self.act_step.step_greedy[
            Self.train_target, Self.ROLLOUT_LEN, Self.MINIBATCH,
        ](self.state, self.actor, obs, action_out, self.action_scale)

    def record_transition(
        mut self,
        ref obs: List[Scalar[DT]],
        ref action: List[Scalar[DT]],
        reward: Scalar[DT],
        ref next_obs: List[Scalar[DT]],
        done: Scalar[DT],
    ) raises:
        _ = action  # env-ready action ignored (cached unbounded used)
        self.record_step.step[Self.train_target, Self.MINIBATCH](
            self.state, obs, reward, next_obs, done,
        )
        self.tracker.add_reward(reward)

    def mark_terminal(mut self) raises:
        self.record_step.mark_terminal[Self.train_target, Self.MINIBATCH](
            self.state,
        )

    def end_episode(mut self):
        self.tracker.end_episode()

    def train_step(mut self, step_idx: Int) raises -> Bool:
        _ = step_idx
        if self.state.rollout_idx < Self.ROLLOUT_LEN:
            return False

        # ── GAE: bootstrap V(s_T) + backward pass.
        self.gae_step.step[Self.train_target, Self.ACT_DIM, Self.MINIBATCH](
            self.state, self.critic, self.gamma, self.gae_lambda,
        )

        # ── K-epoch minibatch SGD. Reset indices ONCE per rollout
        # (matches legacy ordering for bit-identity); epoch shuffles
        # operate on whatever state the previous epoch left behind.
        self.gather_step.reset_indices[Self.train_target](self.state)
        for _epoch in range(Self.N_EPOCHS):
            self.gather_step.shuffle_epoch[Self.train_target](self.state)
            for mb in range(Self.N_MINIBATCHES):
                self.gather_step.gather[Self.train_target](self.state, mb)
                _ = self.actor_train.step[
                    Self.train_target, Self.ROLLOUT_LEN,
                ](self.state, self.actor, self.actor_opt)
                _ = self.critic_train.step[
                    Self.train_target, Self.ACT_DIM, Self.ROLLOUT_LEN,
                ](self.state, self.critic, self.critic_opt)

        # ── Reset rollout cursor + clear term buf.
        self.record_step.reset_rollout[Self.train_target, Self.MINIBATCH](
            self.state,
        )
        return True

    def mean_return(self) -> Scalar[DT]:
        return self.tracker.mean_return()

    def ep_count(self) -> Int:
        return self.tracker.ep_count
