"""TD3TrainerV2 — J.1.e — TD3 trainer composed via TrainerGraph[*BLOCKS].

CPU only. Differences from DDPGTrainerV2:
  - Twin critics (reuses SAC's TwinCriticStepBlock).
  - TD3TargetYBlock (target smoothing on a').
  - TD3DelayedActorPolyakBlock — bundled actor update + 3 polyaks gated
    by an internal `policy_delay` counter.

Block decomposition:
  TrainerGraph[
      UniformSampleCpuBlock,
      TD3TargetYStepBlock,
      TwinCriticStepBlock,           # reused from SAC
      TD3DelayedActorPolyakBlock,    # actor + 3 polyaks, gated
  ]
"""

from layout import TileTensor, row_major

from ..constants import DT
from ..core import Module
from ..core.online_target_pair import OnlineTargetPair
from ..initializer import Xavier
from ..optimizer.adam import Adam
from ..loss.ddpg_actor_loss import DDPGActorLoss
from ..loss.critic_update_block import TwinCriticUpdateBlock
from ..combinators.trainer_graph import TrainerGraph
from .td3_target_y_block import TD3TargetYBlock
from .action_sampling_block import ActionSamplingBlock
from .episode_tracker import EpisodeTracker
from .trainer_block import TrainerState
from .driver_cpu import OffPolicyTrainable
from .blocks.uniform_sample_cpu_block import UniformSampleCpuBlock
from .blocks.td3_target_y_step_block import TD3TargetYStepBlock
from .blocks.twin_critic_step_block import TwinCriticStepBlock
from .blocks.td3_delayed_actor_polyak_block import TD3DelayedActorPolyakBlock


struct TD3TrainerV2[
    ACTOR: Module,
    CRITIC: Module,
    OBS_DIM: Int,
    ACT_DIM: Int,
    BATCH: Int,
    REPLAY_CAPACITY: Int,
](OffPolicyTrainable):

    comptime SampleB = UniformSampleCpuBlock[
        Self.OBS_DIM, Self.ACT_DIM, Self.BATCH, Self.REPLAY_CAPACITY,
    ]
    comptime TargetYB = TD3TargetYStepBlock[
        Self.OBS_DIM, Self.ACT_DIM, Self.BATCH, Self.ACTOR, Self.CRITIC,
    ]
    comptime TwinB = TwinCriticStepBlock[
        Self.OBS_DIM, Self.ACT_DIM, Self.BATCH, Self.CRITIC,
    ]
    comptime ActorPolyakB = TD3DelayedActorPolyakBlock[
        Self.OBS_DIM, Self.ACT_DIM, Self.BATCH, Self.ACTOR, Self.CRITIC,
    ]

    comptime Graph = TrainerGraph[
        Self.OBS_DIM, Self.ACT_DIM, Self.BATCH,
        Self.SampleB, Self.TargetYB, Self.TwinB, Self.ActorPolyakB,
    ]

    var actor_pair:  OnlineTargetPair[Self.ACTOR]
    var pair1:       OnlineTargetPair[Self.CRITIC]
    var pair2:       OnlineTargetPair[Self.CRITIC]
    var actor_opt:   Adam
    var critic1_opt: Adam
    var critic2_opt: Adam

    var actor_loss: DDPGActorLoss[Self.ACTOR, Self.CRITIC, Self.BATCH]
    var twin_critic_block: TwinCriticUpdateBlock[
        Self.CRITIC, Self.BATCH, Self.OBS_DIM, Self.ACT_DIM
    ]
    var target_y_block: TD3TargetYBlock[
        Self.ACTOR, Self.CRITIC, Self.BATCH, Self.OBS_DIM, Self.ACT_DIM
    ]
    var policy_head: ActionSamplingBlock[
        Self.ACTOR, Self.OBS_DIM, Self.ACT_DIM, Self.ACT_DIM
    ]

    var graph:   Self.Graph
    var tracker: EpisodeTracker

    var gamma:             Scalar[DT]
    var tau:               Scalar[DT]
    var action_scale:      Scalar[DT]
    var exploration_noise: Scalar[DT]
    var policy_delay:      Int
    var learning_starts:   Int

    var _actor_L_accum:  Scalar[DT]
    var _critic_L_accum: Scalar[DT]
    var _actor_updates:  Int
    var _critic_updates: Int

    def __init__(out self):
        self.actor_pair = OnlineTargetPair[Self.ACTOR]()
        self.pair1 = OnlineTargetPair[Self.CRITIC]()
        self.pair2 = OnlineTargetPair[Self.CRITIC]()
        self.actor_opt = Adam()
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
        self.graph = Self.Graph()
        self.tracker = EpisodeTracker(
            window=List[Scalar[DT]](), window_size=0, idx=0,
            current_return=Scalar[DT](0.0), ep_count=0,
        )
        self.gamma = Scalar[DT](0.99)
        self.tau = Scalar[DT](0.005)
        self.action_scale = Scalar[DT](1.0)
        self.exploration_noise = Scalar[DT](0.1)
        self.policy_delay = 2
        self.learning_starts = 1_000
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
        max_grad_norm: Scalar[DT] = Scalar[DT](0.0),
    ) raises -> Self:
        comptime assert target == "cpu", "TD3TrainerV2: CPU only"
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
        t.actor_opt = Adam.make[target="cpu", M=Self.ACTOR](
            t.actor_pair.online
        )
        t.actor_opt.lr = actor_lr
        t.actor_opt.max_grad_norm = max_grad_norm
        t.critic1_opt = Adam.make[target="cpu", M=Self.CRITIC](t.pair1.online)
        t.critic1_opt.lr = critic_lr
        t.critic1_opt.max_grad_norm = max_grad_norm
        t.critic2_opt = Adam.make[target="cpu", M=Self.CRITIC](t.pair2.online)
        t.critic2_opt.lr = critic_lr
        t.critic2_opt.max_grad_norm = max_grad_norm

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
        t.tracker = EpisodeTracker.new(
            window_size=window_size, initial_fill=initial_episode_fill
        )

        t.graph = Self.Graph()
        t.graph.state = TrainerState[
            Self.OBS_DIM, Self.ACT_DIM, Self.BATCH
        ].make_cpu()

        t.gamma = gamma
        t.tau = tau
        t.action_scale = action_scale
        t.exploration_noise = exploration_noise
        t.policy_delay = policy_delay
        t.learning_starts = learning_starts

        t.graph.blocks[0].setup(learning_starts)
        t.graph.blocks[3].setup(policy_delay, tau)
        return t^

    def _wire_blocks(mut self):
        self.graph.blocks[1].bind(
            UnsafePointer(to=self.actor_pair.target_net),
            UnsafePointer(to=self.pair1.target_net),
            UnsafePointer(to=self.pair2.target_net),
            UnsafePointer(to=self.target_y_block),
        )
        self.graph.blocks[2].bind(
            UnsafePointer(to=self.pair1.online),
            UnsafePointer(to=self.critic1_opt),
            UnsafePointer(to=self.pair2.online),
            UnsafePointer(to=self.critic2_opt),
            UnsafePointer(to=self.twin_critic_block),
        )
        self.graph.blocks[3].bind(
            UnsafePointer(to=self.actor_pair.online),
            UnsafePointer(to=self.actor_opt),
            UnsafePointer(to=self.pair1.online),
            UnsafePointer(to=self.actor_pair),
            UnsafePointer(to=self.pair1),
            UnsafePointer(to=self.pair2),
            UnsafePointer(to=self.actor_loss),
        )

    # ─── OffPolicyTrainable surface ───────────────────────────────────

    def select_action(
        mut self,
        ref obs: List[Scalar[DT]],
        mut action_out: List[Scalar[DT]],
        step_idx: Int,
    ) raises:
        self.policy_head.select_deterministic_with_noise["cpu"](
            self.actor_pair.online, obs, action_out,
            step_idx=step_idx, learning_starts=self.learning_starts,
            action_scale=self.action_scale,
            noise_scale=self.exploration_noise,
        )

    def select_greedy_action(
        mut self,
        ref obs: List[Scalar[DT]],
        mut action_out: List[Scalar[DT]],
    ) raises:
        self.policy_head.select_deterministic_with_noise["cpu"](
            self.actor_pair.online, obs, action_out,
            step_idx=self.learning_starts,
            learning_starts=self.learning_starts,
            action_scale=self.action_scale,
            noise_scale=Scalar[DT](0.0),
        )

    def record(
        mut self,
        ref obs: List[Scalar[DT]],
        ref action: List[Scalar[DT]],
        reward: Scalar[DT],
        ref next_obs: List[Scalar[DT]],
        done: Scalar[DT],
    ) raises:
        self.tracker.add_reward(reward)
        self.graph.blocks[0].add(obs, action, reward, next_obs, done)

    def end_episode(mut self):
        self.tracker.end_episode()

    def train_step(mut self, step_idx: Int) raises -> Bool:
        self._wire_blocks()
        var ran = self.graph.step["cpu"](step_idx)
        if not ran:
            return False
        self._critic_L_accum += self.graph.state.critic_loss
        self._critic_updates += 1
        # actor update fires only every policy_delay steps; the
        # TD3DelayedActorPolyakBlock overwrites state.actor_loss only
        # when it actually ran. We detect that by checking if the
        # actor block's internal counter just reset (==0 after step).
        # Simpler: TD3DelayedActorPolyakBlock sets state.actor_loss
        # only on update; rely on the value being valid and bump
        # _actor_updates accordingly. Counter exposed via graph.
        if self.graph.blocks[3]._counter == 0:
            self._actor_L_accum += self.graph.state.actor_loss
            self._actor_updates += 1
        return True

    def mean_return(self) -> Scalar[DT]:
        return self.tracker.mean_return()

    def ep_count(self) -> Int:
        return self.tracker.ep_count
