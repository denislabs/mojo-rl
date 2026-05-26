"""DDPGTrainerV2 — J.1.e — DDPG trainer composed via TrainerGraph[*BLOCKS].

CPU only. Reuses the existing DDPGActorLoss, DDPGTargetYBlock,
CriticUpdateBlock as inner blocks; wraps them in TrainerBlocks.

Block decomposition:
  TrainerGraph[
      UniformSampleCpuBlock,
      DDPGTargetYStepBlock,
      SingleCriticStepBlock,
      DDPGActorStepBlock,
      DDPGPolyakBlock,
  ]
"""

from std.random import random_float64
from layout import TileTensor, row_major

from ..constants import DT
from ..core import Module
from ..core.online_target_pair import OnlineTargetPair
from ..initializer import Xavier
from ..optimizer.adam import Adam
from ..loss.ddpg_actor_loss import DDPGActorLoss
from ..loss.critic_update_block import CriticUpdateBlock
from ..combinators.trainer_graph import TrainerGraph
from .ddpg_target_y_block import DDPGTargetYBlock
from .action_sampling_block import ActionSamplingBlock
from .episode_tracker import EpisodeTracker
from .trainer_block import TrainerState
from .driver_cpu import OffPolicyTrainable
from .blocks.uniform_sample_cpu_block import UniformSampleCpuBlock
from .blocks.ddpg_target_y_step_block import DDPGTargetYStepBlock
from .blocks.single_critic_step_block import SingleCriticStepBlock
from .blocks.ddpg_actor_step_block import DDPGActorStepBlock
from .blocks.ddpg_polyak_block import DDPGPolyakBlock


struct DDPGTrainerV2[
    ACTOR: Module,
    CRITIC: Module,
    OBS_DIM: Int,
    ACT_DIM: Int,
    BATCH: Int,
    REPLAY_CAPACITY: Int,
](OffPolicyTrainable):
    comptime SA_DIM = Self.OBS_DIM + Self.ACT_DIM

    comptime SampleB = UniformSampleCpuBlock[
        Self.OBS_DIM, Self.ACT_DIM, Self.BATCH, Self.REPLAY_CAPACITY,
    ]
    comptime TargetYB = DDPGTargetYStepBlock[
        Self.OBS_DIM, Self.ACT_DIM, Self.BATCH, Self.ACTOR, Self.CRITIC,
    ]
    comptime CriticB = SingleCriticStepBlock[
        Self.OBS_DIM, Self.ACT_DIM, Self.BATCH, Self.CRITIC,
    ]
    comptime ActorB = DDPGActorStepBlock[
        Self.OBS_DIM, Self.ACT_DIM, Self.BATCH, Self.ACTOR, Self.CRITIC,
    ]
    comptime PolyakB = DDPGPolyakBlock[
        Self.OBS_DIM, Self.ACT_DIM, Self.BATCH, Self.ACTOR, Self.CRITIC,
    ]

    comptime Graph = TrainerGraph[
        Self.OBS_DIM, Self.ACT_DIM, Self.BATCH,
        Self.SampleB, Self.TargetYB, Self.CriticB,
        Self.ActorB, Self.PolyakB,
    ]

    var actor_pair:  OnlineTargetPair[Self.ACTOR]
    var critic_pair: OnlineTargetPair[Self.CRITIC]
    var actor_opt:   Adam
    var critic_opt:  Adam

    var actor_loss: DDPGActorLoss[Self.ACTOR, Self.CRITIC, Self.BATCH]
    var critic_block: CriticUpdateBlock[
        Self.CRITIC, Self.BATCH, Self.SA_DIM
    ]
    var target_y_block: DDPGTargetYBlock[
        Self.ACTOR, Self.CRITIC, Self.BATCH, Self.OBS_DIM, Self.ACT_DIM
    ]
    var policy_head: ActionSamplingBlock[
        Self.ACTOR, Self.OBS_DIM, Self.ACT_DIM, Self.ACT_DIM
    ]

    var graph:   Self.Graph
    var tracker: EpisodeTracker

    var gamma:           Scalar[DT]
    var tau:             Scalar[DT]
    var action_scale:    Scalar[DT]
    var noise_scale:     Scalar[DT]
    var learning_starts: Int

    var _actor_L_accum:  Scalar[DT]
    var _critic_L_accum: Scalar[DT]
    var _update_count:   Int

    def __init__(out self):
        self.actor_pair = OnlineTargetPair[Self.ACTOR]()
        self.critic_pair = OnlineTargetPair[Self.CRITIC]()
        self.actor_opt = Adam()
        self.critic_opt = Adam()
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
        self.graph = Self.Graph()
        self.tracker = EpisodeTracker(
            window=List[Scalar[DT]](), window_size=0, idx=0,
            current_return=Scalar[DT](0.0), ep_count=0,
        )
        self.gamma = Scalar[DT](0.99)
        self.tau = Scalar[DT](0.005)
        self.action_scale = Scalar[DT](1.0)
        self.noise_scale = Scalar[DT](0.1)
        self.learning_starts = 1_000
        self._actor_L_accum = Scalar[DT](0.0)
        self._critic_L_accum = Scalar[DT](0.0)
        self._update_count = 0

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
        comptime assert target == "cpu", "DDPGTrainerV2: CPU only"
        var t = Self()
        t.actor_pair = OnlineTargetPair[Self.ACTOR].make[
            target="cpu", INIT=Xavier
        ]()
        t.critic_pair = OnlineTargetPair[Self.CRITIC].make[
            target="cpu", INIT=Xavier
        ]()
        t.actor_opt = Adam.make[target="cpu", M=Self.ACTOR](t.actor_pair.online)
        t.actor_opt.lr = actor_lr
        t.actor_opt.max_grad_norm = max_grad_norm
        t.critic_opt = Adam.make[target="cpu", M=Self.CRITIC](
            t.critic_pair.online
        )
        t.critic_opt.lr = critic_lr
        t.critic_opt.max_grad_norm = max_grad_norm
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
        t.noise_scale = noise_scale
        t.learning_starts = learning_starts

        t.graph.blocks[0].setup(learning_starts)
        t.graph.blocks[2].setup()  # SingleCriticStepBlock allocates _mb_sa
        return t^

    def _wire_blocks(mut self):
        self.graph.blocks[1].bind(
            UnsafePointer(to=self.actor_pair.target_net),
            UnsafePointer(to=self.critic_pair.target_net),
            UnsafePointer(to=self.target_y_block),
        )
        self.graph.blocks[2].bind(
            UnsafePointer(to=self.critic_pair.online),
            UnsafePointer(to=self.critic_opt),
            UnsafePointer(to=self.critic_block),
        )
        self.graph.blocks[3].bind(
            UnsafePointer(to=self.actor_pair.online),
            UnsafePointer(to=self.actor_opt),
            UnsafePointer(to=self.critic_pair.online),
            UnsafePointer(to=self.actor_loss),
        )
        self.graph.blocks[4].bind(
            UnsafePointer(to=self.actor_pair),
            UnsafePointer(to=self.critic_pair),
            self.tau,
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
            action_scale=self.action_scale, noise_scale=self.noise_scale,
        )

    def select_greedy_action(
        mut self,
        ref obs: List[Scalar[DT]],
        mut action_out: List[Scalar[DT]],
    ) raises:
        self.policy_head.select_deterministic_with_noise["cpu"](
            self.actor_pair.online, obs, action_out,
            step_idx=self.learning_starts,  # past warmup → no random
            learning_starts=self.learning_starts,
            action_scale=self.action_scale,
            noise_scale=Scalar[DT](0.0),    # zero noise = greedy
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
        self._actor_L_accum  += self.graph.state.actor_loss
        self._critic_L_accum += self.graph.state.critic_loss
        self._update_count   += 1
        return True

    def mean_return(self) -> Scalar[DT]:
        return self.tracker.mean_return()

    def ep_count(self) -> Int:
        return self.tracker.ep_count
