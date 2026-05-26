"""SACTrainerV2 — J.1 — SAC trainer composed via TrainerGraph[*BLOCKS].

Same algorithm + RNG-consumption order as SACTrainer. The 6 train_step
helpers (`_train_compute_target_y`, `_train_critic_update`,
`_train_actor_update`, `_train_alpha_update`, `_train_polyak`, plus the
inline sample branch) are lifted into 6 TrainerBlock-conforming structs;
`train_step` is now `state.alpha = fexp(alpha_opt.value); graph.step()`.

CPU only for J.1.c. GPU + PER come in J.1.d.

Bit-identity gate (J.1.c): seed=42 Pendulum 30k → mean_ret(10) = −167.572.
"""

from std.math import exp as fexp, log as flog, tanh as ftanh
from std.random import random_float64
from layout import TileTensor, row_major

from ..constants import DT
from ..core import Module
from ..core.online_target_pair import OnlineTargetPair
from ..core.scratch import Scratch
from ..core.scratch_walkers import init_scratch_auto
from ..initializer import Xavier
from ..optimizer.adam import Adam
from ..optimizer.scalar_adam import ScalarAdam
from ..loss.sac_actor_loss_cg import SACActorLossCG
from ..loss.critic_update_block import TwinCriticUpdateBlock
from ..combinators.trainer_graph import TrainerGraph
from .target_y_block import TargetYBlock
from .episode_tracker import EpisodeTracker
from .trainer_block import TrainerState
from .driver_cpu import OffPolicyTrainable
from .blocks.uniform_sample_cpu_block import UniformSampleCpuBlock
from .blocks.target_y_step_block import TargetYStepBlock
from .blocks.twin_critic_step_block import TwinCriticStepBlock
from .blocks.sac_actor_step_block import SACActorStepBlock
from .blocks.alpha_update_block import AlphaUpdateBlock
from .blocks.polyak_block import PolyakBlock


struct SACTrainerV2[
    ACTOR: Module,
    CRITIC: Module,
    OBS_DIM: Int,
    ACT_DIM: Int,
    BATCH: Int,
    REPLAY_CAPACITY: Int,
](OffPolicyTrainable):

    alias SampleB = UniformSampleCpuBlock[
        Self.OBS_DIM, Self.ACT_DIM, Self.BATCH, Self.REPLAY_CAPACITY,
    ]
    alias TargetYB = TargetYStepBlock[
        Self.OBS_DIM, Self.ACT_DIM, Self.BATCH, Self.ACTOR, Self.CRITIC,
    ]
    alias TwinB = TwinCriticStepBlock[
        Self.OBS_DIM, Self.ACT_DIM, Self.BATCH, Self.CRITIC,
    ]
    alias ActorB = SACActorStepBlock[
        Self.OBS_DIM, Self.ACT_DIM, Self.BATCH, Self.ACTOR, Self.CRITIC,
    ]
    alias AlphaB = AlphaUpdateBlock[
        Self.OBS_DIM, Self.ACT_DIM, Self.BATCH,
    ]
    alias PolyakB = PolyakBlock[
        Self.OBS_DIM, Self.ACT_DIM, Self.BATCH, Self.CRITIC,
    ]

    alias Graph = TrainerGraph[
        Self.OBS_DIM, Self.ACT_DIM, Self.BATCH,
        Self.SampleB, Self.TargetYB, Self.TwinB,
        Self.ActorB, Self.AlphaB, Self.PolyakB,
    ]

    var actor:       Self.ACTOR
    var pair1:       OnlineTargetPair[Self.CRITIC]
    var pair2:       OnlineTargetPair[Self.CRITIC]
    var actor_opt:   Adam
    var critic1_opt: Adam
    var critic2_opt: Adam
    var alpha_opt:   ScalarAdam

    var actor_loss: SACActorLossCG[Self.ACTOR, Self.CRITIC, Self.BATCH]
    var twin_critic_block: TwinCriticUpdateBlock[
        Self.CRITIC, Self.BATCH, Self.OBS_DIM, Self.ACT_DIM
    ]
    var target_y_block: TargetYBlock[
        Self.ACTOR, Self.CRITIC, Self.BATCH, Self.OBS_DIM, Self.ACT_DIM
    ]

    var graph:   Self.Graph
    var tracker: EpisodeTracker

    var _ob1:  Scratch["ob1",  Self.OBS_DIM, True]
    var _ao1:  Scratch["ao1",  2 * Self.ACT_DIM, True]
    var _alp1: Scratch["alp1", Self.ACT_DIM + 1, True]

    var gamma:           Scalar[DT]
    var tau:             Scalar[DT]
    var action_scale:    Scalar[DT]
    var target_entropy:  Scalar[DT]
    var learning_starts: Int

    var _actor_L_accum:  Scalar[DT]
    var _critic_L_accum: Scalar[DT]
    var _alpha_accum:    Scalar[DT]
    var _update_count:   Int

    def __init__(out self):
        self.actor = Self.ACTOR()
        self.pair1 = OnlineTargetPair[Self.CRITIC]()
        self.pair2 = OnlineTargetPair[Self.CRITIC]()
        self.actor_opt   = Adam()
        self.critic1_opt = Adam()
        self.critic2_opt = Adam()
        self.alpha_opt = ScalarAdam(
            value=0.0, m=0.0, v=0.0, t=0,
            lr=0.0003, beta1=0.9, beta2=0.999, eps=1e-8,
        )
        self.actor_loss = SACActorLossCG[
            Self.ACTOR, Self.CRITIC, Self.BATCH
        ]()
        self.twin_critic_block = TwinCriticUpdateBlock[
            Self.CRITIC, Self.BATCH, Self.OBS_DIM, Self.ACT_DIM
        ]()
        self.target_y_block = TargetYBlock[
            Self.ACTOR, Self.CRITIC, Self.BATCH, Self.OBS_DIM, Self.ACT_DIM
        ]()
        self.graph = Self.Graph()
        self.tracker = EpisodeTracker(
            window=List[Scalar[DT]](), window_size=0, idx=0,
            current_return=Scalar[DT](0.0), ep_count=0,
        )
        self._ob1  = Scratch["ob1",  Self.OBS_DIM, True]()
        self._ao1  = Scratch["ao1",  2 * Self.ACT_DIM, True]()
        self._alp1 = Scratch["alp1", Self.ACT_DIM + 1, True]()
        self.gamma = Scalar[DT](0.99)
        self.tau   = Scalar[DT](0.005)
        self.action_scale   = Scalar[DT](1.0)
        self.target_entropy = Scalar[DT](-1.0)
        self.learning_starts = 1_000
        self._actor_L_accum  = Scalar[DT](0.0)
        self._critic_L_accum = Scalar[DT](0.0)
        self._alpha_accum    = Scalar[DT](0.0)
        self._update_count   = 0

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
        max_grad_norm: Scalar[DT] = Scalar[DT](0.0),
    ) raises -> Self:
        comptime assert target == "cpu", (
            "SACTrainerV2.make[target='gpu'] — GPU comes in J.1.d"
        )
        var t = Self()
        t.actor = Self.ACTOR.make[target="cpu", INIT=Xavier]()
        t.pair1 = OnlineTargetPair[Self.CRITIC].make[
            target="cpu", INIT=Xavier
        ]()
        t.pair2 = OnlineTargetPair[Self.CRITIC].make[
            target="cpu", INIT=Xavier
        ]()
        t.actor_opt = Adam.make[target="cpu", M=Self.ACTOR](t.actor)
        t.actor_opt.lr = actor_lr
        t.actor_opt.max_grad_norm = max_grad_norm
        t.critic1_opt = Adam.make[target="cpu", M=Self.CRITIC](t.pair1.online)
        t.critic1_opt.lr = critic_lr
        t.critic1_opt.max_grad_norm = max_grad_norm
        t.critic2_opt = Adam.make[target="cpu", M=Self.CRITIC](t.pair2.online)
        t.critic2_opt.lr = critic_lr
        t.critic2_opt.max_grad_norm = max_grad_norm
        t.alpha_opt = ScalarAdam.new(flog(init_alpha), alpha_lr)
        t.actor_loss = SACActorLossCG[
            Self.ACTOR, Self.CRITIC, Self.BATCH
        ].make["cpu"](action_scale=action_scale)
        t.twin_critic_block = TwinCriticUpdateBlock[
            Self.CRITIC, Self.BATCH, Self.OBS_DIM, Self.ACT_DIM
        ].make["cpu"]()
        t.target_y_block = TargetYBlock[
            Self.ACTOR, Self.CRITIC, Self.BATCH, Self.OBS_DIM, Self.ACT_DIM
        ].make["cpu"](action_scale=action_scale, gamma=gamma)
        t.tracker = EpisodeTracker.new(
            window_size=window_size, initial_fill=initial_episode_fill
        )

        t.graph = Self.Graph()
        t.graph.state = TrainerState[
            Self.OBS_DIM, Self.ACT_DIM, Self.BATCH
        ].make_cpu()

        init_scratch_auto[Self, target="cpu"](t)

        t.gamma = gamma
        t.tau = tau
        t.action_scale = action_scale
        t.target_entropy = target_entropy
        t.learning_starts = learning_starts

        # SampleBlock owns its replay buffer (no model pointers).
        # Other blocks are rewired per-step (`_wire_blocks` from
        # `train_step`) because `return t^` here moves the trainer,
        # invalidating any `UnsafePointer(to=t.<field>)` taken here.
        # Matches ComputeGraph's `set_external` discipline.
        t.graph.blocks[0].setup(learning_starts)
        return t^

    def _wire_blocks(mut self):
        """Bind block→trainer-field pointers. Called from train_step
        every call (cost: ~6 pointer stores; mirror's ComputeGraph's
        `set_external` per-forward pattern). Catches any post-construction
        moves (e.g. if a trainer field is reseated)."""
        self.graph.blocks[1].bind(
            UnsafePointer(to=self.actor),
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
            UnsafePointer(to=self.actor),
            UnsafePointer(to=self.actor_opt),
            UnsafePointer(to=self.pair1.online),
            UnsafePointer(to=self.pair2.online),
            UnsafePointer(to=self.actor_loss),
        )
        self.graph.blocks[4].bind(
            UnsafePointer(to=self.alpha_opt), self.target_entropy,
        )
        self.graph.blocks[5].bind(
            UnsafePointer(to=self.pair1),
            UnsafePointer(to=self.pair2),
            self.tau,
        )

    # ─── OffPolicyTrainable surface ───────────────────────────────────

    def select_action(
        mut self,
        ref obs: List[Scalar[DT]],
        mut action_out: List[Scalar[DT]],
        step_idx: Int,
    ) raises:
        if step_idx < self.learning_starts:
            for j in range(Self.ACT_DIM):
                var u = Scalar[DT](2.0 * random_float64() - 1.0)
                action_out[j] = u * self.action_scale
            return
        var ob1_cpu_p = self._ob1.cpu_ptr()
        var ao1_cpu_p = self._ao1.cpu_ptr()
        var alp1_cpu_p = self._alp1.cpu_ptr()
        for d in range(Self.OBS_DIM):
            ob1_cpu_p[d] = obs[d]
        var ob1_t = TileTensor(ob1_cpu_p, row_major[1, Self.OBS_DIM]())
        var ao1_t = TileTensor(ao1_cpu_p, row_major[1, 2 * Self.ACT_DIM]())
        self.actor.forward["cpu", 1](ob1_t, output=ao1_t)
        var alp1_t = TileTensor(
            alp1_cpu_p, row_major[1, Self.ACT_DIM + 1]()
        )
        self.actor_loss.rsample.forward["cpu", 1](ao1_t, output=alp1_t)
        for j in range(Self.ACT_DIM):
            var a = alp1_cpu_p[j]
            if a > self.action_scale:
                a = self.action_scale
            elif a < -self.action_scale:
                a = -self.action_scale
            action_out[j] = a

    def select_greedy_action(
        mut self,
        ref obs: List[Scalar[DT]],
        mut action_out: List[Scalar[DT]],
    ) raises:
        var ob1_cpu_p = self._ob1.cpu_ptr()
        var ao1_cpu_p = self._ao1.cpu_ptr()
        for d in range(Self.OBS_DIM):
            ob1_cpu_p[d] = obs[d]
        var ob1_t = TileTensor(ob1_cpu_p, row_major[1, Self.OBS_DIM]())
        var ao1_t = TileTensor(ao1_cpu_p, row_major[1, 2 * Self.ACT_DIM]())
        self.actor.forward["cpu", 1](ob1_t, output=ao1_t)
        for j in range(Self.ACT_DIM):
            var mean = ao1_cpu_p[j]
            var a = ftanh(mean) * self.action_scale
            if a > self.action_scale:
                a = self.action_scale
            elif a < -self.action_scale:
                a = -self.action_scale
            action_out[j] = a

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
        self.graph.state.alpha = fexp(self.alpha_opt.value)
        var ran = self.graph.step["cpu"](step_idx)
        if not ran:
            return False
        self._actor_L_accum  += self.graph.state.actor_loss
        self._critic_L_accum += self.graph.state.critic_loss
        self._alpha_accum    += fexp(self.alpha_opt.value)
        self._update_count   += 1
        return True

    def mean_return(self) -> Scalar[DT]:
        return self.tracker.mean_return()

    def ep_count(self) -> Int:
        return self.tracker.ep_count
