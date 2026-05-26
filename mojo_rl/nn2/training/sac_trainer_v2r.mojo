"""SACTrainerV2R — J.1.g-redesign-v2 — SAC via ref-based block calls.

NO TrainerGraph. NO bind. NO _wire_blocks. NO PortPack. NO UnsafePointer
fields on blocks.

Each block (in `blocks_ref/`) is a small struct that owns its inner
LossBlock if any, and exposes `step[target](mut state, ref/mut ...)`.
The trainer's `train_step` body IS the pipeline.

Bit-identity gate: seed=42 Pendulum 30k → mean_ret(10) = -169.04118
(matches SACTrainer / SACTrainerV2 exactly).
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
from .episode_tracker import EpisodeTracker
from .trainer_block import TrainerState
from .driver_cpu import OffPolicyTrainable
from .blocks_ref import (
    UniformSampleCpuStep,
    TargetYStep,
    TwinCriticStep,
    SACActorStep,
    AlphaUpdateStep,
    PolyakStep,
)


struct SACTrainerV2R[
    ACTOR: Module,
    CRITIC: Module,
    OBS_DIM: Int,
    ACT_DIM: Int,
    BATCH: Int,
    REPLAY_CAPACITY: Int,
](OffPolicyTrainable):

    comptime AGENT_OBS_DIM: Int = Self.OBS_DIM
    comptime AGENT_ACT_DIM: Int = Self.ACT_DIM

    var actor:       Self.ACTOR
    var pair1:       OnlineTargetPair[Self.CRITIC]
    var pair2:       OnlineTargetPair[Self.CRITIC]
    var actor_opt:   Adam
    var critic1_opt: Adam
    var critic2_opt: Adam
    var alpha_opt:   ScalarAdam

    var sample_blk: UniformSampleCpuStep[
        Self.OBS_DIM, Self.ACT_DIM, Self.BATCH, Self.REPLAY_CAPACITY,
    ]
    var target_y_blk: TargetYStep[
        Self.OBS_DIM, Self.ACT_DIM, Self.BATCH, Self.ACTOR, Self.CRITIC,
    ]
    var twin_critic_blk: TwinCriticStep[
        Self.OBS_DIM, Self.ACT_DIM, Self.BATCH, Self.CRITIC,
    ]
    var actor_blk: SACActorStep[
        Self.OBS_DIM, Self.ACT_DIM, Self.BATCH, Self.ACTOR, Self.CRITIC,
    ]
    var alpha_blk: AlphaUpdateStep[Self.OBS_DIM, Self.ACT_DIM, Self.BATCH]
    var polyak_blk: PolyakStep[
        Self.OBS_DIM, Self.ACT_DIM, Self.BATCH, Self.CRITIC,
    ]

    var state:   TrainerState[Self.OBS_DIM, Self.ACT_DIM, Self.BATCH]
    var tracker: EpisodeTracker

    var _ob1:  Scratch["ob1",  Self.OBS_DIM, True]
    var _ao1:  Scratch["ao1",  2 * Self.ACT_DIM, True]
    var _alp1: Scratch["alp1", Self.ACT_DIM + 1, True]

    var action_scale:    Scalar[DT]
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
        self.sample_blk = UniformSampleCpuStep[
            Self.OBS_DIM, Self.ACT_DIM, Self.BATCH, Self.REPLAY_CAPACITY,
        ]()
        self.target_y_blk = TargetYStep[
            Self.OBS_DIM, Self.ACT_DIM, Self.BATCH, Self.ACTOR, Self.CRITIC,
        ]()
        self.twin_critic_blk = TwinCriticStep[
            Self.OBS_DIM, Self.ACT_DIM, Self.BATCH, Self.CRITIC,
        ]()
        self.actor_blk = SACActorStep[
            Self.OBS_DIM, Self.ACT_DIM, Self.BATCH, Self.ACTOR, Self.CRITIC,
        ]()
        self.alpha_blk = AlphaUpdateStep[
            Self.OBS_DIM, Self.ACT_DIM, Self.BATCH,
        ]()
        self.polyak_blk = PolyakStep[
            Self.OBS_DIM, Self.ACT_DIM, Self.BATCH, Self.CRITIC,
        ]()
        self.state = TrainerState[
            Self.OBS_DIM, Self.ACT_DIM, Self.BATCH,
        ]()
        self.tracker = EpisodeTracker(
            window=List[Scalar[DT]](), window_size=0, idx=0,
            current_return=Scalar[DT](0.0), ep_count=0,
        )
        self._ob1  = Scratch["ob1",  Self.OBS_DIM, True]()
        self._ao1  = Scratch["ao1",  2 * Self.ACT_DIM, True]()
        self._alp1 = Scratch["alp1", Self.ACT_DIM + 1, True]()
        self.action_scale = Scalar[DT](1.0)
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
            "SACTrainerV2R.make[target='gpu'] — use SACTrainerV2RGpu"
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

        t.target_y_blk = TargetYStep[
            Self.OBS_DIM, Self.ACT_DIM, Self.BATCH, Self.ACTOR, Self.CRITIC,
        ].make["cpu"](action_scale=action_scale, gamma=gamma)
        t.twin_critic_blk = TwinCriticStep[
            Self.OBS_DIM, Self.ACT_DIM, Self.BATCH, Self.CRITIC,
        ].make["cpu"]()
        t.actor_blk = SACActorStep[
            Self.OBS_DIM, Self.ACT_DIM, Self.BATCH, Self.ACTOR, Self.CRITIC,
        ].make["cpu"](action_scale=action_scale)
        t.alpha_blk = AlphaUpdateStep[
            Self.OBS_DIM, Self.ACT_DIM, Self.BATCH,
        ].make(target_entropy=target_entropy)
        t.polyak_blk = PolyakStep[
            Self.OBS_DIM, Self.ACT_DIM, Self.BATCH, Self.CRITIC,
        ].make(tau=tau)

        t.tracker = EpisodeTracker.new(
            window_size=window_size, initial_fill=initial_episode_fill
        )

        t.state = TrainerState[
            Self.OBS_DIM, Self.ACT_DIM, Self.BATCH,
        ].make_cpu()

        init_scratch_auto[Self, target="cpu"](t)

        t.action_scale = action_scale
        t.learning_starts = learning_starts

        t.sample_blk.setup(learning_starts)
        return t^

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
        # rsample lives on actor_blk.inner (was on trainer.actor_loss before).
        self.actor_blk.inner.rsample.forward["cpu", 1](ao1_t, output=alp1_t)
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
        self.sample_blk.add(obs, action, reward, next_obs, done)

    def end_episode(mut self):
        self.tracker.end_episode()

    def train_step(mut self, step_idx: Int) raises -> Bool:
        """The pipeline. ZERO bind, ZERO wire, ZERO PortPack — each block
        is invoked with the refs it needs at the call site."""
        self.state.step_idx = step_idx
        self.state.did_step = True
        self.state.alpha = fexp(self.alpha_opt.value)

        self.sample_blk.step(self.state)
        if not self.state.did_step:
            return False

        self.target_y_blk.step["cpu"](
            self.state, self.actor,
            self.pair1.target_net, self.pair2.target_net,
        )
        self.twin_critic_blk.step["cpu"](
            self.state,
            self.pair1.online, self.critic1_opt,
            self.pair2.online, self.critic2_opt,
        )
        self.actor_blk.step["cpu"](
            self.state, self.actor, self.actor_opt,
            self.pair1.online, self.pair2.online,
        )
        self.alpha_blk.step(self.state, self.alpha_opt)
        self.polyak_blk.step["cpu"](
            self.state, self.pair1, self.pair2,
        )

        self._actor_L_accum  += self.state.actor_loss
        self._critic_L_accum += self.state.critic_loss
        self._alpha_accum    += fexp(self.alpha_opt.value)
        self._update_count   += 1
        return True

    def mean_return(self) -> Scalar[DT]:
        return self.tracker.mean_return()

    def ep_count(self) -> Int:
        return self.tracker.ep_count
