"""DDPGTrainer — J.1.g-redesign-v2 Step 3 — DDPG via ref-based blocks.

CPU only. Pipeline (5 blocks):
  Sample → DDPGTargetY → SingleCritic → DDPGActor → DDPGPolyak

`policy_head` is kept as a plain helper field for select_action (not a
pipeline block — it's used in env-interaction, not the train_step graph).

Conforms to `OffPolicyAgentUnifiedGpu` so it's drivable through the
Tier-3 `run_offpolicy_train_batched` (CPU env path only). The GPU record
stubs raise — unreachable on the CPU env branch which the Tier-3 driver
comptime-elides for `env_target == "cpu"`.
"""

from std.random import random_float64
from std.gpu.host import DeviceContext, DeviceBuffer
from layout import TileTensor, row_major

from ..constants import DT
from ..core import Module
from ..core.online_target_pair import OnlineTargetPair
from ..core.scratch_walkers import init_scratch_auto
from ..data.n_step_replay import GPUNStepBuffer
from ..initializer import Xavier
from ..optimizer.adam import Adam
from ..random.box_muller import box_muller_normal
from .action_sampling_block import ActionSamplingBlock
from .driver_unified import OffPolicyAgentUnifiedGpu
from .episode_tracker import EpisodeTracker
from .trainer_block import TrainerState
from .blocks import (
    UniformSampleCpuStep,
    DDPGTargetYStep,
    SingleCriticStep,
    DDPGActorStep,
    DDPGPolyakStep,
)


struct DDPGTrainer[
    ACTOR: Module,
    CRITIC: Module,
    OBS_DIM: Int,
    ACT_DIM: Int,
    BATCH: Int,
    REPLAY_CAPACITY: Int,
](OffPolicyAgentUnifiedGpu):
    comptime AGENT_OBS_DIM: Int = Self.OBS_DIM
    comptime AGENT_ACT_DIM: Int = Self.ACT_DIM
    # DDPG is CPU-only; the OffPolicyAgentUnifiedGpu GPU stubs raise.
    comptime AGENT_TRAIN_TARGET: StaticString = "cpu"

    var actor_pair: OnlineTargetPair[Self.ACTOR]
    var critic_pair: OnlineTargetPair[Self.CRITIC]
    var actor_opt: Adam
    var critic_opt: Adam

    var sample_blk: UniformSampleCpuStep[
        Self.OBS_DIM,
        Self.ACT_DIM,
        Self.BATCH,
        Self.REPLAY_CAPACITY,
    ]
    var target_y_blk: DDPGTargetYStep[
        Self.OBS_DIM,
        Self.ACT_DIM,
        Self.BATCH,
        Self.ACTOR,
        Self.CRITIC,
    ]
    var critic_blk: SingleCriticStep[
        Self.OBS_DIM,
        Self.ACT_DIM,
        Self.BATCH,
        Self.CRITIC,
    ]
    var actor_blk: DDPGActorStep[
        Self.OBS_DIM,
        Self.ACT_DIM,
        Self.BATCH,
        Self.ACTOR,
        Self.CRITIC,
    ]
    var polyak_blk: DDPGPolyakStep[
        Self.OBS_DIM,
        Self.ACT_DIM,
        Self.BATCH,
        Self.ACTOR,
        Self.CRITIC,
    ]

    var policy_head: ActionSamplingBlock[
        Self.ACTOR, Self.OBS_DIM, Self.ACT_DIM, Self.ACT_DIM
    ]

    var state: TrainerState[Self.OBS_DIM, Self.ACT_DIM, Self.BATCH]
    var tracker: EpisodeTracker

    var action_scale: Scalar[DT]
    var noise_scale: Scalar[DT]
    var learning_starts: Int

    var _actor_L_accum: Scalar[DT]
    var _critic_L_accum: Scalar[DT]
    var _update_count: Int

    def __init__(out self):
        self.actor_pair = OnlineTargetPair[Self.ACTOR]()
        self.critic_pair = OnlineTargetPair[Self.CRITIC]()
        self.actor_opt = Adam()
        self.critic_opt = Adam()
        self.sample_blk = UniformSampleCpuStep[
            Self.OBS_DIM,
            Self.ACT_DIM,
            Self.BATCH,
            Self.REPLAY_CAPACITY,
        ]()
        self.target_y_blk = DDPGTargetYStep[
            Self.OBS_DIM,
            Self.ACT_DIM,
            Self.BATCH,
            Self.ACTOR,
            Self.CRITIC,
        ]()
        self.critic_blk = SingleCriticStep[
            Self.OBS_DIM,
            Self.ACT_DIM,
            Self.BATCH,
            Self.CRITIC,
        ]()
        self.actor_blk = DDPGActorStep[
            Self.OBS_DIM,
            Self.ACT_DIM,
            Self.BATCH,
            Self.ACTOR,
            Self.CRITIC,
        ]()
        self.polyak_blk = DDPGPolyakStep[
            Self.OBS_DIM,
            Self.ACT_DIM,
            Self.BATCH,
            Self.ACTOR,
            Self.CRITIC,
        ]()
        self.policy_head = ActionSamplingBlock[
            Self.ACTOR,
            Self.OBS_DIM,
            Self.ACT_DIM,
            Self.ACT_DIM,
        ]()
        self.state = TrainerState[
            Self.OBS_DIM,
            Self.ACT_DIM,
            Self.BATCH,
        ]()
        self.tracker = EpisodeTracker(
            window=List[Scalar[DT]](),
            window_size=0,
            idx=0,
            current_return=Scalar[DT](0.0),
            ep_count=0,
        )
        self.action_scale = Scalar[DT](1.0)
        self.noise_scale = Scalar[DT](0.1)
        self.learning_starts = 1_000
        self._actor_L_accum = Scalar[DT](0.0)
        self._critic_L_accum = Scalar[DT](0.0)
        self._update_count = 0

    @staticmethod
    def make[
        target: StaticString
    ](
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
        t.actor_opt = Adam.make[target="cpu", M=Self.ACTOR](t.actor_pair.online)
        t.actor_opt.lr = actor_lr
        t.actor_opt.max_grad_norm = max_grad_norm
        t.critic_opt = Adam.make[target="cpu", M=Self.CRITIC](
            t.critic_pair.online
        )
        t.critic_opt.lr = critic_lr
        t.critic_opt.max_grad_norm = max_grad_norm

        t.target_y_blk = DDPGTargetYStep[
            Self.OBS_DIM,
            Self.ACT_DIM,
            Self.BATCH,
            Self.ACTOR,
            Self.CRITIC,
        ].make["cpu"](action_scale=action_scale, gamma=gamma)
        t.critic_blk = SingleCriticStep[
            Self.OBS_DIM,
            Self.ACT_DIM,
            Self.BATCH,
            Self.CRITIC,
        ].make["cpu"]()
        t.actor_blk = DDPGActorStep[
            Self.OBS_DIM,
            Self.ACT_DIM,
            Self.BATCH,
            Self.ACTOR,
            Self.CRITIC,
        ].make["cpu"]()
        t.polyak_blk = DDPGPolyakStep[
            Self.OBS_DIM,
            Self.ACT_DIM,
            Self.BATCH,
            Self.ACTOR,
            Self.CRITIC,
        ].make(tau=tau)
        t.policy_head = ActionSamplingBlock[
            Self.ACTOR, Self.OBS_DIM, Self.ACT_DIM, Self.ACT_DIM
        ].make["cpu"]()

        t.tracker = EpisodeTracker.new(
            window_size=window_size, initial_fill=initial_episode_fill
        )
        t.state = TrainerState[
            Self.OBS_DIM,
            Self.ACT_DIM,
            Self.BATCH,
        ].make["cpu"]()

        init_scratch_auto[Self, target="cpu"](t)

        t.action_scale = action_scale
        t.noise_scale = noise_scale
        t.learning_starts = learning_starts

        t.sample_blk.setup(learning_starts)
        return t^

    # ─── Direct-callable (host-list) surface ─────────────────────────
    # Used by smoke tests that call the trainer directly without a
    # driver. The Tier-3 driver uses the `*_unified` methods below.

    def select_action(
        mut self,
        ref obs: List[Scalar[DT]],
        mut action_out: List[Scalar[DT]],
        step_idx: Int,
    ) raises:
        self.policy_head.select_deterministic_with_noise["cpu"](
            self.actor_pair.online,
            obs,
            action_out,
            step_idx=step_idx,
            learning_starts=self.learning_starts,
            action_scale=self.action_scale,
            noise_scale=self.noise_scale,
        )

    def select_greedy_action(
        mut self,
        ref obs: List[Scalar[DT]],
        mut action_out: List[Scalar[DT]],
    ) raises:
        self.policy_head.select_deterministic_with_noise["cpu"](
            self.actor_pair.online,
            obs,
            action_out,
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
        self.sample_blk.add(obs, action, reward, next_obs, done)

    def end_episode(mut self):
        self.tracker.end_episode()

    def train_step(mut self, step_idx: Int) raises -> Bool:
        self.state.step_idx = step_idx
        self.state.did_step = True

        self.sample_blk.step(self.state)
        if not self.state.did_step:
            return False

        self.target_y_blk.step["cpu"](
            self.state,
            self.actor_pair.target_net,
            self.critic_pair.target_net,
        )
        self.critic_blk.step["cpu"](
            self.state,
            self.critic_pair.online,
            self.critic_opt,
        )
        self.actor_blk.step["cpu"](
            self.state,
            self.actor_pair.online,
            self.actor_opt,
            self.critic_pair.online,
        )
        self.polyak_blk.step["cpu"](
            self.state,
            self.actor_pair,
            self.critic_pair,
        )

        self._actor_L_accum += self.state.actor_loss
        self._critic_L_accum += self.state.critic_loss
        self._update_count += 1
        return True

    def mean_return(self) -> Scalar[DT]:
        return self.tracker.mean_return()

    def ep_count(self) -> Int:
        return self.tracker.ep_count

    # ─── OffPolicyAgentUnifiedGpu surface (Tier-3 driver) ────────────
    #
    # DDPG is CPU-only — the GPU record stubs raise. The Tier-3 driver
    # comptime-elides those branches when env_target == "cpu", so the
    # stubs are never invoked from a correctly-built driver. Pattern
    # mirrors MBPOTrainer's unified surface.

    def select_action_unified[
        N_ENVS: Int
    ](
        mut self,
        obs_ptr: UnsafePointer[Scalar[DT], MutAnyOrigin],
        action_ptr: UnsafePointer[Scalar[DT], MutAnyOrigin],
        ao_scratch_ptr: UnsafePointer[Scalar[DT], MutAnyOrigin],
        alp_scratch_ptr: UnsafePointer[Scalar[DT], MutAnyOrigin],
        step_idx: Int,
    ) raises:
        comptime assert N_ENVS > 0, "N_ENVS must be > 0"
        comptime OBS = Self.OBS_DIM
        comptime ACT = Self.ACT_DIM

        if step_idx < self.learning_starts:
            for i in range(N_ENVS * ACT):
                var u = Scalar[DT](2.0 * random_float64() - 1.0)
                action_ptr[i] = u * self.action_scale
            return

        # Actor output: N_ENVS × ACT (uses first N_ENVS*ACT scalars of
        # the driver's N_ENVS*2*ACT ao scratch — surplus is harmless).
        var obs_t = TileTensor(obs_ptr, row_major[N_ENVS, OBS]())
        var ao_t = TileTensor(ao_scratch_ptr, row_major[N_ENVS, ACT]())
        self.actor_pair.online.forward["cpu", N_ENVS](obs_t, output=ao_t)

        # Gaussian noise into alp_scratch_ptr (capacity N_ENVS*(ACT+1)
        # ≥ N_ENVS*ACT). Same RNG source as the legacy path's
        # box_muller_normal(self._noise, ACT_DIM).
        box_muller_normal(alp_scratch_ptr, N_ENVS * ACT)
        var sigma = self.noise_scale * self.action_scale
        for i in range(N_ENVS * ACT):
            var a = ao_scratch_ptr[i] + alp_scratch_ptr[i] * sigma
            if a > self.action_scale:
                a = self.action_scale
            elif a < -self.action_scale:
                a = -self.action_scale
            action_ptr[i] = a

    def train_step_unified(mut self, step_idx: Int) raises -> Bool:
        return self.train_step(step_idx)

    def select_greedy_action_unified(
        mut self,
        ref obs: List[Scalar[DT]],
        mut action_out: List[Scalar[DT]],
    ) raises:
        self.select_greedy_action(obs, action_out)

    def add_complete_return(mut self, ret: Scalar[DT]):
        self.tracker.add_complete_return(ret)

    def record_batch_cpu[
        N_ENVS: Int
    ](
        mut self,
        prev_obs_ptr: UnsafePointer[Scalar[DT], MutAnyOrigin],
        action_ptr: UnsafePointer[Scalar[DT], MutAnyOrigin],
        reward_ptr: UnsafePointer[Scalar[DT], MutAnyOrigin],
        next_obs_ptr: UnsafePointer[Scalar[DT], MutAnyOrigin],
        done_ptr: UnsafePointer[Scalar[DT], MutAnyOrigin],
    ) raises:
        comptime OBS = Self.OBS_DIM
        comptime ACT = Self.ACT_DIM
        var obs_lane = List[Scalar[DT]](length=OBS, fill=Scalar[DT](0.0))
        var act_lane = List[Scalar[DT]](length=ACT, fill=Scalar[DT](0.0))
        var nxt_lane = List[Scalar[DT]](length=OBS, fill=Scalar[DT](0.0))
        for env_idx in range(N_ENVS):
            for d in range(OBS):
                obs_lane[d] = prev_obs_ptr[env_idx * OBS + d]
                nxt_lane[d] = next_obs_ptr[env_idx * OBS + d]
            for j in range(ACT):
                act_lane[j] = action_ptr[env_idx * ACT + j]
            self.sample_blk.add(
                obs_lane,
                act_lane,
                reward_ptr[env_idx],
                nxt_lane,
                done_ptr[env_idx],
            )

    def record_batch_gpu[
        N_ENVS: Int
    ](
        mut self,
        ctx: DeviceContext,
        prev_obs_dev: DeviceBuffer[DT],
        action_dev: DeviceBuffer[DT],
        reward_dev: DeviceBuffer[DT],
        obs_dev: DeviceBuffer[DT],
        done_dev: DeviceBuffer[DT],
    ) raises:
        raise Error(
            "DDPGTrainer is CPU-only; record_batch_gpu unreachable"
            " via the Tier-3 cpu env path"
        )

    def record_batch_gpu_nstep[
        N_ENVS: Int, NS: Int
    ](
        mut self,
        ctx: DeviceContext,
        mut nstep_buf: GPUNStepBuffer[
            NS, Self.AGENT_OBS_DIM, Self.AGENT_ACT_DIM, N_ENVS,
        ],
        prev_obs_dev: DeviceBuffer[DT],
        action_dev: DeviceBuffer[DT],
        reward_dev: DeviceBuffer[DT],
        obs_dev: DeviceBuffer[DT],
        done_dev: DeviceBuffer[DT],
    ) raises:
        raise Error(
            "DDPGTrainer is CPU-only; record_batch_gpu_nstep unreachable"
            " via the Tier-3 cpu env path"
        )
