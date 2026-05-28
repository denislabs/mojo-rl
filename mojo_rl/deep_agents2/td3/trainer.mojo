"""TD3Trainer — J.1.g-redesign-v2 Step 3 — TD3 via ref-based blocks.

CPU only. Pipeline (4 blocks):
  Sample → TD3TargetY → TwinCritic [reused from SAC] → TD3DelayedActorPolyak

TD3DelayedActorPolyakStep bundles actor update + 3 polyaks, gated by an
internal counter (no state pollution).

Conforms to `OffPolicyAgentGpu` so it's drivable through the
Tier-3 `run_offpolicy_train_batched` (CPU env path only). GPU record
stubs raise — unreachable on the CPU env branch.
"""

from std.random import random_float64
from std.gpu.host import DeviceContext, DeviceBuffer
from layout import TileTensor, row_major

from mojo_rl.core.logger import Logger, NoOpLogger
from mojo_rl.nn2.constants import DT
from mojo_rl.nn2.core import Module
from mojo_rl.nn2.core.log_bundle import log_bundle
from mojo_rl.nn2.core.metric import LogScalar
from ..core.online_target_pair import OnlineTargetPair
from mojo_rl.nn2.core.scratch_walkers import init_scratch_auto
from ..data.n_step_replay import GPUNStepBuffer
from mojo_rl.nn2.initializer import Xavier
from mojo_rl.nn2.optimizer.adam import Adam
from mojo_rl.nn2.random.box_muller import box_muller_normal
from ..training.action_sampling_block import ActionSamplingBlock
from ..training.driver_offpolicy import OffPolicyAgentGpu
from ..training.episode_tracker import EpisodeTracker
from ..training.trainer_block import TrainerState
from ..training.blocks import UniformSampleCpuStep, TwinCriticStep
from .blocks.target_y_step import TD3TargetYStep
from .blocks.delayed_actor_polyak_step import TD3DelayedActorPolyakStep
from .metrics import TD3Metrics


struct TD3Trainer[
    ACTOR: Module,
    CRITIC: Module,
    OBS_DIM: Int,
    ACT_DIM: Int,
    BATCH: Int,
    REPLAY_CAPACITY: Int,
](OffPolicyAgentGpu):
    comptime AGENT_OBS_DIM: Int = Self.OBS_DIM
    comptime AGENT_ACT_DIM: Int = Self.ACT_DIM
    # TD3 is CPU-only; the OffPolicyAgentGpu GPU stubs raise.
    comptime AGENT_TRAIN_TARGET: StaticString = "cpu"

    var actor_pair: OnlineTargetPair[Self.ACTOR]
    var pair1: OnlineTargetPair[Self.CRITIC]
    var pair2: OnlineTargetPair[Self.CRITIC]
    var actor_opt: Adam
    var critic1_opt: Adam
    var critic2_opt: Adam

    var sample_blk: UniformSampleCpuStep[
        Self.OBS_DIM,
        Self.ACT_DIM,
        Self.BATCH,
        Self.REPLAY_CAPACITY,
    ]
    var target_y_blk: TD3TargetYStep[
        Self.OBS_DIM,
        Self.ACT_DIM,
        Self.BATCH,
        Self.ACTOR,
        Self.CRITIC,
    ]
    var twin_critic_blk: TwinCriticStep[
        Self.OBS_DIM,
        Self.ACT_DIM,
        Self.BATCH,
        Self.CRITIC,
    ]
    var actor_polyak_blk: TD3DelayedActorPolyakStep[
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
    var exploration_noise: Scalar[DT]
    var learning_starts: Int

    var _actor_L_accum: Scalar[DT]
    var _critic_L_accum: Scalar[DT]
    var _actor_updates: Int
    var _critic_updates: Int

    def __init__(out self):
        self.actor_pair = OnlineTargetPair[Self.ACTOR]()
        self.pair1 = OnlineTargetPair[Self.CRITIC]()
        self.pair2 = OnlineTargetPair[Self.CRITIC]()
        self.actor_opt = Adam()
        self.critic1_opt = Adam()
        self.critic2_opt = Adam()

        self.sample_blk = UniformSampleCpuStep[
            Self.OBS_DIM,
            Self.ACT_DIM,
            Self.BATCH,
            Self.REPLAY_CAPACITY,
        ]()
        self.target_y_blk = TD3TargetYStep[
            Self.OBS_DIM,
            Self.ACT_DIM,
            Self.BATCH,
            Self.ACTOR,
            Self.CRITIC,
        ]()
        self.twin_critic_blk = TwinCriticStep[
            Self.OBS_DIM,
            Self.ACT_DIM,
            Self.BATCH,
            Self.CRITIC,
        ]()
        self.actor_polyak_blk = TD3DelayedActorPolyakStep[
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
        self.exploration_noise = Scalar[DT](0.1)
        self.learning_starts = 1_000
        self._actor_L_accum = Scalar[DT](0.0)
        self._critic_L_accum = Scalar[DT](0.0)
        self._actor_updates = 0
        self._critic_updates = 0

    @staticmethod
    def make[
        target: StaticString
    ](
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
        t.actor_opt = Adam.make[target="cpu", M=Self.ACTOR](t.actor_pair.online)
        t.actor_opt.lr = actor_lr
        t.actor_opt.max_grad_norm = max_grad_norm
        t.critic1_opt = Adam.make[target="cpu", M=Self.CRITIC](t.pair1.online)
        t.critic1_opt.lr = critic_lr
        t.critic1_opt.max_grad_norm = max_grad_norm
        t.critic2_opt = Adam.make[target="cpu", M=Self.CRITIC](t.pair2.online)
        t.critic2_opt.lr = critic_lr
        t.critic2_opt.max_grad_norm = max_grad_norm

        t.target_y_blk = TD3TargetYStep[
            Self.OBS_DIM,
            Self.ACT_DIM,
            Self.BATCH,
            Self.ACTOR,
            Self.CRITIC,
        ].make["cpu"](
            action_scale=action_scale,
            gamma=gamma,
            noise_std=target_policy_noise,
            noise_clip=target_noise_clip,
        )
        t.twin_critic_blk = TwinCriticStep[
            Self.OBS_DIM,
            Self.ACT_DIM,
            Self.BATCH,
            Self.CRITIC,
        ].make["cpu"]()
        t.actor_polyak_blk = TD3DelayedActorPolyakStep[
            Self.OBS_DIM,
            Self.ACT_DIM,
            Self.BATCH,
            Self.ACTOR,
            Self.CRITIC,
        ].make["cpu"](policy_delay=policy_delay, tau=tau)
        t.policy_head = ActionSamplingBlock[
            Self.ACTOR,
            Self.OBS_DIM,
            Self.ACT_DIM,
            Self.ACT_DIM,
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
        t.exploration_noise = exploration_noise
        t.learning_starts = learning_starts

        t.sample_blk.setup(learning_starts)
        return t^

    # ─── Direct-callable (host-list) surface ─────────────────────────
    # Used by smoke tests that call the trainer directly without a
    # driver, and by the off-policy driver via the OffPolicyAgent trait.

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
            noise_scale=self.exploration_noise,
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
            self.pair1.target_net,
            self.pair2.target_net,
        )
        self.twin_critic_blk.step["cpu"](
            self.state,
            self.pair1.online,
            self.critic1_opt,
            self.pair2.online,
            self.critic2_opt,
        )
        self._critic_L_accum += self.state.critic_loss
        self._critic_updates += 1

        # TD3 actor + 3-pair polyak (gated by internal counter). Block
        # accesses actor via actor_pair.online + critic1 via pair1.online
        # internally — avoids Mojo aliasing rejection of passing pair +
        # pair.online simultaneously.
        self.actor_polyak_blk.step["cpu"](
            self.state,
            self.actor_opt,
            self.actor_pair,
            self.pair1,
            self.pair2,
        )
        # Block resets _counter to 0 when it fires; reads state.actor_loss
        # only when the block actually ran.
        if self.actor_polyak_blk._counter == 0:
            self._actor_L_accum += self.state.actor_loss
            self._actor_updates += 1
        return True

    def mean_return(self) -> Scalar[DT]:
        return self.tracker.mean_return()

    def ep_count(self) -> Int:
        return self.tracker.ep_count

    # ─── OffPolicyAgentGpu surface (Tier-3 driver) ────────────
    #
    # TD3 is CPU-only — the GPU record stubs raise. The Tier-3 driver
    # comptime-elides those branches when env_target == "cpu", so the
    # stubs are never invoked from a correctly-built driver. Pattern
    # mirrors MBPOTrainer / DDPGTrainer.

    def select_action_batched[
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

        # Actor output: N_ENVS × ACT.
        var obs_t = TileTensor(obs_ptr, row_major[N_ENVS, OBS]())
        var ao_t = TileTensor(ao_scratch_ptr, row_major[N_ENVS, ACT]())
        self.actor_pair.online.forward["cpu", N_ENVS](obs_t, output=ao_t)

        # Gaussian noise into alp_scratch_ptr (≥ N_ENVS*ACT capacity).
        box_muller_normal(alp_scratch_ptr, N_ENVS * ACT)
        var sigma = self.exploration_noise * self.action_scale
        for i in range(N_ENVS * ACT):
            var a = ao_scratch_ptr[i] + alp_scratch_ptr[i] * sigma
            if a > self.action_scale:
                a = self.action_scale
            elif a < -self.action_scale:
                a = -self.action_scale
            action_ptr[i] = a

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
            "TD3Trainer is CPU-only; record_batch_gpu unreachable"
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
            "TD3Trainer is CPU-only; record_batch_gpu_nstep unreachable"
            " via the Tier-3 cpu env path"
        )

    # ─── Logging surface (parity with SACTrainer) ────────────────────────

    def flush_train_log(
        mut self,
    ) -> Tuple[Scalar[DT], Scalar[DT], Int, Int]:
        """Return (mean_actor_loss, mean_critic_loss, n_actor_updates,
        n_critic_updates) since last flush. TD3 has separate counters
        because the actor is updated on a `policy_delay` cadence.
        Resets accumulators."""
        var na = self._actor_updates if self._actor_updates > 0 else 1
        var nc = self._critic_updates if self._critic_updates > 0 else 1
        var inv_a = Scalar[DT](1.0) / Scalar[DT](na)
        var inv_c = Scalar[DT](1.0) / Scalar[DT](nc)
        var out = (
            self._actor_L_accum * inv_a,
            self._critic_L_accum * inv_c,
            self._actor_updates,
            self._critic_updates,
        )
        self._actor_L_accum = Scalar[DT](0.0)
        self._critic_L_accum = Scalar[DT](0.0)
        self._actor_updates = 0
        self._critic_updates = 0
        return out

    def flush_metrics[
        L: Logger = NoOpLogger
    ](
        mut self,
        logger: Optional[UnsafePointer[L, MutAnyOrigin]] = None,
        step: Int = 0,
    ) raises -> TD3Metrics:
        """Drain accumulators into a TD3Metrics bundle. If a logger
        pointer is wired, also emit one log_scalar per metric field.
        Resets accumulators on every call."""
        var na = self._actor_updates if self._actor_updates > 0 else 1
        var nc = self._critic_updates if self._critic_updates > 0 else 1
        var inv_a = Scalar[DT](1.0) / Scalar[DT](na)
        var inv_c = Scalar[DT](1.0) / Scalar[DT](nc)
        var bundle = TD3Metrics(
            actor_loss=LogScalar[DT](self._actor_L_accum * inv_a),
            critic_loss=LogScalar[DT](self._critic_L_accum * inv_c),
            n_actor_updates=LogScalar[DT](Scalar[DT](self._actor_updates)),
            n_critic_updates=LogScalar[DT](Scalar[DT](self._critic_updates)),
        )
        self._actor_L_accum = Scalar[DT](0.0)
        self._critic_L_accum = Scalar[DT](0.0)
        self._actor_updates = 0
        self._critic_updates = 0
        if Bool(logger):
            log_bundle(logger.value()[], bundle, step)
        return bundle^

    def flush_timer_log(mut self) -> String:
        """No timer instrumentation yet (CPU-only trainer). Returns a
        placeholder for API parity with SACTrainer/DQNTrainer."""
        return String("TD3Trainer: no timer instrumentation")
