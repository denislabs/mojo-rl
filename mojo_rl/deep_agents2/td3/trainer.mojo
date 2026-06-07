"""TD3Trainer — unified TD3 trainer: CPU/GPU × uniform/PER replay.

Phase 4.2 migration to the SAC-style signature
`TD3Trainer[train_target, SAMPLE, ACTOR, CRITIC]` (dims derived from
SAMPLE). Replaces the prior CPU-only
`TD3Trainer[ACTOR, CRITIC, OBS, ACT, BATCH, REPLAY]`.

Pipeline (4 blocks):
  Sample → TD3TargetY [twin-critic min + target-policy smoothing] →
  TwinCritic [shared with SAC] → TD3DelayedActorPolyak

`TD3DelayedActorPolyakStep` bundles actor update (DPG on critic1, reusing
`DDPGActorLoss`) + all 3 polyaks (actor + both critics), gated by an
internal `policy_delay` counter. `policy_head` (`ActionSamplingBlock`) is
the env-interaction head; it already carries a GPU path, so the host-list
selects work for both targets.

CPU is bit-identical to the prior TD3Trainer (validated by the TD3 smoke +
metrics tests). GPU mirrors DDPG/SAC: twin-critic loss + DPG actor loss
accumulated on-device (no per-step D2H), drained at flush; target-policy
smoothing noise sampled on-device (Philox). CUDA-graph capture deferred to
Phase 4.4 (trait defaults, never reached with USE_TRAIN_CUDA_GRAPH=False).

Conforms to `OffPolicyAgentGpu`.
"""

from std.random import random_float64
from std.time import perf_counter_ns
from std.gpu import block_dim, block_idx, thread_idx
from std.gpu.host import DeviceContext, DeviceBuffer
from std.random.philox import Random as PhiloxRandom
from layout import Layout, LayoutTensor, TileTensor, row_major

from mojo_rl.core.logger import Logger, NoOpLogger
from mojo_rl.nn2.constants import DT, TPB
from mojo_rl.nn2.core import Module
from mojo_rl.nn2.core.amp import AMPPolicy, NoAMP, Bf16Compute
from mojo_rl.nn2.core.checkpoint import (
    save_state_v2_body, load_state_v2_body,
    save_state_v2_body_gpu, load_state_v2_body_gpu,
)
from mojo_rl.nn2.core.log_bundle import log_bundle
from mojo_rl.nn2.core.map_params import hard_copy_params
from mojo_rl.nn2.core.metric import LogScalar
from ..core.checkpoint_helpers import (
    save_optimizer_v2_body, load_optimizer_v2_body,
    save_optimizer_v2_body_gpu, load_optimizer_v2_body_gpu,
    save_counter_v2_body, load_counter_v2_body,
    split_lines_v2, read_file_v2, expect_v2_header,
)
from ..core.online_target_pair import OnlineTargetPair
from mojo_rl.nn2.core.scratch import Scratch
from mojo_rl.nn2.core.scratch_walkers import init_scratch_auto
from ..data.n_step_replay import GPUNStepBuffer
from mojo_rl.nn2.initializer import Xavier
from mojo_rl.nn2.optimizer.adam import Adam
from mojo_rl.nn2.random.box_muller import box_muller_normal, box_muller_normal_gpu
from mojo_rl.nn2.training.timer import Timer
from ..training.action_sampling_block import ActionSamplingBlock
from ..training.driver_offpolicy import OffPolicyAgentGpu
from ..training.episode_tracker import EpisodeTracker
from ..training.device_mean_accum import DeviceMeanAccum
from ..training.trainer_block import TrainerState
from ..training.blocks import SampleBlock, TwinCriticStep
from .blocks.target_y_step import TD3TargetYStep
from .blocks.delayed_actor_polyak_step import TD3DelayedActorPolyakStep
from .metrics import TD3Metrics


# ──────────────────────────────────────────────────────────────────────
# Batched GPU action kernels (deterministic actor + Gaussian exploration;
# per-trainer copies, same convention as SAC/DDPG).
# ──────────────────────────────────────────────────────────────────────


def _td3_warmup_uniform_kernel[
    N_ENVS: Int, ACT: Int
](
    action_dest: LayoutTensor[DT, Layout.row_major(N_ENVS, ACT), MutAnyOrigin],
    action_scale: Scalar[DT],
    seed: UInt64,
    offset_base: UInt64,
):
    """Per-lane Philox uniform → [N_ENVS, ACT] of Uniform(-scale, +scale)."""
    var i = Int(block_dim.x * block_idx.x + thread_idx.x)
    var total = N_ENVS * ACT
    if i >= total:
        return
    var philox = PhiloxRandom(seed=seed + UInt64(i), offset=offset_base)
    var u = Float32(philox.step_uniform()[0])
    var s = Scalar[DT](2.0) * Scalar[DT](u) - Scalar[DT](1.0)
    var env = i // ACT
    var j = i % ACT
    action_dest[env, j] = s * action_scale


def _td3_noise_clamp_kernel[
    N_ENVS: Int, ACT: Int
](
    ao: LayoutTensor[DT, Layout.row_major(N_ENVS, ACT), MutAnyOrigin],
    noise: LayoutTensor[DT, Layout.row_major(N_ENVS, ACT), MutAnyOrigin],
    action_out: LayoutTensor[DT, Layout.row_major(N_ENVS, ACT), MutAnyOrigin],
    sigma: Scalar[DT],
    action_scale: Scalar[DT],
):
    """`action_out = clamp(ao + noise·sigma, ±scale)` per lane."""
    var i = Int(block_dim.x * block_idx.x + thread_idx.x)
    var total = N_ENVS * ACT
    if i >= total:
        return
    var env = i // ACT
    var j = i % ACT
    var a = ao[env, j] + noise[env, j] * sigma
    if a > action_scale:
        a = action_scale
    elif a < -action_scale:
        a = -action_scale
    action_out[env, j] = a


struct TD3Trainer[
    train_target: StaticString,
    SAMPLE: SampleBlock,
    ACTOR: Module,
    CRITIC: Module,
](OffPolicyAgentGpu):
    """Dimensions (OBS / ACT / BATCH) derived from SAMPLE (mirrors SAC)."""

    comptime OBS_DIM: Int = Self.SAMPLE.OBS
    comptime ACT_DIM: Int = Self.SAMPLE.ACT
    comptime BATCH: Int = Self.SAMPLE.BATCH

    comptime AGENT_OBS_DIM: Int = Self.OBS_DIM
    comptime AGENT_ACT_DIM: Int = Self.ACT_DIM
    comptime AGENT_TRAIN_TARGET: StaticString = Self.train_target

    # Timer section indices — order matches `add_section` calls in `make`.
    comptime _T_SAMPLE = 0
    comptime _T_TARGET_Y = 1
    comptime _T_CRITIC = 2
    comptime _T_ACTOR_POLYAK = 3
    comptime _T_DIAG = 4

    var actor_pair: OnlineTargetPair[Self.ACTOR]
    var pair1: OnlineTargetPair[Self.CRITIC]
    var pair2: OnlineTargetPair[Self.CRITIC]
    var actor_opt: Adam
    var critic1_opt: Adam
    var critic2_opt: Adam

    var sample_blk: Self.SAMPLE
    var target_y_blk: TD3TargetYStep[
        Self.OBS_DIM, Self.ACT_DIM, Self.BATCH, Self.ACTOR, Self.CRITIC,
    ]
    var twin_critic_blk: TwinCriticStep[
        Self.OBS_DIM, Self.ACT_DIM, Self.BATCH, Self.CRITIC,
    ]
    var actor_polyak_blk: TD3DelayedActorPolyakStep[
        Self.OBS_DIM, Self.ACT_DIM, Self.BATCH, Self.ACTOR, Self.CRITIC,
    ]

    var policy_head: ActionSamplingBlock[
        Self.ACTOR, Self.OBS_DIM, Self.ACT_DIM, Self.ACT_DIM
    ]

    var state: TrainerState[Self.OBS_DIM, Self.ACT_DIM, Self.BATCH]
    var tracker: EpisodeTracker
    var ctx: Optional[DeviceContext]

    var action_scale: Scalar[DT]
    var exploration_noise: Scalar[DT]
    var learning_starts: Int
    var _use_bf16: Bool
    # Philox state for batched warmup + exploration noise (gpu path only).
    var _warmup_rng_seed: UInt64
    var _warmup_rng_offset: UInt64
    var _noise_rng_seed: UInt64
    var _noise_rng_offset: UInt64

    var _actor_L_accum: Scalar[DT]
    var _critic_L_accum: Scalar[DT]
    var _actor_updates: Int
    var _critic_updates: Int
    # Never reset by `flush_*` — emitted as `train_steps`.
    var _total_train_steps: Int

    # Per-batch diagnostic accumulators (CPU-only diag walk; GPU leaves 0).
    # Accumulated on the critic cadence → averaged by `_critic_updates`.
    var _q_accum: Scalar[DT]
    var _target_accum: Scalar[DT]
    var _reward_accum: Scalar[DT]
    var _done_accum: Scalar[DT]
    # GPU device-resident mirrors (CPU keeps the host scalars above).
    var _q_mean_dev: DeviceMeanAccum
    var _target_mean_dev: DeviceMeanAccum
    var _reward_mean_dev: DeviceMeanAccum
    var _done_mean_dev: DeviceMeanAccum

    var timer: Timer

    def __init__(out self):
        self.actor_pair = OnlineTargetPair[Self.ACTOR]()
        self.pair1 = OnlineTargetPair[Self.CRITIC]()
        self.pair2 = OnlineTargetPair[Self.CRITIC]()
        self.actor_opt = Adam()
        self.critic1_opt = Adam()
        self.critic2_opt = Adam()
        self.sample_blk = Self.SAMPLE()
        self.target_y_blk = TD3TargetYStep[
            Self.OBS_DIM, Self.ACT_DIM, Self.BATCH, Self.ACTOR, Self.CRITIC,
        ]()
        self.twin_critic_blk = TwinCriticStep[
            Self.OBS_DIM, Self.ACT_DIM, Self.BATCH, Self.CRITIC,
        ]()
        self.actor_polyak_blk = TD3DelayedActorPolyakStep[
            Self.OBS_DIM, Self.ACT_DIM, Self.BATCH, Self.ACTOR, Self.CRITIC,
        ]()
        self.policy_head = ActionSamplingBlock[
            Self.ACTOR, Self.OBS_DIM, Self.ACT_DIM, Self.ACT_DIM,
        ]()
        self.state = TrainerState[
            Self.OBS_DIM, Self.ACT_DIM, Self.BATCH,
        ]()
        self.tracker = EpisodeTracker(
            window=List[Scalar[DT]](),
            window_size=0,
            idx=0,
            current_return=Scalar[DT](0.0),
            ep_count=0,
        )
        self.ctx = None
        self.action_scale = Scalar[DT](1.0)
        self.exploration_noise = Scalar[DT](0.1)
        self.learning_starts = 1_000
        self._use_bf16 = False
        self._warmup_rng_seed = UInt64(0xC0FFEE_C0DE)
        self._warmup_rng_offset = UInt64(0)
        self._noise_rng_seed = UInt64(0xD15EA5E_D00D)
        self._noise_rng_offset = UInt64(0)
        self._actor_L_accum = Scalar[DT](0.0)
        self._critic_L_accum = Scalar[DT](0.0)
        self._actor_updates = 0
        self._critic_updates = 0
        self._total_train_steps = 0
        self._q_accum = Scalar[DT](0.0)
        self._target_accum = Scalar[DT](0.0)
        self._reward_accum = Scalar[DT](0.0)
        self._done_accum = Scalar[DT](0.0)
        self._q_mean_dev = DeviceMeanAccum()
        self._target_mean_dev = DeviceMeanAccum()
        self._reward_mean_dev = DeviceMeanAccum()
        self._done_mean_dev = DeviceMeanAccum()
        self.timer = Timer.new()

    @staticmethod
    def make(
        ctx: Optional[DeviceContext] = None,
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
        use_bf16: Bool = False,
    ) raises -> Self:
        """Unified factory. `ctx` is required for `train_target='gpu'`.
        `use_bf16=True` (GPU only) runs the train step under `Bf16Compute`."""
        comptime assert (
            Self.train_target == "cpu" or Self.train_target == "gpu"
        ), "TD3Trainer: target must be 'cpu' or 'gpu'"
        comptime if Self.train_target == "gpu":
            if not ctx:
                raise Error("TD3Trainer.make[target='gpu']: ctx required")

        var t = Self()
        t.ctx = ctx

        t.actor_pair = OnlineTargetPair[Self.ACTOR].make[
            target=Self.train_target, INIT=Xavier
        ](ctx=ctx)
        t.pair1 = OnlineTargetPair[Self.CRITIC].make[
            target=Self.train_target, INIT=Xavier
        ](ctx=ctx)
        t.pair2 = OnlineTargetPair[Self.CRITIC].make[
            target=Self.train_target, INIT=Xavier
        ](ctx=ctx)
        t.actor_opt = Adam.make[target=Self.train_target, M=Self.ACTOR](
            t.actor_pair.online, ctx=ctx,
        )
        t.actor_opt.lr = actor_lr
        t.actor_opt.max_grad_norm = max_grad_norm
        t.critic1_opt = Adam.make[target=Self.train_target, M=Self.CRITIC](
            t.pair1.online, ctx=ctx,
        )
        t.critic1_opt.lr = critic_lr
        t.critic1_opt.max_grad_norm = max_grad_norm
        t.critic2_opt = Adam.make[target=Self.train_target, M=Self.CRITIC](
            t.pair2.online, ctx=ctx,
        )
        t.critic2_opt.lr = critic_lr
        t.critic2_opt.max_grad_norm = max_grad_norm

        t.target_y_blk = TD3TargetYStep[
            Self.OBS_DIM, Self.ACT_DIM, Self.BATCH, Self.ACTOR, Self.CRITIC,
        ].make[Self.train_target](
            action_scale=action_scale, gamma=gamma,
            noise_std=target_policy_noise, noise_clip=target_noise_clip,
            ctx=ctx,
        )
        t.twin_critic_blk = TwinCriticStep[
            Self.OBS_DIM, Self.ACT_DIM, Self.BATCH, Self.CRITIC,
        ].make[Self.train_target](ctx=ctx)
        t.actor_polyak_blk = TD3DelayedActorPolyakStep[
            Self.OBS_DIM, Self.ACT_DIM, Self.BATCH, Self.ACTOR, Self.CRITIC,
        ].make[Self.train_target](
            policy_delay=policy_delay, tau=tau, ctx=ctx,
        )
        t.policy_head = ActionSamplingBlock[
            Self.ACTOR, Self.OBS_DIM, Self.ACT_DIM, Self.ACT_DIM,
        ].make[Self.train_target](ctx=ctx)

        t.tracker = EpisodeTracker.new(
            window_size=window_size, initial_fill=initial_episode_fill
        )
        t.state = TrainerState[
            Self.OBS_DIM, Self.ACT_DIM, Self.BATCH,
        ].make[Self.train_target](ctx=ctx)

        init_scratch_auto[Self, target=Self.train_target](t, ctx)

        comptime if Self.train_target == "gpu":
            # Device-resident mean accumulators for the GPU diag path.
            t._q_mean_dev = DeviceMeanAccum.make["gpu"](ctx=ctx)
            t._target_mean_dev = DeviceMeanAccum.make["gpu"](ctx=ctx)
            t._reward_mean_dev = DeviceMeanAccum.make["gpu"](ctx=ctx)
            t._done_mean_dev = DeviceMeanAccum.make["gpu"](ctx=ctx)

        t.action_scale = action_scale
        t.exploration_noise = exploration_noise
        t.learning_starts = learning_starts
        t._use_bf16 = use_bf16

        t.sample_blk.setup(learning_starts, ctx=ctx)

        # Timer sections — index order MUST match the `_T_*` constants.
        t.timer.add_section("sample")
        t.timer.add_section("target_y")
        t.timer.add_section("critic")
        t.timer.add_section("actor_polyak")
        t.timer.add_section("diag")
        return t^

    # ─── Direct-callable (host-list) surface ─────────────────────────

    def select_action(
        mut self,
        ref obs: List[Scalar[DT]],
        mut action_out: List[Scalar[DT]],
        step_idx: Int,
    ) raises:
        self.policy_head.select_deterministic_with_noise[Self.train_target](
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
        self.policy_head.select_deterministic_with_noise[Self.train_target](
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
        self.sample_blk.add(obs, action, reward, next_obs, done, ctx=self.ctx)

    # `end_episode` / `mean_return` / `ep_count` / `add_complete_return`
    # are OffPolicyAgent trait defaults (S6) over this single accessor.
    def _tracker_ptr(self) -> UnsafePointer[EpisodeTracker, MutAnyOrigin]:
        return rebind[UnsafePointer[EpisodeTracker, MutAnyOrigin]](
            UnsafePointer(to=self.tracker)
        )

    # ─── train_step ───────────────────────────────────────────────────

    def train_step(mut self, step_idx: Int) raises -> Bool:
        comptime if Self.train_target == "cpu":
            return self._train_step_impl[NoAMP](step_idx)
        else:
            if self._use_bf16:
                return self._train_step_impl[Bf16Compute](step_idx)
            return self._train_step_impl[NoAMP](step_idx)

    def _train_step_impl[
        POLICY: AMPPolicy = NoAMP,
    ](mut self, step_idx: Int) raises -> Bool:
        self.state.step_idx = step_idx
        self.state.did_step = True
        comptime if Self.train_target == "gpu":
            self.state.ctx = self.ctx

        var t_sample = perf_counter_ns()
        self.sample_blk.step(self.state)
        if not self.state.did_step:
            return False
        self.timer.accumulate(Self._T_SAMPLE, t_sample)

        var t_ty = perf_counter_ns()
        self.target_y_blk.step[Self.train_target, POLICY](
            self.state,
            self.actor_pair.target_net,
            self.pair1.target_net,
            self.pair2.target_net,
        )
        self.timer.accumulate(Self._T_TARGET_Y, t_ty)

        var t_crit = perf_counter_ns()
        # GPU: accumulate both critics' loss on-device (no per-step D2H).
        self.twin_critic_blk.step[
            Self.train_target, POLICY, ACCUMULATE = Self.train_target == "gpu"
        ](
            self.state,
            self.pair1.online,
            self.critic1_opt,
            self.pair2.online,
            self.critic2_opt,
        )
        self.timer.accumulate(Self._T_CRITIC, t_crit)
        self._critic_L_accum += self.state.critic_loss
        self._critic_updates += 1

        # Per-batch diagnostics — `mean_q` reads twin_critic c1's Q1(s, a);
        # target/reward/done read the minibatch scratches. Fires every step
        # (critic cadence). CPU sums the host scratches; GPU folds the same
        # `[BATCH]` device buffers into device-resident running means.
        var t_diag = perf_counter_ns()
        comptime if Self.train_target == "cpu":
            var inv_b: Scalar[DT] = Scalar[DT](1.0) / Scalar[DT](Self.BATCH)
            var q_p = self.twin_critic_blk.inner.c1._mb_q.target_ptr["cpu"]()
            var y_p = self.state.mb_y.target_ptr["cpu"]()
            var r_p = self.state.mb_r.target_ptr["cpu"]()
            var d_p = self.state.mb_d.target_ptr["cpu"]()
            var sum_q: Scalar[DT] = 0.0
            var sum_y: Scalar[DT] = 0.0
            var sum_r: Scalar[DT] = 0.0
            var sum_d: Scalar[DT] = 0.0
            for i in range(Self.BATCH):
                sum_q += q_p[i]
                sum_y += y_p[i]
                sum_r += r_p[i]
                sum_d += d_p[i]
            self._q_accum += sum_q * inv_b
            self._target_accum += sum_y * inv_b
            self._reward_accum += sum_r * inv_b
            self._done_accum += sum_d * inv_b
        else:
            var q_ptr = self.twin_critic_blk.inner.c1._mb_q.target_ptr["gpu"]()
            var y_ptr = self.state.mb_y.target_ptr["gpu"]()
            var r_ptr = self.state.mb_r.target_ptr["gpu"]()
            var d_ptr = self.state.mb_d.target_ptr["gpu"]()
            self._q_mean_dev.accumulate_gpu[Self.BATCH](q_ptr)
            self._target_mean_dev.accumulate_gpu[Self.BATCH](y_ptr)
            self._reward_mean_dev.accumulate_gpu[Self.BATCH](r_ptr)
            self._done_mean_dev.accumulate_gpu[Self.BATCH](d_ptr)
        self.timer.accumulate(Self._T_DIAG, t_diag)

        # TD3 actor + 3-pair polyak (gated by internal counter). Block reads
        # actor via actor_pair.online + critic1 via pair1.online internally.
        var t_act = perf_counter_ns()
        self.actor_polyak_blk.step[Self.train_target, POLICY](
            self.state,
            self.actor_opt,
            self.actor_pair,
            self.pair1,
            self.pair2,
        )
        self.timer.accumulate(Self._T_ACTOR_POLYAK, t_act)
        # Block resets _counter to 0 when it fires; the actor loss is valid
        # (CPU) / accumulated on-device (GPU) only then.
        if self.actor_polyak_blk._counter == 0:
            self._actor_L_accum += self.state.actor_loss
            self._actor_updates += 1
        self._total_train_steps += 1
        return True

    # ─── OffPolicyAgentGpu surface (drivers) ──────────────────────────

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
            comptime if Self.train_target == "cpu":
                for i in range(N_ENVS * ACT):
                    var u = Scalar[DT](2.0 * random_float64() - 1.0)
                    action_ptr[i] = u * self.action_scale
            else:
                var action_lt = LayoutTensor[
                    DT, Layout.row_major(N_ENVS, ACT), MutAnyOrigin,
                ](action_ptr)
                comptime total = N_ENVS * ACT
                comptime n_blocks = (total + TPB - 1) // TPB
                comptime warmup_kernel = _td3_warmup_uniform_kernel[N_ENVS, ACT]
                var ctx = self.ctx.value()
                ctx.enqueue_function[warmup_kernel](
                    action_lt,
                    self.action_scale,
                    self._warmup_rng_seed,
                    self._warmup_rng_offset,
                    grid_dim=n_blocks,
                    block_dim=TPB,
                )
                self._warmup_rng_offset += UInt64(N_ENVS * ACT * 2)
            return

        var sigma = self.exploration_noise * self.action_scale
        comptime if Self.train_target == "cpu":
            var obs_t = TileTensor(obs_ptr, row_major[N_ENVS, OBS]())
            var ao_t = TileTensor(ao_scratch_ptr, row_major[N_ENVS, ACT]())
            self.actor_pair.online.forward["cpu", N_ENVS](obs_t, output=ao_t)
            box_muller_normal(alp_scratch_ptr, N_ENVS * ACT)
            for i in range(N_ENVS * ACT):
                var a = ao_scratch_ptr[i] + alp_scratch_ptr[i] * sigma
                if a > self.action_scale:
                    a = self.action_scale
                elif a < -self.action_scale:
                    a = -self.action_scale
                action_ptr[i] = a
        else:
            var ctx = self.ctx.value()
            var obs_t = TileTensor(obs_ptr, row_major[N_ENVS, OBS]())
            var ao_t = TileTensor(ao_scratch_ptr, row_major[N_ENVS, ACT]())
            self.actor_pair.online.forward["gpu", N_ENVS](obs_t, output=ao_t)
            comptime total = N_ENVS * ACT
            box_muller_normal_gpu[total](
                ctx, alp_scratch_ptr,
                self._noise_rng_seed, self._noise_rng_offset,
            )
            self._noise_rng_offset += UInt64(((total + 1) // 2) * 2)
            var ao_lt = LayoutTensor[
                DT, Layout.row_major(N_ENVS, ACT), MutAnyOrigin,
            ](ao_scratch_ptr)
            var noise_lt = LayoutTensor[
                DT, Layout.row_major(N_ENVS, ACT), MutAnyOrigin,
            ](alp_scratch_ptr)
            var action_lt = LayoutTensor[
                DT, Layout.row_major(N_ENVS, ACT), MutAnyOrigin,
            ](action_ptr)
            comptime n_blocks = (total + TPB - 1) // TPB
            comptime nc_kernel = _td3_noise_clamp_kernel[N_ENVS, ACT]
            ctx.enqueue_function[nc_kernel](
                ao_lt, noise_lt, action_lt, sigma, self.action_scale,
                grid_dim=n_blocks, block_dim=TPB,
            )

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
                ctx=self.ctx,
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
        """Delegates to the sample block's add_batch_gpu (GPU blocks)."""
        self.sample_blk.add_batch_gpu[N_ENVS](
            ctx, prev_obs_dev, action_dev, reward_dev, obs_dev, done_dev,
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
        nstep_buf.process(
            ctx, prev_obs_dev, action_dev, reward_dev, obs_dev, done_dev,
        )
        self.sample_blk.store_via_block_gpu[N_ENVS, NS](ctx, nstep_buf)

    # ─── Logging surface (parity with SACTrainer) ────────────────────────

    def flush_train_log(
        mut self,
    ) -> Tuple[Scalar[DT], Scalar[DT], Int, Int]:
        """Return (mean_actor_loss, mean_critic_loss, n_actor_updates,
        n_critic_updates) since last flush. Resets accumulators. CPU
        host-scalar path; for GPU use `flush_metrics`."""
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
        self._q_accum = Scalar[DT](0.0)
        self._target_accum = Scalar[DT](0.0)
        self._reward_accum = Scalar[DT](0.0)
        self._done_accum = Scalar[DT](0.0)
        return out

    def total_train_steps(self) -> Int:
        """Cumulative training updates since trainer was made. Not reset
        by `flush_*`."""
        return self._total_train_steps

    def flush_metrics[
        L: Logger = NoOpLogger
    ](
        mut self,
        logger: Optional[UnsafePointer[L, MutAnyOrigin]] = None,
        step: Int = 0,
    ) raises -> TD3Metrics:
        """Drain accumulators into a TD3Metrics bundle. On GPU the actor +
        critic losses are read from the on-device accumulators (no per-step
        D2H); per-batch diag means are device-resident on GPU (Q / target /
        reward / done reductions folded in each critic update)."""
        var na = self._actor_updates if self._actor_updates > 0 else 1
        var nc = self._critic_updates if self._critic_updates > 0 else 1
        var inv_a = Scalar[DT](1.0) / Scalar[DT](na)
        var inv_c = Scalar[DT](1.0) / Scalar[DT](nc)
        var actor_mean: Scalar[DT]
        var critic_mean: Scalar[DT]
        var q_mean: Scalar[DT]
        var target_mean: Scalar[DT]
        var reward_mean: Scalar[DT]
        var done_mean: Scalar[DT]
        comptime if Self.train_target == "gpu":
            actor_mean = self.actor_polyak_blk.read_loss_accum()
            var cl1 = self.twin_critic_blk.inner.c1.mse_loss.read_accum["gpu"]()
            var cl2 = self.twin_critic_blk.inner.c2.mse_loss.read_accum["gpu"]()
            critic_mean = cl1 + cl2
            q_mean = self._q_mean_dev.read["gpu"]()
            target_mean = self._target_mean_dev.read["gpu"]()
            reward_mean = self._reward_mean_dev.read["gpu"]()
            done_mean = self._done_mean_dev.read["gpu"]()
        else:
            actor_mean = self._actor_L_accum * inv_a
            critic_mean = self._critic_L_accum * inv_c
            q_mean = self._q_accum * inv_c
            target_mean = self._target_accum * inv_c
            reward_mean = self._reward_accum * inv_c
            done_mean = self._done_accum * inv_c
        var bundle = TD3Metrics(
            actor_loss=LogScalar[DT](actor_mean),
            critic_loss=LogScalar[DT](critic_mean),
            mean_q=LogScalar[DT](q_mean),
            mean_target=LogScalar[DT](target_mean),
            mean_reward=LogScalar[DT](reward_mean),
            mean_done=LogScalar[DT](done_mean),
            train_steps=LogScalar[DT](Scalar[DT](self._total_train_steps)),
            n_actor_updates=LogScalar[DT](Scalar[DT](self._actor_updates)),
            n_critic_updates=LogScalar[DT](Scalar[DT](self._critic_updates)),
        )
        self._actor_L_accum = Scalar[DT](0.0)
        self._critic_L_accum = Scalar[DT](0.0)
        self._actor_updates = 0
        self._critic_updates = 0
        self._q_accum = Scalar[DT](0.0)
        self._target_accum = Scalar[DT](0.0)
        self._reward_accum = Scalar[DT](0.0)
        self._done_accum = Scalar[DT](0.0)
        comptime if Self.train_target == "gpu":
            self.twin_critic_blk.inner.c1.mse_loss.reset_accum["gpu"]()
            self.twin_critic_blk.inner.c2.mse_loss.reset_accum["gpu"]()
            self.actor_polyak_blk.reset_loss_accum()
            self._q_mean_dev.reset["gpu"]()
            self._target_mean_dev.reset["gpu"]()
            self._reward_mean_dev.reset["gpu"]()
            self._done_mean_dev.reset["gpu"]()
        if Bool(logger):
            log_bundle(logger.value()[], bundle, step)
        return bundle^

    # ─── Trait-uniform cadence hooks (consumed by the driver) ─────────

    def flush_metrics_through_logger[L: Logger](
        mut self,
        logger: Optional[UnsafePointer[L, MutAnyOrigin]],
        step: Int,
    ) raises:
        _ = self.flush_metrics[L](logger, step)

    def save_state(mut self, path: String) raises:
        """One-file v2 checkpoint of every TD3 module + optimizer.
        Sections: `actor.*`, `critic1.*`, `critic2.*`, `actor_opt.*`,
        `critic1_opt.*`, `critic2_opt.*`."""
        var body = String("")
        comptime if Self.train_target == "gpu":
            var c = self.ctx.value()
            save_state_v2_body_gpu(self.actor_pair.online, body, "actor", c)
            save_state_v2_body_gpu(self.pair1.online, body, "critic1", c)
            save_state_v2_body_gpu(self.pair2.online, body, "critic2", c)
            save_optimizer_v2_body_gpu(self.actor_opt, body, "actor_opt")
            save_optimizer_v2_body_gpu(self.critic1_opt, body, "critic1_opt")
            save_optimizer_v2_body_gpu(self.critic2_opt, body, "critic2_opt")
        else:
            save_state_v2_body(self.actor_pair.online, body, "actor")
            save_state_v2_body(self.pair1.online, body, "critic1")
            save_state_v2_body(self.pair2.online, body, "critic2")
            save_optimizer_v2_body(self.actor_opt, body, "actor_opt")
            save_optimizer_v2_body(self.critic1_opt, body, "critic1_opt")
            save_optimizer_v2_body(self.critic2_opt, body, "critic2_opt")
        save_counter_v2_body(self._total_train_steps, body, "_total_train_steps")
        var content = String("nn2-ckpt v2\n") + body
        with open(path, "w") as f:
            f.write(content)

    def load_state(mut self, path: String) raises:
        """Inverse of `save_state`. Target nets are hard-copied from their
        online twins after the online params are restored."""
        var content = read_file_v2(path)
        var lines = split_lines_v2(content)
        expect_v2_header(lines)
        var idx: Int = 1
        comptime if Self.train_target == "gpu":
            var c = self.ctx.value()
            load_state_v2_body_gpu(self.actor_pair.online, lines, idx, "actor", c)
            load_state_v2_body_gpu(self.pair1.online, lines, idx, "critic1", c)
            load_state_v2_body_gpu(self.pair2.online, lines, idx, "critic2", c)
            load_optimizer_v2_body_gpu(self.actor_opt, lines, idx, "actor_opt")
            load_optimizer_v2_body_gpu(self.critic1_opt, lines, idx, "critic1_opt")
            load_optimizer_v2_body_gpu(self.critic2_opt, lines, idx, "critic2_opt")
            hard_copy_params["gpu", M=Self.ACTOR](
                self.actor_pair.online, self.actor_pair.target_net, self.ctx,
            )
            hard_copy_params["gpu", M=Self.CRITIC](
                self.pair1.online, self.pair1.target_net, self.ctx,
            )
            hard_copy_params["gpu", M=Self.CRITIC](
                self.pair2.online, self.pair2.target_net, self.ctx,
            )
        else:
            load_state_v2_body(self.actor_pair.online, lines, idx, "actor")
            load_state_v2_body(self.pair1.online, lines, idx, "critic1")
            load_state_v2_body(self.pair2.online, lines, idx, "critic2")
            load_optimizer_v2_body(self.actor_opt, lines, idx, "actor_opt")
            load_optimizer_v2_body(self.critic1_opt, lines, idx, "critic1_opt")
            load_optimizer_v2_body(self.critic2_opt, lines, idx, "critic2_opt")
            hard_copy_params["cpu", M=Self.ACTOR](
                self.actor_pair.online, self.actor_pair.target_net, None,
            )
            hard_copy_params["cpu", M=Self.CRITIC](
                self.pair1.online, self.pair1.target_net, None,
            )
            hard_copy_params["cpu", M=Self.CRITIC](
                self.pair2.online, self.pair2.target_net, None,
            )
        load_counter_v2_body(
            self._total_train_steps, lines, idx, "_total_train_steps"
        )

    def flush_timer_log(mut self) -> String:
        """Per-section wall-time report (sample / target_y / critic /
        actor_polyak / diag) and reset the accumulators."""
        var report = self.timer.format_report()
        self.timer.reset()
        return report
