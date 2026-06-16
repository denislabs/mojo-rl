"""C51Trainer — distributional DQN trainer (Bellemare et al. 2017).

CPU-only initial port. Pipeline body mirrors `dqn/trainer.mojo`:
sample → target-Y (categorical projection) → q-update (cross-entropy)
→ polyak. Differences from DQN:

  - `Q_NET.OUT_DIM == NA · N_ATOMS` (per-atom logits instead of Q-values).
  - Target is a distribution `m [B, N_ATOMS]` (NOT a scalar) — the
    trainer owns its own `_mb_m` scratch instead of using `state.mb_y`.
  - Action selection picks `argmax_a Σ_k softmax(logits[b, a])_k · z_k`
    (expected Q from the distribution), not a plain `argmax` over Q.

Conforms to `OffPolicyDiscreteAgent` so it slots into the existing
`run_offpolicy_discrete_train` driver. PER + N-step plumbing inherited
from the SAMPLE block + target-Y γ^n.
"""

from std.math import exp as fexp, log as flog
from std.random import random_float64
from std.gpu import global_idx
from std.gpu.host import DeviceContext, DeviceBuffer
from layout import TileTensor, row_major

from mojo_rl.nn.constants import TPB
from mojo_rl.nn.core.module import mptr

from mojo_rl.core.logger import Logger, NoOpLogger
from mojo_rl.nn.constants import DT
from mojo_rl.nn.core import Module
from mojo_rl.nn.core.amp import AMPPolicy, NoAMP
from mojo_rl.nn.core.checkpoint import (
    save_state_v2_body, load_state_v2_body,
    save_state_v2_body_gpu, load_state_v2_body_gpu,
)
from mojo_rl.nn.core.log_bundle import log_bundle
from mojo_rl.nn.core.map_params import hard_copy_params
from mojo_rl.nn.core.save_scalar import SaveScalar
from mojo_rl.nn.core.metric import LogScalar
from mojo_rl.nn.core.scratch import Scratch
from mojo_rl.nn.core.scratch_walkers import init_scratch_auto
from mojo_rl.nn.initializer import Xavier
from mojo_rl.nn.optimizer.adam import Adam
from mojo_rl.nn.training.timer import Timer
from ..core.checkpoint_helpers import (
    save_optimizer_v2_body, load_optimizer_v2_body,
    save_optimizer_v2_body_gpu, load_optimizer_v2_body_gpu,
    save_counter_v2_body, load_counter_v2_body,
    split_lines_v2, read_file_v2, expect_v2_header,
)
from ..core.online_target_pair import OnlineTargetPair
from ..training.episode_tracker import EpisodeTracker
from ..training.device_mean_accum import DeviceMeanAccum
from ..training.trainer_block import TrainerState
from ..training.driver_offpolicy_discrete import (
    OffPolicyDiscreteAgent,
    OffPolicyDiscreteAgentGpu,
)
from ..training.blocks import SampleBlock, SinglePolyakStep
from ..data.n_step_replay import GPUNStepBuffer
from .target_y_block import C51TargetYBlock
from .blocks.q_update_step import C51QUpdateStep
from .metrics import C51Metrics


# ──────────────────────────────────────────────────────────────────────────
# GPU per-sample distributional diag kernel. One thread per batch row b:
# softmax the taken-action logit row, then write the expected Q
# (`eq = Σ p_k z_k`), the categorical entropy (`ent = −Σ p_k log p_k`), and
# the target-distribution expected value (`tq = Σ m_k z_k`) into three
# `[BATCH]` buffers. The trainer reduces those via `DeviceMeanAccum`. Mirrors
# the CPU walk in `_train_step_impl` exactly (same max-shift softmax + 1e-12
# entropy floor).
# ──────────────────────────────────────────────────────────────────────────
def _c51_diag_kernel[BATCH: Int, NK: Int](
    logits: UnsafePointer[Scalar[DT], MutAnyOrigin],
    m: UnsafePointer[Scalar[DT], MutAnyOrigin],
    z: UnsafePointer[Scalar[DT], MutAnyOrigin],
    eq_out: UnsafePointer[Scalar[DT], MutAnyOrigin],
    ent_out: UnsafePointer[Scalar[DT], MutAnyOrigin],
    tq_out: UnsafePointer[Scalar[DT], MutAnyOrigin],
):
    var b = Int(global_idx.x)
    if b >= BATCH:
        return
    var base = b * NK
    var maxl = logits[base]
    for k in range(1, NK):
        if logits[base + k] > maxl:
            maxl = logits[base + k]
    var sum_exp: Scalar[DT] = 0.0
    for k in range(NK):
        sum_exp += fexp(logits[base + k] - maxl)
    var eq: Scalar[DT] = 0.0
    var ent: Scalar[DT] = 0.0
    var tq: Scalar[DT] = 0.0
    for k in range(NK):
        var p = fexp(logits[base + k] - maxl) / sum_exp
        eq += p * z[k]
        if p > Scalar[DT](1e-12):
            ent -= p * flog(p)
        tq += m[base + k] * z[k]
    eq_out[b] = eq
    ent_out[b] = ent
    tq_out[b] = tq


struct C51Trainer[
    train_target: StaticString,
    SAMPLE: SampleBlock,
    Q_NET: Module,
    N_ATOMS: Int = 51,
    NUM_ACTIONS: Int = 2,
    DOUBLE: Bool = False,
](OffPolicyDiscreteAgentGpu):
    """Q_NET.OUT_DIM must equal NUM_ACTIONS · N_ATOMS (per-atom logits).
    Standard Rainbow defaults: N_ATOMS=51, V_min=-10, V_max=+10."""

    comptime OBS_DIM: Int = Self.SAMPLE.OBS
    comptime ACT_DIM: Int = Self.SAMPLE.ACT
    comptime BATCH: Int = Self.SAMPLE.BATCH

    comptime AGENT_TRAIN_TARGET: StaticString = Self.train_target
    comptime AGENT_OBS_DIM: Int = Self.OBS_DIM
    comptime AGENT_ACT_DIM: Int = Self.ACT_DIM
    comptime AGENT_NUM_ACTIONS: Int = Self.NUM_ACTIONS

    comptime _T_SAMPLE = 0
    comptime _T_TARGET_Y = 1
    comptime _T_CRITIC = 2
    comptime _T_POLYAK = 3
    comptime _T_DIAG = 4

    var pair: OnlineTargetPair[Self.Q_NET]
    var q_opt: Adam
    var sample_blk: Self.SAMPLE
    var target_y_blk: C51TargetYBlock[
        Self.Q_NET, Self.BATCH, Self.OBS_DIM, Self.NUM_ACTIONS,
        Self.N_ATOMS, Self.DOUBLE,
    ]
    var q_update_blk: C51QUpdateStep[
        Self.OBS_DIM, Self.ACT_DIM, Self.BATCH, Self.NUM_ACTIONS,
        Self.N_ATOMS, Self.Q_NET,
    ]
    var polyak_blk: SinglePolyakStep[
        Self.OBS_DIM, Self.ACT_DIM, Self.BATCH, Self.Q_NET,
    ]

    var state: TrainerState[Self.OBS_DIM, Self.ACT_DIM, Self.BATCH]

    # Target distribution scratch — replaces `state.mb_y`'s [B,1] role.
    var _mb_m: Scratch["mb_m", Self.BATCH * Self.N_ATOMS]

    # Action-selection scratch.
    var _ob1: Scratch["ob1", Self.OBS_DIM, True]
    var _q_logits: Scratch["q_logits", Self.NUM_ACTIONS * Self.N_ATOMS, True]
    var _q_batch: Scratch["q_batch", Self.NUM_ACTIONS * Self.N_ATOMS]

    var tracker: EpisodeTracker
    var ctx: Optional[DeviceContext]

    var epsilon: Scalar[DT]
    var epsilon_decay: Scalar[DT]
    var epsilon_min: Scalar[DT]
    var learning_starts: Int

    var _action_list: List[Scalar[DT]]

    # Lazily-sized scratch for the GPU batched action path (N_ENVS is a
    # method-comptime param, unknown at construction). Allocated once on the
    # first `select_action_batched[N_ENVS]` call with N_ENVS>1 and reused —
    # avoids the per-step `enqueue_create_buffer` that explodes disk on NVIDIA.
    var _batch_q_dev: Optional[DeviceBuffer[DT]]
    var _batch_q_host: List[Scalar[DT]]
    var _batch_act_host: List[Scalar[DT]]
    var _batch_n: Int

    var _loss_accum: Scalar[DT]
    # Per-batch distributional diagnostic accumulators (CPU-only diag walk;
    # see `_train_step_impl`). Drained + reset by `flush_metrics`.
    var _q_accum: Scalar[DT]
    var _target_accum: Scalar[DT]
    var _dist_entropy_accum: Scalar[DT]
    var _reward_accum: Scalar[DT]
    var _done_accum: Scalar[DT]
    # GPU device-resident mirrors (CPU keeps the host scalars above). The
    # distributional means are derived by `_c51_diag_kernel` into three
    # `[BATCH]` device buffers, then reduced via these accumulators.
    var _q_mean_dev: DeviceMeanAccum
    var _target_mean_dev: DeviceMeanAccum
    var _dist_entropy_mean_dev: DeviceMeanAccum
    var _reward_mean_dev: DeviceMeanAccum
    var _done_mean_dev: DeviceMeanAccum
    var _diag_eq_dev: Optional[DeviceBuffer[DT]]
    var _diag_ent_dev: Optional[DeviceBuffer[DT]]
    var _diag_tq_dev: Optional[DeviceBuffer[DT]]
    var _update_count: Int
    # Never reset by `flush_*` — emitted as `train_steps` so the
    # downstream monitor can plot cumulative updates over time.
    var _total_train_steps: Int
    var timer: Timer

    def __init__(out self):
        self.pair = OnlineTargetPair[Self.Q_NET]()
        self.q_opt = Adam()
        self.sample_blk = Self.SAMPLE()
        self.target_y_blk = C51TargetYBlock[
            Self.Q_NET, Self.BATCH, Self.OBS_DIM, Self.NUM_ACTIONS,
            Self.N_ATOMS, Self.DOUBLE,
        ]()
        self.q_update_blk = C51QUpdateStep[
            Self.OBS_DIM, Self.ACT_DIM, Self.BATCH, Self.NUM_ACTIONS,
            Self.N_ATOMS, Self.Q_NET,
        ]()
        self.polyak_blk = SinglePolyakStep[
            Self.OBS_DIM, Self.ACT_DIM, Self.BATCH, Self.Q_NET,
        ]()
        self.state = TrainerState[
            Self.OBS_DIM, Self.ACT_DIM, Self.BATCH,
        ]()
        self._mb_m = Scratch["mb_m", Self.BATCH * Self.N_ATOMS]()
        self._ob1 = Scratch["ob1", Self.OBS_DIM, True]()
        self._q_logits = Scratch[
            "q_logits", Self.NUM_ACTIONS * Self.N_ATOMS, True,
        ]()
        self._q_batch = Scratch[
            "q_batch", Self.NUM_ACTIONS * Self.N_ATOMS,
        ]()
        self.tracker = EpisodeTracker(
            window=List[Scalar[DT]](),
            window_size=0,
            idx=0,
            current_return=Scalar[DT](0.0),
            ep_count=0,
        )
        self.ctx = None
        self.epsilon = Scalar[DT](1.0)
        self.epsilon_decay = Scalar[DT](0.995)
        self.epsilon_min = Scalar[DT](0.01)
        self.learning_starts = 1_000
        self._action_list = List[Scalar[DT]](length=1, fill=Scalar[DT](0.0))
        self._batch_q_dev = None
        self._batch_q_host = List[Scalar[DT]]()
        self._batch_act_host = List[Scalar[DT]]()
        self._batch_n = 0
        self._loss_accum = Scalar[DT](0.0)
        self._q_accum = Scalar[DT](0.0)
        self._target_accum = Scalar[DT](0.0)
        self._dist_entropy_accum = Scalar[DT](0.0)
        self._reward_accum = Scalar[DT](0.0)
        self._done_accum = Scalar[DT](0.0)
        self._q_mean_dev = DeviceMeanAccum()
        self._target_mean_dev = DeviceMeanAccum()
        self._dist_entropy_mean_dev = DeviceMeanAccum()
        self._reward_mean_dev = DeviceMeanAccum()
        self._done_mean_dev = DeviceMeanAccum()
        self._diag_eq_dev = None
        self._diag_ent_dev = None
        self._diag_tq_dev = None
        self._update_count = 0
        self._total_train_steps = 0
        self.timer = Timer.new()

    @staticmethod
    def make(
        ctx: Optional[DeviceContext] = None,
        lr: Scalar[DT] = Scalar[DT](1e-4),
        gamma: Scalar[DT] = Scalar[DT](0.99),
        tau: Scalar[DT] = Scalar[DT](0.005),
        epsilon: Scalar[DT] = Scalar[DT](1.0),
        epsilon_decay: Scalar[DT] = Scalar[DT](0.995),
        epsilon_min: Scalar[DT] = Scalar[DT](0.05),
        learning_starts: Int = 1_000,
        target_update_freq: Int = 500,
        window_size: Int = 10,
        initial_episode_fill: Scalar[DT] = Scalar[DT](0.0),
        max_grad_norm: Scalar[DT] = Scalar[DT](0.0),
        per_alpha: Scalar[DT] = Scalar[DT](0.6),
        per_beta: Scalar[DT] = Scalar[DT](0.4),
        per_epsilon: Scalar[DT] = Scalar[DT](1e-6),
        nstep: Int = 1,
        v_min: Scalar[DT] = Scalar[DT](-10.0),
        v_max: Scalar[DT] = Scalar[DT](10.0),
    ) raises -> Self:
        comptime assert (
            Self.train_target == "cpu" or Self.train_target == "gpu"
        ), "C51Trainer: train_target must be 'cpu' or 'gpu'"
        comptime assert (
            Self.ACT_DIM == 1
        ), "C51Trainer: SAMPLE.ACT must be 1 (discrete action index)"
        comptime assert (
            Self.Q_NET.IN_DIMS[0] == Self.OBS_DIM
        ), "C51Trainer: Q_NET.IN_DIM must equal SAMPLE.OBS"
        comptime assert (
            Self.Q_NET.OUT_DIM == Self.NUM_ACTIONS * Self.N_ATOMS
        ), "C51Trainer: Q_NET.OUT_DIM must equal NUM_ACTIONS · N_ATOMS"

        var t = Self()
        t.ctx = ctx
        t.epsilon = epsilon
        t.epsilon_decay = epsilon_decay
        t.epsilon_min = epsilon_min
        t.learning_starts = learning_starts

        t.pair = OnlineTargetPair[Self.Q_NET].make[
            target=Self.train_target, INIT=Xavier,
        ](ctx=ctx)
        t.q_opt = Adam.make[target=Self.train_target, M=Self.Q_NET](
            t.pair.online, ctx=ctx,
        )
        t.q_opt.lr = lr
        t.q_opt.max_grad_norm = max_grad_norm

        t.target_y_blk = C51TargetYBlock[
            Self.Q_NET, Self.BATCH, Self.OBS_DIM, Self.NUM_ACTIONS,
            Self.N_ATOMS, Self.DOUBLE,
        ].make[Self.train_target](
            gamma=gamma, nstep=nstep,
            v_min=v_min, v_max=v_max, ctx=ctx,
        )

        t.q_update_blk = C51QUpdateStep[
            Self.OBS_DIM, Self.ACT_DIM, Self.BATCH, Self.NUM_ACTIONS,
            Self.N_ATOMS, Self.Q_NET,
        ].make[Self.train_target](ctx=ctx)

        t.polyak_blk = SinglePolyakStep[
            Self.OBS_DIM, Self.ACT_DIM, Self.BATCH, Self.Q_NET,
        ].make(tau=tau, update_every=target_update_freq)

        t.state = TrainerState[
            Self.OBS_DIM, Self.ACT_DIM, Self.BATCH,
        ].make[Self.train_target](ctx=ctx)

        t.tracker = EpisodeTracker.new(
            window_size=window_size,
            initial_fill=initial_episode_fill,
        )

        init_scratch_auto[Self, target=Self.train_target](t, ctx)

        comptime if Self.train_target == "gpu":
            # Device-resident mean accumulators + per-sample diag scratch
            # ([BATCH] eq/ent/tq buffers filled by `_c51_diag_kernel`).
            var c = ctx.value()
            t._q_mean_dev = DeviceMeanAccum.make["gpu"](ctx=ctx)
            t._target_mean_dev = DeviceMeanAccum.make["gpu"](ctx=ctx)
            t._dist_entropy_mean_dev = DeviceMeanAccum.make["gpu"](ctx=ctx)
            t._reward_mean_dev = DeviceMeanAccum.make["gpu"](ctx=ctx)
            t._done_mean_dev = DeviceMeanAccum.make["gpu"](ctx=ctx)
            t._diag_eq_dev = c.enqueue_create_buffer[DT](Self.BATCH)
            t._diag_ent_dev = c.enqueue_create_buffer[DT](Self.BATCH)
            t._diag_tq_dev = c.enqueue_create_buffer[DT](Self.BATCH)

        t.sample_blk.configure_per(
            alpha=per_alpha, beta=per_beta, epsilon=per_epsilon,
        )
        t.sample_blk.configure_gamma(gamma)
        t.sample_blk.setup(learning_starts, ctx=ctx)

        t.timer.add_section("sample")
        t.timer.add_section("target_y")
        t.timer.add_section("critic")
        t.timer.add_section("polyak")
        t.timer.add_section("diag")
        return t^

    # ─── Train step ──────────────────────────────────────────────────

    def _train_step_impl[POLICY: AMPPolicy = NoAMP](
        mut self, step_idx: Int,
    ) raises -> Bool:
        from std.time import perf_counter_ns

        self.state.step_idx = step_idx
        self.state.did_step = True

        var t_sample = perf_counter_ns()
        self.sample_blk.step(self.state)
        if not self.state.did_step:
            return False
        self.timer.accumulate(Self._T_SAMPLE, t_sample)

        # Shared device-kernel sequence (target_y → q_update → polyak → PER
        # tail → GPU diag) — the body the CUDA-graph capture path replays.
        self._train_post_sample_kernels[POLICY]()

        # Per-batch distributional diagnostics — CPU-only host walk (the GPU
        # counterpart is folded into `_train_post_sample_kernels`). For the
        # taken action: softmax its N_ATOMS logit row, take expected Q
        # (Σ p_k z_k) and entropy (−Σ p_k log p_k). `_mb_m` holds the projected
        # target distribution, so its expected value Σ m_k z_k is the target-Q.
        var t_diag = perf_counter_ns()
        comptime if Self.train_target == "cpu":
            comptime NK = Self.N_ATOMS
            var inv_b: Scalar[DT] = Scalar[DT](1.0) / Scalar[DT](Self.BATCH)
            var z_p = self.target_y_blk.z_ptr()
            var lg_p = self.q_update_blk.inner._logits_a.target_ptr["cpu"]()
            var m_p = self._mb_m.target_ptr["cpu"]()
            var r_p = self.state.mb_r.target_ptr["cpu"]()
            var d_p = self.state.mb_d.target_ptr["cpu"]()
            var sum_q: Scalar[DT] = 0.0
            var sum_tq: Scalar[DT] = 0.0
            var sum_ent: Scalar[DT] = 0.0
            var sum_r: Scalar[DT] = 0.0
            var sum_d: Scalar[DT] = 0.0
            for b in range(Self.BATCH):
                var base = b * NK
                var maxl = lg_p[base]
                for k in range(1, NK):
                    if lg_p[base + k] > maxl:
                        maxl = lg_p[base + k]
                var sum_exp: Scalar[DT] = 0.0
                for k in range(NK):
                    sum_exp += fexp(lg_p[base + k] - maxl)
                var eq: Scalar[DT] = 0.0
                var ent: Scalar[DT] = 0.0
                var tq: Scalar[DT] = 0.0
                for k in range(NK):
                    var p = fexp(lg_p[base + k] - maxl) / sum_exp
                    eq += p * z_p[k]
                    if p > Scalar[DT](1e-12):
                        ent -= p * flog(p)
                    tq += m_p[base + k] * z_p[k]
                sum_q += eq
                sum_tq += tq
                sum_ent += ent
                sum_r += r_p[b]
                sum_d += d_p[b]
            self._q_accum += sum_q * inv_b
            self._target_accum += sum_tq * inv_b
            self._dist_entropy_accum += sum_ent * inv_b
            self._reward_accum += sum_r * inv_b
            self._done_accum += sum_d * inv_b
        self.timer.accumulate(Self._T_DIAG, t_diag)

        self.note_train_update()
        return True

    # ─── Shared post-sample kernel sequence ───────────────────────────
    #
    # target_y → q_update (ACCUMULATE on GPU) → polyak → PER tail → GPU diag.
    # Called by BOTH `_train_step_impl` (non-captured) and
    # `train_device_kernels` (the CUDA-graph capture closure body), so the two
    # paths enqueue an identical kernel sequence — bit-identity by
    # construction. The `perf_counter_ns` timers are host-only: harmless during
    # capture (not kernels, so not recorded) and don't fire on replay.
    def _train_post_sample_kernels[
        POLICY: AMPPolicy = NoAMP,
    ](mut self) raises:
        from std.time import perf_counter_ns

        var t_ty = perf_counter_ns()
        var m_ptr = self._mb_m.target_ptr[Self.train_target]()
        self.target_y_blk.step[Self.train_target, POLICY](
            self.pair.target_net,
            self.pair.online,
            self.state.mb_sp.target_ptr[Self.train_target](),
            self.state.mb_r.target_ptr[Self.train_target](),
            self.state.mb_d.target_ptr[Self.train_target](),
            m_ptr,
        )
        self.timer.accumulate(Self._T_TARGET_Y, t_ty)

        # On GPU, accumulate the CE loss on-device (no per-step D2H,
        # CUDA-graph capturable); the host reads it at flush via read_accum.
        var t_crit = perf_counter_ns()
        self.q_update_blk.step[
            Self.train_target, POLICY, ACCUMULATE = Self.train_target == "gpu"
        ](
            self.state, self.pair.online, self.q_opt, m_ptr,
        )
        self.timer.accumulate(Self._T_CRITIC, t_crit)

        var t_poly = perf_counter_ns()
        self.polyak_blk.step[Self.train_target](self.state, self.pair)
        self.timer.accumulate(Self._T_POLYAK, t_poly)

        # PER tail (no-op for uniform blocks).
        self.sample_blk.update_priorities(self.state)

        # GPU per-batch distributional diag — device kernels + device-resident
        # running means (no D2H → capture-safe). The CPU host-walk counterpart
        # lives in `_train_step_impl`.
        comptime if Self.train_target == "gpu":
            var t_diag = perf_counter_ns()
            var ctx_v = self.ctx.value()
            var lg_ptr = self.q_update_blk.inner._logits_a.target_ptr["gpu"]()
            var m_ptr_g = self._mb_m.target_ptr["gpu"]()
            var z_ptr = self.target_y_blk._z.target_ptr["gpu"]()
            var eq_ptr = self._diag_eq_dev.value().unsafe_ptr()
            var ent_ptr = self._diag_ent_dev.value().unsafe_ptr()
            var tq_ptr = self._diag_tq_dev.value().unsafe_ptr()
            comptime n_blocks = (Self.BATCH + TPB - 1) // TPB
            comptime diag_k = _c51_diag_kernel[Self.BATCH, Self.N_ATOMS]
            ctx_v.enqueue_function[diag_k](
                lg_ptr, m_ptr_g, z_ptr, eq_ptr, ent_ptr, tq_ptr,
                grid_dim=n_blocks, block_dim=TPB,
            )
            self._q_mean_dev.accumulate_gpu[Self.BATCH](eq_ptr)
            self._target_mean_dev.accumulate_gpu[Self.BATCH](tq_ptr)
            self._dist_entropy_mean_dev.accumulate_gpu[Self.BATCH](ent_ptr)
            self._reward_mean_dev.accumulate_gpu[Self.BATCH](
                self.state.mb_r.target_ptr["gpu"]()
            )
            self._done_mean_dev.accumulate_gpu[Self.BATCH](
                self.state.mb_d.target_ptr["gpu"]()
            )
            self.timer.accumulate(Self._T_DIAG, t_diag)

    # ─── Host bookkeeping (counters + metric accumulator) ─────────────
    #
    # One logical update's host accounting. Called by `_train_step_impl` and —
    # on the capture path — by the driver once per replayed update, so the
    # counters stay correct whether the device work ran directly or via graph
    # replay. On GPU `state.critic_loss` is a 0 sentinel (loss read from the
    # device accumulator at flush), so the `+=` is a harmless `+= 0`.
    def note_train_update(mut self):
        self._loss_accum += self.state.critic_loss
        self._update_count += 1
        self._total_train_steps += 1

    # ─── CUDA-graph capture surface ───────────────────────────────────
    #
    # `train_device_kernels` is the pure device-kernel train step — sampling
    # (device RNG → fresh minibatch each replay) + the shared post-sample
    # sequence, with NO host gate, NO counters. It is the body of the capture
    # closure passed to `maybe_capture_replay`. GPU-only; the caller gates on
    # `learning_starts_count()` and advances host counters via
    # `note_train_update()`.
    def _train_device_kernels_impl[
        POLICY: AMPPolicy = NoAMP,
    ](mut self) raises:
        # Pin `state.step_idx = learning_starts` so the sample block's warmup
        # gate passes (this method has no step_idx of its own). The driver only
        # calls it once the buffer holds ≥ BATCH transitions.
        self.state.step_idx = self.learning_starts
        self.state.did_step = True
        self.sample_blk.step(self.state)
        self._train_post_sample_kernels[POLICY]()

    def train_device_kernels(mut self) raises:
        comptime assert Self.train_target == "gpu", (
            "train_device_kernels is GPU-only (CUDA-graph capture path)"
        )
        self._train_device_kernels_impl[NoAMP]()

    def learning_starts_count(self) -> Int:
        """Cumulative env-step threshold after which the replay buffer is
        warm enough to train — the driver gates the capture path on this."""
        return self.learning_starts

    def train_step(mut self, step_idx: Int) raises -> Bool:
        return self._train_step_impl[NoAMP](step_idx)

    def set_beta(mut self, beta: Scalar[DT]):
        self.sample_blk.set_beta(beta)

    # ─── Record ──────────────────────────────────────────────────────

    def record(
        mut self,
        ref obs: List[Scalar[DT]],
        action_idx: Int,
        reward: Scalar[DT],
        ref next_obs: List[Scalar[DT]],
        done: Scalar[DT],
    ) raises:
        self.tracker.add_reward(reward)
        self._action_list[0] = Scalar[DT](action_idx)
        self.sample_blk.add(
            obs, self._action_list, reward, next_obs, done, ctx=self.ctx,
        )

    def record_batch_cpu[
        N_ENVS: Int,
    ](
        mut self,
        prev_obs_ptr: UnsafePointer[Scalar[DT], MutAnyOrigin],
        action_ptr: UnsafePointer[Scalar[DT], MutAnyOrigin],
        reward_ptr: UnsafePointer[Scalar[DT], MutAnyOrigin],
        next_obs_ptr: UnsafePointer[Scalar[DT], MutAnyOrigin],
        done_ptr: UnsafePointer[Scalar[DT], MutAnyOrigin],
    ) raises:
        comptime assert (
            Self.train_target == "cpu"
        ), "record_batch_cpu: trainer must be cpu"
        comptime OBS = Self.OBS_DIM
        var obs_lane = List[Scalar[DT]](length=OBS, fill=Scalar[DT](0.0))
        var act_lane = List[Scalar[DT]](length=1, fill=Scalar[DT](0.0))
        var nxt_lane = List[Scalar[DT]](length=OBS, fill=Scalar[DT](0.0))
        for env_idx in range(N_ENVS):
            for d in range(OBS):
                obs_lane[d] = prev_obs_ptr[env_idx * OBS + d]
                nxt_lane[d] = next_obs_ptr[env_idx * OBS + d]
            act_lane[0] = action_ptr[env_idx]
            self.sample_blk.add(
                obs_lane, act_lane, reward_ptr[env_idx], nxt_lane,
                done_ptr[env_idx], ctx=self.ctx,
            )

    # ─── Batched device record (GPU-batched-env driver) ──────────────

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
        """1-step batched device record — forwards N_ENVS device
        transitions to the sample block's `add_batch_gpu`. The block
        (GPU uniform / PER) writes them directly into device replay; no
        host round trip. `done_dev` carries `terminated`."""
        self.sample_blk.add_batch_gpu[N_ENVS](
            ctx, prev_obs_dev, action_dev, reward_dev, obs_dev, done_dev,
        )

    def record_batch_gpu_nstep[
        N_ENVS: Int, NS: Int
    ](
        mut self,
        ctx: DeviceContext,
        mut nstep_buf: GPUNStepBuffer[
            NS, Self.AGENT_OBS_DIM, Self.AGENT_ACT_DIM, N_ENVS
        ],
        prev_obs_dev: DeviceBuffer[DT],
        action_dev: DeviceBuffer[DT],
        reward_dev: DeviceBuffer[DT],
        obs_dev: DeviceBuffer[DT],
        done_dev: DeviceBuffer[DT],
    ) raises:
        """N-step batched device record (Rainbow). The driver-owned
        `GPUNStepBuffer` ring-updates all N_ENVS lanes (per-env
        accumulators — no cross-env mixing) and emits compressed n-step
        transitions, which the sample block routes through its device
        replay. Keep `NS` aligned with the target-Y γ^N bootstrap (the
        Rainbow preset wires both from the same `NSTEP`)."""
        nstep_buf.process(
            ctx, prev_obs_dev, action_dev, reward_dev, obs_dev, done_dev,
        )
        self.sample_blk.store_via_block_gpu[N_ENVS, NS](ctx, nstep_buf)

    # ─── Action selection (expected-Q argmax over softmax·z) ─────────

    def _ensure_batch_scratch[
        N_ENVS: Int
    ](mut self, ctx: DeviceContext) raises:
        """Lazily (re)allocate the GPU batched-action scratch for N_ENVS:
        a device `[N_ENVS·NA·NK]` logits buffer + host mirrors for the
        D2H logits readback and the H2D action indices. One allocation per
        distinct N_ENVS (cached via `_batch_n`)."""
        comptime NA = Self.NUM_ACTIONS
        comptime NK = Self.N_ATOMS
        if self._batch_n == N_ENVS:
            return
        self._batch_q_dev = Optional(
            ctx.enqueue_create_buffer[DT](N_ENVS * NA * NK)
        )
        self._batch_q_host = List[Scalar[DT]](
            length=N_ENVS * NA * NK, fill=Scalar[DT](0.0)
        )
        self._batch_act_host = List[Scalar[DT]](
            length=N_ENVS, fill=Scalar[DT](0.0)
        )
        self._batch_n = N_ENVS

    def _expected_q_argmax(
        mut self, q_p: UnsafePointer[Scalar[DT], MutAnyOrigin],
    ) -> Int:
        """Compute argmax_a Σ_k softmax(logits[a])_k · z_k from a flat
        NUM_ACTIONS · N_ATOMS logits row (single sample)."""
        comptime NA = Self.NUM_ACTIONS
        comptime NK = Self.N_ATOMS
        var z_p = self.target_y_blk.z_ptr()
        var best_a: Int = 0
        var best_eq: Scalar[DT] = Scalar[DT](0.0)
        for a in range(NA):
            var base = a * NK
            var mx = q_p[base]
            for i in range(1, NK):
                if q_p[base + i] > mx:
                    mx = q_p[base + i]
            var s_exp: Scalar[DT] = Scalar[DT](0.0)
            for i in range(NK):
                s_exp = s_exp + fexp(q_p[base + i] - mx)
            var eq: Scalar[DT] = Scalar[DT](0.0)
            for i in range(NK):
                var p = fexp(q_p[base + i] - mx) / s_exp
                eq = eq + p * z_p[i]
            if a == 0 or eq > best_eq:
                best_eq = eq
                best_a = a
        return best_a

    def select_action_batched[
        N_ENVS: Int,
    ](
        mut self,
        obs_ptr: UnsafePointer[Scalar[DT], MutAnyOrigin],
        action_ptr: UnsafePointer[Scalar[DT], MutAnyOrigin],
        step_idx: Int,
    ) raises:
        comptime NA = Self.NUM_ACTIONS
        comptime NK = Self.N_ATOMS
        comptime OBS = Self.OBS_DIM

        # Warmup: uniform random action.
        if step_idx < self.learning_starts:
            comptime if Self.train_target == "cpu":
                for i in range(N_ENVS):
                    var r = random_float64()
                    action_ptr[i] = Scalar[DT](Int(r * Float64(NA)))
            else:
                # GPU warmup: draw N_ENVS uniform action indices on the host
                # and H2D them into the (device) action buffer. At N_ENVS=1
                # this consumes exactly one `random_float64` draw, matching the
                # legacy single-env path's RNG order.
                var ctx = self.ctx.value()
                self._ensure_batch_scratch[N_ENVS](ctx)
                var act_h = self._batch_act_host.unsafe_ptr()
                for i in range(N_ENVS):
                    act_h[i] = Scalar[DT](Int(random_float64() * Float64(NA)))
                var action_dev = DeviceBuffer[DT](
                    ctx, action_ptr, N_ENVS, owning=False,
                )
                ctx.enqueue_copy(action_dev, act_h)
            return

        comptime if Self.train_target == "cpu":
            # Policy: batched forward then per-env expected-Q argmax.
            var q_buf = List[Scalar[DT]](
                length=N_ENVS * NA * NK, fill=Scalar[DT](0.0),
            )
            var q_ptr = mptr(q_buf.unsafe_ptr())
            var obs_t = TileTensor(obs_ptr, row_major[N_ENVS, OBS]())
            var q_t = TileTensor(q_ptr, row_major[N_ENVS, NA * NK]())
            self.pair.online.forward[Self.train_target, N_ENVS](
                obs_t, output=q_t,
            )
            for i in range(N_ENVS):
                var r = random_float64()
                if r < Float64(self.epsilon):
                    action_ptr[i] = Scalar[DT](
                        Int(random_float64() * Float64(NA))
                    )
                else:
                    action_ptr[i] = Scalar[DT](
                        self._expected_q_argmax(q_ptr + i * NA * NK)
                    )
        else:
            # GPU policy: ONE batched device forward over all N_ENVS obs,
            # then D2H the [N_ENVS, NA·NK] logits and run the expected-Q
            # argmax (Σ_k softmax(logits[a])_k · z_k) per env on the host —
            # there is no batched device argmax kernel, and the readback is
            # tiny (N_ENVS·NA·NK floats). At N_ENVS=1 this reproduces the
            # legacy single-env behaviour (one forward, one host argmax).
            var ctx = self.ctx.value()
            self._ensure_batch_scratch[N_ENVS](ctx)
            var qdev = self._batch_q_dev.value()
            var qdev_ptr = mptr(qdev.unsafe_ptr())
            var obs_t = TileTensor(obs_ptr, row_major[N_ENVS, OBS]())
            var q_t = TileTensor(
                qdev_ptr, row_major[N_ENVS, NA * NK](),
            )
            self.pair.online.forward[Self.train_target, N_ENVS](
                obs_t, output=q_t,
            )
            var qh = self._batch_q_host.unsafe_ptr()
            ctx.enqueue_copy(qh, qdev)
            ctx.synchronize()
            var act_h = self._batch_act_host.unsafe_ptr()
            for i in range(N_ENVS):
                var r = random_float64()
                if r < Float64(self.epsilon):
                    act_h[i] = Scalar[DT](Int(random_float64() * Float64(NA)))
                else:
                    act_h[i] = Scalar[DT](
                        self._expected_q_argmax(qh + i * NA * NK)
                    )
            var action_dev = DeviceBuffer[DT](
                ctx, action_ptr, N_ENVS, owning=False,
            )
            ctx.enqueue_copy(action_dev, act_h)

    def set_noise_scale(mut self, scale: Scalar[DT]) raises:
        """Broadcast `noise_scale` to the online (acting) net's NoisyLinear
        layers: 1.0 = normal noisy exploration, 0.0 = deterministic mean
        weights. Bracket a greedy-eval rollout with `set_noise_scale(0)` …
        `set_noise_scale(1)`. No-op if the net has no Noisy layers (their
        non-overriding siblings ignore the `set_attr` broadcast)."""
        self.pair.online.set_attr["noise_scale"](scale)

    def select_greedy_action_batched[
        N_ENVS: Int,
    ](
        mut self,
        obs_ptr: UnsafePointer[Scalar[DT], MutAnyOrigin],
        action_ptr: UnsafePointer[Scalar[DT], MutAnyOrigin],
    ) raises:
        """Pure greedy: one batched forward over N_ENVS obs, expected-Q
        argmax per env — no epsilon, no warmup gate. Mirrors the policy
        branch of `select_action_batched` minus exploration. Pair with
        `set_noise_scale(0)` for a deterministic eval rollout."""
        comptime NA = Self.NUM_ACTIONS
        comptime NK = Self.N_ATOMS
        comptime OBS = Self.OBS_DIM
        comptime if Self.train_target == "cpu":
            var q_buf = List[Scalar[DT]](
                length=N_ENVS * NA * NK, fill=Scalar[DT](0.0),
            )
            var q_ptr = mptr(q_buf.unsafe_ptr())
            var obs_t = TileTensor(obs_ptr, row_major[N_ENVS, OBS]())
            var q_t = TileTensor(q_ptr, row_major[N_ENVS, NA * NK]())
            self.pair.online.forward[Self.train_target, N_ENVS](
                obs_t, output=q_t,
            )
            for i in range(N_ENVS):
                action_ptr[i] = Scalar[DT](
                    self._expected_q_argmax(q_ptr + i * NA * NK)
                )
        else:
            var ctx = self.ctx.value()
            self._ensure_batch_scratch[N_ENVS](ctx)
            var qdev = self._batch_q_dev.value()
            var qdev_ptr = mptr(qdev.unsafe_ptr())
            var obs_t = TileTensor(obs_ptr, row_major[N_ENVS, OBS]())
            var q_t = TileTensor(qdev_ptr, row_major[N_ENVS, NA * NK]())
            self.pair.online.forward[Self.train_target, N_ENVS](
                obs_t, output=q_t,
            )
            var qh = self._batch_q_host.unsafe_ptr()
            ctx.enqueue_copy(qh, qdev)
            ctx.synchronize()
            var act_h = self._batch_act_host.unsafe_ptr()
            for i in range(N_ENVS):
                act_h[i] = Scalar[DT](
                    self._expected_q_argmax(qh + i * NA * NK)
                )
            var action_dev = DeviceBuffer[DT](
                ctx, action_ptr, N_ENVS, owning=False,
            )
            ctx.enqueue_copy(action_dev, act_h)

    def select_greedy_action(
        mut self,
        ref obs: List[Scalar[DT]],
    ) raises -> Int:
        comptime NA = Self.NUM_ACTIONS
        comptime NK = Self.N_ATOMS
        comptime OBS = Self.OBS_DIM
        var ob1_p = self._ob1.cpu_ptr()
        for d in range(OBS):
            ob1_p[d] = obs[d]
        comptime if Self.train_target == "cpu":
            var ob1_t = TileTensor(ob1_p, row_major[1, OBS]())
            var q_p = self._q_logits.cpu_ptr()
            var q_t = TileTensor(q_p, row_major[1, NA * NK]())
            self.pair.online.forward[Self.train_target, 1](
                ob1_t, output=q_t,
            )
            return self._expected_q_argmax(q_p)
        else:
            var ctx = self.ctx.value()
            ctx.enqueue_copy(self._ob1.dev.value(), ob1_p)
            var ob1_t = TileTensor(
                self._ob1.dev_ptr(), row_major[1, OBS](),
            )
            var q_t = TileTensor(
                self._q_logits.dev_ptr(), row_major[1, NA * NK](),
            )
            self.pair.online.forward[Self.train_target, 1](
                ob1_t, output=q_t,
            )
            ctx.enqueue_copy(
                self._q_logits.cpu_ptr(), self._q_logits.dev.value(),
            )
            ctx.synchronize()
            return self._expected_q_argmax(self._q_logits.cpu_ptr())

    # ─── Episode tracking ────────────────────────────────────────────

    def end_episode(mut self):
        self.tracker.end_episode()
        self.epsilon = self.epsilon * self.epsilon_decay
        if self.epsilon < self.epsilon_min:
            self.epsilon = self.epsilon_min

    def add_complete_return(mut self, ret: Scalar[DT]):
        self.tracker.add_complete_return(ret)

    def mean_return(self) -> Scalar[DT]:
        return self.tracker.mean_return()

    def ep_count(self) -> Int:
        return self.tracker.ep_count

    # ─── Logging ─────────────────────────────────────────────────────

    def flush_train_log(
        mut self,
    ) -> Tuple[Scalar[DT], Scalar[DT], Int]:
        var n = self._update_count if self._update_count > 0 else 1
        var inv = Scalar[DT](1.0) / Scalar[DT](n)
        var out = (
            self._loss_accum * inv,
            self.epsilon,
            self._update_count,
        )
        self._loss_accum = Scalar[DT](0.0)
        # Keep diagnostic accumulators in lock-step with the chunk counter
        # (this legacy tuple API drops them; a later flush_metrics must not
        # divide a multi-chunk sum by one chunk's `n`).
        self._q_accum = Scalar[DT](0.0)
        self._target_accum = Scalar[DT](0.0)
        self._dist_entropy_accum = Scalar[DT](0.0)
        self._reward_accum = Scalar[DT](0.0)
        self._done_accum = Scalar[DT](0.0)
        self._update_count = 0
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
    ) raises -> C51Metrics:
        """Drain accumulators into a C51Metrics bundle. If a logger
        pointer is wired, also emit one log_scalar per metric field.
        Resets per-chunk accumulators on every call; the cumulative
        `_total_train_steps` counter is NOT reset."""
        var n = self._update_count if self._update_count > 0 else 1
        var inv = Scalar[DT](1.0) / Scalar[DT](n)
        # Distributional diag means: device-resident on GPU (derived by
        # `_c51_diag_kernel`), host scalars on CPU.
        var q_mean: Scalar[DT]
        var target_mean: Scalar[DT]
        var dist_entropy_mean: Scalar[DT]
        var reward_mean: Scalar[DT]
        var done_mean: Scalar[DT]
        # Loss: device-accumulated on GPU (no per-step D2H; read once here),
        # host-summed on CPU.
        var loss_mean: Scalar[DT]
        comptime if Self.train_target == "gpu":
            loss_mean = self.q_update_blk.inner.ce_loss.read_accum["gpu"]()
            q_mean = self._q_mean_dev.read["gpu"]()
            target_mean = self._target_mean_dev.read["gpu"]()
            dist_entropy_mean = self._dist_entropy_mean_dev.read["gpu"]()
            reward_mean = self._reward_mean_dev.read["gpu"]()
            done_mean = self._done_mean_dev.read["gpu"]()
        else:
            loss_mean = self._loss_accum * inv
            q_mean = self._q_accum * inv
            target_mean = self._target_accum * inv
            dist_entropy_mean = self._dist_entropy_accum * inv
            reward_mean = self._reward_accum * inv
            done_mean = self._done_accum * inv
        var bundle = C51Metrics(
            loss=LogScalar[DT](loss_mean),
            epsilon=LogScalar[DT](self.epsilon),
            mean_q=LogScalar[DT](q_mean),
            mean_target=LogScalar[DT](target_mean),
            dist_entropy=LogScalar[DT](dist_entropy_mean),
            mean_reward=LogScalar[DT](reward_mean),
            mean_done=LogScalar[DT](done_mean),
            train_steps=LogScalar[DT](Scalar[DT](self._total_train_steps)),
            n_updates=LogScalar[DT](Scalar[DT](self._update_count)),
        )
        self._loss_accum = Scalar[DT](0.0)
        self._q_accum = Scalar[DT](0.0)
        self._target_accum = Scalar[DT](0.0)
        self._dist_entropy_accum = Scalar[DT](0.0)
        self._reward_accum = Scalar[DT](0.0)
        self._done_accum = Scalar[DT](0.0)
        comptime if Self.train_target == "gpu":
            self.q_update_blk.inner.ce_loss.reset_accum["gpu"]()
            self._q_mean_dev.reset["gpu"]()
            self._target_mean_dev.reset["gpu"]()
            self._dist_entropy_mean_dev.reset["gpu"]()
            self._reward_mean_dev.reset["gpu"]()
            self._done_mean_dev.reset["gpu"]()
        self._update_count = 0
        if Bool(logger):
            log_bundle(logger.value()[], bundle, step)
        return bundle^

    # ─── Trait-uniform cadence hooks (consumed by the driver) ─────────

    def flush_metrics_through_logger[L: Logger](
        mut self,
        logger: Optional[UnsafePointer[L, MutAnyOrigin]],
        step: Int,
    ) raises:
        """Trait-uniform passthrough: drains the C51 metric accumulators
        through `flush_metrics` and discards the typed bundle. The
        driver calls this at the user's `diag_every` cadence so no
        chunking is needed."""
        _ = self.flush_metrics[L](logger, step)

    def save_state(mut self, path: String) raises:
        """One-file v2 checkpoint of the C51 module + optimizer + the
        ε-greedy exploration state. Sections: `q_net.*`, `q_opt.*`, then
        `eps.{epsilon,epsilon_decay,epsilon_min}`. Overwrites `path`. The
        on-disk format is byte-identical CPU vs GPU (device params synced
        to host on save); the ε state is a host scalar in both targets, so
        resume continues the decay schedule instead of restarting at ε=1.
        Replay buffer + episode tracker NOT included."""
        var body = String("")
        comptime if Self.train_target == "cpu":
            save_state_v2_body(self.pair.online, body, "q_net")
            save_optimizer_v2_body(self.q_opt, body, "q_opt")
        else:
            var c = self.ctx.value()
            save_state_v2_body_gpu(self.pair.online, body, "q_net", c)
            save_optimizer_v2_body_gpu(self.q_opt, body, "q_opt")
        SaveScalar[DT](self.epsilon).save(body, "eps.epsilon")
        SaveScalar[DT](self.epsilon_decay).save(body, "eps.epsilon_decay")
        SaveScalar[DT](self.epsilon_min).save(body, "eps.epsilon_min")
        save_counter_v2_body(self._total_train_steps, body, "_total_train_steps")
        var content = String("nn-ckpt v2\n") + body
        with open(path, "w") as f:
            f.write(content)

    def load_state(mut self, path: String) raises:
        """Inverse of `save_state`. Target net is hard-copied from the
        online net after the online params are restored. On GPU the
        device params + Adam moments are restored via host staging; the
        on-disk format is byte-identical to the CPU path."""
        var content = read_file_v2(path)
        var lines = split_lines_v2(content)
        expect_v2_header(lines)
        var idx: Int = 1
        comptime if Self.train_target == "cpu":
            load_state_v2_body(self.pair.online, lines, idx, "q_net")
            load_optimizer_v2_body(self.q_opt, lines, idx, "q_opt")
        else:
            var c = self.ctx.value()
            load_state_v2_body_gpu(self.pair.online, lines, idx, "q_net", c)
            load_optimizer_v2_body_gpu(self.q_opt, lines, idx, "q_opt")
        # ε-greedy exploration state (host scalar in both targets).
        var eps_w = SaveScalar[DT](self.epsilon)
        eps_w.load(lines, idx, "eps.epsilon")
        self.epsilon = eps_w.v
        var eps_decay_w = SaveScalar[DT](self.epsilon_decay)
        eps_decay_w.load(lines, idx, "eps.epsilon_decay")
        self.epsilon_decay = eps_decay_w.v
        var eps_min_w = SaveScalar[DT](self.epsilon_min)
        eps_min_w.load(lines, idx, "eps.epsilon_min")
        self.epsilon_min = eps_min_w.v
        load_counter_v2_body(
            self._total_train_steps, lines, idx, "_total_train_steps"
        )
        hard_copy_params[Self.train_target, M=Self.Q_NET](
            self.pair.online, self.pair.target_net, self.ctx,
        )

    def flush_timer_log(mut self) -> String:
        var report = self.timer.format_report()
        self.timer.reset()
        return report
