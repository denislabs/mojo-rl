"""MBPOTrainer — CPU/GPU model-based policy optimization.

Pipeline (6 blocks):
  DualSample → TargetY → TwinCritic → SACActor → AlphaUpdate → Polyak
(5 of 6 blocks reused unchanged from SAC; all carry GPU paths.)

Dynamics ensemble training + synthetic rollouts are NOT pipeline blocks —
they're trainer methods invoked from train_step on a `model_train_freq`
cadence (block decomposition isn't the right fit for multi-epoch /
multi-step orchestration).

Phase 4.3 added the GPU train path: `train_target` is the first comptime
param, the dual-sample block type is comptime-selected (CPU vs GPU), and
the dynamics-train + rollout phases have device implementations. The SAC
sub-update reuses the already-GPU SAC blocks (twin-critic / actor / alpha
on-device accumulation, no per-step D2H). The synthetic rollout runs fully
on-device: real-buffer start-state draw → batched actor forward + rsample
→ elite dynamics forward → posterior Gaussian sampling (device box-muller
noise) → device batch store into the GPU synth replay.

CPU is bit-identical to the prior CPU-only MBPOTrainer. Conforms to
`OffPolicyAgentGpu`.
"""

from std.math import exp as fexp, sqrt as fsqrt, log as flog
from std.memory import alloc
from std.random import random_float64, randn_float64
from std.time import perf_counter_ns
from std.gpu import block_dim, block_idx, thread_idx, global_idx
from std.gpu.host import DeviceContext, DeviceBuffer
from std.random.philox import Random as PhiloxRandom
from layout import Layout, LayoutTensor, TileTensor, row_major

from mojo_rl.core.logger import Logger, NoOpLogger
from mojo_rl.nn.constants import DT, TPB
from mojo_rl.nn.core import Module
from mojo_rl.nn.core.amp import AMPPolicy, NoAMP, Bf16Compute
from mojo_rl.nn.core.checkpoint import (
    save_state_v2_body, load_state_v2_body,
    save_state_v2_body_gpu, load_state_v2_body_gpu,
)
from mojo_rl.nn.core.log_bundle import log_bundle
from mojo_rl.nn.core.map_params import hard_copy_params
from mojo_rl.nn.core.metric import LogScalar
from mojo_rl.nn.core.save_scalar import SaveI
from ..core.checkpoint_helpers import (
    save_optimizer_v2_body, load_optimizer_v2_body,
    save_optimizer_v2_body_gpu, load_optimizer_v2_body_gpu,
    save_scalar_adam_v2_body, load_scalar_adam_v2_body,
    save_scalar_adam_v2_body_gpu, load_scalar_adam_v2_body_gpu,
    save_counter_v2_body, load_counter_v2_body,
    split_lines_v2, read_file_v2, expect_v2_header,
)
from ..core.online_target_pair import OnlineTargetPair
from mojo_rl.nn.core.scratch import Scratch
from mojo_rl.nn.core.scratch_walkers import init_scratch_auto
from ..data.n_step_replay import GPUNStepBuffer
from mojo_rl.nn.initializer import Xavier, Kaiming
from mojo_rl.nn.optimizer.adam import Adam
from mojo_rl.nn.optimizer.scalar_adam import ScalarAdam
from mojo_rl.nn.random.box_muller import box_muller_normal_gpu
from mojo_rl.nn.training.timer import Timer
from .dynamics_ensemble_block import DynamicsEnsembleBlock
from ..training.episode_tracker import EpisodeTracker
from ..training.device_mean_accum import DeviceMeanAccum
from ..training.trainer_block import TrainerState
from ..training.driver_offpolicy import OffPolicyAgentGpu
from ..training.off_policy_critic import concat_sa_gpu
from ..training.blocks import DualSampleStep, TwinCriticStep, PolyakStep
from ..sac.target_y_block import TargetYBlock
from ..sac.blocks.actor_step import SACActorStep
from ..sac.blocks.alpha_update_step import AlphaUpdateStep
from .metrics import MBPOMetrics


# ──────────────────────────────────────────────────────────────────────
# Batched GPU action kernels (mirror SAC) + MBPO rollout/dynamics kernels.
# ──────────────────────────────────────────────────────────────────────


def _mbpo_warmup_uniform_kernel[
    N_ENVS: Int, ACT: Int
](
    action_dest: LayoutTensor[DT, Layout.row_major(N_ENVS, ACT), MutAnyOrigin],
    action_scale: Scalar[DT],
    seed: UInt64,
    offset_base: UInt64,
):
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


def _mbpo_action_clamp_kernel[
    N: Int, ACT: Int
](
    alp: LayoutTensor[DT, Layout.row_major(N, ACT + 1), MutAnyOrigin],
    action_out: LayoutTensor[DT, Layout.row_major(N, ACT), MutAnyOrigin],
    action_scale: Scalar[DT],
):
    """Extract the first ACT lanes of the rsample output (drop log_prob),
    clamp into action_out. Used for both env-step batched action and the
    rollout policy action on imagined obs."""
    var i = Int(block_dim.x * block_idx.x + thread_idx.x)
    var total = N * ACT
    if i >= total:
        return
    var env = i // ACT
    var j = i % ACT
    var a = alp[env, j]
    if a > action_scale:
        a = action_scale
    elif a < -action_scale:
        a = -action_scale
    action_out[env, j] = a


def _rollout_posterior_kernel[
    BATCH: Int, OBS: Int, PRED: Int
](
    obs: LayoutTensor[DT, Layout.row_major(BATCH, OBS), MutAnyOrigin],
    mu: LayoutTensor[DT, Layout.row_major(BATCH, PRED), MutAnyOrigin],
    lv: LayoutTensor[DT, Layout.row_major(BATCH, PRED), MutAnyOrigin],
    noise: LayoutTensor[DT, Layout.row_major(BATCH, PRED), MutAnyOrigin],
    out_nxt: LayoutTensor[DT, Layout.row_major(BATCH, OBS), MutAnyOrigin],
    out_rew: LayoutTensor[DT, Layout.row_major(BATCH), MutAnyOrigin],
):
    """Posterior sample one rollout step: PRED = 1 + OBS (reward + Δobs).
    `rew = μ_r + exp(0.5·lv_r)·z_r`; `nxt[d] = obs[d] + μ_δ[d] +
    exp(0.5·lv_δ[d])·z_δ[d]`. Standard-normal `z` is the pre-sampled
    device box-muller `noise`. One thread per lane."""
    var k = Int(block_dim.x * block_idx.x + thread_idx.x)
    if k >= BATCH:
        return
    var mu_r = rebind[Scalar[DT]](mu[k, 0])
    var lv_r = rebind[Scalar[DT]](lv[k, 0])
    var z_r = rebind[Scalar[DT]](noise[k, 0])
    # Legacy parity (mbpo_agent.mojo clamp_rewards_kernel): clamp the
    # model-predicted synthetic reward to [-100, 100] and map NaN→0. Guards
    # against NaN cascades + caps the worst OOD optimism spikes the policy
    # could otherwise exploit.
    var r = mu_r + fexp(Scalar[DT](0.5) * lv_r) * z_r
    if r != r:
        r = Scalar[DT](0.0)
    elif r < Scalar[DT](-100.0):
        r = Scalar[DT](-100.0)
    elif r > Scalar[DT](100.0):
        r = Scalar[DT](100.0)
    out_rew[k] = r
    for d in range(OBS):
        var mu_d = rebind[Scalar[DT]](mu[k, 1 + d])
        var lv_d = rebind[Scalar[DT]](lv[k, 1 + d])
        var z_d = rebind[Scalar[DT]](noise[k, 1 + d])
        var delta = mu_d + fexp(Scalar[DT](0.5) * lv_d) * z_d
        out_nxt[k, d] = rebind[Scalar[DT]](obs[k, d]) + delta


def _mbpo_elite_assign_kernel[BATCH: Int](
    slot: LayoutTensor[DT, Layout.row_major(BATCH), MutAnyOrigin],
    n_elites: Int32,
    seed: UInt64,
    offset: UInt64,
):
    """Per-transition random elite slot ∈ [0, n_elites), stored as DT. One
    thread per lane. Mirrors legacy `sample_elite_assignment_kernel` /
    softlearning `np.random.choice(elite_inds, size=batch)`."""
    var b = Int(block_dim.x * block_idx.x + thread_idx.x)
    if b >= BATCH:
        return
    var philox = PhiloxRandom(seed=seed + UInt64(b), offset=offset)
    var u = Float32(philox.step_uniform()[0])
    var ne = Int(n_elites)
    var s = Int(u * Float32(ne))
    if s >= ne:
        s = ne - 1
    if s < 0:
        s = 0
    slot[b] = Scalar[DT](s)


def _mbpo_elite_gather_kernel[BATCH: Int, NELITES: Int, PRED: Int](
    mu_all: LayoutTensor[
        DT, Layout.row_major(NELITES * BATCH, PRED), MutAnyOrigin
    ],
    lv_all: LayoutTensor[
        DT, Layout.row_major(NELITES * BATCH, PRED), MutAnyOrigin
    ],
    slot: LayoutTensor[DT, Layout.row_major(BATCH), MutAnyOrigin],
    out_mu: LayoutTensor[DT, Layout.row_major(BATCH, PRED), MutAnyOrigin],
    out_lv: LayoutTensor[DT, Layout.row_major(BATCH, PRED), MutAnyOrigin],
):
    """Per-transition gather: `out[b,:] = all[slot[b]·BATCH + b, :]` (the
    stacked elite outputs are laid out [NELITES, BATCH, PRED]). One thread
    per lane."""
    var b = Int(block_dim.x * block_idx.x + thread_idx.x)
    if b >= BATCH:
        return
    var s = Int(rebind[Scalar[DT]](slot[b]))
    if s < 0:
        s = 0
    if s >= NELITES:
        s = NELITES - 1
    var row = s * BATCH + b
    for d in range(PRED):
        out_mu[b, d] = rebind[Scalar[DT]](mu_all[row, d])
        out_lv[b, d] = rebind[Scalar[DT]](lv_all[row, d])


def _build_dyn_target_kernel[
    BATCH: Int, OBS: Int, PRED: Int
](
    rew: LayoutTensor[DT, Layout.row_major(BATCH), MutAnyOrigin],
    s: LayoutTensor[DT, Layout.row_major(BATCH, OBS), MutAnyOrigin],
    sp: LayoutTensor[DT, Layout.row_major(BATCH, OBS), MutAnyOrigin],
    out_tgt: LayoutTensor[DT, Layout.row_major(BATCH, PRED), MutAnyOrigin],
):
    """Dynamics target = [reward, Δobs = s' − s]. PRED = 1 + OBS. One
    thread per lane."""
    var k = Int(block_dim.x * block_idx.x + thread_idx.x)
    if k >= BATCH:
        return
    out_tgt[k, 0] = rebind[Scalar[DT]](rew[k])
    for d in range(OBS):
        out_tgt[k, 1 + d] = (
            rebind[Scalar[DT]](sp[k, d]) - rebind[Scalar[DT]](s[k, d])
        )


def _normalize_input_kernel[
    BATCH: Int, D: Int
](
    data: LayoutTensor[DT, Layout.row_major(BATCH, D), MutAnyOrigin],
    mean: LayoutTensor[DT, Layout.row_major(D), MutAnyOrigin],
    std: LayoutTensor[DT, Layout.row_major(D), MutAnyOrigin],
):
    """In-place per-column z-score of the dynamics input: data[b, c] =
    (data[b, c] − mean[c]) / std[c]. One thread per element. Matches the
    legacy MBPO `normalize_input_kernel` so unbounded obs (HalfCheetah-
    style) are whitened before the ensemble forward."""
    var i = Int(block_dim.x * block_idx.x + thread_idx.x)
    if i >= BATCH * D:
        return
    var b = i // D
    var c = i % D
    var v = rebind[Scalar[DT]](data[b, c])
    var m = rebind[Scalar[DT]](mean[c])
    var s = rebind[Scalar[DT]](std[c])
    data[b, c] = (v - m) / s


struct MBPOTrainer[
    train_target: StaticString,
    ACTOR: Module,
    CRITIC: Module,
    DynNet: Module,
    OBS_DIM: Int,
    ACT_DIM: Int,
    BATCH: Int,
    REPLAY_CAPACITY: Int,
    SYNTH_CAPACITY: Int,
    N_ENSEMBLE: Int,
    NUM_ELITES: Int,
    REAL_RATIO_PCT: Int = 5,
    LOGVAR_MIN: Float64 = -10.0,
    LOGVAR_MAX: Float64 = -2.0,
](OffPolicyAgentGpu):
    comptime DYN_IN: Int = Self.OBS_DIM + Self.ACT_DIM
    comptime DYN_PRED: Int = 1 + Self.OBS_DIM
    comptime DYN_OUT: Int = 2 * Self.DYN_PRED
    comptime REAL_BS: Int = (Self.BATCH * Self.REAL_RATIO_PCT) // 100
    comptime SYNTH_BS: Int = Self.BATCH - Self.REAL_BS

    comptime ENSEMBLE = DynamicsEnsembleBlock[
        Self.DynNet,
        Self.N_ENSEMBLE,
        Self.NUM_ELITES,
        Self.DYN_IN,
        Self.DYN_OUT,
        Self.BATCH,
        Self.LOGVAR_MIN,
        Self.LOGVAR_MAX,
    ]

    # Dual-sample block — one concrete dual-storage type for both targets
    # (CPU + GPU storage carried together; only the matching one is built
    # in setup[target]). Avoids the ternary-over-two-struct-types issue.
    comptime SampleBlk = DualSampleStep[
        Self.OBS_DIM, Self.ACT_DIM, Self.BATCH,
        Self.REPLAY_CAPACITY, Self.SYNTH_CAPACITY,
        Self.REAL_BS, Self.SYNTH_BS,
    ]

    comptime AGENT_OBS_DIM: Int = Self.OBS_DIM
    comptime AGENT_ACT_DIM: Int = Self.ACT_DIM
    comptime AGENT_TRAIN_TARGET: StaticString = Self.train_target

    comptime _T_DYN_TRAIN = 0
    comptime _T_ROLLOUT   = 1
    comptime _T_SAMPLE    = 2
    comptime _T_TARGET_Y  = 3
    comptime _T_CRITIC    = 4
    comptime _T_ACTOR     = 5
    comptime _T_ALPHA     = 6
    comptime _T_POLYAK    = 7
    comptime _T_DIAG      = 8

    var actor: Self.ACTOR
    var pair1: OnlineTargetPair[Self.CRITIC]
    var pair2: OnlineTargetPair[Self.CRITIC]
    var actor_opt: Adam
    var critic1_opt: Adam
    var critic2_opt: Adam
    var alpha_opt: ScalarAdam

    var sample_blk: Self.SampleBlk
    var target_y_blk: TargetYBlock[
        Self.ACTOR, Self.CRITIC, Self.BATCH, Self.OBS_DIM, Self.ACT_DIM,
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

    var ensemble: Self.ENSEMBLE
    var state: TrainerState[Self.OBS_DIM, Self.ACT_DIM, Self.BATCH]
    var tracker: EpisodeTracker
    var ctx: Optional[DeviceContext]

    # select_action scratches (staging — host mirror + device).
    var _ob1: Scratch["ob1", Self.OBS_DIM, True]
    var _ao1: Scratch["ao1", 2 * Self.ACT_DIM, True]
    var _alp1: Scratch["alp1", Self.ACT_DIM + 1, True]

    # Dynamics training / rollout scratches.
    var _dyn_in: Scratch["dyn_in", Self.BATCH * Self.DYN_IN]
    var _dyn_tgt: Scratch["dyn_tgt", Self.BATCH * Self.DYN_PRED]
    var _ro_obs: Scratch["ro_obs", Self.BATCH * Self.OBS_DIM]
    var _ro_act: Scratch["ro_act", Self.BATCH * Self.ACT_DIM]
    var _ro_nxt: Scratch["ro_nxt", Self.BATCH * Self.OBS_DIM]
    var _ro_mu: Scratch["ro_mu", Self.BATCH * Self.DYN_PRED]
    var _ro_lv: Scratch["ro_lv", Self.BATCH * Self.DYN_PRED]
    # Stacked elite outputs [NUM_ELITES, BATCH, DYN_PRED] + per-transition
    # elite slot [BATCH]: rollouts forward ALL elites then gather a random
    # elite PER TRANSITION (legacy `sample_elite_assignment_kernel` /
    # softlearning fake_env: `np.random.choice(elite_inds, size=batch)`).
    # One-elite-per-chunk lets the policy exploit a single model's bias →
    # over-confident/optimistic synthetic reward (the remaining gap after the
    # dyn early-stopping fix).
    var _ro_mu_all: Scratch[
        "ro_mu_all", Self.NUM_ELITES * Self.BATCH * Self.DYN_PRED
    ]
    var _ro_lv_all: Scratch[
        "ro_lv_all", Self.NUM_ELITES * Self.BATCH * Self.DYN_PRED
    ]
    var _ro_slot: Scratch["ro_slot", Self.BATCH]
    # GPU-only rollout extras (allocated on both targets; unused on CPU).
    var _ro_rew: Scratch["ro_rew", Self.BATCH]
    var _ro_done: Scratch["ro_done", Self.BATCH]
    var _ro_noise: Scratch["ro_noise", Self.BATCH * Self.DYN_PRED]
    var _ro_ao: Scratch["ro_ao", Self.BATCH * 2 * Self.ACT_DIM]
    var _ro_alp: Scratch["ro_alp", Self.BATCH * (Self.ACT_DIM + 1)]

    # Dynamics input scaler — per-DYN_IN-dim z-score (mean/std), staging
    # (host mirror + device). Re-fit from the real buffer at the start of
    # every dynamics-train round and applied to `dyn_in` in BOTH training
    # and rollout so the world model always sees whitened inputs. Identity
    # (mean=0, std=1) until the first fit. Runtime-only — not checkpointed
    # (the ensemble is refit each round, so the scaler is too).
    var _in_mean: Scratch["in_mean", Self.DYN_IN, True]
    var _in_std: Scratch["in_std", Self.DYN_IN, True]

    var action_scale: Scalar[DT]
    var learning_starts: Int

    var model_train_freq: Int
    var dyn_epochs_per_round: Int
    var rollout_length: Int
    var num_rollouts_per_step: Int
    var sac_updates_per_step: Int
    var dyn_batch_size: Int
    var last_dyn_step: Int
    var _use_bf16: Bool
    # Dynamics-training early-stopping (legacy `train_on_buffer` discipline):
    # split the real buffer into a fixed train/holdout set and train each
    # member until the HELD-OUT NLL stops improving, instead of a fixed
    # `dyn_epochs_per_round` on the whole buffer. Prevents the world model
    # from overfitting → over-confident → optimistic on the policy's OOD
    # rollout actions (the root cause of the nn-MBPO convergence gap).
    var dyn_holdout_ratio: Scalar[DT]  # fraction held out (legacy 0.2)
    var dyn_max_epochs: Int            # epoch ceiling (legacy 100)
    var dyn_patience: Int              # stop after N checks w/o improve (5)
    var dyn_holdout_check_every: Int   # eval holdout every N epochs (5)

    # Philox state for GPU warmup actions + rollout noise.
    var _warmup_rng_seed: UInt64
    var _warmup_rng_offset: UInt64
    var _roll_rng_seed: UInt64
    var _roll_rng_offset: UInt64
    var _elite_rng_offset: UInt64  # per-transition elite-slot assignment

    var _actor_L_accum: Scalar[DT]
    var _critic_L_accum: Scalar[DT]
    var _update_count: Int
    var _total_train_steps: Int

    var _q_accum: Scalar[DT]
    var _reward_accum: Scalar[DT]
    # Per-update batch diagnostics (legacy parity): mean TD target `mb_y` and
    # done-ratio `mean(mb_d)`. Same CPU-host-scalar / GPU-device-accum split
    # as `_q_accum` / `_q_mean_dev`.
    var _td_accum: Scalar[DT]
    var _done_accum: Scalar[DT]
    # Mean |action| over the mixed batch (legacy `mean_abs_action`): the
    # cleanest proxy for "is the policy committing" vs staying timid.
    var _action_abs_accum: Scalar[DT]
    var _dyn_loss_accum: Scalar[DT]
    var _dyn_step_count: Int
    # Last computed dynamics-NLL mean. The ensemble only trains on the
    # `model_train_freq` cadence, so most diag flushes (now keyed on
    # TRAIN steps, far finer than the dyn-train cadence) see zero new dyn
    # member-steps. Reporting `accum/count`=0 on those flushes produced a
    # 0↔value sawtooth. We instead HOLD the last real round's mean between
    # rounds (piecewise-constant, like legacy's per-round `dyn_holdout_loss`).
    var _dyn_loss_last: Scalar[DT]
    # Dynamics holdout suite — refreshed each model-train round (from the dyn
    # eval that already feeds `update_elites`), held between rounds. Surfaced
    # via MBPOMetrics for clean overlay with legacy MBPO.
    var _dyn_holdout_loss: Scalar[DT]
    var _dyn_holdout_min: Scalar[DT]
    var _dyn_holdout_max: Scalar[DT]
    var _dyn_input_std_mean: Scalar[DT]

    # GPU-only device-resident mean accumulators for `mean_q` / `mean_reward`
    # (CPU uses the `_q_accum` / `_reward_accum` host scalars above).
    var _q_mean_dev: DeviceMeanAccum
    var _reward_mean_dev: DeviceMeanAccum
    var _td_mean_dev: DeviceMeanAccum
    var _done_mean_dev: DeviceMeanAccum
    var _action_abs_mean_dev: DeviceMeanAccum

    var timer: Timer

    def __init__(out self):
        comptime assert (
            Self.DynNet.IN_DIMS[0] == Self.DYN_IN
        ), "MBPOTrainer: DynNet.IN_DIM must equal OBS_DIM + ACT_DIM"
        comptime assert (
            Self.DynNet.OUT_DIM == Self.DYN_OUT
        ), "MBPOTrainer: DynNet.OUT_DIM must equal 2 * (1 + OBS_DIM)"
        comptime assert (
            Self.REAL_RATIO_PCT >= 0 and Self.REAL_RATIO_PCT <= 100
        ), "REAL_RATIO_PCT must be in [0, 100]"
        comptime assert Self.REAL_BS >= 1, "REAL_BS must be >= 1"
        comptime assert Self.SYNTH_BS >= 1, "SYNTH_BS must be >= 1"

        self.actor = Self.ACTOR()
        self.pair1 = OnlineTargetPair[Self.CRITIC]()
        self.pair2 = OnlineTargetPair[Self.CRITIC]()
        self.actor_opt = Adam()
        self.critic1_opt = Adam()
        self.critic2_opt = Adam()
        self.alpha_opt = ScalarAdam(
            value=0.0, m=0.0, v=0.0, t=0,
            lr=0.0003, beta1=0.9, beta2=0.999, eps=1e-8,
        )
        self.sample_blk = Self.SampleBlk()
        self.target_y_blk = TargetYBlock[
            Self.ACTOR, Self.CRITIC, Self.BATCH, Self.OBS_DIM, Self.ACT_DIM,
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
        self.ensemble = Self.ENSEMBLE()
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
        self._ob1 = Scratch["ob1", Self.OBS_DIM, True]()
        self._ao1 = Scratch["ao1", 2 * Self.ACT_DIM, True]()
        self._alp1 = Scratch["alp1", Self.ACT_DIM + 1, True]()
        self._dyn_in = Scratch["dyn_in", Self.BATCH * Self.DYN_IN]()
        self._dyn_tgt = Scratch["dyn_tgt", Self.BATCH * Self.DYN_PRED]()
        self._ro_obs = Scratch["ro_obs", Self.BATCH * Self.OBS_DIM]()
        self._ro_act = Scratch["ro_act", Self.BATCH * Self.ACT_DIM]()
        self._ro_nxt = Scratch["ro_nxt", Self.BATCH * Self.OBS_DIM]()
        self._ro_mu = Scratch["ro_mu", Self.BATCH * Self.DYN_PRED]()
        self._ro_lv = Scratch["ro_lv", Self.BATCH * Self.DYN_PRED]()
        self._ro_mu_all = Scratch[
            "ro_mu_all", Self.NUM_ELITES * Self.BATCH * Self.DYN_PRED
        ]()
        self._ro_lv_all = Scratch[
            "ro_lv_all", Self.NUM_ELITES * Self.BATCH * Self.DYN_PRED
        ]()
        self._ro_slot = Scratch["ro_slot", Self.BATCH]()
        self._ro_rew = Scratch["ro_rew", Self.BATCH]()
        self._ro_done = Scratch["ro_done", Self.BATCH]()
        self._ro_noise = Scratch["ro_noise", Self.BATCH * Self.DYN_PRED]()
        self._ro_ao = Scratch["ro_ao", Self.BATCH * 2 * Self.ACT_DIM]()
        self._ro_alp = Scratch["ro_alp", Self.BATCH * (Self.ACT_DIM + 1)]()
        self._in_mean = Scratch["in_mean", Self.DYN_IN, True]()
        self._in_std = Scratch["in_std", Self.DYN_IN, True]()
        self.action_scale = Scalar[DT](1.0)
        self.learning_starts = 1_000
        self.model_train_freq = 250
        self.dyn_epochs_per_round = 4
        self.rollout_length = 1
        self.num_rollouts_per_step = 400
        self.sac_updates_per_step = 20
        self.dyn_batch_size = 256
        self.dyn_holdout_ratio = Scalar[DT](0.2)
        # Ceiling only — early-stopping on the held-out NLL is the real
        # regularizer and typically stops well before this. Kept below
        # legacy's 100 because nn's dyn-train loop isn't CUDA-graphed (one
        # loss D2H per member-step), so the ceiling bounds worst-case wall
        # time per model-train round.
        self.dyn_max_epochs = 40
        self.dyn_patience = 5
        self.dyn_holdout_check_every = 5
        self.last_dyn_step = -1
        self._use_bf16 = False
        self._warmup_rng_seed = UInt64(0xC0FFEE_C0DE)
        self._warmup_rng_offset = UInt64(0)
        self._roll_rng_seed = UInt64(0xB0A75E_D00D)
        self._roll_rng_offset = UInt64(0)
        self._elite_rng_offset = UInt64(0)
        self._actor_L_accum = Scalar[DT](0.0)
        self._critic_L_accum = Scalar[DT](0.0)
        self._update_count = 0
        self._total_train_steps = 0
        self._q_accum = Scalar[DT](0.0)
        self._reward_accum = Scalar[DT](0.0)
        self._td_accum = Scalar[DT](0.0)
        self._done_accum = Scalar[DT](0.0)
        self._action_abs_accum = Scalar[DT](0.0)
        self._dyn_loss_accum = Scalar[DT](0.0)
        self._dyn_step_count = 0
        self._dyn_loss_last = Scalar[DT](0.0)
        self._dyn_holdout_loss = Scalar[DT](0.0)
        self._dyn_holdout_min = Scalar[DT](0.0)
        self._dyn_holdout_max = Scalar[DT](0.0)
        self._dyn_input_std_mean = Scalar[DT](0.0)
        self._q_mean_dev = DeviceMeanAccum()
        self._reward_mean_dev = DeviceMeanAccum()
        self._td_mean_dev = DeviceMeanAccum()
        self._done_mean_dev = DeviceMeanAccum()
        self._action_abs_mean_dev = DeviceMeanAccum()
        self.timer = Timer.new()

    @staticmethod
    def make(
        ctx: Optional[DeviceContext] = None,
        actor_lr: Scalar[DT] = Scalar[DT](3e-4),
        critic_lr: Scalar[DT] = Scalar[DT](3e-4),
        alpha_lr: Scalar[DT] = Scalar[DT](3e-4),
        model_lr: Scalar[DT] = Scalar[DT](1e-3),
        gamma: Scalar[DT] = Scalar[DT](0.99),
        tau: Scalar[DT] = Scalar[DT](0.005),
        action_scale: Scalar[DT] = Scalar[DT](1.0),
        init_alpha: Scalar[DT] = Scalar[DT](0.2),
        target_entropy: Scalar[DT] = Scalar[DT](-1.0),
        learning_starts: Int = 1_000,
        window_size: Int = 10,
        initial_episode_fill: Scalar[DT] = Scalar[DT](-1250.0),
        model_train_freq: Int = 250,
        dyn_epochs_per_round: Int = 4,
        rollout_length: Int = 1,
        num_rollouts_per_step: Int = 400,
        sac_updates_per_step: Int = 20,
        dyn_batch_size: Int = 256,
        dyn_max_epochs: Int = 40,
        dyn_weight_decay: Scalar[DT] = Scalar[DT](5e-5),
        dyn_learnable_bounds: Bool = False,
        use_bf16: Bool = False,
    ) raises -> Self:
        comptime assert (
            Self.train_target == "cpu" or Self.train_target == "gpu"
        ), "MBPOTrainer: target must be 'cpu' or 'gpu'"
        comptime if Self.train_target == "gpu":
            if not ctx:
                raise Error("MBPOTrainer.make[target='gpu']: ctx required")

        var t = Self()
        t.ctx = ctx

        t.actor = Self.ACTOR.make[target=Self.train_target, INIT=Xavier](ctx=ctx)
        t.pair1 = OnlineTargetPair[Self.CRITIC].make[
            target=Self.train_target, INIT=Xavier
        ](ctx=ctx)
        t.pair2 = OnlineTargetPair[Self.CRITIC].make[
            target=Self.train_target, INIT=Xavier
        ](ctx=ctx)
        t.actor_opt = Adam.make[target=Self.train_target, M=Self.ACTOR](
            t.actor, ctx=ctx,
        )
        t.actor_opt.lr = actor_lr
        t.critic1_opt = Adam.make[target=Self.train_target, M=Self.CRITIC](
            t.pair1.online, ctx=ctx,
        )
        t.critic1_opt.lr = critic_lr
        t.critic2_opt = Adam.make[target=Self.train_target, M=Self.CRITIC](
            t.pair2.online, ctx=ctx,
        )
        t.critic2_opt.lr = critic_lr

        comptime if Self.train_target == "gpu":
            t.alpha_opt = ScalarAdam.new_device(
                ctx.value(), flog(init_alpha), alpha_lr,
            )
        else:
            t.alpha_opt = ScalarAdam.new(flog(init_alpha), alpha_lr)

        t.target_y_blk = TargetYBlock[
            Self.ACTOR, Self.CRITIC, Self.BATCH, Self.OBS_DIM, Self.ACT_DIM,
        ].make[Self.train_target](
            action_scale=action_scale, gamma=gamma, ctx=ctx,
        )
        t.twin_critic_blk = TwinCriticStep[
            Self.OBS_DIM, Self.ACT_DIM, Self.BATCH, Self.CRITIC,
        ].make[Self.train_target](ctx=ctx)
        t.actor_blk = SACActorStep[
            Self.OBS_DIM, Self.ACT_DIM, Self.BATCH, Self.ACTOR, Self.CRITIC,
        ].make[Self.train_target](action_scale=action_scale, ctx=ctx)

        comptime if Self.train_target == "gpu":
            var alpha_p = t.alpha_opt.alpha_dev_ptr()
            t.target_y_blk.set_alpha_ptr(alpha_p)
            t.actor_blk.set_alpha_ptr(alpha_p)
            # Device-resident mean accumulators for the GPU diag path.
            t._q_mean_dev = DeviceMeanAccum.make["gpu"](ctx=ctx)
            t._reward_mean_dev = DeviceMeanAccum.make["gpu"](ctx=ctx)
            t._td_mean_dev = DeviceMeanAccum.make["gpu"](ctx=ctx)
            t._done_mean_dev = DeviceMeanAccum.make["gpu"](ctx=ctx)
            t._action_abs_mean_dev = DeviceMeanAccum.make["gpu"](ctx=ctx)

        t.alpha_blk = AlphaUpdateStep[
            Self.OBS_DIM, Self.ACT_DIM, Self.BATCH,
        ].make(target_entropy=target_entropy)
        t.polyak_blk = PolyakStep[
            Self.OBS_DIM, Self.ACT_DIM, Self.BATCH, Self.CRITIC,
        ].make(tau=tau)

        comptime if Self.train_target == "gpu":
            t.ensemble = Self.ENSEMBLE.make[Self.train_target, INIT=Kaiming](
                ctx.value()
            )
        else:
            t.ensemble = Self.ENSEMBLE.make[Self.train_target, INIT=Kaiming]()
        t.ensemble.set_lr(model_lr)
        t.ensemble.set_weight_decay(dyn_weight_decay)
        if dyn_learnable_bounds:
            t.ensemble.enable_learnable_bounds()

        t.tracker = EpisodeTracker.new(
            window_size=window_size, initial_fill=initial_episode_fill
        )
        t.state = TrainerState[
            Self.OBS_DIM, Self.ACT_DIM, Self.BATCH,
        ].make[Self.train_target](ctx=ctx)

        init_scratch_auto[Self, target=Self.train_target](t, ctx)
        t._set_scaler_identity[Self.train_target]()

        t.action_scale = action_scale
        t.learning_starts = learning_starts
        t.model_train_freq = model_train_freq
        t.dyn_epochs_per_round = dyn_epochs_per_round
        t.rollout_length = rollout_length
        t.num_rollouts_per_step = num_rollouts_per_step
        t.sac_updates_per_step = sac_updates_per_step
        t.dyn_batch_size = dyn_batch_size
        t.dyn_max_epochs = dyn_max_epochs
        t._use_bf16 = use_bf16

        t.sample_blk.setup[Self.train_target](learning_starts, ctx=ctx)

        t.timer.add_section("dyn_train")
        t.timer.add_section("rollout")
        t.timer.add_section("sample")
        t.timer.add_section("target_y")
        t.timer.add_section("critic")
        t.timer.add_section("actor")
        t.timer.add_section("alpha")
        t.timer.add_section("polyak")
        t.timer.add_section("diag")
        return t^

    # ─── Direct-callable (host-list) surface ─────────────────────────

    def select_action(
        mut self,
        ref obs: List[Scalar[DT]],
        mut action_out: List[Scalar[DT]],
        step_idx: Int,
    ) raises:
        comptime if Self.train_target == "cpu":
            self._select_action_cpu(obs, action_out, step_idx)
        else:
            # GPU: stage obs H2D into _ob1, delegate to the batched device
            # path (writes the clamped action into the first ACT scalars of
            # _alp1), D2H the action back.
            var ob1_cpu_p = self._ob1.cpu_ptr()
            for d in range(Self.OBS_DIM):
                ob1_cpu_p[d] = obs[d]
            self.ctx.value().enqueue_copy(self._ob1.dev.value(), ob1_cpu_p)
            self.select_action_batched[1](
                self._ob1.target_ptr["gpu"](),
                self._alp1.target_ptr["gpu"](),
                self._ao1.target_ptr["gpu"](),
                self._alp1.target_ptr["gpu"](),
                step_idx,
            )
            var ctx = self.ctx.value()
            ctx.enqueue_copy(self._alp1.cpu_ptr(), self._alp1.dev.value())
            ctx.synchronize()
            var alp1_cpu_p = self._alp1.cpu_ptr()
            for j in range(Self.ACT_DIM):
                action_out[j] = alp1_cpu_p[j]

    def _select_action_cpu(
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
        var alp1_t = TileTensor(alp1_cpu_p, row_major[1, Self.ACT_DIM + 1]())
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
        from std.math import tanh as ftanh

        var ob1_cpu_p = self._ob1.cpu_ptr()
        var ao1_cpu_p = self._ao1.cpu_ptr()
        for d in range(Self.OBS_DIM):
            ob1_cpu_p[d] = obs[d]
        comptime if Self.train_target == "cpu":
            var ob1_t = TileTensor(ob1_cpu_p, row_major[1, Self.OBS_DIM]())
            var ao1_t = TileTensor(ao1_cpu_p, row_major[1, 2 * Self.ACT_DIM]())
            self.actor.forward["cpu", 1](ob1_t, output=ao1_t)
        else:
            var ctx = self.ctx.value()
            ctx.enqueue_copy(self._ob1.dev.value(), ob1_cpu_p)
            var ob1_t = TileTensor(self._ob1.dev_ptr(), row_major[1, Self.OBS_DIM]())
            var ao1_t = TileTensor(
                self._ao1.dev_ptr(), row_major[1, 2 * Self.ACT_DIM]()
            )
            self.actor.forward["gpu", 1](ob1_t, output=ao1_t)
            ctx.enqueue_copy(ao1_cpu_p, self._ao1.dev.value())
            ctx.synchronize()
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
        self.sample_blk.real_add[Self.train_target](
            obs, action, reward, next_obs, done, ctx=self.ctx,
        )

    # `end_episode` / `mean_return` / `ep_count` / `add_complete_return`
    # are OffPolicyAgent trait defaults (S6) over this single accessor.
    def _tracker_ptr(self) -> UnsafePointer[EpisodeTracker, MutAnyOrigin]:
        return rebind[UnsafePointer[EpisodeTracker, MutAnyOrigin]](
            UnsafePointer(to=self.tracker)
        )

    def train_step(mut self, step_idx: Int) raises -> Bool:
        if step_idx < self.learning_starts:
            return False

        var should_train_dyn = (
            self.last_dyn_step < 0
            or step_idx - self.last_dyn_step >= self.model_train_freq
        )
        if should_train_dyn:
            # Re-fit the input scaler on the latest real buffer BEFORE both
            # the dynamics train and the rollout so they share one whitening.
            comptime if Self.train_target == "gpu":
                self._fit_input_scaler_gpu()
            else:
                self._fit_input_scaler_cpu()

            var t_dyn = perf_counter_ns()
            comptime if Self.train_target == "gpu":
                self._train_dynamics_ensemble_gpu()
            else:
                self._train_dynamics_ensemble()
            self.timer.accumulate(Self._T_DYN_TRAIN, t_dyn)

            var t_ro = perf_counter_ns()
            comptime if Self.train_target == "gpu":
                self._generate_synthetic_rollouts_gpu()
            else:
                self._generate_synthetic_rollouts()
            self.timer.accumulate(Self._T_ROLLOUT, t_ro)
            self.last_dyn_step = step_idx

        if self.sample_blk.real_count[Self.train_target]() < Self.REAL_BS:
            return False
        if self.sample_blk.synth_count[Self.train_target]() < Self.SYNTH_BS:
            return False

        comptime if Self.train_target == "cpu":
            return self._run_sac_updates[NoAMP](step_idx)
        else:
            if self._use_bf16:
                return self._run_sac_updates[Bf16Compute](step_idx)
            return self._run_sac_updates[NoAMP](step_idx)

    def _run_sac_updates[
        POLICY: AMPPolicy = NoAMP,
    ](mut self, step_idx: Int) raises -> Bool:
        """The `sac_updates_per_step` inner SAC mini-updates against the
        mixed real+synth buffer. `POLICY` (NoAMP / Bf16Compute) threads
        through the SAC sub-blocks; the dynamics ensemble + rollout phase
        stay fp32 (they ran in the caller before the readiness gate)."""
        var any = False
        for _ in range(self.sac_updates_per_step):
            self.state.step_idx = step_idx
            self.state.did_step = True
            comptime if Self.train_target == "cpu":
                self.state.alpha = fexp(self.alpha_opt.value)
            else:
                self.state.ctx = self.ctx

            var t_sample = perf_counter_ns()
            self.sample_blk.step[Self.train_target](self.state)
            if not self.state.did_step:
                continue
            self.timer.accumulate(Self._T_SAMPLE, t_sample)

            var t_ty = perf_counter_ns()
            self.target_y_blk.step[Self.train_target, POLICY](
                self.state,
                self.actor,
                self.pair1.target_net,
                self.pair2.target_net,
            )
            self.timer.accumulate(Self._T_TARGET_Y, t_ty)

            var t_crit = perf_counter_ns()
            self.twin_critic_blk.step[
                Self.train_target, POLICY,
                ACCUMULATE = Self.train_target == "gpu",
            ](
                self.state,
                self.pair1.online,
                self.critic1_opt,
                self.pair2.online,
                self.critic2_opt,
            )
            self.timer.accumulate(Self._T_CRITIC, t_crit)

            var t_act = perf_counter_ns()
            self.actor_blk.step[Self.train_target, POLICY](
                self.state,
                self.actor,
                self.actor_opt,
                self.pair1.online,
                self.pair2.online,
            )
            self.timer.accumulate(Self._T_ACTOR, t_act)

            var t_alp = perf_counter_ns()
            comptime if Self.train_target == "cpu":
                self.alpha_blk.step["cpu"](self.state, self.alpha_opt)
            else:
                self.alpha_blk.step["gpu"](
                    self.state,
                    self.alpha_opt,
                    self.actor_blk.lp_mean_dev_ptr(),
                    self.ctx,
                )
            self.timer.accumulate(Self._T_ALPHA, t_alp)

            var t_pol = perf_counter_ns()
            self.polyak_blk.step[Self.train_target](
                self.state,
                self.pair1,
                self.pair2,
            )
            self.timer.accumulate(Self._T_POLYAK, t_pol)

            # Per-batch diagnostics — mean Q1(s, a) + mean reward. CPU sums the
            # host scratches; GPU folds the same `[BATCH]` device buffers into
            # device-resident running means (no per-step D2H) read at flush.
            var t_diag = perf_counter_ns()
            comptime if Self.train_target == "cpu":
                var inv_b: Scalar[DT] = Scalar[DT](1.0) / Scalar[DT](Self.BATCH)
                var q_p = self.twin_critic_blk.inner.c1._mb_q.target_ptr["cpu"]()
                var r_p = self.state.mb_r.target_ptr["cpu"]()
                var y_p = self.state.mb_y.target_ptr["cpu"]()
                var d_p = self.state.mb_d.target_ptr["cpu"]()
                var a_p = self.state.mb_a.target_ptr["cpu"]()
                var sum_q: Scalar[DT] = 0.0
                var sum_r: Scalar[DT] = 0.0
                var sum_y: Scalar[DT] = 0.0
                var sum_d: Scalar[DT] = 0.0
                for i in range(Self.BATCH):
                    sum_q += q_p[i]
                    sum_r += r_p[i]
                    sum_y += y_p[i]
                    sum_d += d_p[i]
                var sum_a: Scalar[DT] = 0.0
                for i in range(Self.BATCH * Self.ACT_DIM):
                    sum_a += abs(a_p[i])
                self._q_accum += sum_q * inv_b
                self._reward_accum += sum_r * inv_b
                self._td_accum += sum_y * inv_b
                self._done_accum += sum_d * inv_b
                self._action_abs_accum += sum_a / Scalar[DT](
                    Self.BATCH * Self.ACT_DIM
                )
            else:
                var q_ptr = self.twin_critic_blk.inner.c1._mb_q.target_ptr[
                    "gpu"
                ]()
                var r_ptr = self.state.mb_r.target_ptr["gpu"]()
                var y_ptr = self.state.mb_y.target_ptr["gpu"]()
                var d_ptr = self.state.mb_d.target_ptr["gpu"]()
                var a_ptr = self.state.mb_a.target_ptr["gpu"]()
                self._q_mean_dev.accumulate_gpu[Self.BATCH](q_ptr)
                self._reward_mean_dev.accumulate_gpu[Self.BATCH](r_ptr)
                self._td_mean_dev.accumulate_gpu[Self.BATCH](y_ptr)
                self._done_mean_dev.accumulate_gpu[Self.BATCH](d_ptr)
                self._action_abs_mean_dev.accumulate_gpu_abs[
                    Self.BATCH * Self.ACT_DIM
                ](a_ptr)
            self.timer.accumulate(Self._T_DIAG, t_diag)

            self._actor_L_accum += self.state.actor_loss
            self._critic_L_accum += self.state.critic_loss
            self._update_count += 1
            self._total_train_steps += 1
            any = True
        return any

    # ─── Logging surface (parity with SACTrainer) ────────────────────────

    def flush_train_log(
        mut self,
    ) -> Tuple[Scalar[DT], Scalar[DT], Scalar[DT], Int]:
        var n = self._update_count if self._update_count > 0 else 1
        var inv = Scalar[DT](1.0) / Scalar[DT](n)
        var alpha_now = fexp(self.alpha_opt.value)
        var out = (
            self._actor_L_accum * inv,
            self._critic_L_accum * inv,
            alpha_now,
            self._update_count,
        )
        self._actor_L_accum = Scalar[DT](0.0)
        self._critic_L_accum = Scalar[DT](0.0)
        self._update_count = 0
        self._q_accum = Scalar[DT](0.0)
        self._reward_accum = Scalar[DT](0.0)
        self._td_accum = Scalar[DT](0.0)
        self._done_accum = Scalar[DT](0.0)
        self._action_abs_accum = Scalar[DT](0.0)
        self._dyn_loss_accum = Scalar[DT](0.0)
        self._dyn_step_count = 0
        return out

    def total_train_steps(self) -> Int:
        return self._total_train_steps

    def flush_metrics[
        L: Logger = NoOpLogger
    ](
        mut self,
        logger: Optional[UnsafePointer[L, MutAnyOrigin]] = None,
        step: Int = 0,
    ) raises -> MBPOMetrics:
        """Drain accumulators into an MBPOMetrics bundle. On GPU the SAC
        actor/critic losses + alpha are read from the on-device accumulators;
        the dynamics NLL is host-accumulated on both targets (D2H'd per
        member-step on the periodic cadence). `mean_q` / `mean_reward` are
        device-resident on GPU (Q1 + reward reductions folded in by
        `_run_sac_updates`)."""
        var n = self._update_count if self._update_count > 0 else 1
        var inv = Scalar[DT](1.0) / Scalar[DT](n)
        # Refresh the held dynamics-NLL ONLY when this flush window actually
        # contained dyn member-steps; otherwise keep the last real value so
        # the curve is piecewise-constant between model-train rounds instead
        # of dropping to 0 (the diag cadence is far finer than the dyn-train
        # cadence — see `_dyn_loss_last`).
        if self._dyn_step_count > 0:
            self._dyn_loss_last = (
                self._dyn_loss_accum / Scalar[DT](self._dyn_step_count)
            )
        var actor_mean: Scalar[DT]
        var critic_mean: Scalar[DT]
        var alpha_val: Scalar[DT]
        var q_mean: Scalar[DT]
        var reward_mean: Scalar[DT]
        var td_mean: Scalar[DT]
        var done_mean: Scalar[DT]
        var act_abs_mean: Scalar[DT]
        comptime if Self.train_target == "gpu":
            actor_mean = self.actor_blk.read_loss_accum()
            var cl1 = self.twin_critic_blk.inner.c1.mse_loss.read_accum["gpu"]()
            var cl2 = self.twin_critic_blk.inner.c2.mse_loss.read_accum["gpu"]()
            critic_mean = cl1 + cl2
            alpha_val = self.alpha_opt.read_alpha()
            q_mean = self._q_mean_dev.read["gpu"]()
            reward_mean = self._reward_mean_dev.read["gpu"]()
            td_mean = self._td_mean_dev.read["gpu"]()
            done_mean = self._done_mean_dev.read["gpu"]()
            act_abs_mean = self._action_abs_mean_dev.read["gpu"]()
        else:
            actor_mean = self._actor_L_accum * inv
            critic_mean = self._critic_L_accum * inv
            alpha_val = fexp(self.alpha_opt.value)
            q_mean = self._q_accum * inv
            reward_mean = self._reward_accum * inv
            td_mean = self._td_accum * inv
            done_mean = self._done_accum * inv
            act_abs_mean = self._action_abs_accum * inv
        var bundle = MBPOMetrics(
            actor_loss=LogScalar[DT](actor_mean),
            critic_loss=LogScalar[DT](critic_mean),
            alpha=LogScalar[DT](alpha_val),
            mean_q=LogScalar[DT](q_mean),
            mean_reward=LogScalar[DT](reward_mean),
            td_target=LogScalar[DT](td_mean),
            done_ratio=LogScalar[DT](done_mean),
            mean_abs_action=LogScalar[DT](act_abs_mean),
            dyn_loss=LogScalar[DT](self._dyn_loss_last),
            dyn_holdout_loss=LogScalar[DT](self._dyn_holdout_loss),
            dyn_holdout_min=LogScalar[DT](self._dyn_holdout_min),
            dyn_holdout_max=LogScalar[DT](self._dyn_holdout_max),
            dyn_holdout_spread=LogScalar[DT](
                self._dyn_holdout_max - self._dyn_holdout_min
            ),
            dyn_input_std_mean=LogScalar[DT](self._dyn_input_std_mean),
            train_steps=LogScalar[DT](Scalar[DT](self._total_train_steps)),
            n_updates=LogScalar[DT](Scalar[DT](self._update_count)),
        )
        self._actor_L_accum = Scalar[DT](0.0)
        self._critic_L_accum = Scalar[DT](0.0)
        self._update_count = 0
        self._q_accum = Scalar[DT](0.0)
        self._reward_accum = Scalar[DT](0.0)
        self._td_accum = Scalar[DT](0.0)
        self._done_accum = Scalar[DT](0.0)
        self._action_abs_accum = Scalar[DT](0.0)
        self._dyn_loss_accum = Scalar[DT](0.0)
        self._dyn_step_count = 0
        comptime if Self.train_target == "gpu":
            self.twin_critic_blk.inner.c1.mse_loss.reset_accum["gpu"]()
            self.twin_critic_blk.inner.c2.mse_loss.reset_accum["gpu"]()
            self.actor_blk.reset_loss_accum()
            self._q_mean_dev.reset["gpu"]()
            self._reward_mean_dev.reset["gpu"]()
            self._td_mean_dev.reset["gpu"]()
            self._done_mean_dev.reset["gpu"]()
            self._action_abs_mean_dev.reset["gpu"]()
        if Bool(logger):
            log_bundle(logger.value()[], bundle, step)
        return bundle^

    def flush_metrics_through_logger[L: Logger](
        mut self,
        logger: Optional[UnsafePointer[L, MutAnyOrigin]],
        step: Int,
    ) raises:
        # Key the SAC/dynamics diagnostics (mean_q, critic_loss, alpha,
        # dyn_loss, train_steps, ...) on the cumulative SAC train-step count
        # rather than the driver's env-step `step`. At UTD=40 these differ by
        # ~40x; the legacy MBPO run logged the same diagnostics against
        # train-steps, so this makes the dashboards overlay directly (and
        # removes the "looks like UTD=1" artifact — the diag panels were
        # capped at the env-step horizon while the updates ran 40x faster).
        # Episode reward / episode count stay on the env-step axis; the driver
        # emits those separately, matching legacy.
        _ = step
        _ = self.flush_metrics[L](logger, self._total_train_steps)

    def save_state(mut self, path: String) raises:
        """One-file v2 checkpoint of the full MBPO trainer state: the SAC
        modules + optimizers AND the dynamics ensemble (every member net +
        its Adam moments), the elite-member indices, and the rollout
        length. So a resumed run continues from the learned world model
        instead of re-training it from scratch. The on-disk format is
        byte-identical CPU vs GPU (device state synced to host on save).
        NOT saved: replay buffers + episode tracker (matches every other
        trainer)."""
        var body = String("")
        comptime if Self.train_target == "gpu":
            var c = self.ctx.value()
            save_state_v2_body_gpu(self.actor, body, "actor", c)
            save_state_v2_body_gpu(self.pair1.online, body, "critic1", c)
            save_state_v2_body_gpu(self.pair2.online, body, "critic2", c)
            save_optimizer_v2_body_gpu(self.actor_opt, body, "actor_opt")
            save_optimizer_v2_body_gpu(self.critic1_opt, body, "critic1_opt")
            save_optimizer_v2_body_gpu(self.critic2_opt, body, "critic2_opt")
            save_scalar_adam_v2_body_gpu(self.alpha_opt, body, "alpha_opt")
            for i in range(Self.N_ENSEMBLE):
                save_state_v2_body_gpu(
                    self.ensemble.members[i], body, "dyn_member" + String(i), c
                )
                save_optimizer_v2_body_gpu(
                    self.ensemble.opts[i], body, "dyn_opt" + String(i)
                )
        else:
            save_state_v2_body(self.actor, body, "actor")
            save_state_v2_body(self.pair1.online, body, "critic1")
            save_state_v2_body(self.pair2.online, body, "critic2")
            save_optimizer_v2_body(self.actor_opt, body, "actor_opt")
            save_optimizer_v2_body(self.critic1_opt, body, "critic1_opt")
            save_optimizer_v2_body(self.critic2_opt, body, "critic2_opt")
            save_scalar_adam_v2_body(self.alpha_opt, body, "alpha_opt")
            for i in range(Self.N_ENSEMBLE):
                save_state_v2_body(
                    self.ensemble.members[i], body, "dyn_member" + String(i)
                )
                save_optimizer_v2_body(
                    self.ensemble.opts[i], body, "dyn_opt" + String(i)
                )
        # Elite indices + rollout length: host ints, identical both targets.
        SaveI(len(self.ensemble.elite_indices)).save(body, "dyn_n_elites")
        for i in range(len(self.ensemble.elite_indices)):
            SaveI(self.ensemble.elite_indices[i]).save(
                body, "dyn_elite" + String(i)
            )
        SaveI(self.rollout_length).save(body, "dyn_rollout_length")
        save_counter_v2_body(self._total_train_steps, body, "_total_train_steps")
        var content = String("nn-ckpt v2\n") + body
        with open(path, "w") as f:
            f.write(content)

    def load_state(mut self, path: String) raises:
        var content = read_file_v2(path)
        var lines = split_lines_v2(content)
        expect_v2_header(lines)
        var idx: Int = 1
        comptime if Self.train_target == "gpu":
            var c = self.ctx.value()
            load_state_v2_body_gpu(self.actor, lines, idx, "actor", c)
            load_state_v2_body_gpu(self.pair1.online, lines, idx, "critic1", c)
            load_state_v2_body_gpu(self.pair2.online, lines, idx, "critic2", c)
            load_optimizer_v2_body_gpu(self.actor_opt, lines, idx, "actor_opt")
            load_optimizer_v2_body_gpu(self.critic1_opt, lines, idx, "critic1_opt")
            load_optimizer_v2_body_gpu(self.critic2_opt, lines, idx, "critic2_opt")
            load_scalar_adam_v2_body_gpu(self.alpha_opt, lines, idx, "alpha_opt")
            for i in range(Self.N_ENSEMBLE):
                load_state_v2_body_gpu(
                    self.ensemble.members[i], lines, idx,
                    "dyn_member" + String(i), c,
                )
                load_optimizer_v2_body_gpu(
                    self.ensemble.opts[i], lines, idx, "dyn_opt" + String(i)
                )
            hard_copy_params["gpu", M=Self.CRITIC](
                self.pair1.online, self.pair1.target_net, self.ctx,
            )
            hard_copy_params["gpu", M=Self.CRITIC](
                self.pair2.online, self.pair2.target_net, self.ctx,
            )
        else:
            load_state_v2_body(self.actor, lines, idx, "actor")
            load_state_v2_body(self.pair1.online, lines, idx, "critic1")
            load_state_v2_body(self.pair2.online, lines, idx, "critic2")
            load_optimizer_v2_body(self.actor_opt, lines, idx, "actor_opt")
            load_optimizer_v2_body(self.critic1_opt, lines, idx, "critic1_opt")
            load_optimizer_v2_body(self.critic2_opt, lines, idx, "critic2_opt")
            load_scalar_adam_v2_body(self.alpha_opt, lines, idx, "alpha_opt")
            for i in range(Self.N_ENSEMBLE):
                load_state_v2_body(
                    self.ensemble.members[i], lines, idx,
                    "dyn_member" + String(i),
                )
                load_optimizer_v2_body(
                    self.ensemble.opts[i], lines, idx, "dyn_opt" + String(i)
                )
            hard_copy_params["cpu", M=Self.CRITIC](
                self.pair1.online, self.pair1.target_net, None,
            )
            hard_copy_params["cpu", M=Self.CRITIC](
                self.pair2.online, self.pair2.target_net, None,
            )
        # Elite indices + rollout length (host ints, identical both targets).
        var n_elites_w = SaveI(0)
        n_elites_w.load(lines, idx, "dyn_n_elites")
        for i in range(n_elites_w.v):
            var elite_w = SaveI(0)
            elite_w.load(lines, idx, "dyn_elite" + String(i))
            self.ensemble.elite_indices[i] = elite_w.v
        var rl_w = SaveI(0)
        rl_w.load(lines, idx, "dyn_rollout_length")
        self.rollout_length = rl_w.v
        load_counter_v2_body(
            self._total_train_steps, lines, idx, "_total_train_steps"
        )

    def flush_timer_log(mut self) -> String:
        var report = self.timer.format_report()
        self.timer.reset()
        return report

    # ─── OffPolicyAgentGpu surface ────────────────────────────────────

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
                comptime warmup_kernel = _mbpo_warmup_uniform_kernel[N_ENVS, ACT]
                var ctx = self.ctx.value()
                ctx.enqueue_function[warmup_kernel](
                    action_lt, self.action_scale,
                    self._warmup_rng_seed, self._warmup_rng_offset,
                    grid_dim=n_blocks, block_dim=TPB,
                )
                self._warmup_rng_offset += UInt64(N_ENVS * ACT * 2)
            return

        var obs_t = TileTensor(obs_ptr, row_major[N_ENVS, OBS]())
        var ao_t = TileTensor(ao_scratch_ptr, row_major[N_ENVS, 2 * ACT]())
        var alp_t = TileTensor(alp_scratch_ptr, row_major[N_ENVS, ACT + 1]())
        self.actor.forward[Self.train_target, N_ENVS](obs_t, output=ao_t)
        self.actor_blk.inner.rsample.forward[Self.train_target, N_ENVS](
            ao_t, output=alp_t
        )
        comptime if Self.train_target == "cpu":
            for env in range(N_ENVS):
                var src = alp_scratch_ptr + env * (ACT + 1)
                var dst = action_ptr + env * ACT
                for j in range(ACT):
                    var a = src[j]
                    if a > self.action_scale:
                        a = self.action_scale
                    elif a < -self.action_scale:
                        a = -self.action_scale
                    dst[j] = a
        else:
            var alp_lt = LayoutTensor[
                DT, Layout.row_major(N_ENVS, ACT + 1), MutAnyOrigin,
            ](alp_scratch_ptr)
            var action_lt = LayoutTensor[
                DT, Layout.row_major(N_ENVS, ACT), MutAnyOrigin,
            ](action_ptr)
            comptime total = N_ENVS * ACT
            comptime n_blocks = (total + TPB - 1) // TPB
            comptime clamp_kernel = _mbpo_action_clamp_kernel[N_ENVS, ACT]
            var ctx = self.ctx.value()
            ctx.enqueue_function[clamp_kernel](
                alp_lt, action_lt, self.action_scale,
                grid_dim=n_blocks, block_dim=TPB,
            )

    def _replay_add(
        mut self,
        ref obs: List[Scalar[DT]],
        ref action: List[Scalar[DT]],
        reward: Scalar[DT],
        ref next_obs: List[Scalar[DT]],
        done: Scalar[DT],
    ) raises:
        # `record_batch_cpu`'s staging loop is the OffPolicyAgent trait
        # default (S6 follow-on); MBPO's hook routes to the DualSampleStep's
        # `real_add` (the REAL buffer only — synthetic comes from rollouts),
        # which the uniform `sample_blk.add` default can't name.
        self.sample_blk.real_add[Self.train_target](
            obs, action, reward, next_obs, done, ctx=self.ctx,
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
            "MBPOTrainer.record_batch_gpu: GPU-env batched record not"
            " supported (MBPO uses the cpu-env single-env driver path)"
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
            "MBPOTrainer.record_batch_gpu_nstep: not supported"
        )

    # ─── Dynamics input scaler (per-DYN_IN-dim z-score) ───────────────

    def _set_scaler_identity[target: StaticString](mut self) raises:
        """Reset the input scaler to identity (mean=0, std=1) so it is a
        no-op until the first fit. Called once at construction."""
        var mean_p = self._in_mean.cpu_ptr()
        var std_p = self._in_std.cpu_ptr()
        for c in range(Self.DYN_IN):
            mean_p[c] = Scalar[DT](0.0)
            std_p[c] = Scalar[DT](1.0)
        comptime if target == "gpu":
            var ctx = self.ctx.value()
            ctx.enqueue_copy(self._in_mean.dev.value(), mean_p)
            ctx.enqueue_copy(self._in_std.dev.value(), std_p)

    def _compute_scaler_host(
        mut self,
        obs_p: UnsafePointer[Scalar[DT], MutAnyOrigin],
        act_p: UnsafePointer[Scalar[DT], MutAnyOrigin],
        n_data: Int,
    ):
        """Fit per-DYN_IN-dim mean/std from `n_data` real transitions laid
        out as `obs_p[i*OBS + d]` / `act_p[i*ACT + j]`. Writes the result
        into the host mirror of `_in_mean` / `_in_std`. A near-zero std is
        floored to 1.0 so (near-)constant dims pass through unscaled
        instead of exploding."""
        var mean_p = self._in_mean.cpu_ptr()
        var std_p = self._in_std.cpu_ptr()
        for c in range(Self.DYN_IN):
            mean_p[c] = Scalar[DT](0.0)
            std_p[c] = Scalar[DT](0.0)
        var inv_n = Scalar[DT](1.0) / Scalar[DT](n_data)
        for i in range(n_data):
            for d in range(Self.OBS_DIM):
                mean_p[d] += obs_p[i * Self.OBS_DIM + d]
            for j in range(Self.ACT_DIM):
                mean_p[Self.OBS_DIM + j] += act_p[i * Self.ACT_DIM + j]
        for c in range(Self.DYN_IN):
            mean_p[c] *= inv_n
        for i in range(n_data):
            for d in range(Self.OBS_DIM):
                var diff = obs_p[i * Self.OBS_DIM + d] - mean_p[d]
                std_p[d] += diff * diff
            for j in range(Self.ACT_DIM):
                var diff = (
                    act_p[i * Self.ACT_DIM + j] - mean_p[Self.OBS_DIM + j]
                )
                std_p[Self.OBS_DIM + j] += diff * diff
        var std_sum = Scalar[DT](0.0)
        for c in range(Self.DYN_IN):
            var v = fsqrt(std_p[c] * inv_n)
            if v < Scalar[DT](1e-12):
                v = Scalar[DT](1.0)
            std_p[c] = v
            std_sum += v
        # Diagnostic: mean input-scaler std (legacy `dyn_input_std_mean`).
        self._dyn_input_std_mean = std_sum / Scalar[DT](Self.DYN_IN)

    def _record_holdout_stats(mut self, ref holdout: List[Scalar[DT]]):
        """Capture per-member holdout-NLL mean / min / max into the diag
        fields (held between model-train rounds). Called by both dyn-train
        paths right after they build the `holdout` list for `update_elites`."""
        if len(holdout) == 0:
            return
        var s = holdout[0]
        var mn = holdout[0]
        var mx = holdout[0]
        for i in range(1, len(holdout)):
            var v = holdout[i]
            s += v
            if v < mn:
                mn = v
            if v > mx:
                mx = v
        self._dyn_holdout_loss = s / Scalar[DT](len(holdout))
        self._dyn_holdout_min = mn
        self._dyn_holdout_max = mx

    def _fit_input_scaler_cpu(mut self):
        var n_data = self.sample_blk.real_count["cpu"]()
        if n_data < 2:
            return
        self._compute_scaler_host(
            self.sample_blk.real_cpu.value().obs,
            self.sample_blk.real_cpu.value().act,
            n_data,
        )

    def _fit_input_scaler_gpu(mut self) raises:
        """Fit the scaler on GPU: D2H the real-buffer obs/act, reuse the
        host arithmetic (bit-identical to the CPU path), then H2D the
        mean/std to device for the normalize kernel."""
        var n_data = self.sample_blk.real_count["gpu"]()
        if n_data < 2:
            return
        var ctx = self.ctx.value()
        comptime cap_obs = Self.REPLAY_CAPACITY * Self.OBS_DIM
        comptime cap_act = Self.REPLAY_CAPACITY * Self.ACT_DIM
        var host_obs = alloc[Scalar[DT]](cap_obs)
        var host_act = alloc[Scalar[DT]](cap_act)
        ctx.enqueue_copy(host_obs, self.sample_blk.real_gpu.value().obs)
        ctx.enqueue_copy(host_act, self.sample_blk.real_gpu.value().act)
        ctx.synchronize()
        self._compute_scaler_host(host_obs, host_act, n_data)
        ctx.enqueue_copy(self._in_mean.dev.value(), self._in_mean.cpu_ptr())
        ctx.enqueue_copy(self._in_std.dev.value(), self._in_std.cpu_ptr())
        ctx.synchronize()
        host_obs.free()
        host_act.free()

    def _normalize_dyn_in_cpu(mut self):
        """In-place z-score the BATCH×DYN_IN host `dyn_in` scratch."""
        var dyn_in_p = self._dyn_in.cpu_ptr()
        var mean_p = self._in_mean.cpu_ptr()
        var std_p = self._in_std.cpu_ptr()
        for k in range(Self.BATCH):
            var base = k * Self.DYN_IN
            for c in range(Self.DYN_IN):
                dyn_in_p[base + c] = (
                    dyn_in_p[base + c] - mean_p[c]
                ) / std_p[c]

    def _normalize_dyn_in_gpu(mut self) raises:
        """In-place z-score the BATCH×DYN_IN device `dyn_in` scratch."""
        var ctx = self.ctx.value()
        var data_lt = LayoutTensor[
            DT, Layout.row_major(Self.BATCH, Self.DYN_IN), MutAnyOrigin,
        ](self._dyn_in.dev_ptr())
        var mean_lt = LayoutTensor[
            DT, Layout.row_major(Self.DYN_IN), MutAnyOrigin,
        ](self._in_mean.dev_ptr())
        var std_lt = LayoutTensor[
            DT, Layout.row_major(Self.DYN_IN), MutAnyOrigin,
        ](self._in_std.dev_ptr())
        comptime total = Self.BATCH * Self.DYN_IN
        comptime n_blocks = (total + TPB - 1) // TPB
        comptime norm_kernel = _normalize_input_kernel[Self.BATCH, Self.DYN_IN]
        ctx.enqueue_function[norm_kernel](
            data_lt, mean_lt, std_lt,
            grid_dim=n_blocks, block_dim=TPB,
        )

    # ─── Dynamics training + synthetic rollouts (CPU) ─────────────────

    def _fill_dyn_batch_cpu(mut self, k: Int, idx: Int):
        """Fill row `k` of the host dyn_in/dyn_tgt scratch from real-buffer
        transition `idx`: dyn_in = [obs, act] (raw, whitened by caller),
        dyn_tgt = [reward, Δobs]."""
        var dyn_in_p = self._dyn_in.cpu_ptr()
        var dyn_tgt_p = self._dyn_tgt.cpu_ptr()
        var rb_obs = self.sample_blk.real_cpu.value().obs
        var rb_act = self.sample_blk.real_cpu.value().act
        var rb_rew = self.sample_blk.real_cpu.value().rew
        var rb_nxt = self.sample_blk.real_cpu.value().nxt
        for d in range(Self.OBS_DIM):
            dyn_in_p[k * Self.DYN_IN + d] = rb_obs[idx * Self.OBS_DIM + d]
        for j in range(Self.ACT_DIM):
            dyn_in_p[k * Self.DYN_IN + Self.OBS_DIM + j] = rb_act[
                idx * Self.ACT_DIM + j
            ]
        dyn_tgt_p[k * Self.DYN_PRED + 0] = rb_rew[idx]
        for d in range(Self.OBS_DIM):
            dyn_tgt_p[k * Self.DYN_PRED + 1 + d] = (
                rb_nxt[idx * Self.OBS_DIM + d] - rb_obs[idx * Self.OBS_DIM + d]
            )

    def _eval_member_holdout_cpu(
        mut self, m: Int, n_train: Int, n_holdout: Int,
    ) raises -> Scalar[DT]:
        """Mean GaussianNLL of member `m` over the held-out slice
        `[n_train, n_train + n_holdout)` of the real buffer, evaluated in
        BATCH chunks (whitened inputs, raw targets — training convention).
        This is a TRUE holdout (never trained on), so it detects overfit."""
        var dyn_in_p = self._dyn_in.cpu_ptr()
        var dyn_tgt_p = self._dyn_tgt.cpu_ptr()
        var n_chunks = n_holdout // Self.BATCH
        if n_chunks < 1:
            n_chunks = 1
        var total = Scalar[DT](0.0)
        for c in range(n_chunks):
            for k in range(Self.BATCH):
                var idx = n_train + ((c * Self.BATCH + k) % n_holdout)
                self._fill_dyn_batch_cpu(k, idx)
            self._normalize_dyn_in_cpu()
            var in_t = TileTensor(dyn_in_p, row_major[Self.BATCH, Self.DYN_IN]())
            var tgt_t = TileTensor(
                dyn_tgt_p, row_major[Self.BATCH, Self.DYN_PRED]()
            )
            total += self.ensemble.eval_member_loss["cpu"](m, in_t, tgt_t)
        return total / Scalar[DT](n_chunks)

    def _train_dynamics_ensemble(mut self) raises:
        var n_data = self.sample_blk.real_count["cpu"]()
        if n_data < self.dyn_batch_size:
            return
        # Fixed train/holdout split (legacy `holdout_ratio`): train each
        # member on the first `n_train`, validate + early-stop + rank elites
        # on the last `n_holdout`. A TRUE holdout (never trained on) is what
        # lets early-stopping prevent the overfit→over-confident→optimistic
        # failure mode (the old code trained on the whole buffer for a fixed
        # `dyn_epochs_per_round` and "validated" on data resampled from it).
        var n_holdout = Int(Scalar[DT](n_data) * self.dyn_holdout_ratio)
        if n_holdout < 1:
            n_holdout = 1
        var n_train = n_data - n_holdout
        if n_train < self.dyn_batch_size:
            # Buffer too small to split (only the first round or two) — train
            # on all, eval on all. Harmless; the split kicks in once n_data
            # grows past `dyn_batch_size / (1 - holdout_ratio)`.
            n_train = n_data
            n_holdout = n_data
        var bs = self.dyn_batch_size
        var steps_per_epoch = n_train // bs
        if steps_per_epoch < 1:
            steps_per_epoch = 1
        var dyn_in_p = self._dyn_in.cpu_ptr()
        var dyn_tgt_p = self._dyn_tgt.cpu_ptr()
        var n_checks = self.dyn_max_epochs // self.dyn_holdout_check_every
        if n_checks < 1:
            n_checks = 1

        for m in range(Self.N_ENSEMBLE):
            var best = Scalar[DT](1e30)
            var since = 0
            var stop = False
            for _check in range(n_checks):
                if stop:
                    break
                for _ep in range(self.dyn_holdout_check_every):
                    for _ in range(steps_per_epoch):
                        for k in range(Self.BATCH):
                            var idx = Int(random_float64() * Float64(n_train))
                            if idx >= n_train:
                                idx = n_train - 1
                            self._fill_dyn_batch_cpu(k, idx)
                        self._normalize_dyn_in_cpu()
                        var in_t = TileTensor(
                            dyn_in_p, row_major[Self.BATCH, Self.DYN_IN]()
                        )
                        var tgt_t = TileTensor(
                            dyn_tgt_p, row_major[Self.BATCH, Self.DYN_PRED]()
                        )
                        var dyn_loss = self.ensemble.train_member_step["cpu"](
                            m, in_t, tgt_t,
                        )
                        self._dyn_loss_accum += dyn_loss
                        self._dyn_step_count += 1
                # Early-stop check on the TRUE holdout.
                var hl = self._eval_member_holdout_cpu(m, n_train, n_holdout)
                if hl < best - Scalar[DT](1e-2):
                    best = hl
                    since = 0
                else:
                    since += 1
                    if since >= self.dyn_patience:
                        stop = True

        # Elite ranking on the same held-out set (now a TRUE holdout).
        var holdout = List[Scalar[DT]]()
        for m in range(Self.N_ENSEMBLE):
            holdout.append(self._eval_member_holdout_cpu(m, n_train, n_holdout))
        self._record_holdout_stats(holdout)
        self.ensemble.update_elites(holdout)

    def _generate_synthetic_rollouts(mut self) raises:
        var real_buf_size = self.sample_blk.real_count["cpu"]()
        if real_buf_size < 1:
            return

        var rollouts_done = 0
        while rollouts_done < self.num_rollouts_per_step:
            var this_batch = Self.BATCH
            var remaining = self.num_rollouts_per_step - rollouts_done
            if remaining < this_batch:
                this_batch = remaining

            var roll_obs_p = self._ro_obs.cpu_ptr()
            var roll_act_p = self._ro_act.cpu_ptr()
            var roll_nxt_p = self._ro_nxt.cpu_ptr()
            var dyn_in_p = self._dyn_in.cpu_ptr()

            var rb_obs = self.sample_blk.real_cpu.value().obs
            for k in range(this_batch):
                var idx = Int(random_float64() * Float64(real_buf_size))
                if idx >= real_buf_size:
                    idx = real_buf_size - 1
                for d in range(Self.OBS_DIM):
                    roll_obs_p[k * Self.OBS_DIM + d] = rb_obs[
                        idx * Self.OBS_DIM + d
                    ]

            for _ in range(self.rollout_length):
                for k in range(this_batch):
                    var obs_list = List[Scalar[DT]](capacity=Self.OBS_DIM)
                    for d in range(Self.OBS_DIM):
                        obs_list.append(roll_obs_p[k * Self.OBS_DIM + d])
                    var act_list = List[Scalar[DT]](capacity=Self.ACT_DIM)
                    for _ in range(Self.ACT_DIM):
                        act_list.append(Scalar[DT](0.0))
                    self._select_action_cpu(
                        obs_list, act_list, self.learning_starts + 1,
                    )
                    for j in range(Self.ACT_DIM):
                        roll_act_p[k * Self.ACT_DIM + j] = act_list[j]

                for k in range(this_batch):
                    for d in range(Self.OBS_DIM):
                        dyn_in_p[k * Self.DYN_IN + d] = roll_obs_p[
                            k * Self.OBS_DIM + d
                        ]
                    for j in range(Self.ACT_DIM):
                        dyn_in_p[
                            k * Self.DYN_IN + Self.OBS_DIM + j
                        ] = roll_act_p[k * Self.ACT_DIM + j]
                # Whiten inputs to match the dynamics-train convention.
                self._normalize_dyn_in_cpu()
                var dyn_in_t = TileTensor(
                    dyn_in_p, row_major[Self.BATCH, Self.DYN_IN]()
                )
                # Per-transition elite mixing (CPU): forward ALL elites into
                # stacked host buffers; the posterior loop below picks a random
                # elite PER transition. (One elite per chunk lets the policy
                # exploit a single model's bias → optimistic synthetic reward.)
                var n_elites = len(self.ensemble.elite_indices)
                var mu_all_p = self._ro_mu_all.cpu_ptr()
                var lv_all_p = self._ro_lv_all.cpu_ptr()
                for e in range(n_elites):
                    var off = e * Self.BATCH * Self.DYN_PRED
                    var mu_e_t = TileTensor(
                        mu_all_p + off, row_major[Self.BATCH, Self.DYN_PRED]()
                    )
                    var lv_e_t = TileTensor(
                        lv_all_p + off, row_major[Self.BATCH, Self.DYN_PRED]()
                    )
                    self.ensemble.predict_member["cpu"](
                        self.ensemble.elite_indices[e], dyn_in_t, mu_e_t, lv_e_t,
                    )

                var s_list = List[Scalar[DT]](capacity=Self.OBS_DIM)
                var a_list = List[Scalar[DT]](capacity=Self.ACT_DIM)
                var sp_list = List[Scalar[DT]](capacity=Self.OBS_DIM)
                for _ in range(Self.OBS_DIM):
                    s_list.append(Scalar[DT](0.0))
                    sp_list.append(Scalar[DT](0.0))
                for _ in range(Self.ACT_DIM):
                    a_list.append(Scalar[DT](0.0))
                for k in range(this_batch):
                    var e = Int(random_float64() * Float64(n_elites))
                    if e >= n_elites:
                        e = n_elites - 1
                    var base = (
                        e * Self.BATCH * Self.DYN_PRED + k * Self.DYN_PRED
                    )
                    var mu_r = mu_all_p[base + 0]
                    var lv_r = lv_all_p[base + 0]
                    var std_r = fsqrt(fexp(lv_r))
                    var noise_r = Scalar[DT](randn_float64())
                    var rew = mu_r + std_r * noise_r
                    # Legacy parity: clamp synth reward to [-100, 100], NaN→0.
                    if rew != rew:
                        rew = Scalar[DT](0.0)
                    elif rew < Scalar[DT](-100.0):
                        rew = Scalar[DT](-100.0)
                    elif rew > Scalar[DT](100.0):
                        rew = Scalar[DT](100.0)
                    for d in range(Self.OBS_DIM):
                        s_list[d] = roll_obs_p[k * Self.OBS_DIM + d]
                        var mu_d = mu_all_p[base + 1 + d]
                        var lv_d = lv_all_p[base + 1 + d]
                        var std_d = fsqrt(fexp(lv_d))
                        var noise = Scalar[DT](randn_float64())
                        var delta = mu_d + std_d * noise
                        var nxt = roll_obs_p[k * Self.OBS_DIM + d] + delta
                        sp_list[d] = nxt
                        roll_nxt_p[k * Self.OBS_DIM + d] = nxt
                    for j in range(Self.ACT_DIM):
                        a_list[j] = roll_act_p[k * Self.ACT_DIM + j]
                    self.sample_blk.synth_add(
                        s_list, a_list, rew, sp_list, Scalar[DT](0.0),
                    )

                for k in range(this_batch * Self.OBS_DIM):
                    roll_obs_p[k] = roll_nxt_p[k]

            rollouts_done += this_batch

    # ─── Dynamics training + synthetic rollouts (GPU) ─────────────────

    def _build_dyn_batch_gpu(mut self, lo: Int, hi: Int) raises:
        """Sample BATCH transitions from real-buffer index range [lo, hi)
        into the rollout scratches, then materialize the device tiles
        `self._dyn_in` (whitened [obs, act]) + `self._dyn_tgt` ([reward,
        Δobs]) ready for `train_member_step` / `eval_member_loss`."""
        var ctx = self.ctx.value()
        comptime n_blocks = (Self.BATCH + TPB - 1) // TPB
        comptime tgt_kernel = _build_dyn_target_kernel[
            Self.BATCH, Self.OBS_DIM, Self.DYN_PRED
        ]
        self.sample_blk.real_sample_range[Self.BATCH](
            ctx, lo, hi,
            self._ro_obs.dev.value(),
            self._ro_act.dev.value(),
            self._ro_rew.dev.value(),
            self._ro_nxt.dev.value(),
            self._ro_done.dev.value(),
        )
        concat_sa_gpu[Self.OBS_DIM, Self.ACT_DIM, Self.BATCH](
            ctx,
            self._ro_obs.dev_ptr(),
            self._ro_act.dev_ptr(),
            self._dyn_in.dev_ptr(),
        )
        self._normalize_dyn_in_gpu()
        var rew_lt = LayoutTensor[
            DT, Layout.row_major(Self.BATCH), MutAnyOrigin,
        ](self._ro_rew.dev_ptr())
        var s_lt = LayoutTensor[
            DT, Layout.row_major(Self.BATCH, Self.OBS_DIM), MutAnyOrigin,
        ](self._ro_obs.dev_ptr())
        var sp_lt = LayoutTensor[
            DT, Layout.row_major(Self.BATCH, Self.OBS_DIM), MutAnyOrigin,
        ](self._ro_nxt.dev_ptr())
        var tgt_lt = LayoutTensor[
            DT, Layout.row_major(Self.BATCH, Self.DYN_PRED), MutAnyOrigin,
        ](self._dyn_tgt.dev_ptr())
        ctx.enqueue_function[tgt_kernel](
            rew_lt, s_lt, sp_lt, tgt_lt,
            grid_dim=n_blocks, block_dim=TPB,
        )

    def _eval_member_holdout_gpu(
        mut self, m: Int, lo: Int, hi: Int,
    ) raises -> Scalar[DT]:
        """Mean GaussianNLL of member `m` over the holdout range [lo, hi),
        averaged across up to 4 BATCH draws (a Monte-Carlo estimate of the
        held-out NLL — cheaper than a full reduction; averaging stabilizes
        the early-stop signal)."""
        var span = hi - lo
        var n_chunks = span // Self.BATCH
        if n_chunks < 1:
            n_chunks = 1
        if n_chunks > 4:
            n_chunks = 4
        var total = Scalar[DT](0.0)
        for _ in range(n_chunks):
            self._build_dyn_batch_gpu(lo, hi)
            var in_t = TileTensor(
                self._dyn_in.dev_ptr(), row_major[Self.BATCH, Self.DYN_IN]()
            )
            var tgt_t = TileTensor(
                self._dyn_tgt.dev_ptr(), row_major[Self.BATCH, Self.DYN_PRED]()
            )
            total += self.ensemble.eval_member_loss["gpu"](m, in_t, tgt_t)
        return total / Scalar[DT](n_chunks)

    def _train_dynamics_ensemble_gpu(mut self) raises:
        var n_data = self.sample_blk.real_count["gpu"]()
        if n_data < self.dyn_batch_size:
            return
        # Fixed train/holdout split + per-member early stopping on the TRUE
        # holdout (mirrors the CPU path + legacy `train_on_buffer`). Train on
        # [0, n_train); validate + early-stop + rank elites on [hold_lo,
        # hold_hi). Prevents the overfit→over-confident→optimistic world model
        # the policy was exploiting.
        var n_holdout = Int(Scalar[DT](n_data) * self.dyn_holdout_ratio)
        if n_holdout < 1:
            n_holdout = 1
        var n_train = n_data - n_holdout
        var hold_lo = n_train
        var hold_hi = n_data
        if n_train < self.dyn_batch_size:
            # Buffer too small to split (first round or two) — train + eval
            # on all; the split engages once n_data grows.
            n_train = n_data
            hold_lo = 0
            hold_hi = n_data
        var bs = self.dyn_batch_size
        var steps_per_epoch = n_train // bs
        if steps_per_epoch < 1:
            steps_per_epoch = 1
        var n_checks = self.dyn_max_epochs // self.dyn_holdout_check_every
        if n_checks < 1:
            n_checks = 1

        for m in range(Self.N_ENSEMBLE):
            var best = Scalar[DT](1e30)
            var since = 0
            var stop = False
            for _check in range(n_checks):
                if stop:
                    break
                for _ep in range(self.dyn_holdout_check_every):
                    for _ in range(steps_per_epoch):
                        self._build_dyn_batch_gpu(0, n_train)
                        var dyn_in_t = TileTensor(
                            self._dyn_in.dev_ptr(),
                            row_major[Self.BATCH, Self.DYN_IN](),
                        )
                        var dyn_tgt_t = TileTensor(
                            self._dyn_tgt.dev_ptr(),
                            row_major[Self.BATCH, Self.DYN_PRED](),
                        )
                        var dyn_loss = self.ensemble.train_member_step["gpu"](
                            m, dyn_in_t, dyn_tgt_t,
                        )
                        self._dyn_loss_accum += dyn_loss
                        self._dyn_step_count += 1
                var hl = self._eval_member_holdout_gpu(m, hold_lo, hold_hi)
                if hl < best - Scalar[DT](1e-2):
                    best = hl
                    since = 0
                else:
                    since += 1
                    if since >= self.dyn_patience:
                        stop = True

        # Elite ranking on the TRUE holdout.
        var holdout = List[Scalar[DT]]()
        for m in range(Self.N_ENSEMBLE):
            holdout.append(self._eval_member_holdout_gpu(m, hold_lo, hold_hi))
        self._record_holdout_stats(holdout)
        self.ensemble.update_elites(holdout)

    def _generate_synthetic_rollouts_gpu(mut self) raises:
        if self.sample_blk.real_count["gpu"]() < 1:
            return
        var ctx = self.ctx.value()

        # Synthetic transitions never terminate.
        self._ro_done.dev.value().enqueue_fill(Scalar[DT](0.0))

        comptime n_lane_blocks = (Self.BATCH + TPB - 1) // TPB
        comptime clamp_kernel = _mbpo_action_clamp_kernel[Self.BATCH, Self.ACT_DIM]
        comptime post_kernel = _rollout_posterior_kernel[
            Self.BATCH, Self.OBS_DIM, Self.DYN_PRED
        ]
        comptime n_noise = Self.BATCH * Self.DYN_PRED

        # Process in fixed BATCH chunks; rounds num_rollouts up to a
        # multiple of BATCH (extra synthetic data is harmless).
        var chunks = (self.num_rollouts_per_step + Self.BATCH - 1) // Self.BATCH
        for _ in range(chunks):
            # Start states: draw BATCH transitions, keep s (= ro_obs); the
            # a/r/sp/d destinations are throwaway here.
            self.sample_blk.real_sample[Self.BATCH](
                ctx,
                self._ro_obs.dev.value(),
                self._ro_act.dev.value(),
                self._ro_rew.dev.value(),
                self._ro_nxt.dev.value(),
                self._ro_done.dev.value(),
            )
            self._ro_done.dev.value().enqueue_fill(Scalar[DT](0.0))

            for _ in range(self.rollout_length):
                # Policy action on imagined obs: actor → rsample → clamp.
                var obs_t = TileTensor(
                    self._ro_obs.dev_ptr(), row_major[Self.BATCH, Self.OBS_DIM]()
                )
                var ao_t = TileTensor(
                    self._ro_ao.dev_ptr(),
                    row_major[Self.BATCH, 2 * Self.ACT_DIM](),
                )
                self.actor.forward["gpu", Self.BATCH](obs_t, output=ao_t)
                var alp_t = TileTensor(
                    self._ro_alp.dev_ptr(),
                    row_major[Self.BATCH, Self.ACT_DIM + 1](),
                )
                self.actor_blk.inner.rsample.forward["gpu", Self.BATCH](
                    ao_t, output=alp_t
                )
                var alp_lt = LayoutTensor[
                    DT, Layout.row_major(Self.BATCH, Self.ACT_DIM + 1),
                    MutAnyOrigin,
                ](self._ro_alp.dev_ptr())
                var act_lt = LayoutTensor[
                    DT, Layout.row_major(Self.BATCH, Self.ACT_DIM), MutAnyOrigin,
                ](self._ro_act.dev_ptr())
                ctx.enqueue_function[clamp_kernel](
                    alp_lt, act_lt, self.action_scale,
                    grid_dim=n_lane_blocks, block_dim=TPB,
                )

                # dyn_in = concat(obs, action).
                concat_sa_gpu[Self.OBS_DIM, Self.ACT_DIM, Self.BATCH](
                    ctx,
                    self._ro_obs.dev_ptr(),
                    self._ro_act.dev_ptr(),
                    self._dyn_in.dev_ptr(),
                )
                # Whiten inputs to match the dynamics-train convention.
                self._normalize_dyn_in_gpu()

                # Elite dynamics forward → (mu, lv).
                var dyn_in_t = TileTensor(
                    self._dyn_in.dev_ptr(),
                    row_major[Self.BATCH, Self.DYN_IN](),
                )
                # Per-transition elite mixing: forward ALL elites into the
                # stacked (mu, lv) buffers, then gather a random elite PER
                # transition. (One elite per chunk lets the policy exploit a
                # single model's bias → optimistic synthetic reward.)
                var n_elites = len(self.ensemble.elite_indices)
                for e in range(n_elites):
                    var off = e * Self.BATCH * Self.DYN_PRED
                    var mu_e_t = TileTensor(
                        self._ro_mu_all.dev_ptr() + off,
                        row_major[Self.BATCH, Self.DYN_PRED](),
                    )
                    var lv_e_t = TileTensor(
                        self._ro_lv_all.dev_ptr() + off,
                        row_major[Self.BATCH, Self.DYN_PRED](),
                    )
                    self.ensemble.predict_member["gpu"](
                        self.ensemble.elite_indices[e], dyn_in_t, mu_e_t, lv_e_t,
                    )
                comptime assign_kernel = _mbpo_elite_assign_kernel[Self.BATCH]
                var slot_lt = LayoutTensor[
                    DT, Layout.row_major(Self.BATCH), MutAnyOrigin,
                ](self._ro_slot.dev_ptr())
                ctx.enqueue_function[assign_kernel](
                    slot_lt,
                    Int32(n_elites),
                    self._roll_rng_seed + UInt64(0x5107),
                    self._elite_rng_offset,
                    grid_dim=n_lane_blocks, block_dim=TPB,
                )
                self._elite_rng_offset += UInt64(Self.BATCH)
                comptime gather_kernel = _mbpo_elite_gather_kernel[
                    Self.BATCH, Self.NUM_ELITES, Self.DYN_PRED
                ]
                var mu_all_lt = LayoutTensor[
                    DT,
                    Layout.row_major(
                        Self.NUM_ELITES * Self.BATCH, Self.DYN_PRED
                    ),
                    MutAnyOrigin,
                ](self._ro_mu_all.dev_ptr())
                var lv_all_lt = LayoutTensor[
                    DT,
                    Layout.row_major(
                        Self.NUM_ELITES * Self.BATCH, Self.DYN_PRED
                    ),
                    MutAnyOrigin,
                ](self._ro_lv_all.dev_ptr())
                var g_mu_lt = LayoutTensor[
                    DT, Layout.row_major(Self.BATCH, Self.DYN_PRED), MutAnyOrigin,
                ](self._ro_mu.dev_ptr())
                var g_lv_lt = LayoutTensor[
                    DT, Layout.row_major(Self.BATCH, Self.DYN_PRED), MutAnyOrigin,
                ](self._ro_lv.dev_ptr())
                ctx.enqueue_function[gather_kernel](
                    mu_all_lt, lv_all_lt, slot_lt, g_mu_lt, g_lv_lt,
                    grid_dim=n_lane_blocks, block_dim=TPB,
                )

                # Posterior sample: device box-muller noise → (rew, nxt).
                box_muller_normal_gpu[n_noise](
                    ctx, self._ro_noise.dev_ptr(),
                    self._roll_rng_seed, self._roll_rng_offset,
                )
                self._roll_rng_offset += UInt64(((n_noise + 1) // 2) * 2)
                var obs_lt = LayoutTensor[
                    DT, Layout.row_major(Self.BATCH, Self.OBS_DIM), MutAnyOrigin,
                ](self._ro_obs.dev_ptr())
                var mu_lt = LayoutTensor[
                    DT, Layout.row_major(Self.BATCH, Self.DYN_PRED), MutAnyOrigin,
                ](self._ro_mu.dev_ptr())
                var lv_lt = LayoutTensor[
                    DT, Layout.row_major(Self.BATCH, Self.DYN_PRED), MutAnyOrigin,
                ](self._ro_lv.dev_ptr())
                var noise_lt = LayoutTensor[
                    DT, Layout.row_major(Self.BATCH, Self.DYN_PRED), MutAnyOrigin,
                ](self._ro_noise.dev_ptr())
                var nxt_lt = LayoutTensor[
                    DT, Layout.row_major(Self.BATCH, Self.OBS_DIM), MutAnyOrigin,
                ](self._ro_nxt.dev_ptr())
                var rew_lt = LayoutTensor[
                    DT, Layout.row_major(Self.BATCH), MutAnyOrigin,
                ](self._ro_rew.dev_ptr())
                ctx.enqueue_function[post_kernel](
                    obs_lt, mu_lt, lv_lt, noise_lt, nxt_lt, rew_lt,
                    grid_dim=n_lane_blocks, block_dim=TPB,
                )

                # Store BATCH synthetic transitions (s=obs, a=act, r=rew,
                # sp=nxt, d=0) in one device batch.
                self.sample_blk.synth_add_batch[Self.BATCH](
                    ctx,
                    self._ro_obs.dev.value(),
                    self._ro_act.dev.value(),
                    self._ro_rew.dev.value(),
                    self._ro_nxt.dev.value(),
                    self._ro_done.dev.value(),
                )

                # Roll forward: obs ← nxt.
                ctx.enqueue_copy(self._ro_obs.dev.value(), self._ro_nxt.dev.value())
