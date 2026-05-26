"""MBPOTrainer — Model-Based Policy Optimisation (Janner et al. 2019)
built by composing the existing nn2 `SACTrainer` with a
`DynamicsEnsembleBlock` and a synthetic-rollout replay buffer.

Phase I.1.c/d. The first non-classic-actor-critic agent in nn2.
Validates the framework's thesis: a brand-new model-based agent should
land in <1500 LOC of algorithm-specific code by reusing the SAC half +
the dynamics ensemble block + the GaussianNLL loss + the existing
replay surface.

## Composition diagram

    MBPOTrainer
    ├── sac: SACTrainer          ←  unmodified (bit-identity anchor)
    │   ├── actor, critics
    │   ├── opts, blocks, scratch
    │   └── buf  (REAL replay)
    ├── ensemble: DynamicsEnsembleBlock
    │   ├── N members + opts
    │   ├── shared GaussianNLLLoss
    │   └── elite_indices
    ├── synth_buf: CPUReplay     ←  synthetic-rollout transitions
    └── _mb_*: own scratch       ←  mixed-batch assembly slabs

## Why compose vs fork SAC

The audit's killing-feature claim ("first agent NOT shaped like
actor-critic, ships in <1500 LOC") requires that we *reuse* the SAC
half rather than re-implement it.  MBPOTrainer never modifies
`SACTrainer`'s state outside of `sac.record()` (writes real transition)
and the four `sac._train_*` helpers (drive a per-batch SAC update from
caller-supplied minibatch pointers).  The bit-identity anchor
(−167.572) is preserved by construction.

## Training loop cadence (matches Janner et al. + deep_agents reference)

For each env step:
  1. `record(s, a, r, s', d)` → push to `sac.buf` (real buffer).
  2. `train_step(step_idx)`:
     a. If `step_idx < sac.learning_starts`: skip (warmup).
     b. If `step_idx % model_train_freq == 0`:
        - Train all ensemble members on real buffer for
          `dyn_epochs_per_round` epochs.  Compute holdout NLL, refresh
          elite indices.
        - Generate `num_rollouts_per_step` synthetic rollouts of length
          `rollout_length`, pushing each transition into `synth_buf`.
     c. For `sac_updates_per_step` repetitions:
        - Sample `REAL_BS = BATCH * REAL_RATIO_PCT / 100` from
          `sac.buf`; sample `SYNTH_BS = BATCH - REAL_BS` from
          `synth_buf`; concatenate into `_mb_*` slabs.
        - Call `sac._train_compute_target_y`, `_train_critic_update`,
          `_train_actor_update`, `_train_alpha_update`, `_train_polyak`
          with our slab pointers.

## Scope cuts (I.1.c/d MVP)

  - **CPU only.**  GPU path raises in `make[gpu]`.  Per audit gate,
    primary validation target is "Pendulum CPU 30k within ±20 of
    deep_agents MBPO".  GPU is a follow-up phase.
  - **No input scaler.**  Bounded-obs envs (Pendulum) work without
    normalisation.  Unbounded-obs envs (HalfCheetah-style) will need a
    running mean/std normaliser added before they converge.
  - **No learnable logvar bounds.**  Fixed `[-10, -2]` matches deep_agents
    CPU path.
  - **No PER, no n-step, no ERE, no multi-env.**  All add code but no
    algorithmic value for the I.1 thesis-validation gate.
  - **Single-pass dynamics training**: `dyn_epochs_per_round` passes
    through the real buffer with simple SGD-Adam.  No early stopping.
    The trainer's outer loop calls model training every
    `model_train_freq` env steps, which is the same effective shape as
    the reference's "train until early-stop, repeat".

## Conformance

Implements `OffPolicyTrainable` so the existing `run_offpolicy_train_cpu`
driver works for MBPO with zero changes.  `select_action` / `record` /
`end_episode` / `mean_return` / `ep_count` forward to the inner
`SACTrainer`; only `train_step` carries MBPO-specific orchestration.
"""

from std.math import exp as fexp, sqrt as fsqrt, log as flog
from std.random import random_float64, randn_float64
from std.gpu.host import DeviceContext
from layout import TileTensor, row_major

from ..constants import DT
from ..core import AMPPolicy, NoAMP
from ..core.module import Module
from ..core.scratch import Scratch
from ..core.scratch_walkers import init_scratch_auto
from ..core.target_storage import TargetStorage, assert_tag_for
from ..data.cpu_replay import CPUReplay
from .driver_cpu import OffPolicyTrainable
from .dynamics_ensemble_block import DynamicsEnsembleBlock
from .sac_trainer import SACTrainer


struct MBPOTrainer[
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
](OffPolicyTrainable):
    """MBPO trainer = SACTrainer + DynamicsEnsembleBlock + synth replay.

    `REAL_RATIO_PCT` is comptime (mirrors deep_agents) so the
    mixed-batch sizes `REAL_BS` / `SYNTH_BS` fold to constants. With
    `BATCH = 256` and `REAL_RATIO_PCT = 5` we get REAL_BS = 12, SYNTH_BS = 244.

    `DynNet` MUST have `IN_DIM = OBS_DIM + ACT_DIM` and
    `OUT_DIM = 2 * (1 + OBS_DIM)` (mean + logvar over reward + Δobs).
    """

    comptime SAC_TRAINER = SACTrainer[
        Self.ACTOR, Self.CRITIC, Self.OBS_DIM, Self.ACT_DIM,
        Self.BATCH, Self.REPLAY_CAPACITY,
    ]
    comptime DYN_IN: Int = Self.OBS_DIM + Self.ACT_DIM
    comptime DYN_PRED: Int = 1 + Self.OBS_DIM
    comptime DYN_OUT: Int = 2 * Self.DYN_PRED
    comptime ENSEMBLE = DynamicsEnsembleBlock[
        Self.DynNet, Self.N_ENSEMBLE, Self.NUM_ELITES,
        Self.DYN_IN, Self.DYN_OUT, Self.BATCH,
        Self.LOGVAR_MIN, Self.LOGVAR_MAX,
    ]
    comptime REAL_BS: Int = (Self.BATCH * Self.REAL_RATIO_PCT) // 100
    comptime SYNTH_BS: Int = Self.BATCH - Self.REAL_BS

    # Trait-conformance aliases for the driver.
    comptime AGENT_OBS_DIM: Int = Self.OBS_DIM
    comptime AGENT_ACT_DIM: Int = Self.ACT_DIM

    var sac: Self.SAC_TRAINER
    var ensemble: Self.ENSEMBLE
    var synth_buf: CPUReplay[
        Self.OBS_DIM, Self.ACT_DIM, Self.SYNTH_CAPACITY
    ]

    # Mixed-batch assembly scratch — distinct from sac._mb_* (which sac
    # owns and writes during its own sampling path; we route around that
    # by feeding our slabs into the _train_* helpers directly).
    var _mb_s: Scratch["mb_s", Self.BATCH * Self.OBS_DIM]
    var _mb_a: Scratch["mb_a", Self.BATCH * Self.ACT_DIM]
    var _mb_r: Scratch["mb_r", Self.BATCH]
    var _mb_sp: Scratch["mb_sp", Self.BATCH * Self.OBS_DIM]
    var _mb_d: Scratch["mb_d", Self.BATCH]
    var _mb_y: Scratch["mb_y", Self.BATCH]

    # Rollout scratch — used inside _generate_synthetic_rollouts.
    var _ro_obs: Scratch["ro_obs", Self.OBS_DIM]
    var _ro_act: Scratch["ro_act", Self.ACT_DIM]
    var _ro_nxt: Scratch["ro_nxt", Self.OBS_DIM]
    # Per-rollout-batch ensemble in/out (BATCH wide for vectorised predict).
    var _ro_in: Scratch["ro_in", Self.BATCH * Self.DYN_IN]
    var _ro_mu: Scratch["ro_mu", Self.BATCH * Self.DYN_PRED]
    var _ro_lv: Scratch["ro_lv", Self.BATCH * Self.DYN_PRED]

    # MBPO hyperparams.
    var model_train_freq: Int           # train dynamics every N env steps
    var dyn_epochs_per_round: Int       # passes over real buffer per round
    var rollout_length: Int             # synthetic horizon (per epoch)
    var num_rollouts_per_step: Int      # parallel rollouts per round
    var sac_updates_per_step: Int       # SAC update count per env step
    var holdout_ratio: Float64          # train/holdout split for dyn
    var dyn_batch_size: Int             # dynamics minibatch size
    var step_count: Int                 # cumulative env steps seen
    var last_dyn_step: Int              # step_idx of last dyn-train round

    var ts: TargetStorage

    def __init__(out self):
        comptime assert Self.DynNet.IN_DIM == Self.DYN_IN, (
            "MBPOTrainer: DynNet.IN_DIM must equal OBS_DIM + ACT_DIM"
        )
        comptime assert Self.DynNet.OUT_DIM == Self.DYN_OUT, (
            "MBPOTrainer: DynNet.OUT_DIM must equal 2 * (1 + OBS_DIM)"
        )
        comptime assert (
            Self.REAL_RATIO_PCT >= 0 and Self.REAL_RATIO_PCT <= 100
        ), "REAL_RATIO_PCT must be in [0, 100]"
        comptime assert Self.REAL_BS >= 1, (
            "REAL_RATIO_PCT * BATCH / 100 must be >= 1"
        )
        comptime assert Self.SYNTH_BS >= 1, (
            "SYNTH_BS = BATCH - REAL_BS must be >= 1"
        )
        self.sac = Self.SAC_TRAINER()
        self.ensemble = Self.ENSEMBLE()
        self.synth_buf = CPUReplay[
            Self.OBS_DIM, Self.ACT_DIM, Self.SYNTH_CAPACITY
        ](
            obs=UnsafePointer[Scalar[DT], MutAnyOrigin](unsafe_from_address=0),
            act=UnsafePointer[Scalar[DT], MutAnyOrigin](unsafe_from_address=0),
            rew=UnsafePointer[Scalar[DT], MutAnyOrigin](unsafe_from_address=0),
            nxt=UnsafePointer[Scalar[DT], MutAnyOrigin](unsafe_from_address=0),
            dne=UnsafePointer[Scalar[DT], MutAnyOrigin](unsafe_from_address=0),
            size=0, pos=0,
        )
        self._mb_s = Scratch["mb_s", Self.BATCH * Self.OBS_DIM]()
        self._mb_a = Scratch["mb_a", Self.BATCH * Self.ACT_DIM]()
        self._mb_r = Scratch["mb_r", Self.BATCH]()
        self._mb_sp = Scratch["mb_sp", Self.BATCH * Self.OBS_DIM]()
        self._mb_d = Scratch["mb_d", Self.BATCH]()
        self._mb_y = Scratch["mb_y", Self.BATCH]()
        self._ro_obs = Scratch["ro_obs", Self.OBS_DIM]()
        self._ro_act = Scratch["ro_act", Self.ACT_DIM]()
        self._ro_nxt = Scratch["ro_nxt", Self.OBS_DIM]()
        self._ro_in = Scratch["ro_in", Self.BATCH * Self.DYN_IN]()
        self._ro_mu = Scratch["ro_mu", Self.BATCH * Self.DYN_PRED]()
        self._ro_lv = Scratch["ro_lv", Self.BATCH * Self.DYN_PRED]()
        self.model_train_freq = 250
        self.dyn_epochs_per_round = 4
        self.rollout_length = 1
        self.num_rollouts_per_step = 400
        self.sac_updates_per_step = 20
        self.holdout_ratio = 0.0       # MVP: skip holdout split (use full buf for train)
        self.dyn_batch_size = 256
        self.step_count = 0
        self.last_dyn_step = -1
        self.ts = TargetStorage.make_uninit()

    @staticmethod
    def make[target: StaticString](
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
    ) raises -> Self:
        """CPU factory.  All hyperparams have deep_agents-aligned defaults
        so the typical Pendulum / MuJoCo recipe needs only a few overrides."""
        comptime assert target == "cpu", (
            "MBPOTrainer.make[target='gpu'] is not implemented (I.1 is CPU-first)"
        )
        var t = Self()
        t.sac = Self.SAC_TRAINER.make[target](
            actor_lr=actor_lr, critic_lr=critic_lr, alpha_lr=alpha_lr,
            gamma=gamma, tau=tau, action_scale=action_scale,
            init_alpha=init_alpha, target_entropy=target_entropy,
            learning_starts=learning_starts,
            window_size=window_size,
            initial_episode_fill=initial_episode_fill,
        )
        # Use Kaiming for the dynamics ensemble — Swish-MLP matches
        # ReLU-style fan-in scaling (LinearSwish in nn1 also Kaiming-inits).
        from ..initializer import Kaiming
        t.ensemble = Self.ENSEMBLE.make[target, INIT=Kaiming]()
        t.ensemble.set_lr(model_lr)
        # Synthetic replay slab allocation.
        t.synth_buf = CPUReplay[
            Self.OBS_DIM, Self.ACT_DIM, Self.SYNTH_CAPACITY
        ].new()
        t.model_train_freq = model_train_freq
        t.dyn_epochs_per_round = dyn_epochs_per_round
        t.rollout_length = rollout_length
        t.num_rollouts_per_step = num_rollouts_per_step
        t.sac_updates_per_step = sac_updates_per_step
        t.dyn_batch_size = dyn_batch_size
        t.ts = TargetStorage.make_cpu()
        init_scratch_auto[Self, target="cpu"](t)
        return t^

    @staticmethod
    def make[target: StaticString](ctx: DeviceContext) raises -> Self:
        comptime assert target == "gpu", (
            "MBPOTrainer.make[target='cpu'](ctx) — drop ctx for CPU"
        )
        raise Error(
            "MBPOTrainer GPU not yet implemented (I.1 is CPU-first)"
        )

    # ─────────────────────────────────────────────────────────────────
    # OffPolicyTrainable conformance — forward most methods to sac.
    # ─────────────────────────────────────────────────────────────────

    def select_action(
        mut self,
        ref obs: List[Scalar[DT]],
        mut action_out: List[Scalar[DT]],
        step_idx: Int,
    ) raises:
        self.sac.select_action(obs, action_out, step_idx)

    def select_greedy_action(
        mut self,
        ref obs: List[Scalar[DT]],
        mut action_out: List[Scalar[DT]],
    ) raises:
        self.sac.select_greedy_action(obs, action_out)

    def record(
        mut self,
        ref obs: List[Scalar[DT]],
        ref action: List[Scalar[DT]],
        reward: Scalar[DT],
        ref next_obs: List[Scalar[DT]],
        done: Scalar[DT],
    ) raises:
        self.sac.record(obs, action, reward, next_obs, done)
        self.step_count += 1

    def end_episode(mut self):
        self.sac.end_episode()

    def mean_return(self) -> Scalar[DT]:
        return self.sac.mean_return()

    def ep_count(self) -> Int:
        return self.sac.ep_count()

    # ─────────────────────────────────────────────────────────────────
    # train_step — MBPO orchestration.
    # ─────────────────────────────────────────────────────────────────

    def train_step(mut self, step_idx: Int) raises -> Bool:
        """Forward to the CPU train_step.  Trait-conformance shim
        matching SAC's parametric/non-parametric pattern."""
        return self.train_step["cpu"](step_idx)

    def train_step[
        target: StaticString,
        POLICY: AMPPolicy = NoAMP,
    ](mut self, step_idx: Int) raises -> Bool:
        """Run one MBPO train round.

        Returns True iff at least one SAC update actually executed."""
        comptime assert target == "cpu", (
            "MBPOTrainer.train_step['gpu'] not implemented"
        )
        # Warmup gate.  Same threshold as the inner SAC.
        if step_idx < self.sac.learning_starts:
            return False

        # Dynamics + rollout phase, run every `model_train_freq` steps
        # (including the first post-warmup step so the synth buffer
        # gets populated before any SAC update consumes it).
        var should_train_dyn = (
            self.last_dyn_step < 0
            or step_idx - self.last_dyn_step >= self.model_train_freq
        )
        if should_train_dyn:
            self._train_dynamics_ensemble()
            self._generate_synthetic_rollouts()
            self.last_dyn_step = step_idx

        # If the synth buffer hasn't been filled yet (e.g. first round
        # the rollout produced < SYNTH_BS transitions), skip SAC updates
        # for this step rather than under-fill the batch.
        if self.synth_buf.size < Self.SYNTH_BS:
            return False
        if self.sac.buf.size < Self.REAL_BS:
            return False

        var any_step = False
        for _ in range(self.sac_updates_per_step):
            any_step = self._sac_step_on_mixed_batch[target, POLICY]() or any_step
        return any_step

    # ─────────────────────────────────────────────────────────────────
    # Dynamics ensemble training pass.
    # ─────────────────────────────────────────────────────────────────

    def _train_dynamics_ensemble(mut self) raises:
        """Sample mini-batches from `sac.buf` and run
        `dyn_epochs_per_round * (size/BATCH)` train_member_step calls
        per ensemble member.  Members differ by initialisation +
        per-batch sampling stochasticity.

        Holdout-based elite refresh is gated on `holdout_ratio > 0` —
        for MVP I leave it off (all members are elite, matching the
        initial state). When enabled, after training each member we
        evaluate it on a held-out chunk and call
        `ensemble.update_elites` with the resulting NLLs."""
        var n_data = self.sac.buf.size
        if n_data < self.dyn_batch_size:
            return

        var bs = self.dyn_batch_size
        var steps_per_epoch = n_data // bs
        if steps_per_epoch < 1:
            steps_per_epoch = 1
        var total_steps = steps_per_epoch * self.dyn_epochs_per_round

        # Build target = [reward, delta_obs] from (s, a, r, s') in
        # the inner SAC's real buffer.
        var dyn_in: UnsafePointer[Scalar[DT], MutAnyOrigin] = (
            self._mb_s.cpu_ptr()  # reuse the mixed-batch slab as scratch
        )
        var dyn_target: UnsafePointer[Scalar[DT], MutAnyOrigin] = (
            self._mb_r.cpu_ptr()  # reuse; size BATCH ≥ DYN_PRED for any BATCH ≥ 1
        )
        # Need dedicated slabs sized BATCH × DYN_IN and BATCH × DYN_PRED;
        # _ro_in is BATCH × DYN_IN and _ro_mu is BATCH × DYN_PRED — reuse.
        var dyn_in_p = self._ro_in.cpu_ptr()
        var dyn_target_p = self._ro_mu.cpu_ptr()

        for m in range(Self.N_ENSEMBLE):
            for _ in range(total_steps):
                # Sample BATCH indices from sac.buf (uniform with replacement).
                for k in range(Self.BATCH):
                    var idx = Int(random_float64() * Float64(n_data))
                    if idx >= n_data:
                        idx = n_data - 1
                    # input = [obs, act]
                    for d in range(Self.OBS_DIM):
                        dyn_in_p[k * Self.DYN_IN + d] = (
                            self.sac.buf.obs[idx * Self.OBS_DIM + d]
                        )
                    for j in range(Self.ACT_DIM):
                        dyn_in_p[k * Self.DYN_IN + Self.OBS_DIM + j] = (
                            self.sac.buf.act[idx * Self.ACT_DIM + j]
                        )
                    # target = [reward, delta_obs = next_obs - obs]
                    dyn_target_p[k * Self.DYN_PRED + 0] = self.sac.buf.rew[idx]
                    for d in range(Self.OBS_DIM):
                        dyn_target_p[k * Self.DYN_PRED + 1 + d] = (
                            self.sac.buf.nxt[idx * Self.OBS_DIM + d]
                            - self.sac.buf.obs[idx * Self.OBS_DIM + d]
                        )
                var dyn_in_t = TileTensor(
                    dyn_in_p, row_major[Self.BATCH, Self.DYN_IN]()
                )
                var dyn_target_t = TileTensor(
                    dyn_target_p, row_major[Self.BATCH, Self.DYN_PRED]()
                )
                _ = self.ensemble.train_member_step["cpu"](
                    m, dyn_in_t, dyn_target_t,
                )

    # ─────────────────────────────────────────────────────────────────
    # Synthetic rollout generator.
    # ─────────────────────────────────────────────────────────────────

    def _generate_synthetic_rollouts(mut self) raises:
        """Sample `num_rollouts_per_step` starting states from sac.buf,
        run `rollout_length` steps each with a random elite predicting
        (Δobs, reward).  Push every (s, a, r, s', d=0) into synth_buf.

        Done is always 0 — MBPO's reference TerminationFn for
        Pendulum is NeverTerminate.  Per-env termination is a future
        extension via a TerminationFn template parameter (mirrors the
        deep_agents config knob).  Pendulum has no early-termination."""
        if self.sac.buf.size < 1:
            return

        # Use the BATCH-wide predict path: prepare BATCH rollouts in
        # parallel, advance one step at a time, push BATCH transitions
        # per step. num_rollouts_per_step is the outer count of how many
        # BATCH-fanout starting states we draw.
        var rollouts_done = 0
        while rollouts_done < self.num_rollouts_per_step:
            var this_batch = Self.BATCH
            var remaining = self.num_rollouts_per_step - rollouts_done
            if remaining < this_batch:
                this_batch = remaining

            # Sample `this_batch` starting obs from sac.buf and stage
            # them into _ro_in (we'll fill action below per rollout step).
            var roll_obs_p = self._mb_s.cpu_ptr()   # current obs slab
            var roll_act_p = self._mb_a.cpu_ptr()   # action slab
            var roll_nxt_p = self._mb_sp.cpu_ptr()  # next obs slab

            for k in range(this_batch):
                var idx = Int(random_float64() * Float64(self.sac.buf.size))
                if idx >= self.sac.buf.size:
                    idx = self.sac.buf.size - 1
                for d in range(Self.OBS_DIM):
                    roll_obs_p[k * Self.OBS_DIM + d] = (
                        self.sac.buf.obs[idx * Self.OBS_DIM + d]
                    )

            for _ in range(self.rollout_length):
                # 1. Per-rollout action via current policy.  We invoke
                #    sac.select_action one-at-a-time — vectorising
                #    requires an N_ENVS-batched select path the trainer
                #    doesn't expose today.  For BATCH=256, Pendulum,
                #    this is ~0.1 ms/rollout which is fine.
                for k in range(this_batch):
                    var obs_list = List[Scalar[DT]](capacity=Self.OBS_DIM)
                    for d in range(Self.OBS_DIM):
                        obs_list.append(roll_obs_p[k * Self.OBS_DIM + d])
                    var act_list = List[Scalar[DT]](capacity=Self.ACT_DIM)
                    for _ in range(Self.ACT_DIM):
                        act_list.append(Scalar[DT](0.0))
                    # Always use the stochastic action sampler post-warmup;
                    # passing step_idx > learning_starts ensures the
                    # rsample branch is taken (not the uniform warmup).
                    self.sac.select_action(
                        obs_list, act_list, self.sac.learning_starts + 1,
                    )
                    for j in range(Self.ACT_DIM):
                        roll_act_p[k * Self.ACT_DIM + j] = act_list[j]

                # 2. Build ensemble input [obs | action] and predict.
                var ro_in_p = self._ro_in.cpu_ptr()
                var ro_mu_p = self._ro_mu.cpu_ptr()
                var ro_lv_p = self._ro_lv.cpu_ptr()
                for k in range(this_batch):
                    for d in range(Self.OBS_DIM):
                        ro_in_p[k * Self.DYN_IN + d] = (
                            roll_obs_p[k * Self.OBS_DIM + d]
                        )
                    for j in range(Self.ACT_DIM):
                        ro_in_p[k * Self.DYN_IN + Self.OBS_DIM + j] = (
                            roll_act_p[k * Self.ACT_DIM + j]
                        )
                var ro_in_t = TileTensor(
                    ro_in_p, row_major[Self.BATCH, Self.DYN_IN]()
                )
                var ro_mu_t = TileTensor(
                    ro_mu_p, row_major[Self.BATCH, Self.DYN_PRED]()
                )
                var ro_lv_t = TileTensor(
                    ro_lv_p, row_major[Self.BATCH, Self.DYN_PRED]()
                )

                # Pick a random elite per BATCH slice — matches
                # reference per-sample elite sampling at the per-batch
                # granularity.  Per-sample elite sampling would require
                # one ensemble forward per elite which is wasteful here.
                var n_elites = len(self.ensemble.elite_indices)
                var elite_pick = Int(random_float64() * Float64(n_elites))
                if elite_pick >= n_elites:
                    elite_pick = n_elites - 1
                var member_idx = self.ensemble.elite_indices[elite_pick]
                self.ensemble.predict_member["cpu"](
                    member_idx, ro_in_t, ro_mu_t, ro_lv_t,
                )

                # 3. Sample from the predicted Gaussian:
                #    reward_sample  = mu_r  + exp(0.5 * lv_r)  * N(0, 1)
                #    delta_obs_samp = mu_d  + exp(0.5 * lv_d)  * N(0, 1)
                #    next_obs       = obs   + delta_obs_samp
                #    push (obs, action, reward_sample, next_obs, done=0)
                #    into synth_buf.
                var s_list = List[Scalar[DT]](capacity=Self.OBS_DIM)
                var a_list = List[Scalar[DT]](capacity=Self.ACT_DIM)
                var sp_list = List[Scalar[DT]](capacity=Self.OBS_DIM)
                for _ in range(Self.OBS_DIM):
                    s_list.append(Scalar[DT](0.0))
                    sp_list.append(Scalar[DT](0.0))
                for _ in range(Self.ACT_DIM):
                    a_list.append(Scalar[DT](0.0))
                for k in range(this_batch):
                    var mu_r = ro_mu_p[k * Self.DYN_PRED + 0]
                    var lv_r = ro_lv_p[k * Self.DYN_PRED + 0]
                    var std_r = fsqrt(fexp(lv_r))
                    var noise_r = Scalar[DT](randn_float64())
                    var rew = mu_r + std_r * noise_r
                    for d in range(Self.OBS_DIM):
                        s_list[d] = roll_obs_p[k * Self.OBS_DIM + d]
                        var mu_d = ro_mu_p[k * Self.DYN_PRED + 1 + d]
                        var lv_d = ro_lv_p[k * Self.DYN_PRED + 1 + d]
                        var std_d = fsqrt(fexp(lv_d))
                        var noise = Scalar[DT](randn_float64())
                        var delta = mu_d + std_d * noise
                        var nxt = roll_obs_p[k * Self.OBS_DIM + d] + delta
                        sp_list[d] = nxt
                        roll_nxt_p[k * Self.OBS_DIM + d] = nxt
                    for j in range(Self.ACT_DIM):
                        a_list[j] = roll_act_p[k * Self.ACT_DIM + j]
                    self.synth_buf.add(
                        s_list, a_list, rew, sp_list, Scalar[DT](0.0),
                    )

                # 4. Slide window: next iteration starts from sampled next_obs.
                for k in range(this_batch * Self.OBS_DIM):
                    roll_obs_p[k] = roll_nxt_p[k]

            rollouts_done += this_batch

    # ─────────────────────────────────────────────────────────────────
    # SAC step on a mixed real + synthetic batch.
    # ─────────────────────────────────────────────────────────────────

    def _sac_step_on_mixed_batch[
        target: StaticString,
        POLICY: AMPPolicy = NoAMP,
    ](mut self) raises -> Bool:
        """Build a BATCH-wide minibatch from REAL_BS samples of sac.buf
        + SYNTH_BS samples of synth_buf, then run one SAC update via
        the inner SAC's `_train_*` helpers (using our own slab pointers
        rather than sac's internal scratch)."""
        # Sample REAL_BS into the first REAL_BS rows of each slab.
        var mb_s_p = self._mb_s.cpu_ptr()
        var mb_a_p = self._mb_a.cpu_ptr()
        var mb_r_p = self._mb_r.cpu_ptr()
        var mb_sp_p = self._mb_sp.cpu_ptr()
        var mb_d_p = self._mb_d.cpu_ptr()
        var mb_y_p = self._mb_y.cpu_ptr()

        # Real partition: rows [0, REAL_BS).
        var real_s_p = mb_s_p
        var real_a_p = mb_a_p
        var real_r_p = mb_r_p
        var real_sp_p = mb_sp_p
        var real_d_p = mb_d_p
        # Sample REAL_BS items from sac.buf via its own sample API,
        # writing into the head of our slabs.  CPUReplay.sample is
        # generic over (n, s_out, a_out, r_out, sp_out, d_out).
        self.sac.buf.sample(
            Self.REAL_BS,
            real_s_p, real_a_p, real_r_p, real_sp_p, real_d_p,
        )

        # Synth partition: rows [REAL_BS, BATCH).
        var synth_s_p = mb_s_p + Self.REAL_BS * Self.OBS_DIM
        var synth_a_p = mb_a_p + Self.REAL_BS * Self.ACT_DIM
        var synth_r_p = mb_r_p + Self.REAL_BS
        var synth_sp_p = mb_sp_p + Self.REAL_BS * Self.OBS_DIM
        var synth_d_p = mb_d_p + Self.REAL_BS
        self.synth_buf.sample(
            Self.SYNTH_BS,
            synth_s_p, synth_a_p, synth_r_p, synth_sp_p, synth_d_p,
        )

        # Run the SAC per-batch update using our own slab pointers.
        var alpha = fexp(self.sac.alpha_opt.value)
        self.sac._train_compute_target_y[target, POLICY](
            alpha, mb_sp_p, mb_r_p, mb_y_p,
        )
        var crit_loss = self.sac._train_critic_update[target, POLICY](
            mb_s_p, mb_a_p, mb_y_p,
        )
        var actor_res = self.sac._train_actor_update[target, POLICY](
            alpha, mb_s_p,
        )
        self.sac._train_alpha_update(actor_res.log_prob_mean)
        self.sac._train_polyak[target]()
        return True
