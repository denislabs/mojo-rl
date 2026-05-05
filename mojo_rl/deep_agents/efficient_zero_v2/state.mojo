"""EfficientZero V2 CPU state — five networks + replay buffer + per-step
MCTS targets + K-step + SimSiam scratch.

Originally I tried composing this on top of `MuZeroCPUState` to share the
buffer plumbing, but Mojo nightly's parameter inference crashes when
`Network.forward` is called on a `state.core.representation` whose type
goes through a multi-level trait-alias chain
(`EZV2DiscreteCPUState[Cfg].MuZeroCPUState[Cfg].RepModel`). Owning all
five networks directly at this struct's level keeps every model type one
alias deep — `EZV2DiscreteCPUState[Cfg].RepModel`, etc. — and the
inference machinery can unify them with the `Config.RepModel` path used
by callers.

Layout:

  • Five networks: `representation`, `dynamics`, `prediction` from MuZero
    + `projector`, `predictor` from SimSiam.
  • Replay buffer + per-step MCTS targets (policies, root values,
    player-to-move).
  • K-step unroll scratch — hidden states, prediction outputs, dynamics
    reward logits along the rolled-out branch (mirrors MuZeroCPUState's
    `_hidden_states` / `_pred_outputs` / `_dyn_reward_logits`).
  • SimSiam scratch — projector and predictor outputs at each unroll step
    on the dynamics branch + projector outputs on the obs branch
    (detached / stop-grad).
  • Caches for backward through every network at every K position.
  • Gradient scratch for the K-step BPTT plus the consistency-loss path.

The reward-prefix LSTM head (paper App. G) is intentionally absent —
risk register defers it until after CartPole converges.
"""

from std.memory import alloc, memset
from layout import Layout, LayoutTensor
from mojo_rl.nn.constants import dtype
from mojo_rl.nn.initializer import Kaiming
from mojo_rl.nn.training import NetworkState
from mojo_rl.deep_agents.core.replay.sequence_replay_buffer import (
    SequenceReplayBuffer,
)
from mojo_rl.deep_agents.efficient_zero_v2.configs import EZV2DiscreteConfig


struct EZV2DiscreteCPUState[
    Config: EZV2DiscreteConfig,
    _CAP: Int = 50000,
](Movable):
    """CPU state for EZ-V2 (discrete) training.

    Parameters:
        Config: `EZV2DiscreteConfig` providing all dimensions, networks,
            optimizer, training hyperparameters, and EZ-V2 loss weights.
        _CAP: Replay-buffer capacity (default 50000).
    """

    # ── Shorthand compile-time constants ─────────────────────────────────
    comptime OBS: Int = Self.Config.obs_dim
    comptime ACT: Int = Self.Config.action_dim
    comptime LATENT: Int = Self.Config.latent_dim
    comptime BINS: Int = Self.Config.num_bins
    comptime BATCH: Int = Self.Config.batch_size
    comptime K: Int = Self.Config.unroll_steps
    comptime N: Int = Self.Config.td_steps
    comptime DYN_IN: Int = Self.Config.DYN_IN
    comptime DYN_OUT: Int = Self.Config.DYN_OUT
    comptime PRED_OUT: Int = Self.Config.PRED_OUT
    comptime PROJ: Int = Self.Config.proj_dim

    # ── Networks ─────────────────────────────────────────────────────────
    # Use `Self.Config.X` directly rather than re-aliasing to `Self.X` so
    # callers that pass these networks back to Mojo functions parameterized
    # by `Config: EZV2DiscreteConfig` see exactly the alias path they
    # expect (`Cfg.RepModel`, not `EZV2DiscreteCPUState[Cfg].RepModel`).
    # The latter form trips up Mojo nightly's overload resolution when the
    # call site infers the model type via parameter inference.
    var representation: NetworkState[Self.Config.RepModel, Self.Config.OptType]
    var dynamics: NetworkState[Self.Config.DynModel, Self.Config.OptType]
    var prediction: NetworkState[Self.Config.PredModel, Self.Config.OptType]
    var projector: NetworkState[
        Self.Config.ProjectorModel, Self.Config.OptType
    ]
    var predictor: NetworkState[
        Self.Config.PredictorModel, Self.Config.OptType
    ]

    # ── Replay buffer ────────────────────────────────────────────────────
    # Use `Self.Config.X` directly (rather than `Self.X` aliases) so the
    # buffer's type parameters expose to callers as `Config.obs_dim` /
    # `Config.action_dim`, which Mojo's overload checker can unify with
    # the literal-derived shapes used elsewhere.
    var buffer: SequenceReplayBuffer[
        Self._CAP, Self.Config.obs_dim, Self.Config.action_dim
    ]

    # ── MCTS target storage (parallel to replay buffer) ──────────────────
    var mcts_policies: UnsafePointer[
        Scalar[dtype], MutAnyOrigin
    ]  # [_CAP * ACT]
    var mcts_values: UnsafePointer[Scalar[dtype], MutAnyOrigin]  # [_CAP]
    var mcts_to_play: UnsafePointer[
        Scalar[DType.uint8], MutAnyOrigin
    ]  # [_CAP]

    # `train_step_count` at the time each transition was written. Used by
    # the mixed-value-target blend (paper Eq. 16) to compute per-sample
    # data age in train-steps. uint32 wraps at ~4·10⁹ which is well past
    # any practical training run.
    var step_at_write: UnsafePointer[
        Scalar[DType.uint32], MutAnyOrigin
    ]  # [_CAP]

    # Per-transition priority (paper App. A "Priority Precalculation").
    # New transitions are stamped with `max_priority` so they're sampled
    # at least once; thereafter `train_step` overwrites the entry with
    # |TD error| (= the per-sample value-CE loss at unroll position k=0)
    # so future samples bias toward the windows where the current model
    # is most uncertain.
    var priorities: UnsafePointer[Scalar[dtype], MutAnyOrigin]  # [_CAP]

    # ── K-step unroll scratch (dynamics branch) ──────────────────────────
    var _hidden_states: UnsafePointer[
        Scalar[dtype], MutAnyOrigin
    ]  # [(K+1) * BATCH * LATENT]
    var _pred_outputs: UnsafePointer[
        Scalar[dtype], MutAnyOrigin
    ]  # [(K+1) * BATCH * PRED_OUT]
    var _dyn_reward_logits: UnsafePointer[
        Scalar[dtype], MutAnyOrigin
    ]  # [K * BATCH * BINS]

    # ── SimSiam scratch (k = 1..K) ───────────────────────────────────────
    var _proj_dyn: UnsafePointer[
        Scalar[dtype], MutAnyOrigin
    ]  # [K * BATCH * PROJ] — projector(z_dyn[k])
    var _pred_dyn: UnsafePointer[
        Scalar[dtype], MutAnyOrigin
    ]  # [K * BATCH * PROJ] — predictor(proj_dyn[k])
    var _proj_obs: UnsafePointer[
        Scalar[dtype], MutAnyOrigin
    ]  # [K * BATCH * PROJ] — projector(rep(o[k]))   (stop-grad target)
    var _rep_obs: UnsafePointer[
        Scalar[dtype], MutAnyOrigin
    ]  # [K * BATCH * LATENT] — fresh rep encodings of o[1..K]

    # ── Caches (per-step, for backward) ──────────────────────────────────
    var _rep_cache: UnsafePointer[
        Scalar[dtype], MutAnyOrigin
    ]  # [BATCH * RepModel.CACHE_SIZE]
    var _dyn_caches: UnsafePointer[
        Scalar[dtype], MutAnyOrigin
    ]  # [K * BATCH * DynModel.CACHE_SIZE]
    var _pred_caches: UnsafePointer[
        Scalar[dtype], MutAnyOrigin
    ]  # [(K+1) * BATCH * PredModel.CACHE_SIZE]
    var _proj_dyn_caches: UnsafePointer[
        Scalar[dtype], MutAnyOrigin
    ]  # [K * BATCH * ProjectorModel.CACHE_SIZE]
    var _pred_dyn_caches: UnsafePointer[
        Scalar[dtype], MutAnyOrigin
    ]  # [K * BATCH * PredictorModel.CACHE_SIZE]
    var _proj_obs_caches: UnsafePointer[
        Scalar[dtype], MutAnyOrigin
    ]  # [K * BATCH * ProjectorModel.CACHE_SIZE]
    var _rep_obs_caches: UnsafePointer[
        Scalar[dtype], MutAnyOrigin
    ]  # [K * BATCH * RepModel.CACHE_SIZE]

    # ── Gradient scratch ─────────────────────────────────────────────────
    var _grad_hidden: UnsafePointer[
        Scalar[dtype], MutAnyOrigin
    ]  # [BATCH * LATENT]
    var _grad_pred_out: UnsafePointer[
        Scalar[dtype], MutAnyOrigin
    ]  # [BATCH * PRED_OUT]
    var _grad_dyn_out: UnsafePointer[
        Scalar[dtype], MutAnyOrigin
    ]  # [BATCH * DYN_OUT]
    var _grad_pred_dyn: UnsafePointer[
        Scalar[dtype], MutAnyOrigin
    ]  # [BATCH * PROJ]   — d L_G / d (predictor output)
    var _grad_proj_dyn: UnsafePointer[
        Scalar[dtype], MutAnyOrigin
    ]  # [BATCH * PROJ]   — d L_G / d (projector dyn output)

    # ══════════════════════════════════════════════════════════════════════
    # Constructors
    # ══════════════════════════════════════════════════════════════════════

    def __init__(out self):
        """Allocate networks, replay buffer, and all scratch buffers."""

        # ── Networks ─────────────────────────────────────────────────────
        self.representation = NetworkState[Self.Config.RepModel, Self.Config.OptType]()
        self.representation.initialize[Kaiming[]]()

        self.dynamics = NetworkState[Self.Config.DynModel, Self.Config.OptType]()
        self.dynamics.initialize[Kaiming[]]()

        self.prediction = NetworkState[Self.Config.PredModel, Self.Config.OptType]()
        self.prediction.initialize[Kaiming[]]()

        self.projector = NetworkState[Self.Config.ProjectorModel, Self.Config.OptType]()
        self.projector.initialize[Kaiming[]]()

        self.predictor = NetworkState[Self.Config.PredictorModel, Self.Config.OptType]()
        self.predictor.initialize[Kaiming[]]()

        # ── Replay buffer + MCTS targets ─────────────────────────────────
        self.buffer = SequenceReplayBuffer[
            Self._CAP, Self.Config.obs_dim, Self.Config.action_dim
        ]()

        comptime POLICY_SIZE = Self._CAP * Self.ACT
        self.mcts_policies = alloc[Scalar[dtype]](POLICY_SIZE)
        memset(self.mcts_policies, 0, POLICY_SIZE)

        self.mcts_values = alloc[Scalar[dtype]](Self._CAP)
        memset(self.mcts_values, 0, Self._CAP)

        self.mcts_to_play = alloc[Scalar[DType.uint8]](Self._CAP)
        memset(self.mcts_to_play, 0, Self._CAP)

        self.step_at_write = alloc[Scalar[DType.uint32]](Self._CAP)
        memset(self.step_at_write, 0, Self._CAP)

        self.priorities = alloc[Scalar[dtype]](Self._CAP)
        memset(self.priorities, 0, Self._CAP)

        # ── K-step unroll scratch (dynamics branch) ──────────────────────
        comptime HIDDEN_SIZE = (Self.K + 1) * Self.BATCH * Self.LATENT
        self._hidden_states = alloc[Scalar[dtype]](HIDDEN_SIZE)
        memset(self._hidden_states, 0, HIDDEN_SIZE)

        comptime PRED_SIZE = (Self.K + 1) * Self.BATCH * Self.PRED_OUT
        self._pred_outputs = alloc[Scalar[dtype]](PRED_SIZE)
        memset(self._pred_outputs, 0, PRED_SIZE)

        comptime DYN_REW_SIZE = Self.K * Self.BATCH * Self.BINS
        self._dyn_reward_logits = alloc[Scalar[dtype]](DYN_REW_SIZE)
        memset(self._dyn_reward_logits, 0, DYN_REW_SIZE)

        # ── SimSiam K-step scratch (k=1..K) ──────────────────────────────
        comptime PROJ_BUF = Self.K * Self.BATCH * Self.PROJ
        self._proj_dyn = alloc[Scalar[dtype]](PROJ_BUF)
        memset(self._proj_dyn, 0, PROJ_BUF)
        self._pred_dyn = alloc[Scalar[dtype]](PROJ_BUF)
        memset(self._pred_dyn, 0, PROJ_BUF)
        self._proj_obs = alloc[Scalar[dtype]](PROJ_BUF)
        memset(self._proj_obs, 0, PROJ_BUF)

        comptime LAT_BUF = Self.K * Self.BATCH * Self.LATENT
        self._rep_obs = alloc[Scalar[dtype]](LAT_BUF)
        memset(self._rep_obs, 0, LAT_BUF)

        # ── Cache scratch ────────────────────────────────────────────────
        comptime REP_CS = Self.Config.RepModel.CACHE_SIZE
        comptime REP_CACHE_SIZE = Self.BATCH * REP_CS
        self._rep_cache = alloc[Scalar[dtype]](REP_CACHE_SIZE)
        memset(self._rep_cache, 0, REP_CACHE_SIZE)

        comptime DYN_CS = Self.Config.DynModel.CACHE_SIZE
        comptime DYN_CACHE_SIZE = Self.K * Self.BATCH * DYN_CS
        self._dyn_caches = alloc[Scalar[dtype]](DYN_CACHE_SIZE)
        memset(self._dyn_caches, 0, DYN_CACHE_SIZE)

        comptime PRED_CS = Self.Config.PredModel.CACHE_SIZE
        comptime PRED_CACHE_SIZE = (Self.K + 1) * Self.BATCH * PRED_CS
        self._pred_caches = alloc[Scalar[dtype]](PRED_CACHE_SIZE)
        memset(self._pred_caches, 0, PRED_CACHE_SIZE)

        comptime PROJ_CS = Self.Config.ProjectorModel.CACHE_SIZE
        comptime PROJ_DYN_CACHE_SIZE = Self.K * Self.BATCH * PROJ_CS
        self._proj_dyn_caches = alloc[Scalar[dtype]](PROJ_DYN_CACHE_SIZE)
        memset(self._proj_dyn_caches, 0, PROJ_DYN_CACHE_SIZE)

        comptime PRED_SS_CS = Self.Config.PredictorModel.CACHE_SIZE
        comptime PRED_DYN_CACHE_SIZE = Self.K * Self.BATCH * PRED_SS_CS
        self._pred_dyn_caches = alloc[Scalar[dtype]](PRED_DYN_CACHE_SIZE)
        memset(self._pred_dyn_caches, 0, PRED_DYN_CACHE_SIZE)

        comptime PROJ_OBS_CACHE_SIZE = Self.K * Self.BATCH * PROJ_CS
        self._proj_obs_caches = alloc[Scalar[dtype]](PROJ_OBS_CACHE_SIZE)
        memset(self._proj_obs_caches, 0, PROJ_OBS_CACHE_SIZE)

        comptime REP_OBS_CACHE_SIZE = Self.K * Self.BATCH * REP_CS
        self._rep_obs_caches = alloc[Scalar[dtype]](REP_OBS_CACHE_SIZE)
        memset(self._rep_obs_caches, 0, REP_OBS_CACHE_SIZE)

        # ── Gradient scratch ─────────────────────────────────────────────
        comptime GRAD_HIDDEN_SIZE = Self.BATCH * Self.LATENT
        self._grad_hidden = alloc[Scalar[dtype]](GRAD_HIDDEN_SIZE)
        memset(self._grad_hidden, 0, GRAD_HIDDEN_SIZE)

        comptime GRAD_PRED_SIZE = Self.BATCH * Self.PRED_OUT
        self._grad_pred_out = alloc[Scalar[dtype]](GRAD_PRED_SIZE)
        memset(self._grad_pred_out, 0, GRAD_PRED_SIZE)

        comptime GRAD_DYN_OUT_SIZE = Self.BATCH * Self.DYN_OUT
        self._grad_dyn_out = alloc[Scalar[dtype]](GRAD_DYN_OUT_SIZE)
        memset(self._grad_dyn_out, 0, GRAD_DYN_OUT_SIZE)

        comptime GRAD_PROJ_SIZE = Self.BATCH * Self.PROJ
        self._grad_pred_dyn = alloc[Scalar[dtype]](GRAD_PROJ_SIZE)
        memset(self._grad_pred_dyn, 0, GRAD_PROJ_SIZE)
        self._grad_proj_dyn = alloc[Scalar[dtype]](GRAD_PROJ_SIZE)
        memset(self._grad_proj_dyn, 0, GRAD_PROJ_SIZE)

    def __init__(out self, *, deinit take: Self):
        self.representation = take.representation^
        self.dynamics = take.dynamics^
        self.prediction = take.prediction^
        self.projector = take.projector^
        self.predictor = take.predictor^
        self.buffer = take.buffer^
        self.mcts_policies = take.mcts_policies
        self.mcts_values = take.mcts_values
        self.mcts_to_play = take.mcts_to_play
        self.step_at_write = take.step_at_write
        self.priorities = take.priorities
        self._hidden_states = take._hidden_states
        self._pred_outputs = take._pred_outputs
        self._dyn_reward_logits = take._dyn_reward_logits
        self._proj_dyn = take._proj_dyn
        self._pred_dyn = take._pred_dyn
        self._proj_obs = take._proj_obs
        self._rep_obs = take._rep_obs
        self._rep_cache = take._rep_cache
        self._dyn_caches = take._dyn_caches
        self._pred_caches = take._pred_caches
        self._proj_dyn_caches = take._proj_dyn_caches
        self._pred_dyn_caches = take._pred_dyn_caches
        self._proj_obs_caches = take._proj_obs_caches
        self._rep_obs_caches = take._rep_obs_caches
        self._grad_hidden = take._grad_hidden
        self._grad_pred_out = take._grad_pred_out
        self._grad_dyn_out = take._grad_dyn_out
        self._grad_pred_dyn = take._grad_pred_dyn
        self._grad_proj_dyn = take._grad_proj_dyn

    def __del__(deinit self):
        self.mcts_policies.free()
        self.mcts_values.free()
        self.mcts_to_play.free()
        self.step_at_write.free()
        self.priorities.free()
        self._hidden_states.free()
        self._pred_outputs.free()
        self._dyn_reward_logits.free()
        self._proj_dyn.free()
        self._pred_dyn.free()
        self._proj_obs.free()
        self._rep_obs.free()
        self._rep_cache.free()
        self._dyn_caches.free()
        self._pred_caches.free()
        self._proj_dyn_caches.free()
        self._pred_dyn_caches.free()
        self._proj_obs_caches.free()
        self._rep_obs_caches.free()
        self._grad_hidden.free()
        self._grad_pred_out.free()
        self._grad_dyn_out.free()
        self._grad_pred_dyn.free()
        self._grad_proj_dyn.free()

    # ══════════════════════════════════════════════════════════════════════
    # Convenience accessors
    # ══════════════════════════════════════════════════════════════════════

    def is_ready(self) -> Bool:
        """True when the replay buffer holds enough transitions to sample
        a full K+N+1 window. Reads `.size` directly rather than calling
        `.len()` because Mojo nightly's overload checker rejects method
        dispatch on a struct whose comptime parameters resolve through a
        Config alias chain."""
        comptime needed = Self.K + Self.N + 1
        return self.buffer.size >= needed
