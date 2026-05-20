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

from std.memory import alloc, memset, memcpy
from layout import Layout, LayoutTensor
from std.gpu.host import DeviceContext, DeviceBuffer, HostBuffer
from mojo_rl.nn.constants import dtype
from mojo_rl.nn.initializer import Kaiming, Xavier
from mojo_rl.nn.training import NetworkState, GPUNetworkState
from mojo_rl.nn.model import LSTMCell
from mojo_rl.core.sum_tree import SumTree
from mojo_rl.deep_agents.core.replay.sequence_replay_buffer import (
    SequenceReplayBuffer,
)
from mojo_rl.deep_agents.efficient_zero_v2.configs import EZV2DiscreteConfig
from mojo_rl.deep_agents.efficient_zero_v2.networks import RewardPrefixHeadMLP
from mojo_rl.deep_agents.efficient_zero_v2.gpu_mcts import (
    EZV2GPUMCTSState,
    run_gumbel_search_gpu,
)


# ════════════════════════════════════════════════════════════════════════════
# Phase 2 staging helper (EZV2_CONTINUOUS_TRAINING_PERF.md)
# ════════════════════════════════════════════════════════════════════════════


def copy_window_dtype[
    ELEM: Int
](
    dst: UnsafePointer[Scalar[dtype], MutAnyOrigin],
    src: UnsafePointer[Scalar[dtype], MutAnyOrigin],
    start: Int,
    n_rows: Int,
    capacity: Int,
):
    """Copy `n_rows` consecutive ELEM-wide rows from a circular source
    array (`src`, capacity `capacity` rows) to a packed destination.

    Source row k starts at `src[((start + k) % capacity) * ELEM]`.
    Destination row k starts at `dst[k * ELEM]`.

    Splits into two memcpys when start + n_rows > capacity (wraparound).
    Replaces the per-(sample, k, d) scalar copy loops in the
    `train_step_gpu` staging block (Phase 2). For OBS=17, K_ROOT*ACT=96,
    K+1=6 rows per sample × 256 samples, this collapses ~210k scalar
    reads/writes into a few hundred memcpys.
    """
    var pre_wrap = capacity - start
    if pre_wrap >= n_rows:
        memcpy(
            dest=dst,
            src=src + start * ELEM,
            count=n_rows * ELEM,
        )
    else:
        memcpy(
            dest=dst,
            src=src + start * ELEM,
            count=pre_wrap * ELEM,
        )
        memcpy(
            dest=dst + pre_wrap * ELEM,
            src=src,
            count=(n_rows - pre_wrap) * ELEM,
        )


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
    var projector: NetworkState[Self.Config.ProjectorModel, Self.Config.OptType]
    var predictor: NetworkState[Self.Config.PredictorModel, Self.Config.OptType]

    # Target networks for MuZero-style Reanalyze (paper App. A): a
    # slowly-tracking copy of rep / dyn / pred is used to re-run Gumbel
    # search on stale replay-buffer transitions and refresh their stored
    # MCTS policies + root values. The projector and predictor are
    # SimSiam-training-only — no consistency loss is computed during
    # reanalyze — so they don't need a target copy.
    var representation_target: NetworkState[
        Self.Config.RepModel, Self.Config.OptType
    ]
    var dynamics_target: NetworkState[Self.Config.DynModel, Self.Config.OptType]
    var prediction_target: NetworkState[
        Self.Config.PredModel, Self.Config.OptType
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
    var mcts_to_play: UnsafePointer[Scalar[DType.uint8], MutAnyOrigin]  # [_CAP]

    # Continuous-action EZ-V2 full-π loss (paper Eq. 6) targets.
    # `mcts_sampled_actions`: K root-sampled candidate action vectors per
    # replay slot. `mcts_improved_policy`: softmax(completed_Q + log_prior
    # + gumbel) weights over those K candidates. Used by the new
    # `ezv2_policy_loss_grad_continuous_fullpi_kernel` when action_dim==1.
    # For discrete configs or action_dim>1, these stay zero-initialized
    # and the simple-best path (which reads `mcts_policies`) is used.
    var mcts_sampled_actions: UnsafePointer[
        Scalar[dtype], MutAnyOrigin
    ]  # [_CAP * K_ROOT * ACT]
    var mcts_improved_policy: UnsafePointer[
        Scalar[dtype], MutAnyOrigin
    ]  # [_CAP * K_ROOT]

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

    # Sum-tree mirror of `priorities`, gated by K-step window validity:
    # `priority_tree.get(j)` = max(priorities[j], 1e-8) when slot j is a
    # valid K-step window start (not in the leading edge, no `done` in
    # `dones[j..j+K-1]`), else 0. Lets `train_step_gpu` sample BATCH
    # window starts in O(BATCH * log _CAP) instead of the O(_CAP * BATCH)
    # cum_prio linear scan. Maintained incrementally by `on_flush_write`
    # and the post-train priority writeback. See
    # docs/EZV2_CONTINUOUS_TRAINING_PERF.md (Phase 1).
    var priority_tree: SumTree[dtype]

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

    # ── Reward-prefix LSTM head (paper App. G) ───────────────────────────
    # Always-allocated. Used only when `Config.use_reward_prefix=True`;
    # storage cost at LSTM_HIDDEN=64 is on the order of 100KB so we keep
    # it simple and skip conditional-field gymnastics. The training loop
    # branches at compile time on `use_reward_prefix`.
    #
    # The cell itself is parameterized as `LSTMCell[LATENT, LSTM_HIDDEN]`
    # since its input is the post-dynamics latent `hidden[k+1]`. The cell
    # is *not* Model-trait-conforming (it has explicit (h, c) plumbing
    # for BPTT), so we manage its params/grads/Adam state by hand — but
    # the post-LSTM MLP head IS a `Sequential` that fits `NetworkState`.
    comptime _LSTMHead = LSTMCell[
        Self.Config.latent_dim, Self.Config.lstm_hidden
    ]
    comptime _RewardPrefixMLPModel = RewardPrefixHeadMLP[
        Self.Config.lstm_hidden,
        Self.Config.lstm_mlp_hidden,
        Self.Config.num_bins,
    ]
    comptime _LSTM_PS: Int = Self._LSTMHead.PARAM_SIZE
    comptime _LSTM_CS: Int = Self._LSTMHead.CACHE_SIZE
    comptime _MLP_HEAD_CS: Int = Self._RewardPrefixMLPModel.CACHE_SIZE

    # LSTM trainable state — params, grads, Adam (m, v).
    var lstm_params: UnsafePointer[Scalar[dtype], MutAnyOrigin]  # [PS]
    var lstm_grads: UnsafePointer[Scalar[dtype], MutAnyOrigin]  # [PS]
    var lstm_opt_state: UnsafePointer[
        Scalar[dtype], MutAnyOrigin
    ]  # [PS * 2] — Adam m, v interleaved per-param

    # MLP head NetworkState — has its own params/grads/Adam state.
    var reward_prefix_mlp: NetworkState[
        Self._RewardPrefixMLPModel, Self.Config.OptType
    ]

    # Per-step LSTM hidden + cell state (time-major: layout matches
    # `_hidden_states`). Indexed at k=0..K so we can BPTT through it; h_lstm[0]
    # = c_lstm[0] = 0 at horizon-aligned start, and at every horizon
    # boundary the input slot for the *next* step is reset to zero.
    var _lstm_h_states: UnsafePointer[
        Scalar[dtype], MutAnyOrigin
    ]  # [(K+1) * BATCH * LSTM_HIDDEN]
    var _lstm_c_states: UnsafePointer[
        Scalar[dtype], MutAnyOrigin
    ]  # [(K+1) * BATCH * LSTM_HIDDEN]
    var _lstm_caches: UnsafePointer[
        Scalar[dtype], MutAnyOrigin
    ]  # [K * BATCH * LSTM_CS]
    var _mlp_head_caches: UnsafePointer[
        Scalar[dtype], MutAnyOrigin
    ]  # [K * BATCH * MLP_HEAD_CS]

    # Cumulative reward target scratch — `cum_rewards[b, k] = Σ_{j≤k} reward[b, j]`,
    # filled on-the-fly per batch.
    var _cum_rewards: UnsafePointer[Scalar[dtype], MutAnyOrigin]  # [BATCH * K]

    # ══════════════════════════════════════════════════════════════════════
    # Constructors
    # ══════════════════════════════════════════════════════════════════════

    def __init__(out self):
        """Allocate networks, replay buffer, and all scratch buffers."""

        # ── Networks ─────────────────────────────────────────────────────
        self.representation = NetworkState[
            Self.Config.RepModel, Self.Config.OptType
        ]()
        self.representation.initialize[Kaiming[]]()

        self.dynamics = NetworkState[
            Self.Config.DynModel, Self.Config.OptType
        ]()
        self.dynamics.initialize[Kaiming[]]()

        self.prediction = NetworkState[
            Self.Config.PredModel, Self.Config.OptType
        ]()
        self.prediction.initialize[Kaiming[]]()

        self.projector = NetworkState[
            Self.Config.ProjectorModel, Self.Config.OptType
        ]()
        self.projector.initialize[Kaiming[]]()

        self.predictor = NetworkState[
            Self.Config.PredictorModel, Self.Config.OptType
        ]()
        self.predictor.initialize[Kaiming[]]()

        # Target networks: independent Kaiming init; the agent
        # `update_target_networks(tau=1.0)` call in __init__ overwrites
        # them with the online params so they start synced.
        self.representation_target = NetworkState[
            Self.Config.RepModel, Self.Config.OptType
        ]()
        self.representation_target.initialize[Kaiming[]]()

        self.dynamics_target = NetworkState[
            Self.Config.DynModel, Self.Config.OptType
        ]()
        self.dynamics_target.initialize[Kaiming[]]()

        self.prediction_target = NetworkState[
            Self.Config.PredModel, Self.Config.OptType
        ]()
        self.prediction_target.initialize[Kaiming[]]()

        # ── Replay buffer + MCTS targets ─────────────────────────────────
        self.buffer = SequenceReplayBuffer[
            Self._CAP, Self.Config.obs_dim, Self.Config.action_dim
        ]()

        comptime POLICY_SIZE = Self._CAP * Self.ACT
        self.mcts_policies = alloc[Scalar[dtype]](POLICY_SIZE)
        memset(self.mcts_policies, 0, POLICY_SIZE)

        comptime SAMP_ACTIONS_SIZE = (
            Self._CAP * Self.Config.num_root_candidates * Self.ACT
        )
        self.mcts_sampled_actions = alloc[Scalar[dtype]](SAMP_ACTIONS_SIZE)
        memset(self.mcts_sampled_actions, 0, SAMP_ACTIONS_SIZE)

        comptime IPI_SIZE = (Self._CAP * Self.Config.num_root_candidates)
        self.mcts_improved_policy = alloc[Scalar[dtype]](IPI_SIZE)
        memset(self.mcts_improved_policy, 0, IPI_SIZE)

        self.mcts_values = alloc[Scalar[dtype]](Self._CAP)
        memset(self.mcts_values, 0, Self._CAP)

        self.mcts_to_play = alloc[Scalar[DType.uint8]](Self._CAP)
        memset(self.mcts_to_play, 0, Self._CAP)

        self.step_at_write = alloc[Scalar[DType.uint32]](Self._CAP)
        memset(self.step_at_write, 0, Self._CAP)

        self.priorities = alloc[Scalar[dtype]](Self._CAP)
        memset(self.priorities, 0, Self._CAP)

        self.priority_tree = SumTree[dtype](capacity=Self._CAP)

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

        # ── Reward-prefix LSTM head (always allocated) ──────────────────
        comptime LSTM_PS = Self._LSTM_PS
        comptime LSTM_CS = Self._LSTM_CS
        comptime LSTM_HIDDEN = Self.Config.lstm_hidden
        comptime MLP_HEAD_CS = Self._MLP_HEAD_CS

        self.lstm_params = alloc[Scalar[dtype]](LSTM_PS)
        var lstm_params_view = LayoutTensor[
            dtype, Layout.row_major(LSTM_PS), MutAnyOrigin
        ](self.lstm_params)
        Self._LSTMHead.initialize_params[Xavier[]](lstm_params_view)

        self.lstm_grads = alloc[Scalar[dtype]](LSTM_PS)
        memset(self.lstm_grads, 0, LSTM_PS)

        # Adam state — STATE_PER_PARAM=2 (m, v). We don't go through the
        # `Optimizer.step` trait method (LSTMCell isn't a Model so the
        # NetworkState plumbing doesn't apply); the agent calls
        # `Self.Config.OptType.step[LSTM_PS](...)` directly.
        comptime LSTM_OPT_STATE_SIZE = LSTM_PS * Self.Config.OptType.STATE_PER_PARAM
        self.lstm_opt_state = alloc[Scalar[dtype]](LSTM_OPT_STATE_SIZE)
        memset(self.lstm_opt_state, 0, LSTM_OPT_STATE_SIZE)

        self.reward_prefix_mlp = NetworkState[
            Self._RewardPrefixMLPModel, Self.Config.OptType
        ]()
        self.reward_prefix_mlp.initialize[Kaiming[]]()

        comptime LSTM_HC_SIZE = (Self.K + 1) * Self.BATCH * LSTM_HIDDEN
        self._lstm_h_states = alloc[Scalar[dtype]](LSTM_HC_SIZE)
        memset(self._lstm_h_states, 0, LSTM_HC_SIZE)
        self._lstm_c_states = alloc[Scalar[dtype]](LSTM_HC_SIZE)
        memset(self._lstm_c_states, 0, LSTM_HC_SIZE)

        comptime LSTM_CACHES_SIZE = Self.K * Self.BATCH * LSTM_CS
        self._lstm_caches = alloc[Scalar[dtype]](LSTM_CACHES_SIZE)
        memset(self._lstm_caches, 0, LSTM_CACHES_SIZE)

        comptime MLP_HEAD_CACHES_SIZE = Self.K * Self.BATCH * MLP_HEAD_CS
        self._mlp_head_caches = alloc[Scalar[dtype]](MLP_HEAD_CACHES_SIZE)
        memset(self._mlp_head_caches, 0, MLP_HEAD_CACHES_SIZE)

        comptime CUM_REW_SIZE = Self.BATCH * Self.K
        self._cum_rewards = alloc[Scalar[dtype]](CUM_REW_SIZE)
        memset(self._cum_rewards, 0, CUM_REW_SIZE)

    def __init__(out self, *, deinit take: Self):
        self.representation = take.representation^
        self.dynamics = take.dynamics^
        self.prediction = take.prediction^
        self.projector = take.projector^
        self.predictor = take.predictor^
        self.representation_target = take.representation_target^
        self.dynamics_target = take.dynamics_target^
        self.prediction_target = take.prediction_target^
        self.buffer = take.buffer^
        self.mcts_policies = take.mcts_policies
        self.mcts_sampled_actions = take.mcts_sampled_actions
        self.mcts_improved_policy = take.mcts_improved_policy
        self.mcts_values = take.mcts_values
        self.mcts_to_play = take.mcts_to_play
        self.step_at_write = take.step_at_write
        self.priorities = take.priorities
        self.priority_tree = take.priority_tree^
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
        self.lstm_params = take.lstm_params
        self.lstm_grads = take.lstm_grads
        self.lstm_opt_state = take.lstm_opt_state
        self.reward_prefix_mlp = take.reward_prefix_mlp^
        self._lstm_h_states = take._lstm_h_states
        self._lstm_c_states = take._lstm_c_states
        self._lstm_caches = take._lstm_caches
        self._mlp_head_caches = take._mlp_head_caches
        self._cum_rewards = take._cum_rewards

    def __del__(deinit self):
        self.mcts_policies.free()
        self.mcts_sampled_actions.free()
        self.mcts_improved_policy.free()
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
        self.lstm_params.free()
        self.lstm_grads.free()
        self.lstm_opt_state.free()
        self._lstm_h_states.free()
        self._lstm_c_states.free()
        self._lstm_caches.free()
        self._mlp_head_caches.free()
        self._cum_rewards.free()

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

    # ══════════════════════════════════════════════════════════════════════
    # Priority-tree maintenance (Phase 1 of EZV2_CONTINUOUS_TRAINING_PERF)
    # ══════════════════════════════════════════════════════════════════════

    def _window_is_valid(self, idx: Int) -> Bool:
        """True iff the K-step window starting at `idx` has no `done` in
        slots `idx..idx+K-1` (mod _CAP). Matches the linear-scan validity
        rule in the OLD `train_step_gpu` sampling block — used by
        `on_flush_write` to gate tree priorities."""
        for k in range(Self.K):
            var iidx = (idx + k) % Self._CAP
            if Float64(self.buffer.dones[iidx]) > 0.5:
                return False
        return True

    def on_flush_write(mut self, p: Int):
        """Maintain `priority_tree` after a single `buffer.add_with_termination`
        write at slot `p`. Must be called AFTER `priorities[p]` has been
        set to `max_priority`.

        Invariants enforced:
          • `priority_tree.get(p) = 0` — slot p just entered the K-wide
            "leading edge" (its window extends past the newest slot).
          • `priority_tree.get(p-K+1)` = max(priorities[p-K+1], 1e-8) iff
            slot `(p-K+1) mod _CAP` is a valid K-step start (no done in
            `dones[p-K+1..p]`), else 0. This is the "newly maturing" slot
            on each write — at the same distance K-1 from the new buffer
            head that the old linear scan would have just included.

        The 1e-8 floor mirrors the OLD scan's `if p < 1e-8: p = 1e-8`
        guard.
        """
        # Slot p enters leading edge. Zero its tree weight (overwrites any
        # stale priority left from the previous lap when the buffer wraps).
        self.priority_tree.update(p, Scalar[dtype](0.0))

        # Mature slot (p - K + 1) iff the buffer now has at least K
        # transitions — fewer than K writes can't form a complete window.
        if self.buffer.size >= Self.K:
            var j = (p - Self.K + 1 + Self._CAP) % Self._CAP
            if self._window_is_valid(j):
                var pr = Float64(self.priorities[j])
                if pr < 1e-8:
                    pr = 1e-8
                self.priority_tree.update(j, Scalar[dtype](pr))
            # else: tree[j] already at 0 (set when slot j entered the
            # leading edge K-1 writes ago); no update needed.

    def on_priority_writeback(mut self, idx: Int, new_priority: Float64):
        """Mirror a per-batch priority update onto the tree. `idx` is one
        of the slots returned by `priority_tree.sample` during the
        current train step — so it's known-valid by construction; no
        re-validation needed. Floors below 1e-8 to mirror the OLD scan.
        """
        var pr = new_priority
        if pr < 1e-8:
            pr = 1e-8
        self.priority_tree.update(idx, Scalar[dtype](pr))

    def rebuild_priority_tree(mut self):
        """Rebuild `priority_tree` from current `priorities` + `dones` +
        `buffer.ptr` + `buffer.size`. O(_CAP * K) — for tests or callers
        that pre-populate the buffer by directly mutating `priorities` /
        `buffer.dones` without going through `_flush_episode`. The normal
        training path uses incremental maintenance via `on_flush_write`
        and `on_priority_writeback`.

        Encodes the same validity rule as the OLD linear scan in
        `train_step_gpu`:
          • Slot j is sampleable iff j is at offset 0..buf_size-K-1 from
            `oldest` (so [j..j+K-1] is fully within the buffered range),
          • AND `dones[j..j+K-1]` are all 0.
        """
        var bsize = self.buffer.size
        if bsize <= Self.K:
            # No valid windows possible — zero every leaf.
            for j in range(Self._CAP):
                self.priority_tree.update(j, Scalar[dtype](0.0))
            return

        var oldest = (self.buffer.ptr - bsize + Self._CAP) % Self._CAP
        # First, zero every slot — including those outside the
        # buffered range and the K-wide leading edge.
        for j in range(Self._CAP):
            self.priority_tree.update(j, Scalar[dtype](0.0))
        # Then mark every valid window start with its (floored) priority.
        for offset in range(bsize - Self.K):
            var j = (oldest + offset) % Self._CAP
            if self._window_is_valid(j):
                var pr = Float64(self.priorities[j])
                if pr < 1e-8:
                    pr = 1e-8
                self.priority_tree.update(j, Scalar[dtype](pr))


# ════════════════════════════════════════════════════════════════════════════
# GPU state — DeviceBuffers + GPUNetworkStates for `train_step_gpu`
# ════════════════════════════════════════════════════════════════════════════
#
# Hybrid design (first-cut GPU port): batch sampling is still done on the
# CPU side from `EZV2DiscreteCPUState` (priority-weighted, with the
# done-flag / start-validity logic the CPU `train_step` already
# implements), then the sampled window is uploaded once per train step
# into the GPU buffers below. The forward / backward / optimizer passes
# all run on device. Per-sample value-CE losses come back to the host so
# the priority array (also CPU-resident) can be refreshed at the
# matching `batch_start_idx[b]`.
#
# Target nets are intentionally absent — reanalyze stays on CPU for now,
# and the GPU `train_step_gpu` does not bootstrap the value target from
# a target-net forward (it uses the stored MCTS root values). When a
# future GPU reanalyze lands, the appropriate `*_target` GPUNetworkStates
# can be added next to the online ones.


struct EZV2GPUStateBase[Config: EZV2DiscreteConfig](Movable):
    """GPU-resident state for EZ-V2 (discrete) training.

    Owns:
      * Five `GPUNetworkState`s mirroring the CPU networks.
      * Device buffers for the sampled batch, K-step unroll scratch,
        SimSiam projection / prediction outputs, per-network caches,
        per-output gradient buffers, a workspace big enough for the
        widest network's per-sample workspace, plus tiny accumulators
        for the four loss components and per-sample priorities.
      * Pinned host buffers for the upload (batch + age) and download
        (per-sample priorities + the four loss component scalars) paths.

    Parameters:
        Config: Same `EZV2DiscreteConfig` used by the CPU state.
    """

    # Compile-time constants matching the CPU state's layout.
    comptime OBS: Int = Self.Config.obs_dim
    comptime ACT: Int = Self.Config.action_dim
    comptime LATENT: Int = Self.Config.latent_dim
    comptime BINS: Int = Self.Config.num_bins
    comptime BATCH: Int = Self.Config.batch_size
    comptime K: Int = Self.Config.unroll_steps
    comptime PROJ: Int = Self.Config.proj_dim
    comptime DYN_IN: Int = Self.Config.DYN_IN
    comptime DYN_OUT: Int = Self.Config.DYN_OUT
    comptime PRED_OUT: Int = Self.Config.PRED_OUT

    # ── Networks (live on GPU) ───────────────────────────────────────────
    var representation: GPUNetworkState[
        Self.Config.RepModel, Self.Config.OptType
    ]
    var dynamics: GPUNetworkState[Self.Config.DynModel, Self.Config.OptType]
    var prediction: GPUNetworkState[Self.Config.PredModel, Self.Config.OptType]
    var projector: GPUNetworkState[
        Self.Config.ProjectorModel, Self.Config.OptType
    ]
    var predictor: GPUNetworkState[
        Self.Config.PredictorModel, Self.Config.OptType
    ]

    # ── Target networks (Phase 3) ────────────────────────────────────────
    # MIXED / SARSA value-target modes need a `target-net` forward to
    # produce boot-V at every (sample, k); same-shape mirrors of the
    # online rep/dyn/pred networks. Sync from CPU targets at the existing
    # `target_sync_interval` cadence via `upload_targets_from`. `dyn` is
    # included here for the Phase 4 GPU-batched reanalyze path; the
    # MIXED boot-V forward only touches `representation_target` and
    # `prediction_target`.
    var representation_target: GPUNetworkState[
        Self.Config.RepModel, Self.Config.OptType
    ]
    var dynamics_target: GPUNetworkState[
        Self.Config.DynModel, Self.Config.OptType
    ]
    var prediction_target: GPUNetworkState[
        Self.Config.PredModel, Self.Config.OptType
    ]

    # ── Sampled batch (uploaded each train step) ─────────────────────────
    # Per-sample-time-major to match `compute_loss_components`'s scratch
    # layout: `batch_obs[(b * (K+1) + k) * OBS + d]`.
    var batch_obs_buf: DeviceBuffer[dtype]  # [BATCH * (K+1) * OBS]
    var batch_actions_buf: DeviceBuffer[dtype]  # [BATCH * K * ACT] one-hot
    var batch_rewards_buf: DeviceBuffer[dtype]  # [BATCH * K]
    var batch_mcts_pol_buf: DeviceBuffer[dtype]  # [BATCH * (K+1) * ACT]
    var batch_mcts_val_buf: DeviceBuffer[dtype]  # [BATCH * (K+1)]
    var batch_age_buf: DeviceBuffer[DType.int32]  # [BATCH * (K+1)]
    # Continuous full-π (paper Eq. 6) targets — K root-sampled candidate
    # actions and their improved-policy weights per replay slot. Used by
    # `ezv2_policy_loss_grad_continuous_fullpi_kernel` when action_dim==1.
    var batch_mcts_samp_act_buf: DeviceBuffer[
        dtype
    ]  # [BATCH * (K+1) * K_ROOT * ACT]
    var batch_mcts_imp_pi_buf: DeviceBuffer[dtype]  # [BATCH * (K+1) * K_ROOT]

    # ── Forward scratch (TIME-MAJOR: hidden[k * BATCH * LATENT + b * LATENT + d]) ─
    var hidden_buf: DeviceBuffer[dtype]  # [(K+1) * BATCH * LATENT]
    var dyn_out_buf: DeviceBuffer[dtype]  # [K * BATCH * DYN_OUT]
    var pred_out_buf: DeviceBuffer[dtype]  # [(K+1) * BATCH * PRED_OUT]
    var rep_input_buf: DeviceBuffer[dtype]  # [BATCH * OBS] (k=0 obs gathered)
    var dyn_input_buf: DeviceBuffer[dtype]  # [BATCH * DYN_IN]
    var obs_step_buf: DeviceBuffer[
        dtype
    ]  # [BATCH * OBS] (k>0 obs gathered, target branch)
    var rep_obs_buf: DeviceBuffer[
        dtype
    ]  # [BATCH * LATENT] (rep target output for current k)

    # ── SimSiam scratch (k = 1..K, indexed by k_offset = k-1) ────────────
    var proj_dyn_buf: DeviceBuffer[dtype]  # [K * BATCH * PROJ]
    var pred_dyn_buf: DeviceBuffer[
        dtype
    ]  # [K * BATCH * PROJ] (predictor output)
    var proj_obs_buf: DeviceBuffer[
        dtype
    ]  # [K * BATCH * PROJ] (target branch — stop-grad)

    # ── Caches (per-network forward → backward) ─────────────────────────
    var rep_cache_buf: DeviceBuffer[dtype]  # [BATCH * RepModel.CACHE_SIZE]
    var dyn_caches_buf: DeviceBuffer[dtype]  # [K * BATCH * DynModel.CACHE_SIZE]
    var pred_caches_buf: DeviceBuffer[
        dtype
    ]  # [(K+1) * BATCH * PredModel.CACHE_SIZE]
    var proj_dyn_caches_buf: DeviceBuffer[
        dtype
    ]  # [K * BATCH * ProjectorModel.CACHE_SIZE]
    var pred_dyn_caches_buf: DeviceBuffer[
        dtype
    ]  # [K * BATCH * PredictorModel.CACHE_SIZE]

    # Target-branch scratch caches (SimSiam stop-grad).
    # Used so that the target rep + projector forward can run in
    # *training-mode* (batch-stat BN), matching the PyTorch reference
    # where both branches use `train()` mode. The cache writes are wasted
    # (no backward through the target), but BN normalization stays
    # consistent between the online and target branches and the trivial
    # constant-output collapse path is properly killed at every step.
    # One-slice buffer reused across k=1..K.
    var rep_obs_cache_buf: DeviceBuffer[
        dtype
    ]  # [BATCH * RepModel.CACHE_SIZE], target rep scratch
    var proj_obs_cache_buf: DeviceBuffer[
        dtype
    ]  # [BATCH * ProjectorModel.CACHE_SIZE], target proj scratch

    # ── Target-net boot-V scratch (Phase 3b) ────────────────────────────
    # Per-(sample) staging for `rep_target → pred_target → decode` chain,
    # called K+1 times per train step (one per timestep k=0..K) under
    # MIXED/SARSA value-target modes. All overwritten each k_step.
    # `boot_v_buf` is per-sample-time-major so the Phase 3c kernel can
    # read it the same way it reads `batch_mcts_val_buf`.
    var tgt_rep_input_buf: DeviceBuffer[dtype]  # [BATCH * OBS]
    var tgt_z_buf: DeviceBuffer[dtype]  # [BATCH * LATENT]
    var tgt_pred_out_buf: DeviceBuffer[dtype]  # [BATCH * PRED_OUT]
    var boot_v_buf: DeviceBuffer[dtype]  # [BATCH * (K+1)]

    # ── Per-output gradient buffers (upstream grads + accumulators) ─────
    var grad_pred_out_buf: DeviceBuffer[dtype]  # [(K+1) * BATCH * PRED_OUT]
    var grad_dyn_out_buf: DeviceBuffer[dtype]  # [K * BATCH * DYN_OUT]
    var grad_pred_dyn_buf: DeviceBuffer[dtype]  # [K * BATCH * PROJ]
    var grad_hidden_buf: DeviceBuffer[dtype]  # [(K+1) * BATCH * LATENT]

    # Per-step backward scratch (overwritten each k):
    var grad_pred_in_step_buf: DeviceBuffer[dtype]  # [BATCH * LATENT]
    var grad_predr_in_step_buf: DeviceBuffer[dtype]  # [BATCH * PROJ]
    var grad_proj_in_step_buf: DeviceBuffer[dtype]  # [BATCH * LATENT]
    var grad_dyn_out_step_buf: DeviceBuffer[dtype]  # [BATCH * DYN_OUT]
    var grad_dyn_in_step_buf: DeviceBuffer[dtype]  # [BATCH * DYN_IN]
    var grad_rep_in_buf: DeviceBuffer[dtype]  # [BATCH * OBS] (discarded)

    # ── Two-hot target dist (per-k slice; built before each value/reward grad) ─
    # The mixed-value-target (paper Eq. 16) is computed on host during
    # sampling and uploaded as a [BATCH * (K+1)] scalar tensor; per-step
    # gathering + scalar_transform + two-hot encode runs on GPU below.
    var value_target_full_buf: DeviceBuffer[dtype]  # [BATCH * (K+1)] uploaded
    var value_target_scalar_buf: DeviceBuffer[
        dtype
    ]  # [BATCH] per-k gathered slice
    var value_target_dist_buf: DeviceBuffer[dtype]  # [BATCH * BINS] two-hot
    var reward_target_scalar_buf: DeviceBuffer[dtype]  # [BATCH] reward at k
    var reward_target_dist_buf: DeviceBuffer[dtype]  # [BATCH * BINS] two-hot
    # Per-k policy target gathered from `batch_mcts_pol_buf`; the per-sample-
    # time-major source layout means we can't just `LayoutTensor`-view a
    # slice — gather kernel runs once per k_step.
    var policy_target_step_buf: DeviceBuffer[dtype]  # [BATCH * ACT]
    # Per-k full-π targets (paper Eq. 6) gathered from the K-candidate
    # time-major batch buffers each k_step.
    var fullpi_target_actions_step_buf: DeviceBuffer[
        dtype
    ]  # [BATCH * K_ROOT * ACT]
    var fullpi_target_policy_step_buf: DeviceBuffer[dtype]  # [BATCH * K_ROOT]

    # ── Loss accumulators (1 scalar each; downloaded at end of step) ────
    var L_R_buf: DeviceBuffer[dtype]
    var L_P_buf: DeviceBuffer[dtype]
    var L_V_buf: DeviceBuffer[dtype]
    var L_G_buf: DeviceBuffer[dtype]
    # Diagnostics — written once per train step by `ezv2_z_feature_var_kernel`
    # / `ezv2_v_pred_var_kernel` on the k=0 outputs of the encoder + value
    # head. Used to detect SimSiam encoder collapse (z_var ≈ 0) and value-
    # head state-collapse (v_pred_var ≈ 0) without re-running forwards on
    # the host. 1 scalar each.
    var z_var_buf: DeviceBuffer[dtype]
    var v_pred_var_buf: DeviceBuffer[dtype]
    var per_sample_loss_scratch_buf: DeviceBuffer[
        dtype
    ]  # [BATCH] reused per kernel
    var per_sample_v_loss_k0_buf: DeviceBuffer[
        dtype
    ]  # [BATCH] saved for priority
    var priorities_out_buf: DeviceBuffer[dtype]  # [BATCH] = v_loss + 1e-3

    # ── Network workspace (max across all 5 networks) ────────────────────
    var workspace_buf: DeviceBuffer[dtype]

    # ── Gradient-clipping partial-sums scratch ───────────────────────────
    # Sized for the largest network's `(PARAM_SIZE + TPB - 1) // TPB`
    # (TPB=256). Reused sequentially across networks during the per-
    # network clip pass before each `optimizer_step`.
    var grad_clip_ps: DeviceBuffer[dtype]

    # ── Host buffers (pinned) for upload/download ────────────────────────
    var batch_obs_host: HostBuffer[dtype]
    var batch_actions_host: HostBuffer[dtype]
    var batch_rewards_host: HostBuffer[dtype]
    var batch_mcts_pol_host: HostBuffer[dtype]
    var batch_mcts_val_host: HostBuffer[dtype]
    var batch_age_host: HostBuffer[DType.int32]
    var batch_mcts_samp_act_host: HostBuffer[dtype]
    var batch_mcts_imp_pi_host: HostBuffer[dtype]
    var value_target_full_host: HostBuffer[dtype]

    var L_R_host: HostBuffer[dtype]
    var L_P_host: HostBuffer[dtype]
    var L_V_host: HostBuffer[dtype]
    var L_G_host: HostBuffer[dtype]
    var priorities_out_host: HostBuffer[dtype]
    # Diagnostics host mirrors (read once per train step from the driver).
    var z_var_host: HostBuffer[dtype]
    var v_pred_var_host: HostBuffer[dtype]

    # ── Reward-prefix LSTM head — GPU buffers (always allocated) ─────────
    # Mirrors the CPU state's LSTM/MLP head fields. Used only when
    # `Config.use_reward_prefix=True`; otherwise the buffers exist but
    # the kernels aren't dispatched. Same comptime aliases as the CPU
    # state so dimensional resolution stays consistent.
    comptime _LSTMHead = LSTMCell[
        Self.Config.latent_dim, Self.Config.lstm_hidden
    ]
    comptime _RewardPrefixMLPModel = RewardPrefixHeadMLP[
        Self.Config.lstm_hidden,
        Self.Config.lstm_mlp_hidden,
        Self.Config.num_bins,
    ]
    comptime _LSTM_PS: Int = Self._LSTMHead.PARAM_SIZE
    comptime _LSTM_CS: Int = Self._LSTMHead.CACHE_SIZE
    comptime _MLP_HEAD_CS: Int = Self._RewardPrefixMLPModel.CACHE_SIZE
    comptime _LSTM_HIDDEN: Int = Self.Config.lstm_hidden

    # LSTM trainable state (the cell isn't Model-conforming, so we manage
    # by hand instead of using GPUNetworkState). Adam state spans
    # `LSTM_PS * STATE_PER_PARAM`; opt-global is `OptType.GLOBAL_STATE_SIZE`.
    var lstm_params_buf: DeviceBuffer[dtype]
    var lstm_grads_buf: DeviceBuffer[dtype]
    var lstm_opt_state_buf: DeviceBuffer[dtype]
    var lstm_opt_global_buf: DeviceBuffer[dtype]
    var lstm_params_host: HostBuffer[dtype]
    var lstm_opt_state_host: HostBuffer[dtype]
    var lstm_opt_global_host: HostBuffer[dtype]
    # Track the host-side step counter used for Adam's bias correction;
    # the real on-device counter lives in `lstm_opt_global_buf` slot 0
    # for graph-safe replay (matches GPUNetworkState's `optimizer_step`
    # bookkeeping for the full networks).
    var lstm_step_num: Int

    # MLP head — Model-conforming so we use a full GPUNetworkState.
    var reward_prefix_mlp_gpu: GPUNetworkState[
        Self._RewardPrefixMLPModel, Self.Config.OptType
    ]

    # Per-step state buffers (TIME-MAJOR like `hidden_buf`).
    var lstm_h_states_buf: DeviceBuffer[dtype]  # [(K+1) * BATCH * LSTM_HIDDEN]
    var lstm_c_states_buf: DeviceBuffer[dtype]  # [(K+1) * BATCH * LSTM_HIDDEN]
    var lstm_caches_buf: DeviceBuffer[dtype]  # [K * BATCH * LSTM_CS]
    var mlp_head_caches_buf: DeviceBuffer[dtype]  # [K * BATCH * MLP_HEAD_CS]

    # Per-step input scratch (filled per-k with either the corresponding
    # h/c state slot or zeros at horizon boundary).
    var lstm_h_input_buf: DeviceBuffer[dtype]  # [BATCH * LSTM_HIDDEN]
    var lstm_c_input_buf: DeviceBuffer[dtype]  # [BATCH * LSTM_HIDDEN]

    # Reward-prefix loss + grad scratch.
    var rew_pref_logits_buf: DeviceBuffer[dtype]  # [K * BATCH * BINS]
    var grad_rew_pref_logits_buf: DeviceBuffer[dtype]  # [K * BATCH * BINS]
    var rew_pref_target_dist_buf: DeviceBuffer[dtype]  # [BATCH * BINS]

    # Backward grad accumulators across LSTM time steps.
    var grad_h_lstm_buf: DeviceBuffer[dtype]  # [(K+1) * BATCH * LSTM_HIDDEN]
    var grad_c_lstm_buf: DeviceBuffer[dtype]  # [(K+1) * BATCH * LSTM_HIDDEN]

    # Backward per-step scratch.
    var grad_mlp_in_step_buf: DeviceBuffer[dtype]  # [BATCH * LSTM_HIDDEN]
    var grad_x_lstm_buf: DeviceBuffer[dtype]  # [BATCH * LATENT]
    var grad_h_prev_lstm_buf: DeviceBuffer[dtype]  # [BATCH * LSTM_HIDDEN]
    var grad_c_prev_lstm_buf: DeviceBuffer[dtype]  # [BATCH * LSTM_HIDDEN]
    # `d_combined` workspace required by `LSTMCell.step_backward_gpu` —
    # passes the assembled per-gate pre-activation gradient between the
    # input-grad and param-grad GPU kernels. Shape `[BATCH, 4 * HIDDEN]`.
    var lstm_d_combined_ws_buf: DeviceBuffer[dtype]

    # Cumulative reward target — computed on host during sampling, then
    # uploaded once per train step.
    var cum_rewards_buf: DeviceBuffer[dtype]  # [BATCH * K]
    var cum_rewards_host: HostBuffer[dtype]

    # GPU sampling scratch buffers (consumed by `ezv2_gpu_sample_and_gather`).
    # Sized at fixed CAP=50000 to match `train_step_gpu`'s comptime CAP. When
    # `train_step_gpu_with_replay` is the entry point, these replace the host-
    # side cum_prio / cand_starts / batch_start_idx allocations.
    var cum_prio_buf: DeviceBuffer[dtype]  # [CAP]
    var cand_starts_buf: DeviceBuffer[DType.int32]  # [CAP]
    var n_valid_buf: DeviceBuffer[DType.int32]  # [1]
    var total_prio_buf: DeviceBuffer[dtype]  # [1]
    var batch_start_idx_buf: DeviceBuffer[DType.int32]  # [BATCH]
    var batch_start_idx_host: HostBuffer[DType.int32]  # [BATCH]

    # ══════════════════════════════════════════════════════════════════════
    # Constructor
    # ══════════════════════════════════════════════════════════════════════

    def __init__(out self, ctx: DeviceContext) raises:
        """Allocate all GPU buffers and network states."""

        # ── Networks ─────────────────────────────────────────────────────
        self.representation = GPUNetworkState[
            Self.Config.RepModel, Self.Config.OptType
        ](ctx)
        self.dynamics = GPUNetworkState[
            Self.Config.DynModel, Self.Config.OptType
        ](ctx)
        self.prediction = GPUNetworkState[
            Self.Config.PredModel, Self.Config.OptType
        ](ctx)
        self.projector = GPUNetworkState[
            Self.Config.ProjectorModel, Self.Config.OptType
        ](ctx)
        self.predictor = GPUNetworkState[
            Self.Config.PredictorModel, Self.Config.OptType
        ](ctx)

        # ── Target networks (Phase 3) ───────────────────────────────────
        # Fresh-allocated; first `upload_targets_from(cpu, ctx)` (called
        # right after `gpu.upload_from(cpu, ctx)` at training start)
        # overwrites the random init with CPU target params.
        self.representation_target = GPUNetworkState[
            Self.Config.RepModel, Self.Config.OptType
        ](ctx)
        self.dynamics_target = GPUNetworkState[
            Self.Config.DynModel, Self.Config.OptType
        ](ctx)
        self.prediction_target = GPUNetworkState[
            Self.Config.PredModel, Self.Config.OptType
        ](ctx)

        # ── Sampled batch (TIME-MAJOR-PER-SAMPLE) ───────────────────────
        self.batch_obs_buf = ctx.enqueue_create_buffer[dtype](
            Self.BATCH * (Self.K + 1) * Self.OBS
        )
        self.batch_actions_buf = ctx.enqueue_create_buffer[dtype](
            Self.BATCH * Self.K * Self.ACT
        )
        self.batch_rewards_buf = ctx.enqueue_create_buffer[dtype](
            Self.BATCH * Self.K
        )
        self.batch_mcts_pol_buf = ctx.enqueue_create_buffer[dtype](
            Self.BATCH * (Self.K + 1) * Self.ACT
        )
        self.batch_mcts_val_buf = ctx.enqueue_create_buffer[dtype](
            Self.BATCH * (Self.K + 1)
        )
        self.batch_age_buf = ctx.enqueue_create_buffer[DType.int32](
            Self.BATCH * (Self.K + 1)
        )
        comptime _K_ROOT = Self.Config.num_root_candidates
        self.batch_mcts_samp_act_buf = ctx.enqueue_create_buffer[dtype](
            Self.BATCH * (Self.K + 1) * _K_ROOT * Self.ACT
        )
        self.batch_mcts_imp_pi_buf = ctx.enqueue_create_buffer[dtype](
            Self.BATCH * (Self.K + 1) * _K_ROOT
        )

        # ── Forward scratch ─────────────────────────────────────────────
        self.hidden_buf = ctx.enqueue_create_buffer[dtype](
            (Self.K + 1) * Self.BATCH * Self.LATENT
        )
        self.dyn_out_buf = ctx.enqueue_create_buffer[dtype](
            Self.K * Self.BATCH * Self.DYN_OUT
        )
        self.pred_out_buf = ctx.enqueue_create_buffer[dtype](
            (Self.K + 1) * Self.BATCH * Self.PRED_OUT
        )
        self.rep_input_buf = ctx.enqueue_create_buffer[dtype](
            Self.BATCH * Self.OBS
        )
        self.dyn_input_buf = ctx.enqueue_create_buffer[dtype](
            Self.BATCH * Self.DYN_IN
        )
        self.obs_step_buf = ctx.enqueue_create_buffer[dtype](
            Self.BATCH * Self.OBS
        )
        self.rep_obs_buf = ctx.enqueue_create_buffer[dtype](
            Self.BATCH * Self.LATENT
        )

        # ── Target-net boot-V scratch (Phase 3b) ────────────────────────
        self.tgt_rep_input_buf = ctx.enqueue_create_buffer[dtype](
            Self.BATCH * Self.OBS
        )
        self.tgt_z_buf = ctx.enqueue_create_buffer[dtype](
            Self.BATCH * Self.LATENT
        )
        self.tgt_pred_out_buf = ctx.enqueue_create_buffer[dtype](
            Self.BATCH * Self.PRED_OUT
        )
        self.boot_v_buf = ctx.enqueue_create_buffer[dtype](
            Self.BATCH * (Self.K + 1)
        )

        # ── SimSiam scratch ─────────────────────────────────────────────
        self.proj_dyn_buf = ctx.enqueue_create_buffer[dtype](
            Self.K * Self.BATCH * Self.PROJ
        )
        self.pred_dyn_buf = ctx.enqueue_create_buffer[dtype](
            Self.K * Self.BATCH * Self.PROJ
        )
        self.proj_obs_buf = ctx.enqueue_create_buffer[dtype](
            Self.K * Self.BATCH * Self.PROJ
        )

        # ── Caches ──────────────────────────────────────────────────────
        comptime REP_CS = Self.Config.RepModel.CACHE_SIZE
        comptime DYN_CS = Self.Config.DynModel.CACHE_SIZE
        comptime PRED_CS = Self.Config.PredModel.CACHE_SIZE
        comptime PROJ_CS = Self.Config.ProjectorModel.CACHE_SIZE
        comptime PREDR_CS = Self.Config.PredictorModel.CACHE_SIZE
        self.rep_cache_buf = ctx.enqueue_create_buffer[dtype](
            Self.BATCH * REP_CS if REP_CS > 0 else 1
        )
        self.dyn_caches_buf = ctx.enqueue_create_buffer[dtype](
            Self.K * Self.BATCH * DYN_CS if DYN_CS > 0 else 1
        )
        self.pred_caches_buf = ctx.enqueue_create_buffer[dtype](
            (Self.K + 1) * Self.BATCH * PRED_CS if PRED_CS > 0 else 1
        )
        self.proj_dyn_caches_buf = ctx.enqueue_create_buffer[dtype](
            Self.K * Self.BATCH * PROJ_CS if PROJ_CS > 0 else 1
        )
        self.pred_dyn_caches_buf = ctx.enqueue_create_buffer[dtype](
            Self.K * Self.BATCH * PREDR_CS if PREDR_CS > 0 else 1
        )
        # Target-branch scratch caches (single slice, reused per k).
        self.rep_obs_cache_buf = ctx.enqueue_create_buffer[dtype](
            Self.BATCH * REP_CS if REP_CS > 0 else 1
        )
        self.proj_obs_cache_buf = ctx.enqueue_create_buffer[dtype](
            Self.BATCH * PROJ_CS if PROJ_CS > 0 else 1
        )

        # ── Gradient buffers ────────────────────────────────────────────
        self.grad_pred_out_buf = ctx.enqueue_create_buffer[dtype](
            (Self.K + 1) * Self.BATCH * Self.PRED_OUT
        )
        self.grad_dyn_out_buf = ctx.enqueue_create_buffer[dtype](
            Self.K * Self.BATCH * Self.DYN_OUT
        )
        self.grad_pred_dyn_buf = ctx.enqueue_create_buffer[dtype](
            Self.K * Self.BATCH * Self.PROJ
        )
        self.grad_hidden_buf = ctx.enqueue_create_buffer[dtype](
            (Self.K + 1) * Self.BATCH * Self.LATENT
        )
        self.grad_pred_in_step_buf = ctx.enqueue_create_buffer[dtype](
            Self.BATCH * Self.LATENT
        )
        self.grad_predr_in_step_buf = ctx.enqueue_create_buffer[dtype](
            Self.BATCH * Self.PROJ
        )
        self.grad_proj_in_step_buf = ctx.enqueue_create_buffer[dtype](
            Self.BATCH * Self.LATENT
        )
        self.grad_dyn_out_step_buf = ctx.enqueue_create_buffer[dtype](
            Self.BATCH * Self.DYN_OUT
        )
        self.grad_dyn_in_step_buf = ctx.enqueue_create_buffer[dtype](
            Self.BATCH * Self.DYN_IN
        )
        self.grad_rep_in_buf = ctx.enqueue_create_buffer[dtype](
            Self.BATCH * Self.OBS
        )

        # ── Target dist scratch ─────────────────────────────────────────
        self.value_target_full_buf = ctx.enqueue_create_buffer[dtype](
            Self.BATCH * (Self.K + 1)
        )
        self.value_target_scalar_buf = ctx.enqueue_create_buffer[dtype](
            Self.BATCH
        )
        self.value_target_dist_buf = ctx.enqueue_create_buffer[dtype](
            Self.BATCH * Self.BINS
        )
        self.reward_target_scalar_buf = ctx.enqueue_create_buffer[dtype](
            Self.BATCH
        )
        self.reward_target_dist_buf = ctx.enqueue_create_buffer[dtype](
            Self.BATCH * Self.BINS
        )
        self.policy_target_step_buf = ctx.enqueue_create_buffer[dtype](
            Self.BATCH * Self.ACT
        )
        self.fullpi_target_actions_step_buf = ctx.enqueue_create_buffer[dtype](
            Self.BATCH * _K_ROOT * Self.ACT
        )
        self.fullpi_target_policy_step_buf = ctx.enqueue_create_buffer[dtype](
            Self.BATCH * _K_ROOT
        )

        # ── Loss accumulators (zeroed at the start of every train step) ─
        self.L_R_buf = ctx.enqueue_create_buffer[dtype](1)
        self.L_P_buf = ctx.enqueue_create_buffer[dtype](1)
        self.L_V_buf = ctx.enqueue_create_buffer[dtype](1)
        self.L_G_buf = ctx.enqueue_create_buffer[dtype](1)
        # Diagnostics (overwritten each train step — no pre-zero needed).
        self.z_var_buf = ctx.enqueue_create_buffer[dtype](1)
        self.v_pred_var_buf = ctx.enqueue_create_buffer[dtype](1)
        self.per_sample_loss_scratch_buf = ctx.enqueue_create_buffer[dtype](
            Self.BATCH
        )
        self.per_sample_v_loss_k0_buf = ctx.enqueue_create_buffer[dtype](
            Self.BATCH
        )
        self.priorities_out_buf = ctx.enqueue_create_buffer[dtype](Self.BATCH)

        # ── Workspace (max across 5 networks; *BATCH samples) ───────────
        comptime WS_REP = Self.Config.RepModel.WORKSPACE_SIZE_PER_SAMPLE
        comptime WS_DYN = Self.Config.DynModel.WORKSPACE_SIZE_PER_SAMPLE
        comptime WS_PRED = Self.Config.PredModel.WORKSPACE_SIZE_PER_SAMPLE
        comptime WS_PROJ = Self.Config.ProjectorModel.WORKSPACE_SIZE_PER_SAMPLE
        comptime WS_PREDR = Self.Config.PredictorModel.WORKSPACE_SIZE_PER_SAMPLE
        comptime WS_M1 = WS_REP if WS_REP > WS_DYN else WS_DYN
        comptime WS_M2 = WS_M1 if WS_M1 > WS_PRED else WS_PRED
        comptime WS_M3 = WS_M2 if WS_M2 > WS_PROJ else WS_PROJ
        comptime WS_MAX = WS_M3 if WS_M3 > WS_PREDR else WS_PREDR
        comptime WS_TOTAL = (Self.BATCH * WS_MAX if WS_MAX > 0 else 1)
        self.workspace_buf = ctx.enqueue_create_buffer[dtype](WS_TOTAL)

        # Gradient-clipping partial-sums scratch — sized for the largest
        # of the 5 networks (TPB=256 matches the kernel constant in
        # efficient_zero_v2.train_step_gpu). Reused sequentially.
        comptime _CLIP_TPB: Int = 256
        comptime _CLIP_PS_REP = (
            Self.Config.RepModel.PARAM_SIZE + _CLIP_TPB - 1
        ) // _CLIP_TPB
        comptime _CLIP_PS_DYN = (
            Self.Config.DynModel.PARAM_SIZE + _CLIP_TPB - 1
        ) // _CLIP_TPB
        comptime _CLIP_PS_PRED = (
            Self.Config.PredModel.PARAM_SIZE + _CLIP_TPB - 1
        ) // _CLIP_TPB
        comptime _CLIP_PS_PROJ = (
            Self.Config.ProjectorModel.PARAM_SIZE + _CLIP_TPB - 1
        ) // _CLIP_TPB
        comptime _CLIP_PS_PREDR = (
            Self.Config.PredictorModel.PARAM_SIZE + _CLIP_TPB - 1
        ) // _CLIP_TPB
        comptime _CLIP_M1 = (
            _CLIP_PS_REP if _CLIP_PS_REP > _CLIP_PS_DYN else _CLIP_PS_DYN
        )
        comptime _CLIP_M2 = (
            _CLIP_M1 if _CLIP_M1 > _CLIP_PS_PRED else _CLIP_PS_PRED
        )
        comptime _CLIP_M3 = (
            _CLIP_M2 if _CLIP_M2 > _CLIP_PS_PROJ else _CLIP_PS_PROJ
        )
        comptime _CLIP_PS_MAX = (
            _CLIP_M3 if _CLIP_M3 > _CLIP_PS_PREDR else _CLIP_PS_PREDR
        )
        self.grad_clip_ps = ctx.enqueue_create_buffer[dtype](_CLIP_PS_MAX)

        # ── Host pinned buffers ─────────────────────────────────────────
        self.batch_obs_host = ctx.enqueue_create_host_buffer[dtype](
            Self.BATCH * (Self.K + 1) * Self.OBS
        )
        self.batch_actions_host = ctx.enqueue_create_host_buffer[dtype](
            Self.BATCH * Self.K * Self.ACT
        )
        self.batch_rewards_host = ctx.enqueue_create_host_buffer[dtype](
            Self.BATCH * Self.K
        )
        self.batch_mcts_pol_host = ctx.enqueue_create_host_buffer[dtype](
            Self.BATCH * (Self.K + 1) * Self.ACT
        )
        self.batch_mcts_val_host = ctx.enqueue_create_host_buffer[dtype](
            Self.BATCH * (Self.K + 1)
        )
        self.batch_age_host = ctx.enqueue_create_host_buffer[DType.int32](
            Self.BATCH * (Self.K + 1)
        )
        self.batch_mcts_samp_act_host = ctx.enqueue_create_host_buffer[dtype](
            Self.BATCH * (Self.K + 1) * _K_ROOT * Self.ACT
        )
        self.batch_mcts_imp_pi_host = ctx.enqueue_create_host_buffer[dtype](
            Self.BATCH * (Self.K + 1) * _K_ROOT
        )
        self.value_target_full_host = ctx.enqueue_create_host_buffer[dtype](
            Self.BATCH * (Self.K + 1)
        )
        self.L_R_host = ctx.enqueue_create_host_buffer[dtype](1)
        self.L_P_host = ctx.enqueue_create_host_buffer[dtype](1)
        self.L_V_host = ctx.enqueue_create_host_buffer[dtype](1)
        self.L_G_host = ctx.enqueue_create_host_buffer[dtype](1)
        self.priorities_out_host = ctx.enqueue_create_host_buffer[dtype](
            Self.BATCH
        )
        self.z_var_host = ctx.enqueue_create_host_buffer[dtype](1)
        self.v_pred_var_host = ctx.enqueue_create_host_buffer[dtype](1)

        # ── Reward-prefix LSTM head GPU buffers ─────────────────────────
        comptime LSTM_PS = Self._LSTM_PS
        comptime LSTM_CS = Self._LSTM_CS
        comptime LSTM_HIDDEN = Self._LSTM_HIDDEN
        comptime MLP_HEAD_CS = Self._MLP_HEAD_CS
        comptime LSTM_OPT_STATE_SIZE = (
            LSTM_PS * Self.Config.OptType.STATE_PER_PARAM
        )
        comptime OPT_GLOBAL_SIZE = Self.Config.OptType.GLOBAL_STATE_SIZE

        self.lstm_params_buf = ctx.enqueue_create_buffer[dtype](LSTM_PS)
        self.lstm_grads_buf = ctx.enqueue_create_buffer[dtype](LSTM_PS)
        ctx.enqueue_memset(self.lstm_grads_buf, 0)
        self.lstm_opt_state_buf = ctx.enqueue_create_buffer[dtype](
            LSTM_OPT_STATE_SIZE
        )
        ctx.enqueue_memset(self.lstm_opt_state_buf, 0)
        self.lstm_opt_global_buf = ctx.enqueue_create_buffer[dtype](
            OPT_GLOBAL_SIZE
        )
        ctx.enqueue_memset(self.lstm_opt_global_buf, 0)

        self.lstm_params_host = ctx.enqueue_create_host_buffer[dtype](LSTM_PS)
        self.lstm_opt_state_host = ctx.enqueue_create_host_buffer[dtype](
            LSTM_OPT_STATE_SIZE
        )
        self.lstm_opt_global_host = ctx.enqueue_create_host_buffer[dtype](
            OPT_GLOBAL_SIZE
        )
        self.lstm_step_num = 0

        self.reward_prefix_mlp_gpu = GPUNetworkState[
            Self._RewardPrefixMLPModel, Self.Config.OptType
        ](ctx)

        comptime LSTM_HC_SIZE = (Self.K + 1) * Self.BATCH * LSTM_HIDDEN
        self.lstm_h_states_buf = ctx.enqueue_create_buffer[dtype](LSTM_HC_SIZE)
        self.lstm_c_states_buf = ctx.enqueue_create_buffer[dtype](LSTM_HC_SIZE)
        ctx.enqueue_memset(self.lstm_h_states_buf, 0)
        ctx.enqueue_memset(self.lstm_c_states_buf, 0)

        comptime LSTM_CACHES_SIZE = Self.K * Self.BATCH * LSTM_CS
        self.lstm_caches_buf = ctx.enqueue_create_buffer[dtype](
            LSTM_CACHES_SIZE if LSTM_CACHES_SIZE > 0 else 1
        )
        comptime MLP_HEAD_CACHES_SIZE = Self.K * Self.BATCH * MLP_HEAD_CS
        self.mlp_head_caches_buf = ctx.enqueue_create_buffer[dtype](
            MLP_HEAD_CACHES_SIZE if MLP_HEAD_CACHES_SIZE > 0 else 1
        )

        self.lstm_h_input_buf = ctx.enqueue_create_buffer[dtype](
            Self.BATCH * LSTM_HIDDEN
        )
        self.lstm_c_input_buf = ctx.enqueue_create_buffer[dtype](
            Self.BATCH * LSTM_HIDDEN
        )

        self.rew_pref_logits_buf = ctx.enqueue_create_buffer[dtype](
            Self.K * Self.BATCH * Self.BINS
        )
        self.grad_rew_pref_logits_buf = ctx.enqueue_create_buffer[dtype](
            Self.K * Self.BATCH * Self.BINS
        )
        self.rew_pref_target_dist_buf = ctx.enqueue_create_buffer[dtype](
            Self.BATCH * Self.BINS
        )

        self.grad_h_lstm_buf = ctx.enqueue_create_buffer[dtype](LSTM_HC_SIZE)
        self.grad_c_lstm_buf = ctx.enqueue_create_buffer[dtype](LSTM_HC_SIZE)

        self.grad_mlp_in_step_buf = ctx.enqueue_create_buffer[dtype](
            Self.BATCH * LSTM_HIDDEN
        )
        self.grad_x_lstm_buf = ctx.enqueue_create_buffer[dtype](
            Self.BATCH * Self.LATENT
        )
        self.grad_h_prev_lstm_buf = ctx.enqueue_create_buffer[dtype](
            Self.BATCH * LSTM_HIDDEN
        )
        self.grad_c_prev_lstm_buf = ctx.enqueue_create_buffer[dtype](
            Self.BATCH * LSTM_HIDDEN
        )
        self.lstm_d_combined_ws_buf = ctx.enqueue_create_buffer[dtype](
            Self.BATCH * 4 * LSTM_HIDDEN
        )

        self.cum_rewards_buf = ctx.enqueue_create_buffer[dtype](
            Self.BATCH * Self.K
        )
        self.cum_rewards_host = ctx.enqueue_create_host_buffer[dtype](
            Self.BATCH * Self.K
        )

        # GPU sampling scratch — sized to `Config.buffer_capacity`
        # (matches the comptime CAP in `train_step_gpu`). Allocated
        # unconditionally so the move ctor / upload paths don't have a
        # comptime branch; the cost scales linearly with CAP (~12 bytes
        # per slot) which is rounding error vs the network-state buffers
        # at any reasonable buffer size. 2026-05-16: was hardcoded 50000
        # regardless of Config — now reads Config.buffer_capacity.
        comptime CAP = Self.Config.buffer_capacity
        self.cum_prio_buf = ctx.enqueue_create_buffer[dtype](CAP)
        self.cand_starts_buf = ctx.enqueue_create_buffer[DType.int32](CAP)
        self.n_valid_buf = ctx.enqueue_create_buffer[DType.int32](1)
        self.total_prio_buf = ctx.enqueue_create_buffer[dtype](1)
        self.batch_start_idx_buf = ctx.enqueue_create_buffer[DType.int32](
            Self.BATCH
        )
        self.batch_start_idx_host = ctx.enqueue_create_host_buffer[DType.int32](
            Self.BATCH
        )

    def __init__(out self, *, deinit take: Self):
        """Move constructor."""
        self.representation = take.representation^
        self.dynamics = take.dynamics^
        self.prediction = take.prediction^
        self.projector = take.projector^
        self.predictor = take.predictor^
        self.representation_target = take.representation_target^
        self.dynamics_target = take.dynamics_target^
        self.prediction_target = take.prediction_target^
        self.batch_obs_buf = take.batch_obs_buf^
        self.batch_actions_buf = take.batch_actions_buf^
        self.batch_rewards_buf = take.batch_rewards_buf^
        self.batch_mcts_pol_buf = take.batch_mcts_pol_buf^
        self.batch_mcts_val_buf = take.batch_mcts_val_buf^
        self.batch_age_buf = take.batch_age_buf^
        self.batch_mcts_samp_act_buf = take.batch_mcts_samp_act_buf^
        self.batch_mcts_imp_pi_buf = take.batch_mcts_imp_pi_buf^
        self.hidden_buf = take.hidden_buf^
        self.dyn_out_buf = take.dyn_out_buf^
        self.pred_out_buf = take.pred_out_buf^
        self.rep_input_buf = take.rep_input_buf^
        self.dyn_input_buf = take.dyn_input_buf^
        self.tgt_rep_input_buf = take.tgt_rep_input_buf^
        self.tgt_z_buf = take.tgt_z_buf^
        self.tgt_pred_out_buf = take.tgt_pred_out_buf^
        self.boot_v_buf = take.boot_v_buf^
        self.obs_step_buf = take.obs_step_buf^
        self.rep_obs_buf = take.rep_obs_buf^
        self.proj_dyn_buf = take.proj_dyn_buf^
        self.pred_dyn_buf = take.pred_dyn_buf^
        self.proj_obs_buf = take.proj_obs_buf^
        self.rep_cache_buf = take.rep_cache_buf^
        self.dyn_caches_buf = take.dyn_caches_buf^
        self.pred_caches_buf = take.pred_caches_buf^
        self.proj_dyn_caches_buf = take.proj_dyn_caches_buf^
        self.pred_dyn_caches_buf = take.pred_dyn_caches_buf^
        self.rep_obs_cache_buf = take.rep_obs_cache_buf^
        self.proj_obs_cache_buf = take.proj_obs_cache_buf^
        self.grad_pred_out_buf = take.grad_pred_out_buf^
        self.grad_dyn_out_buf = take.grad_dyn_out_buf^
        self.grad_pred_dyn_buf = take.grad_pred_dyn_buf^
        self.grad_hidden_buf = take.grad_hidden_buf^
        self.grad_pred_in_step_buf = take.grad_pred_in_step_buf^
        self.grad_predr_in_step_buf = take.grad_predr_in_step_buf^
        self.grad_proj_in_step_buf = take.grad_proj_in_step_buf^
        self.grad_dyn_out_step_buf = take.grad_dyn_out_step_buf^
        self.grad_dyn_in_step_buf = take.grad_dyn_in_step_buf^
        self.grad_rep_in_buf = take.grad_rep_in_buf^
        self.value_target_full_buf = take.value_target_full_buf^
        self.value_target_scalar_buf = take.value_target_scalar_buf^
        self.value_target_dist_buf = take.value_target_dist_buf^
        self.reward_target_scalar_buf = take.reward_target_scalar_buf^
        self.reward_target_dist_buf = take.reward_target_dist_buf^
        self.policy_target_step_buf = take.policy_target_step_buf^
        self.fullpi_target_actions_step_buf = (
            take.fullpi_target_actions_step_buf^
        )
        self.fullpi_target_policy_step_buf = take.fullpi_target_policy_step_buf^
        self.L_R_buf = take.L_R_buf^
        self.L_P_buf = take.L_P_buf^
        self.L_V_buf = take.L_V_buf^
        self.L_G_buf = take.L_G_buf^
        self.z_var_buf = take.z_var_buf^
        self.v_pred_var_buf = take.v_pred_var_buf^
        self.per_sample_loss_scratch_buf = take.per_sample_loss_scratch_buf^
        self.per_sample_v_loss_k0_buf = take.per_sample_v_loss_k0_buf^
        self.priorities_out_buf = take.priorities_out_buf^
        self.workspace_buf = take.workspace_buf^
        self.grad_clip_ps = take.grad_clip_ps^
        self.batch_obs_host = take.batch_obs_host^
        self.batch_actions_host = take.batch_actions_host^
        self.batch_rewards_host = take.batch_rewards_host^
        self.batch_mcts_pol_host = take.batch_mcts_pol_host^
        self.batch_mcts_val_host = take.batch_mcts_val_host^
        self.batch_age_host = take.batch_age_host^
        self.batch_mcts_samp_act_host = take.batch_mcts_samp_act_host^
        self.batch_mcts_imp_pi_host = take.batch_mcts_imp_pi_host^
        self.value_target_full_host = take.value_target_full_host^
        self.L_R_host = take.L_R_host^
        self.L_P_host = take.L_P_host^
        self.L_V_host = take.L_V_host^
        self.L_G_host = take.L_G_host^
        self.priorities_out_host = take.priorities_out_host^
        self.z_var_host = take.z_var_host^
        self.v_pred_var_host = take.v_pred_var_host^
        self.lstm_params_buf = take.lstm_params_buf^
        self.lstm_grads_buf = take.lstm_grads_buf^
        self.lstm_opt_state_buf = take.lstm_opt_state_buf^
        self.lstm_opt_global_buf = take.lstm_opt_global_buf^
        self.lstm_params_host = take.lstm_params_host^
        self.lstm_opt_state_host = take.lstm_opt_state_host^
        self.lstm_opt_global_host = take.lstm_opt_global_host^
        self.lstm_step_num = take.lstm_step_num
        self.reward_prefix_mlp_gpu = take.reward_prefix_mlp_gpu^
        self.lstm_h_states_buf = take.lstm_h_states_buf^
        self.lstm_c_states_buf = take.lstm_c_states_buf^
        self.lstm_caches_buf = take.lstm_caches_buf^
        self.mlp_head_caches_buf = take.mlp_head_caches_buf^
        self.lstm_h_input_buf = take.lstm_h_input_buf^
        self.lstm_c_input_buf = take.lstm_c_input_buf^
        self.rew_pref_logits_buf = take.rew_pref_logits_buf^
        self.grad_rew_pref_logits_buf = take.grad_rew_pref_logits_buf^
        self.rew_pref_target_dist_buf = take.rew_pref_target_dist_buf^
        self.grad_h_lstm_buf = take.grad_h_lstm_buf^
        self.grad_c_lstm_buf = take.grad_c_lstm_buf^
        self.grad_mlp_in_step_buf = take.grad_mlp_in_step_buf^
        self.grad_x_lstm_buf = take.grad_x_lstm_buf^
        self.grad_h_prev_lstm_buf = take.grad_h_prev_lstm_buf^
        self.grad_c_prev_lstm_buf = take.grad_c_prev_lstm_buf^
        self.lstm_d_combined_ws_buf = take.lstm_d_combined_ws_buf^
        self.cum_rewards_buf = take.cum_rewards_buf^
        self.cum_rewards_host = take.cum_rewards_host^
        self.cum_prio_buf = take.cum_prio_buf^
        self.cand_starts_buf = take.cand_starts_buf^
        self.n_valid_buf = take.n_valid_buf^
        self.total_prio_buf = take.total_prio_buf^
        self.batch_start_idx_buf = take.batch_start_idx_buf^
        self.batch_start_idx_host = take.batch_start_idx_host^

    # ══════════════════════════════════════════════════════════════════════
    # CPU → GPU upload
    # ══════════════════════════════════════════════════════════════════════

    def upload_from(
        mut self,
        cpu: EZV2DiscreteCPUState[Self.Config, Self.Config.buffer_capacity],
        ctx: DeviceContext,
    ) raises:
        """Copy network params/optimizer state/model state from the
        CPU-resident `EZV2DiscreteCPUState` into this GPU state. The CPU
        state remains the source of truth for the replay buffer, MCTS
        targets, and per-transition priorities; only the trainable
        weights are mirrored on device. Call once at training start, and
        again after a host-side `update_target_networks` if the agent
        ever decides to seed GPU weights from a CPU snapshot."""
        self.representation.upload_from(cpu.representation, ctx)
        self.dynamics.upload_from(cpu.dynamics, ctx)
        self.prediction.upload_from(cpu.prediction, ctx)
        self.projector.upload_from(cpu.projector, ctx)
        self.predictor.upload_from(cpu.predictor, ctx)

        # ── Reward-prefix LSTM head ─────────────────────────────────────
        # Always upload (the host-side LSTM/MLP head buffers exist
        # regardless of `use_reward_prefix`, just untouched when False).
        comptime LSTM_PS = Self._LSTM_PS
        comptime LSTM_OPT_STATE_SIZE = (
            LSTM_PS * Self.Config.OptType.STATE_PER_PARAM
        )
        comptime OPT_GLOBAL_SIZE = Self.Config.OptType.GLOBAL_STATE_SIZE

        for i in range(LSTM_PS):
            self.lstm_params_host[i] = (cpu.lstm_params + i)[]
        ctx.enqueue_copy(self.lstm_params_buf, self.lstm_params_host)

        for i in range(LSTM_OPT_STATE_SIZE):
            self.lstm_opt_state_host[i] = (cpu.lstm_opt_state + i)[]
        ctx.enqueue_copy(self.lstm_opt_state_buf, self.lstm_opt_state_host)

        # CPU LSTM has no opt-global mirror; seed device opt-global to
        # zero (the lr_scale slot defaults to 1.0 via `set_lr_scale` for
        # the GPU networks above; the LSTM's separate global state stays
        # zero — Adam's bias correction reads slot 0 as the step counter
        # which `optimizer_step` bumps each call.) For graph-safety we
        # zero on init and let the kernel manage from there.
        for i in range(OPT_GLOBAL_SIZE):
            self.lstm_opt_global_host[i] = Scalar[dtype](0.0)
        ctx.enqueue_copy(self.lstm_opt_global_buf, self.lstm_opt_global_host)
        self.lstm_step_num = 0

        self.reward_prefix_mlp_gpu.upload_from(cpu.reward_prefix_mlp, ctx)

    def upload_targets_from(
        mut self,
        cpu: EZV2DiscreteCPUState[Self.Config, Self.Config.buffer_capacity],
        ctx: DeviceContext,
    ) raises:
        """Mirror the CPU target nets (`representation_target`,
        `dynamics_target`, `prediction_target`) onto the GPU. Call at
        training start (right after `upload_from`) and again at every
        `target_sync_interval` step in the training loop — same cadence
        as the CPU `update_target_networks(tau=1.0)` call that refreshes
        the CPU targets.

        Only the trainable params + optimizer state + model state are
        copied; SimSiam projector/predictor have no target copy (paper:
        target nets are rep/dyn/pred only).
        """
        self.representation_target.upload_from(cpu.representation_target, ctx)
        self.dynamics_target.upload_from(cpu.dynamics_target, ctx)
        self.prediction_target.upload_from(cpu.prediction_target, ctx)

    def download_to(
        mut self,
        mut cpu: EZV2DiscreteCPUState[Self.Config, Self.Config.buffer_capacity],
        ctx: DeviceContext,
    ) raises:
        """Mirror the GPU networks back to the CPU state. Used after a
        burst of `train_step_gpu` calls so subsequent CPU-side
        operations (Gumbel search at action-selection time, reanalyze
        worker, etc.) see fresh weights."""
        self.representation.download_to(cpu.representation, ctx)
        self.dynamics.download_to(cpu.dynamics, ctx)
        self.prediction.download_to(cpu.prediction, ctx)
        self.projector.download_to(cpu.projector, ctx)
        self.predictor.download_to(cpu.predictor, ctx)

        # ── Reward-prefix LSTM head ─────────────────────────────────────
        comptime LSTM_PS = Self._LSTM_PS
        comptime LSTM_OPT_STATE_SIZE = (
            LSTM_PS * Self.Config.OptType.STATE_PER_PARAM
        )
        ctx.enqueue_copy(self.lstm_params_host, self.lstm_params_buf)
        ctx.enqueue_copy(self.lstm_opt_state_host, self.lstm_opt_state_buf)
        ctx.synchronize()
        for i in range(LSTM_PS):
            (cpu.lstm_params + i)[] = self.lstm_params_host[i]
        for i in range(LSTM_OPT_STATE_SIZE):
            (cpu.lstm_opt_state + i)[] = self.lstm_opt_state_host[i]

        self.reward_prefix_mlp_gpu.download_to(cpu.reward_prefix_mlp, ctx)

    # ══════════════════════════════════════════════════════════════════════
    # GPU Gumbel-search wrapper
    # ══════════════════════════════════════════════════════════════════════
    #
    # Thin pass-through to `run_gumbel_search_gpu`. Lives here (and not at
    # the call site) because Mojo nightly's type-checker fails to unify the
    # alias form `EZV2DiscreteMLPConfig[…].RepModel` (used in `gpu.representation`)
    # with the literal `Sequential[AutoFused[…], …]` form that an external
    # type-parameter binding `Config.RepModel` resolves to. Inside this
    # method, both `Self.Config.RepModel` and `self.representation`'s field
    # type spell the same alias, so unification succeeds.

    def mcts_search[
        N_ENVS: Int,
        NODES: Int,
        MAX_K: Int,
        SIMS: Int,
    ](
        mut self,
        ctx: DeviceContext,
        mut mcts_state: EZV2GPUMCTSState[
            N_ENVS, NODES, Self.ACT, Self.LATENT, Self.BINS, MAX_K
        ],
        obs_buf: DeviceBuffer[dtype],
        workspace_buf: DeviceBuffer[dtype],
        v_min: Float64,
        v_max: Float64,
        gamma: Float64 = 0.997,
        rng_seed: UInt32 = UInt32(0),
        apply_legal: Bool = False,
        k_actual: Int = MAX_K,
        c_visit: Float64 = 50.0,
        c_scale: Float64 = 0.1,
    ) raises:
        """Drive `run_gumbel_search_gpu` over the on-device networks. After
        return, `mcts_state.policies_out` holds the improved policy
        distribution and `mcts_state.visit_count` / `total_value` at the
        root nodes can be downloaded for SVE.
        """
        run_gumbel_search_gpu[
            N_ENVS,
            NODES,
            Self.ACT,
            Self.LATENT,
            Self.BINS,
            MAX_K,
            SIMS,
            Self.Config.RepModel,
            Self.Config.DynModel,
            Self.Config.PredModel,
            Self.Config.OptType,
            Self.Config.OptType,
            Self.Config.OptType,
        ](
            ctx,
            mcts_state,
            obs_buf,
            self.representation,
            self.dynamics,
            self.prediction,
            workspace_buf,
            v_min=v_min,
            v_max=v_max,
            apply_legal=apply_legal,
            k_actual=k_actual,
            c_visit=c_visit,
            c_scale=c_scale,
            gamma=gamma,
            rng_seed=rng_seed,
        )
