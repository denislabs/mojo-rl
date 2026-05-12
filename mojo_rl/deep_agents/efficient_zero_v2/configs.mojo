"""EfficientZero V2 configs — trait + MLP variant.

`EZV2DiscreteConfig` extends `MuZeroConfig` with everything EZ-V2 adds on
top of MuZero:

  • SimSiam-style consistency networks: `ProjectorModel` (representation
    → 1024-d projection) and `PredictorModel` (1024 → 512 → 1024
    asymmetric bottleneck). Both are Model-conforming Sequentials assembled
    from the composites in `efficient_zero_v2/networks.mojo`.

  • Loss weights from paper Eq. 3 / Table 3:

        L = λ_R·L_R + λ_P·L_P + λ_V·L_V + λ_G·L_G  +  λ_H·H[π]

    with defaults λ_R=1.0, λ_P=1.0, λ_V=0.25, λ_G=2.0, λ_H=5e-3.

  • Mixed-value-target staleness thresholds `t_fresh`, `t_stale` (paper
    Eq. 16 + Table 3 defaults 20000 / 40000 train steps).

The trait inherits from `MuZeroConfig` so any `EZV2DiscreteConfig` can be
fed to MuZero infrastructure that wants a `MuZeroConfig` (e.g.
`MuZeroCPUState`, the GumbelMCTS search machinery on the agent's online
networks). The reverse is *not* true — a MuZeroConfig is missing the
projector/predictor + loss weights.

`EZV2DiscreteMLPConfig` is the standalone-MLP variant suitable for clean
state-based environments (CartPole, classic control, DMC proprio). Atari
CNN + DMC vision variants are deferred to Phase 4.

Reward-prefix LSTM head intentionally absent — Phase 2 risk register
(`docs/EFFICIENTZERO_V2_PLAN.md`) defers it until after CartPole converges
with a plain reward head.
"""

from mojo_rl.nn.model import (
    Model,
    Linear,
    LinearReLU,
    LinearMish,
    LayerNorm,
    MinMaxNorm,
    Sequential,
    Parallel,
    Identity,
    ReLU,
)
from mojo_rl.nn.autodiff.combinators import SplitApply
from mojo_rl.nn.optimizer import Optimizer, Adam
from mojo_rl.deep_agents.muzero.configs import MuZeroConfig
from mojo_rl.deep_agents.muzero.strategies import (
    SearchMode,
    LearnedDynamics,
    ValueEncoding,
    CategoricalEncoding,
    HiddenScaling,
    MinMaxScale,
    ExplorationNoise,
    DirichletNoise,
    PUCTFormula,
    AlphaGoPUCT,
    BackupMode,
    NStepBootstrap,
    PlayerMode,
    SinglePlayer,
)
from mojo_rl.deep_agents.efficient_zero_v2.networks import (
    ProjectionMLP,
    PredictionMLP,
    ActionEmbedding,
)
from mojo_rl.deep_agents.efficient_zero_v2.action_space import (
    ActionSpace,
    DiscreteActionSpace,
    ContinuousActionSpace,
)


# ═════════════════════════════════════════════════════════════════════════
# Value-target mode constants (paper Eq. 16, EZ-V2 reference
# `value_target` config field). Mode is comptime — pick once per agent.
#
#   VALUE_TARGET_SEARCH = 0  → pure stored MCTS root value (`sve`).
#                              Default. The bootstrap (boot_v) is not
#                              computed at all in this mode. This was
#                              EZ-V2's original "search" value-target
#                              mode and the de-facto behaviour of the
#                              agent before Lever 1 was wired in.
#
#   VALUE_TARGET_SARSA  = 1  → pure n-step TD with **fresh** target-net
#                              bootstrap (Lever 1, EZ-V2 paper App. A.4).
#                              `boot_v[k+n_eff]` from a forward through
#                              `representation_target + prediction_target`
#                              on `o_{t+k+n_eff}` replaces the stored MCTS
#                              value at the bootstrap position.
#
#   VALUE_TARGET_MIXED  = 2  → blend SVE → SARSA based on transition age,
#                              gated by `t_fresh` / `t_stale`. Matches
#                              `MixedValueTarget.compute(sve, td, age)`.
#                              Note: thresholds are inverted vs the EZ-V2
#                              reference's `value_target='mixed'` mode
#                              (which uses pure n-step TD for early
#                              training and blends in fresh search later).
#                              See work-unit 8 in
#                              `docs/EFFICIENTZERO_V2_PLAN.md` for the
#                              empirical rationale: at smoke configs the
#                              stored MCTS root carries more reward
#                              signal than a single value-head forward,
#                              so SVE is preferred while training is
#                              young and the value head is uninformative.
# ═════════════════════════════════════════════════════════════════════════

comptime VALUE_TARGET_SEARCH: Int = 0
comptime VALUE_TARGET_SARSA: Int = 1
comptime VALUE_TARGET_MIXED: Int = 2


# ═════════════════════════════════════════════════════════════════════════
# Config trait
# ═════════════════════════════════════════════════════════════════════════


trait EZV2DiscreteConfig(MuZeroConfig):
    """Compile-time configuration for EfficientZero V2 (discrete actions).

    Extends `MuZeroConfig` with EZ-V2 specifics: SimSiam consistency
    networks, paper-Eq.-3 loss weights, and mixed-value-target staleness
    thresholds. Strategy types (ValueTarget / PolicyLoss) live in
    `efficient_zero_v2/strategies.mojo` and are wired in directly at the
    agent training loop, not via this trait — they need different bound
    parameters per dispatch site so binding them here would force every
    config to commit to one set.
    """

    # ── SimSiam-style consistency networks ────────────────────────────────
    comptime ProjectorModel: Model
    """Projection MLP for the SimSiam consistency loss.
    IN_DIM = latent_dim, OUT_DIM = proj_dim."""

    comptime PredictorModel: Model
    """Predictor MLP applied only on the dynamics branch (asymmetric bottleneck).
    IN_DIM = OUT_DIM = proj_dim."""

    comptime proj_dim: Int
    """Projection space dimension (paper default 1024)."""

    # ── Gumbel-search hyperparameters (in addition to MuZeroConfig) ──────
    comptime num_root_candidates: Int
    """K candidates sampled at the root via Gumbel-Top-k. Must be ≤
    action_dim and a power of two; the search machinery clips at runtime
    if not."""

    # ── Action-space dispatch (per `docs/EZV2_MODULAR_ARCHITECTURE.md`) ──
    comptime ActSpace: ActionSpace
    """Carries the policy-head loss/grad kernel hook plus dimensional
    knobs (`POLICY_OUT_DIM`, `POLICY_TARGET_DIM`, `K_ROOT`). For discrete
    configs, set this to `DiscreteActionSpace[ACT, K_GUMBEL]` —
    `ActSpace.K_ROOT` then mirrors `num_root_candidates`. The continuous
    EZ-V2 variant (Phase 3) supplies `ContinuousActionSpace[...]`
    instead, with a different `policy_loss_grad_gpu` kernel."""

    # ── Loss weights (paper Eq. 3 + entropy regularizer) ─────────────────
    comptime lambda_reward: Float64
    comptime lambda_policy: Float64
    comptime lambda_value: Float64
    comptime lambda_consistency: Float64
    comptime entropy_weight: Float64

    # ── Value-target mode (paper Eq. 16, EZ-V2 reference `value_target`) ─
    comptime value_target_mode: Int
    """One of `VALUE_TARGET_SEARCH` (0), `VALUE_TARGET_SARSA` (1), or
    `VALUE_TARGET_MIXED` (2). See module docstring for semantics. Defaults
    to SEARCH so existing agents keep their behaviour. Only `t_fresh`/
    `t_stale` are consulted when mode == MIXED; SARSA always uses the
    fresh target-net bootstrap and SEARCH never computes it."""

    # ── Mixed-value-target staleness thresholds (paper Eq. 16) ───────────
    comptime t_fresh: Int
    """Below this train-step age use pure SVE (paper default 20000)."""

    comptime t_stale: Int
    """Above this train-step age use pure n-step TD (paper default 40000).
    Linear blend in between."""

    # ── Reward-prefix LSTM head (EZ-V1 carry-over, paper App. G) ─────────
    # When `use_reward_prefix=True`, the per-step reward CE through the
    # dynamics network's reward head is replaced with a CE on
    #     reward_prefix_logits[k] = MLP_head( LSTM(hidden[k+1]) )
    # against `two_hot(scalar_transform( Σ_{j=0..k} reward[j] ))`. The
    # LSTM state resets to zero every `lstm_horizon_len` unroll steps to
    # cap BPTT depth. When `use_reward_prefix=False` the head buffers are
    # still allocated (small footprint) but no gradient flows through
    # them — the existing per-step reward CE through the dyn-network's
    # reward output stays the loss.
    comptime use_reward_prefix: Bool
    comptime lstm_hidden: Int
    """LSTM hidden / cell state dim. Paper App. G default 64."""

    comptime lstm_horizon_len: Int
    """Number of unroll steps before resetting the LSTM (h, c) to zero.
    Paper App. G default 5. Caps BPTT depth."""

    comptime lstm_mlp_hidden: Int
    """Hidden dim of the post-LSTM MLP that maps h_lstm → reward-prefix
    logits. Paper App. G default 64."""

    # ── init_zero head parameter ranges ──────────────────────────────────
    # Computed from each config's concrete `PredModel` / `DynModel`
    # structure (Sequential / Parallel offsets) — read as plain Ints by
    # `GenericEZV2ContinuousAgent._init_zero_output_heads` to bypass the
    # trait-erasure that hides `Sequential.model_types` / `_param_offset`
    # when `PredModel` / `DynModel` are accessed through this trait.
    comptime pred_policy_head_param_start: Int
    """Index into `PredModel.params` where the policy head's params begin.
    Continuous: branch 0 of the trailing `Parallel[PolicyHead, ValueHead]`.
    """

    comptime pred_policy_head_param_size: Int
    """Length of the policy head's contiguous param slice in
    `PredModel.params`. Continuous: `branch_types[0].PARAM_SIZE` of the
    trailing Parallel."""

    comptime dyn_reward_head_param_start: Int
    """Index into `DynModel.params` where the reward head's params begin.
    Both discrete and continuous: branch 1 of the trailing
    `Parallel[NextLatent, RewardHead]`."""

    comptime dyn_reward_head_param_size: Int
    """Length of the reward head's contiguous param slice in
    `DynModel.params`. `branch_types[1].PARAM_SIZE` of the trailing
    Parallel."""


# ═════════════════════════════════════════════════════════════════════════
# MLP variant
# ═════════════════════════════════════════════════════════════════════════


struct EZV2DiscreteMLPConfig[
    OBS: Int,
    ACT: Int,
    LATENT: Int = 128,
    HIDDEN: Int = 128,
    PROJ: Int = 256,
    PRED_BOTTLENECK: Int = 128,
    BINS: Int = 51,
    LR: Float64 = 1e-3,
    WD: Float64 = 1e-4,
    CAP: Int = 50000,
    BS: Int = 64,
    K_UNROLL: Int = 5,
    N_TD: Int = 10,
    SIMS: Int = 32,
    NODES: Int = 64,
    K_GUMBEL: Int = 8,
    LAMBDA_R: Float64 = 1.0,
    LAMBDA_P: Float64 = 1.0,
    LAMBDA_V: Float64 = 0.25,
    LAMBDA_G: Float64 = 2.0,
    ENT_WEIGHT: Float64 = 5e-3,
    # Value-target mode. Default = SEARCH = pure stored MCTS root value.
    # Set to VALUE_TARGET_SARSA (1) to enable Lever 1 (fresh target-net
    # bootstrap for n-step TD), or VALUE_TARGET_MIXED (2) for the age-
    # gated blend.
    VALUE_TARGET_MODE: Int = VALUE_TARGET_SEARCH,
    T_FRESH: Int = 20000,
    T_STALE: Int = 40000,
    # Reward-prefix LSTM head (paper App. G). Off by default — the head
    # is wired into `train_step` only when `USE_REWARD_PREFIX=True`. Even
    # when off, the LSTM/MLP buffers are still allocated (small footprint)
    # so the state struct's field layout doesn't depend on the flag.
    USE_REWARD_PREFIX: Bool = False,
    LSTM_HIDDEN: Int = 64,
    LSTM_HORIZON_LEN: Int = 5,
    LSTM_MLP_HIDDEN: Int = 64,
](EZV2DiscreteConfig):
    """Standalone-MLP EZ-V2 for clean state-based observations.

    Network topology mirrors `MuZeroMLPConfig` for rep/dyn/pred — three
    `LinearMish` layers ending in `MinMaxNorm` for the rep + dyn-hidden
    branches (preserves MuZero's normalization discipline; LayerNorm is
    what the SimSiam projector adds on top). Defaults below are tuned for
    quick smoke tests; the paper-Table-3 numbers (PROJ=1024, BS=256, etc.)
    can be passed through the parameter list when running real training.
    """

    # ── MuZeroConfig fields ──────────────────────────────────────────────
    comptime NAME: String = "EZV2-MLP"

    comptime obs_dim: Int = Self.OBS
    comptime action_dim: Int = Self.ACT
    comptime latent_dim: Int = Self.LATENT
    comptime num_bins: Int = Self.BINS
    comptime DYN_IN: Int = Self.LATENT + Self.ACT
    comptime DYN_OUT: Int = Self.LATENT + Self.BINS
    comptime PRED_OUT: Int = Self.ACT + Self.BINS

    # Representation: obs → latent. Final MinMaxNorm matches MuZero's
    # autograd-aware MinMaxNorm pattern (Phase G post-mortem 2026-05-04 —
    # post-hoc scaling outside autograd lets pre-scale activations
    # explode). MinMaxNorm is followed by the projector at training time
    # (which adds its own LayerNorm).
    comptime RepModel = Sequential[
        LinearMish[Self.OBS, Self.HIDDEN],
        LinearMish[Self.HIDDEN, Self.HIDDEN],
        Linear[Self.HIDDEN, Self.LATENT],
        MinMaxNorm[Self.LATENT],
    ]

    # Dynamics: (latent, one-hot action) → (next_latent, reward_logits).
    comptime DynModel = Sequential[
        LinearMish[Self.DYN_IN, Self.HIDDEN],
        LinearMish[Self.HIDDEN, Self.HIDDEN],
        Parallel[
            # Latent branch: emits `delta_z` for the residual
            # `next_z = hidden + delta_z` applied externally in
            # `train_step_core` (see `ezv2_extract_hidden_after_dyn_kernel`).
            # LayerNorm[LATENT] (was MinMaxNorm) keeps delta_z mean=0,
            # std=1 so the K-step residual unroll has bounded magnitude
            # growth (~sqrt(K)). Reference uses no output norm and
            # ImproveResidualBlocks for stability — we use the simpler
            # LN-output approach. MinMaxNorm was load-bearing for collapse
            # (degenerate gradient near constant input + bounded [0,1]
            # output incompatible with residual stacking).
            Sequential[
                Linear[Self.HIDDEN, Self.LATENT],
                LayerNorm[Self.LATENT],
            ],
            Linear[Self.HIDDEN, Self.BINS],
        ],
    ]

    # Prediction f-net: latent → (policy_logits, value_logits).
    comptime PredModel = Sequential[
        LinearMish[Self.LATENT, Self.HIDDEN],
        Parallel[
            Linear[Self.HIDDEN, Self.ACT],
            Linear[Self.HIDDEN, Self.BINS],
        ],
    ]

    comptime OptType = Adam[LR=Self.LR, WEIGHT_DECAY=Self.WD]

    comptime batch_size: Int = Self.BS
    comptime buffer_capacity: Int = Self.CAP
    comptime unroll_steps: Int = Self.K_UNROLL
    comptime td_steps: Int = Self.N_TD

    comptime num_simulations: Int = Self.SIMS
    comptime max_nodes: Int = Self.NODES

    comptime Search = LearnedDynamics
    comptime Encoding = CategoricalEncoding
    comptime Scaling = MinMaxScale
    comptime Noise = DirichletNoise[0.25, 0.25]
    comptime PUCT = AlphaGoPUCT[1.0]
    comptime Backup = NStepBootstrap
    comptime Players = SinglePlayer

    comptime USE_REANALYZE: Bool = True

    # ── EZ-V2-specific fields ────────────────────────────────────────────
    comptime ProjectorModel = ProjectionMLP[
        HIDDEN=Self.LATENT, PROJ=Self.PROJ
    ]
    comptime PredictorModel = PredictionMLP[
        PROJ=Self.PROJ, BOTTLENECK=Self.PRED_BOTTLENECK
    ]
    comptime proj_dim: Int = Self.PROJ
    comptime num_root_candidates: Int = Self.K_GUMBEL
    comptime ActSpace = DiscreteActionSpace[Self.ACT, Self.K_GUMBEL]

    comptime lambda_reward: Float64 = Self.LAMBDA_R
    comptime lambda_policy: Float64 = Self.LAMBDA_P
    comptime lambda_value: Float64 = Self.LAMBDA_V
    comptime lambda_consistency: Float64 = Self.LAMBDA_G
    comptime entropy_weight: Float64 = Self.ENT_WEIGHT

    comptime value_target_mode: Int = Self.VALUE_TARGET_MODE
    comptime t_fresh: Int = Self.T_FRESH
    comptime t_stale: Int = Self.T_STALE

    # ── Reward-prefix LSTM head ──────────────────────────────────────────
    comptime use_reward_prefix: Bool = Self.USE_REWARD_PREFIX
    comptime lstm_hidden: Int = Self.LSTM_HIDDEN
    comptime lstm_horizon_len: Int = Self.LSTM_HORIZON_LEN
    comptime lstm_mlp_hidden: Int = Self.LSTM_MLP_HIDDEN

    # ── init_zero head parameter ranges ──────────────────────────────────
    comptime pred_policy_head_param_start: Int = (
        Self.PredModel._param_offset[Self.PredModel.N - 1]()
        + Self.PredModel.model_types[Self.PredModel.N - 1]._param_offset[0]()
    )
    comptime pred_policy_head_param_size: Int = (
        Self.PredModel.model_types[Self.PredModel.N - 1].branch_types[0].PARAM_SIZE
    )
    comptime dyn_reward_head_param_start: Int = (
        Self.DynModel._param_offset[Self.DynModel.N - 1]()
        + Self.DynModel.model_types[Self.DynModel.N - 1]._param_offset[1]()
    )
    comptime dyn_reward_head_param_size: Int = (
        Self.DynModel.model_types[Self.DynModel.N - 1].branch_types[1].PARAM_SIZE
    )


# ═════════════════════════════════════════════════════════════════════════
# Continuous MLP variant (Phase 3 — squashed-Gaussian policy head)
# ═════════════════════════════════════════════════════════════════════════
#
# Conforms to the same `EZV2DiscreteConfig` trait — the trait is action-
# space-agnostic at the field level (`Config.ActSpace` carries the
# discrete-vs-continuous dispatch). The "Discrete" in the trait name is a
# misnomer kept for git-history continuity; rename to `EZV2Config` is a
# follow-up rename when continuous landed and shaken down.
#
# Key differences from `EZV2DiscreteMLPConfig`:
#   • `action_dim` = real-vector dimension (not discrete count).
#   • `PredModel` outputs `2*ACT_DIM + BINS` (μ_raw ‖ σ_raw ‖ value bins).
#   • `ActSpace` = ContinuousActionSpace[ACT_DIM, K_ROOT, MAX, MIN_STD, ...].
#   • Replay buffer stores raw [ACT_DIM] action vectors per slot (the
#     dyn-input concat layer handles the action embedding implicitly —
#     see `docs/EZV2_CONTINUOUS_PHASE3.md` Option 2).


struct EZV2ContinuousMLPConfig[
    OBS: Int,
    ACT_DIM: Int,
    LATENT: Int = 256,
    HIDDEN: Int = 256,
    PROJ: Int = 1024,
    # SimSiam projector inner-hidden width. Reference
    # (`dmc_state.yaml: proj_hid_shape=512`) expands then contracts:
    # `LATENT → PROJ_HID → PROJ_HID → PROJ`. Defaults to `PROJ` (uniform
    # width) to preserve the original Pendulum baseline behavior; override
    # to ~512 on bigger envs (HalfCheetah) where the wider inner gives
    # the projector enough capacity to carry a non-trivial cosine
    # alignment that's also state-discriminative.
    PROJ_HID: Int = PROJ,
    PRED_BOTTLENECK: Int = 512,
    BINS: Int = 51,
    # Action embedding width (paper App. G / ez_dmc_state.py). Reference
    # uses 64. The action goes through Linear(ACT_DIM→ACT_EMBED) + LN +
    # ReLU inside the dyn network (via SplitApply), so the dyn first
    # linear receives [LATENT ‖ ACT_EMBED] instead of [LATENT ‖ ACT_DIM].
    # Critical for low-ACT_DIM envs (Pendulum, 1-dim) where the raw
    # action would otherwise have ~1.5% of dyn input variance.
    ACT_EMBED: Int = 64,
    LR: Float64 = 1e-3,
    WD: Float64 = 1e-4,
    CAP: Int = 50000,
    BS: Int = 64,
    K_UNROLL: Int = 5,
    N_TD: Int = 5,
    SIMS: Int = 32,
    NODES: Int = 64,
    K_ROOT: Int = 16,
    K_NON_ROOT: Int = 8,
    LAMBDA_R: Float64 = 1.0,
    LAMBDA_P: Float64 = 1.0,
    LAMBDA_V: Float64 = 0.25,
    LAMBDA_G: Float64 = 2.0,
    ENT_WEIGHT: Float64 = 5e-3,
    # Squashed-Gaussian hyperparameters (paper App. G).
    MAX_ACTION: Float64 = 1.0,
    MIN_STD: Float64 = 0.1,
    STD_MAGNIFICATION: Float64 = 3.0,
    # Sampled-Gumbel root sampling mode (see
    # `SampledGumbelMCTS.N_POLICY_AT_ROOT`). Default `K_ROOT` preserves
    # legacy magnified-policy behavior (Pendulum baseline). Set to e.g.
    # 4 with `K_ROOT=16` to enable reference DMC root sampling (4 from
    # policy + 12 uniform random in [-MAX_ACTION, MAX_ACTION]) which
    # decouples exploration from the current policy bias.
    N_POLICY_AT_ROOT: Int = K_ROOT,
    VALUE_TARGET_MODE: Int = VALUE_TARGET_SEARCH,
    T_FRESH: Int = 20000,
    T_STALE: Int = 40000,
    USE_REWARD_PREFIX: Bool = False,
    LSTM_HIDDEN: Int = 64,
    LSTM_HORIZON_LEN: Int = 5,
    LSTM_MLP_HIDDEN: Int = 64,
](EZV2DiscreteConfig):
    """Standalone-MLP EZ-V2 for continuous-action proprio environments.

    Defaults track paper Table 3 for DMC Proprio (Pendulum / HalfCheetah /
    etc.): LATENT=256, PROJ=1024, BINS=51, K_ROOT=16. Override via the
    parameter list when running smoke tests (smaller/faster) or full
    convergence (larger BS/PROJ).

    See `docs/EZV2_CONTINUOUS_PHASE3.md` for the design rationale.
    """

    # ── MuZeroConfig fields ──────────────────────────────────────────────
    comptime NAME: String = "EZV2-MLP-Continuous"

    comptime obs_dim: Int = Self.OBS
    # Continuous: action_dim is the real-vector dim. The buffer slot
    # width per transition matches: `[ACT_DIM]` floats per stored action
    # and per stored chosen-action policy target.
    comptime action_dim: Int = Self.ACT_DIM
    comptime latent_dim: Int = Self.LATENT
    comptime num_bins: Int = Self.BINS
    comptime DYN_IN: Int = Self.LATENT + Self.ACT_DIM
    comptime DYN_OUT: Int = Self.LATENT + Self.BINS
    # Pred output: μ_raw ‖ σ_raw ‖ value_bins.
    comptime PRED_OUT: Int = 2 * Self.ACT_DIM + Self.BINS

    # Representation: obs → latent.
    #
    # No output norm — matches reference `dmc_state.yaml: state_norm: False`
    # (`ez_dmc_state.py:180-190` returns post-ResBlock output with no final
    # squeeze). Earlier `MinMaxNorm[LATENT]` per-sample squashed obs into
    # `[0,1]^LATENT` and caused SimSiam projector collapse: even when
    # `L_V` showed encoder was state-discriminative (down to 0.28), the
    # projector could still find a constant-direction mapping that
    # satisfied SimSiam at cos≈1 (G → -1). Removing the output squash
    # preserves magnitude+direction info that the projector now has to
    # deliberately discard. Found 2026-05-13 after iterative SimSiam
    # debugging on HalfCheetah.
    comptime RepModel = Sequential[
        LinearMish[Self.OBS, Self.HIDDEN],
        LinearMish[Self.HIDDEN, Self.HIDDEN],
        Linear[Self.HIDDEN, Self.LATENT],
    ]

    # Dynamics: (latent, raw_action_vector) → (next_latent, reward_logits).
    # The dyn input from `build_dyn_input_kernel` is `[LATENT ‖ ACT_DIM]`
    # (same layout for all configs). For continuous, the dyn network's
    # first stage `SplitApply` keeps the LATENT slice unchanged and
    # routes the ACT_DIM slice through an `ActionEmbedding`
    # (`Linear[ACT_DIM, ACT_EMBED] → LayerNorm → ReLU`) — bringing the
    # action's share of the hidden-layer input to a meaningful fraction
    # (ACT_EMBED / (LATENT + ACT_EMBED)) instead of ACT_DIM / (LATENT +
    # ACT_DIM). Critical for ACT_DIM==1 envs (Pendulum). See
    # `docs/EZV2_CONTINUOUS_PHASE3_POSTMORTEM.md`.
    comptime DynModel = Sequential[
        SplitApply[
            Identity[Self.LATENT],
            ActionEmbedding[Self.ACT_DIM, Self.ACT_EMBED],
            Self.LATENT,
        ],
        LinearMish[Self.LATENT + Self.ACT_EMBED, Self.HIDDEN],
        LinearMish[Self.HIDDEN, Self.HIDDEN],
        Parallel[
            # Latent branch: emits `delta_z` for the residual
            # `next_z = hidden + delta_z` applied externally in
            # `train_step_core` (see `ezv2_extract_hidden_after_dyn_kernel`).
            # LayerNorm[LATENT] (was MinMaxNorm) keeps delta_z mean=0,
            # std=1 so the K-step residual unroll has bounded magnitude
            # growth (~sqrt(K)). Reference uses no output norm and
            # ImproveResidualBlocks for stability — we use the simpler
            # LN-output approach. MinMaxNorm was load-bearing for collapse
            # (degenerate gradient near constant input + bounded [0,1]
            # output incompatible with residual stacking).
            Sequential[
                Linear[Self.HIDDEN, Self.LATENT],
                LayerNorm[Self.LATENT],
            ],
            Linear[Self.HIDDEN, Self.BINS],
        ],
    ]

    # Prediction f-net: latent → (μ_raw ‖ σ_raw ‖ value_logits).
    # The 2*ACT_DIM policy outputs feed the squashed-Gaussian — see
    # `kernels.ezv2_policy_loss_grad_continuous_kernel`.
    comptime PredModel = Sequential[
        LinearMish[Self.LATENT, Self.HIDDEN],
        Parallel[
            Linear[Self.HIDDEN, 2 * Self.ACT_DIM],
            Linear[Self.HIDDEN, Self.BINS],
        ],
    ]

    comptime OptType = Adam[LR=Self.LR, WEIGHT_DECAY=Self.WD]

    comptime batch_size: Int = Self.BS
    comptime buffer_capacity: Int = Self.CAP
    comptime unroll_steps: Int = Self.K_UNROLL
    comptime td_steps: Int = Self.N_TD

    comptime num_simulations: Int = Self.SIMS
    comptime max_nodes: Int = Self.NODES

    comptime Search = LearnedDynamics
    comptime Encoding = CategoricalEncoding
    comptime Scaling = MinMaxScale
    comptime Noise = DirichletNoise[0.25, 0.25]
    comptime PUCT = AlphaGoPUCT[1.0]
    comptime Backup = NStepBootstrap
    comptime Players = SinglePlayer

    comptime USE_REANALYZE: Bool = True

    # ── EZ-V2-specific fields ────────────────────────────────────────────
    comptime ProjectorModel = ProjectionMLP[
        HIDDEN=Self.LATENT, PROJ=Self.PROJ, PROJ_HID=Self.PROJ_HID
    ]
    comptime PredictorModel = PredictionMLP[
        PROJ=Self.PROJ, BOTTLENECK=Self.PRED_BOTTLENECK
    ]
    comptime proj_dim: Int = Self.PROJ
    comptime num_root_candidates: Int = Self.K_ROOT
    comptime ActSpace = ContinuousActionSpace[
        Self.ACT_DIM,
        Self.K_ROOT,
        Self.MAX_ACTION,
        Self.MIN_STD,
        Self.STD_MAGNIFICATION,
        Self.N_POLICY_AT_ROOT,
    ]

    comptime lambda_reward: Float64 = Self.LAMBDA_R
    comptime lambda_policy: Float64 = Self.LAMBDA_P
    comptime lambda_value: Float64 = Self.LAMBDA_V
    comptime lambda_consistency: Float64 = Self.LAMBDA_G
    comptime entropy_weight: Float64 = Self.ENT_WEIGHT

    comptime value_target_mode: Int = Self.VALUE_TARGET_MODE
    comptime t_fresh: Int = Self.T_FRESH
    comptime t_stale: Int = Self.T_STALE

    # ── Reward-prefix LSTM head ──────────────────────────────────────────
    comptime use_reward_prefix: Bool = Self.USE_REWARD_PREFIX
    comptime lstm_hidden: Int = Self.LSTM_HIDDEN
    comptime lstm_horizon_len: Int = Self.LSTM_HORIZON_LEN
    comptime lstm_mlp_hidden: Int = Self.LSTM_MLP_HIDDEN

    # ── init_zero head parameter ranges ──────────────────────────────────
    # Same `Sequential[..., Parallel[A, B]]` shape as the discrete config
    # — branch 0 of the trailing Parallel is the policy head, branch 1 is
    # the value head (PredModel) / reward head (DynModel).
    comptime pred_policy_head_param_start: Int = (
        Self.PredModel._param_offset[Self.PredModel.N - 1]()
        + Self.PredModel.model_types[Self.PredModel.N - 1]._param_offset[0]()
    )
    comptime pred_policy_head_param_size: Int = (
        Self.PredModel.model_types[Self.PredModel.N - 1].branch_types[0].PARAM_SIZE
    )
    comptime dyn_reward_head_param_start: Int = (
        Self.DynModel._param_offset[Self.DynModel.N - 1]()
        + Self.DynModel.model_types[Self.DynModel.N - 1]._param_offset[1]()
    )
    comptime dyn_reward_head_param_size: Int = (
        Self.DynModel.model_types[Self.DynModel.N - 1].branch_types[1].PARAM_SIZE
    )
