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
)
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
            Sequential[
                Linear[Self.HIDDEN, Self.LATENT],
                MinMaxNorm[Self.LATENT],
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
