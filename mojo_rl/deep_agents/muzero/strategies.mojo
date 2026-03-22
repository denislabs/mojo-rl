"""MuZero Strategy Traits — Composable building blocks for MuZero variants.

Follows the same strategy pattern as DQN/SAC (core/strategies/).
Each trait defines a behavior axis that varies across MuZero family algorithms.
Configs bundle strategy choices to define complete algorithm variants.

Strategy traits:
  - SearchMode: Learned dynamics (MuZero) vs true game rules (AlphaZero)
  - ValueEncoding: How values/rewards are encoded (Categorical, Scalar)
  - HiddenScaling: Hidden state normalization (MinMax, None)
  - ExplorationNoise: Root exploration (Dirichlet, Epsilon, None)
  - PUCTFormula: UCB exploration formula (MuZero, AlphaGo, UCB1)
  - BackupMode: Return computation (NStep, MonteCarlo, Lambda)
"""

from std.math import sqrt, log


# ═══════════════════════════════════════════════════════════════════════════
# SearchMode — Learned dynamics vs true game rules
# ═══════════════════════════════════════════════════════════════════════════


trait SearchMode:
    """Determines whether MCTS uses learned dynamics or true game rules.

    MuZero: Uses learned representation + dynamics + prediction networks.
    AlphaZero: Uses true game rules for state transitions, only learns
               policy + value networks (no dynamics model needed).
    """

    comptime USE_LEARNED_DYNAMICS: Bool
    """True for MuZero (learned model), False for AlphaZero (true rules)."""

    comptime NEEDS_REPRESENTATION: Bool
    """True if observations need encoding to latent space (MuZero).
    False if search operates on raw game state (AlphaZero)."""

    comptime NEEDS_REWARD_HEAD: Bool
    """True if reward must be predicted (MuZero).
    False if reward comes from game rules (AlphaZero)."""


struct LearnedDynamics(SearchMode):
    """MuZero-style: learn dynamics from data, search in latent space."""

    comptime USE_LEARNED_DYNAMICS: Bool = True
    comptime NEEDS_REPRESENTATION: Bool = True
    comptime NEEDS_REWARD_HEAD: Bool = True


struct TrueGameRules(SearchMode):
    """AlphaZero-style: use true game simulator, search in observation space.

    When using this mode:
    - RepModel maps obs -> latent for policy/value prediction only
    - DynModel is unused (game rules provide next state)
    - PredModel still predicts policy and value from game state
    """

    comptime USE_LEARNED_DYNAMICS: Bool = False
    comptime NEEDS_REPRESENTATION: Bool = False
    comptime NEEDS_REWARD_HEAD: Bool = False


# ═══════════════════════════════════════════════════════════════════════════
# ValueEncoding — How values and rewards are encoded
# ═══════════════════════════════════════════════════════════════════════════


trait ValueEncoding:
    """Determines how scalar values/rewards are encoded for network I/O.

    Categorical: Distributional encoding with NUM_BINS support bins +
                 two-hot encoding. Most stable for large value ranges.
    Scalar: Direct scalar prediction. Simpler but less stable.
    """

    comptime IS_DISTRIBUTIONAL: Bool
    """True for categorical/distributional encoding, False for scalar."""

    comptime USE_SCALAR_TRANSFORM: Bool
    """True to apply h(x) = sign(x)(sqrt(|x|+1)-1)+eps*x before encoding."""


struct CategoricalEncoding(ValueEncoding):
    """Distributional encoding with categorical support (default MuZero).

    Values are encoded as soft two-hot distributions over NUM_BINS bins.
    Most stable for large value ranges (Atari scores up to 10K+).
    """

    comptime IS_DISTRIBUTIONAL: Bool = True
    comptime USE_SCALAR_TRANSFORM: Bool = True


struct ScalarEncoding(ValueEncoding):
    """Direct scalar value prediction (simpler, for bounded-reward envs).

    Values predicted directly as single float. Works well when rewards
    are naturally bounded (e.g., CartPole reward = 1.0 per step).
    No scalar transform needed since values are small.
    """

    comptime IS_DISTRIBUTIONAL: Bool = False
    comptime USE_SCALAR_TRANSFORM: Bool = False


struct SymlogEncoding(ValueEncoding):
    """Scalar prediction with symlog transform (DreamerV3-style).

    Uses scalar transform for compression but predicts single value
    instead of categorical distribution. Good middle ground.
    """

    comptime IS_DISTRIBUTIONAL: Bool = False
    comptime USE_SCALAR_TRANSFORM: Bool = True


# ═══════════════════════════════════════════════════════════════════════════
# HiddenScaling — Hidden state normalization
# ═══════════════════════════════════════════════════════════════════════════


trait HiddenScaling:
    """Determines how hidden states are normalized after dynamics.

    Prevents hidden state magnitudes from growing unboundedly through
    repeated dynamics applications.
    """

    comptime ENABLED: Bool
    """Whether to apply scaling after each dynamics step."""

    comptime SCALE_METHOD: Int
    """0=MinMax [0,1], 1=LayerNorm, 2=SimNorm."""


struct MinMaxScale(HiddenScaling):
    """Min-max normalization to [0, 1] (default MuZero).

    Fast, no parameters. Each hidden state vector independently
    normalized: h = (h - min(h)) / (max(h) - min(h)).
    """

    comptime ENABLED: Bool = True
    comptime SCALE_METHOD: Int = 0


struct NoScale(HiddenScaling):
    """No hidden state scaling.

    Relies on network initialization and gradient clipping
    to keep hidden state magnitudes bounded.
    """

    comptime ENABLED: Bool = False
    comptime SCALE_METHOD: Int = 0


# ═══════════════════════════════════════════════════════════════════════════
# ExplorationNoise — Root prior exploration
# ═══════════════════════════════════════════════════════════════════════════


trait ExplorationNoise:
    """Determines how exploration noise is added to the root prior.

    Ensures MCTS explores diverse actions at the root, preventing
    the search from collapsing to a single action early.
    """

    comptime NOISE_TYPE: Int
    """0=Dirichlet, 1=Uniform epsilon, 2=None."""

    comptime NOISE_FRACTION: Float64
    """Fraction of noise mixed into the prior: p = (1-f)*prior + f*noise."""

    comptime NOISE_ALPHA: Float64
    """Dirichlet alpha parameter (only used when NOISE_TYPE=0).
    Smaller alpha = more concentrated noise (good for many actions).
    Typical: 0.03 (Go), 0.3 (Chess), 0.25 (Atari)."""


struct DirichletNoise[
    fraction: Float64 = 0.25,
    alpha: Float64 = 0.25,
](ExplorationNoise):
    """Dirichlet noise (default MuZero/AlphaZero).

    Samples noise from Dirichlet(alpha) distribution, mixes with prior.
    Alpha should scale inversely with action space size.
    """

    comptime NOISE_TYPE: Int = 0
    comptime NOISE_FRACTION: Float64 = Self.fraction
    comptime NOISE_ALPHA: Float64 = Self.alpha


struct EpsilonNoise[
    fraction: Float64 = 0.25,
](ExplorationNoise):
    """Uniform epsilon noise — simpler alternative to Dirichlet.

    With probability `fraction`, replaces prior with uniform distribution.
    Less sophisticated but easier to tune.
    """

    comptime NOISE_TYPE: Int = 1
    comptime NOISE_FRACTION: Float64 = Self.fraction
    comptime NOISE_ALPHA: Float64 = 0.0


struct NoNoise(ExplorationNoise):
    """No exploration noise — pure exploitation from prior.

    Useful for evaluation/inference mode.
    """

    comptime NOISE_TYPE: Int = 2
    comptime NOISE_FRACTION: Float64 = 0.0
    comptime NOISE_ALPHA: Float64 = 0.0


# ═══════════════════════════════════════════════════════════════════════════
# PUCTFormula — UCB exploration formula
# ═══════════════════════════════════════════════════════════════════════════


trait PUCTFormula:
    """Determines the exploration formula used during MCTS selection.

    Controls the exploration-exploitation tradeoff in tree search.
    """

    comptime PUCT_TYPE: Int
    """0=MuZero (log-based), 1=AlphaGo (constant c), 2=UCB1 (classic)."""

    comptime C_BASE: Float64
    """Base constant for MuZero PUCT (default: 19652)."""

    comptime C_INIT: Float64
    """Initial exploration constant (default: 1.25)."""


struct MuZeroPUCT[
    c_base: Float64 = 19652.0,
    c_init: Float64 = 1.25,
](PUCTFormula):
    """MuZero PUCT formula (Schrittwieser et al., 2020).

    c(s) = log((1 + N(s) + c_base) / c_base) + c_init
    score = Q(s,a) + c(s) * P(s,a) * sqrt(N(s)) / (1 + N(s,a))

    The log-based c increases exploration as the tree grows deeper.
    """

    comptime PUCT_TYPE: Int = 0
    comptime C_BASE: Float64 = Self.c_base
    comptime C_INIT: Float64 = Self.c_init


struct AlphaGoPUCT[
    c_puct: Float64 = 2.5,
](PUCTFormula):
    """AlphaGo/AlphaZero PUCT formula (Silver et al., 2017).

    score = Q(s,a) + c * P(s,a) * sqrt(N(s)) / (1 + N(s,a))

    Constant exploration parameter — simpler than MuZero's log-based.
    """

    comptime PUCT_TYPE: Int = 1
    comptime C_BASE: Float64 = 0.0
    comptime C_INIT: Float64 = Self.c_puct


struct UCB1Formula[
    c: Float64 = 1.414,
](PUCTFormula):
    """Classic UCB1 formula (Auer et al., 2002).

    score = Q(s,a) + c * sqrt(ln(N(s)) / N(s,a))

    No prior — pure UCB exploration. Mainly for comparison.
    """

    comptime PUCT_TYPE: Int = 2
    comptime C_BASE: Float64 = 0.0
    comptime C_INIT: Float64 = Self.c


# ═══════════════════════════════════════════════════════════════════════════
# BackupMode — Return computation strategy
# ═══════════════════════════════════════════════════════════════════════════


trait BackupMode:
    """Determines how training targets (value/reward returns) are computed.

    N-step bootstrapped returns are standard for MuZero. Monte Carlo
    returns (full episode) can be used for short-episode environments.
    Lambda returns blend both approaches.
    """

    comptime BACKUP_TYPE: Int
    """0=N-step bootstrap, 1=Monte Carlo (full episode), 2=Lambda return."""

    comptime LAMBDA: Float64
    """Lambda parameter for TD(lambda) returns. Only used when BACKUP_TYPE=2.
    0.0 = 1-step TD, 1.0 = Monte Carlo."""


struct NStepBootstrap(BackupMode):
    """N-step bootstrapped returns (default MuZero).

    z(t) = sum_{i=0}^{n-1} gamma^i * r_{t+i} + gamma^n * v_{t+n}

    N is set by Config.td_steps. Best for online RL with long episodes.
    """

    comptime BACKUP_TYPE: Int = 0
    comptime LAMBDA: Float64 = 0.0


struct MonteCarloReturn(BackupMode):
    """Full-episode Monte Carlo returns (no bootstrapping).

    z(t) = sum_{i=0}^{T-t} gamma^i * r_{t+i}

    Best for short-episode environments (board games, CartPole).
    Unbiased but high variance.
    """

    comptime BACKUP_TYPE: Int = 1
    comptime LAMBDA: Float64 = 1.0


struct LambdaReturn[
    lambda_: Float64 = 0.95,
](BackupMode):
    """TD(lambda) returns — exponentially weighted blend of n-step returns.

    Smoothly interpolates between 1-step TD (lambda=0) and
    Monte Carlo (lambda=1). Good default for most environments.
    """

    comptime BACKUP_TYPE: Int = 2
    comptime LAMBDA: Float64 = Self.lambda_
