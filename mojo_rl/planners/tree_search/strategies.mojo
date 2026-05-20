"""MCTS Strategy Traits — Composable building blocks with embedded logic.

Unlike simple flag-only configs, strategies contain actual compute methods
that are called at compile-time dispatch points. Follows the DQN/SAC
strategy pattern (e.g., QTarget.compute_targets_gpu).

Promoted from `mojo_rl/deep_agents/muzero/strategies.mojo` in Phase 0 of the
planners package refactor (see `docs/PLANNERS_PACKAGE.md`). ValueEncoding was
split out to `planners/common/value_encoding.mojo` since trajectory optimizers
also use it.

Strategy traits:
  - SearchMode: Learned dynamics (MuZero) vs true game rules (AlphaZero)
  - HiddenScaling: Hidden state normalization (MinMax, None)
  - ExplorationNoise: Root exploration (Dirichlet, Epsilon, None)
  - PUCTFormula: UCB exploration formula with compute_score() logic
  - BackupMode: Return computation with compute_return() logic
  - PlayerMode: Single-player vs self-play with transform_value() logic
"""

from std.math import sqrt, log


# ═══════════════════════════════════════════════════════════════════════════
# SearchMode — Learned dynamics vs true game rules
# ═══════════════════════════════════════════════════════════════════════════


trait SearchMode:
    """Determines how MCTS expands leaf nodes.

    LearnedDynamics: Use dynamics network g(hidden, action) → next_hidden
    TrueGameRules: Use env.step(state, action) → next_state (requires game states in tree)
    """

    comptime USE_LEARNED_DYNAMICS: Bool
    """True = use dynamics network. False = use game rules."""

    comptime NEEDS_GAME_STATE: Bool
    """True = MCTS tree stores game states per node (for env.step).
    False = MCTS tree stores hidden states per node (for dynamics net)."""


struct LearnedDynamics(SearchMode):
    """MuZero: learn dynamics from data, search in latent space."""

    comptime USE_LEARNED_DYNAMICS: Bool = True
    comptime NEEDS_GAME_STATE: Bool = False


struct TrueGameRules(SearchMode):
    """AlphaZero: use true game rules, search in observation space.
    Tree nodes store actual game states. Expansion calls env.step()."""

    comptime USE_LEARNED_DYNAMICS: Bool = False
    comptime NEEDS_GAME_STATE: Bool = True


# ═══════════════════════════════════════════════════════════════════════════
# HiddenScaling
# ═══════════════════════════════════════════════════════════════════════════


trait HiddenScaling:
    """Hidden state normalization after dynamics."""

    comptime ENABLED: Bool
    comptime SCALE_METHOD: Int  # 0=MinMax, 1=LayerNorm, 2=SimNorm


struct MinMaxScale(HiddenScaling):
    """Min-max normalization to [0, 1]."""

    comptime ENABLED: Bool = True
    comptime SCALE_METHOD: Int = 0


struct NoScale(HiddenScaling):
    """No scaling — for AlphaZero where tree stores real game states."""

    comptime ENABLED: Bool = False
    comptime SCALE_METHOD: Int = 0


# ═══════════════════════════════════════════════════════════════════════════
# ExplorationNoise — with embedded sampling logic
# ═══════════════════════════════════════════════════════════════════════════


trait ExplorationNoise:
    """Root exploration noise strategy.

    Provides compile-time parameters for noise generation.
    Noise sampling runs inside GPU kernels using these constants.
    """

    comptime NOISE_TYPE: Int  # 0=Dirichlet, 1=Uniform, 2=None
    comptime NOISE_FRACTION: Float64
    comptime NOISE_ALPHA: Float64


struct DirichletNoise[
    fraction: Float64 = 0.25,
    alpha: Float64 = 0.25,
](ExplorationNoise):
    """Dirichlet noise (default MuZero/AlphaZero).
    Alpha: 0.03 (Go/Chess), 0.25 (Atari/small games)."""

    comptime NOISE_TYPE: Int = 0
    comptime NOISE_FRACTION: Float64 = Self.fraction
    comptime NOISE_ALPHA: Float64 = Self.alpha


struct EpsilonNoise[
    fraction: Float64 = 0.25,
](ExplorationNoise):
    """Uniform epsilon noise — simpler alternative."""

    comptime NOISE_TYPE: Int = 1
    comptime NOISE_FRACTION: Float64 = Self.fraction
    comptime NOISE_ALPHA: Float64 = 0.0


struct NoNoise(ExplorationNoise):
    """No noise — pure exploitation. For evaluation."""

    comptime NOISE_TYPE: Int = 2
    comptime NOISE_FRACTION: Float64 = 0.0
    comptime NOISE_ALPHA: Float64 = 0.0


# ═══════════════════════════════════════════════════════════════════════════
# PUCTFormula — with embedded compute_score() logic
# ═══════════════════════════════════════════════════════════════════════════


trait PUCTFormula:
    """UCB exploration formula with embedded computation logic.

    Unlike a simple flag, this trait provides the actual formula
    that computes the exploration constant c(s) from parent visit count.
    """

    comptime C_BASE: Float64
    comptime C_INIT: Float64

    @staticmethod
    def compute_c(parent_visits: Float64, cb: Float64, ci: Float64) -> Float64:
        """Compute exploration constant c(s) from parent visit count.

        Called inside the MCTS selection kernel (must be @staticmethod + inline).

        Args:
            parent_visits: N(s) — total visits at parent node.
            cb: Base constant (from config).
            ci: Initial constant (from config).

        Returns:
            Exploration constant c(s).
        """
        ...


struct MuZeroPUCT[
    c_base: Float64 = 19652.0,
    c_init: Float64 = 1.25,
](PUCTFormula):
    """MuZero: c(s) = log((1 + N(s) + c_base) / c_base) + c_init.
    Log-based c increases exploration as tree grows."""

    comptime C_BASE: Float64 = Self.c_base
    comptime C_INIT: Float64 = Self.c_init

    @staticmethod
    def compute_c(parent_visits: Float64, cb: Float64, ci: Float64) -> Float64:
        return log((1.0 + parent_visits + cb) / cb) + ci


struct AlphaGoPUCT[
    c_puct: Float64 = 2.5,
](PUCTFormula):
    """AlphaGo/AlphaZero: c(s) = c_puct (constant).
    Simpler than MuZero's log-based formula."""

    comptime C_BASE: Float64 = 0.0
    comptime C_INIT: Float64 = Self.c_puct

    @staticmethod
    def compute_c(parent_visits: Float64, cb: Float64, ci: Float64) -> Float64:
        return ci


struct UCB1Formula[
    c: Float64 = 1.414,
](PUCTFormula):
    """Classic UCB1: score = Q + c * sqrt(ln(N) / n). No prior."""

    comptime C_BASE: Float64 = 0.0
    comptime C_INIT: Float64 = Self.c

    @staticmethod
    def compute_c(parent_visits: Float64, cb: Float64, ci: Float64) -> Float64:
        return ci


# ═══════════════════════════════════════════════════════════════════════════
# BackupMode — with embedded return computation logic
# ═══════════════════════════════════════════════════════════════════════════


trait BackupMode:
    """Return computation strategy with embedded logic.

    Determines how training targets (value returns) are computed from
    rewards and bootstrap values.
    """

    comptime BACKUP_TYPE: Int  # 0=N-step, 1=MonteCarlo, 2=Lambda
    comptime LAMBDA: Float64

    @staticmethod
    def should_bootstrap(steps_used: Int, n: Int, hit_terminal: Bool) -> Bool:
        """Whether to add bootstrap value to the return.

        Args:
            steps_used: How many reward steps were accumulated.
            n: Maximum n-step horizon.
            hit_terminal: Whether a terminal state was hit.

        Returns:
            True if bootstrap value should be added.
        """
        ...


struct NStepBootstrap(BackupMode):
    """N-step bootstrap: z = sum gamma^i r_i + gamma^n V(s_{t+n})."""

    comptime BACKUP_TYPE: Int = 0
    comptime LAMBDA: Float64 = 0.0

    @staticmethod
    def should_bootstrap(steps_used: Int, n: Int, hit_terminal: Bool) -> Bool:
        return not hit_terminal and steps_used == n


struct MonteCarloReturn(BackupMode):
    """Full-episode return: z = sum gamma^i r_i. No bootstrapping.
    Best for short-episode games (board games)."""

    comptime BACKUP_TYPE: Int = 1
    comptime LAMBDA: Float64 = 1.0

    @staticmethod
    def should_bootstrap(steps_used: Int, n: Int, hit_terminal: Bool) -> Bool:
        return False  # Never bootstrap — use full episode return


struct LambdaReturn[
    lambda_: Float64 = 0.95,
](BackupMode):
    """TD(lambda) returns. Lambda=0 → 1-step TD, Lambda=1 → Monte Carlo."""

    comptime BACKUP_TYPE: Int = 2
    comptime LAMBDA: Float64 = Self.lambda_

    @staticmethod
    def should_bootstrap(steps_used: Int, n: Int, hit_terminal: Bool) -> Bool:
        return not hit_terminal and steps_used == n


# ═══════════════════════════════════════════════════════════════════════════
# PlayerMode — with embedded value transform logic
# ═══════════════════════════════════════════════════════════════════════════


trait PlayerMode:
    """Single-player vs two-player self-play with embedded transform logic.

    Provides the value transformation applied during MCTS backup.
    For zero-sum games, values are negated at each tree level
    (parent sees opposite of child's value).
    """

    comptime IS_SELF_PLAY: Bool
    comptime NEGATE_BACKUP: Bool
    comptime USE_LEGAL_MASK: Bool

    @staticmethod
    def backup_transform(
        value: Float64, reward: Float64, gamma: Float64
    ) -> Float64:
        """Transform value during MCTS backup.

        Called at each level when propagating values from leaf to root.

        Args:
            value: Accumulated value from child nodes.
            reward: Reward at this edge (from dynamics or game rules).
            gamma: Discount factor.

        Returns:
            Transformed value for the parent node.
        """
        ...


struct SinglePlayer(PlayerMode):
    """Standard single-player: value = reward + gamma * child_value."""

    comptime IS_SELF_PLAY: Bool = False
    comptime NEGATE_BACKUP: Bool = False
    comptime USE_LEGAL_MASK: Bool = False

    @staticmethod
    def backup_transform(
        value: Float64, reward: Float64, gamma: Float64
    ) -> Float64:
        return reward + gamma * value


struct SelfPlay(PlayerMode):
    """Two-player zero-sum: value = -child_value (no discount, no per-step reward).
    Parent's perspective is opposite to child's perspective."""

    comptime IS_SELF_PLAY: Bool = True
    comptime NEGATE_BACKUP: Bool = True
    comptime USE_LEGAL_MASK: Bool = True

    @staticmethod
    def backup_transform(
        value: Float64, reward: Float64, gamma: Float64
    ) -> Float64:
        return -value
