"""AlphaZero Configs — Compile-time configuration for AlphaZero agents.

AlphaZero is simpler than MuZero:
  - 1 network: PredNet(obs) → (policy_logits, value)
  - No representation, no dynamics, no latent space
  - MCTS uses true game rules (env.step) for expansion
  - Training: supervised CE(policy, mcts_π) + MSE(value, outcome)

Reuses strategies from muzero/strategies.mojo for composability.
"""

from mojo_rl.nn.model import (
    Model, Linear, LinearReLU, LinearMish, Sequential, Parallel,
)
from mojo_rl.nn.optimizer import Optimizer, Adam
from mojo_rl.nn.autodiff.combinators import Residual
from mojo_rl.deep_agents.muzero.strategies import (
    ExplorationNoise, DirichletNoise,
    PUCTFormula, AlphaGoPUCT, MuZeroPUCT,
    PlayerMode, SelfPlay,
)


# ═══════════════════════════════════════════════════════════════════════════
# AlphaZero Config Trait
# ═══════════════════════════════════════════════════════════════════════════


trait AlphaZeroConfig:
    """Compile-time configuration for AlphaZero agents.

    Simpler than MuZeroConfig — only one network, no dynamics.
    """

    comptime NAME: String

    # ── Dimensions ────────────────────────────────────────────────
    comptime obs_dim: Int         # Observation dimension (canonical)
    comptime action_dim: Int      # Number of discrete actions

    # ── Network ───────────────────────────────────────────────────
    comptime PredModel: Model     # f(obs) → (policy_logits[action_dim], value[1])
    comptime OptType: Optimizer

    # ── Training ──────────────────────────────────────────────────
    comptime batch_size: Int
    comptime buffer_capacity: Int

    # ── MCTS ──────────────────────────────────────────────────────
    comptime num_simulations: Int
    comptime max_nodes: Int

    # ── Strategies (shared with MuZero) ───────────────────────────
    comptime Noise: ExplorationNoise
    comptime PUCT: PUCTFormula
    comptime Players: PlayerMode


# ═══════════════════════════════════════════════════════════════════════════
# TicTacToe Config
# ═══════════════════════════════════════════════════════════════════════════


struct AlphaZeroTicTacToeConfig[
    HIDDEN: Int = 128,
    LR: Float64 = 1e-3,
    BS: Int = 128,
    CAP: Int = 50000,
    SIMS: Int = 50,
    NODES: Int = 64,
](AlphaZeroConfig):
    """AlphaZero for TicTacToe (27D obs, 9 actions)."""

    comptime NAME: String = "AlphaZero-TicTacToe"
    comptime obs_dim: Int = 27
    comptime action_dim: Int = 9

    comptime PredModel = Sequential[
        LinearReLU[27, Self.HIDDEN],
        LinearReLU[Self.HIDDEN, Self.HIDDEN],
        Parallel[
            Linear[Self.HIDDEN, 9],   # Policy head
            Linear[Self.HIDDEN, 1],   # Scalar value head
        ],
    ]
    comptime OptType = Adam[LR=Self.LR]

    comptime batch_size: Int = Self.BS
    comptime buffer_capacity: Int = Self.CAP
    comptime num_simulations: Int = Self.SIMS
    comptime max_nodes: Int = Self.NODES

    comptime Noise = DirichletNoise[0.25, 0.25]
    comptime PUCT = AlphaGoPUCT[2.5]
    comptime Players = SelfPlay


# ═══════════════════════════════════════════════════════════════════════════
# ConnectFour Config
# ═══════════════════════════════════════════════════════════════════════════


struct AlphaZeroConnectFourConfig[
    HIDDEN: Int = 256,
    LR: Float64 = 5e-4,
    BS: Int = 128,
    CAP: Int = 100000,
    SIMS: Int = 100,
    NODES: Int = 128,
](AlphaZeroConfig):
    """AlphaZero for ConnectFour (126D obs, 7 actions)."""

    comptime NAME: String = "AlphaZero-ConnectFour"
    comptime obs_dim: Int = 126
    comptime action_dim: Int = 7

    comptime PredModel = Sequential[
        LinearReLU[126, Self.HIDDEN],
        LinearReLU[Self.HIDDEN, Self.HIDDEN],
        LinearReLU[Self.HIDDEN, Self.HIDDEN],
        Parallel[
            Linear[Self.HIDDEN, 7],
            Linear[Self.HIDDEN, 1],
        ],
    ]
    comptime OptType = Adam[LR=Self.LR]

    comptime batch_size: Int = Self.BS
    comptime buffer_capacity: Int = Self.CAP
    comptime num_simulations: Int = Self.SIMS
    comptime max_nodes: Int = Self.NODES

    comptime Noise = DirichletNoise[0.25, 0.25]
    comptime PUCT = AlphaGoPUCT[2.5]
    comptime Players = SelfPlay


# ═══════════════════════════════════════════════════════════════════════════
# Chess Config
# ═══════════════════════════════════════════════════════════════════════════


struct AlphaZeroChessConfig[
    HIDDEN: Int = 512,
    LR: Float64 = 1e-4,
    BS: Int = 256,
    CAP: Int = 500000,
    SIMS: Int = 800,
    NODES: Int = 512,
](AlphaZeroConfig):
    """AlphaZero for Chess (896D obs, 4672 actions)."""

    comptime NAME: String = "AlphaZero-Chess"
    comptime obs_dim: Int = 896
    comptime action_dim: Int = 4672

    comptime PredModel = Sequential[
        LinearReLU[896, Self.HIDDEN],
        LinearReLU[Self.HIDDEN, Self.HIDDEN],
        LinearReLU[Self.HIDDEN, Self.HIDDEN],
        Parallel[
            Linear[Self.HIDDEN, 4672],
            Linear[Self.HIDDEN, 1],
        ],
    ]
    comptime OptType = Adam[LR=Self.LR]

    comptime batch_size: Int = Self.BS
    comptime buffer_capacity: Int = Self.CAP
    comptime num_simulations: Int = Self.SIMS
    comptime max_nodes: Int = Self.NODES

    comptime Noise = DirichletNoise[0.25, 0.03]  # Small alpha for large action space
    comptime PUCT = AlphaGoPUCT[2.5]
    comptime Players = SelfPlay
