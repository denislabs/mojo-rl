"""AlphaZero Configs — Compile-time configuration for AlphaZero agents.

AlphaZero is simpler than MuZero:
  - 1 network: PredNet(obs) → (policy_logits, value)
  - No representation, no dynamics, no latent space
  - MCTS uses true game rules (env.step) for expansion
  - Training: supervised CE(policy, mcts_π) + MSE(value, outcome)

Reuses strategies from muzero/strategies.mojo for composability.
"""

from mojo_rl.nn.model import (
    Model,
    Linear,
    LinearReLU,
    LinearMish,
    Sequential,
    Parallel,
    Conv2DReLU,
    FlattenLayer,
)
from mojo_rl.nn.optimizer import Optimizer, Adam, AdamW
from mojo_rl.nn.autodiff.combinators import Residual
from mojo_rl.deep_agents.muzero.strategies import (
    ExplorationNoise,
    DirichletNoise,
    PUCTFormula,
    AlphaGoPUCT,
    MuZeroPUCT,
    PlayerMode,
    SelfPlay,
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
    comptime obs_dim: Int  # Observation dimension (canonical)
    comptime action_dim: Int  # Number of discrete actions

    # ── Network ───────────────────────────────────────────────────
    comptime PredModel: Model  # f(obs) → (policy_logits[action_dim], value[1])
    comptime OptType: Optimizer

    # ── Training ──────────────────────────────────────────────────
    comptime batch_size: Int
    comptime buffer_capacity: Int
    comptime history_window: Int  # Keep last K iterations of self-play data

    # ── MCTS ──────────────────────────────────────────────────────
    comptime num_simulations: Int
    comptime max_nodes: Int
    comptime temp_threshold: Int  # Use temp=1 for first N moves, temp=0 after

    # ── Strategies (shared with MuZero) ───────────────────────────
    comptime Noise: ExplorationNoise
    comptime PUCT: PUCTFormula
    comptime Players: PlayerMode


# ═══════════════════════════════════════════════════════════════════════════
# TicTacToe Config (MLP — lightweight)
# ═══════════════════════════════════════════════════════════════════════════


struct AlphaZeroTicTacToeConfig[
    HIDDEN: Int = 128,
    LR: Float64 = 1e-3,
    BS: Int = 64,
    CAP: Int = 50000,
    SIMS: Int = 25,
    NODES: Int = 64,
    C_PUCT: Float64 = 1.0,
](AlphaZeroConfig):
    """AlphaZero for TicTacToe (27D obs, 9 actions) — MLP variant."""

    comptime NAME: String = "AlphaZero-TicTacToe"
    comptime obs_dim: Int = 27
    comptime action_dim: Int = 9

    comptime PredModel = Sequential[
        LinearReLU[27, Self.HIDDEN],
        LinearReLU[Self.HIDDEN, Self.HIDDEN],
        Parallel[
            Linear[Self.HIDDEN, 9],  # Policy head
            Linear[Self.HIDDEN, 1],  # Scalar value head
        ],
    ]
    comptime OptType = Adam[LR=Self.LR]

    comptime batch_size: Int = Self.BS
    comptime buffer_capacity: Int = Self.CAP
    comptime history_window: Int = 20  # Like alpha-zero-general
    comptime num_simulations: Int = Self.SIMS
    comptime max_nodes: Int = Self.NODES
    comptime temp_threshold: Int = 15  # temp=1 first 15 moves, then temp=0

    comptime Noise = DirichletNoise[0.25, 0.25]
    comptime PUCT = AlphaGoPUCT[Self.C_PUCT]
    comptime Players = SelfPlay


# ═══════════════════════════════════════════════════════════════════════════
# TicTacToe Config (CNN — matches alpha-zero-general architecture)
# ═══════════════════════════════════════════════════════════════════════════


struct AlphaZeroTicTacToeCNNConfig[
    FILTERS: Int = 128,
    LR: Float64 = 1e-3,
    BS: Int = 64,
    CAP: Int = 100_000,
    SIMS: Int = 25,
    NODES: Int = 64,
    C_PUCT: Float64 = 1.0,
](AlphaZeroConfig):
    """AlphaZero for TicTacToe — CNN variant matching alpha-zero-general.

    Input 27D = 3 channels × 3×3 board (one-hot: mine, opponent, empty).
    3× Conv2D(3×3, same padding) → Conv2D(3×3, valid) → flatten → FC heads.
    """

    comptime NAME: String = "AlphaZero-TicTacToe-CNN"
    comptime obs_dim: Int = 27
    comptime action_dim: Int = 9

    # Conv2DReLU[ic, oc, k, s, p, h, w]
    # 3 channels, 3×3 board with same padding (p=1)
    # After 3× same-padding convs: still 3×3
    # After valid conv (p=0): (3+0-3)/1+1 = 1×1
    comptime PredModel = Sequential[
        Conv2DReLU[3, Self.FILTERS, 3, 1, 1, 3, 3],  # 3ch→F, 3×3→3×3
        Conv2DReLU[Self.FILTERS, Self.FILTERS, 3, 1, 1, 3, 3],  # F→F, 3×3→3×3
        Conv2DReLU[Self.FILTERS, Self.FILTERS, 3, 1, 1, 3, 3],  # F→F, 3×3→3×3
        Conv2DReLU[Self.FILTERS, Self.FILTERS, 3, 1, 0, 3, 3],  # F→F, 3×3→1×1
        FlattenLayer[Self.FILTERS],  # F×1×1 → F
        LinearReLU[Self.FILTERS, Self.FILTERS * 2],  # F → 2F
        LinearReLU[Self.FILTERS * 2, Self.FILTERS],  # 2F → F
        Parallel[
            Linear[Self.FILTERS, 9],  # Policy head
            Linear[Self.FILTERS, 1],  # Value head
        ],
    ]
    comptime OptType = Adam[LR=Self.LR]

    comptime batch_size: Int = Self.BS
    comptime buffer_capacity: Int = Self.CAP
    comptime history_window: Int = 20  # Like alpha-zero-general
    comptime num_simulations: Int = Self.SIMS
    comptime max_nodes: Int = Self.NODES
    comptime temp_threshold: Int = 15

    comptime Noise = DirichletNoise[0.25, 0.25]
    comptime PUCT = AlphaGoPUCT[Self.C_PUCT]
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
    comptime history_window: Int = 20
    comptime num_simulations: Int = Self.SIMS
    comptime max_nodes: Int = Self.NODES
    comptime temp_threshold: Int = 15

    comptime Noise = DirichletNoise[0.25, 0.25]
    comptime PUCT = AlphaGoPUCT[2.5]
    comptime Players = SelfPlay


# ═══════════════════════════════════════════════════════════════════════════
# ConnectFour Config (CNN — matches alpha-zero-general architecture)
# ═══════════════════════════════════════════════════════════════════════════


struct AlphaZeroConnectFourCNNConfig[
    FILTERS: Int = 128,
    LR: Float64 = 1e-3,
    BS: Int = 64,
    CAP: Int = 200000,
    SIMS: Int = 25,
    NODES: Int = 128,
    C_PUCT: Float64 = 1.0,
](AlphaZeroConfig):
    """AlphaZero for ConnectFour — CNN variant.

    Input 126D = 3 channels × 6 rows × 7 cols.
    3× Conv2D(3×3, same) → Conv2D(3×3, valid) → flatten → FC → heads.

    Alpha-zero-general uses 20 ResNet blocks with 128 filters.
    We use a simpler 4-conv architecture for faster iteration.
    """

    comptime NAME: String = "AlphaZero-ConnectFour-CNN"
    comptime obs_dim: Int = 126
    comptime action_dim: Int = 7

    # Conv2DReLU[ic, oc, k, s, p, h, w]
    # Input: 3 channels, 6 rows, 7 cols (column-major in obs, but Conv2D is row-major)
    # Note: obs layout is 3 planes of 42 = 7cols × 6rows (col-major)
    # Conv2D expects (channels, height, width) = (3, 6, 7)
    comptime PredModel = Sequential[
        Conv2DReLU[3, Self.FILTERS, 3, 1, 1, 6, 7],       # 3ch→F, 6×7→6×7
        Conv2DReLU[Self.FILTERS, Self.FILTERS, 3, 1, 1, 6, 7],  # F→F, 6×7→6×7
        Conv2DReLU[Self.FILTERS, Self.FILTERS, 3, 1, 1, 6, 7],  # F→F, 6×7→6×7
        Conv2DReLU[Self.FILTERS, Self.FILTERS, 3, 1, 0, 6, 7],  # F→F, 6×7→4×5
        FlattenLayer[Self.FILTERS * 4 * 5],                       # F×4×5 → 20F
        LinearReLU[Self.FILTERS * 4 * 5, Self.FILTERS * 2],
        LinearReLU[Self.FILTERS * 2, Self.FILTERS],
        Parallel[
            Linear[Self.FILTERS, 7],   # Policy head
            Linear[Self.FILTERS, 1],   # Value head
        ],
    ]
    comptime OptType = Adam[LR=Self.LR]

    comptime batch_size: Int = Self.BS
    comptime buffer_capacity: Int = Self.CAP
    comptime history_window: Int = 20
    comptime num_simulations: Int = Self.SIMS
    comptime max_nodes: Int = Self.NODES
    comptime temp_threshold: Int = 15

    comptime Noise = DirichletNoise[0.25, 0.25]
    comptime PUCT = AlphaGoPUCT[Self.C_PUCT]
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
    comptime history_window: Int = 20
    comptime num_simulations: Int = Self.SIMS
    comptime max_nodes: Int = Self.NODES
    comptime temp_threshold: Int = 30

    comptime Noise = DirichletNoise[
        0.25, 0.03
    ]  # Small alpha for large action space
    comptime PUCT = AlphaGoPUCT[2.5]
    comptime Players = SelfPlay
