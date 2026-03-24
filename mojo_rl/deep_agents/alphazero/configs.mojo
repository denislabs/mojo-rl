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
    Conv2DLayer,
    BatchNorm2D,
    Dropout,
    FlattenLayer,
    ReLU,
    Tanh,
    Softmax,
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
    LR: Float64 = 0.01,
    BS: Int = 16,
    CAP: Int = 50000,
    SIMS: Int = 100,
    NODES: Int = 128,
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
    LR: Float64 = 0.001,
    BS: Int = 64,
    CAP: Int = 100_000,
    SIMS: Int = 100,
    NODES: Int = 128,
    C_PUCT: Float64 = 1.0,
](AlphaZeroConfig):
    """AlphaZero for TicTacToe — CNN variant matching alpha-zero-general.

    Input 27D = 3 channels × 3×3 board (one-hot: mine, opponent, empty).
    3× Conv2D(3×3, same padding) → Conv2D(3×3, valid) → flatten → FC heads.
    LR=0.001 (not 0.01) to prevent dying ReLU in the deeper CNN backbone.
    """

    comptime NAME: String = "AlphaZero-TicTacToe-CNN"
    comptime obs_dim: Int = 27
    comptime action_dim: Int = 9

    # Conv2D → BatchNorm → ReLU (matching alpha-zero-general)
    comptime PredModel = Sequential[
        Conv2DLayer[3, Self.FILTERS, 3, 1, 1, 3, 3],       # 3ch→F, 3×3→3×3
        BatchNorm2D[Self.FILTERS, 3, 3],
        ReLU[Self.FILTERS * 3 * 3],
        Conv2DLayer[Self.FILTERS, Self.FILTERS, 3, 1, 1, 3, 3],
        BatchNorm2D[Self.FILTERS, 3, 3],
        ReLU[Self.FILTERS * 3 * 3],
        Conv2DLayer[Self.FILTERS, Self.FILTERS, 3, 1, 1, 3, 3],
        BatchNorm2D[Self.FILTERS, 3, 3],
        ReLU[Self.FILTERS * 3 * 3],
        Conv2DLayer[Self.FILTERS, Self.FILTERS, 3, 1, 0, 3, 3],  # 3×3→1×1
        BatchNorm2D[Self.FILTERS, 1, 1],
        ReLU[Self.FILTERS],
        FlattenLayer[Self.FILTERS],
        # FC: Linear → BN1D → ReLU → Dropout (matching alpha-zero-general)
        Linear[Self.FILTERS, Self.FILTERS * 2],
        BatchNorm2D[Self.FILTERS * 2, 1, 1],  # BN1D
        ReLU[Self.FILTERS * 2],
        Dropout[Self.FILTERS * 2, 0.3, 42, True],
        Linear[Self.FILTERS * 2, Self.FILTERS],
        BatchNorm2D[Self.FILTERS, 1, 1],  # BN1D
        ReLU[Self.FILTERS],
        Dropout[Self.FILTERS, 0.3, 137, True],
        Parallel[
            Linear[Self.FILTERS, 9],   # Policy (softmax applied in loss kernel)
            Linear[Self.FILTERS, 1],   # Value (tanh applied in loss kernel)
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
# ConnectFour Config
# ═══════════════════════════════════════════════════════════════════════════


struct AlphaZeroConnectFourConfig[
    HIDDEN: Int = 256,
    LR: Float64 = 0.001,
    WD: Float64 = 1e-4,
    BS: Int = 64,
    CAP: Int = 100000,
    SIMS: Int = 25,
    NODES: Int = 64,
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
    comptime OptType = AdamW[LR=Self.LR, WEIGHT_DECAY=Self.WD]

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
    LR: Float64 = 0.001,
    WD: Float64 = 1e-4,
    BS: Int = 64,
    CAP: Int = 200000,
    SIMS: Int = 25,
    NODES: Int = 64,
    C_PUCT: Float64 = 1.0,
](AlphaZeroConfig):
    """AlphaZero for ConnectFour — CNN variant.

    Input 126D = 3 channels × 6 rows × 7 cols.
    3× Conv2D(3×3, same) → Conv2D(3×3, valid) → flatten → FC → heads.

    Tuned to match alpha-zero-general proven settings:
    - 25 MCTS sims (not 100) so prior matters for the feedback loop
    - AdamW with weight decay 1e-4 to prevent weight explosion
    - max_nodes=64 (enough for 25 sims)
    """

    comptime NAME: String = "AlphaZero-ConnectFour-CNN"
    comptime obs_dim: Int = 126
    comptime action_dim: Int = 7

    # Conv2D → BatchNorm → ReLU (matching alpha-zero-general)
    comptime PredModel = Sequential[
        Conv2DLayer[3, Self.FILTERS, 3, 1, 1, 6, 7],       # 3ch→F, 6×7→6×7
        BatchNorm2D[Self.FILTERS, 6, 7],
        ReLU[Self.FILTERS * 6 * 7],
        Conv2DLayer[Self.FILTERS, Self.FILTERS, 3, 1, 1, 6, 7],
        BatchNorm2D[Self.FILTERS, 6, 7],
        ReLU[Self.FILTERS * 6 * 7],
        Conv2DLayer[Self.FILTERS, Self.FILTERS, 3, 1, 1, 6, 7],
        BatchNorm2D[Self.FILTERS, 6, 7],
        ReLU[Self.FILTERS * 6 * 7],
        Conv2DLayer[Self.FILTERS, Self.FILTERS, 3, 1, 0, 6, 7],  # 6×7→4×5
        BatchNorm2D[Self.FILTERS, 4, 5],
        ReLU[Self.FILTERS * 4 * 5],
        FlattenLayer[Self.FILTERS * 4 * 5],
        # FC: Linear → BN1D → ReLU → Dropout (matching alpha-zero-general)
        Linear[Self.FILTERS * 4 * 5, Self.FILTERS * 2],
        BatchNorm2D[Self.FILTERS * 2, 1, 1],
        ReLU[Self.FILTERS * 2],
        Dropout[Self.FILTERS * 2, 0.3, 42, True],
        Linear[Self.FILTERS * 2, Self.FILTERS],
        BatchNorm2D[Self.FILTERS, 1, 1],
        ReLU[Self.FILTERS],
        Dropout[Self.FILTERS, 0.3, 137, True],
        Parallel[
            Linear[Self.FILTERS, 7],
            Linear[Self.FILTERS, 1],
        ],
    ]
    comptime OptType = AdamW[LR=Self.LR, WEIGHT_DECAY=Self.WD]

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
# ConnectFour Config (ResNet — closer to original AlphaZero)
# ═══════════════════════════════════════════════════════════════════════════

# ResNet blocks via composition: Conv2DReLU → Conv2D → Residual add → ReLU
comptime ResBlock6x7[F: Int] = Sequential[
    Residual[Sequential[
        Conv2DReLU[F, F, 3, 1, 1, 6, 7],
        Conv2DLayer[F, F, 3, 1, 1, 6, 7],
    ]],
    ReLU[F * 6 * 7],
]

comptime ResBlock3x3[F: Int] = Sequential[
    Residual[Sequential[
        Conv2DReLU[F, F, 3, 1, 1, 3, 3],
        Conv2DLayer[F, F, 3, 1, 1, 3, 3],
    ]],
    ReLU[F * 3 * 3],
]


struct AlphaZeroConnectFourResNetConfig[
    FILTERS: Int = 128,
    LR: Float64 = 0.001,
    WD: Float64 = 1e-4,
    BS: Int = 64,
    CAP: Int = 200000,
    SIMS: Int = 25,
    NODES: Int = 64,
    C_PUCT: Float64 = 1.0,
](AlphaZeroConfig):
    """AlphaZero for ConnectFour — ResNet with 4 residual blocks.

    Closer to original AlphaZero architecture:
    - Initial Conv → 4× ResBlock(Conv+ReLU → Conv → skip+ReLU) → FC heads
    - 100 MCTS simulations (vs 25 in CNN config)
    - max_nodes=256 for deeper search trees
    """

    comptime NAME: String = "AlphaZero-ConnectFour-ResNet"
    comptime obs_dim: Int = 126
    comptime action_dim: Int = 7

    comptime PredModel = Sequential[
        Conv2DReLU[3, Self.FILTERS, 3, 1, 1, 6, 7],       # Initial: 3ch→F
        ResBlock6x7[Self.FILTERS],                              # ResBlock 1
        ResBlock6x7[Self.FILTERS],                              # ResBlock 2
        ResBlock6x7[Self.FILTERS],                              # ResBlock 3
        ResBlock6x7[Self.FILTERS],                              # ResBlock 4
        Conv2DReLU[Self.FILTERS, Self.FILTERS, 3, 1, 0, 6, 7],  # Reduce: 6×7→4×5
        FlattenLayer[Self.FILTERS * 4 * 5],
        LinearReLU[Self.FILTERS * 4 * 5, Self.FILTERS * 2],
        LinearReLU[Self.FILTERS * 2, Self.FILTERS],
        Parallel[
            Linear[Self.FILTERS, 7],
            Linear[Self.FILTERS, 1],
        ],
    ]
    comptime OptType = AdamW[LR=Self.LR, WEIGHT_DECAY=Self.WD]

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
# TicTacToe ResNet Config
# ═══════════════════════════════════════════════════════════════════════════


struct AlphaZeroTicTacToeResNetConfig[
    FILTERS: Int = 128,
    LR: Float64 = 1e-3,
    BS: Int = 64,
    CAP: Int = 50000,
    SIMS: Int = 50,
    NODES: Int = 64,
    C_PUCT: Float64 = 1.0,
](AlphaZeroConfig):
    """AlphaZero for TicTacToe — ResNet with 4 residual blocks + 50 MCTS sims."""

    comptime NAME: String = "AlphaZero-TicTacToe-ResNet"
    comptime obs_dim: Int = 27
    comptime action_dim: Int = 9

    comptime PredModel = Sequential[
        Conv2DReLU[3, Self.FILTERS, 3, 1, 1, 3, 3],       # Initial: 3ch→F
        ResBlock3x3[Self.FILTERS],                              # ResBlock 1
        ResBlock3x3[Self.FILTERS],                              # ResBlock 2
        ResBlock3x3[Self.FILTERS],                              # ResBlock 3
        ResBlock3x3[Self.FILTERS],                              # ResBlock 4
        Conv2DReLU[Self.FILTERS, Self.FILTERS, 3, 1, 0, 3, 3],  # Reduce: 3×3→1×1
        FlattenLayer[Self.FILTERS],
        LinearReLU[Self.FILTERS, Self.FILTERS * 2],
        LinearReLU[Self.FILTERS * 2, Self.FILTERS],
        Parallel[
            Linear[Self.FILTERS, 9],
            Linear[Self.FILTERS, 1],
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
