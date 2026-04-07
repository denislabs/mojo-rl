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
    LinearBatchNormReLU,
    Sequential,
    Parallel,
    Conv2DReLU,
    Conv2DLayer,
    BatchNorm2D,
    Conv2DBatchNormReLU,
    Dropout,
    FlattenLayer,
    ReLU,
    Tanh,
    Softmax,
)
from mojo_rl.nn.optimizer import Optimizer, Adam, AdamW
from mojo_rl.nn.autodiff.combinators import Residual, Repeat
from mojo_rl.nn.model.resblock_conv2d_bn import ResBlockConv2DBN
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
    comptime temp_threshold: Int  # Use temp=1 for first N moves, then anneal
    comptime temp_min: Float64  # Min temperature after threshold (0.0=greedy, 0.3=AlphaZero.jl)
    comptime batch_sims: Int  # Parallel MCTS sims per round (8, 16, or 32)
    comptime invalid_action_penalty: Float64  # Penalty for prob mass on illegal moves (1.0=AlphaZero.jl)

    # ── Value target ─────────────────────────────────────────────
    # Blend between game outcome (z) and MCTS root Q-value (q):
    #   value_target = (1 - w) * z + w * q
    # 0.0 = pure z (original AlphaZero), 1.0 = pure q, 0.5 = average
    comptime value_target_q_weight: Float64

    # ── GPU episode tracking ────────────────────────────────────────
    comptime max_episode_length: Int  # Max steps per episode (for GPU staging)
    comptime board_rows: Int  # Board height (for augmentation kernel)
    comptime board_cols: Int  # Board width (for augmentation kernel)
    comptime board_planes: Int  # Obs planes (for augmentation kernel)
    comptime num_symmetries: Int  # Data augmentation symmetries (1=none, 2=flip)

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
    comptime temp_min: Float64 = 0.0
    comptime batch_sims: Int = 8
    comptime invalid_action_penalty: Float64 = 0.0
    comptime value_target_q_weight: Float64 = 0.0
    comptime max_episode_length: Int = 9  # 3×3 board
    comptime board_rows: Int = 3
    comptime board_cols: Int = 3
    comptime board_planes: Int = 3
    comptime num_symmetries: Int = 2  # identity + horizontal flip

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

    # Fused Conv2D+BN+ReLU (matching alpha-zero-general, fewer kernel launches)
    comptime PredModel = Sequential[
        Conv2DBatchNormReLU[3, Self.FILTERS, 3, 1, 1, 3, 3],  # 3ch→F, 3×3→3×3
        Conv2DBatchNormReLU[Self.FILTERS, Self.FILTERS, 3, 1, 1, 3, 3],
        Conv2DBatchNormReLU[Self.FILTERS, Self.FILTERS, 3, 1, 1, 3, 3],
        Conv2DBatchNormReLU[
            Self.FILTERS, Self.FILTERS, 3, 1, 0, 3, 3
        ],  # 3×3→1×1
        FlattenLayer[Self.FILTERS],
        # FC: Fused Linear+BN+ReLU → Dropout (matching alpha-zero-general)
        LinearBatchNormReLU[Self.FILTERS, Self.FILTERS * 2],
        Dropout[Self.FILTERS * 2, 0.3, 42, True],
        LinearBatchNormReLU[Self.FILTERS * 2, Self.FILTERS],
        Dropout[Self.FILTERS, 0.3, 137, True],
        Parallel[
            Linear[Self.FILTERS, 9],  # Policy (softmax applied in loss kernel)
            Linear[Self.FILTERS, 1],  # Value (tanh applied in loss kernel)
        ],
    ]
    comptime OptType = Adam[LR=Self.LR]

    comptime batch_size: Int = Self.BS
    comptime buffer_capacity: Int = Self.CAP
    comptime history_window: Int = 20
    comptime num_simulations: Int = Self.SIMS
    comptime max_nodes: Int = Self.NODES
    comptime temp_threshold: Int = 15
    comptime temp_min: Float64 = 0.0
    comptime batch_sims: Int = 8
    comptime invalid_action_penalty: Float64 = 0.0
    comptime value_target_q_weight: Float64 = 0.0
    comptime max_episode_length: Int = 9
    comptime board_rows: Int = 3
    comptime board_cols: Int = 3
    comptime board_planes: Int = 3
    comptime num_symmetries: Int = 2

    comptime Noise = DirichletNoise[0.25, 0.25]
    comptime PUCT = AlphaGoPUCT[Self.C_PUCT]
    comptime Players = SelfPlay


# ═══════════════════════════════════════════════════════════════════════════
# ConnectFour Config
# ═══════════════════════════════════════════════════════════════════════════


struct AlphaZeroConnectFourConfig[
    HIDDEN: Int = 256,
    LR: Float64 = 2e-3,
    WD: Float64 = 1e-4,
    BS: Int = 64,
    CAP: Int = 400000,
    SIMS: Int = 600,
    NODES: Int = 1024,
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
    comptime temp_threshold: Int = 20
    comptime temp_min: Float64 = 0.0
    comptime batch_sims: Int = 8
    comptime invalid_action_penalty: Float64 = 0.0
    comptime value_target_q_weight: Float64 = 0.0
    comptime max_episode_length: Int = 42  # 6×7 board
    comptime board_rows: Int = 6
    comptime board_cols: Int = 7
    comptime board_planes: Int = 3
    comptime num_symmetries: Int = 2

    comptime Noise = DirichletNoise[0.25, 1.0]  # alpha=1.0 for C4
    comptime PUCT = AlphaGoPUCT[2.0]
    comptime Players = SelfPlay


# ═══════════════════════════════════════════════════════════════════════════
# ConnectFour Config (CNN — matches alpha-zero-general architecture)
# ═══════════════════════════════════════════════════════════════════════════


struct AlphaZeroConnectFourCNNConfig[
    FILTERS: Int = 128,
    LR: Float64 = 2e-3,
    WD: Float64 = 1e-4,
    BS: Int = 64,
    CAP: Int = 400000,
    SIMS: Int = 600,
    NODES: Int = 1024,
    C_PUCT: Float64 = 2.0,
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

    # Non-fused Conv2D → BN → ReLU (for debugging — isolate fusion issues)
    comptime PredModel = Sequential[
        Conv2DBatchNormReLU[3, Self.FILTERS, 3, 1, 1, 6, 7],  # 3ch→F, 6×7→6×7
        Conv2DBatchNormReLU[Self.FILTERS, Self.FILTERS, 3, 1, 1, 6, 7],
        Conv2DBatchNormReLU[Self.FILTERS, Self.FILTERS, 3, 1, 1, 6, 7],
        Conv2DBatchNormReLU[
            Self.FILTERS, Self.FILTERS, 3, 1, 0, 6, 7
        ],  # 6×7→4×5
        FlattenLayer[Self.FILTERS * 4 * 5],
        # FC: Fused Linear+BN+ReLU → Dropout (matching alpha-zero-general)
        LinearBatchNormReLU[Self.FILTERS * 4 * 5, Self.FILTERS * 2],
        Dropout[Self.FILTERS * 2, 0.3, 42, True],
        LinearBatchNormReLU[Self.FILTERS * 2, Self.FILTERS],
        Dropout[Self.FILTERS, 0.3, 137, True],
        Parallel[
            Linear[Self.FILTERS, 7],
            Linear[Self.FILTERS, 1],
        ],
    ]
    comptime OptType = AdamW[
        LR=Self.LR, WEIGHT_DECAY=Self.WD
    ]  # L2=1e-4 (matches AlphaZero.jl)

    comptime batch_size: Int = Self.BS
    comptime buffer_capacity: Int = Self.CAP
    comptime history_window: Int = 20
    comptime num_simulations: Int = Self.SIMS
    comptime max_nodes: Int = Self.NODES
    comptime temp_threshold: Int = 20
    comptime temp_min: Float64 = 0.0
    comptime batch_sims: Int = 8
    comptime invalid_action_penalty: Float64 = 0.0
    comptime value_target_q_weight: Float64 = 0.0
    comptime max_episode_length: Int = 42
    comptime board_rows: Int = 6
    comptime board_cols: Int = 7
    comptime board_planes: Int = 3
    comptime num_symmetries: Int = 2

    comptime Noise = DirichletNoise[
        0.25, 1.0
    ]  # alpha=1.0 for C4 (AlphaZero.jl)
    comptime PUCT = AlphaGoPUCT[Self.C_PUCT]
    comptime Players = SelfPlay


# ═══════════════════════════════════════════════════════════════════════════
# ConnectFour Config (ResNet — closer to original AlphaZero)
# ═══════════════════════════════════════════════════════════════════════════

# ResNet blocks with BatchNorm (matching alpha-zero-general):
# Conv1 → BN → ReLU → Conv2 → BN → (+skip) → ReLU
comptime ResBlockBN6x7[F: Int] = Sequential[
    Residual[
        Sequential[
            Conv2DBatchNormReLU[F, F, 3, 1, 1, 6, 7],  # Conv1 → BN1 → ReLU
            Conv2DLayer[F, F, 3, 1, 1, 6, 7],  # Conv2 (no act)
            BatchNorm2D[F, 6, 7],  # BN2
        ]
    ],
    ReLU[F * 6 * 7],  # skip add → ReLU
]

comptime ResBlockBNFused6x7[F: Int] = ResBlockConv2DBN[F, 3, 1, 6, 7]

comptime ResBlockBN3x3[F: Int] = Sequential[
    Residual[
        Sequential[
            Conv2DBatchNormReLU[F, F, 3, 1, 1, 3, 3],
            Conv2DLayer[F, F, 3, 1, 1, 3, 3],
            BatchNorm2D[F, 3, 3],
        ]
    ],
    ReLU[F * 3 * 3],
]

comptime ResBlockBNFused3x3[F: Int] = ResBlockConv2DBN[F, 3, 1, 3, 3]

# Legacy ResBlocks without BN (for backwards compatibility)
comptime ResBlock6x7[F: Int] = Sequential[
    Residual[
        Sequential[
            Conv2DReLU[F, F, 3, 1, 1, 6, 7],
            Conv2DLayer[F, F, 3, 1, 1, 6, 7],
        ]
    ],
    ReLU[F * 6 * 7],
]

comptime ResBlock3x3[F: Int] = Sequential[
    Residual[
        Sequential[
            Conv2DReLU[F, F, 3, 1, 1, 3, 3],
            Conv2DLayer[F, F, 3, 1, 1, 3, 3],
        ]
    ],
    ReLU[F * 3 * 3],
]


struct AlphaZeroConnectFourResNetConfig[
    FILTERS: Int = 128,
    NUM_BLOCKS: Int = 5,
    LR: Float64 = 2e-3,
    WD: Float64 = 1e-4,
    BS: Int = 64,
    CAP: Int = 400000,
    SIMS: Int = 600,
    NODES: Int = 1024,
    C_PUCT: Float64 = 2.0,
](AlphaZeroConfig):
    """AlphaZero for ConnectFour — ResNet with BatchNorm.

    NUM_BLOCKS controls depth (5=AlphaZero.jl, 10=medium, 20=original AZ).
    Uses Repeat[N] for weight-shared blocks (efficient parameter usage).

    Note: Repeat shares weights across all N blocks. For independent weights,
    list blocks explicitly or use the FusedResNet config.
    """

    comptime NAME: String = "AlphaZero-ConnectFour-ResNet"
    comptime obs_dim: Int = 126
    comptime action_dim: Int = 7

    comptime PredModel = Sequential[
        Conv2DBatchNormReLU[3, Self.FILTERS, 3, 1, 1, 6, 7],  # Initial: 3ch→F
        Repeat[
            Self.NUM_BLOCKS, ResBlockBN6x7[Self.FILTERS], shared=False
        ],  # N× independent ResBlocks
        FlattenLayer[Self.FILTERS * 6 * 7],
        LinearBatchNormReLU[Self.FILTERS * 6 * 7, Self.FILTERS * 2],
        Dropout[Self.FILTERS * 2, 0.3, 42, True],
        LinearBatchNormReLU[Self.FILTERS * 2, Self.FILTERS],
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
    comptime temp_threshold: Int = 20
    comptime temp_min: Float64 = 0.0
    comptime batch_sims: Int = 8
    comptime invalid_action_penalty: Float64 = 0.0
    comptime value_target_q_weight: Float64 = 0.0
    comptime max_episode_length: Int = 42
    comptime board_rows: Int = 6
    comptime board_cols: Int = 7
    comptime board_planes: Int = 3
    comptime num_symmetries: Int = 2

    comptime Noise = DirichletNoise[0.25, 1.0]
    comptime PUCT = AlphaGoPUCT[Self.C_PUCT]
    comptime Players = SelfPlay


# ═══════════════════════════════════════════════════════════════════════════
# ConnectFour Config (ResNet — Fused ResBlocks for fewer kernel launches)
# ═══════════════════════════════════════════════════════════════════════════


struct AlphaZeroConnectFourFusedResNetConfig[
    FILTERS: Int = 128,
    NUM_BLOCKS: Int = 5,
    LR: Float64 = 2e-3,
    WD: Float64 = 1e-4,
    BS: Int = 1024,
    CAP: Int = 400000,
    SIMS: Int = 600,
    NODES: Int = 1024,
    C_PUCT: Float64 = 2.0,
](AlphaZeroConfig):
    """AlphaZero for ConnectFour — Fused ResNet with ResBlockConv2DBN.

    Same architecture as AlphaZeroConnectFourResNetConfig but uses
    fused ResBlockConv2DBN which merges BN2+skip+ReLU into one kernel.
    Fewer kernel launches per training step and smaller workspace.
    """

    comptime NAME: String = "AlphaZero-ConnectFour-FusedResNet"
    comptime obs_dim: Int = 126
    comptime action_dim: Int = 7

    comptime PredModel = Sequential[
        Conv2DBatchNormReLU[3, Self.FILTERS, 3, 1, 1, 6, 7],  # Initial: 3ch→F
        Repeat[
            Self.NUM_BLOCKS, ResBlockBNFused6x7[Self.FILTERS], shared=False
        ],  # N× independent ResBlocks
        FlattenLayer[Self.FILTERS * 6 * 7],
        LinearBatchNormReLU[Self.FILTERS * 6 * 7, Self.FILTERS * 2],
        # Dropout[Self.FILTERS * 2, 0.3, 42, True],
        LinearBatchNormReLU[Self.FILTERS * 2, Self.FILTERS],
        # Dropout[Self.FILTERS, 0.3, 137, True],
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
    comptime temp_threshold: Int = 20
    comptime temp_min: Float64 = 0.3  # AlphaZero.jl: temp=0.3 after move 20
    comptime batch_sims: Int = 8
    comptime invalid_action_penalty: Float64 = 1.0  # AlphaZero.jl: nonvalidity_penalty=1.0
    comptime value_target_q_weight: Float64 = 0.0
    comptime max_episode_length: Int = 42
    comptime board_rows: Int = 6
    comptime board_cols: Int = 7
    comptime board_planes: Int = 3
    comptime num_symmetries: Int = 2

    comptime Noise = DirichletNoise[0.25, 1.0]
    comptime PUCT = AlphaGoPUCT[Self.C_PUCT]
    comptime Players = SelfPlay


# ═══════════════════════════════════════════════════════════════════════════
# TicTacToe ResNet Config
# ═══════════════════════════════════════════════════════════════════════════


struct AlphaZeroTicTacToeResNetConfig[
    FILTERS: Int = 128,
    NUM_BLOCKS: Int = 4,
    LR: Float64 = 1e-3,
    BS: Int = 64,
    CAP: Int = 50000,
    SIMS: Int = 50,
    NODES: Int = 64,
    C_PUCT: Float64 = 1.0,
](AlphaZeroConfig):
    """AlphaZero for TicTacToe — ResNet with BN. NUM_BLOCKS controls depth."""

    comptime NAME: String = "AlphaZero-TicTacToe-ResNet"
    comptime obs_dim: Int = 27
    comptime action_dim: Int = 9

    comptime PredModel = Sequential[
        Conv2DBatchNormReLU[3, Self.FILTERS, 3, 1, 1, 3, 3],  # Initial: 3ch→F
        Repeat[
            Self.NUM_BLOCKS, ResBlockBN3x3[Self.FILTERS], shared=False
        ],  # N× independent ResBlocks
        Conv2DBatchNormReLU[
            Self.FILTERS, Self.FILTERS, 3, 1, 0, 3, 3
        ],  # Reduce: 3×3→1×1
        FlattenLayer[Self.FILTERS],
        LinearBatchNormReLU[Self.FILTERS, Self.FILTERS * 2],
        Dropout[Self.FILTERS * 2, 0.3, 42, True],
        LinearBatchNormReLU[Self.FILTERS * 2, Self.FILTERS],
        Dropout[Self.FILTERS, 0.3, 137, True],
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
    comptime temp_min: Float64 = 0.0
    comptime batch_sims: Int = 8
    comptime invalid_action_penalty: Float64 = 0.0
    comptime value_target_q_weight: Float64 = 0.0
    comptime max_episode_length: Int = 9
    comptime board_rows: Int = 3
    comptime board_cols: Int = 3
    comptime board_planes: Int = 3
    comptime num_symmetries: Int = 2

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
    comptime temp_min: Float64 = 0.0
    comptime batch_sims: Int = 8
    comptime invalid_action_penalty: Float64 = 0.0
    comptime value_target_q_weight: Float64 = 0.0
    comptime max_episode_length: Int = 512
    comptime board_rows: Int = 8
    comptime board_cols: Int = 8
    comptime board_planes: Int = 14
    comptime num_symmetries: Int = 1  # No augmentation for chess

    comptime Noise = DirichletNoise[
        0.25, 0.03
    ]  # Small alpha for large action space
    comptime PUCT = AlphaGoPUCT[2.5]
    comptime Players = SelfPlay
