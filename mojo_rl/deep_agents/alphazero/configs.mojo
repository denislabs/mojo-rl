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
    SinglePlayer,
    BackupMode,
    MonteCarloReturn,
    NStepBootstrap,
)
from .strategies import (
    BoardAugmenter,
    IdentityAugmenter,
    D4SquareAugmenter,
    HFlipColumnAugmenter,
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
    comptime virtual_loss: Int  # PUCT virtual-loss magnitude per pick within a round (3=AlphaGo default; lower → more concentrated visits in small-action games)
    comptime invalid_action_penalty: Float64  # Penalty for prob mass on illegal moves (1.0=AlphaZero.jl)
    comptime max_grad_norm: Float64  # Max gradient norm for clipping (0.0=disabled)

    # ── Value target ─────────────────────────────────────────────
    # Blend between game outcome (z) and MCTS root Q-value (q):
    #   value_target = (1 - w) * z + w * q
    # 0.0 = pure z (original AlphaZero), 1.0 = pure q, 0.5 = average
    comptime value_target_q_weight: Float64

    # Whether the value head is squashed through tanh during loss.
    # True (default for board games): targets ∈ [-1, +1], loss is
    #   `(tanh(raw) - target)²`. Bounded outputs.
    # False (single-player envs with unbounded returns): targets are
    #   the raw discounted return, loss is `(raw - target)²`. Tanh would
    #   saturate to zero gradient for targets > 1.
    comptime value_squash: Bool

    # ── GPU episode tracking ────────────────────────────────────────
    comptime max_episode_length: Int  # Max steps per episode (for GPU staging)

    # ── Board layout (display only — used by replay-buffer dump diagnostics)
    comptime board_rows: Int
    comptime board_cols: Int
    comptime board_planes: Int

    # ── Strategies (shared with MuZero) ───────────────────────────
    comptime Noise: ExplorationNoise
    comptime PUCT: PUCTFormula
    comptime Players: PlayerMode
    comptime Backup: BackupMode

    # ── Strategies (AZ-specific) ──────────────────────────────────
    comptime Aug: BoardAugmenter

    # ── Planner refactor toggle ───────────────────────────────────
    comptime USE_NEW_MCTS: Bool
    """Route GPU action selection through
    ``planners.tree_search.GenericGPUMCTS.search_gpu_alphazero`` instead of
    the inline kernel block. Defaults to ``False`` so production training
    is unchanged until the rewiring is flipped on per-config."""


# ═══════════════════════════════════════════════════════════════════════════
# TicTacToe Config (MLP — lightweight)
# ═══════════════════════════════════════════════════════════════════════════


struct AlphaZeroTicTacToeConfig[
    HIDDEN: Int = 128,
    LR: Float64 = 0.01,
    BS: Int = 16,
    CAP: Int = 120000,
    SIMS: Int = 100,
    NODES: Int = 128,
    C_PUCT: Float64 = 1.0,
    BATCH_SIMS: Int = 8,
    VLOSS: Int = 3,
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
    comptime temp_threshold: Int = 4  # temp=1 first 4 moves, then temp_min
    comptime temp_min: Float64 = 0.0
    comptime batch_sims: Int = Self.BATCH_SIMS
    comptime virtual_loss: Int = Self.VLOSS
    comptime invalid_action_penalty: Float64 = 0.0
    comptime max_grad_norm: Float64 = 0.0
    comptime value_target_q_weight: Float64 = 0.0
    comptime value_squash: Bool = True
    comptime max_episode_length: Int = 9  # 3×3 board
    comptime board_rows: Int = 3
    comptime board_cols: Int = 3
    comptime board_planes: Int = 3

    comptime Noise = DirichletNoise[0.25, 0.25]
    comptime PUCT = AlphaGoPUCT[Self.C_PUCT]
    comptime Players = SelfPlay
    comptime Backup = MonteCarloReturn
    comptime Aug = D4SquareAugmenter[3, 3]
    comptime USE_NEW_MCTS: Bool = False


# ═══════════════════════════════════════════════════════════════════════════
# TicTacToe Config (CNN — matches alpha-zero-general architecture)
# ═══════════════════════════════════════════════════════════════════════════


struct AlphaZeroTicTacToeCNNConfig[
    FILTERS: Int = 128,
    LR: Float64 = 0.001,
    BS: Int = 64,
    CAP: Int = 120_000,
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
        # Dropout[Self.FILTERS * 2, 0.3, 42, True],
        LinearBatchNormReLU[Self.FILTERS * 2, Self.FILTERS],
        # Dropout[Self.FILTERS, 0.3, 137, True],
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
    comptime temp_threshold: Int = 4
    comptime temp_min: Float64 = 0.0
    comptime batch_sims: Int = 8
    comptime virtual_loss: Int = 3
    comptime invalid_action_penalty: Float64 = 0.0
    comptime max_grad_norm: Float64 = 0.0
    comptime value_target_q_weight: Float64 = 0.0
    comptime value_squash: Bool = True
    comptime max_episode_length: Int = 9
    comptime board_rows: Int = 3
    comptime board_cols: Int = 3
    comptime board_planes: Int = 3

    comptime Noise = DirichletNoise[0.25, 0.25]
    comptime PUCT = AlphaGoPUCT[Self.C_PUCT]
    comptime Players = SelfPlay
    comptime Backup = MonteCarloReturn
    comptime Aug = D4SquareAugmenter[3, 3]
    comptime USE_NEW_MCTS: Bool = False


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
    comptime virtual_loss: Int = 3
    comptime invalid_action_penalty: Float64 = 0.0
    comptime max_grad_norm: Float64 = 0.0
    comptime value_target_q_weight: Float64 = 0.0
    comptime value_squash: Bool = True
    comptime max_episode_length: Int = 42  # 6×7 board
    comptime board_rows: Int = 6
    comptime board_cols: Int = 7
    comptime board_planes: Int = 3

    comptime Noise = DirichletNoise[0.25, 1.0]  # alpha=1.0 for C4
    comptime PUCT = AlphaGoPUCT[2.0]
    comptime Players = SelfPlay
    comptime Backup = MonteCarloReturn
    comptime Aug = HFlipColumnAugmenter[6, 7, 3]
    comptime USE_NEW_MCTS: Bool = False


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
    comptime virtual_loss: Int = 3
    comptime invalid_action_penalty: Float64 = 0.0
    comptime max_grad_norm: Float64 = 0.0
    comptime value_target_q_weight: Float64 = 0.0
    comptime value_squash: Bool = True
    comptime max_episode_length: Int = 42
    comptime board_rows: Int = 6
    comptime board_cols: Int = 7
    comptime board_planes: Int = 3

    comptime Noise = DirichletNoise[
        0.25, 1.0
    ]  # alpha=1.0 for C4 (AlphaZero.jl)
    comptime PUCT = AlphaGoPUCT[Self.C_PUCT]
    comptime Players = SelfPlay
    comptime Backup = MonteCarloReturn
    comptime Aug = HFlipColumnAugmenter[6, 7, 3]
    comptime USE_NEW_MCTS: Bool = False


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
    comptime virtual_loss: Int = 3
    comptime invalid_action_penalty: Float64 = 0.0
    comptime max_grad_norm: Float64 = 0.0
    comptime value_target_q_weight: Float64 = 0.0
    comptime value_squash: Bool = True
    comptime max_episode_length: Int = 42
    comptime board_rows: Int = 6
    comptime board_cols: Int = 7
    comptime board_planes: Int = 3

    comptime Noise = DirichletNoise[0.25, 1.0]
    comptime PUCT = AlphaGoPUCT[Self.C_PUCT]
    comptime Players = SelfPlay
    comptime Backup = MonteCarloReturn
    comptime Aug = HFlipColumnAugmenter[6, 7, 3]
    comptime USE_NEW_MCTS: Bool = False


# ═══════════════════════════════════════════════════════════════════════════
# ConnectFour Config (ResNet — Fused ResBlocks for fewer kernel launches)
# ═══════════════════════════════════════════════════════════════════════════


struct AlphaZeroConnectFourFusedResNetConfig[
    FILTERS: Int = 128,
    NUM_BLOCKS: Int = 5,
    HEAD_FILTERS: Int = 32,
    LR: Float64 = 2e-3,
    WD: Float64 = 1e-4,
    BS: Int = 1024,
    CAP: Int = 1000000,
    SIMS: Int = 600,
    NODES: Int = 1024,
    C_PUCT: Float64 = 2.0,
](AlphaZeroConfig):
    """AlphaZero for ConnectFour — Fused ResNet matching AlphaZero.jl architecture.

    Separate policy/value heads with 1x1 conv (matching AlphaZero.jl ResNetHP):
      Policy: Conv1x1(F→HF)+BN+ReLU → Flatten → Dense(HF*42 → 7)
      Value:  Conv1x1(F→HF)+BN+ReLU → Flatten → Dense(HF*42 → F)+ReLU → Dense(F → 1)
    """

    comptime NAME: String = "AlphaZero-ConnectFour-FusedResNet"
    comptime obs_dim: Int = 126
    comptime action_dim: Int = 7

    # Head intermediate dim: HEAD_FILTERS * board_size
    comptime HEAD_DIM: Int = Self.HEAD_FILTERS * 6 * 7

    comptime PredModel = Sequential[
        Conv2DBatchNormReLU[3, Self.FILTERS, 3, 1, 1, 6, 7],  # Initial: 3ch→F
        Repeat[
            Self.NUM_BLOCKS, ResBlockBNFused6x7[Self.FILTERS], shared=False
        ],  # N× independent ResBlocks
        # Separate conv heads (matching AlphaZero.jl / DeepMind AlphaZero)
        Parallel[
            # Policy head: Conv1x1+BN+ReLU → Flatten → FC → logits
            Sequential[
                Conv2DBatchNormReLU[
                    Self.FILTERS, Self.HEAD_FILTERS, 1, 1, 0, 6, 7
                ],
                FlattenLayer[Self.HEAD_DIM],
                Linear[Self.HEAD_DIM, 7],
            ],
            # Value head: Conv1x1+BN+ReLU → Flatten → FC+ReLU → FC → scalar
            Sequential[
                Conv2DBatchNormReLU[
                    Self.FILTERS, Self.HEAD_FILTERS, 1, 1, 0, 6, 7
                ],
                FlattenLayer[Self.HEAD_DIM],
                LinearReLU[Self.HEAD_DIM, Self.FILTERS],
                Linear[Self.FILTERS, 1],
            ],
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
    comptime virtual_loss: Int = 3
    comptime invalid_action_penalty: Float64 = 1.0  # AlphaZero.jl: nonvalidity_penalty=1.0
    comptime max_grad_norm: Float64 = 0.0
    comptime value_target_q_weight: Float64 = 0.0
    comptime value_squash: Bool = True
    comptime max_episode_length: Int = 42
    comptime board_rows: Int = 6
    comptime board_cols: Int = 7
    comptime board_planes: Int = 3

    comptime Noise = DirichletNoise[0.25, 1.0]
    comptime PUCT = AlphaGoPUCT[Self.C_PUCT]
    comptime Players = SelfPlay
    comptime Backup = MonteCarloReturn
    comptime Aug = HFlipColumnAugmenter[6, 7, 3]
    comptime USE_NEW_MCTS: Bool = False


# ═══════════════════════════════════════════════════════════════════════════
# TicTacToe ResNet Config
# ═══════════════════════════════════════════════════════════════════════════


struct AlphaZeroTicTacToeResNetConfig[
    FILTERS: Int = 128,
    NUM_BLOCKS: Int = 4,
    LR: Float64 = 1e-3,
    BS: Int = 64,
    CAP: Int = 120000,
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
        # Dropout[Self.FILTERS * 2, 0.3, 42, True],
        LinearBatchNormReLU[Self.FILTERS * 2, Self.FILTERS],
        # Dropout[Self.FILTERS, 0.3, 137, True],
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
    comptime temp_threshold: Int = 4
    comptime temp_min: Float64 = 0.0
    comptime batch_sims: Int = 8
    comptime virtual_loss: Int = 3
    comptime invalid_action_penalty: Float64 = 0.0
    comptime max_grad_norm: Float64 = 0.0
    comptime value_target_q_weight: Float64 = 0.0
    comptime value_squash: Bool = True
    comptime max_episode_length: Int = 9
    comptime board_rows: Int = 3
    comptime board_cols: Int = 3
    comptime board_planes: Int = 3

    comptime Noise = DirichletNoise[0.25, 0.25]
    comptime PUCT = AlphaGoPUCT[Self.C_PUCT]
    comptime Players = SelfPlay
    comptime Backup = MonteCarloReturn
    comptime Aug = D4SquareAugmenter[3, 3]
    comptime USE_NEW_MCTS: Bool = False


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
    comptime virtual_loss: Int = 3
    comptime invalid_action_penalty: Float64 = 0.0
    comptime max_grad_norm: Float64 = 0.0
    comptime value_target_q_weight: Float64 = 0.0
    comptime value_squash: Bool = True
    comptime max_episode_length: Int = 512
    comptime board_rows: Int = 8
    comptime board_cols: Int = 8
    comptime board_planes: Int = 14

    comptime Noise = DirichletNoise[
        0.25, 0.03
    ]  # Small alpha for large action space
    comptime PUCT = AlphaGoPUCT[2.5]
    comptime Players = SelfPlay
    comptime Backup = MonteCarloReturn
    comptime Aug = IdentityAugmenter
    comptime USE_NEW_MCTS: Bool = False


# ═══════════════════════════════════════════════════════════════════════════
# CartPole Config (single-player MLP — for AZ-vs-MuZero comparison)
# ═══════════════════════════════════════════════════════════════════════════


struct AlphaZeroCartPoleConfig[
    HIDDEN: Int = 64,
    LR: Float64 = 0.001,
    BS: Int = 64,
    CAP: Int = 50000,
    SIMS: Int = 25,
    NODES: Int = 64,
    C_PUCT: Float64 = 1.25,
    MAX_EP: Int = 500,
](AlphaZeroConfig):
    """AlphaZero on CartPole — single-player MCTS-with-true-rules baseline.

    Used to validate AZ's value-learning machinery on a non-board-game env
    where MuZero is currently broken. AZ here uses the env as a perfect
    model, isolating "MCTS + value learning works" from the learned-model
    side that MuZero is debugging.
    """

    comptime NAME: String = "AlphaZero-CartPole"
    comptime obs_dim: Int = 4
    comptime action_dim: Int = 2

    comptime PredModel = Sequential[
        LinearReLU[4, Self.HIDDEN],
        LinearReLU[Self.HIDDEN, Self.HIDDEN],
        Parallel[
            Linear[Self.HIDDEN, 2],  # Policy head
            Linear[Self.HIDDEN, 1],  # Scalar value head
        ],
    ]
    comptime OptType = Adam[LR=Self.LR]

    comptime batch_size: Int = Self.BS
    comptime buffer_capacity: Int = Self.CAP
    comptime history_window: Int = 20
    comptime num_simulations: Int = Self.SIMS
    comptime max_nodes: Int = Self.NODES
    comptime temp_threshold: Int = 50  # Long horizon — broad exploration
    comptime temp_min: Float64 = 0.0
    comptime batch_sims: Int = 5
    comptime virtual_loss: Int = 3
    comptime invalid_action_penalty: Float64 = 0.0
    comptime max_grad_norm: Float64 = 0.0
    comptime value_target_q_weight: Float64 = 0.0
    comptime value_squash: Bool = False
    comptime max_episode_length: Int = Self.MAX_EP
    # Board layout fields are required by the trait but unused for non-board
    # envs — pick a 1×OBS×1 shape so the diagnostic dump_replay still works.
    comptime board_rows: Int = 1
    comptime board_cols: Int = 4
    comptime board_planes: Int = 1

    comptime Noise = DirichletNoise[0.25, 0.25]
    comptime PUCT = MuZeroPUCT[19652.0, Self.C_PUCT]
    comptime Players = SinglePlayer
    comptime Backup = MonteCarloReturn
    comptime Aug = IdentityAugmenter
    comptime USE_NEW_MCTS: Bool = False
