"""MuZero Configs — Compile-time configuration for MuZero family agents.

Follows the same composable pattern as DQN/SAC configs:
  - MuZeroConfig trait defines the interface
  - Concrete configs (MuZeroMLPConfig, MuZeroCNNConfig, MuZeroResNetConfig)
    bundle network architectures + hyperparameters
  - GenericMuZeroAgent[Config] works with any config

Usage:
    # Standard MuZero on CartPole
    var agent = GenericMuZeroAgent[MuZeroMLPConfig[4, 2]]()

    # MuZero CNN on Pong pixel observations
    var agent = GenericMuZeroAgent[MuZeroCNNConfig[3]]()

    # MuZero with ResNet on complex environments
    var agent = GenericMuZeroAgent[MuZeroResNetConfig[17, 6]]()
"""

from mojo_rl.nn.model import (
    Model,
    Linear,
    LinearReLU,
    LinearMish,
    Sequential,
    Parallel,
)
from mojo_rl.nn.model import Conv2DReLU, FlattenLayer
from mojo_rl.nn.optimizer import Optimizer, Adam
from mojo_rl.nn.autodiff.combinators import Residual
from .strategies import (
    SearchMode,
    LearnedDynamics,
    TrueGameRules,
    ValueEncoding,
    CategoricalEncoding,
    ScalarEncoding,
    SymlogEncoding,
    HiddenScaling,
    MinMaxScale,
    NoScale,
    ExplorationNoise,
    DirichletNoise,
    EpsilonNoise,
    NoNoise,
    PUCTFormula,
    MuZeroPUCT,
    AlphaGoPUCT,
    UCB1Formula,
    BackupMode,
    NStepBootstrap,
    MonteCarloReturn,
    LambdaReturn,
)


# ═══════════════════════════════════════════════════════════════════════════
# MuZero Config Trait
# ═══════════════════════════════════════════════════════════════════════════


trait MuZeroConfig:
    """Compile-time configuration for MuZero family agents.

    Bundles network architectures, optimizer, dimensions, and training
    hyperparameters. Concrete implementations define the full algorithm
    variant (MLP, CNN, ResNet, etc.).

    All MuZero variants share the same training loop (K-step unrolled
    forward/backward with MCTS planning). Only the networks differ.
    """

    comptime NAME: String

    # ── Dimensions ────────────────────────────────────────────────
    comptime obs_dim: Int         # Observation space dimension
    comptime action_dim: Int      # Number of discrete actions
    comptime latent_dim: Int      # Hidden state dimension
    comptime num_bins: Int        # Distributional value/reward bins

    # ── Derived dimensions ────────────────────────────────────────
    comptime DYN_IN: Int          # = latent_dim + action_dim
    comptime DYN_OUT: Int         # = latent_dim + num_bins
    comptime PRED_OUT: Int        # = action_dim + num_bins

    # ── Network Architectures ─────────────────────────────────────
    comptime RepModel: Model      # h(obs) → hidden_state
    comptime DynModel: Model      # g(hidden || one_hot_action) → (next_hidden || reward_logits)
    comptime PredModel: Model     # f(hidden) → (policy_logits || value_logits)
    comptime OptType: Optimizer   # Shared optimizer for all three networks

    # ── Training Hyperparameters ──────────────────────────────────
    comptime batch_size: Int      # Training batch size
    comptime buffer_capacity: Int # Replay buffer capacity
    comptime unroll_steps: Int    # K-step unroll depth
    comptime td_steps: Int        # N-step bootstrap horizon

    # ── MCTS Hyperparameters ──────────────────────────────────────
    comptime num_simulations: Int # MCTS simulations per action
    comptime max_nodes: Int       # Maximum tree nodes per search

    # ── Strategy Types ────────────────────────────────────────────
    comptime Search: SearchMode           # Learned dynamics vs true game rules
    comptime Encoding: ValueEncoding      # Categorical vs scalar value encoding
    comptime Scaling: HiddenScaling       # Hidden state normalization
    comptime Noise: ExplorationNoise      # Root exploration noise
    comptime PUCT: PUCTFormula            # UCB exploration formula
    comptime Backup: BackupMode           # Return computation strategy

    # ── Flags ─────────────────────────────────────────────────────
    comptime USE_REANALYZE: Bool  # Enable MuZero Reanalyze


# ═══════════════════════════════════════════════════════════════════════════
# MuZero MLP Config (standard, clean observations)
# ═══════════════════════════════════════════════════════════════════════════


struct MuZeroMLPConfig[
    OBS: Int,
    ACT: Int,
    LATENT: Int = 256,
    HIDDEN: Int = 256,
    BINS: Int = 101,
    LR: Float64 = 3e-4,
    CAP: Int = 100000,
    BS: Int = 128,
    K: Int = 5,
    N: Int = 10,
    SIMS: Int = 50,
    NODES: Int = 64,
](MuZeroConfig):
    """Standard MuZero with MLP networks for clean observations.

    Three-layer MLP for representation, dynamics, and prediction.
    Suitable for CartPole (4D), Pong clean obs (6D), etc.
    """

    comptime NAME: String = "MuZero-MLP"

    # Dimensions
    comptime obs_dim: Int = Self.OBS
    comptime action_dim: Int = Self.ACT
    comptime latent_dim: Int = Self.LATENT
    comptime num_bins: Int = Self.BINS
    comptime DYN_IN: Int = Self.LATENT + Self.ACT
    comptime DYN_OUT: Int = Self.LATENT + Self.BINS
    comptime PRED_OUT: Int = Self.ACT + Self.BINS

    # Networks
    comptime RepModel = Sequential[
        LinearMish[Self.OBS, Self.HIDDEN],
        LinearMish[Self.HIDDEN, Self.HIDDEN],
        Linear[Self.HIDDEN, Self.LATENT],
    ]

    comptime DynModel = Sequential[
        LinearMish[Self.DYN_IN, Self.HIDDEN],
        LinearMish[Self.HIDDEN, Self.HIDDEN],
        Linear[Self.HIDDEN, Self.DYN_OUT],
    ]

    comptime PredModel = Sequential[
        LinearMish[Self.LATENT, Self.HIDDEN],
        Parallel[
            Linear[Self.HIDDEN, Self.ACT],
            Linear[Self.HIDDEN, Self.BINS],
        ],
    ]

    comptime OptType = Adam[LR=Self.LR]

    # Training
    comptime batch_size: Int = Self.BS
    comptime buffer_capacity: Int = Self.CAP
    comptime unroll_steps: Int = Self.K
    comptime td_steps: Int = Self.N

    # MCTS
    comptime num_simulations: Int = Self.SIMS
    comptime max_nodes: Int = Self.NODES

    # Strategy types
    comptime Search = LearnedDynamics
    comptime Encoding = CategoricalEncoding
    comptime Scaling = MinMaxScale
    comptime Noise = DirichletNoise[0.25, 0.25]
    comptime PUCT = MuZeroPUCT[19652.0, 1.25]
    comptime Backup = NStepBootstrap

    comptime USE_REANALYZE: Bool = False


# ═══════════════════════════════════════════════════════════════════════════
# MuZero CNN Config (pixel observations, 4x84x84)
# ═══════════════════════════════════════════════════════════════════════════


struct MuZeroCNNConfig[
    ACT: Int,
    LATENT: Int = 512,
    BINS: Int = 101,
    LR: Float64 = 2.5e-4,
    CAP: Int = 100000,
    BS: Int = 64,
    K: Int = 5,
    N: Int = 10,
    SIMS: Int = 50,
    NODES: Int = 64,
](MuZeroConfig):
    """MuZero with CNN representation for 4x84x84 pixel observations.

    NatureDQN-style Conv2D representation network (Mnih et al., 2015).
    Dynamics and prediction use MLP on the latent space.
    Suitable for Pong, Breakout, Space Invaders pixel modes.
    """

    comptime NAME: String = "MuZero-CNN"

    # Dimensions
    comptime obs_dim: Int = 4 * 84 * 84  # 28224
    comptime action_dim: Int = Self.ACT
    comptime latent_dim: Int = Self.LATENT
    comptime num_bins: Int = Self.BINS
    comptime DYN_IN: Int = Self.LATENT + Self.ACT
    comptime DYN_OUT: Int = Self.LATENT + Self.BINS
    comptime PRED_OUT: Int = Self.ACT + Self.BINS

    # CNN Representation (NatureDQN downsampling → latent)
    comptime RepModel = Sequential[
        Conv2DReLU[4, 32, 8, 4, 0, 84, 84],     # → 20x20
        Conv2DReLU[32, 64, 4, 2, 0, 20, 20],    # → 9x9
        Conv2DReLU[64, 64, 3, 1, 0, 9, 9],      # → 7x7
        FlattenLayer[64 * 7 * 7],                 # → 3136
        LinearReLU[64 * 7 * 7, 512],
        Linear[512, Self.LATENT],
    ]

    # Dynamics + Prediction in latent space (same MLP as standard MuZero)
    comptime DynModel = Sequential[
        LinearMish[Self.DYN_IN, 256],
        LinearMish[256, 256],
        Linear[256, Self.DYN_OUT],
    ]

    comptime PredModel = Sequential[
        LinearMish[Self.LATENT, 256],
        Parallel[
            Linear[256, Self.ACT],
            Linear[256, Self.BINS],
        ],
    ]

    comptime OptType = Adam[LR=Self.LR]

    # Training
    comptime batch_size: Int = Self.BS
    comptime buffer_capacity: Int = Self.CAP
    comptime unroll_steps: Int = Self.K
    comptime td_steps: Int = Self.N

    # MCTS
    comptime num_simulations: Int = Self.SIMS
    comptime max_nodes: Int = Self.NODES

    # Strategy types
    comptime Search = LearnedDynamics
    comptime Encoding = CategoricalEncoding
    comptime Scaling = MinMaxScale
    comptime Noise = DirichletNoise[0.25, 0.25]
    comptime PUCT = MuZeroPUCT[19652.0, 1.25]
    comptime Backup = NStepBootstrap

    comptime USE_REANALYZE: Bool = False


# ═══════════════════════════════════════════════════════════════════════════
# MuZero ResNet Config (deeper networks for harder tasks)
# ═══════════════════════════════════════════════════════════════════════════


struct MuZeroResNetConfig[
    OBS: Int,
    ACT: Int,
    LATENT: Int = 256,
    HIDDEN: Int = 256,
    BINS: Int = 101,
    LR: Float64 = 3e-4,
    CAP: Int = 100000,
    BS: Int = 128,
    K: Int = 5,
    N: Int = 10,
    SIMS: Int = 50,
    NODES: Int = 128,
](MuZeroConfig):
    """MuZero with ResNet blocks for deeper representation.

    Uses Residual connections in representation and dynamics networks
    for better gradient flow in deeper models. Suitable for more
    complex environments where the standard MLP underfits.
    """

    comptime NAME: String = "MuZero-ResNet"

    # Dimensions
    comptime obs_dim: Int = Self.OBS
    comptime action_dim: Int = Self.ACT
    comptime latent_dim: Int = Self.LATENT
    comptime num_bins: Int = Self.BINS
    comptime DYN_IN: Int = Self.LATENT + Self.ACT
    comptime DYN_OUT: Int = Self.LATENT + Self.BINS
    comptime PRED_OUT: Int = Self.ACT + Self.BINS

    # Representation with ResBlocks
    comptime RepModel = Sequential[
        LinearMish[Self.OBS, Self.HIDDEN],
        Residual[Sequential[
            LinearMish[Self.HIDDEN, Self.HIDDEN],
            Linear[Self.HIDDEN, Self.HIDDEN],
        ]],
        Residual[Sequential[
            LinearMish[Self.HIDDEN, Self.HIDDEN],
            Linear[Self.HIDDEN, Self.HIDDEN],
        ]],
        Linear[Self.HIDDEN, Self.LATENT],
    ]

    # Dynamics with ResBlock
    comptime DynModel = Sequential[
        LinearMish[Self.DYN_IN, Self.HIDDEN],
        Residual[Sequential[
            LinearMish[Self.HIDDEN, Self.HIDDEN],
            Linear[Self.HIDDEN, Self.HIDDEN],
        ]],
        Linear[Self.HIDDEN, Self.DYN_OUT],
    ]

    # Prediction with deeper heads
    comptime PredModel = Sequential[
        LinearMish[Self.LATENT, Self.HIDDEN],
        LinearMish[Self.HIDDEN, Self.HIDDEN],
        Parallel[
            Linear[Self.HIDDEN, Self.ACT],
            Linear[Self.HIDDEN, Self.BINS],
        ],
    ]

    comptime OptType = Adam[LR=Self.LR]

    # Training
    comptime batch_size: Int = Self.BS
    comptime buffer_capacity: Int = Self.CAP
    comptime unroll_steps: Int = Self.K
    comptime td_steps: Int = Self.N

    # MCTS
    comptime num_simulations: Int = Self.SIMS
    comptime max_nodes: Int = Self.NODES

    # Strategy types
    comptime Search = LearnedDynamics
    comptime Encoding = CategoricalEncoding
    comptime Scaling = MinMaxScale
    comptime Noise = DirichletNoise[0.25, 0.25]
    comptime PUCT = MuZeroPUCT[19652.0, 1.25]
    comptime Backup = NStepBootstrap

    comptime USE_REANALYZE: Bool = False


# ═══════════════════════════════════════════════════════════════════════════
# MuZero Large Config (for competitive performance)
# ═══════════════════════════════════════════════════════════════════════════


struct MuZeroLargeConfig[
    OBS: Int,
    ACT: Int,
    LR: Float64 = 1e-4,
    SIMS: Int = 100,
](MuZeroConfig):
    """MuZero with large networks for competitive performance.

    512-dim latent, 301-bin distributional, 100 MCTS simulations,
    larger buffer and batch for more stable training.
    """

    comptime NAME: String = "MuZero-Large"
    comptime obs_dim: Int = Self.OBS
    comptime action_dim: Int = Self.ACT
    comptime latent_dim: Int = 512
    comptime num_bins: Int = 301
    comptime DYN_IN: Int = 512 + Self.ACT
    comptime DYN_OUT: Int = 512 + 301
    comptime PRED_OUT: Int = Self.ACT + 301

    comptime RepModel = Sequential[
        LinearMish[Self.OBS, 512],
        LinearMish[512, 512],
        Residual[Sequential[LinearMish[512, 512], Linear[512, 512]]],
        Linear[512, 512],
    ]

    comptime DynModel = Sequential[
        LinearMish[Self.DYN_IN, 512],
        Residual[Sequential[LinearMish[512, 512], Linear[512, 512]]],
        Linear[512, Self.DYN_OUT],
    ]

    comptime PredModel = Sequential[
        LinearMish[512, 512],
        LinearMish[512, 512],
        Parallel[Linear[512, Self.ACT], Linear[512, 301]],
    ]

    comptime OptType = Adam[LR=Self.LR]

    comptime batch_size: Int = 256
    comptime buffer_capacity: Int = 500000
    comptime unroll_steps: Int = 5
    comptime td_steps: Int = 10
    comptime num_simulations: Int = Self.SIMS
    comptime max_nodes: Int = 256

    # Strategy types
    comptime Search = LearnedDynamics
    comptime Encoding = CategoricalEncoding
    comptime Scaling = MinMaxScale
    comptime Noise = DirichletNoise[0.25, 0.25]
    comptime PUCT = MuZeroPUCT[19652.0, 1.25]
    comptime Backup = NStepBootstrap

    comptime USE_REANALYZE: Bool = True


# ═══════════════════════════════════════════════════════════════════════════
# AlphaZero Config (true game rules, no learned dynamics)
# ═══════════════════════════════════════════════════════════════════════════


struct AlphaZeroConfig[
    OBS: Int,
    ACT: Int,
    HIDDEN: Int = 256,
    LR: Float64 = 1e-3,
    BS: Int = 256,
    SIMS: Int = 800,
    NODES: Int = 512,
](MuZeroConfig):
    """AlphaZero-style config: true game rules, no learned dynamics.

    Uses real game simulator for state transitions. Only learns
    policy + value networks. For board games (Chess, Go, etc.)
    where the game rules are known and deterministic.

    The DynModel is still required by the trait but is a minimal stub
    (unused when Search = TrueGameRules).
    """

    comptime NAME: String = "AlphaZero"
    comptime obs_dim: Int = Self.OBS
    comptime action_dim: Int = Self.ACT
    comptime latent_dim: Int = Self.OBS  # Latent = obs space for AlphaZero
    comptime num_bins: Int = 1           # No distributional (scalar values)
    comptime DYN_IN: Int = Self.OBS + Self.ACT
    comptime DYN_OUT: Int = Self.OBS + 1  # Stub
    comptime PRED_OUT: Int = Self.ACT + 1  # Policy + scalar value

    # Policy + Value heads (no representation needed)
    comptime RepModel = Sequential[
        Linear[Self.OBS, Self.OBS],  # Identity-like (pass-through)
    ]

    # Dynamics stub (unused with TrueGameRules)
    comptime DynModel = Sequential[
        Linear[Self.DYN_IN, Self.DYN_OUT],
    ]

    # Policy + scalar value prediction
    comptime PredModel = Sequential[
        LinearReLU[Self.OBS, Self.HIDDEN],
        LinearReLU[Self.HIDDEN, Self.HIDDEN],
        Parallel[
            Linear[Self.HIDDEN, Self.ACT],    # Policy head
            Linear[Self.HIDDEN, 1],            # Scalar value head
        ],
    ]

    comptime OptType = Adam[LR=Self.LR]

    comptime batch_size: Int = Self.BS
    comptime buffer_capacity: Int = 100000
    comptime unroll_steps: Int = 1   # No unroll needed (true rules)
    comptime td_steps: Int = 0       # Full episode returns for board games

    comptime num_simulations: Int = Self.SIMS
    comptime max_nodes: Int = Self.NODES

    # AlphaZero strategy choices:
    comptime Search = TrueGameRules           # Use real game rules, not learned model
    comptime Encoding = ScalarEncoding         # Scalar value (game outcome in [-1, 1])
    comptime Scaling = NoScale                 # No hidden scaling (obs space is stable)
    comptime Noise = DirichletNoise[0.25, 0.03]  # Lower alpha for large action spaces
    comptime PUCT = AlphaGoPUCT[2.5]          # Constant c (not log-based)
    comptime Backup = MonteCarloReturn         # Full episode returns for board games

    comptime USE_REANALYZE: Bool = False


# ═══════════════════════════════════════════════════════════════════════════
# EfficientZero Config (MuZero + self-supervised consistency)
# ═══════════════════════════════════════════════════════════════════════════


struct EfficientZeroConfig[
    OBS: Int,
    ACT: Int,
    LATENT: Int = 256,
    HIDDEN: Int = 256,
    BINS: Int = 101,
    LR: Float64 = 3e-4,
    SIMS: Int = 50,
](MuZeroConfig):
    """EfficientZero-style config (Ye et al., 2021).

    Extends MuZero with:
    - Self-supervised consistency loss (not yet implemented in training)
    - Symlog value encoding for better sample efficiency
    - Lambda returns for smoother value targets

    Network architectures same as MuZero MLP.
    Strategy choices optimize for sample efficiency.
    """

    comptime NAME: String = "EfficientZero"
    comptime obs_dim: Int = Self.OBS
    comptime action_dim: Int = Self.ACT
    comptime latent_dim: Int = Self.LATENT
    comptime num_bins: Int = Self.BINS
    comptime DYN_IN: Int = Self.LATENT + Self.ACT
    comptime DYN_OUT: Int = Self.LATENT + Self.BINS
    comptime PRED_OUT: Int = Self.ACT + Self.BINS

    comptime RepModel = Sequential[
        LinearMish[Self.OBS, Self.HIDDEN],
        LinearMish[Self.HIDDEN, Self.HIDDEN],
        Linear[Self.HIDDEN, Self.LATENT],
    ]
    comptime DynModel = Sequential[
        LinearMish[Self.DYN_IN, Self.HIDDEN],
        LinearMish[Self.HIDDEN, Self.HIDDEN],
        Linear[Self.HIDDEN, Self.DYN_OUT],
    ]
    comptime PredModel = Sequential[
        LinearMish[Self.LATENT, Self.HIDDEN],
        Parallel[
            Linear[Self.HIDDEN, Self.ACT],
            Linear[Self.HIDDEN, Self.BINS],
        ],
    ]
    comptime OptType = Adam[LR=Self.LR]

    comptime batch_size: Int = 128
    comptime buffer_capacity: Int = 100000
    comptime unroll_steps: Int = 5
    comptime td_steps: Int = 10

    comptime num_simulations: Int = Self.SIMS
    comptime max_nodes: Int = 64

    # EfficientZero strategy choices:
    comptime Search = LearnedDynamics
    comptime Encoding = CategoricalEncoding
    comptime Scaling = MinMaxScale
    comptime Noise = DirichletNoise[0.25, 0.25]
    comptime PUCT = MuZeroPUCT[19652.0, 1.25]
    comptime Backup = LambdaReturn[0.95]       # Lambda returns for sample efficiency

    comptime USE_REANALYZE: Bool = True         # Always use Reanalyze
